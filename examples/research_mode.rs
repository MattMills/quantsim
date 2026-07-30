//! Computational quantum research mode, end to end:
//!
//! 1. define a custom gate set ("bbq-37" — an fSim-style hardware set),
//! 2. **conformance**: prove every backend agrees with the reference on
//!    every gate in the set before trusting any numbers,
//! 3. **benchmark**: measure the efficiency each representation buys on
//!    named workloads — wall time, memory, support size — with correctness
//!    re-checked in the same run.
//!
//! A future backend (matrix-product states, say) plugs into exactly this
//! flow: register it by name, run the same two calls, read the same report.
//!
//! Run with `cargo run --release --example research_mode`.

use quantsim::prelude::*;

/// The "bbq-37" research gate set: fSim(θ, φ) plus a doubly controlled
/// phase, layered over the standard library.
fn register_bbq37(sim: &mut Simulator) -> Result<()> {
    sim.registry_mut().register_parametric(
        "bbq_fsim",
        "fSim(θ, φ): XY rotation + conditional phase",
        2,
        2,
        |p| {
            let (theta, phi) = (p[0], p[1]);
            let (c, s) = (theta.cos(), theta.sin());
            let (o, l) = (c64(1.0, 0.0), c64(0.0, 0.0));
            GateMatrix::from_vec(
                4,
                vec![
                    o,
                    l,
                    l,
                    l,
                    l,
                    c64(c, 0.0),
                    c64(0.0, -s),
                    l,
                    l,
                    c64(0.0, -s),
                    c64(c, 0.0),
                    l,
                    l,
                    l,
                    l,
                    cis(-phi),
                ],
            )
        },
    )?;
    sim.registry_mut().register_parametric(
        "bbq_ccphase",
        "doubly controlled phase",
        3,
        1,
        |p| {
            let mut m = GateMatrix::identity(8)?;
            m.set(7, 7, cis(p[0]));
            Ok(m)
        },
    )?;
    Ok(())
}

fn main() -> Result<()> {
    let mut sim: Simulator = Simulator::new();
    register_bbq37(&mut sim)?;
    println!(
        "registry: {} gates (bbq-37 layered over standard)\n",
        sim.registry().len()
    );

    // The two research representations register by name like any other.
    sim.backends_mut()
        .register("phase-field", |n| Ok(Box::new(PhaseFieldState::new(n)?)))?;
    sim.backends_mut()
        .register("braided", |n| Ok(Box::new(BraidedState::new(n)?)))?;

    // ── Step 1: conformance before benchmarks ─────────────────────────────
    // Every standard representation must pass the registry-wide sweep —
    // the custom bbq-37 gates included, discovered automatically — before
    // a single benchmark number is trusted.
    let cfg = ConformanceConfig::default();
    for backend in [
        "sparse",
        "adaptive",
        "factored",
        "mps",
        "mera",
        "phase-field",
        "braided",
    ] {
        let report = verify_backend(&sim, backend, &cfg)?;
        print!("{report}");
        assert!(report.passed(), "do not benchmark an unverified backend");
    }

    // ── Step 2: measure what each representation buys ─────────────────────
    let workloads = vec![
        Workload::qft(12),
        Workload::from_circuit(
            "bbq37-random-10q",
            random_registry_circuit(sim.registry(), 10, 120, 37),
        ),
        Workload::from_circuit("grover-12", library::grover(12, 1337, 3)?),
    ];
    let report = compare_backends(
        &sim,
        &workloads,
        &["dense", "sparse", "adaptive", "factored", "mps", "mera"],
        &BenchConfig::default(),
    )?;
    println!("\n{report}");

    // GHZ-20 separately: the mera rung-1 limitation is that a chain
    // crossing the root cut materializes the 2^20 block (a slow SVD), so
    // it sits this workload out — stated, not hidden. MPS holds the same
    // state at bond 2.
    let ghz = compare_backends(
        &sim,
        &[Workload::ghz(20)],
        &["dense", "sparse", "adaptive", "factored", "mps"],
        &BenchConfig::default(),
    )?;
    println!("{ghz}");

    // ── The two research representations, on the families they assume ─────
    // phase-field's assumption is a diagonal core over an affine subcube;
    // braided's is that the circuit is a word in the realization's
    // generators. Both are priced here against dense on a width all of
    // them can hold (braided derives 2n Majorana strands, capped at 16).
    let iqp_core = |n: usize| {
        let mut c: Circuit = Circuit::new(n);
        for q in 0..n {
            c.h(q);
        }
        for a in 0..n {
            for b in a + 1..n {
                c.cz(a, b);
            }
        }
        for q in 0..n {
            c.t(q);
        }
        c
    };
    let braid = |n: usize| {
        let gens = Realization::Majorana.generators(n).unwrap();
        let all: Vec<usize> = (0..n).collect();
        let mut c: Circuit = Circuit::new(n);
        for k in 0..40 {
            c.raw(
                format!("sigma{}", k % gens.len()),
                gens[k % gens.len()].clone(),
                all.clone(),
            );
        }
        c
    };
    let research = compare_backends(
        &sim,
        &[
            Workload::from_circuit("iqp-core-8", iqp_core(8)),
            Workload::from_circuit("iqp-full-8", {
                let mut c = iqp_core(8);
                for q in 0..8 {
                    c.h(q);
                }
                c
            }),
            Workload::from_circuit("braid-word-8", braid(8)),
        ],
        &["dense", "sparse", "mps", "phase-field", "braided"],
        &BenchConfig::default(),
    )?;
    println!("\n{research}");
    println!(
        "(iqp-core holds phase-field's assumption; iqp-full breaks it with the\n \
         closing Hadamard layer; braid-word-8 is a word in braided's generators)"
    );
    println!(
        "(mera skips ghz-20: a root-crossing chain pays the 2^20 block — the\n rung-1 cost path updates on the roadmap remove; see examples/coarse_register)"
    );
    let max_dev = report.max_deviation().max(ghz.max_deviation());
    println!("\nmax deviation anywhere: {max_dev:.2e}");
    let report = ghz;

    // ── The headline numbers, extracted programmatically ─────────────────
    if let (Some(speed), Some(mem)) = (
        report.speedup("ghz-20", "sparse"),
        report.memory_ratio("ghz-20", "sparse"),
    ) {
        println!(
            "\nghz-20 via sparse: {speed:.0}× faster, {mem:.0}× smaller — verified identical."
        );
    }
    Ok(())
}
