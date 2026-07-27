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

    // ── Step 1: conformance before benchmarks ─────────────────────────────
    let cfg = ConformanceConfig::default();
    for backend in ["sparse", "adaptive"] {
        let report = verify_backend(&sim, backend, &cfg)?;
        print!("{report}");
        assert!(report.passed(), "do not benchmark an unverified backend");
    }

    // ── Step 2: measure what each representation buys ─────────────────────
    let workloads = vec![
        Workload::ghz(20),
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
        &["dense", "sparse", "adaptive"],
        &BenchConfig::default(),
    )?;
    println!("\n{report}");
    println!("max deviation anywhere: {:.2e}", report.max_deviation());

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
