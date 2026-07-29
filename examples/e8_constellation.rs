//! The infinite E8 constellation, live: E8 copies positioned at the
//! points of E8 itself — one byte of coset address per scale level —
//! and the n-qubit backend built on the tower, from the measured
//! 256-class theorem to a 63-qubit GHZ stored as two lattice points.
//!
//! Run with `cargo run --release --example e8_constellation`.

use quantsim::e8::constellation::{self, E8ConstellationState, Shell};
use quantsim::prelude::*;

fn shell_name(s: Shell) -> &'static str {
    match s {
        Shell::Center => "center",
        Shell::RootSphere => "√2-sphere",
        Shell::FrameSphere => "2-sphere",
    }
}

fn main() -> Result<()> {
    // ── 1. The identity-position theorem: E8/2E8, measured ───────────
    println!("── one byte positions an E8 inside its parent (all measured)");
    let n4 = constellation::norm4_vectors();
    println!(
        "  shells of the parent copy: 1 origin, 240 roots (√2-sphere), {} norm-2",
        n4.len()
    );
    let mut census = [0usize; 3];
    for c in 0u16..256 {
        match constellation::shell_of(c as u8) {
            Shell::Center => census[0] += 1,
            Shell::RootSphere => census[1] += 1,
            Shell::FrameSphere => census[2] += 1,
        }
    }
    println!("  vectors (2-sphere). Mod 2E8 they fall into EXACTLY 256 classes:");
    println!(
        "  {} origin + {} antipodal root pairs + {} frames of 16 = 256 = one byte.",
        census[0], census[1], census[2]
    );
    println!("  The address map is linear (class(x+y) = class(x) XOR class(y)), so");
    println!("  \"the identity position of THIS E8\" is one byte choosing a point on");
    println!("  the concentric spheres of the copy one scale up.\n");

    // ── 2. The scale tower: an infinite constellation ────────────────
    println!("── the tower: position = Σ 2ᵏ·rep(digitₖ), one E8 per level");
    let digits = [137u8, 42, 200];
    let p = constellation::compose(&digits);
    println!("  digits {digits:?} compose to the lattice point {p:?};");
    println!(
        "  decompose recovers {:?} exactly.",
        constellation::decompose(&p, 3).unwrap()
    );
    for (k, &d) in digits.iter().enumerate() {
        println!(
            "    level {k}: digit {d:>3} sits on the {} at scale 2^{k}",
            shell_name(constellation::shell_of(d))
        );
    }
    let doubled: constellation::Point = std::array::from_fn(|k| 2 * p[k]);
    println!(
        "  doubling the point prepends digit 0: {:?} — the constellation",
        constellation::decompose(&doubled, 4).unwrap()
    );
    println!("  contains itself at every scale; the recursion never ends.\n");

    // ── 3. The n-qubit expansion ─────────────────────────────────────
    println!("── n qubits = one E8 point at resolution 2^⌈n/8⌉");
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut().register("e8-constellation", |n| {
        Ok(Box::new(E8ConstellationState::new(n)?))
    })?;
    sim.backends_mut().register("e8-rep", |n| {
        Ok(Box::new(quantsim::e8::rep::E8RepState::new(n)?))
    })?;
    let err = DenseState::<C64>::new(63).unwrap_err();
    println!("  dense at 63 qubits: {err}");
    let state = sim.run_on("e8-constellation", &library::ghz(63))?;
    let rep = state
        .as_any()
        .downcast_ref::<E8ConstellationState>()
        .expect("registered representation");
    println!(
        "  GHZ-63 on the constellation: {} stored points, {} bytes:",
        state.nonzero_count(),
        state.memory_bytes()
    );
    for point in rep.stored_points() {
        println!(
            "    {point:?}  (digits {:?})",
            constellation::decompose(&point, 8).unwrap()
        );
    }
    println!("  — eight scale levels deep; the census at every level is one center");
    println!(
        "    digit and one √2-sphere digit: {:?}.\n",
        rep.shell_census()[0]
    );

    // ── 4. The standard frameworks ───────────────────────────────────
    println!("── conformance + benchmark, beside the other representations");
    let cfg = ConformanceConfig {
        random_circuits: 8,
        orderings: 2,
        ..ConformanceConfig::default()
    };
    let report = verify_backend(&sim, "e8-constellation", &cfg)?;
    println!(
        "  conformance 'e8-constellation': {} over {} gates, max deviation {:.1e}",
        if report.passed() { "PASSED" } else { "FAILED" },
        report.gate_checks.len(),
        report.max_amplitude_deviation
    );
    let workloads = [
        Workload::from_circuit("ghz-16", library::ghz(16)),
        Workload::from_circuit("rainbow-16", library::rainbow(16)),
        Workload::from_circuit("qft-12", library::qft(12)),
    ];
    let backends = [
        "dense",
        "sparse",
        "factored",
        "mps",
        "e8-rep",
        "e8-constellation",
    ];
    let bench = compare_backends(&sim, &workloads, &backends, &BenchConfig::default())?;
    print!("{bench}");
    println!("  (e8-rep's rows record its measured native-8 wall; the constellation");
    println!("  runs every width. Concentrated states are where the tower pays —");
    println!("  GHZ-16 in two 80-byte points — and saturated QFT-12 shows the honest");
    println!("  cost: full support in lattice keys is larger than dense's flat");
    println!("  vector. Every completed run is amplitude-verified in-table.)");
    Ok(())
}
