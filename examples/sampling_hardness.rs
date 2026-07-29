//! The sampling task, measured: XEB against exact references, exact
//! sampling on the certified-easy fragments, per-shot Clifford
//! sampling, and the measured price of spoofing the candidate family.
//!
//! Run with `cargo run --release --example sampling_hardness`.

use quantsim::exact::ExactState;
use quantsim::prelude::*;
use std::collections::HashMap;

fn main() -> Result<()> {
    let reg = GateRegistry::<C64>::standard();

    // ── 1. Calibration ───────────────────────────────────────────────
    println!("── XEB calibration (random circuit, n = 10, 4096 shots)");
    let rc = library::random_circuit(10, 300, 7);
    let ideal = Simulator::<C64>::new().run(&rc)?;
    let s = score_samples(ideal.as_ref(), &ideal.sample(4096, &mut Prng::new(5))?);
    println!(
        "  ideal sampler:   raw {:+.3}  ceiling {:.3}  normalized {:+.3}",
        s.raw,
        s.ceiling,
        s.normalized.unwrap()
    );
    let mut uniform: HashMap<u64, u64> = HashMap::new();
    let mut rng = Prng::new(9);
    for _ in 0..4096 {
        *uniform.entry(rng.next_u64() & 1023).or_insert(0) += 1;
    }
    let s = score_samples(ideal.as_ref(), &uniform);
    println!(
        "  uniform sampler: raw {:+.3}  ceiling {:.3}  normalized {:+.3}",
        s.raw,
        s.ceiling,
        s.normalized.unwrap()
    );
    println!("  scores are normalized by the MEASURED ceiling (2^n Σp² − 1), not an");
    println!("  assumed Porter–Thomas 1 — GHZ's ceiling is 2^(n−1) − 1, and a uniform");
    println!("  output distribution has ceiling exactly 0: no XEB signal exists there.\n");

    // ── 2. The certified-easy fragments SAMPLE at certified cost ─────
    println!("── easy fragments: the atlas verdicts carry over to the task");
    let ghz = library::ghz(16);
    let ghz_ideal = Simulator::<C64>::new().run(&ghz)?;
    let sparse = Simulator::<C64>::new().run_on("sparse", &ghz)?;
    let s = score_samples(ghz_ideal.as_ref(), &sparse.sample(4096, &mut Prng::new(5))?);
    println!(
        "  ghz-16 via sparse ({} B): normalized {:+.6}",
        sparse.memory_bytes(),
        s.normalized.unwrap()
    );
    let wide = library::ghz(20);
    let wide_ideal = Simulator::<C64>::new().run_on("sparse", &wide)?;
    let t0 = std::time::Instant::now();
    let counts = clifford_sample(&wide, &reg, 256, &mut Prng::new(11))?;
    let s = score_samples(wide_ideal.as_ref(), &counts);
    println!(
        "  ghz-20 via per-shot tableau measurement: normalized {:+.6}, 256 shots in {:.1?}",
        s.normalized.unwrap(),
        t0.elapsed()
    );
    println!("  (Gottesman–Knill for the TASK: polynomial per shot, no state flush.)");
    let flat = library::brickwork(10, 10, &(0..10).collect::<Vec<_>>());
    let flat_ideal = Simulator::<C64>::new().run(&flat)?;
    let s = score_samples(
        flat_ideal.as_ref(),
        &clifford_sample(&flat, &reg, 64, &mut Prng::new(7))?,
    );
    println!(
        "  uniform-output Clifford circuit: ceiling {:.1e} → normalized {:?} (no signal,",
        s.ceiling, s.normalized
    );
    println!("  reported as such rather than as a fake fidelity).\n");

    // ── 3. The exact reference ───────────────────────────────────────
    let mut ct: Circuit = Circuit::new(8);
    for q in 0..8 {
        ct.h(q);
    }
    ct.t(0)
        .t(3)
        .cx(0, 1)
        .cx(2, 3)
        .t(1)
        .cx(4, 5)
        .t(6)
        .cx(6, 7)
        .h(0)
        .h(3);
    let ct_ideal = Simulator::<C64>::new().run(&ct)?;
    let samples = ct_ideal.sample(2048, &mut Prng::new(17))?;
    let dense_score = linear_xeb(ct_ideal.as_ref(), &samples);
    let exact_score = linear_xeb_exact(&ExactState::run(&ct)?, &samples)?;
    println!("── the exact reference (Clifford+T fragment)");
    println!(
        "  same samples, dense vs D[ω] probabilities: {dense_score:.6} vs {exact_score:.6} (Δ {:.1e})",
        (dense_score - exact_score).abs()
    );
    println!("  the task can be scored with no float in the reference path.\n");

    // ── 4. Spoofing the candidate family ─────────────────────────────
    println!("── truncated-MPS spoofing of a depth-3n² random circuit (n = 10, 2048 shots)");
    println!(
        "  {:>6} {:>10} {:>12} {:>12} {:>12}",
        "χ cap", "peak χ", "memory", "time", "normalized"
    );
    for p in mps_spoof_curve(
        &library::random_circuit(10, 300, 7),
        &reg,
        &[2, 4, 8, 32],
        2048,
        13,
    )? {
        println!(
            "  {:>6} {:>10} {:>10} B {:>9.0} ms {:>+12.3}",
            p.max_bond,
            p.peak_bond,
            p.memory,
            p.nanos as f64 / 1e6,
            p.score.normalized.unwrap()
        );
    }
    println!("  every truncated cap collapses below 0.3 — compounding truncation error —");
    println!("  and only FULL rank (χ = 2^(n/2)) reaches the ceiling. The wall is a");
    println!("  cliff, not a slope: fidelity is bought with the full state bound.\n");

    println!("── fixed-budget spoofing vs family size (χ = 8, 2048 shots)");
    for d in spoof_decay(
        |n| library::random_circuit(n, 3 * n * n, 7),
        &[8, 10, 12, 14],
        8,
        2048,
        13,
    )? {
        println!(
            "  n = {:>2}: normalized {:+.3}",
            d.size,
            d.normalized.unwrap()
        );
    }
    println!("  the fixed budget stops scoring almost immediately: the sampling task");
    println!("  inherits the state bounds — which is precisely what an advantage");
    println!("  experiment claims, here as a measured artifact with fixed seeds.");
    Ok(())
}
