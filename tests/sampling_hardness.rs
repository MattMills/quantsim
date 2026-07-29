//! Sampling-task hardness, measured: XEB calibration against dense and
//! exact references, exact sampling at certified cost on the easy
//! fragments, polynomial per-shot Clifford sampling, and the measured
//! price of spoofing the candidate family with truncated tensor
//! networks. All seeds fixed — every number here reproduces.

mod common;

use quantsim::exact::ExactState;
use quantsim::prelude::*;
use std::collections::HashMap;

#[test]
fn xeb_is_calibrated_on_ideal_and_uniform_samplers() {
    // The score must read ≈ ceiling for the ideal sampler and ≈ 0 for
    // the uniform one, on a Porter–Thomas-like random circuit.
    let rc = library::random_circuit(10, 300, 7);
    let ideal = Simulator::<C64>::new().run(&rc).unwrap();

    let s = score_samples(
        ideal.as_ref(),
        &ideal.sample(4096, &mut Prng::new(5)).unwrap(),
    );
    let norm = s.normalized.unwrap();
    assert!(
        (norm - 1.0).abs() < 0.15,
        "ideal sampler scores ≈ 1: {norm}"
    );

    let mut uniform: HashMap<u64, u64> = HashMap::new();
    let mut rng = Prng::new(9);
    for _ in 0..4096 {
        *uniform.entry(rng.next_u64() & 1023).or_insert(0) += 1;
    }
    let s = score_samples(ideal.as_ref(), &uniform);
    let norm = s.normalized.unwrap();
    assert!(norm.abs() < 0.15, "uniform sampler scores ≈ 0: {norm}");

    // Structured families have wildly non-unit ceilings — GHZ's is
    // 2^{n-1} − 1 — which is exactly why scores are normalized by the
    // measured ceiling instead of an assumed Porter–Thomas 1.
    let ghz = library::ghz(10);
    let ghz_ideal = Simulator::<C64>::new().run(&ghz).unwrap();
    assert!((ideal_xeb(ghz_ideal.as_ref()) - 511.0).abs() < 1e-6);
    let s = score_samples(
        ghz_ideal.as_ref(),
        &ghz_ideal.sample(4096, &mut Prng::new(5)).unwrap(),
    );
    assert!((s.normalized.unwrap() - 1.0).abs() < 1e-9);
}

#[test]
fn the_exact_reference_scores_the_task_with_no_float_in_the_path() {
    // On the Clifford+T fragment the same samples score identically
    // against dense probabilities and against the exact D[ω] ring.
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
    let ideal = Simulator::<C64>::new().run(&ct).unwrap();
    let samples = ideal.sample(2048, &mut Prng::new(17)).unwrap();
    let dense_score = linear_xeb(ideal.as_ref(), &samples);
    let exact_score = linear_xeb_exact(&ExactState::run(&ct).unwrap(), &samples).unwrap();
    assert!(
        (dense_score - exact_score).abs() < 1e-9,
        "{dense_score} vs {exact_score}"
    );
}

#[test]
fn easy_families_are_sampled_exactly_at_certified_cost() {
    // The atlas certifies GHZ classical via sparse; the TASK follows:
    // the sparse representation's own Born sampling scores at the
    // ceiling, from 125 bytes.
    let ghz = library::ghz(16);
    let ideal = Simulator::<C64>::new().run(&ghz).unwrap();
    let sparse = Simulator::<C64>::new().run_on("sparse", &ghz).unwrap();
    assert_eq!(sparse.memory_bytes(), 125);
    let s = score_samples(
        ideal.as_ref(),
        &sparse.sample(4096, &mut Prng::new(5)).unwrap(),
    );
    assert!((s.normalized.unwrap() - 1.0).abs() < 1e-9);
}

#[test]
fn clifford_sampling_is_polynomial_per_shot() {
    // Per-shot native tableau measurement: bit convention pinned
    // (x(0) samples index 1, deterministically), then GHZ at width 20
    // sampled at the ceiling in milliseconds — no state-vector flush,
    // the Gottesman–Knill sampling story as a measured artifact.
    let reg = GateRegistry::<C64>::standard();
    let mut c: Circuit = Circuit::new(2);
    c.x(0);
    let counts = clifford_sample(&c, &reg, 16, &mut Prng::new(3)).unwrap();
    assert_eq!(counts.get(&1), Some(&16), "{counts:?}");

    let ghz = library::ghz(20);
    let ideal = Simulator::<C64>::new().run_on("sparse", &ghz).unwrap();
    let started = std::time::Instant::now();
    let counts = clifford_sample(&ghz, &reg, 256, &mut Prng::new(11)).unwrap();
    let elapsed = started.elapsed();
    assert_eq!(counts.len(), 2, "GHZ samples exactly two outcomes");
    let s = score_samples(ideal.as_ref(), &counts);
    assert!((s.normalized.unwrap() - 1.0).abs() < 1e-9);
    assert!(
        elapsed.as_secs_f64() < 5.0,
        "256 shots at width 20 stay cheap: {elapsed:.1?}"
    );

    // A uniform-output Clifford circuit carries NO XEB signal — the
    // ceiling is exactly zero and the harness says None instead of
    // reporting a fake fidelity.
    let flat = library::brickwork(10, 10, &(0..10).collect::<Vec<_>>());
    let flat_ideal = Simulator::<C64>::new().run(&flat).unwrap();
    let counts = clifford_sample(&flat, &reg, 64, &mut Prng::new(7)).unwrap();
    let s = score_samples(flat_ideal.as_ref(), &counts);
    assert!(s.ceiling.abs() < 1e-9);
    assert!(s.normalized.is_none());
}

#[test]
fn truncated_mps_spoofing_pays_full_rank_for_fidelity() {
    // The real-world spoofing strategy, measured on a depth-3n² random
    // circuit at n = 10: every truncated bond cap collapses the
    // normalized score below 0.3 (compounding truncation error), and
    // only the full-rank spoofer (χ = 2^{n/2} = 32) reaches the
    // ceiling. The curve is NOT a smooth rise — that is the measured
    // shape of the wall.
    let reg = GateRegistry::<C64>::standard();
    let points = mps_spoof_curve(
        &library::random_circuit(10, 300, 7),
        &reg,
        &[2, 4, 8, 32],
        2048,
        13,
    )
    .unwrap();
    for p in &points[..3] {
        let norm = p.score.normalized.unwrap();
        assert!(
            norm < 0.3,
            "χ = {} spoofing stays low-fidelity: {norm}",
            p.max_bond
        );
    }
    let full = points.last().unwrap();
    assert_eq!(full.peak_bond, 32);
    assert!(
        full.score.normalized.unwrap() > 0.9,
        "full rank samples the task: {:?}",
        full.score.normalized
    );
    assert!(
        full.memory > 3 * points[0].memory,
        "fidelity is bought with rank: {} vs {}",
        full.memory,
        points[0].memory
    );
}

#[test]
fn fixed_budget_spoofing_decays_with_size() {
    // Fix the spoofer's bond at χ = 8 and grow the candidate family:
    // the normalized score decays from ~0.41 at n = 8 to noise —
    // the sampling task inherits the state bounds.
    let decay = spoof_decay(
        |n| library::random_circuit(n, 3 * n * n, 7),
        &[8, 10, 12, 14],
        8,
        2048,
        13,
    )
    .unwrap();
    let first = decay[0].normalized.unwrap();
    assert!(first > 0.3, "χ = 8 still scores at n = 8: {first}");
    for d in &decay[1..] {
        let norm = d.normalized.unwrap();
        assert!(
            norm < 0.3,
            "n = {}: the fixed budget stops scoring: {norm}",
            d.size
        );
    }
}
