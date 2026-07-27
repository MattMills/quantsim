//! The exact D[ω] evaluator as the absolute reference: the dense
//! backend's float error becomes a *measured* quantity, exact zeros are
//! decidable, and unitarity holds exactly rather than within tolerance.

mod common;

use common::assert_close;
use quantsim::exact::{DOmega, ExactReal, ExactState};
use quantsim::prelude::*;

/// A deterministic circuit over the exact-supported fragment: Clifford+T
/// plus snapped-angle rotations.
fn exact_fragment_circuit(n: usize, gates: usize, seed: u64) -> Circuit {
    let pi4 = std::f64::consts::FRAC_PI_4;
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for _ in 0..gates {
        let q = (rng.next_u64() % n as u64) as usize;
        let r = ((q + 1) + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
        match rng.next_u64() % 12 {
            0 => c.h(q),
            1 => c.t(q),
            2 => c.tdg(q),
            3 => c.s(q),
            4 => c.x(q),
            5 => c.sx(q),
            6 => c.rz(q, 2.0 * pi4 * (1 + rng.next_u64() % 7) as f64),
            7 => c.cx(q, r),
            8 => c.cz(q, r),
            9 => c.cp(q, r, pi4 * (1 + rng.next_u64() % 7) as f64),
            10 => c.swap(q, r),
            _ => c.ry(q, 2.0 * pi4 * (1 + rng.next_u64() % 7) as f64),
        };
    }
    c
}

#[test]
fn exact_certifies_the_dense_reference() {
    // The reference backend's true float error, measured against the ring
    // — not against itself. Also: total Born weight is *exactly* 1, an
    // assertion no float backend can make.
    for (n, gates, seed) in [(6usize, 120usize, 1u64), (8, 160, 2), (10, 200, 3)] {
        let circuit = exact_fragment_circuit(n, gates, seed);
        let exact = ExactState::run(&circuit).unwrap();
        let weight = exact.total_weight_exact().unwrap();
        assert_eq!(
            weight,
            ExactReal {
                int: 1,
                sqrt2: 0,
                k: 0
            },
            "unitarity holds exactly in the ring"
        );

        let sim: Simulator = Simulator::new();
        let dense = sim.run(&circuit).unwrap();
        let dev = exact.max_deviation_vs(dense.as_ref());
        assert!(
            dev < 1e-12,
            "dense float error vs exact: {dev:.3e} (n={n}, {gates} gates)"
        );
        // Sparse measured against the same absolute yardstick.
        let sparse = sim.run_on("sparse", &circuit).unwrap();
        let dev_sparse = exact.max_deviation_vs(sparse.as_ref());
        assert!(dev_sparse < 1e-12, "sparse vs exact: {dev_sparse:.3e}");
    }
}

#[test]
fn exact_zeros_are_exact_where_floats_leave_residue() {
    // H · T⁴ · H = H·Z·H = X, so |0⟩ ↦ |1⟩ with amplitude(|0⟩) exactly
    // zero. The float path accumulates cis(π/4)⁴ ≠ −1 rounding and leaves
    // a ~1e-16 residue — real, measured here — while the ring answers
    // zero, exactly, and support 1, decidably.
    let mut c: Circuit = Circuit::new(1);
    c.h(0).t(0).t(0).t(0).t(0).h(0);
    let exact = ExactState::run(&c).unwrap();
    assert!(exact.amplitude_exact(0).is_zero(), "exact zero is zero");
    assert_eq!(exact.support_exact(), 1);
    assert_eq!(exact.amplitude_exact(1), DOmega::int(-1).mul(DOmega::omega_pow(4)).unwrap());

    let sim: Simulator = Simulator::new();
    let dense = sim.run(&c).unwrap();
    let residue = dense.amplitude(0).norm();
    assert!(
        residue > 0.0 && residue < 1e-14,
        "the float residue the exact path removes: {residue:.3e}"
    );
    // And T⁸ = identity — exactly, not within tolerance.
    let mut c8: Circuit = Circuit::new(1);
    c8.h(0);
    for _ in 0..8 {
        c8.t(0);
    }
    c8.h(0);
    let exact8 = ExactState::run(&c8).unwrap();
    assert_eq!(exact8.amplitude_exact(0), DOmega::int(1));
    assert_eq!(exact8.support_exact(), 1);
}

#[test]
fn ghz_amplitudes_are_exactly_one_over_sqrt2() {
    let n = 12;
    let exact = ExactState::run(&library::ghz(n)).unwrap();
    assert_eq!(exact.support_exact(), 2);
    let h = DOmega::inv_sqrt2_pow(1);
    assert_eq!(exact.amplitude_exact(0), h);
    assert_eq!(exact.amplitude_exact((1 << n) - 1), h);
    let p = exact.probability_exact(0).unwrap();
    assert_eq!(
        p,
        ExactReal {
            int: 1,
            sqrt2: 0,
            k: 1
        },
        "P(|0…0⟩) is exactly 1/2"
    );
}

#[test]
fn grover_diagonal_oracles_run_exactly() {
    // The ±1 phase-flip diagonals are eighth-turn phases, so the whole
    // Grover circuit evaluates in the ring; the closed-form success
    // probability is reproduced through one final rounding.
    let (n, marked, iterations) = (5usize, 19u64, 3usize);
    let circuit = library::grover::<C64>(n, marked, iterations).unwrap();
    let exact = ExactState::run(&circuit).unwrap();
    assert_eq!(
        exact.total_weight_exact().unwrap(),
        ExactReal {
            int: 1,
            sqrt2: 0,
            k: 0
        }
    );
    let sim: Simulator = Simulator::new();
    let dense = sim.run(&circuit).unwrap();
    assert!(exact.max_deviation_vs(dense.as_ref()) < 1e-12);
    let theta = (1.0 / (1u64 << n) as f64).sqrt().asin();
    let closed_form = ((2 * iterations + 1) as f64 * theta).sin().powi(2);
    assert_close(
        exact.probability_exact(marked).unwrap().to_f64(),
        closed_form,
        1e-12,
    );
}

#[test]
fn overflow_fails_loudly_instead_of_wrapping() {
    let big = DOmega::int(i64::MAX);
    let sq = big.mul(big).unwrap(); // ~2^126: still fits
    assert!(sq.mul(big).is_err(), "third power must overflow i128");
}
