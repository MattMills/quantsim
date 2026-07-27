//! Reproductions of canonical quantum algorithms and states, checked
//! against their closed-form results. These are the BQP acceptance tests:
//! if the simulator disagrees with any of them, something fundamental broke.

#![allow(clippy::assertions_on_constants)] // algebra-flag pins are intentional

mod common;

use common::*;
use quantsim::prelude::*;
use std::f64::consts::{FRAC_1_SQRT_2, TAU};

#[test]
fn ghz_state_on_every_algebra() {
    // GHZ uses only h and cx — the real subset — so it runs over every
    // shipped algebra, dense and sparse alike.
    fn check<S: Scalar>() {
        let sim = Simulator::<S>::new();
        for backend in ["dense", "sparse", "adaptive"] {
            let state = sim.run_on(backend, &library::ghz(4)).unwrap();
            let top = (1u64 << 4) - 1;
            assert!(
                state
                    .amplitude(0)
                    .approx_eq(S::from_re(FRAC_1_SQRT_2), 1e-9),
                "{}/{backend}",
                S::algebra_name()
            );
            assert!(state
                .amplitude(top)
                .approx_eq(S::from_re(FRAC_1_SQRT_2), 1e-9));
            assert_eq!(state.nonzero_count(), 2);
        }
    }
    check::<f64>();
    check::<C64>();
    check::<CComplex>();
    check::<Quaternion>();
    check::<Octonion>();
    check::<Sedenion>();
    check::<SplitComplex>();
}

#[test]
fn qft_matches_the_dft_formula() {
    // QFT|x⟩ = 2^{-n/2} Σ_y ω^{xy} |y⟩ with ω = e^{2πi/2^n}.
    let n = 4;
    let dim = 1u64 << n;
    for x in [0u64, 1, 5, 9, 15] {
        let mut c = Circuit::new(n);
        for q in 0..n {
            if (x >> q) & 1 == 1 {
                c.x(q);
            }
        }
        c.append(&library::qft(n), &(0..n).collect::<Vec<_>>());
        let state = run_dense(&c);
        let norm = 1.0 / (dim as f64).sqrt();
        for y in 0..dim {
            let phase = TAU * (x * y % dim) as f64 / dim as f64;
            let expected = c64(norm * phase.cos(), norm * phase.sin());
            let got = state.amplitude(y);
            assert!(
                got.approx_eq(expected, 1e-9),
                "x={x} y={y}: {got} vs {expected}"
            );
        }
    }
}

#[test]
fn quantum_phase_estimation_exact() {
    // Estimate φ of the eigenphase e^{2πiφ} of p(2πφ) on |1⟩, with t = 4
    // counting qubits. For φ = k/16 the answer is exact and deterministic.
    let t = 4;
    for k in [1u64, 5, 11, 15] {
        let phi = k as f64 / 16.0;
        let mut c = Circuit::new(t + 1);
        c.x(t); // eigenstate |1⟩ on the target
        for j in 0..t {
            c.h(j);
        }
        for j in 0..t {
            // Controlled-P^{2^j}: phase 2π φ 2^j, control = counting qubit j.
            c.cp(j, t, TAU * phi * (1u64 << j) as f64);
        }
        c.append(&library::iqft(t), &(0..t).collect::<Vec<_>>());
        let state = run_dense(&c);
        // Counting register reads k with certainty (target bit stays set).
        let expected_index = k | (1u64 << t);
        assert_close(state.probability(expected_index), 1.0, 1e-9);
    }
}

#[test]
fn grover_success_probability_closed_form() {
    // After k iterations on n qubits with one marked state,
    // P(marked) = sin²((2k+1)·asin(2^{-n/2})).
    for (n, marked, iters) in [(3usize, 5u64, 2usize), (4, 11, 3), (5, 19, 4)] {
        let circuit = library::grover(n, marked, iters).unwrap();
        let state = run_dense(&circuit);
        let theta = (1.0 / (1u64 << n) as f64).sqrt().asin();
        let expected = ((2 * iters + 1) as f64 * theta).sin().powi(2);
        assert_close(state.probability(marked), expected, 1e-9);
        // The marked state dominates every other outcome.
        let probs = state.probabilities();
        let best = probs.iter().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        assert_eq!(best.0, marked, "n={n}");
        // And the classic figures: 94.5% at (3,2), 96.1% at (4,3).
        if n == 3 {
            assert_close(expected, 0.9453125, 1e-7);
        }
        if n == 4 {
            assert_close(expected, 0.96131897, 1e-7);
        }
    }
}

#[test]
fn grover_agrees_across_backends() {
    let circuit = library::grover(4, 6, 3).unwrap();
    let dense = run_dense(&circuit);
    for backend in ["sparse", "adaptive"] {
        let other = run_named(backend, &circuit);
        for i in 0..16u64 {
            assert!(
                dense.amplitude(i).approx_eq(other.amplitude(i), 1e-9),
                "{backend} idx {i}"
            );
        }
    }
}

#[test]
fn deutsch_jozsa_constant_vs_balanced() {
    let n = 3; // query register q0..q2, ancilla q3
    let all_zero_prob = |oracle: &dyn Fn(&mut Circuit)| -> f64 {
        let mut c = Circuit::new(n + 1);
        c.x(n).h(n);
        for q in 0..n {
            c.h(q);
        }
        oracle(&mut c);
        for q in 0..n {
            c.h(q);
        }
        let state = run_dense(&c);
        // Marginal probability that the query register reads 0…0.
        state.probability(0) + state.probability(1 << n)
    };

    // Constant f ≡ 0 and f ≡ 1: query register is certainly all zeros.
    assert_close(all_zero_prob(&|_c| {}), 1.0, 1e-9);
    assert_close(
        all_zero_prob(&|c| {
            c.x(n);
        }),
        1.0,
        1e-9,
    );
    // Balanced parity f(x) = x0 ⊕ x1 ⊕ x2: never all zeros.
    assert_close(
        all_zero_prob(&|c| {
            c.cx(0, n).cx(1, n).cx(2, n);
        }),
        0.0,
        1e-9,
    );
    // Balanced f(x) = x1: never all zeros.
    assert_close(
        all_zero_prob(&|c| {
            c.cx(1, n);
        }),
        0.0,
        1e-9,
    );
}

#[test]
fn bernstein_vazirani_recovers_hidden_string() {
    let n = 5;
    for s in [0b10110u64, 0b00001, 0b11111, 0b01010] {
        let mut c = Circuit::new(n + 1);
        c.x(n).h(n);
        for q in 0..n {
            c.h(q);
        }
        for q in 0..n {
            if (s >> q) & 1 == 1 {
                c.cx(q, n);
            }
        }
        for q in 0..n {
            c.h(q);
        }
        let state = run_dense(&c);
        // Query register reads s with certainty (ancilla in |−⟩: two indices).
        let p = state.probability(s) + state.probability(s | (1 << n));
        assert_close(p, 1.0, 1e-9);
    }
}

#[test]
fn superdense_coding_transmits_two_bits() {
    for b1 in [false, true] {
        for b0 in [false, true] {
            let mut c = Circuit::new(2);
            c.h(0).cx(0, 1); // shared Bell pair
            if b0 {
                c.x(0); // encode low bit
            }
            if b1 {
                c.z(0); // encode high bit
            }
            c.cx(0, 1).h(0); // decode
            let state = run_dense(&c);
            // Deterministic readout: q0 = b1, q1 = b0.
            let expected = (b1 as u64) | ((b0 as u64) << 1);
            assert_close(state.probability(expected), 1.0, 1e-9);
        }
    }
}

#[test]
fn simon_like_parity_correlations_in_ghz() {
    use Pauli::*;
    // GHZ(3): ⟨XXX⟩ = 1, ⟨XYY⟩ = ⟨YXY⟩ = ⟨YYX⟩ = −1 — the Mermin/GHZ
    // paradox correlations (no local hidden-variable model matches all four).
    let state = run_dense(&library::ghz(3));
    let e = |ops: &[(usize, Pauli)]| pauli_expectation(state.as_ref(), ops).unwrap().re;
    assert_close(e(&[(0, X), (1, X), (2, X)]), 1.0, TOL);
    assert_close(e(&[(0, X), (1, Y), (2, Y)]), -1.0, TOL);
    assert_close(e(&[(0, Y), (1, X), (2, Y)]), -1.0, TOL);
    assert_close(e(&[(0, Y), (1, Y), (2, X)]), -1.0, TOL);
}

#[test]
fn w_state_via_load_and_measurement_cascade() {
    // Load the 3-qubit W state directly (backends accept arbitrary states),
    // then check the defining property: measuring any qubit as 0 leaves the
    // remaining pair in a smaller W; measuring 1 collapses the rest to |00⟩.
    let w = 1.0 / 3.0f64.sqrt();
    let entries: Vec<(u64, C64)> = vec![
        (0b001, c64(w, 0.0)),
        (0b010, c64(w, 0.0)),
        (0b100, c64(w, 0.0)),
    ];
    let mut state = DenseState::<C64>::new(3).unwrap();
    state.load(&entries).unwrap();

    let mut rng = Prng::new(3);
    let outcome = state.measure(0, &mut rng).unwrap();
    if outcome {
        // Remaining qubits are |00⟩.
        assert_close(state.probability(0b001), 1.0, TOL);
    } else {
        // Remaining pair is (|01⟩ + |10⟩)/√2 on qubits 1, 2.
        assert_close(state.probability(0b010), 0.5, TOL);
        assert_close(state.probability(0b100), 0.5, TOL);
    }
}

#[test]
fn quaternion_and_octonion_runs_embed_complex_results() {
    // A complex circuit simulated over ℍ or 𝕆 must reproduce the ℂ result
    // exactly in the first two coordinates, with nothing leaking into the
    // extra imaginary units — the embedding ℂ ⊂ ℍ ⊂ 𝕆 is a homomorphism.
    let circuit_c: Circuit = library::random_circuit(4, 30, 777);
    let circuit_h: Circuit<Quaternion> = library::random_circuit(4, 30, 777);
    let circuit_o: Circuit<Octonion> = library::random_circuit(4, 30, 777);

    let sc = Simulator::<C64>::new().run(&circuit_c).unwrap();
    let sh = Simulator::<Quaternion>::new().run(&circuit_h).unwrap();
    let so = Simulator::<Octonion>::new().run(&circuit_o).unwrap();

    for i in 0..16u64 {
        let z = sc.amplitude(i);
        let q = sh.amplitude(i).coeffs();
        let o = so.amplitude(i).coeffs();
        assert_close(q[0], z.re, 1e-9);
        assert_close(q[1], z.im, 1e-9);
        assert!(
            q[2].abs() < 1e-9 && q[3].abs() < 1e-9,
            "leak into j/k at {i}"
        );
        assert_close(o[0], z.re, 1e-9);
        assert_close(o[1], z.im, 1e-9);
        assert!(
            o[2..].iter().all(|c| c.abs() < 1e-9),
            "leak into octonion units at {i}"
        );
    }
}

#[test]
fn real_algebra_simulates_real_circuits() {
    // Over ℝ, the available gates are real; a CHSH-style Ry/CX circuit runs
    // and preserves norm.
    let mut c: Circuit<f64> = Circuit::new(2);
    c.ry(0, 1.2).cx(0, 1).ry(1, -0.7).cz(0, 1).h(0);
    let state = Simulator::<f64>::new().run(&c).unwrap();
    assert_close(state.total_weight(), 1.0, 1e-9);
    // And trying to bind a complex gate over ℝ fails loudly.
    let mut bad: Circuit<f64> = Circuit::new(1);
    bad.gate("s", Vec::new(), vec![0]);
    assert!(matches!(
        bad.bind(Simulator::<f64>::new().registry()),
        Err(Error::UnknownGate(name)) if name == "s"
    ));
}

#[test]
fn sedenion_zero_divisors_break_born_conservation_by_construction() {
    // Unitary-looking gates built from zero-divisor directions do not
    // conserve Born weight over 𝕊 — the simulator surfaces this rather than
    // hiding it. Build a state along one zero-divisor factor and "rotate"
    // with a raw matrix along the other; DIVISION = false warns us.
    assert!(!Sedenion::DIVISION);
    // (e1 + e10) and (e5 + e14) multiply to zero in the CD basis (one of the
    // standard sedenion zero-divisor pairs; verified programmatically in the
    // scalar unit tests).
    let x = Sedenion::basis(1) + Sedenion::basis(10);
    let y = Sedenion::basis(5) + Sedenion::basis(14);
    let prod = x * y;
    // If this particular pair is nonzero under our basis convention, find
    // one that is zero — existence is guaranteed.
    let is_zero = prod.is_zero(1e-12);
    let (x, _y) = if is_zero {
        (x, y)
    } else {
        let mut found = None;
        'outer: for p in 1..16 {
            for q in (p + 1)..16 {
                for r in 1..16 {
                    for s in (r + 1)..16 {
                        let a = Sedenion::basis(p) + Sedenion::basis(q);
                        let b = Sedenion::basis(r) - Sedenion::basis(s);
                        if (a * b).is_zero(1e-12) {
                            found = Some((a, b));
                            break 'outer;
                        }
                    }
                }
            }
        }
        found.expect("sedenions contain zero divisors")
    };
    let half = x.scale(0.5); // |x|² = 2 → x/2 has Born weight 1/2 per slot
    let mut state = DenseState::<Sedenion>::new(1).unwrap();
    state.load(&[(0, half), (1, half)]).unwrap();
    assert_close(state.total_weight(), 1.0, 1e-9);
    // A norm-1 sedenion that fails multiplicativity against x scrambles the
    // total weight when used as a "phase":
    let u = (Sedenion::basis(5) + Sedenion::basis(14)).scale(FRAC_1_SQRT_2);
    assert_close(u.abs_sqr(), 1.0, 1e-12);
    let mut m = GateMatrix::<Sedenion>::identity(2).unwrap();
    m.set(1, 1, u);
    // m†m = I entrywise (conj(u)·u = |u|² = 1 in this algebra? Not
    // necessarily — compute and accept either; the point is weight drift).
    state.apply(&m, &[0]).unwrap();
    let w = state.total_weight();
    // Weight moved away from 1: (u · x/2) has weight ≠ 1/2 because
    // |u·x| ≠ |u||x| near zero-divisor directions.
    assert!(
        (w - 1.0).abs() > 1e-3,
        "expected Born-weight drift near sedenion zero divisors, got {w}"
    );
}
