//! Backend semantics: measurement, collapse, sampling determinism,
//! state loading, adaptive promotion, and expectation values — across all
//! three built-in representations.

mod common;

use common::*;
use quantsim::prelude::*;
use std::f64::consts::FRAC_1_SQRT_2;

const BACKENDS: [&str; 8] = [
    "dense", "sparse", "adaptive", "factored", "mps", "mera", "bulk", "mosaic",
];

#[test]
fn initial_state_is_all_zeros() {
    for name in BACKENDS {
        let state = sim().run_on(name, &Circuit::new(3)).unwrap();
        assert!(state.amplitude(0).approx_eq(c64(1.0, 0.0), TOL), "{name}");
        assert_eq!(state.nonzero_count(), 1, "{name}");
        assert_close(state.total_weight(), 1.0, TOL);
    }
}

#[test]
fn measurement_collapses_and_renormalizes() {
    for name in BACKENDS {
        let mut c = Circuit::new(2);
        c.h(0).cx(0, 1); // Bell pair
        let mut state = sim().run_on(name, &c).unwrap();
        let mut rng = Prng::new(1234);
        let outcome = state.measure(0, &mut rng).unwrap();
        // Collapse: the other qubit must now agree with certainty.
        let idx = if outcome { 0b11 } else { 0b00 };
        assert_close(state.probability(idx), 1.0, TOL);
        assert_close(state.total_weight(), 1.0, TOL);
        // Second measurement is deterministic regardless of rng.
        let outcome_q1 = state.measure(1, &mut rng).unwrap();
        assert_eq!(outcome_q1, outcome, "{name}: Bell correlation broken");
    }
}

#[test]
fn measurement_statistics_unbiased() {
    // 1000 fresh |+⟩ states with seeds 0..1000: outcome counts must sit
    // within a generous binomial window (deterministic given the seeds).
    for name in BACKENDS {
        let mut ones = 0u32;
        for seed in 0..1000 {
            let mut c = Circuit::new(1);
            c.h(0);
            let mut state = sim().run_on(name, &c).unwrap();
            if state.measure(0, &mut Prng::new(seed)).unwrap() {
                ones += 1;
            }
        }
        // Mean 500, σ ≈ 15.8; allow 4σ.
        assert!((437..=563).contains(&ones), "{name}: {ones}/1000 ones");
    }
}

#[test]
fn sampling_matches_probabilities_and_is_backend_independent() {
    let mut c = Circuit::new(3);
    c.h(0).h(1).h(2); // uniform over 8 outcomes
    let dense_counts = {
        let state = run_named("dense", &c);
        state.sample(8192, &mut Prng::new(42)).unwrap()
    };
    let total: u64 = dense_counts.values().sum();
    assert_eq!(total, 8192);
    for idx in 0..8u64 {
        let count = *dense_counts.get(&idx).unwrap_or(&0);
        // Mean 1024, σ ≈ 30; allow 4σ.
        assert!((904..=1144).contains(&count), "outcome {idx}: {count}");
    }
    // Same seed, other representations: byte-identical counts, because
    // sampling accumulates weights in basis order on every backend.
    for name in ["sparse", "adaptive", "factored", "mps", "mera", "bulk", "mosaic"] {
        let state = run_named(name, &c);
        let counts = state.sample(8192, &mut Prng::new(42)).unwrap();
        assert_eq!(counts, dense_counts, "{name} sampling differs from dense");
    }
}

#[test]
fn bell_sampling_never_yields_odd_parity() {
    for name in BACKENDS {
        let state = run_named(name, &library::bell());
        let counts = state.sample(4096, &mut Prng::new(7)).unwrap();
        assert_eq!(*counts.get(&0b01).unwrap_or(&0), 0, "{name}");
        assert_eq!(*counts.get(&0b10).unwrap_or(&0), 0, "{name}");
        let (z, o) = (
            *counts.get(&0b00).unwrap_or(&0),
            *counts.get(&0b11).unwrap_or(&0),
        );
        assert_eq!(z + o, 4096);
        assert!((1792..=2304).contains(&z), "{name}: {z} zeros"); // 4σ ≈ 256
    }
}

#[test]
fn project_manual() {
    for name in BACKENDS {
        let mut c = Circuit::new(1);
        c.h(0);
        let mut state = sim().run_on(name, &c).unwrap();
        state.project(0, true, std::f64::consts::SQRT_2);
        assert!(state.amplitude(1).approx_eq(c64(1.0, 0.0), TOL), "{name}");
        assert!(state.amplitude(0).approx_eq(c64(0.0, 0.0), TOL), "{name}");
    }
}

#[test]
fn load_and_reset() {
    let w = 1.0 / (3.0f64).sqrt();
    let entries: Vec<(u64, C64)> = vec![
        (0b001, c64(w, 0.0)),
        (0b010, c64(0.0, w)),
        (0b100, c64(-w, 0.0)),
    ];
    for name in BACKENDS {
        let mut state = sim().backends().create(name, 3).unwrap();
        state.load(&entries).unwrap();
        assert_close(state.total_weight(), 1.0, TOL);
        assert!(state.amplitude(0b010).approx_eq(c64(0.0, w), TOL), "{name}");
        assert_eq!(state.nonzero_count(), 3);
        state.reset();
        assert!(state.amplitude(0).approx_eq(c64(1.0, 0.0), TOL), "{name}");
        assert_eq!(state.nonzero_count(), 1);
        // Out-of-range load is rejected.
        assert!(state.load(&[(8, c64(1.0, 0.0))]).is_err(), "{name}");
    }
}

#[test]
fn probabilities_listing_sorted() {
    let state = run_dense(&library::ghz(3));
    let probs = state.probabilities();
    assert_eq!(probs.len(), 2);
    assert_eq!(probs[0].0, 0b000);
    assert_eq!(probs[1].0, 0b111);
    assert_close(probs[0].1, 0.5, TOL);
    assert_close(probs[1].1, 0.5, TOL);
}

#[test]
fn adaptive_promotion_on_density() {
    let sim: Simulator = Simulator::new();
    // GHZ keeps 2 nonzeros: stays sparse at any width.
    let state = sim.run_on("adaptive", &library::ghz(12)).unwrap();
    let adaptive = state.as_any().downcast_ref::<AdaptiveState<C64>>().unwrap();
    assert!(!adaptive.is_dense(), "GHZ must stay sparse");
    assert_eq!(state.nonzero_count(), 2);

    // A Hadamard layer saturates the space: must promote.
    let mut h_layer = Circuit::new(10);
    for q in 0..10 {
        h_layer.h(q);
    }
    let state = sim.run_on("adaptive", &h_layer).unwrap();
    let adaptive = state.as_any().downcast_ref::<AdaptiveState<C64>>().unwrap();
    assert!(adaptive.is_dense(), "uniform superposition must promote");
    assert_eq!(state.nonzero_count(), 1024);
    // And the amplitudes survive promotion intact.
    let expected = 1.0 / 32.0;
    for i in 0..1024u64 {
        assert!(state.amplitude(i).approx_eq(c64(expected, 0.0), TOL));
    }
}

#[test]
fn sparse_stays_exact_on_stabilizer_circuits() {
    // 40-qubit GHZ is far past dense reach; sparse handles it exactly.
    let state = run_named("sparse", &library::ghz(40));
    assert_eq!(state.nonzero_count(), 2);
    assert!(state.amplitude(0).approx_eq(c64(FRAC_1_SQRT_2, 0.0), TOL));
    assert!(state
        .amplitude((1u64 << 40) - 1)
        .approx_eq(c64(FRAC_1_SQRT_2, 0.0), TOL));
}

#[test]
fn pauli_expectations() {
    use Pauli::*;
    let bell = run_dense(&library::bell());
    assert_close(
        pauli_expectation(bell.as_ref(), &[(0, Z), (1, Z)])
            .unwrap()
            .re,
        1.0,
        TOL,
    );
    assert_close(
        pauli_expectation(bell.as_ref(), &[(0, X), (1, X)])
            .unwrap()
            .re,
        1.0,
        TOL,
    );
    assert_close(
        pauli_expectation(bell.as_ref(), &[(0, Y), (1, Y)])
            .unwrap()
            .re,
        -1.0,
        TOL,
    );
    assert_close(
        pauli_expectation(bell.as_ref(), &[(0, Z)]).unwrap().re,
        0.0,
        TOL,
    );
    assert_close(
        pauli_expectation(bell.as_ref(), &[(0, I), (1, Z)])
            .unwrap()
            .re,
        0.0,
        TOL,
    );

    let mut plus = Circuit::new(1);
    plus.h(0);
    let plus = run_dense(&plus);
    assert_close(
        pauli_expectation(plus.as_ref(), &[(0, X)]).unwrap().re,
        1.0,
        TOL,
    );
    assert_close(
        pauli_expectation(plus.as_ref(), &[(0, Z)]).unwrap().re,
        0.0,
        TOL,
    );

    // T|+⟩: ⟨X⟩ = cos(π/4).
    let mut c = Circuit::new(1);
    c.h(0).t(0);
    let state = run_dense(&c);
    assert_close(
        pauli_expectation(state.as_ref(), &[(0, X)]).unwrap().re,
        FRAC_1_SQRT_2,
        TOL,
    );

    // Y over ℝ cannot exist.
    let real_state = DenseState::<f64>::new(1).unwrap();
    assert!(matches!(
        pauli_expectation(&real_state, &[(0, Y)]),
        Err(Error::UnsupportedForAlgebra { .. })
    ));
    // But Z does.
    assert_close(pauli_expectation(&real_state, &[(0, Z)]).unwrap(), 1.0, TOL);
}

#[test]
fn exotic_algebra_measurement_can_fail_meaningfully() {
    // A split-complex state on the null cone has zero total Born weight:
    // measurement must refuse rather than divide by zero.
    let mut state = DenseState::<SplitComplex>::new(1).unwrap();
    let s = FRAC_1_SQRT_2;
    state
        .load(&[(0, SplitComplex::new(s, s))]) // born_weight = 0
        .unwrap();
    let err = state.measure(0, &mut Prng::new(0)).unwrap_err();
    assert!(matches!(err, Error::InvalidState(_)));
    // Sampling refuses for the same reason.
    let err = state.sample(16, &mut Prng::new(0)).unwrap_err();
    assert!(matches!(err, Error::InvalidState(_)));
    // A timelike state measures fine.
    state.load(&[(0, SplitComplex::new(1.0, 0.5))]).unwrap();
    let outcome = state.measure(0, &mut Prng::new(0)).unwrap();
    assert!(!outcome, "only |0⟩ is populated");
}

#[test]
fn measure_out_of_range_errors() {
    let mut state = DenseState::<C64>::new(2).unwrap();
    assert!(matches!(
        state.measure(2, &mut Prng::new(0)),
        Err(Error::QubitOutOfRange {
            qubit: 2,
            num_qubits: 2
        })
    ));
}

#[test]
fn width_limits_enforced() {
    // The structural bound is u64 indexing; below it, over-scale widths
    // are inhibited by the resource guard at REAL capacity: a 44-qubit
    // dense vector is 256 TiB, refused with the measured numbers rather
    // than a presumed width constant (see tests/capacity.rs for the
    // guard's own suite).
    assert!(matches!(
        DenseState::<C64>::new(64),
        Err(Error::TooManyQubits {
            requested: 64,
            max: 63
        })
    ));
    match DenseState::<C64>::new(44) {
        Err(Error::OutOfMemory {
            requested,
            available,
            ..
        }) => {
            assert_eq!(requested, 16 << 44);
            assert!(available < requested, "measured budget must be real");
        }
        other => panic!("44 dense qubits must be inhibited by measurement: {other:?}"),
    }
    assert!(SparseState::<C64>::new(63).is_ok());
    assert!(matches!(
        SparseState::<C64>::new(64),
        Err(Error::TooManyQubits { .. })
    ));
    // Width mismatch between circuit and backend.
    let reg: GateRegistry = GateRegistry::standard();
    let bound = library::bell::<C64>().bind(&reg).unwrap();
    let mut state = DenseState::<C64>::new(3).unwrap();
    assert!(matches!(
        bound.run(&mut state),
        Err(Error::WidthMismatch {
            circuit: 2,
            backend: 3
        })
    ));
}

#[test]
fn teleportation_with_mid_circuit_measurement() {
    // Teleport ψ = u(θ, φ, 0)|0⟩ from q0 to q2 with real measurements and
    // classically controlled corrections, over many seeds.
    let (theta, phi) = (1.234f64, 2.345f64);
    let alpha = c64((theta / 2.0).cos(), 0.0);
    let beta = c64(phi.cos(), phi.sin()) * (theta / 2.0).sin();
    let sim: Simulator = Simulator::new();
    let mut outcomes_seen = std::collections::HashSet::new();

    for seed in 0..24 {
        let mut c = Circuit::new(3);
        c.u(0, theta, phi, 0.0); // prepare ψ on q0
        c.h(1).cx(1, 2); // Bell pair on q1, q2
        c.cx(0, 1).h(0); // Bell measurement basis on q0, q1
        let mut state = sim.run(&c).unwrap();

        let mut rng = Prng::new(seed);
        let m0 = state.measure(0, &mut rng).unwrap();
        let m1 = state.measure(1, &mut rng).unwrap();
        if m1 {
            sim.apply(state.as_mut(), "x", &[], &[2]).unwrap();
        }
        if m0 {
            sim.apply(state.as_mut(), "z", &[], &[2]).unwrap();
        }
        outcomes_seen.insert((m0, m1));

        // q2 now carries ψ exactly; q0, q1 are collapsed to |m0⟩, |m1⟩.
        let base = (m0 as u64) | ((m1 as u64) << 1);
        let a0 = state.amplitude(base);
        let a1 = state.amplitude(base | 0b100);
        assert!(a0.approx_eq(alpha, TOL), "seed {seed}: {a0} vs {alpha}");
        assert!(a1.approx_eq(beta, TOL), "seed {seed}: {a1} vs {beta}");
    }
    assert!(
        outcomes_seen.len() >= 3,
        "expected varied Bell outcomes: {outcomes_seen:?}"
    );
}
