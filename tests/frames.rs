//! Local frames: representation cost is not physics, and frames are the
//! dial that separates them. Conformance holds framed backends to reference
//! tolerance; the transverse-field tests measure what frames buy.

mod common;

use common::*;
use quantsim::prelude::*;
use std::f64::consts::FRAC_1_SQRT_2;

fn sim_with_frames() -> Simulator {
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("framed-sparse", |n| {
            Ok(Box::new(FramedState::new(Box::new(
                SparseState::<C64>::new(n)?,
            ))))
        })
        .unwrap();
    sim.backends_mut()
        .register("framed-dense", |n| {
            Ok(Box::new(FramedState::new(Box::new(
                DenseState::<C64>::new(n)?,
            ))))
        })
        .unwrap();
    sim.backends_mut()
        .register("framed-mps", |n| {
            Ok(Box::new(FramedState::new(Box::new(MpsState::<C64>::new(
                n,
            )?))))
        })
        .unwrap();
    sim
}

#[test]
fn framed_backends_conform_over_the_full_registry() {
    // Lazy absorption + conjugation + flush-on-observe must be invisible
    // to physics, over every registered gate and every inner representation.
    let sim = sim_with_frames();
    let cfg = ConformanceConfig {
        random_circuits: 10,
        orderings: 2,
        ..ConformanceConfig::default()
    };
    for backend in ["framed-sparse", "framed-dense", "framed-mps"] {
        let report = verify_backend(&sim, backend, &cfg).unwrap();
        assert!(report.passed(), "{report}");
        assert!(report.max_amplitude_deviation < 1e-9, "{report}");
        assert_eq!(report.sampling_mismatches, 0, "{backend}");
        assert_eq!(report.collapse_violations, 0, "{backend}");
    }
}

#[test]
fn single_qubit_gates_absorb_and_cancel() {
    let mut state = FramedState::new(Box::new(SparseState::<C64>::new(4).unwrap()));
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();

    // An H-layer never touches the inner state.
    for q in 0..4 {
        state.apply(&h, &[q]).unwrap();
    }
    assert_eq!(state.active_frames(), 4);
    assert_eq!(state.stored_nonzero_count(), 1, "inner still |0000⟩");
    assert_eq!(state.stats().absorbed_1q, 4);
    assert_eq!(state.stats().flushes, 0);

    // A second H-layer cancels frame-algebraically: H·H = I.
    for q in 0..4 {
        state.apply(&h, &[q]).unwrap();
    }
    assert_eq!(state.active_frames(), 0, "inverse pairs cancel in metadata");
    assert_eq!(state.stats().flushes, 0, "still zero amplitude work");
    assert!(state.amplitude(0).approx_eq(c64(1.0, 0.0), TOL));

    // With one live H, observation flushes exactly that frame.
    let mut state = FramedState::new(Box::new(SparseState::<C64>::new(2).unwrap()));
    state.apply(&h, &[0]).unwrap();
    assert!(state.amplitude(1).approx_eq(c64(FRAC_1_SQRT_2, 0.0), TOL));
    assert_eq!(state.stats().flushes, 1);
    assert_eq!(state.active_frames(), 0);
}

#[test]
fn transverse_field_bulk_stays_support_one_under_frames() {
    // h-layer; L × { rx on every qubit, rxx on the chain }; h-layer.
    // Physically this is an X-basis evolution; in the Z computational basis
    // the raw sparse representation blows up to 2^n support. Frames absorb
    // the h and rx layers (rx commutes with rxx, so conjugation stays
    // clean) and diagonalize every rxx into an rzz — the stored support
    // never leaves 1. Same physics, verified against dense at the end.
    let n = 14;
    let layers = 3;
    let reg = GateRegistry::<C64>::standard();
    let mut circuit: Circuit = Circuit::new(n);
    for q in 0..n {
        circuit.h(q);
    }
    for l in 0..layers {
        for q in 0..n {
            circuit.rx(q, 0.3 + 0.07 * (l * n + q) as f64);
        }
        for q in 0..n - 1 {
            circuit.rxx(q, q + 1, 0.5 + 0.05 * l as f64);
        }
    }
    for q in 0..n {
        circuit.h(q);
    }
    let bound = circuit.bind(&reg).unwrap();

    // Framed sparse: support 1 throughout the bulk.
    let mut framed = FramedState::new(Box::new(SparseState::<C64>::new(n).unwrap()));
    bound.run(&mut framed).unwrap();
    let stats = framed.stats();
    assert_eq!(
        stats.absorbed_1q,
        2 * n + layers * n,
        "h + rx layers absorb"
    );
    assert_eq!(
        stats.diagonalized,
        layers * (n - 1),
        "every rxx became an rzz"
    );
    assert_eq!(stats.conjugated, 0, "nothing needed a dense conjugate");
    assert_eq!(
        framed.stored_nonzero_count(),
        1,
        "stored support never grew (read pre-flush)"
    );
    let framed_peak = framed.peak_inner_memory();

    // Raw sparse: peak support 2^n while the bulk runs.
    let mut raw = SparseState::<C64>::new(n).unwrap();
    let mut raw_peak = raw.memory_bytes();
    for gate in bound.gates() {
        match &gate.kernel {
            GateKernel::Matrix(m) => raw.apply(m, &gate.qubits).unwrap(),
            GateKernel::Diagonal(d) => raw.apply_diagonal(d, &gate.qubits).unwrap(),
        }
        raw_peak = raw_peak.max(raw.memory_bytes());
    }
    assert!(
        raw_peak > (1usize << n) * 16,
        "raw sparse must have paid 2^n support: {raw_peak}"
    );
    assert!(
        framed_peak * 100 < raw_peak,
        "frames must beat raw peak by >100×: {framed_peak} vs {raw_peak}"
    );

    // Identical physics: compare against dense, amplitude for amplitude.
    let mut dense = DenseState::<C64>::new(n).unwrap();
    bound.run(&mut dense).unwrap();
    let deviation = max_amplitude_deviation(&dense, &framed);
    assert!(deviation < 1e-9, "framed deviates by {deviation}");
}

#[test]
fn adopt_frame_rewrites_representation_not_physics() {
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let n = 8;

    // Build |+…+⟩ the expensive way (flushed into the inner state).
    let mut state = FramedState::new(Box::new(SparseState::<C64>::new(n).unwrap()));
    for q in 0..n {
        state.apply(&h, &[q]).unwrap();
    }
    state.flush().unwrap();
    assert_eq!(state.stored_nonzero_count(), 1 << n, "fully dense support");

    // Adopt the H frame on every qubit: stored collapses to |0…0⟩.
    for q in 0..n {
        state.adopt_frame(q, &h).unwrap();
    }
    assert_eq!(state.stored_nonzero_count(), 1, "support 1 after adoption");
    assert_eq!(state.active_frames(), n);

    // Physical state is untouched: amplitudes read back as |+…+⟩.
    let expected = 1.0 / ((1u64 << n) as f64).sqrt();
    assert!(state.amplitude(0).approx_eq(c64(expected, 0.0), TOL));
    assert!(state
        .amplitude((1 << n) - 1)
        .approx_eq(c64(expected, 0.0), TOL));

    // release_frame is the inverse of adopt_frame.
    let mut state = FramedState::new(Box::new(SparseState::<C64>::new(2).unwrap()));
    state.apply(&h, &[0]).unwrap();
    state.release_frame(0).unwrap();
    assert_eq!(state.active_frames(), 0);
    assert!(state.amplitude(1).approx_eq(c64(FRAC_1_SQRT_2, 0.0), TOL));

    // Validation: non-unitary and out-of-range adoption rejected.
    let mut bad = GateMatrix::<C64>::identity(2).unwrap();
    bad.set(0, 0, c64(2.0, 0.0));
    let mut state = FramedState::new(Box::new(SparseState::<C64>::new(2).unwrap()));
    assert!(matches!(
        state.adopt_frame(0, &bad),
        Err(Error::NotUnitary { .. })
    ));
    let good = GateMatrix::<C64>::identity(2).unwrap();
    assert!(matches!(
        state.adopt_frame(5, &good),
        Err(Error::QubitOutOfRange { .. })
    ));
}

#[test]
fn measurement_flushes_only_its_qubit() {
    let sim = sim_with_frames();
    let mut c = Circuit::new(3);
    c.h(0).h(1).h(2);
    let mut state = sim.run_on("framed-sparse", &c).unwrap();
    // Three frames live; measuring q1 flushes exactly one.
    let framed = state.as_any().downcast_ref::<FramedState<C64>>().unwrap();
    assert_eq!(framed.active_frames(), 3);
    let _ = state.measure(1, &mut Prng::new(4)).unwrap();
    let framed = state.as_any().downcast_ref::<FramedState<C64>>().unwrap();
    assert_eq!(framed.active_frames(), 2);
    assert_eq!(framed.stats().flushes, 1);
    assert_close(state.total_weight(), 1.0, TOL);
}

#[test]
fn wide_gates_fall_back_by_flushing() {
    // A 7-qubit diagonal exceeds the conjugation cap: the wrapper flushes
    // the involved frames and passes through — correctness over cleverness.
    let sim = sim_with_frames();
    let n = 7;
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    c.diagonal(
        "flip",
        library::phase_flip(n, 3),
        (0..n).collect::<Vec<_>>(),
    );
    for q in 0..n {
        c.h(q);
    }
    let framed = sim.run_on("framed-sparse", &c).unwrap();
    // Read stats before any observation API flushes the trailing h-frames.
    let stats = framed
        .as_any()
        .downcast_ref::<FramedState<C64>>()
        .unwrap()
        .stats();
    assert_eq!(stats.flushes, n, "cap fallback flushed the frames once");
    let dense = sim.run(&c).unwrap();
    let deviation = max_amplitude_deviation(dense.as_ref(), framed.as_ref());
    assert!(deviation < 1e-9, "{deviation}");
}

#[test]
fn grover_still_exact_through_frames() {
    // Frames neither help nor harm Grover (H layers absorb, oracles hit the
    // wide-gate fallback): the closed form must survive untouched.
    let sim = sim_with_frames();
    let circuit = library::grover(3, 5, 2).unwrap();
    let state = sim.run_on("framed-sparse", &circuit).unwrap();
    assert_close(state.probability(5), 0.9453125, 1e-9);
}
