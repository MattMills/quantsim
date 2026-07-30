//! Split quaternions as an inclusion/exclusion pair, measured: the
//! coquaternion algebra itself, the full standard gate set it carries
//! (unlike every other non-division algebra here), the exact channel
//! decomposition a standard circuit produces, the boost that pumps
//! matched constructive and destructive weight at conserved net, the
//! exchange that is measurably not unitary, and the null cone where
//! measurement refuses.

#![allow(clippy::assertions_on_constants)] // algebra-flag pins are intentional

mod common;

use quantsim::conformance::{verify_backend, ConformanceConfig};
use quantsim::prelude::*;
use quantsim::qudit::{algebra_capacity, dual_algebra_report, embedded_action_is_component_linear};

type SQ = SplitQuaternion;

fn z(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

/// A circuit exercising real gates, phases and entanglers.
fn workload<S: Scalar>() -> Circuit<S> {
    let mut c: Circuit<S> = Circuit::new(3);
    c.h(0).cx(0, 1).t(1).ry(2, 0.7).cz(1, 2).h(2).s(0).cx(2, 0);
    c
}

// ── the algebra ──────────────────────────────────────────────────────

#[test]
fn the_algebra_is_the_coquaternions_with_an_indefinite_multiplicative_norm() {
    assert_eq!(SQ::DIM, 4);
    assert!(!SQ::COMMUTATIVE);
    assert!(SQ::ASSOCIATIVE);
    assert!(!SQ::DIVISION);
    assert_eq!(SQ::algebra_name(), "H_split");

    // The Born form is |inc|² − |exc|², indefinite, and multiplicative
    // (it is the determinant of the M₂(ℝ) picture).
    let x = SQ::pair(z(1.0, 2.0), z(3.0, 4.0));
    assert_eq!(x.born_weight(), -20.0);
    assert_eq!(x.abs_sqr(), 30.0);
    let y = SQ::pair(z(-2.0, 0.5), z(1.0, -3.0));
    common::assert_close(
        (x * y).born_weight(),
        x.born_weight() * y.born_weight(),
        1e-12,
    );
}

#[test]
fn the_grading_is_the_inclusion_exclusion_rule() {
    // Two exclusions make an inclusion; one of each makes an exclusion.
    let a = SQ::excluded(z(2.0, 0.0));
    let b = SQ::excluded(z(3.0, 0.0));
    assert_eq!((a * b).exclusion(), z(0.0, 0.0));
    assert_eq!((a * b).inclusion(), z(6.0, 0.0));

    let c = SQ::included(z(2.0, 0.0));
    assert_eq!((c * b).inclusion(), z(0.0, 0.0));
    assert_eq!((c * b).exclusion(), z(6.0, 0.0));
}

// ── the gate set ─────────────────────────────────────────────────────

#[test]
fn it_is_the_first_non_division_algebra_here_to_carry_the_full_gate_set() {
    // ℂ embeds (i is present), so nothing is missing.
    let full = GateRegistry::<C64>::standard().names();
    assert_eq!(GateRegistry::<SQ>::standard().names(), full);
    // Contrast: the split-complex numbers have no i and get the real
    // subset only.
    assert!(GateRegistry::<SplitComplex>::standard().names().len() < full.len());
}

#[test]
fn every_shipped_backend_that_can_hold_it_conforms_and_the_rest_refuse_by_reason() {
    let sim = Simulator::<SQ>::new();
    let cfg = ConformanceConfig::default();
    for backend in ["sparse", "adaptive", "factored"] {
        let report = verify_backend(&sim, backend, &cfg).unwrap();
        assert!(report.passed(), "{backend}: {:?}", report.failures);
        assert_eq!(report.algebra, "H_split");
        assert!(report.max_amplitude_deviation < 1e-12);
    }
    // The tensor-network backends need a commutative division algebra
    // and say so rather than producing wrong answers.
    for backend in ["mps", "mera"] {
        let text = match sim.backends().create(backend, 3) {
            Ok(_) => panic!("{backend} must refuse H_split"),
            Err(e) => format!("{e}"),
        };
        assert!(text.contains("commutative division algebra"), "{text}");
        assert!(text.contains("H_split"), "{text}");
    }
    match CliffordFramedState::<SQ>::new(3) {
        Ok(_) => panic!("the Clifford frame needs a commutative division algebra"),
        Err(e) => assert!(format!("{e}").contains("H_split"), "{e}"),
    }
}

#[test]
fn the_other_backends_agree_with_dense_over_the_new_algebra() {
    let bound = workload::<SQ>().bind(&GateRegistry::standard()).unwrap();
    let mut dense = DenseState::<SQ>::new(3).unwrap();
    bound.run(&mut dense).unwrap();

    let mut framed = FramedState::<SQ>::new(Box::new(DenseState::<SQ>::new(3).unwrap()));
    bound.run(&mut framed).unwrap();
    assert!(max_amplitude_deviation(&framed, &dense) < 1e-12);

    let mut interference = InterferenceState::<SQ>::new(3).unwrap();
    bound.run(&mut interference).unwrap();
    assert!(max_amplitude_deviation(&interference, &dense) < 1e-12);

    let mut device = DeviceState::with_latency(
        Topology::linear(3),
        LatencyMap::uniform(DurationModel::ibm_falcon_like()),
        ArityPolicy::default(),
        Box::new(SparseState::<SQ>::new(3).unwrap()),
    )
    .unwrap();
    bound.run(&mut device).unwrap();
    assert!(max_amplitude_deviation(&device, &dense) < 1e-12);

    // The interference ledger works over the new algebra and reproduces
    // the crate's pinned complex result exactly.
    let mut hh: Circuit<SQ> = Circuit::new(1);
    hh.h(0).h(0);
    let mut ledger = InterferenceState::<SQ>::new(1).unwrap();
    hh.bind(&GateRegistry::standard())
        .unwrap()
        .run(&mut ledger)
        .unwrap();
    common::assert_close(ledger.total_destroyed(), 1.0, 1e-12);
}

// ── the pair: two channels, one circuit ──────────────────────────────

#[test]
fn a_standard_circuit_drives_both_channels_and_never_mixes_them() {
    // One split-quaternion run carrying ψ in inclusion and φ in
    // exclusion is bit-for-bit two ordinary complex runs.
    let psi = [(0u64, z(0.6, 0.1)), (3, z(0.5, -0.4)), (5, z(0.4, 0.2))];
    let phi = [(1u64, z(0.3, 0.5)), (3, z(0.1, 0.0)), (6, z(-0.2, 0.1))];

    let mut paired: Vec<(u64, SQ)> = Vec::new();
    for i in 0..8u64 {
        let inc = psi.iter().find(|e| e.0 == i).map_or(z(0.0, 0.0), |e| e.1);
        let exc = phi.iter().find(|e| e.0 == i).map_or(z(0.0, 0.0), |e| e.1);
        if inc != z(0.0, 0.0) || exc != z(0.0, 0.0) {
            paired.push((i, SQ::pair(inc, exc)));
        }
    }

    let mut joint = DenseState::<SQ>::new(3).unwrap();
    joint.load(&paired).unwrap();
    workload::<SQ>()
        .bind(&GateRegistry::standard())
        .unwrap()
        .run(&mut joint)
        .unwrap();

    let run = |entries: &[(u64, C64)]| {
        let mut s = DenseState::<C64>::new(3).unwrap();
        s.load(entries).unwrap();
        workload::<C64>()
            .bind(&GateRegistry::standard())
            .unwrap()
            .run(&mut s)
            .unwrap();
        s
    };
    let a = run(&psi);
    let b = run(&phi);

    for i in 0..8u64 {
        let q = joint.amplitude(i);
        // Exact, not approximate: the same complex arithmetic ran twice.
        assert_eq!(q.inclusion(), a.amplitude(i));
        assert_eq!(q.exclusion(), b.amplitude(i));
        // And the Born rule reads the difference natively.
        common::assert_close(q.born_weight(), a.probability(i) - b.probability(i), 1e-15);
    }
    common::assert_close(
        joint.total_weight(),
        a.total_weight() - b.total_weight(),
        1e-12,
    );
    common::assert_close(
        joint.total_abs_sqr(),
        a.total_abs_sqr() + b.total_abs_sqr(),
        1e-12,
    );
}

#[test]
fn unitary_gates_conserve_the_net_weight_even_when_it_is_negative() {
    // A state with more exclusion than inclusion: net = 0.36 − 0.64.
    let mut state = DenseState::<SQ>::new(3).unwrap();
    state
        .load(&[
            (0, SQ::included(z(0.6, 0.0))),
            (1, SQ::excluded(z(0.8, 0.0))),
        ])
        .unwrap();
    let before = state.total_weight();
    common::assert_close(before, -0.28, 1e-15);

    // Every gate in the registry, applied in turn.
    let registry = GateRegistry::<SQ>::standard();
    let mut worst = 0.0f64;
    for name in registry.names() {
        let def = registry.resolve(&name).unwrap();
        if def.arity() > 3 {
            continue;
        }
        let params = vec![0.37; def.param_count()];
        let matrix = def.matrix(&params).unwrap();
        let targets: Vec<usize> = (0..def.arity()).collect();
        state.apply(&matrix, &targets).unwrap();
        worst = worst.max((state.total_weight() - before).abs());
    }
    assert!(worst < 1e-14, "net weight drifted by {worst}");
}

// ── moving weight between the ledgers ────────────────────────────────

#[test]
fn the_boost_is_unitary_and_pumps_path_weight_at_constant_net() {
    for t in [0.0, 0.25, 0.75, 1.5] {
        let u = SQ::boost(t);
        let mut m = GateMatrix::<SQ>::identity(2).unwrap();
        m.set(0, 0, u);
        m.set(1, 1, u);
        assert!(
            m.unitarity_deviation() < 1e-14,
            "t={t}: {}",
            m.unitarity_deviation()
        );

        let mut state = DenseState::<SQ>::new(2).unwrap();
        state.load(&[(0, SQ::included(z(1.0, 0.0)))]).unwrap();
        state.apply(&m, &[0]).unwrap();
        // Net conserved exactly...
        common::assert_close(state.total_weight(), 1.0, 1e-13);
        // ...while the path weight follows the closed form cosh(2t):
        // constructive and destructive amplitude created in matched
        // pairs, at zero net cost.
        common::assert_close(state.total_abs_sqr(), (2.0 * t).cosh(), 1e-12);
    }
    // Unit norm is not a small group: SL(2,ℝ) is non-compact, so the
    // path weight is unbounded above at fixed net weight.
    assert!(SQ::boost(4.0).path_weight() > 1000.0);
    common::assert_close(SQ::boost(4.0).born_weight(), 1.0, 1e-9);
}

#[test]
fn a_boost_that_mixes_basis_states_is_unitary_too() {
    // [[cosh, sinh·j], [sinh·j, cosh]] — the channel transfer entangled
    // with a basis rotation, still norm-preserving.
    let t: f64 = 0.9;
    let (ch, sh) = (
        SQ::included(z(t.cosh(), 0.0)),
        SQ::excluded(z(t.sinh(), 0.0)),
    );
    let mut m = GateMatrix::<SQ>::zeros(2).unwrap();
    m.set(0, 0, ch);
    m.set(0, 1, sh);
    m.set(1, 0, sh);
    m.set(1, 1, ch);
    assert!(m.unitarity_deviation() < 1e-14);

    let mut state = DenseState::<SQ>::new(1).unwrap();
    state.load(&[(0, SQ::included(z(1.0, 0.0)))]).unwrap();
    state.apply(&m, &[0]).unwrap();
    common::assert_close(state.total_weight(), 1.0, 1e-12);
    assert!(state.total_abs_sqr() > 1.5);
}

#[test]
fn the_exchange_is_measurably_not_unitary_and_negates_the_net() {
    let mut m = GateMatrix::<SQ>::identity(2).unwrap();
    m.set(0, 0, SQ::exchange());
    m.set(1, 1, SQ::exchange());
    // N(j) = −1, so M†M = −I: deviation exactly 2.
    common::assert_close(m.unitarity_deviation(), 2.0, 1e-15);

    // The registry refuses it, naming the deviation.
    let mut registry = GateRegistry::<SQ>::standard();
    let err = registry
        .register_fixed("jx", "inclusion/exclusion exchange", m.clone())
        .unwrap_err();
    assert!(format!("{err}").contains("not unitary"), "{err}");

    // Applied through the raw path it does exactly what it says: swap
    // the ledgers, flip the sign of the net, leave the path alone.
    let mut state = DenseState::<SQ>::new(1).unwrap();
    state
        .load(&[
            (0, SQ::included(z(0.8, 0.0))),
            (1, SQ::included(z(0.6, 0.0))),
        ])
        .unwrap();
    let (net, path) = (state.total_weight(), state.total_abs_sqr());
    state.apply(&m, &[0]).unwrap();
    common::assert_close(state.total_weight(), -net, 1e-14);
    common::assert_close(state.total_abs_sqr(), path, 1e-14);
    assert_eq!(state.amplitude(0).inclusion(), z(0.0, 0.0));
    assert_eq!(state.amplitude(0).exclusion(), z(0.8, 0.0));
}

// ── the null cone ────────────────────────────────────────────────────

#[test]
fn exact_cancellation_is_decidable_and_measurement_refuses_it() {
    // Balanced ledgers: net exactly 0 with path weight positive — the
    // representation can tell "nothing happened" from "everything
    // cancelled", which a complex amplitude cannot.
    let mut state = DenseState::<SQ>::new(1).unwrap();
    state
        .load(&[(0, SQ::pair(z(0.5, 0.0), z(0.5, 0.0)))])
        .unwrap();
    assert_eq!(state.total_weight(), 0.0);
    common::assert_close(state.total_abs_sqr(), 0.5, 1e-15);
    assert!(state.amplitude(0).is_null(1e-12));

    let mut rng = Prng::new(1);
    let err = state.measure(0, &mut rng).unwrap_err();
    assert!(format!("{err}").contains("not positive"), "{err}");

    // A net-negative state cannot be sampled either. Note what the
    // sampler actually does: it drops non-positive branches, so the
    // reported total is the surviving 0, not the state's −1.
    let mut all_exc = DenseState::<SQ>::new(1).unwrap();
    all_exc.load(&[(0, SQ::excluded(z(1.0, 0.0)))]).unwrap();
    common::assert_close(all_exc.total_weight(), -1.0, 1e-15);
    let err = all_exc.sample(4, &mut rng).unwrap_err();
    assert!(format!("{err}").contains("cannot sample"), "{err}");

    // A net-positive state samples normally: the exclusion channel just
    // reduces the weight of the states it cancels.
    let mut mixed = DenseState::<SQ>::new(1).unwrap();
    mixed
        .load(&[
            (0, SQ::pair(z(1.0, 0.0), z(0.6, 0.0))),
            (1, SQ::included(z(1.0, 0.0))),
        ])
        .unwrap();
    let counts = mixed.sample(4000, &mut Prng::new(9)).unwrap();
    let zeros = *counts.get(&0).unwrap_or(&0) as f64;
    let ones = *counts.get(&1).unwrap_or(&0) as f64;
    // Weights 1 − 0.36 = 0.64 against 1.0.
    common::assert_close(zeros / (zeros + ones), 0.64 / 1.64, 0.03);
}

// ── the hierarchical register ────────────────────────────────────────

#[test]
fn it_works_as_an_algebra_sector_qudit() {
    assert_eq!(algebra_capacity::<SQ>(), 1);
    // Embedded-ℂ multiplication acts component-wise on (inc, exc) — the
    // same property that makes standard gates channel-wise — so the
    // site sector runs natively.
    assert!(embedded_action_is_component_linear::<SQ>());

    let report = dual_algebra_report::<SQ>().unwrap();
    assert_eq!(report.sandwich_rank, 16);
    assert_eq!(report.operator_space, 16);
    assert!(report.max_linear_residual < 1e-12);

    let register =
        AlgebraicRegister::<SQ>::new(2, 1, Box::new(DenseState::<SQ>::new(2).unwrap())).unwrap();
    assert_eq!(register.logical_qubits(), 3);
    assert!(register.native_site_path());
}

// ── it benchmarks like any other algebra ─────────────────────────────

#[test]
fn the_harness_prices_it_next_to_every_other_backend() {
    let sim = Simulator::<SQ>::new();
    let report = compare_backends(
        &sim,
        &[Workload::new("ghz-10", || library::ghz(10))],
        &["dense", "sparse", "adaptive", "factored"],
        &BenchConfig::default(),
    )
    .unwrap();
    assert_eq!(report.records.len(), 4);
    for record in &report.records {
        assert!(
            record.deviation.map_or(true, |d| d < 1e-12),
            "{}: {:?}",
            record.backend,
            record.deviation
        );
    }
    let sparse = report
        .records
        .iter()
        .find(|r| r.backend == "sparse")
        .unwrap();
    let dense = report
        .records
        .iter()
        .find(|r| r.backend == "dense")
        .unwrap();
    // Structure still pays, at 4 reals an amplitude instead of 2.
    assert!(sparse.memory_bytes * 100 < dense.memory_bytes);
}
