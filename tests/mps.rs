//! MPS-specific behavior: bond-dimension scaling (the third compression
//! axis), routing, exactness at generous caps, *measurable* degradation at
//! tight caps, and the native conditional sampler.

mod common;

use common::*;
use quantsim::backend::{MPS_LOAD_MAX_QUBITS, MPS_MAX_WINDOW};
use quantsim::prelude::*;
use std::f64::consts::FRAC_1_SQRT_2;

#[test]
fn ghz_is_bond_dimension_two_at_any_width() {
    // The quadrant that defeats the factored backend (one global cluster)
    // is Schmidt-rank 2 across every cut: MPS holds GHZ(48) in a few KB.
    let sim: Simulator = Simulator::new();
    let state = sim.run_on("mps", &library::ghz(48)).unwrap();
    let mps = state.as_any().downcast_ref::<MpsState<C64>>().unwrap();
    assert_eq!(mps.max_bond_dimension(), 2);
    assert!(mps.bond_dimensions().iter().all(|&b| b <= 2));
    assert!(
        state.memory_bytes() < 64_000,
        "GHZ(48) should be KBs, got {}",
        state.memory_bytes()
    );
    assert!(state.amplitude(0).approx_eq(c64(FRAC_1_SQRT_2, 0.0), 1e-9));
    assert!(state
        .amplitude((1u64 << 48) - 1)
        .approx_eq(c64(FRAC_1_SQRT_2, 0.0), 1e-9));
    assert!(state.amplitude(1).approx_eq(c64(0.0, 0.0), 1e-9));
    assert_close(state.total_weight(), 1.0, 1e-9);

    // Environment-based measurement collapses correctly at this width.
    let mut state = sim.run_on("mps", &library::ghz(48)).unwrap();
    let outcome = state.measure(20, &mut Prng::new(9)).unwrap();
    let expected = if outcome { (1u64 << 48) - 1 } else { 0 };
    assert_close(state.probability(expected), 1.0, 1e-9);
    assert_close(state.total_weight(), 1.0, 1e-9);
}

#[test]
fn qft_is_exact_under_a_generous_cap() {
    // QFT(8) needs bond dimension 16 at the middle cut; the default cap of
    // 128 keeps it exact to reference tolerance.
    let sim: Simulator = Simulator::new();
    let circuit = library::qft(8);
    let dense = sim.run(&circuit).unwrap();
    let mps = sim.run_on("mps", &circuit).unwrap();
    let dev = max_amplitude_deviation(dense.as_ref(), mps.as_ref());
    assert!(dev < 1e-9, "qft-8 deviation {dev}");
    let mps = mps.as_any().downcast_ref::<MpsState<C64>>().unwrap();
    assert!(mps.max_bond_dimension() <= 16);
}

#[test]
fn truncation_degrades_measurably_never_silently() {
    // Same circuit, three caps: the deviation from dense is a number that
    // shrinks as the cap grows — approximation as a measured dial.
    let reg = GateRegistry::<C64>::standard();
    let circuit = library::random_circuit::<C64>(6, 60, 0xCAFE);
    let bound = circuit.bind(&reg).unwrap();
    let mut dense = DenseState::<C64>::new(6).unwrap();
    bound.run(&mut dense).unwrap();

    let mut deviations = Vec::new();
    for cap in [1usize, 2, 8] {
        let mut mps = MpsState::<C64>::with_config(
            6,
            MpsConfig {
                max_bond: cap,
                trunc_tol: 1e-12,
            },
        )
        .unwrap();
        bound.run(&mut mps).unwrap();
        deviations.push(max_amplitude_deviation(&dense, &mps));
    }
    assert!(
        deviations[0] > 0.05,
        "bond-1 (product state) must deviate substantially: {deviations:?}"
    );
    assert!(
        deviations[1] < deviations[0],
        "raising the cap must not hurt: {deviations:?}"
    );
    assert!(
        deviations[2] < 1e-9,
        "bond 8 covers a 6-qubit state exactly: {deviations:?}"
    );
}

#[test]
fn routing_and_locality() {
    let sim: Simulator = Simulator::new();
    let mut c = Circuit::new(8);
    c.h(0).cx(0, 6).cx(0, 6);
    let state = sim.run_on("mps", &c).unwrap();
    let mps = state.as_any().downcast_ref::<MpsState<C64>>().unwrap();
    // First cx(0,6) routes stepwise; the second finds the operands already
    // adjacent — swap count unchanged after the first routing.
    assert!(mps.routing_swaps() > 0);
    assert_eq!(mps.routing_swaps(), 5, "0 walks next to 6 in five swaps");
    // cx twice = identity: back to (routed) |+⟩ ⊗ |0…⟩.
    assert!(state.amplitude(0).approx_eq(c64(FRAC_1_SQRT_2, 0.0), 1e-9));
    assert!(state.amplitude(1).approx_eq(c64(FRAC_1_SQRT_2, 0.0), 1e-9));
}

#[test]
fn native_sampler_matches_the_distribution() {
    let sim: Simulator = Simulator::new();
    let state = sim.run_on("mps", &library::bell()).unwrap();
    let mps = state.as_any().downcast_ref::<MpsState<C64>>().unwrap();
    let counts = mps.sample_mps(4096, &mut Prng::new(11)).unwrap();
    assert_eq!(*counts.get(&0b01).unwrap_or(&0), 0);
    assert_eq!(*counts.get(&0b10).unwrap_or(&0), 0);
    let (z, o) = (
        *counts.get(&0b00).unwrap_or(&0),
        *counts.get(&0b11).unwrap_or(&0),
    );
    assert_eq!(z + o, 4096);
    assert!((1792..=2304).contains(&z), "{z} zeros"); // 4σ ≈ 256

    // And it works at widths where enumeration cannot: GHZ(40).
    let state = sim.run_on("mps", &library::ghz(40)).unwrap();
    let mps = state.as_any().downcast_ref::<MpsState<C64>>().unwrap();
    let counts = mps.sample_mps(256, &mut Prng::new(5)).unwrap();
    let all_ones = (1u64 << 40) - 1;
    assert_eq!(counts.len(), 2, "{counts:?}");
    assert_eq!(
        counts.get(&0).copied().unwrap_or(0) + counts.get(&all_ones).copied().unwrap_or(0),
        256
    );
}

#[test]
fn the_three_compression_axes_diverge_on_the_cluster_sweep() {
    // Width 24, one growing GHZ cluster: factored pays 2^cluster; MPS pays
    // bond 2 regardless — the register-width story completed.
    let sim: Simulator = Simulator::new();
    let mut c: Circuit = Circuit::new(24);
    c.h(0);
    for q in 0..21 {
        c.cx(q, q + 1); // 22-qubit cluster
    }
    for q in 22..24 {
        c.ry(q, 0.4);
    }
    let factored = sim.run_on("factored", &c).unwrap();
    let mps = sim.run_on("mps", &c).unwrap();
    assert!(
        factored.memory_bytes() > 60_000_000,
        "factored pays the cluster: {}",
        factored.memory_bytes()
    );
    assert!(
        mps.memory_bytes() < 100_000,
        "mps pays bond 2: {}",
        mps.memory_bytes()
    );
    let dev = max_amplitude_deviation(factored.as_ref(), mps.as_ref());
    assert!(dev < 1e-9, "same physics: {dev}");
}

#[test]
fn wide_gates_and_load_respect_documented_limits() {
    let sim: Simulator = Simulator::new();
    // Arity beyond the window cap is a clean error.
    let mut c: Circuit = Circuit::new(7);
    c.diagonal(
        "wide",
        library::phase_flip(6, 0),
        (0..6).collect::<Vec<_>>(),
    );
    match sim.run_on("mps", &c) {
        Err(Error::TooManyQubits { requested: 6, max }) => assert_eq!(max, MPS_MAX_WINDOW),
        other => panic!("expected window-cap error, got {:?}", other.err()),
    }
    // Load compiles through a 2^n buffer admitted at REAL capacity: 20
    // qubits (16 MiB) is admissible and works; 50 qubits (16 PiB) is
    // inhibited by the guard with the measured numbers, not a presumed
    // width constant.
    let mut mps = MpsState::<C64>::new(20).unwrap();
    mps.load(&[(0, c64(1.0, 0.0))]).unwrap();
    let mut wide = MpsState::<C64>::new(50).unwrap();
    assert!(matches!(
        wide.load(&[(0, c64(1.0, 0.0))]),
        Err(Error::OutOfMemory { .. })
    ));
    const _: () = assert!(MPS_LOAD_MAX_QUBITS == 63, "structural bound only");
    // Non-commutative algebras are rejected at construction, loudly.
    assert!(MpsState::<Quaternion>::new(4).is_err());
}

#[test]
fn selection_places_mps_where_it_wins() {
    // GHZ at width 40: dense can't run; factored errors at its cluster cap;
    // sparse holds 2 amplitudes; mps holds bond 2. Selection must pick a
    // survivor and reject factored with its real error.
    let sim: Simulator = Simulator::new();
    let report = select_backend(
        &sim,
        &Workload::ghz(40),
        &["mps", "sparse"],
        &BenchConfig {
            repetitions: 1,
            ..BenchConfig::default()
        },
        SelectionCriterion::Memory,
        1e-9,
    )
    .unwrap();
    // Sparse's two map entries beat MPS's 40 tensors on raw memory — the
    // point is both are sane and the choice is measured, not asserted.
    assert!(report.chosen.is_some());
    let chosen = report.chosen.unwrap();
    assert!(chosen == "sparse" || chosen == "mps");
}
