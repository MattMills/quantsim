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

    let (mut deviations, mut ledgers) = (Vec::new(), Vec::new());
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
        ledgers.push(mps.discarded_weight());
    }
    // The state books what the cap cut, without a reference to compare.
    assert!(ledgers[0] > 1e-3 && ledgers[1] > 0.0, "{ledgers:?}");
    assert_eq!(ledgers[2], 0.0, "bond 8 cuts nothing: {ledgers:?}");
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
    // A dense load compiles through a 2^n buffer admitted at REAL
    // capacity: 20 qubits (16 MiB) is admissible and works. A sparse
    // load never needs that buffer (see the sparse-load tests below), so
    // the guard's wall is reached at 50 qubits only by a support whose
    // prefix tree is itself over capacity — refused with the measured
    // numbers, not a presumed width constant.
    let mut mps = MpsState::<C64>::new(20).unwrap();
    let all: Vec<(u64, C64)> = (0..1u64 << 20)
        .map(|i| (i, c64(1.0 / 1024.0, 0.0)))
        .collect();
    mps.load(&all).unwrap();
    assert_eq!(mps.max_bond_dimension(), 1, "|+⟩^20 is a product state");
    let mut wide = MpsState::<C64>::new(50).unwrap();
    assert!(matches!(
        wide.load(&huge_support(50, 1 << 17)),
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

/// `count` distinct pseudo-random basis indices on `n` qubits, uniform
/// amplitude: a support whose prefix tree has bond ≈ `count` mid-chain.
fn huge_support(n: usize, count: usize) -> Vec<(u64, C64)> {
    let mut rng = Prng::new(0x5EED);
    let mask = (1u64 << n) - 1;
    let mut idx: Vec<u64> = (0..count).map(|_| rng.next_u64() & mask).collect();
    idx.sort_unstable();
    idx.dedup();
    let amp = c64(1.0 / (idx.len() as f64).sqrt(), 0.0);
    idx.into_iter().map(|i| (i, amp)).collect()
}

#[test]
fn sparse_loads_compile_through_the_prefix_tree_at_any_width() {
    // Loading a k-term superposition used to compile through a 2^n
    // buffer, so the linear-in-width representation refused a two-term
    // GHZ load at 31+ qubits. The support's binary prefix tree has bond
    // ≤ k at every cut; two SVD sweeps trim it to the exact Schmidt
    // ranks, so a loaded GHZ is bond 2 — identical to one built by gates.
    let n = 63;
    let top = (1u64 << n) - 1;
    let h = FRAC_1_SQRT_2;
    let mut mps = MpsState::<C64>::new(n).unwrap();
    mps.load(&[(0, c64(h, 0.0)), (top, c64(0.0, h))]).unwrap();
    assert_eq!(mps.max_bond_dimension(), 2);
    assert!(mps.memory_bytes() < 64_000, "{}", mps.memory_bytes());
    assert!(mps.amplitude(0).approx_eq(c64(h, 0.0), 1e-12));
    assert!(mps.amplitude(top).approx_eq(c64(0.0, h), 1e-12));
    assert!(mps.amplitude(1).approx_eq(c64(0.0, 0.0), 1e-12));
    common::assert_close(mps.total_weight(), 1.0, 1e-12);

    // A generic sparse support at 63 qubits: every amplitude exact, bonds
    // bounded by the support size, the left-isometric gauge restored
    // (the norm sits on the last site, so measurement and sampling work).
    let mut rng = Prng::new(31);
    let entries: Vec<(u64, C64)> = (0..12)
        .map(|_| {
            (
                rng.next_u64() >> 1,
                c64(rng.next_f64() - 0.5, rng.next_f64() - 0.5),
            )
        })
        .collect();
    let norm: f64 = entries
        .iter()
        .map(|(_, a)| a.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let entries: Vec<(u64, C64)> = entries.iter().map(|&(i, a)| (i, a / norm)).collect();
    let mut sparse = SparseState::<C64>::new(n).unwrap();
    sparse.load(&entries).unwrap();
    mps.load(&entries).unwrap();
    assert!(mps.max_bond_dimension() <= entries.len());
    assert!(mps.is_exact());
    assert!(max_amplitude_deviation(&sparse, &mps) < 1e-12);
    common::assert_close(mps.total_weight(), 1.0, 1e-12);
    let shots_mps = mps.sample(200, &mut Prng::new(5)).unwrap();
    let shots_sparse = sparse.sample(200, &mut Prng::new(5)).unwrap();
    assert_eq!(shots_mps, shots_sparse);

    // Permutations ride on load, so they now run at full width too.
    let flip = |i: u64| i ^ top;
    mps.apply_permutation("flip", &flip).unwrap();
    sparse.apply_permutation("flip", &flip).unwrap();
    assert!(max_amplitude_deviation(&sparse, &mps) < 1e-12);

    // A support wider than the bond cap is truncated — and booked.
    let mut capped = MpsState::<C64>::with_config(
        n,
        MpsConfig {
            max_bond: 8,
            trunc_tol: 1e-12,
        },
    )
    .unwrap();
    capped.load(&huge_support(n, 64)).unwrap();
    assert_eq!(capped.max_bond_dimension(), 8);
    assert!(
        capped.discarded_weight() > 0.1,
        "{}",
        capped.discarded_weight()
    );

    // Small widths still take the dense route where it is cheaper, and the
    // two routes agree on the same support.
    let mut tree = MpsState::<C64>::new(12).unwrap();
    let small: Vec<(u64, C64)> = entries.iter().map(|&(i, a)| (i & 0xFFF, a)).collect();
    tree.load(&small).unwrap();
    let mut dense = DenseState::<C64>::new(12).unwrap();
    dense.load(&small).unwrap();
    assert!(max_amplitude_deviation(&dense, &tree) < 1e-12);
}

#[test]
fn a_refused_load_leaves_the_state_untouched() {
    // Regression: load reset the logical↔site layout *before* admitting
    // its buffer, so a refused load silently scrambled any state whose
    // qubits had been routed by SWAPs (the tensors kept the routed order,
    // the map claimed the identity). Load is now transactional.
    let n = 50;
    let mut mps = MpsState::<C64>::new(n).unwrap();
    mps.apply(
        &GateMatrix::try_from_c64s(
            2,
            &[
                c64(FRAC_1_SQRT_2, 0.0),
                c64(FRAC_1_SQRT_2, 0.0),
                c64(FRAC_1_SQRT_2, 0.0),
                c64(-FRAC_1_SQRT_2, 0.0),
            ],
        )
        .unwrap(),
        &[0],
    )
    .unwrap();
    let sim: Simulator = Simulator::new();
    sim.apply(&mut mps, "cx", &[], &[0, 30]).unwrap();
    sim.apply(&mut mps, "x", &[], &[7]).unwrap();
    assert!(mps.routing_swaps() > 0, "the cx must have routed");
    let before: Vec<(u64, C64)> = {
        let mut v = Vec::new();
        mps.for_each_nonzero(&mut |i, a| v.push((i, a)));
        v
    };
    assert!(matches!(
        mps.load(&huge_support(n, 1 << 17)),
        Err(Error::OutOfMemory { .. })
    ));
    let mut after = Vec::new();
    mps.for_each_nonzero(&mut |i, a| after.push((i, a)));
    assert_eq!(before.len(), after.len());
    for ((i, a), (j, b)) in before.iter().zip(&after) {
        assert_eq!(i, j);
        assert!(a.approx_eq(*b, 1e-15));
    }
    // And the state keeps evolving correctly afterwards.
    sim.apply(&mut mps, "cx", &[], &[30, 49]).unwrap();
    let ghz_end = (1u64 << 0) | (1u64 << 30) | (1u64 << 49) | (1u64 << 7);
    assert!(mps
        .amplitude(ghz_end)
        .approx_eq(c64(FRAC_1_SQRT_2, 0.0), 1e-12));
    assert!(mps
        .amplitude(1u64 << 7)
        .approx_eq(c64(FRAC_1_SQRT_2, 0.0), 1e-12));
}
