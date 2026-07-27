//! Physical reproduction of real register geometries and their
//! operation-latency maps: the coupling presets are structurally right,
//! routing cost is a property of the geometry (measured, physics
//! invariant), latency maps move the clock and only the clock, and
//! parallelism is a measured ratio.

mod common;

use common::assert_close;
use quantsim::prelude::*;

#[test]
fn real_geometries_are_wired_correctly() {
    // IBM Falcon-r4 heavy-hex, 27 qubits: 28 couplers, degree ≤ 3,
    // connected, with the known lattice adjacencies.
    let falcon = Topology::heavy_hex_falcon27();
    assert_eq!(falcon.num_sites(), 27);
    assert_eq!(falcon.num_edges(), 28);
    assert_eq!(falcon.max_degree(), 3);
    assert!(falcon.is_connected());
    assert!(falcon.adjacent(1, 4) && falcon.adjacent(12, 15) && falcon.adjacent(25, 26));
    assert!(!falcon.adjacent(0, 2) && !falcon.adjacent(4, 5));

    // Sycamore-class diagonal lattice, 54 sites: degree ≤ 4, connected,
    // vertical (rows−1)·cols plus alternating diagonals.
    let sycamore = Topology::sycamore_like(6, 9);
    assert_eq!(sycamore.num_sites(), 54);
    assert_eq!(sycamore.num_edges(), 5 * 9 + 5 * 8);
    assert_eq!(sycamore.max_degree(), 4);
    assert!(sycamore.is_connected());

    // Trapped-ion all-to-all.
    let ion = Topology::complete(27);
    assert_eq!(ion.num_edges(), 27 * 26 / 2);
    assert_eq!(ion.max_degree(), 26);
    for a in 0..27 {
        for b in 0..27 {
            assert_eq!(ion.adjacent(a, b), a != b);
        }
    }

    // The Falcon corner sub-map stays connected (qubit selection).
    let corner = Topology::heavy_hex_falcon_corner(12).unwrap();
    assert_eq!(corner.num_sites(), 12);
    assert!(corner.is_connected());
    assert!(corner.max_degree() <= 3);
    assert!(Topology::heavy_hex_falcon_corner(28).is_err());
}

#[test]
fn routing_cost_is_geometry_physics_is_not() {
    // The same chip-scale GHZ chain over three real geometries, sparse
    // inner: swap counts differ with the coupling map; the amplitudes do
    // not. All-to-all must never swap.
    let n = 27;
    let ghz = library::ghz(n);
    let sim: Simulator = Simulator::new();
    let reference = sim.run_on("sparse", &ghz).unwrap();
    let reg = GateRegistry::<C64>::standard();
    let bound = ghz.bind(&reg).unwrap();

    let mut swaps = Vec::new();
    for topology in [
        Topology::linear(n),
        Topology::heavy_hex_falcon27(),
        Topology::complete(n),
    ] {
        let mut device = DeviceState::with_latency(
            topology,
            LatencyMap::uniform(DurationModel::ibm_falcon_like()),
            ArityPolicy::default(),
            Box::new(SparseState::<C64>::new(n).unwrap()),
        )
        .unwrap();
        bound.run(&mut device).unwrap();
        let dev = max_amplitude_deviation(reference.as_ref(), &device);
        assert!(dev < 1e-12, "geometry changed the physics: {dev}");
        swaps.push(device.swap_count());
    }
    assert_eq!(swaps[0], 0, "a chain-ordered GHZ is native on a chain");
    assert!(
        swaps[1] > 0,
        "the heavy-hex lattice must route a linear chain: {swaps:?}"
    );
    assert_eq!(swaps[2], 0, "all-to-all never swaps");
}

#[test]
fn latency_overrides_move_the_clock_and_only_the_clock() {
    let n = 4;
    let reg = GateRegistry::<C64>::standard();
    let mut c: Circuit = Circuit::new(n);
    c.h(0).cx(0, 1).cx(1, 2).cx(2, 3);
    let bound = c.bind(&reg).unwrap();

    let base = DurationModel::ibm_falcon_like();
    let run = |latency: LatencyMap| -> (u64, Vec<(u64, f64)>) {
        let mut device = DeviceState::<C64>::with_latency(
            Topology::linear(n),
            latency,
            ArityPolicy::default(),
            Box::new(DenseState::new(n).unwrap()),
        )
        .unwrap();
        bound.run(&mut device).unwrap();
        (device.elapsed(), device.probabilities())
    };

    let (uniform_elapsed, uniform_probs) = run(LatencyMap::uniform(base));
    // Slow one coupler on the critical path by exactly 1000 ticks.
    let mut slowed = LatencyMap::uniform(base);
    slowed.set_two_q(1, 2, base.two_q + 1000);
    let (slowed_elapsed, slowed_probs) = run(slowed);
    assert_eq!(
        slowed_elapsed,
        uniform_elapsed + 1000,
        "the slow edge sits on the critical path: the delta is exact"
    );
    assert_eq!(uniform_probs, slowed_probs, "latency never touches physics");

    // A per-site 1q override off the critical path must NOT move the end
    // (qubit 3's clock is dominated by the cx chain).
    let mut off_path = LatencyMap::uniform(base);
    off_path.set_one_q(3, base.one_q + 5);
    let (off_elapsed, _) = run(off_path);
    assert_eq!(off_elapsed, uniform_elapsed);
}

#[test]
fn parallelism_is_a_measured_ratio() {
    // Two disjoint couplers fire in parallel: elapsed is one gate, the
    // serial total is two.
    let base = DurationModel::ibm_falcon_like();
    let reg = GateRegistry::<C64>::standard();
    let mut c: Circuit = Circuit::new(4);
    c.cx(0, 1).cx(2, 3);
    let mut device = DeviceState::<C64>::with_latency(
        Topology::linear(4),
        LatencyMap::uniform(base),
        ArityPolicy::default(),
        Box::new(DenseState::new(4).unwrap()),
    )
    .unwrap();
    c.bind(&reg).unwrap().run(&mut device).unwrap();
    assert_eq!(device.elapsed(), base.two_q);
    assert_eq!(device.serial_time(), 2 * base.two_q);
}

#[test]
fn timing_models_rescale_identical_schedules() {
    // Same geometry, same circuit, three era clocks: the physical op
    // SEQUENCE is identical; only durations change. Ion clocks are
    // orders of magnitude above superconducting ones.
    let corner = Topology::heavy_hex_falcon_corner(8).unwrap();
    let reg = GateRegistry::<C64>::standard();
    let circuit = library::qft(8);
    let bound = circuit.bind(&reg).unwrap();

    let mut results = Vec::new();
    for model in [
        DurationModel::ibm_falcon_like(),
        DurationModel::sycamore_like(),
        DurationModel::ion_trap_like(),
    ] {
        let mut device = DeviceState::<C64>::with_latency(
            corner.clone(),
            LatencyMap::uniform(model),
            ArityPolicy::default(),
            Box::new(DenseState::new(8).unwrap()),
        )
        .unwrap();
        bound.run(&mut device).unwrap();
        let labels: Vec<(String, Vec<usize>)> = device
            .physical_log()
            .iter()
            .map(|op| (op.label.clone(), op.sites.clone()))
            .collect();
        results.push((device.elapsed(), device.swap_count(), labels));
    }
    assert_eq!(results[0].2, results[1].2, "op order is clock-independent");
    assert_eq!(results[0].2, results[2].2);
    assert_eq!(results[0].1, results[1].1);
    let (ibm, ion) = (results[0].0, results[2].0);
    assert!(ion > 100 * ibm, "ion clocks dominate: {ion} ns vs {ibm} ns");
}

#[test]
fn chip_scale_geometry_runs_on_a_sparse_inner() {
    // GHZ across all 54 Sycamore-class sites: a 2^54 dense vector is
    // impossible; the sparse inner holds 2 amplitudes while the router
    // walks the real diagonal lattice.
    let n = 54;
    let ghz = library::ghz(n);
    let reg = GateRegistry::<C64>::standard();
    let mut device = DeviceState::with_latency(
        Topology::sycamore_like(6, 9),
        LatencyMap::uniform(DurationModel::sycamore_like()),
        ArityPolicy::default(),
        Box::new(SparseState::<C64>::new(n).unwrap()),
    )
    .unwrap();
    ghz.bind(&reg).unwrap().run(&mut device).unwrap();
    assert_eq!(device.nonzero_count(), 2);
    assert_close(device.probability(0), 0.5, 1e-9);
    assert_close(device.probability((1 << n) - 1), 0.5, 1e-9);
    assert!(device.swap_count() > 0, "the chain crosses the lattice");
    assert!(device.memory_bytes() < 1 << 20, "sparse inner stays small");
}
