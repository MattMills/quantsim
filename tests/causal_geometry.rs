//! Causal geometry of qubit registers, measured across every backend.
//!
//! The causality between the elements of a register — which sites an
//! operation can influence, how far apart interacting qubits sit, and
//! in which direction the influence flows — is what bounds how much of
//! the system's full manifold a computation inhabits. This suite
//! measures that from every side the library has:
//!
//! * the **register metric**: causally connecting two sites costs
//!   exactly the coupling-graph distance (measured through the router,
//!   per geometry) — and registers whose fabric is *organized by causal
//!   scale* ([`Topology::hierarchical`], [`Topology::hypercube`])
//!   collapse that metric from linear to logarithmic, which is
//!   measurable as availability and as wall-clock;
//! * **representation pricing**: mobile internal geometry (device
//!   routing, MPS windows) converts causal range into *time*, fixed
//!   internal geometry (mera's tree) converts it into *memory* (rank at
//!   the crossed cut), clustering and Clifford-frame representations
//!   can be blind to it entirely;
//! * **causal diamonds**: operations outside the backward cone of an
//!   observation provably cannot move it — pruning to the diamond
//!   shrinks the computation while every observed marginal stays
//!   identical;
//! * **dual time**: a forward evolution from the preparation boundary
//!   and a backward evolution from the observation boundary resolve
//!   toward each other at a cut, each paying only its own cone's
//!   support — the balanced cut costs the square root of the
//!   one-directional support;
//! * none of it may move the physics — certified against the exact
//!   D[ω] ring wherever the workload family is Clifford.

mod common;

use common::assert_close;
use quantsim::exact::ExactState;
use quantsim::prelude::*;

const TOL: f64 = 1e-12;

/// Marginal probability of measuring 1 on `q`, from the nonzero support.
fn prob_one(state: &dyn Backend<C64>, q: usize) -> f64 {
    let mut p = 0.0;
    state.for_each_nonzero(&mut |idx, amp| {
        if (idx >> q) & 1 == 1 {
            p += amp.norm_sqr();
        }
    });
    p
}

fn device_on(topology: Topology, n: usize) -> DeviceState<C64> {
    DeviceState::with_latency(
        topology,
        LatencyMap::uniform(DurationModel::ibm_falcon_like()),
        ArityPolicy::default(),
        Box::new(SparseState::<C64>::new(n).unwrap()),
    )
    .unwrap()
}

/// The measured routing cost of `cx(a, b)` on a fresh device.
fn connection_cost(topology: &Topology, a: usize, b: usize) -> (usize, u64) {
    let n = topology.num_sites();
    let reg = GateRegistry::<C64>::standard();
    let mut c: Circuit = Circuit::new(n);
    c.h(a).cx(a, b);
    let mut dev = device_on(topology.clone(), n);
    c.bind(&reg).unwrap().run(&mut dev).unwrap();
    (dev.swap_count(), dev.elapsed())
}

#[test]
fn the_register_metric_prices_causal_connection() {
    // Connecting two sites costs (graph distance − 1) swaps — the
    // register's causal metric IS its coupling-graph metric, measured
    // through the router on every geometry, with the wall clock in
    // exact agreement.
    let base = DurationModel::ibm_falcon_like();
    let geometries = [
        Topology::linear(16),
        Topology::ring(16),
        Topology::grid(4, 4),
        Topology::heavy_hex_falcon27(),
        Topology::sycamore_like(3, 4),
        Topology::complete(16),
    ];
    for topology in &geometries {
        let n = topology.num_sites();
        for b in [1, n / 3, n - 1] {
            let dist = topology.shortest_path(0, b).expect("connected").len() - 1;
            let (swaps, elapsed) = connection_cost(topology, 0, b);
            assert_eq!(
                swaps,
                dist - 1,
                "router must pay the graph metric: dist({b}) = {dist}"
            );
            assert_eq!(
                elapsed,
                base.one_q + swaps as u64 * base.swap + base.two_q,
                "the clock is the metric in time units"
            );
        }
    }
    // Closed forms pin two of them: a ring halves the worst case.
    assert_eq!(connection_cost(&Topology::linear(16), 0, 15).0, 14);
    assert_eq!(connection_cost(&Topology::ring(16), 0, 15).0, 0);
    assert_eq!(connection_cost(&Topology::ring(16), 0, 8).0, 7);
    // All-to-all has a trivial metric: every connection is native.
    assert_eq!(connection_cost(&Topology::complete(16), 0, 15).0, 0);
}

#[test]
fn causal_range_is_priced_by_representation_geometry() {
    // ranged_pairs(12, d) holds gate count, arity profile and the output
    // state (n/2 Bell pairs) fixed while varying ONLY the interaction
    // range d. Each representation prices the range through its own
    // internal geometry.
    let n = 12;
    let reg = GateRegistry::<C64>::standard();
    let sim: Simulator = Simulator::new();

    let mut device_swaps = Vec::new();
    let mut mps_swaps = Vec::new();
    for d in [1, 2, 3, 6] {
        let circuit = library::ranged_pairs(n, d);
        let bound = circuit.bind(&reg).unwrap();
        let dense = sim.run(&circuit).unwrap();

        // The exact D[ω] reference: the family is Clifford, so every
        // backend below is checked against absolute values.
        let exact = ExactState::run(&circuit).unwrap();
        for idx in 0..(1u64 << n) {
            let e = exact.amplitude_c64(idx);
            let a = dense.amplitude(idx);
            assert!((a - e).norm() < TOL, "dense vs exact at {idx}");
        }

        // Sparse sees branching (2^{n/2} from the H's), never range.
        let sparse = sim.run_on("sparse", &circuit).unwrap();
        assert_eq!(sparse.nonzero_count(), 1 << (n / 2));

        // Factored sees the true causal clusters: n/2 two-qubit
        // factors at every range — range-blind clustering.
        let mut factored = FactoredState::<C64>::new(n).unwrap();
        bound.run(&mut factored).unwrap();
        assert_eq!(factored.factor_count(), n / 2);
        assert_eq!(factored.largest_factor_qubits(), 2);

        // MPS has *mobile* linear geometry: it drags operands adjacent
        // and entangles locally — bond stays 2 (the pair product), and
        // the range is paid in routing swaps instead, exactly like
        // hardware.
        let mut mps = MpsState::<C64>::new(n).unwrap();
        bound.run(&mut mps).unwrap();
        assert_eq!(
            mps.max_bond_dimension(),
            2,
            "mobile geometry: range costs time, not rank"
        );
        mps_swaps.push(mps.routing_swaps());
        assert!(max_amplitude_deviation(dense.as_ref(), &mps) < TOL);

        // The device is the same mobile story with a clock attached.
        let mut dev = device_on(Topology::linear(n), n);
        bound.run(&mut dev).unwrap();
        device_swaps.push(dev.swap_count());
        assert!(max_amplitude_deviation(dense.as_ref(), &dev) < TOL);

        // Interference: the pair cones never collide — CX is a
        // permutation and each H acts once on a fresh qubit, so
        // destroyed is exactly zero at every range.
        let mut intf = InterferenceState::new(n).unwrap();
        bound.run(&mut intf).unwrap();
        assert_eq!(intf.total_destroyed(), 0.0);

        // Clifford frames: the whole family is group structure — the
        // causal range is absorbed as metadata at stored support 1.
        let mut cf = CliffordFramedState::<C64>::new(n).unwrap();
        bound.run(&mut cf).unwrap();
        assert_eq!(cf.peak_stored_support(), 1);
        assert!(max_amplitude_deviation(dense.as_ref(), &cf) < TOL);
    }

    // Mobile geometries pay time monotonically in range: d = 1 is
    // native (zero swaps) and every widening costs more.
    assert_eq!(device_swaps[0], 0);
    assert_eq!(mps_swaps[0], 0);
    assert!(
        device_swaps.windows(2).all(|w| w[0] < w[1]),
        "{device_swaps:?}"
    );
    assert!(mps_swaps.windows(2).all(|w| w[0] < w[1]), "{mps_swaps:?}");
}

#[test]
fn fixed_tree_geometry_pays_causal_range_in_rank() {
    // mera's binary tree is FIXED: an interaction crossing a tree cut
    // pays rank at that cut. Leaf-aligned pairs (d = 1) are native;
    // root-crossing pairs (d = n/2) force rank 2^{n/2} at the root —
    // and a tree whose bond cap cannot afford the range truncates
    // *measurably*, never silently.
    let n = 12;
    let reg = GateRegistry::<C64>::standard();
    let sim: Simulator = Simulator::new();

    let local = library::ranged_pairs(n, 1);
    let crossing = library::ranged_pairs(n, 6);
    let dense_crossing = sim.run(&crossing).unwrap();

    let mut leaf = MeraState::<C64>::new(n).unwrap();
    local.bind(&reg).unwrap().run(&mut leaf).unwrap();
    assert!(leaf.is_exact());

    let mut root = MeraState::<C64>::new(n).unwrap();
    crossing.bind(&reg).unwrap().run(&mut root).unwrap();
    assert!(
        root.is_exact(),
        "2^6 = 64 fits the default bond cap exactly"
    );
    assert!(max_amplitude_deviation(dense_crossing.as_ref(), &root) < TOL);
    assert!(
        root.peak_block_elements() > leaf.peak_block_elements(),
        "root-crossing range must widen the materialized blocks: {} vs {}",
        root.peak_block_elements(),
        leaf.peak_block_elements()
    );
    assert!(
        root.memory_bytes() > leaf.memory_bytes(),
        "fixed geometry pays range in memory: {} vs {}",
        root.memory_bytes(),
        leaf.memory_bytes()
    );

    // Same circuit under a cap that cannot afford rank 64: the excess
    // Schmidt weight lands in the discarded ledger and is_exact drops.
    let mut starved = MeraState::<C64>::with_config(
        n,
        MeraConfig {
            max_bond: 32,
            ..MeraConfig::default()
        },
    )
    .unwrap();
    crossing.bind(&reg).unwrap().run(&mut starved).unwrap();
    assert!(!starved.is_exact());
    assert!(starved.discarded_weight() > 0.0);
}

#[test]
fn the_light_cone_is_measured_at_chip_scale() {
    // Brickwork spreads influence at most one site per layer. At width
    // 63 the cone is measured two ways at once: on the state side the
    // marginals outside the causal cone are EXACTLY zero (sparse
    // support 2^seeds in a 2^63 space), and on the device side the
    // schedule length is the causal depth, not the gate count.
    let n = 63;
    let depth = 10;
    let seeds = [30usize, 31, 32];
    let circuit = library::brickwork(n, depth, &seeds);
    let reg = GateRegistry::<C64>::standard();
    let bound = circuit.bind(&reg).unwrap();

    // Replay the gate list as pure causal reachability: control → target.
    let mut reachable = [false; 63];
    for &s in &seeds {
        reachable[s] = true;
    }
    for op in circuit.ops() {
        if let Op::Named { name, qubits, .. } = op {
            if name == "cx" && reachable[qubits[0]] {
                reachable[qubits[1]] = true;
            }
        }
    }

    let sim: Simulator = Simulator::new();
    let state = sim.run_on("sparse", &circuit).unwrap();
    assert_eq!(state.nonzero_count(), 1 << seeds.len());
    for (q, &in_cone) in reachable.iter().enumerate() {
        let p = prob_one(state.as_ref(), q);
        if in_cone {
            assert!(p > 0.0, "inside the cone, qubit {q} is disturbed");
            // The light cone is bounded by one site per layer.
            let dist = seeds.iter().map(|&s| s.abs_diff(q)).min().unwrap();
            assert!(dist <= depth, "influence outran the cone at {q}");
        } else {
            assert_eq!(p, 0.0, "outside the cone the marginal is exactly zero");
        }
    }

    // Device side, native geometry: layers are disjoint, so the wall
    // clock is EXACTLY seed layer + depth two-qubit slots, while the
    // serial total counts all 310 gates — the ratio is the measured
    // width of the causal structure.
    let base = DurationModel::ibm_falcon_like();
    let mut dev = device_on(Topology::linear(n), n);
    bound.run(&mut dev).unwrap();
    assert_eq!(dev.swap_count(), 0, "brickwork is native on a chain");
    assert_eq!(dev.elapsed(), base.one_q + depth as u64 * base.two_q);
    let cx_count = (depth * (n / 2)) as u64;
    assert_eq!(
        dev.serial_time(),
        seeds.len() as u64 * base.one_q + cx_count * base.two_q
    );

    // A geometry that does not contain the chain stretches the cone:
    // the same brickwork on the heavy-hex lattice must route.
    let n27 = 27;
    let c27 = library::brickwork(n27, 6, &[13]);
    let bound27 = c27.bind(&reg).unwrap();
    let mut chain = device_on(Topology::linear(n27), n27);
    bound27.run(&mut chain).unwrap();
    let mut hex = device_on(Topology::heavy_hex_falcon27(), n27);
    bound27.run(&mut hex).unwrap();
    assert_eq!(chain.swap_count(), 0);
    assert!(
        hex.swap_count() > 0,
        "heavy-hex does not contain the index chain"
    );
    assert!(hex.elapsed() > chain.elapsed());
    assert!(max_amplitude_deviation(&chain, &hex) < TOL);
}

#[test]
fn rainbow_prices_one_state_across_every_internal_geometry() {
    // The rainbow state: n/2 Bell pairs nested around the centre. One
    // causal structure, four prices — mera's fixed tree pays maximal
    // rank at the root, mobile MPS pays maximal routing, factored pays
    // nothing beyond the clusters, dense pays 2^n regardless.
    let n = 12;
    let circuit = library::rainbow(n);
    let reg = GateRegistry::<C64>::standard();
    let bound = circuit.bind(&reg).unwrap();
    let sim: Simulator = Simulator::new();
    let dense = sim.run(&circuit).unwrap();

    let exact = ExactState::run(&circuit).unwrap();
    for idx in 0..(1u64 << n) {
        assert!((dense.amplitude(idx) - exact.amplitude_c64(idx)).norm() < TOL);
    }

    let mut factored = FactoredState::<C64>::new(n).unwrap();
    bound.run(&mut factored).unwrap();
    assert_eq!(factored.factor_count(), n / 2);
    assert_eq!(factored.largest_factor_qubits(), 2);

    let mut mps = MpsState::<C64>::new(n).unwrap();
    bound.run(&mut mps).unwrap();
    assert_eq!(mps.max_bond_dimension(), 2);
    assert!(mps.routing_swaps() > 0);
    assert!(max_amplitude_deviation(dense.as_ref(), &mps) < TOL);

    let mut mera = MeraState::<C64>::new(n).unwrap();
    bound.run(&mut mera).unwrap();
    assert!(mera.is_exact());
    assert!(max_amplitude_deviation(dense.as_ref(), &mera) < TOL);

    // The ordering of memory is the ordering of geometry match:
    // clustering < mobile chain < fixed tree — and the fixed tree at
    // MAXIMAL causal range saturates to dense-scale cost (rank 2^{n/2}
    // at the root is the whole state), the honest price of a geometry
    // that cannot move.
    assert!(factored.memory_bytes() < mps.memory_bytes());
    assert!(mps.memory_bytes() < mera.memory_bytes());
    assert!(
        mera.memory_bytes() >= dense.memory_bytes() / 2,
        "maximal range saturates the fixed tree to dense scale: {} vs dense {}",
        mera.memory_bytes(),
        dense.memory_bytes()
    );

    // All-to-all hardware absorbs the nesting; a chain pays for it.
    let mut chain_dev = device_on(Topology::linear(n), n);
    bound.run(&mut chain_dev).unwrap();
    let mut ion_dev = device_on(Topology::complete(n), n);
    bound.run(&mut ion_dev).unwrap();
    assert!(chain_dev.swap_count() > 0);
    assert_eq!(ion_dev.swap_count(), 0);
    assert!(max_amplitude_deviation(dense.as_ref(), &chain_dev) < TOL);
}

#[test]
fn destruction_happens_only_where_a_cone_folds_back() {
    // Straight causal cones cannot interfere: H acts once per fresh
    // qubit and CX is a permutation, so ranged_pairs destroys exactly
    // zero. Fold the cones back through their own H's (run the inverse)
    // and each seed's H destroys exactly one unit of path weight — the
    // destruction ledger localizes the folds.
    let n = 12;
    let reg = GateRegistry::<C64>::standard();
    let circuit = library::ranged_pairs(n, 3);
    let bound = circuit.bind(&reg).unwrap();

    let mut straight = InterferenceState::new(n).unwrap();
    bound.run(&mut straight).unwrap();
    assert_eq!(straight.total_destroyed(), 0.0);
    assert!(straight.records().iter().all(|r| r.destroyed() == 0.0));

    let mut folded = InterferenceState::new(n).unwrap();
    bound.run(&mut folded).unwrap();
    bound.inverse().run(&mut folded).unwrap();
    // Each fold destroys the √-branch-volume at its cut: uncomputing
    // pair j (reverse order) merges 2^j branch pairs of magnitude
    // (1/√2)^{j+1}, destroying exactly (√2)^j — a geometric series in
    // the causal depth, Σ_{j<6} (√2)^j.
    let expected: f64 = (0..n / 2).map(|j| 2f64.powf(j as f64 / 2.0)).sum();
    assert_close(folded.total_destroyed(), expected, 1e-9);
    // And it is localized: exactly the n/2 inverse H's destroy, every
    // other record (all the CX permutations) is silent.
    let mut destroyed: Vec<f64> = folded
        .records()
        .iter()
        .filter(|r| r.destroyed() > 1e-9)
        .map(|r| {
            assert_eq!(r.qubits.len(), 1, "the folds are the 1q H's");
            r.destroyed()
        })
        .collect();
    assert_eq!(destroyed.len(), n / 2);
    destroyed.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (j, d) in destroyed.iter().enumerate() {
        assert_close(*d, 2f64.powf(j as f64 / 2.0), 1e-9);
    }
    // The folded state is |0…0⟩ again.
    assert_close(folded.probability(0), 1.0, 1e-9);
}

#[test]
fn clifford_causal_structure_is_metadata_until_magic_arrives() {
    // Width 50: rainbow + deep brickwork — causal structure that costs
    // every amplitude representation — absorbs into the Clifford frame
    // at stored support 1. The first T gate is what forces amplitudes,
    // and even then only within its own cone's basis.
    let n = 50;
    let reg = GateRegistry::<C64>::standard();
    let mut cf = CliffordFramedState::<C64>::new(n).unwrap();

    library::rainbow(n)
        .bind(&reg)
        .unwrap()
        .run(&mut cf)
        .unwrap();
    library::brickwork(n, 8, &[25])
        .bind(&reg)
        .unwrap()
        .run(&mut cf)
        .unwrap();
    assert_eq!(
        cf.peak_stored_support(),
        1,
        "all-Clifford causality is free"
    );
    assert!(cf.frame_gates() > 0);
    assert!(cf.stats().absorbed_clifford > 0);

    let mut with_t: Circuit = Circuit::new(n);
    with_t.t(25);
    with_t.bind(&reg).unwrap().run(&mut cf).unwrap();
    let support = cf.peak_stored_support();
    assert!(
        (2..=4).contains(&support),
        "one T scatters into at most a handful of stored states, got {support}"
    );
}

#[test]
fn causal_workloads_run_identically_on_every_backend_certified_exactly() {
    // The whole-library sweep at n = 8: every registered virtual
    // backend, the model backends, and the device over six geometries
    // run the causal family; every amplitude is checked against the
    // exact D[ω] ring (the family is Clifford), and a Ball run
    // certifies the float results by containment.
    let n = 8;
    let reg = GateRegistry::<C64>::standard();
    let sim: Simulator = Simulator::new();
    let circuits: Vec<(&str, Circuit)> = vec![
        ("ranged-2", library::ranged_pairs(n, 2)),
        ("rainbow", library::rainbow(n)),
        ("brickwork", library::brickwork(n, 4, &[3])),
    ];

    for (name, circuit) in &circuits {
        let exact = ExactState::run(circuit).unwrap();
        let bound = circuit.bind(&reg).unwrap();
        let check = |state: &dyn Backend<C64>, label: &str| {
            for idx in 0..(1u64 << n) {
                let dev = (state.amplitude(idx) - exact.amplitude_c64(idx)).norm();
                assert!(dev < TOL, "{label} vs exact on {name} at {idx}: {dev}");
            }
        };

        for backend in ["dense", "sparse", "adaptive", "factored", "mps", "mera"] {
            let state = sim.run_on(backend, circuit).unwrap();
            check(state.as_ref(), backend);
        }

        let mut intf = InterferenceState::new(n).unwrap();
        bound.run(&mut intf).unwrap();
        check(&intf, "interference");

        let mut framed = FramedState::new(Box::new(SparseState::<C64>::new(n).unwrap()));
        bound.run(&mut framed).unwrap();
        check(&framed, "framed");

        let mut cf = CliffordFramedState::<C64>::new(n).unwrap();
        bound.run(&mut cf).unwrap();
        check(&cf, "clifford-framed");

        for topology in [
            Topology::linear(n),
            Topology::ring(n),
            Topology::grid(2, 4),
            Topology::heavy_hex_falcon_corner(n).unwrap(),
            Topology::sycamore_like(2, 4),
            Topology::complete(n),
        ] {
            let label = format!(
                "device({} sites, {} edges)",
                topology.num_sites(),
                topology.num_edges()
            );
            let mut dev = device_on(topology, n);
            bound.run(&mut dev).unwrap();
            check(&dev, &label);
        }

        // Ball arithmetic: the certified intervals of a coarse-grained
        // run must CONTAIN the exact ring values.
        let ball_circuit: Circuit<Ball> = match *name {
            "ranged-2" => library::ranged_pairs(n, 2),
            "rainbow" => library::rainbow(n),
            _ => library::brickwork(n, 4, &[3]),
        };
        let ball_state = Simulator::<Ball>::new().run(&ball_circuit).unwrap();
        for idx in 0..(1u64 << n) {
            assert!(
                ball_state.amplitude(idx).contains(exact.amplitude_c64(idx)),
                "ball containment on {name} at {idx}"
            );
        }
    }
}

/// A depth-branching staircase: step `l` is `h(l); cx(l, l+1)`, so the
/// one-directional sparse support doubles per step — the workload whose
/// full manifold is `2^D` but whose dual-time halves are `2^{D/2}`.
fn staircase(n: usize, steps: usize) -> Circuit {
    assert!(steps < n);
    let mut c: Circuit = Circuit::new(n);
    for l in 0..steps {
        c.h(l).cx(l, l + 1);
    }
    c
}

#[test]
fn causally_organized_registers_collapse_the_metric() {
    // Registers whose coupling fabric is organized by causal scale —
    // the hierarchical skip register and the hypercube — have
    // logarithmic causal horizons where flat fabrics have linear or
    // polynomial ones. Diameter, ball growth (curvature signature) and
    // pair availability all measure it, and the router turns it into
    // wall-clock.
    let n = 32;
    let linear = Topology::linear(n);
    let ring = Topology::ring(n);
    let grid = Topology::grid(4, 8);
    let hier = Topology::hierarchical(n);
    let cube = Topology::hypercube(5);
    let all = Topology::complete(n);

    // Causal horizons.
    assert_eq!(linear.diameter(), 31);
    assert_eq!(ring.diameter(), 16);
    assert_eq!(grid.diameter(), 10);
    assert_eq!(cube.diameter(), 5, "hypercube horizon is exactly log2 n");
    assert!(
        hier.diameter() <= 10,
        "hierarchical horizon is logarithmic: {}",
        hier.diameter()
    );
    assert_eq!(all.diameter(), 1);
    assert!(hier.mean_distance() < ring.mean_distance());
    assert!(cube.mean_distance() < grid.mean_distance());

    // Curvature signature: ball volumes around a scale hub grow
    // exponentially on the causal fabrics, linearly on the chain.
    let flat = &linear.ball_sizes(16);
    let hyperbolic = &hier.ball_sizes(0);
    let homogeneous = &cube.ball_sizes(0);
    assert_eq!(flat[3], 7, "a chain ball is 2r + 1");
    assert!(hyperbolic[3] >= 3 * flat[3], "{hyperbolic:?}");
    assert_eq!(homogeneous[3], 1 + 5 + 10 + 10, "Σ C(5,k): exponential");

    // Availability: the fraction of pairs interactable within a swap
    // budget. The causal fabrics put most of the register within reach
    // where the chain has almost nothing.
    let budget = 2;
    let avail_linear = linear.pair_availability(budget);
    let avail_hier = hier.pair_availability(budget);
    let avail_cube = cube.pair_availability(budget);
    assert!(
        avail_hier > 2.0 * avail_linear,
        "{avail_hier} vs {avail_linear}"
    );
    assert!(
        avail_cube > 3.0 * avail_linear,
        "{avail_cube} vs {avail_linear}"
    );
    assert_eq!(all.pair_availability(0), 1.0);

    // And the router pays the metric: the worst-case single connection
    // costs the diameter, so the causal fabrics beat the chain by the
    // same logarithmic margin — in swaps and on the clock.
    let (swaps_linear, t_linear) = connection_cost(&linear, 0, 31);
    let (swaps_hier, t_hier) = connection_cost(&hier, 0, 31);
    let (swaps_cube, t_cube) = connection_cost(&cube, 0, 31);
    assert_eq!(swaps_linear, 30);
    assert!(swaps_hier <= 9, "{swaps_hier}");
    assert_eq!(swaps_cube, 4, "hypercube: 5 hops − 1");
    assert!(t_hier < t_linear / 4);
    assert!(t_cube < t_linear / 4);

    // A real algorithm inherits the collapse. The QFT couples every
    // pair; on the 16-site hypercube its coupling distance is Hamming
    // distance, and the measured schedule beats the chain.
    let reg = GateRegistry::<C64>::standard();
    let qft = library::qft(16).bind(&reg).unwrap();
    let mut on_chain = device_on(Topology::linear(16), 16);
    qft.run(&mut on_chain).unwrap();
    let mut on_cube = device_on(Topology::hypercube(4), 16);
    qft.run(&mut on_cube).unwrap();
    assert!(
        on_cube.swap_count() * 2 < on_chain.swap_count(),
        "hypercube must halve QFT routing: {} vs {}",
        on_cube.swap_count(),
        on_chain.swap_count()
    );
    assert!(on_cube.elapsed() < on_chain.elapsed());
    assert!(max_amplitude_deviation(&on_chain, &on_cube) < TOL);
}

#[test]
fn the_causal_diamond_bounds_what_must_be_computed() {
    // Operations outside the backward cone of an observation cannot
    // move it. Pruning to the diamond shrinks the circuit by the
    // geometry's factor while every observed marginal stays identical —
    // the measurable sense in which causal structure makes the manifold
    // computable.
    let n = 63;
    let depth = 8;
    let circuit = library::brickwork(n, depth, &[31]);
    let sim: Simulator = Simulator::new();

    // Observe one edge qubit: its backward cone is a corner of the
    // spacetime volume.
    let observed = [0usize];
    let (pruned, report) = causal_diamond(&circuit, &observed);
    assert_eq!(report.kept + report.dropped, circuit.len());
    assert!(
        report.kept * 4 < circuit.len(),
        "the diamond is a small corner: kept {} of {}",
        report.kept,
        circuit.len()
    );
    assert!(
        report.cone_qubits <= 2 + depth,
        "the cone widens at most one site per layer: {}",
        report.cone_qubits
    );

    // Observationally identical: the full joint distribution over the
    // observed set matches between full and pruned runs.
    let full = sim.run_on("sparse", &circuit).unwrap();
    let small = sim.run_on("sparse", &pruned).unwrap();
    for q in &observed {
        assert_close(
            prob_one(full.as_ref(), *q),
            prob_one(small.as_ref(), *q),
            1e-12,
        );
    }

    // Observing the seed's own site keeps a cone widening in BOTH
    // directions — about twice the edge corner's — because the diamond
    // volume tracks the observable's place in the causal order, not a
    // global property.
    let (_, central) = causal_diamond(&circuit, &[31]);
    assert!(
        central.kept > report.kept * 3 / 2,
        "central {} vs corner {}",
        central.kept,
        report.kept
    );

    // Observing everything keeps everything reachable backward from
    // the final surface.
    let all: Vec<usize> = (0..n).collect();
    let (_, everything) = causal_diamond(&circuit, &all);
    assert_eq!(everything.dropped, 0);
}

#[test]
fn dual_time_directions_resolve_toward_each_other() {
    // ⟨t|U|0⟩ two ways: one time direction carrying the whole manifold
    // to the observation boundary, or two opposed directions — forward
    // from preparation, backward from observation — resolving at a cut.
    // On a depth-branching workload the one-way support is 2^D; each
    // dual-time direction pays 2^{D/2} at the balanced cut. The
    // interface is the measured meeting surface.
    let n = 17;
    let steps = 16;
    let circuit = staircase(n, steps);
    let reg = GateRegistry::<C64>::standard();
    let sim: Simulator = Simulator::new();
    let dense = sim.run(&circuit).unwrap();

    // One-directional cost: the full 2^16 support at the boundary.
    let one_way = sim.run_on("sparse", &circuit).unwrap();
    assert_eq!(one_way.nonzero_count(), 1 << steps);

    for target in [0u64, 0b11, (1 << 5) | (1 << 4)] {
        let direct = dense.amplitude(target);
        // The resolution is cut-invariant: every meeting surface gives
        // the same amplitude.
        for cut in [0, 8, 16, 24, 32] {
            let res = dual_time_amplitude(&reg, &circuit, cut, target).unwrap();
            assert!(
                (res.amplitude - direct).norm() < TOL,
                "cut {cut}: {} vs {direct}",
                res.amplitude
            );
        }
    }

    // The balanced cut: each direction presents 2^{D/2}, quadratically
    // below the one-way manifold; the interface is at most the smaller
    // side.
    let mid = dual_time_amplitude(&reg, &circuit, steps, 0).unwrap();
    assert_eq!(mid.forward_gates, steps);
    assert_eq!(mid.backward_gates, steps);
    assert_eq!(mid.forward_support, 1 << (steps / 2));
    assert_eq!(mid.backward_support, 1 << (steps / 2));
    assert!(mid.interface_terms <= mid.forward_support.min(mid.backward_support));
    assert!(
        mid.peak_support() * mid.peak_support() <= (1 << steps),
        "the balanced resolution costs the square root of the manifold"
    );

    // Unbalanced cuts shift cost between the directions; the product
    // of supports stays the manifold size (nothing is free — the cut
    // chooses where it is paid).
    for cut in [8, 24] {
        let res = dual_time_amplitude(&reg, &circuit, cut, 0).unwrap();
        assert_eq!(
            res.forward_support * res.backward_support,
            1 << steps,
            "cut {cut}"
        );
        assert!(res.peak_support() > mid.peak_support());
    }

    // The same two opposed clocks on hardware: each half-schedule
    // presents half the serial wall, so the dual-time wall is max of
    // the halves — exactly half the one-directional schedule on the
    // strictly serial staircase.
    let base = DurationModel::ibm_falcon_like();
    let (front, back) = circuit.split_at(steps);
    let mut full_dev = device_on(Topology::linear(n), n);
    circuit.bind(&reg).unwrap().run(&mut full_dev).unwrap();
    let mut fwd_dev = device_on(Topology::linear(n), n);
    front.bind(&reg).unwrap().run(&mut fwd_dev).unwrap();
    let mut bwd_dev = device_on(Topology::linear(n), n);
    back.bind(&reg)
        .unwrap()
        .inverse()
        .run(&mut bwd_dev)
        .unwrap();
    let step_cost = base.one_q + base.two_q;
    assert_eq!(full_dev.elapsed(), steps as u64 * step_cost);
    assert_eq!(fwd_dev.elapsed(), (steps / 2) as u64 * step_cost);
    assert_eq!(bwd_dev.elapsed(), (steps / 2) as u64 * step_cost);
    assert_eq!(
        fwd_dev.elapsed().max(bwd_dev.elapsed()) * 2,
        full_dev.elapsed(),
        "opposed clocks meet halfway"
    );
}

#[test]
fn the_workload_harness_measures_the_causal_family() {
    // The causal workloads through the benchmarking harness, with the
    // model backends registered alongside the virtual ones: every run
    // is verified against dense in-run, and the memory ledger shows
    // which representation geometry matched the causal geometry.
    let n = 12;
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("interference", |n| Ok(Box::new(InterferenceState::new(n)?)))
        .unwrap();
    sim.backends_mut()
        .register("framed-sparse", |n| {
            Ok(Box::new(FramedState::new(Box::new(
                SparseState::<C64>::new(n)?,
            ))))
        })
        .unwrap();
    sim.backends_mut()
        .register("clifford-framed", |n| {
            Ok(Box::new(CliffordFramedState::<C64>::new(n)?))
        })
        .unwrap();
    sim.backends_mut()
        .register("device-falcon", |n| {
            Ok(Box::new(DeviceState::with_latency(
                Topology::heavy_hex_falcon_corner(n)?,
                LatencyMap::uniform(DurationModel::ibm_falcon_like()),
                ArityPolicy::default(),
                Box::new(SparseState::<C64>::new(n)?),
            )?))
        })
        .unwrap();

    let workloads = [
        Workload::from_circuit("ranged-1", library::ranged_pairs(n, 1)),
        Workload::from_circuit("ranged-6", library::ranged_pairs(n, 6)),
        Workload::from_circuit("rainbow", library::rainbow(n)),
        Workload::from_circuit(
            "brickwork",
            library::brickwork(n, 3, &(0..n).collect::<Vec<_>>()),
        ),
    ];
    let backends = [
        "dense",
        "sparse",
        "adaptive",
        "factored",
        "mps",
        "mera",
        "interference",
        "framed-sparse",
        "clifford-framed",
        "device-falcon",
    ];
    let report = compare_backends(&sim, &workloads, &backends, &BenchConfig::default()).unwrap();

    // Every (workload, backend) cell ran and verified against dense.
    assert_eq!(report.records.len(), workloads.len() * backends.len());
    assert!(report.max_deviation() < 1e-9, "harness-verified physics");

    // The memory ledger: clustering beats the chain beats dense on the
    // pair states; the Clifford frame beats everyone on Clifford work.
    for w in ["ranged-1", "ranged-6", "rainbow"] {
        let mem = |b: &str| report.record(w, b).unwrap().memory_bytes;
        assert!(mem("factored") < mem("dense"), "{w}");
        assert!(mem("sparse") < mem("dense"), "{w}");
        assert!(mem("clifford-framed") < mem("dense"), "{w}");
    }
}
