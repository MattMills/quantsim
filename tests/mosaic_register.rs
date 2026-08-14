//! The multi-representation register: conformance from singleton
//! regions, refusal-driven migration at structural era boundaries,
//! measured merge decisions, and the flagship — every portion of a
//! wide register in the representation ideal for its portion of the
//! circuit-problem axis, at the sum of the ideal costs.

mod common;

use common::{assert_close, sim};
use quantsim::backend::{BulkState, MosaicPolicy, MosaicState};
use quantsim::prelude::*;

/// Local expander-ish graph circuit (H wall + seeded long-range CZs).
fn graph_circuit(n: usize, edges: usize, seed: u64) -> Circuit<C64> {
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    let mut placed = 0;
    while placed < edges {
        let a = (rng.next_u64() % n as u64) as usize;
        let b = (rng.next_u64() % n as u64) as usize;
        if a != b {
            c.gate("cz", [], [a, b]);
            placed += 1;
        }
    }
    c
}

fn assert_matches_dense(mosaic: &MosaicState<C64>, dense: &dyn Backend<C64>, n: usize, what: &str) {
    let mut map = std::collections::HashMap::new();
    mosaic.for_each_nonzero(&mut |i, a| {
        map.insert(i, a);
    });
    let mut dev = 0.0f64;
    for i in 0..(1u64 << n) {
        let d = dense.amplitude(i);
        let m = map.get(&i).copied().unwrap_or(c64(0.0, 0.0));
        dev = dev.max((d - m).norm());
    }
    assert!(dev < 1e-9, "{what}: deviation {dev}");
}

/// Comparison up to one global phase — the mosaic's contract when a
/// phase-loose (bundle) region participates: a region's global phase
/// is global to the whole product, hence unobservable (module docs).
fn assert_matches_dense_up_to_phase(
    mosaic: &MosaicState<C64>,
    dense: &dyn Backend<C64>,
    n: usize,
    what: &str,
) {
    let mut map = std::collections::HashMap::new();
    let mut anchor = (0u64, 0.0f64);
    mosaic.for_each_nonzero(&mut |i, a| {
        if a.norm() > anchor.1 {
            anchor = (i, a.norm());
        }
        map.insert(i, a);
    });
    let d0 = dense.amplitude(anchor.0);
    assert!(d0.norm() > 1e-12, "{what}: anchor missing in reference");
    let phase = map[&anchor.0] / d0;
    assert_close(phase.norm(), 1.0, 1e-9);
    let mut dev = 0.0f64;
    for i in 0..(1u64 << n) {
        let d = dense.amplitude(i) * phase;
        let m = map.get(&i).copied().unwrap_or(c64(0.0, 0.0));
        dev = dev.max((d - m).norm());
    }
    assert!(dev < 1e-9, "{what}: deviation {dev} beyond a global phase");
}

#[test]
fn mosaic_conforms_over_the_full_registry() {
    let report = verify_backend(&sim(), "mosaic", &ConformanceConfig::default()).unwrap();
    assert!(report.passed(), "{report}");
    assert!(report.max_amplitude_deviation < 1e-9, "{report}");
    assert_eq!(report.sampling_mismatches, 0);
    assert_eq!(report.collapse_violations, 0);
}

#[test]
fn structure_emerges_from_gates_and_merges_are_measured() {
    // From singleton regions, gates sculpt the partition: local gates
    // stay local, entangling gates merge with the predicted-cost
    // comparison ledgered.
    let mut m = MosaicState::<C64>::new(6).unwrap();
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let cx = reg.resolve("cx").unwrap().matrix(&[]).unwrap();
    m.apply(&h, &[0]).unwrap();
    assert_eq!(m.layout().len(), 6, "single-qubit gates keep singletons");
    m.apply(&cx, &[0, 1]).unwrap();
    m.apply(&cx, &[3, 4]).unwrap();
    assert_eq!(m.layout().len(), 4, "two merges happened");
    let merges: Vec<_> = m.events().iter().filter(|e| e.kind == "merge").collect();
    assert_eq!(merges.len(), 2);
    assert!(
        merges[0].cause.contains("predicted sparse"),
        "the choice must carry its prediction: {}",
        merges[0].cause
    );
    // The state is still exactly right.
    let mut c = Circuit::new(6);
    c.h(0).cx(0, 1).cx(3, 4);
    let dense = sim().run(&c).unwrap();
    assert_matches_dense(&m, dense.as_ref(), 6, "sculpted mosaic");
    assert_close(m.total_abs_sqr(), 1.0, 1e-12);
}

#[test]
fn migration_is_refusal_driven_and_exact() {
    // A bundle region refuses its first T by name; the mosaic converts
    // the region down the policy list, retries, and records why. The
    // era boundary in the circuit becomes a representation change in
    // the register — with no loss of exactness.
    let n = 6;
    let graph = {
        let mut b = sim().backends().create("bundle", n).unwrap();
        let c = graph_circuit(n, 6, 3);
        c.bind(sim().registry()).unwrap().run(b.as_mut()).unwrap();
        b
    };
    let mut m =
        MosaicState::with_regions(n, vec![((0..n).collect(), graph)], MosaicPolicy::default())
            .unwrap();
    assert_eq!(m.layout()[0].1, "bundle");

    let reg = GateRegistry::<C64>::standard();
    let t = reg.resolve("t").unwrap().matrix(&[]).unwrap();
    m.apply(&t, &[2]).unwrap();
    assert_eq!(m.layout()[0].1, "sparse", "the region migrated");
    let migration = m
        .events()
        .iter()
        .find(|e| e.kind == "migrate")
        .expect("a migration event");
    assert_eq!(migration.from, "bundle");
    assert_eq!(migration.to, "sparse");
    assert!(migration.cause.contains("refused"), "{}", migration.cause);
    assert!(m.conversions_bytes() > 0);

    // Exactness across the migration, against dense.
    let mut c = graph_circuit(n, 6, 3);
    c.t(2);
    let dense = sim().run(&c).unwrap();
    assert_matches_dense(&m, dense.as_ref(), n, "post-migration state");
}

/// The era scenario: region A is an `na`-qubit Clifford expander graph
/// (bundle), region B (`nb` qubits, two halves) lives through a
/// Clifford → T → entangling era sequence. Returns the mosaic and the
/// full circuit for referees.
fn era_scenario(na: usize, nb: usize) -> (MosaicState<C64>, Circuit<C64>) {
    let n = na + nb;
    let reg = GateRegistry::<C64>::standard();
    // Region A: expander graph state, built in the bundle.
    let ga = graph_circuit(na, 3 * na / 2, 9);
    let mut a = sim().backends().create("bundle", na).unwrap();
    ga.bind(&reg).unwrap().run(a.as_mut()).unwrap();
    // Region B: two halves, each starting as a fresh bundle.
    let b1 = sim().backends().create("bundle", nb / 2).unwrap();
    let b2 = sim().backends().create("bundle", nb / 2).unwrap();
    let mut m = MosaicState::with_regions(
        n,
        vec![
            ((0..na).collect(), a),
            ((na..na + nb / 2).collect(), b1),
            ((na + nb / 2..n).collect(), b2),
        ],
        MosaicPolicy::default(),
    )
    .unwrap();
    // The full circuit, for the dense referee.
    let mut full: Circuit<C64> = Circuit::new(n);
    for op in ga.ops() {
        if let Op::Named {
            name,
            params,
            qubits,
        } = op
        {
            full.gate(name.clone(), params.clone(), qubits.clone());
        }
    }
    // B's story: Clifford prep, then the T era, then a cross-half
    // entangler.
    let reg2 = GateRegistry::<C64>::standard();
    let h = reg2.resolve("h").unwrap().matrix(&[]).unwrap();
    let cx = reg2.resolve("cx").unwrap().matrix(&[]).unwrap();
    let t = reg2.resolve("t").unwrap().matrix(&[]).unwrap();
    for (q0, width) in [(na, nb / 2), (na + nb / 2, nb / 2)] {
        m.apply(&h, &[q0]).unwrap();
        full.h(q0);
        for q in q0..q0 + width - 1 {
            m.apply(&cx, &[q, q + 1]).unwrap();
            full.cx(q, q + 1);
        }
    }
    m.apply(&t, &[na + 1]).unwrap();
    full.t(na + 1);
    m.apply(&t, &[na + nb / 2 + 1]).unwrap();
    full.t(na + nb / 2 + 1);
    m.apply(&cx, &[na + 1, na + nb / 2 + 1]).unwrap();
    full.cx(na + 1, na + nb / 2 + 1);
    (m, full)
}

#[test]
fn each_portion_lives_in_its_ideal_representation() {
    // The flagship, dense-verified at width 16 then held at width 40:
    // region A is a Clifford expander graph; region B lives through
    // its eras. See era_scenario.
    let run_scenario = era_scenario;

    // Verified regime: width 16 against dense, exact up to the one
    // global phase the bundle regions' reduced CX chains rotate
    // (e^{−iπ/4} each — the mosaic's documented phase contract).
    let (m, full) = run_scenario(8, 8);
    let dense = sim().run(&full).unwrap();
    assert_matches_dense_up_to_phase(&m, dense.as_ref(), 16, "width-16 scenario");
    let layout = m.layout();
    assert_eq!(layout[0].1, "bundle", "A never left the bundle");
    assert!(
        m.events().iter().any(|e| e.kind == "migrate")
            && m.events().iter().any(|e| e.kind == "merge"),
        "the ledger must show both era transitions: {:?}",
        m.events().len()
    );

    // Holding regime: width 40 — A is a 20-qubit expander no
    // bond/cluster/support representation affords, B carries magic the
    // bundle refuses, and the mosaic's price is the sum of parts.
    let (m, _) = run_scenario(20, 20);
    let layout = m.layout();
    assert_eq!(layout[0].1, "bundle");
    assert!(layout[0].0.len() == 20);
    let total = m.memory_bytes();
    assert!(
        total < 64 * 1024,
        "the mosaic must stay at the sum of ideal costs: {total} B"
    );
    assert_close(m.total_abs_sqr(), 1.0, 1e-9);
    // The same object under any single shipped lens, for contrast:
    // dense is 2^40 · 16 B = 16 TiB, sparse pays A's 2^20 support ×
    // B's, the bundle refuses B's T gates by name, and every
    // bond/cluster representation pays A's expander rank. (The n = 16
    // dense check above is what certifies the mosaic's answer.)
}

#[test]
fn merge_predictions_choose_dense_when_support_saturates() {
    // Two H-walled (full-support) regions coupled: the predicted
    // sparse cost exceeds dense and the merge says so.
    let mut m = MosaicState::<C64>::new(8).unwrap();
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let cx = reg.resolve("cx").unwrap().matrix(&[]).unwrap();
    for q in 0..8 {
        m.apply(&h, &[q]).unwrap();
    }
    for q in 0..3 {
        m.apply(&cx, &[q, q + 1]).unwrap();
        m.apply(&cx, &[q + 4, q + 5]).unwrap();
    }
    // Coupling the two full-support 4-wide regions: 16×16 = 256
    // sparse entries × ~40 B ≥ 2^8 × 16 B — dense wins the prediction.
    m.apply(&cx, &[3, 4]).unwrap();
    let last = m.events().last().unwrap();
    assert_eq!(last.kind, "merge");
    assert_eq!(last.to, "dense", "{}", last.cause);

    let mut c = Circuit::new(8);
    for q in 0..8 {
        c.h(q);
    }
    for q in 0..3 {
        c.cx(q, q + 1);
        c.cx(q + 4, q + 5);
    }
    c.cx(3, 4);
    let dense = sim().run(&c).unwrap();
    assert_matches_dense(&m, dense.as_ref(), 8, "dense-merged mosaic");
}

#[test]
fn partial_graphs_split_before_migrating() {
    // A 12-qubit bundle region whose graph has three components: the
    // first T fractures the region along its own structure — the two
    // untouched components stay graphs, only the hit component leaves.
    let n = 12;
    let reg = GateRegistry::<C64>::standard();
    let mut prep: Circuit<C64> = Circuit::new(n);
    for q in 0..n {
        prep.h(q);
    }
    for block in 0..3 {
        let b = block * 4;
        prep.gate("cz", [], [b, b + 1])
            .gate("cz", [], [b + 1, b + 2])
            .gate("cz", [], [b + 2, b + 3]);
    }
    let mut bundle = sim().backends().create("bundle", n).unwrap();
    prep.bind(&reg).unwrap().run(bundle.as_mut()).unwrap();
    let mut m =
        MosaicState::with_regions(n, vec![((0..n).collect(), bundle)], MosaicPolicy::default())
            .unwrap();
    assert_eq!(m.layout().len(), 1);

    let t = reg.resolve("t").unwrap().matrix(&[]).unwrap();
    m.apply(&t, &[5]).unwrap();

    let layout = m.layout();
    assert_eq!(layout.len(), 3, "three components after the fracture");
    let mut names: Vec<(Vec<usize>, String)> =
        layout.into_iter().map(|(q, name, _)| (q, name)).collect();
    names.sort_by_key(|(q, _)| q[0]);
    assert_eq!(names[0].1, "bundle", "component 0..4 untouched");
    assert_eq!(
        names[1].1, "sparse",
        "component 4..8 took the T and migrated"
    );
    assert_eq!(names[2].1, "bundle", "component 8..12 untouched");
    let kinds: Vec<&str> = m.events().iter().map(|e| e.kind).collect();
    assert_eq!(
        kinds,
        vec!["split", "migrate"],
        "split before any conversion"
    );

    // Exact against dense (up to the mosaic's phase contract).
    let mut full = prep;
    full.t(5);
    let dense = sim().run(&full).unwrap();
    assert_matches_dense_up_to_phase(&m, dense.as_ref(), n, "post-fracture state");
}

#[test]
fn a_graph_of_graphs_composes() {
    // A region can itself be a mosaic: the partition is recursive by
    // construction. The outer mosaic sees one "mosaic" tile; the inner
    // one keeps its own partition — and the whole thing stays exact.
    let reg = GateRegistry::<C64>::standard();
    let mut inner = MosaicState::<C64>::new(4).unwrap();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let cx = reg.resolve("cx").unwrap().matrix(&[]).unwrap();
    inner.apply(&h, &[0]).unwrap();
    inner.apply(&cx, &[0, 1]).unwrap();
    assert_eq!(inner.layout().len(), 3, "inner partition: (0,1), (2), (3)");

    let mut ghz2 = sim().backends().create("sparse", 2).unwrap();
    let bell: Circuit<C64> = library::bell();
    bell.bind(&reg).unwrap().run(ghz2.as_mut()).unwrap();

    let outer = MosaicState::with_regions(
        6,
        vec![
            ((0..4).collect(), Box::new(inner)),
            ((4..6).collect(), ghz2),
        ],
        MosaicPolicy::default(),
    )
    .unwrap();
    let names: Vec<String> = outer.layout().into_iter().map(|(_, n, _)| n).collect();
    assert_eq!(names, vec!["mosaic".to_string(), "sparse".to_string()]);

    let mut full: Circuit<C64> = Circuit::new(6);
    full.h(0).cx(0, 1).h(4).cx(4, 5);
    let dense = sim().run(&full).unwrap();
    assert_matches_dense(&outer, dense.as_ref(), 6, "mosaic of mosaics");
}

#[test]
fn heterogeneous_regions_compose_with_the_dynamic_register() {
    // A bulk region inside a mosaic: the dynamic hierarchy holds a
    // T-doped dyadic block while a sparse region holds a GHZ — the
    // representations that earlier increments built are tiles here.
    let n = 12;
    let reg = GateRegistry::<C64>::standard();
    let mut bulk_region = BulkState::<C64>::new(8).unwrap();
    let mut local: Circuit<C64> = Circuit::new(8);
    local.h(0).cx(0, 1).t(1).cx(1, 2).cx(2, 3).t(3);
    local.bind(&reg).unwrap().run(&mut bulk_region).unwrap();
    let mut sparse_region = sim().backends().create("sparse", 4).unwrap();
    let ghz4: Circuit<C64> = library::ghz(4);
    ghz4.bind(&reg)
        .unwrap()
        .run(sparse_region.as_mut())
        .unwrap();

    let m = MosaicState::with_regions(
        n,
        vec![
            ((0..8).collect(), Box::new(bulk_region)),
            ((8..12).collect(), sparse_region),
        ],
        MosaicPolicy::default(),
    )
    .unwrap();
    let mut full: Circuit<C64> = Circuit::new(n);
    full.h(0).cx(0, 1).t(1).cx(1, 2).cx(2, 3).t(3);
    full.h(8).cx(8, 9).cx(9, 10).cx(10, 11);
    let dense = sim().run(&full).unwrap();
    assert_matches_dense(&m, dense.as_ref(), n, "bulk⊗sparse mosaic");
    let names: Vec<String> = m.layout().into_iter().map(|(_, name, _)| name).collect();
    assert_eq!(names, vec!["bulk".to_string(), "sparse".to_string()]);
}

#[test]
fn the_mosaic_memory_law_is_the_sum_of_ideal_costs() {
    // The performance claim, as a measured law: sweeping the era
    // family's width, the mosaic's memory stays sub-exponential (the
    // expander tile is O(n + edges) in the bundle, the migrated T-half
    // support-bounded in sparse) while the best fixed lens that can
    // hold the family at all — sparse — is measured exponential on the
    // very same family (the expander half populates 2^na basis
    // states). One family, two laws, and a ×20+ measured ratio at the
    // shared width where both still run.
    let sizes = [8usize, 12, 16, 20]; // na; total width n = 2·na
    let mut mosaic_costs = Vec::new();
    for &na in &sizes {
        let (m, _) = era_scenario(na, na);
        assert_close(m.total_abs_sqr(), 1.0, 1e-9);
        mosaic_costs.push(m.memory_bytes());
    }
    let mosaic_law = classify_law(&sizes, &mosaic_costs);
    assert!(
        !matches!(mosaic_law, Law::Exponential { .. }),
        "mosaic memory must stay sub-exponential on its family: {mosaic_law:?} from {mosaic_costs:?}"
    );

    let sparse_sizes = [6usize, 8, 10];
    let mut sparse_costs = Vec::new();
    for &na in &sparse_sizes {
        let (_, full) = era_scenario(na, na);
        let state = sim().run_on("sparse", &full).unwrap();
        sparse_costs.push(state.memory_bytes());
    }
    let sparse_law = classify_law(&sparse_sizes, &sparse_costs);
    assert!(
        matches!(sparse_law, Law::Exponential { .. }),
        "sparse must pay the expander exponentially: {sparse_law:?} from {sparse_costs:?}"
    );

    // The measured ratio at n = 32, where sparse still runs.
    let (m, full) = era_scenario(16, 16);
    let sparse_fixed = sim().run_on("sparse", &full).unwrap();
    assert!(
        m.memory_bytes() * 20 < sparse_fixed.memory_bytes(),
        "mosaic {} B vs sparse-fixed {} B at width 32",
        m.memory_bytes(),
        sparse_fixed.memory_bytes()
    );
}

#[test]
fn static_structure_costs_no_events_and_stays_at_the_parts() {
    // The overhead of being a mosaic, measured: when the partition
    // already fits the circuit, dispatch performs no merges, no
    // migrations, no conversions — and the resting memory is the
    // parts' sum plus fixed bookkeeping.
    let n = 24;
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let cx = reg.resolve("cx").unwrap().matrix(&[]).unwrap();

    let regions: Vec<(Vec<usize>, Box<dyn Backend<C64>>)> = (0..3)
        .map(|b| {
            let qubits: Vec<usize> = (b * 8..(b + 1) * 8).collect();
            (qubits, sim().backends().create("sparse", 8).unwrap())
        })
        .collect();
    let mut m = MosaicState::with_regions(n, regions, MosaicPolicy::default()).unwrap();
    let mut parts_sum = 0usize;
    for b in 0..3usize {
        let base = b * 8;
        m.apply(&h, &[base]).unwrap();
        for q in base..base + 7 {
            m.apply(&cx, &[q, q + 1]).unwrap();
        }
        // The same block on a standalone sparse backend, for the sum.
        let mut fixed = sim().backends().create("sparse", 8).unwrap();
        let ghz8: Circuit<C64> = library::ghz(8);
        ghz8.bind(&reg).unwrap().run(fixed.as_mut()).unwrap();
        parts_sum += fixed.memory_bytes();
    }
    assert!(
        m.events().is_empty(),
        "static structure must cost no events"
    );
    assert_eq!(m.conversions_bytes(), 0);
    assert!(
        m.memory_bytes() <= parts_sum + 1024,
        "mosaic {} B must be the parts' {} B plus fixed bookkeeping",
        m.memory_bytes(),
        parts_sum
    );
    // And the joint state is exactly the product of block GHZs.
    let f = 1.0 / (2f64).sqrt();
    let all = (1u64 << 8) - 1;
    for pattern in 0..8u64 {
        let mut index = 0u64;
        for b in 0..3 {
            if (pattern >> b) & 1 == 1 {
                index |= all << (b * 8);
            }
        }
        let expect = f * f * f;
        let got = m.amplitude(index);
        assert!(
            (got.re - expect).abs() < 1e-9 && got.im.abs() < 1e-12,
            "block pattern {pattern:b}: {got}"
        );
    }
}
