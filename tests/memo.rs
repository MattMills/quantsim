//! Progressive gate-result memoization: the plan must reproduce its
//! circuit exactly at every fusion width, the entry table must collapse
//! repeated geometry and only repeated geometry, and the journal must
//! unwind and rewind to states indistinguishable from a fresh run.

use quantsim::memo::{explore, Explorer, MemoConfig, MemoPlan};
use quantsim::prelude::*;

fn brickwork(n: usize, depth: usize) -> Circuit {
    let mut c = Circuit::new(n);
    for layer in 0..depth {
        for q in (layer % 2..n.saturating_sub(1)).step_by(2) {
            c.h(q);
            c.cx(q, q + 1);
            c.gate("rz", vec![0.7], vec![q + 1]);
        }
    }
    c
}

/// The circuit that caught the ordering bug, minimized by deletion from a
/// failing random circuit.
///
/// Emitting plan operations in each group's *opening* order is wrong: the
/// `{2}` group opens at gate 2, is still open when the `{1,7,8}` group
/// commits, then acquires qubit 1 at gate 8 — so its opening position
/// predates `cx[1,8]`, and sorting by it applied `cz[1,2]` before a gate
/// that must precede it. Deviation was 1.408. Commit order is the fix.
fn ordering_regression() -> Circuit {
    let mut c = Circuit::new(10);
    c.gate("t", vec![], vec![8]);
    c.h(7);
    c.gate("ry", vec![3.333_444_309_562_009], vec![2]);
    c.cx(7, 8);
    c.cx(1, 8);
    c.gate("t", vec![], vec![4]);
    c.x(1);
    c.gate("cp", vec![0.279_773_967_967_552_4], vec![4, 8]);
    c.cz(1, 2);
    c
}

fn deviation(sim: &Simulator<C64>, backend: &str, circuit: &Circuit, max_fuse: usize) -> f64 {
    let truth = sim.run_on(backend, circuit).expect("reference run");
    let cfg = MemoConfig {
        max_fuse,
        ..Default::default()
    };
    let plan = MemoPlan::from_circuit(circuit, sim.registry(), &cfg).expect("plan");
    let mut state = sim
        .backends()
        .create(backend, circuit.num_qubits())
        .expect("state");
    state.reset();
    plan.run(state.as_mut()).expect("run");
    max_amplitude_deviation(truth.as_ref(), state.as_ref())
}

#[test]
fn every_plan_reproduces_its_circuit_exactly_at_every_fusion_width() {
    let sim = Simulator::<C64>::new();
    let cases: Vec<(&str, Circuit)> = vec![
        ("ghz", library::ghz(10)),
        ("qft", library::qft(7)),
        ("brickwork", brickwork(9, 5)),
        ("rainbow", library::rainbow(8)),
        ("iqp", library::iqp(8, 12, true, 0x11)),
        ("random-a", library::random_circuit(9, 90, 0xABC)),
        ("random-b", library::random_circuit(9, 90, 0xDEF)),
        ("random-c", library::random_circuit(7, 150, 0x1234)),
        ("doped", library::doped_clifford(8, 40, 6, 0x99)),
    ];
    for (name, circuit) in &cases {
        for max_fuse in 1..=5 {
            let dev = deviation(&sim, "dense", circuit, max_fuse);
            assert!(
                dev < 1e-12,
                "{name} at max_fuse={max_fuse}: deviation {dev:.3e} — the plan must be \
                 the same unitary as the circuit, not an approximation of it"
            );
        }
    }
}

#[test]
fn the_commit_order_regression_stays_fixed() {
    let sim = Simulator::<C64>::new();
    let circuit = ordering_regression();
    for max_fuse in 1..=6 {
        let dev = deviation(&sim, "dense", &circuit, max_fuse);
        assert!(
            dev < 1e-12,
            "max_fuse={max_fuse}: deviation {dev:.3e}; opening-order emission gave 1.408 here"
        );
    }
    // The shape that made it possible: a group that outlives another
    // group's whole lifetime and then takes over one of its qubits.
    let plan =
        MemoPlan::from_circuit(&circuit, sim.registry(), &MemoConfig::default()).expect("plan");
    let mut seen_before: Vec<usize> = Vec::new();
    for op in plan.ops() {
        seen_before.extend(op.support.iter().copied());
    }
    assert!(
        seen_before.len() > plan.ops().len(),
        "at least one qubit is touched by more than one operation, which is the \
         precondition for the ordering to matter at all"
    );
}

#[test]
fn the_same_geometry_anywhere_shares_one_entry() {
    let sim = Simulator::<C64>::new();
    // Four disjoint copies of the same two-qubit block.
    let mut circuit = Circuit::new(8);
    for pair in 0..4 {
        circuit.h(2 * pair);
        circuit.cx(2 * pair, 2 * pair + 1);
    }
    let plan =
        MemoPlan::from_circuit(&circuit, sim.registry(), &MemoConfig::default()).expect("plan");
    let stats = plan.stats();
    assert_eq!(stats.gates_in, 8);
    assert_eq!(stats.ops_out, 4, "one fused operation per pair");
    assert_eq!(
        stats.entries, 1,
        "and all four are the same operator, so one entry holds them"
    );
    assert_eq!(stats.hits, 3);
    assert_eq!(stats.misses, 1);
    assert!((stats.reuse() - 4.0).abs() < 1e-12);
    // Every operation points at the same entry.
    assert!(plan.ops().iter().all(|op| op.entry == 0));
    // Absolute position is not part of the identity.
    assert_eq!(plan.ops()[0].support, vec![0, 1]);
    assert_eq!(plan.ops()[3].support, vec![6, 7]);
}

#[test]
fn qubit_order_within_a_support_is_part_of_the_identity() {
    let sim = Simulator::<C64>::new();
    // cx(0,1) and cx(3,2) differ: the control sits at a different
    // position within the ascending support, so the matrices differ and
    // the entries must too.
    let mut circuit = Circuit::new(4);
    circuit.cx(0, 1);
    circuit.cx(3, 2);
    let plan =
        MemoPlan::from_circuit(&circuit, sim.registry(), &MemoConfig::default()).expect("plan");
    assert_eq!(plan.stats().ops_out, 2);
    assert_eq!(
        plan.stats().entries,
        2,
        "content addressing must not conflate a flipped control with an offset"
    );

    // Whereas the same orientation at a different offset does share.
    let mut same = Circuit::new(4);
    same.cx(0, 1);
    same.cx(2, 3);
    let plan = MemoPlan::from_circuit(&same, sim.registry(), &MemoConfig::default()).expect("plan");
    assert_eq!(plan.stats().entries, 1);
}

#[test]
fn fusion_trades_operations_for_entry_bytes_monotonically() {
    let sim = Simulator::<C64>::new();
    let circuit = brickwork(10, 6);
    let mut previous: Option<(usize, usize, usize)> = None;
    for max_fuse in 1..=6 {
        let cfg = MemoConfig {
            max_fuse,
            ..Default::default()
        };
        let plan = MemoPlan::from_circuit(&circuit, sim.registry(), &cfg).expect("plan");
        let s = plan.stats();
        // A *fused* support never exceeds max_fuse, but a single gate
        // wider than the limit keeps its own support — the brickwork's
        // widest gate is a two-qubit cx.
        assert!(
            s.widest_support <= max_fuse.max(2),
            "max_fuse={max_fuse} produced a support of {}",
            s.widest_support
        );
        if let Some((ops, bytes, widest)) = previous {
            assert!(
                s.ops_out <= ops,
                "raising max_fuse must not add operations: {ops} -> {}",
                s.ops_out
            );
            assert!(
                s.entry_bytes >= bytes,
                "the operations it removes are paid for in entry bytes: {bytes} -> {}",
                s.entry_bytes
            );
            assert!(s.widest_support >= widest);
        }
        previous = Some((s.ops_out, s.entry_bytes, s.widest_support));
    }
    // The dial has real range on this family.
    let narrow = MemoPlan::from_circuit(
        &circuit,
        sim.registry(),
        &MemoConfig {
            max_fuse: 1,
            ..Default::default()
        },
    )
    .expect("plan");
    let wide = MemoPlan::from_circuit(
        &circuit,
        sim.registry(),
        &MemoConfig {
            max_fuse: 6,
            ..Default::default()
        },
    )
    .expect("plan");
    assert!(
        wide.stats().ops_out * 4 < narrow.stats().ops_out,
        "fusing to six qubits should collapse the brickwork severalfold: {} vs {}",
        wide.stats().ops_out,
        narrow.stats().ops_out
    );
}

#[test]
fn the_plan_accounting_adds_up() {
    let sim = Simulator::<C64>::new();
    for circuit in [
        library::random_circuit::<C64>(8, 100, 0x5EED),
        brickwork(8, 4),
        library::qft(6),
    ] {
        let plan =
            MemoPlan::from_circuit(&circuit, sim.registry(), &MemoConfig::default()).expect("plan");
        let s = plan.stats();
        assert_eq!(
            s.hits + s.misses,
            s.ops_out,
            "every operation is one or the other"
        );
        assert_eq!(s.entries, s.misses, "an entry is created exactly on a miss");
        assert_eq!(s.ops_out, plan.ops().len());
        assert_eq!(s.entries, plan.entries().len());
        let folded: usize = plan.ops().iter().map(|op| op.fused_from).sum();
        assert_eq!(
            folded, s.gates_in,
            "every source gate must be folded into exactly one operation"
        );
        assert!(plan.ops().iter().all(|op| op.entry < s.entries));
    }
}

#[test]
fn a_gate_wider_than_the_fusion_limit_is_memoized_but_never_fused() {
    let sim = Simulator::<C64>::new();
    let mut circuit = Circuit::new(6);
    // Two three-qubit gates with a fusion limit of two.
    circuit.gate("ccx", vec![], vec![0, 1, 2]);
    circuit.gate("ccx", vec![], vec![3, 4, 5]);
    let cfg = MemoConfig {
        max_fuse: 2,
        ..Default::default()
    };
    let plan = MemoPlan::from_circuit(&circuit, sim.registry(), &cfg).expect("plan");
    let s = plan.stats();
    assert_eq!(s.ops_out, 2);
    assert_eq!(s.widest_support, 3, "the gate keeps its own support");
    assert_eq!(
        s.entries, 1,
        "and is still memoized: the two are the same operator"
    );
    assert!(deviation(&sim, "dense", &circuit, 2) < 1e-12);
}

#[test]
fn a_lone_diagonal_stays_diagonal_and_fusing_promotes_it() {
    let sim = Simulator::<C64>::new();
    let mut circuit = Circuit::new(3);
    circuit.diagonal(
        "oracle",
        vec![
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(-1.0, 0.0),
        ],
        vec![0, 1],
    );
    let plan =
        MemoPlan::from_circuit(&circuit, sim.registry(), &MemoConfig::default()).expect("plan");
    assert!(
        matches!(plan.entries()[0], GateKernel::Diagonal(_)),
        "a lone diagonal must not be promoted to a dense matrix — the backends \
         apply it in O(support)"
    );

    let off = MemoConfig {
        preserve_diagonal: false,
        ..Default::default()
    };
    let plan = MemoPlan::from_circuit(&circuit, sim.registry(), &off).expect("plan");
    assert!(matches!(plan.entries()[0], GateKernel::Matrix(_)));
    assert!(deviation(&sim, "dense", &circuit, 4) < 1e-12);

    // A diagonal that gets something fused onto it becomes a matrix, and
    // stays correct.
    let mut fused = circuit.clone();
    fused.h(0);
    let plan =
        MemoPlan::from_circuit(&fused, sim.registry(), &MemoConfig::default()).expect("plan");
    assert!(matches!(plan.entries()[0], GateKernel::Matrix(_)));
    assert!(deviation(&sim, "dense", &fused, 4) < 1e-12);
}

#[test]
fn the_explorer_rewinds_to_states_indistinguishable_from_a_fresh_run() {
    let sim = Simulator::<C64>::new();
    let circuit = brickwork(8, 5);
    let plan =
        MemoPlan::from_circuit(&circuit, sim.registry(), &MemoConfig::default()).expect("plan");
    let total = plan.ops().len();
    assert!(total >= 6, "the sweep needs somewhere to rewind to");

    // Ground truth at every step: run the plan's first k operations fresh.
    let truth: Vec<Vec<(u64, C64)>> = (0..=total)
        .map(|k| {
            let mut state = sim
                .backends()
                .create("dense", plan.num_qubits())
                .expect("state");
            state.reset();
            plan.run_range(state.as_mut(), 0, k).expect("run");
            let mut stored = Vec::new();
            state.for_each_nonzero(&mut |i, a| stored.push((i, a)));
            stored.sort_by_key(|&(i, _)| i);
            stored
        })
        .collect();

    let mut explorer = Explorer::new(&sim, "dense", &plan, 3).expect("explorer");
    explorer.advance().expect("advance");
    assert_eq!(explorer.cursor(), total);
    assert!(explorer.checkpoints() >= 2);

    // Rewind to every step, descending, and check the state exactly.
    for target in (0..=total).rev() {
        explorer.rewind(target).expect("rewind");
        assert_eq!(explorer.cursor(), target);
        let got = explorer.snapshot_now();
        assert_eq!(
            got.len(),
            truth[target].len(),
            "support size differs after rewinding to {target}"
        );
        let mut worst = 0.0f64;
        for (&(gi, ga), &(ti, ta)) in got.iter().zip(&truth[target]) {
            assert_eq!(gi, ti, "support differs after rewinding to {target}");
            worst = worst.max((ga - ta).norm());
        }
        assert!(
            worst < 1e-12,
            "rewinding to {target} must land on the fresh state, not near it: {worst:.3e}"
        );
    }
    assert_eq!(explorer.rewinds(), total + 1);
    assert!(
        explorer.applied() > total,
        "rewinding replays, and the counted work says so"
    );
}

#[test]
fn the_explorer_refuses_to_run_time_backwards_by_accident() {
    let sim = Simulator::<C64>::new();
    let plan = MemoPlan::from_circuit(&brickwork(6, 3), sim.registry(), &MemoConfig::default())
        .expect("plan");
    let mut explorer = Explorer::new(&sim, "dense", &plan, 2).expect("explorer");
    explorer.advance_to(4).expect("advance");
    assert!(
        explorer.advance_to(2).is_err(),
        "advancing backwards is a mistake, not a rewind"
    );
    assert!(
        explorer.rewind(9_999).is_err(),
        "rewinding ahead of the cursor is a mistake, not an advance"
    );
    assert_eq!(explorer.cursor(), 4, "and neither moved the state");
}

#[test]
fn exploring_variants_shares_the_prefix_and_stays_exact() {
    let sim = Simulator::<C64>::new();
    let prefix = brickwork(9, 5);
    let mut previous = 0.0f64;
    for count in [2usize, 4, 8] {
        let variants: Vec<Circuit> = (0..count)
            .map(|k| {
                let mut c = Circuit::new(9);
                c.gate("rz", vec![0.1 * (k as f64 + 1.0)], vec![0]);
                c.cx(0, 1);
                c
            })
            .collect();
        let r =
            explore(&sim, "dense", &prefix, &variants, &MemoConfig::default()).expect("explore");
        assert!(
            r.deviation < 1e-12,
            "sharing the prefix must not change any answer: {:.3e}",
            r.deviation
        );
        assert_eq!(r.variants, count);
        assert!(r.applied_memoized < r.applied_naive);
        assert!(
            r.counted_speedup() > previous,
            "the more variants share the prefix, the more the sharing is worth: \
             {:.2} was not above {previous:.2}",
            r.counted_speedup()
        );
        previous = r.counted_speedup();
        assert!(r.checkpoints >= 1 && r.journal_amplitudes > 0);
    }
    assert!(
        previous > 4.0,
        "eight variants over a shared prefix should save severalfold: {previous:.2}"
    );
}

#[test]
fn exploring_needs_a_variant_and_a_matching_width() {
    let sim = Simulator::<C64>::new();
    let prefix = brickwork(6, 2);
    assert!(explore(&sim, "dense", &prefix, &[], &MemoConfig::default()).is_err());
    let wrong = Circuit::new(5);
    assert!(explore(&sim, "dense", &prefix, &[wrong], &MemoConfig::default()).is_err());
}

#[test]
fn a_zero_fusion_width_refuses() {
    let sim = Simulator::<C64>::new();
    let cfg = MemoConfig {
        max_fuse: 0,
        ..Default::default()
    };
    assert!(MemoPlan::from_circuit(&library::ghz::<C64>(4), sim.registry(), &cfg).is_err());
}

#[test]
fn the_plan_is_exact_on_every_backend_that_can_hold_it() {
    let sim = Simulator::<C64>::new();
    let circuit = library::random_circuit::<C64>(8, 80, 0xC0FFEE);
    for backend in ["dense", "sparse", "adaptive"] {
        let dev = deviation(&sim, backend, &circuit, 4);
        assert!(dev < 1e-12, "{backend}: deviation {dev:.3e}");
    }
}

#[test]
fn the_content_key_is_generic_over_the_amplitude_algebra() {
    // The memo keys on Scalar::coeffs, so it works over any algebra
    // rather than only over C64 — and repeated geometry still collapses.
    let sim = Simulator::<Quaternion>::new();
    let mut circuit = Circuit::<Quaternion>::new(8);
    for pair in 0..4 {
        circuit.h(2 * pair);
        circuit.cx(2 * pair, 2 * pair + 1);
    }
    let plan =
        MemoPlan::from_circuit(&circuit, sim.registry(), &MemoConfig::default()).expect("plan");
    assert_eq!(plan.stats().ops_out, 4);
    assert_eq!(plan.stats().entries, 1);

    let truth = sim.run_on("dense", &circuit).expect("reference");
    let mut state = sim.backends().create("dense", 8).expect("state");
    state.reset();
    plan.run(state.as_mut()).expect("run");
    assert!(max_amplitude_deviation(truth.as_ref(), state.as_ref()) < 1e-12);
}

#[test]
fn the_whole_registry_survives_memoization() {
    // The strongest acceptance test available: circuits drawn from the
    // registry's own gates — whatever is registered, including aliases and
    // parametric gates at random angles — planned and run through the
    // memo, compared against the unmemoized run.
    let sim = Simulator::<C64>::new();
    let mut worst = 0.0f64;
    for case in 0..24u64 {
        let n = 3 + (case as usize % 4);
        let circuit = random_registry_circuit(sim.registry(), n, 40, 0x9E37 ^ case);
        for max_fuse in [1usize, 2, 4] {
            let dev = deviation(&sim, "dense", &circuit, max_fuse);
            worst = worst.max(dev);
            assert!(
                dev < 1e-12,
                "registry circuit #{case} (n={n}) at max_fuse={max_fuse}: {dev:.3e}"
            );
        }
    }
    assert!(worst < 1e-12, "worst deviation over the sweep: {worst:.3e}");
}
