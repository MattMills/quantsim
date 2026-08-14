//! The time system inside the register: Page–Wootters conditioning
//! over the bulk register's scale axis, native scale snapshots, the
//! clock-controlled tick, scale interferometry, mandatory
//! selector↔system entanglement measured — and the payoff: a selector
//! qudit over representation-heterogeneous branches whose joint cost
//! is the sum of its slices, measured against the single-backend
//! alternatives.

mod common;

use common::{assert_close, sim};
use quantsim::backend::{BulkState, DenseState, FactoredState, MpsState, SparseState};
use quantsim::prelude::*;

/// Tree-aligned pairs plus cross couplings: bond 2 at every level of
/// the depth-3 hierarchy (same state family as the bulk suite).
fn structured_bulk() -> BulkState<C64> {
    let mut c: Circuit<C64> = Circuit::new(8);
    for q in [0usize, 2, 4, 6] {
        c.h(q).cx(q, q + 1);
    }
    c.t(1).cx(1, 2).cx(5, 6).cx(3, 4).s(4);
    let state = sim().run_on("bulk", &c).unwrap();
    state
        .as_any()
        .downcast_ref::<BulkState<C64>>()
        .unwrap()
        .clone()
}

fn overlap(a: &dyn Backend<C64>, b: &dyn Backend<C64>) -> C64 {
    let mut acc = c64(0.0, 0.0);
    a.for_each_nonzero(&mut |i, x| acc += x.conj() * b.amplitude(i));
    acc
}

/// Full-basis comparison by hash-join on the branched side: one
/// enumeration per branch, no per-index queries into branches whose
/// point lookups are expensive (the graph-state bundle's are ~ms).
fn assert_contracts_to(branched: &BranchedRegister<C64>, dense: &DenseState<C64>, what: &str) {
    let mut map = std::collections::HashMap::new();
    branched.for_each_nonzero(&mut |i, a| {
        map.insert(i, a);
    });
    let mut dev = 0.0f64;
    for i in 0..(1u64 << dense.num_qubits()) {
        let d = dense.amplitude(i);
        let b = map.get(&i).copied().unwrap_or(c64(0.0, 0.0));
        dev = dev.max((d - b).norm());
    }
    assert!(dev < 1e-9, "{what}: deviation {dev}");
}

#[test]
fn scale_snapshots_equal_the_unfold_without_gates() {
    // The O(tree) node surgery and the replayed unfold produce the
    // same state at every depth: the projective store read out as a
    // family of states, one per scale, with no gate ever applied.
    let bulk = structured_bulk();
    let prog = bulk.unfold_program().unwrap();
    for depth in 0..=bulk.depth() {
        let snap = bulk.scale_snapshot(depth).unwrap();
        let mut replay = DenseState::<C64>::new(8).unwrap();
        prog.run_to_depth(&mut replay, depth).unwrap();
        let dev = max_amplitude_deviation(&snap, &replay);
        assert!(dev < 1e-12, "depth {depth}: surgery vs unfold {dev}");
        assert_close(snap.total_abs_sqr(), 1.0, 1e-9);
    }
    // Depth beyond the hierarchy is refused, not guessed.
    assert!(bulk.scale_snapshot(bulk.depth() + 1).is_err());
}

#[test]
fn conditioning_the_clock_reads_out_the_scale() {
    // Page–Wootters over the depth axis: the history register is one
    // static entangled state, and conditioning its clock on |ℓ⟩ yields
    // exactly the scale-ℓ view of the system. Uniform weights over all
    // four scales of the depth-3 hierarchy.
    let bulk = structured_bulk();
    let scales = bulk.depth() + 1;
    let w = c64(1.0 / (scales as f64).sqrt(), 0.0);
    for level in 0..scales {
        let mut hist = scale_history(&bulk, &vec![w; scales]).unwrap();
        assert_eq!(hist.selector_dim(), scales);
        let p = hist.condition(level).unwrap();
        assert_close(p, 1.0 / scales as f64, 1e-9);
        let snap = bulk.scale_snapshot(level).unwrap();
        let dev = max_amplitude_deviation(&hist, &snap);
        assert!(dev < 1e-9, "conditioned slice {level} deviates: {dev}");
    }
}

#[test]
fn the_clock_is_mandatorily_entangled_with_the_system() {
    // The selector↔system Schmidt rank equals the number of distinct
    // scale views held in superposition — measured from the slice
    // Gram, never by enumeration of the joint register. Conditioning
    // collapses it to 1. Measuring the selector draws from the Born
    // distribution over slices and collapses the same way.
    let bulk = structured_bulk();
    let scales = bulk.depth() + 1;
    let w = c64(1.0 / (scales as f64).sqrt(), 0.0);
    let hist = scale_history(&bulk, &vec![w; scales]).unwrap();
    assert_eq!(hist.selector_schmidt_rank().unwrap(), scales);
    let probs = hist.selector_probabilities().unwrap();
    for &p in &probs {
        assert_close(p, 1.0 / scales as f64, 1e-9);
    }

    let mut seen = [false; 4];
    for seed in 0..12 {
        let mut h = scale_history(&bulk, &vec![w; scales]).unwrap();
        let outcome = h.measure_selector(&mut Prng::new(seed)).unwrap();
        seen[outcome] = true;
        assert_eq!(h.selector_schmidt_rank().unwrap(), 1);
        let probs = h.selector_probabilities().unwrap();
        assert_close(probs[outcome], 1.0, 1e-9);
    }
    assert!(
        seen.iter().filter(|&&s| s).count() >= 2,
        "several scales must be drawn across seeds"
    );
}

#[test]
fn the_tick_is_one_level_of_information_flow() {
    // The clock-controlled refinement |ℓ⟩⟨ℓ|⊗V_ℓ with |ℓ⟩→|ℓ+1⟩:
    // after a tick, the slice that was at scale ℓ *is* the scale-(ℓ+1)
    // view — verified against independently built snapshots — and
    // slice norms are preserved (the levels are isometries).
    let bulk = structured_bulk();
    let prog = bulk.unfold_program().unwrap();
    let scales = bulk.depth() + 1; // clock dimension 4, scales 0..2 occupied
    let w = c64(1.0 / 3f64.sqrt(), 0.0);
    let mut hist = scale_history(&bulk, &[w, w, w]).unwrap();
    // scale_history sizes the qudit by the weight list; rebuild with
    // headroom for the tick.
    assert_eq!(hist.selector_dim(), 3);
    let mut with_headroom: Vec<C64> = vec![w, w, w];
    with_headroom.resize(scales, c64(0.0, 0.0));
    hist = scale_history(&bulk, &with_headroom).unwrap();
    assert_eq!(hist.selector_dim(), scales);

    tick(&mut hist, &prog).unwrap();
    for level in 0..3 {
        let target = bulk.scale_snapshot(level + 1).unwrap();
        for i in 0..(1u64 << 8) {
            let got = hist.flagged_amplitude(level + 1, i);
            let expect = w * target.amplitude(i);
            assert!(
                (got - expect).norm() < 1e-9,
                "post-tick slice {} index {i}",
                level + 1
            );
        }
    }
    let probs = hist.selector_probabilities().unwrap();
    assert_close(probs[0], 0.0, 1e-9);
    for &p in &probs[1..] {
        assert_close(p, 1.0 / 3.0, 1e-9);
    }

    // A clock at the top of its dimension refuses to tick, by name.
    let full = vec![c64(0.5, 0.0); scales];
    let mut topped = scale_history(&bulk, &full).unwrap();
    match tick(&mut topped, &prog) {
        Err(Error::InvalidState(msg)) => assert!(msg.contains("top of the clock"), "{msg}"),
        other => panic!("expected top-of-clock refusal, got {other:?}"),
    }

    // Interfering the clock shares states across scales; ticking that
    // is refused, by name — condition or tick before interfering.
    let two = c64(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let mut mixed = scale_history(&bulk, &[two, two]).unwrap();
    let h = vec![two, two, two, -two];
    mixed.selector_mix(&h).unwrap();
    match tick(&mut mixed, &prog) {
        Err(Error::InvalidState(msg)) => assert!(msg.contains("shared across scales"), "{msg}"),
        other => panic!("expected shared-scale refusal, got {other:?}"),
    }
}

#[test]
fn scale_interferometry_measures_the_flow() {
    // Interfere the clock between two adjacent scales: the Born
    // distribution over the mixed selector reads
    // p± = (1 ± Re⟨snap_ℓ|snap_{ℓ+1}⟩)/2 — the overlap between
    // renormalization steps, i.e. how far one level of refinement
    // moves the state, measured as clock statistics inside the
    // register.
    let bulk = structured_bulk();
    let f = std::f64::consts::FRAC_1_SQRT_2;
    let two = c64(f, 0.0);
    let h = vec![two, two, two, -two];

    let mut hist = scale_history(&bulk, &[two, two]).unwrap();
    hist.selector_mix(&h).unwrap();
    let probs = hist.selector_probabilities().unwrap();
    let snap0 = bulk.scale_snapshot(0).unwrap();
    let snap1 = bulk.scale_snapshot(1).unwrap();
    let ov = overlap(&snap0, &snap1).re;
    assert_close(probs[0], (1.0 + ov) / 2.0, 1e-9);
    assert_close(probs[1], (1.0 - ov) / 2.0, 1e-9);
    assert!(
        probs[1] > 0.05,
        "an entangling state must move under refinement: p₋ = {}",
        probs[1]
    );

    // A register with nothing to refine shows full visibility: the
    // clock interferes to certainty, the flow speed is zero.
    let still = BulkState::<C64>::new(8).unwrap();
    let mut quiet = scale_history(&still, &[two, two]).unwrap();
    quiet.selector_mix(&h).unwrap();
    let probs = quiet.selector_probabilities().unwrap();
    assert_close(probs[0], 1.0, 1e-9);
    assert_close(probs[1], 0.0, 1e-9);
}

/// A random long-range graph state: an `H` wall and seeded `cz` edges.
/// All-Clifford (bundle-native); its cut rank survives any qubit
/// reordering, so routing tricks buy nothing.
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

#[test]
fn the_qudit_lowers_the_representation() {
    // Four width-16 slices, one per structured backend, each hostile
    // to the others' bets: GHZ (sparse: support 2; one 16-wide cluster
    // defeats factored), rainbow (factored: 8 disjoint pairs; support
    // 2^8 defeats sparse), a seeded brickwork (MPS: bond ≤ 8; support
    // 2^16 defeats sparse), and a random long-range graph state
    // (bundle: O(n + |E|); permutation-robust cut rank, dense support).
    // The selector qudit holds their superposition at the *sum* of the
    // four cheap costs; the measured single-backend alternatives pay
    // orders of magnitude more wherever one slice clashes with their
    // structural bet. More dimensions above, less structure below.
    //
    // (A finding worth keeping: rainbow does NOT defeat this MPS — its
    // persistent SWAP routing re-localizes the nested pairs to bond 2.
    // The graph state is the permutation-robust replacement.)
    let n = 16;
    let s = sim();
    let reg = GateRegistry::<C64>::standard();
    let ghz_c = library::ghz::<C64>(n);
    let rainbow_c = library::rainbow::<C64>(n);
    let brick_c = library::brickwork::<C64>(n, 3, &(0..n).collect::<Vec<_>>());
    let graph_c = graph_circuit(n, 24, 9);

    let run_in = |circuit: &Circuit<C64>, state: &mut dyn Backend<C64>| {
        circuit.bind(&reg).unwrap().run(state).unwrap();
    };
    let mut ghz_b: Box<dyn Backend<C64>> = Box::new(SparseState::<C64>::new(n).unwrap());
    let mut rainbow_b: Box<dyn Backend<C64>> = Box::new(FactoredState::<C64>::new(n).unwrap());
    let mut brick_b: Box<dyn Backend<C64>> = Box::new(MpsState::<C64>::new(n).unwrap());
    let mut graph_b = s.backends().create("bundle", n).unwrap();
    run_in(&ghz_c, ghz_b.as_mut());
    run_in(&rainbow_c, rainbow_b.as_mut());
    run_in(&brick_c, brick_b.as_mut());
    run_in(&graph_c, graph_b.as_mut());

    let w = c64(0.5, 0.0);
    let branched = BranchedRegister::from_branches(
        n,
        4,
        vec![
            (0, w, ghz_b),
            (1, w, rainbow_b),
            (2, w, brick_b),
            (3, w, graph_b),
        ],
    )
    .unwrap();
    let branched_bytes = branched.memory_bytes();
    assert_eq!(branched.selector_schmidt_rank().unwrap(), 4);

    // The contracted view is the exact weighted sum — full conformance
    // against a dense reference assembled slice by slice.
    let mut reference = vec![c64(0.0, 0.0); 1 << n];
    for c in [&ghz_c, &rainbow_c, &brick_c, &graph_c] {
        let d = s.run(c).unwrap();
        d.for_each_nonzero(&mut |i, a| reference[i as usize] += w * a);
    }
    let mut dense_sum = DenseState::<C64>::new(n).unwrap();
    dense_sum
        .load(
            &reference
                .iter()
                .enumerate()
                .filter(|(_, a)| a.norm_sqr() > 0.0)
                .map(|(i, &a)| (i as u64, a))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    assert_contracts_to(&branched, &dense_sum, "contracted view vs the sum");

    // The measured single-backend alternatives, on the slices that
    // clash with their bets: each pays an order of magnitude or more
    // over the entire four-slice register.
    let mut ghz_on_factored = FactoredState::<C64>::new(n).unwrap();
    run_in(&ghz_c, &mut ghz_on_factored);
    let mut brick_on_sparse = SparseState::<C64>::new(n).unwrap();
    run_in(&brick_c, &mut brick_on_sparse);
    let mut graph_on_factored = FactoredState::<C64>::new(n).unwrap();
    run_in(&graph_c, &mut graph_on_factored);
    for (label, bytes) in [
        ("ghz-on-factored", ghz_on_factored.memory_bytes()),
        ("brick-on-sparse", brick_on_sparse.memory_bytes()),
        ("graph-on-factored", graph_on_factored.memory_bytes()),
        ("dense sum", dense_sum.memory_bytes()),
    ] {
        assert!(
            branched_bytes * 10 < bytes,
            "branched {branched_bytes} B vs {label} {bytes} B"
        );
    }

    // System gates broadcast: evolve both sides by the same unitary
    // and the contracted view still conforms — additivity survives
    // dynamics. Diagonal gates are the phase-faithful sector for every
    // branch here (the graph-state bundle included).
    let mut branched = branched;
    let s_gate = reg.resolve("s").unwrap().matrix(&[]).unwrap();
    let cz = reg.resolve("cz").unwrap().matrix(&[]).unwrap();
    branched.apply(&s_gate, &[3]).unwrap();
    branched.apply(&cz, &[3, 4]).unwrap();
    dense_sum.apply(&s_gate, &[3]).unwrap();
    dense_sum.apply(&cz, &[3, 4]).unwrap();
    assert_contracts_to(&branched, &dense_sum, "broadcast dynamics");
    assert_eq!(branched.unique_state_count(), 4, "no branch was copied");

    // The measured boundary of the contract: a superposition register
    // makes branch *global* phases relative, hence physical. The
    // bundle maintains states only up to global phase (its own
    // documented convention — vertex-operator reductions rotate it;
    // classified in development: after H then CX every nonzero bundle
    // amplitude sits at exactly e^{−iπ/4} times dense). Alone that is
    // unobservable; inside the selector superposition it corrupts the
    // contracted sum — measured here, so the requirement is explicit:
    // branches under broadcast dynamics must be phase-faithful, and a
    // phase-loose representation is safe only as a static slice.
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let cx = reg.resolve("cx").unwrap().matrix(&[]).unwrap();
    branched.apply(&h, &[3]).unwrap();
    branched.apply(&cx, &[3, 4]).unwrap();
    dense_sum.apply(&h, &[3]).unwrap();
    dense_sum.apply(&cx, &[3, 4]).unwrap();
    let mut map = std::collections::HashMap::new();
    branched.for_each_nonzero(&mut |i, a| {
        map.insert(i, a);
    });
    let mut dev = 0.0f64;
    for i in 0..(1u64 << n) {
        let d = dense_sum.amplitude(i);
        let b = map.get(&i).copied().unwrap_or(c64(0.0, 0.0));
        dev = dev.max((d - b).norm());
    }
    assert!(
        dev > 1e-4,
        "the phase-loose boundary must be visible, not silent: dev = {dev:.3e}"
    );
}

#[test]
fn wide_scale_histories_stay_native() {
    // A width-32 GHZ register's full scale history — six scales in one
    // clock — built entirely by node surgery: kilobytes, no dense
    // object anywhere, weights and conditioning still exact. The wide
    // GHZ itself is built the dynamic way (grow-as-you-entangle keeps
    // every gate block small; a full-width chain would honestly refuse
    // its 2^32 root block).
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let cx = reg.resolve("cx").unwrap().matrix(&[]).unwrap();
    let mut grown = BulkState::<C64>::new(2).unwrap();
    grown.apply(&h, &[0]).unwrap();
    grown.apply(&cx, &[0, 1]).unwrap();
    while grown.num_qubits() < 32 {
        let q = grown.num_qubits();
        grown.grow(1).unwrap();
        grown.apply(&cx, &[q - 1, q]).unwrap();
    }
    let bulk = &grown;
    let scales = bulk.depth() + 1;
    let w = c64(1.0 / (scales as f64).sqrt(), 0.0);
    let hist = scale_history(bulk, &vec![w; scales]).unwrap();
    assert_eq!(hist.selector_dim(), scales);
    assert!(
        hist.memory_bytes() < 128 * 1024,
        "history memory: {} bytes",
        hist.memory_bytes()
    );
    let probs = hist.selector_probabilities().unwrap();
    for &p in &probs {
        assert_close(p, 1.0 / scales as f64, 1e-9);
    }
    assert_eq!(hist.selector_schmidt_rank().unwrap(), scales);
    // The finest slice is the register itself: spot-check the GHZ
    // amplitudes through the flagged view.
    let f = std::f64::consts::FRAC_1_SQRT_2;
    let fine = bulk.depth();
    let a0 = hist.flagged_amplitude(fine, 0);
    let a1 = hist.flagged_amplitude(fine, (1u64 << 32) - 1);
    assert_close(a0.re, w.re * f, 1e-9);
    assert_close(a1.re, w.re * f, 1e-9);
}
