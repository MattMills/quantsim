//! The dynamically scaled bulk–boundary register: conformance, the
//! gauge invariant (norm as a bulk-record read), exact disturbance-free
//! growth, verified release, streaming past the register's own width,
//! environment-weighted truncation with a certified per-depth ledger,
//! and the unfold — the depth store replayed as a width-growing circuit
//! whose wires are entangled only along their structural sequences.

mod common;

use common::{assert_close, sim};
use quantsim::backend::{BulkConfig, BulkState, MeraConfig, MeraState};
use quantsim::math::svd_thin;
use quantsim::prelude::*;

fn ghz_matrices() -> (GateMatrix<C64>, GateMatrix<C64>) {
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let cx = reg.resolve("cx").unwrap().matrix(&[]).unwrap();
    (h, cx)
}

/// Euclidean distance between two states over the full 2^n basis.
fn l2_distance(a: &dyn Backend<C64>, b: &dyn Backend<C64>, n: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..(1u64 << n) {
        sum += (a.amplitude(i) - b.amplitude(i)).norm_sqr();
    }
    sum.sqrt()
}

#[test]
fn bulk_conforms_over_the_full_registry() {
    let report = verify_backend(&sim(), "bulk", &ConformanceConfig::default()).unwrap();
    assert!(report.passed(), "{report}");
    assert!(report.max_amplitude_deviation < 1e-9, "{report}");
    assert_eq!(report.sampling_mismatches, 0);
    assert_eq!(report.collapse_violations, 0);
}

#[test]
fn norm_is_a_bulk_record_read_at_any_width() {
    // The gauge invariant: every tensor below the record is isometric,
    // so ‖ψ‖² is Σ|top|² — an O(χ) read at width 40, where a flat
    // register could never even be materialized. Blocks sit at 8-qubit
    // dyadic boundaries: the capacity tree is a perfect binary tree
    // (the price of O(1) growth), so "hierarchy-local" means aligned
    // to its dyadic blocks. Cross-checked against the honest
    // enumeration at width 8.
    let n = 40;
    let mut state = BulkState::<C64>::new(n).unwrap();
    let (h, cx) = ghz_matrices();
    let reg = GateRegistry::<C64>::standard();
    let t = reg.resolve("t").unwrap().matrix(&[]).unwrap();
    for block in 0..5 {
        let base = block * 8;
        state.apply(&h, &[base]).unwrap();
        state.apply(&cx, &[base, base + 1]).unwrap();
        state.apply(&t, &[base + 1]).unwrap();
        state.apply(&cx, &[base + 1, base + 2]).unwrap();
        state.apply(&cx, &[base + 2, base + 3]).unwrap();
        state.apply(&cx, &[base + 3, base + 4]).unwrap();
    }
    assert_close(state.total_abs_sqr(), 1.0, 1e-9);
    assert_close(state.total_weight(), 1.0, 1e-9);
    assert!(state.top().len() <= state.config().max_bond);
    assert!(
        state.memory_bytes() < 64 * 1024,
        "hierarchical memory: {} bytes",
        state.memory_bytes()
    );

    // At small width the record read agrees with the enumeration.
    let mut small = BulkState::<C64>::new(8).unwrap();
    let c = library::random_circuit(8, 30, 5);
    c.bind(&reg).unwrap().run(&mut small).unwrap();
    let mut enumerated = 0.0;
    small.for_each_nonzero(&mut |_, a| enumerated += a.norm_sqr());
    assert_close(small.total_abs_sqr(), enumerated, 1e-9);
}

#[test]
fn the_record_is_the_depth_zero_view_and_ghz_is_a_coarse_pair() {
    // coarse_state(0) *is* the bulk record; GHZ(16) at depth 1 is the
    // maximally entangled super-site pair, exactly as on the rung-1
    // hierarchy — the projective store preserves the coarse semantics.
    let n = 16;
    let state = sim().run_on("bulk", &library::ghz(n)).unwrap();
    let bulk = state.as_any().downcast_ref::<BulkState<C64>>().unwrap();

    assert!(bulk.is_exact());
    assert_eq!(bulk.max_bond_dimension(), 2);

    let (dims0, coeffs0) = bulk.coarse_state(0).unwrap();
    assert_eq!(dims0, vec![bulk.top().len()]);
    for (a, b) in coeffs0.iter().zip(bulk.top().iter()) {
        assert!((*a - *b).norm() < 1e-12, "depth-0 view must be the record");
    }

    let (cdims, coeffs) = bulk.coarse_state(1).unwrap();
    assert_eq!(cdims, vec![2, 2]);
    let m = [coeffs[0], coeffs[1], coeffs[2], coeffs[3]];
    let gram00 = (m[0] * m[0].conj() + m[2] * m[2].conj()).re;
    let gram11 = (m[1] * m[1].conj() + m[3] * m[3].conj()).re;
    let gram01 = m[0] * m[1].conj() + m[2] * m[3].conj();
    assert_close(gram00, 0.5, 1e-9);
    assert_close(gram11, 0.5, 1e-9);
    assert!(gram01.norm() < 1e-9);

    // Weight is conserved at every resolution.
    for depth in 0..=bulk.depth() {
        let (_, cs) = bulk.coarse_state(depth).unwrap();
        let w: f64 = cs.iter().map(|a| a.norm_sqr()).sum();
        assert_close(w, 1.0, 1e-9);
    }
}

#[test]
fn growth_is_exact_disturbance_free_and_amortized() {
    // Grow a live, entangled GHZ register one qubit at a time from 2 to
    // 32. Amplitudes stay exact against the closed form the whole way,
    // re-roots happen log-many times (capacity doublings), and memory
    // stays linear. Growth within capacity is pure bookkeeping.
    let (h, cx) = ghz_matrices();
    let mut s = BulkState::<C64>::new(2).unwrap();
    s.apply(&h, &[0]).unwrap();
    s.apply(&cx, &[0, 1]).unwrap();
    let f = std::f64::consts::FRAC_1_SQRT_2;
    while s.num_qubits() < 32 {
        let q = s.num_qubits();
        s.grow(1).unwrap();
        s.apply(&cx, &[q - 1, q]).unwrap();
        let n = s.num_qubits();
        let a0 = s.amplitude(0);
        let a1 = s.amplitude((1u64 << n) - 1);
        assert!(
            (a0.re - f).abs() < 1e-9 && (a1.re - f).abs() < 1e-9,
            "GHZ broken at width {n}"
        );
        assert!(s.is_exact());
        assert_eq!(s.max_bond_dimension(), 2);
    }
    assert_eq!(s.num_qubits(), 32);
    assert_eq!(s.capacity(), 32);
    // 2 → 32 is exactly four capacity doublings: 4 re-roots for 30 grows.
    assert_eq!(s.reroots(), 4, "growth must amortize by doubling");
    assert!(s.memory_bytes() < 16 * 1024, "{} bytes", s.memory_bytes());

    // Against dense at small width: growth then entangling equals the
    // circuit built at the full width from the start.
    let mut grown = BulkState::<C64>::new(2).unwrap();
    grown.apply(&h, &[0]).unwrap();
    grown.apply(&cx, &[0, 1]).unwrap();
    grown.grow(3).unwrap();
    for q in 1..4 {
        grown.apply(&cx, &[q, q + 1]).unwrap();
    }
    let dense = sim().run(&library::ghz(5)).unwrap();
    let dev = max_amplitude_deviation(dense.as_ref(), &grown);
    assert!(
        dev < 1e-12,
        "grown register deviates from fresh dense: {dev}"
    );

    // Within capacity (5 → 8), growth costs no re-root and no tensors.
    let before = grown.reroots();
    grown.grow(3).unwrap();
    assert_eq!(grown.reroots(), before);
    assert_eq!(grown.num_qubits(), 8);
    assert_close(grown.total_abs_sqr(), 1.0, 1e-12);
}

#[test]
fn release_is_verified_never_assumed() {
    let (h, cx) = ghz_matrices();

    // An entangled boundary qubit refuses release, with the measured
    // leakage in the error.
    let mut s = BulkState::<C64>::new(3).unwrap();
    s.apply(&h, &[0]).unwrap();
    s.apply(&cx, &[0, 1]).unwrap();
    s.apply(&cx, &[1, 2]).unwrap();
    match s.release(1) {
        Err(Error::InvalidState(msg)) => {
            assert!(msg.contains("5.000e-1"), "leakage must be measured: {msg}")
        }
        other => panic!("entangled release must refuse: {other:?}"),
    }
    assert_eq!(s.num_qubits(), 3, "refusal must not shrink the register");

    // Uncompute, then release: exact, and the remaining state is the
    // untouched Bell pair.
    s.apply(&cx, &[1, 2]).unwrap();
    s.release(1).unwrap();
    assert_eq!(s.num_qubits(), 2);
    let dense = sim().run(&library::bell()).unwrap();
    let dev = max_amplitude_deviation(dense.as_ref(), &s);
    assert!(dev < 1e-12, "release disturbed the surviving state: {dev}");

    // Width-1 registers refuse to shrink further.
    let mut one = BulkState::<C64>::new(1).unwrap();
    assert!(one.release(1).is_err());

    // Measured release always succeeds and collapses correctly: on a
    // Bell pair, releasing qubit 1 forces qubit 0 to the same outcome.
    let mut seen = [false, false];
    for seed in 0..20 {
        let mut b = BulkState::<C64>::new(2).unwrap();
        b.apply(&h, &[0]).unwrap();
        b.apply(&cx, &[0, 1]).unwrap();
        let mut rng = Prng::new(seed);
        let outs = b.release_measured(1, &mut rng).unwrap();
        assert_eq!(b.num_qubits(), 1);
        let idx = u64::from(outs[0]);
        assert_close(b.probability(idx), 1.0, 1e-9);
        assert_close(b.total_abs_sqr(), 1.0, 1e-9);
        seen[usize::from(outs[0])] = true;
    }
    assert!(seen[0] && seen[1], "both outcomes must occur across seeds");
}

#[test]
fn a_stream_wider_than_the_register() {
    // Sequential emission: one source qubit entangles a fresh boundary
    // qubit per round, which is measured and released. 48 logical
    // qubits pass through a register that is never wider than 2 — width
    // tracks live entanglement, not problem size. All outcomes agree
    // (the GHZ-stream correlation), and both stream values occur across
    // seeds.
    let (h, cx) = ghz_matrices();
    let mut seen = [false, false];
    for seed in [3u64, 4, 5, 6] {
        let mut s = BulkState::<C64>::new(1).unwrap();
        s.apply(&h, &[0]).unwrap();
        let mut rng = Prng::new(seed);
        let mut outcomes = Vec::new();
        for _ in 0..48 {
            s.grow(1).unwrap();
            s.apply(&cx, &[0, 1]).unwrap();
            outcomes.extend(s.release_measured(1, &mut rng).unwrap());
        }
        assert_eq!(outcomes.len(), 48);
        assert!(
            outcomes.iter().all(|&o| o == outcomes[0]),
            "stream must be perfectly correlated (seed {seed})"
        );
        seen[usize::from(outcomes[0])] = true;
        assert_eq!(
            s.peak_width(),
            2,
            "the register never held more than 2 qubits"
        );
        assert_eq!(s.releases(), 48);
        assert_eq!(s.num_qubits(), 1);
        assert_close(s.total_abs_sqr(), 1.0, 1e-9);
    }
    assert!(
        seen[0] && seen[1],
        "both stream values must occur across seeds"
    );
}

#[test]
fn truncation_is_environment_weighted_with_a_certified_ledger() {
    // Under a tight bond cap, every discarded singular value is global
    // (environment-weighted) Schmidt weight, so Σ√ε certifies the L2
    // error against dense — asserted across seeds. The per-depth ledger
    // localizes where resolution was spent, ‖top‖ honestly reflects the
    // loss, and on a pinned seed the weighted rebuild beats the
    // block-local one by more than 4×. (No dominance claim: greedy
    // sequential truncations do not commute, and seeds exist where the
    // orderings trade places — the certified bound is the invariant.)
    let s = sim();
    let reg = GateRegistry::<C64>::standard();
    for seed in 0..5u64 {
        let n = 8;
        let c = library::random_circuit(n, 40, seed);
        let dense = s.run(&c).unwrap();
        let mut bulk = BulkState::<C64>::with_config(
            n,
            BulkConfig {
                max_bond: 2,
                ..BulkConfig::default()
            },
        )
        .unwrap();
        c.bind(&reg).unwrap().run(&mut bulk).unwrap();
        let err = l2_distance(dense.as_ref(), &bulk, n);
        let bound = bulk.l2_error_bound();
        assert!(
            err <= bound + 1e-9,
            "seed {seed}: measured L2 {err} exceeds certified bound {bound}"
        );
        let by_depth = bulk.discarded_by_depth();
        assert_close(by_depth.iter().sum::<f64>(), bulk.discarded_weight(), 1e-12);
        if bulk.discarded_weight() > 1e-9 {
            assert!(
                bulk.total_abs_sqr() < 1.0 - 1e-9,
                "the record must reflect projective loss"
            );
        }
    }

    // Pinned: seed 1 at cap 2 — environment weighting keeps the state
    // block-local truncation throws away.
    let n = 8;
    let c = library::random_circuit(n, 40, 1);
    let dense = s.run(&c).unwrap();
    let bound_c = c.bind(&reg).unwrap();
    let mut bulk = BulkState::<C64>::with_config(
        n,
        BulkConfig {
            max_bond: 2,
            ..BulkConfig::default()
        },
    )
    .unwrap();
    let mut mera = MeraState::<C64>::with_config(
        n,
        MeraConfig {
            max_bond: 2,
            trunc_tol: 1e-12,
            max_block: 63,
        },
    )
    .unwrap();
    bound_c.run(&mut bulk).unwrap();
    bound_c.run(&mut mera).unwrap();
    let eb = l2_distance(dense.as_ref(), &bulk, n);
    let em = l2_distance(dense.as_ref(), &mera, n);
    assert!(
        eb * 4.0 < em,
        "environment weighting must beat block-local here: bulk {eb:.4} vs mera {em:.4}"
    );

    // Generous cap on the same circuit: exact, empty ledger.
    let mut generous = BulkState::<C64>::new(n).unwrap();
    bound_c.run(&mut generous).unwrap();
    assert!(generous.is_exact());
    assert!(generous.l2_error_bound() < 1e-9);
    let dev = max_amplitude_deviation(dense.as_ref(), &generous);
    assert!(dev < 1e-9, "generous cap deviates: {dev}");
}

/// The structured test state: tree-aligned Bell pairs plus cross-pair
/// couplings, so every level of the depth-3 hierarchy carries bond 2.
fn structured_state() -> BulkState<C64> {
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

#[test]
fn the_unfold_replays_the_depth_store_exactly() {
    let bulk = structured_state();
    assert!(bulk.is_exact());
    let prog = bulk.unfold_program().unwrap();
    assert!(prog.max_step_qubits() <= quantsim::backend::UNFOLD_MAX_STEP_QUBITS);

    let mut dense = DenseState::<C64>::new(8).unwrap();
    prog.run_on(&mut dense).unwrap();
    let dev = max_amplitude_deviation(&bulk, &dense);
    assert!(dev < 1e-12, "unfold replay deviates: {dev}");

    // The dilated steps are honest unitaries.
    for step in &prog.steps {
        let d = step.matrix.dim();
        for i in 0..d {
            for j in 0..d {
                let mut acc = c64(0.0, 0.0);
                for k in 0..d {
                    acc += step.matrix.get(k, i).conj() * step.matrix.get(k, j);
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (acc.re - expect).abs() < 1e-9 && acc.im.abs() < 1e-9,
                    "step at depth {} not unitary",
                    step.depth
                );
            }
        }
    }
}

#[test]
fn unfold_snapshots_are_the_coarse_states() {
    // Stopping the unfold after the steps of depth < ℓ leaves exactly
    // the coarse state at depth ℓ on the channel wires, every
    // not-yet-injected wire exactly |0⟩: the projective depth store,
    // replayed level by level.
    let bulk = structured_state();
    let prog = bulk.unfold_program().unwrap();
    let n = 8;
    let clog2 = |x: usize| x.next_power_of_two().trailing_zeros() as usize;

    for depth in 0..=bulk.depth() {
        let mut snap = DenseState::<C64>::new(n).unwrap();
        prog.run_to_depth(&mut snap, depth).unwrap();
        let frontier = bulk.coarse_dims(depth);
        let (dims, coeffs) = bulk.coarse_state(depth).unwrap();
        assert_eq!(dims.len(), frontier.len());
        // Carrier wires per frontier site: [lo, lo + ⌈log₂ bond⌉).
        let carriers: Vec<(usize, usize)> = frontier
            .iter()
            .map(|&(lo, _, bond)| (lo, clog2(bond)))
            .collect();
        for index in 0..(1u64 << n) {
            // Decode the frontier digits from the carrier bits; any
            // weight outside carriers, or an out-of-range digit, must
            // vanish.
            let mut rest = index;
            let mut digit_index = 0usize;
            let mut stride = 1usize;
            let mut valid = true;
            for (site, &(lo, c)) in carriers.iter().enumerate() {
                let mut digit = 0usize;
                for i in 0..c {
                    if (index >> (lo + i)) & 1 == 1 {
                        digit |= 1 << i;
                        rest &= !(1u64 << (lo + i));
                    }
                }
                if digit >= dims[site] {
                    valid = false;
                }
                digit_index += digit * stride;
                stride *= dims[site];
            }
            let expected = if rest != 0 || !valid {
                c64(0.0, 0.0)
            } else {
                coeffs[digit_index]
            };
            let got = snap.amplitude(index);
            assert!(
                (got - expected).norm() < 1e-9,
                "depth {depth} index {index}: snapshot {got} vs coarse {expected}"
            );
        }
    }
}

#[test]
fn entanglement_is_sequenced_and_channel_bounded() {
    // Each wire's interaction history is its structural sequence: a
    // nested chain of spans down the tree, nothing else. Entanglement
    // between a tree-aligned region and the rest is carried by the one
    // channel crossing the cut: Schmidt rank ≤ bond, measured by SVD on
    // the replayed state.
    let bulk = structured_state();
    let prog = bulk.unfold_program().unwrap();
    let n = 8;

    for q in 0..n {
        let seq = prog.sequence(q);
        assert!(!seq.is_empty(), "wire {q} must participate");
        for w in seq.windows(2) {
            let ((d0, (l0, h0)), (d1, (l1, h1))) = (w[0], w[1]);
            assert!(d1 > d0, "wire {q}: sequence must descend in depth");
            assert!(l1 >= l0 && h1 <= h0, "wire {q}: spans must nest");
            assert!((l1..h1).contains(&q));
        }
        assert!(prog.injection_depth(q).is_some());
    }

    // Tree-aligned cuts: region [lo, hi) vs rest, Schmidt rank from the
    // dense replay must not exceed the bond of the crossing channel.
    let mut dense = DenseState::<C64>::new(n).unwrap();
    prog.run_on(&mut dense).unwrap();
    for depth in 1..=bulk.depth() {
        for &(lo, hi, bond) in &bulk.coarse_dims(depth) {
            let w = hi - lo;
            let rows = 1usize << w;
            let cols = 1usize << (n - w);
            let mut m = vec![c64(0.0, 0.0); rows * cols];
            for i in 0..(1u64 << n) {
                let inner = ((i >> lo) & ((1 << w) - 1)) as usize;
                let outer = (((i >> hi) << lo) | (i & ((1 << lo) - 1))) as usize;
                m[inner * cols + outer] = dense.amplitude(i);
            }
            let svd = svd_thin(rows, cols, &m, 0.0, 0.0).unwrap();
            let rank = svd.sigma.iter().filter(|&&s| s > 1e-9).count();
            assert!(
                rank <= bond,
                "region [{lo},{hi}): measured Schmidt rank {rank} exceeds channel bond {bond}"
            );
        }
    }
}

#[test]
fn the_depth_ledger_localizes_resolution_spend() {
    // Leaf-local circuits never truncate; a cross-root entangler under
    // a tight cap spends resolution at the root levels — the ledger
    // names the depth.
    let (h, cx) = ghz_matrices();
    let mut local = BulkState::<C64>::with_config(
        8,
        BulkConfig {
            max_bond: 2,
            ..BulkConfig::default()
        },
    )
    .unwrap();
    for q in [0usize, 2, 4, 6] {
        local.apply(&h, &[q]).unwrap();
        local.apply(&cx, &[q, q + 1]).unwrap();
    }
    assert!(local.is_exact(), "pair-local circuits are exact at cap 2");

    // Rainbow pairs across the root at cap 2: rank demand 4 across the
    // middle, truncated — and the ledger localizes the loss at the top.
    let mut cross = BulkState::<C64>::with_config(
        8,
        BulkConfig {
            max_bond: 2,
            ..BulkConfig::default()
        },
    )
    .unwrap();
    let c: Circuit<C64> = library::rainbow(8);
    c.bind(sim().registry()).unwrap().run(&mut cross).unwrap();
    assert!(!cross.is_exact());
    let by_depth = cross.discarded_by_depth();
    let top_levels: f64 = by_depth.iter().take(2).sum();
    let leaf_levels: f64 = by_depth.iter().skip(2).sum();
    assert!(
        top_levels > leaf_levels,
        "rainbow loss must sit at the root: {by_depth:?}"
    );
}

#[test]
fn bulk_composes_with_ball_certified_scalars() {
    // The projective store over Ball: splits run on midpoints, radii
    // ride along — coarse representation, dynamic width and certified
    // numbers compose.
    let mut c: Circuit<Ball> = Circuit::new(6);
    c.h(0).cx(0, 1).cx(1, 2).t(2).cx(2, 3).cx(3, 4).cx(4, 5);
    let state = Simulator::<Ball>::new().run_on("bulk", &c).unwrap();
    assert_close(state.total_weight(), 1.0, 1e-9);
    let a = state.amplitude(0);
    assert!(a.rad >= 0.0 && a.rad < 1e-9);
    assert_close(a.mid.norm_sqr(), 0.5, 1e-9);

    // Dynamic scaling composes too: grow a Ball register and extend.
    let mut b = BulkState::<Ball>::new(2).unwrap();
    let reg = GateRegistry::<Ball>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let cx = reg.resolve("cx").unwrap().matrix(&[]).unwrap();
    b.apply(&h, &[0]).unwrap();
    b.apply(&cx, &[0, 1]).unwrap();
    b.grow(1).unwrap();
    b.apply(&cx, &[1, 2]).unwrap();
    assert_close(b.total_weight(), 1.0, 1e-9);
    let a = b.amplitude(0b111);
    assert!(a.rad < 1e-9);
    assert_close(a.mid.norm_sqr(), 0.5, 1e-9);
}
