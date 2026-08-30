//! The overlay as a register: it holds the state, and the layout is a
//! constraint on it rather than commentary about it.

use quantsim::curve::{GridOrder, Order, Overlay};
use quantsim::overlay::*;
use quantsim::{Backend, DenseState, Error, GateMatrix, GateRegistry, Prng, Scalar, C64};

/// Every lattice edge of a `side × side` grid, as site pairs.
fn edges(side: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for y in 0..side {
        for x in 0..side {
            let s = y * side + x;
            if x + 1 < side {
                out.push((s, s + 1));
            }
            if y + 1 < side {
                out.push((s, s + side));
            }
        }
    }
    out
}

/// Edges inside each `q × q` patch — a circuit that is local in the
/// lattice, which is the case a layout is supposed to serve.
fn patch_edges(side: usize, q: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for by in (0..side).step_by(q) {
        for bx in (0..side).step_by(q) {
            for y in by..by + q {
                for x in bx..bx + q {
                    let s = y * side + x;
                    if x + 1 < bx + q {
                        out.push((s, s + 1));
                    }
                    if y + 1 < by + q {
                        out.push((s, s + side));
                    }
                }
            }
        }
    }
    out
}

/// A circuit with two localities in it: horizontal strips of four on
/// the left half of the lattice, `q × q` patches on the right. One
/// ordering serves one half well and the other badly.
fn strips_and_patches(side: usize, q: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let half = side / 2;
    for y in 0..side {
        for bx in (0..half).step_by(q) {
            for x in bx..(bx + q).min(half) - 1 {
                out.push((y * side + x, y * side + x + 1));
            }
        }
    }
    for by in (0..side).step_by(q) {
        for bx in (half..side).step_by(q) {
            for y in by..by + q {
                for x in bx..bx + q {
                    let s = y * side + x;
                    if x + 1 < bx + q {
                        out.push((s, s + 1));
                    }
                    if y + 1 < by + q {
                        out.push((s, s + side));
                    }
                }
            }
        }
    }
    out
}

fn cz_gate() -> GateMatrix<C64> {
    let reg: GateRegistry<C64> = GateRegistry::standard();
    reg.get("cz").unwrap().matrix(&[]).unwrap()
}

fn gates() -> (GateMatrix<C64>, GateMatrix<C64>, GateMatrix<C64>) {
    let reg: GateRegistry<C64> = GateRegistry::standard();
    let m = |n: &str| reg.get(n).unwrap().matrix(&[]).unwrap();
    (m("h"), m("t"), m("cx"))
}

/// A local 2D circuit: a layer of `h`, then `cx` along a shuffled
/// subset of lattice edges with a `t` between them.
fn local_circuit(side: usize, seed: u64, take: usize) -> Vec<(usize, Vec<usize>)> {
    let mut rng = Prng::new(seed);
    let mut es = edges(side);
    for i in (1..es.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        es.swap(i, j);
    }
    let mut ops: Vec<(usize, Vec<usize>)> = (0..side * side).map(|s| (0, vec![s])).collect();
    for &(a, b) in es.iter().take(take) {
        ops.push((1, vec![a]));
        ops.push((2, vec![a, b]));
    }
    ops
}

#[test]
fn the_register_holds_the_state_a_dense_backend_holds() {
    let side = 4;
    let n = side * side;
    let (h, t, cx) = gates();
    let mats = [&h, &t, &cx];
    for seed in 0..6u64 {
        let ops = local_circuit(side, seed, 6);
        let mut dense = DenseState::<C64>::new(n).unwrap();
        let mut reg = OverlayRegister::<C64>::family(side, Order::Hilbert).unwrap();
        for (g, qs) in &ops {
            dense.apply(mats[*g], qs).unwrap();
            reg.apply(mats[*g], qs).unwrap();
        }
        for i in 0..(1u64 << n) {
            let (a, b) = (dense.amplitude(i), reg.amplitude(i));
            assert!(
                (a - b).abs_sqr() < 1e-20,
                "seed {seed} index {i}: dense {a:?} vs register {b:?}"
            );
        }
        assert!(
            (reg.total_weight() - 1.0).abs() < 1e-12,
            "seed {seed}: weight {}",
            reg.total_weight()
        );
    }
}

#[test]
fn every_layout_holds_the_same_state() {
    let side = 4;
    let n = side * side;
    let (h, t, cx) = gates();
    let mats = [&h, &t, &cx];
    let ops = local_circuit(side, 11, 8);
    let mut states = Vec::new();
    for mut reg in [
        OverlayRegister::<C64>::single(side, Order::RowMajor).unwrap(),
        OverlayRegister::<C64>::single(side, Order::Snake).unwrap(),
        OverlayRegister::<C64>::single(side, Order::Hilbert).unwrap(),
        OverlayRegister::<C64>::family(side, Order::Hilbert).unwrap(),
    ] {
        for (g, qs) in &ops {
            reg.apply(mats[*g], qs).unwrap();
        }
        states.push(
            (0..(1u64 << n))
                .map(|i| reg.amplitude(i))
                .collect::<Vec<_>>(),
        );
    }
    for s in &states[1..] {
        for (i, (a, b)) in states[0].iter().zip(s).enumerate() {
            assert!((*a - *b).abs_sqr() < 1e-20, "layouts disagree at {i}");
        }
    }
}

#[test]
fn regions_partition_the_lattice_and_are_always_member_blocks() {
    let side = 4;
    let (h, t, cx) = gates();
    let mats = [&h, &t, &cx];
    let mut reg = OverlayRegister::<C64>::family(side, Order::Hilbert).unwrap();
    let ops = local_circuit(side, 3, 7);
    for (i, (g, qs)) in ops.iter().enumerate() {
        reg.apply(mats[*g], qs).unwrap();
        let mut seen = vec![false; reg.sites()];
        for (r, p) in reg.regions().iter().zip(reg.placements()) {
            assert_eq!(r.len(), p.width(), "op {i}: region size is the block size");
            // The region's sites are exactly the block's sites, in the
            // block's own chain order.
            assert_eq!(*r, reg.block_sites(p), "op {i}: region is its block");
            for &s in r {
                assert!(!seen[s], "op {i}: site {s} in two regions");
                seen[s] = true;
            }
        }
        assert!(seen.iter().all(|&b| b), "op {i}: every site is somewhere");
    }
}

#[test]
fn memory_is_the_sum_of_regions_not_two_to_the_width() {
    // A graph state on 2×2 patches of an 8×8 lattice: genuinely
    // entangled, and entangled only locally. A dense register is 2^64
    // amplitudes; this holds sixteen four-site regions.
    let side = 8;
    let (h, _, _) = gates();
    let cz = cz_gate();
    let mut reg = OverlayRegister::<C64>::single(side, Order::Hilbert).unwrap();
    assert_eq!(reg.region_count(), 64);
    assert_eq!(
        reg.memory_amplitudes(),
        128,
        "sixty-four sites, one qubit each"
    );
    for s in 0..reg.sites() {
        reg.apply(&h, &[s]).unwrap();
    }
    for (a, b) in patch_edges(side, 2) {
        reg.apply(&cz, &[a, b]).unwrap();
    }
    assert_eq!(reg.region_count(), 16, "one region per patch");
    assert_eq!(reg.widest(), 4);
    assert_eq!(reg.memory_amplitudes(), 16 * 16, "sixteen regions of 2^4");
    assert!(
        reg.memory_amplitudes() < 1u128 << 20,
        "a dense 64-qubit register does not exist"
    );
}

#[test]
fn a_layout_that_cannot_keep_the_gate_local_refuses_by_name() {
    // One vertical edge on an 8×8 lattice. Row-major puts its
    // endpoints eight chain positions apart, so the smallest block
    // holding both is sixteen sites — 65536 amplitudes for a two-site
    // gate. The Hilbert curve puts them adjacent.
    let side = 8;
    let (h, _, cx) = gates();
    let mut row = OverlayRegister::<C64>::single(side, Order::RowMajor).unwrap();
    let mut hil = OverlayRegister::<C64>::single(side, Order::Hilbert).unwrap();
    row.set_region_cap(4);
    hil.set_region_cap(4);
    row.apply(&h, &[0]).unwrap();
    hil.apply(&h, &[0]).unwrap();
    assert!(
        matches!(
            row.apply(&cx, &[0, side]),
            Err(Error::TooManyQubits {
                requested: 16,
                max: 4
            })
        ),
        "row-major names the block it needed"
    );
    hil.apply(&cx, &[0, side]).unwrap();
    assert_eq!(hil.widest(), 2, "the curve keeps the vertical edge local");
}

#[test]
fn the_curve_completes_a_local_circuit_the_rows_cannot() {
    let side = 8;
    let (h, t, cx) = gates();
    let mats = [&h, &t, &cx];
    let _ = mats;
    let cz = cz_gate();
    let ops = patch_edges(side, 4);
    let mut done = Vec::new();
    for order in [Order::RowMajor, Order::Snake, Order::Hilbert] {
        let mut reg = OverlayRegister::<C64>::single(side, order).unwrap();
        reg.set_region_cap(16);
        for s in 0..reg.sites() {
            reg.apply(&h, &[s]).unwrap();
        }
        let mut ok = 0usize;
        for &(a, b) in &ops {
            match reg.apply(&cz, &[a, b]) {
                Ok(()) => ok += 1,
                Err(Error::TooManyQubits { .. }) => break,
                Err(e) => panic!("{order:?}: {e}"),
            }
        }
        done.push((order, ok, reg.ledger().peak_width()));
    }
    assert_eq!(done[2].1, ops.len(), "the curve finishes: {done:?}");
    assert!(done[0].1 < ops.len(), "the rows do not: {done:?}");
    assert!(done[1].1 < ops.len(), "nor the snake: {done:?}");
    assert_eq!(done[2].2, 16, "and it never held more than one 4×4 patch");
}

#[test]
fn the_overlay_never_places_worse_than_one_member() {
    let side = 8;
    let one = OverlayRegister::<C64>::single(side, Order::Hilbert).unwrap();
    let many = OverlayRegister::<C64>::family(side, Order::Hilbert).unwrap();
    for &(a, b) in edges(side).iter() {
        let p1 = one.admits(&[a, b]).unwrap();
        let pm = many.admits(&[a, b]).unwrap();
        assert!(
            pm.level <= p1.level,
            "edge ({a},{b}): overlay {} vs single {}",
            pm.level,
            p1.level
        );
    }
}

#[test]
fn the_election_fires_where_the_rotor_leaves_a_choice() {
    // A vertical domino inside a quadrant: whether it survives depends
    // on which way the rotor splits that quadrant, so the members
    // disagree and some member wins outright.
    let side = 8;
    let many = OverlayRegister::<C64>::family(side, Order::Hilbert).unwrap();
    let decided = edges(side)
        .iter()
        .filter(|&&(a, b)| many.admits(&[a, b]).unwrap().contenders < many.members())
        .count();
    assert!(
        decided > 0,
        "some edges are placed by a strict subset of the members"
    );
    // And the whole-lattice block is the same under every member, so
    // there the election is a tie by construction.
    let corners = [0usize, side * side - 1];
    let p = many.admits(&corners).unwrap();
    assert_eq!(p.level, (side * side).ilog2());
    assert_eq!(p.contenders, many.members(), "the top block is a tie");
}

#[test]
fn the_election_shows_up_in_the_ledger_of_a_run() {
    let side = 8;
    let (h, t, cx) = gates();
    let mats = [&h, &t, &cx];
    let _ = mats;
    let cz = cz_gate();
    let mut reg = OverlayRegister::<C64>::family(side, Order::Hilbert).unwrap();
    reg.set_region_cap(16);
    for s in 0..reg.sites() {
        reg.apply(&h, &[s]).unwrap();
    }
    for (a, b) in patch_edges(side, 4) {
        reg.apply(&cz, &[a, b]).unwrap();
    }
    let l = reg.ledger();
    assert!(l.gates() > 0 && l.merge_count() > 0);
    assert_eq!(
        l.merge_count() + l.local_gates(),
        l.gates(),
        "every gate either migrated or was already local"
    );
    assert!(l.peak_width() >= 2 && l.peak_width() <= reg.region_cap());
    assert!(l.decided() <= l.merge_count());
    assert!(
        l.elections() > 0,
        "the overlay changed the answer somewhere"
    );
}

#[test]
fn padding_is_what_the_layout_charged_beyond_the_gate() {
    let side = 8;
    let (_, _, cx) = gates();
    // Row-major, one vertical edge: two sites demanded, sixteen taken.
    let mut row = OverlayRegister::<C64>::single(side, Order::RowMajor).unwrap();
    row.set_region_cap(16);
    row.apply(&cx, &[0, side]).unwrap();
    let m = row.ledger().merges()[0];
    assert_eq!((m.demand, m.width(), m.padding()), (2, 16, 14));
    // The curve: two demanded, two taken, nothing wasted.
    let mut hil = OverlayRegister::<C64>::single(side, Order::Hilbert).unwrap();
    hil.apply(&cx, &[0, side]).unwrap();
    let m = hil.ledger().merges()[0];
    assert_eq!((m.demand, m.width(), m.padding()), (2, 2, 0));
}

#[test]
fn the_closure_swallows_what_the_block_cuts_into_and_the_split_hands_it_back() {
    // Entangle two neighbours, then reach from that region across to
    // the far corner. The only block holding both is the whole
    // lattice, so the migration swallows all sixteen regions — and
    // then the split gives back the fourteen it never entangled.
    let side = 4;
    let (h, _, cx) = gates();
    let mut reg = OverlayRegister::<C64>::single(side, Order::Hilbert).unwrap();
    reg.apply(&h, &[0]).unwrap();
    reg.apply(&h, &[side * side - 1]).unwrap();
    reg.apply(&cx, &[0, 1]).unwrap();
    assert_eq!(reg.region_count(), 15, "one pair, fourteen singles");
    // The far corner is in |+⟩, which is a cx eigenstate: the gate
    // migrates but changes nothing.
    reg.apply(&cx, &[0, side * side - 1]).unwrap();
    let m = *reg.ledger().merges().last().unwrap();
    assert_eq!(m.placement.level, (side * side).ilog2());
    assert_eq!(m.width(), side * side, "the whole lattice");
    assert_eq!(m.demand, 3, "the gate asked for a pair and a single");
    assert_eq!(m.absorbed, 15, "and the block cut into every region");
    assert_eq!(m.padding(), 13);
    assert_eq!(reg.region_count(), 15, "given back");
    assert_eq!(reg.widest(), 2);
    assert!(reg.ledger().reclaimed() > 0 && reg.ledger().splits() > 0);
}

#[test]
fn an_entangled_pair_across_the_block_cut_cannot_be_given_back() {
    // The other side of the same coin, and the reason the election is
    // worth making. Once two sites at opposite ends of a block are
    // genuinely entangled, the block's own bipartition has rank two
    // and the register is stuck holding all of it.
    let side = 4;
    let (h, _, _) = gates();
    let cz = cz_gate();
    let mut reg = OverlayRegister::<C64>::single(side, Order::Hilbert).unwrap();
    reg.apply(&h, &[0]).unwrap();
    reg.apply(&h, &[side * side - 1]).unwrap();
    reg.apply(&cz, &[0, side * side - 1]).unwrap();
    assert_eq!(reg.region_count(), 1, "one region, the whole lattice");
    assert_eq!(reg.widest(), side * side);
    assert_eq!(reg.ledger().splits(), 0, "nothing factors across that cut");
    // Two entangled sites cost a sixteen-site region: the padding is
    // spent, not borrowed.
    assert_eq!(reg.memory_amplitudes(), 1 << 16);
}

#[test]
fn a_register_refuses_a_lattice_it_has_no_tree_for() {
    assert!(OverlayRegister::<C64>::single(6, Order::RowMajor).is_err());
    assert!(OverlayRegister::<C64>::single(0, Order::RowMajor).is_err());
    let elsewhere = Overlay::new(vec![GridOrder::new(4, Order::Hilbert).unwrap()]).unwrap();
    assert!(OverlayRegister::<C64>::new(8, &elsewhere).is_err());
}

#[test]
fn reset_returns_every_site_to_its_own_region() {
    let side = 4;
    let (h, _, cx) = gates();
    let mut reg = OverlayRegister::<C64>::family(side, Order::Hilbert).unwrap();
    reg.apply(&h, &[0]).unwrap();
    reg.apply(&cx, &[0, 1]).unwrap();
    assert!(reg.region_count() < 16);
    reg.reset();
    assert_eq!(reg.region_count(), 16);
    assert_eq!(reg.ledger().merge_count(), 0);
    assert_eq!(reg.amplitude(0), C64::new(1.0, 0.0));
}

// ─────────────── the placement laws, which are exact ───────────────

/// Summed placement width over every lattice edge, on a fresh
/// register: what a nearest-neighbour layer costs *before* any
/// commitment, which is the layout's own number.
fn placement_total(side: usize, ov: &Overlay) -> usize {
    let r = OverlayRegister::<C64>::new(side, ov).unwrap();
    edges(side)
        .iter()
        .map(|&(a, b)| r.admits(&[a, b]).unwrap().width())
        .sum()
}

#[test]
fn per_edge_placement_follows_exact_closed_forms() {
    // Verified at every power-of-two side from 4 to 64.
    for side in [4usize, 8, 16, 32, 64] {
        let lg = side.ilog2() as usize;
        let (s2, n) = (side * side, side * side);
        let one = |o| Overlay::new(vec![GridOrder::new(side, o).unwrap()]).unwrap();
        assert_eq!(
            placement_total(side, &one(Order::RowMajor)),
            s2 * (side + 1) * lg,
            "one row-major ordering at side {side}: s²(s+1)·log₂s"
        );
        assert_eq!(
            placement_total(side, &one(Order::Snake)),
            s2 * (side + 1) * lg,
            "the snake pays exactly what the rows do"
        );
        assert_eq!(
            placement_total(side, &Overlay::family(side, Order::RowMajor).unwrap()),
            2 * s2 * lg,
            "the row-major D₄ family at side {side}: n·log₂n"
        );
        assert_eq!(
            placement_total(side, &one(Order::Hilbert)),
            3 * s2 * (side - 1),
            "one Hilbert ordering at side {side}: 3s²(s−1)"
        );
        assert_eq!(
            placement_total(side, &Overlay::family(side, Order::Hilbert).unwrap()),
            2 * s2 * (side - 1),
            "the Hilbert D₄ family at side {side}: 2s²(s−1)"
        );
        assert_eq!(n.ilog2() as usize, 2 * lg);
    }
}

#[test]
fn the_overlay_saves_a_third_of_the_curve_and_a_factor_of_the_rows() {
    for side in [4usize, 8, 16, 32, 64] {
        let one = |o| Overlay::new(vec![GridOrder::new(side, o).unwrap()]).unwrap();
        // The Hilbert family: exactly two thirds, at every size. Its
        // members are rotors of a self-similar curve, so they agree on
        // every quadrant and can only disagree inside one.
        let (h1, h8) = (
            placement_total(side, &one(Order::Hilbert)),
            placement_total(side, &Overlay::family(side, Order::Hilbert).unwrap()),
        );
        assert_eq!(
            3 * h8,
            2 * h1,
            "side {side}: the curve's family saves a third"
        );
        // The row-major family: a factor of (s+1)/2, which grows
        // without bound. Its members have no quadrants to agree about,
        // and the transpose disagrees with it everywhere.
        let (r1, r8) = (
            placement_total(side, &one(Order::RowMajor)),
            placement_total(side, &Overlay::family(side, Order::RowMajor).unwrap()),
        );
        assert_eq!(2 * r1, (side + 1) * r8, "side {side}: (s+1)/2");
        // So the overlay reverses the ranking. One ordering: the curve
        // wins. A family of them: the rows win, and by more at every
        // size.
        assert!(h1 < r1, "side {side}: one curve beats one row-major");
        assert!(
            r8 < h8,
            "side {side}: the row-major family beats the curve's"
        );
    }
}

#[test]
fn two_members_are_the_whole_row_major_gain() {
    // An ordering and its transpose. Members three through eight add
    // exactly nothing, because a lattice edge is horizontal or
    // vertical and those two members already cover both.
    for side in [4usize, 8, 16, 32] {
        let all = Overlay::family(side, Order::RowMajor).unwrap();
        let full = placement_total(side, &all);
        let by_count: Vec<usize> = (1..=8)
            .map(|k| {
                let sub = Overlay::new(all.members()[..k].to_vec()).unwrap();
                placement_total(side, &sub)
            })
            .collect();
        assert!(by_count[0] > by_count[1], "the second member does the work");
        assert!(
            by_count[1..].iter().all(|&v| v == full),
            "side {side}: saturated at two, {by_count:?}"
        );
    }
}

#[test]
fn the_heterogeneous_overlay_is_strictly_the_best_of_both() {
    // The claim the module exists to make. Neither family does both
    // jobs; putting them in one register does.
    for side in [8usize, 16] {
        let one = |o| Overlay::new(vec![GridOrder::new(side, o).unwrap()]).unwrap();
        let rm = Overlay::family(side, Order::RowMajor).unwrap();
        let hb = Overlay::family(side, Order::Hilbert).unwrap();
        let mix = Overlay::families(side, &[Order::RowMajor, Order::Hilbert]).unwrap();

        // Placement: the mixed overlay ties the row-major family, which
        // is the best there is, and beats the Hilbert family outright.
        let (p_rm, p_hb, p_mix) = (
            placement_total(side, &rm),
            placement_total(side, &hb),
            placement_total(side, &mix),
        );
        assert_eq!(p_mix, p_rm, "side {side}: placement ties the best");
        assert!(p_mix < p_hb, "side {side}: {p_mix} < {p_hb}");

        // A run: a graph state that is row-local on the left half of
        // the lattice and 4×4-patch-local on the right — a circuit
        // with two different localities in it, which is the case no
        // single family serves. The row-major family refuses it, and
        // the mixed overlay, which contains that family, completes it
        // and holds less than anything else that does.
        let (h, _, _) = gates();
        let cz = cz_gate();
        let ops = strips_and_patches(side, 4);
        let run = |ov: &Overlay| {
            let mut r = OverlayRegister::<C64>::new(side, ov).unwrap();
            r.set_region_cap(16);
            for s in 0..r.sites() {
                r.apply(&h, &[s]).unwrap();
            }
            let done = ops
                .iter()
                .take_while(|&&(a, b)| r.apply(&cz, &[a, b]).is_ok())
                .count();
            (done, r.ledger().total_padding(), r.ledger().peak_memory())
        };
        let (d_rm, _, _) = run(&rm);
        let (d_hb, pad_hb, mem_hb) = run(&hb);
        let (d_one, pad_one, mem_one) = run(&one(Order::Hilbert));
        let (d_mix, pad_mix, mem_mix) = run(&mix);
        assert!(d_rm < ops.len(), "side {side}: the rows refuse");
        assert_eq!(d_mix, ops.len(), "side {side}: the mix completes");
        assert_eq!(d_hb, ops.len());
        assert_eq!(d_one, ops.len());
        assert!(
            pad_mix < pad_hb && pad_hb < pad_one,
            "side {side}: padding, mix {pad_mix} < hilbert family {pad_hb} < hilbert {pad_one}"
        );
        assert!(
            mem_mix < mem_hb && mem_hb < mem_one,
            "side {side}: peak memory, mix {mem_mix} < {mem_hb} < {mem_one}"
        );
    }
}

#[test]
fn a_tie_in_the_curves_family_is_only_a_labelling() {
    // Where several Hilbert rotors reach the minimal level they name
    // the *same block*, every time — which is why that family saves
    // only a constant. The row-major family's ties are real: its
    // contenders cover different sites.
    for side in [4usize, 8, 16] {
        let hb = OverlayRegister::<C64>::family(side, Order::Hilbert).unwrap();
        let rm = OverlayRegister::<C64>::family(side, Order::RowMajor).unwrap();
        let mut rm_disagreed = 0usize;
        for a in 0..side * side {
            for b in (a + 1)..side * side {
                let same = |r: &OverlayRegister<C64>| {
                    let mut sets: Vec<Vec<usize>> = r
                        .contenders(&[a, b])
                        .unwrap()
                        .iter()
                        .map(|&p| {
                            let mut v = r.block_sites(p);
                            v.sort_unstable();
                            v
                        })
                        .collect();
                    sets.dedup();
                    sets.len() == 1
                };
                assert!(
                    same(&hb),
                    "side {side}: hilbert ties on ({a},{b}) name one block"
                );
                if !same(&rm) {
                    rm_disagreed += 1;
                }
            }
        }
        assert!(
            rm_disagreed > 0,
            "side {side}: the row-major family's ties are a real choice"
        );
    }
}

#[test]
fn uncomputation_gives_the_whole_lattice_back() {
    // The split is not decoration. A graph-state layer applied twice
    // is the identity, and the register has to notice: every region
    // factors again and the peak is *history*, not present cost.
    let side = 8;
    let (h, _, _) = gates();
    let cz = cz_gate();
    let mut reg = OverlayRegister::<C64>::mixed(side, &[Order::RowMajor, Order::Hilbert]).unwrap();
    reg.set_region_cap(16);
    for s in 0..reg.sites() {
        reg.apply(&h, &[s]).unwrap();
    }
    let ops = patch_edges(side, 4);
    for &(a, b) in &ops {
        reg.apply(&cz, &[a, b]).unwrap();
    }
    assert_eq!(reg.widest(), 16, "one region per 4×4 patch");
    for &(a, b) in ops.iter().rev() {
        reg.apply(&cz, &[a, b]).unwrap();
    }
    assert_eq!(reg.region_count(), reg.sites(), "back to singles");
    assert_eq!(reg.memory_amplitudes(), 2 * reg.sites() as u128);
    assert_eq!(reg.ledger().peak_width(), 16, "the peak is remembered");
    assert!(reg.ledger().reclaimed() > 0);
}

#[test]
fn splitting_can_be_turned_off_and_then_the_register_only_grows() {
    let side = 8;
    let (h, _, _) = gates();
    let cz = cz_gate();
    let mut reg = OverlayRegister::<C64>::single(side, Order::Hilbert).unwrap();
    reg.set_auto_split(false);
    reg.set_region_cap(16);
    for s in 0..reg.sites() {
        reg.apply(&h, &[s]).unwrap();
    }
    let ops = patch_edges(side, 4);
    for &(a, b) in ops.iter().chain(ops.iter().rev()) {
        reg.apply(&cz, &[a, b]).unwrap();
    }
    assert_eq!(reg.ledger().splits(), 0);
    assert_eq!(reg.widest(), 16, "still holding what it no longer needs");
    // And asking for it back works, because the state really is a product.
    assert!(reg.compact() > 0);
    assert_eq!(reg.region_count(), reg.sites());
}

#[test]
fn a_region_cap_is_clamped_to_what_a_usize_index_can_address() {
    let mut reg = OverlayRegister::<C64>::single(16, Order::Hilbert).unwrap();
    reg.set_region_cap(1000);
    assert_eq!(reg.region_cap(), quantsim::overlay::MAX_REGION_SITES);
    // The whole 16×16 lattice is always a block, and naming it must not
    // overflow the amplitude count it would cost.
    let corners = [0usize, 16 * 16 - 1];
    let p = reg.admits(&corners).unwrap();
    assert_eq!(p.width(), 256);
    assert_eq!(p.amplitudes(), u128::MAX, "saturates rather than wraps");
    // And it is refused before anything is allocated.
    let (h, _, _) = gates();
    let cz = cz_gate();
    reg.apply(&h, &[corners[0]]).unwrap();
    reg.apply(&h, &[corners[1]]).unwrap();
    assert!(matches!(
        reg.apply(&cz, &corners),
        Err(Error::TooManyQubits {
            requested: 256,
            max: 63
        })
    ));
}
