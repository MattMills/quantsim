//! Space-filling curve orderings: the closed forms, and the refutation
//! of the expectation that motivated the module.

use quantsim::curve::*;
use quantsim::polar::PolarSpace;

const ORDERS: [Order; 3] = [Order::RowMajor, Order::Snake, Order::Hilbert];

#[test]
fn the_orderings_are_bijections_and_the_curves_are_continuous() {
    for side in [2usize, 4, 8, 16] {
        for order in ORDERS {
            let g = GridOrder::new(side, order).unwrap();
            let mut seen = vec![false; g.sites()];
            for i in 0..g.sites() {
                let (x, y) = g.lattice_point(i).unwrap();
                assert_eq!(g.chain_index(x, y).unwrap(), i, "{order:?} round-trips");
                assert!(!seen[y * side + x], "{order:?} hits {x},{y} twice");
                seen[y * side + x] = true;
            }
            assert!(seen.iter().all(|&b| b), "{order:?} covers the lattice");
            // Snake and Hilbert are curves; row-major is not.
            assert_eq!(
                g.is_continuous(),
                order != Order::RowMajor,
                "{order:?} continuity at side {side}"
            );
        }
    }
    assert!(
        GridOrder::new(6, Order::Hilbert).is_err(),
        "needs a power of two"
    );
    assert!(GridOrder::new(6, Order::Snake).is_ok());
    assert!(GridOrder::new(0, Order::RowMajor).is_err());
}

#[test]
fn the_closed_forms_hold_at_every_size() {
    // Every law pinned against enumeration from side 2 to 128.
    for k in 1..=7u32 {
        let side = 1usize << k;
        for order in ORDERS {
            let g = GridOrder::new(side, order).unwrap();
            assert_eq!(
                g.dilation().max,
                order.max_dilation_law(side).unwrap(),
                "{order:?} max dilation at side {side}"
            );
            if let Some(want) = order.cutwidth_law(side) {
                assert_eq!(g.cutwidth(), want, "{order:?} cutwidth at side {side}");
            }
        }
    }
    // The specific laws, stated.
    assert_eq!(Order::RowMajor.max_dilation_law(32), Some(32));
    assert_eq!(Order::Snake.max_dilation_law(32), Some(63));
    assert_eq!(Order::Hilbert.max_dilation_law(32), Some(853));
    assert_eq!(
        Order::Hilbert.max_dilation_law(6),
        None,
        "power of two only"
    );
}

#[test]
fn the_hilbert_curve_is_worse_on_every_measure_the_chain_cares_about() {
    // The refutation. The expectation was that a locality-preserving
    // curve would lay a lattice onto a chain better than reading rows.
    // It does not, and the gap widens with size.
    let mut cut_ratio = Vec::new();
    for side in [8usize, 16, 32, 64] {
        let row = GridOrder::new(side, Order::RowMajor).unwrap();
        let hil = GridOrder::new(side, Order::Hilbert).unwrap();

        assert!(hil.cutwidth() > row.cutwidth(), "side {side}: cutwidth");
        assert!(
            hil.total_crossings() > row.total_crossings(),
            "side {side}: bonds"
        );
        assert!(
            hil.dilation().routing_cost() > row.dilation().routing_cost(),
            "side {side}: routing"
        );
        assert!(
            hil.dilation().max > row.dilation().max,
            "side {side}: worst edge"
        );
        cut_ratio.push(hil.cutwidth() as f64 / row.cutwidth() as f64);
    }
    // And the cutwidth ratio converges to exactly 2 from below.
    assert!(cut_ratio.windows(2).all(|w| w[1] > w[0]), "{cut_ratio:?}");
    assert!(*cut_ratio.last().unwrap() > 1.9, "{cut_ratio:?}");
    assert!(cut_ratio.iter().all(|&r| r < 2.0), "{cut_ratio:?}");
}

#[test]
fn the_half_of_the_expectation_that_survived() {
    // Cutwidth is Θ(side) under every ordering — a property of the
    // grid, not of the curve — so no relabelling beats it
    // asymptotically. That half was right.
    for side in [8usize, 16, 32, 64] {
        for order in ORDERS {
            let g = GridOrder::new(side, order).unwrap();
            assert!(
                g.cutwidth() >= side,
                "{order:?} at {side}: below the grid's own bound"
            );
            assert!(
                g.cutwidth() <= 2 * side,
                "{order:?} at {side}: within a factor of 2"
            );
        }
    }
}

#[test]
fn the_snake_is_the_one_that_gets_both_halves() {
    // Continuous like the curve, and cheap like the rows: the snake
    // matches row-major's cutwidth exactly while being an actual
    // space-filling ordering.
    for side in [8usize, 16, 32, 64] {
        let row = GridOrder::new(side, Order::RowMajor).unwrap();
        let snake = GridOrder::new(side, Order::Snake).unwrap();
        assert_eq!(snake.cutwidth(), row.cutwidth());
        assert_eq!(snake.total_crossings(), row.total_crossings());
        assert_eq!(
            snake.dilation().routing_cost(),
            row.dilation().routing_cost()
        );
        assert!(snake.is_continuous() && !row.is_continuous());
        // It buys the adjacency the curve was wanted for, at no cost.
        assert!(snake.dilation().adjacent_fraction() > row.dilation().adjacent_fraction());
    }
}

// ─────────── keeping the recursion instead of flattening it ───────────

#[test]
fn the_rotor_is_a_dihedral_group_and_the_hilbert_rule_is_four_of_them() {
    let all = Rotor::all();
    assert_eq!(all.len(), 8, "D₄");
    // Every rotor is a bijection of the cell, and reflections are
    // exactly the transposing half.
    for r in &all {
        let mut seen = [false; 16];
        for x in 0..4 {
            for y in 0..4 {
                let (a, b) = r.apply(4, x, y);
                assert!(a < 4 && b < 4);
                assert!(!seen[b * 4 + a], "{r:?} is not injective");
                seen[b * 4 + a] = true;
            }
        }
        assert_eq!(r.is_reflection(), r.transpose);
    }
    assert_eq!(all.iter().filter(|r| r.is_reflection()).count(), 4);

    // The Hilbert rule: transpose, identity, identity, anti-transpose.
    let h = hilbert_rotors();
    assert_eq!(h.len(), 4);
    assert_eq!(h[0], Rotor::new(true, 0));
    assert_eq!(h[1], Rotor::IDENTITY);
    assert_eq!(h[2], Rotor::IDENTITY);
    assert_eq!(h[3], Rotor::new(true, 2));
    assert_eq!(
        h.iter().filter(|r| r.is_reflection()).count(),
        2,
        "two reflections and two rotations is what joins the sub-curves"
    );
    // Identity acts as the identity; four quarter turns is the identity.
    assert_eq!(Rotor::IDENTITY.apply(8, 3, 5), (3, 5));
    assert_eq!(Rotor::new(false, 4), Rotor::IDENTITY);
}

#[test]
fn the_recursion_matches_the_chain_on_the_worst_cut() {
    // The grid's own bound. No layout of any kind beats it, and the
    // tree does not either — this is the half of the earlier
    // expectation that survived, confirmed from the other side.
    for side in [8usize, 16, 32, 64, 128] {
        let b = Bisection::quadrants(side).unwrap();
        let row = GridOrder::new(side, Order::RowMajor).unwrap();
        assert_eq!(b.max_separator(), side);
        assert!(
            b.max_separator() <= row.cutwidth(),
            "side {side}: the tree never exceeds the chain's worst cut"
        );
        assert!(row.cutwidth() - b.max_separator() <= 1);
    }
}

#[test]
fn the_recursion_beats_every_chain_on_the_total_and_the_gap_grows() {
    // Both totals have exact closed forms, so the ratio does too:
    //   chain (best ordering) = side³ − side      = Θ(N^{3/2})
    //   tree                  = 2·side² − 2·side  = Θ(N)
    //   ratio                 = (side + 1) / 2    = Θ(√N)
    for side in [8usize, 16, 32, 64, 128] {
        let tree = Bisection::quadrants(side).unwrap().total_separator();
        // Against the *best* chain ordering, not a convenient one.
        let chain = ORDERS
            .iter()
            .map(|&o| GridOrder::new(side, o).unwrap().total_crossings())
            .min()
            .unwrap();
        assert_eq!(chain, side * side * side - side, "chain law at {side}");
        assert_eq!(tree, 2 * side * side - 2 * side, "tree law at {side}");
        assert_eq!(2 * chain, (side + 1) * tree, "the ratio is (side+1)/2");
        assert!(tree < chain);
    }
    // 64.5x at side 128, and unbounded in the side.
    let tree = Bisection::quadrants(128).unwrap().total_separator();
    let chain = GridOrder::new(128, Order::RowMajor)
        .unwrap()
        .total_crossings();
    assert!(chain as f64 / tree as f64 > 64.0);
}

#[test]
fn the_tree_is_shallow_and_its_cost_sits_at_the_leaves() {
    // One cut per internal node, log₂N levels — and the profile is
    // *not* flat: it doubles every two levels, so the total is
    // dominated by the deepest cuts, not by the root.
    for side in [8usize, 16, 32, 64, 128] {
        let b = Bisection::quadrants(side).unwrap();
        assert_eq!(b.nodes(), side * side - 1, "one cut per internal node");
        assert_eq!(
            b.depth() + 1,
            (side * side).ilog2() as usize,
            "log₂N levels"
        );
        let profile = b.profile();
        for (level, &p) in profile.iter().enumerate() {
            assert_eq!(
                p,
                side * (1 << (level / 2)),
                "side {side} level {level}: the profile law"
            );
        }
        assert_eq!(profile[0], side, "the root cut is the grid's own bound");
        assert!(
            *profile.last().unwrap() > profile[0],
            "the leaves dominate, not the root"
        );
    }
}

// ───────────────── the census, from two directions ─────────────────

#[test]
fn the_polar_geometry_counts_the_stabilizer_states() {
    // `generators × 2^n` — a maximal commuting set is a basis, and each
    // basis carries 2^n states. The sequence is the one a sibling
    // program's census arrives at independently.
    for (n, want) in [
        (1usize, 6u128),
        (2, 60),
        (3, 1080),
        (4, 36_720),
        (5, 2_423_520),
    ] {
        let w = PolarSpace::new(n).unwrap();
        assert_eq!(w.stabilizer_states(), want, "n = {n}");
        assert_eq!(w.stabilizer_states(), w.generators() * (1u128 << n));
    }
}

#[test]
fn the_symplectic_order_has_an_exactly_square_dyadic_part() {
    // v₂|Sp(2n,2)| = n², with second differences identically 2 — one of
    // the quadratic-island laws, here as a property of the polar
    // space's own symmetry group.
    let mut valuations = Vec::new();
    for n in 1..=8usize {
        let w = PolarSpace::new(n).unwrap();
        assert_eq!(w.symplectic_dyadic_valuation(), n * n);
        valuations.push((n * n) as i64);
        if n <= 5 {
            // The stated order, factored: 2^{n²} times an odd number.
            let ord = w.symplectic_order();
            assert_eq!(ord.trailing_zeros() as usize, n * n, "n = {n}");
        }
    }
    let d1: Vec<i64> = valuations.windows(2).map(|w| w[1] - w[0]).collect();
    let d2: Vec<i64> = d1.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(d2.iter().all(|&d| d == 2), "second differences: {d2:?}");
    // |Sp(2,2)| = 6, |Sp(4,2)| = 720, |Sp(6,2)| = 1451520.
    assert_eq!(PolarSpace::new(1).unwrap().symplectic_order(), 6);
    assert_eq!(PolarSpace::new(2).unwrap().symplectic_order(), 720);
    assert_eq!(PolarSpace::new(3).unwrap().symplectic_order(), 1_451_520);
}

// ──────── where the rotor earns its place: the tree's own blocks ────────

#[test]
fn the_curve_wins_where_the_chain_lost_it() {
    // A balanced tree over a chain has the contiguous 2^k blocks as its
    // subtrees, and a hierarchy's cost at a subtree is that block's
    // boundary. On exactly the measure a chain does not care about,
    // the ordering that lost every chain metric wins every block size.
    for side in [8usize, 16, 32] {
        let row = GridOrder::new(side, Order::RowMajor).unwrap();
        let hil = GridOrder::new(side, Order::Hilbert).unwrap();
        let rb = row.block_boundaries();
        let hb = hil.block_boundaries();
        assert_eq!(rb.len(), hb.len());
        assert!(!rb.is_empty());
        for (r, h) in rb.iter().zip(hb.iter()) {
            assert_eq!(r.size, h.size);
            assert!(
                h.max <= r.max,
                "side {side} block {}: hilbert max {} vs row {}",
                r.size,
                h.max,
                r.max
            );
            assert!(
                h.mean() <= r.mean(),
                "side {side} block {}: hilbert mean {} vs row {}",
                r.size,
                h.mean(),
                r.mean()
            );
        }
        // And strictly better somewhere, by a real margin.
        assert!(
            hb.iter().zip(rb.iter()).any(|(h, r)| 2 * h.max <= r.max),
            "side {side}: at least a 2x on some block size"
        );
    }
}

#[test]
fn the_row_orders_blocks_are_strips_and_that_is_the_whole_reason() {
    // A row-major block of exactly `side` positions is one entire row,
    // so its boundary is 2·side — the two long edges. The curve's block
    // of the same size is a compact region and is bounded well below
    // that. This is the mechanism, isolated.
    for side in [8usize, 16, 32] {
        let k = side.trailing_zeros();
        let row = GridOrder::new(side, Order::RowMajor)
            .unwrap()
            .block_boundary(k);
        let hil = GridOrder::new(side, Order::Hilbert)
            .unwrap()
            .block_boundary(k);
        assert_eq!(row.size, side);
        // Interior rows have both long edges; the two outer rows have one.
        assert_eq!(row.max, 2 * side, "a row-major block IS a row");
        assert!(
            hil.max < row.max,
            "side {side}: compact {} vs strip {}",
            hil.max,
            row.max
        );
    }
}

#[test]
fn the_bisection_separators_do_not_depend_on_the_rotor() {
    // The tension the block measurement resolves: the quadrant tree's
    // numbers are a property of the grid, so the rotor contributes
    // nothing to them. It contributes to which *linear* blocks are
    // those quadrants, which is the block measurement above.
    for side in [8usize, 16, 32] {
        let b = Bisection::quadrants(side).unwrap();
        assert_eq!(b.max_separator(), side);
        assert_eq!(b.total_separator(), 2 * side * side - 2 * side);
    }
    // Stated as the reason both results can hold at once: the curve
    // loses the chain metrics and wins the block metric.
    let side = 32;
    let row = GridOrder::new(side, Order::RowMajor).unwrap();
    let hil = GridOrder::new(side, Order::Hilbert).unwrap();
    assert!(hil.cutwidth() > row.cutwidth(), "loses the chain");
    let k = side.trailing_zeros();
    assert!(
        hil.block_boundary(k).max < row.block_boundary(k).max,
        "wins the tree"
    );
}
