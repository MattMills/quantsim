//! The residual as a mosaic: component factoring, canonical-form
//! merges, branch-and-reduce, and a branching election that is measured
//! rather than assumed.
//!
//! The growth laws here are properties of the *circuit*, not of the
//! solver, and the suite measures the ones that collapse and the one
//! that does not.

use quantsim::pathsum::{PathSum, Pivot};
use quantsim::rng::Prng;

const BUDGET: u64 = 5_000_000;
const POLICIES: [Pivot; 4] = [
    Pivot::First,
    Pivot::MaxDegree,
    Pivot::MinRemainder,
    Pivot::Elected,
];

/// `(HT)^k H` on one qubit — the thinnest possible magic.
fn ht_chain(k: usize) -> PathSum {
    let mut p = PathSum::new(1);
    for _ in 0..k {
        p.h(0).unwrap();
        p.t(0).unwrap();
    }
    p.h(0).unwrap();
    p.reduce();
    p
}

/// `n` qubits in a line, `layers` × (H all, chain CZ, T all).
fn chain_family(n: usize, layers: usize) -> PathSum {
    let mut p = PathSum::new(n);
    for _ in 0..layers {
        for q in 0..n {
            p.h(q).unwrap();
        }
        for q in 0..n.saturating_sub(1) {
            p.cz(q, q + 1).unwrap();
        }
        for q in 0..n {
            p.t(q).unwrap();
        }
    }
    p.reduce();
    p
}

/// `side × side` grid: the same qubit count, T-count and depth as
/// `chain_family(side*side, layers)`, coupled in two dimensions.
fn grid_family(side: usize, layers: usize) -> PathSum {
    let n = side * side;
    let mut p = PathSum::new(n);
    for _ in 0..layers {
        for q in 0..n {
            p.h(q).unwrap();
        }
        for r in 0..side {
            for c in 0..side {
                let q = r * side + c;
                if c + 1 < side {
                    p.cz(q, q + 1).unwrap();
                }
                if r + 1 < side {
                    p.cz(q, q + side).unwrap();
                }
            }
        }
        for q in 0..n {
            p.t(q).unwrap();
        }
    }
    p.reduce();
    p
}

// ───────────────────────── the answer is the answer ─────────────────────────

#[test]
fn merged_equals_enumerated_on_random_circuits() {
    // The merged route is a different way to the same number, and the
    // deviation is float summation order, not approximation.
    let mut rng = Prng::new(0x9E3779B9);
    let mut worst = 0.0f64;
    for _ in 0..200 {
        let n = 2 + (rng.next_u64() % 3) as usize;
        let mut p = PathSum::new(n);
        for _ in 0..(6 + rng.next_u64() % 10) {
            let q = (rng.next_u64() as usize) % n;
            let r = (rng.next_u64() as usize) % n;
            match rng.next_u64() % 5 {
                0 => p.h(q).unwrap(),
                1 => p.t(q).unwrap(),
                2 => p.s(q).unwrap(),
                3 => {
                    if q != r {
                        p.cz(q, r).unwrap()
                    }
                }
                _ => {
                    if q != r {
                        p.cnot(q, r).unwrap()
                    }
                }
            }
        }
        p.reduce();
        for b in 0..(1u64 << n) {
            let want = p.amplitude(b);
            for pv in POLICIES {
                let (got, _) = p.amplitude_merged_with(b, BUDGET, pv).unwrap();
                worst = worst.max((want - got).norm());
                assert!(
                    (want - got).norm() < 1e-9,
                    "{pv:?} disagrees on bits {b}: {want:?} vs {got:?}"
                );
            }
        }
    }
    assert!(worst < 1e-12, "worst deviation {worst:e}");
}

#[test]
fn a_clifford_circuit_never_reaches_the_solver() {
    // The rewrite rules are complete on the Clifford fragment, so the
    // residual is empty and the merged route is one leaf. Whatever the
    // solver could add to Gottesman–Knill, it is not needed here.
    let mut p = PathSum::new(8);
    p.h(0).unwrap();
    for q in 0..7 {
        p.cnot(q, q + 1).unwrap();
    }
    for q in 0..8 {
        p.s(q).unwrap();
    }
    p.reduce();
    assert_eq!(p.internal_vars(), 0);
    let (v, st) = p.amplitude_merged(0, BUDGET).unwrap();
    assert_eq!(st.nodes, 1);
    assert_eq!(st.leaves, 1);
    assert_eq!(st.distinct_forms, 0, "nothing was branched");
    assert!((v - p.amplitude(0)).norm() < 1e-12);
}

#[test]
fn the_budget_is_a_named_refusal() {
    let p = chain_family(8, 8);
    let err = p.amplitude_merged(0, 50).unwrap_err().to_string();
    assert!(err.contains("node budget"), "{err}");
    assert!(err.contains("refuses rather than"), "{err}");
}

// ──────────────────── what collapses, measured ────────────────────

#[test]
fn thin_magic_is_logarithmic_where_enumeration_is_exponential() {
    // `(HT)^k H` has h* = k, so enumeration pays 2^k. Cutting the
    // residual path in the middle leaves two halves that factor *and*
    // memoize against each other, and the node count is logarithmic.
    let mut last = 0u64;
    for k in [8usize, 16, 32, 64] {
        let p = ht_chain(k);
        assert_eq!(p.internal_vars(), k, "h* is the T-count on this family");
        let (v, st) = p.amplitude_merged(0, BUDGET).unwrap();
        assert!(
            st.nodes < 4 * (k as u64).ilog2() as u64 + 40,
            "k = {k}: {} nodes is not logarithmic",
            st.nodes
        );
        assert!(st.memo_hits > 0, "k = {k}: the merges are what pay for it");
        assert!(st.nodes > last, "monotone in k");
        last = st.nodes;
        if k <= 20 {
            assert!((v - p.amplitude(0)).norm() < 1e-9);
        }
    }
    // The headline: h* = 64 is 1.8e19 by enumeration.
    let (_, st) = ht_chain(64).amplitude_merged(0, BUDGET).unwrap();
    assert!(st.nodes < 100, "{} nodes at h* = 64", st.nodes);
}

#[test]
fn width_is_innocent_and_the_coupling_geometry_is_not() {
    // Same qubit count, same T-count, same depth; only the coupling
    // differs. The 1D twin stays linear while the 2D one pulls away,
    // and the gap widens with size — the obstruction is separator
    // growth, not register size.
    let mut ratios = Vec::new();
    for side in [3usize, 4, 5, 6] {
        let n = side * side;
        let line = chain_family(n, 2);
        let grid = grid_family(side, 2);
        assert_eq!(line.internal_vars(), grid.internal_vars(), "same h*");
        let (_, ls) = line.amplitude_merged(0, BUDGET).unwrap();
        let (_, gs) = grid.amplitude_merged(0, BUDGET).unwrap();
        assert!(
            ls.nodes < 4 * n as u64,
            "1D at {n} qubits: {} nodes is not linear",
            ls.nodes
        );
        ratios.push(gs.nodes as f64 / ls.nodes as f64);
    }
    assert!(
        ratios.windows(2).all(|w| w[1] > w[0]),
        "the 2D/1D gap must widen: {ratios:?}"
    );
    assert!(*ratios.last().unwrap() > 10.0, "{ratios:?}");
}

#[test]
fn the_square_family_grows_in_the_square_root_of_the_t_count() {
    // n qubits, n layers, t = n², treewidth ~ n = √t. Enumeration pays
    // 2^{h*} with h* = n(n−1); the solver pays 2^{c·n}, and c settles.
    let mut slopes = Vec::new();
    for n in 5..=8usize {
        let p = chain_family(n, n);
        assert_eq!(p.internal_vars(), n * (n - 1));
        let (_, st) = p.amplitude_merged(0, BUDGET).unwrap();
        let slope = (st.nodes as f64).log2() / n as f64;
        slopes.push(slope);
        assert!(
            (st.nodes as f64).log2() < 0.35 * p.internal_vars() as f64,
            "n = {n}: {} nodes against 2^{}",
            st.nodes,
            p.internal_vars()
        );
    }
    // log2(nodes)/n settles rather than growing: the law is 2^{c√t}.
    let spread = slopes.iter().fold(f64::MIN, |a, &b| a.max(b))
        - slopes.iter().fold(f64::MAX, |a, &b| a.min(b));
    assert!(spread < 0.35, "slopes {slopes:?} have not settled");
    assert!(slopes.iter().all(|&s| s > 1.0 && s < 2.0), "{slopes:?}");
}

// ────────────────────────── the election ──────────────────────────

#[test]
fn no_single_signal_wins_and_the_composed_election_does() {
    // The mosaic's thesis on the residual's axis: connectivity is the
    // whole game on a chain and flat on a shallow grid, degree is the
    // reverse, and composing them beats both parents in both regimes.
    let nodes = |p: &PathSum, pv: Pivot| p.amplitude_merged_with(0, BUDGET, pv).unwrap().1.nodes;

    let chain = ht_chain(32);
    assert!(
        nodes(&chain, Pivot::MinRemainder) < nodes(&chain, Pivot::MaxDegree),
        "on a chain the connectivity signal wins"
    );

    let grid = grid_family(5, 2);
    assert!(
        nodes(&grid, Pivot::MaxDegree) < nodes(&grid, Pivot::MinRemainder),
        "on a shallow grid the degree signal wins"
    );

    // And the election is at least as good as both, everywhere.
    for p in [ht_chain(32), grid_family(5, 2), chain_family(7, 7)] {
        let e = nodes(&p, Pivot::Elected);
        for pv in [Pivot::First, Pivot::MaxDegree, Pivot::MinRemainder] {
            assert!(
                e <= nodes(&p, pv),
                "Elected ({e}) lost to {pv:?} ({})",
                nodes(&p, pv)
            );
        }
    }
    // Strictly better than both parents where they disagree most.
    let sq = chain_family(8, 8);
    let e = nodes(&sq, Pivot::Elected);
    assert!(e * 2 < nodes(&sq, Pivot::MinRemainder), "{e}");
    assert!(e * 20 < nodes(&sq, Pivot::MaxDegree), "{e}");
}

#[test]
fn every_policy_returns_the_same_amplitude() {
    // The election is a cost decision and never a numerical one.
    for p in [ht_chain(12), grid_family(3, 2), chain_family(5, 5)] {
        let want = p.amplitude(0);
        for pv in POLICIES {
            let (got, _) = p.amplitude_merged_with(0, BUDGET, pv).unwrap();
            assert!((want - got).norm() < 1e-9, "{pv:?}");
        }
    }
}

#[test]
fn the_ledger_reports_where_the_work_went() {
    let p = chain_family(6, 6);
    let (_, st) = p.amplitude_merged(0, BUDGET).unwrap();
    assert!(st.nodes > 0 && st.leaves > 0);
    assert!(st.memo_hits > 0, "merges happened");
    assert!(st.distinct_forms > 0 && st.distinct_forms < st.nodes);
    assert!(st.max_depth > 0);
    // Distinct forms, not 2^{h*}, is what the route paid.
    assert!((st.distinct_forms as f64).log2() < p.internal_vars() as f64);
}
