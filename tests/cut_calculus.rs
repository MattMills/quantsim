//! The cut calculus: Schmidt rank predicted from the interaction graph
//! alone, what the prediction is worth, and the open conjecture it
//! settles — in the negative.

use quantsim::cut::CutGraph;
use quantsim::support::Support;
use quantsim::prelude::*;

/// Every graph on these arities, and every non-trivial cut.
fn sweep(arities: &[usize], mut f: impl FnMut(&CutGraph, &Support)) {
    let n = arities.len();
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }
    for mask in 0..1u32 << pairs.len() {
        let mut g = CutGraph::new(arities.to_vec()).unwrap();
        for (b, &(i, j)) in pairs.iter().enumerate() {
            if mask >> b & 1 == 1 {
                g.bond(i, j).unwrap();
            }
        }
        for bits in 1..(1u32 << n) - 1 {
            let cut: Support = (0..n).filter(|&i| bits >> i & 1 == 1).collect();
            f(&g, &cut);
        }
    }
}

// ── the bonds themselves ─────────────────────────────────────────────

#[test]
fn a_single_bond_has_rank_min_of_its_arities() {
    for (a, b) in [(2usize, 2usize), (2, 3), (3, 5), (4, 2), (6, 3), (5, 5)] {
        let mut g = CutGraph::new(vec![a, b]).unwrap();
        g.bond(0, 1).unwrap();
        assert_eq!(
            g.exact_rank(&CutGraph::cut_of(&[0])).unwrap(),
            a.min(b),
            "a {a}×{b} bond should carry rank {}",
            a.min(b)
        );
        assert_eq!(g.schur_bound(&CutGraph::cut_of(&[0])), a.min(b) as u128);
    }
}

#[test]
fn only_the_crossing_bonds_matter() {
    // A bond inside one side is a diagonal unitary on that side, and
    // local unitaries never move Schmidt rank. So adding them must
    // change nothing at all.
    let mut bare = CutGraph::new(vec![2, 3, 2, 3]).unwrap();
    bare.bond(0, 2).unwrap().bond(1, 3).unwrap();
    let cut = CutGraph::cut_of(&[0, 1]); // A = {0,1}
    let want = bare.exact_rank(&cut).unwrap();

    let mut padded = bare.clone();
    padded.bond(0, 1).unwrap(); // inside A
    padded.bond(2, 3).unwrap(); // inside B
    assert_eq!(padded.crossing(&cut), bare.crossing(&cut));
    assert_eq!(padded.exact_rank(&cut).unwrap(), want);
    assert_eq!(padded.schur_bound(&cut), bare.schur_bound(&cut));
}

// ── the bounds are sound ─────────────────────────────────────────────

#[test]
fn the_graph_bounds_are_never_violated() {
    // Exhaustive: every graph on four mixed-arity sites, every cut.
    for arities in [
        vec![2usize, 2, 2, 2],
        vec![2, 3, 2, 3],
        vec![6, 2, 3, 2],
        vec![4, 2, 2, 3],
    ] {
        let mut cases = 0usize;
        sweep(&arities, |g, cut| {
            let exact = g.exact_rank(cut).unwrap() as u128;
            assert!(
                exact <= g.schur_bound(cut),
                "{arities:?} bonds {:?} cut {cut:?}: rank {exact} > Schur {}",
                g.bonds(),
                g.schur_bound(cut)
            );
            assert!(
                exact <= g.capacity_bound(cut),
                "{arities:?} bonds {:?} cut {cut:?}: rank {exact} > capacity {}",
                g.bonds(),
                g.capacity_bound(cut)
            );
            assert!(exact <= g.predicted_rank(cut).unwrap());
            cases += 1;
        });
        assert_eq!(cases, 896, "the sweep should be exhaustive");
    }
}

#[test]
fn a_matching_saturates_the_bound() {
    // Vertex-disjoint crossing bonds factorize, so Schur is attained —
    // the network-flow regime.
    for arities in [vec![2usize, 3, 2, 3], vec![6, 2, 3, 2], vec![4, 2, 2, 3]] {
        let mut checked = 0usize;
        sweep(&arities, |g, cut| {
            if !g.is_matching(cut) || g.crossing(cut).is_empty() {
                return;
            }
            assert_eq!(
                g.exact_rank(cut).unwrap() as u128,
                g.schur_bound(cut),
                "{arities:?} bonds {:?} cut {cut:?}: a matching must saturate",
                g.bonds()
            );
            checked += 1;
        });
        assert!(checked > 100, "only {checked} matchings seen");
    }
}

#[test]
fn a_wide_shared_vertex_saturates_and_a_narrow_one_collapses() {
    // The distinction the source theory draws: capacity, not
    // disjointness, is the constraint.
    let mut wide = CutGraph::new(vec![6, 2, 3]).unwrap();
    wide.bond(0, 1).unwrap().bond(0, 2).unwrap();
    assert!(!wide.is_matching(&CutGraph::cut_of(&[0])), "the bonds share site 0");
    assert_eq!(
        wide.exact_rank(&CutGraph::cut_of(&[0])).unwrap(),
        6,
        "an arity-6 hub carries both a 2-bond and a 3-bond"
    );
    assert_eq!(wide.schur_bound(&CutGraph::cut_of(&[0])), 6);

    // The qubit 4-cycle, fully crossed: every bound says 4, the truth
    // is 3, because the two A-side digits reach B only through their
    // sum.
    let mut cycle = CutGraph::uniform(4, 2).unwrap();
    cycle
        .bond(0, 1)
        .unwrap()
        .bond(1, 2)
        .unwrap()
        .bond(2, 3)
        .unwrap()
        .bond(3, 0)
        .unwrap();
    let cut = CutGraph::cut_of(&[0, 2]); // A = {0,2}
    assert_eq!(cycle.crossing(&cut).len(), 4, "all four bonds cross");
    assert_eq!(cycle.exact_rank(&cut).unwrap(), 3);
    assert_eq!(cycle.capacity_bound(&cut), 4);
    assert_eq!(
        cycle.character_count(&cut).unwrap(),
        3,
        "the frequencies see only d0+d2"
    );
}

// ── the open conjecture, settled ─────────────────────────────────────

#[test]
fn the_character_count_formula_is_false() {
    // The source theory leaves this open: "whether rank = |{θ(d_A)}|
    // holds in full generality is a named develop item". It does not,
    // and the smallest counterexample is four qubits with two bonds
    // sharing a vertex.
    //
    // A = {1,2}, B = {0,3}. Only site 0 receives, so every row is
    // (1, exp(2πi·θ₀)) — three distinct frequencies, but three vectors
    // in a two-dimensional column space cannot be independent.
    let mut g = CutGraph::uniform(4, 2).unwrap();
    g.bond(0, 1).unwrap().bond(0, 2).unwrap();
    let cut = CutGraph::cut_of(&[1, 2]);
    assert_eq!(g.character_count(&cut).unwrap(), 3);
    assert_eq!(g.exact_rank(&cut).unwrap(), 2, "the count over-states the rank");

    // Counting on the other side repairs this one — but not in general.
    // Six distinct rows and six distinct columns, rank 4.
    let mut h = CutGraph::new(vec![2, 3, 2, 3]).unwrap();
    h.bond(0, 2).unwrap().bond(0, 3).unwrap().bond(1, 2).unwrap();
    let cut = CutGraph::cut_of(&[0, 1]);
    assert_eq!(h.character_count(&cut).unwrap(), 6);
    assert_eq!(h.character_count(&CutGraph::cut_of(&[2, 3])).unwrap(), 6);
    assert_eq!(h.predicted_rank(&cut).unwrap(), 6);
    assert_eq!(
        h.exact_rank(&cut).unwrap(),
        4,
        "neither side's character count reaches the rank"
    );
}

#[test]
fn the_prediction_is_sound_everywhere_and_exact_on_most_of_the_sweep() {
    // What the cheap bound is actually worth, stated as a rate rather
    // than a slogan — and asserted from below so a regression shows up.
    let mut cases = 0usize;
    let mut tight = 0usize;
    for arities in [vec![2usize, 2, 2, 2], vec![2, 3, 2, 3], vec![6, 2, 3, 2]] {
        sweep(&arities, |g, cut| {
            let exact = g.exact_rank(cut).unwrap() as u128;
            let predicted = g.predicted_rank(cut).unwrap();
            assert!(exact <= predicted, "the prediction must never under-state");
            cases += 1;
            if exact == predicted {
                tight += 1;
            }
        });
    }
    let rate = tight as f64 / cases as f64;
    assert!(
        rate > 0.95,
        "the prediction was tight on only {:.1}% of {cases} cases",
        100.0 * rate
    );
    assert!(
        rate < 1.0,
        "if it were exact everywhere the conjecture would hold, and it does not"
    );
}

// ── why a shared vertex sometimes saturates ──────────────────────────

#[test]
fn a_shared_vertex_saturates_when_its_channel_is_injective() {
    // Two bonds into one hub reach it only through d₀/a₀ + d₁/a₁. That
    // map is injective — so nothing collapses — when the denominators
    // are coprime (Chinese remainder) *or* when they nest positionally.
    // It fails on a divisibility chain, where the digits add.
    for arities in [
        vec![2usize, 3, 5], // coprime: CRT injective
        vec![3, 5, 7],      // coprime
        vec![2, 4, 6],      // not coprime, but d₀/2 + d₁/4 is positional
        vec![4, 4, 4],      // the far side binds first
    ] {
        let mut loose = 0usize;
        sweep(&arities, |g, cut| {
            if g.exact_rank(cut).unwrap() as u128 != g.schur_bound(cut) {
                loose += 1;
            }
        });
        assert_eq!(loose, 0, "{arities:?} should attain Schur at every cut");
    }

    // And the chains, where it does collapse.
    for (arities, want) in [
        (vec![2usize, 2, 4], 3usize),
        (vec![2, 4, 8], 6),
        (vec![3, 9, 27], 15),
    ] {
        let mut g = CutGraph::new(arities.clone()).unwrap();
        g.bond(0, 2).unwrap().bond(1, 2).unwrap();
        let cut = CutGraph::cut_of(&[0, 1]); // both feeding the wide hub
        assert_eq!(
            g.exact_rank(&cut).unwrap(),
            want,
            "{arities:?}: a divisibility chain should collapse"
        );
        assert!(
            (g.exact_rank(&cut).unwrap() as u128) < g.schur_bound(&cut),
            "{arities:?}: and fall short of Schur"
        );
    }
}

// ── the topology of onset ────────────────────────────────────────────

#[test]
fn entanglement_onset_is_graph_connectivity() {
    // The source theory's first law: the sides entangle exactly when
    // the graph joins them.
    let mut g = CutGraph::new(vec![2, 3, 5, 3]).unwrap();
    g.bond(0, 1).unwrap().bond(2, 3).unwrap();
    let cut = CutGraph::cut_of(&[0, 1]); // A = {0,1} against B = {2,3}
    assert!(!g.is_connected());
    assert_eq!(g.crossing(&cut).len(), 0);
    assert_eq!(g.exact_rank(&cut).unwrap(), 1, "disconnected means product");

    g.bond(1, 2).unwrap();
    assert!(g.is_connected());
    assert!(g.exact_rank(&cut).unwrap() > 1, "one junction entangles");
    assert_eq!(g.betti(), 0, "a tree carries no cycle");

    g.bond(0, 3).unwrap();
    assert_eq!(g.betti(), 1, "the second junction closes the loop");
}

#[test]
fn the_betti_number_counts_independent_cycles() {
    let mut path = CutGraph::uniform(4, 2).unwrap();
    path.bond(0, 1).unwrap().bond(1, 2).unwrap().bond(2, 3).unwrap();
    assert_eq!(path.betti(), 0);

    let mut triangle = CutGraph::uniform(3, 2).unwrap();
    triangle.bond(0, 1).unwrap().bond(1, 2).unwrap().bond(2, 0).unwrap();
    assert_eq!(triangle.betti(), 1);

    let mut theta = CutGraph::uniform(4, 2).unwrap();
    theta
        .bond(0, 1)
        .unwrap()
        .bond(1, 2)
        .unwrap()
        .bond(2, 3)
        .unwrap()
        .bond(3, 0)
        .unwrap()
        .bond(0, 2)
        .unwrap();
    assert_eq!(theta.betti(), 2);
}

// ── provisioning, and refusals ───────────────────────────────────────

#[test]
fn provisioning_meets_a_demand_or_says_it_cannot() {
    let arities = [6usize, 5, 4, 3, 2];
    let bought = CutGraph::provision(&arities, 12).unwrap();
    let mut g = CutGraph::new(arities.to_vec()).unwrap();
    for &(i, j) in &bought {
        g.bond(i, j).unwrap();
    }
    // the bought bonds form a matching, so Schur is attained
    let cut: Support = bought.iter().map(|&(i, _)| i).collect();
    assert!(g.is_matching(&cut));
    assert!(g.schur_bound(&cut) >= 12);

    // and a demand past what a matching can carry is refused, not
    // silently under-served
    match CutGraph::provision(&[2, 2, 2, 2], 1000) {
        Err(Error::InvalidState(msg)) => assert!(msg.contains("short of the demand"), "{msg}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_malformed_graph_is_refused() {
    assert!(CutGraph::new(vec![2, 1, 3]).is_err(), "arity 1 is not a qudit");
    let mut g = CutGraph::uniform(3, 2).unwrap();
    assert!(g.bond(0, 0).is_err(), "a self-bond is a local phase");
    g.bond(0, 1).unwrap();
    assert!(g.bond(1, 0).is_err(), "a duplicate bond is a different strength");
    assert!(g.bond(0, 9).is_err());
}

#[test]
fn the_predictor_costs_the_graph_and_the_verifier_costs_the_cut() {
    // The whole point: the bound is O(|E|) and knows nothing about
    // dimension, so it prices a register no state could represent.
    let n = 400usize;
    let mut g = CutGraph::uniform(n, 3).unwrap();
    for q in 0..n - 1 {
        g.bond(q, q + 1).unwrap();
    }
    let cut: Support = (0..32).collect(); // the first 32 sites against the rest
    assert_eq!(g.crossing(&cut).len(), 1, "a chain cut crosses one bond");
    assert_eq!(g.schur_bound(&cut), 3, "and carries rank 3, at any width");
    assert_eq!(g.betti(), 0);
    // the verifier, by contrast, refuses — as it should
    assert!(g.exact_rank(&cut).is_err());
}
