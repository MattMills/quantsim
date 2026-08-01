//! Supports as a shared graph: that the algebra is the crate's own
//! algebra with the width ceiling removed, that the sharing is real
//! rather than asserted, and that the Heisenberg walk built on it
//! reproduces the bounded walk exactly and then keeps going.

use std::collections::HashSet;

use quantsim::heisenberg::{self, commutes, pauli_mul, tfim_trotter, PauliKey, PauliSum, Rotation};
use quantsim::prelude::*;
use quantsim::support::bench::{shared_nodes, unshared_nodes};
use quantsim::support::{
    clear_memo, propagate_wide, Support, WideConfig, WidePauli, WidePauliSum, WideRotation,
};

// ── the set itself ───────────────────────────────────────────────────

#[test]
fn a_support_is_the_set_of_indices_it_was_given() {
    for idx in &[
        vec![0usize],
        vec![63],
        vec![64],
        vec![65],
        vec![0, 63, 64, 127, 128, 1000],
        vec![5, 5000, 50_000, 500_000],
    ] {
        let s: Support = idx.iter().copied().collect();
        assert_eq!(s.weight(), idx.len(), "{idx:?}");
        assert_eq!(&s.iter().collect::<Vec<_>>(), idx, "{idx:?}");
        for &i in idx {
            assert!(s.contains(i), "{idx:?} should contain {i}");
        }
        for i in [1usize, 62, 66, 999, 100_001] {
            if !idx.contains(&i) {
                assert!(!s.contains(i), "{idx:?} should not contain {i}");
            }
        }
        assert_eq!(s.max_index(), idx.iter().copied().max());
    }
}

#[test]
fn the_set_operations_are_the_set_operations() {
    // Against a `HashSet<usize>` over indices spread far past 64, so the
    // trie is genuinely several levels deep.
    let mut rng = Prng::new(5);
    for _ in 0..200 {
        let pick = |rng: &mut Prng| -> Vec<usize> {
            (0..12)
                .map(|_| (rng.next_u64() % 4096) as usize)
                .collect::<HashSet<_>>()
                .into_iter()
                .collect()
        };
        let (a, b) = (pick(&mut rng), pick(&mut rng));
        let (sa, sb): (Support, Support) = (a.iter().copied().collect(), b.iter().copied().collect());
        let (ha, hb): (HashSet<usize>, HashSet<usize>) =
            (a.iter().copied().collect(), b.iter().copied().collect());

        let sorted = |h: HashSet<usize>| {
            let mut v: Vec<usize> = h.into_iter().collect();
            v.sort_unstable();
            v
        };
        assert_eq!(
            sa.or(&sb).iter().collect::<Vec<_>>(),
            sorted(&ha | &hb),
            "union"
        );
        assert_eq!(
            sa.and(&sb).iter().collect::<Vec<_>>(),
            sorted(&ha & &hb),
            "intersection"
        );
        assert_eq!(
            sa.xor(&sb).iter().collect::<Vec<_>>(),
            sorted(&ha ^ &hb),
            "symmetric difference"
        );
        assert_eq!(
            sa.without(&sb).iter().collect::<Vec<_>>(),
            sorted(&ha - &hb),
            "difference"
        );
        assert_eq!(sa.intersects(&sb), !(&ha & &hb).is_empty());
        assert_eq!(sa.and_parity(&sb), (&ha & &hb).len() % 2 == 1);
    }
}

#[test]
fn equal_sets_are_equal_however_they_were_built() {
    // Canonicalization is what makes the type usable as a map key: two
    // supports built by different routes must hash and compare equal.
    let a: Support = [0usize, 100, 5000].iter().copied().collect();
    let b = Support::single(5000)
        .or(&Support::single(0))
        .or(&Support::single(100));
    let c = Support::single(0)
        .or(&Support::single(100))
        .or(&Support::single(5000))
        .or(&Support::single(7))
        .without(&Support::single(7));
    assert_eq!(a, b);
    assert_eq!(a, c);
    let mut set = HashSet::new();
    set.insert(a.clone());
    assert!(set.contains(&b) && set.contains(&c));
    assert_eq!(set.len(), 1);
    // and unequal sets stay unequal
    assert_ne!(a, a.or(&Support::single(1)));
    assert_ne!(a, a.without(&Support::single(0)));
}

#[test]
fn there_is_no_index_ceiling() {
    // A million qubits. Nothing in the module scales with the width, so
    // there is nothing to cap.
    let far = 1_000_000usize;
    let s = Support::single(far).or(&Support::single(3));
    assert_eq!(s.weight(), 2);
    assert!(s.contains(far) && s.contains(3));
    assert_eq!(s.max_index(), Some(far));
    // and it costs a path, not a million bits
    assert!(s.nodes() < 64, "{} nodes for two indices", s.nodes());
}

// ── the sharing is real ──────────────────────────────────────────────

#[test]
fn near_identical_supports_share_their_structure() {
    // The claim the whole design rests on. Terms that differ in two
    // places out of n must not each carry a private copy of n.
    clear_memo();
    let n = 8192usize;
    let mut rng = Prng::new(3);
    let mut cur = Support::single(n / 2);
    let mut terms = Vec::new();
    for _ in 0..2000 {
        let q = (rng.next_u64() as usize) % (n - 1);
        cur = cur.xor(&Support::single(q)).xor(&Support::single(q + 1));
        terms.push(cur.clone());
    }
    let shared = shared_nodes(&terms);
    let unshared = unshared_nodes(&terms);
    assert!(
        unshared as f64 / shared as f64 > 8.0,
        "sharing only {:.1}x ({unshared} unshared vs {shared} shared)",
        unshared as f64 / shared as f64
    );
}

#[test]
fn combining_costs_a_path_and_not_the_width() {
    // The precise claim, which is weaker than "constant" and much
    // stronger than "linear": changing two bits rebuilds the root-to-leaf
    // path, so the nodes introduced grow with the *depth* of the trie —
    // logarithmically in the register — while a flat bitset copies all
    // n/64 words. Measured, that is 2, 7 and 13 new nodes at n = 64,
    // 1024 and 65536, against 1, 16 and 1024 words.
    let mut sizes = Vec::new();
    for n in [64usize, 1024, 65536] {
        clear_memo();
        let a: Support = (0..n).step_by(3).collect();
        let b = Support::single(n / 2).or(&Support::single(n / 2 + 1));
        let before = shared_nodes(std::slice::from_ref(&a));
        let c = a.xor(&b);
        let after = shared_nodes(&[a, b, c]);
        sizes.push((n, after - before));
    }
    for &(n, new) in &sizes {
        let depth = (n / 64).max(1).ilog2() as usize;
        assert!(
            new <= 2 * depth + 4,
            "n={n}: {new} new nodes exceeds a path of depth {depth}: {sizes:?}"
        );
    }
    // The growth is logarithmic: a 1024-fold wider register costs a
    // handful of extra nodes rather than 1024 extra words.
    assert!(
        sizes[2].1 <= sizes[0].1 + 16,
        "the growth is not logarithmic: {sizes:?}"
    );
    // And the crossover is where it should be, stated rather than
    // glossed: at n = 64 the trie is *worse* than a single machine word,
    // and by n = 65536 it is two orders of magnitude better. Asserting
    // only the good end would be picking the flattering half.
    assert!(sizes[0].1 > 64 / 64, "at n=64 a flat word should still win");
    assert!(
        sizes[2].1 * 32 < 65536 / 64,
        "at n=65536 the path should be far cheaper than {} flat words: {sizes:?}",
        65536 / 64
    );
}

// ── it is the crate's algebra ────────────────────────────────────────

#[test]
fn the_wide_pauli_agrees_with_the_u64_one() {
    let mut rng = Prng::new(7);
    for _ in 0..20_000 {
        let p: PauliKey = (rng.next_u64(), rng.next_u64());
        let q: PauliKey = (rng.next_u64(), rng.next_u64());
        let (wp, wq) = (WidePauli::from_key(p), WidePauli::from_key(q));
        assert_eq!(commutes(p, q), wp.commutes(&wq), "commutation");
        let (pk, ps) = pauli_mul(p, q);
        let (wk, ws) = wp.mul(&wq);
        assert_eq!(Some(pk), wk.to_key(), "product");
        assert_eq!(ps, ws, "reordering sign");
        assert_eq!((p.0 | p.1).count_ones() as usize, wp.weight(), "weight");
    }
}

#[test]
fn a_wide_pauli_past_64_has_no_bounded_form_and_says_so() {
    let p = WidePauli {
        x: Support::single(200),
        z: Support::empty(),
    };
    assert_eq!(p.to_key(), None, "it must refuse to lie about fitting");
    assert_eq!(p.weight(), 1);
    assert_eq!(p.support(), vec![200]);
    let q = WidePauli::from_key((1, 1));
    assert!(q.to_key().is_some());
    assert!(p.commutes(&q), "disjoint supports commute");
}

// ── and the walk on top of it ────────────────────────────────────────

#[test]
fn the_wide_walk_agrees_with_the_bounded_one() {
    // Term for term and coefficient for coefficient — and *exactly*,
    // not to a tolerance: it is the same arithmetic in the same order,
    // and only the key type changed.
    for &(n, steps) in &[(8usize, 3usize), (16, 3), (24, 2), (40, 2)] {
        let rots: Vec<Rotation> = tfim_trotter(n, 1.0, 0.7, 0.35, steps);
        let wide: Vec<WideRotation> = rots
            .iter()
            .map(|r| WideRotation {
                theta: r.theta,
                axis: WidePauli::from_key(r.axis),
            })
            .collect();
        let cfg = heisenberg::Config {
            threshold: 0.0,
            max_terms: None,
            checkpoint_every: 0,
            exclusion: false,
            retire_frozen: false,
        };
        let narrow =
            heisenberg::propagate(&PauliSum::from_key((0, 1u64 << (n / 2))), &rots, &cfg).unwrap();
        clear_memo();
        let w = propagate_wide(&WidePauliSum::z_at(n / 2), &wide, &WideConfig::default());
        assert_eq!(narrow.sum.len(), w.sum.len(), "n={n}: term counts");
        for (k, c) in narrow.sum.terms() {
            let got = w.sum.get(&WidePauli::from_key(k));
            assert_eq!((got.re, got.im), (c.re, c.im), "n={n}: coefficient of {k:?}");
        }
    }
}

#[test]
fn the_wide_walk_runs_where_the_bounded_one_cannot() {
    // 4096 qubits. The bounded walk cannot represent a single term.
    let n = 4096usize;
    let mut rots = Vec::new();
    for _ in 0..2 {
        for q in 0..n - 1 {
            rots.push(WideRotation::rzz(q, q + 1, 0.35));
        }
        for q in 0..n {
            rots.push(WideRotation::rx(q, 0.245));
        }
    }
    clear_memo();
    let w = propagate_wide(
        &WidePauliSum::z_at(n / 2),
        &rots,
        &WideConfig {
            threshold: 1e-8,
            max_terms: Some(20_000),
        },
    );
    assert!(w.sum.len() > 1, "the walk should have branched");
    assert!(
        w.max_weight < 16,
        "a nearest-neighbour cone should stay narrow: weight {}",
        w.max_weight
    );
    assert_eq!(w.discarded_l1, 0.0, "nothing needed discarding at this depth");
    // and every term's support is genuinely past the u64 ceiling's reach
    assert!(
        w.sum.terms().all(|(p, _)| p.to_key().is_none()),
        "the terms should live past index 64, or the test proves nothing"
    );
}

#[test]
fn the_walk_reports_what_it_discarded() {
    let n = 64usize;
    let mut rots = Vec::new();
    for _ in 0..4 {
        for q in 0..n - 1 {
            rots.push(WideRotation::rzz(q, q + 1, 0.6));
        }
        for q in 0..n {
            rots.push(WideRotation::rx(q, 0.5));
        }
    }
    clear_memo();
    let exact = propagate_wide(&WidePauliSum::z_at(n / 2), &rots, &WideConfig::default());
    clear_memo();
    let capped = propagate_wide(
        &WidePauliSum::z_at(n / 2),
        &rots,
        &WideConfig {
            threshold: 0.0,
            max_terms: Some(64),
        },
    );
    assert!(capped.sum.len() <= 64);
    assert!(exact.sum.len() > capped.sum.len(), "the cap should have bitten");
    assert!(
        capped.discarded_l1 > 0.0,
        "and the walk must say what it dropped"
    );
    // the bound is honest: the answer moved by no more than the L1 dropped
    let moved = (exact.sum.vacuum_expectation() - capped.sum.vacuum_expectation()).abs();
    assert!(
        moved <= capped.discarded_l1 + 1e-12,
        "moved {moved} against a certified {}",
        capped.discarded_l1
    );
}

#[test]
fn the_memo_is_an_optimization_and_never_an_answer() {
    // Clearing the memo tables mid-flight must change nothing but speed.
    let n = 512usize;
    let rots: Vec<WideRotation> = (0..n - 1)
        .map(|q| WideRotation::rzz(q, q + 1, 0.35))
        .chain((0..n).map(|q| WideRotation::rx(q, 0.245)))
        .collect();
    clear_memo();
    let a = propagate_wide(&WidePauliSum::z_at(n / 2), &rots, &WideConfig::default());
    let b = propagate_wide(&WidePauliSum::z_at(n / 2), &rots, &WideConfig::default());
    clear_memo();
    let c = propagate_wide(&WidePauliSum::z_at(n / 2), &rots, &WideConfig::default());
    assert_eq!(a.sum.len(), b.sum.len());
    assert_eq!(a.sum.len(), c.sum.len());
    for (p, coeff) in a.sum.terms() {
        assert_eq!(b.sum.get(p).re, coeff.re);
        assert_eq!(c.sum.get(p).re, coeff.re);
    }
}
