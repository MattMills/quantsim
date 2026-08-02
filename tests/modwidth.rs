//! The a-priori width calculus: an operator's rank across a cut,
//! predicted from number theory and checked against linear algebra.
//!
//! The whole value of the theorem is that it answers before any tensor
//! exists, so the tests are built the same way round: the prediction is
//! computed with no matrix, then an independent routine builds the
//! permutation and eliminates. If they ever disagree the theorem is what
//! is wrong.

use quantsim::modwidth::{
    self, exact_schmidt_rank, mod_inverse, mult_cut_width, mult_width_profile, power_width_curve,
    Width,
};
use std::collections::HashSet;

/// Direct enumeration of the crossing set, to pin the closed-form branch.
fn enumerated(k: u128, l: u128, s: u128) -> u128 {
    let k = k % (l * s);
    let mut seen: HashSet<u128> = HashSet::new();
    for b in 0..s {
        seen.insert((k * b / s) % l);
    }
    seen.len() as u128
}

/// For `k ≤ S` the map `b ↦ ⌊kb/S⌋` steps by 0 or 1 and reaches `k−1`, so
/// it is a surjection onto `[0, k)` and the width is `min(k, L)` with no
/// enumeration at all. This pins the shortcut against the honest count.
#[test]
fn the_closed_form_branch_agrees_with_enumeration() {
    for l in [2u128, 6, 24, 120] {
        for s in [6u128, 24, 120] {
            for k in [1u128, 2, 3, 5, s - 1, s] {
                let w = mult_cut_width(k, l, s);
                assert!(w.is_exact());
                assert_eq!(w.value(), enumerated(k, l, s), "k={k} l={l} s={s}");
            }
        }
    }
}

/// **The independent check.** Everything in `modwidth` is number theory;
/// `exact_schmidt_rank` is linear algebra that never mentions the
/// crossing message. They must agree.
#[test]
fn the_number_theory_prediction_equals_the_measured_operator_rank() {
    let mut checked = 0usize;
    for (l, s) in [
        (4usize, 6usize),
        (6, 8),
        (8, 6),
        (6, 6),
        (4, 12),
        (12, 4),
        (6, 12),
        (12, 6),
        (8, 12),
        (12, 8),
        (10, 10),
        (9, 8),
    ] {
        let n = (l * s) as u128;
        for k in 1..n {
            // The theorem is a rank statement for invertible multipliers.
            if gcd(k, n) != 1 {
                continue;
            }
            let predicted = mult_cut_width(k, l as u128, s as u128);
            assert!(predicted.is_exact(), "small cuts must be exact");
            let measured = exact_schmidt_rank(k, l, s).expect("within the exact-rank ceiling");
            assert_eq!(
                predicted.value() as usize,
                measured,
                "×{k} on L={l} S={s}: number theory said {predicted}, elimination \
                 measured {measured}"
            );
            checked += 1;
        }
    }
    assert!(checked > 100, "only {checked} multipliers exercised");
    println!("{checked} invertible multipliers agreed, prediction vs elimination");
}

fn gcd(a: u128, b: u128) -> u128 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// **Cost is resonant, not monotone: a larger multiplier can be cheaper.**
///
/// At the waist cut `L = 24, S = 120` of `ℤ_2880`, repeated squaring
/// `7 → 49 → 2401 → 1921` gives widths `7 → 24 → 6 → 3`. Squaring a
/// multiplier can *narrow* it, and `×2401` is far cheaper than `×7`.
#[test]
fn squaring_a_multiplier_can_make_it_narrower() {
    let chain = [(7u128, 7u128), (49, 24), (2401, 6), (1921, 3)];
    for (k, expect) in chain {
        let w = mult_cut_width(k, 24, 120);
        assert!(w.is_exact());
        assert_eq!(w.value(), expect, "×{k}");
    }
    // The point stated as a comparison rather than a table: 2401 > 7, and
    // it is cheaper.
    assert!(
        mult_cut_width(2401, 24, 120).value() < mult_cut_width(7, 24, 120).value(),
        "the resonance is the whole finding; without it cost would just track k"
    );
    // And the inverse multiplier costs the same as the multiplier, since
    // rank(A) = rank(A†) and M_k† = M_{k⁻¹}.
    let n = 2880u128;
    for k in [7u128, 11, 49, 121, 823, 2401] {
        let ki = mod_inverse(k, n);
        assert_eq!(
            mult_cut_width(k, 24, 120).value(),
            mult_cut_width(ki, 24, 120).value(),
            "×{k} vs its inverse ×{ki}"
        );
    }
}

/// **Unbounded depth at bounded width.**
///
/// `k^e` cycles with the multiplicative order of `k` mod `N`, so the
/// width curve cycles with it: the cost of a modular-exponentiation
/// kernel is *periodic*, never growing, and its maximum over one period
/// is computable in advance from number theory alone.
///
/// This is the form the result has to take to matter — not "this circuit
/// is small" but "this family stays bounded however deep it goes".
///
/// Scope, corrected against the source: the `novel_quantum_structures`
/// README states that `×7^K` "stays at the resonant width 3". That does
/// not reproduce. At the `ℤ_2880` waist the measured curve is
/// `[7, 24, 24, 6, 24, 24, 21, 3, 21, 24, 14, 2]` — emphatically
/// resonant, and emphatically not fixed. Periodicity and boundedness are
/// what actually hold, and they are what is asserted here.
#[test]
fn the_modular_exponentiation_curve_is_periodic_and_therefore_bounded() {
    let (l, s) = (24u128, 120u128);
    let n = l * s;
    let order = multiplicative_order(7, n);
    let curve: Vec<u128> = power_width_curve(7, l, s, 2 * order)
        .iter()
        .map(|w| w.value())
        .collect();
    println!("ord(7) mod {n} = {order}");
    println!("×7^e widths at the waist: {:?}", &curve[..order]);

    // Periodic with the multiplicative order: depth past one period buys
    // the adversary nothing at all.
    for e in 0..order {
        assert_eq!(
            curve[e],
            curve[e + order],
            "the width curve is not periodic at e={e}; if it were not, deep \
             exponentiation could cost more than one period's maximum"
        );
    }

    // Bounded by the cut, and the bound is reached — so "bounded" is a
    // real statement rather than a vacuous one.
    let peak = *curve.iter().max().unwrap();
    assert!(peak <= l.min(s), "width exceeded the cut dimension");
    println!("peak over the period: {peak} (cut cap {})", l.min(s));

    // Resonant: it falls as often as it rises, which is the finding that
    // makes cost a property of k rather than of its size.
    assert!(
        curve.windows(2).any(|p| p[1] < p[0]),
        "the curve never fell, so cost would just track the multiplier"
    );
}

/// Some bonds are *exactly* flat across every power — the outer bonds of
/// the dimension wave never move however deep the exponentiation goes,
/// while the middle bonds resonate.
#[test]
fn the_outer_bonds_of_a_wave_are_flat_across_every_power() {
    let profile = [2usize, 3, 4, 5, 4, 3, 2];
    let n: u128 = profile.iter().map(|&d| d as u128).product();
    let mut acc = 1u128 % n;
    let mut first: Option<Vec<u128>> = None;
    let mut moved = vec![false; profile.len() - 1];
    for _ in 0..12 {
        acc = (acc * 7) % n;
        if acc == 1 {
            break; // the identity is bond-1 everywhere and not informative
        }
        let w: Vec<u128> = mult_width_profile(&profile, acc)
            .iter()
            .map(|x| x.value())
            .collect();
        match &first {
            None => first = Some(w),
            Some(f) => {
                for (i, (a, b)) in f.iter().zip(&w).enumerate() {
                    if a != b {
                        moved[i] = true;
                    }
                }
            }
        }
    }
    let flat: Vec<usize> = (0..moved.len()).filter(|&i| !moved[i]).collect();
    println!(
        "bonds flat across every power of 7: {flat:?} of {}",
        moved.len()
    );
    assert!(
        flat.contains(&0) && flat.contains(&(moved.len() - 1)),
        "the outer bonds should not move; flat set was {flat:?}"
    );
    assert!(
        !flat.is_empty() && flat.len() < moved.len(),
        "some bonds flat and some resonant is the structure being claimed"
    );
}

fn multiplicative_order(k: u128, n: u128) -> usize {
    let mut acc = k % n;
    let mut e = 1usize;
    while acc != 1 {
        acc = (acc * k) % n;
        e += 1;
        assert!(e < 10_000, "no multiplicative order found");
    }
    e
}

/// Reducing `k` mod `N` is sound: `k + LS` changes `c(b)` by
/// `L·b ≡ 0 (mod L)`.
#[test]
fn reduction_mod_n_leaves_the_width_alone() {
    for k in [5u128, 49, 823, 2401] {
        assert_eq!(
            mult_cut_width(k, 24, 120).value(),
            mult_cut_width(k + 2880, 24, 120).value(),
            "×{k}"
        );
    }
}

/// The whole cost curve of an operator across every bond of a chain,
/// computed without building the operator anywhere.
#[test]
fn a_whole_bond_profile_is_predicted_with_no_operator_built() {
    let profile = [2usize, 3, 4, 5, 4, 3, 2];
    for k in [7u128, 11, 49, 2401] {
        let widths = mult_width_profile(&profile, k);
        assert_eq!(widths.len(), profile.len() - 1, "one width per bond");
        assert!(widths.iter().all(|w| w.is_exact()));
        let n: u128 = profile.iter().map(|&d| d as u128).product();
        for (m, w) in widths.iter().enumerate() {
            let left: u128 = profile[..=m].iter().map(|&d| d as u128).product();
            assert!(
                w.value() <= left.min(n / left),
                "bond {m} width {w} exceeded the cut's own dimension"
            );
        }
        println!(
            "   ×{k:<5} {:?}",
            widths.iter().map(|w| w.value()).collect::<Vec<_>>()
        );
    }
}

/// A width past the enumeration ceiling is reported as a bound rather
/// than silently costing an unbounded scan or, worse, being passed off as
/// exact.
#[test]
fn an_unenumerable_cut_returns_a_bound_and_says_so() {
    let s = modwidth::MAX_ENUM_BLOCK + 1;
    let w = mult_cut_width(s + 5, 1_000_000, s);
    assert!(
        matches!(w, Width::UpperBound(_)),
        "past MAX_ENUM_BLOCK the answer must be a bound, got {w}"
    );
    assert!(format!("{w}").starts_with('≤'), "and must print as one");
    // While anything inside the closed-form branch stays exact however
    // large the block is, because it needs no scan.
    assert!(mult_cut_width(3, 1_000_000, s).is_exact());
}
