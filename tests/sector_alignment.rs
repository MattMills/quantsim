//! Can a wider signature close the sums the elliptic sector cannot?
//!
//! `h*` is capped by the Hadamard count, and roughly half of a
//! skeleton's variables survive reduction. This file asks *why* they
//! survive, and whether extending the Hadamard to normal, hyperbolic and
//! ultrahyperbolic components — letting each variable align in whichever
//! space is aligned — could reach them.
//!
//! The census answers the first question: **29% of surviving variables
//! stall on alignment alone, and every one sits on an odd eighth** —
//! `1/8` or `3/8`, which are exactly the primitive 8th roots of unity,
//! the `T` gate showing up directly in the coefficient. The other 71%
//! are structural (51% shape, 20% coupling) and no change of scalar
//! algebra touches them. So the target is not a rich alignment set; it
//! is one missing closure *order*.
//!
//! The second question then has a short and negative answer, and the
//! reason is worth more than the answer. Summing a boolean variable with
//! self-coefficient `c` gives `1 + u` for `u` a root of unity of order
//! `1/c`, and the magnitude that has to be absorbed is
//!
//! ```text
//!   |1 + u|² = 2 + 2cos(2πc)
//! ```
//!
//! which depends **only on the order of `u`** — not on the signature it
//! lives in. Concretely:
//!
//! * **Hyperbolic** (`j² = +1`): the unit group `{a + bj : a² − b² = 1}`
//!   is `(cosh t, sinh t)`, and the `n`-th power is `(cosh nt, sinh nt)`,
//!   which is the identity only at `t = 0`. The group is **torsion-free**
//!   — it has no element of order 8, or even of order 4. Strictly poorer
//!   than `ℂ`, not richer.
//! * **Ultrahyperbolic** `(2,2)` — split quaternions, `≅ M₂(ℝ)`, unit
//!   group `SL(2,ℝ)`: order-8 elements *do* exist. But
//!   `det(I + R_{π/4}) = 2 + √2`, identical to the elliptic value. The
//!   extra time direction buys nothing here.
//!
//! So the obstruction is not the metric. It is the **ring**:
//! `2 + √2 = √2·(1 + √2)`, and `1 + √2` is the fundamental unit of
//! `ℤ[√2]` — while `3/8` gives `2 − √2 = √2·(√2 − 1)`, the same unit
//! inverted. The path sum's magnitude bookkeeping (`e_half`) counts only
//! integer powers of `√2`. Closing an odd-eighth self-term means carrying
//! `ℤ[√2]` units — equivalently moving the coefficient ring from
//! `ℤ[ζ₈, 1/√2]` up to `ℤ[ζ₁₆]`. That is a cyclotomic tower, and each
//! level buys exactly one more closure point while introducing the next
//! stall one level finer.
//!
//! The census also bounds the prize before anyone builds it: closing
//! every `1/8` stall removes 29% of surviving variables, taking `h*/n`
//! from about `0.53` to about `0.38`. Worth roughly `1.4×` the reachable
//! width. Real, and not a change of growth class.
//!
//! ## The other 71%, and which stratum it lives in
//!
//! The structural stalls turn out to be far more specific than "compound
//! monomials". **Every one measured is at degree exactly three** — none
//! at degree 2, none at 4 or above. Rules `[E]` and `[G]` take a variable
//! appearing as `y` or `y·L`, degree 1 and 2; degree 3 is the first shape
//! they cannot take, and nothing climbs higher because `CCZ` is the
//! widest gate in the fragment and rule `[V]` splits reduce degree. On
//! `CCZ`-rich circuits the cubic stratum is *every* stall.
//!
//! And that stratum is **sparse**: about `2n` terms against the `~n³/6`
//! available, under 0.1% occupancy at `n = 64`. So reaching it is not a
//! storage problem — the terms are few and already held exactly. The
//! problem is that `[E]` and `[G]` are quadratic while a degree-3 trap
//! `y·p·q` is cubic, leaving only rule `[V]`, which splits one monomial
//! into three rather than eliminating anything.
//!
//! ## And the cubic case is a constraint, not a Gauss sum
//!
//! Every cubic-trapped variable on a `CCZ`-rich circuit carries
//! coefficients that are exactly `½`. That settles which of the two
//! possible cubic problems is the real one. Summing `y` out gives
//! `1 + ω^A`; at half turns `A ∈ {0, ½}`, so the sum is `2` or `0` — it
//! **vanishes unless a constraint holds**, exactly like rule `[E]`, with
//! no varying magnitude anywhere. Quarter-turn coefficients *would* have
//! given magnitudes `2` and `√2`, genuinely path-dependent and needing a
//! normalization exponent that varies per path. That case does not occur.
//!
//! So the open target is not a cubic Gauss sum. It is rule `[E]` lifted
//! from *affine* constraints to **quadratic** ones: the constraint handed
//! to it is `L ⊕ p·q = c`.
//!
//! That locates the blocker exactly. Solving such a constraint means
//! substituting a quadratic expression where a parity is expected, and a
//! `Mask` is linear by construction. Expanding `a ⊕ b = a + b − 2ab` to
//! return to monomials *is* rule `[V]` — the split the whole exercise was
//! trying to avoid. **The obstruction is the parity representation of the
//! substitution target**, not the scalar algebra and not the signature.
//!
//! Three bottlenecks, all now measured rather than guessed:
//!
//! | share | stall | blocked by |
//! |---|---|---|
//! | 29% | odd-eighth closure *order* | the coefficient ring, `ℤ[√2]` units |
//! | 71% | cubic closure *degree* | affine-only substitution (`Mask` is a parity) |
//! | — | signature | ruled out: closure depends on the order of `u`, not the metric |

use quantsim::circuit::Op;
use quantsim::pathsum::{self, Stall};
use quantsim::prelude::*;

fn skeleton(n: usize, layers: usize, seed: u64) -> Circuit<C64> {
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for _ in 0..layers {
        for q in 0..n {
            c.gate("h", vec![], vec![q]);
        }
        for _ in 0..n {
            let a = (rng.next_u64() % n as u64) as usize;
            let b = (a + 1 + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
            c.gate("cx", vec![], vec![a, b]);
        }
    }
    c
}

fn with_magic(skel: &Circuit<C64>, t: usize, seed: u64) -> Circuit<C64> {
    let n = skel.num_qubits();
    let mut rng = Prng::new(seed);
    let mut slots: Vec<usize> = (0..t)
        .map(|_| (rng.next_u64() % (skel.len() as u64 + 1)) as usize)
        .collect();
    slots.sort_unstable();
    let mut c = Circuit::new(n);
    let mut next = 0usize;
    for (i, op) in skel.ops().iter().enumerate() {
        while next < slots.len() && slots[next] == i {
            c.gate("t", vec![], vec![(rng.next_u64() % n as u64) as usize]);
            next += 1;
        }
        if let Op::Named {
            name,
            params,
            qubits,
        } = op
        {
            c.gate(name, params.clone(), qubits.clone());
        }
    }
    while next < slots.len() {
        c.gate("t", vec![], vec![(rng.next_u64() % n as u64) as usize]);
        next += 1;
    }
    c
}

/// **The census.** How much of `h*` is even reachable by a wider algebra?
///
/// Every alignment stall lands on an *odd eighth* — `1/8`, `3/8`, … —
/// which are exactly the primitive 8th roots of unity, so they are the
/// `T` gate showing up directly in the coefficient.
///
/// Structural stalls are facts about the monomial shape and are immune to
/// the choice of scalars. Alignment stalls are the opposite: the shape is
/// exactly what a rule wants and only the self-coefficient is off. This
/// measures the split, and it is what bounds the prize before any
/// machinery is built.
#[test]
fn most_stalls_are_structural_and_every_alignment_stall_is_an_eighth() {
    let mut align = 0usize;
    let mut total = 0usize;
    for (n, layers, t) in [
        (8usize, 4usize, 8usize),
        (8, 4, 32),
        (16, 2, 16),
        (16, 4, 16),
        (32, 2, 32),
        (32, 4, 32),
        (64, 2, 64),
    ] {
        let ps = pathsum::operator(&with_magic(&skeleton(n, layers, 11), t, 5)).unwrap();
        for (_, s) in ps.stall_census() {
            total += 1;
            if let Stall::Alignment(c) = s {
                align += 1;
                let eighths = c / pathsum::EIGHTH;
                assert!(
                    c % pathsum::EIGHTH == 0 && eighths % 2 == 1,
                    "an alignment stall sat at {c:#x}, which is not an ODD eighth. \
                     Odd eighths are exactly the primitive 8th roots of unity — \
                     the T-gate points — and the whole cyclotomic-tower reading \
                     depends on that being where they land."
                );
            }
        }
    }
    let share = align as f64 / total as f64;
    println!("alignment stalls: {align} of {total} ({:.0}%)", 100.0 * share);
    assert!(total > 50, "corpus too small to conclude from");
    assert!(
        align > 0,
        "no alignment stalls at all, so a wider algebra has nothing to reach"
    );
    assert!(
        share < 0.6,
        "alignment stalls are {:.0}% of the total. They were 29% when the \
         cyclotomic-tower conclusion was drawn; a majority would mean the \
         structural rules, not the coefficient ring, became the bottleneck.",
        100.0 * share
    );
}

/// **The closure factor depends on the order of the root of unity, not
/// on the signature it lives in.**
///
/// Summing a boolean variable with self-coefficient `c` produces `1 + u`
/// with `u` of order `1/c`, and `|1 + u|² = 2 + 2cos(2πc)` in every
/// signature. The elliptic sector absorbs it only when that is a power of
/// two, which happens at `0`, `¼`, `½`, `¾` and nowhere else.
#[test]
fn the_closure_factor_is_a_power_of_two_only_at_the_quarter_turns() {
    let f = |c: f64| 2.0 + 2.0 * (std::f64::consts::TAU * c).cos();
    for (label, c, closes) in [
        ("0", 0.0, true),
        ("1/4", 0.25, true),
        ("1/2", 0.5, true),
        ("3/4", 0.75, true),
        ("1/8", 0.125, false),
        ("3/8", 0.375, false),
    ] {
        let v = f(c);
        let is_pow2 = v == 0.0 || (v.log2() - v.log2().round()).abs() < 1e-12;
        println!("  c={label:5} |1+u|² = {v:.6}   power of two: {is_pow2}");
        assert_eq!(is_pow2, closes, "closure at c={label} was misclassified");
    }
    // The eighth is the one the census says matters, and it lands on the
    // fundamental unit of ℤ[√2] rather than on a power of √2.
    let eighth = f(0.125);
    assert!((eighth - (2.0 + 2f64.sqrt())).abs() < 1e-12);
    assert!((eighth - 2f64.sqrt() * (1.0 + 2f64.sqrt())).abs() < 1e-12);
}

/// **The hyperbolic unit group is torsion-free**, so it does not contain
/// an eighth root of unity — or a fourth. It is poorer than `ℂ`, not
/// richer, and cannot close what the elliptic sector cannot.
#[test]
fn the_hyperbolic_unit_group_has_no_element_of_order_eight() {
    // Unit hyperbola: (cosh t, sinh t). The n-th power is (cosh nt, sinh nt),
    // so it returns to the identity only at t = 0 — for EVERY order n, not
    // just 8. Kept in a range where cosh² is still exact in f64; the claim
    // is algebraic and the sampling only illustrates it.
    for step in 1..40 {
        let t = step as f64 * 0.05;
        for n in [2.0f64, 4.0, 8.0] {
            let (a, b) = ((n * t).cosh(), (n * t).sinh());
            assert!(
                (a - 1.0).abs() > 1e-9 || b.abs() > 1e-9,
                "a nonzero boost t={t} returned to the identity at power {n}"
            );
            // and it never leaves the unit hyperbola, so the norm cannot
            // supply the missing factor either
            assert!((a * a - b * b - 1.0).abs() < 1e-6 * (a * a).max(1.0));
        }
    }
}

/// **The ultrahyperbolic sector does contain order-8 elements — and they
/// give the same factor.**
///
/// Split quaternions are `M₂(ℝ)` and their norm-one units are `SL(2,ℝ)`,
/// which has elements of order 8. But `det(I + R_θ) = 2 + 2cos θ`, the
/// elliptic value. The second time direction adds conjugacy classes, not
/// closure points.
#[test]
fn the_ultrahyperbolic_order_eight_element_gives_the_elliptic_factor() {
    let th = std::f64::consts::FRAC_PI_4;
    let det = (1.0 + th.cos()).powi(2) + th.sin().powi(2);
    assert!(
        (det - (2.0 + 2.0 * th.cos())).abs() < 1e-12,
        "det(I+R) should be 2+2cos θ"
    );
    assert!(
        (det - (2.0 + 2f64.sqrt())).abs() < 1e-12,
        "an order-8 element of SL(2,R) gives {det}, and the elliptic sector \
         gives 2+√2 — if these ever differ, the signature route is back open"
    );
}

// ─────────────────────────────────────────────────────────────────────
// The strata: which DEGREE the structural stalls live at.
// ─────────────────────────────────────────────────────────────────────

/// `CCZ`-rich circuits, which push cubic terms into the phase polynomial.
fn cubic(n: usize, layers: usize, seed: u64) -> Circuit<C64> {
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for _ in 0..layers {
        for q in 0..n {
            c.gate("h", vec![], vec![q]);
        }
        for _ in 0..n {
            let a = (rng.next_u64() % n as u64) as usize;
            let b = (a + 1 + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
            let mut d = (rng.next_u64() % n as u64) as usize;
            while d == a || d == b {
                d = (d + 1) % n;
            }
            c.gate("ccz", vec![], vec![a, b, d]);
        }
    }
    c
}

/// **Every structural stall is at degree exactly three.**
///
/// Not one at degree 2, not one at degree 4 or above. The reason is
/// visible in the rules: `[E]` and `[G]` handle a variable appearing as
/// `y` or as `y·L` — degree 1 and 2 — and degree 3 is the first shape
/// they cannot take. Nothing climbs past 3 because `CCZ` is the widest
/// gate in the fragment and rule `[V]` splits reduce degree.
///
/// So the structural bottleneck is not "compound monomials" in general.
/// It is one stratum: the cubic one. On `CCZ`-rich circuits it is
/// *every* stall — zero alignment stalls at all.
#[test]
fn every_structural_stall_sits_in_the_cubic_stratum() {
    for (label, c) in [
        ("skeleton n=32", with_magic(&skeleton(32, 2, 11), 32, 5)),
        ("skeleton n=64", with_magic(&skeleton(64, 2, 11), 64, 5)),
        ("cubic n=16", cubic(16, 2, 11)),
        ("cubic n=32", cubic(32, 2, 11)),
    ] {
        let ps = pathsum::operator(&c).unwrap();
        let mut cubic_stalls = 0usize;
        for (_, st) in ps.stall_census() {
            if let Stall::Shape { degree } = st {
                assert_eq!(
                    degree, 3,
                    "{label}: a structural stall sat at degree {degree}. Every one \
                     measured has been cubic, and the whole reading — that the \
                     bottleneck is one stratum rather than compound monomials in \
                     general — depends on that."
                );
                cubic_stalls += 1;
            }
        }
        println!("   {label:16} h*={:3}  cubic stalls={cubic_stalls}", ps.internal_vars());
    }
}

/// **The cubic stratum is `O(n)`-occupied, not `O(n³)`.**
///
/// There are `~n³/6` possible degree-3 monomials, but a reduced path sum
/// only ever holds about `2n` of them — occupancy at `n = 64` is under
/// 0.1%. The `n²` stratum behaves the same way, around `2n` against
/// `n²/2` possible.
///
/// That matters for what a richer representation would have to do.
/// Reaching the cubic stratum is **not a storage problem** — the terms
/// are already few and already held exactly. The problem is that rules
/// `[E]` and `[G]` are *quadratic* and a degree-3 trap `y·p·q` is cubic,
/// so the only move available is rule `[V]`, which splits one monomial
/// into three rather than eliminating anything.
///
/// The open target is therefore a **cubic elimination rule**, and it is
/// a question about closing a cubic Gauss sum, not about holding `n³`
/// coefficients.
#[test]
fn the_cubic_stratum_is_sparse_so_reaching_it_is_not_a_storage_problem() {
    println!("      n   deg2   deg3   deg2/n²   deg3/n³");
    let mut last_ratio = 1.0f64;
    for n in [8usize, 16, 32, 64] {
        let ps = pathsum::operator(&cubic(n, 2, 11)).unwrap();
        let prof = ps.degree_profile();
        let g = |d: usize| *prof.get(&d).unwrap_or(&0);
        let (d2, d3) = (g(2), g(3));
        let r3 = d3 as f64 / (n * n * n) as f64;
        println!(
            "   {n:6} {d2:6} {d3:6}   {:7.4}   {r3:8.5}",
            d2 as f64 / (n * n) as f64
        );
        assert!(
            d3 <= 8 * n,
            "the cubic stratum held {d3} terms at n={n}, past the ~2n that has \
             been measured; if it ever grows like n³ this stops being sparse \
             and becomes a storage problem after all"
        );
        assert!(
            r3 < last_ratio,
            "cubic occupancy did not fall with n ({r3} vs {last_ratio})"
        );
        last_ratio = r3;
        assert!(
            prof.keys().all(|&d| d <= 3),
            "a monomial of degree {} appeared; the fragment is closed at cubic",
            prof.keys().max().unwrap()
        );
    }
}

/// **The cubic stall is a quadratic CONSTRAINT, not a missing Gauss sum.**
///
/// Every cubic-trapped variable on a `CCZ`-rich circuit carries
/// quarter-multiple coefficients — in fact all of them are exactly `½`.
/// That decides which of the two possible cubic problems is the real one.
///
/// Summing `y` out of a trap gives `1 + ω^A` with `A` a polynomial in the
/// remaining parities. At half turns `A ∈ {0, ½}`, so the sum is `2` or
/// `0`: it **vanishes unless a constraint holds**, exactly like rule
/// `[E]`, and the magnitude never varies. Had the coefficients been
/// quarter turns instead, the magnitudes would have been `2` and `√2` —
/// genuinely path-dependent, needing a normalization exponent that varies
/// per path. That case does not occur.
///
/// So the cubic bottleneck is not a missing cubic Gauss sum. It is rule
/// `[E]` being restricted to *affine* constraints while the constraint it
/// is handed, `L ⊕ p·q = c`, is **quadratic**.
///
/// And that locates the blocker precisely: solving it means substituting
/// a quadratic expression where a parity is expected, and a `Mask` is
/// linear by construction. Expanding `a ⊕ b = a + b − 2ab` to get back to
/// monomials *is* rule `[V]` — which is the split the whole exercise was
/// trying to avoid. The obstruction is the parity representation of the
/// substitution target, not the scalar algebra and not the signature.
#[test]
fn cubic_stalls_are_half_turn_constraints_rather_than_gauss_sums() {
    for (label, n) in [("cubic n=16", 16usize), ("cubic n=32", 32), ("cubic n=64", 64)] {
        let ps = pathsum::operator(&cubic(n, 2, 11)).unwrap();
        let mut seen = 0usize;
        for (v, st) in ps.stall_census() {
            if let Stall::Shape { degree: 3 } = st {
                seen += 1;
                for (deg, c) in ps.terms_containing(v) {
                    assert_eq!(
                        c,
                        pathsum::HALF,
                        "{label}: a degree-{deg} term on a cubic-trapped variable \
                         carried {c:#x} rather than a half turn. Half turns make the \
                         trap a constraint (sum is 2 or 0); anything else would make \
                         it a varying-magnitude sum and change what the fix has to be."
                    );
                }
            }
        }
        assert!(seen > 0, "{label}: no cubic stalls to characterise");
        println!("   {label:12} {seen} cubic stalls, all half-turn constraints");
    }
}

/// The contrast that fixes the reading: a half-turn trap has a *constant*
/// magnitude and a quarter-turn one would not.
#[test]
fn a_half_turn_trap_keeps_one_magnitude_and_a_quarter_turn_trap_would_not() {
    let mag = |a: f64| {
        let (s, c) = (std::f64::consts::TAU * a).sin_cos();
        ((1.0 + c) * (1.0 + c) + s * s).sqrt()
    };
    // Half turns: the two branches are 2 and 0 — a constraint, and the
    // nonzero branch has a single magnitude that `e_half` already holds.
    assert!((mag(0.0) - 2.0).abs() < 1e-12);
    assert!(mag(0.5).abs() < 1e-12);
    // Quarter turns: two DIFFERENT nonzero magnitudes, which no single
    // global normalization exponent can carry.
    assert!((mag(0.0) - 2.0).abs() < 1e-12);
    assert!((mag(0.25) - 2f64.sqrt()).abs() < 1e-12);
    assert!(
        (mag(0.0) - mag(0.25)).abs() > 0.5,
        "if these ever agreed, the quarter-turn cubic case would close too"
    );
}
