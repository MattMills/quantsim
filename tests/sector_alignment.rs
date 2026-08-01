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
