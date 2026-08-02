//! Operator flows: that `U^t` really is a one-parameter group, and that
//! its width across a cut detects when the partial operation lands on a
//! lattice site.

use quantsim::opflow::{compose, deviation, schmidt_rank, shift_power, width_curve};
use quantsim::scalar::C64;

fn identity(n: usize) -> Vec<Vec<C64>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| C64::new(if i == j { 1.0 } else { 0.0 }, 0.0))
                .collect()
        })
        .collect()
}

/// **The group law, exactly.** `U^s ∘ U^t = U^{s+t}` at every pair tried,
/// including irrational parameters where no matrix power exists.
#[test]
fn the_flow_is_a_genuine_one_parameter_group() {
    let n = 24;
    let c = 5.0;
    for (s, t) in [
        (0.0, 1.0),
        (0.5, 0.5),
        (0.25, 0.75),
        (0.3, 0.4),
        (1.5, -0.5),
        (0.1234, 0.8766),
        (std::f64::consts::FRAC_1_PI, 0.2),
    ] {
        let a = shift_power(n, c, s).unwrap();
        let b = shift_power(n, c, t).unwrap();
        let joint = shift_power(n, c, s + t).unwrap();
        let dev = deviation(&compose(&a, &b), &joint);
        assert!(
            dev < 1e-9,
            "U^{s} ∘ U^{t} ≠ U^{{{}}}: deviation {dev}. The flow being an exact \
             group is what makes fractional powers meaningful at all.",
            s + t
        );
    }
}

/// The endpoints are what they should be: `t = 0` is the identity and
/// `t = 1` is the crisp permutation `x → x + c`.
#[test]
fn the_endpoints_are_the_identity_and_the_whole_operation() {
    let n = 16;
    let c = 3.0;
    assert!(deviation(&shift_power(n, c, 0.0).unwrap(), &identity(n)) < 1e-9);

    let u = shift_power(n, c, 1.0).unwrap();
    for (xp, row) in u.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            let want = if xp == (x + 3) % n { 1.0 } else { 0.0 };
            assert!(
                (cell.norm() - want).abs() < 1e-9,
                "U^1[{xp}][{x}] should be {want}, got {}",
                cell.norm()
            );
        }
    }
}

/// **Operator width sees the integers.**
///
/// At integer `t·c` the Dirichlet kernel collapses to a delta and the
/// operator is a narrow permutation; at fractional `t·c` nothing cancels
/// and the rank jumps. Width is therefore not a smooth interpolation
/// between the endpoints — it reads off arithmetic.
#[test]
fn the_width_dips_exactly_where_the_partial_shift_is_an_integer() {
    let (n, l, s) = (24usize, 4usize, 6usize);
    let c = 4.0; // so t·c is an integer at t = 0, .25, .5, .75, 1
    let curve = width_curve(n, c, l, s, 20);
    println!("   t      t·c    rank");
    let mut integral = Vec::new();
    let mut fractional = Vec::new();
    for &(t, rank) in &curve {
        let tc = t * c;
        let is_int = (tc - tc.round()).abs() < 1e-9;
        println!(
            "  {t:.2}   {tc:5.2}   {rank:4}{}",
            if is_int { "  <- integer" } else { "" }
        );
        if is_int {
            integral.push(rank);
        } else {
            fractional.push(rank);
        }
    }
    assert!(!integral.is_empty() && !fractional.is_empty());
    let worst_integral = *integral.iter().max().unwrap();
    let best_fractional = *fractional.iter().min().unwrap();
    assert!(
        worst_integral < best_fractional,
        "the widest integer-shift rank ({worst_integral}) was not below the \
         narrowest fractional one ({best_fractional}); if these overlap, width \
         no longer separates the lattice points"
    );
}

/// **A translation's width is the carry, wherever you cut it.**
///
/// The integer shift has operator Schmidt rank exactly 2 at every cut of
/// the ring — carry or no carry — independent of where the split lands.
/// The fractional shift is never narrower and is strictly wider wherever
/// there is room to see it.
///
/// The one exception is measured rather than hidden: at the most extreme
/// cut `2|12` the rank ceiling is low enough that the smeared operator
/// also comes out at 2, so the dip is invisible there. Everywhere with
/// `L ≥ 3` it is strict.
#[test]
fn a_translation_costs_one_carry_at_every_cut_and_a_fraction_costs_more() {
    let n = 24usize;
    let c = 1.0;
    let mut strict = 0usize;
    for (l, s) in [(2usize, 12usize), (3, 8), (4, 6), (6, 4), (8, 3), (12, 2)] {
        let crisp = schmidt_rank(&shift_power(n, c, 1.0).unwrap(), l, s);
        let smeared = schmidt_rank(&shift_power(n, c, 0.5).unwrap(), l, s);
        println!("   cut {l}|{s}: integer rank {crisp}, half-integer rank {smeared}");
        assert_eq!(
            crisp, 2,
            "at cut {l}|{s} the integer shift cost {crisp}, not the single carry \
             a translation should need wherever it is cut"
        );
        assert!(
            smeared >= crisp,
            "at cut {l}|{s} smearing the shift made it NARROWER ({smeared} < {crisp})"
        );
        if l >= 3 {
            assert!(
                smeared > crisp,
                "at cut {l}|{s} the half-integer shift ({smeared}) was not wider \
                 than the integer one ({crisp}); with L >= 3 there is room for the \
                 dip and it should be strict"
            );
            strict += 1;
        }
    }
    assert!(strict >= 4, "only {strict} cuts had room to show the dip");
}

/// The flow passes *through* the operation and keeps going: `U^2` is the
/// double shift, and it is exactly as narrow as `U^1`. Depth along the
/// flow does not cost width, only fractional position does.
#[test]
fn whole_powers_stay_narrow_however_far_along_the_flow() {
    let (n, l, s) = (24usize, 4usize, 6usize);
    let c = 1.0;
    let base = schmidt_rank(&shift_power(n, c, 1.0).unwrap(), l, s);
    for t in [2.0f64, 3.0, 5.0, 8.0, 13.0] {
        let r = schmidt_rank(&shift_power(n, c, t).unwrap(), l, s);
        assert_eq!(
            r, base,
            "U^{t} had rank {r} against {base} for U^1; whole powers are all \
             permutations and should cost the same"
        );
    }
    // And a fractional step anywhere along it is wide.
    let mid = schmidt_rank(&shift_power(n, c, 8.5).unwrap(), l, s);
    assert!(
        mid > base,
        "the half-step at t=8.5 ({mid}) should be wider than the whole ones ({base})"
    );
}
