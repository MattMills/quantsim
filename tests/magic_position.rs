//! Positions in the magic n-fold as exact arithmetic: the layer
//! decomposition pinned against the magic square, the closed form on the
//! octonionic line, the Bott grading as a position-only convention, the
//! two folds, and the refusal where the compound tower leaves `u128`.

use quantsim::magic::{octonionic_bott, octonionic_dim, Composition, Position, Slot};
use Composition::{C, H, O, R};

/// **The layer sum is the magic square.** Nothing here is fitted: the
/// dimensions fall out of the slots' `(t, der)` and the Tits layers, and
/// they land on the known exceptional values exactly.
#[test]
fn the_layer_formula_reproduces_the_magic_square() {
    let cases: [(&[Composition], u128, &str); 7] = [
        (&[R, O], 52, "F4"),
        (&[C, O], 78, "E6"),
        (&[H, O], 133, "E7"),
        (&[O, O], 248, "E8"),
        (&[H, H], 66, "so(12)"),
        (&[C, C], 16, ""),
        (&[O], 52, "F4 again — R is the identity slot"),
    ];
    for (algs, want, what) in cases {
        let p = Position::algebras(algs).unwrap();
        assert_eq!(
            p.dim().unwrap(),
            want,
            "{} should be {want} {what}",
            p.label()
        );
    }

    // And the layers add up to the total, term by term, on E8.
    let e8 = Position::algebras(&[O, O]).unwrap();
    let l = e8.layers().unwrap();
    assert_eq!(
        (l.frame, l.derivations, l.linear, l.pairwise),
        (3, 28, 70, 147)
    );
    assert_eq!(l.higher, 0, "a 2-slot position has no term above pairwise");
    assert_eq!(
        l.frame + l.derivations + l.linear + l.pairwise + l.higher,
        248
    );
}

/// **The closed form agrees with the layer sum**, and its leading term is
/// `dim 𝕆^{⊗n}` — the n-fold is the octonion tensor power plus a quadratic.
#[test]
fn the_closed_form_agrees_with_the_layer_sum() {
    for n in 1..=8usize {
        let layered = Position::octonionic(n).unwrap().dim().unwrap();
        let closed = octonionic_dim(n as u32).unwrap();
        assert_eq!(layered, closed, "the two routes disagree at n={n}");
    }
    // the known rungs
    assert_eq!(octonionic_dim(2).unwrap(), 248);
    assert_eq!(octonionic_dim(3).unwrap(), 934);
    assert_eq!(octonionic_dim(4).unwrap(), 4854);
    // and the leading term really is 8^n
    for n in 1..=10u32 {
        let d = octonionic_dim(n).unwrap();
        assert!(
            d > 8u128.pow(n),
            "8^n must be a strict lower bound at n={n}"
        );
        assert!(
            d < 8u128.pow(n) + 200 * (n as u128) * (n as u128),
            "the correction above 8^n must stay quadratic at n={n}"
        );
    }
}

/// **The grading is a quadratic with period 8**, so it is `O(1)` from
/// coordinates however far out the position sits.
#[test]
fn the_bott_grading_is_a_quadratic_with_period_eight() {
    // 8^n fits u128 up to n = 42.
    for n in 1..=42u32 {
        let d = octonionic_dim(n).unwrap();
        assert_eq!(
            (d % 8) as u8,
            octonionic_bott(n),
            "the closed-form grading disagrees with dim mod 8 at n={n}"
        );
    }
    for n in 1..=34u32 {
        assert_eq!(
            octonionic_bott(n),
            octonionic_bott(n + 8),
            "the grading must have period 8"
        );
    }
    // the first turn of the circle, as measured
    let turn: Vec<u8> = (1..=8).map(octonionic_bott).collect();
    assert_eq!(turn, vec![4, 0, 6, 6, 0, 4, 2, 2]);
}

/// **Every magic n-fold lands in the even sector.** `n(n+1)` is even for
/// every `n`, so `n² + n + 2` is too — the odd classes are unreachable on
/// this line, and it is a one-line proof rather than a sweep.
#[test]
fn every_magic_n_fold_lands_in_the_even_sector() {
    for n in 1..=42u32 {
        assert_eq!(octonionic_bott(n) % 2, 0, "n={n} reached an odd class");
    }
    let seen: std::collections::BTreeSet<u8> = (1..=42).map(octonionic_bott).collect();
    assert_eq!(seen, [0u8, 2, 4, 6].into_iter().collect());
}

/// **Folding inward** is the codim-1 face: omit one slot. Every face of
/// `M(𝕆,𝕆,𝕆)` is `M(𝕆,𝕆) = 248`.
#[test]
fn folding_inward_gives_the_lower_folds() {
    let m3 = Position::octonionic(3).unwrap();
    let faces = m3.faces();
    assert_eq!(faces.len(), 3, "one face per slot omitted");
    for f in &faces {
        assert_eq!(f.arity(), 2);
        assert_eq!(f.dim().unwrap(), 248, "every face of M(O,O,O) is E8");
    }
    // a single-slot position has no faces to fold into
    assert!(Position::octonionic(1).unwrap().faces().is_empty());
}

/// **The interior is the n-body term no face can see**: `∏ tᵢ = 7ⁿ`, which
/// is exactly an n-site register of arity-7 qudits.
#[test]
fn the_interior_is_the_n_body_term_no_face_sees() {
    for n in 2..=5u32 {
        let p = Position::octonionic(n as usize).unwrap();
        assert_eq!(
            p.interior().unwrap(),
            7u128.pow(n),
            "interior must be 7^{n}"
        );
    }
    // and it is strictly bigger than any single face's whole dimension
    // once the fold is deep enough
    let m4 = Position::octonionic(4).unwrap();
    assert!(m4.interior().unwrap() < m4.dim().unwrap());
    assert_eq!(m4.interior().unwrap(), 2401);
}

/// **Folding outward in depth**: a position fed back in as a slot. The
/// first rung above `E₈` is `M(E₈,E₈) = 185,556`.
#[test]
fn folding_outward_reproduces_the_compound_tower() {
    let e8 = Position::algebras(&[O, O]).unwrap();
    assert_eq!(e8.dim().unwrap(), 248);
    assert_eq!(
        e8.derivation_layer().unwrap(),
        28,
        "g2 + g2, frame excluded"
    );

    let l1 = e8.nest(2).unwrap();
    assert_eq!(l1.depth(), 2);
    assert_eq!(l1.label(), "M(M(O,O),M(O,O))");
    assert_eq!(l1.dim().unwrap(), 185_556);

    let l2 = l1.nest(2).unwrap();
    assert_eq!(l2.depth(), 3);
    assert_eq!(l2.dim().unwrap(), 103_293_829_740);

    // the growth is doubly exponential: each rung is ~3x the square below
    let (a, b) = (l1.dim().unwrap(), l2.dim().unwrap());
    assert!(b > a * a, "the tower must square, not merely multiply");
    assert!(b < 4 * a * a);
}

/// **The tower refuses rather than overflowing.** It leaves `u128` at the
/// fourth rung, and a wrapped dimension would be worse than no dimension.
#[test]
fn the_tower_refuses_rather_than_overflowing() {
    let mut p = Position::algebras(&[O, O]).unwrap();
    let mut last_ok = 0usize;
    for rung in 1..=6 {
        p = p.nest(2).unwrap();
        match p.dim() {
            Ok(_) => last_ok = rung,
            Err(e) => {
                assert!(
                    e.to_string().contains("exceeds u128"),
                    "the refusal must name the limit, got: {e}"
                );
                assert!(rung >= 4, "u128 should carry at least three rungs");
                return;
            }
        }
    }
    panic!("the tower never refused; it reached rung {last_ok} silently");
}

/// **Hurwitz closes the slot axis.** Four algebras, and the derivation
/// ladder `0, 0, 3, 14` that grows to `g₂` and stops.
#[test]
fn hurwitz_closes_the_slot_axis() {
    assert_eq!(Composition::ALL.len(), 4);
    let dims: Vec<u128> = Composition::ALL.iter().map(|a| a.dim()).collect();
    assert_eq!(dims, vec![1, 2, 4, 8]);
    let ders: Vec<u128> = Composition::ALL.iter().map(|a| a.derivations()).collect();
    assert_eq!(ders, vec![0, 0, 3, 14], "Der grows to g2 and freezes");
    let ts: Vec<u128> = Composition::ALL.iter().map(|a| a.interior()).collect();
    assert_eq!(ts, vec![0, 1, 3, 7]);
}

/// **The grading needs no communication.** Two holders of the same
/// coordinates, building the position by different routes, agree on the
/// grading without exchanging anything.
#[test]
fn the_grading_is_a_function_of_position_alone() {
    // route A: built from algebras
    let a = Position::algebras(&[O, O, O]).unwrap();
    // route B: built slot by slot, in a different order of construction
    let b = Position::new(vec![Slot::Algebra(O), Slot::Algebra(O), Slot::Algebra(O)]).unwrap();
    assert_eq!(a.bott().unwrap(), b.bott().unwrap());
    assert_eq!(a.bott().unwrap(), octonionic_bott(3));
    assert_eq!(a.dim().unwrap(), b.dim().unwrap());

    // and the grading of a nested position is equally positional
    let deep = Position::algebras(&[O, O]).unwrap().nest(2).unwrap();
    assert_eq!(deep.bott().unwrap(), (185_556 % 8) as u8);
}
