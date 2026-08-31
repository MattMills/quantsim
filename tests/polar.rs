//! The embedding — `PG(2n−1,2)` and its polar space `W(2n−1,2)` — and
//! the effective geometry an assembly of frames closes onto.

use quantsim::backend::{CliffordStep, PauliString};
use quantsim::polar::*;
use quantsim::retro::{Code, SurfaceCode, ToricCode};
use quantsim::stitch::Volume;
use quantsim::volqudit::VolQudit;

fn ps(x: u64, z: u64) -> PauliString {
    PauliString {
        x,
        z,
        negative: false,
    }
}

// ───────────────────── the ambient, named and checked ─────────────────────

#[test]
fn the_closed_forms_are_what_enumeration_finds() {
    for n in 1..=4usize {
        let w = PolarSpace::new(n).unwrap();
        assert_eq!(w.homogeneous_coordinates(), 2 * n);
        assert_eq!(w.projective_dimension(), 2 * n - 1);
        assert_eq!(w.points(), (1u128 << (2 * n)) - 1);

        let pts = w.enumerate_points().unwrap();
        assert_eq!(pts.len() as u128, w.points(), "n = {n}: point count");

        let lines = w.enumerate_lines().unwrap();
        assert_eq!(
            lines.len() as u128,
            w.totally_isotropic_lines(),
            "n = {n}: line count against the closed form"
        );
        for l in &lines {
            assert_eq!(l.len(), 3, "a projective line over 𝔽₂ has three points");
        }
        // The degree is constant and matches `2^{2n−2} − 1`.
        for p in pts.iter().take(8) {
            let deg = lines
                .iter()
                .filter(|l| l.iter().any(|q| (q.x, q.z) == (p.x, p.z)))
                .count();
            assert_eq!(deg as u128, w.lines_through_a_point(), "n = {n}");
        }
    }
}

#[test]
fn two_qubits_is_the_doily() {
    let w = PolarSpace::new(2).unwrap();
    assert!(w.is_doily());
    assert_eq!(
        w.homogeneous_coordinates(),
        4,
        "four homogeneous coordinates"
    );
    assert_eq!(w.points(), 15);
    assert_eq!(w.totally_isotropic_lines(), 15);
    assert_eq!(w.lines_through_a_point(), 3);
    assert_eq!(
        w.generators(),
        15,
        "the maximal commuting sets are the lines"
    );

    // The two properties that earn the name.
    assert_eq!(
        w.triangles().unwrap(),
        0,
        "a generalized quadrangle has none"
    );
    assert!(w.gq_axiom().unwrap(), "GQ(2,2)");
}

#[test]
fn wider_registers_have_triangles_and_are_not_quadrangles() {
    // The doily is special. `W(5,2)` is a polar space too, but it is
    // not a generalized quadrangle, and saying so keeps the claim from
    // spreading past where it was checked.
    let w = PolarSpace::new(3).unwrap();
    assert!(!w.is_doily());
    assert!(w.triangles().unwrap() > 0);
    assert!(!w.gq_axiom().unwrap());
}

#[test]
fn a_wider_register_is_a_combinatorial_space_of_doilies() {
    assert_eq!(PolarSpace::new(1).unwrap().doilies().unwrap(), 0);
    assert_eq!(
        PolarSpace::new(2).unwrap().doilies().unwrap(),
        1,
        "two qubits is exactly one doily"
    );
    let three = PolarSpace::new(3).unwrap().doilies().unwrap();
    assert_eq!(three, 336, "three qubits carries 336 overlapping doilies");
    // The enumeration is the check on the closed form, at both sizes
    // where enumeration is affordable.
    for n in 2..=3usize {
        let w = PolarSpace::new(n).unwrap();
        assert_eq!(
            w.doilies().unwrap() as u128,
            w.doily_count(),
            "n = {n}: enumeration against the closed form"
        );
    }
    // And the closed form answers where enumeration cannot.
    assert_eq!(PolarSpace::new(4).unwrap().doily_count(), 91_392);
    assert_eq!(PolarSpace::new(5).unwrap().doily_count(), 23_744_512);
    // Past the walk's cap it refuses rather than grinding.
    assert!(PolarSpace::new(6).unwrap().doilies().is_err());
    assert!(PolarSpace::new(6).unwrap().enumerate_points().is_err());
    // But the closed forms answer at any width.
    let big = PolarSpace::new(40).unwrap();
    assert_eq!(big.points(), (1u128 << 80) - 1);
    // Past u128 these saturate rather than wrapping: an honest "too
    // large to name" instead of a wrong number.
    assert_eq!(big.generators(), u128::MAX);
    assert_eq!(big.doily_count(), u128::MAX);
    assert!(PolarSpace::new(12).unwrap().doily_count() < u128::MAX);
}

#[test]
fn generators_are_the_maximal_commuting_sets() {
    // `∏(2^i + 1)`, and at n = 2 the generators are exactly the lines.
    for (n, want) in [(1usize, 3u128), (2, 15), (3, 135), (4, 2295)] {
        assert_eq!(PolarSpace::new(n).unwrap().generators(), want, "n = {n}");
    }
    // A maximal isotropic Volume is one of them, and it carves a point.
    let v = Volume::span(3, &[ps(0, 0b001), ps(0, 0b010), ps(0, 0b100)]).unwrap();
    assert!(v.is_maximal_isotropic());
    assert_eq!(v.carves(), 1);
}

// ────────────────────── every frame is a flat of it ──────────────────────

#[test]
fn the_codes_are_flats_of_the_polar_space() {
    // The connection the module exists to make: a stabilizer code is a
    // totally isotropic flat, its points are points of PG(2n−1,2), and
    // the qudit it carries is what the flat's perp bounds.
    for (label, n, gens) in [
        (
            "toric L=2",
            8usize,
            ToricCode::new(2, 0).unwrap().generators(),
        ),
        (
            "surface d=3",
            9,
            SurfaceCode::new(3, 0).unwrap().generators(),
        ),
    ] {
        let v = Volume::span(n, &gens).unwrap();
        assert!(v.is_isotropic(), "{label} is a flat of W({},2)", 2 * n - 1);
        let w = PolarSpace::new(n).unwrap();
        // The flat's own points, as a projective subspace.
        let flat_points = (1u128 << v.rank()) - 1;
        assert!(flat_points < w.points(), "{label}: a proper flat");
        let q = VolQudit::new(n, v).unwrap();
        assert_eq!(q.levels(), 1u128 << (n - q.frame().rank()));
    }
}

// ───────────────────────── effective geometry ─────────────────────────

fn frame(n: usize, gens: &[PauliString]) -> Volume {
    Volume::span(n, gens).unwrap()
}

#[test]
fn the_closure_is_the_constraint_intersection_and_it_is_idempotent() {
    let a = Assembly::new(
        4,
        vec![
            frame(4, &[ps(0, 0b0011)]),
            frame(4, &[ps(0, 0b0110)]),
            frame(4, &[ps(0b1111, 0)]),
        ],
    )
    .unwrap();
    assert!(a.is_idempotent().unwrap(), "Γ² = Γ");
    // Im Γ = (⋁ Vᵢ)^⊥, and it contains every frame it was built from.
    let c = a.closure().unwrap();
    for f in a.frames() {
        for p in f.basis() {
            assert!(c.contains(p), "a frame is admissible for itself");
        }
    }
    assert_eq!(c.rank(), 2 * 4 - a.joined().unwrap().rank());
}

#[test]
fn the_effective_dimension_is_not_the_sum_of_the_parts() {
    // Three single-constraint frames on four qubits. Each alone leaves
    // h = 3; together they leave far less, and the deficit is what
    // closing them destroys.
    let a = Assembly::new(
        4,
        vec![
            frame(4, &[ps(0, 0b0011)]),
            frame(4, &[ps(0, 0b0110)]),
            frame(4, &[ps(0, 0b1100)]),
        ],
    )
    .unwrap();
    assert_eq!(a.local_ranks().unwrap(), vec![3, 3, 3]);
    // Three independent Z-parities, so the join has rank 3 and only one
    // logical direction survives all of them.
    assert_eq!(a.joined().unwrap().rank(), 3);
    assert_eq!(a.effective_rank().unwrap(), 1);
    assert_eq!(
        a.closure_deficit().unwrap(),
        9 - 1,
        "closing costs 8 of the 9"
    );
    // And the throat bounds it.
    assert!(a.effective_rank().unwrap() <= a.throat().unwrap());
}

#[test]
fn the_throat_bound_is_tight_exactly_on_a_nested_chain() {
    // Nested: each constraint contains the last, so the narrowest frame
    // already decides, and the bound is an equality.
    let nested = Assembly::new(
        4,
        vec![
            frame(4, &[ps(0, 0b0001)]),
            frame(4, &[ps(0, 0b0001), ps(0, 0b0010)]),
            frame(4, &[ps(0, 0b0001), ps(0, 0b0010), ps(0, 0b0100)]),
        ],
    )
    .unwrap();
    assert!(nested.is_nested());
    assert_eq!(nested.local_ranks().unwrap(), vec![3, 2, 1]);
    assert_eq!(nested.throat().unwrap(), 1);
    assert_eq!(
        nested.effective_rank().unwrap(),
        nested.throat().unwrap(),
        "the hourglass law, where it actually holds"
    );

    // Not nested: the bound holds and is slack, because the frames
    // constrain different directions and closing them costs more than
    // the narrowest alone.
    let crossed = Assembly::new(
        4,
        vec![
            frame(4, &[ps(0, 0b0011), ps(0, 0b1100)]),
            frame(4, &[ps(0b0011, 0), ps(0b1100, 0)]),
        ],
    )
    .unwrap();
    assert!(!crossed.is_nested());
    assert!(
        crossed.effective_rank().unwrap() < crossed.throat().unwrap(),
        "eff {} vs throat {}",
        crossed.effective_rank().unwrap(),
        crossed.throat().unwrap()
    );
}

#[test]
fn a_local_invariant_need_not_be_globally_effective() {
    // Two assemblies with the *same* closure. The effective rank agrees
    // — it factors through Γ. The sum of the local ranks does not, so
    // it is a local invariant and the closure erases it.
    let x = Assembly::new(
        3,
        vec![frame(3, &[ps(0, 0b011)]), frame(3, &[ps(0, 0b110)])],
    )
    .unwrap();
    let y = Assembly::new(3, vec![frame(3, &[ps(0, 0b011), ps(0, 0b110)])]).unwrap();
    assert!(x.closure().unwrap().is_same(&y.closure().unwrap()));

    let eff = |a: &Assembly| a.effective_rank();
    assert_eq!(x.agrees_on_closure(&y, &eff).unwrap(), Some(true));

    let local = |a: &Assembly| -> quantsim::Result<usize> { Ok(a.local_ranks()?.iter().sum()) };
    assert_eq!(
        x.agrees_on_closure(&y, &local).unwrap(),
        Some(false),
        "Σ hᵢ is local: 2+2 against 1"
    );

    // Assemblies with different closures are not comparable at all, and
    // that is reported rather than answered.
    let z = Assembly::new(3, vec![frame(3, &[ps(0b111, 0)])]).unwrap();
    assert_eq!(x.agrees_on_closure(&z, &eff).unwrap(), None);
}

#[test]
fn dynamic_sufficiency_is_the_loop_predicate_with_a_number() {
    let v = frame(3, &[ps(0, 0b011), ps(0, 0b110)]);
    let q = VolQudit::new(3, v.clone()).unwrap();

    // A transport that fixes the frame: Φ = 0, and `closes` agrees.
    let loop_steps = [
        CliffordStep::Cx(0, 1),
        CliffordStep::Cx(1, 0),
        CliffordStep::Cx(0, 1),
    ];
    let phi = dynamic_obstruction(&v, &loop_steps).unwrap();
    assert_eq!(phi == 0, q.closes(&loop_steps).unwrap());

    // One that does not: Φ counts the directions the frame is missing.
    let open = [CliffordStep::H(0)];
    let phi = dynamic_obstruction(&v, &open).unwrap();
    assert!(phi > 0);
    assert!(!q.closes(&open).unwrap());
    assert!(phi <= v.rank(), "at most the whole frame can leave");

    // The identity transport is flat by construction.
    assert_eq!(dynamic_obstruction(&v, &[]).unwrap(), 0);
}

#[test]
fn an_assembly_refuses_what_is_not_a_position() {
    assert!(Assembly::new(2, vec![]).is_err());
    let bad = frame(2, &[ps(0b01, 0), ps(0, 0b01)]);
    let e = Assembly::new(2, vec![bad]).unwrap_err().to_string();
    assert!(e.contains("not isotropic"), "{e}");
    assert!(Assembly::new(3, vec![frame(4, &[ps(0, 1)])]).is_err());
}
