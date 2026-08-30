//! Geometric qudits: levels read off the geometry, a position that
//! moves, holonomy from motion, and an interior no face can see.

use quantsim::backend::{CliffordStep, PauliString};
use quantsim::retro::{Code, SurfaceCode, ToricCode};
use quantsim::stitch::Volume;
use quantsim::volqudit::*;

fn ps(x: u64, z: u64) -> PauliString {
    PauliString {
        x,
        z,
        negative: false,
    }
}

/// The GHZ frame: `⟨Z₀Z₁, Z₁Z₂⟩`, one logical qubit on three.
fn ghz() -> VolQudit {
    let v = Volume::span(3, &[ps(0, 0b011), ps(0, 0b110)]).unwrap();
    VolQudit::new(3, v).unwrap()
}

fn from_code(n: usize, gens: Vec<PauliString>) -> VolQudit {
    VolQudit::new(n, Volume::span(n, &gens).unwrap()).unwrap()
}

// ───────────────── the level count is the geometry's ─────────────────

#[test]
fn levels_are_derived_from_the_obstruction() {
    // Nothing is told the qudit how many levels it has. `2^h` with
    // `h = n − r` comes out of the centraliser, and it agrees with
    // every code's known logical count.
    for (label, n, gens, want_h) in [
        ("free 1", 1usize, vec![], 1usize),
        ("free 4", 4, vec![], 4),
        ("GHZ", 3, vec![ps(0, 0b011), ps(0, 0b110)], 1),
        (
            "toric L=2",
            8,
            ToricCode::new(2, 0).unwrap().generators(),
            2,
        ),
        (
            "toric L=3",
            18,
            ToricCode::new(3, 0).unwrap().generators(),
            2,
        ),
        (
            "surface d=3",
            9,
            SurfaceCode::new(3, 0).unwrap().generators(),
            1,
        ),
        (
            "surface d=5",
            25,
            SurfaceCode::new(5, 0).unwrap().generators(),
            1,
        ),
    ] {
        let q = from_code(n, gens);
        assert_eq!(q.logical_rank(), want_h, "{label}: h");
        assert_eq!(q.levels(), 1u128 << want_h, "{label}: levels");
        // The boundary is exactly `2n − r`, and it contains the frame.
        assert_eq!(q.boundary().rank(), 2 * n - q.frame().rank(), "{label}");
        for g in q.frame().basis() {
            assert!(q.boundary().contains(g), "{label}: V ⊆ V^⊥");
        }
        // The conjugate pairs really are conjugate, and really are
        // logical: each anticommutes with its partner and commutes
        // with every frame element.
        for (i, &(x, z)) in q.logical_pairs().iter().enumerate() {
            assert!(!x.commutes_with(z), "{label}: pair {i} must anticommute");
            for g in q.frame().basis() {
                assert!(
                    x.commutes_with(g) && z.commutes_with(g),
                    "{label}: pair {i}"
                );
            }
        }
    }
}

#[test]
fn a_frame_that_is_not_a_position_is_refused() {
    // X₀ and Z₀ anticommute, so their span is no abelian subgroup: it
    // carves nothing and bounds nothing.
    let v = Volume::span(2, &[ps(0b01, 0), ps(0, 0b01)]).unwrap();
    let e = VolQudit::new(2, v).unwrap_err().to_string();
    assert!(e.contains("not isotropic"), "{e}");
    assert!(e.contains("bounds no"), "{e}");

    // And a frame from the wrong ambient is refused rather than padded.
    let v = Volume::span(3, &[ps(0, 0b011)]).unwrap();
    assert!(VolQudit::new(4, v).is_err());
}

#[test]
fn the_admissible_algebra_is_the_small_one() {
    // Naming an admissible operation costs 2h bits; naming a general
    // element of End on the same level space costs 4^h complex
    // parameters. The gap is the theory's `A_V ⊊ End(V)`, forced by
    // the geometry rather than imposed.
    let q = VolQudit::free(8).unwrap();
    let c = q.operative_cost();
    assert_eq!(c.levels, 256);
    assert_eq!(c.admissible_bits, 16);
    assert_eq!(c.full_end_params, 65_536);
    assert!(c.admissible_bits as u128 * 4000 < c.full_end_params);

    // A code qudit: a large frame, a tiny admissible algebra.
    let q = from_code(25, SurfaceCode::new(5, 0).unwrap().generators());
    let c = q.operative_cost();
    assert_eq!(c.frame_bits, 24 * 50);
    assert_eq!(c.admissible_bits, 2);
}

// ─────────────────────── position, motion, holonomy ───────────────────────

#[test]
fn transport_moves_the_qudit_and_preserves_what_it_is() {
    // A Clifford is a symplectic map, so it carries flats to flats.
    // The frame moves; the signature does not. That is the covariance
    // condition, asserted rather than assumed.
    let q = ghz();
    let sig = q.signature();
    for steps in [
        vec![CliffordStep::Cx(0, 1)],
        vec![CliffordStep::H(2)],
        vec![
            CliffordStep::S(0),
            CliffordStep::Cx(1, 2),
            CliffordStep::H(0),
        ],
    ] {
        let moved = q.transport(&steps).unwrap();
        assert_eq!(
            moved.signature(),
            sig,
            "signature is invariant under motion"
        );
        assert_eq!(moved.levels(), q.levels());
    }
    // At least one of those genuinely moved it.
    let moved = q.transport(&[CliffordStep::H(0)]).unwrap();
    assert!(
        !moved.frame().is_same(q.frame()),
        "the qudit actually moved"
    );
}

#[test]
fn a_loop_returns_the_frame_and_need_not_return_the_qudit() {
    // The knot-register: move the qudit around and bring it back, and
    // what it bounds has been acted on.
    let q = VolQudit::free(1).unwrap();
    // A free frame is fixed by everything, so every word is a loop.
    assert!(q.closes(&[CliffordStep::H(0)]).unwrap());
    let h = q.holonomy(&[CliffordStep::H(0)]).unwrap();
    assert!(!h.is_flat(), "H swaps X̄ and Z̄");
    assert_eq!(h.order(8), Some(2));
    assert_eq!(h.matrix(), &[vec![false, true], vec![true, false]]);

    // S is also a loop, of order 2 on the F₂ action, and different.
    let s = q.holonomy(&[CliffordStep::S(0)]).unwrap();
    assert!(!s.is_flat());
    assert_ne!(s, h);

    // HS has order 3 — the full SL(2,𝔽₂) ≅ S₃ shows up.
    let orders: Vec<Option<usize>> = q
        .holonomy_search(&[CliffordStep::H(0), CliffordStep::S(0)], 3)
        .iter()
        .map(|g| g.order(12))
        .collect();
    assert!(orders.contains(&Some(3)), "{orders:?}");
    assert!(orders.contains(&Some(2)), "{orders:?}");

    // Flat loops report zero curvature and are excluded from the search.
    let flat = q
        .holonomy(&[CliffordStep::H(0), CliffordStep::H(0)])
        .unwrap();
    assert!(flat.is_flat());
    assert_eq!(flat.curvature(), 0);
}

#[test]
fn a_constrained_qudit_has_loops_too() {
    // The GHZ frame is fixed by far fewer words, and the search finds
    // the ones that survive rather than assuming any exist.
    let q = ghz();
    let gens = [
        CliffordStep::Cx(0, 1),
        CliffordStep::Cx(1, 2),
        CliffordStep::Cx(1, 0),
        CliffordStep::S(0),
        CliffordStep::S(1),
        CliffordStep::S(2),
    ];
    let found = q.holonomy_search(&gens, 3);
    assert!(!found.is_empty(), "a loop with curvature exists here");
    for g in &found {
        assert!(!g.is_flat());
        assert!(g.curvature() > 0);
        assert_eq!(g.logical_rank(), 1);
        assert!(g.order(16).is_some(), "a finite-order logical action");
    }
}

#[test]
fn holonomy_of_an_open_path_is_refused() {
    let q = ghz();
    let open = [CliffordStep::H(0)];
    assert!(!q.closes(&open).unwrap());
    let e = q.holonomy(&open).unwrap_err().to_string();
    assert!(e.contains("does not return the frame"), "{e}");
}

// ────────────────────────── frame in frame ──────────────────────────

#[test]
fn a_qudit_nests_inside_another_qudits_level_space() {
    let outer = VolQudit::free(6).unwrap();
    assert_eq!(outer.logical_rank(), 6);
    let inner = outer
        .nest(Volume::span(6, &[ps(0, 0b000011), ps(0, 0b001100)]).unwrap())
        .unwrap();
    assert_eq!(
        inner.ambient(),
        6,
        "the inner ambient IS the outer level space"
    );
    assert_eq!(inner.logical_rank(), 4);
    assert_eq!(inner.levels(), 16);
    // The inner frame is priced in the outer's logical coordinates, so
    // it costs O(h) rather than O(n): the outer geometry already paid.
    assert_eq!(inner.operative_cost().frame_bits, 2 * 2 * 6);

    // Nesting refuses a frame of the wrong width rather than padding.
    let small = VolQudit::free(2).unwrap();
    assert!(small.nest(Volume::span(5, &[ps(0, 1)]).unwrap()).is_err());
}

#[test]
fn promotion_exposes_the_interface_and_nothing_else() {
    // Two structurally different frames with the same signature promote
    // to equal atoms — which is what makes the outer level able to
    // ignore the interior.
    let a = from_code(9, SurfaceCode::new(3, 0).unwrap().generators());
    let b = VolQudit::new(
        9,
        Volume::span(9, &(0..8).map(|q| ps(0, 1 << q)).collect::<Vec<_>>()).unwrap(),
    )
    .unwrap();
    assert!(!a.frame().is_same(b.frame()), "genuinely different frames");
    assert_eq!(a.signature(), b.signature());
    assert_eq!(a.promote(), b.promote());
    assert_eq!(a.promote().levels(), 2);

    // And a qudit with a different logical rank does not collide.
    let c = from_code(8, ToricCode::new(2, 0).unwrap().generators());
    assert_ne!(a.promote(), c.promote());
}

// ──────────────────── the interior no face can see ────────────────────

#[test]
fn independent_qudits_have_no_interior() {
    // The control. Three free qubits in three slots: every logical
    // class has a single-slot representative, so every face sees it,
    // the skeleton is all of `V^⊥/V`, and the vol is empty.
    let c = CompoundQudit::new(VolQudit::free(3).unwrap(), &[(0, 1), (1, 1), (2, 1)]).unwrap();
    assert_eq!(c.qudit().logical_rank(), 3);
    for i in 0..3 {
        assert_eq!(
            c.face_rank(i).unwrap(),
            4,
            "face {i} sees the other two slots"
        );
    }
    assert_eq!(c.skeleton_rank().unwrap(), 6);
    assert_eq!(
        c.interior_rank().unwrap(),
        0,
        "nothing is irreducibly 3-body"
    );
    assert!(!c.is_brunnian().unwrap());
}

#[test]
fn the_ghz_frame_has_an_interior_of_exactly_one() {
    // `Z̄` can be rewritten onto a single qubit by multiplying in a
    // stabilizer, so it is visible on every other face. `X̄ = XXX`
    // cannot be rewritten off any qubit at all. One of the two logical
    // dimensions is therefore irreducibly 3-body, and that is the vol.
    let c = CompoundQudit::new(ghz(), &[(0, 1), (1, 1), (2, 1)]).unwrap();
    assert_eq!(c.qudit().logical_rank(), 1);
    for i in 0..3 {
        assert_eq!(c.face_rank(i).unwrap(), 1);
    }
    assert_eq!(c.skeleton_rank().unwrap(), 1);
    assert_eq!(c.interior_rank().unwrap(), 1);
    // Not Brunnian: the faces are not empty, they are just incomplete.
    assert!(!c.is_brunnian().unwrap());
}

#[test]
fn a_code_split_across_slots_carries_its_interior_with_it() {
    let q = from_code(8, ToricCode::new(2, 0).unwrap().generators());
    let c = CompoundQudit::new(q, &[(0, 4), (4, 4)]).unwrap();
    assert_eq!(c.qudit().logical_rank(), 2);
    assert_eq!(
        c.interior_rank().unwrap(),
        2,
        "half the logical space needs both halves"
    );
    assert!(c.interior_rank().unwrap() <= 2 * c.qudit().logical_rank());
}

#[test]
fn a_compound_needs_slots_that_partition() {
    let q = VolQudit::free(4).unwrap();
    assert!(
        CompoundQudit::new(q.clone(), &[(0, 4)]).is_err(),
        "one slot poses nothing"
    );
    assert!(
        CompoundQudit::new(q, &[(0, 3), (2, 2)]).is_err(),
        "overlapping slots are refused"
    );
}

#[test]
fn every_quantity_is_polynomial_in_the_ambient() {
    // The whole module is 𝔽₂ mask algebra: a 40-qubit ambient with a
    // 2^36-level qudit is a few hundred bytes and microseconds, and
    // nothing anywhere materializes 2^n.
    let n = 40;
    let gens: Vec<PauliString> = (0..4).map(|q| ps(0, 1 << q)).collect();
    let t0 = std::time::Instant::now();
    let q = from_code(n, gens);
    let c = CompoundQudit::new(q.clone(), &[(0, 20), (20, 20)]).unwrap();
    let interior = c.interior_rank().unwrap();
    let elapsed = t0.elapsed();
    assert_eq!(q.logical_rank(), 36);
    assert_eq!(q.levels(), 1u128 << 36);
    assert_eq!(interior, 0, "an unentangled split has no interior");
    assert!(elapsed.as_millis() < 500, "{elapsed:?}");
    assert!(q.operative_cost().frame_bits < 1000);
}
