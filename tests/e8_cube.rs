//! E8 as a 2×2×2 cube volume, measured: the eight ambient coordinates
//! arranged on the cube's eight vertices, the lattice condition read as
//! a parity law on the volume, the cube's symmetry group inside
//! `W(E8)`, the scale tower as a cube of cubes, and inward/outward
//! interaction — a volume participating in its own interior to a
//! finite, measured depth.

mod common;

use quantsim::e8::constellation::{self, E8ConstellationState};
use quantsim::e8::cube;
use quantsim::prelude::*;

// ── the volume ───────────────────────────────────────────────────────

#[test]
fn eight_coordinates_are_the_eight_vertices_of_one_cube() {
    let edges = cube::edges();
    assert_eq!(edges.len(), 12);
    // Every edge joins vertices differing in exactly one bit.
    for (a, b) in &edges {
        assert_eq!((a ^ b).count_ones(), 1);
    }
    // Three axes, four parallel edges each, together covering all 12.
    let mut all = Vec::new();
    for axis in 0..3 {
        let pairs = cube::axis_pairs(axis).unwrap();
        assert_eq!(pairs.len(), 4);
        all.extend(pairs);
    }
    all.sort_unstable();
    assert_eq!(all, edges);
    assert!(cube::axis_pairs(3).is_err());
}

#[test]
fn the_lattice_condition_is_a_parity_law_on_the_volume() {
    // Every root lives wholly on the cube or wholly on the body-centred
    // copy — never mixed, which is exactly what E8 forbids.
    let (on_cube, body_centred) = cube::sector_census();
    assert_eq!((on_cube, body_centred), (112, 128));
    assert_eq!(on_cube + body_centred, 240);

    // And every root's eight vertex values sum to a multiple of four.
    for r in quantsim::e8::roots() {
        let p: [i64; 8] = std::array::from_fn(|k| i64::from(r[k]));
        assert_eq!(
            cube::vertex_parity(&p),
            0,
            "root {r:?} breaks the parity law"
        );
        assert!(cube::sector(&p).is_some());
    }

    // A vector with mixed vertex parities is in neither sector — the
    // law rejecting a non-lattice volume.
    let mixed = [1i64, 2, 0, 0, 0, 0, 0, 0];
    assert!(cube::sector(&mixed).is_none());
}

#[test]
fn the_cubes_symmetry_group_sits_inside_the_lattices() {
    // Eight vertex translations times six axis relabelings.
    assert_eq!(cube::cube_symmetry_order(), (48, 48));

    // Each translation is a symmetry of the volume and of the lattice.
    for shift in 0u8..8 {
        let perm = cube::translation(shift);
        assert!(cube::preserves_cube(&perm));
        assert!(cube::is_lattice_automorphism(&perm));
    }

    // An axis relabeling is a cube symmetry; a shear is invertible on
    // the vertex labels but moves edges to diagonals.
    let swap_axes = cube::linear_map([2, 1, 4]).unwrap();
    assert!(cube::preserves_cube(&swap_axes));
    let shear = cube::linear_map([1, 3, 4]).unwrap();
    assert!(cube::is_lattice_automorphism(&shear));
    assert!(!cube::preserves_cube(&shear));

    // A dependent triple is not invertible at all.
    assert!(cube::linear_map([1, 2, 3]).is_none());
}

#[test]
fn cube_symmetries_act_on_lattice_states() {
    let mut state = E8ConstellationState::new(16).unwrap();
    state
        .load(&[(0, C64::new(0.6, 0.0)), (0x0105, C64::new(0.8, 0.0))])
        .unwrap();
    let before = state.total_abs_sqr();

    let perm = cube::translation(5);
    state.permute_coordinates(&perm).unwrap();
    common::assert_close(state.total_abs_sqr(), before, 1e-12);

    // A vertex translation is an involution: applying it twice returns
    // the state exactly.
    let mut twice = E8ConstellationState::new(16).unwrap();
    twice
        .load(&[(0, C64::new(0.6, 0.0)), (0x0105, C64::new(0.8, 0.0))])
        .unwrap();
    state.permute_coordinates(&perm).unwrap();
    assert!(max_amplitude_deviation(&state, &twice) < 1e-15);

    // Not a permutation, and the register's blocks must be whole.
    assert!(state
        .permute_coordinates(&[0, 0, 1, 2, 3, 4, 5, 6])
        .is_err());
    let mut partial = E8ConstellationState::new(12).unwrap();
    assert!(partial.permute_coordinates(&perm).is_err());
}

// ── the tower: a cube of cubes ───────────────────────────────────────

#[test]
fn a_point_is_a_cube_whose_every_vertex_is_a_cube() {
    let digits = [3u8, 17, 200];
    let p = cube::from_tower(&digits);
    assert_eq!(cube::tower(&p, 3).unwrap(), digits.to_vec());
    // Its eight vertex values are what the volume reads.
    let values = cube::vertex_values(&p).unwrap();
    assert_eq!(values.len(), 8);
    // Doubling the point prepends an empty finest cube: the
    // self-similarity of the tower, read as recursion.
    let doubled: [i64; 8] = p.map(|c| c << 1);
    let mut expected = vec![0u8];
    expected.extend_from_slice(&digits);
    assert_eq!(cube::tower(&doubled, 4).unwrap(), expected);
}

#[test]
fn the_elementary_displacements_are_lattice_vectors_at_every_scale() {
    for vertex in 0..8 {
        for level in 0..4 {
            let s = cube::step(vertex, level).unwrap();
            assert!(constellation::class_of(&s).is_some());
            // It is the vertex direction and nothing else.
            for (k, &c) in s.iter().enumerate() {
                assert_eq!(c != 0, k == vertex);
            }
        }
    }
    assert!(cube::step(8, 0).is_err());

    // Inward is one scale finer, outward one scale coarser.
    assert_eq!(cube::inward(2, 0).unwrap(), cube::step(0, 1).unwrap());
    assert_eq!(cube::outward(2, 0).unwrap(), cube::step(0, 3).unwrap());
    // The finest volume has no interior, and says so.
    let err = cube::inward(0, 0).unwrap_err();
    assert!(format!("{err}").contains("no interior"));
}

// ── inward and outward ───────────────────────────────────────────────

#[test]
fn different_cube_vertices_are_exactly_independent_channels() {
    // The ambient coordinates are orthogonal, so a displacement at one
    // vertex and a modulation at another commute exactly — at every
    // pair of scales.
    for j in 0..3 {
        for k in 0..3 {
            for other in [1usize, 3, 7] {
                let i = cube::interaction(24, (j, 0), (k, other)).unwrap();
                assert!(i.commutes);
                assert!(i.coupling < 1e-14, "{}", i.coupling);
            }
        }
    }
}

#[test]
fn a_volume_participates_in_its_own_interior_to_a_finite_measured_depth() {
    // Depth-3 tower: only the very finest pair still interacts.
    let ladder = cube::interaction_ladder(24, 0).unwrap();
    assert_eq!(ladder.len(), 9);
    for i in &ladder {
        let separation = i.translate_level + i.modulate_level;
        // The measured horizon: interaction survives exactly while
        // j + k < m − 2.
        assert_eq!(!i.commutes, separation < 3 - 2, "levels {separation}");
    }
    // And where it does interact, the phase is an honest −1.
    let close = ladder.iter().find(|i| !i.commutes).unwrap();
    common::assert_close(close.phase.re, -1.0, 1e-12);
    common::assert_close(close.phase.im, 0.0, 1e-12);
    common::assert_close(close.coupling, 2.0, 1e-12);
}

#[test]
fn deeper_participation_is_finer_participation() {
    // Depth-5 tower: the interaction is not one strength but a ladder
    // of phases, `exp(2πi · 2^{j+k+2−m})` — exactly −1 on the horizon,
    // halving in angle for every level further inside.
    let m = 5usize;
    for i in cube::interaction_ladder(8 * m, 0).unwrap() {
        let exponent = i.translate_level as i64 + i.modulate_level as i64 + 2 - m as i64;
        let turns = 2f64.powi(exponent as i32);
        if exponent >= 0 {
            assert!(i.commutes, "levels {exponent} should commute exactly");
            continue;
        }
        // The sign follows the crate's modulation character convention;
        // the magnitude law below is convention-free.
        let angle = -std::f64::consts::TAU * turns;
        common::assert_close(i.phase.re, angle.cos(), 1e-12);
        common::assert_close(i.phase.im, angle.sin(), 1e-12);
        // |e^{iθ} − 1| = 2|sin(θ/2)|
        common::assert_close(i.coupling, 2.0 * (angle / 2.0).sin().abs(), 1e-12);
    }
    // On the horizon it is a half turn; one level inside, a quarter.
    let horizon = cube::interaction(8 * m, (0, 0), (m - 3, 0)).unwrap();
    common::assert_close(horizon.coupling, 2.0, 1e-12);
    let inside = cube::interaction(8 * m, (0, 0), (m - 4, 0)).unwrap();
    common::assert_close(inside.coupling, std::f64::consts::SQRT_2, 1e-12);
}

#[test]
fn the_self_reference_depth_grows_with_the_tower_and_is_reported_when_absent() {
    // m = 2: nothing in the volume can reach itself at all.
    assert_eq!(cube::self_reference_depth(16, 0).unwrap(), None);
    // m = 3, 4, 5: the horizon opens one level at a time.
    assert_eq!(cube::self_reference_depth(24, 0).unwrap(), Some(0));
    assert_eq!(cube::self_reference_depth(32, 0).unwrap(), Some(1));
    assert_eq!(cube::self_reference_depth(40, 0).unwrap(), Some(2));
}

#[test]
fn the_measured_commutator_is_cross_checked_against_the_lattice_pairing() {
    // `interaction` refuses if the phase read off the evolved states
    // disagrees with the exact integer pairing, so a passing sweep is a
    // measurement agreeing with arithmetic, not either one alone.
    for j in 0..5 {
        for k in 0..5 {
            let i = cube::interaction(40, (j, 2), (k, 2)).unwrap();
            assert_eq!(i.translate_level, j);
            assert_eq!(i.modulate_level, k);
            assert!(i.commutes == (i.coupling < 1e-12));
        }
    }
}
