//! Recursive quantum systems, measured: a site that is either a point
//! or an entire lattice of the same kind; the block-spin flow that
//! decides whether the substitution means anything; the harmonic
//! lattice where it is exact; the same statement executed on a real
//! backend; and a computation that participates in itself.

mod common;

use quantsim::prelude::*;
use quantsim::recursive::*;

// ── the structure ────────────────────────────────────────────────────

#[test]
fn four_qubits_in_a_square_then_the_same_structure_again() {
    let flat = RecursiveLattice::new(Shape::SQUARE).unwrap();
    assert_eq!(flat.width(), 4);
    assert_eq!(flat.depth(), 1);
    assert_eq!(flat.bond_pairs(), vec![(0, 1), (0, 3), (1, 2), (2, 3)]);

    // The same structure again, bonded laterally.
    let nested = RecursiveLattice::nest(Shape::SQUARE, 2).unwrap();
    assert_eq!(nested.width(), 16);
    assert_eq!(nested.depth(), 2);
    let bonds = nested.bonds();
    // Four internal squares of four bonds, plus four lateral bonds
    // joining the blocks corner to corner.
    assert_eq!(bonds.len(), 20);
    assert_eq!(bonds.iter().filter(|b| b.depth == 1).count(), 16);
    let lateral: Vec<(usize, usize)> = bonds
        .iter()
        .filter(|b| b.depth == 0)
        .map(|b| (b.a, b.b))
        .collect();
    assert_eq!(lateral, vec![(1, 4), (3, 12), (6, 9), (11, 14)]);

    // Every lateral bond leaves its block: the blocks really are joined
    // to each other, not to themselves.
    let blocks = nested.blocks(1);
    assert_eq!(blocks, vec![(0, 4), (4, 8), (8, 12), (12, 16)]);
    for (a, b) in lateral {
        let block_of = |q: usize| blocks.iter().position(|&(lo, hi)| (lo..hi).contains(&q));
        assert_ne!(block_of(a), block_of(b));
    }
}

#[test]
fn the_recursion_is_scale_free() {
    // Three levels: a square of squares of squares.
    let deep = RecursiveLattice::nest(Shape::SQUARE, 3).unwrap();
    assert_eq!(deep.width(), 64);
    assert_eq!(deep.depth(), 3);
    // 16 innermost squares (4 bonds) + 4 mid-level lateral (4 bonds
    // each) + 4 top-level lateral.
    let bonds = deep.bonds();
    assert_eq!(bonds.iter().filter(|b| b.depth == 2).count(), 64);
    assert_eq!(bonds.iter().filter(|b| b.depth == 1).count(), 16);
    assert_eq!(bonds.iter().filter(|b| b.depth == 0).count(), 4);
    assert_eq!(bonds.len(), 84);
}

#[test]
fn the_lattice_is_a_device_geometry_like_any_other() {
    let nested = RecursiveLattice::nest(Shape::SQUARE, 2).unwrap();
    let t = nested.topology().unwrap();
    assert_eq!(t.num_sites(), 16);
    assert_eq!(t.num_edges(), 20);
    assert!(t.is_connected());
    assert_eq!(t.diameter(), 6);

    // And it runs a circuit through the real device backend, answering
    // in logical indices.
    let mut device = DeviceState::with_latency(
        t,
        LatencyMap::uniform(DurationModel::ibm_falcon_like()),
        ArityPolicy::default(),
        Box::new(SparseState::<C64>::new(16).unwrap()),
    )
    .unwrap();
    let mut c: Circuit = Circuit::new(16);
    c.h(0);
    for q in 0..15 {
        c.cx(q, q + 1);
    }
    c.bind(&GateRegistry::standard())
        .unwrap()
        .run(&mut device)
        .unwrap();
    // A 16-qubit GHZ across the recursive fabric.
    common::assert_close(device.probability(0), 0.5, 1e-9);
    common::assert_close(device.probability((1 << 16) - 1), 0.5, 1e-9);
}

#[test]
fn refine_and_coarsen_are_structural_inverses_and_refuse_nonsense() {
    let mut l = RecursiveLattice::new(Shape::SQUARE).unwrap();
    let before = l.clone();
    l.refine(&[1], Shape::SQUARE).unwrap();
    assert_eq!(l.width(), 7);

    // Refining an existing lattice is refused rather than silently
    // nesting one more level.
    let err = l.refine(&[1], Shape::SQUARE).unwrap_err();
    assert!(format!("{err}").contains("already a lattice"));

    // A point has no sub-sites to address.
    let err = l.refine(&[0, 2], Shape::SQUARE).unwrap_err();
    assert!(format!("{err}").contains("no sub-sites"));

    l.coarsen(&[1]).unwrap();
    assert_eq!(l, before);
    assert!(l.coarsen(&[1]).is_err());
}

#[test]
fn the_cube_shape_is_the_e8_volume() {
    // The 2×2×2 cube as a lattice block agrees, edge for edge, with the
    // cube the eight E8 coordinates are arranged on.
    assert_eq!(Shape::CUBE.arity(), 8);
    assert_eq!(Shape::CUBE.bonds(), quantsim::e8::cube::edges());

    let nested = RecursiveLattice::nest(Shape::CUBE, 2).unwrap();
    assert_eq!(nested.width(), 64);
    // Eight inner cubes of twelve edges, plus the outer cube's twelve
    // lateral bonds.
    assert_eq!(nested.bonds().len(), 8 * 12 + 12);
}

// ── the Ising block: a lattice standing where a point stood ──────────

#[test]
fn a_block_reads_its_own_effective_description_off_its_spectrum() {
    let block = RecursiveLattice::new(Shape::SQUARE).unwrap();
    let rg = block_rg(&block, 1.0, 1.0).unwrap();

    // The two-state isometry really is one.
    assert!(rg.isometry_error < 1e-12, "{}", rg.isometry_error);
    // Both effective couplings come out of measurement, and both are
    // physical: a positive gap, a boundary element inside [-1, 1].
    assert!(rg.gap > 0.0);
    assert!(rg.port_element.abs() <= 1.0);
    common::assert_close(rg.renorm_field, rg.gap / 2.0, 1e-15);
    common::assert_close(
        rg.renorm_coupling,
        rg.coupling * rg.port_element * rg.port_element,
        1e-15,
    );

    // Measured values for the square at J = h = 1.
    common::assert_close(rg.port_element.abs(), 0.849_088, 1e-6);
    common::assert_close(rg.renorm_coupling, 0.720_951, 1e-6);
    common::assert_close(rg.renorm_field, 0.198_912, 1e-6);
}

#[test]
fn a_zero_transverse_field_is_refused_rather_than_divided_by() {
    let block = RecursiveLattice::new(Shape::SQUARE).unwrap();
    let err = block_rg(&block, 1.0, 0.0).unwrap_err();
    assert!(format!("{err}").contains("degenerate"));
}

#[test]
fn the_flow_runs_both_ways_and_says_when_it_runs_out_of_resolution() {
    // Disordered side: the ratio collapses monotonically toward zero.
    let weak = rg_flow(Shape::SQUARE, 0.3, 1.0, 8).unwrap();
    assert!(weak.terminated.is_none());
    assert_eq!(weak.steps.len(), 9);
    for pair in weak.steps.windows(2) {
        assert!(pair[1].ratio < pair[0].ratio);
    }
    assert!(weak.steps.last().unwrap().ratio < 1e-4);

    // Ordered side: the block gap collapses geometrically and the flow
    // reports that double precision, not physics, ended it.
    let strong = rg_flow(Shape::SQUARE, 1.5, 1.0, 12).unwrap();
    let reason = strong.terminated.expect("the ordered flow must terminate");
    assert!(reason.contains("double-precision"), "{reason}");
    assert!(strong.steps.len() < 12);
    assert!(strong.steps[1].ratio > strong.steps[0].ratio);
}

#[test]
fn the_flow_has_a_measured_fixed_point_where_a_lattice_is_its_own_point() {
    let fp = rg_fixed_point(Shape::Chain(2), 2.0).unwrap();
    assert!(fp.residual < 1e-12, "{}", fp.residual);
    common::assert_close(fp.ratio, 0.783_243, 1e-5);
    // Relevant direction: the flow runs away from the fixed point.
    assert!(fp.eigenvalue > 1.0);
    common::assert_close(fp.eigenvalue, 1.596_338, 1e-4);
    // The block-spin exponent against the exactly known ν = 1 of the
    // 1D transverse Ising chain: this is the size of the Kadanoff
    // approximation, measured rather than hidden.
    common::assert_close(fp.exponent, 1.482, 1e-3);

    let square = rg_fixed_point(Shape::SQUARE, 2.0).unwrap();
    assert!(square.residual < 1e-11, "{}", square.residual);
    common::assert_close(square.ratio, 0.554_090, 1e-5);
    assert!(square.eigenvalue > 1.0);
}

#[test]
fn the_substitution_holds_below_the_block_gap_and_measurably_fails_above_it() {
    // Two squares bonded laterally, against two points at the square's
    // own renormalized couplings.
    let r = substitution_report(Shape::SQUARE, 0.2, 1.0, 3).unwrap();
    assert_eq!(r.fine_width, 8);
    assert!(r.residual < 1e-10, "{}", r.residual);

    // Inside the band the two spectra track each other to about 1%.
    assert_eq!(r.levels_in_band, 2);
    assert!(r.deviation_in_band < 0.02, "{}", r.deviation_in_band);
    // Above the block's internal gap the effective point invents a
    // level that is not there, and the report shows it.
    assert!(r.deviation > 0.3, "{}", r.deviation);
    assert!(r.coarse_gaps[2] > r.internal_gap);
    assert!(r.fine_gaps[2] < r.internal_gap);
}

#[test]
fn the_substitution_error_is_a_measured_number_not_a_claim() {
    // Across the coupling range the in-band error stays a few percent
    // and never silently vanishes.
    for g in [0.2, 1.0, 3.0] {
        let r = substitution_report(Shape::SQUARE, g, 1.0, 3).unwrap();
        assert!(r.deviation_in_band > 0.0);
        assert!(
            r.deviation_in_band < 0.1,
            "g={g}: in-band deviation {}",
            r.deviation_in_band
        );
    }
}

// ── phonons: where the duality is exact ─────────────────────────────

#[test]
fn a_harmonic_block_is_its_own_point_with_no_frequency_shift_at_all() {
    let b = phonon_block(Shape::SQUARE, 1.0, 0.3).unwrap();
    // The internal springs annihilate the uniform displacement exactly.
    assert!(b.frequency_shift < 1e-14, "{}", b.frequency_shift);
    common::assert_close(b.collective_frequency, 1.0, 1e-14);
    // Four sites share the collective coordinate equally.
    common::assert_close(b.port_participation, 0.5, 1e-12);
    common::assert_close(b.renorm_spring, 0.3 / 4.0, 1e-12);
    common::assert_close(b.uniform_renorm_spring, 0.3 / 4.0, 1e-12);
    // And the internal modes are genuinely above it.
    assert!(b.internal_gap > 0.2);
}

#[test]
fn uniform_bonding_makes_the_substitution_exact_and_port_bonding_does_not() {
    let uniform = phonon_substitution(Shape::SQUARE, 1.0, 0.5, 0.3, Bonding::Uniform).unwrap();
    assert!(uniform.deviation < 1e-12, "{}", uniform.deviation);
    assert_eq!(uniform.fine_width, 16);
    common::assert_close(uniform.renorm_spring, 0.3 / 4.0, 1e-15);

    let port = phonon_substitution(Shape::SQUARE, 1.0, 0.5, 0.3, Bonding::Port).unwrap();
    assert!(port.deviation > 1e-3, "{}", port.deviation);
}

#[test]
fn the_port_bonded_substitution_error_is_second_order_in_the_lateral_spring() {
    let mut previous: Option<f64> = None;
    for lateral in [0.3, 0.03, 0.003] {
        let s = phonon_substitution(Shape::SQUARE, 1.0, 0.5, lateral, Bonding::Port).unwrap();
        if let Some(prev) = previous {
            // A tenfold weaker lateral spring buys roughly a
            // hundredfold smaller error: second order, measured.
            let ratio = prev / s.deviation;
            assert!(ratio > 50.0 && ratio < 200.0, "ratio {ratio}");
        }
        previous = Some(s.deviation);
    }
}

#[test]
fn a_phonon_walks_the_recursive_lattice_on_a_real_backend() {
    let w = phonon_walk(Shape::SQUARE, 1.0, 0.01, 20.0 / 160.0, 160).unwrap();
    assert_eq!(w.width, 16);
    common::assert_close(w.coarse_spring, 0.01 / 4.0, 1e-15);

    // The xy gate conserves excitation number exactly: the simulator
    // never leaves the one-phonon sector.
    assert!(w.leakage < 1e-12, "{}", w.leakage);
    // The substitution error is tiny and independent of the Trotter
    // step; the Trotter error shrinks with it.
    assert!(
        w.substitution_deviation < 1e-3,
        "{}",
        w.substitution_deviation
    );

    let coarse_step = phonon_walk(Shape::SQUARE, 1.0, 0.01, 20.0 / 40.0, 40).unwrap();
    assert!(coarse_step.trotter_deviation > w.trotter_deviation);
    common::assert_close(
        coarse_step.substitution_deviation,
        w.substitution_deviation,
        1e-9,
    );
}

#[test]
fn the_walks_substitution_error_is_second_order_too() {
    // Same total walk time (20.0) as the sibling test, in 40 coarse
    // steps rather than 400 fine ones: the substitution deviation is
    // step-independent — that test pins it to 1e-9 across a 4x change
    // in dt — so the coarse walk measures the same quantity for a
    // tenth of the arithmetic.
    let strong = phonon_walk(Shape::SQUARE, 1.0, 0.1, 0.5, 40).unwrap();
    let weak = phonon_walk(Shape::SQUARE, 1.0, 0.01, 0.5, 40).unwrap();
    let ratio = strong.substitution_deviation / weak.substitution_deviation;
    assert!(ratio > 50.0 && ratio < 200.0, "ratio {ratio}");
}

// ── the computation that participates in itself ─────────────────────

#[test]
fn a_block_that_is_its_own_environment_converges_to_a_measured_fixed_point() {
    let block = RecursiveLattice::new(Shape::SQUARE).unwrap();

    // Weak coupling: the computation talks itself down to nothing.
    let weak = self_participation(&block, 0.2, 1.0, 2, 0.5, 1e-10, 300).unwrap();
    assert!(weak.converged);
    assert!(weak.magnetization.abs() < 1e-8, "{}", weak.magnetization);

    // Strong coupling: a nonzero answer sustains itself, and the field
    // it sits in is exactly the one its own answer produces.
    let strong = self_participation(&block, 1.0, 1.0, 2, 0.5, 1e-10, 300).unwrap();
    assert!(strong.converged);
    assert!(strong.magnetization > 0.9);
    common::assert_close(strong.field, 1.0 * 2.0 * strong.magnetization, 1e-9);
    assert!(strong.residual <= 1e-10);
    // It really iterated rather than starting at the answer.
    assert!(strong.rounds.len() > 3);
    assert!(strong.rounds[0].magnetization != strong.magnetization);
}

#[test]
fn self_participation_works_at_any_recursion_depth() {
    // A chain of chains: nine qubits, the block itself a lattice.
    let deep = RecursiveLattice::nest(Shape::Chain(3), 2).unwrap();
    assert_eq!(deep.width(), 9);
    let p = self_participation(&deep, 1.0, 1.0, 2, 0.5, 1e-8, 100).unwrap();
    assert!(p.converged, "residual {}", p.residual);
    assert!(p.magnetization > 0.8);
    assert_eq!(p.width, 9);
}

#[test]
fn the_self_feeding_iteration_has_a_measured_critical_coupling() {
    let block = RecursiveLattice::new(Shape::SQUARE).unwrap();
    let t = participation_transition(&block, 1.0, 2, 1e-3, (0.05, 3.0), 40).unwrap();
    common::assert_close(t.coupling, 0.267_675, 1e-5);
    assert!(t.bracket < 1e-9);
    // Ten percent either side of it: an order-unity answer above,
    // nothing below.
    assert!(t.magnetization_above > 0.3, "{}", t.magnetization_above);
    assert!(t.magnetization_below < 1e-6, "{}", t.magnetization_below);
}

#[test]
fn an_unbracketed_transition_is_refused_with_the_reason() {
    let block = RecursiveLattice::new(Shape::SQUARE).unwrap();
    let err = participation_transition(&block, 1.0, 2, 1e-3, (0.05, 0.1), 20).unwrap_err();
    assert!(format!("{err}").contains("widen it upward"));
    let err = participation_transition(&block, 1.0, 2, 1e-3, (1.0, 3.0), 20).unwrap_err();
    assert!(format!("{err}").contains("widen it downward"));
}

#[test]
fn registers_wider_than_the_exact_eigensolver_are_refused_structurally() {
    let wide = RecursiveLattice::nest(Shape::SQUARE, 2).unwrap();
    assert_eq!(wide.width(), 16);
    let err = self_participation(&wide, 1.0, 1.0, 2, 0.5, 1e-8, 10).unwrap_err();
    assert!(format!("{err}").contains("qubits"));
}
