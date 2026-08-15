//! The register that holds only engineered entanglement, and the exact
//! price the magic charges it.

use quantsim::dcs::{Dcs, Doping};
use quantsim::engineered;
use quantsim::prelude::*;
use quantsim::sweep;

/// Branches the tests will spend. Each is a full sweep of the volume,
/// and the structure being checked is visible well below the module's
/// own ceiling.
const TEST_MAX_T: usize = 7;

/// `T = a·I + b·Z` with both arms Clifford, so every branch must be a
/// single stabilizer term and the weighted sum must be the surface
/// itself. If either fails the decomposition has leaked and every
/// number after it is meaningless.
#[test]
fn every_branch_is_clifford_and_the_sum_is_the_surface() {
    for n in [10usize, 12, 16] {
        let d = Dcs::scaled(n);
        let c = d.circuit();
        for bond in [0usize, 1, 3] {
            if engineered::magic_behind(&c, bond) > TEST_MAX_T {
                continue;
            }
            let e = engineered::engineered_surface(&c, bond, 0).unwrap();
            assert_eq!(
                e.stabilizer_branches, e.branches,
                "n={n} bond {bond}: {} of {} branches were not one stabilizer term",
                e.stabilizer_branches, e.branches
            );
            assert!(
                e.deviation < 1e-12,
                "n={n} bond {bond}: the branch sum deviates by {:e}",
                e.deviation
            );
            assert_eq!(e.branches, 1 << e.t_gates);
        }
    }
}

/// The two arms of the split are the two Clifford gates that add to a
/// `T`: `a + b = 1` and `a − b = e^{iπ/4}`.
#[test]
fn the_arms_add_to_a_t_gate() {
    let (a, b) = (engineered::branch_identity(), engineered::branch_z());
    let sum = a + b;
    let diff = a - b;
    assert!((sum - C64::new(1.0, 0.0)).norm() < 1e-15);
    let w = C64::new(
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    );
    assert!((diff - w).norm() < 1e-15);
}

/// What the register pays is the span, and the span is capped twice:
/// by the branches above it and by the surface's own dimension.
#[test]
fn the_span_is_bounded_by_the_branches_and_by_the_surface() {
    let d = Dcs::scaled(16);
    let c = d.circuit();
    for s in sweep::surfaces(&c, 0).unwrap() {
        if engineered::magic_behind(&c, s.after_qubit) > TEST_MAX_T {
            break;
        }
        let e = engineered::engineered_surface(&c, s.after_qubit, 0).unwrap();
        assert!(e.span_rank <= e.branches, "span past the branch count");
        assert!(
            (e.span_rank as u128) <= e.dense(),
            "span past the surface's dimension"
        );
    }
}

/// The finding that survives the saturation: the branches share their
/// affine geometry. Thousands of Clifford branches land on a handful of
/// cosets, so the engineered part of the entanglement is stored once
/// and the magic lives in the phase.
#[test]
fn the_branches_share_their_affine_geometry() {
    for n in [12usize, 16] {
        let d = Dcs::scaled(n);
        let c = d.circuit();
        for s in sweep::surfaces(&c, 0).unwrap() {
            if engineered::magic_behind(&c, s.after_qubit) > TEST_MAX_T {
                break;
            }
            let e = engineered::engineered_surface(&c, s.after_qubit, 0).unwrap();
            assert!(
                e.distinct_supports <= 4,
                "n={n} bond {}: {} cosets across {} branches",
                e.after_qubit,
                e.distinct_supports,
                e.branches
            );
        }
    }
}

/// And the boundary, which is a counting question the experiment can be
/// asked directly: the register is cheaper than the dense surface only
/// while the magic behind a bond is smaller than the legs on it. On the
/// experiment the two curves cross within the first handful of bonds.
#[test]
fn the_experiment_crosses_the_boundary_almost_immediately() {
    let exp = Dcs::experiment();
    let c = exp.circuit();
    let plan = sweep::plan(&c).unwrap();
    let covered: Vec<usize> = (0..exp.qubits - 1)
        .filter(|&b| engineered::magic_behind(&c, b) < plan.legs_per_bond[b])
        .collect();
    assert!(
        covered.len() < 10,
        "the engineered register covered {} of {} bonds",
        covered.len(),
        exp.qubits - 1
    );
    assert!(
        covered.iter().all(|&b| b < 10),
        "coverage reached past the first ten bonds"
    );
    // Past the crossing the magic only grows, so nothing comes back.
    let deep = engineered::magic_behind(&c, exp.qubits / 2);
    assert!(deep > plan.legs_per_bond[exp.qubits / 2] * 5);
}

/// Concentrating the magic does not help, and the reason is that the
/// count is cumulative along the sweep.
///
/// `magic_behind` accumulates from qubit 0 upward, so the bonds the
/// engineered register can hold are always a *prefix*. Banding the same
/// `T` gates onto few world-lines front-loads that profile and lands
/// the magic behind exactly the bonds that were the only candidates —
/// so it is strictly worse than spreading it, not better. Only magic on
/// the world-lines the sweep reaches last would extend the prefix, and
/// the sweep cannot be reordered to arrange it: it requires
/// nearest-neighbour `CZ`, so the circuit's connectivity fixes the
/// order.
#[test]
fn banding_the_magic_onto_few_world_lines_is_strictly_worse() {
    fn covered(d: Dcs) -> usize {
        let c = d.circuit();
        let plan = sweep::plan(&c).unwrap();
        (0..d.qubits - 1)
            .filter(|&b| {
                let legs = plan.legs_per_bond[b];
                let t = engineered::magic_behind(&c, b);
                let dense = 1u128 << legs.min(100);
                let span = if t >= legs { dense } else { 1u128 << t };
                span.saturating_mul((legs as u128 + 1).pow(2)) < dense
            })
            .count()
    }
    let exp = Dcs::experiment();
    let uniform = covered(exp);
    assert!(uniform > 0 && uniform < 10, "uniform covered {uniform}");
    for (w, g) in [(1usize, 9usize), (5, 30), (10, 60), (35, 35)] {
        let banded = covered(exp.with_doping(Doping::Banded {
            width: w,
            gap: g,
            layers: exp.depth,
            late: false,
        }));
        assert!(
            banded <= uniform,
            "banding width {w} gap {g} covered {banded} bonds against uniform's {uniform}"
        );
    }

    // Cumulative, so the profile never decreases along the sweep.
    let c = exp.circuit();
    let mut last = 0usize;
    for b in 0..exp.qubits - 1 {
        let t = engineered::magic_behind(&c, b);
        assert!(t >= last, "the magic behind bond {b} went down");
        last = t;
    }
    // The last bond sits below the top world-line, so its volume is
    // everything except that line's own magic.
    assert!(last < exp.t_gates);
    assert_eq!(
        engineered::magic_behind(&c, exp.qubits - 1),
        exp.t_gates,
        "every T gate is behind the top of the register"
    );
}

/// The branch budget is stated and refused at, rather than discovered
/// by waiting: each branch is a full sweep of the volume.
#[test]
fn too_much_magic_behind_a_bond_is_refused_by_name() {
    let d = Dcs::scaled(24);
    let c = d.circuit();
    let deep = sweep::surfaces(&c, 0).unwrap().len() - 1;
    let e = engineered::engineered_surface(&c, deep, 0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("branch sweeps"), "got {e}");
}
