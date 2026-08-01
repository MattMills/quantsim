//! The latent past and future as fields: that the invariant really is
//! invariant, that the influence field's perturbation bound really
//! bounds a perturbation actually inserted, and that the field says
//! things the light cone cannot.

use quantsim::backend::pauli_expectation;
use quantsim::heisenberg::*;
use quantsim::horizon::{self, Config, Forward, Horizon};
use quantsim::prelude::*;

fn exact(samples: usize) -> Config {
    Config {
        samples,
        threshold: 0.0,
        forward: Forward::Dense,
    }
}

/// `⟨0…0| U† (X^x Z^z) U |0…0⟩`, run densely — the ground truth.
fn dense_value(rotations: &[Rotation], n: usize, key: PauliKey) -> f64 {
    let mut d = DenseState::<C64>::new(n).unwrap();
    for r in rotations {
        let (m, s) = r.gate().unwrap();
        d.apply(&m, &s).unwrap();
    }
    let ops: Vec<(usize, Pauli)> = (0..n)
        .filter_map(|q| match (key.0 >> q & 1, key.1 >> q & 1) {
            (1, 0) => Some((q, Pauli::X)),
            (0, 1) => Some((q, Pauli::Z)),
            (1, 1) => Some((q, Pauli::Y)),
            _ => None,
        })
        .collect();
    let herm = pauli_expectation(&d as &dyn Backend<C64>, &ops).unwrap();
    (herm / axis_operator_phase(key)).re
}

/// A rotation about a single-qubit axis, indexed as the field stores it.
fn axis_rotation(qubit: usize, axis: usize, delta: f64) -> Rotation {
    let bit = 1u64 << qubit;
    Rotation {
        theta: delta,
        axis: match axis {
            0 => (bit, 0),
            1 => (bit, bit),
            _ => (0, bit),
        },
    }
}

fn chain(n: usize, steps: usize) -> Vec<Rotation> {
    tfim_trotter(n, 1.0, 0.7, 0.35, steps)
}

// ── the invariant ────────────────────────────────────────────────────

#[test]
fn the_contraction_is_the_same_number_at_every_cut() {
    for &(n, steps) in &[(6usize, 3usize), (8, 4), (9, 2)] {
        let rots = chain(n, steps);
        let obs = (0u64, 1u64 << (n / 2));
        let h = horizon::horizon(obs, &rots, n, &exact(12)).unwrap();
        let truth = dense_value(&rots, n, obs);

        assert!(
            h.value_spread() < 1e-11,
            "n={n}: the invariant moved by {:.3e} across cuts",
            h.value_spread()
        );
        for (c, &v) in h.value.iter().enumerate() {
            assert!(
                (v - truth).abs() < 1e-11,
                "n={n} cut {}: {v} vs dense {truth}",
                h.cuts[c]
            );
        }
    }
}

#[test]
fn the_two_ends_are_the_pure_schrodinger_and_pure_heisenberg_answers() {
    let n = 8;
    let rots = chain(n, 4);
    let obs = (0u64, 1u64 << 3);
    let h = horizon::horizon(obs, &rots, n, &exact(8)).unwrap();

    // cut 0: nothing has run forward, so the past field is |0…0⟩ — every
    // Bloch length exactly 1 — and the whole answer sits in the operator.
    assert_eq!(h.cuts[0], 0);
    for q in 0..n {
        assert!(
            (h.commitment[0][q] - 1.0).abs() < 1e-12,
            "q{q} at cut 0 has Bloch length {}",
            h.commitment[0][q]
        );
    }
    // cut `gates`: the operator has not moved, so the future field is the
    // observable itself — one qubit, one axis.
    let last = h.cuts.len() - 1;
    assert_eq!(h.cuts[last], rots.len());
    assert_eq!(h.terms[last], 1);
    assert_eq!(h.max_weight[last], 1);
    for q in 0..n {
        let f = h.influence[last][q];
        if q == 3 {
            // Z_3 anticommutes with X_3 and Y_3, and commutes with Z_3
            assert!((f[0] - 1.0).abs() < 1e-12, "{f:?}");
            assert!((f[1] - 1.0).abs() < 1e-12, "{f:?}");
            assert!(f[2].abs() < 1e-12, "{f:?}");
        } else {
            assert!(f.iter().all(|v| v.abs() < 1e-12), "q{q}: {f:?}");
        }
    }
}

// ── the field's operational meaning ──────────────────────────────────

#[test]
fn the_perturbation_bound_bounds_a_perturbation_actually_inserted() {
    // The whole claim of the influence field: it is not a picture, it is
    // a bound. Insert a real rotation at a real cell, recompute the real
    // answer densely, and check the bound held.
    let n = 7;
    let rots = chain(n, 3);
    let obs = (0u64, 1u64 << 3);
    let h = horizon::horizon(obs, &rots, n, &exact(6)).unwrap();
    let truth = dense_value(&rots, n, obs);

    let mut checks = 0usize;
    let mut worst_slack = f64::INFINITY;
    let mut hit_zero_field = 0usize;
    for delta in [0.05f64, 0.3, 1.0] {
        for (c, &cut) in h.cuts.iter().enumerate() {
            for q in 0..n {
                for axis in 0..3 {
                    let mut perturbed = rots.clone();
                    perturbed.insert(cut, axis_rotation(q, axis, delta));
                    let moved = (dense_value(&perturbed, n, obs) - truth).abs();
                    let bound = h.perturbation_bound(c, q, axis, delta);
                    assert!(
                        moved <= bound + 1e-9,
                        "cut {cut} q{q} axis {axis} δ={delta}: moved {moved:.6e} > bound {bound:.6e}"
                    );
                    if bound == 0.0 {
                        // a zero field is the strongest possible claim:
                        // the answer cannot move at all
                        assert!(moved < 1e-12, "zero field but moved {moved:.3e}");
                        hit_zero_field += 1;
                    } else {
                        worst_slack = worst_slack.min(bound / moved.max(1e-300));
                    }
                    checks += 1;
                }
            }
        }
    }
    assert!(checks >= 400, "only {checks} checks");
    assert!(
        hit_zero_field > 0,
        "the zero-field case was never exercised"
    );
    // The bound is real but loose, as an L1 bound must be; recorded so
    // it is read as a certificate and not as an estimate.
    assert!(worst_slack >= 1.0);
}

#[test]
fn a_zero_field_cell_is_exactly_the_outside_of_the_light_cone() {
    // The cone is the field's support, so the two must agree cell for
    // cell: a gate outside the backward cone is one the walk skips.
    let n = 8;
    let rots = chain(n, 2);
    let obs = (0u64, 1u64 << 4);
    let h = horizon::horizon(obs, &rots, n, &exact(8)).unwrap();

    let mut outside = 0usize;
    for (c, &cut) in h.cuts.iter().enumerate() {
        for q in 0..n {
            if h.influence_max(c, q) == 0.0 {
                outside += 1;
                // a full π rotation is the largest single-qubit kick
                // there is; if the cell is outside the cone even that
                // changes nothing
                let mut perturbed = rots.clone();
                perturbed.insert(cut, axis_rotation(q, 2, std::f64::consts::PI));
                let truth = dense_value(&rots, n, obs);
                let moved = (dense_value(&perturbed, n, obs) - truth).abs();
                assert!(
                    moved < 1e-12,
                    "cut {cut} q{q} is outside the cone but a π kick moved it {moved:.3e}"
                );
            }
        }
    }
    assert!(outside > 0, "this circuit has no outside to check");
}

// ── what the field says that the cone cannot ─────────────────────────

#[test]
fn influence_falls_somewhere_which_a_cone_can_never_do() {
    // A cone only grows. The field can shrink: anticommuting terms
    // cancel against one another as the walk continues, so a qubit's
    // grip on the answer weakens without leaving the cone.
    let n = 8;
    let rots = chain(n, 4);
    let obs = (0u64, 1u64 << 4);
    let h = horizon::horizon(obs, &rots, n, &exact(12)).unwrap();
    assert!(
        h.non_monotone_cells() > 0,
        "no cell's influence fell; the field would then be a cone in disguise"
    );
    // and the cone is nowhere near full
    let fill = h.cone_fill();
    assert!(
        fill > 0.0 && fill < 0.9,
        "cone fill {fill} — a cone drawn as a set claims 1"
    );
}

#[test]
fn the_front_moves_outward_at_a_measurable_speed() {
    // Nearest-neighbour couplings: the cone's edge should advance about
    // one qubit per layer, and the front is where that gets measured.
    let n = 9;
    let steps = 4;
    let rots = chain(n, steps);
    let obs = (0u64, 1u64 << 4);
    let h = horizon::horizon(obs, &rots, n, &exact(12)).unwrap();
    let front = h.front();

    // the observable's own qubit is in the cone all the way to the end
    assert_eq!(front[4], Some(h.gates));
    // and the front recedes monotonically with distance from it
    for d in 1..4usize {
        let (near, far) = (front[4 - d + 1], front[4 - d]);
        if let (Some(a), Some(b)) = (near, far) {
            assert!(
                b <= a,
                "qubit {} enters at {b} but its neighbour {} at {a}",
                4 - d,
                4 - d + 1
            );
        }
    }
    // every qubit is reached eventually on a chain this deep
    assert!(
        front.iter().filter(|f| f.is_some()).count() >= n - 1,
        "front {front:?}"
    );
}

#[test]
fn what_the_future_can_see_stops_being_about_single_qubits() {
    // The readout neither field gives alone. It starts at exactly 1 —
    // the input is a product state, so everything the observable can see
    // is held in single-qubit marginals — and then falls, because the
    // information migrates into correlations while the cone keeps
    // covering the same qubits.
    let n = 8;
    let rots = chain(n, 4);
    let obs = (0u64, 1u64 << 4);
    let h = horizon::horizon(obs, &rots, n, &exact(12)).unwrap();

    let settled = h.settled();
    assert!(
        (settled[0] - 1.0).abs() < 1e-12,
        "a product input settles everything it shows: {}",
        settled[0]
    );
    assert!(
        *settled.last().unwrap() < 0.95,
        "locality never drained: {settled:?}"
    );
    for w in settled.windows(2) {
        assert!(w[1] <= w[0] + 1e-9, "settled rose: {settled:?}");
    }

    // and the stake — visible AND committed — only ever falls
    let stake = h.stake();
    for w in stake.windows(2) {
        assert!(w[1] <= w[0] + 1e-9, "stake rose: {stake:?}");
    }
    // while the visible mass rises going backward, i.e. falls in `t`
    let visible = h.visible();
    assert!(
        visible[0] > *visible.last().unwrap(),
        "the cone must be widest at the start of the circuit: {visible:?}"
    );
}

// ── the retrodictive field ───────────────────────────────────────────

#[test]
fn truncation_blame_lands_on_the_qubits_that_lost_the_weight() {
    let n = 8;
    let rots = chain(n, 4);
    let obs = (0u64, 1u64 << 4);
    let h = horizon::horizon(
        obs,
        &rots,
        n,
        &Config {
            samples: 12,
            threshold: 1e-3,
            forward: Forward::Dense,
        },
    )
    .unwrap();

    let total: f64 = h.blame[0].iter().sum();
    assert!(total > 0.0, "a 1e-3 threshold should discard something");
    // Blame accumulates as the walk runs backward, so the earliest cut
    // carries the most and the last carries none.
    let last = h.cuts.len() - 1;
    assert_eq!(h.blame[last].iter().sum::<f64>(), 0.0);
    for c in 0..last {
        let a: f64 = h.blame[c].iter().sum();
        let b: f64 = h.blame[c + 1].iter().sum();
        assert!(a >= b - 1e-12, "blame fell from cut {c} to {}", c + 1);
    }
    // And it is spent near the observable, not uniformly.
    let near: f64 = (3..=5).map(|q| h.blame[0][q]).sum();
    assert!(
        near > total / n as f64,
        "blame {near} around the observable vs uniform share {}",
        total / n as f64
    );
}

#[test]
fn truncation_shows_up_as_the_invariant_ceasing_to_be_invariant() {
    // The exact walk holds the same number at every cut. A truncated one
    // does not, and the spread is the honest size of the error.
    let n = 7;
    let rots = chain(n, 4);
    let obs = (0u64, 1u64 << 3);
    let truth = dense_value(&rots, n, obs);

    let mut previous = f64::INFINITY;
    for &th in &[1e-2f64, 1e-3, 1e-4, 0.0] {
        let h = horizon::horizon(
            obs,
            &rots,
            n,
            &Config {
                samples: 10,
                threshold: th,
                forward: Forward::Dense,
            },
        )
        .unwrap();
        let spread = h.value_spread();
        assert!(
            spread <= previous + 1e-12,
            "spread grew as the threshold tightened: {spread} after {previous}"
        );
        previous = spread;
        // the cut-0 column is the full backward walk, so it carries the
        // truncation's actual error
        assert!(
            (h.value[0] - truth).abs() <= spread + 1e-9,
            "the spread must cover the error at threshold {th}"
        );
    }
    assert!(previous < 1e-11, "the exact walk should close: {previous}");
}

// ── shape ────────────────────────────────────────────────────────────

#[test]
fn the_sheet_is_shaped_the_way_the_renderers_assume() {
    let n = 6;
    let rots = chain(n, 3);
    let h = horizon::horizon((0, 1u64 << 2), &rots, n, &exact(9)).unwrap();
    assert_eq!(h.qubits, n);
    assert_eq!(h.gates, rots.len());
    assert_eq!(*h.cuts.first().unwrap(), 0);
    assert_eq!(*h.cuts.last().unwrap(), rots.len());
    assert!(h.cuts.windows(2).all(|w| w[0] < w[1]), "{:?}", h.cuts);
    let columns = h.cuts.len();
    assert_eq!(h.value.len(), columns);
    assert_eq!(h.influence.len(), columns);
    assert_eq!(h.commitment.len(), columns);
    assert_eq!(h.blame.len(), columns);
    assert_eq!(h.terms.len(), columns);
    assert_eq!(h.l1.len(), columns);
    assert_eq!(h.max_weight.len(), columns);
    for c in 0..h.cuts.len() {
        assert_eq!(h.influence[c].len(), n);
        assert_eq!(h.commitment[c].len(), n);
        assert_eq!(h.blame[c].len(), n);
        for q in 0..n {
            for a in 0..3 {
                let v = h.influence[c][q][a];
                assert!((0.0..=1.0).contains(&v), "influence {v} out of range");
            }
            let r: f64 = h.commitment[c][q];
            assert!((0.0..=1.0 + 1e-12).contains(&r), "Bloch length {r}");
        }
    }
}

#[test]
fn a_forward_representation_that_can_hold_the_state_gives_the_same_sheet() {
    let n = 8;
    let rots = chain(n, 3);
    let obs = (0u64, 1u64 << 4);
    let dense = horizon::horizon(obs, &rots, n, &exact(8)).unwrap();
    for forward in [Forward::Sparse, Forward::Mps { max_bond: 64 }] {
        let other = horizon::horizon(
            obs,
            &rots,
            n,
            &Config {
                forward,
                ..exact(8)
            },
        )
        .unwrap();
        for c in 0..dense.cuts.len() {
            assert!(
                (dense.value[c] - other.value[c]).abs() < 1e-11,
                "{forward:?} cut {c}"
            );
            for q in 0..n {
                assert!(
                    (dense.commitment[c][q] - other.commitment[c][q]).abs() < 1e-10,
                    "{forward:?} cut {c} q{q}"
                );
            }
        }
    }
}

/// Renderers index this way; pin it so a picture cannot silently
/// transpose.
#[test]
fn a_wide_shallow_circuit_has_a_narrow_cone_and_a_deep_one_a_wide_cone() {
    let n = 10;
    let shallow: Horizon = horizon::horizon((0, 1u64 << 5), &chain(n, 1), n, &exact(6)).unwrap();
    let deep: Horizon = horizon::horizon((0, 1u64 << 5), &chain(n, 4), n, &exact(6)).unwrap();
    let reach = |h: &Horizon| h.front().iter().filter(|f| f.is_some()).count();
    assert!(
        reach(&shallow) < reach(&deep),
        "shallow reached {} qubits, deep {}",
        reach(&shallow),
        reach(&deep)
    );
}
