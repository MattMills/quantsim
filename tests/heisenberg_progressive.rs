//! Progressive stateless computation: exactness against dense, the
//! certified error bound actually bounding, the retrodictive journal
//! locating where precision was spent, checkpointed refinement, and the
//! shared walk over an observable basis.

use quantsim::heisenberg::*;
use quantsim::prelude::*;

fn exact_cfg() -> Config {
    Config {
        threshold: 0.0,
        max_terms: None,
        checkpoint_every: 0,
        exclusion: true,
        retire_frozen: true,
    }
}

/// The same, with exclusion off — the reference the exclusion mode must
/// agree with.
fn no_exclusion_cfg() -> Config {
    Config {
        exclusion: false,
        retire_frozen: false,
        ..exact_cfg()
    }
}

/// Evolve `|0…0⟩` through the rotations on a dense backend.
fn dense_run(rotations: &[Rotation], n: usize) -> DenseState<C64> {
    let mut d = DenseState::<C64>::new(n).unwrap();
    for r in rotations {
        let (m, s) = r.gate().unwrap();
        d.apply(&m, &s).unwrap();
    }
    d
}

// ── the algebra ──────────────────────────────────────────────────────

#[test]
fn the_pauli_basis_multiplies_and_commutes_correctly() {
    // exhaustive on two qubits, against explicit matrices
    for ax in 0..4u64 {
        for az in 0..4u64 {
            for bx in 0..4u64 {
                for bz in 0..4u64 {
                    let (p, q) = ((ax, az), (bx, bz));
                    // commutation from the symplectic form vs from the
                    // product in both orders
                    let (k1, s1) = pauli_mul(p, q);
                    let (k2, s2) = pauli_mul(q, p);
                    assert_eq!(k1, k2, "product keys must agree");
                    let same_sign = s1 == s2;
                    assert_eq!(
                        commutes(p, q),
                        same_sign,
                        "commutation disagrees for {p:?} {q:?}"
                    );
                }
            }
        }
    }
    // the Hermitian axis phases
    assert_eq!(axis_operator_phase((0, 0)), C64::new(1.0, 0.0));
    assert_eq!(axis_operator_phase((1, 1)), C64::new(0.0, 1.0));
    assert_eq!(axis_operator_phase((3, 3)), C64::new(-1.0, 0.0));
}

#[test]
fn rotation_gates_are_unitary_and_land_on_their_support() {
    for rot in [
        Rotation::rz(0, 0.3),
        Rotation::rx(2, -1.1),
        Rotation::rzz(0, 3, 0.7),
        Rotation::rxx(1, 4, 2.2),
        Rotation {
            theta: 0.9,
            axis: (0b101, 0b011),
        },
    ] {
        let (m, s) = rot.gate().unwrap();
        assert!(m.is_unitary(1e-13), "axis {:?} gate not unitary", rot.axis);
        assert_eq!(s.len(), rot.weight());
        assert_eq!(m.dim(), 1usize << rot.weight());
    }
    assert!(Rotation { theta: 1.0, axis: (0, 0) }.gate().is_err());
}

// ── exactness ────────────────────────────────────────────────────────

#[test]
fn the_propagation_is_exact_against_dense_at_non_clifford_angles() {
    let mut worst = 0.0f64;
    let mut checks = 0usize;
    for n in 2..=9usize {
        for &dt in &[0.11f64, 0.3, 0.7] {
            let rots = tfim_trotter(n, 1.0, 0.7, dt, 3);
            let d = dense_run(&rots, n);
            for q in 0..n {
                for (obs, ops) in [
                    (PauliSum::z(q), vec![(q, Pauli::Z)]),
                    (PauliSum::x(q), vec![(q, Pauli::X)]),
                    (PauliSum::y(q), vec![(q, Pauli::Y)]),
                ] {
                    let p = propagate(&obs, &rots, &exact_cfg()).unwrap();
                    let reference =
                        pauli_expectation(&d as &dyn Backend<C64>, &ops).unwrap().re;
                    worst = worst.max((p.expectation() - reference).abs());
                    checks += 1;
                }
            }
        }
    }
    assert!(checks >= 396, "only {checks} checks");
    assert!(worst < 1e-13, "exact propagation deviated by {worst:.3e}");
}

#[test]
fn a_clifford_angle_never_branches_the_sum() {
    // dt·J = dt·h = π/4 is the Clifford point: every rotation maps one
    // Pauli to one Pauli, so the sum stays a single term forever.
    let n = 12;
    let dt = std::f64::consts::FRAC_PI_4;
    let rots = tfim_trotter(n, 1.0, 1.0, dt, 20);
    let p = propagate(&PauliSum::z(5), &rots, &no_exclusion_cfg()).unwrap();
    assert_eq!(p.sum.len(), 1, "Clifford rotations must not branch");
    assert_eq!(p.discarded_l1, 0.0);
    assert!((p.sum.l2_squared() - 1.0).abs() < 1e-12);
}

#[test]
fn the_l2_weight_is_conserved_by_conjugation() {
    let n = 8;
    let rots = tfim_trotter(n, 1.0, 0.7, 0.4, 4);
    // Exclusion deliberately removes terms, so L2 conservation is a
    // property of the *unexcluded* walk.
    let p = propagate(&PauliSum::z(3), &rots, &no_exclusion_cfg()).unwrap();
    // unitary conjugation preserves the Frobenius norm of the observable
    assert!(
        (p.sum.l2_squared() - 1.0).abs() < 1e-12,
        "l2 drifted to {}",
        p.sum.l2_squared()
    );
    for step in p.journal.steps() {
        assert!((step.l2_squared - 1.0).abs() < 1e-12);
    }
}

#[test]
fn exclusion_is_exact_and_never_changes_the_answer() {
    // Exclusion drops terms the remaining circuit provably cannot bring
    // back to the X-free sector. If the reachability test is sound, the
    // answer is bit-identical to the unexcluded walk — that is the whole
    // correctness criterion, and it is checked against dense too.
    let mut worst_vs_plain = 0.0f64;
    let mut worst_vs_dense = 0.0f64;
    let mut total_excluded = 0usize;
    let mut checks = 0usize;
    for n in 2..=8usize {
        for &dt in &[0.2f64, 0.55] {
            let rots = tfim_trotter(n, 1.0, 0.7, dt, 3);
            let d = dense_run(&rots, n);
            for q in 0..n {
                let with = propagate(&PauliSum::z(q), &rots, &exact_cfg()).unwrap();
                let without = propagate(&PauliSum::z(q), &rots, &no_exclusion_cfg()).unwrap();
                let reference = pauli_expectation(&d as &dyn Backend<C64>, &[(q, Pauli::Z)])
                    .unwrap()
                    .re;
                worst_vs_plain =
                    worst_vs_plain.max((with.expectation() - without.expectation()).abs());
                worst_vs_dense = worst_vs_dense.max((with.expectation() - reference).abs());
                total_excluded += with.excluded_terms;
                // exclusion is exact: it never adds to the error ledger
                assert_eq!(with.discarded_l1, 0.0, "exact mode must discard nothing");
                assert!(with.sum.len() <= without.sum.len());
                checks += 1;
            }
        }
    }
    assert!(checks >= 60, "only {checks} checks");
    assert!(
        worst_vs_plain < 1e-13,
        "exclusion changed the answer by {worst_vs_plain:.3e}"
    );
    assert!(
        worst_vs_dense < 1e-13,
        "excluded walk deviated from dense by {worst_vs_dense:.3e}"
    );
    assert!(
        total_excluded > 0,
        "exclusion never fired — the test proves nothing"
    );
}

#[test]
fn the_x_span_filtration_is_a_correct_gf2_basis() {
    let rots = tfim_trotter(6, 1.0, 0.7, 0.3, 3);
    let span = XSpan::of(&rots);
    // rank is at most the width, and reached because every site gets an X
    assert!(span.rank() <= 6);
    assert_eq!(span.rank(), 6);
    // zero is always reachable, from anywhere
    for g in 0..=rots.len() {
        assert!(span.reachable(0, g));
    }
    // with no gates left, only zero is reachable
    for x in 1u64..64 {
        assert!(!span.reachable(x, 0), "x={x} reachable with no gates left");
    }
    // with every gate available, the full span is reachable
    for x in 0u64..64 {
        assert!(span.reachable(x, rots.len()));
    }
    // and membership is monotone in the number of remaining gates
    for x in 0u64..64 {
        let mut seen_true = false;
        for g in 0..=rots.len() {
            let r = span.reachable(x, g);
            if r {
                seen_true = true;
            }
            assert!(!seen_true || r, "reachability un-monotone for x={x}");
        }
    }
}

// ── the certified bound ──────────────────────────────────────────────

#[test]
fn the_error_bound_actually_bounds_the_error() {
    let n = 10;
    let rots = tfim_trotter(n, 1.0, 0.7, 0.35, 5);
    let d = dense_run(&rots, n);
    for q in [0usize, 4, 9] {
        let reference = pauli_expectation(&d as &dyn Backend<C64>, &[(q, Pauli::Z)])
            .unwrap()
            .re;
        let mut last_error = f64::INFINITY;
        for &th in &[1e-1f64, 1e-2, 1e-3, 1e-5, 1e-8] {
            let cfg = Config {
                threshold: th,
                max_terms: None,
                checkpoint_every: 0,
        exclusion: true,
        retire_frozen: true,
            };
            let p = propagate(&PauliSum::z(q), &rots, &cfg).unwrap();
            let error = (p.expectation() - reference).abs();
            assert!(
                error <= p.error_bound() + 1e-12,
                "q={q} th={th:e}: error {error:.3e} exceeded its bound {:.3e}",
                p.error_bound()
            );
            // and tightening the threshold never makes it worse
            assert!(
                error <= last_error + 1e-9,
                "q={q} th={th:e}: error grew from {last_error:.3e} to {error:.3e}"
            );
            last_error = error;
        }
        // threshold 1e-8 leaves an error of that order; exactness needs
        // threshold 0, which the floor now makes safe.
        assert!(last_error < 1e-7, "did not converge: {last_error:.3e}");
        let p = propagate(&PauliSum::z(q), &rots, &exact_cfg()).unwrap();
        assert!(
            (p.expectation() - reference).abs() < 1e-13,
            "exact mode deviated by {:.3e}",
            (p.expectation() - reference).abs()
        );
    }
}

#[test]
fn the_term_cap_is_respected_and_reported() {
    let n = 16;
    let rots = tfim_trotter(n, 1.0, 0.7, 0.3, 6);
    let cfg = Config {
        threshold: 0.0,
        max_terms: Some(200),
        checkpoint_every: 0,
        exclusion: true,
        retire_frozen: true,
    };
    let p = propagate(&PauliSum::z(8), &rots, &cfg).unwrap();
    assert!(p.hit_cap, "the cap should have bound");
    assert!(p.peak_terms <= 201, "peak {} exceeded the cap", p.peak_terms);
    // dropping terms to meet a cap is still accounted
    assert!(p.discarded_l1 > 0.0);
}

// ── the retrodictive journal ─────────────────────────────────────────

#[test]
fn the_journal_is_monotone_and_retrodiction_matches_a_linear_scan() {
    let n = 14;
    let rots = tfim_trotter(n, 1.0, 0.7, 0.3, 6);
    let cfg = Config {
        threshold: 1e-6,
        max_terms: None,
        checkpoint_every: 16,
        exclusion: true,
        retire_frozen: true,
    };
    let p = propagate(&PauliSum::z(7), &rots, &cfg).unwrap();
    assert_eq!(p.journal.steps().len(), rots.len());

    // monotone — the property that makes the binary search sound
    let mut prev = 0.0;
    for s in p.journal.steps() {
        assert!(s.cumulative_l1 >= prev - 1e-15, "cumulative decreased");
        prev = s.cumulative_l1;
    }

    // the walk is backwards: step k is gate L-1-k
    for (k, s) in p.journal.steps().iter().enumerate() {
        assert_eq!(s.gate_index, rots.len() - 1 - k);
    }

    // binary search agrees with a linear scan at every budget
    for budget_exp in 1..12 {
        let budget = 10f64.powi(-budget_exp);
        let linear = p
            .journal
            .steps()
            .iter()
            .position(|s| s.cumulative_l1 > budget);
        assert_eq!(
            p.journal.retrodict(budget),
            linear,
            "retrodict disagreed at budget {budget:e}"
        );
        // and the gate it blames is the gate at that step
        if let Some(i) = linear {
            assert_eq!(
                p.journal.blame_gate(budget),
                Some(p.journal.steps()[i].gate_index)
            );
        }
    }
    // a budget above the total is never exceeded
    assert_eq!(p.journal.retrodict(p.discarded_l1 * 2.0 + 1.0), None);
}

#[test]
fn refinement_restarts_from_a_checkpoint_and_improves_the_answer() {
    let n = 12;
    let rots = tfim_trotter(n, 1.0, 0.7, 0.35, 6);
    let d = dense_run(&rots, n);
    let reference = pauli_expectation(&d as &dyn Backend<C64>, &[(6, Pauli::Z)])
        .unwrap()
        .re;

    let coarse = Config {
        threshold: 1e-3,
        max_terms: None,
        checkpoint_every: 8,
        exclusion: true,
        retire_frozen: true,
    };
    let p = propagate(&PauliSum::z(6), &rots, &coarse).unwrap();
    assert!(p.journal.checkpoint_count() > 1);
    let coarse_error = (p.expectation() - reference).abs();

    // Refining from step 0 — where the checkpoint is the pristine
    // observable — is a full re-run and recovers everything.
    let fine = Config {
        threshold: 1e-9,
        max_terms: None,
        checkpoint_every: 8,
        exclusion: true,
        retire_frozen: true,
    };
    let (full, walked_full) = p.refine_from(&rots, 0, &fine).unwrap();
    assert_eq!(walked_full, rots.len());
    let full_error = (full.expectation() - reference).abs();
    assert!(
        full_error < coarse_error / 10.0,
        "full refinement barely helped: {coarse_error:.3e} → {full_error:.3e}"
    );

    // Refining from a *late* checkpoint walks less but cannot undo what
    // that checkpoint already threw away — the honest limit of the
    // space-time dial, asserted rather than assumed.
    let budget = p.discarded_l1 / 4.0;
    let step = p.journal.retrodict(budget).expect("budget must be blown");
    let (partial, walked_partial) = p.refine_from(&rots, step, &fine).unwrap();
    assert!(
        walked_partial <= rots.len(),
        "partial refinement walked {walked_partial} of {}",
        rots.len()
    );
    let partial_error = (partial.expectation() - reference).abs();
    assert!(
        partial_error >= full_error - 1e-12,
        "a late checkpoint cannot beat a full re-run: {partial_error:.3e} vs {full_error:.3e}"
    );
    assert!(partial.discarded_l1 > 0.0, "carried error must be kept");
}

// ── the shared walk ──────────────────────────────────────────────────

#[test]
fn the_shared_walk_matches_the_sum_of_separate_walks_and_costs_less() {
    let n = 14;
    let rots = tfim_trotter(n, 1.0, 0.7, 0.3, 5);
    let (obs, w) = tfim_energy_basis(n, 1.0, 0.7);
    assert_eq!(obs.len(), 2 * n - 1);

    let cfg = Config {
        threshold: 1e-7,
        max_terms: None,
        checkpoint_every: 0,
        exclusion: true,
        retire_frozen: true,
    };
    let r = propagate_basis(&obs, &w, &rots, &cfg, true).unwrap();

    // the shared total is the sum of the individual contributions
    let summed: f64 = r.breakdown.as_ref().unwrap().iter().sum();
    assert!(
        (r.total - summed).abs() < 1e-6,
        "shared {} vs summed {}",
        r.total,
        summed
    );

    // and it is much cheaper than walking them one at a time
    let factor = r.sharing_factor().unwrap();
    assert!(
        factor > 3.0,
        "sharing bought only {factor:.2}× ({} vs {})",
        r.peak_terms,
        r.separate_peak_total.unwrap()
    );

    // against dense, the energy is right
    let d = dense_run(&rots, n);
    let mut reference = 0.0;
    for k in 0..n - 1 {
        reference -= pauli_expectation(&d as &dyn Backend<C64>, &[(k, Pauli::Z), (k + 1, Pauli::Z)])
            .unwrap()
            .re;
    }
    for k in 0..n {
        reference -= 0.7 * pauli_expectation(&d as &dyn Backend<C64>, &[(k, Pauli::X)])
            .unwrap()
            .re;
    }
    assert!(
        (r.total - reference).abs() < 1e-4,
        "energy {} vs dense {}",
        r.total,
        reference
    );
}

#[test]
fn propagate_basis_validates_its_inputs() {
    let rots = tfim_trotter(4, 1.0, 0.7, 0.3, 2);
    let cfg = exact_cfg();
    assert!(propagate_basis(&[], &[], &rots, &cfg, false).is_err());
    assert!(propagate_basis(&[PauliSum::z(0)], &[1.0, 2.0], &rots, &cfg, false).is_err());
}

// ── the cone, measured while walking ─────────────────────────────────

#[test]
fn commuting_gates_are_skipped_and_that_is_the_light_cone() {
    // A local observable at one end of a long chain cannot be reached by
    // gates at the other end within a shallow circuit, so most rotations
    // must skip.
    let n = 30;
    let rots = tfim_trotter(n, 1.0, 0.7, 0.3, 2);
    let p = propagate(&PauliSum::z(0), &rots, &exact_cfg()).unwrap();
    assert!(
        p.commuting_skips > p.branchings,
        "expected the cone to dominate: {} skips vs {} branchings",
        p.commuting_skips,
        p.branchings
    );
    assert_eq!(p.commuting_skips + p.branchings, rots.len());
    // and the observable really has only spread a bounded distance
    assert!(
        p.sum.max_weight() <= 2 * 2 + 1,
        "spread to weight {} in 2 layers",
        p.sum.max_weight()
    );
}
