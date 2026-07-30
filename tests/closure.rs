//! A journalled state representation whose errors are found by
//! geometric closure, retrodictively — measured.
//!
//! The loops a geometry happens to have are checks nobody declared;
//! breaking one localizes in space; breaking it *at some point* localizes
//! in time; and the history is read backwards to find when.

mod common;

use quantsim::bundle::PolarityBundle;
use quantsim::closure::*;
use quantsim::prelude::*;

fn grid(w: usize, h: usize) -> PolarityBundle {
    let mut b = PolarityBundle::new(w * h).unwrap();
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as u32;
            if x + 1 < w {
                b.link(i, i + 1).unwrap();
            }
            if y + 1 < h {
                b.link(i, i + w as u32).unwrap();
            }
        }
    }
    b
}

fn ring(n: usize) -> PolarityBundle {
    let mut b = PolarityBundle::new(n).unwrap();
    for i in 0..n as u32 {
        b.link(i, (i + 1) % n as u32).unwrap();
    }
    b
}

// ── what the geometry supplies without being asked ───────────────────

#[test]
fn every_independent_loop_is_a_check_nobody_declared() {
    for (bundle, expected_rank) in [
        (ring(3), 1),
        (ring(6), 1),
        (grid(4, 4), 24 - 16 + 1),
        (grid(6, 6), 60 - 36 + 1),
    ] {
        let links = bundle.profile().links;
        let sites = bundle.sites();
        let basis = closure_basis(&bundle);
        // rank = |E| − |V| + components, exactly.
        assert_eq!(basis.rank, expected_rank);
        assert_eq!(basis.rank, links - sites + 1);
        assert_eq!(basis.loops.len(), basis.rank);
        // Every loop really closes: consecutive sites are linked, and
        // the last returns to the first.
        for cycle in &basis.loops {
            assert!(cycle.len() >= 3);
            for (a, b) in cycle.links() {
                assert!(bundle.linked(a, b), "loop step {a}→{b} is not a link");
            }
        }
        // The cost of evaluating closure is the total loop length, and
        // it is reported rather than assumed small.
        assert_eq!(
            basis.total_length,
            basis.loops.iter().map(|l| l.len()).sum::<usize>()
        );
    }
}

#[test]
fn a_geometry_with_no_loops_has_no_reach_and_says_so() {
    let mut tree = PolarityBundle::new(5).unwrap();
    for (a, b) in [(0u32, 1u32), (1, 2), (1, 3), (3, 4)] {
        tree.link(a, b).unwrap();
    }
    let basis = closure_basis(&tree);
    assert_eq!(basis.rank, 0);
    assert_eq!(basis.bridges.len(), 4);
    assert_eq!(basis.total_length, 0);

    let mut history = ClosureHistory::new(tree).unwrap();
    history.perturb_spin(3).unwrap();
    let broken = history.break_report().unwrap();
    assert!(broken.intact(), "a tree cannot see anything");
    let localization = history.localize(&broken);
    assert!(localization.undetectable);
    assert!(!localization.unique);
}

// ── closure holds across intended history and breaks otherwise ───────

#[test]
fn intended_changes_keep_closure_and_unrecorded_ones_break_it() {
    let mut history = ClosureHistory::new(grid(4, 4)).unwrap();
    assert!(history.break_report().unwrap().intact());

    // A long run of legitimate changes: closure never moves.
    for step in 0..30u32 {
        history.set_spin(step % 16, step % 3 == 0).unwrap();
        assert!(
            history.break_report().unwrap().intact(),
            "intended step {step} broke closure"
        );
    }

    // One change that did not go through the history.
    history.perturb_spin(5).unwrap();
    let broken = history.break_report().unwrap();
    assert!(!broken.intact());
    assert!(broken.chirality.is_empty(), "a spin change is a spin break");
    assert!(!broken.spin.is_empty());
}

#[test]
fn a_break_is_exactly_the_loops_through_the_site() {
    let bundle = grid(5, 5);
    let basis = closure_basis(&bundle);
    for site in 0..bundle.sites() as u32 {
        let mut history = ClosureHistory::new(bundle.clone()).unwrap();
        history.perturb_spin(site).unwrap();
        let broken = history.break_report().unwrap().spin;
        let expected: Vec<usize> = (0..basis.rank)
            .filter(|&i| basis.loops[i].passes_through(site))
            .collect();
        // Geometric, not statistical: the loops that break are the loops
        // that pass through, and no others.
        assert_eq!(broken, expected, "site {site}");
    }
}

// ── localization ─────────────────────────────────────────────────────

#[test]
fn the_geometry_names_the_site_when_it_can_and_declines_when_it_cannot() {
    let bundle = grid(6, 6);
    let (mut unique, mut ambiguous, mut invisible) = (0, 0, 0);
    for site in 0..bundle.sites() as u32 {
        let mut history = ClosureHistory::new(bundle.clone()).unwrap();
        history.perturb_spin(site).unwrap();
        let localization = history.localize(&history.break_report().unwrap());
        if localization.undetectable {
            invisible += 1;
        } else if localization.unique {
            // When it names one site, it names the right one.
            assert_eq!(localization.sites, vec![site]);
            unique += 1;
        } else {
            // When it cannot, the true site is still among the
            // candidates — the ambiguity is honest, not wrong.
            assert!(localization.sites.contains(&site), "site {site} was lost");
            ambiguous += 1;
        }
    }
    assert_eq!(unique + ambiguous + invisible, 36);
    assert_eq!(invisible, 0, "every grid site lies on some loop");
    assert!(unique >= 20, "only {unique} sites localized exactly");
}

#[test]
fn one_loop_cannot_localize_and_reports_the_whole_loop() {
    // A ring has a single independent loop, so it can tell that
    // something changed but not where. Every site on the loop is a
    // candidate, and the report says so instead of picking one.
    let mut history = ClosureHistory::new(ring(6)).unwrap();
    history.perturb_spin(2).unwrap();
    let localization = history.localize(&history.break_report().unwrap());
    assert!(!localization.undetectable);
    assert!(!localization.unique);
    assert_eq!(localization.sites.len(), 6);
    assert!(localization.sites.contains(&2));
}

// ── retrodiction ─────────────────────────────────────────────────────

#[test]
fn the_history_is_read_backwards_for_the_moment_closure_broke() {
    let mut history = ClosureHistory::new(grid(6, 6)).unwrap();
    history.stamp().unwrap();
    for step in 0..40u32 {
        history.set_spin(step % 36, step % 3 == 0).unwrap();
        if step % 4 == 3 {
            history.stamp().unwrap();
        }
    }
    let last_clean_step = history.step();

    // The perturbation is not recorded as a perturbation. It is simply
    // something that happened.
    history.perturb_spin(14).unwrap();
    let injected_at = history.step();

    for step in 0..40u32 {
        history.set_spin((step * 7) % 36, step % 2 == 0).unwrap();
        if step % 4 == 3 {
            history.stamp().unwrap();
        }
    }
    history.stamp().unwrap();

    let retro = history.retrodict().unwrap();
    // Time: the interval between the last intact stamp and the first
    // broken one brackets the moment.
    let last_intact = retro.last_intact_step.expect("closure held at first");
    let first_broken = retro.step.expect("closure is broken now");
    assert!(last_intact <= last_clean_step);
    assert!(last_intact < injected_at);
    assert!(first_broken >= injected_at);

    // Space: the same act names the site.
    assert!(retro.localization.unique);
    assert_eq!(retro.localization.sites, vec![14]);

    // And it took logarithmically many comparisons of a few bits each.
    let stamps = retro.linear_comparisons as f64;
    assert!(
        (retro.comparisons as f64) <= stamps.log2().ceil() + 2.0,
        "{} comparisons for {stamps} stamps",
        retro.comparisons
    );
    assert!(retro.comparisons < retro.linear_comparisons);
}

#[test]
fn an_unbroken_history_retrodicts_to_nothing() {
    let mut history = ClosureHistory::new(grid(4, 4)).unwrap();
    history.stamp().unwrap();
    for step in 0..12u32 {
        history.set_spin(step % 16, true).unwrap();
        history.stamp().unwrap();
    }
    let retro = history.retrodict().unwrap();
    assert!(retro.step.is_none());
    assert!(retro.localization.undetectable);
    assert!(retro.last_intact_step.is_some());
}

#[test]
fn the_cost_of_being_able_to_retrodict_is_two_bits_per_loop() {
    let mut history = ClosureHistory::new(grid(6, 6)).unwrap();
    let rank = history.basis().rank;
    assert_eq!(history.stamp_bits(), 0);
    history.stamp().unwrap();
    assert_eq!(history.stamp_bits(), 2 * rank);
    history.stamp().unwrap();
    assert_eq!(history.stamp_bits(), 4 * rank);
    // A stamp is not a state: 36 sites cost 50 bits a stamp, and the
    // number of stamps is the caller's choice.
    assert_eq!(2 * rank, 50);
}

// ── correction ───────────────────────────────────────────────────────

#[test]
fn correction_restores_closure_and_the_denoted_state_exactly() {
    let mut clean = grid(4, 4);
    clean.set_spin(6, true).unwrap();
    clean.set_spin(9, true).unwrap();

    // Pick a site the geometry can name exactly.
    let target = (0..16u32)
        .find(|&site| {
            let mut probe = ClosureHistory::new(clean.clone()).unwrap();
            probe.perturb_spin(site).unwrap();
            let l = probe.localize(&probe.break_report().unwrap());
            l.unique && l.sites[0] == site
        })
        .expect("a grid names some of its sites exactly");

    let mut history = ClosureHistory::new(clean.clone()).unwrap();
    history.stamp().unwrap();
    history.perturb_spin(target).unwrap();
    history.stamp().unwrap();

    let retro = history.retrodict().unwrap();
    let report = history.correct(&retro).unwrap();
    assert_eq!(report.site, target);
    assert!(report.broken_before > 0);
    assert_eq!(report.broken_after, 0);
    assert!(report.restored);
    assert!(history.break_report().unwrap().intact());

    // Not merely closure: the state the bundle denotes is restored
    // exactly.
    let repaired = history.into_bundle();
    let before = clean.to_state().unwrap();
    let after = repaired.to_state().unwrap();
    assert_eq!(
        max_amplitude_deviation(before.as_ref(), after.as_ref()),
        0.0
    );
}

#[test]
fn correction_refuses_to_guess() {
    // Ambiguous: a ring gives six candidates and no reason to prefer
    // one.
    let mut history = ClosureHistory::new(ring(6)).unwrap();
    history.stamp().unwrap();
    history.perturb_spin(2).unwrap();
    history.stamp().unwrap();
    let retro = history.retrodict().unwrap();
    let err = history.correct(&retro).unwrap_err();
    assert!(format!("{err}").contains("will not invent one"), "{err}");

    // Nothing broken: nothing to correct.
    let mut clean = ClosureHistory::new(grid(4, 4)).unwrap();
    clean.stamp().unwrap();
    let retro = clean.retrodict().unwrap();
    let err = clean.correct(&retro).unwrap_err();
    assert!(format!("{err}").contains("nothing to correct"), "{err}");
}

// ── the chirality channel ────────────────────────────────────────────

#[test]
fn an_ordering_break_localizes_to_a_link_not_a_site() {
    let mut history = ClosureHistory::new(grid(4, 4)).unwrap();
    assert!(history.break_report().unwrap().intact());

    // Two adjacent positions swap without the history accounting for
    // it. Positions 5 and 6 hold sites 5 and 6, which are linked.
    history.perturb_order(5).unwrap();
    let broken = history.break_report().unwrap();
    assert!(broken.spin.is_empty(), "no fiber sign moved");
    assert!(!broken.chirality.is_empty());

    let localization = history.localize(&broken);
    // The chirality channel names the link the ordering crossed.
    assert!(localization.sites.is_empty());
    assert_eq!(localization.links, vec![(5, 6)]);
    assert!(localization.unique);

    // And correction declines: the link is named, the transposition is
    // not.
    let retro = history.retrodict().unwrap();
    let err = history.correct(&retro).unwrap_err();
    assert!(format!("{err}").contains("names the link"), "{err}");
}

#[test]
fn a_deliberate_reordering_keeps_closure() {
    let mut history = ClosureHistory::new(grid(4, 4)).unwrap();
    for position in 0..8 {
        history.swap_order(position).unwrap();
        assert!(
            history.break_report().unwrap().intact(),
            "intended reordering at {position} broke closure"
        );
    }
}

// ── the history carries its entanglement ─────────────────────────────

#[test]
fn the_history_reconstructs_past_entanglement_not_just_past_signs() {
    let mut bundle = PolarityBundle::new(6).unwrap();
    bundle.link(0, 1).unwrap();
    bundle.link(1, 2).unwrap();
    bundle.link(2, 0).unwrap();
    let early_links = bundle.profile().links;
    let early_signature = bundle.link_signature();

    // Grow the entanglement, then look back.
    bundle.link(3, 4).unwrap();
    bundle.link(4, 5).unwrap();
    bundle.link(5, 3).unwrap();
    assert_eq!(bundle.profile().links, 6);

    let past = bundle.rewind(3).unwrap();
    assert_eq!(past.profile().links, early_links);
    assert_eq!(past.link_signature(), early_signature);

    // And closure is a property of that past geometry, recomputable.
    assert_eq!(closure_basis(&past).rank, 1);
    assert_eq!(closure_basis(&bundle).rank, 2);
}

// ── scale ────────────────────────────────────────────────────────────

#[test]
fn closure_runs_on_a_geometry_no_amplitude_could_hold() {
    // A 120×120 grid: 14 400 sites, 14 161 independent loops, and a
    // stamp of 28 322 bits — about 3.5 kB — is the entire cost of
    // being able to retrodict to this moment.
    let bundle = grid(120, 120);
    let mut history = ClosureHistory::new(bundle).unwrap();
    let basis_rank = history.basis().rank;
    assert_eq!(basis_rank, 28560 - 14400 + 1);

    history.stamp().unwrap();
    assert_eq!(history.stamp_bits(), 2 * basis_rank);
    assert!(history.stamp_bits() < 32_000);

    history.perturb_spin(7_000).unwrap();
    history.stamp().unwrap();
    let retro = history.retrodict().unwrap();
    assert!(retro.step.is_some());
    assert!(!retro.localization.broken.intact());
    assert!(retro.localization.sites.contains(&7_000));

    // The honest cost of evaluating closure is the total loop length,
    // which a spanning-forest basis does not keep small.
    assert!(history.basis().total_length > basis_rank);
}
