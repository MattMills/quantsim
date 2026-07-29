//! Polarity as a fibered, re-orderable, journalled co-bundle, measured:
//! the GHZ sign living in a fiber (which is where the previous module's
//! marginal obstruction does not reach), chirality as a consequence of
//! twist plus ordering, entanglement as a budgeted resource, interaction
//! confined to commonality, journal replay, fiber tomography against
//! real simulated states, and the whole thing at 10⁵–10⁶ sites.

mod common;

use quantsim::bundle::*;
use quantsim::polarity::{pairwise_signature, PolaritySystem};
use quantsim::prelude::*;

fn overlap(a: &dyn Backend<C64>, b: &dyn Backend<C64>) -> f64 {
    let mut acc = C64::new(0.0, 0.0);
    a.for_each_nonzero(&mut |i, x| acc += x.conj() * b.amplitude(i));
    acc.norm()
}

// ── the correction: the sign is fiber data ───────────────────────────

#[test]
fn the_ghz_sign_lives_in_a_fiber_not_in_the_base() {
    for n in [3usize, 4, 6, 8] {
        let plus = ghz_bundle(n, false).unwrap();
        let minus = ghz_bundle(n, true).unwrap();

        // The base — the entanglement structure — is bit-for-bit the
        // same. Only one fiber's sign differs.
        assert_eq!(plus.link_signature(), minus.link_signature());
        assert_eq!(plus.profile().links, minus.profile().links);
        assert_eq!(plus.profile().links, n - 1);
        assert!(!plus.inspect(0).unwrap().fiber.spin);
        assert!(minus.inspect(0).unwrap().fiber.spin);
        for q in 1..n as u32 {
            assert_eq!(
                plus.inspect(q).unwrap().fiber,
                minus.inspect(q).unwrap().fiber
            );
        }

        // The states they denote really are GHZ± — and orthogonal.
        let sp = plus.to_state().unwrap();
        let sm = minus.to_state().unwrap();
        let top = (1u64 << n) - 1;
        common::assert_close(sp.probability(0), 0.5, 1e-12);
        common::assert_close(sp.probability(top), 0.5, 1e-12);
        common::assert_close(sp.amplitude(top).re, -sm.amplitude(top).re, 1e-12);
        assert!(overlap(sp.as_ref(), sm.as_ref()) < 1e-14);

        // Two-body marginals remain blind — the earlier measurement is
        // reproduced here, unchanged — while the bundle is not.
        let mp = pairwise_signature(sp.as_ref()).unwrap();
        let mm = pairwise_signature(sm.as_ref()).unwrap();
        assert_eq!(mp.max_deviation(&mm), 0.0);

        // Fiber tomography reads the sign straight back off the state.
        let read_plus = plus.verify_against(sp.as_ref()).unwrap();
        let read_minus = plus.verify_against(sm.as_ref()).unwrap();
        assert!(read_plus.deviation < 1e-12);
        assert!(read_minus.deviation < 1e-12);
        assert!(!read_plus.spins[0]);
        assert!(
            read_minus.spins[0],
            "the − state must read as a flipped fiber"
        );
        assert_eq!(read_plus.spins[1..], read_minus.spins[1..]);
    }
}

#[test]
fn tomography_round_trips_a_mixed_frame_bundle() {
    let mut bundle = PolarityBundle::new(8).unwrap();
    for (a, b) in [
        (0u32, 1u32),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (2, 7),
    ] {
        bundle.link(a, b).unwrap();
    }
    bundle.set_frame(3, Frame::X).unwrap();
    bundle.set_frame(6, Frame::Y).unwrap();
    bundle.set_spin(2, true).unwrap();
    bundle.set_spin(5, true).unwrap();

    let state = bundle.to_state().unwrap();
    let read = bundle.verify_against(state.as_ref()).unwrap();
    // Every stabilizer generator reads ±1: the state is in the sector.
    assert!(read.deviation < 1e-12, "{}", read.deviation);
    let expected: Vec<bool> = (0..8)
        .map(|q| bundle.inspect(q).unwrap().fiber.spin)
        .collect();
    assert_eq!(read.spins, expected);
}

#[test]
fn a_state_outside_the_sector_is_reported_not_smoothed_over() {
    let mut bundle = PolarityBundle::new(4).unwrap();
    bundle.link(0, 1).unwrap();
    bundle.link(2, 3).unwrap();

    // The state it denotes verifies cleanly...
    let good = bundle.to_state().unwrap();
    assert!(bundle.verify_against(good.as_ref()).unwrap().deviation < 1e-12);

    // ...and a T-rotated one does not. The bundle says so.
    let mut c: Circuit = Circuit::new(4);
    c.h(0).h(1).h(2).h(3).cz(0, 1).cz(2, 3).t(0);
    let escaped = Simulator::new().run(&c).unwrap();
    let read = bundle.verify_against(escaped.as_ref()).unwrap();
    assert!(read.deviation > 0.1, "deviation {}", read.deviation);

    // Width mismatches refuse rather than reading garbage.
    assert!(bundle.verify_against(good.as_ref()).is_ok());
    let wide = PolarityBundle::new(5).unwrap().to_state().unwrap();
    assert!(bundle.verify_against(wide.as_ref()).is_err());
}

// ── twist, ordering, chirality ───────────────────────────────────────

#[test]
fn chirality_is_the_reordering_sign_the_twist_forces() {
    // A bundle's chirality after a reordering must equal the sign
    // relating the ordered products of the corresponding polarity
    // generators — measured against `PolaritySystem`, not asserted.
    let links = [(0usize, 1usize), (1, 2), (0, 3), (2, 4)];
    let n = 5;
    let system = PolaritySystem::new(n, &links, &[]).unwrap();

    let ordered_sign = |order: &[u32]| -> f64 {
        let mut mask = 0u64;
        let mut sign = 1.0;
        for &site in order {
            let (m, s) = system.product(mask, 1u64 << site);
            mask = m;
            sign *= s;
        }
        sign
    };

    for order in [
        vec![0u32, 1, 2, 3, 4],
        vec![4, 3, 2, 1, 0],
        vec![2, 0, 4, 1, 3],
        vec![1, 3, 0, 2, 4],
    ] {
        let mut bundle = PolarityBundle::new(n).unwrap();
        for &(a, b) in &links {
            bundle.link(a as u32, b as u32).unwrap();
        }
        bundle.reorder(&order).unwrap();
        let expected = ordered_sign(&order) / ordered_sign(&[0, 1, 2, 3, 4]);
        assert_eq!(
            bundle.chirality() as f64,
            expected,
            "order {order:?}: bundle {} vs algebra {expected}",
            bundle.chirality()
        );
    }
}

#[test]
fn adjacent_swaps_agree_with_a_wholesale_reorder() {
    let mut a = PolarityBundle::new(4).unwrap();
    let mut b = PolarityBundle::new(4).unwrap();
    for (x, y) in [(0u32, 1u32), (1, 2), (2, 3)] {
        a.link(x, y).unwrap();
        b.link(x, y).unwrap();
    }
    // Bubble 0 to the end one adjacent swap at a time.
    for pos in 0..3 {
        a.swap_order(pos).unwrap();
    }
    b.reorder(&[1, 2, 3, 0]).unwrap();
    assert_eq!(a.order(), b.order());
    assert_eq!(a.chirality(), b.chirality());

    assert!(a.swap_order(3).is_err());
    assert!(b.reorder(&[0, 0, 1, 2]).is_err());
    assert!(b.reorder(&[0, 1, 2]).is_err());
}

#[test]
fn the_forward_census_follows_the_ordering() {
    let mut bundle = PolarityBundle::new(4).unwrap();
    for (a, b) in [(0u32, 1u32), (0, 2), (0, 3)] {
        bundle.link(a, b).unwrap();
    }
    // Centre first: it sees all three links running forward.
    assert_eq!(bundle.forward_census(), vec![3, 0, 0, 0]);
    // Centre last: none do.
    bundle.reorder(&[1, 2, 3, 0]).unwrap();
    assert_eq!(bundle.forward_census(), vec![1, 1, 1, 0]);
}

// ── entanglement as a managed resource ───────────────────────────────

#[test]
fn coarse_graining_spends_links_and_accounts_for_every_one() {
    let mut bundle = PolarityBundle::new(9).unwrap();
    // Two triangles joined by a bridge.
    for (a, b) in [(0u32, 1u32), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)] {
        bundle.link(a, b).unwrap();
    }
    let before = bundle.profile();
    assert_eq!(before.links, 7);
    assert_eq!(before.components, 4); // two triangles joined, plus 6,7,8

    let report = bundle.coarse_grain(&[0, 1, 2]).unwrap();
    assert_eq!(report.absorbed_fibers, 2);
    assert_eq!(report.absorbed_links, 3); // the triangle's own edges
    let after = bundle.profile();
    assert_eq!(after.links, 4); // three triangle edges gone
    assert_eq!(after.live_sites, 7);
    assert_eq!(bundle.inspect(0).unwrap().fiber.weight, 3);
    assert!(!bundle.inspect(1).unwrap().fiber.live);
    // The bridge survived, re-pointed onto the survivor.
    assert!(bundle.linked(0, 3));
    // Absorbed fibers are no longer addressable.
    assert!(bundle.link(1, 4).is_err());
    assert!(bundle.coarse_grain(&[0]).is_err());
}

#[test]
fn parallel_links_collapse_and_are_counted_separately() {
    // Two members sharing one external neighbour: merging them removes
    // two links and leaves one. That collapse is entanglement spent by
    // a different route, and it is reported on its own line.
    let mut bundle = PolarityBundle::new(4).unwrap();
    bundle.link(0, 1).unwrap();
    bundle.link(0, 3).unwrap();
    bundle.link(1, 3).unwrap();
    let report = bundle.coarse_grain(&[0, 1]).unwrap();
    assert_eq!(report.absorbed_links, 1); // the 0–1 edge
    assert_eq!(report.collapsed_links, 1); // two edges to 3 became one
    assert_eq!(bundle.profile().links, 1);
    assert_eq!(bundle.profile().absorbed_links, 2);
}

#[test]
fn a_link_budget_is_met_and_priced() {
    let mut bundle = scaling_bundle(2000, 2).unwrap();
    let start = bundle.profile();
    assert!(start.links > 5000);

    for budget in [3000usize, 1000, 200] {
        let report = bundle.coarse_grain_to_budget(budget).unwrap();
        let now = bundle.profile();
        assert!(now.links <= budget, "budget {budget}: {} links", now.links);
        assert!(report.merges > 0);
        // Everything removed is accounted for on one of the two lines.
        assert!(report.absorbed_links + report.collapsed_links > 0);
        // And the surviving fibers still cover every original site.
        let covered: u32 = (0..bundle.sites() as u32)
            .filter_map(|s| bundle.inspect(s).ok())
            .filter(|v| v.fiber.live)
            .map(|v| v.fiber.weight)
            .sum();
        assert_eq!(covered as usize, 2000);
    }
}

#[test]
fn the_profile_separates_independent_entangled_clusters() {
    let mut bundle = PolarityBundle::new(12).unwrap();
    for (a, b) in [(0u32, 1u32), (1, 2), (2, 3)] {
        bundle.link(a, b).unwrap();
    }
    for (a, b) in [(6u32, 7u32), (7, 8)] {
        bundle.link(a, b).unwrap();
    }
    let p = bundle.profile();
    // Two clusters (sizes 4 and 3) plus five isolated fibers.
    assert_eq!(p.components, 7);
    assert_eq!(p.largest_component, 4);
    assert_eq!(p.links, 5);
    assert_eq!(p.max_degree, 2);
    let histogram = degree_histogram(&bundle);
    assert_eq!(histogram.get(&0), Some(&5));
    assert_eq!(histogram.get(&1), Some(&4));
    assert_eq!(histogram.get(&2), Some(&3));
}

// ── the co-bundle: interaction on commonality ────────────────────────

#[test]
fn interaction_is_confined_to_the_commonality() {
    let mut a = PolarityBundle::new(6).unwrap();
    for (x, y) in [(0u32, 1u32), (1, 2), (3, 4)] {
        a.link(x, y).unwrap();
    }
    a.set_frame(4, Frame::X).unwrap();

    let mut b = PolarityBundle::new(6).unwrap();
    for (x, y) in [(1u32, 2u32), (2, 3), (4, 5)] {
        b.link(x, y).unwrap();
    }
    b.set_frame(4, Frame::Y).unwrap();
    b.set_spin(0, true).unwrap();

    let common = a.commonality(&b);
    assert_eq!(common.shared_sites, 6);
    assert_eq!(common.shared_links, 1); // 1–2
                                        // Site 4 carries X against Y: incompatible, so out of the interaction.
    assert_eq!(common.incompatible_fibers, 1);
    assert_eq!(common.compatible_fibers, 5);
    assert_eq!(common.interactable, vec![0, 1, 2, 3, 5]);

    let before_45 = a.linked(4, 5);
    a.interact(&b).unwrap();
    // Toggled inside the commonality: 1–2 cancelled, 2–3 added.
    assert!(!a.linked(1, 2));
    assert!(a.linked(2, 3));
    // Untouched outside it: 4–5 lives on an incompatible fiber.
    assert_eq!(a.linked(4, 5), before_45);
    assert!(a.linked(3, 4)); // a's own link, never in b
                             // Fiber signs XOR on the interactable set only.
    assert!(a.inspect(0).unwrap().fiber.spin);
}

#[test]
fn frame_compatibility_is_what_gates_the_interaction() {
    assert!(Frame::Z.compatible(Frame::X));
    assert!(Frame::X.compatible(Frame::X));
    assert!(!Frame::X.compatible(Frame::Y));
    assert!(Frame::Y.compatible(Frame::Z));
}

// ── the journal ──────────────────────────────────────────────────────

#[test]
fn the_journal_is_the_state() {
    let mut bundle = PolarityBundle::new(6).unwrap();
    bundle.link(0, 1).unwrap();
    bundle.set_frame(2, Frame::X).unwrap();
    bundle.link(2, 3).unwrap();
    bundle.set_spin(1, true).unwrap();
    bundle.swap_order(0).unwrap();
    bundle.coarse_grain(&[2, 3]).unwrap();
    assert_eq!(bundle.journal().len(), 6);

    // Replaying the whole journal reproduces the bundle exactly.
    let replayed = bundle.rewind(bundle.journal().len()).unwrap();
    assert_eq!(replayed.link_signature(), bundle.link_signature());
    assert_eq!(replayed.chirality(), bundle.chirality());
    assert_eq!(replayed.profile().links, bundle.profile().links);
    assert_eq!(replayed.profile().live_sites, bundle.profile().live_sites);
    for s in 0..6u32 {
        assert_eq!(
            replayed.inspect(s).unwrap().fiber,
            bundle.inspect(s).unwrap().fiber
        );
    }

    // And any prefix is the bundle as it stood then.
    let early = bundle.rewind(2).unwrap();
    assert_eq!(early.profile().links, 1);
    assert_eq!(early.inspect(2).unwrap().fiber.frame, Frame::X);
    assert!(!early.inspect(1).unwrap().fiber.spin);
    assert!(early.inspect(3).unwrap().fiber.live);

    assert!(bundle.rewind(99).is_err());
}

#[test]
fn a_coarse_grained_bundle_refuses_to_pretend_it_still_denotes_a_state() {
    let mut bundle = ghz_bundle(5, false).unwrap();
    assert!(bundle.to_state().is_ok());
    bundle.coarse_grain(&[1, 2]).unwrap();
    match bundle.to_state() {
        Ok(_) => panic!("a coarse-grained bundle must not materialize a state"),
        Err(e) => assert!(format!("{e}").contains("coarse-grained"), "{e}"),
    }
    // Rewinding past the merge restores a materializable bundle.
    let before = bundle.rewind(bundle.journal().len() - 1).unwrap();
    assert!(before.to_state().is_ok());
}

// ── scale ────────────────────────────────────────────────────────────

#[test]
fn the_bundle_runs_where_no_amplitude_could() {
    // A hundred thousand sites: the state vector would be 2^100000
    // amplitudes and does not exist. The bundle is a few tens of MB and
    // every operation stays sparse.
    let sites = 100_000usize;
    let mut bundle = scaling_bundle(sites, 2).unwrap();
    let profile = bundle.profile();
    assert_eq!(profile.sites, sites);
    assert_eq!(profile.live_sites, sites);
    assert!(profile.links >= sites, "{} links", profile.links);
    assert!(profile.max_degree <= 8);
    // Sparse storage: the structure is linear in sites + links, not
    // exponential. The journal is the dominant term and is reported
    // separately, because auditability is a cost with a name.
    let structure = profile.bytes - profile.journal_bytes;
    assert!(
        structure < 128 * sites,
        "{structure} structural bytes for {sites} sites"
    );
    assert!(
        profile.journal_bytes > structure / 2,
        "the journal should dominate here"
    );

    // Checkpointing declines to pay for it.
    let dropped = bundle.checkpoint();
    assert_eq!(dropped, profile.journal_ops);
    let lean = bundle.profile();
    assert_eq!(lean.journal_bytes, 0);
    assert!(lean.bytes < profile.bytes / 2);
    // ...and rewinding past the checkpoint is no longer possible.
    assert_eq!(bundle.journal().len(), 0);

    // Ordering the whole thing is one linear pass.
    let reversed: Vec<u32> = (0..sites as u32).rev().collect();
    bundle.reorder(&reversed).unwrap();
    assert!(bundle.chirality() == 1 || bundle.chirality() == -1);
    assert_eq!(bundle.order()[0], sites as u32 - 1);

    // Inspection stays local at any depth.
    let view = bundle.inspect(50_000).unwrap();
    assert_eq!(view.site, 50_000);
    assert!(!view.links.is_empty());

    // And the amplitude picture correctly refuses.
    match bundle.to_state() {
        Ok(_) => panic!("100k qubits cannot be a dense state"),
        Err(e) => assert!(matches!(e, quantsim::Error::TooManyQubits { .. })),
    }
}

#[test]
fn degenerate_bundles_are_refused_by_reason() {
    assert!(PolarityBundle::new(0).is_err());
    assert!(ghz_bundle(1, false).is_err());
    let bundle = PolarityBundle::new(3).unwrap();
    assert!(bundle.inspect(7).is_err());
}
