//! Graph-state measurement on the polarity bundle: the collapse is done in
//! the description, and it has to agree with projecting the materialized
//! state — for every graph shape, every decoration, every site and both
//! outcomes, and across sequences of measurements.

use quantsim::bundle::{vop, PolarityBundle};
use quantsim::prelude::*;

/// Dense reference: materialize, project qubit `q` onto `outcome`,
/// renormalize. `None` when the outcome has no weight.
fn dense_collapse(state: &dyn Backend<C64>, q: usize, outcome: bool) -> Option<Vec<C64>> {
    let n = state.num_qubits();
    let mut amps = vec![C64::new(0.0, 0.0); 1 << n];
    state.for_each_nonzero(&mut |i, a| amps[i as usize] = a);
    let mask = 1u64 << q;
    let weight: f64 = amps
        .iter()
        .enumerate()
        .filter(|(i, _)| ((*i as u64 & mask) != 0) == outcome)
        .map(|(_, a)| a.norm_sqr())
        .sum();
    if weight <= 1e-15 {
        return None;
    }
    let scale = C64::new(1.0 / weight.sqrt(), 0.0);
    for (i, a) in amps.iter_mut().enumerate() {
        if ((i as u64 & mask) != 0) == outcome {
            *a *= scale;
        } else {
            *a = C64::new(0.0, 0.0);
        }
    }
    Some(amps)
}

fn amps_of(bundle: &PolarityBundle) -> Vec<C64> {
    let state = bundle.to_state().expect("materialize");
    let n = state.num_qubits();
    let mut amps = vec![C64::new(0.0, 0.0); 1 << n];
    state.for_each_nonzero(&mut |i, a| amps[i as usize] = a);
    amps
}

/// Worst amplitude difference after dividing out one global phase.
fn deviation_up_to_phase(want: &[C64], got: &[C64]) -> f64 {
    let pivot = want
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.norm().total_cmp(&y.1.norm()))
        .map(|(i, _)| i)
        .expect("non-empty");
    if want[pivot].norm() < 1e-12 || got[pivot].norm() < 1e-12 {
        return want
            .iter()
            .zip(got)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max);
    }
    let ratio = want[pivot] / got[pivot];
    let phase = ratio / C64::new(ratio.norm(), 0.0);
    want.iter()
        .zip(got)
        .map(|(x, y)| (x - y * phase).norm())
        .fold(0.0f64, f64::max)
}

fn shapes(n: u32) -> Vec<(&'static str, Vec<(u32, u32)>)> {
    let path: Vec<(u32, u32)> = (0..n - 1).map(|a| (a, a + 1)).collect();
    let star: Vec<(u32, u32)> = (1..n).map(|a| (0, a)).collect();
    let mut cycle = path.clone();
    cycle.push((0, n - 1));
    let mut complete = Vec::new();
    for a in 0..n {
        for b in (a + 1)..n {
            complete.push((a, b));
        }
    }
    let mut broken = path.clone();
    broken.pop();
    vec![
        ("empty", vec![]),
        ("single-edge", vec![(0, 1)]),
        ("path", path),
        ("star", star),
        ("cycle", cycle),
        ("complete", complete),
        ("path-minus-one", broken),
    ]
}

/// Build a decorated bundle. `decoration`: 0 bare, 1 vops, 2 spins, 3 both.
fn decorated(n: u32, edges: &[(u32, u32)], decoration: usize, seed: usize) -> PolarityBundle {
    let mut bundle = PolarityBundle::new(n as usize).expect("sites");
    for &(a, b) in edges {
        bundle.link(a, b).expect("link");
    }
    for q in 0..n {
        if decoration == 1 || decoration == 3 {
            let v = (((q as usize + 1) * seed) % vop::count()) as vop::Vop;
            bundle.apply_vop(q, v).expect("vop");
        }
        if (decoration == 2 || decoration == 3) && (q as usize + seed) % 2 == 0 {
            bundle.set_spin(q, true).expect("spin");
        }
    }
    bundle
}

#[test]
fn the_collapse_agrees_with_projecting_the_materialized_state() {
    let n = 4u32;
    let mut worst = 0.0f64;
    let mut cases = 0usize;
    for (name, edges) in shapes(n) {
        for decoration in 0..4usize {
            for seed in [7usize, 10, 13] {
                for site in 0..n {
                    for outcome in [false, true] {
                        let mut bundle = decorated(n, &edges, decoration, seed);
                        let reference = bundle.to_state().expect("materialize");
                        let want = dense_collapse(reference.as_ref(), site as usize, outcome);
                        let probability = bundle.collapse(site, outcome).expect("collapse");
                        cases += 1;
                        match want {
                            None => assert!(
                                probability <= 1e-12,
                                "{name}/dec{decoration}/seed{seed}/site{site}/{outcome}: \
                                 dense says the outcome is impossible, the bundle says \
                                 p = {probability}"
                            ),
                            Some(want) => {
                                assert!(probability > 1e-12);
                                let dev = deviation_up_to_phase(&want, &amps_of(&bundle));
                                worst = worst.max(dev);
                                assert!(
                                    dev < 1e-9,
                                    "{name}/dec{decoration}/seed{seed}/site{site}/{outcome}: \
                                     deviation {dev:.3e}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(cases >= 600, "the sweep should be broad: {cases} cases");
    assert!(
        worst < 1e-9,
        "worst deviation {worst:.3e} over {cases} cases"
    );
}

#[test]
fn a_sequence_of_collapses_stays_exact() {
    // One measurement being right is not enough — the post-measurement
    // description has to be a description a further measurement can use.
    let n = 5u32;
    let mut worst = 0.0f64;
    for (name, edges) in shapes(n) {
        for decoration in 0..4usize {
            for pattern in 0..8u32 {
                let mut bundle = decorated(n, &edges, decoration, 11);
                let mut reference = bundle.to_state().expect("materialize");
                let mut alive = true;
                for site in 0..n.min(3) {
                    let outcome = pattern >> site & 1 == 1;
                    let want = dense_collapse(reference.as_ref(), site as usize, outcome);
                    let probability = bundle.collapse(site, outcome).expect("collapse");
                    match want {
                        None => {
                            assert!(probability <= 1e-12);
                            alive = false;
                            break;
                        }
                        Some(want) => {
                            let dev = deviation_up_to_phase(&want, &amps_of(&bundle));
                            worst = worst.max(dev);
                            assert!(
                                dev < 1e-9,
                                "{name}/dec{decoration}/pattern{pattern}/after site {site}: \
                                 deviation {dev:.3e}"
                            );
                            let mut next = DenseState::<C64>::new(n as usize).expect("dense");
                            next.load(
                                &want
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, a)| a.norm() > 0.0)
                                    .map(|(i, &a)| (i as u64, a))
                                    .collect::<Vec<_>>(),
                            )
                            .expect("load");
                            reference = Box::new(next);
                        }
                    }
                }
                let _ = alive;
            }
        }
    }
    assert!(worst < 1e-9, "worst deviation {worst:.3e}");
}

#[test]
fn measuring_collapses_the_state_rather_than_only_reporting_an_outcome() {
    // The bug this fixes: `measure` used to draw a correct outcome and then
    // leave the state alone, so a second measurement of the same qubit
    // could disagree with the first. It must not.
    let mut bundle = quantsim::bundle::ghz_bundle(6, false).expect("ghz");
    let mut rng = Prng::new(0xC0FFEE);
    let first = Backend::<C64>::measure(&mut bundle, 0, &mut rng).expect("measure");
    for _ in 0..8 {
        let again = Backend::<C64>::measure(&mut bundle, 0, &mut rng).expect("measure");
        assert_eq!(
            again, first,
            "a collapsed qubit must keep returning the outcome it collapsed to"
        );
    }
    // And on a GHZ state every other qubit must agree with the first.
    for q in 1..6 {
        let other = Backend::<C64>::measure(&mut bundle, q, &mut rng).expect("measure");
        assert_eq!(other, first, "GHZ outcomes are perfectly correlated");
    }
}

#[test]
fn a_measured_bundle_still_denotes_the_state_it_says_it_does() {
    // Tomography against the bundle's own stabilizer generators: after a
    // collapse the description must still be a consistent graph state.
    let mut rng = Prng::new(0x51DE);
    for shape in 0..4usize {
        let mut bundle = decorated(5, &shapes(5)[shape + 2].1, 3, 9);
        for site in 0..3u32 {
            let outcome =
                Backend::<C64>::measure(&mut bundle, site as usize, &mut rng).expect("measure");
            let state = bundle.to_state().expect("materialize");
            let report = bundle.verify_against(state.as_ref()).expect("tomography");
            assert!(
                report.deviation < 1e-9,
                "shape {shape}, site {site} -> {outcome}: the description drifted from \
                 the state by {:.3e}",
                report.deviation
            );
        }
    }
}

#[test]
fn an_isolated_fiber_can_be_deterministic_and_says_so() {
    // A site with no links carries a product state, so an outcome can be
    // certain — and the impossible one is reported as probability zero
    // rather than silently collapsed to.
    let mut bundle = PolarityBundle::new(2).expect("sites");
    // reset-equivalent: a bare bundle's fibers are |+>, so both outcomes
    // are even. Put site 0 into |0> by making its operator a Hadamard.
    bundle.apply_vop(0, vop::hadamard()).expect("vop");
    let mut probe = bundle.clone();
    assert!((probe.collapse(0, false).expect("collapse") - 1.0).abs() < 1e-12);
    let mut probe = bundle.clone();
    assert_eq!(probe.collapse(0, true).expect("collapse"), 0.0);
    // And the impossible collapse left the description untouched.
    let peak = amps_of(&probe)
        .iter()
        .map(|a| a.norm())
        .fold(0.0f64, f64::max);
    assert!(
        peak > 0.5,
        "an impossible projection must not zero the state; peak amplitude {peak}"
    );

    // An undecorated fiber is unbiased.
    let mut bundle = PolarityBundle::new(2).expect("sites");
    assert!((bundle.collapse(0, true).expect("collapse") - 0.5).abs() < 1e-12);
}

#[test]
fn the_collapse_is_journalled_and_replays() {
    let mut bundle = quantsim::bundle::ghz_bundle(5, true).expect("ghz");
    let before = bundle.journal().len();
    bundle.collapse(2, true).expect("collapse");
    let journal = bundle.journal();
    assert!(journal.len() > before);
    assert!(
        journal.iter().any(|op| matches!(
            op,
            quantsim::bundle::BundleOp::Collapse {
                site: 2,
                outcome: true,
                ..
            }
        )),
        "the collapse must appear in the journal: {journal:?}"
    );
    // Replaying the whole journal reproduces the same state.
    let replayed = bundle.rewind(journal.len()).expect("rewind");
    let dev = deviation_up_to_phase(&amps_of(&bundle), &amps_of(&replayed));
    assert!(dev < 1e-9, "replay deviated by {dev:.3e}");
    // And rewinding to before the collapse restores the entanglement.
    let earlier = bundle.rewind(before).expect("rewind");
    assert!(
        earlier.profile().links > 0,
        "the pre-collapse description still holds its links"
    );
}

#[test]
fn measurement_probabilities_match_the_materialized_distribution() {
    // The native `measure` computes its bias from the description. Over
    // many shots it must reproduce the dense distribution.
    let mut bundle = PolarityBundle::new(4).expect("sites");
    bundle.link(0, 1).expect("link");
    bundle.link(1, 2).expect("link");
    bundle.link(2, 3).expect("link");
    bundle.apply_vop(1, vop::hadamard()).expect("vop");
    let reference = bundle.to_state().expect("materialize");
    let mut expected = [0.0f64; 4];
    reference.for_each_nonzero(&mut |i, a: C64| {
        for (q, slot) in expected.iter_mut().enumerate() {
            if i >> q & 1 == 1 {
                *slot += a.norm_sqr();
            }
        }
    });

    let shots = 4000;
    let mut counts = [0u64; 4];
    let mut rng = Prng::new(0xBEEF);
    for _ in 0..shots {
        let mut shot = bundle.clone();
        for (q, slot) in counts.iter_mut().enumerate() {
            if Backend::<C64>::measure(&mut shot, q, &mut rng).expect("measure") {
                *slot += 1;
            }
        }
    }
    for (q, (&count, &want)) in counts.iter().zip(&expected).enumerate() {
        let got = count as f64 / shots as f64;
        assert!(
            (got - want).abs() < 0.05,
            "qubit {q}: sampled {got:.3}, dense says {want:.3}"
        );
    }
}
