//! The composite register: one mixed-radix space held as a product of
//! regions in heterogeneous backends, the partition following the space
//! as it changes. Conformance over the full gate registry from singleton
//! qubit regions; agreement with the radix-dense store along random paths
//! of spaces (gates, measurements, merges, splits, grows, releases,
//! promotes, demotes, lossy steps opted in) with the dropped weight equal to the dense
//! store's; grow as a new region that touches no amplitude; a workspace
//! re-separated by the rank-1 test before its release, and a reused
//! workspace that stays one pair only because regions re-separate; refusals and drops
//! carrying the **global** weight; measurement splitting the measured
//! site out exactly and seed-for-seed with dense; an MPS region and radix
//! regions in one register, the crossing ledgered; and a 40-qutrit stream
//! of virtual dimension `3^40` held in a few kilobytes.

mod common;

use common::{assert_close, sim};
use quantsim::backend::{max_amplitude_deviation, CompositeState, MpsState};
use quantsim::prelude::*;
use quantsim::space::{cshift, fourier};

fn c(re: f64) -> C64 {
    C64::new(re, 0.0)
}

fn composite(radices: &[usize]) -> CompositeState<C64> {
    CompositeState::new(Space::new(radices).unwrap()).unwrap()
}

/// One random unitary on one or two random sites of `space`.
fn random_step(space: &Space, rng: &mut Prng) -> (Vec<usize>, SiteMatrix<C64>) {
    let n = space.sites();
    let pick = |rng: &mut Prng, n: usize| (rng.next_f64() * n as f64) as usize % n;
    let a = pick(rng, n);
    let sites = if n == 1 || rng.next_f64() < 0.5 {
        vec![a]
    } else {
        let mut b = pick(rng, n);
        if b == a {
            b = (a + 1) % n;
        }
        vec![a, b]
    };
    let d = space.sub_dim(&sites);
    let m = SiteMatrix::from_vec(d, common::random_unitary(d, rng)).unwrap();
    (sites, m)
}

/// A random legal transition from `space`, keeping the register small
/// enough for the dense store to follow.
fn random_transition(space: &Space, rng: &mut Prng) -> Transition {
    let pick = |rng: &mut Prng, n: usize| (rng.next_f64() * n as f64) as usize % n;
    loop {
        let n = space.sites();
        let site = pick(rng, n);
        let r = space.radix(site);
        let t = match pick(rng, 6) {
            0 => Transition::Merge { site },
            1 => Transition::Split {
                site,
                low: 2 + pick(rng, 3),
            },
            2 => Transition::Grow {
                at: pick(rng, n + 1),
                radix: 2 + pick(rng, 3),
            },
            3 => Transition::Release { site },
            4 => Transition::Promote {
                site,
                radix: r + 1 + pick(rng, 2),
            },
            _ => Transition::Demote {
                site,
                radix: 2 + pick(rng, r.saturating_sub(2).max(1)),
            },
        };
        if let Ok(to) = t.target(space) {
            if to.dim() <= 2000 && to.sites() <= 6 && to.radices().iter().all(|&r| r <= 12) {
                return t;
            }
        }
    }
}

// ── conformance ──────────────────────────────────────────────────────

#[test]
fn composite_conforms_over_the_full_registry() {
    let report = verify_backend(&sim(), "composite", &ConformanceConfig::default()).unwrap();
    assert!(report.passed(), "{report}");
    assert!(report.max_amplitude_deviation < 1e-9, "{report}");
    assert_eq!(report.sampling_mismatches, 0);
    assert_eq!(report.collapse_violations, 0);
}

#[test]
fn composite_agrees_with_dense_along_random_paths_of_spaces() {
    let mut separations = 0;
    let mut merges = 0;
    for seed in 0..12u64 {
        let mut rng = Prng::new(900 + seed);
        let mut space = Space::new(&[2, 3, 2]).unwrap();
        let mut dense = RadixDenseState::<C64>::new(space.clone()).unwrap();
        let mut comp = CompositeState::<C64>::new(space.clone()).unwrap();
        let (mut lost_dense, mut lost_comp) = (0.0, 0.0);
        for _ in 0..80 {
            // Sparse in two-site gates so regions have room to separate.
            let roll = rng.next_f64();
            if roll < 0.4 {
                let (sites, m) = random_step(&space, &mut rng);
                dense.apply_sites(&m, &sites).unwrap();
                comp.apply_sites(&m, &sites).unwrap();
            } else if roll < 0.5 && dense.total_weight() > 1e-6 {
                // A measurement, drawn from the same seed on both: Haar
                // gates rarely leave a rank-1 cut, measurement always does.
                let site = (rng.next_f64() * space.sites() as f64) as usize % space.sites();
                let draw = rng.next_u64();
                let a = comp.measure_site(site, &mut Prng::new(draw)).unwrap();
                let b = dense.measure_site(site, &mut Prng::new(draw)).unwrap();
                assert_eq!(a, b, "seed {seed}: measured site {site}");
            } else {
                let t = random_transition(&space, &mut rng);
                space = t.target(&space).unwrap();
                lost_dense += dense.transition(&t, true).unwrap();
                lost_comp += comp.transition(&t, true).unwrap();
            }
            assert_eq!(comp.space(), space);
            let dev = max_amplitude_deviation(&comp, &dense);
            assert!(dev < 1e-9, "seed {seed}: deviation {dev} on {space}");
        }
        assert_close(lost_comp, lost_dense, 1e-10);
        // Measurement renormalizes, so the books balance only up to it.
        assert!(comp.total_abs_sqr() <= 1.0 + 1e-9);
        separations += comp
            .events()
            .iter()
            .filter(|e| e.kind == "separate")
            .count();
        merges += comp.events().iter().filter(|e| e.kind == "merge").count();
    }
    // The partition really moved both ways along these paths.
    assert!(merges > 50, "{merges} merges");
    assert!(separations > 20, "{separations} separations");
}

// ── the space over time ──────────────────────────────────────────────

#[test]
fn grow_is_a_new_region_and_touches_no_amplitude() {
    let mut comp = composite(&[3, 3]);
    let f3 = fourier::<C64>(3).unwrap();
    comp.apply_sites(&f3, &[0]).unwrap();
    comp.apply_sites(&cshift::<C64>(3, 3), &[0, 1]).unwrap();
    assert_eq!(comp.region_count(), 1);
    let before = comp.events().len();
    for k in 0..20 {
        comp.transition(
            &Transition::Grow {
                at: 1,
                radix: 2 + k % 4,
            },
            false,
        )
        .unwrap();
    }
    // 22 sites, one entangled pair: 21 regions, no merge, no conversion.
    assert_eq!(comp.space().sites(), 22);
    assert_eq!(comp.region_count(), 21);
    assert_eq!(comp.events().len(), before);
    assert_eq!(comp.largest_region_dim(), 9);
    // The pair moved to the ends of the register; the growth sits between.
    let last = comp.space().sites() - 1;
    let space = comp.space();
    for d in 0..3 {
        let mut digits = vec![0; 22];
        digits[0] = d;
        digits[last] = d;
        let i = space.encode(&digits).unwrap();
        assert!((comp.amplitude(i) - c(1.0 / 3f64.sqrt())).norm() < 1e-12);
    }
    assert_close(comp.total_abs_sqr(), 1.0, 1e-12);
}

#[test]
fn an_uncomputed_workspace_separates_before_its_release() {
    // The space module's workspace example, on the composite.
    let mut circuit = SiteCircuit::<C64>::new(Space::new(&[3, 2]).unwrap());
    circuit
        .fourier(0)
        .unwrap()
        .grow(2, 3)
        .unwrap()
        .cshift(0, 2)
        .unwrap()
        .cshift(2, 1)
        .unwrap()
        .cshift(0, 2)
        .unwrap()
        .cshift(0, 2)
        .unwrap();
    let mut comp = composite(&[3, 2]);
    circuit.run(&mut comp).unwrap();
    // Uncomputed: the workspace is its own region in |0⟩ again.
    let layout = comp.layout();
    assert!(
        layout
            .iter()
            .any(|(sites, _, dim, nnz)| sites == &[2] && *dim == 3 && *nnz == 1),
        "{layout:?}"
    );
    assert!(comp
        .events()
        .iter()
        .any(|e| e.kind == "separate" && e.sites == [2]));
    // So the release moves nothing and loses nothing.
    let dropped = comp
        .transition(&Transition::Release { site: 2 }, false)
        .unwrap();
    assert_eq!(dropped, 0.0);
    let mut reference = circuit.clone();
    reference.release(2).unwrap();
    let dense = sim().run_space("dense", &reference).unwrap();
    assert!(max_amplitude_deviation(&comp, dense.as_ref()) < 1e-12);
}

#[test]
fn a_reused_workspace_stays_small_only_because_regions_re_separate() {
    // Eight superposed qutrits served in turn by one workspace qutrit:
    // copy in, phase, uncompute (the phase kicks back onto the qutrit).
    // With re-separation the workspace leaves each qutrit's region as soon
    // as it is back in |0⟩; without it, the regions accrete.
    let n = 8;
    let z = quantsim::space::clock::<C64>(3).unwrap();
    let mut peaks = Vec::new();
    for split in [true, false] {
        let mut comp = composite(&[3; 8]);
        comp.set_auto_split(split);
        for k in 0..n {
            comp.apply_sites(&fourier::<C64>(3).unwrap(), &[k]).unwrap();
        }
        comp.transition(&Transition::Grow { at: n, radix: 3 }, false)
            .unwrap();
        for k in 0..n {
            comp.apply_sites(&cshift::<C64>(3, 3), &[k, n]).unwrap();
            comp.apply_sites(&z, &[n]).unwrap();
            comp.apply_sites(&cshift::<C64>(3, 3), &[k, n]).unwrap();
            comp.apply_sites(&cshift::<C64>(3, 3), &[k, n]).unwrap();
        }
        let dropped = comp
            .transition(&Transition::Release { site: n }, false)
            .unwrap();
        assert_eq!(dropped, 0.0);
        // Either way the state is ⊗_k (Σ_d ω^d |d⟩)/√3.
        let w = C64::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
        let space = comp.space();
        for i in [0u64, 1, 5, 100, 6560] {
            let digits = space.decode(i);
            let phase = w.powu(digits.iter().sum::<usize>() as u32);
            let want = phase / 3f64.powi(4);
            assert!((comp.amplitude(i) - want).norm() < 1e-12, "index {i}");
        }
        peaks.push(comp.peak().0);
    }
    assert_eq!(peaks, vec![9, 3u128.pow(9)]);
}

#[test]
fn releases_and_demotes_are_refused_or_dropped_on_the_global_weight() {
    // Site 1 entangled with site 0; site 2 a separate superposition, so
    // the region-local weight is not the global one when site 2 is lossy.
    let mut comp = composite(&[2, 3, 3]);
    let h = SiteMatrix::from_vec(2, {
        let s = 1.0 / 2f64.sqrt();
        vec![c(s), c(s), c(s), c(-s)]
    })
    .unwrap();
    comp.apply_sites(&h, &[0]).unwrap();
    comp.apply_sites(&cshift::<C64>(2, 3), &[0, 1]).unwrap();
    comp.apply_sites(&fourier::<C64>(3).unwrap(), &[2]).unwrap();
    let mut dense = RadixDenseState::<C64>::new(comp.space()).unwrap();
    dense
        .load(&{
            let mut e = Vec::new();
            comp.for_each_nonzero(&mut |i, a| e.push((i, a)));
            e
        })
        .unwrap();

    // Refused, with the weight named and the state untouched.
    let err = comp
        .transition(&Transition::Release { site: 1 }, false)
        .unwrap_err();
    match err {
        Error::WouldDiscard { weight, .. } => assert_close(weight, 0.5, 1e-12),
        other => panic!("expected WouldDiscard, got {other}"),
    }
    assert!(max_amplitude_deviation(&comp, &dense) < 1e-15);

    // Demote site 2 to 2 levels, opted in: a third of the weight goes.
    let t = Transition::Demote { site: 2, radix: 2 };
    let lost = comp.transition(&t, true).unwrap();
    assert_close(lost, dense.transition(&t, true).unwrap(), 1e-12);
    assert_close(lost, 1.0 / 3.0, 1e-12);
    // Now release site 1: the region loses half of *its* weight, and the
    // global figure is that times the demoted region's 2/3.
    let t = Transition::Release { site: 1 };
    let lost = comp.transition(&t, true).unwrap();
    assert_close(lost, dense.transition(&t, true).unwrap(), 1e-12);
    assert_close(lost, 0.5 * 2.0 / 3.0, 1e-12);
    assert!(max_amplitude_deviation(&comp, &dense) < 1e-12);
    assert_close(comp.total_abs_sqr(), 1.0 / 3.0, 1e-12);
}

#[test]
fn a_merge_across_regions_then_a_split_re_separates() {
    let mut comp = composite(&[2, 3, 4]);
    let mut rng = Prng::new(5);
    for s in 0..3 {
        let d = comp.space().radix(s);
        let m = SiteMatrix::from_vec(d, common::random_unitary(d, &mut rng)).unwrap();
        comp.apply_sites(&m, &[s]).unwrap();
    }
    assert_eq!(comp.region_count(), 3);
    comp.transition(&Transition::Merge { site: 1 }, false)
        .unwrap();
    assert_eq!(comp.space().radices(), &[2, 12]);
    assert_eq!(comp.region_count(), 2);
    // A product state split back: the rank-1 test finds the two factors.
    comp.transition(&Transition::Split { site: 1, low: 3 }, false)
        .unwrap();
    assert_eq!(comp.space().radices(), &[2, 3, 4]);
    assert_eq!(comp.region_count(), 3);
    // Split the other way: a radix-12 digit read as (4, 3) is entangled.
    comp.transition(&Transition::Merge { site: 1 }, false)
        .unwrap();
    comp.transition(&Transition::Split { site: 1, low: 4 }, false)
        .unwrap();
    assert_eq!(comp.space().radices(), &[2, 4, 3]);
    assert_eq!(comp.region_count(), 2);
}

// ── measurement ──────────────────────────────────────────────────────

#[test]
fn measurement_splits_the_site_out_and_matches_dense_seed_for_seed() {
    let space = Space::new(&[3, 2, 4, 3]).unwrap();
    let mut rng = Prng::new(77);
    let steps: Vec<_> = (0..14).map(|_| random_step(&space, &mut rng)).collect();
    for seed in 0..8u64 {
        let mut comp = CompositeState::<C64>::new(space.clone()).unwrap();
        let mut dense = RadixDenseState::<C64>::new(space.clone()).unwrap();
        for (sites, m) in &steps {
            comp.apply_sites(m, sites).unwrap();
            dense.apply_sites(m, sites).unwrap();
        }
        let (mut rc, mut rd) = (Prng::new(seed), Prng::new(seed));
        for site in [2, 0, 3] {
            let a = comp.measure_site(site, &mut rc).unwrap();
            let b = dense.measure_site(site, &mut rd).unwrap();
            assert_eq!(a, b, "seed {seed}, site {site}");
            // The measured site is a singleton region holding |a⟩.
            let layout = comp.layout();
            assert!(layout
                .iter()
                .any(|(s, _, _, nnz)| s == &[site] && *nnz == 1));
        }
        assert!(comp.measure(1, &mut rc).unwrap() == dense.measure(1, &mut rd).unwrap());
        assert!(max_amplitude_deviation(&comp, &dense) < 1e-10);
        assert_eq!(comp.region_count(), 4);
    }
}

// ── several backends, one space ──────────────────────────────────────

#[test]
fn an_mps_region_and_radix_regions_share_one_register() {
    // Sites 0..24 are qubits held as an MPS; site 24 a qutrit, site 25 a
    // ququart, both radix stores.
    let mut radices = vec![2; 24];
    radices.extend([3, 4]);
    let space = Space::new(&radices).unwrap();
    let mut mps = MpsState::<C64>::new(24).unwrap();
    let mut ghz = Circuit::new(24);
    ghz.h(0);
    for q in 0..23 {
        ghz.cx(q, q + 1);
    }
    let bound = ghz.bind(sim().registry()).unwrap();
    for g in bound.gates() {
        match &g.kernel {
            GateKernel::Matrix(m) => mps.apply(m, &g.qubits).unwrap(),
            GateKernel::Diagonal(d) => mps.apply_diagonal(d, &g.qubits).unwrap(),
        }
    }
    let regions: Vec<(Vec<usize>, Box<dyn Backend<C64>>)> = vec![
        ((0..24).collect(), Box::new(mps)),
        (
            vec![24],
            Box::new(RadixSparseState::<C64>::new(Space::new(&[3]).unwrap()).unwrap()),
        ),
        (
            vec![25],
            Box::new(RadixDenseState::<C64>::new(Space::new(&[4]).unwrap()).unwrap()),
        ),
    ];
    let mut comp = CompositeState::with_regions(space.clone(), regions).unwrap();

    // Inside a region: native, nothing merges, the MPS stays an MPS.
    let x = SiteMatrix::from_vec(2, vec![c(0.0), c(1.0), c(1.0), c(0.0)]).unwrap();
    comp.apply_sites(&x, &[5]).unwrap();
    comp.apply_sites(&fourier::<C64>(4).unwrap(), &[25])
        .unwrap();
    assert_eq!(comp.layout()[0].1, "mps");
    assert!(comp.events().is_empty());

    // A crossing: qubit 23 controls a shift of the qutrit. The merged
    // region has dimension 3·2^24 but support 2, so it is elected sparse.
    let ctrl_shift = {
        // control = site 23 (low digit), target = site 24.
        let mut m = vec![c(0.0); 36];
        for t in 0..3 {
            m[(2 * t) * 6 + 2 * t] = c(1.0); // control 0: identity
            m[(1 + 2 * ((t + 1) % 3)) * 6 + 1 + 2 * t] = c(1.0); // control 1: shift
        }
        SiteMatrix::from_vec(6, m).unwrap()
    };
    comp.apply_sites(&ctrl_shift, &[23, 24]).unwrap();
    let merge = comp
        .events()
        .iter()
        .find(|e| e.kind == "merge")
        .expect("the crossing is ledgered");
    assert_eq!(merge.from, "mps+radix-sparse");
    assert_eq!(merge.to, "radix-sparse");
    let layout = comp.layout();
    assert_eq!(layout.len(), 2, "{layout:?}");
    assert_eq!(layout[0].2, 3 << 24);
    assert_eq!(layout[0].3, 2);
    assert_eq!(comp.largest_region_dim(), 3 << 24);

    // The state: GHZ with qubit 5 flipped, the qutrit at the parity of
    // qubit 23, the ququart uniform.
    let mut want = 0;
    for branch in 0..2usize {
        for q in 0..4 {
            let mut digits = vec![branch; 24];
            digits[5] ^= 1;
            digits.extend([branch, q]);
            let i = space.encode(&digits).unwrap();
            assert!((comp.amplitude(i) - c(0.5 / 2f64.sqrt())).norm() < 1e-12);
            want += 1;
        }
    }
    assert_eq!(comp.nonzero_count(), want);
    assert_close(comp.total_abs_sqr(), 1.0, 1e-12);
}

// ── a virtual space no flat store can hold ───────────────────────────

#[test]
fn a_forty_qutrit_stream_lives_in_kilobytes() {
    // Grow a qutrit chain one site at a time: each new site is entangled
    // with its predecessor (Fourier, then a controlled shift), and the
    // predecessor is then measured. The space path ends at 3^40 ≈ 1.2e19
    // — inside the u64 index, far past any dense allocation — while the
    // largest region the composite ever holds is one pair.
    let mut comp = composite(&[3]);
    let f3 = fourier::<C64>(3).unwrap();
    let cs = cshift::<C64>(3, 3);
    let mut rng = Prng::new(2026);
    let mut outcomes = Vec::new();
    comp.apply_sites(&f3, &[0]).unwrap();
    for k in 1..40 {
        comp.transition(&Transition::Grow { at: k, radix: 3 }, false)
            .unwrap();
        comp.apply_sites(&cs, &[k - 1, k]).unwrap();
        comp.apply_sites(&f3, &[k]).unwrap();
        outcomes.push(comp.measure_site(k - 1, &mut rng).unwrap());
    }
    let space = comp.space();
    assert_eq!(space.sites(), 40);
    assert_eq!(space.dim(), 3u128.pow(40));
    let (peak_dim, peak_stored) = comp.peak();
    assert_eq!(peak_dim, 9);
    assert!(peak_stored <= 2 * 40 + 9, "{peak_stored}");
    assert!(comp.memory_bytes() < 64 * 1024, "{} B", comp.memory_bytes());
    assert_eq!(comp.region_count(), 40);
    // Every measured site holds its outcome; the last is uniform.
    let mut digits: Vec<usize> = outcomes.clone();
    digits.push(0);
    for d in 0..3 {
        digits[39] = d;
        let i = space.encode(&digits).unwrap();
        assert!((comp.amplitude(i).norm() - 1.0 / 3f64.sqrt()).abs() < 1e-12);
    }
    assert_close(comp.total_abs_sqr(), 1.0, 1e-10);
    // The outcomes are not degenerate (the stream really branched).
    assert!(outcomes.contains(&1) && outcomes.contains(&2));
}
