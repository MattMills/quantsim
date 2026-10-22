//! Space transitions, measured: a register whose [`Space`] is a path
//! rather than a constant. Every transition's index map against its
//! digit-level specification; merge and split as free relabelings that
//! reproduce the two-site gate exactly; grow and promote certified
//! against an independent [`CompoundRegister`] built on the final space
//! (promote as `U ⊕ I`); a workspace grown, used, uncomputed and released
//! exactly; release and demote refused with the state untouched when they
//! would remove amplitude, or dropping it with the weight **returned** when
//! the caller opts in; dense and sparse agreeing along random paths over ℂ
//! and ℍ with `weight + discarded = 1`; and every refusal by name — a
//! qubit backend asked to change shape, an illegal step, a circuit that
//! would discard without a way to report it.

mod common;

use std::collections::HashSet;

use quantsim::backend::{max_amplitude_deviation, DenseState};
use quantsim::mixed::CompoundRegister;
use quantsim::prelude::*;

type Steps = Vec<(Vec<usize>, Vec<C64>)>;

/// One random unitary on one or two random sites of `space`.
fn random_step(space: &Space, rng: &mut Prng) -> (Vec<usize>, Vec<C64>) {
    let n = space.sites();
    let a = (rng.next_f64() * n as f64) as usize % n;
    let sites = if n == 1 || rng.next_f64() < 0.5 {
        vec![a]
    } else {
        let mut b = (rng.next_f64() * n as f64) as usize % n;
        if b == a {
            b = (a + 1) % n;
        }
        vec![a, b]
    };
    let d = space.sub_dim(&sites);
    (sites, common::random_unitary(d, rng))
}

fn random_steps(space: &Space, count: usize, rng: &mut Prng) -> Steps {
    (0..count).map(|_| random_step(space, rng)).collect()
}

fn apply_steps(state: &mut dyn Backend<C64>, steps: &Steps) {
    let space = state.space();
    for (sites, m) in steps {
        let sm = SiteMatrix::from_vec(space.sub_dim(sites), m.clone()).unwrap();
        state.apply_sites(&sm, sites).unwrap();
    }
}

/// Both native stores over `space`, in `|0…0⟩`.
fn both(space: &Space) -> [Box<dyn Backend<C64>>; 2] {
    [
        Box::new(RadixDenseState::<C64>::new(space.clone()).unwrap()),
        Box::new(RadixSparseState::<C64>::new(space.clone()).unwrap()),
    ]
}

/// Largest amplitude gap between a backend and the oracle, which must hold
/// the same radices.
fn oracle_gap(state: &dyn Backend<C64>, oracle: &CompoundRegister) -> f64 {
    let space = state.space();
    assert_eq!(space.radices(), oracle.dims());
    (0..space.dim() as u64)
        .map(|i| (state.amplitude(i) - oracle.amplitude(&space.decode(i)).unwrap()).norm())
        .fold(0.0, f64::max)
}

/// `m` over `sites` of `old` as an operator on the same sites of `new`,
/// whose radices there are at least as wide: `m` on the old levels,
/// identity on the rest.
fn embed(m: &[C64], sites: &[usize], old: &Space, new: &Space) -> Vec<C64> {
    let r_old: Vec<usize> = sites.iter().map(|&s| old.radix(s)).collect();
    let r_new: Vec<usize> = sites.iter().map(|&s| new.radix(s)).collect();
    let (d_old, d_new) = (old.sub_dim(sites), new.sub_dim(sites));
    let old_sub = |mut sub: usize| {
        let mut out = 0;
        let mut stride = 1;
        for (k, &r) in r_new.iter().enumerate() {
            let digit = sub % r;
            sub /= r;
            if digit >= r_old[k] {
                return None;
            }
            out += digit * stride;
            stride *= r_old[k];
        }
        Some(out)
    };
    let mut out = vec![C64::new(0.0, 0.0); d_new * d_new];
    for row in 0..d_new {
        for col in 0..d_new {
            out[row * d_new + col] = match (old_sub(row), old_sub(col)) {
                (Some(r), Some(c)) => m[r * d_old + c],
                (None, None) if row == col => C64::new(1.0, 0.0),
                _ => C64::new(0.0, 0.0),
            };
        }
    }
    out
}

// ── the index maps ───────────────────────────────────────────────────

#[test]
fn every_transition_moves_digits_as_specified() {
    let from = Space::new(&[2, 3, 4, 3]).unwrap();
    let cases = [
        Transition::Merge { site: 1 },
        Transition::Split { site: 2, low: 2 },
        Transition::Grow { at: 0, radix: 3 },
        Transition::Grow { at: 2, radix: 5 },
        Transition::Grow { at: 4, radix: 2 },
        Transition::Release { site: 1 },
        Transition::Promote { site: 1, radix: 5 },
        Transition::Demote { site: 2, radix: 3 },
    ];
    for t in cases {
        let to = t.target(&from).unwrap();
        let mut seen = HashSet::new();
        for i in 0..from.dim() as u64 {
            let mut d = from.decode(i);
            let want = match t {
                Transition::Merge { site } => {
                    let fused = d[site] + from.radix(site) * d[site + 1];
                    d.splice(site..site + 2, [fused]);
                    Some(d)
                }
                Transition::Split { site, low } => {
                    let v = d[site];
                    d.splice(site..site + 1, [v % low, v / low]);
                    Some(d)
                }
                Transition::Grow { at, .. } => {
                    d.insert(at, 0);
                    Some(d)
                }
                Transition::Release { site } => (d[site] == 0).then(|| {
                    d.remove(site);
                    d
                }),
                Transition::Promote { .. } => Some(d),
                Transition::Demote { site, radix } => (d[site] < radix).then_some(d),
            };
            let got = t.map_index(&from, i);
            match want {
                None => assert_eq!(got, None, "{t}: index {i} should be removed"),
                Some(digits) => {
                    let j = got.unwrap_or_else(|| panic!("{t}: index {i} was dropped"));
                    assert!((j as u128) < to.dim());
                    assert_eq!(to.decode(j), digits, "{t}: index {i}");
                    assert!(seen.insert(j), "{t}: two indices land on {j}");
                }
            }
        }
        // Refactorings and embeddings keep every index; only release and
        // demote can lose any.
        if !t.can_discard() {
            assert_eq!(seen.len() as u128, from.dim(), "{t}");
        }
        if t.is_relabeling() {
            assert_eq!(to.dim(), from.dim());
            for i in 0..from.dim() as u64 {
                assert_eq!(t.map_index(&from, i), Some(i), "{t} moved index {i}");
            }
        }
    }
}

// ── refactorings: merge and split ────────────────────────────────────

#[test]
fn merging_two_sites_is_free_and_reproduces_the_two_site_gate() {
    let space = Space::new(&[2, 3, 4]).unwrap();
    let mut rng = Prng::new(11);
    let prep = random_steps(&space, 12, &mut rng);
    let u = SiteMatrix::from_vec(12, common::random_unitary(12, &mut rng)).unwrap();
    for (mut direct, mut merged) in both(&space).into_iter().zip(both(&space)) {
        apply_steps(direct.as_mut(), &prep);
        apply_steps(merged.as_mut(), &prep);
        direct.apply_sites(&u, &[1, 2]).unwrap();

        let discarded = merged
            .transition(&Transition::Merge { site: 1 }, false)
            .unwrap();
        assert_eq!(discarded, 0.0);
        assert_eq!(merged.space().radices(), &[2, 12]);
        assert_eq!(merged.num_qubits(), 2);
        // The radix-12 site *is* sites 1 and 2: one-site gate, same result.
        merged.apply_sites(&u, &[1]).unwrap();
        assert!(max_amplitude_deviation(merged.as_ref(), direct.as_ref()) < 1e-12);

        merged
            .transition(&Transition::Split { site: 1, low: 3 }, false)
            .unwrap();
        assert_eq!(merged.space(), space);
        assert!(max_amplitude_deviation(merged.as_ref(), direct.as_ref()) < 1e-12);
        common::assert_close(merged.total_weight(), 1.0, 1e-10);
    }
}

#[test]
fn a_qubit_pair_becomes_a_ququart_and_back_inside_one_circuit() {
    // Bell pair (|00⟩ + |11⟩)/√2 → merged into one radix-4 digit {0, 3} →
    // shift by one → {1, 0} → split back into qubits: |+⟩ on qubit 0,
    // |0⟩ on qubit 1 → H on qubit 0 → |00⟩.
    let sim: Simulator = Simulator::new();
    let mut bell = Circuit::new(2);
    bell.h(0).cx(0, 1);
    let mut h0 = Circuit::new(2);
    h0.h(0);
    let (bell, h0) = (
        bell.bind(sim.registry()).unwrap(),
        h0.bind(sim.registry()).unwrap(),
    );
    let mut c = SiteCircuit::<C64>::new(Space::qubits(2));
    c.place(&bell, &[0, 1]).unwrap();
    c.merge(0).unwrap().shift(0).unwrap().split(0, 2).unwrap();
    c.place(&h0, &[0, 1]).unwrap();
    assert_eq!(c.space_path().len(), 3);
    assert_eq!(c.space_path()[1], Space::new(&[4]).unwrap());
    assert!(c.final_space().is_qubits());

    for backend in ["dense", "sparse", "composite"] {
        // A circuit that changes shape runs on the native store, even when
        // it starts and ends on qubits.
        let state = sim.run_space(backend, &c).unwrap();
        assert!(state.name().starts_with("radix-") || state.name() == "composite");
        assert_eq!(state.space(), Space::qubits(2));
        common::assert_close(state.probability(0), 1.0, 1e-12);
    }
    // A backend without a native store cannot follow the path.
    assert!(matches!(
        sim.run_space("mps", &c),
        Err(Error::UnsupportedForSpace { .. })
    ));
}

// ── embeddings: grow and promote ─────────────────────────────────────

#[test]
fn growing_sites_matches_the_compound_register_on_the_final_space() {
    let start = Space::new(&[3, 2]).unwrap();
    for seed in 0..3u64 {
        let mut rng = Prng::new(200 + seed);
        let phase1 = random_steps(&start, 10, &mut rng);
        let mid = Transition::Grow { at: 1, radix: 4 }.target(&start).unwrap();
        let phase2 = random_steps(&mid, 10, &mut rng);
        let end = Transition::Grow { at: 3, radix: 3 }.target(&mid).unwrap();
        let phase3 = random_steps(&end, 10, &mut rng);
        assert_eq!(end.radices(), &[3, 4, 2, 3]);

        // The oracle lives on the final space from the start; the sites
        // that will be grown simply sit in |0⟩ until they are used.
        let mut oracle = CompoundRegister::new(end.radices()).unwrap();
        for (sites, m) in &phase1 {
            let moved: Vec<usize> = sites.iter().map(|&s| if s == 1 { 2 } else { s }).collect();
            oracle.apply_k(&moved, m).unwrap();
        }
        for (sites, m) in phase2.iter().chain(&phase3) {
            oracle.apply_k(sites, m).unwrap();
        }

        for mut state in both(&start) {
            apply_steps(state.as_mut(), &phase1);
            let stored = state.nonzero_count();
            let d1 = state
                .transition(&Transition::Grow { at: 1, radix: 4 }, false)
                .unwrap();
            // An embedding moves amplitudes; it creates none.
            assert_eq!(state.nonzero_count(), stored);
            apply_steps(state.as_mut(), &phase2);
            let d2 = state
                .transition(&Transition::Grow { at: 3, radix: 3 }, false)
                .unwrap();
            apply_steps(state.as_mut(), &phase3);
            assert_eq!(d1 + d2, 0.0);
            let gap = oracle_gap(state.as_ref(), &oracle);
            assert!(gap < 1e-10, "seed {seed} {}: gap {gap}", state.name());
        }
    }
}

#[test]
fn promoting_a_site_is_the_embedding_u_plus_identity() {
    let start = Space::new(&[2, 3]).unwrap();
    let wide = Transition::Promote { site: 0, radix: 4 }
        .target(&start)
        .unwrap();
    let mut rng = Prng::new(31);
    let before = random_steps(&start, 12, &mut rng);
    let after = random_steps(&wide, 12, &mut rng);
    let mut oracle = CompoundRegister::new(wide.radices()).unwrap();
    for (sites, m) in &before {
        oracle
            .apply_k(sites, &embed(m, sites, &start, &wide))
            .unwrap();
    }
    for (sites, m) in &after {
        oracle.apply_k(sites, m).unwrap();
    }
    // Weight the later gates put in levels 2 and 3 of site 0.
    let upper: f64 = (0..wide.dim() as u64)
        .filter(|&i| wide.digit(i, 0) >= 2)
        .map(|i| oracle.probability(&wide.decode(i)).unwrap())
        .sum();
    assert!(upper > 0.05, "the test needs weight above the old levels");

    for mut state in both(&start) {
        apply_steps(state.as_mut(), &before);
        let promote = Transition::Promote { site: 0, radix: 4 };
        assert_eq!(state.transition(&promote, false).unwrap(), 0.0);
        apply_steps(state.as_mut(), &after);
        assert!(oracle_gap(state.as_ref(), &oracle) < 1e-10);

        // Demoting back would remove that weight: refused, state untouched.
        let demote = Transition::Demote { site: 0, radix: 2 };
        match state.transition(&demote, false) {
            Err(Error::WouldDiscard { weight, .. }) => common::assert_close(weight, upper, 1e-10),
            other => panic!("expected WouldDiscard, got {other:?}"),
        }
        assert_eq!(state.space(), wide);
        assert!(oracle_gap(state.as_ref(), &oracle) < 1e-10);

        // Opted in: the weight is dropped, returned, and not renormalized
        // away; what survives is exactly the oracle's lower block.
        let dropped = state.transition(&demote, true).unwrap();
        common::assert_close(dropped, upper, 1e-10);
        assert_eq!(state.space(), start);
        common::assert_close(state.total_weight(), 1.0 - upper, 1e-10);
        for i in 0..start.dim() as u64 {
            let want = oracle.amplitude(&start.decode(i)).unwrap();
            assert!((state.amplitude(i) - want).norm() < 1e-10);
        }
    }
}

#[test]
fn promote_then_demote_is_the_identity_when_the_new_levels_stay_empty() {
    let space = Space::new(&[3, 4]).unwrap();
    let mut rng = Prng::new(5);
    let prep = random_steps(&space, 10, &mut rng);
    for (mut state, mut reference) in both(&space).into_iter().zip(both(&space)) {
        apply_steps(state.as_mut(), &prep);
        apply_steps(reference.as_mut(), &prep);
        state
            .transition(&Transition::Promote { site: 1, radix: 6 }, false)
            .unwrap();
        // A gate that never leaves the old levels of site 1.
        state.apply_sites(&quantsim::space::shift(3), &[0]).unwrap();
        reference
            .apply_sites(&quantsim::space::shift(3), &[0])
            .unwrap();
        let back = state
            .transition(&Transition::Demote { site: 1, radix: 4 }, false)
            .unwrap();
        assert_eq!(back, 0.0);
        assert_eq!(state.space(), space);
        assert_eq!(
            max_amplitude_deviation(state.as_ref(), reference.as_ref()),
            0.0
        );
    }
}

// ── release: a workspace that comes and goes ─────────────────────────

/// `prep` on `[3, 2, 5]`, then a radix-3 workspace grown at the end and
/// loaded with a copy of site 0, used to drive site 1.
fn workspace_circuit(prep: &Steps) -> SiteCircuit<C64> {
    let space = Space::new(&[3, 2, 5]).unwrap();
    let mut c = SiteCircuit::<C64>::new(space.clone());
    for (sites, m) in prep {
        let sm = SiteMatrix::from_vec(space.sub_dim(sites), m.clone()).unwrap();
        c.gate("u", sm, sites).unwrap();
    }
    c.grow(3, 3)
        .unwrap()
        .cshift(0, 3)
        .unwrap()
        .cshift(3, 1)
        .unwrap();
    c
}

#[test]
fn a_workspace_grown_used_uncomputed_and_released_is_exact() {
    let space = Space::new(&[3, 2, 5]).unwrap();
    let mut rng = Prng::new(77);
    let prep = random_steps(&space, 15, &mut rng);
    // Oracle: the workspace's effect, d1 += d0 (mod 2), applied directly.
    let mut oracle = CompoundRegister::new(space.radices()).unwrap();
    for (sites, m) in &prep {
        oracle.apply_k(sites, m).unwrap();
    }
    let direct = quantsim::space::cshift::<C64>(3, 2);
    oracle.apply_k(&[0, 1], direct.data()).unwrap();

    let mut c = workspace_circuit(&prep);
    // Uncompute: two more copies make 3·d0 ≡ 0 (mod 3).
    c.cshift(0, 3)
        .unwrap()
        .cshift(0, 3)
        .unwrap()
        .release(3)
        .unwrap();
    assert_eq!(
        c.space_path(),
        &[
            space.clone(),
            Space::new(&[3, 2, 5, 3]).unwrap(),
            space.clone()
        ]
    );
    assert!(!c.allows_discard());

    let sim: Simulator = Simulator::new();
    for backend in ["dense", "sparse", "composite"] {
        let (state, discarded) = sim.run_space_accounted(backend, &c).unwrap();
        assert_eq!(discarded, 0.0);
        assert!(oracle_gap(state.as_ref(), &oracle) < 1e-10, "{backend}");
        assert!(oracle_gap(sim.run_space(backend, &c).unwrap().as_ref(), &oracle) < 1e-10);
    }
}

#[test]
fn releasing_an_entangled_site_is_refused_or_accounted() {
    let space = Space::new(&[3, 2, 5]).unwrap();
    let mut rng = Prng::new(78);
    let prep = random_steps(&space, 15, &mut rng);
    let mut oracle = CompoundRegister::new(space.radices()).unwrap();
    for (sites, m) in &prep {
        oracle.apply_k(sites, m).unwrap();
    }
    // The workspace holds a copy of d0, so releasing it removes every
    // branch with d0 ≠ 0.
    let entangled: f64 = (0..space.dim() as u64)
        .filter(|&i| space.digit(i, 0) != 0)
        .map(|i| oracle.probability(&space.decode(i)).unwrap())
        .sum();
    assert!(entangled > 0.1);

    let sim: Simulator = Simulator::new();
    let mut refusing = workspace_circuit(&prep);
    refusing.release(3).unwrap();
    let mut accounting = workspace_circuit(&prep);
    accounting
        .transition(Transition::Release { site: 3 }, true)
        .unwrap();
    assert!(accounting.allows_discard());
    for backend in ["dense", "sparse", "composite"] {
        match sim.run_space(backend, &refusing) {
            Err(Error::WouldDiscard { weight, .. }) => {
                common::assert_close(weight, entangled, 1e-10)
            }
            other => panic!("expected WouldDiscard, got {:?}", other.map(|s| s.space())),
        }
        // run_space has nowhere to put the weight, so it will not run a
        // circuit that may discard.
        assert!(matches!(
            sim.run_space(backend, &accounting),
            Err(Error::InvalidState(_))
        ));
        let (state, dropped) = sim.run_space_accounted(backend, &accounting).unwrap();
        common::assert_close(dropped, entangled, 1e-10);
        assert_eq!(state.space(), Space::new(&[3, 2, 5]).unwrap());
        common::assert_close(state.total_weight(), 1.0 - entangled, 1e-10);
        // What survives is the d0 = 0 branch, where the workspace did
        // nothing.
        for i in 0..space.dim() as u64 {
            let want = if space.digit(i, 0) == 0 {
                oracle.amplitude(&space.decode(i)).unwrap()
            } else {
                C64::new(0.0, 0.0)
            };
            assert!((state.amplitude(i) - want).norm() < 1e-10);
        }
    }
}

#[test]
fn a_refused_transition_leaves_the_state_exactly_as_it_was() {
    let space = Space::new(&[3, 4, 2]).unwrap();
    let mut rng = Prng::new(9);
    let prep = random_steps(&space, 10, &mut rng);
    let mut dense = RadixDenseState::<C64>::new(space.clone()).unwrap();
    let mut sparse = RadixSparseState::<C64>::new(space.clone()).unwrap();
    apply_steps(&mut dense, &prep);
    apply_steps(&mut sparse, &prep);
    let (dense0, sparse0) = (dense.clone(), sparse.clone());
    for t in [
        Transition::Release { site: 1 },
        Transition::Demote { site: 1, radix: 2 },
    ] {
        assert!(matches!(
            dense.transition(&t, false),
            Err(Error::WouldDiscard { .. })
        ));
        assert!(matches!(
            sparse.transition(&t, false),
            Err(Error::WouldDiscard { .. })
        ));
        assert_eq!(dense.space(), space);
        assert_eq!(dense.amplitudes(), dense0.amplitudes());
        assert_eq!(sparse.space(), space);
        assert_eq!(max_amplitude_deviation(&sparse, &sparse0), 0.0);
        assert_eq!(sparse.nonzero_count(), sparse0.nonzero_count());
    }
    // An illegal step is refused before anything is looked at.
    for t in [
        Transition::Merge { site: 2 },
        Transition::Split { site: 1, low: 3 },
        Transition::Promote { site: 0, radix: 3 },
        Transition::Grow { at: 4, radix: 2 },
    ] {
        assert!(matches!(
            dense.transition(&t, true),
            Err(Error::UnsupportedForSpace { .. })
        ));
        assert_eq!(dense.amplitudes(), dense0.amplitudes());
    }
}

// ── dense and sparse along random paths ──────────────────────────────

/// A random legal transition from `space`, keeping the register small.
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
            if to.dim() <= 720 && to.sites() <= 5 && to.radices().iter().all(|&r| r <= 12) {
                return t;
            }
        }
    }
}

fn random_path_agrees<S: Scalar>(seed: u64) {
    let mut rng = Prng::new(seed);
    let mut space = Space::new(&[2, 3, 2]).unwrap();
    let mut dense = RadixDenseState::<S>::new(space.clone()).unwrap();
    let mut sparse = RadixSparseState::<S>::new(space.clone()).unwrap();
    let (mut lost_dense, mut lost_sparse) = (0.0, 0.0);
    let mut transitions = 0;
    for _ in 0..60 {
        if rng.next_f64() < 0.5 {
            let (sites, m) = random_step(&space, &mut rng);
            let sm = SiteMatrix::<S>::try_from_c64s("u", space.sub_dim(&sites), &m).unwrap();
            dense.apply_sites(&sm, &sites).unwrap();
            sparse.apply_sites(&sm, &sites).unwrap();
        } else {
            let t = random_transition(&space, &mut rng);
            space = t.target(&space).unwrap();
            lost_dense += dense.transition(&t, true).unwrap();
            lost_sparse += sparse.transition(&t, true).unwrap();
            transitions += 1;
        }
        assert_eq!(dense.space(), space);
        assert_eq!(sparse.space(), space);
        // The dense store is always exactly the current space's size.
        assert_eq!(dense.amplitudes().len() as u128, space.dim());
    }
    assert!(transitions > 15);
    assert!(max_amplitude_deviation(&dense, &sparse) < 1e-10);
    common::assert_close(lost_dense, lost_sparse, 1e-12);
    // Unitaries keep the weight and transitions move it, so everything
    // missing is exactly what the transitions reported dropping.
    common::assert_close(dense.total_weight() + lost_dense, 1.0, 1e-10);
    common::assert_close(sparse.total_weight() + lost_sparse, 1.0, 1e-10);
}

#[test]
fn dense_and_sparse_agree_along_random_paths_of_spaces() {
    for seed in 0..6 {
        random_path_agrees::<C64>(400 + seed);
    }
}

#[test]
fn dense_and_sparse_agree_along_random_paths_over_the_quaternions() {
    for seed in 0..3 {
        random_path_agrees::<Quaternion>(500 + seed);
    }
}

// ── refusals ─────────────────────────────────────────────────────────

#[test]
fn backends_with_a_fixed_shape_refuse_transitions() {
    let mut dense = DenseState::<C64>::new(3).unwrap();
    for t in [
        Transition::Grow { at: 3, radix: 3 },
        Transition::Merge { site: 0 },
    ] {
        match dense.transition(&t, true) {
            Err(Error::UnsupportedForSpace { operation, space }) => {
                assert!(operation.contains("'dense'"), "{operation}");
                assert_eq!(space, "3 qubits");
            }
            other => panic!("expected UnsupportedForSpace, got {other:?}"),
        }
    }
    assert_eq!(dense.space(), Space::qubits(3));

    let sim: Simulator = Simulator::new();
    let mut c = SiteCircuit::<C64>::new(Space::qubits(3));
    c.grow(3, 3).unwrap();
    for backend in ["mps", "factored"] {
        assert!(matches!(
            sim.run_space(backend, &c),
            Err(Error::UnsupportedForSpace { .. })
        ));
    }
    assert!(matches!(
        sim.backends().create_native("mps", &Space::qubits(3)),
        Err(Error::UnsupportedForSpace { .. })
    ));
    assert!(matches!(
        sim.backends()
            .create_native("no-such-backend", &Space::qubits(3)),
        Err(Error::UnknownBackend(_))
    ));
    // Handed the circuit directly, a qubit backend refuses at the step.
    let mut qubit = DenseState::<C64>::new(3).unwrap();
    assert!(matches!(
        c.run(&mut qubit),
        Err(Error::UnsupportedForSpace { .. })
    ));
}

#[test]
fn circuits_validate_each_step_against_the_space_reached_so_far() {
    let mut c = SiteCircuit::<C64>::new(Space::new(&[2, 3]).unwrap());
    c.grow(2, 5)
        .unwrap()
        .merge(0)
        .unwrap()
        .promote(1, 7)
        .unwrap();
    let path: Vec<Vec<usize>> = c
        .space_path()
        .iter()
        .map(|s| s.radices().to_vec())
        .collect();
    assert_eq!(
        path,
        vec![vec![2, 3], vec![2, 3, 5], vec![6, 5], vec![6, 7]]
    );
    assert_eq!(c.space().radices(), &[2, 3]);
    assert!(c.has_transitions() && !c.allows_discard());

    // Gates see the current space: site 0 is now radix 6.
    c.fourier(0).unwrap();
    match c.ops().last() {
        Some(SiteOp::Sites { label, .. }) => assert_eq!(label, "fourier6"),
        other => panic!("unexpected op {other:?}"),
    }
    assert!(c.gate("u", SiteMatrix::identity(6), &[0, 1]).is_err());
    assert!(c.swap(0, 1).is_err());

    let ops = c.len();
    let refused = [
        Transition::Merge { site: 1 },
        Transition::Split { site: 0, low: 4 },
        Transition::Split { site: 0, low: 6 },
        Transition::Promote { site: 0, radix: 6 },
        Transition::Demote { site: 1, radix: 7 },
        Transition::Demote { site: 1, radix: 1 },
        Transition::Grow { at: 3, radix: 2 },
        Transition::Grow { at: 0, radix: 1 },
        Transition::Release { site: 2 },
    ];
    for t in refused {
        assert!(c.transition(t.clone(), false).is_err(), "{t} was accepted");
    }
    assert_eq!(c.len(), ops);
    assert_eq!(c.space_path().len(), 4);

    // A register keeps at least one site, and stays indexable.
    let mut one = SiteCircuit::<C64>::new(Space::new(&[5]).unwrap());
    assert!(one.release(0).is_err());
    let mut huge = SiteCircuit::<C64>::new(Space::new(&[1 << 31, 1 << 31]).unwrap());
    assert!(huge.grow(2, 8).is_err());

    assert_eq!(
        Transition::Merge { site: 0 }.to_string(),
        "merge sites 0 and 1"
    );
    assert_eq!(
        Transition::Demote { site: 2, radix: 3 }.to_string(),
        "demote site 2 to radix 3"
    );
}
