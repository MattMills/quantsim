//! Differential tests for two shared primitives that changed underneath
//! everything else: `PhaseFieldState::for_each_nonzero`, which was
//! rewritten from an `O(2^n)` sweep to an `O(support)` enumeration, and
//! `Backend::apply_permutation`, which is a new trait method with one
//! override.
//!
//! Neither is tested by asserting the new code agrees with itself. Both
//! are tested against the thing they replaced.

use quantsim::backend::{Backend, PhaseFieldState};
use quantsim::prelude::*;
use quantsim::shor;

fn sim_with_phase_field() -> Simulator {
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("phase-field", |n| Ok(Box::new(PhaseFieldState::new(n)?)))
        .unwrap();
    sim
}

/// Exactly the old implementation: sweep every index and filter.
fn brute_force_support(state: &dyn Backend<C64>) -> Vec<(u64, C64)> {
    let n = state.num_qubits();
    (0..(1u64 << n))
        .map(|i| (i, state.amplitude(i)))
        .filter(|(_, a)| a.abs_sqr() > 0.0)
        .collect()
}

fn walked_support(state: &dyn Backend<C64>) -> Vec<(u64, C64)> {
    let mut out = Vec::new();
    state.for_each_nonzero(&mut |i, a| out.push((i, a)));
    out
}

/// Drive a phase-field register with gates that keep it in the field
/// class: Hadamards free bits, diagonals write the phase polynomial.
fn evolve_field(sim: &Simulator, state: &mut dyn Backend<C64>, seed: u64) {
    let n = state.num_qubits();
    let mut rng = Prng::new(seed);
    for q in 0..n {
        if rng.next_u64() % 3 != 0 {
            sim.apply(state, "h", &[], &[q]).unwrap();
        }
    }
    for _ in 0..2 * n {
        let q = (rng.next_u64() % n as u64) as usize;
        let p = (rng.next_u64() % n as u64) as usize;
        match rng.next_u64() % 4 {
            0 => sim.apply(state, "s", &[], &[q]).unwrap(),
            1 => sim.apply(state, "t", &[], &[q]).unwrap(),
            2 => sim.apply(state, "z", &[], &[q]).unwrap(),
            _ => {
                if q != p {
                    sim.apply(state, "cz", &[], &[q, p]).unwrap();
                }
            }
        }
    }
}

#[test]
fn the_phase_field_support_walk_matches_the_sweep_it_replaced() {
    let sim = sim_with_phase_field();
    let mut checked = 0;
    for n in 1..=10usize {
        for seed in 0..12u64 {
            let mut state = sim.backends().create("phase-field", n).unwrap();
            evolve_field(&sim, state.as_mut(), seed * 31 + n as u64);

            let mut want = brute_force_support(state.as_ref());
            let mut got = walked_support(state.as_ref());
            want.sort_by_key(|e| e.0);
            got.sort_by_key(|e| e.0);

            assert_eq!(
                got.len(),
                want.len(),
                "n={n} seed={seed}: walked {} entries, sweep found {}",
                got.len(),
                want.len()
            );
            for (a, b) in got.iter().zip(want.iter()) {
                assert_eq!(a.0, b.0, "n={n} seed={seed}: index mismatch");
                assert!(
                    (a.1.re - b.1.re).abs() < 1e-15 && (a.1.im - b.1.im).abs() < 1e-15,
                    "n={n} seed={seed} index {}: {:?} vs {:?}",
                    a.0,
                    a.1,
                    b.1
                );
            }
            // And the closed-form count is the count actually walked.
            assert_eq!(state.nonzero_count(), got.len(), "n={n} seed={seed}");
            checked += 1;
        }
    }
    assert!(checked >= 100, "only {checked} states compared");
}

#[test]
fn the_phase_field_walk_is_still_in_ascending_index_order() {
    // The sweep it replaced emitted indices ascending; anything relying
    // on that ordering must not have been broken by the rewrite.
    let sim = sim_with_phase_field();
    for n in 1..=10usize {
        for seed in 0..6u64 {
            let mut state = sim.backends().create("phase-field", n).unwrap();
            evolve_field(&sim, state.as_mut(), seed + 7);
            let got = walked_support(state.as_ref());
            for pair in got.windows(2) {
                assert!(pair[0].0 < pair[1].0, "n={n} seed={seed}: not ascending");
            }
        }
    }
}

#[test]
fn the_phase_field_walk_is_cheap_at_a_width_the_sweep_could_not_reach() {
    // The rewrite's whole point: a pinned state on a wide register. The
    // old sweep was 2^n here; if the new one were too, this would not
    // return. Two amplitudes at 40 qubits.
    let sim = sim_with_phase_field();
    let mut state = sim.backends().create("phase-field", 40).unwrap();
    sim.apply(state.as_mut(), "x", &[], &[3]).unwrap();
    sim.apply(state.as_mut(), "h", &[], &[17]).unwrap();
    assert_eq!(state.nonzero_count(), 2);
    let got = walked_support(state.as_ref());
    assert_eq!(got.len(), 2);
    for (i, _) in &got {
        assert!(i >> 3 & 1 == 1, "the pinned bit was lost");
    }
    assert_ne!(got[0].0, got[1].0);
}

/// The permutation used to cross-check the trait method: a modular
/// multiplication on the low bits, which is what `shor` actually applies.
fn perm(index: u64, work: &[usize], multiplier: u64, modulus: u64) -> u64 {
    let y = shor::gather(index, work);
    if y >= modulus {
        return index;
    }
    shor::scatter(index, work, y * multiplier % modulus)
}

#[test]
fn the_sparse_permutation_override_agrees_with_the_trait_default() {
    // `SparseState` overrides `apply_permutation`; every other
    // representation inherits the default. They must be the same map.
    let sim: Simulator = Simulator::new();
    let n = 8usize;
    let work: Vec<usize> = (0..6).collect();
    let modulus = 35u64;
    for multiplier in [2u64, 3, 4, 6, 8, 9, 11, 12, 13, 16, 17] {
        if quantsim::padic::gcd(multiplier, modulus) != 1 {
            continue;
        }
        let mut states: Vec<(&str, Box<dyn Backend<C64>>)> = Vec::new();
        for name in ["sparse", "dense", "adaptive", "mosaic"] {
            let mut s = sim.backends().create(name, n).unwrap();
            // A spread-out start state, so the map is exercised broadly.
            let mut c: Circuit = Circuit::new(n);
            c.x(0).h(6).h(7).h(1);
            c.bind(sim.registry()).unwrap().run(s.as_mut()).unwrap();
            states.push((name, s));
        }
        for (_, s) in states.iter_mut() {
            s.apply_permutation("mul", &|i| perm(i, &work, multiplier, modulus))
                .unwrap();
        }
        // Every representation must now hold the same state.
        let reference: Vec<(u64, C64)> = {
            let mut v = walked_support(states[0].1.as_ref());
            v.sort_by_key(|e| e.0);
            v
        };
        for (name, s) in states.iter().skip(1) {
            let mut v = walked_support(s.as_ref());
            v.sort_by_key(|e| e.0);
            assert_eq!(
                v.len(),
                reference.len(),
                "{name} disagreed on support size for multiplier {multiplier}"
            );
            for (a, b) in v.iter().zip(reference.iter()) {
                assert_eq!(a.0, b.0, "{name}: index mismatch");
                assert!(
                    (a.1.re - b.1.re).abs() < 1e-12 && (a.1.im - b.1.im).abs() < 1e-12,
                    "{name}: amplitude mismatch at {}",
                    a.0
                );
            }
        }
    }
}

#[test]
fn every_representation_refuses_a_non_injective_permutation() {
    let sim: Simulator = Simulator::new();
    for name in ["sparse", "dense", "adaptive", "mosaic", "factored"] {
        let mut s = sim.backends().create(name, 4).unwrap();
        let mut c: Circuit = Circuit::new(4);
        c.h(0).h(1);
        c.bind(sim.registry()).unwrap().run(s.as_mut()).unwrap();
        let err = s
            .apply_permutation("collapse", &|_| 0)
            .expect_err("{name} accepted a collapsing map")
            .to_string();
        assert!(
            err.contains("not injective"),
            "{name} gave the wrong error: {err}"
        );
    }
}

#[test]
fn a_permutation_out_of_range_is_refused_rather_than_wrapped() {
    let sim: Simulator = Simulator::new();
    for name in ["sparse", "dense", "adaptive"] {
        let mut s = sim.backends().create(name, 4).unwrap();
        let err = s
            .apply_permutation("escape", &|i| i + 64)
            .expect_err("{name} accepted an out-of-range map")
            .to_string();
        assert!(
            err.contains("outside") || err.contains("range"),
            "{name} gave the wrong error: {err}"
        );
    }
}
