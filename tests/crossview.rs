//! The cross-sparsified view: every Pauli string from three transforms,
//! and a sparsification that is a proof rather than a threshold.

use quantsim::crossview::{fwht, CrossView};
use quantsim::prelude::*;

fn word(n: usize, gates: usize, seed: u64) -> Circuit<C64> {
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for _ in 0..gates {
        let a = (rng.next_u64() % n as u64) as usize;
        let b = (rng.next_u64() % n as u64) as usize;
        match rng.next_u64() % 7 {
            0 => c.gate("h", vec![], vec![a]),
            1 => c.gate("t", vec![], vec![a]),
            2 => c.gate("s", vec![], vec![a]),
            3 => c.gate("rz", vec![0.37], vec![a]),
            4 => c.gate("ry", vec![0.61], vec![a]),
            _ => {
                if a != b {
                    c.gate("cx", vec![], vec![a, b])
                } else {
                    c.gate("h", vec![], vec![a])
                }
            }
        };
    }
    c
}

fn ghz(n: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    c.gate("h", vec![], vec![0]);
    for q in 1..n {
        c.gate("cx", vec![], vec![0, q]);
    }
    c
}

// ── the transform ────────────────────────────────────────────────────

#[test]
fn the_transform_is_its_own_inverse_up_to_scale() {
    let mut rng = Prng::new(3);
    for k in 1..=8usize {
        let n = 1usize << k;
        let v: Vec<f64> = (0..n)
            .map(|_| (rng.next_u64() % 1000) as f64 - 500.0)
            .collect();
        let mut w = v.clone();
        fwht(&mut w);
        fwht(&mut w);
        for (a, b) in v.iter().zip(&w) {
            assert!((a * n as f64 - b).abs() < 1e-9, "W(W(v)) must be N·v");
        }
    }
}

// ── it is every expectation, exactly ─────────────────────────────────

#[test]
fn three_transforms_give_every_pure_string() {
    let sim: Simulator = Simulator::new();
    let mut worst = 0.0f64;
    for seed in 0..30u64 {
        for n in 2..=4usize {
            let c = word(n, 10, seed * 5 + n as u64);
            let st = sim.run(&c).unwrap();
            let cv = CrossView::of(&*st).unwrap();
            for a in 0..1u64 << n {
                let want_x = pauli_expectation(&*st, &cv.ops(a, 0)).unwrap();
                worst = worst.max((want_x.re - cv.x_string(a)).abs());
                let want_z = pauli_expectation(&*st, &cv.ops(0, a)).unwrap();
                worst = worst.max((want_z.re - cv.z_string(a)).abs());
            }
        }
    }
    assert!(worst < 1e-12, "worst deviation {worst:e}");
}

#[test]
fn one_more_transform_gives_every_mixed_string() {
    // The drill: all 2^n values of ⟨X_A Z_B⟩ for a fixed A, from a
    // single transform, in the crate's own Hermitian normalization.
    let sim: Simulator = Simulator::new();
    let mut worst = 0.0f64;
    let mut checked = 0usize;
    for seed in 0..12u64 {
        for n in 2..=4usize {
            let c = word(n, 10, seed * 7 + n as u64);
            let st = sim.run(&c).unwrap();
            let cv = CrossView::of(&*st).unwrap();
            for a in 0..1u64 << n {
                let drilled = cv.drill(a);
                for b in 0..1u64 << n {
                    let want = pauli_expectation(&*st, &cv.ops(a, b)).unwrap();
                    worst = worst.max((want.re - drilled[b as usize]).abs());
                    // and the one-off route agrees with the batched one
                    worst = worst.max((cv.mixed(a, b) - drilled[b as usize]).abs());
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 3000, "only {checked} strings checked");
    assert!(worst < 1e-12, "worst deviation {worst:e}");
}

// ── the sparsification is a proof ────────────────────────────────────

#[test]
fn an_unfired_difference_contributes_exactly_zero() {
    // The claim that makes the sparsification exact rather than a
    // threshold: zero mass forces every mixed string on that difference
    // to vanish, so skipping it approximates nothing.
    let sim: Simulator = Simulator::new();
    for n in [3usize, 5, 8] {
        let st = sim.run(&ghz(n)).unwrap();
        let cv = CrossView::of(&*st).unwrap();
        let mut unfired = 0usize;
        for a in 0..1u64 << n {
            if cv.mass(a) > 1e-12 {
                continue;
            }
            unfired += 1;
            for b in 0..1u64 << n {
                assert_eq!(
                    cv.mixed(a, b),
                    0.0,
                    "n={n}: difference {a} has no mass but ⟨X_{a} Z_{b}⟩ ≠ 0"
                );
            }
        }
        assert!(
            unfired > 0,
            "n={n}: nothing was skipped, so nothing is proved"
        );
    }
}

#[test]
fn structure_fires_almost_nothing_and_the_generic_case_fires_everything() {
    // Both halves matter. A GHZ state uses two of its differences; a
    // scrambled one uses all of them, and the report says so instead of
    // quoting only the flattering case.
    let sim: Simulator = Simulator::new();
    for n in [8usize, 10, 12] {
        let st = sim.run(&ghz(n)).unwrap();
        let cv = CrossView::of(&*st).unwrap();
        assert_eq!(cv.fired(1e-12).len(), 2, "GHZ on {n} qubits");
        assert!(cv.sparsity(1e-12) < 0.01);
    }
    // Depth matters here, and it is worth being exact about why: at 40
    // gates a random word sometimes leaves a qubit untouched and the
    // sparsity swings between 0.03 and 1.0. By 80 it is 1.0 for every
    // seed, which is the generic case this is claiming.
    for seed in 0..5u64 {
        let st = sim.run(&word(8, 80, seed + 9)).unwrap();
        let cv = CrossView::of(&*st).unwrap();
        assert!(
            cv.sparsity(1e-12) > 0.99,
            "seed {seed}: a scrambled state should fire everything, got {:.3}",
            cv.sparsity(1e-12)
        );
    }
}

#[test]
fn the_mass_is_blind_to_phases_and_so_survives_a_diagonal_gate() {
    // The property the feedback loop leans on: the mass depends only on
    // amplitude moduli, so no diagonal gate can change which
    // differences fire.
    let sim: Simulator = Simulator::new();
    let n = 6usize;
    let mut c = word(n, 12, 4);
    let st = sim.run(&c).unwrap();
    let before = CrossView::of(&*st).unwrap();
    for q in 0..n {
        c.gate("rz", vec![0.3 + 0.2 * q as f64], vec![q]);
        c.gate("t", vec![], vec![q]);
    }
    let st = sim.run(&c).unwrap();
    let after = CrossView::of(&*st).unwrap();
    for a in 0..1u64 << n {
        assert!(
            (before.mass(a) - after.mass(a)).abs() < 1e-12,
            "difference {a}: mass moved under a diagonal gate"
        );
    }
}

// ── the derived views ────────────────────────────────────────────────

#[test]
fn the_entanglement_complex_separates_product_from_entangled() {
    let sim: Simulator = Simulator::new();
    let n = 6usize;
    let mut prod = Circuit::new(n);
    for q in 0..n {
        prod.gate("ry", vec![0.7], vec![q]);
    }
    let st = sim.run(&prod).unwrap();
    let cv = CrossView::of(&*st).unwrap();
    assert!(
        cv.entanglement_complex(1e-9).is_empty(),
        "a product state has no connected pairs: {:?}",
        cv.entanglement_complex(1e-9)
    );

    let st = sim.run(&ghz(n)).unwrap();
    let cv = CrossView::of(&*st).unwrap();
    let complex = cv.entanglement_complex(1e-9);
    assert_eq!(complex.len(), n * (n - 1) / 2, "GHZ correlates every pair");
    for (_, _, c) in complex {
        assert!((c - 1.0).abs() < 1e-12, "GHZ pairs correlate exactly: {c}");
    }
}

#[test]
fn a_width_past_reach_is_refused_rather_than_attempted() {
    assert!(CrossView::of_amplitudes(&[]).is_err());
    assert!(CrossView::of_amplitudes(&[C64::new(1.0, 0.0); 3]).is_err());
}

#[test]
fn the_transform_count_is_reported_and_constant() {
    let sim: Simulator = Simulator::new();
    for n in [2usize, 6, 10] {
        let st = sim.run(&word(n, 8, 2)).unwrap();
        let cv = CrossView::of(&*st).unwrap();
        assert_eq!(cv.transforms(), 5, "the cost must not grow with n");
        assert_eq!(cv.qubits(), n);
    }
}
