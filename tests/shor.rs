//! Shor's algorithm: the permutation kernels, the two phase-estimation
//! forms, the support law, and factoring end to end.

use quantsim::prelude::*;
use quantsim::shor::{
    self, apply_permutation, controlled_mul_mod, gather, mul_register_into, multiplicative_order,
    pow_mod, scatter, xor_copy, xor_square_into, OrderFinder, PhaseForm, Route,
};

/// Every (N, a) pair the suite exercises, with the order it must find.
fn cases() -> Vec<(u64, u64, u64)> {
    [
        (15u64, 2u64),
        (15, 7),
        (21, 2),
        (21, 5),
        (33, 5),
        (35, 2),
        (35, 6),
        (55, 2),
    ]
    .iter()
    .map(|&(n, a)| (n, a, multiplicative_order(a, n).expect("coprime")))
    .collect()
}

#[test]
fn modular_arithmetic_matches_direct_computation() {
    for n in 2u64..60 {
        for a in 0..n {
            for e in 0..8u64 {
                let mut want = 1u64 % n;
                for _ in 0..e {
                    want = want * a % n;
                }
                assert_eq!(pow_mod(a, e, n), want, "{a}^{e} mod {n}");
            }
        }
    }
    // The order is the first return to 1, by definition.
    for (n, a, r) in cases() {
        assert_eq!(pow_mod(a, r, n), 1);
        for k in 1..r {
            assert_ne!(pow_mod(a, k, n), 1, "{a} mod {n} returned early at {k}");
        }
    }
}

#[test]
fn gather_and_scatter_are_inverse_on_the_named_qubits() {
    let qubits = [1usize, 3, 4];
    for index in 0..64u64 {
        let v = gather(index, &qubits);
        assert!(v < 8);
        assert_eq!(scatter(index, &qubits, v), index);
        for w in 0..8u64 {
            let moved = scatter(index, &qubits, w);
            assert_eq!(gather(moved, &qubits), w);
            // Untouched bits stay untouched.
            let mask: u64 = qubits.iter().map(|&q| 1u64 << q).sum();
            assert_eq!(moved & !mask, index & !mask);
        }
    }
}

#[test]
fn a_non_injective_map_is_refused_rather_than_losing_amplitude() {
    let sim: Simulator = Simulator::new();
    let mut state = sim.backends().create("dense", 3).unwrap();
    let mut c: Circuit = Circuit::new(3);
    c.h(0).h(1);
    c.bind(sim.registry()).unwrap().run(state.as_mut()).unwrap();
    // Collapse everything onto |0⟩: not a permutation.
    let err = apply_permutation(state.as_mut(), "collapse", &|_| 0).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("not injective"), "unexpected error: {text}");
}

#[test]
fn controlled_multiplication_is_the_permutation_it_claims() {
    let sim: Simulator = Simulator::new();
    let n = 15u64;
    let work: Vec<usize> = (0..4).collect();
    let control = 4usize;
    for a in [2u64, 4, 7, 11, 13] {
        for index in 0..32u64 {
            let mut state = sim.backends().create("sparse", 5).unwrap();
            state.load(&[(index, C64::new(1.0, 0.0))]).unwrap();
            controlled_mul_mod(state.as_mut(), Some(control), &work, a, n).unwrap();
            let mut got = None;
            state.for_each_nonzero(&mut |i, _| got = Some(i));
            let got = got.unwrap();
            let y = gather(index, &work);
            let want = if (index >> control) & 1 == 1 && y < n {
                scatter(index, &work, a * y % n)
            } else {
                index
            };
            assert_eq!(got, want, "a={a} index={index}");
        }
    }
}

#[test]
fn multiplying_by_the_inverse_undoes_the_multiplication() {
    let sim: Simulator = Simulator::new();
    let n = 35u64;
    let work: Vec<usize> = (0..6).collect();
    let mut state = sim.backends().create("dense", 7).unwrap();
    let mut c: Circuit = Circuit::new(7);
    c.h(6);
    for q in 0..3 {
        c.h(q);
    }
    c.bind(sim.registry()).unwrap().run(state.as_mut()).unwrap();
    let before: Vec<(u64, C64)> = {
        let mut v = Vec::new();
        state.for_each_nonzero(&mut |i, a| v.push((i, a)));
        v.sort_by_key(|e| e.0);
        v
    };
    let a = 13u64;
    let inv = quantsim::padic::mod_inv(a, n).unwrap();
    controlled_mul_mod(state.as_mut(), Some(6), &work, a, n).unwrap();
    controlled_mul_mod(state.as_mut(), Some(6), &work, inv, n).unwrap();
    let mut after: Vec<(u64, C64)> = Vec::new();
    state.for_each_nonzero(&mut |i, x| after.push((i, x)));
    after.sort_by_key(|e| e.0);
    assert_eq!(before.len(), after.len());
    for ((i, x), (j, y)) in before.iter().zip(after.iter()) {
        assert_eq!(i, j);
        assert!((x.re - y.re).abs() < 1e-12 && (x.im - y.im).abs() < 1e-12);
    }
}

#[test]
fn the_reversible_primitives_are_involutions() {
    let sim: Simulator = Simulator::new();
    let n = 21u64;
    let src: Vec<usize> = (0..5).collect();
    let dst: Vec<usize> = (5..10).collect();
    for value in 0..32u64 {
        // xor_square_into, twice, is the identity.
        let mut state = sim.backends().create("sparse", 10).unwrap();
        let start = scatter(0, &src, value);
        state.load(&[(start, C64::new(1.0, 0.0))]).unwrap();
        xor_square_into(state.as_mut(), &src, &dst, n).unwrap();
        if value < n {
            let mut mid = None;
            state.for_each_nonzero(&mut |i, _| mid = Some(i));
            assert_eq!(gather(mid.unwrap(), &dst), value * value % n);
        }
        xor_square_into(state.as_mut(), &src, &dst, n).unwrap();
        let mut back = None;
        state.for_each_nonzero(&mut |i, _| back = Some(i));
        assert_eq!(
            back.unwrap(),
            start,
            "square is not an involution at {value}"
        );

        // xor_copy, twice, is the identity.
        let mut state = sim.backends().create("sparse", 10).unwrap();
        state.load(&[(start, C64::new(1.0, 0.0))]).unwrap();
        xor_copy(state.as_mut(), &src, &dst).unwrap();
        xor_copy(state.as_mut(), &src, &dst).unwrap();
        let mut back = None;
        state.for_each_nonzero(&mut |i, _| back = Some(i));
        assert_eq!(back.unwrap(), start);
    }
}

#[test]
fn register_multiplication_inverts_itself() {
    let sim: Simulator = Simulator::new();
    let n = 35u64;
    let target: Vec<usize> = (0..6).collect();
    let source: Vec<usize> = (6..12).collect();
    for x in 0..n {
        for y in [1u64, 2, 3, 4, 6, 8, 9, 11, 12, 13, 16, 17] {
            let start = scatter(scatter(0, &target, x), &source, y);
            let mut state = sim.backends().create("sparse", 12).unwrap();
            state.load(&[(start, C64::new(1.0, 0.0))]).unwrap();
            mul_register_into(state.as_mut(), &target, &source, n, false).unwrap();
            let mut mid = None;
            state.for_each_nonzero(&mut |i, _| mid = Some(i));
            assert_eq!(gather(mid.unwrap(), &target), x * y % n, "x={x} y={y}");
            mul_register_into(state.as_mut(), &target, &source, n, true).unwrap();
            let mut back = None;
            state.for_each_nonzero(&mut |i, _| back = Some(i));
            assert_eq!(back.unwrap(), start, "x={x} y={y}");
        }
    }
}

#[test]
fn semiclassical_phase_estimation_recovers_the_order() {
    let sim: Simulator = Simulator::new();
    for (n, a, r) in cases() {
        let finder = OrderFinder::new(n, a).unwrap();
        let mut hits = 0;
        for seed in 0..12u64 {
            let mut rng = Prng::new(seed);
            let est = finder
                .estimate(&sim, PhaseForm::Semiclassical, &mut rng)
                .unwrap();
            if shor::order_from_phase(est.phase, n, a, n) == Some(r) {
                hits += 1;
            }
        }
        // The s = 0 branch has probability 1/r and yields nothing; a
        // handful of seeds must still land the order most of the time.
        assert!(hits >= 6, "N={n} a={a} r={r}: only {hits}/12 recovered");
    }
}

#[test]
fn both_phase_forms_recover_the_same_order() {
    let sim: Simulator = Simulator::new();
    for (n, a, r) in [(15u64, 7u64, 4u64), (21, 2, 6), (35, 6, 2)] {
        let finder = OrderFinder::new(n, a).unwrap();
        let mut full = 0;
        for seed in 0..8u64 {
            let mut rng = Prng::new(seed);
            let est = finder
                .estimate(&sim, PhaseForm::FullRegister, &mut rng)
                .unwrap();
            if shor::order_from_phase(est.phase, n, a, n) == Some(r) {
                full += 1;
            }
        }
        assert!(
            full >= 4,
            "N={n} a={a}: full-register found the order {full}/8"
        );
    }
}

#[test]
fn the_semiclassical_form_is_narrower_by_the_whole_phase_register() {
    for (n, a, _) in cases() {
        let finder = OrderFinder::new(n, a).unwrap();
        let w = finder.work_bits();
        assert_eq!(finder.width(PhaseForm::Semiclassical), w + 1);
        assert_eq!(finder.width(PhaseForm::FullRegister), w + finder.phase_bits);
        assert_eq!(finder.phase_bits, 2 * w + 1);
    }
}

#[test]
fn the_orbit_is_the_order_and_the_support_is_the_orbit_or_twice_it() {
    let sim: Simulator = Simulator::new();
    for (n, a, r) in cases() {
        let finder = OrderFinder::new(n, a).unwrap();
        let mut rng = Prng::new(3);
        let est = finder
            .estimate(&sim, PhaseForm::Semiclassical, &mut rng)
            .unwrap();
        // The orbit is exactly r; the support is the orbit or twice it,
        // the factor of two being the ancilla mid-round.
        assert_eq!(est.peak_orbit, r as usize, "N={n} a={a}: orbit is not r");
        assert!(
            est.peak_support == est.peak_orbit || est.peak_support == 2 * est.peak_orbit,
            "N={n} a={a}: support {} is neither r nor 2r",
            est.peak_support
        );
        assert!(est.peak_support <= 2 * r as usize);
        assert_eq!(est.modular_multiplications, finder.phase_bits);
    }
    // Same order, different widths: 2 has order 12 mod both 35 and 65,
    // and the cost is the order, not the width.
    let mut peaks = Vec::new();
    for n in [35u64, 65] {
        let finder = OrderFinder::new(n, 2).unwrap();
        let mut rng = Prng::new(11);
        let est = finder
            .estimate(&sim, PhaseForm::Semiclassical, &mut rng)
            .unwrap();
        peaks.push((finder.work_bits(), est.peak_orbit));
    }
    assert_ne!(peaks[0].0, peaks[1].0, "the widths were meant to differ");
    assert_eq!(peaks[0].1, peaks[1].1, "the orbit followed the width");
}

#[test]
fn the_support_regime_is_decided_by_the_parity_of_the_order() {
    // peak_support = r for even r, 2r for odd r — and the mechanism is
    // the 2-adic valuation: the ladder's multipliers a^{2^k mod r}
    // generate the odd part of the orbit, so the trajectory plateaus at
    // r / 2^{v₂(r)} and doubles on each of the last v₂(r) rounds.
    let sim: Simulator = Simulator::new();
    let mut checked = 0;
    for n in [
        15u64, 21, 33, 35, 39, 55, 65, 91, 95, 119, 221, 899, 4087, 7387, 32399,
    ] {
        for a in [2u64, 3, 5, 7] {
            if a >= n || quantsim::padic::gcd(a, n) != 1 {
                continue;
            }
            let r = multiplicative_order(a, n).expect("coprime");
            let finder = OrderFinder::new(n, a).unwrap();
            let mut rng = Prng::new(1);
            let est = finder
                .estimate(&sim, PhaseForm::Semiclassical, &mut rng)
                .unwrap();
            let predicted = if r % 2 == 0 { r } else { 2 * r };
            assert_eq!(
                est.peak_support as u64, predicted,
                "N={n} a={a} r={r}: support law broken"
            );
            assert_eq!(est.peak_orbit as u64, r, "N={n} a={a}: orbit is not r");

            // The plateau is the odd part, and it is where the trajectory
            // sits v₂(r) rounds from the end.
            let v2 = r.trailing_zeros() as usize;
            let traj = &est.orbit_trajectory;
            assert_eq!(
                traj[traj.len() - 1 - v2] as u64,
                r >> v2,
                "N={n} a={a} r={r}: plateau is not the odd part"
            );
            // And support is exactly twice the largest orbit a round doubled.
            let doubled = *traj[..traj.len() - 1].iter().max().unwrap();
            assert_eq!(est.peak_support, 2 * doubled);
            assert_eq!(est.peak_orbit, *traj.iter().max().unwrap());
            checked += 1;
        }
    }
    assert!(checked >= 40, "only {checked} pairs exercised");
}

#[test]
fn the_support_reaches_twice_the_orbit_once_saturation_beats_the_last_round() {
    // The regime the small cases do not show: when the orbit fills before
    // the final round, a later round doubles the register's nonzero
    // count. Which regime a case lands in is a property of the orbit's
    // saturation step, not of the width — 32399 shows both, base by base.
    let sim: Simulator = Simulator::new();
    let (n, a) = (32_399u64, 3u64); // 179 × 181
    let r = multiplicative_order(a, n).expect("coprime");
    let finder = OrderFinder::new(n, a).unwrap();
    let mut rng = Prng::new(1);
    let est = finder
        .estimate(&sim, PhaseForm::Semiclassical, &mut rng)
        .unwrap();
    assert_eq!(est.peak_orbit, r as usize);
    assert_eq!(est.peak_support, 2 * r as usize, "expected the 2r regime");
    assert_eq!(est.qubits, finder.work_bits() + 1);

    // Same modulus, same width, base 2 instead: the other regime.
    let finder = OrderFinder::new(n, 2).unwrap();
    let r2 = multiplicative_order(2, n).expect("coprime");
    let mut rng = Prng::new(1);
    let est2 = finder
        .estimate(&sim, PhaseForm::Semiclassical, &mut rng)
        .unwrap();
    assert_eq!(est2.peak_orbit, r2 as usize);
    assert_eq!(est2.peak_support, r2 as usize, "expected the r regime");
    assert_eq!(
        est.qubits, est2.qubits,
        "the two regimes are not a width effect"
    );
}

#[test]
fn the_banded_transform_costs_fewer_gates_and_still_finds_the_order() {
    let sim: Simulator = Simulator::new();
    let (n, a, r) = (21u64, 2u64, 6u64);
    let exact = OrderFinder::new(n, a).unwrap();
    let banded = OrderFinder::new(n, a)
        .unwrap()
        .with_qft_epsilon(std::f64::consts::PI / 16.0);
    let mut rng = Prng::new(5);
    let e = exact
        .estimate(&sim, PhaseForm::FullRegister, &mut rng)
        .unwrap();
    let mut rng = Prng::new(5);
    let b = banded
        .estimate(&sim, PhaseForm::FullRegister, &mut rng)
        .unwrap();
    assert!(
        b.phase_gates < e.phase_gates,
        "{} !< {}",
        b.phase_gates,
        e.phase_gates
    );
    let mut hits = 0;
    for seed in 0..8u64 {
        let mut rng = Prng::new(seed);
        let est = banded
            .estimate(&sim, PhaseForm::FullRegister, &mut rng)
            .unwrap();
        if shor::order_from_phase(est.phase, n, a, n) == Some(r) {
            hits += 1;
        }
    }
    assert!(hits >= 3, "banded transform recovered the order {hits}/8");
}

#[test]
fn the_ladder_width_curve_is_periodic_in_the_order() {
    // modwidth's claim, on this module's ladder: a^{2^k} cycles with the
    // multiplicative order of 2 mod r, so the width curve cycles with it.
    for (n, a, r) in cases() {
        let widths = shor::ladder_widths(n, a, 16);
        let odd = r >> r.trailing_zeros();
        if odd == 1 {
            continue; // 2 has no order mod 1
        }
        let period = multiplicative_order(2, odd).unwrap() as usize;
        let start = r.trailing_zeros() as usize;
        for k in start..widths.len() - period {
            assert_eq!(
                widths[k],
                widths[k + period],
                "N={n} a={a} r={r}: width curve broke period {period} at {k}"
            );
        }
    }
}

#[test]
fn the_classical_reduction_only_splits_when_it_should() {
    for (n, a, r) in cases() {
        let split = shor::split_from_order(n, a, r);
        if let Some((p, q)) = split {
            assert_eq!(p * q, n);
            assert!(p > 1 && q > 1);
            assert_eq!(r % 2, 0);
            assert_ne!(pow_mod(a, r / 2, n), n - 1);
        }
    }
    // An odd order can never split: 4 has order 3 mod 21.
    assert_eq!(multiplicative_order(4, 21), Some(3));
    assert_eq!(shor::split_from_order(21, 4, 3), None);
    // Nor can a half-power of −1: 14 ≡ −1 mod 15 has order 2.
    assert_eq!(multiplicative_order(14, 15), Some(2));
    assert_eq!(pow_mod(14, 1, 15), 14);
    assert_eq!(shor::split_from_order(15, 14, 2), None);
    // But an even order whose half-power is neither ±1 does split.
    assert_eq!(shor::split_from_order(35, 6, 2), Some((5, 7)));
}

#[test]
fn order_recovery_survives_a_convergent_that_undershoots() {
    // 5/12 has convergents 1/2, 2/5, 5/12; the order 12 is only the last.
    let phase = 5.0 / 12.0;
    assert_eq!(shor::order_from_phase(phase, 35, 2, 35), Some(12));
    // A phase of zero carries no order at all, and says so.
    assert_eq!(shor::order_from_phase(0.0, 35, 2, 35), None);
}

#[test]
fn factoring_takes_the_classical_shortcuts_and_names_them() {
    let sim: Simulator = Simulator::new();
    let mut rng = Prng::new(1);
    let even = shor::factor(&sim, 34, PhaseForm::Semiclassical, &mut rng, 4).unwrap();
    assert_eq!(even.route, Route::Even);
    assert_eq!(even.factors, Some((2, 17)));

    let power = shor::factor(&sim, 27, PhaseForm::Semiclassical, &mut rng, 4).unwrap();
    assert_eq!(power.route, Route::PerfectPower(3, 3));

    let prime = shor::factor(&sim, 31, PhaseForm::Semiclassical, &mut rng, 4).unwrap();
    assert_eq!(prime.route, Route::Prime);
    assert_eq!(prime.factors, None);
}

#[test]
fn the_raised_modulus_cap_is_usable_and_still_bounded() {
    // MAX_MODULUS went from 2^31 to its structural bound 2^62, and the
    // default phase-bit rule changed from *erroring* above w = 30 to
    // clamping at MAX_PHASE_BITS. Both halves of that need to hold.
    let sim: Simulator = Simulator::new();
    assert_eq!(shor::MAX_MODULUS, 1 << 62);

    // Below w = 31 the default 2w+1 is under MAX_PHASE_BITS and stands.
    // a = N-1 has order 2 for every N, so the support is 2 at any width.
    let n = 1_040_399u64; // 20 bits
    let finder = OrderFinder::new(n, n - 1).unwrap();
    assert_eq!(finder.work_bits(), 20);
    assert_eq!(finder.phase_bits, 41, "clamp fired too early");
    let mut rng = Prng::new(1);
    let est = finder
        .estimate(&sim, PhaseForm::Semiclassical, &mut rng)
        .unwrap();
    assert_eq!(est.qubits, 21);
    assert_eq!(est.peak_orbit, 2);

    // A 40-bit modulus: past the old 2^31 cap, and it runs.
    let n = 1_099_511_627_689u64; // 40 bits, odd
    let finder = OrderFinder::new(n, n - 1).unwrap();
    assert_eq!(finder.work_bits(), 40);
    assert_eq!(
        finder.phase_bits,
        shor::MAX_PHASE_BITS,
        "clamp did not fire"
    );
    let mut rng = Prng::new(1);
    let est = finder
        .estimate(&sim, PhaseForm::Semiclassical, &mut rng)
        .unwrap();
    assert_eq!(est.qubits, 41);
    assert_eq!(est.peak_orbit, 2);

    // Past w = 31 the default 2w+1 exceeds MAX_PHASE_BITS and is clamped
    // rather than refused.
    let wide_n = (1u64 << 45) - 1;
    let finder = OrderFinder::new(wide_n, wide_n - 1).unwrap();
    assert_eq!(finder.work_bits(), 45);
    assert_eq!(finder.phase_bits, shor::MAX_PHASE_BITS);
    let mut rng = Prng::new(1);
    let est = finder
        .estimate(&sim, PhaseForm::Semiclassical, &mut rng)
        .unwrap();
    assert_eq!(est.qubits, 46);
    assert_eq!(est.peak_orbit, 2);

    // And the register ceiling is still enforced: a modulus needing more
    // than 62 work qubits leaves no room for the ancilla.
    assert!(OrderFinder::new(u64::MAX, u64::MAX - 2).is_err());
}

#[test]
fn factoring_end_to_end() {
    let sim: Simulator = Simulator::new();
    for n in [15u64, 21, 33, 35, 39, 51, 55] {
        for seed in 0..4u64 {
            let mut rng = Prng::new(seed);
            let report = shor::factor(&sim, n, PhaseForm::Semiclassical, &mut rng, 16).unwrap();
            let (p, q) = report
                .factors
                .unwrap_or_else(|| panic!("N={n} seed={seed} found nothing"));
            assert_eq!(p * q, n);
            assert!(p > 1 && q > 1);
            assert!(report.modular_multiplications > 0 || report.attempts.is_empty());
        }
    }
}

#[test]
fn factoring_end_to_end_on_the_full_register_form() {
    let sim: Simulator = Simulator::new();
    for n in [15u64, 21, 35] {
        let mut rng = Prng::new(2);
        let report = shor::factor(&sim, n, PhaseForm::FullRegister, &mut rng, 16).unwrap();
        let (p, q) = report
            .factors
            .unwrap_or_else(|| panic!("N={n} found nothing"));
        assert_eq!(p * q, n);
    }
}
