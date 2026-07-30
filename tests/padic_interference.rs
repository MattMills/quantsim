//! The p-adic phase register: CRT factorization, the interference
//! character in the trailing digits, fringe periods from valuations, and
//! the measured cost of the sweep order.

use quantsim::padic::{
    best_rational, character, compare_diagonals, crt_phase_factors, factor, factored_field,
    fringe_period, gcd, geometry_radix, lcm, mod_inv, predicted_writes_per_point, resolve, sweep,
    CrtDiagonal, Order, Radix, Sweep, Wave, WaveSystem,
};
use quantsim::prelude::*;

// ── the arithmetic the representation rests on ───────────────────────

#[test]
fn factorization_and_lcm_are_right() {
    assert_eq!(factor(75600), vec![(2, 4), (3, 3), (5, 2), (7, 1)]);
    assert_eq!(factor(97), vec![(97, 1)]);
    assert_eq!(factor(1024), vec![(2, 10)]);
    assert_eq!(lcm(12, 18), Some(36));
    assert_eq!(gcd(75600, 37800), 37800);
    // no lcm may exceed the module's structural bound
    assert_eq!(lcm(1 << 39, 3), None);
}

#[test]
fn mod_inv_is_an_inverse_or_absent() {
    for m in 2u64..60 {
        for a in 0u64..m {
            match mod_inv(a, m) {
                Some(inv) => {
                    assert_eq!(gcd(a, m), 1, "inverse offered for non-coprime {a} mod {m}");
                    assert_eq!((a as u128 * inv as u128 % m as u128) as u64, 1);
                }
                None => assert_ne!(gcd(a, m), 1, "inverse refused for coprime {a} mod {m}"),
            }
        }
    }
}

#[test]
fn crt_is_a_bijection_onto_the_component_residues() {
    for m in [2 * 3 * 5u64, 8 * 9, 16 * 27 * 25, 4 * 9 * 5 * 7] {
        let radix = Radix::of(m).unwrap();
        let mut seen = std::collections::HashSet::new();
        for a in 0..m {
            let r = radix.residues(a);
            assert!(seen.insert(r.clone()), "residue vector repeated at {a}");
            assert_eq!(radix.reconstruct(&r).unwrap(), a);
        }
        assert_eq!(seen.len() as u64, m);
    }
}

// ── fact 1: the phase field factors, exactly ─────────────────────────

#[test]
fn crt_phase_factors_reproduce_the_global_phase() {
    let radix = Radix::of(8 * 9 * 5).unwrap();
    let m = radix.modulus();
    let dims = radix.component_moduli();
    for a in [0u64, 1, 7, 179, 359] {
        let f = crt_phase_factors(a, &radix);
        for x in 0..m {
            let point = radix.residues(x);
            // ∏_i e^{2πi c_i x_i / m_i}  ==  e^{2πi a x / M}
            let mut prod = C64::new(1.0, 0.0);
            for i in 0..dims.len() {
                let e = (f[i] as u128 * point[i] as u128) % dims[i] as u128;
                prod *= cis(std::f64::consts::TAU * e as f64 / dims[i] as f64);
            }
            let want = cis(
                std::f64::consts::TAU * ((a as u128 * x as u128) % m as u128) as f64 / m as f64,
            );
            assert!(
                (prod - want).norm() < 1e-12,
                "a={a} x={x}: {prod:?} vs {want:?}"
            );
        }
    }
}

#[test]
fn the_field_stays_a_product_and_matches_the_direct_sum() {
    let radix = Radix::of(16 * 27 * 25 * 7).unwrap();
    let system = WaveSystem::new(
        radix.clone(),
        1,
        vec![
            Wave::new(0, vec![1]),
            Wave::new(1234, vec![37]),
            Wave::new(7, vec![2100]),
            Wave::new(999, vec![-11]),
        ],
    )
    .unwrap();
    let probes: Vec<u64> = (0..64).map(|k| (k * 3121 + 7) % radix.modulus()).collect();
    let field = factored_field(&system, &probes).unwrap();

    assert_eq!(field.rank, 4);
    assert_eq!(field.dims, vec![16, 27, 25, 7]);
    assert_eq!(field.dense_entries, 75_600);
    assert_eq!(field.factored_entries, 4 * (16 + 27 + 25 + 7));
    assert!(field.compression() > 250.0);
    // no interaction ever coupled two CRT components
    assert!(field.stayed_product(), "volumes {:?}", field.volumes);
    assert!(
        field.max_deviation < 1e-13,
        "deviation {:.3e}",
        field.max_deviation
    );
}

#[test]
fn factored_field_refuses_a_multidimensional_argument() {
    let radix = Radix::of(60).unwrap();
    let sys = WaveSystem::new(radix, 2, vec![Wave::new(0, vec![1, 1])]).unwrap();
    assert!(factored_field(&sys, &[0, 1]).is_err());
}

// ── fact 2: the character lives in the trailing digits ───────────────

#[test]
fn character_is_the_root_of_unity_order_everywhere() {
    for m in [8 * 9u64, 16 * 25, 2 * 3 * 5 * 7] {
        let radix = Radix::of(m).unwrap();
        for delta in 0..m {
            let c = character(delta, &radix);
            let g = if delta == 0 { m } else { gcd(delta, m) };
            assert_eq!(c.coherence, g, "delta {delta} mod {m}");
            assert_eq!(c.order, m / g);
            assert_eq!(c.constructive(), delta == 0);
            assert_eq!(c.antiphase(), c.order == 2);
            // and the antiphase claim is the physics: the amplitude cancels
            if c.antiphase() {
                let sum = C64::new(1.0, 0.0)
                    + cis(std::f64::consts::TAU * delta as f64 / m as f64);
                assert!(sum.norm() < 1e-12, "antiphase did not cancel: {sum:?}");
            }
        }
    }
}

#[test]
fn character_cost_is_in_the_prime_count_not_the_modulus() {
    // Same primes, very different depth and modulus.
    let shallow = Radix::of(2u64.pow(10) * 3u64.pow(5)).unwrap();
    let deep = Radix::of(2u64.pow(30) * 3u64.pow(5)).unwrap();
    assert_eq!(shallow.depth(), 15);
    assert_eq!(deep.depth(), 35);
    assert!(deep.modulus() / shallow.modulus() > 1_000_000);

    let mut rng = Prng::new(5);
    let mean = |r: &Radix, rng: &mut Prng| {
        let n = 40_000;
        let total: usize = (0..n)
            .map(|_| character(rng.next_u64() % r.modulus(), r).digit_reads)
            .sum();
        total as f64 / n as f64
    };
    let ms = mean(&shallow, &mut rng);
    let md = mean(&deep, &mut rng);

    // Σ_i p_i/(p_i−1) = 2 + 1.5 = 3.5 for both: the cost is in the primes.
    for (label, m) in [("shallow", ms), ("deep", md)] {
        assert!((m - 3.5).abs() < 0.1, "{label} mean {m}");
    }
    assert!((ms - md).abs() < 0.1, "{ms} vs {md}");
    // and it is far below either full depth
    assert!(md < 0.2 * deep.depth() as f64);

    // a depth-1 radix is the degenerate case: every component reads its
    // single digit, so the cost is exactly the component count.
    let flat = Radix::of(2 * 3 * 5 * 7).unwrap();
    for delta in 0..flat.modulus() {
        assert_eq!(character(delta, &flat).digit_reads, 4);
    }
}

#[test]
fn resolution_decides_exactly_and_the_residual_shrinks() {
    let radix = Radix::of(2u64.pow(6) * 3 * 5 * 7).unwrap();
    let mut rng = Prng::new(19);
    for d in [
        CrtDiagonal::Interleaved,
        CrtDiagonal::Sequential,
        CrtDiagonal::CoarsestFirst,
        CrtDiagonal::Shuffled(3),
    ] {
        // every ordering consumes every digit exactly once
        let steps = d.steps(&radix);
        assert_eq!(steps.len(), radix.depth() as usize);
        let mut per_comp = vec![Vec::new(); radix.components().len()];
        for &(ci, level) in &steps {
            per_comp[ci].push(level);
        }
        for (ci, levels) in per_comp.iter().enumerate() {
            let want: Vec<u32> = (0..radix.components()[ci].1).collect();
            assert_eq!(levels, &want, "component {ci} digits out of order");
        }
        // the verdict is exact, and the residual only ever shrinks
        for _ in 0..500 {
            let delta = rng.next_u64() % radix.modulus();
            let r = resolve(delta, &radix, d);
            assert_eq!(r.in_phase, Some(delta == 0));
            assert!(r.classes <= radix.modulus());
            assert_eq!(r.classes * r.modulus, radix.modulus());
        }
        assert_eq!(resolve(0, &radix, d).classes, 1);
    }
}

#[test]
fn coarsest_first_decides_fastest_and_sequential_slowest() {
    let radix = Radix::of(16 * 27 * 25 * 7).unwrap();
    let cmp = compare_diagonals(&radix, 20_000, 0xC0FFEE);
    assert_eq!(cmp.full_depth, 10);
    let get = |name: &str| cmp.orderings.iter().find(|o| o.0 == name).unwrap().1;
    let (coarse, inter, seq) = (
        get("coarsest-first"),
        get("interleaved"),
        get("sequential"),
    );
    assert!(coarse < inter, "{coarse} !< {inter}");
    assert!(inter < seq, "{inter} !< {seq}");
    // every ordering is far below the full depth: the character is cheap
    for &(label, mean, _) in &cmp.orderings {
        assert!(mean < 2.5, "{label} mean {mean}");
    }
}

// ── the fringe period, from digits alone ─────────────────────────────

#[test]
fn fringe_period_matches_a_brute_force_scan() {
    let radix = Radix::of(2u64.pow(4) * 3 * 5 * 7).unwrap();
    let m = radix.modulus();
    let system = WaveSystem::new(
        radix,
        1,
        vec![
            Wave::new(0, vec![1]),
            Wave::new(11, vec![7]),
            Wave::new(3, vec![84]),
            Wave::new(0, vec![1]),
        ],
    )
    .unwrap();
    for (a, b) in [(0usize, 1usize), (0, 2), (1, 2), (0, 3)] {
        let claimed = fringe_period(&system, (a, b), 0);
        let base = system.delta(a, b, &[0]);
        let scanned = (1..=m).find(|&t| system.delta(a, b, &[t as i64]) == base);
        match claimed {
            // a varying pair repeats first at exactly the claimed period
            Some(p) => assert_eq!(scanned, Some(p), "pair ({a},{b})"),
            // a non-varying pair is constant, so it "repeats" at one step
            None => assert_eq!(scanned, Some(1), "pair ({a},{b})"),
        }
        if let Some(p) = claimed {
            // the character really does repeat at that period, everywhere
            for k in 0..20i64 {
                assert_eq!(
                    system.character_at(a, b, &[k]),
                    system.character_at(a, b, &[k + p as i64])
                );
            }
        }
    }
    // identical waves never vary: no period at all
    assert_eq!(fringe_period(&system, (0, 3), 0), None);
}

// ── the sweep: the cost of the visiting order ────────────────────────

#[test]
fn lattice_orders_hit_the_closed_form_and_scrambling_does_not() {
    for (p, n, side) in [(2u64, 12u32, 64i64), (3, 8, 81)] {
        let radix = Radix::of(p.pow(n)).unwrap();
        // the grid covers ℤ/M exactly once, so the closed form applies
        let system = WaveSystem::new(
            radix.clone(),
            2,
            vec![Wave::new(0, vec![1, side]), Wave::new(0, vec![0, 0])],
        )
        .unwrap();
        assert_eq!((side * side) as u64, radix.modulus());

        let lattice_pred = predicted_writes_per_point(&radix, true);
        let scrambled_pred = predicted_writes_per_point(&radix, false);

        for order in [Order::Axis(0), Order::Axis(1)] {
            let r = sweep(
                &system,
                (0, 1),
                &[side, side],
                order,
                CrtDiagonal::Interleaved,
                false,
            )
            .unwrap();
            assert_eq!(r.points, (side * side) as usize);
            assert!(
                (r.writes_per_point - lattice_pred).abs() < 0.01,
                "p={p}: measured {} vs predicted {lattice_pred}",
                r.writes_per_point
            );
        }

        let shuffled = sweep(
            &system,
            (0, 1),
            &[side, side],
            Order::Shuffled(7),
            CrtDiagonal::Interleaved,
            false,
        )
        .unwrap();
        assert!(
            (shuffled.writes_per_point - scrambled_pred).abs() < 0.1,
            "p={p}: scrambled {} vs predicted {scrambled_pred}",
            shuffled.writes_per_point
        );
        // and the gap is the whole claim
        assert!(shuffled.writes_per_point > 2.5 * lattice_pred);
    }
}

#[test]
fn morton_order_needs_power_of_two_extents() {
    assert!(Order::Morton.points(&[8, 8]).is_ok());
    assert!(Order::Morton.points(&[8, 6]).is_err());
    // and it is a permutation of the same grid
    let mut a = Order::Morton.points(&[8, 4]).unwrap();
    let mut b = Order::Axis(0).points(&[8, 4]).unwrap();
    a.sort();
    b.sort();
    assert_eq!(a, b);
}

#[test]
fn every_order_visits_every_point_exactly_once() {
    for order in [
        Order::Axis(0),
        Order::Axis(1),
        Order::Axis(2),
        Order::Morton,
        Order::Shuffled(42),
    ] {
        let pts = order.points(&[4, 8, 2]).unwrap();
        assert_eq!(pts.len(), 64);
        let uniq: std::collections::HashSet<_> = pts.iter().cloned().collect();
        assert_eq!(uniq.len(), 64);
        for p in &pts {
            assert!(p[0] < 4 && p[1] < 8 && p[2] < 2, "{p:?} outside the grid");
        }
    }
    assert!(Order::Axis(3).points(&[4, 8, 2]).is_err());
}

#[test]
fn the_journal_resumes_and_rewinds_without_changing_the_answer() {
    let radix = Radix::of(2u64.pow(10) * 3).unwrap();
    let system = WaveSystem::new(
        radix.clone(),
        2,
        vec![Wave::new(5, vec![1, 32]), Wave::new(0, vec![0, 0])],
    )
    .unwrap();

    let one_shot = sweep(
        &system,
        (0, 1),
        &[32, 32],
        Order::Axis(0),
        CrtDiagonal::Interleaved,
        false,
    )
    .unwrap();

    // the same sweep, interrupted three times
    let mut s = Sweep::new(
        &system,
        (0, 1),
        &[32, 32],
        Order::Axis(0),
        CrtDiagonal::Interleaved,
        false,
    )
    .unwrap();
    assert_eq!(s.resume(&system, 100), 100);
    assert_eq!(s.resume(&system, 300), 300);
    assert_eq!(s.resume(&system, 10_000), 1024 - 400);
    assert_eq!(s.resume(&system, 10), 0);
    let staged = s.report(&radix);

    assert_eq!(staged.points, one_shot.points);
    assert_eq!(staged.steps, one_shot.steps);
    assert_eq!(staged.in_phase, one_shot.in_phase);
    // digit writes differ only by the two resumption boundaries, where the
    // register state is re-primed; the journal records that honestly.
    assert!(staged.digit_writes >= one_shot.digit_writes);

    let verdicts: Vec<_> = s.entries().iter().map(|e| e.resolution.in_phase).collect();
    s.rewind_to(400);
    assert_eq!(s.entries().len(), 400);
    assert_eq!(s.remaining(), 1024 - 400);
    s.resume(&system, 10_000);
    let replayed: Vec<_> = s.entries().iter().map(|e| e.resolution.in_phase).collect();
    assert_eq!(verdicts, replayed, "rewind + replay changed a verdict");
}

#[test]
fn the_verdict_cache_never_lies() {
    let radix = Radix::of(2u64.pow(8) * 3 * 5).unwrap();
    let system = WaveSystem::new(
        radix.clone(),
        2,
        vec![Wave::new(0, vec![1, 40]), Wave::new(0, vec![0, 0])],
    )
    .unwrap();
    let mut s = Sweep::new(
        &system,
        (0, 1),
        &[40, 40],
        Order::Axis(0),
        CrtDiagonal::Interleaved,
        true,
    )
    .unwrap();
    s.resume(&system, 10_000);
    for e in s.entries() {
        assert_eq!(
            e.resolution.in_phase,
            Some(e.delta == 0),
            "cache answered wrong at {:?}",
            e.point
        );
    }
    let rep = s.report(&radix);
    assert!(rep.reused > 0);
    // the cache is priced: hits are never reported without their probes
    assert!(rep.cache_probes >= rep.reused);
}

// ── the radix the geometry supplies ──────────────────────────────────

#[test]
fn commensurate_geometry_is_exact_and_small() {
    let g = geometry_radix(&[0.5, 0.25, 0.125, 1.0 / 3.0], 4096).unwrap();
    assert!(g.exact);
    assert_eq!(g.max_path_error, 0.0);
    assert_eq!(g.radix.modulus(), 24);
    assert_eq!(g.radix.components(), &[(2, 3), (3, 1)]);
}

#[test]
fn best_rational_is_a_convergent() {
    let (n, d, err) = best_rational(std::f64::consts::PI, 200);
    assert_eq!((n, d), (355, 113));
    assert!(err < 1e-6);
    let (n, d, _) = best_rational(0.5, 100);
    assert_eq!((n, d), (1, 2));
    let (n, d, _) = best_rational(-0.25, 100);
    assert_eq!((n, d), (-1, 4));
}

#[test]
fn the_golden_ratio_is_the_worst_approximable_geometry() {
    // Hurwitz: err·q² ≥ 1/√5 infinitely often, with equality approached
    // only for φ. So φ's denominators must grow fastest — the
    // quasi-periodic pattern is the expensive one, measured.
    let phi = ((1.0 + 5f64.sqrt()) / 2.0).fract();
    let mut phi_q = Vec::new();
    for budget in [8u64, 32, 128, 512, 2048] {
        let g = geometry_radix(&[phi], budget).unwrap();
        let q = g.radix.modulus() as f64;
        phi_q.push(g.radix.modulus());
        let hurwitz = g.max_path_error * q * q;
        assert!(
            (hurwitz - 1.0 / 5f64.sqrt()).abs() < 0.02,
            "φ Hurwitz {hurwitz} at budget {budget}"
        );
        assert!(!g.exact);
        assert!(g.max_phase_error > 0.0);

        // every other tested irrational is approximated at least as well
        for x in [2f64.sqrt().fract(), std::f64::consts::PI.fract(), 3f64.sqrt().fract()] {
            let o = geometry_radix(&[x], budget).unwrap();
            let oq = o.radix.modulus() as f64;
            assert!(
                o.max_path_error * oq * oq <= hurwitz + 1e-9,
                "{x} beat φ at budget {budget}"
            );
        }
    }
    // the denominators are the Fibonacci numbers, growing without bound
    assert_eq!(phi_q, vec![8, 21, 89, 377, 1597]);
}

#[test]
fn a_modulus_beyond_the_bound_is_refused_not_truncated() {
    assert!(Radix::of(1).is_err());
    assert!(Radix::of(0).is_err());
    assert!(Radix::of(u64::MAX).is_err());
    assert!(Radix::from_denominators(&[]).is_err());
    assert!(Radix::from_denominators(&[0]).is_err());
    assert!(Radix::from_denominators(&[3, 5, 7]).is_ok());
}
