//! Regev's algorithm: the Zeckendorf digits, the three exponentiation
//! schedules and their agreement, and factoring end to end.

use quantsim::prelude::*;
use quantsim::regev::{
    self, coprime_primes, fibonacci, zeckendorf, zeckendorf_len, ExpPlan, ExpSchedule, Layout,
    Regev, Superposition,
};
use quantsim::shor;

const SCHEDULES: [ExpSchedule; 3] = [
    ExpSchedule::Sequential,
    ExpSchedule::Regev,
    ExpSchedule::Fibonacci,
];

#[test]
fn small_primes_skip_the_ones_that_divide_n() {
    let (primes, divisors) = coprime_primes(15, 3);
    assert_eq!(primes, vec![2, 7, 11]);
    assert_eq!(divisors, vec![3, 5]);
    for n in [15u64, 21, 33, 35, 51, 55, 91] {
        let (primes, divisors) = coprime_primes(n, 4);
        assert_eq!(primes.len(), 4);
        for p in &primes {
            assert_ne!(n % p, 0, "{p} divides {n} and was still used");
        }
        for d in &divisors {
            assert_eq!(n % d, 0, "{d} was skipped but does not divide {n}");
        }
    }
}

#[test]
fn zeckendorf_digits_reconstruct_the_value() {
    for bits in 1..=8usize {
        let k = zeckendorf_len(bits);
        let fibs = fibonacci(k + 1);
        // The stated bound: F_{K+1} ≥ 2^bits.
        assert!(fibs[k] >= 1u64 << bits, "F_{{{}}} too small", k + 1);
        for value in 0..(1u64 << bits) {
            let mask = zeckendorf(value, &fibs[..k]);
            let sum: u64 = (0..k)
                .filter(|&j| (mask >> j) & 1 == 1)
                .map(|j| fibs[j])
                .sum();
            assert_eq!(sum, value, "bits={bits} value={value} mask={mask:b}");
            assert_eq!(mask & 1, 0, "F₁ was used for {value}");
            assert_eq!(mask & (mask >> 1), 0, "adjacent digits for {value}");
        }
    }
}

#[test]
fn fibonacci_is_the_fibonacci_sequence() {
    let f = fibonacci(12);
    assert_eq!(f, vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]);
}

#[test]
fn every_schedule_computes_the_same_permutation_and_leaves_no_garbage() {
    let sim: Simulator = Simulator::new();
    for &(n, d, r) in &[(15u64, 2usize, 3usize), (21, 2, 3), (35, 2, 3), (33, 3, 2)] {
        let bases: Vec<u64> = coprime_primes(n, d).0.iter().map(|p| p * p % n).collect();
        let span = 1u64 << r;
        for schedule in SCHEDULES {
            let layout = Layout::new(schedule, d, r, shor::work_bits(n));
            assert!(layout.qubits <= regev::MAX_WIDTH, "{schedule} too wide");
            let plan = ExpPlan {
                modulus: n,
                bases: bases.clone(),
                exponent_bits: r,
                schedule,
                layout: layout.clone(),
            };
            for code in 0..span.pow(d as u32) {
                let mut state = sim.backends().create("sparse", layout.qubits).unwrap();
                let mut rest = code;
                let mut start = 0u64;
                let mut z = Vec::with_capacity(d);
                for reg in &layout.exponent {
                    let zi = rest % span;
                    rest /= span;
                    start = shor::scatter(start, reg, zi);
                    z.push(zi);
                }
                state.load(&[(start, C64::new(1.0, 0.0))]).unwrap();
                regev::exponentiate(&sim, state.as_mut(), &plan).unwrap();

                let mut got = None;
                state.for_each_nonzero(&mut |i, _| got = Some(i));
                let got = got.unwrap();
                let out = &layout.work[layout.output];
                let mut want = 1u64 % n;
                for (i, &b) in bases.iter().enumerate() {
                    want = shor::mul_mod(want, shor::pow_mod(b, z[i], n), n);
                }
                assert_eq!(shor::gather(got, out), want, "{schedule} N={n} z={z:?}");
                assert_eq!(
                    shor::scatter(got, out, 0),
                    start,
                    "{schedule} N={n} z={z:?} left garbage behind"
                );
            }
        }
    }
}

#[test]
fn the_measured_multiplication_counts_are_the_advertised_ones() {
    let sim: Simulator = Simulator::new();
    let (n, d, r) = (35u64, 2usize, 4usize);
    let bases: Vec<u64> = coprime_primes(n, d).0.iter().map(|p| p * p % n).collect();
    for schedule in SCHEDULES {
        let layout = Layout::new(schedule, d, r, shor::work_bits(n));
        let plan = ExpPlan {
            modulus: n,
            bases: bases.clone(),
            exponent_bits: r,
            schedule,
            layout: layout.clone(),
        };
        let mut state = sim.backends().create("sparse", layout.qubits).unwrap();
        state.load(&[(0, C64::new(1.0, 0.0))]).unwrap();
        let cost = regev::exponentiate(&sim, state.as_mut(), &plan).unwrap();
        let (full, small) = schedule.multiplications(d, r);
        assert_eq!(cost.full_multiplications, full, "{schedule} full");
        assert_eq!(cost.small_multiplications, small, "{schedule} small");
        assert_eq!(
            cost.work_registers,
            schedule.work_registers(r),
            "{schedule} regs"
        );
    }
    // And the table's shape: Regev's schedule buys full-width
    // multiplications only once d > 2, and Fibonacci pays 1.44× for them
    // to hold three registers instead of R + 2.
    let k = zeckendorf_len(r);
    assert_eq!(ExpSchedule::Sequential.multiplications(4, r).0, 4 * r);
    assert_eq!(ExpSchedule::Regev.multiplications(4, r).0, 2 * r);
    assert_eq!(ExpSchedule::Fibonacci.multiplications(4, r).0, 2 * k);
    // K = ⌈R/log₂φ⌉ + O(1): the ratio tends to 1.4404 from above,
    // because F_{K+1} ≥ 2^R carries an additive log_φ√5 ≈ 1.67.
    assert!(k > r);
    let ratio = |r: usize| zeckendorf_len(r) as f64 / r as f64;
    assert!(ratio(4) > ratio(16) && ratio(16) > ratio(48));
    for rr in 2..=48usize {
        let kk = zeckendorf_len(rr) as f64;
        let asymptote = rr as f64 * 2f64.log(1.618_033_988_749_895);
        assert!(
            kk >= asymptote && kk <= asymptote + 3.0,
            "R={rr}: K={kk} is not within O(1) of {asymptote}"
        );
    }
    assert_eq!(ExpSchedule::Regev.work_registers(r), r + 2);
    assert_eq!(ExpSchedule::Fibonacci.work_registers(r), 3);
}

#[test]
fn a_lattice_vector_is_a_square_root_of_one() {
    // The construction's whole point: bᵢ = pᵢ², so v ∈ L gives u² ≡ 1.
    for n in [15u64, 21, 33, 35, 55] {
        let primes = coprime_primes(n, 3).0;
        for v in [[1i64, -1, 0], [2, 0, -1], [-3, 1, 1]] {
            let u = regev::evaluate(&primes, &v, n).unwrap();
            let mut check = 1u64;
            for (i, &p) in primes.iter().enumerate() {
                let b = p * p % n;
                let e = v[i];
                let term = if e >= 0 {
                    shor::pow_mod(b, e as u64, n)
                } else {
                    shor::pow_mod(quantsim::padic::mod_inv(b, n).unwrap(), (-e) as u64, n)
                };
                check = shor::mul_mod(check, term, n);
            }
            assert_eq!(shor::mul_mod(u, u, n), check, "u² must be ∏bᵢ^vᵢ");
        }
    }
}

#[test]
fn a_witness_that_claims_a_square_root_really_is_one() {
    let sim: Simulator = Simulator::new();
    for n in [15u64, 21, 33, 35, 39, 51, 55] {
        for d in [2usize, 3] {
            let mut rng = Prng::new(4);
            let report = regev::factor(&sim, n, d, &mut rng).unwrap();
            let w = report
                .witness
                .unwrap_or_else(|| panic!("N={n} d={d} found no witness"));
            assert_eq!(w.factors.0 * w.factors.1, n);
            assert!(w.factors.0 > 1 && w.factors.1 > 1);
            if w.square_root_of_one {
                assert_eq!(shor::mul_mod(w.root, w.root, n), 1, "N={n}: u² ≠ 1");
                assert_ne!(w.root, 1);
                assert_ne!(w.root, n - 1);
            }
        }
    }
}

#[test]
fn factoring_end_to_end() {
    let sim: Simulator = Simulator::new();
    for n in [15u64, 21, 33, 35, 39, 51, 55] {
        for seed in 0..4u64 {
            let mut rng = Prng::new(seed);
            let report = regev::factor(&sim, n, 2, &mut rng).unwrap();
            let (p, q) = report
                .factors
                .unwrap_or_else(|| panic!("N={n} seed={seed} found nothing"));
            assert_eq!(p * q, n);
            assert_eq!(report.samples.len(), 6);
            assert!(report.cost.full_multiplications > 0);
        }
    }
}

#[test]
fn every_schedule_factors_end_to_end() {
    let sim: Simulator = Simulator::new();
    for schedule in SCHEDULES {
        for n in [15u64, 21] {
            let regev = Regev::new(n, 2)
                .unwrap()
                .with_exponent_bits(3)
                .with_schedule(schedule);
            assert!(regev.layout().qubits <= regev::MAX_WIDTH);
            let mut found = false;
            for seed in 0..4u64 {
                let mut rng = Prng::new(seed);
                let report = regev.run(&sim, &mut rng, 6).unwrap();
                if let Some((p, q)) = report.factors {
                    assert_eq!(p * q, n);
                    found = true;
                }
            }
            assert!(found, "{schedule} never factored {n}");
        }
    }
}

#[test]
fn the_gaussian_preparation_normalizes_and_still_factors() {
    let sim: Simulator = Simulator::new();
    let n = 35u64;
    let regev = Regev::new(n, 2)
        .unwrap()
        .with_superposition(Superposition::Gaussian(8.0));
    let mut found = false;
    for seed in 0..4u64 {
        let mut rng = Prng::new(seed);
        let report = regev.run(&sim, &mut rng, 6).unwrap();
        if let Some((p, q)) = report.factors {
            assert_eq!(p * q, n);
            found = true;
        }
    }
    assert!(found, "the Gaussian preparation never factored {n}");
}

#[test]
fn the_banded_transform_costs_fewer_gates_per_register() {
    let sim: Simulator = Simulator::new();
    let n = 35u64;
    let exact = Regev::new(n, 2).unwrap();
    let banded = Regev::new(n, 2)
        .unwrap()
        .with_qft_epsilon(std::f64::consts::PI / 8.0);
    let mut rng = Prng::new(6);
    let a = exact.sample(&sim, &mut rng).unwrap();
    let mut rng = Prng::new(6);
    let b = banded.sample(&sim, &mut rng).unwrap();
    assert!(
        b.cost.phase_gates < a.cost.phase_gates,
        "{} !< {}",
        b.cost.phase_gates,
        a.cost.phase_gates
    );
    // The saving is per exponent register, so it scales with d.
    assert_eq!(a.cost.phase_gates % 2, 0);
}

#[test]
fn a_modulus_too_small_for_its_dimension_is_refused_by_name() {
    let err = Regev::new(9, 8).unwrap_err().to_string();
    assert!(err.contains("coprime prime bases"), "unexpected: {err}");
}
