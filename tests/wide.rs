//! Registers past the `u64` basis index: the arithmetic, the register,
//! and order finding at four figures of qubits.

use quantsim::prelude::*;
use quantsim::shor::{self, OrderFinder, PhaseForm, WideOrderFinder};
use quantsim::wide::{Montgomery, Wide, WideRegister};

fn w(v: u64) -> Wide {
    Wide::from_u64(v)
}

#[test]
fn wide_arithmetic_matches_u128() {
    let mut rng = Prng::new(5);
    for _ in 0..20_000 {
        let a = rng.next_u64() as u128 * rng.next_u64() as u128;
        let b = rng.next_u64() as u128 * rng.next_u64() as u128;
        let wa = Wide::from_limbs(vec![a as u64, (a >> 64) as u64]);
        let wb = Wide::from_limbs(vec![b as u64, (b >> 64) as u64]);
        assert_eq!(wa.cmp(&wb), a.cmp(&b));
        let sum = wa.add(&wb);
        assert_eq!(
            sum.limbs().first().copied().unwrap_or(0),
            a.wrapping_add(b) as u64
        );
        if a >= b {
            let d = wa.sub(&wb);
            assert_eq!(d.limbs().first().copied().unwrap_or(0), (a - b) as u64);
            assert_eq!(d.add(&wb), wa, "sub then add is not the identity");
        }
        assert_eq!(wa.double(), wa.add(&wa));
    }
    // Multiplication against u128 where the product fits.
    for _ in 0..20_000 {
        let a = rng.next_u64() >> 1;
        let b = rng.next_u64() >> 1;
        let p = a as u128 * b as u128;
        let got = w(a).mul(&w(b));
        assert_eq!(got.limbs().first().copied().unwrap_or(0), p as u64);
        assert_eq!(got.limbs().get(1).copied().unwrap_or(0), (p >> 64) as u64);
    }
}

#[test]
fn division_by_a_small_divisor_matches_u128() {
    let mut rng = Prng::new(9);
    for _ in 0..20_000 {
        let a = rng.next_u64() as u128 * rng.next_u64() as u128;
        let d = (rng.next_u64() % 1000) + 1;
        let wa = Wide::from_limbs(vec![a as u64, (a >> 64) as u64]);
        let (q, r) = wa.div_small(d);
        assert_eq!(r as u128, a % d as u128);
        // q·d + r == a, exactly.
        assert_eq!(q.mul(&w(d)).add(&w(r)), wa);
    }
}

#[test]
fn bit_access_round_trips() {
    let mut rng = Prng::new(11);
    for _ in 0..2000 {
        let limbs: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        let v = Wide::from_limbs(limbs);
        for i in [0usize, 1, 63, 64, 65, 127, 200, 255] {
            let before = v.bit(i);
            let mut flipped = v.clone();
            flipped.set_bit(i, !before);
            assert_eq!(flipped.bit(i), !before);
            flipped.set_bit(i, before);
            assert_eq!(flipped, v, "set_bit did not round trip at {i}");
        }
        for n in [1usize, 7, 64, 65, 130] {
            let low = v.low_bits(n);
            assert!(low.bits() <= n);
            for i in 0..n {
                assert_eq!(low.bit(i), v.bit(i));
            }
            // Replacing the low bits with what was there is the identity.
            assert_eq!(v.with_low_bits(n, &low), v);
        }
    }
}

#[test]
fn montgomery_matches_u128_and_refuses_an_even_modulus() {
    let mut rng = Prng::new(13);
    for _ in 0..3000 {
        let n = (rng.next_u64() % (1u64 << 40)) | 1;
        if n < 3 {
            continue;
        }
        let m = Montgomery::new(w(n)).unwrap();
        let a = rng.next_u64() % n;
        let b = rng.next_u64() % n;
        let got = m.from_montgomery(&m.mul(&m.to_montgomery(&w(a)), &m.to_montgomery(&w(b))));
        assert_eq!(
            got.to_u64(),
            Some(((a as u128 * b as u128) % n as u128) as u64)
        );
        let e = rng.next_u64() % 4096;
        assert_eq!(m.pow(&w(a), &w(e)).to_u64(), Some(shor::pow_mod(a, e, n)));
        // The representative of 1 really is R mod N.
        assert_eq!(m.from_montgomery(&m.one()), Wide::one());
    }
    let err = Montgomery::new(w(30)).unwrap_err().to_string();
    assert!(err.contains("odd modulus"), "unexpected: {err}");
    assert!(Montgomery::new(Wide::one()).is_err());
}

#[test]
fn the_carmichael_identity_holds_at_a_thousand_bits() {
    let mut primes: Vec<u64> = Vec::new();
    let mut c = (1u64 << 31) - 1;
    while primes.len() < 33 {
        if shor::is_prime(c) {
            primes.push(c);
        }
        c -= 2;
    }
    let mut n = Wide::one();
    let mut multiple = Wide::one();
    for &p in &primes {
        n = n.mul(&w(p));
        multiple = multiple.mul(&w(p - 1));
    }
    assert!(n.bits() > 1000, "modulus is only {} bits", n.bits());
    let m = Montgomery::new(n).unwrap();
    for base in [3u64, 5, 7, 11, 13] {
        assert_eq!(m.pow(&w(base), &multiple), Wide::one(), "base {base}");
    }
}

#[test]
fn a_non_injective_map_is_refused() {
    let mut reg: WideRegister<C64> = WideRegister::new(8);
    reg.load(vec![(w(1), C64::new(1.0, 0.0)), (w(2), C64::new(1.0, 0.0))])
        .unwrap();
    let err = reg
        .apply_permutation("collapse", &|_| Wide::zero())
        .unwrap_err()
        .to_string();
    assert!(err.contains("not injective"), "unexpected: {err}");
}

#[test]
fn a_one_qubit_gate_matches_the_dense_backend() {
    // The same Hadamard layer on both, amplitude for amplitude.
    let sim: Simulator = Simulator::new();
    let n = 6usize;
    let mut dense = sim.backends().create("dense", n).unwrap();
    let mut wide: WideRegister<C64> = WideRegister::new(n);
    let k = std::f64::consts::FRAC_1_SQRT_2;
    let h = quantsim::math::GateMatrix::<C64>::from_vec(
        2,
        vec![
            C64::new(k, 0.0),
            C64::new(k, 0.0),
            C64::new(k, 0.0),
            C64::new(-k, 0.0),
        ],
    )
    .unwrap();
    for q in [0usize, 2, 5, 3] {
        sim.apply(dense.as_mut(), "h", &[], &[q]).unwrap();
        wide.apply_1q(&h, q).unwrap();
    }
    assert_eq!(wide.nonzero_count(), dense.nonzero_count());
    for i in 0..(1u64 << n) {
        let a = dense.amplitude(i);
        let b = wide.amplitude(&w(i));
        assert!(
            (a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12,
            "index {i}: {a:?} vs {b:?}"
        );
    }
}

#[test]
fn measurement_collapses_and_renormalizes() {
    let mut reg: WideRegister<C64> = WideRegister::new(4);
    let k = 0.5;
    reg.load(vec![
        (w(0b0000), C64::new(k, 0.0)),
        (w(0b0001), C64::new(k, 0.0)),
        (w(0b0010), C64::new(k, 0.0)),
        (w(0b0011), C64::new(k, 0.0)),
    ])
    .unwrap();
    let mut rng = Prng::new(2);
    let bit = reg.measure(0, &mut rng).unwrap();
    assert_eq!(reg.nonzero_count(), 2);
    for (i, _) in reg.entries() {
        assert_eq!(i.bit(0), bit);
    }
    assert!((reg.total_weight() - 1.0).abs() < 1e-12, "not renormalized");
}

#[test]
fn the_wide_path_reproduces_the_u64_path_exactly() {
    let sim: Simulator = Simulator::new();
    for (n, a) in [(35u64, 2u64), (221, 3), (899, 5), (4087, 7), (32399, 3)] {
        let f = OrderFinder::new(n, a).unwrap();
        let wf = WideOrderFinder::new(w(n), w(a))
            .unwrap()
            .with_phase_bits(f.phase_bits);
        assert_eq!(wf.work_bits(), f.work_bits());
        assert_eq!(wf.width(), f.width(PhaseForm::Semiclassical));
        for seed in 0..8u64 {
            let e = f
                .estimate(&sim, PhaseForm::Semiclassical, &mut Prng::new(seed))
                .unwrap();
            let we = wf.estimate::<C64>(&mut Prng::new(seed)).unwrap();
            assert_eq!(e.value, we.value, "N={n} a={a} seed={seed}");
            assert_eq!(e.bits, we.bits);
            assert_eq!(e.peak_support, we.peak_support);
            assert_eq!(e.peak_orbit, we.peak_orbit);
            assert_eq!(e.orbit_trajectory, we.orbit_trajectory);
        }
    }
}

#[test]
fn order_finding_runs_past_a_thousand_qubits() {
    // A modulus far past any u64 basis index, with an order small enough
    // to hold: the whole point is that those are independent.
    let mut primes: Vec<u64> = Vec::new();
    let mut c = (1u64 << 31) - 1;
    while primes.len() < 33 {
        if shor::is_prime(c) {
            primes.push(c);
        }
        c -= 2;
    }
    let (mut n, mut multiple) = (Wide::one(), Wide::one());
    let mut factors: Vec<u64> = Vec::new();
    for &p in &primes {
        n = n.mul(&w(p));
        multiple = multiple.mul(&w(p - 1));
        let mut x = p - 1;
        let mut d = 2u64;
        while d * d <= x {
            if x % d == 0 {
                while x % d == 0 {
                    x /= d;
                }
                factors.push(d);
            }
            d += if d == 2 { 1 } else { 2 };
        }
        if x > 1 {
            factors.push(x);
        }
    }
    factors.sort_unstable();
    factors.dedup();
    let mont = Montgomery::new(n.clone()).unwrap();
    let order_of = |g: &Wide, bound: &Wide| {
        let mut r = bound.clone();
        for &p in &factors {
            loop {
                let (q, rem) = r.div_small(p);
                if rem != 0 {
                    break;
                }
                if mont.pow(g, &q) == Wide::one() {
                    r = q;
                } else {
                    break;
                }
            }
        }
        r
    };
    let full = order_of(&w(3), &multiple);
    let mut exponent = full.clone();
    let mut target = Wide::one();
    for _ in 0..6 {
        let (q, rem) = exponent.div_small(2);
        if rem != 0 {
            break;
        }
        exponent = q;
        target = target.double();
    }
    let a = mont.pow(&w(3), &exponent);
    let r = order_of(&a, &full).to_u64().expect("order fits u64");
    assert_eq!(Some(r), target.to_u64());
    assert!(r > 1, "the constructed base is trivial");

    let wf = WideOrderFinder::new(n.clone(), a).unwrap();
    assert!(wf.width() > 1000, "only {} qubits", wf.width());
    let est = wf.estimate::<C64>(&mut Prng::new(3)).unwrap();
    assert_eq!(est.qubits, n.bits() + 1);
    // The support is the orbit, and the orbit is r — at a thousand qubits.
    assert_eq!(est.peak_orbit as u64, r);
    assert!(est.peak_support as u64 <= 2 * r);
    assert_eq!(wf.order_from_phase(est.phase, 1 << 20), Some(r));
}

#[test]
fn the_arithmetic_alone_is_free_of_the_width() {
    // One basis state, modular multiplication, at widths a u64 index
    // cannot address: the support stays 1 and the register stays tiny.
    for k in [17usize, 67] {
        let mut primes: Vec<u64> = Vec::new();
        let mut c = (1u64 << 31) - 1;
        while primes.len() < k {
            if shor::is_prime(c) {
                primes.push(c);
            }
            c -= 2;
        }
        let mut n = Wide::one();
        for &p in &primes {
            n = n.mul(&w(p));
        }
        let width = n.bits();
        assert!(width > 63, "not past the u64 index");
        let mont = Montgomery::new(n.clone()).unwrap();
        let mut reg: WideRegister<C64> = WideRegister::new(width);
        reg.load(vec![(mont.one(), C64::new(1.0, 0.0))]).unwrap();
        let m = mont.to_montgomery(&w(3));
        for _ in 0..32 {
            reg.apply_permutation("mul", &|i: &Wide| {
                let y = i.low_bits(width);
                if y >= n {
                    return i.clone();
                }
                i.with_low_bits(width, &mont.mul(&y, &m))
            })
            .unwrap();
            assert_eq!(reg.nonzero_count(), 1, "a permutation changed the support");
        }
        // 32 multiplications by 3 leaves 3^32 in Montgomery form.
        let held = reg.entries().next().unwrap().0.clone();
        assert_eq!(
            mont.from_montgomery(&held),
            mont.pow(&w(3), &w(32)),
            "wrong residue at width {width}"
        );
    }
}
