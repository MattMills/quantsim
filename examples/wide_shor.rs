//! Order finding past the `u64` basis index.
//!
//! Run with `cargo run --release --example wide_shor`.
//!
//! Every `Backend` in the crate indexes basis states by `u64`, capping a
//! register at 63 qubits. `shor`'s claim is that modular exponentiation
//! is a permutation — `O(support)` at any width — and that the support is
//! the multiplicative order `r`. Neither quantity mentions the modulus's
//! bit length, and on a `u64`-indexed register that is untestable past 63
//! qubits because the register cannot address the modulus.
//! [`quantsim::wide`] supplies the index type instead, and this runs the
//! same algorithm at four figures of qubits.

use std::time::Instant;

use quantsim::prelude::*;
use quantsim::shor::{self, OrderFinder, PhaseForm, WideOrderFinder};
use quantsim::wide::{Montgomery, Wide};

/// `N` = product of `k` distinct primes just below `2^31`, with a known
/// multiple of `λ(N)` and that multiple's prime factors.
fn build(k: usize) -> (Wide, Wide, Vec<u64>) {
    let mut primes = Vec::new();
    let mut c = (1u64 << 31) - 1;
    while primes.len() < k {
        if shor::is_prime(c) {
            primes.push(c);
        }
        c -= 2;
    }
    let mut n = Wide::one();
    let mut multiple = Wide::one();
    let mut factors: Vec<u64> = Vec::new();
    for &p in &primes {
        n = n.mul(&Wide::from_u64(p));
        multiple = multiple.mul(&Wide::from_u64(p - 1));
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
    (n, multiple, factors)
}

/// Exact order of `g`, by dividing a known multiple down by its primes.
fn exact_order(mont: &Montgomery, g: &Wide, multiple: &Wide, primes: &[u64]) -> Wide {
    let mut r = multiple.clone();
    for &p in primes {
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
}

fn bytes(b: usize) -> String {
    match b {
        v if v < 10_000 => format!("{v}B"),
        v if v < 10_000_000 => format!("{:.1}KB", v as f64 / 1e3),
        v => format!("{:.1}MB", v as f64 / 1e6),
    }
}

fn main() -> Result<()> {
    let sim: Simulator = Simulator::new();

    println!("== the wide path against the u64 path, same seeds ==");
    println!(
        "  {:>8} {:>4} {:>7}  {:>16}  {:>16}",
        "N", "a", "r", "u64 (value/support)", "wide"
    );
    for (n, a) in [
        (35u64, 2u64),
        (221, 3),
        (899, 5),
        (4087, 7),
        (32399, 3),
        (126727, 2),
    ] {
        let f = OrderFinder::new(n, a)?;
        let wf = WideOrderFinder::new(Wide::from_u64(n), Wide::from_u64(a))?
            .with_phase_bits(f.phase_bits);
        let mut agree = 0;
        let (mut sv, mut ss, mut wv, mut ws) = (0, 0, 0, 0);
        for seed in 0..16u64 {
            let e = f.estimate(&sim, PhaseForm::Semiclassical, &mut Prng::new(seed))?;
            let we = wf.estimate::<C64>(&mut Prng::new(seed))?;
            if e.value == we.value && e.peak_support == we.peak_support {
                agree += 1;
            }
            if seed == 0 {
                sv = e.value;
                ss = e.peak_support;
                wv = we.value;
                ws = we.peak_support;
            }
        }
        let r = shor::multiplicative_order(a, n).unwrap();
        println!(
            "  {n:>8} {a:>4} {r:>7}  {:>16}  {:>16}   {agree}/16 identical",
            format!("{sv}/{ss}"),
            format!("{wv}/{ws}")
        );
        assert_eq!(agree, 16);
    }

    println!();
    println!("== order finding past 63 qubits ==");
    println!(
        "  {:>8} {:>7} {:>6} {:>7} {:>11} {:>9} {:>10} {:>10}",
        "N bits", "qubits", "limbs", "r", "time", "support", "bytes", "recovered"
    );
    for k in [3usize, 5, 9, 17, 33, 67, 133] {
        let (n, multiple, factors) = build(k);
        let mont = Montgomery::new(n.clone())?;
        let g = Wide::from_u64(3);
        let full = exact_order(&mont, &g, &multiple, &factors);
        // Walk the order down to a small power of two: the support stays
        // holdable while the width does not.
        let mut exponent = full.clone();
        let mut target = Wide::one();
        for _ in 0..8 {
            let (q, rem) = exponent.div_small(2);
            if rem != 0 {
                break;
            }
            exponent = q;
            target = target.double();
        }
        let a = mont.pow(&g, &exponent);
        let r = exact_order(&mont, &a, &full, &factors)
            .to_u64()
            .unwrap_or(0);
        let wf = WideOrderFinder::new(n.clone(), a)?;
        let t = Instant::now();
        let est = wf.estimate::<C64>(&mut Prng::new(3))?;
        let elapsed = t.elapsed().as_nanos() as u64;
        let recovered = wf.order_from_phase(est.phase, 1 << 20);
        println!(
            "  {:>8} {:>7} {:>6} {:>7} {:>10.2}ms {:>9} {:>10} {:>10}",
            n.bits(),
            est.qubits,
            n.limbs().len(),
            r,
            elapsed as f64 / 1e6,
            est.peak_support,
            bytes(est.peak_bytes),
            recovered
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".into())
        );
        assert_eq!(est.peak_orbit as u64, r);
        assert_eq!(target.to_u64(), Some(r));
    }

    println!();
    println!("== the arithmetic alone, one basis state, no interference ==");
    println!(
        "  {:>8} {:>7} {:>7} {:>11} {:>9}",
        "N bits", "qubits", "support", "time/mul", "bytes"
    );
    for k in [17usize, 67, 133, 267] {
        let (n, _, _) = build(k);
        let mont = Montgomery::new(n.clone())?;
        let w = n.bits();
        let mut reg: quantsim::wide::WideRegister<C64> = quantsim::wide::WideRegister::new(w);
        reg.load(vec![(mont.one(), C64::new(1.0, 0.0))])?;
        let m = mont.to_montgomery(&Wide::from_u64(3));
        let modulus = n.clone();
        let t = Instant::now();
        let rounds = 64;
        for _ in 0..rounds {
            reg.apply_permutation("modular multiplication", &|i: &Wide| {
                let y = i.low_bits(w);
                if y >= modulus {
                    return i.clone();
                }
                i.with_low_bits(w, &mont.mul(&y, &m))
            })?;
        }
        println!(
            "  {:>8} {:>7} {:>7} {:>10.1}µs {:>9}",
            n.bits(),
            reg.qubits(),
            reg.nonzero_count(),
            t.elapsed().as_nanos() as f64 / 1e3 / rounds as f64,
            bytes(reg.memory_bytes())
        );
    }
    Ok(())
}
