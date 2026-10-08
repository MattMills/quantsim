//! The D[ζ_N] evaluator: exact past Clifford+T, equal to the D[ω]
//! evaluator at N = 8, and the QFT equal to its closed form.

mod common;

use om_core::cyclotomic::Cyclo;
use quantsim::cyclotomic::{CyclotomicState, DZeta};
use quantsim::exact::ExactState;
use quantsim::library::{iqft, qft};
use quantsim::prelude::*;
use std::f64::consts::PI;

/// A random circuit over the gates and angles exact in D[ζ_N]: phases at
/// multiples of 2π/N, half-angle rotations at multiples of 4π/N.
fn random_circuit<const N: u32>(n: usize, gates: usize, seed: u64) -> Circuit {
    let mut rng = Prng::new(seed);
    let step = 2.0 * PI / f64::from(N);
    let mut c = Circuit::new(n);
    for _ in 0..gates {
        let q = (rng.next_u64() % n as u64) as usize;
        let r = (q + 1 + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
        let m = (rng.next_u64() % u64::from(N)) as f64;
        match rng.next_u64() % 14 {
            0 => c.h(q),
            1 => c.t(q),
            2 => c.sx(q),
            3 => c.y(q),
            4 => c.cx(q, r),
            5 => c.cz(q, r),
            6 => c.swap(q, r),
            7 => c.p(q, m * step),
            8 => c.rz(q, 2.0 * m * step),
            9 => c.rx(q, 2.0 * m * step),
            10 => c.ry(q, 2.0 * m * step),
            11 => c.cp(q, r, m * step),
            12 => c.crz(q, r, 2.0 * m * step),
            _ => c.u(q, 2.0 * m * step, m * step, 3.0 * m * step),
        };
    }
    c
}

#[test]
fn equals_the_d_omega_evaluator_at_n_8() {
    for seed in 0..40 {
        let c = random_circuit::<8>(4, 60, seed);
        let a = ExactState::run(&c).unwrap();
        let b = CyclotomicState::<8>::run(&c).unwrap();
        for i in 0..16 {
            let (dc, dk) = a.amplitude_exact(i).parts();
            let z = b.amplitude_exact(i);
            assert_eq!(z.parts(), (&dc[..], dk), "seed {seed}, index {i}");
        }
    }
}

fn unitary_and_close_to_dense<const N: u32>(seeds: u64) {
    for seed in 0..seeds {
        let c = random_circuit::<N>(4, 50, seed);
        let exact = CyclotomicState::<N>::run(&c).unwrap();
        assert_eq!(
            exact.total_weight_exact().unwrap(),
            DZeta::int(1),
            "N = {N}, seed {seed}"
        );
        assert!(exact.max_deviation_vs(common::run_dense(&c).as_ref()) < 1e-12);
    }
}

#[test]
fn finer_phases_are_exact_and_unitary() {
    unitary_and_close_to_dense::<16>(20);
    unitary_and_close_to_dense::<64>(10);
    unitary_and_close_to_dense::<256>(4);
}

#[test]
fn qft_is_the_dft_exactly() {
    fn check<const N: u32>(n: usize) {
        let dim = 1u64 << n;
        for x in 0..dim {
            let mut c = Circuit::new(n);
            for q in (0..n).filter(|q| x >> q & 1 == 1) {
                c.x(q);
            }
            c.append(&qft(n), &(0..n).collect::<Vec<_>>());
            let s = CyclotomicState::<N>::run(&c).unwrap();
            // 2^{-n/2} ζ_{2^n}^{xy}, with ζ_{2^n} = ζ_N^{N/2^n}.
            let scale = DZeta::<N>::inv_sqrt2_pow(n as u32);
            for y in 0..dim {
                let e = (x * y % dim) * (u64::from(N) / dim);
                let want = DZeta::<N>::zeta_pow(e as i64).mul(&scale).unwrap();
                assert_eq!(s.amplitude_exact(y), want, "n = {n}, x = {x}, y = {y}");
            }
            assert!(s.max_deviation_vs(common::run_dense(&c).as_ref()) < 1e-12);
        }
    }
    check::<8>(2);
    check::<8>(3);
    check::<16>(4);
    check::<32>(5);
    check::<64>(6);
}

#[test]
fn qft_then_inverse_returns_the_input_exactly() {
    let n = 7;
    let mut rng = Prng::new(9);
    for _ in 0..4 {
        let x = rng.next_u64() % (1 << n);
        let mut c = Circuit::new(n);
        for q in (0..n).filter(|q| x >> q & 1 == 1) {
            c.x(q);
        }
        let all: Vec<usize> = (0..n).collect();
        c.append(&qft(n), &all).append(&iqft(n), &all);
        let s = CyclotomicState::<128>::run(&c).unwrap();
        assert_eq!(s.support_exact(), 1);
        assert_eq!(s.amplitude_exact(x), DZeta::int(1));
    }
}

#[test]
fn sqrt_t_is_exact_where_d_omega_refuses_it() {
    let mut c = Circuit::new(1);
    c.p(0, PI / 8.0);
    assert!(matches!(
        ExactState::run(&c),
        Err(Error::UnsupportedForAlgebra { .. })
    ));
    // √T · √T = T, and H (√T)^16 H = I, exactly.
    let mut sqrt_t_twice = Circuit::new(1);
    sqrt_t_twice.h(0).p(0, PI / 8.0).p(0, PI / 8.0);
    let mut t = Circuit::new(1);
    t.h(0).t(0);
    let (a, b) = (
        CyclotomicState::<16>::run(&sqrt_t_twice).unwrap(),
        CyclotomicState::<16>::run(&t).unwrap(),
    );
    assert_eq!(a.amplitude_exact(1), b.amplitude_exact(1));
    let mut cycle = Circuit::new(1);
    cycle.h(0);
    for _ in 0..16 {
        cycle.p(0, PI / 8.0);
    }
    cycle.h(0);
    let s = CyclotomicState::<16>::run(&cycle).unwrap();
    assert_eq!(s.amplitude_exact(0), DZeta::int(1));
    assert!(s.amplitude_exact(1).is_zero());
}

#[test]
fn off_grid_angles_and_raw_matrices_are_refused() {
    let mut c = Circuit::new(1);
    c.p(0, 0.1);
    assert!(matches!(
        CyclotomicState::<64>::run(&c),
        Err(Error::UnsupportedForAlgebra { .. })
    ));
    let mut c = Circuit::new(1);
    c.p(0, PI / 64.0);
    assert!(CyclotomicState::<64>::run(&c).is_err());
    assert!(CyclotomicState::<128>::run(&c).is_ok());
}

/// OM's `Cyclo<N>` is an independent implementation of `ℤ[ζ_N]` in the
/// same power basis: products and conjugates must agree coefficient for
/// coefficient.
fn ring_agrees_with_om<const N: u32>() {
    let to_om = |c: &[i128]| {
        c.iter()
            .enumerate()
            .fold(Cyclo::<N>::zero(), |acc, (j, &x)| {
                acc.add(&Cyclo::zeta_pow(j as i64).scale(x))
            })
    };
    let mut rng = Prng::new(u64::from(N));
    let mut random = || {
        let c: Vec<i128> = (0..N / 2)
            .map(|_| (rng.next_u64() % 21) as i128 - 10)
            .collect();
        DZeta::<N>::from_parts(&c, 0).unwrap()
    };
    for _ in 0..200 {
        let (a, b) = (random(), random());
        let ab = a.mul(&b).unwrap();
        let om_ab = to_om(a.parts().0).mul(&to_om(b.parts().0));
        assert_eq!(ab.parts().0, om_ab.coefficients());
        assert_eq!(a.conj().parts().0, to_om(a.parts().0).conj().coefficients());
    }
}

#[test]
fn ring_agrees_with_opposed_mathematics() {
    ring_agrees_with_om::<8>();
    ring_agrees_with_om::<16>();
    ring_agrees_with_om::<64>();
}
