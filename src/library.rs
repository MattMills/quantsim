//! A small library of named circuit constructions, used by the tests,
//! benchmarks and examples — and as executable documentation of the crate's
//! conventions.
//!
//! Constructions are generic over the amplitude algebra; whether a given
//! circuit *binds* over an algebra depends on which gates the registry could
//! install there (e.g. [`qft`] needs `cp`, so it binds over ℂ/ℍ/𝕆 but not ℝ,
//! while [`ghz`] binds everywhere).

use crate::circuit::Circuit;
use crate::error::Result;
use crate::math::GateMatrix;
use crate::rng::Prng;
use crate::scalar::Scalar;

/// Bell pair preparation: `H(0); CX(0, 1)` giving `(|00⟩ + |11⟩)/√2`.
pub fn bell<S: Scalar>() -> Circuit<S> {
    let mut c = Circuit::new(2);
    c.h(0).cx(0, 1);
    c
}

/// GHZ state on `n ≥ 1` qubits: `(|0…0⟩ + |1…1⟩)/√2`.
pub fn ghz<S: Scalar>(n: usize) -> Circuit<S> {
    assert!(n >= 1, "GHZ needs at least one qubit");
    let mut c = Circuit::new(n);
    c.h(0);
    for q in 1..n {
        c.cx(q - 1, q);
    }
    c
}

/// Quantum Fourier transform on `n` qubits (little-endian):
/// `|x⟩ → 2^{-n/2} Σ_y e^{2πi x y / 2^n} |y⟩`, including the final qubit
/// reversal swaps. Invert with [`crate::circuit::BoundCircuit::inverse`].
pub fn qft<S: Scalar>(n: usize) -> Circuit<S> {
    let mut c = Circuit::new(n);
    for j in (0..n).rev() {
        c.h(j);
        for m in (0..j).rev() {
            let angle = std::f64::consts::PI / (1u64 << (j - m)) as f64;
            c.cp(m, j, angle);
        }
    }
    for k in 0..n / 2 {
        c.swap(k, n - 1 - k);
    }
    c
}

/// A multi-controlled Z on all `n` qubits as a raw diagonal matrix
/// (`diag(1, …, 1, −1)`); real-valued, so it exists over every algebra.
pub fn mcz_matrix<S: Scalar>(n: usize) -> Result<GateMatrix<S>> {
    let dim = 1usize << n;
    let mut m = GateMatrix::identity(dim)?;
    m.set(dim - 1, dim - 1, -S::one());
    Ok(m)
}

/// Grover search on `n` qubits for the basis state `marked`, running
/// `iterations` rounds of oracle + diffusion. The optimal iteration count is
/// roughly `π/4 · √(2^n)`.
pub fn grover<S: Scalar>(n: usize, marked: u64, iterations: usize) -> Result<Circuit<S>> {
    assert!(n >= 1 && n <= 32, "grover: unreasonable width");
    assert!(marked < (1u64 << n), "grover: marked state out of range");
    let mcz = mcz_matrix::<S>(n)?;
    let all: Vec<usize> = (0..n).collect();
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    for _ in 0..iterations {
        // Oracle: phase-flip |marked⟩ (conjugate an all-ones MCZ by X on the
        // zero bits of the marked pattern).
        for q in 0..n {
            if (marked >> q) & 1 == 0 {
                c.x(q);
            }
        }
        c.raw("mcz", mcz.clone(), all.clone());
        for q in 0..n {
            if (marked >> q) & 1 == 0 {
                c.x(q);
            }
        }
        // Diffusion: reflect about the uniform superposition.
        for q in 0..n {
            c.h(q);
        }
        for q in 0..n {
            c.x(q);
        }
        c.raw("mcz", mcz.clone(), all.clone());
        for q in 0..n {
            c.x(q);
        }
        for q in 0..n {
            c.h(q);
        }
    }
    Ok(c)
}

/// A deterministic pseudo-random circuit over a hardware-ish gate pool,
/// intended for benchmarks and cross-backend agreement tests. Uses the
/// full standard library, so it binds over algebras containing ℂ.
pub fn random_circuit<S: Scalar>(n: usize, gates: usize, seed: u64) -> Circuit<S> {
    assert!(n >= 2, "random_circuit needs at least two qubits");
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    let tau = std::f64::consts::TAU;
    for _ in 0..gates {
        let mut pick_q = || (rng.next_u64() % n as u64) as usize;
        let q0 = pick_q();
        let mut q1 = pick_q();
        while q1 == q0 {
            q1 = pick_q();
        }
        match rng.next_u64() % 12 {
            0 => c.h(q0),
            1 => c.x(q0),
            2 => c.t(q0),
            3 => c.s(q0),
            4 => c.sx(q0),
            5 => c.rx(q0, rng.next_f64() * tau),
            6 => c.ry(q0, rng.next_f64() * tau),
            7 => c.rz(q0, rng.next_f64() * tau),
            8 => c.cx(q0, q1),
            9 => c.cz(q0, q1),
            10 => c.cp(q0, q1, rng.next_f64() * tau),
            _ => c.swap(q0, q1),
        };
    }
    c
}
