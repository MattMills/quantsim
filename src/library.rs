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

/// Brickwork light-cone circuit: `H` on each qubit in `seeds`, then
/// `depth` alternating layers of adjacent `CX(q, q+1)` — even-`q` pairs
/// on even layers, odd-`q` pairs on odd layers. All-Clifford.
///
/// This is the canonical *causally local* workload: each layer is a set
/// of disjoint nearest-neighbour gates, so information spreads at most
/// one site per layer and the causal cone of the seed set widens
/// linearly with depth. Everything outside the cone stays exactly |0⟩.
pub fn brickwork<S: Scalar>(n: usize, depth: usize, seeds: &[usize]) -> Circuit<S> {
    let mut c = Circuit::new(n);
    for &s in seeds {
        c.h(s);
    }
    for layer in 0..depth {
        let start = layer % 2;
        let mut q = start;
        while q + 1 < n {
            c.cx(q, q + 1);
            q += 2;
        }
    }
    c
}

/// A layer of `n/2` disjoint Bell pairs, each spanning interaction
/// distance `d`: `H(a); CX(a, a+d)` for every pair. Requires
/// `n % (2·d) == 0`; pairs tile in blocks of `2d`
/// (`(b·2d + j, b·2d + j + d)` for `j < d`), so every qubit belongs to
/// exactly one pair. All-Clifford.
///
/// The family holds the gate count, arity profile and output state
/// *shape* (a product of `n/2` Bell pairs) fixed while varying only the
/// causal range `d` — the knob for measuring how each representation
/// and each device geometry prices interaction distance.
pub fn ranged_pairs<S: Scalar>(n: usize, d: usize) -> Circuit<S> {
    assert!(d >= 1 && n % (2 * d) == 0, "pairs at range d must tile n");
    let mut c = Circuit::new(n);
    for block in 0..n / (2 * d) {
        for j in 0..d {
            let a = block * 2 * d + j;
            c.h(a).cx(a, a + d);
        }
    }
    c
}

/// The rainbow state on even `n`: `H(i); CX(i, n−1−i)` for `i < n/2` —
/// `n/2` disjoint Bell pairs nested around the centre, at interaction
/// distances `n−1, n−3, …, 1`. All-Clifford.
///
/// Every pair crosses the central cut, so any representation that pays
/// per *linear or hierarchical cut* (MPS bonds, mera's root) faces rank
/// `2^{n/2}` there, while the state remains a product of pairs that a
/// clustering representation stores in `O(n)`.
pub fn rainbow<S: Scalar>(n: usize) -> Circuit<S> {
    assert!(n >= 2 && n % 2 == 0, "rainbow needs even n");
    let mut c = Circuit::new(n);
    for i in 0..n / 2 {
        c.h(i).cx(i, n - 1 - i);
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

/// Inverse quantum Fourier transform on `n` qubits: the exact op-by-op
/// reversal of [`qft`] with negated phases.
pub fn iqft<S: Scalar>(n: usize) -> Circuit<S> {
    let mut c = Circuit::new(n);
    for k in (0..n / 2).rev() {
        c.swap(k, n - 1 - k);
    }
    for j in 0..n {
        for m in 0..j {
            let angle = -std::f64::consts::PI / (1u64 << (j - m)) as f64;
            c.cp(m, j, angle);
        }
        c.h(j);
    }
    c
}

/// A multi-controlled Z on all `n` qubits as a dense matrix
/// (`diag(1, …, 1, −1)`); real-valued, so it exists over every algebra.
/// Costs `O(4^n)` storage — for anything beyond a handful of qubits use
/// [`phase_flip`] with [`Circuit::diagonal`] instead.
pub fn mcz_matrix<S: Scalar>(n: usize) -> Result<GateMatrix<S>> {
    let dim = 1usize << n;
    let mut m = GateMatrix::identity(dim)?;
    m.set(dim - 1, dim - 1, -S::one());
    Ok(m)
}

/// Diagonal entries of an `n`-qubit phase flip about basis state `index`:
/// identity except `−1` at `index`. `O(2^n)` storage; real-valued, so it
/// exists over every algebra. Feed to [`Circuit::diagonal`].
pub fn phase_flip<S: Scalar>(n: usize, index: u64) -> Vec<S> {
    let dim = 1usize << n;
    assert!(index < dim as u64, "phase_flip: index out of range");
    let mut d = vec![S::one(); dim];
    d[index as usize] = -S::one();
    d
}

/// Grover search on `n` qubits for the basis state `marked`, running
/// `iterations` rounds of oracle + diffusion. The optimal iteration count is
/// roughly `π/4 · √(2^n)`.
///
/// The oracle is a diagonal phase flip at `marked`; the diffusion is
/// `H⊗n · (phase flip at 0) · H⊗n` (equal to the textbook reflection
/// `2|s⟩⟨s| − I` up to a global phase). Both use [`Circuit::diagonal`], so
/// circuit memory is `O(2^n)`, not `O(4^n)`.
pub fn grover<S: Scalar>(n: usize, marked: u64, iterations: usize) -> Result<Circuit<S>> {
    assert!((1..=63).contains(&n), "grover: width must fit u64 indices");
    assert!(marked < (1u64 << n), "grover: marked state out of range");
    let oracle = phase_flip::<S>(n, marked);
    let flip_zero = phase_flip::<S>(n, 0);
    let all: Vec<usize> = (0..n).collect();
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    for _ in 0..iterations {
        c.diagonal("oracle", oracle.clone(), all.clone());
        for q in 0..n {
            c.h(q);
        }
        c.diagonal("flip|0…0⟩", flip_zero.clone(), all.clone());
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
