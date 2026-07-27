//! Exact amplitudes for the Clifford+T fragment: absolute reference
//! values with no floating-point error at all.
//!
//! The dense backend is the crate's *reference*, but it is still a float
//! machine: every gate multiplies IEEE doubles, so a long circuit's
//! amplitudes carry ~1e-16-per-op drift, exact zeros come out as ~1e-17
//! residue, and "how wrong is this backend?" bottoms out at "as wrong as
//! the reference". This module removes that floor for the fragment where
//! it can be removed honestly: circuits over the ring
//!
//! ```text
//! D[ω] = ℤ[1/√2, ω],   ω = e^{iπ/4}
//! ```
//!
//! — the ring of Clifford+T amplitudes (Kliuchnikov–Maslov–Mosca,
//! Ross–Selinger). Every element is `(a + bω + cω² + dω³)/√2^k` with
//! integer coefficients, closed under the matrix entries of `h`, `t`,
//! `cx`, the full Clifford set, and every standard rotation at
//! eighth-turn angles. [`ExactState`] evaluates a [`Circuit`] in this
//! ring with `i128` coefficients: results are **exact** — amplitude
//! equality, exact zeros and Born weights are decidable, not
//! approximate — and [`ExactState::max_deviation_vs`] turns any backend's
//! output into an *absolute* error measurement instead of a
//! relative-to-dense one.
//!
//! Scope, stated plainly:
//!
//! * gates whose entries (at the given parameters) lie in `D[ω]` are
//!   supported — the whole Clifford+T set, and parametric gates
//!   (`p`, `cp`, `rx`, `ry`, `rz`, `crx`, `cry`, `crz`, `rxx`, `ryy`,
//!   `rzz`, `u`) at angles that are exact multiples of π/4 (half-angle
//!   gates: θ/2 a multiple of π/4). Anything else fails loudly with the
//!   offending gate named — no silent rounding;
//! * diagonal kernels are accepted when every entry is a unit
//!   eighth-turn phase `±ω^m` (phase oracles, multi-controlled Z);
//! * raw float matrices are rejected: recognizing arbitrary floats as
//!   ring elements would smuggle approximation into the exact path;
//! * coefficients are `i128` and checked: a circuit deep enough to
//!   overflow errors out rather than wrapping. Width is capped at
//!   [`EXACT_MAX_QUBITS`] (the state is a dense `2^n` vector of 68-byte
//!   elements).
//!
//! The exact evaluator is itself the certification anchor for the rest of
//! the crate: `tests/exact_reference.rs` measures the dense backend's
//! true float error against it, and the [`Ball`](crate::scalar) scalar's
//! certified radii are validated by containment against these values.

use crate::circuit::{Circuit, Op};
use crate::error::{Error, Result};
use crate::scalar::C64;

/// Maximum register width for the exact evaluator (dense `2^n` storage of
/// 68-byte elements; 20 qubits ≈ 68 MiB).
pub const EXACT_MAX_QUBITS: usize = 20;

/// Tolerance for recognizing a parameter as an exact multiple of π/4.
const ANGLE_SNAP_TOL: f64 = 1e-9;

fn overflow() -> Error {
    Error::InvalidState(
        "exact arithmetic overflow: circuit too deep for i128 D[ω] coefficients".to_string(),
    )
}

fn cadd(x: i128, y: i128) -> Result<i128> {
    x.checked_add(y).ok_or_else(overflow)
}

fn cmul(x: i128, y: i128) -> Result<i128> {
    x.checked_mul(y).ok_or_else(overflow)
}

/// An element of `D[ω] = ℤ[1/√2, ω]`, `ω = e^{iπ/4}`: the value
/// `(c₀ + c₁ω + c₂ω² + c₃ω³)/√2^k` held exactly with integer
/// coefficients. Values are kept in canonical form (no factor of √2
/// divides the numerator while `k > 0`), so derived `PartialEq`/`Eq` is
/// exact equality in the ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DOmega {
    /// Coefficients of `1, ω, ω², ω³`.
    c: [i128; 4],
    /// Power of √2 in the denominator.
    k: u32,
}

// The arithmetic methods deliberately return `Result` (checked i128
// coefficients), so they cannot implement the std operator traits; the
// familiar names are kept because they are the type's primary operations.
#[allow(clippy::should_implement_trait)]
impl DOmega {
    /// The zero element.
    pub fn zero() -> Self {
        DOmega { c: [0; 4], k: 0 }
    }

    /// An integer.
    pub fn int(n: i64) -> Self {
        DOmega {
            c: [n as i128, 0, 0, 0],
            k: 0,
        }
        .reduced()
    }

    /// `ω^m` for any integer `m` (`ω⁸ = 1`).
    pub fn omega_pow(m: i64) -> Self {
        let m = m.rem_euclid(8) as usize;
        let mut c = [0i128; 4];
        if m < 4 {
            c[m] = 1;
        } else {
            c[m - 4] = -1;
        }
        DOmega { c, k: 0 }
    }

    /// `1/√2^k`.
    pub fn inv_sqrt2_pow(k: u32) -> Self {
        DOmega { c: [1, 0, 0, 0], k }.reduced()
    }

    /// The raw `(coefficients, √2-exponent)` pair, canonical form.
    pub fn parts(&self) -> ([i128; 4], u32) {
        (self.c, self.k)
    }

    /// Canonical form: divide numerator and denominator by √2 while
    /// possible (`√2·(p,q,r,s) = (q−s, p+r, q+s, r−p)`, so divisibility
    /// is the parity condition `c₀≡c₂, c₁≡c₃ (mod 2)`).
    fn reduced(mut self) -> Self {
        if self.c == [0; 4] {
            self.k = 0;
            return self;
        }
        while self.k > 0 {
            let [a, b, c, d] = self.c;
            if (a - c) % 2 != 0 || (b - d) % 2 != 0 {
                break;
            }
            self.c = [(b - d) / 2, (a + c) / 2, (b + d) / 2, (c - a) / 2];
            self.k -= 1;
        }
        self
    }

    /// Multiply numerator by √2 (raise `k` by one without changing the
    /// value) — the inverse of one reduction step.
    fn raised(self) -> Result<Self> {
        let [p, q, r, s] = self.c;
        Ok(DOmega {
            c: [cadd(q, -s)?, cadd(p, r)?, cadd(q, s)?, cadd(r, -p)?],
            k: self.k + 1,
        })
    }

    /// Exact sum.
    pub fn add(self, rhs: Self) -> Result<Self> {
        let (mut a, mut b) = (self, rhs);
        while a.k < b.k {
            a = a.raised()?;
        }
        while b.k < a.k {
            b = b.raised()?;
        }
        let mut c = [0i128; 4];
        for (slot, (&x, &y)) in c.iter_mut().zip(a.c.iter().zip(b.c.iter())) {
            *slot = cadd(x, y)?;
        }
        Ok(DOmega { c, k: a.k }.reduced())
    }

    /// Exact difference.
    pub fn sub(self, rhs: Self) -> Result<Self> {
        self.add(rhs.neg())
    }

    /// Negation.
    pub fn neg(mut self) -> Self {
        for x in &mut self.c {
            *x = -*x;
        }
        self
    }

    /// Exact product (`ω⁴ = −1` cyclotomic convolution).
    pub fn mul(self, rhs: Self) -> Result<Self> {
        let mut c = [0i128; 4];
        for i in 0..4 {
            if self.c[i] == 0 {
                continue;
            }
            for j in 0..4 {
                if rhs.c[j] == 0 {
                    continue;
                }
                let term = cmul(self.c[i], rhs.c[j])?;
                let m = i + j;
                if m < 4 {
                    c[m] = cadd(c[m], term)?;
                } else {
                    c[m - 4] = cadd(c[m - 4], -term)?;
                }
            }
        }
        Ok(DOmega {
            c,
            k: self.k + rhs.k,
        }
        .reduced())
    }

    /// Complex conjugate (`conj(ω) = −ω³`, `conj(ω²) = −ω²`).
    pub fn conj(self) -> Self {
        DOmega {
            c: [self.c[0], -self.c[3], -self.c[2], -self.c[1]],
            k: self.k,
        }
    }

    /// Exactly zero?
    pub fn is_zero(&self) -> bool {
        self.c == [0; 4]
    }

    /// `|z|² = z·conj(z)` as an exact real element of `ℤ[√2]/2^k`.
    pub fn born_weight_exact(self) -> Result<ExactReal> {
        let mut m = self.mul(self.conj())?;
        // Real elements of D[ω] have c₂ = 0 and c₁ = −c₃:
        // value = (c₀ + c₁√2)/√2^k. Canonical k may be odd — raise once
        // so the denominator is a plain power of 2.
        if m.k % 2 == 1 {
            m = m.raised()?;
        }
        debug_assert_eq!(m.c[2], 0, "|z|² must be real");
        debug_assert_eq!(m.c[1], -m.c[3], "|z|² must be real");
        Ok(ExactReal {
            int: m.c[0],
            sqrt2: m.c[1],
            k: m.k / 2,
        })
    }

    /// Convert to floating point (the one deliberately inexact door).
    pub fn to_c64(self) -> C64 {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let re = self.c[0] as f64 + (self.c[1] as f64 - self.c[3] as f64) * s;
        let im = self.c[2] as f64 + (self.c[1] as f64 + self.c[3] as f64) * s;
        let scale = std::f64::consts::SQRT_2.powi(self.k as i32).recip();
        C64::new(re * scale, im * scale)
    }

    /// Recognize a complex number as a unit eighth-turn phase `±ω^m`
    /// (used to accept diagonal kernels exactly). Returns `None` when the
    /// value is not unimodular or its argument is not a multiple of π/4.
    pub fn from_eighth_turn(z: C64) -> Option<Self> {
        if (z.norm_sqr() - 1.0).abs() > 1e-9 {
            return None;
        }
        let turns = z.arg() / std::f64::consts::FRAC_PI_4;
        let m = turns.round();
        if (turns - m).abs() > ANGLE_SNAP_TOL {
            return None;
        }
        Some(DOmega::omega_pow(m as i64))
    }
}

/// An exact real number `(int + sqrt2·√2)/2^k` — the form Born weights
/// take in `D[ω]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactReal {
    /// Integer part of the numerator.
    pub int: i128,
    /// Coefficient of √2 in the numerator.
    pub sqrt2: i128,
    /// Power of 2 in the denominator.
    pub k: u32,
}

// Same trade as `DOmega`: checked arithmetic returns `Result`, so the
// std operator traits cannot apply.
#[allow(clippy::should_implement_trait)]
impl ExactReal {
    /// The exact zero.
    pub fn zero() -> Self {
        ExactReal {
            int: 0,
            sqrt2: 0,
            k: 0,
        }
    }

    /// Exact sum.
    pub fn add(self, rhs: Self) -> Result<Self> {
        let (mut a, mut b) = (self, rhs);
        while a.k < b.k {
            a = ExactReal {
                int: cmul(a.int, 2)?,
                sqrt2: cmul(a.sqrt2, 2)?,
                k: a.k + 1,
            };
        }
        while b.k < a.k {
            b = ExactReal {
                int: cmul(b.int, 2)?,
                sqrt2: cmul(b.sqrt2, 2)?,
                k: b.k + 1,
            };
        }
        let mut out = ExactReal {
            int: cadd(a.int, b.int)?,
            sqrt2: cadd(a.sqrt2, b.sqrt2)?,
            k: a.k,
        };
        while out.k > 0 && out.int % 2 == 0 && out.sqrt2 % 2 == 0 {
            out.int /= 2;
            out.sqrt2 /= 2;
            out.k -= 1;
        }
        Ok(out)
    }

    /// Convert to floating point.
    pub fn to_f64(self) -> f64 {
        (self.int as f64 + self.sqrt2 as f64 * std::f64::consts::SQRT_2)
            / (2f64).powi(self.k as i32)
    }
}

/// cos(m·π/4) as an exact ring element.
fn cos_pi4(m: i64) -> DOmega {
    let h = || DOmega::inv_sqrt2_pow(1);
    match m.rem_euclid(8) {
        0 => DOmega::int(1),
        1 | 7 => h(),
        2 | 6 => DOmega::zero(),
        3 | 5 => h().neg(),
        _ => DOmega::int(-1),
    }
}

/// sin(m·π/4) as an exact ring element.
fn sin_pi4(m: i64) -> DOmega {
    cos_pi4(m - 2)
}

/// Recognize `x` as an exact multiple of π/4; `None` otherwise.
fn eighth_turns(x: f64) -> Option<i64> {
    let turns = x / std::f64::consts::FRAC_PI_4;
    let m = turns.round();
    ((turns - m).abs() < ANGLE_SNAP_TOL && m.abs() < 1e15).then_some(m as i64)
}

/// Controlled version of a 2×2 exact matrix, control at sub-index bit 0
/// (matching [`crate::math::GateMatrix::controlled`]).
fn controlled(u: &[DOmega; 4]) -> Vec<DOmega> {
    let (o, l) = (DOmega::int(1), DOmega::zero());
    let mut m = vec![l; 16];
    m[0] = o;
    m[2 * 4 + 2] = o;
    m[4 + 1] = u[0];
    m[4 + 3] = u[1];
    m[3 * 4 + 1] = u[2];
    m[3 * 4 + 3] = u[3];
    m
}

/// Doubly-controlled version (controls at sub-index bits 0 and 1).
fn controlled2(u: &[DOmega; 4]) -> Vec<DOmega> {
    let (o, l) = (DOmega::int(1), DOmega::zero());
    let mut m = vec![l; 64];
    for j in 0..8 {
        if j & 3 != 3 {
            m[j * 8 + j] = o;
        }
    }
    m[3 * 8 + 3] = u[0];
    m[3 * 8 + 7] = u[1];
    m[7 * 8 + 3] = u[2];
    m[7 * 8 + 7] = u[3];
    m
}

/// `rz(θ)`-style diagonal pair `(e^{−iθ/2}, e^{iθ/2})` at snapped angles.
fn rz_pair(theta: f64) -> Option<[DOmega; 2]> {
    let m = eighth_turns(theta / 2.0)?;
    Some([DOmega::omega_pow(-m), DOmega::omega_pow(m)])
}

/// `rx(θ)` as a 2×2 exact matrix at snapped angles.
fn rx_matrix(theta: f64) -> Option<[DOmega; 4]> {
    let m = eighth_turns(theta / 2.0)?;
    let (c, s) = (cos_pi4(m), sin_pi4(m));
    let minus_i = DOmega::omega_pow(6);
    let mis = minus_i.mul(s).ok()?;
    Some([c, mis, mis, c])
}

/// `ry(θ)` as a 2×2 exact matrix at snapped angles.
fn ry_matrix(theta: f64) -> Option<[DOmega; 4]> {
    let m = eighth_turns(theta / 2.0)?;
    let (c, s) = (cos_pi4(m), sin_pi4(m));
    Some([c, s.neg(), s, c])
}

/// The exact matrix of a named standard gate at the given parameters, or
/// an [`Error::UnsupportedForAlgebra`] naming the gate when it (or its
/// parameter values) leave the ring.
fn exact_gate(name: &str, params: &[f64]) -> Result<Vec<DOmega>> {
    let unsupported = || Error::UnsupportedForAlgebra {
        gate: format!("{name}{params:?}"),
        algebra: "D[ω] (exact)".to_string(),
    };
    let (o, l) = (DOmega::int(1), DOmega::zero());
    let h_ = DOmega::inv_sqrt2_pow(1);
    let i_ = DOmega::omega_pow(2);
    let mi = DOmega::omega_pow(6);
    let m: Vec<DOmega> = match name {
        "id" | "identity" => vec![o, l, l, o],
        "x" | "not" => vec![l, o, o, l],
        "y" => vec![l, mi, i_, l],
        "z" => vec![o, l, l, o.neg()],
        "h" => vec![h_, h_, h_, h_.neg()],
        "s" => vec![o, l, l, i_],
        "sdg" => vec![o, l, l, mi],
        "t" => vec![o, l, l, DOmega::omega_pow(1)],
        "tdg" => vec![o, l, l, DOmega::omega_pow(-1)],
        "sx" => {
            // (1±i)/2 entries.
            let p = DOmega {
                c: [1, 0, 1, 0],
                k: 2,
            }
            .reduced();
            let q = DOmega {
                c: [1, 0, -1, 0],
                k: 2,
            }
            .reduced();
            vec![p, q, q, p]
        }
        "sxdg" => {
            let p = DOmega {
                c: [1, 0, 1, 0],
                k: 2,
            }
            .reduced();
            let q = DOmega {
                c: [1, 0, -1, 0],
                k: 2,
            }
            .reduced();
            vec![q, p, p, q]
        }
        "p" | "phase" => {
            let m = eighth_turns(params[0]).ok_or_else(unsupported)?;
            vec![o, l, l, DOmega::omega_pow(m)]
        }
        "rz" => {
            let [a, b] = rz_pair(params[0]).ok_or_else(unsupported)?;
            vec![a, l, l, b]
        }
        "rx" => rx_matrix(params[0]).ok_or_else(unsupported)?.to_vec(),
        "ry" => ry_matrix(params[0]).ok_or_else(unsupported)?.to_vec(),
        "u" | "u3" => {
            let mt = eighth_turns(params[0] / 2.0).ok_or_else(unsupported)?;
            let mp = eighth_turns(params[1]).ok_or_else(unsupported)?;
            let ml = eighth_turns(params[2]).ok_or_else(unsupported)?;
            let (c, s) = (cos_pi4(mt), sin_pi4(mt));
            vec![
                c,
                DOmega::omega_pow(ml).mul(s)?.neg(),
                DOmega::omega_pow(mp).mul(s)?,
                DOmega::omega_pow(mp + ml).mul(c)?,
            ]
        }
        "cx" | "cnot" => controlled(&[l, o, o, l]),
        "cy" => controlled(&[l, mi, i_, l]),
        "cz" => controlled(&[o, l, l, o.neg()]),
        "ch" => controlled(&[h_, h_, h_, h_.neg()]),
        "cp" | "cphase" => {
            let m = eighth_turns(params[0]).ok_or_else(unsupported)?;
            controlled(&[o, l, l, DOmega::omega_pow(m)])
        }
        "crx" => controlled(&rx_matrix(params[0]).ok_or_else(unsupported)?),
        "cry" => controlled(&ry_matrix(params[0]).ok_or_else(unsupported)?),
        "crz" => {
            let [a, b] = rz_pair(params[0]).ok_or_else(unsupported)?;
            controlled(&[a, l, l, b])
        }
        "swap" => vec![o, l, l, l, l, l, o, l, l, o, l, l, l, l, l, o],
        "iswap" => vec![o, l, l, l, l, l, i_, l, l, i_, l, l, l, l, l, o],
        "rzz" => {
            let [a, b] = rz_pair(params[0]).ok_or_else(unsupported)?;
            let mut m = vec![l; 16];
            for (j, e) in [a, b, b, a].into_iter().enumerate() {
                m[j * 4 + j] = e;
            }
            m
        }
        "rxx" | "ryy" => {
            let mm = eighth_turns(params[0] / 2.0).ok_or_else(unsupported)?;
            let (c, s) = (cos_pi4(mm), sin_pi4(mm));
            let off = mi.mul(s)?;
            let corner = if name == "rxx" { off } else { off.neg() };
            let mut m = vec![l; 16];
            for j in 0..4 {
                m[j * 4 + j] = c;
            }
            m[3] = corner;
            m[4 + 2] = off;
            m[2 * 4 + 1] = off;
            m[3 * 4] = corner;
            m
        }
        "ccx" | "toffoli" => controlled2(&[l, o, o, l]),
        "ccz" => controlled2(&[o, l, l, o.neg()]),
        "cswap" | "fredkin" => {
            // Swap sub-bits 1 and 2 when sub-bit 0 (the control) is set.
            let mut m = vec![l; 64];
            for j in 0..8usize {
                let i = if j & 1 == 1 {
                    (j & 1) | ((j >> 1) & 1) << 2 | ((j >> 2) & 1) << 1
                } else {
                    j
                };
                m[i * 8 + j] = o;
            }
            m
        }
        _ => return Err(unsupported()),
    };
    Ok(m)
}

/// A dense `2^n` state vector over [`DOmega`]: the exact evaluator.
#[derive(Debug, Clone)]
pub struct ExactState {
    num_qubits: usize,
    amps: Vec<DOmega>,
}

impl ExactState {
    /// `|0…0⟩` on `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits > EXACT_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: num_qubits,
                max: EXACT_MAX_QUBITS,
            });
        }
        let mut amps = vec![DOmega::zero(); 1usize << num_qubits];
        amps[0] = DOmega::int(1);
        Ok(ExactState { num_qubits, amps })
    }

    /// Evaluate a circuit exactly. Named gates resolve through the exact
    /// gate table; diagonal kernels are accepted at unit eighth-turn
    /// phases; raw matrices are rejected (see the module docs).
    pub fn run(circuit: &Circuit<C64>) -> Result<Self> {
        let mut state = ExactState::new(circuit.num_qubits())?;
        for op in circuit.ops() {
            match op {
                Op::Named {
                    name,
                    params,
                    qubits,
                } => {
                    let matrix = exact_gate(name, params)?;
                    state.apply(&matrix, qubits)?;
                }
                Op::Diagonal {
                    label,
                    entries,
                    qubits,
                } => {
                    let exact: Option<Vec<DOmega>> = entries
                        .iter()
                        .map(|&e| DOmega::from_eighth_turn(e))
                        .collect();
                    let exact = exact.ok_or_else(|| Error::UnsupportedForAlgebra {
                        gate: format!("diagonal '{label}' (non-eighth-turn entry)"),
                        algebra: "D[ω] (exact)".to_string(),
                    })?;
                    state.apply_diagonal(&exact, qubits)?;
                }
                Op::Raw { label, .. } => {
                    return Err(Error::UnsupportedForAlgebra {
                        gate: format!("raw '{label}' (float matrix)"),
                        algebra: "D[ω] (exact)".to_string(),
                    });
                }
            }
        }
        Ok(state)
    }

    /// Register width.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Apply an exact `2^k × 2^k` matrix (row-major) to the given qubits.
    pub fn apply(&mut self, matrix: &[DOmega], qubits: &[usize]) -> Result<()> {
        crate::circuit::validate_targets(self.num_qubits, qubits)?;
        let k = qubits.len();
        let d = 1usize << k;
        if matrix.len() != d * d {
            return Err(Error::BadDimension {
                expected: d,
                got: (matrix.len() as f64).sqrt() as usize,
            });
        }
        let mut sorted = qubits.to_vec();
        sorted.sort_unstable();
        let scatter = crate::backend::scatter_table(qubits);
        let groups = self.amps.len() >> k;
        let mut scratch = vec![DOmega::zero(); d];
        for g in 0..groups {
            let base = crate::backend::expand_index(g as u64, &sorted);
            for (j, slot) in scratch.iter_mut().enumerate() {
                *slot = self.amps[(base | scatter[j]) as usize];
            }
            for r in 0..d {
                let mut acc = DOmega::zero();
                for (l, &s) in scratch.iter().enumerate() {
                    let e = matrix[r * d + l];
                    if e.is_zero() || s.is_zero() {
                        continue;
                    }
                    acc = acc.add(e.mul(s)?)?;
                }
                self.amps[(base | scatter[r]) as usize] = acc;
            }
        }
        Ok(())
    }

    /// Apply an exact diagonal (little-endian sub-index over `qubits`).
    pub fn apply_diagonal(&mut self, entries: &[DOmega], qubits: &[usize]) -> Result<()> {
        crate::circuit::validate_targets(self.num_qubits, qubits)?;
        if entries.len() != 1usize << qubits.len() {
            return Err(Error::BadDimension {
                expected: 1usize << qubits.len(),
                got: entries.len(),
            });
        }
        for (i, a) in self.amps.iter_mut().enumerate() {
            if a.is_zero() {
                continue;
            }
            let mut sub = 0usize;
            for (b, &q) in qubits.iter().enumerate() {
                sub |= ((i >> q) & 1) << b;
            }
            *a = entries[sub].mul(*a)?;
        }
        Ok(())
    }

    /// The exact amplitude of a basis state.
    pub fn amplitude_exact(&self, index: u64) -> DOmega {
        self.amps
            .get(index as usize)
            .copied()
            .unwrap_or_else(DOmega::zero)
    }

    /// The amplitude as floating point (correct to f64 conversion of the
    /// exact value — one rounding, not one per gate).
    pub fn amplitude_c64(&self, index: u64) -> C64 {
        self.amplitude_exact(index).to_c64()
    }

    /// The exact Born weight of a basis state.
    pub fn probability_exact(&self, index: u64) -> Result<ExactReal> {
        self.amplitude_exact(index).born_weight_exact()
    }

    /// The exact total Born weight (1 for a unitary circuit — exactly,
    /// not within tolerance).
    pub fn total_weight_exact(&self) -> Result<ExactReal> {
        let mut total = ExactReal::zero();
        for a in &self.amps {
            if !a.is_zero() {
                total = total.add(a.born_weight_exact()?)?;
            }
        }
        Ok(total)
    }

    /// Number of *exactly* nonzero amplitudes — a support count no float
    /// backend can provide (a 1e-17 residue is nonzero to a float scan).
    pub fn support_exact(&self) -> usize {
        self.amps.iter().filter(|a| !a.is_zero()).count()
    }

    /// Largest absolute amplitude deviation of a backend's state from the
    /// exact values — the crate's absolute error yardstick.
    pub fn max_deviation_vs(&self, state: &dyn crate::backend::Backend<C64>) -> f64 {
        let mut dev = 0.0f64;
        for (i, a) in self.amps.iter().enumerate() {
            let d = (state.amplitude(i as u64) - a.to_c64()).norm();
            dev = dev.max(d);
        }
        dev
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::c64;

    fn approx(a: C64, b: C64) -> bool {
        (a - b).norm() < 1e-12
    }

    #[test]
    fn ring_constants_and_conjugation() {
        let omega = DOmega::omega_pow(1);
        // ω⁸ = 1, ω⁴ = −1.
        let mut p = DOmega::int(1);
        for _ in 0..8 {
            p = p.mul(omega).unwrap();
        }
        assert_eq!(p, DOmega::int(1));
        assert_eq!(DOmega::omega_pow(4), DOmega::int(-1));
        // √2 = ω − ω³ and (1/√2)² = 1/2.
        let sqrt2 = DOmega::omega_pow(1).sub(DOmega::omega_pow(3)).unwrap();
        assert!(approx(sqrt2.to_c64(), c64(std::f64::consts::SQRT_2, 0.0)));
        let h = DOmega::inv_sqrt2_pow(1);
        assert!(approx(h.mul(h).unwrap().to_c64(), c64(0.5, 0.0)));
        assert_eq!(h.mul(sqrt2).unwrap(), DOmega::int(1));
        // conj(ω)·ω = 1.
        assert_eq!(omega.conj().mul(omega).unwrap(), DOmega::int(1));
        // Born weight of ω^m/√2 is exactly 1/2.
        for m in 0..8 {
            let z = DOmega::omega_pow(m).mul(h).unwrap();
            let w = z.born_weight_exact().unwrap();
            assert_eq!((w.int, w.sqrt2, w.k), (1, 0, 1));
        }
    }

    #[test]
    fn reduction_is_canonical() {
        // 2/√2² reduces to 1; equal values compare equal whatever their
        // construction path.
        let a = DOmega {
            c: [2, 0, 0, 0],
            k: 2,
        }
        .reduced();
        assert_eq!(a, DOmega::int(1));
        let b = DOmega::inv_sqrt2_pow(3)
            .mul(DOmega::omega_pow(1).sub(DOmega::omega_pow(3)).unwrap())
            .unwrap();
        assert_eq!(b, DOmega::inv_sqrt2_pow(2));
    }

    #[test]
    fn exact_gate_matrices_match_the_registry() {
        // Every supported gate's exact matrix must agree entrywise with
        // the float registry at the same (snapped) parameters.
        use crate::registry::GateRegistry;
        let reg = GateRegistry::<C64>::standard();
        let pi4 = std::f64::consts::FRAC_PI_4;
        let cases: Vec<(&str, Vec<f64>)> = vec![
            ("id", vec![]),
            ("x", vec![]),
            ("y", vec![]),
            ("z", vec![]),
            ("h", vec![]),
            ("s", vec![]),
            ("sdg", vec![]),
            ("t", vec![]),
            ("tdg", vec![]),
            ("sx", vec![]),
            ("sxdg", vec![]),
            ("p", vec![3.0 * pi4]),
            ("rz", vec![2.0 * pi4]),
            ("rx", vec![2.0 * pi4]),
            ("ry", vec![6.0 * pi4]),
            ("u", vec![2.0 * pi4, pi4, -pi4]),
            ("cx", vec![]),
            ("cy", vec![]),
            ("cz", vec![]),
            ("ch", vec![]),
            ("cp", vec![-5.0 * pi4]),
            ("crx", vec![2.0 * pi4]),
            ("cry", vec![-2.0 * pi4]),
            ("crz", vec![4.0 * pi4]),
            ("swap", vec![]),
            ("iswap", vec![]),
            ("rzz", vec![2.0 * pi4]),
            ("rxx", vec![2.0 * pi4]),
            ("ryy", vec![-6.0 * pi4]),
            ("ccx", vec![]),
            ("ccz", vec![]),
            ("cswap", vec![]),
        ];
        for (name, params) in cases {
            let exact = exact_gate(name, &params).unwrap();
            let float = reg.resolve(name).unwrap().matrix(&params).unwrap();
            let d = float.dim();
            assert_eq!(exact.len(), d * d, "{name}: dimension");
            for r in 0..d {
                for c in 0..d {
                    assert!(
                        approx(exact[r * d + c].to_c64(), float.get(r, c)),
                        "{name}[{r}][{c}]: exact {:?} vs float {}",
                        exact[r * d + c].to_c64(),
                        float.get(r, c)
                    );
                }
            }
        }
    }

    #[test]
    fn generic_angles_and_raw_ops_fail_loudly() {
        assert!(matches!(
            exact_gate("rz", &[0.7365]),
            Err(Error::UnsupportedForAlgebra { .. })
        ));
        assert!(matches!(
            exact_gate("nonexistent", &[]),
            Err(Error::UnsupportedForAlgebra { .. })
        ));
        let mut c: Circuit = Circuit::new(1);
        c.raw(
            "scramble",
            crate::math::GateMatrix::identity(2).unwrap(),
            vec![0],
        );
        assert!(matches!(
            ExactState::run(&c),
            Err(Error::UnsupportedForAlgebra { .. })
        ));
    }
}
