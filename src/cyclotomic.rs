//! Exact amplitudes past Clifford+T: circuits over
//!
//! ```text
//! D[ζ_N] = ℤ[1/√2, ζ_N],   ζ_N = e^{2πi/N},   N = 2^m ≥ 8
//! ```
//!
//! [`crate::exact`] evaluates the Clifford+T fragment in `D[ω]`, `ω = ζ_8`,
//! and rejects any phase finer than an eighth turn. Here `N` is any power of
//! two from 8 up, so phases at multiples of `2π/N` are exact: `p`, `cp` and
//! the diagonal kernels at multiples of `2π/N`, the half-angle rotations
//! (`rz`, `rx`, `ry`, `u`, `crz`, `rzz`, …) at multiples of `4π/N`. That
//! takes in `√T` at `N = 16` and the controlled phases of the `n`-qubit QFT
//! at `N = 2^n`. At `N = 8` this is `D[ω]` again, and the two evaluators
//! agree amplitude for amplitude.
//!
//! An element is `z / √2^k`: `z ∈ ℤ[ζ_N] = ℤ[x]/(x^{N/2} + 1)` as `N/2`
//! checked `i128` coefficients on `1, ζ, …, ζ^{N/2−1}`, and `k` the least
//! power of `√2 = ζ^{N/8} − ζ^{3N/8}` that makes the numerator integral, so
//! derived equality is exact equality in the ring and exact zeros are
//! decidable. A product is a negacyclic convolution, `O(N²)` in general and
//! `O(N)` when one factor is a root of unity. Gate recognition follows
//! [`crate::exact`]: named gates at snapped angles and unit diagonal
//! phases; raw float matrices are refused.

use crate::backend::{expand_index, scatter_table, Backend};
use crate::circuit::{validate_targets, Circuit, Op};
use crate::error::{Error, Result};
use crate::exact::EXACT_MAX_QUBITS;
use crate::scalar::C64;

/// Tolerance for recognizing an angle as an exact multiple of `2π/N`.
const ANGLE_SNAP_TOL: f64 = 1e-9;

fn overflow() -> Error {
    Error::InvalidState("D[ζ_N] coefficient overflow (exceeds i128)".to_string())
}

fn cadd(x: i128, y: i128) -> Result<i128> {
    x.checked_add(y).ok_or_else(overflow)
}

fn cmul(x: i128, y: i128) -> Result<i128> {
    x.checked_mul(y).ok_or_else(overflow)
}

/// An element `z / √2^k` of `D[ζ_N]`, `N` a power of two `≥ 8`, in
/// canonical form (no factor `√2` of the numerator while `k > 0`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DZeta<const N: u32> {
    /// Coefficients of `1, ζ, …, ζ^{N/2−1}`.
    c: Vec<i128>,
    /// Power of √2 in the denominator.
    k: u32,
}

// Checked arithmetic returns `Result`, so the std operator traits cannot
// apply; the names match `DOmega`'s.
#[allow(clippy::should_implement_trait)]
impl<const N: u32> DZeta<N> {
    const VALID: () = assert!(
        N >= 8 && N.is_power_of_two(),
        "D[ζ_N] needs N a power of two, at least 8"
    );
    const HALF: usize = (N / 2) as usize;

    /// The zero element.
    pub fn zero() -> Self {
        let () = Self::VALID;
        DZeta {
            c: vec![0; Self::HALF],
            k: 0,
        }
    }

    /// An integer.
    pub fn int(n: i128) -> Self {
        let mut z = Self::zero();
        z.c[0] = n;
        z
    }

    /// `ζ_N^m` for any integer `m`.
    pub fn zeta_pow(m: i64) -> Self {
        let mut z = Self::zero();
        let e = m.rem_euclid(i64::from(N)) as usize;
        if e < Self::HALF {
            z.c[e] = 1;
        } else {
            z.c[e - Self::HALF] = -1;
        }
        z
    }

    /// `1/√2^k`.
    pub fn inv_sqrt2_pow(k: u32) -> Self {
        let mut z = Self::int(1);
        z.k = k;
        z.reduced().expect("1/√2^k reduces without overflow")
    }

    /// `(Σ_j c_j ζ^j) / √2^k` from `N/2` coefficients, put in canonical
    /// form.
    ///
    /// # Errors
    ///
    /// [`Error::BadDimension`] unless `c.len() == N/2`, or
    /// [`Error::InvalidState`] on overflow.
    pub fn from_parts(c: &[i128], k: u32) -> Result<Self> {
        let mut z = Self::zero();
        if c.len() != z.c.len() {
            return Err(Error::BadDimension {
                expected: z.c.len(),
                got: c.len(),
            });
        }
        z.c.copy_from_slice(c);
        z.k = k;
        z.reduced()
    }

    /// The canonical `(coefficients, √2-exponent)` pair.
    pub fn parts(&self) -> (&[i128], u32) {
        (&self.c, self.k)
    }

    /// Exact zero test.
    pub fn is_zero(&self) -> bool {
        self.c.iter().all(|&x| x == 0)
    }

    /// The numerator times `ζ^s`: a negacyclic shift (`ζ^{N/2} = −1`).
    fn shifted(c: &[i128], s: usize) -> Result<Vec<i128>> {
        let h = c.len();
        let mut out = vec![0; h];
        for (i, &x) in c.iter().enumerate() {
            let e = (i + s) % (2 * h);
            if e < h {
                out[e] = x;
            } else {
                out[e - h] = x.checked_neg().ok_or_else(overflow)?;
            }
        }
        Ok(out)
    }

    /// The numerator times `√2 = ζ^{N/8} − ζ^{3N/8}`.
    fn times_sqrt2(c: &[i128]) -> Result<Vec<i128>> {
        let a = (N / 8) as usize;
        let (p, q) = (Self::shifted(c, a)?, Self::shifted(c, 3 * a)?);
        p.iter().zip(&q).map(|(&x, &y)| cadd(x, -y)).collect()
    }

    /// Canonical form: divide numerator and denominator by `√2` while the
    /// numerator is divisible, i.e. while `z·√2` has even coefficients.
    fn reduced(mut self) -> Result<Self> {
        if self.is_zero() {
            self.k = 0;
            return Ok(self);
        }
        while self.k > 0 {
            let t = Self::times_sqrt2(&self.c)?;
            if t.iter().any(|x| x % 2 != 0) {
                break;
            }
            self.c = t.into_iter().map(|x| x / 2).collect();
            self.k -= 1;
        }
        Ok(self)
    }

    /// The same value over `√2^{k+d}`.
    fn raised(&self, d: u32) -> Result<Vec<i128>> {
        let mut c = self.c.clone();
        for _ in 0..d {
            c = Self::times_sqrt2(&c)?;
        }
        Ok(c)
    }

    /// Exact sum.
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        let k = self.k.max(rhs.k);
        let (a, b) = (self.raised(k - self.k)?, rhs.raised(k - rhs.k)?);
        let c = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| cadd(x, y))
            .collect::<Result<_>>()?;
        DZeta { c, k }.reduced()
    }

    /// Negation.
    pub fn neg(&self) -> Self {
        DZeta {
            c: self.c.iter().map(|&x| -x).collect(),
            k: self.k,
        }
    }

    /// Exact difference.
    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.add(&rhs.neg())
    }

    /// Exact product: negacyclic convolution of the numerators.
    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        let h = self.c.len();
        let mut c = vec![0i128; h];
        for (i, &x) in self.c.iter().enumerate().filter(|(_, &x)| x != 0) {
            for (j, &y) in rhs.c.iter().enumerate().filter(|(_, &y)| y != 0) {
                let p = cmul(x, y)?;
                let (slot, p) = if i + j < h {
                    (i + j, p)
                } else {
                    (i + j - h, -p)
                };
                c[slot] = cadd(c[slot], p)?;
            }
        }
        DZeta {
            c,
            k: self.k + rhs.k,
        }
        .reduced()
    }

    /// Complex conjugate, `ζ^j ↦ ζ^{−j} = −ζ^{N/2−j}`.
    pub fn conj(&self) -> Self {
        let h = self.c.len();
        let mut c = vec![0; h];
        c[0] = self.c[0];
        for j in 1..h {
            c[h - j] = -self.c[j];
        }
        DZeta { c, k: self.k }
    }

    /// The exact Born weight `|z|²`, a real element of `D[ζ_N]`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidState`] on overflow.
    pub fn born_weight(&self) -> Result<Self> {
        self.conj().mul(self)
    }

    /// Convert to floating point (one rounding of the exact value).
    pub fn to_c64(&self) -> C64 {
        let step = 2.0 * std::f64::consts::PI / f64::from(N);
        let z = self
            .c
            .iter()
            .enumerate()
            .filter(|(_, &x)| x != 0)
            .fold(C64::new(0.0, 0.0), |acc, (j, &x)| {
                acc + C64::from_polar(x as f64, step * j as f64)
            });
        z / std::f64::consts::SQRT_2.powi(self.k as i32)
    }

    /// Recognize a unit complex number as an exact `N`-th root of unity.
    pub fn from_root_of_unity(z: C64) -> Option<Self> {
        if (z.norm_sqr() - 1.0).abs() > 1e-9 {
            return None;
        }
        turns::<N>(z.arg()).map(Self::zeta_pow)
    }
}

/// Recognize `x` as an exact multiple of `2π/N`.
fn turns<const N: u32>(x: f64) -> Option<i64> {
    let t = x / (2.0 * std::f64::consts::PI / f64::from(N));
    let m = t.round();
    ((t - m).abs() < ANGLE_SNAP_TOL && m.abs() < 1e15).then_some(m as i64)
}

/// `cos(2πm/N)` and `sin(2πm/N)`: `(ζ^m ± ζ^{−m})/2`, the sine divided by `i`.
fn cos_sin<const N: u32>(m: i64) -> Result<(DZeta<N>, DZeta<N>)> {
    let half = DZeta::<N>::inv_sqrt2_pow(2);
    let (p, q) = (DZeta::<N>::zeta_pow(m), DZeta::<N>::zeta_pow(-m));
    let minus_i = DZeta::<N>::zeta_pow(-i64::from(N / 4));
    let cos = p.add(&q)?.mul(&half)?;
    let sin = minus_i.mul(&p.sub(&q)?)?.mul(&half)?;
    Ok((cos, sin))
}

/// Controlled version of a 2×2 matrix, control at sub-index bit 0.
fn controlled<T: Clone>(u: &[T; 4], one: &T, zero: &T) -> Vec<T> {
    let mut m = vec![zero.clone(); 16];
    m[0] = one.clone();
    m[2 * 4 + 2] = one.clone();
    m[4 + 1] = u[0].clone();
    m[4 + 3] = u[1].clone();
    m[3 * 4 + 1] = u[2].clone();
    m[3 * 4 + 3] = u[3].clone();
    m
}

/// Doubly-controlled version, controls at sub-index bits 0 and 1.
fn controlled2<T: Clone>(u: &[T; 4], one: &T, zero: &T) -> Vec<T> {
    let mut m = vec![zero.clone(); 64];
    for j in 0..8 {
        if j & 3 != 3 {
            m[j * 8 + j] = one.clone();
        }
    }
    m[3 * 8 + 3] = u[0].clone();
    m[3 * 8 + 7] = u[1].clone();
    m[7 * 8 + 3] = u[2].clone();
    m[7 * 8 + 7] = u[3].clone();
    m
}

/// The exact matrix of a named standard gate at the given parameters, or
/// [`Error::UnsupportedForAlgebra`] naming the gate when it leaves the ring.
fn gate_matrix<const N: u32>(name: &str, params: &[f64]) -> Result<Vec<DZeta<N>>> {
    type Z<const N: u32> = DZeta<N>;
    let unsupported = || Error::UnsupportedForAlgebra {
        gate: format!("{name}{params:?}"),
        algebra: format!("D[ζ_{N}] (exact)"),
    };
    let angle = |i: usize| params.get(i).copied().ok_or_else(unsupported);
    let snap = |x: f64| turns::<N>(x).ok_or_else(unsupported);
    let q = i64::from(N / 4);
    let (o, l) = (Z::<N>::int(1), Z::<N>::zero());
    let i_ = Z::<N>::zeta_pow(q);
    let mi = Z::<N>::zeta_pow(-q);
    let h_ = Z::<N>::inv_sqrt2_pow(1);
    let rz_pair = |theta: f64| -> Result<[Z<N>; 2]> {
        let m = snap(theta / 2.0)?;
        Ok([Z::zeta_pow(-m), Z::zeta_pow(m)])
    };
    let rx = |theta: f64| -> Result<[Z<N>; 4]> {
        let (c, s) = cos_sin::<N>(snap(theta / 2.0)?)?;
        let mis = mi.mul(&s)?;
        Ok([c.clone(), mis.clone(), mis, c])
    };
    let ry = |theta: f64| -> Result<[Z<N>; 4]> {
        let (c, s) = cos_sin::<N>(snap(theta / 2.0)?)?;
        Ok([c.clone(), s.neg(), s, c])
    };
    // (1 ± i)/2, the entries of √X.
    let sx_p = o.add(&i_)?.mul(&Z::inv_sqrt2_pow(2))?;
    let sx_q = o.sub(&i_)?.mul(&Z::inv_sqrt2_pow(2))?;
    let ctl = |u: &[Z<N>; 4]| controlled(u, &o, &l);
    let m: Vec<Z<N>> = match name {
        "id" | "identity" => vec![o.clone(), l.clone(), l.clone(), o.clone()],
        "x" | "not" => vec![l.clone(), o.clone(), o.clone(), l.clone()],
        "y" => vec![l.clone(), mi.clone(), i_.clone(), l.clone()],
        "z" => vec![o.clone(), l.clone(), l.clone(), o.neg()],
        "h" => vec![h_.clone(), h_.clone(), h_.clone(), h_.neg()],
        "s" => vec![o.clone(), l.clone(), l.clone(), i_.clone()],
        "sdg" => vec![o.clone(), l.clone(), l.clone(), mi.clone()],
        "t" => vec![o.clone(), l.clone(), l.clone(), Z::zeta_pow(q / 2)],
        "tdg" => vec![o.clone(), l.clone(), l.clone(), Z::zeta_pow(-q / 2)],
        "sx" => vec![sx_p.clone(), sx_q.clone(), sx_q, sx_p],
        "sxdg" => vec![sx_q.clone(), sx_p.clone(), sx_p, sx_q],
        "p" | "phase" => vec![
            o.clone(),
            l.clone(),
            l.clone(),
            Z::zeta_pow(snap(angle(0)?)?),
        ],
        "rz" => {
            let [a, b] = rz_pair(angle(0)?)?;
            vec![a, l.clone(), l.clone(), b]
        }
        "rx" => rx(angle(0)?)?.to_vec(),
        "ry" => ry(angle(0)?)?.to_vec(),
        "u" | "u3" => {
            let (c, s) = cos_sin::<N>(snap(angle(0)? / 2.0)?)?;
            let (mp, ml) = (snap(angle(1)?)?, snap(angle(2)?)?);
            vec![
                c.clone(),
                Z::zeta_pow(ml).mul(&s)?.neg(),
                Z::zeta_pow(mp).mul(&s)?,
                Z::zeta_pow(mp + ml).mul(&c)?,
            ]
        }
        "cx" | "cnot" => ctl(&[l.clone(), o.clone(), o.clone(), l.clone()]),
        "cy" => ctl(&[l.clone(), mi.clone(), i_.clone(), l.clone()]),
        "cz" => ctl(&[o.clone(), l.clone(), l.clone(), o.neg()]),
        "ch" => ctl(&[h_.clone(), h_.clone(), h_.clone(), h_.neg()]),
        "cp" | "cphase" => ctl(&[
            o.clone(),
            l.clone(),
            l.clone(),
            Z::zeta_pow(snap(angle(0)?)?),
        ]),
        "crx" => ctl(&rx(angle(0)?)?),
        "cry" => ctl(&ry(angle(0)?)?),
        "crz" => {
            let [a, b] = rz_pair(angle(0)?)?;
            ctl(&[a, l.clone(), l.clone(), b])
        }
        "swap" | "iswap" => {
            let off = if name == "swap" {
                o.clone()
            } else {
                i_.clone()
            };
            let mut m = vec![l.clone(); 16];
            m[0] = o.clone();
            m[4 + 2] = off.clone();
            m[2 * 4 + 1] = off;
            m[15] = o.clone();
            m
        }
        "rzz" => {
            let [a, b] = rz_pair(angle(0)?)?;
            let mut m = vec![l.clone(); 16];
            for (j, e) in [a.clone(), b.clone(), b, a].into_iter().enumerate() {
                m[j * 4 + j] = e;
            }
            m
        }
        "rxx" | "ryy" => {
            let (c, s) = cos_sin::<N>(snap(angle(0)? / 2.0)?)?;
            let off = mi.mul(&s)?;
            let corner = if name == "rxx" {
                off.clone()
            } else {
                off.neg()
            };
            let mut m = vec![l.clone(); 16];
            for j in 0..4 {
                m[j * 4 + j] = c.clone();
            }
            m[3] = corner.clone();
            m[4 + 2] = off.clone();
            m[2 * 4 + 1] = off;
            m[3 * 4] = corner;
            m
        }
        "ccx" | "toffoli" => controlled2(&[l.clone(), o.clone(), o.clone(), l.clone()], &o, &l),
        "ccz" => controlled2(&[o.clone(), l.clone(), l.clone(), o.neg()], &o, &l),
        "cswap" | "fredkin" => {
            let mut m = vec![l.clone(); 64];
            for j in 0..8usize {
                let i = if j & 1 == 1 {
                    (j & 1) | ((j >> 1) & 1) << 2 | ((j >> 2) & 1) << 1
                } else {
                    j
                };
                m[i * 8 + j] = o.clone();
            }
            m
        }
        _ => return Err(unsupported()),
    };
    Ok(m)
}

/// A dense `2^n` state vector over [`DZeta<N>`]: the exact evaluator for
/// circuits with phases at multiples of `2π/N`.
#[derive(Debug, Clone)]
pub struct CyclotomicState<const N: u32> {
    num_qubits: usize,
    amps: Vec<DZeta<N>>,
}

impl<const N: u32> CyclotomicState<N> {
    /// `|0…0⟩` on `num_qubits` qubits.
    ///
    /// # Errors
    ///
    /// [`Error::TooManyQubits`] past [`EXACT_MAX_QUBITS`], or
    /// [`Error::OutOfMemory`] when the resource guard refuses the vector.
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits > EXACT_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: num_qubits,
                max: EXACT_MAX_QUBITS,
            });
        }
        let mut amps = crate::guard::try_vec(
            1usize << num_qubits,
            DZeta::zero(),
            &format!("D[ζ_{N}] state ({num_qubits} qubits)"),
        )?;
        amps[0] = DZeta::int(1);
        Ok(CyclotomicState { num_qubits, amps })
    }

    /// Evaluate a circuit exactly: named gates through the exact gate
    /// table, diagonal kernels whose entries are `N`-th roots of unity.
    ///
    /// # Errors
    ///
    /// [`Error::UnsupportedForAlgebra`] naming the first gate outside
    /// `D[ζ_N]` (raw float matrices always are), or
    /// [`Error::InvalidState`] on coefficient overflow.
    pub fn run(circuit: &Circuit<C64>) -> Result<Self> {
        let _scope = crate::guard::enter();
        let mut state = Self::new(circuit.num_qubits())?;
        for op in circuit.ops() {
            match op {
                Op::Named {
                    name,
                    params,
                    qubits,
                } => state.apply(&gate_matrix::<N>(name, params)?, qubits)?,
                Op::Diagonal {
                    label,
                    entries,
                    qubits,
                } => {
                    let exact: Option<Vec<DZeta<N>>> = entries
                        .iter()
                        .map(|&e| DZeta::from_root_of_unity(e))
                        .collect();
                    let exact = exact.ok_or_else(|| Error::UnsupportedForAlgebra {
                        gate: format!("diagonal '{label}' (entry not an {N}-th root of unity)"),
                        algebra: format!("D[ζ_{N}] (exact)"),
                    })?;
                    state.apply_diagonal(&exact, qubits)?;
                }
                Op::Raw { label, .. } => {
                    return Err(Error::UnsupportedForAlgebra {
                        gate: format!("raw '{label}' (float matrix)"),
                        algebra: format!("D[ζ_{N}] (exact)"),
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
    ///
    /// # Errors
    ///
    /// Target validation errors, [`Error::BadDimension`], or overflow.
    pub fn apply(&mut self, matrix: &[DZeta<N>], qubits: &[usize]) -> Result<()> {
        validate_targets(self.num_qubits, qubits)?;
        let d = 1usize << qubits.len();
        if matrix.len() != d * d {
            return Err(Error::BadDimension {
                expected: d,
                got: (matrix.len() as f64).sqrt() as usize,
            });
        }
        let mut sorted = qubits.to_vec();
        sorted.sort_unstable();
        let scatter = scatter_table(qubits);
        let mut scratch = vec![DZeta::zero(); d];
        for g in 0..self.amps.len() >> qubits.len() {
            if g % (1 << 16) == 0 {
                crate::guard::checkpoint()?;
            }
            let base = expand_index(g as u64, &sorted);
            for (j, slot) in scratch.iter_mut().enumerate() {
                *slot =
                    std::mem::replace(&mut self.amps[(base | scatter[j]) as usize], DZeta::zero());
            }
            for r in 0..d {
                let mut acc = DZeta::zero();
                for (s, e) in scratch.iter().zip(&matrix[r * d..(r + 1) * d]) {
                    if !e.is_zero() && !s.is_zero() {
                        acc = acc.add(&e.mul(s)?)?;
                    }
                }
                self.amps[(base | scatter[r]) as usize] = acc;
            }
        }
        Ok(())
    }

    /// Apply an exact diagonal (little-endian sub-index over `qubits`).
    ///
    /// # Errors
    ///
    /// Target validation errors, [`Error::BadDimension`], or overflow.
    pub fn apply_diagonal(&mut self, entries: &[DZeta<N>], qubits: &[usize]) -> Result<()> {
        validate_targets(self.num_qubits, qubits)?;
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
            let sub = qubits
                .iter()
                .enumerate()
                .fold(0usize, |acc, (b, &q)| acc | ((i >> q) & 1) << b);
            *a = entries[sub].mul(a)?;
        }
        Ok(())
    }

    /// The exact amplitude of a basis state.
    pub fn amplitude_exact(&self, index: u64) -> DZeta<N> {
        self.amps
            .get(index as usize)
            .cloned()
            .unwrap_or_else(DZeta::zero)
    }

    /// The amplitude as floating point (one rounding of the exact value).
    pub fn amplitude_c64(&self, index: u64) -> C64 {
        self.amplitude_exact(index).to_c64()
    }

    /// The exact Born weight of a basis state.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidState`] on overflow.
    pub fn probability_exact(&self, index: u64) -> Result<DZeta<N>> {
        self.amplitude_exact(index).born_weight()
    }

    /// The exact total Born weight: `1` for a unitary circuit, exactly.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidState`] on overflow.
    pub fn total_weight_exact(&self) -> Result<DZeta<N>> {
        let mut total = DZeta::zero();
        for a in self.amps.iter().filter(|a| !a.is_zero()) {
            total = total.add(&a.born_weight()?)?;
        }
        Ok(total)
    }

    /// Number of exactly nonzero amplitudes.
    pub fn support_exact(&self) -> usize {
        self.amps.iter().filter(|a| !a.is_zero()).count()
    }

    /// Largest absolute amplitude deviation of a backend's state from the
    /// exact values.
    pub fn max_deviation_vs(&self, state: &dyn Backend<C64>) -> f64 {
        self.amps
            .iter()
            .enumerate()
            .map(|(i, a)| (state.amplitude(i as u64) - a.to_c64()).norm())
            .fold(0.0, f64::max)
    }
}
