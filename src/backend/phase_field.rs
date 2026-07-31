//! Phase-field backend: the state as an **exact phase polynomial over
//! `ℤ/M`** on an affine subcube, promoting to dense the moment a gate
//! leaves that class.
//!
//! # The represented class
//!
//! A state is in *phase-field form* when it is
//!
//! ```text
//! |ψ⟩ = scale · Σ_{x ∈ C} ω_M^{P(x)} |x⟩,     ω_M = e^{2πi/M}
//! ```
//!
//! where `C` is an affine subcube (each qubit is either **free** or
//! **pinned** to a known bit) and `P` is a multilinear polynomial in the
//! free bits with coefficients in `ℤ/M`. Every amplitude has the same
//! modulus and the entire phase structure lives in `P`.
//!
//! Three things follow, and they are the reason this is a representation
//! rather than a special case:
//!
//! * **Storage is the monomial count.** `P` is stored as a
//!   `monomial → coefficient` table, so a state on `n` free qubits with
//!   `m` monomials costs `O(m)` and not `2^n`. A depth-`d` diagonal
//!   circuit of `k`-local gates produces at most `d·2^k` monomials
//!   regardless of the width.
//! * **The phase is exact.** Coefficients are integers mod `M`; nothing
//!   is a float until an amplitude is asked for. Two circuits whose
//!   phases cancel cancel *exactly*, and the modulus is grown by `lcm`
//!   as gates demand it — which is the [`padic`](crate::padic) radix
//!   doing its job inside a backend.
//! * **The class boundary is exactly where the physics is.** `h` on a
//!   *pinned* qubit frees it and stays in the class; diagonal gates whose
//!   entries are roots of unity stay in the class; `h` on an already-free
//!   qubit — the interference step — leaves it, and so does any gate that
//!   is not diagonal-with-root-of-unity-entries. So an IQP core is held
//!   in monomials and the final Hadamard layer is exactly the moment the
//!   representation gives up. That is not a limitation smuggled in; it is
//!   the same boundary the sampling-hardness measurements sit on.
//!
//! Anything outside the class **materializes to dense** and the backend
//! keeps running, exactly as [`AdaptiveState`](crate::backend::AdaptiveState)
//! promotes sparse → dense. Conformance therefore holds over the whole
//! registry, and [`PhaseFieldState::is_field`] reports which side of the
//! boundary a given run ended on rather than leaving it to be inferred.

use std::collections::HashMap;

use super::{validate_apply, validate_apply_diagonal, Backend, DenseState};
use crate::backend::dense::DENSE_MAX_QUBITS;
use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::padic::lcm;
use crate::scalar::C64;

/// Largest root-of-unity order a single gate entry may have before the
/// state is materialized. Covers Clifford+T (8), every eighth-turn
/// rotation, and the qudit phases the crate ships.
pub const MAX_ROOT_ORDER: u64 = 4096;

/// Largest phase modulus the field form will grow to by `lcm` before
/// giving up and materializing.
pub const MAX_FIELD_MODULUS: u64 = 1 << 34;

/// Tolerance for recognizing an entry as an exact root of unity.
const ROOT_TOL: f64 = 1e-9;

/// The phase polynomial and the subcube it lives on.
#[derive(Debug, Clone)]
struct Field {
    /// Bit `q` set ⇒ qubit `q` is pinned (not in superposition).
    pinned_mask: u64,
    /// Values of the pinned qubits (bits outside `pinned_mask` are 0).
    pinned_bits: u64,
    /// Phase modulus `M`; always even, so a sign is representable.
    modulus: u64,
    /// `monomial bitmask over free qubits → coefficient in ℤ/M`. The
    /// empty mask is the constant term.
    poly: HashMap<u64, u64>,
    /// Global complex scale, carrying normalization and any non-phase
    /// factor a projection introduced.
    scale: C64,
}

impl Field {
    fn new(n: usize) -> Self {
        let mask = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
        Field {
            pinned_mask: mask,
            pinned_bits: 0,
            modulus: 8,
            poly: HashMap::new(),
            scale: C64::new(1.0, 0.0),
        }
    }

    fn is_zero(&self) -> bool {
        self.scale.norm() == 0.0
    }

    /// Evaluate `P` at a basis index (only free bits matter).
    fn evaluate(&self, index: u64) -> u64 {
        let mut acc: u128 = 0;
        for (&mono, &coeff) in &self.poly {
            if index & mono == mono {
                acc += coeff as u128;
            }
        }
        (acc % self.modulus as u128) as u64
    }

    fn amplitude(&self, index: u64) -> C64 {
        if self.is_zero() || (index & self.pinned_mask) != self.pinned_bits {
            return C64::new(0.0, 0.0);
        }
        let e = self.evaluate(index);
        self.scale * crate::math::cis(std::f64::consts::TAU * e as f64 / self.modulus as f64)
    }

    /// Rescale every coefficient into a larger modulus.
    fn rescale(&mut self, new_modulus: u64) {
        if new_modulus == self.modulus {
            return;
        }
        let factor = new_modulus / self.modulus;
        for c in self.poly.values_mut() {
            *c = (*c * factor) % new_modulus;
        }
        self.modulus = new_modulus;
    }

    fn add_term(&mut self, mono: u64, coeff: u64) {
        if coeff % self.modulus == 0 {
            return;
        }
        let e = self.poly.entry(mono).or_insert(0);
        *e = (*e + coeff) % self.modulus;
        if *e == 0 {
            self.poly.remove(&mono);
        }
    }

    fn free_count(&self, n: usize) -> usize {
        let mask = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
        (mask & !self.pinned_mask).count_ones() as usize
    }
}

#[derive(Debug, Clone)]
enum Repr {
    Field(Field),
    Dense(DenseState<C64>),
}

/// A state held as an exact phase polynomial while the circuit stays in
/// the phase-field class, and as a dense vector once it does not.
///
/// See the [`phase_field`](crate::backend) module docs for the class and
/// why its boundary is the interesting part.
#[derive(Debug, Clone)]
pub struct PhaseFieldState {
    n: usize,
    repr: Repr,
    /// Gates that forced materialization, for the record.
    escapes: usize,
}

impl PhaseFieldState {
    /// `|0…0⟩` on `num_qubits` qubits, in phase-field form.
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits > DENSE_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: num_qubits,
                max: DENSE_MAX_QUBITS,
            });
        }
        Ok(PhaseFieldState {
            n: num_qubits,
            repr: Repr::Field(Field::new(num_qubits)),
            escapes: 0,
        })
    }

    /// Whether the state is still in phase-field form.
    pub fn is_field(&self) -> bool {
        matches!(self.repr, Repr::Field(_))
    }

    /// Monomials currently stored; `None` once materialized.
    pub fn monomials(&self) -> Option<usize> {
        match &self.repr {
            Repr::Field(f) => Some(f.poly.len()),
            Repr::Dense(_) => None,
        }
    }

    /// The current phase modulus `M`; `None` once materialized.
    pub fn modulus(&self) -> Option<u64> {
        match &self.repr {
            Repr::Field(f) => Some(f.modulus),
            Repr::Dense(_) => None,
        }
    }

    /// Qubits currently in superposition; `None` once materialized.
    pub fn free_qubits(&self) -> Option<usize> {
        match &self.repr {
            Repr::Field(f) => Some(f.free_count(self.n)),
            Repr::Dense(_) => None,
        }
    }

    /// How many gates left the phase-field class.
    pub fn escapes(&self) -> usize {
        self.escapes
    }

    /// Force the dense representation, keeping the state exactly.
    fn materialize(&mut self) -> Result<()> {
        if let Repr::Field(f) = &self.repr {
            let mut dense = DenseState::<C64>::new(self.n)?;
            if !f.is_zero() {
                let total = 1u64 << self.n;
                let entries: Vec<(u64, C64)> = (0..total)
                    .filter(|&i| (i & f.pinned_mask) == f.pinned_bits)
                    .map(|i| (i, f.amplitude(i)))
                    .collect();
                dense.load(&entries)?;
            }
            self.repr = Repr::Dense(dense);
            self.escapes += 1;
        }
        Ok(())
    }

    fn dense_mut(&mut self) -> Result<&mut DenseState<C64>> {
        self.materialize()?;
        match &mut self.repr {
            Repr::Dense(d) => Ok(d),
            Repr::Field(_) => unreachable!("materialize leaves a dense repr"),
        }
    }

    /// The smallest `q ≤ MAX_ROOT_ORDER` with `z = ω_q^k`, and that `k`.
    fn as_root(z: C64) -> Option<(u64, u64)> {
        if (z.norm() - 1.0).abs() > ROOT_TOL {
            return None;
        }
        let mut turn = z.im.atan2(z.re) / std::f64::consts::TAU;
        if turn < 0.0 {
            turn += 1.0;
        }
        for q in 1..=MAX_ROOT_ORDER {
            let k = turn * q as f64;
            let r = k.round();
            if (k - r).abs() < ROOT_TOL * q as f64 && r >= 0.0 {
                return Some((q, (r as u64) % q));
            }
        }
        None
    }

    /// Diagonal entries as root-of-unity exponents over a common order.
    fn diagonal_exponents(entries: &[C64]) -> Option<(u64, Vec<u64>)> {
        let mut order = 2u64;
        let mut roots = Vec::with_capacity(entries.len());
        for &z in entries {
            let (q, k) = Self::as_root(z)?;
            roots.push((q, k));
            order = lcm(order, q)?;
            if order > MAX_FIELD_MODULUS {
                return None;
            }
        }
        Some((
            order,
            roots.iter().map(|&(q, k)| k * (order / q)).collect(),
        ))
    }

    /// Whether a matrix is exactly the diagonal it looks like.
    fn as_diagonal(matrix: &GateMatrix<C64>) -> Option<Vec<C64>> {
        let d = matrix.dim();
        for r in 0..d {
            for c in 0..d {
                if r != c && matrix.get(r, c).norm() > ROOT_TOL {
                    return None;
                }
            }
        }
        Some((0..d).map(|i| matrix.get(i, i)).collect())
    }

    /// Whether a single-qubit matrix is the Hadamard.
    fn is_hadamard(matrix: &GateMatrix<C64>) -> bool {
        if matrix.dim() != 2 {
            return false;
        }
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let want = [
            C64::new(s, 0.0),
            C64::new(s, 0.0),
            C64::new(s, 0.0),
            C64::new(-s, 0.0),
        ];
        (0..4).all(|i| (matrix.data()[i] - want[i]).norm() < ROOT_TOL)
    }

    /// Apply a diagonal in the field form, or report that it cannot be.
    fn try_diagonal(&mut self, entries: &[C64], qubits: &[usize]) -> Result<bool> {
        let Repr::Field(f) = &mut self.repr else {
            return Ok(false);
        };
        let Some((order, exps)) = Self::diagonal_exponents(entries) else {
            return Ok(false);
        };
        let Some(modulus) = lcm(f.modulus, order.max(2)) else {
            return Ok(false);
        };
        if modulus > MAX_FIELD_MODULUS {
            return Ok(false);
        }
        f.rescale(modulus);
        let grow = modulus / order;

        // Split the targets into pinned (value known) and free.
        let mut free_targets = Vec::new();
        let mut pinned_contrib = 0usize;
        for (b, &q) in qubits.iter().enumerate() {
            if f.pinned_mask >> q & 1 == 1 {
                if f.pinned_bits >> q & 1 == 1 {
                    pinned_contrib |= 1 << b;
                }
            } else {
                free_targets.push((b, q));
            }
        }
        let k = free_targets.len();
        if k > 20 {
            return Ok(false);
        }

        // Restrict the exponent function to the free targets and
        // Möbius-invert into a multilinear polynomial.
        let mut restricted = vec![0u64; 1usize << k];
        for (s, slot) in restricted.iter_mut().enumerate() {
            let mut sub = pinned_contrib;
            for (j, &(b, _)) in free_targets.iter().enumerate() {
                if s >> j & 1 == 1 {
                    sub |= 1 << b;
                }
            }
            *slot = (exps[sub] * grow) % modulus;
        }
        // coeff[T] = Σ_{S ⊆ T} (−1)^{|T|−|S|} e[S]
        for j in 0..k {
            for t in 0..(1usize << k) {
                if t >> j & 1 == 1 {
                    let lower = restricted[t & !(1 << j)];
                    let cur = restricted[t];
                    restricted[t] = (cur + modulus - lower % modulus) % modulus;
                }
            }
        }
        for (t, &coeff) in restricted.iter().enumerate() {
            if coeff == 0 {
                continue;
            }
            let mut mono = 0u64;
            for (j, &(_, q)) in free_targets.iter().enumerate() {
                if t >> j & 1 == 1 {
                    mono |= 1 << q;
                }
            }
            f.add_term(mono, coeff);
        }
        Ok(true)
    }

    /// Apply a Hadamard that frees a pinned qubit, or report that it
    /// cannot be applied in the field form.
    fn try_free_hadamard(&mut self, qubit: usize) -> bool {
        let Repr::Field(f) = &mut self.repr else {
            return false;
        };
        if f.pinned_mask >> qubit & 1 == 0 {
            // already in superposition: this is the interference step,
            // and it is exactly what leaves the class
            return false;
        }
        let one = f.pinned_bits >> qubit & 1 == 1;
        f.pinned_mask &= !(1 << qubit);
        f.pinned_bits &= !(1 << qubit);
        f.scale *= C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        if one {
            let half = f.modulus / 2;
            f.add_term(1u64 << qubit, half);
        }
        true
    }
}

impl Backend<C64> for PhaseFieldState {
    fn name(&self) -> &str {
        "phase-field"
    }

    fn num_qubits(&self) -> usize {
        self.n
    }

    fn apply(&mut self, matrix: &GateMatrix<C64>, qubits: &[usize]) -> Result<()> {
        validate_apply(self.n, matrix, qubits)?;
        if self.is_field() {
            if let Some(diag) = Self::as_diagonal(matrix) {
                if self.try_diagonal(&diag, qubits)? {
                    return Ok(());
                }
            } else if qubits.len() == 1
                && Self::is_hadamard(matrix)
                && self.try_free_hadamard(qubits[0])
            {
                return Ok(());
            }
        }
        self.dense_mut()?.apply(matrix, qubits)
    }

    fn apply_diagonal(&mut self, entries: &[C64], qubits: &[usize]) -> Result<()> {
        validate_apply_diagonal(self.n, entries, qubits)?;
        if self.is_field() && self.try_diagonal(entries, qubits)? {
            return Ok(());
        }
        self.dense_mut()?.apply_diagonal(entries, qubits)
    }

    fn amplitude(&self, index: u64) -> C64 {
        match &self.repr {
            Repr::Field(f) => {
                if index >= 1u64 << self.n {
                    C64::new(0.0, 0.0)
                } else {
                    f.amplitude(index)
                }
            }
            Repr::Dense(d) => d.amplitude(index),
        }
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, C64)) {
        match &self.repr {
            Repr::Field(field) => {
                if field.is_zero() {
                    return;
                }
                let total = 1u64 << self.n;
                for i in 0..total {
                    if (i & field.pinned_mask) == field.pinned_bits {
                        f(i, field.amplitude(i));
                    }
                }
            }
            Repr::Dense(d) => d.for_each_nonzero(f),
        }
    }

    fn nonzero_count(&self) -> usize {
        match &self.repr {
            Repr::Field(f) => {
                if f.is_zero() {
                    0
                } else {
                    1usize << f.free_count(self.n)
                }
            }
            Repr::Dense(d) => d.nonzero_count(),
        }
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        if let Repr::Field(f) = &mut self.repr {
            if qubit >= self.n {
                return;
            }
            if f.pinned_mask >> qubit & 1 == 1 {
                // already pinned: agreeing keeps the state, disagreeing
                // annihilates it
                let bit = f.pinned_bits >> qubit & 1 == 1;
                if bit != outcome {
                    f.scale = C64::new(0.0, 0.0);
                    f.poly.clear();
                } else {
                    f.scale *= C64::new(renorm, 0.0);
                }
                return;
            }
            // substitute x_q = outcome into the polynomial
            let mask = 1u64 << qubit;
            let terms: Vec<(u64, u64)> = f.poly.iter().map(|(&m, &c)| (m, c)).collect();
            let modulus = f.modulus;
            f.poly.clear();
            for (m, c) in terms {
                if m & mask == 0 {
                    let e = f.poly.entry(m).or_insert(0);
                    *e = (*e + c) % modulus;
                } else if outcome {
                    let e = f.poly.entry(m & !mask).or_insert(0);
                    *e = (*e + c) % modulus;
                }
            }
            f.poly.retain(|_, c| *c % modulus != 0);
            f.pinned_mask |= mask;
            if outcome {
                f.pinned_bits |= mask;
            } else {
                f.pinned_bits &= !mask;
            }
            f.scale *= C64::new(renorm, 0.0);
            return;
        }
        if let Repr::Dense(d) = &mut self.repr {
            d.project(qubit, outcome, renorm);
        }
    }

    fn reset(&mut self) {
        self.repr = Repr::Field(Field::new(self.n));
    }

    fn load(&mut self, entries: &[(u64, C64)]) -> Result<()> {
        // an arbitrary amplitude list is outside the class by definition
        self.dense_mut()?.load(entries)
    }

    fn memory_bytes(&self) -> usize {
        match &self.repr {
            Repr::Field(f) => {
                std::mem::size_of::<Self>() + f.poly.len() * (2 * std::mem::size_of::<u64>() + 8)
            }
            Repr::Dense(d) => std::mem::size_of::<Self>() + d.memory_bytes(),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots_of_unity_are_recognized_exactly() {
        assert_eq!(PhaseFieldState::as_root(C64::new(1.0, 0.0)), Some((1, 0)));
        assert_eq!(PhaseFieldState::as_root(C64::new(-1.0, 0.0)), Some((2, 1)));
        assert_eq!(PhaseFieldState::as_root(C64::new(0.0, 1.0)), Some((4, 1)));
        // e^{iπ/4} is an eighth root
        let t = crate::math::cis(std::f64::consts::FRAC_PI_4);
        assert_eq!(PhaseFieldState::as_root(t), Some((8, 1)));
        // not on the unit circle, and not a root of small order
        assert_eq!(PhaseFieldState::as_root(C64::new(0.5, 0.0)), None);
        assert_eq!(PhaseFieldState::as_root(crate::math::cis(1.0)), None);
    }

    #[test]
    fn a_hadamard_layer_then_diagonals_stays_a_field() {
        let mut s = PhaseFieldState::new(6).unwrap();
        let h = GateMatrix::<C64>::try_from_c64s(
            2,
            &[
                C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                C64::new(-std::f64::consts::FRAC_1_SQRT_2, 0.0),
            ],
        )
        .unwrap();
        for q in 0..6 {
            s.apply(&h, &[q]).unwrap();
        }
        assert!(s.is_field());
        assert_eq!(s.free_qubits(), Some(6));
        // a cz on every pair: still a field, and the monomial count is
        // the pair count, not 2^6
        let cz = [
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(-1.0, 0.0),
        ];
        for a in 0..6 {
            for b in a + 1..6 {
                s.apply_diagonal(&cz, &[a, b]).unwrap();
            }
        }
        assert!(s.is_field());
        assert_eq!(s.monomials(), Some(15));
        assert_eq!(s.escapes(), 0);
        // and a second Hadamard on a free qubit is exactly what leaves
        s.apply(&h, &[0]).unwrap();
        assert!(!s.is_field());
        assert_eq!(s.escapes(), 1);
    }
}
