//! Registers wider than a `u64` basis index, and the modular arithmetic
//! they need.
//!
//! Every [`Backend`](crate::backend::Backend) in the crate indexes basis
//! states by `u64`, which caps a register at 63 qubits. For most
//! representations that ceiling is academic — they cannot hold 63 qubits
//! of state anyway. For a **permutation kernel on a sparse support** it is
//! the only thing in the way, and it is in the way of exactly the claim
//! [`shor`](crate::shor) is built to make: modular exponentiation is a
//! basis permutation, so it costs `O(support)` at any width, and the
//! support is the multiplicative order `r`. Neither quantity mentions the
//! modulus's bit length. A register that cannot *address* a wide modulus
//! cannot demonstrate that.
//!
//! So this module supplies the index type instead of widening the trait:
//! a [`WideRegister`] keyed by [`Wide`], an arbitrary-precision unsigned
//! integer. It is deliberately **not** a `Backend` — the same reason
//! [`pathsum`](crate::pathsum) and [`query`](crate::query) are not. The
//! trait's contract is `u64`-indexed and widening it would change
//! nineteen implementors to serve one.
//!
//! ## The arithmetic
//!
//! Reduction is Montgomery ([`Montgomery`]), not division: `REDC` needs
//! only multiply-accumulate over limbs, so the module never implements a
//! multi-precision divide. The register holds Montgomery representatives
//! throughout, which is a relabelling of `(ℤ/N)*` by the bijection
//! `y ↦ yR mod N` — the orbit structure, and therefore the order `r` and
//! the whole period-finding argument, is identical under it. Only the
//! final readout converts back.
//!
//! Montgomery requires an odd modulus. Every modulus factoring cares
//! about is odd (an even one is split by inspection), and an even one is
//! refused by name.

use std::cmp::Ordering;

use rustc_hash::FxHashMap;

use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::rng::Prng;
use crate::scalar::Scalar;

/// An arbitrary-precision unsigned integer, little-endian in 64-bit
/// limbs, kept normalized (no trailing zero limb).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Wide {
    limbs: Vec<u64>,
}

impl Wide {
    /// Zero.
    pub fn zero() -> Self {
        Wide { limbs: Vec::new() }
    }

    /// One.
    pub fn one() -> Self {
        Wide::from_u64(1)
    }

    /// From a `u64`.
    pub fn from_u64(v: u64) -> Self {
        let mut w = Wide { limbs: vec![v] };
        w.normalize();
        w
    }

    /// From little-endian limbs.
    pub fn from_limbs(limbs: Vec<u64>) -> Self {
        let mut w = Wide { limbs };
        w.normalize();
        w
    }

    /// Little-endian limbs, normalized.
    pub fn limbs(&self) -> &[u64] {
        &self.limbs
    }

    /// The value as a `u64`, or `None` if it does not fit.
    pub fn to_u64(&self) -> Option<u64> {
        match self.limbs.len() {
            0 => Some(0),
            1 => Some(self.limbs[0]),
            _ => None,
        }
    }

    fn normalize(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    /// Whether the value is zero.
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Number of significant bits.
    pub fn bits(&self) -> usize {
        match self.limbs.last() {
            None => 0,
            Some(&top) => self.limbs.len() * 64 - top.leading_zeros() as usize,
        }
    }

    /// Bit `i`, counting from the least significant.
    pub fn bit(&self, i: usize) -> bool {
        let (limb, off) = (i / 64, i % 64);
        self.limbs.get(limb).is_some_and(|&v| (v >> off) & 1 == 1)
    }

    /// Set bit `i` to `value`.
    pub fn set_bit(&mut self, i: usize, value: bool) {
        let (limb, off) = (i / 64, i % 64);
        if value {
            if self.limbs.len() <= limb {
                self.limbs.resize(limb + 1, 0);
            }
            self.limbs[limb] |= 1u64 << off;
        } else if limb < self.limbs.len() {
            self.limbs[limb] &= !(1u64 << off);
            self.normalize();
        }
    }

    /// The low `n` bits, as a value.
    pub fn low_bits(&self, n: usize) -> Wide {
        let full = n / 64;
        let rem = n % 64;
        let mut out: Vec<u64> = self.limbs.iter().take(full).copied().collect();
        if rem > 0 {
            let extra = self.limbs.get(full).copied().unwrap_or(0) & ((1u64 << rem) - 1);
            out.push(extra);
        }
        Wide::from_limbs(out)
    }

    /// Replace the low `n` bits with `value` (which must fit in `n` bits).
    pub fn with_low_bits(&self, n: usize, value: &Wide) -> Wide {
        let mut out = self.clone();
        for i in 0..n {
            out.set_bit(i, value.bit(i));
        }
        out
    }

    /// `self + rhs`.
    pub fn add(&self, rhs: &Wide) -> Wide {
        let len = self.limbs.len().max(rhs.limbs.len());
        let mut out = Vec::with_capacity(len + 1);
        let mut carry = 0u64;
        for i in 0..len {
            let a = self.limbs.get(i).copied().unwrap_or(0) as u128;
            let b = rhs.limbs.get(i).copied().unwrap_or(0) as u128;
            let s = a + b + carry as u128;
            out.push(s as u64);
            carry = (s >> 64) as u64;
        }
        if carry != 0 {
            out.push(carry);
        }
        Wide::from_limbs(out)
    }

    /// `self − rhs`, which panics if `rhs > self`.
    pub fn sub(&self, rhs: &Wide) -> Wide {
        debug_assert!(self >= rhs, "Wide::sub would go negative");
        let mut out = Vec::with_capacity(self.limbs.len());
        let mut borrow = 0i128;
        for i in 0..self.limbs.len() {
            let a = self.limbs[i] as i128;
            let b = rhs.limbs.get(i).copied().unwrap_or(0) as i128;
            let mut d = a - b - borrow;
            if d < 0 {
                d += 1i128 << 64;
                borrow = 1;
            } else {
                borrow = 0;
            }
            out.push(d as u64);
        }
        Wide::from_limbs(out)
    }

    /// `self · 2`.
    pub fn double(&self) -> Wide {
        self.add(self)
    }

    /// `self / d` and `self mod d` for a single-limb divisor.
    ///
    /// Multi-limb division is never needed — Montgomery removes it from
    /// the reduction path — but dividing by a *small* known factor is,
    /// wherever an exponent has to be walked down to a chosen order.
    pub fn div_small(&self, d: u64) -> (Wide, u64) {
        assert!(d != 0, "division by zero");
        let mut quotient = vec![0u64; self.limbs.len()];
        let mut rem = 0u128;
        for i in (0..self.limbs.len()).rev() {
            let acc = (rem << 64) | self.limbs[i] as u128;
            quotient[i] = (acc / d as u128) as u64;
            rem = acc % d as u128;
        }
        (Wide::from_limbs(quotient), rem as u64)
    }

    /// Schoolbook `self · rhs`.
    pub fn mul(&self, rhs: &Wide) -> Wide {
        if self.is_zero() || rhs.is_zero() {
            return Wide::zero();
        }
        let mut out = vec![0u64; self.limbs.len() + rhs.limbs.len()];
        for (i, &a) in self.limbs.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &b) in rhs.limbs.iter().enumerate() {
                let cur = out[i + j] as u128 + a as u128 * b as u128 + carry;
                out[i + j] = cur as u64;
                carry = cur >> 64;
            }
            let mut k = i + rhs.limbs.len();
            while carry != 0 {
                let cur = out[k] as u128 + carry;
                out[k] = cur as u64;
                carry = cur >> 64;
                k += 1;
            }
        }
        Wide::from_limbs(out)
    }
}

impl Ord for Wide {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.limbs.len().cmp(&other.limbs.len()) {
            Ordering::Equal => {
                for i in (0..self.limbs.len()).rev() {
                    match self.limbs[i].cmp(&other.limbs[i]) {
                        Ordering::Equal => continue,
                        o => return o,
                    }
                }
                Ordering::Equal
            }
            o => o,
        }
    }
}

impl PartialOrd for Wide {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl std::fmt::Display for Wide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        // Repeated division by 10^19, the largest power of ten in a u64.
        let mut chunks: Vec<u64> = Vec::new();
        let mut cur = self.limbs.clone();
        const BASE: u128 = 10_000_000_000_000_000_000;
        while cur.iter().any(|&l| l != 0) {
            let mut rem = 0u128;
            for limb in cur.iter_mut().rev() {
                let acc = (rem << 64) | *limb as u128;
                *limb = (acc / BASE) as u64;
                rem = acc % BASE;
            }
            chunks.push(rem as u64);
            while cur.last() == Some(&0) {
                cur.pop();
            }
        }
        let mut s = chunks.pop().unwrap_or(0).to_string();
        while let Some(c) = chunks.pop() {
            s.push_str(&format!("{c:019}"));
        }
        write!(f, "{s}")
    }
}

/// Montgomery arithmetic modulo an odd [`Wide`].
///
/// `R = 2^{64k}` for a `k`-limb modulus. Values are held as `xR mod N`;
/// [`Montgomery::mul`] computes `abR⁻¹ mod N`, so a product of two
/// representatives is the representative of the product. No division is
/// implemented anywhere: `R mod N` and `R² mod N` are built by repeated
/// doubling with conditional subtraction, which is the only place the
/// modulus is reduced against directly.
#[derive(Clone, Debug)]
pub struct Montgomery {
    modulus: Wide,
    limbs: usize,
    n0inv: u64,
    r_mod_n: Wide,
    r2_mod_n: Wide,
}

impl Montgomery {
    /// Build a context for an odd modulus greater than 1.
    pub fn new(modulus: Wide) -> Result<Self> {
        if modulus <= Wide::one() {
            return Err(Error::InvalidState(
                "Montgomery modulus must exceed 1".into(),
            ));
        }
        if !modulus.bit(0) {
            return Err(Error::InvalidState(
                "Montgomery needs an odd modulus; an even one is split by inspection".into(),
            ));
        }
        let limbs = modulus.limbs().len();
        // −N⁻¹ mod 2^64 by Newton iteration on the low limb.
        let n0 = modulus.limbs()[0];
        let mut inv = 1u64;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(n0.wrapping_mul(inv)));
        }
        let n0inv = inv.wrapping_neg();

        // R mod N and R² mod N by doubling, `64·limbs` times each.
        let mut r = Wide::one();
        for _ in 0..64 * limbs {
            r = r.double();
            if r >= modulus {
                r = r.sub(&modulus);
            }
        }
        let mut r2 = r.clone();
        for _ in 0..64 * limbs {
            r2 = r2.double();
            if r2 >= modulus {
                r2 = r2.sub(&modulus);
            }
        }
        Ok(Montgomery {
            modulus,
            limbs,
            n0inv,
            r_mod_n: r,
            r2_mod_n: r2,
        })
    }

    /// The modulus.
    pub fn modulus(&self) -> &Wide {
        &self.modulus
    }

    /// Limb count of the modulus.
    pub fn limbs(&self) -> usize {
        self.limbs
    }

    /// `1` in Montgomery form, i.e. `R mod N`.
    pub fn one(&self) -> Wide {
        self.r_mod_n.clone()
    }

    /// `abR⁻¹ mod N` by CIOS.
    // The inner loops walk `t`, the modulus limbs and the (possibly
    // shorter, hence `get`-with-default) operand limbs together at
    // different offsets, so an index is the clear form here.
    #[allow(clippy::needless_range_loop)]
    pub fn mul(&self, a: &Wide, b: &Wide) -> Wide {
        let k = self.limbs;
        let n = self.modulus.limbs();
        let mut t = vec![0u64; k + 2];
        for i in 0..k {
            let bi = b.limbs().get(i).copied().unwrap_or(0) as u128;
            let mut carry = 0u128;
            for j in 0..k {
                let aj = a.limbs().get(j).copied().unwrap_or(0) as u128;
                let cur = t[j] as u128 + aj * bi + carry;
                t[j] = cur as u64;
                carry = cur >> 64;
            }
            let cur = t[k] as u128 + carry;
            t[k] = cur as u64;
            t[k + 1] = (cur >> 64) as u64;

            let m = (t[0] as u128 * self.n0inv as u128) as u64;
            let cur = t[0] as u128 + m as u128 * n[0] as u128;
            let mut carry = cur >> 64;
            for j in 1..k {
                let cur = t[j] as u128 + m as u128 * n[j] as u128 + carry;
                t[j - 1] = cur as u64;
                carry = cur >> 64;
            }
            let cur = t[k] as u128 + carry;
            t[k - 1] = cur as u64;
            t[k] = t[k + 1] + (cur >> 64) as u64;
        }
        let mut out = Wide::from_limbs(t[..=k].to_vec());
        if out >= self.modulus {
            out = out.sub(&self.modulus);
        }
        out
    }

    /// `x → xR mod N`.
    pub fn to_montgomery(&self, x: &Wide) -> Wide {
        self.mul(x, &self.r2_mod_n)
    }

    /// `x → xR⁻¹ mod N`, the inverse of [`to_montgomery`](Self::to_montgomery).
    pub fn from_montgomery(&self, x: &Wide) -> Wide {
        self.mul(x, &Wide::one())
    }

    /// `base^exp mod N`, with both argument and result in ordinary form.
    pub fn pow(&self, base: &Wide, exp: &Wide) -> Wide {
        let mut acc = self.one();
        let mut b = self.to_montgomery(base);
        for i in 0..exp.bits() {
            if exp.bit(i) {
                acc = self.mul(&acc, &b);
            }
            b = self.mul(&b, &b);
        }
        self.from_montgomery(&acc)
    }
}

// ── the register ─────────────────────────────────────────────────────

/// A sparse state over an arbitrary number of qubits, keyed by [`Wide`].
///
/// The whole point is the index type. Amplitudes are held in a hash map
/// exactly as [`SparseState`](crate::backend::SparseState) holds them,
/// and every cost is `O(support)` — but the support is addressed by a
/// multi-limb integer, so the register has no width ceiling at all. A
/// 2048-qubit register holding two amplitudes costs two entries.
///
/// It is not a [`Backend`](crate::backend::Backend): that trait's
/// `amplitude`, `for_each_nonzero` and `load` are `u64`-indexed, and
/// widening them would change nineteen implementors to serve one.
#[derive(Clone, Debug)]
pub struct WideRegister<S: Scalar> {
    qubits: usize,
    map: FxHashMap<Wide, S>,
}

impl<S: Scalar> WideRegister<S> {
    /// `|0…0⟩` on `qubits` qubits, at any width.
    pub fn new(qubits: usize) -> Self {
        let mut map = FxHashMap::default();
        map.insert(Wide::zero(), S::one());
        WideRegister { qubits, map }
    }

    /// Register width.
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    /// Nonzero amplitudes held.
    pub fn nonzero_count(&self) -> usize {
        self.map.len()
    }

    /// Heap bytes, counting each key's limbs.
    pub fn memory_bytes(&self) -> usize {
        let entry = std::mem::size_of::<(Wide, S)>() + 1;
        let limbs: usize = self.map.keys().map(|k| k.limbs().len() * 8).sum();
        std::mem::size_of::<Self>() + self.map.len() * entry * 8 / 7 + limbs
    }

    /// The stored amplitudes.
    pub fn entries(&self) -> impl Iterator<Item = (&Wide, S)> + '_ {
        self.map.iter().map(|(k, &v)| (k, v))
    }

    /// Amplitude at a basis index.
    pub fn amplitude(&self, index: &Wide) -> S {
        self.map.get(index).copied().unwrap_or_else(S::zero)
    }

    /// Replace the state with an explicit amplitude list.
    pub fn load(&mut self, entries: Vec<(Wide, S)>) -> Result<()> {
        self.map.clear();
        for (i, a) in entries {
            if i.bits() > self.qubits {
                return Err(Error::InvalidState(format!(
                    "index needs {} bits, past the {}-qubit register",
                    i.bits(),
                    self.qubits
                )));
            }
            if a.abs_sqr() > 0.0 {
                self.map.insert(i, a);
            }
        }
        Ok(())
    }

    /// Apply a basis permutation in place, in `O(support)`.
    ///
    /// Injectivity is checked, as in
    /// [`shor::apply_permutation`](crate::shor::apply_permutation): a map
    /// that collides is not unitary and is refused rather than quietly
    /// losing weight.
    pub fn apply_permutation(&mut self, label: &str, map: &dyn Fn(&Wide) -> Wide) -> Result<usize> {
        let old = std::mem::take(&mut self.map);
        let mut next: FxHashMap<Wide, S> = FxHashMap::default();
        next.reserve(old.len());
        for (step, (i, a)) in old.into_iter().enumerate() {
            if step % 4096 == 0 {
                crate::guard::checkpoint()?;
            }
            let j = map(&i);
            if j.bits() > self.qubits {
                return Err(Error::InvalidState(format!(
                    "{label} sent an index to {} bits, past the {}-qubit register",
                    j.bits(),
                    self.qubits
                )));
            }
            if next.insert(j, a).is_some() {
                return Err(Error::InvalidState(format!(
                    "{label} is not injective: two indices share an image"
                )));
            }
        }
        self.map = next;
        Ok(self.map.len())
    }

    /// Apply a one-qubit gate.
    pub fn apply_1q(&mut self, matrix: &GateMatrix<S>, qubit: usize) -> Result<()> {
        if matrix.dim() != 2 {
            return Err(Error::BadDimension {
                expected: 2,
                got: matrix.dim(),
            });
        }
        if qubit >= self.qubits {
            return Err(Error::QubitOutOfRange {
                qubit,
                num_qubits: self.qubits,
            });
        }
        crate::guard::checkpoint()?;
        let old = std::mem::take(&mut self.map);
        let mut pairs: FxHashMap<Wide, [S; 2]> = FxHashMap::default();
        pairs.reserve(old.len());
        for (i, a) in old {
            let bit = i.bit(qubit);
            let mut base = i;
            base.set_bit(qubit, false);
            let slot = pairs.entry(base).or_insert([S::zero(), S::zero()]);
            slot[usize::from(bit)] = a;
        }
        let mut next: FxHashMap<Wide, S> = FxHashMap::default();
        next.reserve(pairs.len() * 2);
        for (base, [a0, a1]) in pairs {
            let out0 = matrix.get(0, 0) * a0 + matrix.get(0, 1) * a1;
            let out1 = matrix.get(1, 0) * a0 + matrix.get(1, 1) * a1;
            if out0.abs_sqr() > 0.0 {
                next.insert(base.clone(), out0);
            }
            if out1.abs_sqr() > 0.0 {
                let mut one = base;
                one.set_bit(qubit, true);
                next.insert(one, out1);
            }
        }
        self.map = next;
        Ok(())
    }

    /// Total Born weight.
    pub fn total_weight(&self) -> f64 {
        self.map.values().map(|a| a.born_weight()).sum()
    }

    /// Measure one qubit, collapsing and renormalizing.
    pub fn measure(&mut self, qubit: usize, rng: &mut Prng) -> Result<bool> {
        if qubit >= self.qubits {
            return Err(Error::QubitOutOfRange {
                qubit,
                num_qubits: self.qubits,
            });
        }
        let mut total = 0.0;
        let mut one_weight = 0.0;
        for (i, a) in &self.map {
            let w = a.born_weight();
            total += w;
            if i.bit(qubit) {
                one_weight += w;
            }
        }
        if total <= 0.0 || !total.is_finite() {
            return Err(Error::InvalidState(format!(
                "total Born weight {total} is not positive; cannot sample a measurement"
            )));
        }
        let p_one = (one_weight / total).clamp(0.0, 1.0);
        let outcome = rng.next_f64() < p_one;
        let p_outcome = if outcome { p_one } else { 1.0 - p_one };
        let renorm = 1.0 / (p_outcome * total).sqrt();
        self.map.retain(|i, _| i.bit(qubit) == outcome);
        for a in self.map.values_mut() {
            *a = a.scale(renorm);
        }
        Ok(outcome)
    }
}
