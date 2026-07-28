//! Sparse state-vector backend: hash map of nonzero amplitudes.

use rustc_hash::FxHashMap;

use super::{scatter_table, validate_apply, Backend, PRUNE_TOL};
use crate::error::{Error, Result};
use crate::math::{GateMatrix, MATRIX_ZERO_TOL};
use crate::scalar::Scalar;

/// Maximum width for the sparse representation (basis indices are `u64`).
pub const SPARSE_MAX_QUBITS: usize = 63;

/// Approximate bytes one stored map entry costs (key + value + control
/// byte at hashbrown's 7/8 load factor) — the unit sparse growth
/// admission is measured in.
fn entry_bytes<S>() -> usize {
    (std::mem::size_of::<u64>() + std::mem::size_of::<S>() + 1) * 8 / 7
}

/// Nonzero amplitudes stored in an `FxHashMap<u64, S>`.
///
/// Gate cost scales with the number of nonzero amplitudes times the gate
/// matrix's column fill — permutation-like gates (X, CX, SWAP) and diagonal
/// gates (Z, S, T, CP, RZZ) cause **no** fill, so circuits that stay
/// concentrated (GHZ, oracles, arithmetic) simulate in memory proportional
/// to the support size, far past the dense width limit. Superposition-heavy
/// circuits (H layers, QFT) fill up to `2^n` entries and are strictly worse
/// than dense — see the `adaptive` backend and the width benchmarks.
#[derive(Debug, Clone)]
pub struct SparseState<S: Scalar> {
    num_qubits: usize,
    map: FxHashMap<u64, S>,
}

impl<S: Scalar> SparseState<S> {
    /// `|0…0⟩` on `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits > SPARSE_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: num_qubits,
                max: SPARSE_MAX_QUBITS,
            });
        }
        let mut map = FxHashMap::default();
        map.insert(0u64, S::one());
        Ok(SparseState { num_qubits, map })
    }

    /// Read-only view of the stored amplitudes.
    pub fn entries(&self) -> impl Iterator<Item = (u64, S)> + '_ {
        self.map.iter().map(|(&i, &a)| (i, a))
    }

    /// Apply the Pauli-axis rotation `exp(−iθ/2 · σP)` natively, where `P`
    /// is the Hermitian Pauli string with X-support `x_mask`, Z-support
    /// `z_mask` (qubits in both are Y), and `σ = −1` when `negate` is set.
    ///
    /// Cost is `O(support)` with at most 2× support growth — independent of
    /// the string's weight, which is what makes Clifford-frame simulation
    /// pay: arbitrarily long conjugated axes stay cheap. Pure-Z strings are
    /// diagonal and grow nothing. Requires an algebra containing `i`
    /// (the rotation coefficient is `−i sin`); fails loudly otherwise.
    pub fn apply_pauli_rotation(
        &mut self,
        theta: f64,
        x_mask: u64,
        z_mask: u64,
        negate: bool,
    ) -> Result<()> {
        let width_mask = if self.num_qubits == 64 {
            u64::MAX
        } else {
            (1u64 << self.num_qubits) - 1
        };
        if x_mask & !width_mask != 0 || z_mask & !width_mask != 0 {
            return Err(Error::QubitOutOfRange {
                qubit: (64 - (x_mask | z_mask).leading_zeros()) as usize,
                num_qubits: self.num_qubits,
            });
        }
        let theta = if negate { -theta } else { theta };
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        if x_mask == 0 {
            // Diagonal: amp ← e^{−iθ/2·(−1)^{|idx∧z|}} · amp.
            let plus = S::try_from_c64(crate::math::c64(c, -s)).ok_or_else(|| {
                Error::UnsupportedForAlgebra {
                    gate: "pauli rotation".into(),
                    algebra: S::algebra_name(),
                }
            })?;
            let minus = S::try_from_c64(crate::math::c64(c, s)).ok_or_else(|| {
                Error::UnsupportedForAlgebra {
                    gate: "pauli rotation".into(),
                    algebra: S::algebra_name(),
                }
            })?;
            for (idx, a) in self.map.iter_mut() {
                let sign = (idx & z_mask).count_ones() & 1 == 1;
                *a = if sign { minus * *a } else { plus * *a };
            }
            return Ok(());
        }
        // ⟨idx⊕x| σP |idx⟩ per entry; cos keeps, −i·sin scatters. The
        // scatter coefficient is −i·s·i^k where k counts the element's
        // quarter-turns: Z contributes (−1)^bit on z-only qubits (k += 2),
        // Y contributes +i on |0⟩ and −i on |1⟩. Only two values of k occur
        // (its parity is fixed by the string's Y-count), so both are
        // precomputed — and the embedding failure over algebras without the
        // needed imaginary part (e.g. an X-string over ℝ, while Y-strings
        // stay real) surfaces before the state is touched.
        let cos_s = S::from_re(c);
        let y_mask = x_mask & z_mask;
        let z_only = z_mask & !x_mask;
        let y_total = y_mask.count_ones();
        let k_lo = y_total & 1;
        let unit = |k: u32| match k & 3 {
            0 => crate::math::c64(1.0, 0.0),
            1 => crate::math::c64(0.0, 1.0),
            2 => crate::math::c64(-1.0, 0.0),
            _ => crate::math::c64(0.0, -1.0),
        };
        let base = S::try_from_c64(unit(k_lo) * crate::math::c64(0.0, -s)).ok_or_else(|| {
            Error::UnsupportedForAlgebra {
                gate: "pauli rotation".into(),
                algebra: S::algebra_name(),
            }
        })?;
        crate::guard::admit_growth(
            self.map.len().saturating_mul(2 * entry_bytes::<S>()),
            "sparse Pauli-rotation growth",
        )?;
        let mut out = FxHashMap::default();
        out.reserve(self.map.len() * 2);
        for (&idx, &a) in &self.map {
            if c != 0.0 {
                let slot = out.entry(idx).or_insert_with(S::zero);
                *slot = *slot + cos_s * a;
            }
            if s != 0.0 {
                // i^{y_total} · (−i)^{2·y_ones} = i^{y_total − 2·y_ones}
                let y_ones = (idx & y_mask).count_ones();
                let mut k = (y_total as i32 - 2 * y_ones as i32).rem_euclid(4) as u32;
                if (idx & z_only).count_ones() & 1 == 1 {
                    k = (k + 2) & 3;
                }
                let coeff = if k == k_lo { base } else { -base };
                let slot = out.entry(idx ^ x_mask).or_insert_with(S::zero);
                *slot = *slot + coeff * a;
            }
        }
        out.retain(|_, a| a.abs_sqr() > PRUNE_TOL);
        self.map = out;
        Ok(())
    }

    /// Measure the Hermitian Pauli string `σP` natively: draw the ±1
    /// outcome from the (generalized) Born distribution, project onto the
    /// observed eigenspace with `(I ± σP)/2`, and renormalize the branch
    /// to weight 1. Returns `true` for the **−1** outcome — measuring
    /// `Z_q` returns `true` exactly when qubit `q` reads 1, matching
    /// [`Backend::measure`]'s bit and RNG-draw conventions so outcomes are
    /// seed-identical across backends.
    ///
    /// Cost is `O(support)`; projection never more than doubles support —
    /// it shrinks toward eigenstates of `P` but *grows* when the state is
    /// far from one, and repeated measurements compound either way. Same
    /// algebra rule as the rotations: the
    /// element phase `i^{|x∧z|}` must embed (even-Y strings are real,
    /// odd-Y strings need `i` — measuring Y on a rebit state genuinely
    /// has no real-valued post-measurement state).
    pub fn measure_pauli(
        &mut self,
        x_mask: u64,
        z_mask: u64,
        negate: bool,
        rng: &mut crate::rng::Prng,
    ) -> Result<bool> {
        let width_mask = if self.num_qubits == 64 {
            u64::MAX
        } else {
            (1u64 << self.num_qubits) - 1
        };
        if x_mask & !width_mask != 0 || z_mask & !width_mask != 0 {
            return Err(Error::QubitOutOfRange {
                qubit: (64 - (x_mask | z_mask).leading_zeros()) as usize,
                num_qubits: self.num_qubits,
            });
        }
        // P|j⟩ = σ·i^{|x∧z|}·(−1)^{|j∧z|}·|j⊕x⟩; only ±(σ·i^y) occurs.
        let y = (x_mask & z_mask).count_ones();
        let unit = |k: u32| match k & 3 {
            0 => crate::math::c64(1.0, 0.0),
            1 => crate::math::c64(0.0, 1.0),
            2 => crate::math::c64(-1.0, 0.0),
            _ => crate::math::c64(0.0, -1.0),
        };
        let sigma = if negate { -1.0 } else { 1.0 };
        let base = S::try_from_c64(unit(y) * crate::math::c64(sigma, 0.0)).ok_or_else(|| {
            Error::UnsupportedForAlgebra {
                gate: "pauli measurement".into(),
                algebra: S::algebra_name(),
            }
        })?;
        let phase_of = |j: u64| -> S {
            if (j & z_mask).count_ones() & 1 == 1 {
                -base
            } else {
                base
            }
        };
        // ⟨P⟩ = Σ_j conj(a[j⊕x])·ph(j)·a[j]; real for Hermitian P.
        let mut total = 0.0;
        let mut expectation = 0.0;
        for (&j, &a) in &self.map {
            total += a.born_weight();
            if let Some(&b) = self.map.get(&(j ^ x_mask)) {
                expectation += (b.conj() * (phase_of(j) * a)).re();
            }
        }
        if total <= 0.0 || !total.is_finite() {
            return Err(Error::InvalidState(format!(
                "total Born weight {total} is not positive; cannot sample a measurement"
            )));
        }
        let p_minus = ((total - expectation) / (2.0 * total)).clamp(0.0, 1.0);
        let outcome_minus = rng.next_f64() < p_minus;
        let eps = if outcome_minus { -1.0 } else { 1.0 };
        let branch_weight = if outcome_minus {
            p_minus * total
        } else {
            (1.0 - p_minus) * total
        };
        let renorm = 1.0 / branch_weight.sqrt();
        // out = (a + ε·P·a)/2 · renorm, entry-wise over the pair graph.
        let mut out = FxHashMap::default();
        out.reserve(self.map.len());
        for (&i, &a) in &self.map {
            let mut v = a;
            if let Some(&b) = self.map.get(&(i ^ x_mask)) {
                // (Pψ)[i] = ph(i⊕x)·a[i⊕x]
                let pa = phase_of(i ^ x_mask) * b;
                v = if eps > 0.0 { v + pa } else { v - pa };
            } else {
                // Partner absent: |i⊕x⟩ gains ε·ph(i)·a[i]/2 too.
                let pa = phase_of(i) * a;
                let w = if eps > 0.0 { pa } else { -pa };
                let slot = out.entry(i ^ x_mask).or_insert_with(S::zero);
                *slot = *slot + w.scale(0.5 * renorm);
            }
            let slot = out.entry(i).or_insert_with(S::zero);
            *slot = *slot + v.scale(0.5 * renorm);
        }
        out.retain(|_, a| a.abs_sqr() > PRUNE_TOL);
        self.map = out;
        Ok(outcome_minus)
    }
}

impl<S: Scalar> Backend<S> for SparseState<S> {
    fn name(&self) -> &str {
        "sparse"
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        validate_apply(self.num_qubits, matrix, qubits)?;
        let k = qubits.len();
        let d = 1usize << k;
        let scatter = scatter_table(qubits);
        let target_mask: u64 = qubits.iter().fold(0, |acc, &q| acc | (1u64 << q));

        // Column-wise nonzero structure: for input sub-index l, the output
        // sub-indices r with M[r][l] != 0. Permutation/diagonal gates have
        // exactly one entry per column, so they cause no fill.
        let mdata = matrix.data();
        let mut cols: Vec<Vec<(usize, S)>> = vec![Vec::new(); d];
        for r in 0..d {
            for (l, col) in cols.iter_mut().enumerate() {
                let e = mdata[r * d + l];
                if e.abs_sqr() > MATRIX_ZERO_TOL {
                    col.push((r, e));
                }
            }
        }

        // Worst-case fill is support × matrix column fill; admit that
        // growth against measured memory before building it.
        let max_fill = cols.iter().map(|c| c.len()).max().unwrap_or(1);
        crate::guard::admit_growth(
            self.map
                .len()
                .saturating_mul(max_fill)
                .saturating_mul(entry_bytes::<S>()),
            "sparse state growth",
        )?;
        let mut out: FxHashMap<u64, S> = FxHashMap::default();
        out.reserve(self.map.len());
        for (count, (&idx, &amp)) in self.map.iter().enumerate() {
            if count % (1 << 18) == 0 {
                crate::guard::checkpoint()?;
            }
            let rest = idx & !target_mask;
            let mut sub = 0usize;
            for (b, &q) in qubits.iter().enumerate() {
                sub |= (((idx >> q) & 1) as usize) << b;
            }
            for &(r, e) in &cols[sub] {
                let slot = out.entry(rest | scatter[r]).or_insert_with(S::zero);
                *slot = *slot + e * amp;
            }
        }
        out.retain(|_, a| a.abs_sqr() > PRUNE_TOL);
        self.map = out;
        Ok(())
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        super::validate_apply_diagonal(self.num_qubits, entries, qubits)?;
        for (&idx, a) in self.map.iter_mut() {
            *a = entries[super::sub_index(idx, qubits)] * *a;
        }
        Ok(())
    }

    fn amplitude(&self, index: u64) -> S {
        self.map.get(&index).copied().unwrap_or_else(S::zero)
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        for (&i, &a) in &self.map {
            f(i, a);
        }
    }

    fn nonzero_count(&self) -> usize {
        self.map.len()
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        let mask = 1u64 << qubit;
        self.map.retain(|&i, _| ((i & mask) != 0) == outcome);
        for a in self.map.values_mut() {
            *a = a.scale(renorm);
        }
    }

    fn reset(&mut self) {
        self.map.clear();
        self.map.insert(0, S::one());
    }

    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        let limit = if self.num_qubits == 64 {
            u64::MAX
        } else {
            1u64 << self.num_qubits
        };
        for &(i, _) in entries {
            if i >= limit {
                return Err(Error::QubitOutOfRange {
                    qubit: 64 - i.leading_zeros() as usize,
                    num_qubits: self.num_qubits,
                });
            }
        }
        self.map.clear();
        for &(i, a) in entries {
            if a.abs_sqr() > 0.0 {
                self.map.insert(i, a);
            }
        }
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        // hashbrown stores (key, value) slots plus one control byte each, at
        // 7/8 maximum load factor. An estimate, not an allocator audit.
        let slot = std::mem::size_of::<u64>() + std::mem::size_of::<S>() + 1;
        self.map.capacity() * slot * 8 / 7 + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
