//! Sparse state-vector backend: hash map of nonzero amplitudes.

use rustc_hash::FxHashMap;

use super::{scatter_table, validate_apply, Backend, PRUNE_TOL};
use crate::error::{Error, Result};
use crate::math::{GateMatrix, MATRIX_ZERO_TOL};
use crate::scalar::Scalar;

/// Maximum width for the sparse representation (basis indices are `u64`).
pub const SPARSE_MAX_QUBITS: usize = 63;

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

        let mut out: FxHashMap<u64, S> = FxHashMap::default();
        out.reserve(self.map.len());
        for (&idx, &amp) in &self.map {
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
