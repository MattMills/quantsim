//! Dense state-vector backend: the BQP reference implementation.

use super::{expand_index, scatter_table, validate_apply, Backend};
use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::scalar::Scalar;

/// Maximum width for the dense representation (memory is
/// `2^n · size_of::<S>()`; 32 qubits of `C64` is already 64 GiB).
pub const DENSE_MAX_QUBITS: usize = 32;

/// A flat `2^n` amplitude vector.
///
/// Time per `k`-qubit gate is `O(4^k / 2^k · 2^n)` = `O(2^{n+k})`; memory is
/// exactly one amplitude per basis state. With the universal gate set in the
/// standard registry this simulates any polynomial-size BQP circuit — at
/// exponential cost in width, which is what the width benchmarks measure.
#[derive(Debug, Clone)]
pub struct DenseState<S: Scalar> {
    num_qubits: usize,
    amps: Vec<S>,
}

impl<S: Scalar> DenseState<S> {
    /// `|0…0⟩` on `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits > DENSE_MAX_QUBITS {
            return Err(Error::TooManyQubits { requested: num_qubits, max: DENSE_MAX_QUBITS });
        }
        let mut amps = vec![S::zero(); 1usize << num_qubits];
        amps[0] = S::one();
        Ok(DenseState { num_qubits, amps })
    }

    /// Direct read-only view of the amplitude vector.
    pub fn amplitudes(&self) -> &[S] {
        &self.amps
    }

    fn apply_single(&mut self, m: &GateMatrix<S>, q: usize) {
        let (m00, m01, m10, m11) = (m.get(0, 0), m.get(0, 1), m.get(1, 0), m.get(1, 1));
        let mask = 1usize << q;
        let half = self.amps.len() >> 1;
        for i in 0..half {
            let low = i & (mask - 1);
            let i0 = ((i >> q) << (q + 1)) | low;
            let i1 = i0 | mask;
            let a0 = self.amps[i0];
            let a1 = self.amps[i1];
            self.amps[i0] = m00 * a0 + m01 * a1;
            self.amps[i1] = m10 * a0 + m11 * a1;
        }
    }

    fn apply_general(&mut self, m: &GateMatrix<S>, qubits: &[usize]) {
        let k = qubits.len();
        let d = 1usize << k;
        let mut sorted = qubits.to_vec();
        sorted.sort_unstable();
        let scatter = scatter_table(qubits);
        let groups = self.amps.len() >> k;
        let mdata = m.data();
        let mut scratch = vec![S::zero(); d];
        for g in 0..groups {
            let base = expand_index(g as u64, &sorted);
            for (j, slot) in scratch.iter_mut().enumerate() {
                *slot = self.amps[(base | scatter[j]) as usize];
            }
            for r in 0..d {
                let mut acc = S::zero();
                for (l, &s) in scratch.iter().enumerate() {
                    acc = acc + mdata[r * d + l] * s;
                }
                self.amps[(base | scatter[r]) as usize] = acc;
            }
        }
    }
}

impl<S: Scalar> Backend<S> for DenseState<S> {
    fn name(&self) -> &str {
        "dense"
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        validate_apply(self.num_qubits, matrix, qubits)?;
        if qubits.len() == 1 {
            self.apply_single(matrix, qubits[0]);
        } else {
            self.apply_general(matrix, qubits);
        }
        Ok(())
    }

    fn amplitude(&self, index: u64) -> S {
        self.amps.get(index as usize).copied().unwrap_or_else(S::zero)
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        for (i, &a) in self.amps.iter().enumerate() {
            if a.abs_sqr() > 0.0 {
                f(i as u64, a);
            }
        }
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        let mask = 1usize << qubit;
        for (i, a) in self.amps.iter_mut().enumerate() {
            if ((i & mask) != 0) == outcome {
                *a = a.scale(renorm);
            } else {
                *a = S::zero();
            }
        }
    }

    fn reset(&mut self) {
        for a in self.amps.iter_mut() {
            *a = S::zero();
        }
        self.amps[0] = S::one();
    }

    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        let len = self.amps.len() as u64;
        for &(i, _) in entries {
            if i >= len {
                return Err(Error::QubitOutOfRange {
                    qubit: 64 - i.leading_zeros() as usize,
                    num_qubits: self.num_qubits,
                });
            }
        }
        for a in self.amps.iter_mut() {
            *a = S::zero();
        }
        for &(i, a) in entries {
            self.amps[i as usize] = a;
        }
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        self.amps.capacity() * std::mem::size_of::<S>() + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
