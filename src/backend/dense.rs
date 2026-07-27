//! Dense state-vector backend: the BQP reference implementation.

use super::{validate_apply, Backend};
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
            return Err(Error::TooManyQubits {
                requested: num_qubits,
                max: DENSE_MAX_QUBITS,
            });
        }
        let mut amps = vec![S::zero(); 1usize << num_qubits];
        amps[0] = S::one();
        Ok(DenseState { num_qubits, amps })
    }

    /// Direct read-only view of the amplitude vector.
    pub fn amplitudes(&self) -> &[S] {
        &self.amps
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
            super::apply_single_in_place(&mut self.amps, matrix, qubits[0]);
        } else {
            super::apply_general_in_place(&mut self.amps, matrix, qubits);
        }
        Ok(())
    }

    fn amplitude(&self, index: u64) -> S {
        self.amps
            .get(index as usize)
            .copied()
            .unwrap_or_else(S::zero)
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        super::validate_apply_diagonal(self.num_qubits, entries, qubits)?;
        for (i, a) in self.amps.iter_mut().enumerate() {
            *a = entries[super::sub_index(i as u64, qubits)] * *a;
        }
        Ok(())
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
