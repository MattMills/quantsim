//! Adaptive backend: sparse until density says otherwise, then dense.

use super::{Backend, DenseState, SparseState};
use crate::backend::dense::DENSE_MAX_QUBITS;
use crate::error::Result;
use crate::math::GateMatrix;
use crate::scalar::Scalar;

/// Promote to dense once the nonzero count reaches this fraction of `2^n`
/// (a hash-map entry costs several times a dense slot, so 1/4 density is
/// already past break-even).
const PROMOTE_DENSITY_DENOM: u64 = 4;

#[derive(Debug, Clone)]
enum Repr<S: Scalar> {
    Sparse(SparseState<S>),
    Dense(DenseState<S>),
}

/// Starts sparse; promotes itself to dense when the state grows dense enough
/// that the flat vector is cheaper. Widths beyond the dense limit
/// ([`DENSE_MAX_QUBITS`]) never promote.
///
/// Promotion is one-way: measurement can re-sparsify a state, but the
/// simpler invariant is easier to reason about, and a dense state that
/// collapsed still fits in memory by construction.
#[derive(Debug, Clone)]
pub struct AdaptiveState<S: Scalar> {
    repr: Repr<S>,
}

impl<S: Scalar> AdaptiveState<S> {
    /// `|0…0⟩` on `num_qubits` qubits (sparse representation).
    pub fn new(num_qubits: usize) -> Result<Self> {
        Ok(AdaptiveState {
            repr: Repr::Sparse(SparseState::new(num_qubits)?),
        })
    }

    /// Whether the state has promoted to the dense representation.
    pub fn is_dense(&self) -> bool {
        matches!(self.repr, Repr::Dense(_))
    }

    fn inner(&self) -> &dyn Backend<S> {
        match &self.repr {
            Repr::Sparse(s) => s,
            Repr::Dense(d) => d,
        }
    }

    fn inner_mut(&mut self) -> &mut dyn Backend<S> {
        match &mut self.repr {
            Repr::Sparse(s) => s,
            Repr::Dense(d) => d,
        }
    }

    fn maybe_promote(&mut self) -> Result<()> {
        if let Repr::Sparse(s) = &self.repr {
            let n = s.num_qubits();
            if n > DENSE_MAX_QUBITS {
                return Ok(());
            }
            let threshold = (1u64 << n) / PROMOTE_DENSITY_DENOM;
            if (s.nonzero_count() as u64) >= threshold.max(1) {
                let mut dense = DenseState::new(n)?;
                let entries: Vec<(u64, S)> = s.entries().collect();
                dense.load(&entries)?;
                self.repr = Repr::Dense(dense);
            }
        }
        Ok(())
    }
}

impl<S: Scalar> Backend<S> for AdaptiveState<S> {
    fn name(&self) -> &str {
        "adaptive"
    }

    fn num_qubits(&self) -> usize {
        self.inner().num_qubits()
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        self.inner_mut().apply(matrix, qubits)?;
        self.maybe_promote()
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        // Diagonal gates never change the support, so no promotion check.
        self.inner_mut().apply_diagonal(entries, qubits)
    }

    fn amplitude(&self, index: u64) -> S {
        self.inner().amplitude(index)
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        self.inner().for_each_nonzero(f)
    }

    fn nonzero_count(&self) -> usize {
        self.inner().nonzero_count()
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        self.inner_mut().project(qubit, outcome, renorm)
    }

    fn reset(&mut self) {
        let n = self.num_qubits();
        self.repr = Repr::Sparse(SparseState::new(n).expect("width was already validated"));
    }

    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        self.inner_mut().load(entries)?;
        self.maybe_promote()
    }

    fn memory_bytes(&self) -> usize {
        self.inner().memory_bytes()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
