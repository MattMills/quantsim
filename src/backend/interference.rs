//! Interference-instrumented backend: constructive and destructive
//! interference modeled independently.
//!
//! Amplitudes evolve exactly as in the dense backend, but every gate
//! application also records, per output amplitude, the **path weight**
//! `P = Σ_k |contribution_k|` — the magnitude the output would have if all
//! incoming contributions interfered constructively — next to the **net**
//! magnitude `|Σ_k contribution_k|`. The gap `P − |net| ≥ 0` (triangle
//! inequality) is the destructive interference that gate performed at that
//! output; `P ≈ |net|` means the paths aligned constructively.
//!
//! The ledger is kept two ways:
//!
//! * per gate — an [`InterferenceRecord`] with total path/net weight and
//!   the derived destruction, in application order;
//! * per basis state — a cumulative [`destruction_map`] of how much
//!   amplitude ever cancelled *at* each output state.
//!
//! Basis-dependence is inherent (interference is a basis-relative notion);
//! records are relative to the computational basis the simulator works in.
//! Diagonal kernels move phases but sum nothing, so they destroy exactly 0 —
//! a useful sanity anchor. Fidelity is untouched: this is the dense backend
//! plus bookkeeping, and the conformance suite holds it to that.
//!
//! [`destruction_map`]: InterferenceState::destruction_map

use super::{validate_apply, validate_apply_diagonal, Backend};
use crate::backend::dense::DENSE_MAX_QUBITS;
use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::scalar::Scalar;

/// Interference ledger entry for one gate application.
#[derive(Debug, Clone)]
pub struct InterferenceRecord {
    /// 0-based application index.
    pub gate_index: usize,
    /// Target qubits.
    pub qubits: Vec<usize>,
    /// Σ over outputs of Σ|contributions| — the fully-constructive bound.
    pub path_weight: f64,
    /// Σ over outputs of |net amplitude|.
    pub net_weight: f64,
}

impl InterferenceRecord {
    /// Amplitude magnitude destroyed by cancellation in this gate.
    pub fn destroyed(&self) -> f64 {
        (self.path_weight - self.net_weight).max(0.0)
    }
    /// Fraction of the path weight that survived (1 = fully constructive).
    pub fn constructive_fraction(&self) -> f64 {
        if self.path_weight > 0.0 {
            self.net_weight / self.path_weight
        } else {
            1.0
        }
    }
}

/// Dense amplitudes plus an independent constructive/destructive ledger.
#[derive(Debug, Clone)]
pub struct InterferenceState<S: Scalar> {
    num_qubits: usize,
    amps: Vec<S>,
    records: Vec<InterferenceRecord>,
    destroyed_at: Vec<f64>,
}

impl<S: Scalar> InterferenceState<S> {
    /// `|0…0⟩` with an empty ledger.
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits > DENSE_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: num_qubits,
                max: DENSE_MAX_QUBITS,
            });
        }
        let mut amps = crate::guard::try_vec(
            1usize << num_qubits,
            S::zero(),
            &format!("interference state ({num_qubits} qubits)"),
        )?;
        amps[0] = S::one();
        let destroyed_at = crate::guard::try_vec(
            1usize << num_qubits,
            0.0f64,
            &format!("interference ledger ({num_qubits} qubits)"),
        )?;
        Ok(InterferenceState {
            num_qubits,
            amps,
            records: Vec::new(),
            destroyed_at,
        })
    }

    /// The per-gate ledger, in application order.
    pub fn records(&self) -> &[InterferenceRecord] {
        &self.records
    }

    /// Total amplitude magnitude destroyed across the whole run.
    pub fn total_destroyed(&self) -> f64 {
        self.records.iter().map(|r| r.destroyed()).sum()
    }

    /// Cumulative destruction that happened *at* each basis state.
    pub fn destruction_map(&self) -> &[f64] {
        &self.destroyed_at
    }

    fn record(&mut self, qubits: &[usize], path: f64, net: f64) {
        self.records.push(InterferenceRecord {
            gate_index: self.records.len(),
            qubits: qubits.to_vec(),
            path_weight: path,
            net_weight: net,
        });
    }
}

impl<S: Scalar> Backend<S> for InterferenceState<S> {
    fn name(&self) -> &str {
        "interference"
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        validate_apply(self.num_qubits, matrix, qubits)?;
        let k = qubits.len();
        let d = 1usize << k;
        let mut sorted = qubits.to_vec();
        sorted.sort_unstable();
        let scatter = super::scatter_table(qubits);
        let groups = self.amps.len() >> k;
        let mdata = matrix.data();
        let mut scratch = vec![S::zero(); d];
        let mut path_total = 0.0f64;
        let mut net_total = 0.0f64;
        for g in 0..groups {
            if g % (1 << 20) == 0 {
                crate::guard::checkpoint()?;
            }
            let base = super::expand_index(g as u64, &sorted);
            for (j, slot) in scratch.iter_mut().enumerate() {
                *slot = self.amps[(base | scatter[j]) as usize];
            }
            for r in 0..d {
                let mut acc = S::zero();
                let mut path = 0.0f64;
                for (l, &s) in scratch.iter().enumerate() {
                    let term = mdata[r * d + l] * s;
                    path += term.abs_sqr().sqrt();
                    acc = acc + term;
                }
                let net = acc.abs_sqr().sqrt();
                let index = (base | scatter[r]) as usize;
                self.amps[index] = acc;
                self.destroyed_at[index] += (path - net).max(0.0);
                path_total += path;
                net_total += net;
            }
        }
        self.record(qubits, path_total, net_total);
        Ok(())
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        validate_apply_diagonal(self.num_qubits, entries, qubits)?;
        let mut weight = 0.0f64;
        for (i, a) in self.amps.iter_mut().enumerate() {
            *a = entries[super::sub_index(i as u64, qubits)] * *a;
            weight += a.abs_sqr().sqrt();
        }
        // One contribution per output: nothing can cancel.
        self.record(qubits, weight, weight);
        Ok(())
    }

    fn amplitude(&self, index: u64) -> S {
        self.amps
            .get(index as usize)
            .copied()
            .unwrap_or_else(S::zero)
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
        self.records.clear();
        for d in self.destroyed_at.iter_mut() {
            *d = 0.0;
        }
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
        self.amps.capacity() * std::mem::size_of::<S>()
            + self.destroyed_at.capacity() * std::mem::size_of::<f64>()
            + self.records.capacity() * std::mem::size_of::<InterferenceRecord>()
            + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
