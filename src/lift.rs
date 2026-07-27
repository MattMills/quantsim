//! Dimensional lift: rewrite a circuit into a **feedback loop in a larger
//! Clifford space**.
//!
//! [`to_clifford_feedback`] takes an `n`-qubit circuit whose non-Clifford
//! content is its `t`/`tdg` gates and produces an `(n + t)`-qubit
//! [`Schedule`] in which **every unitary is Clifford** and each former T
//! gate has become gate teleportation: a resource ancilla, a CX, a
//! measurement, and a classically-conditioned Clifford correction —
//! measurement outcomes steering the computation as queue messages. The
//! non-Cliffordness of the original circuit does not disappear; it is
//! relocated from *operators spread through the circuit* into *states*
//! (the `t` resource ancillas `|T⟩ = T·H|0⟩`), after which the dynamics
//! live entirely in the lifted Clifford space.
//!
//! The gadget (for `t` on data qubit `q` with ancilla `a`):
//!
//! ```text
//!   a: |0⟩ ─ H ─ T ─────●────── measure ──►  outcome 1: apply S to q
//!   q: ─────────────────X──────────────────  (outcome 0: done)
//! ```
//!
//! Per outcome the data register carries exactly `T|ψ⟩` — outcome 1 up to
//! a known global phase `e^{iπ/4}` (`e^{−iπ/4}` for `tdg`), which
//! [`outcome_phase`] reports so verification can be **exact**, not
//! up-to-phase. Run on a [`CliffordFramedState`], every unitary in the
//! lifted schedule absorbs into the frame, measurements project natively
//! through the tableau, and the only amplitude work left in the entire
//! computation is the resource preparation — [`ResourcePrep`] chooses
//! *when* that cost is paid (all upfront, or just-in-time per gadget),
//! which relocates the peak without changing the physics. Where the cost
//! actually condenses is a **measured** property of the run, not an
//! assumption of the rewrite: see `examples/clifford_lift.rs` and
//! `tests/clifford_lift.rs`.
//!
//! Scope: the lift recognizes literal `t` / `tdg` ops (the standard
//! non-Clifford currency; corrections `s`/`sdg` are Clifford, so one
//! round of feedback terminates). Generic-angle rotations would need
//! resource ladders with non-Clifford corrections (repeat-until-success)
//! — a further rung, deliberately not folded into this one.

use crate::circuit::{Circuit, Op};
use crate::error::{Error, Result};
use crate::scalar::Scalar;
use crate::schedule::{Schedule, TimedOp};

/// When the lifted schedule prepares its resource ancillas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourcePrep {
    /// All `|T⟩` ancillas prepared before the circuit body: the magic
    /// arrives as one up-front resource, the body is pure
    /// Clifford + measurement.
    Upfront,
    /// Each ancilla prepared immediately before its gadget consumes it:
    /// same physics, the preparation cost interleaves with consumption.
    JustInTime,
}

/// A lifted circuit: the feedback-loop schedule plus its geometry.
pub struct LiftedCircuit<S: Scalar> {
    /// The `(data + ancillas)`-qubit schedule; all unitaries Clifford.
    pub schedule: Schedule<S>,
    /// Original register width (data qubits `0..data_qubits`).
    pub data_qubits: usize,
    /// Number of resource ancillas (`==` T-count), at
    /// `data_qubits..data_qubits + ancillas`, in order of consumption.
    pub ancillas: usize,
    /// `+1` for each `t`, `−1` for each `tdg`, in ancilla order — the
    /// outcome-1 phase bookkeeping for [`outcome_phase`].
    pub phase_signs: Vec<i32>,
}

impl<S: Scalar> LiftedCircuit<S> {
    /// The known global phase `e^{iπ/4·Σ signs·outcomes}` accumulated by
    /// the gadgets, given the measured ancilla outcomes in ancilla order
    /// — divide it out for exact amplitude comparison against the
    /// unlifted circuit.
    pub fn outcome_phase(&self, outcomes: &[bool]) -> crate::scalar::C64 {
        let quarter: i32 = self
            .phase_signs
            .iter()
            .zip(outcomes)
            .map(|(&s, &o)| if o { s } else { 0 })
            .sum();
        crate::math::cis(std::f64::consts::FRAC_PI_4 * quarter as f64)
    }
}

/// Lift a circuit into the enlarged Clifford space. Errors if the
/// combined register would exceed the sparse mask width, or on a `t`/`tdg`
/// op with parameters (malformed).
pub fn to_clifford_feedback<S: Scalar>(
    circuit: &Circuit<S>,
    prep: ResourcePrep,
) -> Result<LiftedCircuit<S>> {
    let n = circuit.num_qubits();
    let t_count = circuit
        .ops()
        .iter()
        .filter(|op| matches!(op, Op::Named { name, .. } if name == "t" || name == "tdg"))
        .count();
    let total = n + t_count;
    if total > crate::backend::SPARSE_MAX_QUBITS {
        return Err(Error::TooManyQubits {
            requested: total,
            max: crate::backend::SPARSE_MAX_QUBITS,
        });
    }

    // Tick layout: stride 8 per original op leaves room for a gadget's
    // prep(2) + cx(1) + measure(1) + feedback(+1) with strict causality.
    let horizon = 8 * (circuit.ops().len() as u64 + 2);
    let mut schedule: Schedule<S> = Schedule::new(total, horizon);
    let mut tick: u64 = 0;
    let mut ancilla = n;
    let mut phase_signs = Vec::with_capacity(t_count);

    if prep == ResourcePrep::Upfront {
        let mut a = n;
        for op in circuit.ops() {
            if let Op::Named { name, .. } = op {
                if name == "t" || name == "tdg" {
                    schedule.at(0, "h", [], [a]);
                    schedule.at(1, name.clone(), [], [a]);
                    a += 1;
                }
            }
        }
        tick = 8;
    }

    for op in circuit.ops() {
        match op {
            Op::Named {
                name,
                params,
                qubits,
            } if name == "t" || name == "tdg" => {
                if !params.is_empty() {
                    return Err(Error::ParamCountMismatch {
                        gate: name.clone(),
                        expected: 0,
                        got: params.len(),
                    });
                }
                let q = qubits[0];
                let a = ancilla;
                ancilla += 1;
                phase_signs.push(if name == "t" { 1 } else { -1 });
                if prep == ResourcePrep::JustInTime {
                    schedule.at(tick, "h", [], [a]);
                    schedule.at(tick + 1, name.clone(), [], [a]);
                }
                let correction = if name == "t" { "s" } else { "sdg" };
                schedule.at(tick + 2, "cx", [], [q, a]);
                schedule.measure_at(
                    tick + 3,
                    a,
                    vec![],
                    vec![TimedOp::gate(1, correction, [], [q])],
                );
                tick += 8;
            }
            Op::Named {
                name,
                params,
                qubits,
            } => {
                schedule.at(tick, name.clone(), params.clone(), qubits.clone());
                tick += 8;
            }
            Op::Raw {
                label,
                matrix,
                qubits,
            } => {
                schedule.raw_at(tick, label.clone(), matrix.clone(), qubits.clone());
                tick += 8;
            }
            Op::Diagonal {
                label,
                entries,
                qubits,
            } => {
                schedule.diagonal_at(tick, label.clone(), entries.clone(), qubits.clone());
                tick += 8;
            }
        }
    }

    Ok(LiftedCircuit {
        schedule,
        data_qubits: n,
        ancillas: t_count,
        phase_signs,
    })
}
