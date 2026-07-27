//! Discovery of computation-transparent structure.
//!
//! Two levels of "gates that do not affect the computation":
//!
//! 1. **Point stabilizers** — a single gate that acts as the identity (up
//!    to a global phase) on the *current* state: [`stabilizes_state`] tests
//!    one, [`discover_stabilizers`] searches a parametric family. These are
//!    the legal insertion points: anything discovered here can be spliced
//!    into the present without changing physics.
//!
//! 2. **Signal threads** ([`Insertion`]) — ops spliced at *several* points
//!    of the circuit graph: a signal is added at one point, transformed or
//!    exploited in between, and removed at another. The parts are **not**
//!    individually identity — only the whole n-wide unit is, which is
//!    exactly what [`verify_transparent`] checks: it runs the base and the
//!    spliced circuit and demands the *final* states agree (up to phase),
//!    while the thread is free to carry structure mid-flight (parity
//!    handles, expectation anchors, re-routed entanglement).
//!
//! Whether an insertion made the circuit *more computable* is a measured
//! question, not an asserted one: [`verify_transparent`] also runs both
//! variants on the [`FactoredState`] backend and reports the peak memory
//! and peak factor width each one forced.

use crate::backend::{Backend, DenseState, FactoredState, DENSE_MAX_QUBITS};
use crate::circuit::{Circuit, GateKernel, Op};
use crate::error::{Error, Result};
use crate::gates::GateDef;
use crate::scalar::Scalar;
use crate::sim::Simulator;

/// Largest entrywise deviation between `b` and `u·a` for the best global
/// unit `u` (division algebras align phases at the largest amplitude of
/// `a`; other algebras compare exactly). Scans the union support.
pub fn state_deviation_up_to_phase<S: Scalar>(a: &dyn Backend<S>, b: &dyn Backend<S>) -> f64 {
    // Pivot: largest amplitude of a.
    let mut pivot: Option<(u64, f64)> = None;
    a.for_each_nonzero(&mut |i, x| {
        let w = x.abs_sqr();
        if pivot.map(|(_, best)| w > best).unwrap_or(true) {
            pivot = Some((i, w));
        }
    });
    let phase = match pivot {
        Some((i, w)) if S::DIVISION && w > 0.0 => {
            // u = b(i)·conj(a(i)) / |a(i)|² so that u·a(i) = b(i).
            (b.amplitude(i) * a.amplitude(i).conj()).scale(1.0 / w)
        }
        _ => S::one(),
    };
    let mut deviation = 0.0f64;
    a.for_each_nonzero(&mut |i, x| {
        deviation = deviation.max((phase * x - b.amplitude(i)).abs_sqr().sqrt());
    });
    b.for_each_nonzero(&mut |i, x| {
        deviation = deviation.max((phase * a.amplitude(i) - x).abs_sqr().sqrt());
    });
    deviation
}

/// Result of a point-stabilizer test.
#[derive(Debug, Clone, Copy)]
pub struct StabilizerCheck {
    /// Deviation of `U|ψ⟩` from `|ψ⟩` up to a global unit.
    pub deviation: f64,
    /// Whether the deviation is within the requested tolerance.
    pub stabilizes: bool,
}

/// Does this kernel act as the identity (up to global phase) on the current
/// state? Snapshots the state into a dense copy, applies, compares.
pub fn stabilizes_state<S: Scalar>(
    state: &dyn Backend<S>,
    kernel: &GateKernel<S>,
    qubits: &[usize],
    tol: f64,
) -> Result<StabilizerCheck> {
    let n = state.num_qubits();
    if n > DENSE_MAX_QUBITS {
        return Err(Error::TooManyQubits {
            requested: n,
            max: DENSE_MAX_QUBITS,
        });
    }
    let mut entries = Vec::new();
    state.for_each_nonzero(&mut |i, a| entries.push((i, a)));
    let mut original = DenseState::<S>::new(n)?;
    original.load(&entries)?;
    let mut evolved = DenseState::<S>::new(n)?;
    evolved.load(&entries)?;
    match kernel {
        GateKernel::Matrix(m) => evolved.apply(m, qubits)?,
        GateKernel::Diagonal(d) => evolved.apply_diagonal(d, qubits)?,
    }
    let deviation = state_deviation_up_to_phase(&original, &evolved);
    Ok(StabilizerCheck {
        deviation,
        stabilizes: deviation <= tol,
    })
}

/// Search a parametric gate family for point stabilizers of the current
/// state. Returns the parameter sets that stabilize, with their deviations.
pub fn discover_stabilizers<S: Scalar>(
    state: &dyn Backend<S>,
    family: &dyn GateDef<S>,
    qubits: &[usize],
    param_sets: &[Vec<f64>],
    tol: f64,
) -> Result<Vec<(Vec<f64>, f64)>> {
    let mut found = Vec::new();
    for params in param_sets {
        if params.len() != family.param_count() {
            return Err(Error::ParamCountMismatch {
                gate: family.name().to_string(),
                expected: family.param_count(),
                got: params.len(),
            });
        }
        let matrix = family.matrix(params)?;
        let check = stabilizes_state(state, &GateKernel::Matrix(matrix), qubits, tol)?;
        if check.stabilizes {
            found.push((params.clone(), check.deviation));
        }
    }
    Ok(found)
}

/// A signal thread: ops spliced at multiple positions of a base circuit,
/// forming **one n-wide compute unit**. `ops` holds `(position, op)` pairs;
/// each op is inserted *before* the base op at that index (`position ==
/// base.len()` appends), same-position ops in insertion order.
#[derive(Debug, Clone)]
pub struct Insertion<S: Scalar> {
    /// Thread name (for reports).
    pub name: String,
    /// `(position, op)` splice list.
    pub ops: Vec<(usize, Op<S>)>,
}

impl<S: Scalar> Insertion<S> {
    /// An empty thread.
    pub fn new(name: impl Into<String>) -> Self {
        Insertion {
            name: name.into(),
            ops: Vec::new(),
        }
    }

    /// Splice a named gate before base position `position`.
    pub fn gate(
        mut self,
        position: usize,
        name: impl Into<String>,
        params: impl Into<Vec<f64>>,
        qubits: impl Into<Vec<usize>>,
    ) -> Self {
        self.ops.push((
            position,
            Op::Named {
                name: name.into(),
                params: params.into(),
                qubits: qubits.into(),
            },
        ));
        self
    }

    /// Splice a diagonal kernel before base position `position`.
    pub fn diagonal(
        mut self,
        position: usize,
        label: impl Into<String>,
        entries: Vec<S>,
        qubits: impl Into<Vec<usize>>,
    ) -> Self {
        self.ops.push((
            position,
            Op::Diagonal {
                label: label.into(),
                entries,
                qubits: qubits.into(),
            },
        ));
        self
    }

    /// The spliced circuit.
    pub fn splice(&self, base: &Circuit<S>) -> Circuit<S> {
        let mut out = Circuit::new(base.num_qubits());
        let push = |out: &mut Circuit<S>, op: &Op<S>| match op {
            Op::Named {
                name,
                params,
                qubits,
            } => {
                out.gate(name.clone(), params.clone(), qubits.clone());
            }
            Op::Raw {
                label,
                matrix,
                qubits,
            } => {
                out.raw(label.clone(), matrix.clone(), qubits.clone());
            }
            Op::Diagonal {
                label,
                entries,
                qubits,
            } => {
                out.diagonal(label.clone(), entries.clone(), qubits.clone());
            }
        };
        for (index, base_op) in base.ops().iter().enumerate() {
            for (position, op) in &self.ops {
                if *position == index {
                    push(&mut out, op);
                }
            }
            push(&mut out, base_op);
        }
        for (position, op) in &self.ops {
            if *position >= base.ops().len() {
                push(&mut out, op);
            }
        }
        out
    }
}

/// Outcome of a transparency check: is the whole thread computation-neutral,
/// and what did it cost (or save) geometrically?
#[derive(Debug, Clone)]
pub struct TransparencyReport {
    /// Thread name.
    pub name: String,
    /// Final-state deviation (up to global phase) between base and spliced.
    pub final_deviation: f64,
    /// `|total weight − 1|` of the spliced run.
    pub weight_drift: f64,
    /// Whether the thread is transparent within tolerance.
    pub transparent: bool,
    /// Factored-backend peaks `(total amplitudes, largest factor qubits)`
    /// for the base circuit…
    pub peak_without: (usize, usize),
    /// …and for the spliced circuit — the measured cost of the thread.
    pub peak_with: (usize, usize),
}

/// Verify an [`Insertion`] is transparent — the *whole* unit leaves the end
/// state unchanged up to phase — and measure its geometric cost on the
/// factored backend.
pub fn verify_transparent<S: Scalar>(
    sim: &Simulator<S>,
    base: &Circuit<S>,
    insertion: &Insertion<S>,
    tol: f64,
) -> Result<TransparencyReport> {
    let spliced = insertion.splice(base);
    let base_state = sim.run(base)?;
    let spliced_state = sim.run(&spliced)?;
    let final_deviation = state_deviation_up_to_phase(base_state.as_ref(), spliced_state.as_ref());
    let weight_drift = (spliced_state.total_weight() - 1.0).abs();

    let peak = |circuit: &Circuit<S>| -> Result<(usize, usize)> {
        let state = sim.run_on("factored", circuit)?;
        let factored = state
            .as_any()
            .downcast_ref::<FactoredState<S>>()
            .expect("'factored' backend is FactoredState");
        Ok(factored.peak())
    };
    Ok(TransparencyReport {
        name: insertion.name.clone(),
        final_deviation,
        weight_drift,
        transparent: final_deviation <= tol,
        peak_without: peak(base)?,
        peak_with: peak(&spliced)?,
    })
}
