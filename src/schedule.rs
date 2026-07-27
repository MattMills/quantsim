//! Evented gate scheduling: simultaneous loops of quantum gates on one
//! register, message-queue style.
//!
//! Instead of one static gate sequence, a [`Schedule`] holds:
//!
//! * **loops** ([`GateLoop`]) — bodies of timed ops repeating with a period
//!   and phase, running *simultaneously* with every other loop,
//! * **one-shot events** — ops at absolute ticks,
//! * **measurement events** ([`MeasureEvent`]) — measurements whose
//!   outcomes enqueue further [`FeedbackOp`]s: gates, or **further
//!   measurements with their own branches**, recursively. A feedback
//!   branch is a finite tree, so adaptive protocols of any bounded depth —
//!   outcome-dependent measurement choices, bounded repeat-until-success
//!   ladders, gate-teleportation cascades — run inside one schedule, and
//!   termination is structural (the tree is finite data), not hoped for.
//!
//! Execution drains a deterministic priority queue ordered by
//! `(tick, sequence)`. Simultaneity is real: multiple events can share a
//! tick. Same-tick ordering is deterministic (one-shots, then loops in
//! insertion order, then measurements, then feedback in enqueue order), and
//! [`OverlapPolicy::Error`] instead rejects schedules where same-tick
//! events touch overlapping qubits — use it when "simultaneous" must mean
//! "commuting by construction". (The policy is checked over the *static*
//! schedule; feedback ops are causally ordered by construction and exempt.)
//!
//! A measurement-free schedule can be flattened to an ordinary
//! [`Circuit`] with [`Schedule::to_circuit`], which is also how schedule
//! semantics are pinned in the tests: run-through-the-queue must equal the
//! flattened circuit exactly.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::backend::Backend;
use crate::circuit::Circuit;
use crate::error::{Error, Result};
use crate::math::{GateMatrix, UNITARY_TOL};
use crate::rng::Prng;
use crate::scalar::{Scalar, C64};
use crate::sim::Simulator;

/// What a scheduled event applies.
#[derive(Debug, Clone)]
pub enum ScheduledKernel<S: Scalar> {
    /// A registry gate by name.
    Named {
        /// Gate name or alias.
        name: String,
        /// Real parameters.
        params: Vec<f64>,
    },
    /// A diagonal unitary (see [`Circuit::diagonal`]).
    Diagonal {
        /// Display label.
        label: String,
        /// `2^k` diagonal entries.
        entries: Vec<S>,
    },
    /// An explicit unitary matrix.
    Raw {
        /// Display label.
        label: String,
        /// The unitary.
        matrix: GateMatrix<S>,
    },
}

impl<S: Scalar> ScheduledKernel<S> {
    fn label(&self) -> &str {
        match self {
            ScheduledKernel::Named { name, .. } => name,
            ScheduledKernel::Diagonal { label, .. } | ScheduledKernel::Raw { label, .. } => label,
        }
    }
}

/// One op at a tick. Inside a [`GateLoop`] body or a [`MeasureEvent`]
/// branch, `at` is a *relative* offset; as a one-shot it is absolute.
#[derive(Debug, Clone)]
pub struct TimedOp<S: Scalar> {
    /// Tick (absolute or relative — see above).
    pub at: u64,
    /// The kernel to apply.
    pub kernel: ScheduledKernel<S>,
    /// Target qubits.
    pub qubits: Vec<usize>,
}

impl<S: Scalar> TimedOp<S> {
    /// Named-gate op.
    pub fn gate(
        at: u64,
        name: impl Into<String>,
        params: impl Into<Vec<f64>>,
        qubits: impl Into<Vec<usize>>,
    ) -> Self {
        TimedOp {
            at,
            kernel: ScheduledKernel::Named {
                name: name.into(),
                params: params.into(),
            },
            qubits: qubits.into(),
        }
    }
}

/// A repeating body of ops: `body` offsets are relative to each iteration's
/// start; iteration `k` begins at `start + k·period`.
#[derive(Debug, Clone)]
pub struct GateLoop<S: Scalar> {
    /// Loop name (appears in traces).
    pub name: String,
    /// First iteration's start tick.
    pub start: u64,
    /// Ticks between iteration starts (must be > 0; body offsets < period).
    pub period: u64,
    /// Number of iterations, or `None` to repeat until the horizon.
    pub iterations: Option<u64>,
    /// The loop body (relative offsets).
    pub body: Vec<TimedOp<S>>,
}

/// One feedback action enqueued by a measurement outcome: apply a gate, or
/// perform a **further measurement** whose own branches may nest deeper.
/// Offsets are relative to the parent measurement's tick. Because branches
/// are owned finite trees, adaptivity has structural termination: every
/// run fires finitely many events regardless of outcomes.
#[derive(Debug, Clone)]
pub enum FeedbackOp<S: Scalar> {
    /// Apply a kernel at a relative offset.
    Gate(TimedOp<S>),
    /// Measure again at a relative offset (`event.at` is relative here),
    /// with its own `on_zero`/`on_one` feedback branches.
    Measure(Box<MeasureEvent<S>>),
}

impl<S: Scalar> FeedbackOp<S> {
    /// Named-gate feedback op at a relative offset.
    pub fn gate(
        at: u64,
        name: impl Into<String>,
        params: impl Into<Vec<f64>>,
        qubits: impl Into<Vec<usize>>,
    ) -> Self {
        FeedbackOp::Gate(TimedOp::gate(at, name, params, qubits))
    }

    /// Nested measurement at a relative offset, with its own branches.
    pub fn measure(
        at: u64,
        qubit: usize,
        on_zero: Vec<FeedbackOp<S>>,
        on_one: Vec<FeedbackOp<S>>,
    ) -> Self {
        FeedbackOp::Measure(Box::new(MeasureEvent {
            at,
            qubit,
            on_zero,
            on_one,
        }))
    }
}

impl<S: Scalar> From<TimedOp<S>> for FeedbackOp<S> {
    fn from(op: TimedOp<S>) -> Self {
        FeedbackOp::Gate(op)
    }
}

/// A measurement at a tick; the outcome enqueues one of two feedback
/// branches at offsets relative to the measurement tick (message-queue
/// feedback). Branches may contain further measurements — see
/// [`FeedbackOp`]. As a top-level event `at` is absolute; nested inside a
/// branch it is relative to the parent measurement's tick.
#[derive(Debug, Clone)]
pub struct MeasureEvent<S: Scalar> {
    /// Measurement tick (absolute at top level, relative when nested).
    pub at: u64,
    /// Qubit to measure.
    pub qubit: usize,
    /// Feedback enqueued when the outcome is 0.
    pub on_zero: Vec<FeedbackOp<S>>,
    /// Feedback enqueued when the outcome is 1.
    pub on_one: Vec<FeedbackOp<S>>,
}

/// How to treat same-tick events with overlapping qubits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverlapPolicy {
    /// Apply in the deterministic same-tick order (the default).
    #[default]
    InsertionOrder,
    /// Reject the schedule at run time: simultaneity must be disjoint.
    /// (Checked over the static schedule; feedback ops are causally ordered
    /// by construction and exempt.)
    Error,
}

/// An evented gate program on `num_qubits` qubits, bounded by a horizon.
#[derive(Debug, Clone)]
pub struct Schedule<S: Scalar = C64> {
    num_qubits: usize,
    horizon: u64,
    policy: OverlapPolicy,
    ops: Vec<TimedOp<S>>,
    loops: Vec<GateLoop<S>>,
    measures: Vec<MeasureEvent<S>>,
}

/// One applied event in a [`ScheduleTrace`].
#[derive(Debug, Clone)]
pub struct TraceEntry {
    /// Tick at which the event fired.
    pub time: u64,
    /// Kernel label (gate name) or `measure`.
    pub label: String,
    /// Target qubits.
    pub qubits: Vec<usize>,
}

/// Everything that happened during one run.
#[derive(Debug, Clone, Default)]
pub struct ScheduleTrace {
    /// Applied events in execution order.
    pub events: Vec<TraceEntry>,
    /// `(tick, qubit, outcome)` per measurement.
    pub measurements: Vec<(u64, usize, bool)>,
    /// Events skipped because they fell past the horizon.
    pub skipped_past_horizon: u64,
}

/// `(time, sequence, item)` triples from static-schedule expansion, plus
/// the next free sequence number and the count of past-horizon skips.
type StaticExpansion<S> = (Vec<(u64, u64, Item<S>)>, u64, u64);

#[derive(Debug, Clone)]
enum Item<S: Scalar> {
    Gate {
        kernel: ScheduledKernel<S>,
        qubits: Vec<usize>,
    },
    /// An owned measurement event (static events are cloned in; feedback
    /// events are spliced from their parent's branches). The heap entry's
    /// time is authoritative; the event's own `at` is not consulted again.
    Measure { event: MeasureEvent<S> },
}

struct HeapEntry<S: Scalar> {
    time: u64,
    seq: u64,
    item: Item<S>,
}

impl<S: Scalar> PartialEq for HeapEntry<S> {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.seq == other.seq
    }
}
impl<S: Scalar> Eq for HeapEntry<S> {}
impl<S: Scalar> PartialOrd for HeapEntry<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<S: Scalar> Ord for HeapEntry<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for a min-heap on (time, seq).
        (other.time, other.seq).cmp(&(self.time, self.seq))
    }
}

impl<S: Scalar> Schedule<S> {
    /// An empty schedule on `num_qubits` qubits, running through tick
    /// `horizon` inclusive.
    pub fn new(num_qubits: usize, horizon: u64) -> Self {
        Schedule {
            num_qubits,
            horizon,
            policy: OverlapPolicy::default(),
            ops: Vec::new(),
            loops: Vec::new(),
            measures: Vec::new(),
        }
    }

    /// Register width.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Set the same-tick overlap policy.
    pub fn policy(&mut self, policy: OverlapPolicy) -> &mut Self {
        self.policy = policy;
        self
    }

    /// One-shot named gate at an absolute tick.
    pub fn at(
        &mut self,
        time: u64,
        name: impl Into<String>,
        params: impl Into<Vec<f64>>,
        qubits: impl Into<Vec<usize>>,
    ) -> &mut Self {
        self.ops.push(TimedOp::gate(time, name, params, qubits));
        self
    }

    /// One-shot diagonal kernel at an absolute tick.
    pub fn diagonal_at(
        &mut self,
        time: u64,
        label: impl Into<String>,
        entries: Vec<S>,
        qubits: impl Into<Vec<usize>>,
    ) -> &mut Self {
        self.ops.push(TimedOp {
            at: time,
            kernel: ScheduledKernel::Diagonal {
                label: label.into(),
                entries,
            },
            qubits: qubits.into(),
        });
        self
    }

    /// One-shot raw unitary at an absolute tick.
    pub fn raw_at(
        &mut self,
        time: u64,
        label: impl Into<String>,
        matrix: GateMatrix<S>,
        qubits: impl Into<Vec<usize>>,
    ) -> &mut Self {
        self.ops.push(TimedOp {
            at: time,
            kernel: ScheduledKernel::Raw {
                label: label.into(),
                matrix,
            },
            qubits: qubits.into(),
        });
        self
    }

    /// Add a simultaneous loop.
    pub fn add_loop(&mut self, gate_loop: GateLoop<S>) -> &mut Self {
        self.loops.push(gate_loop);
        self
    }

    /// Add a measurement event whose branches apply gates only (the common
    /// case; see [`Schedule::measure_branching_at`] for adaptive trees).
    pub fn measure_at(
        &mut self,
        time: u64,
        qubit: usize,
        on_zero: Vec<TimedOp<S>>,
        on_one: Vec<TimedOp<S>>,
    ) -> &mut Self {
        self.measure_branching_at(
            time,
            qubit,
            on_zero.into_iter().map(FeedbackOp::Gate).collect(),
            on_one.into_iter().map(FeedbackOp::Gate).collect(),
        )
    }

    /// Add a measurement event with full feedback branches — gates and
    /// nested measurements ([`FeedbackOp`]), adaptive to any finite depth.
    pub fn measure_branching_at(
        &mut self,
        time: u64,
        qubit: usize,
        on_zero: Vec<FeedbackOp<S>>,
        on_one: Vec<FeedbackOp<S>>,
    ) -> &mut Self {
        self.measures.push(MeasureEvent {
            at: time,
            qubit,
            on_zero,
            on_one,
        });
        self
    }

    /// Expand the static (non-feedback) schedule into `(time, seq, item)`
    /// triples with the documented deterministic same-tick order. Returns
    /// the triples, the next free sequence number, and the count of events
    /// skipped for falling past the horizon.
    fn static_items(&self) -> Result<StaticExpansion<S>> {
        let mut items = Vec::new();
        let mut seq = 0u64;
        let mut skipped = 0u64;
        for op in &self.ops {
            if op.at > self.horizon {
                skipped += 1;
                continue;
            }
            items.push((
                op.at,
                seq,
                Item::Gate {
                    kernel: op.kernel.clone(),
                    qubits: op.qubits.clone(),
                },
            ));
            seq += 1;
        }
        for gate_loop in &self.loops {
            if gate_loop.period == 0 {
                return Err(Error::InvalidState(format!(
                    "loop '{}' has period 0",
                    gate_loop.name
                )));
            }
            for op in &gate_loop.body {
                if op.at >= gate_loop.period {
                    return Err(Error::InvalidState(format!(
                        "loop '{}': body offset {} exceeds period {}",
                        gate_loop.name, op.at, gate_loop.period
                    )));
                }
            }
            let mut iteration = 0u64;
            loop {
                if let Some(max) = gate_loop.iterations {
                    if iteration >= max {
                        break;
                    }
                }
                let base = gate_loop.start + iteration * gate_loop.period;
                if base > self.horizon {
                    break;
                }
                for op in &gate_loop.body {
                    let time = base + op.at;
                    if time > self.horizon {
                        skipped += 1;
                        continue;
                    }
                    items.push((
                        time,
                        seq,
                        Item::Gate {
                            kernel: op.kernel.clone(),
                            qubits: op.qubits.clone(),
                        },
                    ));
                    seq += 1;
                }
                iteration += 1;
            }
        }
        for measure in &self.measures {
            if measure.at > self.horizon {
                skipped += 1;
                continue;
            }
            items.push((
                measure.at,
                seq,
                Item::Measure {
                    event: measure.clone(),
                },
            ));
            seq += 1;
        }
        if self.policy == OverlapPolicy::Error {
            self.check_static_overlaps(&items)?;
        }
        Ok((items, seq, skipped))
    }

    fn check_static_overlaps(&self, items: &[(u64, u64, Item<S>)]) -> Result<()> {
        let mut sorted: Vec<&(u64, u64, Item<S>)> = items.iter().collect();
        sorted.sort_by_key(|(time, seq, _)| (*time, *seq));
        let mut i = 0;
        while i < sorted.len() {
            let tick = sorted[i].0;
            let mut used: Vec<usize> = Vec::new();
            let mut j = i;
            while j < sorted.len() && sorted[j].0 == tick {
                let qubits: Vec<usize> = match &sorted[j].2 {
                    Item::Gate { qubits, .. } => qubits.clone(),
                    Item::Measure { event } => vec![event.qubit],
                };
                for q in qubits {
                    if used.contains(&q) {
                        return Err(Error::InvalidState(format!(
                            "overlap policy: qubit {q} touched by multiple events at tick {tick}"
                        )));
                    }
                    used.push(q);
                }
                j += 1;
            }
            i = j;
        }
        Ok(())
    }

    /// Flatten a measurement-free schedule to a [`Circuit`] in execution
    /// order. Fails with [`Error::InvalidState`] if measurement events are
    /// present (their feedback cannot be expressed as a static circuit).
    pub fn to_circuit(&self) -> Result<Circuit<S>> {
        if !self.measures.is_empty() {
            return Err(Error::InvalidState(
                "schedule contains measurement events; cannot flatten to a circuit".to_string(),
            ));
        }
        let (mut items, _, _) = self.static_items()?;
        items.sort_by_key(|(time, seq, _)| (*time, *seq));
        let mut circuit = Circuit::new(self.num_qubits);
        for (_, _, item) in items {
            if let Item::Gate { kernel, qubits } = item {
                match kernel {
                    ScheduledKernel::Named { name, params } => {
                        circuit.gate(name, params, qubits);
                    }
                    ScheduledKernel::Diagonal { label, entries } => {
                        circuit.diagonal(label, entries, qubits);
                    }
                    ScheduledKernel::Raw { label, matrix } => {
                        circuit.raw(label, matrix, qubits);
                    }
                }
            }
        }
        Ok(circuit)
    }

    /// Run the schedule on a named backend. Measurement randomness comes
    /// from `seed`; the run is fully deterministic given `(schedule, seed)`.
    pub fn run_on(
        &self,
        sim: &Simulator<S>,
        backend: &str,
        seed: u64,
    ) -> Result<(Box<dyn Backend<S>>, ScheduleTrace)> {
        let mut state = sim.backends().create(backend, self.num_qubits)?;
        let (items, mut next_seq, skipped) = self.static_items()?;
        let mut trace = ScheduleTrace {
            skipped_past_horizon: skipped,
            ..ScheduleTrace::default()
        };
        let mut heap: BinaryHeap<HeapEntry<S>> = items
            .into_iter()
            .map(|(time, seq, item)| HeapEntry { time, seq, item })
            .collect();
        let mut rng = Prng::new(seed);

        while let Some(entry) = heap.pop() {
            match entry.item {
                Item::Gate { kernel, qubits } => {
                    let label = kernel.label().to_string();
                    apply_kernel(sim, state.as_mut(), &kernel, &qubits)?;
                    trace.events.push(TraceEntry {
                        time: entry.time,
                        label,
                        qubits,
                    });
                }
                Item::Measure { event } => {
                    let outcome = state.measure(event.qubit, &mut rng)?;
                    trace.measurements.push((entry.time, event.qubit, outcome));
                    trace.events.push(TraceEntry {
                        time: entry.time,
                        label: format!("measure→{}", outcome as u8),
                        qubits: vec![event.qubit],
                    });
                    let branch = if outcome { event.on_one } else { event.on_zero };
                    for op in branch {
                        let (time, item) = match op {
                            FeedbackOp::Gate(op) => (
                                entry.time + op.at,
                                Item::Gate {
                                    kernel: op.kernel,
                                    qubits: op.qubits,
                                },
                            ),
                            FeedbackOp::Measure(nested) => {
                                (entry.time + nested.at, Item::Measure { event: *nested })
                            }
                        };
                        if time > self.horizon {
                            // A skipped nested measurement drops its whole
                            // subtree; it counts once here.
                            trace.skipped_past_horizon += 1;
                            continue;
                        }
                        heap.push(HeapEntry {
                            time,
                            seq: next_seq,
                            item,
                        });
                        next_seq += 1;
                    }
                }
            }
        }
        Ok((state, trace))
    }

    /// Run on the default dense backend.
    pub fn run(
        &self,
        sim: &Simulator<S>,
        seed: u64,
    ) -> Result<(Box<dyn Backend<S>>, ScheduleTrace)> {
        self.run_on(sim, "dense", seed)
    }
}

fn apply_kernel<S: Scalar>(
    sim: &Simulator<S>,
    state: &mut dyn Backend<S>,
    kernel: &ScheduledKernel<S>,
    qubits: &[usize],
) -> Result<()> {
    match kernel {
        ScheduledKernel::Named { name, params } => sim.apply(state, name, params, qubits),
        ScheduledKernel::Diagonal { label, entries } => {
            let mut deviation: f64 = 0.0;
            for &d in entries {
                deviation = deviation.max((d.conj() * d - S::one()).abs_sqr().sqrt());
            }
            if deviation > UNITARY_TOL {
                return Err(Error::NotUnitary {
                    label: label.clone(),
                    deviation,
                });
            }
            state.apply_diagonal(entries, qubits)
        }
        ScheduledKernel::Raw { label, matrix } => {
            let deviation = matrix.unitarity_deviation();
            if deviation > UNITARY_TOL {
                return Err(Error::NotUnitary {
                    label: label.clone(),
                    deviation,
                });
            }
            state.apply(matrix, qubits)
        }
    }
}
