//! What a representation must **answer**, not how it must store.
//!
//! [`crate::backend::Backend`] requires `amplitude(index) -> S`,
//! `for_each_nonzero`, and `load` — so every representation registered
//! against it must be able to expose itself in the `2^n` computational
//! basis. That makes the exponential object the ground truth and every
//! "alternative representation" a *compression* of it: the Clifford
//! frame, the MPS, the MERA tree, the phase field and the braided state
//! all flush to a dense inner the moment they are asked for amplitudes.
//!
//! The tell is where this crate's sub-exponential work actually lives.
//! [`crate::heisenberg`], [`crate::horizon`], [`crate::dyadic`] and
//! [`crate::blocks`] are not backends. They are free functions over a
//! [`PauliSum`], because they cannot answer
//! `amplitude()` and were never going to. The parts of the library that
//! beat `2^n` had to escape its central abstraction to exist.
//!
//! This module is that abstraction, rebuilt around the question rather
//! than the storage. A [`Resolver`] answers queries. It is never asked
//! for an amplitude, never asked to enumerate its support, and never
//! asked what the state of the whole register is — that last one is a
//! question a resolver is entitled to refuse to need.
//!
//! ## Cost is part of the answer
//!
//! Every [`Answer`] carries the [`Cost`] that produced it: how many
//! qubits were instantiated, how large a space was actually simulated,
//! how many gates ran. A resolver that answers a local observable on a
//! 40-qubit circuit by instantiating 13 qubits says so, in the same
//! value it returns. That turns "is this sub-exponential" from a claim
//! into a field on a struct.
//!
//! Three resolvers ship here, and the point is that they are not the
//! same kind of object:
//!
//! * [`StateResolver`] wraps any [`Backend`] —
//!   it runs the whole circuit and pays the whole width. The baseline.
//! * [`ConeResolver`] records the circuit and applies nothing until a
//!   query arrives, then **relabels the query's backward cone onto a
//!   compact register** and simulates only that. This is the piece
//!   [`crate::causal::causal_diamond`] was missing: pruning the gates
//!   outside the cone still leaves a circuit `n` qubits wide, so the
//!   simulation is still `2^n`. Relabelling makes it `2^{|cone|}`.
//! * [`HeisenbergResolver`] propagates the observable backwards and
//!   never holds a state at all. It is a resolver that could not be a
//!   backend under any refactoring, which is the structural point.
//!
//! ## What it does not claim
//!
//! Cone restriction gates the growth; it does not remove it. The cone
//! widens with depth, and once it covers the register the cost is the
//! register's. [`ConeResolver::cone_width`] reports where that happens
//! for a given query rather than assuming it, so the boundary is
//! measured. A refusal ([`Error::TooManyQubits`]) is a measurement too.

use std::collections::HashMap;

use crate::backend::{pauli_expectation, Backend, DenseState, MpsConfig, MpsState, SparseState};
use crate::circuit::{Circuit, GateKernel, Op};
use crate::error::{Error, Result};
use crate::gates::Pauli;
use crate::heisenberg::{self, PauliSum, Rotation};
use crate::math::GateMatrix;
use crate::registry::GateRegistry;
use crate::scalar::C64;

/// What answering a query actually cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cost {
    /// Qubits instantiated — the width of the register that was really
    /// built, which is the number that decides whether this was
    /// sub-exponential.
    pub qubits: usize,
    /// Width of the circuit the query was asked about.
    pub width: usize,
    /// Gates actually applied.
    pub gates: usize,
    /// Bytes the representation held at its peak.
    pub bytes: usize,
}

impl Cost {
    /// Dimension of the space simulated, `2^qubits`, saturating.
    pub fn dim(&self) -> u128 {
        if self.qubits >= 127 {
            u128::MAX
        } else {
            1u128 << self.qubits
        }
    }

    /// Dimension a full-width state vector would have needed.
    pub fn full_dim(&self) -> u128 {
        if self.width >= 127 {
            u128::MAX
        } else {
            1u128 << self.width
        }
    }

    /// How many times smaller the simulated space was than the register.
    pub fn compression(&self) -> f64 {
        self.full_dim() as f64 / self.dim().max(1) as f64
    }

    /// Nothing was simulated — the answer was structural.
    pub fn is_free(&self) -> bool {
        self.qubits == 0 && self.gates == 0
    }
}

/// A value and what it cost to get.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Answer {
    /// The answer.
    pub value: f64,
    /// What produced it.
    pub cost: Cost,
}

/// Something that answers questions about a circuit.
///
/// Deliberately missing: `amplitude`, `for_each_nonzero`, `load`,
/// `project`. A resolver is never required to be able to write itself
/// out in the computational basis, which is the requirement that makes
/// every [`Backend`] a compression of `2^n`.
pub trait Resolver {
    /// Name, for reports.
    fn name(&self) -> &str;

    /// Width of the circuit being asked about.
    fn width(&self) -> usize;

    /// `⟨O⟩` for a product of single-qubit Paulis at the end of the
    /// circuit.
    fn expectation(&mut self, ops: &[(usize, Pauli)]) -> Result<Answer>;

    /// How much `⟨O⟩` moves when `perturbation` is inserted on `site`
    /// just before gate `at_gate`.
    ///
    /// The default runs two expectations and differences them. A
    /// resolver that knows the causal structure should override it: when
    /// the site cannot reach the observable the answer is **exactly
    /// zero**, and no simulation is needed to say so.
    fn response(
        &mut self,
        at_gate: usize,
        site: usize,
        perturbation: Rotation,
        ops: &[(usize, Pauli)],
    ) -> Result<Answer> {
        let base = self.expectation(ops)?;
        let moved = self.perturbed_expectation(at_gate, site, perturbation, ops)?;
        Ok(Answer {
            value: moved.value - base.value,
            cost: Cost {
                qubits: base.cost.qubits.max(moved.cost.qubits),
                width: base.cost.width,
                gates: base.cost.gates + moved.cost.gates,
                bytes: base.cost.bytes.max(moved.cost.bytes),
            },
        })
    }

    /// `⟨O⟩` with `perturbation` inserted — the second branch of
    /// [`Resolver::response`].
    fn perturbed_expectation(
        &mut self,
        at_gate: usize,
        site: usize,
        perturbation: Rotation,
        ops: &[(usize, Pauli)],
    ) -> Result<Answer>;
}

// ── the baseline: a resolver that pays the whole width ───────────────

/// Which state representation an evaluating resolver builds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Inner {
    /// Dense amplitudes.
    Dense,
    /// Sparse amplitudes.
    Sparse,
    /// Matrix product state with a bond ceiling.
    Mps {
        /// Bond ceiling.
        max_bond: usize,
    },
}

impl Inner {
    fn build(&self, qubits: usize) -> Result<Box<dyn Backend<C64>>> {
        Ok(match *self {
            Inner::Dense => Box::new(DenseState::<C64>::new(qubits)?),
            Inner::Sparse => Box::new(SparseState::<C64>::new(qubits)?),
            Inner::Mps { max_bond } => Box::new(MpsState::<C64>::with_config(
                qubits,
                MpsConfig {
                    max_bond,
                    trunc_tol: 1e-14,
                },
            )?),
        })
    }
}

/// Runs the whole circuit on a full-width state. The honest baseline
/// every other resolver is measured against.
pub struct StateResolver<'a> {
    circuit: Circuit<C64>,
    registry: &'a GateRegistry<C64>,
    inner: Inner,
    name: String,
}

impl<'a> StateResolver<'a> {
    /// Wrap a circuit.
    pub fn new(circuit: Circuit<C64>, registry: &'a GateRegistry<C64>, inner: Inner) -> Self {
        StateResolver {
            circuit,
            registry,
            inner,
            name: "state".into(),
        }
    }

    fn run(
        &self,
        extra: Option<(usize, Rotation)>,
    ) -> Result<(f64, Cost, Box<dyn Backend<C64>>)> {
        let n = self.circuit.num_qubits();
        let mut state = self.inner.build(n)?;
        let bound = self.circuit.bind(self.registry)?;
        let mut gates = 0usize;
        for (i, g) in bound.gates().iter().enumerate() {
            if let Some((at, rot)) = extra {
                if i == at {
                    let (m, q) = rot.gate()?;
                    state.apply(&m, &q)?;
                    gates += 1;
                }
            }
            match &g.kernel {
                GateKernel::Matrix(m) => state.apply(m, &g.qubits)?,
                GateKernel::Diagonal(d) => state.apply_diagonal(d, &g.qubits)?,
            }
            gates += 1;
        }
        if let Some((at, rot)) = extra {
            if at >= bound.gates().len() {
                let (m, q) = rot.gate()?;
                state.apply(&m, &q)?;
                gates += 1;
            }
        }
        let cost = Cost {
            qubits: n,
            width: n,
            gates,
            bytes: state.memory_bytes(),
        };
        Ok((0.0, cost, state))
    }
}

impl Resolver for StateResolver<'_> {
    fn name(&self) -> &str {
        &self.name
    }
    fn width(&self) -> usize {
        self.circuit.num_qubits()
    }
    fn expectation(&mut self, ops: &[(usize, Pauli)]) -> Result<Answer> {
        let (_, cost, state) = self.run(None)?;
        let value = pauli_expectation(state.as_ref(), ops)?.re;
        Ok(Answer { value, cost })
    }
    fn perturbed_expectation(
        &mut self,
        at_gate: usize,
        site: usize,
        perturbation: Rotation,
        ops: &[(usize, Pauli)],
    ) -> Result<Answer> {
        let _ = site;
        let (_, cost, state) = self.run(Some((at_gate, perturbation)))?;
        let value = pauli_expectation(state.as_ref(), ops)?.re;
        Ok(Answer { value, cost })
    }
}

/// A rotation's gate matrix on compacted qubit indices.
fn remap_rotation(
    rot: Rotation,
    map: &HashMap<usize, usize>,
) -> Result<(GateMatrix<C64>, Vec<usize>)> {
    let (m, qs) = rot.gate()?;
    let mapped: Option<Vec<usize>> = qs.iter().map(|q| map.get(q).copied()).collect();
    let mapped = mapped.ok_or_else(|| {
        Error::InvalidState("perturbation acts outside the query's cone".into())
    })?;
    Ok((m, mapped))
}

// ── the cone resolver: nothing runs until a question arrives ─────────

/// Records a circuit and applies **nothing** until asked. Each query is
/// answered on its own backward cone, relabelled onto a compact
/// register, so the simulated dimension is `2^{|cone|}` and not `2^n`.
pub struct ConeResolver<'a> {
    circuit: Circuit<C64>,
    registry: &'a GateRegistry<C64>,
    inner: Inner,
    /// Widest cone that will be simulated before refusing.
    pub max_cone: usize,
    name: String,
}

impl<'a> ConeResolver<'a> {
    /// Record a circuit. No gate is applied here.
    pub fn new(circuit: Circuit<C64>, registry: &'a GateRegistry<C64>, inner: Inner) -> Self {
        ConeResolver {
            circuit,
            registry,
            inner,
            max_cone: 26,
            name: "cone".into(),
        }
    }

    /// Set the refusal width.
    pub fn with_max_cone(mut self, max_cone: usize) -> Self {
        self.max_cone = max_cone;
        self
    }

    /// The backward cone of `observed` over the gates in `from..`:
    /// `(cone qubits ascending, indices of the gates inside it)`.
    ///
    /// This is [`crate::causal::backward_cone`] plus the qubit set,
    /// which is the half that decides the cost.
    pub fn cone(&self, observed: &[usize], from: usize) -> (Vec<usize>, Vec<usize>) {
        let n = self.circuit.num_qubits();
        let mut active = vec![false; n];
        for &q in observed {
            if q < n {
                active[q] = true;
            }
        }
        let ops = self.circuit.ops();
        let mut keep = Vec::new();
        for i in (from..ops.len()).rev() {
            if ops[i].qubits().iter().any(|&q| q < n && active[q]) {
                keep.push(i);
                for &q in ops[i].qubits() {
                    if q < n {
                        active[q] = true;
                    }
                }
            }
        }
        keep.reverse();
        let cone: Vec<usize> = (0..n).filter(|&q| active[q]).collect();
        (cone, keep)
    }

    /// Qubits the backward cone of `observed` reaches — the cost of the
    /// query before paying it.
    pub fn cone_width(&self, observed: &[usize]) -> usize {
        self.cone(observed, 0).0.len()
    }

    /// Build the compact circuit on the cone's own indices.
    fn compact(&self, cone: &[usize], keep: &[usize]) -> (Circuit<C64>, HashMap<usize, usize>) {
        let map: HashMap<usize, usize> =
            cone.iter().enumerate().map(|(i, &q)| (q, i)).collect();
        let mut out = Circuit::new(cone.len());
        for &i in keep {
            let relabel = |qs: &[usize]| -> Vec<usize> { qs.iter().map(|q| map[q]).collect() };
            match &self.circuit.ops()[i] {
                Op::Named {
                    name,
                    params,
                    qubits,
                } => {
                    out.gate(name.clone(), params.clone(), relabel(qubits));
                }
                Op::Raw {
                    label,
                    matrix,
                    qubits,
                } => {
                    out.raw(label.clone(), matrix.clone(), relabel(qubits));
                }
                Op::Diagonal {
                    label,
                    entries,
                    qubits,
                } => {
                    out.diagonal(label.clone(), entries.clone(), relabel(qubits));
                }
            }
        }
        (out, map)
    }

    fn answer(
        &self,
        ops: &[(usize, Pauli)],
        extra: Option<(usize, Rotation)>,
    ) -> Result<Answer> {
        let observed: Vec<usize> = ops.iter().map(|&(q, _)| q).collect();
        let (cone, keep) = self.cone(&observed, 0);
        if cone.len() > self.max_cone {
            return Err(Error::TooManyQubits {
                requested: cone.len(),
                max: self.max_cone,
            });
        }
        let (compact, map) = self.compact(&cone, &keep);
        let mut state = self.inner.build(cone.len())?;
        let bound = compact.bind(self.registry)?;
        let mut gates = 0usize;
        // `extra` is positioned by ORIGINAL gate index; find where that
        // lands among the kept gates.
        let insert_before = extra.map(|(at, _)| keep.iter().filter(|&&i| i < at).count());
        for (i, g) in bound.gates().iter().enumerate() {
            if let (Some(pos), Some((_, rot))) = (insert_before, extra) {
                if i == pos {
                    let (m, q) = remap_rotation(rot, &map)?;
                    state.apply(&m, &q)?;
                    gates += 1;
                }
            }
            match &g.kernel {
                GateKernel::Matrix(m) => state.apply(m, &g.qubits)?,
                GateKernel::Diagonal(d) => state.apply_diagonal(d, &g.qubits)?,
            }
            gates += 1;
        }
        if let (Some(pos), Some((_, rot))) = (insert_before, extra) {
            if pos >= bound.gates().len() {
                let (m, q) = remap_rotation(rot, &map)?;
                state.apply(&m, &q)?;
                gates += 1;
            }
        }
        let remapped: Vec<(usize, Pauli)> = ops.iter().map(|&(q, p)| (map[&q], p)).collect();
        let value = pauli_expectation(state.as_ref(), &remapped)?.re;
        Ok(Answer {
            value,
            cost: Cost {
                qubits: cone.len(),
                width: self.circuit.num_qubits(),
                gates,
                bytes: state.memory_bytes(),
            },
        })
    }
}

impl Resolver for ConeResolver<'_> {
    fn name(&self) -> &str {
        &self.name
    }
    fn width(&self) -> usize {
        self.circuit.num_qubits()
    }
    fn expectation(&mut self, ops: &[(usize, Pauli)]) -> Result<Answer> {
        self.answer(ops, None)
    }
    fn perturbed_expectation(
        &mut self,
        at_gate: usize,
        site: usize,
        perturbation: Rotation,
        ops: &[(usize, Pauli)],
    ) -> Result<Answer> {
        let _ = site;
        self.answer(ops, Some((at_gate, perturbation)))
    }

    /// Overridden so that a causally disconnected perturbation is
    /// answered **exactly**, at **zero** cost: if the site is not in the
    /// observable's backward cone over the gates that follow it, nothing
    /// done there can move the answer, and no simulation is needed to
    /// establish that.
    fn response(
        &mut self,
        at_gate: usize,
        site: usize,
        perturbation: Rotation,
        ops: &[(usize, Pauli)],
    ) -> Result<Answer> {
        let observed: Vec<usize> = ops.iter().map(|&(q, _)| q).collect();
        let (reach, _) = self.cone(&observed, at_gate);
        if !reach.contains(&site) {
            return Ok(Answer {
                value: 0.0,
                cost: Cost {
                    qubits: 0,
                    width: self.circuit.num_qubits(),
                    gates: 0,
                    bytes: 0,
                },
            });
        }
        let base = self.expectation(ops)?;
        let moved = self.perturbed_expectation(at_gate, site, perturbation, ops)?;
        Ok(Answer {
            value: moved.value - base.value,
            cost: Cost {
                qubits: base.cost.qubits.max(moved.cost.qubits),
                width: base.cost.width,
                gates: base.cost.gates + moved.cost.gates,
                bytes: base.cost.bytes.max(moved.cost.bytes),
            },
        })
    }
}

// ── a resolver that holds no state at all ────────────────────────────

/// Answers by propagating the **observable** backwards, never building a
/// state.
///
/// This is the structural point of the module. There is no refactoring
/// of [`Backend`] that would admit this object:
/// it has no amplitudes to hand out, no support to enumerate, and no
/// basis to load into. It answers questions, and answering questions is
/// all a representation has to do. Its [`Cost::qubits`] is **zero** —
/// not small, zero — because no register is ever instantiated.
///
/// Its cost lives in a different currency: terms in the Pauli sum, which
/// grows with the circuit's *branching* gates rather than its width.
pub struct HeisenbergResolver {
    rotations: Vec<Rotation>,
    qubits: usize,
    /// Propagation settings; a nonzero threshold trades exactness for
    /// terms and reports the bound it bought.
    pub config: heisenberg::Config,
    /// `Σ|c|` discarded by the last answer — zero for an exact walk.
    pub last_error_bound: f64,
    name: String,
}

impl HeisenbergResolver {
    /// Record a circuit as Pauli rotations.
    pub fn new(rotations: Vec<Rotation>, qubits: usize) -> Self {
        HeisenbergResolver {
            rotations,
            qubits,
            config: heisenberg::Config {
                threshold: 0.0,
                max_terms: None,
                checkpoint_every: 0,
                exclusion: true,
                retire_frozen: false,
            },
            last_error_bound: 0.0,
            name: "heisenberg".into(),
        }
    }

    /// Set the truncation threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// The observable as a Pauli sum on the raw `X^x Z^z` basis.
    fn observable(&self, ops: &[(usize, Pauli)]) -> PauliSum {
        let (mut x, mut z) = (0u64, 0u64);
        for &(q, p) in ops {
            match p {
                Pauli::X => x |= 1 << q,
                Pauli::Z => z |= 1 << q,
                Pauli::Y => {
                    x |= 1 << q;
                    z |= 1 << q;
                }
                Pauli::I => {}
            }
        }
        // `PauliSum` carries coefficients of the raw product; the
        // Hermitian observable is `i^{|x∧z|}` times it.
        let mut s = PauliSum::zero();
        s.add((x, z), crate::heisenberg::axis_operator_phase((x, z)));
        s
    }
}

impl Resolver for HeisenbergResolver {
    fn name(&self) -> &str {
        &self.name
    }
    fn width(&self) -> usize {
        self.qubits
    }
    fn expectation(&mut self, ops: &[(usize, Pauli)]) -> Result<Answer> {
        let obs = self.observable(ops);
        let p = heisenberg::propagate(&obs, &self.rotations, &self.config)?;
        self.last_error_bound = p.error_bound();
        Ok(Answer {
            value: p.expectation(),
            cost: Cost {
                qubits: 0,
                width: self.qubits,
                gates: self.rotations.len(),
                bytes: p.peak_terms * 32,
            },
        })
    }
    fn perturbed_expectation(
        &mut self,
        at_gate: usize,
        _site: usize,
        perturbation: Rotation,
        ops: &[(usize, Pauli)],
    ) -> Result<Answer> {
        let mut rots = self.rotations.clone();
        rots.insert(at_gate.min(rots.len()), perturbation);
        let obs = self.observable(ops);
        let p = heisenberg::propagate(&obs, &rots, &self.config)?;
        self.last_error_bound = p.error_bound();
        Ok(Answer {
            value: p.expectation(),
            cost: Cost {
                qubits: 0,
                width: self.qubits,
                gates: rots.len(),
                bytes: p.peak_terms * 32,
            },
        })
    }
}
