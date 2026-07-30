//! Progressive gate-result memoization: qubit operations as `n`-wide
//! operation objects, with the computational geometry memoized away, the
//! whole computational path held at once, and a journal that unwinds and
//! rewinds the state so different configurations can be re-explored.
//!
//! # Operations as `n`-wide objects
//!
//! A circuit is normally a sequence of "this gate, on those qubits". That
//! spelling ties an operation to *where it sits*, so a simulator rebuilds
//! the same arithmetic every time the same structure recurs — every rung
//! of a brickwork, every repeat of an oracle, every branch of a parameter
//! sweep.
//!
//! [`MemoPlan`] rewrites the circuit into [`PlanOp`]s: each one an
//! operation over a **support** (a set of qubits) whose content lives in a
//! shared entry table. Consecutive gates are fused into one operator per
//! support while they fit inside [`MemoConfig::max_fuse`], and the fused
//! operator is looked up **by its exact contents** — every coefficient of
//! every amplitude, bit for bit, via [`crate::scalar::Scalar::coeffs`].
//!
//! Content addressing is what memoizes the geometry away. A fused
//! operator over a sorted support is indexed by position *within* that
//! support, so it does not know which qubits it sits on: a `cx` on
//! `(0, 1)` and a `cx` on `(7, 8)` produce the identical matrix and
//! therefore the identical entry. Nothing has to be canonicalized by
//! hand, and because the key is the exact bit pattern rather than a hash,
//! a hit is an identity, never a guess.
//!
//! # Fusion is sound by disjointness
//!
//! Several supports stay open at once, and they are **disjoint by
//! construction**. A gate joins the open groups it touches when their
//! union still fits inside `max_fuse` — merging disjoint groups is exact
//! because disjoint operators commute, so the merged operator is their
//! product in any order. When the union would overflow, the touched
//! groups are **committed** and the gate opens a fresh one.
//!
//! Committed operations are emitted in **commit order**, which is a valid
//! linearization for a reason worth stating: a qubit is owned by at most
//! one open group, ownership transfers only when that group commits, and a
//! group commits only after every gate it contains has been seen. So for
//! any qubit, the groups touching it commit in the same order as its
//! gates appear.
//!
//! Emitting in *opening* order is **not** valid, and the difference is not
//! academic — a still-open group can acquire a qubit that an
//! already-committed group used, at which point its opening position
//! predates gates it does not contain. That produced a measured
//! 1.4-amplitude error on a random circuit before it was fixed, and
//! `tests/memo.rs` pins the case.
//!
//! # The path, and re-exploring it
//!
//! [`Explorer`] runs a plan against any registered backend while
//! journalling: state snapshots at a configurable stride, so
//! [`Explorer::rewind`] restores the nearest checkpoint and replays
//! forward to any earlier step. [`explore`] uses that to run a set of
//! variants sharing a prefix — the prefix is evolved once and rewound to,
//! rather than recomputed per variant — and reports both the counted work
//! and the measured wall clock, together with the deviation against
//! running each variant from scratch.

use std::collections::HashMap;

use crate::backend::Backend;
use crate::circuit::{BoundCircuit, Circuit, GateKernel};
use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::scalar::Scalar;
use crate::sim::Simulator;

/// Widest support a fused operator may cover, by default — a 4-qubit
/// operator is a 16×16 matrix, small enough that fusing is cheaper than
/// the gates it replaces.
pub const DEFAULT_MAX_FUSE: usize = 4;

/// Knobs for [`MemoPlan::plan`].
#[derive(Debug, Clone)]
pub struct MemoConfig {
    /// Widest support a fused operator may cover (default
    /// [`DEFAULT_MAX_FUSE`]). One means no fusion at all, which is still
    /// memoized — repeated single-qubit gates share entries.
    pub max_fuse: usize,
    /// Whether to keep diagonal kernels diagonal when a group never grows
    /// past one gate. Diagonal application is `O(support)` with `O(2^k)`
    /// storage in the shipped backends, so a lone diagonal gate is
    /// cheaper left alone than promoted to a dense matrix (default true).
    pub preserve_diagonal: bool,
}

impl Default for MemoConfig {
    fn default() -> Self {
        MemoConfig {
            max_fuse: DEFAULT_MAX_FUSE,
            preserve_diagonal: true,
        }
    }
}

/// One operation of a plan: a support, and the entry holding what to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanOp {
    /// Target qubits, ascending — the operator's sub-index bit `b`
    /// corresponds to `support[b]`.
    pub support: Vec<usize>,
    /// Index into [`MemoPlan::entries`].
    pub entry: usize,
    /// Original circuit gates folded into this operation.
    pub fused_from: usize,
}

/// What the memo achieved, counted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoStats {
    /// Gates in the source circuit.
    pub gates_in: usize,
    /// Operations in the plan after fusion.
    pub ops_out: usize,
    /// Distinct entries stored — the number of operators actually built.
    pub entries: usize,
    /// Plan operations that found an existing entry.
    pub hits: usize,
    /// Plan operations that had to build one.
    pub misses: usize,
    /// Bytes the entry table holds.
    pub entry_bytes: usize,
    /// Widest support any operation covers.
    pub widest_support: usize,
}

impl MemoStats {
    /// Operations per distinct entry — how much the geometry repeated.
    /// One means every operation was structurally unique.
    pub fn reuse(&self) -> f64 {
        if self.entries == 0 {
            return 1.0;
        }
        self.ops_out as f64 / self.entries as f64
    }

    /// Fraction of plan operations that hit an existing entry.
    pub fn hit_rate(&self) -> f64 {
        if self.ops_out == 0 {
            return 0.0;
        }
        self.hits as f64 / self.ops_out as f64
    }

    /// Source gates per plan operation — how much fusion collapsed.
    pub fn fusion(&self) -> f64 {
        if self.ops_out == 0 {
            return 1.0;
        }
        self.gates_in as f64 / self.ops_out as f64
    }
}

/// A circuit rewritten as `n`-wide operations over a shared entry table.
#[derive(Debug, Clone)]
pub struct MemoPlan<S: Scalar> {
    num_qubits: usize,
    entries: Vec<GateKernel<S>>,
    ops: Vec<PlanOp>,
    stats: MemoStats,
}

/// An open fusion group: a support and the operator accumulated on it.
struct Group<S: Scalar> {
    support: Vec<usize>,
    /// `None` while the group holds exactly one diagonal kernel that has
    /// not been promoted, so a lone diagonal can stay diagonal.
    matrix: Option<GateMatrix<S>>,
    diagonal: Option<Vec<S>>,
    gates: usize,
    /// Source position of the group's first gate, so committed groups can
    /// be emitted in a deterministic order.
    opened_at: usize,
}

impl<S: Scalar> MemoPlan<S> {
    /// Rewrite `circuit` into memoized `n`-wide operations.
    pub fn plan(circuit: &BoundCircuit<S>, cfg: &MemoConfig) -> Result<Self> {
        if cfg.max_fuse == 0 {
            return Err(Error::InvalidState(
                "max_fuse must be at least 1: an operation covers at least its own qubits".into(),
            ));
        }
        let num_qubits = circuit.num_qubits();
        let mut open: Vec<Group<S>> = Vec::new();
        // Commit order, which is the emission order — see the module docs
        // for why opening order would be wrong.
        let mut committed: Vec<(Vec<usize>, GateKernel<S>, usize)> = Vec::new();

        for (position, gate) in circuit.gates().iter().enumerate() {
            let mut targets: Vec<usize> = gate.qubits.clone();
            targets.sort_unstable();
            targets.dedup();

            // Which open groups does this gate touch? Open supports are
            // disjoint, so this is the set of current owners of its qubits.
            let touched: Vec<usize> = open
                .iter()
                .enumerate()
                .filter(|(_, g)| g.support.iter().any(|q| targets.contains(q)))
                .map(|(i, _)| i)
                .collect();

            let mut union = targets.clone();
            for &i in &touched {
                union.extend(open[i].support.iter().copied());
            }
            union.sort_unstable();
            union.dedup();

            if union.len() <= cfg.max_fuse && !touched.is_empty() {
                // Merge every touched group into one operator over the
                // union, then multiply the gate on. Exact: the groups are
                // pairwise disjoint, so their operators commute and the
                // product is order-independent.
                let mut merged: Option<GateMatrix<S>> = None;
                let mut gates = 0usize;
                let mut opened_at = position;
                for &i in touched.iter().rev() {
                    let mut group = open.remove(i);
                    promote(&mut group, &union)?;
                    let part = group.matrix.take().expect("promoted");
                    merged = Some(match merged {
                        None => part,
                        Some(acc) => part.matmul(&acc),
                    });
                    gates += group.gates;
                    opened_at = opened_at.min(group.opened_at);
                }
                let acc = merged.expect("touched is non-empty");
                let addition = embed(&gate.kernel, &gate.qubits, &union)?;
                open.push(Group {
                    support: union,
                    matrix: Some(addition.matmul(&acc)),
                    diagonal: None,
                    gates: gates + 1,
                    opened_at,
                });
            } else {
                // Commit the touched groups so their qubit lines are free
                // before the new gate lands on them. Removal goes by
                // DESCENDING INDEX — `open` is not ordered by `opened_at`,
                // because merging pushes a group carrying the minimum of
                // its parents' opening positions — and the committed
                // groups are then ordered oldest first for determinism.
                let mut closing = Vec::with_capacity(touched.len());
                for &i in touched.iter().rev() {
                    closing.push(open.remove(i));
                }
                closing.sort_by_key(|g| g.opened_at);
                for group in closing {
                    committed.push(close(group)?);
                }
                if targets.len() > cfg.max_fuse {
                    // Wider than any group may be: its own operation,
                    // memoized but never fused.
                    let kernel = normalize(&gate.kernel, &gate.qubits, &targets)?;
                    committed.push((targets, kernel, 1));
                } else {
                    open.push(new_group(gate, &targets, position, cfg)?);
                }
            }
        }
        // Leftovers are all still open, hence pairwise disjoint, so any
        // order is exact; oldest first keeps the plan deterministic.
        let mut leftovers: Vec<Group<S>> = open;
        leftovers.sort_by_key(|g| g.opened_at);
        for group in leftovers {
            committed.push(close(group)?);
        }

        // The memo: content-addressed on the exact coefficient bits, so a
        // hit is an identity rather than a hash guess.
        let mut table: HashMap<Vec<u64>, usize> = HashMap::new();
        let mut entries: Vec<GateKernel<S>> = Vec::new();
        let mut ops = Vec::with_capacity(committed.len());
        let (mut hits, mut misses) = (0usize, 0usize);
        let mut widest_support = 0usize;
        let gates_in = circuit.gates().len();
        for (support, kernel, fused_from) in committed {
            let key = kernel_key(&kernel);
            let entry = match table.get(&key) {
                Some(&index) => {
                    hits += 1;
                    index
                }
                None => {
                    misses += 1;
                    let index = entries.len();
                    table.insert(key, index);
                    entries.push(kernel);
                    index
                }
            };
            widest_support = widest_support.max(support.len());
            ops.push(PlanOp {
                support,
                entry,
                fused_from,
            });
        }
        let entry_bytes = entries.iter().map(kernel_bytes::<S>).sum();
        let stats = MemoStats {
            gates_in,
            ops_out: ops.len(),
            entries: entries.len(),
            hits,
            misses,
            entry_bytes,
            widest_support,
        };
        Ok(MemoPlan {
            num_qubits,
            entries,
            ops,
            stats,
        })
    }

    /// Plan straight from an unbound circuit.
    pub fn from_circuit(
        circuit: &Circuit<S>,
        registry: &crate::registry::GateRegistry<S>,
        cfg: &MemoConfig,
    ) -> Result<Self> {
        Self::plan(&circuit.bind(registry)?, cfg)
    }

    /// Circuit width.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The plan's operations, in order.
    pub fn ops(&self) -> &[PlanOp] {
        &self.ops
    }

    /// The shared entry table — each distinct operator, stored once.
    pub fn entries(&self) -> &[GateKernel<S>] {
        &self.entries
    }

    /// What the memo achieved.
    pub fn stats(&self) -> &MemoStats {
        &self.stats
    }

    /// Apply one plan operation to a backend.
    pub fn step(&self, backend: &mut dyn Backend<S>, index: usize) -> Result<()> {
        let op = self
            .ops
            .get(index)
            .ok_or_else(|| Error::InvalidState(format!("plan has no operation {index}")))?;
        match &self.entries[op.entry] {
            GateKernel::Matrix(m) => backend.apply(m, &op.support),
            GateKernel::Diagonal(d) => backend.apply_diagonal(d, &op.support),
        }
    }

    /// Apply operations `from..to` in order.
    pub fn run_range(&self, backend: &mut dyn Backend<S>, from: usize, to: usize) -> Result<()> {
        if backend.num_qubits() != self.num_qubits {
            return Err(Error::WidthMismatch {
                circuit: self.num_qubits,
                backend: backend.num_qubits(),
            });
        }
        let _scope = crate::guard::enter();
        for index in from..to.min(self.ops.len()) {
            crate::guard::checkpoint()?;
            self.step(backend, index)?;
        }
        Ok(())
    }

    /// Apply the whole plan.
    pub fn run(&self, backend: &mut dyn Backend<S>) -> Result<()> {
        self.run_range(backend, 0, self.ops.len())
    }
}

/// A group that has only ever seen one gate keeps that gate's kernel; one
/// that has to accumulate is promoted to a dense matrix over its support.
fn new_group<S: Scalar>(
    gate: &crate::circuit::BoundGate<S>,
    targets: &[usize],
    position: usize,
    cfg: &MemoConfig,
) -> Result<Group<S>> {
    let support = targets.to_vec();
    match (&gate.kernel, cfg.preserve_diagonal) {
        (GateKernel::Diagonal(d), true) if gate.qubits.len() == support.len() => {
            // Reorder the diagonal into ascending support order.
            let reordered = reorder_diagonal(d, &gate.qubits, &support)?;
            Ok(Group {
                support,
                matrix: None,
                diagonal: Some(reordered),
                gates: 1,
                opened_at: position,
            })
        }
        _ => Ok(Group {
            matrix: Some(embed(&gate.kernel, &gate.qubits, &support)?),
            support,
            diagonal: None,
            gates: 1,
            opened_at: position,
        }),
    }
}

/// Make sure a group carries a matrix over `support`, materializing a
/// held diagonal and widening an existing matrix as needed.
fn promote<S: Scalar>(group: &mut Group<S>, support: &[usize]) -> Result<()> {
    if let Some(d) = group.diagonal.take() {
        group.matrix = Some(embed(
            &GateKernel::Diagonal(d),
            &group.support.clone(),
            support,
        )?);
        return Ok(());
    }
    let current = group
        .matrix
        .take()
        .ok_or_else(|| Error::InvalidState("a group holds neither matrix nor diagonal".into()))?;
    if group.support == support {
        group.matrix = Some(current);
    } else {
        group.matrix = Some(embed(
            &GateKernel::Matrix(current),
            &group.support.clone(),
            support,
        )?);
    }
    Ok(())
}

/// Close a group into a committed operation.
#[allow(clippy::type_complexity)]
fn close<S: Scalar>(group: Group<S>) -> Result<(Vec<usize>, GateKernel<S>, usize)> {
    let Group {
        support,
        matrix,
        diagonal,
        gates,
        opened_at,
    } = group;
    let _ = opened_at;
    let kernel = match (matrix, diagonal) {
        (Some(m), _) => GateKernel::Matrix(m),
        (None, Some(d)) => GateKernel::Diagonal(d),
        (None, None) => {
            return Err(Error::InvalidState(
                "a group holds neither matrix nor diagonal".into(),
            ))
        }
    };
    Ok((support, kernel, gates))
}

/// A kernel expressed over `support` in ascending order, without fusing.
fn normalize<S: Scalar>(
    kernel: &GateKernel<S>,
    qubits: &[usize],
    support: &[usize],
) -> Result<GateKernel<S>> {
    match kernel {
        GateKernel::Diagonal(d) if qubits.len() == support.len() => {
            Ok(GateKernel::Diagonal(reorder_diagonal(d, qubits, support)?))
        }
        other => Ok(GateKernel::Matrix(embed(other, qubits, support)?)),
    }
}

/// Reorder a diagonal from `qubits` order into ascending `support` order.
fn reorder_diagonal<S: Scalar>(
    entries: &[S],
    qubits: &[usize],
    support: &[usize],
) -> Result<Vec<S>> {
    let k = support.len();
    if entries.len() != 1usize << k {
        return Err(Error::BadDimension {
            expected: 1usize << k,
            got: entries.len(),
        });
    }
    let positions: Vec<usize> = qubits
        .iter()
        .map(|q| {
            support
                .iter()
                .position(|s| s == q)
                .ok_or_else(|| Error::InvalidState(format!("qubit {q} is outside the support")))
        })
        .collect::<Result<_>>()?;
    let mut out = vec![S::zero(); 1usize << k];
    for (index, slot) in out.iter_mut().enumerate() {
        // Gather the gate's own sub-index out of the support index.
        let sub = positions
            .iter()
            .enumerate()
            .fold(0usize, |acc, (bit, &pos)| {
                acc | (((index >> pos) & 1) << bit)
            });
        *slot = entries[sub];
    }
    Ok(out)
}

/// Embed a kernel acting on `qubits` into a dense matrix over `support`
/// (ascending), the identity on the rest of the support.
fn embed<S: Scalar>(
    kernel: &GateKernel<S>,
    qubits: &[usize],
    support: &[usize],
) -> Result<GateMatrix<S>> {
    let k = support.len();
    let dim = 1usize << k;
    let positions: Vec<usize> = qubits
        .iter()
        .map(|q| {
            support
                .iter()
                .position(|s| s == q)
                .ok_or_else(|| Error::InvalidState(format!("qubit {q} is outside the support")))
        })
        .collect::<Result<_>>()?;
    let gate_mask = positions.iter().fold(0usize, |acc, &p| acc | (1 << p));
    let gdim = 1usize << positions.len();
    let mut out = GateMatrix::zeros(dim)?;
    match kernel {
        GateKernel::Diagonal(d) => {
            if d.len() != gdim {
                return Err(Error::BadDimension {
                    expected: gdim,
                    got: d.len(),
                });
            }
            for index in 0..dim {
                let sub = gather(index, &positions);
                out.set(index, index, d[sub]);
            }
        }
        GateKernel::Matrix(m) => {
            if m.dim() != gdim {
                return Err(Error::BadDimension {
                    expected: gdim,
                    got: m.dim(),
                });
            }
            for col in 0..dim {
                let sub = gather(col, &positions);
                let rest = col & !gate_mask;
                for row_sub in 0..gdim {
                    let value = m.get(row_sub, sub);
                    if !value.is_zero(0.0) {
                        out.set(rest | scatter(row_sub, &positions), col, value);
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Gather the sub-index the gate sees out of a support index.
fn gather(index: usize, positions: &[usize]) -> usize {
    positions
        .iter()
        .enumerate()
        .fold(0usize, |acc, (bit, &pos)| {
            acc | (((index >> pos) & 1) << bit)
        })
}

/// Scatter a gate sub-index back into support bit positions.
fn scatter(sub: usize, positions: &[usize]) -> usize {
    positions
        .iter()
        .enumerate()
        .fold(0usize, |acc, (bit, &pos)| acc | (((sub >> bit) & 1) << pos))
}

/// The exact content key of a kernel: a tag, the dimension, and every
/// coefficient of every amplitude as raw bits.
///
/// Exact bits rather than a hash, so a table hit is an identity. `-0.0`
/// is folded to `0.0` first, because the two are numerically equal and
/// would otherwise key differently.
fn kernel_key<S: Scalar>(kernel: &GateKernel<S>) -> Vec<u64> {
    let (tag, values): (u64, Vec<S>) = match kernel {
        GateKernel::Matrix(m) => (0, m.data().to_vec()),
        GateKernel::Diagonal(d) => (1, d.clone()),
    };
    let mut key = Vec::with_capacity(2 + values.len() * S::DIM);
    key.push(tag);
    key.push(values.len() as u64);
    for value in values {
        for coefficient in value.coeffs() {
            let normalized = if coefficient == 0.0 { 0.0 } else { coefficient };
            key.push(normalized.to_bits());
        }
    }
    key
}

/// Bytes a stored kernel occupies.
fn kernel_bytes<S: Scalar>(kernel: &GateKernel<S>) -> usize {
    let count = match kernel {
        GateKernel::Matrix(m) => m.dim() * m.dim(),
        GateKernel::Diagonal(d) => d.len(),
    };
    count * std::mem::size_of::<S>()
}

/// A journalled state snapshot.
#[derive(Debug, Clone)]
struct Checkpoint<S: Scalar> {
    step: usize,
    stored: Vec<(u64, S)>,
}

/// A plan under execution, journalled so the state can be unwound and
/// rewound to re-explore different configurations.
pub struct Explorer<'a, S: Scalar> {
    plan: &'a MemoPlan<S>,
    state: Box<dyn Backend<S>>,
    checkpoints: Vec<Checkpoint<S>>,
    stride: usize,
    cursor: usize,
    replayed: usize,
    rewinds: usize,
}

impl<'a, S: Scalar> Explorer<'a, S> {
    /// Start an explorer on `backend` at step 0, snapshotting every
    /// `stride` operations (`stride = 0` snapshots only the origin).
    pub fn new(
        sim: &Simulator<S>,
        backend: &str,
        plan: &'a MemoPlan<S>,
        stride: usize,
    ) -> Result<Self> {
        let mut state = sim.backends().create(backend, plan.num_qubits())?;
        state.reset();
        let mut explorer = Explorer {
            plan,
            state,
            checkpoints: Vec::new(),
            stride,
            cursor: 0,
            replayed: 0,
            rewinds: 0,
        };
        explorer.snapshot();
        Ok(explorer)
    }

    fn snapshot(&mut self) {
        let mut stored = Vec::with_capacity(self.state.nonzero_count());
        self.state
            .for_each_nonzero(&mut |index, amp| stored.push((index, amp)));
        stored.sort_by_key(|&(index, _)| index);
        self.checkpoints.push(Checkpoint {
            step: self.cursor,
            stored,
        });
    }

    /// The operation index the state currently sits at.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// The evolved state.
    pub fn state(&self) -> &dyn Backend<S> {
        self.state.as_ref()
    }

    /// Snapshots held.
    pub fn checkpoints(&self) -> usize {
        self.checkpoints.len()
    }

    /// Amplitudes across every snapshot — the journal's measured cost.
    pub fn journal_amplitudes(&self) -> usize {
        self.checkpoints.iter().map(|c| c.stored.len()).sum()
    }

    /// Plan operations applied in total, replays included — the counted
    /// work, as distinct from the cursor's position.
    pub fn applied(&self) -> usize {
        self.replayed
    }

    /// Rewinds performed.
    pub fn rewinds(&self) -> usize {
        self.rewinds
    }

    /// The current state as sorted stored amplitudes — the journal entry
    /// a fork starts from.
    pub fn snapshot_now(&self) -> Vec<(u64, S)> {
        let mut stored = Vec::with_capacity(self.state.nonzero_count());
        self.state
            .for_each_nonzero(&mut |index, amp| stored.push((index, amp)));
        stored.sort_by_key(|&(index, _)| index);
        stored
    }

    /// Advance to operation `target`, applying whatever lies between and
    /// snapshotting on the stride.
    pub fn advance_to(&mut self, target: usize) -> Result<()> {
        let target = target.min(self.plan.ops().len());
        if target < self.cursor {
            return Err(Error::InvalidState(format!(
                "cannot advance backwards from {} to {target}; use rewind",
                self.cursor
            )));
        }
        let _scope = crate::guard::enter();
        while self.cursor < target {
            crate::guard::checkpoint()?;
            self.plan.step(self.state.as_mut(), self.cursor)?;
            self.cursor += 1;
            self.replayed += 1;
            if self.stride > 0 && self.cursor % self.stride == 0 {
                self.snapshot();
            }
        }
        Ok(())
    }

    /// Advance to the end of the plan.
    pub fn advance(&mut self) -> Result<()> {
        self.advance_to(self.plan.ops().len())
    }

    /// Unwind to operation `target`: restore the nearest snapshot at or
    /// before it, then replay forward. Exact — the restored state is the
    /// snapshot's amplitudes, and the replay is the plan's own operators.
    pub fn rewind(&mut self, target: usize) -> Result<()> {
        if target > self.cursor {
            return Err(Error::InvalidState(format!(
                "rewind target {target} is ahead of the cursor {}",
                self.cursor
            )));
        }
        let nearest = self
            .checkpoints
            .iter()
            .filter(|c| c.step <= target)
            .max_by_key(|c| c.step)
            .ok_or_else(|| Error::InvalidState("no snapshot at or before the target".into()))?;
        let (step, stored) = (nearest.step, nearest.stored.clone());
        self.state.reset();
        self.state.load(&stored)?;
        self.cursor = step;
        self.rewinds += 1;
        // Snapshots ahead of the restore point describe a path no longer
        // being followed; dropping them keeps the journal a history of
        // where the state actually is.
        self.checkpoints.retain(|c| c.step <= step);
        // Replay without snapshotting mid-way is wrong: the stride must
        // keep producing snapshots so a later rewind is still cheap.
        self.advance_to(target)
    }
}

/// What exploring a set of variants over a shared prefix cost.
#[derive(Debug, Clone)]
pub struct BranchReport {
    /// Plan operations in the shared prefix.
    pub prefix_ops: usize,
    /// Variants explored.
    pub variants: usize,
    /// Plan operations applied when each variant runs from scratch.
    pub applied_naive: usize,
    /// Plan operations applied when the prefix is evolved once and
    /// rewound to.
    pub applied_memoized: usize,
    /// Median nanoseconds, running every variant from scratch.
    pub nanos_naive: usize,
    /// Median nanoseconds, sharing the prefix through the journal.
    pub nanos_memoized: usize,
    /// Snapshots the journal held.
    pub checkpoints: usize,
    /// Amplitudes the journal stored.
    pub journal_amplitudes: usize,
    /// Worst amplitude deviation between the two routes, over every
    /// variant — the correctness of the sharing, measured.
    pub deviation: f64,
}

impl BranchReport {
    /// Counted work saved, as a ratio.
    pub fn counted_speedup(&self) -> f64 {
        if self.applied_memoized == 0 {
            return 1.0;
        }
        self.applied_naive as f64 / self.applied_memoized as f64
    }

    /// Measured wall-clock speedup.
    pub fn measured_speedup(&self) -> f64 {
        if self.nanos_memoized == 0 {
            return 1.0;
        }
        self.nanos_naive as f64 / self.nanos_memoized as f64
    }
}

/// Explore `variants` over a shared `prefix`, both ways, and compare.
///
/// The memoized route evolves the prefix once and rewinds to it per
/// variant; the naive route runs `prefix + variant` from scratch each
/// time. Both are real runs on `backend`, and the deviation between their
/// final states is measured for every variant — sharing that changed an
/// answer would show up here rather than as a silent speedup.
pub fn explore<S: Scalar>(
    sim: &Simulator<S>,
    backend: &str,
    prefix: &Circuit<S>,
    variants: &[Circuit<S>],
    cfg: &MemoConfig,
) -> Result<BranchReport> {
    if variants.is_empty() {
        return Err(Error::InvalidState(
            "exploring needs at least one variant".into(),
        ));
    }
    let all: Vec<usize> = (0..prefix.num_qubits()).collect();
    let mut plans = Vec::with_capacity(variants.len());
    let mut tails = Vec::with_capacity(variants.len());
    for variant in variants {
        if variant.num_qubits() != prefix.num_qubits() {
            return Err(Error::WidthMismatch {
                circuit: variant.num_qubits(),
                backend: prefix.num_qubits(),
            });
        }
        let mut whole = prefix.clone();
        whole.append(variant, &all);
        // The naive route gets the BETTER plan: fusing the whole circuit
        // can merge across the prefix/variant boundary, which the shared
        // route cannot. That is the honest comparison to lose or win.
        plans.push(MemoPlan::from_circuit(&whole, sim.registry(), cfg)?);
        tails.push(MemoPlan::from_circuit(variant, sim.registry(), cfg)?);
    }
    let prefix_plan = MemoPlan::from_circuit(prefix, sim.registry(), cfg)?;
    let prefix_ops = prefix_plan.ops().len();

    // Naive: every variant from scratch, three times, median kept.
    let mut naive_samples = Vec::with_capacity(3);
    let mut naive_final: Vec<Vec<(u64, S)>> = Vec::new();
    for repetition in 0..3 {
        let start = std::time::Instant::now();
        let mut finals = Vec::with_capacity(plans.len());
        for plan in &plans {
            let mut state = sim.backends().create(backend, plan.num_qubits())?;
            state.reset();
            plan.run(state.as_mut())?;
            let mut stored = Vec::new();
            state.for_each_nonzero(&mut |index, amp| stored.push((index, amp)));
            stored.sort_by_key(|&(index, _)| index);
            finals.push(stored);
        }
        naive_samples.push(start.elapsed().as_nanos() as usize);
        if repetition == 0 {
            naive_final = finals;
        }
    }
    naive_samples.sort_unstable();

    // Memoized: the prefix once, rewound to per variant.
    let mut memo_samples = Vec::with_capacity(3);
    let mut memo_final: Vec<Vec<(u64, S)>> = Vec::new();
    let mut applied_memoized = 0usize;
    let mut checkpoints = 0usize;
    let mut journal_amplitudes = 0usize;
    for repetition in 0..3 {
        let start = std::time::Instant::now();
        let mut finals = Vec::with_capacity(plans.len());
        let mut explorer = Explorer::new(sim, backend, &prefix_plan, prefix_ops.max(1))?;
        explorer.advance()?;
        // The prefix state, evolved once and journalled.
        let shared = explorer.snapshot_now();
        let applied_prefix = explorer.applied();
        checkpoints = explorer.checkpoints();
        journal_amplitudes = explorer.journal_amplitudes();
        let mut applied_tails = 0usize;
        for tail in &tails {
            let mut state = sim.backends().create(backend, tail.num_qubits())?;
            state.reset();
            state.load(&shared)?;
            // Only the variant's own tail is applied, planned on its own —
            // the combined plan's operation boundary is NOT a valid cut,
            // because fusion can merge prefix gates with variant gates.
            tail.run(state.as_mut())?;
            applied_tails += tail.ops().len();
            let mut stored = Vec::new();
            state.for_each_nonzero(&mut |index, amp| stored.push((index, amp)));
            stored.sort_by_key(|&(index, _)| index);
            finals.push(stored);
        }
        memo_samples.push(start.elapsed().as_nanos() as usize);
        applied_memoized = applied_prefix + applied_tails;
        if repetition == 0 {
            memo_final = finals;
        }
    }
    memo_samples.sort_unstable();

    let applied_naive: usize = plans.iter().map(|p| p.ops().len()).sum();
    let mut deviation = 0.0f64;
    for (want, got) in naive_final.iter().zip(&memo_final) {
        deviation = deviation.max(compare(want, got));
    }
    Ok(BranchReport {
        prefix_ops,
        variants: variants.len(),
        applied_naive,
        applied_memoized,
        nanos_naive: naive_samples[1].max(1),
        nanos_memoized: memo_samples[1].max(1),
        checkpoints,
        journal_amplitudes,
        deviation,
    })
}

/// Worst amplitude difference between two sorted stored-support lists.
fn compare<S: Scalar>(a: &[(u64, S)], b: &[(u64, S)]) -> f64 {
    let mut worst = 0.0f64;
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() || j < b.len() {
        match (a.get(i), b.get(j)) {
            (Some(&(ai, av)), Some(&(bi, bv))) if ai == bi => {
                worst = worst.max(magnitude(av, bv));
                i += 1;
                j += 1;
            }
            (Some(&(ai, av)), Some(&(bi, _))) if ai < bi => {
                worst = worst.max(av.abs_sqr().sqrt());
                i += 1;
            }
            (Some(_), Some(&(_, bv))) => {
                worst = worst.max(bv.abs_sqr().sqrt());
                j += 1;
            }
            (Some(&(_, av)), None) => {
                worst = worst.max(av.abs_sqr().sqrt());
                i += 1;
            }
            (None, Some(&(_, bv))) => {
                worst = worst.max(bv.abs_sqr().sqrt());
                j += 1;
            }
            (None, None) => break,
        }
    }
    worst
}

fn magnitude<S: Scalar>(a: S, b: S) -> f64 {
    a.coeffs()
        .iter()
        .zip(b.coeffs())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}
