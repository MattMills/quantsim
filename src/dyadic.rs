//! A dyadic cone: the same two-directional structure, recursively, at
//! every scale — and the two directions pruning each other.
//!
//! [`crate::horizon`] samples the sheet along a *line* of cuts. This
//! bisects it instead. The gate axis is halved, then each half halved,
//! down `depth` levels, so level `k` holds `2^k` segments of length
//! `L / 2^k` and the structure is self-similar by construction. Every
//! node carries both directions:
//!
//! * **past** — the qubits the input can have reached by the node's left
//!   edge: the forward light cone of the gates before it;
//! * **future** — the qubits the observable can still feel at the node's
//!   right edge: the backward cone, graded by the influence field;
//! * **diamond** — their intersection, restricted to the gates the node
//!   actually owns. Outside it, either the past cannot get there or the
//!   future cannot see it, so nothing that happens there can matter.
//!
//! Children partition their parent's gates exactly, so the diamond is a
//! **measure over a dyadic tree** rather than a set over a rectangle.
//! That has a consequence worth stating plainly, because it decides what
//! the recursion is good for: every *additive* quantity — live area,
//! total front motion — is identical at every level, by construction.
//! Those are checks the recursion performs on itself, not measurements.
//! What carries new information as the scale sharpens is the *extremal*
//! quantities. [`DyadicCone::peak_velocity_by_level`] climbs with depth
//! because a coarse window averages the front's bursts together with the
//! stretches where it is saturated and cannot move; halving separates
//! them. Where it stops climbing is the circuit's own time scale, read
//! off without being told what that scale is.
//!
//! ## Where the two directions interact
//!
//! At a cut both objects exist at once, and that changes what may be
//! discarded. A backward walk on its own can only bound a dropped term
//! by `|c_P|`, because `|⟨ψ|P|ψ⟩| ≤ 1` is all that is known without the
//! state — there is no sound refinement of that bound, and reaching for
//! one is a mistake. But at a cut the state *is* known, so
//!
//! ```text
//! |Δ⟨P⟩| = |Σ_{dropped} c_P ⟨ψ|P|ψ⟩| ≤ Σ_{dropped} |c_P| · |⟨ψ|P|ψ⟩|
//! ```
//!
//! is computable, still rigorous, and never worse. [`Node::two_sided_l1`]
//! is that sum against [`Node::l1`], and [`DyadicCone::tightening`] is
//! the ratio — the measured amount by which a one-directional bound
//! overstates what a two-directional one can certify.
//!
//! The tree also self-checks harder than a line does. Every node
//! contracts its own midpoint, so all `2^{depth+1} − 1` of them return
//! the same number; [`DyadicCone::value_spread`] is how far they differ,
//! and it is machine precision for an exact walk.

use crate::backend::{pauli_expectation, Backend, DenseState, MpsConfig, MpsState, SparseState};
use crate::error::{Error, Result};
use crate::gates::Pauli;
use crate::heisenberg::{
    axis_operator_phase, step_through, PauliKey, PauliSum, Rotation, MAX_QUBITS,
};
use crate::horizon::Forward;
use crate::scalar::C64;

/// Deepest bisection allowed: level `d` needs `2^d + 1` stored backward
/// sums, so the tree's memory is exponential in the depth even when each
/// sum is small.
pub const MAX_DEPTH: usize = 12;

/// How a dyadic cone is built.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Levels of bisection. Level 0 is the whole circuit; level `depth`
    /// has `2^depth` leaves.
    pub depth: usize,
    /// Drop backward terms below this `|c|`. Zero is exact.
    pub threshold: f64,
    /// How the forward state is held.
    pub forward: Forward,
    /// Skip the two-sided sum on any node whose backward sum is larger
    /// than this, since it costs one state expectation per term. Zero
    /// means never skip.
    pub two_sided_max_terms: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            depth: 4,
            threshold: 0.0,
            forward: Forward::Dense,
            two_sided_max_terms: 4096,
        }
    }
}

/// One node of the bisection: a gate range, both directions' reach at
/// its edges, and the answer contracted at its own midpoint.
#[derive(Debug, Clone)]
pub struct Node {
    /// Bisection level; 0 is the whole circuit.
    pub level: usize,
    /// Index within the level, left to right.
    pub index: usize,
    /// Gate range `[a, b)` this node owns.
    pub span: (usize, usize),
    /// The cut this node contracts at — its own midpoint.
    pub mid: usize,
    /// `⟨ψ_mid|O_mid|ψ_mid⟩`. Every node returns the same number.
    pub value: f64,
    /// Qubits the input can have reached by the left edge.
    pub past: u64,
    /// Qubits the observable can still feel at the right edge.
    pub future: u64,
    /// Every qubit the two directions overlap on somewhere in this span
    /// — the union of the per-gate intersections.
    pub diamond: u64,
    /// Qubits the future's front advanced by while crossing this node.
    pub advance: usize,
    /// Cells the node owns: `span length × qubits`.
    pub cells: usize,
    /// Cells inside the diamond.
    pub live_cells: usize,
    /// Terms in the backward sum at the midpoint.
    pub terms: usize,
    /// `Σ|c|` at the midpoint — what a one-directional walk can certify.
    pub l1: f64,
    /// `Σ|c|·|⟨ψ|P|ψ⟩|` at the midpoint — what both directions can.
    /// `None` when the sum was too large to evaluate term by term.
    pub two_sided_l1: Option<f64>,
}

impl Node {
    /// Gates in the span.
    pub fn width(&self) -> usize {
        self.span.1 - self.span.0
    }

    /// Fraction of the node's cells that are inside the diamond.
    pub fn density(&self) -> f64 {
        self.live_cells as f64 / self.cells.max(1) as f64
    }

    /// How much the one-directional bound overstates the two-directional
    /// one here. `None` when the two-sided sum was skipped.
    pub fn tightening(&self) -> Option<f64> {
        self.two_sided_l1
            .map(|t| if t > 0.0 { self.l1 / t } else { f64::INFINITY })
    }
}

/// The tree.
#[derive(Debug, Clone)]
pub struct DyadicCone {
    /// Register width.
    pub qubits: usize,
    /// Gates in the circuit.
    pub gates: usize,
    /// Levels built.
    pub depth: usize,
    /// Every node, level by level, left to right within a level.
    pub nodes: Vec<Node>,
}

impl DyadicCone {
    /// The nodes at one level.
    pub fn level(&self, level: usize) -> Vec<&Node> {
        self.nodes.iter().filter(|n| n.level == level).collect()
    }

    /// How far the tree's nodes disagree about the answer. Machine
    /// precision for an exact walk — and every node is an independent
    /// contraction, so this is a far stronger check than a line of cuts.
    pub fn value_spread(&self) -> f64 {
        let (lo, hi) = self
            .nodes
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), n| {
                (a.min(n.value), b.max(n.value))
            });
        if lo.is_finite() {
            hi - lo
        } else {
            0.0
        }
    }

    /// Live area at each level, as a share of the level's cells.
    ///
    /// This is **conserved**: children partition their parent's gates,
    /// so the total is identical at every level. It is therefore a check
    /// on the recursion rather than a measurement of the physics — if it
    /// drifts, the bisection has lost or double-counted cells.
    pub fn density_by_level(&self) -> Vec<f64> {
        (0..=self.depth)
            .map(|l| {
                let nodes = self.level(l);
                let live: usize = nodes.iter().map(|n| n.live_cells).sum();
                let cells: usize = nodes.iter().map(|n| n.cells).sum();
                live as f64 / cells.max(1) as f64
            })
            .collect()
    }

    /// Mean front velocity at each level, in qubits per gate.
    ///
    /// Also **conserved**, and for the same reason: the advances across a
    /// level's nodes telescope to the front's total motion over the whole
    /// circuit. Any additive quantity does. That is worth stating plainly,
    /// because it is what a multi-scale decomposition is *for*: the
    /// additive quantities are invariants the recursion can check itself
    /// against, and only the extremal ones carry new information as the
    /// scale sharpens — see [`DyadicCone::peak_velocity_by_level`].
    pub fn velocity_by_level(&self) -> Vec<f64> {
        (0..=self.depth)
            .map(|l| {
                let nodes = self.level(l);
                let adv: usize = nodes.iter().map(|n| n.advance).sum();
                let span: usize = nodes.iter().map(|n| n.width()).sum();
                adv as f64 / span.max(1) as f64
            })
            .collect()
    }

    /// **Peak** local front velocity at each level.
    ///
    /// This is the number that resolves. At a coarse scale the front's
    /// bursts are averaged with the stretches where it is saturated and
    /// cannot move at all; halving the window separates them, so the peak
    /// climbs with depth until the window is short enough to sit inside a
    /// single burst. Where it stops climbing is the circuit's own time
    /// scale, read off without being told what that scale is.
    pub fn peak_velocity_by_level(&self) -> Vec<f64> {
        (0..=self.depth)
            .map(|l| {
                self.level(l)
                    .iter()
                    .map(|n| n.advance as f64 / n.width().max(1) as f64)
                    .fold(0.0f64, f64::max)
            })
            .collect()
    }

    /// Mean diamond width at each level, in qubits. Resolves: a shorter
    /// window overlaps fewer qubits, so this falls with depth even
    /// though the total live area does not move.
    pub fn mean_diamond_by_level(&self) -> Vec<f64> {
        (0..=self.depth)
            .map(|l| {
                let nodes = self.level(l);
                nodes
                    .iter()
                    .map(|n| n.diamond.count_ones() as f64)
                    .sum::<f64>()
                    / nodes.len().max(1) as f64
            })
            .collect()
    }

    /// How much sharper the finest scale sees the front than the
    /// coarsest: peak velocity at the deepest level over peak velocity
    /// at level 0. `1.0` would mean the bisection resolved nothing.
    pub fn resolution_gain(&self) -> f64 {
        let peaks = self.peak_velocity_by_level();
        match (peaks.last(), peaks.first()) {
            (Some(&last), Some(&first)) if first > 0.0 => last / first,
            _ => 1.0,
        }
    }

    /// Largest measured ratio of the one-directional bound to the
    /// two-directional one, over the nodes that evaluated it.
    pub fn tightening(&self) -> f64 {
        self.nodes
            .iter()
            .filter_map(|n| n.tightening())
            .filter(|t| t.is_finite())
            .fold(1.0f64, f64::max)
    }

    /// Mean tightening over the nodes that evaluated it.
    pub fn mean_tightening(&self) -> f64 {
        let vals: Vec<f64> = self
            .nodes
            .iter()
            .filter_map(|n| n.tightening())
            .filter(|t| t.is_finite())
            .collect();
        if vals.is_empty() {
            1.0
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    }

    /// Cells the whole tree's leaves consider live, against the level-0
    /// rectangle — how much the recursion localizes the work.
    pub fn leaf_localization(&self) -> f64 {
        let leaves = self.level(self.depth);
        let live: usize = leaves.iter().map(|n| n.live_cells).sum();
        let root = self.nodes.first().map(|n| n.cells).unwrap_or(1);
        live as f64 / root.max(1) as f64
    }
}

/// Qubits the sum acts on non-trivially.
fn support(sum: &PauliSum) -> u64 {
    sum.terms().fold(0u64, |m, (k, _)| m | k.0 | k.1)
}

/// `⟨ψ| X^x Z^z |ψ⟩` for the raw key.
fn pauli_on_state(state: &dyn Backend<C64>, key: PauliKey) -> Result<C64> {
    let ops: Vec<(usize, Pauli)> = (0..MAX_QUBITS)
        .filter_map(|q| match (key.0 >> q & 1, key.1 >> q & 1) {
            (1, 0) => Some((q, Pauli::X)),
            (0, 1) => Some((q, Pauli::Z)),
            (1, 1) => Some((q, Pauli::Y)),
            _ => None,
        })
        .collect();
    Ok(pauli_expectation(state, &ops)? / axis_operator_phase(key))
}

/// The dyadic cut positions, ascending and de-duplicated.
fn cut_positions(gates: usize, depth: usize) -> Vec<usize> {
    let n = 1usize << depth;
    let mut cuts: Vec<usize> = (0..=n).map(|i| i * gates / n).collect();
    cuts.dedup();
    cuts
}

/// Build the tree.
///
/// One backward walk stores the sum at every dyadic position; one
/// forward run meets each of them. Every node then contracts at its own
/// midpoint, so the tree's agreement is `2^{depth+1} − 1` independent
/// evaluations of the same number rather than one.
pub fn dyadic_cone(
    observable: PauliKey,
    rotations: &[Rotation],
    qubits: usize,
    cfg: &Config,
) -> Result<DyadicCone> {
    if qubits == 0 || qubits > MAX_QUBITS {
        return Err(Error::InvalidState(format!(
            "dyadic_cone: {qubits} qubits outside 1..={MAX_QUBITS}"
        )));
    }
    if cfg.depth > MAX_DEPTH {
        return Err(Error::InvalidState(format!(
            "dyadic_cone: depth {} exceeds {MAX_DEPTH}",
            cfg.depth
        )));
    }
    let gates = rotations.len();
    if gates < (1usize << cfg.depth) {
        return Err(Error::InvalidState(format!(
            "dyadic_cone: {gates} gates cannot be bisected {} times",
            cfg.depth
        )));
    }
    let cuts = cut_positions(gates, cfg.depth);
    let index_of = |g: usize| cuts.iter().position(|&c| c == g);

    // ── backward: store the sum at every dyadic position ─────────────
    let mut sum = PauliSum::from_key(observable);
    let mut backward: Vec<PauliSum> = vec![PauliSum::zero(); cuts.len()];
    // The future's reach at EVERY position, not just the dyadic ones —
    // the diamond is a per-gate measure, so that children partition
    // their parent exactly rather than approximately.
    let mut future = vec![0u64; gates + 1];
    future[gates] = support(&sum);
    let mut next = cuts.len() - 1;
    while cuts[next] == gates {
        backward[next] = sum.clone();
        if next == 0 {
            break;
        }
        next -= 1;
    }
    for i in (0..gates).rev() {
        let (mut stepped, _, _, _) = step_through(&sum, &rotations[i], None);
        if cfg.threshold > 0.0 {
            stepped.truncate(cfg.threshold);
        }
        sum = stepped;
        future[i] = support(&sum);
        while cuts[next] == i {
            backward[next] = sum.clone();
            if next == 0 {
                break;
            }
            next -= 1;
        }
    }

    // ── forward: meet each stored sum where it sits ──────────────────
    let mut state: Box<dyn Backend<C64>> = match cfg.forward {
        Forward::Dense => Box::new(DenseState::<C64>::new(qubits)?),
        Forward::Sparse => Box::new(SparseState::<C64>::new(qubits)?),
        Forward::Mps { max_bond } => Box::new(MpsState::<C64>::with_config(
            qubits,
            MpsConfig {
                max_bond,
                trunc_tol: 1e-14,
            },
        )?),
    };
    let mut value = vec![0.0f64; cuts.len()];
    let mut l1 = vec![0.0f64; cuts.len()];
    let mut terms = vec![0usize; cuts.len()];
    let mut two_sided = vec![None; cuts.len()];

    let mut settle = |col: usize, state: &dyn Backend<C64>| -> Result<()> {
        let s = &backward[col];
        let mut acc = C64::new(0.0, 0.0);
        let mut weighted = 0.0f64;
        let evaluate_two_sided =
            cfg.two_sided_max_terms == 0 || s.len() <= cfg.two_sided_max_terms;
        for (key, c) in s.terms() {
            let e = pauli_on_state(state, key)?;
            acc += c * e;
            if evaluate_two_sided {
                weighted += c.norm() * e.norm();
            }
        }
        value[col] = acc.re;
        l1[col] = s.l1();
        terms[col] = s.len();
        two_sided[col] = evaluate_two_sided.then_some(weighted);
        Ok(())
    };

    let mut col = 0usize;
    while col < cuts.len() && cuts[col] == 0 {
        settle(col, state.as_ref())?;
        col += 1;
    }
    for (i, r) in rotations.iter().enumerate() {
        let (m, s) = r.gate()?;
        state.apply(&m, &s)?;
        while col < cuts.len() && cuts[col] == i + 1 {
            settle(col, state.as_ref())?;
            col += 1;
        }
    }

    // ── the tree ─────────────────────────────────────────────────────
    // `past[g]` is the forward light cone of everything before position
    // `g`; `future[g]` is what the observable can still feel there. The
    // diamond is their intersection AT EACH GATE, so a node's live area
    // is a sum over its own gates and children partition their parent.
    let mut past = vec![0u64; gates + 1];
    for i in 0..gates {
        past[i + 1] = past[i] | rotations[i].axis.0 | rotations[i].axis.1;
    }

    let mut nodes = Vec::new();
    for level in 0..=cfg.depth {
        let count = 1usize << level;
        for index in 0..count {
            let a = index * gates / count;
            let b = (index + 1) * gates / count;
            let mid = (a + b) / 2;
            // Contract at the nearest stored position to this node's
            // midpoint; at the tree's own depth that is the midpoint.
            let col = index_of(mid).unwrap_or_else(|| {
                cuts.iter()
                    .enumerate()
                    .min_by_key(|(_, &c)| c.abs_diff(mid))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            });
            let width = b - a;
            let mut diamond = 0u64;
            let mut live_cells = 0usize;
            for g in a..b {
                let d = past[g] & future[g];
                diamond |= d;
                live_cells += d.count_ones() as usize;
            }
            // The front's advance across this node, in qubits: the
            // future's reach grows as the walk runs backward, so this is
            // how far it moved while crossing the node's own gates.
            let advance = future[a].count_ones().saturating_sub(future[b].count_ones()) as usize;
            nodes.push(Node {
                level,
                index,
                span: (a, b),
                mid,
                value: value[col],
                past: past[a],
                future: future[b],
                diamond,
                advance,
                cells: width * qubits,
                live_cells,
                terms: terms[col],
                l1: l1[col],
                two_sided_l1: two_sided[col],
            });
        }
    }

    Ok(DyadicCone {
        qubits,
        gates,
        depth: cfg.depth,
        nodes,
    })
}

/// The two-directional error bound at one cut, and the one-directional
/// bound beside it.
///
/// Returns `(Σ|c|, Σ|c|·|⟨ψ|P|ψ⟩|)`. The second is what a truncation at
/// this cut can actually be certified to cost; the first is what a walk
/// that has not met its state has to assume.
pub fn two_sided_bound(sum: &PauliSum, state: &dyn Backend<C64>) -> Result<(f64, f64)> {
    let mut one = 0.0f64;
    let mut two = 0.0f64;
    for (key, c) in sum.terms() {
        let w = c.norm();
        one += w;
        two += w * pauli_on_state(state, key)?.norm();
    }
    Ok((one, two))
}

/// Discard the terms a **two-directional** criterion says cannot matter,
/// and report what was dropped and what it provably cost.
///
/// The key is `|c_P| · |⟨ψ|P|ψ⟩|`, not `|c_P|`. Both are rigorous; only
/// the second is available once the state is in hand, and it is never
/// looser. Terms are dropped smallest-first until the accumulated
/// certified error would exceed `budget`, so the returned error is a
/// bound on the whole discard and not on any single term.
///
/// Returns `(kept sum, terms dropped, certified error)`.
pub fn prune_two_sided(
    sum: &PauliSum,
    state: &dyn Backend<C64>,
    budget: f64,
) -> Result<(PauliSum, usize, f64)> {
    let mut scored: Vec<(PauliKey, C64, f64)> = Vec::with_capacity(sum.len());
    for (key, c) in sum.terms() {
        scored.push((key, c, c.norm() * pauli_on_state(state, key)?.norm()));
    }
    scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut spent = 0.0f64;
    let mut dropped = 0usize;
    let mut kept = PauliSum::zero();
    for (key, c, score) in &scored {
        if spent + score <= budget {
            spent += score;
            dropped += 1;
        } else {
            kept.add(*key, *c);
        }
    }
    Ok((kept, dropped, spent))
}

