//! **Cut simulation**: the state as a sum over the branches of the gates
//! that cross one cut — cost set by the *interface*, not by the circuit.
//!
//! Every representation in this crate compresses along some axis of the
//! state: support, clusters, bonds, hierarchy, T-count. This one does
//! not compress the state at all. It splits the *circuit*.
//!
//! Pick a cut. A two-qubit gate lying wholly on one side is a local
//! gate there and costs nothing across the cut. A gate that **crosses**
//! it decomposes into a sum of products,
//!
//! ```text
//!   CZ = Σ_{b ∈ {0,1}}  |b⟩⟨b|_A ⊗ Z^b_B
//! ```
//!
//! so branching on `b` for each of the `k` crossing gates writes the
//! whole state as a sum of `2^k` product states:
//!
//! ```text
//!   |ψ⟩ = Σ_{p ∈ 𝔽₂^k} |ψ_A^p⟩ ⊗ |ψ_B^p⟩,
//!   ⟨x|ψ⟩ = Σ_p ⟨x_A|ψ_A^p⟩ · ⟨x_B|ψ_B^p⟩
//! ```
//!
//! and each half is an independent simulation on its own qubits.
//!
//! ## What the cost is, and what it is not
//!
//! ```text
//!   2^k paths  ×  two halves of 2^{n/2} amplitudes
//! ```
//!
//! `k` is the number of **crossing gates**, which for a brickwork on a
//! line is `depth/2` — a bond is active every other layer — regardless
//! of how many two-qubit gates the circuit has in total.
//!
//! The part worth saying out loud: **single-qubit gates are free**.
//! Every one of them lies inside a half by construction, so a `T` costs
//! exactly what an `S` costs, which is nothing. The T-count does not
//! appear in the cost at all. That is the whole point of putting this
//! next to a stabilizer-rank method, whose cost is `2^{αt}` and nothing
//! else: the two price completely disjoint features of the same circuit,
//! and on a circuit that is wide in one and narrow in the other the gap
//! is not small.
//!
//! This is the hybrid Schrödinger–Feynman decomposition, which is
//! standard for two-dimensional supremacy circuits and is the obvious
//! thing to point at a *one-dimensional* one, where the interface is as
//! thin as it can be. [`crate::cut`] predicts `k` from the graph with no
//! simulation; this pays it.
//!
//! ## Scope, stated plainly
//!
//! Crossing gates must be `CZ`. That is not a hidden restriction — a
//! general two-qubit gate decomposes across the cut into at most four
//! product terms rather than two, which changes `2^k` to `4^k` and is
//! usually not worth it; a `CZ` is rank 2 and is what makes this cheap.
//! Anything else is refused rather than approximated.


use crate::backend::Backend;
use crate::circuit::{Circuit, Op};
use crate::error::{Error, Result};
use crate::scalar::C64;

/// The crossing structure of a circuit at one cut.
#[derive(Debug, Clone)]
pub struct CutPlan {
    /// Qubits `0..cut` are the low half.
    pub cut: usize,
    /// Register width.
    pub qubits: usize,
    /// Operation indices of the gates that cross the cut, in order.
    pub crossings: Vec<usize>,
    /// Two-qubit gates lying wholly inside a half — free across the cut.
    pub interior_two_qubit: usize,
    /// Single-qubit gates — free, whatever they are.
    pub single_qubit: usize,
}

impl CutPlan {
    /// `2^k` — the branches the sum runs over.
    pub fn paths(&self) -> u128 {
        1u128 << self.crossings.len().min(127)
    }

    /// The exponent that actually governs the cost: crossings plus the
    /// wider half's width.
    pub fn cost_exponent(&self) -> usize {
        self.crossings.len() + self.cut.max(self.qubits - self.cut)
    }
}

/// Count what crosses `cut` and what does not.
pub fn plan(circuit: &Circuit<C64>, cut: usize) -> Result<CutPlan> {
    let n = circuit.num_qubits();
    if cut == 0 || cut >= n {
        return Err(Error::InvalidState(format!(
            "cutsim: cut {cut} does not split a {n}-qubit register"
        )));
    }
    let mut crossings = Vec::new();
    let mut interior_two_qubit = 0;
    let mut single_qubit = 0;
    for (i, op) in circuit.ops().iter().enumerate() {
        let qs = op.qubits();
        match qs.len() {
            0 => {}
            1 => single_qubit += 1,
            2 => {
                let (a, b) = (qs[0], qs[1]);
                if (a < cut) != (b < cut) {
                    crossings.push(i);
                } else {
                    interior_two_qubit += 1;
                }
            }
            _ => {
                return Err(Error::InvalidState(
                    "cutsim: gates on three or more qubits do not cut this way".into(),
                ))
            }
        }
    }
    Ok(CutPlan {
        cut,
        qubits: n,
        crossings,
        interior_two_qubit,
        single_qubit,
    })
}

/// The cut with the fewest crossing gates, and its plan.
pub fn best_plan(circuit: &Circuit<C64>) -> Result<CutPlan> {
    let n = circuit.num_qubits();
    let mut best: Option<CutPlan> = None;
    for cut in 1..n {
        let p = plan(circuit, cut)?;
        if best
            .as_ref()
            .map_or(true, |b| p.cost_exponent() < b.cost_exponent())
        {
            best = Some(p);
        }
    }
    best.ok_or_else(|| Error::InvalidState("cutsim: no cut available".into()))
}

/// The two halves' interior circuits, split at each crossing gate.
///
/// `k` crossings give `k + 1` segments per side. Binding once and
/// replaying per branch is what keeps the `2^k` loop paying only for the
/// amplitude work.
struct Segments {
    low: Vec<crate::circuit::BoundCircuit<C64>>,
    high: Vec<crate::circuit::BoundCircuit<C64>>,
    /// Per crossing: the qubit on the low side, and on the high side
    /// (already re-indexed onto their halves).
    pivots: Vec<(usize, usize)>,
}

fn segments(circuit: &Circuit<C64>, p: &CutPlan) -> Result<Segments> {
    let reg = crate::registry::GateRegistry::<C64>::standard();
    let k = p.crossings.len();
    let mut low: Vec<Circuit<C64>> = (0..=k).map(|_| Circuit::new(p.cut)).collect();
    let mut high: Vec<Circuit<C64>> = (0..=k)
        .map(|_| Circuit::new(p.qubits - p.cut))
        .collect();
    let mut pivots = Vec::with_capacity(k);
    let mut seg = 0usize;
    for (i, op) in circuit.ops().iter().enumerate() {
        let Op::Named {
            name,
            params,
            qubits,
        } = op
        else {
            return Err(Error::InvalidState(
                "cutsim: only named registry gates cut".into(),
            ));
        };
        if p.crossings.get(seg) == Some(&i) {
            if name != "cz" {
                return Err(Error::InvalidState(format!(
                    "cutsim: crossing gate `{name}` is not a CZ; a general two-qubit \
                     gate cuts into four product terms rather than two, and is refused \
                     rather than silently repriced"
                )));
            }
            // CZ is symmetric, so which endpoint is "low" is decided by
            // the cut and not by the gate's argument order.
            let (lo_q, hi_q) = if qubits[0] < p.cut {
                (qubits[0], qubits[1])
            } else {
                (qubits[1], qubits[0])
            };
            pivots.push((lo_q, hi_q - p.cut));
            seg += 1;
            continue;
        }
        let all_low = qubits.iter().all(|&q| q < p.cut);
        let all_high = qubits.iter().all(|&q| q >= p.cut);
        if all_low {
            low[seg].gate(name.clone(), params.clone(), qubits.clone());
        } else if all_high {
            let remapped: Vec<usize> = qubits.iter().map(|&q| q - p.cut).collect();
            high[seg].gate(name.clone(), params.clone(), remapped);
        }
    }
    Ok(Segments {
        low: low
            .iter()
            .map(|c| c.bind(&reg))
            .collect::<Result<Vec<_>>>()?,
        high: high
            .iter()
            .map(|c| c.bind(&reg))
            .collect::<Result<Vec<_>>>()?,
        pivots,
    })
}

/// Amplitudes `⟨x|C|0…0⟩` at every requested `bits`, by summing the
/// cut's branches.
///
/// Exact, and the exactness is the point: this is not a truncation with
/// an error bar, it is the same number the dense vector would give,
/// reached by paying `2^k` for the *interface* instead of `2^n` for the
/// register.
///
/// **The batch is the unit of work.** Each branch produces the two half
/// vectors in full, so every requested amplitude reads them for free:
/// a thousand amplitudes cost what one costs. Sampling wants many
/// amplitudes, which is exactly the shape this has.
///
/// Costs `2^k` branch pairs at `k = plan.crossings.len()`, each two
/// dense sweeps on `cut` and `n − cut` qubits, in `2^{max half}`
/// memory rather than `2^n`. The branches are independent, so the whole
/// loop is embarrassingly parallel. Nothing in any of that depends on
/// the circuit's single-qubit content — the `T` gates are free.
pub fn amplitudes_at(circuit: &Circuit<C64>, p: &CutPlan, bits: &[u64]) -> Result<Vec<C64>> {
    let segs = segments(circuit, p)?;
    let k = p.crossings.len();
    let split: Vec<(u64, u64)> = bits
        .iter()
        .map(|&x| (x & ((1u64 << p.cut) - 1), x >> p.cut))
        .collect();
    let mut total = vec![C64::new(0.0, 0.0); bits.len()];
    let minus = [C64::new(1.0, 0.0), C64::new(-1.0, 0.0)];
    let mut a = crate::backend::DenseState::<C64>::new(p.cut)?;
    let mut b = crate::backend::DenseState::<C64>::new(p.qubits - p.cut)?;
    for path in 0..(1u64 << k) {
        crate::guard::checkpoint()?;
        // Low half: interior gates, then a projector at each crossing.
        a.reset();
        for j in 0..=k {
            segs.low[j].run(&mut a)?;
            if j < k {
                a.project(segs.pivots[j].0, (path >> j) & 1 == 1, 1.0);
            }
        }
        // Most branches annihilate the low half outright; when they do,
        // the high half is never built.
        if split.iter().all(|&(x_lo, _)| a.amplitude(x_lo).norm_sqr() < 1e-300) {
            continue;
        }
        b.reset();
        for j in 0..=k {
            segs.high[j].run(&mut b)?;
            if j < k && (path >> j) & 1 == 1 {
                b.apply_diagonal(&minus, &[segs.pivots[j].1])?;
            }
        }
        for (t, &(x_lo, x_hi)) in total.iter_mut().zip(&split) {
            *t += a.amplitude(x_lo) * b.amplitude(x_hi);
        }
    }
    Ok(total)
}

/// The full amplitude vector, restricted to a chosen subset of branches.
///
/// The `2^k` sum is only worth paying in full if its terms carry
/// comparable weight. They need not: each branch is
/// `|ψ_A^p⟩ ⊗ |ψ_B^p⟩` with its own norm, and the norms are a
/// *structural* property of the cut — computable per branch, before any
/// amplitude is used. If the weight concentrates, the sum truncates, and
/// a task that only needs fidelity `F` only needs the branches carrying
/// `F` of the weight.
///
/// Returns `(branch weights, truncated amplitude vector)` with the
/// weights in branch order. `keep` is the number of heaviest branches
/// retained; pass `usize::MAX` for the exact sum.
///
/// Exponential in the width by construction — this is the measuring
/// instrument for the truncation question, not a way to run it at scale.
pub fn branch_weights_and_vector(
    circuit: &Circuit<C64>,
    p: &CutPlan,
    keep: usize,
) -> Result<(Vec<f64>, Vec<C64>)> {
    let segs = segments(circuit, p)?;
    let k = p.crossings.len();
    let n = p.qubits;
    let hi_w = n - p.cut;
    let minus = [C64::new(1.0, 0.0), C64::new(-1.0, 0.0)];

    // First pass: every branch's weight, ‖ψ_A^p‖·‖ψ_B^p‖.
    let mut halves: Vec<(Vec<C64>, Vec<C64>)> = Vec::new();
    let mut weights = Vec::with_capacity(1 << k);
    for path in 0..(1u64 << k) {
        crate::guard::checkpoint()?;
        let mut a = crate::backend::DenseState::<C64>::new(p.cut)?;
        for j in 0..=k {
            segs.low[j].run(&mut a)?;
            if j < k {
                a.project(segs.pivots[j].0, (path >> j) & 1 == 1, 1.0);
            }
        }
        let mut b = crate::backend::DenseState::<C64>::new(hi_w)?;
        for j in 0..=k {
            segs.high[j].run(&mut b)?;
            if j < k && (path >> j) & 1 == 1 {
                b.apply_diagonal(&minus, &[segs.pivots[j].1])?;
            }
        }
        let va: Vec<C64> = (0..(1u64 << p.cut)).map(|i| a.amplitude(i)).collect();
        let vb: Vec<C64> = (0..(1u64 << hi_w)).map(|i| b.amplitude(i)).collect();
        let na: f64 = va.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let nb: f64 = vb.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        weights.push(na * nb);
        halves.push((va, vb));
    }

    // Second pass: sum only the heaviest `keep` branches.
    let mut order: Vec<usize> = (0..halves.len()).collect();
    order.sort_by(|&i, &j| weights[j].partial_cmp(&weights[i]).unwrap());
    order.truncate(keep.min(halves.len()));
    let mut out = vec![C64::new(0.0, 0.0); 1usize << n];
    for &idx in &order {
        let (va, vb) = &halves[idx];
        for (hi, zb) in vb.iter().enumerate() {
            if zb.norm_sqr() == 0.0 {
                continue;
            }
            let base = hi << p.cut;
            for (lo, za) in va.iter().enumerate() {
                out[base | lo] += *za * *zb;
            }
        }
    }
    Ok((weights, out))
}

/// Every branch as a full-register amplitude vector, with its weight.
///
/// The cut writes `|ψ⟩ = Σ_p |ψ_A^p⟩ ⊗ |ψ_B^p⟩`, which is precisely the
/// object a [`BranchedRegister`](crate::clock::BranchedRegister) holds:
/// a weighted superposition of slices whose cost is the *sum* over
/// slices rather than the product. Handing the branches to it makes
/// `selector_schmidt_rank` answer a question this module cannot ask
/// itself — how many of the `2^k` branches are linearly *independent*.
/// Flat weights do not imply independent slices, and the difference is
/// exactly how much the branch sum could be compressed.
///
/// Exponential in the width by construction: this materializes each
/// branch on all `n` qubits. It is the instrument for the independence
/// question at small width, not a way to run the decomposition.
pub fn branch_states(circuit: &Circuit<C64>, p: &CutPlan) -> Result<Vec<(f64, Vec<C64>)>> {
    let segs = segments(circuit, p)?;
    let k = p.crossings.len();
    let n = p.qubits;
    let hi_w = n - p.cut;
    let minus = [C64::new(1.0, 0.0), C64::new(-1.0, 0.0)];
    let mut out = Vec::new();
    for path in 0..(1u64 << k) {
        crate::guard::checkpoint()?;
        let mut a = crate::backend::DenseState::<C64>::new(p.cut)?;
        for j in 0..=k {
            segs.low[j].run(&mut a)?;
            if j < k {
                a.project(segs.pivots[j].0, (path >> j) & 1 == 1, 1.0);
            }
        }
        let mut b = crate::backend::DenseState::<C64>::new(hi_w)?;
        for j in 0..=k {
            segs.high[j].run(&mut b)?;
            if j < k && (path >> j) & 1 == 1 {
                b.apply_diagonal(&minus, &[segs.pivots[j].1])?;
            }
        }
        let mut full = vec![C64::new(0.0, 0.0); 1usize << n];
        let mut norm = 0.0;
        for hi in 0..(1u64 << hi_w) {
            let zb = b.amplitude(hi);
            if zb.norm_sqr() == 0.0 {
                continue;
            }
            let base = (hi as usize) << p.cut;
            for lo in 0..(1u64 << p.cut) {
                let v = a.amplitude(lo) * zb;
                norm += v.norm_sqr();
                full[base | lo as usize] = v;
            }
        }
        out.push((norm.sqrt(), full));
    }
    Ok(out)
}

/// One amplitude — [`amplitudes_at`] with a single target.
pub fn amplitude(circuit: &Circuit<C64>, p: &CutPlan, bits: u64) -> Result<C64> {
    Ok(amplitudes_at(circuit, p, &[bits])?[0])
}
