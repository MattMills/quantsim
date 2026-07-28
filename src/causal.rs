//! Causal geometry of a computation over a qubit register.
//!
//! The elements of a register are causally related through the
//! operations that couple them; that relation — not the raw qubit count
//! — is what bounds how much of the system's full manifold a
//! computation actually inhabits, and therefore how much of it must be
//! computed. This module makes the relation operational, in three
//! layers:
//!
//! * **Backward light cones** ([`backward_cone`]): the set of earlier
//!   operations that can influence a chosen observation surface, walked
//!   directly off the circuit's coupling structure.
//! * **Causal diamonds** ([`causal_diamond`]): the circuit restricted
//!   to that cone. Operations outside the diamond provably cannot move
//!   any marginal on the observed qubits — the pruned circuit is
//!   smaller but *observationally identical*, which is the precise
//!   sense in which causal structure makes the manifold more
//!   computable.
//! * **Dual-time resolution** ([`dual_time_amplitude`]): two opposed
//!   evolution directions that resolve toward each other. The
//!   preparation boundary evolves *forward* through the front of the
//!   circuit while the observation boundary evolves *backward* (the
//!   adjoint) through the back, and the two meet at a chosen cut:
//!   `⟨t|U|0⟩ = ⟨U₂†t | U₁0⟩`. Each direction only ever grows the
//!   support its own boundary demands, so a computation whose one-way
//!   support is `2^D` resolves at `2^{D/2}` a side when the cut is
//!   balanced — the meeting surface ([`DualTimeResolution`]) carries
//!   the measured cost of resolution.
//!
//! The same two clocks exist on hardware: scheduled on a
//! [`DeviceState`](crate::backend::DeviceState), the front half and the
//! inverted back half each present half the serial wall, so the wall
//! the schedule presents is `max` of the two half-schedules rather than
//! their sum. The register-side counterpart of all of this lives on
//! [`Topology`](crate::backend::Topology): `distances_from`,
//! `diameter`, `ball_sizes` and `pair_availability` measure the causal
//! metric a coupling fabric imposes, and the `hierarchical` /
//! `hypercube` constructors build registers whose fabric is *organized*
//! by causal scale, so that the metric — and with it routing cost and
//! availability — is logarithmic instead of linear.

use crate::backend::{Backend, SparseState};
use crate::circuit::Circuit;
use crate::error::{Error, Result};
use crate::registry::GateRegistry;
use crate::scalar::{Scalar, C64};

/// Per-operation membership in the backward light cone of `observed`:
/// entry `i` is `true` iff operation `i` can influence a measurement on
/// any of the observed qubits. Walked in reverse: an operation is in
/// the cone iff it touches a qubit the cone has already claimed, and
/// membership claims all of its qubits.
pub fn backward_cone<S: Scalar>(circuit: &Circuit<S>, observed: &[usize]) -> Vec<bool> {
    let n = circuit.num_qubits();
    let mut active = vec![false; n];
    for &q in observed {
        if q < n {
            active[q] = true;
        }
    }
    let mut keep = vec![false; circuit.len()];
    for (i, op) in circuit.ops().iter().enumerate().rev() {
        if op.qubits().iter().any(|&q| active[q]) {
            keep[i] = true;
            for &q in op.qubits() {
                active[q] = true;
            }
        }
    }
    keep
}

/// What [`causal_diamond`] measured while pruning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiamondReport {
    /// Operations inside the diamond (kept).
    pub kept: usize,
    /// Operations outside the diamond (dropped).
    pub dropped: usize,
    /// Qubits the backward cone reached — the diamond's spatial width
    /// at the preparation boundary.
    pub cone_qubits: usize,
}

/// Restrict `circuit` to the causal diamond of the observed qubits:
/// the sub-circuit of operations inside [`backward_cone`], in original
/// order. Every marginal (joint outcome distribution) on `observed` is
/// identical between the full and the pruned circuit — operations
/// outside the backward cone act, at each point of the reversed order,
/// on a disjoint tensor factor, so they cannot move it.
pub fn causal_diamond<S: Scalar>(
    circuit: &Circuit<S>,
    observed: &[usize],
) -> (Circuit<S>, DiamondReport) {
    let keep = backward_cone(circuit, observed);
    let n = circuit.num_qubits();
    let mut cone = vec![false; n];
    for &q in observed {
        if q < n {
            cone[q] = true;
        }
    }
    let mut pruned = Circuit::new(n);
    let mut kept = 0usize;
    for (i, op) in circuit.ops().iter().enumerate() {
        if keep[i] {
            kept += 1;
            for &q in op.qubits() {
                cone[q] = true;
            }
            push_clone(&mut pruned, op);
        }
    }
    let report = DiamondReport {
        kept,
        dropped: circuit.len() - kept,
        cone_qubits: cone.iter().filter(|&&c| c).count(),
    };
    (pruned, report)
}

fn push_clone<S: Scalar>(target: &mut Circuit<S>, op: &crate::circuit::Op<S>) {
    use crate::circuit::Op;
    match op {
        Op::Named {
            name,
            params,
            qubits,
        } => {
            target.gate(name.clone(), params.clone(), qubits.clone());
        }
        Op::Raw {
            label,
            matrix,
            qubits,
        } => {
            target.raw(label.clone(), matrix.clone(), qubits.clone());
        }
        Op::Diagonal {
            label,
            entries,
            qubits,
        } => {
            target.diagonal(label.clone(), entries.clone(), qubits.clone());
        }
    }
}

/// The meeting surface of a dual-time resolution: what each direction
/// carried to the cut, and what it cost to resolve them.
#[derive(Debug, Clone)]
pub struct DualTimeResolution {
    /// `⟨target | U | 0…0⟩`, resolved at the cut.
    pub amplitude: C64,
    /// Operations the forward direction evolved through.
    pub forward_gates: usize,
    /// Operations the backward direction evolved through (inverted).
    pub backward_gates: usize,
    /// Nonzero support the forward direction presented at the cut.
    pub forward_support: usize,
    /// Nonzero support the backward direction presented at the cut.
    pub backward_support: usize,
    /// Basis states where the two directions overlap — the terms the
    /// resolution actually summed.
    pub interface_terms: usize,
}

impl DualTimeResolution {
    /// The larger of the two supports — the peak either direction had
    /// to hold, against the one-directional cost of carrying the full
    /// evolution to the observation boundary.
    pub fn peak_support(&self) -> usize {
        self.forward_support.max(self.backward_support)
    }
}

/// Resolve `⟨target|U|0…0⟩` by two opposed time directions meeting at
/// `cut`: the preparation boundary `|0…0⟩` evolves forward through
/// `circuit[..cut]`, the observation boundary `|target⟩` evolves
/// backward through the adjoint of `circuit[cut..]`, and the two
/// resolve as an inner product on their supports' overlap
/// (`⟨t|U₂U₁|0⟩ = ⟨U₂†t|U₁0⟩`).
///
/// Both directions run sparse, so each pays only the support its own
/// boundary generates up to the cut — the reason a balanced cut costs
/// the square root of the one-directional support on depth-branching
/// circuits. The returned [`DualTimeResolution`] reports the measured
/// meeting surface.
pub fn dual_time_amplitude(
    registry: &GateRegistry<C64>,
    circuit: &Circuit<C64>,
    cut: usize,
    target: u64,
) -> Result<DualTimeResolution> {
    let n = circuit.num_qubits();
    if cut > circuit.len() {
        return Err(Error::InvalidState(format!(
            "dual-time cut {cut} past the end of a {}-op circuit",
            circuit.len()
        )));
    }
    if n < 64 && target >> n != 0 {
        return Err(Error::InvalidState(format!(
            "target state {target:#x} does not fit {n} qubits"
        )));
    }
    let (front, back) = circuit.split_at(cut);

    let mut forward = SparseState::<C64>::new(n)?;
    front.bind(registry)?.run(&mut forward)?;

    let mut backward = SparseState::<C64>::new(n)?;
    backward.load(&[(target, C64::new(1.0, 0.0))])?;
    back.bind(registry)?.inverse().run(&mut backward)?;

    // Resolve on the smaller support; the inner product conjugates the
    // backward (left) factor per the crate's left-module convention.
    let mut amplitude = C64::new(0.0, 0.0);
    let mut interface = 0usize;
    let (small, large): (&SparseState<C64>, &SparseState<C64>) =
        if forward.nonzero_count() <= backward.nonzero_count() {
            (&forward, &backward)
        } else {
            (&backward, &forward)
        };
    small.for_each_nonzero(&mut |idx, amp_small| {
        let amp_large = large.amplitude(idx);
        if amp_large.norm_sqr() > 0.0 {
            interface += 1;
            let (fwd, bwd) = if std::ptr::eq(small, &forward) {
                (amp_small, amp_large)
            } else {
                (amp_large, amp_small)
            };
            amplitude += bwd.conj() * fwd;
        }
    });

    Ok(DualTimeResolution {
        amplitude,
        forward_gates: front.len(),
        backward_gates: back.len(),
        forward_support: forward.nonzero_count(),
        backward_support: backward.nonzero_count(),
        interface_terms: interface,
    })
}
