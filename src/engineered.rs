//! **A register that holds only engineered entanglement**, and what the
//! magic costs it.
//!
//! [`crate::surface`] measured that a Clifford surface is an affine
//! coset carrying a degree-2 phase over `ℤ/4` — `O(m²)` numbers where
//! the surface has `2^m` indices — and that three `T` gates behind a
//! bond are enough to end that. Engineered entanglement is exactly the
//! entanglement that form can hold, and this module asks the obvious
//! next question: when the form breaks, by how much?
//!
//! ## The automaton
//!
//! A `T` gate is diagonal, so it is a combination of two **Clifford**
//! gates:
//!
//! ```text
//!   T = diag(1, ω) = a·I + b·Z,     ω = e^{iπ/4}
//!   a = (1 + ω)/2                   b = (1 − ω)/2
//! ```
//!
//! A circuit with `t` of them is therefore a weighted sum of `2^t`
//! Clifford circuits, each of which the engineered register holds
//! exactly and for free. Nothing is approximated and nothing is
//! truncated: the branch set reproduces the surface to rounding, which
//! [`EngineeredSurface::deviation`] measures rather than assumes.
//!
//! So the register scales *dynamically*, and what drives it is the
//! magic rather than the width. The question is whether it scales at
//! the naive rate.
//!
//! ## What the register actually costs
//!
//! `2^t` branches is the count before anything looks at them. The
//! number the register pays is the **dimension of their span** — how
//! many linearly independent surfaces the branch set actually contains
//! — because a dependent branch is one the others already write.
//! [`EngineeredSurface::span_rank`] measures it exactly.
//!
//! Three numbers then bound each other, and their order is the whole
//! result:
//!
//! ```text
//!   span_rank  ≤  2^t          the branches, before distillation
//!   span_rank  ≤  2^m          the surface's own dimension
//! ```
//!
//! Below `2^t` the distillation is real. At `2^m` the engineered
//! register has stopped being cheaper than writing the surface out, and
//! the bond where that happens is where this representation ends.

use crate::circuit::{Circuit, Op};
use crate::error::{Error, Result};
use crate::math::svd_thin;
use crate::scalar::{Scalar, C64};
use crate::surface::{self, PhasePoly};
use crate::sweep;

/// `T` gates behind a bond past which the branch set is refused.
///
/// The set has `2^t` members and each is a full sweep of the volume, so
/// this is a statement about what may be spent, not about where the
/// representation stops working.
pub const MAX_BRANCH_T: usize = 14;

/// `a = (1 + e^{iπ/4})/2` — the identity arm of a `T`.
pub fn branch_identity() -> C64 {
    let w = C64::new(
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    );
    (C64::new(1.0, 0.0) + w) * C64::new(0.5, 0.0)
}

/// `b = (1 − e^{iπ/4})/2` — the `Z` arm of a `T`.
pub fn branch_z() -> C64 {
    let w = C64::new(
        std::f64::consts::FRAC_1_SQRT_2,
        std::f64::consts::FRAC_1_SQRT_2,
    );
    (C64::new(1.0, 0.0) - w) * C64::new(0.5, 0.0)
}

/// A bond's surface as a weighted sum of Clifford surfaces.
#[derive(Debug, Clone)]
pub struct EngineeredSurface {
    /// The bond, between this qubit and the one above it.
    pub after_qubit: usize,
    /// Legs on the bond; the surface has `2^legs` indices.
    pub legs: usize,
    /// `T` gates in the volume behind the bond.
    pub t_gates: usize,
    /// `2^t` — branches before anything looks at them.
    pub branches: usize,
    /// Distinct affine supports among the branches. Two branches on the
    /// same coset are candidates to combine; two on different cosets
    /// are not.
    pub distinct_supports: usize,
    /// Branches that really were single stabilizer terms. Anything less
    /// than all of them would mean the decomposition had leaked.
    pub stabilizer_branches: usize,
    /// Dimension of the branches' span — what the register pays.
    pub span_rank: usize,
    /// Largest deviation of the reassembled sum from the true surface.
    pub deviation: f64,
}

impl EngineeredSurface {
    /// `2^legs` — what writing the surface out costs instead.
    pub fn dense(&self) -> u128 {
        1u128 << self.legs.min(127)
    }

    /// Whether the span is genuinely smaller than the branch count, so
    /// distillation did something.
    pub fn distilled(&self) -> bool {
        self.span_rank < self.branches
    }

    /// Whether the register is still cheaper than the dense surface,
    /// counting each term as the `O(m²)` an affine phase polynomial
    /// takes.
    pub fn cheaper_than_dense(&self) -> bool {
        (self.span_rank as u128) * (self.legs as u128 + 1).pow(2) < self.dense()
    }
}

/// The volume behind `after_qubit` with its `T` gates replaced by the
/// arm `mask` selects: bit `i` clear takes `I`, set takes `Z`.
fn branch_circuit(circuit: &Circuit<C64>, after_qubit: usize, mask: u64) -> Result<Circuit<C64>> {
    let mut out = Circuit::<C64>::new(circuit.num_qubits());
    let mut seen = 0usize;
    for op in circuit.ops() {
        let Op::Named {
            name,
            params,
            qubits,
        } = op
        else {
            return Err(Error::InvalidState(
                "engineered: only named registry gates branch".into(),
            ));
        };
        let magic = (name == "t" || name == "tdg") && qubits[0] <= after_qubit;
        if magic {
            if (mask >> seen) & 1 == 1 {
                out.gate("z", vec![], qubits.clone());
            }
            seen += 1;
        } else {
            out.gate(name.clone(), params.clone(), qubits.clone());
        }
    }
    Ok(out)
}

/// `T` gates in the volume behind a bond.
pub fn magic_behind(circuit: &Circuit<C64>, after_qubit: usize) -> usize {
    circuit
        .ops()
        .iter()
        .filter(|op| match op {
            Op::Named { name, qubits, .. } => {
                (name == "t" || name == "tdg") && qubits[0] <= after_qubit
            }
            _ => false,
        })
        .count()
}

/// Decompose one bond's surface into Clifford branches and measure what
/// the engineered register pays for it.
pub fn engineered_surface(
    circuit: &Circuit<C64>,
    after_qubit: usize,
    bits: u64,
) -> Result<EngineeredSurface> {
    let t = magic_behind(circuit, after_qubit);
    if t > MAX_BRANCH_T {
        return Err(Error::InvalidState(format!(
            "engineered: {t} T gates behind bond {after_qubit} would take 2^{t} branch \
             sweeps, past the {MAX_BRANCH_T} this will spend"
        )));
    }
    let truth = sweep::surfaces(circuit, bits)?
        .into_iter()
        .find(|s| s.after_qubit == after_qubit)
        .ok_or_else(|| Error::InvalidState(format!("engineered: no bond {after_qubit}")))?;
    let m = truth.legs.len();
    let width = 1usize << m;

    let (a, b) = (branch_identity(), branch_z());
    let branches = 1usize << t;
    let mut rows: Vec<C64> = vec![C64::zero(); branches * width];
    let mut summed = vec![C64::zero(); width];
    let mut supports: std::collections::HashSet<(u64, Vec<u64>)> = std::collections::HashSet::new();
    let mut stabilizer_branches = 0usize;

    for mask in 0..branches {
        crate::guard::checkpoint()?;
        let bc = branch_circuit(circuit, after_qubit, mask as u64)?;
        let s = sweep::surfaces(&bc, bits)?
            .into_iter()
            .find(|s| s.after_qubit == after_qubit)
            .ok_or_else(|| Error::InvalidState("engineered: branch lost the bond".into()))?;
        // Weight is one arm per T gate, by the bits of `mask`.
        let mut w = C64::new(1.0, 0.0);
        for i in 0..t {
            w *= if (mask >> i) & 1 == 1 { b } else { a };
        }
        for (x, z) in s.amps.iter().enumerate() {
            let v = *z * w;
            rows[mask * width + x] = v;
            summed[x] += v;
        }
        // Every branch is a Clifford circuit, so each must be a single
        // stabilizer term; anything else means the split leaked.
        if let Some(pp) = surface::phase_poly(&s.amps, m)? {
            stabilizer_branches += 1;
            supports.insert((pp.offset, pp.basis.clone()));
        }
    }

    let svd = svd_thin(branches, width, &rows, 1e-12, 1e-300)?;
    let deviation = truth
        .amps
        .iter()
        .zip(&summed)
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0, f64::max);

    Ok(EngineeredSurface {
        after_qubit,
        legs: m,
        t_gates: t,
        branches,
        distinct_supports: supports.len(),
        stabilizer_branches,
        span_rank: svd.rank,
        deviation,
    })
}

/// One branch's surface as the phase polynomial the register holds it
/// in, or `None` if that branch was not a single stabilizer term.
pub fn branch_term(
    circuit: &Circuit<C64>,
    after_qubit: usize,
    bits: u64,
    mask: u64,
) -> Result<Option<PhasePoly>> {
    let bc = branch_circuit(circuit, after_qubit, mask)?;
    let s = sweep::surfaces(&bc, bits)?
        .into_iter()
        .find(|s| s.after_qubit == after_qubit)
        .ok_or_else(|| Error::InvalidState("engineered: branch lost the bond".into()))?;
    let m = s.legs.len();
    surface::phase_poly(&s.amps, m)
}
