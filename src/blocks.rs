//! One representation per engineered block: the circuit solved as `k`
//! independent problems, each in whichever representation suits it.
//!
//! [`crate::coupling`] turns a circuit that looks fully coupled into `k`
//! symplectically orthogonal pieces on disjoint qubits. That is a
//! statement about the *operator*, and [`crate::coupling::propagate_engineered`]
//! cashes it in one way: a Pauli sum per block, whose cost is the sum
//! rather than the product of the blocks' term counts.
//!
//! But once the blocks are genuinely independent there is no reason for
//! them to share a representation at all. In the engineered frame the
//! whole computation is
//!
//! ```text
//! ⟨0…0|U†PU|0…0⟩ = ∏_b ⟨φ_b| U_b† P_b U_b |φ_b⟩
//! ```
//!
//! — a product of `k` complete quantum simulations, each on its own `w`
//! qubits, each with its own input `|φ_b⟩`, its own circuit `U_b`, and
//! its own observable `P_b`. Nothing couples them, so each may be solved
//! by whatever is cheapest for *it*: a Pauli walk where the block is
//! nearly Clifford, an MPS where it is wide but lightly entangled, a
//! dense vector where it is narrow enough that `2^w` is nothing.
//! [`BlockSolver`] is that choice and [`propagate_blocked`] makes it per
//! block.
//!
//! This is where the representational win is largest, because bond
//! dimension is not additive across a frame change. A global MPS on the
//! scrambled circuit pays the entanglement of the *lab* frame, which the
//! scrambler made maximal; `k` MPS on `w` sites each pay only what their
//! own block carries. The bound goes from `2^{n/2}` to `k · 2^{w/2}`.
//!
//! ## What has to be true, and is checked
//!
//! The product form needs `|φ⟩ = V†|0…0⟩` to factor across the blocks —
//! the same condition [`crate::coupling::EngineeredReport::separable_input`]
//! reports. Here it is not optional: without it there is no per-block
//! problem to pose, so [`propagate_blocked`] refuses rather than
//! returning a product that is not one.
//!
//! Given separability, each `|φ_b⟩` is a stabilizer state on the block's
//! own qubits, and [`preparation`] synthesizes a Clifford circuit making
//! it from `|0…0⟩` — which is what lets an ordinary [`Backend`] be
//! pointed at a block at all.

use crate::backend::{
    conjugate_by_step, pauli_expectation, Backend, CliffordStep, DenseState, MpsConfig, MpsState,
    PauliString, SparseState,
};
use crate::coupling::{coupling_of, decoupling_frame, Stabilizer};
use crate::error::{Error, Result};
use crate::gates::Pauli;
use crate::heisenberg::{
    axis_operator_phase, step_through, PauliKey, PauliSum, Rotation, MAX_QUBITS,
};
use crate::scalar::C64;

/// How one block is solved. The blocks are independent, so this is a
/// per-block choice and not a global one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockSolver {
    /// Heisenberg Pauli walk inside the block, contracted against the
    /// block's stabilizer input. Exponential in the block's *branching*
    /// gates, free on Clifford ones.
    Pauli,
    /// Forward MPS on the block's own sites. Exponential in the block's
    /// *internal* entanglement, and blind to how entangled the lab frame
    /// was.
    Mps {
        /// Bond ceiling for this block alone.
        max_bond: usize,
    },
    /// Forward dense vector on the block's own sites: `2^w`, which for a
    /// narrow block is smaller than either of the above.
    Dense,
    /// Forward sparse vector on the block's own sites.
    Sparse,
}

impl BlockSolver {
    /// Short label for reports.
    pub fn label(&self) -> String {
        match self {
            BlockSolver::Pauli => "pauli".into(),
            BlockSolver::Mps { max_bond } => format!("mps χ≤{max_bond}"),
            BlockSolver::Dense => "dense".into(),
            BlockSolver::Sparse => "sparse".into(),
        }
    }
}

/// What one block cost and what it returned.
#[derive(Debug, Clone)]
pub struct BlockOutcome {
    /// Qubits this block occupies in the engineered frame.
    pub mask: u64,
    /// Width of the block.
    pub qubits: usize,
    /// Rotations that landed in it.
    pub rotations: usize,
    /// How it was solved.
    pub solver: BlockSolver,
    /// `⟨φ_b|U_b†P_bU_b|φ_b⟩` for the Hermitian `P_b`.
    pub value: f64,
    /// Peak bytes this block's representation held.
    pub bytes: usize,
    /// The representation's own parameter — terms, bond, or support.
    pub detail: String,
}

/// What [`propagate_blocked`] measured.
#[derive(Debug, Clone)]
pub struct BlockedReport {
    /// `⟨0…0|U†PU|0…0⟩`, assembled as the product over blocks.
    pub value: f64,
    /// One entry per block, in frame order.
    pub blocks: Vec<BlockOutcome>,
    /// Elementary Cliffords in the frame.
    pub frame_steps: usize,
    /// Widest block, in qubits — the width any single representation
    /// actually had to hold.
    pub widest_block: usize,
    /// Rotations deleted as inert before any of this ran.
    pub inert_rotations: usize,
}

impl BlockedReport {
    /// Bytes summed over blocks: what the whole computation held at
    /// once, since nothing couples them.
    pub fn total_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.bytes).sum()
    }

    /// Largest single block's bytes — the peak of any one representation.
    pub fn peak_block_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.bytes).max().unwrap_or(0)
    }
}

// ── preparing a block's share of the input ───────────────────────────

/// A Clifford circuit making a stabilizer state from `|0…0⟩`.
#[derive(Debug, Clone, Default)]
pub struct Preparation {
    /// Qubits to flip before the Clifford runs.
    pub flips: Vec<usize>,
    /// Clifford steps, **in application order**.
    pub steps: Vec<CliffordStep>,
}

/// Synthesize a circuit preparing the stabilizer state whose group is
/// generated by `gens`, on `qubits` sites.
///
/// The routine is the same symplectic steering
/// [`crate::coupling::decoupling_frame`] uses, restricted to a
/// mutually-commuting family and targeting Z: canonicalize each
/// generator to `±Z` on a fresh qubit, reducing the rest by it as it
/// goes — legal here, and only here, because a stabilizer group is one
/// group rather than several that must stay apart.
///
/// If `W† g_i W = s_i Z_{q_i}` then `W†|φ⟩` is the computational basis
/// state with a 1 wherever `s_i = −1`, so `|φ⟩ = W|x⟩` and the circuit
/// is `X^x` followed by `W`. Since `W = s₁s₂⋯s_m` as an operator, the
/// steps are applied to the ket in **reverse**, which is what
/// [`Preparation::steps`] stores.
pub fn preparation(gens: &[PauliString], qubits: usize) -> Result<Preparation> {
    let mut rem: Vec<PauliString> = gens.to_vec();
    let mut steps: Vec<CliffordStep> = Vec::new();
    let mut allocated = 0u64;
    let mut flips: Vec<usize> = Vec::new();

    macro_rules! emit {
        ($step:expr) => {{
            let s = $step;
            steps.push(s);
            for g in rem.iter_mut() {
                *g = conjugate_by_step(*g, s);
            }
        }};
    }

    for i in 0..rem.len() {
        // Reduce against the already-canonicalized `±Z` pivots. Every
        // survivor commutes with them, so it carries no X there and a
        // single multiplication clears the Z.
        let mut probe = allocated & (rem[i].x | rem[i].z);
        while probe != 0 {
            let q = probe.trailing_zeros() as usize;
            probe &= probe - 1;
            let pivot = rem[..i]
                .iter()
                .find(|g| g.z == 1u64 << q && g.x == 0)
                .copied()
                .ok_or_else(|| {
                    Error::InvalidState("preparation: no pivot for an allocated qubit".into())
                })?;
            rem[i] = rem[i]
                .times(pivot)
                .ok_or_else(|| Error::InvalidState("preparation: generators anticommute".into()))?;
        }
        if rem[i].x | rem[i].z == 0 {
            return Err(Error::InvalidState(
                "preparation: generators are not independent".into(),
            ));
        }

        // Steer to all-Z: H turns X into Z, S then H turns Y into Z.
        loop {
            let g = rem[i];
            let Some(q) = (0..qubits).find(|q| g.x >> q & 1 == 1) else {
                break;
            };
            if g.z >> q & 1 == 1 {
                emit!(CliffordStep::S(q));
            }
            emit!(CliffordStep::H(q));
        }

        // Fold onto a fresh pivot: `Z_pZ_q` under CX(p→q) is `Z_q`.
        let support = rem[i].z & !allocated;
        let q = support.trailing_zeros() as usize;
        if support == 0 {
            return Err(Error::InvalidState(
                "preparation: generator reached only allocated qubits".into(),
            ));
        }
        loop {
            let rest = rem[i].z & !(1u64 << q);
            if rest == 0 {
                break;
            }
            emit!(CliffordStep::Cx(rest.trailing_zeros() as usize, q));
        }

        if rem[i].negative {
            flips.push(q);
        }
        allocated |= 1u64 << q;
    }

    // `W = s₁⋯s_m` acting on a ket runs its factors right to left.
    steps.reverse();
    Ok(Preparation { flips, steps })
}

impl Preparation {
    /// Run the preparation on a backend holding `|0…0⟩`.
    pub fn apply(&self, state: &mut dyn Backend<C64>) -> Result<()> {
        for &q in &self.flips {
            let (m, s) = Rotation::rx(q, std::f64::consts::PI).gate()?;
            state.apply(&m, &s)?;
        }
        for step in &self.steps {
            let (m, s) = step_gate(*step)?;
            state.apply(&m, &s)?;
        }
        Ok(())
    }
}

/// The unitary of one elementary Clifford, for the qubits it acts on.
fn step_gate(step: CliffordStep) -> Result<(crate::math::GateMatrix<C64>, Vec<usize>)> {
    let r = std::f64::consts::FRAC_1_SQRT_2;
    match step {
        CliffordStep::H(q) => {
            let mut m = crate::math::GateMatrix::<C64>::zeros(2)?;
            m.set(0, 0, C64::new(r, 0.0));
            m.set(0, 1, C64::new(r, 0.0));
            m.set(1, 0, C64::new(r, 0.0));
            m.set(1, 1, C64::new(-r, 0.0));
            Ok((m, vec![q]))
        }
        CliffordStep::S(q) => {
            let mut m = crate::math::GateMatrix::<C64>::zeros(2)?;
            m.set(0, 0, C64::new(1.0, 0.0));
            m.set(1, 1, C64::new(0.0, 1.0));
            Ok((m, vec![q]))
        }
        CliffordStep::Cx(c, t) => {
            // sub-index bit 0 is the control, per the crate's convention
            let mut m = crate::math::GateMatrix::<C64>::zeros(4)?;
            for (row, col) in [(0, 0), (1, 3), (2, 2), (3, 1)] {
                m.set(row, col, C64::new(1.0, 0.0));
            }
            Ok((m, vec![c, t]))
        }
    }
}

// ── the blocked computation ──────────────────────────────────────────

/// Compact a mask's qubits to `0..w`, returning the ascending list.
fn compact(mask: u64) -> Vec<usize> {
    (0..MAX_QUBITS).filter(|q| mask >> q & 1 == 1).collect()
}

/// Remap a key onto compacted indices; bits outside `sites` are dropped.
fn remap(key: PauliKey, sites: &[usize]) -> PauliKey {
    let mut out = (0u64, 0u64);
    for (i, &q) in sites.iter().enumerate() {
        out.0 |= ((key.0 >> q) & 1) << i;
        out.1 |= ((key.1 >> q) & 1) << i;
    }
    out
}

/// The Hermitian Pauli of a key, as backend ops.
fn ops_of(key: PauliKey) -> Vec<(usize, Pauli)> {
    (0..MAX_QUBITS)
        .filter_map(|q| match (key.0 >> q & 1, key.1 >> q & 1) {
            (1, 0) => Some((q, Pauli::X)),
            (0, 1) => Some((q, Pauli::Z)),
            (1, 1) => Some((q, Pauli::Y)),
            _ => None,
        })
        .collect()
}

/// Solve one block forward on a concrete backend.
fn forward_block(
    mut state: Box<dyn Backend<C64>>,
    prep: &Preparation,
    rots: &[Rotation],
    observable: PauliKey,
) -> Result<f64> {
    prep.apply(state.as_mut())?;
    for r in rots {
        let (m, s) = r.gate()?;
        state.apply(&m, &s)?;
    }
    let value = pauli_expectation(state.as_ref(), &ops_of(observable))?.re;
    Ok(value)
}

/// Propagate an observable with **one representation per engineered
/// block**, chosen by `choose` from the block's own shape.
///
/// The circuit is decoupled first ([`crate::coupling::decoupling_frame`]),
/// then split: each block gets its own qubits, its own rotations, its
/// own share of the observable, and its own input state, and is solved
/// on its own. The answer is the product.
///
/// Errors when the frame leaves `V†|0…0⟩` entangled across the blocks —
/// there is then no per-block problem, and saying so is the only honest
/// option.
pub fn propagate_blocked_with(
    observable: PauliKey,
    rotations: &[Rotation],
    qubits: usize,
    choose: impl Fn(usize, u64, usize) -> BlockSolver,
) -> Result<BlockedReport> {
    let coupling = coupling_of(observable, rotations);
    let live = coupling.live_rotations(rotations);
    let frame = decoupling_frame(observable, &live, qubits)?;
    let framed = frame.rewrite(&live);
    let (obs_key, obs_sign) = frame.conjugate(observable);
    let stab = frame.input_stabilizer();
    let blocks = frame.blocks().to_vec();

    if !stab.separable(&blocks, qubits) {
        return Err(Error::InvalidState(
            "propagate_blocked: the frame left the input entangled across the blocks, \
             so there is no per-block problem to pose"
                .into(),
        ));
    }

    let mut outcomes = Vec::with_capacity(blocks.len());
    // The Hermitian observable is the product of its per-block
    // restrictions, because `|x ∧ z|` is additive over disjoint support.
    let mut product = 1.0f64;

    for (bi, &mask) in blocks.iter().enumerate() {
        let sites = compact(mask);
        let w = sites.len();
        let block_rots: Vec<Rotation> = framed
            .iter()
            .filter(|r| (r.axis.0 | r.axis.1) & mask != 0)
            .map(|r| Rotation {
                theta: r.theta,
                axis: remap(r.axis, &sites),
            })
            .collect();
        let block_key_global = (obs_key.0 & mask, obs_key.1 & mask);
        let block_key = remap(block_key_global, &sites);
        let solver = choose(bi, mask, block_rots.len());

        let (value, bytes, detail) = match solver {
            BlockSolver::Pauli => {
                // The block's own Pauli walk, contracted against the
                // block's own stabilizer input.
                let mut sum = PauliSum::from_key(block_key);
                let mut peak = sum.len();
                for r in block_rots.iter().rev() {
                    let (next, _, _, _) = step_through(&sum, r, None);
                    sum = next;
                    peak = peak.max(sum.len());
                }
                let local = Stabilizer::new(
                    stab.block_generators(mask)
                        .into_iter()
                        .map(|g| PauliString {
                            x: remap((g.x, g.z), &sites).0,
                            z: remap((g.x, g.z), &sites).1,
                            negative: g.negative,
                        })
                        .collect(),
                );
                let mut acc = C64::new(0.0, 0.0);
                for (key, c) in sum.terms() {
                    let s = local.expectation(key);
                    if s != 0.0 {
                        acc += c * C64::new(s, 0.0) / axis_operator_phase(key);
                    }
                }
                let herm = (acc * axis_operator_phase(block_key)).re;
                (herm, peak * 32, format!("terms {peak}"))
            }
            _ => {
                let local: Vec<PauliString> = stab
                    .block_generators(mask)
                    .into_iter()
                    .map(|g| {
                        let (x, z) = remap((g.x, g.z), &sites);
                        PauliString {
                            x,
                            z,
                            negative: g.negative,
                        }
                    })
                    .collect();
                let prep = preparation(&local, w)?;
                match solver {
                    BlockSolver::Mps { max_bond } => {
                        let mut s = MpsState::<C64>::with_config(
                            w,
                            MpsConfig {
                                max_bond,
                                trunc_tol: 1e-14,
                            },
                        )?;
                        prep.apply(&mut s)?;
                        for r in &block_rots {
                            let (m, q) = r.gate()?;
                            s.apply(&m, &q)?;
                        }
                        let v = pauli_expectation(&s as &dyn Backend<C64>, &ops_of(block_key))?.re;
                        let bond = s.max_bond_dimension();
                        (v, s.memory_bytes(), format!("bond {bond}"))
                    }
                    BlockSolver::Dense => {
                        let s = DenseState::<C64>::new(w)?;
                        let bytes = s.memory_bytes();
                        let v = forward_block(Box::new(s), &prep, &block_rots, block_key)?;
                        (v, bytes, format!("amps {}", 1usize << w))
                    }
                    BlockSolver::Sparse => {
                        let mut s = SparseState::<C64>::new(w)?;
                        prep.apply(&mut s)?;
                        for r in &block_rots {
                            let (m, q) = r.gate()?;
                            s.apply(&m, &q)?;
                        }
                        let support = s.nonzero_count();
                        let v = pauli_expectation(&s as &dyn Backend<C64>, &ops_of(block_key))?.re;
                        (v, s.memory_bytes(), format!("support {support}"))
                    }
                    BlockSolver::Pauli => unreachable!("handled above"),
                }
            }
        };

        product *= value;
        outcomes.push(BlockOutcome {
            mask,
            qubits: w,
            rotations: block_rots.len(),
            solver,
            value,
            bytes,
            detail,
        });
    }

    // Back out of the Hermitian normalization, exactly as
    // `propagate_engineered` does.
    let phase = axis_operator_phase(observable);
    let value = (C64::new(product * obs_sign, 0.0) / phase).re;

    Ok(BlockedReport {
        value,
        blocks: outcomes,
        frame_steps: frame.len(),
        widest_block: blocks
            .iter()
            .map(|m| m.count_ones() as usize)
            .max()
            .unwrap_or(0),
        inert_rotations: coupling.inert_rotations(),
    })
}

/// [`propagate_blocked_with`] using one solver for every block.
pub fn propagate_blocked(
    observable: PauliKey,
    rotations: &[Rotation],
    qubits: usize,
    solver: BlockSolver,
) -> Result<BlockedReport> {
    propagate_blocked_with(observable, rotations, qubits, |_, _, _| solver)
}
