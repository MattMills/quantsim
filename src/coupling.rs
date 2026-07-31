//! Engineered coupling structure: decoupling an operator that the
//! circuit's wiring says is fully coupled.
//!
//! [`crate::heisenberg::FactoredPauliSum`] holds the observable as a
//! tensor product of blocks and merges two blocks whenever a gate's axis
//! **touches both**. That rule is correct but it is not the right rule,
//! and the difference is the whole content of this module.
//!
//! Touching the same qubits is a statement about the computational
//! basis. What actually governs whether conjugation couples two regions
//! is the **symplectic form**: `exp(−iθP/2)` maps `Q ↦ Q` exactly when
//! `[P, Q] = 0`, and branches only when they anticommute. So build the
//! graph whose nodes are the rotation axes, with an edge wherever two of
//! them *anticommute*. Two facts follow, and both are exact:
//!
//! 1. **Distinct components are symplectically orthogonal.** If `a` and
//!    `b` lie in different components then `ω(a, b) = 0` by construction,
//!    and because `ω` is bilinear over `𝔽₂` the *spans* are orthogonal
//!    too — every element of one commutes with every element of the
//!    other. The evolution factorizes across components no matter how
//!    their qubit supports overlap.
//! 2. **Components the observable cannot feel are inert.** Split
//!    `U = U₁U₂` by component; `[U₁, U₂] = 0`, and if every axis of
//!    component 2 commutes with `P` then `U₂†PU₂ = P` — and the same
//!    holds for every term `P·a` the other components can ever produce,
//!    because `ω(Pa, b) = ω(P, b) + ω(a, b) = 0`. So `U†PU = U₁†PU₁`
//!    and those rotations can be **deleted**, not approximated.
//!    [`Coupling::live_rotations`] does the deleting.
//!
//! Note what is *not* a node: the observable's single-site factors.
//! Splitting `P` by qubit is itself a basis-dependent move — a Clifford
//! change of frame scrambles which site goes where — so including it
//! would make the partition depend on how the problem was written down.
//! The axes' partition is a Clifford invariant, and the observable
//! enters only through the one basis-free question that matters: does it
//! anticommute with anything in this component?
//!
//! Point 1 answers "is the qubit partition the natural one". It is not:
//! it is the partition you get by refusing to change basis. Since
//! components are symplectically orthogonal, a Clifford `V` generally
//! exists carrying each onto its **own disjoint set of qubits** —
//! [`decoupling_frame`] constructs one, as an explicit list of H, S and
//! CX steps, by symplectic Gram–Schmidt over `𝔽₂`. In that frame the
//! circuit is a tensor product of independent circuits and
//! [`FactoredPauliSum`] never merges. The lab-frame circuit can be
//! all-to-all; the engineered frame does not care. Measured on a
//! block-structured circuit conjugated by a Clifford scrambler until
//! every axis is wide: the support rule sees **one** block at every
//! size, the frame recovers all `k`, stored terms grow linearly and the
//! flat count the factorization avoids grows geometrically.
//!
//! ## Where the cost went
//!
//! Nothing is free, and this module is built to show where it went
//! rather than to hide it. Conjugating the circuit by `V` moves the
//! input too: `⟨0…0|U†PU|0…0⟩ = ⟨φ|U′†P′U′|φ⟩` with `|φ⟩ = V†|0…0⟩`, a
//! stabilizer state. The block product `⟨φ|A⊗B|φ⟩ = ⟨A⟩⟨B⟩` is valid
//! only when `|φ⟩` is unentangled across the engineered blocks, which is
//! a property of `V` and is **not** automatic. [`Stabilizer::separable`]
//! tests it and [`EngineeredReport::separable_input`] reports it; when
//! it fails, [`propagate_engineered`] says so and falls back to a flat
//! contraction (bounded by [`FLAT_CONTRACTION_CAP`]) rather than
//! returning a wrong number.
//!
//! Measured, on a block-structured circuit in a scrambled frame: a
//! CX-only scrambler fixes `|0…0⟩` and the input stays a product in
//! **every** case; a scrambler drawn from the full Clifford generating
//! set leaves it entangled in roughly half. The operator decoupled
//! either way. That is the honest shape of the result — the *operator*
//! side can always be decoupled, the pair (operator, input) cannot, and
//! which one you get is decided by the circuit and its input, not by the
//! representation.

use crate::backend::{conjugate_by_step, CliffordStep, PauliString};
use crate::error::{Error, Result};
use crate::heisenberg::{
    axis_operator_phase, commutes, FactoredPauliSum, PauliKey, Rotation, MAX_QUBITS,
};
use crate::scalar::C64;

/// Most flat terms [`propagate_engineered`] will expand when the input
/// state does not factor across the engineered blocks and the product
/// shortcut is therefore invalid.
pub const FLAT_CONTRACTION_CAP: u128 = 1 << 22;

// ── the intrinsic partition ──────────────────────────────────────────

/// The anticommutation components of a circuit's rotation axes.
///
/// This is the partition the operator *actually* obeys, as opposed to
/// the partition its qubit labels suggest. Built by union–find over the
/// symplectic form, in `O(m²)` mask operations.
///
/// The nodes are the **rotation axes only**. That is deliberate and it
/// is the second thing the qubit-support rule gets wrong: splitting the
/// observable into single-site factors is itself a statement about the
/// computational basis, and a Clifford change of frame scrambles which
/// sites go where. The axes' partition, by contrast, is a Clifford
/// invariant — conjugation is a bijection on axes and preserves `ω` — so
/// it is a property of the circuit and not of how it was written down.
/// The observable enters only through the one basis-free question that
/// matters: does it anticommute with anything in this component?
#[derive(Debug, Clone)]
pub struct Coupling {
    /// Component id per rotation, indexed as the input slice.
    rot_comp: Vec<usize>,
    /// Canonical component ids, ascending.
    comps: Vec<usize>,
    /// Components holding at least one axis that anticommutes with the
    /// observable.
    live: Vec<usize>,
}

/// Union–find with path halving.
struct Dsu(Vec<usize>);

impl Dsu {
    fn new(n: usize) -> Self {
        Dsu((0..n).collect())
    }
    fn find(&mut self, mut a: usize) -> usize {
        while self.0[a] != a {
            self.0[a] = self.0[self.0[a]];
            a = self.0[a];
        }
        a
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.0[ra] = rb;
        }
    }
}

/// The single-site factors of a Pauli key, one per non-identity qubit.
fn sites(key: PauliKey) -> Vec<(usize, PauliKey)> {
    let (x, z) = key;
    (0..MAX_QUBITS)
        .filter(|q| (x >> q | z >> q) & 1 == 1)
        .map(|q| (q, ((x >> q & 1) << q, (z >> q & 1) << q)))
        .collect()
}

/// Find the anticommutation components of a circuit, and which of them
/// the observable can feel.
pub fn coupling_of(observable: PauliKey, rotations: &[Rotation]) -> Coupling {
    let axes: Vec<PauliKey> = rotations.iter().map(|r| r.axis).collect();
    let mut dsu = Dsu::new(axes.len());
    for i in 0..axes.len() {
        for j in (i + 1)..axes.len() {
            if !commutes(axes[i], axes[j]) {
                dsu.union(i, j);
            }
        }
    }
    let rot_comp: Vec<usize> = (0..axes.len()).map(|i| dsu.find(i)).collect();

    let mut comps = rot_comp.clone();
    comps.sort_unstable();
    comps.dedup();

    // A component is live when some axis in it fails to commute with the
    // observable. If they all commute then so does every element of the
    // span (`ω` is bilinear), and `U_c` fixes not just `P` but every term
    // the other components can ever produce from it.
    let mut live: Vec<usize> = axes
        .iter()
        .zip(&rot_comp)
        .filter(|(a, _)| !commutes(observable, **a))
        .map(|(_, &c)| c)
        .collect();
    live.sort_unstable();
    live.dedup();

    Coupling {
        rot_comp,
        comps,
        live,
    }
}

impl Coupling {
    /// Number of anticommutation components.
    pub fn components(&self) -> usize {
        self.comps.len()
    }

    /// Components the observable can feel — the number of independent
    /// problems the circuit poses.
    pub fn live_components(&self) -> usize {
        self.live.len()
    }

    /// Rotations in a component the observable cannot feel. Deleting
    /// them is exact, not an approximation.
    pub fn inert_rotations(&self) -> usize {
        self.rot_comp
            .iter()
            .filter(|c| !self.live.contains(c))
            .count()
    }

    /// The circuit with the inert rotations removed, order preserved.
    pub fn live_rotations(&self, rotations: &[Rotation]) -> Vec<Rotation> {
        rotations
            .iter()
            .zip(&self.rot_comp)
            .filter(|(_, c)| self.live.contains(c))
            .map(|(r, _)| *r)
            .collect()
    }

    /// Qubit mask of each live component's axes, in the **lab** frame —
    /// where they generally overlap, which is why a frame change is
    /// needed to use them as tensor factors.
    pub fn live_masks(&self, rotations: &[Rotation]) -> Vec<u64> {
        self.live
            .iter()
            .map(|&c| {
                rotations
                    .iter()
                    .zip(&self.rot_comp)
                    .filter(|(_, &cc)| cc == c)
                    .fold(0u64, |m, (r, _)| m | r.axis.0 | r.axis.1)
            })
            .collect()
    }

    /// Every axis, grouped by component with the live components first.
    /// The order is what steers the frame: eliminating a whole component
    /// before starting the next is what keeps its image on a contiguous
    /// block of its own.
    fn grouped_axes(&self, rotations: &[Rotation]) -> Vec<Vec<PauliKey>> {
        let mut order: Vec<usize> = self.live.clone();
        order.extend(self.comps.iter().filter(|c| !self.live.contains(c)));
        order
            .iter()
            .map(|&c| {
                rotations
                    .iter()
                    .zip(&self.rot_comp)
                    .filter(|(_, &cc)| cc == c)
                    .map(|(r, _)| r.axis)
                    .collect()
            })
            .collect()
    }
}

/// Find a Pauli that anticommutes with `g` and commutes with every
/// element of `others`, using only the qubits in `allowed`.
///
/// This is what keeps a *radical* elimination round from leaking one
/// component's span into another's. A round that pairs `g` with a real
/// generator `h` is safe on its own: anything that fails to commute with
/// the pair is, by definition of the graph, already in the same
/// component. A round with no such `h` has to invent one, and inventing
/// it carelessly is what merges blocks — an invented partner that some
/// other component's generator can feel drags that generator onto this
/// component's qubit. Demanding orthogonality to everything else makes
/// the pivot invisible outside the round, so no clearing is needed and
/// nothing is contaminated.
///
/// Solvable exactly when `g` is independent of `others`: the constraints
/// are `⟨ω(·, g), h⟩ = 1` and `⟨ω(·, rᵢ), h⟩ = 0`, an inhomogeneous
/// `𝔽₂` system whose only obstruction is `ω(·, g)` lying in the span of
/// the `ω(·, rᵢ)`.
fn orthogonal_partner(g: PauliKey, others: &[PauliKey], allowed: u64) -> Option<PauliKey> {
    // Column `i` of the solution vector is `h.x[i]`, column `64 + i` is
    // `h.z[i]`; the row for a Pauli `v` is `(v.z ‖ v.x)`.
    let row_of = |v: PauliKey| ((v.1 & allowed) as u128) | (((v.0 & allowed) as u128) << 64);
    let mut rows: Vec<(u128, u8)> = vec![(row_of(g), 1)];
    rows.extend(others.iter().map(|&r| (row_of(r), 0)));

    let mut pivots: Vec<(usize, usize)> = Vec::new(); // (column, row index)
    let mut rank = 0usize;
    for col in 0..128usize {
        let Some(p) = (rank..rows.len()).find(|&i| rows[i].0 >> col & 1 == 1) else {
            continue;
        };
        rows.swap(rank, p);
        let pivot = rows[rank];
        for (i, row) in rows.iter_mut().enumerate() {
            if i != rank && row.0 >> col & 1 == 1 {
                row.0 ^= pivot.0;
                row.1 ^= pivot.1;
            }
        }
        pivots.push((col, rank));
        rank += 1;
    }
    // inconsistent: `0 = 1`
    if rows.iter().any(|&(v, b)| v == 0 && b == 1) {
        return None;
    }
    let mut h = 0u128;
    for &(col, r) in &pivots {
        if rows[r].1 == 1 {
            h |= 1u128 << col;
        }
    }
    let key = ((h & u64::MAX as u128) as u64, (h >> 64) as u64);
    if key.0 | key.1 == 0 {
        None
    } else {
        Some(key)
    }
}

/// Support-overlap components of a set of Pauli masks.
fn overlap_blocks(masks: impl IntoIterator<Item = u64>) -> Vec<u64> {
    let mut blocks: Vec<u64> = Vec::new();
    for m in masks {
        if m == 0 {
            continue;
        }
        let hit: Vec<usize> = (0..blocks.len()).filter(|&i| blocks[i] & m != 0).collect();
        let mut merged = m;
        for &i in hit.iter().rev() {
            merged |= blocks.remove(i);
        }
        blocks.push(merged);
    }
    blocks.sort_unstable();
    blocks
}

/// The number of blocks a support-overlap partition finds — what
/// [`crate::heisenberg::propagate_factored`] discovers, computed here so
/// the two partitions can be compared on the same circuit.
pub fn support_blocks(observable: PauliKey, rotations: &[Rotation]) -> usize {
    let mut masks: Vec<u64> = sites(observable).iter().map(|&(q, _)| 1u64 << q).collect();
    for r in rotations {
        let m = r.axis.0 | r.axis.1;
        let hit: Vec<usize> = (0..masks.len()).filter(|&i| masks[i] & m != 0).collect();
        if hit.is_empty() {
            continue;
        }
        let mut merged = m;
        for &i in hit.iter().rev() {
            merged |= masks.remove(i);
        }
        masks.push(merged);
    }
    masks.len()
}

// ── the decoupling frame ─────────────────────────────────────────────

/// A Clifford held as an explicit list of elementary steps, together
/// with the disjoint qubit blocks it produces.
///
/// Named for what [`decoupling_frame`] builds it for, but the type is
/// general: [`DecouplingFrame::from_steps`] takes any Clifford, which is
/// how a problem gets *posed* in a scrambled frame as well as solved in
/// a decoupled one.
///
/// Conjugation by the frame is `V†(·)V` for `V = s₁s₂⋯s_k`, applied by
/// running [`conjugate_by_step`] left to right. Signs come out of the
/// tableau rules exactly; no phase is dropped.
#[derive(Debug, Clone)]
pub struct DecouplingFrame {
    steps: Vec<CliffordStep>,
    blocks: Vec<u64>,
    qubits: usize,
}

impl DecouplingFrame {
    /// The identity frame on `n` qubits, with no blocks.
    pub fn identity(qubits: usize) -> Self {
        DecouplingFrame::from_steps(Vec::new(), qubits)
    }

    /// An arbitrary Clifford, given as elementary steps. Carries no
    /// block structure — use it to *pose* a problem in a chosen frame,
    /// not to solve one.
    pub fn from_steps(steps: Vec<CliffordStep>, qubits: usize) -> Self {
        DecouplingFrame {
            steps,
            blocks: Vec::new(),
            qubits,
        }
    }

    /// Elementary Clifford count — the cost of entering the frame.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the frame is the identity.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The steps, in application order.
    pub fn steps(&self) -> &[CliffordStep] {
        &self.steps
    }

    /// The engineered blocks: support-overlap components of the framed
    /// axes and the framed observable's sites, so a block is a set of
    /// qubits no framed gate ever reaches out of. Disjoint by
    /// construction. This counts every axis; a walk that starts from the
    /// observable only ever visits the blocks the observable reaches, so
    /// [`EngineeredReport::engineered_blocks`] can be smaller.
    pub fn blocks(&self) -> &[u64] {
        &self.blocks
    }

    /// `V† P V` for a signed Hermitian string.
    pub fn conjugate_string(&self, p: PauliString) -> PauliString {
        self.steps
            .iter()
            .fold(p, |acc, &s| conjugate_by_step(acc, s))
    }

    /// `V† Q V` for `Q = i^{|x∧z|}X^xZ^z`, returned as `(key, sign)` in
    /// the same Hermitian normalization.
    pub fn conjugate(&self, key: PauliKey) -> (PauliKey, f64) {
        let img = self.conjugate_string(PauliString {
            x: key.0,
            z: key.1,
            negative: false,
        });
        (
            (img.x, img.z),
            if img.negative { -1.0 } else { 1.0 },
        )
    }

    /// The circuit rewritten in the frame: each axis conjugated, with a
    /// sign flip absorbed into the angle (`exp(−iθ(−Q)/2) = exp(iθQ/2)`).
    pub fn rewrite(&self, rotations: &[Rotation]) -> Vec<Rotation> {
        rotations
            .iter()
            .map(|r| {
                let (axis, sign) = self.conjugate(r.axis);
                Rotation {
                    theta: r.theta * sign,
                    axis,
                }
            })
            .collect()
    }

    /// The stabilizer group of `|φ⟩ = V†|0…0⟩`, generated by
    /// `V†Z_qV` — because `Z_q|0…0⟩ = |0…0⟩` implies
    /// `(V†Z_qV)(V†|0…0⟩) = V†|0…0⟩`.
    pub fn input_stabilizer(&self) -> Stabilizer {
        Stabilizer::new(
            (0..self.qubits)
                .map(|q| {
                    self.conjugate_string(PauliString {
                        x: 0,
                        z: 1u64 << q,
                        negative: false,
                    })
                })
                .collect(),
        )
    }
}

/// Build a Clifford that compresses the circuit's axis span to canonical
/// form, so the operator factors across as many independent blocks as
/// the circuit really has.
///
/// The algorithm is a symplectic Gram–Schmidt over `𝔽₂`. Take the first
/// remaining generator `g`:
///
/// * if some remaining `h` **anticommutes** with it, `(g, h)` is a
///   hyperbolic pair: steer `g` to `X_q` and `h` to `Z_q` on one fresh
///   qubit `q`;
/// * otherwise `g` lies in the radical of what is left: steer it to
///   `Z_q` alone.
///
/// Either way `q` is then cleared out of every remaining generator by
/// multiplying in `X_q`/`Z_q` — legal because those now lie in the span,
/// so this is a change of basis and nothing is lost. That clearing is
/// what makes the next pivot always fresh: a hyperbolic round leaves
/// every survivor commuting with both `X_q` and `Z_q`, a radical round
/// leaves them commuting with `Z_q` and (because `g` was in the radical)
/// already carrying no `X` there.
///
/// A hyperbolic pair spends one qubit for two dimensions and a radical
/// element one for one, so the budget is `k + r ≤ n` — the bound that an
/// isotropic subspace of a `2n`-dimensional symplectic space has
/// dimension at most `n` — and the construction never runs out.
///
/// **Generators are fed in component order, live components first.**
/// That ordering is the whole of the block engineering: eliminating one
/// component before starting the next keeps its image on a contiguous
/// run of fresh qubits. It is a heuristic, not a theorem — two
/// components can share a radical direction, in which case their images
/// overlap and the blocks merge — so the resulting partition is
/// **measured** from the framed axes ([`DecouplingFrame::blocks`]) rather than
/// assumed, and [`EngineeredReport::engineered_blocks`] reports what was
/// actually achieved.
pub fn decoupling_frame(
    observable: PauliKey,
    rotations: &[Rotation],
    qubits: usize,
) -> Result<DecouplingFrame> {
    if qubits > MAX_QUBITS {
        return Err(Error::InvalidState(format!(
            "decoupling_frame: {qubits} qubits exceeds {MAX_QUBITS}"
        )));
    }
    let width = |k: PauliKey| 64 - (k.0 | k.1).leading_zeros() as usize;
    let used = rotations
        .iter()
        .map(|r| width(r.axis))
        .chain(std::iter::once(width(observable)))
        .max()
        .unwrap_or(0);
    if used > qubits {
        return Err(Error::InvalidState(format!(
            "decoupling_frame: circuit touches qubit {} but was given {qubits}",
            used - 1
        )));
    }
    let coupling = coupling_of(observable, rotations);
    // One flat list, ordered by component: the elimination is global (so
    // a pivot is always fresh) while the ordering is what groups each
    // component's image onto its own qubits.
    let mut gens: Vec<PauliKey> = coupling
        .grouped_axes(rotations)
        .concat()
        .into_iter()
        .filter(|&(x, z)| x | z != 0)
        .collect();

    let mut steps: Vec<CliffordStep> = Vec::new();
    let mut allocated = 0u64;

    // Emitting a step transforms every remaining generator; only supports
    // matter for building `V`, so this runs unsigned.
    macro_rules! emit {
        ($step:expr, $gens:expr) => {{
            let s = $step;
            steps.push(s);
            for k in $gens.iter_mut() {
                let img = conjugate_by_step(
                    PauliString {
                        x: k.0,
                        z: k.1,
                        negative: false,
                    },
                    s,
                );
                *k = (img.x, img.z);
            }
        }};
    }

    {
        let groups = &mut gens;
        loop {
            groups.retain(|&(x, z)| x | z != 0);
            // Repeated Trotter steps hand the same axis in many times;
            // a duplicate would make the partner system inconsistent, so
            // the list is kept as a set.
            let mut seen = std::collections::HashSet::new();
            groups.retain(|k| seen.insert(*k));
            let Some(&g) = groups.first() else { break };
            let mut partner = (1..groups.len()).find(|&j| !commutes(g, groups[j]));
            if partner.is_none() {
                // Radical round: invent a partner nothing else can feel,
                // so the pivot stays invisible to the other components.
                let others: Vec<PauliKey> = groups[1..].to_vec();
                match orthogonal_partner(g, &others, !allocated) {
                    Some(h) => {
                        groups.push(h);
                        partner = Some(groups.len() - 1);
                    }
                    // No such partner exists exactly when `g` lies in the
                    // span of the rest, in which case it needs no qubit of
                    // its own: its image is already a product of theirs.
                    None => {
                        groups[0] = (0, 0);
                        continue;
                    }
                }
            }

            // Every round is hyperbolic — the partner is either a real
            // generator or an invented one — so `g` always steers to a
            // single-site X. Make it X-type first: S maps Y to X and H
            // maps Z to X, both local to the site they act on.
            let partner = partner.expect("every round has a partner");
            loop {
                let (x, z) = groups[0];
                let Some(q) = (0..qubits).find(|q| z >> q & 1 == 1) else {
                    break;
                };
                if x >> q & 1 == 1 {
                    emit!(CliffordStep::S(q), groups);
                } else {
                    emit!(CliffordStep::H(q), groups);
                }
            }

            // Pivot: lowest supported qubit. The clearing step below is
            // what makes it fresh; this check pins that invariant.
            let (gx, gz) = groups[0];
            let support = gx | gz;
            let q = support.trailing_zeros() as usize;
            if support & allocated != 0 || support == 0 {
                return Err(Error::InvalidState(
                    "decoupling_frame: generator reached an allocated qubit".into(),
                ));
            }

            // Fold the rest of the support onto the pivot: `X_qX_p` under
            // CX(q→p) is `X_q`.
            loop {
                let (gx, gz) = groups[0];
                let rest = (gx | gz) & !(1u64 << q);
                if rest == 0 {
                    break;
                }
                let p = rest.trailing_zeros() as usize;
                emit!(CliffordStep::Cx(q, p), groups);
            }

            {
                let j = partner;
                // The partner anticommutes with `X_q`, so it carries Z or Y there.
                // so it carries Z or Y there. `HSH` fixes X and maps
                // Y ↦ −Z, so normalize it to Z first.
                if groups[j].0 >> q & 1 == 1 {
                    emit!(CliffordStep::H(q), groups);
                    emit!(CliffordStep::S(q), groups);
                    emit!(CliffordStep::H(q), groups);
                }
                // Strip its support away from the pivot: make each other
                // site Z, then CX(p→q) turns `Z_pZ_q` into `Z_q` and
                // leaves `X_q` alone.
                loop {
                    let (hx, hz) = groups[j];
                    let rest = (hx | hz) & !(1u64 << q);
                    if rest == 0 {
                        break;
                    }
                    let p = rest.trailing_zeros() as usize;
                    match (hx >> p & 1, hz >> p & 1) {
                        (1, 1) => {
                            emit!(CliffordStep::S(p), groups);
                            emit!(CliffordStep::H(p), groups);
                        }
                        (1, 0) => emit!(CliffordStep::H(p), groups),
                        _ => emit!(CliffordStep::Cx(p, q), groups),
                    }
                }
            }

            // Spend the qubit and clear it from every remaining generator
            // by multiplying in `X_q` / `Z_q` — a basis change inside the
            // span, so no information is lost.
            allocated |= 1u64 << q;
            let bit = 1u64 << q;
            groups[0] = (0, 0);
            groups[partner] = (0, 0);
            for k in groups.iter_mut() {
                // Both `X_q` and `Z_q` are in the span now, so either
                // bit can be multiplied away. With an invented partner
                // this is a no-op: nothing else can feel the pivot.
                k.0 &= !bit;
                k.1 &= !bit;
            }
        }
    }

    // The partition the frame actually achieved, read off the images.
    let frame = DecouplingFrame {
        steps,
        blocks: Vec::new(),
        qubits,
    };
    // The observable enters site by site — it is already a tensor
    // product, so it must not be the thing that fuses two blocks.
    let obs_image = frame.conjugate(observable).0;
    let blocks = overlap_blocks(
        rotations
            .iter()
            .map(|r| {
                let k = frame.conjugate(r.axis).0;
                k.0 | k.1
            })
            .chain(sites(obs_image).into_iter().map(|(q, _)| 1u64 << q)),
    );
    Ok(DecouplingFrame { blocks, ..frame })
}

// ── contracting against the framed input ─────────────────────────────

/// The stabilizer group of a stabilizer state, in reduced form.
///
/// Two things are asked of it: the sign of a Pauli's expectation
/// (`±1` if the Pauli is in the group, `0` otherwise — every other
/// Pauli is traceless against a stabilizer state), and whether the state
/// factorizes across a set of disjoint qubit blocks.
#[derive(Debug, Clone)]
pub struct Stabilizer {
    /// Generators reduced so each owns a distinct leading bit of the
    /// `2n`-bit `(x‖z)` vector.
    gens: Vec<PauliString>,
    /// Leading bit index of each generator.
    pivots: Vec<usize>,
}

/// Leading set bit of `(x‖z)`, X bits first, or `None` for identity.
fn leading(p: PauliString) -> Option<usize> {
    if p.x != 0 {
        Some(p.x.trailing_zeros() as usize)
    } else if p.z != 0 {
        Some(MAX_QUBITS + p.z.trailing_zeros() as usize)
    } else {
        None
    }
}

impl Stabilizer {
    /// Reduce a generating set to echelon form.
    pub fn new(gens: Vec<PauliString>) -> Self {
        let mut rows: Vec<PauliString> = Vec::new();
        let mut pivots: Vec<usize> = Vec::new();
        for mut g in gens {
            while let Some(lead) = leading(g) {
                match pivots.iter().position(|&p| p == lead) {
                    Some(i) => {
                        // Stabilizer generators commute, so the product
                        // stays Hermitian.
                        match g.times(rows[i]) {
                            Some(p) => g = p,
                            None => break,
                        }
                    }
                    None => {
                        rows.push(g);
                        pivots.push(lead);
                        break;
                    }
                }
            }
        }
        Stabilizer { gens: rows, pivots }
    }

    /// Number of independent generators.
    pub fn rank(&self) -> usize {
        self.gens.len()
    }

    /// `⟨φ| i^{|x∧z|}X^xZ^z |φ⟩` — `±1` when the string is in the group,
    /// `0` otherwise.
    pub fn expectation(&self, key: PauliKey) -> f64 {
        let mut p = PauliString {
            x: key.0,
            z: key.1,
            negative: false,
        };
        if self.gens.iter().any(|g| !g.commutes_with(p)) {
            return 0.0;
        }
        loop {
            let Some(lead) = leading(p) else {
                return if p.negative { -1.0 } else { 1.0 };
            };
            let Some(i) = self.pivots.iter().position(|&q| q == lead) else {
                return 0.0;
            };
            match p.times(self.gens[i]) {
                Some(next) => p = next,
                None => return 0.0,
            }
        }
    }

    /// Dimension of the subgroup supported inside `mask`.
    pub fn block_rank(&self, mask: u64) -> usize {
        // Eliminate on the complement: the generators whose complement
        // part vanishes span the block-local subgroup, and their count is
        // `rank − rank(complement projection)`.
        let mut rows: Vec<(u64, u64)> = self
            .gens
            .iter()
            .map(|g| (g.x & !mask, g.z & !mask))
            .collect();
        let mut rank = 0usize;
        for bit in 0..(2 * MAX_QUBITS) {
            let sel = |r: &(u64, u64)| {
                if bit < MAX_QUBITS {
                    r.0 >> bit & 1 == 1
                } else {
                    r.1 >> (bit - MAX_QUBITS) & 1 == 1
                }
            };
            let Some(p) = (rank..rows.len()).find(|&i| sel(&rows[i])) else {
                continue;
            };
            rows.swap(rank, p);
            let pivot = rows[rank];
            for (i, row) in rows.iter_mut().enumerate() {
                if i != rank && sel(row) {
                    row.0 ^= pivot.0;
                    row.1 ^= pivot.1;
                }
            }
            rank += 1;
        }
        self.gens.len() - rank
    }

    /// Whether the state is a product across `blocks` (plus the
    /// unclaimed qubits, taken one per site). True exactly when the
    /// block-local subgroups already account for the full rank.
    pub fn separable(&self, blocks: &[u64], qubits: usize) -> bool {
        let covered = blocks.iter().fold(0u64, |a, b| a | b);
        let mut total: usize = blocks.iter().map(|&m| self.block_rank(m)).sum();
        for q in 0..qubits {
            if covered >> q & 1 == 0 {
                total += self.block_rank(1u64 << q);
            }
        }
        total >= self.gens.len()
    }
}

// ── the whole pipeline ───────────────────────────────────────────────

/// What [`propagate_engineered`] measured.
#[derive(Debug, Clone)]
pub struct EngineeredReport {
    /// `⟨0…0|U†PU|0…0⟩`, exact.
    pub value: f64,
    /// Blocks a support-overlap partition finds in the lab frame — the
    /// number [`crate::heisenberg::propagate_factored`] would discover.
    pub support_blocks: usize,
    /// Blocks the walk ended with in the engineered frame: the same rule
    /// applied to the framed circuit, so the two numbers are directly
    /// comparable.
    pub engineered_blocks: usize,
    /// Anticommutation components containing the observable — the number
    /// of independent problems the circuit poses, and the ceiling the
    /// frame is trying to reach.
    pub live_components: usize,
    /// Rotations deleted as inert. Exact, not approximate.
    pub inert_rotations: usize,
    /// Elementary Cliffords in the frame.
    pub frame_steps: usize,
    /// Widest engineered block, in qubits.
    pub widest_block: usize,
    /// Final blocks as `(qubit mask, terms)`.
    pub blocks: Vec<(u64, usize)>,
    /// Peak stored terms — the sum over blocks.
    pub peak_stored: usize,
    /// Peak flat term count — the product over blocks, i.e. what the
    /// unfactored walk would have carried.
    pub peak_flat: u128,
    /// Whether `V†|0…0⟩` factors across the engineered blocks. When
    /// false the product shortcut is invalid and the contraction was
    /// done flat — the entanglement the operator shed is in the input.
    pub separable_input: bool,
}

impl EngineeredReport {
    /// Terms the factorization avoided, at peak.
    pub fn factor_saving(&self) -> f64 {
        self.peak_flat as f64 / self.peak_stored.max(1) as f64
    }
}

/// Propagate an observable backwards in the **engineered frame**: drop
/// the inert components, change basis so the live ones sit on disjoint
/// qubits, and hold the operator factored across them.
///
/// Exact throughout — no threshold, no truncation. The claim being
/// measured is structural: how many independent problems the circuit
/// poses, and how many terms never had to exist.
pub fn propagate_engineered(
    observable: PauliKey,
    rotations: &[Rotation],
    qubits: usize,
) -> Result<EngineeredReport> {
    let coupling = coupling_of(observable, rotations);
    let live = coupling.live_rotations(rotations);
    let frame = decoupling_frame(observable, &live, qubits)?;
    let framed = frame.rewrite(&live);
    let (obs_key, obs_sign) = frame.conjugate(observable);
    let stab = frame.input_stabilizer();

    let mut f = FactoredPauliSum::from_key(obs_key);
    let mut peak_stored = f.stored_terms();
    let mut peak_flat = f.flat_terms();
    for rot in framed.iter().rev() {
        f.conjugate(rot);
        peak_stored = peak_stored.max(f.stored_terms());
        peak_flat = peak_flat.max(f.flat_terms());
    }

    // Separability has to be tested against the partition the walk
    // actually ended with, not the frame's: the walk starts from the
    // observable's own support and so can end up strictly finer, and a
    // state that factors across a coarse partition need not factor
    // across a finer one.
    let blocks: Vec<u64> = f.blocks().iter().map(|&(m, _)| m).collect();
    let separable = stab.separable(&blocks, qubits);

    let value = if separable {
        contract_factored(&f, &stab)
    } else {
        contract_flat(&f, &stab)?
    };

    // `from_key` normalizes the observable to `X^xZ^z`; the Hermitian
    // form carries `i^{|x∧z|}` and the frame's sign, and the phase of
    // the pre-frame key divides back out.
    let phase = axis_operator_phase(obs_key) / axis_operator_phase(observable);
    let value = (value * phase * C64::new(obs_sign, 0.0)).re;

    Ok(EngineeredReport {
        value,
        support_blocks: support_blocks(observable, rotations),
        engineered_blocks: blocks.len(),
        live_components: coupling.live_components(),
        inert_rotations: coupling.inert_rotations(),
        frame_steps: frame.len(),
        widest_block: blocks
            .iter()
            .map(|m| m.count_ones() as usize)
            .max()
            .unwrap_or(0),
        blocks: f.blocks(),
        peak_stored,
        peak_flat,
        separable_input: separable,
    })
}

/// `⟨φ|A⊗B⊗…|φ⟩` as a product over blocks — valid when `|φ⟩` factors
/// across them, which is what the caller has checked.
fn contract_factored(f: &FactoredPauliSum, stab: &Stabilizer) -> C64 {
    let mut acc = f.scale();
    for (_, sum) in f.block_sums() {
        let mut b = C64::new(0.0, 0.0);
        for (key, c) in sum.terms() {
            let s = stab.expectation(key);
            if s != 0.0 {
                // `terms` are coefficients of the raw `X^xZ^z`, whose
                // expectation is the Hermitian one divided by `i^{|x∧z|}`.
                b += c * C64::new(s, 0.0) / axis_operator_phase(key);
            }
        }
        acc *= b;
    }
    acc
}

/// The same contraction without the product shortcut: expand the blocks
/// into the flat sum and evaluate term by term. Correct for any input
/// state, and exponential in the number of blocks — which is exactly why
/// [`EngineeredReport::separable_input`] is worth reporting.
fn contract_flat(f: &FactoredPauliSum, stab: &Stabilizer) -> Result<C64> {
    if f.flat_terms() > FLAT_CONTRACTION_CAP {
        return Err(Error::InvalidState(format!(
            "engineered frame left the input entangled across blocks and the flat \
             contraction needs {} terms (cap {FLAT_CONTRACTION_CAP})",
            f.flat_terms()
        )));
    }
    let mut flat: Vec<(PauliKey, C64)> = vec![((0, 0), f.scale())];
    for (_, sum) in f.block_sums() {
        let mut next = Vec::with_capacity(flat.len() * sum.len().max(1));
        for (ka, ca) in &flat {
            for (kb, cb) in sum.terms() {
                // disjoint supports, so the product carries no sign
                next.push(((ka.0 | kb.0, ka.1 | kb.1), *ca * cb));
            }
        }
        flat = next;
    }
    let mut acc = C64::new(0.0, 0.0);
    for (key, c) in flat {
        let s = stab.expectation(key);
        if s != 0.0 {
            acc += c * C64::new(s, 0.0) / axis_operator_phase(key);
        }
    }
    Ok(acc)
}

// ── circuits that look coupled and are not ───────────────────────────

/// Conjugate a circuit by a Clifford scrambler, producing an equivalent
/// problem whose axes are wide and whose qubit-support partition is a
/// single block.
///
/// This is the demonstration that the support partition measures the
/// basis and not the physics: `V` preserves the symplectic form, so the
/// anticommutation components — and therefore the engineered block
/// structure and every expectation value — are untouched, while the
/// supports spread to all-to-all.
pub fn scramble(
    rotations: &[Rotation],
    observable: PauliKey,
    qubits: usize,
) -> (Vec<Rotation>, PauliKey, C64) {
    let mut steps = Vec::new();
    // A CX ladder up and back, then a stride sweep: enough to give every
    // axis support across the whole register, and built from CX only, so
    // the scrambler fixes `|0…0⟩` and the comparison against a direct
    // simulation needs no extra state preparation.
    for q in 0..qubits.saturating_sub(1) {
        steps.push(CliffordStep::Cx(q, q + 1));
    }
    for q in (1..qubits).rev() {
        steps.push(CliffordStep::Cx(q, q - 1));
    }
    for q in 0..qubits {
        let t = (q + qubits / 2) % qubits.max(1);
        if t != q {
            steps.push(CliffordStep::Cx(q, t));
        }
    }
    let frame = DecouplingFrame {
        steps,
        blocks: Vec::new(),
        qubits,
    };
    let (key, sign) = frame.conjugate(observable);
    // `V†(X^xZ^z)V = σ · i^{|x'∧z'|}/i^{|x∧z|} · X^{x'}Z^{z'}`.
    let coeff = C64::new(sign, 0.0) * axis_operator_phase(key)
        / axis_operator_phase(observable);
    (frame.rewrite(rotations), key, coeff)
}
