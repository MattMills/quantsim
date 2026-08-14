//! Dynamically scaled register with a recursive bulk–boundary
//! representation: the state stored **across depth, projectively**.
//!
//! A [`BulkState`] keeps the register as a perfect binary tree of
//! **isometries** over a power-of-two capacity, with every scalar of
//! state weight held in one explicit **bulk record** — the [`top`]
//! coefficient vector at the root. Reading the structure inward from
//! the physical qubits (the *boundary*, at the leaves), each level is
//! an isometric coarse-graining; reading it outward from [`top`], each
//! level is a projective refinement. The state is never stored flat:
//! it **is** the stack of per-depth isometries applied to the bulk
//! record, and the image subspaces of the partial stacks form a nested
//! projector tower `P₀ ⊇ P₁ ⊇ …` in which the represented state sits
//! exactly. Truncation events are the only projective losses, and they
//! are ledgered **per depth** ([`BulkState::discarded_by_depth`]) with
//! a running global bound ([`BulkState::l2_error_bound`]).
//!
//! Three things distinguish this from the rung-1 hierarchy
//! ([`MeraState`](super::MeraState)):
//!
//! * **The gauge is maintained.** Every node tensor is kept isometric
//!   (up-bond → subtree), so `‖ψ‖ = ‖top‖` *identically* — norm and
//!   total weight are `O(χ)` reads at any width — and truncations are
//!   taken against the **true environment**: before a subtree is
//!   re-compressed, the environment's Gram factor is transported down
//!   from `top` (an `X` with `X†X = ρ`), and every split optimizes the
//!   weighted state `X·B`. Discarded singular values are therefore
//!   *global* Schmidt weight, not block-local weight — the roadmap's
//!   gauge-maintenance rung, delivered by environment-weighted
//!   rebuilds rather than by center transport. (Sequential truncations
//!   still do not commute; the certified statement is the accumulated
//!   bound `‖ψ_ideal − ψ_stored‖ ≤ Σ√εᵢ`, measured in the tests.)
//! * **The register scales dynamically.** [`BulkState::grow`] appends
//!   fresh `|0⟩` qubits at the boundary in amortized `O(1)` tensors:
//!   within capacity, growth is a bookkeeping change (the dormant
//!   suffix of the capacity tree is pristine by invariant); past
//!   capacity, the register **re-roots** — the whole current tree
//!   becomes the left child of a new root, i.e. the register becomes a
//!   *site* of a larger register, the point-is-a-lattice move of
//!   [`recursive`](crate::recursive) applied to the state
//!   representation itself. [`BulkState::release`] removes boundary
//!   qubits again, refusing with the **measured leakage** unless the
//!   qubit is verifiably `|0⟩`; [`BulkState::release_measured`]
//!   measures first, so it always succeeds. Width becomes a trajectory
//!   ([`BulkState::peak_width`], [`BulkState::reroots`]), and a
//!   register can process a stream of logical qubits far wider than it
//!   ever is.
//! * **The depth store unfolds into structured entanglement.**
//!   [`BulkState::unfold_program`] compiles the stored representation
//!   into an explicit [`UnfoldProgram`]: a seed state on the bulk
//!   channel plus one dilated unitary per node, in depth order, each
//!   consuming fresh `|0⟩` wires (Stinespring dilation of the level
//!   isometries). Replaying the program on any backend reproduces the
//!   boundary state **exactly**, and the intermediate state after the
//!   first `ℓ` levels *is* the coarse state at depth `ℓ`
//!   ([`BulkState::coarse_state`]) on the channel wires — asserted in
//!   the tests. Each wire's entanglement history is its
//!   [`UnfoldProgram::sequence`]: a qubit interacts **only** along its
//!   designated path through the hierarchy, so entanglement between
//!   tree-aligned boundary regions is carried by nameable channel
//!   wires with Schmidt rank capped by the crossing bond — dimension
//!   added *above* the problem separates its entanglement into
//!   structured, addressable channels (the same up-a-dimension move
//!   that [`lift`](crate::lift) and [`upembed`](crate::upembed) make
//!   for magic, made here for entanglement).
//!
//! Rung-1 honesty carries over: a gate spanning both halves of a
//! subtree materializes that subtree's **live** dense block (bounded by
//! [`BulkConfig::max_block`] and the [resource guard](crate::guard)),
//! applies the gate exactly, and re-compresses by environment-weighted
//! Schmidt splits. Disentanglers and operator-level path updates remain
//! the next rungs (see the roadmap).

use super::{validate_apply, validate_apply_diagonal, Backend};
use crate::error::{Error, Result};
use crate::math::{svd_thin, GateMatrix};
use crate::scalar::Scalar;

/// Maximum register width (basis indices are `u64`).
pub const BULK_MAX_QUBITS: usize = 63;

/// Widest dilated step [`BulkState::unfold_program`] will emit
/// (the unitary is dense: `4^k` scalars, completed by Gram–Schmidt).
pub const UNFOLD_MAX_STEP_QUBITS: usize = 6;

/// Configuration for the bulk register.
#[derive(Debug, Clone, Copy)]
pub struct BulkConfig {
    /// Hard cap on any bond dimension; excess Schmidt weight is
    /// truncated and ledgered per depth.
    pub max_bond: usize,
    /// Relative singular-value cutoff at each split.
    pub trunc_tol: f64,
    /// Widest **live** subtree a single gate may force into a dense
    /// block — a policy cap below the machine's real limits (the
    /// transient buffer is additionally admitted by the resource
    /// guard).
    pub max_block: usize,
    /// Maximum measured weight outside `|0⟩` a qubit may carry and
    /// still be [`release`](BulkState::release)d. Above it, release
    /// refuses and reports the number.
    pub release_tol: f64,
}

impl Default for BulkConfig {
    fn default() -> Self {
        BulkConfig {
            max_bond: 64,
            trunc_tol: 1e-12,
            max_block: 63,
            release_tol: 1e-12,
        }
    }
}

/// One node of the capacity tree over the qubit range `[lo, hi)`.
///
/// Semantics: the subtree is a map `A[u][p]` from its up-bond `u`
/// (dimension `bond`) to amplitudes over its **live** physical range
/// (little-endian: bit `i` of `p` is qubit `lo + i`). A leaf stores
/// `A` directly (`[bond][2]`); an internal node stores the combiner
/// `W[u][a][b]` with
/// `A[u][(pr << wl) | pl] = Σ W[u][a][b]·L[a][pl]·R[b][pr]`.
///
/// Gauge invariant: every node tensor has orthonormal rows (it is an
/// isometry from the up-bond into the children's joint space), so the
/// subtree map is an isometry and all state weight lives in
/// [`BulkState::top`]. Nodes entirely at or beyond the live width are
/// **pristine**: bond 1, `|0…0⟩`, structurally identical to freshly
/// grown ones.
#[derive(Debug, Clone)]
struct Node<S: Scalar> {
    lo: usize,
    hi: usize,
    bond: usize,
    tensor: Vec<S>,
    children: Children<S>,
}

type Children<S> = Option<(Box<Node<S>>, Box<Node<S>>)>;

impl<S: Scalar> Node<S> {
    /// Fresh `|0…⟩` subtree, bond 1 everywhere (isometric trivially).
    fn pristine(lo: usize, hi: usize) -> Self {
        if hi - lo == 1 {
            return Node {
                lo,
                hi,
                bond: 1,
                tensor: vec![S::one(), S::zero()],
                children: None,
            };
        }
        let mid = lo + (hi - lo).div_ceil(2);
        Node {
            lo,
            hi,
            bond: 1,
            tensor: vec![S::one()],
            children: Some((
                Box::new(Node::pristine(lo, mid)),
                Box::new(Node::pristine(mid, hi)),
            )),
        }
    }

    /// Structural midpoint (capacity split, not live split).
    fn mid(&self) -> usize {
        self.lo + (self.hi - self.lo).div_ceil(2)
    }

    /// Live width under boundary width `n`.
    fn live_width(&self, n: usize) -> usize {
        self.hi.min(n).saturating_sub(self.lo)
    }

    /// Whether the subtree is entirely dormant under width `n`.
    fn dormant(&self, n: usize) -> bool {
        self.lo >= n
    }

    /// Dense block `[bond][2^live_width]` of this subtree's map,
    /// skipping the pristine dormant suffix.
    fn contract_live(&self, n: usize) -> Result<Vec<S>> {
        if self.dormant(n) {
            debug_assert_eq!(self.bond, 1, "dormant subtree must be pristine");
            return Ok(vec![S::one()]);
        }
        match &self.children {
            None => Ok(self.tensor.clone()),
            Some((l, r)) => {
                let (wl, wr) = (l.live_width(n), r.live_width(n));
                let (dl, dr) = (1usize << wl, 1usize << wr);
                let bl = l.contract_live(n)?;
                let br = r.contract_live(n)?;
                let mut out = crate::guard::try_vec(
                    self.bond << (wl + wr),
                    S::zero(),
                    "bulk block contraction",
                )?;
                for u in 0..self.bond {
                    crate::guard::checkpoint()?;
                    for a in 0..l.bond {
                        for b in 0..r.bond {
                            let w = self.tensor[(u * l.bond + a) * r.bond + b];
                            if w.abs_sqr() == 0.0 {
                                continue;
                            }
                            for pr in 0..dr {
                                let rv = br[b * dr + pr];
                                if rv.abs_sqr() == 0.0 {
                                    continue;
                                }
                                let wrb = w * rv;
                                let base = (u << (wl + wr)) | (pr << wl);
                                for pl in 0..dl {
                                    let lv = bl[a * dl + pl];
                                    if lv.abs_sqr() == 0.0 {
                                        continue;
                                    }
                                    out[base | pl] = out[base | pl] + wrb * lv;
                                }
                            }
                        }
                    }
                }
                Ok(out)
            }
        }
    }

    /// Bond-vector `A[·][p]` for the bits of `index` in `[lo, hi)`.
    fn amplitude_vec(&self, index: u64, n: usize) -> Vec<S> {
        if self.dormant(n) {
            return vec![S::one()];
        }
        match &self.children {
            None => {
                let bit = ((index >> self.lo) & 1) as usize;
                (0..self.bond).map(|u| self.tensor[u * 2 + bit]).collect()
            }
            Some((l, r)) => {
                let lv = l.amplitude_vec(index, n);
                let rv = r.amplitude_vec(index, n);
                (0..self.bond)
                    .map(|u| {
                        let mut acc = S::zero();
                        for (a, &la) in lv.iter().enumerate() {
                            if la.abs_sqr() == 0.0 {
                                continue;
                            }
                            for (b, &rb) in rv.iter().enumerate() {
                                acc = acc + self.tensor[(u * l.bond + a) * r.bond + b] * la * rb;
                            }
                        }
                        acc
                    })
                    .collect()
            }
        }
    }

    /// Depth-first support walk over the live range, all-zero vectors
    /// pruned. Exponential for generic states (documented trait
    /// behavior), linear-ish for concentrated ones.
    fn support(&self, n: usize) -> Vec<(u64, Vec<S>)> {
        if self.dormant(n) {
            return vec![(0, vec![S::one()])];
        }
        match &self.children {
            None => (0..2u64)
                .filter_map(|p| {
                    let v: Vec<S> = (0..self.bond)
                        .map(|u| self.tensor[u * 2 + p as usize])
                        .collect();
                    v.iter().any(|x| x.abs_sqr() > 0.0).then_some((p, v))
                })
                .collect(),
            Some((l, r)) => {
                let wl = l.live_width(n);
                let ls = l.support(n);
                let rs = r.support(n);
                let mut out = Vec::new();
                for (ri, rv) in &rs {
                    for (li, lv) in &ls {
                        let v: Vec<S> = (0..self.bond)
                            .map(|u| {
                                let mut acc = S::zero();
                                for (a, &la) in lv.iter().enumerate() {
                                    if la.abs_sqr() == 0.0 {
                                        continue;
                                    }
                                    for (b, &rb) in rv.iter().enumerate() {
                                        acc = acc
                                            + self.tensor[(u * l.bond + a) * r.bond + b] * la * rb;
                                    }
                                }
                                acc
                            })
                            .collect();
                        if v.iter().any(|x| x.abs_sqr() > 0.0) {
                            out.push(((ri << wl) | li, v));
                        }
                    }
                }
                out
            }
        }
    }

    fn max_bond(&self, n: usize) -> usize {
        if self.dormant(n) {
            return 1;
        }
        let own = self.bond;
        match &self.children {
            None => own,
            Some((l, r)) => own.max(l.max_bond(n)).max(r.max_bond(n)),
        }
    }

    fn tensor_elements(&self) -> usize {
        self.tensor.len()
            + self
                .children
                .as_ref()
                .map(|(l, r)| l.tensor_elements() + r.tensor_elements())
                .unwrap_or(0)
    }

    /// Live frontier at `depth`: `(lo, live_hi, bond)` triples, left to
    /// right, dormant subtrees skipped.
    fn frontier(&self, depth: usize, n: usize, out: &mut Vec<(usize, usize, usize)>) {
        if self.dormant(n) {
            return;
        }
        if depth == 0 || self.children.is_none() {
            out.push((self.lo, self.hi.min(n), self.bond));
            return;
        }
        let (l, r) = self.children.as_ref().expect("checked");
        l.frontier(depth - 1, n, out);
        r.frontier(depth - 1, n, out);
    }

    /// Contraction of everything above the depth-`d` live frontier:
    /// the map `[bond][frontier product]`, frontier order little-endian
    /// (first frontier node varies fastest).
    fn coarse(&self, depth: usize, n: usize) -> Result<(Vec<usize>, Vec<S>)> {
        if self.dormant(n) {
            debug_assert_eq!(self.bond, 1);
            return Ok((Vec::new(), vec![S::one()]));
        }
        if depth == 0 || self.children.is_none() {
            let mut id = vec![S::zero(); self.bond * self.bond];
            for u in 0..self.bond {
                id[u * self.bond + u] = S::one();
            }
            return Ok((vec![self.bond], id));
        }
        let (l, r) = self.children.as_ref().expect("checked");
        let (ldims, lc) = l.coarse(depth - 1, n)?;
        let (rdims, rc) = r.coarse(depth - 1, n)?;
        let lprod: usize = ldims.iter().product();
        let rprod: usize = rdims.iter().product();
        let mut out =
            crate::guard::try_vec(self.bond * lprod * rprod, S::zero(), "bulk coarse view")?;
        for u in 0..self.bond {
            for a in 0..l.bond {
                for b in 0..r.bond {
                    let w = self.tensor[(u * l.bond + a) * r.bond + b];
                    if w.abs_sqr() == 0.0 {
                        continue;
                    }
                    for jr in 0..rprod {
                        let rv = rc[b * rprod + jr];
                        if rv.abs_sqr() == 0.0 {
                            continue;
                        }
                        let wrb = w * rv;
                        let base = (u * rprod + jr) * lprod;
                        for jl in 0..lprod {
                            let lv = lc[a * lprod + jl];
                            if lv.abs_sqr() == 0.0 {
                                continue;
                            }
                            out[base + jl] = out[base + jl] + wrb * lv;
                        }
                    }
                }
            }
        }
        let mut dims = ldims;
        dims.extend(rdims);
        Ok((dims, out))
    }
}

/// `⌈log₂ x⌉` for `x ≥ 1`.
fn clog2(x: usize) -> usize {
    debug_assert!(x >= 1);
    x.next_power_of_two().trailing_zeros() as usize
}

/// Isometrize a `rows × cols` tensor: return `(vt, m, rank)` with
/// `tensor = m · vt`, `vt` having orthonormal rows (the new isometric
/// tensor) and `m = U·Σ` (`rows × rank`) the factor the parent absorbs.
/// Exact bookkeeping — nothing is truncated here beyond hard numerical
/// rank.
fn isometrize<S: Scalar>(rows: usize, cols: usize, t: &[S]) -> Result<(Vec<S>, Vec<S>, usize)> {
    let svd = svd_thin(rows, cols, t, 0.0, 0.0)?;
    if svd.rank == 0 {
        // Zero tensor (a fully projected branch): keep a rank-1 frame so
        // shapes stay well-formed; the zero weight lives in `m`.
        let mut vt = vec![S::zero(); cols];
        if cols > 0 {
            vt[0] = S::one();
        }
        return Ok((vt, vec![S::zero(); rows], 1));
    }
    let rank = svd.rank;
    let mut m = vec![S::zero(); rows * rank];
    for u in 0..rows {
        for k in 0..rank {
            m[u * rank + k] = svd.u[u * rank + k].scale(svd.sigma[k]);
        }
    }
    Ok((svd.vt, m, rank))
}

/// Row-compress an environment factor `c` (`rows × cols`) into an
/// equivalent `x` (`rank × cols`) with `x†x = c†c`. The Gram factor is
/// all the weighted splits consume, so the row count never grows down
/// the tree.
fn compress_env<S: Scalar>(rows: usize, cols: usize, c: &[S]) -> Result<(Vec<S>, usize)> {
    let svd = svd_thin(rows, cols, c, 0.0, 0.0)?;
    if svd.rank == 0 {
        return Ok((vec![S::zero(); cols.max(1)], 1));
    }
    let mut x = vec![S::zero(); svd.rank * cols];
    for k in 0..svd.rank {
        for j in 0..cols {
            x[k * cols + j] = svd.vt[k * cols + j].scale(svd.sigma[k]);
        }
    }
    Ok((x, svd.rank))
}

/// Truncation ledger: per-depth discarded σ² plus the running certified
/// bound `Σ√ε` over events.
#[derive(Debug, Clone, Default)]
struct Ledger {
    by_depth: Vec<f64>,
    sum_sqrt: f64,
}

impl Ledger {
    fn add(&mut self, depth: usize, event: f64) {
        if event <= 0.0 {
            return;
        }
        if self.by_depth.len() <= depth {
            self.by_depth.resize(depth + 1, 0.0);
        }
        self.by_depth[depth] += event;
        self.sum_sqrt += event.sqrt();
    }

    fn total(&self) -> f64 {
        self.by_depth.iter().sum()
    }

    /// A re-root pushes every recorded depth one level down.
    fn shift_root(&mut self) {
        if !self.by_depth.is_empty() {
            self.by_depth.insert(0, 0.0);
        }
    }
}

/// The dynamically scaled bulk–boundary register. See the module docs.
#[derive(Debug, Clone)]
pub struct BulkState<S: Scalar> {
    config: BulkConfig,
    root: Node<S>,
    /// The bulk record: state coefficients over the root's up-bond.
    top: Vec<S>,
    /// Live boundary width (`≤ root.hi`, the capacity).
    n: usize,
    ledger: Ledger,
    peak_block_elements: usize,
    peak_width: usize,
    reroots: usize,
    releases: usize,
}

impl<S: Scalar> BulkState<S> {
    /// `|0…0⟩` of width `num_qubits` with the default configuration.
    pub fn new(num_qubits: usize) -> Result<Self> {
        Self::with_config(num_qubits, BulkConfig::default())
    }

    /// `|0…0⟩` with an explicit configuration. Requires a commutative
    /// division algebra (the Jacobi SVD assumes commuting scalars).
    pub fn with_config(num_qubits: usize, config: BulkConfig) -> Result<Self> {
        if !(S::COMMUTATIVE && S::DIVISION) {
            return Err(Error::InvalidState(format!(
                "bulk requires a commutative division algebra; {} is not",
                S::algebra_name()
            )));
        }
        if num_qubits == 0 || num_qubits > BULK_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: num_qubits,
                max: BULK_MAX_QUBITS,
            });
        }
        if config.max_bond == 0 {
            return Err(Error::InvalidState("bulk: max_bond must be ≥ 1".into()));
        }
        let capacity = num_qubits.next_power_of_two();
        Ok(BulkState {
            config,
            root: Node::pristine(0, capacity),
            top: vec![S::one()],
            n: num_qubits,
            ledger: Ledger::default(),
            peak_block_elements: 1,
            peak_width: num_qubits,
            reroots: 0,
            releases: 0,
        })
    }

    /// The bulk record: the state's coefficients over the root bond
    /// basis. `‖top‖²` **is** `‖ψ‖²` (gauge invariant).
    pub fn top(&self) -> &[S] {
        &self.top
    }

    /// The configuration in effect.
    pub fn config(&self) -> BulkConfig {
        self.config
    }

    /// Capacity of the current tree (`≥ num_qubits`; grows by doubling).
    pub fn capacity(&self) -> usize {
        self.root.hi
    }

    /// Total discarded singular-value weight `Σσ²` over every
    /// truncation since construction/reset. Zero means the depth store
    /// is an exact representation of everything applied to it.
    pub fn discarded_weight(&self) -> f64 {
        self.ledger.total()
    }

    /// Discarded weight resolved **per depth** (index 0 = the root
    /// bond, deeper indices toward the boundary): where across the
    /// hierarchy resolution was spent.
    pub fn discarded_by_depth(&self) -> &[f64] {
        &self.ledger.by_depth
    }

    /// Certified accumulated L2 error bound:
    /// `‖ψ_ideal − ψ_stored‖ ≤ Σ_events √ε_event`. Each event's ε is
    /// global (environment-weighted) discarded weight, so the triangle
    /// inequality over events certifies the sum; the tests measure its
    /// tightness against dense.
    pub fn l2_error_bound(&self) -> f64 {
        self.ledger.sum_sqrt
    }

    /// Whether nothing above numerical noise has been truncated.
    pub fn is_exact(&self) -> bool {
        self.discarded_weight() < 1e-20
    }

    /// Largest dense block any single gate ever forced (elements).
    pub fn peak_block_elements(&self) -> usize {
        self.peak_block_elements
    }

    /// Largest bond dimension anywhere in the live hierarchy.
    pub fn max_bond_dimension(&self) -> usize {
        self.root.max_bond(self.n).max(self.top.len())
    }

    /// Depth of the capacity tree (root = 0; leaves at `log₂ capacity`).
    pub fn depth(&self) -> usize {
        clog2(self.capacity())
    }

    /// Widest the register has ever been.
    pub fn peak_width(&self) -> usize {
        self.peak_width
    }

    /// How many times growth re-rooted the tree (capacity doublings).
    pub fn reroots(&self) -> usize {
        self.reroots
    }

    /// How many qubits have been released back.
    pub fn releases(&self) -> usize {
        self.releases
    }

    // ── coarse views ────────────────────────────────────────────────

    /// The live frontier at `depth`: for each super-site (left to
    /// right) its qubit range and bond dimension.
    pub fn coarse_dims(&self, depth: usize) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        self.root.frontier(depth, self.n, &mut out);
        out
    }

    /// The exact state on the super-site basis at `depth`, with the
    /// bulk record folded in: `(per-site bond dimensions,
    /// coefficients)`, coefficient index little-endian over frontier
    /// sites. Depth 0 is the bulk record itself; leaf depth is the full
    /// state. Errs when the coarse space exceeds 2^20 coefficients.
    pub fn coarse_state(&self, depth: usize) -> Result<(Vec<usize>, Vec<S>)> {
        let dims: Vec<usize> = self.coarse_dims(depth).iter().map(|&(_, _, b)| b).collect();
        let prod: usize = dims.iter().product();
        if prod > (1 << 20) {
            return Err(Error::InvalidState(format!(
                "coarse space at depth {depth} has {prod} coefficients (> 2^20)"
            )));
        }
        let (dims, raw) = self.root.coarse(depth, self.n)?;
        let prod: usize = dims.iter().product();
        let mut out = vec![S::zero(); prod];
        for (j, slot) in out.iter_mut().enumerate() {
            let mut acc = S::zero();
            for (u, &t) in self.top.iter().enumerate() {
                acc = acc + t * raw[u * prod + j];
            }
            *slot = acc;
        }
        Ok((dims, out))
    }

    /// The refinement dictionary of the `position`-th live frontier
    /// node at `depth` (see [`MeraState::refine_basis`]'s conventions):
    /// `(up_bond, left_bond, right_bond, tensor)` for an internal node,
    /// `(bond, 2, 1, tensor)` for a leaf.
    ///
    /// [`MeraState::refine_basis`]: super::MeraState::refine_basis
    pub fn refine_basis(
        &self,
        depth: usize,
        position: usize,
    ) -> Result<(usize, usize, usize, Vec<S>)> {
        fn find<'a, S: Scalar>(
            node: &'a Node<S>,
            depth: usize,
            n: usize,
            pos: &mut usize,
        ) -> Option<&'a Node<S>> {
            if node.dormant(n) {
                return None;
            }
            if depth == 0 || node.children.is_none() {
                if *pos == 0 {
                    return Some(node);
                }
                *pos -= 1;
                return None;
            }
            let (l, r) = node.children.as_ref().expect("checked");
            if let Some(hit) = find(l, depth - 1, n, pos) {
                return Some(hit);
            }
            find(r, depth - 1, n, pos)
        }
        let mut p = position;
        let node = find(&self.root, depth, self.n, &mut p).ok_or_else(|| {
            Error::InvalidState(format!("no live frontier node {position} at depth {depth}"))
        })?;
        match &node.children {
            None => Ok((node.bond, 2, 1, node.tensor.clone())),
            Some((l, r)) => Ok((node.bond, l.bond, r.bond, node.tensor.clone())),
        }
    }

    /// The register re-expressed **at scale `depth`**, as a state of
    /// the same boundary width: the upper `depth` levels and the bulk
    /// record are kept, and every deeper level is replaced by the
    /// **channel-basis embedding** — each depth-`depth` super-site's
    /// bond index written as bits on the first `⌈log₂ bond⌉` wires of
    /// its span, all other wires exactly `|0⟩`. This is precisely the
    /// state [`UnfoldProgram::run_to_depth`] produces at `depth`
    /// (asserted in the tests), built here by `O(tree)` node surgery
    /// with no gate ever applied — the projective depth store read out
    /// as a *family of states, one per scale*. The snapshot is exact
    /// with respect to the stored representation (fresh ledger).
    pub fn scale_snapshot(&self, depth: usize) -> Result<Self> {
        if depth > self.depth() {
            return Err(Error::InvalidState(format!(
                "scale_snapshot: depth {depth} exceeds hierarchy depth {}",
                self.depth()
            )));
        }
        // Channel-basis embedding over `[lo, hi)`: `|u⟩ ↦` bits of `u`
        // on the first wires of the span, `|0⟩` elsewhere. Isometric by
        // construction (each row a single unit entry).
        fn embed<S: Scalar>(lo: usize, hi: usize, bond: usize, n: usize) -> Node<S> {
            if lo >= n {
                debug_assert_eq!(bond, 1, "dormant embedding must be trivial");
                return Node::pristine(lo, hi);
            }
            if hi - lo == 1 {
                let tensor = if bond == 2 {
                    vec![S::one(), S::zero(), S::zero(), S::one()]
                } else {
                    vec![S::one(), S::zero()]
                };
                return Node {
                    lo,
                    hi,
                    bond,
                    tensor,
                    children: None,
                };
            }
            let mid = lo + (hi - lo).div_ceil(2);
            let c = clog2(bond);
            let ca_bits = c.min(mid - lo);
            let ca = bond.min(1 << ca_bits);
            let cb = bond.div_ceil(1 << ca_bits);
            let mut tensor = vec![S::zero(); bond * ca * cb];
            for u in 0..bond {
                let a = u & ((1 << ca_bits) - 1);
                let b = u >> ca_bits;
                tensor[(u * ca + a) * cb + b] = S::one();
            }
            Node {
                lo,
                hi,
                bond,
                tensor,
                children: Some((
                    Box::new(embed(lo, mid, ca, n)),
                    Box::new(embed(mid, hi, cb, n)),
                )),
            }
        }
        fn splice<S: Scalar>(node: &Node<S>, depth: usize, n: usize) -> Node<S> {
            if node.dormant(n) {
                return node.clone();
            }
            if depth == 0 || node.children.is_none() {
                return embed(node.lo, node.hi, node.bond, n);
            }
            let (l, r) = node.children.as_ref().expect("checked");
            Node {
                lo: node.lo,
                hi: node.hi,
                bond: node.bond,
                tensor: node.tensor.clone(),
                children: Some((
                    Box::new(splice(l, depth - 1, n)),
                    Box::new(splice(r, depth - 1, n)),
                )),
            }
        }
        Ok(BulkState {
            config: self.config,
            root: splice(&self.root, depth, self.n),
            top: self.top.clone(),
            n: self.n,
            ledger: Ledger::default(),
            peak_block_elements: 1,
            peak_width: self.n,
            reroots: 0,
            releases: 0,
        })
    }

    // ── dynamic scaling ─────────────────────────────────────────────

    /// Append `k` fresh `|0⟩` qubits at the boundary (new indices
    /// `n…n+k−1`). Within capacity this is pure bookkeeping (the
    /// dormant suffix is pristine by invariant); past capacity the tree
    /// **re-roots** — the register becomes the left site of a register
    /// twice its size — at `O(capacity)` trivial tensors per doubling,
    /// amortized `O(1)` per qubit. Entanglement, bonds and the bulk
    /// record are untouched: growth is exact and disturbance-free.
    pub fn grow(&mut self, k: usize) -> Result<()> {
        let new_n = self.n + k;
        if new_n > BULK_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: new_n,
                max: BULK_MAX_QUBITS,
            });
        }
        while self.root.hi < new_n {
            let cap = self.root.hi;
            let bond = self.root.bond;
            let old = std::mem::replace(&mut self.root, Node::pristine(0, 1));
            let mut tensor = vec![S::zero(); bond * bond];
            for u in 0..bond {
                tensor[u * bond + u] = S::one();
            }
            self.root = Node {
                lo: 0,
                hi: 2 * cap,
                bond,
                tensor,
                children: Some((Box::new(old), Box::new(Node::pristine(cap, 2 * cap)))),
            };
            self.ledger.shift_root();
            self.reroots += 1;
        }
        self.n = new_n;
        self.peak_width = self.peak_width.max(self.n);
        Ok(())
    }

    /// Release the last `k` qubits (highest indices first), verifying
    /// each is `|0⟩` before detaching it. A qubit carrying measured
    /// weight above [`BulkConfig::release_tol`] outside `|0⟩` refuses
    /// with the number — release is checked, never assumed. Refuses to
    /// shrink below width 1.
    pub fn release(&mut self, k: usize) -> Result<()> {
        for _ in 0..k {
            self.release_one()?;
        }
        Ok(())
    }

    fn release_one(&mut self) -> Result<()> {
        if self.n <= 1 {
            return Err(Error::InvalidState(
                "release refused: register width is already 1".into(),
            ));
        }
        let q = self.n - 1;
        let leak = self.boundary_leakage(q)?;
        if leak > self.config.release_tol {
            return Err(Error::InvalidState(format!(
                "release refused: qubit {q} carries measured weight {leak:.3e} outside |0⟩ \
                 (tolerance {:.0e}); measure or uncompute it first",
                self.config.release_tol
            )));
        }
        // Snap the verified-negligible |1⟩ component to exactly zero and
        // let the rebuild collapse the leaf's bonds to 1.
        let renorm = if leak < 1.0 {
            1.0 / (1.0 - leak).sqrt()
        } else {
            1.0
        };
        let mut m = GateMatrix::<S>::zeros(2)?;
        m.set(0, 0, S::one().scale(renorm));
        self.apply_via_tree(&[q], |amps, local| {
            super::apply_single_in_place(amps, &m, local[0])
        })?;
        self.pristinize_leaf(q)?;
        self.n = q;
        self.releases += 1;
        Ok(())
    }

    /// Measure the last `k` qubits (highest first), flip any `1`
    /// outcome back to `|0⟩`, and release them. Always succeeds on a
    /// well-formed state; returns the outcomes in release order.
    pub fn release_measured(&mut self, k: usize, rng: &mut crate::rng::Prng) -> Result<Vec<bool>> {
        let mut outcomes = Vec::with_capacity(k);
        for _ in 0..k {
            let q = self.n - 1;
            let outcome = Backend::measure(self, q, rng)?;
            if outcome {
                let mut x = GateMatrix::<S>::zeros(2)?;
                x.set(0, 1, S::one());
                x.set(1, 0, S::one());
                self.apply_via_tree(&[q], |amps, local| {
                    super::apply_single_in_place(amps, &x, local[0])
                })?;
            }
            self.release(1)?;
            outcomes.push(outcome);
        }
        Ok(outcomes)
    }

    /// Measured weight qubit `q` carries outside `|0⟩`, from the
    /// environment-transported Gram factor: `w₁ / (w₀ + w₁)`.
    fn boundary_leakage(&self, q: usize) -> Result<f64> {
        let (x, xr, leaf) = self.leaf_env(q)?;
        let mut w = [0.0f64; 2];
        for (p, slot) in w.iter_mut().enumerate() {
            for k in 0..xr {
                let mut acc = S::zero();
                for u in 0..leaf.bond {
                    acc = acc + x[k * leaf.bond + u] * leaf.tensor[u * 2 + p];
                }
                *slot += acc.abs_sqr();
            }
        }
        let total = w[0] + w[1];
        if total <= 0.0 || !total.is_finite() {
            return Err(Error::InvalidState(format!(
                "release: total weight {total} is not positive"
            )));
        }
        Ok(w[1] / total)
    }

    /// Environment Gram factor transported from `top` down to leaf `q`.
    fn leaf_env(&self, q: usize) -> Result<(Vec<S>, usize, &Node<S>)> {
        let mut node = &self.root;
        let mut x: Vec<S> = self.top.clone();
        let mut xr = 1usize;
        while let Some((l, r)) = &node.children {
            let take_left = q < l.hi;
            let (child, sibling_first) = if take_left { (l, true) } else { (r, false) };
            let (ca, cb) = (l.bond, r.bond);
            let cols = child.bond;
            let rows = xr * if sibling_first { cb } else { ca };
            // C[(k, sib)][child] = Σ_u x[k][u] W[u][a][b]
            let mut c = vec![S::zero(); rows * cols];
            for k in 0..xr {
                for u in 0..node.bond {
                    let xv = x[k * node.bond + u];
                    if xv.abs_sqr() == 0.0 {
                        continue;
                    }
                    for a in 0..ca {
                        for b in 0..cb {
                            let w = node.tensor[(u * ca + a) * cb + b];
                            if w.abs_sqr() == 0.0 {
                                continue;
                            }
                            let (row, col) = if take_left {
                                (k * cb + b, a)
                            } else {
                                (k * ca + a, b)
                            };
                            c[row * cols + col] = c[row * cols + col] + xv * w;
                        }
                    }
                }
            }
            let (nx, nxr) = compress_env(rows, cols, &c)?;
            x = nx;
            xr = nxr;
            node = child;
        }
        if node.lo != q {
            return Err(Error::QubitOutOfRange {
                qubit: q,
                num_qubits: self.n,
            });
        }
        Ok((x, xr, node))
    }

    /// Reset leaf `q` (already verified `bond = 1`, `|0⟩`-supported) to
    /// the pristine tensor, folding its unit phase into the parent.
    fn pristinize_leaf(&mut self, q: usize) -> Result<()> {
        fn go<S: Scalar>(node: &mut Node<S>, q: usize) -> Result<()> {
            let (l, r) = node
                .children
                .as_mut()
                .ok_or_else(|| Error::InvalidState("pristinize: no children".into()))?;
            let (child, is_left) = if q < l.hi { (l, true) } else { (r, false) };
            if child.children.is_some() {
                let (l3, r3) = node.children.as_mut().expect("checked");
                return go(if q < l3.hi { l3 } else { r3 }, q);
            }
            if child.bond != 1 {
                return Err(Error::InvalidState(format!(
                    "release: leaf {q} kept bond {} after projection",
                    child.bond
                )));
            }
            let phase = child.tensor[0];
            child.tensor = vec![S::one(), S::zero()];
            // Fold the unit factor into the parent combiner on this
            // child's (bond-1) slot.
            let (l4, r4) = node.children.as_ref().expect("checked");
            let (ca, cb) = (l4.bond, r4.bond);
            for u in 0..node.bond {
                for a in 0..ca {
                    for b in 0..cb {
                        let sel = if is_left { a == 0 } else { b == 0 };
                        if sel {
                            let idx = (u * ca + a) * cb + b;
                            node.tensor[idx] = node.tensor[idx] * phase;
                        }
                    }
                }
            }
            Ok(())
        }
        go(&mut self.root, q)
    }

    // ── the weighted rebuild machinery ──────────────────────────────

    /// Rank after the configured cap and relative tolerance; dropped σ²
    /// are one ledger event at `depth`.
    fn truncated_rank(
        sigma: &[f64],
        config: &BulkConfig,
        ledger: &mut Ledger,
        depth: usize,
    ) -> usize {
        let smax = sigma.first().copied().unwrap_or(0.0);
        let mut rank = 0usize;
        let mut dropped = 0.0;
        for (i, &s) in sigma.iter().enumerate() {
            if i < config.max_bond && s > smax * config.trunc_tol {
                rank = i + 1;
            } else {
                dropped += s * s;
            }
        }
        ledger.add(depth, dropped);
        rank.max(1)
    }

    /// Rebuild the subtree over `[lo, hi)` from a live dense block
    /// `[bond][2^w_live]` by environment-weighted Schmidt splits.
    /// `x` (`xr × bond`) is the environment Gram factor (`x†x = ρ`).
    /// Returns the new subtree (all tensors isometric) plus the
    /// absorb factor `m` (`bond × new_bond`) for the caller.
    #[allow(clippy::too_many_arguments)]
    fn build(
        lo: usize,
        hi: usize,
        bond: usize,
        block: Vec<S>,
        x: &[S],
        xr: usize,
        n: usize,
        config: &BulkConfig,
        ledger: &mut Ledger,
        depth: usize,
    ) -> Result<(Node<S>, Vec<S>, usize)> {
        if hi - lo == 1 {
            let (vt, m, rank) = isometrize(bond, 2, &block)?;
            return Ok((
                Node {
                    lo,
                    hi,
                    bond: rank,
                    tensor: vt,
                    children: None,
                },
                m,
                rank,
            ));
        }
        let mid = lo + (hi - lo).div_ceil(2);
        let live_hi = hi.min(n);
        if mid >= live_hi {
            // The right half is entirely dormant: pass the block through
            // to the left child unchanged and keep a pristine right.
            let (lnode, ml, rbl) =
                Self::build(lo, mid, bond, block, x, xr, n, config, ledger, depth + 1)?;
            // W[u][a][0] = ml[u][a], then isometrize.
            let (vt, m, rank) = isometrize(bond, rbl, &ml)?;
            return Ok((
                Node {
                    lo,
                    hi,
                    bond: rank,
                    tensor: vt,
                    children: Some((Box::new(lnode), Box::new(Node::pristine(mid, hi)))),
                },
                m,
                rank,
            ));
        }
        let (wl, wr) = (mid - lo, live_hi - mid);
        let (dl, dr) = (1usize << wl, 1usize << wr);
        let w = wl + wr;

        // Weighted block B̃[k][p] = Σ_u x[k][u]·B[u][p]: the state the
        // splits actually optimize.
        crate::guard::checkpoint()?;
        let mut bt = crate::guard::try_vec(xr << w, S::zero(), "bulk weighted block")?;
        for k in 0..xr {
            for u in 0..bond {
                let xv = x[k * bond + u];
                if xv.abs_sqr() == 0.0 {
                    continue;
                }
                let src = &block[u << w..(u + 1) << w];
                let dst = &mut bt[k << w..(k + 1) << w];
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d = *d + xv * s;
                }
            }
        }

        // Left split on the weighted state: M1[pl][(k, pr)].
        let cols1 = xr * dr;
        let mut m1 = crate::guard::try_vec(dl * cols1, S::zero(), "bulk left split")?;
        for k in 0..xr {
            for pr in 0..dr {
                for pl in 0..dl {
                    m1[pl * cols1 + k * dr + pr] = bt[(k << w) | (pr << wl) | pl];
                }
            }
        }
        let svd1 = svd_thin(dl, cols1, &m1, 0.0, 0.0)?;
        let r1 = Self::truncated_rank(&svd1.sigma, config, ledger, depth + 1).min(svd1.rank.max(1));
        let mut al = vec![S::zero(); r1 * dl];
        for pl in 0..dl {
            for a in 0..r1.min(svd1.rank) {
                al[a * dl + pl] = svd1.u[pl * svd1.rank + a];
            }
        }
        if svd1.rank == 0 {
            al[0] = S::one();
        }

        // Exact remainder in the original up-index:
        // RB[u][(a, pr)] = Σ_pl conj(AL[a][pl])·B[u][(pr<<wl)|pl].
        let mut rb = crate::guard::try_vec(bond * r1 * dr, S::zero(), "bulk remainder")?;
        for u in 0..bond {
            for a in 0..r1 {
                for pr in 0..dr {
                    let mut acc = S::zero();
                    for pl in 0..dl {
                        acc = acc + al[a * dl + pl].conj() * block[(u << w) | (pr << wl) | pl];
                    }
                    rb[(u * r1 + a) * dr + pr] = acc;
                }
            }
        }
        // Weighted remainder R̃[k][(a, pr)] = Σ_pl conj(AL)·B̃.
        let mut rt = crate::guard::try_vec(xr * r1 * dr, S::zero(), "bulk weighted remainder")?;
        for k in 0..xr {
            for a in 0..r1 {
                for pr in 0..dr {
                    let mut acc = S::zero();
                    for pl in 0..dl {
                        acc = acc + al[a * dl + pl].conj() * bt[(k << w) | (pr << wl) | pl];
                    }
                    rt[(k * r1 + a) * dr + pr] = acc;
                }
            }
        }

        // Right split on the weighted remainder: M2[pr][(k, a)].
        let cols2 = xr * r1;
        let mut m2 = crate::guard::try_vec(dr * cols2, S::zero(), "bulk right split")?;
        for k in 0..xr {
            for a in 0..r1 {
                for pr in 0..dr {
                    m2[pr * cols2 + k * r1 + a] = rt[(k * r1 + a) * dr + pr];
                }
            }
        }
        let svd2 = svd_thin(dr, cols2, &m2, 0.0, 0.0)?;
        let r2 = Self::truncated_rank(&svd2.sigma, config, ledger, depth + 1).min(svd2.rank.max(1));
        let mut ar = vec![S::zero(); r2 * dr];
        for pr in 0..dr {
            for b in 0..r2.min(svd2.rank) {
                ar[b * dr + pr] = svd2.u[pr * svd2.rank + b];
            }
        }
        if svd2.rank == 0 {
            ar[0] = S::one();
        }

        // Combiner in the original up-index:
        // W[u][a][b] = Σ_pr conj(AR[b][pr])·RB[u][(a, pr)].
        let mut wt = vec![S::zero(); bond * r1 * r2];
        for u in 0..bond {
            for a in 0..r1 {
                for b in 0..r2 {
                    let mut acc = S::zero();
                    for pr in 0..dr {
                        acc = acc + ar[b * dr + pr].conj() * rb[(u * r1 + a) * dr + pr];
                    }
                    wt[(u * r1 + a) * r2 + b] = acc;
                }
            }
        }

        // Children environments: C_L[(k,b)][a] = Σ_u x[k][u]·W[u][a][b]
        // (and mirrored for the right child), row-compressed so the
        // factor never grows.
        let mut cl = vec![S::zero(); xr * r2 * r1];
        let mut cr = vec![S::zero(); xr * r1 * r2];
        for k in 0..xr {
            for u in 0..bond {
                let xv = x[k * bond + u];
                if xv.abs_sqr() == 0.0 {
                    continue;
                }
                for a in 0..r1 {
                    for b in 0..r2 {
                        let wv = wt[(u * r1 + a) * r2 + b];
                        if wv.abs_sqr() == 0.0 {
                            continue;
                        }
                        let v = xv * wv;
                        cl[(k * r2 + b) * r1 + a] = cl[(k * r2 + b) * r1 + a] + v;
                        cr[(k * r1 + a) * r2 + b] = cr[(k * r1 + a) * r2 + b] + v;
                    }
                }
            }
        }
        let (xl, xlr) = compress_env(xr * r2, r1, &cl)?;
        let (xrv, xrr) = compress_env(xr * r1, r2, &cr)?;

        let (lnode, ml, rbl) =
            Self::build(lo, mid, r1, al, &xl, xlr, n, config, ledger, depth + 1)?;
        let (rnode, mr, rbr) =
            Self::build(mid, hi, r2, ar, &xrv, xrr, n, config, ledger, depth + 1)?;

        // Absorb the children's exact factors, then isometrize.
        let mut wt2 = vec![S::zero(); bond * rbl * rbr];
        for u in 0..bond {
            for a in 0..r1 {
                for b in 0..r2 {
                    let wv = wt[(u * r1 + a) * r2 + b];
                    if wv.abs_sqr() == 0.0 {
                        continue;
                    }
                    for a2 in 0..rbl {
                        let mla = ml[a * rbl + a2];
                        if mla.abs_sqr() == 0.0 {
                            continue;
                        }
                        for b2 in 0..rbr {
                            let idx = (u * rbl + a2) * rbr + b2;
                            wt2[idx] = wt2[idx] + wv * mla * mr[b * rbr + b2];
                        }
                    }
                }
            }
        }
        let (vt, m, rank) = isometrize(bond, rbl * rbr, &wt2)?;
        Ok((
            Node {
                lo,
                hi,
                bond: rank,
                tensor: vt,
                children: Some((Box::new(lnode), Box::new(rnode))),
            },
            m,
            rank,
        ))
    }

    /// Descend to the smallest live subtree spanning the gate,
    /// transporting the environment factor; contract, apply,
    /// weighted-rebuild; re-isometrize the path on unwind and fold the
    /// final factor into `top`.
    fn apply_via_tree(
        &mut self,
        qubits: &[usize],
        apply: impl Fn(&mut [S], &[usize]) -> Result<()>,
    ) -> Result<()> {
        let (lo, hi) = (
            *qubits.iter().min().expect("validated nonempty"),
            *qubits.iter().max().expect("validated nonempty") + 1,
        );
        #[allow(clippy::too_many_arguments)]
        fn go<S: Scalar, F: Fn(&mut [S], &[usize]) -> Result<()>>(
            node: &mut Node<S>,
            lo: usize,
            hi: usize,
            qubits: &[usize],
            apply: &F,
            x: &[S],
            xr: usize,
            n: usize,
            config: &BulkConfig,
            ledger: &mut Ledger,
            peak: &mut usize,
            depth: usize,
        ) -> Result<(Vec<S>, usize)> {
            let fits_child = match &node.children {
                None => None,
                Some((l, _)) => {
                    if hi <= l.hi {
                        Some(true)
                    } else if lo >= l.hi {
                        Some(false)
                    } else {
                        None
                    }
                }
            };
            if let Some(take_left) = fits_child {
                // Transport the environment one level down.
                let (ca, cb) = {
                    let (l, r) = node.children.as_ref().expect("internal");
                    (l.bond, r.bond)
                };
                let cols = if take_left { ca } else { cb };
                let rows = xr * if take_left { cb } else { ca };
                let mut c = vec![S::zero(); rows * cols];
                for k in 0..xr {
                    for u in 0..node.bond {
                        let xv = x[k * node.bond + u];
                        if xv.abs_sqr() == 0.0 {
                            continue;
                        }
                        for a in 0..ca {
                            for b in 0..cb {
                                let wv = node.tensor[(u * ca + a) * cb + b];
                                if wv.abs_sqr() == 0.0 {
                                    continue;
                                }
                                let (row, col) = if take_left {
                                    (k * cb + b, a)
                                } else {
                                    (k * ca + a, b)
                                };
                                c[row * cols + col] = c[row * cols + col] + xv * wv;
                            }
                        }
                    }
                }
                let (cx, cxr) = compress_env(rows, cols, &c)?;
                let (l, r) = node.children.as_mut().expect("internal");
                let child = if take_left { l } else { r };
                let (m, new_child_bond) = go(
                    child,
                    lo,
                    hi,
                    qubits,
                    apply,
                    &cx,
                    cxr,
                    n,
                    config,
                    ledger,
                    peak,
                    depth + 1,
                )?;
                // Absorb the child's factor into this combiner slot.
                let (ca2, cb2) = {
                    let (l2, r2) = node.children.as_ref().expect("internal");
                    (l2.bond, r2.bond)
                };
                let (old_ca, old_cb) = if take_left {
                    (m.len() / new_child_bond.max(1), cb2)
                } else {
                    (ca2, m.len() / new_child_bond.max(1))
                };
                let mut wt = vec![S::zero(); node.bond * ca2 * cb2];
                for u in 0..node.bond {
                    for a in 0..ca2 {
                        for b in 0..cb2 {
                            let mut acc = S::zero();
                            if take_left {
                                for a0 in 0..old_ca {
                                    let wv = node.tensor[(u * old_ca + a0) * cb2 + b];
                                    if wv.abs_sqr() == 0.0 {
                                        continue;
                                    }
                                    acc = acc + wv * m[a0 * ca2 + a];
                                }
                            } else {
                                for b0 in 0..old_cb {
                                    let wv = node.tensor[(u * ca2 + a) * old_cb + b0];
                                    if wv.abs_sqr() == 0.0 {
                                        continue;
                                    }
                                    acc = acc + wv * m[b0 * cb2 + b];
                                }
                            }
                            wt[(u * ca2 + a) * cb2 + b] = acc;
                        }
                    }
                }
                let (vt, m_up, rank) = isometrize(node.bond, ca2 * cb2, &wt)?;
                let old_bond = node.bond;
                node.tensor = vt;
                node.bond = rank;
                debug_assert_eq!(m_up.len(), old_bond * rank);
                return Ok((m_up, rank));
            }
            // This node is the LCA: materialize the live block.
            let w = node.live_width(n);
            if w > config.max_block {
                return Err(Error::TooManyQubits {
                    requested: w,
                    max: config.max_block,
                });
            }
            let mut block = node.contract_live(n)?;
            *peak = (*peak).max(block.len());
            let local: Vec<usize> = qubits.iter().map(|&q| q - node.lo).collect();
            let d = 1usize << w;
            for u in 0..node.bond {
                apply(&mut block[u * d..(u + 1) * d], &local)?;
            }
            let (nlo, nhi, nbond) = (node.lo, node.hi, node.bond);
            let (rebuilt, m_up, rank) =
                BulkState::<S>::build(nlo, nhi, nbond, block, x, xr, n, config, ledger, depth)?;
            *node = rebuilt;
            Ok((m_up, rank))
        }

        let config = self.config;
        let mut ledger = std::mem::take(&mut self.ledger);
        let mut peak = self.peak_block_elements;
        let x: Vec<S> = self.top.clone();
        let result = go(
            &mut self.root,
            lo,
            hi,
            qubits,
            &apply,
            &x,
            1,
            self.n,
            &config,
            &mut ledger,
            &mut peak,
            0,
        );
        self.ledger = ledger;
        self.peak_block_elements = peak;
        let (m, rank) = result?;
        // Fold the root factor into the bulk record.
        let old = self.top.clone();
        debug_assert_eq!(m.len(), old.len() * rank);
        let mut top = vec![S::zero(); rank];
        for (u, &t) in old.iter().enumerate() {
            if t.abs_sqr() == 0.0 {
                continue;
            }
            for (k, slot) in top.iter_mut().enumerate() {
                *slot = *slot + t * m[u * rank + k];
            }
        }
        self.top = top;
        Ok(())
    }
}

// ── the structured unfold ───────────────────────────────────────────

/// One dilated level step of an [`UnfoldProgram`]: a unitary on
/// `qubits` (input channel wires first, then the wires injected fresh
/// at this step), realizing one node's isometry by Stinespring
/// dilation.
#[derive(Debug, Clone)]
pub struct UnfoldStep<S: Scalar> {
    /// Depth of the node this step expands (root = 0).
    pub depth: usize,
    /// Boundary span `[lo, hi)` of the node.
    pub span: (usize, usize),
    /// Wires the unitary acts on (matrix sub-index bit `b` ↔
    /// `qubits[b]`).
    pub qubits: Vec<usize>,
    /// The wires in `qubits` that must be `|0⟩` before this step —
    /// the dimension injected at this level.
    pub injected: Vec<usize>,
    /// The dilated unitary.
    pub matrix: GateMatrix<S>,
}

/// The depth store compiled into an explicit width-growing circuit:
/// a seed state on the bulk channel plus one dilated unitary per node
/// in depth order. Replaying it reproduces the boundary state exactly;
/// stopping after depth `< ℓ` reproduces the coarse state at depth `ℓ`
/// on the channel wires. See [`BulkState::unfold_program`].
#[derive(Debug, Clone)]
pub struct UnfoldProgram<S: Scalar> {
    /// Boundary width the program unfolds to.
    pub num_qubits: usize,
    /// Wires carrying the bulk record initially (positions `0…c`).
    pub seed_qubits: Vec<usize>,
    /// The bulk record as seed amplitudes over `seed_qubits`
    /// (little-endian; padded with zeros to the next power of two).
    pub seed: Vec<S>,
    /// Level steps in depth order.
    pub steps: Vec<UnfoldStep<S>>,
}

impl<S: Scalar> UnfoldProgram<S> {
    /// Replay the whole program on `state` (must have width
    /// `num_qubits`; its current contents are replaced).
    pub fn run_on(&self, state: &mut dyn Backend<S>) -> Result<()> {
        self.run_to_depth(state, usize::MAX)
    }

    /// Replay the seed and every step of node-depth `< depth`. The
    /// resulting state is the coarse state at `depth` on the channel
    /// wires, all other wires exactly `|0⟩`.
    pub fn run_to_depth(&self, state: &mut dyn Backend<S>, depth: usize) -> Result<()> {
        if state.num_qubits() != self.num_qubits {
            return Err(Error::BadDimension {
                expected: self.num_qubits,
                got: state.num_qubits(),
            });
        }
        let entries: Vec<(u64, S)> = self
            .seed
            .iter()
            .enumerate()
            .filter(|(_, a)| a.abs_sqr() > 0.0)
            .map(|(i, &a)| (i as u64, a))
            .collect();
        state.load(&entries)?;
        for step in self.steps.iter().filter(|s| s.depth < depth) {
            state.apply(&step.matrix, &step.qubits)?;
        }
        Ok(())
    }

    /// The structural interaction sequence of one boundary wire: every
    /// `(depth, span)` step it participates in, in order. A wire is
    /// entangled **only** through this sequence — its first entry is
    /// where the wire is injected (or the seed), and each later entry
    /// couples it exactly to its node's span.
    pub fn sequence(&self, qubit: usize) -> Vec<(usize, (usize, usize))> {
        self.steps
            .iter()
            .filter(|s| s.qubits.contains(&qubit))
            .map(|s| (s.depth, s.span))
            .collect()
    }

    /// Depth at which a wire enters the computation: 0 for seed wires,
    /// otherwise the depth of the step that injects it. `None` for a
    /// wire that never participates (a pristine `|0⟩` boundary qubit).
    pub fn injection_depth(&self, qubit: usize) -> Option<usize> {
        if self.seed_qubits.contains(&qubit) {
            return Some(0);
        }
        self.steps
            .iter()
            .find(|s| s.injected.contains(&qubit))
            .map(|s| s.depth)
    }

    /// Widest step in the program (qubits per dilated unitary).
    pub fn max_step_qubits(&self) -> usize {
        self.steps.iter().map(|s| s.qubits.len()).max().unwrap_or(0)
    }
}

/// Complete a set of orthonormal columns to a full `d × d` unitary by
/// Gram–Schmidt over the standard basis.
fn complete_unitary<S: Scalar>(d: usize, cols: &[(usize, Vec<S>)]) -> Result<GateMatrix<S>> {
    let mut m = GateMatrix::<S>::zeros(d)?;
    let mut have: Vec<Vec<S>> = Vec::with_capacity(d);
    let mut placed = vec![false; d];
    for (j, col) in cols {
        for (i, &v) in col.iter().enumerate() {
            m.set(i, *j, v);
        }
        have.push(col.clone());
        placed[*j] = true;
    }
    let mut cursor = 0usize;
    for (j, placed_j) in placed.iter_mut().enumerate() {
        if *placed_j {
            continue;
        }
        // Next standard basis vector with enough residual.
        let mut chosen: Option<Vec<S>> = None;
        while cursor < d {
            let mut v = vec![S::zero(); d];
            v[cursor] = S::one();
            cursor += 1;
            for h in &have {
                let mut ip = S::zero();
                for (hv, vv) in h.iter().zip(v.iter()) {
                    ip = ip + hv.conj() * *vv;
                }
                if ip.abs_sqr() > 0.0 {
                    for (vv, hv) in v.iter_mut().zip(h.iter()) {
                        *vv = *vv - ip * *hv;
                    }
                }
            }
            let norm: f64 = v.iter().map(|a| a.abs_sqr()).sum();
            if norm > 1e-6 {
                let scale = 1.0 / norm.sqrt();
                for vv in v.iter_mut() {
                    *vv = vv.scale(scale);
                }
                chosen = Some(v);
                break;
            }
        }
        let v = chosen.ok_or_else(|| {
            Error::InvalidState("unfold: unitary completion ran out of basis vectors".into())
        })?;
        for (i, &vv) in v.iter().enumerate() {
            m.set(i, j, vv);
        }
        have.push(v);
        *placed_j = true;
    }
    Ok(m)
}

impl<S: Scalar> BulkState<S> {
    /// Compile the depth store into an [`UnfoldProgram`]: the bulk
    /// record as a seed on channel wires `0…⌈log₂ χ_root⌉`, then one
    /// dilated unitary per live node in depth order, each consuming
    /// fresh `|0⟩` wires. Errs when any step would exceed
    /// [`UNFOLD_MAX_STEP_QUBITS`] wires.
    pub fn unfold_program(&self) -> Result<UnfoldProgram<S>> {
        let cu_root = clog2(self.top.len().max(1));
        let mut seed = vec![S::zero(); 1 << cu_root];
        seed[..self.top.len()].copy_from_slice(&self.top);
        let mut steps: Vec<UnfoldStep<S>> = Vec::new();
        self.unfold_node(&self.root, 0, &mut steps)?;
        steps.sort_by_key(|s| s.depth);
        Ok(UnfoldProgram {
            num_qubits: self.n,
            seed_qubits: (0..cu_root).collect(),
            seed,
            steps,
        })
    }

    fn unfold_node(
        &self,
        node: &Node<S>,
        depth: usize,
        steps: &mut Vec<UnfoldStep<S>>,
    ) -> Result<()> {
        if node.dormant(self.n) {
            return Ok(());
        }
        let cu = clog2(node.bond);
        match &node.children {
            None => {
                // Leaf: map the channel wire to the physical qubit.
                let d = 2usize;
                let mut cols = Vec::new();
                for u in 0..node.bond {
                    let col: Vec<S> = (0..d).map(|p| node.tensor[u * 2 + p]).collect();
                    cols.push((u, col));
                }
                let matrix = complete_unitary(d, &cols)?;
                steps.push(UnfoldStep {
                    depth,
                    span: (node.lo, node.hi.min(self.n)),
                    qubits: vec![node.lo],
                    injected: if cu == 0 { vec![node.lo] } else { vec![] },
                    matrix,
                });
                Ok(())
            }
            Some((l, r)) => {
                let (ca, cb) = (
                    clog2(if l.dormant(self.n) { 1 } else { l.bond }),
                    clog2(if r.dormant(self.n) { 1 } else { r.bond }),
                );
                let mid = node.mid();
                // Wire set: input channel [lo, lo+cu), output channels
                // [lo, lo+ca) and [mid, mid+cb); dedup and sort.
                let mut set: Vec<usize> = (node.lo..node.lo + cu.max(ca)).collect();
                set.extend(mid..mid + cb);
                set.sort_unstable();
                set.dedup();
                if set.len() > UNFOLD_MAX_STEP_QUBITS {
                    return Err(Error::TooManyQubits {
                        requested: set.len(),
                        max: UNFOLD_MAX_STEP_QUBITS,
                    });
                }
                if !set.is_empty() {
                    let pos = |q: usize| set.iter().position(|&p| p == q).expect("in set");
                    let d = 1usize << set.len();
                    let mut cols = Vec::new();
                    let lb = if l.dormant(self.n) { 1 } else { l.bond };
                    let rb = if r.dormant(self.n) { 1 } else { r.bond };
                    for u in 0..node.bond {
                        // Input embedding: bit i of u sits on wire lo+i.
                        let mut input = 0usize;
                        for i in 0..cu {
                            if (u >> i) & 1 == 1 {
                                input |= 1 << pos(node.lo + i);
                            }
                        }
                        let mut col = vec![S::zero(); d];
                        for a in 0..lb {
                            for b in 0..rb {
                                let wv = node.tensor[(u * l.bond + a) * r.bond + b];
                                if wv.abs_sqr() == 0.0 {
                                    continue;
                                }
                                let mut out = 0usize;
                                for i in 0..ca {
                                    if (a >> i) & 1 == 1 {
                                        out |= 1 << pos(node.lo + i);
                                    }
                                }
                                for i in 0..cb {
                                    if (b >> i) & 1 == 1 {
                                        out |= 1 << pos(mid + i);
                                    }
                                }
                                col[out] = wv;
                            }
                        }
                        cols.push((input, col));
                    }
                    let matrix = complete_unitary(d, &cols)?;
                    let injected: Vec<usize> = set
                        .iter()
                        .copied()
                        .filter(|&q| !(node.lo..node.lo + cu).contains(&q))
                        .collect();
                    steps.push(UnfoldStep {
                        depth,
                        span: (node.lo, node.hi.min(self.n)),
                        qubits: set,
                        injected,
                        matrix,
                    });
                }
                self.unfold_node(l, depth + 1, steps)?;
                self.unfold_node(r, depth + 1, steps)?;
                Ok(())
            }
        }
    }
}

impl<S: Scalar> Backend<S> for BulkState<S> {
    fn name(&self) -> &str {
        "bulk"
    }

    fn num_qubits(&self) -> usize {
        self.n
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        validate_apply(self.n, matrix, qubits)?;
        self.apply_via_tree(qubits, |amps, local| {
            if local.len() == 1 {
                super::apply_single_in_place(amps, matrix, local[0])
            } else {
                super::apply_general_in_place(amps, matrix, local)
            }
        })
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        validate_apply_diagonal(self.n, entries, qubits)?;
        self.apply_via_tree(qubits, |amps, local| {
            super::apply_diagonal_in_place(amps, entries, local)
        })
    }

    fn amplitude(&self, index: u64) -> S {
        let v = self.root.amplitude_vec(index, self.n);
        let mut acc = S::zero();
        for (u, &t) in self.top.iter().enumerate() {
            if t.abs_sqr() == 0.0 {
                continue;
            }
            acc = acc + t * v[u];
        }
        acc
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        for (index, v) in self.root.support(self.n) {
            let mut acc = S::zero();
            for (u, &t) in self.top.iter().enumerate() {
                if t.abs_sqr() == 0.0 {
                    continue;
                }
                acc = acc + t * v[u];
            }
            if acc.abs_sqr() > 0.0 {
                f(index, acc);
            }
        }
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        let mut m = GateMatrix::<S>::zeros(2).expect("2 is a power of two");
        let slot = usize::from(outcome);
        m.set(slot, slot, S::one().scale(renorm));
        let _ = self.apply_via_tree(&[qubit], |amps, local| {
            super::apply_single_in_place(amps, &m, local[0])
        });
    }

    fn reset(&mut self) {
        self.root = Node::pristine(0, self.capacity());
        self.top = vec![S::one()];
        self.ledger = Ledger::default();
        self.peak_block_elements = 1;
    }

    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        let n = self.n;
        let limit = 1u64 << n;
        let mut block = crate::guard::try_vec(1usize << n, S::zero(), "bulk load compilation")?;
        for &(i, a) in entries {
            if i >= limit {
                return Err(Error::QubitOutOfRange {
                    qubit: 64 - i.leading_zeros() as usize,
                    num_qubits: n,
                });
            }
            block[i as usize] = a;
        }
        let config = self.config;
        let mut ledger = std::mem::take(&mut self.ledger);
        let x = [S::one()];
        let (root, m, rank) = Self::build(
            0,
            self.capacity(),
            1,
            block,
            &x,
            1,
            n,
            &config,
            &mut ledger,
            0,
        )?;
        self.ledger = ledger;
        self.root = root;
        self.top = (0..rank).map(|k| m[k]).collect();
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        (self.root.tensor_elements() + self.top.len()) * std::mem::size_of::<S>()
            + std::mem::size_of::<Self>()
    }

    /// `‖top‖²` — exact by the gauge invariant, `O(χ)` at any width.
    fn total_abs_sqr(&self) -> f64 {
        self.top.iter().map(|a| a.abs_sqr()).sum()
    }

    /// Equal to [`total_abs_sqr`](Backend::total_abs_sqr) on the
    /// commutative division algebras this backend requires.
    fn total_weight(&self) -> f64 {
        self.top.iter().map(|a| a.born_weight()).sum()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::C64;

    #[test]
    fn pristine_tree_is_isometric_and_zero_state() {
        let s = BulkState::<C64>::new(5).unwrap();
        assert_eq!(s.num_qubits(), 5);
        assert_eq!(s.capacity(), 8);
        assert_eq!(s.amplitude(0), C64::new(1.0, 0.0));
        assert_eq!(s.top().len(), 1);
        assert!((s.total_abs_sqr() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn complete_unitary_fills_orthonormal_columns() {
        let cols = vec![(
            0usize,
            vec![
                C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            ],
        )];
        let m = complete_unitary(2, &cols).unwrap();
        // U†U = I.
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = C64::new(0.0, 0.0);
                for k in 0..2 {
                    acc += m.get(k, i).conj() * m.get(k, j);
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((acc.re - expect).abs() < 1e-12 && acc.im.abs() < 1e-12);
            }
        }
    }
}
