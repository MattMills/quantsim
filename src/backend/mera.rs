//! Hierarchical (MERA-family) register backend: the state as a
//! renormalization hierarchy, with coarse views at every scale.
//!
//! A [`MeraState`] stores the register as a balanced binary tree of
//! tensors — the isometry layer of the MERA family (a tree tensor
//! network; the disentangler layer is the named next rung, see the
//! roadmap). Each node coarse-grains its two children into one
//! **super-site** whose bond dimension is capped at
//! [`MeraConfig::max_bond`]: physically, the tree *is* the
//! coarse-graining structure of a renormalization group — qubits at the
//! leaves, effective degrees of freedom at every internal level, the
//! whole register summarized at the root.
//!
//! What the hierarchy buys:
//!
//! * **Coarse views of the gross register** — [`MeraState::coarse_state`]
//!   returns the exact state *on the super-site basis at any depth*: a
//!   16-qubit GHZ at depth 1 is literally a two-super-site maximally
//!   entangled pair (asserted in the tests). [`MeraState::refine_basis`]
//!   returns the dictionary that expands one super-site one level — the
//!   refinement primitive: evaluate coarse first, descend on demand,
//!   down to physical amplitudes.
//! * **Hierarchy-local memory** — a gate's cost is the *smallest subtree
//!   spanning its targets*: gates inside a block never touch the rest of
//!   the tree, and bonds carry exactly the entanglement each cut needs
//!   (GHZ: bond 2 at every level). Persistent memory is the sum of
//!   tensor sizes.
//! * **Measured resolution** — truncation to `max_bond` accumulates the
//!   discarded singular-value weight ([`MeraState::discarded_weight`]);
//!   zero discarded means the representation is exact, and the
//!   conformance suite holds the default configuration to reference
//!   tolerance. Combined with the [`Ball`](crate::scalar::Ball) scalar
//!   the numbers themselves carry certified resolution through the SVDs.
//!
//! Rung-1 scope, stated plainly: a gate spanning both halves of a
//! subtree **materializes that subtree's dense block** (bounded by
//! [`MeraConfig::max_block`], error past it), applies the gate exactly,
//! and re-compresses by recursive Schmidt splits — honest transient
//! cost, instrumented via [`MeraState::peak_block_elements`], exactly
//! the degradation mode of the factored backend. Removing it (rank-≤4
//! operator splits + hierarchical rounding along the tree path, then
//! disentanglers, then ascending superoperators so *operators* renormalize
//! instead of blocks) is the MERA completion on the roadmap; truncation
//! optimality is block-local pending gauge maintenance, like the MPS
//! backend's.

use super::{validate_apply, validate_apply_diagonal, Backend};
use crate::error::{Error, Result};
use crate::math::{svd_thin, GateMatrix};
use crate::scalar::Scalar;

/// Maximum register width (basis indices are `u64`).
pub const MERA_MAX_QUBITS: usize = 63;

/// Structural width bound for [`Backend::load`] (basis indices fit
/// `u64`); the exponential compilation buffer itself is admitted by the
/// [resource guard](crate::guard) against measured memory.
pub const MERA_LOAD_MAX_QUBITS: usize = 63;

/// Configuration for the hierarchical backend.
#[derive(Debug, Clone, Copy)]
pub struct MeraConfig {
    /// Hard cap on any super-site bond dimension; excess Schmidt weight
    /// is truncated and accumulated in the discarded ledger.
    pub max_bond: usize,
    /// Relative singular-value cutoff at each split.
    pub trunc_tol: f64,
    /// Widest subtree a single gate may force into a dense block — a
    /// *policy* cap for callers who want structural refusal below the
    /// machine's real limits. The default is the structural bound: the
    /// transient `2^block` buffer is admitted by the [resource
    /// guard](crate::guard) against measured memory, and its SVD
    /// re-compression is bounded by the guard's time budget when one is
    /// armed.
    pub max_block: usize,
}

impl Default for MeraConfig {
    fn default() -> Self {
        MeraConfig {
            max_bond: 64,
            trunc_tol: 1e-12,
            max_block: 63,
        }
    }
}

/// One node of the hierarchy over the qubit range `[lo, hi)`.
///
/// Semantics: the subtree is a map `A[u][p]` from its up-bond `u`
/// (dimension `bond`) to amplitudes over its physical range `p`
/// (little-endian: bit `i` of `p` is qubit `lo + i`). A leaf stores
/// `A` directly (`[bond][2]`); an internal node stores the combiner
/// `W[u][a][b]` with `A[u][(pr << wl) | pl] = Σ W[u][a][b]·L[a][pl]·R[b][pr]`.
#[derive(Debug, Clone)]
struct Node<S: Scalar> {
    lo: usize,
    hi: usize,
    bond: usize,
    tensor: Vec<S>,
    children: Children<S>,
}

/// The two subtrees of an internal node (`None` for a leaf).
type Children<S> = Option<(Box<Node<S>>, Box<Node<S>>)>;

impl<S: Scalar> Node<S> {
    fn width(&self) -> usize {
        self.hi - self.lo
    }

    /// Fresh `|0…⟩` subtree (all bonds 1).
    fn fresh(lo: usize, hi: usize) -> Self {
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
                Box::new(Node::fresh(lo, mid)),
                Box::new(Node::fresh(mid, hi)),
            )),
        }
    }

    /// Dense block `[bond][2^width]` of this subtree's map.
    fn contract(&self) -> crate::error::Result<Vec<S>> {
        match &self.children {
            None => Ok(self.tensor.clone()),
            Some((l, r)) => {
                let (wl, wr) = (l.width(), r.width());
                let (dl, dr) = (1usize << wl, 1usize << wr);
                let bl = l.contract()?;
                let br = r.contract()?;
                let mut out = crate::guard::try_vec(
                    self.bond << (wl + wr),
                    S::zero(),
                    "mera block contraction",
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
                                let rb = br[b * dr + pr];
                                if rb.abs_sqr() == 0.0 {
                                    continue;
                                }
                                let wrb = w * rb;
                                let base = (u << (wl + wr)) | (pr << wl);
                                for pl in 0..dl {
                                    let la = bl[a * dl + pl];
                                    if la.abs_sqr() == 0.0 {
                                        continue;
                                    }
                                    out[base | pl] = out[base | pl] + wrb * la;
                                }
                            }
                        }
                    }
                }
                Ok(out)
            }
        }
    }

    /// Amplitude column of this subtree at a basis assignment: the
    /// `bond`-vector `A[·][p]` for the bits of `index` in `[lo, hi)`.
    fn amplitude_vec(&self, index: u64) -> Vec<S> {
        match &self.children {
            None => {
                let bit = ((index >> self.lo) & 1) as usize;
                (0..self.bond).map(|u| self.tensor[u * 2 + bit]).collect()
            }
            Some((l, r)) => {
                let lv = l.amplitude_vec(index);
                let rv = r.amplitude_vec(index);
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

    /// Depth-first support walk: `(relative index, bond-vector)` pairs
    /// with all-zero vectors pruned. Exponential for generic states
    /// (documented trait behavior), linear-ish for concentrated ones.
    fn support(&self) -> Vec<(u64, Vec<S>)> {
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
                let wl = l.width();
                let ls = l.support();
                let rs = r.support();
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

    fn max_bond(&self) -> usize {
        let own = self.bond;
        match &self.children {
            None => own,
            Some((l, r)) => own.max(l.max_bond()).max(r.max_bond()),
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

    /// Nodes of the frontier at `depth` (a leaf shallower than `depth`
    /// is itself frontier), left to right.
    fn frontier(&self, depth: usize, out: &mut Vec<(usize, usize, usize)>) {
        // (lo, hi, bond) triples.
        if depth == 0 || self.children.is_none() {
            out.push((self.lo, self.hi, self.bond));
            return;
        }
        let (l, r) = self.children.as_ref().expect("checked");
        l.frontier(depth - 1, out);
        r.frontier(depth - 1, out);
    }

    /// Contraction of everything **above** the depth-`d` frontier: the
    /// coarse state over the frontier's bond product, little-endian in
    /// frontier order (first frontier node = least significant factor).
    fn coarse(&self, depth: usize) -> crate::error::Result<(Vec<usize>, Vec<S>)> {
        if depth == 0 || self.children.is_none() {
            // Identity over the own bond: the frontier index *is* u.
            let mut id = vec![S::zero(); self.bond * self.bond];
            for u in 0..self.bond {
                id[u * self.bond + u] = S::one();
            }
            return Ok((vec![self.bond], id));
        }
        let (l, r) = self.children.as_ref().expect("checked");
        let (ldims, lc) = l.coarse(depth - 1)?;
        let (rdims, rc) = r.coarse(depth - 1)?;
        let lprod: usize = ldims.iter().product();
        let rprod: usize = rdims.iter().product();
        let mut out =
            crate::guard::try_vec(self.bond * lprod * rprod, S::zero(), "mera coarse view")?;
        for u in 0..self.bond {
            for a in 0..l.bond {
                for b in 0..r.bond {
                    let w = self.tensor[(u * l.bond + a) * r.bond + b];
                    if w.abs_sqr() == 0.0 {
                        continue;
                    }
                    for jr in 0..rprod {
                        let rb = rc[b * rprod + jr];
                        if rb.abs_sqr() == 0.0 {
                            continue;
                        }
                        let wrb = w * rb;
                        let base = (u * rprod + jr) * lprod;
                        for jl in 0..lprod {
                            let la = lc[a * lprod + jl];
                            if la.abs_sqr() == 0.0 {
                                continue;
                            }
                            out[base + jl] = out[base + jl] + wrb * la;
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

/// The hierarchical register backend. See the module docs.
#[derive(Debug, Clone)]
pub struct MeraState<S: Scalar> {
    config: MeraConfig,
    root: Node<S>,
    discarded: f64,
    peak_block_elements: usize,
}

impl<S: Scalar> MeraState<S> {
    /// `|0…0⟩` with the default configuration.
    pub fn new(num_qubits: usize) -> Result<Self> {
        Self::with_config(num_qubits, MeraConfig::default())
    }

    /// `|0…0⟩` with an explicit configuration. Requires a commutative
    /// division algebra (the Jacobi SVD's rotations assume commuting
    /// scalars), like the MPS backend.
    pub fn with_config(num_qubits: usize, config: MeraConfig) -> Result<Self> {
        if !(S::COMMUTATIVE && S::DIVISION) {
            return Err(Error::InvalidState(format!(
                "mera requires a commutative division algebra; {} is not",
                S::algebra_name()
            )));
        }
        if num_qubits == 0 || num_qubits > MERA_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: num_qubits,
                max: MERA_MAX_QUBITS,
            });
        }
        if config.max_bond == 0 {
            return Err(Error::InvalidState("mera: max_bond must be ≥ 1".into()));
        }
        Ok(MeraState {
            config,
            root: Node::fresh(0, num_qubits),
            discarded: 0.0,
            peak_block_elements: 1,
        })
    }

    /// Total discarded singular-value weight (Σσ² over every truncation
    /// since construction/reset). Zero means the hierarchy is an exact
    /// representation of everything applied to it; nonzero is the
    /// measured resolution loss (block-local, pending gauge maintenance
    /// — see the module docs).
    pub fn discarded_weight(&self) -> f64 {
        self.discarded
    }

    /// Whether nothing above numerical noise has been truncated.
    pub fn is_exact(&self) -> bool {
        self.discarded < 1e-20
    }

    /// Largest dense block any single gate ever forced (elements).
    pub fn peak_block_elements(&self) -> usize {
        self.peak_block_elements
    }

    /// Largest bond dimension anywhere in the current hierarchy.
    pub fn max_bond_dimension(&self) -> usize {
        self.root.max_bond()
    }

    /// Depth of the hierarchy (root = 0; a perfect tree has
    /// `ceil(log2 n)` levels of super-sites above the leaves).
    pub fn depth(&self) -> usize {
        fn go<S: Scalar>(n: &Node<S>) -> usize {
            match &n.children {
                None => 0,
                Some((l, r)) => 1 + go(l).max(go(r)),
            }
        }
        go(&self.root)
    }

    /// The frontier at `depth`: for each super-site (left to right) its
    /// qubit range and bond dimension — the shape of the gross register
    /// at that resolution.
    pub fn coarse_dims(&self, depth: usize) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        self.root.frontier(depth, &mut out);
        out
    }

    /// The exact state expressed on the super-site basis at `depth`:
    /// `(per-site bond dimensions, coefficients)` with the coefficient
    /// index little-endian over frontier sites (first site's index
    /// varies fastest, mixed-radix by the returned dimensions). At
    /// depth 0 this is the trivial `[1]`; at leaf depth it is the full
    /// state. Errs when the coarse space exceeds 2^20 coefficients.
    pub fn coarse_state(&self, depth: usize) -> Result<(Vec<usize>, Vec<S>)> {
        let dims: Vec<usize> = self.coarse_dims(depth).iter().map(|&(_, _, b)| b).collect();
        let prod: usize = dims.iter().product();
        if prod > (1 << 20) {
            return Err(Error::InvalidState(format!(
                "coarse space at depth {depth} has {prod} coefficients (> 2^20)"
            )));
        }
        let (dims, coeffs) = self.root.coarse(depth)?;
        // Root bond is 1: drop the singleton u index.
        Ok((dims, coeffs))
    }

    /// The refinement dictionary of the `position`-th frontier node at
    /// `depth`: for an internal node, its combiner `W` as
    /// `(up_bond, left_bond, right_bond, tensor)` — how one super-site
    /// expands into its two children one level down; for a leaf, its
    /// `(bond, 2, 1, tensor)` map onto the physical qubit.
    pub fn refine_basis(
        &self,
        depth: usize,
        position: usize,
    ) -> Result<(usize, usize, usize, Vec<S>)> {
        fn find<'a, S: Scalar>(
            n: &'a Node<S>,
            depth: usize,
            pos: &mut usize,
        ) -> Option<&'a Node<S>> {
            if depth == 0 || n.children.is_none() {
                if *pos == 0 {
                    return Some(n);
                }
                *pos -= 1;
                return None;
            }
            let (l, r) = n.children.as_ref().expect("checked");
            if let Some(hit) = find(l, depth - 1, pos) {
                return Some(hit);
            }
            find(r, depth - 1, pos)
        }
        let mut p = position;
        let node = find(&self.root, depth, &mut p).ok_or_else(|| {
            Error::InvalidState(format!("no frontier node {position} at depth {depth}"))
        })?;
        match &node.children {
            None => Ok((node.bond, 2, 1, node.tensor.clone())),
            Some((l, r)) => Ok((node.bond, l.bond, r.bond, node.tensor.clone())),
        }
    }

    /// Rebuild the subtree over `[lo, hi)` from a dense block
    /// `[bond][2^width]` by recursive Schmidt splits, truncating at the
    /// configured cap and accumulating discarded weight.
    fn build(
        lo: usize,
        hi: usize,
        bond: usize,
        block: Vec<S>,
        config: &MeraConfig,
        discarded: &mut f64,
    ) -> Result<Node<S>> {
        let w = hi - lo;
        if w == 1 {
            return Ok(Node {
                lo,
                hi,
                bond,
                tensor: block,
                children: None,
            });
        }
        let mid = lo + w.div_ceil(2);
        let (wl, wr) = (mid - lo, hi - mid);
        let (dl, dr) = (1usize << wl, 1usize << wr);

        // Split off the left half: M[pl][(u, pr)].
        let cols1 = bond * dr;
        crate::guard::checkpoint()?;
        let mut m = crate::guard::try_vec(dl * cols1, S::zero(), "mera left split")?;
        for u in 0..bond {
            for pr in 0..dr {
                for pl in 0..dl {
                    m[pl * cols1 + u * dr + pr] = block[(u << w) | (pr << wl) | pl];
                }
            }
        }
        let svd1 = svd_thin(dl, cols1, &m, 0.0, 0.0)?;
        let r1 = Self::truncated_rank(&svd1.sigma, config, discarded);
        // Left child map AL[a][pl] and the remainder B[a][(u, pr)] = σ·V†.
        let mut al = vec![S::zero(); r1 * dl];
        for pl in 0..dl {
            for a in 0..r1 {
                al[a * dl + pl] = svd1.u[pl * svd1.rank + a];
            }
        }
        let mut bmat = vec![S::zero(); r1 * cols1];
        for a in 0..r1 {
            for j in 0..cols1 {
                bmat[a * cols1 + j] = svd1.vt[a * cols1 + j].scale(svd1.sigma[a]);
            }
        }

        // Split off the right half: N[pr][(u, a)].
        let cols2 = bond * r1;
        let mut n = crate::guard::try_vec(dr * cols2, S::zero(), "mera right split")?;
        for a in 0..r1 {
            for u in 0..bond {
                for pr in 0..dr {
                    n[pr * cols2 + u * r1 + a] = bmat[a * cols1 + u * dr + pr];
                }
            }
        }
        let svd2 = svd_thin(dr, cols2, &n, 0.0, 0.0)?;
        let r2 = Self::truncated_rank(&svd2.sigma, config, discarded);
        let mut ar = vec![S::zero(); r2 * dr];
        for pr in 0..dr {
            for b in 0..r2 {
                ar[b * dr + pr] = svd2.u[pr * svd2.rank + b];
            }
        }
        // Combiner W[u][a][b] = σ_b · vt2[b][(u, a)].
        let mut wt = vec![S::zero(); bond * r1 * r2];
        for u in 0..bond {
            for a in 0..r1 {
                for b in 0..r2 {
                    wt[(u * r1 + a) * r2 + b] =
                        svd2.vt[b * cols2 + u * r1 + a].scale(svd2.sigma[b]);
                }
            }
        }

        let left = Self::build(lo, mid, r1, al, config, discarded)?;
        let right = Self::build(mid, hi, r2, ar, config, discarded)?;
        Ok(Node {
            lo,
            hi,
            bond,
            tensor: wt,
            children: Some((Box::new(left), Box::new(right))),
        })
    }

    /// Rank after applying the configured cap and relative tolerance;
    /// dropped σ² feed the discarded ledger.
    fn truncated_rank(sigma: &[f64], config: &MeraConfig, discarded: &mut f64) -> usize {
        let smax = sigma.first().copied().unwrap_or(0.0);
        let mut rank = 0usize;
        for (i, &s) in sigma.iter().enumerate() {
            if i < config.max_bond && s > smax * config.trunc_tol {
                rank = i + 1;
            } else {
                *discarded += s * s;
            }
        }
        rank.max(1)
    }

    /// Smallest subtree spanning all target qubits, materialized, gate
    /// applied, re-compressed.
    fn apply_in_block(
        &mut self,
        qubits: &[usize],
        apply: impl Fn(&mut [S], &[usize]) -> crate::error::Result<()>,
    ) -> Result<()> {
        let (lo, hi) = (
            *qubits.iter().min().expect("validated nonempty"),
            *qubits.iter().max().expect("validated nonempty") + 1,
        );
        // Descend to the LCA node.
        fn lca<S: Scalar>(n: &mut Node<S>, lo: usize, hi: usize) -> &mut Node<S> {
            let (go_left, go_right) = match &n.children {
                None => (false, false),
                Some((l, _)) => {
                    let mid = l.hi;
                    (hi <= mid, lo >= mid)
                }
            };
            if go_left {
                let (l, _) = n.children.as_mut().expect("internal node");
                lca(l, lo, hi)
            } else if go_right {
                let (_, r) = n.children.as_mut().expect("internal node");
                lca(r, lo, hi)
            } else {
                n
            }
        }
        let config = self.config;
        let mut discarded = self.discarded;
        let mut peak = self.peak_block_elements;
        {
            let node = lca(&mut self.root, lo, hi);
            let w = node.width();
            if w > config.max_block {
                return Err(Error::TooManyQubits {
                    requested: w,
                    max: config.max_block,
                });
            }
            let mut block = node.contract()?;
            peak = peak.max(block.len());
            let local: Vec<usize> = qubits.iter().map(|&q| q - node.lo).collect();
            let d = 1usize << w;
            for u in 0..node.bond {
                apply(&mut block[u * d..(u + 1) * d], &local)?;
            }
            let (nlo, nhi, nbond) = (node.lo, node.hi, node.bond);
            let rebuilt = Self::build(nlo, nhi, nbond, block, &config, &mut discarded)?;
            *node = rebuilt;
        }
        self.discarded = discarded;
        self.peak_block_elements = peak;
        Ok(())
    }
}

impl<S: Scalar> Backend<S> for MeraState<S> {
    fn name(&self) -> &str {
        "mera"
    }

    fn num_qubits(&self) -> usize {
        self.root.hi
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        validate_apply(self.num_qubits(), matrix, qubits)?;
        self.apply_in_block(qubits, |amps, local| {
            if local.len() == 1 {
                super::apply_single_in_place(amps, matrix, local[0])
            } else {
                super::apply_general_in_place(amps, matrix, local)
            }
        })
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        validate_apply_diagonal(self.num_qubits(), entries, qubits)?;
        self.apply_in_block(qubits, |amps, local| {
            super::apply_diagonal_in_place(amps, entries, local)
        })
    }

    fn amplitude(&self, index: u64) -> S {
        self.root.amplitude_vec(index)[0]
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        for (index, v) in self.root.support() {
            if v[0].abs_sqr() > 0.0 {
                f(index, v[0]);
            }
        }
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        // A projector is a (non-unitary) 1-qubit kernel: leaf-local.
        let mut m = GateMatrix::<S>::zeros(2).expect("2 is a power of two");
        let slot = usize::from(outcome);
        m.set(slot, slot, S::one().scale(renorm));
        let _ = self.apply_in_block(&[qubit], |amps, local| {
            super::apply_single_in_place(amps, &m, local[0])
        });
    }

    fn reset(&mut self) {
        self.root = Node::fresh(0, self.num_qubits());
        self.discarded = 0.0;
        self.peak_block_elements = 1;
    }

    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        let n = self.num_qubits();
        if n > MERA_LOAD_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: n,
                max: MERA_LOAD_MAX_QUBITS,
            });
        }
        let limit = 1u64 << n;
        let mut block = crate::guard::try_vec(1usize << n, S::zero(), "mera load compilation")?;
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
        self.root = Self::build(0, n, 1, block, &config, &mut self.discarded)?;
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        self.root.tensor_elements() * std::mem::size_of::<S>() + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
