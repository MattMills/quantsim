//! Recursive quantum systems: a site that is either a point **or an
//! entire lattice of the same kind**, and a computation that
//! participates in itself.
//!
//! Every other register in this crate is *flat at the bottom*: a
//! backend addresses `n` two-level sites, a qudit register addresses
//! `m` sites of fixed arity, a MERA tree coarse-grains a fixed leaf
//! set. This module breaks the bottom out. A [`RecursiveLattice`] is a
//! [`Shape`] (four sites bonded in a square, say) whose sites are
//! [`Site`]s, and a site is *either* a single point *or another
//! lattice*. [`RecursiveLattice::refine`] turns one point into a
//! lattice; [`RecursiveLattice::nest`] does it uniformly, so the square
//! of four points becomes a square of four squares — sixteen qubits,
//! four internal squares, and four **lateral** bonds joining the blocks
//! corner to corner, exactly as the blocks' own bonds join their
//! points. The construction is scale-free: nothing in the bond rule
//! knows what depth it is at.
//!
//! That is the *structure*. The physics question is whether the
//! structure means anything — whether a lattice really can stand where
//! a point stood. Three measurements answer it, and they disagree in
//! an informative way:
//!
//! * **Ising blocks — approximately, with the error measured.**
//!   [`block_rg`] diagonalizes one block exactly, reads the effective
//!   field off its measured gap (`h' = Δ/2`) and the effective coupling
//!   off its measured boundary matrix element (`J' = J·μ²`, `μ =
//!   ⟨g₀|Z_port|g₁⟩`) — nothing is assumed, both numbers come out of
//!   the block's own spectrum. [`substitution_report`] then puts the
//!   claim on trial: it diagonalizes two blocks bonded laterally
//!   (the fine system) and two points at `(J', h')` (the coarse
//!   system) and reports the deviation between their *excitation
//!   spectra*. [`rg_fixed_point`] finds where the flow stands still —
//!   the coupling ratio at which the lattice is its own point — and
//!   linearizes there for the RG eigenvalue and correlation exponent.
//! * **Phonons — exactly, when the bonding is uniform.** A harmonic
//!   block's collective mode is the *exact* zero mode of its internal
//!   springs, so the block's lowest mode sits at the site frequency
//!   with no shift at all, and the external spring renormalizes by the
//!   port's measured participation. [`phonon_substitution`] measures
//!   both regimes: bonding the blocks through single corner ports
//!   leaves a finite, measured deviation (the lateral spring couples
//!   the collective coordinate to the internal modes), while bonding
//!   them uniformly makes the substitution exact to machine precision.
//!   The point-is-a-lattice duality is not a metaphor here; it is a
//!   number near 1e-16.
//! * **On the simulator, dynamically.** [`phonon_walk`] loads a single
//!   phonon into a *block's collective coordinate* on a real
//!   [`Backend`], evolves it with Trotterized `xy` hops through the
//!   recursive lattice's own bonds, and compares the coarse-grained
//!   amplitudes against the coarse lattice's exact evolution at the
//!   renormalized hop. Trotter error and substitution error are
//!   reported separately, and excitation-number leakage is measured
//!   rather than assumed.
//!
//! **The computation that participates in itself.**
//! [`self_participation`] closes the loop. A block is solved exactly in
//! the presence of a mean field; its own boundary magnetization — read
//! back off a loaded [`Backend`] through
//! [`pauli_expectation`], not off
//! the eigenvector — *is* the field its neighbours impose, so the
//! output of the computation is its input. The iteration is the system
//! being its own environment, and it either converges to a measured
//! fixed point or reports that it did not. Because the block is a
//! [`RecursiveLattice`], the same call works when the block is itself a
//! lattice of lattices: self-participation at every scale, with the
//! same code. [`participation_transition`] bisects the coupling at
//! which the self-consistent magnetization becomes nonzero — a
//! measured critical point produced entirely by a computation feeding
//! on itself.
//!
//! Honesty about what this is: the block-spin recursion is the
//! classical Kadanoff construction, and it is *approximate* by
//! construction — the measured deviation in [`SubstitutionReport`] is
//! the size of the approximation, reported rather than hidden. The
//! phonon substitution and the excitation-number conservation are
//! exact, and measured to be.

use crate::backend::{pauli_expectation, Backend, DenseState, Topology};
use crate::circuit::Circuit;
use crate::error::{Error, Result};
use crate::gates::Pauli;
use crate::math::{c64, GateMatrix};
use crate::registry::GateRegistry;
use crate::rng::Prng;
use crate::scalar::C64;

// ── the recursive structure ──────────────────────────────────────────

/// How the sites of one block are bonded to each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// `k` sites in a path: `0–1–…–(k−1)`.
    Chain(usize),
    /// `k` sites in a cycle. [`Shape::SQUARE`] is `Cycle(4)`.
    Cycle(usize),
    /// `k` sites, every pair bonded.
    Clique(usize),
    /// `2^dim` sites at the vertices of a `dim`-cube, bonded along
    /// edges. [`Shape::CUBE`] is `Hypercube(3)` — the 2×2×2 volume
    /// whose eight vertices are the eight coordinates of an E8 point
    /// (see [`e8::cube`](crate::e8::cube)), so a nested `CUBE` lattice
    /// is the qubit-register form of the E8 scale tower's cube of
    /// cubes.
    Hypercube(usize),
}

impl Shape {
    /// Four sites bonded in a square — the shape of the whole
    /// investigation.
    pub const SQUARE: Shape = Shape::Cycle(4);

    /// The 2×2×2 volume: eight sites, twelve edges.
    pub const CUBE: Shape = Shape::Hypercube(3);

    /// Number of sites in the block.
    pub fn arity(self) -> usize {
        match self {
            Shape::Chain(k) | Shape::Cycle(k) | Shape::Clique(k) => k,
            Shape::Hypercube(dim) => 1usize << dim,
        }
    }

    /// The block's bonds as site-slot pairs, `a < b`, sorted.
    pub fn bonds(self) -> Vec<(usize, usize)> {
        let k = self.arity();
        let mut out = Vec::new();
        match self {
            Shape::Chain(_) => {
                for i in 0..k.saturating_sub(1) {
                    out.push((i, i + 1));
                }
            }
            Shape::Cycle(_) => {
                for i in 0..k {
                    let j = (i + 1) % k;
                    out.push((i.min(j), i.max(j)));
                }
            }
            Shape::Clique(_) => {
                for i in 0..k {
                    for j in (i + 1)..k {
                        out.push((i, j));
                    }
                }
            }
            Shape::Hypercube(dim) => {
                for v in 0..k {
                    for axis in 0..dim {
                        let w = v ^ (1 << axis);
                        if v < w {
                            out.push((v, w));
                        }
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// **The dimensional lift**: one more axis.
    ///
    /// [`Shape::CUBE`] lifts to the tesseract, and a `dim`-cube to a
    /// `dim+1`-cube — the block doubles, every existing bond survives,
    /// and each old site gains exactly one partner. Only hypercubes lift:
    /// the operation is `G □ K₂` and a cycle or a clique lands outside
    /// its own family, so those refuse by name rather than returning
    /// something that is no longer the shape it claims to be.
    /// [`Topology::lift`](crate::backend::Topology::lift) is the same move
    /// on a register fabric, where it is defined for every topology.
    pub fn lift(self) -> Result<Shape> {
        match self {
            Shape::Hypercube(dim) => {
                let lifted = Shape::Hypercube(dim + 1);
                lifted.validate()?;
                Ok(lifted)
            }
            other => Err(Error::InvalidState(format!(
                "lift: {other:?} is not a hypercube; G □ K₂ leaves its family. \
                 Lift the register fabric instead (Topology::lift), which is \
                 defined for every topology."
            ))),
        }
    }

    /// Reject degenerate shapes: a block needs at least two sites, and
    /// a cycle at least three (two sites in a "cycle" is one bond
    /// counted twice).
    pub fn validate(self) -> Result<()> {
        let k = self.arity();
        if k < 2 {
            return Err(Error::InvalidState(format!(
                "a block needs at least two sites; got {k}"
            )));
        }
        if matches!(self, Shape::Cycle(_)) && k < 3 {
            return Err(Error::InvalidState(
                "a cycle needs at least three sites (two would double a bond)".into(),
            ));
        }
        if let Shape::Hypercube(dim) = self {
            if dim > 20 {
                return Err(Error::InvalidState(format!(
                    "a {dim}-cube has 2^{dim} sites; that is a register, not a block"
                )));
            }
        }
        Ok(())
    }
}

/// A site of a [`RecursiveLattice`]: a single point, or the same kind
/// of structure one scale finer.
#[derive(Debug, Clone, PartialEq)]
pub enum Site {
    /// One qubit — a lattice point.
    Point,
    /// An entire lattice standing where a point would stand.
    Lattice(Box<RecursiveLattice>),
}

impl Site {
    /// Number of leaf points under this site.
    pub fn width(&self) -> usize {
        match self {
            Site::Point => 1,
            Site::Lattice(l) => l.width(),
        }
    }

    /// Recursion depth below this site (a point has depth 0).
    pub fn depth(&self) -> usize {
        match self {
            Site::Point => 0,
            Site::Lattice(l) => l.depth(),
        }
    }

    /// Whether this site is a point.
    pub fn is_point(&self) -> bool {
        matches!(self, Site::Point)
    }
}

/// One bond of the flattened register, tagged with the recursion depth
/// at which it was created: depth 0 is the outermost block's own
/// bonds (the *lateral* bonds joining whole sub-lattices), higher
/// depths are bonds internal to those sub-lattices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bond {
    /// Lower leaf index.
    pub a: usize,
    /// Higher leaf index.
    pub b: usize,
    /// Recursion depth of the block that produced the bond.
    pub depth: usize,
}

/// A lattice whose sites may themselves be lattices.
///
/// ```
/// use quantsim::recursive::{RecursiveLattice, Shape};
///
/// // Four qubits bonded in a square.
/// let flat = RecursiveLattice::new(Shape::SQUARE).unwrap();
/// assert_eq!(flat.width(), 4);
/// assert_eq!(flat.bonds().len(), 4);
///
/// // The same structure again, bonded laterally: a square of squares.
/// let nested = RecursiveLattice::nest(Shape::SQUARE, 2).unwrap();
/// assert_eq!(nested.width(), 16);
/// // four inner squares (4 bonds each) plus four lateral bonds
/// assert_eq!(nested.bonds().len(), 20);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RecursiveLattice {
    shape: Shape,
    sites: Vec<Site>,
}

impl RecursiveLattice {
    /// A flat lattice: every site a point.
    pub fn new(shape: Shape) -> Result<Self> {
        shape.validate()?;
        Ok(RecursiveLattice {
            shape,
            sites: vec![Site::Point; shape.arity()],
        })
    }

    /// `depth` levels of `shape` nested uniformly: `nest(SQUARE, 1)` is
    /// four points in a square, `nest(SQUARE, 2)` is four squares in a
    /// square, and so on.
    pub fn nest(shape: Shape, depth: usize) -> Result<Self> {
        shape.validate()?;
        if depth == 0 {
            return Err(Error::InvalidState(
                "a lattice needs at least one level; depth 0 is a bare point".into(),
            ));
        }
        let mut lattice = RecursiveLattice::new(shape)?;
        for _ in 1..depth {
            lattice = lattice.refined_all(shape)?;
        }
        Ok(lattice)
    }

    /// The block shape of this level.
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// The sites of this level.
    pub fn sites(&self) -> &[Site] {
        &self.sites
    }

    /// Total leaf points — the width of the flattened qubit register.
    pub fn width(&self) -> usize {
        self.sites.iter().map(Site::width).sum()
    }

    /// Recursion depth: 1 for a flat lattice, 2 for a lattice of flat
    /// lattices, and so on.
    pub fn depth(&self) -> usize {
        1 + self.sites.iter().map(Site::depth).max().unwrap_or(0)
    }

    /// Leaf index where site `slot` begins.
    pub fn leaf_offset(&self, slot: usize) -> usize {
        self.sites[..slot].iter().map(Site::width).sum()
    }

    /// The site addressed by `path` (empty path is not a site; use the
    /// lattice itself).
    pub fn site_at(&self, path: &[usize]) -> Result<&Site> {
        let (&head, rest) = path
            .split_first()
            .ok_or_else(|| Error::InvalidState("empty site path".into()))?;
        let site = self
            .sites
            .get(head)
            .ok_or_else(|| Error::InvalidState(format!("no site {head} at this level")))?;
        if rest.is_empty() {
            return Ok(site);
        }
        match site {
            Site::Point => Err(Error::InvalidState(format!(
                "site {head} is a point; it has no sub-sites to address"
            ))),
            Site::Lattice(l) => l.site_at(rest),
        }
    }

    fn site_at_mut(&mut self, path: &[usize]) -> Result<&mut Site> {
        let (&head, rest) = path
            .split_first()
            .ok_or_else(|| Error::InvalidState("empty site path".into()))?;
        let len = self.sites.len();
        let site = self
            .sites
            .get_mut(head)
            .ok_or_else(|| Error::InvalidState(format!("no site {head} of {len} at this level")))?;
        if rest.is_empty() {
            return Ok(site);
        }
        match site {
            Site::Point => Err(Error::InvalidState(format!(
                "site {head} is a point; it has no sub-sites to address"
            ))),
            Site::Lattice(l) => l.site_at_mut(rest),
        }
    }

    /// **The point becomes a lattice.** Replace the point at `path`
    /// with a flat lattice of `shape`. Refusing to refine a site that
    /// is already a lattice is deliberate: refinement is a statement
    /// about a *point*, and silently nesting one more level would make
    /// the depth of a register depend on call order.
    pub fn refine(&mut self, path: &[usize], shape: Shape) -> Result<()> {
        shape.validate()?;
        let site = self.site_at_mut(path)?;
        match site {
            Site::Point => {
                *site = Site::Lattice(Box::new(RecursiveLattice::new(shape)?));
                Ok(())
            }
            Site::Lattice(_) => Err(Error::InvalidState(format!(
                "site {path:?} is already a lattice; coarsen it first to refine it differently"
            ))),
        }
    }

    /// **The lattice becomes a point.** The structural inverse of
    /// [`refine`](Self::refine); the *state-level* inverse is the
    /// measured substitution in [`substitution_report`].
    pub fn coarsen(&mut self, path: &[usize]) -> Result<()> {
        let site = self.site_at_mut(path)?;
        match site {
            Site::Lattice(_) => {
                *site = Site::Point;
                Ok(())
            }
            Site::Point => Err(Error::InvalidState(format!(
                "site {path:?} is already a point"
            ))),
        }
    }

    /// Every point at every depth becomes a lattice of `shape` — one
    /// uniform scale step outward.
    pub fn refined_all(&self, shape: Shape) -> Result<Self> {
        shape.validate()?;
        let sites = self
            .sites
            .iter()
            .map(|s| match s {
                Site::Point => Ok(Site::Lattice(Box::new(RecursiveLattice::new(shape)?))),
                Site::Lattice(l) => Ok(Site::Lattice(Box::new(l.refined_all(shape)?))),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RecursiveLattice {
            shape: self.shape,
            sites,
        })
    }

    /// The leaf a sub-lattice exposes in direction `direction` — its
    /// **corner**, reached by descending into slot `direction` at every
    /// level. The rule is scale-free: it names the same relative corner
    /// however deep the structure goes, which is what makes lateral
    /// bonding well defined between blocks of any depth.
    pub fn corner(&self, direction: usize) -> usize {
        let slot = direction % self.sites.len();
        let off = self.leaf_offset(slot);
        match &self.sites[slot] {
            Site::Point => off,
            Site::Lattice(l) => off + l.corner(direction),
        }
    }

    /// The leaf of site `slot` that carries this level's bond toward
    /// site `direction`: the site itself when it is a point, its
    /// corner in that direction when it is a lattice.
    pub fn port(&self, slot: usize, direction: usize) -> usize {
        let off = self.leaf_offset(slot);
        match &self.sites[slot] {
            Site::Point => off,
            Site::Lattice(l) => off + l.corner(direction),
        }
    }

    /// Every bond of the flattened register, depth-tagged, sorted and
    /// deduplicated. Bonds at depth 0 are this level's own — the
    /// lateral bonds between whole sub-lattices; deeper bonds are
    /// internal to them.
    pub fn bonds(&self) -> Vec<Bond> {
        let mut out = Vec::new();
        self.collect_bonds(0, 0, &mut out);
        out.sort_unstable_by_key(|b| (b.a, b.b, b.depth));
        out.dedup_by_key(|b| (b.a, b.b));
        out
    }

    fn collect_bonds(&self, base: usize, depth: usize, out: &mut Vec<Bond>) {
        for (i, j) in self.shape.bonds() {
            let a = base + self.port(i, j);
            let b = base + self.port(j, i);
            if a != b {
                out.push(Bond {
                    a: a.min(b),
                    b: a.max(b),
                    depth,
                });
            }
        }
        let mut off = base;
        for site in &self.sites {
            if let Site::Lattice(l) = site {
                l.collect_bonds(off, depth + 1, out);
            }
            off += site.width();
        }
    }

    /// The bonds as bare index pairs.
    pub fn bond_pairs(&self) -> Vec<(usize, usize)> {
        self.bonds().iter().map(|b| (b.a, b.b)).collect()
    }

    /// The register fabric as a [`Topology`], so the recursive lattice
    /// runs on [`DeviceState`](crate::backend::DeviceState) and reports
    /// its causal metrics like any other device geometry.
    pub fn topology(&self) -> Result<Topology> {
        Topology::custom(self.width(), &self.bond_pairs())
    }

    /// Leaf ranges `[lo, hi)` of the sub-lattices at recursion depth
    /// `depth`: depth 0 is the whole register as one range, depth 1 the
    /// immediate sites, and so on. Sites shallower than `depth` are
    /// returned at their own depth — the partition always covers every
    /// leaf exactly once.
    pub fn blocks(&self, depth: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        self.collect_blocks(0, depth, &mut out);
        out
    }

    fn collect_blocks(&self, base: usize, depth: usize, out: &mut Vec<(usize, usize)>) {
        if depth == 0 {
            out.push((base, base + self.width()));
            return;
        }
        let mut off = base;
        for site in &self.sites {
            match site {
                Site::Point => out.push((off, off + 1)),
                Site::Lattice(l) => l.collect_blocks(off, depth - 1, out),
            }
            off += site.width();
        }
    }

    /// A one-line structural summary, e.g. `Cycle(4){Cycle(4){•⁴}⁴}`.
    pub fn describe(&self) -> String {
        let inner: Vec<String> = self
            .sites
            .iter()
            .map(|s| match s {
                Site::Point => "•".to_string(),
                Site::Lattice(l) => l.describe(),
            })
            .collect();
        format!("{:?}{{{}}}", self.shape, inner.join(","))
    }
}

// ── small dense real-symmetric eigensolvers ──────────────────────────

/// Cyclic Jacobi eigendecomposition of a real symmetric `n × n` matrix
/// (row-major). Returns eigenvalues ascending and the matching
/// eigenvectors as the *columns* of a row-major `n × n` matrix.
fn sym_eigen(n: usize, a_in: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut a = a_in.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let scale: f64 = a.iter().map(|x| x * x).sum::<f64>().max(1.0);
    for _sweep in 0..100 {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off <= 1e-30 * scale {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| a[x * n + x].total_cmp(&a[y * n + y]));
    let values: Vec<f64> = order.iter().map(|&j| a[j * n + j]).collect();
    let mut vecs = vec![0.0f64; n * n];
    for (slot, &j) in order.iter().enumerate() {
        for k in 0..n {
            vecs[k * n + slot] = v[k * n + j];
        }
    }
    (values, vecs)
}

/// Modified Gram–Schmidt orthonormalization of `block` column vectors
/// stored as `vs[j]`. Columns that collapse are re-seeded from the
/// deterministic stream so the block never loses rank.
fn orthonormalize(vs: &mut [Vec<f64>], rng: &mut Prng) {
    for j in 0..vs.len() {
        for i in 0..j {
            let dot: f64 = vs[j].iter().zip(&vs[i]).map(|(x, y)| x * y).sum();
            for k in 0..vs[j].len() {
                vs[j][k] -= dot * vs[i][k];
            }
        }
        let norm: f64 = vs[j].iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for x in vs[j].iter_mut() {
                *x /= norm;
            }
        } else {
            for x in vs[j].iter_mut() {
                *x = rng.next_f64() - 0.5;
            }
            let norm: f64 = vs[j].iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in vs[j].iter_mut() {
                *x /= norm;
            }
        }
    }
}

/// The lowest `want` eigenpairs of a real symmetric operator given only
/// as a matrix-vector product, by **block Lanczos with full
/// reorthogonalization** and Rayleigh–Ritz extraction.
///
/// Block, not plain, Lanczos: a single starting vector spans only one
/// direction of each eigenspace, so exact degeneracies — and the
/// transverse-field Ising model on a symmetric block has them — would
/// silently come back with the wrong multiplicity. A starting block of
/// `want + 2` vectors resolves multiplicities up to that width, and the
/// residual is returned so an unconverged answer is visible rather than
/// assumed.
fn lowest_pairs(
    dim: usize,
    matvec: &dyn Fn(&[f64], &mut [f64]),
    want: usize,
    blocks: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>, f64)> {
    let p = (want + 2).min(dim);
    let mut rng = Prng::new(0x5EED_1CE5);
    let mut basis: Vec<Vec<f64>> = (0..p)
        .map(|_| (0..dim).map(|_| rng.next_f64() - 0.5).collect())
        .collect();
    orthonormalize(&mut basis, &mut rng);

    let mut hv = vec![0.0f64; dim];
    let mut frontier = 0usize;
    for _ in 0..blocks {
        crate::guard::checkpoint()?;
        let next = basis.len();
        if next >= dim {
            break;
        }
        for i in frontier..next {
            matvec(&basis[i], &mut hv);
            let mut w = hv.clone();
            // Full reorthogonalization against the whole basis, twice:
            // one pass loses orthogonality exactly where Lanczos is
            // known to, and the second pass is cheap at these widths.
            for _ in 0..2 {
                for q in basis.iter() {
                    let dot: f64 = q.iter().zip(&w).map(|(a, b)| a * b).sum();
                    for (x, y) in w.iter_mut().zip(q) {
                        *x -= dot * y;
                    }
                }
            }
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-9 {
                for x in w.iter_mut() {
                    *x /= norm;
                }
                basis.push(w);
            }
            if basis.len() >= dim {
                break;
            }
        }
        if basis.len() == next {
            break; // the Krylov space closed: nothing more to find
        }
        frontier = next;
    }

    // Rayleigh–Ritz over the accumulated basis.
    let b = basis.len();
    let mut t = vec![0.0f64; b * b];
    for i in 0..b {
        matvec(&basis[i], &mut hv);
        for j in 0..b {
            t[j * b + i] = basis[j].iter().zip(&hv).map(|(a, c)| a * c).sum();
        }
    }
    for i in 0..b {
        for j in (i + 1)..b {
            let m = 0.5 * (t[i * b + j] + t[j * b + i]);
            t[i * b + j] = m;
            t[j * b + i] = m;
        }
    }
    let (values, y) = sym_eigen(b, &t);

    let mut out_values = Vec::with_capacity(want);
    let mut vectors = Vec::with_capacity(want);
    let mut residual = 0.0f64;
    for slot in 0..want.min(b) {
        let mut v = vec![0.0f64; dim];
        for (i, q) in basis.iter().enumerate() {
            let coeff = y[i * b + slot];
            if coeff != 0.0 {
                for (x, qk) in v.iter_mut().zip(q) {
                    *x += coeff * qk;
                }
            }
        }
        matvec(&v, &mut hv);
        let lambda: f64 = v.iter().zip(&hv).map(|(x, z)| x * z).sum();
        let r: f64 = hv
            .iter()
            .zip(&v)
            .map(|(z, x)| (z - lambda * x) * (z - lambda * x))
            .sum::<f64>()
            .sqrt();
        residual = residual.max(r);
        out_values.push(values[slot]);
        vectors.push(v);
    }
    Ok((out_values, vectors, residual))
}

// ── the transverse-field Ising model on a bond list ──────────────────

/// Widest register this module will diagonalize; past it the state
/// vector alone is gigabytes and the answer belongs to a tensor-network
/// backend, not an exact eigensolver.
pub const EXACT_MAX_QUBITS: usize = 14;

/// Diagonal `−J Σ_bonds Z_iZ_j − b Σ_i Z_i` for every basis state.
fn ising_diagonal(n: usize, bonds: &[(usize, usize)], j: f64, longitudinal: f64) -> Vec<f64> {
    let dim = 1usize << n;
    (0..dim)
        .map(|idx| {
            let sign = |q: usize| if (idx >> q) & 1 == 1 { -1.0 } else { 1.0 };
            let zz: f64 = bonds.iter().map(|&(a, b)| sign(a) * sign(b)).sum();
            let z: f64 = (0..n).map(sign).sum();
            -j * zz - longitudinal * z
        })
        .collect()
}

/// `H = −J Σ_bonds Z_iZ_j − h Σ_i X_i − b Σ_i Z_i` applied to a vector.
fn ising_matvec(n: usize, diag: &[f64], h: f64, x: &[f64], y: &mut [f64]) {
    let dim = 1usize << n;
    for i in 0..dim {
        y[i] = diag[i] * x[i];
    }
    if h != 0.0 {
        for i in 0..dim {
            let mut acc = 0.0;
            for q in 0..n {
                acc += x[i ^ (1 << q)];
            }
            y[i] -= h * acc;
        }
    }
}

/// Dense `2^n × 2^n` matrix of the same Hamiltonian, for the small
/// blocks where the whole spectrum is wanted.
fn ising_dense(n: usize, bonds: &[(usize, usize)], j: f64, h: f64, longitudinal: f64) -> Vec<f64> {
    let dim = 1usize << n;
    let diag = ising_diagonal(n, bonds, j, longitudinal);
    let mut m = vec![0.0f64; dim * dim];
    for i in 0..dim {
        m[i * dim + i] = diag[i];
        for q in 0..n {
            m[i * dim + (i ^ (1 << q))] -= h;
        }
    }
    m
}

/// Lowest `want` eigenvalues of the transverse-field Ising model on a
/// bond list, with the eigensolver's measured residual.
fn ising_levels(
    n: usize,
    bonds: &[(usize, usize)],
    j: f64,
    h: f64,
    want: usize,
) -> Result<(Vec<f64>, f64)> {
    if n > EXACT_MAX_QUBITS {
        return Err(Error::TooManyQubits {
            requested: n,
            max: EXACT_MAX_QUBITS,
        });
    }
    let dim = 1usize << n;
    if dim <= 64 {
        let m = ising_dense(n, bonds, j, h, 0.0);
        let (values, _) = sym_eigen(dim, &m);
        return Ok((values.into_iter().take(want).collect(), 0.0));
    }
    let diag = ising_diagonal(n, bonds, j, 0.0);
    let (values, _, residual) =
        lowest_pairs(dim, &|x, y| ising_matvec(n, &diag, h, x, y), want, 40)?;
    Ok((values, residual))
}

// ── block-spin renormalization: the lattice as a point ───────────────

/// What one block of the recursive lattice measures itself to be when
/// it stands in for a single point.
///
/// Nothing here is postulated: the effective field is half the block's
/// *measured* gap (the tunneling splitting between the two lowest
/// states, which is what a single effective spin's transverse field
/// produces), and the effective coupling is the bare coupling times the
/// square of the *measured* boundary matrix element `μ = ⟨g₀|Z_port|g₁⟩`
/// — the amplitude with which the block's effective spin actually
/// reaches its port.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockRg {
    /// Block shape that was diagonalized.
    pub shape: Shape,
    /// Bare coupling `J`.
    pub coupling: f64,
    /// Bare transverse field `h`.
    pub field: f64,
    /// Ground-state energy of the isolated block.
    pub ground_energy: f64,
    /// Measured gap `E₁ − E₀`.
    pub gap: f64,
    /// Measured boundary matrix element `μ = ⟨g₀|Z_port|g₁⟩`.
    pub port_element: f64,
    /// Renormalized coupling `J' = J·μ²` per lateral bond.
    pub renorm_coupling: f64,
    /// Renormalized field `h' = gap/2`.
    pub renorm_field: f64,
    /// Orthonormality deviation of the two-state isometry `V†V − I`.
    pub isometry_error: f64,
}

impl BlockRg {
    /// The dimensionless coupling ratio before and after the step.
    pub fn ratio(&self) -> (f64, f64) {
        (
            self.coupling / self.field,
            self.renorm_coupling / self.renorm_field,
        )
    }
}

/// Diagonalize one block exactly and read its effective single-point
/// description off the spectrum.
///
/// The block is any [`RecursiveLattice`] — a flat square, or a square
/// of squares, or anything else within [`EXACT_MAX_QUBITS`]; the
/// procedure never asks how deep it is.
pub fn block_rg(lattice: &RecursiveLattice, j: f64, h: f64) -> Result<BlockRg> {
    let n = lattice.width();
    if n > EXACT_MAX_QUBITS {
        return Err(Error::TooManyQubits {
            requested: n,
            max: EXACT_MAX_QUBITS,
        });
    }
    if h == 0.0 {
        return Err(Error::InvalidState(
            "a zero transverse field leaves the block's two lowest states exactly degenerate; \
             the effective field it defines is 0/0"
                .into(),
        ));
    }
    let dim = 1usize << n;
    let bonds = lattice.bond_pairs();
    let m = ising_dense(n, &bonds, j, h, 0.0);
    let (values, vecs) = sym_eigen(dim, &m);
    let g0: Vec<f64> = (0..dim).map(|k| vecs[k * dim]).collect();
    let g1: Vec<f64> = (0..dim).map(|k| vecs[k * dim + 1]).collect();

    let port = lattice.corner(0);
    let sign = |idx: usize, q: usize| if (idx >> q) & 1 == 1 { -1.0 } else { 1.0 };
    let mu: f64 = (0..dim).map(|i| g0[i] * sign(i, port) * g1[i]).sum();

    let n00: f64 = g0.iter().map(|x| x * x).sum::<f64>() - 1.0;
    let n11: f64 = g1.iter().map(|x| x * x).sum::<f64>() - 1.0;
    let n01: f64 = g0.iter().zip(&g1).map(|(x, y)| x * y).sum();
    let isometry_error = n00.abs().max(n11.abs()).max(n01.abs());

    let gap = values[1] - values[0];
    Ok(BlockRg {
        shape: lattice.shape(),
        coupling: j,
        field: h,
        ground_energy: values[0],
        gap,
        port_element: mu,
        renorm_coupling: j * mu * mu,
        renorm_field: gap / 2.0,
        isometry_error,
    })
}

/// One step of the measured coupling flow.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RgStep {
    /// How many block substitutions deep this step is.
    pub scale: usize,
    /// Coupling at this scale.
    pub coupling: f64,
    /// Transverse field at this scale.
    pub field: f64,
    /// The dimensionless ratio `J/h` that actually flows.
    pub ratio: f64,
}

/// A measured coupling flow, with the reason it stopped.
#[derive(Debug, Clone, PartialEq)]
pub struct RgFlow {
    /// The flow, scale by scale.
    pub steps: Vec<RgStep>,
    /// Why the flow ended, when it ended before the requested depth.
    /// The ordered phase drives the block gap to zero geometrically, so
    /// a double-precision flow *always* runs out of resolution rather
    /// than out of physics — this says so instead of continuing into
    /// numerical noise.
    pub terminated: Option<String>,
}

/// Iterate [`block_rg`], feeding each step's renormalized couplings
/// back in as the next step's bare couplings: the flow of the lattice
/// under repeated substitution of a block for a point.
pub fn rg_flow(shape: Shape, j: f64, h: f64, steps: usize) -> Result<RgFlow> {
    let block = RecursiveLattice::new(shape)?;
    let mut out = vec![RgStep {
        scale: 0,
        coupling: j,
        field: h,
        ratio: j / h,
    }];
    let (mut j, mut h) = (j, h);
    let mut terminated = None;
    for scale in 1..=steps {
        let scale_of = j.abs().max(h.abs()).max(1.0);
        if h.abs() <= 1e-12 * scale_of {
            terminated = Some(format!(
                "block gap fell to {h:.3e} at scale {scale}, below double-precision \
                 resolution against a coupling of {j:.3e}: the flow has resolved into \
                 the ordered phase and further steps would be noise"
            ));
            break;
        }
        let step = block_rg(&block, j, h)?;
        j = step.renorm_coupling;
        h = step.renorm_field;
        if !j.is_finite() || !h.is_finite() {
            terminated = Some(format!("non-finite couplings at scale {scale}"));
            break;
        }
        out.push(RgStep {
            scale,
            coupling: j,
            field: h,
            ratio: j / h,
        });
    }
    Ok(RgFlow {
        steps: out,
        terminated,
    })
}

/// The coupling ratio at which the flow stands still — where a lattice
/// and a point are the *same* system — with the flow's linearization
/// there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RgFixedPoint {
    /// Block shape.
    pub shape: Shape,
    /// Fixed-point ratio `g* = J/h`.
    pub ratio: f64,
    /// `|g'(g*) − g*|` — how still the flow actually stands.
    pub residual: f64,
    /// The linearized flow `dg'/dg` at the fixed point.
    pub eigenvalue: f64,
    /// Correlation-length exponent `ν = ln b / ln λ` for the supplied
    /// length rescaling `b`.
    pub exponent: f64,
    /// The length rescaling assumed for the exponent.
    pub rescaling: f64,
    /// Bisection steps taken.
    pub iterations: usize,
}

/// The dimensionless RG map `g ↦ g'` at fixed `h = 1`.
fn rg_map(block: &RecursiveLattice, g: f64) -> Result<f64> {
    let step = block_rg(block, g, 1.0)?;
    Ok(step.renorm_coupling / step.renorm_field)
}

/// Bisect for the nontrivial fixed point of the block-spin flow.
///
/// The bracket is found by measurement, not assumption: at small `g`
/// the block's gap is the single-spin-flip gap and the port element is
/// spread over all `k` sites, so `g' < g` (flow to the disordered
/// point); at large `g` the block's two lowest states are the ordered
/// pair whose splitting collapses, so `g' > g`. The sign change between
/// them is the fixed point.
pub fn rg_fixed_point(shape: Shape, rescaling: f64) -> Result<RgFixedPoint> {
    let block = RecursiveLattice::new(shape)?;
    let f = |g: f64| -> Result<f64> { Ok(rg_map(&block, g)? - g) };
    let (mut lo, mut hi) = (1e-3, 1e-3);
    let mut bracketed = false;
    for step in 0..40 {
        let candidate = 1e-3 * 1.5f64.powi(step);
        if f(candidate)? > 0.0 {
            hi = candidate;
            bracketed = true;
            break;
        }
        lo = candidate;
    }
    if !bracketed {
        return Err(Error::InvalidState(format!(
            "no sign change in the block-spin flow for {shape:?} up to g = {lo:.3e}; \
             the flow has no measured fixed point in that range"
        )));
    }
    let mut iterations = 0usize;
    for _ in 0..200 {
        iterations += 1;
        let mid = 0.5 * (lo + hi);
        if f(mid)? > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
        if (hi - lo).abs() <= 1e-13 * hi.abs().max(1.0) {
            break;
        }
    }
    let g = 0.5 * (lo + hi);
    let residual = f(g)?.abs();
    let eps = 1e-5 * g.max(1.0);
    let eigenvalue = (rg_map(&block, g + eps)? - rg_map(&block, g - eps)?) / (2.0 * eps);
    let exponent = if eigenvalue > 1.0 {
        rescaling.ln() / eigenvalue.ln()
    } else {
        f64::NAN
    };
    Ok(RgFixedPoint {
        shape,
        ratio: g,
        residual,
        eigenvalue,
        exponent,
        rescaling,
        iterations,
    })
}

/// The claim "a lattice may stand where a point stood", put on trial.
///
/// The fine system is two blocks bonded laterally — literally
/// `Chain(2)` with both sites refined into `shape`, so the bond list
/// comes from the same scale-free rule as everything else. The coarse
/// system is two *points* at the block's own renormalized couplings.
/// Both are diagonalized exactly; the comparison is between their
/// **excitation spectra** (energies above the ground state), which is
/// the scale-free part — the ground energies differ by the block
/// condensation energy the substitution deliberately absorbs.
///
/// The duality has a **resolution horizon**, and it is measured rather
/// than glossed: a point carries two states, a block carries `2^k`, so
/// the correspondence can only hold below the block's own internal gap
/// [`SubstitutionReport::internal_gap`]. Below it the two spectra track
/// each other; above it the fine system has excitations the coarse
/// description has no room for, and
/// [`SubstitutionReport::deviation`] reports the mismatch honestly.
/// [`SubstitutionReport::deviation_in_band`] is the number the claim
/// actually rests on.
#[derive(Debug, Clone, PartialEq)]
pub struct SubstitutionReport {
    /// Block shape substituted for a point.
    pub shape: Shape,
    /// Bare couplings.
    pub coupling: f64,
    /// Bare transverse field.
    pub field: f64,
    /// Renormalized `(J', h')` read off the isolated block.
    pub renorm: (f64, f64),
    /// Width of the fine (two-block) register.
    pub fine_width: usize,
    /// The isolated block's gap to its *third* level — the first
    /// excitation the two-state effective description cannot hold.
    pub internal_gap: f64,
    /// Excitation energies of the fine system.
    pub fine_gaps: Vec<f64>,
    /// Excitation energies of the coarse two-point system.
    pub coarse_gaps: Vec<f64>,
    /// Per-level relative deviation.
    pub level_deviations: Vec<f64>,
    /// Largest relative deviation across every compared level.
    pub deviation: f64,
    /// Largest relative deviation among levels *below* the block's
    /// internal gap — where the substitution actually claims to hold.
    pub deviation_in_band: f64,
    /// How many levels fell in that band.
    pub levels_in_band: usize,
    /// Worst eigensolver residual behind those numbers.
    pub residual: f64,
}

/// Measure the substitution error for `levels` excitations.
pub fn substitution_report(
    shape: Shape,
    j: f64,
    h: f64,
    levels: usize,
) -> Result<SubstitutionReport> {
    let block = RecursiveLattice::new(shape)?;
    let rg = block_rg(&block, j, h)?;

    let mut pair = RecursiveLattice::new(Shape::Chain(2))?;
    pair.refine(&[0], shape)?;
    pair.refine(&[1], shape)?;
    let fine_width = pair.width();
    let (fine, res_fine) = ising_levels(fine_width, &pair.bond_pairs(), j, h, levels + 1)?;

    let coarse_lattice = RecursiveLattice::new(Shape::Chain(2))?;
    let (coarse, res_coarse) = ising_levels(
        2,
        &coarse_lattice.bond_pairs(),
        rg.renorm_coupling,
        rg.renorm_field,
        levels + 1,
    )?;

    // The block's own third level: the first excitation the two-state
    // effective point cannot represent.
    let (block_levels, _) = ising_levels(block.width(), &block.bond_pairs(), j, h, 3)?;
    let internal_gap = block_levels[2] - block_levels[0];

    let fine_gaps: Vec<f64> = fine[1..].iter().map(|e| e - fine[0]).collect();
    let coarse_gaps: Vec<f64> = coarse[1..].iter().map(|e| e - coarse[0]).collect();
    let level_deviations: Vec<f64> = fine_gaps
        .iter()
        .zip(&coarse_gaps)
        .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1e-12))
        .collect();
    let deviation = level_deviations.iter().copied().fold(0.0f64, f64::max);
    // A level is inside the duality's band only when *both* spectra put
    // it below the block's internal gap: above that the fine system has
    // block excitations the effective point cannot hold, and the coarse
    // system invents levels that are not there.
    let in_band: Vec<f64> = (0..level_deviations.len())
        .filter(|&i| fine_gaps[i] < internal_gap && coarse_gaps[i] < internal_gap)
        .map(|i| level_deviations[i])
        .collect();
    let levels_in_band = in_band.len();
    let deviation_in_band = in_band.into_iter().fold(0.0f64, f64::max);

    Ok(SubstitutionReport {
        shape,
        coupling: j,
        field: h,
        renorm: (rg.renorm_coupling, rg.renorm_field),
        fine_width,
        internal_gap,
        fine_gaps,
        coarse_gaps,
        level_deviations,
        deviation,
        deviation_in_band,
        levels_in_band,
        residual: res_fine.max(res_coarse),
    })
}

// ── phonons: the harmonic lattice, where the duality is exact ────────

/// A weighted spring between two leaves.
pub type Spring = (usize, usize, f64);

/// Springs of a recursive lattice, with a weight per recursion depth
/// (`weights[d]` for depth `d`; the last entry repeats for deeper
/// bonds).
pub fn lattice_springs(lattice: &RecursiveLattice, weights: &[f64]) -> Result<Vec<Spring>> {
    if weights.is_empty() {
        return Err(Error::InvalidState(
            "at least one spring weight is needed".into(),
        ));
    }
    Ok(lattice
        .bonds()
        .iter()
        .map(|b| {
            let w = weights[b.depth.min(weights.len() - 1)];
            (b.a, b.b, w)
        })
        .collect())
}

/// The dynamical matrix `D = ω²I + L(springs)` of a harmonic lattice
/// with unit masses: on-site restoring force `ω²` plus the weighted
/// graph Laplacian of the springs.
pub fn dynamical_matrix(n: usize, springs: &[Spring], omega: f64) -> Vec<f64> {
    let mut d = vec![0.0f64; n * n];
    for i in 0..n {
        d[i * n + i] = omega * omega;
    }
    for &(a, b, w) in springs {
        d[a * n + a] += w;
        d[b * n + b] += w;
        d[a * n + b] -= w;
        d[b * n + a] -= w;
    }
    d
}

/// Normal modes of a harmonic lattice: frequencies ascending with their
/// mode vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct PhononModes {
    /// Normal-mode frequencies, ascending.
    pub frequencies: Vec<f64>,
    /// Mode vectors, one per frequency.
    pub vectors: Vec<Vec<f64>>,
}

/// Diagonalize the dynamical matrix exactly.
pub fn phonon_modes(n: usize, springs: &[Spring], omega: f64) -> PhononModes {
    let d = dynamical_matrix(n, springs, omega);
    let (values, vecs) = sym_eigen(n, &d);
    let frequencies = values.iter().map(|&l| l.max(0.0).sqrt()).collect();
    let vectors = (0..n)
        .map(|k| (0..n).map(|i| vecs[i * n + k]).collect())
        .collect();
    PhononModes {
        frequencies,
        vectors,
    }
}

/// How a harmonic block presents itself as a single mode.
///
/// The block's internal springs annihilate the uniform displacement
/// exactly, so its lowest mode sits at the *bare* site frequency with
/// no shift at all — measured, not assumed
/// ([`PhononBlock::frequency_shift`] is the number). What renormalizes
/// is the coupling: an external spring attached at one port sees the
/// collective coordinate only through that port's participation
/// `c_port`, so `g' = g·c_port²`.
#[derive(Debug, Clone, PartialEq)]
pub struct PhononBlock {
    /// Block shape.
    pub shape: Shape,
    /// Number of sites in the block.
    pub sites: usize,
    /// Frequency of the block's collective (lowest) mode.
    pub collective_frequency: f64,
    /// `|ω_collective − ω|` — measured, and exactly zero for a
    /// connected block.
    pub frequency_shift: f64,
    /// Gap to the first internal mode: how well separated the
    /// collective coordinate is.
    pub internal_gap: f64,
    /// Port participation `c_port` in the collective mode.
    pub port_participation: f64,
    /// Renormalized spring `g' = g·c_port²` for a single port bond.
    pub renorm_spring: f64,
    /// Renormalized spring for *uniform* bonding, where every site of
    /// one block couples to every site of the other with weight
    /// `g/k²`: the participation is exactly 1 and `g' = g/k`.
    pub uniform_renorm_spring: f64,
}

/// Measure a block's collective mode and the coupling it renormalizes.
pub fn phonon_block(shape: Shape, omega: f64, spring: f64) -> Result<PhononBlock> {
    let block = RecursiveLattice::new(shape)?;
    let k = block.width();
    let springs = lattice_springs(&block, &[spring])?;
    let modes = phonon_modes(k, &springs, omega);
    let port = block.corner(0);
    let collective = &modes.vectors[0];
    let norm: f64 = collective.iter().map(|x| x * x).sum::<f64>().sqrt();
    let participation = (collective[port] / norm).abs();
    Ok(PhononBlock {
        shape,
        sites: k,
        collective_frequency: modes.frequencies[0],
        frequency_shift: (modes.frequencies[0] - omega).abs(),
        internal_gap: modes.frequencies[1] - modes.frequencies[0],
        port_participation: participation,
        renorm_spring: spring * participation * participation,
        uniform_renorm_spring: spring / k as f64,
    })
}

/// How the blocks of a refined lattice are joined to each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bonding {
    /// One spring per lateral bond, attached at the blocks' corner
    /// ports — the structure [`RecursiveLattice::bonds`] builds.
    Port,
    /// Every site of one block coupled to every site of the other with
    /// weight `g/k²`. The collective subspace is then *exactly*
    /// invariant, and the substitution is exact.
    Uniform,
}

/// The phonon substitution measured in both bonding regimes.
#[derive(Debug, Clone, PartialEq)]
pub struct PhononSubstitution {
    /// Coarse shape whose sites were refined.
    pub shape: Shape,
    /// Sites per block.
    pub block_sites: usize,
    /// Width of the refined register.
    pub fine_width: usize,
    /// Lowest fine frequencies (as many as the coarse lattice has
    /// modes).
    pub fine_frequencies: Vec<f64>,
    /// The coarse lattice's frequencies at the renormalized spring.
    pub coarse_frequencies: Vec<f64>,
    /// Renormalized spring used for the coarse lattice.
    pub renorm_spring: f64,
    /// Largest absolute deviation between the two frequency lists.
    pub deviation: f64,
}

/// Refine every site of `shape` into a block of `shape`, then ask
/// whether the refined lattice's lowest band *is* the coarse lattice.
///
/// With [`Bonding::Uniform`] the answer is yes to machine precision;
/// with [`Bonding::Port`] the lateral spring mixes the collective
/// coordinate into the internal modes and the deviation is finite,
/// measured, and second order in the lateral spring.
pub fn phonon_substitution(
    shape: Shape,
    omega: f64,
    internal: f64,
    lateral: f64,
    bonding: Bonding,
) -> Result<PhononSubstitution> {
    let coarse = RecursiveLattice::new(shape)?;
    let k = shape.arity();
    let fine = coarse.refined_all(shape)?;
    let n = fine.width();

    let mut springs: Vec<Spring> = fine
        .bonds()
        .iter()
        .filter(|b| b.depth > 0)
        .map(|b| (b.a, b.b, internal))
        .collect();
    let block_ranges = fine.blocks(1);
    match bonding {
        Bonding::Port => {
            for b in fine.bonds().iter().filter(|b| b.depth == 0) {
                springs.push((b.a, b.b, lateral));
            }
        }
        Bonding::Uniform => {
            let per = lateral / (k * k) as f64;
            for (i, jx) in shape.bonds() {
                let (lo_i, hi_i) = block_ranges[i];
                let (lo_j, hi_j) = block_ranges[jx];
                for a in lo_i..hi_i {
                    for b in lo_j..hi_j {
                        springs.push((a, b, per));
                    }
                }
            }
        }
    }

    let fine_modes = phonon_modes(n, &springs, omega);
    let renorm = lateral / k as f64;
    let coarse_springs: Vec<Spring> = coarse.bonds().iter().map(|b| (b.a, b.b, renorm)).collect();
    let coarse_modes = phonon_modes(k, &coarse_springs, omega);

    let fine_frequencies: Vec<f64> = fine_modes.frequencies[..k].to_vec();
    let deviation = fine_frequencies
        .iter()
        .zip(&coarse_modes.frequencies)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    Ok(PhononSubstitution {
        shape,
        block_sites: k,
        fine_width: n,
        fine_frequencies,
        coarse_frequencies: coarse_modes.frequencies.clone(),
        renorm_spring: renorm,
        deviation,
    })
}

// ── the same statement, executed on the simulator ────────────────────

/// The `xy` hopping gate: a rotation by `θ/2` inside `{|01⟩, |10⟩}`,
/// which is `exp(−i·θ/2·(XX+YY)/2)` — the phonon hop.
fn register_xy(reg: &mut GateRegistry<C64>) -> Result<()> {
    if reg.contains("xy") {
        return Ok(());
    }
    reg.register_parametric("xy", "XY hopping interaction", 2, 1, |p| {
        let (c, s) = ((p[0] / 2.0).cos(), (p[0] / 2.0).sin());
        let (o, l) = (c64(1.0, 0.0), c64(0.0, 0.0));
        #[rustfmt::skip]
        let entries = [
            o, l,            l,            l,
            l, c64(c, 0.0),  c64(0.0, -s), l,
            l, c64(0.0, -s), c64(c, 0.0),  l,
            l, l,            l,            o,
        ];
        GateMatrix::try_from_c64s(4, &entries).ok_or(Error::UnsupportedForAlgebra {
            gate: "xy".into(),
            algebra: "non-complex".into(),
        })
    })
}

/// A single phonon walked through the recursive lattice on a real
/// backend, with every error source separated and measured.
#[derive(Debug, Clone, PartialEq)]
pub struct PhononWalk {
    /// Width of the simulated register.
    pub width: usize,
    /// Trotter steps taken.
    pub steps: usize,
    /// Time step per Trotter layer.
    pub dt: f64,
    /// Coarse spring the substitution predicts, `lateral/k`.
    pub coarse_spring: f64,
    /// Weight the simulator left outside the one-excitation sector.
    /// The `xy` gate conserves excitation number, so this is a
    /// conservation law measured rather than assumed.
    pub leakage: f64,
    /// Simulator vs exact single-particle evolution of the *same* fine
    /// lattice: pure Trotter error.
    pub trotter_deviation: f64,
    /// Exact fine evolution, coarse-grained onto the blocks, vs the
    /// coarse lattice's own exact evolution at the renormalized hop:
    /// pure substitution error.
    pub substitution_deviation: f64,
    /// Simulator vs the coarse prediction — what an experiment that
    /// only ever saw the coarse register would measure.
    pub coarse_deviation: f64,
}

/// Exact `exp(−iAt)` applied to `v` for a real symmetric `n × n` `A`.
fn evolve_symmetric(n: usize, a: &[f64], v: &[f64], t: f64) -> Vec<C64> {
    let (values, vecs) = sym_eigen(n, a);
    let mut out = vec![C64::new(0.0, 0.0); n];
    for k in 0..n {
        let overlap: f64 = (0..n).map(|i| vecs[i * n + k] * v[i]).sum();
        if overlap == 0.0 {
            continue;
        }
        let phase = C64::new((values[k] * t).cos(), -(values[k] * t).sin());
        for (i, slot) in out.iter_mut().enumerate() {
            *slot += phase * overlap * vecs[i * n + k];
        }
    }
    out
}

/// Load one phonon into a block's **collective coordinate**, evolve it
/// through the recursive lattice's own bonds on a [`Backend`], and
/// compare what a coarse observer sees against the coarse lattice's
/// exact evolution at the renormalized coupling.
///
/// The generator is the spring **Laplacian**, not a bare hopping
/// matrix, and the distinction is the whole point: only the Laplacian
/// annihilates a block's uniform displacement exactly, which is what
/// makes the collective coordinate a protected degree of freedom rather
/// than one mode among many. The internal springs must dominate the
/// lateral ones for the block to hold together as a point — pass
/// `internal ≫ lateral` and the substitution error is small and
/// measured; pass them comparable and the report says so.
///
/// The initial state is the uniform superposition over block 0's sites
/// — the collective mode that is supposed to *be* the coarse point's
/// mode. Nothing is fitted: the coarse spring is `lateral/k`, the
/// participation-renormalized value measured in [`phonon_block`].
pub fn phonon_walk(
    shape: Shape,
    internal: f64,
    lateral: f64,
    dt: f64,
    steps: usize,
) -> Result<PhononWalk> {
    let coarse = RecursiveLattice::new(shape)?;
    let k = shape.arity();
    let fine = coarse.refined_all(shape)?;
    let n = fine.width();
    if n > EXACT_MAX_QUBITS + 4 {
        return Err(Error::TooManyQubits {
            requested: n,
            max: EXACT_MAX_QUBITS + 4,
        });
    }
    let blocks = fine.blocks(1);

    // Single-excitation generators: the weighted graph Laplacians. The
    // on-site term is the site's total spring load, so a block's
    // uniform displacement is annihilated exactly.
    let weight = |depth: usize| if depth == 0 { lateral } else { internal };
    let mut fine_h = vec![0.0f64; n * n];
    for b in fine.bonds() {
        let w = weight(b.depth);
        fine_h[b.a * n + b.b] -= w;
        fine_h[b.b * n + b.a] -= w;
        fine_h[b.a * n + b.a] += w;
        fine_h[b.b * n + b.b] += w;
    }
    let coarse_spring = lateral / k as f64;
    let mut coarse_h = vec![0.0f64; k * k];
    for b in coarse.bonds() {
        coarse_h[b.a * k + b.b] -= coarse_spring;
        coarse_h[b.b * k + b.a] -= coarse_spring;
        coarse_h[b.a * k + b.a] += coarse_spring;
        coarse_h[b.b * k + b.b] += coarse_spring;
    }

    // Initial collective coordinate of block 0.
    let amp = 1.0 / (k as f64).sqrt();
    let (lo0, hi0) = blocks[0];
    let mut fine_v = vec![0.0f64; n];
    for slot in fine_v.iter_mut().take(hi0).skip(lo0) {
        *slot = amp;
    }
    let mut coarse_v = vec![0.0f64; k];
    coarse_v[0] = 1.0;

    let t = dt * steps as f64;
    let exact_fine = evolve_symmetric(n, &fine_h, &fine_v, t);
    let exact_coarse = evolve_symmetric(k, &coarse_h, &coarse_v, t);

    // The simulator: load the collective coordinate, Trotter the
    // Laplacian — `xy` for the off-diagonal hops, `p` for the on-site
    // spring load.
    let mut reg = GateRegistry::<C64>::standard();
    register_xy(&mut reg)?;
    let mut circuit: Circuit<C64> = Circuit::new(n);
    for _ in 0..steps {
        for b in fine.bonds() {
            circuit.gate("xy", [-2.0 * weight(b.depth) * dt], [b.a, b.b]);
        }
        for q in 0..n {
            let load = fine_h[q * n + q];
            if load != 0.0 {
                circuit.gate("p", [-load * dt], [q]);
            }
        }
    }
    let bound = circuit.bind(&reg)?;
    let mut state = DenseState::<C64>::new(n)?;
    let entries: Vec<(u64, C64)> = (lo0..hi0)
        .map(|q| (1u64 << q, C64::new(amp, 0.0)))
        .collect();
    state.load(&entries)?;
    bound.run(&mut state)?;

    let sim: Vec<C64> = (0..n).map(|q| state.amplitude(1u64 << q)).collect();
    let mut leakage = 0.0f64;
    state.for_each_nonzero(&mut |idx, a| {
        if idx.count_ones() != 1 {
            leakage += a.norm_sqr();
        }
    });

    let project = |v: &[C64]| -> Vec<C64> {
        blocks
            .iter()
            .map(|&(lo, hi)| {
                let mut acc = C64::new(0.0, 0.0);
                for slot in v.iter().take(hi).skip(lo) {
                    acc += *slot;
                }
                acc * amp
            })
            .collect()
    };

    let trotter_deviation = sim
        .iter()
        .zip(&exact_fine)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    let substitution_deviation = project(&exact_fine)
        .iter()
        .zip(&exact_coarse)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);
    let coarse_deviation = project(&sim)
        .iter()
        .zip(&exact_coarse)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f64, f64::max);

    Ok(PhononWalk {
        width: n,
        steps,
        dt,
        coarse_spring,
        leakage: leakage.sqrt(),
        trotter_deviation,
        substitution_deviation,
        coarse_deviation,
    })
}

// ── the computation that participates in itself ──────────────────────

/// One round of a computation feeding on its own output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticipationRound {
    /// Round index.
    pub iteration: usize,
    /// The field the block was solved in — produced by the previous
    /// round's own answer.
    pub field: f64,
    /// The magnetization the block answered with, read off a loaded
    /// backend through [`pauli_expectation`].
    pub magnetization: f64,
    /// `|m_new − m_old|` — how far the computation moved itself.
    pub residual: f64,
}

/// A block that is its own environment.
#[derive(Debug, Clone, PartialEq)]
pub struct SelfParticipation {
    /// The full trajectory of the self-feeding iteration.
    pub rounds: Vec<ParticipationRound>,
    /// Fixed-point magnetization.
    pub magnetization: f64,
    /// Fixed-point field.
    pub field: f64,
    /// Whether the iteration reached the requested tolerance.
    pub converged: bool,
    /// Final residual.
    pub residual: f64,
    /// Width of the block that participated.
    pub width: usize,
}

/// Ground-state magnetization of a block in a longitudinal field,
/// measured *through the simulator*: the eigenvector is loaded into a
/// [`DenseState`] and the observable read with
/// [`pauli_expectation`] like any other measurement in this crate.
fn block_magnetization(
    lattice: &RecursiveLattice,
    j: f64,
    h: f64,
    longitudinal: f64,
) -> Result<f64> {
    let n = lattice.width();
    let dim = 1usize << n;
    let bonds = lattice.bond_pairs();
    let ground = if dim <= 64 {
        let m = ising_dense(n, &bonds, j, h, longitudinal);
        let (_, vecs) = sym_eigen(dim, &m);
        (0..dim).map(|k| vecs[k * dim]).collect::<Vec<f64>>()
    } else {
        let diag = ising_diagonal(n, &bonds, j, longitudinal);
        let (_, vectors, _) = lowest_pairs(dim, &|x, y| ising_matvec(n, &diag, h, x, y), 1, 40)?;
        vectors.into_iter().next().expect("one vector requested")
    };
    let mut state = DenseState::<C64>::new(n)?;
    let entries: Vec<(u64, C64)> = ground
        .iter()
        .enumerate()
        .filter(|(_, &a)| a != 0.0)
        .map(|(i, &a)| (i as u64, C64::new(a, 0.0)))
        .collect();
    state.load(&entries)?;
    let port = lattice.corner(0);
    Ok(pauli_expectation(&state as &dyn Backend<C64>, &[(port, Pauli::Z)])?.re)
}

/// **The computation that participates in itself.**
///
/// A block is solved in the mean field its neighbours impose; the field
/// its neighbours impose is `J·z·m` where `m` is the block's own
/// boundary magnetization; so the block's output is the block's input
/// and the iteration is the system standing in for its own
/// environment. The block may itself be a lattice of lattices — the
/// same call, at any depth.
///
/// `seed` breaks the `m = 0` symmetry deliberately: `m = 0` is always a
/// fixed point, and starting there would report "converged" while
/// measuring nothing.
pub fn self_participation(
    lattice: &RecursiveLattice,
    j: f64,
    h: f64,
    neighbours: usize,
    seed: f64,
    tol: f64,
    max_iter: usize,
) -> Result<SelfParticipation> {
    let width = lattice.width();
    if width > EXACT_MAX_QUBITS {
        return Err(Error::TooManyQubits {
            requested: width,
            max: EXACT_MAX_QUBITS,
        });
    }
    let mut m = seed;
    let mut rounds = Vec::with_capacity(max_iter);
    let mut converged = false;
    let mut residual = f64::INFINITY;
    let mut field = 0.0;
    for iteration in 0..max_iter {
        field = j * neighbours as f64 * m;
        let next = block_magnetization(lattice, j, h, field)?;
        residual = (next - m).abs();
        rounds.push(ParticipationRound {
            iteration,
            field,
            magnetization: next,
            residual,
        });
        m = next;
        if residual <= tol {
            converged = true;
            break;
        }
    }
    Ok(SelfParticipation {
        rounds,
        magnetization: m,
        field,
        converged,
        residual,
        width,
    })
}

/// The coupling at which a computation feeding on itself stops
/// answering zero.
///
/// Below it the only self-consistent answer is `m = 0` — the
/// computation talks itself down to nothing; above it a nonzero
/// magnetization sustains itself. The crossing is bisected on measured
/// output, with no assumed form for the transition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticipationTransition {
    /// Critical coupling.
    pub coupling: f64,
    /// Sustained magnetization at 1.1× the critical coupling.
    pub magnetization_above: f64,
    /// Sustained magnetization at 0.9× the critical coupling — the
    /// computation talking itself down to nothing.
    pub magnetization_below: f64,
    /// Bisection width reached.
    pub bracket: f64,
    /// Bisection steps.
    pub iterations: usize,
}

/// Bisect the self-participation coupling at which the sustained
/// magnetization first exceeds `threshold`.
pub fn participation_transition(
    lattice: &RecursiveLattice,
    h: f64,
    neighbours: usize,
    threshold: f64,
    bracket: (f64, f64),
    steps: usize,
) -> Result<ParticipationTransition> {
    let sustained = |j: f64| -> Result<f64> {
        Ok(
            self_participation(lattice, j, h, neighbours, 0.5, 1e-10, 200)?
                .magnetization
                .abs(),
        )
    };
    let (mut lo, mut hi) = bracket;
    if sustained(lo)? > threshold {
        return Err(Error::InvalidState(format!(
            "the low end of the bracket (J = {lo}) already sustains a magnetization \
             above {threshold}; widen it downward"
        )));
    }
    if sustained(hi)? <= threshold {
        return Err(Error::InvalidState(format!(
            "the high end of the bracket (J = {hi}) sustains nothing above {threshold}; \
             widen it upward"
        )));
    }
    let mut iterations = 0usize;
    for _ in 0..steps {
        iterations += 1;
        let mid = 0.5 * (lo + hi);
        if sustained(mid)? > threshold {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let coupling = 0.5 * (lo + hi);
    Ok(ParticipationTransition {
        coupling,
        magnetization_above: sustained(1.1 * coupling)?,
        magnetization_below: sustained(0.9 * coupling)?,
        bracket: hi - lo,
        iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_of_squares_has_sixteen_leaves_and_twenty_bonds() {
        let l = RecursiveLattice::nest(Shape::SQUARE, 2).unwrap();
        assert_eq!(l.width(), 16);
        assert_eq!(l.depth(), 2);
        let bonds = l.bonds();
        assert_eq!(bonds.len(), 20);
        assert_eq!(bonds.iter().filter(|b| b.depth == 0).count(), 4);
        assert_eq!(bonds.iter().filter(|b| b.depth == 1).count(), 16);
    }

    #[test]
    fn sym_eigen_matches_a_known_spectrum() {
        // [[2,1],[1,2]] has eigenvalues 1 and 3.
        let (v, _) = sym_eigen(2, &[2.0, 1.0, 1.0, 2.0]);
        assert!((v[0] - 1.0).abs() < 1e-12);
        assert!((v[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn refine_then_coarsen_is_the_identity() {
        let mut l = RecursiveLattice::new(Shape::SQUARE).unwrap();
        let before = l.clone();
        l.refine(&[2], Shape::Chain(3)).unwrap();
        assert_eq!(l.width(), 6);
        l.coarsen(&[2]).unwrap();
        assert_eq!(l, before);
    }
}
