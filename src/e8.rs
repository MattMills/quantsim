//! The E8 root system, built and measured — the correlation fabric for
//! compound qudits.
//!
//! Everything here is *constructed and verified programmatically*, not
//! asserted from tables: the 240 roots are generated (112 integer
//! roots `±eᵢ ± eⱼ` and 128 half-integer spinor roots `(±½)⁸` with an
//! even number of minus signs), their count, norms and closure under
//! negation are checked, and the sub-structures are found by *search*
//! over the constructed system.
//!
//! The compound-qudit content: an `A_{d−1}` chain of roots carries the
//! `su(d)` symmetry of a `d`-level qudit, so chains for
//! `d = 2, 3, 4, 5` are the binary/ternary/quaternary/quintary
//! sub-qudits' symmetry frames. Two measured facts shape the compound
//! structure:
//!
//! * an **orthogonal pair of `A₄` chains exists** (the `su(5) × su(5)`
//!   decomposition — exhibited by search, orthogonality checked), so
//!   fully independent equal-arity sub-structures are possible;
//! * the four chains `A₁, A₂, A₃, A₄` have total rank
//!   `1 + 2 + 3 + 4 = 10 > 8`, so they **cannot be mutually
//!   orthogonal** in E8 — verified by exhaustive failure of the
//!   orthogonal-completion search, not just arithmetic. Any embedding
//!   of all four arities *must* share root directions: the sub-qudits
//!   are correlated by geometric necessity, and the measured overlap
//!   ([`ChainEmbedding::gram`], [`ChainEmbedding::coupling`]) is
//!   exactly the "correlated but independent in a horizontal set"
//!   structure — which [`crate::mixed`] imports as the compound
//!   qudit's coupling fabric.
//!
//! Coordinates are stored doubled (`2·root`) so every inner product is
//! exact integer arithmetic; a root has doubled norm² = 8.

use crate::error::{Error, Result};

/// One E8 root in doubled coordinates (all entries integers; real
/// coordinates are these halved).
pub type Root = [i32; 8];

/// Exact doubled inner product (`4 · ⟨α, β⟩` in real units).
pub fn dot(a: &Root, b: &Root) -> i32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Construct the 240 E8 roots: 112 integer roots (±2, ±2 in two
/// coordinates, doubled) and 128 even-sign half-spinor roots ((±1)⁸
/// doubled, even number of minus signs).
pub fn roots() -> Vec<Root> {
    let mut out = Vec::with_capacity(240);
    for i in 0..8 {
        for j in (i + 1)..8 {
            for (si, sj) in [(2, 2), (2, -2), (-2, 2), (-2, -2)] {
                let mut r = [0i32; 8];
                r[i] = si;
                r[j] = sj;
                out.push(r);
            }
        }
    }
    for signs in 0u32..256 {
        if signs.count_ones() % 2 == 0 {
            let mut r = [1i32; 8];
            for (bit, slot) in r.iter_mut().enumerate() {
                if (signs >> bit) & 1 == 1 {
                    *slot = -1;
                }
            }
            out.push(r);
        }
    }
    out
}

/// Verify the constructed system: 240 roots, every doubled norm² = 8,
/// closed under negation, and the integer/spinor split is 112 + 128.
pub fn verify() -> Result<()> {
    let rs = roots();
    if rs.len() != 240 {
        return Err(Error::InvalidState(format!("{} roots ≠ 240", rs.len())));
    }
    let mut integer = 0;
    let mut spinor = 0;
    for r in &rs {
        if dot(r, r) != 8 {
            return Err(Error::InvalidState(format!("root {r:?} has norm² ≠ 2")));
        }
        if r.iter().all(|&c| c % 2 == 0) {
            integer += 1;
        } else {
            spinor += 1;
        }
        let neg: Root = std::array::from_fn(|k| -r[k]);
        if !rs.contains(&neg) {
            return Err(Error::InvalidState(format!("−{r:?} missing")));
        }
    }
    if integer != 112 || spinor != 128 {
        return Err(Error::InvalidState(format!(
            "split {integer} + {spinor} ≠ 112 + 128"
        )));
    }
    Ok(())
}

/// Whether `chain` is a simple-root chain of type `A_k` (`k` roots,
/// consecutive pairs at doubled inner product −4, all other pairs
/// orthogonal): the Cartan pattern of `su(k+1)`.
pub fn is_a_chain(chain: &[Root]) -> bool {
    for (i, a) in chain.iter().enumerate() {
        if dot(a, a) != 8 {
            return false;
        }
        for (j, b) in chain.iter().enumerate().skip(i + 1) {
            let expected = if j == i + 1 { -4 } else { 0 };
            if dot(a, b) != expected {
                return false;
            }
        }
    }
    true
}

/// Search the root system for an `A_k` chain orthogonal to every root
/// in `avoid`, by depth-first extension. Returns the first chain
/// found, or `None` when the search space is exhausted — an exhausted
/// search is a *measured* impossibility, which is how the rank
/// obstruction below is demonstrated rather than assumed.
pub fn find_a_chain(k: usize, avoid: &[Root]) -> Option<Vec<Root>> {
    let rs = roots();
    let ok = |r: &Root| avoid.iter().all(|a| dot(r, a) == 0);
    fn extend(rs: &[Root], ok: &dyn Fn(&Root) -> bool, chain: &mut Vec<Root>, k: usize) -> bool {
        if chain.len() == k {
            return true;
        }
        for r in rs {
            if !ok(r) {
                continue;
            }
            let fits = chain
                .iter()
                .enumerate()
                .all(|(i, c)| dot(c, r) == if i + 1 == chain.len() { -4 } else { 0 });
            if fits {
                chain.push(*r);
                if extend(rs, ok, chain, k) {
                    return true;
                }
                chain.pop();
            }
        }
        false
    }
    let mut chain = Vec::new();
    if extend(&rs, &ok, &mut chain, k) {
        Some(chain)
    } else {
        None
    }
}

/// The measured neighbourhood of a root: how many of the OTHER 239
/// roots sit at each doubled inner product (−8, −4, 0, 4, 8). For E8
/// this comes out (1, 56, 126, 56, 0) at every root — the antipode,
/// 56 at −1, 126 orthogonal, 56 at +1, and nothing but the root
/// itself at +2. Measured here, not quoted.
pub fn neighbor_profile(root: &Root) -> [usize; 5] {
    let mut out = [0usize; 5];
    for r in roots() {
        if r == *root {
            continue;
        }
        match dot(root, &r) {
            -8 => out[0] += 1,
            -4 => out[1] += 1,
            0 => out[2] += 1,
            4 => out[3] += 1,
            8 => out[4] += 1,
            other => panic!("impossible inner product {other}"),
        }
    }
    out
}

/// The −1 adjacency (doubled inner product −4) as an edge list over
/// root indices into [`roots`], each pair once.
pub fn minus_one_edges() -> Vec<(usize, usize)> {
    let rs = roots();
    let mut edges = Vec::new();
    for i in 0..rs.len() {
        for j in (i + 1)..rs.len() {
            if dot(&rs[i], &rs[j]) == -4 {
                edges.push((i, j));
            }
        }
    }
    edges
}

/// The zero-sum triangles {α, β, γ} with α + β + γ = 0: because every
/// norm-2 vector of the E8 lattice is a root, EVERY −1 edge closes
/// into exactly one such triangle — the 2-cells of the root complex
/// are the additive relations themselves.
pub fn zero_sum_triangles() -> Vec<[usize; 3]> {
    let rs = roots();
    let mut triangles = Vec::new();
    for i in 0..rs.len() {
        for j in (i + 1)..rs.len() {
            if dot(&rs[i], &rs[j]) != -4 {
                continue;
            }
            let gamma: Root = std::array::from_fn(|k| -rs[i][k] - rs[j][k]);
            if let Some(g) = rs.iter().position(|r| *r == gamma) {
                if g > j {
                    triangles.push([i, j, g]);
                }
            }
        }
    }
    triangles
}

/// The GF(2) first Betti number of the root complex (vertices = the
/// 240 roots, edges = the −1 pairs, 2-cells = the zero-sum
/// triangles): the dimension of edge-stored data that is a cocycle
/// but NOT a coboundary — the invariant storage capacity of the
/// complex beyond point data. Computed by rank over GF(2), never
/// assumed.
pub fn triangle_complex_b1() -> usize {
    let edges = minus_one_edges();
    let triangles = zero_sum_triangles();
    let mut edge_id = std::collections::HashMap::new();
    for (id, &e) in edges.iter().enumerate() {
        edge_id.insert(e, id);
    }
    let words = edges.len().div_ceil(64);
    let key = |a: usize, b: usize| (a.min(b), a.max(b));
    // Column per triangle over GF(2), eliminated to count the rank of ∂₂.
    let mut basis: Vec<Vec<u64>> = Vec::new();
    let mut rank = 0usize;
    for t in &triangles {
        let mut col = vec![0u64; words];
        for &(a, b) in &[(t[0], t[1]), (t[0], t[2]), (t[1], t[2])] {
            let id = edge_id[&key(a, b)];
            col[id / 64] ^= 1u64 << (id % 64);
        }
        for row in &basis {
            let pivot = row.iter().rposition(|&w| w != 0).unwrap();
            let bit = 63 - row[pivot].leading_zeros() as usize;
            if col[pivot] >> bit & 1 == 1 {
                for (c, r) in col.iter_mut().zip(row) {
                    *c ^= r;
                }
            }
        }
        if col.iter().any(|&w| w != 0) {
            basis.push(col);
            rank += 1;
        }
    }
    // b₁ = E − V + components − rank ∂₂ (the −1 graph is connected: one
    // component, verified cheaply here).
    let mut parent: Vec<usize> = (0..240).collect();
    fn find(p: &mut Vec<usize>, x: usize) -> usize {
        if p[x] != x {
            let r = find(p, p[x]);
            p[x] = r;
        }
        p[x]
    }
    for &(a, b) in &edges {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        parent[ra] = rb;
    }
    let components = (0..240).filter(|&v| find(&mut parent, v) == v).count();
    edges.len() - 240 + components - rank
}

/// The E8×E8 qubit representation: 8 qubits held ON the root
/// geometry, with entanglement structure and state adjacency read off
/// as measured root geometry. See [`rep`].
pub mod rep {
    use super::{dot, Root};
    use crate::backend::Backend;
    use crate::math::svd_thin;
    use crate::scalar::C64;

    /// The spinor root of an even-parity 8-bit string: bit `i` = 0 maps
    /// to coordinate +1, bit 1 to −1 (doubled units). `None` for odd
    /// parity — odd strings live on the SECOND E8 copy (see
    /// [`sector_of`]).
    pub fn spinor_of_bits(bits: u8) -> Option<Root> {
        if bits.count_ones() % 2 != 0 {
            return None;
        }
        Some(std::array::from_fn(|i| {
            if (bits >> i) & 1 == 0 {
                1
            } else {
                -1
            }
        }))
    }

    /// The bit string of a spinor root (inverse of [`spinor_of_bits`]).
    pub fn bits_of_spinor(root: &Root) -> Option<u8> {
        if root.iter().any(|&c| c != 1 && c != -1) {
            return None;
        }
        let mut bits = 0u8;
        for (i, &c) in root.iter().enumerate() {
            if c == -1 {
                bits |= 1 << i;
            }
        }
        Some(bits)
    }

    /// Which E8 copy an 8-qubit basis state lives on: copy 0 holds the
    /// even-parity sector directly; copy 1 holds the odd sector via
    /// the fixed pairing `s ↦ s ⊕ 1` (flip qubit 0). Together the two
    /// spinor sectors are exactly the 256 basis states — E8×E8 as the
    /// full 8-qubit space.
    pub fn sector_of(bits: u8) -> (usize, Root) {
        if bits.count_ones() % 2 == 0 {
            (0, spinor_of_bits(bits).unwrap())
        } else {
            (1, spinor_of_bits(bits ^ 1).unwrap())
        }
    }

    /// The integer root connecting two SAME-sector basis states at
    /// Hamming distance 2: their spinor difference, which is always an
    /// integer root (±2 at exactly the two flipped positions). `None`
    /// when the states are not a distance-2 same-parity pair — the
    /// measured content: the 112 integer roots ARE the 2-local
    /// transition labels of the representation.
    pub fn transition_root(x: u8, y: u8) -> Option<Root> {
        if x.count_ones() % 2 != y.count_ones() % 2 || (x ^ y).count_ones() != 2 {
            return None;
        }
        let (sx, sy) = (sector_of(x).1, sector_of(y).1);
        let diff: Root = std::array::from_fn(|i| sy[i] - sx[i]);
        debug_assert_eq!(dot(&diff, &diff), 8);
        Some(diff)
    }

    /// The measured geometry of a state's basis support on the root
    /// system.
    #[derive(Debug, Clone)]
    pub struct SupportGeometry {
        /// Occupied basis states.
        pub points: usize,
        /// Split across the two E8 copies (even, odd sector).
        pub sectors: (usize, usize),
        /// Root-adjacent pairs inside the support (same sector,
        /// Hamming distance 2 — i.e. pairs connected by an integer
        /// root).
        pub root_edges: usize,
        /// GF(2) affine dimension of the support's bit strings — the
        /// smallest coset the state lives in.
        pub affine_dim: usize,
        /// Antipodal pairs in the support (a copy's maximal-distance
        /// geometry — GHZ lives on exactly one).
        pub antipodal_pairs: usize,
    }

    /// Measure the support geometry of an 8-qubit state.
    pub fn support_geometry(state: &dyn Backend<C64>) -> SupportGeometry {
        assert_eq!(state.num_qubits(), 8, "the E8 representation is 8 qubits");
        let mut support: Vec<u8> = Vec::new();
        state.for_each_nonzero(&mut |idx, _| support.push(idx as u8));
        support.sort_unstable();
        let mut sectors = (0, 0);
        for &s in &support {
            if s.count_ones() % 2 == 0 {
                sectors.0 += 1;
            } else {
                sectors.1 += 1;
            }
        }
        let mut root_edges = 0;
        let mut antipodal_pairs = 0;
        for (i, &x) in support.iter().enumerate() {
            for &y in &support[i + 1..] {
                if transition_root(x, y).is_some() {
                    root_edges += 1;
                }
                if x ^ y == 0xff {
                    antipodal_pairs += 1;
                }
            }
        }
        // Affine dimension over GF(2): rank of {s ⊕ s₀}.
        let mut basis: Vec<u8> = Vec::new();
        if let Some(&s0) = support.first() {
            for &s in &support[1..] {
                let mut v = s ^ s0;
                for &b in &basis {
                    let pivot = 7 - b.leading_zeros() as usize;
                    if (v >> pivot) & 1 == 1 {
                        v ^= b;
                    }
                }
                if v != 0 {
                    basis.push(v);
                }
            }
        }
        SupportGeometry {
            points: support.len(),
            sectors,
            root_edges,
            affine_dim: basis.len(),
            antipodal_pairs,
        }
    }

    /// Schmidt rank and entanglement entropy (bits) across the cut
    /// whose LOW side is the qubits set in `cut_mask`, computed by SVD
    /// of the reshaped amplitude matrix.
    pub fn schmidt(state: &dyn Backend<C64>, cut_mask: u8) -> (usize, f64) {
        assert_eq!(state.num_qubits(), 8);
        let low: Vec<usize> = (0..8).filter(|q| (cut_mask >> q) & 1 == 1).collect();
        let high: Vec<usize> = (0..8).filter(|q| (cut_mask >> q) & 1 == 0).collect();
        let (rows, cols) = (1usize << low.len(), 1usize << high.len());
        let mut m = vec![C64::new(0.0, 0.0); rows * cols];
        state.for_each_nonzero(&mut |idx, amp| {
            let mut r = 0usize;
            for (bit, &q) in low.iter().enumerate() {
                r |= (((idx >> q) & 1) as usize) << bit;
            }
            let mut c = 0usize;
            for (bit, &q) in high.iter().enumerate() {
                c |= (((idx >> q) & 1) as usize) << bit;
            }
            m[r * cols + c] = amp;
        });
        let svd = svd_thin::<C64>(rows, cols, &m, 1e-12, 1e-13).expect("C64 svd");
        let weights: Vec<f64> = svd.sigma.iter().map(|s| s * s).collect();
        let total: f64 = weights.iter().sum();
        let mut entropy = 0.0;
        let mut rank = 0;
        for w in weights {
            let p = w / total;
            if p > 1e-12 {
                rank += 1;
                entropy -= p * p.log2();
            }
        }
        (rank, entropy)
    }

    /// The projected-support bound: Schmidt rank across a cut can
    /// never exceed the number of distinct low-side (or high-side)
    /// bit patterns in the support — a geometric bound on
    /// entanglement, checkable against [`schmidt`].
    pub fn projected_support_bound(state: &dyn Backend<C64>, cut_mask: u8) -> usize {
        let mut low = std::collections::HashSet::new();
        let mut high = std::collections::HashSet::new();
        state.for_each_nonzero(&mut |idx, _| {
            low.insert((idx as u8) & cut_mask);
            high.insert((idx as u8) & !cut_mask);
        });
        low.len().min(high.len())
    }
}

/// An embedding of the four arity chains (`A₁ … A₄`, i.e. the su(2),
/// su(3), su(4), su(5) frames of binary/ternary/quaternary/quintary
/// sub-qudits) into the one E8 root system, with the measured overlap
/// structure between them.
#[derive(Debug, Clone)]
pub struct ChainEmbedding {
    /// The chains, indexed by sub-qudit (0 ↦ d=2 … 3 ↦ d=5).
    pub chains: Vec<Vec<Root>>,
}

impl ChainEmbedding {
    /// Build the canonical embedding: each chain is found by search,
    /// orthogonal to as many *earlier* chains as rank permits (the
    /// greedy horizontal layout). The last chain necessarily overlaps —
    /// see [`ChainEmbedding::gram`].
    pub fn canonical() -> Result<Self> {
        let mut chains: Vec<Vec<Root>> = Vec::new();
        for k in 1..=4usize {
            let avoid: Vec<Root> = chains.iter().flatten().copied().collect();
            let chain = match find_a_chain(k, &avoid) {
                Some(c) => c,
                None => {
                    // Rank exhausted: place this chain with the minimal
                    // overlap the geometry allows — orthogonal to all
                    // chains but the largest earlier one.
                    let avoid: Vec<Root> = chains
                        .iter()
                        .take(chains.len().saturating_sub(1))
                        .flatten()
                        .copied()
                        .collect();
                    find_a_chain(k, &avoid).ok_or_else(|| {
                        Error::InvalidState(format!("no A_{k} chain found at all"))
                    })?
                }
            };
            chains.push(chain);
        }
        Ok(ChainEmbedding { chains })
    }

    /// The doubled inner products between every root of chain `i` and
    /// every root of chain `j` — the measured correlation of the two
    /// sub-qudits' frames (all zeros = fully independent).
    pub fn gram(&self, i: usize, j: usize) -> Vec<Vec<i32>> {
        self.chains[i]
            .iter()
            .map(|a| self.chains[j].iter().map(|b| dot(a, b)).collect())
            .collect()
    }

    /// The compound qudit's coupling fabric: `coupling[i][j]` is true
    /// when chains `i` and `j` share any non-orthogonal root pair —
    /// the sub-qudit pairs the E8 geometry correlates.
    pub fn coupling(&self) -> Vec<Vec<bool>> {
        let n = self.chains.len();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| i != j && self.gram(i, j).iter().flatten().any(|&g| g != 0))
                    .collect()
            })
            .collect()
    }
}
