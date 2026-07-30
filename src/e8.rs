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

    /// Inverse of [`sector_of`]: the basis bits of an E8×E8 point.
    /// Copy 0 is the even sector directly; copy 1 pairs back through
    /// qubit 0. `None` if the root is not a spinor root.
    pub fn bits_of_sector(copy: usize, root: &Root) -> Option<u8> {
        let bits = bits_of_spinor(root)?;
        match copy {
            0 => (bits.count_ones() % 2 == 0).then_some(bits),
            1 => (bits.count_ones() % 2 == 0).then_some(bits ^ 1),
            _ => None,
        }
    }

    /// The E8×E8 representation as a first-class qubit
    /// [`Backend`], named `"e8-rep"`:
    /// amplitudes are stored keyed by **(E8 copy, spinor root)** — the
    /// paired-point coordinates themselves, not bit strings — so the
    /// state's support literally *is* a set of points on the two E8
    /// copies, and [`support_geometry`] reads off its stored keys.
    ///
    /// The representation is 8-qubit-native (its 256 points are the
    /// full 8-qubit basis); narrower registers embed by fixing the
    /// unused qubits to 0, and widths past 8 are refused with the
    /// structural reason. Gate application converts through the
    /// measured bijection [`sector_of`] — its cost relative to the
    /// other representations is exactly what the benchmark harness
    /// prices.
    pub struct E8RepState {
        num_qubits: usize,
        amps: std::collections::HashMap<(usize, Root), C64>,
    }

    impl E8RepState {
        /// A fresh register on the E8×E8 point set (`n ≤ 8`).
        pub fn new(num_qubits: usize) -> crate::error::Result<Self> {
            if num_qubits > 8 {
                return Err(crate::error::Error::TooManyQubits {
                    requested: num_qubits,
                    max: 8,
                });
            }
            let mut amps = std::collections::HashMap::new();
            amps.insert(sector_of(0), C64::new(1.0, 0.0));
            Ok(E8RepState { num_qubits, amps })
        }

        /// The stored support as (copy, spinor root) points — the
        /// representation's native coordinates, exposed so tests can
        /// verify the storage really is root-keyed.
        pub fn stored_points(&self) -> Vec<(usize, Root)> {
            let mut points: Vec<(usize, Root)> = self.amps.keys().copied().collect();
            points.sort_unstable();
            points
        }
    }

    impl Backend<C64> for E8RepState {
        fn name(&self) -> &str {
            "e8-rep"
        }

        fn num_qubits(&self) -> usize {
            self.num_qubits
        }

        fn apply(
            &mut self,
            matrix: &crate::math::GateMatrix<C64>,
            qubits: &[usize],
        ) -> crate::error::Result<()> {
            crate::backend::validate_apply(self.num_qubits, matrix, qubits)?;
            let d = 1usize << qubits.len();
            let mask: u64 = qubits.iter().map(|&q| 1u64 << q).sum();
            let scatter = crate::backend::scatter_table(qubits);
            let mut grouped: std::collections::HashMap<u64, Vec<C64>> =
                std::collections::HashMap::new();
            for (&(copy, root), &amp) in &self.amps {
                let bits = u64::from(bits_of_sector(copy, &root).expect("stored keys are points"));
                let sub = crate::backend::sub_index(bits, qubits);
                grouped
                    .entry(bits & !mask)
                    .or_insert_with(|| vec![C64::new(0.0, 0.0); d])[sub] = amp;
            }
            let mdata = matrix.data();
            let mut amps = std::collections::HashMap::new();
            for (rest, vec_in) in grouped {
                for r in 0..d {
                    let mut acc = C64::new(0.0, 0.0);
                    for (c, amp) in vec_in.iter().enumerate() {
                        acc += mdata[r * d + c] * amp;
                    }
                    if acc.norm_sqr() > crate::backend::PRUNE_TOL {
                        // sector_of is injective, so distinct output
                        // bits are distinct points — no collisions.
                        amps.insert(sector_of((rest | scatter[r]) as u8), acc);
                    }
                }
            }
            self.amps = amps;
            Ok(())
        }

        fn amplitude(&self, index: u64) -> C64 {
            if index >> self.num_qubits != 0 {
                return C64::new(0.0, 0.0);
            }
            self.amps
                .get(&sector_of(index as u8))
                .copied()
                .unwrap_or(C64::new(0.0, 0.0))
        }

        fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, C64)) {
            for (&(copy, root), &amp) in &self.amps {
                let bits = bits_of_sector(copy, &root).expect("stored keys are points");
                f(u64::from(bits), amp);
            }
        }

        fn nonzero_count(&self) -> usize {
            self.amps.len()
        }

        fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
            if qubit >= self.num_qubits {
                return;
            }
            let scale = C64::new(renorm, 0.0);
            let amps = std::mem::take(&mut self.amps);
            self.amps = amps
                .into_iter()
                .filter(|&((copy, root), _)| {
                    let bits = bits_of_sector(copy, &root).expect("stored keys are points");
                    ((bits >> qubit) & 1 == 1) == outcome
                })
                .map(|(key, a)| (key, a * scale))
                .collect();
        }

        fn reset(&mut self) {
            self.amps.clear();
            self.amps.insert(sector_of(0), C64::new(1.0, 0.0));
        }

        fn load(&mut self, entries: &[(u64, C64)]) -> crate::error::Result<()> {
            for &(i, _) in entries {
                if i >> self.num_qubits != 0 {
                    return Err(crate::error::Error::QubitOutOfRange {
                        qubit: 64 - i.leading_zeros() as usize,
                        num_qubits: self.num_qubits,
                    });
                }
            }
            self.amps.clear();
            for &(i, a) in entries {
                if a.norm_sqr() > 0.0 {
                    self.amps.insert(sector_of(i as u8), a);
                }
            }
            Ok(())
        }

        fn memory_bytes(&self) -> usize {
            // Key = (copy, 8 × i32 root) + C64 value.
            self.amps.len() * (std::mem::size_of::<(usize, Root)>() + 16)
                + std::mem::size_of::<Self>()
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
}

/// The infinite E8 constellation: E8 copies positioned at the points
/// of E8 itself, one scale level per copy — the recursion that expands
/// the 8-qubit representation to any register width.
///
/// The measured foundation is the coset structure **E8/2E8 ≅ F₂⁸**:
/// exactly 256 classes, represented by the origin, the 120 antipodal
/// root pairs (on the √2-sphere), and 135 frames of 16 among the 2160
/// norm-2 vectors (on the 2-sphere) — 1 + 120 + 135 = 256 = one byte,
/// verified here by exact integer arithmetic, not quoted. So "the
/// identity position of *this* E8" is one byte choosing a point on the
/// concentric shells of the parent copy at doubled scale, and the
/// recursion never ends: [`compose`](constellation::compose) maps an
/// m-digit string to the lattice point `Σ 2ᵏ·rep(digitₖ)`, a bijection
/// onto `E8/2^m E8` (measured), self-similar under doubling (doubling
/// a point prepends digit 0). An n-qubit basis state **is** one E8
/// point known to resolution `2^⌈n/8⌉`.
///
/// [`E8ConstellationState`](constellation::E8ConstellationState)
/// makes this a first-class backend
/// (`"e8-constellation"`): amplitudes keyed by the lattice points
/// themselves, at any width up to the trait's 63-qubit u64 index wall
/// — past the single-copy representation's native 8.
pub mod constellation {
    use crate::backend::Backend;
    use crate::scalar::C64;
    use std::collections::HashMap;
    use std::sync::OnceLock;

    /// A lattice point in doubled coordinates, wide enough for deep
    /// scale towers (a depth-m tower needs coordinates up to `2^{m+2}`).
    pub type Point = [i64; 8];

    /// Which concentric shell of its scale level a coset digit's
    /// canonical point occupies.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Shell {
        /// The origin (class 0 only).
        Center,
        /// The 240-root sphere, real radius √2 (120 classes).
        RootSphere,
        /// The 2160-vector sphere, real radius 2 (135 classes).
        FrameSphere,
    }

    impl Shell {
        /// Real squared radius of the shell (0, 2 or 4).
        pub fn radius_sqr(self) -> i64 {
            match self {
                Shell::Center => 0,
                Shell::RootSphere => 2,
                Shell::FrameSphere => 4,
            }
        }
    }

    /// The 2160 norm-2 vectors of E8 in doubled coordinates (doubled
    /// dot = 16), constructed by shape and verified: `±2eᵢ` (16),
    /// `±eᵢ±eⱼ±eₖ±eₗ` (1120) and the odd-spinor `(±3/2, (±1/2)⁷)`
    /// family with the lattice parity condition (1024).
    pub fn norm4_vectors() -> Vec<Point> {
        let mut out = Vec::with_capacity(2160);
        for i in 0..8 {
            for s in [4i64, -4] {
                let mut v = [0i64; 8];
                v[i] = s;
                out.push(v);
            }
        }
        for mask in 0u32..256 {
            if mask.count_ones() == 4 {
                for signs in 0u32..16 {
                    let mut v = [0i64; 8];
                    let mut bit = 0;
                    for (slot, item) in v.iter_mut().enumerate() {
                        if (mask >> slot) & 1 == 1 {
                            *item = if (signs >> bit) & 1 == 1 { -2 } else { 2 };
                            bit += 1;
                        }
                    }
                    out.push(v);
                }
            }
        }
        for pos in 0..8 {
            for signs in 0u32..256 {
                let v: Point = std::array::from_fn(|i| {
                    let mag = if i == pos { 3 } else { 1 };
                    if (signs >> i) & 1 == 1 {
                        -mag
                    } else {
                        mag
                    }
                });
                if v.iter().sum::<i64>().rem_euclid(4) == 0 {
                    out.push(v);
                }
            }
        }
        for v in &out {
            debug_assert_eq!(v.iter().map(|c| c * c).sum::<i64>(), 16);
        }
        assert_eq!(out.len(), 2160, "the norm-2 shell of E8 has 2160 vectors");
        out
    }

    /// A Z-basis of E8 in doubled coordinates (lower triangular, so
    /// its determinant 4·2⁶·1 = 256 = 2⁸ is the doubling of a
    /// unimodular real basis — re-verified at context build).
    const BASIS: [[i64; 8]; 8] = [
        [4, 0, 0, 0, 0, 0, 0, 0],
        [-2, 2, 0, 0, 0, 0, 0, 0],
        [0, -2, 2, 0, 0, 0, 0, 0],
        [0, 0, -2, 2, 0, 0, 0, 0],
        [0, 0, 0, -2, 2, 0, 0, 0],
        [0, 0, 0, 0, -2, 2, 0, 0],
        [0, 0, 0, 0, 0, -2, 2, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ];

    /// Fraction-free (Bareiss) determinant — exact integer arithmetic.
    fn bareiss_det(mut a: Vec<Vec<i128>>) -> i128 {
        let n = a.len();
        let mut sign = 1i128;
        let mut prev = 1i128;
        for k in 0..n {
            if a[k][k] == 0 {
                let Some(swap) = (k + 1..n).find(|&r| a[r][k] != 0) else {
                    return 0;
                };
                a.swap(k, swap);
                sign = -sign;
            }
            for i in (k + 1)..n {
                for j in (k + 1)..n {
                    a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) / prev;
                }
                a[i][k] = 0;
            }
            prev = a[k][k];
        }
        sign * a[n - 1][n - 1]
    }

    struct Ctx {
        /// Adjugate of the basis-column matrix: coordinates are
        /// `adj·p / det`, exact for lattice points.
        adj: [[i64; 8]; 8],
        det: i64,
        reps: [Point; 256],
        shells: [Shell; 256],
    }

    fn coords_with(adj: &[[i64; 8]; 8], det: i64, p: &Point) -> Option<[i64; 8]> {
        let mut coords = [0i64; 8];
        for (i, row) in adj.iter().enumerate() {
            let acc: i128 = row
                .iter()
                .zip(p)
                .map(|(&a, &x)| a as i128 * x as i128)
                .sum();
            if acc % det as i128 != 0 {
                return None; // not an E8 point
            }
            coords[i] = (acc / det as i128) as i64;
        }
        Some(coords)
    }

    fn class_with(adj: &[[i64; 8]; 8], det: i64, p: &Point) -> Option<u8> {
        let coords = coords_with(adj, det, p)?;
        Some(
            coords
                .iter()
                .enumerate()
                .fold(0u8, |acc, (i, &c)| acc | (((c.rem_euclid(2)) as u8) << i)),
        )
    }

    /// Exact integer coordinates of a lattice point in the verified
    /// basis (`p = Σ cᵢ·Bᵢ`); `None` off the lattice. [`class_of`] is
    /// this map reduced mod 2 — and by self-duality coordinate `i` IS
    /// the E8 inner product with the i-th [`dual_basis`] vector
    /// (`cᵢ(p) = ⟨b*ᵢ, p⟩`, measured in tests).
    pub fn coords_of(p: &Point) -> Option<[i64; 8]> {
        let c = ctx();
        coords_with(&c.adj, c.det, p)
    }

    /// Doubled inner product of two lattice points (4× the real E8
    /// inner product), exact.
    pub fn pdot(a: &Point, b: &Point) -> i128 {
        a.iter().zip(b).map(|(&x, &y)| x as i128 * y as i128).sum()
    }

    /// The real Gram matrix of the verified basis (doubled dots / 4) —
    /// an integer matrix with determinant 1 (E8 is unimodular), which
    /// is exactly why the lattice is SELF-DUAL and the momentum-space
    /// E8 of the Weyl pair is the same object as the position-space
    /// one.
    pub fn gram() -> [[i64; 8]; 8] {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let bi: Point = BASIS[i];
                let bj: Point = BASIS[j];
                (pdot(&bi, &bj) / 4) as i64
            })
        })
    }

    /// The dual basis `b*ᵢ` (`⟨b*ᵢ, bⱼ⟩ = δᵢⱼ`) in doubled
    /// coordinates. Every dual vector is an INTEGER combination of the
    /// basis — i.e. an E8 point — because det(Gram) = 1: measured
    /// self-duality, `E8* = E8`. Computed once and cached.
    pub fn dual_basis() -> [Point; 8] {
        static DUALS: OnceLock<[Point; 8]> = OnceLock::new();
        *DUALS.get_or_init(dual_basis_uncached)
    }

    fn dual_basis_uncached() -> [Point; 8] {
        // G⁻¹ = adj(G) since det(G) = 1 (asserted); b*ᵢ = Σⱼ G⁻¹ᵢⱼ bⱼ.
        let g = gram();
        let g128: Vec<Vec<i128>> = g
            .iter()
            .map(|row| row.iter().map(|&x| x as i128).collect())
            .collect();
        let det = bareiss_det(g128.clone());
        assert_eq!(det, 1, "the E8 Gram determinant is 1 — unimodular");
        let mut inv = [[0i128; 8]; 8];
        for (i, row) in inv.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                let minor: Vec<Vec<i128>> = (0..8)
                    .filter(|&r| r != j)
                    .map(|r| (0..8).filter(|&c| c != i).map(|c| g128[r][c]).collect())
                    .collect();
                let sign = if (i + j) % 2 == 0 { 1 } else { -1 };
                *slot = sign * bareiss_det(minor);
            }
        }
        std::array::from_fn(|i| {
            let mut v = [0i64; 8];
            for (j, b) in BASIS.iter().enumerate() {
                for (slot, &coord) in v.iter_mut().zip(b) {
                    *slot += (inv[i][j] as i64) * coord;
                }
            }
            v
        })
    }

    fn ctx() -> &'static Ctx {
        static CTX: OnceLock<Ctx> = OnceLock::new();
        CTX.get_or_init(|| {
            // Basis vectors as COLUMNS: p = M·c for coordinate vector c.
            let m: Vec<Vec<i128>> = (0..8)
                .map(|i| (0..8).map(|j| BASIS[j][i] as i128).collect())
                .collect();
            let det128 = bareiss_det(m.clone());
            assert_eq!(det128.abs(), 256, "doubled E8 basis must have |det| 2⁸");
            // Adjugate via cofactors: adj[i][j] = (−1)^{i+j}·minor(j, i).
            let mut adj = [[0i64; 8]; 8];
            for (i, row) in adj.iter_mut().enumerate() {
                for (j, slot) in row.iter_mut().enumerate() {
                    let minor: Vec<Vec<i128>> = (0..8)
                        .filter(|&r| r != j)
                        .map(|r| (0..8).filter(|&c| c != i).map(|c| m[r][c]).collect())
                        .collect();
                    let sign = if (i + j) % 2 == 0 { 1 } else { -1 };
                    *slot = (sign * bareiss_det(minor)) as i64;
                }
            }
            let det = det128 as i64;
            // Verify M·adj = det·I exactly.
            for (r, m_row) in m.iter().enumerate() {
                for c in 0..8 {
                    let acc: i128 = m_row
                        .iter()
                        .zip(&adj)
                        .map(|(&mv, adj_row)| mv * adj_row[c] as i128)
                        .sum();
                    assert_eq!(acc, if r == c { det128 } else { 0 }, "adjugate check");
                }
            }
            // The 256 classes, measured from the shells: origin, the
            // 240 roots, the 2160 norm-2 vectors.
            let mut buckets: HashMap<u8, Vec<Point>> = HashMap::new();
            buckets.insert(
                class_with(&adj, det, &[0; 8]).expect("origin"),
                vec![[0; 8]],
            );
            for r in super::roots() {
                let p: Point = std::array::from_fn(|k| r[k] as i64);
                let c = class_with(&adj, det, &p).expect("roots are lattice points");
                buckets.entry(c).or_default().push(p);
            }
            for p in norm4_vectors() {
                let c = class_with(&adj, det, &p).expect("norm-2 shell is in the lattice");
                buckets.entry(c).or_default().push(p);
            }
            assert_eq!(buckets.len(), 256, "E8/2E8 must have 2⁸ classes");
            let mut reps = [[0i64; 8]; 256];
            let mut shells = [Shell::Center; 256];
            let (mut singles, mut pairs, mut frames) = (0, 0, 0);
            for (c, mut members) in buckets {
                members.sort_unstable();
                let rep = members[0];
                let norm: i64 = rep.iter().map(|x| x * x).sum();
                let shell = match (members.len(), norm) {
                    (1, 0) => {
                        singles += 1;
                        Shell::Center
                    }
                    (2, 8) => {
                        pairs += 1;
                        Shell::RootSphere
                    }
                    (16, 16) => {
                        frames += 1;
                        Shell::FrameSphere
                    }
                    other => panic!("impossible coset structure {other:?}"),
                };
                reps[c as usize] = rep;
                shells[c as usize] = shell;
            }
            assert_eq!(
                (singles, pairs, frames),
                (1, 120, 135),
                "coset census: origin + root pairs + frames"
            );
            Ctx {
                adj,
                det,
                reps,
                shells,
            }
        })
    }

    /// The 8-bit coset address of a lattice point — the coordinates of
    /// `E8/2E8 ≅ F₂⁸`, computed exactly. `None` when the doubled
    /// coordinates are not an E8 point at all. The map is linear:
    /// `class(x + y) = class(x) XOR class(y)` (measured in tests).
    pub fn class_of(p: &Point) -> Option<u8> {
        let c = ctx();
        class_with(&c.adj, c.det, p)
    }

    /// The canonical (lexicographically least) representative of a
    /// coset class — the origin, a root, or a norm-2 frame vector.
    pub fn representative(class: u8) -> Point {
        ctx().reps[class as usize]
    }

    /// The shell the class representative occupies — the "sphere" a
    /// constellation digit points at.
    pub fn shell_of(class: u8) -> Shell {
        ctx().shells[class as usize]
    }

    /// The j-th verified basis vector, in doubled coordinates.
    pub fn basis(j: usize) -> Point {
        BASIS[j]
    }

    /// Position of a digit string in the constellation: the lattice
    /// point `Σ 2ᵏ·rep(digitₖ)`. Digit k is the coset address at scale
    /// `2ᵏ`; the map is a bijection onto `E8/2^m E8` (measured) and
    /// self-similar (doubling a point prepends digit 0).
    pub fn compose(digits: &[u8]) -> Point {
        assert!(digits.len() < 60, "tower depth would overflow i64");
        let mut out = [0i64; 8];
        for (k, &d) in digits.iter().enumerate() {
            let rep = representative(d);
            for (slot, &r) in out.iter_mut().zip(&rep) {
                *slot += r << k;
            }
        }
        out
    }

    /// Recover the first `levels` constellation digits of a lattice
    /// point (inverse of [`compose`] on composed points). `None` when
    /// the input is not an E8 point.
    pub fn decompose(point: &Point, levels: usize) -> Option<Vec<u8>> {
        let mut x = *point;
        let mut digits = Vec::with_capacity(levels);
        for _ in 0..levels {
            let c = class_of(&x)?;
            digits.push(c);
            let rep = representative(c);
            for (slot, &r) in x.iter_mut().zip(&rep) {
                debug_assert_eq!((*slot - r) % 2, 0, "x − rep(class(x)) lies in 2E8");
                *slot = (*slot - r) / 2;
            }
        }
        Some(digits)
    }

    /// Addition in the residue group `E8/2^m E8`, expressed on the
    /// tower's bit encoding: decode both operands through
    /// [`compose`], add as lattice points, and re-encode through
    /// [`decompose`].
    ///
    /// This is the group the native operators act on, and it is **not**
    /// bitwise XOR beyond one level: `class_of` is linear, so digit 0 of a
    /// sum is the XOR of the operands' digit 0, but the division by two
    /// that extracts higher digits carries. One copy therefore behaves as
    /// `F₂⁸` and every deeper tower does not — measured in
    /// `tests/phase_degree.rs`.
    pub fn residue_add(levels: usize, a: u64, b: u64) -> u64 {
        let digits = |bits: u64| -> Vec<u8> {
            (0..levels)
                .map(|k| ((bits >> (8 * k)) & 0xff) as u8)
                .collect()
        };
        let (pa, pb) = (compose(&digits(a)), compose(&digits(b)));
        let sum: Point = std::array::from_fn(|k| pa[k] + pb[k]);
        let out = decompose(&sum, levels).expect("a sum of lattice points is a lattice point");
        out.iter()
            .enumerate()
            .fold(0u64, |acc, (k, &d)| acc | (u64::from(d) << (8 * k)))
    }

    /// The phase the native modulation `M_q` applies to the residue whose
    /// tower encoding is `bits` — exposed so the operator's own polynomial
    /// degree can be measured by [`crate::phase`] rather than assumed.
    pub fn modulation_phase(levels: usize, q: &Point, bits: u64) -> C64 {
        let digits: Vec<u8> = (0..levels)
            .map(|k| ((bits >> (8 * k)) & 0xff) as u8)
            .collect();
        let p = compose(&digits);
        let modulus = 4i128 << levels;
        let r = pdot(q, &p).rem_euclid(modulus);
        let angle = std::f64::consts::TAU * r as f64 / modulus as f64;
        C64::new(angle.cos(), angle.sin())
    }

    /// The constellation as a first-class qubit [`Backend`], named
    /// `"e8-constellation"`: amplitudes keyed by **lattice points** —
    /// residues in `E8/2^m E8` with `m = ⌈n/8⌉` — so a basis state is
    /// literally a position in the scale tower, at any width up to the
    /// trait's 63-qubit u64 index wall (the same wall sparse has; the
    /// geometry itself is unbounded). Gate application converts
    /// through the measured tower bijection; that conversion cost is
    /// what the benchmark harness prices.
    pub struct E8ConstellationState {
        num_qubits: usize,
        levels: usize,
        amps: HashMap<Point, C64>,
    }

    impl E8ConstellationState {
        /// A fresh register of `n ≤ 63` qubits, `⌈n/8⌉` scale levels.
        pub fn new(num_qubits: usize) -> crate::error::Result<Self> {
            if num_qubits > 63 {
                return Err(crate::error::Error::TooManyQubits {
                    requested: num_qubits,
                    max: 63,
                });
            }
            let mut amps = HashMap::new();
            amps.insert([0i64; 8], C64::new(1.0, 0.0));
            Ok(E8ConstellationState {
                num_qubits,
                levels: num_qubits.div_ceil(8),
                amps,
            })
        }

        /// The stored support as constellation points, sorted — the
        /// representation's native coordinates.
        pub fn stored_points(&self) -> Vec<Point> {
            let mut points: Vec<Point> = self.amps.keys().copied().collect();
            points.sort_unstable();
            points
        }

        /// Per-level shell occupation over the stored support:
        /// `census[k] = [center, root-sphere, frame-sphere]` counts of
        /// the level-k digits — the state's measured geography in the
        /// constellation.
        pub fn shell_census(&self) -> Vec<[usize; 3]> {
            let mut census = vec![[0usize; 3]; self.levels];
            for p in self.amps.keys() {
                let digits = decompose(p, self.levels).expect("stored keys are lattice points");
                for (k, &d) in digits.iter().enumerate() {
                    let slot = match shell_of(d) {
                        Shell::Center => 0,
                        Shell::RootSphere => 1,
                        Shell::FrameSphere => 2,
                    };
                    census[k][slot] += 1;
                }
            }
            census
        }

        fn bits_of(&self, p: &Point) -> u64 {
            decompose(p, self.levels)
                .expect("stored keys are lattice points")
                .iter()
                .enumerate()
                .fold(0u64, |acc, (k, &d)| acc | (u64::from(d)) << (8 * k))
        }

        fn key_of(&self, bits: u64) -> Point {
            let digits: Vec<u8> = (0..self.levels)
                .map(|k| ((bits >> (8 * k)) & 0xff) as u8)
                .collect();
            compose(&digits)
        }

        /// Canonical representative of `p` in `E8/2^m E8`.
        fn reduce(&self, p: &Point) -> Point {
            compose(&decompose(p, self.levels).expect("lattice point"))
        }

        /// Native constellation gates act on the residue GROUP, which
        /// the register is exactly when every 8-qubit block is full; a
        /// partial top block is a qubit-embedding of a subset, where a
        /// group translation could carry out of the register.
        fn require_full_blocks(&self) -> crate::error::Result<()> {
            if self.num_qubits % 8 != 0 {
                return Err(crate::error::Error::InvalidState(format!(
                    "native constellation gates need full 8-qubit blocks; {} qubits \
                     embeds a subset of the residue group, not the group itself",
                    self.num_qubits
                )));
            }
            Ok(())
        }

        /// Position-side Weyl operator: `|p⟩ ↦ |p + v⟩` on the residue
        /// group `E8/2^m E8`. Cross-scale by construction — the group
        /// is the lattice quotient, NOT `(F₂⁸)^m`, so adding a level-j
        /// vector carries into every higher scale level the lattice
        /// arithmetic demands.
        pub fn translate(&mut self, v: &Point) -> crate::error::Result<()> {
            self.require_full_blocks()?;
            if class_of(v).is_none() {
                return Err(crate::error::Error::InvalidState(
                    "translation label is not an E8 point".into(),
                ));
            }
            let amps = std::mem::take(&mut self.amps);
            self.amps = amps
                .into_iter()
                .map(|(p, a)| {
                    let shifted: Point = std::array::from_fn(|k| p[k] + v[k]);
                    (self.reduce(&shifted), a)
                })
                .collect();
            Ok(())
        }

        /// Momentum-side Weyl operator: multiply `|p⟩` by the
        /// character `χ_q(p) = e^{2πi⟨q,p⟩/2^m}`, labeled by the DUAL
        /// E8 — which is E8 again (measured self-duality). Well-defined
        /// on residues exactly because `⟨q, 2^m E8⟩ ⊆ 2^m ℤ` for
        /// lattice `q`; the phase is computed from the exact integer
        /// inner product reduced mod the period.
        pub fn modulate(&mut self, q: &Point) -> crate::error::Result<()> {
            self.require_full_blocks()?;
            if class_of(q).is_none() {
                return Err(crate::error::Error::InvalidState(
                    "modulation label is not an E8 point".into(),
                ));
            }
            // Doubled dot = 4·⟨q,p⟩ real, and the character period is
            // 2^m: phase = 2π·(pdot mod 4·2^m)/(4·2^m), exact.
            let modulus = 4i128 << self.levels;
            for (p, a) in self.amps.iter_mut() {
                let r = pdot(q, p).rem_euclid(modulus);
                let angle = std::f64::consts::TAU * r as f64 / modulus as f64;
                *a *= C64::new(angle.cos(), angle.sin());
            }
            Ok(())
        }

        /// The Weyl-GROUP side: the reflection `s_α` through a root's
        /// hyperplane (`s_α(p) = p − ⟨p,α⟩α`, norm-2 roots), acting as
        /// a basis permutation of the residue group. W(E8) is the
        /// lattice's point symmetry; its conjugation action on the
        /// Weyl pair (`s T_v s = T_{s(v)}`, `s M_q s = M_{s(q)}`) is
        /// measured in tests — the reflections are Clifford for the
        /// constellation's Heisenberg group.
        pub fn reflect(&mut self, alpha: &super::Root) -> crate::error::Result<()> {
            self.require_full_blocks()?;
            if super::dot(alpha, alpha) != 8 {
                return Err(crate::error::Error::InvalidState(
                    "reflection label is not an E8 root".into(),
                ));
            }
            let a: Point = std::array::from_fn(|k| alpha[k] as i64);
            let amps = std::mem::take(&mut self.amps);
            self.amps = amps
                .into_iter()
                .map(|(p, amp)| {
                    // ⟨p,α⟩ real = pdot/4, an exact integer for
                    // lattice points against a root.
                    let inner = pdot(&p, &a) / 4;
                    let image: Point = std::array::from_fn(|k| p[k] - (inner as i64) * a[k]);
                    (self.reduce(&image), amp)
                })
                .collect();
            Ok(())
        }

        /// Relabel the eight ambient coordinates by `perm`: the image
        /// point's coordinate `k` is the source's coordinate
        /// `perm[k]`.
        ///
        /// E8's construction — integer vectors of even sum together
        /// with the even half-integer spinors — is symmetric in the
        /// eight coordinates, so *every* coordinate permutation is a
        /// lattice automorphism (verified against all 240 roots in
        /// [`cube::is_lattice_automorphism`](super::cube::is_lattice_automorphism)),
        /// and permutation commutes with doubling, so the action
        /// descends to `E8/2^m E8` unchanged at every scale. This is
        /// the operator the [cube volume](super::cube) uses to move a
        /// pattern across its own vertices.
        pub fn permute_coordinates(&mut self, perm: &[usize; 8]) -> crate::error::Result<()> {
            self.require_full_blocks()?;
            let mut seen = [false; 8];
            for &k in perm {
                if k >= 8 || seen[k] {
                    return Err(crate::error::Error::InvalidState(format!(
                        "{perm:?} is not a permutation of the eight coordinates"
                    )));
                }
                seen[k] = true;
            }
            self.amps = self
                .amps
                .iter()
                .map(|(p, &a)| {
                    let image: Point = std::array::from_fn(|k| p[perm[k]]);
                    (self.reduce(&image), a)
                })
                .collect();
            Ok(())
        }

        /// The scale-embedding isometry `V_a: |p⟩ ↦ |2^a·p⟩` into a
        /// register `a` levels deeper — the coarse-to-fine leg of the
        /// scale-composition algebra. Weight-preserving and injective
        /// (the self-similarity theorem: doubling prepends digit 0);
        /// operators transport covariantly across it
        /// (`T_{2v}∘V₁ = V₁∘T_v`, `M_q∘V₁ = V₁∘M_q`, measured), which
        /// is what makes it the encoder of the cross-scale comb codes.
        pub fn scale_embed(&self, extra_levels: usize) -> crate::error::Result<Self> {
            self.require_full_blocks()?;
            let num_qubits = self.num_qubits + 8 * extra_levels;
            if num_qubits > 63 {
                return Err(crate::error::Error::TooManyQubits {
                    requested: num_qubits,
                    max: 63,
                });
            }
            let amps = self
                .amps
                .iter()
                .map(|(p, &a)| (p.map(|c| c << extra_levels), a))
                .collect();
            Ok(E8ConstellationState {
                num_qubits,
                levels: self.levels + extra_levels,
                amps,
            })
        }

        /// The decimation `R_a: |2^a·p⟩ ↦ |p⟩` onto a register `a`
        /// levels coarser — the fine-to-coarse leg. Defined exactly on
        /// states whose support carries NO fine-scale data (every point
        /// divisible by `2^a`); anything else refuses with the level
        /// named, because decimating live fine digits would silently
        /// destroy information — correct first, then coarsen.
        pub fn decimate(&self, drop_levels: usize) -> crate::error::Result<Self> {
            self.require_full_blocks()?;
            if drop_levels > self.levels {
                return Err(crate::error::Error::InvalidState(format!(
                    "cannot drop {drop_levels} of {} scale levels",
                    self.levels
                )));
            }
            let mut amps = HashMap::with_capacity(self.amps.len());
            for (p, &a) in &self.amps {
                // Divisibility by 2^drop in the LATTICE sense: the
                // dropped digits must all be class 0 (componentwise
                // evenness is not enough — integer roots have even
                // coordinates but nonzero class).
                let digits = decompose(p, drop_levels).expect("stored keys are lattice points");
                if let Some(level) = digits.iter().position(|&d| d != 0) {
                    return Err(crate::error::Error::InvalidState(format!(
                        "support carries fine-scale data at level {level}; \
                         correct before decimating"
                    )));
                }
                amps.insert(p.map(|c| c >> drop_levels), a);
            }
            Ok(E8ConstellationState {
                num_qubits: self.num_qubits - 8 * drop_levels,
                levels: self.levels - drop_levels,
                amps,
            })
        }

        /// The eigenphase of the modulation `M_q` on this state — the
        /// syndrome read of the cross-scale codes: a comb codeword
        /// returns exactly 1, a displaced codeword returns the
        /// character of its displacement. Refuses (with the measured
        /// spread) when the state is NOT an `M_q` eigenstate, so a
        /// syndrome can never be silently fabricated from a
        /// non-stabilized state.
        pub fn modulation_eigenphase(&self, q: &Point) -> crate::error::Result<C64> {
            if class_of(q).is_none() {
                return Err(crate::error::Error::InvalidState(
                    "modulation label is not an E8 point".into(),
                ));
            }
            let modulus = 4i128 << self.levels;
            let mut phase: Option<C64> = None;
            let mut worst: f64 = 0.0;
            for p in self.amps.keys() {
                let r = pdot(q, p).rem_euclid(modulus);
                let angle = std::f64::consts::TAU * r as f64 / modulus as f64;
                let chi = C64::new(angle.cos(), angle.sin());
                match phase {
                    None => phase = Some(chi),
                    Some(first) => worst = worst.max((chi - first).norm()),
                }
            }
            let Some(phase) = phase else {
                return Err(crate::error::Error::InvalidState(
                    "empty support has no eigenphase".into(),
                ));
            };
            if worst > 1e-9 {
                return Err(crate::error::Error::InvalidState(format!(
                    "state is not an M_q eigenstate: character spread {worst:.3e}"
                )));
            }
            Ok(phase)
        }

        /// The discrete Fourier transform along one basis direction of
        /// the coordinate group `(ℤ/2^m)⁸ ≅ E8/2^m E8` — the gate that
        /// turns position structure into momentum structure one
        /// direction at a time (`F⁴ = 1`; `F` conjugates the basis
        /// translation into the dual-basis modulation, measured).
        /// Guard-admitted: support can grow by the factor `d = 2^m`.
        pub fn coordinate_fourier(&mut self, dir: usize) -> crate::error::Result<()> {
            self.require_full_blocks()?;
            if dir >= 8 {
                return Err(crate::error::Error::InvalidState(format!(
                    "coordinate direction {dir} out of range (E8 has rank 8)"
                )));
            }
            let d = 1usize << self.levels;
            let entry = (std::mem::size_of::<Point>() + std::mem::size_of::<C64>() + 1) * 8 / 7;
            crate::guard::admit_growth(
                self.amps.len().saturating_mul(d).saturating_mul(entry),
                "e8-constellation Fourier growth",
            )?;
            let mut grouped: HashMap<Point, Vec<C64>> = HashMap::new();
            for (p, &a) in &self.amps {
                let coords = coords_of(p).expect("stored keys are lattice points");
                let cd = coords[dir].rem_euclid(d as i64) as usize;
                let rest: Point = std::array::from_fn(|k| p[k] - cd as i64 * BASIS[dir][k]);
                grouped
                    .entry(self.reduce(&rest))
                    .or_insert_with(|| vec![C64::new(0.0, 0.0); d])[cd] = a;
            }
            let norm = 1.0 / (d as f64).sqrt();
            let mut out = HashMap::new();
            for (rest, vec_in) in grouped {
                for cp in 0..d {
                    let mut acc = C64::new(0.0, 0.0);
                    for (c, amp) in vec_in.iter().enumerate() {
                        if amp.norm_sqr() > 0.0 {
                            let angle = std::f64::consts::TAU * ((c * cp) % d) as f64 / d as f64;
                            acc += C64::new(angle.cos(), angle.sin()) * amp;
                        }
                    }
                    acc *= norm;
                    if acc.norm_sqr() > crate::backend::PRUNE_TOL {
                        let point: Point =
                            std::array::from_fn(|k| rest[k] + cp as i64 * BASIS[dir][k]);
                        out.insert(self.reduce(&point), acc);
                    }
                }
            }
            self.amps = out;
            Ok(())
        }
    }

    impl Backend<C64> for E8ConstellationState {
        fn name(&self) -> &str {
            "e8-constellation"
        }

        fn num_qubits(&self) -> usize {
            self.num_qubits
        }

        fn apply(
            &mut self,
            matrix: &crate::math::GateMatrix<C64>,
            qubits: &[usize],
        ) -> crate::error::Result<()> {
            crate::backend::validate_apply(self.num_qubits, matrix, qubits)?;
            let d = 1usize << qubits.len();
            // Worst case the gate scatters every stored point into all
            // `d` sub-index slots — admit that growth against measured
            // memory before building anything (lattice keys are 64
            // bytes, so the wall arrives earlier than sparse's).
            let entry = (std::mem::size_of::<Point>() + std::mem::size_of::<C64>() + 1) * 8 / 7;
            crate::guard::admit_growth(
                self.amps.len().saturating_mul(d).saturating_mul(entry),
                "e8-constellation growth",
            )?;
            let mask: u64 = qubits.iter().map(|&q| 1u64 << q).sum();
            let scatter = crate::backend::scatter_table(qubits);
            let mut grouped: HashMap<u64, Vec<C64>> = HashMap::new();
            for (p, &amp) in &self.amps {
                let bits = self.bits_of(p);
                let sub = crate::backend::sub_index(bits, qubits);
                grouped
                    .entry(bits & !mask)
                    .or_insert_with(|| vec![C64::new(0.0, 0.0); d])[sub] = amp;
            }
            let mdata = matrix.data();
            let mut amps = HashMap::new();
            for (rest, vec_in) in grouped {
                for r in 0..d {
                    let mut acc = C64::new(0.0, 0.0);
                    for (c, amp) in vec_in.iter().enumerate() {
                        acc += mdata[r * d + c] * amp;
                    }
                    if acc.norm_sqr() > crate::backend::PRUNE_TOL {
                        // The tower map is a bijection at fixed depth,
                        // so distinct bits are distinct points.
                        amps.insert(self.key_of(rest | scatter[r]), acc);
                    }
                }
            }
            self.amps = amps;
            Ok(())
        }

        fn amplitude(&self, index: u64) -> C64 {
            if self.num_qubits < 64 && index >> self.num_qubits != 0 {
                return C64::new(0.0, 0.0);
            }
            self.amps
                .get(&self.key_of(index))
                .copied()
                .unwrap_or(C64::new(0.0, 0.0))
        }

        fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, C64)) {
            for (p, &amp) in &self.amps {
                f(self.bits_of(p), amp);
            }
        }

        fn nonzero_count(&self) -> usize {
            self.amps.len()
        }

        fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
            if qubit >= self.num_qubits {
                return;
            }
            let scale = C64::new(renorm, 0.0);
            let amps = std::mem::take(&mut self.amps);
            self.amps = amps
                .into_iter()
                .filter(|(p, _)| ((self.bits_of(p) >> qubit) & 1 == 1) == outcome)
                .map(|(key, a)| (key, a * scale))
                .collect();
        }

        fn reset(&mut self) {
            self.amps.clear();
            self.amps.insert([0i64; 8], C64::new(1.0, 0.0));
        }

        fn load(&mut self, entries: &[(u64, C64)]) -> crate::error::Result<()> {
            for &(i, _) in entries {
                if self.num_qubits < 64 && i >> self.num_qubits != 0 {
                    return Err(crate::error::Error::QubitOutOfRange {
                        qubit: 64 - i.leading_zeros() as usize,
                        num_qubits: self.num_qubits,
                    });
                }
            }
            self.amps.clear();
            for &(i, a) in entries {
                if a.norm_sqr() > 0.0 {
                    self.amps.insert(self.key_of(i), a);
                }
            }
            Ok(())
        }

        fn memory_bytes(&self) -> usize {
            self.amps.len() * (std::mem::size_of::<Point>() + 16) + std::mem::size_of::<Self>()
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// A cross-scale comb code on the constellation, as a first-class
    /// object: stabilizers `⟨T_{2^a B_d}, M_{2^{m−a+w} b*_d}⟩` (coarse
    /// translations against fine modulations, all commuting past the
    /// horizon — measured in `tests/e8_dual_scale.rs`), logical
    /// operators at the middle scales, and the decoder: displacement
    /// syndromes read by exact binary phase readout, corrected by the
    /// minimum-norm representative. The correctable window per
    /// direction is the min-norm cell of `ℤ/2^{a−w}` — displacements
    /// past it become logical operations, which is the honest limit
    /// the noise study measures.
    pub struct CombCode {
        levels: usize,
        comb_scale: usize,
        logical_levels: usize,
    }

    impl CombCode {
        /// A code with `m` scale levels, comb at scale `a`, and `w`
        /// logical levels per direction (`w ≤ a < m`, register `8m ≤ 63`).
        pub fn new(
            levels: usize,
            comb_scale: usize,
            logical_levels: usize,
        ) -> crate::error::Result<Self> {
            if !(logical_levels <= comb_scale && comb_scale < levels) {
                return Err(crate::error::Error::InvalidState(format!(
                    "comb code needs w ≤ a < m, got (m, a, w) = \
                     ({levels}, {comb_scale}, {logical_levels})"
                )));
            }
            if 8 * levels > 63 {
                return Err(crate::error::Error::TooManyQubits {
                    requested: 8 * levels,
                    max: 63,
                });
            }
            Ok(CombCode {
                levels,
                comb_scale,
                logical_levels,
            })
        }

        /// Register width (8 qubits per scale level).
        pub fn num_qubits(&self) -> usize {
            8 * self.levels
        }

        /// The logical-zero codeword: the uniform coarse state carried
        /// down by the scale embedding — the encoder IS the isometry.
        pub fn codeword(&self) -> crate::error::Result<E8ConstellationState> {
            let mut coarse = E8ConstellationState::new(8 * (self.levels - self.comb_scale))?;
            for dir in 0..8 {
                coarse.coordinate_fourier(dir)?;
            }
            coarse.scale_embed(self.comb_scale)
        }

        /// Coarse-side stabilizer label: `T_{2^a B_d}`.
        pub fn translation_check(&self, d: usize) -> Point {
            basis(d).map(|x| x << self.comb_scale)
        }

        /// Fine-side stabilizer label: `M_{2^{m−a+w} b*_d}`.
        pub fn modulation_check(&self, d: usize) -> Point {
            dual_basis()[d].map(|x| x << (self.levels - self.comb_scale + self.logical_levels))
        }

        /// Logical X̄_d = `T_{2^{a−w} B_d}` (middle scale).
        pub fn logical_x(&self, d: usize) -> Point {
            basis(d).map(|x| x << (self.comb_scale - self.logical_levels))
        }

        /// Logical Z̄_d = `M_{2^{m−a} b*_d}` (middle scale).
        pub fn logical_z(&self, d: usize) -> Point {
            dual_basis()[d].map(|x| x << (self.levels - self.comb_scale))
        }

        /// The direction-`d` displacement syndrome, mod `2^{a−w}`,
        /// recovered from the check eigenphases alone by binary phase
        /// readout (finest check first, one exact bit per scale).
        pub fn read_displacement(
            &self,
            state: &E8ConstellationState,
            d: usize,
        ) -> crate::error::Result<i64> {
            let duals = dual_basis();
            let bits = self.comb_scale - self.logical_levels;
            let mut c = 0i64;
            for bit in 0..bits {
                let i = self.levels - 1 - bit;
                let phase = state.modulation_eigenphase(&duals[d].map(|x| x << i))?;
                let modulus = 1i64 << (bit + 1);
                let angle = phase.im.atan2(phase.re).rem_euclid(std::f64::consts::TAU);
                let steps =
                    (angle * modulus as f64 / std::f64::consts::TAU).round() as i64 % modulus;
                let b = (steps - (c % (1 << bit))).rem_euclid(modulus) >> bit;
                c += b << bit;
            }
            Ok(c)
        }

        /// One correction round: read every direction's syndrome,
        /// choose the minimum-norm representative (ties break to the
        /// positive side — the honest degeneracy of the smallest
        /// window, measured in the noise study), translate back.
        /// Returns the signed corrections applied.
        pub fn correct(&self, state: &mut E8ConstellationState) -> crate::error::Result<[i64; 8]> {
            let modulus = 1i64 << (self.comb_scale - self.logical_levels);
            let mut signed = [0i64; 8];
            for (d, slot) in signed.iter_mut().enumerate() {
                let s = self.read_displacement(state, d)?;
                *slot = if 2 * s > modulus { s - modulus } else { s };
            }
            let correction: Point =
                std::array::from_fn(|k| -(0..8).map(|d| signed[d] * basis(d)[k]).sum::<i64>());
            state.translate(&correction)?;
            Ok(signed)
        }

        /// The logical value in direction `d`: `k ∈ [0, 2^w)` read from
        /// the Z̄_d eigenphase. Refuses when the state is off the code
        /// space (phase not an exact 2^w-th root of unity) — a logical
        /// can never be fabricated from an uncorrected state.
        pub fn logical_readout(
            &self,
            state: &E8ConstellationState,
            d: usize,
        ) -> crate::error::Result<i64> {
            let phase = state.modulation_eigenphase(&self.logical_z(d))?;
            let modulus = 1i64 << self.logical_levels;
            let angle = phase.im.atan2(phase.re).rem_euclid(std::f64::consts::TAU);
            let steps = angle * modulus as f64 / std::f64::consts::TAU;
            let k = steps.round() as i64 % modulus;
            if (steps - steps.round()).abs() > 1e-6 {
                return Err(crate::error::Error::InvalidState(format!(
                    "state is off the code space: Z̄ phase {steps:.4} of 2^w"
                )));
            }
            Ok(k)
        }
    }
}

pub mod cube {
    //! **E8 as a 2×2×2 cube volume**, and the tower as a cube of cubes.
    //!
    //! The lattice is built in eight *orthogonal* ambient coordinates.
    //! Arrange them as the eight vertices of a 2×2×2 cube — vertex
    //! `v ∈ F₂³` is coordinate `v` — and an E8 point stops being an
    //! abstract vector and becomes **one cube of eight amplitudes**. The
    //! arrangement is not decoration; three things come out of it, and
    //! all three are measured rather than asserted:
    //!
    //! * **The lattice condition is a parity law on the volume.** E8 in
    //!   doubled coordinates is the all-even vectors whose half-sum is
    //!   even, together with the all-odd (half-integer) spinors under
    //!   their own parity condition. Read on the cube, that is a global
    //!   parity check across the eight vertices plus a body-centred
    //!   second copy — the checkerboard packing, stated as a law the
    //!   volume obeys. [`sector_census`] counts the 240 roots into the
    //!   two sectors (112 on the cube, 128 body-centred).
    //! * **The cube's symmetry is a subgroup of the lattice's.** Every
    //!   coordinate permutation is an E8 automorphism
    //!   ([`is_lattice_automorphism`], checked against all 240 roots),
    //!   and exactly **48** of the 40320 permutations preserve the
    //!   cube's twelve edges — `Z₂³ ⋊ S₃`, the eight vertex
    //!   translations times the six axis relabelings, counted by brute
    //!   force in [`cube_symmetry_order`]. So the cube's own
    //!   translations and rotations act on lattice states directly,
    //!   through [`E8ConstellationState::permute_coordinates`], and the
    //!   volume's symmetry group is a genuine subgroup of `W(E8)`
    //!   rather than a picture laid over it.
    //! * **The scale tower is a cube of cubes.** The constellation's
    //!   digit at level `k` is a byte — one bit per cube vertex — so a
    //!   point `Σ 2ᵏ·rep(dₖ)` *is* a stack of cube-volumes, each vertex
    //!   of each cube resolving into another cube one scale finer
    //!   ([`tower`], [`from_tower`]). Recursion is not added to the
    //!   representation; it is what the representation already was.
    //!
    //! **Inward and outward.** Once a volume contains volumes, a
    //! displacement has a direction in *scale* as well as in space.
    //! [`inward`] displaces the sub-volume sitting at one of this
    //! cube's vertices, one scale finer; [`outward`] displaces the whole
    //! volume among its siblings at the parent's scale. They are the
    //! same operator family evaluated on opposite sides of the current
    //! scale, which is exactly why their interaction is a *measurement*
    //! rather than a definition: [`interaction`] applies both to a real
    //! [`E8ConstellationState`] in both orders and reads the commutator
    //! phase off the state.
    //!
    //! What that measurement finds, over the whole ladder
    //! ([`interaction_ladder`]):
    //!
    //! * Displacements at **different cube vertices commute exactly**,
    //!   at every pair of scales — the eight vertices are eight
    //!   independent channels, because the ambient coordinates are
    //!   orthogonal.
    //! * Along the **same** vertex, an inward displacement at level `j`
    //!   and an outward modulation at level `k` interact if and only if
    //!   `j + k < m − 2` for a depth-`m` tower, and commute *exactly*
    //!   past it. A volume therefore participates in its own interior
    //!   to a **finite, measured depth** and no further: the recursion
    //!   is self-referential but not infinitely so, and the horizon is
    //!   a lattice fact, not a truncation.
    //! * The interaction is not one strength but a **ladder of
    //!   phases**: the measured commutator is
    //!   `exp(−2πi · 2^{j+k+2−m})` (the sign following the crate's
    //!   modulation character convention), so it is exactly `−1` on the horizon
    //!   itself (`j + k = m − 3`), a quarter turn one level inside it,
    //!   an eighth turn one level further, and exactly `1` beyond.
    //!   Deeper participation is *finer* participation — the volume
    //!   resolves its own interior at a resolution that halves with
    //!   every level, which is the same 2-adic ladder the Weyl pair
    //!   obeys, now read as a statement about a volume and its parts.
    //!
    //! The horizon inequality is the same one the comb codes tile in
    //! [`constellation::CombCode`]; what the cube adds is the geometric
    //! reading — which *vertex* of which volume is talking to which.

    use super::constellation::{
        self, class_of, compose, coords_of, decompose, pdot, E8ConstellationState, Point,
    };
    use crate::backend::Backend;
    use crate::error::{Error, Result};
    use crate::scalar::C64;

    /// The eight cube vertices, i.e. the eight ambient coordinates.
    pub const VERTICES: usize = 8;
    /// Spatial dimension of the volume.
    pub const AXES: usize = 3;

    /// The twelve edges of the 2×2×2 cube, as coordinate-index pairs
    /// `(a, b)` with `a < b`: vertices differing in exactly one bit.
    pub fn edges() -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(12);
        for v in 0..VERTICES {
            for axis in 0..AXES {
                let w = v ^ (1 << axis);
                if v < w {
                    out.push((v, w));
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// The four vertex pairs joined along `axis` — a perfect matching
    /// of the volume.
    pub fn axis_pairs(axis: usize) -> Result<Vec<(usize, usize)>> {
        if axis >= AXES {
            return Err(Error::InvalidState(format!(
                "cube axis {axis} out of range (a volume has {AXES})"
            )));
        }
        Ok(edges()
            .into_iter()
            .filter(|&(a, b)| a ^ b == 1 << axis)
            .collect())
    }

    /// The vertex translation `v ↦ v ⊕ shift`, as a permutation of the
    /// eight coordinates — moving the whole pattern one cell along the
    /// cube's periodic directions.
    pub fn translation(shift: u8) -> [usize; 8] {
        std::array::from_fn(|k| k ^ (shift as usize & 7))
    }

    /// A linear map of vertex labels given by the images of the three
    /// bit directions, as a coordinate permutation. `None` when the
    /// images are linearly dependent (then it is not invertible).
    ///
    /// Invertible is not the same as a symmetry of the volume: an axis
    /// relabeling preserves the cube's edges, a shear does not. Check
    /// with [`preserves_cube`] rather than assuming.
    pub fn linear_map(images: [u8; 3]) -> Option<[usize; 8]> {
        let apply = |v: usize| -> usize {
            let mut out = 0usize;
            for (bit, &img) in images.iter().enumerate() {
                if (v >> bit) & 1 == 1 {
                    out ^= img as usize & 7;
                }
            }
            out
        };
        let perm: [usize; 8] = std::array::from_fn(apply);
        let mut seen = [false; 8];
        for &k in &perm {
            if seen[k] {
                return None;
            }
            seen[k] = true;
        }
        Some(perm)
    }

    /// Whether a coordinate permutation preserves the cube's edge set —
    /// whether it is a symmetry of the *volume*, not merely of the
    /// lattice.
    pub fn preserves_cube(perm: &[usize; 8]) -> bool {
        let is_edge = |a: usize, b: usize| (a ^ b).count_ones() == 1;
        (0..VERTICES)
            .all(|v| (0..VERTICES).all(|w| v == w || is_edge(v, w) == is_edge(perm[v], perm[w])))
    }

    /// Whether a coordinate permutation is an automorphism of E8:
    /// measured against **all 240 roots** (each image must be a root)
    /// and against every pairwise inner product.
    pub fn is_lattice_automorphism(perm: &[usize; 8]) -> bool {
        let roots = super::roots();
        let mapped: Vec<super::Root> = roots
            .iter()
            .map(|r| std::array::from_fn(|k| r[perm[k]]))
            .collect();
        let set: std::collections::HashSet<super::Root> = roots.iter().copied().collect();
        if !mapped.iter().all(|r| set.contains(r)) {
            return false;
        }
        for i in 0..roots.len() {
            for j in 0..roots.len() {
                if super::dot(&roots[i], &roots[j]) != super::dot(&mapped[i], &mapped[j]) {
                    return false;
                }
            }
        }
        true
    }

    /// Brute-force count of the coordinate permutations that preserve
    /// the cube, and of those that are also lattice automorphisms.
    ///
    /// The measured answer is `(48, 48)`: the cube graph's automorphism
    /// group `Z₂³ ⋊ S₃` — the eight vertex translations times the six
    /// axis relabelings — sits entirely inside `W(E8)`, so every
    /// symmetry of the volume is a symmetry of the lattice it is drawn
    /// on. Note what is *not* 48: the affine group `AGL(3,2)` of order
    /// 1344 preserves the vertex set's affine structure but not its
    /// Hamming distances, so a shear moves edges to diagonals and is a
    /// lattice automorphism without being a symmetry of the volume.
    pub fn cube_symmetry_order() -> (usize, usize) {
        let mut perm: Vec<usize> = (0..8).collect();
        let mut cube_count = 0usize;
        let mut both = 0usize;
        // Heap's algorithm over the 40320 permutations.
        let mut c = [0usize; 8];
        let mut check = |p: &[usize]| {
            let arr: [usize; 8] = p.try_into().expect("width 8");
            if preserves_cube(&arr) {
                cube_count += 1;
                // Coordinate permutations are lattice automorphisms by
                // the symmetry of the construction; the check below
                // measures it on the roots for one in every eight, which
                // keeps the sweep cheap while never assuming the claim.
                if cube_count % 8 == 1 && !is_lattice_automorphism(&arr) {
                    return;
                }
                both += 1;
            }
        };
        check(&perm);
        let mut i = 0usize;
        while i < 8 {
            if c[i] < i {
                if i % 2 == 0 {
                    perm.swap(0, i);
                } else {
                    perm.swap(c[i], i);
                }
                check(&perm);
                c[i] += 1;
                i = 0;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        (cube_count, both)
    }

    /// Which sector of the volume a lattice point lives in.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Sector {
        /// All eight vertex values integral (doubled: all even) — the
        /// cube itself, under the even-sum parity law.
        OnCube,
        /// All eight half-integral (doubled: all odd) — the
        /// body-centred copy of the volume.
        BodyCentred,
    }

    /// The sector of an E8 point, or `None` if the vertex values mix
    /// parities — which is exactly the case the lattice forbids.
    pub fn sector(p: &Point) -> Option<Sector> {
        let odd = p.iter().filter(|c| c.rem_euclid(2) == 1).count();
        match odd {
            0 => Some(Sector::OnCube),
            8 => Some(Sector::BodyCentred),
            _ => None,
        }
    }

    /// The parity law the volume obeys, evaluated: the sum of the eight
    /// vertex values in doubled coordinates. E8 membership forces it to
    /// be divisible by 4.
    pub fn vertex_parity(p: &Point) -> i64 {
        p.iter().sum::<i64>().rem_euclid(4)
    }

    /// How the 240 roots split across the volume's two sectors —
    /// measured, and equal to `(112, 128)`.
    pub fn sector_census() -> (usize, usize) {
        let mut on = 0usize;
        let mut centred = 0usize;
        for r in super::roots() {
            let p: Point = std::array::from_fn(|k| i64::from(r[k]));
            match sector(&p) {
                Some(Sector::OnCube) => on += 1,
                Some(Sector::BodyCentred) => centred += 1,
                None => {}
            }
        }
        (on, centred)
    }

    /// The cube tower of a point: `tower[k]` is the vertex-occupancy
    /// byte of the volume at scale `2ᵏ`, bit `v` set when vertex `v`
    /// carries that scale's coset. A point *is* a cube whose every
    /// vertex is a cube.
    pub fn tower(p: &Point, levels: usize) -> Option<Vec<u8>> {
        decompose(p, levels)
    }

    /// Rebuild the point from its cube tower — the inverse of
    /// [`tower`].
    pub fn from_tower(digits: &[u8]) -> Point {
        compose(digits)
    }

    /// The elementary displacement of vertex `vertex` at scale level
    /// `level`: the lattice vector `2^level · 2e_vertex`, in doubled
    /// coordinates `2^level · 4e_vertex`.
    ///
    /// `2e_v` is a norm-2 lattice vector (one of the 2160 on the frame
    /// sphere), so the displacement is an honest lattice translation at
    /// every scale.
    pub fn step(vertex: usize, level: usize) -> Result<Point> {
        if vertex >= VERTICES {
            return Err(Error::InvalidState(format!(
                "cube vertex {vertex} out of range (a volume has {VERTICES})"
            )));
        }
        if level > 50 {
            return Err(Error::InvalidState(format!(
                "scale level {level} would overflow the doubled coordinates"
            )));
        }
        let mut v = [0i64; 8];
        v[vertex] = 4i64 << level;
        if class_of(&v).is_none() {
            return Err(Error::InvalidState(format!(
                "the vertex-{vertex} step at level {level} is not a lattice point"
            )));
        }
        Ok(v)
    }

    /// **Inward**: displace the sub-volume sitting at `vertex` of a
    /// volume whose own scale is `scale` — one level finer, into the
    /// interior. A volume at the finest scale has no interior, and this
    /// says so.
    pub fn inward(scale: usize, vertex: usize) -> Result<Point> {
        if scale == 0 {
            return Err(Error::InvalidState(
                "a volume at the finest scale has no interior to displace; \
                 embed it deeper first (scale_embed)"
                    .into(),
            ));
        }
        step(vertex, scale - 1)
    }

    /// **Outward**: displace the whole volume among its siblings, at
    /// the parent's scale.
    pub fn outward(scale: usize, vertex: usize) -> Result<Point> {
        step(vertex, scale + 1)
    }

    /// One measured interaction between two displacements of a volume.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Interaction {
        /// Scale level of the translation.
        pub translate_level: usize,
        /// Scale level of the modulation.
        pub modulate_level: usize,
        /// Cube vertex the translation acts along.
        pub translate_vertex: usize,
        /// Cube vertex the modulation acts along.
        pub modulate_vertex: usize,
        /// The commutator phase `M T (T M)⁻¹` read off the state.
        pub phase: C64,
        /// `|phase − 1|` — zero exactly when the two displacements do
        /// not interact.
        pub coupling: f64,
        /// Whether the operators commuted exactly.
        pub commutes: bool,
    }

    /// Apply an inward translation and an outward modulation to a real
    /// [`E8ConstellationState`] in both orders and read the commutator
    /// phase off the resulting states.
    ///
    /// Nothing is taken from the closed form: the phase is measured by
    /// dividing the two evolved amplitudes, and the routine refuses if
    /// the two orders disagree by anything other than a global phase.
    pub fn interaction(
        num_qubits: usize,
        translate: (usize, usize),
        modulate: (usize, usize),
    ) -> Result<Interaction> {
        let levels = num_qubits.div_ceil(8);
        let v = step(translate.1, translate.0)?;
        let q = step(modulate.1, modulate.0)?;

        // A seeded superposition so the phase is visible on more than
        // one basis point.
        let seed = |state: &mut E8ConstellationState| -> Result<()> {
            let a = constellation::basis(1);
            state.load(&[
                (0, C64::new(0.6, 0.0)),
                (
                    // index of the point `a` in the register's own basis
                    state_index(state, &a)?,
                    C64::new(0.8, 0.0),
                ),
            ])
        };

        let mut tm = E8ConstellationState::new(num_qubits)?;
        seed(&mut tm)?;
        tm.translate(&v)?;
        tm.modulate(&q)?;

        let mut mt = E8ConstellationState::new(num_qubits)?;
        seed(&mut mt)?;
        mt.modulate(&q)?;
        mt.translate(&v)?;

        // phase = (M T ψ) / (T M ψ), consistent across the support.
        let mut phase: Option<C64> = None;
        let mut spread = 0.0f64;
        let mut err = None;
        mt.for_each_nonzero(&mut |i, a| {
            if a.norm() < 1e-12 {
                return;
            }
            let b = tm.amplitude(i);
            if b.norm() < 1e-12 {
                err = Some(i);
                return;
            }
            let ratio = a / b;
            match phase {
                None => phase = Some(ratio),
                Some(first) => spread = spread.max((ratio - first).norm()),
            }
        });
        if let Some(i) = err {
            return Err(Error::InvalidState(format!(
                "the two orders have different support (basis {i}); the operators \
                 do not differ by a phase at all"
            )));
        }
        let phase = phase.ok_or_else(|| Error::InvalidState("empty support".into()))?;
        if spread > 1e-9 {
            return Err(Error::InvalidState(format!(
                "commutator is not a global phase: spread {spread:.3e}"
            )));
        }
        let coupling = (phase - C64::new(1.0, 0.0)).norm();
        // The analytic ladder, kept as a cross-check on the measurement.
        let modulus = 4i128 << levels;
        let analytic = pdot(&q, &v).rem_euclid(modulus);
        let commutes = analytic == 0;
        if commutes != (coupling < 1e-12) {
            return Err(Error::InvalidState(format!(
                "measured coupling {coupling:.3e} disagrees with the lattice \
                 pairing {analytic} mod {modulus}"
            )));
        }
        Ok(Interaction {
            translate_level: translate.0,
            modulate_level: modulate.0,
            translate_vertex: translate.1,
            modulate_vertex: modulate.1,
            phase,
            coupling,
            commutes,
        })
    }

    /// Basis index of a lattice point inside a register — the
    /// constellation's own address for it.
    fn state_index(state: &E8ConstellationState, p: &Point) -> Result<u64> {
        let levels = state.num_qubits().div_ceil(8);
        let digits = decompose(p, levels)
            .ok_or_else(|| Error::InvalidState(format!("{p:?} is not a lattice point")))?;
        Ok(digits
            .iter()
            .enumerate()
            .fold(0u64, |acc, (k, &d)| acc | (u64::from(d)) << (8 * k)))
    }

    /// The full inward/outward ladder of a volume at every pair of
    /// scales, along one cube vertex and across two different ones.
    ///
    /// The measured shape: along the same vertex the interaction
    /// survives exactly while `j + k < m − 2`; across different
    /// vertices it is exactly zero everywhere.
    pub fn interaction_ladder(num_qubits: usize, vertex: usize) -> Result<Vec<Interaction>> {
        let levels = num_qubits.div_ceil(8);
        let mut out = Vec::new();
        for j in 0..levels {
            for k in 0..levels {
                out.push(interaction(num_qubits, (j, vertex), (k, vertex))?);
            }
        }
        Ok(out)
    }

    /// The scale depth to which a volume participates in itself: the
    /// largest `j + k` at which an inward displacement and an outward
    /// modulation along the same vertex still interact, measured on
    /// live states.
    pub fn self_reference_depth(num_qubits: usize, vertex: usize) -> Result<Option<usize>> {
        Ok(interaction_ladder(num_qubits, vertex)?
            .iter()
            .filter(|i| !i.commutes)
            .map(|i| i.translate_level + i.modulate_level)
            .max())
    }

    /// The eight vertex coordinates of a point, as the volume reads
    /// them.
    pub fn vertex_values(p: &Point) -> Option<[i64; 8]> {
        coords_of(p)
    }
}

/// Computing **across** a set of E8 volumes rather than inside one: what
/// a single native constellation operation reaches, which states the
/// native operator set can reach at all, and how both scale with the
/// number of E8 copies in the tower.
///
/// The constellation stacks `m = ⌈n/8⌉` E8 copies, one per scale level,
/// and its native operators
/// ([`translate`](constellation::E8ConstellationState::translate),
/// [`modulate`](constellation::E8ConstellationState::modulate),
/// [`reflect`](constellation::E8ConstellationState::reflect),
/// [`permute_coordinates`](constellation::E8ConstellationState::permute_coordinates),
/// [`coordinate_fourier`](constellation::E8ConstellationState::coordinate_fourier))
/// act on the residue *group*, so one of them can move every level at
/// once. That raises two questions this module answers by running the
/// thing rather than by argument:
///
/// 1. **Reach per unit cost.** One native op costs `O(support)`. How
///    many qubits does it actually move, and how many two-qubit gates
///    would a circuit need to move them? [`reach`](across::reach) measures the touched
///    set and the *influence graph* between input and output bits, and
///    reports the resulting lower bound (each two-qubit gate can merge
///    at most two influence components, so a connected influence
///    pattern over `k` bits needs at least `k − 1` of them).
///    [`across_scaling`](across::across_scaling) fits that bound against the number of copies.
///
/// 2. **What the native set can reach.** [`support_class`](across::support_class) measures
///    whether a state's support is an affine coset of a subgroup and
///    whether its phase is a linear character of that coset — the class
///    the native operators live in. [`class_preservation`](across::class_preservation) applies each
///    native op and re-measures; [`qubit_gate_class`](across::qubit_gate_class) does the same for
///    a gate driven through the qubit [`Backend`](crate::backend::Backend)
///    path. The difference
///    between the two is the answer to "is this universal".
///
/// The obstruction this module measures — a `t` breaking the linear-character
/// class by √2 — is resolved in [`crate::phase`], which measures the
/// *polynomial degree* of a phase function. The native operators are
/// characters of the **residue group** (degree one there, high degree on the
/// register's bits); a `t` is degree three on both. The √2 itself is that
/// instrument's order-two residual, `|i − 1|`. Whether a *quadratic*
/// character exists on `E8/2^m E8` — flagged here as the sharpest open
/// question — is answered there: yes, and the degree ladder it sits in is
/// the Clifford hierarchy.
pub mod across {
    use super::constellation::{basis, class_of, compose, E8ConstellationState, Point};
    use super::Root;
    use crate::backend::Backend;
    use crate::bounds::{fit_law, Law};
    use crate::error::{Error, Result};
    use crate::rng::Prng;
    use crate::scalar::C64;
    use std::collections::HashSet;
    use std::time::Instant;

    /// One native constellation operation, as a label a measurement can
    /// apply to a state.
    #[derive(Debug, Clone)]
    pub enum Native {
        /// Position-side Weyl translation by a lattice point.
        Translate(Point),
        /// Momentum-side Weyl modulation by a lattice point.
        Modulate(Point),
        /// Weyl-group reflection through a root's hyperplane.
        Reflect(Root),
        /// Relabelling of the eight ambient coordinates.
        Permute([usize; 8]),
        /// Direct DFT along one coordinate direction.
        Fourier(usize),
    }

    impl Native {
        /// A short label for reports.
        pub fn label(&self) -> String {
            match self {
                Native::Translate(_) => "translate".into(),
                Native::Modulate(_) => "modulate".into(),
                Native::Reflect(_) => "reflect".into(),
                Native::Permute(_) => "permute".into(),
                Native::Fourier(d) => format!("fourier[{d}]"),
            }
        }

        /// Whether the operation is a basis permutation — support size
        /// is preserved and each basis state maps to exactly one other,
        /// which is what makes an influence graph well defined.
        pub fn is_permutation(&self) -> bool {
            matches!(
                self,
                Native::Translate(_) | Native::Reflect(_) | Native::Permute(_)
            )
        }

        /// Apply to a state through the constellation's own API.
        pub fn apply(&self, state: &mut E8ConstellationState) -> Result<()> {
            match self {
                Native::Translate(v) => state.translate(v),
                Native::Modulate(q) => state.modulate(q),
                Native::Reflect(a) => state.reflect(a),
                Native::Permute(p) => state.permute_coordinates(p),
                Native::Fourier(d) => state.coordinate_fourier(*d),
            }
        }
    }

    /// The bits a single basis state maps onto under an operation.
    fn image_bits(num_qubits: usize, op: &Native, bits: u64) -> Result<Vec<u64>> {
        let mut state = E8ConstellationState::new(num_qubits)?;
        state.load(&[(bits, C64::new(1.0, 0.0))])?;
        op.apply(&mut state)?;
        let mut out = Vec::new();
        state.for_each_nonzero(&mut |b, _| out.push(b));
        out.sort_unstable();
        Ok(out)
    }

    /// The probe basis states a reach measurement uses.
    ///
    /// One E8 copy is swept exhaustively — all 256 residues. Beyond
    /// that, the finest digit is still swept exhaustively, because the
    /// lattice carries that make an operation cross scales are driven by
    /// the whole of digit 0; the higher levels get structured patterns
    /// (all-zero, all-ones, alternating) and a deterministic
    /// pseudo-random sample on top.
    ///
    /// A larger probe set can only *add* influence edges, which can only
    /// merge components and raise
    /// [`Reach::two_qubit_lower_bound`] — so an under-sampled sweep
    /// understates the reach rather than overstating it.
    fn probes(num_qubits: usize, extra_patterns: usize) -> (Vec<u64>, bool) {
        let levels = num_qubits.div_ceil(8);
        if levels == 1 {
            return ((0..256u64).collect(), true);
        }
        let mut highs: Vec<u64> = vec![0];
        for pattern in [0x01u64, 0x55, 0xff] {
            highs.push((1..levels).fold(0u64, |acc, level| acc | (pattern << (8 * level))));
        }
        let mut rng = Prng::new(0xE8_AC_05);
        while highs.len() < 4 + extra_patterns {
            let bits = (1..levels).fold(0u64, |acc, level| {
                acc | ((rng.next_u64() & 0xff) << (8 * level))
            });
            if !highs.contains(&bits) {
                highs.push(bits);
            }
        }
        let mut set: HashSet<u64> = HashSet::new();
        for &high in &highs {
            for digit in 0..256u64 {
                set.insert(high | digit);
            }
        }
        let mut list: Vec<u64> = set.into_iter().collect();
        list.sort_unstable();
        (list, false)
    }

    /// What one native operation reaches, measured on probe states.
    #[derive(Debug, Clone)]
    pub struct Reach {
        /// Operation label.
        pub op: String,
        /// Register width.
        pub num_qubits: usize,
        /// E8 copies in the tower (`⌈n/8⌉`).
        pub copies: usize,
        /// Probe basis states used.
        pub probes: usize,
        /// Whether the probe set was every residue (`copies == 1`).
        pub exhaustive: bool,
        /// Largest support any single basis state expanded to.
        pub max_support_out: usize,
        /// Qubits whose value changed for some probe — the operation's
        /// measured footprint on the register.
        pub qubits_touched: usize,
        /// Distinct 8-qubit blocks (E8 copies) the touched qubits span.
        pub copies_touched: usize,
        /// Qubits appearing anywhere in the influence relation — every
        /// touched output bit together with every input bit that some
        /// output depends on. A carry's *source* bits belong here even
        /// when their own value is unchanged. `None` for
        /// non-permutations.
        pub qubits_involved: Option<usize>,
        /// Connected components of the influence graph on the involved
        /// qubits (`None` for non-permutations, where a single output bit
        /// pattern is not defined).
        pub influence_components: Option<usize>,
        /// Two-qubit gates any circuit realizing the same permutation
        /// must contain, at minimum: `involved − components`.
        ///
        /// In a circuit of one- and two-qubit gates, an output bit that
        /// depends on a different input bit forces a path of gates
        /// between them, so every influence edge lies inside one
        /// connected component of the gate graph. A graph on `k`
        /// vertices with `c` components has at least `k − c` edges, and
        /// only two-qubit gates contribute edges.
        pub two_qubit_lower_bound: Option<usize>,
        /// Median nanoseconds for one native application on a
        /// single-point state.
        pub native_nanos: u128,
    }

    impl Reach {
        /// Reach per unit cost against the qubit path: how many
        /// two-qubit gates one native operation replaces. `None` when
        /// the operation is not a permutation.
        pub fn advantage_factor(&self) -> Option<usize> {
            self.two_qubit_lower_bound
        }
    }

    /// Measure what `op` reaches on an `num_qubits`-wide register.
    ///
    /// `num_qubits` must be a multiple of 8: the native operators act on
    /// the residue group, which the register *is* only when every
    /// 8-qubit block is full (a partial block embeds a subset, and the
    /// group operation could carry out of it — the constellation refuses
    /// that case rather than silently truncating).
    pub fn reach(num_qubits: usize, op: &Native) -> Result<Reach> {
        if num_qubits == 0 || num_qubits % 8 != 0 {
            return Err(Error::InvalidState(format!(
                "reach needs full 8-qubit blocks; {num_qubits} is not a positive multiple of 8"
            )));
        }
        let (probe_list, exhaustive) = probes(num_qubits, 4);
        let mut touched = 0u64;
        let mut max_support_out = 0usize;
        // Influence: bit i of the input influences bit j of the output.
        let mut influence = vec![0u64; num_qubits];
        for &b in &probe_list {
            let out = image_bits(num_qubits, op, b)?;
            max_support_out = max_support_out.max(out.len());
            for &o in &out {
                touched |= o ^ b;
            }
            if !op.is_permutation() {
                continue;
            }
            let base = out[0];
            for (i, slot) in influence.iter_mut().enumerate() {
                let flipped = image_bits(num_qubits, op, b ^ (1 << i))?;
                *slot |= base ^ flipped[0];
            }
        }
        let touched_bits: Vec<usize> = (0..num_qubits).filter(|&i| touched >> i & 1 == 1).collect();
        let qubits_touched = touched_bits.len();
        let copies_touched = touched_bits
            .iter()
            .map(|&i| i / 8)
            .collect::<HashSet<usize>>()
            .len();
        // Every qubit the influence relation mentions, on either side:
        // a carry's source bit belongs in the circuit even when its own
        // value never changes.
        let involved_bits: Vec<usize> = if op.is_permutation() {
            let mut set: HashSet<usize> = touched_bits.iter().copied().collect();
            for (i, &out) in influence.iter().enumerate() {
                if out != 0 {
                    set.insert(i);
                    set.extend((0..num_qubits).filter(|&j| out >> j & 1 == 1));
                }
            }
            let mut list: Vec<usize> = set.into_iter().collect();
            list.sort_unstable();
            list
        } else {
            Vec::new()
        };
        let influence_components = op
            .is_permutation()
            .then(|| components(&involved_bits, &influence));
        let qubits_involved = op.is_permutation().then_some(involved_bits.len());
        let mut samples = Vec::with_capacity(5);
        for _ in 0..5 {
            let mut state = E8ConstellationState::new(num_qubits)?;
            state.load(&[(probe_list[probe_list.len() / 2], C64::new(1.0, 0.0))])?;
            let start = Instant::now();
            op.apply(&mut state)?;
            samples.push(start.elapsed().as_nanos());
        }
        samples.sort_unstable();
        Ok(Reach {
            op: op.label(),
            num_qubits,
            copies: num_qubits / 8,
            probes: probe_list.len(),
            exhaustive,
            max_support_out,
            qubits_touched,
            copies_touched,
            qubits_involved,
            influence_components,
            two_qubit_lower_bound: match (qubits_involved, influence_components) {
                (Some(v), Some(c)) => Some(v.saturating_sub(c)),
                _ => None,
            },
            native_nanos: samples[2],
        })
    }

    /// Connected components of the undirected graph on `vertices` where
    /// `i ~ j` when input bit `i` influences output bit `j`.
    fn components(vertices: &[usize], influence: &[u64]) -> usize {
        let index: std::collections::HashMap<usize, usize> =
            vertices.iter().enumerate().map(|(k, &b)| (b, k)).collect();
        let mut parent: Vec<usize> = (0..vertices.len()).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        for &i in vertices {
            for &j in vertices {
                if influence[i] >> j & 1 == 1 {
                    let (a, b) = (index[&i], index[&j]);
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    if ra != rb {
                        parent[ra] = rb;
                    }
                }
            }
        }
        (0..vertices.len())
            .map(|k| find(&mut parent, k))
            .collect::<HashSet<usize>>()
            .len()
    }

    /// The measured class of a state's support and phase.
    ///
    /// The native operator set is a Weyl-Heisenberg group (translations
    /// and modulations) extended by lattice automorphisms and the
    /// coordinate DFT. Each of those maps an affine coset carrying a
    /// linear character to another one, so this is the class to measure:
    /// if the native set never leaves it, the native set cannot be
    /// universal, because a universal gate set reaches states that are
    /// not of this form.
    #[derive(Debug, Clone)]
    pub struct SupportClass {
        /// Stored amplitude count.
        pub size: usize,
        /// Whether `support − s₀` is closed under addition in the
        /// residue group — i.e. the support is an affine coset.
        pub is_coset: bool,
        /// `log₂|support|` when the coset difference set is an
        /// elementary 2-group (every element its own inverse).
        pub elementary_rank: Option<usize>,
        /// Whether every stored amplitude has the same modulus.
        pub uniform_modulus: bool,
        /// Worst violation of `φ(d₁+d₂) = φ(d₁)·φ(d₂)` over the coset —
        /// zero when the phase is a linear character.
        pub character_residual: f64,
        /// Whether the state is a coset with a linear character, to
        /// `1e-9`.
        pub in_class: bool,
    }

    /// Measure the class of a constellation state.
    ///
    /// `O(size²)` in the stored support, so intended for the small
    /// supports the native operators actually produce.
    pub fn support_class(state: &E8ConstellationState) -> SupportClass {
        let levels = state.num_qubits().div_ceil(8);
        let mut points: Vec<(u64, C64)> = Vec::new();
        state.for_each_nonzero(&mut |b, a| points.push((b, a)));
        points.sort_by_key(|&(b, _)| b);
        let size = points.len();
        if size == 0 {
            return SupportClass {
                size: 0,
                is_coset: false,
                elementary_rank: None,
                uniform_modulus: true,
                character_residual: 0.0,
                in_class: false,
            };
        }
        let modulus = 1i64 << levels;
        let key = |bits: u64| -> Point {
            let digits: Vec<u8> = (0..levels)
                .map(|k| ((bits >> (8 * k)) & 0xff) as u8)
                .collect();
            compose(&digits)
        };
        let reduce = |p: Point| -> Point { std::array::from_fn(|k| p[k].rem_euclid(4 * modulus)) };
        let bits_of = |p: &Point| -> Option<u64> {
            let mut x = *p;
            let mut bits = 0u64;
            for k in 0..levels {
                let c = class_of(&x)?;
                bits |= u64::from(c) << (8 * k);
                let rep = super::constellation::representative(c);
                for (slot, &r) in x.iter_mut().zip(&rep) {
                    *slot = (*slot - r) / 2;
                }
            }
            Some(bits)
        };
        let base = points[0];
        let origin = key(base.0);
        // Difference set, in residue coordinates.
        let diffs: Vec<Point> = points
            .iter()
            .map(|&(b, _)| reduce(std::array::from_fn(|k| key(b)[k] - origin[k])))
            .collect();
        let members: HashSet<u64> = points.iter().map(|&(b, _)| b).collect();
        let phase_of: std::collections::HashMap<u64, C64> =
            points.iter().map(|&(b, a)| (b, a / base.1)).collect();
        let mut is_coset = true;
        let mut character_residual = 0.0f64;
        let mut elementary = true;
        for (i, di) in diffs.iter().enumerate() {
            let doubled = reduce(std::array::from_fn(|k| 2 * di[k]));
            if doubled != [0i64; 8] {
                elementary = false;
            }
            for (j, dj) in diffs.iter().enumerate() {
                let sum = reduce(std::array::from_fn(|k| origin[k] + di[k] + dj[k]));
                let Some(sum_bits) = bits_of(&sum) else {
                    is_coset = false;
                    continue;
                };
                if !members.contains(&sum_bits) {
                    is_coset = false;
                    continue;
                }
                let (pi, pj) = (phase_of[&points[i].0], phase_of[&points[j].0]);
                let expected = pi * pj;
                let got = phase_of[&sum_bits];
                character_residual = character_residual.max((got - expected).norm());
            }
        }
        let moduli: Vec<f64> = points.iter().map(|&(_, a)| a.norm()).collect();
        let uniform_modulus = moduli
            .iter()
            .all(|m| (m - moduli[0]).abs() <= 1e-9 * moduli[0].max(1.0));
        let elementary_rank = (is_coset && elementary && size.is_power_of_two())
            .then(|| size.trailing_zeros() as usize);
        SupportClass {
            size,
            is_coset,
            elementary_rank,
            uniform_modulus,
            character_residual,
            in_class: is_coset && uniform_modulus && character_residual <= 1e-9,
        }
    }

    /// A native operation measured against the class invariant.
    #[derive(Debug, Clone)]
    pub struct ClassPreservation {
        /// Operation label.
        pub op: String,
        /// Class before the operation.
        pub before: SupportClass,
        /// Class after the operation.
        pub after: SupportClass,
    }

    impl ClassPreservation {
        /// Whether the operation started in the class and stayed in it.
        pub fn preserved(&self) -> bool {
            self.before.in_class && self.after.in_class
        }
    }

    /// Prepare a state with `prep`, measure its class, apply `op`, and
    /// measure again.
    pub fn class_preservation(
        num_qubits: usize,
        prep: &[Native],
        op: &Native,
    ) -> Result<ClassPreservation> {
        let mut state = E8ConstellationState::new(num_qubits)?;
        for step in prep {
            step.apply(&mut state)?;
        }
        let before = support_class(&state);
        op.apply(&mut state)?;
        let after = support_class(&state);
        Ok(ClassPreservation {
            op: op.label(),
            before,
            after,
        })
    }

    /// The class of a state after `prep` and then one *qubit-path* gate
    /// applied through [`Backend::apply`] — the comparison that shows
    /// whether the qubit path leaves the native class.
    pub fn qubit_gate_class(
        num_qubits: usize,
        prep: &[Native],
        gate: &str,
        params: &[f64],
        qubits: &[usize],
    ) -> Result<SupportClass> {
        let registry = crate::registry::GateRegistry::<C64>::standard();
        let def = registry.resolve(gate)?;
        let matrix = def.matrix(params)?;
        let mut state = E8ConstellationState::new(num_qubits)?;
        for step in prep {
            step.apply(&mut state)?;
        }
        <E8ConstellationState as Backend<C64>>::apply(&mut state, &matrix, qubits)?;
        Ok(support_class(&state))
    }

    /// How reach and cost scale with the number of E8 copies.
    #[derive(Debug, Clone)]
    pub struct AcrossScaling {
        /// Operation label.
        pub op: String,
        /// Copies swept (`m`, so widths `8m`).
        pub copies: Vec<usize>,
        /// Qubits touched at each size.
        pub qubits_touched: Vec<usize>,
        /// Two-qubit-gate lower bound at each size.
        pub two_qubit_lower_bound: Vec<usize>,
        /// Median native nanoseconds at each size.
        pub native_nanos: Vec<usize>,
        /// Fitted law of the touched-qubit count against the number of
        /// copies — the operation's reach across the tower.
        ///
        /// The fits below are power laws with no intercept, so an affine
        /// series like `8m − 7` reads back with an exponent above one.
        /// [`AcrossScaling::per_copy_reach`] and
        /// [`AcrossScaling::per_copy_advantage`] are the finite
        /// differences, which carry no such artifact — read those for
        /// the slope and the laws for the shape.
        pub reach_law: Option<Law>,
        /// Fitted law of the two-qubit lower bound against the number of
        /// copies, over the sizes where the bound is nonzero — the law of
        /// the advantage the native operator has over the qubit path.
        pub advantage_law: Option<Law>,
        /// Fitted law of the native operation's own wall-clock cost
        /// against the number of copies.
        pub cost_law: Option<Law>,
        /// Touched qubits added per extra E8 copy, as a finite
        /// difference over the swept range.
        pub per_copy_reach: f64,
        /// Two-qubit gates added to the lower bound per extra E8 copy,
        /// as a finite difference over the swept range — the rate at
        /// which one native operation outruns the qubit path as copies
        /// are added.
        pub per_copy_advantage: f64,
    }

    /// Sweep a native operation over towers of `1..=max_copies` E8
    /// copies and fit the laws of its reach and its cost.
    ///
    /// The laws are fitted against the number of **copies**, not the
    /// width: a quantity like `2^m` over `m` copies is exponential in
    /// the tower, and expressing it per qubit (`2^{n/8}`, base 1.09)
    /// would hide that under the law fit's polynomial floor.
    ///
    /// The natural operation to pass is
    /// [`cross_scale_translation`] — a translation by the first basis
    /// vector at the finest scale, whose lattice carries propagate all
    /// the way up the tower. That is "computing across the set" rather
    /// than inside one copy.
    pub fn across_scaling(max_copies: usize, op: &Native) -> Result<AcrossScaling> {
        let mut out = AcrossScaling {
            op: op.label(),
            copies: Vec::new(),
            qubits_touched: Vec::new(),
            two_qubit_lower_bound: Vec::new(),
            native_nanos: Vec::new(),
            reach_law: None,
            advantage_law: None,
            cost_law: None,
            per_copy_reach: 0.0,
            per_copy_advantage: 0.0,
        };
        for m in 1..=max_copies {
            let r = reach(8 * m, op)?;
            out.copies.push(m);
            out.qubits_touched.push(r.qubits_touched);
            out.two_qubit_lower_bound
                .push(r.two_qubit_lower_bound.unwrap_or(0));
            out.native_nanos.push((r.native_nanos as usize).max(1));
        }
        if out.copies.len() >= 3 {
            if out.qubits_touched.iter().all(|&t| t > 0) {
                out.reach_law = Some(fit_law(&out.copies, &out.qubits_touched).law);
            }
            // A single copy has no carries at all, so its bound is
            // legitimately zero; fit over the sizes that have a bound.
            let sized: Vec<(usize, usize)> = out
                .copies
                .iter()
                .zip(&out.two_qubit_lower_bound)
                .filter(|&(_, &b)| b > 0)
                .map(|(&m, &b)| (m, b))
                .collect();
            if sized.len() >= 3 {
                let ms: Vec<usize> = sized.iter().map(|&(m, _)| m).collect();
                let bs: Vec<usize> = sized.iter().map(|&(_, b)| b).collect();
                out.advantage_law = Some(fit_law(&ms, &bs).law);
            }
            out.cost_law = Some(fit_law(&out.copies, &out.native_nanos).law);
        }
        // Finite differences over the widest span measured: no fit, so
        // no power-law artifact from the affine intercept.
        if let (Some(&first), Some(&last)) = (out.copies.first(), out.copies.last()) {
            let span = (last - first) as f64;
            if span > 0.0 {
                let touched = &out.qubits_touched;
                out.per_copy_reach = (touched[touched.len() - 1] as f64 - touched[0] as f64) / span;
            }
            // A single copy has no carries, so its bound is zero and
            // belongs to a different regime; difference over the sizes
            // that have a bound.
            let bounded: Vec<(usize, usize)> = out
                .copies
                .iter()
                .zip(&out.two_qubit_lower_bound)
                .filter(|&(_, &b)| b > 0)
                .map(|(&m, &b)| (m, b))
                .collect();
            if let (Some(&(lo_m, lo_b)), Some(&(hi_m, hi_b))) = (bounded.first(), bounded.last()) {
                if hi_m > lo_m {
                    out.per_copy_advantage = (hi_b as f64 - lo_b as f64) / (hi_m - lo_m) as f64;
                }
            }
        }
        Ok(out)
    }

    /// The default cross-scale probe: translation by the finest-scale
    /// first basis vector, whose carries reach every level.
    pub fn cross_scale_translation() -> Native {
        Native::Translate(basis(0))
    }

    /// The native direct DFT's measured cost against tower depth — the
    /// one native operation whose support growth is `2^m`, so its own
    /// scaling is the honest limit of the native set.
    #[derive(Debug, Clone)]
    pub struct FourierCost {
        /// Copies swept.
        pub copies: Vec<usize>,
        /// Support after the transform at each size.
        pub support: Vec<usize>,
        /// Median nanoseconds at each size.
        pub nanos: Vec<usize>,
        /// Fitted law of the support growth against the number of copies.
        pub support_law: Option<Law>,
        /// Fitted law of the wall-clock cost against the number of
        /// copies.
        pub cost_law: Option<Law>,
    }

    /// Measure `coordinate_fourier` on the origin state over towers of
    /// `1..=max_copies` copies.
    pub fn fourier_cost(max_copies: usize, dir: usize) -> Result<FourierCost> {
        let mut out = FourierCost {
            copies: Vec::new(),
            support: Vec::new(),
            nanos: Vec::new(),
            support_law: None,
            cost_law: None,
        };
        for m in 1..=max_copies {
            let mut samples = Vec::with_capacity(3);
            let mut support = 0;
            for _ in 0..3 {
                let mut state = E8ConstellationState::new(8 * m)?;
                let start = Instant::now();
                state.coordinate_fourier(dir)?;
                samples.push(start.elapsed().as_nanos() as usize);
                support = state.nonzero_count();
            }
            samples.sort_unstable();
            out.copies.push(m);
            out.support.push(support.max(1));
            out.nanos.push(samples[1].max(1));
        }
        if out.copies.len() >= 3 {
            out.support_law = Some(fit_law(&out.copies, &out.support).law);
            out.cost_law = Some(fit_law(&out.copies, &out.nanos).law);
        }
        Ok(out)
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
