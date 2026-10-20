//! The register's state space as a first-class object: one radix per
//! site, mixed freely, and no geometry.
//!
//! Every backend in this crate was written against one shape — `n`
//! two-level sites, basis indices read as bit strings. A [`Space`] names
//! the shape instead of assuming it: site `k` has `radix(k) ≥ 2` levels,
//! and the register's Hilbert space is the tensor product
//!
//! ```text
//! ℋ = ℂ^{r_0} ⊗ ℂ^{r_1} ⊗ … ⊗ ℂ^{r_{n−1}},     dim ℋ = Π r_k
//! ```
//!
//! (over whatever amplitude algebra the backend uses). A qubit register
//! is the special case `r_k = 2` for every `k`, and every backend reports
//! its space through [`Backend::space`](crate::backend::Backend::space) —
//! the default is exactly that qubit space, so nothing that predates this
//! module changes meaning.
//!
//! ## Indexing
//!
//! Basis indices are **little-endian mixed radix**, the crate's qubit
//! convention generalized: site 0 is the least significant digit, and
//!
//! ```text
//! index = Σ_k digit_k · stride_k,      stride_k = Π_{j<k} r_j
//! ```
//!
//! so on a qubit space a digit is a bit and `stride_k = 2^k`. A `k`-site
//! [`SiteMatrix`] is row-major over the same convention restricted to its
//! targets: `sub = Σ_b digit(sites[b]) · Π_{c<b} radix(sites[c])`, which
//! is `qubits[b] ↔ bit b` when every radix is 2, and is the convention
//! [`crate::mixed::CompoundRegister`] already uses.
//!
//! ## What a space is not
//!
//! It carries no positions, no couplings and no ordering beyond the index
//! convention. Where sites sit, and which pairs interact, is geometry and
//! stays in the layers that already own it ([`crate::lattice`],
//! [`crate::curve`], the device [`Topology`](crate::backend::Topology)).
//! The space says *what* the register is; the geometry says *where* its
//! sites are; the backend says *how* it is stored. They change
//! independently.

use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::scalar::{Scalar, C64};
use std::fmt;

/// The largest register dimension a backend can index: basis indices are
/// `u64`.
pub const SPACE_MAX_DIM: u128 = u64::MAX as u128;

/// A register's state space: one radix per site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Space {
    radices: Vec<usize>,
    strides: Vec<u64>,
    dim: u128,
    qubits: bool,
}

impl Space {
    /// A mixed-radix space. Every radix must be at least 2, and the total
    /// dimension must fit a `u64` basis index.
    pub fn new(radices: &[usize]) -> Result<Space> {
        let mut strides = Vec::with_capacity(radices.len());
        let mut dim: u128 = 1;
        for (site, &r) in radices.iter().enumerate() {
            if r < 2 {
                return Err(Error::InvalidState(format!(
                    "space: site {site} has radix {r}; every site needs at least 2 levels"
                )));
            }
            strides.push(dim as u64);
            dim = dim.saturating_mul(r as u128);
            if dim > SPACE_MAX_DIM {
                return Err(Error::InvalidState(format!(
                    "space: dimension of {radices:?} exceeds the u64 index range"
                )));
            }
        }
        Ok(Space {
            radices: radices.to_vec(),
            strides,
            dim,
            qubits: radices.iter().all(|&r| r == 2),
        })
    }

    /// `n` qubits. Never fails, so every backend can report its width:
    /// the dimension is stored as `u128`, and on a register wider than 64
    /// qubits the sites past bit 63 have no digit in a `u64` index
    /// ([`Space::digit`] reads them as 0).
    pub fn qubits(n: usize) -> Space {
        Space {
            radices: vec![2; n],
            strides: (0..n)
                .map(|k| 1u64.checked_shl(k as u32).unwrap_or(0))
                .collect(),
            dim: 1u128 << n.min(127),
            qubits: true,
        }
    }

    /// Number of sites.
    pub fn sites(&self) -> usize {
        self.radices.len()
    }

    /// All radices, site 0 first.
    pub fn radices(&self) -> &[usize] {
        &self.radices
    }

    /// Radix of one site.
    ///
    /// # Panics
    /// Panics if `site >= self.sites()`.
    pub fn radix(&self, site: usize) -> usize {
        self.radices[site]
    }

    /// Index stride of one site, `Π_{j<site} r_j`.
    ///
    /// # Panics
    /// Panics if `site >= self.sites()`.
    pub fn stride(&self, site: usize) -> u64 {
        self.strides[site]
    }

    /// Total dimension `Π r_k`.
    pub fn dim(&self) -> u128 {
        self.dim
    }

    /// Whether every site is a qubit.
    pub fn is_qubits(&self) -> bool {
        self.qubits
    }

    /// Digit of `site` in basis index `index`.
    #[inline]
    pub fn digit(&self, index: u64, site: usize) -> usize {
        match self.strides[site] {
            0 => 0,
            stride => ((index / stride) % self.radices[site] as u64) as usize,
        }
    }

    /// `index` with the digit of `site` replaced by `digit`.
    #[inline]
    pub fn with_digit(&self, index: u64, site: usize, digit: usize) -> u64 {
        let old = self.digit(index, site) as u64;
        index - old * self.strides[site] + digit as u64 * self.strides[site]
    }

    /// Basis index of a full digit string (one digit per site).
    pub fn encode(&self, digits: &[usize]) -> Result<u64> {
        if digits.len() != self.sites() {
            return Err(Error::BadDimension {
                expected: self.sites(),
                got: digits.len(),
            });
        }
        let mut index = 0u64;
        for (site, &d) in digits.iter().enumerate() {
            if d >= self.radices[site] {
                return Err(Error::InvalidState(format!(
                    "space: digit {d} out of range for site {site} of radix {}",
                    self.radices[site]
                )));
            }
            index += d as u64 * self.strides[site];
        }
        Ok(index)
    }

    /// Digit string of a basis index.
    pub fn decode(&self, index: u64) -> Vec<usize> {
        (0..self.sites()).map(|s| self.digit(index, s)).collect()
    }

    /// Check a target list: in range and pairwise distinct.
    pub fn validate_sites(&self, sites: &[usize]) -> Result<()> {
        crate::circuit::validate_targets(self.sites(), sites)
    }

    /// Dimension of the sub-space a gate on `sites` acts on.
    pub fn sub_dim(&self, sites: &[usize]) -> usize {
        sites.iter().map(|&s| self.radices[s]).product()
    }

    /// Little-endian sub-index of `index` over `sites`.
    #[inline]
    pub fn sub_index(&self, index: u64, sites: &[usize]) -> usize {
        let mut sub = 0usize;
        let mut weight = 1usize;
        for &s in sites {
            sub += self.digit(index, s) * weight;
            weight *= self.radices[s];
        }
        sub
    }

    /// For each sub-index of a gate on `sites`, the basis-index offset its
    /// digits occupy — the mixed-radix scatter table.
    pub fn offsets(&self, sites: &[usize]) -> Vec<u64> {
        let d = self.sub_dim(sites);
        (0..d)
            .map(|mut sub| {
                let mut off = 0u64;
                for &s in sites {
                    let r = self.radices[s];
                    off += (sub % r) as u64 * self.strides[s];
                    sub /= r;
                }
                off
            })
            .collect()
    }
}

impl fmt::Display for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.qubits {
            write!(f, "{} qubits", self.sites())
        } else {
            let r: Vec<String> = self.radices.iter().map(|r| r.to_string()).collect();
            write!(f, "radices [{}]", r.join(", "))
        }
    }
}

/// A square matrix of any dimension over `S`, row-major: the site-level
/// counterpart of [`GateMatrix`], which is restricted to powers of two.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteMatrix<S: Scalar> {
    dim: usize,
    data: Vec<S>,
}

impl<S: Scalar> SiteMatrix<S> {
    /// From a row-major entry vector of length `dim²`.
    pub fn from_vec(dim: usize, data: Vec<S>) -> Result<Self> {
        if dim == 0 || data.len() != dim * dim {
            return Err(Error::BadDimension {
                expected: dim * dim,
                got: data.len(),
            });
        }
        Ok(SiteMatrix { dim, data })
    }

    /// From row-major complex entries, if every entry embeds in `S`;
    /// otherwise [`Error::UnsupportedForAlgebra`] naming `label`.
    pub fn try_from_c64s(label: &str, dim: usize, entries: &[C64]) -> Result<Self> {
        let unsupported = || Error::UnsupportedForAlgebra {
            gate: label.to_string(),
            algebra: S::algebra_name(),
        };
        let data = entries
            .iter()
            .map(|&z| S::try_from_c64(z).ok_or_else(unsupported))
            .collect::<Result<Vec<S>>>()?;
        SiteMatrix::from_vec(dim, data)
    }

    /// The same matrix, viewed at site level.
    pub fn from_gate(m: &GateMatrix<S>) -> Self {
        SiteMatrix {
            dim: m.dim(),
            data: m.data().to_vec(),
        }
    }

    /// Back to a [`GateMatrix`], when the dimension is a power of two.
    pub fn to_gate(&self) -> Result<GateMatrix<S>> {
        GateMatrix::from_vec(self.dim, self.data.clone())
    }

    /// Identity of dimension `dim`.
    pub fn identity(dim: usize) -> Self {
        let mut data = vec![S::zero(); dim * dim];
        for i in 0..dim {
            data[i * dim + i] = S::one();
        }
        SiteMatrix { dim, data }
    }

    /// Dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Row-major entries.
    pub fn data(&self) -> &[S] {
        &self.data
    }

    /// Entry at `(row, col)`.
    pub fn get(&self, row: usize, col: usize) -> S {
        self.data[row * self.dim + col]
    }

    /// Product `self · rhs` (rows of `self` against columns of `rhs`,
    /// entries multiplied left to right).
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.dim != rhs.dim {
            return Err(Error::BadDimension {
                expected: self.dim,
                got: rhs.dim,
            });
        }
        let d = self.dim;
        let mut out = vec![S::zero(); d * d];
        for r in 0..d {
            for c in 0..d {
                let mut acc = S::zero();
                for k in 0..d {
                    acc = acc + self.data[r * d + k] * rhs.data[k * d + c];
                }
                out[r * d + c] = acc;
            }
        }
        Ok(SiteMatrix { dim: d, data: out })
    }

    /// Max-entry deviation of `M† M` from the identity, in the Euclidean
    /// coordinate metric.
    pub fn unitarity_deviation(&self) -> f64 {
        let d = self.dim;
        let mut worst = 0.0f64;
        for c1 in 0..d {
            for c2 in 0..d {
                let mut acc = S::zero();
                for r in 0..d {
                    acc = acc + self.data[r * d + c1].conj() * self.data[r * d + c2];
                }
                let target = if c1 == c2 { S::one() } else { S::zero() };
                worst = worst.max((acc - target).abs_sqr().sqrt());
            }
        }
        worst
    }

    /// Fail with [`Error::NotUnitary`] unless `M† M = 1` to within `tol`.
    pub fn check_unitary(&self, label: &str, tol: f64) -> Result<()> {
        let deviation = self.unitarity_deviation();
        if deviation > tol {
            return Err(Error::NotUnitary {
                label: label.to_string(),
                deviation,
            });
        }
        Ok(())
    }
}

/// Generalized Pauli shift `X_d |j⟩ = |j + 1 mod d⟩`. Real, so it exists
/// over every algebra.
pub fn shift<S: Scalar>(d: usize) -> SiteMatrix<S> {
    let mut data = vec![S::zero(); d * d];
    for j in 0..d {
        data[((j + 1) % d) * d + j] = S::one();
    }
    SiteMatrix { dim: d, data }
}

/// Generalized Pauli clock `Z_d |j⟩ = ω^j |j⟩`, `ω = e^{2πi/d}`. Needs an
/// algebra containing the `d`-th roots of unity (any containing ℂ; over ℝ
/// only `d = 2`).
pub fn clock<S: Scalar>(d: usize) -> Result<SiteMatrix<S>> {
    let mut entries = vec![C64::new(0.0, 0.0); d * d];
    for j in 0..d {
        entries[j * d + j] = crate::math::cis(std::f64::consts::TAU * j as f64 / d as f64);
    }
    snap_real(&mut entries);
    SiteMatrix::try_from_c64s(&format!("clock{d}"), d, &entries)
}

/// The `d`-level discrete Fourier transform (the qudit Hadamard),
/// `F[r][c] = ω^{rc} / √d`. Over ℝ only `d = 2` exists.
pub fn fourier<S: Scalar>(d: usize) -> Result<SiteMatrix<S>> {
    let norm = 1.0 / (d as f64).sqrt();
    let mut entries = vec![C64::new(0.0, 0.0); d * d];
    for r in 0..d {
        for c in 0..d {
            let angle = std::f64::consts::TAU * ((r * c) % d) as f64 / d as f64;
            entries[r * d + c] = crate::math::cis(angle) * norm;
        }
    }
    snap_real(&mut entries);
    SiteMatrix::try_from_c64s(&format!("fourier{d}"), d, &entries)
}

/// The mixed-radix entangler `|a, b⟩ ↦ |a, b + a mod d_b⟩` on a site pair
/// of any radices (`a` is the low sub-digit). A permutation, so it exists
/// over every algebra.
pub fn cshift<S: Scalar>(da: usize, db: usize) -> SiteMatrix<S> {
    let dim = da * db;
    let mut data = vec![S::zero(); dim * dim];
    for a in 0..da {
        for b in 0..db {
            data[(a + da * ((b + a) % db)) * dim + (a + da * b)] = S::one();
        }
    }
    SiteMatrix { dim, data }
}

/// Swap of two sites of equal radix `d` — the only arity pattern for which
/// a swap is dimensionally well-formed.
pub fn swap<S: Scalar>(d: usize) -> SiteMatrix<S> {
    let dim = d * d;
    let mut data = vec![S::zero(); dim * dim];
    for a in 0..d {
        for b in 0..d {
            data[(b + d * a) * dim + (a + d * b)] = S::one();
        }
    }
    SiteMatrix { dim, data }
}

/// Zero the imaginary parts that are rounding residue of real roots of
/// unity (`ω^{d/2} = −1`), so real-valued entries embed over ℝ.
fn snap_real(entries: &mut [C64]) {
    for z in entries.iter_mut() {
        if z.im.abs() < 1e-15 {
            z.im = 0.0;
        }
    }
}

// ── kernels shared by space-native backends ─────────────────────────

/// Check a site-level application against `space`: targets valid and the
/// matrix dimension equal to the product of their radices.
pub(crate) fn validate_sites<S: Scalar>(
    space: &Space,
    matrix: &SiteMatrix<S>,
    sites: &[usize],
) -> Result<()> {
    space.validate_sites(sites)?;
    let expected = space.sub_dim(sites);
    if matrix.dim() != expected {
        return Err(Error::BadDimension {
            expected,
            got: matrix.dim(),
        });
    }
    Ok(())
}

/// Refuse a qubit-level operation on any non-binary target site.
pub(crate) fn require_binary(space: &Space, sites: &[usize], what: &str) -> Result<()> {
    space.validate_sites(sites)?;
    if let Some(&s) = sites.iter().find(|&&s| space.radix(s) != 2) {
        return Err(Error::UnsupportedForSpace {
            operation: format!("{what} on site {s} (radix {})", space.radix(s)),
            space: space.to_string(),
        });
    }
    Ok(())
}

/// Apply a site matrix to a flat mixed-radix amplitude vector in place.
pub(crate) fn apply_sites_in_place<S: Scalar>(
    amps: &mut [S],
    space: &Space,
    m: &SiteMatrix<S>,
    sites: &[usize],
) -> Result<()> {
    let d = m.dim();
    let offsets = space.offsets(sites);
    let data = m.data();
    let mut scratch = vec![S::zero(); d];
    for base in 0..amps.len() as u64 {
        if base % (1 << 20) == 0 {
            crate::guard::checkpoint()?;
        }
        if sites.iter().any(|&s| space.digit(base, s) != 0) {
            continue;
        }
        for (j, slot) in scratch.iter_mut().enumerate() {
            *slot = amps[(base + offsets[j]) as usize];
        }
        for r in 0..d {
            let mut acc = S::zero();
            for (l, &s) in scratch.iter().enumerate() {
                acc = acc + data[r * d + l] * s;
            }
            amps[(base + offsets[r]) as usize] = acc;
        }
    }
    Ok(())
}

// ── circuits over a space ───────────────────────────────────────────

/// One operation of a [`SiteCircuit`].
#[derive(Debug, Clone)]
pub enum SiteOp<S: Scalar> {
    /// A site-level unitary, run through
    /// [`Backend::apply_sites`](crate::backend::Backend::apply_sites).
    Sites {
        /// Display label.
        label: String,
        /// The unitary, validated when the op was added.
        matrix: SiteMatrix<S>,
        /// Target sites (`sites[0]` is the low sub-digit).
        sites: Vec<usize>,
    },
    /// A bound qubit gate placed on radix-2 sites, run through the
    /// backend's qubit kernels (`apply` / `apply_diagonal`).
    Qubit(crate::circuit::BoundGate<S>),
}

/// A gate sequence over a mixed-radix [`Space`]: the site-level
/// counterpart of [`Circuit`](crate::circuit::Circuit).
///
/// Every operation is validated when it is added — targets against the
/// space, matrix dimension against the targets' radices, unitarity to
/// [`UNITARY_TOL`](crate::math::UNITARY_TOL) — so [`SiteCircuit::run`]
/// only has to check that the backend holds the same space. Standard
/// qubit gates join through [`SiteCircuit::place`], which maps a bound
/// qubit circuit onto radix-2 sites and refuses any other target.
#[derive(Debug, Clone)]
pub struct SiteCircuit<S: Scalar = C64> {
    space: Space,
    ops: Vec<SiteOp<S>>,
}

impl<S: Scalar> SiteCircuit<S> {
    /// An empty circuit over `space`.
    pub fn new(space: Space) -> Self {
        SiteCircuit {
            space,
            ops: Vec::new(),
        }
    }

    /// The space this circuit acts on.
    pub fn space(&self) -> &Space {
        &self.space
    }

    /// The operations, in order.
    pub fn ops(&self) -> &[SiteOp<S>] {
        &self.ops
    }

    /// Number of operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the circuit is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Append a site-level unitary.
    pub fn gate(
        &mut self,
        label: impl Into<String>,
        matrix: SiteMatrix<S>,
        sites: &[usize],
    ) -> Result<&mut Self> {
        let label = label.into();
        validate_sites(&self.space, &matrix, sites)?;
        matrix.check_unitary(&label, crate::math::UNITARY_TOL)?;
        self.ops.push(SiteOp::Sites {
            label,
            matrix,
            sites: sites.to_vec(),
        });
        Ok(self)
    }

    /// The radix-`d` Fourier transform on one site.
    pub fn fourier(&mut self, site: usize) -> Result<&mut Self> {
        self.space.validate_sites(&[site])?;
        let d = self.space.radix(site);
        self.gate(format!("fourier{d}"), fourier(d)?, &[site])
    }

    /// The shift `X_d` on one site.
    pub fn shift(&mut self, site: usize) -> Result<&mut Self> {
        self.space.validate_sites(&[site])?;
        let d = self.space.radix(site);
        self.gate(format!("shift{d}"), shift(d), &[site])
    }

    /// The clock `Z_d` on one site.
    pub fn clock(&mut self, site: usize) -> Result<&mut Self> {
        self.space.validate_sites(&[site])?;
        let d = self.space.radix(site);
        self.gate(format!("clock{d}"), clock(d)?, &[site])
    }

    /// `|a, b⟩ ↦ |a, b + a mod d_b⟩` with `control` as `a`.
    pub fn cshift(&mut self, control: usize, target: usize) -> Result<&mut Self> {
        self.space.validate_sites(&[control, target])?;
        let (da, db) = (self.space.radix(control), self.space.radix(target));
        self.gate(
            format!("cshift{da}x{db}"),
            cshift(da, db),
            &[control, target],
        )
    }

    /// Swap two sites; refused unless their radices are equal.
    pub fn swap(&mut self, a: usize, b: usize) -> Result<&mut Self> {
        self.space.validate_sites(&[a, b])?;
        let (da, db) = (self.space.radix(a), self.space.radix(b));
        if da != db {
            return Err(Error::UnsupportedForSpace {
                operation: format!("swap of site {a} (radix {da}) with site {b} (radix {db})"),
                space: self.space.to_string(),
            });
        }
        self.gate(format!("swap{da}"), swap(da), &[a, b])
    }

    /// Place a bound qubit circuit on this space: qubit `q` of `bound`
    /// lands on site `map[q]`, which must have radix 2.
    pub fn place(
        &mut self,
        bound: &crate::circuit::BoundCircuit<S>,
        map: &[usize],
    ) -> Result<&mut Self> {
        if map.len() != bound.num_qubits() {
            return Err(Error::WidthMismatch {
                circuit: bound.num_qubits(),
                backend: map.len(),
            });
        }
        require_binary(&self.space, map, "place")?;
        for g in bound.gates() {
            let mut placed = g.clone();
            placed.qubits = g.qubits.iter().map(|&q| map[q]).collect();
            self.ops.push(SiteOp::Qubit(placed));
        }
        Ok(self)
    }

    /// Run every operation, in order, on `backend`, which must hold this
    /// circuit's space.
    pub fn run(&self, backend: &mut dyn crate::backend::Backend<S>) -> Result<()> {
        let held = backend.space();
        if held != self.space {
            return Err(Error::UnsupportedForSpace {
                operation: format!("a circuit over {}", self.space),
                space: held.to_string(),
            });
        }
        let _scope = crate::guard::enter();
        for op in &self.ops {
            crate::guard::checkpoint()?;
            match op {
                SiteOp::Sites { matrix, sites, .. } => backend.apply_sites(matrix, sites)?,
                SiteOp::Qubit(g) => match &g.kernel {
                    crate::circuit::GateKernel::Matrix(m) => backend.apply(m, &g.qubits)?,
                    crate::circuit::GateKernel::Diagonal(d) => {
                        backend.apply_diagonal(d, &g.qubits)?
                    }
                },
            }
        }
        Ok(())
    }
}
