//! Hierarchical qubit register via algebraic qudits and a dual-algebra.
//!
//! A flat register is `n` two-level sites indexed linearly. This module
//! breaks that shape: the register splits into a **site sector** (any
//! [`Backend`] representation over `m` sites) and an **algebra sector**
//! — `k` further logical qubits carried *inside every stored scalar*,
//! using the Cayley–Dickson tower over ℂ as the qudit space. One
//! `CD^k⟨ℂ⟩` scalar has `2^k` complex coordinates, so a stored
//! amplitude *is* a width-`k` qudit: the logical register is
//! `m + k` qubits wide while the site representation only ever
//! addresses `2^m` indices. The qudit structure is *varied* by
//! construction — choose the split `m / k` per register, and the site
//! sector itself may be any structured representation (sparse support,
//! factored clusters, tree bonds), so logical qubits live at several
//! widths of the hierarchy at once.
//!
//! **Dual-algebra gate synthesis.** Gates on algebra qubits are
//! synthesized from the algebra's own multiplication: the operators
//! `x ↦ (a·x)·b` — the algebra acting on itself from the *left* and
//! the *right* (formally `A ⊗ A^op`, the algebra paired with its
//! opposite: two opposed actions resolving an endomorphism together,
//! the same duality shape as the dual-time engine in [`crate::causal`]).
//! [`dual_algebra_report`] measures, per doubling level, how much of
//! the qudit's real operator space the sandwich span actually resolves
//! and whether every ℂ-linear gate is reachable; [`synthesize_sandwich`]
//! solves a concrete gate into sandwich terms with a measured residual.
//! Nothing is assumed: where the tower's twist breaks a property (the
//! embedded-ℂ action stops being component-linear past ℍ; sedenion
//! zero divisors), the report says so and the register routes around it.
//!
//! **Honesty about feasibility.** For *arbitrary* states this is a
//! reshaping, not a compression: `2^m` sites of `2^k`-dimensional
//! qudits store exactly `2^{m+k}` complex numbers. What the hierarchy
//! buys, measured in the tests: (1) the site sector keeps its
//! structure (sparse support counts **site** indices — the algebra
//! sector rides inside each entry); (2) gates on site qubits run
//! natively on the inner backend whenever the embedded-ℂ action is
//! verified component-linear (measured at construction, true for ℍ);
//! (3) the logical width ceiling breaks: every flat backend in this
//! crate indexes states by `u64` and stops at 63 qubits, while an
//! algebraic register reaches `m + k` logical qubits with exact
//! amplitudes addressed as (site index, qudit component) pairs.

use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::math::{svd_thin, GateMatrix};
use crate::scalar::{Scalar, C64};
use std::collections::HashMap;

/// Number of algebra qubits a scalar can carry: `log2` of its complex
/// coordinate count (0 for ℂ itself, 1 for ℍ, 2 for 𝕆, 3 for 𝕊).
pub fn algebra_capacity<A: Scalar>() -> usize {
    debug_assert!(A::DIM.is_power_of_two() && A::DIM >= 2);
    (A::DIM / 2).trailing_zeros() as usize
}

/// The complex coordinates of a scalar, pairing consecutive real
/// coefficients: component `c` is `coeffs[2c] + i·coeffs[2c+1]`.
/// Component-index bit `j` is algebra qubit `j`; the top bit selects
/// the outermost Cayley–Dickson half.
pub fn components<A: Scalar>(x: A) -> Vec<C64> {
    let co = x.coeffs();
    co.chunks_exact(2).map(|p| C64::new(p[0], p[1])).collect()
}

/// Rebuild a scalar from its complex coordinates (inverse of
/// [`components`]).
pub fn from_components<A: Scalar>(comps: &[C64]) -> A {
    let mut co = Vec::with_capacity(A::DIM);
    for z in comps {
        co.push(z.re);
        co.push(z.im);
    }
    A::from_coeffs(&co)
}

/// Whether left multiplication by embedded complex numbers acts
/// component-wise on this algebra's complex coordinates — measured on
/// random probes, never assumed. True for ℂ and ℍ; false from 𝕆 on,
/// where the Cayley–Dickson twist conjugates a coordinate. Gates on
/// site qubits may run natively through the inner backend exactly when
/// this holds.
pub fn embedded_action_is_component_linear<A: Scalar>() -> bool {
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for _ in 0..8 {
        let x = A::from_coeffs(&(0..A::DIM).map(|_| next()).collect::<Vec<_>>());
        let z = C64::new(next(), next());
        let embedded = A::try_from_c64(z).expect("C embeds in the CD tower");
        let lhs = components::<A>(embedded * x);
        let rhs: Vec<C64> = components::<A>(x).into_iter().map(|c| z * c).collect();
        if lhs
            .iter()
            .zip(&rhs)
            .any(|(l, r)| (l - r).norm_sqr() > 1e-20)
        {
            return false;
        }
    }
    true
}

/// One term of a dual-algebra operator: `x ↦ (left · x) · right`
/// (evaluated in exactly that order — it matters once associativity is
/// gone).
#[derive(Debug, Clone)]
pub struct SandwichTerm<A: Scalar> {
    /// Left factor.
    pub left: A,
    /// Right factor.
    pub right: A,
}

/// A sum of dual-algebra sandwich terms — a qudit gate expressed in
/// the algebra's own multiplication.
#[derive(Debug, Clone)]
pub struct SandwichOp<A: Scalar> {
    /// The terms; the operator is their sum.
    pub terms: Vec<SandwichTerm<A>>,
}

impl<A: Scalar> SandwichOp<A> {
    /// Apply to a scalar: `Σᵢ (leftᵢ · x) · rightᵢ`.
    pub fn apply(&self, x: A) -> A {
        let mut out = A::zero();
        for t in &self.terms {
            out = out + (t.left * x) * t.right;
        }
        out
    }
}

/// The real matrix (on the algebra's coefficient space) of
/// `x ↦ (basis(i) · x) · basis(j)`, column-major over basis inputs.
fn sandwich_matrix<A: Scalar>(i: usize, j: usize) -> Vec<f64> {
    let dim = A::DIM;
    let (bi, bj) = (A::basis(i), A::basis(j));
    let mut m = vec![0.0; dim * dim];
    for c in 0..dim {
        let y = (bi * A::basis(c)) * bj;
        for (r, v) in y.coeffs().iter().enumerate() {
            m[r * dim + c] = *v;
        }
    }
    m
}

/// Realify a complex matrix over the component space into the
/// coefficient space (interleaved re/im coordinates).
fn realify(dim_c: usize, m: &[C64]) -> Vec<f64> {
    let d = 2 * dim_c;
    let mut r = vec![0.0; d * d];
    for row in 0..dim_c {
        for col in 0..dim_c {
            let z = m[row * dim_c + col];
            r[(2 * row) * d + 2 * col] = z.re;
            r[(2 * row) * d + 2 * col + 1] = -z.im;
            r[(2 * row + 1) * d + 2 * col] = z.im;
            r[(2 * row + 1) * d + 2 * col + 1] = z.re;
        }
    }
    r
}

/// The dual-algebra operator space of one scalar type, measured.
#[derive(Debug, Clone, PartialEq)]
pub struct DualAlgebraReport {
    /// Real dimension of the sandwich span `{L_{eᵢ} R_{eⱼ}}`.
    pub sandwich_rank: usize,
    /// Real dimension of the full operator space `End_ℝ` of the qudit.
    pub operator_space: usize,
    /// Worst synthesis residual over a basis of all ℂ-linear qudit
    /// gates — ≈ 0 means every gate on the algebra qubits is a
    /// dual-algebra sandwich.
    pub max_linear_residual: f64,
    /// Whether the embedded-ℂ action is component-linear (measured).
    pub embedded_linear: bool,
}

fn sandwich_basis<A: Scalar>() -> (usize, Vec<f64>) {
    let dim = A::DIM;
    let n = dim * dim; // operator coordinates
    let mut basis = vec![0.0; n * n]; // column per (i, j) pair
    for i in 0..dim {
        for j in 0..dim {
            let col = i * dim + j;
            for (r, v) in sandwich_matrix::<A>(i, j).iter().enumerate() {
                basis[r * n + col] = *v;
            }
        }
    }
    (n, basis)
}

/// Precomputed least-squares machinery over the sandwich span: the
/// basis SVD runs once and serves many targets.
struct SandwichSolver {
    n: usize,
    basis: Vec<f64>,
    svd: crate::math::Svd<f64>,
}

impl SandwichSolver {
    fn build<A: Scalar>() -> Result<Self> {
        let (n, basis) = sandwich_basis::<A>();
        let svd = svd_thin::<f64>(n, n, &basis, 1e-10, 1e-11)?;
        Ok(SandwichSolver { n, basis, svd })
    }

    /// Pseudo-solve `basis · c = target` (`c = V Σ⁻¹ Uᵀ t`); returns
    /// the coefficients and the measured residual.
    fn solve(&self, target_real: &[f64]) -> (Vec<f64>, f64) {
        let n = self.n;
        let mut ut = vec![0.0; self.svd.rank];
        for (r, slot) in ut.iter_mut().enumerate() {
            let acc: f64 = target_real
                .iter()
                .enumerate()
                .map(|(row, t)| self.svd.u[row * self.svd.rank + r] * t)
                .sum();
            *slot = acc / self.svd.sigma[r];
        }
        let mut coeff = vec![0.0; n];
        for (c, slot) in coeff.iter_mut().enumerate() {
            *slot = ut
                .iter()
                .enumerate()
                .map(|(r, u)| self.svd.vt[r * n + c] * u)
                .sum();
        }
        let residual: f64 = target_real
            .iter()
            .enumerate()
            .map(|(row, t)| {
                let acc: f64 = coeff
                    .iter()
                    .enumerate()
                    .map(|(col, c)| self.basis[row * n + col] * c)
                    .sum();
                (acc - t).powi(2)
            })
            .sum();
        (coeff, residual.sqrt())
    }
}

fn op_from_coeffs<A: Scalar>(coeff: &[f64]) -> SandwichOp<A> {
    let dim = A::DIM;
    let mut terms = Vec::new();
    for i in 0..dim {
        for j in 0..dim {
            let c = coeff[i * dim + j];
            if c.abs() > 1e-12 {
                terms.push(SandwichTerm {
                    left: A::basis(i).scale(c),
                    right: A::basis(j),
                });
            }
        }
    }
    SandwichOp { terms }
}

/// Synthesize a complex gate on the algebra qubits (a
/// `2^k_cap × 2^k_cap` matrix over the scalar's complex components)
/// into dual-algebra sandwich terms. Returns the operator and the
/// measured residual — exactly zero-ish means the gate *is* a sum of
/// left×right multiplications.
pub fn synthesize_sandwich<A: Scalar>(matrix_c: &[C64]) -> Result<(SandwichOp<A>, f64)> {
    let dim_c = A::DIM / 2;
    if matrix_c.len() != dim_c * dim_c {
        return Err(Error::BadDimension {
            expected: dim_c * dim_c,
            got: matrix_c.len(),
        });
    }
    let solver = SandwichSolver::build::<A>()?;
    let (coeff, residual) = solver.solve(&realify(dim_c, matrix_c));
    Ok((op_from_coeffs::<A>(&coeff), residual))
}

/// Measure the dual-algebra operator space of scalar `A`: the rank of
/// the sandwich span, and the worst residual synthesizing a basis of
/// every ℂ-linear qudit gate.
pub fn dual_algebra_report<A: Scalar>() -> Result<DualAlgebraReport> {
    let dim_c = A::DIM / 2;
    let solver = SandwichSolver::build::<A>()?;
    let mut worst = 0.0f64;
    for r in 0..dim_c {
        for c in 0..dim_c {
            for phase in [C64::new(1.0, 0.0), C64::new(0.0, 1.0)] {
                let mut m = vec![C64::new(0.0, 0.0); dim_c * dim_c];
                m[r * dim_c + c] = phase;
                let (_, residual) = solver.solve(&realify(dim_c, &m));
                worst = worst.max(residual);
            }
        }
    }
    Ok(DualAlgebraReport {
        sandwich_rank: solver.svd.rank,
        operator_space: solver.n,
        max_linear_residual: worst,
        embedded_linear: embedded_action_is_component_linear::<A>(),
    })
}

/// Execution counters of an [`AlgebraicRegister`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuditStats {
    /// Gates on site qubits delegated natively to the inner backend.
    pub native_site_gates: usize,
    /// Gates routed through the component path (algebra-touching, or
    /// site gates on a scalar whose embedded action is not
    /// component-linear).
    pub component_gates: usize,
}

/// The hierarchical register: `site_qubits` handled by any inner
/// [`Backend`] over the algebra `A`, plus `algebra_qubits` carried in
/// each scalar's complex components. Implements `Backend<C64>` over
/// the full `site + algebra` logical width (trait index methods
/// require the total to fit `u64`; wider registers use the
/// `*_parts` accessors).
///
/// Logical qubit `q < site_qubits` is site bit `q`; logical qubit
/// `q ≥ site_qubits` is algebra component bit `q − site_qubits`.
pub struct AlgebraicRegister<A: Scalar> {
    inner: Box<dyn Backend<A>>,
    site_qubits: usize,
    algebra_qubits: usize,
    embedded_linear: bool,
    stats: QuditStats,
}

impl<A: Scalar> AlgebraicRegister<A> {
    /// Wrap an inner site-sector backend, carrying `algebra_qubits`
    /// further logical qubits inside each scalar. Fails if the inner
    /// width mismatches, or the scalar cannot carry that many qubits.
    pub fn new(
        site_qubits: usize,
        algebra_qubits: usize,
        inner: Box<dyn Backend<A>>,
    ) -> Result<Self> {
        if inner.num_qubits() != site_qubits {
            return Err(Error::WidthMismatch {
                circuit: site_qubits,
                backend: inner.num_qubits(),
            });
        }
        let capacity = algebra_capacity::<A>();
        if algebra_qubits > capacity {
            return Err(Error::InvalidState(format!(
                "{} carries at most {capacity} algebra qubits, asked for {algebra_qubits}",
                A::algebra_name()
            )));
        }
        Ok(AlgebraicRegister {
            inner,
            site_qubits,
            algebra_qubits,
            embedded_linear: embedded_action_is_component_linear::<A>(),
            stats: QuditStats::default(),
        })
    }

    /// Site-sector width.
    pub fn site_qubits(&self) -> usize {
        self.site_qubits
    }

    /// Algebra-sector width.
    pub fn algebra_qubits(&self) -> usize {
        self.algebra_qubits
    }

    /// Total logical width — valid beyond 63, unlike any flat index.
    pub fn logical_qubits(&self) -> usize {
        self.site_qubits + self.algebra_qubits
    }

    /// Whether site gates run natively on the inner backend (measured
    /// embedded-ℂ component-linearity of the scalar).
    pub fn native_site_path(&self) -> bool {
        self.embedded_linear
    }

    /// Execution counters.
    pub fn stats(&self) -> QuditStats {
        self.stats
    }

    /// Stored site-sector support (the algebra sector rides inside
    /// each entry).
    pub fn site_support(&self) -> usize {
        self.inner.nonzero_count()
    }

    /// Amplitude by (site index, algebra component) — the wide
    /// addressing that works at any logical width.
    pub fn amplitude_parts(&self, site: u64, component: usize) -> C64 {
        components::<A>(self.inner.amplitude(site))[component]
    }

    /// Visit nonzero amplitudes as (site index, component, value) —
    /// wide-safe.
    pub fn for_each_nonzero_parts(&self, f: &mut dyn FnMut(u64, usize, C64)) {
        let kc = 1usize << self.algebra_qubits;
        self.inner.for_each_nonzero(&mut |site, x| {
            for (c, z) in components::<A>(x).into_iter().take(kc).enumerate() {
                if z.norm_sqr() > 0.0 {
                    f(site, c, z);
                }
            }
        });
    }

    fn assert_indexable(&self) {
        assert!(
            self.logical_qubits() <= 63,
            "{} logical qubits exceed u64 flat indexing — use the *_parts accessors",
            self.logical_qubits()
        );
    }

    fn gather(&self) -> Vec<(u64, Vec<C64>)> {
        let mut entries = Vec::new();
        self.inner.for_each_nonzero(&mut |site, x| {
            entries.push((site, components::<A>(x)));
        });
        entries
    }

    fn store(&mut self, entries: Vec<(u64, Vec<C64>)>) -> Result<()> {
        let packed: Vec<(u64, A)> = entries
            .into_iter()
            .map(|(site, comps)| (site, from_components::<A>(&comps)))
            .filter(|(_, x)| x.abs_sqr() > 0.0)
            .collect();
        self.inner.load(&packed)
    }

    /// The component path: apply an arbitrary gate over mixed
    /// site/algebra qubits by bucketing stored entries on the involved
    /// site bits and multiplying the local blocks out in ℂ — exact,
    /// structural (no flat `site+algebra` index is ever formed), and
    /// `O(support · 2^{algebra})`.
    fn component_apply(&mut self, matrix: &GateMatrix<C64>, qubits: &[usize]) -> Result<()> {
        let t = qubits.len();
        let cap = A::DIM / 2;
        let site_positions: Vec<(usize, usize)> = qubits
            .iter()
            .enumerate()
            .filter(|(_, &q)| q < self.site_qubits)
            .map(|(g, &q)| (g, q))
            .collect();
        let alg_positions: Vec<(usize, usize)> = qubits
            .iter()
            .enumerate()
            .filter(|(_, &q)| q >= self.site_qubits)
            .map(|(g, &q)| (g, q - self.site_qubits))
            .collect();
        let site_mask: u64 = site_positions.iter().map(|&(_, q)| 1u64 << q).sum();
        let alg_mask: usize = alg_positions.iter().map(|&(_, b)| 1usize << b).sum();

        // Bucket entries by the site bits the gate does not touch.
        let mut buckets: HashMap<u64, HashMap<u64, Vec<C64>>> = HashMap::new();
        for (site, comps) in self.gather() {
            buckets
                .entry(site & !site_mask)
                .or_default()
                .insert(site & site_mask, comps);
        }

        let mut out: Vec<(u64, Vec<C64>)> = Vec::new();
        for (base, mut branches) in buckets {
            // Every site-branch pattern the gate can reach.
            let patterns: Vec<u64> = (0..1u64 << site_positions.len())
                .map(|p| {
                    site_positions
                        .iter()
                        .enumerate()
                        .filter(|(bit, _)| (p >> bit) & 1 == 1)
                        .map(|(_, &(_, q))| 1u64 << q)
                        .sum()
                })
                .collect();
            let mut local: HashMap<u64, Vec<C64>> = HashMap::new();
            for &p in &patterns {
                local.insert(
                    p,
                    branches
                        .remove(&p)
                        .unwrap_or_else(|| vec![C64::new(0.0, 0.0); cap]),
                );
            }
            // For each spectator component pattern, one 2^t block.
            for spec in 0..cap {
                if spec & alg_mask != 0 {
                    continue;
                }
                let mut v = vec![C64::new(0.0, 0.0); 1 << t];
                for l in 0..1u64 << t {
                    let mut sbits = 0u64;
                    for (bit, &(g, q)) in site_positions.iter().enumerate() {
                        let _ = bit;
                        if (l >> g) & 1 == 1 {
                            sbits |= 1u64 << q;
                        }
                    }
                    let mut comp = spec;
                    for &(g, b) in &alg_positions {
                        if (l >> g) & 1 == 1 {
                            comp |= 1 << b;
                        }
                    }
                    v[l as usize] = local[&sbits][comp];
                }
                let mut w = vec![C64::new(0.0, 0.0); 1 << t];
                for (r, slot) in w.iter_mut().enumerate() {
                    let mut acc = C64::new(0.0, 0.0);
                    for (c, amp) in v.iter().enumerate() {
                        acc += matrix.get(r, c) * amp;
                    }
                    *slot = acc;
                }
                for (l, value) in w.into_iter().enumerate() {
                    let mut sbits = 0u64;
                    for &(g, q) in &site_positions {
                        if (l >> g) & 1 == 1 {
                            sbits |= 1u64 << q;
                        }
                    }
                    let mut comp = spec;
                    for &(g, b) in &alg_positions {
                        if (l >> g) & 1 == 1 {
                            comp |= 1 << b;
                        }
                    }
                    local.get_mut(&sbits).expect("pattern present")[comp] = value;
                }
            }
            for (p, comps) in local {
                out.push((base | p, comps));
            }
        }
        self.stats.component_gates += 1;
        self.store(out)
    }
}

impl<A: Scalar> Backend<C64> for AlgebraicRegister<A> {
    fn name(&self) -> &str {
        "algebraic"
    }

    fn num_qubits(&self) -> usize {
        self.logical_qubits()
    }

    fn apply(&mut self, matrix: &GateMatrix<C64>, qubits: &[usize]) -> Result<()> {
        let n = self.logical_qubits();
        if matrix.dim() != 1 << qubits.len() {
            return Err(Error::BadDimension {
                expected: 1 << qubits.len(),
                got: matrix.dim(),
            });
        }
        for (i, &q) in qubits.iter().enumerate() {
            if q >= n {
                return Err(Error::QubitOutOfRange {
                    qubit: q,
                    num_qubits: n,
                });
            }
            if qubits[..i].contains(&q) {
                return Err(Error::DuplicateQubits {
                    qubits: qubits.to_vec(),
                });
            }
        }
        let site_only = qubits.iter().all(|&q| q < self.site_qubits);
        if site_only && self.embedded_linear {
            let embedded = GateMatrix::<A>::try_from_c64s(matrix.dim(), matrix.data())
                .expect("C embeds in the CD tower");
            self.stats.native_site_gates += 1;
            return self.inner.apply(&embedded, qubits);
        }
        self.component_apply(matrix, qubits)
    }

    fn apply_diagonal(&mut self, entries: &[C64], qubits: &[usize]) -> Result<()> {
        let n = self.logical_qubits();
        for &q in qubits {
            if q >= n {
                return Err(Error::QubitOutOfRange {
                    qubit: q,
                    num_qubits: n,
                });
            }
        }
        let site_only = qubits.iter().all(|&q| q < self.site_qubits);
        if site_only && self.embedded_linear {
            let embedded: Vec<A> = entries
                .iter()
                .map(|&z| A::try_from_c64(z).expect("C embeds in the CD tower"))
                .collect();
            self.stats.native_site_gates += 1;
            return self.inner.apply_diagonal(&embedded, qubits);
        }
        // Diagonals never mix branches: phase each stored coordinate.
        let kc = 1usize << self.algebra_qubits;
        let gathered = self.gather();
        let mut out = Vec::with_capacity(gathered.len());
        for (site, mut comps) in gathered {
            for (c, z) in comps.iter_mut().enumerate().take(kc) {
                let mut sub = 0usize;
                for (g, &q) in qubits.iter().enumerate() {
                    let bit = if q < self.site_qubits {
                        ((site >> q) & 1) as usize
                    } else {
                        (c >> (q - self.site_qubits)) & 1
                    };
                    sub |= bit << g;
                }
                *z *= entries[sub];
            }
            out.push((site, comps));
        }
        self.stats.component_gates += 1;
        self.store(out)
    }

    fn amplitude(&self, index: u64) -> C64 {
        self.assert_indexable();
        let site = index & ((1u64 << self.site_qubits) - 1);
        let comp = (index >> self.site_qubits) as usize;
        self.amplitude_parts(site, comp)
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, C64)) {
        self.assert_indexable();
        let shift = self.site_qubits;
        self.for_each_nonzero_parts(&mut |site, c, z| f(site | ((c as u64) << shift), z));
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        if qubit < self.site_qubits {
            self.inner.project(qubit, outcome, renorm);
            return;
        }
        let bit = qubit - self.site_qubits;
        let gathered = self.gather();
        let mut out = Vec::with_capacity(gathered.len());
        for (site, mut comps) in gathered {
            for (c, z) in comps.iter_mut().enumerate() {
                if ((c >> bit) & 1 == 1) != outcome {
                    *z = C64::new(0.0, 0.0);
                } else {
                    *z *= renorm;
                }
            }
            out.push((site, comps));
        }
        self.store(out).expect("projection preserves validity");
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn load(&mut self, entries: &[(u64, C64)]) -> Result<()> {
        self.assert_indexable();
        let cap = A::DIM / 2;
        let kc = 1usize << self.algebra_qubits;
        let mut per_site: HashMap<u64, Vec<C64>> = HashMap::new();
        for &(index, z) in entries {
            if self.logical_qubits() < 64 && index >> self.logical_qubits() != 0 {
                return Err(Error::QubitOutOfRange {
                    qubit: 64 - index.leading_zeros() as usize,
                    num_qubits: self.logical_qubits(),
                });
            }
            let site = index & ((1u64 << self.site_qubits) - 1);
            let comp = (index >> self.site_qubits) as usize;
            if comp >= kc {
                return Err(Error::QubitOutOfRange {
                    qubit: self.site_qubits + kc.trailing_zeros() as usize,
                    num_qubits: self.logical_qubits(),
                });
            }
            per_site
                .entry(site)
                .or_insert_with(|| vec![C64::new(0.0, 0.0); cap])[comp] += z;
        }
        self.store(per_site.into_iter().collect())
    }

    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
