//! Krein metrics over a mixed-radix [`Space`]: any signature `(p, q)`,
//! and the pseudo-orthogonal algebras `so(p, q)` on a single site.
//!
//! [`crate::krein`] puts the indefinite metric on qubits as a Pauli
//! string, and a non-trivial Pauli string always splits the register in
//! half: its signatures are exactly `(2^{n−1}, 2^{n−1})`. That covers
//! twistor space, `(2, 2)`, and nothing unequal — not ℝ^{3,2}, the
//! ambient space of AdS₄, nor ℝ^{3,1} itself. A qudit site removes the
//! restriction: give a site of radix `p + q` the metric
//! `diag(+1 ×p, −1 ×q)` and that site **is** ℝ^{p,q} (complexified), so
//! a register's state is a vector of the pseudo-Euclidean space and the
//! gates that preserve the metric are its symmetry group.
//!
//! ## The metric
//!
//! A [`SiteMetric`] is a tensor product over the sites of a [`Space`] of
//! one Hermitian involution per site (`H = H†`, `H² = 1`; identity by
//! default) — a diagonal sign vector ([`SiteMetric::signs`]) or any
//! involution matrix ([`SiteMetric::involution`]). A site of radix `r`
//! with involution `H` has signature `((r + tr H)/2, (r − tr H)/2)`, and
//! signatures multiply: `(p₁, q₁) ⊗ (p₂, q₂) = (p₁p₂ + q₁q₂, p₁q₂ + q₁p₂)`.
//! Every Pauli-string metric of [`crate::krein::KreinMetric`] is the
//! all-qubit special case, and the tests check the two agree.
//!
//! [`SiteKreinRegister`] is the register: any `Backend<C64>` whose
//! [`Backend::space`] is the metric's space, with gates checked for
//! pseudo-unitarity against the metric restricted to their sites
//! (refused by name otherwise) and the same [`KreinLedger`] as the qubit
//! register. Dense and sparse backends hold mixed-radix spaces natively.
//!
//! ## `so(p, q)` on one site
//!
//! [`so_algebra`] is the real Lie algebra `{X real : Xᵀη + ηX = 0}` in
//! its defining representation, dimension `(p+q)(p+q−1)/2`, and
//! [`crate::krein::killing_signature_sites`] names its real form.
//! Measured (`tests/krein_sites.rs`), with the non-compact directions
//! positive:
//!
//! | signature | algebra   | Killing `(+, −, 0)` | also realised as |
//! |-----------|-----------|---------------------|------------------|
//! | (3, 1)    | so(3,1)   | (3, 3, 0)           | the Lorentz algebra |
//! | (2, 2)    | so(2,2)   | (4, 2, 0)           | `sl(2,ℝ) ⊕ sl(2,ℝ)` |
//! | (3, 2)    | so(3,2)   | (6, 4, 0)           | the AdS₄ stabilizer of a rank-4 infinity twistor |
//! | (4, 1)    | so(4,1)   | (4, 6, 0)           | the dS₄ stabilizer |
//! | (4, 2)    | so(4,2)   | (8, 7, 0)           | `su(2,2)` on twistors |
//! | (3, 3)    | so(3,3)   | (9, 6, 0)           | `sl(4,ℝ)` |
//! | (4, 4)    | so(4,4)   | (16, 12, 0)         | the triality algebra |
//!
//! The third column is a cross-check across representations: the 5-dim
//! vector of `so(3,2)` on one qudit and the 4-dim twistor `sp(4,ℝ)` on
//! two qubits give the same inertia, as `Spin(3,2) ≅ Sp(4,ℝ)` says they
//! must.
//!
//! **AdS₄ as a state.** A real vector `x` on a `(3, 2)` site with
//! `⟨x, x⟩_η = −R²` is a point of the hyperboloid whose induced metric is
//! Lorentzian `(3, 1)`: AdS₄. The `so(3,2)` gates are real, so they move
//! such a state along the hyperboloid, `η`-norm conserved exactly, while
//! the path weight `Σ x_k²` (the Euclidean distance from the origin of the
//! embedding, not anything physical) grows along the non-compact
//! directions. Measured in the tests.
//!
//! ## Boundaries
//!
//! As in [`crate::krein`]: the `η`-norm is conserved but is not a Born
//! rule, and non-compact gates spend precision, not memory. Metrics are
//! products over sites; an entangling metric (one that is not a product)
//! is not expressible here, and nothing in this module pretends a
//! product metric covers it.

use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::krein::{KreinLedger, KREIN_TOL};
use crate::scalar::C64;
use crate::space::{SiteMatrix, Space};

fn c(re: f64) -> C64 {
    C64::new(re, 0.0)
}

fn max_dev(a: &SiteMatrix<C64>, b: &SiteMatrix<C64>) -> f64 {
    a.data()
        .iter()
        .zip(b.data())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).norm()))
}

fn dagger(m: &SiteMatrix<C64>) -> SiteMatrix<C64> {
    let d = m.dim();
    let data = (0..d * d)
        .map(|k| m.get(k % d, k / d).conj())
        .collect::<Vec<_>>();
    SiteMatrix::from_vec(d, data).expect("square")
}

/// A product metric `η = ⊗_k H_k` over the sites of a [`Space`]. See the
/// module docs.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteMetric {
    space: Space,
    factors: Vec<Option<SiteMatrix<C64>>>,
}

impl SiteMetric {
    /// The positive-definite metric on `space` (every factor the
    /// identity).
    pub fn new(space: Space) -> Self {
        let factors = vec![None; space.sites()];
        SiteMetric { space, factors }
    }

    /// One site of radix `p + q` carrying `diag(+1 ×p, −1 ×q)`: the
    /// space ℝ^{p,q}.
    pub fn split(p: usize, q: usize) -> Result<Self> {
        let space = Space::new(&[p + q])?;
        let signs: Vec<i8> = (0..p + q).map(|k| if k < p { 1 } else { -1 }).collect();
        SiteMetric::new(space).signs(0, &signs)
    }

    /// Set site `site`'s factor to `diag(signs)`; every entry must be ±1
    /// and there must be one per level.
    pub fn signs(self, site: usize, signs: &[i8]) -> Result<Self> {
        self.space.validate_sites(&[site])?;
        let r = self.space.radix(site);
        if signs.len() != r || signs.iter().any(|s| s.abs() != 1) {
            return Err(Error::InvalidState(format!(
                "site metric: site {site} has radix {r}; need {r} signs of ±1, got {signs:?}"
            )));
        }
        let mut data = vec![c(0.0); r * r];
        for (k, &s) in signs.iter().enumerate() {
            data[k * r + k] = c(f64::from(s));
        }
        self.involution(site, SiteMatrix::from_vec(r, data)?)
    }

    /// Set site `site`'s factor to the Hermitian involution `h`
    /// (`h = h†`, `h² = 1`, checked to [`KREIN_TOL`]).
    pub fn involution(mut self, site: usize, h: SiteMatrix<C64>) -> Result<Self> {
        self.space.validate_sites(&[site])?;
        let r = self.space.radix(site);
        if h.dim() != r {
            return Err(Error::BadDimension {
                expected: r,
                got: h.dim(),
            });
        }
        let herm = max_dev(&h, &dagger(&h));
        let invol = max_dev(&h.matmul(&h)?, &SiteMatrix::identity(r));
        if herm > KREIN_TOL || invol > KREIN_TOL {
            return Err(Error::InvalidState(format!(
                "site metric: site {site}'s factor is not a Hermitian involution \
                 (|H − H†| = {herm:.3e}, |H² − 1| = {invol:.3e})"
            )));
        }
        self.factors[site] = Some(h);
        Ok(self)
    }

    /// The space the metric lives on.
    pub fn space(&self) -> &Space {
        &self.space
    }

    /// Signature `(p, q)` of one site's factor.
    pub fn site_signature(&self, site: usize) -> (u128, u128) {
        let r = self.space.radix(site);
        match &self.factors[site] {
            None => (r as u128, 0),
            Some(h) => {
                let tr: f64 = (0..r).map(|k| h.get(k, k).re).sum();
                let p = ((r as f64 + tr) / 2.0).round() as u128;
                (p, r as u128 - p)
            }
        }
    }

    /// Signature `(p, q)` of the whole metric, by the product rule.
    pub fn signature(&self) -> (u128, u128) {
        (0..self.space.sites()).fold((1, 0), |(p, q), k| {
            let (a, b) = self.site_signature(k);
            (p * a + q * b, p * b + q * a)
        })
    }

    /// `η` restricted to `sites`, over their mixed-radix sub-index
    /// (`sites[0]` the low digit).
    pub fn restricted(&self, sites: &[usize]) -> Result<SiteMatrix<C64>> {
        self.space.validate_sites(sites)?;
        let radices: Vec<usize> = sites.iter().map(|&s| self.space.radix(s)).collect();
        let d: usize = radices.iter().product();
        let digits = |mut x: usize| -> Vec<usize> {
            radices
                .iter()
                .map(|&r| {
                    let v = x % r;
                    x /= r;
                    v
                })
                .collect()
        };
        let mut data = vec![c(0.0); d * d];
        for row in 0..d {
            let rd = digits(row);
            for col in 0..d {
                let cd = digits(col);
                let mut v = c(1.0);
                for (b, &s) in sites.iter().enumerate() {
                    v *= match &self.factors[s] {
                        None if rd[b] == cd[b] => c(1.0),
                        None => c(0.0),
                        Some(h) => h.get(rd[b], cd[b]),
                    };
                }
                data[row * d + col] = v;
            }
        }
        SiteMatrix::from_vec(d, data)
    }

    /// `max |(M†η_SM − η_S)_{rc}|` for a gate `m` on `sites`.
    pub fn pseudo_unitarity_deviation(&self, m: &SiteMatrix<C64>, sites: &[usize]) -> Result<f64> {
        let eta = self.restricted(sites)?;
        if m.dim() != eta.dim() {
            return Err(Error::BadDimension {
                expected: eta.dim(),
                got: m.dim(),
            });
        }
        Ok(max_dev(&dagger(m).matmul(&eta)?.matmul(m)?, &eta))
    }

    /// The boost mixing levels `a` and `b` of a site whose factor is
    /// diagonal with opposite signs there: `cosh θ` on both, `sinh θ`
    /// between them, identity elsewhere. Pseudo-unitary and not unitary —
    /// the qudit ghost pump.
    pub fn level_boost(
        &self,
        site: usize,
        a: usize,
        b: usize,
        theta: f64,
    ) -> Result<SiteMatrix<C64>> {
        self.space.validate_sites(&[site])?;
        let r = self.space.radix(site);
        let refuse = |why: &str| {
            Err(Error::InvalidState(format!(
                "level boost on site {site} between levels {a} and {b}: {why}"
            )))
        };
        if a >= r || b >= r || a == b {
            return refuse("levels out of range or equal");
        }
        let Some(h) = &self.factors[site] else {
            return refuse("the site's metric is definite");
        };
        let diagonal = (0..r).all(|i| (0..r).all(|j| i == j || h.get(i, j).norm() == 0.0));
        if !diagonal || h.get(a, a).re * h.get(b, b).re >= 0.0 {
            return refuse("needs a diagonal factor with opposite signs at the two levels");
        }
        let mut m = SiteMatrix::<C64>::identity(r).data().to_vec();
        let (ch, sh) = (theta.cosh(), theta.sinh());
        m[a * r + a] = c(ch);
        m[b * r + b] = c(ch);
        m[a * r + b] = c(sh);
        m[b * r + a] = c(sh);
        SiteMatrix::from_vec(r, m)
    }

    /// `⟨ψ|η|ψ⟩`, evaluated on the backend's nonzero amplitudes. Each
    /// non-diagonal factor fans an amplitude out over its column's
    /// nonzeros, so the cost is the support times that product.
    pub fn eta_norm(&self, state: &dyn Backend<C64>) -> Result<f64> {
        if state.space() != self.space {
            return Err(Error::InvalidState(format!(
                "site metric: metric space {} but state space {}",
                self.space,
                state.space()
            )));
        }
        let mut acc = c(0.0);
        state.for_each_nonzero(&mut |i, amp| {
            // η|i⟩ = Σ_targets coeff |target⟩, built site by site.
            let mut images: Vec<(u64, C64)> = vec![(i, c(1.0))];
            for (k, f) in self.factors.iter().enumerate() {
                let Some(h) = f else { continue };
                let col = self.space.digit(i, k);
                let mut next = Vec::new();
                for &(idx, coef) in &images {
                    for row in 0..h.dim() {
                        let e = h.get(row, col);
                        if e.norm() != 0.0 {
                            next.push((self.space.with_digit(idx, k, row), coef * e));
                        }
                    }
                }
                images = next;
            }
            for (t, coef) in images {
                acc += state.amplitude(t).conj() * coef * amp;
            }
        });
        Ok(acc.re)
    }
}

/// A register with a [`SiteMetric`]: any `Backend<C64>` on the metric's
/// space. See the module docs.
pub struct SiteKreinRegister {
    state: Box<dyn Backend<C64>>,
    metric: SiteMetric,
    tol: f64,
}

impl SiteKreinRegister {
    /// Wrap `state`; its space must be the metric's.
    pub fn new(state: Box<dyn Backend<C64>>, metric: SiteMetric) -> Result<Self> {
        if state.space() != *metric.space() {
            return Err(Error::InvalidState(format!(
                "site Krein register: metric space {} but state space {}",
                metric.space(),
                state.space()
            )));
        }
        Ok(SiteKreinRegister {
            state,
            metric,
            tol: KREIN_TOL,
        })
    }

    /// The pseudo-unitarity tolerance (default [`KREIN_TOL`]).
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// The metric.
    pub fn metric(&self) -> &SiteMetric {
        &self.metric
    }

    /// The underlying state.
    pub fn state(&self) -> &dyn Backend<C64> {
        self.state.as_ref()
    }

    /// Apply `m` on `sites` if it preserves the metric; refuse by name
    /// (with the measured deviation) otherwise.
    pub fn apply_sites(&mut self, m: &SiteMatrix<C64>, sites: &[usize]) -> Result<()> {
        let dev = self.metric.pseudo_unitarity_deviation(m, sites)?;
        if dev > self.tol {
            return Err(Error::InvalidState(format!(
                "site Krein register: gate on sites {sites:?} is not pseudo-unitary \
                 for the metric (deviation {dev:.3e})"
            )));
        }
        self.state.apply_sites(m, sites)
    }

    /// The register's weights, as [`crate::krein::KreinRegister::ledger`].
    pub fn ledger(&self) -> Result<KreinLedger> {
        let path = self.state.total_abs_sqr();
        let eta = self.metric.eta_norm(self.state.as_ref())?;
        Ok(KreinLedger {
            path,
            eta,
            positive: 0.5 * (path + eta),
            negative: 0.5 * (path - eta),
        })
    }
}

/// A real basis of `so(p, q)` in its defining representation on a site
/// with metric `diag(signs)`: `X_ab = E_ab η_b − E_ba η_a` for `a < b`.
pub fn so_algebra(signs: &[i8]) -> Result<Vec<SiteMatrix<C64>>> {
    let r = signs.len();
    if r < 2 || signs.iter().any(|s| s.abs() != 1) {
        return Err(Error::InvalidState(format!(
            "so_algebra: need at least two signs of ±1, got {signs:?}"
        )));
    }
    let mut basis = Vec::new();
    for a in 0..r {
        for b in (a + 1)..r {
            let mut data = vec![c(0.0); r * r];
            data[a * r + b] = c(f64::from(signs[b]));
            data[b * r + a] = c(-f64::from(signs[a]));
            basis.push(SiteMatrix::from_vec(r, data)?);
        }
    }
    Ok(basis)
}

/// `exp(θX)` for a site matrix, by scaling and squaring with a Taylor
/// series (the [`crate::braided::mat_exp`] scheme, for any dimension).
pub fn exponentiate_site(x: &SiteMatrix<C64>, theta: f64) -> Result<SiteMatrix<C64>> {
    let d = x.dim();
    let norm = x.data().iter().map(|z| z.norm()).fold(0.0f64, f64::max) * d as f64 * theta.abs();
    let squarings = ((norm.max(1e-300)).log2().ceil().max(0.0) as u32 + 1).min(60);
    let scale = theta / (1u64 << squarings) as f64;
    let a = SiteMatrix::from_vec(d, x.data().iter().map(|z| z * scale).collect())?;
    let mut result = SiteMatrix::<C64>::identity(d);
    let mut term = SiteMatrix::<C64>::identity(d);
    for k in 1..=24u32 {
        let t = term.matmul(&a)?;
        term = SiteMatrix::from_vec(d, t.data().iter().map(|z| z / f64::from(k)).collect())?;
        result = SiteMatrix::from_vec(
            d,
            result
                .data()
                .iter()
                .zip(term.data())
                .map(|(p, q)| p + q)
                .collect(),
        )?;
    }
    for _ in 0..squarings {
        result = result.matmul(&result)?;
    }
    Ok(result)
}
