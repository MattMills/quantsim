//! A qubit as a 2×2 Hermitian matrix over any amplitude algebra — and the
//! spacetime that algebra makes out of it.
//!
//! Every other register in this crate takes its geometry from outside:
//! a width, a lattice, a curve through a grid. This one takes it from the
//! number system. Over an algebra `K` with conjugation, the Hermitian
//! matrices
//!
//! ```text
//! X = ⎡ a    z ⎤        a, b ∈ ℝ,  z ∈ K
//!     ⎣ z̄    b ⎦
//! ```
//!
//! form a real space of dimension `K::DIM + 2`, and the determinant
//!
//! ```text
//! det X = a·b − N(z),       N(z) = z·z̄ = born_weight(z)
//! ```
//!
//! is a quadratic form on it. Writing `a = t + x`, `b = t − x` makes it
//! `t² − x² − N(z)`: a Minkowski-type form whose signature is fixed by `K`
//! alone. Over ℂ this is the textbook identification of a qubit's density
//! matrices with the light cone of ℝ^{3,1}: the trace is time, the Bloch
//! vector is space, and pure states are exactly the null vectors. Over the
//! other algebras the same construction gives a different spacetime, and
//! [`signature`] measures which one rather than quoting it. Measured in
//! `tests/hermitian2.rs`, with `(positive, negative)` counting time-like
//! and space-like directions of `det`:
//!
//! | `K`               | `DIM` | signature of `det` | pure spinors null? |
//! |-------------------|-------|--------------------|--------------------|
//! | ℝ (`f64`)         | 1     | (1, 2)             | yes |
//! | ℂ ([`C64`](crate::scalar::C64)) | 2 | (1, 3)             | yes |
//! | split-complex     | 2     | (2, 2)             | yes |
//! | ℍ                 | 4     | (1, 5)             | yes |
//! | split-quaternions | 4     | (3, 3)             | yes |
//! | 𝕆                 | 8     | (1, 9)             | yes |
//! | 𝕊 (sedenions)     | 16    | (1, 17)            | **no** |
//!
//! The split-complex row is the compound "2-dimensional, irreducibly
//! 2-dimensional" space with split signature `(2, 2)`. The octonion row is
//! ℝ^{9,1}; its pure states `ψψ†` form `𝕆P¹`, an 8-sphere.
//!
//! ## What acts on it
//!
//! A 2×2 matrix `M` over `K` acts by `X ↦ M X M†`. For non-associative `K`
//! that product needs a bracketing, so the always-valid generators are
//! given entry by entry, from the algebra's own products:
//!
//! * [`Hermitian2::shear_upper`] / [`Hermitian2::shear_lower`] — null
//!   rotations by `[[1, w], [0, 1]]` and `[[1, 0], [w, 1]]`, any `w ∈ K`;
//! * [`Hermitian2::boost`] — `diag(λ, 1/λ)`, real `λ ≠ 0`.
//!
//! Each preserves `det` in **every** algebra here, sedenions included: the
//! proof uses only that `N` is a quadratic form whose polarization is
//! `Re(z·w̄)`, and the tests check it numerically. Over ℂ these generate
//! `SL(2, ℂ)`, the Lorentz group's double cover. The general
//! [`Hermitian2::conjugate`] is offered only for associative `K` and
//! refuses the rest.
//!
//! ## Where the boundaries are
//!
//! * **Pure states are null only for composition algebras.**
//!   `det(ψψ†) = N(ψ₀)·N(ψ₁) − N(ψ₀·ψ̄₁)`, which vanishes exactly when the
//!   norm is multiplicative. That holds through the octonions and for both
//!   split algebras; for the sedenions it fails, and
//!   [`Hermitian2::from_spinor`] produces measurably non-null "pure" states.
//! * **Probabilities need a definite algebra.** [`Hermitian2::trace_pairing`]
//!   is `Re tr(XY)`, the Jordan trace form. Over ℂ it is `Tr(ρσ)`, and on
//!   pure states `|⟨ψ|φ⟩|²`. Over a split algebra it is indefinite, so a
//!   state can have negative "probability" against itself; it is reported,
//!   not clamped.
//! * **One system per register.** There is no tensor product here. For
//!   non-associative `K` none exists — the octonionic projective line and
//!   plane are the whole of octonionic projective geometry, and the
//!   exceptional Jordan algebra of 3×3 octonion matrices is the largest
//!   octonionic observable algebra. A composite of `Hermitian2` registers
//!   is not defined, and nothing in this module pretends otherwise.
//! * **This is a state space, not a backend.** It does not run circuits; it
//!   is the object a backend for these algebras would have to preserve.

use crate::error::{Error, Result};
use crate::scalar::Scalar;

/// A 2×2 Hermitian matrix `[[a, z], [z̄, b]]` over the algebra `K`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hermitian2<K: Scalar> {
    a: f64,
    b: f64,
    z: K,
}

/// Causal type of a [`Hermitian2`] under its determinant form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalClass {
    /// `det > tol`: inside the cone (over ℂ, a mixed state up to scale).
    Timelike,
    /// `|det| ≤ tol`: on the cone (over ℂ, a pure state up to scale).
    Null,
    /// `det < −tol`: outside the cone.
    Spacelike,
}

/// Inertia of the determinant form over an algebra: how many directions
/// are time-like, space-like and degenerate. See [`signature`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormSignature {
    /// Directions on which `det` is positive (time-like).
    pub positive: usize,
    /// Directions on which `det` is negative (space-like).
    pub negative: usize,
    /// Degenerate directions.
    pub null: usize,
}

impl FormSignature {
    /// Total real dimension, `K::DIM + 2`.
    pub fn dim(&self) -> usize {
        self.positive + self.negative + self.null
    }
}

/// The polarization of the algebra's quadratic form,
/// `B(u, v) = (N(u + v) − N(u) − N(v)) / 2`.
fn polar<K: Scalar>(u: K, v: K) -> f64 {
    0.5 * ((u + v).born_weight() - u.born_weight() - v.born_weight())
}

/// Whether `x` is a real multiple of the identity, to within `tol` in
/// every non-identity coordinate.
fn is_real<K: Scalar>(x: K, tol: f64) -> bool {
    x.coeffs().iter().skip(1).all(|c| c.abs() <= tol)
}

impl<K: Scalar> Hermitian2<K> {
    /// `[[a, z], [z̄, b]]`.
    pub fn new(a: f64, b: f64, z: K) -> Self {
        Hermitian2 { a, b, z }
    }

    /// From spacetime coordinates: `a = t + x`, `b = t − x`.
    pub fn from_spacetime(t: f64, x: f64, z: K) -> Self {
        Hermitian2::new(t + x, t - x, z)
    }

    /// The projector-like matrix `ψψ†` of a two-component spinor
    /// `ψ = (ψ₀, ψ₁)`: `a = N(ψ₀)`, `b = N(ψ₁)`, `z = ψ₀·ψ̄₁`.
    ///
    /// Over ℂ with `|ψ₀|² + |ψ₁|² = 1` this is the density matrix of the
    /// pure state `ψ₀|0⟩ + ψ₁|1⟩`. It is null exactly when `K`'s norm is
    /// multiplicative on `ψ₀, ψ̄₁` (see the module docs).
    pub fn from_spinor(psi0: K, psi1: K) -> Self {
        Hermitian2::new(psi0.born_weight(), psi1.born_weight(), psi0 * psi1.conj())
    }

    /// From the coordinate vector `[t, x, z₀, …, z_{DIM−1}]`.
    ///
    /// # Panics
    /// Panics if `c.len() != K::DIM + 2`.
    pub fn from_coords(c: &[f64]) -> Self {
        assert_eq!(
            c.len(),
            K::DIM + 2,
            "Hermitian2 over {} has {} coordinates",
            K::algebra_name(),
            K::DIM + 2
        );
        Hermitian2::from_spacetime(c[0], c[1], K::from_coeffs(&c[2..]))
    }

    /// Upper diagonal entry `a`.
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Lower diagonal entry `b`.
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Off-diagonal entry `z` (the lower one is `z̄`).
    pub fn z(&self) -> K {
        self.z
    }

    /// Time coordinate `t = (a + b) / 2`, half the trace.
    pub fn t(&self) -> f64 {
        0.5 * (self.a + self.b)
    }

    /// First space coordinate `x = (a − b) / 2`.
    pub fn x(&self) -> f64 {
        0.5 * (self.a - self.b)
    }

    /// `tr X = a + b`.
    pub fn trace(&self) -> f64 {
        self.a + self.b
    }

    /// `det X = a·b − N(z)`.
    pub fn det(&self) -> f64 {
        self.a * self.b - self.z.born_weight()
    }

    /// Coordinate vector `[t, x, z₀, …]`, length `K::DIM + 2`.
    pub fn coords(&self) -> Vec<f64> {
        let mut c = Vec::with_capacity(K::DIM + 2);
        c.push(self.t());
        c.push(self.x());
        c.extend(self.z.coeffs());
        c
    }

    /// The space part over the time part, `[x, z₀, …] / t`.
    ///
    /// Over ℂ, for a density matrix, this is the Bloch vector in the order
    /// `(⟨Z⟩, ⟨X⟩, −⟨Y⟩)`: `z = ρ₀₁ = (⟨X⟩ − i⟨Y⟩) / 2`. Its length is 1
    /// exactly on the null cone.
    pub fn bloch_vector(&self) -> Vec<f64> {
        let t = self.t();
        self.coords()[1..].iter().map(|c| c / t).collect()
    }

    /// Causal type under `det`, with `|det| ≤ tol` counted as null.
    pub fn causal_class(&self, tol: f64) -> CausalClass {
        let d = self.det();
        if d > tol {
            CausalClass::Timelike
        } else if d < -tol {
            CausalClass::Spacelike
        } else {
            CausalClass::Null
        }
    }

    /// The polar form of `det`: `½(a·b' + a'·b) − B(z, z')`, so that
    /// `det_pairing(X, X) = det X`.
    pub fn det_pairing(&self, other: &Self) -> f64 {
        0.5 * (self.a * other.b + other.a * self.b) - polar(self.z, other.z)
    }

    /// `Re tr(X·Y)`, the Jordan trace form, computed from the algebra's own
    /// products: `a·a' + b·b' + Re(z·z̄' + z̄·z')`.
    ///
    /// Over ℂ this is `Tr(ρσ)`; between pure states it is `|⟨ψ|φ⟩|²`. It is
    /// positive on every nonzero `X` only when `N` is positive-definite —
    /// over the split algebras it can be negative, and is returned as is.
    pub fn trace_pairing(&self, other: &Self) -> f64 {
        self.a * other.a
            + self.b * other.b
            + (self.z * other.z.conj() + self.z.conj() * other.z).re()
    }

    /// `M X M†` for `M = [[1, w], [0, 1]]`, written out entry by entry so
    /// no bracketing of a triple product is needed:
    /// `a' = a + Re(w·z̄ + z·w̄) + b·Re(w·w̄)`, `z' = z + w·b`, `b' = b`.
    /// Preserves `det` in every algebra.
    pub fn shear_upper(&self, w: K) -> Self {
        let cross = (w * self.z.conj() + self.z * w.conj()).re();
        Hermitian2::new(
            self.a + cross + self.b * (w * w.conj()).re(),
            self.b,
            self.z + w.scale(self.b),
        )
    }

    /// `M X M†` for `M = [[1, 0], [w, 1]]`:
    /// `a' = a`, `z' = z + a·w̄`, `b' = b + Re(w·z + z̄·w̄) + a·Re(w·w̄)`.
    /// Preserves `det` in every algebra.
    pub fn shear_lower(&self, w: K) -> Self {
        let cross = (w * self.z + self.z.conj() * w.conj()).re();
        Hermitian2::new(
            self.a,
            self.b + cross + self.a * (w * w.conj()).re(),
            self.z + w.conj().scale(self.a),
        )
    }

    /// `M X M†` for `M = diag(λ, 1/λ)`, real `λ ≠ 0`: a boost along `x`
    /// with rapidity `2·ln|λ|`. Preserves `det`.
    ///
    /// # Panics
    /// Panics if `lambda == 0.0`.
    pub fn boost(&self, lambda: f64) -> Self {
        assert!(lambda != 0.0, "boost: λ must be nonzero");
        Hermitian2::new(self.a * lambda * lambda, self.b / (lambda * lambda), self.z)
    }

    /// The general action `M X M†` for an arbitrary 2×2 matrix `M` over an
    /// associative `K` (rows `m[0]`, `m[1]`), computed as `(M X) M†`.
    ///
    /// Over ℂ, `det(M X M†) = |det M|²·det X`. Refuses non-associative
    /// algebras with [`Error::UnsupportedForAlgebra`] — there `(M X) M†`
    /// and `M (X M†)` differ and neither is the action — and returns
    /// [`Error::InvalidState`] if the result is not Hermitian to within
    /// `tol` (possible for algebras whose self-conjugate elements are not
    /// all real).
    pub fn conjugate(&self, m: [[K; 2]; 2], tol: f64) -> Result<Self> {
        if !K::ASSOCIATIVE {
            return Err(Error::UnsupportedForAlgebra {
                gate: "Hermitian2::conjugate".into(),
                algebra: K::algebra_name(),
            });
        }
        let x = [
            [K::from_re(self.a), self.z],
            [self.z.conj(), K::from_re(self.b)],
        ];
        let mut p = [[K::zero(); 2]; 2];
        for (i, row) in p.iter_mut().enumerate() {
            for (j, e) in row.iter_mut().enumerate() {
                *e = m[i][0] * x[0][j] + m[i][1] * x[1][j];
            }
        }
        let mut q = [[K::zero(); 2]; 2];
        for (i, row) in q.iter_mut().enumerate() {
            for (j, e) in row.iter_mut().enumerate() {
                *e = p[i][0] * m[j][0].conj() + p[i][1] * m[j][1].conj();
            }
        }
        if !is_real(q[0][0], tol)
            || !is_real(q[1][1], tol)
            || !q[1][0].approx_eq(q[0][1].conj(), tol)
        {
            return Err(Error::InvalidState(format!(
                "Hermitian2::conjugate over {}: M X M† is not Hermitian to within {tol}",
                K::algebra_name()
            )));
        }
        Ok(Hermitian2::new(q[0][0].re(), q[1][1].re(), q[0][1]))
    }
}

/// Eigenvalues of a real symmetric `n×n` matrix (row-major) by cyclic
/// Jacobi rotations. `n` here is at most `DIM + 2`, so the cost is trivial.
fn symmetric_eigenvalues(mut m: Vec<f64>, n: usize) -> Vec<f64> {
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|p| (0..n).filter(move |&q| q != p).map(move |q| (p, q)))
            .map(|(p, q)| m[p * n + q] * m[p * n + q])
            .sum();
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (kp, kq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * kp - s * kq;
                    m[k * n + q] = s * kp + c * kq;
                }
                for k in 0..n {
                    let (pk, qk) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * pk - s * qk;
                    m[q * n + k] = s * pk + c * qk;
                }
            }
        }
    }
    (0..n).map(|i| m[i * n + i]).collect()
}

/// The signature of `det` on the Hermitian 2×2 matrices over `K`,
/// measured: the Gram matrix of [`Hermitian2::det_pairing`] on the
/// coordinate basis `[t, x, z₀, …]`, diagonalized, and its eigenvalues
/// counted by sign (Sylvester's law of inertia). Nothing is assumed about
/// the algebra's basis being orthogonal.
pub fn signature<K: Scalar>() -> FormSignature {
    let n = K::DIM + 2;
    let basis: Vec<Hermitian2<K>> = (0..n)
        .map(|i| {
            let mut c = vec![0.0; n];
            c[i] = 1.0;
            Hermitian2::from_coords(&c)
        })
        .collect();
    let mut gram = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            gram[i * n + j] = basis[i].det_pairing(&basis[j]);
        }
    }
    let eig = symmetric_eigenvalues(gram, n);
    let scale = eig.iter().fold(0.0f64, |m, e| m.max(e.abs())).max(1.0);
    let tol = 1e-9 * scale;
    FormSignature {
        positive: eig.iter().filter(|&&e| e > tol).count(),
        negative: eig.iter().filter(|&&e| e < -tol).count(),
        null: eig.iter().filter(|&&e| e.abs() <= tol).count(),
    }
}
