//! Swappable amplitude algebras.
//!
//! Every state representation and every gate matrix in this crate is generic
//! over a [`Scalar`]: the algebra the amplitudes live in. Standard quantum
//! mechanics uses the complex numbers ([`C64`], the crate default), but the
//! simulator is a research tool and the coefficient algebra is an explicit
//! extension axis:
//!
//! | type                | algebra                | dim | notes |
//! |---------------------|------------------------|-----|-------|
//! | `f64`               | ℝ (reals)              | 1   | "rebit" simulation; only the real gate subset exists |
//! | [`C64`]             | ℂ (complex)            | 2   | the default; full standard gate set |
//! | [`CComplex`]        | ℂ via Cayley–Dickson   | 2   | `CD<f64>`, isomorphic to `C64` (used to validate the doubling) |
//! | [`Quaternion`]      | ℍ                      | 4   | non-commutative; ℂ embeds, so the full gate set works |
//! | [`Octonion`]        | 𝕆                      | 8   | non-associative; still a division algebra |
//! | [`Sedenion`]        | 𝕊                      | 16  | zero divisors; Born weights need not be conserved |
//! | [`SplitComplex`]    | ℝ⊕ℝj, j² = +1          | 2   | non-Cayley–Dickson; indefinite Born form |
//!
//! Two magnitude notions are deliberately separate:
//!
//! * [`Scalar::abs_sqr`] — the sum of squared coordinates. Always ≥ 0. Used
//!   for numerical concerns: pruning, tolerances, unitarity deviations.
//! * [`Scalar::born_weight`] — the algebra's own quadratic form, used as the
//!   (generalized) Born rule for measurement. For ℝ, ℂ, ℍ, 𝕆 it coincides
//!   with `abs_sqr`; for split-complex it is `a² − b²` and may be negative.
//!   What measurement *means* outside the division algebras is exactly the
//!   kind of question this hook exists to explore.
//!
//! Planned scalars (see `ROADMAP.md`): truncated p-adics ℚ_p (which are not
//! ℝ-algebras — `scale` and the norm hooks will grow p-adic-aware variants)
//! and further non-Cayley–Dickson constructions.

mod cayley_dickson;
mod complex;
mod real;
mod split_complex;

pub use cayley_dickson::{CComplex, Octonion, Quaternion, Sedenion, CD};
pub use split_complex::SplitComplex;

/// The default amplitude type: `num_complex::Complex<f64>`.
pub type C64 = num_complex::Complex<f64>;

use std::fmt::Debug;
use std::ops::{Add, Mul, Neg, Sub};

/// An amplitude algebra: a finite-dimensional unital algebra over ℝ with a
/// conjugation, usable as the coefficient type of a quantum state.
///
/// Implementations must satisfy, for all `x`, `y`:
///
/// * ring axioms for `+`, `*` (associativity of `*` is **not** required —
///   octonions and sedenions are supported; set [`Scalar::ASSOCIATIVE`]
///   accordingly),
/// * `conj` is a linear involution with `conj(x * y) = conj(y) * conj(x)`,
/// * `scale(k)` is multiplication by the real number `k` (reals are central),
/// * `abs_sqr` is the squared Euclidean length of the coordinate vector,
/// * `try_from_c64` embeds ℂ (or its real subfield if `i` does not embed)
///   unitally: `1 ↦ one()`, and the embedded `i` squares to `−one()`.
///
/// Multiplication order matters for non-commutative scalars. This crate
/// treats states as **left modules**: gate application computes
/// `Σ M[r][c] * amp[c]` with the matrix entry on the left. Inner products are
/// `⟨φ|ψ⟩ = Σ conj(φ_i) * ψ_i`.
pub trait Scalar:
    Copy
    + Clone
    + Debug
    + PartialEq
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    /// Real dimension of the algebra (1 for ℝ, 2 for ℂ, 4 for ℍ, ...).
    const DIM: usize;
    /// Whether multiplication commutes.
    const COMMUTATIVE: bool;
    /// Whether multiplication associates.
    const ASSOCIATIVE: bool;
    /// Whether this is a normed division algebra (no zero divisors, and
    /// `born_weight` is positive-definite and multiplicative). When this is
    /// `false`, "unitary" gates need not conserve total Born weight — that
    /// is a feature for research, not a bug in the simulator.
    const DIVISION: bool;
    /// Whether conjugation is the identity (true only for ℝ-like scalars).
    const TRIVIAL_CONJ: bool;

    /// Human-readable algebra label, e.g. `"C"`, `"H"`, `"C_split"`.
    fn algebra_name() -> String;
    /// Additive identity.
    fn zero() -> Self;
    /// Multiplicative identity.
    fn one() -> Self;
    /// Conjugation.
    fn conj(self) -> Self;
    /// Multiplication by a real number.
    fn scale(self, k: f64) -> Self;
    /// Coefficient of the identity (the "real part").
    fn re(self) -> f64;
    /// Sum of squared coordinates; always ≥ 0.
    fn abs_sqr(self) -> f64;
    /// The algebra's quadratic form, used as the generalized Born rule.
    /// Defaults to [`Scalar::abs_sqr`]; may be indefinite for exotic algebras.
    fn born_weight(self) -> f64 {
        self.abs_sqr()
    }
    /// Embed a complex number, if this algebra contains ℂ; embed reals
    /// (`im == 0.0`) always. Returns `None` when the value cannot be
    /// represented (e.g. `i` over ℝ) — the gate registry uses this to decide
    /// which standard gates exist over a given algebra.
    fn try_from_c64(z: C64) -> Option<Self>;
    /// Embed a real number. Always succeeds (the algebra is unital over ℝ).
    fn from_re(x: f64) -> Self {
        Self::try_from_c64(C64::new(x, 0.0)).expect("reals embed in every unital R-algebra")
    }
    /// Coordinate vector of length [`Scalar::DIM`].
    fn coeffs(self) -> Vec<f64>;
    /// Build from a coordinate vector of length [`Scalar::DIM`].
    ///
    /// # Panics
    /// Panics if `c.len() != Self::DIM`.
    fn from_coeffs(c: &[f64]) -> Self;
    /// The `k`-th basis element (one-hot coordinate vector).
    ///
    /// # Panics
    /// Panics if `k >= Self::DIM`.
    fn basis(k: usize) -> Self {
        assert!(
            k < Self::DIM,
            "basis index {k} out of range for dim {}",
            Self::DIM
        );
        let mut c = vec![0.0; Self::DIM];
        c[k] = 1.0;
        Self::from_coeffs(&c)
    }
    /// Approximate equality in the Euclidean coordinate metric.
    fn approx_eq(self, rhs: Self, tol: f64) -> bool {
        (self - rhs).abs_sqr() <= tol * tol
    }
    /// Whether the Euclidean magnitude is below `tol`.
    fn is_zero(self, tol: f64) -> bool {
        self.abs_sqr() <= tol * tol
    }
}
