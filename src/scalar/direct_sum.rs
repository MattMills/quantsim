//! Direct-sum (product) algebras: `T ⊕ U` with componentwise
//! operations.
//!
//! As a scalar this is the simplest way to put *several independent
//! qudit blocks inside one stored amplitude*: the complex coordinates
//! of `T ⊕ U` are the coordinates of `T` followed by those of `U`, so
//! an [`AlgebraicRegister`](crate::qudit::AlgebraicRegister) over
//! `DirectSum<Quaternion, Quaternion>` carries two *independent*
//! 1-qubit blocks per scalar — a varied qudit structure within a
//! single register.
//!
//! The algebraic price is exactly the interesting part: a direct sum
//! multiplies blockwise, so its dual-algebra sandwiches `(a·x)·b` can
//! never move weight *between* blocks — the sandwich span is the
//! block-diagonal operator algebra, and
//! [`dual_algebra_report`](crate::qudit::dual_algebra_report) measures
//! that boundary (rank `Σ blockᵢ²` of the full `(Σ blockᵢ)²`, with
//! cross-block gates showing a large synthesis residual). The
//! register still runs cross-block gates exactly — through the
//! component path — so the boundary costs routing, never correctness.
//!
//! Direct sums always contain zero divisors (`(x, 0)·(0, y) = 0`), so
//! [`Scalar::DIVISION`] is `false`; ℂ embeds diagonally
//! (`z ↦ (z, z)`), which keeps the embedding unital.

use super::{Scalar, C64};

/// The direct sum `T ⊕ U` with componentwise ring operations. See the
/// module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectSum<T: Scalar, U: Scalar> {
    /// The first block.
    pub first: T,
    /// The second block.
    pub second: U,
}

impl<T: Scalar, U: Scalar> DirectSum<T, U> {
    /// Build from the two blocks.
    pub fn new(first: T, second: U) -> Self {
        DirectSum { first, second }
    }
}

impl<T: Scalar, U: Scalar> std::ops::Add for DirectSum<T, U> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        DirectSum::new(self.first + rhs.first, self.second + rhs.second)
    }
}

impl<T: Scalar, U: Scalar> std::ops::Sub for DirectSum<T, U> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        DirectSum::new(self.first - rhs.first, self.second - rhs.second)
    }
}

impl<T: Scalar, U: Scalar> std::ops::Neg for DirectSum<T, U> {
    type Output = Self;
    fn neg(self) -> Self {
        DirectSum::new(-self.first, -self.second)
    }
}

impl<T: Scalar, U: Scalar> std::ops::Mul for DirectSum<T, U> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        DirectSum::new(self.first * rhs.first, self.second * rhs.second)
    }
}

impl<T: Scalar, U: Scalar> Scalar for DirectSum<T, U> {
    const DIM: usize = T::DIM + U::DIM;
    const COMMUTATIVE: bool = T::COMMUTATIVE && U::COMMUTATIVE;
    const ASSOCIATIVE: bool = T::ASSOCIATIVE && U::ASSOCIATIVE;
    // (x, 0) · (0, y) = 0: a direct sum always has zero divisors.
    const DIVISION: bool = false;
    const TRIVIAL_CONJ: bool = T::TRIVIAL_CONJ && U::TRIVIAL_CONJ;

    fn algebra_name() -> String {
        format!("{} (+) {}", T::algebra_name(), U::algebra_name())
    }

    fn zero() -> Self {
        DirectSum::new(T::zero(), U::zero())
    }

    fn one() -> Self {
        DirectSum::new(T::one(), U::one())
    }

    fn conj(self) -> Self {
        DirectSum::new(self.first.conj(), self.second.conj())
    }

    fn scale(self, k: f64) -> Self {
        DirectSum::new(self.first.scale(k), self.second.scale(k))
    }

    fn re(self) -> f64 {
        // Coefficient of the identity (1, 1) under the orthogonal
        // coordinate decomposition.
        (self.first.re() + self.second.re()) / 2.0
    }

    fn abs_sqr(self) -> f64 {
        self.first.abs_sqr() + self.second.abs_sqr()
    }

    fn try_from_c64(z: C64) -> Option<Self> {
        Some(DirectSum::new(T::try_from_c64(z)?, U::try_from_c64(z)?))
    }

    fn coeffs(self) -> Vec<f64> {
        let mut c = self.first.coeffs();
        c.extend(self.second.coeffs());
        c
    }

    fn from_coeffs(c: &[f64]) -> Self {
        assert_eq!(c.len(), Self::DIM, "direct-sum coordinate length");
        DirectSum::new(T::from_coeffs(&c[..T::DIM]), U::from_coeffs(&c[T::DIM..]))
    }
}
