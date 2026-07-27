//! Split-complex numbers: a non-Cayley–Dickson exploration algebra.
//!
//! Elements are `a + b·j` with `j² = +1` (instead of ℂ's `−1`). The algebra
//! is commutative and associative but **not** a division algebra: the null
//! cone `a = ±b` consists of zero divisors, and the natural quadratic form
//! `a² − b²` (exposed via [`Scalar::born_weight`]) is indefinite — "Born
//! weights" of split-complex states can be negative or vanish for nonzero
//! amplitudes. The simulator machinery runs anyway; interpreting the results
//! is the research question.

use super::{Scalar, C64};

/// A split-complex number `a + b·j`, `j² = +1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitComplex {
    /// Real part.
    pub a: f64,
    /// Coefficient of `j`.
    pub b: f64,
}

impl SplitComplex {
    /// Build `a + b·j`.
    pub fn new(a: f64, b: f64) -> Self {
        SplitComplex { a, b }
    }
}

impl std::ops::Add for SplitComplex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        SplitComplex::new(self.a + rhs.a, self.b + rhs.b)
    }
}

impl std::ops::Sub for SplitComplex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        SplitComplex::new(self.a - rhs.a, self.b - rhs.b)
    }
}

impl std::ops::Neg for SplitComplex {
    type Output = Self;
    fn neg(self) -> Self {
        SplitComplex::new(-self.a, -self.b)
    }
}

impl std::ops::Mul for SplitComplex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // (a + bj)(c + dj) = (ac + bd) + (ad + bc)j
        SplitComplex::new(
            self.a * rhs.a + self.b * rhs.b,
            self.a * rhs.b + self.b * rhs.a,
        )
    }
}

impl Scalar for SplitComplex {
    const DIM: usize = 2;
    const COMMUTATIVE: bool = true;
    const ASSOCIATIVE: bool = true;
    const DIVISION: bool = false;
    const TRIVIAL_CONJ: bool = false;

    fn algebra_name() -> String {
        "C_split".to_string()
    }
    fn zero() -> Self {
        SplitComplex::new(0.0, 0.0)
    }
    fn one() -> Self {
        SplitComplex::new(1.0, 0.0)
    }
    fn conj(self) -> Self {
        SplitComplex::new(self.a, -self.b)
    }
    fn scale(self, k: f64) -> Self {
        SplitComplex::new(self.a * k, self.b * k)
    }
    fn re(self) -> f64 {
        self.a
    }
    fn abs_sqr(self) -> f64 {
        self.a * self.a + self.b * self.b
    }
    fn born_weight(self) -> f64 {
        // The split-complex modulus: z * conj(z) = a^2 - b^2. Indefinite.
        self.a * self.a - self.b * self.b
    }
    fn try_from_c64(z: C64) -> Option<Self> {
        // x^2 = -1 has no split-complex solution, so only ℝ embeds.
        if z.im == 0.0 {
            Some(SplitComplex::new(z.re, 0.0))
        } else {
            None
        }
    }
    fn coeffs(self) -> Vec<f64> {
        vec![self.a, self.b]
    }
    fn from_coeffs(c: &[f64]) -> Self {
        assert_eq!(c.len(), 2, "C_split has dimension 2");
        SplitComplex::new(c[0], c[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn j_squares_to_plus_one() {
        let j = SplitComplex::basis(1);
        assert_eq!(j * j, SplitComplex::one());
    }

    #[test]
    fn null_cone_zero_divisors() {
        let x = SplitComplex::new(1.0, 1.0);
        let y = SplitComplex::new(1.0, -1.0);
        assert_eq!(x * y, SplitComplex::zero());
        assert_eq!(x.born_weight(), 0.0);
        assert!(x.abs_sqr() > 0.0);
    }

    #[test]
    fn born_weight_indefinite() {
        assert!(SplitComplex::new(0.0, 1.0).born_weight() < 0.0);
        assert!(SplitComplex::new(1.0, 0.0).born_weight() > 0.0);
    }
}
