//! ℂ (via `num_complex::Complex<f64>`) as the default amplitude algebra.

use super::{Scalar, C64};

impl Scalar for C64 {
    const DIM: usize = 2;
    const COMMUTATIVE: bool = true;
    const ASSOCIATIVE: bool = true;
    const DIVISION: bool = true;
    const TRIVIAL_CONJ: bool = false;

    fn algebra_name() -> String {
        "C".to_string()
    }
    fn zero() -> Self {
        C64::new(0.0, 0.0)
    }
    fn one() -> Self {
        C64::new(1.0, 0.0)
    }
    fn conj(self) -> Self {
        C64::new(self.re, -self.im)
    }
    fn scale(self, k: f64) -> Self {
        C64::new(self.re * k, self.im * k)
    }
    fn re(self) -> f64 {
        self.re
    }
    fn abs_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    fn try_from_c64(z: C64) -> Option<Self> {
        Some(z)
    }
    fn coeffs(self) -> Vec<f64> {
        vec![self.re, self.im]
    }
    fn from_coeffs(c: &[f64]) -> Self {
        assert_eq!(c.len(), 2, "C has dimension 2");
        C64::new(c[0], c[1])
    }
}
