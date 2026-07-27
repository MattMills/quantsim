//! ℝ as an amplitude algebra ("rebit" simulation).
//!
//! Real-amplitude quantum mechanics is universal for BQP given one extra
//! qubit (the real/complex encoding), and is a useful sanity domain: only the
//! real subset of the standard gate library exists over ℝ (`x`, `z`, `h`,
//! `ry`, `cx`, `cz`, `swap`, `ccx`, ...), which the registry discovers
//! automatically via [`Scalar::try_from_c64`].

use super::{Scalar, C64};

impl Scalar for f64 {
    const DIM: usize = 1;
    const COMMUTATIVE: bool = true;
    const ASSOCIATIVE: bool = true;
    const DIVISION: bool = true;
    const TRIVIAL_CONJ: bool = true;

    fn algebra_name() -> String {
        "R".to_string()
    }
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn conj(self) -> Self {
        self
    }
    fn scale(self, k: f64) -> Self {
        self * k
    }
    fn re(self) -> f64 {
        self
    }
    fn abs_sqr(self) -> f64 {
        self * self
    }
    fn try_from_c64(z: C64) -> Option<Self> {
        if z.im == 0.0 {
            Some(z.re)
        } else {
            None
        }
    }
    fn coeffs(self) -> Vec<f64> {
        vec![self]
    }
    fn from_coeffs(c: &[f64]) -> Self {
        assert_eq!(c.len(), 1, "R has dimension 1");
        c[0]
    }
}
