//! The **split** Cayley–Dickson doubling, and the split octonions it
//! reaches.
//!
//! [`SplitCD<T>`] doubles an algebra with the new unit squaring to `+1`
//! instead of `−1`:
//!
//! ```text
//! (a, b) · (c, d) = (a·c + conj(d)·b,  d·a + b·conj(c))
//! conj((a, b))    = (conj(a), −b)
//! N((a, b))       = N(a) − N(b)
//! ```
//!
//! It is [`CD`](super::CD) with one sign flipped, and the flip is the
//! whole story: the norm stops being a sum of squares and becomes a
//! *difference* of the two halves' norms. Over a positive-definite base
//! of dimension `n` the result has norm signature `(n, n)` — the split
//! signature, the most real null directions a quadratic form can have.
//!
//! * `SplitCD<f64>` is the split-complex numbers (cross-checked against
//!   [`SplitComplex`](super::SplitComplex)),
//! * `SplitCD<C64>` is the split quaternions `M₂(ℝ)` (cross-checked
//!   against [`SplitQuaternion`](super::SplitQuaternion), whose
//!   inclusion/exclusion product is this formula),
//! * `SplitCD<Quaternion>` is the **split octonions** 𝕆ₛ — see
//!   [`SplitOctonion`]. Norm signature `(4, 4)`.
//!
//! ## The split octonions
//!
//! 𝕆ₛ is the split real form of the octonions: 8-dimensional,
//! non-associative but **alternative**, with a norm that is still
//! **multiplicative** (`N(xy) = N(x)N(y)` — it is a composition algebra,
//! like ℝ, ℂ, ℍ, 𝕆 and their split forms, and unlike the sedenions) but
//! indefinite, so null vectors are zero divisors. Its norm form *is*
//! `ℝ^{4,4}`, the arena of `Spin(4,4)` triality, and the 2×2 Hermitian
//! matrices over it ([`Hermitian2<SplitOctonion>`](crate::hermitian2))
//! form `ℝ^{5,5}`, on which `SL(2, 𝕆ₛ) ≅ Spin(5,5)` acts — the conformal
//! group of a `(4,4)` world. Measured in `tests/split_octonion.rs`.
//!
//! **ℂ embeds** through the base quaternions, so the full standard gate
//! library projects in, as for [`SplitQuaternion`](super::SplitQuaternion).
//! A standard gate's complex entry `m = (m, 0)` acts as
//! `m·(a, b) = (m·a, b·m)`: the first half is driven on the left, the
//! second on the right, and both by a norm-preserving map — so under a
//! standard circuit both halves keep their weight separately, and the net
//! (Born) weight `N(a) − N(b)` is conserved along with the path weight.
//! The element `cosh θ + sinh θ·ℓ` (with `ℓ = (0, 1)`, `ℓ² = +1`) has
//! norm 1 and pumps path weight while conserving net weight: the
//! non-compact direction every split algebra has and no division algebra
//! does.
//!
//! The Born weight is the split norm, so it is indefinite; what
//! measurement *means* here is the open question the split algebras
//! exist in this crate to pose, exactly as for
//! [`SplitComplex`](super::SplitComplex).

use super::{Quaternion, Scalar, C64};

/// One split Cayley–Dickson doubling of `T`. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitCD<T: Scalar> {
    /// First half (the base algebra).
    pub a: T,
    /// Second half: coefficients of the new unit `ℓ`, with `ℓ² = +1`.
    pub b: T,
}

/// The split octonions 𝕆ₛ: the quaternions doubled with `ℓ² = +1`.
pub type SplitOctonion = SplitCD<Quaternion>;

impl<T: Scalar> SplitCD<T> {
    /// Build from the two halves.
    pub fn new(a: T, b: T) -> Self {
        SplitCD { a, b }
    }

    /// The doubling unit `ℓ = (0, 1)`, which squares to `+1`.
    pub fn ell() -> Self {
        SplitCD {
            a: T::zero(),
            b: T::one(),
        }
    }

    /// `cosh θ + sinh θ·ℓ`: a norm-1 element in the non-compact
    /// direction. Multiplying by it conserves the net (Born) weight and
    /// changes the path weight `abs_sqr` by a factor up to `e^{2|θ|}`.
    pub fn boost(theta: f64) -> Self {
        SplitCD {
            a: T::from_re(theta.cosh()),
            b: T::from_re(theta.sinh()),
        }
    }
}

impl<T: Scalar> std::ops::Add for SplitCD<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        SplitCD {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}

impl<T: Scalar> std::ops::Sub for SplitCD<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        SplitCD {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}

impl<T: Scalar> std::ops::Neg for SplitCD<T> {
    type Output = Self;
    fn neg(self) -> Self {
        SplitCD {
            a: -self.a,
            b: -self.b,
        }
    }
}

impl<T: Scalar> std::ops::Mul for SplitCD<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (self.a, self.b);
        let (c, d) = (rhs.a, rhs.b);
        SplitCD {
            a: a * c + d.conj() * b,
            b: d * a + b * c.conj(),
        }
    }
}

impl<T: Scalar> Scalar for SplitCD<T> {
    const DIM: usize = 2 * T::DIM;
    // Same flag algebra as the ordinary doubling (the sign does not change
    // which identities survive); never a division algebra, since `(1, 1)`
    // is null.
    const COMMUTATIVE: bool = T::COMMUTATIVE && T::TRIVIAL_CONJ;
    const ASSOCIATIVE: bool = T::COMMUTATIVE && T::ASSOCIATIVE;
    const DIVISION: bool = false;
    const TRIVIAL_CONJ: bool = false;

    fn algebra_name() -> String {
        match T::algebra_name().as_str() {
            "R" => "C_split (as SplitCD<R>)".to_string(),
            "C" => "H_split (as SplitCD<C>)".to_string(),
            "H (quaternions)" => "O_split (split octonions)".to_string(),
            other => format!("SplitCD<{other}>"),
        }
    }
    fn zero() -> Self {
        SplitCD {
            a: T::zero(),
            b: T::zero(),
        }
    }
    fn one() -> Self {
        SplitCD {
            a: T::one(),
            b: T::zero(),
        }
    }
    fn conj(self) -> Self {
        SplitCD {
            a: self.a.conj(),
            b: -self.b,
        }
    }
    fn scale(self, k: f64) -> Self {
        SplitCD {
            a: self.a.scale(k),
            b: self.b.scale(k),
        }
    }
    fn re(self) -> f64 {
        self.a.re()
    }
    fn abs_sqr(self) -> f64 {
        self.a.abs_sqr() + self.b.abs_sqr()
    }
    fn born_weight(self) -> f64 {
        // N((a, b)) = (a, b)·conj((a, b)) = N(a) − N(b).
        self.a.born_weight() - self.b.born_weight()
    }
    fn try_from_c64(z: C64) -> Option<Self> {
        // Only the base can supply `i`: the new unit squares to +1.
        T::try_from_c64(z).map(|a| SplitCD { a, b: T::zero() })
    }
    fn coeffs(self) -> Vec<f64> {
        let mut c = self.a.coeffs();
        c.extend(self.b.coeffs());
        c
    }
    fn from_coeffs(c: &[f64]) -> Self {
        assert_eq!(
            c.len(),
            Self::DIM,
            "SplitCD algebra has dimension {}",
            Self::DIM
        );
        SplitCD {
            a: T::from_coeffs(&c[..T::DIM]),
            b: T::from_coeffs(&c[T::DIM..]),
        }
    }
}

#[cfg(test)]
#[allow(clippy::assertions_on_constants)] // algebra-flag pins are intentional
mod tests {
    use super::*;
    use crate::scalar::{SplitComplex, SplitQuaternion};

    const TOL: f64 = 1e-12;

    fn rand<S: Scalar>(prng: &mut crate::rng::Prng) -> S {
        let c: Vec<f64> = (0..S::DIM).map(|_| prng.next_f64() * 2.0 - 1.0).collect();
        S::from_coeffs(&c)
    }

    #[test]
    fn split_cd_of_reals_is_the_split_complex_numbers() {
        let mut prng = crate::rng::Prng::new(5);
        for _ in 0..100 {
            let (x, y): (SplitCD<f64>, SplitCD<f64>) = (rand(&mut prng), rand(&mut prng));
            let (sx, sy) = (
                SplitComplex::from_coeffs(&x.coeffs()),
                SplitComplex::from_coeffs(&y.coeffs()),
            );
            let (p, sp) = (x * y, sx * sy);
            for (u, v) in p.coeffs().iter().zip(sp.coeffs()) {
                assert!((u - v).abs() < TOL);
            }
            assert!((x.born_weight() - sx.born_weight()).abs() < TOL);
        }
        // No i: the split unit squares to +1, so only reals embed.
        assert!(SplitCD::<f64>::try_from_c64(C64::new(0.0, 1.0)).is_none());
        assert!(SplitCD::<f64>::COMMUTATIVE && SplitCD::<f64>::ASSOCIATIVE);
    }

    #[test]
    fn split_cd_of_complexes_is_the_split_quaternions() {
        let mut prng = crate::rng::Prng::new(6);
        for _ in 0..100 {
            let (x, y): (SplitCD<C64>, SplitCD<C64>) = (rand(&mut prng), rand(&mut prng));
            let (qx, qy) = (
                SplitQuaternion::from_coeffs(&x.coeffs()),
                SplitQuaternion::from_coeffs(&y.coeffs()),
            );
            for (u, v) in (x * y).coeffs().iter().zip((qx * qy).coeffs()) {
                assert!((u - v).abs() < TOL);
            }
            assert!((x.born_weight() - qx.born_weight()).abs() < TOL);
            for (u, v) in x.conj().coeffs().iter().zip(qx.conj().coeffs()) {
                assert!((u - v).abs() < TOL);
            }
        }
        assert!(!SplitCD::<C64>::COMMUTATIVE && SplitCD::<C64>::ASSOCIATIVE);
    }

    #[test]
    fn ell_squares_to_plus_one_and_the_embedded_i_to_minus_one() {
        let ell = SplitOctonion::ell();
        assert!((ell * ell).approx_eq(SplitOctonion::one(), TOL));
        assert_eq!(ell.born_weight(), -1.0);
        let i = SplitOctonion::try_from_c64(C64::new(0.0, 1.0)).unwrap();
        assert!((i * i).approx_eq(-SplitOctonion::one(), TOL));
        assert_eq!(
            SplitOctonion::algebra_name(),
            "O_split (split octonions)".to_string()
        );
    }
}
