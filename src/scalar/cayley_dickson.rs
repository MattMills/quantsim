//! The Cayley–Dickson doubling construction.
//!
//! [`CD<T>`] doubles an algebra: elements are pairs `(a, b)` with
//!
//! ```text
//! (a, b) · (c, d) = (a·c − conj(d)·b,  d·a + b·conj(c))
//! conj((a, b))    = (conj(a), −b)
//! ```
//!
//! Iterating from ℝ climbs the classical tower — each level trading away a
//! property:
//!
//! * `CD<f64>` ≅ ℂ (loses trivial conjugation) — see [`CComplex`],
//! * `CD<C64>` = ℍ quaternions (loses commutativity) — see [`Quaternion`],
//! * `CD<Quaternion>` = 𝕆 octonions (loses associativity) — see [`Octonion`],
//! * `CD<Octonion>` = 𝕊 sedenions (loses the division property: zero
//!   divisors appear, and Born weight is no longer multiplicative) — see
//!   [`Sedenion`].
//!
//! Nothing stops you from doubling further (`CD<Sedenion>`, ...) or doubling
//! a non-standard base (`CD<SplitComplex>` gives the split-quaternions).
//!
//! Basis ordering: `coeffs()` concatenates the coordinates of `a` then `b`,
//! so for [`Quaternion`] the basis is `[1, i, j, k]` with `i = (i, 0)`,
//! `j = (0, 1)`, `k = (0, i)`, and standard relations `i·j = k`, `j·i = −k`.

use super::{Scalar, C64};

/// One Cayley–Dickson doubling of the algebra `T`. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CD<T: Scalar> {
    /// First component ("lower half" of the doubled algebra).
    pub a: T,
    /// Second component (coefficients of the new imaginary units).
    pub b: T,
}

/// ℂ rebuilt as `CD<f64>`; isomorphic to [`C64`], kept as a cross-check of
/// the doubling construction itself.
pub type CComplex = CD<f64>;
/// The quaternions ℍ.
pub type Quaternion = CD<C64>;
/// The octonions 𝕆.
pub type Octonion = CD<Quaternion>;
/// The sedenions 𝕊 (first Cayley–Dickson algebra with zero divisors).
pub type Sedenion = CD<Octonion>;

impl<T: Scalar> CD<T> {
    /// Build from the two halves.
    pub fn new(a: T, b: T) -> Self {
        CD { a, b }
    }
}

impl<T: Scalar> std::ops::Add for CD<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        CD { a: self.a + rhs.a, b: self.b + rhs.b }
    }
}

impl<T: Scalar> std::ops::Sub for CD<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        CD { a: self.a - rhs.a, b: self.b - rhs.b }
    }
}

impl<T: Scalar> std::ops::Neg for CD<T> {
    type Output = Self;
    fn neg(self) -> Self {
        CD { a: -self.a, b: -self.b }
    }
}

impl<T: Scalar> std::ops::Mul for CD<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (self.a, self.b);
        let (c, d) = (rhs.a, rhs.b);
        CD { a: a * c - d.conj() * b, b: d * a + b * c.conj() }
    }
}

impl<T: Scalar> Scalar for CD<T> {
    const DIM: usize = 2 * T::DIM;
    // Doubling is commutative iff the base is commutative with trivial
    // conjugation; associative iff the base is commutative and associative;
    // a division algebra iff the base is an associative division algebra.
    const COMMUTATIVE: bool = T::COMMUTATIVE && T::TRIVIAL_CONJ;
    const ASSOCIATIVE: bool = T::COMMUTATIVE && T::ASSOCIATIVE;
    const DIVISION: bool = T::ASSOCIATIVE && T::DIVISION;
    const TRIVIAL_CONJ: bool = false;

    fn algebra_name() -> String {
        match T::algebra_name().as_str() {
            "R" => "C (as CD<R>)".to_string(),
            "C" | "C (as CD<R>)" => "H (quaternions)".to_string(),
            "H (quaternions)" => "O (octonions)".to_string(),
            "O (octonions)" => "S (sedenions)".to_string(),
            other => format!("CD<{other}>"),
        }
    }
    fn zero() -> Self {
        CD { a: T::zero(), b: T::zero() }
    }
    fn one() -> Self {
        CD { a: T::one(), b: T::zero() }
    }
    fn conj(self) -> Self {
        CD { a: self.a.conj(), b: -self.b }
    }
    fn scale(self, k: f64) -> Self {
        CD { a: self.a.scale(k), b: self.b.scale(k) }
    }
    fn re(self) -> f64 {
        self.a.re()
    }
    fn abs_sqr(self) -> f64 {
        self.a.abs_sqr() + self.b.abs_sqr()
    }
    fn try_from_c64(z: C64) -> Option<Self> {
        if let Some(a) = T::try_from_c64(z) {
            // Nested embedding: the tower C ⊂ H ⊂ O ⊂ ... shares one `i`.
            return Some(CD { a, b: T::zero() });
        }
        // The base holds only reals; the doubling unit (0, 1) squares to −1
        // and serves as `i`, making CD<T> itself a complex embedding.
        Some(CD { a: T::from_re(z.re), b: T::from_re(z.im) })
    }
    fn coeffs(self) -> Vec<f64> {
        let mut c = self.a.coeffs();
        c.extend(self.b.coeffs());
        c
    }
    fn from_coeffs(c: &[f64]) -> Self {
        assert_eq!(c.len(), Self::DIM, "CD algebra has dimension {}", Self::DIM);
        CD { a: T::from_coeffs(&c[..T::DIM]), b: T::from_coeffs(&c[T::DIM..]) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn quaternion_multiplication_table() {
        let e = |k| Quaternion::basis(k); // [1, i, j, k]
        let (one, i, j, k) = (e(0), e(1), e(2), e(3));
        assert!((i * i).approx_eq(-one, TOL));
        assert!((j * j).approx_eq(-one, TOL));
        assert!((k * k).approx_eq(-one, TOL));
        assert!((i * j).approx_eq(k, TOL));
        assert!((j * i).approx_eq(-k, TOL));
        assert!((j * k).approx_eq(i, TOL));
        assert!((k * j).approx_eq(-i, TOL));
        assert!((k * i).approx_eq(j, TOL));
        assert!((i * k).approx_eq(-j, TOL));
        assert!((i * j * k).approx_eq(-one, TOL));
    }

    #[test]
    fn octonion_units_square_to_minus_one_and_anticommute() {
        let one = Octonion::one();
        for p in 1..8 {
            let ep = Octonion::basis(p);
            assert!((ep * ep).approx_eq(-one, TOL), "e{p}^2 != -1");
            for q in (p + 1)..8 {
                let eq = Octonion::basis(q);
                assert!((ep * eq).approx_eq(-(eq * ep), TOL), "e{p}, e{q} should anticommute");
            }
        }
    }

    #[test]
    fn octonion_non_associativity_witness() {
        // Find some triple of imaginary units with (x·y)·z != x·(y·z);
        // existence is convention-independent.
        let mut found = false;
        'outer: for p in 1..8 {
            for q in 1..8 {
                for r in 1..8 {
                    let (x, y, z) =
                        (Octonion::basis(p), Octonion::basis(q), Octonion::basis(r));
                    if !((x * y) * z).approx_eq(x * (y * z), TOL) {
                        found = true;
                        break 'outer;
                    }
                }
            }
        }
        assert!(found, "octonions must not be associative");
        assert!(!Octonion::ASSOCIATIVE);
        assert!(Octonion::DIVISION);
    }

    #[test]
    fn quaternions_are_associative() {
        for p in 0..4 {
            for q in 0..4 {
                for r in 0..4 {
                    let (x, y, z) =
                        (Quaternion::basis(p), Quaternion::basis(q), Quaternion::basis(r));
                    assert!(((x * y) * z).approx_eq(x * (y * z), TOL));
                }
            }
        }
        assert!(Quaternion::ASSOCIATIVE);
        assert!(!Quaternion::COMMUTATIVE);
    }

    #[test]
    fn sedenion_zero_divisors_exist() {
        // The sedenions are not a division algebra: some (e_p + e_q)(e_r − e_s)
        // vanishes with both factors nonzero. The exact pairs depend on basis
        // conventions, so search rather than hard-code.
        let mut found = None;
        'outer: for p in 1..16 {
            for q in (p + 1)..16 {
                for r in 1..16 {
                    for s in (r + 1)..16 {
                        let x = Sedenion::basis(p) + Sedenion::basis(q);
                        let y = Sedenion::basis(r) - Sedenion::basis(s);
                        if (x * y).is_zero(TOL) {
                            found = Some((p, q, r, s));
                            break 'outer;
                        }
                    }
                }
            }
        }
        let (p, q, r, s) = found.expect("sedenions must contain zero divisors");
        assert!(!Sedenion::DIVISION);
        // Sanity: the same search over octonions must find nothing.
        for p in 1..8 {
            for q in (p + 1)..8 {
                for r in 1..8 {
                    for s in (r + 1)..8 {
                        let x = Octonion::basis(p) + Octonion::basis(q);
                        let y = Octonion::basis(r) - Octonion::basis(s);
                        assert!(!(x * y).is_zero(TOL));
                    }
                }
            }
        }
        // Keep the found witness visible in test output on failure elsewhere.
        eprintln!("sedenion zero divisor: (e{p} + e{q})(e{r} - e{s}) = 0");
    }

    #[test]
    fn cd_complex_matches_num_complex() {
        let mut prng = crate::rng::Prng::new(11);
        for _ in 0..200 {
            let (a, b, c, d) = (
                prng.next_f64() * 2.0 - 1.0,
                prng.next_f64() * 2.0 - 1.0,
                prng.next_f64() * 2.0 - 1.0,
                prng.next_f64() * 2.0 - 1.0,
            );
            let x = CComplex::from_coeffs(&[a, b]);
            let y = CComplex::from_coeffs(&[c, d]);
            let zx = C64::new(a, b);
            let zy = C64::new(c, d);
            let prod = x * y;
            let zprod = zx * zy;
            assert!((prod.coeffs()[0] - zprod.re).abs() < TOL);
            assert!((prod.coeffs()[1] - zprod.im).abs() < TOL);
            let conj = x.conj();
            assert!((conj.coeffs()[0] - zx.re).abs() < TOL);
            assert!((conj.coeffs()[1] + zx.im).abs() < TOL);
        }
        assert!(CComplex::COMMUTATIVE && CComplex::ASSOCIATIVE && CComplex::DIVISION);
    }

    #[test]
    fn complex_embedding_shares_i_down_the_tower() {
        // try_from_c64(i) into H must be (i, 0), i.e. coeffs [0,1,0,0].
        let i_h = Quaternion::try_from_c64(C64::new(0.0, 1.0)).unwrap();
        assert_eq!(i_h.coeffs(), vec![0.0, 1.0, 0.0, 0.0]);
        let i_o = Octonion::try_from_c64(C64::new(0.0, 1.0)).unwrap();
        assert_eq!(i_o.coeffs(), vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // And in CD<f64>, i is the doubling unit (0, 1).
        let i_c = CComplex::try_from_c64(C64::new(0.0, 1.0)).unwrap();
        assert_eq!(i_c.coeffs(), vec![0.0, 1.0]);
        // The embedded i squares to -1 everywhere.
        assert!((i_h * i_h).approx_eq(-Quaternion::one(), TOL));
        assert!((i_o * i_o).approx_eq(-Octonion::one(), TOL));
        assert!((i_c * i_c).approx_eq(-CComplex::one(), TOL));
    }

    #[test]
    fn norm_multiplicative_for_division_algebras_only() {
        let mut prng = crate::rng::Prng::new(23);
        let mut rand_coeffs = |dim: usize| -> Vec<f64> {
            (0..dim).map(|_| prng.next_f64() * 2.0 - 1.0).collect()
        };
        for _ in 0..100 {
            let x = Quaternion::from_coeffs(&rand_coeffs(4));
            let y = Quaternion::from_coeffs(&rand_coeffs(4));
            assert!(((x * y).abs_sqr() - x.abs_sqr() * y.abs_sqr()).abs() < 1e-9);
            let x = Octonion::from_coeffs(&rand_coeffs(8));
            let y = Octonion::from_coeffs(&rand_coeffs(8));
            assert!(((x * y).abs_sqr() - x.abs_sqr() * y.abs_sqr()).abs() < 1e-9);
        }
        // Sedenions violate multiplicativity (any zero-divisor pair shows it).
        let mut violated = false;
        for _ in 0..100 {
            let x = Sedenion::from_coeffs(&rand_coeffs(16));
            let y = Sedenion::from_coeffs(&rand_coeffs(16));
            if ((x * y).abs_sqr() - x.abs_sqr() * y.abs_sqr()).abs() > 1e-6 {
                violated = true;
            }
        }
        assert!(violated, "sedenion norm should not be multiplicative");
    }
}
