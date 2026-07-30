//! Split quaternions as an **inclusion/exclusion pair**: one amplitude
//! carrying a constructive and a destructive complex component, tracked
//! independently, with measurement reading their difference.
//!
//! An element is `q = z₊ + z₋·j` with `z₊, z₋ ∈ ℂ = span{1, i}` and
//! `i² = −1`, `j² = +1`, `k = ij`, `ji = −ij`. Writing it as the pair
//! `(z₊, z₋)` is not a notational convenience — it is the algebra's own
//! ℤ₂ grading, and the product respects it exactly:
//!
//! ```text
//! (z₊ + z₋j)(w₊ + w₋j) = (z₊w₊ + z₋·conj(w₋)) + (z₊w₋ + z₋·conj(w₊))·j
//! ```
//!
//! Read as bookkeeping: **two exclusions make an inclusion, one of each
//! makes an exclusion.** That is the inclusion–exclusion parity rule,
//! and here it is the multiplication table rather than a convention laid
//! on top of one.
//!
//! The two magnitudes this crate keeps separate finally separate
//! *usefully*:
//!
//! * [`Scalar::abs_sqr`] = `|z₊|² + |z₋|²` — the **path weight**, every
//!   contribution counted regardless of sign.
//! * [`Scalar::born_weight`] = `|z₊|² − |z₋|²` — the **net weight**,
//!   inclusion minus exclusion. Indefinite, and zero on the null cone
//!   `|z₊| = |z₋|`: exact cancellation, decidable, with the path weight
//!   still positive to say how much was cancelled.
//!
//! **ℂ embeds** (`i` is right there), so unlike
//! [`SplitComplex`](super::SplitComplex) this algebra carries the *full*
//! standard gate library, and every ordinary circuit runs over it
//! unmodified. What such a circuit does is the first measured fact: an
//! embedded-ℂ matrix entry `m` acts as `m·(z₊, z₋) = (m z₊, m z₋)`, so
//! **a standard gate drives both channels with the same complex matrix
//! and never mixes them**. A split-quaternion register under a standard
//! circuit is exactly two ordinary complex registers evolving in
//! lockstep, with the Born rule reading their difference — independent
//! tracking, in one pass, at 2× the amplitude footprint.
//!
//! That grading also bounds what the representation can do, and the
//! bound is worth stating plainly: because the product is ℤ₂-graded,
//! *no* gate can evolve the two channels differently. Entries with
//! `z₋ = 0` drive both identically; entries with `z₋ ≠ 0` mix them. The
//! pair is a pair, not two independent registers.
//!
//! **Moving weight between the ledgers.** The unit-norm elements
//! (`N(q) = 1`) are `SL(2,ℝ)` — split quaternions are `M₂(ℝ)` and the
//! norm is the determinant — which is *non-compact*, and that is the
//! representation's point. [`SplitQuaternion::boost`] is
//! `cosh t + sinh t·j`: a genuinely unitary gate (its
//! `unitarity_deviation` measures to ~1e-16) that pumps matched
//! constructive and destructive weight into a state, so the **path
//! weight grows without bound while the net weight is conserved
//! exactly**. Constructive and destructive amplitude are created in
//! pairs at zero net cost — which is what interference does, here made
//! into a gate rather than inferred from a ledger.
//!
//! [`SplitQuaternion::exchange`] is `j` itself, with `N(j) = −1`: it
//! swaps the two channels (conjugating them) and therefore *negates* the
//! net weight. It is measurably not unitary — deviation exactly 2 — so
//! the gate registry refuses it and it must be applied through
//! [`Circuit::raw`](crate::circuit::Circuit::raw), which is the honest
//! status of an operation that turns constructive weight into
//! destructive weight outright.
//!
//! Consequences the simulator will report rather than hide: a state with
//! more exclusion than inclusion has non-positive total Born weight, and
//! measurement refuses it by the existing rule; a balanced state sits on
//! the null cone with net exactly 0 and path weight positive. Both are
//! measured in `tests/split_quaternion.rs`.

use super::{Scalar, C64};

/// A split quaternion held as an inclusion/exclusion pair
/// `q = inc + exc·j`, with `i² = −1` and `j² = +1`.
///
/// Coordinates are `[inc.re, inc.im, exc.re, exc.im]` — the coefficients
/// of `1, i, j, k` in that order, since `(c + d i)·j = c j + d k`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitQuaternion {
    /// The constructive component: the part measurement counts `+`.
    pub inc: C64,
    /// The destructive component: the part measurement counts `−`.
    pub exc: C64,
}

impl SplitQuaternion {
    /// Build from an explicit inclusion/exclusion pair.
    pub fn pair(inc: C64, exc: C64) -> Self {
        SplitQuaternion { inc, exc }
    }

    /// A purely constructive amplitude — the image of the ℂ embedding.
    pub fn included(z: C64) -> Self {
        SplitQuaternion::pair(z, C64::new(0.0, 0.0))
    }

    /// A purely destructive amplitude: `z·j`.
    pub fn excluded(z: C64) -> Self {
        SplitQuaternion::pair(C64::new(0.0, 0.0), z)
    }

    /// The constructive component.
    pub fn inclusion(self) -> C64 {
        self.inc
    }

    /// The destructive component.
    pub fn exclusion(self) -> C64 {
        self.exc
    }

    /// `|inc|² − |exc|²` — the net weight measurement uses. Same as
    /// [`Scalar::born_weight`], named for what it means here.
    pub fn net_weight(self) -> f64 {
        self.inc.norm_sqr() - self.exc.norm_sqr()
    }

    /// `|inc|² + |exc|²` — every contribution counted regardless of
    /// sign. Same as [`Scalar::abs_sqr`].
    pub fn path_weight(self) -> f64 {
        self.inc.norm_sqr() + self.exc.norm_sqr()
    }

    /// On the null cone: the two ledgers cancel exactly
    /// (`net ≈ 0`) while something was actually there
    /// (`path > tol`). Complete destructive interference, decidable.
    pub fn is_null(self, tol: f64) -> bool {
        self.net_weight().abs() <= tol && self.path_weight() > tol
    }

    /// The boost `cosh t + sinh t·j`: unit norm, hence a legitimate
    /// unitary gate entry, and the generator of weight transfer between
    /// the two ledgers. Applying it multiplies the path weight without
    /// touching the net weight.
    pub fn boost(rapidity: f64) -> Self {
        SplitQuaternion::pair(
            C64::new(rapidity.cosh(), 0.0),
            C64::new(rapidity.sinh(), 0.0),
        )
    }

    /// `j` itself: swaps the two ledgers (conjugating them) and negates
    /// the net weight, because `N(j) = −1`. Not unitary — see the module
    /// docs.
    pub fn exchange() -> Self {
        SplitQuaternion::excluded(C64::new(1.0, 0.0))
    }

    /// The orthogonal idempotents `e± = (1 ± j)/2`: `e±² = e±`,
    /// `e₊e₋ = 0`, `e₊ + e₋ = 1`. The split algebra's own decomposition
    /// into two independent halves — the reason "inclusion/exclusion
    /// pair" is a structural statement and not a metaphor.
    pub fn idempotents() -> (Self, Self) {
        let half = C64::new(0.5, 0.0);
        (
            SplitQuaternion::pair(half, half),
            SplitQuaternion::pair(half, -half),
        )
    }
}

impl std::ops::Add for SplitQuaternion {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        SplitQuaternion::pair(self.inc + rhs.inc, self.exc + rhs.exc)
    }
}

impl std::ops::Sub for SplitQuaternion {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        SplitQuaternion::pair(self.inc - rhs.inc, self.exc - rhs.exc)
    }
}

impl std::ops::Neg for SplitQuaternion {
    type Output = Self;
    fn neg(self) -> Self {
        SplitQuaternion::pair(-self.inc, -self.exc)
    }
}

impl std::ops::Mul for SplitQuaternion {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // (z₊ + z₋j)(w₊ + w₋j), using j·w = conj(w)·j and j² = +1:
        //   inclusion ← z₊w₊ + z₋·conj(w₋)   (two exclusions make an inclusion)
        //   exclusion ← z₊w₋ + z₋·conj(w₊)   (one of each makes an exclusion)
        SplitQuaternion::pair(
            self.inc * rhs.inc + self.exc * rhs.exc.conj(),
            self.inc * rhs.exc + self.exc * rhs.inc.conj(),
        )
    }
}

impl Scalar for SplitQuaternion {
    const DIM: usize = 4;
    const COMMUTATIVE: bool = false;
    const ASSOCIATIVE: bool = true;
    const DIVISION: bool = false;
    const TRIVIAL_CONJ: bool = false;

    fn algebra_name() -> String {
        "H_split".to_string()
    }
    fn zero() -> Self {
        SplitQuaternion::pair(C64::new(0.0, 0.0), C64::new(0.0, 0.0))
    }
    fn one() -> Self {
        SplitQuaternion::included(C64::new(1.0, 0.0))
    }
    fn conj(self) -> Self {
        // conj(a + bi + cj + dk) = a − bi − cj − dk = conj(inc) − exc·j.
        SplitQuaternion::pair(self.inc.conj(), -self.exc)
    }
    fn scale(self, k: f64) -> Self {
        SplitQuaternion::pair(self.inc * k, self.exc * k)
    }
    fn re(self) -> f64 {
        self.inc.re
    }
    fn abs_sqr(self) -> f64 {
        self.path_weight()
    }
    fn born_weight(self) -> f64 {
        // q·conj(q) = |inc|² − |exc|². Indefinite, signature (2,2).
        self.net_weight()
    }
    fn try_from_c64(z: C64) -> Option<Self> {
        // i is present, so all of ℂ embeds — into the inclusion channel.
        Some(SplitQuaternion::included(z))
    }
    fn coeffs(self) -> Vec<f64> {
        vec![self.inc.re, self.inc.im, self.exc.re, self.exc.im]
    }
    fn from_coeffs(c: &[f64]) -> Self {
        assert_eq!(c.len(), 4, "H_split has dimension 4");
        SplitQuaternion::pair(C64::new(c[0], c[1]), C64::new(c[2], c[3]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(a: f64, b: f64, c: f64, d: f64) -> SplitQuaternion {
        SplitQuaternion::from_coeffs(&[a, b, c, d])
    }

    #[test]
    fn the_coquaternion_multiplication_table() {
        let (one, i, j, k) = (
            SplitQuaternion::one(),
            SplitQuaternion::basis(1),
            SplitQuaternion::basis(2),
            SplitQuaternion::basis(3),
        );
        assert_eq!(i * i, -one);
        assert_eq!(j * j, one);
        assert_eq!(k * k, one);
        assert_eq!(i * j, k);
        assert_eq!(j * i, -k);
        assert_eq!(j * k, -i);
        assert_eq!(k * j, i);
        assert_eq!(k * i, j);
        assert_eq!(i * k, -j);
    }

    #[test]
    fn the_norm_is_the_split_form_and_multiplicative() {
        let x = q(1.0, 2.0, 3.0, 4.0);
        assert_eq!(x.born_weight(), 1.0 + 4.0 - 9.0 - 16.0);
        assert_eq!(x.abs_sqr(), 1.0 + 4.0 + 9.0 + 16.0);
        // N(xy) = N(x)N(y): the determinant of M₂(ℝ).
        let y = q(-2.0, 0.5, 1.0, -3.0);
        let lhs = (x * y).born_weight();
        let rhs = x.born_weight() * y.born_weight();
        assert!((lhs - rhs).abs() < 1e-12, "{lhs} vs {rhs}");
    }

    #[test]
    fn the_null_cone_carries_zero_divisors() {
        let x = SplitQuaternion::one() + SplitQuaternion::exchange();
        let y = SplitQuaternion::one() - SplitQuaternion::exchange();
        assert_eq!(x * y, SplitQuaternion::zero());
        assert!(x.is_null(1e-12));
        assert_eq!(x.net_weight(), 0.0);
        assert!(x.path_weight() > 0.0);
    }

    #[test]
    fn the_idempotents_split_the_algebra() {
        let (ep, em) = SplitQuaternion::idempotents();
        assert_eq!(ep * ep, ep);
        assert_eq!(em * em, em);
        assert_eq!(ep * em, SplitQuaternion::zero());
        assert_eq!(em * ep, SplitQuaternion::zero());
        assert_eq!(ep + em, SplitQuaternion::one());
    }

    #[test]
    fn a_boost_is_unit_norm_and_the_exchange_is_not() {
        for t in [0.0, 0.5, 1.5, -2.0] {
            let b = SplitQuaternion::boost(t);
            assert!((b.born_weight() - 1.0).abs() < 1e-12);
            // Path weight grows as cosh(2t) while the net stays 1.
            assert!((b.path_weight() - (2.0 * t).cosh()).abs() < 1e-12);
        }
        assert_eq!(SplitQuaternion::exchange().born_weight(), -1.0);
    }

    #[test]
    fn embedded_complex_entries_drive_both_channels_identically() {
        let m = SplitQuaternion::included(C64::new(0.3, -0.7));
        let x = q(1.0, 2.0, 3.0, 4.0);
        let out = m * x;
        assert_eq!(out.inc, C64::new(0.3, -0.7) * x.inc);
        assert_eq!(out.exc, C64::new(0.3, -0.7) * x.exc);
    }

    #[test]
    fn the_exchange_swaps_the_ledgers_and_flips_the_net() {
        let x = q(1.0, 2.0, 0.5, -0.25);
        let out = SplitQuaternion::exchange() * x;
        assert_eq!(out.inc, x.exc.conj());
        assert_eq!(out.exc, x.inc.conj());
        assert!((out.net_weight() + x.net_weight()).abs() < 1e-12);
        assert!((out.path_weight() - x.path_weight()).abs() < 1e-12);
    }
}
