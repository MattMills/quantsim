//! `n` abstract polarities in one amplitude: the split-quaternion
//! inclusion/exclusion pair, generalized.
//!
//! [`SplitQuaternion`](super::SplitQuaternion) carries *one* polarity —
//! a single `j` with `j² = +1` splitting each amplitude into a
//! constructive and a destructive component. [`Polarity<N>`] carries
//! `N` of them: generators `j₀ … j_{N−1}`, each squaring to `+1`, every
//! distinct pair **anticommuting**:
//!
//! ```text
//! j_a² = +1,      j_a j_b = −j_b j_a   (a ≠ b)
//! ```
//!
//! The anticommutation is the *twist*, and it is what makes the system
//! more than `N` independent ledgers. A basis element is a subset of
//! the generators — a **polarity monomial** — so the algebra has real
//! dimension `2^N`, and the two magnitudes split by monomial **degree
//! parity**:
//!
//! * [`Scalar::abs_sqr`] = `Σ c_g²` — the path weight.
//! * [`Scalar::born_weight`] = `Σ (−1)^{deg g} c_g²` — the net weight:
//!   even-degree monomials count as inclusions, odd-degree ones as
//!   exclusions. (That falls out of `q·conj(q)`; it is not imposed.)
//!
//! `Polarity<1>` is the split-complex numbers and `Polarity<2>` **is**
//! the split quaternions — the isomorphism `1↔1, i↔j₀j₁, j↔j₀, k↔−j₁`
//! is checked entrywise in `tests/polarity.rs`. From `N ≥ 2` onward
//! `j₀j₁` squares to `−1`, so ℂ embeds and the full standard gate
//! library exists over every one of these algebras.
//!
//! **What this costs, stated first.** `N` polarities per amplitude is
//! `2^N` reals per amplitude. Tracking one polarity per qubit of an
//! `n`-qubit register therefore costs `2^n` *per stored amplitude* — the
//! exponential does not go away, it moves from the register into the
//! scalar. That is measured in `examples/polarity_systems.rs` rather
//! than argued: the width sweep shows `Polarity<4>` paying 16 reals an
//! amplitude for the same states `C64` holds in 2. What the twist
//! actually buys, and the precise sector where polarity data *is*
//! locally storable, is the subject of [`crate::polarity`].

use super::{Scalar, C64};

/// Largest polarity count this representation stores.
pub const MAX_POLARITIES: usize = 4;

/// Coefficient storage: `2^MAX_POLARITIES` slots, of which the first
/// `2^N` are live.
const SLOTS: usize = 1 << MAX_POLARITIES;

/// Product of two polarity monomials in the fully twisted system:
/// every generator squares to `+1` and every distinct pair
/// anticommutes. Returns the resulting monomial and its sign.
///
/// Monomials are held in increasing generator order, so multiplying in
/// `j_g` means carrying it left past every generator of the running
/// monomial with a *higher* index — each such swap flips the sign — and
/// then either cancelling its twin (`j_g² = +1`) or inserting it.
pub(crate) fn monomial_product(a: usize, b: usize, n: usize) -> (usize, f64) {
    let mut mask = a;
    let mut sign = 1.0f64;
    for g in 0..n {
        if (b >> g) & 1 == 0 {
            continue;
        }
        if (mask >> (g + 1)).count_ones() % 2 == 1 {
            sign = -sign;
        }
        mask ^= 1 << g;
    }
    (mask, sign)
}

/// Sign of Clifford conjugation on a degree-`d` monomial:
/// `(−1)^{d(d+1)/2}`, the composition of reversion with the grade
/// involution — the anti-automorphism the [`Scalar`] contract wants.
fn conj_sign(mask: usize) -> f64 {
    let d = mask.count_ones() as usize;
    if (d * (d + 1) / 2) % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// An amplitude carrying `N` abstract polarities.
///
/// Coordinates are indexed by polarity monomial: coordinate `m` is the
/// coefficient of `∏_{g ∈ m} j_g`, with `m` read as a bitmask over the
/// generators.
#[derive(Debug, Clone, Copy)]
pub struct Polarity<const N: usize> {
    c: [f64; SLOTS],
}

impl<const N: usize> PartialEq for Polarity<N> {
    fn eq(&self, other: &Self) -> bool {
        self.c[..1 << N] == other.c[..1 << N]
    }
}

impl<const N: usize> Polarity<N> {
    /// Number of polarity generators.
    pub const POLARITIES: usize = N;

    /// The zero element with the width assertion in one place.
    fn blank() -> Self {
        assert!(
            N >= 1 && N <= MAX_POLARITIES,
            "Polarity<{N}> is out of range: 1..={MAX_POLARITIES} polarities are stored"
        );
        Polarity { c: [0.0; SLOTS] }
    }

    /// The coefficient of a polarity monomial, given as a generator
    /// bitmask.
    pub fn coefficient(self, monomial: usize) -> f64 {
        assert!(monomial < 1 << N, "monomial {monomial} out of range");
        self.c[monomial]
    }

    /// Build from a single monomial and coefficient.
    pub fn monomial(monomial: usize, coefficient: f64) -> Self {
        let mut out = Self::blank();
        assert!(monomial < 1 << N, "monomial {monomial} out of range");
        out.c[monomial] = coefficient;
        out
    }

    /// The `g`-th polarity generator `j_g`.
    pub fn generator(g: usize) -> Self {
        assert!(g < N, "polarity {g} out of range for Polarity<{N}>");
        Self::monomial(1 << g, 1.0)
    }

    /// The embedded imaginary unit `j₀j₁`, which squares to `−1`.
    /// Requires `N ≥ 2`.
    pub fn imaginary() -> Option<Self> {
        (N >= 2).then(|| Self::monomial(0b11, 1.0))
    }

    /// Total weight on monomials of even degree — the **inclusion**
    /// side of the ledger.
    pub fn inclusion_weight(self) -> f64 {
        (0..1usize << N)
            .filter(|m| m.count_ones() % 2 == 0)
            .map(|m| self.c[m] * self.c[m])
            .sum()
    }

    /// Total weight on monomials of odd degree — the **exclusion**
    /// side.
    pub fn exclusion_weight(self) -> f64 {
        (0..1usize << N)
            .filter(|m| m.count_ones() % 2 == 1)
            .map(|m| self.c[m] * self.c[m])
            .sum()
    }

    /// Weight carried at each polarity degree `0..=N` — how the
    /// amplitude is spread across the polarity grading.
    pub fn degree_profile(self) -> Vec<f64> {
        let mut out = vec![0.0; N + 1];
        for m in 0..1usize << N {
            out[m.count_ones() as usize] += self.c[m] * self.c[m];
        }
        out
    }
}

impl<const N: usize> std::ops::Add for Polarity<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut out = Self::blank();
        for m in 0..1 << N {
            out.c[m] = self.c[m] + rhs.c[m];
        }
        out
    }
}

impl<const N: usize> std::ops::Sub for Polarity<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut out = Self::blank();
        for m in 0..1 << N {
            out.c[m] = self.c[m] - rhs.c[m];
        }
        out
    }
}

impl<const N: usize> std::ops::Neg for Polarity<N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut out = Self::blank();
        for m in 0..1 << N {
            out.c[m] = -self.c[m];
        }
        out
    }
}

impl<const N: usize> std::ops::Mul for Polarity<N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut out = Self::blank();
        for a in 0..1 << N {
            if self.c[a] == 0.0 {
                continue;
            }
            for b in 0..1 << N {
                if rhs.c[b] == 0.0 {
                    continue;
                }
                let (m, sign) = monomial_product(a, b, N);
                out.c[m] += sign * self.c[a] * rhs.c[b];
            }
        }
        out
    }
}

impl<const N: usize> Scalar for Polarity<N> {
    const DIM: usize = 1 << N;
    // One polarity is just split-complex: commutative. Two or more
    // anticommute.
    const COMMUTATIVE: bool = N < 2;
    const ASSOCIATIVE: bool = true;
    const DIVISION: bool = false;
    const TRIVIAL_CONJ: bool = false;

    fn algebra_name() -> String {
        format!("Pol{N}")
    }
    fn zero() -> Self {
        Self::blank()
    }
    fn one() -> Self {
        Self::monomial(0, 1.0)
    }
    fn conj(self) -> Self {
        let mut out = Self::blank();
        for m in 0..1 << N {
            out.c[m] = conj_sign(m) * self.c[m];
        }
        out
    }
    fn scale(self, k: f64) -> Self {
        let mut out = Self::blank();
        for m in 0..1 << N {
            out.c[m] = self.c[m] * k;
        }
        out
    }
    fn re(self) -> f64 {
        self.c[0]
    }
    fn abs_sqr(self) -> f64 {
        self.c[..1 << N].iter().map(|x| x * x).sum()
    }
    fn born_weight(self) -> f64 {
        // Scalar part of q·conj(q): even-degree monomials contribute
        // +, odd-degree ones −. Inclusion minus exclusion, by degree
        // parity.
        self.inclusion_weight() - self.exclusion_weight()
    }
    fn try_from_c64(z: C64) -> Option<Self> {
        if z.im == 0.0 {
            return Some(Self::monomial(0, z.re));
        }
        // j₀j₁ squares to −1, so all of ℂ embeds once there are two
        // polarities to twist against each other.
        if N < 2 {
            return None;
        }
        let mut out = Self::monomial(0, z.re);
        out.c[0b11] = z.im;
        Some(out)
    }
    fn coeffs(self) -> Vec<f64> {
        self.c[..1 << N].to_vec()
    }
    fn from_coeffs(c: &[f64]) -> Self {
        assert_eq!(c.len(), 1 << N, "Polarity<{N}> has dimension {}", 1 << N);
        let mut out = Self::blank();
        out.c[..1 << N].copy_from_slice(c);
        out
    }
}

/// Two polarities: the split quaternions.
pub type Polarity2 = Polarity<2>;
/// Three polarities.
pub type Polarity3 = Polarity<3>;
/// Four polarities.
pub type Polarity4 = Polarity<4>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generators_square_to_one_and_anticommute() {
        let (a, b) = (Polarity::<3>::generator(0), Polarity::<3>::generator(1));
        assert_eq!(a * a, Polarity::<3>::one());
        assert_eq!(b * b, Polarity::<3>::one());
        assert_eq!(a * b, -(b * a));
    }

    #[test]
    fn two_twisted_polarities_give_an_imaginary_unit() {
        let i = Polarity::<2>::imaginary().unwrap();
        assert_eq!(i * i, -Polarity::<2>::one());
        assert!(Polarity::<1>::imaginary().is_none());
        // ...and therefore ℂ embeds from two polarities upward.
        assert!(Polarity::<1>::try_from_c64(C64::new(0.0, 1.0)).is_none());
        assert!(Polarity::<2>::try_from_c64(C64::new(0.0, 1.0)).is_some());
        assert!(Polarity::<4>::try_from_c64(C64::new(0.0, 1.0)).is_some());
    }

    #[test]
    fn the_born_form_splits_by_degree_parity() {
        let mut c = vec![0.0; 8];
        c[0] = 1.0; // degree 0 → inclusion
        c[0b001] = 2.0; // degree 1 → exclusion
        c[0b011] = 3.0; // degree 2 → inclusion
        c[0b111] = 4.0; // degree 3 → exclusion
        let q = Polarity::<3>::from_coeffs(&c);
        assert_eq!(q.inclusion_weight(), 1.0 + 9.0);
        assert_eq!(q.exclusion_weight(), 4.0 + 16.0);
        assert_eq!(q.born_weight(), 10.0 - 20.0);
        assert_eq!(q.abs_sqr(), 30.0);
        assert_eq!(q.degree_profile(), vec![1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn multiplication_is_associative_across_the_whole_basis() {
        for a in 0..8usize {
            for b in 0..8usize {
                for c in 0..8usize {
                    let (x, y, z) = (
                        Polarity::<3>::monomial(a, 1.0),
                        Polarity::<3>::monomial(b, 1.0),
                        Polarity::<3>::monomial(c, 1.0),
                    );
                    assert_eq!((x * y) * z, x * (y * z));
                }
            }
        }
    }

    #[test]
    fn conjugation_is_an_anti_automorphism() {
        for a in 0..16usize {
            for b in 0..16usize {
                let (x, y) = (
                    Polarity::<4>::monomial(a, 1.0),
                    Polarity::<4>::monomial(b, 1.0),
                );
                assert_eq!((x * y).conj(), y.conj() * x.conj());
            }
        }
    }
}
