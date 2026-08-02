//! The structured tail-radix phase web: why a mixed-radix Fourier
//! transform is *banded*, and what its couplings cost to represent
//! exactly.
//!
//! A qudit chain with local dimensions `d_0 … d_{n−1}` (site 0 most
//! significant) encodes the ring `ℤ_N`, `N = Π d_k`. Writing the input in
//! ordinary place value and the output digit of site `j` in **reversed**
//! place value (`Q_j = Π_{k<j} d_k`), the Fourier kernel factorizes:
//!
//! ```text
//!   exp(2πi·x·y/N) = Π_j Π_{i ≥ j} exp(2πi · x_i · y'_j / D_{j,i}),
//!   D_{j,i} = d_j · d_{j+1} ··· d_i          (the radix segment product)
//! ```
//!
//! The reason the other half vanishes is worth stating, because it is the
//! whole structure: for `i < j` the place values contribute
//! `Π_{i<k<j} d_k`, an **integer**, so the phase is a whole number of
//! turns and the coupling is exactly absent. Output digit `j` is
//! phase-correlated only with the *tail* of the register.
//!
//! ## Why that makes it banded
//!
//! `D_{j,i}` is a product over every dimension between the two sites, so
//! it grows at least like `2^{i−j}` and the coupling angle `2π/D_{j,i}`
//! decays at least exponentially with distance. Truncating at an angle
//! threshold therefore keeps a *band* rather than a triangle: the web has
//! `n(n+1)/2` couplings in principle and `O(n·log(1/ε))` above any fixed
//! `ε`. Segments only grow along a row, so the scan can stop at the first
//! coupling below threshold.
//!
//! ## What it costs to hold exactly
//!
//! The angles are `2π/D` for `D` a product of local dimensions, so they
//! are roots of unity of *composite* order. [`crate::pathsum`] is exact
//! over dyadic angles and refuses everything else rather than rounding,
//! which makes "how much of this web is dyadic?" a sharp question with a
//! measured answer: all of it on a binary chain, almost none of it once
//! an odd dimension appears anywhere in a segment. See
//! [`dyadic_share`].
//!
//! Ported from the `novel_quantum_structures` research package
//! (`radix.rs`), with the factorization identity checked numerically
//! rather than assumed.

use std::f64::consts::TAU;

/// Largest ring size [`verify_factorization`] will sweep exhaustively.
/// Named so `grep 'pub const MAX'` finds every ceiling in the crate.
pub const MAX_SWEEP_N: u128 = 4096;

/// One surviving coupling of the phase web.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coupling {
    /// Output digit index `j` (reversed place value).
    pub out: usize,
    /// Input digit index `i`, always `≥ out`.
    pub inp: usize,
    /// The radix segment product `D_{j,i} = d_j ··· d_i`, which is the
    /// denominator of the coupling's angle.
    pub denom: u128,
}

impl Coupling {
    /// The coupling angle `2π / D_{j,i}` in radians.
    pub fn angle(&self) -> f64 {
        TAU / self.denom as f64
    }

    /// Distance along the chain — the band index.
    pub fn span(&self) -> usize {
        self.inp - self.out
    }

    /// Whether the angle is a dyadic fraction of a turn, i.e. whether
    /// `pathsum` could hold it exactly rather than refusing it.
    pub fn is_dyadic(&self) -> bool {
        self.denom.is_power_of_two()
    }
}

/// The radix segment product `D_{j,i} = d_j · d_{j+1} ··· d_i`.
pub fn segment(profile: &[usize], j: usize, i: usize) -> u128 {
    assert!(j <= i && i < profile.len(), "segment out of range");
    profile[j..=i].iter().map(|&d| d as u128).product()
}

/// The full tail web: every coupling with `i ≥ j`, unpruned.
///
/// `n(n+1)/2` entries — the triangle before banding is applied.
pub fn full_web(profile: &[usize]) -> Vec<Coupling> {
    let n = profile.len();
    let mut out = Vec::with_capacity(n * (n + 1) / 2);
    for j in 0..n {
        for i in j..n {
            out.push(Coupling {
                out: j,
                inp: i,
                denom: segment(profile, j, i),
            });
        }
    }
    out
}

/// The **banded** web: couplings whose angle is at least `min_angle`.
///
/// Segments only grow along a row, so once one coupling falls below the
/// threshold every later one on that row does too and the scan stops —
/// which is why this is cheap even when the full triangle is not.
pub fn phase_web(profile: &[usize], min_angle: f64) -> Vec<Coupling> {
    let n = profile.len();
    let mut out = Vec::new();
    for j in 0..n {
        for i in j..n {
            let denom = segment(profile, j, i);
            let c = Coupling {
                out: j,
                inp: i,
                denom,
            };
            if c.angle() < min_angle {
                break; // segments only grow: every later coupling is smaller
            }
            out.push(c);
        }
    }
    out
}

/// Digits of `x` in ordinary place value, site 0 most significant.
pub fn digits(profile: &[usize], x: u128) -> Vec<u128> {
    let mut place: u128 = profile.iter().map(|&d| d as u128).product();
    profile
        .iter()
        .map(|&d| {
            place /= d as u128;
            (x / place) % d as u128
        })
        .collect()
}

/// Digits of `y` in **reversed** place value: digit `j` has place
/// `Q_j = Π_{k<j} d_k`.
pub fn reversed_digits(profile: &[usize], y: u128) -> Vec<u128> {
    let mut place: u128 = 1;
    profile
        .iter()
        .map(|&d| {
            let v = (y / place) % d as u128;
            place *= d as u128;
            v
        })
        .collect()
}

/// The **independent check**: the largest deviation between the Fourier
/// kernel `exp(2πi·x·y/N)` and the tail-web product, over all `x, y`.
///
/// Nothing else in this module evaluates the kernel; this does, and if it
/// ever returns a nonzero residual the factorization — not the code
/// around it — is what is wrong. Returns `None` past [`MAX_SWEEP_N`]
/// rather than sweeping forever.
pub fn verify_factorization(profile: &[usize]) -> Option<f64> {
    let n: u128 = profile.iter().map(|&d| d as u128).product();
    if n > MAX_SWEEP_N {
        return None;
    }
    let mut worst = 0f64;
    for x in 0..n {
        let xd = digits(profile, x);
        for y in 0..n {
            let yd = reversed_digits(profile, y);
            // The kernel, evaluated directly.
            let direct = TAU * (x * y % n) as f64 / n as f64;
            // The product over surviving couplings, as a total angle.
            let mut acc = 0f64;
            for (j, &yj) in yd.iter().enumerate() {
                for (i, &xi) in xd.iter().enumerate().skip(j) {
                    let d = segment(profile, j, i) as f64;
                    acc += TAU * (xi * yj) as f64 / d;
                }
            }
            let diff = (direct - acc) / TAU;
            // Equality is mod one whole turn.
            let frac = diff - diff.round();
            worst = worst.max(frac.abs());
        }
    }
    Some(worst)
}

/// The fraction of the full web whose angle is dyadic — the part
/// [`crate::pathsum`] could hold exactly rather than refuse.
///
/// A coupling's denominator is a product of local dimensions, so it is a
/// power of two exactly when every dimension in its segment is. On a
/// binary chain that is all of them; one odd dimension poisons every
/// segment spanning it.
pub fn dyadic_share(profile: &[usize]) -> f64 {
    let web = full_web(profile);
    let dyadic = web.iter().filter(|c| c.is_dyadic()).count();
    dyadic as f64 / web.len() as f64
}

/// Couplings per output digit at a threshold — the measured bandwidth.
///
/// The point of the whole structure: this stays bounded as the chain
/// grows, where the full triangle would grow like `n`.
pub fn bandwidth(profile: &[usize], min_angle: f64) -> usize {
    let web = phase_web(profile, min_angle);
    let n = profile.len();
    (0..n)
        .map(|j| web.iter().filter(|c| c.out == j).count())
        .max()
        .unwrap_or(0)
}
