//! The a-priori width calculus of modular multiplication: an operator's
//! Schmidt rank across a cut, computed from number theory before any
//! tensor exists.
//!
//! [`crate::cut`] predicts a cut from the coupling graph and returns
//! *bounds*. This module does the same job for one operator family and
//! returns the **exact** rank, in closed form, with no state and no
//! matrix built at any point.
//!
//! Split the ring `ℤ_N` at a cut with a left ring of size `L` and a right
//! block of size `S`, `N = L·S`, and write `x = a·S + b`. Then the
//! multiplier `M_k : |x⟩ → |k·x mod N⟩` factors branch-wise over the
//! **crossing message** — the multiplication carry entering the cut:
//!
//! ```text
//!   M_k |a, b⟩ = |(k·a + c(b)) mod L,  k·b mod S⟩,     c(b) = ⌊k·b / S⌋
//! ```
//!
//! and the operator Schmidt rank across the cut is exactly
//!
//! ```text
//!   |{ c(b) mod L : b ∈ [0, S) }|
//! ```
//!
//! Two regimes. For `k ≤ S` the map `b ↦ ⌊kb/S⌋` steps by 0 or 1 and
//! reaches `k−1`, so it is a surjection onto `[0, k)` and the width is
//! `min(k, L)` with no enumeration at all. For `k > S` the sequence skips
//! and number theory takes over — that is where the resonances live.
//!
//! ## What that buys: cost is resonant, not monotone
//!
//! The striking consequence is that **multiplying by a larger number can
//! be cheaper**. At the waist cut `L = 24, S = 120` of `ℤ_2880`, the
//! repeated-squaring chain `7 → 49 → 2401 → 1921` gives widths
//! `7 → 24 → 6 → 3`. The cost of `×k` is not monotone in `k`; it is a
//! property of `k`'s arithmetic relationship to the cut.
//!
//! And it persists under composition, which is what makes it useful:
//! `×7^K` stays at width 3 however large `K` gets, so a reversible
//! computation of unbounded depth can sit at fixed width. That is a
//! statement about *which* computations are cheap rather than how big
//! they are, which is the only kind of statement that helps.
//!
//! ## Scope, stated plainly
//!
//! This is exact for modular multiplication on a chain cut and says
//! nothing about circuits in general. The rank it returns is the operator
//! Schmidt rank of one permutation across one cut. Ported from the
//! `novel_quantum_structures` research package (`width.rs`, THEORY.md §8).

use std::collections::HashSet;

/// Largest right block this module will enumerate over.
///
/// Past it the crossing set cannot be scanned and the answer degrades to
/// a bound rather than silently costing more. Named so that
/// `grep 'pub const MAX'` finds every ceiling in the crate.
pub const MAX_ENUM_BLOCK: u128 = 8_000_000;

/// Largest `L` for which distinct residues are counted with a bitset
/// rather than a hash set. Purely an allocation policy — the answer is
/// identical either way.
pub const MAX_BITSET_RING: u128 = 1 << 25;

/// Largest `N = L·S` for which [`exact_schmidt_rank`] will build the
/// permutation and eliminate. The independent check is `O(N²)` in memory,
/// so it anchors the theorem at small `N` rather than everywhere.
pub const MAX_EXACT_RANK_N: usize = 4096;

/// A per-cut width statement: the exact rank, or only a bound.
///
/// The distinction is reported rather than blurred — a caller can always
/// tell whether the number is the answer or an upper bound on it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Width {
    /// The exact operator Schmidt rank across the cut.
    Exact(u128),
    /// Only the bound `min(k mod N, L, S)`: the crossing set was past
    /// [`MAX_ENUM_BLOCK`] and no closed form applied.
    UpperBound(u128),
}

impl Width {
    /// The number, whichever kind it is.
    pub fn value(self) -> u128 {
        match self {
            Width::Exact(v) | Width::UpperBound(v) => v,
        }
    }

    /// Whether this is the rank rather than a bound on it.
    pub fn is_exact(self) -> bool {
        matches!(self, Width::Exact(_))
    }
}

impl std::fmt::Display for Width {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Width::Exact(v) => write!(f, "{v}"),
            Width::UpperBound(v) => write!(f, "≤{v}"),
        }
    }
}

/// Operator width of `×k` across a cut with left ring `l` and right block
/// `s`: the count `|{⌊k·b/s⌋ mod l : b ∈ [0, s)}|`.
///
/// This is the operator Schmidt rank of the multiplier across the cut
/// when `gcd(k, l·s) = 1`; the count is well defined either way.
/// Reducing `k` mod `N` is sound, since replacing `k` by `k + LS` changes
/// `c(b)` by `L·b ≡ 0 (mod L)`.
pub fn mult_cut_width(k: u128, l: u128, s: u128) -> Width {
    assert!(l >= 1 && s >= 1, "cut sizes must be positive");
    let k = k % (l * s);
    if k == 0 {
        return Width::Exact(1);
    }
    if k <= s {
        // `b ↦ ⌊kb/s⌋` steps by 0 or 1 and reaches `k−1`: a surjection
        // onto `[0, k)`, so no enumeration is needed.
        return Width::Exact(k.min(l));
    }
    if s <= MAX_ENUM_BLOCK {
        let count = if l <= MAX_BITSET_RING {
            let mut seen = vec![false; l as usize];
            let mut count = 0u128;
            for b in 0..s {
                let c = ((k * b / s) % l) as usize;
                if !seen[c] {
                    seen[c] = true;
                    count += 1;
                }
            }
            count
        } else {
            let mut seen: HashSet<u128> = HashSet::new();
            for b in 0..s {
                seen.insert((k * b / s) % l);
            }
            seen.len() as u128
        };
        return Width::Exact(count);
    }
    Width::UpperBound(k.min(l).min(s))
}

/// Per-bond width profile of `×k` on a chain of local dimensions.
///
/// Entry `m` is the width across the bond between sites `m` and `m+1`,
/// with left ring `Π_{i≤m} d_i` and right block `Π_{i>m} d_i`. This is the
/// whole cost curve of the operator, computed without building it.
pub fn mult_width_profile(profile: &[usize], k: u128) -> Vec<Width> {
    assert!(profile.len() >= 2, "a chain needs at least two sites");
    let n: u128 = profile.iter().map(|&d| d as u128).product();
    let mut left = 1u128;
    profile[..profile.len() - 1]
        .iter()
        .map(|&d| {
            left *= d as u128;
            mult_cut_width(k, left, n / left)
        })
        .collect()
}

/// The **independent** check: build the permutation and take its rank.
///
/// Everything above is number theory; this is linear algebra, and it
/// never mentions the crossing message `c(b)`. It builds `M_k` straight
/// from `|x⟩ → |kx mod N⟩`, reshapes it to the operator matrix
/// `R[(a', a)][(b', b)]`, and eliminates. If the two ever disagree, the
/// theorem — not the implementation — is what is wrong.
///
/// Returns `None` past [`MAX_EXACT_RANK_N`] rather than allocating.
pub fn exact_schmidt_rank(k: u128, l: usize, s: usize) -> Option<usize> {
    let n = l.checked_mul(s)?;
    if n > MAX_EXACT_RANK_N {
        return None;
    }
    // R is indexed by (a', a) rows and (b', b) columns, and holds the
    // permutation's entry at |a',b'⟩⟨a,b|.
    let rows = l * l;
    let cols = s * s;
    let mut r = vec![vec![0f64; cols]; rows];
    for x in 0..n {
        let (a, b) = (x / s, x % s);
        let y = ((k * x as u128) % n as u128) as usize;
        let (a2, b2) = (y / s, y % s);
        r[a2 * l + a][b2 * s + b] = 1.0;
    }
    // Gaussian elimination. Entries are 0/1 and the matrix is sparse and
    // structured, so plain partial pivoting is ample here.
    let mut rank = 0usize;
    let mut pivot_row = 0usize;
    for col in 0..cols {
        let Some(sel) = (pivot_row..rows).find(|&i| r[i][col].abs() > 1e-9) else {
            continue;
        };
        r.swap(pivot_row, sel);
        let p = r[pivot_row][col];
        for i in 0..rows {
            if i != pivot_row && r[i][col].abs() > 1e-9 {
                let f = r[i][col] / p;
                let (lo, hi) = if i < pivot_row {
                    let (a, b) = r.split_at_mut(pivot_row);
                    (&mut a[i], &b[0])
                } else {
                    let (a, b) = r.split_at_mut(i);
                    (&mut b[0], &a[pivot_row])
                };
                for (dst, src) in lo[col..].iter_mut().zip(&hi[col..]) {
                    *dst -= f * src;
                }
            }
        }
        pivot_row += 1;
        rank += 1;
        if pivot_row == rows {
            break;
        }
    }
    Some(rank)
}

/// `k⁻¹ mod n`, for the inversion symmetry. Panics unless `gcd(k, n) = 1`.
pub fn mod_inverse(k: u128, n: u128) -> u128 {
    let (mut old_r, mut r) = (k as i128, n as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
    }
    assert_eq!(old_r, 1, "gcd(k, n) must be 1 to invert");
    old_s.rem_euclid(n as i128) as u128
}

/// The width of `k^e mod N` across a cut — the cost curve of modular
/// exponentiation, which is what a period-finding kernel actually runs.
///
/// Returned per exponent so the *shape* of the curve is visible: it is
/// resonant rather than monotone, and some bases settle at a fixed width
/// and stay there however deep the computation goes.
pub fn power_width_curve(k: u128, l: u128, s: u128, exponents: usize) -> Vec<Width> {
    let n = l * s;
    let mut acc = 1u128 % n;
    (0..exponents)
        .map(|_| {
            acc = (acc * k) % n;
            mult_cut_width(acc, l, s)
        })
        .collect()
}
