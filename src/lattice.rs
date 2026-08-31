//! Integer lattices: Gram–Schmidt, LLL reduction, and the kernel of a
//! system of congruences.
//!
//! The classical half of [`regev`](crate::regev) is a lattice problem, and
//! it is the half that decides whether the quantum samples are worth
//! anything. A run of Regev's algorithm returns vectors `w ∈ ℤ_q^d` that
//! are approximately orthogonal to a hidden lattice `L ⊆ ℤ^d`; recovering
//! `L` means finding short vectors of
//!
//! ```text
//! Λ = { v ∈ ℤ^d : ⟨w_j, v⟩ ≡ 0 (mod q) for every sampled w_j }
//! ```
//!
//! which [`congruence_kernel`] does by LLL on the standard embedding
//!
//! ```text
//! ⎡ I_d   c·Wᵀ ⎤
//! ⎣  0    c·q I ⎦
//! ```
//!
//! LLL can always absorb a multiple of `c·q` into a tail coordinate, so
//! the tail it actually minimizes is the **centred** residue
//! `⟨w_j, v⟩ mod± q`, and the objective is
//!
//! ```text
//! ‖v‖² + c² · Σ_j (⟨w_j, v⟩ mod± q)²
//! ```
//!
//! The weight `c` is a dial, and getting it wrong is the failure mode
//! that matters. A sampled `w` is only the *rounded* dual vector — the
//! quantum step returns `w_i ≈ q·y_i` with `|y|` irrational in general —
//! so a genuine `v ∈ L` has residue `O(‖v‖₁/2)`, **not** zero, and that
//! bound does not shrink as `q` grows. Setting `c` large therefore
//! demands an exactness the samples never had, and LLL answers with the
//! trivial kernel `qℤ^d`. `c = 1` weighs a unit of residue against a unit
//! of length, which is the right objective; `examples/regev.rs` measures
//! the whole dial and shows where it breaks.
//!
//! Everything is exact integer arithmetic on the basis; only the
//! Gram–Schmidt coefficients are floating point, which is the standard
//! LLL compromise and is safe at the dimensions this module sees
//! (`d + m ≤ 16`).

/// Inner product, widened so intermediate products cannot wrap.
pub fn dot(a: &[i64], b: &[i64]) -> i128 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i128 * y as i128)
        .sum()
}

/// Squared Euclidean norm.
pub fn norm_sq(a: &[i64]) -> i128 {
    dot(a, a)
}

/// A Gram–Schmidt orthogonalization of an integer basis.
#[derive(Debug, Clone)]
pub struct Gso {
    /// The orthogonalized vectors `b*_i`.
    pub star: Vec<Vec<f64>>,
    /// Coefficients `μ_{i,j} = ⟨b_i, b*_j⟩ / ‖b*_j‖²` for `j < i`.
    pub mu: Vec<Vec<f64>>,
    /// `‖b*_i‖²`.
    pub norms: Vec<f64>,
}

/// Gram–Schmidt orthogonalize an integer basis (rows are vectors).
pub fn gso(basis: &[Vec<i64>]) -> Gso {
    let n = basis.len();
    let dim = basis.first().map_or(0, |v| v.len());
    let mut star: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut mu: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    let mut norms: Vec<f64> = Vec::with_capacity(n);
    for (i, row) in basis.iter().enumerate() {
        let mut v: Vec<f64> = row.iter().map(|&x| x as f64).collect();
        for j in 0..i {
            let nj = norms[j];
            let m = if nj > 0.0 {
                let d: f64 = row
                    .iter()
                    .zip(star[j].iter())
                    .map(|(&x, &y)| x as f64 * y)
                    .sum();
                d / nj
            } else {
                0.0
            };
            mu[i][j] = m;
            if m != 0.0 {
                for k in 0..dim {
                    v[k] -= m * star[j][k];
                }
            }
        }
        norms.push(v.iter().map(|&x| x * x).sum());
        star.push(v);
    }
    Gso { star, mu, norms }
}

/// LLL-reduce `basis` in place with parameter `delta ∈ (0.25, 1)`
/// (`0.75` is Lenstra–Lenstra–Lovász's own choice). Returns the number of
/// swaps performed — a cheap measure of how far from reduced the input
/// was.
///
/// The Gram–Schmidt data is recomputed after every basis change. That is
/// `O(n³)` per step rather than the incremental `O(n)` update, and it is
/// the right trade at these dimensions: correctness with no accumulated
/// drift, in a routine that runs on bases of at most a dozen rows.
pub fn lll(basis: &mut [Vec<i64>], delta: f64) -> usize {
    let n = basis.len();
    if n < 2 {
        return 0;
    }
    let mut g = gso(basis);
    let mut swaps = 0usize;
    let mut k = 1usize;
    let mut guard = 0usize;
    let limit = 1000 * n * n + 10_000;
    while k < n && guard < limit {
        guard += 1;
        // Size reduction against every earlier vector. Subtracting a
        // multiple of `b_j` leaves every `b*_i` untouched, so the
        // Gram–Schmidt vectors and their norms survive and only the `μ`
        // row of `k` has to move — updated exactly, not refitted.
        for j in (0..k).rev() {
            let m = g.mu[k][j];
            if m.abs() > 0.5 {
                let q = m.round() as i64;
                if q != 0 {
                    let sub: Vec<i64> = basis[j].iter().map(|&x| q * x).collect();
                    for (slot, s) in basis[k].iter_mut().zip(sub.iter()) {
                        *slot -= s;
                    }
                    let qf = q as f64;
                    for i in 0..j {
                        g.mu[k][i] -= qf * g.mu[j][i];
                    }
                    g.mu[k][j] -= qf;
                }
            }
        }
        // Lovász condition.
        let lhs = g.norms[k];
        let rhs = (delta - g.mu[k][k - 1] * g.mu[k][k - 1]) * g.norms[k - 1];
        if lhs >= rhs {
            k += 1;
        } else {
            basis.swap(k, k - 1);
            g = gso(basis);
            swaps += 1;
            k = k.saturating_sub(1).max(1);
        }
    }
    swaps
}

/// Short vectors `v ∈ ℤ^d` with `⟨w_j, v⟩ ≡ 0 (mod q)` for every row of
/// `rows`, by LLL on the embedding described in the module docs.
///
/// Returns the head blocks of the reduced basis, deduplicated and
/// ordered by increasing norm. Near-misses are kept and not marked:
/// with the weight set correctly they are the expected output, since the
/// sampled `w` are rounded and a true lattice vector need not have a
/// zero residue. The caller's test — does this vector give a square root
/// of one? — is cheaper than the certainty would be.
pub fn congruence_kernel(rows: &[Vec<i64>], d: usize, q: i64, weight: i64) -> Vec<Vec<i64>> {
    let m = rows.len();
    let total = d + m;
    let mut basis: Vec<Vec<i64>> = Vec::with_capacity(total);
    for i in 0..d {
        let mut row = vec![0i64; total];
        row[i] = 1;
        for (j, w) in rows.iter().enumerate() {
            let entry = w.get(i).copied().unwrap_or(0).rem_euclid(q);
            row[d + j] = weight * entry;
        }
        basis.push(row);
    }
    for j in 0..m {
        let mut row = vec![0i64; total];
        row[d + j] = weight * q;
        basis.push(row);
    }
    lll(&mut basis, 0.75);

    let mut heads: Vec<Vec<i64>> = basis
        .iter()
        .map(|row| row[..d].to_vec())
        .filter(|head| head.iter().any(|&x| x != 0))
        .collect();
    heads.sort_by_key(|v| norm_sq(v));
    heads.dedup();
    heads
}

/// Small integer combinations of the first `take` basis vectors with
/// coefficients in `[-radius, radius]`, shortest first and deduplicated up
/// to sign. Regev's short vector is not always a basis vector, and at
/// these dimensions enumerating the neighbourhood is cheaper than
/// enumerating excuses.
pub fn short_combinations(basis: &[Vec<i64>], take: usize, radius: i64) -> Vec<Vec<i64>> {
    let take = take.min(basis.len());
    if take == 0 || radius < 1 {
        return Vec::new();
    }
    let dim = basis[0].len();
    let span = (2 * radius + 1) as usize;
    let total = span.saturating_pow(take as u32);
    let mut out: Vec<Vec<i64>> = Vec::new();
    for code in 0..total {
        let mut rest = code;
        let mut v = vec![0i64; dim];
        let mut all_zero = true;
        let mut first_sign = 0i64;
        for row in basis.iter().take(take) {
            let c = (rest % span) as i64 - radius;
            rest /= span;
            if c != 0 {
                all_zero = false;
                if first_sign == 0 {
                    first_sign = c.signum();
                }
                for (k, slot) in v.iter_mut().enumerate() {
                    *slot += c * row[k];
                }
            }
        }
        // Keep one of each ± pair.
        if all_zero || first_sign < 0 {
            continue;
        }
        out.push(v);
    }
    out.sort_by_key(|v| norm_sq(v));
    out.dedup();
    out
}
