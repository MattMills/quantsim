//! Integer lattices: what LLL guarantees, and what the congruence kernel
//! actually returns.

use quantsim::lattice::{congruence_kernel, dot, gso, lll, norm_sq, short_combinations};
use quantsim::Prng;

/// Determinant of a square integer basis by fraction-free elimination —
/// the invariant LLL must not change (up to sign).
fn det(basis: &[Vec<i64>]) -> i128 {
    let n = basis.len();
    let mut m: Vec<Vec<i128>> = basis
        .iter()
        .map(|r| r.iter().map(|&x| x as i128).collect())
        .collect();
    let mut sign = 1i128;
    let mut prev = 1i128;
    for k in 0..n {
        if m[k][k] == 0 {
            match (k + 1..n).find(|&i| m[i][k] != 0) {
                Some(i) => {
                    m.swap(i, k);
                    sign = -sign;
                }
                None => return 0,
            }
        }
        for i in k + 1..n {
            for j in k + 1..n {
                m[i][j] = (m[i][j] * m[k][k] - m[i][k] * m[k][j]) / prev;
            }
        }
        prev = m[k][k];
    }
    sign * m[n - 1][n - 1]
}

fn random_basis(n: usize, rng: &mut Prng, spread: i64) -> Vec<Vec<i64>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let r = (rng.next_u64() % (2 * spread as u64 + 1)) as i64 - spread;
                    if i == j {
                        r + spread + 1
                    } else {
                        r
                    }
                })
                .collect()
        })
        .collect()
}

#[test]
fn reduction_preserves_the_lattice() {
    let mut rng = Prng::new(9);
    for _ in 0..40 {
        for n in 2..=5usize {
            let basis = random_basis(n, &mut rng, 12);
            let before = det(&basis);
            if before == 0 {
                continue;
            }
            let mut reduced = basis.clone();
            lll(&mut reduced, 0.75);
            assert_eq!(before.abs(), det(&reduced).abs(), "determinant moved");
        }
    }
}

#[test]
fn reduction_satisfies_size_reduction_and_lovasz() {
    let mut rng = Prng::new(17);
    let delta = 0.75;
    for _ in 0..40 {
        for n in 2..=5usize {
            let mut basis = random_basis(n, &mut rng, 12);
            if det(&basis) == 0 {
                continue;
            }
            lll(&mut basis, delta);
            let g = gso(&basis);
            for k in 1..n {
                for j in 0..k {
                    assert!(
                        g.mu[k][j].abs() <= 0.5 + 1e-9,
                        "μ[{k}][{j}] = {} is not size reduced",
                        g.mu[k][j]
                    );
                }
                let rhs = (delta - g.mu[k][k - 1] * g.mu[k][k - 1]) * g.norms[k - 1];
                assert!(
                    g.norms[k] >= rhs - 1e-9,
                    "Lovász failed at {k}: {} < {rhs}",
                    g.norms[k]
                );
            }
        }
    }
}

#[test]
fn the_first_vector_meets_the_lll_bound() {
    // The theorem, not a hope: for δ = 3/4, ‖b₁‖² ≤ 2^{n−1} · det^{2/n}.
    let mut rng = Prng::new(23);
    let mut improved = 0;
    let mut total = 0;
    for _ in 0..60 {
        for n in 2..=5usize {
            let basis = random_basis(n, &mut rng, 20);
            let d = det(&basis);
            if d == 0 {
                continue;
            }
            let before = basis.iter().map(|v| norm_sq(v)).min().unwrap();
            let mut reduced = basis.clone();
            lll(&mut reduced, 0.75);
            let after = norm_sq(&reduced[0]);
            let bound = 2f64.powi(n as i32 - 1) * (d.unsigned_abs() as f64).powf(2.0 / n as f64);
            assert!(
                (after as f64) <= bound * (1.0 + 1e-9),
                "n={n}: ‖b₁‖² = {after} exceeds the LLL bound {bound}"
            );
            total += 1;
            if after < before {
                improved += 1;
            }
        }
    }
    // And in practice it does better than the bound: most inputs get a
    // strictly shorter first vector than anything they came in with.
    assert!(
        improved * 2 > total,
        "reduction shortened only {improved}/{total}"
    );
}

#[test]
fn the_congruence_kernel_respects_its_congruences_at_high_weight() {
    // With a large weight LLL is pushed to exact kernel vectors, and the
    // shortest ones must satisfy every congruence exactly.
    let q = 64i64;
    let rows = vec![vec![13i64, 51], vec![26, 38]];
    let out = congruence_kernel(&rows, 2, q, 4 * q);
    assert!(!out.is_empty());
    let exact = out
        .iter()
        .filter(|v| rows.iter().all(|w| dot(w, v).rem_euclid(q as i128) == 0))
        .count();
    assert!(exact > 0, "no exact kernel vector among {out:?}");
    // (1, 1) is in the kernel of both rows and is the shortest such.
    assert!(out
        .iter()
        .any(|v| (v[0].abs(), v[1].abs()) == (1, 1) && v[0] == v[1]));
}

#[test]
fn the_kernel_is_sorted_shortest_first_and_never_zero() {
    let q = 32i64;
    let rows = vec![vec![7i64, 11, 3], vec![5, 2, 30], vec![18, 18, 18]];
    let out = congruence_kernel(&rows, 3, q, 1);
    assert!(!out.is_empty());
    for v in &out {
        assert!(v.iter().any(|&x| x != 0), "a zero vector was returned");
    }
    for pair in out.windows(2) {
        assert!(
            norm_sq(&pair[0]) <= norm_sq(&pair[1]),
            "not sorted: {out:?}"
        );
    }
}

#[test]
fn a_large_weight_forces_the_trivial_kernel_when_the_samples_are_generic() {
    // The failure mode the weight exists to avoid: generic rows have no
    // short exact kernel vector, so demanding exactness returns q·e_i.
    let q = 64i64;
    let rows = vec![vec![13i64, 27], vec![41, 5], vec![7, 50]];
    let strict = congruence_kernel(&rows, 2, q, 4 * q);
    let shortest_strict = norm_sq(&strict[0]);
    let loose = congruence_kernel(&rows, 2, q, 1);
    let shortest_loose = norm_sq(&loose[0]);
    assert!(
        shortest_loose < shortest_strict,
        "weight made no difference: {shortest_loose} vs {shortest_strict}"
    );
}

#[test]
fn short_combinations_enumerates_without_duplicates_or_signs() {
    let basis = vec![vec![1i64, 0], vec![0, 1]];
    let out = short_combinations(&basis, 2, 1);
    // (1,0) (0,1) (1,1) (1,-1) — one of each ± pair, no zero.
    assert_eq!(out.len(), 4, "{out:?}");
    for v in &out {
        assert!(v.iter().any(|&x| x != 0));
    }
    let mut seen = out.clone();
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), out.len(), "duplicates in {out:?}");
    for pair in out.windows(2) {
        assert!(norm_sq(&pair[0]) <= norm_sq(&pair[1]));
    }
}

#[test]
fn gram_schmidt_is_orthogonal_and_reproduces_the_basis() {
    let mut rng = Prng::new(31);
    for _ in 0..20 {
        let basis = random_basis(4, &mut rng, 9);
        let g = gso(&basis);
        for (i, row) in basis.iter().enumerate() {
            for j in 0..i {
                let d: f64 = g.star[i]
                    .iter()
                    .zip(g.star[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(d.abs() < 1e-6, "b*[{i}]·b*[{j}] = {d}");
            }
            // b_i = b*_i + Σ_{j<i} μ_ij b*_j
            for (c, &entry) in row.iter().enumerate() {
                let mut v = g.star[i][c];
                for j in 0..i {
                    v += g.mu[i][j] * g.star[j][c];
                }
                assert!((v - entry as f64).abs() < 1e-6);
            }
        }
    }
}
