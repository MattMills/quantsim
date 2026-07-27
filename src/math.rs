//! Small dense matrices over a [`Scalar`] algebra, used as gate unitaries.

use crate::error::{Error, Result};
use crate::scalar::{Scalar, C64};

/// Shorthand constructor for a complex number.
#[inline]
pub fn c64(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

/// `e^{iθ}` as a complex number.
#[inline]
pub fn cis(theta: f64) -> C64 {
    C64::new(theta.cos(), theta.sin())
}

/// Matrix entries with squared magnitude below this are treated as structural
/// zeros when building sparse application plans.
pub const MATRIX_ZERO_TOL: f64 = 1e-30;

/// Default tolerance for unitarity validation.
pub const UNITARY_TOL: f64 = 1e-9;

/// A `dim × dim` row-major matrix over the amplitude algebra `S`.
///
/// `dim` is validated to be a power of two at construction, so a matrix
/// always corresponds to a gate on `dim.trailing_zeros()` qubits. Sub-index
/// convention (little-endian, matching the state backends): bit `b` of a
/// row/column index corresponds to `qubits[b]` of the gate application, so
/// for controlled gates built with [`GateMatrix::controlled`] the control is
/// `qubits[0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct GateMatrix<S: Scalar> {
    dim: usize,
    data: Vec<S>,
}

impl<S: Scalar> GateMatrix<S> {
    /// Zero matrix of the given power-of-two dimension.
    pub fn zeros(dim: usize) -> Result<Self> {
        if dim == 0 || !dim.is_power_of_two() {
            return Err(Error::NonPowerOfTwoDim(dim));
        }
        Ok(GateMatrix {
            dim,
            data: vec![S::zero(); dim * dim],
        })
    }

    /// Identity matrix of the given power-of-two dimension.
    pub fn identity(dim: usize) -> Result<Self> {
        let mut m = Self::zeros(dim)?;
        for i in 0..dim {
            m.data[i * dim + i] = S::one();
        }
        Ok(m)
    }

    /// Build from a row-major entry vector of length `dim * dim`.
    pub fn from_vec(dim: usize, data: Vec<S>) -> Result<Self> {
        if dim == 0 || !dim.is_power_of_two() {
            return Err(Error::NonPowerOfTwoDim(dim));
        }
        if data.len() != dim * dim {
            return Err(Error::BadDimension {
                expected: dim * dim,
                got: data.len(),
            });
        }
        Ok(GateMatrix { dim, data })
    }

    /// Convert a row-major complex matrix into this algebra entry-by-entry.
    /// Returns `None` if any entry does not embed (e.g. `i` over ℝ) — the
    /// registry uses this to decide which gates exist per algebra.
    ///
    /// # Panics
    /// Panics if `entries.len() != dim * dim` or `dim` is not a power of two
    /// (programmer error in a gate definition).
    pub fn try_from_c64s(dim: usize, entries: &[C64]) -> Option<Self> {
        assert!(
            dim.is_power_of_two(),
            "gate dimension must be a power of two"
        );
        assert_eq!(entries.len(), dim * dim, "entry count must be dim^2");
        let mut data = Vec::with_capacity(entries.len());
        for &z in entries {
            data.push(S::try_from_c64(z)?);
        }
        Some(GateMatrix { dim, data })
    }

    /// Matrix dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of qubits this matrix acts on (`log2(dim)`).
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.dim.trailing_zeros() as usize
    }

    /// Row-major entries.
    #[inline]
    pub fn data(&self) -> &[S] {
        &self.data
    }

    /// Entry at `(row, col)`.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> S {
        self.data[row * self.dim + col]
    }

    /// Set entry at `(row, col)`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: S) {
        self.data[row * self.dim + col] = value;
    }

    /// Matrix product `self · rhs` (entry products taken left-to-right,
    /// which matters for non-commutative scalars).
    ///
    /// # Panics
    /// Panics on dimension mismatch.
    pub fn matmul(&self, rhs: &Self) -> Self {
        assert_eq!(self.dim, rhs.dim, "matrix dimension mismatch");
        let d = self.dim;
        let mut out = vec![S::zero(); d * d];
        for r in 0..d {
            for k in 0..d {
                let a = self.data[r * d + k];
                if a.abs_sqr() <= MATRIX_ZERO_TOL {
                    continue;
                }
                for c in 0..d {
                    out[r * d + c] = out[r * d + c] + a * rhs.data[k * d + c];
                }
            }
        }
        GateMatrix { dim: d, data: out }
    }

    /// Conjugate transpose.
    pub fn dagger(&self) -> Self {
        let d = self.dim;
        let mut out = vec![S::zero(); d * d];
        for r in 0..d {
            for c in 0..d {
                out[c * d + r] = self.data[r * d + c].conj();
            }
        }
        GateMatrix { dim: d, data: out }
    }

    /// Controlled version of this gate. The new control occupies sub-index
    /// bit 0 (i.e. `qubits[0]` at application time); the original gate's
    /// qubits shift up by one.
    pub fn controlled(&self) -> Self {
        let d = self.dim;
        let nd = 2 * d;
        let mut out = vec![S::zero(); nd * nd];
        for r in 0..d {
            // Control clear: identity block.
            out[(2 * r) * nd + (2 * r)] = S::one();
            // Control set: the gate.
            for c in 0..d {
                out[(2 * r + 1) * nd + (2 * c + 1)] = self.data[r * d + c];
            }
        }
        GateMatrix { dim: nd, data: out }
    }

    /// Kronecker product `self ⊗ rhs`, with `rhs` occupying the **low**
    /// sub-index bits. A two-qubit gate `G` applied to `[q0, q1]` therefore
    /// decomposes as `G = high.kron(&low)` with `low` acting on `q0`.
    pub fn kron(&self, rhs: &Self) -> Self {
        let (da, db) = (self.dim, rhs.dim);
        let nd = da * db;
        let mut out = vec![S::zero(); nd * nd];
        for ra in 0..da {
            for ca in 0..da {
                let a = self.data[ra * da + ca];
                if a.abs_sqr() <= MATRIX_ZERO_TOL {
                    continue;
                }
                for rb in 0..db {
                    for cb in 0..db {
                        out[(ra * db + rb) * nd + (ca * db + cb)] = a * rhs.data[rb * db + cb];
                    }
                }
            }
        }
        GateMatrix { dim: nd, data: out }
    }

    /// Maximum entry magnitude of `self† · self − I`.
    ///
    /// For associative scalars a small deviation guarantees the gate
    /// preserves total Born weight; for non-associative algebras (octonions
    /// onward) that implication can fail even at deviation 0.
    pub fn unitarity_deviation(&self) -> f64 {
        let prod = self.dagger().matmul(self);
        let d = self.dim;
        let mut worst: f64 = 0.0;
        for r in 0..d {
            for c in 0..d {
                let expect = if r == c { S::one() } else { S::zero() };
                worst = worst.max((prod.data[r * d + c] - expect).abs_sqr().sqrt());
            }
        }
        worst
    }

    /// Whether `self† · self ≈ I` within `tol`.
    pub fn is_unitary(&self, tol: f64) -> bool {
        self.unitarity_deviation() <= tol
    }

    /// Entrywise approximate equality.
    pub fn approx_eq(&self, rhs: &Self, tol: f64) -> bool {
        self.dim == rhs.dim
            && self
                .data
                .iter()
                .zip(rhs.data.iter())
                .all(|(&a, &b)| a.approx_eq(b, tol))
    }
}

/// A thin singular value decomposition `A ≈ U · diag(σ) · V†` with rank `r`.
///
/// Produced by [`jacobi_svd`]. `u` is `rows × r` row-major with orthonormal
/// columns, `sigma` holds `r` singular values in descending order, `vt` is
/// `r × cols` row-major with orthonormal rows.
#[derive(Debug, Clone)]
pub struct Svd<S: Scalar> {
    /// Left singular vectors, `rows × rank`, row-major.
    pub u: Vec<S>,
    /// Singular values, descending.
    pub sigma: Vec<f64>,
    /// Conjugate-transposed right singular vectors, `rank × cols`, row-major.
    pub vt: Vec<S>,
    /// Numerical rank retained.
    pub rank: usize,
}

/// One-sided Jacobi SVD of a `rows × cols` row-major matrix over a
/// **commutative division algebra** (ℝ, ℂ — the scalars with the complex
/// arithmetic the rotations rely on). Dependency-free, numerically robust
/// for the small matrices that appear in tensor-network splits.
///
/// Columns with singular value ≤ `max(abs_floor, σ_max · rel_tol)` are
/// dropped; at least one column is always kept. Rotations repeat until the
/// normalized off-diagonal Gram mass falls below 1e-28 (or 100 sweeps).
pub fn jacobi_svd<S: Scalar>(
    rows: usize,
    cols: usize,
    a: &[S],
    rel_tol: f64,
    abs_floor: f64,
) -> Result<Svd<S>> {
    if !(S::COMMUTATIVE && S::DIVISION) {
        return Err(Error::InvalidState(format!(
            "jacobi_svd requires a commutative division algebra; {} is not",
            S::algebra_name()
        )));
    }
    if a.len() != rows * cols || rows == 0 || cols == 0 {
        return Err(Error::BadDimension {
            expected: rows * cols,
            got: a.len(),
        });
    }
    // Work column-major internally: col j = work[j*rows..(j+1)*rows].
    let mut work = vec![S::zero(); rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            work[c * rows + r] = a[r * cols + c];
        }
    }
    // V accumulates right rotations, column-major cols × cols identity.
    let mut v = vec![S::zero(); cols * cols];
    for j in 0..cols {
        v[j * cols + j] = S::one();
    }

    for _sweep in 0..100 {
        let mut off = 0.0f64;
        for i in 0..cols {
            for j in (i + 1)..cols {
                let (ci, cj) = (i * rows, j * rows);
                let mut aii = 0.0f64;
                let mut ajj = 0.0f64;
                let mut aij = S::zero();
                for k in 0..rows {
                    aii += work[ci + k].abs_sqr();
                    ajj += work[cj + k].abs_sqr();
                    aij = aij + work[ci + k].conj() * work[cj + k];
                }
                let mag = aij.abs_sqr().sqrt();
                if aii <= 0.0 || ajj <= 0.0 || mag * mag <= 1e-32 * aii * ajj {
                    continue;
                }
                off += (mag * mag) / (aii * ajj);
                // Diagonalize [[aii, aij], [conj(aij), ajj]] with the
                // rotation R = [[cosθ, −û sinθ], [conj(û) sinθ, cosθ]],
                // û = aij/|aij|, tan(2θ) = 2|aij|/(aii − ajj).
                let unit = aij.scale(1.0 / mag);
                let tau = (aii - ajj) / (2.0 * mag);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cs = 1.0 / (1.0 + t * t).sqrt();
                let sn = cs * t;
                let u_sn = unit.scale(sn);
                let u_conj_sn = unit.conj().scale(sn);
                for k in 0..rows {
                    let x = work[ci + k];
                    let y = work[cj + k];
                    work[ci + k] = x.scale(cs) + y * u_conj_sn;
                    work[cj + k] = y.scale(cs) - x * u_sn;
                }
                for k in 0..cols {
                    let x = v[i * cols + k];
                    let y = v[j * cols + k];
                    v[i * cols + k] = x.scale(cs) + y * u_conj_sn;
                    v[j * cols + k] = y.scale(cs) - x * u_sn;
                }
            }
        }
        if off < 1e-28 {
            break;
        }
    }

    // Singular values, sorted descending.
    let mut order: Vec<usize> = (0..cols).collect();
    let norms: Vec<f64> = (0..cols)
        .map(|j| {
            (0..rows)
                .map(|k| work[j * rows + k].abs_sqr())
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    order.sort_by(|&x, &y| norms[y].total_cmp(&norms[x]));
    let sigma_max = norms[order[0]];
    let cutoff = abs_floor.max(sigma_max * rel_tol);
    let rank = order.iter().filter(|&&j| norms[j] > cutoff).count().max(1);

    let mut u = vec![S::zero(); rows * rank];
    let mut sigma = Vec::with_capacity(rank);
    let mut vt = vec![S::zero(); rank * cols];
    for (slot, &j) in order.iter().take(rank).enumerate() {
        let s = norms[j];
        sigma.push(s);
        if s > 0.0 {
            let inv = 1.0 / s;
            for k in 0..rows {
                u[k * rank + slot] = work[j * rows + k].scale(inv);
            }
        }
        for k in 0..cols {
            vt[slot * cols + k] = v[j * cols + k].conj();
        }
    }
    Ok(Svd { u, sigma, vt, rank })
}

/// [`jacobi_svd`] with the cheaper orientation chosen automatically: when
/// `cols > rows` the decomposition runs on `A†` (Jacobi cost grows with the
/// column count) and the factors are mapped back. Same output contract.
pub fn svd_thin<S: Scalar>(
    rows: usize,
    cols: usize,
    a: &[S],
    rel_tol: f64,
    abs_floor: f64,
) -> Result<Svd<S>> {
    if cols <= rows {
        return jacobi_svd(rows, cols, a, rel_tol, abs_floor);
    }
    // B = A† (cols × rows). A = UΣV† ⇒ B = VΣU†, so svd(B).u = V and
    // svd(B).vt = U†; map back.
    let mut b = vec![S::zero(); cols * rows];
    for r in 0..rows {
        for c in 0..cols {
            b[c * rows + r] = a[r * cols + c].conj();
        }
    }
    let svd_b = jacobi_svd(cols, rows, &b, rel_tol, abs_floor)?;
    let rank = svd_b.rank;
    let mut u = vec![S::zero(); rows * rank];
    let mut vt = vec![S::zero(); rank * cols];
    for r in 0..rows {
        for k in 0..rank {
            u[r * rank + k] = svd_b.vt[k * rows + r].conj();
        }
    }
    for k in 0..rank {
        for c in 0..cols {
            vt[k * cols + c] = svd_b.u[c * rank + k].conj();
        }
    }
    Ok(Svd {
        u,
        sigma: svd_b.sigma,
        vt,
        rank,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_1_SQRT_2;

    fn h() -> GateMatrix<C64> {
        let s = FRAC_1_SQRT_2;
        GateMatrix::try_from_c64s(2, &[c64(s, 0.0), c64(s, 0.0), c64(s, 0.0), c64(-s, 0.0)])
            .unwrap()
    }

    fn x() -> GateMatrix<C64> {
        GateMatrix::try_from_c64s(
            2,
            &[c64(0.0, 0.0), c64(1.0, 0.0), c64(1.0, 0.0), c64(0.0, 0.0)],
        )
        .unwrap()
    }

    #[test]
    fn identity_and_dims() {
        let i4 = GateMatrix::<C64>::identity(4).unwrap();
        assert_eq!(i4.num_qubits(), 2);
        assert!(i4.is_unitary(1e-12));
        assert!(GateMatrix::<C64>::zeros(3).is_err());
        assert!(GateMatrix::<C64>::zeros(0).is_err());
    }

    #[test]
    fn h_squared_is_identity() {
        let hh = h().matmul(&h());
        assert!(hh.approx_eq(&GateMatrix::identity(2).unwrap(), 1e-12));
    }

    #[test]
    fn hxh_is_z() {
        let z = GateMatrix::try_from_c64s(
            2,
            &[c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(-1.0, 0.0)],
        )
        .unwrap();
        assert!(h().matmul(&x()).matmul(&h()).approx_eq(&z, 1e-12));
    }

    #[test]
    fn controlled_x_matches_little_endian_cx() {
        // Control = sub-bit 0, target = sub-bit 1: columns map
        // |c t⟩: 00→00, 01(c=1)→11, 10→10, 11→01.
        let cx = x().controlled();
        let expect = GateMatrix::from_vec(
            4,
            vec![
                c64(1.0, 0.0),
                c64(0.0, 0.0),
                c64(0.0, 0.0),
                c64(0.0, 0.0),
                c64(0.0, 0.0),
                c64(0.0, 0.0),
                c64(0.0, 0.0),
                c64(1.0, 0.0),
                c64(0.0, 0.0),
                c64(0.0, 0.0),
                c64(1.0, 0.0),
                c64(0.0, 0.0),
                c64(0.0, 0.0),
                c64(1.0, 0.0),
                c64(0.0, 0.0),
                c64(0.0, 0.0),
            ],
        )
        .unwrap();
        assert!(cx.approx_eq(&expect, 1e-12));
    }

    #[test]
    fn dagger_inverts_unitaries() {
        let m = h().matmul(&x());
        let prod = m.matmul(&m.dagger());
        assert!(prod.approx_eq(&GateMatrix::identity(2).unwrap(), 1e-12));
    }

    #[test]
    fn kron_dimensions_and_entries() {
        let k = x().kron(&h());
        assert_eq!(k.dim(), 4);
        // (X ⊗ H)[0, 2] = X[0,1] * H[0,0]
        assert!(k.get(0, 2).approx_eq(c64(FRAC_1_SQRT_2, 0.0), 1e-12));
        assert!(k.get(0, 0).approx_eq(c64(0.0, 0.0), 1e-12));
    }

    #[test]
    fn real_algebra_rejects_complex_entries() {
        let s = GateMatrix::<f64>::try_from_c64s(
            2,
            &[c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(0.0, 1.0)],
        );
        assert!(s.is_none());
        let hr = GateMatrix::<f64>::try_from_c64s(
            2,
            &[
                c64(FRAC_1_SQRT_2, 0.0),
                c64(FRAC_1_SQRT_2, 0.0),
                c64(FRAC_1_SQRT_2, 0.0),
                c64(-FRAC_1_SQRT_2, 0.0),
            ],
        );
        assert!(hr.is_some());
    }

    fn svd_reconstruction_error<S: Scalar>(rows: usize, cols: usize, a: &[S]) -> f64 {
        let svd = jacobi_svd(rows, cols, a, 1e-14, 1e-14).unwrap();
        let mut worst = 0.0f64;
        for r in 0..rows {
            for c in 0..cols {
                let mut acc = S::zero();
                for k in 0..svd.rank {
                    acc =
                        acc + (svd.u[r * svd.rank + k] * svd.vt[k * cols + c]).scale(svd.sigma[k]);
                }
                worst = worst.max((acc - a[r * cols + c]).abs_sqr().sqrt());
            }
        }
        // U†U = I and V V† = I on the retained rank.
        for i in 0..svd.rank {
            for j in 0..svd.rank {
                let mut uu = S::zero();
                let mut vv = S::zero();
                for r in 0..rows {
                    uu = uu + svd.u[r * svd.rank + i].conj() * svd.u[r * svd.rank + j];
                }
                for c in 0..cols {
                    vv = vv + svd.vt[i * cols + c] * svd.vt[j * cols + c].conj();
                }
                let expect = if i == j { S::one() } else { S::zero() };
                worst = worst.max((uu - expect).abs_sqr().sqrt());
                worst = worst.max((vv - expect).abs_sqr().sqrt());
            }
        }
        // Descending singular values.
        assert!(svd.sigma.windows(2).all(|w| w[0] >= w[1] - 1e-12));
        worst
    }

    #[test]
    fn jacobi_svd_reconstructs_random_complex_matrices() {
        let mut rng = crate::rng::Prng::new(0x5D);
        for &(rows, cols) in &[(1usize, 1usize), (3, 2), (2, 3), (6, 6), (8, 5), (4, 12)] {
            let a: Vec<C64> = (0..rows * cols)
                .map(|_| c64(rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0))
                .collect();
            let err = svd_reconstruction_error(rows, cols, &a);
            assert!(err < 1e-10, "{rows}x{cols}: reconstruction error {err}");
        }
        // Real algebra too.
        let a: Vec<f64> = (0..30).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        let err = svd_reconstruction_error(5, 6, &a);
        assert!(err < 1e-10, "real 5x6: {err}");
    }

    #[test]
    fn jacobi_svd_detects_low_rank() {
        // Outer product = rank 1 exactly.
        let u = [c64(0.6, 0.3), c64(-0.2, 0.7), c64(0.1, -0.4)];
        let v = [c64(0.9, -0.1), c64(0.3, 0.5)];
        let mut a = vec![c64(0.0, 0.0); 6];
        for r in 0..3 {
            for c in 0..2 {
                a[r * 2 + c] = u[r] * v[c];
            }
        }
        let svd = jacobi_svd(3, 2, &a, 1e-10, 1e-12).unwrap();
        assert_eq!(svd.rank, 1);
        assert!(svd_reconstruction_error(3, 2, &a) < 1e-10);
    }

    #[test]
    fn jacobi_svd_rejects_noncommutative_algebras() {
        use crate::scalar::Quaternion;
        let a = vec![Quaternion::one(); 4];
        assert!(jacobi_svd(2, 2, &a, 1e-12, 1e-12).is_err());
    }
}
