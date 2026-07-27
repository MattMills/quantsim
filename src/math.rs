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
}
