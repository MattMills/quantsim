//! Operator flows: a circuit block as a *one-parameter group* rather than
//! a single unitary, and what its width does in between.
//!
//! A block usually exists at one granularity — the whole unitary `U`.
//! This module treats it as the flow `t ↦ U^t`, the path through operator
//! space from the identity (`t = 0`) to the full operation (`t = 1`) and
//! past it. Fractional powers are exact in a diagonalizing Fourier frame:
//!
//! ```text
//!   U^t = F† · D(t) · F,      D(t) = diag(ω^{t·c·y/N}) linear in t
//! ```
//!
//! so the flow is a genuine one-parameter group, `U^s ∘ U^t = U^{s+t}`,
//! with no approximation anywhere.
//!
//! ## Operator width sees the integers
//!
//! For the ring translation this is the continuous shift `x → x + t·c`,
//! and its matrix is a Dirichlet kernel:
//!
//! ```text
//!   U_t[x', x] = (1/N) Σ_y ω^{y·(x − x' + t·c)/N}
//! ```
//!
//! At **integer** `t·c` that sum collapses to a delta: the operator is a
//! crisp permutation and narrow across any cut. At **fractional** `t·c`
//! nothing cancels, the kernel delocalizes over the whole ring, and the
//! operator Schmidt rank jumps. So the width of a partially-applied
//! operation is not a smooth interpolation between its endpoints — it
//! *detects arithmetic*, reading off exactly when the shift lands on a
//! lattice site.
//!
//! That makes width a probe rather than a cost: sweeping `t` and watching
//! the rank recovers which fractions of an operation are themselves
//! operations.
//!
//! Ported from the `novel_quantum_structures` research package
//! (`flow.rs`), with the group law and the rank behaviour measured rather
//! than assumed.

use crate::scalar::C64;
use std::f64::consts::TAU;

/// Largest ring size these routines will build a dense operator for.
/// Named so `grep 'pub const MAX'` finds every ceiling in the crate.
pub const MAX_FLOW_N: usize = 256;

/// Tolerance for counting a singular direction as present.
pub const RANK_TOL: f64 = 1e-9;

/// The flow `U^t` of the ring translation by `c` on `ℤ_N`, as a dense
/// matrix `u[row][col]` with `row = x'`, `col = x`.
///
/// Exact at every real `t`: this is `F† D(t) F` evaluated in closed form,
/// not a matrix power or an exponential series.
pub fn shift_power(n: usize, c: f64, t: f64) -> Option<Vec<Vec<C64>>> {
    if n > MAX_FLOW_N || n == 0 {
        return None;
    }
    let shift = t * c;
    let mut u = vec![vec![C64::new(0.0, 0.0); n]; n];
    for (xp, row) in u.iter_mut().enumerate() {
        for (x, cell) in row.iter_mut().enumerate() {
            let d = x as f64 - xp as f64 + shift;
            let mut acc = C64::new(0.0, 0.0);
            for y in 0..n {
                let ang = TAU * y as f64 * d / n as f64;
                acc += C64::new(ang.cos(), ang.sin());
            }
            *cell = acc / n as f64;
        }
    }
    Some(u)
}

/// Matrix product `a · b`.
pub fn compose(a: &[Vec<C64>], b: &[Vec<C64>]) -> Vec<Vec<C64>> {
    let n = a.len();
    let mut out = vec![vec![C64::new(0.0, 0.0); n]; n];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut acc = C64::new(0.0, 0.0);
            for k in 0..n {
                acc += a[i][k] * b[k][j];
            }
            *cell = acc;
        }
    }
    out
}

/// Largest entrywise deviation between two matrices.
pub fn deviation(a: &[Vec<C64>], b: &[Vec<C64>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| (x - y).norm()))
        .fold(0.0, f64::max)
}

/// The **operator Schmidt rank** of `u` across the cut `N = l·s`.
///
/// Reshapes `u[(a',b')][(a,b)]` to `R[(a',a)][(b',b)]` and eliminates.
/// This is the bond dimension the operator would need at that cut.
pub fn schmidt_rank(u: &[Vec<C64>], l: usize, s: usize) -> usize {
    let (rows, cols) = (l * l, s * s);
    let mut r = vec![vec![C64::new(0.0, 0.0); cols]; rows];
    for (xp, urow) in u.iter().enumerate() {
        let (ap, bp) = (xp / s, xp % s);
        for (x, &v) in urow.iter().enumerate() {
            let (a, b) = (x / s, x % s);
            r[ap * l + a][bp * s + b] = v;
        }
    }
    let mut rank = 0usize;
    let mut pivot = 0usize;
    for col in 0..cols {
        let Some(sel) = (pivot..rows).find(|&i| r[i][col].norm() > RANK_TOL) else {
            continue;
        };
        r.swap(pivot, sel);
        let p = r[pivot][col];
        for i in 0..rows {
            if i != pivot && r[i][col].norm() > RANK_TOL {
                let f = r[i][col] / p;
                let (lo, hi) = if i < pivot {
                    let (x, y) = r.split_at_mut(pivot);
                    (&mut x[i], &y[0])
                } else {
                    let (x, y) = r.split_at_mut(i);
                    (&mut y[0], &x[pivot])
                };
                for (dst, src) in lo[col..].iter_mut().zip(&hi[col..]) {
                    *dst -= f * src;
                }
            }
        }
        pivot += 1;
        rank += 1;
        if pivot == rows {
            break;
        }
    }
    rank
}

/// Sweep the flow and report `(t, rank)` across the cut `l | s`.
///
/// The curve is the point: it dips exactly where `t·c` is an integer.
pub fn width_curve(n: usize, c: f64, l: usize, s: usize, steps: usize) -> Vec<(f64, usize)> {
    (0..=steps)
        .filter_map(|k| {
            let t = k as f64 / steps as f64;
            let u = shift_power(n, c, t)?;
            Some((t, schmidt_rank(&u, l, s)))
        })
        .collect()
}
