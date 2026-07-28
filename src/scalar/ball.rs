//! Ball arithmetic: complex amplitudes carrying a certified resolution.
//!
//! A [`Ball`] is a midpoint–radius pair `(mid, rad)` representing *every*
//! complex number within distance `rad` of `mid` — the amplitude analogue
//! of a measurement with an error bar, and the crate's **coarse-grained
//! number type**: the value is known *at a resolution*, and every
//! arithmetic operation propagates that resolution soundly (the true
//! result of operating on the true inputs always lies inside the result
//! ball). Three sources feed the radius:
//!
//! * **rounding** — each operation inflates by a conservative multiple of
//!   machine epsilon (documented as conservative inflation, not IEEE
//!   directed rounding);
//! * **input coarseness** — radii combine by the triangle inequality
//!   (`|A·β| + |B·α| + |α·β|` for products), so coarse inputs yield
//!   honestly coarse outputs;
//! * **deliberate coarse-graining** — [`Ball::quantize`] snaps the
//!   midpoint to a dyadic grid of chosen spacing and *pays the snap
//!   distance into the radius*. This is the resolution dial: quantize
//!   aggressively and every downstream number still certifies its own
//!   error; recompute at a finer grid and the radii shrink toward the
//!   float floor. "Compute coarse first, refine later" becomes a sound
//!   operation on numbers, not a hope.
//!
//! The midpoint arithmetic is bit-identical to [`C64`] arithmetic, so a
//! `Simulator<Ball>` reproduces the complex simulator's physics exactly
//! while carrying certified error alongside — and the certification is
//! itself *tested*, not assumed: `tests/ball_certification.rs` runs
//! Clifford+T circuits over `Ball` and checks containment of the
//! [`exact`](crate::exact) ring values in every final amplitude ball, at
//! full resolution and under deliberately quantized (coarse) gates.
//!
//! Scope note: the radius is representation metadata, not an algebra
//! coordinate — [`Scalar::coeffs`] exposes the midpoint only, and sparse
//! backends' support pruning consults midpoints (a pruned entry's radius
//! is dropped with it). Certified end-to-end runs should use the dense
//! backend, which never discards amplitudes; per-representation certified
//! truncation is the [`mera`](crate::backend) backend's job, where
//! discarded weight is tracked explicitly.

use super::{Scalar, C64};

/// Conservative per-operation rounding inflation (a small multiple of
/// f64 machine epsilon; see the module docs).
const EPS_INFLATE: f64 = 8.0 * f64::EPSILON;

/// A complex midpoint with a certified error radius. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ball {
    /// Midpoint (the working value; arithmetic on midpoints is exactly
    /// `C64` arithmetic).
    pub mid: C64,
    /// Certified radius: the represented true value lies within `rad` of
    /// `mid`.
    pub rad: f64,
}

impl Ball {
    /// A ball of radius zero (exact float value).
    pub fn exactly(mid: C64) -> Self {
        Ball { mid, rad: 0.0 }
    }

    /// A ball with an explicit radius.
    pub fn with_radius(mid: C64, rad: f64) -> Self {
        Ball {
            mid,
            rad: rad.max(0.0),
        }
    }

    /// Whether `z` lies inside this ball (containment is the soundness
    /// relation the certification tests check).
    pub fn contains(&self, z: C64) -> bool {
        (z - self.mid).norm() <= self.rad
    }

    /// Coarse-grain to a grid of spacing `scale`: the midpoint snaps to
    /// the nearest grid point componentwise and the snap distance is paid
    /// into the radius, so the result still contains everything the
    /// input contained. `scale ≤ 0` is a no-op.
    pub fn quantize(self, scale: f64) -> Self {
        if scale <= 0.0 {
            return self;
        }
        let snap = |x: f64| (x / scale).round() * scale;
        let snapped = C64::new(snap(self.mid.re), snap(self.mid.im));
        let moved = (snapped - self.mid).norm();
        Ball {
            mid: snapped,
            rad: self.rad + moved + EPS_INFLATE * snapped.norm(),
        }
    }
}

impl std::ops::Add for Ball {
    type Output = Ball;
    fn add(self, rhs: Ball) -> Ball {
        let mid = self.mid + rhs.mid;
        Ball {
            mid,
            rad: self.rad + rhs.rad + EPS_INFLATE * mid.norm(),
        }
    }
}

impl std::ops::Sub for Ball {
    type Output = Ball;
    fn sub(self, rhs: Ball) -> Ball {
        let mid = self.mid - rhs.mid;
        Ball {
            mid,
            rad: self.rad + rhs.rad + EPS_INFLATE * mid.norm(),
        }
    }
}

impl std::ops::Neg for Ball {
    type Output = Ball;
    fn neg(self) -> Ball {
        Ball {
            mid: -self.mid,
            rad: self.rad,
        }
    }
}

impl std::ops::Mul for Ball {
    type Output = Ball;
    fn mul(self, rhs: Ball) -> Ball {
        let (a, b) = (self.mid.norm(), rhs.mid.norm());
        let mid = self.mid * rhs.mid;
        Ball {
            mid,
            rad: a * rhs.rad
                + b * self.rad
                + self.rad * rhs.rad
                + EPS_INFLATE * (a * b + mid.norm()),
        }
    }
}

impl Scalar for Ball {
    const DIM: usize = 2;
    const COMMUTATIVE: bool = true;
    const ASSOCIATIVE: bool = true;
    const DIVISION: bool = true;
    const TRIVIAL_CONJ: bool = false;

    fn algebra_name() -> String {
        "C (ball)".to_string()
    }

    fn zero() -> Self {
        Ball::exactly(C64::new(0.0, 0.0))
    }

    fn one() -> Self {
        Ball::exactly(C64::new(1.0, 0.0))
    }

    fn conj(self) -> Self {
        Ball {
            mid: self.mid.conj(),
            rad: self.rad,
        }
    }

    fn scale(self, k: f64) -> Self {
        let mid = self.mid * k;
        Ball {
            mid,
            rad: self.rad * k.abs() + EPS_INFLATE * mid.norm(),
        }
    }

    fn re(self) -> f64 {
        self.mid.re
    }

    fn abs_sqr(self) -> f64 {
        self.mid.norm_sqr()
    }

    fn try_from_c64(z: C64) -> Option<Self> {
        Some(Ball::exactly(z))
    }

    /// Midpoint coordinates only — the radius is resolution metadata,
    /// not an algebra coordinate (see the module docs).
    fn coeffs(self) -> Vec<f64> {
        vec![self.mid.re, self.mid.im]
    }

    fn from_coeffs(c: &[f64]) -> Self {
        assert_eq!(c.len(), 2, "Ball has 2 coordinates");
        Ball::exactly(C64::new(c[0], c[1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(re: f64, im: f64, rad: f64) -> Ball {
        Ball::with_radius(C64::new(re, im), rad)
    }

    #[test]
    fn midpoints_track_c64_exactly() {
        let (x, y) = (C64::new(0.73, -0.21), C64::new(-1.4, 0.9));
        let (bx, by) = (Ball::exactly(x), Ball::exactly(y));
        assert_eq!((bx + by).mid, x + y);
        assert_eq!((bx * by).mid, x * y);
        assert_eq!((bx - by).mid, x - y);
        assert_eq!(bx.conj().mid, x.conj());
        assert_eq!(bx.scale(1.7).mid, x * 1.7);
    }

    #[test]
    fn radii_are_sound_for_interval_endpoints() {
        // True values at the edge of each input ball must land inside the
        // output ball, for products and sums.
        let bx = b(0.6, -0.3, 1e-3);
        let by = b(-0.2, 0.8, 2e-3);
        let sum = bx + by;
        let prod = bx * by;
        for (dx, dy) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
            let tx = bx.mid + C64::new(dx * bx.rad, 0.0);
            let ty = by.mid + C64::new(0.0, dy * by.rad);
            assert!(sum.contains(tx + ty));
            assert!(prod.contains(tx * ty));
        }
    }

    #[test]
    fn quantize_pays_the_snap_into_the_radius() {
        let x = b(0.736_812, -0.214_9, 1e-6);
        let q = x.quantize(1.0 / 16.0);
        // Midpoint on the grid.
        assert_eq!(q.mid.re, (x.mid.re * 16.0).round() / 16.0);
        assert_eq!(q.mid.im, (x.mid.im * 16.0).round() / 16.0);
        // Everything the input contained is still contained.
        assert!(q.contains(x.mid));
        assert!(q.rad >= (q.mid - x.mid).norm() + x.rad - 1e-18);
        // Finer grids cost less radius.
        let fine = x.quantize(1.0 / 1024.0);
        assert!(fine.rad < q.rad);
        // Non-positive scale is a no-op.
        assert_eq!(x.quantize(0.0), x);
    }
}
