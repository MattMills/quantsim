//! **The surface between volumes, and a horizontal competition to
//! describe it.**
//!
//! [`crate::sweep`] contracts a circuit one world-line at a time. What
//! passes from the processed part to the rest is a
//! [`Surface`](crate::sweep::Surface): the open legs on one bond, one
//! leg per `CZ`, indexed in circuit order. On the 70-qubit experiment
//! that is 35 legs, so the sweep's peak is `2^36` rather than the
//! register's `2^70`.
//!
//! `2^35` is the *raw* size of the surface — how many indices it has,
//! not how many numbers it takes to write down. Those are different,
//! and this module measures the second one.
//!
//! ## Two geometric modes
//!
//! The sweep walks **space**: qubit `0`, then `1`, then `2`. The
//! surface it carries is indexed by **time**: leg `p` is the `p`-th
//! `CZ` on that bond. So the object in flight is a register in the time
//! direction being transported along the space direction, and the
//! chain of surfaces is a register whose sites are bonds. What each
//! site costs is what decides whether volumes compose — a volume of
//! world-lines hands the next volume however many numbers its surface
//! takes, and the volumes chain at that width no matter how many of
//! them there are.
//!
//! ## Why a competition, and not a decomposition
//!
//! A surface has `m` legs. A singular value decomposition can only ever
//! see it through a **bipartition**: legs `0..j` against the rest, one
//! two-way flattening at a time. That is a real invariant, but it is
//! not the tensor's irreducibility — it is the irreducibility of one
//! shadow of it. A description can be short for reasons no bipartition
//! records: the amplitudes can be flat on an affine subspace with the
//! whole content in a phase polynomial, or sparse in the character
//! basis rather than the computational one, or a short sum of rank-one
//! terms over all `m` legs at once, which is a genuinely `m`-way
//! statement that no sequence of two-way cuts reconstructs.
//!
//! So every technique here runs on the same surface and is scored on
//! **two** axes, because collapsing them to one is what hides the real
//! trade:
//!
//! * `scalars` — numbers that must cross the bond. A stored index
//!   counts as one and a stored amplitude counts as one, since a
//!   technique that hides its bookkeeping in indices has not made the
//!   surface smaller.
//! * `read_cost` — scalar operations to get **one** entry back out.
//!
//! A description can be short to write and dear to read. The path sum
//! is exactly that, and a single score would either crown it or
//! disqualify it rather than showing what it is.
//!
//! | technique | what it bets on | scalars | read |
//! |---|---|---|---|
//! | [`Technique::Dense`] | nothing | `2^m` | `1` |
//! | [`Technique::Support`] | the surface is sparse | `2·nnz` | `1` |
//! | [`Technique::Walsh`] | it is sparse in the *character* basis | `2·nnz(Ŵ)` | `nnz(Ŵ)` |
//! | [`Technique::Mps`] | early legs barely signal late ones | `Σ 2·χ_p·χ_{p+1}` | `2mχ²` |
//! | [`Technique::WalshMps`] | the same, in the dual basis | as above | `2^m·2mχ²` |
//! | [`Technique::PhasePoly`] | affine support, polynomial phase | `2 + k + 2·monomials` | `monomials` |
//! | [`Technique::Cp`] | short `m`-way rank-one sum | `2·R·m` | `R·m` |
//! | [`Technique::PathSum`] | the *volume*, not the surface | `2·monomials` | `2^{h*}·monomials` |
//!
//! Techniques that cannot represent a given surface say so
//! ([`Decomposition::applicable`]) rather than reporting a large
//! number, deviations are measured by reading entries back rather than
//! inferred from discarded weight, and
//! [`checked`](Decomposition::checked) records how many entries that
//! measurement actually covered — `0` marks a claim rather than a
//! result.
//!
//! ## Arity
//!
//! Every one of those techniques is **bipolar**, and so is the price of
//! a volume: [`crate::sweep::plan_volumes`] charges an independent
//! volume for two surfaces. Arity two is a choice. Two functions ask
//! what higher arity finds, and they answer differently:
//!
//! * [`grain_scan`] groups the legs into digits of arity `2^b` and
//!   reads the surface in that axis's own `ℤ_d` characters. On DCS the
//!   answer is nothing — occupancy stays at 98–100% from arity two up to
//!   arity `2^m`, at every bond.
//! * [`phase_arity`] asks the same question of the **phase** group
//!   instead of the harmonic group, and there it pays. A Clifford
//!   surface's phases never leave `ℤ/4`, at any bond, at any depth, and
//!   the affine description writes such a surface in `O(m²)` numbers
//!   while the bipartition reports a bond dimension and calls it
//!   irreducible. Doping is what ends it, at a measured rate: three `T`
//!   gates behind a bond are already enough.

use std::collections::HashMap;
use std::f64::consts::TAU;

use crate::error::{Error, Result};
use crate::math::svd_thin;
use crate::rng::Prng;
use crate::scalar::{Scalar, C64};
use crate::sweep::Surface;

/// Bond dimension past which the matrix product chain is abandoned. At
/// `χ = 4096` its own factors cost more than the surface it replaces,
/// so continuing would only spend time to confirm failure.
pub const MAX_CHI: usize = 4096;

/// Default weight discarded per bond by the truncating techniques.
pub const DEFAULT_EPSILON: f64 = 1e-10;

/// Amplitude below which a surface entry counts as absent.
pub const ZERO_TOLERANCE: f64 = 1e-12;

/// Wires past which [`symbolic`] refuses to build the volume behind a
/// bond.
///
/// Not a property of the path sum, which has no width ceiling — a
/// property of how its budget is enforced. [`crate::pathsum`] observes
/// an armed deadline between reduction sweeps, and on a volume this
/// wide a *single* sweep over the grown polynomial can outlast any
/// budget set around it. Measured on DCS at `n = 20`: the volume behind
/// bond 7 (18 wires) reduces in seconds, and the one behind bond 9 (20
/// wires) had not returned in two minutes under an eight-second budget.
/// So the ceiling is stated and refused at, rather than discovered by
/// waiting. Raise it deliberately with [`symbolic_within`].
pub const MAX_VOLUME_WIRES: usize = 18;

// ---------------------------------------------------------------------
// The matrix product chain — one entrant, in its own basis.
// ---------------------------------------------------------------------

/// How much of a surface is occupied, cut by cut, in time order.
#[derive(Debug, Clone)]
pub struct SurfaceCode {
    /// The bond this surface crosses: between `after_qubit` and the
    /// qubit above it.
    pub after_qubit: usize,
    /// Legs on the bond — the surface's raw index count is `2^legs`.
    pub legs: usize,
    /// Numerical rank across each time-ordered cut, `exact[j]`
    /// separating legs `0..=j` from the rest.
    pub exact: Vec<usize>,
    /// Rank retaining all but `epsilon` of the squared weight.
    pub truncated: Vec<usize>,
    /// Entanglement entropy across each cut, in bits.
    pub entropy: Vec<f64>,
    /// Set when the chain stopped at [`MAX_CHI`]; the vectors then
    /// cover only the cuts reached.
    pub ceiling_hit: bool,
}

impl SurfaceCode {
    /// Widest exact cut — the dimension a volume must hand on if the
    /// bipartite description is the one used.
    pub fn code_dim(&self) -> usize {
        self.exact.iter().copied().max().unwrap_or(1)
    }

    /// Widest cut after discarding `epsilon` per bond.
    pub fn truncated_dim(&self) -> usize {
        self.truncated.iter().copied().max().unwrap_or(1)
    }

    /// `2^legs` — what the surface costs held flat.
    pub fn raw_dim(&self) -> u128 {
        1u128 << self.legs.min(127)
    }

    /// Largest cut entropy, in bits: `legs/2` for a surface maximally
    /// entangled in time, `0` for a product one.
    pub fn peak_entropy(&self) -> f64 {
        self.entropy.iter().copied().fold(0.0, f64::max)
    }
}

/// The chain itself, kept so the surface can be rebuilt and the
/// deviation measured rather than bounded.
struct Chain {
    /// Site `p` as `χ_p × 2 × χ_{p+1}`, row-major.
    sites: Vec<Vec<C64>>,
    dims: Vec<usize>,
    code: SurfaceCode,
}

impl Chain {
    fn scalars(&self) -> u128 {
        self.sites.iter().map(|s| s.len() as u128).sum()
    }

    fn rebuild(&self, m: usize) -> Vec<C64> {
        // Left to right: `acc` is indexed by (legs so far) × bond.
        let mut acc = vec![C64::new(1.0, 0.0)];
        let mut chi = 1usize;
        for (p, site) in self.sites.iter().enumerate() {
            let next = self.dims[p + 1];
            let mut out = vec![C64::zero(); acc.len() / chi * 2 * next];
            let blocks = acc.len() / chi;
            for blk in 0..blocks {
                for b in 0..2 {
                    for beta in 0..next {
                        let mut z = C64::zero();
                        for alpha in 0..chi {
                            z += acc[blk * chi + alpha] * site[(alpha * 2 + b) * next + beta];
                        }
                        out[((b << p) + blk) * next + beta] = z;
                    }
                }
            }
            acc = out;
            chi = next;
        }
        debug_assert_eq!(chi, 1);
        debug_assert_eq!(acc.len(), 1usize << m);
        acc
    }
}

/// Split a surface into a matrix product over its time-ordered legs.
///
/// `epsilon` is the squared weight discarded per bond; when `truncate`
/// is false the chain keeps full numerical rank and the reported
/// truncated dimensions are advisory only.
fn decompose(surface: &Surface, epsilon: f64, truncate: bool) -> Result<Chain> {
    let m = surface.legs.len();
    if surface.amps.len() != 1usize << m {
        return Err(Error::InvalidState(format!(
            "surface: {} amplitudes for {m} legs; expected {}",
            surface.amps.len(),
            1usize << m
        )));
    }
    let mut code = SurfaceCode {
        after_qubit: surface.after_qubit,
        legs: m,
        exact: Vec::new(),
        truncated: Vec::new(),
        entropy: Vec::new(),
        ceiling_hit: false,
    };
    let mut chain = Chain {
        sites: Vec::new(),
        dims: vec![1],
        code: code.clone(),
    };
    if m == 0 {
        chain.code = code;
        return Ok(chain);
    }

    // `mat` is χ × 2^{m-p}: rows the bond carried in from the legs
    // already split off, columns the legs still to come.
    let mut mat = surface.amps.clone();
    let mut chi = 1usize;
    for p in 0..(m - 1) {
        crate::guard::checkpoint()?;
        let rest = 1usize << (m - p - 1);
        let rows = chi * 2;
        let mut a = vec![C64::zero(); rows * rest];
        for alpha in 0..chi {
            for y in 0..(rest * 2) {
                a[(alpha * 2 + (y & 1)) * rest + (y >> 1)] = mat[alpha * (rest * 2) + y];
            }
        }
        let svd = svd_thin(rows, rest, &a, 1e-13, 1e-300)?;
        let total: f64 = svd.sigma.iter().map(|s| s * s).sum();

        let mut trunc = svd.rank;
        let mut kept = 0.0;
        for (i, s) in svd.sigma.iter().enumerate() {
            kept += s * s;
            if total > 0.0 && kept >= (1.0 - epsilon) * total {
                trunc = i + 1;
                break;
            }
        }
        code.exact.push(svd.rank);
        code.truncated.push(trunc);
        code.entropy.push(if total > 0.0 {
            -svd.sigma
                .iter()
                .map(|s| s * s / total)
                .filter(|p| *p > 0.0)
                .map(|p| p * p.log2())
                .sum::<f64>()
        } else {
            0.0
        });

        if svd.rank > MAX_CHI {
            code.ceiling_hit = true;
            chain.code = code;
            return Ok(chain);
        }

        let keep = if truncate { trunc } else { svd.rank };
        let mut site = vec![C64::zero(); rows * keep];
        for r in 0..rows {
            for k in 0..keep {
                site[r * keep + k] = svd.u[r * svd.rank + k];
            }
        }
        chain.sites.push(site);
        chain.dims.push(keep);

        mat = vec![C64::zero(); keep * rest];
        for k in 0..keep {
            for c in 0..rest {
                mat[k * rest + c] = svd.vt[k * rest + c] * C64::new(svd.sigma[k], 0.0);
            }
        }
        chi = keep;
    }
    // The remainder is the last site: χ × 2 with a trivial right bond.
    chain.sites.push(mat);
    chain.dims.push(1);
    chain.code = code;
    Ok(chain)
}

/// Decompose a surface along its own (time) direction and report the
/// width of every cut.
pub fn code(surface: &Surface, epsilon: f64) -> Result<SurfaceCode> {
    Ok(decompose(surface, epsilon, false)?.code)
}

/// [`code`] for every bond of a circuit at one measured outcome.
pub fn census(
    circuit: &crate::circuit::Circuit<C64>,
    bits: u64,
    epsilon: f64,
) -> Result<Vec<SurfaceCode>> {
    crate::sweep::surfaces(circuit, bits)?
        .iter()
        .map(|s| code(s, epsilon))
        .collect()
}

// ---------------------------------------------------------------------
// The phase polynomial — affine support, and everything else in a phase.
// ---------------------------------------------------------------------

/// A surface written as a flat amplitude on an affine subspace times a
/// multilinear phase polynomial over `ℤ/denominator`.
///
/// This is what a Clifford+`T` computation produces natively, and it is
/// invisible to a bipartition: such a surface can have full rank across
/// every cut and still be `O(k³)` numbers.
#[derive(Debug, Clone)]
pub struct PhasePoly {
    /// Offset of the affine support.
    pub offset: u64,
    /// Basis of the support's linear part, in reduced echelon form.
    pub basis: Vec<u64>,
    /// Pivot bit of each basis vector.
    pub pivots: Vec<usize>,
    /// The amplitude at the offset. Every occupied entry is this
    /// number times a root of unity, so its phase is part of the
    /// description and not just its modulus.
    pub reference: C64,
    /// Denominator of the phase group: phases are `ω^t`, `ω = e^{2πi/D}`.
    pub denominator: i64,
    /// Nonzero monomials as `(subset of coordinates, coefficient)`.
    pub monomials: Vec<(u64, i64)>,
    /// Highest monomial degree present.
    pub degree: usize,
}

impl PhasePoly {
    /// Coordinates, offset, magnitude and monomials, counted alike.
    pub fn scalars(&self) -> u128 {
        2 + self.basis.len() as u128 + 2 * self.monomials.len() as u128
    }

    /// Rebuild the surface this describes.
    pub fn rebuild(&self, m: usize) -> Vec<C64> {
        let k = self.basis.len();
        let mut out = vec![C64::zero(); 1usize << m];
        // Zeta transform: values from monomials.
        let mut t = vec![0i64; 1usize << k];
        for (s, c) in &self.monomials {
            t[*s as usize] = *c;
        }
        for i in 0..k {
            for s in 0..(1usize << k) {
                if (s >> i) & 1 == 1 {
                    t[s] = (t[s] + t[s ^ (1 << i)]).rem_euclid(self.denominator);
                }
            }
        }
        for (s, ts) in t.iter().enumerate() {
            let mut x = self.offset;
            for (i, b) in self.basis.iter().enumerate() {
                if (s >> i) & 1 == 1 {
                    x ^= b;
                }
            }
            let theta = TAU * (*ts as f64) / (self.denominator as f64);
            out[x as usize] = self.reference * C64::new(theta.cos(), theta.sin());
        }
        out
    }
}

/// Fit a [`PhasePoly`] exactly, or report why the surface is not one.
///
/// Fails — returning `Ok(None)` — when the support is not an affine
/// subspace, the magnitudes are not flat, or the phases do not lie in
/// any `ℤ/2^j` up to `2^20`. No approximate fit is ever returned.
pub fn phase_poly(amps: &[C64], m: usize) -> Result<Option<PhasePoly>> {
    let support: Vec<u64> = amps
        .iter()
        .enumerate()
        .filter(|(_, z)| z.norm() > ZERO_TOLERANCE)
        .map(|(x, _)| x as u64)
        .collect();
    if support.is_empty() {
        return Ok(None);
    }
    let offset = support[0];
    let reference = amps[offset as usize];
    let magnitude = reference.norm();

    // Reduced echelon basis of {x ⊕ offset}.
    let mut basis: Vec<u64> = Vec::new();
    let mut pivots: Vec<usize> = Vec::new();
    for &x in &support {
        let mut v = x ^ offset;
        for (i, b) in basis.iter().enumerate() {
            if (v >> pivots[i]) & 1 == 1 {
                v ^= *b;
            }
        }
        if v == 0 {
            continue;
        }
        let p = 63 - v.leading_zeros() as usize;
        for (i, b) in basis.iter_mut().enumerate() {
            let _ = i;
            if (*b >> p) & 1 == 1 {
                *b ^= v;
            }
        }
        basis.push(v);
        pivots.push(p);
    }
    let k = basis.len();
    if support.len() != 1usize << k {
        return Ok(None); // support is not an affine subspace
    }

    // Flat magnitude, and the phase at each coordinate vector.
    let mut turns: Vec<f64> = vec![f64::NAN; 1usize << k];
    for &x in &support {
        let z = amps[x as usize];
        if (z.norm() - magnitude).abs() > 1e-9 * magnitude.max(1.0) {
            return Ok(None); // not flat
        }
        let v = x ^ offset;
        let mut s = 0usize;
        let mut check = 0u64;
        for (i, b) in basis.iter().enumerate() {
            if (v >> pivots[i]) & 1 == 1 {
                s |= 1 << i;
                check ^= *b;
            }
        }
        if check != v {
            return Ok(None);
        }
        let mut turn = (z / amps[offset as usize]).arg() / TAU;
        if turn < 0.0 {
            turn += 1.0;
        }
        turns[s] = turn;
    }
    if turns.iter().any(|t| t.is_nan()) {
        return Ok(None);
    }

    // Smallest power-of-two denominator holding every phase.
    let mut denominator = 0i64;
    let mut d = 1i64;
    while d <= 1 << 20 {
        if turns
            .iter()
            .all(|t| (t * d as f64 - (t * d as f64).round()).abs() < 1e-6)
        {
            denominator = d;
            break;
        }
        d <<= 1;
    }
    if denominator == 0 {
        return Ok(None);
    }

    // Möbius transform: values to multilinear coefficients over ℤ/D.
    let mut a: Vec<i64> = turns
        .iter()
        .map(|t| ((t * denominator as f64).round() as i64).rem_euclid(denominator))
        .collect();
    for i in 0..k {
        for s in 0..(1usize << k) {
            if (s >> i) & 1 == 1 {
                a[s] = (a[s] - a[s ^ (1 << i)]).rem_euclid(denominator);
            }
        }
    }
    let monomials: Vec<(u64, i64)> = a
        .iter()
        .enumerate()
        .filter(|(_, c)| **c != 0)
        .map(|(s, c)| (s as u64, *c))
        .collect();
    let degree = monomials
        .iter()
        .map(|(s, _)| s.count_ones() as usize)
        .max()
        .unwrap_or(0);
    let _ = m;
    Ok(Some(PhasePoly {
        offset,
        basis,
        pivots,
        reference,
        denominator,
        monomials,
        degree,
    }))
}

// ---------------------------------------------------------------------
// Canonical polyadic — the m-way rank, seen all at once.
// ---------------------------------------------------------------------

/// A surface as `Σ_{r<R} ⊗_p v_p^{(r)}`: rank-one terms over every leg
/// simultaneously, which is the notion of rank a bipartition cannot
/// express.
#[derive(Debug, Clone)]
pub struct CpFit {
    /// Terms in the sum.
    pub rank: usize,
    /// Factor `p` as `2 × rank`, row-major.
    pub factors: Vec<Vec<C64>>,
    /// Relative Frobenius deviation achieved.
    pub error: f64,
    /// Alternating least-squares sweeps used.
    pub sweeps: usize,
}

impl CpFit {
    /// Two numbers per leg per term.
    pub fn scalars(&self) -> u128 {
        2 * self.rank as u128 * self.factors.len() as u128
    }

    /// Rebuild the surface this describes.
    pub fn rebuild(&self, m: usize) -> Vec<C64> {
        let mut out = vec![C64::zero(); 1usize << m];
        for (x, o) in out.iter_mut().enumerate() {
            for r in 0..self.rank {
                let mut z = C64::new(1.0, 0.0);
                for (p, f) in self.factors.iter().enumerate() {
                    z *= f[((x >> p) & 1) * self.rank + r];
                }
                *o += z;
            }
        }
        out
    }
}

/// Alternating least squares for a rank-`rank` polyadic fit.
fn cp_als(amps: &[C64], m: usize, rank: usize, sweeps: usize, seed: u64) -> Result<CpFit> {
    let mut rng = Prng::new(seed);
    let mut factors: Vec<Vec<C64>> = (0..m)
        .map(|_| {
            (0..2 * rank)
                .map(|_| C64::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5))
                .collect()
        })
        .collect();
    let norm = amps.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let mut error = f64::INFINITY;
    let mut used = 0usize;

    for sweep in 0..sweeps {
        crate::guard::checkpoint()?;
        used = sweep + 1;
        for p in 0..m {
            // Gram of every other factor, Hadamard-multiplied.
            let mut a = vec![C64::new(1.0, 0.0); rank * rank];
            for (q, f) in factors.iter().enumerate() {
                if q == p {
                    continue;
                }
                for r in 0..rank {
                    for s in 0..rank {
                        let g = f[r] * f[s].conj() + f[rank + r] * f[rank + s].conj();
                        a[r * rank + s] *= g;
                    }
                }
            }
            // Matricized-tensor-times-Khatri-Rao product.
            let mut mt = vec![C64::zero(); 2 * rank];
            for (x, z) in amps.iter().enumerate() {
                if z.norm() == 0.0 {
                    continue;
                }
                let i = (x >> p) & 1;
                for r in 0..rank {
                    let mut w = *z;
                    for (q, f) in factors.iter().enumerate() {
                        if q == p {
                            continue;
                        }
                        w *= f[((x >> q) & 1) * rank + r].conj();
                    }
                    mt[i * rank + r] += w;
                }
            }
            // F = M · A⁻¹, through the pseudo-inverse so a rank-deficient
            // Gram degrades instead of exploding.
            let svd = svd_thin(rank, rank, &a, 1e-12, 1e-300)?;
            let mut next = vec![C64::zero(); 2 * rank];
            for i in 0..2 {
                for s in 0..rank {
                    let mut acc = C64::zero();
                    for kk in 0..svd.rank {
                        let mut uh = C64::zero();
                        for r in 0..rank {
                            uh += mt[i * rank + r] * svd.u[r * svd.rank + kk].conj();
                        }
                        acc += uh / C64::new(svd.sigma[kk], 0.0) * svd.vt[kk * rank + s].conj();
                    }
                    next[i * rank + s] = acc;
                }
            }
            factors[p] = next;
        }

        let fit = CpFit {
            rank,
            factors: factors.clone(),
            error: 0.0,
            sweeps: used,
        };
        let rebuilt = fit.rebuild(m);
        let dev = amps
            .iter()
            .zip(&rebuilt)
            .map(|(a, b)| (*a - *b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let rel = if norm > 0.0 { dev / norm } else { dev };
        if error - rel < 1e-12 {
            error = rel;
            break;
        }
        error = rel;
    }
    Ok(CpFit {
        rank,
        factors,
        error,
        sweeps: used,
    })
}

/// Smallest polyadic rank reaching `epsilon` relative deviation, by
/// doubling the rank until it does or `max_rank` is reached.
pub fn cp_fit(
    amps: &[C64],
    m: usize,
    epsilon: f64,
    max_rank: usize,
    sweeps: usize,
    seed: u64,
) -> Result<CpFit> {
    let mut best = cp_als(amps, m, 1, sweeps, seed)?;
    let mut rank = 1usize;
    while best.error > epsilon && rank < max_rank {
        rank = (rank * 2).min(max_rank);
        let f = cp_als(amps, m, rank, sweeps, seed)?;
        if f.error < best.error {
            best = f;
        }
    }
    Ok(best)
}

// ---------------------------------------------------------------------
// Grain — the same surface read at higher polarity arity.
// ---------------------------------------------------------------------

/// One rung of the coarse-to-fine ladder: the surface's legs read as
/// digits of arity `radix` instead of as bits.
///
/// Every technique above is **bipolar**. A bipartition is a ±
/// split, the Walsh characters are the `±1` characters of `F₂^m`, and
/// [`crate::sweep::plan_volumes`] prices a volume at *two* surfaces.
/// That is arity two everywhere, and arity two is a choice.
///
/// A leg is one `CZ` — one entangling event, at one time. Grouping `b`
/// consecutive legs into a digit makes an inclusion/exclusion axis of
/// arity `d = 2^b` out of `b` consecutive entangling events, in the
/// sense [`crate::polarity`] means: a generator splitting the surface
/// into `d` sectors rather than two. The harmonics of that axis are the
/// `ℤ_d` characters `ω^{jk}`, `ω = e^{2πi/d}`, which are **not** the
/// `ℤ₂^b` characters the Walsh transform uses on the same `b` legs —
/// same legs, different group, different notion of what is irreducible.
///
/// So the ladder runs from `b = 1` (bipolar, the reading everything
/// else uses) to `b = m` (one digit of arity `2^m`, no decomposition at
/// all), and reports at each rung how much of the surface is occupied
/// in each reading.
#[derive(Debug, Clone)]
pub struct Grain {
    /// Legs per digit.
    pub legs_per_digit: usize,
    /// Digit arity, `2^legs_per_digit`.
    pub radix: usize,
    /// Digits the surface splits into.
    pub sites: usize,
    /// Occupied coefficients in the `ℤ_d` character basis — the
    /// arity-`d` polarity harmonics.
    pub cyclic_nnz: usize,
    /// Occupied coefficients in the `ℤ₂^m` character basis, which is
    /// the bipolar reading and does not depend on the grouping.
    pub walsh_nnz: usize,
    /// Occupied entries in the computational basis, likewise.
    pub direct_nnz: usize,
    /// Widest bond of the matrix product over `radix`-ary sites.
    pub chi: usize,
    /// Largest cut entropy at this grain, in bits.
    pub entropy: f64,
}

impl Grain {
    /// Numbers to describe the surface at this grain, taking whichever
    /// of the three readings is smallest.
    pub fn scalars(&self) -> u128 {
        let chain =
            2 * self.radix as u128 * self.chi as u128 * self.chi as u128 * self.sites as u128;
        (2 * self.cyclic_nnz as u128)
            .min(2 * self.walsh_nnz as u128)
            .min(2 * self.direct_nnz as u128)
            .min(chain)
    }
}

/// The `ℤ_d` character transform, digit by digit, in place. Unitary, so
/// the occupancy it reports is comparable with the Walsh transform's.
fn cyclic_transform(amps: &mut [C64], radix: usize, sites: usize) {
    let n = amps.len();
    let root: Vec<C64> = (0..radix)
        .map(|k| {
            let th = -TAU * k as f64 / radix as f64;
            C64::new(th.cos(), th.sin())
        })
        .collect();
    let scale = 1.0 / (radix as f64).sqrt();
    let mut stride = 1usize;
    let mut buf = vec![C64::zero(); radix];
    for _ in 0..sites {
        let span = stride * radix;
        for block in (0..n).step_by(span) {
            for off in 0..stride {
                for (a, b) in buf.iter_mut().enumerate() {
                    *b = amps[block + off + a * stride];
                }
                for k in 0..radix {
                    let mut acc = C64::zero();
                    for (a, b) in buf.iter().enumerate() {
                        acc += *b * root[(a * k) % radix];
                    }
                    amps[block + off + k * stride] = acc * C64::new(scale, 0.0);
                }
            }
        }
        stride = span;
    }
}

/// The smallest `ℤ_D` holding every one of the surface's phases,
/// measured against its largest entry, or `None` if no `D` up to `2^20`
/// does.
///
/// This is the *other* arity, and the one the measurements point at. A
/// surface's harmonics can be spread across the whole character group
/// at every reading ([`grain_scan`]) while its phases still sit in a
/// small cyclic group — those are independent facts. A single stabilizer
/// term over Clifford+`T` has `D = 8`; a sum of them generally has no
/// `D` at all, because the sum of eighth roots is not a root of unity.
/// So this rises with how much circuit stands behind the bond, and
/// where it stops existing is where the surface stopped being one term.
pub fn phase_arity(amps: &[C64]) -> Option<i64> {
    let peak = amps
        .iter()
        .max_by(|a, b| a.norm().partial_cmp(&b.norm()).unwrap())
        .copied()?;
    if peak.norm() <= ZERO_TOLERANCE {
        return None;
    }
    let mut d = 1i64;
    while d <= 1 << 20 {
        let ok = amps.iter().filter(|z| z.norm() > ZERO_TOLERANCE).all(|z| {
            let t = (*z / peak).arg() / TAU * d as f64;
            (t - t.round()).abs() < 1e-6
        });
        if ok {
            return Some(d);
        }
        d <<= 1;
    }
    None
}

/// Occupied entries, relative to the largest — so a change of
/// normalization cannot change the count.
fn nnz_relative(v: &[C64], rel: f64) -> usize {
    let peak = v.iter().map(|z| z.norm()).fold(0.0, f64::max);
    if peak == 0.0 {
        return 0;
    }
    v.iter().filter(|z| z.norm() > rel * peak).count()
}

/// Walk the surface from bipolar up to a single compound axis, and
/// report what is occupied at every rung.
///
/// Grains are the digit widths dividing the leg count, so every rung
/// tiles the legs exactly.
pub fn grain_scan(surface: &Surface, epsilon: f64) -> Result<Vec<Grain>> {
    let m = surface.legs.len();
    let mut out = Vec::new();
    if m == 0 {
        return Ok(out);
    }
    let direct_nnz = nnz_relative(&surface.amps, 1e-12);
    let mut walsh = surface.amps.clone();
    crate::crossview::fwht_c(&mut walsh);
    let walsh_nnz = nnz_relative(&walsh, 1e-12);

    for b in 1..=m {
        if m % b != 0 {
            continue;
        }
        crate::guard::checkpoint()?;
        let radix = 1usize << b;
        let sites = m / b;
        let mut hat = surface.amps.clone();
        cyclic_transform(&mut hat, radix, sites);

        // The matrix product over d-ary sites: the same chain, read at
        // this grain. A cut between digits is a cut between legs, so
        // the widths are the bipolar chain's widths at multiples of b.
        let chain = decompose(surface, epsilon, false)?;
        let (chi, entropy) = if chain.code.ceiling_hit {
            (usize::MAX, f64::NAN)
        } else {
            let mut chi = 1usize;
            let mut ent: f64 = 0.0;
            for (j, r) in chain.code.exact.iter().enumerate() {
                if (j + 1) % b == 0 {
                    chi = chi.max(*r);
                    ent = ent.max(chain.code.entropy[j]);
                }
            }
            (chi, ent)
        };

        out.push(Grain {
            legs_per_digit: b,
            radix,
            sites,
            cyclic_nnz: nnz_relative(&hat, 1e-12),
            walsh_nnz,
            direct_nnz,
            chi,
            entropy,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------
// The symbolic surface — the volume behind it, never its values.
// ---------------------------------------------------------------------

/// The volume to the left of a bond, held as a path sum over its own
/// variables rather than as the values it takes on the surface.
///
/// A leg is qubit `q`'s value at the moment a `CZ` fired, so the volume
/// that produces the surface is the sub-circuit on qubits `0..=q` with
/// each of those `CZ`s replaced by a `CNOT` copying `q` onto a fresh
/// wire. Post-selecting the data qubits on the measured outcome leaves
/// a state on the leg wires whose amplitudes *are* the surface — and
/// the path sum holds it as a polynomial, so its description is the
/// monomial count while its evaluation is `2^{h*}` per entry.
///
/// The two numbers are independent, which is the point: a surface can
/// be short to write and expensive to read.
#[derive(Debug, Clone)]
pub struct SymbolicSurface {
    /// The bond, between this qubit and the one above it.
    pub after_qubit: usize,
    /// Legs on the bond.
    pub legs: usize,
    /// Internal variables surviving reduction. Reading one entry binds
    /// the outputs, reduces again, then sums over the `2^{h*}`
    /// assignments that survive — so this exponent is the read's
    /// scaling part, not its whole cost.
    pub h_star: usize,
    /// Monomials in the reduced phase polynomial.
    pub terms: usize,
    /// Whether reduction stopped on the step cap rather than a fixpoint.
    pub cut_short: bool,
    sum: crate::pathsum::PathSum,
    data_qubits: usize,
    bits: u64,
}

impl SymbolicSurface {
    /// One surface entry, evaluated from the polynomial.
    pub fn amplitude(&self, b: u64) -> C64 {
        self.sum.amplitude(self.bits | (b << self.data_qubits))
    }

    /// Monomial and coefficient apiece.
    pub fn scalars(&self) -> u128 {
        2 * self.terms as u128
    }
}

/// The volume behind one bond, reduced, up to [`MAX_VOLUME_WIRES`].
pub fn symbolic(
    circuit: &crate::circuit::Circuit<C64>,
    after_qubit: usize,
    bits: u64,
) -> Result<SymbolicSurface> {
    symbolic_within(circuit, after_qubit, bits, MAX_VOLUME_WIRES)
}

/// [`symbolic`] with the wire ceiling chosen by the caller.
pub fn symbolic_within(
    circuit: &crate::circuit::Circuit<C64>,
    after_qubit: usize,
    bits: u64,
    max_wires: usize,
) -> Result<SymbolicSurface> {
    use crate::circuit::{Circuit, Op};
    let q = after_qubit;
    let mut legs = 0usize;
    for op in circuit.ops() {
        if let Op::Named { qubits, .. } = op {
            if qubits.len() == 2 {
                let (lo, hi) = (qubits[0].min(qubits[1]), qubits[0].max(qubits[1]));
                if lo == q && hi == q + 1 {
                    legs += 1;
                }
            }
        }
    }
    let width = q + 1 + legs;
    if width > 64 {
        return Err(Error::InvalidState(format!(
            "surface: the volume behind bond {q} needs {width} wires, past the 64 \
             an amplitude index addresses"
        )));
    }
    if width > max_wires {
        return Err(Error::InvalidState(format!(
            "surface: the volume behind bond {q} needs {width} wires, past the {max_wires} \
             where one reduction sweep can outlast the budget set around it"
        )));
    }
    let mut left = Circuit::<C64>::new(width);
    let mut j = 0usize;
    for op in circuit.ops() {
        let Op::Named {
            name,
            params,
            qubits,
        } = op
        else {
            return Err(Error::InvalidState(
                "surface: only named registry gates form a volume".into(),
            ));
        };
        match qubits.len() {
            1 => {
                if qubits[0] <= q {
                    left.gate(name.clone(), params.clone(), qubits.clone());
                }
            }
            2 => {
                let (lo, hi) = (qubits[0].min(qubits[1]), qubits[0].max(qubits[1]));
                if hi <= q {
                    left.gate(name.clone(), params.clone(), vec![lo, hi]);
                } else if lo == q && hi == q + 1 {
                    // The leg records q's value here, which is a copy.
                    left.cx(q, q + 1 + j);
                    j += 1;
                }
            }
            k => {
                return Err(Error::InvalidState(format!(
                    "surface: {k}-qubit gates do not split into legs"
                )))
            }
        }
    }
    let sum = crate::pathsum::PathSum::from_circuit(&left)?;
    Ok(SymbolicSurface {
        after_qubit: q,
        legs,
        h_star: sum.internal_vars(),
        terms: sum.terms(),
        cut_short: sum.reduction_cut_short(),
        data_qubits: q + 1,
        bits: bits & ((1u64 << (q + 1)) - 1),
        sum,
    })
}

// ---------------------------------------------------------------------
// The competition.
// ---------------------------------------------------------------------

/// Which description of a surface was tried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Technique {
    /// Every amplitude, written out.
    Dense,
    /// Occupied entries only.
    Support,
    /// Occupied entries of the Walsh–Hadamard transform.
    Walsh,
    /// Matrix product over the time-ordered legs.
    Mps,
    /// Matrix product over the time-ordered *characters*.
    WalshMps,
    /// Affine support with a multilinear phase.
    PhasePoly,
    /// Sum of rank-one terms over all legs at once.
    Cp,
    /// The volume behind the surface, as a phase polynomial.
    PathSum,
}

impl Technique {
    /// Name as printed.
    pub fn name(&self) -> &'static str {
        match self {
            Technique::Dense => "dense",
            Technique::Support => "support",
            Technique::Walsh => "walsh",
            Technique::Mps => "mps-svd",
            Technique::WalshMps => "walsh-mps",
            Technique::PhasePoly => "phase-poly",
            Technique::Cp => "cp-als",
            Technique::PathSum => "path-sum",
        }
    }
}

/// One entrant's result on one surface.
///
/// Two costs, because they are genuinely independent and a single
/// number hides the trade: a description can be short to *write* and
/// expensive to *read*, and the path sum is exactly that. Scoring only
/// the first would call it the winner; scoring only the second would
/// never let it enter.
#[derive(Debug, Clone)]
pub struct Decomposition {
    /// The technique.
    pub technique: Technique,
    /// Numbers that must cross the bond — indices and amplitudes alike.
    pub scalars: u128,
    /// Scalar operations to read **one** surface entry back out.
    pub read_cost: u128,
    /// Largest amplitude deviation measured over `checked` entries, or
    /// `NaN` when reading back was itself unaffordable.
    pub error: f64,
    /// Entries actually compared. `0` means the deviation is unmeasured
    /// and the entry is a claim, not a result.
    pub checked: usize,
    /// Whether the technique could represent this surface at all.
    pub applicable: bool,
    /// What it found, in one line.
    pub detail: String,
}

impl Decomposition {
    /// Verified to `tolerance` on at least one entry.
    pub fn verified(&self, tolerance: f64) -> bool {
        self.applicable && self.checked > 0 && self.error <= tolerance
    }
}

/// Every entrant on one surface.
#[derive(Debug, Clone)]
pub struct Competition {
    /// The bond, between this qubit and the one above it.
    pub after_qubit: usize,
    /// Legs on the bond.
    pub legs: usize,
    /// `2^legs`.
    pub raw: u128,
    /// Results, in the order tried.
    pub entries: Vec<Decomposition>,
}

impl Competition {
    /// Shortest description that was verified to `tolerance`.
    pub fn winner(&self, tolerance: f64) -> Option<&Decomposition> {
        self.entries
            .iter()
            .filter(|e| e.verified(tolerance))
            .min_by_key(|e| e.scalars)
    }

    /// Cheapest to read back, among the verified.
    pub fn fastest(&self, tolerance: f64) -> Option<&Decomposition> {
        self.entries
            .iter()
            .filter(|e| e.verified(tolerance))
            .min_by_key(|e| e.read_cost)
    }

    /// One entrant by name.
    pub fn get(&self, technique: Technique) -> Option<&Decomposition> {
        self.entries.iter().find(|e| e.technique == technique)
    }
}

/// What the competition is allowed to spend.
#[derive(Debug, Clone)]
pub struct CompeteConfig {
    /// Squared weight the truncating techniques may discard per bond.
    pub epsilon: f64,
    /// Relative deviation the polyadic fit aims for.
    pub cp_epsilon: f64,
    /// Largest polyadic rank tried. Each alternating step inverts a
    /// `rank × rank` Gram through a Jacobi decomposition, so this is
    /// the term that sets what a fit costs — not the surface's size.
    pub cp_max_rank: usize,
    /// Legs past which the polyadic fit is skipped — its sweeps cost
    /// `O(m² · 2^m · R)` and stop being worth running.
    pub cp_max_legs: usize,
    /// Alternating least-squares sweeps per rank.
    pub cp_sweeps: usize,
    /// Seed for the polyadic fit's start.
    pub seed: u64,
    /// Enumeration steps the path-sum entrant may spend proving
    /// itself. A read is a re-reduction *plus* a sum over `2^{h*}`
    /// assignments, and this budgets the second part — the one that
    /// scales — so a surface whose reads are unaffordable reports
    /// `checked: 0` rather than being quietly believed.
    pub pathsum_budget: u128,
    /// Wall-clock the path sum may spend reducing the volume before it
    /// is abandoned and reports why.
    pub symbolic_budget: std::time::Duration,
    /// Wires the volume behind a bond may have; see
    /// [`MAX_VOLUME_WIRES`] for why a wall-clock budget alone is not
    /// enough.
    pub symbolic_max_wires: usize,
}

impl Default for CompeteConfig {
    fn default() -> Self {
        Self {
            epsilon: DEFAULT_EPSILON,
            cp_epsilon: 1e-8,
            cp_max_rank: 16,
            cp_max_legs: 12,
            cp_sweeps: 25,
            seed: 0x51DE_0F00,
            pathsum_budget: 1 << 24,
            symbolic_budget: std::time::Duration::from_secs(10),
            symbolic_max_wires: MAX_VOLUME_WIRES,
        }
    }
}

/// Which assumption the affine/phase description failed on. The two
/// obstructions are different facts about the surface — an affine
/// support with uneven magnitudes is a *sum* of stabilizer terms, which
/// is a much weaker failure than a support that is not a coset at all.
fn phase_poly_obstruction(amps: &[C64]) -> String {
    let support: Vec<u64> = amps
        .iter()
        .enumerate()
        .filter(|(_, z)| z.norm() > ZERO_TOLERANCE)
        .map(|(x, _)| x as u64)
        .collect();
    if support.is_empty() {
        return "surface is identically zero".into();
    }
    let offset = support[0];
    let mut basis: Vec<u64> = Vec::new();
    let mut pivots: Vec<usize> = Vec::new();
    for &x in &support {
        let mut v = x ^ offset;
        for (i, b) in basis.iter().enumerate() {
            if (v >> pivots[i]) & 1 == 1 {
                v ^= *b;
            }
        }
        if v != 0 {
            pivots.push(63 - v.leading_zeros() as usize);
            basis.push(v);
        }
    }
    if support.len() != 1usize << basis.len() {
        return format!(
            "support is not a coset: {} points inside a span of {}",
            support.len(),
            1usize << basis.len()
        );
    }
    let mut levels = std::collections::HashSet::new();
    for &x in &support {
        levels.insert((amps[x as usize].norm() * 1e9).round() as u64);
    }
    format!(
        "support is a coset of dimension {}, but magnitudes take {} values — a \
         sum of stabilizer terms, not one",
        basis.len(),
        levels.len()
    )
}

fn deviation(a: &[C64], b: &[C64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0, f64::max)
}

/// Run every technique on one surface and score them alike.
///
/// `volume` is the circuit the surface came from; pass it to let the
/// symbolic entrant reconstruct the volume behind the bond, or `None`
/// to run only the techniques that read the surface's values.
pub fn compete(
    surface: &Surface,
    volume: Option<&crate::circuit::Circuit<C64>>,
    bits: u64,
    cfg: &CompeteConfig,
) -> Result<Competition> {
    let m = surface.legs.len();
    let amps = &surface.amps;
    let mut entries = Vec::new();

    let n_entries = amps.len();
    entries.push(Decomposition {
        technique: Technique::Dense,
        scalars: n_entries as u128,
        read_cost: 1,
        error: 0.0,
        checked: n_entries,
        applicable: true,
        detail: format!("{n_entries} amplitudes"),
    });

    let nnz = amps.iter().filter(|z| z.norm() > ZERO_TOLERANCE).count();
    entries.push(Decomposition {
        technique: Technique::Support,
        scalars: 2 * nnz as u128,
        read_cost: 1,
        error: 0.0,
        checked: n_entries,
        applicable: true,
        detail: format!(
            "{nnz} of {n_entries} occupied ({:.1}%)",
            100.0 * nnz as f64 / n_entries as f64
        ),
    });

    // The character basis. W(W(v)) = N·v, so the inverse is the same
    // transform scaled — both directions are exact.
    let mut hat = amps.clone();
    crate::crossview::fwht_c(&mut hat);
    let scale = 1.0 / amps.len() as f64;
    for z in hat.iter_mut() {
        *z *= C64::new(scale, 0.0);
    }
    let hnnz = hat.iter().filter(|z| z.norm() > ZERO_TOLERANCE).count();
    let mut back = hat.clone();
    crate::crossview::fwht_c(&mut back);
    entries.push(Decomposition {
        technique: Technique::Walsh,
        scalars: 2 * hnnz as u128,
        read_cost: hnnz as u128,
        error: deviation(amps, &back),
        checked: n_entries,
        applicable: true,
        detail: format!(
            "{hnnz} of {n_entries} characters ({:.1}%)",
            100.0 * hnnz as f64 / n_entries as f64
        ),
    });

    let chain = decompose(surface, cfg.epsilon, false)?;
    if chain.code.ceiling_hit {
        entries.push(Decomposition {
            technique: Technique::Mps,
            scalars: u128::MAX,
            read_cost: u128::MAX,
            error: f64::NAN,
            checked: 0,
            applicable: false,
            detail: format!("bond exceeded {MAX_CHI} at cut {}", chain.code.exact.len()),
        });
    } else {
        let chi = chain.code.code_dim() as u128;
        entries.push(Decomposition {
            technique: Technique::Mps,
            scalars: chain.scalars(),
            read_cost: 2 * m as u128 * chi * chi,
            error: deviation(amps, &chain.rebuild(m)),
            checked: n_entries,
            applicable: true,
            detail: format!(
                "χ ≤ {chi}, peak entropy {:.2} bits of {:.1} possible",
                chain.code.peak_entropy(),
                m as f64 / 2.0
            ),
        });
    }

    let hat_surface = Surface {
        after_qubit: surface.after_qubit,
        legs: surface.legs.clone(),
        amps: hat.clone(),
    };
    let hchain = decompose(&hat_surface, cfg.epsilon, false)?;
    if hchain.code.ceiling_hit {
        entries.push(Decomposition {
            technique: Technique::WalshMps,
            scalars: u128::MAX,
            read_cost: u128::MAX,
            error: f64::NAN,
            checked: 0,
            applicable: false,
            detail: format!("bond exceeded {MAX_CHI} at cut {}", hchain.code.exact.len()),
        });
    } else {
        let mut rebuilt = hchain.rebuild(m);
        crate::crossview::fwht_c(&mut rebuilt);
        let chi = hchain.code.code_dim() as u128;
        entries.push(Decomposition {
            technique: Technique::WalshMps,
            scalars: hchain.scalars(),
            // One entry needs the whole inverse transform, so the
            // dual-basis chain is cheap to store and dear to read.
            read_cost: n_entries as u128 * 2 * m as u128 * chi * chi,
            error: deviation(amps, &rebuilt),
            checked: n_entries,
            applicable: true,
            detail: format!("χ ≤ {chi} in the character basis"),
        });
    }

    match phase_poly(amps, m)? {
        Some(pp) => entries.push(Decomposition {
            technique: Technique::PhasePoly,
            scalars: pp.scalars(),
            read_cost: pp.monomials.len() as u128,
            error: deviation(amps, &pp.rebuild(m)),
            checked: n_entries,
            applicable: true,
            detail: format!(
                "affine dim {} of {m}, degree {} over ℤ/{}, {} monomials",
                pp.basis.len(),
                pp.degree,
                pp.denominator,
                pp.monomials.len()
            ),
        }),
        None => entries.push(Decomposition {
            technique: Technique::PhasePoly,
            scalars: u128::MAX,
            read_cost: u128::MAX,
            error: f64::NAN,
            checked: 0,
            applicable: false,
            detail: phase_poly_obstruction(amps),
        }),
    }

    if m <= cfg.cp_max_legs {
        let fit = cp_fit(
            amps,
            m,
            cfg.cp_epsilon,
            cfg.cp_max_rank,
            cfg.cp_sweeps,
            cfg.seed,
        )?;
        let err = deviation(amps, &fit.rebuild(m));
        entries.push(Decomposition {
            technique: Technique::Cp,
            scalars: fit.scalars(),
            read_cost: fit.rank as u128 * m as u128,
            error: err,
            checked: n_entries,
            applicable: fit.error <= cfg.cp_epsilon,
            detail: format!(
                "rank {} after {} sweeps, relative {:.2e}{}",
                fit.rank,
                fit.sweeps,
                fit.error,
                if fit.rank >= cfg.cp_max_rank && fit.error > cfg.cp_epsilon {
                    " (rank ceiling)"
                } else {
                    ""
                }
            ),
        });
    } else {
        entries.push(Decomposition {
            technique: Technique::Cp,
            scalars: u128::MAX,
            read_cost: u128::MAX,
            error: f64::NAN,
            checked: 0,
            applicable: false,
            detail: format!("skipped above {} legs", cfg.cp_max_legs),
        });
    }

    if let Some(circuit) = volume {
        match crate::guard::with_time_budget(cfg.symbolic_budget, || {
            symbolic_within(circuit, surface.after_qubit, bits, cfg.symbolic_max_wires)
        }) {
            Ok(sym) => {
                // One entry sums over 2^{h*} internal assignments and
                // touches every monomial at each, so the reads are
                // budgeted by that product rather than by the entries.
                let per_read =
                    (1u128 << sym.h_star.min(100)).saturating_mul(sym.terms.max(1) as u128);
                let reads = (cfg.pathsum_budget / per_read.max(1)).min(n_entries as u128) as usize;
                let mut worst: f64 = 0.0;
                let mut rng = Prng::new(cfg.seed);
                for i in 0..reads {
                    crate::guard::checkpoint()?;
                    let b = if reads == n_entries {
                        i as u64
                    } else {
                        rng.next_u64() & ((1u64 << m) - 1)
                    };
                    worst = worst.max((amps[b as usize] - sym.amplitude(b)).norm());
                }
                entries.push(Decomposition {
                    technique: Technique::PathSum,
                    scalars: sym.scalars(),
                    read_cost: per_read,
                    error: if reads == 0 { f64::NAN } else { worst },
                    checked: reads,
                    applicable: !sym.cut_short,
                    detail: format!(
                        "{} monomials, h* = {}, one entry sums 2^{}{}",
                        sym.terms,
                        sym.h_star,
                        sym.h_star,
                        if sym.cut_short {
                            " — REDUCTION CUT SHORT"
                        } else {
                            ""
                        }
                    ),
                });
            }
            Err(e) => entries.push(Decomposition {
                technique: Technique::PathSum,
                scalars: u128::MAX,
                read_cost: u128::MAX,
                error: f64::NAN,
                checked: 0,
                applicable: false,
                detail: format!("{e}"),
            }),
        }
    }

    Ok(Competition {
        after_qubit: surface.after_qubit,
        legs: m,
        raw: 1u128 << m.min(127),
        entries,
    })
}

/// How often each technique won across a set of surfaces.
pub fn tally(runs: &[Competition], tolerance: f64) -> HashMap<Technique, usize> {
    let mut out = HashMap::new();
    for r in runs {
        if let Some(w) = r.winner(tolerance) {
            *out.entry(w.technique).or_insert(0) += 1;
        }
    }
    out
}
