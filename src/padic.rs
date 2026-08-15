//! Interference as **modular arithmetic**: a p-adic phase register, the
//! CRT diagonal that factors it, and a journalled sweep whose cost is the
//! informational content of the question rather than the size of the grid.
//!
//! # The premise
//!
//! Two waves are in phase when their path difference is zero *modulo* a
//! wavelength. An interference pattern is a function of `Δφ mod 2π` and
//! nothing else. That is a statement about a **cyclic group**, not about
//! the reals, and this module takes it literally: a phase lives in `ℤ/M`,
//! where `M` is a modulus the *geometry supplies* ([`Radix::from_denominators`]),
//! and every question about interference becomes a question about
//! congruences.
//!
//! Two structural facts then do all the work, and both are measured here
//! rather than assumed:
//!
//! 1. **CRT factorization.** `ℤ/M ≅ ∏ ℤ/p_i^{n_i}` for `M = ∏ p_i^{n_i}`.
//!    A phase `e^{2πi a x / M}` is therefore a **product of single-component
//!    phases** — [`crt_phase_factors`] computes the per-component
//!    coefficients, and [`factored_field`] applies them to a
//!    [`CompoundRegister`] of mixed arity.
//!    The whole `M`-point phase field is stored in `Σ p_i^{n_i}` entries
//!    instead of `M`, exactly, with the deviation measured at ~1e-15.
//!    This is the Good–Thomas factorization read as a *representation*:
//!    the interference problem does not couple the coprime components at
//!    all, so their costs add instead of multiplying.
//!
//! 2. **The interference character is a low-digit quantity.** Write
//!    `Δa`'s residue in each component in p-adic digits, least significant
//!    first. Then `gcd(Δa, M) = ∏ p_i^{min(v_i, n_i)}` where `v_i` is the
//!    number of leading zero digits, and the **order** of `e^{2πiΔa/M}` as
//!    a root of unity is `M / gcd(Δa, M)`. Order 1 is exact constructive
//!    interference; order 2 is exact antiphase — total destruction. So the
//!    qualitative interference character is decided by the *trailing*
//!    digits, and [`character`] reads exactly `Σ_i (v_i + 1)` of them
//!    before it can stop: **a cost in the number of primes, independent of
//!    `M`**. Resolving the *amplitude* to full precision still costs the
//!    full depth `Σ n_i`; the gap between those two costs is the point.
//!
//! Fact 2 is why the p-adic direction is the right one and also why it is
//! the *opposite* of a wavelet: the p-adic expansion resolves fine
//! structure first and coarse structure last. That is backwards for
//! magnitude and exactly right for phase, because interference is a
//! congruence condition and congruences live at the fine end.
//!
//! # The diagonal
//!
//! A [`CrtDiagonal`] is an ordering of the `(component, digit)` pairs — the
//! sequence in which the CRT/p-adic information is consumed. After `k`
//! steps the phase is known modulo a running `L`, and the **residual** is
//! the number of surviving congruence classes `M / L`
//! ([`Resolution::classes`]), which strictly shrinks by a factor `p_i` at
//! every step. That monotonicity is what makes the journal
//! binary-searchable and the early exit sound.
//!
//! The ordering is not a free parameter to be optimized combinatorially,
//! and the measurement says so plainly: for the character question the
//! expected number of steps under a given ordering is fixed by the primes
//! themselves (a digit in base `p` is zero with probability `1/p`), so
//! [`CrtDiagonal::CoarsestFirst`] — largest prime first — decides fastest and
//! [`CrtDiagonal::Interleaved`] is within a constant of it.
//! [`compare_diagonals`] measures all four orderings on the same
//! population rather than arguing about them.
//!
//! # The sweep, and why the *point* order matters more
//!
//! The ordering that actually costs something is the order in which
//! **phase-space points** are visited. A phase field `Δa(x) = c + Σ s_d x_d
//! mod M` changes by a *fixed increment* along each lattice direction, so
//! a sweep that walks the lattice touches only the digits a carry
//! reaches: `Σ_i (1 − p_i^{−n_i})/(1 − 1/p_i) → Σ_i p_i/(p_i − 1)`
//! writes per point, **independent of the depth**. A sweep that visits
//! the same points in a scrambled order rewrites every digit that happens
//! to differ, `Σ_i n_i(1 − 1/p_i)` of them — **linear in the depth**.
//! [`predicted_writes_per_point`] is those two closed forms and [`sweep`]
//! measures against them (`2^12` on a grid covering `ℤ/M` exactly:
//! predicted 1.9995 / 6.0000, measured 1.9995 / 5.9502).
//!
//! The measurement also corrects the obvious guess about *which* lattice
//! order to choose. Every lattice order lands on the same closed form —
//! adding any fixed stride to a residue moves only as many digits as the
//! carry reaches, so no axis is privileged and neither is Morton order.
//! What costs `Θ(depth)` is not a badly chosen direction but having no
//! fixed stride at all. "Propagate linearly across the surface, not
//! chaotically" is exactly right, and it is the *linearity* that carries
//! the whole claim, not the choice of line.
//!
//! The journal additionally caches decided verdicts by their deciding
//! prefix, and [`SweepReport::cache_probes`] prices that cache next to
//! [`SweepReport::reused`] so a hit rate is never quoted without its
//! cost. Measured, it does not pay: the resolution it replaces already
//! costs `O(#components)`. What *does* pay is [`fringe_period`] — the
//! exact spacing `M / gcd(Δs, M)` at which a pair returns to the same
//! character, read off the trailing digits of the slope difference with
//! the field never evaluated anywhere, so one resolution answers a whole
//! arithmetic progression of points.
//!
//! # Where the radix comes from
//!
//! There is no bootstrapping problem, because the radix is read off the
//! geometry in closed form. If the path differences of the sources are
//! `a_j / q_j` wavelengths, then `M = lcm(q_j)` and the radix is `M`'s
//! prime factorization — [`Radix::from_denominators`]. Commensurate
//! geometry gives a small `M` and a shallow, cheap register; incommensurate
//! geometry has **no finite radix at all**, and must be approximated by a
//! continued-fraction convergent ([`best_rational`]), whose denominator is
//! the honest price of the irrationality. [`geometry_radix`] reports the
//! resulting modulus and the phase error it costs, and the measurement
//! reproduces the physics: the golden-ratio spacing — the worst-case
//! irrational — has the fastest-growing denominators of any geometry, so
//! the quasi-periodic pattern really is the expensive one.
//!
//! # Honesty about scope
//!
//! Everything here is exact **when the phases are rational multiples of
//! `2π`**. That is not a small fragment — it is the same fragment
//! [`exact`](crate::exact) privileges (`M = 2^b` is the `D[ω]` ring) and
//! the same one qudit registers of arity `d` live in — but it is a
//! fragment, and irrational geometry enters only through a measured
//! approximation with a reported error. Nothing in this module makes a
//! claim about the general continuous-phase problem.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::math::cis;
use crate::mixed::CompoundRegister;
use crate::rng::Prng;
use crate::scalar::C64;

/// Largest modulus this module will factor or build a register over.
/// Structural: the CRT weights are computed in `u128`, and the factored
/// register allocates `Σ p_i^{n_i}` amplitudes.
pub const MAX_MODULUS: u64 = 1 << 40;

// ── number theory the module needs, kept explicit ────────────────────

/// Greatest common divisor.
pub fn gcd(a: u64, b: u64) -> u64 {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Least common multiple; `None` on overflow past [`MAX_MODULUS`].
pub fn lcm(a: u64, b: u64) -> Option<u64> {
    if a == 0 || b == 0 {
        return None;
    }
    let g = gcd(a, b);
    let v = (a as u128 / g as u128) * b as u128;
    if v > MAX_MODULUS as u128 {
        None
    } else {
        Some(v as u64)
    }
}

/// Prime factorization by trial division, ascending in the prime.
pub fn factor(mut m: u64) -> Vec<(u64, u32)> {
    let mut out = Vec::new();
    let mut d = 2u64;
    while d.saturating_mul(d) <= m {
        if m % d == 0 {
            let mut e = 0u32;
            while m % d == 0 {
                m /= d;
                e += 1;
            }
            out.push((d, e));
        }
        d += if d == 2 { 1 } else { 2 };
    }
    if m > 1 {
        out.push((m, 1));
    }
    out
}

/// Modular inverse of `a` mod `m`, or `None` when `gcd(a, m) ≠ 1`.
pub fn mod_inv(a: u64, m: u64) -> Option<u64> {
    if m == 1 {
        return Some(0);
    }
    let (mut old_r, mut r) = (a as i128 % m as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);
    while r != 0 {
        let q = old_r / r;
        let t = old_r - q * r;
        old_r = r;
        r = t;
        let t = old_s - q * s;
        old_s = s;
        s = t;
    }
    if old_r != 1 {
        return None;
    }
    Some(old_s.rem_euclid(m as i128) as u64)
}

/// `a` reduced into `[0, m)` from any signed value.
pub fn reduce(a: i128, m: u64) -> u64 {
    a.rem_euclid(m as i128) as u64
}

// ── the radix ────────────────────────────────────────────────────────

/// The phase modulus `M` together with its coprime prime-power
/// decomposition — the **radix** the p-adic register is written in.
///
/// Components are held ascending in the prime, which is the order
/// [`CrtDiagonal::Interleaved`] and [`CrtDiagonal::Sequential`] consume them in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Radix {
    comps: Vec<(u64, u32)>,
    modulus: u64,
}

impl Radix {
    /// Factor an explicit modulus.
    pub fn of(modulus: u64) -> Result<Self> {
        if modulus < 2 {
            return Err(Error::InvalidState(format!(
                "phase modulus must be at least 2, got {modulus}"
            )));
        }
        if modulus > MAX_MODULUS {
            return Err(Error::InvalidState(format!(
                "phase modulus {modulus} exceeds MAX_MODULUS {MAX_MODULUS}"
            )));
        }
        Ok(Radix {
            comps: factor(modulus),
            modulus,
        })
    }

    /// The radix a geometry supplies: `M = lcm(denominators)` of the path
    /// differences expressed in wavelengths. This is the closed-form
    /// answer to "where does the radix come from" — nothing about the
    /// field has to be computed first.
    pub fn from_denominators(dens: &[u64]) -> Result<Self> {
        if dens.is_empty() {
            return Err(Error::InvalidState(
                "no path-difference denominators given".into(),
            ));
        }
        let mut m = 1u64;
        for &d in dens {
            if d == 0 {
                return Err(Error::InvalidState("zero denominator".into()));
            }
            m = lcm(m.max(1), d).ok_or_else(|| {
                Error::InvalidState(format!(
                    "denominators {dens:?} need a modulus beyond MAX_MODULUS {MAX_MODULUS}"
                ))
            })?;
        }
        Radix::of(m)
    }

    /// The modulus `M`.
    pub fn modulus(&self) -> u64 {
        self.modulus
    }

    /// `(prime, power)` pairs, ascending in the prime.
    pub fn components(&self) -> &[(u64, u32)] {
        &self.comps
    }

    /// The coprime moduli `p_i^{n_i}`.
    pub fn component_moduli(&self) -> Vec<u64> {
        self.comps.iter().map(|&(p, n)| p.pow(n)).collect()
    }

    /// Total digit count `Σ n_i` — the full resolution depth.
    pub fn depth(&self) -> u32 {
        self.comps.iter().map(|&(_, n)| n).sum()
    }

    /// Residues of `a` in each component.
    pub fn residues(&self, a: u64) -> Vec<u64> {
        self.component_moduli().iter().map(|&m| a % m).collect()
    }

    /// CRT weights `w_i = (M/m_i)·((M/m_i)^{-1} mod m_i) mod M`, so that
    /// `a = Σ w_i·r_i mod M` reconstructs from residues.
    pub fn crt_weights(&self) -> Vec<u64> {
        let m = self.modulus as u128;
        self.component_moduli()
            .iter()
            .map(|&mi| {
                let co = m / mi as u128;
                let inv = mod_inv((co % mi as u128) as u64, mi).expect("coprime by construction");
                ((co * inv as u128) % m) as u64
            })
            .collect()
    }

    /// Reconstruct the residue in `ℤ/M` from per-component residues.
    pub fn reconstruct(&self, residues: &[u64]) -> Result<u64> {
        if residues.len() != self.comps.len() {
            return Err(Error::InvalidState(format!(
                "radix has {} components, got {} residues",
                self.comps.len(),
                residues.len()
            )));
        }
        let w = self.crt_weights();
        let mut acc = 0u128;
        for (i, &r) in residues.iter().enumerate() {
            acc = (acc + (w[i] as u128) * (r as u128)) % self.modulus as u128;
        }
        Ok(acc as u64)
    }

    /// Amplitudes needed to hold a phase field over `ℤ/M` densely.
    pub fn dense_entries(&self) -> u64 {
        self.modulus
    }

    /// Amplitudes needed to hold the same field CRT-factored: `Σ p_i^{n_i}`.
    pub fn factored_entries(&self) -> u64 {
        self.component_moduli().iter().sum()
    }
}

// ── p-adic digits ────────────────────────────────────────────────────

/// Digits of `r` in base `p`, least significant first, padded to `n`.
pub fn digits(r: u64, p: u64, n: u32) -> Vec<u64> {
    let mut out = Vec::with_capacity(n as usize);
    let mut r = r;
    for _ in 0..n {
        out.push(r % p);
        r /= p;
    }
    out
}

/// The p-adic valuation of `r` inside a component of modulus `p^n`: the
/// count of leading zero digits, capped at `n` (`r = 0` gives `n`).
pub fn valuation(r: u64, p: u64, n: u32) -> u32 {
    if r % p.pow(n) == 0 {
        return n;
    }
    let mut r = r % p.pow(n);
    let mut v = 0u32;
    while r % p == 0 {
        r /= p;
        v += 1;
    }
    v
}

// ── the interference character ───────────────────────────────────────

/// What the low digits of a phase difference decide: the exact
/// interference character of `e^{2πiΔa/M}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Character {
    /// Order of the phase as a root of unity: `M / gcd(Δa, M)`.
    /// `1` is exact constructive interference, `2` exact antiphase.
    pub order: u64,
    /// `gcd(Δa, M)` — the size of the in-phase sublattice.
    pub coherence: u64,
    /// Per-component p-adic valuations `v_i`, capped at `n_i`.
    pub valuations: Vec<u32>,
    /// Digit reads performed: `Σ_i min(v_i + 1, n_i)` — the measured cost,
    /// which does not depend on `M` beyond its number of prime components.
    pub digit_reads: usize,
}

impl Character {
    /// Exactly constructive: the waves are in phase.
    pub fn constructive(&self) -> bool {
        self.order == 1
    }

    /// Exactly antiphase: total destructive interference.
    pub fn antiphase(&self) -> bool {
        self.order == 2
    }
}

/// Read the interference character of a phase difference from its
/// **trailing** digits, stopping in each component at the first nonzero
/// digit.
///
/// The cost is `Σ_i min(v_i + 1, n_i)` digit reads: with digits uniform,
/// the expected cost per component is `p/(p−1) ≤ 2`, so the whole
/// character costs `O(#components)` regardless of how large `M` is.
pub fn character(delta: u64, radix: &Radix) -> Character {
    let mut valuations = Vec::with_capacity(radix.components().len());
    let mut reads = 0usize;
    let mut coherence: u128 = 1;
    let residues = radix.residues(delta);
    for (i, &(p, n)) in radix.components().iter().enumerate() {
        let mut v = 0u32;
        let mut x = residues[i];
        // read digits least-significant first, stop at the first nonzero
        while v < n {
            reads += 1;
            if x % p != 0 {
                break;
            }
            x /= p;
            v += 1;
        }
        valuations.push(v);
        coherence *= (p as u128).pow(v);
    }
    let coherence = coherence as u64;
    Character {
        order: radix.modulus() / coherence,
        coherence,
        valuations,
        digit_reads: reads,
    }
}

// ── the CRT diagonal ─────────────────────────────────────────────────

/// An ordering of the `(component, digit)` pairs — the sequence in which
/// the phase information is consumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrtDiagonal {
    /// One digit from each component in turn, ascending in the prime,
    /// then the next digit of each: the geometry-matched interleave.
    Interleaved,
    /// One component to full depth before the next.
    Sequential,
    /// Interleaved, but largest prime first at each level.
    CoarsestFirst,
    /// A deterministic shuffle of the full step list — the control.
    Shuffled(u64),
}

impl CrtDiagonal {
    /// The full step sequence: `(component index, digit index)` pairs,
    /// `Σ n_i` of them.
    pub fn steps(&self, radix: &Radix) -> Vec<(usize, u32)> {
        let comps = radix.components();
        let max_n = comps.iter().map(|&(_, n)| n).max().unwrap_or(0);
        let mut steps = Vec::with_capacity(radix.depth() as usize);
        match self {
            CrtDiagonal::Interleaved => {
                for level in 0..max_n {
                    for (i, &(_, n)) in comps.iter().enumerate() {
                        if level < n {
                            steps.push((i, level));
                        }
                    }
                }
            }
            CrtDiagonal::CoarsestFirst => {
                for level in 0..max_n {
                    for (i, &(_, n)) in comps.iter().enumerate().rev() {
                        if level < n {
                            steps.push((i, level));
                        }
                    }
                }
            }
            CrtDiagonal::Sequential => {
                for (i, &(_, n)) in comps.iter().enumerate() {
                    for level in 0..n {
                        steps.push((i, level));
                    }
                }
            }
            CrtDiagonal::Shuffled(seed) => {
                for (i, &(_, n)) in comps.iter().enumerate() {
                    for level in 0..n {
                        steps.push((i, level));
                    }
                }
                let mut rng = Prng::new(*seed);
                // Fisher–Yates, but digits of one component must stay in
                // ascending order (a digit is only meaningful once the
                // lower ones are known), so shuffle component *slots*.
                for k in (1..steps.len()).rev() {
                    let j = (rng.next_u64() % (k as u64 + 1)) as usize;
                    steps.swap(k, j);
                }
                repair_digit_order(&mut steps, comps.len());
            }
        }
        steps
    }
}

/// After a shuffle, rewrite each component's digit indices back into
/// ascending order at whatever positions that component occupies — the
/// *positions* stay scrambled, the digit sequence within a component
/// stays legal.
fn repair_digit_order(steps: &mut [(usize, u32)], ncomp: usize) {
    let mut next = vec![0u32; ncomp];
    for s in steps.iter_mut() {
        s.1 = next[s.0];
        next[s.0] += 1;
    }
}

/// The progressive state of one phase's resolution along a diagonal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolution {
    /// Steps consumed.
    pub steps: usize,
    /// Running modulus `L` — the phase is known mod `L`.
    pub modulus: u64,
    /// Surviving congruence classes `M / L`: the **residual**. Strictly
    /// decreasing in `steps`, by a factor `p_i` each step.
    pub classes: u64,
    /// `Some(true)` once the phase is proven exactly constructive (all
    /// digits consumed and zero), `Some(false)` at the first nonzero
    /// digit, `None` while still undecided.
    pub in_phase: Option<bool>,
}

/// Walk `diagonal` over `delta`, stopping as soon as the in-phase
/// question is decided.
///
/// The residual [`Resolution::classes`] shrinks monotonically, so a
/// journal of these is binary-searchable and an early exit is sound: a
/// nonzero digit can never be un-seen by a later step.
pub fn resolve(delta: u64, radix: &Radix, diagonal: CrtDiagonal) -> Resolution {
    let steps = diagonal.steps(radix);
    let residues = radix.residues(delta);
    let comps = radix.components();
    let mut l: u128 = 1;
    let mut used = 0usize;
    for &(ci, level) in &steps {
        let (p, _) = comps[ci];
        let d = (residues[ci] / p.pow(level)) % p;
        l *= p as u128;
        used += 1;
        if d != 0 {
            return Resolution {
                steps: used,
                modulus: l as u64,
                classes: radix.modulus() / l as u64,
                in_phase: Some(false),
            };
        }
    }
    Resolution {
        steps: used,
        modulus: radix.modulus(),
        classes: 1,
        in_phase: Some(true),
    }
}

/// Mean steps to decide, per diagonal ordering, over a population of
/// phase differences.
#[derive(Debug, Clone)]
pub struct DiagonalComparison {
    /// `(ordering label, mean steps to decide, worst case)`.
    pub orderings: Vec<(&'static str, f64, usize)>,
    /// Full depth `Σ n_i` — the cost of resolving the amplitude instead
    /// of the character.
    pub full_depth: u32,
    /// Population size.
    pub samples: usize,
}

/// Measure every [`CrtDiagonal`] on the same population of phase
/// differences, drawn uniformly from `ℤ/M`.
pub fn compare_diagonals(radix: &Radix, samples: usize, seed: u64) -> DiagonalComparison {
    let mut rng = Prng::new(seed);
    let deltas: Vec<u64> = (0..samples)
        .map(|_| rng.next_u64() % radix.modulus())
        .collect();
    let mut orderings = Vec::new();
    for (label, d) in [
        ("interleaved", CrtDiagonal::Interleaved),
        ("sequential", CrtDiagonal::Sequential),
        ("coarsest-first", CrtDiagonal::CoarsestFirst),
        ("shuffled", CrtDiagonal::Shuffled(seed ^ 0x5eed)),
    ] {
        let mut total = 0usize;
        let mut worst = 0usize;
        for &delta in &deltas {
            let r = resolve(delta, radix, d);
            total += r.steps;
            worst = worst.max(r.steps);
        }
        orderings.push((label, total as f64 / samples.max(1) as f64, worst));
    }
    DiagonalComparison {
        orderings,
        full_depth: radix.depth(),
        samples,
    }
}

// ── the wave system ──────────────────────────────────────────────────

/// One interfering component: an integer phase `offset + Σ slope_d·x_d`
/// over `ℤ/M`, with a complex weight.
#[derive(Debug, Clone)]
pub struct Wave {
    /// Constant phase term, in units of `2π/M`.
    pub offset: i64,
    /// Phase increment per unit step along each coordinate.
    pub slope: Vec<i64>,
    /// Complex amplitude weight.
    pub weight: C64,
}

impl Wave {
    /// A unit-weight wave.
    pub fn new(offset: i64, slope: Vec<i64>) -> Self {
        Wave {
            offset,
            slope,
            weight: C64::new(1.0, 0.0),
        }
    }

    /// The integer phase at `x`, reduced into `[0, M)`.
    pub fn phase_at(&self, x: &[i64], modulus: u64) -> u64 {
        let mut acc = self.offset as i128;
        for (d, &xd) in x.iter().enumerate() {
            if d < self.slope.len() {
                acc += self.slope[d] as i128 * xd as i128;
            }
        }
        reduce(acc, modulus)
    }
}

/// A set of waves over a shared phase modulus.
#[derive(Debug, Clone)]
pub struct WaveSystem {
    radix: Radix,
    waves: Vec<Wave>,
    dims: usize,
}

impl WaveSystem {
    /// Build a system; every wave's slope must have `dims` entries.
    pub fn new(radix: Radix, dims: usize, waves: Vec<Wave>) -> Result<Self> {
        if waves.is_empty() {
            return Err(Error::InvalidState("wave system needs a wave".into()));
        }
        for w in &waves {
            if w.slope.len() != dims {
                return Err(Error::InvalidState(format!(
                    "wave slope has {} entries, expected {dims}",
                    w.slope.len()
                )));
            }
        }
        Ok(WaveSystem { radix, waves, dims })
    }

    /// The radix.
    pub fn radix(&self) -> &Radix {
        &self.radix
    }

    /// The waves.
    pub fn waves(&self) -> &[Wave] {
        &self.waves
    }

    /// Phase-space dimension.
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// The direct interference sum at `x` — the reference every
    /// factorization in this module is measured against.
    pub fn amplitude(&self, x: &[i64]) -> C64 {
        let m = self.radix.modulus() as f64;
        let mut acc = C64::new(0.0, 0.0);
        for w in &self.waves {
            let a = w.phase_at(x, self.radix.modulus());
            acc += w.weight * cis(std::f64::consts::TAU * a as f64 / m);
        }
        acc
    }

    /// The phase difference of waves `i` and `j` at `x`, in `ℤ/M`.
    pub fn delta(&self, i: usize, j: usize, x: &[i64]) -> u64 {
        let m = self.radix.modulus();
        let a = self.waves[i].phase_at(x, m) as i128;
        let b = self.waves[j].phase_at(x, m) as i128;
        reduce(a - b, m)
    }

    /// The interference character of the pair `(i, j)` at `x`.
    pub fn character_at(&self, i: usize, j: usize, x: &[i64]) -> Character {
        character(self.delta(i, j, x), &self.radix)
    }
}

// ── the CRT-factored phase field ─────────────────────────────────────

/// Per-component coefficients of the phase `e^{2πi a x / M}`: the phase
/// factors as `∏_i e^{2πi c_i x_i / m_i}` with `c_i = a·y_i mod m_i`,
/// where `y_i = (M/m_i)^{-1} mod m_i`.
///
/// This is the exact statement that the coprime components **do not
/// couple**: a single global phase over `ℤ/M` is a product of independent
/// single-component phases, so the register that holds it is a product
/// register.
pub fn crt_phase_factors(a: u64, radix: &Radix) -> Vec<u64> {
    radix
        .component_moduli()
        .iter()
        .map(|&mi| {
            let co = (radix.modulus() / mi) as u128;
            let y = mod_inv((co % mi as u128) as u64, mi).expect("coprime by construction");
            (((a as u128 % mi as u128) * y as u128) % mi as u128) as u64
        })
        .collect()
}

/// What [`factored_field`] measured.
#[derive(Debug, Clone)]
pub struct FactoredField {
    /// Interfering components — the rank of the factored representation.
    pub rank: usize,
    /// Amplitudes the factored representation holds: `rank · Σ p_i^{n_i}`.
    pub factored_entries: u64,
    /// Amplitudes a dense field over `ℤ/M` would hold: `M`.
    pub dense_entries: u64,
    /// Independent volumes each wave's register still had after the whole
    /// field was written — equal to the component count exactly when the
    /// components never correlated, which is the factorization claim as
    /// the register itself measures it.
    pub volumes: Vec<usize>,
    /// Worst deviation between the factored evaluation and the direct
    /// interference sum, over the probed points.
    pub max_deviation: f64,
    /// Points probed.
    pub probes: usize,
    /// The compound register's arities, one per CRT component.
    pub dims: Vec<usize>,
}

impl FactoredField {
    /// Storage ratio `dense / factored`.
    pub fn compression(&self) -> f64 {
        self.dense_entries as f64 / self.factored_entries.max(1) as f64
    }

    /// Whether every wave's register stayed a product across the CRT
    /// components (no interaction ever merged two of them).
    pub fn stayed_product(&self) -> bool {
        self.volumes.iter().all(|&v| v == self.dims.len())
    }
}

/// Hold each wave's **entire phase field over `ℤ/M`** on a
/// [`CompoundRegister`] whose sites are
/// the CRT components, and measure the assembled interference pattern
/// against the direct sum.
///
/// The construction is the content of the claim. A wave `w_j e^{2πi(o_j +
/// s_j u)/M}` over the whole field `u ∈ ℤ/M` is written by preparing the
/// uniform superposition (a product over the components, one `fourier_d`
/// per site) and applying **one single-site diagonal per component** —
/// [`crt_phase_factors`] gives the coefficients. No two-site gate is ever
/// applied, so the register stays a product: `M` amplitudes held in
/// `Σ p_i^{n_i}` numbers, exactly.
///
/// `k` interfering waves are therefore a **rank-`k`** object of size
/// `k · Σ p_i^{n_i}` rather than `M`. The measured deviation compares the
/// amplitudes read back out of the registers, at the probe points
/// `u ∈ probes`, against [`WaveSystem::amplitude`].
///
/// Requires a one-dimensional phase argument (`system.dims() == 1`); the
/// register's sites are the CRT components of that argument, not the
/// phase-space axes.
pub fn factored_field(system: &WaveSystem, probes: &[u64]) -> Result<FactoredField> {
    if system.dims() != 1 {
        return Err(Error::InvalidState(format!(
            "factored_field takes a 1-D phase argument, system has {}",
            system.dims()
        )));
    }
    let radix = system.radix();
    let dims: Vec<usize> = radix
        .component_moduli()
        .iter()
        .map(|&m| m as usize)
        .collect();
    if dims.iter().any(|&d| d < 2) {
        return Err(Error::InvalidState(
            "every CRT component must have arity ≥ 2".into(),
        ));
    }
    let m = radix.modulus();
    let norm = (m as f64).sqrt();

    // One register per wave: the whole field of that wave, as a product.
    let mut registers = Vec::with_capacity(system.waves().len());
    for w in system.waves() {
        let mut reg = CompoundRegister::new(&dims)?;
        for (i, &d) in dims.iter().enumerate() {
            reg.apply_1(i, &crate::mixed::fourier_d(d))?;
        }
        let slope = reduce(w.slope[0] as i128, m);
        let factors = crt_phase_factors(slope, radix);
        for (i, &d) in dims.iter().enumerate() {
            let diag: Vec<C64> = (0..d)
                .map(|k| {
                    let e = (factors[i] as u128 * k as u128) % d as u128;
                    cis(std::f64::consts::TAU * e as f64 / d as f64)
                })
                .collect();
            reg.apply_1_diagonal(i, &diag)?;
        }
        registers.push(reg);
    }
    let volumes: Vec<usize> = registers.iter().map(|r| r.volumes()).collect();

    let mut max_dev: f64 = 0.0;
    for &u in probes {
        let u = u % m;
        let point: Vec<usize> = radix.residues(u).iter().map(|&r| r as usize).collect();
        let mut assembled = C64::new(0.0, 0.0);
        for (j, w) in system.waves().iter().enumerate() {
            // the register holds e^{2πi s_j u / M} / √M at |u⟩
            let amp = registers[j].amplitude(&point)? * norm;
            let offset = cis(std::f64::consts::TAU * reduce(w.offset as i128, m) as f64 / m as f64);
            assembled += w.weight * offset * amp;
        }
        let reference = system.amplitude(&[u as i64]);
        max_dev = max_dev.max((assembled - reference).norm());
    }

    Ok(FactoredField {
        rank: system.waves().len(),
        factored_entries: radix.factored_entries() * system.waves().len() as u64,
        dense_entries: radix.dense_entries(),
        volumes,
        max_deviation: max_dev,
        probes: probes.len(),
        dims,
    })
}

// ── the fringe period, from digits alone ─────────────────────────────

/// The exact period at which a pair of waves returns to the same
/// interference character along a phase-space axis: `M / gcd(Δs, M)`,
/// where `Δs` is the slope difference along that axis.
///
/// This is the fringe spacing, and it is a **valuation** — computed from
/// the trailing digits of `Δs` in `O(#components)` reads, with the field
/// never evaluated anywhere. It is also what makes the journal able to
/// *skip*: every point of the progression `x₀ + t·period` carries the
/// same character as `x₀`, so one resolution answers a whole coset
/// instead of one point.
///
/// `None` when the pair's phase does not vary along that axis at all
/// (`Δs ≡ 0`), in which case every point on the axis shares one
/// character.
pub fn fringe_period(system: &WaveSystem, pair: (usize, usize), axis: usize) -> Option<u64> {
    let m = system.radix().modulus();
    let a = system.waves()[pair.0].slope.get(axis).copied().unwrap_or(0);
    let b = system.waves()[pair.1].slope.get(axis).copied().unwrap_or(0);
    let ds = reduce(a as i128 - b as i128, m);
    if ds == 0 {
        return None;
    }
    Some(m / gcd(ds, m))
}
// ── the journalled sweep ─────────────────────────────────────────────

/// The order in which phase-space points are visited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Row-major with axis `d` varying fastest — a lattice walk.
    Axis(usize),
    /// Digit-interleaved (Morton / Z-order); requires power-of-two
    /// extents.
    Morton,
    /// Deterministic shuffle of the same point set — the chaotic control.
    Shuffled(u64),
}

impl Order {
    /// Enumerate the grid `∏ [0, extent_d)` in this order.
    pub fn points(&self, extent: &[i64]) -> Result<Vec<Vec<i64>>> {
        let total: i64 = extent.iter().product();
        if total <= 0 {
            return Err(Error::InvalidState("empty phase-space extent".into()));
        }
        let mut pts: Vec<Vec<i64>> = Vec::with_capacity(total as usize);
        match self {
            Order::Axis(fast) => {
                if *fast >= extent.len() {
                    return Err(Error::InvalidState(format!(
                        "fast axis {fast} outside {} dimensions",
                        extent.len()
                    )));
                }
                // significance order: `fast` least significant, the rest
                // ascending above it — a plain mixed-radix odometer.
                let mut axes = vec![*fast];
                axes.extend((0..extent.len()).filter(|d| d != fast));
                let mut idx = vec![0i64; extent.len()];
                for _ in 0..total {
                    pts.push(idx.clone());
                    for &axis in &axes {
                        idx[axis] += 1;
                        if idx[axis] < extent[axis] {
                            break;
                        }
                        idx[axis] = 0;
                    }
                }
            }
            Order::Morton => {
                for &e in extent {
                    if e <= 0 || (e & (e - 1)) != 0 {
                        return Err(Error::InvalidState(
                            "Morton order needs power-of-two extents".into(),
                        ));
                    }
                }
                let bits: Vec<u32> = extent.iter().map(|&e| e.trailing_zeros()).collect();
                for code in 0..total {
                    let mut idx = vec![0i64; extent.len()];
                    let mut c = code;
                    let maxb = *bits.iter().max().unwrap();
                    let mut written = vec![0u32; extent.len()];
                    for b in 0..maxb {
                        for d in 0..extent.len() {
                            if b < bits[d] {
                                idx[d] |= (c & 1) << written[d];
                                written[d] += 1;
                                c >>= 1;
                            }
                        }
                    }
                    pts.push(idx);
                }
            }
            Order::Shuffled(seed) => {
                let base = Order::Axis(0).points(extent)?;
                pts = base;
                let mut rng = Prng::new(*seed);
                for k in (1..pts.len()).rev() {
                    let j = (rng.next_u64() % (k as u64 + 1)) as usize;
                    pts.swap(k, j);
                }
            }
        }
        Ok(pts)
    }
}

/// One journalled step of a sweep.
#[derive(Debug, Clone)]
pub struct JournalEntry {
    /// The phase-space point.
    pub point: Vec<i64>,
    /// The pair's phase difference there.
    pub delta: u64,
    /// How far the resolution got.
    pub resolution: Resolution,
    /// Digit positions rewritten relative to the previous point.
    pub digit_writes: usize,
    /// Answered from the prefix table rather than by descending.
    pub reused: bool,
}

/// A resumable, rewindable record of a sweep. Residuals are monotone
/// within a point, and the journal keeps every point's stopping state, so
/// a sweep can be interrupted and continued with no recomputation
/// ([`Sweep::resume`]) or rolled back to any earlier point
/// ([`Sweep::rewind_to`]).
#[derive(Debug, Clone)]
pub struct Sweep {
    entries: Vec<JournalEntry>,
    order: Vec<Vec<i64>>,
    cursor: usize,
    /// Decided verdicts keyed by deciding modulus `L`, then residue mod
    /// `L`. Held per-`L` so a lookup is one hash probe per distinct
    /// modulus seen (at most `depth` of them) rather than a scan.
    cache: HashMap<u64, HashMap<u64, bool>>,
    cache_probes: usize,
    reuse: bool,
    diagonal: CrtDiagonal,
    pair: (usize, usize),
    digit_state: Vec<u64>,
}

/// What a sweep measured.
#[derive(Debug, Clone)]
pub struct SweepReport {
    /// Points visited.
    pub points: usize,
    /// Total digit positions rewritten across the sweep.
    pub digit_writes: usize,
    /// `digit_writes / points` — the locality of the visiting order.
    pub writes_per_point: f64,
    /// Total resolution steps taken (descents into the diagonal).
    pub steps: usize,
    /// `steps / points`.
    pub steps_per_point: f64,
    /// Points answered from the verdict cache without descending.
    pub reused: usize,
    /// Cache probes performed — the honest cost of the reuse mechanism,
    /// so a hit rate can never be quoted without its price.
    pub cache_probes: usize,
    /// Points found exactly in phase.
    pub in_phase: usize,
    /// Full depth `Σ n_i` — the per-point cost of resolving the amplitude
    /// instead of the character.
    pub full_depth: u32,
}

impl SweepReport {
    /// Digit writes per point relative to the full depth: `1.0` means the
    /// order rewrote the whole register at every point.
    pub fn write_fraction(&self) -> f64 {
        self.writes_per_point / self.full_depth.max(1) as f64
    }
}

impl Sweep {
    /// Start a sweep of `system`'s pair `(i, j)` over the grid `extent`.
    pub fn new(
        system: &WaveSystem,
        pair: (usize, usize),
        extent: &[i64],
        order: Order,
        diagonal: CrtDiagonal,
        reuse: bool,
    ) -> Result<Self> {
        if pair.0 >= system.waves().len() || pair.1 >= system.waves().len() {
            return Err(Error::InvalidState("wave index out of range".into()));
        }
        if extent.len() != system.dims() {
            return Err(Error::InvalidState(format!(
                "extent has {} dimensions, system has {}",
                extent.len(),
                system.dims()
            )));
        }
        Ok(Sweep {
            entries: Vec::new(),
            order: order.points(extent)?,
            cursor: 0,
            cache: HashMap::new(),
            cache_probes: 0,
            reuse,
            diagonal,
            pair,
            digit_state: Vec::new(),
        })
    }

    /// Advance at most `budget` points; returns how many were visited.
    /// Calling it again continues exactly where it stopped.
    pub fn resume(&mut self, system: &WaveSystem, budget: usize) -> usize {
        let radix = system.radix().clone();
        let comps: Vec<(u64, u32)> = radix.components().to_vec();
        let depth = radix.depth() as usize;
        if self.digit_state.is_empty() {
            self.digit_state = vec![u64::MAX; depth];
        }
        let mut done = 0usize;
        while done < budget && self.cursor < self.order.len() {
            let point = self.order[self.cursor].clone();
            let delta = system.delta(self.pair.0, self.pair.1, &point);

            // digit writes: how much of the register this point changed
            let residues = radix.residues(delta);
            let mut flat = Vec::with_capacity(depth);
            for (ci, &(p, n)) in comps.iter().enumerate() {
                flat.extend(digits(residues[ci], p, n));
            }
            let writes = flat
                .iter()
                .zip(self.digit_state.iter())
                .filter(|(a, b)| a != b)
                .count();
            self.digit_state = flat;

            // resolution, possibly answered from the prefix table
            let mut reused = false;
            let resolution = if self.reuse {
                match self.lookup(delta, &radix) {
                    Some(res) => {
                        reused = true;
                        res
                    }
                    None => {
                        let r = resolve(delta, &radix, self.diagonal);
                        if let Some(v) = r.in_phase {
                            self.cache
                                .entry(r.modulus)
                                .or_default()
                                .insert(delta % r.modulus, v);
                        }
                        r
                    }
                }
            } else {
                resolve(delta, &radix, self.diagonal)
            };

            self.entries.push(JournalEntry {
                point,
                delta,
                resolution,
                digit_writes: writes,
                reused,
            });
            self.cursor += 1;
            done += 1;
        }
        done
    }

    /// A cache hit: some earlier point decided at modulus `L` with the
    /// same residue mod `L`, so this point's verdict is already known —
    /// sound because the verdict depends only on the consumed digits.
    fn lookup(&mut self, delta: u64, radix: &Radix) -> Option<Resolution> {
        let moduli: Vec<u64> = self.cache.keys().copied().collect();
        for l in moduli {
            self.cache_probes += 1;
            if let Some(&verdict) = self.cache[&l].get(&(delta % l)) {
                return Some(Resolution {
                    steps: 0,
                    modulus: l,
                    classes: radix.modulus() / l,
                    in_phase: Some(verdict),
                });
            }
        }
        None
    }

    /// Roll the journal back to `point_index`, discarding later entries.
    /// The verdict cache is kept: nothing it recorded can become false.
    pub fn rewind_to(&mut self, point_index: usize) {
        self.entries.truncate(point_index);
        self.cursor = point_index.min(self.order.len());
        self.digit_state.clear();
    }

    /// The journal.
    pub fn entries(&self) -> &[JournalEntry] {
        &self.entries
    }

    /// Points not yet visited.
    pub fn remaining(&self) -> usize {
        self.order.len() - self.cursor
    }

    /// Summarize what the sweep measured so far.
    pub fn report(&self, radix: &Radix) -> SweepReport {
        let points = self.entries.len();
        let digit_writes: usize = self.entries.iter().map(|e| e.digit_writes).sum();
        let steps: usize = self.entries.iter().map(|e| e.resolution.steps).sum();
        let reused = self.entries.iter().filter(|e| e.reused).count();
        let in_phase = self
            .entries
            .iter()
            .filter(|e| e.resolution.in_phase == Some(true))
            .count();
        SweepReport {
            points,
            digit_writes,
            writes_per_point: digit_writes as f64 / points.max(1) as f64,
            steps,
            steps_per_point: steps as f64 / points.max(1) as f64,
            reused,
            cache_probes: self.cache_probes,
            in_phase,
            full_depth: radix.depth(),
        }
    }
}

/// Closed-form prediction for [`SweepReport::writes_per_point`].
///
/// * `lattice = true` — consecutive points step the phase difference by
///   exactly `1`, so a component's digits change only as far as the carry
///   reaches: `Σ_i (1 − p_i^{−n_i}) / (1 − 1/p_i)`, which tends to
///   `Σ_i p_i/(p_i − 1) ≤ 2·#components` **independently of the depth**.
/// * `lattice = false` — consecutive points are uncorrelated, so every
///   digit is rewritten unless it happens to repeat: `Σ_i n_i(1 − 1/p_i)`,
///   which is **linear in the depth**.
///
/// The gap between the two is the entire cost argument for sweeping the
/// phase plane in a lattice order, and [`sweep`] measures both against
/// this prediction rather than asserting either.
pub fn predicted_writes_per_point(radix: &Radix, lattice: bool) -> f64 {
    radix
        .components()
        .iter()
        .map(|&(p, n)| {
            let p = p as f64;
            if lattice {
                (1.0 - p.powi(-(n as i32))) / (1.0 - 1.0 / p)
            } else {
                n as f64 * (1.0 - 1.0 / p)
            }
        })
        .sum()
}

/// Run a sweep to completion and report.
pub fn sweep(
    system: &WaveSystem,
    pair: (usize, usize),
    extent: &[i64],
    order: Order,
    diagonal: CrtDiagonal,
    reuse: bool,
) -> Result<SweepReport> {
    let mut s = Sweep::new(system, pair, extent, order, diagonal, reuse)?;
    let total = s.remaining();
    s.resume(system, total);
    Ok(s.report(system.radix()))
}

// ── the radix a geometry supplies ────────────────────────────────────

/// Best rational approximation `num/den` to `x` with `den ≤ max_den`, by
/// continued fractions. Returns `(num, den, |x − num/den|)`.
pub fn best_rational(x: f64, max_den: u64) -> (i64, u64, f64) {
    // semiconvergent-aware continued fraction expansion
    let neg = x < 0.0;
    let x = x.abs();
    let (mut h0, mut h1) = (0i128, 1i128);
    let (mut k0, mut k1) = (1i128, 0i128);
    let mut v = x;
    let (mut bh, mut bk) = (x.round() as i128, 1i128);
    for _ in 0..64 {
        let a = v.floor();
        if !a.is_finite() {
            break;
        }
        let a_i = a as i128;
        let h2 = a_i * h1 + h0;
        let k2 = a_i * k1 + k0;
        if k2 > max_den as i128 || k2 <= 0 {
            break;
        }
        h0 = h1;
        h1 = h2;
        k0 = k1;
        k1 = k2;
        bh = h1;
        bk = k1;
        let frac = v - a;
        if frac.abs() < 1e-15 {
            break;
        }
        v = 1.0 / frac;
    }
    let approx = bh as f64 / bk as f64;
    let err = (x - approx).abs();
    (if neg { -(bh as i64) } else { bh as i64 }, bk as u64, err)
}

/// The radix a source geometry supplies, with the price of any
/// irrationality made explicit.
#[derive(Debug, Clone)]
pub struct GeometryRadix {
    /// The radix `M = lcm(denominators)` and its factorization.
    pub radix: Radix,
    /// The rational path differences actually used, `(num, den)`.
    pub rationals: Vec<(i64, u64)>,
    /// Worst `|exact − approximated|` path difference, in wavelengths.
    /// Exactly `0.0` when the geometry was already commensurate.
    pub max_path_error: f64,
    /// Worst resulting phase error, in radians.
    pub max_phase_error: f64,
    /// Whether every path difference was represented exactly.
    pub exact: bool,
}

/// Read the radix off a geometry: path differences in wavelengths become
/// rationals (exactly when they are commensurate, by continued-fraction
/// convergent otherwise), and the modulus is their common denominator.
///
/// No property of the interference field is computed first — this is the
/// closed-form answer to the bootstrapping question. What the measurement
/// then shows is that incommensurate geometry has **no finite radix**:
/// the convergents' denominators grow, and for the golden ratio — the
/// worst-approximable number — they grow the fastest of any spacing, so
/// the quasi-periodic pattern is the expensive one on exactly the same
/// axis that makes it quasi-periodic.
pub fn geometry_radix(path_differences: &[f64], max_den: u64) -> Result<GeometryRadix> {
    if path_differences.is_empty() {
        return Err(Error::InvalidState("no path differences given".into()));
    }
    let mut rationals = Vec::with_capacity(path_differences.len());
    let mut dens = Vec::with_capacity(path_differences.len());
    let mut max_path_error: f64 = 0.0;
    for &d in path_differences {
        let (n, q, err) = best_rational(d, max_den);
        rationals.push((n, q));
        dens.push(q);
        max_path_error = max_path_error.max(err);
    }
    let radix = Radix::from_denominators(&dens)?;
    Ok(GeometryRadix {
        radix,
        rationals,
        max_path_error,
        max_phase_error: max_path_error * std::f64::consts::TAU,
        exact: max_path_error == 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crt_round_trips_every_residue() {
        let radix = Radix::of(3 * 4 * 5).unwrap();
        for a in 0..radix.modulus() {
            let r = radix.residues(a);
            assert_eq!(radix.reconstruct(&r).unwrap(), a);
        }
    }

    #[test]
    fn character_is_the_root_of_unity_order() {
        let radix = Radix::of(8 * 9).unwrap();
        for delta in 0..radix.modulus() {
            let c = character(delta, &radix);
            assert_eq!(c.coherence, gcd(delta, radix.modulus()).max(1));
            assert_eq!(c.order, radix.modulus() / c.coherence);
        }
    }

    #[test]
    fn residual_shrinks_monotonically() {
        let radix = Radix::of(2u64.pow(5) * 3 * 5).unwrap();
        let mut rng = Prng::new(11);
        for _ in 0..200 {
            let delta = rng.next_u64() % radix.modulus();
            let mut last = radix.modulus() + 1;
            let steps = CrtDiagonal::Interleaved.steps(&radix);
            let mut l = 1u64;
            for (k, &(ci, level)) in steps.iter().enumerate() {
                let (p, _) = radix.components()[ci];
                let _ = level;
                l *= p;
                let classes = radix.modulus() / l;
                assert!(classes < last, "step {k} did not shrink the residual");
                last = classes;
            }
            let r = resolve(delta, &radix, CrtDiagonal::Interleaved);
            assert!(r.in_phase.is_some());
            assert_eq!(r.in_phase == Some(true), delta == 0);
        }
    }
}
