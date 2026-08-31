//! Regev's factoring algorithm, and the three exponentiation schedules
//! that decide what it costs.
//!
//! Shor spends `Θ(n)` full-width modular multiplications on one exponent
//! register. Regev (2023) spends `Θ(√n)` of them on `d ≈ √n` registers,
//! and pays for it in a lattice problem afterwards. This module builds
//! both halves and *measures* the trade rather than asserting it.
//!
//! ## The construction
//!
//! Pick `d` small primes `p₁ … p_d` coprime to `N` and set `bᵢ = pᵢ²`.
//! Superpose over `z ∈ [0, 2^R)^d`, compute `∏ bᵢ^{zᵢ} mod N` into a work
//! register, measure it, Fourier-transform every exponent register and
//! measure. The samples `w` are (approximately) orthogonal to
//!
//! ```text
//! L = { z ∈ ℤ^d : ∏ bᵢ^{zᵢ} ≡ 1 (mod N) }
//! ```
//!
//! and [`crate::lattice::congruence_kernel`]
//! recovers short vectors of it. The squares are the point: a `v ∈ L`
//! gives `u = ∏ pᵢ^{vᵢ}` with `u² = ∏ bᵢ^{vᵢ} ≡ 1 (mod N)`, so every
//! recovered lattice vector is a square root of one, and any square root
//! other than `±1` splits `N`. Shor gets one square root per successful
//! order-finding run; Regev gets one per short lattice vector.
//!
//! ## The three schedules, and why there are three
//!
//! `∏ bᵢ^{zᵢ}` can be scheduled in more than one way, and the choice is
//! the whole efficiency argument:
//!
//! | [`ExpSchedule`] | full-width mults | small mults | work registers |
//! |---|---|---|---|
//! | [`Sequential`](ExpSchedule::Sequential) | `d·R` | 0 | 1 |
//! | [`Regev`](ExpSchedule::Regev) | `2R` | `2dR` | `R + 2` |
//! | [`Fibonacci`](ExpSchedule::Fibonacci) | `2K`, `K → 1.44R` | `2dK` | 3 |
//!
//! *Sequential* is Shor's schedule widened: one controlled multiplication
//! by the classical constant `bᵢ^{2ʲ}` per exponent bit. With `d ≈ R ≈ √n`
//! that is `d·R ≈ n` full-width multiplications — **no saving at all**.
//! The saving is not in the multi-register idea; it is in the schedule.
//!
//! *Regev* is Horner over the bit planes: `A ← A²·∏ bᵢ^{z_{i,j}}`. Only
//! `R` squarings are full-width; the rest are multiplications by *small*
//! constants. But `y ↦ y²` is not injective mod `N`, so each squaring
//! needs a fresh register ([`shor::xor_square_into`]), and the
//! intermediates must then be uncomputed. `R + 2` registers of `n` bits
//! is the `Õ(n^{3/2})` qubit count Regev's paper reports.
//!
//! *Fibonacci* is the Ragavan–Vaikuntanathan repair. Replace the
//! irreversible `a ↦ a²` with the reversible `(x, y) ↦ (y, x·y)` — one
//! in-place register-into-register multiplication
//! ([`shor::mul_register_into`]), no fresh register, no garbage. The
//! exponent weights that recurrence generates are Fibonacci numbers, not
//! powers of two, so the exponents are read in Zeckendorf digits
//! ([`zeckendorf`]); reaching `2^R` then takes `K` steps, where
//! `F_{K+1} ≥ 2^R` gives `K = R/log₂φ + log_φ√5 + O(1) ≈ 1.44R + 1.7`.
//! The ratio `K/R` tends to 1.4404 *from above* — at `R = 4` it is 1.75,
//! at `R = 48` it is 1.48 — so the "1.44×" is an asymptote, not a
//! constant, and the additive term is what a small simulation actually
//! pays. Three work registers instead of `R + 2` for that. The trade,
//! including where the two qubit counts cross, is measured in
//! `examples/regev.rs`.
//!
//! ## The honest boundary
//!
//! Two things this module does not pretend:
//!
//! * **Regev's advantage is asymptotic and this simulator is not.** At
//!   `d = 2` and `R = 6` the schedule table above is dominated by
//!   constants. What is verifiable at these sizes is that the counts
//!   *are* what the table says, that all three schedules compute the same
//!   permutation, and that the algorithm recovers the factors. The
//!   asymptotic claim is inherited from the counts, not measured.
//! * **The uniform superposition is not Regev's Gaussian.** His analysis
//!   needs a Gaussian-weighted state; [`Superposition::Gaussian`] builds
//!   one by loading amplitudes directly, which is free here and is *not*
//!   free on hardware. `Uniform` is what a Hadamard layer gives and is
//!   what the default uses. Measured, at these sizes the two are
//!   indistinguishable in both cost and success — the preparations
//!   differ, the outcomes do not.
//! * **The lattice weight is a real dial that is not currently binding.**
//!   [`crate::lattice::congruence_kernel`]
//!   explains why demanding exact congruences should fail on rounded
//!   samples, and `tests/lattice.rs` shows it failing on generic rows.
//!   On the samples this algorithm actually produces, a correctly
//!   size-reduced LLL recovers the short kernel vector at every weight
//!   from 1 to 65536 — measured in `examples/regev.rs`. The default of
//!   `1` is the principled choice, not a measured necessity.

use crate::backend::Backend;
use crate::circuit::Circuit;
use crate::error::{Error, Result};
use crate::lattice;
use crate::library;
use crate::padic::{gcd, mod_inv};
use crate::rng::Prng;
use crate::scalar::Scalar;
use crate::shor::{
    self, controlled_mul_mod, mul_register_into, pow_mod, work_bits, xor_copy, xor_square_into,
};
use crate::sim::Simulator;

/// Largest lattice dimension `d` this module will build.
pub const MAX_DIMENSION: usize = 8;

/// Widest register a sampling run may allocate — the sparse backend's
/// `u64` basis index, less one bit of headroom.
pub const MAX_WIDTH: usize = 62;

// ── the small primes ─────────────────────────────────────────────────

/// The first `d` primes coprime to `n`, and the small primes skipped
/// along the way because they divide `n`.
///
/// Regev's construction needs bases in `(ℤ/N)*`, so a small prime that
/// divides `N` cannot be used — and *is* a factor, found for free. It is
/// returned rather than swallowed: at the widths this module simulates,
/// that free byproduct usually finishes the job before the quantum step
/// starts, and a report that hid it would be measuring the wrong thing.
pub fn coprime_primes(n: u64, d: usize) -> (Vec<u64>, Vec<u64>) {
    let mut primes = Vec::with_capacity(d);
    let mut divisors = Vec::new();
    let mut candidate = 2u64;
    while primes.len() < d && candidate < n {
        if shor::is_prime(candidate) {
            if n % candidate == 0 {
                divisors.push(candidate);
            } else {
                primes.push(candidate);
            }
        }
        candidate += 1;
    }
    (primes, divisors)
}

// ── Zeckendorf digits ────────────────────────────────────────────────

/// `F₁ … F_len` with `F₁ = F₂ = 1`.
pub fn fibonacci(len: usize) -> Vec<u64> {
    let mut f = Vec::with_capacity(len);
    for k in 0..len {
        f.push(match k {
            0 | 1 => 1,
            _ => f[k - 1] + f[k - 2],
        });
    }
    f
}

/// Digits `K` needed to represent every value below `2^bits` in
/// Zeckendorf form: the least `K` with `F_{K+1} ≥ 2^bits`.
pub fn zeckendorf_len(bits: usize) -> usize {
    let target = 1u64 << bits;
    let mut k = 2usize;
    loop {
        let f = fibonacci(k + 1);
        if f[k] >= target {
            return k;
        }
        k += 1;
    }
}

/// Greedy Zeckendorf expansion of `value` over `F₁ … F_len`: a bit mask
/// whose bit `k` is the coefficient of `F_{k+1}`. `F₁` is never used
/// (it duplicates `F₂`), so bit 0 is always clear and no two set bits are
/// adjacent.
pub fn zeckendorf(value: u64, fibs: &[u64]) -> u64 {
    let mut rest = value;
    let mut mask = 0u64;
    for k in (1..fibs.len()).rev() {
        if fibs[k] <= rest {
            rest -= fibs[k];
            mask |= 1u64 << k;
        }
    }
    mask
}

// ── schedules and their layout ───────────────────────────────────────

/// How `∏ bᵢ^{zᵢ} mod N` is scheduled onto registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpSchedule {
    /// One controlled multiplication by the classical constant `bᵢ^{2ʲ}`
    /// per exponent bit: `d·R` full-width multiplications, one work
    /// register, nothing to uncompute. Shor's schedule, widened.
    Sequential,
    /// Horner over bit planes with a fresh register per squaring:
    /// `2R` full-width multiplications, `2dR` small ones, `R + 2`
    /// registers. Regev's own schedule.
    Regev,
    /// The Fibonacci ladder `(x, y) ↦ (y, x·y)`: `2K` full-width
    /// register-into-register multiplications, `2dK` small ones, three
    /// registers. Ragavan–Vaikuntanathan's schedule.
    Fibonacci,
}

impl std::fmt::Display for ExpSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `f.pad` rather than `write!`, so `{schedule:<12}` in a caller's
        // table actually pads.
        f.pad(match self {
            ExpSchedule::Sequential => "sequential",
            ExpSchedule::Regev => "regev",
            ExpSchedule::Fibonacci => "fibonacci",
        })
    }
}

impl ExpSchedule {
    /// Number of `n`-bit work registers the schedule holds at once.
    pub fn work_registers(&self, exponent_bits: usize) -> usize {
        match self {
            ExpSchedule::Sequential => 1,
            ExpSchedule::Regev => exponent_bits + 2,
            ExpSchedule::Fibonacci => 3,
        }
    }

    /// The multiplication counts, before any circuit is built: full-width
    /// first, then multiplications by a small constant.
    pub fn multiplications(&self, dimension: usize, exponent_bits: usize) -> (usize, usize) {
        let r = exponent_bits;
        match self {
            ExpSchedule::Sequential => (dimension * r, 0),
            ExpSchedule::Regev => (2 * r, 2 * dimension * r),
            ExpSchedule::Fibonacci => {
                let k = zeckendorf_len(r);
                (2 * k, 2 * dimension * k)
            }
        }
    }
}

/// Which qubit does what.
#[derive(Debug, Clone)]
pub struct Layout {
    /// `d` exponent registers of `R` qubits, little-endian.
    pub exponent: Vec<Vec<usize>>,
    /// Zeckendorf digit registers (Fibonacci schedule only).
    pub digits: Vec<Vec<usize>>,
    /// Work registers of `w` qubits.
    pub work: Vec<Vec<usize>>,
    /// Index into [`work`](Layout::work) of the register that ends up
    /// holding `∏ bᵢ^{zᵢ}`.
    pub output: usize,
    /// Total width.
    pub qubits: usize,
}

impl Layout {
    /// Lay out the registers a schedule needs.
    pub fn new(schedule: ExpSchedule, dimension: usize, exponent_bits: usize, w: usize) -> Self {
        let mut next = 0usize;
        let mut take = |len: usize| {
            let r: Vec<usize> = (next..next + len).collect();
            next += len;
            r
        };
        let exponent: Vec<Vec<usize>> = (0..dimension).map(|_| take(exponent_bits)).collect();
        let digits: Vec<Vec<usize>> = match schedule {
            ExpSchedule::Fibonacci => {
                let k = zeckendorf_len(exponent_bits);
                (0..dimension).map(|_| take(k)).collect()
            }
            _ => Vec::new(),
        };
        let registers = schedule.work_registers(exponent_bits);
        let work: Vec<Vec<usize>> = (0..registers).map(|_| take(w)).collect();
        let output = match schedule {
            ExpSchedule::Sequential => 0,
            ExpSchedule::Regev => registers - 1,
            ExpSchedule::Fibonacci => 2,
        };
        Layout {
            exponent,
            digits,
            work,
            output,
            qubits: next,
        }
    }
}

/// What one exponentiation actually cost.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Cost {
    /// Multiplications where both operands are full-width residues.
    pub full_multiplications: usize,
    /// Multiplications by one of the small primes `bᵢ`.
    pub small_multiplications: usize,
    /// Ops in the `d` inverse-QFT blocks. Regev pays the transform once
    /// per exponent register, so the banded form
    /// ([`Regev::qft_epsilon`]) saves `d ×` what it saves Shor.
    pub phase_gates: usize,
    /// Reversible Zeckendorf conversions (Fibonacci schedule only).
    pub digit_conversions: usize,
    /// `n`-bit work registers held at once.
    pub work_registers: usize,
    /// Register width the run allocated.
    pub qubits: usize,
    /// Largest support the state ever held.
    pub peak_support: usize,
    /// Largest reported representation size.
    pub peak_bytes: usize,
}

/// Apply `|z⟩|1⟩ ↦ |z⟩|∏ bᵢ^{zᵢ} mod N⟩` under `schedule`, leaving every
/// register other than the output back at `|0⟩`.
///
/// The work registers must start at `|0…0⟩`; the routine seeds and clears
/// them itself.
pub fn exponentiate<S: Scalar>(
    sim: &Simulator<S>,
    state: &mut dyn Backend<S>,
    plan: &ExpPlan,
) -> Result<Cost> {
    match plan.schedule {
        ExpSchedule::Sequential => exponentiate_sequential(sim, state, plan),
        ExpSchedule::Regev => exponentiate_regev(sim, state, plan),
        ExpSchedule::Fibonacci => exponentiate_fibonacci(sim, state, plan),
    }
}

/// Everything [`exponentiate`] needs that is not the state itself.
#[derive(Debug, Clone)]
pub struct ExpPlan {
    /// The modulus.
    pub modulus: u64,
    /// The bases `bᵢ = pᵢ²`.
    pub bases: Vec<u64>,
    /// Exponent-register width `R`.
    pub exponent_bits: usize,
    /// The schedule.
    pub schedule: ExpSchedule,
    /// The register layout.
    pub layout: Layout,
}

fn exponentiate_sequential<S: Scalar>(
    sim: &Simulator<S>,
    state: &mut dyn Backend<S>,
    plan: &ExpPlan,
) -> Result<Cost> {
    let n = plan.modulus;
    let work = &plan.layout.work[0];
    let mut cost = Cost {
        work_registers: 1,
        qubits: plan.layout.qubits,
        ..Cost::default()
    };
    sim.apply(state, "x", &[], &[work[0]])?;
    for (i, &b) in plan.bases.iter().enumerate() {
        for j in 0..plan.exponent_bits {
            let c = plan.layout.exponent[i][j];
            let support = controlled_mul_mod(state, Some(c), work, pow_mod(b, 1u64 << j, n), n)?;
            cost.full_multiplications += 1;
            cost.peak_support = cost.peak_support.max(support);
        }
    }
    Ok(cost)
}

fn exponentiate_regev<S: Scalar>(
    sim: &Simulator<S>,
    state: &mut dyn Backend<S>,
    plan: &ExpPlan,
) -> Result<Cost> {
    let n = plan.modulus;
    let r = plan.exponent_bits;
    let work = &plan.layout.work;
    let mut cost = Cost {
        work_registers: work.len(),
        qubits: plan.layout.qubits,
        ..Cost::default()
    };
    // work[0] is the seed A = 1; work[s+1] holds the bit plane R−1−s.
    sim.apply(state, "x", &[], &[work[0][0]])?;
    for s in 0..r {
        let j = r - 1 - s;
        let support = xor_square_into(state, &work[s], &work[s + 1], n)?;
        cost.full_multiplications += 1;
        cost.peak_support = cost.peak_support.max(support);
        for (i, &b) in plan.bases.iter().enumerate() {
            let c = plan.layout.exponent[i][j];
            let support = controlled_mul_mod(state, Some(c), &work[s + 1], b, n)?;
            cost.small_multiplications += 1;
            cost.peak_support = cost.peak_support.max(support);
        }
    }
    xor_copy(state, &work[r], &work[r + 1])?;
    // Uncompute, exactly backwards.
    for s in (0..r).rev() {
        let j = r - 1 - s;
        for (i, &b) in plan.bases.iter().enumerate().rev() {
            let inv = mod_inv(b % n, n).ok_or_else(|| {
                Error::InvalidState(format!("base {b} is not invertible mod {n}"))
            })?;
            let c = plan.layout.exponent[i][j];
            let support = controlled_mul_mod(state, Some(c), &work[s + 1], inv, n)?;
            cost.small_multiplications += 1;
            cost.peak_support = cost.peak_support.max(support);
        }
        let support = xor_square_into(state, &work[s], &work[s + 1], n)?;
        cost.full_multiplications += 1;
        cost.peak_support = cost.peak_support.max(support);
    }
    sim.apply(state, "x", &[], &[work[0][0]])?;
    Ok(cost)
}

fn exponentiate_fibonacci<S: Scalar>(
    sim: &Simulator<S>,
    state: &mut dyn Backend<S>,
    plan: &ExpPlan,
) -> Result<Cost> {
    let n = plan.modulus;
    let k = zeckendorf_len(plan.exponent_bits);
    let fibs = fibonacci(k + 1);
    let work = &plan.layout.work;
    let mut cost = Cost {
        work_registers: 3,
        qubits: plan.layout.qubits,
        ..Cost::default()
    };
    // Zeckendorf digits, reversibly, from the binary exponent registers.
    for (i, digits) in plan.layout.digits.iter().enumerate() {
        let exponent = plan.layout.exponent[i].clone();
        let digits = digits.clone();
        let f = fibs.clone();
        let support = shor::apply_permutation(state, "zeckendorf digits", &|idx| {
            let z = shor::gather(idx, &exponent);
            let e = shor::gather(idx, &digits);
            shor::scatter(idx, &digits, e ^ zeckendorf(z, &f))
        })?;
        cost.digit_conversions += 1;
        cost.peak_support = cost.peak_support.max(support);
    }
    // S and T start at 1; `roles` tracks which physical register is which.
    sim.apply(state, "x", &[], &[work[0][0]])?;
    sim.apply(state, "x", &[], &[work[1][0]])?;
    let (mut s_reg, mut t_reg) = (0usize, 1usize);
    for step in (1..=k).rev() {
        for (i, &b) in plan.bases.iter().enumerate() {
            let c = plan.layout.digits[i][step - 1];
            let support = controlled_mul_mod(state, Some(c), &work[t_reg], b, n)?;
            cost.small_multiplications += 1;
            cost.peak_support = cost.peak_support.max(support);
        }
        let support = mul_register_into(state, &work[s_reg], &work[t_reg], n, false)?;
        cost.full_multiplications += 1;
        cost.peak_support = cost.peak_support.max(support);
        std::mem::swap(&mut s_reg, &mut t_reg);
    }
    xor_copy(state, &work[s_reg], &work[2])?;
    // Uncompute the ladder.
    for step in 1..=k {
        std::mem::swap(&mut s_reg, &mut t_reg);
        let support = mul_register_into(state, &work[s_reg], &work[t_reg], n, true)?;
        cost.full_multiplications += 1;
        cost.peak_support = cost.peak_support.max(support);
        for (i, &b) in plan.bases.iter().enumerate().rev() {
            let inv = mod_inv(b % n, n).ok_or_else(|| {
                Error::InvalidState(format!("base {b} is not invertible mod {n}"))
            })?;
            let c = plan.layout.digits[i][step - 1];
            let support = controlled_mul_mod(state, Some(c), &work[t_reg], inv, n)?;
            cost.small_multiplications += 1;
            cost.peak_support = cost.peak_support.max(support);
        }
    }
    sim.apply(state, "x", &[], &[work[0][0]])?;
    sim.apply(state, "x", &[], &[work[1][0]])?;
    for (i, digits) in plan.layout.digits.iter().enumerate() {
        let exponent = plan.layout.exponent[i].clone();
        let digits = digits.clone();
        let f = fibs.clone();
        let support = shor::apply_permutation(state, "zeckendorf digits", &|idx| {
            let z = shor::gather(idx, &exponent);
            let e = shor::gather(idx, &digits);
            shor::scatter(idx, &digits, e ^ zeckendorf(z, &f))
        })?;
        cost.digit_conversions += 1;
        cost.peak_support = cost.peak_support.max(support);
    }
    Ok(cost)
}

// ── the sampling run ─────────────────────────────────────────────────

/// How the exponent registers are prepared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Superposition {
    /// A Hadamard layer: uniform over the box `[0, 2^R)^d`. What a
    /// circuit gives for free.
    Uniform,
    /// A discrete Gaussian of the given width, centred in the box,
    /// prepared by loading amplitudes. What Regev's analysis assumes —
    /// and *not* free on hardware, which is why the default is uniform.
    Gaussian(f64),
}

/// One run of the quantum step.
#[derive(Debug, Clone)]
pub struct Sample {
    /// The measured dual vector `w ∈ ℤ_{2^R}^d`.
    pub dual: Vec<u64>,
    /// The residue the work register collapsed to.
    pub work_value: u64,
    /// What the run cost.
    pub cost: Cost,
}

/// Regev's algorithm over one modulus.
#[derive(Debug, Clone)]
pub struct Regev {
    /// The modulus `N`.
    pub modulus: u64,
    /// Lattice dimension `d`. Regev takes `d ≈ √(log N)`.
    pub dimension: usize,
    /// Exponent-register width `R`.
    pub exponent_bits: usize,
    /// The primes `pᵢ`; the bases are `bᵢ = pᵢ²`.
    pub primes: Vec<u64>,
    /// Small primes skipped because they divide `N` — free factors, kept
    /// in the report so the quantum step is not credited with them.
    pub trivial_divisors: Vec<u64>,
    /// Exponentiation schedule.
    pub schedule: ExpSchedule,
    /// Exponent-register preparation.
    pub superposition: Superposition,
    /// Backend name. `"sparse"` by default: the schedules with many work
    /// registers are wide but never dense.
    pub backend: String,
    /// Embedding weight for the lattice step. `1` by default, which
    /// weighs a unit of congruence residue against a unit of vector
    /// length — the right objective, because the sampled duals are
    /// rounded and a true lattice vector's residue is `O(‖v‖₁/2)` rather
    /// than zero. See [`lattice::congruence_kernel`].
    pub lattice_weight: i64,
    /// Angle floor for the per-register inverse QFT. `0.0` is exact; a
    /// positive value uses the banded transform
    /// [`crate::library::aqft`].
    pub qft_epsilon: f64,
}

impl Regev {
    /// A default configuration for `n`: `d` small primes coprime to `N`,
    /// `R = ⌈log₂ N⌉` exponent bits, sequential schedule.
    pub fn new(modulus: u64, dimension: usize) -> Result<Self> {
        if !(2..=shor::MAX_MODULUS).contains(&modulus) {
            return Err(Error::InvalidState(format!(
                "modulus {modulus} outside [2, {}]",
                shor::MAX_MODULUS
            )));
        }
        if !(1..=MAX_DIMENSION).contains(&dimension) {
            return Err(Error::InvalidState(format!(
                "dimension {dimension} outside [1, {MAX_DIMENSION}]"
            )));
        }
        let (primes, trivial_divisors) = coprime_primes(modulus, dimension);
        if primes.len() < dimension {
            return Err(Error::InvalidState(format!(
                "{modulus} is too small to supply {dimension} coprime prime bases"
            )));
        }
        let r = work_bits(modulus);
        Ok(Regev {
            modulus,
            dimension,
            exponent_bits: r,
            primes,
            trivial_divisors,
            schedule: ExpSchedule::Sequential,
            superposition: Superposition::Uniform,
            backend: "sparse".to_string(),
            lattice_weight: 1,
            qft_epsilon: 0.0,
        })
    }

    /// Override the exponent-register width.
    pub fn with_exponent_bits(mut self, bits: usize) -> Self {
        self.exponent_bits = bits;
        self
    }

    /// Override the schedule.
    pub fn with_schedule(mut self, schedule: ExpSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Override the exponent-register preparation.
    pub fn with_superposition(mut self, superposition: Superposition) -> Self {
        self.superposition = superposition;
        self
    }

    /// Override the lattice embedding weight.
    pub fn with_lattice_weight(mut self, weight: i64) -> Self {
        self.lattice_weight = weight;
        self
    }

    /// Override the inverse-QFT angle floor (see
    /// [`qft_epsilon`](Regev::qft_epsilon)).
    pub fn with_qft_epsilon(mut self, epsilon: f64) -> Self {
        self.qft_epsilon = epsilon;
        self
    }

    /// Override the backend.
    pub fn on_backend(mut self, name: &str) -> Self {
        self.backend = name.to_string();
        self
    }

    /// The bases `bᵢ = pᵢ² mod N`.
    pub fn bases(&self) -> Vec<u64> {
        self.primes
            .iter()
            .map(|&p| (p * p) % self.modulus)
            .collect()
    }

    /// The register layout this configuration needs.
    pub fn layout(&self) -> Layout {
        Layout::new(
            self.schedule,
            self.dimension,
            self.exponent_bits,
            work_bits(self.modulus),
        )
    }

    /// One quantum sample: prepare, exponentiate, measure the work
    /// register, Fourier-transform each exponent register, measure.
    pub fn sample<S: Scalar>(&self, sim: &Simulator<S>, rng: &mut Prng) -> Result<Sample> {
        // One sample is one scope; see `shor::OrderFinder::estimate`.
        let _scope = crate::guard::enter();
        let layout = self.layout();
        if layout.qubits > MAX_WIDTH {
            return Err(Error::TooManyQubits {
                requested: layout.qubits,
                max: MAX_WIDTH,
            });
        }
        let mut state = sim.backends().create(&self.backend, layout.qubits)?;
        self.prepare_for(sim, state.as_mut(), &layout)?;
        let plan = ExpPlan {
            modulus: self.modulus,
            bases: self.bases(),
            exponent_bits: self.exponent_bits,
            schedule: self.schedule,
            layout: layout.clone(),
        };
        let mut cost = exponentiate(sim, state.as_mut(), &plan)?;
        cost.peak_bytes = cost.peak_bytes.max(state.memory_bytes());

        // Measure the work register: this is what collapses z onto a
        // coset of L, and it is the only measurement whose outcome the
        // algorithm throws away.
        let out = &layout.work[layout.output];
        let mut work_value = 0u64;
        for (k, &q) in out.iter().enumerate() {
            if state.measure(q, rng)? {
                work_value |= 1u64 << k;
            }
        }
        // Inverse QFT on each exponent register, then measure it.
        let mut wrapper: Circuit<S> = Circuit::new(layout.qubits);
        let transform = if self.qft_epsilon > 0.0 {
            library::aqft::<S>(self.exponent_bits, self.qft_epsilon)
        } else {
            library::qft::<S>(self.exponent_bits)
        };
        cost.phase_gates = transform.len() * layout.exponent.len();
        for reg in &layout.exponent {
            wrapper.append(&transform, reg);
        }
        let bound = wrapper.bind(sim.registry())?;
        bound.inverse().run(state.as_mut())?;
        cost.peak_support = cost.peak_support.max(state.nonzero_count());
        cost.peak_bytes = cost.peak_bytes.max(state.memory_bytes());

        let mut dual = Vec::with_capacity(self.dimension);
        for reg in &layout.exponent {
            let mut v = 0u64;
            for (k, &q) in reg.iter().enumerate() {
                if state.measure(q, rng)? {
                    v |= 1u64 << k;
                }
            }
            dual.push(v);
        }
        Ok(Sample {
            dual,
            work_value,
            cost,
        })
    }

    /// Prepare the exponent registers alone, without running anything
    /// else — the only step where [`Superposition`] variants differ, and
    /// so the one worth measuring on its own.
    pub fn prepare_for<S: Scalar>(
        &self,
        sim: &Simulator<S>,
        state: &mut dyn Backend<S>,
        layout: &Layout,
    ) -> Result<()> {
        match self.superposition {
            Superposition::Uniform => {
                for reg in &layout.exponent {
                    for &q in reg {
                        sim.apply(state, "h", &[], &[q])?;
                    }
                }
                Ok(())
            }
            Superposition::Gaussian(sigma) => {
                let span = 1u64 << self.exponent_bits;
                let centre = (span as f64 - 1.0) / 2.0;
                let mut entries: Vec<(u64, S)> = Vec::new();
                let mut norm = 0.0f64;
                let total = span.checked_pow(self.dimension as u32).ok_or_else(|| {
                    Error::InvalidState("Gaussian preparation: box too large".to_string())
                })?;
                for code in 0..total {
                    let mut rest = code;
                    let mut index = 0u64;
                    let mut exponent_sq = 0.0f64;
                    for reg in &layout.exponent {
                        let z = rest % span;
                        rest /= span;
                        let dz = z as f64 - centre;
                        exponent_sq += dz * dz;
                        index = shor::scatter(index, reg, z);
                    }
                    let amp = (-std::f64::consts::PI * exponent_sq / (sigma * sigma)).exp();
                    if amp > 1e-12 {
                        norm += amp * amp;
                        entries.push((index, S::from_re(amp)));
                    }
                }
                let scale = 1.0 / norm.sqrt();
                for e in entries.iter_mut() {
                    e.1 = e.1.scale(scale);
                }
                state.load(&entries)
            }
        }
    }
}

// ── classical recovery ───────────────────────────────────────────────

/// `∏ pᵢ^{vᵢ} mod N`, with negative exponents taken through modular
/// inverses. `None` when some `pᵢ` is not invertible — which cannot
/// happen for primes chosen coprime to `N`, and is reported rather than
/// assumed away.
pub fn evaluate(primes: &[u64], v: &[i64], n: u64) -> Option<u64> {
    let mut acc = 1u64 % n;
    for (&p, &e) in primes.iter().zip(v.iter()) {
        let base = if e >= 0 { p % n } else { mod_inv(p % n, n)? };
        acc = shor::mul_mod(acc, pow_mod(base, e.unsigned_abs(), n), n);
    }
    Some(acc)
}

/// The witness a candidate lattice vector produced.
#[derive(Debug, Clone)]
pub struct Witness {
    /// The lattice vector.
    pub vector: Vec<i64>,
    /// `u = ∏ pᵢ^{vᵢ} mod N`.
    pub root: u64,
    /// Whether `u² ≡ 1 (mod N)` — i.e. whether the vector really was in
    /// `L`, rather than a near miss that happened to split `N` anyway.
    pub square_root_of_one: bool,
    /// The split.
    pub factors: (u64, u64),
}

/// Test one candidate vector: does `∏ pᵢ^{vᵢ}` split `N`?
pub fn test_candidate(n: u64, primes: &[u64], v: &[i64]) -> Option<Witness> {
    let u = evaluate(primes, v, n)?;
    if u == 0 {
        return None;
    }
    let is_root = shor::mul_mod(u, u, n) == 1;
    for cand in [gcd(u + 1, n), gcd(u.wrapping_sub(1), n)] {
        if cand > 1 && cand < n {
            return Some(Witness {
                vector: v.to_vec(),
                root: u,
                square_root_of_one: is_root,
                factors: (cand.min(n / cand), cand.max(n / cand)),
            });
        }
    }
    None
}

/// The whole run: what was sampled, what the lattice gave back, and what
/// it cost.
#[derive(Debug, Clone)]
pub struct RegevReport {
    /// The modulus.
    pub modulus: u64,
    /// The primes used.
    pub primes: Vec<u64>,
    /// Small primes that divide `N`, noticed while choosing the bases.
    pub trivial_divisors: Vec<u64>,
    /// Lattice dimension.
    pub dimension: usize,
    /// Exponent-register width.
    pub exponent_bits: usize,
    /// The schedule.
    pub schedule: ExpSchedule,
    /// Every sample taken.
    pub samples: Vec<Sample>,
    /// Candidate lattice vectors tested.
    pub candidates: usize,
    /// The witness, when one was found.
    pub witness: Option<Witness>,
    /// The split.
    pub factors: Option<(u64, u64)>,
    /// Summed cost over every sample.
    pub cost: Cost,
}

impl Regev {
    /// Run the algorithm: `samples` quantum runs, then the lattice step.
    ///
    /// Regev takes `d + 4` samples; fewer leaves the kernel lattice too
    /// large, more only costs runs.
    pub fn run<S: Scalar>(
        &self,
        sim: &Simulator<S>,
        rng: &mut Prng,
        samples: usize,
    ) -> Result<RegevReport> {
        let _scope = crate::guard::enter();
        let mut report = RegevReport {
            modulus: self.modulus,
            primes: self.primes.clone(),
            trivial_divisors: self.trivial_divisors.clone(),
            dimension: self.dimension,
            exponent_bits: self.exponent_bits,
            schedule: self.schedule,
            samples: Vec::with_capacity(samples),
            candidates: 0,
            witness: None,
            factors: None,
            cost: Cost::default(),
        };
        for _ in 0..samples {
            let s = self.sample(sim, rng)?;
            report.cost.full_multiplications += s.cost.full_multiplications;
            report.cost.small_multiplications += s.cost.small_multiplications;
            report.cost.phase_gates += s.cost.phase_gates;
            report.cost.digit_conversions += s.cost.digit_conversions;
            report.cost.work_registers = s.cost.work_registers;
            report.cost.qubits = s.cost.qubits;
            report.cost.peak_support = report.cost.peak_support.max(s.cost.peak_support);
            report.cost.peak_bytes = report.cost.peak_bytes.max(s.cost.peak_bytes);
            report.samples.push(s);
        }
        let q = 1i64 << self.exponent_bits;
        let rows: Vec<Vec<i64>> = report
            .samples
            .iter()
            .map(|s| s.dual.iter().map(|&x| x as i64).collect())
            .collect();
        let basis = lattice::congruence_kernel(&rows, self.dimension, q, self.lattice_weight);
        let mut candidates = basis.clone();
        candidates.extend(lattice::short_combinations(&basis, 3, 2));
        // Prefer a witness that really is a square root of one — the
        // lattice did its job — over a short vector that happened to
        // share a factor with N. Both split it; only the first is
        // evidence the quantum step contributed anything, so the search
        // does not stop at the first split it stumbles into.
        let mut lucky: Option<Witness> = None;
        for v in &candidates {
            report.candidates += 1;
            match test_candidate(self.modulus, &self.primes, v) {
                Some(w) if w.square_root_of_one => {
                    report.factors = Some(w.factors);
                    report.witness = Some(w);
                    return Ok(report);
                }
                Some(w) => lucky = lucky.or(Some(w)),
                None => {}
            }
        }
        if let Some(w) = lucky {
            report.factors = Some(w.factors);
            report.witness = Some(w);
        }
        Ok(report)
    }
}

/// Factor `n` with Regev's algorithm at dimension `d`, taking `d + 4`
/// samples — the default configuration, for callers that do not want to
/// choose a schedule.
pub fn factor<S: Scalar>(
    sim: &Simulator<S>,
    n: u64,
    dimension: usize,
    rng: &mut Prng,
) -> Result<RegevReport> {
    let regev = Regev::new(n, dimension)?;
    let samples = dimension + 4;
    regev.run(sim, rng, samples)
}
