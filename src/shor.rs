//! Shor's algorithm, resource-honest: the arithmetic is a permutation,
//! the interference is the cost.
//!
//! Factoring is usually demonstrated by compiling modular exponentiation
//! into ripple-carry adders and running the result on a dense state
//! vector. That measures the *compiler*, not the algorithm. This module
//! takes the decomposition the structure actually offers:
//!
//! * **Modular exponentiation is a basis permutation.** `|y⟩ ↦ |a·y mod N⟩`
//!   sends each basis index to one other basis index. Applied to a state
//!   with `k` nonzero amplitudes it costs `O(k)` — no matrix, no ancillas,
//!   no width penalty. [`controlled_mul_mod`] is that map, applied straight
//!   to any [`Backend`] through [`apply_permutation`], and it *refuses* a
//!   non-injective map by name rather than silently dropping amplitude.
//! * **The control register collapses to one recycled qubit.** In the
//!   semiclassical (Griffiths–Niu / Kitaev) form the inverse QFT is
//!   replaced by measurement plus a classical feedback rotation, so the
//!   `2n+1` control qubits become **one** ancilla measured and reset once
//!   per phase bit. [`PhaseForm::Semiclassical`] runs at width `w + 1`;
//!   [`PhaseForm::FullRegister`] is the textbook form kept as the
//!   cross-check reference, at width `w + t`.
//! * **What is left is the interference, and it is not free.** The work
//!   register only ever holds powers of `a`, so its support is bounded by
//!   the orbit `|⟨a⟩| = r` — the very number being computed.
//!
//! ## The measured law
//!
//! Two numbers, and they are not the same number:
//!
//! * **The orbit is exactly `r`.** The work register's distinct-value
//!   count ([`PhaseEstimate::peak_orbit`]) saturates at the multiplicative
//!   order of the base, at every width, in every case measured — `w = 4`
//!   through `w = 20`.
//! * **The support is `r` for even `r` and `2r` for odd `r`** — exactly,
//!   in 51 of 51 `(N, a)` pairs measured. Support counts *amplitudes*:
//!   mid-round the ancilla is in superposition, so `|y, 0⟩` and
//!   `|a^{2^k}·y, 1⟩` are two entries, and since `a^{2^k} ∈ ⟨a⟩` the
//!   multiplier never enlarges the set of reachable residues — it doubles
//!   the count of pairs.
//!
//!   The mechanism is the 2-adic valuation of `r`. The ladder's
//!   multipliers are `a^{2^k mod r}`, which for large `k` generate
//!   `⟨a^{2^{v₂(r)}}⟩` — the odd part of the orbit, of size
//!   `r / 2^{v₂(r)}`. The orbit sits on that plateau for most of the run
//!   and doubles on each of the last `v₂(r)` rounds
//!   ([`PhaseEstimate::orbit_trajectory`] shows it). So `r` odd means the
//!   plateau is already `r` and every later round doubles the pair count,
//!   giving `2r`; `r` even means the orbit only reaches `r` on the final
//!   round, which nothing doubles, giving `r`.
//!
//!
//! Neither is a function of the register width `w`. A 4-bit modulus and a
//! 20-bit modulus with the same order cost the same. That is the honest
//! statement of where Shor's quantum content lives: not in the arithmetic,
//! which a permutation kernel carries at any width with support 1 until
//! the first Hadamard, but in the period itself. Since `r` is typically
//! `Θ(N)`, the simulation is exponential in `log N` — as it must be, or
//! the algorithm would not be worth building hardware for.
//!
//! ## What this module does not claim
//!
//! At the widths that fit a simulator, trial division factors `N` outright
//! and does so faster than any of this. [`factor`] therefore reports the
//! *route* it took, and the classical shortcuts (even `N`, perfect powers,
//! a lucky `gcd`) are taken and named rather than skipped for effect. The
//! deliverable is the circuit's structure and its measured cost law, not a
//! race against `u64` arithmetic.
//!
//! ```
//! use quantsim::prelude::*;
//! use quantsim::shor;
//!
//! let sim: Simulator = Simulator::new();
//! let mut rng = Prng::new(7);
//! let report = shor::factor(&sim, 21, shor::PhaseForm::Semiclassical, &mut rng, 12)?;
//! let (p, q) = report.factors.expect("21 factors");
//! assert_eq!(p * q, 21);
//! # Ok::<(), quantsim::Error>(())
//! ```

use std::f64::consts::PI;

use crate::backend::Backend;
use crate::circuit::Circuit;
use crate::error::{Error, Result};
use crate::library;
use crate::math::GateMatrix;
use crate::padic::gcd;
use crate::rng::Prng;
use crate::scalar::{Scalar, C64};
use crate::sim::Simulator;

/// Largest modulus this module will accept, and the bound is structural
/// rather than chosen: products are taken in `u128`, so the arithmetic is
/// exact to `2^64`, and the binding constraint is the register — a
/// `w`-bit work register plus one phase ancilla must fit the 63-qubit
/// `u64` basis index, so `w ≤ 62`.
pub const MAX_MODULUS: u64 = 1 << 62;

/// Most phase bits a run can take: the ladder computes `2^{t-1}` as a
/// `u64` exponent.
pub const MAX_PHASE_BITS: usize = 63;

// ── modular arithmetic, kept explicit ────────────────────────────────

/// `a·b mod m`, widened so the product cannot wrap.
pub fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// `base^exp mod m` by square-and-multiply.
pub fn pow_mod(base: u64, exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let mut acc = 1u64;
    let mut b = base % m;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            acc = mul_mod(acc, b, m);
        }
        b = mul_mod(b, b, m);
        e >>= 1;
    }
    acc
}

/// Multiplicative order of `a` mod `m` by direct iteration — the classical
/// answer the quantum estimate is checked against. `None` when
/// `gcd(a, m) ≠ 1`, and capped at `m` steps.
pub fn multiplicative_order(a: u64, m: u64) -> Option<u64> {
    if m < 2 || gcd(a % m, m) != 1 {
        return None;
    }
    let mut x = a % m;
    let mut r = 1u64;
    while x != 1 {
        x = mul_mod(x, a, m);
        r += 1;
        if r > m {
            return None;
        }
    }
    Some(r)
}

/// Bits needed to hold every residue mod `m`, i.e. `⌈log₂ m⌉` rounded up to
/// admit `m − 1`.
pub fn work_bits(m: u64) -> usize {
    (64 - (m - 1).leading_zeros()) as usize
}

/// Trial-division primality, honest about its cost: `O(√m)`, which at the
/// widths this module simulates is cheaper than everything else here.
pub fn is_prime(m: u64) -> bool {
    if m < 2 {
        return false;
    }
    if m % 2 == 0 {
        return m == 2;
    }
    let mut d = 3u64;
    while d.saturating_mul(d) <= m {
        if m % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// `m = b^k` for some `k ≥ 2`? Returns `(b, k)` for the smallest such `b`.
pub fn perfect_power(m: u64) -> Option<(u64, u32)> {
    if m < 4 {
        return None;
    }
    let max_k = 64 - m.leading_zeros();
    for k in 2..=max_k {
        let root = (m as f64).powf(1.0 / k as f64).round() as u64;
        for b in root.saturating_sub(1)..=root + 1 {
            if b < 2 {
                continue;
            }
            let mut acc = 1u64;
            let mut overflow = false;
            for _ in 0..k {
                match acc.checked_mul(b) {
                    Some(v) => acc = v,
                    None => {
                        overflow = true;
                        break;
                    }
                }
            }
            if !overflow && acc == m {
                return Some((b, k));
            }
        }
    }
    None
}

// ── permutation kernels ──────────────────────────────────────────────

/// Read the value held by `qubits` out of a basis index: `qubits[k]` is
/// bit `k` of the value.
pub fn gather(index: u64, qubits: &[usize]) -> u64 {
    let mut v = 0u64;
    for (k, &q) in qubits.iter().enumerate() {
        if (index >> q) & 1 == 1 {
            v |= 1u64 << k;
        }
    }
    v
}

/// Write `value` into the `qubits` positions of a basis index, leaving
/// every other bit alone. Inverse of [`gather`] on those positions.
pub fn scatter(index: u64, qubits: &[usize], value: u64) -> u64 {
    let mut out = index;
    for (k, &q) in qubits.iter().enumerate() {
        let bit = (value >> k) & 1;
        out = (out & !(1u64 << q)) | (bit << q);
    }
    out
}

/// Apply a basis permutation to a state in place, in `O(support)`.
///
/// This is the operation a `GateMatrix` cannot express without paying
/// `4^n`: `map` sends each occupied basis index to its image, amplitudes
/// ride along unchanged. Injectivity is *checked* — a map that collides
/// is not unitary, and is rejected as [`Error::InvalidState`] naming the
/// two colliding indices rather than quietly losing weight.
///
/// Returns the resulting support (number of nonzero amplitudes).
pub fn apply_permutation<S: Scalar>(
    state: &mut dyn Backend<S>,
    label: &str,
    map: &dyn Fn(u64) -> u64,
) -> Result<usize> {
    // The rewrite holds the support, so it is admitted like every other
    // exponential kernel in the crate rather than trusted to be small
    // because the algorithm says so — but the check is only taken where
    // it can bind.
    //
    // With no explicit limit configured, `guard::admit` measures
    // availability by reading `/proc/meminfo` and the cgroup files on
    // every call: **13.5 µs**, against 4 ns when a limit is set and the
    // budget is an atomic load. One call per controlled multiplication
    // is `2w+1` of those per order-finding run, which at small support
    // costs 40× the work it is guarding and was the whole of an
    // unexplained fixed overhead. So: always check under a configured
    // limit, where the check is free, and otherwise only above a floor
    // no plausible budget would refuse.
    const ADMIT_FLOOR: usize = 1 << 20;
    let entry = std::mem::size_of::<(u64, S)>();
    let bytes = 3usize
        .saturating_mul(state.nonzero_count())
        .saturating_mul(entry);
    if bytes >= ADMIT_FLOOR || crate::guard::memory_limit().is_some() {
        crate::guard::admit(bytes, label)?;
    }
    state.apply_permutation(label, map)?;
    Ok(state.nonzero_count())
}

/// The controlled modular multiplier `|c⟩|y⟩ ↦ |c⟩|a·y mod N⟩` (acting
/// only when `c = 1`, and only on `y < N` so the map stays a permutation
/// of the whole register).
///
/// `control = None` applies it unconditionally. `work[k]` is bit `k` of
/// the residue. One call is one *modular multiplication* of circuit
/// resource — the unit Shor and Regev are counted in — and `O(support)`
/// of simulator work.
pub fn controlled_mul_mod<S: Scalar>(
    state: &mut dyn Backend<S>,
    control: Option<usize>,
    work: &[usize],
    multiplier: u64,
    modulus: u64,
) -> Result<usize> {
    if !(2..=MAX_MODULUS).contains(&modulus) {
        return Err(Error::InvalidState(format!(
            "modulus {modulus} outside [2, {MAX_MODULUS}]"
        )));
    }
    if work.len() < work_bits(modulus) {
        return Err(Error::InvalidState(format!(
            "work register of {} qubits cannot hold residues mod {modulus}",
            work.len()
        )));
    }
    let a = multiplier % modulus;
    if gcd(a, modulus) != 1 {
        return Err(Error::InvalidState(format!(
            "multiplier {a} shares a factor with {modulus}, so |y⟩ ↦ |a·y⟩ is not injective"
        )));
    }
    let n = state.num_qubits();
    for &q in work.iter().chain(control.iter()) {
        if q >= n {
            return Err(Error::QubitOutOfRange {
                qubit: q,
                num_qubits: n,
            });
        }
    }
    apply_permutation(state, "controlled modular multiplication", &|i| {
        if let Some(c) = control {
            if (i >> c) & 1 == 0 {
                return i;
            }
        }
        let y = gather(i, work);
        if y >= modulus {
            return i;
        }
        scatter(i, work, mul_mod(a, y, modulus))
    })
}

/// Multiply one register into another: `|x⟩|y⟩ ↦ |x·y mod N⟩|y⟩`.
///
/// A permutation whenever `gcd(y, N) = 1` — and the identity on the rows
/// where it is not, which keeps the whole map injective. This is the
/// register-by-register multiplication that
/// [`regev::ExpSchedule::Fibonacci`](crate::regev::ExpSchedule::Fibonacci)
/// needs and that squaring cannot supply: `y ↦ y²` is not injective mod
/// `N`, which is exactly why Regev's own circuit has to keep its
/// intermediates.
/// `inverse` runs it backwards, `|x⟩|y⟩ ↦ |x·y⁻¹ mod N⟩|y⟩`, which is
/// what an uncomputation pass needs.
pub fn mul_register_into<S: Scalar>(
    state: &mut dyn Backend<S>,
    target: &[usize],
    source: &[usize],
    modulus: u64,
    inverse: bool,
) -> Result<usize> {
    if !(2..=MAX_MODULUS).contains(&modulus) {
        return Err(Error::InvalidState(format!(
            "modulus {modulus} outside [2, {MAX_MODULUS}]"
        )));
    }
    apply_permutation(state, "register-register modular multiplication", &|i| {
        let y = gather(i, source);
        let x = gather(i, target);
        if y >= modulus || x >= modulus || gcd(y, modulus) != 1 {
            return i;
        }
        let factor = if inverse {
            match crate::padic::mod_inv(y, modulus) {
                Some(v) => v,
                None => return i,
            }
        } else {
            y
        };
        scatter(i, target, mul_mod(x, factor, modulus))
    })
}

/// `|A⟩|y⟩ ↦ |A⟩|y ⊕ (A² mod N)⟩` — out-of-place modular squaring.
///
/// Squaring is *not* injective mod `N`, so it cannot be done in place;
/// XOR-ing into a second register is the standard reversible workaround,
/// and it is an involution, so the same call undoes it. This is the
/// operation whose per-step fresh register gives Regev's schedule its
/// `Õ(n^{3/2})` qubit count.
pub fn xor_square_into<S: Scalar>(
    state: &mut dyn Backend<S>,
    source: &[usize],
    target: &[usize],
    modulus: u64,
) -> Result<usize> {
    apply_permutation(state, "out-of-place modular squaring", &|i| {
        let a = gather(i, source);
        if a >= modulus {
            return i;
        }
        let y = gather(i, target);
        scatter(i, target, y ^ mul_mod(a, a, modulus))
    })
}

/// `|x⟩|y⟩ ↦ |x⟩|y ⊕ x⟩` on two equal-width registers: the reversible
/// copy that lets a result survive its own uncomputation.
pub fn xor_copy<S: Scalar>(
    state: &mut dyn Backend<S>,
    source: &[usize],
    target: &[usize],
) -> Result<usize> {
    apply_permutation(state, "reversible copy", &|i| {
        let x = gather(i, source);
        let y = gather(i, target);
        scatter(i, target, y ^ x)
    })
}

// ── the ancilla's fused gates ────────────────────────────────────────

fn two_by_two<S: Scalar>(entries: [C64; 4], label: &str) -> Result<GateMatrix<S>> {
    let mut data = Vec::with_capacity(4);
    for e in entries {
        data.push(
            S::try_from_c64(e).ok_or_else(|| Error::UnsupportedForAlgebra {
                gate: label.to_string(),
                algebra: S::algebra_name(),
            })?,
        );
    }
    GateMatrix::from_vec(2, data)
}

/// `H`, built once instead of resolved from the registry per round.
fn hadamard<S: Scalar>() -> Result<GateMatrix<S>> {
    let k = std::f64::consts::FRAC_1_SQRT_2;
    two_by_two(
        [
            C64::new(k, 0.0),
            C64::new(k, 0.0),
            C64::new(k, 0.0),
            C64::new(-k, 0.0),
        ],
        "h",
    )
}

/// `H·X` — the ancilla reset folded into the next round's Hadamard, so a
/// measured 1 costs no extra pass over the support.
fn hadamard_after_flip<S: Scalar>() -> Result<GateMatrix<S>> {
    let k = std::f64::consts::FRAC_1_SQRT_2;
    two_by_two(
        [
            C64::new(k, 0.0),
            C64::new(k, 0.0),
            C64::new(-k, 0.0),
            C64::new(k, 0.0),
        ],
        "h·x",
    )
}

/// `H·P(θ)` — the feedback rotation and the closing Hadamard as one
/// matrix. Two passes over the support become one, and the profile in
/// `examples/factoring_scale.rs` is why.
fn hadamard_after_phase<S: Scalar>(theta: f64) -> Result<GateMatrix<S>> {
    let k = std::f64::consts::FRAC_1_SQRT_2;
    let e = crate::math::cis(theta);
    two_by_two(
        [
            C64::new(k, 0.0),
            C64::new(k * e.re, k * e.im),
            C64::new(k, 0.0),
            C64::new(-k * e.re, -k * e.im),
        ],
        "h·p",
    )
}

// ── phase estimation ─────────────────────────────────────────────────

/// How the phase register is realized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseForm {
    /// One ancilla, measured and reset once per phase bit, with the
    /// inverse QFT replaced by classical feedback rotations
    /// (Griffiths–Niu). Width `w + 1`.
    Semiclassical,
    /// The textbook `t`-qubit control register and an inverse QFT.
    /// Width `w + t`; kept as the cross-check reference.
    FullRegister,
}

impl std::fmt::Display for PhaseForm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.pad(match self {
            PhaseForm::Semiclassical => "semiclassical",
            PhaseForm::FullRegister => "full-register",
        })
    }
}

/// What one phase-estimation run cost and what it returned.
#[derive(Debug, Clone)]
pub struct PhaseEstimate {
    /// The estimate `ω = 0.b₁b₂…b_t ∈ [0, 1)` of `s/r`.
    pub phase: f64,
    /// The same number as an integer: `ω · 2^t`.
    pub value: u64,
    /// `b₁ … b_t`, most significant first.
    pub bits: Vec<bool>,
    /// Phase bits requested.
    pub phase_bits: usize,
    /// Register width the run actually used.
    pub qubits: usize,
    /// Controlled modular multiplications applied — the circuit's cost
    /// in the unit that survives compilation.
    pub modular_multiplications: usize,
    /// Phase-rotation ops the form spent: the inverse-QFT block for
    /// [`PhaseForm::FullRegister`] (what [`OrderFinder::qft_epsilon`]
    /// trades against accuracy), and the feedback rotations on the single
    /// ancilla for [`PhaseForm::Semiclassical`]. The comparison is the
    /// point — `t(t+1)/2` controlled phases against at most `t`
    /// uncontrolled ones.
    pub phase_gates: usize,
    /// Largest number of nonzero amplitudes held at any point.
    ///
    /// Amplitudes, not residues: mid-round the ancilla is in
    /// superposition, so `|y, 0⟩` and `|a^{2^k}·y, 1⟩` are two entries
    /// even though `a^{2^k}·y` is in the same subgroup `⟨a⟩`. The
    /// multiplier never enlarges the *set* of reachable residues — it
    /// cannot, being itself a power of `a` — it doubles the count of
    /// *pairs*.
    ///
    /// Hence, exactly:
    ///
    /// ```text
    /// peak_support = 2 · max over rounds that ran   of orbit_trajectory
    /// peak_orbit   =     max over all entries       of orbit_trajectory
    /// ```
    ///
    /// The final entry is the orbit after the last measurement, which no
    /// round doubles, so the two coincide precisely when the orbit first
    /// fills there. Measured, that is decided by the parity of `r`:
    /// `peak_support = r` for even `r`, `2r` for odd `r`, 51/51. Nothing
    /// about the width enters either side.
    pub peak_support: usize,
    /// Largest `memory_bytes` the representation reported — the number
    /// that actually differs between backends, since support is a
    /// representation-independent count.
    pub peak_bytes: usize,
    /// Largest number of distinct residues the work register ever held —
    /// the orbit actually reached. Measured equal to `r` at every width
    /// tried, and never a function of the width.
    pub peak_orbit: usize,
    /// Distinct residues the work register held entering each round,
    /// starting at 1. The ancilla is reset (or its reset folded into the
    /// next round's Hadamard), so every surviving index carries the same
    /// ancilla bit and this count *is* the support at those moments.
    ///
    /// This is what makes the `r` / `2r` split mechanical rather than a
    /// story: see [`PhaseEstimate::peak_support`].
    pub orbit_trajectory: Vec<usize>,
}

/// Order finding by phase estimation over `|y⟩ ↦ |a·y mod N⟩`.
#[derive(Debug, Clone)]
pub struct OrderFinder {
    /// The modulus `N`.
    pub modulus: u64,
    /// The base `a`, coprime to `N`.
    pub base: u64,
    /// Phase bits `t`. Default `2⌈log₂N⌉ + 1`, which makes the
    /// continued-fraction recovery of `r < N` unambiguous.
    pub phase_bits: usize,
    /// Backend name. Defaults to `"sparse"`, which is the representation
    /// the support law rewards; the full-register form is better served
    /// by `"dense"`.
    pub backend: String,
    /// Angle floor for the inverse QFT of
    /// [`PhaseForm::FullRegister`]. `0.0` is the exact transform; a
    /// positive value uses the banded approximation
    /// [`crate::library::aqft`] — Coppersmith's
    /// construction, whose surviving coupling count
    /// [`radixweb`](crate::radixweb) derives and verifies — cutting the
    /// controlled-phase count from `t(t+1)/2` to `O(t·log(1/ε))`.
    ///
    /// It does not change the semiclassical form, whose corresponding
    /// band is a rounding of the feedback accumulator rather than a
    /// pruned triangle.
    pub qft_epsilon: f64,
}

impl OrderFinder {
    /// Order finder for `base` mod `modulus`, with default phase bits.
    pub fn new(modulus: u64, base: u64) -> Result<Self> {
        if !(2..=MAX_MODULUS).contains(&modulus) {
            return Err(Error::InvalidState(format!(
                "modulus {modulus} outside [2, {MAX_MODULUS}]"
            )));
        }
        if gcd(base % modulus, modulus) != 1 {
            return Err(Error::InvalidState(format!(
                "base {base} is not coprime to {modulus}; gcd already factors it"
            )));
        }
        let w = work_bits(modulus);
        if w + 1 > 63 {
            return Err(Error::TooManyQubits {
                requested: w + 1,
                max: 63,
            });
        }
        // `2w+1` bits resolve any order below `N`; past `MAX_PHASE_BITS`
        // that is unrepresentable, and unnecessary — `t` bits resolve
        // orders to about `2^{t/2}`, so 63 covers `r` up to ~2^31, far
        // past what the support (which *is* `r`) can be held in.
        let t = (2 * w + 1).min(MAX_PHASE_BITS);
        Ok(OrderFinder {
            modulus,
            base: base % modulus,
            phase_bits: t,
            backend: "sparse".to_string(),
            qft_epsilon: 0.0,
        })
    }

    /// Override the inverse-QFT angle floor (see
    /// [`qft_epsilon`](OrderFinder::qft_epsilon)).
    pub fn with_qft_epsilon(mut self, epsilon: f64) -> Self {
        self.qft_epsilon = epsilon;
        self
    }

    /// Override the phase-bit count.
    pub fn with_phase_bits(mut self, bits: usize) -> Self {
        self.phase_bits = bits;
        self
    }

    /// Override the backend the run allocates.
    pub fn on_backend(mut self, name: &str) -> Self {
        self.backend = name.to_string();
        self
    }

    /// Work-register width `w = ⌈log₂ N⌉`.
    pub fn work_bits(&self) -> usize {
        work_bits(self.modulus)
    }

    /// Total register width for a form — the whole point of the
    /// semiclassical variant is that this is `w + 1` rather than `w + t`.
    pub fn width(&self, form: PhaseForm) -> usize {
        match form {
            PhaseForm::Semiclassical => self.work_bits() + 1,
            PhaseForm::FullRegister => self.work_bits() + self.phase_bits,
        }
    }

    /// Run one phase estimation.
    pub fn estimate<S: Scalar>(
        &self,
        sim: &Simulator<S>,
        form: PhaseForm,
        rng: &mut Prng,
    ) -> Result<PhaseEstimate> {
        match form {
            PhaseForm::Semiclassical => self.estimate_semiclassical(sim, rng),
            PhaseForm::FullRegister => self.estimate_full_register(sim, rng),
        }
    }

    fn estimate_semiclassical<S: Scalar>(
        &self,
        sim: &Simulator<S>,
        rng: &mut Prng,
    ) -> Result<PhaseEstimate> {
        // One run is one scope: a configured time budget has to bound the
        // whole ladder, not each gate inside it, or a long permutation
        // kernel runs unbounded between two checks that both pass.
        let _scope = crate::guard::enter();
        let w = self.work_bits();
        let t = self.phase_bits;
        if t == 0 || t > MAX_PHASE_BITS {
            return Err(Error::InvalidState(format!(
                "phase_bits {t} outside [1, {MAX_PHASE_BITS}]"
            )));
        }
        let work: Vec<usize> = (0..w).collect();
        let anc = w;
        let mut state = sim.backends().create(&self.backend, w + 1)?;
        // |1⟩ in the work register: the cyclic group's identity, whose
        // orbit under multiplication by `a` is exactly ⟨a⟩.
        sim.apply(state.as_mut(), "x", &[], &[work[0]])?;

        let mut omega = 0.0f64;
        let mut measured: Vec<bool> = Vec::with_capacity(t);
        let mut peak = state.nonzero_count();
        let mut bytes = state.memory_bytes();
        let mut orbit = 1usize;
        let mut trajectory = vec![1usize];
        let mut muls = 0usize;
        let mut rotations = 0usize;
        let h = hadamard::<S>()?;
        let hx = hadamard_after_flip::<S>()?;
        let mut pending_reset = false;
        for j in (1..=t).rev() {
            // Opening Hadamard, with any pending ancilla reset folded in.
            state.apply(if pending_reset { &hx } else { &h }, &[anc])?;
            let a_pow = pow_mod(self.base, 1u64 << (j - 1), self.modulus);
            let support =
                controlled_mul_mod(state.as_mut(), Some(anc), &work, a_pow, self.modulus)?;
            muls += 1;
            peak = peak.max(support);
            // Feedback and closing Hadamard as one matrix: strip the bits
            // already known, leaving ±1 on the ancilla. ω here is
            // 0.b_{j+1}…b_t, so the correction is −πω.
            state.apply(&hadamard_after_phase::<S>(-PI * omega)?, &[anc])?;
            if omega > 0.0 {
                rotations += 1;
            }
            let bit = state.measure(anc, rng)?;
            pending_reset = bit;
            measured.push(bit);
            omega = (if bit { 1.0 } else { 0.0 } + omega) * 0.5;
            // Every surviving index now carries the same ancilla bit, so
            // the count of distinct residues *is* the support — no second
            // pass, and `tests/shor.rs` pins the identity.
            let support = state.nonzero_count();
            peak = peak.max(support);
            bytes = bytes.max(state.memory_bytes());
            orbit = orbit.max(support);
            trajectory.push(support);
        }
        measured.reverse(); // b₁ … b_t
        let mut value = 0u64;
        for (k, &b) in measured.iter().enumerate() {
            if b {
                value |= 1u64 << (t - 1 - k);
            }
        }
        Ok(PhaseEstimate {
            phase: omega,
            value,
            bits: measured,
            phase_bits: t,
            qubits: w + 1,
            modular_multiplications: muls,
            phase_gates: rotations,
            peak_support: peak,
            peak_bytes: bytes,
            peak_orbit: orbit,
            orbit_trajectory: trajectory,
        })
    }

    fn estimate_full_register<S: Scalar>(
        &self,
        sim: &Simulator<S>,
        rng: &mut Prng,
    ) -> Result<PhaseEstimate> {
        let _scope = crate::guard::enter();
        let w = self.work_bits();
        let t = self.phase_bits;
        if t == 0 || t > MAX_PHASE_BITS {
            return Err(Error::InvalidState(format!(
                "phase_bits {t} outside [1, {MAX_PHASE_BITS}]"
            )));
        }
        let work: Vec<usize> = (0..w).collect();
        let controls: Vec<usize> = (w..w + t).collect();
        let mut state = sim.backends().create(&self.backend, w + t)?;
        sim.apply(state.as_mut(), "x", &[], &[work[0]])?;
        for &c in &controls {
            sim.apply(state.as_mut(), "h", &[], &[c])?;
        }
        let mut peak = state.nonzero_count();
        let mut bytes = state.memory_bytes();
        let mut orbit = 1usize;
        let mut muls = 0usize;
        for (k, &c) in controls.iter().enumerate() {
            let a_pow = pow_mod(self.base, 1u64 << k, self.modulus);
            let support = controlled_mul_mod(state.as_mut(), Some(c), &work, a_pow, self.modulus)?;
            muls += 1;
            peak = peak.max(support);
            bytes = bytes.max(state.memory_bytes());
            orbit = orbit.max(distinct_values(state.as_ref(), &work));
        }
        // Inverse QFT on the control register only, exact or banded.
        let mut wrapper: Circuit<S> = Circuit::new(w + t);
        let transform = if self.qft_epsilon > 0.0 {
            library::aqft::<S>(t, self.qft_epsilon)
        } else {
            library::qft::<S>(t)
        };
        let phase_gates = transform.len();
        wrapper.append(&transform, &controls);
        let bound = wrapper.bind(sim.registry())?;
        bound.inverse().run(state.as_mut())?;
        peak = peak.max(state.nonzero_count());
        bytes = bytes.max(state.memory_bytes());

        let mut value = 0u64;
        for (k, &c) in controls.iter().enumerate() {
            if state.measure(c, rng)? {
                value |= 1u64 << k;
            }
        }
        let bits: Vec<bool> = (0..t).map(|k| (value >> (t - 1 - k)) & 1 == 1).collect();
        Ok(PhaseEstimate {
            phase: value as f64 / (1u64 << t) as f64,
            value,
            bits,
            phase_bits: t,
            qubits: w + t,
            modular_multiplications: muls,
            phase_gates,
            peak_support: peak,
            peak_bytes: bytes,
            peak_orbit: orbit,
            orbit_trajectory: Vec::new(),
        })
    }
}

/// Predicted operator width of the exponentiation ladder, across every
/// nontrivial factorization cut of `N`, from
/// [`modwidth`](crate::modwidth) — the whole cost curve of
/// `|y⟩ ↦ |a^{2^k}·y mod N⟩` computed before any state exists.
///
/// Returns one entry per ladder step: the classical multiplier
/// `a^{2^k} mod N` and its width across each cut `N = l·s`.
///
/// **The scope, because it is narrower than it looks.** `modwidth` prices
/// the map on a *ℤ_N-native* register, where the cuts are the divisors of
/// `N`. The register this module actually simulates is `w = ⌈log₂N⌉`
/// qubits, so it carries the same permutation with an identity block on
/// `[N, 2^w)` — a different operator across a binary cut whenever
/// `N ≠ 2^w`, which for an odd `N` is always. So this is a forecast for
/// the qudit form ([`mixed::CompoundRegister`](crate::mixed)), and an
/// upper-bound intuition for the binary one. It is *not* what sets the
/// measured cost here: on the sparse representation the cost is the
/// orbit, and the orbit is `r` whatever the cut rank says.
///
/// What it does say, exactly, is `modwidth`'s own result: `a^{2^k}` cycles
/// with the multiplicative order of `2` mod `r`, so the width curve
/// cycles with it and modular exponentiation is *periodic in width* —
/// bounded, not growing, along the ladder.
pub fn ladder_widths(modulus: u64, base: u64, phase_bits: usize) -> Vec<(u64, Vec<u128>)> {
    let cuts: Vec<(u128, u128)> = (2..modulus)
        .filter(|d| modulus % d == 0)
        .map(|d| (d as u128, (modulus / d) as u128))
        .collect();
    (0..phase_bits)
        .map(|k| {
            let m = pow_mod(base, 1u64 << k, modulus);
            let widths = cuts
                .iter()
                .map(|&(l, s)| crate::modwidth::mult_cut_width(m as u128, l, s).value())
                .collect();
            (m, widths)
        })
        .collect()
}

/// How many distinct values the `qubits` sub-register holds across the
/// state's support — the orbit size, read off rather than assumed.
pub fn distinct_values<S: Scalar>(state: &dyn Backend<S>, qubits: &[usize]) -> usize {
    let mut seen: rustc_hash::FxHashSet<u64> = rustc_hash::FxHashSet::default();
    state.for_each_nonzero(&mut |i, _| {
        seen.insert(gather(i, qubits));
    });
    seen.len()
}

// ── continued-fraction recovery ──────────────────────────────────────

/// Recover the order from a phase estimate: the smallest `r ≤ max_order`
/// with `base^r ≡ 1 (mod modulus)` among the convergents of `phase`
/// ([`padic::convergents`](crate::padic::convergents)) and their small
/// multiples. A convergent whose denominator is a proper divisor of `r`
/// is the common near miss, which is what the multiples loop catches.
pub fn order_from_phase(phase: f64, modulus: u64, base: u64, max_order: u64) -> Option<u64> {
    if phase <= 0.0 || phase.is_nan() {
        return None;
    }
    for (_, den) in crate::padic::convergents(phase, max_order) {
        let mut r = den;
        while r <= max_order {
            if pow_mod(base, r, modulus) == 1 {
                return Some(r);
            }
            r += den;
        }
    }
    None
}

// ── the classical reduction ──────────────────────────────────────────

/// Split `n` from a base and its order, the standard reduction: an even
/// order whose half-power is not `−1` gives `gcd(a^{r/2} ∓ 1, n)`.
pub fn split_from_order(n: u64, base: u64, order: u64) -> Option<(u64, u64)> {
    if order == 0 || order % 2 != 0 {
        return None;
    }
    let x = pow_mod(base, order / 2, n);
    if x == n - 1 || x == 1 {
        return None;
    }
    for cand in [gcd(x + 1, n), gcd(x - 1, n)] {
        if cand > 1 && cand < n {
            return Some((cand.min(n / cand), cand.max(n / cand)));
        }
    }
    None
}

/// Which road actually produced the factors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    /// `n` is even.
    Even,
    /// `n = b^k`; no period finding needed.
    PerfectPower(u64, u32),
    /// The random base shared a factor with `n` — free, and reported
    /// rather than hidden, because it happens.
    LuckyGcd(u64),
    /// Order finding succeeded with this base and order.
    OrderFinding {
        /// The base whose order was estimated.
        base: u64,
        /// The recovered order.
        order: u64,
    },
    /// `n` is prime; there is nothing to split.
    Prime,
    /// Every attempt was spent without a split.
    Exhausted,
}

impl std::fmt::Display for Route {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Route::Even => write!(f, "even"),
            Route::PerfectPower(b, k) => write!(f, "perfect power {b}^{k}"),
            Route::LuckyGcd(a) => write!(f, "lucky gcd with base {a}"),
            Route::OrderFinding { base, order } => write!(f, "order of {base} is {order}"),
            Route::Prime => write!(f, "prime"),
            Route::Exhausted => write!(f, "exhausted"),
        }
    }
}

/// One base tried, and what became of it.
#[derive(Debug, Clone)]
pub struct Attempt {
    /// The base.
    pub base: u64,
    /// The phase estimate, when one was taken.
    pub estimate: Option<PhaseEstimate>,
    /// The order recovered from it.
    pub order: Option<u64>,
    /// Why the attempt ended.
    pub verdict: &'static str,
}

/// The whole factoring run: outcome, route, and the cost of getting there.
#[derive(Debug, Clone)]
pub struct FactorReport {
    /// The modulus.
    pub modulus: u64,
    /// The split, when one was found.
    pub factors: Option<(u64, u64)>,
    /// How it was found.
    pub route: Route,
    /// Every base tried, in order.
    pub attempts: Vec<Attempt>,
    /// Register width the quantum part used.
    pub qubits: usize,
    /// Controlled modular multiplications over the whole run.
    pub modular_multiplications: usize,
    /// Largest support held at any point of any run.
    pub peak_support: usize,
    /// Largest reported representation size over the run.
    pub peak_bytes: usize,
    /// Largest work-register orbit reached over the run.
    pub peak_orbit: usize,
}

/// Factor `n` by order finding, taking (and naming) the classical
/// shortcuts on the way.
///
/// `attempts` bounds the number of random bases tried. The `rng` seeds
/// both the base choice and the measurement outcomes, so a run is
/// reproducible from its seed.
pub fn factor<S: Scalar>(
    sim: &Simulator<S>,
    n: u64,
    form: PhaseForm,
    rng: &mut Prng,
    attempts: usize,
) -> Result<FactorReport> {
    if !(2..=MAX_MODULUS).contains(&n) {
        return Err(Error::InvalidState(format!(
            "modulus {n} outside [2, {MAX_MODULUS}]"
        )));
    }
    let _scope = crate::guard::enter();
    let mut report = FactorReport {
        modulus: n,
        factors: None,
        route: Route::Exhausted,
        attempts: Vec::new(),
        qubits: 0,
        modular_multiplications: 0,
        peak_support: 0,
        peak_bytes: 0,
        peak_orbit: 0,
    };
    if n % 2 == 0 {
        report.factors = Some((2, n / 2));
        report.route = Route::Even;
        return Ok(report);
    }
    if let Some((b, k)) = perfect_power(n) {
        report.factors = Some((b, n / b));
        report.route = Route::PerfectPower(b, k);
        return Ok(report);
    }
    if is_prime(n) {
        report.route = Route::Prime;
        return Ok(report);
    }
    // The width the quantum part will use, whether or not a lucky gcd
    // gets there first.
    let w = work_bits(n);
    report.qubits = match form {
        PhaseForm::Semiclassical => w + 1,
        PhaseForm::FullRegister => w + 2 * w + 1,
    };
    for _ in 0..attempts {
        let a = 2 + rng.next_u64() % (n - 3);
        let g = gcd(a, n);
        if g > 1 {
            report.factors = Some((g.min(n / g), g.max(n / g)));
            report.route = Route::LuckyGcd(a);
            return Ok(report);
        }
        let finder = OrderFinder::new(n, a)?;
        report.qubits = finder.width(form);
        let est = finder.estimate(sim, form, rng)?;
        report.modular_multiplications += est.modular_multiplications;
        report.peak_support = report.peak_support.max(est.peak_support);
        report.peak_bytes = report.peak_bytes.max(est.peak_bytes);
        report.peak_orbit = report.peak_orbit.max(est.peak_orbit);
        let order = order_from_phase(est.phase, n, a, n);
        let split = order.and_then(|r| split_from_order(n, a, r));
        let verdict = match (order, &split) {
            (None, _) => "no order from the convergents",
            (Some(r), None) if r % 2 != 0 => "order is odd",
            (Some(_), None) => "half-power is −1",
            (Some(_), Some(_)) => "split",
        };
        report.attempts.push(Attempt {
            base: a,
            estimate: Some(est),
            order,
            verdict,
        });
        if let (Some(r), Some(pq)) = (order, split) {
            report.factors = Some(pq);
            report.route = Route::OrderFinding { base: a, order: r };
            return Ok(report);
        }
    }
    Ok(report)
}
