//! The polynomial degree of a phase function on an abelian group, measured
//! by iterated discrete derivatives — the instrument that resolves what
//! "linearizing the diagonal" can and cannot reach.
//!
//! # Why a degree
//!
//! A diagonal unitary is a phase function `f: G → U(1)` on the group `G`
//! that indexes the register. [`crate::e8::across`] measured that the E8
//! native operator set never leaves the class of *characters* — phase
//! functions with `f(p+q) = f(p)f(q)` — and that a `t` gate breaks it by
//! √2. [`crate::selfhost`] measured that a stack of E8 volumes reduces a
//! diagonal's **multilinear degree in the bits** to one, turning a `ccz`
//! into a character, and that a `t` is untouched because it is already
//! multilinear-degree one.
//!
//! Those two facts sit together awkwardly: `t` is degree one and yet is not
//! a character. This module explains why, by measuring the *other* degree.
//!
//! For `f: G → U(1)` define the derivative in direction `a`:
//!
//! ```text
//! (Δ_a f)(p) = f(p + a) / f(p)
//! ```
//!
//! and iterate. `f` is a **polynomial phase function of degree ≤ d** when
//! every `(d+1)`-fold derivative is identically 1. Degree 0 is a constant,
//! and **degree 1 is exactly a character** (`Δ_a f` constant in `p` means
//! `f(p+a)/f(p)` does not depend on `p`). So "is it a character" and "what
//! is its degree" are the same question, and the degree says *how far* from
//! one a diagonal is rather than merely that it is.
//!
//! The `k`-fold derivative is computed from its subset expansion,
//! `Δ_{a₁..a_k} f(p) = Π_{S ⊆ [k]} f(p + Σ_{i∈S} a_i)^{(−1)^{k−|S|}}`, so
//! order `k` costs `2^k` phase evaluations per probe.
//!
//! # What a measurement here proves
//!
//! Asymmetric, and the API keeps the asymmetry visible:
//!
//! * A **non-vanishing** derivative is a *witness*: one `(p, a₁..a_k)` with
//!   `Δ^k f(p) ≠ 1` proves the degree is at least `k`. Sampling is sound
//!   for this direction — [`PhaseDegree::exceeds`].
//! * A **vanishing** derivative over a sampled probe set proves nothing;
//!   under-sampling can only miss a violation. A degree is therefore only
//!   *certified* when the sweep was exhaustive —
//!   [`PhaseDegree::certified_degree`] — and otherwise reported as an upper
//!   bound.

use crate::rng::Prng;
use crate::scalar::C64;

/// Probe tuples above this count switch a sweep from exhaustive to
/// sampled. Order `k` enumerates `|G|^{k+1}` tuples, so the budget binds
/// quickly.
pub const EXHAUSTIVE_BUDGET: usize = 4_000_000;

/// A measured polynomial degree.
#[derive(Debug, Clone)]
pub struct PhaseDegree {
    /// Worst `|Δ^k f − 1|` over the probes, for `k = 1 ..= max_order`.
    /// `order_residuals[k-1]` is order `k`.
    pub order_residuals: Vec<f64>,
    /// The smallest order whose derivative vanished everywhere probed;
    /// the degree is that order minus one. `None` when no order up to
    /// `max_order` vanished.
    pub vanishing_order: Option<u32>,
    /// Highest order tried.
    pub max_order: u32,
    /// Probe tuples used at each order.
    pub probes: Vec<usize>,
    /// Whether every order's sweep enumerated all tuples.
    pub exhaustive: bool,
    /// Tolerance a residual had to fall under to count as vanishing.
    pub tolerance: f64,
}

impl PhaseDegree {
    /// The degree, as an upper bound: `vanishing_order − 1`. `None` when
    /// nothing vanished up to `max_order`.
    pub fn degree(&self) -> Option<u32> {
        self.vanishing_order.map(|order| order.saturating_sub(1))
    }

    /// The degree, only when the sweep was exhaustive and therefore proves
    /// it. A sampled sweep returns `None` here however clean it looked.
    pub fn certified_degree(&self) -> Option<u32> {
        self.exhaustive.then(|| self.degree()).flatten()
    }

    /// Whether the measurement holds a *witness* that the degree exceeds
    /// `k`: order `k` had a non-vanishing derivative. Sound under
    /// sampling — a witness is a witness.
    pub fn exceeds(&self, k: u32) -> bool {
        if k == 0 {
            return self
                .order_residuals
                .first()
                .is_some_and(|r| *r > self.tolerance);
        }
        self.order_residuals
            .get(k as usize - 1)
            .is_some_and(|r| *r > self.tolerance)
            && self
                .order_residuals
                .get(k as usize)
                .is_some_and(|r| *r > self.tolerance)
    }

    /// Whether the function is a character of the group — degree one, i.e.
    /// the second derivative vanishes.
    pub fn is_character(&self) -> bool {
        self.degree() == Some(1)
    }
}

/// A group to differentiate over: its elements (or a sample of them) and
/// its addition.
pub struct PhaseGroup<'a> {
    /// Element encodings. Addition must be closed over this set for an
    /// exhaustive sweep to mean what it says.
    pub elements: &'a [u64],
    /// Whether `elements` is the whole group.
    pub complete: bool,
    /// The group operation on encodings.
    pub add: &'a dyn Fn(u64, u64) -> u64,
}

impl<'a> PhaseGroup<'a> {
    /// The bit group `(Z/2)^n` under XOR — the group a register's basis
    /// indices form when a diagonal is read as a multilinear polynomial.
    pub fn bits(n: usize, xor: &'a dyn Fn(u64, u64) -> u64, elements: &'a [u64]) -> Self {
        let _ = n;
        PhaseGroup {
            elements,
            complete: true,
            add: xor,
        }
    }
}

/// Every element of `(Z/2)^n`, for `n ≤ 20`.
pub fn bit_group_elements(n: usize) -> Vec<u64> {
    assert!(n <= 20, "enumerating 2^{n} elements is not intended here");
    (0..1u64 << n).collect()
}

/// The `k`-fold derivative of `phase` at `p` along `dirs`, from the subset
/// expansion.
fn derivative(
    add: &dyn Fn(u64, u64) -> u64,
    phase: &dyn Fn(u64) -> C64,
    p: u64,
    dirs: &[u64],
) -> C64 {
    let k = dirs.len();
    let mut acc = C64::new(1.0, 0.0);
    for mask in 0u32..(1u32 << k) {
        let mut point = p;
        for (i, &a) in dirs.iter().enumerate() {
            if mask >> i & 1 == 1 {
                point = add(point, a);
            }
        }
        let value = phase(point);
        // Exponent (−1)^{k−|S|}: multiply on even, divide on odd.
        if (k as u32 - mask.count_ones()) % 2 == 0 {
            acc *= value;
        } else {
            acc /= value;
        }
    }
    acc
}

/// Measure the polynomial degree of `phase` on `group`, trying orders
/// `1 ..= max_order`.
///
/// Each order enumerates all `(p, a₁..a_k)` tuples when that fits inside
/// [`EXHAUSTIVE_BUDGET`], and otherwise draws `sampled` deterministic
/// pseudo-random tuples. The sweep stops at the first order that vanishes:
/// higher derivatives of a vanishing one vanish too, so there is nothing
/// past it to measure.
pub fn phase_degree(
    group: &PhaseGroup<'_>,
    phase: &dyn Fn(u64) -> C64,
    max_order: u32,
    sampled: usize,
    tolerance: f64,
) -> PhaseDegree {
    let size = group.elements.len();
    let mut order_residuals = Vec::new();
    let mut probes = Vec::new();
    let mut exhaustive = group.complete;
    let mut vanishing_order = None;
    let mut rng = Prng::new(0x00DE_62EE);

    for k in 1..=max_order {
        let tuples = (size as u128).checked_pow(k + 1);
        let full = group.complete
            && tuples.is_some_and(|count| count <= EXHAUSTIVE_BUDGET as u128)
            && size > 0;
        let mut worst = 0.0f64;
        let mut count = 0usize;
        if full {
            // Every base point against every direction tuple.
            let mut dirs = vec![0usize; k as usize];
            loop {
                for &p in group.elements {
                    let directions: Vec<u64> =
                        dirs.iter().map(|&index| group.elements[index]).collect();
                    let value = derivative(group.add, phase, p, &directions);
                    worst = worst.max((value - C64::new(1.0, 0.0)).norm());
                    count += 1;
                }
                // Odometer over direction indices.
                let mut position = 0usize;
                loop {
                    if position == dirs.len() {
                        break;
                    }
                    dirs[position] += 1;
                    if dirs[position] < size {
                        break;
                    }
                    dirs[position] = 0;
                    position += 1;
                }
                if position == dirs.len() {
                    break;
                }
            }
        } else {
            exhaustive = false;
            for _ in 0..sampled {
                let p = group.elements[(rng.next_u64() as usize) % size];
                let directions: Vec<u64> = (0..k)
                    .map(|_| group.elements[(rng.next_u64() as usize) % size])
                    .collect();
                let value = derivative(group.add, phase, p, &directions);
                worst = worst.max((value - C64::new(1.0, 0.0)).norm());
                count += 1;
            }
        }
        order_residuals.push(worst);
        probes.push(count);
        if worst <= tolerance {
            vanishing_order = Some(k);
            break;
        }
    }

    PhaseDegree {
        order_residuals,
        vanishing_order,
        max_order,
        probes,
        exhaustive,
        tolerance,
    }
}

/// The phase function of a diagonal gate given as `2^k` diagonal entries
/// over the whole register, for use with [`phase_degree`].
pub fn diagonal_phase(entries: &[C64]) -> impl Fn(u64) -> C64 + '_ {
    move |index| {
        entries
            .get(index as usize)
            .copied()
            .unwrap_or(C64::new(1.0, 0.0))
    }
}

/// The measured relationship between a diagonal's two degrees.
///
/// A diagonal has a **multilinear degree** — the largest monomial of its
/// phase polynomial in the register bits, which is what
/// [`crate::selfhost`] reduces to one — and a **phase degree**, measured
/// here, which is what decides whether it is a character. They are not the
/// same number, and the gap is the whole content of the √2 obstruction.
#[derive(Debug, Clone)]
pub struct DegreeComparison {
    /// Label for reports.
    pub label: String,
    /// Largest monomial in the phase polynomial over the register bits.
    pub multilinear_degree: u32,
    /// `log₂` of the phase denominator: 1 for a ±1 diagonal, 3 for a `t`.
    pub denominator_log: u32,
    /// The measured phase degree on the bit group.
    pub phase_degree: Option<u32>,
    /// Whether the phase-degree sweep was exhaustive.
    pub certified: bool,
    /// Whether the diagonal is a character of the bit group.
    pub is_character: bool,
}

impl DegreeComparison {
    /// The degree predicted by `multilinear + log₂(denominator) − 1`.
    ///
    /// Not a definition — a conjecture the measurements either confirm or
    /// refute, which is why it is computed separately from
    /// [`DegreeComparison::phase_degree`] rather than used to fill it in.
    pub fn predicted(&self) -> u32 {
        self.multilinear_degree + self.denominator_log - 1
    }

    /// Whether the prediction matched the measurement.
    pub fn matches_prediction(&self) -> bool {
        self.phase_degree == Some(self.predicted())
    }
}

/// Pair a diagonal's declared multilinear degree and denominator against a
/// measured phase degree.
pub fn compare_degrees(
    label: impl Into<String>,
    multilinear_degree: u32,
    denominator_log: u32,
    measured: &PhaseDegree,
) -> DegreeComparison {
    DegreeComparison {
        label: label.into(),
        multilinear_degree,
        denominator_log,
        phase_degree: measured.degree(),
        certified: measured.exhaustive,
        is_character: measured.is_character(),
    }
}
