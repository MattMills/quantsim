//! A geometric object that computes itself: a stack of E8 volumes whose
//! each layer holds the previous layers' own diagonal, progressively, so
//! that the diagonal becomes **linear** — and a linear diagonal is a
//! product of single-qubit phases.
//!
//! # The object
//!
//! One E8 volume is the coset group `E8/2E8 ≅ F₂⁸` — exactly one byte,
//! eight coordinates, and (measured in [`crate::e8`]) the class map is
//! F₂-linear. That linearity is the whole reason this works: a volume's
//! coordinate can hold the value of *any* F₂ function of the substrate,
//! and remain a coordinate.
//!
//! A [`SelfHostedStack`] is a substrate register together with a tower of
//! such volumes:
//!
//! * **layer 0** is the substrate — the register the computation is
//!   *about*, the object acting "on its own purpose";
//! * **layer k** holds the degree-`k+1` monomials of the substrate bits,
//!   one per coordinate — the object *computing itself*, since a layer's
//!   content is a function of the layers below it;
//! * the stack is [`expand`](SelfHostedStack::expand)ed by evaluating
//!   those monomials, which is the feedback: the object's own output
//!   becomes the next layer's coordinates.
//!
//! # Linearizing the diagonal
//!
//! Any diagonal unitary on `n` qubits is a phase polynomial
//! ([`Diagonal`]): `phase(x) = exp(2πi Σ_M c_M · x_M / 2^bits)`, the sum
//! over monomials `x_M = Π_{j∈M} x_j`. Its **degree** is the largest
//! monomial. Degree 1 is special:
//!
//! > A degree-1 diagonal is a product of *single-qubit* phase gates.
//!
//! It entangles nothing, costs one gate per qubit, and — on a group whose
//! coordinates are the bits — it is a **character**, the one kind of
//! diagonal the E8 native operator set can apply
//! ([`crate::e8::across`] measures that boundary: every native operator
//! stays inside coset-with-linear-character, and a `t` through the qubit
//! path breaks it).
//!
//! [`SelfHostedStack::linearize`] rewrites a degree-`d` diagonal as a
//! degree-**1** diagonal over the stack, by replacing each higher
//! monomial with the stack coordinate that holds it. So a `ccz` — a
//! degree-3, genuinely non-Clifford diagonal — becomes three single-qubit
//! phase gates on a stack two layers deep. That is the recursive
//! expansion of capacity: **each layer buys one more degree**, measured
//! in [`recursive_expansion`].
//!
//! # What it costs, honestly
//!
//! The expansion is not free and cannot be: if it were, the native
//! operator set would manufacture non-Clifford diagonals from nothing.
//! Writing a degree-`k` monomial into a coordinate is a `k`-controlled
//! X — precisely the non-native work, priced by
//! [`SelfHostedStack::self_computation_cost`]. The object pays that once
//! and then applies the diagonal as single-qubit phases as often as it
//! likes, so the question is amortization, and [`amortization`] measures
//! the crossover on real runs rather than asserting one.

use std::collections::BTreeMap;

use crate::circuit::Circuit;
use crate::error::{Error, Result};
use crate::scalar::C64;
use crate::sim::Simulator;

/// Coordinates in one E8 volume: the eight of `E8/2E8 ≅ F₂⁸`, one byte.
pub const VOLUME_COORDINATES: usize = 8;

/// A diagonal unitary as a phase polynomial over the register's bits.
///
/// `phase(x) = exp(2πi · Σ_M c_M · x_M / 2^bits)`, where the sum runs
/// over monomials keyed by their bit mask and `x_M = Π_{j∈M} x_j`. The
/// empty mask is a global phase.
///
/// Every diagonal unitary whose entries are `2^bits`-th roots of unity
/// has such a form, so this is a representation rather than a
/// restriction. `cz` is `bits = 1` with the single term `{0,1} ↦ 1`;
/// `ccz` is `bits = 1` with `{0,1,2} ↦ 1`; `t` is `bits = 3` with
/// `{0} ↦ 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagonal {
    bits: u32,
    terms: BTreeMap<u64, i64>,
}

impl Diagonal {
    /// A zero polynomial with phases valued in `2^bits`-th roots of
    /// unity. `bits` must be in `1..=32`.
    pub fn new(bits: u32) -> Result<Self> {
        if bits == 0 || bits > 32 {
            return Err(Error::InvalidState(format!(
                "phase denominator 2^{bits} out of range (need 1..=32)"
            )));
        }
        Ok(Diagonal {
            bits,
            terms: BTreeMap::new(),
        })
    }

    /// Add `coeff` to the monomial over `mask`'s bits, reduced modulo
    /// `2^bits`. A coefficient reducing to zero is dropped, so the term
    /// list stays canonical and [`Diagonal::degree`] never counts a
    /// monomial that contributes nothing.
    pub fn term(&mut self, mask: u64, coeff: i64) -> &mut Self {
        let modulus = 1i64 << self.bits;
        let existing = self.terms.get(&mask).copied().unwrap_or(0);
        let reduced = (existing + coeff).rem_euclid(modulus);
        if reduced == 0 {
            self.terms.remove(&mask);
        } else {
            self.terms.insert(mask, reduced);
        }
        self
    }

    /// The phase denominator exponent.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// The monomials and their coefficients, ordered by mask.
    pub fn terms(&self) -> &BTreeMap<u64, i64> {
        &self.terms
    }

    /// The largest monomial degree present; `0` for a global phase alone.
    pub fn degree(&self) -> u32 {
        self.terms.keys().map(|m| m.count_ones()).max().unwrap_or(0)
    }

    /// Whether every monomial has degree at most one — the case where
    /// the diagonal factorizes into single-qubit phases.
    pub fn is_linear(&self) -> bool {
        self.degree() <= 1
    }

    /// The monomials of degree at least two, ordered by (degree, mask) —
    /// the ones a stack has to hold, coarsest degree last so layers can
    /// be filled progressively.
    pub fn nonlinear_monomials(&self) -> Vec<u64> {
        let mut masks: Vec<u64> = self
            .terms
            .keys()
            .copied()
            .filter(|m| m.count_ones() >= 2)
            .collect();
        masks.sort_by_key(|m| (m.count_ones(), *m));
        masks
    }

    /// The phase this diagonal applies to basis state `x`.
    pub fn phase(&self, x: u64) -> C64 {
        let modulus = 1i64 << self.bits;
        let mut acc = 0i64;
        for (&mask, &coeff) in &self.terms {
            if x & mask == mask {
                acc = (acc + coeff).rem_euclid(modulus);
            }
        }
        let angle = std::f64::consts::TAU * acc as f64 / modulus as f64;
        C64::new(angle.cos(), angle.sin())
    }

    /// Worst violation of `phase(x ⊕ y) = phase(x)·phase(y)` — the
    /// failure of the diagonal to be a **character of `(Z/2)^width`**.
    ///
    /// This is the same instrument
    /// [`crate::e8::across::SupportClass::character_residual`] applies to
    /// the constellation, pointed at the F₂ group an E8 volume actually
    /// is (`E8/2E8 ≅ F₂⁸`, measured in [`crate::e8`]). A character is the
    /// one diagonal an F₂ volume can apply natively, so this number is
    /// what "linearizing the diagonal" has to drive to zero.
    ///
    /// Exhaustive over all pairs when `2^width ≤ 64`, and over a
    /// deterministic pseudo-random sample beyond — reported by
    /// [`Diagonal::character_sample`], since an under-sampled sweep can
    /// only *miss* a violation, never invent one.
    pub fn character_residual(&self, width: usize) -> f64 {
        let (pairs, _) = self.character_pairs(width);
        let mut worst = 0.0f64;
        for (x, y) in pairs {
            let got = self.phase(x ^ y);
            let want = self.phase(x) * self.phase(y);
            worst = worst.max((got - want).norm());
        }
        worst
    }

    /// Pairs used by [`Diagonal::character_residual`] at this width, and
    /// whether they were exhaustive.
    pub fn character_sample(&self, width: usize) -> (usize, bool) {
        let (pairs, exhaustive) = self.character_pairs(width);
        (pairs.len(), exhaustive)
    }

    fn character_pairs(&self, width: usize) -> (Vec<(u64, u64)>, bool) {
        let span = 1u64 << width.min(63);
        if span <= 64 {
            let mut pairs = Vec::with_capacity((span * span) as usize);
            for x in 0..span {
                for y in 0..span {
                    pairs.push((x, y));
                }
            }
            return (pairs, true);
        }
        let mut rng = crate::rng::Prng::new(0x5E1F_1105);
        let mask = span - 1;
        let pairs = (0..2048)
            .map(|_| (rng.next_u64() & mask, rng.next_u64() & mask))
            .collect();
        (pairs, false)
    }

    /// Whether the diagonal is a character of `(Z/2)^width`, to `1e-12`.
    pub fn is_f2_character(&self, width: usize) -> bool {
        self.character_residual(width) <= 1e-12
    }

    /// The circuit applying this diagonal through the *diagonal kernel*
    /// path — one kernel per monomial, over that monomial's own qubits.
    ///
    /// Both routes in [`amortization`] use this, so the comparison is
    /// between gate *arities* rather than between two different code
    /// paths in the backend.
    pub fn kernel_circuit(&self, width: usize) -> Result<Circuit<C64>> {
        let mut c = Circuit::new(width);
        let modulus = (1u64 << self.bits) as f64;
        for (&mask, &coeff) in &self.terms {
            let qubits: Vec<usize> = (0..width).filter(|&q| mask >> q & 1 == 1).collect();
            if qubits.len() != mask.count_ones() as usize {
                return Err(Error::InvalidState(format!(
                    "monomial {mask:#x} refers to qubits outside a {width}-qubit register"
                )));
            }
            let angle = std::f64::consts::TAU * coeff as f64 / modulus;
            let phase = C64::new(angle.cos(), angle.sin());
            match qubits.len() {
                0 => c.diagonal("global-phase", vec![phase, phase], vec![0]),
                k => {
                    let d = 1usize << k;
                    let mut entries = vec![C64::new(1.0, 0.0); d];
                    entries[d - 1] = phase;
                    c.diagonal(format!("phase-deg{k}"), entries, qubits)
                }
            };
        }
        Ok(c)
    }

    /// The largest gate arity [`Diagonal::kernel_circuit`] needs.
    pub fn max_arity(&self) -> u32 {
        self.degree().max(1)
    }

    /// Monomials of degree at least two — the gates that entangle, and
    /// the ones a stack removes from every application.
    pub fn multicontrolled_gates(&self) -> usize {
        self.terms.keys().filter(|m| m.count_ones() >= 2).count()
    }

    /// A linear diagonal as its single-qubit phase angles, one per qubit
    /// carrying a degree-1 term, plus the global angle from the empty
    /// monomial. Refuses when the diagonal is not linear — the whole
    /// point of [`SelfHostedStack::linearize`] is to make this succeed.
    pub fn single_qubit_angles(&self) -> Result<(f64, Vec<(usize, f64)>)> {
        if !self.is_linear() {
            return Err(Error::InvalidState(format!(
                "a degree-{} diagonal does not factorize into single-qubit phases",
                self.degree()
            )));
        }
        let modulus = (1u64 << self.bits) as f64;
        let mut global = 0.0;
        let mut angles = Vec::new();
        for (&mask, &coeff) in &self.terms {
            let angle = std::f64::consts::TAU * coeff as f64 / modulus;
            if mask == 0 {
                global += angle;
            } else {
                angles.push((mask.trailing_zeros() as usize, angle));
            }
        }
        angles.sort_by_key(|&(q, _)| q);
        Ok((global, angles))
    }

    /// A circuit applying this diagonal on `width` qubits.
    ///
    /// A linear diagonal becomes one `p` gate per qubit. A higher-degree
    /// monomial becomes a multi-controlled phase, built as a raw diagonal
    /// kernel over its own qubits — which is exactly the expensive thing
    /// the stack exists to avoid.
    pub fn circuit(&self, width: usize) -> Result<Circuit<C64>> {
        let mut c = Circuit::new(width);
        let modulus = (1u64 << self.bits) as f64;
        for (&mask, &coeff) in &self.terms {
            let qubits: Vec<usize> = (0..width).filter(|&q| mask >> q & 1 == 1).collect();
            if qubits.len() != mask.count_ones() as usize {
                return Err(Error::InvalidState(format!(
                    "monomial {mask:#x} refers to qubits outside a {width}-qubit register"
                )));
            }
            let angle = std::f64::consts::TAU * coeff as f64 / modulus;
            match qubits.len() {
                0 => {
                    // A global phase: one p gate on qubit 0 would be
                    // wrong (it is conditional), so apply it as a
                    // one-qubit diagonal with both entries equal.
                    let e = C64::new(angle.cos(), angle.sin());
                    c.diagonal("global-phase", vec![e, e], vec![0]);
                }
                1 => {
                    c.gate("p", vec![angle], qubits);
                }
                k => {
                    let d = 1usize << k;
                    let mut entries = vec![C64::new(1.0, 0.0); d];
                    entries[d - 1] = C64::new(angle.cos(), angle.sin());
                    c.diagonal(format!("mcp-{k}"), entries, qubits);
                }
            }
        }
        Ok(c)
    }
}

/// One layer of the stack: an E8 volume tower holding the monomials of a
/// single degree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layer {
    /// The monomial degree this layer holds.
    pub degree: u32,
    /// The monomial masks, in coordinate order.
    pub monomials: Vec<u64>,
    /// E8 volumes this layer occupies (8 coordinates each).
    pub volumes: usize,
}

impl Layer {
    /// Coordinates the layer's volumes provide.
    pub fn capacity(&self) -> usize {
        self.volumes * VOLUME_COORDINATES
    }

    /// Coordinates left unused — the layer's spare capacity for further
    /// monomials of the same degree.
    pub fn spare(&self) -> usize {
        self.capacity() - self.monomials.len()
    }
}

/// The measured cost of the object computing itself: writing each
/// monomial into its coordinate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfComputationCost {
    /// Multi-controlled X gates, one per held monomial.
    pub gates: usize,
    /// Control count per gate, ascending — the monomial degrees.
    pub arities: Vec<u32>,
    /// Toffoli-equivalents, counting a `k`-controlled X as `2k − 3` for
    /// `k ≥ 2` (the standard ancilla-free ladder) and 1 for `k = 2`.
    pub toffoli_equivalents: usize,
}

/// A stack of E8 volumes over a substrate register: the object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelfHostedStack {
    substrate: usize,
    layers: Vec<Layer>,
    slots: BTreeMap<u64, usize>,
}

impl SelfHostedStack {
    /// Plan the stack a diagonal needs: one layer per monomial degree
    /// from 2 up to the diagonal's degree, each filled with that degree's
    /// monomials, taking as many E8 volumes as the count requires.
    ///
    /// Progressive by construction — layer `k` holds degree `k+1`, so the
    /// object's depth *is* the diagonal's degree minus one.
    pub fn plan(substrate: usize, diagonal: &Diagonal) -> Result<Self> {
        if substrate == 0 || substrate > 32 {
            return Err(Error::InvalidState(format!(
                "substrate width {substrate} out of range (need 1..=32)"
            )));
        }
        for &mask in diagonal.terms.keys() {
            if mask >> substrate != 0 {
                return Err(Error::InvalidState(format!(
                    "monomial {mask:#x} refers to qubits outside the {substrate}-qubit substrate"
                )));
            }
        }
        let mut layers: Vec<Layer> = Vec::new();
        let mut slots = BTreeMap::new();
        let mut next = substrate;
        for degree in 2..=diagonal.degree() {
            let monomials: Vec<u64> = diagonal
                .nonlinear_monomials()
                .into_iter()
                .filter(|m| m.count_ones() == degree)
                .collect();
            // A layer exists for every degree in range, even an empty
            // one: the stack's depth is the degree it reaches, and
            // hiding a gap would misreport the recursion.
            let volumes = monomials.len().div_ceil(VOLUME_COORDINATES).max(1);
            for &mask in &monomials {
                slots.insert(mask, next);
                next += 1;
            }
            // Coordinates the layer reserves but does not fill stay
            // reserved, so slot indices are layer-aligned.
            next = substrate
                + layers.iter().map(Layer::capacity).sum::<usize>()
                + volumes * VOLUME_COORDINATES;
            layers.push(Layer {
                degree,
                monomials,
                volumes,
            });
        }
        Ok(SelfHostedStack {
            substrate,
            layers,
            slots,
        })
    }

    /// The substrate width — layer 0, the register the computation is
    /// about.
    pub fn substrate(&self) -> usize {
        self.substrate
    }

    /// The volume layers above the substrate.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Layers above the substrate — the recursion depth.
    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    /// Total register width: substrate plus every layer's volumes.
    pub fn width(&self) -> usize {
        self.substrate + self.layers.iter().map(Layer::capacity).sum::<usize>()
    }

    /// E8 volumes in the whole stack.
    pub fn volumes(&self) -> usize {
        self.layers.iter().map(|l| l.volumes).sum()
    }

    /// Coordinates actually holding a monomial, over coordinates
    /// available — how much of the geometry the diagonal uses.
    pub fn occupancy(&self) -> f64 {
        let capacity: usize = self.layers.iter().map(Layer::capacity).sum();
        if capacity == 0 {
            return 1.0;
        }
        self.slots.len() as f64 / capacity as f64
    }

    /// The stack qubit holding a monomial, if the stack holds it.
    pub fn slot(&self, mask: u64) -> Option<usize> {
        self.slots.get(&mask).copied()
    }

    /// The self-computation: a substrate basis index extended with every
    /// held monomial's value. This is the feedback step — the object
    /// evaluating its own structure into its own coordinates.
    pub fn expand(&self, x: u64) -> u64 {
        let mut out = x & ((1u64 << self.substrate) - 1);
        for (&mask, &slot) in &self.slots {
            if x & mask == mask {
                out |= 1 << slot;
            }
        }
        out
    }

    /// Whether a stack basis index is consistent — every coordinate
    /// really holds its monomial's value. Indices failing this are off
    /// the object's own image and carry no meaning.
    pub fn consistent(&self, y: u64) -> bool {
        self.expand(y) == y & ((1u64 << self.width()) - 1)
    }

    /// Rewrite `diagonal` as a **degree-1** diagonal over the stack, by
    /// replacing each higher monomial with the coordinate holding it.
    ///
    /// This is the linearization. The result satisfies
    /// [`Diagonal::is_linear`] and therefore
    /// [`Diagonal::single_qubit_angles`] — an entangling multi-qubit
    /// diagonal has become a layer of single-qubit phases.
    pub fn linearize(&self, diagonal: &Diagonal) -> Result<Diagonal> {
        let mut out = Diagonal::new(diagonal.bits)?;
        for (&mask, &coeff) in &diagonal.terms {
            let target = if mask.count_ones() >= 2 {
                let slot = self.slot(mask).ok_or_else(|| {
                    Error::InvalidState(format!(
                        "this stack holds no coordinate for monomial {mask:#x}"
                    ))
                })?;
                1u64 << slot
            } else {
                mask
            };
            // Two monomials can share a slot only if they are equal, so
            // `term`'s accumulation keeps the result canonical.
            out.term(target, coeff);
        }
        Ok(out)
    }

    /// The circuit that performs the self-computation: one
    /// multi-controlled X per held monomial, writing it into its
    /// coordinate.
    pub fn self_computation_circuit(&self) -> Result<Circuit<C64>> {
        let width = self.width();
        let mut c = Circuit::new(width);
        for (&mask, &slot) in &self.slots {
            let mut qubits: Vec<usize> = (0..self.substrate)
                .filter(|&q| mask >> q & 1 == 1)
                .collect();
            qubits.push(slot);
            let d = 1usize << qubits.len();
            // A controlled X on the target: identity except on the block
            // where every control is set, where it swaps 0 and 1.
            let mut m = crate::math::GateMatrix::zeros(d)?;
            for i in 0..d {
                let controls_set = i & (d / 2 - 1) == d / 2 - 1;
                let j = if controls_set { i ^ (d / 2) } else { i };
                m.set(j, i, C64::new(1.0, 0.0));
            }
            c.raw(format!("write-deg{}", mask.count_ones()), m, qubits);
        }
        Ok(c)
    }

    /// The cost of the object computing itself, counted.
    pub fn self_computation_cost(&self) -> SelfComputationCost {
        let mut arities: Vec<u32> = self.slots.keys().map(|m| m.count_ones()).collect();
        arities.sort_unstable();
        let toffoli_equivalents = arities
            .iter()
            .map(|&k| if k <= 2 { 1 } else { 2 * k as usize - 3 })
            .sum();
        SelfComputationCost {
            gates: arities.len(),
            arities,
            toffoli_equivalents,
        }
    }
}

/// What a linearization achieved, measured against the substrate.
#[derive(Debug, Clone)]
pub struct LinearizationReport {
    /// Substrate width.
    pub substrate: usize,
    /// Full stack width.
    pub width: usize,
    /// Layers above the substrate.
    pub depth: usize,
    /// E8 volumes used.
    pub volumes: usize,
    /// Degree of the diagonal on the substrate.
    pub original_degree: u32,
    /// Degree after linearization — the claim is 1.
    pub linear_degree: u32,
    /// Single-qubit phase gates the linearized diagonal needs.
    pub phase_gates: usize,
    /// Toffoli-equivalents the self-computation costs.
    pub toffoli_equivalents: usize,
    /// Worst amplitude deviation between the stack route and applying
    /// the diagonal directly on the substrate.
    pub deviation: f64,
    /// The substrate diagonal's failure to be a character of `(Z/2)^n` —
    /// nonzero exactly when it entangles.
    pub substrate_character_residual: f64,
    /// The linearized diagonal's failure to be a character of
    /// `(Z/2)^width`. Zero for a real (`bits = 1`) diagonal: the
    /// entangling diagonal has become something an F₂ volume applies
    /// natively.
    pub stack_character_residual: f64,
}

impl LinearizationReport {
    /// Whether the stack route reproduced the substrate diagonal exactly
    /// (to `1e-12`) *and* actually linearized it.
    pub fn exact(&self) -> bool {
        self.linear_degree <= 1 && self.deviation <= 1e-12
    }

    /// Whether the linearization also turned the diagonal into a
    /// character of the stack's F₂ group — true for real diagonals, and
    /// **false** for a `t`-like diagonal, whose 8th-root phases factorize
    /// into single-qubit gates without ever becoming ±1 valued. The
    /// distinction matters and this field keeps it visible.
    pub fn became_character(&self) -> bool {
        self.stack_character_residual <= 1e-12
    }
}

/// Verify a linearization end to end on real states.
///
/// Applies `diagonal` directly to a uniform superposition of the
/// substrate, then takes the stack route — expand the register, apply the
/// linearized diagonal as single-qubit phases, restrict back — and
/// compares amplitude by amplitude. Both routes run on `backend` through
/// the simulator, so this is a measured agreement, not an algebraic
/// argument.
pub fn verify(
    sim: &Simulator<C64>,
    backend: &str,
    substrate: usize,
    diagonal: &Diagonal,
) -> Result<LinearizationReport> {
    let stack = SelfHostedStack::plan(substrate, diagonal)?;
    let linear = stack.linearize(diagonal)?;
    let cost = stack.self_computation_cost();

    // Route A: the diagonal applied directly on the substrate.
    let amp = 1.0 / (1u64 << substrate) as f64;
    let direct: Vec<(u64, C64)> = (0..1u64 << substrate)
        .map(|x| (x, C64::new(amp.sqrt(), 0.0) * diagonal.phase(x)))
        .collect();

    // Route B: expand, apply single-qubit phases, read back.
    let mut state = sim.backends().create(backend, stack.width())?;
    state.reset();
    let entries: Vec<(u64, C64)> = (0..1u64 << substrate)
        .map(|x| (stack.expand(x), C64::new(amp.sqrt(), 0.0)))
        .collect();
    state.load(&entries)?;
    let (global, angles) = linear.single_qubit_angles()?;
    linear
        .kernel_circuit(stack.width())?
        .bind(sim.registry())?
        .run(state.as_mut())?;
    // kernel_circuit already applies the global term, so it is divided
    // out here rather than multiplied in.
    let _ = global;
    let global_phase = C64::new(1.0, 0.0);

    let mut deviation: f64 = 0.0;
    for &(x, want) in &direct {
        let got = state.amplitude(stack.expand(x)) * global_phase;
        deviation = deviation.max((got - want).norm());
    }
    // Nothing may have leaked off the object's own image.
    let mut leaked = 0.0f64;
    let image: std::collections::HashSet<u64> =
        direct.iter().map(|&(x, _)| stack.expand(x)).collect();
    state.for_each_nonzero(&mut |y, a| {
        if !image.contains(&y) {
            leaked = leaked.max(a.norm());
        }
    });
    deviation = deviation.max(leaked);

    Ok(LinearizationReport {
        substrate,
        width: stack.width(),
        depth: stack.depth(),
        volumes: stack.volumes(),
        original_degree: diagonal.degree(),
        linear_degree: linear.degree(),
        phase_gates: angles.len(),
        toffoli_equivalents: cost.toffoli_equivalents,
        deviation,
        substrate_character_residual: diagonal.character_residual(substrate),
        stack_character_residual: linear.character_residual(stack.width()),
    })
}

/// The full-monomial diagonal of a given degree on a substrate: every
/// monomial of exactly `degree` bits, coefficient 1. The densest
/// diagonal of its degree, so its stack is the widest that degree needs.
pub fn full_degree(substrate: usize, degree: u32, bits: u32) -> Result<Diagonal> {
    let mut d = Diagonal::new(bits)?;
    for mask in 0..1u64 << substrate {
        if mask.count_ones() == degree {
            d.term(mask, 1);
        }
    }
    Ok(d)
}

/// Sweep degrees `2..=max_degree` on a substrate and report each
/// linearization — the measured recursive expansion of capacity.
///
/// The expected shape, if the object works as described: depth grows by
/// exactly one per degree, the linearized degree stays 1 throughout, and
/// every route is exact.
pub fn recursive_expansion(
    sim: &Simulator<C64>,
    backend: &str,
    substrate: usize,
    max_degree: u32,
) -> Result<Vec<LinearizationReport>> {
    let mut out = Vec::new();
    for degree in 2..=max_degree {
        // One monomial per degree keeps the stack narrow, so the sweep
        // measures the DEPTH the degree costs rather than the width a
        // dense polynomial happens to need.
        let mut d = Diagonal::new(1)?;
        d.term((1u64 << degree) - 1, 1);
        out.push(verify(sim, backend, substrate, &d)?);
    }
    Ok(out)
}

/// Where the stack starts paying: the crossover in repetitions, counted
/// in entangling operations and separately measured in wall clock.
///
/// The counted crossover is the meaningful one. The resource the object
/// trades is *gate arity* — an entangling multi-qubit diagonal per use,
/// versus a one-off write and single-qubit phases forever after. A
/// classical simulator applying a diagonal kernel barely cares about
/// arity (it is `O(support)` either way), so the wall-clock columns are
/// reported for honesty and are expected NOT to show the win.
#[derive(Debug, Clone)]
pub struct Amortization {
    /// Substrate width.
    pub substrate: usize,
    /// Full stack width.
    pub width: usize,
    /// The diagonal's degree.
    pub degree: u32,
    /// Repetition counts measured.
    pub repetitions: Vec<usize>,
    /// Entangling (arity ≥ 2) operations the direct route needs — one set
    /// of multi-controlled phases per repetition.
    pub direct_entangling: Vec<usize>,
    /// Entangling operations the stack route needs — the self-computation
    /// once, and nothing per repetition.
    pub stack_entangling: Vec<usize>,
    /// Largest gate arity each route uses.
    pub direct_max_arity: u32,
    /// Largest gate arity the stack route uses after the one-off write.
    pub stack_steady_max_arity: u32,
    /// Smallest repetition count at which the stack route uses fewer
    /// entangling operations.
    pub counted_crossover: Option<usize>,
    /// Nanoseconds applying the diagonal directly, per repetition count.
    pub direct_nanos: Vec<usize>,
    /// Nanoseconds for the stack route — self-computation once, then the
    /// single-qubit phases per repetition.
    pub stack_nanos: Vec<usize>,
    /// Smallest measured repetition count at which the stack route is
    /// faster in wall clock, or `None` if it never was.
    pub measured_crossover: Option<usize>,
}

/// Measure the amortization: apply `diagonal` `r` times directly, versus
/// paying the self-computation once and applying single-qubit phases `r`
/// times, for each `r` in `repetitions`.
///
/// Both routes are real runs on `backend`, median of three.
pub fn amortization(
    sim: &Simulator<C64>,
    backend: &str,
    substrate: usize,
    diagonal: &Diagonal,
    repetitions: &[usize],
) -> Result<Amortization> {
    let stack = SelfHostedStack::plan(substrate, diagonal)?;
    let linear = stack.linearize(diagonal)?;
    let amp = (1.0 / (1u64 << substrate) as f64).sqrt();

    // Both routes go through the diagonal-kernel path, so the comparison
    // is between arities and counts rather than between two different
    // code paths inside the backend.
    let direct_once = diagonal.kernel_circuit(substrate)?;
    let steady_once = linear.kernel_circuit(stack.width())?;
    let cost = stack.self_computation_cost();
    let mut out = Amortization {
        substrate,
        width: stack.width(),
        degree: diagonal.degree(),
        repetitions: repetitions.to_vec(),
        direct_entangling: repetitions
            .iter()
            .map(|&r| r * diagonal.multicontrolled_gates())
            .collect(),
        stack_entangling: repetitions.iter().map(|_| cost.gates).collect(),
        direct_max_arity: diagonal.max_arity(),
        stack_steady_max_arity: linear.max_arity(),
        counted_crossover: None,
        direct_nanos: Vec::new(),
        stack_nanos: Vec::new(),
        measured_crossover: None,
    };
    out.counted_crossover = repetitions
        .iter()
        .zip(out.direct_entangling.iter().zip(&out.stack_entangling))
        .find(|(_, (&d, &s))| s < d)
        .map(|(&r, _)| r);

    for &reps in repetitions {
        // Direct: the multi-controlled phase, r times.
        let mut direct = Circuit::new(substrate);
        for _ in 0..reps {
            direct.append(&direct_once, &(0..substrate).collect::<Vec<_>>());
        }
        let direct_bound = direct.bind(sim.registry())?;
        let entries: Vec<(u64, C64)> = (0..1u64 << substrate)
            .map(|x| (x, C64::new(amp, 0.0)))
            .collect();
        let mut direct_samples = Vec::with_capacity(3);
        for _ in 0..3 {
            let mut state = sim.backends().create(backend, substrate)?;
            state.reset();
            state.load(&entries)?;
            let start = std::time::Instant::now();
            direct_bound.run(state.as_mut())?;
            direct_samples.push(start.elapsed().as_nanos() as usize);
        }
        direct_samples.sort_unstable();

        // Stack: self-computation once, then single-qubit phases r times.
        let mut route = stack.self_computation_circuit()?;
        let all: Vec<usize> = (0..stack.width()).collect();
        for _ in 0..reps {
            route.append(&steady_once, &all);
        }
        let route_bound = route.bind(sim.registry())?;
        let stack_entries: Vec<(u64, C64)> = (0..1u64 << substrate)
            .map(|x| (x, C64::new(amp, 0.0)))
            .collect();
        let mut stack_samples = Vec::with_capacity(3);
        for _ in 0..3 {
            let mut state = sim.backends().create(backend, stack.width())?;
            state.reset();
            state.load(&stack_entries)?;
            let start = std::time::Instant::now();
            route_bound.run(state.as_mut())?;
            stack_samples.push(start.elapsed().as_nanos() as usize);
        }
        stack_samples.sort_unstable();

        let (d, s) = (direct_samples[1].max(1), stack_samples[1].max(1));
        out.direct_nanos.push(d);
        out.stack_nanos.push(s);
        if out.measured_crossover.is_none() && s < d {
            out.measured_crossover = Some(reps);
        }
    }
    Ok(out)
}
