//! The state as composed topology: a sum over paths, reduced.
//!
//! Every other representation in this crate answers the question "how do
//! I store `2^n` amplitudes more cheaply". This one declines the
//! question. The amplitude vector is a *representation choice*, not the
//! state; what is held instead is the circuit's closed form as a sum
//! over path variables,
//!
//! ```text
//! |ψ⟩ = 2^e · Σ_{y ∈ 𝔽₂^h} ω^{Q(y)} |f(y)⟩,        ω = e^{2πi}
//! ```
//!
//! with one affine **form** `f_q(y) = ⟨m_q, y⟩ ⊕ c_q` per qubit and `Q`
//! an exact phase polynomial: a sum of rational turns on monomials of
//! *parities*. Nothing of size `2^n` is ever built — composition is
//! `𝔽₂` linear algebra on the forms plus polynomial bookkeeping on `Q`,
//! and the whole fragment (X, Z, S, T, CZ, CCZ, CNOT, SWAP) is closed at
//! bounded degree.
//!
//! Coefficients are **exact**. A turn is held as a `u64` numerator over
//! `2^64`, so adding turns is wrapping integer addition, reducing them
//! is exact, and there is no floating point anywhere in the algebra. A
//! gate whose angle is not a dyadic fraction of a turn is refused rather
//! than rounded.
//!
//! ## The wall, and what reduction removes
//!
//! Only one letter introduces a path variable: the Hadamard, or **wall**,
//! which couples the qubit's current form to a fresh `y` through a
//! half-turn term and then *becomes* that variable. Everything else acts
//! in place.
//!
//! A variable that appears in no output form is **internal** — summed
//! over but unobservable — and three rules eliminate it:
//!
//! * **\[E\]** a variable whose terms are `½·y·L` (plus an optional `½·y`
//!   self-term) sums to an affine *constraint* `L = const`, which is
//!   substituted away: two variables leave at once.
//! * **\[G\]** a `¼·y` or `¾·y` self-term is a Gauss sum: the variable
//!   leaves for a `±⅛` global phase and a `∓¼` parity term.
//! * **\[V\]** a variable trapped inside a compound parity is *split*
//!   multilinearly, `p(K) = y + p(r) − 2·y·p(r)`, which exposes the
//!   couplings the other two rules need.
//!
//! What survives is `h*` — [`PathSum::internal_vars`] — the state's
//! irreducible topological content, and readout costs `2^{h*}`.
//!
//! ## What that buys, measured
//!
//! `h*` is **zero for every Clifford circuit**, at any width: the rules
//! consume every variable a wall introduces. That is Gottesman–Knill,
//! recovered with no stabilizer tableau anywhere — from reduction alone.
//! Adding T gates makes `h*` grow with the **T-count and not the
//! width**, so the exponential that remains is indexed by a measured
//! property of the circuit rather than by the size of the register.
//!
//! `h*` is not monotone in the T-count: how much reduces depends on
//! structure, not just on how much magic is present. It is measured
//! after the fact, never predicted.
//!
//! ## The operator formulation, and what needs no tableau
//!
//! [`PathSum::identity`] starts from the identity *operator* rather than
//! a state: each qubit's form is its own free input variable. That one
//! change is what the absence of a tableau buys.
//!
//! A stabilizer tableau represents a stabilizer *state*. It cannot hold
//! a non-Clifford operator at all, so a tableau-based tool has to decide
//! up front which fragment it is in and branch. The path sum has one
//! code path for every circuit, and `h*` *reports* where it landed.
//!
//! Two capabilities follow, neither available to a tableau:
//!
//! * [`equivalent`] decides circuit equality by reducing `V† ∘ U` — and
//!   `T·T = S`, `T⁸ = I` are statements a tableau cannot even express.
//! * Reduction **discovers Clifford-ness the gate list hides**. A circuit
//!   with 128 `T` gates that cancel reduces to `h* = 0` and is certified
//!   Clifford; a T-counting cost model calls it hard, and a tableau
//!   simulator must refuse or fall back. Magic that does *not* cancel is
//!   not certified away, which is what makes the certificate mean
//!   something.
//!
//! Soundness runs one way and is reported that way: reduction only ever
//! rewrites the sum into an equal one, so `true` is a proof. A `false` is
//! "the rewrite system stalled" — complete for the Clifford fragment,
//! not in general — and [`equivalent_verdict`] returns the surviving
//! `h*` so the two can be told apart.
//!
//! Ported from the `octonion_triality` research package's `pathsum` /
//! `pathunit` modules, whose reduction rules this follows.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

use crate::circuit::{Circuit, Op};
use crate::error::{Error, Result};
use crate::scalar::C64;

/// A turn: an exact angle held as a numerator over `2^64`, so a full
/// turn is zero and addition is wrapping integer addition.
///
/// `1/2` is `1 << 63`, `1/4` is `1 << 62`, `1/8` is `1 << 61`. Every
/// coefficient the fragment produces is dyadic, so this is exact
/// arithmetic and not a discretization.
pub type Turn = u64;

/// Half a turn.
pub const HALF: Turn = 1 << 63;
/// A quarter turn.
pub const QUARTER: Turn = 1 << 62;
/// Three quarters of a turn.
pub const THREE_QUARTER: Turn = 3 << 62;
/// An eighth of a turn.
pub const EIGHTH: Turn = 1 << 61;

/// The exact turn `numerator / 2^log2_denominator`.
///
/// The natural constructor: `turn_from_dyadic(1, 3)` is an eighth, which
/// is `T`. Denominators past `2^64` cannot be represented and are
/// refused.
pub fn turn_from_dyadic(numerator: i64, log2_denominator: u32) -> Result<Turn> {
    if log2_denominator > 64 {
        return Err(Error::InvalidState(format!(
            "pathsum: 1/2^{log2_denominator} of a turn is finer than the u64 grid"
        )));
    }
    // A whole turn is zero, so the step for denominator 1 is `2^64 ≡ 0`.
    let step = match log2_denominator {
        0 => 0u64,
        d => 1u64 << (64 - d),
    };
    Ok((numerator as u64).wrapping_mul(step))
}

/// The finest denominator [`turn_from_radians`] will search for.
///
/// Deliberately far short of the `u64` grid. Past about `2^30` the
/// double's own spacing exceeds the tolerance, so a wider search would
/// start *accepting* non-dyadic angles as dyadic ones — the opposite of
/// the guarantee. Refusing a representable-but-absurd angle is the safe
/// failure; silently rounding an arbitrary one is not.
pub const MAX_DYADIC_DEPTH: u32 = 30;

/// The exact turn for `theta` radians, or a refusal.
///
/// The algebra is closed over dyadic angles and nothing else, so an
/// angle that is not `2π·k/2^m` for some `m ≤ `[`MAX_DYADIC_DEPTH`] is
/// **rejected rather than rounded**. That is the honest boundary of this
/// representation: it is exact where it applies, and it says so where it
/// does not, instead of returning an approximation dressed as a closed
/// form.
pub fn turn_from_radians(theta: f64) -> Result<Turn> {
    if !theta.is_finite() {
        return Err(Error::InvalidState("pathsum: non-finite angle".into()));
    }
    let turns = theta / std::f64::consts::TAU;
    for m in 0..=MAX_DYADIC_DEPTH {
        let scaled = turns * 2f64.powi(m as i32);
        let k = scaled.round();
        if (scaled - k).abs() <= 1e-12 && k.abs() < 2f64.powi(53) {
            return turn_from_dyadic(k as i64, m);
        }
    }
    Err(Error::InvalidState(format!(
        "pathsum: {theta} rad is not a dyadic fraction of a turn; \
         this representation refuses it rather than rounding it"
    )))
}

/// Turns as radians, for readout only — the algebra never uses this.
fn turn_to_c64(t: Turn) -> C64 {
    let frac = t as f64 / 2f64.powi(64);
    let theta = std::f64::consts::TAU * frac;
    C64::new(theta.cos(), theta.sin())
}

// ── an unbounded bitset over path variables ──────────────────────────

/// A parity: the set of path variables whose XOR the parity takes.
///
/// Unbounded, because a wall introduces a variable per Hadamard and a
/// circuit may have many; canonical (trailing zero words trimmed) so it
/// can key a hash map.
#[derive(Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Mask(Vec<u64>);

impl Mask {
    /// The empty parity.
    pub fn zero() -> Self {
        Mask(Vec::new())
    }

    /// The parity of a single variable.
    pub fn single(i: usize) -> Self {
        let mut m = Mask(vec![0; i / 64 + 1]);
        m.0[i / 64] = 1u64 << (i % 64);
        m
    }

    fn trim(&mut self) {
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
    }

    /// Whether variable `i` is in the parity.
    pub fn bit(&self, i: usize) -> bool {
        self.0.get(i / 64).is_some_and(|w| w >> (i % 64) & 1 == 1)
    }

    /// Add variable `i` to the parity.
    pub fn set(&mut self, i: usize) {
        if self.0.len() <= i / 64 {
            self.0.resize(i / 64 + 1, 0);
        }
        self.0[i / 64] |= 1u64 << (i % 64);
    }

    /// Remove variable `i`.
    pub fn clear(&mut self, i: usize) {
        if let Some(w) = self.0.get_mut(i / 64) {
            *w &= !(1u64 << (i % 64));
        }
        self.trim();
    }

    /// Symmetric difference — parities add by XOR.
    pub fn xor(&self, other: &Mask) -> Mask {
        let n = self.0.len().max(other.0.len());
        let mut out = Mask(Vec::with_capacity(n));
        for i in 0..n {
            out.0
                .push(self.0.get(i).copied().unwrap_or(0) ^ other.0.get(i).copied().unwrap_or(0));
        }
        out.trim();
        out
    }

    /// Union.
    pub fn or(&self, other: &Mask) -> Mask {
        let n = self.0.len().max(other.0.len());
        let mut out = Mask(Vec::with_capacity(n));
        for i in 0..n {
            out.0
                .push(self.0.get(i).copied().unwrap_or(0) | other.0.get(i).copied().unwrap_or(0));
        }
        out.trim();
        out
    }

    /// Whether any variable below `i` is present.
    pub fn any_below(&self, i: usize) -> bool {
        let full = i / 64;
        if self.0.iter().take(full).any(|&w| w != 0) {
            return true;
        }
        let rest = i % 64;
        rest != 0 && self.0.get(full).is_some_and(|w| w & ((1u64 << rest) - 1) != 0)
    }

    /// Intersection.
    pub fn and(&self, other: &Mask) -> Mask {
        let n = self.0.len().min(other.0.len());
        let mut out = Mask(Vec::with_capacity(n));
        for i in 0..n {
            out.0.push(self.0[i] & other.0[i]);
        }
        out.trim();
        out
    }

    /// Set difference.
    pub fn without(&self, other: &Mask) -> Mask {
        let mut out = Mask(Vec::with_capacity(self.0.len()));
        for i in 0..self.0.len() {
            out.0.push(self.0[i] & !other.0.get(i).copied().unwrap_or(0));
        }
        out.trim();
        out
    }

    /// Whether the parity is empty.
    pub fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether the two parities share a variable.
    pub fn intersects(&self, other: &Mask) -> bool {
        let n = self.0.len().min(other.0.len());
        (0..n).any(|i| self.0[i] & other.0[i] != 0)
    }

    /// Number of variables in the parity.
    pub fn count(&self) -> usize {
        self.0.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Lowest variable in the parity.
    pub fn lowest(&self) -> Option<usize> {
        self.0
            .iter()
            .enumerate()
            .find(|(_, w)| **w != 0)
            .map(|(i, w)| i * 64 + w.trailing_zeros() as usize)
    }

    /// The variables in the parity, ascending.
    ///
    /// Costs one step per variable *present*, not per bit examined. The
    /// difference is not cosmetic: reduction indexes the polynomial by
    /// variable thousands of times over, and a parity over 400 variables
    /// spans seven words, so a per-bit walk pays 448 tests to report a
    /// handful of members.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        let mut word = 0usize;
        let mut cur = self.0.first().copied().unwrap_or(0);
        std::iter::from_fn(move || loop {
            if cur != 0 {
                let b = cur.trailing_zeros() as usize;
                cur &= cur - 1;
                return Some(word * 64 + b);
            }
            word += 1;
            cur = *self.0.get(word)?;
        })
    }

    /// Parity of the intersection — `⟨self, assignment⟩` over `𝔽₂`.
    pub fn parity_of(&self, assignment: &Mask) -> bool {
        let n = self.0.len().min(assignment.0.len());
        (0..n).fold(0u32, |a, i| a + (self.0[i] & assignment.0[i]).count_ones()) % 2 == 1
    }
}

/// A monomial: the product of a set of parities, each in `{0, 1}`, so
/// the product is `1` exactly when every parity is `1`. A set rather
/// than a multiset, because `p² = p` for a bit.
pub type Term = Vec<Mask>;

fn canonical(mut t: Term) -> Term {
    t.sort();
    t.dedup();
    t
}

// ── the state ────────────────────────────────────────────────────────

/// A state held as a reduced sum over paths.
#[derive(Clone, Debug)]
pub struct PathSum {
    qubits: usize,
    /// Per qubit: the affine output form `(parity mask, constant)`.
    forms: Vec<(Mask, bool)>,
    /// The phase polynomial: monomial → turn.
    poly: HashMap<Term, Turn>,
    /// Scale: the state carries a factor `2^{e_half / 2}`.
    e_half: i64,
    /// Global phase, in turns.
    phase: Turn,
    /// Path variables allocated so far.
    nvars: usize,
    /// Variables still under the sum.
    active: Mask,
    /// The state reduced to zero.
    zero: bool,
    /// Free input variables `0..inputs` — `0` for a state.
    inputs: usize,
    /// Rule-V splits performed — the reduction's measured work.
    splits: u64,
}

/// Why a surviving internal variable resisted elimination — the output
/// of [`PathSum::stall_census`].
///
/// The split that matters is structural-versus-alignment. `Shape`,
/// `Coupling` and `Pivot` are facts about the monomial structure and no
/// change of scalar algebra touches them. `Alignment` is a fact about
/// *where on the circle* the self-coefficient landed, and a sector whose
/// sums close at different points would reach exactly those.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stall {
    /// The variable appears inside a compound monomial, or in more than
    /// two masks. Structural.
    Shape {
        /// Degree of the widest monomial trapping the variable — the
        /// number of parities multiplied together. Degree 2 is the `n²`
        /// stratum, degree 3 the `n³`.
        degree: usize,
    },
    /// The shape is right but a coupling term is off a half turn,
    /// carrying the offending coefficient. Structural.
    Coupling(Turn),
    /// Rule E would fire, but every pivot in the coupling is a free
    /// input and the operator has no such constraint to assert.
    Pivot,
    /// Shape and couplings are exactly what a rule wants; only the
    /// self-coefficient (carried here) is off every point where the
    /// elliptic Gauss sum closes. This is the reachable case.
    Alignment(Turn),
}

impl Stall {
    /// Whether a different signature could plausibly reach this — true
    /// only for [`Stall::Alignment`].
    pub fn is_alignment(&self) -> bool {
        matches!(self, Stall::Alignment(_))
    }
}

impl PathSum {
    /// `|0…0⟩` on `qubits` qubits: no path variables, all forms constant.
    pub fn new(qubits: usize) -> Self {
        PathSum {
            qubits,
            forms: vec![(Mask::zero(), false); qubits],
            poly: HashMap::new(),
            e_half: 0,
            phase: 0,
            nvars: 0,
            active: Mask::zero(),
            zero: false,
            inputs: 0,
            splits: 0,
        }
    }

    /// Register width.
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    /// Path variables allocated over the circuit's life.
    pub fn allocated_vars(&self) -> usize {
        self.nvars
    }

    /// Monomials currently in the phase polynomial.
    pub fn terms(&self) -> usize {
        self.poly.len()
    }

    /// Whether reduction proved the state is exactly zero.
    pub fn is_zero(&self) -> bool {
        self.zero
    }

    /// Rule-V splits performed so far — what reduction actually cost.
    ///
    /// Rules E and G each retire a variable and are bounded by the wall
    /// count; only V can fire repeatedly, trading one monomial for
    /// three. So this counter, not the width and not the gate count, is
    /// what reduction time tracks, and it is reported rather than
    /// predicted.
    pub fn splits(&self) -> u64 {
        self.splits
    }

    /// **`h*`**: internal variables that survived reduction — the state's
    /// irreducible topological content, and the exponent of readout.
    pub fn internal_vars(&self) -> usize {
        self.active.without(&self.live_union()).count()
    }

    /// How many monomials sit at each **degree** — the number of parities
    /// multiplied together in a term.
    ///
    /// Degree 1 is a bare parity; degree 2 is the `n²` stratum, degree 3
    /// the `n³`. Rule V only ever splits a variable out of a compound
    /// monomial, so this profile is what the structural stalls are made
    /// of, and it says which stratum a representation would have to hold
    /// to reach them.
    pub fn degree_profile(&self) -> std::collections::BTreeMap<usize, usize> {
        let mut out = std::collections::BTreeMap::new();
        for t in self.poly.keys() {
            *out.entry(t.len()).or_insert(0) += 1;
        }
        out
    }

    /// Every term mentioning `v`, as `(degree, coefficient)`.
    ///
    /// The diagnostic behind a cubic stall: it says which coefficients a
    /// rule would have to close over, so "would a richer normalization
    /// reach this?" becomes a question about the actual numbers rather
    /// than about the shape alone.
    pub fn terms_containing(&self, v: usize) -> Vec<(usize, Turn)> {
        self.poly
            .iter()
            .filter(|(t, _)| t.iter().any(|m| m.bit(v)))
            .map(|(t, c)| (t.len(), *c))
            .collect()
    }

    /// Why each surviving internal variable resisted elimination.
    ///
    /// [`internal_vars`](Self::internal_vars) reports *how many* variables
    /// survived. This reports *why*, and the distinction is what decides
    /// whether a wider algebra could ever help.
    ///
    /// A [`Stall::Shape`] or [`Stall::Coupling`] is structural: the
    /// variable sits inside a compound monomial, or a coupling term is
    /// off a half turn, and no choice of signature changes that.
    /// [`Stall::Pivot`] is the operator formulation refusing to constrain
    /// a free input. [`Stall::Alignment`] is the opposite of all three —
    /// the shape is exactly what a rule wants, and only the
    /// self-coefficient is off the four points where the elliptic Gauss
    /// sum closes (`0`, `¼`, `½`, `¾`). A `T` gate contributes an eighth,
    /// which is not one of them.
    ///
    /// So the alignment count is the *opportunity*: the number of
    /// variables a sector with different closure points could reach
    /// without touching the rewrite system's structure at all.
    pub fn stall_census(&self) -> Vec<(usize, Stall)> {
        let candidates = self.active.without(&self.live_union());
        let mut out = Vec::new();
        for v in candidates.iter() {
            let pmask = Mask::single(v);
            let mut self_c: Turn = 0;
            let mut couple = Mask::zero();
            let mut shape_ok = true;
            let mut bad_coupling = None;
            let mut mentioned = false;
            let mut worst_degree = 0usize;
            for (t, c) in &self.poly {
                if !t.iter().any(|m| m.bit(v)) {
                    continue;
                }
                mentioned = true;
                let holding: Vec<&Mask> = t.iter().filter(|m| m.bit(v)).collect();
                if holding.len() != 1 || *holding[0] != pmask || t.len() > 2 {
                    shape_ok = false;
                    worst_degree = worst_degree.max(t.len());
                    continue;
                }
                if t.len() == 1 {
                    self_c = *c;
                } else {
                    if *c != HALF {
                        bad_coupling = Some(*c);
                    }
                    if let Some(other) = t.iter().find(|m| !m.bit(v)) {
                        couple = couple.xor(other);
                    }
                }
            }
            if !mentioned {
                continue;
            }
            let stall = if !shape_ok {
                Stall::Shape { degree: worst_degree }
            } else if let Some(c) = bad_coupling {
                Stall::Coupling(c)
            } else if (self_c == 0 || self_c == HALF)
                && !couple.is_zero()
                && couple.and(&self.active).lowest().is_none()
            {
                Stall::Pivot
            } else {
                Stall::Alignment(self_c)
            };
            out.push((v, stall));
        }
        out
    }

    fn live_union(&self) -> Mask {
        let mut out = Mask::zero();
        for (m, _) in &self.forms {
            for v in m.iter() {
                out.set(v);
            }
        }
        out
    }

    /// Add `coeff` to a monomial, folding the empty monomial into the
    /// global phase and dropping monomials that cancel.
    fn add_term(&mut self, term: Term, coeff: Turn) {
        if coeff == 0 {
            return;
        }
        if term.is_empty() {
            self.phase = self.phase.wrapping_add(coeff);
            return;
        }
        match self.poly.entry(term) {
            Entry::Occupied(mut o) => {
                let v = o.get().wrapping_add(coeff);
                if v == 0 {
                    o.remove();
                } else {
                    *o.get_mut() = v;
                }
            }
            Entry::Vacant(slot) => {
                slot.insert(coeff);
            }
        }
    }
}

/// Expand `coeff · Π_i (parity(mask_i) ⊕ flip_i)` into parity monomials.
///
/// The shared engine: gate application, wall coupling and substitution
/// all reduce to this. A factor `(p ⊕ 1)` is `1 − p`, so it splits into
/// two branches with opposite sign; a factor whose mask is empty is the
/// constant `flip`.
pub fn expand_affine_product(factors: &[(Mask, bool)], coeff: Turn) -> Vec<(Term, Turn)> {
    let mut out: HashMap<Term, Turn> = HashMap::new();
    fn rec(
        i: usize,
        factors: &[(Mask, bool)],
        cur: &mut Vec<Mask>,
        c: Turn,
        out: &mut HashMap<Term, Turn>,
    ) {
        if i == factors.len() {
            let e = out.entry(canonical(cur.clone())).or_insert(0);
            *e = e.wrapping_add(c);
            return;
        }
        let (mask, flip) = &factors[i];
        if mask.is_zero() {
            if *flip {
                rec(i + 1, factors, cur, c, out);
            }
            // an empty mask with no flip is the constant 0: the whole
            // product vanishes
            return;
        }
        if !*flip {
            cur.push(mask.clone());
            rec(i + 1, factors, cur, c, out);
            cur.pop();
        } else {
            rec(i + 1, factors, cur, c, out);
            cur.push(mask.clone());
            rec(i + 1, factors, cur, c.wrapping_neg(), out);
            cur.pop();
        }
    }
    let mut cur = Vec::new();
    rec(0, factors, &mut cur, coeff, &mut out);
    out.into_iter().filter(|(_, c)| *c != 0).collect()
}

// ── the gate letters ─────────────────────────────────────────────────

impl PathSum {
    fn check(&self, q: usize) -> Result<()> {
        if q >= self.qubits {
            return Err(Error::InvalidState(format!(
                "pathsum: qubit {q} outside a {}-qubit register",
                self.qubits
            )));
        }
        Ok(())
    }

    /// `X`: flip the qubit's affine constant.
    pub fn x(&mut self, q: usize) -> Result<()> {
        self.check(q)?;
        self.forms[q].1 = !self.forms[q].1;
        Ok(())
    }

    /// A diagonal phase `ω^{turn · f_q}` on one qubit — `Z` at a half
    /// turn, `S` at a quarter, `T` at an eighth.
    pub fn phase_on(&mut self, q: usize, turn: Turn) -> Result<()> {
        self.check(q)?;
        let f = self.forms[q].clone();
        for (t, c) in expand_affine_product(&[f], turn) {
            self.add_term(t, c);
        }
        Ok(())
    }

    /// `Z`.
    pub fn z(&mut self, q: usize) -> Result<()> {
        self.phase_on(q, HALF)
    }
    /// `S`.
    pub fn s(&mut self, q: usize) -> Result<()> {
        self.phase_on(q, QUARTER)
    }
    /// `S†`.
    pub fn sdg(&mut self, q: usize) -> Result<()> {
        self.phase_on(q, QUARTER.wrapping_neg())
    }
    /// `T`.
    pub fn t(&mut self, q: usize) -> Result<()> {
        self.phase_on(q, EIGHTH)
    }
    /// `T†`.
    pub fn tdg(&mut self, q: usize) -> Result<()> {
        self.phase_on(q, EIGHTH.wrapping_neg())
    }

    /// A diagonal phase on a product of forms — `CZ` at a half turn on
    /// two, `CCZ` at a half turn on three.
    pub fn phase_on_all(&mut self, qs: &[usize], turn: Turn) -> Result<()> {
        let mut factors = Vec::with_capacity(qs.len());
        for &q in qs {
            self.check(q)?;
            factors.push(self.forms[q].clone());
        }
        for (t, c) in expand_affine_product(&factors, turn) {
            self.add_term(t, c);
        }
        Ok(())
    }

    /// `CZ`.
    pub fn cz(&mut self, a: usize, b: usize) -> Result<()> {
        self.phase_on_all(&[a, b], HALF)
    }
    /// `CCZ`.
    pub fn ccz(&mut self, a: usize, b: usize, c: usize) -> Result<()> {
        self.phase_on_all(&[a, b, c], HALF)
    }

    /// `CNOT`: the target's form takes the control's XOR.
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<()> {
        self.check(control)?;
        self.check(target)?;
        if control == target {
            return Err(Error::InvalidState("pathsum: cnot on one qubit".into()));
        }
        let (cm, cc) = self.forms[control].clone();
        let (tm, tc) = self.forms[target].clone();
        self.forms[target] = (tm.xor(&cm), tc ^ cc);
        Ok(())
    }

    /// `SWAP`.
    pub fn swap(&mut self, a: usize, b: usize) -> Result<()> {
        self.check(a)?;
        self.check(b)?;
        self.forms.swap(a, b);
        Ok(())
    }

    /// A global phase, in turns. Free: it never touches a form.
    pub fn global_phase(&mut self, turn: Turn) {
        self.phase = self.phase.wrapping_add(turn);
    }

    /// A diagonal phase on the **XOR** of several forms — what `rzz`
    /// needs, and one factor rather than a product, so it costs one
    /// monomial no matter how many qubits it spans.
    pub fn phase_on_parity(&mut self, qs: &[usize], turn: Turn) -> Result<()> {
        let mut mask = Mask::zero();
        let mut konst = false;
        for &q in qs {
            self.check(q)?;
            mask = mask.xor(&self.forms[q].0);
            konst ^= self.forms[q].1;
        }
        for (t, c) in expand_affine_product(&[(mask, konst)], turn) {
            self.add_term(t, c);
        }
        Ok(())
    }

    /// The **wall**: the only letter that introduces a path variable.
    ///
    /// A fresh `y` is coupled to the qubit's current form by a half-turn
    /// term, the form becomes `y`, and the scale picks up `1/√2`.
    pub fn h(&mut self, q: usize) -> Result<()> {
        self.check(q)?;
        let y = self.nvars;
        self.nvars += 1;
        self.active.set(y);
        let ymask = Mask::single(y);
        let f = self.forms[q].clone();
        for (t, c) in expand_affine_product(&[f, (ymask.clone(), false)], HALF) {
            self.add_term(t, c);
        }
        self.forms[q] = (ymask, false);
        self.e_half -= 1;
        Ok(())
    }
}

// ── the crate's front door ───────────────────────────────────────────

impl PathSum {
    /// Run a [`Circuit`] as a path sum, then reduce.
    ///
    /// This is the point of contact with the rest of the crate: the same
    /// circuit that a [`DenseState`](crate::DenseState) evolves can be
    /// *composed* here instead, and the two agree amplitude for
    /// amplitude — including global phase, which a representation that
    /// only ever reports probabilities is free to lose and this one is
    /// not.
    ///
    /// The supported letters are the dyadic fragment plus the rotations
    /// that reduce to it. Anything else — a continuously parameterised
    /// `u3`, a gate at an angle that is not `2π·k/2^m` — is **refused
    /// with its name**, not approximated. A representation that is exact
    /// on a fragment should say where the fragment ends.
    /// Run a [`Circuit`] as a path sum, then reduce.
    ///
    /// This is the point of contact with the rest of the crate: the same
    /// circuit that a [`DenseState`](crate::DenseState) evolves can be
    /// *composed* here instead, and the two agree amplitude for
    /// amplitude — including global phase, which a representation that
    /// only ever reports probabilities is free to lose and this one is
    /// not.
    ///
    /// The supported letters are the dyadic fragment plus the rotations
    /// that reduce to it. Anything else — a continuously parameterised
    /// `u3`, a gate at an angle that is not `2π·k/2^m` — is **refused
    /// with its name**, not approximated. A representation that is exact
    /// on a fragment should say where the fragment ends.
    pub fn from_circuit(circuit: &Circuit<C64>) -> Result<PathSum> {
        let mut ps = PathSum::new(circuit.num_qubits());
        ps.apply_circuit(circuit, false)?;
        ps.reduce();
        Ok(ps)
    }

    /// One gate, optionally inverted.
    fn apply_op(&mut self, op: &Op<C64>, dagger: bool) -> Result<()> {
        let Op::Named {
            name,
            params,
            qubits: qs,
        } = op
        else {
            return Err(Error::InvalidState(
                "pathsum: only named registry gates compose as path sums".into(),
            ));
        };
        // Inverting a gate of this fragment is either a letter swap or a
        // negated angle; every other letter here is self-inverse.
        let (owned_name, params) = if dagger {
            match name.as_str() {
                "s" => ("sdg".to_string(), params.clone()),
                "sdg" => ("s".to_string(), params.clone()),
                "t" => ("tdg".to_string(), params.clone()),
                "tdg" => ("t".to_string(), params.clone()),
                "sx" => ("sxdg".to_string(), params.clone()),
                "sxdg" => ("sx".to_string(), params.clone()),
                "p" | "phase" | "rz" | "rx" | "cp" | "cphase" | "rzz" => {
                    (name.clone(), params.iter().map(|x| -x).collect())
                }
                _ => (name.clone(), params.clone()),
            }
        } else {
            (name.clone(), params.clone())
        };
        let name = &owned_name;
        let p = |i: usize| params.get(i).copied().unwrap_or(0.0);
        // Report the gate's own parameter on refusal. A rotation
        // decomposes into a turn *and* a half-turn of global phase,
        // so the raw converter would otherwise name an angle the
        // caller never wrote.
        let ang = |theta: f64| -> Result<Turn> {
            turn_from_radians(theta).map_err(|_| {
                Error::InvalidState(format!(
                    "pathsum: `{name}` at {} rad is not a dyadic angle; \
                     this representation refuses it rather than rounding it",
                    p(0)
                ))
            })
        };
        match (name.as_str(), qs.len()) {
            ("id", _) => {}
            ("h", 1) => self.h(qs[0])?,
            ("x" | "not", 1) => self.x(qs[0])?,
            ("z", 1) => self.z(qs[0])?,
            ("s", 1) => self.s(qs[0])?,
            ("sdg", 1) => self.sdg(qs[0])?,
            ("t", 1) => self.t(qs[0])?,
            ("tdg", 1) => self.tdg(qs[0])?,
            ("y", 1) => {
                // Y = i·X·Z, so Z, then X, then the quarter turn.
                self.z(qs[0])?;
                self.x(qs[0])?;
                self.global_phase(QUARTER);
            }
            ("p" | "phase", 1) => self.phase_on(qs[0], ang(p(0))?)?,
            ("rz", 1) => {
                // exp(−iθZ/2) = e^{−iθ/2}·diag(1, e^{iθ})
                self.global_phase(ang(-p(0) / 2.0)?);
                self.phase_on(qs[0], ang(p(0))?)?;
            }
            ("rx", 1) => {
                self.h(qs[0])?;
                self.global_phase(ang(-p(0) / 2.0)?);
                self.phase_on(qs[0], ang(p(0))?)?;
                self.h(qs[0])?;
            }
            ("sx", 1) => {
                self.global_phase(EIGHTH);
                self.h(qs[0])?;
                self.global_phase(turn_from_dyadic(-1, 3)?);
                self.phase_on(qs[0], QUARTER)?;
                self.h(qs[0])?;
            }
            ("sxdg", 1) => {
                self.global_phase(EIGHTH.wrapping_neg());
                self.h(qs[0])?;
                self.global_phase(turn_from_dyadic(1, 3)?);
                self.phase_on(qs[0], QUARTER.wrapping_neg())?;
                self.h(qs[0])?;
            }
            ("cx" | "cnot", 2) => self.cnot(qs[0], qs[1])?,
            ("cz", 2) => self.cz(qs[0], qs[1])?,
            ("swap", 2) => self.swap(qs[0], qs[1])?,
            ("cp" | "cphase", 2) => self.phase_on_all(&[qs[0], qs[1]], ang(p(0))?)?,
            ("rzz", 2) => {
                // exp(−iθ Z⊗Z/2) = e^{−iθ/2}·ω^{θ·(a⊕b)}
                self.global_phase(ang(-p(0) / 2.0)?);
                self.phase_on_parity(&[qs[0], qs[1]], ang(p(0))?)?;
            }
            ("ccz", 3) => self.ccz(qs[0], qs[1], qs[2])?,
            ("ccx", 3) => {
                self.h(qs[2])?;
                self.ccz(qs[0], qs[1], qs[2])?;
                self.h(qs[2])?;
            }
            _ => {
                return Err(Error::InvalidState(format!(
                    "pathsum: `{name}` on {} qubits is outside the dyadic fragment",
                    qs.len()
                )))
            }
        }
        Ok(())
    }

}

// ── reduction ────────────────────────────────────────────────────────

impl PathSum {
    /// Substitute `y_p := parity(mask) ⊕ c` everywhere.
    fn substitute(&mut self, p: usize, mask: &Mask, c: bool) {
        let pm = Mask::single(p);
        let items: Vec<(Term, Turn)> = self.poly.drain().collect();
        for (term, coeff) in items {
            if !term.iter().any(|m| m.bit(p)) {
                self.add_term(term, coeff);
                continue;
            }
            let factors: Vec<(Mask, bool)> = term
                .iter()
                .map(|m| {
                    if m.bit(p) {
                        (m.xor(&pm).xor(mask), c)
                    } else {
                        (m.clone(), false)
                    }
                })
                .collect();
            for (t2, c2) in expand_affine_product(&factors, coeff) {
                self.add_term(t2, c2);
            }
        }
        for (m, cc) in self.forms.iter_mut() {
            if m.bit(p) {
                *m = m.xor(&pm).xor(mask);
                *cc ^= c;
            }
        }
    }

    /// **Rule V** — split a compound parity holding an internal
    /// variable: `p(K) = y + p(r) − 2·y·p(r)`. Strictly monotone in the
    /// parities' size, and it exposes the couplings E and G need.
    ///
    /// The victim is chosen **greedily and deterministically**: the
    /// narrowest compound parity, in the shortest term, breaking ties on
    /// the keys themselves. Both halves of that matter. Splitting trades
    /// one monomial for three, so the narrowest parity is the cheapest
    /// trade and the one likeliest to expose a bare `y` that E or G can
    /// then consume — and scanning `poly.keys()` for the first candidate
    /// instead picks by `HashMap` order, which is seeded per process.
    /// That made reduction nondeterministic in *cost*: the same 32-qubit
    /// Clifford circuit reduced to between 82 and 103 monomials across
    /// five runs, and at 64 qubits the spread ran from seconds to minutes
    /// on identical input. The answer never moved; the work to reach it
    /// swung by two orders of magnitude.
    fn split(&mut self) -> bool {
        let internal = self.active.without(&self.live_union());
        let mut found: Option<(Term, Mask)> = None;
        let mut best = (usize::MAX, usize::MAX);
        for term in self.poly.keys() {
            for k in term {
                if k.count() < 2 || !k.intersects(&internal) {
                    continue;
                }
                let score = (k.count(), term.len());
                let better = match &found {
                    None => true,
                    Some((t0, k0)) => (score, k, term) < (best, k0, t0),
                };
                if better {
                    best = score;
                    found = Some((term.clone(), k.clone()));
                }
            }
        }
        let Some((term, k)) = found else { return false };
        let Some(coeff) = self.poly.remove(&term) else {
            return false;
        };
        let inner = k.and(&internal);
        let Some(pv) = inner.lowest() else { return false };
        let p = Mask::single(pv);
        let r = k.xor(&p);
        let rest: Vec<Mask> = term.iter().filter(|m| **m != k).cloned().collect();
        self.splits += 1;

        let mut a = rest.clone();
        a.push(p.clone());
        self.add_term(canonical(a), coeff);
        let mut b = rest.clone();
        b.push(r.clone());
        self.add_term(canonical(b), coeff);
        let mut c = rest;
        c.push(p);
        c.push(r);
        self.add_term(canonical(c), coeff.wrapping_mul(2).wrapping_neg());
        true
    }

    /// One sweep of rules **E** and **G**, plus the trivial case of a
    /// variable appearing nowhere (which simply doubles the scale).
    ///
    /// The index is built once per sweep rather than once per variable,
    /// covers only the variables a rule could actually fire on (internal
    /// and still under the sum), and holds *positions* rather than copies
    /// of the monomials. The last point is the expensive one: keying by
    /// cloned `Term` copies a monomial once for every variable it
    /// mentions, and on a wide Clifford circuit a monomial mentions
    /// hundreds.
    fn pass(&mut self) -> bool {
        let mut fired = false;
        loop {
            if self.zero {
                return fired;
            }
            let live = self.live_union();
            // Only internal, still-summed variables can fire a rule.
            let candidates = self.active.without(&live);
            if candidates.is_zero() {
                return fired;
            }
            // variable -> positions in `keys` of the monomials mentioning it
            let keys: Vec<Term> = self.poly.keys().cloned().collect();
            let mut index: HashMap<usize, Vec<usize>> = HashMap::new();
            for (ki, t) in keys.iter().enumerate() {
                let mut seen = Mask::zero();
                for m in t {
                    for v in m.and(&candidates).iter() {
                        if !seen.bit(v) {
                            seen.set(v);
                            index.entry(v).or_default().push(ki);
                        }
                    }
                }
            }
            let mut changed = false;
            for v in candidates.iter() {
                let Some(at) = index.get(&v) else {
                    // the variable is under the sum but in no term: the
                    // sum over it is just a factor of two
                    self.e_half += 2;
                    self.active.clear(v);
                    changed = true;
                    fired = true;
                    break;
                };
                let terms: Vec<(Term, Turn)> = at
                    .iter()
                    .filter_map(|&ki| self.poly.get(&keys[ki]).map(|c| (keys[ki].clone(), *c)))
                    .collect();
                if terms.is_empty() {
                    self.e_half += 2;
                    self.active.clear(v);
                    changed = true;
                    fired = true;
                    break;
                }
                // The rules need every term to be `y` alone, or `y`
                // times exactly one other parity at a half turn.
                let mut self_c: Turn = 0;
                let mut couple = Mask::zero();
                let mut ok = true;
                let pmask = Mask::single(v);
                for (t, c) in &terms {
                    let holding: Vec<&Mask> = t.iter().filter(|m| m.bit(v)).collect();
                    if holding.len() != 1 || *holding[0] != pmask || t.len() > 2 {
                        ok = false;
                        break;
                    }
                    if t.len() == 1 {
                        self_c = *c;
                    } else {
                        let other = t.iter().find(|m| !m.bit(v)).unwrap();
                        if *c != HALF {
                            ok = false;
                            break;
                        }
                        couple = couple.xor(other);
                    }
                }
                if !ok {
                    continue;
                }

                if self_c == 0 || self_c == HALF {
                    // Rule E: the sum over y forces the affine
                    // constraint `couple = const`.
                    //
                    // The pivot has to be a variable we are *summing
                    // over*. Once input variables exist (the operator
                    // formulation) `couple` can be a condition on the
                    // inputs alone, and an input is free — substituting
                    // into one would assert a constraint the operator
                    // does not have. When that happens the rule simply
                    // does not fire.
                    let konst = self_c == HALF;
                    let piv = couple.and(&self.active).lowest();
                    if !couple.is_zero() && piv.is_none() {
                        continue;
                    }
                    for (t, _) in &terms {
                        self.poly.remove(t);
                    }
                    if couple.is_zero() {
                        if konst {
                            self.zero = true;
                            return true;
                        }
                        self.e_half += 2;
                        self.active.clear(v);
                    } else {
                        let piv = piv.expect("checked above");
                        self.e_half += 2;
                        self.active.clear(v);
                        let rest = couple.xor(&Mask::single(piv));
                        self.substitute(piv, &rest, konst);
                        self.active.clear(piv);
                    }
                    changed = true;
                    fired = true;
                    break;
                }
                if self_c == QUARTER || self_c == THREE_QUARTER {
                    // Rule G: a Gauss sum removes the variable for a
                    // ±1/8 phase and a ∓1/4 parity term.
                    for (t, _) in &terms {
                        self.poly.remove(t);
                    }
                    let positive = self_c == QUARTER;
                    self.phase = if positive {
                        self.phase.wrapping_add(EIGHTH)
                    } else {
                        self.phase.wrapping_sub(EIGHTH)
                    };
                    if !couple.is_zero() {
                        let c = if positive {
                            QUARTER.wrapping_neg()
                        } else {
                            QUARTER
                        };
                        self.add_term(vec![couple.clone()], c);
                    }
                    self.e_half += 1;
                    self.active.clear(v);
                    changed = true;
                    fired = true;
                    break;
                }
            }
            if !changed {
                return fired;
            }
        }
    }

    /// Reduce to normal form: sweep E and G, split when they stall, stop
    /// when neither can fire.
    pub fn reduce(&mut self) {
        let mut guard = 200_000u32;
        while guard > 0 && !self.zero {
            guard -= 1;
            if self.pass() {
                continue;
            }
            if !self.split() {
                break;
            }
        }
    }
}

// ── readout ──────────────────────────────────────────────────────────

impl PathSum {
    /// Sum `ω^{phase + Q}` over the surviving path variables.
    ///
    /// Costs `2^{used}` where `used` counts variables the polynomial
    /// still mentions — never `2^n`.
    fn enumerate_scalar(&self) -> (C64, usize) {
        if self.zero {
            return (C64::new(0.0, 0.0), 0);
        }
        let mut used = Mask::zero();
        for t in self.poly.keys() {
            for m in t {
                for v in m.iter() {
                    used.set(v);
                }
            }
        }
        used = used.and(&self.active);
        let vars: Vec<usize> = used.iter().collect();
        let free = self.active.count() - vars.len();

        let mut total = C64::new(0.0, 0.0);
        for assign in 0..(1u64 << vars.len().min(63)) {
            let mut a = Mask::zero();
            for (i, &v) in vars.iter().enumerate() {
                if assign >> i & 1 == 1 {
                    a.set(v);
                }
            }
            let mut turn = self.phase;
            for (t, c) in &self.poly {
                if t.iter().all(|m| m.parity_of(&a)) {
                    turn = turn.wrapping_add(*c);
                }
            }
            total += turn_to_c64(turn);
        }
        let scale = 2f64.powf((self.e_half as f64) / 2.0 + free as f64);
        (total * C64::new(scale, 0.0), 1usize << vars.len())
    }

    /// `⟨bits|ψ⟩`, by substituting the output constraints and summing
    /// over only the variables that survive — `O(2^{h*})`, not `O(2^n)`.
    pub fn amplitude(&self, bits: u64) -> C64 {
        let mut ps = self.clone();
        for q in 0..ps.qubits {
            let (m, c) = ps.forms[q].clone();
            let want = bits >> q & 1 == 1;
            if m.is_zero() {
                if c != want {
                    return C64::new(0.0, 0.0);
                }
                continue;
            }
            let piv = m.lowest().unwrap();
            let rest = m.xor(&Mask::single(piv));
            ps.substitute(piv, &rest, c ^ want);
            ps.active.clear(piv);
            ps.forms[q] = (Mask::zero(), want);
        }
        ps.reduce();
        ps.enumerate_scalar().0
    }

    /// Every amplitude, for checking against a dense state. Exponential
    /// by construction — this is the verification path, not the point.
    pub fn to_dense(&self) -> Vec<C64> {
        (0..(1u64 << self.qubits))
            .map(|b| self.amplitude(b))
            .collect()
    }
}

// ── the operator formulation: what needs no tableau ──────────────────

impl PathSum {
    /// The **identity operator** on `qubits` qubits: `|x⟩ ↦ |x⟩`.
    ///
    /// The one change that turns a state into an operator — each qubit's
    /// form starts as its own *input variable* rather than a constant.
    /// Input variables are free: they are never summed over, so
    /// reduction leaves them alone and [`PathSum::internal_vars`] counts
    /// only the path variables the walls introduced.
    ///
    /// This is the thing a stabilizer tableau cannot do. A tableau
    /// represents a stabilizer *state*; it cannot hold a non-Clifford
    /// operator at all, so a tableau-based tool has to decide up front
    /// which fragment it is in and branch. The path sum has one code
    /// path for every circuit, and `h*` *reports* where it landed.
    pub fn identity(qubits: usize) -> PathSum {
        let mut ps = PathSum::new(qubits);
        ps.nvars = qubits;
        ps.inputs = qubits;
        for (q, form) in ps.forms.iter_mut().enumerate() {
            *form = (Mask::single(q), false);
        }
        ps
    }

    /// Input variables — `0` for a state, the width for an operator.
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// Whether this is the identity operator exactly, global phase
    /// included.
    ///
    /// The decision procedure behind [`equivalent`]: every output form is
    /// its own input, the phase polynomial is empty, the scale is one and
    /// nothing is left under a sum.
    pub fn is_identity(&self) -> bool {
        self.is_identity_up_to_phase() && self.phase == 0
    }

    /// The same, ignoring an overall phase — which no measurement sees.
    pub fn is_identity_up_to_phase(&self) -> bool {
        !self.zero
            && self.inputs == self.qubits
            && self.poly.is_empty()
            && self.e_half == 0
            && self.active.count() == 0
            && self
                .forms
                .iter()
                .enumerate()
                .all(|(q, (m, c))| !*c && *m == Mask::single(q))
    }

    /// The global phase, in turns.
    pub fn phase(&self) -> Turn {
        self.phase
    }



    /// Apply a circuit's gates, optionally inverted.
    pub fn apply_circuit(&mut self, circuit: &Circuit<C64>, dagger: bool) -> Result<()> {
        if circuit.num_qubits() > self.qubits {
            return Err(Error::InvalidState(format!(
                "pathsum: a {}-qubit circuit does not fit a {}-qubit path sum",
                circuit.num_qubits(),
                self.qubits
            )));
        }
        let ops: Vec<&Op<C64>> = if dagger {
            circuit.ops().iter().rev().collect()
        } else {
            circuit.ops().iter().collect()
        };
        for op in ops {
            self.apply_op(op, dagger)?;
        }
        Ok(())
    }
}

/// Whether two circuits are the same unitary, decided by reduction.
///
/// Builds `V† ∘ U` as one operator path sum and reduces it: the circuits
/// agree exactly when what is left is the identity. No tableau, no
/// simulation, and no `2^n` anything — and, unlike a tableau method, it
/// does not need either circuit to be Clifford.
///
/// Sound in both directions where it answers: reduction only ever
/// rewrites the sum into an equal one, so `true` is a proof. `false` is
/// weaker — the rewrite system is complete for the Clifford fragment but
/// not in general, so a `false` means *this procedure did not reduce it
/// to the identity*, which `equivalent_verdict` states rather than
/// hides.
pub fn equivalent(a: &Circuit<C64>, b: &Circuit<C64>) -> Result<bool> {
    Ok(equivalent_verdict(a, b)?.0)
}

/// [`equivalent`], plus the surviving `h*` — `0` with a non-identity
/// result means the circuits are genuinely different; `> 0` means the
/// reduction stalled and the answer is "not proved equal", not "unequal".
pub fn equivalent_verdict(a: &Circuit<C64>, b: &Circuit<C64>) -> Result<(bool, usize)> {
    let n = a.num_qubits().max(b.num_qubits());
    let mut ps = PathSum::identity(n);
    ps.apply_circuit(a, false)?;
    ps.apply_circuit(b, true)?;
    ps.reduce();
    Ok((ps.is_identity_up_to_phase(), ps.internal_vars()))
}

/// The operator a circuit *is*, reduced — with `h*` its exponent.
pub fn operator(circuit: &Circuit<C64>) -> Result<PathSum> {
    let mut ps = PathSum::identity(circuit.num_qubits());
    ps.apply_circuit(circuit, false)?;
    ps.reduce();
    Ok(ps)
}
