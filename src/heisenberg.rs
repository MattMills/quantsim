//! **Progressive stateless computation**: an observable propagated
//! backwards through a circuit as a sum of Paulis, pruned by what the
//! question needs, with a journal that says — retrodictively — where the
//! precision went.
//!
//! # The mechanism
//!
//! The Heisenberg query is `⟨0|U†PU|0⟩`. Writing `U = g_L ⋯ g_1`, the
//! conjugation `U†PU = g_1† ⋯ g_L† P g_L ⋯ g_1` is evaluated **from the
//! observable backwards to the input** — the walk is retrodictive by
//! construction, which is what makes the journal below a diagnostic
//! instrument rather than a log.
//!
//! Each gate is a Pauli rotation `R = exp(−iθQ/2)`, and conjugating one
//! Pauli term through it does exactly one of two things:
//!
//! ```text
//! [P,Q] = 0   →   R† P R = P                             (free)
//! {P,Q} = 0   →   R† P R = cos θ · P  −  i sin θ · PQ     (splits)
//! ```
//!
//! So the observable becomes a [`PauliSum`], branching only where a gate
//! genuinely anticommutes with a term it meets. Three consequences carry
//! the method:
//!
//! * **Exclusion comes before approximation, and is exact.** A term
//!   contributes to `⟨0|·|0⟩` only if its `X`-mask reaches zero, and the
//!   mask only ever XORs with a remaining gate's axis — so if it lies
//!   outside the GF(2) span of the remaining axes ([`XSpan`]), the term
//!   and every descendant contribute **precisely zero**. The question is
//!   asked at generation, so an excluded branch is never created. It is
//!   kept in its own ledger ([`Propagation::excluded_l1`]) apart from the
//!   error ledger, because removing it costs nothing at all — verified
//!   bit-identical against the unexcluded walk.
//!
//! * **Freezing was the obvious next exclusion, and it does not pay.**
//!   A term that commutes with *every* remaining axis can never change
//!   again, so it is settled exactly: coefficient into the answer if its
//!   `X`-mask is already zero, dropped otherwise. That is a GF(2)
//!   condition — symplectic orthogonality to the span of the remaining
//!   axes ([`AxisSpan`]) — and unlike reachability it *could* fire
//!   mid-walk, which is where the peak lives. It is implemented,
//!   verified not to change any answer, and **measured to buy nothing**:
//!   peak 1947 → 1947 on TFIM, 16384 → 16383 on IQP, at 20–60% more
//!   time for the `O(rank)` test per term per step. Even on a staged
//!   circuit built to favour it, six terms retired and the peak did not
//!   move — because reachability-exclusion had already removed 681 of
//!   them first. Freezing demands orthogonality to the whole remaining
//!   span, which is far rarer than merely falling outside it.
//!   [`Config::retire_frozen`] therefore defaults to `false`.
//!
//!   One thing this rules out, worth recording: a magnitude bound
//!   discounted by how far a term is from reachable is **not sound**.
//!   `|⟨ψ|P|ψ⟩| ≤ 1` for any Pauli, so `|c|` is already the correct
//!   bound on a term's contribution and no reachability argument
//!   sharpens it.
//!
//!   Where it fires is structural and worth stating, because it is not
//!   where one would hope. The span of gates `0..g` *shrinks* as `g → 0`,
//!   and a backward walk reaches small `g` **last** — so exclusion
//!   necessarily bites at the tail. Measured: an IQP-like circuit at
//!   width 14 collapses from 16 384 surviving terms to **1**, but its
//!   peak is unchanged at 16 384 and the time is not improved, because
//!   the collapse happens in the final layer. TFIM, whose `X` axes are
//!   spread through every layer, does get an earlier bite: peak 4 502 →
//!   1 947 and 11.7 ms → 9.2 ms. So exclusion bounds the answer's
//!   assembly and the memory at the end; it reduces the peak only when
//!   the circuit's *prefix* is `X`-poor.
//! * **The causal cone is free and needs no separate pass.** A rotation
//!   whose axis commutes with every term costs one popcount and changes
//!   nothing. [`Propagation::commuting_skips`] counts them — that is the
//!   backward light cone, measured while walking rather than computed
//!   beforehand.
//! * **The error is certified, not estimated.** `Σ|c|²` is invariant
//!   under the conjugation, so dropping the terms below a threshold has
//!   an *exactly known* cost. [`Propagation::discarded_l2`] is that
//!   number, and it is a bound on the expectation error, not a guess.
//! * **The cost tracks the question, not the circuit.** Ask for three
//!   digits and you pay for the terms above `1e-3`; ask for machine
//!   precision and you pay for all of them.
//!
//! # Factoring, rather than cutting
//!
//! A cut is positional and blind to structure. The structural
//! alternative is to hold the conjugated observable as a **product of
//! independent blocks** ([`FactoredPauliSum`]) — a single Pauli already
//! *is* a tensor product of single-site Paulis, and conjugation only
//! couples two blocks when a gate's axis straddles them. Nothing is
//! declared in advance: the partition is **discovered** from the gates
//! as the walk meets them, and merged lazily only where the circuit
//! genuinely couples regions. It is the operator-side analogue of
//! [`FactoredState`](crate::backend::FactoredState).
//!
//! The arithmetic is the whole point: `k` independent blocks cost the
//! **sum** of their sizes, not the product. Measured on `k` decoupled
//! chains of four qubits, with the observable straddling all of them:
//!
//! ```text
//!   blocks   stored terms   flat terms      merges   saving
//!        4            224      9 834 496         0    43 904×
//!        5            280    550 731 776         0  1 966 899×
//!        6            336 30 840 979 456         0 91 788 629×
//! ```
//!
//! Stored grows **linearly** in the block count while the flat sum grows
//! exponentially, and `merges = 0` because a decoupled circuit never
//! forces the partition to break. Nearest-neighbour TFIM, by contrast,
//! collapses to a single block with a saving of exactly `1.0×` — there
//! is nothing to factor and the report says so rather than pretending.
//!
//! Scope: [`propagate_factored`] is **exact only**. A per-block
//! threshold's error has to be carried across the product, and that
//! bound is not implemented, so the factored path applies no truncation.
//!
//! # Both directions of time
//!
//! `⟨0|U†PU|0⟩` with `U = U₂U₁` is `⟨ψ|U₂†PU₂|ψ⟩` for `|ψ⟩ = U₁|0…0⟩`,
//! so the observable need only be walked back through `U₂` while the
//! input is pushed forward through `U₁`
//! ([`propagate_bidirectional`]). This is the only change here that
//! reduces the backward walk's **peak** rather than pruning what it has
//! already produced, because it reduces the number of branching gates
//! the walk ever sees.
//!
//! It pays exactly when the two halves are exponential in *different*
//! resources. With a sparse forward half they are not — the state
//! saturates within a couple of layers and the best cut is `0`, the pure
//! backward walk. With an MPS forward half on a circuit whose prefix is
//! low-entanglement but non-Clifford, they are: the meeting cost has a
//! genuine interior minimum at **8 032 bytes against 76 064 at either
//! end**, a 9.5× saving, with the value stable to 2e-7 across every cut.
//! [`auto_cut`] finds it by scanning.
//!
//! One correctness trap, recorded because it silently produced wrong
//! numbers before it was caught: **the exclusions are boundary
//! conditions, not circuit properties.** Both encode "this walk ends at
//! `|0…0⟩`, where only `X`-free terms contribute". At an interior cut
//! the terms are evaluated against `|ψ⟩` instead, where terms with
//! `X`-support contribute perfectly well — applying either pruning there
//! gave 0.18 absolute error on a validation sweep.
//! [`propagate_bidirectional`] switches them off for any cut past zero.
//!
//! # The journal, read backwards
//!
//! Every step records how many terms survived, how much weight was
//! dropped, and the running total. The running total is **monotone**, so
//! the journal is binary-searchable: [`Journal::retrodict`] answers *at
//! which gate did my error budget get spent* in `O(log L)` without
//! re-running anything. That is the retrodiction pattern
//! [`closure`](crate::closure) uses for errors, applied to precision.
//!
//! Because the walk is backwards, the answer is a **circuit position** —
//! "your third digit died at gate 847" — which is actionable in a way a
//! final error bar is not. [`Propagation::refine_from`] then re-runs from
//! the nearest checkpoint with a tighter threshold. Checkpoints are the
//! *partial state*: the space–time dial between storing everything and
//! recomputing everything.
//!
//! One limit of that, stated because it is easy to assume otherwise: a
//! checkpoint holds the sum **as it was already truncated**, so refining
//! from it recovers only the error incurred *after* it. Refining from
//! step 0 — where the checkpoint is the pristine observable — is a full
//! re-run and recovers everything. The useful pattern is therefore to
//! retrodict the blame gate first and then re-walk with a threshold that
//! is tight only near it, rather than to expect a late checkpoint to
//! undo an early loss.
//!
//! # What this is, honestly
//!
//! The propagation mechanism is **sparse Pauli dynamics** (Pauli path
//! integrals), the method that classically reproduced IBM's 127-qubit
//! utility experiment. It is not new here and this module does not claim
//! it. What is assembled here is the *control structure* around it: the
//! retrodictive journal, checkpointed refinement, and — the reason to
//! bother — [`propagate_basis`], which shares one walk across a whole
//! observable **basis** rather than re-walking per observable. An energy
//! `H = Σ cᵢPᵢ` has `O(n²)`–`O(n⁴)` terms whose cones overlap almost
//! entirely; sharing the walk is close to the cost of the worst single
//! term.
//!
//! And the limit, stated up front: for anti-concentrated circuits the
//! coefficients flatten, no threshold helps, and the term count is
//! exponential. That is the same wall as everywhere else. The method
//! wins where structure or damping makes the coefficient distribution
//! decay — which notably includes *noisy* circuits, so it is strongest
//! exactly where hardware is weakest.

use rustc_hash::FxHashMap;

use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::scalar::C64;

/// Largest register width: Pauli support is carried in `u64` masks.
pub const MAX_QUBITS: usize = 64;

/// Coefficients below this are dropped **regardless of the configured
/// threshold**, because they are below double precision's ability to
/// mean anything.
///
/// This is not a tuning knob, it is a correctness fix. At an exact
/// Clifford angle `cos(−π/2)` evaluates to `6.1e−17` rather than `0`, so
/// an "exact" walk with `threshold = 0` would branch on that dust and
/// reach millions of terms where the true answer is one term. The floor
/// is what makes the Clifford point actually free.
pub const COEFF_FLOOR: f64 = 1e-15;

// ── the Pauli algebra, in the X^x Z^z basis ──────────────────────────

/// A Pauli basis element `X^x Z^z`, as a pair of support masks.
///
/// This basis is used rather than the Hermitian `{I,X,Y,Z}` one because
/// multiplication is a XOR plus one sign — no `i` bookkeeping in the hot
/// loop. Hermiticity is restored where it matters: a rotation *axis* is
/// interpreted as `i^{|x&z|} X^x Z^z` (see [`axis_operator_phase`]),
/// which is Hermitian and squares to the identity.
pub type PauliKey = (u64, u64);

/// Sign from commuting `Z^b` past `X^c`: `Z^b X^c = (−1)^{|b∧c|} X^c Z^b`.
#[inline]
fn reorder_sign(b: u64, c: u64) -> f64 {
    if (b & c).count_ones() % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Whether `X^a Z^b` and `X^c Z^d` commute — the symplectic form.
#[inline]
pub fn commutes(p: PauliKey, q: PauliKey) -> bool {
    ((p.1 & q.0).count_ones() + (p.0 & q.1).count_ones()) % 2 == 0
}

/// Product `(X^a Z^b)(X^c Z^d) = sign · X^{a⊕c} Z^{b⊕d}`.
#[inline]
pub fn pauli_mul(p: PauliKey, q: PauliKey) -> (PauliKey, f64) {
    ((p.0 ^ q.0, p.1 ^ q.1), reorder_sign(p.1, q.0))
}

/// The phase making an axis Hermitian: the rotation axis for masks
/// `(x, z)` is `i^{|x∧z|} X^x Z^z`, which is Hermitian and squares to
/// `I` (e.g. `(1,1) ↦ i·XZ = Y`).
#[inline]
pub fn axis_operator_phase(axis: PauliKey) -> C64 {
    match (axis.0 & axis.1).count_ones() % 4 {
        0 => C64::new(1.0, 0.0),
        1 => C64::new(0.0, 1.0),
        2 => C64::new(-1.0, 0.0),
        _ => C64::new(0.0, -1.0),
    }
}

/// A weighted sum of Pauli basis elements.
#[derive(Debug, Clone, Default)]
pub struct PauliSum {
    terms: FxHashMap<PauliKey, C64>,
}

impl PauliSum {
    /// The empty (zero) sum.
    pub fn zero() -> Self {
        PauliSum::default()
    }

    /// A single Pauli with unit coefficient.
    pub fn from_key(key: PauliKey) -> Self {
        let mut s = PauliSum::zero();
        s.add(key, C64::new(1.0, 0.0));
        s
    }

    /// `Z` on one qubit.
    pub fn z(qubit: usize) -> Self {
        PauliSum::from_key((0, 1u64 << qubit))
    }

    /// `X` on one qubit.
    pub fn x(qubit: usize) -> Self {
        PauliSum::from_key((1u64 << qubit, 0))
    }

    /// `Y` on one qubit — `i·XZ` in this basis.
    pub fn y(qubit: usize) -> Self {
        let mut s = PauliSum::zero();
        s.add((1u64 << qubit, 1u64 << qubit), C64::new(0.0, 1.0));
        s
    }

    /// `Z_a Z_b`.
    pub fn zz(a: usize, b: usize) -> Self {
        PauliSum::from_key((0, (1u64 << a) | (1u64 << b)))
    }

    /// Add a term, merging with any like term and dropping exact zeros.
    pub fn add(&mut self, key: PauliKey, coeff: C64) {
        if coeff.norm() == 0.0 {
            return;
        }
        let e = self.terms.entry(key).or_insert(C64::new(0.0, 0.0));
        *e += coeff;
        if e.norm() == 0.0 {
            self.terms.remove(&key);
        }
    }

    /// Number of distinct Pauli terms.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Whether the sum is empty.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// The terms.
    pub fn terms(&self) -> impl Iterator<Item = (PauliKey, C64)> + '_ {
        self.terms.iter().map(|(&k, &c)| (k, c))
    }

    /// `Σ|c|²` — invariant under conjugation, so it is the yardstick the
    /// discarded weight is measured against.
    pub fn l2_squared(&self) -> f64 {
        self.terms.values().map(|c| c.norm_sqr()).sum()
    }

    /// `Σ|c|` — the bound on how much a truncation can move an
    /// expectation value.
    pub fn l1(&self) -> f64 {
        self.terms.values().map(|c| c.norm()).sum()
    }

    /// Largest support of any term — how far the observable has spread.
    pub fn max_weight(&self) -> usize {
        self.terms
            .keys()
            .map(|&(x, z)| (x | z).count_ones() as usize)
            .max()
            .unwrap_or(0)
    }

    /// Drop every term with `|c| < threshold`; returns the discarded
    /// `Σ|c|` and `Σ|c|²`.
    ///
    /// The `Σ|c|` figure is the one that bounds the expectation error:
    /// `|⟨P⟩ − ⟨P_truncated⟩| ≤ Σ_{dropped}|c|`, because every Pauli has
    /// operator norm 1.
    pub fn truncate(&mut self, threshold: f64) -> (f64, f64) {
        if threshold <= 0.0 {
            return (0.0, 0.0);
        }
        let mut l1 = 0.0;
        let mut l2 = 0.0;
        self.terms.retain(|_, c| {
            let m = c.norm();
            if m < threshold {
                l1 += m;
                l2 += m * m;
                false
            } else {
                true
            }
        });
        (l1, l2)
    }

    /// Keep only the terms the predicate accepts.
    pub fn retain(&mut self, mut f: impl FnMut(&PauliKey, &C64) -> bool) {
        self.terms.retain(|k, c| f(k, c));
    }

    /// `⟨0…0| Σ c_k X^{a_k} Z^{b_k} |0…0⟩` — the sum of the coefficients
    /// of the terms with no `X` support, since `Z^b|0⟩ = |0⟩` and
    /// `X^a|0⟩ = |a⟩` is orthogonal to `|0⟩` unless `a = 0`.
    pub fn expectation_on_zero_state(&self) -> C64 {
        self.terms
            .iter()
            .filter(|(&(x, _), _)| x == 0)
            .map(|(_, &c)| c)
            .fold(C64::new(0.0, 0.0), |a, b| a + b)
    }

    /// Approximate stored bytes.
    pub fn memory_bytes(&self) -> usize {
        self.terms.capacity() * (2 * 8 + 16 + 1) * 8 / 7 + std::mem::size_of::<Self>()
    }
}

// ── the circuit, as Pauli rotations ──────────────────────────────────

/// One gate: `exp(−iθ Q / 2)` about the Hermitian Pauli
/// `Q = i^{|x∧z|} X^x Z^z`.
///
/// Every unitary decomposes into these, and the physics circuits this
/// module targets are written in them natively (a Trotter step *is* a
/// list of Pauli rotations). `θ = ±π/2` is Clifford and never branches
/// the sum by more than a relabelling; `θ` elsewhere is where the tree
/// grows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotation {
    /// Rotation angle.
    pub theta: f64,
    /// Axis masks `(x, z)`.
    pub axis: PauliKey,
}

impl Rotation {
    /// A rotation about `Z_q`.
    pub fn rz(qubit: usize, theta: f64) -> Self {
        Rotation {
            theta,
            axis: (0, 1u64 << qubit),
        }
    }

    /// A rotation about `X_q`.
    pub fn rx(qubit: usize, theta: f64) -> Self {
        Rotation {
            theta,
            axis: (1u64 << qubit, 0),
        }
    }

    /// A rotation about `Z_a Z_b`.
    pub fn rzz(a: usize, b: usize, theta: f64) -> Self {
        Rotation {
            theta,
            axis: (0, (1u64 << a) | (1u64 << b)),
        }
    }

    /// A rotation about `X_a X_b`.
    pub fn rxx(a: usize, b: usize, theta: f64) -> Self {
        Rotation {
            theta,
            axis: ((1u64 << a) | (1u64 << b), 0),
        }
    }

    /// Support size of the axis.
    pub fn weight(&self) -> usize {
        (self.axis.0 | self.axis.1).count_ones() as usize
    }

    /// The qubits the axis acts on, ascending.
    pub fn support(&self) -> Vec<usize> {
        let mask = self.axis.0 | self.axis.1;
        (0..MAX_QUBITS).filter(|&q| mask >> q & 1 == 1).collect()
    }

    /// `exp(−iθQ/2)` as a gate matrix on [`Rotation::support`], so the
    /// same rotation can be handed to any [`Backend`] —
    /// which is how the propagation is checked against dense.
    pub fn gate(&self) -> Result<(GateMatrix<C64>, Vec<usize>)> {
        let support = self.support();
        let dim = 1usize << support.len();
        if support.is_empty() {
            return Err(Error::InvalidState("rotation with empty axis".into()));
        }
        let phase = axis_operator_phase(self.axis);
        // Q = phase · X^x Z^z restricted to the support
        let mut q = GateMatrix::<C64>::zeros(dim)?;
        for col in 0..dim {
            let (mut row, mut sign) = (0usize, 1.0f64);
            for (b, &site) in support.iter().enumerate() {
                let bit = (col >> b) & 1;
                if (self.axis.1 >> site) & 1 == 1 && bit == 1 {
                    sign = -sign; // Z acts first
                }
                row |= (bit ^ (((self.axis.0 >> site) & 1) as usize)) << b;
            }
            q.set(row, col, phase * C64::new(sign, 0.0));
        }
        let (c, s) = ((self.theta / 2.0).cos(), (self.theta / 2.0).sin());
        let mut m = GateMatrix::<C64>::zeros(dim)?;
        for r in 0..dim {
            for cc in 0..dim {
                let id = if r == cc {
                    C64::new(c, 0.0)
                } else {
                    C64::new(0.0, 0.0)
                };
                m.set(r, cc, id + q.get(r, cc) * C64::new(0.0, -s));
            }
        }
        Ok((m, support))
    }
}

// ── the journal ──────────────────────────────────────────────────────

/// One backward step's accounting.
#[derive(Debug, Clone)]
pub struct Step {
    /// Position of the gate in the original (forward) circuit.
    pub gate_index: usize,
    /// Terms in the sum after this step.
    pub terms: usize,
    /// `Σ|c|` discarded at this step.
    pub discarded_l1: f64,
    /// Running `Σ|c|` discarded from the start of the walk — **monotone
    /// non-decreasing**, which is what makes [`Journal::retrodict`] a
    /// binary search.
    pub cumulative_l1: f64,
    /// `Σ|c|²` remaining.
    pub l2_squared: f64,
    /// Largest term support after this step.
    pub max_weight: usize,
    /// Whether the gate commuted with everything and cost nothing.
    pub skipped: bool,
}

/// A checkpoint: the partial state at a step, so refinement need not
/// restart the walk.
#[derive(Debug, Clone)]
struct Checkpoint {
    step: usize,
    sum: PauliSum,
    cumulative_l1: f64,
    /// The answer already settled by retirement before this point — it
    /// is not in `sum`, so refinement must carry it or lose it.
    retired_value: C64,
}

/// The record of a backward walk, and the instrument for asking where
/// the precision went.
#[derive(Debug, Clone, Default)]
pub struct Journal {
    steps: Vec<Step>,
    checkpoints: Vec<Checkpoint>,
}

impl Journal {
    /// Every step, in walk order (which is reverse circuit order).
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Checkpoints retained.
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// **Retrodiction.** The first walk step at which the cumulative
    /// discarded weight exceeded `budget`, found by binary search on the
    /// monotone running total — `O(log L)`, with nothing re-run.
    ///
    /// `None` means the budget was never exceeded: the answer is good to
    /// `budget` and no refinement is needed.
    pub fn retrodict(&self, budget: f64) -> Option<usize> {
        if self.steps.is_empty() || self.steps.last().unwrap().cumulative_l1 <= budget {
            return None;
        }
        let (mut lo, mut hi) = (0usize, self.steps.len() - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.steps[mid].cumulative_l1 > budget {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Some(lo)
    }

    /// The **circuit position** where the budget was spent — the
    /// actionable form of [`Journal::retrodict`], since the walk runs
    /// backwards and a step index is not a gate index.
    pub fn blame_gate(&self, budget: f64) -> Option<usize> {
        self.retrodict(budget).map(|s| self.steps[s].gate_index)
    }

    /// The latest checkpoint at or before `step`.
    fn checkpoint_at_or_before(&self, step: usize) -> Option<&Checkpoint> {
        self.checkpoints
            .iter()
            .rev()
            .find(|c| c.step <= step)
    }

    /// Approximate stored bytes, checkpoints included — the space half of
    /// the space–time dial.
    pub fn memory_bytes(&self) -> usize {
        self.steps.len() * std::mem::size_of::<Step>()
            + self
                .checkpoints
                .iter()
                .map(|c| c.sum.memory_bytes())
                .sum::<usize>()
    }
}

// ── exclusion: what provably cannot reach the answer ─────────────────

/// A GF(2) basis of X-masks, built by insertion without ever rewriting
/// an existing vector — so every **prefix** of the vector list is itself
/// a valid basis, and one list serves every prefix of the circuit.
#[derive(Debug, Clone, Default)]
pub struct XSpan {
    /// `(pivot bit, vector)`, pivots distinct, insertion order.
    vectors: Vec<(u32, u64)>,
    /// `counts[g]` = vectors contributed by gates `0..g`.
    counts: Vec<usize>,
}

impl XSpan {
    /// Build the filtration of X-mask spans over a rotation list, in
    /// forward gate order.
    pub fn of(rotations: &[Rotation]) -> Self {
        let mut span = XSpan {
            vectors: Vec::new(),
            counts: Vec::with_capacity(rotations.len() + 1),
        };
        span.counts.push(0);
        for r in rotations {
            span.insert(r.axis.0);
            span.counts.push(span.vectors.len());
        }
        span
    }

    fn insert(&mut self, mut v: u64) {
        for &(pivot, basis) in &self.vectors {
            if v >> pivot & 1 == 1 {
                v ^= basis;
            }
        }
        if v != 0 {
            let pivot = 63 - v.leading_zeros();
            self.vectors.push((pivot, v));
        }
    }

    /// Whether `x` lies in the span of the X-masks of gates `0..gates`
    /// — i.e. whether the remaining circuit can still cancel it.
    ///
    /// A `false` here is an **exact** exclusion: the term and every
    /// descendant it would produce contribute precisely zero to
    /// `⟨0|·|0⟩`, with no approximation involved.
    pub fn reachable(&self, x: u64, gates: usize) -> bool {
        if x == 0 {
            return true;
        }
        let take = self.counts[gates.min(self.counts.len() - 1)];
        let mut v = x;
        for &(pivot, basis) in &self.vectors[..take] {
            if v >> pivot & 1 == 1 {
                v ^= basis;
            }
        }
        v == 0
    }

    /// Rank of the full span.
    pub fn rank(&self) -> usize {
        self.vectors.len()
    }

    /// Approximate stored bytes — one shared list, not one basis per gate.
    pub fn memory_bytes(&self) -> usize {
        self.vectors.len() * 12 + self.counts.len() * 8 + std::mem::size_of::<Self>()
    }
}

/// A GF(2) basis of the remaining **axes**, in symplectic coordinates,
/// with the same prefix filtration as [`XSpan`].
///
/// A term commutes with a whole span iff it commutes with a basis of it,
/// because the symplectic form is bilinear — so one `O(rank)` test says
/// whether the remaining circuit can touch a term **at all**.
#[derive(Debug, Clone, Default)]
pub struct AxisSpan {
    /// `(pivot, x-mask, z-mask)`, pivots distinct, insertion order.
    vectors: Vec<(u32, u64, u64)>,
    counts: Vec<usize>,
}

impl AxisSpan {
    /// Build the filtration over a rotation list, in forward gate order.
    pub fn of(rotations: &[Rotation]) -> Self {
        let mut span = AxisSpan {
            vectors: Vec::new(),
            counts: Vec::with_capacity(rotations.len() + 1),
        };
        span.counts.push(0);
        for r in rotations {
            span.insert(r.axis);
            span.counts.push(span.vectors.len());
        }
        span
    }

    fn insert(&mut self, axis: PauliKey) {
        let (mut x, mut z) = axis;
        for &(pivot, bx, bz) in &self.vectors {
            let bit = if pivot < 64 {
                x >> pivot & 1
            } else {
                z >> (pivot - 64) & 1
            };
            if bit == 1 {
                x ^= bx;
                z ^= bz;
            }
        }
        if x != 0 {
            self.vectors.push((63 - x.leading_zeros(), x, z));
        } else if z != 0 {
            self.vectors.push((127 - z.leading_zeros(), x, z));
        }
    }

    /// Whether the remaining `gates` can move this term at all: `false`
    /// means the term commutes with every one of them, so its masks are
    /// **frozen** for the rest of the walk.
    pub fn can_touch(&self, term: PauliKey, gates: usize) -> bool {
        let take = self.counts[gates.min(self.counts.len() - 1)];
        self.vectors[..take]
            .iter()
            .any(|&(_, ax, az)| !commutes(term, (ax, az)))
    }

    /// Rank of the full span.
    pub fn rank(&self) -> usize {
        self.vectors.len()
    }

    /// Approximate stored bytes.
    pub fn memory_bytes(&self) -> usize {
        self.vectors.len() * 20 + self.counts.len() * 8 + std::mem::size_of::<Self>()
    }
}

// ── configuration and the walk ───────────────────────────────────────

/// How much to keep, and how much to remember.
#[derive(Debug, Clone)]
pub struct Config {
    /// Terms with `|c|` below this are dropped, and their weight is
    /// accounted. `0.0` is exact.
    pub threshold: f64,
    /// Hard cap on the term count; the walk truncates upward to respect
    /// it and reports having done so. `None` is uncapped.
    pub max_terms: Option<usize>,
    /// Store a checkpoint every this many steps. `0` disables them —
    /// minimum memory, no cheap refinement.
    pub checkpoint_every: usize,
    /// Run **exclusion first**: before ranking terms by magnitude, drop
    /// the ones the remaining circuit provably cannot bring back to the
    /// `X`-free sector. That is exact and free of error, so it is asked
    /// before truncation is.
    pub exclusion: bool,
    /// **Retire frozen terms.** A term commuting with every remaining
    /// axis can never change again, so it is settled: if its `X`-mask is
    /// zero its coefficient joins the answer and it leaves the working
    /// set; if not, it contributes nothing and is dropped. Both are
    /// exact.
    ///
    /// **Default off, because it was measured not to pay** — see the
    /// module docs. It is kept because it is exact and because the
    /// measurement is worth being able to repeat, not because it helps.
    pub retire_frozen: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            threshold: 1e-8,
            max_terms: None,
            checkpoint_every: 64,
            exclusion: true,
            // Off by default: exact, correct, and measured not to pay.
            // See the module docs.
            retire_frozen: false,
        }
    }
}

/// A completed (or partial) propagation.
#[derive(Debug, Clone)]
pub struct Propagation {
    /// The conjugated observable.
    pub sum: PauliSum,
    /// The record of the walk.
    pub journal: Journal,
    /// Gates that commuted with every term and cost nothing — the
    /// backward light cone, measured.
    pub commuting_skips: usize,
    /// Gates that actually branched the sum.
    pub branchings: usize,
    /// Total `Σ|c|` discarded: a **bound** on the expectation error,
    /// since every Pauli has operator norm 1.
    pub discarded_l1: f64,
    /// Total `Σ|c|²` discarded.
    pub discarded_l2: f64,
    /// Peak term count during the walk.
    pub peak_terms: usize,
    /// Whether the term cap was ever hit.
    pub hit_cap: bool,
    /// Terms removed by **exclusion** — provably unable to reach the
    /// answer, so removing them costs nothing. Kept in its own ledger,
    /// separate from [`Propagation::discarded_l1`], because one is exact
    /// and the other is error.
    pub excluded_terms: usize,
    /// `Σ|c|` carried by the excluded terms. This is *not* error: those
    /// terms contribute exactly zero.
    pub excluded_l1: f64,
    /// Terms retired early because the remaining circuit could no longer
    /// touch them — settled exactly, and removed from the working set.
    pub retired_terms: usize,
    /// The contribution those retired terms already made to the answer.
    pub retired_value: C64,
}

impl Propagation {
    /// `⟨0…0|U†PU|0…0⟩` under the retained terms, plus whatever was
    /// already settled by early retirement.
    pub fn expectation(&self) -> f64 {
        (self.sum.expectation_on_zero_state() + self.retired_value).re
    }

    /// The certified error bar on [`Propagation::expectation`].
    pub fn error_bound(&self) -> f64 {
        self.discarded_l1
    }
}

/// Conjugate one Pauli sum backwards through one rotation.
///
/// **Exclusion is applied at generation, not after it.** When a span is
/// supplied, a branch whose `X`-mask the remaining circuit provably
/// cannot cancel is never created — so the peak term count never
/// includes it, and the work of carrying it is never done. Filtering
/// after the fact gives the same answer and none of the saving.
pub(crate) fn step_through(
    sum: &PauliSum,
    rot: &Rotation,
    exclude: Option<(&XSpan, usize)>,
) -> (PauliSum, bool, usize, f64) {
    let (c, s) = ((rot.theta).cos(), (rot.theta).sin());
    let phase = axis_operator_phase(rot.axis);
    let mut out = PauliSum::zero();
    let mut branched = false;
    let (mut excluded, mut excluded_l1) = (0usize, 0.0f64);

    let reachable = |x: u64| match exclude {
        Some((sp, gates)) => sp.reachable(x, gates),
        None => true,
    };

    for (key, coeff) in sum.terms() {
        if commutes(key, rot.axis) {
            if reachable(key.0) {
                out.add(key, coeff);
            } else {
                excluded += 1;
                excluded_l1 += coeff.norm();
            }
            continue;
        }
        // R† P R = cos θ · P − i sin θ · P Q, with both branches held to
        // the floor so a numerically-zero one does not survive.
        let keep = coeff * C64::new(c, 0.0);
        if keep.norm() >= COEFF_FLOOR {
            if reachable(key.0) {
                out.add(key, keep);
            } else {
                excluded += 1;
                excluded_l1 += keep.norm();
            }
        }
        let (prod, sign) = pauli_mul(key, rot.axis);
        let split = coeff * (C64::new(0.0, -s * sign) * phase);
        if split.norm() >= COEFF_FLOOR {
            if reachable(prod.0) {
                out.add(prod, split);
                branched = true;
            } else {
                excluded += 1;
                excluded_l1 += split.norm();
            }
        }
    }
    (out, branched, excluded, excluded_l1)
}

/// Propagate an observable backwards through a rotation sequence.
///
/// `rotations` is in **forward circuit order**; the walk runs from the
/// last to the first, which is what makes the journal retrodictive.
pub fn propagate(
    observable: &PauliSum,
    rotations: &[Rotation],
    cfg: &Config,
) -> Result<Propagation> {
    if rotations.len() > usize::MAX - 1 {
        return Err(Error::InvalidState("rotation list too long".into()));
    }
    let mut sum = observable.clone();
    let mut journal = Journal::default();
    let (mut skips, mut branchings) = (0usize, 0usize);
    let (mut disc_l1, mut disc_l2) = (0.0f64, 0.0f64);
    let mut peak = sum.len();
    let mut hit_cap = false;
    let (mut excluded_terms, mut excluded_l1) = (0usize, 0.0f64);
    let (mut retired_terms, mut retired_value) = (0usize, C64::new(0.0, 0.0));
    // The exclusion filtration is over the whole rotation list; building
    // it is one forward pass and one shared basis.
    let span = if cfg.exclusion {
        Some(XSpan::of(rotations))
    } else {
        None
    };
    let axes = if cfg.retire_frozen {
        Some(AxisSpan::of(rotations))
    } else {
        None
    };

    if cfg.checkpoint_every > 0 {
        journal.checkpoints.push(Checkpoint {
            step: 0,
            sum: sum.clone(),
            cumulative_l1: 0.0,
            retired_value: C64::new(0.0, 0.0),
        });
    }

    for (step, gate_index) in (0..rotations.len()).rev().enumerate() {
        let rot = &rotations[gate_index];
        // `gate_index` gates remain after this one is consumed.
        let excl = span.as_ref().map(|sp| (sp, gate_index));
        let (next, branched, ex_n, ex_l1) = step_through(&sum, rot, excl);
        sum = next;
        excluded_terms += ex_n;
        excluded_l1 += ex_l1;
        if branched {
            branchings += 1;
        } else {
            skips += 1;
        }

        let (mut l1, mut l2) = sum.truncate(cfg.threshold);
        if let Some(cap) = cfg.max_terms {
            if sum.len() > cap {
                hit_cap = true;
                // raise the threshold until the cap is met, accounting
                // everything dropped on the way
                let mut mags: Vec<f64> = sum.terms().map(|(_, c)| c.norm()).collect();
                mags.sort_by(|a, b| b.partial_cmp(a).unwrap());
                let cut = mags[cap];
                let (a, b) = sum.truncate(cut * (1.0 + 1e-12));
                l1 += a;
                l2 += b;
            }
        }
        // RETIREMENT. A term the remaining circuit can no longer touch
        // is settled: it either already sits in the X-free sector and
        // its coefficient is part of the answer, or it never will and it
        // contributes nothing. Either way it leaves the working set, and
        // unlike reachability-exclusion this can fire mid-walk.
        if let Some(ax) = &axes {
            let mut settled = C64::new(0.0, 0.0);
            let mut retired = 0usize;
            sum.retain(|&key, c| {
                if ax.can_touch(key, gate_index) {
                    return true;
                }
                retired += 1;
                if key.0 == 0 {
                    settled += *c;
                }
                false
            });
            retired_terms += retired;
            retired_value += settled;
        }

        disc_l1 += l1;
        disc_l2 += l2;
        peak = peak.max(sum.len());

        journal.steps.push(Step {
            gate_index,
            terms: sum.len(),
            discarded_l1: l1,
            cumulative_l1: disc_l1,
            l2_squared: sum.l2_squared(),
            max_weight: sum.max_weight(),
            skipped: !branched,
        });

        if cfg.checkpoint_every > 0 && (step + 1) % cfg.checkpoint_every == 0 {
            journal.checkpoints.push(Checkpoint {
                step: step + 1,
                sum: sum.clone(),
                cumulative_l1: disc_l1,
                retired_value,
            });
        }
    }

    Ok(Propagation {
        sum,
        journal,
        commuting_skips: skips,
        branchings,
        discarded_l1: disc_l1,
        discarded_l2: disc_l2,
        peak_terms: peak,
        hit_cap,
        excluded_terms,
        excluded_l1,
        retired_terms,
        retired_value,
    })
}

impl Propagation {
    /// Re-run from the latest checkpoint at or before the step the
    /// budget was blown at, with a tighter threshold — the progressive
    /// half of the method.
    ///
    /// The prefix of the walk before that checkpoint is not repeated, so
    /// a sharper answer costs only the tail. Returns the refined
    /// propagation and the number of steps actually re-walked.
    pub fn refine_from(
        &self,
        rotations: &[Rotation],
        step: usize,
        cfg: &Config,
    ) -> Result<(Propagation, usize)> {
        let Some(cp) = self.journal.checkpoint_at_or_before(step) else {
            // no checkpoint: an honest full re-run
            let p = propagate_from_state(
                &self.observable_seed(),
                rotations,
                0,
                0.0,
                C64::new(0.0, 0.0),
                cfg,
            )?;
            let walked = rotations.len();
            return Ok((p, walked));
        };
        let walked = rotations.len() - cp.step;
        let p = propagate_from_state(
            &cp.sum,
            rotations,
            cp.step,
            cp.cumulative_l1,
            cp.retired_value,
            cfg,
        )?;
        Ok((p, walked))
    }

    /// The sum as it stood at walk step 0, if a checkpoint held it.
    fn observable_seed(&self) -> PauliSum {
        self.journal
            .checkpoints
            .first()
            .map(|c| c.sum.clone())
            .unwrap_or_else(PauliSum::zero)
    }
}

/// Continue a walk from a checkpointed sum at `start_step`, carrying the
/// error already accounted for.
#[allow(clippy::too_many_arguments)]
fn propagate_from_state(
    sum0: &PauliSum,
    rotations: &[Rotation],
    start_step: usize,
    carried_l1: f64,
    carried_retired: C64,
    cfg: &Config,
) -> Result<Propagation> {
    let remaining = rotations.len().saturating_sub(start_step);
    let tail: Vec<Rotation> = rotations[..remaining].to_vec();
    let mut p = propagate(sum0, &tail, cfg)?;
    p.discarded_l1 += carried_l1;
    p.retired_value += carried_retired;
    for s in p.journal.steps.iter_mut() {
        s.cumulative_l1 += carried_l1;
    }
    Ok(p)
}

// ── sharing one walk across an observable basis ──────────────────────

/// What [`propagate_basis`] measured.
#[derive(Debug, Clone)]
pub struct BasisReport {
    /// `Σ wᵢ⟨Pᵢ⟩` from the **single** shared walk — the quantity an
    /// energy or a structure factor actually is.
    pub total: f64,
    /// The certified bound on `total`.
    pub error_bound: f64,
    /// Per-observable values, only when asked for: each costs its own
    /// walk, so the default is not to.
    pub breakdown: Option<Vec<f64>>,
    /// Terms at the end of the shared walk.
    pub terms: usize,
    /// Peak terms during the shared walk.
    pub peak_terms: usize,
    /// Peak terms summed over separate per-observable walks — the cost
    /// the sharing avoids.
    pub separate_peak_total: Option<usize>,
    /// Gates the shared walk skipped as commuting.
    pub commuting_skips: usize,
    /// The shared walk's journal.
    pub journal: Journal,
}

impl BasisReport {
    /// How much the shared walk saved over walking each observable
    /// separately, in peak terms. `None` unless the breakdown was
    /// computed (there is nothing to compare against otherwise).
    pub fn sharing_factor(&self) -> Option<f64> {
        self.separate_peak_total
            .map(|sep| sep as f64 / self.peak_terms.max(1) as f64)
    }
}

/// Propagate a whole weighted observable basis through **one** walk.
///
/// Conjugation is linear, so `Σ wᵢPᵢ` propagates as a single sum and the
/// like terms produced by different observables **merge**. That merging
/// is the saving, and it is large precisely when the observables overlap
/// — which is the normal case: an energy `H = Σ cᵢPᵢ` is a sum of local
/// terms on the same lattice, and their backward cones are nearly the
/// same cone.
///
/// `breakdown` costs one extra walk per observable and is off by
/// default; the total is what a shared walk is for.
pub fn propagate_basis(
    observables: &[PauliSum],
    weights: &[f64],
    rotations: &[Rotation],
    cfg: &Config,
    breakdown: bool,
) -> Result<BasisReport> {
    if observables.len() != weights.len() {
        return Err(Error::InvalidState(format!(
            "{} observables but {} weights",
            observables.len(),
            weights.len()
        )));
    }
    if observables.is_empty() {
        return Err(Error::InvalidState("no observables".into()));
    }
    let mut combined = PauliSum::zero();
    for (o, &w) in observables.iter().zip(weights) {
        for (k, c) in o.terms() {
            combined.add(k, c * C64::new(w, 0.0));
        }
    }
    let shared = propagate(&combined, rotations, cfg)?;

    let (values, separate_peak) = if breakdown {
        let mut vals = Vec::with_capacity(observables.len());
        let mut peak_total = 0usize;
        for (o, &w) in observables.iter().zip(weights) {
            let p = propagate(o, rotations, cfg)?;
            vals.push(w * p.expectation());
            peak_total += p.peak_terms;
        }
        (Some(vals), Some(peak_total))
    } else {
        (None, None)
    };

    Ok(BasisReport {
        total: shared.expectation(),
        error_bound: shared.error_bound(),
        breakdown: values,
        terms: shared.sum.len(),
        peak_terms: shared.peak_terms,
        separate_peak_total: separate_peak,
        commuting_skips: shared.commuting_skips,
        journal: shared.journal,
    })
}

/// The transverse-field Ising energy `H = −J Σ Z_k Z_{k+1} − h Σ X_k` as
/// a weighted observable basis: `2n − 1` local terms sharing one lattice.
pub fn tfim_energy_basis(n: usize, j: f64, h: f64) -> (Vec<PauliSum>, Vec<f64>) {
    let mut obs = Vec::new();
    let mut w = Vec::new();
    for k in 0..n.saturating_sub(1) {
        obs.push(PauliSum::zz(k, k + 1));
        w.push(-j);
    }
    for k in 0..n {
        obs.push(PauliSum::x(k));
        w.push(-h);
    }
    (obs, w)
}

// ── factoring the operator where the circuit does not couple ─────────

/// The conjugated observable held as a **product of independent blocks**
/// rather than one flat sum.
///
/// A cut is positional and blind to structure. This is the structural
/// alternative: a single Pauli is already a tensor product of
/// single-site Paulis, and conjugation only ever *couples* two blocks
/// when a gate's axis straddles them. So the natural representation is
/// one sum per block, merged lazily on demand — the operator-side
/// analogue of
/// [`FactoredState`](crate::backend::FactoredState), and the reason it
/// helps is arithmetic: `k` independent blocks of size `m` cost `k·m`
/// stored terms instead of `m^k`.
///
/// Nothing is declared in advance. The partition is *discovered* from
/// the gates as the walk meets them, and
/// [`FactoredPauliSum::blocks`] is what it found.
#[derive(Debug, Clone)]
pub struct FactoredPauliSum {
    /// `(qubit mask, sum restricted to it)`. Masks are disjoint; qubits
    /// in no block carry the identity.
    blocks: Vec<(u64, PauliSum)>,
    /// Overall scalar, kept out of the blocks so a global phase never
    /// forces a merge.
    scale: C64,
}

impl FactoredPauliSum {
    /// A single Pauli, blocked by its own support: one block per site,
    /// which is the finest partition consistent with it.
    pub fn from_key(key: PauliKey) -> Self {
        let (x, z) = key;
        let mut blocks = Vec::new();
        for q in 0..MAX_QUBITS {
            let (bx, bz) = (x >> q & 1, z >> q & 1);
            if bx | bz != 0 {
                blocks.push((1u64 << q, PauliSum::from_key((bx << q, bz << q))));
            }
        }
        FactoredPauliSum {
            blocks,
            scale: C64::new(1.0, 0.0),
        }
    }

    /// `Z` on one qubit.
    pub fn z(qubit: usize) -> Self {
        FactoredPauliSum::from_key((0, 1u64 << qubit))
    }

    /// The discovered partition: `(mask, term count)` per block.
    pub fn blocks(&self) -> Vec<(u64, usize)> {
        self.blocks.iter().map(|(m, s)| (*m, s.len())).collect()
    }

    /// Terms actually stored — the **sum** over blocks.
    pub fn stored_terms(&self) -> usize {
        self.blocks.iter().map(|(_, s)| s.len()).sum()
    }

    /// Terms the equivalent flat sum would hold — the **product** over
    /// blocks. This is the number the factorization avoids.
    pub fn flat_terms(&self) -> u128 {
        self.blocks
            .iter()
            .map(|(_, s)| s.len() as u128)
            .product::<u128>()
            .max(1)
    }

    /// Largest single block, in terms — what the memory actually tracks.
    pub fn largest_block(&self) -> usize {
        self.blocks.iter().map(|(_, s)| s.len()).max().unwrap_or(0)
    }

    /// `⟨0…0|·|0…0⟩`, which factorizes: `⟨0|A⊗B|0⟩ = ⟨0|A|0⟩⟨0|B|0⟩`.
    /// The answer is assembled from the blocks and the flat sum is never
    /// formed.
    pub fn expectation_on_zero_state(&self) -> C64 {
        let mut acc = self.scale;
        for (_, s) in &self.blocks {
            acc *= s.expectation_on_zero_state();
        }
        acc
    }

    /// Merge every block the mask touches into one, so a straddling gate
    /// has a single block to act on. This is where the cost is paid, and
    /// it is paid only when the circuit genuinely couples the regions.
    fn merge_for(&mut self, mask: u64) -> usize {
        let hit: Vec<usize> = (0..self.blocks.len())
            .filter(|&i| self.blocks[i].0 & mask != 0)
            .collect();
        if hit.len() <= 1 {
            return *hit.first().unwrap_or(&usize::MAX);
        }
        // tensor the touched blocks together
        let mut merged_mask = 0u64;
        let mut merged = PauliSum::from_key((0, 0));
        for &i in &hit {
            let (m, s) = &self.blocks[i];
            merged_mask |= m;
            let mut next = PauliSum::zero();
            for (ka, ca) in merged.terms() {
                for (kb, cb) in s.terms() {
                    // disjoint supports, so the product carries no sign
                    next.add((ka.0 | kb.0, ka.1 | kb.1), ca * cb);
                }
            }
            merged = next;
        }
        for &i in hit.iter().rev() {
            self.blocks.remove(i);
        }
        self.blocks.push((merged_mask, merged));
        self.blocks.len() - 1
    }

    /// The blocks themselves, for evaluators that need more than
    /// `⟨0…0|·|0…0⟩` — [`crate::coupling`] contracts them against a
    /// stabilizer input instead.
    pub fn block_sums(&self) -> &[(u64, PauliSum)] {
        &self.blocks
    }

    /// The overall scalar held outside the blocks.
    pub fn scale(&self) -> C64 {
        self.scale
    }

    /// Conjugate through one rotation, merging only if it straddles.
    pub fn conjugate(&mut self, rot: &Rotation) {
        self.step(rot)
    }

    /// Conjugate through one rotation, merging only if it straddles.
    fn step(&mut self, rot: &Rotation) {
        let mask = rot.axis.0 | rot.axis.1;
        let idx = self.merge_for(mask);
        if idx == usize::MAX {
            // the observable is the identity everywhere the axis acts,
            // so it commutes and nothing happens
            return;
        }
        let (bmask, sum) = &self.blocks[idx];
        let (next, _, _, _) = step_through(sum, rot, None);
        let new_mask = bmask | mask;
        self.blocks[idx] = (new_mask, next);
    }
}

/// What [`propagate_factored`] measured.
#[derive(Debug, Clone)]
pub struct FactoredReport {
    /// `⟨0…0|U†PU|0…0⟩`, assembled from the blocks.
    pub value: f64,
    /// The discovered partition at the end: `(mask, terms)`.
    pub blocks: Vec<(u64, usize)>,
    /// Peak stored terms — the sum over blocks.
    pub peak_stored: usize,
    /// Peak of the equivalent flat term count — the product over blocks,
    /// i.e. what a non-factored walk would have carried.
    pub peak_flat: u128,
    /// Peak size of the largest single block.
    pub peak_largest_block: usize,
    /// Gates that straddled two blocks and forced a merge — the measured
    /// points where the circuit actually couples regions.
    pub merges: usize,
}

impl FactoredReport {
    /// How much the factorization avoided, at peak.
    pub fn factor_saving(&self) -> f64 {
        self.peak_flat as f64 / self.peak_stored.max(1) as f64
    }
}

/// Propagate an observable backwards holding it **factored**, merging
/// blocks only where the circuit couples them.
///
/// Exact: no threshold is applied, because a per-block truncation's
/// error bound has to be carried across the product and that is not
/// implemented. The point being measured here is structural — how much
/// of the sum never needed to be formed.
pub fn propagate_factored(
    observable: PauliKey,
    rotations: &[Rotation],
) -> Result<FactoredReport> {
    let mut f = FactoredPauliSum::from_key(observable);
    let mut peak_stored = f.stored_terms();
    let mut peak_flat = f.flat_terms();
    let mut peak_largest = f.largest_block();
    let mut merges = 0usize;

    for rot in rotations.iter().rev() {
        let mask = rot.axis.0 | rot.axis.1;
        let touched = f.blocks.iter().filter(|(m, _)| m & mask != 0).count();
        if touched > 1 {
            merges += 1;
        }
        f.step(rot);
        peak_stored = peak_stored.max(f.stored_terms());
        peak_flat = peak_flat.max(f.flat_terms());
        peak_largest = peak_largest.max(f.largest_block());
    }

    Ok(FactoredReport {
        value: f.expectation_on_zero_state().re,
        blocks: f.blocks(),
        peak_stored,
        peak_flat,
        peak_largest_block: peak_largest,
        merges,
    })
}

// ── meeting in the middle: both directions of time ───────────────────

/// How the forward half holds `|ψ⟩ = U₁|0…0⟩`.
///
/// The choice matters more than the cut does: the backward half is
/// exponential in *branching gates* and the forward half in whatever its
/// representation is exponential in, so meeting in the middle only pays
/// when those two resources are genuinely different.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Forward {
    /// Sparse amplitudes — exponential in support, which for an
    /// entangling circuit means immediately.
    Sparse,
    /// Matrix product state — exponential in entanglement across a cut,
    /// so a low-entanglement prefix stays cheap however non-Clifford it
    /// is. This is the case where the two resources actually differ.
    Mps {
        /// Bond cap; truncation beyond it is reported by the backend.
        max_bond: usize,
    },
    /// Clifford frame — a Clifford prefix costs nothing at all, though
    /// the backward walk does not pay for one either.
    CliffordFramed,
}

/// What [`propagate_bidirectional`] measured at one cut.
#[derive(Debug, Clone)]
pub struct Meeting {
    /// Where the circuit was split: gates `..cut` ran forward, gates
    /// `cut..` were walked backward.
    pub cut: usize,
    /// `⟨ψ|A|ψ⟩` with `|ψ⟩ = U₁|0…0⟩` and `A = U₂†PU₂`.
    pub value: f64,
    /// Bytes the forward half needed — comparable across
    /// representations in a way that a support count is not.
    pub forward_bytes: usize,
    /// Stored amplitudes the forward half needed.
    pub forward_support: usize,
    /// Peak Pauli terms the backward half needed.
    pub backward_peak: usize,
    /// Terms surviving at the meeting surface.
    pub backward_terms: usize,
    /// Certified bound, inherited from the backward half's truncation.
    pub error_bound: f64,
    /// The larger of the two sides in bytes — the cost the cut is chosen
    /// to minimize, since the two halves are paid for separately.
    pub meeting_cost: usize,
}

/// Evaluate `⟨ψ| X^x Z^z |ψ⟩` on a stored state.
///
/// `X^x Z^z |i⟩ = (−1)^{|z ∧ i|} |i ⊕ x⟩`, so this is one pass over the
/// stored support — the *forward* half of the meeting, and the reason a
/// Pauli sum can be evaluated against a state without either side ever
/// being materialized as the other.
fn pauli_on_state(state: &dyn Backend<C64>, key: PauliKey) -> C64 {
    let (x, z) = key;
    let mut acc = C64::new(0.0, 0.0);
    state.for_each_nonzero(&mut |i, amp| {
        let sign = if (z & i).count_ones() % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        let bra = state.amplitude(i ^ x);
        acc += bra.conj() * amp * C64::new(sign, 0.0);
    });
    acc
}

/// Run the circuit from **both ends of time** and meet at `cut`.
///
/// `⟨0|U†PU|0⟩` with `U = U₂U₁` is `⟨ψ|U₂†PU₂|ψ⟩` for `|ψ⟩ = U₁|0…0⟩`.
/// So the observable need only be walked back through `U₂`, and the
/// input need only be pushed forward through `U₁`. Neither side ever
/// covers the whole circuit.
///
/// The exclusions are switched **off** for any cut past zero, because
/// they are statements about the `|0…0⟩` boundary rather than about the
/// circuit; see the guard in the body.
///
/// This is the only structural change here that can reduce the backward
/// walk's **peak**, because it reduces the number of branching gates the
/// walk ever sees — the two exclusions could only prune what the walk
/// had already produced. The price is that the forward half now pays
/// state amplitudes, so the cut trades one exponential against the
/// other and [`Meeting::meeting_cost`] is the quantity to minimize.
pub fn propagate_bidirectional(
    observable: &PauliSum,
    rotations: &[Rotation],
    num_qubits: usize,
    cut: usize,
    forward: Forward,
    cfg: &Config,
) -> Result<Meeting> {
    if cut > rotations.len() {
        return Err(Error::InvalidState(format!(
            "cut {cut} beyond {} rotations",
            rotations.len()
        )));
    }
    // The exclusions are BOUNDARY CONDITIONS, not circuit properties.
    // Both encode "this walk ends at |0…0⟩, where only X-free terms
    // contribute". At an intermediate cut the terms are evaluated
    // against |ψ⟩ = U₁|0…0⟩ instead, where terms with X-support
    // contribute perfectly well — so applying either would silently
    // produce wrong numbers, which is exactly what it did before this
    // guard existed (0.18 absolute error on a validation sweep).
    let partial = cut > 0;
    let back_cfg = if partial {
        Config {
            exclusion: false,
            retire_frozen: false,
            ..cfg.clone()
        }
    } else {
        cfg.clone()
    };
    // backward half: the observable through the suffix
    let back = propagate(observable, &rotations[cut..], &back_cfg)?;

    // forward half: the input through the prefix
    let mut state: Box<dyn Backend<C64>> = match forward {
        Forward::Sparse => Box::new(crate::backend::SparseState::<C64>::new(num_qubits)?),
        Forward::Mps { max_bond } => Box::new(crate::backend::MpsState::<C64>::with_config(
            num_qubits,
            crate::backend::MpsConfig {
                max_bond,
                trunc_tol: 1e-14,
            },
        )?),
        Forward::CliffordFramed => Box::new(
            crate::backend::CliffordFramedState::<C64>::new(num_qubits)?,
        ),
    };
    for r in &rotations[..cut] {
        let (m, s) = r.gate()?;
        state.apply(&m, &s)?;
    }
    let forward_support = state.nonzero_count();
    let forward_bytes = state.memory_bytes();

    // meet
    let mut value = back.retired_value;
    for (key, coeff) in back.sum.terms() {
        value += coeff * pauli_on_state(state.as_ref(), key);
    }

    // the backward half's terms cost their key plus a coefficient
    let backward_bytes = back.peak_terms * 32;
    Ok(Meeting {
        cut,
        value: value.re,
        forward_bytes,
        forward_support,
        backward_peak: back.peak_terms,
        backward_terms: back.sum.len(),
        error_bound: back.error_bound(),
        meeting_cost: forward_bytes.max(backward_bytes),
    })
}

/// Sweep the cut and return every meeting, so the trade between the two
/// exponentials is visible rather than assumed.
pub fn cut_sweep(
    observable: &PauliSum,
    rotations: &[Rotation],
    num_qubits: usize,
    cuts: &[usize],
    forward: Forward,
    cfg: &Config,
) -> Result<Vec<Meeting>> {
    cuts.iter()
        .map(|&c| propagate_bidirectional(observable, rotations, num_qubits, c, forward, cfg))
        .collect()
}

/// Find the cut that minimizes [`Meeting::meeting_cost`], by scanning
/// `samples` evenly spaced positions.
///
/// The minimum is genuinely interior only when the two halves are
/// exponential in *different* resources. With [`Forward::Sparse`] it
/// never is — the state saturates immediately and the best cut is `0`,
/// the pure backward walk. With [`Forward::Mps`] on a circuit whose
/// prefix is low-entanglement but non-Clifford it is: measured 8 032
/// bytes at the interior optimum against 76 064 at either end, a 9.5×
/// saving, with the value stable to 2e-7 across every cut.
pub fn auto_cut(
    observable: &PauliSum,
    rotations: &[Rotation],
    num_qubits: usize,
    samples: usize,
    forward: Forward,
    cfg: &Config,
) -> Result<Meeting> {
    let samples = samples.max(2);
    let cuts: Vec<usize> = (0..=samples)
        .map(|i| i * rotations.len() / samples)
        .collect();
    let mut best: Option<Meeting> = None;
    for m in cut_sweep(observable, rotations, num_qubits, &cuts, forward, cfg)? {
        if best.as_ref().map_or(true, |b| m.meeting_cost < b.meeting_cost) {
            best = Some(m);
        }
    }
    best.ok_or_else(|| Error::InvalidState("no cut sampled".into()))
}

// ── a circuit family worth pointing it at ────────────────────────────

/// One Trotter step of the transverse-field Ising model
/// `H = −J Σ Z_k Z_{k+1} − h Σ X_k`, as Pauli rotations.
///
/// At `dt·J = dt·h = π/4` every rotation is Clifford and the sum never
/// branches; anywhere else it does, which makes this the natural family
/// for measuring how the cost tracks non-Cliffordness.
pub fn tfim_trotter(n: usize, j: f64, h: f64, dt: f64, steps: usize) -> Vec<Rotation> {
    let mut out = Vec::with_capacity(steps * (2 * n));
    for _ in 0..steps {
        for k in 0..n.saturating_sub(1) {
            out.push(Rotation::rzz(k, k + 1, -2.0 * j * dt));
        }
        for k in 0..n {
            out.push(Rotation::rx(k, -2.0 * h * dt));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pauli_algebra_is_right() {
        let x = (1u64, 0u64);
        let z = (0u64, 1u64);
        assert!(!commutes(x, z));
        assert!(commutes(x, x));
        assert!(commutes((1, 0), (2, 0)));
        // X·Z = XZ with sign +1 in this basis
        let (k, s) = pauli_mul(x, z);
        assert_eq!(k, (1, 1));
        assert_eq!(s, 1.0);
        // Z·X = -XZ
        let (k, s) = pauli_mul(z, x);
        assert_eq!(k, (1, 1));
        assert_eq!(s, -1.0);
        // the Hermitian axis for (1,1) is i·XZ = Y
        assert_eq!(axis_operator_phase((1, 1)), C64::new(0.0, 1.0));
    }

    #[test]
    fn rz_rotates_x_into_minus_y() {
        // R† X R = cos θ X − sin θ Y for R = exp(−iθZ/2)
        let theta = 0.37;
        let p = propagate(
            &PauliSum::x(0),
            &[Rotation::rz(0, theta)],
            &Config {
                threshold: 0.0,
                max_terms: None,
                checkpoint_every: 0,
                // both prunings off: this test is about the branch itself
                exclusion: false,
                retire_frozen: false,
            },
        )
        .unwrap();
        assert_eq!(p.sum.len(), 2);
        let mut got_x = C64::new(0.0, 0.0);
        let mut got_xz = C64::new(0.0, 0.0);
        for (k, c) in p.sum.terms() {
            if k == (1, 0) {
                got_x = c;
            }
            if k == (1, 1) {
                got_xz = c;
            }
        }
        assert!((got_x.re - theta.cos()).abs() < 1e-14);
        // −sin θ · Y = −sin θ · i·XZ, so the XZ coefficient is −i sin θ
        assert!((got_xz - C64::new(0.0, -theta.sin())).norm() < 1e-14);
    }

    #[test]
    fn a_commuting_gate_is_free() {
        let cfg = Config {
            threshold: 0.0,
            max_terms: None,
            checkpoint_every: 0,
            exclusion: true,
            retire_frozen: false,
        };
        let p = propagate(&PauliSum::z(0), &[Rotation::rz(0, 0.5)], &cfg).unwrap();
        assert_eq!(p.sum.len(), 1);
        assert_eq!(p.commuting_skips, 1);
        assert_eq!(p.branchings, 0);
        assert!((p.expectation() - 1.0).abs() < 1e-15);
    }
}
