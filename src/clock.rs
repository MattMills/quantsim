//! A time system **inside** the register: a selector qudit whose
//! entanglement with the system is mandatory and structural, and whose
//! axis can be aligned with the coarse/fine-graining recursion of the
//! bulk register — internal, relational time, with scale as its
//! direction.
//!
//! The physics anchors are stated plainly: conditioning a static
//! entangled clock⊗system state on clock readings to recover dynamics
//! is the Page–Wootters mechanism, and a register holding all of its
//! own stages in superposition is a Feynman–Kitaev history state. What
//! this module adds is the *representation* claim, measured: making
//! the stage label an explicit qudit lets every stage live in its own
//! representation, so the joint object costs the **sum** of its
//! slices — while any single flat representation of the summed state
//! pays wherever *one* slice clashes with its structural bet. New
//! dimension above the problem, lower representational structure below
//! it.
//!
//! Two constructions:
//!
//! * [`BranchedRegister`] — a `d`-level **selector qudit** over a
//!   multiset of weighted branches, each branch an arbitrary
//!   [`Backend`] (heterogeneous: one branch sparse, one factored, one
//!   MPS…). Branch states are shared (`Rc`) so selector rotations are
//!   bookkeeping, not copies; interference between branches is
//!   evaluated lazily at amplitude time; branch count is the honest
//!   ledger of clock-side cost. System gates broadcast to each unique
//!   state once (additive). The register implements [`Backend`] in its
//!   **contracted** view — the plain sum `Σₖ wₖ|ψₖ⟩` with the selector
//!   traced against the weights — so every existing instrument
//!   (deviation, expectations, the atlas) applies to the object whose
//!   representation the qudit just lowered. The **flagged** view
//!   ([`BranchedRegister::flagged_amplitude`],
//!   [`BranchedRegister::condition`],
//!   [`BranchedRegister::measure_selector`]) is where the clock is a
//!   quantum degree of freedom: selector↔system entanglement is
//!   *measured* ([`BranchedRegister::selector_schmidt_rank`], computed
//!   polynomially from the pairwise Gram of unique branch states, never
//!   by enumeration), and conditioning reads out one slice.
//! * [`scale_history`] — the selector axis aligned with recursion:
//!   the history register `Σ_ℓ w_ℓ |ℓ⟩⊗|ψ at scale ℓ⟩` over a
//!   [`BulkState`]'s own depth levels, each slice built natively by
//!   [`BulkState::scale_snapshot`] (`O(tree)` node surgery, no
//!   exponential gate ever applied — width 32 histories in kilobytes).
//!   [`tick`] is the clock-controlled refinement `|ℓ⟩⟨ℓ| ⊗ V_ℓ` with
//!   `|ℓ⟩ → |ℓ+1⟩`: one unit of internal time **is** one level of
//!   coarse-to-fine information flow, which is the mandatory flow the
//!   selector carries. Interfering the clock
//!   ([`BranchedRegister::selector_mix`]) turns inter-scale overlaps
//!   into measurable clock statistics — scale interferometry, the
//!   speed of the renormalization flow read off a Born distribution.
//!
//! Honesty about boundaries: enumeration-based trait methods
//! (`for_each_nonzero`, `sample`) cost the union support — documented
//! trait behavior, exponential for dense-support branches, exactly as
//! on the other structured backends. [`tick`] refuses (by name) a
//! register whose branch states are shared across distinct scales —
//! interfere after ticking, or condition first. Selector mixes grow
//! the branch multiset at most to `d × unique_states`, consolidated by
//! `(selector, state)` identity and pruned at zero weight — growth is
//! ledgered ([`BranchedRegister::peak_branches`]), never silent. And a
//! contract the superposition itself imposes, measured in the tests:
//! branch **global phases become relative phases** of the joint state,
//! so a branch evolved through a representation that maintains states
//! only up to global phase (the graph-state bundle's documented
//! convention) is sound as a *static* slice but corrupts the
//! contracted sum under broadcast dynamics — branches evolved in place
//! must be phase-faithful.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::backend::{Backend, BulkState, MosaicState, SparseState, UnfoldProgram};
use crate::error::{Error, Result};
use crate::math::{svd_thin, GateMatrix};
use crate::rng::Prng;
use crate::scalar::Scalar;

/// Branch weights with squared magnitude at or below this are pruned
/// after selector operations.
const BRANCH_PRUNE_TOL: f64 = 1e-24;

type Shared<S> = Rc<RefCell<Box<dyn Backend<S>>>>;

struct Branch<S: Scalar> {
    sel: usize,
    weight: S,
    state: Shared<S>,
}

/// A selector qudit over weighted, representation-heterogeneous
/// branches. See the module docs.
pub struct BranchedRegister<S: Scalar> {
    n: usize,
    dim: usize,
    branches: Vec<Branch<S>>,
    peak_branches: usize,
}

impl<S: Scalar> BranchedRegister<S> {
    /// `|sel = 0⟩ ⊗ |0…0⟩` with a **mosaic** system branch: the two
    /// structures compose — the mosaic elects the ideal representation
    /// per portion of the register *below*, the selector adds qudit
    /// dimension *above* for what no single branch state holds cheaply.
    /// Callers wanting specific branch representations use
    /// [`from_branches`](Self::from_branches).
    pub fn new(num_qubits: usize, selector_dim: usize) -> Result<Self> {
        let state: Box<dyn Backend<S>> = Box::new(MosaicState::new(num_qubits)?);
        Self::from_branches(num_qubits, selector_dim, vec![(0, S::one(), state)])
    }

    /// Assemble a register from explicit `(selector, weight, state)`
    /// branches. Widths must match; selectors must lie below the
    /// qudit dimension. Branch states may be *any* mix of backends.
    pub fn from_branches(
        num_qubits: usize,
        selector_dim: usize,
        branches: Vec<(usize, S, Box<dyn Backend<S>>)>,
    ) -> Result<Self> {
        if selector_dim == 0 {
            return Err(Error::InvalidState(
                "branched: selector_dim must be ≥ 1".into(),
            ));
        }
        if branches.is_empty() {
            return Err(Error::InvalidState("branched: at least one branch".into()));
        }
        let mut out = Vec::with_capacity(branches.len());
        for (sel, weight, state) in branches {
            if sel >= selector_dim {
                return Err(Error::InvalidState(format!(
                    "branched: selector {sel} outside qudit dimension {selector_dim}"
                )));
            }
            if state.num_qubits() != num_qubits {
                return Err(Error::BadDimension {
                    expected: num_qubits,
                    got: state.num_qubits(),
                });
            }
            out.push(Branch {
                sel,
                weight,
                state: Rc::new(RefCell::new(state)),
            });
        }
        let peak = out.len();
        Ok(BranchedRegister {
            n: num_qubits,
            dim: selector_dim,
            branches: out,
            peak_branches: peak,
        })
    }

    /// Selector qudit dimension `d`.
    pub fn selector_dim(&self) -> usize {
        self.dim
    }

    /// Current number of branches in the multiset.
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }

    /// Largest the branch multiset has ever been — the ledger of what
    /// selector-side operations cost.
    pub fn peak_branches(&self) -> usize {
        self.peak_branches
    }

    /// Number of distinct underlying branch states (shared states
    /// counted once).
    pub fn unique_state_count(&self) -> usize {
        self.unique_states().len()
    }

    fn unique_states(&self) -> Vec<Shared<S>> {
        let mut out: Vec<Shared<S>> = Vec::new();
        for b in &self.branches {
            if !out.iter().any(|s| Rc::ptr_eq(s, &b.state)) {
                out.push(Rc::clone(&b.state));
            }
        }
        out
    }

    /// Flagged amplitude `⟨sel, index|Ψ⟩ = Σ_{k: selₖ=sel} wₖ ψₖ(index)`.
    pub fn flagged_amplitude(&self, sel: usize, index: u64) -> S {
        let mut acc = S::zero();
        for b in self.branches.iter().filter(|b| b.sel == sel) {
            acc = acc + b.weight * b.state.borrow().amplitude(index);
        }
        acc
    }

    // ── the Gram machinery: everything selector-side is polynomial ──

    /// Pairwise inner products `⟨ψₚ|ψ_q⟩` of the unique states, by
    /// hash-join over each state's own support enumeration — one
    /// `for_each_nonzero` per state and no per-index amplitude queries
    /// anywhere, so representations with expensive point queries (the
    /// graph-state bundle) cost their enumeration, once.
    fn state_gram(&self, uniques: &[Shared<S>]) -> Vec<S> {
        let m = uniques.len();
        let supports: Vec<HashMap<u64, S>> = uniques
            .iter()
            .map(|u| {
                let mut map = HashMap::new();
                u.borrow().for_each_nonzero(&mut |i, a| {
                    map.insert(i, a);
                });
                map
            })
            .collect();
        let mut g = vec![S::zero(); m * m];
        for p in 0..m {
            let mut norm = 0.0;
            for a in supports[p].values() {
                norm += a.abs_sqr();
            }
            g[p * m + p] = S::one().scale(norm);
            for q in (p + 1)..m {
                let (small, large, flip) = if supports[p].len() <= supports[q].len() {
                    (&supports[p], &supports[q], false)
                } else {
                    (&supports[q], &supports[p], true)
                };
                let mut acc = S::zero();
                for (i, &a) in small {
                    if let Some(&b) = large.get(i) {
                        acc = acc + if flip { b.conj() * a } else { a.conj() * b };
                    }
                }
                g[p * m + q] = acc;
                g[q * m + p] = acc.conj();
            }
        }
        g
    }

    /// Gram matrix of the `d` flagged slices
    /// `|slice_s⟩ = Σ_{k: selₖ=s} wₖ|ψₖ⟩`.
    fn slice_gram(&self) -> Vec<S> {
        let uniques = self.unique_states();
        let idx = |st: &Shared<S>| {
            uniques
                .iter()
                .position(|u| Rc::ptr_eq(u, st))
                .expect("state in unique list")
        };
        let g = self.state_gram(&uniques);
        let m = uniques.len();
        let d = self.dim;
        let mut out = vec![S::zero(); d * d];
        for a in &self.branches {
            for b in &self.branches {
                let inner = g[idx(&a.state) * m + idx(&b.state)];
                out[a.sel * d + b.sel] =
                    out[a.sel * d + b.sel] + a.weight.conj() * inner * b.weight;
            }
        }
        out
    }

    /// Born weights of the selector readings: `‖slice_s‖²`, normalized.
    /// Polynomial in branches — never an amplitude enumeration of the
    /// joint register.
    pub fn selector_probabilities(&self) -> Result<Vec<f64>> {
        let gram = self.slice_gram();
        let d = self.dim;
        let diag: Vec<f64> = (0..d).map(|s| gram[s * d + s].re()).collect();
        let total: f64 = diag.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return Err(Error::InvalidState(format!(
                "branched: total selector weight {total} is not positive"
            )));
        }
        Ok(diag.iter().map(|w| (w / total).max(0.0)).collect())
    }

    /// Schmidt rank across the selector|system cut — the *measured*
    /// mandatory entanglement: how many independent system states the
    /// clock distinguishes. Computed as the rank of the slice Gram.
    /// Requires a commutative division algebra (the Jacobi SVD).
    pub fn selector_schmidt_rank(&self) -> Result<usize> {
        if !(S::COMMUTATIVE && S::DIVISION) {
            return Err(Error::InvalidState(format!(
                "selector_schmidt_rank requires a commutative division algebra; {} is not",
                S::algebra_name()
            )));
        }
        let gram = self.slice_gram();
        let svd = svd_thin(self.dim, self.dim, &gram, 0.0, 0.0)?;
        let smax = svd.sigma.first().copied().unwrap_or(0.0);
        Ok(svd.sigma.iter().filter(|&&s| s > smax * 1e-9).count())
    }

    // ── selector-side operations ────────────────────────────────────

    /// Multiply each slice by a phase/weight: `wₖ ← φ(selₖ)·wₖ`. The
    /// qudit clock operator `clock_d` is the special case
    /// `φ(s) = ω^s`. Free: no branch is copied.
    pub fn selector_phase(&mut self, phases: &[S]) -> Result<()> {
        if phases.len() != self.dim {
            return Err(Error::BadDimension {
                expected: self.dim,
                got: phases.len(),
            });
        }
        for b in &mut self.branches {
            b.weight = phases[b.sel] * b.weight;
        }
        self.consolidate();
        Ok(())
    }

    /// Cyclic shift of the selector: `|s⟩ → |s + k mod d⟩` (the qudit
    /// `shift_d`). Free: relabeling only.
    pub fn selector_shift(&mut self, k: usize) {
        for b in &mut self.branches {
            b.sel = (b.sel + k) % self.dim;
        }
        self.consolidate();
    }

    /// Apply a general `d × d` unitary to the selector qudit
    /// (row-major, column action `|j⟩ → Σᵢ U[i][j]|i⟩`). Branches are
    /// *shared*, so this multiplies the multiset by at most the number
    /// of nonzero entries per column — bookkeeping, not state copies —
    /// then consolidates by `(selector, state)` and prunes zero
    /// weight. Interference between slices becomes real at amplitude
    /// time. Unitarity is validated.
    pub fn selector_mix(&mut self, u: &[S]) -> Result<()> {
        let d = self.dim;
        if u.len() != d * d {
            return Err(Error::BadDimension {
                expected: d * d,
                got: u.len(),
            });
        }
        let mut dev = 0.0f64;
        for i in 0..d {
            for j in 0..d {
                let mut acc = S::zero();
                for k in 0..d {
                    acc = acc + u[k * d + i].conj() * u[k * d + j];
                }
                let target = if i == j { acc - S::one() } else { acc };
                dev = dev.max(target.abs_sqr().sqrt());
            }
        }
        if dev > 1e-9 {
            return Err(Error::NotUnitary {
                label: "selector_mix".into(),
                deviation: dev,
            });
        }
        let mut out: Vec<Branch<S>> = Vec::new();
        for b in &self.branches {
            for i in 0..d {
                let c = u[i * d + b.sel];
                if c.abs_sqr() == 0.0 {
                    continue;
                }
                out.push(Branch {
                    sel: i,
                    weight: c * b.weight,
                    state: Rc::clone(&b.state),
                });
            }
        }
        self.branches = out;
        self.peak_branches = self.peak_branches.max(self.branches.len());
        self.consolidate();
        Ok(())
    }

    /// Merge branches with identical `(selector, state)` and prune
    /// negligible weight.
    fn consolidate(&mut self) {
        let mut merged: Vec<Branch<S>> = Vec::new();
        for b in self.branches.drain(..) {
            if let Some(hit) = merged
                .iter_mut()
                .find(|m| m.sel == b.sel && Rc::ptr_eq(&m.state, &b.state))
            {
                hit.weight = hit.weight + b.weight;
            } else {
                merged.push(b);
            }
        }
        merged.retain(|b| b.weight.abs_sqr() > BRANCH_PRUNE_TOL);
        if merged.is_empty() {
            // A fully cancelled register keeps one zero-weight branch so
            // the shape stays well-formed; the weight says the truth.
            merged.push(Branch {
                sel: 0,
                weight: S::zero(),
                state: Rc::new(RefCell::new(Box::new(
                    SparseState::new(self.n).expect("width validated"),
                ))),
            });
        }
        self.branches = merged;
    }

    /// Post-select the clock on `sel` (Page–Wootters conditioning):
    /// keep that slice, renormalized. Returns the Born probability the
    /// projective reading would have had.
    pub fn condition(&mut self, sel: usize) -> Result<f64> {
        if sel >= self.dim {
            return Err(Error::InvalidState(format!(
                "branched: selector {sel} outside qudit dimension {}",
                self.dim
            )));
        }
        let probs = self.selector_probabilities()?;
        let p = probs[sel];
        if p <= 0.0 {
            return Err(Error::InvalidState(format!(
                "branched: conditioning on selector {sel} with zero weight"
            )));
        }
        let gram = self.slice_gram();
        let norm = gram[sel * self.dim + sel].re().sqrt();
        self.branches.retain(|b| b.sel == sel);
        for b in &mut self.branches {
            b.weight = b.weight.scale(1.0 / norm);
        }
        self.consolidate();
        Ok(p)
    }

    /// Measure the selector qudit: draw a reading from the Born
    /// distribution over slices, collapse onto it, return it.
    pub fn measure_selector(&mut self, rng: &mut Prng) -> Result<usize> {
        let probs = self.selector_probabilities()?;
        let mut u = rng.next_f64();
        let mut outcome = self.dim - 1;
        for (s, &p) in probs.iter().enumerate() {
            if u < p {
                outcome = s;
                break;
            }
            u -= p;
        }
        self.condition(outcome)?;
        Ok(outcome)
    }

    fn for_each_unique(
        &mut self,
        mut f: impl FnMut(&mut dyn Backend<S>) -> Result<()>,
    ) -> Result<()> {
        let uniques = self.unique_states();
        for state in uniques {
            f(state.borrow_mut().as_mut())?;
        }
        Ok(())
    }

    /// Slice-addressed surgery on the record: apply `f` once to every
    /// unique state flagged `sel` — the retrocorrection hook, where a
    /// correction transported back through the intervening dynamics is
    /// applied to a *stored past slice* in register.
    ///
    /// Refused when a target state is shared with a branch of a
    /// **different** selector: rewriting it would silently rewrite the
    /// other slice's history too. Unshare first; surgery does not
    /// operate through walls.
    pub fn apply_at(
        &mut self,
        sel: usize,
        mut f: impl FnMut(&mut dyn Backend<S>) -> Result<()>,
    ) -> Result<()> {
        if sel >= self.dim {
            return Err(Error::InvalidState(format!(
                "branched: selector {sel} outside qudit dimension {}",
                self.dim
            )));
        }
        let targets: Vec<Shared<S>> = {
            let mut out: Vec<Shared<S>> = Vec::new();
            for b in self.branches.iter().filter(|b| b.sel == sel) {
                if !out.iter().any(|s| Rc::ptr_eq(s, &b.state)) {
                    out.push(Rc::clone(&b.state));
                }
            }
            out
        };
        for t in &targets {
            if self
                .branches
                .iter()
                .any(|b| b.sel != sel && Rc::ptr_eq(&b.state, t))
            {
                return Err(Error::InvalidState(format!(
                    "branched: slice {sel} shares a state with another \
                     selector; surgery on it would rewrite that slice's \
                     history too — unshare first"
                )));
            }
        }
        for t in targets {
            f(t.borrow_mut().as_mut())?;
        }
        Ok(())
    }
}

impl<S: Scalar> Backend<S> for BranchedRegister<S> {
    fn name(&self) -> &str {
        "branched"
    }

    fn num_qubits(&self) -> usize {
        self.n
    }

    /// System gates broadcast to each unique branch state **once** —
    /// the additive cost the selector structure exists to enable.
    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        self.for_each_unique(|st| st.apply(matrix, qubits))
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        self.for_each_unique(|st| st.apply_diagonal(entries, qubits))
    }

    /// Contracted view: `Σₖ wₖ ψₖ(index)` — the selector traced
    /// against the weights.
    fn amplitude(&self, index: u64) -> S {
        let mut acc = S::zero();
        for b in &self.branches {
            acc = acc + b.weight * b.state.borrow().amplitude(index);
        }
        acc
    }

    /// Union-support enumeration (documented trait behavior:
    /// exponential when a branch has dense support).
    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        let mut acc: HashMap<u64, S> = HashMap::new();
        for b in &self.branches {
            b.state.borrow().for_each_nonzero(&mut |i, a| {
                let slot = acc.entry(i).or_insert_with(S::zero);
                *slot = *slot + b.weight * a;
            });
        }
        let mut entries: Vec<(u64, S)> =
            acc.into_iter().filter(|(_, a)| a.abs_sqr() > 0.0).collect();
        entries.sort_unstable_by_key(|&(i, _)| i);
        for (i, a) in entries {
            f(i, a);
        }
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        let _ = self.for_each_unique(|st| {
            st.project(qubit, outcome, renorm);
            Ok(())
        });
    }

    fn reset(&mut self) {
        let state: Box<dyn Backend<S>> =
            Box::new(MosaicState::new(self.n).expect("width validated"));
        self.branches = vec![Branch {
            sel: 0,
            weight: S::one(),
            state: Rc::new(RefCell::new(state)),
        }];
    }

    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        let mut state = MosaicState::new(self.n)?;
        state.load(entries)?;
        self.branches = vec![Branch {
            sel: 0,
            weight: S::one(),
            state: Rc::new(RefCell::new(Box::new(state))),
        }];
        Ok(())
    }

    /// Additive: the sum of the unique branch representations plus the
    /// multiset bookkeeping — the measured payoff.
    fn memory_bytes(&self) -> usize {
        let states: usize = self
            .unique_states()
            .iter()
            .map(|s| s.borrow().memory_bytes())
            .sum();
        states
            + self.branches.len() * std::mem::size_of::<Branch<S>>()
            + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ── scale as the clock's axis ───────────────────────────────────────

/// The Page–Wootters history register over a bulk register's **own
/// depth levels**: `Σ_ℓ w_ℓ |ℓ⟩ ⊗ |ψ at scale ℓ⟩`, one weight per
/// scale (`weights[ℓ]`, up to `depth + 1` of them; zero weights build
/// no branch). Every slice is a native
/// [`scale_snapshot`](BulkState::scale_snapshot) — `O(tree)` each, so
/// wide histories stay in kilobytes. The selector qudit dimension is
/// `weights.len()`.
pub fn scale_history<S: Scalar>(bulk: &BulkState<S>, weights: &[S]) -> Result<BranchedRegister<S>> {
    if weights.is_empty() || weights.len() > bulk.depth() + 1 {
        return Err(Error::InvalidState(format!(
            "scale_history: {} weights for a hierarchy of depth {} (need 1..={})",
            weights.len(),
            bulk.depth(),
            bulk.depth() + 1
        )));
    }
    let mut branches: Vec<(usize, S, Box<dyn Backend<S>>)> = Vec::new();
    for (level, &w) in weights.iter().enumerate() {
        if w.abs_sqr() == 0.0 {
            continue;
        }
        branches.push((level, w, Box::new(bulk.scale_snapshot(level)?)));
    }
    if branches.is_empty() {
        return Err(Error::InvalidState(
            "scale_history: all weights are zero".into(),
        ));
    }
    BranchedRegister::from_branches(bulk.num_qubits(), weights.len(), branches)
}

/// One unit of internal time: the clock-controlled refinement
/// `Σ_ℓ |ℓ+1⟩⟨ℓ| ⊗ V_ℓ`, where `V_ℓ` is level `ℓ` of the unfold —
/// every branch at scale `ℓ` receives that level's dilated unitaries
/// and its clock advances. Isometric per slice, so slice norms are
/// preserved. Refuses, by name, a register whose branch states are
/// shared across distinct scales (interfere **after** ticking, or
/// condition first) and a clock already at the top of its dimension.
pub fn tick<S: Scalar>(reg: &mut BranchedRegister<S>, prog: &UnfoldProgram<S>) -> Result<()> {
    // Group branches by underlying state; each state must sit at one
    // scale for a controlled evolution to be applicable branch-wise.
    let mut per_state: Vec<(Shared<S>, usize)> = Vec::new();
    for b in &reg.branches {
        match per_state.iter().find(|(s, _)| Rc::ptr_eq(s, &b.state)) {
            None => per_state.push((Rc::clone(&b.state), b.sel)),
            Some((_, sel)) if *sel == b.sel => {}
            Some((_, sel)) => {
                return Err(Error::InvalidState(format!(
                    "tick: a branch state is shared across scales {sel} and {}; \
                     condition or tick before interfering the clock",
                    b.sel
                )));
            }
        }
    }
    if let Some((_, sel)) = per_state.iter().find(|(_, sel)| sel + 1 >= reg.dim) {
        return Err(Error::InvalidState(format!(
            "tick: scale {sel} is at the top of the clock (dimension {}); grow the qudit",
            reg.dim
        )));
    }
    for (state, sel) in &per_state {
        let mut st = state.borrow_mut();
        for step in prog.steps.iter().filter(|s| s.depth == *sel) {
            st.apply(&step.matrix, &step.qubits)?;
        }
    }
    for b in &mut reg.branches {
        b.sel += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::C64;

    #[test]
    fn selector_ops_are_bookkeeping_not_copies() {
        let mut r = BranchedRegister::<C64>::new(3, 4).unwrap();
        assert_eq!(r.branch_count(), 1);
        r.selector_shift(2);
        assert_eq!(r.branch_count(), 1);
        assert_eq!(r.unique_state_count(), 1);
        // A full mix fans one branch across the qudit, sharing the state.
        let d = 4;
        let mut u = vec![C64::new(0.0, 0.0); d * d];
        for i in 0..d {
            for j in 0..d {
                let phase = 2.0 * std::f64::consts::PI * (i * j) as f64 / d as f64;
                u[i * d + j] = C64::new(phase.cos() / 2.0, phase.sin() / 2.0);
            }
        }
        r.selector_mix(&u).unwrap();
        assert_eq!(r.branch_count(), 4);
        assert_eq!(r.unique_state_count(), 1, "mixing must share, not copy");
        assert_eq!(r.peak_branches(), 4);
    }

    #[test]
    fn a_non_unitary_mix_is_refused_with_the_deviation() {
        let mut r = BranchedRegister::<C64>::new(2, 2).unwrap();
        let u = vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.5, 0.0),
        ];
        match r.selector_mix(&u) {
            Err(Error::NotUnitary { label, deviation }) => {
                assert_eq!(label, "selector_mix");
                assert!(deviation > 0.1);
            }
            other => panic!("expected NotUnitary, got {other:?}"),
        }
    }
}
