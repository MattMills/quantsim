//! Factored state backend: hierarchical separation of the register.
//!
//! The state is maintained as a **tensor product of factors**, each factor a
//! dense amplitude vector over a subset of qubits. Gates merge factors only
//! when they genuinely couple them; projective measurement splits the
//! measured qubit back out exactly; and (over commutative algebras) a
//! rank-1 test re-separates qubits that a gate left unentangled. Memory is
//! the **sum** of factor sizes, so a circuit whose entanglement stays
//! geometrically local simulates in polynomial memory at widths where a
//! dense vector cannot exist — while a globally entangling circuit
//! degenerates, honestly, to one dense factor.
//!
//! This is the substrate for two research programs:
//!
//! * **non-exponential entanglement decomposition** — the factor partition
//!   *is* the current entanglement geometry ([`FactoredState::factors`]),
//!   and [`FactoredState::peak`] records the worst cost the circuit ever
//!   forced, so decomposition strategies are measurable;
//! * **quantum memory interacting with quantum computation** — idle memory
//!   qubits cost two amplitudes each in their own factors; interactions
//!   merge them into the compute region and measurement or uncomputation
//!   releases them again.
//!
//! Conventions and caveats: factors are combined left-to-right in order of
//! their smallest qubit index (only relevant for non-commutative algebras);
//! automatic rank-1 splitting runs only over commutative scalars (the exact
//! split after projection works everywhere); splitting introduces error
//! below `split_tol` per split (default 1e-11), the same order as sparse
//! pruning.

use super::{validate_apply, validate_apply_diagonal, Backend};
use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::scalar::Scalar;

/// Maximum width of a *single factor* (a merged factor is a dense vector;
/// 26 qubits of `C64` is already 1 GiB). The register itself can be as wide
/// as 63 qubits — the point of the representation is that factors stay
/// small when entanglement does.
pub const FACTOR_MAX_QUBITS: usize = 26;

/// Maximum register width (basis indices are `u64`).
pub const FACTORED_MAX_QUBITS: usize = 63;

#[derive(Debug, Clone)]
struct Factor<S: Scalar> {
    /// Global qubit ids; local bit `b` ↔ `qubits[b]`.
    qubits: Vec<usize>,
    /// Dense amplitudes over the factor's qubits.
    amps: Vec<S>,
}

impl<S: Scalar> Factor<S> {
    fn min_qubit(&self) -> usize {
        *self.qubits.iter().min().expect("factors are never empty")
    }
    fn local_position(&self, qubit: usize) -> usize {
        self.qubits
            .iter()
            .position(|&q| q == qubit)
            .expect("qubit_factor map out of sync")
    }
}

/// A state stored as a product of dense factors over qubit subsets.
#[derive(Debug, Clone)]
pub struct FactoredState<S: Scalar> {
    num_qubits: usize,
    factors: Vec<Factor<S>>,
    qubit_factor: Vec<usize>,
    auto_split: bool,
    split_tol: f64,
    peak_total_amps: usize,
    peak_factor_qubits: usize,
}

impl<S: Scalar> FactoredState<S> {
    /// `|0…0⟩` as `n` singleton factors.
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits > FACTORED_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: num_qubits,
                max: FACTORED_MAX_QUBITS,
            });
        }
        let mut state = FactoredState {
            num_qubits,
            factors: Vec::new(),
            qubit_factor: Vec::new(),
            auto_split: true,
            split_tol: 1e-11,
            peak_total_amps: 0,
            peak_factor_qubits: 0,
        };
        state.reset_factors();
        Ok(state)
    }

    fn reset_factors(&mut self) {
        self.factors = (0..self.num_qubits)
            .map(|q| Factor {
                qubits: vec![q],
                amps: vec![S::one(), S::zero()],
            })
            .collect();
        self.qubit_factor = (0..self.num_qubits).collect();
        self.note_peaks();
    }

    /// Enable/disable the automatic rank-1 split attempt after each gate
    /// (commutative algebras only; exact projection splits always run).
    pub fn set_auto_split(&mut self, enabled: bool) {
        self.auto_split = enabled;
    }

    /// Residual tolerance for accepting a rank-1 split.
    pub fn set_split_tolerance(&mut self, tol: f64) {
        self.split_tol = tol;
    }

    /// The current factor partition — the live entanglement geometry.
    /// Sorted by each factor's smallest qubit; qubits within a factor sorted.
    pub fn factors(&self) -> Vec<Vec<usize>> {
        let mut out: Vec<Vec<usize>> = self
            .factors
            .iter()
            .map(|f| {
                let mut qs = f.qubits.clone();
                qs.sort_unstable();
                qs
            })
            .collect();
        out.sort_by_key(|qs| qs[0]);
        out
    }

    /// Number of factors (`num_qubits` when fully separated, 1 when fully
    /// entangled).
    pub fn factor_count(&self) -> usize {
        self.factors.len()
    }

    /// Width of the largest current factor.
    pub fn largest_factor_qubits(&self) -> usize {
        self.factors
            .iter()
            .map(|f| f.qubits.len())
            .max()
            .unwrap_or(0)
    }

    /// Lifetime peaks since construction or [`Backend::reset`]:
    /// `(max total stored amplitudes, max single-factor width)` — the
    /// measurable cost a circuit's entanglement geometry ever forced.
    pub fn peak(&self) -> (usize, usize) {
        (self.peak_total_amps, self.peak_factor_qubits)
    }

    fn note_peaks(&mut self) {
        let total: usize = self.factors.iter().map(|f| f.amps.len()).sum();
        self.peak_total_amps = self.peak_total_amps.max(total);
        self.peak_factor_qubits = self.peak_factor_qubits.max(self.largest_factor_qubits());
    }

    fn sorted_factor_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.factors.len()).collect();
        order.sort_by_key(|&i| self.factors[i].min_qubit());
        order
    }

    /// Merge the factors of `a` and `b` (indices into `self.factors`),
    /// returning the surviving factor index. The factor with the smaller
    /// minimum qubit becomes the left tensor operand.
    fn merge_two(&mut self, a: usize, b: usize) -> Result<usize> {
        debug_assert_ne!(a, b);
        let (left_idx, right_idx) = if self.factors[a].min_qubit() <= self.factors[b].min_qubit() {
            (a, b)
        } else {
            (b, a)
        };
        let k_left = self.factors[left_idx].qubits.len();
        let k_right = self.factors[right_idx].qubits.len();
        if k_left + k_right > FACTOR_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: k_left + k_right,
                max: FACTOR_MAX_QUBITS,
            });
        }
        let right = self.factors[right_idx].clone();
        let left = &self.factors[left_idx];
        let mut qubits = left.qubits.clone();
        qubits.extend(&right.qubits);
        let mut amps = vec![S::zero(); left.amps.len() * right.amps.len()];
        for (ir, &ra) in right.amps.iter().enumerate() {
            if ra.abs_sqr() == 0.0 {
                continue;
            }
            for (il, &la) in left.amps.iter().enumerate() {
                if la.abs_sqr() == 0.0 {
                    continue;
                }
                amps[(ir << k_left) | il] = la * ra;
            }
        }
        // Survivor lives at the smaller of the two slots; drop the other.
        let (keep, drop) = (left_idx.min(right_idx), left_idx.max(right_idx));
        self.factors[keep] = Factor { qubits, amps };
        self.factors.swap_remove(drop);
        self.rebuild_qubit_map();
        Ok(keep)
    }

    fn rebuild_qubit_map(&mut self) {
        for (idx, factor) in self.factors.iter().enumerate() {
            for &q in &factor.qubits {
                self.qubit_factor[q] = idx;
            }
        }
    }

    /// Ensure all target qubits share one factor; returns its index.
    fn merge_for(&mut self, qubits: &[usize]) -> Result<usize> {
        loop {
            let mut involved: Vec<usize> = qubits.iter().map(|&q| self.qubit_factor[q]).collect();
            involved.sort_unstable();
            involved.dedup();
            match involved.len() {
                1 => return Ok(involved[0]),
                _ => {
                    self.merge_two(involved[0], involved[1])?;
                }
            }
        }
    }

    /// Exact split after a projection left `qubit` in the pure basis state
    /// `outcome` within its factor.
    fn split_projected(&mut self, qubit: usize, outcome: bool) {
        let fidx = self.qubit_factor[qubit];
        if self.factors[fidx].qubits.len() == 1 {
            return;
        }
        let p = self.factors[fidx].local_position(qubit);
        let mask = 1usize << p;
        let old = &self.factors[fidx];
        let half = old.amps.len() >> 1;
        let mut rest = vec![S::zero(); half];
        for (r, slot) in rest.iter_mut().enumerate() {
            let low = r & (mask - 1);
            let base = ((r >> p) << (p + 1)) | low;
            *slot = old.amps[base | if outcome { mask } else { 0 }];
        }
        let mut qubits = old.qubits.clone();
        qubits.remove(p);
        self.factors[fidx] = Factor { qubits, amps: rest };
        let single = Factor {
            qubits: vec![qubit],
            amps: if outcome {
                vec![S::zero(), S::one()]
            } else {
                vec![S::one(), S::zero()]
            },
        };
        self.factors.push(single);
        self.rebuild_qubit_map();
    }

    /// Attempt a rank-1 split of `qubit` out of its factor (commutative
    /// algebras only — tensor-order conventions make the general test
    /// ambiguous otherwise).
    fn try_split(&mut self, qubit: usize) {
        if !S::COMMUTATIVE {
            return;
        }
        let fidx = self.qubit_factor[qubit];
        if self.factors[fidx].qubits.len() == 1 {
            return;
        }
        let p = self.factors[fidx].local_position(qubit);
        let mask = 1usize << p;
        let amps = &self.factors[fidx].amps;
        let half = amps.len() >> 1;
        let rest_index = |r: usize| {
            let low = r & (mask - 1);
            ((r >> p) << (p + 1)) | low
        };
        // Pivot column with the largest weight.
        let mut pivot = 0usize;
        let mut best = -1.0f64;
        for r in 0..half {
            let i0 = rest_index(r);
            let w = amps[i0].abs_sqr() + amps[i0 | mask].abs_sqr();
            if w > best {
                best = w;
                pivot = r;
            }
        }
        if best <= 0.0 {
            return;
        }
        let norm = best.sqrt();
        let i0 = rest_index(pivot);
        let alpha0 = amps[i0].scale(1.0 / norm);
        let alpha1 = amps[i0 | mask].scale(1.0 / norm);
        // Projected rest vector and rank-1 residual.
        let mut rest = vec![S::zero(); half];
        let mut residual: f64 = 0.0;
        for (r, slot) in rest.iter_mut().enumerate() {
            let j0 = rest_index(r);
            let j1 = j0 | mask;
            let v = alpha0.conj() * amps[j0] + alpha1.conj() * amps[j1];
            residual = residual.max((alpha0 * v - amps[j0]).abs_sqr());
            residual = residual.max((alpha1 * v - amps[j1]).abs_sqr());
            *slot = v;
        }
        if residual.sqrt() > self.split_tol {
            return;
        }
        let mut qubits = self.factors[fidx].qubits.clone();
        qubits.remove(p);
        self.factors[fidx] = Factor { qubits, amps: rest };
        self.factors.push(Factor {
            qubits: vec![qubit],
            amps: vec![alpha0, alpha1],
        });
        self.rebuild_qubit_map();
    }

    fn local_targets(&self, fidx: usize, qubits: &[usize]) -> Vec<usize> {
        qubits
            .iter()
            .map(|&q| self.factors[fidx].local_position(q))
            .collect()
    }

    /// Fast per-shot sampler exploiting the product structure: each factor
    /// is sampled independently (exact for the Born distribution of a
    /// product state) in `O(shots · Σ factor sizes)` — no support
    /// enumeration. Prefer this over the trait's [`Backend::sample`] for
    /// wide, well-factored states; the trait method is kept enumeration-
    /// based so its counts are bit-identical with the other backends.
    pub fn sample_factored(
        &self,
        shots: u64,
        rng: &mut crate::rng::Prng,
    ) -> Result<std::collections::HashMap<u64, u64>> {
        let order = self.sorted_factor_order();
        // Per-factor cumulative weights.
        let mut tables: Vec<(usize, Vec<(u64, f64)>)> = Vec::new();
        for &fi in &order {
            let f = &self.factors[fi];
            let mut cumulative = 0.0;
            let mut table = Vec::new();
            for (local, &a) in f.amps.iter().enumerate() {
                let w = a.born_weight();
                if w > 0.0 {
                    cumulative += w;
                    let mut global = 0u64;
                    for (b, &q) in f.qubits.iter().enumerate() {
                        global |= (((local >> b) & 1) as u64) << q;
                    }
                    table.push((global, cumulative));
                }
            }
            if cumulative <= 0.0 || !cumulative.is_finite() {
                return Err(Error::InvalidState(format!(
                    "factor Born weight {cumulative} is not positive; cannot sample"
                )));
            }
            tables.push((fi, table));
        }
        let mut counts = std::collections::HashMap::new();
        for _ in 0..shots {
            let mut index = 0u64;
            for (_, table) in &tables {
                let total = table.last().expect("nonempty by construction").1;
                let u = rng.next_f64() * total;
                let pos = table.partition_point(|&(_, c)| c <= u).min(table.len() - 1);
                index |= table[pos].0;
            }
            *counts.entry(index).or_insert(0) += 1;
        }
        Ok(counts)
    }
}

impl<S: Scalar> Backend<S> for FactoredState<S> {
    fn name(&self) -> &str {
        "factored"
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        validate_apply(self.num_qubits, matrix, qubits)?;
        let fidx = self.merge_for(qubits)?;
        let local = self.local_targets(fidx, qubits);
        let amps = &mut self.factors[fidx].amps;
        if local.len() == 1 {
            super::apply_single_in_place(amps, matrix, local[0]);
        } else {
            super::apply_general_in_place(amps, matrix, &local);
        }
        self.note_peaks();
        if self.auto_split {
            for &q in qubits {
                self.try_split(q);
            }
        }
        Ok(())
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        validate_apply_diagonal(self.num_qubits, entries, qubits)?;
        let fidx = self.merge_for(qubits)?;
        let local = self.local_targets(fidx, qubits);
        super::apply_diagonal_in_place(&mut self.factors[fidx].amps, entries, &local);
        self.note_peaks();
        if self.auto_split {
            for &q in qubits {
                self.try_split(q);
            }
        }
        Ok(())
    }

    fn amplitude(&self, index: u64) -> S {
        let mut product = S::one();
        for &fi in &self.sorted_factor_order() {
            let f = &self.factors[fi];
            let mut local = 0usize;
            for (b, &q) in f.qubits.iter().enumerate() {
                local |= (((index >> q) & 1) as usize) << b;
            }
            let a = f.amps[local];
            if a.abs_sqr() == 0.0 {
                return S::zero();
            }
            product = product * a;
        }
        product
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        let order = self.sorted_factor_order();
        fn recurse<S: Scalar>(
            state: &FactoredState<S>,
            order: &[usize],
            depth: usize,
            index: u64,
            amp: S,
            f: &mut dyn FnMut(u64, S),
        ) {
            if depth == order.len() {
                f(index, amp);
                return;
            }
            let factor = &state.factors[order[depth]];
            for (local, &a) in factor.amps.iter().enumerate() {
                if a.abs_sqr() == 0.0 {
                    continue;
                }
                let mut next = index;
                for (b, &q) in factor.qubits.iter().enumerate() {
                    next |= (((local >> b) & 1) as u64) << q;
                }
                recurse(state, order, depth + 1, next, amp * a, f);
            }
        }
        recurse(self, &order, 0, 0, S::one(), f);
    }

    fn nonzero_count(&self) -> usize {
        self.factors
            .iter()
            .map(|f| f.amps.iter().filter(|a| a.abs_sqr() > 0.0).count())
            .fold(1usize, |acc, c| acc.saturating_mul(c))
    }

    fn total_weight(&self) -> f64 {
        if S::DIVISION {
            // Born weight is multiplicative over tensor factors.
            self.factors
                .iter()
                .map(|f| f.amps.iter().map(|a| a.born_weight()).sum::<f64>())
                .product()
        } else {
            let mut total = 0.0;
            self.for_each_nonzero(&mut |_, a| total += a.born_weight());
            total
        }
    }

    fn measure(&mut self, qubit: usize, rng: &mut crate::rng::Prng) -> Result<bool> {
        if qubit >= self.num_qubits {
            return Err(Error::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits,
            });
        }
        if !S::DIVISION {
            // Weight cancellation across factors is invalid for indefinite
            // Born forms; fall back to full-support accumulation.
            return default_measure(self, qubit, rng);
        }
        // Marginal within the qubit's own factor: other factors' weights
        // cancel in the ratio.
        let fidx = self.qubit_factor[qubit];
        let p = self.factors[fidx].local_position(qubit);
        let mask = 1usize << p;
        let mut total = 0.0;
        let mut one = 0.0;
        for (local, &a) in self.factors[fidx].amps.iter().enumerate() {
            let w = a.born_weight();
            total += w;
            if local & mask != 0 {
                one += w;
            }
        }
        if total <= 0.0 || !total.is_finite() {
            return Err(Error::InvalidState(format!(
                "factor Born weight {total} is not positive; cannot measure"
            )));
        }
        let p_one = (one / total).clamp(0.0, 1.0);
        let outcome = rng.next_f64() < p_one;
        let p_outcome = if outcome { p_one } else { 1.0 - p_one };
        let renorm = 1.0 / (p_outcome * total).sqrt();
        self.project(qubit, outcome, renorm);
        Ok(outcome)
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        let fidx = self.qubit_factor[qubit];
        let p = self.factors[fidx].local_position(qubit);
        let mask = 1usize << p;
        for (local, a) in self.factors[fidx].amps.iter_mut().enumerate() {
            if ((local & mask) != 0) == outcome {
                *a = a.scale(renorm);
            } else {
                *a = S::zero();
            }
        }
        // The projected qubit is now exactly separable.
        self.split_projected(qubit, outcome);
        self.note_peaks();
    }

    fn reset(&mut self) {
        self.peak_total_amps = 0;
        self.peak_factor_qubits = 0;
        self.reset_factors();
    }

    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        if self.num_qubits > FACTOR_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: self.num_qubits,
                max: FACTOR_MAX_QUBITS,
            });
        }
        let limit = 1u64 << self.num_qubits;
        for &(i, _) in entries {
            if i >= limit {
                return Err(Error::QubitOutOfRange {
                    qubit: 64 - i.leading_zeros() as usize,
                    num_qubits: self.num_qubits,
                });
            }
        }
        let mut amps = vec![S::zero(); 1usize << self.num_qubits];
        for &(i, a) in entries {
            amps[i as usize] = a;
        }
        self.factors = vec![Factor {
            qubits: (0..self.num_qubits).collect(),
            amps,
        }];
        self.qubit_factor = vec![0; self.num_qubits];
        self.note_peaks();
        if self.auto_split {
            for q in 0..self.num_qubits {
                self.try_split(q);
            }
        }
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        let amps: usize = self.factors.iter().map(|f| f.amps.capacity()).sum();
        let bookkeeping: usize = self
            .factors
            .iter()
            .map(|f| f.qubits.capacity() * std::mem::size_of::<usize>())
            .sum();
        amps * std::mem::size_of::<S>()
            + bookkeeping
            + self.qubit_factor.capacity() * std::mem::size_of::<usize>()
            + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// The trait's default measurement logic (full-support accumulation),
/// reachable from the specialized override's non-division fallback.
fn default_measure<S: Scalar>(
    state: &mut FactoredState<S>,
    qubit: usize,
    rng: &mut crate::rng::Prng,
) -> Result<bool> {
    let mut total = 0.0;
    let mut one = 0.0;
    let mask = 1u64 << qubit;
    state.for_each_nonzero(&mut |i, a| {
        let w = a.born_weight();
        total += w;
        if i & mask != 0 {
            one += w;
        }
    });
    if total <= 0.0 || !total.is_finite() {
        return Err(Error::InvalidState(format!(
            "total Born weight {total} is not positive; cannot sample a measurement"
        )));
    }
    let p_one = (one / total).clamp(0.0, 1.0);
    let outcome = rng.next_f64() < p_one;
    let p_outcome = if outcome { p_one } else { 1.0 - p_one };
    let renorm = 1.0 / (p_outcome * total).sqrt();
    state.project(qubit, outcome, renorm);
    Ok(outcome)
}
