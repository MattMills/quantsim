//! Matrix-product-state backend: width scaling by bond dimension.
//!
//! The state is a chain of site tensors `A_s[left][physical][right]`; the
//! bond dimensions between sites carry exactly as much entanglement as the
//! state has Schmidt rank across each cut. Cost model:
//!
//! * memory `O(n · χ² )` for maximum bond dimension χ — **linear in
//!   register width** whenever entanglement across every cut stays bounded,
//!   which is a *different* compression axis from both sparse (support
//!   size) and factored (cluster width). GHZ — one global cluster that
//!   defeats the factored backend — is Schmidt-rank 2 across every cut:
//!   bond dimension 2, linear memory, at any width.
//! * gates: single-site exact; multi-qubit gates act on a contiguous site
//!   window (non-adjacent operands are routed by stepwise SWAPs, like the
//!   device model) — contract, apply, re-split by SVD with truncation at
//!   [`MpsConfig::trunc_tol`] capped at [`MpsConfig::max_bond`].
//!
//! Truncation makes MPS the crate's first *approximate* backend: with a
//! generous cap it is exact and passes conformance at reference tolerance;
//! with a tight cap it degrades **measurably** (see the truncation tests) —
//! never silently.
//!
//! Requires a commutative division algebra (ℝ, ℂ, `CD<f64>`): the Jacobi
//! SVD rotations assume commuting scalars. Quaternionic MPS is a roadmap
//! item, not a silent wrong answer — construction fails loudly elsewhere.
//!
//! Trait notes: `for_each_nonzero` walks the chain depth-first with
//! branch pruning (cheap for concentrated states, exponential for generic
//! ones — documented trait behavior); `measure`/`total_weight` use norm
//! environments in `O(n·χ³)`; [`MpsState::sample_mps`] is the native
//! conditional sampler (the trait `sample` stays enumeration-based so its
//! counts are bit-identical across backends).

use std::collections::HashMap;

use super::{validate_apply, validate_apply_diagonal, Backend};
use crate::error::{Error, Result};
use crate::math::{svd_thin, GateMatrix};
use crate::rng::Prng;
use crate::scalar::Scalar;

/// Maximum register width (basis indices are `u64`).
pub const MPS_MAX_QUBITS: usize = 63;
/// Widest gate window (contiguous sites contracted at once).
pub const MPS_MAX_WINDOW: usize = 5;
/// Structural width bound for [`Backend::load`] (basis indices fit
/// `u64`); the exponential compilation buffer is admitted by the
/// [resource guard](crate::guard) against measured memory, and the
/// compilation itself checkpoints the guard's time budget.
pub const MPS_LOAD_MAX_QUBITS: usize = 63;

/// Truncation knobs for the MPS backend.
#[derive(Debug, Clone, Copy)]
pub struct MpsConfig {
    /// Hard cap on any bond dimension. Exceeding ranks are truncated —
    /// the approximation trade the representation exists to study.
    pub max_bond: usize,
    /// Relative singular-value cutoff at each split.
    pub trunc_tol: f64,
}

impl Default for MpsConfig {
    fn default() -> Self {
        MpsConfig {
            max_bond: 128,
            trunc_tol: 1e-12,
        }
    }
}

#[derive(Debug, Clone)]
struct SiteTensor<S: Scalar> {
    left: usize,
    right: usize,
    /// Index `((l * 2) + p) * right + r`.
    data: Vec<S>,
}

impl<S: Scalar> SiteTensor<S> {
    fn zero_site() -> Self {
        SiteTensor {
            left: 1,
            right: 1,
            data: vec![S::one(), S::zero()],
        }
    }
    #[inline]
    fn at(&self, l: usize, p: usize, r: usize) -> S {
        self.data[(l * 2 + p) * self.right + r]
    }
}

/// The matrix-product-state backend. See the module docs.
#[derive(Debug, Clone)]
pub struct MpsState<S: Scalar> {
    config: MpsConfig,
    tensors: Vec<SiteTensor<S>>,
    site_of_logical: Vec<usize>,
    logical_at_site: Vec<usize>,
    routing_swaps: usize,
    peak_bond: usize,
    peak_elements: usize,
}

impl<S: Scalar> MpsState<S> {
    /// `|0…0⟩` with default truncation config.
    pub fn new(num_qubits: usize) -> Result<Self> {
        Self::with_config(num_qubits, MpsConfig::default())
    }

    /// `|0…0⟩` with explicit truncation config.
    pub fn with_config(num_qubits: usize, config: MpsConfig) -> Result<Self> {
        if !(S::COMMUTATIVE && S::DIVISION) {
            return Err(Error::InvalidState(format!(
                "mps requires a commutative division algebra; {} is not",
                S::algebra_name()
            )));
        }
        if num_qubits > MPS_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: num_qubits,
                max: MPS_MAX_QUBITS,
            });
        }
        if config.max_bond == 0 {
            return Err(Error::InvalidState("mps: max_bond must be ≥ 1".into()));
        }
        let mut state = MpsState {
            config,
            tensors: Vec::new(),
            site_of_logical: Vec::new(),
            logical_at_site: Vec::new(),
            routing_swaps: 0,
            peak_bond: 0,
            peak_elements: 0,
        };
        state.reset_chain(num_qubits);
        Ok(state)
    }

    fn reset_chain(&mut self, n: usize) {
        self.tensors = (0..n).map(|_| SiteTensor::zero_site()).collect();
        self.site_of_logical = (0..n).collect();
        self.logical_at_site = (0..n).collect();
        self.note_peaks();
    }

    /// Bond dimensions between consecutive sites (`n − 1` entries).
    pub fn bond_dimensions(&self) -> Vec<usize> {
        self.tensors
            .iter()
            .take(self.tensors.len().saturating_sub(1))
            .map(|t| t.right)
            .collect()
    }

    /// Largest current bond dimension.
    pub fn max_bond_dimension(&self) -> usize {
        self.bond_dimensions().into_iter().max().unwrap_or(1)
    }

    /// Lifetime peaks since construction or reset:
    /// `(max bond dimension, max total tensor elements)`.
    pub fn peak(&self) -> (usize, usize) {
        (self.peak_bond, self.peak_elements)
    }

    /// SWAPs inserted to route non-adjacent operands.
    pub fn routing_swaps(&self) -> usize {
        self.routing_swaps
    }

    fn note_peaks(&mut self) {
        self.peak_bond = self.peak_bond.max(self.max_bond_dimension());
        let elements: usize = self.tensors.iter().map(|t| t.data.len()).sum();
        self.peak_elements = self.peak_elements.max(elements);
    }

    // ── window machinery ──────────────────────────────────────────────

    /// Contract sites `start .. start + k` into `Θ[l][p][r]` with window
    /// bit `o` (little-endian) belonging to site `start + o`.
    fn contract_window(&self, start: usize, k: usize) -> (usize, usize, Vec<S>) {
        let first = &self.tensors[start];
        let l_dim = first.left;
        let mut r_dim = first.right;
        let mut bits = 1usize;
        let mut theta = first.data.clone(); // already [l][p][r] with p = 1 bit
        for o in 1..k {
            let next = &self.tensors[start + o];
            let mut merged = vec![S::zero(); l_dim * (1 << (bits + 1)) * next.right];
            for l in 0..l_dim {
                for p in 0..(1 << bits) {
                    for r in 0..r_dim {
                        let amp = theta[(l * (1 << bits) + p) * r_dim + r];
                        if amp.abs_sqr() == 0.0 {
                            continue;
                        }
                        for pn in 0..2 {
                            for rn in 0..next.right {
                                let contribution = amp * next.at(r, pn, rn);
                                let p_new = p + (pn << bits);
                                let idx = (l * (1 << (bits + 1)) + p_new) * next.right + rn;
                                merged[idx] = merged[idx] + contribution;
                            }
                        }
                    }
                }
            }
            theta = merged;
            r_dim = next.right;
            bits += 1;
        }
        (l_dim, r_dim, theta)
    }

    /// Split `Θ` back into `k` site tensors via successive thin SVDs with
    /// truncation.
    fn split_window(
        &mut self,
        start: usize,
        k: usize,
        l_dim: usize,
        r_dim: usize,
        mut theta: Vec<S>,
    ) -> Result<()> {
        let mut cur_l = l_dim;
        let mut bits = k;
        for o in 0..k - 1 {
            let rows = cur_l * 2;
            let rest = 1usize << (bits - 1);
            let cols = rest * r_dim;
            // A[row][col], row = l·2 + p0, col = prest·r_dim + r.
            let mut a = vec![S::zero(); rows * cols];
            for l in 0..cur_l {
                for p in 0..(1 << bits) {
                    let (p0, prest) = (p & 1, p >> 1);
                    for r in 0..r_dim {
                        a[(l * 2 + p0) * cols + (prest * r_dim + r)] =
                            theta[(l * (1 << bits) + p) * r_dim + r];
                    }
                }
            }
            let svd = svd_thin(rows, cols, &a, self.config.trunc_tol, 1e-14)?;
            let chi = svd.rank.min(self.config.max_bond).max(1);
            let mut site = vec![S::zero(); cur_l * 2 * chi];
            for row in 0..rows {
                for c in 0..chi {
                    site[row * chi + c] = svd.u[row * svd.rank + c];
                }
            }
            self.tensors[start + o] = SiteTensor {
                left: cur_l,
                right: chi,
                data: site,
            };
            let mut next = vec![S::zero(); chi * rest * r_dim];
            for c in 0..chi {
                let scale = svd.sigma[c];
                for col in 0..cols {
                    next[c * cols + col] = svd.vt[c * cols + col].scale(scale);
                }
            }
            theta = next;
            cur_l = chi;
            bits -= 1;
        }
        // Last site: Θ is (cur_l, 2, r_dim) in exactly SiteTensor layout.
        self.tensors[start + k - 1] = SiteTensor {
            left: cur_l,
            right: r_dim,
            data: theta,
        };
        self.note_peaks();
        Ok(())
    }

    /// Apply a matrix (site-bit order) or diagonal to a contiguous window.
    fn apply_window_sites(
        &mut self,
        start: usize,
        k: usize,
        matrix: Option<&[S]>,
        diagonal: Option<&[S]>,
    ) -> Result<()> {
        let (l_dim, r_dim, theta) = self.contract_window(start, k);
        let d = 1usize << k;
        let transformed = if let Some(m) = matrix {
            let mut out = vec![S::zero(); theta.len()];
            for l in 0..l_dim {
                for r in 0..r_dim {
                    for p_new in 0..d {
                        let mut acc = S::zero();
                        for p in 0..d {
                            let entry = m[p_new * d + p];
                            if entry.abs_sqr() == 0.0 {
                                continue;
                            }
                            acc = acc + entry * theta[(l * d + p) * r_dim + r];
                        }
                        out[(l * d + p_new) * r_dim + r] = acc;
                    }
                }
            }
            out
        } else {
            let dg = diagonal.expect("matrix or diagonal");
            let mut out = theta;
            for l in 0..l_dim {
                for (p, &scale) in dg.iter().enumerate().take(d) {
                    for r in 0..r_dim {
                        let idx = (l * d + p) * r_dim + r;
                        out[idx] = scale * out[idx];
                    }
                }
            }
            out
        };
        if k == 1 {
            self.tensors[start] = SiteTensor {
                left: l_dim,
                right: r_dim,
                data: transformed,
            };
            self.note_peaks();
            Ok(())
        } else {
            self.split_window(start, k, l_dim, r_dim, transformed)
        }
    }

    /// Swap the qubits at sites `s` and `s + 1` (updates the mapping).
    fn swap_sites(&mut self, s: usize) -> Result<()> {
        let (o, z) = (S::one(), S::zero());
        let swap = [o, z, z, z, z, z, o, z, z, o, z, z, z, z, z, o];
        self.apply_window_sites(s, 2, Some(&swap), None)?;
        let (la, lb) = (self.logical_at_site[s], self.logical_at_site[s + 1]);
        self.logical_at_site[s] = lb;
        self.logical_at_site[s + 1] = la;
        self.site_of_logical[la] = s + 1;
        self.site_of_logical[lb] = s;
        self.routing_swaps += 1;
        Ok(())
    }

    /// Route the gate's qubits onto consecutive sites; returns the window
    /// start and, per window offset, which gate bit lives there.
    fn route_window(&mut self, qubits: &[usize]) -> Result<(usize, Vec<usize>)> {
        let k = qubits.len();
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by_key(|&b| self.site_of_logical[qubits[b]]);
        let anchor = self.site_of_logical[qubits[order[0]]];
        for (offset, &b) in order.iter().enumerate().skip(1) {
            let target = anchor + offset;
            let mut site = self.site_of_logical[qubits[b]];
            while site > target {
                self.swap_sites(site - 1)?;
                site -= 1;
            }
        }
        let gate_bit_at_offset: Vec<usize> = (0..k)
            .map(|offset| {
                let logical = self.logical_at_site[anchor + offset];
                qubits
                    .iter()
                    .position(|&q| q == logical)
                    .expect("routing placed a non-target qubit in the window")
            })
            .collect();
        Ok((anchor, gate_bit_at_offset))
    }

    /// Remap a gate-bit-ordered index table into window-site order.
    fn window_index_map(gate_bit_at_offset: &[usize]) -> Vec<usize> {
        let k = gate_bit_at_offset.len();
        (0..(1usize << k))
            .map(|window_p| {
                let mut gate_p = 0usize;
                for (offset, &bit) in gate_bit_at_offset.iter().enumerate() {
                    gate_p |= ((window_p >> offset) & 1) << bit;
                }
                gate_p
            })
            .collect()
    }

    // ── environments ─────────────────────────────────────────────────

    fn left_environments(&self) -> Vec<Vec<S>> {
        let n = self.tensors.len();
        let mut envs = Vec::with_capacity(n + 1);
        envs.push(vec![S::one()]); // 1×1
        for s in 0..n {
            let t = &self.tensors[s];
            let prev = &envs[s];
            let dim_in = t.left;
            let dim_out = t.right;
            let mut next = vec![S::zero(); dim_out * dim_out];
            for p in 0..2 {
                for a in 0..dim_in {
                    for b in 0..dim_in {
                        let l = prev[a * dim_in + b];
                        if l.abs_sqr() == 0.0 {
                            continue;
                        }
                        for ap in 0..dim_out {
                            let ca = t.at(a, p, ap).conj();
                            if ca.abs_sqr() == 0.0 {
                                continue;
                            }
                            for bp in 0..dim_out {
                                next[ap * dim_out + bp] =
                                    next[ap * dim_out + bp] + ca * l * t.at(b, p, bp);
                            }
                        }
                    }
                }
            }
            envs.push(next);
        }
        envs
    }

    fn right_environments(&self) -> Vec<Vec<S>> {
        let n = self.tensors.len();
        let mut envs = vec![Vec::new(); n + 1];
        envs[n] = vec![S::one()];
        for s in (0..n).rev() {
            let t = &self.tensors[s];
            let after = &envs[s + 1];
            let dim_out = t.right;
            let dim_in = t.left;
            let mut prev = vec![S::zero(); dim_in * dim_in];
            for p in 0..2 {
                for ap in 0..dim_out {
                    for bp in 0..dim_out {
                        let r = after[ap * dim_out + bp];
                        if r.abs_sqr() == 0.0 {
                            continue;
                        }
                        for a in 0..dim_in {
                            let va = t.at(a, p, ap).conj();
                            if va.abs_sqr() == 0.0 {
                                continue;
                            }
                            for b in 0..dim_in {
                                prev[a * dim_in + b] =
                                    prev[a * dim_in + b] + va * r * t.at(b, p, bp);
                            }
                        }
                    }
                }
            }
            envs[s] = prev;
        }
        envs
    }

    /// `(Born weight of bit = 1 at site, total weight)` via environments.
    fn site_marginal(&self, site: usize) -> (f64, f64) {
        let lefts = self.left_environments();
        let rights = self.right_environments();
        let t = &self.tensors[site];
        let left = &lefts[site];
        let right = &rights[site + 1];
        let mut weights = [0.0f64; 2];
        for (p, slot) in weights.iter_mut().enumerate() {
            let mut acc = S::zero();
            for a in 0..t.left {
                for b in 0..t.left {
                    let l = left[a * t.left + b];
                    if l.abs_sqr() == 0.0 {
                        continue;
                    }
                    for ap in 0..t.right {
                        let ca = t.at(a, p, ap).conj();
                        if ca.abs_sqr() == 0.0 {
                            continue;
                        }
                        for bp in 0..t.right {
                            acc = acc + ca * l * t.at(b, p, bp) * right[ap * t.right + bp];
                        }
                    }
                }
            }
            *slot = acc.re();
        }
        (weights[1], weights[0] + weights[1])
    }

    /// Native conditional sampler: `O(shots · n · χ³)` with no support
    /// enumeration — the fast path for wide states. Counts differ from the
    /// enumeration-based trait [`Backend::sample`] for the same seed (the
    /// draws are consumed differently); the distribution is identical.
    pub fn sample_mps(&self, shots: u64, rng: &mut Prng) -> Result<HashMap<u64, u64>> {
        let n = self.tensors.len();
        let rights = self.right_environments();
        let mut counts: HashMap<u64, u64> = HashMap::new();
        for _ in 0..shots {
            let mut index = 0u64;
            let mut cond = vec![S::one()]; // 1×1 conditioned left env
            let mut dim = 1usize;
            for s in 0..n {
                let t = &self.tensors[s];
                let right = &rights[s + 1];
                let mut weights = [0.0f64; 2];
                let mut nexts: [Vec<S>; 2] = [
                    vec![S::zero(); t.right * t.right],
                    vec![S::zero(); t.right * t.right],
                ];
                for p in 0..2 {
                    for a in 0..dim {
                        for b in 0..dim {
                            let l = cond[a * dim + b];
                            if l.abs_sqr() == 0.0 {
                                continue;
                            }
                            for ap in 0..t.right {
                                let ca = t.at(a, p, ap).conj();
                                if ca.abs_sqr() == 0.0 {
                                    continue;
                                }
                                for bp in 0..t.right {
                                    nexts[p][ap * t.right + bp] =
                                        nexts[p][ap * t.right + bp] + ca * l * t.at(b, p, bp);
                                }
                            }
                        }
                    }
                    let mut w = S::zero();
                    for ap in 0..t.right {
                        for bp in 0..t.right {
                            w = w + nexts[p][ap * t.right + bp] * right[ap * t.right + bp];
                        }
                    }
                    weights[p] = w.re();
                }
                let total = weights[0] + weights[1];
                if total <= 0.0 || !total.is_finite() {
                    return Err(Error::InvalidState(format!(
                        "mps sampler: conditional weight {total} is not positive"
                    )));
                }
                let p1 = (weights[1] / total).clamp(0.0, 1.0);
                let bit = rng.next_f64() < p1;
                if bit {
                    index |= 1u64 << self.logical_at_site[s];
                }
                cond = std::mem::take(&mut nexts[bit as usize]);
                // Normalize to keep magnitudes sane over long chains.
                let w = weights[bit as usize];
                if w > 0.0 {
                    let inv = 1.0 / w;
                    for entry in cond.iter_mut() {
                        *entry = entry.scale(inv);
                    }
                }
                dim = t.right;
            }
            *counts.entry(index).or_insert(0) += 1;
        }
        Ok(counts)
    }

    fn dfs_nonzero(&self, site: usize, index: u64, vector: &[S], f: &mut dyn FnMut(u64, S)) {
        if site == self.tensors.len() {
            let amp = vector[0];
            if amp.abs_sqr() > 0.0 {
                f(index, amp);
            }
            return;
        }
        let t = &self.tensors[site];
        for p in 0..2 {
            let mut next = vec![S::zero(); t.right];
            let mut norm = 0.0f64;
            for (r, slot) in next.iter_mut().enumerate() {
                let mut acc = S::zero();
                for (l, &v) in vector.iter().enumerate() {
                    acc = acc + v * t.at(l, p, r);
                }
                norm += acc.abs_sqr();
                *slot = acc;
            }
            if norm <= 1e-28 {
                continue;
            }
            let bit = (p as u64) << self.logical_at_site[site];
            self.dfs_nonzero(site + 1, index | bit, &next, f);
        }
    }
}

impl<S: Scalar> Backend<S> for MpsState<S> {
    fn name(&self) -> &str {
        "mps"
    }

    fn num_qubits(&self) -> usize {
        self.tensors.len()
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        validate_apply(self.num_qubits(), matrix, qubits)?;
        let k = qubits.len();
        if k > MPS_MAX_WINDOW {
            return Err(Error::TooManyQubits {
                requested: k,
                max: MPS_MAX_WINDOW,
            });
        }
        if k == 1 {
            let site = self.site_of_logical[qubits[0]];
            return self.apply_window_sites(site, 1, Some(matrix.data()), None);
        }
        let (start, gate_bits) = self.route_window(qubits)?;
        let map = Self::window_index_map(&gate_bits);
        let d = 1usize << k;
        let mdata = matrix.data();
        let mut permuted = vec![S::zero(); d * d];
        for wr in 0..d {
            for wc in 0..d {
                permuted[wr * d + wc] = mdata[map[wr] * d + map[wc]];
            }
        }
        self.apply_window_sites(start, k, Some(&permuted), None)
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        validate_apply_diagonal(self.num_qubits(), entries, qubits)?;
        let k = qubits.len();
        if k > MPS_MAX_WINDOW {
            return Err(Error::TooManyQubits {
                requested: k,
                max: MPS_MAX_WINDOW,
            });
        }
        if k == 1 {
            let site = self.site_of_logical[qubits[0]];
            return self.apply_window_sites(site, 1, None, Some(entries));
        }
        let (start, gate_bits) = self.route_window(qubits)?;
        let map = Self::window_index_map(&gate_bits);
        let permuted: Vec<S> = (0..entries.len()).map(|wp| entries[map[wp]]).collect();
        self.apply_window_sites(start, k, None, Some(&permuted))
    }

    fn amplitude(&self, index: u64) -> S {
        let mut vector = vec![S::one()];
        for site in 0..self.tensors.len() {
            let t = &self.tensors[site];
            let p = ((index >> self.logical_at_site[site]) & 1) as usize;
            let mut next = vec![S::zero(); t.right];
            for (r, slot) in next.iter_mut().enumerate() {
                let mut acc = S::zero();
                for (l, &v) in vector.iter().enumerate() {
                    acc = acc + v * t.at(l, p, r);
                }
                *slot = acc;
            }
            vector = next;
        }
        vector[0]
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        self.dfs_nonzero(0, 0, &[S::one()], f);
    }

    fn total_weight(&self) -> f64 {
        let lefts = self.left_environments();
        lefts[self.tensors.len()][0].re()
    }

    fn measure(&mut self, qubit: usize, rng: &mut Prng) -> Result<bool> {
        if qubit >= self.num_qubits() {
            return Err(Error::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits(),
            });
        }
        let site = self.site_of_logical[qubit];
        let (one, total) = self.site_marginal(site);
        if total <= 0.0 || !total.is_finite() {
            return Err(Error::InvalidState(format!(
                "total Born weight {total} is not positive; cannot measure"
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
        let site = self.site_of_logical[qubit];
        let t = &mut self.tensors[site];
        let keep = outcome as usize;
        for l in 0..t.left {
            for p in 0..2 {
                for r in 0..t.right {
                    let idx = (l * 2 + p) * t.right + r;
                    if p == keep {
                        t.data[idx] = t.data[idx].scale(renorm);
                    } else {
                        t.data[idx] = S::zero();
                    }
                }
            }
        }
    }

    fn reset(&mut self) {
        let n = self.tensors.len();
        self.routing_swaps = 0;
        self.peak_bond = 0;
        self.peak_elements = 0;
        self.reset_chain(n);
    }

    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        let n = self.tensors.len();
        if n > MPS_LOAD_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: n,
                max: MPS_LOAD_MAX_QUBITS,
            });
        }
        let limit = 1u64 << n;
        for &(i, _) in entries {
            if i >= limit {
                return Err(Error::QubitOutOfRange {
                    qubit: 64 - i.leading_zeros() as usize,
                    num_qubits: n,
                });
            }
        }
        // Identity layout, then compile the dense vector into the chain.
        self.site_of_logical = (0..n).collect();
        self.logical_at_site = (0..n).collect();
        let mut theta = crate::guard::try_vec(
            1usize << n,
            S::zero(),
            &format!("mps load compilation ({n} qubits)"),
        )?;
        for &(i, a) in entries {
            theta[i as usize] = a;
        }
        if n == 0 {
            return Ok(());
        }
        if n == 1 {
            self.tensors[0] = SiteTensor {
                left: 1,
                right: 1,
                data: theta,
            };
            self.note_peaks();
            return Ok(());
        }
        self.split_window(0, n, 1, 1, theta)
    }

    fn memory_bytes(&self) -> usize {
        let elements: usize = self.tensors.iter().map(|t| t.data.capacity()).sum();
        elements * std::mem::size_of::<S>()
            + 2 * self.site_of_logical.capacity() * std::mem::size_of::<usize>()
            + self.tensors.capacity() * std::mem::size_of::<SiteTensor<S>>()
            + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
