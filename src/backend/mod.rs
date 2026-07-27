//! State backends: swappable representations of an `n`-qubit state.
//!
//! A [`Backend`] owns amplitudes over the algebra `S` and knows how to apply
//! gate matrices, project on measurement outcomes, and report its memory
//! footprint. The crate ships three:
//!
//! * [`DenseState`] — flat `2^n` amplitude vector; the BQP reference
//!   implementation,
//! * [`SparseState`] — hash map of nonzero amplitudes; exponential savings
//!   whenever the state stays concentrated (stabilizer-like circuits, GHZ,
//!   oracles),
//! * [`AdaptiveState`] — starts sparse, promotes itself to dense when the
//!   state's density makes dense cheaper. A first example of a "research"
//!   backend built from the same trait.
//!
//! Additional representations (matrix-product states are on the roadmap)
//! implement the same trait and can be registered by name in a
//! [`BackendRegistry`], exactly like research gates in the gate registry.

mod adaptive;
mod dense;
mod sparse;

pub use adaptive::AdaptiveState;
pub use dense::DenseState;
pub use sparse::SparseState;

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::gates::Pauli;
use crate::math::GateMatrix;
use crate::rng::Prng;
use crate::scalar::{Scalar, C64};

/// Amplitudes with squared magnitude at or below this are pruned by sparse
/// representations after each gate.
pub const PRUNE_TOL: f64 = 1e-24;

/// A state representation over the amplitude algebra `S`.
///
/// Object-safe: backends are used as `Box<dyn Backend<S>>` so they can be
/// selected by name at runtime. Measurement and sampling are provided
/// methods built on the required primitives — a representation with a
/// smarter native strategy (e.g. MPS conditional sampling) can override
/// them.
pub trait Backend<S: Scalar> {
    /// Short backend name (e.g. `"dense"`).
    fn name(&self) -> &str;
    /// Register width.
    fn num_qubits(&self) -> usize;
    /// Apply a `k`-qubit gate matrix to the given qubits
    /// (`qubits[b]` ↔ matrix sub-index bit `b`).
    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()>;
    /// Amplitude of a basis state.
    fn amplitude(&self, index: u64) -> S;
    /// Visit every stored nonzero amplitude as `(basis_index, amplitude)`.
    /// Dense backends visit all non-vanishing entries in index order; sparse
    /// backends visit in arbitrary order.
    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S));
    /// Number of stored nonzero amplitudes.
    fn nonzero_count(&self) -> usize {
        let mut count = 0usize;
        self.for_each_nonzero(&mut |_, _| count += 1);
        count
    }
    /// Project qubit `q` onto `outcome`, zeroing the other branch and
    /// multiplying surviving amplitudes by the real factor `renorm`.
    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64);
    /// Reset to `|0…0⟩`.
    fn reset(&mut self);
    /// Replace the state with the given amplitudes (unlisted indices become
    /// zero). Indices are bounds-checked; normalization is the caller's
    /// choice — measurement copes with unnormalized states.
    fn load(&mut self, entries: &[(u64, S)]) -> Result<()>;
    /// Estimated heap + struct memory in bytes for the current state.
    fn memory_bytes(&self) -> usize;
    /// Downcast support for backend-specific inspection.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Total Born weight `Σ born_weight(amp)`. Equals the squared norm for
    /// division algebras; may be anything for exotic scalars.
    fn total_weight(&self) -> f64 {
        let mut total = 0.0;
        self.for_each_nonzero(&mut |_, a| total += a.born_weight());
        total
    }

    /// Sum of `abs_sqr` over amplitudes — the Euclidean squared norm of the
    /// coordinate vector, always ≥ 0.
    fn total_abs_sqr(&self) -> f64 {
        let mut total = 0.0;
        self.for_each_nonzero(&mut |_, a| total += a.abs_sqr());
        total
    }

    /// Born weight of one basis state.
    fn probability(&self, index: u64) -> f64 {
        self.amplitude(index).born_weight()
    }

    /// All nonzero `(index, born_weight)` pairs, sorted by index.
    fn probabilities(&self) -> Vec<(u64, f64)> {
        let mut probs = Vec::new();
        self.for_each_nonzero(&mut |i, a| probs.push((i, a.born_weight())));
        probs.sort_unstable_by_key(|&(i, _)| i);
        probs
    }

    /// Measure qubit `q` in the computational basis: draws an outcome from
    /// the (generalized) Born distribution, collapses the state, and
    /// renormalizes. Deterministically reproducible via the seeded [`Prng`].
    ///
    /// Fails with [`Error::InvalidState`] if the total Born weight is not
    /// positive (possible for non-division algebras).
    fn measure(&mut self, qubit: usize, rng: &mut Prng) -> Result<bool> {
        if qubit >= self.num_qubits() {
            return Err(Error::QubitOutOfRange { qubit, num_qubits: self.num_qubits() });
        }
        let mut total = 0.0;
        let mut one_weight = 0.0;
        let mask = 1u64 << qubit;
        self.for_each_nonzero(&mut |i, a| {
            let w = a.born_weight();
            total += w;
            if i & mask != 0 {
                one_weight += w;
            }
        });
        if !(total > 0.0) || !total.is_finite() {
            return Err(Error::InvalidState(format!(
                "total Born weight {total} is not positive; cannot sample a measurement"
            )));
        }
        let p_one = (one_weight / total).clamp(0.0, 1.0);
        let outcome = rng.next_f64() < p_one;
        let p_outcome = if outcome { p_one } else { 1.0 - p_one };
        // Branch weight = p_outcome * total; rescale it to 1.
        let renorm = 1.0 / (p_outcome * total).sqrt();
        self.project(qubit, outcome, renorm);
        Ok(outcome)
    }

    /// Draw `shots` full-register samples from the Born distribution without
    /// disturbing the state. Returns `basis_index → count`.
    ///
    /// Weights are accumulated in basis-index order, so results for a given
    /// seed agree across backends of every representation.
    fn sample(&self, shots: u64, rng: &mut Prng) -> Result<HashMap<u64, u64>> {
        let mut entries = Vec::new();
        self.for_each_nonzero(&mut |i, a| {
            let w = a.born_weight();
            if w > 0.0 {
                entries.push((i, w));
            }
        });
        entries.sort_unstable_by_key(|&(i, _)| i);
        let mut cumulative = 0.0;
        for e in &mut entries {
            cumulative += e.1;
            e.1 = cumulative;
        }
        if !(cumulative > 0.0) || !cumulative.is_finite() {
            return Err(Error::InvalidState(format!(
                "total Born weight {cumulative} is not positive; cannot sample"
            )));
        }
        let mut counts: HashMap<u64, u64> = HashMap::new();
        for _ in 0..shots {
            let u = rng.next_f64() * cumulative;
            let pos = entries.partition_point(|&(_, c)| c <= u).min(entries.len() - 1);
            *counts.entry(entries[pos].0).or_insert(0) += 1;
        }
        Ok(counts)
    }
}

/// Expectation value `⟨ψ| P |ψ⟩` of a Pauli string, without modifying or
/// copying the state. `ops` lists `(qubit, Pauli)` pairs; unlisted qubits
/// are implicitly identity. Returns a scalar; for division algebras with a
/// normalized state its `re()` is the physical expectation value.
///
/// Inner-product convention: `⟨φ|ψ⟩ = Σ conj(φ_i) ψ_i`, products taken
/// left-to-right (relevant for non-commutative and non-associative `S`).
pub fn pauli_expectation<S: Scalar>(
    state: &dyn Backend<S>,
    ops: &[(usize, Pauli)],
) -> Result<S> {
    let n = state.num_qubits();
    let qubits: Vec<usize> = ops.iter().map(|&(q, _)| q).collect();
    crate::circuit::validate_targets(n, &qubits)?;

    // P|i⟩ = phase · |i ^ flip_mask⟩ with a per-index complex phase.
    let mut err = None;
    let mut acc = S::zero();
    state.for_each_nonzero(&mut |i, a| {
        if err.is_some() {
            return;
        }
        let mut target = i;
        let mut phase = C64::new(1.0, 0.0);
        for &(q, p) in ops {
            let bit = (i >> q) & 1 == 1;
            match p {
                Pauli::I => {}
                Pauli::X => target ^= 1 << q,
                Pauli::Y => {
                    target ^= 1 << q;
                    // Y|0⟩ = i|1⟩, Y|1⟩ = −i|0⟩.
                    phase *= if bit { C64::new(0.0, -1.0) } else { C64::new(0.0, 1.0) };
                }
                Pauli::Z => {
                    if bit {
                        phase = -phase;
                    }
                }
            }
        }
        match S::try_from_c64(phase) {
            Some(ph) => {
                // conj(ψ[target]) · phase · ψ[i], left-to-right.
                acc = acc + state.amplitude(target).conj() * ph * a;
            }
            None => {
                err = Some(Error::UnsupportedForAlgebra {
                    gate: "pauli_expectation(Y)".to_string(),
                    algebra: S::algebra_name(),
                })
            }
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(acc),
    }
}

type BackendCtor<S> = Box<dyn Fn(usize) -> Result<Box<dyn Backend<S>>> + Send + Sync>;

/// A name → constructor catalog of state representations, mirroring the
/// gate registry: research backends register here and become selectable by
/// name through the [`Simulator`](crate::sim::Simulator).
pub struct BackendRegistry<S: Scalar = C64> {
    ctors: HashMap<String, BackendCtor<S>>,
}

impl<S: Scalar> Default for BackendRegistry<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> BackendRegistry<S> {
    /// An empty registry.
    pub fn new() -> Self {
        BackendRegistry { ctors: HashMap::new() }
    }

    /// A registry with the built-in `"dense"`, `"sparse"` and `"adaptive"`
    /// backends.
    pub fn standard() -> Self {
        let mut reg = Self::new();
        reg.register("dense", |n| Ok(Box::new(DenseState::<S>::new(n)?)))
            .expect("fresh registry");
        reg.register("sparse", |n| Ok(Box::new(SparseState::<S>::new(n)?)))
            .expect("fresh registry");
        reg.register("adaptive", |n| Ok(Box::new(AdaptiveState::<S>::new(n)?)))
            .expect("fresh registry");
        reg
    }

    /// Register a backend constructor under a new name.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        ctor: impl Fn(usize) -> Result<Box<dyn Backend<S>>> + Send + Sync + 'static,
    ) -> Result<()> {
        let name = name.into();
        if self.ctors.contains_key(&name) {
            return Err(Error::DuplicateBackend(name));
        }
        self.ctors.insert(name, Box::new(ctor));
        Ok(())
    }

    /// Construct a backend by name.
    pub fn create(&self, name: &str, num_qubits: usize) -> Result<Box<dyn Backend<S>>> {
        match self.ctors.get(name) {
            Some(ctor) => ctor(num_qubits),
            None => Err(Error::UnknownBackend(name.to_string())),
        }
    }

    /// Registered backend names, sorted.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.ctors.keys().cloned().collect();
        names.sort();
        names
    }

    /// Whether a backend with this name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.ctors.contains_key(name)
    }
}

/// Shared validation for gate application: matrix dimension vs target count,
/// and target validity vs register width.
pub(crate) fn validate_apply<S: Scalar>(
    num_qubits: usize,
    matrix: &GateMatrix<S>,
    qubits: &[usize],
) -> Result<()> {
    crate::circuit::validate_targets(num_qubits, qubits)?;
    let expected = 1usize
        .checked_shl(qubits.len() as u32)
        .ok_or(Error::TooManyQubits { requested: qubits.len(), max: 63 })?;
    if matrix.dim() != expected {
        return Err(Error::BadDimension { expected, got: matrix.dim() });
    }
    Ok(())
}

/// Expand `group` (an index over non-target bit patterns) into a full basis
/// index with zeros at the sorted target positions.
#[inline]
pub(crate) fn expand_index(group: u64, sorted_targets: &[usize]) -> u64 {
    let mut idx = group;
    for &t in sorted_targets {
        let low = idx & ((1u64 << t) - 1);
        idx = ((idx >> t) << (t + 1)) | low;
    }
    idx
}

/// For each sub-index `j` of a `k`-qubit gate, the basis-index offset formed
/// by scattering `j`'s bits to the target qubit positions (original order).
#[inline]
pub(crate) fn scatter_table(qubits: &[usize]) -> Vec<u64> {
    let d = 1usize << qubits.len();
    (0..d as u64)
        .map(|j| {
            qubits
                .iter()
                .enumerate()
                .fold(0u64, |acc, (b, &q)| acc | (((j >> b) & 1) << q))
        })
        .collect()
}
