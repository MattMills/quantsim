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
mod braided_state;
mod bulk;
mod clifford_frame;
mod dense;
mod device;
mod factored;
mod frames;
mod interference;
mod mera;
mod mosaic;
mod mps;
mod phase_field;
mod radix;
mod sparse;

pub use adaptive::AdaptiveState;
pub use braided_state::{BraidedState, Realization};
pub use bulk::{
    BulkConfig, BulkState, UnfoldProgram, UnfoldStep, BULK_MAX_QUBITS, UNFOLD_MAX_STEP_QUBITS,
};
pub use clifford_frame::{
    conjugate_by_step, CliffordFrameStats, CliffordFramedState, CliffordStep, PauliString,
    CLIFFORD_DIAGONAL_MAX, CLIFFORD_RECOGNITION_MAX,
};
pub use dense::{DenseState, DENSE_MAX_QUBITS};
pub use device::{ArityPolicy, DeviceState, DurationModel, LatencyMap, PhysicalOp, Topology};
pub use factored::{FactoredState, FACTORED_MAX_QUBITS, FACTOR_MAX_QUBITS};
pub use frames::{FrameStats, FramedState, FRAME_CONJUGATION_MAX};
pub use interference::{InterferenceRecord, InterferenceState};
pub use mera::{MeraConfig, MeraState, MERA_LOAD_MAX_QUBITS, MERA_MAX_QUBITS};
pub use mosaic::{MosaicEvent, MosaicPolicy, MosaicState, MOSAIC_MAX_QUBITS};
pub use mps::{MpsConfig, MpsState, MPS_LOAD_MAX_QUBITS, MPS_MAX_QUBITS, MPS_MAX_WINDOW};
pub use phase_field::{PhaseFieldState, MAX_DYADIC_ROOT_DEPTH, MAX_FIELD_MODULUS, MAX_ROOT_ORDER};
pub use radix::{RadixDenseState, RadixSparseState};
pub use sparse::{SparseState, SPARSE_MAX_QUBITS};

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::gates::Pauli;
use crate::math::GateMatrix;
use crate::rng::Prng;
use crate::scalar::{Scalar, C64};
use crate::space::{require_binary, validate_sites, SiteMatrix, Space, Transition};

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
    /// Apply a diagonal unitary given as its `2^k` diagonal entries, indexed
    /// by little-endian sub-index over `qubits`. Shipped backends run this
    /// in `O(states)` with `O(2^k)` storage; the default implementation
    /// materializes the full matrix and calls [`Backend::apply`], so custom
    /// backends work unmodified but should override for large `k`.
    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        let d = entries.len();
        if !d.is_power_of_two() || d != 1usize << qubits.len() {
            return Err(Error::BadDimension {
                expected: 1usize << qubits.len(),
                got: d,
            });
        }
        let mut m = GateMatrix::zeros(d)?;
        for (i, &e) in entries.iter().enumerate() {
            m.set(i, i, e);
        }
        self.apply(&m, qubits)
    }
    /// The register's state space (see [`crate::space`]). The default is
    /// `num_qubits()` qubits; a backend that stores mixed-radix registers
    /// overrides it, and then `num_qubits()` is its number of **sites**
    /// and basis indices are little-endian mixed radix.
    fn space(&self) -> Space {
        Space::qubits(self.num_qubits())
    }
    /// Apply a unitary over the mixed-radix sub-index of `sites`
    /// (`sites[0]` is the low digit; the matrix dimension is the product
    /// of their radices).
    ///
    /// The default serves qubit-only backends: every target must have
    /// radix 2, and the matrix goes through [`Backend::apply`]. A
    /// non-binary target is refused with [`Error::UnsupportedForSpace`] —
    /// never reinterpreted as bits.
    fn apply_sites(&mut self, matrix: &SiteMatrix<S>, sites: &[usize]) -> Result<()> {
        let space = self.space();
        require_binary(&space, sites, "apply_sites")?;
        validate_sites(&space, matrix, sites)?;
        self.apply(&matrix.to_gate()?, sites)
    }
    /// Project site `site` onto `digit`, zeroing every other branch and
    /// multiplying surviving amplitudes by `renorm`. The default handles
    /// radix-2 sites through [`Backend::project`] and refuses the rest.
    fn project_digit(&mut self, site: usize, digit: usize, renorm: f64) -> Result<()> {
        let space = self.space();
        require_binary(&space, &[site], "project_digit")?;
        if digit > 1 {
            return Err(Error::InvalidState(format!(
                "digit {digit} out of range for site {site} of radix 2"
            )));
        }
        self.project(site, digit == 1, renorm);
        Ok(())
    }
    /// Change the register's space by one [`Transition`], returning the
    /// weight (summed `abs_sqr`) it removed. A release or demote that would
    /// remove weight above [`PRUNE_TOL`] is refused with
    /// [`Error::WouldDiscard`] — state untouched — unless `allow_discard`,
    /// in which case the amplitude is dropped and its weight returned,
    /// never silently and never renormalized.
    ///
    /// The default refuses every transition with
    /// [`Error::UnsupportedForSpace`]: a backend whose shape is fixed at
    /// construction cannot change it. Native mixed-radix stores override
    /// it.
    fn transition(&mut self, t: &Transition, allow_discard: bool) -> Result<f64> {
        let _ = allow_discard;
        Err(Error::UnsupportedForSpace {
            operation: format!("space transition ({t}) on backend '{}'", self.name()),
            space: self.space().to_string(),
        })
    }
    /// Measure one site in its computational basis, collapsing and
    /// renormalizing the state: the site-level [`Backend::measure`].
    /// A radix-2 site is measured by [`Backend::measure`] itself, so qubit
    /// results are seed-for-seed identical; a wider site draws its digit
    /// from the cumulative Born weights in digit order.
    fn measure_site(&mut self, site: usize, rng: &mut Prng) -> Result<usize> {
        let space = self.space();
        space.validate_sites(&[site])?;
        let r = space.radix(site);
        if r == 2 {
            return self.measure(site, rng).map(usize::from);
        }
        let mut weights = vec![0.0f64; r];
        self.for_each_nonzero(&mut |i, a| weights[space.digit(i, site)] += a.born_weight());
        let total: f64 = weights.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return Err(Error::InvalidState(format!(
                "total Born weight {total} is not positive; cannot sample a measurement"
            )));
        }
        let u = rng.next_f64() * total;
        let mut cumulative = 0.0;
        let mut digit = (0..r).rev().find(|&d| weights[d] > 0.0).unwrap_or(0);
        for (d, &w) in weights.iter().enumerate() {
            cumulative += w;
            if w > 0.0 && u < cumulative {
                digit = d;
                break;
            }
        }
        self.project_digit(site, digit, 1.0 / weights[digit].sqrt())?;
        Ok(digit)
    }
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
    /// Apply a basis permutation in place: each occupied index moves to
    /// `map(index)`, amplitudes unchanged.
    ///
    /// The default collects the support, rewrites it and reloads — three
    /// passes and two hash tables. A representation that stores its
    /// support directly can do it in one, and
    /// [`SparseState`] does; the measured profile of
    /// [`shor`](crate::shor) is dominated by this call, which is why it
    /// is a trait method with an override rather than a free function.
    ///
    /// A non-injective `map` is not unitary and is reported as
    /// [`Error::InvalidState`] rather than silently applied.
    fn apply_permutation(&mut self, label: &str, map: &dyn Fn(u64) -> u64) -> Result<()> {
        let space = self.space();
        let mut entries: Vec<(u64, S)> = Vec::with_capacity(self.nonzero_count());
        self.for_each_nonzero(&mut |i, a| entries.push((i, a)));
        let mut seen: HashMap<u64, u64> = HashMap::with_capacity(entries.len());
        let mut out: Vec<(u64, S)> = Vec::with_capacity(entries.len());
        for (step, &(i, a)) in entries.iter().enumerate() {
            if step % 4096 == 0 {
                crate::guard::checkpoint()?;
            }
            let j = map(i);
            if j as u128 >= space.dim() {
                return Err(Error::InvalidState(format!(
                    "{label} sent index {i} to {j}, outside the register of {space}"
                )));
            }
            if let Some(prev) = seen.insert(j, i) {
                return Err(Error::InvalidState(format!(
                    "{label} is not injective: indices {prev} and {i} both map to {j}"
                )));
            }
            out.push((j, a));
        }
        self.load(&out)
    }

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
            return Err(Error::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits(),
            });
        }
        let space = self.space();
        if !space.is_qubits() {
            require_binary(&space, &[qubit], "measure")?;
        }
        let mut total = 0.0;
        let mut one_weight = 0.0;
        let mask = 1u64 << qubit;
        let qubits = space.is_qubits();
        self.for_each_nonzero(&mut |i, a| {
            let w = a.born_weight();
            total += w;
            let one = if qubits {
                i & mask != 0
            } else {
                space.digit(i, qubit) == 1
            };
            if one {
                one_weight += w;
            }
        });
        if total <= 0.0 || !total.is_finite() {
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
        if cumulative <= 0.0 || !cumulative.is_finite() {
            return Err(Error::InvalidState(format!(
                "total Born weight {cumulative} is not positive; cannot sample"
            )));
        }
        let mut counts: HashMap<u64, u64> = HashMap::new();
        for _ in 0..shots {
            let u = rng.next_f64() * cumulative;
            let pos = entries
                .partition_point(|&(_, c)| c <= u)
                .min(entries.len() - 1);
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
pub fn pauli_expectation<S: Scalar>(state: &dyn Backend<S>, ops: &[(usize, Pauli)]) -> Result<S> {
    let space = state.space();
    if !space.is_qubits() {
        return Err(Error::UnsupportedForSpace {
            operation: "pauli_expectation".into(),
            space: space.to_string(),
        });
    }
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
                    phase *= if bit {
                        C64::new(0.0, -1.0)
                    } else {
                        C64::new(0.0, 1.0)
                    };
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

/// Largest entrywise amplitude distance between two states, scanned over
/// the union of their supports — `O(nnz)` on both sides, so it works at
/// widths where a full `2^n` sweep does not. The workhorse of backend
/// conformance checking.
pub fn max_amplitude_deviation<S: Scalar>(a: &dyn Backend<S>, b: &dyn Backend<S>) -> f64 {
    let mut dev = 0.0f64;
    a.for_each_nonzero(&mut |i, x| {
        dev = dev.max((x - b.amplitude(i)).abs_sqr().sqrt());
    });
    b.for_each_nonzero(&mut |i, x| {
        dev = dev.max((x - a.amplitude(i)).abs_sqr().sqrt());
    });
    dev
}

type BackendCtor<S> = Box<dyn Fn(usize) -> Result<Box<dyn Backend<S>>> + Send + Sync>;
type SpaceCtor<S> = Box<dyn Fn(Space) -> Result<Box<dyn Backend<S>>> + Send + Sync>;

/// A name → constructor catalog of state representations, mirroring the
/// gate registry: research backends register here and become selectable by
/// name through the [`Simulator`](crate::sim::Simulator).
pub struct BackendRegistry<S: Scalar = C64> {
    ctors: HashMap<String, BackendCtor<S>>,
    space_ctors: HashMap<String, SpaceCtor<S>>,
}

impl<S: Scalar> Default for BackendRegistry<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> BackendRegistry<S> {
    /// An empty registry.
    pub fn new() -> Self {
        BackendRegistry {
            ctors: HashMap::new(),
            space_ctors: HashMap::new(),
        }
    }

    /// A registry with the built-in `"dense"`, `"sparse"`, `"adaptive"`,
    /// `"factored"`, `"mps"`, `"mera"`, `"bulk"`, `"mosaic"`, `"bundle"`
    /// and `"branched"` backends.
    /// (`"bundle"` is the graph-state
    /// [`PolarityBundle`](crate::bundle::PolarityBundle): exact and
    /// `O(n + |E|)` on the Clifford sector, refusing anything else by
    /// name.) (`"branched"` is the
    /// [`BranchedRegister`](crate::clock::BranchedRegister) with a
    /// trivial selector over one **mosaic** branch: representation
    /// election below, selector dimension above — the flat Backend
    /// contract holds either way.) (`"mps"`, `"mera"` and `"bulk"`
    /// require a commutative division algebra and report an error at
    /// creation elsewhere.)
    pub fn standard() -> Self {
        let mut reg = Self::new();
        reg.register("dense", |n| Ok(Box::new(DenseState::<S>::new(n)?)))
            .expect("fresh registry");
        reg.register("sparse", |n| Ok(Box::new(SparseState::<S>::new(n)?)))
            .expect("fresh registry");
        reg.register_space("dense", |space| {
            Ok(Box::new(RadixDenseState::<S>::new(space)?))
        })
        .expect("fresh registry");
        reg.register_space("sparse", |space| {
            Ok(Box::new(RadixSparseState::<S>::new(space)?))
        })
        .expect("fresh registry");
        reg.register("adaptive", |n| Ok(Box::new(AdaptiveState::<S>::new(n)?)))
            .expect("fresh registry");
        reg.register("factored", |n| Ok(Box::new(FactoredState::<S>::new(n)?)))
            .expect("fresh registry");
        reg.register("mps", |n| Ok(Box::new(MpsState::<S>::new(n)?)))
            .expect("fresh registry");
        reg.register("mera", |n| Ok(Box::new(MeraState::<S>::new(n)?)))
            .expect("fresh registry");
        reg.register("bulk", |n| Ok(Box::new(BulkState::<S>::new(n)?)))
            .expect("fresh registry");
        reg.register("mosaic", |n| Ok(Box::new(MosaicState::<S>::new(n)?)))
            .expect("fresh registry");
        reg.register("bundle", |n| {
            let mut bundle = crate::bundle::PolarityBundle::new(n)?;
            // The Backend contract starts at |0…0⟩; the bundle's own
            // description convention starts at |+…+⟩.
            <crate::bundle::PolarityBundle as Backend<S>>::reset(&mut bundle);
            Ok(Box::new(bundle) as Box<dyn Backend<S>>)
        })
        .expect("fresh registry");
        reg.register("branched", |n| {
            Ok(Box::new(crate::clock::BranchedRegister::<S>::new(n, 1)?))
        })
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

    /// Register a constructor for mixed-radix spaces under a backend name
    /// (the name may also have a qubit constructor; the two are separate
    /// tables). Only backends that store non-binary sites natively should
    /// register here.
    pub fn register_space(
        &mut self,
        name: impl Into<String>,
        ctor: impl Fn(Space) -> Result<Box<dyn Backend<S>>> + Send + Sync + 'static,
    ) -> Result<()> {
        let name = name.into();
        if self.space_ctors.contains_key(&name) {
            return Err(Error::DuplicateBackend(name));
        }
        self.space_ctors.insert(name, Box::new(ctor));
        Ok(())
    }

    /// Instantiate a backend over `space`. A qubit space goes to the
    /// ordinary constructor, so every backend accepts it; a mixed-radix
    /// space needs a [`register_space`](Self::register_space) entry, and a
    /// backend without one is refused with [`Error::UnsupportedForSpace`]
    /// rather than handed a register it would read as bits.
    pub fn create_with_space(&self, name: &str, space: &Space) -> Result<Box<dyn Backend<S>>> {
        if space.is_qubits() {
            return self.create(name, space.sites());
        }
        self.create_native(name, space)
    }

    /// Instantiate the named backend's native mixed-radix store over
    /// `space`, **even a qubit space**: what a register whose space will
    /// change needs, since only native stores implement
    /// [`Backend::transition`]. A backend without a
    /// [`register_space`](Self::register_space) entry is refused with
    /// [`Error::UnsupportedForSpace`].
    pub fn create_native(&self, name: &str, space: &Space) -> Result<Box<dyn Backend<S>>> {
        match self.space_ctors.get(name) {
            Some(ctor) => ctor(space.clone()),
            None if self.ctors.contains_key(name) => Err(Error::UnsupportedForSpace {
                operation: format!("backend '{name}' (no native mixed-radix store)"),
                space: space.to_string(),
            }),
            None => Err(Error::UnknownBackend(name.to_string())),
        }
    }

    /// Names of backends that accept mixed-radix spaces, sorted.
    pub fn space_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.space_ctors.keys().cloned().collect();
        names.sort();
        names
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

impl BackendRegistry<C64> {
    /// Add the representations that exist **only** over ℂ, because they
    /// store complex amplitudes natively rather than being generic over
    /// the algebra: the E8 scale tower (`"e8-constellation"`) and the
    /// single-copy E8×E8 register (`"e8-rep"`).
    ///
    /// They cannot live in [`standard`](BackendRegistry::standard),
    /// which is generic over every [`Scalar`], so they are opt-in by
    /// this call — and that is a statement about them, not an oversight.
    pub fn register_e8(&mut self) -> Result<()> {
        self.register("e8-constellation", |n| {
            Ok(Box::new(
                crate::e8::constellation::E8ConstellationState::new(n)?,
            ))
        })?;
        self.register("e8-rep", |n| {
            Ok(Box::new(crate::e8::rep::E8RepState::new(n)?))
        })
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
        .ok_or(Error::TooManyQubits {
            requested: qubits.len(),
            max: 63,
        })?;
    if matrix.dim() != expected {
        return Err(Error::BadDimension {
            expected,
            got: matrix.dim(),
        });
    }
    Ok(())
}

/// Shared validation for diagonal application.
pub(crate) fn validate_apply_diagonal<S: Scalar>(
    num_qubits: usize,
    entries: &[S],
    qubits: &[usize],
) -> Result<()> {
    crate::circuit::validate_targets(num_qubits, qubits)?;
    let expected = 1usize
        .checked_shl(qubits.len() as u32)
        .ok_or(Error::TooManyQubits {
            requested: qubits.len(),
            max: 63,
        })?;
    if entries.len() != expected {
        return Err(Error::BadDimension {
            expected,
            got: entries.len(),
        });
    }
    Ok(())
}

/// Extract the little-endian sub-index of `index` over `qubits`.
#[inline]
pub(crate) fn sub_index(index: u64, qubits: &[usize]) -> usize {
    let mut sub = 0usize;
    for (b, &q) in qubits.iter().enumerate() {
        sub |= (((index >> q) & 1) as usize) << b;
    }
    sub
}

// In-place gate kernels over a flat amplitude slice, shared by the dense and
// factored backends (the factored backend calls them with factor-local qubit
// positions). All checkpoint the resource guard's deadline at coarse
// intervals so an over-scale sweep aborts mid-gate instead of running
// unbounded.

/// Deadline-checkpoint stride inside amplitude sweeps (in loop
/// iterations): coarse enough to be free, fine enough that a single
/// over-budget gate is caught within a fraction of a second.
const CHECK_STRIDE: usize = 1 << 20;

pub(crate) fn apply_single_in_place<S: Scalar>(
    amps: &mut [S],
    m: &GateMatrix<S>,
    q: usize,
) -> Result<()> {
    let (m00, m01, m10, m11) = (m.get(0, 0), m.get(0, 1), m.get(1, 0), m.get(1, 1));
    let mask = 1usize << q;
    let half = amps.len() >> 1;
    for i in 0..half {
        if i % CHECK_STRIDE == 0 {
            crate::guard::checkpoint()?;
        }
        let low = i & (mask - 1);
        let i0 = ((i >> q) << (q + 1)) | low;
        let i1 = i0 | mask;
        let a0 = amps[i0];
        let a1 = amps[i1];
        amps[i0] = m00 * a0 + m01 * a1;
        amps[i1] = m10 * a0 + m11 * a1;
    }
    Ok(())
}

pub(crate) fn apply_general_in_place<S: Scalar>(
    amps: &mut [S],
    m: &GateMatrix<S>,
    qubits: &[usize],
) -> Result<()> {
    let k = qubits.len();
    let d = 1usize << k;
    let mut sorted = qubits.to_vec();
    sorted.sort_unstable();
    let scatter = scatter_table(qubits);
    let groups = amps.len() >> k;
    let mdata = m.data();
    let mut scratch = vec![S::zero(); d];
    for g in 0..groups {
        if g % CHECK_STRIDE == 0 {
            crate::guard::checkpoint()?;
        }
        let base = expand_index(g as u64, &sorted);
        for (j, slot) in scratch.iter_mut().enumerate() {
            *slot = amps[(base | scatter[j]) as usize];
        }
        for r in 0..d {
            let mut acc = S::zero();
            for (l, &s) in scratch.iter().enumerate() {
                acc = acc + mdata[r * d + l] * s;
            }
            amps[(base | scatter[r]) as usize] = acc;
        }
    }
    Ok(())
}

pub(crate) fn apply_diagonal_in_place<S: Scalar>(
    amps: &mut [S],
    entries: &[S],
    qubits: &[usize],
) -> Result<()> {
    for (i, a) in amps.iter_mut().enumerate() {
        if i % CHECK_STRIDE == 0 {
            crate::guard::checkpoint()?;
        }
        *a = entries[sub_index(i as u64, qubits)] * *a;
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
