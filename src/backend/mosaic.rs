//! The multi-representation register: every portion of the system in
//! the representation ideal for **its** portion of the circuit-problem
//! axis.
//!
//! Every representation in this crate is a *factorization lens* — an
//! attribute it factors out of the state, and a cost that is
//! exponential exactly when that attribute grows:
//!
//! | representation | what it factors                    |
//! |----------------|------------------------------------|
//! | sparse         | basis-alignment (support)          |
//! | factored       | spatial product structure          |
//! | mps            | linear cut rank (bonds)            |
//! | mera / bulk    | hierarchical cut rank              |
//! | bundle         | stabilizer group structure         |
//! | clifford-framed| Clifford conjugation (magic count) |
//! | phase-field    | diagonal phase polynomials         |
//! | branched       | class rank across slices           |
//!
//! A fixed choice applies one lens to the whole register for the whole
//! circuit. A [`MosaicState`] instead partitions the boundary into
//! **regions**, each held by its own [`Backend`] — one region a graph
//! state in the bundle, its neighbour a T-doped block in sparse, a
//! third in MPS — and lets the assignment follow the circuit:
//!
//! * **Gates inside a region** run natively in that region's
//!   representation.
//! * **Gates across regions** merge the regions (the
//!   [`FactoredState`](super::FactoredState) geometry, generalized:
//!   factors are no longer dense blocks but arbitrary backends), with
//!   the merged representation **chosen by measurement** — predicted
//!   sparse cost (product of supports) against predicted dense cost
//!   (`2^w`), the choice and both predictions ledgered so
//!   mispredictions are data, not surprises.
//! * **Migration is refusal-driven, and partial before total.** A
//!   representation that cannot apply a gate refuses by name — the
//!   bundle on its first T — and the mosaic treats the refusal as the
//!   measurement that the region's structural era ended. Before
//!   converting anything it asks the region for structure it carries
//!   for free: a bundle region whose graph has several components is
//!   **split into one bundle region per component** (exact — a graph
//!   state is the product of its components), so only the component
//!   the gate actually touched migrates and the rest stay graphs: the
//!   graph, and the graph of graphs. Only then does the affected
//!   region convert down the policy's candidate list and retry. Era
//!   boundaries in the *circuit* axis become representation changes in
//!   the *register*, automatically, and every split and migration is
//!   ledgered with its cause. (Regions may themselves be mosaics — a
//!   region is any [`Backend`] — so the partition composes
//!   recursively.)
//!
//! Rung-1 honesty: regions are **product factors** — entanglement
//! between regions forces a merge into one representation, exactly as
//! in the factored backend. A cross-representation *bond* (two regions
//! entangled while each keeps its own lens, via the mutual-encoding
//! machinery) is the named next rung, not a silent assumption. Merges
//! and conversions enumerate region supports (guard-admitted,
//! exponential when a region's support is — measured, never hidden),
//! and re-separation of disentangled regions is a further rung.
//!
//! Phase contract, the dual of the clock module's: a **product**
//! composition forgives phase-loose representations — a region's
//! global phase is a global phase of the whole product, so a bundle
//! region (defined up to phase; its vertex-operator reductions rotate
//! it, measured at exactly `e^{−iπ/4}` per reduced CX chain) is sound
//! as a mosaic tile, and the mosaic's amplitudes are then defined up
//! to one global phase. A **superposition** composition
//! ([`BranchedRegister`](crate::clock::BranchedRegister)) makes the
//! same phases relative, hence physical — which is why the clock
//! module demands phase-faithful branches while the mosaic does not
//! need to.

use super::{
    validate_apply, validate_apply_diagonal, Backend, BulkState, DenseState, FactoredState,
    MpsState, SparseState,
};
use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::scalar::Scalar;

/// Maximum register width (basis indices are `u64`).
pub const MOSAIC_MAX_QUBITS: usize = 63;

/// How the mosaic chooses and changes representations.
#[derive(Debug, Clone)]
pub struct MosaicPolicy {
    /// Candidate representations for refusal-driven migration, tried
    /// in order. Must be constructible names from
    /// {"sparse", "dense", "mps", "bulk", "factored"}.
    pub fallbacks: Vec<String>,
    /// Merged regions pick sparse when the predicted support cost
    /// stays below this fraction of the predicted dense cost.
    pub sparse_bias: f64,
}

impl Default for MosaicPolicy {
    fn default() -> Self {
        MosaicPolicy {
            fallbacks: vec!["sparse".into(), "dense".into()],
            sparse_bias: 1.0,
        }
    }
}

/// One region: a set of global qubits (sorted; position = local bit)
/// held by its own backend of local width.
struct Region<S: Scalar> {
    qubits: Vec<usize>,
    state: Box<dyn Backend<S>>,
}

impl<S: Scalar> Region<S> {
    fn local(&self, global: usize) -> usize {
        self.qubits
            .iter()
            .position(|&q| q == global)
            .expect("qubit in region")
    }
}

/// One ledger entry: something the mosaic decided, with its cause.
#[derive(Debug, Clone)]
pub struct MosaicEvent {
    /// "migrate" or "merge".
    pub kind: &'static str,
    /// The qubits involved.
    pub qubits: Vec<usize>,
    /// Representations before → after.
    pub from: String,
    /// The representation chosen.
    pub to: String,
    /// Why (the refusal text, or the predicted-cost comparison).
    pub cause: String,
}

/// The multi-representation register. See the module docs.
pub struct MosaicState<S: Scalar> {
    n: usize,
    regions: Vec<Region<S>>,
    policy: MosaicPolicy,
    events: Vec<MosaicEvent>,
    conversions_bytes: usize,
}

impl<S: Scalar> MosaicState<S> {
    /// `|0…0⟩` as per-qubit sparse singleton regions with the default
    /// policy — the conformance-facing constructor; structure emerges
    /// from the gates.
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits == 0 || num_qubits > MOSAIC_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: num_qubits,
                max: MOSAIC_MAX_QUBITS,
            });
        }
        let regions = (0..num_qubits)
            .map(|q| {
                Ok(Region {
                    qubits: vec![q],
                    state: Box::new(SparseState::<S>::new(1)?) as Box<dyn Backend<S>>,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(MosaicState {
            n: num_qubits,
            regions,
            policy: MosaicPolicy::default(),
            events: Vec::new(),
            conversions_bytes: 0,
        })
    }

    /// Assemble a mosaic from explicit `(qubits, state)` regions.
    /// Regions must partition `0..num_qubits`; each state's width must
    /// equal its region size (local bit `i` ↔ `qubits[i]`).
    pub fn with_regions(
        num_qubits: usize,
        regions: Vec<(Vec<usize>, Box<dyn Backend<S>>)>,
        policy: MosaicPolicy,
    ) -> Result<Self> {
        let mut seen = vec![false; num_qubits];
        let mut out = Vec::with_capacity(regions.len());
        for (qubits, state) in regions {
            if state.num_qubits() != qubits.len() {
                return Err(Error::BadDimension {
                    expected: qubits.len(),
                    got: state.num_qubits(),
                });
            }
            for &q in &qubits {
                if q >= num_qubits || seen[q] {
                    return Err(Error::InvalidState(format!(
                        "mosaic: qubit {q} out of range or claimed twice"
                    )));
                }
                seen[q] = true;
            }
            out.push(Region { qubits, state });
        }
        if !seen.iter().all(|&s| s) {
            return Err(Error::InvalidState(
                "mosaic: regions must partition the register".into(),
            ));
        }
        Ok(MosaicState {
            n: num_qubits,
            regions: out,
            policy,
            events: Vec::new(),
            conversions_bytes: 0,
        })
    }

    /// The live partition: per region its global qubits, backend name
    /// and resting memory — the mosaic as data.
    pub fn layout(&self) -> Vec<(Vec<usize>, String, usize)> {
        self.regions
            .iter()
            .map(|r| {
                (
                    r.qubits.clone(),
                    r.state.name().to_string(),
                    r.state.memory_bytes(),
                )
            })
            .collect()
    }

    /// Every migration and merge the mosaic performed, with causes.
    pub fn events(&self) -> &[MosaicEvent] {
        &self.events
    }

    /// Total bytes moved through representation conversions.
    pub fn conversions_bytes(&self) -> usize {
        self.conversions_bytes
    }

    /// Construct a fresh backend by candidate name at `width`.
    fn construct(name: &str, width: usize) -> Result<Box<dyn Backend<S>>> {
        Ok(match name {
            "sparse" => Box::new(SparseState::<S>::new(width)?),
            "dense" => Box::new(DenseState::<S>::new(width)?),
            "mps" => Box::new(MpsState::<S>::new(width)?),
            "bulk" => Box::new(BulkState::<S>::new(width)?),
            "factored" => Box::new(FactoredState::<S>::new(width)?),
            other => {
                return Err(Error::UnknownBackend(other.to_string()));
            }
        })
    }

    /// Convert a region's state into `name`, by support enumeration
    /// (guard-admitted; the ledger records the bytes moved).
    fn convert(&mut self, region: usize, name: &str) -> Result<()> {
        let width = self.regions[region].qubits.len();
        let mut entries = Vec::new();
        self.regions[region]
            .state
            .for_each_nonzero(&mut |i, a| entries.push((i, a)));
        let mut fresh = Self::construct(name, width)?;
        fresh.load(&entries)?;
        self.conversions_bytes += entries.len() * (std::mem::size_of::<S>() + 8);
        self.regions[region].state = fresh;
        Ok(())
    }

    /// Index of the region containing `qubit`.
    fn region_of(&self, qubit: usize) -> usize {
        self.regions
            .iter()
            .position(|r| r.qubits.contains(&qubit))
            .expect("validated qubit")
    }

    /// Merge every region touching `qubits` into one, choosing the
    /// merged representation from measured support predictions.
    fn merge_for(&mut self, qubits: &[usize]) -> Result<usize> {
        let mut involved: Vec<usize> = qubits.iter().map(|&q| self.region_of(q)).collect();
        involved.sort_unstable();
        involved.dedup();
        if involved.len() == 1 {
            return Ok(involved[0]);
        }
        // Predict: sparse pays the product of supports; dense pays 2^w.
        let mut support_product = 1f64;
        let mut width = 0usize;
        let mut names = Vec::new();
        for &r in &involved {
            support_product *= self.regions[r].state.nonzero_count().max(1) as f64;
            width += self.regions[r].qubits.len();
            names.push(self.regions[r].state.name().to_string());
        }
        let entry = (std::mem::size_of::<S>() + 24) as f64;
        let sparse_cost = support_product * entry;
        let dense_cost = (1u128 << width.min(80)) as f64 * std::mem::size_of::<S>() as f64;
        let target = if sparse_cost <= dense_cost * self.policy.sparse_bias {
            "sparse"
        } else {
            "dense"
        };
        // Gather the merged product state by cross-enumeration.
        let mut acc: Vec<(u64, S)> = vec![(0, S::one())];
        let mut merged_qubits: Vec<usize> = Vec::new();
        for &r in &involved {
            let offset = merged_qubits.len();
            let mut local = Vec::new();
            self.regions[r]
                .state
                .for_each_nonzero(&mut |i, a| local.push((i, a)));
            let mut next = Vec::with_capacity(acc.len() * local.len());
            for &(ia, va) in &acc {
                for &(ib, vb) in &local {
                    next.push((ia | (ib << offset), va * vb));
                }
            }
            crate::guard::checkpoint()?;
            acc = next;
            merged_qubits.extend(self.regions[r].qubits.iter().copied());
        }
        let mut fresh = Self::construct(target, width)?;
        fresh.load(&acc)?;
        self.conversions_bytes += acc.len() * (std::mem::size_of::<S>() + 8);
        self.events.push(MosaicEvent {
            kind: "merge",
            qubits: merged_qubits.clone(),
            from: names.join("⊗"),
            to: target.to_string(),
            cause: format!(
                "predicted sparse {:.0} B vs dense {:.0} B",
                sparse_cost, dense_cost
            ),
        });
        // Replace the first involved region; drop the rest.
        let keep = involved[0];
        self.regions[keep] = Region {
            qubits: merged_qubits,
            state: fresh,
        };
        for &r in involved[1..].iter().rev() {
            self.regions.remove(r);
        }
        Ok(keep)
    }

    /// Split a region along structure its representation carries for
    /// free: a bundle region whose graph has several components
    /// becomes one **bundle region per component** — the graph and the
    /// graph of graphs — each still in its cheap representation, split
    /// exactly (a graph state is the product of its components).
    /// Returns whether a split happened.
    fn try_split_components(&mut self, region: usize) -> bool {
        use crate::bundle::PolarityBundle;
        let built: Option<Vec<(Vec<usize>, PolarityBundle)>> = {
            let r = &self.regions[region];
            r.state
                .as_any()
                .downcast_ref::<PolarityBundle>()
                .and_then(|b| {
                    let comps = b.graph_components();
                    if comps.len() < 2 {
                        return None;
                    }
                    let mut subs = Vec::with_capacity(comps.len());
                    for comp in &comps {
                        let global: Vec<usize> =
                            comp.iter().map(|&l| r.qubits[l as usize]).collect();
                        match b.restrict(comp) {
                            Ok(sub) => subs.push((global, sub)),
                            Err(_) => return None,
                        }
                    }
                    Some(subs)
                })
        };
        let Some(subs) = built else {
            return false;
        };
        let count = subs.len();
        let from_qubits = self.regions[region].qubits.clone();
        let mut iter = subs.into_iter();
        let (first_qubits, first_state) = iter.next().expect("at least two components");
        self.regions[region] = Region {
            qubits: first_qubits,
            state: Box::new(first_state),
        };
        for (qubits, state) in iter {
            self.regions.push(Region {
                qubits,
                state: Box::new(state),
            });
        }
        self.events.push(MosaicEvent {
            kind: "split",
            qubits: from_qubits,
            from: "bundle".into(),
            to: format!("bundle×{count}"),
            cause: "graph components factor the region exactly".into(),
        });
        true
    }

    /// Route a gate: merge the touched regions, then apply with
    /// refusal-driven structure discovery — first try splitting the
    /// region along its own components so only the affected part
    /// leaves its representation, then convert down the policy's
    /// candidate list and retry.
    fn dispatch(
        &mut self,
        qubits: &[usize],
        apply: &impl Fn(&mut dyn Backend<S>, &[usize]) -> Result<()>,
    ) -> Result<()> {
        let region = self.merge_for(qubits)?;
        let local: Vec<usize> = qubits
            .iter()
            .map(|&q| self.regions[region].local(q))
            .collect();
        let first = apply(self.regions[region].state.as_mut(), &local);
        let refusal = match first {
            Ok(()) => return Ok(()),
            Err(e @ Error::UnsupportedForAlgebra { .. }) | Err(e @ Error::InvalidState(_)) => e,
            Err(other) => return Err(other),
        };
        // Partial structure first: fracture the region along its own
        // graph so untouched components keep their representation.
        if self.try_split_components(region) {
            return self.dispatch(qubits, apply);
        }
        let from = self.regions[region].state.name().to_string();
        let fallbacks = self.policy.fallbacks.clone();
        for name in &fallbacks {
            if *name == from {
                continue;
            }
            if self.convert(region, name).is_err() {
                continue;
            }
            if apply(self.regions[region].state.as_mut(), &local).is_ok() {
                self.events.push(MosaicEvent {
                    kind: "migrate",
                    qubits: self.regions[region].qubits.clone(),
                    from,
                    to: name.clone(),
                    cause: format!("refused: {refusal}"),
                });
                return Ok(());
            }
        }
        Err(refusal)
    }
}

impl<S: Scalar> Backend<S> for MosaicState<S> {
    fn name(&self) -> &str {
        "mosaic"
    }

    fn num_qubits(&self) -> usize {
        self.n
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        validate_apply(self.n, matrix, qubits)?;
        self.dispatch(qubits, &|s, local| s.apply(matrix, local))
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        validate_apply_diagonal(self.n, entries, qubits)?;
        self.dispatch(qubits, &|s, local| s.apply_diagonal(entries, local))
    }

    fn amplitude(&self, index: u64) -> S {
        let mut acc = S::one();
        for r in &self.regions {
            let mut local = 0u64;
            for (bit, &q) in r.qubits.iter().enumerate() {
                if (index >> q) & 1 == 1 {
                    local |= 1 << bit;
                }
            }
            acc = acc * r.state.amplitude(local);
            if acc.abs_sqr() == 0.0 {
                return acc;
            }
        }
        acc
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        // Cross product over regions (documented trait behavior:
        // exponential when the joint support is).
        let mut acc: Vec<(u64, S)> = vec![(0, S::one())];
        for r in &self.regions {
            let mut local = Vec::new();
            r.state.for_each_nonzero(&mut |i, a| local.push((i, a)));
            let mut next = Vec::with_capacity(acc.len() * local.len());
            for &(ia, va) in &acc {
                for &(il, vl) in &local {
                    let mut global = ia;
                    for (bit, &q) in r.qubits.iter().enumerate() {
                        if (il >> bit) & 1 == 1 {
                            global |= 1u64 << q;
                        }
                    }
                    let v = va * vl;
                    if v.abs_sqr() > 0.0 {
                        next.push((global, v));
                    }
                }
            }
            acc = next;
        }
        acc.sort_unstable_by_key(|&(i, _)| i);
        for (i, a) in acc {
            f(i, a);
        }
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        let region = self.region_of(qubit);
        let local = self.regions[region].local(qubit);
        self.regions[region].state.project(local, outcome, renorm);
    }

    fn reset(&mut self) {
        let policy = self.policy.clone();
        *self = MosaicState::new(self.n).expect("width already validated");
        self.policy = policy;
    }

    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        let mut state = SparseState::<S>::new(self.n)?;
        state.load(entries)?;
        self.regions = vec![Region {
            qubits: (0..self.n).collect(),
            state: Box::new(state),
        }];
        self.events.clear();
        self.conversions_bytes = 0;
        Ok(())
    }

    /// The sum of the regions — each portion priced in its own
    /// representation.
    fn memory_bytes(&self) -> usize {
        self.regions
            .iter()
            .map(|r| r.state.memory_bytes() + r.qubits.len() * std::mem::size_of::<usize>())
            .sum::<usize>()
            + std::mem::size_of::<Self>()
    }

    fn total_abs_sqr(&self) -> f64 {
        self.regions
            .iter()
            .map(|r| r.state.total_abs_sqr())
            .product()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Deterministic same-index accumulation for the conformance sampler:
/// the mosaic's `for_each_nonzero` already visits in index order, so
/// the default `sample` and `measure` apply unchanged.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::C64;

    #[test]
    fn singleton_mosaic_is_the_zero_state() {
        let m = MosaicState::<C64>::new(4).unwrap();
        assert_eq!(m.num_qubits(), 4);
        assert_eq!(m.layout().len(), 4);
        assert!((m.amplitude(0).re - 1.0).abs() < 1e-12);
        assert!((m.total_abs_sqr() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn regions_must_partition() {
        let s: Box<dyn Backend<C64>> = Box::new(SparseState::<C64>::new(2).unwrap());
        let err = MosaicState::with_regions(3, vec![(vec![0, 1], s)], MosaicPolicy::default());
        assert!(err.is_err(), "a hole in the partition must refuse");
    }
}
