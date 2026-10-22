//! The composite register: one mixed-radix [`Space`], held as a product
//! of **regions**, each region a sub-space in its own [`Backend`], and the
//! partition changing as the space itself changes over time.
//!
//! The space module made the register's shape a path
//! ([`Transition`]s), and the native stores walk that path by moving
//! every amplitude to the new index map: growing a qutrit onto a register
//! of dimension `D` costs a fresh `3D` allocation, whatever the new site
//! does. A [`CompositeState`] walks the same path with cost set by the
//! *entanglement* the path creates rather than by the widest space on it:
//!
//! * **Grow is a new region.** `|ψ⟩ ↦ |ψ⟩ ⊗ |0⟩` is literally a product,
//!   so the new site becomes a singleton region and no existing amplitude
//!   is touched — `O(regions)` bookkeeping, at any virtual dimension.
//! * **Gates run in the region that owns their targets**, natively in
//!   that region's representation. A gate across regions merges them
//!   first (support enumeration, guard-admitted, ledgered), the merged
//!   store **elected** dense or sparse from its measured support.
//! * **Regions re-separate.** After every multi-site step, each touched
//!   site is tested for a rank-1 factorization against the rest of its
//!   region (the generalization of [`FactoredState`](super::FactoredState)'s
//!   qubit test to any radix and any region backend), and a measured or
//!   projected site is split out exactly. A workspace that was entangled,
//!   uncomputed and left in `|0⟩` is its own region again before anyone
//!   asks — so the release that follows costs nothing and moves nothing.
//! * **Transitions are local.** Promote, demote, release, merge and split
//!   run in the owning region (merging regions first when a site merge
//!   straddles two); a removed weight is the region's removed weight times
//!   the other regions' norms — the exact global figure — and is refused
//!   or returned under the same [`Error::WouldDiscard`] contract as the
//!   native stores.
//! * **Regions may be any backend.** [`CompositeState::with_regions`]
//!   seats caller-built regions — an [`MpsState`](super::MpsState) on a
//!   qubit block beside radix-sparse qutrits — and they evolve natively
//!   until a cross-region gate or a transition they cannot perform forces
//!   them into an elected native store (ledgered as a conversion). That
//!   is "several backends simultaneously, one virtual space", with the
//!   price of every crossing written down.
//!
//! ## Boundaries
//!
//! * **Commutative algebras only.** A product of regions is a tensor
//!   product only when a gate on one factor commutes past the others'
//!   amplitudes: `U·(a·b) = a·(U·b)` needs commutativity. Over ℍ or 𝕆 the
//!   constructor refuses by name.
//! * **Rung 1 = product factors**, as for the mosaic: entanglement
//!   between regions forces a merge into one store, and a merge
//!   enumerates the product of the merged supports. Dense stays dense when
//!   the entanglement is global; the composite only wins while it is
//!   clustered.
//! * **Re-separation is single-site.** A rank-1 test per touched site,
//!   tolerance [`CompositeState::set_split_tolerance`] (default `1e-11`,
//!   the factored backend's); a region whose sites are pairwise entangled
//!   but separable as two *blocks* is not split (the block test is the
//!   next rung).
//! * **The virtual space is still a [`Space`]**, so its dimension must fit
//!   the `u64` basis index. The composite never allocates it — memory is
//!   the sum of the regions — but the enumeration contract
//!   ([`Backend::amplitude`], [`Backend::for_each_nonzero`]) is keyed by
//!   that index, and a register past it would need a structural-only
//!   contract the trait does not have yet.

use rustc_hash::FxHashMap;

use super::{validate_apply, validate_apply_diagonal, Backend, RadixDenseState, RadixSparseState};
use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::rng::Prng;
use crate::scalar::Scalar;
use crate::space::{admit_discard, require_binary, validate_sites, SiteMatrix, Space, Transition};

/// How the composite elects a store for a region it builds (a merge, a
/// re-separation, a forced conversion).
#[derive(Debug, Clone)]
pub struct CompositePolicy {
    /// Regions of at most this dimension are always dense: below it the
    /// hash map's constants cost more than the zeros.
    pub dense_floor: u128,
    /// Larger regions go dense when their measured support is at least
    /// this fraction of their dimension (the adaptive backend's ¼).
    pub dense_density: f64,
}

impl Default for CompositePolicy {
    fn default() -> Self {
        CompositePolicy {
            dense_floor: 64,
            dense_density: 0.25,
        }
    }
}

impl CompositePolicy {
    /// The store a region of `space` with `support` nonzeros should use.
    pub fn elect(&self, space: &Space, support: usize) -> &'static str {
        let dim = space.dim();
        if dim <= self.dense_floor || support as f64 >= self.dense_density * dim as f64 {
            "radix-dense"
        } else {
            "radix-sparse"
        }
    }
}

/// One region: a set of global sites (sorted; position = local site)
/// held by its own backend over the sub-space they span.
struct Region<S: Scalar> {
    sites: Vec<usize>,
    local: Space,
    state: Box<dyn Backend<S>>,
}

impl<S: Scalar> Region<S> {
    fn position(&self, global: usize) -> usize {
        self.sites
            .iter()
            .position(|&s| s == global)
            .expect("site in region")
    }

    /// The region's local index of a global basis index.
    fn local_index(&self, global: &Space, index: u64) -> u64 {
        self.sites
            .iter()
            .enumerate()
            .map(|(k, &s)| global.digit(index, s) as u64 * self.local.stride(k))
            .sum()
    }

    /// The region's contribution to a global index, from a local index.
    fn global_part(&self, global: &Space, local_index: u64) -> u64 {
        self.sites
            .iter()
            .enumerate()
            .map(|(k, &s)| self.local.digit(local_index, k) as u64 * global.stride(s))
            .sum()
    }

    fn entries(&self) -> Vec<(u64, S)> {
        let mut out = Vec::with_capacity(self.state.nonzero_count());
        self.state.for_each_nonzero(&mut |i, a| out.push((i, a)));
        out
    }
}

/// One ledger entry: something the composite decided, with its cause.
#[derive(Debug, Clone, PartialEq)]
pub struct CompositeEvent {
    /// `"merge"`, `"separate"`, `"convert"` or `"reelect"`.
    pub kind: &'static str,
    /// The global sites of the region(s) involved, at the time.
    pub sites: Vec<usize>,
    /// The store(s) before.
    pub from: String,
    /// The store after.
    pub to: String,
    /// Why (the gate or transition, or the refusal text).
    pub cause: String,
}

/// A mixed-radix register held as a product of regions in heterogeneous
/// backends. See the module docs.
pub struct CompositeState<S: Scalar> {
    space: Space,
    regions: Vec<Region<S>>,
    site_region: Vec<usize>,
    policy: CompositePolicy,
    auto_split: bool,
    split_tol: f64,
    events: Vec<CompositeEvent>,
    peak_region_dim: u128,
    peak_stored: usize,
}

fn check_algebra<S: Scalar>() -> Result<()> {
    if !(S::COMMUTATIVE && S::ASSOCIATIVE) {
        return Err(Error::InvalidState(format!(
            "composite requires a commutative algebra (a product of regions is a tensor \
             product only when gates commute past the other factors); {} is not",
            S::algebra_name()
        )));
    }
    Ok(())
}

/// Build the elected native store over `space` holding `entries`.
fn native<S: Scalar>(
    name: &str,
    space: Space,
    entries: &[(u64, S)],
) -> Result<Box<dyn Backend<S>>> {
    let mut state: Box<dyn Backend<S>> = if name == "radix-dense" {
        Box::new(RadixDenseState::<S>::new(space)?)
    } else {
        Box::new(RadixSparseState::<S>::new(space)?)
    };
    state.load(entries)?;
    Ok(state)
}

impl<S: Scalar> CompositeState<S> {
    /// `|0…0⟩` over `space`, one dense singleton region per site.
    pub fn new(space: Space) -> Result<Self> {
        check_algebra::<S>()?;
        if space.sites() == 0 {
            return Err(Error::InvalidState(
                "a composite register needs at least one site".into(),
            ));
        }
        crate::guard::admit(
            space
                .sites()
                .saturating_mul(std::mem::size_of::<Region<S>>() + 64),
            "composite regions",
        )?;
        let regions = (0..space.sites())
            .map(|s| {
                let local = Space::new(&[space.radix(s)])?;
                Ok(Region {
                    sites: vec![s],
                    state: Box::new(RadixDenseState::<S>::new(local.clone())?)
                        as Box<dyn Backend<S>>,
                    local,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut state = CompositeState {
            space,
            regions,
            site_region: Vec::new(),
            policy: CompositePolicy::default(),
            auto_split: true,
            split_tol: 1e-11,
            events: Vec::new(),
            peak_region_dim: 0,
            peak_stored: 0,
        };
        state.rebuild();
        Ok(state)
    }

    /// A register over `space` seated with caller-built regions: each
    /// entry is a set of global sites and a backend whose space is the
    /// sub-space they span (radices in ascending site order). The regions
    /// must partition the sites; the register is their product. Any
    /// backend is accepted — an [`MpsState`](super::MpsState) on a qubit
    /// block, a radix store on a qudit block — and evolves natively until
    /// a crossing forces a conversion.
    pub fn with_regions(
        space: Space,
        regions: Vec<(Vec<usize>, Box<dyn Backend<S>>)>,
    ) -> Result<Self> {
        let mut state = Self::new(space)?;
        let n = state.space.sites();
        let mut seen = vec![false; n];
        let mut seated = Vec::with_capacity(regions.len());
        for (mut sites, backend) in regions {
            sites.sort_unstable();
            for &s in &sites {
                if s >= n || seen[s] {
                    return Err(Error::InvalidState(format!(
                        "composite: regions must partition sites 0..{n}; site {s} is \
                         out of range or repeated"
                    )));
                }
                seen[s] = true;
            }
            let radices: Vec<usize> = sites.iter().map(|&s| state.space.radix(s)).collect();
            let local = Space::new(&radices)?;
            if backend.space() != local {
                return Err(Error::UnsupportedForSpace {
                    operation: format!(
                        "seating backend '{}' (space {}) as the region over sites {sites:?}",
                        backend.name(),
                        backend.space()
                    ),
                    space: local.to_string(),
                });
            }
            seated.push(Region {
                sites,
                local,
                state: backend,
            });
        }
        if let Some(s) = seen.iter().position(|&b| !b) {
            return Err(Error::InvalidState(format!(
                "composite: regions must partition sites 0..{n}; site {s} is in none"
            )));
        }
        state.regions = seated;
        state.rebuild();
        Ok(state)
    }

    /// The live partition: per region (in ascending order of first site),
    /// its global sites, its store's name, its dimension and its stored
    /// nonzero count.
    pub fn layout(&self) -> Vec<(Vec<usize>, String, u128, usize)> {
        self.regions
            .iter()
            .map(|r| {
                (
                    r.sites.clone(),
                    r.state.name().to_string(),
                    r.local.dim(),
                    r.state.nonzero_count(),
                )
            })
            .collect()
    }

    /// Number of regions.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Dimension of the largest region now.
    pub fn largest_region_dim(&self) -> u128 {
        self.regions
            .iter()
            .map(|r| r.local.dim())
            .max()
            .unwrap_or(1)
    }

    /// Lifetime peaks: (largest region dimension, largest total of stored
    /// nonzeros summed over regions).
    pub fn peak(&self) -> (u128, usize) {
        (self.peak_region_dim, self.peak_stored)
    }

    /// The decision ledger.
    pub fn events(&self) -> &[CompositeEvent] {
        &self.events
    }

    /// The store-election policy.
    pub fn policy(&self) -> &CompositePolicy {
        &self.policy
    }

    /// Replace the store-election policy (applies to regions built from
    /// now on).
    pub fn set_policy(&mut self, policy: CompositePolicy) {
        self.policy = policy;
    }

    /// Enable/disable the rank-1 re-separation attempt after multi-site
    /// steps. (The exact split after a measurement or projection always
    /// runs.)
    pub fn set_auto_split(&mut self, enabled: bool) {
        self.auto_split = enabled;
    }

    /// Residual tolerance for accepting a rank-1 split.
    pub fn set_split_tolerance(&mut self, tol: f64) {
        self.split_tol = tol;
    }

    // ── bookkeeping ──────────────────────────────────────────────────

    fn rebuild(&mut self) {
        self.regions.sort_by_key(|r| r.sites[0]);
        self.site_region = vec![0; self.space.sites()];
        for (ri, r) in self.regions.iter().enumerate() {
            for &s in &r.sites {
                self.site_region[s] = ri;
            }
        }
        let dim = self.largest_region_dim();
        let stored: usize = self.regions.iter().map(|r| r.state.nonzero_count()).sum();
        self.peak_region_dim = self.peak_region_dim.max(dim);
        self.peak_stored = self.peak_stored.max(stored);
    }

    /// Relabel every global site id through `f` (used when a transition
    /// inserts or removes a site).
    fn renumber(&mut self, f: impl Fn(usize) -> usize) {
        for r in &mut self.regions {
            for s in &mut r.sites {
                *s = f(*s);
            }
        }
    }

    fn log(
        &mut self,
        kind: &'static str,
        sites: Vec<usize>,
        from: String,
        to: &str,
        cause: String,
    ) {
        self.events.push(CompositeEvent {
            kind,
            sites,
            from,
            to: to.to_string(),
            cause,
        });
    }

    /// Product of every region's Euclidean norm² except `skip`'s: the
    /// factor turning a region-local removed weight into the global one.
    fn others_abs_sqr(&self, skip: usize) -> f64 {
        self.regions
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != skip)
            .map(|(_, r)| r.state.total_abs_sqr())
            .product()
    }

    /// Merge every region owning one of `sites` into one elected native
    /// store; returns its index. One region already: nothing happens.
    fn merge_for(&mut self, sites: &[usize], cause: &str) -> Result<usize> {
        let mut involved: Vec<usize> = sites.iter().map(|&s| self.site_region[s]).collect();
        involved.sort_unstable();
        involved.dedup();
        if involved.len() == 1 {
            return Ok(involved[0]);
        }
        let mut merged: Vec<usize> = involved
            .iter()
            .flat_map(|&ri| self.regions[ri].sites.iter().copied())
            .collect();
        merged.sort_unstable();
        let radices: Vec<usize> = merged.iter().map(|&s| self.space.radix(s)).collect();
        let local = Space::new(&radices)?;
        let support: u128 = involved
            .iter()
            .map(|&ri| self.regions[ri].state.nonzero_count() as u128)
            .product();
        let support = usize::try_from(support).map_err(|_| Error::OutOfMemory {
            what: format!("composite merge over sites {merged:?}"),
            requested: usize::MAX,
            available: 0,
        })?;
        crate::guard::admit(
            support.saturating_mul(std::mem::size_of::<(u64, S)>()),
            "composite merge",
        )?;
        // Each region's entries re-keyed into the merged local index.
        let pos: FxHashMap<usize, usize> =
            merged.iter().enumerate().map(|(k, &s)| (s, k)).collect();
        let mut acc: Vec<(u64, S)> = vec![(0, S::one())];
        for &ri in &involved {
            let r = &self.regions[ri];
            let keyed: Vec<(u64, S)> = r
                .entries()
                .into_iter()
                .map(|(i, a)| {
                    let key = r
                        .sites
                        .iter()
                        .enumerate()
                        .map(|(k, s)| r.local.digit(i, k) as u64 * local.stride(pos[s]))
                        .sum();
                    (key, a)
                })
                .collect();
            let mut next = Vec::with_capacity(acc.len() * keyed.len());
            for (step, &(i, a)) in acc.iter().enumerate() {
                if step % 4096 == 0 {
                    crate::guard::checkpoint()?;
                }
                for &(j, b) in &keyed {
                    next.push((i + j, a * b));
                }
            }
            acc = next;
        }
        let from = involved
            .iter()
            .map(|&ri| self.regions[ri].state.name().to_string())
            .collect::<Vec<_>>()
            .join("+");
        let name = self.policy.elect(&local, acc.len());
        let state = native(name, local.clone(), &acc)?;
        for &ri in involved.iter().rev() {
            self.regions.remove(ri);
        }
        self.regions.push(Region {
            sites: merged.clone(),
            local,
            state,
        });
        self.rebuild();
        self.log("merge", merged, from, name, cause.to_string());
        Ok(self.site_region[sites[0]])
    }

    /// Replace region `ri`'s store with an elected native one holding the
    /// same amplitudes — for a seated backend asked to do what only a
    /// native store can (change shape).
    fn convert(&mut self, ri: usize, cause: String) -> Result<()> {
        let entries = self.regions[ri].entries();
        let local = self.regions[ri].local.clone();
        let name = self.policy.elect(&local, entries.len());
        let from = self.regions[ri].state.name().to_string();
        self.regions[ri].state = native(name, local, &entries)?;
        let sites = self.regions[ri].sites.clone();
        self.log("convert", sites, from, name, cause);
        Ok(())
    }

    /// A sparse region whose support has filled past the dense rule goes
    /// dense (one direction only, so a region never thrashes).
    fn reelect(&mut self, ri: usize, cause: &str) -> Result<()> {
        if self.regions[ri].state.name() != "radix-sparse" {
            return Ok(());
        }
        let support = self.regions[ri].state.nonzero_count();
        if self.policy.elect(&self.regions[ri].local, support) == "radix-dense" {
            let entries = self.regions[ri].entries();
            let local = self.regions[ri].local.clone();
            self.regions[ri].state = native("radix-dense", local, &entries)?;
            let sites = self.regions[ri].sites.clone();
            self.log(
                "reelect",
                sites,
                "radix-sparse".into(),
                "radix-dense",
                format!("support {support} filled past the dense rule after {cause}"),
            );
        }
        Ok(())
    }

    /// Detach `site` from its region as a singleton holding `single`
    /// (amplitudes over its radix), the region keeping `rest` over its
    /// remaining sites.
    fn detach(
        &mut self,
        site: usize,
        single: Vec<S>,
        rest: Vec<(u64, S)>,
        cause: String,
    ) -> Result<()> {
        let ri = self.site_region[site];
        let p = self.regions[ri].position(site);
        let mut sites = self.regions[ri].sites.clone();
        sites.remove(p);
        let radices: Vec<usize> = sites.iter().map(|&s| self.space.radix(s)).collect();
        let rest_space = Space::new(&radices)?;
        let from = self.regions[ri].state.name().to_string();
        let name = self.policy.elect(&rest_space, rest.len());
        let state = native(name, rest_space.clone(), &rest)?;
        let single_space = Space::new(&[self.space.radix(site)])?;
        let mut single_state = RadixDenseState::<S>::new(single_space.clone())?;
        let single_entries: Vec<(u64, S)> = single
            .into_iter()
            .enumerate()
            .map(|(d, a)| (d as u64, a))
            .collect();
        single_state.load(&single_entries)?;
        self.regions[ri] = Region {
            sites: sites.clone(),
            local: rest_space,
            state,
        };
        self.regions.push(Region {
            sites: vec![site],
            local: single_space,
            state: Box::new(single_state),
        });
        self.rebuild();
        self.log("separate", vec![site], from, name, cause);
        Ok(())
    }

    /// Split `site` out of its region when the region's state is
    /// `|α⟩_site ⊗ |rest⟩` to the split tolerance: the rank-1 test,
    /// against the heaviest column as pivot.
    fn try_split(&mut self, site: usize, cause: &str) -> Result<bool> {
        let ri = self.site_region[site];
        if self.regions[ri].sites.len() == 1 {
            return Ok(false);
        }
        let r = &self.regions[ri];
        let p = r.position(site);
        let radix = r.local.radix(p);
        let stride = r.local.stride(p);
        // Group amplitudes by the index of the remaining sites.
        let mut columns: FxHashMap<u64, Vec<S>> = FxHashMap::default();
        r.state.for_each_nonzero(&mut |i, a| {
            let d = r.local.digit(i, p);
            let rest = i % stride + (i / (stride * radix as u64)) * stride;
            columns
                .entry(rest)
                .or_insert_with(|| vec![S::zero(); radix])[d] = a;
        });
        let Some((_, pivot)) = columns.iter().max_by(|x, y| {
            let w = |v: &Vec<S>| v.iter().map(|a| a.abs_sqr()).sum::<f64>();
            w(x.1).total_cmp(&w(y.1))
        }) else {
            return Ok(false);
        };
        let norm = pivot.iter().map(|a| a.abs_sqr()).sum::<f64>().sqrt();
        if norm <= 0.0 {
            return Ok(false);
        }
        let alpha: Vec<S> = pivot.iter().map(|a| a.scale(1.0 / norm)).collect();
        let mut rest = Vec::with_capacity(columns.len());
        for (&k, col) in &columns {
            let mut v = S::zero();
            for (al, &a) in alpha.iter().zip(col) {
                v = v + al.conj() * a;
            }
            for (al, &a) in alpha.iter().zip(col) {
                if (*al * v - a).abs_sqr().sqrt() > self.split_tol {
                    return Ok(false);
                }
            }
            if v.abs_sqr() > 0.0 {
                rest.push((k, v));
            }
        }
        self.detach(site, alpha, rest, format!("rank-1 after {cause}"))?;
        Ok(true)
    }

    /// After a projection onto `digit`, `site` is exactly `|digit⟩`:
    /// split it out with the amplitude left in the rest (exact for every
    /// algebra this backend admits).
    fn split_projected(&mut self, site: usize, digit: usize) -> Result<()> {
        let ri = self.site_region[site];
        if self.regions[ri].sites.len() == 1 {
            return Ok(());
        }
        let r = &self.regions[ri];
        let p = r.position(site);
        let radix = r.local.radix(p) as u64;
        let stride = r.local.stride(p);
        let mut rest = Vec::with_capacity(r.state.nonzero_count());
        r.state.for_each_nonzero(&mut |i, a| {
            if r.local.digit(i, p) == digit {
                rest.push((i % stride + (i / (stride * radix)) * stride, a));
            }
        });
        let mut single = vec![S::zero(); radix as usize];
        single[digit] = S::one();
        self.detach(site, single, rest, format!("projected onto digit {digit}"))
    }

    /// Re-separation pass over the sites a step touched.
    fn resplit(&mut self, sites: &[usize], cause: &str) -> Result<()> {
        if !self.auto_split {
            return Ok(());
        }
        for &s in sites {
            self.try_split(s, cause)?;
        }
        Ok(())
    }

    fn locals(&self, ri: usize, sites: &[usize]) -> Vec<usize> {
        sites
            .iter()
            .map(|&s| self.regions[ri].position(s))
            .collect()
    }

    /// Route a gate: merge its regions, apply natively, re-elect, re-split.
    fn route(
        &mut self,
        sites: &[usize],
        cause: &str,
        op: impl FnOnce(&mut dyn Backend<S>, &[usize]) -> Result<()>,
    ) -> Result<()> {
        let ri = self.merge_for(sites, cause)?;
        let locals = self.locals(ri, sites);
        op(self.regions[ri].state.as_mut(), &locals)?;
        self.reelect(ri, cause)?;
        self.rebuild();
        if sites.len() > 1 {
            self.resplit(sites, cause)?;
        }
        Ok(())
    }

    /// Run `t` on region `ri` in local terms, converting a seated backend
    /// that cannot change shape; returns the region-local removed weight.
    fn region_transition(&mut self, ri: usize, t: &Transition, allow_discard: bool) -> Result<f64> {
        match self.regions[ri].state.transition(t, allow_discard) {
            Err(Error::UnsupportedForSpace { operation, .. })
                if self.regions[ri].state.name() != "radix-dense"
                    && self.regions[ri].state.name() != "radix-sparse" =>
            {
                self.convert(ri, operation)?;
                self.regions[ri].state.transition(t, allow_discard)
            }
            other => other,
        }
    }

    /// Release a site that is its own region: its digit-0 amplitude `c`
    /// survives and is folded into another region (exact over a
    /// commutative algebra); the other levels are the removed weight.
    fn release_singleton(
        &mut self,
        t: &Transition,
        site: usize,
        allow_discard: bool,
    ) -> Result<f64> {
        let ri = self.site_region[site];
        let mut c = S::zero();
        let mut local_dropped = 0.0;
        self.regions[ri].state.for_each_nonzero(&mut |i, a| {
            if i == 0 {
                c = a;
            } else {
                local_dropped += a.abs_sqr();
            }
        });
        let dropped = local_dropped * self.others_abs_sqr(ri);
        admit_discard(t, &self.space, dropped, allow_discard)?;
        self.regions.remove(ri);
        if !c.approx_eq(S::one(), 0.0) {
            // Fold into the smallest remaining region.
            let target = (0..self.regions.len())
                .min_by_key(|&k| self.regions[k].state.nonzero_count())
                .expect("a register keeps at least one site");
            let scaled: Vec<(u64, S)> = self.regions[target]
                .entries()
                .into_iter()
                .map(|(i, a)| (i, c * a))
                .collect();
            self.regions[target].state.load(&scaled)?;
        }
        Ok(dropped)
    }
}

impl<S: Scalar> Backend<S> for CompositeState<S> {
    fn name(&self) -> &str {
        "composite"
    }

    fn num_qubits(&self) -> usize {
        self.space.sites()
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        validate_apply(self.space.sites(), matrix, qubits)?;
        require_binary(&self.space, qubits, "apply")?;
        self.route(qubits, "a qubit gate", |state, locals| {
            state.apply(matrix, locals)
        })
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        validate_apply_diagonal(self.space.sites(), entries, qubits)?;
        require_binary(&self.space, qubits, "apply_diagonal")?;
        self.route(qubits, "a diagonal gate", |state, locals| {
            state.apply_diagonal(entries, locals)
        })
    }

    fn space(&self) -> Space {
        self.space.clone()
    }

    fn apply_sites(&mut self, matrix: &SiteMatrix<S>, sites: &[usize]) -> Result<()> {
        validate_sites(&self.space, matrix, sites)?;
        self.route(sites, "a site gate", |state, locals| {
            state.apply_sites(matrix, locals)
        })
    }

    fn project_digit(&mut self, site: usize, digit: usize, renorm: f64) -> Result<()> {
        self.space.validate_sites(&[site])?;
        if digit >= self.space.radix(site) {
            return Err(Error::InvalidState(format!(
                "digit {digit} out of range for site {site} of radix {}",
                self.space.radix(site)
            )));
        }
        let ri = self.site_region[site];
        let p = self.regions[ri].position(site);
        self.regions[ri].state.project_digit(p, digit, renorm)?;
        self.split_projected(site, digit)
    }

    fn transition(&mut self, t: &Transition, allow_discard: bool) -> Result<f64> {
        let to = t.target(&self.space)?;
        let mut dropped = 0.0;
        match *t {
            Transition::Grow { at, radix } => {
                self.renumber(|s| if s >= at { s + 1 } else { s });
                let local = Space::new(&[radix])?;
                self.regions.push(Region {
                    sites: vec![at],
                    state: Box::new(RadixDenseState::<S>::new(local.clone())?),
                    local,
                });
            }
            Transition::Release { site } => {
                if self.regions[self.site_region[site]].sites.len() == 1 {
                    dropped = self.release_singleton(t, site, allow_discard)?;
                } else {
                    let ri = self.site_region[site];
                    let p = self.regions[ri].position(site);
                    let others = self.others_abs_sqr(ri);
                    // Refuse on the global figure, then let the region drop.
                    let local_weight = {
                        let r = &self.regions[ri];
                        let mut w = 0.0;
                        r.state.for_each_nonzero(&mut |i, a| {
                            if r.local.digit(i, p) != 0 {
                                w += a.abs_sqr();
                            }
                        });
                        w
                    };
                    admit_discard(t, &self.space, local_weight * others, allow_discard)?;
                    let local = Transition::Release { site: p };
                    dropped = self.region_transition(ri, &local, true)? * others;
                    self.regions[ri].sites.remove(p);
                    self.regions[ri].local = self.regions[ri].state.space();
                }
                self.renumber(|s| if s > site { s - 1 } else { s });
            }
            Transition::Promote { site, radix } | Transition::Demote { site, radix } => {
                let ri = self.site_region[site];
                let p = self.regions[ri].position(site);
                let others = self.others_abs_sqr(ri);
                let local = match *t {
                    Transition::Promote { .. } => Transition::Promote { site: p, radix },
                    _ => Transition::Demote { site: p, radix },
                };
                if local.can_discard() {
                    let r = &self.regions[ri];
                    let mut w = 0.0;
                    r.state.for_each_nonzero(&mut |i, a| {
                        if r.local.digit(i, p) >= radix {
                            w += a.abs_sqr();
                        }
                    });
                    admit_discard(t, &self.space, w * others, allow_discard)?;
                }
                dropped = self.region_transition(ri, &local, true)? * others;
                self.regions[ri].local = self.regions[ri].state.space();
            }
            Transition::Merge { site } => {
                let ri = self.merge_for(&[site, site + 1], &t.to_string())?;
                let p = self.regions[ri].position(site);
                self.region_transition(ri, &Transition::Merge { site: p }, false)?;
                self.regions[ri].sites.remove(p + 1);
                self.regions[ri].local = self.regions[ri].state.space();
                self.renumber(|s| if s > site + 1 { s - 1 } else { s });
            }
            Transition::Split { site, low } => {
                let ri = self.site_region[site];
                let p = self.regions[ri].position(site);
                self.region_transition(ri, &Transition::Split { site: p, low }, false)?;
                self.renumber(|s| if s > site { s + 1 } else { s });
                let r = &mut self.regions[ri];
                r.sites.insert(p + 1, site + 1);
                r.local = r.state.space();
            }
        }
        self.space = to;
        self.rebuild();
        if let Transition::Split { site, .. } = *t {
            self.resplit(&[site, site + 1], &t.to_string())?;
        }
        Ok(dropped)
    }

    fn measure_site(&mut self, site: usize, rng: &mut Prng) -> Result<usize> {
        self.space.validate_sites(&[site])?;
        let ri = self.site_region[site];
        let p = self.regions[ri].position(site);
        let digit = self.regions[ri].state.measure_site(p, rng)?;
        // The region renormalized its own branch; the contract renormalizes
        // the whole register, so the other regions' weight (not 1 after an
        // opted-in discard) is divided out too.
        let others: f64 = self
            .regions
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != ri)
            .map(|(_, r)| r.state.total_weight())
            .product();
        if others.is_finite() && others > 0.0 && (others - 1.0).abs() > 1e-15 {
            let target = (0..self.regions.len())
                .filter(|&k| k != ri)
                .min_by_key(|&k| self.regions[k].state.nonzero_count())
                .expect("weight other than 1 means another region exists");
            let k = 1.0 / others.sqrt();
            let scaled: Vec<(u64, S)> = self.regions[target]
                .entries()
                .into_iter()
                .map(|(i, a)| (i, a.scale(k)))
                .collect();
            self.regions[target].state.load(&scaled)?;
        }
        self.split_projected(site, digit)?;
        Ok(digit)
    }

    fn measure(&mut self, qubit: usize, rng: &mut Prng) -> Result<bool> {
        if qubit >= self.space.sites() {
            return Err(Error::QubitOutOfRange {
                qubit,
                num_qubits: self.space.sites(),
            });
        }
        require_binary(&self.space, &[qubit], "measure")?;
        Ok(self.measure_site(qubit, rng)? == 1)
    }

    fn amplitude(&self, index: u64) -> S {
        if index as u128 >= self.space.dim() {
            return S::zero();
        }
        let mut amp = S::one();
        for r in &self.regions {
            amp = amp * r.state.amplitude(r.local_index(&self.space, index));
        }
        amp
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        let parts: Vec<Vec<(u64, S)>> = self
            .regions
            .iter()
            .map(|r| {
                r.entries()
                    .into_iter()
                    .map(|(i, a)| (r.global_part(&self.space, i), a))
                    .collect()
            })
            .collect();
        fn recurse<S: Scalar>(
            parts: &[Vec<(u64, S)>],
            index: u64,
            amp: S,
            f: &mut dyn FnMut(u64, S),
        ) {
            match parts.split_first() {
                None => f(index, amp),
                Some((head, tail)) => {
                    for &(i, a) in head {
                        recurse(tail, index + i, amp * a, f);
                    }
                }
            }
        }
        recurse(&parts, 0, S::one(), f);
    }

    fn nonzero_count(&self) -> usize {
        self.regions
            .iter()
            .map(|r| r.state.nonzero_count())
            .fold(1usize, |acc, c| acc.saturating_mul(c))
    }

    fn total_abs_sqr(&self) -> f64 {
        self.regions
            .iter()
            .map(|r| r.state.total_abs_sqr())
            .product()
    }

    fn total_weight(&self) -> f64 {
        if S::DIVISION {
            return self
                .regions
                .iter()
                .map(|r| r.state.total_weight())
                .product();
        }
        let mut total = 0.0;
        self.for_each_nonzero(&mut |_, a| total += a.born_weight());
        total
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        let digit = usize::from(outcome);
        let ri = self.site_region[qubit];
        let p = self.regions[ri].position(qubit);
        self.regions[ri].state.project(p, outcome, renorm);
        // The exact split only re-keys what the region already holds.
        let _ = self.split_projected(qubit, digit);
    }

    fn reset(&mut self) {
        let space = self.space.clone();
        let policy = self.policy.clone();
        let (auto_split, split_tol) = (self.auto_split, self.split_tol);
        *self = Self::new(space).expect("the space was admitted once already");
        self.policy = policy;
        self.auto_split = auto_split;
        self.split_tol = split_tol;
    }

    /// Loaded amplitudes are one region over every site (structure is not
    /// guessed from a flat list).
    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        for &(i, _) in entries {
            if i as u128 >= self.space.dim() {
                return Err(Error::InvalidState(format!(
                    "index {i} is outside the register of {}",
                    self.space
                )));
            }
        }
        let name = self.policy.elect(&self.space, entries.len());
        let state = native(name, self.space.clone(), entries)?;
        self.regions = vec![Region {
            sites: (0..self.space.sites()).collect(),
            local: self.space.clone(),
            state,
        }];
        self.rebuild();
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        self.regions
            .iter()
            .map(|r| {
                r.state.memory_bytes()
                    + r.sites.capacity() * std::mem::size_of::<usize>()
                    + std::mem::size_of::<Region<S>>()
            })
            .sum::<usize>()
            + self.site_region.capacity() * std::mem::size_of::<usize>()
            + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::C64;

    #[test]
    fn grow_is_a_singleton_region() {
        let mut c = CompositeState::<C64>::new(Space::new(&[3, 2]).unwrap()).unwrap();
        c.transition(&Transition::Grow { at: 1, radix: 5 }, false)
            .unwrap();
        assert_eq!(c.space().radices(), &[3, 5, 2]);
        assert_eq!(c.region_count(), 3);
        assert!(c.events().is_empty());
        assert!((c.amplitude(0) - C64::new(1.0, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn quaternions_are_refused_by_name() {
        let err = CompositeState::<crate::scalar::Quaternion>::new(Space::qubits(2))
            .err()
            .unwrap();
        assert!(err.to_string().contains("commutative"), "{err}");
    }
}
