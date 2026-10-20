//! Mixed-radix state backends: the dense and sparse representations over a
//! [`Space`] of arbitrary per-site radices.
//!
//! [`RadixDenseState`] is a flat `Π r_k` amplitude vector and
//! [`RadixSparseState`] a hash map of nonzero amplitudes, both indexed
//! little-endian mixed radix (see [`crate::space`]). Site-level operations
//! are native ([`Backend::apply_sites`], [`Backend::project_digit`],
//! [`Backend::measure_site`]); the qubit-level ones (`apply`,
//! `apply_diagonal`, `project`, `measure`) act on radix-2 sites and refuse
//! any other target with [`Error::UnsupportedForSpace`].
//!
//! They are separate types from [`DenseState`](super::DenseState) and
//! [`SparseState`](super::SparseState) on purpose: the qubit backends keep
//! their bit-twiddling kernels and their measured footprints untouched,
//! and these register under the same names as **space** constructors
//! ([`BackendRegistry::create_with_space`](super::BackendRegistry::create_with_space)),
//! so `run_space("dense", …)` reaches the mixed-radix dense store. On a
//! qubit space they compute exactly what the qubit backends compute — the
//! tests certify that — only through the general kernel.

use rustc_hash::FxHashMap;

use super::{validate_apply, validate_apply_diagonal, Backend, PRUNE_TOL};
use crate::error::{Error, Result};
use crate::math::{GateMatrix, MATRIX_ZERO_TOL};
use crate::scalar::Scalar;
use crate::space::{apply_sites_in_place, require_binary, validate_sites, SiteMatrix, Space};

/// Heap bytes a [`Space`] holds (its radix and stride tables).
fn space_bytes(space: &Space) -> usize {
    space.sites() * (std::mem::size_of::<usize>() + std::mem::size_of::<u64>())
}

fn check_digit(space: &Space, site: usize, digit: usize) -> Result<()> {
    space.validate_sites(&[site])?;
    if digit >= space.radix(site) {
        return Err(Error::InvalidState(format!(
            "digit {digit} out of range for site {site} of radix {}",
            space.radix(site)
        )));
    }
    Ok(())
}

fn check_load<S: Scalar>(space: &Space, entries: &[(u64, S)]) -> Result<()> {
    for &(i, _) in entries {
        if i as u128 >= space.dim() {
            return Err(Error::InvalidState(format!(
                "index {i} is outside the register of {space}"
            )));
        }
    }
    Ok(())
}

/// A flat mixed-radix amplitude vector. See the module docs.
#[derive(Debug, Clone)]
pub struct RadixDenseState<S: Scalar> {
    space: Space,
    amps: Vec<S>,
}

impl<S: Scalar> RadixDenseState<S> {
    /// `|0…0⟩` over `space`: `Π r_k` amplitudes, admitted by the resource
    /// guard like any dense allocation.
    pub fn new(space: Space) -> Result<Self> {
        let len = usize::try_from(space.dim()).map_err(|_| {
            Error::InvalidState(format!("dense state: {space} does not fit memory indices"))
        })?;
        let mut amps = crate::guard::try_vec(len, S::zero(), &format!("dense state ({space})"))?;
        amps[0] = S::one();
        Ok(RadixDenseState { space, amps })
    }

    /// Direct read-only view of the amplitude vector.
    pub fn amplitudes(&self) -> &[S] {
        &self.amps
    }

    fn project_onto(&mut self, site: usize, digit: usize, renorm: f64) {
        for (i, a) in self.amps.iter_mut().enumerate() {
            if self.space.digit(i as u64, site) == digit {
                *a = a.scale(renorm);
            } else {
                *a = S::zero();
            }
        }
    }
}

impl<S: Scalar> Backend<S> for RadixDenseState<S> {
    fn name(&self) -> &str {
        "radix-dense"
    }

    fn num_qubits(&self) -> usize {
        self.space.sites()
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        validate_apply(self.space.sites(), matrix, qubits)?;
        require_binary(&self.space, qubits, "apply")?;
        apply_sites_in_place(
            &mut self.amps,
            &self.space,
            &SiteMatrix::from_gate(matrix),
            qubits,
        )
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        validate_apply_diagonal(self.space.sites(), entries, qubits)?;
        require_binary(&self.space, qubits, "apply_diagonal")?;
        for (i, a) in self.amps.iter_mut().enumerate() {
            *a = entries[self.space.sub_index(i as u64, qubits)] * *a;
        }
        Ok(())
    }

    fn space(&self) -> Space {
        self.space.clone()
    }

    fn apply_sites(&mut self, matrix: &SiteMatrix<S>, sites: &[usize]) -> Result<()> {
        validate_sites(&self.space, matrix, sites)?;
        apply_sites_in_place(&mut self.amps, &self.space, matrix, sites)
    }

    fn project_digit(&mut self, site: usize, digit: usize, renorm: f64) -> Result<()> {
        check_digit(&self.space, site, digit)?;
        self.project_onto(site, digit, renorm);
        Ok(())
    }

    fn amplitude(&self, index: u64) -> S {
        self.amps
            .get(index as usize)
            .copied()
            .unwrap_or_else(S::zero)
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        for (i, &a) in self.amps.iter().enumerate() {
            if a.abs_sqr() > 0.0 {
                f(i as u64, a);
            }
        }
    }

    /// Qubit-level projection: `measure` has already refused non-binary
    /// sites, so the outcome is read as the digit.
    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        self.project_onto(qubit, usize::from(outcome), renorm);
    }

    fn reset(&mut self) {
        for a in self.amps.iter_mut() {
            *a = S::zero();
        }
        self.amps[0] = S::one();
    }

    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        check_load(&self.space, entries)?;
        for a in self.amps.iter_mut() {
            *a = S::zero();
        }
        for &(i, a) in entries {
            self.amps[i as usize] = a;
        }
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        self.amps.capacity() * std::mem::size_of::<S>()
            + space_bytes(&self.space)
            + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Approximate bytes one stored map entry costs (key + value + control
/// byte at hashbrown's 7/8 load factor).
fn entry_bytes<S>() -> usize {
    (std::mem::size_of::<u64>() + std::mem::size_of::<S>() + 1) * 8 / 7
}

/// Nonzero mixed-radix amplitudes in an `FxHashMap<u64, S>`. See the
/// module docs. As for qubit sparse, permutation-like and diagonal site
/// gates (shift, cshift, swap, clock) cause no fill.
#[derive(Debug, Clone)]
pub struct RadixSparseState<S: Scalar> {
    space: Space,
    map: FxHashMap<u64, S>,
}

impl<S: Scalar> RadixSparseState<S> {
    /// `|0…0⟩` over `space`.
    pub fn new(space: Space) -> Result<Self> {
        let mut map = FxHashMap::default();
        map.insert(0u64, S::one());
        Ok(RadixSparseState { space, map })
    }

    /// Read-only view of the stored amplitudes.
    pub fn entries(&self) -> impl Iterator<Item = (u64, S)> + '_ {
        self.map.iter().map(|(&i, &a)| (i, a))
    }

    fn apply_map(&mut self, matrix: &SiteMatrix<S>, sites: &[usize]) -> Result<()> {
        let d = matrix.dim();
        let offsets = self.space.offsets(sites);
        let mdata = matrix.data();
        // Column-wise nonzero structure, as in the qubit sparse kernel.
        let mut cols: Vec<Vec<(usize, S)>> = vec![Vec::new(); d];
        for r in 0..d {
            for (l, col) in cols.iter_mut().enumerate() {
                let e = mdata[r * d + l];
                if e.abs_sqr() > MATRIX_ZERO_TOL {
                    col.push((r, e));
                }
            }
        }
        let max_fill = cols.iter().map(|c| c.len()).max().unwrap_or(1);
        crate::guard::admit_growth(
            self.map
                .len()
                .saturating_mul(max_fill)
                .saturating_mul(entry_bytes::<S>()),
            "sparse state growth",
        )?;
        let mut out: FxHashMap<u64, S> = FxHashMap::default();
        out.reserve(self.map.len());
        for (count, (&idx, &amp)) in self.map.iter().enumerate() {
            if count % (1 << 18) == 0 {
                crate::guard::checkpoint()?;
            }
            let sub = self.space.sub_index(idx, sites);
            let rest = idx - offsets[sub];
            for &(r, e) in &cols[sub] {
                let slot = out.entry(rest + offsets[r]).or_insert_with(S::zero);
                *slot = *slot + e * amp;
            }
        }
        out.retain(|_, a| a.abs_sqr() > PRUNE_TOL);
        self.map = out;
        Ok(())
    }

    fn project_onto(&mut self, site: usize, digit: usize, renorm: f64) {
        let space = &self.space;
        self.map.retain(|&i, _| space.digit(i, site) == digit);
        for a in self.map.values_mut() {
            *a = a.scale(renorm);
        }
    }
}

impl<S: Scalar> Backend<S> for RadixSparseState<S> {
    fn name(&self) -> &str {
        "radix-sparse"
    }

    fn num_qubits(&self) -> usize {
        self.space.sites()
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        validate_apply(self.space.sites(), matrix, qubits)?;
        require_binary(&self.space, qubits, "apply")?;
        self.apply_map(&SiteMatrix::from_gate(matrix), qubits)
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        validate_apply_diagonal(self.space.sites(), entries, qubits)?;
        require_binary(&self.space, qubits, "apply_diagonal")?;
        for (&idx, a) in self.map.iter_mut() {
            *a = entries[self.space.sub_index(idx, qubits)] * *a;
        }
        Ok(())
    }

    fn space(&self) -> Space {
        self.space.clone()
    }

    fn apply_sites(&mut self, matrix: &SiteMatrix<S>, sites: &[usize]) -> Result<()> {
        validate_sites(&self.space, matrix, sites)?;
        self.apply_map(matrix, sites)
    }

    fn project_digit(&mut self, site: usize, digit: usize, renorm: f64) -> Result<()> {
        check_digit(&self.space, site, digit)?;
        self.project_onto(site, digit, renorm);
        Ok(())
    }

    fn amplitude(&self, index: u64) -> S {
        self.map.get(&index).copied().unwrap_or_else(S::zero)
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        for (&i, &a) in &self.map {
            f(i, a);
        }
    }

    fn nonzero_count(&self) -> usize {
        self.map.len()
    }

    /// Qubit-level projection: `measure` has already refused non-binary
    /// sites, so the outcome is read as the digit.
    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        self.project_onto(qubit, usize::from(outcome), renorm);
    }

    fn reset(&mut self) {
        self.map.clear();
        self.map.insert(0, S::one());
    }

    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        check_load(&self.space, entries)?;
        self.map.clear();
        for &(i, a) in entries {
            if a.abs_sqr() > 0.0 {
                self.map.insert(i, a);
            }
        }
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        let slot = std::mem::size_of::<u64>() + std::mem::size_of::<S>() + 1;
        self.map.capacity() * slot * 8 / 7 + space_bytes(&self.space) + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
