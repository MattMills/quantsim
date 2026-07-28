//! Mixed-arity compound qudit registers: representation, interaction
//! and flow as three explicit, separately-measured layers.
//!
//! The two problems the design keeps apart:
//!
//! 1. **How information is maintained** — the representational layer.
//!    A [`CompoundRegister`] is a set of sites of *arbitrary* arity
//!    (binary, ternary, quaternary, quintary, …), stored as
//!    **horizontal volumes**: every site starts as its own independent
//!    amplitude volume, and volumes stay separate until an interaction
//!    genuinely correlates them. "Correlated but independent in a
//!    horizontal set" is the resting state of the representation, not
//!    an approximation of it.
//! 2. **How information meets information** — the interaction layer.
//!    Cross-volume gates merge exactly the volumes they touch (guard-
//!    admitted, so over-scale merges refuse with measured numbers),
//!    and the coupling fabric between sites is an explicit object
//!    ([`fabric`]) with measured bond density and one structural
//!    consequence of mixed arity: **swap-routing only exists between
//!    equal-dimension sites** (a swap between a ternary and a quintary
//!    site is not even dimensionally well-formed), so the fabric
//!    decomposes into swap classes and every cross-arity bond is
//!    forced native — [`FabricReport`] measures the split.
//!
//! The **flow** through the full structure is recorded, not inferred:
//! every interaction is logged, [`CompoundRegister::merge_timeline`]
//! shows when volumes correlated, and
//! [`CompoundRegister::reachable_from`] replays the log as the
//! interaction cone of a site.
//!
//! Gates are plain unitary matrices over mixed-radix sub-indices
//! (validated at application): [`fourier_d`] (the d-level DFT — the
//! qudit Hadamard), the generalized Pauli pair [`shift_d`]/[`clock_d`]
//! (`Z_d X_d = ω X_d Z_d`, tested), the entangler [`cshift`]
//! (`|a, b⟩ ↦ |a, b + a mod d_b⟩` across any arity pair), and
//! [`swap_dd`] for equal dims only.
//!
//! The E8 connection ([`crate::e8`]): a compound qudit of arities
//! (2, 3, 4, 5) takes its coupling fabric from the measured overlap of
//! the su(2)/su(3)/su(4)/su(5) root chains inside the one E8 system —
//! chains that provably cannot all be orthogonal (rank 10 > 8), so
//! the geometry itself dictates which sub-qudits are correlated.

use crate::error::{Error, Result};
use crate::guard;
use crate::rng::Prng;
use crate::scalar::C64;
use std::collections::HashMap;

/// The d-level discrete Fourier transform (the qudit Hadamard):
/// `F[r][c] = ω^{rc}/√d` with `ω = e^{2πi/d}`. Row-major, `d × d`.
pub fn fourier_d(d: usize) -> Vec<C64> {
    let norm = 1.0 / (d as f64).sqrt();
    let mut m = vec![C64::new(0.0, 0.0); d * d];
    for r in 0..d {
        for c in 0..d {
            let angle = std::f64::consts::TAU * (r * c % d) as f64 / d as f64;
            m[r * d + c] = C64::new(angle.cos() * norm, angle.sin() * norm);
        }
    }
    m
}

/// The generalized Pauli shift `X_d |j⟩ = |j + 1 mod d⟩`.
pub fn shift_d(d: usize) -> Vec<C64> {
    let mut m = vec![C64::new(0.0, 0.0); d * d];
    for j in 0..d {
        m[((j + 1) % d) * d + j] = C64::new(1.0, 0.0);
    }
    m
}

/// The generalized Pauli clock `Z_d |j⟩ = ω^j |j⟩`.
pub fn clock_d(d: usize) -> Vec<C64> {
    let mut m = vec![C64::new(0.0, 0.0); d * d];
    for j in 0..d {
        let angle = std::f64::consts::TAU * j as f64 / d as f64;
        m[j * d + j] = C64::new(angle.cos(), angle.sin());
    }
    m
}

/// The mixed-arity entangler `|a, b⟩ ↦ |a, b + a mod d_b⟩` (control is
/// the LOW sub-digit, site `a`); well-formed for ANY arity pair —
/// unlike swap. Row-major over sub-index `a + d_a · b`.
pub fn cshift(da: usize, db: usize) -> Vec<C64> {
    let dim = da * db;
    let mut m = vec![C64::new(0.0, 0.0); dim * dim];
    for a in 0..da {
        for b in 0..db {
            let from = a + da * b;
            let to = a + da * ((b + a) % db);
            m[to * dim + from] = C64::new(1.0, 0.0);
        }
    }
    m
}

/// Swap of two EQUAL-dimension sites — the only arity pattern for
/// which swap exists at all (the structural fact behind
/// [`FabricReport::swap_classes`]).
pub fn swap_dd(d: usize) -> Vec<C64> {
    let dim = d * d;
    let mut m = vec![C64::new(0.0, 0.0); dim * dim];
    for a in 0..d {
        for b in 0..d {
            m[(b + d * a) * dim + (a + d * b)] = C64::new(1.0, 0.0);
        }
    }
    m
}

fn check_unitary(dim: usize, m: &[C64]) -> Result<()> {
    if m.len() != dim * dim {
        return Err(Error::BadDimension {
            expected: dim * dim,
            got: m.len(),
        });
    }
    for c1 in 0..dim {
        for c2 in c1..dim {
            let mut acc = C64::new(0.0, 0.0);
            for r in 0..dim {
                acc += m[r * dim + c1].conj() * m[r * dim + c2];
            }
            let expected = if c1 == c2 { 1.0 } else { 0.0 };
            if (acc - C64::new(expected, 0.0)).norm() > 1e-9 {
                return Err(Error::InvalidState(format!(
                    "matrix is not unitary at columns ({c1}, {c2})"
                )));
            }
        }
    }
    Ok(())
}

/// One recorded interaction — the unit of information flow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interaction {
    /// Application step (0-based over the register's lifetime).
    pub step: usize,
    /// Sites the gate touched.
    pub sites: Vec<usize>,
    /// Whether this interaction merged previously-independent volumes.
    pub merged: bool,
}

/// One independent amplitude volume over a subset of sites.
#[derive(Debug, Clone)]
struct Volume {
    /// Member sites, ascending.
    sites: Vec<usize>,
    /// Sparse amplitudes over the packed mixed-radix member index.
    amps: HashMap<u64, C64>,
}

impl Volume {
    fn stride_of(&self, dims: &[usize], site: usize) -> (u64, usize) {
        let mut stride = 1u64;
        for &s in &self.sites {
            if s == site {
                return (stride, dims[s]);
            }
            stride *= dims[s] as u64;
        }
        unreachable!("site not in volume");
    }

    fn states(&self, dims: &[usize]) -> u128 {
        self.sites.iter().map(|&s| dims[s] as u128).product()
    }
}

/// A mixed-arity register held as horizontal volumes. See the module
/// docs for the representation/interaction/flow separation.
pub struct CompoundRegister {
    dims: Vec<usize>,
    volumes: Vec<Volume>,
    site_volume: Vec<usize>,
    log: Vec<Interaction>,
    steps: usize,
}

impl CompoundRegister {
    /// A register with the given site arities (each ≥ 2), every site
    /// its own volume in state |0⟩.
    pub fn new(dims: &[usize]) -> Result<Self> {
        if dims.iter().any(|&d| d < 2) {
            return Err(Error::InvalidState(
                "every site needs at least two levels".into(),
            ));
        }
        let volumes = (0..dims.len())
            .map(|s| Volume {
                sites: vec![s],
                amps: HashMap::from([(0u64, C64::new(1.0, 0.0))]),
            })
            .collect();
        Ok(CompoundRegister {
            dims: dims.to_vec(),
            volumes,
            site_volume: (0..dims.len()).collect(),
            log: Vec::new(),
            steps: 0,
        })
    }

    /// Site arities.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Number of independent volumes right now.
    pub fn volumes(&self) -> usize {
        self.volumes.len()
    }

    /// The sites of each volume — the horizontal set.
    pub fn volume_sites(&self) -> Vec<Vec<usize>> {
        self.volumes.iter().map(|v| v.sites.clone()).collect()
    }

    /// Basis states spanned by the largest volume — the measure of how
    /// much of the register has been drawn into one correlation.
    pub fn largest_volume_states(&self) -> u128 {
        self.volumes
            .iter()
            .map(|v| v.states(&self.dims))
            .max()
            .unwrap_or(1)
    }

    /// Stored amplitude entries across all volumes.
    pub fn stored_entries(&self) -> usize {
        self.volumes.iter().map(|v| v.amps.len()).sum()
    }

    /// Approximate stored bytes.
    pub fn memory_bytes(&self) -> usize {
        self.volumes
            .iter()
            .map(|v| v.sites.len() * 8 + v.amps.len() * 24 + 48)
            .sum()
    }

    /// The recorded flow: every interaction, in order, with whether it
    /// merged volumes.
    pub fn merge_timeline(&self) -> &[Interaction] {
        &self.log
    }

    /// Replay the log as the interaction cone of `site`: everything a
    /// perturbation there could have reached (interactions carry
    /// influence both ways — this is the conservative bound).
    pub fn reachable_from(&self, site: usize) -> Vec<bool> {
        let mut reach = vec![false; self.dims.len()];
        reach[site] = true;
        for i in &self.log {
            if i.sites.iter().any(|&s| reach[s]) {
                for &s in &i.sites {
                    reach[s] = true;
                }
            }
        }
        reach
    }

    /// Total Born weight (product over volumes; 1 for unitary flows).
    pub fn born_weight(&self) -> f64 {
        self.volumes
            .iter()
            .map(|v| v.amps.values().map(|a| a.norm_sqr()).sum::<f64>())
            .product()
    }

    /// Amplitude of the joint basis state `basis` (one digit per site,
    /// `basis[s] < dims[s]`) — the product over volumes.
    pub fn amplitude(&self, basis: &[usize]) -> Result<C64> {
        self.check_basis(basis)?;
        let mut out = C64::new(1.0, 0.0);
        for v in &self.volumes {
            let mut idx = 0u64;
            let mut stride = 1u64;
            for &s in &v.sites {
                idx += basis[s] as u64 * stride;
                stride *= self.dims[s] as u64;
            }
            out *= v.amps.get(&idx).copied().unwrap_or(C64::new(0.0, 0.0));
        }
        Ok(out)
    }

    /// Probability of the joint basis state.
    pub fn probability(&self, basis: &[usize]) -> Result<f64> {
        Ok(self.amplitude(basis)?.norm_sqr())
    }

    fn check_basis(&self, basis: &[usize]) -> Result<()> {
        if basis.len() != self.dims.len() {
            return Err(Error::BadDimension {
                expected: self.dims.len(),
                got: basis.len(),
            });
        }
        for (s, (&b, &d)) in basis.iter().zip(&self.dims).enumerate() {
            if b >= d {
                return Err(Error::InvalidState(format!(
                    "digit {b} out of range for the {d}-level site {s}"
                )));
            }
        }
        Ok(())
    }

    fn check_site(&self, site: usize) -> Result<()> {
        if site >= self.dims.len() {
            return Err(Error::QubitOutOfRange {
                qubit: site,
                num_qubits: self.dims.len(),
            });
        }
        Ok(())
    }

    /// Apply a `d × d` unitary to one site.
    pub fn apply_1(&mut self, site: usize, matrix: &[C64]) -> Result<()> {
        self.check_site(site)?;
        let d = self.dims[site];
        check_unitary(d, matrix)?;
        let v = self.site_volume[site];
        let (stride, _) = self.volumes[v].stride_of(&self.dims, site);
        let mut grouped: HashMap<u64, Vec<C64>> = HashMap::new();
        for (&idx, &amp) in &self.volumes[v].amps {
            let digit = (idx / stride) % d as u64;
            let rest = idx - digit * stride;
            grouped
                .entry(rest)
                .or_insert_with(|| vec![C64::new(0.0, 0.0); d])[digit as usize] = amp;
        }
        let mut amps = HashMap::new();
        for (rest, vec_in) in grouped {
            for r in 0..d {
                let mut acc = C64::new(0.0, 0.0);
                for (c, amp) in vec_in.iter().enumerate() {
                    acc += matrix[r * d + c] * amp;
                }
                if acc.norm_sqr() > 0.0 {
                    amps.insert(rest + r as u64 * stride, acc);
                }
            }
        }
        self.volumes[v].amps = amps;
        self.log.push(Interaction {
            step: self.steps,
            sites: vec![site],
            merged: false,
        });
        self.steps += 1;
        Ok(())
    }

    /// Apply a `(d_a·d_b) × (d_a·d_b)` unitary to a site pair (site
    /// `a` is the LOW sub-digit). Merges the two volumes when they are
    /// still independent — the recorded correlation event.
    pub fn apply_2(&mut self, a: usize, b: usize, matrix: &[C64]) -> Result<()> {
        self.check_site(a)?;
        self.check_site(b)?;
        if a == b {
            return Err(Error::DuplicateQubits { qubits: vec![a, b] });
        }
        let (da, db) = (self.dims[a], self.dims[b]);
        check_unitary(da * db, matrix)?;
        let merged = self.site_volume[a] != self.site_volume[b];
        if merged {
            self.merge(self.site_volume[a], self.site_volume[b])?;
        }
        let v = self.site_volume[a];
        let (stride_a, _) = self.volumes[v].stride_of(&self.dims, a);
        let (stride_b, _) = self.volumes[v].stride_of(&self.dims, b);
        let mut grouped: HashMap<u64, Vec<C64>> = HashMap::new();
        for (&idx, &amp) in &self.volumes[v].amps {
            let xa = (idx / stride_a) % da as u64;
            let xb = (idx / stride_b) % db as u64;
            let rest = idx - xa * stride_a - xb * stride_b;
            let sub = xa as usize + da * xb as usize;
            grouped
                .entry(rest)
                .or_insert_with(|| vec![C64::new(0.0, 0.0); da * db])[sub] = amp;
        }
        let dim = da * db;
        let mut amps = HashMap::new();
        for (rest, vec_in) in grouped {
            for r in 0..dim {
                let mut acc = C64::new(0.0, 0.0);
                for (c, amp) in vec_in.iter().enumerate() {
                    acc += matrix[r * dim + c] * amp;
                }
                if acc.norm_sqr() > 0.0 {
                    let ra = (r % da) as u64;
                    let rb = (r / da) as u64;
                    amps.insert(rest + ra * stride_a + rb * stride_b, acc);
                }
            }
        }
        self.volumes[v].amps = amps;
        self.log.push(Interaction {
            step: self.steps,
            sites: vec![a, b],
            merged,
        });
        self.steps += 1;
        Ok(())
    }

    /// Apply a diagonal phase field to one site: `entries[j]` multiplies
    /// every amplitude whose site digit is `j`. Validated exactly in
    /// O(d) (each phase must be unimodular) — the fast path for
    /// coboundary-style phase storage on wide qudits.
    pub fn apply_1_diagonal(&mut self, site: usize, entries: &[C64]) -> Result<()> {
        self.check_site(site)?;
        let d = self.dims[site];
        if entries.len() != d {
            return Err(Error::BadDimension {
                expected: d,
                got: entries.len(),
            });
        }
        for (j, e) in entries.iter().enumerate() {
            if (e.norm() - 1.0).abs() > 1e-9 {
                return Err(Error::InvalidState(format!(
                    "diagonal entry {j} is not unimodular"
                )));
            }
        }
        let v = self.site_volume[site];
        let (stride, _) = self.volumes[v].stride_of(&self.dims, site);
        for (idx, amp) in self.volumes[v].amps.iter_mut() {
            let digit = ((idx / stride) % d as u64) as usize;
            *amp *= entries[digit];
        }
        self.log.push(Interaction {
            step: self.steps,
            sites: vec![site],
            merged: false,
        });
        self.steps += 1;
        Ok(())
    }

    /// Apply a classical reversible map to a site pair:
    /// `perm(a, b) = (a', b')` must be a bijection on the digit pairs —
    /// validated exactly in O(d_a·d_b), which is what makes wide-qudit
    /// permutation gates (a 240-level `cshift`, its inverse, a swap)
    /// affordable where a dense unitarity check would not be. Merges
    /// volumes like any interaction.
    pub fn apply_2_permutation(
        &mut self,
        a: usize,
        b: usize,
        perm: &dyn Fn(usize, usize) -> (usize, usize),
    ) -> Result<()> {
        self.check_site(a)?;
        self.check_site(b)?;
        if a == b {
            return Err(Error::DuplicateQubits { qubits: vec![a, b] });
        }
        let (da, db) = (self.dims[a], self.dims[b]);
        let mut seen = vec![false; da * db];
        for xa in 0..da {
            for xb in 0..db {
                let (ya, yb) = perm(xa, xb);
                if ya >= da || yb >= db {
                    return Err(Error::InvalidState(format!(
                        "permutation image ({ya}, {yb}) out of range"
                    )));
                }
                let slot = ya + da * yb;
                if seen[slot] {
                    return Err(Error::InvalidState(
                        "map is not a bijection on digit pairs".into(),
                    ));
                }
                seen[slot] = true;
            }
        }
        let merged = self.site_volume[a] != self.site_volume[b];
        if merged {
            self.merge(self.site_volume[a], self.site_volume[b])?;
        }
        let v = self.site_volume[a];
        let (stride_a, _) = self.volumes[v].stride_of(&self.dims, a);
        let (stride_b, _) = self.volumes[v].stride_of(&self.dims, b);
        let amps = std::mem::take(&mut self.volumes[v].amps);
        let mut out = HashMap::with_capacity(amps.len());
        for (idx, amp) in amps {
            let xa = ((idx / stride_a) % da as u64) as usize;
            let xb = ((idx / stride_b) % db as u64) as usize;
            let (ya, yb) = perm(xa, xb);
            let new = idx - (xa as u64) * stride_a - (xb as u64) * stride_b
                + (ya as u64) * stride_a
                + (yb as u64) * stride_b;
            out.insert(new, amp);
        }
        self.volumes[v].amps = out;
        self.log.push(Interaction {
            step: self.steps,
            sites: vec![a, b],
            merged,
        });
        self.steps += 1;
        Ok(())
    }

    fn merge(&mut self, va: usize, vb: usize) -> Result<()> {
        let (keep, drop) = (va.min(vb), va.max(vb));
        // Admission first: the merged volume's worst case is the entry
        // product — measured against the guard, never presumed fine.
        let bytes = self.volumes[keep]
            .amps
            .len()
            .saturating_mul(self.volumes[drop].amps.len())
            .saturating_mul(24);
        guard::admit_growth(bytes, "compound volume merge")?;
        let dropped = self.volumes.remove(drop);
        let kept = &mut self.volumes[keep];
        let mut sites = kept.sites.clone();
        sites.extend(&dropped.sites);
        sites.sort_unstable();
        let mut amps = HashMap::with_capacity(kept.amps.len() * dropped.amps.len());
        let repack = |old: &Volume, idx: u64, dims: &[usize], sites: &[usize]| -> u64 {
            let mut out = 0u64;
            let mut old_stride = 1u64;
            for &s in &old.sites {
                let digit = (idx / old_stride) % dims[s] as u64;
                old_stride *= dims[s] as u64;
                let mut new_stride = 1u64;
                for &t in sites {
                    if t == s {
                        break;
                    }
                    new_stride *= dims[t] as u64;
                }
                out += digit * new_stride;
            }
            out
        };
        let kept_snapshot: Vec<(u64, C64)> = kept.amps.iter().map(|(&i, &a)| (i, a)).collect();
        for (i1, a1) in kept_snapshot {
            let base = repack(kept, i1, &self.dims, &sites);
            for (&i2, &a2) in &dropped.amps {
                let add = repack(&dropped, i2, &self.dims, &sites);
                amps.insert(base + add, a1 * a2);
            }
        }
        kept.sites = sites;
        kept.amps = amps;
        for sv in self.site_volume.iter_mut() {
            if *sv == drop {
                *sv = keep;
            } else if *sv > drop {
                *sv -= 1;
            }
        }
        Ok(())
    }

    /// Deterministic Born sampling of the full register: independent
    /// volumes sample independently (that is what independence means),
    /// and each shot is a digit string over all sites.
    pub fn sample(&self, shots: u64, rng: &mut Prng) -> Result<HashMap<Vec<usize>, u64>> {
        let mut per_volume: Vec<Vec<(u64, f64)>> = Vec::new();
        for v in &self.volumes {
            let mut entries: Vec<(u64, f64)> = v
                .amps
                .iter()
                .map(|(&i, &a)| (i, a.norm_sqr()))
                .filter(|&(_, w)| w > 0.0)
                .collect();
            entries.sort_unstable_by_key(|&(i, _)| i);
            let mut acc = 0.0;
            for e in &mut entries {
                acc += e.1;
                e.1 = acc;
            }
            if acc <= 0.0 || !acc.is_finite() {
                return Err(Error::InvalidState(format!(
                    "volume weight {acc} is not positive; cannot sample"
                )));
            }
            per_volume.push(entries);
        }
        let mut counts: HashMap<Vec<usize>, u64> = HashMap::new();
        for _ in 0..shots {
            let mut outcome = vec![0usize; self.dims.len()];
            for (v, entries) in self.volumes.iter().zip(&per_volume) {
                let total = entries.last().unwrap().1;
                let u = rng.next_f64() * total;
                let hit = entries.partition_point(|&(_, c)| c <= u);
                let idx = entries[hit.min(entries.len() - 1)].0;
                let mut stride = 1u64;
                for &s in &v.sites {
                    outcome[s] = ((idx / stride) % self.dims[s] as u64) as usize;
                    stride *= self.dims[s] as u64;
                }
            }
            *counts.entry(outcome).or_insert(0) += 1;
        }
        Ok(counts)
    }
}

/// The interaction fabric of a mixed-arity register, measured.
#[derive(Debug, Clone)]
pub struct FabricReport {
    /// The declared bonds.
    pub bonds: Vec<(usize, usize)>,
    /// Bond density: bonds / possible pairs.
    pub density: f64,
    /// Swap classes: connected components of the EQUAL-dimension
    /// sub-fabric — the only regions where swap-routing exists at all.
    pub swap_classes: Vec<Vec<usize>>,
    /// Bonds between different arities: dimensionally unable to swap,
    /// forced to interact natively.
    pub cross_arity_bonds: usize,
}

/// Measure a coupling fabric over mixed-arity sites: density, the
/// swap-class decomposition, and the cross-arity bonds that can never
/// route.
pub fn fabric(dims: &[usize], bonds: &[(usize, usize)]) -> Result<FabricReport> {
    let n = dims.len();
    for &(a, b) in bonds {
        if a >= n || b >= n || a == b {
            return Err(Error::InvalidState(format!("bad bond ({a}, {b})")));
        }
    }
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            let root = find(parent, parent[x]);
            parent[x] = root;
        }
        parent[x]
    }
    let mut cross = 0;
    for &(a, b) in bonds {
        if dims[a] == dims[b] {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            parent[ra] = rb;
        } else {
            cross += 1;
        }
    }
    let mut classes: HashMap<usize, Vec<usize>> = HashMap::new();
    for s in 0..n {
        let root = find(&mut parent, s);
        classes.entry(root).or_default().push(s);
    }
    let mut swap_classes: Vec<Vec<usize>> = classes.into_values().collect();
    swap_classes.sort();
    let possible = n * (n - 1) / 2;
    Ok(FabricReport {
        bonds: bonds.to_vec(),
        density: if possible == 0 {
            0.0
        } else {
            bonds.len() as f64 / possible as f64
        },
        swap_classes,
        cross_arity_bonds: cross,
    })
}
