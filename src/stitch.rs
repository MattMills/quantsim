//! The stitch: sharding a register by its own symplectic form.
//!
//! [`lateral`](crate::lateral) distributes the *control plane* — the
//! Pauli frame, 27 bytes a tick — and leaves each node holding its own
//! patch. That is the narrow reading of a distributed register, and it
//! is not the one the symplectic form actually supports.
//!
//! The wide one is this. The phase-free Pauli group is an 𝔽₂ vector
//! space carrying an **alternating** form (`⟨a,b⟩ = 0` iff `a` and `b`
//! commute), and every alternating form has a symplectic normal form:
//!
//! ```text
//! span(data) = radical ⊥ H₁ ⊥ H₂ ⊥ … ⊥ H_h
//! ```
//!
//! with the radical isotropic — an abelian subgroup, which is to say a
//! **stabilizer** — and each `Hᵢ` a **hyperbolic pair** `(eᵢ, fᵢ)` with
//! `⟨eᵢ, fᵢ⟩ = 1`. [`orthogonalize`] computes it in `O(rank² · n)`.
//!
//! Two facts follow, and together they are the sharding rule:
//!
//! 1. **The radical carves; the pairs cannot.** `r` commuting
//!    directions cut the `2ⁿ` module to `2^{n−r}` — that is stabilizer
//!    encoding. A hyperbolic pair names no region at all: `eᵢ` and `fᵢ`
//!    anticommute, so **no single abelian subgroup, and therefore no
//!    single node, can hold both**.
//! 2. **So the impossible part partitions.** Each pair forces a binary
//!    choice — extend by `eᵢ` *or* by `fᵢ`, never both — giving exactly
//!    `2^h` maximal isotropic extensions ([`Stitch::selections`]), each
//!    of rank `r + h`, each carving `2^{n−r−h}`. Two distinct
//!    selections contain both `eᵢ` and `fᵢ` for some `i`, and a vector
//!    fixed by both would satisfy `v = ½(eᵢfᵢ + fᵢeᵢ)v = 0`, so their
//!    regions meet in zero. The `2^h` slices are a **direct sum**:
//!
//!    ```text
//!    2^h · 2^{n−r−h}  =  2^{n−r}
//!    ```
//!
//!    exact, not approximate — [`Stitch::closes`].
//!
//! ## What that is, said plainly
//!
//! A `2^{n−r}`-dimensional code space is the direct sum of `2^h` slices,
//! **each of which is a single stabilizer state** — a rank-`n` tableau,
//! `O(n²)` bits, with no amplitude stored anywhere. Give each slice to a
//! node and no node holds `2ⁿ`, or `2^k`, or anything exponential: it
//! holds a tableau and one coefficient. The exponential moves out of
//! per-node memory and into the **node count**, which is what
//! distributing an exponential object means.
//!
//! `h` is the number of lateral axes, and `log₂` of the node count. It
//! is also, exactly, the number of logical qubits: for a code with `r`
//! independent stabilizer generators on `n` qubits,
//! [`orthogonalize`] over the full normalizer returns `radical_rank = r`
//! and `witt = n − r = k`, **measured off the generators** rather than
//! asserted — `tests/stitch.rs` pins it on the toric and surface
//! families.
//!
//! ## Why this is cross-*lateral*
//!
//! The pair `(eᵢ, fᵢ)` is the lateral axis, and it is lateral precisely
//! because it is *impossible*: anticommutation is what forbids one node
//! from holding both halves, and therefore what forces the split. The
//! partition is not imposed on the algebra from outside — it is the
//! algebra's own obstruction, read as a shard boundary.
//! [`Stitch::hidden_bits`] is `h`, and the name is earned: `2^{n−r}` is
//! the region's dimension whatever `h` is, so **the size of the object
//! does not reveal how many slices it is in**.
//!
//! ## The accounting, all three columns
//!
//! Per node: one rank-`(r+h)` tableau at `2n` bits a row, plus one
//! coefficient. Across the network: `2^h` of them. Measured by
//! [`ShardPlan`], on shapes rather than on enumerations:
//!
//! ```text
//!   n    r    h |     nodes | per node |     network |    monolithic
//!  18   16    2 |         4 |    106 B |       424 B |       4.19 MB
//!  30   20    5 |        32 |    216 B |     6.91 kB |      17.18 GB
//!  30   20   10 |      1024 |    256 B |    262.1 kB |      17.18 GB
//!  40   20   20 |   1048576 |    416 B |    436.2 MB |      17.59 TB
//! ```
//!
//! Both of the interesting columns move. **Per node** the cost is
//! polynomial in `n` and *does not mention the node count* — 1024 nodes
//! and 32 nodes on the same register differ by 40 bytes each. And the
//! **network total** is `2^h · O(n²)`, exponential in the *logical*
//! count `h`, not in the register width `n`: a code with `r` stabilizer
//! generators has already pulled the exponent from `n` down to
//! `k = n − r`, and the shard plan simply spends it as machines instead
//! of as memory.
//!
//! What it costs, stated: each node carries a whole tableau to hold one
//! coefficient, so the network holds `O(n²)` bytes per coefficient
//! where a bare `2^k` amplitude vector would hold 16. That is the
//! trade — a factor of `n²/64` in total for a per-node object that is
//! polynomial, *closed* (a Clifford acts on a slice's tableau in
//! `O(n²)` locally, with no communication), and independent of every
//! other node. What crosses the wire is coefficient traffic plus
//! [`lateral`](crate::lateral)'s frame.
//!
//! (The construction is `cliff-core`'s `stitch` and `volume`, whose
//! documentation states the correspondence outright — the twist form is
//! the Pauli group's commutator phase, an isotropic volume *is* a
//! stabilizer, a centraliser *is* a normalizer. This module is that
//! decomposition over [`PauliString`], where the slices can be
//! instantiated as actual states and the direct sum checked on
//! amplitudes rather than on dimensions.)

use crate::backend::PauliString;
use crate::error::{Error, Result};

/// Widest register the 63-qubit [`PauliString`] masks address.
pub const MAX_QUBITS: usize = 63;

/// Most hyperbolic pairs [`Stitch::selections`] will enumerate: it
/// returns `2^h` volumes, and the whole point of `h` is that it is an
/// exponent.
pub const MAX_ENUMERATED_PAIRS: usize = 16;

// ────────────────────────────── volumes ──────────────────────────────

/// A region of the Pauli group as an 𝔽₂ subspace: the span of a set of
/// strings, phases discarded, kept in echelon form.
///
/// Named properly, this is a **flat of `PG(2n−1, 2)`**, and an
/// [`is_isotropic`](Self::is_isotropic) one is a flat of the polar
/// space `W(2n−1, 2)` — see [`polar`](crate::polar), which carries the
/// counts and the incidence structure. At `n = 2` that polar space is
/// the doily `GQ(2,2)`, and a maximal isotropic volume there is one of
/// its 15 lines.
///
/// Everything worth asking is linear algebra over one bit, and **none
/// of the costs mention `2ⁿ`** — a volume of rank 40 inside a 60-qubit
/// register contains `2⁴⁰` strings and is a 40-row bit matrix.
///
/// | question | here | in stabilizer language |
/// |---|---|---|
/// | [`is_isotropic`](Self::is_isotropic) | every pair commutes | an abelian subgroup — a stabilizer |
/// | [`centraliser`](Self::centraliser) | commutes with all of mine | the normalizer of that subgroup |
/// | [`is_maximal_isotropic`](Self::is_maximal_isotropic) | isotropic of rank `n` | a maximal commuting set: one state |
/// | [`meet`](Self::meet) | what we share | the common subgroup |
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Volume {
    n: usize,
    rows: Vec<(u64, u64)>,
    pivots: Vec<usize>,
}

fn leading(x: u64, z: u64) -> Option<usize> {
    if x != 0 {
        Some(x.trailing_zeros() as usize)
    } else if z != 0 {
        Some(MAX_QUBITS + 1 + z.trailing_zeros() as usize)
    } else {
        None
    }
}

/// The symplectic form: `0` when the two strings commute, `1` when they
/// anticommute. Bilinear over 𝔽₂ and alternating (`⟨a,a⟩ = 0`), which
/// is the whole reason the normal form below exists.
pub fn symplectic(a: (u64, u64), b: (u64, u64)) -> u32 {
    ((a.1 & b.0).count_ones() + (a.0 & b.1).count_ones()) & 1
}

impl Volume {
    /// The empty volume on `n` qubits.
    pub fn empty(n: usize) -> Result<Self> {
        if n == 0 || n > MAX_QUBITS {
            return Err(Error::InvalidState(format!(
                "volume: {n} qubits outside 1..={MAX_QUBITS}"
            )));
        }
        Ok(Volume {
            n,
            rows: Vec::new(),
            pivots: Vec::new(),
        })
    }

    /// The span of a set of strings, phases discarded.
    pub fn span(n: usize, gens: &[PauliString]) -> Result<Self> {
        let mut v = Volume::empty(n)?;
        for g in gens {
            v.insert((g.x, g.z));
        }
        Ok(v)
    }

    fn reduce(&self, mut x: u64, mut z: u64) -> (u64, u64) {
        while let Some(lead) = leading(x, z) {
            match self.pivots.iter().position(|&p| p == lead) {
                Some(i) => {
                    x ^= self.rows[i].0;
                    z ^= self.rows[i].1;
                }
                None => break,
            }
        }
        (x, z)
    }

    fn insert(&mut self, v: (u64, u64)) {
        let (x, z) = self.reduce(v.0, v.1);
        if let Some(lead) = leading(x, z) {
            self.rows.push((x, z));
            self.pivots.push(lead);
        }
    }

    /// Register width.
    pub fn qubits(&self) -> usize {
        self.n
    }

    /// 𝔽₂ rank — the volume holds `2^rank` strings.
    pub fn rank(&self) -> usize {
        self.rows.len()
    }

    /// Whether the volume is trivial.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The echelon basis, as unsigned strings.
    pub fn basis(&self) -> Vec<PauliString> {
        self.rows
            .iter()
            .map(|&(x, z)| PauliString {
                x,
                z,
                negative: false,
            })
            .collect()
    }

    /// Membership, in `O(rank)` mask algebra.
    pub fn contains(&self, p: PauliString) -> bool {
        self.reduce(p.x, p.z) == (0, 0)
    }

    /// `V + W` — everything the two reach together.
    pub fn join(&self, other: &Volume) -> Result<Volume> {
        if self.n != other.n {
            return Err(Error::InvalidState(format!(
                "volume join: {} qubits against {}",
                self.n, other.n
            )));
        }
        let mut v = self.clone();
        for &r in &other.rows {
            v.insert(r);
        }
        Ok(v)
    }

    /// `V ∩ W` — everything the two share, by elimination on the
    /// stacked system rather than by enumeration.
    pub fn meet(&self, other: &Volume) -> Result<Volume> {
        if self.n != other.n {
            return Err(Error::InvalidState(format!(
                "volume meet: {} qubits against {}",
                self.n, other.n
            )));
        }
        // Row-reduce `self`'s rows while tracking which combination
        // produced each; a combination that also lies in `other` after
        // reduction against `other` is in the intersection.
        let mut out = Volume::empty(self.n)?;
        // Basis of self ∩ other: reduce each element of a basis of self
        // against other's complement is not linear, so do it properly —
        // solve for combinations of self's rows lying in other.
        let rows: Vec<((u64, u64), u128)> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, &r)| (other.reduce(r.0, r.1), 1u128 << i))
            .collect();
        let mut pivots: Vec<usize> = Vec::new();
        let mut basis: Vec<((u64, u64), u128)> = Vec::new();
        for &(row_val, row_track) in &rows {
            let (mut val, mut track) = (row_val, row_track);
            for (i, &p) in pivots.iter().enumerate() {
                if leading_bit(val, p) {
                    val.0 ^= basis[i].0 .0;
                    val.1 ^= basis[i].0 .1;
                    track ^= basis[i].1;
                }
            }
            match leading(val.0, val.1) {
                None => {
                    // This combination of self's rows reduced to zero
                    // against `other`, so it lies in `other` too.
                    let mut acc = (0u64, 0u64);
                    let mut bits = track;
                    while bits != 0 {
                        let i = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        acc.0 ^= self.rows[i].0;
                        acc.1 ^= self.rows[i].1;
                    }
                    out.insert(acc);
                }
                Some(lead) => {
                    basis.push((val, track));
                    pivots.push(lead);
                }
            }
        }
        Ok(out)
    }

    /// Everything that commutes with every element — the **normalizer**
    /// of this subgroup, as an 𝔽₂ kernel. `O(rank · n)` elimination,
    /// no enumeration.
    pub fn centraliser(&self) -> Result<Volume> {
        // A string (x, z) commutes with row (a, b) iff
        // |b & x| + |a & z| ≡ 0, a linear condition on the 2n unknowns.
        // Build the constraint rows in the swapped coordinates and take
        // the kernel.
        let n = self.n;
        let mut constraints: Vec<u128> = self
            .rows
            .iter()
            .map(|&(a, b)| {
                let mut c = 0u128;
                for q in 0..n {
                    if b >> q & 1 == 1 {
                        c |= 1u128 << q; // pairs with x_q
                    }
                    if a >> q & 1 == 1 {
                        c |= 1u128 << (n + q); // pairs with z_q
                    }
                }
                c
            })
            .collect();
        // Gaussian elimination over the 2n unknowns; free columns give
        // the kernel basis.
        let mut pivot_of: Vec<Option<usize>> = vec![None; 2 * n];
        let mut rank = 0usize;
        for (col, slot) in pivot_of.iter_mut().enumerate() {
            if let Some(r) = (rank..constraints.len()).find(|&r| constraints[r] >> col & 1 == 1) {
                constraints.swap(rank, r);
                let p = constraints[rank];
                for (i, c) in constraints.iter_mut().enumerate() {
                    if i != rank && *c >> col & 1 == 1 {
                        *c ^= p;
                    }
                }
                *slot = Some(rank);
                rank += 1;
            }
        }
        let mut out = Volume::empty(n)?;
        for free in 0..2 * n {
            if pivot_of[free].is_some() {
                continue;
            }
            let mut v = 1u128 << free;
            for (col, slot) in pivot_of.iter().enumerate() {
                if let Some(r) = *slot {
                    if constraints[r] >> free & 1 == 1 {
                        v |= 1u128 << col;
                    }
                }
            }
            let x = (v & ((1u128 << n) - 1)) as u64;
            let z = ((v >> n) & ((1u128 << n) - 1)) as u64;
            out.insert((x, z));
        }
        Ok(out)
    }

    /// Does every pair of elements commute? An isotropic volume is an
    /// abelian subgroup — a stabilizer.
    pub fn is_isotropic(&self) -> bool {
        self.rows
            .iter()
            .enumerate()
            .all(|(i, &a)| self.rows[i + 1..].iter().all(|&b| symplectic(a, b) == 0))
    }

    /// Isotropic and of rank `n` — a maximal commuting set, whose
    /// simultaneous eigenspace is a **single state**.
    pub fn is_maximal_isotropic(&self) -> bool {
        self.rank() == self.n && self.is_isotropic()
    }

    /// Whether two volumes are the **same subspace**, independent of
    /// how each was spelled.
    ///
    /// `PartialEq` compares the stored echelon rows, which depend on
    /// insertion order; this compares the spaces. Equal rank plus
    /// mutual containment is the whole test.
    pub fn is_same(&self, other: &Volume) -> bool {
        self.n == other.n
            && self.rank() == other.rank()
            && other.basis().iter().all(|p| self.contains(*p))
    }

    /// The dimension of the region this volume carves: `2^{n − rank}`,
    /// which is `1` exactly when it is maximal isotropic.
    pub fn carves(&self) -> u128 {
        1u128 << (self.n - self.rank()).min(127)
    }
}

fn leading_bit(v: (u64, u64), lead: usize) -> bool {
    if lead <= MAX_QUBITS {
        v.0 >> lead & 1 == 1
    } else {
        v.1 >> (lead - MAX_QUBITS - 1) & 1 == 1
    }
}

// ─────────────────────── the symplectic normal form ───────────────────────

/// **The symplectic normal form of a Pauli set**: an isotropic radical
/// plus hyperbolic pairs.
///
/// `radical ⊥ H₁ ⊥ … ⊥ H_h`, with the radical the part that commutes
/// with the whole span — the stabilizer — and each pair
/// `(eᵢ, fᵢ)` anticommuting. `r` and `h` are what the rest of the
/// module works with, and they are the two numbers that decide how a
/// register shards: `r` says how far it carves, `h` says into how many
/// pieces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stitch {
    n: usize,
    radical: Vec<(u64, u64)>,
    pairs: Vec<((u64, u64), (u64, u64))>,
}

/// **Orthogonalise a Pauli set** under the symplectic form: symplectic
/// Gram–Schmidt over 𝔽₂, `O(rank² · n)`, with no `2ⁿ` anywhere.
///
/// The move is the standard one and it is worth stating because it is
/// the only place the form's *alternating*-ness is used: given a
/// hyperbolic pair `(v, w)` with `⟨v,w⟩ = 1`, the map
/// `u ↦ u ⊕ ⟨w,u⟩v ⊕ ⟨v,u⟩w` sends everything else into the pair's
/// orthogonal complement, because `⟨v,v⟩ = ⟨w,w⟩ = 0`. Anything left
/// with nothing to pair against is in the radical.
pub fn orthogonalize(volume: &Volume) -> Stitch {
    let n = volume.qubits();
    let mut rest: Vec<(u64, u64)> = volume.rows.clone();
    let mut radical: Vec<(u64, u64)> = Vec::new();
    let mut pairs: Vec<((u64, u64), (u64, u64))> = Vec::new();

    while let Some(v) = rest.pop() {
        match rest.iter().position(|&w| symplectic(v, w) == 1) {
            // Nothing pairs with it: it is in the radical of the
            // restricted form.
            None => radical.push(v),
            Some(i) => {
                let w = rest.swap_remove(i);
                for u in rest.iter_mut() {
                    let a = symplectic(v, *u);
                    let b = symplectic(w, *u);
                    if b == 1 {
                        u.0 ^= v.0;
                        u.1 ^= v.1;
                    }
                    if a == 1 {
                        u.0 ^= w.0;
                        u.1 ^= w.1;
                    }
                }
                pairs.push((v, w));
            }
        }
    }
    Stitch { n, radical, pairs }
}

impl Stitch {
    /// A stitch of a given **shape** — `r` radical directions and `h`
    /// hyperbolic pairs on `n` qubits — built directly rather than
    /// decomposed out of a data set.
    ///
    /// The point is [`ShardPlan`]: the accounting questions are about
    /// the shape, and a shape with `h = 20` is a perfectly reasonable
    /// thing to price and an unreasonable thing to enumerate. The
    /// construction is the obvious one (`Z` on the first `r` qubits,
    /// then an `(X, Z)` pair on each of the next `h`), so the returned
    /// stitch is a real one and every invariant holds on it.
    pub fn from_parts(n: usize, r: usize, h: usize) -> Result<Self> {
        if n == 0 || n > MAX_QUBITS {
            return Err(Error::InvalidState(format!(
                "stitch: {n} qubits outside 1..={MAX_QUBITS}"
            )));
        }
        if r + h > n {
            return Err(Error::InvalidState(format!(
                "stitch: r + h = {} exceeds {n} qubits — an isotropic extension cannot \
                 outrank the register",
                r + h
            )));
        }
        Ok(Stitch {
            n,
            radical: (0..r).map(|q| (0u64, 1u64 << q)).collect(),
            pairs: (0..h)
                .map(|i| {
                    let q = r + i;
                    ((1u64 << q, 0u64), (0u64, 1u64 << q))
                })
                .collect(),
        })
    }

    /// Register width.
    pub fn qubits(&self) -> usize {
        self.n
    }

    /// **The isotropic radical** — the part that carves. `r`
    /// commuting directions cut the module to `2^{n−r}`; in stabilizer
    /// language these are the code's generators.
    pub fn radical(&self) -> Vec<PauliString> {
        self.radical
            .iter()
            .map(|&(x, z)| PauliString {
                x,
                z,
                negative: false,
            })
            .collect()
    }

    /// **The hyperbolic pairs** — the part that cannot carve, and
    /// therefore the part that partitions. In stabilizer language
    /// these are the logical `(X̄ᵢ, Z̄ᵢ)` conjugate pairs.
    pub fn pairs(&self) -> Vec<(PauliString, PauliString)> {
        self.pairs
            .iter()
            .map(|&(e, f)| {
                (
                    PauliString {
                        x: e.0,
                        z: e.1,
                        negative: false,
                    },
                    PauliString {
                        x: f.0,
                        z: f.1,
                        negative: false,
                    },
                )
            })
            .collect()
    }

    /// `r`, the radical's rank.
    pub fn radical_rank(&self) -> usize {
        self.radical.len()
    }

    /// `h`, the Witt index of the form restricted to the data — the
    /// number of lateral axes, and the number of logical qubits.
    pub fn witt(&self) -> usize {
        self.pairs.len()
    }

    /// The data's rank, `r + 2h`. Orthogonalising preserves it.
    pub fn rank(&self) -> usize {
        self.radical.len() + 2 * self.pairs.len()
    }

    /// `h` — the bits of structure a stitched region carries **that its
    /// dimension does not show**. The region is `2^{n−r}` whatever `h`
    /// is, so measuring the object's size says nothing about how many
    /// slices it is in.
    pub fn hidden_bits(&self) -> usize {
        self.witt()
    }

    /// One of the `2^h` maximal isotropic extensions: bit `i` of `mask`
    /// picks `fᵢ` over `eᵢ`. Rank `r + h`, isotropic for every mask.
    pub fn selection(&self, mask: u64) -> Result<Volume> {
        let mut v = Volume::empty(self.n)?;
        for &r in &self.radical {
            v.insert(r);
        }
        for (i, &(e, f)) in self.pairs.iter().enumerate() {
            v.insert(if (mask >> i) & 1 == 1 { f } else { e });
        }
        Ok(v)
    }

    /// **All `2^h` maximal isotropic extensions** — one per node.
    /// Refused above [`MAX_ENUMERATED_PAIRS`], because `h` is an
    /// exponent and enumerating it is the caller's decision to make.
    pub fn selections(&self) -> Result<Vec<Volume>> {
        if self.witt() > MAX_ENUMERATED_PAIRS {
            return Err(Error::InvalidState(format!(
                "stitch: {} hyperbolic pairs is {} slices — past the {} this will \
                 enumerate; `h` is an exponent and expanding it is a decision, not a \
                 default",
                self.witt(),
                self.slices(),
                MAX_ENUMERATED_PAIRS
            )));
        }
        (0..(1u64 << self.witt()))
            .map(|m| self.selection(m))
            .collect()
    }

    /// How many slices the stitch cuts: `2^h`, and the node count.
    pub fn slices(&self) -> u128 {
        1u128 << self.witt().min(127)
    }

    /// The reassembled region's dimension, `2^{n−r}` — the code space.
    pub fn region_dimension(&self) -> u128 {
        1u128 << (self.n - self.radical_rank()).min(127)
    }

    /// One slice's dimension, `2^{n−r−h}` — `1` when the extension is
    /// maximal, which is the case that makes a slice a single state.
    pub fn slice_dimension(&self) -> u128 {
        1u128
            << self
                .n
                .saturating_sub(self.radical_rank() + self.witt())
                .min(127)
    }

    /// **Whether the stitch closes**: `2^h · 2^{n−r−h} = 2^{n−r}`,
    /// exact. The arithmetic is trivial; what it asserts is not — the
    /// `2^h` slices are pairwise disjoint (two selections contain both
    /// `eᵢ` and `fᵢ` for some `i`, and nothing is fixed by both), so
    /// their sum is direct and the impossible part costs nothing in the
    /// end. It *partitions*.
    pub fn closes(&self) -> bool {
        self.radical_rank() + self.witt() <= self.n
            && self.slices().saturating_mul(self.slice_dimension()) == self.region_dimension()
    }
}

// ────────────────────────── the shard accounting ──────────────────────────

/// What sharding a register by its stitch actually costs, per node and
/// in total, against holding it monolithically.
///
/// The numbers are the point of the module and they are not flattering
/// in every column: the node count *is* the exponential. What moves is
/// where it lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardPlan {
    qubits: usize,
    radical_rank: usize,
    witt: usize,
}

impl ShardPlan {
    /// Read a stitch as a deployment.
    pub fn new(stitch: &Stitch) -> Self {
        ShardPlan {
            qubits: stitch.qubits(),
            radical_rank: stitch.radical_rank(),
            witt: stitch.witt(),
        }
    }

    /// `2^h` — one per maximal isotropic extension.
    pub fn nodes(&self) -> u128 {
        1u128 << self.witt.min(127)
    }

    /// Bytes one node holds: a rank-`n` tableau over `2n` bits a row,
    /// plus one 16-byte coefficient. **Polynomial in `n`, and it does
    /// not mention the node count.**
    pub fn per_node_bytes(&self) -> u128 {
        let rows = (self.radical_rank + self.witt) as u128;
        rows * (2 * self.qubits as u128).div_ceil(8) + 16
    }

    /// Bytes a dense amplitude vector of the whole register would take.
    /// Saturates; at these widths the number is the point.
    pub fn monolithic_bytes(&self) -> u128 {
        1u128
            .checked_shl(self.qubits as u32)
            .map_or(u128::MAX, |s| s.saturating_mul(16))
    }

    /// Bytes the whole network holds — `nodes × per_node`. This is
    /// **larger** than the code space's own `2^k` coefficients would be,
    /// and saying so is the honest half of the trade: sharding buys a
    /// polynomial per-node object, not a smaller total.
    pub fn network_bytes(&self) -> u128 {
        self.nodes().saturating_mul(self.per_node_bytes())
    }

    /// How much smaller one node's share is than the monolithic
    /// register. `f64` because the ratio is the readable quantity.
    pub fn per_node_ratio(&self) -> f64 {
        self.monolithic_bytes() as f64 / self.per_node_bytes() as f64
    }
}
