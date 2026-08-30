//! Geometric qudits: a level count that is the geometry's, a position
//! that can move, and an interior no face can see.
//!
//! A qudit is normally a declaration — "this register has `d` levels" —
//! and the levels are a label space `ℂ^d` with no geometry inside it.
//! This module takes the other route. Everything here is derived from
//! **one ambient object**: [`stitch::Volume`](crate::stitch::Volume),
//! the phase-free Pauli group on `n` qubits as an 𝔽₂ space under its
//! symplectic form, whose meet, join and centraliser cost `O(rank · n)`
//! and never mention `2ⁿ`.
//!
//! The ambient has a name, and [`polar`](crate::polar) carries it:
//! `𝔽₂^{2n}` minus the origin is the point set of the projective space
//! `PG(2n−1, 2)`, and the commutation form makes it the polar space
//! `W(2n−1, 2)`. A frame is a flat of that polar space. At two qubits
//! it is the doily `GQ(2,2)` — 15 points, 15 lines, no triangles — and
//! a wider register is a combinatorial space of doilies rather than a
//! new kind of object.
//!
//! ## The qudit is the obstruction
//!
//! Put an isotropic flat `V` into that space — an abelian subgroup,
//! which is to say a stabilizer, which is to say a **constraint**. Three
//! things follow immediately, and together they are the whole
//! construction:
//!
//! * `V` is the qudit's **position**: a flat in the ambient, `O(r · n)`
//!   bits, movable.
//! * `V^⊥ = ` [`centraliser`](crate::stitch::Volume::centraliser) is the
//!   **bounded boundary** it induces, of rank `2n − r`, with `V ⊆ V^⊥`.
//! * `V^⊥/V` carries a **non-degenerate** symplectic form of rank `2h`
//!   with `h = n − r`, so the qudit has exactly
//!   ```text
//!   levels = 2^h
//!   ```
//!   — read off the geometry, never declared.
//!
//! That is the sheet-and-obstruction picture made exact. The
//! constraint is the obstruction; the boundary it bounds is `V^⊥`; and
//! the knot-register inside it is `V^⊥/V`, whose conjugate pairs
//! [`orthogonalize`] hands over directly —
//! the radical comes back as `V` itself and the `h` hyperbolic pairs
//! *are* the qudit's logical conjugate pairs. Which member of a pair
//! is called `X̄` and which `Z̄` is a labelling convention and not a
//! fact about the geometry — the pair is what the form determines, and
//! the module reports it as a pair.
//!
//! ## The operative algebra is small on purpose
//!
//! `A_V ⊊ End(V)` is the theory's central restriction, and here it is
//! forced rather than chosen: the admissible operations are the
//! **logical Pauli group**, `2h` bits of 𝔽₂ data, against `4^h`
//! complex parameters for the full endomorphism algebra of the level
//! space. [`OperativeCost`] reports both. A large relational state
//! volume with a small symmetry-admissible operator volume is the shape
//! the theory predicts, and the numbers are measured off the code.
//!
//! ## Frame-in-frame, and motion
//!
//! A Clifford is a symplectic map, so it carries `V ↦ gV`:
//! [`transport`](VolQudit::transport) **moves** the qudit. Its
//! signature — levels, logical rank, ambient — is invariant under
//! every transport, which is the covariance condition frame geometry
//! asks for, and `tests/volqudit.rs` asserts it rather than assuming
//! it.
//!
//! A transport that returns the frame to itself is a **loop**, and a
//! loop need not act trivially on what the frame bounds.
//! [`Holonomy`] records the induced action on `V^⊥/V`; the loops with
//! `H(γ) = I` are flat and the rest carry
//! [`curvature`](Holonomy::is_flat) — a logical operation obtained by
//! *moving the qudit around and bringing it back*. That is the
//! knot-register, and [`VolQudit::holonomy_search`] finds them by
//! enumeration rather than construction, so the ones that exist are
//! measured and the ones that do not are absent by report.
//!
//! And the ambient of one qudit can be the level space of another:
//! [`nest`](VolQudit::nest) is the frame-in-frame move, with the level
//! counts multiplying exactly.
//!
//! ## The interior: what no face can see
//!
//! Split the ambient into `k` blocks and the qudit becomes a
//! **compound** ([`CompoundQudit`]). The face map `d_i` forgets slot
//! `i`, the **skeleton** is everything visible on some proper face, and
//! the vol is the quotient
//!
//! ```text
//! interior = (V^⊥/V) / skeleton
//! ```
//!
//! — the logical content that genuinely requires *every* block at once.
//! Independent qudits side by side have interior rank 0, because every
//! logical operator factors. A frame that couples them can have
//! interior rank above zero, and
//! [`interior_rank`](CompoundQudit::interior_rank) is the measurement.
//! A nonzero interior with every face trivial is the **Brunnian**
//! case, and [`is_brunnian`](CompoundQudit::is_brunnian) tests exactly
//! that.
//!
//! ## Promotion
//!
//! A closed qudit promotes to an [`Atom`] carrying only its
//! [`Signature`]. The theory's *interface sufficiency* is a
//! conjecture; here it is at least a **test**: two structurally
//! different frames with equal signatures are checked to behave
//! identically under a common outer context. That does not prove the
//! conjecture — it makes it falsifiable at this scale, and the module
//! says which.
//!
//! ## Honest scope
//!
//! Everything here is 𝔽₂ and Clifford. The level space is a *Pauli*
//! geometry, so the admissible algebra is the logical Pauli group and
//! not the full logical unitary group; magic lives outside it, exactly
//! as it does in [`retro`](crate::retro) and
//! [`logical`](crate::logical). What this buys is that every quantity
//! above — levels, boundary rank, operative dimension, interior rank,
//! holonomy class — is computed in `O(poly(n))` mask algebra with no
//! amplitude anywhere.

use crate::backend::{conjugate_by_step, CliffordStep, PauliString};
use crate::error::{Error, Result};
use crate::stitch::{orthogonalize, symplectic, Volume};

/// A qudit's externally visible interface — the whole of what an outer
/// context sees after [promotion](VolQudit::promote).
///
/// This is the theory's `σ(𝕧)`: the closed vol's interface signature,
/// deliberately *smaller* than its interior. Two qudits with equal
/// signatures have equal level counts and equal admissible algebra
/// dimensions, whatever their frames look like.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Signature {
    /// Ambient qubits the frame lives in.
    pub ambient: usize,
    /// Rank of the frame itself — the constraint's size.
    pub frame_rank: usize,
    /// `h = n − r`, the number of conjugate pairs the boundary carries.
    pub logical_rank: usize,
}

impl Signature {
    /// `2^h` — the qudit's level count.
    pub fn levels(&self) -> u128 {
        1u128 << self.logical_rank.min(127)
    }
}

/// What it costs to *name* an admissible operation, against what it
/// costs to name a general one on the same level space.
///
/// The theory's restriction `A_V ⊊ End(V)` is not a modelling choice
/// here — the geometry admits the logical Pauli group and nothing
/// else — and the gap is in the units, which is why both are reported.
/// An admissible operation is `2h` **bits**: one per logical `X̄`/`Z̄`.
/// A general element of `End` on the same space is `4^h` **complex
/// numbers**. At `h = 8` that is 16 bits against 65,536 amplitudes,
/// for the same `256`-level qudit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperativeCost {
    /// `2^h`, the level count.
    pub levels: u128,
    /// **Bits** to name one admissible operation: `2h`.
    pub admissible_bits: usize,
    /// How many admissible operations there are: `4^h` Pauli classes.
    pub admissible_order: u128,
    /// **Complex parameters** for a general element of `End` on the
    /// level space: `4^h`.
    pub full_end_params: u128,
    /// Bits to store the frame itself: `rank × 2n`.
    pub frame_bits: usize,
}

/// A geometric qudit: an isotropic flat in an ambient Clifford space,
/// together with everything the flat determines.
#[derive(Clone, Debug)]
pub struct VolQudit {
    ambient: usize,
    frame: Volume,
    boundary: Volume,
    pairs: Vec<(PauliString, PauliString)>,
}

impl VolQudit {
    /// A qudit from a frame, which must be **isotropic**.
    ///
    /// A frame that is not isotropic is not a position: its own
    /// elements do not commute, so it carves no region and bounds no
    /// boundary. That is refused by name rather than projected onto
    /// the nearest legal thing.
    pub fn new(ambient: usize, frame: Volume) -> Result<Self> {
        if frame.qubits() != ambient {
            return Err(Error::InvalidState(format!(
                "vol qudit: a frame on {} qubits cannot sit in a {ambient}-qubit ambient",
                frame.qubits()
            )));
        }
        if !frame.is_isotropic() {
            return Err(Error::InvalidState(
                "vol qudit: the frame is not isotropic — two of its elements anticommute, \
                 so it is not an abelian subgroup, carves no region and bounds no \
                 boundary; refused rather than projected onto the nearest flat"
                    .into(),
            ));
        }
        let boundary = frame.centraliser()?;
        // The boundary's symplectic normal form returns the frame as
        // its radical and the qudit's conjugate pairs as its hyperbolic
        // part — the logical X̄/Z̄, found rather than declared.
        let st = orthogonalize(&boundary);
        let pairs = st.pairs();
        Ok(VolQudit {
            ambient,
            frame,
            boundary,
            pairs,
        })
    }

    /// The **free** qudit on `n` qubits: no constraint at all, so
    /// `h = n` and the levels are `2ⁿ`. A plain register is the
    /// degenerate case of this construction, which is worth being able
    /// to say.
    pub fn free(ambient: usize) -> Result<Self> {
        Self::new(ambient, Volume::empty(ambient)?)
    }

    /// Ambient qubit count.
    pub fn ambient(&self) -> usize {
        self.ambient
    }

    /// The frame — the qudit's position in the ambient.
    pub fn frame(&self) -> &Volume {
        &self.frame
    }

    /// `V^⊥` — the bounded boundary the obstruction induces. Rank
    /// `2n − r`, and it always contains the frame.
    pub fn boundary(&self) -> &Volume {
        &self.boundary
    }

    /// `h`, the number of conjugate pairs in `V^⊥/V`.
    pub fn logical_rank(&self) -> usize {
        self.pairs.len()
    }

    /// `2^h`, the qudit's level count — derived, not declared.
    pub fn levels(&self) -> u128 {
        1u128 << self.logical_rank().min(127)
    }

    /// The `h` conjugate pairs spanning `V^⊥/V`: each anticommutes
    /// with its partner and commutes with the whole frame. Which
    /// member is `X̄` and which is `Z̄` is a naming convention, so the
    /// pair is returned rather than a labelled pair.
    pub fn logical_pairs(&self) -> &[(PauliString, PauliString)] {
        &self.pairs
    }

    /// The interface an outer context sees.
    pub fn signature(&self) -> Signature {
        Signature {
            ambient: self.ambient,
            frame_rank: self.frame.rank(),
            logical_rank: self.logical_rank(),
        }
    }

    /// The admissible algebra against the full one.
    pub fn operative_cost(&self) -> OperativeCost {
        let h = self.logical_rank();
        OperativeCost {
            levels: self.levels(),
            admissible_bits: 2 * h,
            admissible_order: 1u128 << (2 * h).min(127),
            full_end_params: 1u128 << (2 * h).min(127),
            frame_bits: self.frame.rank() * 2 * self.ambient,
        }
    }

    /// **Move the qudit**: conjugate its frame by a Clifford, which is
    /// a symplectic map and therefore carries flats to flats.
    ///
    /// The signature is invariant under this — a qudit that moves is
    /// still the same qudit — which is the frame/object covariance
    /// condition, asserted in the tests rather than assumed here.
    pub fn transport(&self, steps: &[CliffordStep]) -> Result<Self> {
        let moved: Vec<PauliString> = self
            .frame
            .basis()
            .into_iter()
            .map(|p| steps.iter().fold(p, |acc, &s| conjugate_by_step(acc, s)))
            .collect();
        VolQudit::new(self.ambient, Volume::span(self.ambient, &moved)?)
    }

    /// Whether a transport is a **loop**: does it return the frame to
    /// itself, as a subspace?
    pub fn closes(&self, steps: &[CliffordStep]) -> Result<bool> {
        Ok(self.transport(steps)?.frame.is_same(&self.frame))
    }

    /// The action a **closed** transport induces on `V^⊥/V`.
    ///
    /// The frame comes back; what it bounds need not. A loop with
    /// `H(γ) ≠ I` has moved the qudit around the ambient and returned
    /// it rotated — a logical operation obtained from motion, which is
    /// the whole content of a knot-register.
    ///
    /// Refuses a transport that is not a loop, because holonomy of an
    /// open path is not defined.
    pub fn holonomy(&self, steps: &[CliffordStep]) -> Result<Holonomy> {
        if !self.closes(steps)? {
            return Err(Error::InvalidState(
                "holonomy: this transport does not return the frame to itself, so it is \
                 an open path and has no holonomy; use `transport` and compare \
                 signatures instead"
                    .into(),
            ));
        }
        let push = |p: PauliString| steps.iter().fold(p, |acc, &s| conjugate_by_step(acc, s));
        // Each logical's image, read in the (X̄, Z̄) coordinates of
        // V^⊥/V by its commutation pattern — the only basis-free
        // question available, and exactly the symplectic form.
        let h = self.pairs.len();
        let mut matrix = vec![vec![false; 2 * h]; 2 * h];
        for (col, src) in self.pairs.iter().flat_map(|&(x, z)| [x, z]).enumerate() {
            let img = push(src);
            if !self.boundary.contains(img) {
                return Err(Error::InvalidState(
                    "holonomy: a logical left the boundary under a transport that fixes \
                     the frame — impossible for a symplectic map, so this is a bug and \
                     not a measurement"
                        .into(),
                ));
            }
            for (row, &(x, z)) in self.pairs.iter().enumerate() {
                // ω(img, Z̄ᵢ) reads the X̄ᵢ coefficient; ω(img, X̄ᵢ) the Z̄ᵢ.
                matrix[2 * row][col] = symplectic((img.x, img.z), (z.x, z.z)) == 1;
                matrix[2 * row + 1][col] = symplectic((img.x, img.z), (x.x, x.z)) == 1;
            }
        }
        Ok(Holonomy {
            logical_rank: h,
            matrix,
        })
    }

    /// Search short Clifford words for loops, reporting the distinct
    /// non-flat holonomies found.
    ///
    /// Constructing a nontrivial loop by hand needs a symmetry one may
    /// not have; enumerating short words needs only a budget. What
    /// comes back is a measurement — a frame whose loops are all flat
    /// says so by returning an empty list, and that is data.
    pub fn holonomy_search(&self, generators: &[CliffordStep], depth: usize) -> Vec<Holonomy> {
        let mut found: Vec<Holonomy> = Vec::new();
        let mut word: Vec<CliffordStep> = Vec::new();
        self.walk(generators, depth, &mut word, &mut found);
        found
    }

    fn walk(
        &self,
        gens: &[CliffordStep],
        depth: usize,
        word: &mut Vec<CliffordStep>,
        found: &mut Vec<Holonomy>,
    ) {
        if !word.is_empty() {
            if let Ok(true) = self.closes(word) {
                if let Ok(h) = self.holonomy(word) {
                    if !h.is_flat() && !found.contains(&h) {
                        found.push(h);
                    }
                }
            }
        }
        if depth == 0 {
            return;
        }
        for &g in gens {
            word.push(g);
            self.walk(gens, depth - 1, word, found);
            word.pop();
        }
    }

    /// **Frame in frame**: a qudit whose ambient is this one's level
    /// space.
    ///
    /// The inner qudit's own frame is expressed in the outer's logical
    /// coordinates, so the two nest exactly and the level counts
    /// compose: an inner `h'` inside an outer `h` leaves `2^{h'}`
    /// levels, and the inner frame costs `O(h)` rather than `O(n)` —
    /// the outer geometry has already paid for the reduction.
    pub fn nest(&self, inner_frame: Volume) -> Result<VolQudit> {
        if inner_frame.qubits() != self.logical_rank() {
            return Err(Error::InvalidState(format!(
                "nest: the inner frame lives on {} qubits but this qudit's level space \
                 has logical rank {}",
                inner_frame.qubits(),
                self.logical_rank()
            )));
        }
        VolQudit::new(self.logical_rank(), inner_frame)
    }

    /// Promote a closed qudit to an atom carrying only its interface.
    pub fn promote(&self) -> Atom {
        Atom {
            signature: self.signature(),
        }
    }
}

/// The induced action of a closed transport on `V^⊥/V`, as a `2h × 2h`
/// 𝔽₂ matrix in the `(X̄₁, Z̄₁, …, X̄_h, Z̄_h)` basis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Holonomy {
    logical_rank: usize,
    matrix: Vec<Vec<bool>>,
}

impl Holonomy {
    /// `h`.
    pub fn logical_rank(&self) -> usize {
        self.logical_rank
    }

    /// The matrix, row-major.
    pub fn matrix(&self) -> &[Vec<bool>] {
        &self.matrix
    }

    /// Is the loop flat — `H(γ) = I`?
    pub fn is_flat(&self) -> bool {
        self.matrix
            .iter()
            .enumerate()
            .all(|(i, row)| row.iter().enumerate().all(|(j, &b)| b == (i == j)))
    }

    /// `K(γ) = H(γ) − I`, as the count of entries where the loop
    /// differs from the identity. Zero exactly when the loop is flat.
    pub fn curvature(&self) -> usize {
        self.matrix
            .iter()
            .enumerate()
            .map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .filter(|&(j, &b)| b != (i == j))
                    .count()
            })
            .sum()
    }

    /// The loop's order: how many times it must be traversed before the
    /// induced action is the identity. `None` past `max`.
    pub fn order(&self, max: usize) -> Option<usize> {
        let n = 2 * self.logical_rank;
        let mut acc: Vec<Vec<bool>> = (0..n).map(|i| (0..n).map(|j| i == j).collect()).collect();
        for k in 1..=max {
            let mut next = vec![vec![false; n]; n];
            for (i, out) in next.iter_mut().enumerate() {
                for (j, cell) in out.iter_mut().enumerate() {
                    let mut b = false;
                    for (l, row) in acc.iter().enumerate() {
                        b ^= self.matrix[i][l] && row[j];
                    }
                    *cell = b;
                }
            }
            acc = next;
            if acc
                .iter()
                .enumerate()
                .all(|(i, row)| row.iter().enumerate().all(|(j, &b)| b == (i == j)))
            {
                return Some(k);
            }
        }
        None
    }
}

/// A promoted qudit: the interface, and nothing else.
///
/// Promotion is the theory's memoization step — a closed vol becomes
/// one new primitive, and the outer level addresses it without
/// reopening its interior. What survives is exactly [`Signature`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Atom {
    /// The interface the outer context sees.
    pub signature: Signature,
}

impl Atom {
    /// The level count the outer context computes with.
    pub fn levels(&self) -> u128 {
        self.signature.levels()
    }
}

/// A qudit whose ambient is partitioned into slots, so the vol
/// machinery — faces, skeleton, interior — applies.
#[derive(Clone, Debug)]
pub struct CompoundQudit {
    qudit: VolQudit,
    slots: Vec<u64>,
}

/// A slot's mask over the ambient register.
fn block_mask(offset: usize, width: usize) -> u64 {
    if width == 0 || offset >= 64 {
        return 0;
    }
    let w = width.min(64 - offset);
    let m = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
    m << offset
}

impl CompoundQudit {
    /// A compound over `slots`, each `(offset, width)`, which must be
    /// disjoint and cover the ambient.
    pub fn new(qudit: VolQudit, slots: &[(usize, usize)]) -> Result<Self> {
        if slots.len() < 2 {
            return Err(Error::InvalidState(
                "compound qudit: a vol needs at least two slots — with one slot every \
                 face is empty and the interior question is not posed"
                    .into(),
            ));
        }
        let mut seen = 0u64;
        let mut masks = Vec::with_capacity(slots.len());
        for &(off, w) in slots {
            let m = block_mask(off, w);
            if m & seen != 0 {
                return Err(Error::InvalidState(format!(
                    "compound qudit: slot at {off} overlaps an earlier one — slots \
                     partition the ambient, they do not share it"
                )));
            }
            seen |= m;
            masks.push(m);
        }
        Ok(CompoundQudit {
            qudit,
            slots: masks,
        })
    }

    /// The underlying qudit.
    pub fn qudit(&self) -> &VolQudit {
        &self.qudit
    }

    /// Slot count.
    pub fn slots(&self) -> usize {
        self.slots.len()
    }

    /// Every Pauli supported entirely **off** slot `i`.
    fn off_slot(&self, i: usize) -> Result<Volume> {
        let mask = *self.slots.get(i).ok_or_else(|| {
            Error::InvalidState(format!(
                "compound qudit: no slot {i} of {}",
                self.slots.len()
            ))
        })?;
        let n = self.qudit.ambient;
        let mut gens = Vec::new();
        for q in 0..n {
            if mask >> q & 1 == 1 {
                continue;
            }
            gens.push(PauliString {
                x: 1 << q,
                z: 0,
                negative: false,
            });
            gens.push(PauliString {
                x: 0,
                z: 1 << q,
                negative: false,
            });
        }
        Volume::span(n, &gens)
    }

    /// The logical classes visible on the face that **forgets slot
    /// `i`** — those with *some* representative supported entirely off
    /// that slot.
    ///
    /// This is `d_i` in the theory's sense, and it is asked of the
    /// coset rather than of a spelling: a logical operator that can be
    /// rewritten off the slot by multiplying in a stabilizer is
    /// visible on that face, and the answer must not depend on which
    /// representative the basis happened to hold. Computed as
    /// `V^⊥ ∩ (off-slot)`, pushed into `V^⊥/V`.
    pub fn face(&self, i: usize) -> Result<Volume> {
        self.qudit.boundary().meet(&self.off_slot(i)?)
    }

    /// The rank of face `i` **as seen in `V^⊥/V`** — the number of
    /// independent logical classes that survive forgetting slot `i`.
    pub fn face_rank(&self, i: usize) -> Result<usize> {
        let with_frame = self.face(i)?.join(self.qudit.frame())?;
        Ok(with_frame.rank() - self.qudit.frame().rank())
    }

    /// The **skeleton**: everything the proper faces can see between
    /// them, as a rank in `V^⊥/V`.
    ///
    /// Canonical, because each face is a meet of subspaces rather than
    /// a filter on a chosen basis: a logical class counts as visible
    /// exactly when *some* representative of it misses the slot.
    pub fn skeleton_rank(&self) -> Result<usize> {
        let mut acc = self.qudit.frame().clone();
        for i in 0..self.slots.len() {
            acc = acc.join(&self.face(i)?)?;
        }
        Ok(acc.rank() - self.qudit.frame().rank())
    }

    /// `dim(V^⊥/V) − dim(skeleton)` — the logical content that
    /// requires **every** slot at once.
    ///
    /// Zero for independent qudits placed side by side, because every
    /// logical class then has a representative on a single slot and is
    /// therefore visible on every other face. Positive exactly when the
    /// frame couples the slots irreducibly, and that is the vol.
    pub fn interior_rank(&self) -> Result<usize> {
        Ok(2 * self.qudit.logical_rank() - self.skeleton_rank()?)
    }

    /// **Brunnian**: a nonzero interior with every proper face empty.
    ///
    /// The content exists, acts at full arity, and no proper subsystem
    /// can detect it.
    pub fn is_brunnian(&self) -> Result<bool> {
        if self.interior_rank()? == 0 {
            return Ok(false);
        }
        for i in 0..self.slots.len() {
            if self.face_rank(i)? != 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
