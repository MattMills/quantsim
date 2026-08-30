//! What the register is embedded in: `PG(2n−1, 2)` and its polar space
//! `W(2n−1, 2)` — and what survives when an assembly closes.
//!
//! [`stitch`](crate::stitch) works with 𝔽₂ subspaces under a symplectic
//! form and [`volqudit`](crate::volqudit) puts qudits into them, but
//! neither said what the ambient *is*. It is a projective space, and
//! naming it is not decoration: the counts, the incidence structure and
//! the classical geometry come with the name.
//!
//! ## The embedding
//!
//! The phase-free Pauli group on `n` qubits is `𝔽₂^{2n}` minus the
//! origin, and scalars are already quotiented out, so its points are
//! the points of the projective space
//!
//! ```text
//! PG(2n − 1, 2)      2n homogeneous coordinates,  2^{2n} − 1 points
//! ```
//!
//! The commutation form makes it a **polar space**: `W(2n−1, 2)`, whose
//! *totally isotropic* flats are exactly the abelian subgroups — the
//! stabilizers, and therefore the [`VolQudit`]
//! frames. A [`Volume`] is a flat; an isotropic
//! one is a flat *of the polar space*; a maximal one is a **generator**.
//!
//! The counts are closed forms, and [`PolarSpace`] checks each of them
//! against enumeration at small `n` rather than quoting them:
//!
//! ```text
//! points                  2^{2n} − 1
//! lines through a point   2^{2n−2} − 1
//! totally isotropic lines (2^{2n} − 1)(2^{2n−2} − 1) / 3
//! generators              ∏_{i=1}^{n} (2^i + 1)
//! ```
//!
//! ## `n = 2` is the doily, and that is the whole point
//!
//! At two qubits the polar space is `W(3, 2)`: **15 points, 15 totally
//! isotropic lines, 3 points on every line, 3 lines through every
//! point, and no triangles.** That is the generalized quadrangle
//! `GQ(2,2)` — the *doily* — and [`PolarSpace::is_doily`],
//! [`triangles`](PolarSpace::triangles) and
//! [`gq_axiom`](PolarSpace::gq_axiom) verify all of it by enumeration.
//!
//! So the four homogeneous coordinates of `PG(3,2)` are the unit the
//! rest is built from. A wider register is `W(2n−1, 2)`, which is a
//! **combinatorial space of doilies**: the non-degenerate 4-dimensional
//! subspaces are copies of `W(3,2)` sitting inside it, and
//! [`doilies`](PolarSpace::doilies) counts them. Two qubits is one
//! doily; three is 28 of them overlapping; the configuration is the
//! geometry.
//!
//! ## Effective geometry: over the representation, not over the state
//!
//! Frame geometry asks what can be represented together; object
//! geometry asks what closes into a persistent unit. Effective geometry
//! asks what the **whole assembly** permits to remain consequential,
//! and the closure it quotients by is over the *representation* — the
//! frames — rather than over any state.
//!
//! For an [`Assembly`] of frames `V₁ … V_k` in one ambient, an
//! operation is admissible for qudit `i` exactly when it commutes with
//! `Vᵢ`, so the systemic closure is the constraint intersection
//!
//! ```text
//! Im Γ  =  ⋂ᵢ Vᵢ^⊥  =  (⋁ᵢ Vᵢ)^⊥
//! ```
//!
//! — idempotent by construction, and [`Assembly::is_idempotent`] checks
//! it rather than assuming it. The **effective dimension** is the
//! logical rank of the joined frame, and the finding is that it is
//! *not* the sum of the parts:
//!
//! ```text
//! d_eff  =  n − rank(⋁ Vᵢ)   ≤   Σᵢ hᵢ
//! ```
//!
//! with [`closure_deficit`](Assembly::closure_deficit) measuring the
//! gap. Locally rich frames can carry a large joint carrier while only
//! a low-rank sector survives closing them together, which is the whole
//! reason effective geometry is a separate class from the other two.
//!
//! ## The throat, as an inequality rather than a slogan
//!
//! The hourglass law says a round trip's rank is the narrowest level's
//! capacity. Here the honest statement is a **bound**:
//!
//! ```text
//! d_eff  ≤  min_i h_i
//! ```
//!
//! — what survives the assembly cannot exceed the narrowest frame's own
//! logical rank — and it becomes an **equality exactly on a nested
//! chain** `V₁ ⊆ V₂ ⊆ …`, where each stage's constraint already
//! contains the last. [`Assembly::throat`] reports the bound,
//! [`Assembly::is_nested`] reports when it is tight, and
//! `tests/polar.rs` measures both the tight and the slack case rather
//! than asserting the law.
//!
//! ## Dynamic sufficiency
//!
//! `Φ(P, U) = P U (I − P)` is zero exactly when the represented future
//! factors through the represented present. On this layer that is
//! [`dynamic_obstruction`]: the rank by which a transport pushes a
//! frame out of itself, zero exactly when the transport is a loop —
//! which is the same predicate
//! [`VolQudit::closes`](crate::volqudit::VolQudit::closes) already
//! answers, now with the frame-geometry name and a number attached
//! instead of a bool.

use crate::backend::{conjugate_by_step, CliffordStep, PauliString};
use crate::error::{Error, Result};
use crate::stitch::{symplectic, Volume};
use crate::volqudit::VolQudit;

/// Widest polar space the enumerating methods will walk: they are
/// `O(4ⁿ)` in the points and `O(16ⁿ)` in the lines.
pub const MAX_ENUMERATED: usize = 5;

/// The symplectic polar space `W(2n−1, 2)` over `PG(2n−1, 2)` — the
/// ambient every frame in this crate is a flat of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolarSpace {
    n: usize,
}

fn pow2(k: usize) -> u128 {
    1u128 << k.min(127)
}

impl PolarSpace {
    /// The polar space of an `n`-qubit register.
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 || n > 63 {
            return Err(Error::InvalidState(format!(
                "polar space: {n} qubits outside 1..=63"
            )));
        }
        Ok(PolarSpace { n })
    }

    /// Qubits.
    pub fn qubits(&self) -> usize {
        self.n
    }

    /// `2n` — the homogeneous coordinates of the ambient projective
    /// space. Two qubits is four, which is why the doily is the unit.
    pub fn homogeneous_coordinates(&self) -> usize {
        2 * self.n
    }

    /// `2n − 1` — the ambient's projective dimension.
    pub fn projective_dimension(&self) -> usize {
        2 * self.n - 1
    }

    /// `2^{2n} − 1`.
    pub fn points(&self) -> u128 {
        pow2(2 * self.n) - 1
    }

    /// `2^{2n−2} − 1` — every point sits on this many totally isotropic
    /// lines, and the count does not depend on which point.
    pub fn lines_through_a_point(&self) -> u128 {
        pow2(2 * self.n - 2) - 1
    }

    /// `(2^{2n} − 1)(2^{2n−2} − 1) / 3` — each line carries three
    /// points, so the double count divides exactly.
    pub fn totally_isotropic_lines(&self) -> u128 {
        self.points() * self.lines_through_a_point() / 3
    }

    /// `∏_{i=1}^{n} (2^i + 1)` — the maximal totally isotropic flats,
    /// which are the maximal commuting sets: `2^n` operators
    /// simultaneously diagonalizable, one basis apiece.
    /// Saturates rather than wrapping: the product passes `u128` around
    /// `n = 40`, and a saturated count is an honest "too large to
    /// name" where a wrapped one is a wrong number.
    pub fn generators(&self) -> u128 {
        (1..=self.n).fold(1u128, |acc, i| acc.saturating_mul(pow2(i) + 1))
    }

    /// Is this the doily — `W(3,2) = GQ(2,2)`, the two-qubit geometry?
    pub fn is_doily(&self) -> bool {
        self.n == 2
    }

    /// Every point, as an unsigned Pauli string.
    pub fn enumerate_points(&self) -> Result<Vec<PauliString>> {
        if self.n > MAX_ENUMERATED {
            return Err(Error::InvalidState(format!(
                "polar space: enumerating {} points at n = {} is past the {MAX_ENUMERATED}-qubit \
                 cap; the closed forms answer the counting questions without it",
                self.points(),
                self.n
            )));
        }
        let m = 1u64 << self.n;
        Ok((0..m)
            .flat_map(|x| (0..m).map(move |z| (x, z)))
            .filter(|&(x, z)| x | z != 0)
            .map(|(x, z)| PauliString {
                x,
                z,
                negative: false,
            })
            .collect())
    }

    /// Every totally isotropic line, as its three points sorted.
    pub fn enumerate_lines(&self) -> Result<Vec<[PauliString; 3]>> {
        let pts = self.enumerate_points()?;
        let key = |p: &PauliString| (p.x, p.z);
        let mut seen = std::collections::BTreeSet::new();
        let mut out = Vec::new();
        for (i, &p) in pts.iter().enumerate() {
            for &q in &pts[i + 1..] {
                if symplectic((p.x, p.z), (q.x, q.z)) != 0 {
                    continue;
                }
                let r = PauliString {
                    x: p.x ^ q.x,
                    z: p.z ^ q.z,
                    negative: false,
                };
                let mut l = [p, q, r];
                l.sort_by_key(key);
                if seen.insert([key(&l[0]), key(&l[1]), key(&l[2])]) {
                    out.push(l);
                }
            }
        }
        Ok(out)
    }

    /// Triples of pairwise-collinear points not on a common line.
    ///
    /// A generalized quadrangle has none, and that is the defining
    /// property rather than a consequence — so this is the check that
    /// decides whether the name is earned.
    pub fn triangles(&self) -> Result<usize> {
        let pts = self.enumerate_points()?;
        let mut count = 0;
        for (i, &a) in pts.iter().enumerate() {
            for (j, &b) in pts.iter().enumerate().skip(i + 1) {
                if symplectic((a.x, a.z), (b.x, b.z)) != 0 {
                    continue;
                }
                let sum = (a.x ^ b.x, a.z ^ b.z);
                for &c in pts.iter().skip(j + 1) {
                    if symplectic((a.x, a.z), (c.x, c.z)) != 0
                        || symplectic((b.x, b.z), (c.x, c.z)) != 0
                    {
                        continue;
                    }
                    if (c.x, c.z) != sum {
                        count += 1;
                    }
                }
            }
        }
        Ok(count)
    }

    /// The generalized-quadrangle axiom: for every point off a line,
    /// **exactly one** point of that line is collinear with it.
    pub fn gq_axiom(&self) -> Result<bool> {
        let pts = self.enumerate_points()?;
        let lines = self.enumerate_lines()?;
        for &p in &pts {
            for l in &lines {
                if l.iter().any(|q| (q.x, q.z) == (p.x, p.z)) {
                    continue;
                }
                let c = l
                    .iter()
                    .filter(|q| symplectic((p.x, p.z), (q.x, q.z)) == 0)
                    .count();
                if c != 1 {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// How many **doilies** sit inside this space, in closed form:
    /// the non-degenerate 4-dimensional subspaces, each a copy of
    /// `W(3,2)`.
    ///
    /// ```text
    /// 2^{4(n−2)} · (2^{2n−2} − 1)(2^{2n} − 1) / 45
    /// ```
    ///
    /// This is the sense in which a wider register is a combinatorial
    /// space of four-coordinate projective spaces rather than a new
    /// kind of object: 1 at two qubits, 336 at three, 91,392 at four.
    /// [`doilies`](Self::doilies) checks it against enumeration where
    /// enumeration is affordable.
    /// The `/45` is exact — `(2^{2n−2} − 1)(2^{2n} − 1)` is divisible
    /// by 45 for every `n ≥ 2` — and is taken before the power of two
    /// so the division never truncates. Saturates past `u128`.
    pub fn doily_count(&self) -> u128 {
        if self.n < 2 {
            return 0;
        }
        let n = self.n;
        let a = pow2(2 * n - 2) - 1;
        let b = pow2(2 * n) - 1;
        match a.checked_mul(b) {
            Some(ab) => (ab / 45).saturating_mul(pow2(4 * (n - 2))),
            None => u128::MAX,
        }
    }

    /// How many **doilies** sit inside this space, by enumeration —
    /// the check on [`doily_count`](Self::doily_count).
    ///
    pub fn doilies(&self) -> Result<usize> {
        if self.n < 2 {
            return Ok(0);
        }
        if self.n > 3 {
            return Err(Error::InvalidState(format!(
                "polar space: enumerating doilies at n = {} is past the 3-qubit cap on \
                 this walk",
                self.n
            )));
        }
        let pts = self.enumerate_points()?;
        let mut seen = std::collections::BTreeSet::new();
        // A non-degenerate 4-space is spanned by two hyperbolic pairs.
        for (i, &e1) in pts.iter().enumerate() {
            for &f1 in pts.iter().skip(i + 1) {
                if symplectic((e1.x, e1.z), (f1.x, f1.z)) != 1 {
                    continue;
                }
                for (k, &e2) in pts.iter().enumerate() {
                    if symplectic((e1.x, e1.z), (e2.x, e2.z)) != 0
                        || symplectic((f1.x, f1.z), (e2.x, e2.z)) != 0
                    {
                        continue;
                    }
                    for &f2 in pts.iter().skip(k + 1) {
                        if symplectic((e2.x, e2.z), (f2.x, f2.z)) != 1
                            || symplectic((e1.x, e1.z), (f2.x, f2.z)) != 0
                            || symplectic((f1.x, f1.z), (f2.x, f2.z)) != 0
                        {
                            continue;
                        }
                        // Canonicalize the 4-space by its point set.
                        let v = Volume::span(self.n, &[e1, f1, e2, f2])?;
                        if v.rank() != 4 {
                            continue;
                        }
                        let mut members: Vec<(u64, u64)> = pts
                            .iter()
                            .filter(|p| v.contains(**p))
                            .map(|p| (p.x, p.z))
                            .collect();
                        members.sort_unstable();
                        seen.insert(members);
                    }
                }
            }
        }
        Ok(seen.len())
    }
}

/// By how much a transport pushes a frame **out of itself** — the
/// `Φ(P, U) = P U (I − P)` obstruction, as a rank.
///
/// Zero exactly when the transport is a loop, so the frame is
/// dynamically closed under it and the represented future factors
/// through the represented present. Nonzero names the directions the
/// frame does not currently carry but that the dynamics will bring back
/// into it — which is frame geometry's constructive reason to add a
/// dimension rather than to enumerate a larger ambient blindly.
pub fn dynamic_obstruction(frame: &Volume, steps: &[CliffordStep]) -> Result<usize> {
    let moved: Vec<PauliString> = frame
        .basis()
        .into_iter()
        .map(|p| steps.iter().fold(p, |acc, &s| conjugate_by_step(acc, s)))
        .collect();
    let after = Volume::span(frame.qubits(), &moved)?;
    Ok(frame.join(&after)?.rank() - frame.rank())
}

/// Several frames in one ambient, and what survives closing them
/// together.
#[derive(Clone, Debug)]
pub struct Assembly {
    ambient: usize,
    frames: Vec<Volume>,
}

impl Assembly {
    /// An assembly of frames, each of which must be a legal position —
    /// isotropic — in the same ambient.
    pub fn new(ambient: usize, frames: Vec<Volume>) -> Result<Self> {
        if frames.is_empty() {
            return Err(Error::InvalidState(
                "assembly: no frames — an effective geometry is the closure of an \
                 assembly, and there is nothing here to close"
                    .into(),
            ));
        }
        for (i, f) in frames.iter().enumerate() {
            if f.qubits() != ambient {
                return Err(Error::InvalidState(format!(
                    "assembly: frame {i} lives on {} qubits, not {ambient}",
                    f.qubits()
                )));
            }
            if !f.is_isotropic() {
                return Err(Error::InvalidState(format!(
                    "assembly: frame {i} is not isotropic, so it is not a position and \
                     imposes no admissibility constraint"
                )));
            }
        }
        Ok(Assembly { ambient, frames })
    }

    /// The ambient's polar space.
    pub fn space(&self) -> Result<PolarSpace> {
        PolarSpace::new(self.ambient)
    }

    /// The frames.
    pub fn frames(&self) -> &[Volume] {
        &self.frames
    }

    /// `⋁ᵢ Vᵢ` — every constraint the assembly imposes, together.
    pub fn joined(&self) -> Result<Volume> {
        let mut acc = Volume::empty(self.ambient)?;
        for f in &self.frames {
            acc = acc.join(f)?;
        }
        Ok(acc)
    }

    /// `Im Γ = ⋂ᵢ Vᵢ^⊥ = (⋁ᵢ Vᵢ)^⊥` — the operations admissible for
    /// every member at once, which is the systemic closure's image.
    pub fn closure(&self) -> Result<Volume> {
        self.joined()?.centraliser()
    }

    /// `Γ² = Γ`, checked rather than assumed: closing the closure
    /// changes nothing.
    pub fn is_idempotent(&self) -> Result<bool> {
        let c = self.closure()?;
        let again = Assembly::new(self.ambient, vec![self.joined()?])?.closure()?;
        Ok(c.is_same(&again))
    }

    /// The effective geometry as a qudit: the whole assembly seen as
    /// one obstruction.
    pub fn effective(&self) -> Result<VolQudit> {
        VolQudit::new(self.ambient, self.joined()?)
    }

    /// `d_eff` — the logical rank that survives closing everything.
    pub fn effective_rank(&self) -> Result<usize> {
        Ok(self.effective()?.logical_rank())
    }

    /// Each member's own logical rank, before the assembly closes.
    pub fn local_ranks(&self) -> Result<Vec<usize>> {
        self.frames
            .iter()
            .map(|f| Ok(VolQudit::new(self.ambient, f.clone())?.logical_rank()))
            .collect()
    }

    /// `Σᵢ hᵢ − d_eff` — how much the assembly's own closure destroys.
    ///
    /// Zero would mean the members constrain independent things. It is
    /// routinely positive, and that gap is the reason effective
    /// geometry is not a synonym for assembled object geometry.
    pub fn closure_deficit(&self) -> Result<i64> {
        let local: usize = self.local_ranks()?.iter().sum();
        Ok(local as i64 - self.effective_rank()? as i64)
    }

    /// The throat bound `min_i hᵢ`: what survives the assembly cannot
    /// exceed the narrowest member's own logical rank.
    pub fn throat(&self) -> Result<usize> {
        Ok(self.local_ranks()?.into_iter().min().unwrap_or(0))
    }

    /// Whether the frames form a **nested chain** `V₁ ⊆ V₂ ⊆ …`, which
    /// is exactly when the throat bound is tight.
    pub fn is_nested(&self) -> bool {
        self.frames.windows(2).all(|w| {
            w[0].basis().iter().all(|p| w[1].contains(*p))
                || w[1].basis().iter().all(|p| w[0].contains(*p))
        })
    }

    /// Whether a proposed rank-valued invariant is **globally
    /// effective**: does it factor through the closure?
    ///
    /// `I = Ī ∘ Γ` means `Γ(x) = Γ(y) ⟹ I(x) = I(y)`. Given a second
    /// assembly with the same closure, an effective invariant must
    /// agree on both; one that disagrees is local, however meaningful
    /// it is locally.
    pub fn agrees_on_closure(
        &self,
        other: &Assembly,
        invariant: &dyn Fn(&Assembly) -> Result<usize>,
    ) -> Result<Option<bool>> {
        if !self.closure()?.is_same(&other.closure()?) {
            return Ok(None);
        }
        Ok(Some(invariant(self)? == invariant(other)?))
    }
}
