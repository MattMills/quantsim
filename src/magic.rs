//! Positions in the magic n-fold, as exact arithmetic: the dimension of a
//! magic space, its Bott grading, its folds, and the compound tower.
//!
//! The Freudenthal–Tits construction `M(A₁,…,Aₙ)` takes composition-algebra
//! slots and returns a dimension. This module is that arithmetic and
//! **only** that arithmetic: it computes dimensions, layer decompositions
//! and gradings. It does **not** realize the Lie algebras — `M(𝕆,𝕆) = 248`
//! is computed here as a number, and `E₈` as a bracket is somewhere else
//! entirely. Keeping that line visible is the point; a dimension that
//! matches is not a construction that works.
//!
//! ## The layer decomposition
//!
//! Writing `tᵢ` for a slot's interior (`dim Aᵢ − 1`) and `derᵢ` for its
//! derivation algebra, the dimension splits into Tits layers:
//!
//! ```text
//! dim = 3 + Σ derᵢ + 5·e₁(t) + 3·e₂(t) + Σ_{k≥3} e_k(t)
//! ```
//!
//! with `e_k` the elementary symmetric polynomials. The `3` is the shared
//! `so(3)` frame; `e₂` is the pairwise coupling between slots; `e_n = ∏ tᵢ`
//! is the **interior** — the genuinely n-body term, invisible to every
//! face. [`Position::layers`] reports all of them, and
//! `the_layer_formula_reproduces_the_magic_square` pins it against `F₄`,
//! `E₆`, `E₇`, `E₈` and `so(12)`.
//!
//! ## Two infinite axes and a circle
//!
//! * **Arity** — more slots. On the all-octonionic line the layer sum has a
//!   closed form, [`octonionic_dim`]:
//!   `dim M(𝕆ⁿ) = 8ⁿ + 98·C(n,2) + 42n + 2`. The leading term is exactly
//!   `dim 𝕆^{⊗n}`, so the n-fold is the octonion tensor power plus a
//!   quadratic correction.
//! * **Depth** — a whole position fed back in as one slot ([`Position::nest`]).
//!   The recurrence is `dimₖ₊₁ ≈ 3·dimₖ²`: doubly exponential, so the tower
//!   is *generated*, never enumerated. It leaves `u128` at the fourth rung
//!   and this module **refuses** there rather than wrapping — see
//!   `the_tower_refuses_rather_than_overflowing`.
//! * **The Bott grading** is a circle, not an axis: `dim mod 8`, and on the
//!   octonionic line it collapses to the quadratic
//!   `(n² + n + 2) mod 8` ([`octonionic_bott`]) with period 8. Because
//!   `n(n+1)` is always even, **every** magic n-fold lands in the even
//!   sector `{0,2,4,6}`.
//!
//! The grading is the useful part operationally: it is a function of
//! *position alone*, so two parties holding coordinates agree on it with
//! no communication and `O(1)` work. What that buys is a shared
//! **convention** — a label both compute independently. It is not a shared
//! physical phase reference, and nothing here manufactures one.
//!
//! ## The third axis is closed
//!
//! Slots are drawn from [`Composition`], which has exactly four values.
//! That is Hurwitz's theorem, not a modelling choice: past `dim 8` the
//! composition law fails and the norm stops being multiplicative. So the
//! geometry is `∞ × ∞ × ℤ/8 × 4`, and the finiteness of the last factor is
//! a theorem rather than a cap this module imposes.

use crate::error::{Error, Result};

/// A composition algebra slot. Hurwitz's theorem closes this list at four:
/// `ℝ, ℂ, ℍ, 𝕆` are the only normed division algebras, so there is no
/// fifth variant to add.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Composition {
    /// The reals, `dim 1`.
    R,
    /// The complexes, `dim 2`.
    C,
    /// The quaternions, `dim 4`.
    H,
    /// The octonions, `dim 8`.
    O,
}

impl Composition {
    /// Every composition algebra, in dimension order.
    pub const ALL: [Composition; 4] = [
        Composition::R,
        Composition::C,
        Composition::H,
        Composition::O,
    ];

    /// Real dimension: `1, 2, 4, 8`.
    pub fn dim(self) -> u128 {
        match self {
            Composition::R => 1,
            Composition::C => 2,
            Composition::H => 4,
            Composition::O => 8,
        }
    }

    /// The slot's **interior** `t = dim − 1` — its imaginary part, which is
    /// what couples to the other slots.
    pub fn interior(self) -> u128 {
        self.dim() - 1
    }

    /// `dim Der(A)`: `0, 0, 3, 14`. The ladder grows to `g₂` and stops —
    /// and it stays at `g₂` past the octonions, which is why the
    /// automorphism thread does not dead-end where the division-algebra
    /// thread does.
    pub fn derivations(self) -> u128 {
        match self {
            Composition::R | Composition::C => 0,
            Composition::H => 3,
            Composition::O => 14,
        }
    }

    /// Short name.
    pub fn name(self) -> &'static str {
        match self {
            Composition::R => "R",
            Composition::C => "C",
            Composition::H => "H",
            Composition::O => "O",
        }
    }
}

/// One slot of a position: a bare composition algebra, or an entire
/// position nested one level down — the move that makes the depth axis.
#[derive(Debug, Clone, PartialEq)]
pub enum Slot {
    /// A composition algebra.
    Algebra(Composition),
    /// A whole magic position, fed back in as one slot.
    Nested(Box<Position>),
}

/// The Tits layers of a position's dimension, each reported separately so
/// the interior can be told apart from the coupling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layers {
    /// The shared `so(3)` frame — always 3.
    pub frame: u128,
    /// `Σ derᵢ`, the slots' derivation algebras.
    pub derivations: u128,
    /// `5·e₁`, linear in the slot interiors.
    pub linear: u128,
    /// `3·e₂`, the pairwise coupling between slots.
    pub pairwise: u128,
    /// `Σ_{k≥3} e_k`, everything genuinely higher than 2-body.
    pub higher: u128,
    /// The sum of the layers.
    pub total: u128,
}

/// A position in the magic n-fold: an ordered list of slots.
#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    slots: Vec<Slot>,
}

fn add(a: u128, b: u128) -> Result<u128> {
    a.checked_add(b)
        .ok_or_else(|| Error::InvalidState("magic: the position's dimension exceeds u128".into()))
}

fn mul(a: u128, b: u128) -> Result<u128> {
    a.checked_mul(b)
        .ok_or_else(|| Error::InvalidState("magic: the position's dimension exceeds u128".into()))
}

impl Position {
    /// A position from explicit slots. At least one slot is required — a
    /// magic space with no slots is not a degenerate case, it is not a
    /// magic space.
    pub fn new(slots: Vec<Slot>) -> Result<Self> {
        if slots.is_empty() {
            return Err(Error::InvalidState(
                "magic: a position needs at least one slot".into(),
            ));
        }
        Ok(Position { slots })
    }

    /// `M(A₁,…,Aₙ)` from bare algebras.
    pub fn algebras(algebras: &[Composition]) -> Result<Self> {
        Position::new(algebras.iter().copied().map(Slot::Algebra).collect())
    }

    /// `M(𝕆ⁿ)` — the all-octonionic line, the exceptional row.
    pub fn octonionic(n: usize) -> Result<Self> {
        Position::algebras(&vec![Composition::O; n])
    }

    /// The slots.
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    /// Number of slots.
    pub fn arity(&self) -> usize {
        self.slots.len()
    }

    /// Nesting depth: 1 for a flat position, 2 for a position of positions.
    pub fn depth(&self) -> usize {
        1 + self
            .slots
            .iter()
            .map(|s| match s {
                Slot::Algebra(_) => 0,
                Slot::Nested(p) => p.depth(),
            })
            .max()
            .unwrap_or(0)
    }

    /// This position's own derivation layer, `Σ derᵢ` — what it contributes
    /// as a slot one level up. It excludes the `so(3)` frame, which is
    /// shared rather than carried.
    pub fn derivation_layer(&self) -> Result<u128> {
        let mut acc = 0u128;
        for s in &self.slots {
            acc = add(acc, self.slot_der(s)?)?;
        }
        Ok(acc)
    }

    fn slot_der(&self, s: &Slot) -> Result<u128> {
        match s {
            Slot::Algebra(a) => Ok(a.derivations()),
            Slot::Nested(p) => p.derivation_layer(),
        }
    }

    fn slot_interior(&self, s: &Slot) -> Result<u128> {
        match s {
            Slot::Algebra(a) => Ok(a.interior()),
            Slot::Nested(p) => Ok(p.dim()? - 1),
        }
    }

    /// The slots' interiors `tᵢ`.
    pub fn interiors(&self) -> Result<Vec<u128>> {
        self.slots.iter().map(|s| self.slot_interior(s)).collect()
    }

    /// The elementary symmetric polynomial `e_k` of the slot interiors.
    fn esym(t: &[u128], k: usize) -> Result<u128> {
        if k == 0 {
            return Ok(1);
        }
        if k > t.len() {
            return Ok(0);
        }
        let mut dp = vec![0u128; k + 1];
        dp[0] = 1;
        for &x in t {
            for j in (1..=k).rev() {
                dp[j] = add(dp[j], mul(dp[j - 1], x)?)?;
            }
        }
        Ok(dp[k])
    }

    /// The layer decomposition. Every term is exact; the sum is
    /// [`Position::dim`].
    pub fn layers(&self) -> Result<Layers> {
        let t = self.interiors()?;
        let derivations = self.derivation_layer()?;
        let linear = mul(5, Self::esym(&t, 1)?)?;
        let pairwise = mul(3, Self::esym(&t, 2)?)?;
        let mut higher = 0u128;
        for k in 3..=t.len() {
            higher = add(higher, Self::esym(&t, k)?)?;
        }
        let total = add(add(add(add(3, derivations)?, linear)?, pairwise)?, higher)?;
        Ok(Layers {
            frame: 3,
            derivations,
            linear,
            pairwise,
            higher,
            total,
        })
    }

    /// The position's dimension.
    pub fn dim(&self) -> Result<u128> {
        Ok(self.layers()?.total)
    }

    /// The **interior** `∏ tᵢ` — the n-body term no face can see. For a
    /// single slot this is that slot's own interior.
    pub fn interior(&self) -> Result<u128> {
        let t = self.interiors()?;
        let mut acc = 1u128;
        for x in t {
            acc = mul(acc, x)?;
        }
        Ok(acc)
    }

    /// The **Bott grading**: `dim mod 8`.
    ///
    /// A function of position alone, so two holders of coordinates agree on
    /// it with no communication. That makes it a shared *convention*; it is
    /// not a shared physical phase reference.
    pub fn bott(&self) -> Result<u8> {
        Ok((self.dim()? % 8) as u8)
    }

    /// **Folding inward**: the codim-1 faces, one per slot omitted. A face
    /// of `M(𝕆,𝕆,𝕆)` is `M(𝕆,𝕆) = 248`.
    pub fn faces(&self) -> Vec<Position> {
        if self.slots.len() < 2 {
            return Vec::new();
        }
        (0..self.slots.len())
            .map(|i| {
                let mut s = self.slots.clone();
                s.remove(i);
                Position { slots: s }
            })
            .collect()
    }

    /// **Folding outward in depth**: this position fed back in as every
    /// slot of a new one, `arity` times over.
    pub fn nest(&self, arity: usize) -> Result<Position> {
        if arity == 0 {
            return Err(Error::InvalidState(
                "magic: nesting needs at least one slot".into(),
            ));
        }
        Position::new(vec![Slot::Nested(Box::new(self.clone())); arity])
    }

    /// A one-line label, e.g. `M(O,O)` or `M(M(O,O),M(O,O))`.
    pub fn label(&self) -> String {
        let inner: Vec<String> = self
            .slots
            .iter()
            .map(|s| match s {
                Slot::Algebra(a) => a.name().to_string(),
                Slot::Nested(p) => p.label(),
            })
            .collect();
        format!("M({})", inner.join(","))
    }
}

/// The closed form on the all-octonionic line:
/// `dim M(𝕆ⁿ) = 8ⁿ + 98·C(n,2) + 42n + 2`.
///
/// `O(1)` where the layer sum is `O(n²)`, and the leading `8ⁿ` is exactly
/// `dim 𝕆^{⊗n}` — the n-fold is the octonion tensor power plus a quadratic
/// correction. Agreement with the layer sum is pinned by
/// `the_closed_form_agrees_with_the_layer_sum`.
pub fn octonionic_dim(n: u32) -> Result<u128> {
    let pow = 8u128
        .checked_pow(n)
        .ok_or_else(|| Error::InvalidState(format!("magic: 8^{n} exceeds u128")))?;
    let n128 = n as u128;
    let choose2 = mul(n128, n128.saturating_sub(1))? / 2;
    add(add(add(pow, mul(98, choose2)?)?, mul(42, n128)?)?, 2)
}

/// The Bott grading on the octonionic line, `(n² + n + 2) mod 8`.
///
/// Period 8 in `n`, and always even — `n(n+1)` is even for every `n`, so
/// the whole arity axis lives in the even sector `{0,2,4,6}`.
pub fn octonionic_bott(n: u32) -> u8 {
    let n = (n % 8) as u16;
    ((n * n + n + 2) % 8) as u8
}
