//! Space-filling curve orderings, and what a lattice costs a chain
//! register.
//!
//! An MPS, a path-sum variable list and a qubit index are all **one
//! dimensional**. A coupling fabric usually is not. The map between
//! them is an *ordering*, and it is not free: every lattice edge whose
//! endpoints land far apart on the chain is a bond the representation
//! has to carry, or a swap the router has to insert.
//!
//! This module makes the ordering a first-class, measured object.
//!
//! ## The orderings
//!
//! * [`Order::RowMajor`] — the obvious one, and the bad one: the last
//!   site of a row is adjacent to the first site of the next, so every
//!   row boundary is a dilation-`side` edge.
//! * [`Order::Snake`] — boustrophedon, reversing alternate rows. Fixes
//!   the row-boundary edges and nothing else; the vertical edges still
//!   span a row.
//! * [`Order::Hilbert`] — the space-filling curve, on a power-of-two
//!   side. Every step is a lattice step, and locality is preserved at
//!   *every* scale rather than only within a row.
//!
//! ## What is measured
//!
//! Two quantities, and they answer different questions:
//!
//! * **Dilation** ([`GridOrder::dilation`]) — `|i(u) − i(v)|` over the
//!   lattice edges. This is what a *router* pays: a two-site gate
//!   between chain positions `i` and `j` costs `|i − j| − 1` swaps on
//!   a nearest-neighbour chain, so the mean dilation is the swap bill.
//! * **Cut profile** ([`GridOrder::cut_profile`]) and its maximum, the
//!   **cutwidth** ([`GridOrder::cutwidth`]) — how many lattice edges
//!   cross each chain cut. This is what a *bond* pays: an MPS bond at
//!   position `k` has to carry every coupling that straddles `k`.
//!
//! ## The scope, stated first, because it is where the interest is
//!
//! There are two different things one can do with the Hilbert
//! construction, and they come out opposite ways.
//!
//! * Flatten it to a **linear index** and use that as an ordering.
//!   That is what [`GridOrder`] does, and it is *worse* than reading
//!   the rows, by exact laws, on every measure a chain cares about.
//! * Keep the **recursion** — the quadrant tree and its rotor per
//!   cell — and use that as the layout. That is [`Bisection`], and it
//!   beats every linear ordering by a factor that grows without limit:
//!   `Θ(√N)`, measured at 64.5× on a 128×128 grid.
//!
//! So the refutation below is real and it is *narrow*: it refutes the
//! curve **as a linear order**, which is the use that throws the
//! recursion away. The construction's content is the recursion, and
//! the second half of this module measures what keeping it buys.
//!
//! ## The result, which refutes the obvious expectation
//!
//! The expectation this module was written to test — *a curve that
//! preserves locality should lay a lattice onto a chain better than
//! reading it in rows* — is *wrong*, and wrong by an exact law. Every
//! form below holds at every power-of-two side from 2 to 128:
//!
//! ```text
//! ordering    max dilation          cutwidth
//! RowMajor    side                  side + 1
//! Snake       2·side − 1            side + 1
//! Hilbert     (10·4^{k−1} − 1)/3    2·side − 2        (side = 2^k)
//! ```
//!
//! So the Hilbert curve's cutwidth is **exactly twice** row-major's,
//! and its worst dilation is `Θ(N)` — quadratic in the side — against
//! row-major's `Θ(√N)`. Mean dilation, total crossings and the routing
//! bill all move the same way: at side 32, Hilbert pays 36,952 swaps
//! against row-major's 30,752 and a cutwidth of 62 against 33. It is
//! worse on every measure that a chain register cares about, and the
//! only column where it wins — the fraction of lattice edges that land
//! chain-adjacent — is one the snake matches.
//!
//! **The reason is a direction error, and it is worth stating because
//! the intuition is so natural.** A space-filling curve preserves
//! locality from *curve to plane*: nearby indices are nearby points.
//! That is what makes it the right tool for spatial indexing and cache
//! layout. A chain register needs the *opposite* direction — nearby
//! points must get nearby indices — and the Hilbert curve is provably
//! bad at that, its inverse having unbounded dilation. Reading the
//! rows is bad at the first direction and optimal at the second.
//!
//! Half of the expectation stated before the run did survive: cutwidth
//! is `Θ(side)` under every ordering, a property of the grid rather
//! than of the curve, so no relabelling beats it asymptotically. The
//! other half — that the curve would at least buy a better routing
//! bill — did not, and is recorded as refuted.
//!
//! ## Keeping the recursion instead: what the rotor is worth
//!
//! The Hilbert construction is not really an ordering. It is a
//! **rotor per cell, applied recursively**: each quadrant's sub-curve
//! is entered under a dihedral symmetry ([`Rotor`], the eight elements
//! of `D₄`), and the rule is the assignment of rotors to quadrants.
//! [`hilbert_rotors`] is that assignment — transpose, identity,
//! identity, anti-transpose — and it is the whole of what distinguishes
//! the Hilbert curve from any other order-and-twist rule.
//!
//! A linear index throws that structure away: it keeps only the order
//! the recursion happens to visit cells in. [`Bisection`] keeps the
//! recursion itself, as the tree of quadrant cuts, and prices the
//! layout by the separator at each tree node instead of by the prefix
//! cuts of a chain. Measured against the best linear ordering:
//!
//! ```text
//! side   chain total   tree total   ratio     max separator
//!    8           504          112    4.5x     8  both
//!   32        32 736        1 984   16.5x    32  both
//!  128     2 097 024       32 512   64.5x   128  both
//! ```
//!
//! The **maximum** separator is identical — `side`, the grid's own
//! bound, which no layout of any kind beats — and that is the half of
//! the earlier expectation that survived, now confirmed from the other
//! side. But the **totals** have exact closed forms and they differ by
//! an order:
//!
//! ```text
//! chain (best ordering)   side³ − side       = Θ(N^{3/2})
//! tree                    2·side² − 2·side   = Θ(N)
//! ratio                   (side + 1) / 2     = Θ(√N)
//! ```
//!
//! so the gap grows without limit. One correction to the obvious
//! reading of that, since the shape is not what one would guess: the
//! tree's cost is **not** concentrated at the root. Its per-level
//! profile is `side · 2^{⌊ℓ/2⌋}` — doubling every two levels — so the
//! total is dominated by the *deepest* cuts. The separators shrink per
//! node and the node count grows faster. A representation whose cost is the sum over cuts — a
//! hierarchy, [`MeraState`](crate::backend::MeraState) or
//! [`BulkState`](crate::backend::BulkState) — pays the second column; one whose
//! cost is the sum over a chain's cuts pays the first.
//!
//! **So the curve was never the wrong idea; flattening it was.** What
//! is *not* shown here, and is worth separating from what is: nothing
//! below tests whether a specific **overlay of several** rotor
//! assignments — distinct curves used together as independent
//! addressing bits rather than one at a time — improves on plain
//! recursive bisection. That is a stronger claim than the one measured
//! and it stays open.
//!
//! ## Provenance
//!
//! The curve family here is the `n = 2` case of a construction the
//! sibling `fractal_research` program searches in general: a rule is a
//! base order on the `2^n` sub-cells plus a **twist per slot** drawn
//! from the hypercube symmetry group `F₂ⁿ ⋊ Sₙ` of order `2ⁿ n!`, and
//! the curve is the recursive closure of the rule. That program's
//! first result is a *refutation* worth carrying across: the
//! presentation is faithful — at `n = 2` all 98,304 syntactic rules
//! have 98,304 distinct level-3 point sequences, so unlike a gate set
//! there is **no semantic quotient** to collapse the search onto.
//! Nothing here searches the family; this module implements the one
//! rule everyone means by "the Hilbert curve" and prices it.

use crate::error::{Error, Result};

/// How a `side × side` lattice is laid onto a chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Read the rows in order, each left to right.
    RowMajor,
    /// Boustrophedon: alternate rows reversed, so consecutive chain
    /// positions are always lattice neighbours *within* a row.
    Snake,
    /// The Hilbert curve. Requires a power-of-two side, and every
    /// consecutive pair is a lattice neighbour at every scale.
    Hilbert,
}

impl Order {
    /// The largest `|i(u) − i(v)|` this ordering produces on a
    /// `side × side` lattice, in closed form.
    ///
    /// `RowMajor` is `side`, `Snake` is `2·side − 1`, and `Hilbert` is
    /// `(10·4^{k−1} − 1)/3` for `side = 2^k` — quadratic in the side
    /// where the other two are linear. Pinned against enumeration at
    /// every side from 2 to 128 in `tests/curve.rs`.
    pub fn max_dilation_law(self, side: usize) -> Option<usize> {
        match self {
            Order::RowMajor => Some(side),
            Order::Snake => Some(2 * side - 1),
            Order::Hilbert => {
                if !side.is_power_of_two() {
                    return None;
                }
                let k = side.trailing_zeros();
                Some((10 * 4usize.checked_pow(k - 1)? - 1) / 3)
            }
        }
    }

    /// The largest number of lattice edges crossing any chain cut, in
    /// closed form: `side + 1` for the row orders and `2·side − 2` for
    /// the Hilbert curve — exactly twice, for every side above 2.
    pub fn cutwidth_law(self, side: usize) -> Option<usize> {
        if side < 4 {
            return None;
        }
        match self {
            Order::RowMajor | Order::Snake => Some(side + 1),
            Order::Hilbert => side.is_power_of_two().then(|| 2 * side - 2),
        }
    }
}

/// The dilation census of an ordering: what the router pays.
#[derive(Clone, Debug, PartialEq)]
pub struct Dilation {
    /// Largest `|i(u) − i(v)|` over lattice edges.
    pub max: usize,
    /// Mean, as an exact rational rendered late.
    pub total: usize,
    /// Lattice edges counted.
    pub edges: usize,
    /// How many edges have dilation exactly 1 — the free ones.
    pub adjacent: usize,
}

impl Dilation {
    /// Mean dilation.
    pub fn mean(&self) -> f64 {
        if self.edges == 0 {
            0.0
        } else {
            self.total as f64 / self.edges as f64
        }
    }

    /// Fraction of lattice edges that are chain-adjacent, so free to a
    /// nearest-neighbour router.
    pub fn adjacent_fraction(&self) -> f64 {
        if self.edges == 0 {
            0.0
        } else {
            self.adjacent as f64 / self.edges as f64
        }
    }

    /// Swaps a nearest-neighbour router pays to bring every lattice
    /// edge together once: `Σ (dilation − 1)`.
    pub fn routing_cost(&self) -> usize {
        self.total - self.edges
    }
}

/// A `side × side` lattice laid onto a chain by a named ordering.
#[derive(Clone, Debug)]
pub struct GridOrder {
    side: usize,
    order: Order,
    /// `to_chain[y * side + x]` is the chain position of lattice `(x, y)`.
    to_chain: Vec<usize>,
    /// `to_point[i]` is the lattice point at chain position `i`.
    to_point: Vec<(usize, usize)>,
}

/// The standard Hilbert `d → (x, y)` map on a `side × side` grid.
fn hilbert_point(side: usize, mut t: usize) -> (usize, usize) {
    let (mut x, mut y) = (0usize, 0usize);
    let mut s = 1usize;
    while s < side {
        let rx = 1 & (t / 2);
        let ry = 1 & (t ^ rx);
        // Rotate the quadrant into place.
        if ry == 0 {
            if rx == 1 {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        x += s * rx;
        y += s * ry;
        t /= 4;
        s *= 2;
    }
    (x, y)
}

impl GridOrder {
    /// Lay a `side × side` lattice onto a chain.
    ///
    /// [`Order::Hilbert`] needs a power-of-two side and refuses
    /// otherwise rather than padding to one, because a padded grid is
    /// a different lattice and would quietly change what is measured.
    pub fn new(side: usize, order: Order) -> Result<Self> {
        if side == 0 {
            return Err(Error::InvalidState("grid order: side 0".into()));
        }
        if order == Order::Hilbert && !side.is_power_of_two() {
            return Err(Error::InvalidState(format!(
                "grid order: the Hilbert curve needs a power-of-two side, got {side} — \
                 padding to one would measure a different lattice, so this refuses \
                 instead"
            )));
        }
        let n = side * side;
        let mut to_chain = vec![0usize; n];
        let mut to_point = vec![(0usize, 0usize); n];
        for (i, slot) in to_point.iter_mut().enumerate().take(n) {
            let (x, y) = match order {
                Order::RowMajor => (i % side, i / side),
                Order::Snake => {
                    let row = i / side;
                    let col = i % side;
                    (if row % 2 == 0 { col } else { side - 1 - col }, row)
                }
                Order::Hilbert => hilbert_point(side, i),
            };
            *slot = (x, y);
            to_chain[y * side + x] = i;
        }
        Ok(GridOrder {
            side,
            order,
            to_chain,
            to_point,
        })
    }

    /// Lattice side.
    pub fn side(&self) -> usize {
        self.side
    }

    /// Sites.
    pub fn sites(&self) -> usize {
        self.side * self.side
    }

    /// The ordering.
    pub fn order(&self) -> Order {
        self.order
    }

    /// Chain position of a lattice point.
    pub fn chain_index(&self, x: usize, y: usize) -> Result<usize> {
        if x >= self.side || y >= self.side {
            return Err(Error::InvalidState(format!(
                "grid order: ({x}, {y}) is outside a {}×{} lattice",
                self.side, self.side
            )));
        }
        Ok(self.to_chain[y * self.side + x])
    }

    /// Lattice point at a chain position.
    pub fn lattice_point(&self, i: usize) -> Result<(usize, usize)> {
        self.to_point.get(i).copied().ok_or_else(|| {
            Error::InvalidState(format!("grid order: chain position {i} out of range"))
        })
    }

    /// Whether consecutive chain positions are always lattice
    /// neighbours — the defining property of a space-filling curve,
    /// checked rather than assumed.
    pub fn is_continuous(&self) -> bool {
        self.to_point.windows(2).all(|w| {
            let (a, b) = (w[0], w[1]);
            a.0.abs_diff(b.0) + a.1.abs_diff(b.1) == 1
        })
    }

    /// Every lattice edge, as a pair of chain positions.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for y in 0..self.side {
            for x in 0..self.side {
                let a = self.to_chain[y * self.side + x];
                if x + 1 < self.side {
                    out.push((a, self.to_chain[y * self.side + x + 1]));
                }
                if y + 1 < self.side {
                    out.push((a, self.to_chain[(y + 1) * self.side + x]));
                }
            }
        }
        out
    }

    /// What the router pays.
    pub fn dilation(&self) -> Dilation {
        let edges = self.edges();
        let mut max = 0;
        let mut total = 0;
        let mut adjacent = 0;
        for (a, b) in &edges {
            let d = a.abs_diff(*b);
            max = max.max(d);
            total += d;
            if d == 1 {
                adjacent += 1;
            }
        }
        Dilation {
            max,
            total,
            edges: edges.len(),
            adjacent,
        }
    }

    /// How many lattice edges cross each chain cut — what the bonds
    /// pay. Entry `k` counts edges straddling the cut between chain
    /// positions `k` and `k + 1`.
    pub fn cut_profile(&self) -> Vec<usize> {
        let n = self.sites();
        let mut profile = vec![0usize; n.saturating_sub(1)];
        for (a, b) in self.edges() {
            let (lo, hi) = (a.min(b), a.max(b));
            for slot in profile.iter_mut().take(hi).skip(lo) {
                *slot += 1;
            }
        }
        profile
    }

    /// The largest cut — the bound no bond can go under.
    pub fn cutwidth(&self) -> usize {
        self.cut_profile().into_iter().max().unwrap_or(0)
    }

    /// Total edge-crossings summed over every cut. Where `cutwidth` is
    /// the worst bond, this is the whole bond bill, and it is the one
    /// an ordering can actually move.
    pub fn total_crossings(&self) -> usize {
        self.cut_profile().into_iter().sum()
    }
}

// ─────────────── the recursion, which is what the curve is ───────────────

/// A dihedral symmetry of the square: the eight elements of `D₄`.
///
/// This is the object the Hilbert construction is actually made of. A
/// rule assigns one rotor to each sub-cell, and the curve is the
/// recursive closure of that assignment — so the rotor, not the index,
/// is the carrier of the structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Rotor {
    /// Reflect across the leading diagonal (transpose) before turning.
    pub transpose: bool,
    /// Quarter turns applied after the reflection, `0..4`.
    pub quarter_turns: u8,
}

impl Rotor {
    /// The identity rotor.
    pub const IDENTITY: Rotor = Rotor {
        transpose: false,
        quarter_turns: 0,
    };

    /// A rotor, with the turn count reduced mod 4.
    pub fn new(transpose: bool, quarter_turns: u8) -> Rotor {
        Rotor {
            transpose,
            quarter_turns: quarter_turns % 4,
        }
    }

    /// All eight elements of `D₄`, in a fixed order.
    pub fn all() -> Vec<Rotor> {
        (0..4)
            .flat_map(|q| [Rotor::new(false, q), Rotor::new(true, q)])
            .collect()
    }

    /// Apply to a point of a `size × size` cell.
    pub fn apply(self, size: usize, x: usize, y: usize) -> (usize, usize) {
        let (mut x, mut y) = if self.transpose { (y, x) } else { (x, y) };
        for _ in 0..self.quarter_turns {
            let nx = size - 1 - y;
            y = x;
            x = nx;
        }
        (x, y)
    }

    /// Whether this rotor is orientation-reversing — the "reflection"
    /// half of `D₄`, which is what a transpose contributes.
    pub fn is_reflection(self) -> bool {
        self.transpose
    }
}

/// The rotor the Hilbert rule applies to each of the four quadrants,
/// in visit order.
///
/// Transpose, identity, identity, anti-transpose. Two reflections and
/// two rotations — the assignment that makes the four sub-curves join
/// end to end, and the entire content of "the Hilbert curve" as
/// against any other order-and-twist rule.
pub fn hilbert_rotors() -> [Rotor; 4] {
    [
        Rotor::new(true, 0),
        Rotor::IDENTITY,
        Rotor::IDENTITY,
        Rotor::new(true, 2),
    ]
}

/// One node of a recursive bisection: the region it cuts and how many
/// lattice edges the cut severs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cut {
    /// Depth in the recursion, root at 0.
    pub depth: usize,
    /// Lattice edges this cut severs — the node's separator.
    pub separator: usize,
    /// The region cut, as `(x, y, width, height)`.
    pub region: (usize, usize, usize, usize),
}

/// The lattice laid out as a **recursive bisection** rather than as a
/// chain — the Hilbert construction's own structure, kept.
///
/// Where a [`GridOrder`] is priced by the prefix cuts of a chain, this
/// is priced by the separator at each node of the quadrant tree. The
/// two agree on the *worst* cut, which is the grid's own bound, and
/// differ by `Θ(√N)` on the total.
#[derive(Clone, Debug)]
pub struct Bisection {
    side: usize,
    cuts: Vec<Cut>,
}

impl Bisection {
    /// Bisect a `side × side` lattice recursively, always cutting the
    /// longer axis, which is the standard nested-dissection choice and
    /// the one the quadrant recursion makes.
    pub fn quadrants(side: usize) -> Result<Self> {
        if side == 0 {
            return Err(Error::InvalidState("bisection: side 0".into()));
        }
        let mut cuts = Vec::new();
        Self::split(0, 0, side, side, 0, &mut cuts);
        Ok(Bisection { side, cuts })
    }

    fn split(x: usize, y: usize, w: usize, h: usize, depth: usize, out: &mut Vec<Cut>) {
        if w <= 1 && h <= 1 {
            return;
        }
        if w >= h {
            let half = w / 2;
            out.push(Cut {
                depth,
                separator: h,
                region: (x, y, w, h),
            });
            Self::split(x, y, half, h, depth + 1, out);
            Self::split(x + half, y, w - half, h, depth + 1, out);
        } else {
            let half = h / 2;
            out.push(Cut {
                depth,
                separator: w,
                region: (x, y, w, h),
            });
            Self::split(x, y, w, half, depth + 1, out);
            Self::split(x, y + half, w, h - half, depth + 1, out);
        }
    }

    /// Lattice side.
    pub fn side(&self) -> usize {
        self.side
    }

    /// Every cut.
    pub fn cuts(&self) -> &[Cut] {
        &self.cuts
    }

    /// Tree nodes — one per cut.
    pub fn nodes(&self) -> usize {
        self.cuts.len()
    }

    /// Recursion depth.
    pub fn depth(&self) -> usize {
        self.cuts.iter().map(|c| c.depth).max().unwrap_or(0)
    }

    /// The largest separator anywhere in the tree — the grid's own
    /// bound, and equal to what the best chain ordering achieves.
    pub fn max_separator(&self) -> usize {
        self.cuts.iter().map(|c| c.separator).max().unwrap_or(0)
    }

    /// Every separator summed — what a representation paying per cut
    /// actually pays, and the column where the recursion wins.
    pub fn total_separator(&self) -> usize {
        self.cuts.iter().map(|c| c.separator).sum()
    }

    /// Separators summed by depth, root first.
    pub fn profile(&self) -> Vec<usize> {
        let mut out = vec![0usize; self.depth() + 1];
        for c in &self.cuts {
            out[c.depth] += c.separator;
        }
        out
    }
}
