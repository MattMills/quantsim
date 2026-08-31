//! The address algebra: canonical residual forms as **composable
//! addresses**, and evaluation as a fold over the resulting DAG.
//!
//! [`PathSum::amplitude_merged`](crate::pathsum::PathSum::amplitude_merged)
//! already memoizes on a canonical form — two residuals that are the
//! same polynomial up to renaming have the same sum, so the second one
//! costs a lookup. But its memo has the type `CanonKey → C64`: an
//! address maps to a **value**, so nothing can be done with an address
//! except decode it. It is a memo table, not an algebra, and it is
//! rebuilt from empty on every query.
//!
//! This module closes the address space under composition. An
//! [`Addr`] names a node, and a [`Node`] is built *from other
//! addresses*:
//!
//! ```text
//! Zero
//! Product { scale, parts: [Addr] }     ω^turn · √2^half · ∏ parts
//! Branch  { zero: Addr, one: Addr }    the sum of the two children
//! ```
//!
//! Those are the three moves the merge solver already makes — the
//! rewrite rules closing a residual, the interaction graph factoring
//! into independent components, and branching one variable — written
//! as constructors instead of as control flow. So the solver composes
//! *symbols*, and a value is a fold performed once at the end, or never.
//!
//! ## What being an algebra buys, and it is not a micro-optimization
//!
//! * **The scale is exact.** A residual's prefactor was always
//!   `ω^turn · √2^half` with `turn` a dyadic angle
//!   ([`Turn`]) and `half` an integer; the old
//!   memo evaluated it to a `C64` immediately. [`Scale`] keeps it, so
//!   two nodes merge when they are the same symbol rather than when
//!   two floats happen to agree, and
//!   [`value_exact`](AddressSpace::value_exact) evaluates the whole DAG
//!   in the Clifford+T ring [`DOmega`] with no
//!   floating point anywhere.
//! * **Products commute on addresses.** [`product`](AddressSpace::product)
//!   sorts its parts before interning, so two factorings that differ
//!   only in order are one node.
//! * **Equal addresses are equal amplitudes**, decided in `O(1)` with
//!   no arithmetic at all. [`AddressSpace::same`] is the whole test.
//! * **The space outlives the query.** This is the one the memo could
//!   not do. A `CanonKey → C64` map is scoped to one call of
//!   `amplitude_merged`; an address space is content-addressed, so a
//!   second amplitude reuses the forms the first one built — and not
//!   only of the same circuit. That is Gosper's hashlife move —
//!   time-step at the level of canonical node ids rather than of
//!   cells — carried onto the phase polynomial.
//!
//! ## The measurement
//!
//! The claim worth testing is that the *address count* stays flat where
//! the *work* grows. [`AddressStats::builds`] counts composition calls
//! and [`AddressSpace::len`] counts distinct nodes; over a sweep
//! of basis states on one circuit the first grows with the queries and
//! the second saturates. `tests/address.rs` measures it, and
//! `examples/address_algebra.rs` prints it.
//!
//! ## Scope, stated
//!
//! The address is exact for any dyadic turn. *Exact evaluation* is
//! available on the eighth-turn fragment — the ring `D[ω]` the
//! [`exact`](crate::exact) module works in — and refuses by name
//! otherwise rather than rounding.
//! [`value`](AddressSpace::value) always works and is float.
//!
//! ## The rewrites, and what they were measured to be worth
//!
//! Four rewrites act on addresses, before anything is evaluated:
//! `0 + x = x`; `a + a = 2a`; `s·P + t·P = 0` when the scales
//! [cancel](Scale::cancels), which is destructive interference decided
//! **without knowing what `P` is**; and `s·CP + t·CQ = C·(s·P + t·Q)`,
//! the common factor. Plus product absorption, since scales form a
//! group under [`Scale::times`].
//!
//! Measured, they are worth little here, and the useful part of this
//! module is the instrument that says why.
//!
//! * **Common-factor extraction is a trade, not a win**, so it is
//!   **off by default** ([`set_factoring`](AddressSpace::set_factoring)).
//!   It replaces one `Sum` node with four, and on a 3×3 grid over 64
//!   queries it costs 1.6× the addresses to save 3% of the
//!   evaluations; on a 4×4 grid, 1.5× for 17%.
//! * **Product absorption never fires on its own.** A `Product`'s parts
//!   are component sums, so a nested product only appears once another
//!   rewrite has made one. With factoring off, the address count is
//!   identical to rewriting off entirely.
//! * **`a + a = 2a` and the scale cancellation have not been observed
//!   firing.** Not on 40,000 random Clifford+T circuits — and random
//!   circuits are close to the wrong instrument for this, since
//!   cancellation is a structural coincidence and random sampling
//!   destroys structure — nor on the structured families where it
//!   should live: mirror circuits (`C` then `C†`), symmetric graph
//!   states on cycles and complete graphs, and repeated identical
//!   blocks. What those show instead is *where the cancellation went*:
//!   a mirror circuit reduces to `h* = 0` and **two** addresses, so
//!   [`reduce`](crate::pathsum) has already taken all of it before the
//!   address level exists. That is evidence for where to look, not a
//!   proof that the rewrites cannot fire.
//!
//! Which suggests what the cancellation rewrite is actually for. `h*`
//! is what survives the reducer, and the reducer's job *is* to consume
//! interference — so a cancellation at the address level would be
//! interference the rewrite rules **missed**. Read that way
//! [`AddressStats::cancelled`] is not an optimization counter but an
//! **incompleteness detector for the reducer**, and it reads zero on
//! everything tried, structured and random alike. That is evidence the
//! rules are complete on what has been run, and it is the cheapest
//! standing check for the opposite.
//!
//! [`sum_census`](AddressSpace::sum_census) is the instrument, and it
//! bounds what *any* sum-rewrite could reach on a given circuit by
//! classifying every sum's operands. On the grids above, 74–99% of sums
//! have **disjoint** factor sets — no factoring rewrite can touch them
//! — and the `same_parts` row, which is the ceiling for a rewrite that
//! combines coefficients over a shared factor set, is 0.6–16%.
//!
//! ## What is measured about cost
//!
//! For a **single** query the address route performs exactly as many
//! compositions as
//! [`amplitude_merged`](crate::pathsum::PathSum::amplitude_merged)
//! performs nodes — measured equal on every basis state of every
//! circuit in `tests/address.rs`, which is what one expects since it is
//! the same recursion under the same pivot. So the per-query cost *is*
//! the merge solver's, and what this module adds is reuse between
//! queries. Whether the growth law across a circuit family changes is
//! **not measured here** and is not claimed either way.
//!

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::exact::DOmega;
use crate::pathsum::{canonicalize, CanonKey, Component, PathSum, Pivot, Turn, EIGHTH, HALF};
use crate::scalar::C64;

/// A content address: the identity of a node in an [`AddressSpace`].
///
/// Two addresses are equal exactly when they name the same canonical
/// node, and equal nodes have equal value — so [`AddressSpace::same`]
/// decides amplitude equality without evaluating either side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Addr(u32);

impl Addr {
    /// The address's index in its space.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A residual's prefactor, kept as the exact symbol it is:
/// `ω^turn · √2^half`, where `turn` is a dyadic angle and `half` counts
/// **half**-powers of two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Scale {
    /// The angle, exact (see [`Turn`]).
    pub turn: Turn,
    /// The exponent, in half-powers: the factor is `√2^half`.
    pub half: i64,
}

impl Scale {
    /// The unit scale: no phase, no scaling.
    pub const ONE: Scale = Scale { turn: 0, half: 0 };

    /// Whether the angle is an eighth-turn multiple, which is what
    /// [`AddressSpace::value_exact`] needs.
    pub fn is_eighth(&self) -> bool {
        self.turn % EIGHTH == 0
    }

    /// As a float.
    pub fn to_c64(self) -> C64 {
        let frac = self.turn as f64 / 2f64.powi(64);
        let theta = std::f64::consts::TAU * frac;
        C64::new(theta.cos(), theta.sin()) * C64::new(2f64.powf(self.half as f64 / 2.0), 0.0)
    }

    /// In the Clifford+T ring, or a named refusal when the angle is not
    /// an eighth turn.
    pub fn to_exact(self) -> Result<DOmega> {
        if !self.is_eighth() {
            return Err(Error::InvalidState(format!(
                "address: turn {} is not a multiple of an eighth, so it is not in \
                 D[ω] — evaluate this address with `value` instead of `value_exact` \
                 rather than having it rounded",
                self.turn
            )));
        }
        let omega = DOmega::omega_pow((self.turn / EIGHTH) as i64);
        let root = sqrt2_pow(self.half)?;
        omega.mul(root)
    }
}

impl Scale {
    /// The scale of a product of two scales: turns add, exponents add.
    /// Exact, and the reason a `Product` can absorb a nested one.
    pub fn times(self, rhs: Scale) -> Scale {
        Scale {
            turn: self.turn.wrapping_add(rhs.turn),
            half: self.half + rhs.half,
        }
    }

    /// Whether `self + rhs` is exactly zero: the same magnitude at
    /// opposite angles. Decided on the symbols, with no arithmetic.
    pub fn cancels(self, rhs: Scale) -> bool {
        self.half == rhs.half && rhs.turn == self.turn.wrapping_add(HALF)
    }

    /// Doubling, as a scale: `2 = √2²`.
    pub const TWO: Scale = Scale { turn: 0, half: 2 };
}

/// `√2^half`, exactly. Even exponents are integers; odd ones carry one
/// factor of `√2 = ω − ω³`.
fn sqrt2_pow(half: i64) -> Result<DOmega> {
    if half < 0 {
        let k = u32::try_from(-half).map_err(|_| {
            Error::InvalidState(format!("address: √2 exponent {half} out of range"))
        })?;
        return Ok(DOmega::inv_sqrt2_pow(k));
    }
    let (whole, odd) = (half / 2, half % 2 == 1);
    if whole >= 62 {
        return Err(Error::InvalidState(format!(
            "address: √2^{half} overflows the exact ring's i128 coefficients — the \
             residual's scale is past what D[ω] can hold"
        )));
    }
    let mut v = DOmega::int(1i64 << whole);
    if odd {
        // √2 = ω − ω³.
        let r2 = DOmega::omega_pow(1).sub(DOmega::omega_pow(3))?;
        v = v.mul(r2)?;
    }
    Ok(v)
}

/// A node of the address space, built from other addresses.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    /// The rewrite rules proved this sum vanishes.
    Zero,
    /// `scale · ∏ parts` — the factoring move. With no parts this is a
    /// closed leaf: the rules finished it and its value is the scale.
    Product {
        /// The exact prefactor.
        scale: Scale,
        /// Independent factors, sorted so that commutation merges.
        parts: Vec<Addr>,
    },
    /// A sum of two addresses. It arises from branching a variable,
    /// but addition commutes, so the operands are held sorted and
    /// `a + b` and `b + a` are one node.
    Sum {
        /// The smaller operand.
        left: Addr,
        /// The larger.
        right: Addr,
    },
}

/// What building addresses cost, and how much of it was reuse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AddressStats {
    /// Composition calls — the work.
    pub builds: u64,
    /// Canonical forms reused: the merges, across every query the space
    /// has ever answered.
    pub form_hits: u64,
    /// Structural intern hits: a node built twice from the same parts.
    pub node_hits: u64,
    /// `x · 0 = 0` fired: a product collapsed because one factor was
    /// the zero address, with no evaluation.
    pub zero_products: u64,
    /// `0 + x = x` fired: a branch had a vanishing child, so the node
    /// is the surviving child rather than a new one.
    pub zero_sums: u64,
    /// `a + a = 2a` fired.
    pub doubled: u64,
    /// `s·P + t·P = 0` fired: the two scales cancel, so the sum is
    /// exactly zero whatever `P` is — decided without knowing it.
    pub cancelled: u64,
    /// A common factor was pulled out of a sum:
    /// `s·CP + t·CQ = C·(s·P + t·Q)`.
    pub factored: u64,
    /// A nested product was absorbed into its parent's scale.
    pub flattened: u64,
    /// Deepest build nesting.
    pub max_depth: u32,
    /// Nodes whose value was actually computed.
    pub evaluations: u64,
}

/// How the operands of the space's sums relate — the ceiling on any
/// sum-rewrite. See [`AddressSpace::sum_census`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SumCensus {
    /// Sum nodes classified.
    pub sums: usize,
    /// Both operands are products over the **same** factor set: a
    /// coefficient rewrite could combine them into one node.
    pub same_parts: usize,
    /// The factor sets overlap without being equal: a common factor
    /// can be pulled out.
    pub some_common: usize,
    /// The factor sets are disjoint — out of reach of either rewrite.
    pub disjoint: usize,
    /// An operand was not a product (a nested sum).
    pub other: usize,
}

/// A space of composable addresses, and the DAG they name.
///
/// Persist one across many queries: forms and nodes are content-
/// addressed, so a second amplitude of the same circuit reuses
/// everything the first one built.
#[derive(Clone, Debug, Default)]
pub struct AddressSpace {
    nodes: Vec<Node>,
    intern: HashMap<Node, Addr>,
    forms: HashMap<CanonKey, Addr>,
    values: Vec<Option<C64>>,
    exact: Vec<Option<DOmega>>,
    rewrite: bool,
    factor: bool,
    stats: AddressStats,
}

impl AddressSpace {
    /// An empty space, with rewriting on.
    pub fn new() -> Self {
        AddressSpace {
            rewrite: true,
            factor: false,
            ..AddressSpace::default()
        }
    }

    /// Turn the cheap structural rewrites off (product absorption,
    /// `a + a = 2a`, scale cancellation), leaving only interning and
    /// the zero folds. Exists so the rung can be measured rather than
    /// assumed: the same circuit built both ways must give the same
    /// values.
    pub fn set_rewrites(&mut self, on: bool) {
        self.rewrite = on;
    }

    /// Turn common-factor extraction on. **Off by default**, because it
    /// is the one rewrite here that is a trade rather than a win: it
    /// names the shared sub-DAG once instead of twice, but replaces one
    /// `Sum` node with four, so it costs addresses and buys
    /// evaluations. Measured both ways in `examples/address_algebra.rs`.
    pub fn set_factoring(&mut self, on: bool) {
        self.factor = on;
    }

    /// Distinct addresses in the space.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the space holds no addresses.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// What building has cost so far.
    pub fn stats(&self) -> AddressStats {
        self.stats
    }

    /// Distinct canonical residual forms the space has addressed.
    pub fn forms(&self) -> usize {
        self.forms.len()
    }

    /// What an address is built from.
    pub fn node(&self, a: Addr) -> &Node {
        &self.nodes[a.index()]
    }

    /// The node at an index, for walking the whole space.
    pub fn node_at(&self, i: usize) -> &Node {
        &self.nodes[i]
    }

    /// Classify every [`Node::Sum`] by how its two operands' factor
    /// sets relate — a **bound on what any sum-rewrite can reach**,
    /// measured rather than assumed.
    ///
    /// A rewrite that pulls out a common factor can only touch the
    /// `some_common` and `same_parts` rows; one that combines
    /// coefficients over a shared factor set can only touch
    /// `same_parts`. Whatever is `disjoint` is out of reach of both.
    pub fn sum_census(&self) -> SumCensus {
        let mut c = SumCensus::default();
        for n in &self.nodes {
            let Node::Sum { left, right } = n else {
                continue;
            };
            c.sums += 1;
            match (&self.nodes[left.index()], &self.nodes[right.index()]) {
                (Node::Product { parts: a, .. }, Node::Product { parts: b, .. }) => {
                    let (common, ra, rb) = AddressSpace::split_common(a, b);
                    if common.is_empty() {
                        c.disjoint += 1;
                    } else if ra.is_empty() && rb.is_empty() {
                        c.same_parts += 1;
                    } else {
                        c.some_common += 1;
                    }
                }
                _ => c.other += 1,
            }
        }
        c
    }

    /// Whether two addresses denote the same amplitude — decided by
    /// identity, with no arithmetic performed on either side.
    pub fn same(a: Addr, b: Addr) -> bool {
        a == b
    }

    // ─────────────────── the algebra's constructors ───────────────────

    fn intern(&mut self, node: Node) -> Addr {
        if let Some(&a) = self.intern.get(&node) {
            self.stats.node_hits += 1;
            return a;
        }
        let a = Addr(self.nodes.len() as u32);
        self.nodes.push(node.clone());
        self.values.push(None);
        self.exact.push(None);
        self.intern.insert(node, a);
        a
    }

    /// The zero amplitude.
    pub fn zero(&mut self) -> Addr {
        self.intern(Node::Zero)
    }

    /// A closed leaf: a residual the rewrite rules finished, whose sum
    /// is exactly `scale`.
    pub fn scalar(&mut self, scale: Scale) -> Addr {
        self.intern(Node::Product {
            scale,
            parts: Vec::new(),
        })
    }

    /// `scale · ∏ parts`, normalized.
    ///
    /// Parts are sorted, since the product commutes; a zero factor
    /// collapses the whole product; and a part that is itself a
    /// `Product` is **absorbed** — its scale multiplied into this one
    /// and its parts spliced in — because scales form a group under
    /// [`Scale::times`]. All three are rewrites on addresses, done
    /// before anything is evaluated.
    pub fn product(&mut self, scale: Scale, parts: Vec<Addr>) -> Addr {
        if parts.iter().any(|&p| self.nodes[p.index()] == Node::Zero) {
            self.stats.zero_products += 1;
            return self.zero();
        }
        let (mut scale, mut flat, mut absorbed) = (scale, Vec::with_capacity(parts.len()), false);
        for p in parts {
            match &self.nodes[p.index()] {
                Node::Product {
                    scale: inner,
                    parts: qs,
                } if self.rewrite => {
                    absorbed = true;
                    scale = scale.times(*inner);
                    flat.extend(qs.iter().copied());
                }
                _ => flat.push(p),
            }
        }
        if absorbed {
            self.stats.flattened += 1;
        }
        flat.sort_unstable();
        if flat.is_empty() {
            return self.scalar(scale);
        }
        self.intern(Node::Product { scale, parts: flat })
    }

    /// The multiset intersection of two sorted part lists, and what is
    /// left of each — the common factor of a sum.
    fn split_common(a: &[Addr], b: &[Addr]) -> (Vec<Addr>, Vec<Addr>, Vec<Addr>) {
        let (mut common, mut ra, mut rb) = (Vec::new(), Vec::new(), Vec::new());
        let (mut i, mut j) = (0usize, 0usize);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Equal => {
                    common.push(a[i]);
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => {
                    ra.push(a[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    rb.push(b[j]);
                    j += 1;
                }
            }
        }
        ra.extend_from_slice(&a[i..]);
        rb.extend_from_slice(&b[j..]);
        (common, ra, rb)
    }

    /// `a + b`, normalized.
    ///
    /// Four rewrites, each decided on the addresses:
    ///
    /// * `0 + x = x`;
    /// * `a + a = 2a`, which is a `Product` with scale `√2²`;
    /// * `s·P + t·P = 0` when the scales [cancel](Scale::cancels) — an
    ///   exact cancellation recognized **without knowing what `P` is**,
    ///   which is destructive interference decided structurally;
    /// * `s·CP + t·CQ = C·(s·P + t·Q)` — the common factor pulled out,
    ///   so a shared sub-DAG is named once instead of twice.
    pub fn sum(&mut self, a: Addr, b: Addr) -> Addr {
        let za = self.nodes[a.index()] == Node::Zero;
        let zb = self.nodes[b.index()] == Node::Zero;
        match (za, zb) {
            (true, true) => return self.zero(),
            (true, false) => {
                self.stats.zero_sums += 1;
                return b;
            }
            (false, true) => {
                self.stats.zero_sums += 1;
                return a;
            }
            (false, false) => {}
        }
        if self.rewrite {
            if a == b {
                self.stats.doubled += 1;
                return self.product(Scale::TWO, vec![a]);
            }
            let pair = match (&self.nodes[a.index()], &self.nodes[b.index()]) {
                (
                    Node::Product {
                        scale: s,
                        parts: pa,
                    },
                    Node::Product {
                        scale: t,
                        parts: pb,
                    },
                ) => Some((*s, *t, AddressSpace::split_common(pa, pb))),
                _ => None,
            };
            if let Some((s, t, (common, ra, rb))) = pair {
                if ra.is_empty() && rb.is_empty() && s.cancels(t) {
                    self.stats.cancelled += 1;
                    return self.zero();
                }
                if !common.is_empty() && self.factor {
                    self.stats.factored += 1;
                    let la = self.product(s, ra);
                    let lb = self.product(t, rb);
                    let inner = self.sum(la, lb);
                    let mut parts = common;
                    parts.push(inner);
                    return self.product(Scale::ONE, parts);
                }
            }
        }
        let (left, right) = if a <= b { (a, b) } else { (b, a) };
        self.intern(Node::Sum { left, right })
    }

    // ─────────────────────────── building ───────────────────────────

    /// Address `⟨bits|ψ⟩` without evaluating it.
    ///
    /// `budget` caps composition calls; exceeding it is a named refusal
    /// rather than a silent fallback.
    pub fn address(&mut self, ps: &PathSum, bits: u64, budget: u64) -> Result<Addr> {
        self.address_by(ps, &|q| bits >> q & 1 == 1, budget, Pivot::default())
    }

    /// [`address`](Self::address) under a named branching policy.
    pub fn address_by(
        &mut self,
        ps: &PathSum,
        want_bit: &dyn Fn(usize) -> bool,
        budget: u64,
        pivot: Pivot,
    ) -> Result<Addr> {
        match ps.pinned(want_bit) {
            None => Ok(self.zero()),
            Some(pinned) => self.build(&pinned, 0, budget, pivot),
        }
    }

    fn build(&mut self, ps: &PathSum, depth: u32, budget: u64, pivot: Pivot) -> Result<Addr> {
        self.stats.builds += 1;
        self.stats.max_depth = self.stats.max_depth.max(depth);
        if self.stats.builds > budget {
            return Err(Error::InvalidState(format!(
                "address: composition budget {budget} exhausted at depth {depth} with \
                 {} addresses and {} forms — the residual did not merge, and this \
                 refuses rather than quietly enumerating it",
                self.nodes.len(),
                self.forms.len()
            )));
        }
        if ps.is_zero_residual() {
            return Ok(self.zero());
        }
        let (turn, half, comps) = ps.factor_exact();
        let mut parts = Vec::with_capacity(comps.len());
        for c in &comps {
            parts.push(self.component(c, depth, budget, pivot)?);
        }
        Ok(self.product(Scale { turn, half }, parts))
    }

    fn component(
        &mut self,
        comp: &Component,
        depth: u32,
        budget: u64,
        pivot: Pivot,
    ) -> Result<Addr> {
        let key = canonicalize(comp);
        if let Some(&a) = self.forms.get(&key) {
            self.stats.form_hits += 1;
            return Ok(a);
        }
        let piv = pivot.choose(comp);
        let [c0, c1] = PathSum::branch_children(comp, piv);
        let a0 = self.build(&c0, depth + 1, budget, pivot)?;
        let a1 = self.build(&c1, depth + 1, budget, pivot)?;
        let a = self.sum(a0, a1);
        self.forms.insert(key, a);
        Ok(a)
    }

    // ────────────────────────── evaluation ──────────────────────────

    /// Post-order over the DAG, visiting each address at most once.
    fn order(&self, root: Addr) -> Vec<Addr> {
        let mut out = Vec::new();
        let mut seen = vec![false; self.nodes.len()];
        let mut stack = vec![(root, false)];
        while let Some((a, expanded)) = stack.pop() {
            if expanded {
                out.push(a);
                continue;
            }
            if seen[a.index()] {
                continue;
            }
            seen[a.index()] = true;
            stack.push((a, true));
            match &self.nodes[a.index()] {
                Node::Zero => {}
                Node::Product { parts, .. } => {
                    for &p in parts {
                        stack.push((p, false));
                    }
                }
                Node::Sum { left, right } => {
                    stack.push((*left, false));
                    stack.push((*right, false));
                }
            }
        }
        out
    }

    /// The amplitude an address denotes, memoized across every address
    /// in the space.
    pub fn value(&mut self, root: Addr) -> Result<C64> {
        for a in self.order(root) {
            if self.values[a.index()].is_some() {
                continue;
            }
            let v = match self.nodes[a.index()].clone() {
                Node::Zero => C64::new(0.0, 0.0),
                Node::Product { scale, parts } => {
                    let mut acc = scale.to_c64();
                    for p in parts {
                        acc *= self.values[p.index()].expect("post-order visits parts first");
                    }
                    acc
                }
                Node::Sum { left, right } => {
                    self.values[left.index()].expect("post-order")
                        + self.values[right.index()].expect("post-order")
                }
            };
            self.values[a.index()] = Some(v);
            self.stats.evaluations += 1;
        }
        Ok(self.values[root.index()].expect("root was visited"))
    }

    /// The amplitude in the Clifford+T ring — **no floating point at
    /// all**, so equality and exact zero are decidable.
    ///
    /// Refuses by name if any scale in the DAG is not an eighth turn.
    pub fn value_exact(&mut self, root: Addr) -> Result<DOmega> {
        for a in self.order(root) {
            if self.exact[a.index()].is_some() {
                continue;
            }
            let v = match self.nodes[a.index()].clone() {
                Node::Zero => DOmega::zero(),
                Node::Product { scale, parts } => {
                    let mut acc = scale.to_exact()?;
                    for p in parts {
                        let f = self.exact[p.index()].expect("post-order visits parts first");
                        acc = acc.mul(f)?;
                    }
                    acc
                }
                Node::Sum { left, right } => {
                    let a0 = self.exact[left.index()].expect("post-order");
                    let a1 = self.exact[right.index()].expect("post-order");
                    a0.add(a1)?
                }
            };
            self.exact[a.index()] = Some(v);
            self.stats.evaluations += 1;
        }
        Ok(self.exact[root.index()].expect("root was visited"))
    }

    /// Whether an address denotes exactly zero, decided in the ring.
    ///
    /// A `Zero` node answers structurally, with no arithmetic; anything
    /// else is evaluated exactly, so this is a decision and not a
    /// tolerance test.
    pub fn is_zero(&mut self, root: Addr) -> Result<bool> {
        if self.nodes[root.index()] == Node::Zero {
            return Ok(true);
        }
        Ok(self.value_exact(root)?.is_zero())
    }
}
