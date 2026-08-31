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
//!   second amplitude of the same circuit reuses every form the first
//!   one built. That is Gosper's hashlife move — time-step at the level
//!   of canonical node ids rather than of cells — carried onto the
//!   phase polynomial.
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
//! The zero identities (`x · 0 = 0`, `0 + x = x`) are implemented and
//! correct, and **have never fired**: over 4000 random Clifford+T
//! circuits at 3–4 qubits, [`AddressStats::folds`] stayed at zero,
//! because the reduction consumes the lone-half-turn pattern that makes
//! a residual vanish before any branch can produce it as a child. They
//! are kept because `0 + x = x` is what stops a dead branch from
//! splitting one node into two, and reported here rather than
//! advertised.
//!
//! This changes what the solver *is*, not what it costs asymptotically:
//! the growth law is still a property of the circuit's coupling graph.
//! What it changes is that the cost is now amortized across queries and
//! that structural questions are answerable without arithmetic.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::exact::DOmega;
use crate::pathsum::{canonicalize, CanonKey, Component, PathSum, Pivot, Turn, EIGHTH};
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
    /// The sum of the two children of a branched variable.
    Branch {
        /// The child with the pivot set to `false`.
        zero: Addr,
        /// The child with the pivot set to `true`.
        one: Addr,
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
    /// Zero identities that fired on addresses without any evaluation
    /// (`x · 0`, `0 + x`). Measured at **zero** on every circuit tried;
    /// see the module docs.
    pub folds: u64,
    /// Deepest build nesting.
    pub max_depth: u32,
    /// Nodes whose value was actually computed.
    pub evaluations: u64,
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
    stats: AddressStats,
}

impl AddressSpace {
    /// An empty space.
    pub fn new() -> Self {
        AddressSpace::default()
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

    /// `scale · ∏ parts`. Parts are sorted (the product commutes) and a
    /// zero factor collapses the whole product — rewrites performed on
    /// addresses, before anything is evaluated.
    pub fn product(&mut self, scale: Scale, mut parts: Vec<Addr>) -> Addr {
        if parts.iter().any(|&p| self.nodes[p.index()] == Node::Zero) {
            self.stats.folds += 1;
            return self.zero();
        }
        parts.sort_unstable();
        if parts.is_empty() {
            return self.scalar(scale);
        }
        self.intern(Node::Product { scale, parts })
    }

    /// `zero + one`, with `0 + x = x` folded on the addresses.
    pub fn sum(&mut self, zero: Addr, one: Addr) -> Addr {
        let z0 = self.nodes[zero.index()] == Node::Zero;
        let z1 = self.nodes[one.index()] == Node::Zero;
        match (z0, z1) {
            (true, true) => self.zero(),
            (true, false) => {
                self.stats.folds += 1;
                one
            }
            (false, true) => {
                self.stats.folds += 1;
                zero
            }
            (false, false) => self.intern(Node::Branch { zero, one }),
        }
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
                Node::Branch { zero, one } => {
                    stack.push((*zero, false));
                    stack.push((*one, false));
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
                Node::Branch { zero, one } => {
                    self.values[zero.index()].expect("post-order")
                        + self.values[one.index()].expect("post-order")
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
                Node::Branch { zero, one } => {
                    let a0 = self.exact[zero.index()].expect("post-order");
                    let a1 = self.exact[one.index()].expect("post-order");
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
