//! Pauli supports as a shared, memoized graph — no width ceiling, and
//! no per-term copy.
//!
//! The crate's Pauli strings are `(u64, u64)`, which caps a register at
//! [`MAX_QUBITS`](crate::heisenberg::MAX_QUBITS). For a representation
//! whose cost is `2^n` that ceiling is free — 2^64 amplitudes is
//! unreachable long before 64 wires are — but the Heisenberg walk, the
//! coupling partition, the dyadic cone and the up-embedded readout are
//! all *sub-exponential*, and for those the ceiling is a limit of the
//! datatype rather than of the physics.
//!
//! ## Why not just a big flat bitset
//!
//! Because it would be strictly worse than `u64` below 64 qubits and
//! only barely workable above. A [`PauliSum`](crate::heisenberg::PauliSum)
//! is a hash map keyed by support, holding up to millions of terms, and
//! a flat `Vec<u64>` key means a heap allocation per term, an `O(n/64)`
//! hash on every lookup, an `O(n/64)` comparison on every collision —
//! and it discards the one fact that matters about those terms: **they
//! are nearly identical to each other.** Propagation grows a support one
//! rotation at a time, so a sum's terms share almost all of their
//! structure.
//!
//! ## What this is instead
//!
//! A hash-consed binary trie over qubit indices, with 64-index chunks at
//! the leaves. Two supports that agree on a range *share the node for
//! that range*, so:
//!
//! * equality is a pointer comparison in the common case, and a hash
//!   comparison otherwise — never a scan,
//! * hashing is `O(1)`: every node caches its own,
//! * [`Support::weight`] is `O(1)`: every node caches its popcount,
//! * `xor`/`and`/`or` are memoized recursions that return immediately on
//!   identical subtrees, so combining two near-identical supports costs
//!   their *difference* times the depth of the trie — logarithmic in the
//!   register, where a flat bitset is linear in it,
//! * a support of weight `w` over `n` qubits costs `O(w log n)` nodes
//!   before sharing, and sharing is what pays for the log.
//!
//! None of the operations knows how wide the register is, so there is no
//! `MAX_QUBITS` here — the index type is `usize` and that is the only
//! bound.
//!
//! The [`mod@bench`] helpers measure this against
//! both alternatives rather than asserting it; `examples/wide_support.rs`
//! runs them.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::scalar::C64;

/// Qubit indices covered by one leaf.
const CHUNK: usize = 64;

/// Memo entries kept per operation before the table is cleared.
///
/// The tables hold strong references to their operands, so a cap is what
/// keeps a long-running walk from retaining every intermediate support
/// it ever built. Clearing costs nothing but re-derivation.
pub const MEMO_CAPACITY: usize = 1 << 18;

// ── the node ─────────────────────────────────────────────────────────

#[derive(Debug)]
enum Kind {
    /// 64 indices, `base .. base + 64`.
    Leaf(u64),
    /// Two halves of a `64·2^height`-index range.
    Branch(Support, Support),
}

#[derive(Debug)]
struct Node {
    /// 0 for a leaf; a branch at height `h` spans `64·2^h` indices.
    height: u8,
    /// Cached popcount — [`Support::weight`] never walks.
    weight: u32,
    /// Cached hash — hashing a support never walks.
    hash: u64,
    kind: Kind,
}

/// A set of qubit indices, unbounded, shared between the supports that
/// contain it.
///
/// Cheap to clone (one `Arc` bump), cheap to hash and compare (cached),
/// and cheap to combine with a support it resembles (memoized).
#[derive(Clone, Debug, Default)]
pub struct Support(Option<Arc<Node>>);

fn mix(a: u64, b: u64) -> u64 {
    // A cheap, well-mixing combiner; the values being combined are
    // already hashes, so this only needs to avoid collapsing them.
    let mut h = a ^ (b.wrapping_add(0x9e37_79b9_7f4a_7c15));
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

impl Support {
    /// The empty support.
    pub fn empty() -> Support {
        Support(None)
    }

    /// Whether the support holds nothing.
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
    }

    /// Indices in the support — cached, never a scan.
    pub fn weight(&self) -> usize {
        self.0.as_ref().map_or(0, |n| n.weight as usize)
    }

    fn height(&self) -> u8 {
        self.0.as_ref().map_or(0, |n| n.height)
    }

    fn cached_hash(&self) -> u64 {
        self.0.as_ref().map_or(0, |n| n.hash)
    }

    fn leaf(bits: u64) -> Support {
        if bits == 0 {
            return Support::empty();
        }
        Support(Some(Arc::new(Node {
            height: 0,
            weight: bits.count_ones(),
            hash: mix(bits, 0x5bf0_3635),
            kind: Kind::Leaf(bits),
        })))
    }

    /// A branch at `height`, canonicalized: an empty high half collapses
    /// to the low half, so a support's height is a function of its
    /// largest index and equal sets have equal shapes.
    fn branch(lo: Support, hi: Support, height: u8) -> Support {
        if hi.is_empty() {
            return lo;
        }
        let weight = (lo.weight() + hi.weight()) as u32;
        let hash = mix(lo.cached_hash(), mix(hi.cached_hash(), height as u64));
        Support(Some(Arc::new(Node {
            height,
            weight,
            hash,
            kind: Kind::Branch(lo, hi),
        })))
    }

    /// The two halves of this support viewed at `height`, each at
    /// height `height - 1`.
    ///
    /// A support shorter than `height` lies wholly in the low half —
    /// that is what makes canonical collapsing sound.
    fn split(&self, height: u8) -> (Support, Support) {
        match self.0.as_ref() {
            None => (Support::empty(), Support::empty()),
            Some(n) if n.height < height => (self.clone(), Support::empty()),
            Some(n) => match &n.kind {
                Kind::Branch(lo, hi) => (lo.clone(), hi.clone()),
                Kind::Leaf(_) => (self.clone(), Support::empty()),
            },
        }
    }

    /// The single index `i`.
    pub fn single(i: usize) -> Support {
        let mut s = Support::leaf(1u64 << (i % CHUNK));
        let mut chunk = i / CHUNK;
        let mut height = 1u8;
        while chunk > 0 {
            s = if chunk & 1 == 1 {
                Support::branch(Support::empty(), s, height)
            } else {
                Support::branch(s, Support::empty(), height)
            };
            chunk >>= 1;
            height += 1;
        }
        s
    }

    /// A support from a `u64` bitmask over indices `0..64` — the bridge
    /// to the crate's `PauliKey`.
    pub fn from_u64(bits: u64) -> Support {
        Support::leaf(bits)
    }

    /// The low 64 indices as a `u64`, for the same bridge. Indices at or
    /// above 64 are not representable and are reported separately by
    /// [`Support::fits_u64`].
    pub fn to_u64(&self) -> u64 {
        match self.0.as_ref() {
            None => 0,
            Some(n) => match &n.kind {
                Kind::Leaf(b) => *b,
                Kind::Branch(lo, _) => lo.to_u64(),
            },
        }
    }

    /// Whether every index fits below 64, so [`Support::to_u64`] is
    /// lossless.
    pub fn fits_u64(&self) -> bool {
        self.height() == 0
    }

    /// Whether index `i` is present.
    pub fn contains(&self, i: usize) -> bool {
        let mut cur = self.clone();
        loop {
            match cur.0.as_ref() {
                None => return false,
                Some(n) => match &n.kind {
                    Kind::Leaf(b) => {
                        return i < CHUNK && b >> i & 1 == 1;
                    }
                    Kind::Branch(lo, hi) => {
                        let half = CHUNK << (n.height - 1);
                        if i >= half {
                            cur = hi.clone();
                            // `i` is measured from the branch's base
                            return cur.contains(i - half);
                        }
                        cur = lo.clone();
                    }
                },
            }
        }
    }

    /// This support with `i` added.
    pub fn with(&self, i: usize) -> Support {
        self.or(&Support::single(i))
    }

    /// This support with `i` removed.
    pub fn without_index(&self, i: usize) -> Support {
        self.without(&Support::single(i))
    }

    /// Symmetric difference.
    pub fn xor(&self, other: &Support) -> Support {
        zip(self, other, Op::Xor)
    }

    /// Intersection.
    pub fn and(&self, other: &Support) -> Support {
        zip(self, other, Op::And)
    }

    /// Union.
    pub fn or(&self, other: &Support) -> Support {
        zip(self, other, Op::Or)
    }

    /// Set difference.
    pub fn without(&self, other: &Support) -> Support {
        zip(self, other, Op::Without)
    }

    /// Whether the two share an index — short-circuits, so it never
    /// builds the intersection.
    pub fn intersects(&self, other: &Support) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        if ptr_eq(self, other) {
            return true;
        }
        let h = self.height().max(other.height());
        if h == 0 {
            return self.to_u64() & other.to_u64() != 0;
        }
        let (al, ah) = self.split(h);
        let (bl, bh) = other.split(h);
        al.intersects(&bl) || ah.intersects(&bh)
    }

    /// `|self ∧ other| mod 2` — the anticommutation test, without
    /// building the intersection.
    pub fn and_parity(&self, other: &Support) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        if ptr_eq(self, other) {
            return self.weight() % 2 == 1;
        }
        let h = self.height().max(other.height());
        if h == 0 {
            return (self.to_u64() & other.to_u64()).count_ones() % 2 == 1;
        }
        let (al, ah) = self.split(h);
        let (bl, bh) = other.split(h);
        al.and_parity(&bl) ^ ah.and_parity(&bh)
    }

    /// The indices, ascending.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        let mut out = Vec::with_capacity(self.weight());
        self.collect_into(0, &mut out);
        out.into_iter()
    }

    fn collect_into(&self, base: usize, out: &mut Vec<usize>) {
        match self.0.as_ref() {
            None => {}
            Some(n) => match &n.kind {
                Kind::Leaf(b) => {
                    let mut w = *b;
                    while w != 0 {
                        out.push(base + w.trailing_zeros() as usize);
                        w &= w - 1;
                    }
                }
                Kind::Branch(lo, hi) => {
                    let half = CHUNK << (n.height - 1);
                    lo.collect_into(base, out);
                    hi.collect_into(base + half, out);
                }
            },
        }
    }

    /// The largest index present.
    pub fn max_index(&self) -> Option<usize> {
        match self.0.as_ref() {
            None => None,
            Some(n) => match &n.kind {
                Kind::Leaf(b) => Some(63 - b.leading_zeros() as usize),
                Kind::Branch(_, hi) => {
                    let half = CHUNK << (n.height - 1);
                    hi.max_index().map(|i| i + half)
                }
            },
        }
    }

    /// Distinct nodes reachable from this support — what it costs
    /// *after* sharing within itself.
    pub fn nodes(&self) -> usize {
        let mut seen = Vec::new();
        self.walk(&mut seen);
        seen.len()
    }

    fn walk(&self, seen: &mut Vec<*const Node>) {
        let Some(n) = self.0.as_ref() else { return };
        let p = Arc::as_ptr(n);
        if seen.contains(&p) {
            return;
        }
        seen.push(p);
        if let Kind::Branch(lo, hi) = &n.kind {
            lo.walk(seen);
            hi.walk(seen);
        }
    }
}

fn ptr_eq(a: &Support, b: &Support) -> bool {
    match (a.0.as_ref(), b.0.as_ref()) {
        (None, None) => true,
        (Some(x), Some(y)) => Arc::ptr_eq(x, y),
        _ => false,
    }
}

// ── memoized combination ─────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Op {
    Xor,
    And,
    Or,
    Without,
}

thread_local! {
    static MEMO: RefCell<HashMap<(Op, u64, u64), Support>> =
        RefCell::new(HashMap::with_capacity(1024));
}

/// Discard the memo tables. Frees whatever they were holding alive;
/// correctness never depends on them.
pub fn clear_memo() {
    MEMO.with(|m| m.borrow_mut().clear());
}

/// Entries currently memoized — reported so the sharing can be measured
/// rather than assumed.
pub fn memo_len() -> usize {
    MEMO.with(|m| m.borrow().len())
}

fn zip(a: &Support, b: &Support, op: Op) -> Support {
    // The identities, which are what make near-identical operands cheap.
    if ptr_eq(a, b) {
        return match op {
            Op::Xor | Op::Without => Support::empty(),
            Op::And | Op::Or => a.clone(),
        };
    }
    if a.is_empty() {
        return match op {
            Op::And | Op::Without => Support::empty(),
            Op::Xor | Op::Or => b.clone(),
        };
    }
    if b.is_empty() {
        return match op {
            Op::And => Support::empty(),
            _ => a.clone(),
        };
    }

    let key = (op, a.cached_hash(), b.cached_hash());
    if let Some(hit) = MEMO.with(|m| m.borrow().get(&key).cloned()) {
        return hit;
    }

    let h = a.height().max(b.height());
    let out = if h == 0 {
        let (x, y) = (a.to_u64(), b.to_u64());
        Support::leaf(match op {
            Op::Xor => x ^ y,
            Op::And => x & y,
            Op::Or => x | y,
            Op::Without => x & !y,
        })
    } else {
        let (al, ah) = a.split(h);
        let (bl, bh) = b.split(h);
        Support::branch(zip(&al, &bl, op), zip(&ah, &bh, op), h)
    };

    MEMO.with(|m| {
        let mut t = m.borrow_mut();
        if t.len() >= MEMO_CAPACITY {
            t.clear();
        }
        t.insert(key, out.clone());
    });
    out
}

// ── the traits that make it a map key ────────────────────────────────

impl PartialEq for Support {
    fn eq(&self, other: &Self) -> bool {
        if ptr_eq(self, other) {
            return true;
        }
        // Distinct nodes with equal hashes are vanishingly rare and
        // still have to be distinguished, so fall through structurally.
        if self.cached_hash() != other.cached_hash()
            || self.weight() != other.weight()
            || self.height() != other.height()
        {
            return false;
        }
        match (self.0.as_ref(), other.0.as_ref()) {
            (None, None) => true,
            (Some(x), Some(y)) => match (&x.kind, &y.kind) {
                (Kind::Leaf(a), Kind::Leaf(b)) => a == b,
                (Kind::Branch(al, ah), Kind::Branch(bl, bh)) => al == bl && ah == bh,
                _ => false,
            },
            _ => false,
        }
    }
}

impl Eq for Support {}

impl Hash for Support {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.cached_hash());
    }
}

impl Ord for Support {
    /// Lexicographic on the indices, so orderings are stable across
    /// runs — a hash order would not be.
    fn cmp(&self, other: &Self) -> Ordering {
        if ptr_eq(self, other) {
            return Ordering::Equal;
        }
        self.iter().cmp(other.iter())
    }
}

impl PartialOrd for Support {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl FromIterator<usize> for Support {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Support {
        let mut s = Support::empty();
        for i in iter {
            s = s.or(&Support::single(i));
        }
        s
    }
}

// ── a Pauli with no width ceiling ────────────────────────────────────

/// A Pauli string `i^{|x∧z|} X^x Z^z` over an unbounded register.
///
/// The same algebra as the crate's `(u64, u64)`
/// [`PauliKey`](crate::heisenberg::PauliKey), with the
/// same sign conventions and none of the width limit.
/// `the_wide_pauli_agrees_with_the_u64_one` pins the two against each
/// other everywhere both can run.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WidePauli {
    /// X-support.
    pub x: Support,
    /// Z-support.
    pub z: Support,
}

impl WidePauli {
    /// The identity.
    pub fn identity() -> WidePauli {
        WidePauli::default()
    }

    /// From the crate's bounded key.
    pub fn from_key(key: (u64, u64)) -> WidePauli {
        WidePauli {
            x: Support::from_u64(key.0),
            z: Support::from_u64(key.1),
        }
    }

    /// From a pair of [`Mask`](crate::pathsum::Mask) supports.
    ///
    /// The link between the two unbounded Pauli representations in this
    /// crate. `Mask` is what the path sum and
    /// [`crate::upembed`] carry, and what
    /// [`crate::dcs::rotation_axes`] returns; `Support` is what the wide
    /// walk needs. Without this the only register-unbounded *producer*
    /// of Pauli axes and the only register-unbounded *consumer* of them
    /// could not be composed, which is a gap in the plumbing rather than
    /// in the mathematics.
    pub fn from_masks(x: &crate::pathsum::Mask, z: &crate::pathsum::Mask) -> WidePauli {
        let build = |m: &crate::pathsum::Mask| {
            let mut s = Support::empty();
            for i in m.iter() {
                s = s.with(i);
            }
            s
        };
        WidePauli {
            x: build(x),
            z: build(z),
        }
    }

    /// To the crate's bounded key, if it fits.
    pub fn to_key(&self) -> Option<(u64, u64)> {
        (self.x.fits_u64() && self.z.fits_u64()).then(|| (self.x.to_u64(), self.z.to_u64()))
    }

    /// Non-identity tensor factors.
    pub fn weight(&self) -> usize {
        self.x.or(&self.z).weight()
    }

    /// The qubits acted on, ascending.
    pub fn support(&self) -> Vec<usize> {
        self.x.or(&self.z).iter().collect()
    }

    /// Whether the two commute — `|p_z ∧ q_x| + |p_x ∧ q_z|` even.
    pub fn commutes(&self, other: &WidePauli) -> bool {
        !(self.z.and_parity(&other.x) ^ self.x.and_parity(&other.z))
    }

    /// `(X^a Z^b)(X^c Z^d) = sign · X^{a⊕c} Z^{b⊕d}`, with the same
    /// reordering sign the bounded version uses.
    pub fn mul(&self, other: &WidePauli) -> (WidePauli, f64) {
        let sign = if self.z.and_parity(&other.x) {
            -1.0
        } else {
            1.0
        };
        (
            WidePauli {
                x: self.x.xor(&other.x),
                z: self.z.xor(&other.z),
            },
            sign,
        )
    }
}

// ── measurement helpers ──────────────────────────────────────────────

/// Helpers the runnable uses to price this against the alternatives.
///
/// These exist so the claim "shared beats flat on this workload" is a
/// measurement in the repository rather than an assertion in a doc
/// comment.
pub mod bench {
    use super::*;

    /// A flat unbounded bitset — the obvious alternative to `u64`, and
    /// the thing the shared graph has to beat to be worth its
    /// complexity.
    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct Flat(pub Vec<u64>);

    impl Flat {
        /// The single index `i`.
        pub fn single(i: usize) -> Flat {
            let mut v = vec![0; i / 64 + 1];
            v[i / 64] = 1u64 << (i % 64);
            Flat(v)
        }

        /// Symmetric difference.
        pub fn xor(&self, other: &Flat) -> Flat {
            let n = self.0.len().max(other.0.len());
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(self.0.get(i).copied().unwrap_or(0) ^ other.0.get(i).copied().unwrap_or(0));
            }
            while v.last() == Some(&0) {
                v.pop();
            }
            Flat(v)
        }

        /// Union.
        pub fn or(&self, other: &Flat) -> Flat {
            let n = self.0.len().max(other.0.len());
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(self.0.get(i).copied().unwrap_or(0) | other.0.get(i).copied().unwrap_or(0));
            }
            Flat(v)
        }

        /// Indices present — an unavoidable scan.
        pub fn weight(&self) -> usize {
            self.0.iter().map(|w| w.count_ones() as usize).sum()
        }

        /// Words held, which is what it costs per stored term.
        pub fn words(&self) -> usize {
            self.0.len()
        }
    }

    /// Total distinct nodes across a collection — the shared cost, as
    /// against the sum of the individual costs.
    pub fn shared_nodes(supports: &[Support]) -> usize {
        let mut seen: Vec<*const Node> = Vec::new();
        for s in supports {
            s.walk(&mut seen);
        }
        seen.len()
    }

    /// Nodes each support would need on its own — the unshared cost.
    pub fn unshared_nodes(supports: &[Support]) -> usize {
        supports.iter().map(|s| s.nodes()).sum()
    }
}

// ── a Heisenberg walk with no width ceiling ──────────────────────────

/// A rotation `exp(−iθP/2)` about an unbounded Pauli axis.
#[derive(Clone, Debug)]
pub struct WideRotation {
    /// Rotation angle.
    pub theta: f64,
    /// The axis.
    pub axis: WidePauli,
}

impl WideRotation {
    /// A rotation about `Z_q`.
    pub fn rz(qubit: usize, theta: f64) -> WideRotation {
        WideRotation {
            theta,
            axis: WidePauli {
                x: Support::empty(),
                z: Support::single(qubit),
            },
        }
    }

    /// A rotation about `X_q`.
    pub fn rx(qubit: usize, theta: f64) -> WideRotation {
        WideRotation {
            theta,
            axis: WidePauli {
                x: Support::single(qubit),
                z: Support::empty(),
            },
        }
    }

    /// A rotation about `Z_a Z_b`.
    pub fn rzz(a: usize, b: usize, theta: f64) -> WideRotation {
        WideRotation {
            theta,
            axis: WidePauli {
                x: Support::empty(),
                z: Support::single(a).or(&Support::single(b)),
            },
        }
    }
}

/// A sum of unbounded Pauli strings with complex coefficients.
///
/// The same object as [`PauliSum`](crate::heisenberg::PauliSum), keyed
/// by [`WidePauli`] instead of `(u64, u64)` — so the register has no
/// maximum, and the keys share their structure with each other rather
/// than each holding a private copy.
#[derive(Clone, Debug, Default)]
pub struct WidePauliSum {
    terms: HashMap<WidePauli, C64>,
}

impl WidePauliSum {
    /// The empty sum.
    pub fn zero() -> WidePauliSum {
        WidePauliSum::default()
    }

    /// A single Pauli with unit coefficient, in the **raw** `X^x Z^z`
    /// convention the keys use.
    pub fn from_pauli(p: WidePauli) -> WidePauliSum {
        let mut s = WidePauliSum::zero();
        s.add(p, C64::new(1.0, 0.0));
        s
    }

    /// The **Hermitian** observable `i^{|x∧z|} X^x Z^z`, seeded with the
    /// normalization that makes it Hermitian.
    ///
    /// The distinction is not pedantic and it is easy to lose: keys are
    /// raw, so seeding a `Y` observable with coefficient `1` computes
    /// `⟨X Z⟩ = −i⟨Y⟩`, which is purely imaginary — and any readout that
    /// takes a real part then silently returns zero. That is exactly how
    /// the reflexive loop's Jacobian came back identically zero, and it
    /// looked like physics rather than a bug because `⟨X⟩` and `⟨Z⟩`
    /// observables were unaffected.
    pub fn from_observable(p: WidePauli) -> WidePauliSum {
        let phase = match p.x.and(&p.z).weight() % 4 {
            0 => C64::new(1.0, 0.0),
            1 => C64::new(0.0, 1.0),
            2 => C64::new(-1.0, 0.0),
            _ => C64::new(0.0, -1.0),
        };
        let mut s = WidePauliSum::zero();
        s.add(p, phase);
        s
    }

    /// `Z_q` with unit coefficient — the usual observable.
    pub fn z_at(qubit: usize) -> WidePauliSum {
        WidePauliSum::from_pauli(WidePauli {
            x: Support::empty(),
            z: Support::single(qubit),
        })
    }

    /// Accumulate a coefficient, dropping the term when it cancels.
    pub fn add(&mut self, p: WidePauli, coeff: C64) {
        if coeff.norm() < crate::heisenberg::COEFF_FLOOR {
            return;
        }
        match self.terms.entry(p) {
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let v = *o.get() + coeff;
                if v.norm() < crate::heisenberg::COEFF_FLOOR {
                    o.remove();
                } else {
                    *o.get_mut() = v;
                }
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(coeff);
            }
        }
    }

    /// Terms held.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Whether the sum is empty.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// The terms.
    pub fn terms(&self) -> impl Iterator<Item = (&WidePauli, &C64)> {
        self.terms.iter()
    }

    /// `Σ|c|`.
    pub fn l1(&self) -> f64 {
        self.terms.values().map(|c| c.norm()).sum()
    }

    /// The heaviest term's weight.
    pub fn max_weight(&self) -> usize {
        self.terms.keys().map(|p| p.weight()).max().unwrap_or(0)
    }

    /// The coefficient of a term.
    pub fn get(&self, p: &WidePauli) -> C64 {
        self.terms.get(p).copied().unwrap_or(C64::new(0.0, 0.0))
    }

    /// `⟨0…0|Σ|0…0⟩`: only `X`-free terms survive, and each contributes
    /// its coefficient — an `X`-free raw key is `Z^z`, whose vacuum
    /// expectation is `1`.
    ///
    /// Seed with [`WidePauliSum::from_observable`] if the sum is meant
    /// to be a Hermitian observable, or this returns the real part of
    /// something that was never real.
    pub fn vacuum_expectation(&self) -> f64 {
        self.vacuum_expectation_complex().re
    }

    /// The same, without discarding the imaginary part — which should be
    /// zero for a properly seeded Hermitian observable, and is worth
    /// checking rather than assuming.
    pub fn vacuum_expectation_complex(&self) -> C64 {
        self.terms
            .iter()
            .filter(|(p, _)| p.x.is_empty())
            .fold(C64::new(0.0, 0.0), |a, (_, c)| a + *c)
    }
}

/// Pruning policy for [`propagate_wide`].
#[derive(Clone, Copy, Debug, Default)]
pub struct WideConfig {
    /// Terms with `|c|` below this are dropped and accounted. `0.0` is
    /// exact.
    pub threshold: f64,
    /// Hard cap on the term count. `None` is uncapped.
    pub max_terms: Option<usize>,
}

/// What a wide walk produced and what it cost.
#[derive(Clone, Debug)]
pub struct WidePropagation {
    /// The propagated operator.
    pub sum: WidePauliSum,
    /// Largest working set reached.
    pub peak_terms: usize,
    /// `Σ|c|` discarded — the rigorous error bound.
    pub discarded_l1: f64,
    /// Heaviest term seen.
    pub max_weight: usize,
    /// Distinct support nodes across the final sum, against the nodes
    /// the same terms would need unshared.
    pub shared_nodes: usize,
    /// Nodes the final sum would need with no sharing.
    pub unshared_nodes: usize,
}

/// One Heisenberg step: `R† P R = cos θ · P − i sin θ · P Q`.
fn step_wide(sum: &WidePauliSum, rot: &WideRotation) -> WidePauliSum {
    let (c, s) = (rot.theta.cos(), rot.theta.sin());
    // i^{|x∧z|} for the axis, matching the crate's Hermitian convention.
    let phase = match rot.axis.x.and(&rot.axis.z).weight() % 4 {
        0 => C64::new(1.0, 0.0),
        1 => C64::new(0.0, 1.0),
        2 => C64::new(-1.0, 0.0),
        _ => C64::new(0.0, -1.0),
    };
    let mut out = WidePauliSum::zero();
    for (p, coeff) in sum.terms() {
        if p.commutes(&rot.axis) {
            out.add(p.clone(), *coeff);
            continue;
        }
        out.add(p.clone(), *coeff * C64::new(c, 0.0));
        let (prod, sign) = p.mul(&rot.axis);
        out.add(prod, *coeff * (C64::new(0.0, -s * sign) * phase));
    }
    out
}

/// Propagate an observable backwards through a rotation list, with no
/// bound on the register's width.
///
/// The same walk as [`heisenberg::propagate`](crate::heisenberg::propagate)
/// and the same numbers — `the_wide_walk_agrees_with_the_bounded_one`
/// pins them together wherever both can run — but the working set's keys
/// are shared, so a sum whose terms resemble each other costs the size
/// of their differences.
pub fn propagate_wide(
    observable: &WidePauliSum,
    rotations: &[WideRotation],
    cfg: &WideConfig,
) -> WidePropagation {
    let mut sum = observable.clone();
    let mut peak = sum.len();
    let mut discarded = 0.0f64;
    let mut max_weight = sum.max_weight();

    for rot in rotations.iter().rev() {
        sum = step_wide(&sum, rot);
        if cfg.threshold > 0.0 {
            let mut keep = WidePauliSum::zero();
            for (p, c) in sum.terms() {
                if c.norm() >= cfg.threshold {
                    keep.add(p.clone(), *c);
                } else {
                    discarded += c.norm();
                }
            }
            sum = keep;
        }
        if let Some(cap) = cfg.max_terms {
            if sum.len() > cap {
                let mut ranked: Vec<(WidePauli, C64)> =
                    sum.terms().map(|(p, c)| (p.clone(), *c)).collect();
                ranked.sort_by(|a, b| {
                    b.1.norm()
                        .partial_cmp(&a.1.norm())
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                let mut keep = WidePauliSum::zero();
                for (i, (p, c)) in ranked.into_iter().enumerate() {
                    if i < cap {
                        keep.add(p, c);
                    } else {
                        discarded += c.norm();
                    }
                }
                sum = keep;
            }
        }
        peak = peak.max(sum.len());
        max_weight = max_weight.max(sum.max_weight());
    }

    let supports: Vec<Support> = sum.terms().map(|(p, _)| p.x.or(&p.z)).collect();
    WidePropagation {
        shared_nodes: bench::shared_nodes(&supports),
        unshared_nodes: bench::unshared_nodes(&supports),
        sum,
        peak_terms: peak,
        discarded_l1: discarded,
        max_weight,
    }
}
