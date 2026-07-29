//! Polarity as a **fibered, re-orderable, journalled co-bundle**:
//! entanglement held as an explicit, inspectable, coarse-grainable
//! resource rather than as an implicit consequence of amplitude
//! storage.
//!
//! ## Why this exists, and what the previous module got wrong
//!
//! [`crate::polarity`] measured that the two-body *marginals* of a
//! state cannot distinguish `(|0…0⟩ + |1…1⟩)/√2` from
//! `(|0…0⟩ − |1…1⟩)/√2`. That measurement is correct and it settles
//! nothing about a fibered representation, because a bundle does not
//! store marginals. It stores a **base** — which sites are twist-linked
//! to which — and a **fiber** over each site carrying that site's own
//! polarity data. The GHZ sign lives in a *fiber*, not in the base:
//! [`PolarityBundle::set_spin`] on one fiber flips `+` to `−` while the
//! link structure stays bit-for-bit identical, and
//! `tests/bundle.rs::the_ghz_sign_lives_in_a_fiber_not_in_the_base`
//! measures both — same base, orthogonal states. The obstruction was an
//! obstruction to marginals. It was never one to fibers.
//!
//! ## The structure
//!
//! * **Base**: `n` sites. **Fiber** over each: a [`Fiber`] — local
//!   polarity axis ([`Frame`]), sign (`spin`), absorbed weight.
//! * **Twist**: a *sparse* link set. Two sites are linked when their
//!   polarities anticommute. Storage is `O(n + |E|)`, never `2ⁿ`, which
//!   is what lets the thing run at 10⁵–10⁶ sites
//!   (`examples/polarity_bundle.rs` measures it).
//! * **Re-orderable**: the generator sequence is a maintained
//!   permutation, and reordering is not free. Swapping two *linked*
//!   sites flips the bundle's [`chirality`](PolarityBundle::chirality):
//!   the sign a product of anticommuting generators picks up under an
//!   odd permutation. Chirality is a measured consequence of twist plus
//!   ordering, cross-checked against
//!   [`PolaritySystem::product`](crate::polarity::PolaritySystem::product).
//! * **Journalled**: every operation appends a [`BundleOp`].
//!   [`PolarityBundle::rewind`] replays a prefix, so any past state is
//!   reconstructible and inspectable; [`PolarityBundle::inspect`] reads
//!   any fiber, and [`PolarityBundle::verify_against`] reads the fiber
//!   data back off a real simulated state — tomography of the fibers,
//!   not of the amplitudes.
//! * **Co-bundle / interaction on commonality**: two bundles interact
//!   only where they *share* structure.
//!   [`PolarityBundle::commonality`] measures the shared sites, shared
//!   links and frame-compatible fibers;
//!   [`PolarityBundle::interact`] applies the interaction confined to
//!   exactly that set and journals what it touched.
//!
//! ## Entanglement as a managed resource
//!
//! The link set *is* the entanglement, in the open, countable and
//! editable. [`PolarityBundle::profile`] reports degree distribution,
//! connected components (independent entangled clusters) and bytes;
//! [`PolarityBundle::coarse_grain`] merges a group of fibers into one,
//! **absorbing** the internal links and reporting how many were spent;
//! [`PolarityBundle::coarse_grain_to_budget`] drives the total link
//! count under a cap and says what it cost. Entanglement stops being an
//! emergent accident of the representation and becomes a quantity with
//! a budget, an owner and an audit trail.
//!
//! ## What this is exact for, stated first
//!
//! A bundle denotes the state
//! `∏ F_q · ∏ Z_q^{spin_q} · ∏_{(a,b) ∈ E} CZ_{ab} · |+⟩^{⊗n}` — a
//! **graph state dressed by local frames**, i.e. the local-Clifford
//! orbit of graph states. On that sector the representation is *exact*
//! and `O(n + |E|)`, which is why 10⁵ sites is a measured fact rather
//! than an aspiration. That sector is a known classical island
//! (Gottesman–Knill); nothing here moves the boundary of what is
//! efficiently simulable, and this module does not claim to.
//!
//! What is new is the *management*: ordering with measured chirality,
//! coarse-graining with a measured cost, commonality-confined
//! interaction, and a journal that makes all of it auditable — at a
//! scale where the amplitude picture does not exist. Operations outside
//! the sector are refused by name rather than silently approximated;
//! [`PolarityBundle::to_state`] exists only up to
//! [`BUNDLE_STATE_MAX_SITES`] because past that the *comparison* state
//! is what fails, not the bundle.

use std::collections::HashMap;

use crate::backend::{pauli_expectation, Backend, DenseState};
use crate::circuit::Circuit;
use crate::error::{Error, Result};
use crate::gates::Pauli;
use crate::math::GateMatrix;
use crate::registry::GateRegistry;
use crate::scalar::{Scalar, C64};

/// Widest bundle [`PolarityBundle::to_state`] will materialize. The
/// bundle itself has no such limit; the dense comparison state does.
pub const BUNDLE_STATE_MAX_SITES: usize = 20;

/// The local polarity axis a fiber points along.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Frame {
    /// The computational axis — no local rotation.
    #[default]
    Z,
    /// Rotated by `h`.
    X,
    /// Rotated by `h` then `s`.
    Y,
}

impl Frame {
    /// The local gates this frame applies, in order.
    pub fn gates(self) -> &'static [&'static str] {
        match self {
            Frame::Z => &[],
            Frame::X => &["h"],
            Frame::Y => &["h", "s"],
        }
    }

    /// Whether two fibers can interact directly: same axis, or one of
    /// them unrotated.
    pub fn compatible(self, other: Frame) -> bool {
        self == other || self == Frame::Z || other == Frame::Z
    }
}

/// One fiber: the polarity data local to a single site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fiber {
    /// Local polarity axis. Retained as the *description* label; the
    /// authority once gates run is [`Fiber::vop`], and `set_frame`
    /// keeps the two in step.
    pub frame: Frame,
    /// The fiber's vertex operator: its local Clifford. This is what
    /// gate action composes into.
    pub vop: vop::Vop,
    /// Local sign. This is where a GHZ-style global sign lives.
    pub spin: bool,
    /// How many original sites this fiber represents (1 until it
    /// absorbs others through coarse-graining).
    pub weight: u32,
    /// False once the fiber has been absorbed into another.
    pub live: bool,
}

impl Default for Fiber {
    fn default() -> Self {
        Fiber {
            frame: Frame::default(),
            vop: vop::IDENTITY,
            spin: false,
            weight: 1,
            live: true,
        }
    }
}

/// A journalled operation on a bundle. The journal is append-only and
/// replayable, so any earlier configuration is reconstructible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BundleOp {
    /// Twist-link two sites.
    Link(u32, u32),
    /// Remove a twist link.
    Unlink(u32, u32),
    /// Set a fiber's local sign.
    Spin(u32, bool),
    /// Set a fiber's local axis.
    Reframe(u32, Frame),
    /// Compose a vertex operator into a fiber.
    Vertex(u32, vop::Vop),
    /// Local complementation at a site.
    LocalComplement(u32),
    /// Swap two adjacent positions in the generator ordering.
    SwapOrder(u32),
    /// Absorb `absorbed` into `into`, spending `internal` links.
    Merge {
        /// Surviving fiber.
        into: u32,
        /// Fibers absorbed into it.
        absorbed: Vec<u32>,
        /// Links that became internal and were spent.
        internal: u32,
    },
}

/// A read-only view of one fiber and its immediate twist structure.
#[derive(Debug, Clone, PartialEq)]
pub struct FiberView {
    /// Site index.
    pub site: usize,
    /// The fiber's own data.
    pub fiber: Fiber,
    /// Current position in the generator ordering.
    pub position: usize,
    /// Sites this fiber is twist-linked to.
    pub links: Vec<u32>,
}

/// What a bundle's entanglement currently costs.
#[derive(Debug, Clone, PartialEq)]
pub struct EntanglementProfile {
    /// Sites in the base (including absorbed ones).
    pub sites: usize,
    /// Fibers still live.
    pub live_sites: usize,
    /// Twist links — the entanglement, counted.
    pub links: usize,
    /// Highest fiber degree.
    pub max_degree: usize,
    /// Mean degree over live fibers.
    pub mean_degree: f64,
    /// Connected components: independent entangled clusters.
    pub components: usize,
    /// Sites in the largest cluster.
    pub largest_component: usize,
    /// Links spent so far by coarse-graining.
    pub absorbed_links: usize,
    /// Journalled operations retained.
    pub journal_ops: usize,
    /// Bytes the journal alone costs. It is usually the largest term —
    /// auditability is not free, and [`PolarityBundle::checkpoint`]
    /// is how you decline to pay for it.
    pub journal_bytes: usize,
    /// Estimated heap + struct bytes, journal included.
    pub bytes: usize,
}

/// What one coarse-graining step cost and kept.
#[derive(Debug, Clone, PartialEq)]
pub struct CoarseReport {
    /// Merges performed.
    pub merges: usize,
    /// Fibers absorbed.
    pub absorbed_fibers: usize,
    /// Links that became internal to a super-fiber and were spent.
    pub absorbed_links: usize,
    /// Parallel links merged when several members shared an external
    /// neighbour — also spent, by a different route.
    pub collapsed_links: usize,
    /// Links remaining afterwards.
    pub links_after: usize,
    /// Total original sites now represented by the largest fiber.
    pub max_weight: u32,
}

/// Where two bundles overlap, and therefore where they can interact.
#[derive(Debug, Clone, PartialEq)]
pub struct Commonality {
    /// Sites present and live in both.
    pub shared_sites: usize,
    /// Links present in both.
    pub shared_links: usize,
    /// Links in exactly one of them, over the shared sites.
    pub divergent_links: usize,
    /// Shared sites whose frames can interact.
    pub compatible_fibers: usize,
    /// Shared sites whose frames cannot.
    pub incompatible_fibers: usize,
    /// The sites interaction is confined to.
    pub interactable: Vec<u32>,
}

/// A fibered, re-orderable, journalled polarity bundle.
///
/// ```
/// use quantsim::bundle::{Frame, PolarityBundle};
///
/// // A five-site star: the GHZ entanglement structure.
/// let mut bundle = PolarityBundle::new(5).unwrap();
/// for leaf in 1..5 {
///     bundle.link(0, leaf).unwrap();
///     bundle.set_frame(leaf, Frame::X).unwrap();
/// }
/// assert_eq!(bundle.profile().links, 4);
///
/// // The GHZ sign is fiber data, not base data.
/// let base = bundle.link_signature();
/// bundle.set_spin(0, true).unwrap();
/// assert_eq!(bundle.link_signature(), base);
/// ```
#[derive(Debug, Clone)]
pub struct PolarityBundle {
    fibers: Vec<Fiber>,
    links: Vec<Vec<u32>>,
    order: Vec<u32>,
    position: Vec<u32>,
    chirality: i8,
    link_count: usize,
    absorbed_links: usize,
    journal: Vec<BundleOp>,
}

impl PolarityBundle {
    /// A bundle of `sites` unlinked fibers in the default frame.
    pub fn new(sites: usize) -> Result<Self> {
        if sites == 0 {
            return Err(Error::InvalidState(
                "a polarity bundle needs at least one site".into(),
            ));
        }
        if sites > u32::MAX as usize {
            return Err(Error::InvalidState(format!(
                "{sites} sites exceeds the u32 site index"
            )));
        }
        Ok(PolarityBundle {
            fibers: vec![Fiber::default(); sites],
            links: vec![Vec::new(); sites],
            order: (0..sites as u32).collect(),
            position: (0..sites as u32).collect(),
            chirality: 1,
            link_count: 0,
            absorbed_links: 0,
            journal: Vec::new(),
        })
    }

    /// Sites in the base.
    pub fn sites(&self) -> usize {
        self.fibers.len()
    }

    fn check(&self, site: u32) -> Result<usize> {
        let s = site as usize;
        if s >= self.fibers.len() {
            return Err(Error::QubitOutOfRange {
                qubit: s,
                num_qubits: self.fibers.len(),
            });
        }
        if !self.fibers[s].live {
            return Err(Error::InvalidState(format!(
                "site {s} was absorbed by coarse-graining and is no longer a live fiber"
            )));
        }
        Ok(s)
    }

    // ── the base: twist links ────────────────────────────────────────

    /// Twist-link two sites. Idempotent, `O(deg)`.
    pub fn link(&mut self, a: u32, b: u32) -> Result<bool> {
        let (ia, ib) = (self.check(a)?, self.check(b)?);
        if ia == ib {
            return Err(Error::InvalidState(format!(
                "site {ia} cannot be twist-linked to itself"
            )));
        }
        if let Err(pos) = self.links[ia].binary_search(&b) {
            self.links[ia].insert(pos, b);
            let pos_b = self.links[ib].binary_search(&a).unwrap_err();
            self.links[ib].insert(pos_b, a);
            self.link_count += 1;
            self.journal.push(BundleOp::Link(a, b));
            return Ok(true);
        }
        Ok(false)
    }

    /// Remove a twist link. `O(deg)`.
    pub fn unlink(&mut self, a: u32, b: u32) -> Result<bool> {
        let (ia, ib) = (self.check(a)?, self.check(b)?);
        if let Ok(pos) = self.links[ia].binary_search(&b) {
            self.links[ia].remove(pos);
            let pos_b = self.links[ib]
                .binary_search(&a)
                .expect("links are symmetric");
            self.links[ib].remove(pos_b);
            self.link_count -= 1;
            self.journal.push(BundleOp::Unlink(a, b));
            return Ok(true);
        }
        Ok(false)
    }

    /// Whether two sites are twist-linked.
    pub fn linked(&self, a: u32, b: u32) -> bool {
        self.links
            .get(a as usize)
            .is_some_and(|l| l.binary_search(&b).is_ok())
    }

    /// A fiber's twist links.
    pub fn neighbours(&self, site: u32) -> &[u32] {
        &self.links[site as usize]
    }

    /// An order-independent fingerprint of the base — the link
    /// structure alone, with no fiber data. Two bundles with the same
    /// signature carry the same entanglement structure.
    pub fn link_signature(&self) -> u64 {
        let mut acc = 0xcbf2_9ce4_8422_2325u64;
        for (a, neighbours) in self.links.iter().enumerate() {
            for &b in neighbours {
                if (a as u32) < b {
                    let key = ((a as u64) << 32) | b as u64;
                    acc ^= key.wrapping_mul(0x100_0000_01b3);
                    acc = acc.rotate_left(13);
                }
            }
        }
        acc
    }

    // ── the fibers ───────────────────────────────────────────────────

    /// Set a fiber's local sign. This is fiber data: the base is
    /// untouched.
    pub fn set_spin(&mut self, site: u32, spin: bool) -> Result<()> {
        let s = self.check(site)?;
        self.fibers[s].spin = spin;
        self.journal.push(BundleOp::Spin(site, spin));
        Ok(())
    }

    /// Set a fiber's local polarity axis.
    pub fn set_frame(&mut self, site: u32, frame: Frame) -> Result<()> {
        let s = self.check(site)?;
        self.fibers[s].frame = frame;
        self.fibers[s].vop = match frame {
            Frame::Z => vop::IDENTITY,
            Frame::X => vop::hadamard(),
            Frame::Y => vop::compose(vop::phase(), vop::hadamard()),
        };
        self.journal.push(BundleOp::Reframe(site, frame));
        Ok(())
    }

    /// Inspect one fiber and its immediate twist structure.
    pub fn inspect(&self, site: u32) -> Result<FiberView> {
        let s = site as usize;
        if s >= self.fibers.len() {
            return Err(Error::QubitOutOfRange {
                qubit: s,
                num_qubits: self.fibers.len(),
            });
        }
        Ok(FiberView {
            site: s,
            fiber: self.fibers[s],
            position: self.position[s] as usize,
            links: self.links[s].clone(),
        })
    }

    // ── ordering and chirality ───────────────────────────────────────

    /// The current generator ordering: `order()[k]` is the site at
    /// position `k`.
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// The accumulated reordering sign, `±1`.
    ///
    /// Swapping two *linked* (anticommuting) sites flips it; swapping
    /// unlinked ones does not. This is the same sign a product of
    /// polarity generators picks up under reordering — chirality as a
    /// measured consequence of twist plus order, not an extra postulate.
    pub fn chirality(&self) -> i8 {
        self.chirality
    }

    /// Swap the sites at positions `pos` and `pos + 1`. `O(1)`.
    pub fn swap_order(&mut self, pos: usize) -> Result<()> {
        if pos + 1 >= self.order.len() {
            return Err(Error::InvalidState(format!(
                "position {pos} has no successor to swap with in {} sites",
                self.order.len()
            )));
        }
        let (a, b) = (self.order[pos], self.order[pos + 1]);
        if self.linked(a, b) {
            self.chirality = -self.chirality;
        }
        self.order.swap(pos, pos + 1);
        self.position[a as usize] = pos as u32 + 1;
        self.position[b as usize] = pos as u32;
        self.journal.push(BundleOp::SwapOrder(pos as u32));
        Ok(())
    }

    /// Re-order the whole bundle, accumulating chirality in one pass:
    /// the parity of *linked* pairs whose relative order changed.
    /// `O(n + |E|)`.
    pub fn reorder(&mut self, new_order: &[u32]) -> Result<i8> {
        let n = self.order.len();
        if new_order.len() != n {
            return Err(Error::WidthMismatch {
                circuit: n,
                backend: new_order.len(),
            });
        }
        let mut seen = vec![false; n];
        let mut new_position = vec![0u32; n];
        for (pos, &site) in new_order.iter().enumerate() {
            let s = site as usize;
            if s >= n || seen[s] {
                return Err(Error::InvalidState(format!(
                    "the new order is not a permutation of the {n} sites"
                )));
            }
            seen[s] = true;
            new_position[s] = pos as u32;
        }
        // Only linked pairs carry a sign, so only they are counted.
        let mut flips = 0usize;
        for (a, neighbours) in self.links.iter().enumerate() {
            for &b in neighbours {
                if (a as u32) < b {
                    let before = self.position[a] < self.position[b as usize];
                    let after = new_position[a] < new_position[b as usize];
                    if before != after {
                        flips += 1;
                    }
                }
            }
        }
        if flips % 2 == 1 {
            self.chirality = -self.chirality;
        }
        self.order = new_order.to_vec();
        self.position = new_position;
        Ok(self.chirality)
    }

    // ── entanglement as a managed resource ───────────────────────────

    /// Measure what the bundle's entanglement currently costs.
    pub fn profile(&self) -> EntanglementProfile {
        let live: Vec<usize> = (0..self.fibers.len())
            .filter(|&s| self.fibers[s].live)
            .collect();
        let max_degree = live.iter().map(|&s| self.links[s].len()).max().unwrap_or(0);
        let mean_degree = if live.is_empty() {
            0.0
        } else {
            live.iter().map(|&s| self.links[s].len()).sum::<usize>() as f64 / live.len() as f64
        };
        let (components, largest_component) = self.components();
        let journal_bytes = self.journal.capacity() * std::mem::size_of::<BundleOp>();
        let bytes = std::mem::size_of::<Self>()
            + self.fibers.capacity() * std::mem::size_of::<Fiber>()
            + self.links.capacity() * std::mem::size_of::<Vec<u32>>()
            + self.links.iter().map(|l| l.capacity() * 4).sum::<usize>()
            + (self.order.capacity() + self.position.capacity()) * 4
            + journal_bytes;
        EntanglementProfile {
            sites: self.fibers.len(),
            live_sites: live.len(),
            links: self.link_count,
            max_degree,
            mean_degree,
            components,
            largest_component,
            absorbed_links: self.absorbed_links,
            journal_ops: self.journal.len(),
            journal_bytes,
            bytes,
        }
    }

    /// Connected components of the twist graph — independent entangled
    /// clusters — and the size of the largest. `O(n + |E|)`.
    fn components(&self) -> (usize, usize) {
        let n = self.fibers.len();
        let mut seen = vec![false; n];
        let mut stack: Vec<u32> = Vec::new();
        let (mut count, mut largest) = (0usize, 0usize);
        for start in 0..n {
            if seen[start] || !self.fibers[start].live {
                continue;
            }
            count += 1;
            let mut size = 0usize;
            seen[start] = true;
            stack.push(start as u32);
            while let Some(site) = stack.pop() {
                size += 1;
                for &next in &self.links[site as usize] {
                    if !seen[next as usize] {
                        seen[next as usize] = true;
                        stack.push(next);
                    }
                }
            }
            largest = largest.max(size);
        }
        (count, largest)
    }

    /// **Coarse-grain**: absorb `group` into its lowest-indexed member.
    /// Links internal to the group are *spent* (counted, then dropped);
    /// links leaving the group are re-pointed onto the survivor.
    ///
    /// This is the entanglement-management primitive: it trades
    /// resolution for link count, and reports exactly what it traded.
    pub fn coarse_grain(&mut self, group: &[u32]) -> Result<CoarseReport> {
        if group.len() < 2 {
            return Err(Error::InvalidState(
                "coarse-graining needs at least two fibers to merge".into(),
            ));
        }
        let mut members: Vec<u32> = group.to_vec();
        members.sort_unstable();
        members.dedup();
        for &m in &members {
            self.check(m)?;
        }
        let survivor = members[0];
        let absorbed: Vec<u32> = members[1..].to_vec();
        let member_set: std::collections::HashSet<u32> = members.iter().copied().collect();

        let mut internal = 0u32;
        // Every link leaving the group is removed; several members may
        // share one external neighbour, so the number of links removed
        // and the number of distinct neighbours are different counts.
        let mut removed_external = 0usize;
        let mut external: Vec<u32> = Vec::new();
        for &m in &members {
            for &nb in &self.links[m as usize] {
                if member_set.contains(&nb) {
                    if m < nb {
                        internal += 1;
                    }
                } else {
                    removed_external += 1;
                    external.push(nb);
                }
            }
        }
        external.sort_unstable();
        external.dedup();
        let collapsed = removed_external - external.len();

        // Detach every member from the graph.
        for &m in &members {
            let neighbours = std::mem::take(&mut self.links[m as usize]);
            for nb in neighbours {
                if !member_set.contains(&nb) {
                    if let Ok(pos) = self.links[nb as usize].binary_search(&m) {
                        self.links[nb as usize].remove(pos);
                    }
                }
            }
        }
        self.link_count -= internal as usize + removed_external;
        self.absorbed_links += internal as usize + collapsed;

        // Re-attach the survivor to the external boundary.
        self.links[survivor as usize] = external.clone();
        for &nb in &external {
            let l = &mut self.links[nb as usize];
            if let Err(pos) = l.binary_search(&survivor) {
                l.insert(pos, survivor);
            }
        }
        self.link_count += external.len();

        let mut weight = self.fibers[survivor as usize].weight;
        for &m in &absorbed {
            weight += self.fibers[m as usize].weight;
            self.fibers[m as usize].live = false;
            self.fibers[m as usize].weight = 0;
        }
        self.fibers[survivor as usize].weight = weight;

        self.journal.push(BundleOp::Merge {
            into: survivor,
            absorbed: absorbed.clone(),
            internal,
        });
        Ok(CoarseReport {
            merges: 1,
            absorbed_fibers: absorbed.len(),
            absorbed_links: internal as usize,
            collapsed_links: collapsed,
            links_after: self.link_count,
            max_weight: self.fibers.iter().map(|f| f.weight).max().unwrap_or(0),
        })
    }

    /// Drive the total link count under `max_links` by repeatedly
    /// coarse-graining the highest-degree fiber together with its
    /// neighbourhood, and report what it cost.
    ///
    /// Refuses rather than looping forever when the budget cannot be
    /// met — a bundle that is already a single fiber has no links left
    /// to spend.
    pub fn coarse_grain_to_budget(&mut self, max_links: usize) -> Result<CoarseReport> {
        let mut total = CoarseReport {
            merges: 0,
            absorbed_fibers: 0,
            absorbed_links: 0,
            collapsed_links: 0,
            links_after: self.link_count,
            max_weight: 0,
        };
        while self.link_count > max_links {
            let hottest = (0..self.fibers.len())
                .filter(|&s| self.fibers[s].live && !self.links[s].is_empty())
                .max_by_key(|&s| self.links[s].len());
            let Some(centre) = hottest else {
                return Err(Error::InvalidState(format!(
                    "no links remain but the count is {}; the budget of {max_links} \
                     cannot be met by coarse-graining",
                    self.link_count
                )));
            };
            let mut group = vec![centre as u32];
            group.extend_from_slice(&self.links[centre]);
            let step = self.coarse_grain(&group)?;
            total.merges += 1;
            total.absorbed_fibers += step.absorbed_fibers;
            total.absorbed_links += step.absorbed_links;
            total.collapsed_links += step.collapsed_links;
            total.links_after = step.links_after;
            total.max_weight = step.max_weight;
        }
        Ok(total)
    }

    // ── the co-bundle: interaction on commonality ────────────────────

    /// Measure where two bundles overlap: shared sites, shared links,
    /// and which fibers have compatible polarity.
    pub fn commonality(&self, other: &PolarityBundle) -> Commonality {
        let n = self.fibers.len().min(other.fibers.len());
        let mut shared_sites = 0usize;
        let mut compatible = 0usize;
        let mut incompatible = 0usize;
        let mut interactable = Vec::new();
        for s in 0..n {
            if !(self.fibers[s].live && other.fibers[s].live) {
                continue;
            }
            shared_sites += 1;
            if self.fibers[s].frame.compatible(other.fibers[s].frame) {
                compatible += 1;
                interactable.push(s as u32);
            } else {
                incompatible += 1;
            }
        }
        let mut shared_links = 0usize;
        let mut divergent_links = 0usize;
        for a in 0..n {
            for &b in &self.links[a] {
                if (a as u32) < b && (b as usize) < n {
                    if other.linked(a as u32, b) {
                        shared_links += 1;
                    } else {
                        divergent_links += 1;
                    }
                }
            }
            for &b in &other.links[a] {
                if (a as u32) < b && (b as usize) < n && !self.linked(a as u32, b) {
                    divergent_links += 1;
                }
            }
        }
        Commonality {
            shared_sites,
            shared_links,
            divergent_links,
            compatible_fibers: compatible,
            incompatible_fibers: incompatible,
            interactable,
        }
    }

    /// Interact with another bundle, **confined to the commonality**:
    /// links are combined only on frame-compatible shared sites, and
    /// fiber signs XOR there. Everything outside the interactable set
    /// is left untouched, and the journal records exactly what was.
    ///
    /// Combining is symmetric-difference on links — two twists in the
    /// same place cancel, which is the polarity rule the algebra
    /// already obeys.
    pub fn interact(&mut self, other: &PolarityBundle) -> Result<Commonality> {
        let common = self.commonality(other);
        let live: std::collections::HashSet<u32> = common.interactable.iter().copied().collect();
        let mut toggles: Vec<(u32, u32)> = Vec::new();
        for &a in &common.interactable {
            for &b in &other.links[a as usize] {
                if a < b && live.contains(&b) {
                    toggles.push((a, b));
                }
            }
        }
        for (a, b) in toggles {
            if self.linked(a, b) {
                self.unlink(a, b)?;
            } else {
                self.link(a, b)?;
            }
        }
        for &s in &common.interactable {
            let spin = self.fibers[s as usize].spin ^ other.fibers[s as usize].spin;
            self.set_spin(s, spin)?;
        }
        Ok(common)
    }

    // ── the journal ──────────────────────────────────────────────────

    /// Every operation applied, in order.
    pub fn journal(&self) -> &[BundleOp] {
        &self.journal
    }

    /// Drop the journal, keeping the current configuration. Returns how
    /// many operations were discarded.
    ///
    /// The journal is the largest term in
    /// [`EntanglementProfile::bytes`] on a large bundle, so this is the
    /// explicit trade: give up the ability to
    /// [`rewind`](Self::rewind) past this point, get the space back.
    pub fn checkpoint(&mut self) -> usize {
        let dropped = self.journal.len();
        self.journal.clear();
        self.journal.shrink_to_fit();
        dropped
    }

    /// Rebuild the bundle as it stood after the first `steps`
    /// journalled operations. The journal is the state.
    pub fn rewind(&self, steps: usize) -> Result<PolarityBundle> {
        if steps > self.journal.len() {
            return Err(Error::InvalidState(format!(
                "cannot rewind to step {steps} of a {}-step journal",
                self.journal.len()
            )));
        }
        let mut out = PolarityBundle::new(self.fibers.len())?;
        for op in &self.journal[..steps] {
            match op {
                BundleOp::Link(a, b) => {
                    out.link(*a, *b)?;
                }
                BundleOp::Unlink(a, b) => {
                    out.unlink(*a, *b)?;
                }
                BundleOp::Spin(s, v) => out.set_spin(*s, *v)?,
                BundleOp::Reframe(s, f) => out.set_frame(*s, *f)?,
                BundleOp::Vertex(s, v) => out.apply_vop(*s, *v)?,
                BundleOp::LocalComplement(s) => out.local_complement(*s)?,
                BundleOp::SwapOrder(p) => out.swap_order(*p as usize)?,
                BundleOp::Merge { into, absorbed, .. } => {
                    let mut group = vec![*into];
                    group.extend_from_slice(absorbed);
                    out.coarse_grain(&group)?;
                }
            }
        }
        Ok(out)
    }

    // ── the anchor to real states ────────────────────────────────────

    /// Materialize the state the bundle denotes:
    /// `∏ F_q · ∏ Z_q^{spin} · ∏_{(a,b) ∈ E} CZ_{ab} · |+⟩^{⊗n}`.
    ///
    /// Only up to [`BUNDLE_STATE_MAX_SITES`] — the limit is the dense
    /// state vector's, not the bundle's.
    pub fn to_state(&self) -> Result<Box<dyn Backend<C64>>> {
        let n = self.fibers.len();
        if n > BUNDLE_STATE_MAX_SITES {
            return Err(Error::TooManyQubits {
                requested: n,
                max: BUNDLE_STATE_MAX_SITES,
            });
        }
        if self.fibers.iter().any(|f| !f.live) {
            return Err(Error::InvalidState(
                "a coarse-grained bundle no longer denotes a state on its original sites; \
                 rewind past the merge to materialize one"
                    .into(),
            ));
        }
        let mut circuit: Circuit<C64> = Circuit::new(n);
        for q in 0..n {
            circuit.h(q);
        }
        for a in 0..n {
            for &b in &self.links[a] {
                if (a as u32) < b {
                    circuit.cz(a, b as usize);
                }
            }
        }
        for (q, fiber) in self.fibers.iter().enumerate() {
            if fiber.spin {
                circuit.z(q);
            }
            if fiber.vop != vop::IDENTITY {
                let m =
                    vop::matrix::<C64>(fiber.vop).ok_or_else(|| Error::UnsupportedForAlgebra {
                        gate: "vertex operator".into(),
                        algebra: "non-complex".into(),
                    })?;
                circuit.raw("vop", GateMatrix::from_vec(2, m.to_vec())?, vec![q]);
            }
        }
        let registry = GateRegistry::<C64>::standard();
        let mut state = DenseState::<C64>::new(n)?;
        circuit.bind(&registry)?.run(&mut state)?;
        Ok(Box::new(state))
    }

    /// **Fiber tomography**: read the fiber signs back off a state by
    /// measuring each site's stabilizer generator
    /// `F_q X_q F_q† · ∏_{b ∈ N(q)} Z_b`, and report the recovered
    /// signs together with the worst `|⟨g⟩| − 1` deviation.
    ///
    /// A deviation near zero means the state really is the one this
    /// bundle denotes, and the recovered signs are exact. A large
    /// deviation means the state has left the bundle's sector — which
    /// is reported, not smoothed over.
    pub fn verify_against(&self, state: &dyn Backend<C64>) -> Result<Tomography> {
        let n = self.fibers.len();
        if state.num_qubits() != n {
            return Err(Error::WidthMismatch {
                circuit: n,
                backend: state.num_qubits(),
            });
        }
        let mut spins = Vec::with_capacity(n);
        let mut worst = 0.0f64;
        for q in 0..n {
            // F X F†: Z-frame keeps X, X-frame sends it to Z, Y-frame to Y.
            // The generator is F X F† at q and F Z F† at each
            // neighbour, read straight off the vertex operators; the
            // signs they introduce fold into the reading below.
            let pauli_of = |p: u8| match p {
                0 => Pauli::X,
                1 => Pauli::Y,
                _ => Pauli::Z,
            };
            let (xp, mut negative) = vop::map_pauli(self.fibers[q].vop, 0, false);
            let mut ops = vec![(q, pauli_of(xp))];
            for &b in &self.links[q] {
                let nb = b as usize;
                let (zp, zs) = vop::map_pauli(self.fibers[nb].vop, 2, false);
                negative ^= zs;
                ops.push((nb, pauli_of(zp)));
            }
            let mut value = pauli_expectation(state, &ops)?.re;
            if negative {
                value = -value;
            }
            worst = worst.max((value.abs() - 1.0).abs());
            spins.push(value < 0.0);
        }
        Ok(Tomography {
            spins,
            deviation: worst,
        })
    }

    /// Ordering-sensitive census of the bundle by position: how many
    /// links run forward from each position. Together with
    /// [`chirality`](Self::chirality) this is the "co-" side of the
    /// bundle — what an observer reading the fibers in order sees.
    pub fn forward_census(&self) -> Vec<u32> {
        self.order
            .iter()
            .map(|&site| {
                let here = self.position[site as usize];
                self.links[site as usize]
                    .iter()
                    .filter(|&&nb| self.position[nb as usize] > here)
                    .count() as u32
            })
            .collect()
    }
}

/// The result of reading fiber data back off a simulated state.
#[derive(Debug, Clone, PartialEq)]
pub struct Tomography {
    /// Recovered fiber signs, one per site.
    pub spins: Vec<bool>,
    /// Worst `||⟨g⟩| − 1|` over the stabilizer generators. Near zero
    /// means the state is in the bundle's sector.
    pub deviation: f64,
}

/// Build a bundle whose denoted state is the `n`-site GHZ state: a star
/// base with `X` frames on the leaves. `sign = true` gives the `−`
/// state, and does so by fiber data alone.
pub fn ghz_bundle(sites: usize, sign: bool) -> Result<PolarityBundle> {
    if sites < 2 {
        return Err(Error::InvalidState(
            "a GHZ bundle needs at least two sites".into(),
        ));
    }
    let mut bundle = PolarityBundle::new(sites)?;
    for leaf in 1..sites as u32 {
        bundle.link(0, leaf)?;
        bundle.set_frame(leaf, Frame::X)?;
    }
    if sign {
        bundle.set_spin(0, true)?;
    }
    Ok(bundle)
}

/// A deterministic sparse bundle for scaling measurements: a ring plus
/// `extra` chords per site, built without ever forming an amplitude.
pub fn scaling_bundle(sites: usize, extra: usize) -> Result<PolarityBundle> {
    let mut bundle = PolarityBundle::new(sites)?;
    let n = sites as u64;
    for a in 0..sites as u32 {
        bundle.link(a, ((a as u64 + 1) % n) as u32)?;
        for k in 0..extra {
            // A fixed multiplicative stride: deterministic, no RNG.
            let stride = 2654435761u64.wrapping_mul(k as u64 + 1) % n;
            let b = ((a as u64 + stride.max(2)) % n) as u32;
            if b != a {
                bundle.link(a, b)?;
            }
        }
    }
    Ok(bundle)
}

/// Frequency of each fiber degree — the shape of the entanglement, not
/// just its total.
pub fn degree_histogram(bundle: &PolarityBundle) -> HashMap<usize, usize> {
    let mut out = HashMap::new();
    for site in 0..bundle.sites() {
        if bundle.inspect(site as u32).map(|v| v.fiber.live) == Ok(true) {
            *out.entry(bundle.neighbours(site as u32).len()).or_insert(0) += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn links_are_symmetric_and_idempotent() {
        let mut b = PolarityBundle::new(4).unwrap();
        assert!(b.link(0, 2).unwrap());
        assert!(!b.link(0, 2).unwrap());
        assert!(b.linked(0, 2) && b.linked(2, 0));
        assert_eq!(b.profile().links, 1);
        assert!(b.unlink(2, 0).unwrap());
        assert!(!b.linked(0, 2));
        assert_eq!(b.profile().links, 0);
        assert!(b.link(1, 1).is_err());
    }

    #[test]
    fn chirality_flips_only_on_linked_swaps() {
        let mut b = PolarityBundle::new(3).unwrap();
        b.link(0, 1).unwrap();
        assert_eq!(b.chirality(), 1);
        b.swap_order(0).unwrap(); // swaps sites 0 and 1 — linked
        assert_eq!(b.chirality(), -1);
        b.swap_order(1).unwrap(); // now swaps sites 0 and 2 — unlinked
        assert_eq!(b.chirality(), -1);
    }

    #[test]
    fn the_journal_replays_exactly() {
        let mut b = PolarityBundle::new(5).unwrap();
        b.link(0, 1).unwrap();
        b.set_frame(2, Frame::X).unwrap();
        b.link(2, 3).unwrap();
        b.set_spin(1, true).unwrap();
        let replayed = b.rewind(b.journal().len()).unwrap();
        assert_eq!(replayed.link_signature(), b.link_signature());
        assert_eq!(replayed.inspect(1).unwrap(), b.inspect(1).unwrap());
        // And a prefix really is the earlier state.
        let early = b.rewind(1).unwrap();
        assert_eq!(early.profile().links, 1);
        assert!(!early.inspect(1).unwrap().fiber.spin);
    }
}

// ── the single-qubit Clifford group, as vertex operators ─────────────

/// The 24 single-qubit Cliffords, identified by their action on the
/// Pauli generators: `X ↦ ±P`, `Z ↦ ±Q` with `P ≠ Q`.
///
/// This is what a [`Fiber`]'s frame really is once the bundle has to
/// *evolve* rather than merely be described: three axes were enough to
/// denote a state, but gate action needs the whole local group.
pub mod vop {
    use crate::scalar::{Scalar, C64};
    use std::sync::OnceLock;

    /// A vertex operator: index into the 24-element group.
    pub type Vop = u8;

    /// The identity.
    pub const IDENTITY: Vop = 0;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct Action {
        /// Image of X: Pauli index (0=X, 1=Y, 2=Z) and sign.
        xp: u8,
        xs: bool,
        /// Image of Z.
        zp: u8,
        zs: bool,
    }

    /// `P_a · P_b = i^k · P_r` for distinct non-identity Paulis.
    fn pauli_mul(a: u8, b: u8) -> (u8, u8) {
        if (a + 1) % 3 == b {
            ((a + 2) % 3, 1)
        } else {
            ((a + 1) % 3, 3)
        }
    }

    impl Action {
        const fn id() -> Self {
            Action {
                xp: 0,
                xs: false,
                zp: 2,
                zs: false,
            }
        }

        /// Image of a signed Pauli under this Clifford.
        fn map(self, p: u8, sign: bool) -> (u8, bool) {
            match p {
                0 => (self.xp, self.xs ^ sign),
                2 => (self.zp, self.zs ^ sign),
                _ => {
                    // Y = i·X·Z, so the image is i·(image X)(image Z).
                    let (r, k) = pauli_mul(self.xp, self.zp);
                    let negative = ((1 + k) % 4 == 2) ^ self.xs ^ self.zs;
                    (r, negative ^ sign)
                }
            }
        }

        /// `self ∘ other`.
        fn compose(self, other: Action) -> Action {
            let (xp, xs) = self.map(other.xp, other.xs);
            let (zp, zs) = self.map(other.zp, other.zs);
            Action { xp, xs, zp, zs }
        }
    }

    struct Group {
        actions: Vec<Action>,
        table: Vec<Vop>,
        h: Vop,
        s: Vop,
    }

    fn group() -> &'static Group {
        static GROUP: OnceLock<Group> = OnceLock::new();
        GROUP.get_or_init(|| {
            let h = Action {
                xp: 2,
                xs: false,
                zp: 0,
                zs: false,
            };
            let s = Action {
                xp: 1,
                xs: false,
                zp: 2,
                zs: false,
            };
            // Closure of {H, S} under composition, identity first.
            let mut actions = vec![Action::id()];
            let mut frontier = vec![Action::id()];
            while let Some(a) = frontier.pop() {
                for g in [h, s] {
                    let next = g.compose(a);
                    if !actions.contains(&next) {
                        actions.push(next);
                        frontier.push(next);
                    }
                }
            }
            assert_eq!(
                actions.len(),
                24,
                "the single-qubit Clifford group has 24 elements"
            );
            let n = actions.len();
            let mut table = vec![0u8; n * n];
            for (i, &a) in actions.iter().enumerate() {
                for (j, &b) in actions.iter().enumerate() {
                    let c = a.compose(b);
                    table[i * n + j] = actions
                        .iter()
                        .position(|&x| x == c)
                        .expect("group is closed") as u8;
                }
            }
            let h_index = actions.iter().position(|&x| x == h).unwrap() as u8;
            let s_index = actions.iter().position(|&x| x == s).unwrap() as u8;
            Group {
                actions,
                table,
                h: h_index,
                s: s_index,
            }
        })
    }

    /// Number of vertex operators (24).
    pub fn count() -> usize {
        group().actions.len()
    }

    /// `a ∘ b` — apply `b` first.
    pub fn compose(a: Vop, b: Vop) -> Vop {
        let g = group();
        g.table[a as usize * g.actions.len() + b as usize]
    }

    /// The Hadamard vertex operator.
    pub fn hadamard() -> Vop {
        group().h
    }

    /// The phase vertex operator.
    pub fn phase() -> Vop {
        group().s
    }

    /// Whether the operator is a diagonal matrix — equivalently,
    /// whether it maps `Z` to `+Z`. Exactly these four commute with a
    /// `cz`, which is why reduction into this subgroup is what edge
    /// toggling needs.
    pub fn is_diagonal(v: Vop) -> bool {
        let a = group().actions[v as usize];
        a.zp == 2 && !a.zs
    }

    fn find(xp: u8, xs: bool, zp: u8, zs: bool) -> Vop {
        let target = Action { xp, xs, zp, zs };
        group()
            .actions
            .iter()
            .position(|&a| a == target)
            .expect("every Pauli action is realized") as Vop
    }

    /// `exp(−iπ/4 · X)`: `X ↦ X`, `Z ↦ −Y`. The vertex's own update
    /// under local complementation.
    pub fn sqrt_x() -> Vop {
        find(0, false, 1, true)
    }

    /// `exp(−iπ/4 · Z)`: `X ↦ Y`, `Z ↦ Z`. A neighbour's update under
    /// local complementation.
    pub fn sqrt_z() -> Vop {
        find(1, false, 2, false)
    }

    /// Inverse of a vertex operator.
    pub fn inverse(v: Vop) -> Vop {
        (0..count() as Vop)
            .find(|&c| compose(v, c) == IDENTITY)
            .expect("the group has inverses")
    }

    /// Image of a signed Pauli.
    pub fn map_pauli(v: Vop, p: u8, sign: bool) -> (u8, bool) {
        group().actions[v as usize].map(p, sign)
    }

    /// The `2 × 2` matrix of a vertex operator over any algebra
    /// containing ℂ, up to an irrelevant global phase — reconstructed
    /// from its Pauli action so the group and the matrices cannot drift
    /// apart.
    pub fn matrix<S: Scalar>(v: Vop) -> Option<[S; 4]> {
        let m = matrix_c64(v);
        let mut out = [S::zero(); 4];
        for (slot, z) in out.iter_mut().zip(m) {
            *slot = S::try_from_c64(z)?;
        }
        Some(out)
    }

    /// Complex matrices of all 24 operators, built once by conjugating
    /// the Pauli generators and solving, then cached.
    fn matrix_c64(v: Vop) -> [C64; 4] {
        static MATRICES: OnceLock<Vec<[C64; 4]>> = OnceLock::new();
        MATRICES.get_or_init(|| {
            let g = group();
            let h = [
                C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                C64::new(-std::f64::consts::FRAC_1_SQRT_2, 0.0),
            ];
            let s = [
                C64::new(1.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 1.0),
            ];
            let id = [
                C64::new(1.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(0.0, 0.0),
                C64::new(1.0, 0.0),
            ];
            let mul = |a: [C64; 4], b: [C64; 4]| -> [C64; 4] {
                [
                    a[0] * b[0] + a[1] * b[2],
                    a[0] * b[1] + a[1] * b[3],
                    a[2] * b[0] + a[3] * b[2],
                    a[2] * b[1] + a[3] * b[3],
                ]
            };
            // Rebuild every element by the same closure order the group
            // used, so index and matrix agree by construction.
            let mut mats = vec![id; g.actions.len()];
            let mut done = vec![false; g.actions.len()];
            done[0] = true;
            let mut frontier = vec![0usize];
            while let Some(i) = frontier.pop() {
                for (gen_index, gen_mat) in [(g.h, h), (g.s, s)] {
                    let j = compose(gen_index, i as Vop) as usize;
                    if !done[j] {
                        mats[j] = mul(gen_mat, mats[i]);
                        done[j] = true;
                        frontier.push(j);
                    }
                }
            }
            mats
        })[v as usize]
    }
}

// ── gate action: the bundle as a representation, not a description ───

/// Widest register [`PolarityBundle`] will enumerate amplitudes for.
/// The *description* has no such limit; reading amplitudes out of it
/// does, because there are `2^n` of them.
pub const BUNDLE_ENUMERATION_MAX: usize = 24;

impl PolarityBundle {
    /// Apply a vertex operator to one fiber. `O(1)`.
    pub fn apply_vop(&mut self, site: u32, op: vop::Vop) -> Result<()> {
        let s = self.check(site)?;
        self.fibers[s].vop = vop::compose(op, self.fibers[s].vop);
        self.journal.push(BundleOp::Vertex(site, op));
        Ok(())
    }

    /// **Local complementation** at `site`: complement the subgraph on
    /// its neighbourhood and absorb the resulting local Cliffords into
    /// the vertex operators. The denoted state is unchanged — this is a
    /// change of description, and `tests/bundle_backend.rs` measures the
    /// state before and after to say so.
    pub fn local_complement(&mut self, site: u32) -> Result<()> {
        let s = self.check(site)?;
        let neighbours: Vec<u32> = self.links[s].clone();
        for i in 0..neighbours.len() {
            for j in (i + 1)..neighbours.len() {
                let (a, b) = (neighbours[i], neighbours[j]);
                if self.linked(a, b) {
                    self.unlink(a, b)?;
                } else {
                    self.link(a, b)?;
                }
            }
        }
        let sx = vop::inverse(vop::sqrt_x());
        self.fibers[s].vop = vop::compose(self.fibers[s].vop, sx);
        let sz = vop::sqrt_z();
        for &w in &neighbours {
            self.fibers[w as usize].vop = vop::compose(self.fibers[w as usize].vop, sz);
        }
        self.journal.push(BundleOp::LocalComplement(site));
        Ok(())
    }

    /// Bring a vertex operator into the diagonal subgroup using local
    /// complementations, so a `cz` on it becomes an edge toggle.
    ///
    /// Local complementation at the vertex right-multiplies its operator
    /// by one generator; local complementation at a *neighbour*
    /// right-multiplies it by the other. The two generate the whole
    /// group, so a word always exists — provided a neighbour exists.
    fn reduce_vop(&mut self, site: u32, avoid: u32) -> Result<()> {
        let s = site as usize;
        if vop::is_diagonal(self.fibers[s].vop) {
            return Ok(());
        }
        let helper = self.links[s]
            .iter()
            .copied()
            .find(|&w| w != avoid)
            .or_else(|| self.links[s].first().copied());
        let Some(helper) = helper else {
            return Err(Error::InvalidState(format!(
                "site {site} carries a non-diagonal vertex operator and has no neighbour \
                 to complement against; the graph description cannot absorb it"
            )));
        };
        // Breadth-first over words in {complement here, complement at
        // the helper}, tracking only the operator.
        let sx = vop::inverse(vop::sqrt_x());
        let sz = vop::sqrt_z();
        let start = self.fibers[s].vop;
        let mut seen = std::collections::HashMap::new();
        seen.insert(start, Vec::<bool>::new());
        let mut queue = std::collections::VecDeque::from([start]);
        let mut word = None;
        while let Some(current) = queue.pop_front() {
            if vop::is_diagonal(current) {
                word = Some(seen[&current].clone());
                break;
            }
            let base = seen[&current].clone();
            for (here, generator) in [(true, sx), (false, sz)] {
                let next = vop::compose(current, generator);
                if let std::collections::hash_map::Entry::Vacant(slot) = seen.entry(next) {
                    let mut path = base.clone();
                    path.push(here);
                    slot.insert(path);
                    queue.push_back(next);
                }
            }
        }
        let word = word.ok_or_else(|| {
            Error::InvalidState(format!("no complementation word reduces site {site}"))
        })?;
        for here in word {
            if here {
                self.local_complement(site)?;
            } else {
                self.local_complement(helper)?;
            }
        }
        Ok(())
    }

    /// An isolated fiber's state factorizes, so its vertex operator can
    /// be settled without any graph to complement against.
    ///
    /// Its state is the `+1` eigenstate of the operator's image of `X`.
    /// If that image is `±Z` the fiber is a computational basis state
    /// and `Some(one)` says which; otherwise the state is equatorial and
    /// the operator is replaced by the diagonal one with the same
    /// image, which denotes the same state.
    fn settle_isolated(&mut self, site: u32) -> Result<Option<bool>> {
        let s = site as usize;
        let (xp, xs) = vop::map_pauli(self.fibers[s].vop, 0, false);
        if xp == 2 {
            return Ok(Some(xs));
        }
        let diagonal = (0..vop::count() as vop::Vop)
            .find(|&d| vop::is_diagonal(d) && vop::map_pauli(d, 0, false) == (xp, xs))
            .expect("every equatorial image is realized by a diagonal operator");
        self.fibers[s].vop = diagonal;
        Ok(None)
    }

    /// Apply `cz` to a pair: reduce both vertex operators into the
    /// diagonal subgroup, then toggle the link. Exact, and `O(deg²)`
    /// through the complementations.
    pub fn apply_cz(&mut self, a: u32, b: u32) -> Result<()> {
        self.check(a)?;
        self.check(b)?;
        if a == b {
            return Err(Error::InvalidState(format!(
                "cz needs two distinct sites; got {a} twice"
            )));
        }
        // An isolated endpoint in a basis state makes the cz classical:
        // |0⟩ leaves the partner alone, |1⟩ applies Z to it.
        let z_vop = (0..vop::count() as vop::Vop)
            .find(|&v| {
                vop::map_pauli(v, 0, false) == (0, true)
                    && vop::map_pauli(v, 2, false) == (2, false)
            })
            .expect("Z is a vertex operator");
        for (v, other) in [(a, b), (b, a)] {
            if self.links[v as usize].is_empty() {
                match self.settle_isolated(v)? {
                    Some(true) => return self.apply_vop(other, z_vop),
                    Some(false) => return Ok(()),
                    None => {}
                }
            }
        }
        // Reducing one endpoint can disturb the other when they are
        // each other's only neighbour, so alternate until both settle.
        for _ in 0..8 {
            if vop::is_diagonal(self.fibers[a as usize].vop)
                && vop::is_diagonal(self.fibers[b as usize].vop)
            {
                break;
            }
            self.reduce_vop(b, a)?;
            self.reduce_vop(a, b)?;
        }
        if !vop::is_diagonal(self.fibers[a as usize].vop)
            || !vop::is_diagonal(self.fibers[b as usize].vop)
        {
            // Alternating reduction can cycle when the two endpoints
            // are each other's only handle. Search complementation
            // words at the pair directly, rolling back what does not
            // land, rather than refusing something that is reachable.
            let saved = self.clone();
            let mut landed = false;
            // Complementing anywhere in the pair's immediate
            // neighbourhood can move their operators; that is the whole
            // set of handles available.
            let mut sites = vec![a, b];
            for &v in [a, b].iter() {
                for &w in &saved.links[v as usize] {
                    if !sites.contains(&w) {
                        sites.push(w);
                    }
                }
            }
            let base = sites.len() as u32;
            'search: for length in 1..=4usize {
                for word in 0..base.pow(length as u32) {
                    let mut trial = saved.clone();
                    let mut ok = true;
                    let mut code = word;
                    for _ in 0..length {
                        let site = sites[(code % base) as usize];
                        code /= base;
                        if trial.local_complement(site).is_err() {
                            ok = false;
                            break;
                        }
                    }
                    if ok
                        && vop::is_diagonal(trial.fibers[a as usize].vop)
                        && vop::is_diagonal(trial.fibers[b as usize].vop)
                    {
                        *self = trial;
                        landed = true;
                        break 'search;
                    }
                }
            }
            if !landed {
                *self = saved;
                return Err(Error::InvalidState(format!(
                    "could not reduce the vertex operators at {a} and {b} into the diagonal \
                     subgroup; cz is not expressible on this description"
                )));
            }
        }
        if self.linked(a, b) {
            self.unlink(a, b)?;
        } else {
            self.link(a, b)?;
        }
        Ok(())
    }
}

/// What an incoming gate matrix was recognized as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Recognized {
    Local(vop::Vop),
    Cz,
    Cx,
    Swap,
}

/// Eighth roots of unity — the phases a Clifford matrix can carry.
fn clifford_phases() -> [C64; 8] {
    std::array::from_fn(|k| {
        let angle = std::f64::consts::FRAC_PI_4 * k as f64;
        C64::new(angle.cos(), angle.sin())
    })
}

fn matches<S: Scalar>(m: &GateMatrix<S>, target: &[C64], tol: f64) -> bool {
    clifford_phases().iter().any(|&phase| {
        target.iter().enumerate().all(|(k, &z)| {
            S::try_from_c64(z * phase).is_some_and(|want| m.data()[k].approx_eq(want, tol))
        })
    })
}

fn recognize<S: Scalar>(m: &GateMatrix<S>) -> Option<Recognized> {
    let tol = 1e-9;
    match m.dim() {
        2 => (0..vop::count() as vop::Vop).find_map(|v| {
            let target = vop::matrix::<C64>(v).expect("ℂ carries every vertex operator");
            matches(m, &target, tol).then_some(Recognized::Local(v))
        }),
        4 => {
            let o = C64::new(1.0, 0.0);
            let l = C64::new(0.0, 0.0);
            #[rustfmt::skip]
            let cz = [o, l, l, l,  l, o, l, l,  l, l, o, l,  l, l, l, -o];
            #[rustfmt::skip]
            let cx = [o, l, l, l,  l, l, l, o,  l, l, o, l,  l, o, l, l];
            #[rustfmt::skip]
            let swap = [o, l, l, l,  l, l, o, l,  l, o, l, l,  l, l, l, o];
            [
                (cz, Recognized::Cz),
                (cx, Recognized::Cx),
                (swap, Recognized::Swap),
            ]
            .into_iter()
            .find_map(|(t, r)| matches(m, &t, tol).then_some(r))
        }
        _ => None,
    }
}

impl PolarityBundle {
    /// Amplitudes of the denoted state, built from the description.
    ///
    /// The description is `O(n + |E|)`; there are `2^n` amplitudes, so
    /// *reading them out* is exponential while *holding the state* is
    /// not. Everything below the graph part costs `O(|E| + n·2^n)`.
    fn materialize<S: Scalar>(&self) -> Result<Vec<S>> {
        let n = self.fibers.len();
        if n > BUNDLE_ENUMERATION_MAX {
            return Err(Error::TooManyQubits {
                requested: n,
                max: BUNDLE_ENUMERATION_MAX,
            });
        }
        if self.fibers.iter().any(|f| !f.live) {
            return Err(Error::InvalidState(
                "a coarse-grained bundle no longer denotes a state on its original sites".into(),
            ));
        }
        let dim = 1usize << n;
        let amp = S::from_re(1.0 / (dim as f64).sqrt());
        let mut out = vec![amp; dim];
        for (y, slot) in out.iter_mut().enumerate() {
            let mut negative = false;
            for (a, neighbours) in self.links.iter().enumerate() {
                if (y >> a) & 1 == 0 {
                    continue;
                }
                for &b in neighbours {
                    if (a as u32) < b && (y >> b) & 1 == 1 {
                        negative = !negative;
                    }
                }
            }
            for (q, fiber) in self.fibers.iter().enumerate() {
                if fiber.spin && (y >> q) & 1 == 1 {
                    negative = !negative;
                }
            }
            if negative {
                *slot = -*slot;
            }
        }
        for (q, fiber) in self.fibers.iter().enumerate() {
            if fiber.vop == vop::IDENTITY {
                continue;
            }
            let entries = vop::matrix::<S>(fiber.vop).ok_or(Error::UnsupportedForAlgebra {
                gate: "vertex operator".to_string(),
                algebra: S::algebra_name(),
            })?;
            let m = GateMatrix::from_vec(2, entries.to_vec())?;
            crate::backend::apply_single_in_place(&mut out, &m, q)?;
        }
        Ok(out)
    }
}

impl<S: Scalar> Backend<S> for PolarityBundle {
    fn name(&self) -> &str {
        "bundle"
    }

    fn num_qubits(&self) -> usize {
        self.fibers.len()
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        crate::backend::validate_apply(self.fibers.len(), matrix, qubits)?;
        match recognize(matrix) {
            Some(Recognized::Local(v)) => self.apply_vop(qubits[0] as u32, v),
            Some(Recognized::Cz) => self.apply_cz(qubits[0] as u32, qubits[1] as u32),
            Some(Recognized::Cx) => {
                let (c, t) = (qubits[0] as u32, qubits[1] as u32);
                self.apply_vop(t, vop::hadamard())?;
                self.apply_cz(c, t)?;
                self.apply_vop(t, vop::hadamard())
            }
            Some(Recognized::Swap) => {
                let (a, b) = (qubits[0] as u32, qubits[1] as u32);
                for (c, t) in [(a, b), (b, a), (a, b)] {
                    self.apply_vop(t, vop::hadamard())?;
                    self.apply_cz(c, t)?;
                    self.apply_vop(t, vop::hadamard())?;
                }
                Ok(())
            }
            None => Err(Error::UnsupportedForAlgebra {
                gate: format!("non-Clifford {}-qubit gate", qubits.len()),
                algebra: "graph-state bundle".to_string(),
            }),
        }
    }

    fn amplitude(&self, index: u64) -> S {
        self.materialize::<S>()
            .ok()
            .and_then(|v| v.get(index as usize).copied())
            .unwrap_or_else(S::zero)
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        if let Ok(amps) = self.materialize::<S>() {
            for (i, a) in amps.into_iter().enumerate() {
                if !a.is_zero(1e-15) {
                    f(i as u64, a);
                }
            }
        }
    }

    fn project(&mut self, _qubit: usize, _outcome: bool, _renorm: f64) {
        // Measurement keeps a graph state in its class, but the update
        // is a different algorithm from gate action and is not written
        // yet. Silently collapsing to something wrong would be worse
        // than leaving the state alone and letting `measure` fail.
    }

    fn reset(&mut self) {
        let n = self.fibers.len();
        for site in 0..n {
            self.links[site].clear();
            self.fibers[site] = Fiber::default();
            // |0…0⟩ is |+…+⟩ with a Hadamard on every site.
            self.fibers[site].vop = vop::hadamard();
        }
        self.link_count = 0;
        self.journal.clear();
    }

    fn load(&mut self, _entries: &[(u64, S)]) -> Result<()> {
        Err(Error::InvalidState(
            "a graph-state bundle holds a description, not arbitrary amplitudes; build it \
             with link/apply_vop or run a Clifford circuit onto it"
                .into(),
        ))
    }

    fn memory_bytes(&self) -> usize {
        self.profile().bytes
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
