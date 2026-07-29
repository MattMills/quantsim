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
use crate::registry::GateRegistry;
use crate::scalar::C64;

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
    /// Local polarity axis.
    pub frame: Frame,
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
            for gate in fiber.frame.gates() {
                circuit.gate(*gate, Vec::new(), vec![q]);
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
            // F X F†: the Z frame is the identity so X stays X; both
            // rotated frames contain `h`, which sends X to Z.
            let axis = match self.fibers[q].frame {
                Frame::Z => Pauli::X,
                Frame::X | Frame::Y => Pauli::Z,
            };
            let mut ops = vec![(q, axis)];
            for &b in &self.links[q] {
                let nb = b as usize;
                let neighbour_axis = match self.fibers[nb].frame {
                    Frame::Z => Pauli::Z,
                    Frame::X => Pauli::X,
                    Frame::Y => Pauli::Y,
                };
                ops.push((nb, neighbour_axis));
            }
            let value = pauli_expectation(state, &ops)?.re;
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
