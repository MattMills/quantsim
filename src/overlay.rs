//! The overlay as a **register**: a state whose regions are curve
//! blocks.
//!
//! [`curve`](crate::curve) measures orderings. This module *uses* one.
//! It holds amplitudes, applies gates and answers
//! [`amplitude`](OverlayRegister::amplitude) — it is a representation,
//! not a census of one — and what it is built to show is that the
//! layout is not bookkeeping around the state but a constraint **on**
//! it, with a price the register pays in memory and can be made to
//! report.
//!
//! ## The contract
//!
//! The register keeps the lattice partitioned into **regions**. Each
//! holds a dense amplitude vector over its own sites, and regions are
//! never entangled with each other: the state is their product, so
//! memory is the **sum** of `2^{sites}` over regions rather than `2^n`.
//!
//! The layout enters as a single rule:
//!
//! > A region must be a contiguous run of `2^k` chain positions under
//! > **one of the overlay's members**.
//!
//! That is the refusal. A gate whose support straddles two regions
//! cannot simply take their union, because a set of sites is not
//! something a chain-addressed register can hold. It has to
//! **migrate**: elect the smallest member-block containing them,
//! absorb every region that block cuts into, and re-index. The block
//! is generally larger than the union, and the difference —
//! [`Merge::padding`] — is what the layout charged for that gate.
//!
//! So the ordering is not advice — it is a width, and the width is a
//! property of the layout alone. [`projected`](OverlayRegister::projected)
//! reports it without allocating anything: one vertical lattice edge on
//! an 8×8 grid forces a **16-site** block under row-major and a
//! **2-site** block under the Hilbert curve, on any machine and under
//! any memory limit.
//!
//! This module owns **no capacity policy**. Whether a block fits is
//! asked of the [resource guard](crate::guard), which admits every
//! merge against the bytes *measured* available at that moment and
//! fails with [`Error::OutOfMemory`] carrying both counts. A register
//! that refused at a precomputed width would inhibit computations the
//! machine can perform — the mistake `guard` exists to have stopped
//! making — so the only width bound here is
//! [`MAX_REGION_SITES`], which is structural (local basis indices are
//! `usize`). Measured across a 256× range of budgets, the demand a
//! layout puts on the register is the same number every time; only the
//! outcome changes.
//!
//! The other half of the contract is that padding can be *borrowed*.
//! [`compact`](OverlayRegister::compact) — run automatically after
//! every gate — tests each region for a rank-1 factorisation across
//! its block's own bipartition and hands the halves back when it
//! finds one. The closure swallows regions the gate never touched;
//! those are product factors and come straight back, and with them
//! the freedom to be elected into a *different* member next time.
//! What cannot be given back is an entangled pair straddling the
//! block's cut: there the rank is two and the register is stuck
//! holding all of it, which is precisely the case the election exists
//! to avoid.
//!
//! ## What the overlay is worth, exactly
//!
//! Sum [`Placement::width`] over every lattice edge of a fresh
//! register: the cost of a nearest-neighbour layer before anything has
//! committed. Four exact closed forms, verified at every power-of-two
//! side from 4 to 64 (`s` the side, `n = s²`):
//!
//! | overlay | summed placement width | mean per edge |
//! |---|---|---|
//! | one row-major (or snake) ordering | `s²(s+1)·log₂ s` | `≈ s·log₂ s / 2` |
//! | the row-major `D₄` family | `2s²·log₂ s = n·log₂ n` | `→ log₂ s` |
//! | one Hilbert ordering | `3s²(s−1)` | exactly `1.5·s` |
//! | the Hilbert `D₄` family | `2s²(s−1)` | exactly `s` |
//!
//! Two things fall out, and the second is a reversal:
//!
//! * The **Hilbert family saves exactly one third**, at every size, and
//!   never more. Its members are rotors of a self-similar curve, so
//!   they agree on every quadrant; where several of them tie at the
//!   minimal level they name the *same block*, measured, at every size
//!   — the tie is a labelling, not a choice.
//! * The **row-major family saves a factor of `(s+1)/2`**, which grows
//!   without bound, and **two members are the whole of it**: an
//!   ordering and its transpose. Members three through eight add
//!   exactly nothing, because a lattice edge is horizontal or vertical
//!   and those two already cover both.
//!
//! So as a single ordering the curve wins and the rows lose, which is
//! the ranking [`curve`](crate::curve) measures. As an *overlay* the
//! ranking inverts: `n log₂ n` against `2s²(s−1)`, and the gap widens
//! at every size. The overlay's value is the **disagreement** between
//! its members, and a self-similar family agrees with itself too much
//! to have much of it.
//!
//! ## Which is why the register mixes families
//!
//! [`Overlay::families`] puts the `D₄` families of several orderings
//! into one overlay, and that register is strictly the best of both:
//!
//! * it ties the row-major family on placement, which is the best
//!   there is, and beats the Hilbert family outright;
//! * on a run — a graph state that is row-local on half the lattice
//!   and patch-local on the other half, which is the case no single
//!   family serves — it holds less than either: at side 16, padding
//!   120 against the Hilbert family's 168 and a single Hilbert
//!   ordering's 186, peak memory **8 396 800 B** against 8 454 144 and
//!   12 615 680.
//!
//! And the row-major family is not in that comparison because it
//! cannot be run at all: on a graph state over 4×4 patches it reaches
//! a `cz` that demands a **32-site** region — 2³² amplitudes,
//! **68 719 476 736 bytes** — for a two-site gate, while the curve
//! runs the same circuit in **4 MiB**. That is a ratio between two
//! measured demands, not a verdict from a threshold.
//!
//! That is the [mosaic register](crate::backend::MosaicState)'s
//! contract on the
//! layout axis: heterogeneous representations in one register, elected
//! per operation against a measured cost, and every migration
//! ledgered.
//!
//! ## What is not claimed
//!
//! The election is a per-migration minimum, not per-region freedom.
//! Once a region commits to a block, the next migration must contain
//! *that whole block*, so on a run the overlay cannot beat the best
//! single family on peak width — only on padding and on total memory,
//! and only because the mix contains a family that suits each part of
//! the circuit. In particular the row-major family's `(s+1)/2`
//! placement advantage does **not** survive into a run: it demands the
//! same 32-site region the single ordering does. Electing among tied
//! contenders by which one cuts into fewest regions
//! ([`OverlayRegister::contenders`]) is implemented and is the right
//! rule, but has not yet changed a measured outcome.
//!
//! ## Conventions
//!
//! * A **site** is `y · side + x`, the lattice's own labelling, the
//!   same one [`GridOrder::edges_by_site`](crate::curve::GridOrder::edges_by_site)
//!   uses.
//! * Global amplitude index bit `s` is site `s`, so
//!   [`amplitude`](OverlayRegister::amplitude) is directly comparable
//!   with a dense backend of the same width — which is how the tests
//!   establish that this holds the state it claims to.
//! * Inside a region, local bit `b` is `sites[b]`, and `sites` is in
//!   the member's chain order.
//! * Regions multiply in order of their smallest site, matching
//!   [`FactoredState`](crate::backend::FactoredState). Only relevant
//!   over non-commutative scalars, where the split is also disabled.

use crate::backend::{apply_general_in_place, apply_single_in_place};
use crate::curve::{GridOrder, Order, Overlay};
use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::scalar::Scalar;

/// Structural ceiling on the sites one region may hold: local basis
/// indices are `usize`, the same bound
/// [`FACTOR_MAX_QUBITS`](crate::backend::FACTOR_MAX_QUBITS) puts on a
/// factor.
///
/// This is **not** a capacity policy. Whether a region fits is decided
/// by the [resource guard](crate::guard), which admits every merge
/// against the memory *measured* available at that moment and fails
/// with [`Error::OutOfMemory`] carrying both byte counts. A register
/// that refused at a precomputed width would inhibit computations the
/// machine can perform, which is the mistake `guard` exists to have
/// stopped making.
pub const MAX_REGION_SITES: usize = 63;

/// Default rank-1 residual below which a region is judged to factor.
///
/// The same order as [`FactoredState`](crate::backend::FactoredState)'s
/// split tolerance, and for the same reason: a split is exact when the
/// state really is a product and introduces error of this size when it
/// is merely close to one.
pub const DEFAULT_SPLIT_TOLERANCE: f64 = 1e-11;

/// Where a set of sites can live: one member's block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    /// Index of the overlay member whose ordering this block belongs to.
    pub member: usize,
    /// Block level: the block spans `2^level` chain positions.
    pub level: u32,
    /// Which block at that level, counting from the chain's start.
    pub position: usize,
    /// How many members achieve this same level — `1` means the
    /// election was decided, more means it was a tie.
    pub contenders: usize,
}

impl Placement {
    /// Sites the block holds.
    pub fn width(&self) -> usize {
        1usize << self.level
    }

    /// Amplitudes a region filling this block would cost, saturating.
    ///
    /// A block can name more sites than any machine can hold — the
    /// whole lattice is always a block — so this saturates rather than
    /// overflowing. [`OverlayRegister`] refuses such a block before it
    /// is ever allocated.
    pub fn amplitudes(&self) -> u128 {
        if self.width() >= 128 {
            u128::MAX
        } else {
            1u128 << self.width()
        }
    }
}

/// One migration, recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Merge {
    /// The block the register migrated into.
    pub placement: Placement,
    /// Sites the gate's own regions held — what it actually needed.
    pub demand: usize,
    /// Regions the block cut into and therefore had to swallow.
    pub absorbed: usize,
    /// Whether the elected member was not member `0` — i.e. the
    /// overlay changed the answer.
    pub elected: bool,
}

impl Merge {
    /// Sites the block holds.
    pub fn width(&self) -> usize {
        self.placement.width()
    }

    /// Sites the layout charged beyond what the gate needed.
    pub fn padding(&self) -> usize {
        self.placement.width() - self.demand
    }
}

/// Every migration the register made, and the peaks it reached.
#[derive(Clone, Debug, Default)]
pub struct Ledger {
    merges: Vec<Merge>,
    peak_width: usize,
    peak_memory: u128,
    gates: usize,
    local: usize,
    splits: usize,
    reclaimed: usize,
}

impl Ledger {
    /// The migrations, in order.
    pub fn merges(&self) -> &[Merge] {
        &self.merges
    }

    /// How many migrations happened.
    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }

    /// Gates applied.
    pub fn gates(&self) -> usize {
        self.gates
    }

    /// Gates that needed no migration because their support was
    /// already inside one region.
    pub fn local_gates(&self) -> usize {
        self.local
    }

    /// Widest region the register ever held, in sites.
    pub fn peak_width(&self) -> usize {
        self.peak_width
    }

    /// Largest total amplitude count the register ever held.
    pub fn peak_memory(&self) -> u128 {
        self.peak_memory
    }

    /// Summed [`Merge::padding`] — the layout's whole bill.
    pub fn total_padding(&self) -> usize {
        self.merges.iter().map(|m| m.padding()).sum()
    }

    /// Migrations where the overlay's election chose a member other
    /// than the first — the times having several rotors mattered.
    pub fn elections(&self) -> usize {
        self.merges.iter().filter(|m| m.elected).count()
    }

    /// Splits performed: regions that turned out to factor across their
    /// block's own bipartition and were released.
    pub fn splits(&self) -> usize {
        self.splits
    }

    /// Sites of dense width released by splitting — the block the
    /// register handed back.
    ///
    /// A migration's closure swallows every region its block cuts into,
    /// including regions the gate never touched. Those are product
    /// factors, and the split hands them straight back, so much of
    /// [`total_padding`](Self::total_padding) is borrowed rather than
    /// spent. This counts what came back.
    pub fn reclaimed(&self) -> usize {
        self.reclaimed
    }

    /// Migrations that were decided rather than tied: some member
    /// reached a level no other member reached.
    pub fn decided(&self) -> usize {
        self.merges
            .iter()
            .filter(|m| m.placement.contenders == 1)
            .count()
    }
}

/// A region: one member-block, and the amplitudes over it.
#[derive(Clone, Debug)]
struct Region<S: Scalar> {
    /// Site ids in the block's chain order; local bit `b` ↔ `sites[b]`.
    sites: Vec<usize>,
    /// The block.
    placement: Placement,
    /// `2^{sites.len()}` amplitudes.
    amps: Vec<S>,
}

/// A register whose regions are blocks of an [`Overlay`].
#[derive(Clone, Debug)]
pub struct OverlayRegister<S: Scalar> {
    side: usize,
    levels: u32,
    /// Per member, `chain[site]`, and the inverse.
    chain: Vec<Vec<usize>>,
    point: Vec<Vec<usize>>,
    regions: Vec<Region<S>>,
    /// `home[site]` is the index in `regions` holding that site.
    home: Vec<usize>,
    auto_split: bool,
    split_tol: f64,
    ledger: Ledger,
}

impl<S: Scalar> OverlayRegister<S> {
    /// A register over an explicit overlay, in `|0…0⟩`.
    ///
    /// The lattice side must be a power of two: the block hierarchy is
    /// the chain's balanced binary tree, and a non-power-of-two site
    /// count has no such tree.
    pub fn new(side: usize, overlay: &Overlay) -> Result<Self> {
        if !side.is_power_of_two() || side == 0 {
            return Err(Error::InvalidState(format!(
                "overlay register: side {side} is not a power of two — the block \
                 hierarchy is the chain's balanced tree, which needs one"
            )));
        }
        let members = overlay.members();
        if members.iter().any(|m| m.side() != side) {
            return Err(Error::InvalidState(format!(
                "overlay register: members are laid on a different lattice than side {side}"
            )));
        }
        let n = side * side;
        let mut chain = Vec::with_capacity(members.len());
        let mut point = Vec::with_capacity(members.len());
        for m in members {
            let mut c = vec![0usize; n];
            let mut p = vec![0usize; n];
            for (i, slot) in p.iter_mut().enumerate() {
                let (x, y) = m.lattice_point(i)?;
                let site = y * side + x;
                c[site] = i;
                *slot = site;
            }
            chain.push(c);
            point.push(p);
        }
        let mut reg = OverlayRegister {
            side,
            levels: n.ilog2(),
            chain,
            point,
            regions: Vec::new(),
            home: vec![0; n],
            auto_split: true,
            split_tol: DEFAULT_SPLIT_TOLERANCE,
            ledger: Ledger::default(),
        };
        reg.reset();
        Ok(reg)
    }

    /// A register over a single ordering — the balanced tree over one
    /// chain, with no election to make. This is the baseline the
    /// overlay is measured against.
    pub fn single(side: usize, order: Order) -> Result<Self> {
        OverlayRegister::new(side, &Overlay::new(vec![GridOrder::new(side, order)?])?)
    }

    /// A register over the `D₄` family of an ordering: the same rule
    /// under each of the eight global rotors, elected per migration.
    pub fn family(side: usize, order: Order) -> Result<Self> {
        OverlayRegister::new(side, &Overlay::family(side, order)?)
    }

    /// A register over the `D₄` families of several orderings — a
    /// heterogeneous overlay, and the one that wins.
    ///
    /// See [`Overlay::families`].
    pub fn mixed(side: usize, orders: &[Order]) -> Result<Self> {
        OverlayRegister::new(side, &Overlay::families(side, orders)?)
    }

    /// Whether to try splitting regions back apart after every gate.
    ///
    /// On by default, and it is not a micro-optimization: the closure
    /// in a migration swallows regions the gate never touched, and the
    /// split is what gives them back — and with them the freedom to be
    /// elected into a *different* member next time. Without it a region
    /// commits to one member's block permanently, and the overlay's
    /// election is spent the first time it is used.
    pub fn set_auto_split(&mut self, enabled: bool) {
        self.auto_split = enabled;
    }

    /// Set the rank-1 residual below which a region is judged to
    /// factor. See [`DEFAULT_SPLIT_TOLERANCE`].
    pub fn set_split_tolerance(&mut self, tol: f64) {
        self.split_tol = tol;
    }

    /// Lattice side.
    pub fn side(&self) -> usize {
        self.side
    }

    /// Sites, which is also the register width in qubits.
    pub fn sites(&self) -> usize {
        self.side * self.side
    }

    /// Overlay members.
    pub fn members(&self) -> usize {
        self.chain.len()
    }

    /// Back to `|0…0⟩` with every site in its own region, and a fresh
    /// ledger.
    pub fn reset(&mut self) {
        let n = self.sites();
        self.regions = (0..n)
            .map(|site| Region {
                sites: vec![site],
                placement: Placement {
                    member: 0,
                    level: 0,
                    position: self.chain[0][site],
                    contenders: self.chain.len(),
                },
                amps: vec![S::one(), S::zero()],
            })
            .collect();
        self.home = (0..n).collect();
        self.ledger = Ledger::default();
        self.ledger.peak_width = 1;
        self.ledger.peak_memory = 2 * n as u128;
    }

    /// The current partition: each region's sites, in its own chain
    /// order. This *is* the register's entanglement geometry, the same
    /// way [`FactoredState::factors`](crate::backend::FactoredState::factors)
    /// is.
    pub fn regions(&self) -> Vec<Vec<usize>> {
        self.regions.iter().map(|r| r.sites.clone()).collect()
    }

    /// How many regions the state is currently in.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Where each region lives.
    pub fn placements(&self) -> Vec<Placement> {
        self.regions.iter().map(|r| r.placement).collect()
    }

    /// Sites in the widest region — the width a dense representation
    /// of the worst region would need.
    pub fn widest(&self) -> usize {
        self.regions
            .iter()
            .map(|r| r.sites.len())
            .max()
            .unwrap_or(0)
    }

    /// Amplitudes currently held: the **sum** over regions, not `2^n`.
    pub fn memory_amplitudes(&self) -> u128 {
        self.regions.iter().map(|r| 1u128 << r.sites.len()).sum()
    }

    /// Bytes currently held — the unit the [guard](crate::guard)
    /// admits merges in, and the one a layout comparison should be
    /// stated in.
    pub fn memory_bytes(&self) -> u128 {
        self.memory_amplitudes() * std::mem::size_of::<S>() as u128
    }

    /// Bytes at the register's high-water mark.
    pub fn peak_memory_bytes(&self) -> u128 {
        self.ledger.peak_memory * std::mem::size_of::<S>() as u128
    }

    /// The migration record.
    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// Site id of a lattice point.
    pub fn site(&self, x: usize, y: usize) -> Result<usize> {
        if x >= self.side || y >= self.side {
            return Err(Error::InvalidState(format!(
                "overlay register: ({x}, {y}) is outside a {}×{} lattice",
                self.side, self.side
            )));
        }
        Ok(y * self.side + x)
    }

    // ─────────────────────── the election ───────────────────────

    /// The smallest block of any member that contains every one of
    /// `sites` — the migration target, before closure.
    ///
    /// This is the whole of what the overlay buys: with one member the
    /// answer is forced, and with several it is a minimum over
    /// several forced answers.
    pub fn admits(&self, sites: &[usize]) -> Result<Placement> {
        for &s in sites {
            if s >= self.sites() {
                return Err(Error::QubitOutOfRange {
                    qubit: s,
                    num_qubits: self.sites(),
                });
            }
        }
        if sites.is_empty() {
            return Err(Error::InvalidState(
                "overlay register: no sites to place".into(),
            ));
        }
        for level in 0..=self.levels {
            let mut winner: Option<(usize, usize)> = None;
            let mut contenders = 0usize;
            for (m, chain) in self.chain.iter().enumerate() {
                let mut lo = usize::MAX;
                let mut hi = 0usize;
                for &s in sites {
                    let c = chain[s];
                    lo = lo.min(c);
                    hi = hi.max(c);
                }
                if (lo >> level) == (hi >> level) {
                    contenders += 1;
                    if winner.is_none() {
                        winner = Some((m, lo >> level));
                    }
                }
            }
            if let Some((member, position)) = winner {
                return Ok(Placement {
                    member,
                    level,
                    position,
                    contenders,
                });
            }
        }
        // The top block is the whole chain under every member, so this
        // is unreachable; state it rather than panicking.
        Err(Error::InvalidState(
            "overlay register: no block contains the support, which cannot happen \
             since the top block is the whole lattice"
                .into(),
        ))
    }

    /// Every member's block at the minimal level — the ties
    /// [`admits`](Self::admits) breaks by taking the first.
    ///
    /// A migration elects among these by the second criterion the
    /// register actually cares about: how many existing regions the
    /// block cuts into, since those have to be swallowed whole.
    pub fn contenders(&self, sites: &[usize]) -> Result<Vec<Placement>> {
        let best = self.admits(sites)?;
        let mut out = Vec::with_capacity(best.contenders);
        for (m, chain) in self.chain.iter().enumerate() {
            let mut lo = usize::MAX;
            let mut hi = 0usize;
            for &s in sites {
                let c = chain[s];
                lo = lo.min(c);
                hi = hi.max(c);
            }
            if (lo >> best.level) == (hi >> best.level) {
                out.push(Placement {
                    member: m,
                    level: best.level,
                    position: lo >> best.level,
                    contenders: best.contenders,
                });
            }
        }
        Ok(out)
    }

    /// The block, among [`contenders`](Self::contenders), that cuts
    /// into the fewest existing regions.
    fn elect(&self, sites: &[usize]) -> Result<Placement> {
        let cands = self.contenders(sites)?;
        let mut best: Option<(usize, Placement)> = None;
        for p in cands {
            let block = self.block_sites(p);
            let mut homes: Vec<usize> = Vec::new();
            for &s in &block {
                let h = self.home[s];
                if !homes.contains(&h) {
                    homes.push(h);
                }
            }
            if best.map_or(true, |(n, _)| homes.len() < n) {
                best = Some((homes.len(), p));
            }
        }
        Ok(best.expect("admits found at least one contender").1)
    }

    /// The sites a placement's block covers, in that member's chain
    /// order.
    pub fn block_sites(&self, p: Placement) -> Vec<usize> {
        let size = 1usize << p.level;
        let start = p.position * size;
        self.point[p.member][start..start + size].to_vec()
    }

    // ─────────────────────── the migration ───────────────────────

    /// What a gate on `sites` would cost this layout, computed without
    /// allocating or changing anything.
    ///
    /// This is the layout's own signal and it needs **no budget**: the
    /// answer is a block width, not a verdict. A vertical lattice edge
    /// on an 8×8 grid projects to 16 sites under row-major and 2 under
    /// the Hilbert curve, at any memory limit and on any machine.
    /// Whether the register can then *hold* that block is a separate
    /// question, and one the [resource guard](crate::guard) answers by
    /// measuring rather than by a constant.
    ///
    /// `None` means the gate is already inside one region and costs no
    /// migration at all.
    pub fn projected(&self, sites: &[usize]) -> Result<Option<Placement>> {
        let chosen = self.homes_of(sites)?;
        if chosen.len() == 1 {
            return Ok(None);
        }
        Ok(Some(self.plan(chosen)?.0))
    }

    /// The distinct regions holding `sites`, in first-touch order.
    fn homes_of(&self, sites: &[usize]) -> Result<Vec<usize>> {
        let mut chosen: Vec<usize> = Vec::new();
        for &s in sites {
            let h = *self.home.get(s).ok_or(Error::QubitOutOfRange {
                qubit: s,
                num_qubits: self.sites(),
            })?;
            if !chosen.contains(&h) {
                chosen.push(h);
            }
        }
        Ok(chosen)
    }

    /// The closure, read-only: elect a block for the chosen regions;
    /// a block that cuts into another region has to swallow it whole,
    /// and swallowing may force a bigger block, so iterate to a fixed
    /// point. It terminates because the swallowed set only grows and
    /// the whole lattice is a block of every member.
    fn plan(&self, chosen: Vec<usize>) -> Result<(Placement, Vec<usize>, Vec<usize>)> {
        let mut chosen = chosen;
        loop {
            let mut span: Vec<usize> = Vec::new();
            for &i in &chosen {
                span.extend_from_slice(&self.regions[i].sites);
            }
            let placement = self.elect(&span)?;
            let block = self.block_sites(placement);
            let mut needed: Vec<usize> = Vec::new();
            for &s in &block {
                let h = self.home[s];
                if !needed.contains(&h) {
                    needed.push(h);
                }
            }
            if needed.len() == chosen.len() {
                return Ok((placement, block, needed));
            }
            chosen = needed;
        }
    }

    /// Bring every site of `sites` into one region, migrating as the
    /// layout requires, and return that region's index.
    fn coalesce(&mut self, sites: &[usize]) -> Result<usize> {
        let chosen = self.homes_of(sites)?;
        if chosen.len() == 1 {
            self.ledger.local += 1;
            return Ok(chosen[0]);
        }
        let demand: usize = chosen.iter().map(|&i| self.regions[i].sites.len()).sum();
        let (placement, block, chosen) = self.plan(chosen)?;

        // The only width bound here is structural — local basis indices
        // are `usize`. Whether the machine can hold the block is asked
        // of the guard below, in bytes, by measurement.
        if block.len() > MAX_REGION_SITES {
            return Err(Error::TooManyQubits {
                requested: block.len(),
                max: MAX_REGION_SITES,
            });
        }

        // Multiply the constituents into the block's ordering.
        let mut order: Vec<usize> = chosen.clone();
        order.sort_by_key(|&i| {
            *self.regions[i]
                .sites
                .iter()
                .min()
                .expect("regions are never empty")
        });
        let width = block.len();
        let mut local: Vec<Vec<usize>> = Vec::with_capacity(order.len());
        for &i in &order {
            local.push(
                self.regions[i]
                    .sites
                    .iter()
                    .map(|s| {
                        block
                            .iter()
                            .position(|b| b == s)
                            .expect("the closure guarantees every site is in the block")
                    })
                    .collect(),
            );
        }
        let mut amps = crate::guard::try_vec(
            1usize << width,
            S::zero(),
            &format!("overlay region ({width} sites)"),
        )?;
        for (idx, slot) in amps.iter_mut().enumerate() {
            let mut acc = S::one();
            for (r, &i) in order.iter().enumerate() {
                let mut sub = 0usize;
                for (b, &pos) in local[r].iter().enumerate() {
                    sub |= ((idx >> pos) & 1) << b;
                }
                acc = acc * self.regions[i].amps[sub];
            }
            *slot = acc;
        }

        // Replace the constituents with the merged region.
        let mut dead: Vec<usize> = chosen;
        dead.sort_unstable();
        let keep = dead[0];
        self.regions[keep] = Region {
            sites: block.clone(),
            placement,
            amps,
        };
        for &d in dead.iter().skip(1).rev() {
            self.regions.swap_remove(d);
        }
        self.rebuild_home();
        let home = self.home[block[0]];

        self.ledger.merges.push(Merge {
            placement,
            demand,
            absorbed: dead.len(),
            elected: placement.member != 0,
        });
        self.ledger.peak_width = self.ledger.peak_width.max(width);
        self.ledger.peak_memory = self.ledger.peak_memory.max(self.memory_amplitudes());
        Ok(home)
    }

    // ─────────────────── giving the block back ───────────────────

    /// Try to split one region across its block's own bipartition.
    ///
    /// A block of `2^k` sites is two blocks of `2^{k-1}` — the same
    /// member's children — and the region's amplitudes, reshaped over
    /// that cut, are a matrix. If the matrix has rank one the state is
    /// a product across the cut and the register has no business
    /// holding it as one region.
    ///
    /// This is [`FactoredState`](crate::backend::FactoredState)'s
    /// rank-1 test moved onto the layout's own cut, which is what makes
    /// the halves *placeable*: each is a block, so each is free to be
    /// elected into any member on its next migration.
    fn try_split(&mut self, idx: usize) -> Option<(usize, usize)> {
        if !S::COMMUTATIVE {
            return None;
        }
        let width = self.regions[idx].sites.len();
        if width < 2 {
            return None;
        }
        let h = width / 2;
        let (rows, cols) = (1usize << (width - h), 1usize << h);
        let amps = &self.regions[idx].amps;
        // Pivot row: the heaviest slice of the high half.
        let (mut pivot, mut best) = (0usize, -1.0f64);
        for r in 0..rows {
            let w: f64 = (0..cols).map(|c| amps[r * cols + c].abs_sqr()).sum();
            if w > best {
                best = w;
                pivot = r;
            }
        }
        if best <= 0.0 {
            return None;
        }
        let norm = best.sqrt();
        let low: Vec<S> = (0..cols)
            .map(|c| amps[pivot * cols + c].scale(1.0 / norm))
            .collect();
        let mut high = vec![S::zero(); rows];
        let mut residual = 0.0f64;
        for (r, slot) in high.iter_mut().enumerate() {
            let mut v = S::zero();
            for (c, &l) in low.iter().enumerate() {
                v = v + l.conj() * amps[r * cols + c];
            }
            for (c, &l) in low.iter().enumerate() {
                residual = residual.max((l * v - amps[r * cols + c]).abs_sqr());
            }
            *slot = v;
        }
        if residual.sqrt() > self.split_tol {
            return None;
        }
        let sites = self.regions[idx].sites.clone();
        let (lo_sites, hi_sites) = (sites[..h].to_vec(), sites[h..].to_vec());
        let lo_place = self.admits(&lo_sites).ok()?;
        let hi_place = self.admits(&hi_sites).ok()?;
        self.regions[idx] = Region {
            sites: lo_sites,
            placement: lo_place,
            amps: low,
        };
        self.regions.push(Region {
            sites: hi_sites,
            placement: hi_place,
            amps: high,
        });
        let hi_idx = self.regions.len() - 1;
        self.rebuild_home();
        self.ledger.splits += 1;
        self.ledger.reclaimed += h;
        Some((idx, hi_idx))
    }

    /// Split every region as far as it factors.
    fn resplit(&mut self) {
        let mut queue: Vec<usize> = (0..self.regions.len()).collect();
        while let Some(i) = queue.pop() {
            if let Some((a, b)) = self.try_split(i) {
                queue.push(a);
                queue.push(b);
            }
        }
    }

    /// Split every region as far as it factors, and report how many
    /// splits it took. Called automatically after each gate unless
    /// [`set_auto_split`](Self::set_auto_split) turned it off.
    pub fn compact(&mut self) -> usize {
        let before = self.ledger.splits;
        self.resplit();
        self.ledger.splits - before
    }

    fn rebuild_home(&mut self) {
        for (i, r) in self.regions.iter().enumerate() {
            for &s in &r.sites {
                self.home[s] = i;
            }
        }
    }

    /// Apply a gate to lattice **sites**, migrating first if the
    /// support straddles regions.
    ///
    /// `sites[j]` is the gate matrix's qubit `j`.
    ///
    /// The migration allocates through the [resource
    /// guard](crate::guard), so a block the machine cannot hold fails
    /// with [`Error::OutOfMemory`] carrying the bytes needed and the
    /// bytes measured available — never against a width constant.
    /// Nothing is mutated before that allocation succeeds, so a
    /// refused gate leaves the register exactly as it was and
    /// [`projected`](Self::projected) still reports the block it
    /// wanted.
    pub fn apply(&mut self, matrix: &GateMatrix<S>, sites: &[usize]) -> Result<()> {
        crate::circuit::validate_targets(self.sites(), sites)?;
        let expected = 1usize
            .checked_shl(sites.len() as u32)
            .ok_or(Error::TooManyQubits {
                requested: sites.len(),
                max: 63,
            })?;
        if matrix.dim() != expected {
            return Err(Error::BadDimension {
                expected,
                got: matrix.dim(),
            });
        }
        let home = self.coalesce(sites)?;
        let region = &mut self.regions[home];
        let local: Vec<usize> = sites
            .iter()
            .map(|s| {
                region
                    .sites
                    .iter()
                    .position(|t| t == s)
                    .expect("coalesce put every site in this region")
            })
            .collect();
        if local.len() == 1 {
            apply_single_in_place(&mut region.amps, matrix, local[0])?;
        } else {
            apply_general_in_place(&mut region.amps, matrix, &local)?;
        }
        self.ledger.gates += 1;
        if self.auto_split {
            self.resplit();
        }
        Ok(())
    }

    /// [`apply`](Self::apply) addressed by lattice points.
    pub fn apply_at(&mut self, matrix: &GateMatrix<S>, points: &[(usize, usize)]) -> Result<()> {
        let sites = points
            .iter()
            .map(|&(x, y)| self.site(x, y))
            .collect::<Result<Vec<_>>>()?;
        self.apply(matrix, &sites)
    }

    /// The amplitude of a global basis state, index bit `s` ↔ site `s`.
    ///
    /// The state is the product over regions, so this is a product of
    /// their local amplitudes — which is why it is directly comparable
    /// with a dense backend of the same width.
    pub fn amplitude(&self, index: u64) -> S {
        let mut order: Vec<usize> = (0..self.regions.len()).collect();
        order.sort_by_key(|&i| {
            *self.regions[i]
                .sites
                .iter()
                .min()
                .expect("regions are never empty")
        });
        let mut acc = S::one();
        for i in order {
            let r = &self.regions[i];
            let mut sub = 0usize;
            for (b, &s) in r.sites.iter().enumerate() {
                sub |= (((index >> s) & 1) as usize) << b;
            }
            acc = acc * r.amps[sub];
        }
        acc
    }

    /// Total Born weight, which a correct run conserves over the
    /// division algebras.
    pub fn total_weight(&self) -> f64 {
        self.regions
            .iter()
            .map(|r| r.amps.iter().map(|a| a.born_weight()).sum::<f64>())
            .product()
    }
}
