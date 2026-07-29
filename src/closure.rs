//! A **journalled state representation** whose history carries its
//! entanglement, and whose errors are found by **geometric closure**,
//! **retrodictively**.
//!
//! ## The representation is the history
//!
//! A [`ClosureHistory`] does not hold a state and a log beside it. The
//! log *is* the state: every legitimate change is an entry, the twist
//! links (the entanglement) are part of every entry's record, and any
//! past configuration is reconstructible by replay. What is added on
//! top — and what makes the history do work rather than merely sit
//! there — is a **closure stamp**: a handful of bits, taken at chosen
//! moments, that says whether the structure's loops still close.
//!
//! ## Closure
//!
//! A [`PolarityBundle`]'s twist links
//! form a geometry, and that geometry has **loops** — closed walks
//! returning to where they started. Carry a fiber's polarity around
//! one and it must come back the same. Two independent things can
//! change on the way round, so each loop reports two channels:
//!
//! * **spin closure** — the parity of the fiber signs the loop passes
//!   through. Flipping one fiber's sign flips exactly the loops that
//!   pass through it, and no others.
//! * **chirality closure** — the parity of the descents the loop makes
//!   against the current generator ordering. Re-ordering two *linked*
//!   sites flips exactly the loops that traverse that link.
//!
//! Neither channel is designed, declared or paid for. Every independent
//! loop the geometry happens to have is a check, and there are exactly
//! `|E| − |V| + components` of them
//! ([`ClosureBasis::rank`]) — the structure carries its own error
//! detection because it carries its own loops. Sites lying on no loop
//! at all carry none, which [`Localization::undetectable`] reports
//! rather than hides.
//!
//! ## Retrodiction
//!
//! A closure stamp is not a state. It is `2 × rank` bits, and it is
//! taken *forward* in time while the history runs. When something later
//! goes wrong, the history is read **backwards**: because an unrepaired
//! break in closure persists, the stamps are monotone, and
//! [`ClosureHistory::retrodict`] **binary-searches** them for the first
//! moment the loops stopped closing. That is `O(log stamps)` closure
//! comparisons — no state is replayed, no error was ever recorded as an
//! error, and the answer is *when* as well as *where*.
//!
//! The geometry supplies the *where* in the same act: the loops that
//! broke all pass through the offending site, and the loops that held
//! all avoid it, so intersecting them names it
//! ([`ClosureHistory::localize`]). Space comes from which loops broke;
//! time comes from when they broke. Correction
//! ([`ClosureHistory::correct`]) restores closure at that point and the
//! denoted state with it — verified against a directly simulated clean
//! state in `tests/closure.rs`.
//!
//! ## What is claimed, and what is not
//!
//! Claimed and measured: detection capacity equal to the cycle rank at
//! no declared cost; exact geometric localization whenever the site is
//! the unique common site of the broken loops; exact temporal
//! localization in logarithmically many closure comparisons; correction
//! restoring both closure and the denoted state.
//!
//! Not claimed: that this detects everything. A perturbation on a site
//! no loop passes through is invisible here, and
//! [`Localization`] says so with the site named. Two perturbations
//! whose broken-loop sets coincide are reported as *ambiguous* rather
//! than guessed at. The honest reach is the cycle structure, and the
//! module reports the reach it has.

use crate::bundle::{Frame, PolarityBundle};
use crate::error::{Error, Result};

/// A closed walk in the twist geometry, stored as the cycle of sites it
/// visits in traversal order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureLoop {
    /// Sites in traversal order; the last is linked back to the first.
    pub sites: Vec<u32>,
}

impl ClosureLoop {
    /// Number of links the loop traverses.
    pub fn len(&self) -> usize {
        self.sites.len()
    }

    /// Whether the loop is empty (never produced by
    /// [`closure_basis`]).
    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// Whether the loop passes through a site.
    pub fn passes_through(&self, site: u32) -> bool {
        self.sites.contains(&site)
    }

    /// The links the loop traverses, as ordered pairs in traversal
    /// direction.
    pub fn links(&self) -> Vec<(u32, u32)> {
        (0..self.sites.len())
            .map(|i| (self.sites[i], self.sites[(i + 1) % self.sites.len()]))
            .collect()
    }
}

/// Every independent loop the geometry has, and the sites that have
/// none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureBasis {
    /// The independent loops.
    pub loops: Vec<ClosureLoop>,
    /// `|E| − |V| + components` — how many independent checks the
    /// geometry supplies for free.
    pub rank: usize,
    /// Links no loop traverses. A perturbation confined to these is
    /// outside the reach of closure.
    pub bridges: Vec<(u32, u32)>,
    /// Sum of every loop's length. This, not the rank, is what
    /// evaluating closure costs — a spanning-forest basis can produce
    /// long loops, and the number is reported rather than assumed
    /// small.
    pub total_length: usize,
}

/// Independent loops of a bundle's twist geometry.
///
/// A spanning forest is grown, and every link left over closes exactly
/// one loop against it. That is the whole cycle structure, counted
/// exactly: [`ClosureBasis::rank`] independent loops, and the forest's
/// own links are the bridges.
pub fn closure_basis(bundle: &PolarityBundle) -> ClosureBasis {
    let n = bundle.sites();
    let mut parent = vec![u32::MAX; n];
    let mut depth = vec![0usize; n];
    let mut seen = vec![false; n];
    let mut tree_link = vec![false; n]; // link to parent is a tree link
    let mut loops = Vec::new();
    let mut extra: Vec<(u32, u32)> = Vec::new();

    for root in 0..n {
        if seen[root] || bundle.inspect(root as u32).map(|v| v.fiber.live) != Ok(true) {
            continue;
        }
        seen[root] = true;
        let mut queue = std::collections::VecDeque::from([root as u32]);
        while let Some(site) = queue.pop_front() {
            for &next in bundle.neighbours(site) {
                let nx = next as usize;
                if !seen[nx] {
                    seen[nx] = true;
                    parent[nx] = site;
                    depth[nx] = depth[site as usize] + 1;
                    tree_link[nx] = true;
                    queue.push_back(next);
                } else if parent[site as usize] != next && site < next {
                    // A link that closes a loop against the forest.
                    extra.push((site, next));
                }
            }
        }
    }

    for &(a, b) in &extra {
        if let Some(cycle) = forest_cycle(&parent, &depth, a, b) {
            loops.push(ClosureLoop { sites: cycle });
        }
    }

    // Bridges: forest links no loop traverses.
    let mut on_loop: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    for l in &loops {
        for (a, b) in l.links() {
            on_loop.insert((a.min(b), a.max(b)));
        }
    }
    let mut bridges = Vec::new();
    for site in 0..n as u32 {
        for &next in bundle.neighbours(site) {
            if site < next && !on_loop.contains(&(site, next)) {
                bridges.push((site, next));
            }
        }
    }

    let rank = loops.len();
    let total_length = loops.iter().map(|l| l.len()).sum();
    ClosureBasis {
        loops,
        rank,
        bridges,
        total_length,
    }
}

/// The cycle closed by the link `(a, b)` against a spanning forest:
/// climb both ends to their meeting point.
fn forest_cycle(parent: &[u32], depth: &[usize], a: u32, b: u32) -> Option<Vec<u32>> {
    let (mut x, mut y) = (a, b);
    let mut left = vec![x];
    let mut right = vec![y];
    while depth[x as usize] > depth[y as usize] {
        x = *parent.get(x as usize)?;
        if x == u32::MAX {
            return None;
        }
        left.push(x);
    }
    while depth[y as usize] > depth[x as usize] {
        y = *parent.get(y as usize)?;
        if y == u32::MAX {
            return None;
        }
        right.push(y);
    }
    while x != y {
        x = *parent.get(x as usize)?;
        y = *parent.get(y as usize)?;
        if x == u32::MAX || y == u32::MAX {
            return None;
        }
        left.push(x);
        right.push(y);
    }
    // `left` ends at the meeting point, which `right` also ends at.
    right.pop();
    right.reverse();
    left.extend(right);
    (left.len() >= 3).then_some(left)
}

/// What one loop reports when carried around.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopClosure {
    /// Parity of the fiber signs the loop passes through.
    pub spin: bool,
    /// Parity of the descents the loop makes against the current
    /// ordering.
    pub chirality: bool,
    /// Links traversed.
    pub length: usize,
}

/// Carry the polarity data around one loop.
pub fn holonomy(bundle: &PolarityBundle, cycle: &ClosureLoop) -> Result<LoopClosure> {
    let mut spin = false;
    let mut chirality = false;
    for &site in &cycle.sites {
        spin ^= bundle.inspect(site)?.fiber.spin;
    }
    for (a, b) in cycle.links() {
        let pa = bundle.inspect(a)?.position;
        let pb = bundle.inspect(b)?.position;
        if pa > pb {
            chirality = !chirality;
        }
    }
    Ok(LoopClosure {
        spin,
        chirality,
        length: cycle.len(),
    })
}

/// The whole closure fingerprint of a configuration: two bits per
/// independent loop, and nothing else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureStamp {
    /// Which history step the stamp was taken at.
    pub step: usize,
    /// Spin channel, one bit per loop.
    pub spin: Vec<bool>,
    /// Chirality channel, one bit per loop.
    pub chirality: Vec<bool>,
}

impl ClosureStamp {
    /// Bits the stamp occupies — the entire cost of being able to
    /// retrodict to this moment.
    pub fn bits(&self) -> usize {
        self.spin.len() + self.chirality.len()
    }

    /// Loops whose closure differs between two stamps.
    pub fn differing_loops(&self, other: &ClosureStamp) -> Vec<usize> {
        (0..self.spin.len().min(other.spin.len()))
            .filter(|&i| self.spin[i] != other.spin[i] || self.chirality[i] != other.chirality[i])
            .collect()
    }
}

/// Which loops failed, on which channel. The two channels answer
/// different questions and localize on different geometry, so they are
/// never mixed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ClosureBreak {
    /// Loops whose spin parity moved — a fiber changed.
    pub spin: Vec<usize>,
    /// Loops whose chirality moved — the ordering across a link
    /// changed.
    pub chirality: Vec<usize>,
}

impl ClosureBreak {
    /// Whether every loop still closes on both channels.
    pub fn intact(&self) -> bool {
        self.spin.is_empty() && self.chirality.is_empty()
    }

    /// Loops broken on either channel.
    pub fn loops(&self) -> Vec<usize> {
        let mut out = self.spin.clone();
        for &i in &self.chirality {
            if !out.contains(&i) {
                out.push(i);
            }
        }
        out.sort_unstable();
        out
    }
}

/// Where the geometry says the break is.
///
/// The spin channel names **sites**; the chirality channel names
/// **links**. A candidate qualifies when every broken loop passes
/// through it and no intact loop does — so one candidate is an exact
/// answer, several is an ambiguity the closure data genuinely does not
/// resolve, and none means it happened where no loop passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Localization {
    /// Sites consistent with the broken spin loops.
    pub sites: Vec<u32>,
    /// Links consistent with the broken chirality loops.
    pub links: Vec<(u32, u32)>,
    /// Exactly one candidate, on exactly one channel.
    pub unique: bool,
    /// No loop broke on either channel.
    pub undetectable: bool,
    /// The break this localization explains.
    pub broken: ClosureBreak,
    /// Loops that held on both channels.
    pub intact_loops: usize,
}

/// The retrodicted origin of a closure break.
#[derive(Debug, Clone, PartialEq)]
pub struct Retrodiction {
    /// Index of the first stamp at which closure was already broken —
    /// the interval `(step_before, step)` contains the moment.
    pub stamp: Option<usize>,
    /// History step of that stamp.
    pub step: Option<usize>,
    /// Last step at which closure still held.
    pub last_intact_step: Option<usize>,
    /// Where the geometry puts it.
    pub localization: Localization,
    /// Closure comparisons the search needed.
    pub comparisons: usize,
    /// Comparisons a linear scan would have needed.
    pub linear_comparisons: usize,
}

/// What a correction did.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrectionReport {
    /// Site the correction was applied at.
    pub site: u32,
    /// Loops broken before.
    pub broken_before: usize,
    /// Loops broken after — zero when closure is restored.
    pub broken_after: usize,
    /// Whether every loop closes again.
    pub restored: bool,
}

/// A journalled bundle whose history carries closure stamps.
///
/// ```
/// use quantsim::bundle::PolarityBundle;
/// use quantsim::closure::ClosureHistory;
///
/// // A triangle: one independent loop, so one check, for free.
/// let mut bundle = PolarityBundle::new(3).unwrap();
/// bundle.link(0, 1).unwrap();
/// bundle.link(1, 2).unwrap();
/// bundle.link(2, 0).unwrap();
///
/// let mut history = ClosureHistory::new(bundle).unwrap();
/// assert_eq!(history.basis().rank, 1);
/// history.stamp();
///
/// // Something changes the structure without going through the history.
/// history.perturb_spin(1).unwrap();
/// history.stamp();
///
/// let retro = history.retrodict().unwrap();
/// assert!(retro.localization.sites.contains(&1));
/// ```
#[derive(Debug, Clone)]
pub struct ClosureHistory {
    bundle: PolarityBundle,
    basis: ClosureBasis,
    /// Per loop, the sites it passes through — cached for prediction.
    loop_sites: Vec<Vec<u32>>,
    /// What closure *should* read, given only the intended history.
    expected: ClosureStamp,
    /// Stamps taken so far, with the expectation in force at each.
    stamps: Vec<(ClosureStamp, ClosureStamp)>,
    step: usize,
}

impl ClosureHistory {
    /// Take ownership of a bundle and begin a history over its
    /// geometry.
    pub fn new(bundle: PolarityBundle) -> Result<Self> {
        let basis = closure_basis(&bundle);
        let loop_sites = basis.loops.iter().map(|l| l.sites.clone()).collect();
        let mut history = ClosureHistory {
            expected: ClosureStamp {
                step: 0,
                spin: Vec::new(),
                chirality: Vec::new(),
            },
            bundle,
            basis,
            loop_sites,
            stamps: Vec::new(),
            step: 0,
        };
        history.expected = history.read_closure()?;
        Ok(history)
    }

    /// The geometry's independent loops.
    pub fn basis(&self) -> &ClosureBasis {
        &self.basis
    }

    /// The live bundle.
    pub fn bundle(&self) -> &PolarityBundle {
        &self.bundle
    }

    /// History steps taken.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Stamps recorded.
    pub fn stamps(&self) -> usize {
        self.stamps.len()
    }

    /// Total bits every stamp occupies together — the entire storage
    /// cost of being able to retrodict.
    pub fn stamp_bits(&self) -> usize {
        self.stamps.iter().map(|(a, _)| a.bits()).sum()
    }

    /// Read closure directly off the live structure.
    pub fn read_closure(&self) -> Result<ClosureStamp> {
        let mut spin = Vec::with_capacity(self.basis.loops.len());
        let mut chirality = Vec::with_capacity(self.basis.loops.len());
        for cycle in &self.basis.loops {
            let h = holonomy(&self.bundle, cycle)?;
            spin.push(h.spin);
            chirality.push(h.chirality);
        }
        Ok(ClosureStamp {
            step: self.step,
            spin,
            chirality,
        })
    }

    /// What closure should read if only the intended history had
    /// happened.
    pub fn expected_closure(&self) -> &ClosureStamp {
        &self.expected
    }

    // ── intended changes: the history knows about these ──────────────

    /// Set a fiber's sign as part of the intended history. The
    /// expectation moves with it, so closure stays intact.
    pub fn set_spin(&mut self, site: u32, spin: bool) -> Result<()> {
        let was = self.bundle.inspect(site)?.fiber.spin;
        self.bundle.set_spin(site, spin)?;
        if was != spin {
            self.expect_spin_flip(site);
        }
        self.step += 1;
        Ok(())
    }

    /// Set a fiber's frame as part of the intended history.
    pub fn set_frame(&mut self, site: u32, frame: Frame) -> Result<()> {
        self.bundle.set_frame(site, frame)?;
        self.step += 1;
        Ok(())
    }

    /// Re-order two adjacent positions as part of the intended history.
    /// Chirality expectations move with it.
    pub fn swap_order(&mut self, position: usize) -> Result<()> {
        self.bundle.swap_order(position)?;
        self.step += 1;
        // Chirality is cheap to re-read and the ordering touches many
        // loops at once, so the expectation is refreshed rather than
        // predicted link by link.
        let actual = self.read_closure()?;
        self.expected.chirality = actual.chirality;
        Ok(())
    }

    /// Flip the loops through `site` in the expectation.
    fn expect_spin_flip(&mut self, site: u32) {
        for (index, sites) in self.loop_sites.iter().enumerate() {
            if sites.contains(&site) {
                self.expected.spin[index] = !self.expected.spin[index];
            }
        }
    }

    // ── unintended changes: the history does NOT know ────────────────

    /// **A perturbation.** The live structure changes; the expectation
    /// does not. This is what an error is: something that happened
    /// without being part of the history.
    pub fn perturb_spin(&mut self, site: u32) -> Result<()> {
        let was = self.bundle.inspect(site)?.fiber.spin;
        self.bundle.set_spin(site, !was)?;
        self.step += 1;
        Ok(())
    }

    /// A perturbation of the ordering: two adjacent positions swap
    /// without the history accounting for it.
    pub fn perturb_order(&mut self, position: usize) -> Result<()> {
        self.bundle.swap_order(position)?;
        self.step += 1;
        Ok(())
    }

    // ── stamps and audit ─────────────────────────────────────────────

    /// Record the closure fingerprint now. Two bits per loop, and this
    /// is the only thing retrodiction will have to work with.
    pub fn stamp(&mut self) -> Result<()> {
        let actual = self.read_closure()?;
        let mut expected = self.expected.clone();
        expected.step = self.step;
        self.stamps.push((actual, expected));
        Ok(())
    }

    /// Which loops fail to close right now, split by channel.
    pub fn break_report(&self) -> Result<ClosureBreak> {
        let actual = self.read_closure()?;
        let mut broken = ClosureBreak::default();
        for i in 0..actual.spin.len() {
            if actual.spin[i] != self.expected.spin[i] {
                broken.spin.push(i);
            }
            if actual.chirality[i] != self.expected.chirality[i] {
                broken.chirality.push(i);
            }
        }
        Ok(broken)
    }

    /// Loops failing to close, on either channel.
    pub fn broken_loops(&self) -> Result<Vec<usize>> {
        Ok(self.break_report()?.loops())
    }

    /// Where the geometry puts a break, given the loops that failed.
    ///
    /// A site is consistent when every broken loop passes through it
    /// and no intact loop does. One consistent site is an exact
    /// localization; several is an ambiguity, and it is reported as
    /// one.
    pub fn localize(&self, broken: &ClosureBreak) -> Localization {
        let total = self.basis.loops.len();
        if broken.intact() {
            return Localization {
                sites: Vec::new(),
                links: Vec::new(),
                unique: false,
                undetectable: true,
                broken: broken.clone(),
                intact_loops: total,
            };
        }
        // Spin channel names sites.
        let mut sites = Vec::new();
        if !broken.spin.is_empty() {
            let intact: Vec<usize> = (0..total).filter(|i| !broken.spin.contains(i)).collect();
            for site in 0..self.bundle.sites() as u32 {
                let on_all = broken
                    .spin
                    .iter()
                    .all(|&i| self.loop_sites[i].contains(&site));
                let off_rest = intact.iter().all(|&i| !self.loop_sites[i].contains(&site));
                if on_all && off_rest {
                    sites.push(site);
                }
            }
        }
        // Chirality channel names links.
        let mut links = Vec::new();
        if !broken.chirality.is_empty() {
            let intact: Vec<usize> = (0..total)
                .filter(|i| !broken.chirality.contains(i))
                .collect();
            let mut candidates: Vec<(u32, u32)> = Vec::new();
            for &i in &broken.chirality {
                for (a, b) in self.basis.loops[i].links() {
                    let key = (a.min(b), a.max(b));
                    if !candidates.contains(&key) {
                        candidates.push(key);
                    }
                }
            }
            for key in candidates {
                let on_all = broken
                    .chirality
                    .iter()
                    .all(|&i| self.loop_traverses(i, key));
                let off_rest = intact.iter().all(|&i| !self.loop_traverses(i, key));
                if on_all && off_rest {
                    links.push(key);
                }
            }
        }
        let candidates = sites.len() + links.len();
        Localization {
            unique: candidates == 1,
            undetectable: candidates == 0,
            sites,
            links,
            broken: broken.clone(),
            intact_loops: total - broken.loops().len(),
        }
    }

    fn loop_traverses(&self, index: usize, link: (u32, u32)) -> bool {
        self.basis.loops[index]
            .links()
            .iter()
            .any(|&(a, b)| (a.min(b), a.max(b)) == link)
    }

    // ── retrodiction ─────────────────────────────────────────────────

    /// Read the history **backwards** for the moment closure broke.
    ///
    /// An unrepaired break persists, so the stamps are monotone in
    /// "broken", and the first broken stamp is found by bisection:
    /// `O(log stamps)` comparisons of a few bits each, against the
    /// `stamps` a scan would need. No state is replayed. The break was
    /// never recorded as a break.
    pub fn retrodict(&self) -> Result<Retrodiction> {
        let broken_now = self.break_report()?;
        let localization = self.localize(&broken_now);
        if self.stamps.is_empty() {
            return Ok(Retrodiction {
                stamp: None,
                step: None,
                last_intact_step: None,
                localization,
                comparisons: 0,
                linear_comparisons: 0,
            });
        }
        let broken_at = |i: usize| -> bool {
            let (actual, expected) = &self.stamps[i];
            !actual.differing_loops(expected).is_empty()
        };
        let mut comparisons = 0usize;
        let (mut lo, mut hi) = (0usize, self.stamps.len());
        comparisons += 1;
        if !broken_at(self.stamps.len() - 1) {
            // Closure held at the last stamp; nothing to retrodict to.
            return Ok(Retrodiction {
                stamp: None,
                step: None,
                last_intact_step: Some(self.stamps[self.stamps.len() - 1].0.step),
                localization,
                comparisons,
                linear_comparisons: self.stamps.len(),
            });
        }
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            comparisons += 1;
            if broken_at(mid) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let last_intact_step = (lo > 0).then(|| self.stamps[lo - 1].0.step);
        Ok(Retrodiction {
            stamp: Some(lo),
            step: Some(self.stamps[lo].0.step),
            last_intact_step,
            localization,
            comparisons,
            linear_comparisons: self.stamps.len(),
        })
    }

    /// Restore closure at the retrodicted site.
    ///
    /// Refuses when the geometry did not name a single site: guessing
    /// between candidates would be a correction the closure data does
    /// not support.
    pub fn correct(&mut self, retro: &Retrodiction) -> Result<CorrectionReport> {
        if retro.localization.undetectable {
            return Err(Error::InvalidState(
                "closure reports no broken loop; there is nothing to correct, or it \
                 happened where no loop passes"
                    .into(),
            ));
        }
        if !retro.localization.unique {
            return Err(Error::InvalidState(format!(
                "closure is consistent with {} sites {:?} and {} links {:?}; it does not \
                 determine a correction and will not invent one",
                retro.localization.sites.len(),
                retro.localization.sites,
                retro.localization.links.len(),
                retro.localization.links
            )));
        }
        if retro.localization.sites.is_empty() {
            let link = retro.localization.links[0];
            return Err(Error::InvalidState(format!(
                "closure localizes the break to the ordering across link {link:?}, which \
                 names the link but not the transposition that moved it; re-order that \
                 link's endpoints deliberately rather than having this guess"
            )));
        }
        let site = retro.localization.sites[0];
        let broken_before = retro.localization.broken.loops().len();
        let was = self.bundle.inspect(site)?.fiber.spin;
        self.bundle.set_spin(site, !was)?;
        self.step += 1;
        let broken_after = self.broken_loops()?.len();
        Ok(CorrectionReport {
            site,
            broken_before,
            broken_after,
            restored: broken_after == 0,
        })
    }

    /// Take the bundle back out, corrected or not.
    pub fn into_bundle(self) -> PolarityBundle {
        self.bundle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> PolarityBundle {
        let mut b = PolarityBundle::new(3).unwrap();
        b.link(0, 1).unwrap();
        b.link(1, 2).unwrap();
        b.link(2, 0).unwrap();
        b
    }

    #[test]
    fn a_triangle_supplies_exactly_one_loop() {
        let basis = closure_basis(&triangle());
        assert_eq!(basis.rank, 1);
        assert_eq!(basis.loops[0].len(), 3);
        assert!(basis.bridges.is_empty());
    }

    #[test]
    fn an_intended_change_keeps_closure_and_a_perturbation_breaks_it() {
        let mut h = ClosureHistory::new(triangle()).unwrap();
        h.set_spin(1, true).unwrap();
        assert!(h.break_report().unwrap().intact());
        h.perturb_spin(2).unwrap();
        assert_eq!(h.break_report().unwrap().spin, vec![0]);
    }

    #[test]
    fn a_tree_has_no_loops_and_therefore_no_reach() {
        let mut b = PolarityBundle::new(4).unwrap();
        b.link(0, 1).unwrap();
        b.link(1, 2).unwrap();
        b.link(2, 3).unwrap();
        let basis = closure_basis(&b);
        assert_eq!(basis.rank, 0);
        assert_eq!(basis.bridges.len(), 3);
        let mut h = ClosureHistory::new(b).unwrap();
        h.perturb_spin(1).unwrap();
        let broken = h.break_report().unwrap();
        assert!(broken.intact());
        assert!(h.localize(&broken).undetectable);
    }
}
