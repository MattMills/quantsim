//! Device-model backend: reproduce the *physical operation order* of a real
//! quantum computer.
//!
//! A [`DeviceState`] executes the same logical circuit as the reference
//! backend but the way hardware would:
//!
//! * a [`Topology`] (coupling map) restricts native two-qubit gates to
//!   adjacent physical sites; a logical gate on non-adjacent qubits is
//!   **routed** — SWAPs move one operand stepwise through adjacent sites,
//!   so entanglement propagates through the coupling graph the way it does
//!   on the machine;
//! * routing permanently updates the logical→physical mapping (data ends up
//!   living where the computation dragged it, exactly like real routing);
//! * every operation is stamped onto per-qubit clocks with a
//!   [`DurationModel`]: independent qubits advance in parallel, dependent
//!   ones serialize — [`DeviceState::elapsed`] is the schedule length, and
//!   [`DeviceState::physical_log`] is the full physical-order trace.
//!
//! Externally the backend answers in **logical** indices (amplitudes,
//! measurements, sampling translate through the mapping), so its results
//! must be — and are, via the conformance suite — identical to the dense
//! reference. The physics of *how* is what changed, and it is measurable.
//!
//! Gates of arity ≥ 3 are not native on hardware; [`ArityPolicy`] chooses
//! between rejecting them ([`ArityPolicy::Reject`]) and applying them as
//! "virtual" ops with a latency charge ([`ArityPolicy::Virtual`], default —
//! keeps the full gate registry usable while you study routing).
//!
//! **Reproducing existing machines.** [`Topology`] ships the coupling
//! maps of real register geometries — [`Topology::heavy_hex_falcon27`]
//! (the 27-qubit IBM Falcon-r4 heavy-hex lattice),
//! [`Topology::sycamore_like`] (the diagonal-coupler lattice of the
//! Sycamore family), [`Topology::complete`] (trapped-ion all-to-all) —
//! and [`LatencyMap`] carries their operation-latency maps: a base
//! [`DurationModel`] (era-representative presets:
//! [`DurationModel::ibm_falcon_like`], [`DurationModel::sycamore_like`],
//! [`DurationModel::ion_trap_like`]; ticks are nanoseconds) plus
//! **per-site and per-edge overrides**, because real calibration data is
//! heterogeneous. [`DeviceState::with_latency`] additionally takes the
//! inner state representation by value, so chip-scale geometry (GHZ
//! across all 54 Sycamore sites) runs over a sparse inner instead of a
//! 2^54 dense vector. [`DeviceState::elapsed`] is the schedule length
//! under per-qubit clocks and [`DeviceState::serial_time`] the
//! no-parallelism total — their ratio measures how much parallelism the
//! geometry admitted. Routing is currently latency-blind (BFS by edge
//! count); the slow-edge demonstration in
//! `examples/device_reproduction.rs` measures the cost of that, which
//! is the roadmap motivation for latency-aware routing.

use super::{validate_apply, validate_apply_diagonal, Backend};
use crate::backend::dense::DenseState;
use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::scalar::Scalar;

/// An undirected coupling map over physical sites.
#[derive(Debug, Clone)]
pub struct Topology {
    n: usize,
    adjacency: Vec<Vec<usize>>,
}

impl Topology {
    /// A linear chain `0 — 1 — … — n−1`.
    pub fn linear(n: usize) -> Self {
        let mut edges = Vec::new();
        for i in 0..n.saturating_sub(1) {
            edges.push((i, i + 1));
        }
        Self::custom(n, &edges).expect("chain edges are valid")
    }

    /// A ring: the chain plus the closing edge.
    pub fn ring(n: usize) -> Self {
        let mut edges = Vec::new();
        for i in 0..n.saturating_sub(1) {
            edges.push((i, i + 1));
        }
        if n > 2 {
            edges.push((n - 1, 0));
        }
        Self::custom(n, &edges).expect("ring edges are valid")
    }

    /// A `width × height` grid, row-major site numbering.
    pub fn grid(width: usize, height: usize) -> Self {
        let n = width * height;
        let mut edges = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let site = y * width + x;
                if x + 1 < width {
                    edges.push((site, site + 1));
                }
                if y + 1 < height {
                    edges.push((site, site + width));
                }
            }
        }
        Self::custom(n, &edges).expect("grid edges are valid")
    }

    /// All-to-all connectivity on `n` sites — the trapped-ion register
    /// geometry (every pair shares a gate; routing never swaps).
    pub fn complete(n: usize) -> Self {
        let mut edges = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                edges.push((a, b));
            }
        }
        Self::custom(n, &edges).expect("complete-graph edges are valid")
    }

    /// The 27-qubit heavy-hex coupling map of the IBM Falcon r4 family
    /// (the lattice of e.g. the Montreal/Mumbai-class devices): 28
    /// couplers, maximum degree 3.
    pub fn heavy_hex_falcon27() -> Self {
        const EDGES: [(usize, usize); 28] = [
            (0, 1),
            (1, 2),
            (1, 4),
            (2, 3),
            (3, 5),
            (4, 7),
            (5, 8),
            (6, 7),
            (7, 10),
            (8, 9),
            (8, 11),
            (10, 12),
            (11, 14),
            (12, 13),
            (12, 15),
            (13, 14),
            (14, 16),
            (15, 18),
            (16, 19),
            (17, 18),
            (18, 21),
            (19, 20),
            (19, 22),
            (21, 23),
            (22, 25),
            (23, 24),
            (24, 25),
            (25, 26),
        ];
        Self::custom(27, &EDGES).expect("falcon-27 edges are valid")
    }

    /// The induced sub-map of [`Topology::heavy_hex_falcon27`] on its
    /// first `n` sites (a connected corner of the chip for `n ≥ 2`) —
    /// the "qubit selection" step of running a small circuit on a big
    /// device.
    pub fn heavy_hex_falcon_corner(n: usize) -> Result<Self> {
        let full = Self::heavy_hex_falcon27();
        if n == 0 || n > full.num_sites() {
            return Err(Error::QubitOutOfRange {
                qubit: n,
                num_qubits: full.num_sites(),
            });
        }
        let mut edges = Vec::new();
        for a in 0..n {
            for &b in &full.adjacency[a] {
                if b < n && a < b {
                    edges.push((a, b));
                }
            }
        }
        Self::custom(n, &edges)
    }

    /// A diagonal-coupler lattice of the Sycamore family: `rows × cols`
    /// sites, each coupled to its vertical neighbor and one alternating
    /// diagonal neighbor per row parity (maximum degree 4).
    /// `sycamore_like(6, 9)` is the 54-site Sycamore-class geometry.
    pub fn sycamore_like(rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        let site = |r: usize, c: usize| r * cols + c;
        let mut edges = Vec::new();
        for r in 0..rows.saturating_sub(1) {
            for c in 0..cols {
                edges.push((site(r, c), site(r + 1, c)));
                if r % 2 == 0 {
                    if c + 1 < cols {
                        edges.push((site(r, c), site(r + 1, c + 1)));
                    }
                } else if c > 0 {
                    edges.push((site(r, c), site(r + 1, c - 1)));
                }
            }
        }
        Self::custom(n, &edges).expect("sycamore-like edges are valid")
    }

    /// A register whose coupling fabric is a *causal hierarchy*: the
    /// nearest-neighbour chain plus skip couplers `(i, i + 2^k)` at
    /// every scale `k ≥ 1` (for `i` a multiple of `2^k`) — the Hasse
    /// diagram of a binary causal order laid onto hardware, mirroring
    /// the mera tree. Any two sites are within `O(log n)` hops, so the
    /// register's causal metric is *hyperbolic*: ball volumes grow
    /// exponentially and the causal horizon is logarithmic where a
    /// chain's is linear. Degree grows to `O(log n)` at scale hubs —
    /// a research geometry, priced honestly by the router.
    pub fn hierarchical(n: usize) -> Self {
        let mut edges = Vec::new();
        for i in 0..n.saturating_sub(1) {
            edges.push((i, i + 1));
        }
        let mut step = 2usize;
        while step < n {
            let mut i = 0;
            while i + step < n {
                edges.push((i, i + step));
                i += step;
            }
            step *= 2;
        }
        Self::custom(n, &edges).expect("hierarchical edges are valid")
    }

    /// The `dim`-dimensional hypercube register on `2^dim` sites: sites
    /// are bit strings, couplers connect strings differing in one bit.
    /// Diameter and degree are both `dim = log2 n` — a homogeneous
    /// log-horizon causal geometry (every site is a hub, unlike
    /// [`Topology::hierarchical`]).
    pub fn hypercube(dim: usize) -> Self {
        let n = 1usize << dim;
        let mut edges = Vec::new();
        for a in 0..n {
            for k in 0..dim {
                let b = a ^ (1 << k);
                if a < b {
                    edges.push((a, b));
                }
            }
        }
        Self::custom(n, &edges).expect("hypercube edges are valid")
    }

    /// Maximum vertex degree — 3 for heavy-hex, 4 for the diagonal
    /// lattice, `n − 1` for all-to-all.
    pub fn max_degree(&self) -> usize {
        self.adjacency.iter().map(|a| a.len()).max().unwrap_or(0)
    }

    /// Hop distances from `site` to every site (BFS; `usize::MAX` for
    /// unreachable sites).
    pub fn distances_from(&self, site: usize) -> Vec<usize> {
        let mut dist = vec![usize::MAX; self.n];
        if site >= self.n {
            return dist;
        }
        dist[site] = 0;
        let mut queue = std::collections::VecDeque::from([site]);
        while let Some(s) = queue.pop_front() {
            for &next in &self.adjacency[s] {
                if dist[next] == usize::MAX {
                    dist[next] = dist[s] + 1;
                    queue.push_back(next);
                }
            }
        }
        dist
    }

    /// The register's causal horizon: the largest hop distance between
    /// any two connected sites. Linear in `n` for a chain, `O(√n)` for
    /// a grid, `O(log n)` for [`Topology::hierarchical`] and
    /// [`Topology::hypercube`], 1 for all-to-all.
    pub fn diameter(&self) -> usize {
        (0..self.n)
            .flat_map(|s| self.distances_from(s))
            .filter(|&d| d != usize::MAX)
            .max()
            .unwrap_or(0)
    }

    /// Mean hop distance over ordered pairs of distinct connected
    /// sites — the expected causal separation of a random interaction.
    pub fn mean_distance(&self) -> f64 {
        let mut total = 0usize;
        let mut pairs = 0usize;
        for s in 0..self.n {
            for (t, &d) in self.distances_from(s).iter().enumerate() {
                if t != s && d != usize::MAX {
                    total += d;
                    pairs += 1;
                }
            }
        }
        if pairs == 0 {
            0.0
        } else {
            total as f64 / pairs as f64
        }
    }

    /// Causal-ball volumes from `site`: entry `r` is the number of
    /// sites within `r` hops. The growth profile is the register's
    /// curvature signature — linear growth is a flat 1-D fabric,
    /// polynomial is flat higher-D, exponential is hyperbolic.
    pub fn ball_sizes(&self, site: usize) -> Vec<usize> {
        let dist = self.distances_from(site);
        let max = dist
            .iter()
            .filter(|&&d| d != usize::MAX)
            .max()
            .copied()
            .unwrap_or(0);
        (0..=max)
            .map(|r| dist.iter().filter(|&&d| d <= r).count())
            .collect()
    }

    /// Computational availability at a swap budget: the fraction of
    /// unordered site pairs whose two-qubit interaction can be made
    /// native with at most `max_swaps` routing swaps (hop distance
    /// ≤ `max_swaps + 1`). Availability 1.0 at budget 0 is all-to-all;
    /// how fast the curve rises is what a causal register geometry
    /// buys.
    pub fn pair_availability(&self, max_swaps: usize) -> f64 {
        if self.n < 2 {
            return 1.0;
        }
        let mut reachable = 0usize;
        for s in 0..self.n {
            reachable += self
                .distances_from(s)
                .iter()
                .enumerate()
                .filter(|&(t, &d)| t != s && d != usize::MAX && d <= max_swaps + 1)
                .count();
        }
        reachable as f64 / (self.n * (self.n - 1)) as f64
    }

    /// Number of undirected couplers.
    pub fn num_edges(&self) -> usize {
        self.adjacency.iter().map(|a| a.len()).sum::<usize>() / 2
    }

    /// Whether every site can reach every other through couplers.
    pub fn is_connected(&self) -> bool {
        if self.n == 0 {
            return true;
        }
        let mut seen = vec![false; self.n];
        let mut queue = std::collections::VecDeque::from([0usize]);
        seen[0] = true;
        let mut count = 1;
        while let Some(site) = queue.pop_front() {
            for &next in &self.adjacency[site] {
                if !seen[next] {
                    seen[next] = true;
                    count += 1;
                    queue.push_back(next);
                }
            }
        }
        count == self.n
    }

    /// Arbitrary coupling map from an edge list.
    pub fn custom(n: usize, edges: &[(usize, usize)]) -> Result<Self> {
        let mut adjacency = vec![Vec::new(); n];
        for &(a, b) in edges {
            if a >= n || b >= n {
                return Err(Error::QubitOutOfRange {
                    qubit: a.max(b),
                    num_qubits: n,
                });
            }
            if a == b {
                return Err(Error::DuplicateQubits { qubits: vec![a, b] });
            }
            if !adjacency[a].contains(&b) {
                adjacency[a].push(b);
                adjacency[b].push(a);
            }
        }
        Ok(Topology { n, adjacency })
    }

    /// Number of physical sites.
    pub fn num_sites(&self) -> usize {
        self.n
    }

    /// Whether two sites share an edge.
    pub fn adjacent(&self, a: usize, b: usize) -> bool {
        self.adjacency[a].contains(&b)
    }

    /// BFS shortest path including both endpoints; `None` if disconnected.
    pub fn shortest_path(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        if from == to {
            return Some(vec![from]);
        }
        let mut previous = vec![usize::MAX; self.n];
        let mut queue = std::collections::VecDeque::from([from]);
        previous[from] = from;
        while let Some(site) = queue.pop_front() {
            for &next in &self.adjacency[site] {
                if previous[next] == usize::MAX {
                    previous[next] = site;
                    if next == to {
                        let mut path = vec![to];
                        let mut cursor = to;
                        while cursor != from {
                            cursor = previous[cursor];
                            path.push(cursor);
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(next);
                }
            }
        }
        None
    }
}

/// Gate durations in device ticks.
#[derive(Debug, Clone, Copy)]
pub struct DurationModel {
    /// Single-qubit gate duration.
    pub one_q: u64,
    /// Native two-qubit gate duration.
    pub two_q: u64,
    /// SWAP duration (typically ≈ 3 two-qubit gates).
    pub swap: u64,
    /// Measurement/projection duration.
    pub measure: u64,
}

impl Default for DurationModel {
    fn default() -> Self {
        DurationModel {
            one_q: 1,
            two_q: 10,
            swap: 30,
            measure: 50,
        }
    }
}

impl DurationModel {
    /// Era-representative superconducting timings of the IBM Falcon
    /// family, in nanoseconds (1q ≈ 35 ns, CX ≈ 450 ns, SWAP = 3·CX,
    /// readout ≈ 860 ns). Representative of published calibrations of
    /// that hardware generation, not a specific machine's live data.
    pub fn ibm_falcon_like() -> Self {
        DurationModel {
            one_q: 35,
            two_q: 450,
            swap: 1350,
            measure: 860,
        }
    }

    /// Era-representative Sycamore-class timings, in nanoseconds
    /// (1q ≈ 25 ns, 2q ≈ 32 ns, SWAP = 3·2q, readout ≈ 1 µs).
    pub fn sycamore_like() -> Self {
        DurationModel {
            one_q: 25,
            two_q: 32,
            swap: 96,
            measure: 1000,
        }
    }

    /// Era-representative trapped-ion timings, in nanoseconds
    /// (1q ≈ 10 µs, 2q ≈ 210 µs, readout ≈ 130 µs). All-to-all
    /// connectivity means the SWAP entry is rarely exercised.
    pub fn ion_trap_like() -> Self {
        DurationModel {
            one_q: 10_000,
            two_q: 210_000,
            swap: 630_000,
            measure: 130_000,
        }
    }
}

/// A device's operation-latency map: a base [`DurationModel`] plus
/// per-site (1q, measure) and per-edge (2q, SWAP) overrides — the shape
/// real calibration data takes, where every coupler has its own gate
/// time.
#[derive(Debug, Clone)]
pub struct LatencyMap {
    base: DurationModel,
    one_q: std::collections::HashMap<usize, u64>,
    two_q: std::collections::HashMap<(usize, usize), u64>,
    swap: std::collections::HashMap<(usize, usize), u64>,
    measure: std::collections::HashMap<usize, u64>,
}

impl From<DurationModel> for LatencyMap {
    fn from(base: DurationModel) -> Self {
        LatencyMap {
            base,
            one_q: std::collections::HashMap::new(),
            two_q: std::collections::HashMap::new(),
            swap: std::collections::HashMap::new(),
            measure: std::collections::HashMap::new(),
        }
    }
}

fn edge_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

impl LatencyMap {
    /// A map with uniform base durations and no overrides.
    pub fn uniform(base: DurationModel) -> Self {
        base.into()
    }

    /// The base model (used for un-overridden sites/edges and the
    /// virtual arity-≥3 charge).
    pub fn base(&self) -> DurationModel {
        self.base
    }

    /// Override the single-qubit duration at one site.
    pub fn set_one_q(&mut self, site: usize, ticks: u64) -> &mut Self {
        self.one_q.insert(site, ticks);
        self
    }

    /// Override the two-qubit duration on one edge (undirected).
    pub fn set_two_q(&mut self, a: usize, b: usize, ticks: u64) -> &mut Self {
        self.two_q.insert(edge_key(a, b), ticks);
        self
    }

    /// Override the SWAP duration on one edge (undirected).
    pub fn set_swap(&mut self, a: usize, b: usize, ticks: u64) -> &mut Self {
        self.swap.insert(edge_key(a, b), ticks);
        self
    }

    /// Override the measurement duration at one site.
    pub fn set_measure(&mut self, site: usize, ticks: u64) -> &mut Self {
        self.measure.insert(site, ticks);
        self
    }

    /// Effective single-qubit duration at a site.
    pub fn one_q_at(&self, site: usize) -> u64 {
        *self.one_q.get(&site).unwrap_or(&self.base.one_q)
    }

    /// Effective two-qubit duration on an edge.
    pub fn two_q_at(&self, a: usize, b: usize) -> u64 {
        *self.two_q.get(&edge_key(a, b)).unwrap_or(&self.base.two_q)
    }

    /// Effective SWAP duration on an edge.
    pub fn swap_at(&self, a: usize, b: usize) -> u64 {
        *self.swap.get(&edge_key(a, b)).unwrap_or(&self.base.swap)
    }

    /// Effective measurement duration at a site.
    pub fn measure_at(&self, site: usize) -> u64 {
        *self.measure.get(&site).unwrap_or(&self.base.measure)
    }
}

/// What to do with gates wider than the native two-qubit limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArityPolicy {
    /// Apply directly with a latency charge, logged as `virtual*` (default).
    #[default]
    Virtual,
    /// Refuse: the circuit must be decomposed to arity ≤ 2 first.
    Reject,
}

/// One entry of the physical execution trace.
#[derive(Debug, Clone)]
pub struct PhysicalOp {
    /// `gate1q`, `gate2q`, `diag*q`, `swap`, `virtual*q`, or `measure`.
    pub label: String,
    /// Physical sites involved.
    pub sites: Vec<usize>,
    /// Start tick (max clock over the involved sites at issue time).
    pub start: u64,
    /// End tick.
    pub end: u64,
}

/// The device-model backend. See the module docs.
pub struct DeviceState<S: Scalar> {
    topology: Topology,
    latency: LatencyMap,
    policy: ArityPolicy,
    /// Physically-indexed amplitudes, in any representation (dense by
    /// default; sparse makes chip-scale geometry affordable).
    inner: Box<dyn Backend<S>>,
    logical_to_physical: Vec<usize>,
    physical_to_logical: Vec<usize>,
    clocks: Vec<u64>,
    log: Vec<PhysicalOp>,
    swaps: usize,
}

impl<S: Scalar> DeviceState<S> {
    /// `|0…0⟩` on the given topology with logical qubit `q` initially at
    /// physical site `q`, over a dense inner state.
    pub fn new(topology: Topology, durations: DurationModel, policy: ArityPolicy) -> Result<Self> {
        let n = topology.num_sites();
        let inner: Box<dyn Backend<S>> = Box::new(DenseState::new(n)?);
        Self::with_latency(topology, durations.into(), policy, inner)
    }

    /// Full-control constructor: a per-site/per-edge [`LatencyMap`] and
    /// the inner representation by value — `Box::new(SparseState::new(n)?)`
    /// runs chip-scale geometry without a `2^n` dense vector.
    pub fn with_latency(
        topology: Topology,
        latency: LatencyMap,
        policy: ArityPolicy,
        inner: Box<dyn Backend<S>>,
    ) -> Result<Self> {
        let n = topology.num_sites();
        if inner.num_qubits() != n {
            return Err(Error::WidthMismatch {
                circuit: n,
                backend: inner.num_qubits(),
            });
        }
        Ok(DeviceState {
            inner,
            logical_to_physical: (0..n).collect(),
            physical_to_logical: (0..n).collect(),
            clocks: vec![0; n],
            log: Vec::new(),
            swaps: 0,
            topology,
            latency,
            policy,
        })
    }

    /// Sum of every physical op's duration — the schedule length a
    /// fully serial machine would need. `serial_time / elapsed` is the
    /// parallelism the geometry actually admitted.
    pub fn serial_time(&self) -> u64 {
        self.log.iter().map(|op| op.end - op.start).sum()
    }

    /// The operation-latency map in effect.
    pub fn latency(&self) -> &LatencyMap {
        &self.latency
    }

    /// Total schedule length so far (max per-qubit clock).
    pub fn elapsed(&self) -> u64 {
        self.clocks.iter().copied().max().unwrap_or(0)
    }

    /// SWAPs inserted by routing so far.
    pub fn swap_count(&self) -> usize {
        self.swaps
    }

    /// The physical execution trace, in issue order.
    pub fn physical_log(&self) -> &[PhysicalOp] {
        &self.log
    }

    /// Current logical→physical mapping.
    pub fn mapping(&self) -> &[usize] {
        &self.logical_to_physical
    }

    /// The coupling map.
    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    fn stamp(&mut self, label: impl Into<String>, sites: Vec<usize>, duration: u64) {
        let start = sites.iter().map(|&s| self.clocks[s]).max().unwrap_or(0);
        let end = start + duration;
        for &s in &sites {
            self.clocks[s] = end;
        }
        self.log.push(PhysicalOp {
            label: label.into(),
            sites,
            start,
            end,
        });
    }

    fn swap_physical(&mut self, a: usize, b: usize) -> Result<()> {
        debug_assert!(self.topology.adjacent(a, b));
        let swap = swap_matrix::<S>();
        self.inner.apply(&swap, &[a, b])?;
        let (la, lb) = (self.physical_to_logical[a], self.physical_to_logical[b]);
        self.physical_to_logical[a] = lb;
        self.physical_to_logical[b] = la;
        self.logical_to_physical[la] = b;
        self.logical_to_physical[lb] = a;
        let ticks = self.latency.swap_at(a, b);
        self.stamp("swap", vec![a, b], ticks);
        self.swaps += 1;
        Ok(())
    }

    /// Route logical operands until they sit on adjacent sites; returns the
    /// physical pair to apply the gate at.
    fn route_pair(&mut self, a: usize, b: usize) -> Result<(usize, usize)> {
        let (mut pa, pb) = (self.logical_to_physical[a], self.logical_to_physical[b]);
        if self.topology.adjacent(pa, pb) {
            return Ok((pa, pb));
        }
        let path = self.topology.shortest_path(pa, pb).ok_or_else(|| {
            Error::InvalidState(format!(
                "topology: no path between physical sites {pa} and {pb}"
            ))
        })?;
        // Walk operand `a` stepwise through adjacency to the site next to b.
        for window in 0..path.len().saturating_sub(2) {
            self.swap_physical(path[window], path[window + 1])?;
            pa = path[window + 1];
        }
        debug_assert!(self.topology.adjacent(pa, pb));
        Ok((pa, pb))
    }

    fn physical_index(&self, logical: u64) -> u64 {
        let mut physical = 0u64;
        for q in 0..self.topology.num_sites() {
            if (logical >> q) & 1 == 1 {
                physical |= 1u64 << self.logical_to_physical[q];
            }
        }
        physical
    }

    fn logical_index(&self, physical: u64) -> u64 {
        let mut logical = 0u64;
        for p in 0..self.topology.num_sites() {
            if (physical >> p) & 1 == 1 {
                logical |= 1u64 << self.physical_to_logical[p];
            }
        }
        logical
    }
}

fn swap_matrix<S: Scalar>() -> GateMatrix<S> {
    let (o, l) = (S::one(), S::zero());
    GateMatrix::from_vec(4, vec![o, l, l, l, l, l, o, l, l, o, l, l, l, l, l, o])
        .expect("4x4 is a valid dimension")
}

impl<S: Scalar> Backend<S> for DeviceState<S> {
    fn name(&self) -> &str {
        "device"
    }

    fn num_qubits(&self) -> usize {
        self.topology.num_sites()
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        validate_apply(self.num_qubits(), matrix, qubits)?;
        match qubits.len() {
            1 => {
                let p = self.logical_to_physical[qubits[0]];
                self.inner.apply(matrix, &[p])?;
                let ticks = self.latency.one_q_at(p);
                self.stamp("gate1q", vec![p], ticks);
                Ok(())
            }
            2 => {
                let (pa, pb) = self.route_pair(qubits[0], qubits[1])?;
                self.inner.apply(matrix, &[pa, pb])?;
                let ticks = self.latency.two_q_at(pa, pb);
                self.stamp("gate2q", vec![pa, pb], ticks);
                Ok(())
            }
            k => {
                if self.policy == ArityPolicy::Reject {
                    return Err(Error::InvalidState(format!(
                        "device model: {k}-qubit gates are not native (policy: Reject)"
                    )));
                }
                let sites: Vec<usize> = qubits
                    .iter()
                    .map(|&q| self.logical_to_physical[q])
                    .collect();
                self.inner.apply(matrix, &sites)?;
                let ticks = self.latency.base().two_q * k as u64;
                self.stamp(format!("virtual{k}q"), sites, ticks);
                Ok(())
            }
        }
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        validate_apply_diagonal(self.num_qubits(), entries, qubits)?;
        match qubits.len() {
            1 => {
                let p = self.logical_to_physical[qubits[0]];
                self.inner.apply_diagonal(entries, &[p])?;
                let ticks = self.latency.one_q_at(p);
                self.stamp("diag1q", vec![p], ticks);
                Ok(())
            }
            2 => {
                // A two-qubit diagonal is still an interaction: route it.
                let (pa, pb) = self.route_pair(qubits[0], qubits[1])?;
                self.inner.apply_diagonal(entries, &[pa, pb])?;
                let ticks = self.latency.two_q_at(pa, pb);
                self.stamp("diag2q", vec![pa, pb], ticks);
                Ok(())
            }
            k => {
                if self.policy == ArityPolicy::Reject {
                    return Err(Error::InvalidState(format!(
                        "device model: {k}-qubit diagonals are not native (policy: Reject)"
                    )));
                }
                let sites: Vec<usize> = qubits
                    .iter()
                    .map(|&q| self.logical_to_physical[q])
                    .collect();
                self.inner.apply_diagonal(entries, &sites)?;
                let ticks = self.latency.base().two_q * k as u64;
                self.stamp(format!("virtual{k}q"), sites, ticks);
                Ok(())
            }
        }
    }

    fn amplitude(&self, index: u64) -> S {
        self.inner.amplitude(self.physical_index(index))
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        self.inner
            .for_each_nonzero(&mut |physical, a| f(self.logical_index(physical), a));
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        let p = self.logical_to_physical[qubit];
        self.inner.project(p, outcome, renorm);
        let ticks = self.latency.measure_at(p);
        self.stamp("measure", vec![p], ticks);
    }

    fn reset(&mut self) {
        let n = self.topology.num_sites();
        self.inner.reset();
        self.logical_to_physical = (0..n).collect();
        self.physical_to_logical = (0..n).collect();
        self.clocks = vec![0; n];
        self.log.clear();
        self.swaps = 0;
    }

    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        // Loading defines a fresh logical state: reset the layout first.
        let n = self.topology.num_sites();
        self.logical_to_physical = (0..n).collect();
        self.physical_to_logical = (0..n).collect();
        self.inner.load(entries)
    }

    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
            + self.log.capacity() * std::mem::size_of::<PhysicalOp>()
            + 3 * self.clocks.capacity() * std::mem::size_of::<u64>()
            + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
