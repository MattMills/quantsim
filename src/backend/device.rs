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
#[derive(Debug, Clone)]
pub struct DeviceState<S: Scalar> {
    topology: Topology,
    durations: DurationModel,
    policy: ArityPolicy,
    /// Physically-indexed amplitudes.
    inner: DenseState<S>,
    logical_to_physical: Vec<usize>,
    physical_to_logical: Vec<usize>,
    clocks: Vec<u64>,
    log: Vec<PhysicalOp>,
    swaps: usize,
}

impl<S: Scalar> DeviceState<S> {
    /// `|0…0⟩` on the given topology with logical qubit `q` initially at
    /// physical site `q`.
    pub fn new(topology: Topology, durations: DurationModel, policy: ArityPolicy) -> Result<Self> {
        let n = topology.num_sites();
        Ok(DeviceState {
            inner: DenseState::new(n)?,
            logical_to_physical: (0..n).collect(),
            physical_to_logical: (0..n).collect(),
            clocks: vec![0; n],
            log: Vec::new(),
            swaps: 0,
            topology,
            durations,
            policy,
        })
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
        self.stamp("swap", vec![a, b], self.durations.swap);
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
                self.stamp("gate1q", vec![p], self.durations.one_q);
                Ok(())
            }
            2 => {
                let (pa, pb) = self.route_pair(qubits[0], qubits[1])?;
                self.inner.apply(matrix, &[pa, pb])?;
                self.stamp("gate2q", vec![pa, pb], self.durations.two_q);
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
                self.stamp(
                    format!("virtual{k}q"),
                    sites,
                    self.durations.two_q * k as u64,
                );
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
                self.stamp("diag1q", vec![p], self.durations.one_q);
                Ok(())
            }
            2 => {
                // A two-qubit diagonal is still an interaction: route it.
                let (pa, pb) = self.route_pair(qubits[0], qubits[1])?;
                self.inner.apply_diagonal(entries, &[pa, pb])?;
                self.stamp("diag2q", vec![pa, pb], self.durations.two_q);
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
                self.stamp(
                    format!("virtual{k}q"),
                    sites,
                    self.durations.two_q * k as u64,
                );
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
        self.stamp("measure", vec![p], self.durations.measure);
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
