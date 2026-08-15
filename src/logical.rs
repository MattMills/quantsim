//! The code space as a representation bet: constraint-driven
//! compression.
//!
//! [`retro`](crate::retro) measured that a code is a constraint set the
//! simulator can read deterministically. This module turns the
//! constraint set into a **backend**: [`LogicalState`] holds an
//! `n`-qubit register as
//!
//! ```text
//! |ψ⟩ = P · Enc · |l⟩
//! ```
//!
//! — a Pauli frame `P` (the fault record, absorbed exactly), the code
//! layout's encoder `Enc` (recognized gate by gate as it streams in),
//! and a **logical state** `|l⟩` on `k = 2·nodes` wires held in an
//! inner backend. The bet: the circuit is an *encoded computation* —
//! its own encoders first, then logical operations (physical Paulis,
//! which the frame absorbs at any weight, and transversal CX blocks,
//! which commit as logical CXs when their `2L²` physical gates
//! complete). While the bet holds, memory is the logical state plus
//! bookkeeping: `2^k` against the register's `2^n`, an exponent
//! divided by `L²` per node. A gate outside the vocabulary
//! materializes the exact state and the axis honestly reads sparse
//! from then on — the phase-field's escape contract, one level up.
//!
//! Reads are exact and never materialize: an amplitude is a coset
//! membership question over F2 (which codeword orbit does the index
//! sit in, after the frame is peeled off), answered from the X-side
//! generator row space in `O(gens)` plus one inner amplitude.
//!
//! Faults and retrocorrection live *inside* the representation: a
//! physical Pauli fault lands in the frame, the frame's
//! [`signature`](crate::retro::signature) IS the syndrome (no state
//! work at all), and a decoded correction is one more frame update.
//! The constraint predicts, diagnoses, and repairs in bookkeeping
//! space.
//!
//! Phase 1 scope, stated: the logical vocabulary is Clifford (Paulis
//! and transversal CX), the inner default is sparse, and logical
//! magic — like a non-sparse inner for the logical layer (a graph
//! bundle holding the LINK GRAPH rather than its Schmidt expansion) —
//! is the named next rung.

use std::collections::HashMap;

use crate::backend::{conjugate_by_step, Backend, CliffordStep, PauliString, SparseState};
use crate::circuit::Op;
use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::retro::{Code, ToricCode};
use crate::scalar::{Scalar, C64};

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

/// Exact matrix recognition against a 2×2 or 4×4 constant.
fn matches(matrix: &GateMatrix<C64>, want: &[C64]) -> bool {
    let d = matrix.dim();
    if d * d != want.len() {
        return false;
    }
    (0..d * d).all(|i| {
        let got = matrix.data()[i];
        (got - want[i]).norm_sqr() < 1e-24
    })
}

fn is_x(m: &GateMatrix<C64>) -> bool {
    matches(m, &[c(0., 0.), c(1., 0.), c(1., 0.), c(0., 0.)])
}

fn is_y(m: &GateMatrix<C64>) -> bool {
    matches(m, &[c(0., 0.), c(0., -1.), c(0., 1.), c(0., 0.)])
}

fn is_z(m: &GateMatrix<C64>) -> bool {
    matches(m, &[c(1., 0.), c(0., 0.), c(0., 0.), c(-1., 0.)])
}

fn is_h(m: &GateMatrix<C64>) -> bool {
    let f = std::f64::consts::FRAC_1_SQRT_2;
    matches(m, &[c(f, 0.), c(f, 0.), c(f, 0.), c(-f, 0.)])
}

fn is_cx(m: &GateMatrix<C64>) -> bool {
    let mut want = vec![c(0., 0.); 16];
    // Convention: qubits[0] is matrix sub-index bit 0 (the control).
    for (row, col) in [(0usize, 0usize), (2, 2), (1, 3), (3, 1)] {
        want[row * 4 + col] = c(1., 0.);
    }
    matches(m, &want)
}

/// One expected encoder step, precomputed from the codes' circuits.
#[derive(Clone, Debug, PartialEq, Eq)]
enum EncStep {
    H(usize),
    Cx(usize, usize),
}

/// The Pauli frame: `phase · X^x Z^z`, exact.
#[derive(Clone, Copy, Debug)]
struct Frame {
    x: u64,
    z: u64,
    phase: C64,
}

impl Frame {
    fn identity() -> Frame {
        Frame {
            x: 0,
            z: 0,
            phase: c(1., 0.),
        }
    }

    /// Left-multiply by `phase·X^{gx}Z^{gz}` (the new gate acts after
    /// the stored frame): `(X^{gx}Z^{gz})(X^x Z^z) =
    /// (−1)^{gz·x} X^{gx⊕x} Z^{gz⊕z}`.
    fn push(&mut self, gx: u64, gz: u64, phase: C64) {
        let swaps = (gz & self.x).count_ones();
        self.x ^= gx;
        self.z ^= gz;
        self.phase *= phase;
        if swaps % 2 == 1 {
            self.phase *= c(-1., 0.);
        }
    }

    fn as_string(&self) -> PauliString {
        PauliString {
            x: self.x,
            z: self.z,
            negative: false,
        }
    }
}

/// Counters for what the representation did — the measured ledger.
#[derive(Clone, Copy, Debug, Default)]
pub struct LogicalStats {
    /// Encoder gates absorbed by op-matching.
    pub encoder_gates: usize,
    /// Physical Paulis absorbed into the frame.
    pub frame_paulis: usize,
    /// Transversal blocks committed as logical CXs.
    pub logical_cx: usize,
    /// Gates that left the vocabulary (0 while the bet holds).
    pub escapes: usize,
}

/// The constraint-compression backend: toric nodes tiled over the
/// register, an inner backend on the logical wires, a Pauli frame for
/// everything the constraints absorb.
pub struct LogicalState {
    n: usize,
    codes: Vec<ToricCode>,
    /// Reduced-row-echelon basis of the X-generator row space (all
    /// nodes), for coset membership.
    gen_basis: Vec<u64>,
    /// Logical X̄ rows reduced against `gen_basis`, one per wire.
    xbar_red: Vec<u64>,
    inner: SparseState<C64>,
    frame: Frame,
    enc: Vec<EncStep>,
    enc_pos: usize,
    /// Open transversal block per ordered node pair: local edge
    /// indices already seen.
    pending_cx: HashMap<(usize, usize), Vec<usize>>,
    materialized: Option<SparseState<C64>>,
    stats: LogicalStats,
}

impl LogicalState {
    /// Tile `L = 2` toric nodes across the register. A width that does
    /// not tile (not a positive multiple of 8) gets no layout: the bet
    /// is vacuous there, and the register starts materialized — priced
    /// honestly as sparse, never silently wrong.
    pub fn tiled(n: usize) -> Result<Self> {
        if n == 0 || n % 8 != 0 || n > 63 {
            return Self::bare(n);
        }
        let codes = (0..n / 8)
            .map(|i| ToricCode::new(2, 8 * i))
            .collect::<Result<Vec<_>>>()?;
        Self::with_codes(n, codes)
    }

    /// No layout at all: materialized from birth (exact, sparse-priced).
    pub fn bare(n: usize) -> Result<Self> {
        let mut out = Self::with_codes(8, vec![ToricCode::new(2, 0)?])?;
        out.n = n;
        out.codes.clear();
        out.gen_basis.clear();
        out.xbar_red.clear();
        out.enc.clear();
        out.enc_pos = 0;
        out.materialized = Some(SparseState::new(n)?);
        Ok(out)
    }

    /// A register whose constraint layout is the given toric nodes.
    pub fn with_codes(n: usize, codes: Vec<ToricCode>) -> Result<Self> {
        if codes.is_empty() {
            return Err(Error::InvalidState("logical: at least one node".into()));
        }
        if n > 63 {
            return Err(Error::TooManyQubits {
                requested: n,
                max: 63,
            });
        }
        for code in &codes {
            if code.offset() + code.qubits() > n {
                return Err(Error::InvalidState(format!(
                    "logical: node at offset {} overruns the {n}-qubit register",
                    code.offset()
                )));
            }
        }
        // The expected encoder stream: each node's |0̄0̄⟩ circuit in
        // node order, exactly as the library's families emit it.
        let mut enc = Vec::new();
        for code in &codes {
            for op in code.encoder(n, [false, false]).ops() {
                let Op::Named { name, qubits, .. } = op else {
                    unreachable!("the encoder emits named gates")
                };
                match name.as_str() {
                    "h" => enc.push(EncStep::H(qubits[0])),
                    "cx" => enc.push(EncStep::Cx(qubits[0], qubits[1])),
                    other => unreachable!("encoder emits h/cx only, got {other}"),
                }
            }
        }
        // F2 row space of every node's X-side generators, echeloned.
        let mut gen_basis: Vec<u64> = Vec::new();
        let reduce_by = |mut row: u64, basis: &[u64]| -> u64 {
            loop {
                let mut changed = false;
                for &b in basis {
                    if row != 0 && row & (1u64 << b.trailing_zeros()) != 0 {
                        row ^= b;
                        changed = true;
                    }
                }
                if !changed {
                    return row;
                }
            }
        };
        for code in &codes {
            for g in Code::generators(code) {
                let red = reduce_by(g.x, &gen_basis);
                if red != 0 {
                    gen_basis.push(red);
                }
            }
        }
        let mut xbar_red = Vec::new();
        for code in &codes {
            for i in 0..2 {
                let red = reduce_by(code.logical_x(i).x, &gen_basis);
                assert_ne!(red, 0, "a logical is never in the generator span");
                xbar_red.push(red);
            }
        }
        let k = 2 * codes.len();
        Ok(LogicalState {
            n,
            codes,
            gen_basis,
            xbar_red,
            inner: SparseState::new(k)?,
            frame: Frame::identity(),
            enc,
            enc_pos: 0,
            pending_cx: HashMap::new(),
            materialized: None,
            stats: LogicalStats::default(),
        })
    }

    /// Logical wires (`2` per node).
    pub fn logical_qubits(&self) -> usize {
        2 * self.codes.len()
    }

    /// The measured ledger.
    pub fn stats(&self) -> LogicalStats {
        self.stats
    }

    /// Whether the bet still holds (no escape, no materialization).
    pub fn is_logical(&self) -> bool {
        self.materialized.is_none()
    }

    /// Transversal blocks currently open (reads mid-block see the
    /// block as not yet applied).
    pub fn pending_blocks(&self) -> usize {
        self.pending_cx.len()
    }

    /// The frame's syndrome against one node — the fault record read
    /// with **no state work at all**: the constraint diagnoses from
    /// bookkeeping.
    pub fn frame_syndrome(&self, code: &ToricCode) -> Vec<bool> {
        crate::retro::signature(code, self.frame.as_string())
    }

    /// Absorb a Pauli correction into the frame (retrocorrection
    /// in-representation: one bookkeeping update, any weight).
    pub fn correct(&mut self, p: PauliString) {
        let y = (p.x & p.z).count_ones();
        let mut phase = match y % 4 {
            0 => c(1., 0.),
            1 => c(0., 1.),
            2 => c(-1., 0.),
            _ => c(0., -1.),
        };
        if p.negative {
            phase *= c(-1., 0.);
        }
        self.frame.push(p.x, p.z, phase);
        self.stats.frame_paulis += 1;
    }

    /// Reduce an index against the generator row space; the remainder
    /// keys the codeword coset.
    fn reduce(&self, mut row: u64) -> u64 {
        loop {
            let mut changed = false;
            for &b in self.gen_basis.iter() {
                if row & (1u64 << b.trailing_zeros()) != 0 {
                    row ^= b;
                    changed = true;
                }
            }
            if !changed {
                return row;
            }
        }
    }

    /// Which logical basis state's orbit contains `j`, if any: solve
    /// `⊕ bᵢ·x̄ᵢ_red = reduce(j)` over the `k` wires by elimination
    /// (k ≤ 16 here; brute force is honest and instant).
    fn coset_of(&self, j: u64) -> Option<u64> {
        let target = self.reduce(j);
        let k = self.xbar_red.len();
        for b in 0..(1u64 << k) {
            let mut acc = 0u64;
            for (i, &row) in self.xbar_red.iter().enumerate() {
                if b >> i & 1 == 1 {
                    acc ^= row;
                }
            }
            if self.reduce(acc) == target {
                return Some(b);
            }
        }
        None
    }

    fn materialize(&mut self) -> Result<()> {
        if self.materialized.is_some() {
            return Ok(());
        }
        let mut entries: Vec<(u64, C64)> = Vec::new();
        self.enumerate(&mut |i, a| entries.push((i, a)));
        let mut s = SparseState::new(self.n)?;
        s.load(&entries)?;
        self.materialized = Some(s);
        self.stats.escapes += 1;
        Ok(())
    }

    /// Exact enumeration of the physical state from the representation
    /// (used by reads and by the escape).
    fn enumerate(&self, f: &mut dyn FnMut(u64, C64)) {
        if self.enc_pos < self.enc.len() {
            // Mid-encoder: replay the matched prefix concretely.
            let mut s = SparseState::<C64>::new(self.n).expect("width validated");
            let h = GateMatrix::from_vec(
                2,
                vec![
                    c(std::f64::consts::FRAC_1_SQRT_2, 0.),
                    c(std::f64::consts::FRAC_1_SQRT_2, 0.),
                    c(std::f64::consts::FRAC_1_SQRT_2, 0.),
                    c(-std::f64::consts::FRAC_1_SQRT_2, 0.),
                ],
            )
            .expect("2x2");
            let mut cx = vec![c(0., 0.); 16];
            for (row, col) in [(0usize, 0usize), (2, 2), (1, 3), (3, 1)] {
                cx[row * 4 + col] = c(1., 0.);
            }
            let cx = GateMatrix::from_vec(4, cx).expect("4x4");
            for step in &self.enc[..self.enc_pos] {
                match step {
                    EncStep::H(q) => s.apply(&h, &[*q]).expect("replay"),
                    EncStep::Cx(a, b) => s.apply(&cx, &[*a, *b]).expect("replay"),
                }
            }
            s.for_each_nonzero(f);
            return;
        }
        // Post-encoder: orbit enumeration per inner nonzero.
        let orbit_count = 1u64 << self.gen_basis.len();
        let norm = 1.0 / (orbit_count as f64).sqrt();
        self.inner.for_each_nonzero(&mut |b, alpha| {
            let mut base = 0u64;
            for (i, &row) in self.xbar_red.iter().enumerate() {
                if b >> i & 1 == 1 {
                    base ^= row;
                }
            }
            for subset in 0..orbit_count {
                let mut j = base;
                for (bi, &row) in self.gen_basis.iter().enumerate() {
                    if subset >> bi & 1 == 1 {
                        j ^= row;
                    }
                }
                // Apply the frame: index flip by x, sign by z·j, phase.
                let i = j ^ self.frame.x;
                let sign = if (self.frame.z & j).count_ones() % 2 == 1 {
                    -1.0
                } else {
                    1.0
                };
                f(i, alpha * self.frame.phase * c(sign * norm, 0.0));
            }
        });
    }
}

impl Backend<C64> for LogicalState {
    fn name(&self) -> &str {
        "logical"
    }

    fn num_qubits(&self) -> usize {
        self.n
    }

    fn apply(&mut self, matrix: &GateMatrix<C64>, qubits: &[usize]) -> Result<()> {
        if let Some(s) = self.materialized.as_mut() {
            return s.apply(matrix, qubits);
        }
        // Era 1: the encoder, op for op.
        if self.enc_pos < self.enc.len() {
            let expected = &self.enc[self.enc_pos];
            let hit = match (expected, qubits.len()) {
                (EncStep::H(q), 1) => *q == qubits[0] && is_h(matrix),
                (EncStep::Cx(a, b), 2) => *a == qubits[0] && *b == qubits[1] && is_cx(matrix),
                _ => false,
            };
            if hit {
                self.enc_pos += 1;
                self.stats.encoder_gates += 1;
                return Ok(());
            }
            self.materialize()?;
            return self.materialized.as_mut().unwrap().apply(matrix, qubits);
        }
        // Era 2: Paulis into the frame at any time…
        if qubits.len() == 1 {
            let bit = 1u64 << qubits[0];
            if is_x(matrix) {
                self.frame.push(bit, 0, c(1., 0.));
                self.stats.frame_paulis += 1;
                return Ok(());
            }
            if is_y(matrix) {
                self.frame.push(bit, bit, c(0., 1.));
                self.stats.frame_paulis += 1;
                return Ok(());
            }
            if is_z(matrix) {
                self.frame.push(0, bit, c(1., 0.));
                self.stats.frame_paulis += 1;
                return Ok(());
            }
        }
        // …and transversal CX blocks, committed when complete.
        if qubits.len() == 2 && is_cx(matrix) {
            let node_of = |codes: &[ToricCode], q: usize| -> Option<(usize, usize)> {
                codes
                    .iter()
                    .position(|code| q >= code.offset() && q < code.offset() + code.qubits())
                    .map(|i| (i, q - codes[i].offset()))
            };
            let hit = (
                node_of(&self.codes, qubits[0]),
                node_of(&self.codes, qubits[1]),
            );
            if let (Some((na, la)), Some((nb, lb))) = hit {
                if na != nb && la == lb {
                    let entry = self.pending_cx.entry((na, nb)).or_default();
                    if !entry.contains(&la) {
                        entry.push(la);
                        let full = self.codes[na].qubits();
                        if entry.len() == full {
                            // Commit: logical CX on both wire pairs,
                            // frame conjugated through the block.
                            self.pending_cx.remove(&(na, nb));
                            let mut cxm = vec![c(0., 0.); 16];
                            for (row, col) in [(0usize, 0usize), (2, 2), (1, 3), (3, 1)] {
                                cxm[row * 4 + col] = c(1., 0.);
                            }
                            let cxm = GateMatrix::from_vec(4, cxm)?;
                            self.inner.apply(&cxm, &[2 * na, 2 * nb])?;
                            self.inner.apply(&cxm, &[2 * na + 1, 2 * nb + 1])?;
                            // The crate's PauliString is the Hermitian
                            // i^{|x∧z|}X^xZ^z; the frame is raw
                            // phase·X^xZ^z. Conjugation can change the
                            // Y-overlap, so the raw phase picks up
                            // i^{y_after − y_before} beside the sign.
                            let y_before = (self.frame.x & self.frame.z).count_ones() as i64;
                            let mut fr = self.frame.as_string();
                            for q in 0..full {
                                fr = conjugate_by_step(
                                    fr,
                                    CliffordStep::Cx(
                                        self.codes[na].offset() + q,
                                        self.codes[nb].offset() + q,
                                    ),
                                );
                            }
                            let y_after = (fr.x & fr.z).count_ones() as i64;
                            let mut phase = self.frame.phase;
                            if fr.negative {
                                phase *= c(-1., 0.);
                            }
                            phase *= match ((y_after - y_before).rem_euclid(4)) as u32 {
                                0 => c(1., 0.),
                                1 => c(0., 1.),
                                2 => c(-1., 0.),
                                _ => c(0., -1.),
                            };
                            self.frame = Frame {
                                x: fr.x,
                                z: fr.z,
                                phase,
                            };
                            self.stats.logical_cx += 1;
                        }
                        return Ok(());
                    }
                }
            }
        }
        self.materialize()?;
        self.materialized.as_mut().unwrap().apply(matrix, qubits)
    }

    fn apply_diagonal(&mut self, entries: &[C64], qubits: &[usize]) -> Result<()> {
        if let Some(s) = self.materialized.as_mut() {
            return s.apply_diagonal(entries, qubits);
        }
        // Identity and Z are the diagonal vocabulary; the rest escapes.
        if self.enc_pos == self.enc.len() && qubits.len() == 1 && entries.len() == 2 {
            let id = (entries[0] - c(1., 0.)).norm_sqr() < 1e-24;
            if id && (entries[1] - c(1., 0.)).norm_sqr() < 1e-24 {
                return Ok(());
            }
            if id && (entries[1] - c(-1., 0.)).norm_sqr() < 1e-24 {
                self.frame.push(0, 1u64 << qubits[0], c(1., 0.));
                self.stats.frame_paulis += 1;
                return Ok(());
            }
        }
        self.materialize()?;
        self.materialized
            .as_mut()
            .unwrap()
            .apply_diagonal(entries, qubits)
    }

    fn amplitude(&self, index: u64) -> C64 {
        if let Some(s) = self.materialized.as_ref() {
            return s.amplitude(index);
        }
        if self.enc_pos < self.enc.len() {
            let mut acc = c(0., 0.);
            self.enumerate(&mut |i, a| {
                if i == index {
                    acc += a;
                }
            });
            return acc;
        }
        // Peel the frame, key the coset, ask the inner state.
        let j = index ^ self.frame.x;
        // The index must vanish on every bare (non-code, non-frame)
        // qubit and sit in some codeword orbit.
        let Some(b) = self.coset_of(j) else {
            return c(0., 0.);
        };
        // Membership requires the reduction to be exact: j must equal
        // base ⊕ span element — verified by reconstructing.
        let mut base = 0u64;
        for (i, &row) in self.xbar_red.iter().enumerate() {
            if b >> i & 1 == 1 {
                base ^= row;
            }
        }
        if self.reduce(j ^ base) != 0 {
            return c(0., 0.);
        }
        let sign = if (self.frame.z & j).count_ones() % 2 == 1 {
            -1.0
        } else {
            1.0
        };
        let norm = 1.0 / ((1u64 << self.gen_basis.len()) as f64).sqrt();
        self.inner.amplitude(b) * self.frame.phase * c(sign * norm, 0.0)
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, C64)) {
        if let Some(s) = self.materialized.as_ref() {
            s.for_each_nonzero(f);
            return;
        }
        let mut entries: Vec<(u64, C64)> = Vec::new();
        self.enumerate(&mut |i, a| entries.push((i, a)));
        entries.sort_unstable_by_key(|&(i, _)| i);
        for (i, a) in entries {
            if a.abs_sqr() > 0.0 {
                f(i, a);
            }
        }
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        let _ = self.materialize();
        if let Some(s) = self.materialized.as_mut() {
            s.project(qubit, outcome, renorm);
        }
    }

    fn reset(&mut self) {
        self.frame = Frame::identity();
        self.enc_pos = 0;
        self.pending_cx.clear();
        if self.codes.is_empty() {
            self.materialized = Some(SparseState::new(self.n).expect("width validated"));
            return;
        }
        self.inner = SparseState::new(self.logical_qubits()).expect("width validated");
        self.materialized = None;
    }

    fn load(&mut self, entries: &[(u64, C64)]) -> Result<()> {
        let mut s = SparseState::new(self.n)?;
        s.load(entries)?;
        self.materialized = Some(s);
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        if let Some(s) = self.materialized.as_ref() {
            return s.memory_bytes() + std::mem::size_of::<Self>();
        }
        self.inner.memory_bytes()
            + self.gen_basis.len() * 8
            + self.xbar_red.len() * 8
            + self.enc.len() * std::mem::size_of::<EncStep>()
            + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
