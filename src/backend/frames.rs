//! Local frames: deferred per-qubit basis changes as representation
//! metadata.
//!
//! A [`FramedState`] wraps any inner backend and maintains, per qubit, an
//! optional 2×2 unitary frame `W_q`. The physical state is
//! `(⊗_q W_q) · stored`, where `stored` lives in the inner backend. Gates
//! are executed against the *stored* state in whichever form is cheapest:
//!
//! * **single-qubit gates absorb** into the frame (`W_q ← G · W_q`) — zero
//!   amplitude work, and mutually-inverse pairs cancel to the identity
//!   frame without the state ever being touched;
//! * **multi-qubit gates are conjugated** into the frame,
//!   `G' = F† G F` with `F = ⊗ W_q` over the gate's qubits; when `G'`
//!   comes out diagonal it takes the diagonal fast path (no fill on the
//!   sparse inner, no factor merging cost beyond the diagonal's own);
//! * **observation flushes lazily**: measuring qubit `q` flushes only
//!   `W_q`; amplitude/support/sampling queries flush all frames first, so
//!   the [`Backend`] contract is met exactly — conformance holds framed
//!   backends to reference tolerance like everything else.
//!
//! What frames can and cannot buy, stated up front: entanglement structure
//! (Schmidt rank across cuts, cluster width) is **invariant under local
//! unitaries** — frames never shrink the factored or MPS cost axes. What
//! *is* frame-dependent is sparsity and gate diagonality: an `h`-layer
//! followed by an `rx`/`rxx` bulk keeps the stored support at **1** under
//! frames where the raw sparse representation blows up to `2^n` — see the
//! transverse-field tests. Representation cost is not physics, and frames
//! are the dial that separates them.
//!
//! The research API: [`FramedState::adopt_frame`] rewrites the
//! representation (`stored ← W† · stored`, frame ← `W`) without changing
//! the physical state — the concrete mechanism by which "inserted
//! structure" (deliberate basis choices) makes the *future* of a circuit
//! cheaper, with [`FramedState::stats`] and
//! [`FramedState::peak_inner_memory`] measuring exactly what it bought.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use super::Backend;
use crate::error::{Error, Result};
use crate::math::{GateMatrix, UNITARY_TOL};
use crate::rng::Prng;
use crate::scalar::Scalar;

/// Widest gate the conjugation path will build a dense `F` for; wider
/// gates flush the involved frames and pass through.
pub const FRAME_CONJUGATION_MAX: usize = 6;

/// Off-diagonal squared-magnitude tolerance for detecting that a
/// conjugated gate is diagonal.
const DIAG_TOL: f64 = 1e-24;

/// Counters describing how the framed wrapper executed its gates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// Single-qubit gates absorbed into frames (zero amplitude work).
    pub absorbed_1q: usize,
    /// Multi-qubit gates whose frame conjugate was diagonal and took the
    /// diagonal fast path.
    pub diagonalized: usize,
    /// Multi-qubit gates applied as dense conjugated matrices.
    pub conjugated: usize,
    /// Gates passed through with no active frames on their qubits.
    pub passthrough: usize,
    /// Frame applications to the inner state (flush work).
    pub flushes: usize,
}

struct Core<S: Scalar> {
    inner: Box<dyn Backend<S>>,
    frames: Vec<Option<GateMatrix<S>>>,
    stats: FrameStats,
    peak_inner_memory: usize,
}

impl<S: Scalar> Core<S> {
    fn note_peak(&mut self) {
        self.peak_inner_memory = self.peak_inner_memory.max(self.inner.memory_bytes());
    }

    fn flush_qubit(&mut self, q: usize) -> Result<()> {
        if let Some(w) = self.frames[q].take() {
            self.inner.apply(&w, &[q])?;
            self.stats.flushes += 1;
            self.note_peak();
        }
        Ok(())
    }

    fn flush_all(&mut self) -> Result<()> {
        for q in 0..self.frames.len() {
            self.flush_qubit(q)?;
        }
        Ok(())
    }

    fn any_active(&self) -> bool {
        self.frames.iter().any(|f| f.is_some())
    }
}

fn is_identity<S: Scalar>(m: &GateMatrix<S>) -> bool {
    let d = m.dim();
    for r in 0..d {
        for c in 0..d {
            let expect = if r == c { S::one() } else { S::zero() };
            if !(m.get(r, c) - expect).is_zero(1e-12) {
                return false;
            }
        }
    }
    true
}

/// Extract diagonal entries if all off-diagonals vanish within tolerance.
fn try_diagonal<S: Scalar>(m: &GateMatrix<S>) -> Option<Vec<S>> {
    let d = m.dim();
    for r in 0..d {
        for c in 0..d {
            if r != c && m.get(r, c).abs_sqr() > DIAG_TOL {
                return None;
            }
        }
    }
    Some((0..d).map(|i| m.get(i, i)).collect())
}

/// Any backend, wrapped with lazy per-qubit basis frames. See module docs.
pub struct FramedState<S: Scalar> {
    num_qubits: usize,
    name: String,
    /// Fast frame-activity flag readable without borrowing `core` — lets
    /// flush-on-read APIs short-circuit while an outer shared borrow (e.g.
    /// a `for_each_nonzero` callback querying `amplitude`) is live.
    active: Cell<bool>,
    core: RefCell<Core<S>>,
}

impl<S: Scalar> FramedState<S> {
    /// Wrap an inner backend (which must be in its initial or otherwise
    /// caller-understood state; frames start as identity).
    pub fn new(inner: Box<dyn Backend<S>>) -> Self {
        let num_qubits = inner.num_qubits();
        let name = format!("framed-{}", inner.name());
        let peak = inner.memory_bytes();
        FramedState {
            num_qubits,
            name,
            active: Cell::new(false),
            core: RefCell::new(Core {
                inner,
                frames: vec![None; num_qubits],
                stats: FrameStats::default(),
                peak_inner_memory: peak,
            }),
        }
    }

    /// Execution counters so far.
    pub fn stats(&self) -> FrameStats {
        self.core.borrow().stats
    }

    /// Number of qubits currently carrying a non-identity frame.
    pub fn active_frames(&self) -> usize {
        self.core
            .borrow()
            .frames
            .iter()
            .filter(|f| f.is_some())
            .count()
    }

    /// The frame on `q`, if any.
    pub fn frame_of(&self, q: usize) -> Option<GateMatrix<S>> {
        self.core.borrow().frames.get(q).cloned().flatten()
    }

    /// Peak memory the *inner* representation ever reached — the number
    /// frames exist to keep small.
    pub fn peak_inner_memory(&self) -> usize {
        self.core.borrow().peak_inner_memory
    }

    /// Stored-state support size, read **without** flushing (frame-relative
    /// introspection: this is the representation's cost, not the physical
    /// support).
    pub fn stored_nonzero_count(&self) -> usize {
        self.core.borrow().inner.nonzero_count()
    }

    /// Re-express the state in the basis `w` on qubit `q` without changing
    /// the physical state: applies `w† · (existing frame)` to the stored
    /// state and records `w` as the new frame. The deliberate-structure
    /// primitive: choose frames so the stored state gets sparse or the
    /// coming gates get diagonal.
    pub fn adopt_frame(&mut self, q: usize, w: &GateMatrix<S>) -> Result<()> {
        if q >= self.num_qubits {
            return Err(Error::QubitOutOfRange {
                qubit: q,
                num_qubits: self.num_qubits,
            });
        }
        if w.dim() != 2 {
            return Err(Error::BadDimension {
                expected: 2,
                got: w.dim(),
            });
        }
        let deviation = w.unitarity_deviation();
        if deviation > UNITARY_TOL {
            return Err(Error::NotUnitary {
                label: "adopt_frame".into(),
                deviation,
            });
        }
        let mut core = self.core.borrow_mut();
        let existing = core.frames[q].take();
        let mut correction = w.dagger();
        if let Some(f) = existing {
            correction = correction.matmul(&f);
        }
        if !is_identity(&correction) {
            core.inner.apply(&correction, &[q])?;
            core.stats.flushes += 1;
            core.note_peak();
        }
        core.frames[q] = if is_identity(w) {
            None
        } else {
            Some(w.clone())
        };
        let active = core.any_active();
        drop(core);
        self.active.set(active);
        Ok(())
    }

    /// Flush the frame on `q` into the stored state (frame becomes
    /// identity; physical state unchanged).
    pub fn release_frame(&mut self, q: usize) -> Result<()> {
        if q >= self.num_qubits {
            return Err(Error::QubitOutOfRange {
                qubit: q,
                num_qubits: self.num_qubits,
            });
        }
        let mut core = self.core.borrow_mut();
        core.flush_qubit(q)?;
        let active = core.any_active();
        drop(core);
        self.active.set(active);
        Ok(())
    }

    /// Flush every frame (physical state unchanged, frames identity).
    pub fn flush(&mut self) -> Result<()> {
        let mut core = self.core.borrow_mut();
        core.flush_all()?;
        drop(core);
        self.active.set(false);
        Ok(())
    }

    fn flush_all_if_active(&self) -> Result<()> {
        if !self.active.get() {
            return Ok(());
        }
        let mut core = self.core.borrow_mut();
        core.flush_all()?;
        drop(core);
        self.active.set(false);
        Ok(())
    }

    /// `F = ⊗ W_{qubits[b]}` in gate-bit order (bit 0 lowest).
    fn frame_product(core: &Core<S>, qubits: &[usize]) -> Result<GateMatrix<S>> {
        let identity = GateMatrix::<S>::identity(2)?;
        let mut f: Option<GateMatrix<S>> = None;
        for &q in qubits.iter().rev() {
            let w = core.frames[q].as_ref().unwrap_or(&identity);
            f = Some(match f {
                None => w.clone(),
                Some(high) => high.kron(w),
            });
        }
        Ok(f.expect("gate has at least one qubit"))
    }
}

impl<S: Scalar> Backend<S> for FramedState<S> {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        super::validate_apply(self.num_qubits, matrix, qubits)?;
        let mut core = self.core.borrow_mut();
        let k = qubits.len();
        if k == 1 {
            let q = qubits[0];
            let composed = match core.frames[q].take() {
                Some(w) => matrix.matmul(&w),
                None => matrix.clone(),
            };
            core.frames[q] = if is_identity(&composed) {
                None
            } else {
                Some(composed)
            };
            core.stats.absorbed_1q += 1;
            let active = core.any_active();
            drop(core);
            self.active.set(active);
            return Ok(());
        }
        let frames_on_gate = qubits.iter().any(|&q| core.frames[q].is_some());
        if !frames_on_gate {
            core.inner.apply(matrix, qubits)?;
            core.stats.passthrough += 1;
            core.note_peak();
            return Ok(());
        }
        if k > FRAME_CONJUGATION_MAX {
            for &q in qubits {
                core.flush_qubit(q)?;
            }
            core.inner.apply(matrix, qubits)?;
            core.stats.passthrough += 1;
            core.note_peak();
            let active = core.any_active();
            drop(core);
            self.active.set(active);
            return Ok(());
        }
        let f = Self::frame_product(&core, qubits)?;
        let conjugated = f.dagger().matmul(matrix).matmul(&f);
        match try_diagonal(&conjugated) {
            Some(diag) => {
                core.inner.apply_diagonal(&diag, qubits)?;
                core.stats.diagonalized += 1;
            }
            None => {
                core.inner.apply(&conjugated, qubits)?;
                core.stats.conjugated += 1;
            }
        }
        core.note_peak();
        Ok(())
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        super::validate_apply_diagonal(self.num_qubits, entries, qubits)?;
        let mut core = self.core.borrow_mut();
        let frames_on_gate = qubits.iter().any(|&q| core.frames[q].is_some());
        if !frames_on_gate {
            core.inner.apply_diagonal(entries, qubits)?;
            core.stats.passthrough += 1;
            core.note_peak();
            return Ok(());
        }
        let k = qubits.len();
        if k == 1 {
            // Build the 2×2 and absorb like any single-qubit gate.
            let m = GateMatrix::from_vec(2, vec![entries[0], S::zero(), S::zero(), entries[1]])?;
            drop(core);
            return self.apply(&m, qubits);
        }
        if k > FRAME_CONJUGATION_MAX {
            for &q in qubits {
                core.flush_qubit(q)?;
            }
            core.inner.apply_diagonal(entries, qubits)?;
            core.stats.passthrough += 1;
            core.note_peak();
            let active = core.any_active();
            drop(core);
            self.active.set(active);
            return Ok(());
        }
        let d = entries.len();
        let mut m = GateMatrix::zeros(d)?;
        for (i, &e) in entries.iter().enumerate() {
            m.set(i, i, e);
        }
        let f = Self::frame_product(&core, qubits)?;
        let conjugated = f.dagger().matmul(&m).matmul(&f);
        match try_diagonal(&conjugated) {
            Some(diag) => {
                core.inner.apply_diagonal(&diag, qubits)?;
                core.stats.diagonalized += 1;
            }
            None => {
                core.inner.apply(&conjugated, qubits)?;
                core.stats.conjugated += 1;
            }
        }
        core.note_peak();
        Ok(())
    }

    fn amplitude(&self, index: u64) -> S {
        if self.flush_all_if_active().is_err() {
            return S::zero();
        }
        self.core.borrow().inner.amplitude(index)
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        if self.flush_all_if_active().is_err() {
            return;
        }
        self.core.borrow().inner.for_each_nonzero(f)
    }

    fn nonzero_count(&self) -> usize {
        if self.flush_all_if_active().is_err() {
            return 0;
        }
        self.core.borrow().inner.nonzero_count()
    }

    fn total_weight(&self) -> f64 {
        if self.flush_all_if_active().is_err() {
            return f64::NAN;
        }
        self.core.borrow().inner.total_weight()
    }

    fn measure(&mut self, qubit: usize, rng: &mut Prng) -> Result<bool> {
        if qubit >= self.num_qubits {
            return Err(Error::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits,
            });
        }
        let mut core = self.core.borrow_mut();
        core.flush_qubit(qubit)?;
        let outcome = core.inner.measure(qubit, rng)?;
        core.note_peak();
        let active = core.any_active();
        drop(core);
        self.active.set(active);
        Ok(outcome)
    }

    fn sample(&self, shots: u64, rng: &mut Prng) -> Result<HashMap<u64, u64>> {
        self.flush_all_if_active()?;
        self.core.borrow().inner.sample(shots, rng)
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        let mut core = self.core.borrow_mut();
        // Flush errors cannot surface through this signature; a frame that
        // fails to apply would already have failed at absorption time.
        let _ = core.flush_qubit(qubit);
        core.inner.project(qubit, outcome, renorm);
        core.note_peak();
        let active = core.any_active();
        drop(core);
        self.active.set(active);
    }

    fn reset(&mut self) {
        let mut core = self.core.borrow_mut();
        core.inner.reset();
        core.frames = vec![None; self.num_qubits];
        core.stats = FrameStats::default();
        core.peak_inner_memory = core.inner.memory_bytes();
        drop(core);
        self.active.set(false);
    }

    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        let mut core = self.core.borrow_mut();
        core.frames = vec![None; self.num_qubits];
        core.inner.load(entries)?;
        core.note_peak();
        drop(core);
        self.active.set(false);
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        let core = self.core.borrow();
        let frame_bytes: usize = core
            .frames
            .iter()
            .flatten()
            .map(|m| std::mem::size_of_val(m.data()))
            .sum();
        core.inner.memory_bytes() + frame_bytes + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
