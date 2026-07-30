//! Braided backend: the state as a **braid word**, materialized only when
//! an amplitude is actually asked for.
//!
//! # The represented class
//!
//! [`crate::braided`] measures that a path through the braid group is
//! simultaneously the address, the stored state and the computation. This
//! backend is that claim put where the crate can price it: the state is
//! held as `(initial basis index, word)` over a fixed realization's
//! generators, and nothing else. Applying a generator is one `push` —
//! `O(1)`, independent of the register width — and applying anything else
//! materializes the word into a dense vector and hands over, exactly as
//! [`AdaptiveState`](crate::backend::AdaptiveState) promotes sparse →
//! dense.
//!
//! # What the measurement says, including the part that does not flatter it
//!
//! Three honest facts, all of them measured in
//! `tests/representation_conformance.rs` rather than argued:
//!
//! * **The word really is sub-exponential storage.** A depth-`d` braid
//!   circuit on any width costs `O(d)` bytes while it stays a word. The
//!   realization's generators are *derived from the strand count*, not
//!   stored per state, which is why [`BraidedState::memory_bytes`]
//!   excludes them — the same reason a dense state does not charge itself
//!   for the gate registry. [`BraidedState::alphabet_bytes`] reports that
//!   shared cost separately so it is never quietly dropped.
//! * **The time axis is where it loses.** Reading a single amplitude
//!   materializes the whole word: `O(d · 4^n)`. The boundary atlas
//!   certifies an axis only when memory *and* wall-clock are both
//!   sub-exponential, so this representation is correctly **not** a
//!   certified simulation — it trades one exponential for another.
//!   Measured on the pure-braid family: memory **Constant** — 280 bytes
//!   at every width — and time **Exponential** at base 5.6.
//! * **The alphabet is not free either.** The generators are derived, but
//!   deriving them costs `(strands−1)·4^n` amplitudes:
//!   [`BraidedState::alphabet_bytes`] measures 28 KiB at 4 qubits, 720
//!   KiB at 6 and 15 MiB at 8. It is shared across every state in the
//!   representation, exactly like a gate registry, which is why it is not
//!   in `memory_bytes` — but it is exponential and it is reported.
//!
//! One thing worth stating because the measurement contradicted the
//! obvious guess: the Majorana generators *are* Clifford, so
//! [`CliffordFramedState`](crate::backend::CliffordFramedState) ought to
//! hold these circuits cheaply — and on this family it reads exponential
//! too (base 1.69). The frame's assumption is about the **gate set**, not
//! about the unitary: generators arriving as raw matrices rather than as
//! named Clifford gates are outside what it can exploit. That is a real
//! limitation of the frame surfaced by adding this axis, not a point in
//! the braided representation's favour.
//!
//! So the value here is not a new advantage. It is that the braided
//! representation now stands in the same conformance and cost harness as
//! every other representation, and reports what it is worth.

use std::cell::RefCell;

use super::{validate_apply, Backend, DenseState};
use crate::backend::dense::DENSE_MAX_QUBITS;
use crate::braided::{fibonacci_generators, majorana_generators};
use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::scalar::C64;

/// Tolerance for recognizing an applied matrix as a realization
/// generator.
const MATCH_TOL: f64 = 1e-12;

/// Which braid realization a [`BraidedState`] is a word in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Realization {
    /// `σ_i = (1 + γ_i γ_{i+1})/√2` on Jordan–Wigner Majoranas: a `B_n`
    /// representation at every even strand count, and Clifford.
    Majorana,
    /// The 2-dimensional Fibonacci `B_3` representation — universal, and
    /// only defined on one qubit's worth of fusion space.
    Fibonacci,
}

impl Realization {
    /// Whether the realization acts on a register of this width, without
    /// building anything.
    pub fn validate_width(self, qubits: usize) -> Result<()> {
        match self {
            Realization::Majorana => {
                if qubits == 0 || 2 * qubits > crate::braided::MAX_STRANDS {
                    return Err(Error::InvalidState(format!(
                        "the Majorana realization needs {} strands, above the {} supported",
                        2 * qubits,
                        crate::braided::MAX_STRANDS
                    )));
                }
                Ok(())
            }
            Realization::Fibonacci => {
                if qubits != 1 {
                    return Err(Error::InvalidState(format!(
                        "the Fibonacci B_3 realization acts on 1 qubit, not {qubits}"
                    )));
                }
                Ok(())
            }
        }
    }

    /// The generators on a register of `qubits` qubits, or an error when
    /// the realization does not act on that width.
    pub fn generators(self, qubits: usize) -> Result<Vec<GateMatrix<C64>>> {
        match self {
            Realization::Majorana => majorana_generators(2 * qubits),
            Realization::Fibonacci => {
                if qubits != 1 {
                    return Err(Error::InvalidState(format!(
                        "the Fibonacci B_3 realization acts on 1 qubit, not {qubits}"
                    )));
                }
                fibonacci_generators()
            }
        }
    }
}

#[derive(Debug)]
enum Repr {
    /// `word` over the realization's generators; letters are `±(i+1)`.
    Word { init: u64, word: Vec<i32> },
    Dense(DenseState<C64>),
}

/// A state held as a braid word over a fixed realization, materialized
/// on demand.
///
/// See the [`braided_state`](crate::backend) module docs for what this
/// buys and what it does not.
#[derive(Debug)]
pub struct BraidedState {
    n: usize,
    realization: Realization,
    /// The generator alphabet, **derived on first need**. A gate that
    /// does not span the whole register cannot be a generator, so a
    /// circuit that never touches the braided class never pays the
    /// `(strands−1)·4^n` derivation at all.
    gens: RefCell<Option<Vec<GateMatrix<C64>>>>,
    repr: Repr,
    /// Materialized amplitudes, produced the first time one is read.
    cache: RefCell<Option<Vec<C64>>>,
    escapes: usize,
}

impl Clone for BraidedState {
    fn clone(&self) -> Self {
        BraidedState {
            n: self.n,
            realization: self.realization,
            gens: RefCell::new(self.gens.borrow().clone()),
            repr: match &self.repr {
                Repr::Word { init, word } => Repr::Word {
                    init: *init,
                    word: word.clone(),
                },
                Repr::Dense(d) => Repr::Dense(d.clone()),
            },
            cache: RefCell::new(self.cache.borrow().clone()),
            escapes: self.escapes,
        }
    }
}

impl BraidedState {
    /// `|0…0⟩` as the empty word over the Majorana realization.
    pub fn new(num_qubits: usize) -> Result<Self> {
        BraidedState::with_realization(num_qubits, Realization::Majorana)
    }

    /// `|0…0⟩` as the empty word over a chosen realization.
    pub fn with_realization(num_qubits: usize, realization: Realization) -> Result<Self> {
        if num_qubits == 0 || num_qubits > DENSE_MAX_QUBITS {
            return Err(Error::TooManyQubits {
                requested: num_qubits,
                max: DENSE_MAX_QUBITS,
            });
        }
        // Validate the width eagerly — a refusal must be immediate and
        // measured — but do not build the matrices yet.
        realization.validate_width(num_qubits)?;
        Ok(BraidedState {
            n: num_qubits,
            realization,
            gens: RefCell::new(None),
            repr: Repr::Word {
                init: 0,
                word: Vec::new(),
            },
            cache: RefCell::new(None),
            escapes: 0,
        })
    }

    /// Which realization the word is written in.
    pub fn realization(&self) -> Realization {
        self.realization
    }

    /// Whether the state is still a word.
    pub fn is_word(&self) -> bool {
        matches!(self.repr, Repr::Word { .. })
    }

    /// The word, while there is one.
    pub fn word(&self) -> Option<&[i32]> {
        match &self.repr {
            Repr::Word { word, .. } => Some(word),
            Repr::Dense(_) => None,
        }
    }

    /// Gates that left the braided class.
    pub fn escapes(&self) -> usize {
        self.escapes
    }

    /// Whether the amplitudes have been materialized from the word.
    pub fn is_materialized(&self) -> bool {
        self.cache.borrow().is_some() || matches!(self.repr, Repr::Dense(_))
    }

    /// Bytes of the **shared** generator alphabet: `(strands−1)·4^n`
    /// amplitudes, derived from the strand count rather than stored per
    /// state, and reported here so the cost is never invisible.
    ///
    /// Computed from the shape, so asking does not build it.
    pub fn alphabet_bytes(&self) -> usize {
        let count = match self.realization {
            Realization::Majorana => 2 * self.n - 1,
            Realization::Fibonacci => 2,
        };
        let dim = 1usize << self.n;
        count * dim * dim * std::mem::size_of::<C64>()
    }

    /// Whether the generator alphabet has actually been derived.
    pub fn alphabet_built(&self) -> bool {
        self.gens.borrow().is_some()
    }

    /// The generators, deriving them if this is the first need.
    fn with_gens<T>(&self, f: impl FnOnce(&[GateMatrix<C64>]) -> T) -> T {
        if self.gens.borrow().is_none() {
            let built = self
                .realization
                .generators(self.n)
                .expect("the width was validated at construction");
            *self.gens.borrow_mut() = Some(built);
        }
        let borrowed = self.gens.borrow();
        f(borrowed.as_ref().expect("just built"))
    }

    /// Drop the materialized amplitudes, returning to word-only storage.
    /// Only meaningful while the state is still a word.
    pub fn forget_amplitudes(&self) {
        *self.cache.borrow_mut() = None;
    }

    /// Match an applied matrix against the realization's generators.
    fn as_letter(&self, matrix: &GateMatrix<C64>, qubits: &[usize]) -> Option<i32> {
        // A generator acts on the whole register, in order. Checking that
        // first is what keeps a non-braided circuit from ever paying for
        // the alphabet.
        if qubits.len() != self.n || qubits.iter().enumerate().any(|(i, &q)| i != q) {
            return None;
        }
        self.with_gens(|gens| {
            for (i, g) in gens.iter().enumerate() {
                if g.approx_eq(matrix, MATCH_TOL) {
                    return Some(i as i32 + 1);
                }
                if g.dagger().approx_eq(matrix, MATCH_TOL) {
                    return Some(-(i as i32 + 1));
                }
            }
            None
        })
    }

    /// Evaluate the word into amplitudes.
    fn evaluate(&self) -> Vec<C64> {
        let Repr::Word { init, word } = &self.repr else {
            return Vec::new();
        };
        let dim = 1usize << self.n;
        let mut v = vec![C64::new(0.0, 0.0); dim];
        v[(*init as usize).min(dim - 1)] = C64::new(1.0, 0.0);
        if word.is_empty() {
            return v;
        }
        self.with_gens(|gens| {
            for &l in word {
                let g = &gens[l.unsigned_abs() as usize - 1];
                let m = if l < 0 { g.dagger() } else { g.clone() };
                let mut out = vec![C64::new(0.0, 0.0); dim];
                for (r, slot) in out.iter_mut().enumerate() {
                    let mut acc = C64::new(0.0, 0.0);
                    for (c, &x) in v.iter().enumerate() {
                        if x.norm() != 0.0 {
                            acc += m.get(r, c) * x;
                        }
                    }
                    *slot = acc;
                }
                v = out;
            }
            v
        })
    }

    fn amplitudes(&self) -> std::cell::Ref<'_, Option<Vec<C64>>> {
        if self.cache.borrow().is_none() {
            let v = self.evaluate();
            *self.cache.borrow_mut() = Some(v);
        }
        self.cache.borrow()
    }

    fn materialize(&mut self) -> Result<()> {
        if self.is_word() {
            let v = {
                let borrowed = self.amplitudes();
                borrowed.clone().unwrap_or_default()
            };
            let mut dense = DenseState::<C64>::new(self.n)?;
            let entries: Vec<(u64, C64)> = v
                .iter()
                .enumerate()
                .filter(|(_, a)| a.norm() != 0.0)
                .map(|(i, &a)| (i as u64, a))
                .collect();
            dense.load(&entries)?;
            self.repr = Repr::Dense(dense);
            *self.cache.borrow_mut() = None;
            self.escapes += 1;
        }
        Ok(())
    }

    fn dense_mut(&mut self) -> Result<&mut DenseState<C64>> {
        self.materialize()?;
        match &mut self.repr {
            Repr::Dense(d) => Ok(d),
            Repr::Word { .. } => unreachable!("materialize leaves a dense repr"),
        }
    }
}

impl Backend<C64> for BraidedState {
    fn name(&self) -> &str {
        "braided"
    }

    fn num_qubits(&self) -> usize {
        self.n
    }

    fn apply(&mut self, matrix: &GateMatrix<C64>, qubits: &[usize]) -> Result<()> {
        validate_apply(self.n, matrix, qubits)?;
        if self.is_word() {
            if let Some(letter) = self.as_letter(matrix, qubits) {
                if let Repr::Word { word, .. } = &mut self.repr {
                    // free reduction is sound in the group
                    if word.last() == Some(&-letter) {
                        word.pop();
                    } else {
                        word.push(letter);
                    }
                }
                *self.cache.borrow_mut() = None;
                return Ok(());
            }
        }
        self.dense_mut()?.apply(matrix, qubits)
    }

    fn amplitude(&self, index: u64) -> C64 {
        match &self.repr {
            Repr::Word { .. } => {
                if index >= 1u64 << self.n {
                    return C64::new(0.0, 0.0);
                }
                let borrowed = self.amplitudes();
                borrowed.as_ref().map_or(C64::new(0.0, 0.0), |v| v[index as usize])
            }
            Repr::Dense(d) => d.amplitude(index),
        }
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, C64)) {
        match &self.repr {
            Repr::Word { .. } => {
                let borrowed = self.amplitudes();
                if let Some(v) = borrowed.as_ref() {
                    for (i, &a) in v.iter().enumerate() {
                        if a.norm() != 0.0 {
                            f(i as u64, a);
                        }
                    }
                }
            }
            Repr::Dense(d) => d.for_each_nonzero(f),
        }
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        if self.materialize().is_err() {
            return;
        }
        if let Repr::Dense(d) = &mut self.repr {
            d.project(qubit, outcome, renorm);
        }
    }

    fn reset(&mut self) {
        self.repr = Repr::Word {
            init: 0,
            word: Vec::new(),
        };
        *self.cache.borrow_mut() = None;
    }

    fn load(&mut self, entries: &[(u64, C64)]) -> Result<()> {
        // A single basis state is still a word (the empty one); anything
        // else is outside the class.
        if entries.len() == 1 && (entries[0].1 - C64::new(1.0, 0.0)).norm() < MATCH_TOL {
            let idx = entries[0].0;
            if idx < 1u64 << self.n {
                self.repr = Repr::Word {
                    init: idx,
                    word: Vec::new(),
                };
                *self.cache.borrow_mut() = None;
                return Ok(());
            }
        }
        self.dense_mut()?.load(entries)
    }

    fn memory_bytes(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        match &self.repr {
            Repr::Word { word, .. } => {
                let cached = self
                    .cache
                    .borrow()
                    .as_ref()
                    .map_or(0, |v| v.len() * std::mem::size_of::<C64>());
                base + word.len() * std::mem::size_of::<i32>() + cached
            }
            Repr::Dense(d) => base + d.memory_bytes(),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_braid_circuit_stays_a_word_and_costs_its_length() {
        let mut s = BraidedState::with_realization(3, Realization::Majorana).unwrap();
        let gens = Realization::Majorana.generators(3).unwrap();
        assert_eq!(gens.len(), 5);
        let all: Vec<usize> = (0..3).collect();
        for k in 0..200 {
            s.apply(&gens[k % gens.len()], &all).unwrap();
        }
        assert!(s.is_word());
        assert_eq!(s.word().unwrap().len(), 200);
        assert_eq!(s.escapes(), 0);
        // the word is the storage while no amplitude has been read
        assert!(!s.is_materialized());
        let word_only = s.memory_bytes();
        assert!(word_only < 1200, "word-only memory {word_only}");
        // reading one amplitude materializes the whole thing
        let _ = s.amplitude(0);
        assert!(s.is_materialized());
        assert!(s.memory_bytes() > word_only);
    }

    #[test]
    fn free_reduction_keeps_the_word_short() {
        let mut s = BraidedState::new(2).unwrap();
        let gens = Realization::Majorana.generators(2).unwrap();
        let all = [0usize, 1];
        for _ in 0..50 {
            s.apply(&gens[0], &all).unwrap();
            s.apply(&gens[0].dagger(), &all).unwrap();
        }
        assert_eq!(s.word().unwrap().len(), 0);
        assert!((s.amplitude(0) - C64::new(1.0, 0.0)).norm() < 1e-12);
    }
}
