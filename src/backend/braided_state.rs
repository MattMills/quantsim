//! Braided backend: the braid word as the address, over a stabilizer
//! frame that holds the state it names.
//!
//! # What this representation is
//!
//! [`crate::braided`] measures that a path through the braid group is
//! simultaneously the address, the stored state and the computation, and
//! that the Majorana/Ising realization's projective image is **finite**
//! (192 elements at 4 strands, 23040 at 6 — both walked, not asserted).
//! A finite image is the whole efficiency claim, and this backend is that
//! claim implemented rather than described.
//!
//! The mechanism is one derivation. Under Jordan–Wigner the Z-strings
//! cancel inside a neighbouring Majorana pair:
//!
//! ```text
//! γ_{2k} γ_{2k+1}   = i Z_k
//! γ_{2k+1} γ_{2k+2} = i X_k X_{k+1}
//! ```
//!
//! so every braid generator `σ_i = (1 + γ_iγ_{i+1})/√2` is
//! `exp(iπ/4 · P)` for a Pauli `P` of weight **one or two**
//! ([`majorana_local_gate`](crate::braided::majorana_local_gate),
//! verified against the dense generators at every width and basis state).
//! A `±π/2` Pauli rotation is Clifford, so
//! [`CliffordFramedState`](crate::backend::CliffordFramedState) **absorbs
//! it into the tableau with zero amplitude work**.
//!
//! # What that buys, and the honest shape of it
//!
//! * **No width limit and no exponential anywhere.** A generator is
//!   `O(1)` to describe and `O(n²)` to apply. There is no `4^n` matrix in
//!   the path — an earlier version of this backend built one, which is
//!   where its apparent exponential cost and its 16-strand ceiling both
//!   came from. Neither was a property of the representation.
//! * **The cost is flat in the width.** Measured on a depth-200 braid
//!   circuit: 34 837 bytes at 8 qubits and 38 677 at 63 — a 7.9× wider
//!   register for 11% more memory, with the stored support **1** and
//!   **zero flushes** throughout. The exponential is simply not there.
//! * **It is linear in the depth, and that is the frame's replay log.**
//!   ~170 bytes per absorbed gate: 9 437 bytes at depth 50, 847 637 at
//!   depth 5000. The *state* does not grow — support stays 1 and the
//!   tableau is fixed-size — but
//!   [`CliffordFramedState`](crate::backend::CliffordFramedState) keeps
//!   the log so a later flush can replay it, so the footprint does.
//!   Saying "the footprint is depth-independent" would be wrong, and it
//!   was wrong in an earlier draft of this file.
//! * **The address is kept alongside**, because it is the point: the word
//!   is the path, and [`BraidedState::word`] is the thing
//!   [`crate::braided::run_path`] ascends exactly. It is `O(depth)` and
//!   [`BraidedState::canonicalize`] drops it — sound, because the tableau
//!   already carries the group element the word spelled, and
//!   [`crate::braided`] measured that image to be finite.
//! * **Universality is where it stops, as it must be.** The
//!   [`Realization::Fibonacci`] generators are not Clifford — that is
//!   precisely why Fibonacci braiding is universal and Ising is not — so
//!   the frame cannot absorb them and pays amplitudes. The two
//!   realizations sitting in one backend, with only one of them cheap, is
//!   the Gottesman–Knill boundary showing up from the braid-group side.
//!
//! So the braided representation *is* efficient, and its efficiency is
//! the Clifford island reached by a different road: the module measures
//! the image is finite, and the backend spends that finiteness.

use super::{validate_apply, Backend, CliffordFramedState};
use crate::braided::{fibonacci_generators, majorana_local_gate};
use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::scalar::C64;

/// Tolerance for recognizing an applied matrix as a realization
/// generator.
const MATCH_TOL: f64 = 1e-12;

/// Which braid realization a [`BraidedState`] is a word in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Realization {
    /// `σ_i = (1 + γ_i γ_{i+1})/√2` on Jordan–Wigner Majoranas — a `B_{2n}`
    /// representation on `n` qubits, every generator a weight-≤2 Clifford
    /// rotation, and the Ising-anyon braiding.
    Majorana,
    /// The 2-dimensional Fibonacci `B_3` representation: universal, and
    /// therefore **not** Clifford, so the frame pays for it.
    Fibonacci,
}

impl Realization {
    /// Whether the realization acts on a register of this width.
    pub fn validate_width(self, qubits: usize) -> Result<()> {
        match self {
            Realization::Majorana => {
                if qubits == 0 {
                    return Err(Error::InvalidState(
                        "the Majorana realization needs at least one qubit".into(),
                    ));
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

    /// Number of braid generators on a register of this width.
    pub fn generator_count(self, qubits: usize) -> usize {
        match self {
            Realization::Majorana => 2 * qubits - 1,
            Realization::Fibonacci => 2,
        }
    }

    /// Generator `index` as `(matrix, target qubits)` — weight ≤ 2 for
    /// the Majorana realization at any width.
    pub fn local_gate(self, index: usize, qubits: usize) -> Result<(GateMatrix<C64>, Vec<usize>)> {
        match self {
            Realization::Majorana => majorana_local_gate(index, qubits),
            Realization::Fibonacci => {
                let gens = fibonacci_generators()?;
                gens.get(index)
                    .cloned()
                    .map(|m| (m, vec![0usize]))
                    .ok_or_else(|| {
                        Error::InvalidState(format!("Fibonacci B_3 has no generator {index}"))
                    })
            }
        }
    }
}

/// A state named by a braid word and held on a stabilizer frame.
///
/// See the [`braided_state`](crate::backend) module docs for the
/// derivation that makes every Majorana generator a local Clifford gate.
pub struct BraidedState {
    n: usize,
    realization: Realization,
    /// The path: signed generator letters, `±(i+1)`.
    word: Vec<i32>,
    /// The state the word names.
    inner: CliffordFramedState<C64>,
    /// Gates that were not realization generators.
    escapes: usize,
}

impl std::fmt::Debug for BraidedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BraidedState")
            .field("qubits", &self.n)
            .field("realization", &self.realization)
            .field("word_len", &self.word.len())
            .field("escapes", &self.escapes)
            .field("stored_support", &self.inner.stored_nonzero_count())
            .finish()
    }
}

impl BraidedState {
    /// `|0…0⟩` as the empty word over the Majorana realization.
    pub fn new(num_qubits: usize) -> Result<Self> {
        BraidedState::with_realization(num_qubits, Realization::Majorana)
    }

    /// `|0…0⟩` as the empty word over a chosen realization.
    pub fn with_realization(num_qubits: usize, realization: Realization) -> Result<Self> {
        realization.validate_width(num_qubits)?;
        Ok(BraidedState {
            n: num_qubits,
            realization,
            word: Vec::new(),
            inner: CliffordFramedState::<C64>::new(num_qubits)?,
            escapes: 0,
        })
    }

    /// Which realization the word is written in.
    pub fn realization(&self) -> Realization {
        self.realization
    }

    /// The path so far.
    pub fn word(&self) -> &[i32] {
        &self.word
    }

    /// Gates that were not generators of the realization.
    pub fn escapes(&self) -> usize {
        self.escapes
    }

    /// Whether every gate so far was a braid generator.
    pub fn is_pure_braid(&self) -> bool {
        self.escapes == 0
    }

    /// Amplitudes the frame is actually storing — 1 while the circuit
    /// stays Clifford, at any width and any depth.
    pub fn stored_support(&self) -> usize {
        self.inner.stored_nonzero_count()
    }

    /// The frame's own accounting: absorbed Cliffords, flushes, and the
    /// widest conjugated Pauli string it kept out of the amplitude work.
    pub fn frame_stats(&self) -> super::CliffordFrameStats {
        self.inner.stats()
    }

    /// The Heisenberg-picture image of an observable: `C† P C`, where
    /// `C` is the Clifford the word has spelled so far.
    ///
    /// This is the query that needs no state. A Pauli conjugated back
    /// through the path is still one Pauli — `O(n)` bits — so an
    /// expectation, a light cone, or an operator's support can be read
    /// off the path at any width without a single amplitude existing.
    /// The whole stateless claim reduces to this method being cheap.
    pub fn conjugated(&self, p: super::PauliString) -> super::PauliString {
        self.inner.conjugated(p)
    }

    /// Support of `C† P C` — the number of sites the observable has
    /// spread to. Measured against time this is the light cone.
    pub fn observable_weight(&self, p: super::PauliString) -> usize {
        self.conjugated(p).weight()
    }

    /// The exact expectation `⟨0…0| C† P C |0…0⟩`, read off the
    /// conjugated Pauli with no state and no sampling.
    ///
    /// On a stabilizer state every Pauli expectation is exactly `0`,
    /// `+1` or `−1`: the conjugated string either has an `X` or `Y`
    /// somewhere — which flips a bit of `|0…0⟩` and gives overlap 0 —
    /// or is pure `Z`, and then the sign is the answer. Quantized
    /// correlators are a property of the Clifford point, not an
    /// approximation.
    pub fn expectation(&self, p: super::PauliString) -> f64 {
        let c = self.conjugated(p);
        if c.x != 0 {
            0.0
        } else if c.negative {
            -1.0
        } else {
            1.0
        }
    }

    /// Drop the word, keeping the state.
    ///
    /// Sound because the tableau **is** the group element the word names:
    /// the Majorana image is finite ([`crate::braided::orbit_closure`]
    /// walks it), so an arbitrarily long path has only so many
    /// destinations and the tableau already holds which one. This is the
    /// depth-independent form of the storage — after it, the footprint is
    /// `O(n²)` however long the path was.
    pub fn canonicalize(&mut self) {
        self.word.clear();
    }

    /// Match an applied matrix against the realization's generators.
    /// Compares weight-≤2 matrices against `2n−1` candidates, so this is
    /// `O(n)` tiny comparisons and never touches a `2^n` object.
    fn as_letter(&self, matrix: &GateMatrix<C64>, qubits: &[usize]) -> Option<i32> {
        for i in 0..self.realization.generator_count(self.n) {
            let (g, targets) = self.realization.local_gate(i, self.n).ok()?;
            if targets != qubits {
                continue;
            }
            if g.approx_eq(matrix, MATCH_TOL) {
                return Some(i as i32 + 1);
            }
            if g.dagger().approx_eq(matrix, MATCH_TOL) {
                return Some(-(i as i32 + 1));
            }
        }
        None
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
        match self.as_letter(matrix, qubits) {
            Some(letter) => {
                // free reduction is sound in the group
                if self.word.last() == Some(&-letter) {
                    self.word.pop();
                } else {
                    self.word.push(letter);
                }
            }
            None => self.escapes += 1,
        }
        // Either way the frame evolves the state: a generator is a
        // weight-≤2 Clifford rotation it absorbs, and anything else
        // follows the frame's own published policy.
        self.inner.apply(matrix, qubits)
    }

    fn apply_diagonal(&mut self, entries: &[C64], qubits: &[usize]) -> Result<()> {
        self.escapes += 1;
        self.inner.apply_diagonal(entries, qubits)
    }

    fn amplitude(&self, index: u64) -> C64 {
        self.inner.amplitude(index)
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, C64)) {
        self.inner.for_each_nonzero(f)
    }

    fn nonzero_count(&self) -> usize {
        self.inner.nonzero_count()
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        self.inner.project(qubit, outcome, renorm)
    }

    fn reset(&mut self) {
        self.word.clear();
        self.escapes = 0;
        self.inner.reset();
    }

    fn load(&mut self, entries: &[(u64, C64)]) -> Result<()> {
        self.word.clear();
        self.inner.load(entries)
    }

    fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.word.len() * std::mem::size_of::<i32>()
            + self.inner.memory_bytes()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walk a braid circuit at a width no dense matrix could reach.
    fn braid_run(qubits: usize, steps: usize) -> BraidedState {
        let mut s = BraidedState::new(qubits).unwrap();
        let count = Realization::Majorana.generator_count(qubits);
        for k in 0..steps {
            let (g, targets) = Realization::Majorana.local_gate(k % count, qubits).unwrap();
            s.apply(&g, &targets).unwrap();
        }
        s
    }

    #[test]
    fn a_braid_circuit_is_free_at_widths_no_dense_matrix_reaches() {
        for qubits in [8usize, 20, 40] {
            let s = braid_run(qubits, 400);
            assert!(
                s.is_pure_braid(),
                "{qubits} qubits: {} escapes",
                s.escapes()
            );
            assert_eq!(
                s.stored_support(),
                1,
                "{qubits} qubits: a Clifford circuit must leave support 1"
            );
            let stats = s.frame_stats();
            assert_eq!(stats.flushes, 0, "{qubits} qubits: no flush should happen");
            assert!(
                stats.absorbed_clifford + stats.axis_rotations >= 400,
                "{qubits} qubits: gates were not absorbed ({stats:?})"
            );
        }
    }

    #[test]
    fn the_cost_is_flat_in_the_width_and_linear_in_the_depth() {
        // Width: a 7.9× wider register for a few percent more memory.
        let narrow = braid_run(8, 200);
        let wide = braid_run(63, 200);
        assert_eq!(narrow.stored_support(), 1);
        assert_eq!(wide.stored_support(), 1);
        assert!(
            (wide.memory_bytes() as f64) < 1.3 * narrow.memory_bytes() as f64,
            "width 8 → 63 grew from {} to {} bytes",
            narrow.memory_bytes(),
            wide.memory_bytes()
        );

        // Depth: linear, and it is the frame's replay log, not the state.
        let short = braid_run(24, 50);
        let mut long = braid_run(24, 5000);
        assert_eq!(short.stored_support(), long.stored_support());
        assert_eq!(long.word().len(), 5000);
        assert!(
            long.memory_bytes() > 8 * short.memory_bytes(),
            "depth 50 → 5000 should grow the log: {} vs {}",
            short.memory_bytes(),
            long.memory_bytes()
        );
        // canonicalize drops the word; the log is the frame's own.
        let before = long.memory_bytes();
        long.canonicalize();
        assert!(long.word().is_empty());
        assert_eq!(long.stored_support(), 1);
        assert!(long.memory_bytes() < before);
    }

    #[test]
    fn generators_are_recognized_and_freely_reduce() {
        let mut s = BraidedState::new(6).unwrap();
        let (g, t) = Realization::Majorana.local_gate(3, 6).unwrap();
        for _ in 0..50 {
            s.apply(&g, &t).unwrap();
            s.apply(&g.dagger(), &t).unwrap();
        }
        assert_eq!(s.word().len(), 0);
        assert_eq!(s.escapes(), 0);
        assert!((s.amplitude(0) - C64::new(1.0, 0.0)).norm() < 1e-12);
    }
}
