//! Systems of `n` twisted polarities, and the exact sector of a
//! quantum state that pairwise-local polarity data can hold.
//!
//! A **polarity** is an inclusion/exclusion axis: a generator `j` with
//! `j² = ±1` splitting things into what counts `+` and what counts `−`.
//! A **system** of them is `n` such generators together with a **twist**
//! — a choice, for each pair, of whether they commute or anticommute.
//! The twist is the whole story, so it is the object this module
//! measures.
//!
//! Formally a [`PolaritySystem`] is the twisted group algebra
//! `ℝ^τ[F₂ⁿ]`: basis elements are subsets of the generators (polarity
//! **monomials**), multiplication is symmetric difference of subsets
//! carrying a sign that the twist determines exactly. Real dimension
//! `2ⁿ` — [`Polarity<N>`](crate::scalar::Polarity) is the fully twisted
//! case turned into an amplitude type.
//!
//! ## What the twist decides
//!
//! Write the twist as an alternating bilinear form `B` on `F₂ⁿ`
//! (`B(a,b) = 0` iff the monomials `a` and `b` commute). Everything
//! follows from its **rank** `r`, and all of it is measured by brute
//! force in `tests/polarity.rs` rather than quoted:
//!
//! * the **centre** has dimension `2^{n−r}` — the untwisted directions,
//! * a **maximal isotropic** set (pairwise-commuting monomials) is a
//!   subgroup of size `2^{n−r/2}`,
//! * and the algebra factors as `2^{n−r/2} × 2^{r/2}`: an abelian part
//!   that is *simultaneously diagonalizable*, and an irreducible matrix
//!   block of side `2^{r/2}`.
//!
//! That factorization is the answer to the natural hope. An abelian
//! system of polarities is simultaneously diagonalizable, so its state
//! is `n − r/2` independent **sign bits** — genuinely local, `O(n)`
//! storage, each polarity's value readable on its own. Everything
//! outside costs `2^{r/2}`. **The twist rank is exactly the price of
//! non-locality**, and it is a dial: [`PolaritySystem::partial`] builds
//! systems anywhere between untwisted (fully local, `2ⁿ` independent
//! sectors, no correlation) and fully twisted (one matrix block, no
//! local part at all).
//!
//! ## Where quantum states actually sit
//!
//! [`PolaritySystem::pauli`] builds the polarity system of the `n`-qubit
//! Pauli group: `2n` generators (`X_q`, `Z_q`), twisted exactly where
//! `X_q` meets `Z_q`. Measured: rank `2n` — maximally twisted, centre
//! trivial — and maximal isotropic subgroups of size `2ⁿ`. Those
//! subgroups *are* stabilizer groups, and their `n` independent sign
//! bits are the `O(n²)` description Gottesman–Knill runs on (this crate
//! already ships it as
//! [`CliffordFramedState`](crate::backend::CliffordFramedState)). So the
//! locally-storable sector of polarity data is real, it is large, and it
//! is already the best-known classical island.
//!
//! ## The obstruction, measured
//!
//! It is tempting to go further and ask for the polarities of *each
//! qubit against each qubit* — an `O(n²)` table of pairwise
//! inclusion/exclusion relations, held locally, standing in for the
//! state. [`pairwise_signature`] builds exactly that object: every
//! one-body and two-body Pauli expectation, which is the complete
//! content of every two-qubit reduced density matrix — `3n + 9·C(n,2)`
//! real numbers, everything a pairwise-local description could possibly
//! know.
//!
//! It is not enough, and the counterexample is sharp rather than
//! asymptotic. [`ghz_sign_obstruction`] measures it: for `n ≥ 3` the
//! **orthogonal** states `(|0…0⟩ + |1…1⟩)/√2` and
//! `(|0…0⟩ − |1…1⟩)/√2` have **identical** pairwise signatures — every
//! one of those numbers agrees to ~1e-16 — while their overlap is
//! exactly 0. No pairwise-local representation can distinguish them,
//! because the only observable that does is the `n`-body correlator
//! `⟨X^{⊗n}⟩ = ±1`. At `n = 2` the same pair *is* separable by
//! `⟨XX⟩`, so the obstruction switches on exactly at three qubits, and
//! the measurement shows it switching on.
//!
//! The sting is that both states are *stabilizer* states, so this is not
//! a case of pairwise data failing on something exotic. It fails inside
//! the very sector that is efficiently describable — because the
//! efficient description is `O(n²)` in *size* but not **pairwise in
//! structure**: GHZ's stabilizer group needs a generator of weight `n`
//! ([`stabilizer_weight_profile`] measures the profile as
//! `[n, 2, 2, …]`), and dropping it leaves a description that two
//! orthogonal states share.
//!
//! So: polarity systems make the local sector precise and dial-able, and
//! they say exactly what it costs to leave it (`2^{r/2}`). What they do
//! not do is make entanglement pairwise. The non-local part is a
//! measured quantity, not an accounting artifact.

use crate::backend::{pauli_expectation, Backend, DenseState};
use crate::error::{Error, Result};
use crate::gates::Pauli;
use crate::scalar::C64;

/// Widest polarity system this module will enumerate by brute force.
/// The measurements here are `O(4ⁿ)` in the generator count, and the
/// point of them is exactness, not scale.
pub const MAX_GENERATORS: usize = 14;

/// `n` polarity generators with a twist: which pairs anticommute, and
/// what each one squares to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolaritySystem {
    generators: usize,
    /// `twist[a]` has bit `b` set when `j_a j_b = −j_b j_a`.
    twist: Vec<u64>,
    /// `+1` or `−1` per generator.
    squares: Vec<i8>,
}

impl PolaritySystem {
    /// Build from an explicit anticommuting-pair list. Generators
    /// listed in `negative_squares` have `j² = −1`; the rest `j² = +1`.
    pub fn new(
        generators: usize,
        anticommuting: &[(usize, usize)],
        negative_squares: &[usize],
    ) -> Result<Self> {
        if generators == 0 || generators > MAX_GENERATORS {
            return Err(Error::InvalidState(format!(
                "a polarity system needs 1..={MAX_GENERATORS} generators; got {generators}"
            )));
        }
        let mut twist = vec![0u64; generators];
        for &(a, b) in anticommuting {
            if a >= generators || b >= generators {
                return Err(Error::InvalidState(format!(
                    "anticommuting pair ({a}, {b}) is out of range for {generators} polarities"
                )));
            }
            if a == b {
                return Err(Error::InvalidState(
                    "a polarity commutes with itself; ({a}, {a}) is not a twist".into(),
                ));
            }
            twist[a] |= 1 << b;
            twist[b] |= 1 << a;
        }
        let mut squares = vec![1i8; generators];
        for &g in negative_squares {
            if g >= generators {
                return Err(Error::InvalidState(format!(
                    "generator {g} is out of range for {generators} polarities"
                )));
            }
            squares[g] = -1;
        }
        Ok(PolaritySystem {
            generators,
            twist,
            squares,
        })
    }

    /// No twist at all: `n` commuting polarities. Fully local and
    /// completely uncorrelated — `2ⁿ` independent sectors.
    pub fn untwisted(generators: usize) -> Result<Self> {
        PolaritySystem::new(generators, &[], &[])
    }

    /// Every distinct pair anticommutes — the system
    /// [`Polarity<N>`](crate::scalar::Polarity) realizes as an
    /// amplitude type.
    pub fn fully_twisted(generators: usize) -> Result<Self> {
        let pairs: Vec<(usize, usize)> = (0..generators)
            .flat_map(|a| ((a + 1)..generators).map(move |b| (a, b)))
            .collect();
        PolaritySystem::new(generators, &pairs, &[])
    }

    /// The twist dial: the first `twisted_pairs` disjoint pairs
    /// `(0,1), (2,3), …` anticommute and everything else commutes, so
    /// the rank is exactly `2·twisted_pairs` and the local sector
    /// shrinks one bit at a time.
    pub fn partial(generators: usize, twisted_pairs: usize) -> Result<Self> {
        if 2 * twisted_pairs > generators {
            return Err(Error::InvalidState(format!(
                "{twisted_pairs} disjoint twisted pairs need {} generators, not {generators}",
                2 * twisted_pairs
            )));
        }
        let pairs: Vec<(usize, usize)> = (0..twisted_pairs).map(|k| (2 * k, 2 * k + 1)).collect();
        PolaritySystem::new(generators, &pairs, &[])
    }

    /// The polarity system of the `n`-qubit Pauli group: generators
    /// `X_q` (index `2q`) and `Z_q` (index `2q+1`), anticommuting
    /// exactly when they sit on the same qubit.
    pub fn pauli(qubits: usize) -> Result<Self> {
        let pairs: Vec<(usize, usize)> = (0..qubits).map(|q| (2 * q, 2 * q + 1)).collect();
        PolaritySystem::new(2 * qubits, &pairs, &[])
    }

    /// Number of polarity generators.
    pub fn generators(&self) -> usize {
        self.generators
    }

    /// Real dimension of the algebra: `2ⁿ`.
    pub fn dimension(&self) -> usize {
        1 << self.generators
    }

    /// Product of two polarity monomials: the symmetric difference of
    /// the generator sets, with the sign the twist forces.
    pub fn product(&self, a: u64, b: u64) -> (u64, f64) {
        let mut mask = a;
        let mut sign = 1.0f64;
        for g in 0..self.generators {
            if (b >> g) & 1 == 0 {
                continue;
            }
            // Carry j_g left past every higher-indexed generator it
            // anticommutes with.
            let higher = mask & !((1u64 << (g + 1)) - 1);
            if (higher & self.twist[g]).count_ones() % 2 == 1 {
                sign = -sign;
            }
            if (mask >> g) & 1 == 1 {
                if self.squares[g] < 0 {
                    sign = -sign;
                }
                mask &= !(1u64 << g);
            } else {
                mask |= 1u64 << g;
            }
        }
        (mask, sign)
    }

    /// The alternating form: `false` when the two monomials commute.
    /// Bilinear over `F₂`, which is why the commuting sets below are
    /// subgroups rather than merely maximal sets.
    pub fn twisted(&self, a: u64, b: u64) -> bool {
        let mut parity = 0u32;
        for g in 0..self.generators {
            if (a >> g) & 1 == 1 {
                parity += (b & self.twist[g]).count_ones();
            }
        }
        parity % 2 == 1
    }

    /// Whether two polarity monomials commute.
    pub fn commutes(&self, a: u64, b: u64) -> bool {
        !self.twisted(a, b)
    }

    /// `F₂` rank of the twist form. Always even.
    pub fn twist_rank(&self) -> usize {
        let mut rows: Vec<u64> = self.twist.clone();
        let mut rank = 0usize;
        let mut pivot_col = 0usize;
        while pivot_col < self.generators && rank < rows.len() {
            if let Some(r) = (rank..rows.len()).find(|&r| (rows[r] >> pivot_col) & 1 == 1) {
                rows.swap(rank, r);
                let pivot = rows[rank];
                for (other, row) in rows.iter_mut().enumerate() {
                    if other != rank && (*row >> pivot_col) & 1 == 1 {
                        *row ^= pivot;
                    }
                }
                rank += 1;
            }
            pivot_col += 1;
        }
        rank
    }

    /// The monomials that commute with everything — the centre.
    pub fn center(&self) -> Vec<u64> {
        (0..1u64 << self.generators)
            .filter(|&m| (0..self.generators).all(|g| self.commutes(m, 1 << g)))
            .collect()
    }

    /// A maximal pairwise-commuting set of monomials. Because the form
    /// is bilinear this is automatically a subgroup, so its size is a
    /// power of two and its `log₂` is the number of independent sign
    /// bits a state in the abelian sector needs.
    pub fn maximal_isotropic(&self) -> Vec<u64> {
        let mut chosen: Vec<u64> = Vec::new();
        for m in 0..1u64 << self.generators {
            if chosen.iter().all(|&c| self.commutes(m, c)) {
                chosen.push(m);
            }
        }
        chosen
    }

    /// The measured structure of the system.
    pub fn structure(&self) -> PolarityStructure {
        let rank = self.twist_rank();
        let n = self.generators;
        PolarityStructure {
            polarities: n,
            dimension: 1 << n,
            twist_rank: rank,
            center_dimension: 1 << (n - rank),
            local_bits: n - rank / 2,
            isotropic_size: 1 << (n - rank / 2),
            matrix_block: 1 << (rank / 2),
        }
    }
}

/// What a polarity system's twist buys and what it costs.
///
/// The identity that matters: `isotropic_size × matrix_block =
/// dimension`. The first factor is storable as independent sign bits —
/// local, `O(n)` — and the second is the irreducible block that is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolarityStructure {
    /// Generator count.
    pub polarities: usize,
    /// `2ⁿ`.
    pub dimension: usize,
    /// `F₂` rank of the twist form (even).
    pub twist_rank: usize,
    /// `2^{n−rank}` — the untwisted directions.
    pub center_dimension: usize,
    /// `n − rank/2` independent sign bits: the local budget.
    pub local_bits: usize,
    /// `2^{local_bits}` — size of a maximal commuting subgroup.
    pub isotropic_size: usize,
    /// `2^{rank/2}` — the irreducible matrix block, the measured price
    /// of the twist.
    pub matrix_block: usize,
}

// ── the pairwise-locality question ───────────────────────────────────

/// `((a, b), ⟨P_a Q_b⟩)` for one qubit pair, indexed `[P][Q]` over the
/// three Pauli axes.
pub type TwoBodyBlock = ((usize, usize), [[f64; 3]; 3]);

/// Every one- and two-body Pauli expectation of a state: the complete
/// content of all two-qubit reduced density matrices, and therefore
/// everything any pairwise-local description could know.
#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseSignature {
    /// Register width.
    pub num_qubits: usize,
    /// `⟨X_a⟩, ⟨Y_a⟩, ⟨Z_a⟩` per qubit.
    pub one_body: Vec<[f64; 3]>,
    /// `((a, b), ⟨P_a Q_b⟩)` for `a < b`, indexed `[P][Q]` over X, Y, Z.
    pub two_body: Vec<TwoBodyBlock>,
}

impl PairwiseSignature {
    /// How many real numbers the signature holds: `3n + 9·C(n,2)`,
    /// quadratic in the width.
    pub fn values(&self) -> usize {
        3 * self.num_qubits + 9 * self.two_body.len()
    }

    /// Largest disagreement with another signature of the same width.
    pub fn max_deviation(&self, other: &PairwiseSignature) -> f64 {
        if self.num_qubits != other.num_qubits {
            return f64::INFINITY;
        }
        let mut worst = 0.0f64;
        for (a, b) in self.one_body.iter().zip(&other.one_body) {
            for k in 0..3 {
                worst = worst.max((a[k] - b[k]).abs());
            }
        }
        for ((pa, ma), (pb, mb)) in self.two_body.iter().zip(&other.two_body) {
            if pa != pb {
                return f64::INFINITY;
            }
            for p in 0..3 {
                for q in 0..3 {
                    worst = worst.max((ma[p][q] - mb[p][q]).abs());
                }
            }
        }
        worst
    }
}

const AXES: [Pauli; 3] = [Pauli::X, Pauli::Y, Pauli::Z];

/// Measure the full pairwise signature of a state.
pub fn pairwise_signature(state: &dyn Backend<C64>) -> Result<PairwiseSignature> {
    let n = state.num_qubits();
    let mut one_body = Vec::with_capacity(n);
    for q in 0..n {
        let mut row = [0.0f64; 3];
        for (k, &p) in AXES.iter().enumerate() {
            row[k] = pauli_expectation(state, &[(q, p)])?.re;
        }
        one_body.push(row);
    }
    let mut two_body = Vec::new();
    for a in 0..n {
        for b in (a + 1)..n {
            let mut block = [[0.0f64; 3]; 3];
            for (i, &p) in AXES.iter().enumerate() {
                for (j, &q) in AXES.iter().enumerate() {
                    block[i][j] = pauli_expectation(state, &[(a, p), (b, q)])?.re;
                }
            }
            two_body.push(((a, b), block));
        }
    }
    Ok(PairwiseSignature {
        num_qubits: n,
        one_body,
        two_body,
    })
}

/// The measured limit of pairwise-local polarity data.
#[derive(Debug, Clone, PartialEq)]
pub struct SignObstruction {
    /// Register width.
    pub num_qubits: usize,
    /// Real numbers a pairwise description holds, `3n + 9·C(n,2)`.
    pub pairwise_values: usize,
    /// Real numbers a state vector holds, `2^{n+1}`.
    pub state_values: usize,
    /// Largest disagreement between the two states' pairwise
    /// signatures. Zero means pairwise data cannot tell them apart.
    pub signature_deviation: f64,
    /// `|⟨ψ₊|ψ₋⟩|` — zero means they are orthogonal, i.e. perfectly
    /// distinguishable by *some* measurement.
    pub overlap: f64,
    /// `⟨X^{⊗n}⟩` on each state: the `n`-body correlator that does
    /// separate them.
    pub global_correlator: (f64, f64),
    /// Whether pairwise data is provably insufficient here.
    pub pairwise_blind: bool,
}

/// Measure whether pairwise-local data can distinguish the two GHZ
/// sign states `(|0…0⟩ ± |1…1⟩)/√2`.
///
/// They are orthogonal pure states, so *something* separates them. The
/// question is whether anything pairwise does — and from `n = 3` up,
/// nothing does: the deviation between their full pairwise signatures
/// is zero to machine precision while the `n`-body correlator
/// `⟨X^{⊗n}⟩` reads `+1` and `−1`.
pub fn ghz_sign_obstruction(num_qubits: usize) -> Result<SignObstruction> {
    if !(2..=16).contains(&num_qubits) {
        return Err(Error::InvalidState(format!(
            "the GHZ sign pair needs 2..=16 qubits; got {num_qubits}"
        )));
    }
    let top = (1u64 << num_qubits) - 1;
    let amp = std::f64::consts::FRAC_1_SQRT_2;
    let build = |sign: f64| -> Result<DenseState<C64>> {
        let mut state = DenseState::<C64>::new(num_qubits)?;
        state.load(&[(0, C64::new(amp, 0.0)), (top, C64::new(sign * amp, 0.0))])?;
        Ok(state)
    };
    let plus = build(1.0)?;
    let minus = build(-1.0)?;

    let sig_plus = pairwise_signature(&plus)?;
    let sig_minus = pairwise_signature(&minus)?;
    let signature_deviation = sig_plus.max_deviation(&sig_minus);

    let mut overlap = C64::new(0.0, 0.0);
    plus.for_each_nonzero(&mut |i, a| overlap += a.conj() * minus.amplitude(i));

    let all_x: Vec<(usize, Pauli)> = (0..num_qubits).map(|q| (q, Pauli::X)).collect();
    let global_correlator = (
        pauli_expectation(&plus, &all_x)?.re,
        pauli_expectation(&minus, &all_x)?.re,
    );

    Ok(SignObstruction {
        num_qubits,
        pairwise_values: sig_plus.values(),
        state_values: 2 << num_qubits,
        signature_deviation,
        overlap: overlap.norm(),
        pairwise_blind: signature_deviation < 1e-12 && overlap.norm() < 1e-12,
        global_correlator,
    })
}

/// Pauli weights of a canonical GHZ stabilizer generating set, sorted
/// descending: `[n, 2, 2, …, 2]`.
///
/// The efficient description of a stabilizer state is `O(n²)` in size
/// but not pairwise in *structure* — this is the generator that makes
/// it so, and dropping it is exactly what leaves
/// [`ghz_sign_obstruction`] two states sharing one description.
pub fn stabilizer_weight_profile(num_qubits: usize) -> Result<Vec<usize>> {
    if num_qubits < 2 {
        return Err(Error::InvalidState(
            "a GHZ stabilizer group needs at least two qubits".into(),
        ));
    }
    // X^{⊗n}, then Z_a Z_{a+1} for a = 0 .. n−2.
    let mut weights = vec![num_qubits];
    weights.extend(std::iter::repeat(2).take(num_qubits - 1));
    weights.sort_unstable_by(|a, b| b.cmp(a));
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_twist_rank_controls_the_local_budget() {
        for pairs in 0..=3 {
            let system = PolaritySystem::partial(6, pairs).unwrap();
            let s = system.structure();
            assert_eq!(s.twist_rank, 2 * pairs);
            assert_eq!(s.isotropic_size * s.matrix_block, s.dimension);
            assert_eq!(s.center_dimension, 1 << (6 - 2 * pairs));
        }
    }

    #[test]
    fn measured_structure_matches_brute_force() {
        for system in [
            PolaritySystem::untwisted(4).unwrap(),
            PolaritySystem::partial(5, 2).unwrap(),
            PolaritySystem::fully_twisted(4).unwrap(),
            PolaritySystem::pauli(3).unwrap(),
        ] {
            let s = system.structure();
            assert_eq!(system.center().len(), s.center_dimension);
            assert_eq!(system.maximal_isotropic().len(), s.isotropic_size);
        }
    }

    #[test]
    fn the_pauli_system_is_maximally_twisted() {
        for n in 1..=5 {
            let s = PolaritySystem::pauli(n).unwrap().structure();
            assert_eq!(s.polarities, 2 * n);
            assert_eq!(s.twist_rank, 2 * n);
            assert_eq!(s.center_dimension, 1);
            // A maximal commuting subgroup has 2^n elements: a
            // stabilizer group, n independent sign bits.
            assert_eq!(s.local_bits, n);
            assert_eq!(s.isotropic_size, 1 << n);
            assert_eq!(s.matrix_block, 1 << n);
        }
    }
}
