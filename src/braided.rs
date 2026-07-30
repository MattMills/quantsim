//! **Braided boundary encoding**: a register held as `n` mutually
//! encoding boundaries, navigated by a periodic word, with the path, the
//! address and the stored state being one object.
//!
//! # The mutual encoding, and what is actually true about it
//!
//! Split a pure register into a region `A` and its complement `B`. The
//! Schmidt decomposition makes each side a **boundary encoding** of the
//! other, and the sense in which it does is exact and measurable
//! ([`mutual_encoding`]):
//!
//! * the reduced states `ρ_A` and `ρ_B` — computed independently, from
//!   different-sized matrices — have **identical spectra**, measured to
//!   ~1e-16;
//! * `ρ_A` **alone** purifies to a state whose own complement is
//!   spectrally identical to the true `B` ([`MutualEncoding::purification_deviation`]).
//!   Nothing about `B` was consulted. That is the holographic claim in
//!   its checkable form: the boundary of `A` determines `B` up to a
//!   unitary on `B`, and only up to that;
//! * the encoding is **dimension-reducing** exactly when the rank is:
//!   `rank·(2^{|A|} + 2^{|B|})` numbers instead of `2^{|A|+|B|}`.
//!
//! The last point is where enthusiasm has to stop, and this module says
//! so in the type: [`MutualEncoding::stored_entries`] is a measurement,
//! and for a generic state the rank is maximal and there is no saving at
//! all. Boundary encoding is dimension-reducing *when the state is
//! structured*, which is the same condition every other representation
//! in this crate is priced against — it is
//! [`MpsState`](crate::backend::MpsState)'s bond dimension seen from the
//! other side, and the [`bounds`](crate::bounds) atlas already measures
//! that axis. What is new here is not the storage. It is the navigation.
//!
//! # Braiding is the navigation, and it is decidable
//!
//! With `n` boundaries the encodings do not commute — pushing boundary 2
//! through 3 through 1 is not pushing 3 through 1 through 2 — and the
//! group that tracks exactly that non-commutation is the braid group
//! `B_n`. The **Artin action** ([`BraidWord::artin_images`]) is the
//! mutual encoding written as an automorphism of the free group:
//!
//! ```text
//! σ_i:  x_i ↦ x_i x_{i+1} x_i^{-1},   x_{i+1} ↦ x_i
//! ```
//!
//! One generator's image is literally the other generator conjugated by
//! it. Artin's theorem says this action is **faithful**, so braid words
//! are comparable by comparing free-group images — [`BraidWord::equals`]
//! decides path identity exactly, with no normal form to trust. That
//! makes the infinite graph navigable: [`cayley_ball`] measures its
//! growth by deduplicating words into genuine group elements (`B_3` out
//! to radius 6: 1, 5, 17, 47, 115, 263, 577, with the ratio falling
//! from 5.00 toward ≈2.2 as the relations bite — measured, not quoted).
//!
//! # The periodic word, and how many distinct paths there are
//!
//! A [`PeriodicPath`] is a finite period over an `n`-letter alphabet
//! specifying an infinite path — the finite-representation-of-an-infinite-
//! object move applied to the navigation itself. Two periods that differ
//! by rotation are the same orbit, so the orbit types are **necklaces**
//! and the primitive ones are **Lyndon words**; [`necklace_count`] and
//! [`lyndon_count`] are the Burnside and Möbius closed forms, verified
//! against brute-force enumeration. [`lyndon_factorization`] is Duval's
//! algorithm, and its output is the unique non-increasing factorization
//! that indexes a basis of the free Lie algebra.
//!
//! # The bracket, measured rather than posited
//!
//! "What is the geometric residue of folding `A` through `B` versus `B`
//! through `A`" has an exact answer: it is the **Lie bracket**, and
//! [`commutator_residual`] measures the sense in which that is true.
//! The group commutator of two flows agrees with `exp(ε²[A,B])` to
//! order `ε³` — measured by fitting the residual's power law, which
//! comes out at 3.00 — and [`bch_residual`] measures the next term the
//! same way, at 3.97 for a generic pair. So the bracket is not a formal
//! stand-in for the geometry; it *is* the leading residue, to a measured
//! order.
//!
//! The fit is also sensitive enough to catch a degeneracy rather than
//! paper over it: for the `su(2)` pair `iX, iZ` the order-4 BCH term
//! `[B,[A,[A,B]]]` vanishes identically, and the measured exponent comes
//! out at 5.01 instead of 4. The example runs both pairs side by side.
//!
//! The formal algebra is free ([`free_lie_dim`] is the Witt formula, and
//! equals the Lyndon count), but no finite-dimensional realization is.
//! [`realized_rank`] measures the collapse against
//! [`majorana_bilinears`] — the infinitesimal folds `γ_i γ_{i+1}` whose
//! exponentials *are* the braid generators. On 6 strands the free Lie
//! algebra on 5 generators reaches cumulative dimension 5, 15, 55, 205,
//! 829 through degrees 1…5, while the realization reaches 5, 9, 12, 14,
//! **15** and stops: the whole infinite path algebra lands inside
//! `so(6)`, whose dimension is 15. **The collapse is not a defect of the
//! realization; it is where the computation's bound lives.**
//!
//! # What closes and what does not
//!
//! Two unitary realizations of the braiding are built and their relations
//! verified on the actual matrices, to ~1e-16, rather than assumed:
//!
//! * [`majorana_generators`] — `σ_i = (1 + γ_i γ_{i+1})/√2` on
//!   Jordan–Wigner Majoranas, a genuine `B_n` representation for every
//!   even strand count, and the Ising-anyon braiding;
//! * [`fibonacci_generators`] — the 2-dimensional `B_3` representation
//!   with the golden-ratio `F`-matrix.
//!
//! [`orbit_closure`] then measures the thing that decides whether the
//! path can *be* the storage. The Majorana image is **finite at every
//! strand count**, and the closure is walked rather than asserted: 4
//! strands close at 192 projective elements at radius 7, 6 strands at
//! 23040 at radius 16. An arbitrarily long Ising path is therefore
//! stored in `O(1)` — its address is a group element and there are only
//! so many. The Fibonacci ball is still growing at a ratio near 1.9 when
//! the search stops, which is evidence and not proof, so the report says
//! "did not close within the radius" and never "infinite".
//!
//! That is the same boundary this crate already measures from the other
//! side: Ising braiding is Clifford, and
//! [`CliffordFramedState`](crate::backend::CliffordFramedState) simulates
//! it in `size²`; Fibonacci braiding is universal, and nothing does.
//!
//! # The path is the address — where that is true, exactly
//!
//! [`RecursionLedger`] measures it per path. Every path **ascends
//! exactly** (deviation ~1e-13): the descent erased nothing, because the
//! recursion never flattened the geometry into symbols. Whether the
//! *storage* is `O(1)` is a separate question, and the ledger answers it
//! decisively with [`projective_order`] rather than by counting states —
//! a state count at finite tolerance saturates for a dense orbit too, and
//! would have reported closure where there is none.
//!
//! Measured, per period, in the Fibonacci realization: `01` closes at
//! order 3, `001` at 2, `0110` at 5, `01011` at 10 — and `0111001`
//! **does not close at all** within 10⁵ periods. So even in a universal
//! realization many short orbits are finite, and the property belongs to
//! the path, not only to the group. In the Ising realization every path
//! closes, because the group does.

use std::collections::{HashMap, HashSet};

use crate::backend::{Backend, DenseState};
use crate::error::{Error, Result};
use crate::math::{svd_thin, GateMatrix};
use crate::scalar::C64;

/// Largest strand count for which the Majorana realization is built:
/// the matrices are `2^{strands/2}` on a side.
pub const MAX_STRANDS: usize = 16;

/// How far [`run_path`] searches for a period element's projective
/// order before reporting that it did not close.
pub const ORDER_SEARCH_CAP: usize = 100_000;

// ── free group words ─────────────────────────────────────────────────

/// A freely reduced word in the free group on `n` generators: letters
/// `(index, ±1)` with no `x x^{-1}` adjacent pair left.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct FreeWord {
    letters: Vec<(usize, i8)>,
}

impl FreeWord {
    /// The empty word.
    pub fn identity() -> Self {
        FreeWord::default()
    }

    /// The generator `x_i`.
    pub fn generator(i: usize) -> Self {
        FreeWord {
            letters: vec![(i, 1)],
        }
    }

    /// Build from letters, reducing freely.
    pub fn from_letters(letters: &[(usize, i8)]) -> Self {
        let mut w = FreeWord::identity();
        for &(i, s) in letters {
            w.push(i, s);
        }
        w
    }

    fn push(&mut self, i: usize, s: i8) {
        match self.letters.last() {
            Some(&(j, t)) if j == i && t == -s => {
                self.letters.pop();
            }
            _ => self.letters.push((i, s.signum())),
        }
    }

    /// The reduced letters.
    pub fn letters(&self) -> &[(usize, i8)] {
        &self.letters
    }

    /// Reduced length.
    pub fn len(&self) -> usize {
        self.letters.len()
    }

    /// Whether the word reduces to the identity.
    pub fn is_empty(&self) -> bool {
        self.letters.is_empty()
    }

    /// The inverse word.
    pub fn inverse(&self) -> Self {
        FreeWord {
            letters: self
                .letters
                .iter()
                .rev()
                .map(|&(i, s)| (i, -s))
                .collect(),
        }
    }

    /// Concatenation, freely reduced.
    pub fn times(&self, other: &Self) -> Self {
        let mut w = self.clone();
        for &(i, s) in &other.letters {
            w.push(i, s);
        }
        w
    }

    /// Substitute `x_k ↦ images[k]` (and `x_k^{-1} ↦ images[k]^{-1}`).
    pub fn substitute(&self, images: &[FreeWord]) -> Self {
        let mut out = FreeWord::identity();
        for &(i, s) in &self.letters {
            let img = &images[i];
            if s > 0 {
                out = out.times(img);
            } else {
                out = out.times(&img.inverse());
            }
        }
        out
    }
}

// ── braid words and the Artin action ─────────────────────────────────

/// A word in the braid group `B_n`: letters are `±(i+1)` for the
/// generator `σ_i`, `i ∈ [0, n−1)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BraidWord {
    strands: usize,
    letters: Vec<i32>,
}

impl BraidWord {
    /// The identity braid on `strands` strands.
    pub fn identity(strands: usize) -> Result<Self> {
        if strands < 2 {
            return Err(Error::InvalidState(format!(
                "a braid needs at least 2 strands, got {strands}"
            )));
        }
        Ok(BraidWord {
            strands,
            letters: Vec::new(),
        })
    }

    /// The generator `σ_i` (or its inverse when `sign < 0`).
    pub fn generator(strands: usize, i: usize, sign: i8) -> Result<Self> {
        let mut w = BraidWord::identity(strands)?;
        w.push(i, sign)?;
        Ok(w)
    }

    /// Build from signed generator letters.
    pub fn from_letters(strands: usize, letters: &[i32]) -> Result<Self> {
        let mut w = BraidWord::identity(strands)?;
        for &l in letters {
            let i = (l.unsigned_abs() as usize).checked_sub(1).ok_or_else(|| {
                Error::InvalidState("braid letter 0 is not a generator".into())
            })?;
            w.push(i, if l < 0 { -1 } else { 1 })?;
        }
        Ok(w)
    }

    fn push(&mut self, i: usize, sign: i8) -> Result<()> {
        if i + 1 >= self.strands {
            return Err(Error::InvalidState(format!(
                "generator σ_{i} needs {} strands, have {}",
                i + 2,
                self.strands
            )));
        }
        let l = (i as i32 + 1) * if sign < 0 { -1 } else { 1 };
        // free reduction in the generators is always sound
        if self.letters.last() == Some(&-l) {
            self.letters.pop();
        } else {
            self.letters.push(l);
        }
        Ok(())
    }

    /// Strand count.
    pub fn strands(&self) -> usize {
        self.strands
    }

    /// The generator letters.
    pub fn letters(&self) -> &[i32] {
        &self.letters
    }

    /// Word length in generators.
    pub fn len(&self) -> usize {
        self.letters.len()
    }

    /// Whether the word is empty as written (not whether it is trivial in
    /// the group — that is [`BraidWord::is_trivial`]).
    pub fn is_empty(&self) -> bool {
        self.letters.is_empty()
    }

    /// Concatenation.
    pub fn times(&self, other: &Self) -> Result<Self> {
        if other.strands != self.strands {
            return Err(Error::InvalidState(
                "cannot compose braids on different strand counts".into(),
            ));
        }
        let mut w = self.clone();
        for &l in &other.letters {
            w.push(l.unsigned_abs() as usize - 1, if l < 0 { -1 } else { 1 })?;
        }
        Ok(w)
    }

    /// The inverse braid.
    pub fn inverse(&self) -> Self {
        BraidWord {
            strands: self.strands,
            letters: self.letters.iter().rev().map(|l| -l).collect(),
        }
    }

    /// The images of the free generators under the Artin action — the
    /// mutual boundary encoding as a substitution.
    ///
    /// `σ_i` sends `x_i ↦ x_i x_{i+1} x_i^{-1}` and `x_{i+1} ↦ x_i`,
    /// fixing the rest; `σ_i^{-1}` sends `x_i ↦ x_{i+1}` and
    /// `x_{i+1} ↦ x_{i+1}^{-1} x_i x_{i+1}`. The map `w ↦ images` is a
    /// homomorphism into `Aut(F_n)` and, by Artin's theorem, injective.
    pub fn artin_images(&self) -> Vec<FreeWord> {
        let n = self.strands;
        let mut images: Vec<FreeWord> = (0..n).map(FreeWord::generator).collect();
        for &l in &self.letters {
            let i = l.unsigned_abs() as usize - 1;
            let short = generator_images(n, i, l > 0);
            // compose: new[j] = φ_w(φ_σ(x_j))
            images = short.iter().map(|s| s.substitute(&images)).collect();
        }
        images
    }

    /// Exact equality in the braid group, via the faithful Artin action.
    pub fn equals(&self, other: &Self) -> bool {
        self.strands == other.strands && self.artin_images() == other.artin_images()
    }

    /// Whether the braid is the identity element of the group.
    pub fn is_trivial(&self) -> bool {
        self.artin_images()
            .iter()
            .enumerate()
            .all(|(j, w)| *w == FreeWord::generator(j))
    }

    /// The underlying permutation of strands (the braid group's map onto
    /// `S_n`): entry `j` is where strand `j` ends up.
    pub fn permutation(&self) -> Vec<usize> {
        let mut p: Vec<usize> = (0..self.strands).collect();
        for &l in &self.letters {
            let i = l.unsigned_abs() as usize - 1;
            p.swap(i, i + 1);
        }
        p
    }
}

/// The one-generator Artin substitution on `n` free generators.
fn generator_images(n: usize, i: usize, positive: bool) -> Vec<FreeWord> {
    let mut out: Vec<FreeWord> = (0..n).map(FreeWord::generator).collect();
    let (xi, xj) = (FreeWord::generator(i), FreeWord::generator(i + 1));
    if positive {
        out[i] = xi.times(&xj).times(&xi.inverse());
        out[i + 1] = xi;
    } else {
        out[i] = xj.clone();
        out[i + 1] = xj.inverse().times(&xi).times(&xj);
    }
    out
}

/// Sizes of the balls of radius `0..=radius` in the Cayley graph of
/// `B_strands` with the standard generators — the infinite navigation
/// graph, measured.
///
/// Elements are deduplicated by their Artin images, so the count is of
/// genuine group elements, not of words.
pub fn cayley_ball(strands: usize, radius: usize) -> Result<Vec<usize>> {
    let id = BraidWord::identity(strands)?;
    let mut seen: HashSet<Vec<FreeWord>> = HashSet::new();
    seen.insert(id.artin_images());
    let mut frontier = vec![id];
    let mut sizes = vec![1usize];
    for _ in 0..radius {
        let mut next = Vec::new();
        for w in &frontier {
            for i in 0..strands - 1 {
                for sign in [1i8, -1] {
                    let mut c = w.clone();
                    c.push(i, sign)?;
                    let key = c.artin_images();
                    if seen.insert(key) {
                        next.push(c);
                    }
                }
            }
        }
        frontier = next;
        sizes.push(seen.len());
        if frontier.is_empty() {
            break;
        }
    }
    Ok(sizes)
}

// ── periodic navigation words ────────────────────────────────────────

/// A finite period over an `alphabet`-letter alphabet, specifying an
/// infinite path: `step(k)` is the letter at any depth `k`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeriodicPath {
    alphabet: usize,
    period: Vec<usize>,
}

impl PeriodicPath {
    /// A path from one period; every letter must be below `alphabet`.
    pub fn new(alphabet: usize, period: Vec<usize>) -> Result<Self> {
        if alphabet < 2 {
            return Err(Error::InvalidState(
                "a navigation alphabet needs at least 2 letters".into(),
            ));
        }
        if period.is_empty() {
            return Err(Error::InvalidState("empty period".into()));
        }
        if let Some(&bad) = period.iter().find(|&&l| l >= alphabet) {
            return Err(Error::InvalidState(format!(
                "letter {bad} outside an alphabet of {alphabet}"
            )));
        }
        Ok(PeriodicPath { alphabet, period })
    }

    /// The alphabet size — the number of boundaries.
    pub fn alphabet(&self) -> usize {
        self.alphabet
    }

    /// The period.
    pub fn period(&self) -> &[usize] {
        &self.period
    }

    /// The letter at depth `k`, for any `k`: the infinite path, read from
    /// its finite representation.
    pub fn step(&self, k: usize) -> usize {
        self.period[k % self.period.len()]
    }

    /// The first `len` letters.
    pub fn prefix(&self, len: usize) -> Vec<usize> {
        (0..len).map(|k| self.step(k)).collect()
    }

    /// Whether the period is primitive (not itself a repetition).
    pub fn is_primitive(&self) -> bool {
        let n = self.period.len();
        (1..n).all(|d| n % d != 0 || (0..n).any(|k| self.period[k] != self.period[k % d]))
    }

    /// The lexicographically least rotation — the canonical name of the
    /// orbit this path traces.
    pub fn necklace(&self) -> Vec<usize> {
        let n = self.period.len();
        (0..n)
            .map(|r| {
                let mut v = self.period[r..].to_vec();
                v.extend_from_slice(&self.period[..r]);
                v
            })
            .min()
            .unwrap_or_default()
    }

    /// Two paths trace the same orbit iff their necklaces agree.
    pub fn same_orbit(&self, other: &Self) -> bool {
        self.alphabet == other.alphabet
            && self.period.len() == other.period.len()
            && self.necklace() == other.necklace()
    }
}

fn euler_phi(mut n: u64) -> u64 {
    let mut result = n;
    let mut p = 2;
    while p * p <= n {
        if n % p == 0 {
            while n % p == 0 {
                n /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if n > 1 {
        result -= result / n;
    }
    result
}

fn mobius(mut n: u64) -> i64 {
    let mut primes = 0;
    let mut p = 2;
    while p * p <= n {
        if n % p == 0 {
            n /= p;
            if n % p == 0 {
                return 0;
            }
            primes += 1;
        }
        p += 1;
    }
    if n > 1 {
        primes += 1;
    }
    if primes % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Distinct necklaces of length `len` over `alphabet` letters — the
/// number of orbit types a period of that length can trace. Burnside:
/// `(1/len) Σ_{d | len} φ(len/d) · alphabet^d`.
pub fn necklace_count(alphabet: u64, len: u64) -> u128 {
    if len == 0 {
        return 0;
    }
    let mut total: u128 = 0;
    for d in 1..=len {
        if len % d == 0 {
            total += euler_phi(len / d) as u128 * (alphabet as u128).pow(d as u32);
        }
    }
    total / len as u128
}

/// Lyndon words of length `len` over `alphabet` letters — the
/// **primitive** orbits, and the dimension of degree `len` of the free
/// Lie algebra on `alphabet` generators (Witt's formula). Möbius:
/// `(1/len) Σ_{d | len} μ(len/d) · alphabet^d`.
pub fn lyndon_count(alphabet: u64, len: u64) -> u128 {
    if len == 0 {
        return 0;
    }
    let mut total: i128 = 0;
    for d in 1..=len {
        if len % d == 0 {
            total += mobius(len / d) as i128 * (alphabet as i128).pow(d as u32);
        }
    }
    (total / len as i128) as u128
}

/// The dimension of the degree-`degree` component of the free Lie algebra
/// on `generators` generators. Identical to [`lyndon_count`] — the Lyndon
/// words of that length *are* a basis — which the tests measure rather
/// than take on faith.
pub fn free_lie_dim(generators: u64, degree: u64) -> u128 {
    lyndon_count(generators, degree)
}

/// Duval's algorithm: the unique factorization of a word into a
/// non-increasing sequence of Lyndon words.
pub fn lyndon_factorization(word: &[usize]) -> Vec<Vec<usize>> {
    let n = word.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let (mut j, mut k) = (i + 1, i);
        while j < n && word[k] <= word[j] {
            if word[k] < word[j] {
                k = i;
            } else {
                k += 1;
            }
            j += 1;
        }
        while i <= k {
            out.push(word[i..i + j - k].to_vec());
            i += j - k;
        }
    }
    out
}

/// What [`orbit_junction`] measured where two periodic orbits meet.
#[derive(Debug, Clone)]
pub struct OrbitJunction {
    /// The concatenated word.
    pub word: Vec<usize>,
    /// Its Lyndon factorization.
    pub factors: Vec<Vec<usize>>,
    /// Leading factors identical to the first path's own factorization.
    pub kept_from_first: usize,
    /// Trailing factors identical to the second path's own factorization.
    pub kept_from_second: usize,
    /// The factors belonging to neither — the **interface**. Empty when
    /// the two orbits abut with nothing new between them.
    pub interface: Vec<Vec<usize>>,
    /// Letters the interface spans.
    pub interface_len: usize,
}

impl OrbitJunction {
    /// Whether the two orbits abut with no interface at all — which
    /// happens exactly when their factorizations already concatenate
    /// into a non-increasing sequence.
    pub fn seamless(&self) -> bool {
        self.interface.is_empty()
    }
}

/// Concatenate a prefix of one periodic orbit with a prefix of another
/// and factor the result — the exact answer to what a boundary between
/// two orbits *is*.
///
/// The junction is not a new kind of object: Duval's factorization of the
/// joined word is unique and non-increasing, so whatever the seam does is
/// recorded in the same structure the orbits themselves are. What the
/// measurement then shows is that the seam is **sharply two-regime, and
/// decided by lexicographic order alone**:
///
/// * when the first orbit's last factor is `≥` the second's first factor,
///   the concatenated factorizations are already non-increasing, so by
///   uniqueness they *are* the joint factorization. The interface is
///   **empty** — the orbits abut with nothing between them at all.
/// * otherwise no such split exists, because a Lyndon word must be
///   strictly smaller than each of its proper suffixes. The smaller tail
///   absorbs the larger head, and the interface can be **arbitrarily
///   long** — in the measured `[1,1,0]` against `[0,1]` case it swallows
///   13 of the 24 letters and everything after the seam.
///
/// So the honest answer is not "the boundary is local". It is that a
/// junction is either free or total, the two orbits are not symmetric in
/// it, and which regime holds is settled by comparing words — nothing
/// about the geometry enters. [`OrbitJunction::seamless`] is that test.
pub fn orbit_junction(
    first: &PeriodicPath,
    len_a: usize,
    second: &PeriodicPath,
    len_b: usize,
) -> Result<OrbitJunction> {
    if first.alphabet() != second.alphabet() {
        return Err(Error::InvalidState(
            "orbits over different alphabets do not share a word".into(),
        ));
    }
    let a = first.prefix(len_a);
    let b = second.prefix(len_b);
    let mut word = a.clone();
    word.extend_from_slice(&b);

    let fa = lyndon_factorization(&a);
    let fb = lyndon_factorization(&b);
    let factors = lyndon_factorization(&word);

    let kept_from_first = factors
        .iter()
        .zip(fa.iter())
        .take_while(|(x, y)| x == y)
        .count();
    let kept_from_second = factors
        .iter()
        .rev()
        .zip(fb.iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let lo = kept_from_first;
    let hi = factors.len().saturating_sub(kept_from_second).max(lo);
    let interface: Vec<Vec<usize>> = factors[lo..hi].to_vec();
    let interface_len = interface.iter().map(|f| f.len()).sum();

    Ok(OrbitJunction {
        word,
        factors,
        kept_from_first,
        kept_from_second,
        interface,
        interface_len,
    })
}

// ── the boundary encoding ────────────────────────────────────────────

/// What [`mutual_encoding`] measured across one cut.
#[derive(Debug, Clone)]
pub struct MutualEncoding {
    /// The region taken as `A` (sorted qubit indices).
    pub region: Vec<usize>,
    /// Its complement `B`.
    pub complement: Vec<usize>,
    /// Schmidt rank across the cut.
    pub rank: usize,
    /// Schmidt coefficients, descending.
    pub spectrum: Vec<f64>,
    /// Worst difference between the eigenvalues of `ρ_A` and those of
    /// `ρ_B`, each computed independently from its own side. The two
    /// boundaries carry the same spectrum — measured, not assumed.
    pub spectrum_deviation: f64,
    /// Worst deviation of `ρ_A` rebuilt from its own purification: `ρ_A`
    /// alone is purified into a state on `A ⊗ ℂ^rank`, and that state's
    /// `A`-marginal is compared with the original. `B` is never
    /// consulted.
    pub purification_deviation: f64,
    /// Worst difference between the purification's `B`-side spectrum and
    /// the true `ρ_B` spectrum — how much of `B` the boundary of `A`
    /// determines.
    pub purified_spectrum_deviation: f64,
    /// Worst deviation of the joint state rebuilt from the full Schmidt
    /// data.
    pub reconstruction_deviation: f64,
    /// Numbers the boundary encoding stores: `rank·(2^{|A|} + 2^{|B|}) + rank`.
    pub stored_entries: usize,
    /// Numbers the dense state stores: `2^{|A|+|B|}`.
    pub dense_entries: usize,
}

impl MutualEncoding {
    /// Storage ratio `dense / stored`; `≤ 1` means the cut bought
    /// nothing, which is the generic case and is reported, not hidden.
    pub fn compression(&self) -> f64 {
        self.dense_entries as f64 / self.stored_entries.max(1) as f64
    }

    /// The largest rank this cut could have had.
    pub fn max_rank(&self) -> usize {
        (1usize << self.region.len()).min(1usize << self.complement.len())
    }
}

/// Read a cut's boundary data off a loaded backend and measure the
/// mutual encoding in both directions.
pub fn mutual_encoding(state: &dyn Backend<C64>, region: &[usize]) -> Result<MutualEncoding> {
    let n = state.num_qubits();
    let mut a: Vec<usize> = region.to_vec();
    a.sort_unstable();
    a.dedup();
    if a.len() != region.len() {
        return Err(Error::InvalidState("repeated qubit in the region".into()));
    }
    if a.iter().any(|&q| q >= n) {
        return Err(Error::QubitOutOfRange {
            qubit: *a.iter().max().unwrap(),
            num_qubits: n,
        });
    }
    if a.is_empty() || a.len() == n {
        return Err(Error::InvalidState(
            "a boundary needs a region and a nonempty complement".into(),
        ));
    }
    let b: Vec<usize> = (0..n).filter(|q| !a.contains(q)).collect();
    let (da, db) = (1usize << a.len(), 1usize << b.len());

    // ψ as a da × db matrix
    let mut psi = vec![C64::new(0.0, 0.0); da * db];
    for ia in 0..da {
        for ib in 0..db {
            let mut idx = 0u64;
            for (k, &q) in a.iter().enumerate() {
                if ia >> k & 1 == 1 {
                    idx |= 1 << q;
                }
            }
            for (k, &q) in b.iter().enumerate() {
                if ib >> k & 1 == 1 {
                    idx |= 1 << q;
                }
            }
            psi[ia * db + ib] = state.amplitude(idx);
        }
    }

    let svd = svd_thin(da, db, &psi, 1e-12, 1e-14)?;
    let rank = svd.rank;
    let spectrum: Vec<f64> = svd.sigma[..rank].to_vec();

    // ρ_A and ρ_B, each built from its own side, spectra compared
    let rho_a = gram(da, db, &psi, true);
    let rho_b = gram(da, db, &psi, false);
    let eig_a = psd_eigenvalues(da, &rho_a)?;
    let eig_b = psd_eigenvalues(db, &rho_b)?;
    let spectrum_deviation = spectrum_gap(&eig_a, &eig_b);

    // purify ρ_A alone: |χ⟩ = Σ_k √λ_k |u_k⟩|k⟩ on A ⊗ ℂ^rank
    let svd_a = svd_thin(da, da, &rho_a, 1e-12, 1e-14)?;
    let prank = svd_a.rank;
    let mut chi = vec![C64::new(0.0, 0.0); da * prank];
    for ia in 0..da {
        for k in 0..prank {
            chi[ia * prank + k] = svd_a.u[ia * prank + k] * C64::new(svd_a.sigma[k].sqrt(), 0.0);
        }
    }
    let rho_a2 = gram(da, prank, &chi, true);
    let purification_deviation = max_entry_gap(&rho_a, &rho_a2);
    let eig_chi_b = psd_eigenvalues(prank, &gram(da, prank, &chi, false))?;
    let purified_spectrum_deviation = spectrum_gap(&eig_chi_b, &eig_b);

    // the joint state rebuilt from the full Schmidt data
    let mut rebuilt = vec![C64::new(0.0, 0.0); da * db];
    for ia in 0..da {
        for ib in 0..db {
            let mut acc = C64::new(0.0, 0.0);
            for k in 0..rank {
                acc += svd.u[ia * rank + k] * C64::new(svd.sigma[k], 0.0) * svd.vt[k * db + ib];
            }
            rebuilt[ia * db + ib] = acc;
        }
    }
    let reconstruction_deviation = max_entry_gap(&psi, &rebuilt);

    Ok(MutualEncoding {
        region: a.clone(),
        complement: b.clone(),
        rank,
        spectrum,
        spectrum_deviation,
        purification_deviation,
        purified_spectrum_deviation,
        reconstruction_deviation,
        stored_entries: rank * (da + db) + rank,
        dense_entries: da * db,
    })
}

/// `M M†` (when `left`) or `M† M`, as a row-major square matrix.
fn gram(rows: usize, cols: usize, m: &[C64], left: bool) -> Vec<C64> {
    let d = if left { rows } else { cols };
    let mut out = vec![C64::new(0.0, 0.0); d * d];
    for i in 0..d {
        for j in 0..d {
            let mut acc = C64::new(0.0, 0.0);
            if left {
                for k in 0..cols {
                    acc += m[i * cols + k] * m[j * cols + k].conj();
                }
            } else {
                for k in 0..rows {
                    acc += m[k * cols + i].conj() * m[k * cols + j];
                }
            }
            out[i * d + j] = acc;
        }
    }
    out
}

/// Eigenvalues of a positive semidefinite matrix, descending — its
/// singular values.
fn psd_eigenvalues(d: usize, m: &[C64]) -> Result<Vec<f64>> {
    let svd = svd_thin(d, d, m, 0.0, 0.0)?;
    Ok(svd.sigma)
}

/// Worst difference between two spectra, padded with zeros.
fn spectrum_gap(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().max(b.len());
    (0..n)
        .map(|k| {
            let x = a.get(k).copied().unwrap_or(0.0);
            let y = b.get(k).copied().unwrap_or(0.0);
            (x - y).abs()
        })
        .fold(0.0, f64::max)
}

fn max_entry_gap(a: &[C64], b: &[C64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0, f64::max)
}

// ── unitary realizations of the braiding ─────────────────────────────

/// Jordan–Wigner Majorana operators `γ_0 … γ_{2m−1}` on `m` qubits, as
/// `2^m × 2^m` matrices: `γ_{2k} = Z_0…Z_{k−1} X_k`,
/// `γ_{2k+1} = Z_0…Z_{k−1} Y_k`.
pub fn majorana_operators(qubits: usize) -> Result<Vec<GateMatrix<C64>>> {
    if qubits == 0 || qubits > MAX_STRANDS / 2 {
        return Err(Error::InvalidState(format!(
            "majorana_operators supports 1..={} qubits, got {qubits}",
            MAX_STRANDS / 2
        )));
    }
    let d = 1usize << qubits;
    let mut out = Vec::with_capacity(2 * qubits);
    for k in 0..qubits {
        for which in 0..2 {
            let mut m = GateMatrix::<C64>::zeros(d)?;
            for col in 0..d {
                // Z string on qubits < k
                let mut sign = 1.0;
                for q in 0..k {
                    if col >> q & 1 == 1 {
                        sign = -sign;
                    }
                }
                let bit = col >> k & 1;
                let row = col ^ (1 << k);
                let v = if which == 0 {
                    // X_k
                    C64::new(sign, 0.0)
                } else {
                    // Y_k: |0⟩⟨1| ↦ -i, |1⟩⟨0| ↦ +i
                    if bit == 0 {
                        C64::new(0.0, sign)
                    } else {
                        C64::new(0.0, -sign)
                    }
                };
                m.set(row, col, v);
            }
            out.push(m);
        }
    }
    Ok(out)
}

/// The Ising (Majorana) braid generators on `strands` strands:
/// `σ_i = (1 + γ_i γ_{i+1})/√2`, acting on `strands/2` qubits.
///
/// These satisfy the braid relations exactly, for every even strand
/// count, because the Majoranas satisfy the Clifford algebra —
/// [`verify_relations`] measures both.
pub fn majorana_generators(strands: usize) -> Result<Vec<GateMatrix<C64>>> {
    if strands < 2 || strands % 2 != 0 || strands > MAX_STRANDS {
        return Err(Error::InvalidState(format!(
            "majorana_generators needs an even strand count in 2..={MAX_STRANDS}, got {strands}"
        )));
    }
    let g = majorana_operators(strands / 2)?;
    let d = 1usize << (strands / 2);
    let inv = 1.0 / std::f64::consts::SQRT_2;
    let mut out = Vec::with_capacity(strands - 1);
    for i in 0..strands - 1 {
        let prod = g[i].matmul(&g[i + 1]);
        let mut m = GateMatrix::<C64>::zeros(d)?;
        for r in 0..d {
            for c in 0..d {
                let extra = if r == c { C64::new(1.0, 0.0) } else { C64::new(0.0, 0.0) };
                m.set(r, c, (extra + prod.get(r, c)) * C64::new(inv, 0.0));
            }
        }
        out.push(m);
    }
    Ok(out)
}

/// The **infinitesimal** braid generators of the Majorana realization:
/// the bilinears `γ_i γ_{i+1}`, which are anti-Hermitian and satisfy
/// `σ_i = exp((π/4)·γ_i γ_{i+1})`.
///
/// This is the exponential map the path algebra needs: one bilinear is a
/// single infinitesimal fold, and exponentiating it produces the finite
/// braid generator. Their brackets are what [`realized_rank`] measures
/// against the free Lie algebra.
pub fn majorana_bilinears(strands: usize) -> Result<Vec<GateMatrix<C64>>> {
    if strands < 2 || strands % 2 != 0 || strands > MAX_STRANDS {
        return Err(Error::InvalidState(format!(
            "majorana_bilinears needs an even strand count in 2..={MAX_STRANDS}, got {strands}"
        )));
    }
    let g = majorana_operators(strands / 2)?;
    Ok((0..strands - 1).map(|i| g[i].matmul(&g[i + 1])).collect())
}

/// The 2-dimensional Fibonacci-anyon representation of `B_3`:
/// `σ_1 = diag(e^{−4πi/5}, e^{3πi/5})` and `σ_2 = F σ_1 F†` with the
/// golden-ratio `F`-matrix. Its image is dense in `PSU(2)` — universal —
/// which [`orbit_closure`] measures as a ball that never closes.
pub fn fibonacci_generators() -> Result<Vec<GateMatrix<C64>>> {
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    let (a, b) = (1.0 / phi, 1.0 / phi.sqrt());
    let f = GateMatrix::<C64>::from_vec(
        2,
        vec![
            C64::new(a, 0.0),
            C64::new(b, 0.0),
            C64::new(b, 0.0),
            C64::new(-a, 0.0),
        ],
    )?;
    let tau = std::f64::consts::TAU;
    let s1 = GateMatrix::<C64>::from_vec(
        2,
        vec![
            C64::from_polar(1.0, -2.0 * tau / 5.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::from_polar(1.0, 3.0 * tau / 10.0),
        ],
    )?;
    let s2 = f.matmul(&s1).matmul(&f.dagger());
    Ok(vec![s1, s2])
}

/// Measured residuals of the braid group's defining relations on a set
/// of generator matrices.
#[derive(Debug, Clone)]
pub struct BraidRelations {
    /// Worst `‖U†U − I‖_max` over the generators.
    pub unitarity: f64,
    /// Worst `‖σ_i σ_{i+1} σ_i − σ_{i+1} σ_i σ_{i+1}‖_max`.
    pub braid: f64,
    /// Worst `‖σ_i σ_j − σ_j σ_i‖_max` over `|i − j| ≥ 2`; `0.0` when
    /// there is no such pair.
    pub far_commutation: f64,
    /// Generators checked.
    pub generators: usize,
}

impl BraidRelations {
    /// Whether every relation holds to `tol`.
    pub fn hold(&self, tol: f64) -> bool {
        self.unitarity <= tol && self.braid <= tol && self.far_commutation <= tol
    }
}

/// Measure the braid relations on a candidate realization.
pub fn verify_relations(gens: &[GateMatrix<C64>]) -> Result<BraidRelations> {
    if gens.is_empty() {
        return Err(Error::InvalidState("no generators to verify".into()));
    }
    let d = gens[0].dim();
    if gens.iter().any(|g| g.dim() != d) {
        return Err(Error::InvalidState(
            "generators of different dimensions".into(),
        ));
    }
    let unitarity = gens
        .iter()
        .map(|g| g.unitarity_deviation())
        .fold(0.0, f64::max);
    let mut braid: f64 = 0.0;
    for i in 0..gens.len().saturating_sub(1) {
        let l = gens[i].matmul(&gens[i + 1]).matmul(&gens[i]);
        let r = gens[i + 1].matmul(&gens[i]).matmul(&gens[i + 1]);
        braid = braid.max(max_entry_gap(l.data(), r.data()));
    }
    let mut far: f64 = 0.0;
    for i in 0..gens.len() {
        for j in 0..gens.len() {
            if j >= i + 2 {
                let l = gens[i].matmul(&gens[j]);
                let r = gens[j].matmul(&gens[i]);
                far = far.max(max_entry_gap(l.data(), r.data()));
            }
        }
    }
    Ok(BraidRelations {
        unitarity,
        braid,
        far_commutation: far,
        generators: gens.len(),
    })
}

// ── does the path close? ─────────────────────────────────────────────

/// What [`orbit_closure`] measured about the group a realization
/// generates.
#[derive(Debug, Clone)]
pub struct OrbitClosure {
    /// Ball sizes at radius `0, 1, 2, …` — distinct elements up to a
    /// global phase.
    pub ball_sizes: Vec<usize>,
    /// The order of the projective image, when the ball stopped growing
    /// inside the budget. `None` means it did not close — measured, not
    /// concluded.
    pub closed_at: Option<usize>,
    /// Radius at which the ball stopped growing, if it did.
    pub closure_radius: Option<usize>,
    /// Geometric growth ratio of the last three balls; near 1 when the
    /// group is closing, well above 1 when it is not.
    pub growth: f64,
    /// Whether the search stopped because it hit the element cap rather
    /// than because the group closed.
    pub hit_cap: bool,
}

impl OrbitClosure {
    /// A finite image means an arbitrarily long path is stored in `O(1)`:
    /// the path's *address* is a group element, and there are only so
    /// many.
    pub fn finite(&self) -> bool {
        self.closed_at.is_some()
    }
}

/// Canonical key of a unitary up to global phase, rounded so that
/// numerically identical elements collide.
fn phase_free_key(m: &GateMatrix<C64>) -> Vec<i64> {
    let data = m.data();
    let (mut best, mut bi) = (0.0f64, 0usize);
    for (i, z) in data.iter().enumerate() {
        if z.norm() > best + 1e-12 {
            best = z.norm();
            bi = i;
        }
    }
    let pivot = data[bi] / C64::new(data[bi].norm(), 0.0);
    let scale = 1e6;
    data.iter()
        .flat_map(|z| {
            let w = *z / pivot;
            [
                (w.re * scale).round() as i64,
                (w.im * scale).round() as i64,
            ]
        })
        .collect()
}

/// The **projective order** of a unitary: the least `k ≤ cap` with
/// `U^k` a multiple of the identity to `tol`, or `None` when there is
/// none in range.
///
/// This is the decisive test a tolerance-rounded state count cannot
/// give. A periodic path applies one fixed group element per period, so
/// the path's orbit closes exactly when that element has finite
/// projective order — and when it does not, any finite state count is a
/// statement about the counting resolution, not about the orbit.
pub fn projective_order(u: &GateMatrix<C64>, cap: usize, tol: f64) -> Option<usize> {
    let d = u.dim();
    let mut p = u.clone();
    for k in 1..=cap {
        // is p a multiple of the identity?
        let mut pivot = C64::new(0.0, 0.0);
        let mut best = 0.0f64;
        for i in 0..d {
            let z = p.get(i, i);
            if z.norm() > best {
                best = z.norm();
                pivot = z;
            }
        }
        if best > tol {
            let mut ok = true;
            for r in 0..d {
                for c in 0..d {
                    let want = if r == c { pivot } else { C64::new(0.0, 0.0) };
                    if (p.get(r, c) - want).norm() > tol {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    break;
                }
            }
            if ok {
                return Some(k);
            }
        }
        p = p.matmul(u);
    }
    None
}

/// Walk the group the generators produce, breadth first, and measure
/// whether it closes.
///
/// Elements are identified up to a global phase, which is the physically
/// meaningful identification: a global phase is not observable, so two
/// paths differing by one have reached the same state of the register.
pub fn orbit_closure(
    gens: &[GateMatrix<C64>],
    max_radius: usize,
    cap: usize,
) -> Result<OrbitClosure> {
    if gens.is_empty() {
        return Err(Error::InvalidState("no generators".into()));
    }
    let d = gens[0].dim();
    let mut all: Vec<GateMatrix<C64>> = Vec::with_capacity(2 * gens.len());
    for g in gens {
        all.push(g.clone());
        all.push(g.dagger());
    }
    let id = GateMatrix::<C64>::identity(d)?;
    let mut seen: HashSet<Vec<i64>> = HashSet::new();
    seen.insert(phase_free_key(&id));
    let mut frontier = vec![id];
    let mut sizes = vec![1usize];
    let mut hit_cap = false;
    let mut closure_radius = None;
    for r in 1..=max_radius {
        let mut next = Vec::new();
        for m in &frontier {
            for g in &all {
                let p = g.matmul(m);
                if seen.insert(phase_free_key(&p)) {
                    next.push(p);
                }
            }
        }
        sizes.push(seen.len());
        if next.is_empty() {
            closure_radius = Some(r);
            break;
        }
        if seen.len() > cap {
            hit_cap = true;
            break;
        }
        frontier = next;
    }
    let growth = if sizes.len() >= 3 {
        let n = sizes.len();
        let (a, b) = (sizes[n - 2] as f64, sizes[n - 1] as f64);
        if a > 0.0 {
            b / a
        } else {
            1.0
        }
    } else {
        1.0
    };
    let closed_at = closure_radius.map(|_| *sizes.last().unwrap());
    Ok(OrbitClosure {
        ball_sizes: sizes,
        closed_at,
        closure_radius,
        growth,
        hit_cap,
    })
}

// ── the bracket as the geometric residue ─────────────────────────────

/// `exp(M)` by scaling and squaring with a Taylor series — enough for
/// the small matrices the bracket measurements use.
pub fn mat_exp(m: &GateMatrix<C64>) -> Result<GateMatrix<C64>> {
    let d = m.dim();
    let norm = m
        .data()
        .iter()
        .map(|z| z.norm())
        .fold(0.0f64, f64::max)
        * d as f64;
    let squarings = ((norm.max(1e-300)).log2().ceil().max(0.0) as u32 + 1).min(60);
    let scale = C64::new(1.0 / (1u64 << squarings) as f64, 0.0);
    let mut a = GateMatrix::<C64>::zeros(d)?;
    for r in 0..d {
        for c in 0..d {
            a.set(r, c, m.get(r, c) * scale);
        }
    }
    let mut result = GateMatrix::<C64>::identity(d)?;
    let mut term = GateMatrix::<C64>::identity(d)?;
    for k in 1..=24u32 {
        term = term.matmul(&a);
        let f = C64::new(1.0 / (1..=k).map(|i| i as f64).product::<f64>(), 0.0);
        for r in 0..d {
            for c in 0..d {
                let v = result.get(r, c) + term.get(r, c) * f;
                result.set(r, c, v);
            }
        }
    }
    for _ in 0..squarings {
        result = result.matmul(&result);
    }
    Ok(result)
}

/// The Lie bracket `[A, B] = AB − BA`.
pub fn bracket(a: &GateMatrix<C64>, b: &GateMatrix<C64>) -> Result<GateMatrix<C64>> {
    let d = a.dim();
    let ab = a.matmul(b);
    let ba = b.matmul(a);
    let mut out = GateMatrix::<C64>::zeros(d)?;
    for r in 0..d {
        for c in 0..d {
            out.set(r, c, ab.get(r, c) - ba.get(r, c));
        }
    }
    Ok(out)
}

fn scaled(m: &GateMatrix<C64>, s: f64) -> Result<GateMatrix<C64>> {
    let d = m.dim();
    let mut out = GateMatrix::<C64>::zeros(d)?;
    for r in 0..d {
        for c in 0..d {
            out.set(r, c, m.get(r, c) * C64::new(s, 0.0));
        }
    }
    Ok(out)
}

fn sum(terms: &[(f64, &GateMatrix<C64>)]) -> Result<GateMatrix<C64>> {
    let d = terms[0].1.dim();
    let mut out = GateMatrix::<C64>::zeros(d)?;
    for &(w, m) in terms {
        for r in 0..d {
            for c in 0..d {
                let v = out.get(r, c) + m.get(r, c) * C64::new(w, 0.0);
                out.set(r, c, v);
            }
        }
    }
    Ok(out)
}

/// A measured power law: residual `~ ε^order`.
#[derive(Debug, Clone)]
pub struct ResidualLaw {
    /// The `ε` values probed.
    pub epsilons: Vec<f64>,
    /// The residual at each.
    pub residuals: Vec<f64>,
    /// Fitted exponent from a log–log line through the probes.
    pub order: f64,
}

fn fit_order(eps: &[f64], res: &[f64]) -> f64 {
    let n = eps.len() as f64;
    let xs: Vec<f64> = eps.iter().map(|e| e.ln()).collect();
    let ys: Vec<f64> = res.iter().map(|r| r.max(1e-300).ln()).collect();
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let num: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

/// The geometric residue of folding `A` through `B` and back, measured
/// against the bracket.
///
/// The group commutator `exp(εA)exp(εB)exp(−εA)exp(−εB)` equals
/// `exp(ε²[A,B])` up to order `ε³`, so the residual between them must
/// scale as `ε³` — and the fitted exponent is the measurement that says
/// the bracket really is the leading residue rather than a formal
/// stand-in for it.
pub fn commutator_residual(
    a: &GateMatrix<C64>,
    b: &GateMatrix<C64>,
    epsilons: &[f64],
) -> Result<ResidualLaw> {
    let c = bracket(a, b)?;
    let mut residuals = Vec::with_capacity(epsilons.len());
    for &e in epsilons {
        let ea = mat_exp(&scaled(a, e)?)?;
        let eb = mat_exp(&scaled(b, e)?)?;
        let eai = mat_exp(&scaled(a, -e)?)?;
        let ebi = mat_exp(&scaled(b, -e)?)?;
        let lhs = ea.matmul(&eb).matmul(&eai).matmul(&ebi);
        let rhs = mat_exp(&scaled(&c, e * e)?)?;
        residuals.push(max_entry_gap(lhs.data(), rhs.data()));
    }
    let order = fit_order(epsilons, &residuals);
    Ok(ResidualLaw {
        epsilons: epsilons.to_vec(),
        residuals,
        order,
    })
}

/// The Baker–Campbell–Hausdorff series to third order, measured the same
/// way: `exp(εA)exp(εB)` against
/// `exp(ε(A+B) + ε²[A,B]/2 + ε³([A,[A,B]] + [B,[B,A]])/12)`, whose
/// residual must scale as `ε⁴`.
pub fn bch_residual(
    a: &GateMatrix<C64>,
    b: &GateMatrix<C64>,
    epsilons: &[f64],
) -> Result<ResidualLaw> {
    let c = bracket(a, b)?;
    let aac = bracket(a, &c)?;
    let bba = bracket(b, &bracket(b, a)?)?;
    let mut residuals = Vec::with_capacity(epsilons.len());
    for &e in epsilons {
        let lhs = mat_exp(&scaled(a, e)?)?.matmul(&mat_exp(&scaled(b, e)?)?);
        let series = sum(&[
            (e, a),
            (e, b),
            (e * e / 2.0, &c),
            (e * e * e / 12.0, &aac),
            (e * e * e / 12.0, &bba),
        ])?;
        let rhs = mat_exp(&series)?;
        residuals.push(max_entry_gap(lhs.data(), rhs.data()));
    }
    let order = fit_order(epsilons, &residuals);
    Ok(ResidualLaw {
        epsilons: epsilons.to_vec(),
        residuals,
        order,
    })
}

/// What [`realized_rank`] measured: how much of the free Lie algebra a
/// concrete realization actually carries.
#[derive(Debug, Clone)]
pub struct RealizedAlgebra {
    /// Free-algebra dimension newly available at each degree `1..=degree`
    /// (Witt / Lyndon).
    pub free_dims: Vec<u128>,
    /// Cumulative free dimension through each degree.
    pub free_cumulative: Vec<u128>,
    /// Measured rank of the span of all bracket monomials through each
    /// degree, in the realization.
    pub realized: Vec<usize>,
    /// The degree at which the realized rank stopped growing, if it did.
    pub saturated_at: Option<usize>,
    /// The realization's ambient real dimension `2·d²`.
    pub ambient: usize,
}

/// Real rank of a set of complex matrices, by modified Gram–Schmidt over
/// the `2d²` real coordinates.
fn real_rank(mats: &[GateMatrix<C64>], tol: f64) -> usize {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for m in mats {
        let mut v: Vec<f64> = m
            .data()
            .iter()
            .flat_map(|z| [z.re, z.im])
            .collect();
        for b in &basis {
            let dot: f64 = v.iter().zip(b).map(|(x, y)| x * y).sum();
            for (x, y) in v.iter_mut().zip(b) {
                *x -= dot * y;
            }
        }
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > tol {
            for x in v.iter_mut() {
                *x /= n;
            }
            basis.push(v);
        }
    }
    basis.len()
}

/// Measure the collapse: the free Lie algebra on `gens.len()` generators
/// against the algebra those generators actually span in the
/// realization, degree by degree.
///
/// The generators should be the Lie-algebra elements (anti-Hermitian
/// logs, or any matrices whose brackets are of interest), not the group
/// elements.
pub fn realized_rank(gens: &[GateMatrix<C64>], degree: usize) -> Result<RealizedAlgebra> {
    if gens.is_empty() {
        return Err(Error::InvalidState("no generators".into()));
    }
    let d = gens[0].dim();
    let mut layers: Vec<Vec<GateMatrix<C64>>> = vec![gens.to_vec()];
    let mut all: Vec<GateMatrix<C64>> = gens.to_vec();
    let mut realized = vec![real_rank(&all, 1e-9)];
    let mut free_dims = vec![lyndon_count(gens.len() as u64, 1)];
    let mut saturated_at = None;
    for k in 2..=degree {
        let mut layer = Vec::new();
        for lower in &layers[k - 2] {
            for g in gens {
                layer.push(bracket(g, lower)?);
            }
        }
        all.extend(layer.iter().cloned());
        layers.push(layer);
        let r = real_rank(&all, 1e-9);
        if saturated_at.is_none() && r == *realized.last().unwrap() {
            saturated_at = Some(k - 1);
        }
        realized.push(r);
        free_dims.push(lyndon_count(gens.len() as u64, k as u64));
    }
    let mut free_cumulative = Vec::with_capacity(free_dims.len());
    let mut acc = 0u128;
    for &f in &free_dims {
        acc += f;
        free_cumulative.push(acc);
    }
    Ok(RealizedAlgebra {
        free_dims,
        free_cumulative,
        realized,
        saturated_at,
        ambient: 2 * d * d,
    })
}

// ── the recursion ledger ─────────────────────────────────────────────

/// One journalled step of a path.
#[derive(Debug, Clone)]
pub struct LedgerStep {
    /// Depth along the path.
    pub depth: usize,
    /// The generator applied — the path letter.
    pub letter: usize,
    /// Schmidt rank across the reference cut after this step.
    pub rank: usize,
    /// Largest Schmidt coefficient after this step.
    pub leading: f64,
    /// Whether the register's state had been reached before, up to a
    /// global phase.
    pub revisited: bool,
}

/// What [`run_path`] measured.
#[derive(Debug, Clone)]
pub struct RecursionLedger {
    /// One entry per step, in order.
    pub steps: Vec<LedgerStep>,
    /// Distinct register states the path reached, up to a global phase
    /// **and up to the counting tolerance**. A dense orbit saturates
    /// this count too, at whatever resolution the key rounds to, so it
    /// is a description of the run and not a proof of closure — that is
    /// [`RecursionLedger::cycle_order`]'s job.
    pub distinct_states: usize,
    /// The projective order of the group element one full period
    /// applies — the decisive closure test. `Some(k)` means the path
    /// really is a `k`-cycle and the whole infinite path is `k` states;
    /// `None` means it did not close within the search cap, so no finite
    /// state count describes it.
    pub cycle_order: Option<usize>,
    /// Depth at which the state set stopped growing, if it did — the
    /// point past which the path is pure re-addressing and the storage
    /// stops growing.
    pub saturated_at: Option<usize>,
    /// Deviation after ascending the whole path (applying its inverse):
    /// the recursion is exactly reversible, so this is a measurement of
    /// how much information the descent erased.
    pub ascent_deviation: f64,
    /// Bytes to hold the path itself (one letter per step).
    pub path_bytes: usize,
    /// Bytes the dense register holds.
    pub state_bytes: usize,
}

/// Drive a register along a periodic path of braid generators,
/// journalling the boundary at every step, then ascend the whole path
/// and measure what the descent cost.
///
/// `gens` are the realization's generators (from
/// [`majorana_generators`] or [`fibonacci_generators`]); the path's
/// alphabet must not exceed their count. The register is the
/// realization's own space, initialized to `|0…0⟩`.
pub fn run_path(
    gens: &[GateMatrix<C64>],
    path: &PeriodicPath,
    depth: usize,
    cut: &[usize],
) -> Result<RecursionLedger> {
    if gens.is_empty() {
        return Err(Error::InvalidState("no generators".into()));
    }
    if path.alphabet() > gens.len() {
        return Err(Error::InvalidState(format!(
            "path alphabet {} exceeds {} generators",
            path.alphabet(),
            gens.len()
        )));
    }
    let d = gens[0].dim();
    let qubits = d.trailing_zeros() as usize;
    if 1usize << qubits != d {
        return Err(Error::InvalidState(
            "the realization's dimension is not a power of two".into(),
        ));
    }
    let all: Vec<usize> = (0..qubits).collect();

    // the element one full period applies, and whether it closes
    let mut period_element = GateMatrix::<C64>::identity(d)?;
    for k in 0..path.period().len() {
        period_element = gens[path.step(k)].matmul(&period_element);
    }
    let cycle_order = projective_order(&period_element, ORDER_SEARCH_CAP, 1e-9);

    let mut state = DenseState::<C64>::new(qubits)?;
    let mut seen: HashSet<Vec<i64>> = HashSet::new();
    seen.insert(state_key(&state));
    let mut steps = Vec::with_capacity(depth);
    let mut saturated_at = None;
    let mut last_count = seen.len();
    let mut stagnant = 0usize;

    for k in 0..depth {
        let letter = path.step(k);
        state.apply(&gens[letter], &all)?;
        let revisited = !seen.insert(state_key(&state));
        let (rank, leading) = if cut.is_empty() || qubits < 2 {
            (1, 1.0)
        } else {
            let enc = mutual_encoding(&state, cut)?;
            (enc.rank, enc.spectrum.first().copied().unwrap_or(0.0))
        };
        steps.push(LedgerStep {
            depth: k + 1,
            letter,
            rank,
            leading,
            revisited,
        });
        if seen.len() == last_count {
            stagnant += 1;
            if stagnant >= 2 * path.period().len() && saturated_at.is_none() {
                saturated_at = Some(k + 1);
            }
        } else {
            stagnant = 0;
            last_count = seen.len();
        }
    }

    // ascend: the path run backwards, exactly
    for k in (0..depth).rev() {
        let letter = path.step(k);
        state.apply(&gens[letter].dagger(), &all)?;
    }
    let mut ascent_deviation: f64 = 0.0;
    for i in 0..d {
        let want = if i == 0 {
            C64::new(1.0, 0.0)
        } else {
            C64::new(0.0, 0.0)
        };
        ascent_deviation = ascent_deviation.max((state.amplitude(i as u64) - want).norm());
    }

    Ok(RecursionLedger {
        steps,
        distinct_states: seen.len(),
        cycle_order,
        saturated_at,
        ascent_deviation,
        path_bytes: depth,
        state_bytes: d * std::mem::size_of::<C64>(),
    })
}

/// Phase-free key of a register state.
fn state_key(state: &DenseState<C64>) -> Vec<i64> {
    let d = 1usize << state.num_qubits();
    let amps: Vec<C64> = (0..d).map(|i| state.amplitude(i as u64)).collect();
    let (mut best, mut bi) = (0.0f64, 0usize);
    for (i, z) in amps.iter().enumerate() {
        if z.norm() > best + 1e-12 {
            best = z.norm();
            bi = i;
        }
    }
    let pivot = amps[bi] / C64::new(amps[bi].norm().max(1e-300), 0.0);
    let scale = 1e6;
    amps.iter()
        .flat_map(|z| {
            let w = *z / pivot;
            [
                (w.re * scale).round() as i64,
                (w.im * scale).round() as i64,
            ]
        })
        .collect()
}

/// Count the distinct orbits (necklaces) a set of periods traces, by
/// canonical rotation — the measured counterpart of [`necklace_count`].
pub fn distinct_orbits(paths: &[PeriodicPath]) -> usize {
    let mut set: HashMap<(usize, Vec<usize>), usize> = HashMap::new();
    for p in paths {
        *set.entry((p.alphabet(), p.necklace())).or_insert(0) += 1;
    }
    set.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artin_action_satisfies_the_braid_relations() {
        for n in 3..=5 {
            for i in 0..n - 2 {
                let a = BraidWord::from_letters(
                    n,
                    &[i as i32 + 1, i as i32 + 2, i as i32 + 1],
                )
                .unwrap();
                let b = BraidWord::from_letters(
                    n,
                    &[i as i32 + 2, i as i32 + 1, i as i32 + 2],
                )
                .unwrap();
                assert!(a.equals(&b), "braid relation failed at {i} on {n} strands");
            }
        }
    }

    #[test]
    fn free_words_reduce() {
        let w = FreeWord::from_letters(&[(0, 1), (1, 1), (1, -1), (0, -1), (2, 1)]);
        assert_eq!(w.letters(), &[(2, 1)]);
        assert!(w.times(&w.inverse()).is_empty());
    }

    #[test]
    fn lyndon_counts_match_enumeration() {
        for k in 2..=3u64 {
            for n in 1..=6u64 {
                let mut lyn = 0u128;
                let mut necklaces = HashSet::new();
                let total = (k as usize).pow(n as u32);
                for code in 0..total {
                    let mut w = Vec::with_capacity(n as usize);
                    let mut c = code;
                    for _ in 0..n {
                        w.push(c % k as usize);
                        c /= k as usize;
                    }
                    let p = PeriodicPath::new(k as usize, w.clone()).unwrap();
                    necklaces.insert(p.necklace());
                    if p.is_primitive() && p.necklace() == w {
                        lyn += 1;
                    }
                }
                assert_eq!(lyn, lyndon_count(k, n), "lyndon k={k} n={n}");
                assert_eq!(necklaces.len() as u128, necklace_count(k, n));
            }
        }
    }
}
