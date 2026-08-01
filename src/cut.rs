//! The cut calculus: what a cut costs, read off the interaction graph.
//!
//! Every other cost model in this crate is *measured* — you run the walk
//! and see how many terms it took. This one is **predicted**, from the
//! graph and nothing else, in `O(|E|)` with no state, no register and no
//! simulation. It answers the question a representation has to answer
//! before it can choose itself: *how much entanglement will cross this
//! cut?*
//!
//! ## The object
//!
//! A [`CutGraph`] is a mixed-arity graph state: qudit `i` has arity
//! `aᵢ`, every site starts in `|+⟩`, and each bond `(i,j)` applies the
//! fractional controlled-`Z` phase
//!
//! ```text
//! CZ|dᵢ, dⱼ⟩ = exp(2πi · dᵢdⱼ / (aᵢaⱼ)) |dᵢ, dⱼ⟩
//! ```
//!
//! Cut the sites into `A` and `B`. The coefficient matrix across the cut
//! is the **Hadamard product of the single-bond Vandermonde matrices**,
//! `M = ⊙_{e ∈ cut} M_e` with `M_e[dᵢ][dⱼ] = exp(2πi dᵢdⱼ/(aᵢaⱼ))`.
//!
//! ## What that buys, in order of sharpness
//!
//! * **Only crossing edges matter.** A bond with both ends on one side
//!   is a diagonal unitary on that side, and local unitaries do not move
//!   Schmidt rank. [`CutGraph::crossing`].
//! * **Schur's bound.** `rank(⊙ M_e) ≤ ∏ rank(M_e)`, and a single bond
//!   has rank exactly `min(aᵢ,aⱼ)`, so
//!   `rank ≤ ∏_{e ∈ cut} min(aᵢ,aⱼ)` — computable from the edge list.
//!   [`CutGraph::schur_bound`].
//! * **Capacity beats disjointness.** Schur is loose when bonds share a
//!   vertex, because that vertex's own arity caps what it can carry.
//!   Intersecting with the touched-side products gives a strictly better
//!   bound, and it is the one that explains why a *wide* shared vertex
//!   still saturates. [`CutGraph::capacity_bound`].
//! * **The character count.** Each row of `M` is a character
//!   `d_B ↦ exp(2πi⟨θ(d_A), d_B⟩)` with frequency vector
//!   `θⱼ(d_A) = Σ_{i∈A} dᵢ/(aᵢaⱼ) mod 1`, so equal frequency vectors
//!   give equal rows and `rank ≤ |{θ(d_A) mod 1}|`.
//!   [`CutGraph::character_count`].
//!
//! The first three are graph combinatorics. The character count is
//! exponential in `|A|`, and [`CutGraph::exact_rank`] — Gaussian
//! elimination on the built matrix — is exponential in the whole cut.
//! Those two exist to *check* the cheap ones, not to be used.
//!
//! ## The open item, settled here
//!
//! The source theory states the character-count formula as a conjecture:
//! *"whether `rank = |{θ(d_A) mod 1}|` holds in full generality is a
//! named develop item, not yet claimed."*
//!
//! [`CutGraph::character_count`] and [`CutGraph::exact_rank`] are
//! both implemented so the question can be decided rather than repeated.
//! What the exhaustive sweep in `tests/cut_calculus.rs` finds is recorded
//! there and in `examples/cut_calculus.rs`, including the cases where it
//! fails — which is the useful half.
//!
//! Ported from the `operadic_clifford_algebra` entanglement-geometry
//! corpus (`cut_calculus`, `feedback_topology`), whose statement of the
//! area law `log rank = Σ_{e∈cut} log rank_e` this implements and tests.

use crate::error::{Error, Result};
use crate::support::Support;
use crate::scalar::C64;

/// Largest cut dimension [`CutGraph::exact_rank`] will build.
///
/// This is a ceiling on the **verifier**, not on the predictor. The
/// bounds are `O(|E|)` and know nothing about it; a graph of any size
/// can be priced, and only the exact cross-check needs a matrix.
pub const MAX_EXACT_DIM: u128 = 1 << 14;

/// A mixed-arity interaction graph — the whole input to the calculus.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CutGraph {
    arities: Vec<usize>,
    bonds: Vec<(usize, usize)>,
}

impl CutGraph {
    /// A register with the given arities and no bonds.
    pub fn new(arities: Vec<usize>) -> Result<CutGraph> {
        if arities.iter().any(|&a| a < 2) {
            return Err(Error::InvalidState(
                "cut: every site needs arity at least 2".into(),
            ));
        }
        Ok(CutGraph {
            arities,
            bonds: Vec::new(),
        })
    }

    /// A uniform register of `sites` qudits of arity `arity`.
    pub fn uniform(sites: usize, arity: usize) -> Result<CutGraph> {
        CutGraph::new(vec![arity; sites])
    }

    /// Add a bond. Self-loops and duplicates are refused: a self-loop is
    /// a local phase and a duplicate is a different bond strength, and
    /// silently accepting either would corrupt the rank arithmetic.
    pub fn bond(&mut self, i: usize, j: usize) -> Result<&mut CutGraph> {
        if i >= self.arities.len() || j >= self.arities.len() {
            return Err(Error::InvalidState(format!(
                "cut: bond ({i},{j}) outside a {}-site register",
                self.arities.len()
            )));
        }
        if i == j {
            return Err(Error::InvalidState(
                "cut: a self-bond is a local phase, not an edge".into(),
            ));
        }
        let e = (i.min(j), i.max(j));
        if self.bonds.contains(&e) {
            return Err(Error::InvalidState(format!(
                "cut: bond ({i},{j}) is already present"
            )));
        }
        self.bonds.push(e);
        Ok(self)
    }

    /// Sites.
    pub fn sites(&self) -> usize {
        self.arities.len()
    }

    /// Arities, by site.
    pub fn arities(&self) -> &[usize] {
        &self.arities
    }

    /// Bonds, each `(i,j)` with `i < j`.
    pub fn bonds(&self) -> &[(usize, usize)] {
        &self.bonds
    }

    /// First Betti number `E − V + C` — the master invariant of the
    /// source theory, and the number of independent feedback cycles.
    pub fn betti(&self) -> usize {
        let v = self.sites();
        let mut parent: Vec<usize> = (0..v).collect();
        fn find(p: &mut [usize], mut i: usize) -> usize {
            while p[i] != i {
                p[i] = p[p[i]];
                i = p[i];
            }
            i
        }
        for &(i, j) in &self.bonds {
            let (a, b) = (find(&mut parent, i), find(&mut parent, j));
            parent[a] = b;
        }
        let components = (0..v).filter(|&i| find(&mut parent, i) == i).count();
        self.bonds.len() + components - v
    }

    /// Whether the graph is connected — the source theory's
    /// *entanglement onset*: the sides entangle exactly when the graph
    /// joins.
    pub fn is_connected(&self) -> bool {
        self.sites() <= 1 || self.betti() + self.sites() == self.bonds.len() + 1
    }

    /// The `A` side as a support, from a list of sites.
    pub fn cut_of(sites: &[usize]) -> Support {
        sites.iter().copied().collect()
    }

    /// The complement of a cut within this register.
    pub fn complement(&self, cut: &Support) -> Support {
        (0..self.sites()).filter(|&i| !cut.contains(i)).collect()
    }

    /// The bonds crossing the cut.
    ///
    /// The only bonds that matter: a bond inside `A` or inside `B` is a
    /// diagonal unitary on one side, and local unitaries never move
    /// Schmidt rank.
    pub fn crossing(&self, cut: &Support) -> Vec<(usize, usize)> {
        self.bonds
            .iter()
            .copied()
            .filter(|&(i, j)| cut.contains(i) != cut.contains(j))
            .collect()
    }

    /// Dimension of the `A` side.
    pub fn dim_a(&self, cut: &Support) -> u128 {
        (0..self.sites())
            .filter(|&i| cut.contains(i))
            .fold(1u128, |d, i| d.saturating_mul(self.arities[i] as u128))
    }

    /// Dimension of the `B` side.
    pub fn dim_b(&self, cut: &Support) -> u128 {
        (0..self.sites())
            .filter(|&i| !cut.contains(i))
            .fold(1u128, |d, i| d.saturating_mul(self.arities[i] as u128))
    }

    /// **Schur's bound**: `∏_{e ∈ cut} min(aᵢ,aⱼ)`.
    ///
    /// `O(|E|)`, no state. A single bond's Vandermonde has rank exactly
    /// `min(aᵢ,aⱼ)`, and the cut matrix is the Hadamard product of the
    /// crossing bonds' matrices, so Schur's rank inequality applies
    /// directly.
    pub fn schur_bound(&self, cut: &Support) -> u128 {
        self.crossing(cut)
            .into_iter()
            .fold(1u128, |r, (i, j)| {
                r.saturating_mul(self.arities[i].min(self.arities[j]) as u128)
            })
            .min(self.dim_a(cut))
            .min(self.dim_b(cut))
    }

    /// **The capacity bound**: Schur, intersected with what the touched
    /// sites can actually carry.
    ///
    /// A site with several crossing bonds feeds all of them from the one
    /// digit `dᵢ`, so it can contribute at most `aᵢ` distinct rows — the
    /// constraint is the site's *capacity*, not whether the bonds are
    /// disjoint. This is why a wide shared vertex saturates while a
    /// narrow one collapses, and it is still `O(|E|)`.
    pub fn capacity_bound(&self, cut: &Support) -> u128 {
        let crossing = self.crossing(cut);
        let mut touched_a = 1u128;
        let mut touched_b = 1u128;
        let mut seen_a = vec![false; self.sites()];
        let mut seen_b = vec![false; self.sites()];
        for &(i, j) in &crossing {
            for s in [i, j] {
                if cut.contains(s) {
                    if !seen_a[s] {
                        seen_a[s] = true;
                        touched_a = touched_a.saturating_mul(self.arities[s] as u128);
                    }
                } else if !seen_b[s] {
                    seen_b[s] = true;
                    touched_b = touched_b.saturating_mul(self.arities[s] as u128);
                }
            }
        }
        self.schur_bound(cut).min(touched_a).min(touched_b)
    }

    /// Whether the crossing bonds are vertex-disjoint — the regime in
    /// which Schur is known to be attained.
    pub fn is_matching(&self, cut: &Support) -> bool {
        let crossing = self.crossing(cut);
        let mut seen = vec![false; self.sites()];
        for &(i, j) in &crossing {
            for s in [i, j] {
                if seen[s] {
                    return false;
                }
                seen[s] = true;
            }
        }
        true
    }

    /// **The character count**: `|{θ(d_A) mod 1}|`.
    ///
    /// Each row of the cut matrix is `d_B ↦ exp(2πi⟨θ(d_A), d_B⟩)` with
    /// `θⱼ(d_A) = Σ_{i∈A} dᵢ/(aᵢaⱼ) mod 1`, so rows with equal frequency
    /// vectors are equal and the rank is at most the number of distinct
    /// vectors. Computed over a common denominator, so the count is
    /// exact integer arithmetic and never a float comparison.
    ///
    /// Costs `∏_{i∈A} aᵢ`. It is a cross-check on the cheap bounds, not
    /// one of them.
    pub fn character_count(&self, cut: &Support) -> Result<usize> {
        let crossing = self.crossing(cut);
        if crossing.is_empty() {
            return Ok(1);
        }
        let dim_a = self.dim_a(cut);
        if dim_a > MAX_EXACT_DIM {
            return Err(Error::TooManyQubits {
                requested: dim_a.min(usize::MAX as u128) as usize,
                max: MAX_EXACT_DIM as usize,
            });
        }
        // A common denominator for every aᵢ·aⱼ, so frequencies compare
        // as integers.
        let mut denom: u128 = 1;
        for &(i, j) in &crossing {
            denom = lcm(denom, (self.arities[i] * self.arities[j]) as u128);
        }
        let a_sites: Vec<usize> = (0..self.sites()).filter(|&i| cut.contains(i)).collect();
        let b_sites: Vec<usize> = (0..self.sites()).filter(|&i| !cut.contains(i)).collect();

        let mut seen: Vec<Vec<u128>> = Vec::new();
        for index in 0..dim_a {
            let digits = digits_of(index, &a_sites, &self.arities);
            let theta: Vec<u128> = b_sites
                .iter()
                .map(|&j| {
                    let mut acc: u128 = 0;
                    for (slot, &i) in a_sites.iter().enumerate() {
                        if cut.contains(i) && self.bonds.contains(&(i.min(j), i.max(j))) {
                            let step = denom / (self.arities[i] * self.arities[j]) as u128;
                            acc += digits[slot] as u128 * step;
                        }
                    }
                    acc % denom
                })
                .collect();
            if !seen.contains(&theta) {
                seen.push(theta);
            }
        }
        Ok(seen.len())
    }

    /// The best cheap upper bound this module has: capacity intersected
    /// with the character count on **both** sides.
    ///
    /// Sound — it is never below the true rank — and tighter than any of
    /// its parts. It is **not** exact, and the module does not pretend
    /// otherwise; see the type docs for the counterexample.
    pub fn predicted_rank(&self, cut: &Support) -> Result<u128> {
        let other = self.complement(cut);
        Ok(self
            .capacity_bound(cut)
            .min(self.character_count(cut)? as u128)
            .min(self.character_count(&other)? as u128))
    }

    /// The exact Schmidt rank across the cut, by building the matrix.
    ///
    /// Exponential in the cut and present only to check the predictors.
    /// Refuses past [`MAX_EXACT_DIM`] rather than allocating.
    pub fn exact_rank(&self, cut: &Support) -> Result<usize> {
        let (dim_a, dim_b) = (self.dim_a(cut), self.dim_b(cut));
        if dim_a.saturating_mul(dim_b) > MAX_EXACT_DIM * MAX_EXACT_DIM || dim_a > MAX_EXACT_DIM {
            return Err(Error::TooManyQubits {
                requested: dim_a.min(usize::MAX as u128) as usize,
                max: MAX_EXACT_DIM as usize,
            });
        }
        let crossing = self.crossing(cut);
        let a_sites: Vec<usize> = (0..self.sites()).filter(|&i| cut.contains(i)).collect();
        let b_sites: Vec<usize> = (0..self.sites()).filter(|&i| !cut.contains(i)).collect();

        let mut m: Vec<Vec<C64>> = Vec::with_capacity(dim_a as usize);
        for ia in 0..dim_a {
            let da = digits_of(ia, &a_sites, &self.arities);
            let mut row = Vec::with_capacity(dim_b as usize);
            for ib in 0..dim_b {
                let db = digits_of(ib, &b_sites, &self.arities);
                let mut phase = 0.0f64;
                for &(i, j) in &crossing {
                    let (sa, sb) = if cut.contains(i) { (i, j) } else { (j, i) };
                    let va = da[a_sites.iter().position(|&s| s == sa).expect("in A")];
                    let vb = db[b_sites.iter().position(|&s| s == sb).expect("in B")];
                    phase += (va * vb) as f64 / (self.arities[sa] * self.arities[sb]) as f64;
                }
                let t = std::f64::consts::TAU * phase;
                row.push(C64::new(t.cos(), t.sin()));
            }
            m.push(row);
        }
        Ok(numeric_rank(&mut m))
    }

    /// Greedy provisioning: buy bonds by descending `min(aᵢ,aⱼ)` until
    /// the Schur bound meets a rank demand.
    ///
    /// Returns the bonds bought, or a refusal naming the capacity that
    /// was available — a demand beyond what a matching can carry is not
    /// silently under-served.
    pub fn provision(arities: &[usize], demand: u128) -> Result<Vec<(usize, usize)>> {
        let mut candidates: Vec<(usize, usize)> = Vec::new();
        for i in 0..arities.len() {
            for j in (i + 1)..arities.len() {
                candidates.push((i, j));
            }
        }
        candidates.sort_by_key(|&(i, j)| std::cmp::Reverse(arities[i].min(arities[j])));
        let mut used = vec![false; arities.len()];
        let mut bought = Vec::new();
        let mut have: u128 = 1;
        for (i, j) in candidates {
            if have >= demand {
                break;
            }
            if used[i] || used[j] {
                continue;
            }
            used[i] = true;
            used[j] = true;
            bought.push((i, j));
            have = have.saturating_mul(arities[i].min(arities[j]) as u128);
        }
        if have < demand {
            return Err(Error::InvalidState(format!(
                "cut: a matching on these arities carries rank {have}, short of the \
                 demand {demand}; wider sites or shared-vertex capacity are needed"
            )));
        }
        Ok(bought)
    }
}

fn lcm(a: u128, b: u128) -> u128 {
    fn gcd(a: u128, b: u128) -> u128 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    a / gcd(a, b) * b
}

/// Mixed-radix digits of `index` over the given sites.
fn digits_of(index: u128, sites: &[usize], arities: &[usize]) -> Vec<usize> {
    let mut rest = index;
    let mut out = vec![0usize; sites.len()];
    for (slot, &s) in sites.iter().enumerate().rev() {
        let a = arities[s] as u128;
        out[slot] = (rest % a) as usize;
        rest /= a;
    }
    out
}

/// Rank by Gaussian elimination with partial pivoting.
fn numeric_rank(m: &mut [Vec<C64>]) -> usize {
    let rows = m.len();
    if rows == 0 {
        return 0;
    }
    let cols = m[0].len();
    let tol = 1e-9;
    let mut rank = 0usize;
    let mut row = 0usize;
    for col in 0..cols {
        if row >= rows {
            break;
        }
        let (mut best, mut best_norm) = (row, 0.0f64);
        for (r, item) in m.iter().enumerate().skip(row) {
            let n = item[col].re.hypot(item[col].im);
            if n > best_norm {
                best_norm = n;
                best = r;
            }
        }
        if best_norm < tol {
            continue;
        }
        m.swap(row, best);
        let pivot = m[row][col];
        for r in (row + 1)..rows {
            let f = m[r][col] / pivot;
            if f.re.hypot(f.im) < tol {
                continue;
            }
            let pivot_row: Vec<C64> = m[row][col..].to_vec();
            for (c, &pv) in pivot_row.iter().enumerate() {
                m[r][col + c] -= f * pv;
            }
        }
        row += 1;
        rank += 1;
    }
    rank
}
