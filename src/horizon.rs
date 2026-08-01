//! The latent past and the latent future, as fields over one sheet.
//!
//! A circuit's causal structure is usually drawn as a **light cone**: a
//! set of (time, qubit) cells the answer can reach. That picture is
//! true and coarse. What the Heisenberg walk actually carries is
//! strictly more information, and it is quantitative — at every circuit
//! position `t` the backward observable
//!
//! ```text
//! O_t = U_{≥t}† P U_{≥t}
//! ```
//!
//! is a weighted sum of Paulis, and *how much of that weight touches
//! qubit `q`* is a number in `[0, 1]`, not a yes or no. So the cone is
//! only the support of a **field**:
//!
//! ```text
//! Φ_a(t, q) = ( Σ_{P : {P, a_q} = 0} |c_P| ) / Σ_P |c_P|      a ∈ {X, Y, Z}
//! ```
//!
//! and the field is what carries the physics. It is not a heuristic:
//! inserting `exp(−iδ a_q/2)` at position `t` moves the answer by at
//! most
//!
//! ```text
//! |Δ⟨P⟩| ≤ (|1 − cos δ| + |sin δ|) · Φ_a(t, q) · L1(O_t)
//! ```
//!
//! because only the anticommuting terms change at all, and each changes
//! by at most that factor times its own weight. `Φ = 0` is exactly the
//! statement "nothing you do to this qubit at this time can be seen",
//! which is the light cone; `Φ = 0.03` is the statement the light cone
//! cannot make. [`Horizon::influence`] is that field and
//! [`Horizon::perturbation_bound`] is the bound.
//!
//! ## The dual field
//!
//! The future depends locally; the past *commits* locally. Run the
//! circuit forward to `t` and read each qubit's Bloch length
//! `r_q = √(⟨X⟩² + ⟨Y⟩² + ⟨Z⟩²)`. `r_q = 1` says qubit `q` is in a pure
//! local state — the past has settled it, and it is entangled with
//! nothing. `r_q = 0` says the opposite: locally maximally mixed, every
//! bit of what it knows is held jointly with the rest of the register.
//! [`Horizon::commitment`] is that field, and it runs the other way in
//! time: the future's field grows backward from the observable, the
//! past's shrinks forward from `|0…0⟩`.
//!
//! Both live on the same `(cut, qubit)` sheet, and their contraction
//! `⟨ψ_t|O_t|ψ_t⟩` is **the same number at every cut** — that constant
//! is the conservation law of the picture, and
//! [`Horizon::value`] records it at every sampled position so it can be
//! seen rather than asserted.
//!
//! ## What the field shows that the cone cannot
//!
//! Three things, all measured rather than argued:
//!
//! * **Influence is not monotone.** A cone only ever grows. The field
//!   can *fall*: terms that anticommute with `a_q` can cancel against
//!   each other as the walk continues, so a qubit's grip on the answer
//!   weakens even though it stays inside the cone. That is interference
//!   in operator space, and [`Horizon::non_monotone_cells`] counts it.
//! * **The cone overstates dependence, usually by a lot.** The ratio of
//!   cells that are *in* the cone to cells whose influence is above any
//!   given threshold is the size of that overstatement.
//! * **Locality drains away under the future's own gaze.**
//!   [`Horizon::settled`] — the share of what the observable can still
//!   see that the input has committed to single-qubit marginals — starts
//!   at exactly 1 on a product input and falls as the circuit runs. The
//!   future keeps seeing the same qubits; what it sees stops being
//!   *about* them individually.
//! * **Where precision was spent, by qubit.** The journal already knows
//!   *when* a truncation discarded weight; attributing that weight to the
//!   qubits the discarded terms were supported on says *where*.
//!   [`Horizon::blame`] is the retrodictive field.

use crate::backend::{pauli_expectation, Backend, DenseState, MpsConfig, MpsState, SparseState};
use crate::error::{Error, Result};
use crate::gates::Pauli;
use crate::heisenberg::{commutes, step_through, PauliKey, PauliSum, Rotation, MAX_QUBITS};
use crate::scalar::C64;

/// How the forward half of the sheet is held.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Forward {
    /// Dense amplitudes — the reference, exact at any entanglement.
    Dense,
    /// Sparse amplitudes.
    Sparse,
    /// Matrix product state, with a bond ceiling.
    Mps {
        /// Bond ceiling.
        max_bond: usize,
    },
}

/// How a horizon is sampled and truncated.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Number of cuts sampled across the circuit; the sheet has
    /// `samples + 1` columns, including both ends.
    pub samples: usize,
    /// Drop backward terms below this `|c|`. Zero is exact.
    pub threshold: f64,
    /// How the forward state is represented.
    pub forward: Forward,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            samples: 24,
            threshold: 0.0,
            forward: Forward::Dense,
        }
    }
}

/// The three single-qubit axes, in the order the fields store them.
pub const AXES: [Pauli; 3] = [Pauli::X, Pauli::Y, Pauli::Z];

/// The sheet: both fields, the invariant, and the accounting, sampled at
/// a set of circuit positions.
#[derive(Debug, Clone)]
pub struct Horizon {
    /// Register width.
    pub qubits: usize,
    /// Total gates in the circuit.
    pub gates: usize,
    /// Circuit positions sampled, ascending, starting at `0` and ending
    /// at `gates`.
    pub cuts: Vec<usize>,
    /// `⟨ψ_t|O_t|ψ_t⟩` at each cut. Constant up to truncation — that is
    /// the point.
    pub value: Vec<f64>,
    /// `[cut][qubit][axis]` — the latent **future**: the L1 fraction of
    /// `O_t` anticommuting with that axis on that qubit.
    pub influence: Vec<Vec<[f64; 3]>>,
    /// `[cut][qubit]` — the latent **past**: the Bloch length of the
    /// forward state's single-qubit marginal.
    pub commitment: Vec<Vec<f64>>,
    /// `[cut][qubit]` — cumulative discarded `Σ|c|` attributed to the
    /// qubits the discarded terms were supported on.
    pub blame: Vec<Vec<f64>>,
    /// Terms in the backward sum at each cut.
    pub terms: Vec<usize>,
    /// `Σ|c|` of the backward sum at each cut.
    pub l1: Vec<f64>,
    /// Largest term support at each cut — the scrambling front.
    pub max_weight: Vec<usize>,
    /// Peak terms over the whole walk.
    pub peak_terms: usize,
}

impl Horizon {
    /// Column index of the cut nearest `gate`.
    pub fn column_of(&self, gate: usize) -> usize {
        self.cuts
            .iter()
            .enumerate()
            .min_by_key(|(_, &c)| c.abs_diff(gate))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// The influence field collapsed over axes: the largest of the three,
    /// which is the tightest single number bounding *any* single-qubit
    /// perturbation there.
    pub fn influence_max(&self, cut: usize, qubit: usize) -> f64 {
        self.influence[cut][qubit]
            .iter()
            .fold(0.0f64, |a, &b| a.max(b))
    }

    /// How far the answer can move if `exp(−iδ a_q/2)` is inserted at
    /// this cut — the field's operational meaning.
    ///
    /// Only anticommuting terms change at all, and each moves by at most
    /// `|1 − cos δ| + |sin δ|` times its own weight, so the bound is
    /// that factor times the anticommuting weight.
    pub fn perturbation_bound(&self, cut: usize, qubit: usize, axis: usize, delta: f64) -> f64 {
        let factor = (1.0 - delta.cos()).abs() + delta.sin().abs();
        factor * self.influence[cut][qubit][axis] * self.l1[cut]
    }

    /// Cells inside the backward light cone: influence strictly positive.
    pub fn cone_cells(&self) -> usize {
        self.count_cells(0.0)
    }

    /// Cells whose influence exceeds `threshold`.
    pub fn count_cells(&self, threshold: f64) -> usize {
        (0..self.cuts.len())
            .flat_map(|c| (0..self.qubits).map(move |q| (c, q)))
            .filter(|&(c, q)| self.influence_max(c, q) > threshold)
            .count()
    }

    /// How much the light cone overstates dependence at `threshold`: the
    /// ratio of cells in the cone to cells that actually matter that much.
    pub fn cone_overstatement(&self, threshold: f64) -> f64 {
        self.cone_cells() as f64 / self.count_cells(threshold).max(1) as f64
    }

    /// Cells where influence **fell** as the walk went further back —
    /// operator-space interference, which a light cone cannot express
    /// because a cone only grows.
    ///
    /// Counted in walk order: the backward walk visits cuts from `gates`
    /// down to `0`, so a fall means a later (earlier-in-circuit) column
    /// has less influence than the one after it.
    pub fn non_monotone_cells(&self) -> usize {
        let mut n = 0;
        for q in 0..self.qubits {
            for c in 0..self.cuts.len().saturating_sub(1) {
                // column c is EARLIER in the circuit than c+1, so the
                // walk reached it later
                if self.influence_max(c, q) < self.influence_max(c + 1, q) - 1e-12 {
                    n += 1;
                }
            }
        }
        n
    }

    /// The **front**: the latest circuit position at which each qubit is
    /// still inside the backward cone.
    ///
    /// The cone grows as the walk runs backward, so a qubit's influence
    /// is positive on every cut at or before this one and zero after it.
    /// Differencing the front across qubits is how a light-cone velocity
    /// gets measured. `None` for a qubit the observable never reaches.
    pub fn front(&self) -> Vec<Option<usize>> {
        (0..self.qubits)
            .map(|q| {
                (0..self.cuts.len())
                    .rev()
                    .find(|&c| self.influence_max(c, q) > 0.0)
                    .map(|c| self.cuts[c])
            })
            .collect()
    }

    /// How *full* the cone is: total influence mass divided by the number
    /// of cells inside it. A cone drawn as a set implicitly claims this
    /// is 1; it never is.
    pub fn cone_fill(&self) -> f64 {
        let mass: f64 = (0..self.cuts.len())
            .flat_map(|c| (0..self.qubits).map(move |q| (c, q)))
            .map(|(c, q)| self.influence_max(c, q))
            .sum();
        mass / self.cone_cells().max(1) as f64
    }

    /// Where the two fields overlap: `Φ_max · r_q`, the cells where the
    /// past has committed something the future can still see. This is
    /// the product the answer is actually assembled from.
    pub fn overlap(&self, cut: usize, qubit: usize) -> f64 {
        self.influence_max(cut, qubit) * self.commitment[cut][qubit]
    }

    /// **Visible mass**: `Σ_q Φ_max(t, q)` — how much of the register
    /// the future can still see at each cut. Rises as the walk runs
    /// backward and the operator spreads.
    pub fn visible(&self) -> Vec<f64> {
        (0..self.cuts.len())
            .map(|c| (0..self.qubits).map(|q| self.influence_max(c, q)).sum())
            .collect()
    }

    /// **Stake**: `Σ_q Φ_max(t, q) · r_q(t)` — the part of the register
    /// that is both still visible to the future and still locally
    /// committed by the past. Measured monotone non-increasing in `t`:
    /// as the circuit runs, the amount at stake only falls.
    pub fn stake(&self) -> Vec<f64> {
        (0..self.cuts.len())
            .map(|c| (0..self.qubits).map(|q| self.overlap(c, q)).sum())
            .collect()
    }

    /// **Settled fraction**: stake divided by visible mass — of what the
    /// future can still see, how much the past has committed *locally*.
    ///
    /// This is the readout neither field gives alone, and it is the one
    /// that says something surprising. It starts at exactly `1`: the
    /// input is a product state, so everything the operator can see is
    /// held in single-qubit marginals. It then **falls** — measured to
    /// ~0.74 on a six-layer Ising chain — because the information the
    /// observable depends on migrates out of the marginals and into
    /// correlations. The future keeps seeing the same qubits; what it
    /// sees stops being *about* them individually.
    pub fn settled(&self) -> Vec<f64> {
        let (stake, visible) = (self.stake(), self.visible());
        stake
            .iter()
            .zip(&visible)
            .map(|(s, v)| if *v > 0.0 { s / v } else { 0.0 })
            .collect()
    }

    /// Spread of the invariant across cuts — zero for an exact walk, and
    /// the truncation's real error otherwise.
    pub fn value_spread(&self) -> f64 {
        let (lo, hi) = self
            .value
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        if lo.is_finite() {
            hi - lo
        } else {
            0.0
        }
    }
}

/// The anticommuting L1 weight of a sum, per qubit and axis.
fn influence_of(sum: &PauliSum, qubits: usize) -> (Vec<[f64; 3]>, f64) {
    let mut field = vec![[0.0f64; 3]; qubits];
    let mut l1 = 0.0f64;
    for (key, coeff) in sum.terms() {
        let w = coeff.norm();
        l1 += w;
        let touched = (key.0 | key.1) & ((1u64 << qubits) - 1).max(if qubits == 64 { u64::MAX } else { 0 });
        let mut bits = touched;
        while bits != 0 {
            let q = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            for (a, axis) in [(0usize, (1u64 << q, 0u64)), (1, (1 << q, 1 << q)), (2, (0, 1 << q))] {
                if !commutes(key, axis) {
                    field[q][a] += w;
                }
            }
        }
    }
    if l1 > 0.0 {
        for cell in field.iter_mut() {
            for a in cell.iter_mut() {
                *a /= l1;
            }
        }
    }
    (field, l1)
}

/// Attribute a discarded term's weight to the qubits it acted on.
fn blame_into(blame: &mut [f64], key: PauliKey, weight: f64) {
    let mut bits = key.0 | key.1;
    while bits != 0 {
        let q = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        if q < blame.len() {
            blame[q] += weight;
        }
    }
}

/// Bloch lengths of every single-qubit marginal of a forward state.
fn commitment_of(state: &dyn Backend<C64>, qubits: usize) -> Result<Vec<f64>> {
    (0..qubits)
        .map(|q| {
            let mut sq = 0.0f64;
            for p in AXES {
                let e = pauli_expectation(state, &[(q, p)])?.re;
                sq += e * e;
            }
            Ok(sq.sqrt())
        })
        .collect()
}

/// Build the sheet: one backward walk recording the future field at each
/// sampled cut, one forward run recording the past field at the same
/// cuts, and their contraction at every column.
///
/// Exact when `cfg.threshold` is zero, in which case
/// [`Horizon::value_spread`] comes out at machine precision and the
/// invariant is visible rather than asserted.
pub fn horizon(
    observable: PauliKey,
    rotations: &[Rotation],
    qubits: usize,
    cfg: &Config,
) -> Result<Horizon> {
    if qubits == 0 || qubits > MAX_QUBITS {
        return Err(Error::InvalidState(format!(
            "horizon: {qubits} qubits outside 1..={MAX_QUBITS}"
        )));
    }
    let gates = rotations.len();
    let samples = cfg.samples.max(1);
    let mut cuts: Vec<usize> = (0..=samples).map(|i| i * gates / samples).collect();
    cuts.dedup();

    // ── backward: the latent future ──────────────────────────────────
    // The walk runs from the end to the start, so cuts are met in
    // descending order and the columns are filled back to front.
    let mut sum = PauliSum::from_key(observable);
    let mut running_blame = vec![0.0f64; qubits];
    let mut peak_terms = sum.len();
    let mut cols = Columns::new(cuts.len());

    // the last column is the cut at `gates`: the observable itself
    let mut next = cuts.len() - 1;
    while cuts[next] == gates {
        cols.record(next, &sum, &running_blame, qubits);
        if next == 0 {
            break;
        }
        next -= 1;
    }

    for i in (0..gates).rev() {
        let (mut stepped, _, _, _) = step_through(&sum, &rotations[i], None);
        if cfg.threshold > 0.0 {
            let dropped: Vec<(PauliKey, f64)> = stepped
                .terms()
                .filter(|(_, c)| c.norm() < cfg.threshold)
                .map(|(k, c)| (k, c.norm()))
                .collect();
            for (key, w) in dropped {
                blame_into(&mut running_blame, key, w);
            }
            stepped.truncate(cfg.threshold);
        }
        sum = stepped;
        peak_terms = peak_terms.max(sum.len());
        // position `i` is the cut *before* gate `i`
        while cuts[next] == i {
            cols.record(next, &sum, &running_blame, qubits);
            if next == 0 {
                break;
            }
            next -= 1;
        }
    }

    // ── forward: the latent past, and the contraction ────────────────
    let mut state: Box<dyn Backend<C64>> = match cfg.forward {
        Forward::Dense => Box::new(DenseState::<C64>::new(qubits)?),
        Forward::Sparse => Box::new(SparseState::<C64>::new(qubits)?),
        Forward::Mps { max_bond } => Box::new(MpsState::<C64>::with_config(
            qubits,
            MpsConfig {
                max_bond,
                trunc_tol: 1e-14,
            },
        )?),
    };
    let mut commitment = vec![Vec::new(); cuts.len()];
    let mut value = vec![0.0f64; cuts.len()];
    let mut col = 0usize;

    while col < cuts.len() && cuts[col] == 0 {
        settle(col, state.as_ref(), &cols, qubits, &mut commitment, &mut value)?;
        col += 1;
    }
    for (i, r) in rotations.iter().enumerate() {
        let (m, s) = r.gate()?;
        state.apply(&m, &s)?;
        while col < cuts.len() && cuts[col] == i + 1 {
            settle(col, state.as_ref(), &cols, qubits, &mut commitment, &mut value)?;
            col += 1;
        }
    }

    let Columns {
        influence,
        blame,
        terms,
        l1,
        max_weight,
        backward: _,
    } = cols;

    Ok(Horizon {
        qubits,
        gates,
        cuts,
        value,
        influence,
        commitment,
        blame,
        terms,
        l1,
        max_weight,
        peak_terms,
    })
}

/// The backward walk's per-column record, filled back to front.
struct Columns {
    influence: Vec<Vec<[f64; 3]>>,
    blame: Vec<Vec<f64>>,
    terms: Vec<usize>,
    l1: Vec<f64>,
    max_weight: Vec<usize>,
    /// The sums themselves, kept so the forward pass can contract
    /// against them without a second backward walk.
    backward: Vec<PauliSum>,
}

impl Columns {
    fn new(n: usize) -> Self {
        Columns {
            influence: vec![Vec::new(); n],
            blame: vec![Vec::new(); n],
            terms: vec![0; n],
            l1: vec![0.0; n],
            max_weight: vec![0; n],
            backward: vec![PauliSum::zero(); n],
        }
    }

    fn record(&mut self, col: usize, sum: &PauliSum, running_blame: &[f64], qubits: usize) {
        let (field, weight) = influence_of(sum, qubits);
        self.influence[col] = field;
        self.blame[col] = running_blame.to_vec();
        self.terms[col] = sum.len();
        self.l1[col] = weight;
        self.max_weight[col] = sum.max_weight();
        self.backward[col] = sum.clone();
    }
}

/// Read the past field at one column and contract it with the future.
fn settle(
    col: usize,
    state: &dyn Backend<C64>,
    cols: &Columns,
    qubits: usize,
    commitment: &mut [Vec<f64>],
    value: &mut [f64],
) -> Result<()> {
    commitment[col] = commitment_of(state, qubits)?;
    let mut acc = C64::new(0.0, 0.0);
    for (key, c) in cols.backward[col].terms() {
        acc += c * pauli_on_state(state, key)?;
    }
    value[col] = acc.re;
    Ok(())
}

/// `⟨ψ| X^x Z^z |ψ⟩` for the raw key, via the Hermitian form the
/// backends expose.
fn pauli_on_state(state: &dyn Backend<C64>, key: PauliKey) -> Result<C64> {
    let ops: Vec<(usize, Pauli)> = (0..MAX_QUBITS)
        .filter_map(|q| match (key.0 >> q & 1, key.1 >> q & 1) {
            (1, 0) => Some((q, Pauli::X)),
            (0, 1) => Some((q, Pauli::Z)),
            (1, 1) => Some((q, Pauli::Y)),
            _ => None,
        })
        .collect();
    let herm = pauli_expectation(state, &ops)?;
    Ok(herm / crate::heisenberg::axis_operator_phase(key))
}
