//! Clifford frames: the frame group upgraded from local unitaries to the
//! full Clifford group, with the T-count as the measured cost currency.
//!
//! A [`CliffordFramedState`] represents the physical state as
//! `C · stored`, where `C` is a Clifford unitary held as **metadata** — a
//! stabilizer tableau (the images `C† X_q C`, `C† Z_q C`) plus a replay
//! log — and `stored` is a concrete [`SparseState`]. Gates are executed in
//! whichever form is cheapest:
//!
//! * **Clifford gates absorb** into the frame (`C ← g·C`): a tableau
//!   update and a log push, zero amplitude work. A circuit of Clifford
//!   gates keeps the stored support at **1** at any width the sparse map
//!   allows — the *evolution* sector of Gottesman–Knill reproduced by the
//!   frame mechanism. Two scope statements keep that label honest. The
//!   free sector cannot exceed the Clifford group: absorption's acceptance
//!   test — every conjugated generator must land on a single ±1 Pauli — is
//!   the definition of Pauli-normalizer membership (verified bidirectionally
//!   against an independent dense ground truth over the whole registry in
//!   `tests/clifford_frames.rs`), so a non-Clifford gate *cannot* ride for
//!   free; anything else would be a BQP = BPP claim. Measurement runs
//!   natively: [`CliffordFramedState::measure_pauli`] (and `measure`,
//!   which is `Z_q` through it) projects the *stored* state with the
//!   conjugated string — no flush, the frame survives — at the same cost
//!   envelope as the rotations: `O(support)` per measurement, at most 2×
//!   growth. That growth **compounds**: a long adaptive sequence drifts
//!   the stored state away from the frame (the register collapses toward
//!   a basis state `|b⟩` while the stored side becomes `C†|b⟩`,
//!   generically full-support), so `m` measurements are bounded by
//!   `2^m`, not by a polynomial — measured honestly in the tests, cheap
//!   for the fresh-ancilla feedback loops the lift generates, and the
//!   reason *frame repair on measurement* (the true tableau update, which
//!   would re-align the frame after each projection) is a named roadmap
//!   rung rather than a claimed capability. Full amplitude-vector
//!   extraction and batch `sample()` still flush — extracting `2^n`
//!   numbers is not a Gottesman–Knill capability and never was.
//! * **Pauli-axis rotations conjugate**: `exp(−iθ/2·P)` becomes
//!   `exp(−iθ/2·C†PC)`, another signed Pauli string read off the tableau,
//!   applied natively by [`SparseState::apply_pauli_rotation`] in
//!   `O(support)` with at most 2× growth — *independent of the string's
//!   weight*. A T gate costs one such rotation, so a Clifford+T circuit's
//!   stored support is bounded by `2^t` in its T-count `t`, not by
//!   `2^n` in its width or by its gate count.
//! * **Diagonal gates decompose** (Walsh–Hadamard) into Z-string
//!   rotations, each conjugated the same way — `cp`, `crz`, `ccz`, and
//!   small phase oracles all pass through the frame exactly.
//! * **Generic single-qubit gates split** (ZYZ) into three axis rotations.
//! * **Everything else flushes**: the log replays into the stored state
//!   (`C·stored` materialized), the tableau resets, and the gate applies
//!   raw — correctness over cleverness, verified by conformance.
//!
//! Measurement runs natively through the tableau (no flush); amplitude
//! queries, batch sampling and computational projection flush lazily
//! first, so the [`Backend`] contract is met exactly. The
//! representational claims live in [`CliffordFramedState::stats`],
//! [`CliffordFramedState::stored_nonzero_count`] and
//! [`CliffordFramedState::peak_stored_support`] — read them *before* an
//! observation flush, because materializing physical amplitudes
//! necessarily pays the physical support.
//!
//! What this buys and what it cannot: the frame walks gates through the
//! symplectic group Sp(2n, 𝔽₂) at metadata cost, so the *Clifford part*
//! of a circuit is free; what remains in the stored state is the
//! non-Clifford residue ("magic"). The `2^t` here is the crude product
//! bound — stabilizer-rank compression (≈ 2^{0.4t}) and native Pauli
//! measurement through the tableau are roadmap rungs. A frame that
//! absorbed *everything* would prove BQP = BPP; the measured exchange
//! rate into T-count currency is exactly the wall this backend makes
//! visible.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use super::sparse::SparseState;
use super::Backend;
use crate::error::{Error, Result};
use crate::math::{c64, cis, GateMatrix, UNITARY_TOL};
use crate::rng::Prng;
use crate::scalar::{Scalar, C64};

/// Widest gate the Clifford / Pauli-axis recognizers will decompose
/// (recognition cost is `O(8^k)`).
pub const CLIFFORD_RECOGNITION_MAX: usize = 3;

/// Widest diagonal the Walsh–Hadamard rotation decomposition accepts
/// (up to `2^k − 1` Z-string rotations).
pub const CLIFFORD_DIAGONAL_MAX: usize = 5;

/// Tolerance for numeric recognition: Pauli-basis coefficients below this
/// are structural zeros, and Clifford images must match ±1 this closely.
const RECOGNITION_TOL: f64 = 1e-10;

/// Squared-magnitude tolerance for off-diagonal entries when detecting
/// that a matrix is diagonal.
const DIAG_TOL: f64 = 1e-24;

/// A signed Hermitian Pauli string `σ · i^{|x∧z|} · X^x Z^z` on up to 63
/// qubits: bit `q` of `x`/`z` puts X/Z on qubit `q`, a bit in both is Y,
/// and `negative` is the sign `σ`. This normalization is Hermitian with
/// eigenvalues ±1 for every mask pair, which is why Clifford conjugation
/// stays inside the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PauliString {
    /// X-support mask.
    pub x: u64,
    /// Z-support mask.
    pub z: u64,
    /// Overall sign σ.
    pub negative: bool,
}

impl PauliString {
    /// Number of non-identity tensor factors.
    pub fn weight(&self) -> usize {
        (self.x | self.z).count_ones() as usize
    }
}

/// `i^phase · X^x Z^z` — the raw (possibly non-Hermitian) intermediate
/// form produced while multiplying strings.
#[derive(Debug, Clone, Copy)]
struct RawPauli {
    phase: u32,
    x: u64,
    z: u64,
}

impl RawPauli {
    fn from_hermitian(p: PauliString) -> Self {
        let y = (p.x & p.z).count_ones();
        RawPauli {
            phase: (y + if p.negative { 2 } else { 0 }) & 3,
            x: p.x,
            z: p.z,
        }
    }

    /// `self ← self · rhs`, using
    /// `X^{x₁}Z^{z₁} · X^{x₂}Z^{z₂} = (−1)^{|z₁∧x₂|} X^{x₁⊕x₂} Z^{z₁⊕z₂}`.
    fn mul(&mut self, rhs: RawPauli) {
        let swaps = (self.z & rhs.x).count_ones();
        self.phase = (self.phase + rhs.phase + 2 * (swaps & 1)) & 3;
        self.x ^= rhs.x;
        self.z ^= rhs.z;
    }

    fn into_hermitian(self) -> PauliString {
        let y = (self.x & self.z).count_ones() & 3;
        let rel = (self.phase + 4 - y) & 3;
        debug_assert_eq!(
            rel & 1,
            0,
            "product of commuting strings must stay Hermitian"
        );
        PauliString {
            x: self.x,
            z: self.z,
            negative: rel == 2,
        }
    }
}

/// Images of the generators `X_q`, `Z_q` under `C†(·)C` for the current
/// frame Clifford `C` — a stabilizer tableau, used to conjugate Pauli
/// strings into the stored basis in `O(weight)` mask algebra.
#[derive(Debug, Clone)]
struct CliffordTableau {
    x_images: Vec<PauliString>,
    z_images: Vec<PauliString>,
}

impl CliffordTableau {
    fn identity(n: usize) -> Self {
        CliffordTableau {
            x_images: (0..n)
                .map(|q| PauliString {
                    x: 1u64 << q,
                    z: 0,
                    negative: false,
                })
                .collect(),
            z_images: (0..n)
                .map(|q| PauliString {
                    x: 0,
                    z: 1u64 << q,
                    negative: false,
                })
                .collect(),
        }
    }

    /// `C† P C` for an arbitrary signed Hermitian string, as the phased
    /// product of generator images: `P = σ i^{|x∧z|} · Π X_q · Π Z_q`.
    fn image(&self, p: PauliString) -> PauliString {
        let y = (p.x & p.z).count_ones();
        let mut acc = RawPauli {
            phase: (y + if p.negative { 2 } else { 0 }) & 3,
            x: 0,
            z: 0,
        };
        let mut xs = p.x;
        while xs != 0 {
            let q = xs.trailing_zeros() as usize;
            xs &= xs - 1;
            acc.mul(RawPauli::from_hermitian(self.x_images[q]));
        }
        let mut zs = p.z;
        while zs != 0 {
            let q = zs.trailing_zeros() as usize;
            zs &= zs - 1;
            acc.mul(RawPauli::from_hermitian(self.z_images[q]));
        }
        acc.into_hermitian()
    }
}

/// Execution counters: how the wrapper routed its gates, and how heavy the
/// conjugated rotations got.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CliffordFrameStats {
    /// Gates recognized as Clifford and absorbed into the frame
    /// (tableau update + log push; zero amplitude work).
    pub absorbed_clifford: usize,
    /// Gates recognized as single Pauli-axis rotations and applied
    /// natively through the tableau.
    pub axis_rotations: usize,
    /// Z-string rotations produced by Walsh–Hadamard decompositions of
    /// diagonal gates (several may serve one gate).
    pub diagonal_rotations: usize,
    /// Generic single-qubit gates split into ZYZ rotation triples.
    pub zyz_decompositions: usize,
    /// Gates applied raw after a frame flush (no recognition matched).
    pub raw_fallbacks: usize,
    /// Flush events (log replays into the stored state).
    pub flushes: usize,
    /// Logged Clifford gates replayed by flushes.
    pub replayed_gates: usize,
    /// Widest conjugated Pauli string applied — the number the tableau
    /// keeps *out* of the amplitude work.
    pub max_axis_weight: usize,
    /// Measurements performed natively through the tableau — no flush,
    /// the frame survives.
    pub native_measurements: usize,
}

enum LoggedKernel<S: Scalar> {
    Matrix(GateMatrix<S>),
    Diagonal(Vec<S>),
}

struct LoggedGate<S: Scalar> {
    kernel: LoggedKernel<S>,
    qubits: Vec<usize>,
}

struct Core<S: Scalar> {
    state: SparseState<S>,
    tableau: CliffordTableau,
    log: Vec<LoggedGate<S>>,
    stats: CliffordFrameStats,
    peak_inner_memory: usize,
    peak_stored_support: usize,
}

impl<S: Scalar> Core<S> {
    fn note_peak(&mut self) {
        self.peak_inner_memory = self.peak_inner_memory.max(self.state.memory_bytes());
        self.peak_stored_support = self.peak_stored_support.max(self.state.nonzero_count());
    }

    /// Replay the log into the stored state (`stored ← C·stored`), reset
    /// the tableau. Interleaved rotations already live in the stored state
    /// in conjugated form, so replaying the Cliffords in absorption order
    /// reproduces the physical circuit exactly:
    /// `C·r₂′·r₁′ = g₂·g₁·(g₁†g₂†r₂g₂g₁)·(g₁†r₁g₁) = r₂·g₂·r₁·g₁`.
    fn flush(&mut self, num_qubits: usize) -> Result<()> {
        if self.log.is_empty() {
            return Ok(());
        }
        self.stats.flushes += 1;
        for gate in std::mem::take(&mut self.log) {
            match &gate.kernel {
                LoggedKernel::Matrix(m) => self.state.apply(m, &gate.qubits)?,
                LoggedKernel::Diagonal(d) => self.state.apply_diagonal(d, &gate.qubits)?,
            }
            self.stats.replayed_gates += 1;
            self.note_peak();
        }
        self.tableau = CliffordTableau::identity(num_qubits);
        Ok(())
    }

    /// Apply `phase · exp(−iθ/2·P)` to the *physical* state by conjugating
    /// `P` through the tableau and rotating the stored state natively.
    fn rotate(&mut self, theta: f64, p: PauliString, phase: C64) -> Result<()> {
        let img = self.tableau.image(p);
        self.stats.max_axis_weight = self.stats.max_axis_weight.max(img.weight());
        self.state
            .apply_pauli_rotation(theta, img.x, img.z, img.negative)?;
        self.global_phase(phase)?;
        self.note_peak();
        Ok(())
    }

    /// Multiply every stored amplitude by a unit scalar (as a 1-qubit
    /// "diagonal" with equal entries — no dedicated backend API needed).
    fn global_phase(&mut self, phase: C64) -> Result<()> {
        if (phase - c64(1.0, 0.0)).norm_sqr() < 1e-24 {
            return Ok(());
        }
        let p = S::try_from_c64(phase).ok_or_else(|| Error::UnsupportedForAlgebra {
            gate: "global phase".into(),
            algebra: S::algebra_name(),
        })?;
        self.state.apply_diagonal(&[p, p], &[0])
    }
}

/// `i^k` as a complex unit.
fn unit_phase(k: u32) -> C64 {
    match k & 3 {
        0 => c64(1.0, 0.0),
        1 => c64(0.0, 1.0),
        2 => c64(-1.0, 0.0),
        _ => c64(0.0, -1.0),
    }
}

/// Local sub-index bitmask → global qubit mask via the gate's qubit list.
fn to_global(local: usize, qubits: &[usize]) -> u64 {
    let mut mask = 0u64;
    for (b, &q) in qubits.iter().enumerate() {
        if (local >> b) & 1 == 1 {
            mask |= 1u64 << q;
        }
    }
    mask
}

/// View a gate matrix as complex entries. Sound for the constructor-gated
/// algebras (`DIM ≤ 2`, commutative division, containing `i`): the
/// coordinate vector is exactly (re, im).
fn scalar_to_c64<S: Scalar>(v: S) -> C64 {
    let co = v.coeffs();
    c64(co[0], co.get(1).copied().unwrap_or(0.0))
}

fn matrix_to_c64<S: Scalar>(m: &GateMatrix<S>) -> Vec<C64> {
    m.data().iter().map(|&v| scalar_to_c64(v)).collect()
}

/// `A = M† · (i^{|px∧pz|} X^{px} Z^{pz}) · M` over `d × d` complex
/// matrices — the numeric side of Clifford recognition.
fn conjugate_generator(m: &[C64], d: usize, px: usize, pz: usize) -> Vec<C64> {
    let iy = unit_phase((px & pz).count_ones());
    let mut pm = vec![c64(0.0, 0.0); d * d];
    for i in 0..d {
        let src = i ^ px;
        let ph = if (src & pz).count_ones() & 1 == 1 {
            -iy
        } else {
            iy
        };
        for col in 0..d {
            pm[i * d + col] = ph * m[src * d + col];
        }
    }
    let mut a = vec![c64(0.0, 0.0); d * d];
    for r in 0..d {
        for col in 0..d {
            let mut acc = c64(0.0, 0.0);
            for i in 0..d {
                acc += m[i * d + r].conj() * pm[i * d + col];
            }
            a[r * d + col] = acc;
        }
    }
    a
}

/// Nonzero coefficients of `A = Σ c_{x,z} · i^{|x∧z|} X^x Z^z` in the
/// Hermitian Pauli basis, via `c_Q = tr(Q·A)/d` computed along `Q`'s
/// monomial structure (`O(2^k)` per string, `O(8^k)` total).
fn pauli_decompose(a: &[C64], k: usize, tol: f64) -> Vec<(usize, usize, C64)> {
    let d = 1usize << k;
    let mut out = Vec::new();
    for px in 0..d {
        for pz in 0..d {
            let iy = unit_phase((px & pz).count_ones());
            let mut acc = c64(0.0, 0.0);
            for i in 0..d {
                let j = i ^ px;
                let sign = if (j & pz).count_ones() & 1 == 1 {
                    -1.0
                } else {
                    1.0
                };
                acc += iy.scale(sign) * a[j * d + i];
            }
            let coeff = acc / d as f64;
            if coeff.norm_sqr() > tol * tol {
                out.push((px, pz, coeff));
            }
        }
    }
    out
}

/// If `M` is Clifford, the images `M† X_b M`, `M† Z_b M` for every local
/// qubit `b`, as global-masked signed strings; `None` when any generator
/// fails to land on a single ±1 Pauli.
fn recognize_clifford(
    mc: &[C64],
    k: usize,
    qubits: &[usize],
) -> Option<Vec<(PauliString, PauliString)>> {
    let d = 1usize << k;
    let mut images = Vec::with_capacity(k);
    for b in 0..k {
        let mut pair = Vec::with_capacity(2);
        for (px, pz) in [(1usize << b, 0usize), (0usize, 1usize << b)] {
            let a = conjugate_generator(mc, d, px, pz);
            let dec = pauli_decompose(&a, k, RECOGNITION_TOL);
            if dec.len() != 1 {
                return None;
            }
            let (ix, iz, c) = dec[0];
            let negative = if (c - c64(1.0, 0.0)).norm_sqr() < RECOGNITION_TOL * RECOGNITION_TOL {
                false
            } else if (c + c64(1.0, 0.0)).norm_sqr() < RECOGNITION_TOL * RECOGNITION_TOL {
                true
            } else {
                return None;
            };
            pair.push(PauliString {
                x: to_global(ix, qubits),
                z: to_global(iz, qubits),
                negative,
            });
        }
        images.push((pair[0], pair[1]));
    }
    Some(images)
}

/// If `M = e^{iφ}·exp(−iθ/2·P)` for a single Pauli `P`, the parameters
/// `(θ, e^{iφ}, local x, local z)`. The Pauli support of such a matrix is
/// exactly `{I, P}` with `c_I = e^{iφ}cos(θ/2)`, `c_P = −i·e^{iφ}sin(θ/2)`
/// — phase-locked and unit-normed, which is what the residual checks
/// enforce (they also reject non-unitary raw ops).
fn recognize_axis(mc: &[C64], k: usize) -> Option<(f64, C64, usize, usize)> {
    let dec = pauli_decompose(mc, k, RECOGNITION_TOL);
    if dec.is_empty() || dec.len() > 2 {
        return None;
    }
    let mut u = c64(0.0, 0.0);
    let mut axis: Option<(usize, usize)> = None;
    let mut v = c64(0.0, 0.0);
    for &(px, pz, c) in &dec {
        if px == 0 && pz == 0 {
            u = c;
        } else if axis.is_none() {
            axis = Some((px, pz));
            v = c64(0.0, 1.0) * c; // v = i·c_P = e^{iφ} sin(θ/2)
        } else {
            return None;
        }
    }
    let (px, pz) = axis.unwrap_or((0, 0));
    let phi = if u.norm_sqr() >= v.norm_sqr() {
        u.arg()
    } else {
        v.arg()
    };
    let e = cis(-phi);
    let (ru, iu) = ((u * e).re, (u * e).im);
    let (rv, iv) = ((v * e).re, (v * e).im);
    if iu.abs() > 1e-9 || iv.abs() > 1e-9 || (ru * ru + rv * rv - 1.0).abs() > 1e-9 {
        return None;
    }
    let theta = 2.0 * rv.atan2(ru);
    Some((theta, cis(phi), px, pz))
}

/// Diagonal entries if all off-diagonals vanish.
fn extract_diagonal(mc: &[C64], d: usize) -> Option<Vec<C64>> {
    for r in 0..d {
        for c in 0..d {
            if r != c && mc[r * d + c].norm_sqr() > DIAG_TOL {
                return None;
            }
        }
    }
    Some((0..d).map(|i| mc[i * d + i]).collect())
}

/// ZYZ angles `(α, β, γ, δ)` with `U = e^{iα}·RZ(β)·RY(γ)·RZ(δ)` for a
/// generic 2×2 unitary, verified by rebuilding.
fn recognize_zyz(mc: &[C64]) -> Option<(f64, f64, f64, f64)> {
    let (a, b, c, d) = (mc[0], mc[1], mc[2], mc[3]);
    let gamma = 2.0 * c.norm().atan2(a.norm());
    let (alpha, beta, delta) = if a.norm() > 1e-9 && c.norm() > 1e-9 {
        (
            (a.arg() + d.arg()) / 2.0,
            c.arg() - a.arg(),
            d.arg() - c.arg(),
        )
    } else if a.norm() <= 1e-9 {
        // Anti-diagonal: β and δ only appear as β−δ; pin δ = 0.
        (((-b).arg() + c.arg()) / 2.0, c.arg() - (-b).arg(), 0.0)
    } else {
        // Diagonal: pin δ = 0. (Normally routed to the Walsh path first.)
        ((a.arg() + d.arg()) / 2.0, d.arg() - a.arg(), 0.0)
    };
    // Rebuild e^{iα}·RZ(β)·RY(γ)·RZ(δ) and require an entrywise match.
    let (cg, sg) = ((gamma / 2.0).cos(), (gamma / 2.0).sin());
    let ea = cis(alpha);
    let rebuilt = [
        ea * cis(-(beta + delta) / 2.0) * cg,
        ea * cis(-(beta - delta) / 2.0) * -sg,
        ea * cis((beta - delta) / 2.0) * sg,
        ea * cis((beta + delta) / 2.0) * cg,
    ];
    for (got, want) in rebuilt.iter().zip(mc) {
        if (got - want).norm_sqr() > 1e-18 {
            return None;
        }
    }
    Some((alpha, beta, gamma, delta))
}

/// A sparse state behind a lazily-flushed Clifford frame. See module docs.
pub struct CliffordFramedState<S: Scalar> {
    num_qubits: usize,
    /// Frame-activity flag readable without borrowing `core`, so
    /// flush-on-read APIs can short-circuit while an outer shared borrow
    /// (e.g. an `amplitude` call inside `for_each_nonzero`) is live.
    active: Cell<bool>,
    core: RefCell<Core<S>>,
}

impl<S: Scalar> CliffordFramedState<S> {
    /// `|0…0⟩` behind an identity frame. The algebra must be a commutative
    /// division algebra containing `i` (recognition and rotation
    /// coefficients live in ℂ): `C64` and `CComplex` qualify; ℝ,
    /// split-complex and the wider Cayley–Dickson algebras are rejected.
    pub fn new(num_qubits: usize) -> Result<Self> {
        if !(S::COMMUTATIVE && S::DIVISION && S::DIM <= 2)
            || S::try_from_c64(c64(0.0, 1.0)).is_none()
        {
            return Err(Error::UnsupportedForAlgebra {
                gate: "clifford frame".into(),
                algebra: S::algebra_name(),
            });
        }
        let state = SparseState::new(num_qubits)?;
        let peak = state.memory_bytes();
        Ok(CliffordFramedState {
            num_qubits,
            active: Cell::new(false),
            core: RefCell::new(Core {
                state,
                tableau: CliffordTableau::identity(num_qubits),
                log: Vec::new(),
                stats: CliffordFrameStats::default(),
                peak_inner_memory: peak,
                peak_stored_support: 1,
            }),
        })
    }

    /// Execution counters so far.
    pub fn stats(&self) -> CliffordFrameStats {
        self.core.borrow().stats
    }

    /// Clifford gates currently held in the frame (the replay-log length).
    pub fn frame_gates(&self) -> usize {
        self.core.borrow().log.len()
    }

    /// Stored-state support, read **without** flushing — the
    /// representation's cost, not the physical support.
    pub fn stored_nonzero_count(&self) -> usize {
        self.core.borrow().state.nonzero_count()
    }

    /// Peak stored support ever reached (flush replays included).
    pub fn peak_stored_support(&self) -> usize {
        self.core.borrow().peak_stored_support
    }

    /// Peak memory the stored state ever reached.
    pub fn peak_inner_memory(&self) -> usize {
        self.core.borrow().peak_inner_memory
    }

    /// The tableau image `C† P C` of a signed Pauli string under the
    /// current frame — research introspection into the symplectic side.
    pub fn conjugated(&self, p: PauliString) -> PauliString {
        self.core.borrow().tableau.image(p)
    }

    /// Measure the Hermitian Pauli observable `P` on the **physical**
    /// state, natively through the frame — no flush, the frame survives.
    /// Physically projecting `C|s⟩` with `(I ± P)/2` equals projecting the
    /// stored `|s⟩` with `(I ± C†PC)/2`, and `C` preserves inner products,
    /// so both the outcome distribution and the collapsed state are exact.
    /// Returns `true` for the −1 outcome (for `Z_q`, "qubit q read 1").
    ///
    /// Cost is the rotation envelope — `O(support)`, at most 2× growth —
    /// **per measurement**; long adaptive sequences compound it (see the
    /// module docs). This is the feedback-loop primitive for
    /// measurement-driven computation in the lifted Clifford space
    /// ([`crate::lift`]), where each measurement hits a fresh ancilla and
    /// the compounding never engages.
    pub fn measure_pauli(&mut self, p: PauliString, rng: &mut Prng) -> Result<bool> {
        let mut core = self.core.borrow_mut();
        let img = core.tableau.image(p);
        core.stats.max_axis_weight = core.stats.max_axis_weight.max(img.weight());
        let outcome = core.state.measure_pauli(img.x, img.z, img.negative, rng)?;
        core.stats.native_measurements += 1;
        core.note_peak();
        Ok(outcome)
    }

    /// Replay the frame into the stored state now (physical state
    /// unchanged; frame becomes identity).
    pub fn flush(&mut self) -> Result<()> {
        let mut core = self.core.borrow_mut();
        core.flush(self.num_qubits)?;
        drop(core);
        self.active.set(false);
        Ok(())
    }

    fn flush_if_active(&self) -> Result<()> {
        if !self.active.get() {
            return Ok(());
        }
        let mut core = self.core.borrow_mut();
        core.flush(self.num_qubits)?;
        drop(core);
        self.active.set(false);
        Ok(())
    }

    /// Fallback path: materialize the frame, then apply the gate raw.
    fn flush_and_raw(
        &mut self,
        matrix: Option<&GateMatrix<S>>,
        entries: Option<&[S]>,
        qubits: &[usize],
    ) -> Result<()> {
        let mut core = self.core.borrow_mut();
        core.flush(self.num_qubits)?;
        match (matrix, entries) {
            (Some(m), _) => core.state.apply(m, qubits)?,
            (_, Some(d)) => core.state.apply_diagonal(d, qubits)?,
            _ => unreachable!("fallback needs a kernel"),
        }
        core.stats.raw_fallbacks += 1;
        core.note_peak();
        drop(core);
        self.active.set(false);
        Ok(())
    }

    fn absorb(
        &mut self,
        kernel: LoggedKernel<S>,
        qubits: &[usize],
        images: Vec<(PauliString, PauliString)>,
    ) {
        let mut core = self.core.borrow_mut();
        // Map the gate's generator images through the *old* tableau before
        // writing any entry (multiple qubits read it).
        let staged: Vec<(usize, PauliString, PauliString)> = qubits
            .iter()
            .zip(&images)
            .map(|(&q, &(gx, gz))| (q, core.tableau.image(gx), core.tableau.image(gz)))
            .collect();
        for (q, xi, zi) in staged {
            core.tableau.x_images[q] = xi;
            core.tableau.z_images[q] = zi;
        }
        core.log.push(LoggedGate {
            kernel,
            qubits: qubits.to_vec(),
        });
        core.stats.absorbed_clifford += 1;
        drop(core);
        self.active.set(true);
    }

    /// Walsh–Hadamard: split a unit diagonal into Z-string rotations and a
    /// global phase, each conjugated through the tableau. Returns `false`
    /// (untouched state) when an entry is not unimodular.
    fn try_walsh(&mut self, diag: &[C64], qubits: &[usize]) -> Result<bool> {
        let d = diag.len();
        let mut phi = Vec::with_capacity(d);
        for e in diag {
            if (e.norm_sqr() - 1.0).abs() > 1e-9 {
                return Ok(false);
            }
            phi.push(e.arg());
        }
        // In-place fast WHT: slot T ends as Σ_b (−1)^{|b∧T|} φ(b).
        let mut len = 1;
        while len < d {
            for start in (0..d).step_by(2 * len) {
                for i in start..start + len {
                    let (a, b) = (phi[i], phi[i + len]);
                    phi[i] = a + b;
                    phi[i + len] = a - b;
                }
            }
            len <<= 1;
        }
        let scale = 1.0 / d as f64;
        let mut core = self.core.borrow_mut();
        core.global_phase(cis(phi[0] * scale))?;
        for (t, &f) in phi.iter().enumerate().skip(1) {
            let ct = f * scale;
            if ct.abs() < 1e-15 {
                continue;
            }
            let p = PauliString {
                x: 0,
                z: to_global(t, qubits),
                negative: false,
            };
            core.rotate(-2.0 * ct, p, c64(1.0, 0.0))?;
            core.stats.diagonal_rotations += 1;
        }
        core.note_peak();
        Ok(true)
    }
}

impl<S: Scalar> Backend<S> for CliffordFramedState<S> {
    fn name(&self) -> &str {
        "clifford-framed"
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn apply(&mut self, matrix: &GateMatrix<S>, qubits: &[usize]) -> Result<()> {
        super::validate_apply(self.num_qubits, matrix, qubits)?;
        let k = qubits.len();
        if k <= CLIFFORD_RECOGNITION_MAX.max(CLIFFORD_DIAGONAL_MAX)
            && matrix.unitarity_deviation() <= UNITARY_TOL
        {
            let mc = matrix_to_c64(matrix);
            if k <= CLIFFORD_RECOGNITION_MAX {
                if let Some(images) = recognize_clifford(&mc, k, qubits) {
                    self.absorb(LoggedKernel::Matrix(matrix.clone()), qubits, images);
                    return Ok(());
                }
                if let Some((theta, phase, px, pz)) = recognize_axis(&mc, k) {
                    let p = PauliString {
                        x: to_global(px, qubits),
                        z: to_global(pz, qubits),
                        negative: false,
                    };
                    let mut core = self.core.borrow_mut();
                    core.rotate(theta, p, phase)?;
                    core.stats.axis_rotations += 1;
                    return Ok(());
                }
            }
            if let Some(diag) = extract_diagonal(&mc, 1usize << k) {
                if self.try_walsh(&diag, qubits)? {
                    return Ok(());
                }
            }
            if k == 1 {
                if let Some((alpha, beta, gamma, delta)) = recognize_zyz(&mc) {
                    let q = qubits[0];
                    let zq = PauliString {
                        x: 0,
                        z: 1u64 << q,
                        negative: false,
                    };
                    let yq = PauliString {
                        x: 1u64 << q,
                        z: 1u64 << q,
                        negative: false,
                    };
                    let mut core = self.core.borrow_mut();
                    core.rotate(delta, zq, c64(1.0, 0.0))?;
                    core.rotate(gamma, yq, c64(1.0, 0.0))?;
                    core.rotate(beta, zq, cis(alpha))?;
                    core.stats.zyz_decompositions += 1;
                    return Ok(());
                }
            }
        }
        self.flush_and_raw(Some(matrix), None, qubits)
    }

    fn apply_diagonal(&mut self, entries: &[S], qubits: &[usize]) -> Result<()> {
        super::validate_apply_diagonal(self.num_qubits, entries, qubits)?;
        let k = qubits.len();
        let diag: Vec<C64> = entries.iter().map(|&e| scalar_to_c64(e)).collect();
        if k <= CLIFFORD_RECOGNITION_MAX {
            let d = entries.len();
            let mut mc = vec![c64(0.0, 0.0); d * d];
            for (i, &e) in diag.iter().enumerate() {
                mc[i * d + i] = e;
            }
            if let Some(images) = recognize_clifford(&mc, k, qubits) {
                self.absorb(LoggedKernel::Diagonal(entries.to_vec()), qubits, images);
                return Ok(());
            }
        }
        if k <= CLIFFORD_DIAGONAL_MAX && self.try_walsh(&diag, qubits)? {
            return Ok(());
        }
        self.flush_and_raw(None, Some(entries), qubits)
    }

    fn amplitude(&self, index: u64) -> S {
        if self.flush_if_active().is_err() {
            return S::zero();
        }
        self.core.borrow().state.amplitude(index)
    }

    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, S)) {
        if self.flush_if_active().is_err() {
            return;
        }
        self.core.borrow().state.for_each_nonzero(f)
    }

    fn nonzero_count(&self) -> usize {
        if self.flush_if_active().is_err() {
            return 0;
        }
        self.core.borrow().state.nonzero_count()
    }

    fn total_weight(&self) -> f64 {
        if self.flush_if_active().is_err() {
            return f64::NAN;
        }
        self.core.borrow().state.total_weight()
    }

    fn measure(&mut self, qubit: usize, rng: &mut Prng) -> Result<bool> {
        // Computational-basis measurement is the Pauli measurement of
        // Z_q, taken natively through the tableau: the frame survives,
        // and outcomes are seed-identical with every other backend
        // (same draw convention, same distribution — C is unitary).
        if qubit >= self.num_qubits {
            return Err(Error::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits,
            });
        }
        self.measure_pauli(
            PauliString {
                x: 0,
                z: 1u64 << qubit,
                negative: false,
            },
            rng,
        )
    }

    fn sample(&self, shots: u64, rng: &mut Prng) -> Result<HashMap<u64, u64>> {
        self.flush_if_active()?;
        self.core.borrow().state.sample(shots, rng)
    }

    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        // Flush errors cannot surface through this signature; a gate that
        // fails to replay would already have failed at absorption time.
        let _ = self.flush_if_active();
        let mut core = self.core.borrow_mut();
        core.state.project(qubit, outcome, renorm);
        core.note_peak();
    }

    fn reset(&mut self) {
        let mut core = self.core.borrow_mut();
        core.state.reset();
        core.tableau = CliffordTableau::identity(self.num_qubits);
        core.log.clear();
        core.stats = CliffordFrameStats::default();
        core.peak_inner_memory = core.state.memory_bytes();
        core.peak_stored_support = 1;
        drop(core);
        self.active.set(false);
    }

    fn load(&mut self, entries: &[(u64, S)]) -> Result<()> {
        let mut core = self.core.borrow_mut();
        core.tableau = CliffordTableau::identity(self.num_qubits);
        core.log.clear();
        core.state.load(entries)?;
        core.note_peak();
        drop(core);
        self.active.set(false);
        Ok(())
    }

    fn memory_bytes(&self) -> usize {
        let core = self.core.borrow();
        let log_bytes: usize = core
            .log
            .iter()
            .map(|g| {
                let kernel = match &g.kernel {
                    LoggedKernel::Matrix(m) => std::mem::size_of_val(m.data()),
                    LoggedKernel::Diagonal(d) => std::mem::size_of_val(d.as_slice()),
                };
                kernel + g.qubits.len() * std::mem::size_of::<usize>()
            })
            .sum();
        core.state.memory_bytes()
            + log_bytes
            + 2 * self.num_qubits * std::mem::size_of::<PauliString>()
            + std::mem::size_of::<Self>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_pauli_products_track_phases() {
        // X·Z = −i·Y in the Hermitian convention: i^1 X Z with phase i^0·(−1)^0…
        let x = PauliString {
            x: 1,
            z: 0,
            negative: false,
        };
        let z = PauliString {
            x: 0,
            z: 1,
            negative: false,
        };
        let mut acc = RawPauli::from_hermitian(x);
        acc.mul(RawPauli::from_hermitian(z));
        // X·Z = X Z = i^{-1}·(i X Z) = −i·Y: raw phase 0, Hermitian phase 1 ⇒ rel 3.
        assert_eq!(acc.phase, 0);
        assert_eq!((acc.x, acc.z), (1, 1));
        // Z·X = −X·Z = +i·Y.
        let mut zx = RawPauli::from_hermitian(z);
        zx.mul(RawPauli::from_hermitian(x));
        assert_eq!(zx.phase, 2);
    }

    #[test]
    fn tableau_reproduces_hadamard_conjugation() {
        // C = H on qubit 0: τ(X)=Z, τ(Z)=X, and τ(Y) must come out −Y.
        let mut t = CliffordTableau::identity(2);
        t.x_images[0] = PauliString {
            x: 0,
            z: 1,
            negative: false,
        };
        t.z_images[0] = PauliString {
            x: 1,
            z: 0,
            negative: false,
        };
        let y = PauliString {
            x: 1,
            z: 1,
            negative: false,
        };
        assert_eq!(
            t.image(y),
            PauliString {
                x: 1,
                z: 1,
                negative: true
            }
        );
        // Qubit 1 untouched.
        let z1 = PauliString {
            x: 0,
            z: 2,
            negative: false,
        };
        assert_eq!(t.image(z1), z1);
    }

    #[test]
    fn decompose_recognizes_known_matrices() {
        // H = (X + Z)/√2.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let h = [c64(s, 0.0), c64(s, 0.0), c64(s, 0.0), c64(-s, 0.0)];
        let dec = pauli_decompose(&h, 1, 1e-12);
        assert_eq!(dec.len(), 2);
        for (px, pz, c) in dec {
            assert!((px, pz) == (1, 0) || (px, pz) == (0, 1));
            assert!((c - c64(s, 0.0)).norm() < 1e-12);
        }
        // rx(θ) has support {I, X} with the locked phases.
        let th: f64 = 0.83;
        let (cc, ss) = ((th / 2.0).cos(), (th / 2.0).sin());
        let rx = [c64(cc, 0.0), c64(0.0, -ss), c64(0.0, -ss), c64(cc, 0.0)];
        let (theta, phase, px, pz) = recognize_axis(&rx, 1).unwrap();
        assert!((theta - th).abs() < 1e-12);
        assert!((phase - c64(1.0, 0.0)).norm() < 1e-12);
        assert_eq!((px, pz), (1, 0));
    }

    #[test]
    fn clifford_recognition_accepts_h_rejects_t() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let h = [c64(s, 0.0), c64(s, 0.0), c64(s, 0.0), c64(-s, 0.0)];
        let images = recognize_clifford(&h, 1, &[3]).unwrap();
        // H swaps X and Z on the gate's qubit (global qubit 3 ⇒ mask 8).
        assert_eq!(
            images[0].0,
            PauliString {
                x: 0,
                z: 8,
                negative: false
            }
        );
        assert_eq!(
            images[0].1,
            PauliString {
                x: 8,
                z: 0,
                negative: false
            }
        );
        let t = [
            c64(1.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            cis(std::f64::consts::FRAC_PI_4),
        ];
        assert!(recognize_clifford(&t, 1, &[0]).is_none());
    }

    #[test]
    fn zyz_covers_generic_and_antidiagonal() {
        for mc in [
            // Generic: a scrambled unitary (u-gate values).
            {
                let (t, p, l) = (0.7f64, 1.1f64, -0.4f64);
                let (ct, st) = ((t / 2.0).cos(), (t / 2.0).sin());
                [c64(ct, 0.0), -cis(l) * st, cis(p) * st, cis(p + l) * ct]
            },
            // Anti-diagonal with phases.
            [c64(0.0, 0.0), cis(0.9), cis(2.1), c64(0.0, 0.0)],
        ] {
            let (alpha, beta, gamma, delta) = recognize_zyz(&mc).unwrap();
            let (cg, sg) = ((gamma / 2.0).cos(), (gamma / 2.0).sin());
            let ea = cis(alpha);
            let rebuilt = [
                ea * cis(-(beta + delta) / 2.0) * cg,
                ea * cis(-(beta - delta) / 2.0) * -sg,
                ea * cis((beta - delta) / 2.0) * sg,
                ea * cis((beta + delta) / 2.0) * cg,
            ];
            for (got, want) in rebuilt.iter().zip(&mc) {
                assert!((got - want).norm() < 1e-9);
            }
        }
    }
}
