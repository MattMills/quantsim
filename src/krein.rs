//! Krein registers: a qubit register whose inner product is **indefinite**,
//! and the twistor space of Minkowski space as the smallest one.
//!
//! Every backend in this crate keeps the positive-definite inner product
//! `⟨φ|ψ⟩ = Σ conj(φᵢ)ψᵢ`, and "unitary" means unitary for it. A theory
//! with more than one time direction, or a conformal theory written on
//! twistors, needs instead a fixed Hermitian involution `η` and the
//! pairing
//!
//! ```text
//! ⟨φ, ψ⟩_η = ⟨φ| η |ψ⟩,        η = η†,  η² = 1,
//! ```
//!
//! which is indefinite whenever `η ≠ ±1`. The evolutions that respect it
//! are the **pseudo-unitaries** `M†ηM = η` — the group `U(p, q)`, where
//! `(p, q)` counts `η`'s `±1` eigenvalues. This is a Krein space, and it
//! is the setting of every "ghost" problem: vectors of negative norm, and
//! a non-compact symmetry group under which the ordinary (path) norm is
//! unbounded while the `η`-norm is conserved exactly.
//!
//! ## What is here
//!
//! * [`KreinMetric`] — `η` as a Pauli string on chosen qubits. It costs
//!   nothing to store and is evaluated on any [`Backend`] through
//!   [`pauli_expectation`]. Any non-trivial Pauli string has eigenvalues
//!   `±1` in equal number, so these metrics give exactly the **split**
//!   signatures `(2^{n−1}, 2^{n−1})` — which is the case that matters
//!   (twistor space is `(2, 2)`), and the boundary is stated rather than
//!   hidden: unequal signatures need a metric this type cannot express.
//! * [`KreinRegister`] — any `Backend<C64>` plus a metric. Gates are
//!   checked for pseudo-unitarity **against the metric restricted to
//!   their qubits** (`η = η_Q ⊗ η_rest`, so `M†ηM = η` iff
//!   `M†η_QM = η_Q`) and refused by name otherwise. The ledger
//!   ([`KreinLedger`]) reports the path weight `Σ|ψᵢ|²`, the `η`-norm,
//!   and the weights of the positive and negative ("ghost") sectors,
//!   `‖Π_±ψ‖² = ⟨ψ|(1 ± η)/2|ψ⟩`. Nothing is clamped: a negative
//!   `η`-norm is reported as one.
//! * [`KreinMetric::boost`] — `exp(θQ) = cosh θ + sinh θ·Q` for a Pauli
//!   string `Q` that *anticommutes* with `η`: pseudo-unitary, not
//!   unitary, and the elementary ghost pump — matched positive- and
//!   negative-sector weight created at zero `η`-cost.
//!
//! ## Twistor space and the infinity twistor
//!
//! Twistors of Minkowski space are `Z = (ω^A, π_{A'}) ∈ ℂ⁴` with the
//! Hermitian form `Φ(Z) = ω^A π̄_A + π_{A'} ω̄^{A'}`, the matrix
//! `[[0, 1], [1, 0]]` in `(ω, π)` blocks (the conventions of the `twisty`
//! reference library). On two qubits — local index `α = s + 2c`, with
//! the **chirality** qubit `c` choosing `ω` or `π` and the spinor qubit
//! `s` the component — that is `Φ = X` on the chirality qubit:
//! [`KreinMetric::twistor`]. Signature `(2, 2)`; the pseudo-unitary
//! group is `U(2, 2)`, and its `SU(2, 2)` is the conformal group's spin
//! cover `Spin(4, 2)` ([`su22_basis`], measured 15-dimensional).
//!
//! Which spacetime the conformal group is cut down to is decided by one
//! more object, the **infinity twistor** `I_{αβ}`, a skew bilinear form;
//! the spacetime symmetry group is the subgroup preserving it,
//! `MᵀIM = I`. [`infinity_twistor`] builds
//! `I = [[λ ε, 0], [0, ε]]`, and [`stabilizer`] solves for its algebra.
//! Measured (`tests/krein.rs`), with the Killing form's inertia naming the
//! real form:
//!
//! | `λ`   | rank of `I` | stabilizer | Killing `(+, −, 0)` | spacetime |
//! |-------|-------------|------------|---------------------|-----------|
//! | `0`   | 2           | 10-dim     | `(3, 3, 4)`         | Minkowski: Poincaré, the 4 null directions are the translations |
//! | `< 0` | 4           | 10-dim     | `(6, 4, 0)`         | AdS₄: `so(3, 2) = sp(4, ℝ)` |
//! | `> 0` | 4           | 10-dim     | `(4, 6, 0)`         | dS₄: `so(4, 1)` |
//! | `i`   | 4           | 6-dim      | `(3, 3, 0)`         | none: `I` breaks the reality of `Φ`, only Lorentz survives |
//!
//! So the rank of a single twistor decides whether a second time
//! direction is absent (flat space, rank 2) or present and wrapped into
//! the curvature scale (AdS₄ ⊂ ℝ^{3,2}, rank 4), and its sign decides
//! which of the two times it is. The stabilizer exponentiates
//! ([`mat_exp`]) into two-qubit gates that the Krein register accepts, so
//! AdS₄ isometries run on a register while ordinary unitarity refuses
//! them.
//!
//! ## Boundaries
//!
//! * A Krein register is a *state space*, not a probability theory. The
//!   `η`-norm is conserved by every accepted gate, but it is indefinite,
//!   so it is not a Born rule; the sector weights are what a projection
//!   onto the positive sector (Gupta–Bleuler style) would keep. Which of
//!   these, if any, is a probability is the open question the type is
//!   built to pose.
//! * The underlying backend still stores amplitudes in the ordinary
//!   basis, and non-compact gates grow the path weight without bound:
//!   precision, not memory, is what a long ghost-pumping run spends.

use crate::backend::{pauli_expectation, Backend};
use crate::braided::mat_exp;
use crate::error::{Error, Result};
use crate::gates::Pauli;
use crate::hermitian2::FormSignature;
use crate::math::GateMatrix;
use crate::scalar::C64;

/// Default tolerance for pseudo-unitarity checks.
pub const KREIN_TOL: f64 = 1e-9;

fn pauli_entry(p: Pauli, r: usize, c: usize) -> C64 {
    let z = C64::new(0.0, 0.0);
    let one = C64::new(1.0, 0.0);
    match (p, r, c) {
        (Pauli::I, r, c) if r == c => one,
        (Pauli::X, r, c) if r != c => one,
        (Pauli::Y, 0, 1) => C64::new(0.0, -1.0),
        (Pauli::Y, 1, 0) => C64::new(0.0, 1.0),
        (Pauli::Z, 0, 0) => one,
        (Pauli::Z, 1, 1) => -one,
        _ => z,
    }
}

/// Whether two single-qubit Paulis anticommute.
fn anticommute(a: Pauli, b: Pauli) -> bool {
    a != Pauli::I && b != Pauli::I && a != b
}

/// An indefinite metric `η`: a tensor product of Paulis on chosen qubits
/// (identity elsewhere). See the module docs.
#[derive(Debug, Clone, PartialEq)]
pub struct KreinMetric {
    ops: Vec<(usize, Pauli)>,
}

impl KreinMetric {
    /// `η = ⊗ P_q` over the listed qubits. Identity entries are dropped;
    /// an empty list is the ordinary positive-definite metric. Duplicate
    /// qubits are refused.
    pub fn new(ops: &[(usize, Pauli)]) -> Result<Self> {
        let mut kept: Vec<(usize, Pauli)> = Vec::new();
        for &(q, p) in ops {
            if kept.iter().any(|&(k, _)| k == q) {
                return Err(Error::InvalidState(format!(
                    "Krein metric: qubit {q} listed twice"
                )));
            }
            if p != Pauli::I {
                kept.push((q, p));
            }
        }
        Ok(KreinMetric { ops: kept })
    }

    /// The twistor form `Φ`: `X` on the chirality qubit. On a register
    /// whose two qubits are `[spinor, chirality]`, local index
    /// `s + 2c` is the twistor component `(ω⁰, ω¹, π₀', π₁')`.
    pub fn twistor(chirality: usize) -> Self {
        KreinMetric {
            ops: vec![(chirality, Pauli::X)],
        }
    }

    /// The Pauli string.
    pub fn ops(&self) -> &[(usize, Pauli)] {
        &self.ops
    }

    /// Signature `(p, q)` of `η` on an `n`-qubit register: `(2ⁿ, 0)` for
    /// the trivial metric, `(2^{n−1}, 2^{n−1})` otherwise.
    pub fn signature(&self, num_qubits: usize) -> (u64, u64) {
        let d = 1u64 << num_qubits;
        if self.ops.is_empty() {
            (d, 0)
        } else {
            (d / 2, d / 2)
        }
    }

    /// `η` restricted to `qubits`, as a matrix over their little-endian
    /// local index (bit `b` ↔ `qubits[b]`).
    pub fn restricted(&self, qubits: &[usize]) -> Result<GateMatrix<C64>> {
        let k = qubits.len();
        let d = 1usize << k;
        let paulis: Vec<Pauli> = qubits
            .iter()
            .map(|q| {
                self.ops
                    .iter()
                    .find(|&&(m, _)| m == *q)
                    .map_or(Pauli::I, |&(_, p)| p)
            })
            .collect();
        let mut m = GateMatrix::<C64>::zeros(d)?;
        for r in 0..d {
            for c in 0..d {
                let mut v = C64::new(1.0, 0.0);
                for (b, &p) in paulis.iter().enumerate() {
                    v *= pauli_entry(p, (r >> b) & 1, (c >> b) & 1);
                }
                m.set(r, c, v);
            }
        }
        Ok(m)
    }

    /// `max |(M†η_QM − η_Q)_{rc}|` for a gate `m` on `qubits`.
    pub fn pseudo_unitarity_deviation(&self, m: &GateMatrix<C64>, qubits: &[usize]) -> Result<f64> {
        if m.dim() != 1usize << qubits.len() {
            return Err(Error::InvalidState(format!(
                "Krein check: a {}-dim gate on {} qubits",
                m.dim(),
                qubits.len()
            )));
        }
        let eta = self.restricted(qubits)?;
        let lhs = m.dagger().matmul(&eta).matmul(m);
        let mut dev = 0.0f64;
        for r in 0..m.dim() {
            for c in 0..m.dim() {
                dev = dev.max((lhs.get(r, c) - eta.get(r, c)).norm());
            }
        }
        Ok(dev)
    }

    /// `exp(θQ) = cosh θ·1 + sinh θ·Q` for the Pauli string `generator`
    /// (on its own qubits, in the listed order). `Q` must anticommute with
    /// the metric restricted to those qubits — then the result is
    /// pseudo-unitary and not unitary; otherwise it is refused.
    pub fn boost(&self, generator: &[(usize, Pauli)], theta: f64) -> Result<GateMatrix<C64>> {
        let mut flips = 0usize;
        for &(q, p) in generator {
            if let Some(&(_, e)) = self.ops.iter().find(|&&(m, _)| m == q) {
                if anticommute(p, e) {
                    flips += 1;
                }
            }
        }
        if flips % 2 == 0 {
            return Err(Error::InvalidState(
                "Krein boost: the generator commutes with the metric, so exp(θQ) \
                 is not pseudo-unitary"
                    .to_string(),
            ));
        }
        let qubits: Vec<usize> = generator.iter().map(|&(q, _)| q).collect();
        let q = KreinMetric::new(generator)?.restricted(&qubits)?;
        let d = q.dim();
        let mut m = GateMatrix::<C64>::zeros(d)?;
        let (c, s) = (theta.cosh(), theta.sinh());
        for r in 0..d {
            for col in 0..d {
                let id = if r == col { c } else { 0.0 };
                m.set(r, col, C64::new(id, 0.0) + q.get(r, col) * s);
            }
        }
        Ok(m)
    }

    /// `⟨φ, ψ⟩_η = ⟨φ|η|ψ⟩` between two states of equal width.
    pub fn pair(&self, phi: &dyn Backend<C64>, psi: &dyn Backend<C64>) -> Result<C64> {
        if phi.num_qubits() != psi.num_qubits() {
            return Err(Error::InvalidState(
                "Krein pairing: registers of different width".to_string(),
            ));
        }
        let mut acc = C64::new(0.0, 0.0);
        psi.for_each_nonzero(&mut |i, a| {
            // η|i⟩ = phase·|i ^ flips⟩.
            let mut target = i;
            let mut phase = C64::new(1.0, 0.0);
            for &(q, p) in &self.ops {
                let bit = (i >> q) & 1;
                match p {
                    Pauli::I => {}
                    Pauli::X => target ^= 1 << q,
                    Pauli::Y => {
                        target ^= 1 << q;
                        phase *= pauli_entry(Pauli::Y, 1 - bit as usize, bit as usize);
                    }
                    Pauli::Z => {
                        if bit == 1 {
                            phase = -phase;
                        }
                    }
                }
            }
            acc += phi.amplitude(target).conj() * phase * a;
        });
        Ok(acc)
    }
}

/// The weights a Krein register carries. See [`KreinRegister::ledger`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KreinLedger {
    /// `Σ|ψᵢ|²`: every amplitude counted positively.
    pub path: f64,
    /// `⟨ψ|η|ψ⟩`: the conserved, indefinite norm.
    pub eta: f64,
    /// `‖Π₊ψ‖²`, the positive sector's weight, `(path + eta)/2`.
    pub positive: f64,
    /// `‖Π₋ψ‖²`, the negative ("ghost") sector's weight, `(path − eta)/2`.
    pub negative: f64,
}

/// A register with an indefinite metric: any `Backend<C64>` plus a
/// [`KreinMetric`]. See the module docs.
pub struct KreinRegister {
    state: Box<dyn Backend<C64>>,
    metric: KreinMetric,
    tol: f64,
}

impl KreinRegister {
    /// Wrap `state` with `metric`. The metric's qubits must exist.
    pub fn new(state: Box<dyn Backend<C64>>, metric: KreinMetric) -> Result<Self> {
        let n = state.num_qubits();
        if let Some(&(q, _)) = metric.ops.iter().find(|&&(q, _)| q >= n) {
            return Err(Error::InvalidState(format!(
                "Krein metric names qubit {q} on a {n}-qubit register"
            )));
        }
        Ok(KreinRegister {
            state,
            metric,
            tol: KREIN_TOL,
        })
    }

    /// The pseudo-unitarity tolerance (default [`KREIN_TOL`]).
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// The metric.
    pub fn metric(&self) -> &KreinMetric {
        &self.metric
    }

    /// The underlying state.
    pub fn state(&self) -> &dyn Backend<C64> {
        self.state.as_ref()
    }

    /// Apply `m` on `qubits` if it preserves the metric; refuse by name
    /// (with the measured deviation) otherwise. Ordinary unitaries that
    /// do not touch the metric's qubits, or commute with it, pass.
    pub fn apply(&mut self, m: &GateMatrix<C64>, qubits: &[usize]) -> Result<()> {
        let dev = self.metric.pseudo_unitarity_deviation(m, qubits)?;
        if dev > self.tol {
            return Err(Error::InvalidState(format!(
                "Krein register: gate on {qubits:?} is not pseudo-unitary for \
                 metric {:?} (deviation {dev:.3e})",
                self.metric.ops
            )));
        }
        self.state.apply(m, qubits)
    }

    /// The register's weights. The `η`-norm is conserved by every gate
    /// [`KreinRegister::apply`] accepts; the path weight is not.
    pub fn ledger(&self) -> Result<KreinLedger> {
        let path = self.state.total_abs_sqr();
        let eta = if self.metric.ops.is_empty() {
            path
        } else {
            pauli_expectation(self.state.as_ref(), &self.metric.ops)?.re
        };
        Ok(KreinLedger {
            path,
            eta,
            positive: 0.5 * (path + eta),
            negative: 0.5 * (path - eta),
        })
    }
}

// ── twistor space ────────────────────────────────────────────────────

fn mat4(entries: [[C64; 4]; 4]) -> GateMatrix<C64> {
    let data: Vec<C64> = entries.iter().flatten().copied().collect();
    GateMatrix::from_vec(4, data).expect("4 is a power of two")
}

/// The twistor Hermitian form `Φ = [[0, 1], [1, 0]]` in `(ω, π)` blocks,
/// equal to [`KreinMetric::twistor`]`(1).restricted(&[0, 1])`.
pub fn twistor_form() -> GateMatrix<C64> {
    let (o, l) = (C64::new(0.0, 0.0), C64::new(1.0, 0.0));
    mat4([[o, o, l, o], [o, o, o, l], [l, o, o, o], [o, l, o, o]])
}

/// The infinity twistor `I = [[λ ε, 0], [0, ε]]`, `ε = [[0, 1], [−1, 0]]`:
/// a skew bilinear form on twistors. Rank 2 at `λ = 0` (flat space),
/// rank 4 otherwise.
pub fn infinity_twistor(lambda: C64) -> GateMatrix<C64> {
    let (o, l) = (C64::new(0.0, 0.0), C64::new(1.0, 0.0));
    mat4([
        [o, lambda, o, o],
        [-lambda, o, o, o],
        [o, o, o, l],
        [o, o, -l, o],
    ])
}

/// Real basis of the solution space of a set of real-linear constraints
/// on 4×4 complex matrices. Each constraint maps `X` to a matrix that
/// must vanish.
type Constraint<'a> = &'a dyn Fn(&GateMatrix<C64>) -> GateMatrix<C64>;

fn solve_4x4(constraints: &[Constraint]) -> Vec<GateMatrix<C64>> {
    // Unknowns: 16 real parts then 16 imaginary parts of X (row-major).
    let unit = |k: usize| -> GateMatrix<C64> {
        let mut m = GateMatrix::<C64>::zeros(4).expect("4 is a power of two");
        let v = if k < 16 {
            C64::new(1.0, 0.0)
        } else {
            C64::new(0.0, 1.0)
        };
        m.set((k % 16) / 4, k % 4, v);
        m
    };
    let images: Vec<Vec<f64>> = (0..32)
        .map(|k| {
            let x = unit(k);
            let mut col = Vec::new();
            for c in constraints {
                let y = c(&x);
                col.extend(y.data().iter().map(|z| z.re));
                col.extend(y.data().iter().map(|z| z.im));
            }
            col
        })
        .collect();
    let mut rows: Vec<Vec<f64>> = (0..images[0].len())
        .map(|r| images.iter().map(|col| col[r]).collect())
        .collect();
    real_null_space(&mut rows, 32, 1e-10)
        .into_iter()
        .map(|v| {
            let data: Vec<C64> = (0..16).map(|k| C64::new(v[k], v[16 + k])).collect();
            GateMatrix::from_vec(4, data).expect("4 is a power of two")
        })
        .collect()
}

/// Null space of a real matrix (rows given), by Gauss–Jordan elimination
/// with partial pivoting. Returns one vector per free column.
fn real_null_space(rows: &mut [Vec<f64>], cols: usize, tol: f64) -> Vec<Vec<f64>> {
    let scale = rows
        .iter()
        .flatten()
        .fold(0.0f64, |m, x| m.max(x.abs()))
        .max(1.0);
    let mut pivots: Vec<usize> = Vec::new();
    let mut r = 0usize;
    for c in 0..cols {
        if r == rows.len() {
            break;
        }
        let (best, val) = (r..rows.len())
            .map(|i| (i, rows[i][c].abs()))
            .fold((r, 0.0), |acc, x| if x.1 > acc.1 { x } else { acc });
        if val <= tol * scale {
            continue;
        }
        rows.swap(r, best);
        let p = rows[r][c];
        for x in rows[r].iter_mut() {
            *x /= p;
        }
        let pivot_row = rows[r].clone();
        for (i, row) in rows.iter_mut().enumerate() {
            if i != r && row[c] != 0.0 {
                let f = row[c];
                for (x, &y) in row.iter_mut().zip(&pivot_row) {
                    *x -= f * y;
                }
            }
        }
        pivots.push(c);
        r += 1;
    }
    let mut basis = Vec::new();
    for free in (0..cols).filter(|c| !pivots.contains(c)) {
        let mut v = vec![0.0; cols];
        v[free] = 1.0;
        for (i, &pc) in pivots.iter().enumerate() {
            v[pc] = -rows[i][free];
        }
        basis.push(v);
    }
    basis
}

/// A real basis of `su(2, 2)`: traceless `X` with `X†Φ + ΦX = 0`.
pub fn su22_basis() -> Vec<GateMatrix<C64>> {
    let phi = twistor_form();
    let preserve = |x: &GateMatrix<C64>| {
        let a = x.dagger().matmul(&phi);
        let b = phi.matmul(x);
        sum(&a, &b)
    };
    let trace = |x: &GateMatrix<C64>| {
        let mut m = GateMatrix::<C64>::zeros(4).expect("4 is a power of two");
        m.set(0, 0, (0..4).map(|k| x.get(k, k)).sum());
        m
    };
    solve_4x4(&[&preserve, &trace])
}

/// The subalgebra of `su(2, 2)` preserving the skew form `i`:
/// `X ∈ su(2, 2)` with `XᵀI + IX = 0`. See the module docs' table.
pub fn stabilizer(i: &GateMatrix<C64>) -> Vec<GateMatrix<C64>> {
    let phi = twistor_form();
    let preserve = |x: &GateMatrix<C64>| sum(&x.dagger().matmul(&phi), &phi.matmul(x));
    let trace = |x: &GateMatrix<C64>| {
        let mut m = GateMatrix::<C64>::zeros(4).expect("4 is a power of two");
        m.set(0, 0, (0..4).map(|k| x.get(k, k)).sum());
        m
    };
    let skew = |x: &GateMatrix<C64>| sum(&transpose(x).matmul(i), &i.matmul(x));
    solve_4x4(&[&preserve, &trace, &skew])
}

fn sum(a: &GateMatrix<C64>, b: &GateMatrix<C64>) -> GateMatrix<C64> {
    let data: Vec<C64> = a.data().iter().zip(b.data()).map(|(x, y)| x + y).collect();
    GateMatrix::from_vec(a.dim(), data).expect("same dimension")
}

/// Plain transpose (not conjugated).
pub fn transpose(a: &GateMatrix<C64>) -> GateMatrix<C64> {
    let d = a.dim();
    let mut t = GateMatrix::<C64>::zeros(d).expect("square");
    for r in 0..d {
        for c in 0..d {
            t.set(c, r, a.get(r, c));
        }
    }
    t
}

/// `Re tr(A†B)`, the real inner product on matrices.
fn real_inner(a: &GateMatrix<C64>, b: &GateMatrix<C64>) -> f64 {
    a.data()
        .iter()
        .zip(b.data())
        .map(|(x, y)| (x.conj() * y).re)
        .sum()
}

/// Inertia of the Killing form `K(X, Y) = tr(ad X · ad Y)` of the real Lie
/// algebra spanned by `basis` (which must close under commutators). Its
/// signature names the real form: `(n_noncompact, n_compact, n_nilpotent)`
/// up to the convention that compact directions are negative.
pub fn killing_signature(basis: &[GateMatrix<C64>]) -> Result<FormSignature> {
    let n = basis.len();
    if n == 0 {
        return Ok(FormSignature {
            positive: 0,
            negative: 0,
            null: 0,
        });
    }
    // Coordinates in the basis via its Gram matrix.
    let gram: Vec<f64> = (0..n * n)
        .map(|k| real_inner(&basis[k / n], &basis[k % n]))
        .collect();
    let coords = |m: &GateMatrix<C64>| -> Result<Vec<f64>> {
        let rhs: Vec<f64> = basis.iter().map(|b| real_inner(b, m)).collect();
        let c = solve_dense(n, &gram, &rhs)?;
        // Closure check: the reconstruction must be exact.
        let mut back = GateMatrix::<C64>::zeros(m.dim())?;
        for (b, &ck) in basis.iter().zip(&c) {
            for r in 0..m.dim() {
                for col in 0..m.dim() {
                    back.set(r, col, back.get(r, col) + b.get(r, col) * ck);
                }
            }
        }
        if !back.approx_eq(m, 1e-8) {
            return Err(Error::InvalidState(
                "killing_signature: the basis does not close under commutators".to_string(),
            ));
        }
        Ok(c)
    };
    // ad_i as an n×n real matrix: column j = coords([B_i, B_j]).
    let mut ads: Vec<Vec<f64>> = Vec::with_capacity(n);
    for x in basis {
        let mut ad = vec![0.0; n * n];
        for (j, y) in basis.iter().enumerate() {
            let xy = x.matmul(y);
            let yx = y.matmul(x);
            let data: Vec<C64> = xy
                .data()
                .iter()
                .zip(yx.data())
                .map(|(a, b)| a - b)
                .collect();
            let comm = GateMatrix::from_vec(x.dim(), data)?;
            for (i, c) in coords(&comm)?.into_iter().enumerate() {
                ad[i * n + j] = c;
            }
        }
        ads.push(ad);
    }
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let (a, b) = (&ads[i], &ads[j]);
            let mut t = 0.0;
            for p in 0..n {
                for q in 0..n {
                    t += a[p * n + q] * b[q * n + p];
                }
            }
            k[i * n + j] = t;
        }
    }
    let (values, _) = crate::recursive::sym_eigen(n, &k);
    let top = values
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(1e-300);
    let cut = 1e-8 * top;
    Ok(FormSignature {
        positive: values.iter().filter(|&&v| v > cut).count(),
        negative: values.iter().filter(|&&v| v < -cut).count(),
        null: values.iter().filter(|&&v| v.abs() <= cut).count(),
    })
}

/// Solve the `n × n` system `A x = b` by Gaussian elimination.
fn solve_dense(n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|r| {
            let mut row = a[r * n..(r + 1) * n].to_vec();
            row.push(b[r]);
            row
        })
        .collect();
    for c in 0..n {
        let best = (c..n)
            .max_by(|&x, &y| m[x][c].abs().total_cmp(&m[y][c].abs()))
            .expect("non-empty range");
        if m[best][c].abs() < 1e-14 {
            return Err(Error::InvalidState(
                "killing_signature: degenerate basis".to_string(),
            ));
        }
        m.swap(c, best);
        let pivot = m[c].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r != c {
                let f = row[c] / pivot[c];
                for (x, &y) in row.iter_mut().zip(&pivot).skip(c) {
                    *x -= f * y;
                }
            }
        }
    }
    Ok((0..n).map(|r| m[r][n] / m[r][r]).collect())
}

/// `exp(θX)` for an algebra element `x`: a group element of the
/// corresponding subgroup of `SU(2, 2)`, usable as a two-qubit gate on a
/// [`KreinRegister`] with the [`KreinMetric::twistor`] metric.
pub fn exponentiate(x: &GateMatrix<C64>, theta: f64) -> Result<GateMatrix<C64>> {
    let data: Vec<C64> = x.data().iter().map(|z| z * theta).collect();
    mat_exp(&GateMatrix::from_vec(x.dim(), data)?)
}
