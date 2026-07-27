//! Circuits: ordered gate applications, decoupled from any registry or
//! backend until bound.
//!
//! A [`Circuit`] records operations by *name* (plus parameters and target
//! qubits) or as raw matrices; [`Circuit::bind`] resolves names against a
//! [`GateRegistry`], validates everything, and produces a [`BoundCircuit`]
//! of ready-to-apply matrices that can run on any backend of matching width.
//!
//! Qubit convention (little-endian, Qiskit-style): qubit 0 is the least
//! significant bit of a basis-state index. For multi-qubit gates, `qubits[b]`
//! corresponds to sub-index bit `b` of the gate matrix; standard controlled
//! gates put the control at `qubits[0]`.

use crate::error::{Error, Result};
use crate::math::{GateMatrix, UNITARY_TOL};
use crate::registry::GateRegistry;
use crate::scalar::{Scalar, C64};

/// One recorded operation.
#[derive(Debug, Clone)]
pub enum Op<S: Scalar> {
    /// A registry gate referenced by name.
    Named {
        /// Gate name or alias, resolved at bind time.
        name: String,
        /// Real parameters.
        params: Vec<f64>,
        /// Target qubits (`qubits[b]` ↔ matrix sub-index bit `b`).
        qubits: Vec<usize>,
    },
    /// An explicit unitary, for gates not worth registering.
    Raw {
        /// Display label.
        label: String,
        /// The unitary; validated at bind time.
        matrix: GateMatrix<S>,
        /// Target qubits.
        qubits: Vec<usize>,
    },
    /// A diagonal unitary given by its `2^k` diagonal entries. Stores and
    /// applies in `O(2^k)` instead of `O(4^k)` — the difference between a
    /// 262 KiB and a 4 GiB multi-controlled Z at k = 14. The natural shape
    /// for phase oracles.
    Diagonal {
        /// Display label.
        label: String,
        /// Diagonal entries, indexed by sub-index (little-endian over
        /// `qubits`); each must satisfy `conj(d)·d = 1` (validated at bind).
        entries: Vec<S>,
        /// Target qubits.
        qubits: Vec<usize>,
    },
}

impl<S: Scalar> Op<S> {
    fn qubits(&self) -> &[usize] {
        match self {
            Op::Named { qubits, .. } | Op::Raw { qubits, .. } | Op::Diagonal { qubits, .. } => {
                qubits
            }
        }
    }
    fn qubits_mut(&mut self) -> &mut Vec<usize> {
        match self {
            Op::Named { qubits, .. } | Op::Raw { qubits, .. } | Op::Diagonal { qubits, .. } => {
                qubits
            }
        }
    }
}

/// An ordered list of gate applications on `num_qubits` qubits.
///
/// The scalar parameter `S` (default [`C64`]) only matters for raw-matrix
/// operations; named operations are algebra-independent until bound.
#[derive(Debug, Clone)]
pub struct Circuit<S: Scalar = C64> {
    num_qubits: usize,
    ops: Vec<Op<S>>,
}

impl<S: Scalar> Circuit<S> {
    /// An empty circuit on `num_qubits` qubits.
    pub fn new(num_qubits: usize) -> Self {
        Circuit {
            num_qubits,
            ops: Vec::new(),
        }
    }

    /// Circuit width.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Recorded operations.
    pub fn ops(&self) -> &[Op<S>] {
        &self.ops
    }

    /// Number of operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the circuit has no operations.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Append a named gate. Validation happens at [`Circuit::bind`], so this
    /// is chainable without `Result` noise.
    pub fn gate(
        &mut self,
        name: impl Into<String>,
        params: impl Into<Vec<f64>>,
        qubits: impl Into<Vec<usize>>,
    ) -> &mut Self {
        self.ops.push(Op::Named {
            name: name.into(),
            params: params.into(),
            qubits: qubits.into(),
        });
        self
    }

    /// Append a raw unitary (checked for unitarity at bind time).
    pub fn raw(
        &mut self,
        label: impl Into<String>,
        matrix: GateMatrix<S>,
        qubits: impl Into<Vec<usize>>,
    ) -> &mut Self {
        self.ops.push(Op::Raw {
            label: label.into(),
            matrix,
            qubits: qubits.into(),
        });
        self
    }

    /// Append a diagonal unitary given by its `2^k` diagonal entries
    /// (indexed by little-endian sub-index over `qubits`). Each entry must
    /// satisfy `conj(d)·d = 1`, checked at bind time. Phase oracles and
    /// multi-controlled Z gates should use this rather than [`Circuit::raw`]:
    /// storage and application cost `O(2^k)` instead of `O(4^k)`.
    pub fn diagonal(
        &mut self,
        label: impl Into<String>,
        entries: Vec<S>,
        qubits: impl Into<Vec<usize>>,
    ) -> &mut Self {
        self.ops.push(Op::Diagonal {
            label: label.into(),
            entries,
            qubits: qubits.into(),
        });
        self
    }

    /// Append every operation of `other`, remapping its qubit `q` to
    /// `map[q]`. Useful for embedding library circuits (e.g. a QFT) into a
    /// subset of a larger register.
    ///
    /// # Panics
    /// Panics if `map.len() != other.num_qubits()` (a programmer error);
    /// the mapped qubit values themselves are validated at bind time.
    pub fn append(&mut self, other: &Circuit<S>, map: &[usize]) -> &mut Self {
        assert_eq!(
            map.len(),
            other.num_qubits(),
            "append: map length must equal the appended circuit's width"
        );
        for op in &other.ops {
            let mut op = op.clone();
            for q in op.qubits_mut().iter_mut() {
                *q = map[*q];
            }
            self.ops.push(op);
        }
        self
    }

    /// Resolve names against `registry`, validate every operation, and
    /// produce a runnable [`BoundCircuit`].
    pub fn bind(&self, registry: &GateRegistry<S>) -> Result<BoundCircuit<S>> {
        let mut gates = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            validate_targets(self.num_qubits, op.qubits())?;
            match op {
                Op::Named {
                    name,
                    params,
                    qubits,
                } => {
                    let def = registry.resolve(name)?;
                    if def.arity() != qubits.len() {
                        return Err(Error::ArityMismatch {
                            gate: name.clone(),
                            expected: def.arity(),
                            got: qubits.len(),
                        });
                    }
                    if def.param_count() != params.len() {
                        return Err(Error::ParamCountMismatch {
                            gate: name.clone(),
                            expected: def.param_count(),
                            got: params.len(),
                        });
                    }
                    let matrix = def.matrix(params)?;
                    debug_assert_eq!(matrix.dim(), 1 << qubits.len());
                    gates.push(BoundGate {
                        label: name.clone(),
                        kernel: GateKernel::Matrix(matrix),
                        qubits: qubits.clone(),
                    });
                }
                Op::Raw {
                    label,
                    matrix,
                    qubits,
                } => {
                    let expected = 1usize << qubits.len();
                    if matrix.dim() != expected {
                        return Err(Error::BadDimension {
                            expected,
                            got: matrix.dim(),
                        });
                    }
                    let dev = matrix.unitarity_deviation();
                    if dev > UNITARY_TOL {
                        return Err(Error::NotUnitary {
                            label: label.clone(),
                            deviation: dev,
                        });
                    }
                    gates.push(BoundGate {
                        label: label.clone(),
                        kernel: GateKernel::Matrix(matrix.clone()),
                        qubits: qubits.clone(),
                    });
                }
                Op::Diagonal {
                    label,
                    entries,
                    qubits,
                } => {
                    let expected = 1usize << qubits.len();
                    if entries.len() != expected {
                        return Err(Error::BadDimension {
                            expected,
                            got: entries.len(),
                        });
                    }
                    // Unitarity for a diagonal is entrywise conj(d)·d = 1
                    // (the algebra's own conjugation, not just |d| = 1 —
                    // they differ over e.g. split-complex).
                    let mut dev: f64 = 0.0;
                    for &d in entries {
                        dev = dev.max((d.conj() * d - S::one()).abs_sqr().sqrt());
                    }
                    if dev > UNITARY_TOL {
                        return Err(Error::NotUnitary {
                            label: label.clone(),
                            deviation: dev,
                        });
                    }
                    gates.push(BoundGate {
                        label: label.clone(),
                        kernel: GateKernel::Diagonal(entries.clone()),
                        qubits: qubits.clone(),
                    });
                }
            }
        }
        Ok(BoundCircuit {
            num_qubits: self.num_qubits,
            gates,
        })
    }
}

/// Validate a target-qubit list against a register width: in range, and no
/// duplicates.
pub(crate) fn validate_targets(num_qubits: usize, qubits: &[usize]) -> Result<()> {
    for &q in qubits {
        if q >= num_qubits {
            return Err(Error::QubitOutOfRange {
                qubit: q,
                num_qubits,
            });
        }
    }
    for (i, &a) in qubits.iter().enumerate() {
        if qubits[i + 1..].contains(&a) {
            return Err(Error::DuplicateQubits {
                qubits: qubits.to_vec(),
            });
        }
    }
    Ok(())
}

/// The computational kernel of a bound gate.
#[derive(Debug, Clone)]
pub enum GateKernel<S: Scalar> {
    /// A dense unitary matrix.
    Matrix(GateMatrix<S>),
    /// A diagonal unitary, stored as its `2^k` diagonal entries.
    Diagonal(Vec<S>),
}

/// A resolved gate application: kernel plus targets.
#[derive(Debug, Clone)]
pub struct BoundGate<S: Scalar> {
    /// Display label (gate name, or `name†` for inverses).
    pub label: String,
    /// The unitary kernel.
    pub kernel: GateKernel<S>,
    /// Target qubits.
    pub qubits: Vec<usize>,
}

/// A validated, registry-independent sequence of gate matrices.
#[derive(Debug, Clone)]
pub struct BoundCircuit<S: Scalar> {
    num_qubits: usize,
    gates: Vec<BoundGate<S>>,
}

impl<S: Scalar> BoundCircuit<S> {
    /// Circuit width.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// The resolved gate sequence.
    pub fn gates(&self) -> &[BoundGate<S>] {
        &self.gates
    }

    /// Run every gate, in order, on `backend`.
    pub fn run(&self, backend: &mut dyn crate::backend::Backend<S>) -> Result<()> {
        if backend.num_qubits() != self.num_qubits {
            return Err(Error::WidthMismatch {
                circuit: self.num_qubits,
                backend: backend.num_qubits(),
            });
        }
        for g in &self.gates {
            match &g.kernel {
                GateKernel::Matrix(m) => backend.apply(m, &g.qubits)?,
                GateKernel::Diagonal(d) => backend.apply_diagonal(d, &g.qubits)?,
            }
        }
        Ok(())
    }

    /// The inverse circuit: reversed order, each kernel conjugate-transposed.
    ///
    /// Exact for associative scalars; over non-associative algebras
    /// `U · U†` need not be the identity operationally, which is part of
    /// what those algebras are for.
    pub fn inverse(&self) -> Self {
        let gates = self
            .gates
            .iter()
            .rev()
            .map(|g| BoundGate {
                label: format!("{}†", g.label),
                kernel: match &g.kernel {
                    GateKernel::Matrix(m) => GateKernel::Matrix(m.dagger()),
                    GateKernel::Diagonal(d) => {
                        GateKernel::Diagonal(d.iter().map(|&e| e.conj()).collect())
                    }
                },
                qubits: g.qubits.clone(),
            })
            .collect();
        BoundCircuit {
            num_qubits: self.num_qubits,
            gates,
        }
    }
}

// Convenience constructors for every standard gate. All are thin wrappers
// over [`Circuit::gate`]; validity is checked at bind time.
macro_rules! fixed_gates {
    ($( $(#[$doc:meta])* $fn_name:ident($name:literal, $($q:ident),+ ); )*) => {
        impl<S: Scalar> Circuit<S> {
            $(
                $(#[$doc])*
                pub fn $fn_name(&mut self, $($q: usize),+) -> &mut Self {
                    self.gate($name, Vec::new(), vec![$($q),+])
                }
            )*
        }
    };
}

macro_rules! param_gates {
    ($( $(#[$doc:meta])* $fn_name:ident($name:literal, [$($q:ident),+], [$($p:ident),+]); )*) => {
        impl<S: Scalar> Circuit<S> {
            $(
                $(#[$doc])*
                pub fn $fn_name(&mut self, $($q: usize,)+ $($p: f64),+) -> &mut Self {
                    self.gate($name, vec![$($p),+], vec![$($q),+])
                }
            )*
        }
    };
}

fixed_gates! {
    /// Identity on `q`.
    id("id", q);
    /// Pauli X on `q`.
    x("x", q);
    /// Pauli Y on `q`.
    y("y", q);
    /// Pauli Z on `q`.
    z("z", q);
    /// Hadamard on `q`.
    h("h", q);
    /// Phase gate S on `q`.
    s("s", q);
    /// S† on `q`.
    sdg("sdg", q);
    /// T gate on `q`.
    t("t", q);
    /// T† on `q`.
    tdg("tdg", q);
    /// √X on `q`.
    sx("sx", q);
    /// √X† on `q`.
    sxdg("sxdg", q);
    /// Controlled-X with control `c`, target `t`.
    cx("cx", c, t);
    /// Controlled-Y with control `c`, target `t`.
    cy("cy", c, t);
    /// Controlled-Z on `c`, `t` (symmetric).
    cz("cz", c, t);
    /// Controlled-Hadamard with control `c`, target `t`.
    ch("ch", c, t);
    /// Swap qubits `a` and `b`.
    swap("swap", a, b);
    /// iSwap on `a`, `b`.
    iswap("iswap", a, b);
    /// Toffoli with controls `c1`, `c2`, target `t`.
    ccx("ccx", c1, c2, t);
    /// Doubly controlled Z on `a`, `b`, `c` (symmetric).
    ccz("ccz", a, b, c);
    /// Fredkin: swap `a`, `b` when control `c` is set.
    cswap("cswap", c, a, b);
}

param_gates! {
    /// X rotation `exp(-i θ X / 2)` on `q`.
    rx("rx", [q], [theta]);
    /// Y rotation `exp(-i θ Y / 2)` on `q`.
    ry("ry", [q], [theta]);
    /// Z rotation `exp(-i θ Z / 2)` on `q`.
    rz("rz", [q], [theta]);
    /// Phase gate `diag(1, e^{iλ})` on `q`.
    p("p", [q], [lambda]);
    /// Generic single-qubit gate `u(θ, φ, λ)` on `q`.
    u("u", [q], [theta, phi, lambda]);
    /// Controlled phase with control `c`, target `t`.
    cp("cp", [c, t], [lambda]);
    /// Controlled X rotation with control `c`, target `t`.
    crx("crx", [c, t], [theta]);
    /// Controlled Y rotation with control `c`, target `t`.
    cry("cry", [c, t], [theta]);
    /// Controlled Z rotation with control `c`, target `t`.
    crz("crz", [c, t], [theta]);
    /// XX rotation on `a`, `b`.
    rxx("rxx", [a, b], [theta]);
    /// YY rotation on `a`, `b`.
    ryy("ryy", [a, b], [theta]);
    /// ZZ rotation on `a`, `b`.
    rzz("rzz", [a, b], [theta]);
}
