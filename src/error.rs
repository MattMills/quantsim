//! Error types for the `quantsim` crate.

use std::fmt;

/// Convenience alias for `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// All errors produced by this crate.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A gate name was not found in the registry.
    UnknownGate(String),
    /// A backend name was not found in the backend registry.
    UnknownBackend(String),
    /// A gate (or alias) with this name is already registered.
    DuplicateGate(String),
    /// A backend with this name is already registered.
    DuplicateBackend(String),
    /// A qubit index was out of range for the circuit or backend width.
    QubitOutOfRange {
        /// The offending qubit index.
        qubit: usize,
        /// The width of the circuit/backend.
        num_qubits: usize,
    },
    /// The same qubit was passed more than once to a multi-qubit gate.
    DuplicateQubits {
        /// The qubit list as given.
        qubits: Vec<usize>,
    },
    /// A gate was applied to the wrong number of qubits.
    ArityMismatch {
        /// Gate name or label.
        gate: String,
        /// Number of qubits the gate acts on.
        expected: usize,
        /// Number of qubits provided.
        got: usize,
    },
    /// A gate received the wrong number of parameters.
    ParamCountMismatch {
        /// Gate name or label.
        gate: String,
        /// Number of parameters the gate takes.
        expected: usize,
        /// Number of parameters provided.
        got: usize,
    },
    /// A matrix dimension was not the expected power of two.
    BadDimension {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        got: usize,
    },
    /// A matrix dimension was not a power of two.
    NonPowerOfTwoDim(usize),
    /// A matrix failed the unitarity check.
    NotUnitary {
        /// Gate name or label.
        label: String,
        /// Max-entry deviation of `U† U` from the identity.
        deviation: f64,
    },
    /// A gate exists but cannot be constructed over the requested algebra
    /// (e.g. the phase gate `s` over the reals, which has no imaginary unit).
    UnsupportedForAlgebra {
        /// Gate name.
        gate: String,
        /// Algebra label (see `Scalar::algebra_name`).
        algebra: String,
    },
    /// Requested width exceeds what the backend supports.
    TooManyQubits {
        /// Requested width.
        requested: usize,
        /// Maximum supported width for the backend.
        max: usize,
    },
    /// Circuit width does not match backend width.
    WidthMismatch {
        /// Circuit width.
        circuit: usize,
        /// Backend width.
        backend: usize,
    },
    /// A state initialization or measurement found an invalid state
    /// (e.g. non-normalizable, or non-positive total Born weight).
    InvalidState(String),
    /// An allocation was refused by the resource guard: the request
    /// exceeds the memory measured available to this process at that
    /// moment (or the explicitly configured limit), or the allocator
    /// itself declined the reservation. See [`crate::guard`].
    OutOfMemory {
        /// Bytes the operation needed.
        requested: usize,
        /// Byte budget measured (or configured) at admission time.
        available: usize,
        /// What was being allocated (e.g. `"dense state (24 qubits)"`).
        what: String,
    },
    /// The armed wall-clock budget expired and the computation was
    /// aborted mid-operation; the state it was acting on is torn and
    /// must be discarded. See [`crate::guard::set_time_budget`].
    Timeout {
        /// The budget that was armed, in milliseconds.
        budget_ms: u64,
        /// Measured elapsed time at abort, in milliseconds.
        elapsed_ms: u64,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnknownGate(name) => write!(f, "unknown gate '{name}'"),
            Error::UnknownBackend(name) => write!(f, "unknown backend '{name}'"),
            Error::DuplicateGate(name) => write!(f, "gate '{name}' is already registered"),
            Error::DuplicateBackend(name) => write!(f, "backend '{name}' is already registered"),
            Error::QubitOutOfRange { qubit, num_qubits } => {
                write!(f, "qubit {qubit} out of range for width {num_qubits}")
            }
            Error::DuplicateQubits { qubits } => {
                write!(f, "duplicate qubit in target list {qubits:?}")
            }
            Error::ArityMismatch {
                gate,
                expected,
                got,
            } => {
                write!(f, "gate '{gate}' acts on {expected} qubit(s), got {got}")
            }
            Error::ParamCountMismatch {
                gate,
                expected,
                got,
            } => {
                write!(f, "gate '{gate}' takes {expected} parameter(s), got {got}")
            }
            Error::BadDimension { expected, got } => {
                write!(
                    f,
                    "matrix dimension {got} does not match expected {expected}"
                )
            }
            Error::NonPowerOfTwoDim(dim) => {
                write!(f, "matrix dimension {dim} is not a power of two")
            }
            Error::NotUnitary { label, deviation } => {
                write!(
                    f,
                    "matrix '{label}' is not unitary (deviation {deviation:.3e})"
                )
            }
            Error::UnsupportedForAlgebra { gate, algebra } => {
                write!(
                    f,
                    "gate '{gate}' cannot be constructed over algebra {algebra}"
                )
            }
            Error::TooManyQubits { requested, max } => {
                write!(
                    f,
                    "{requested} qubits requested, backend supports at most {max}"
                )
            }
            Error::WidthMismatch { circuit, backend } => {
                write!(
                    f,
                    "circuit width {circuit} does not match backend width {backend}"
                )
            }
            Error::InvalidState(msg) => write!(f, "invalid state: {msg}"),
            Error::OutOfMemory {
                requested,
                available,
                what,
            } => {
                write!(
                    f,
                    "out of memory: {what} needs {requested} bytes, {available} available (measured)"
                )
            }
            Error::Timeout {
                budget_ms,
                elapsed_ms,
            } => {
                write!(
                    f,
                    "time budget exceeded: aborted after {elapsed_ms} ms (budget {budget_ms} ms)"
                )
            }
        }
    }
}

impl std::error::Error for Error {}
