//! Gate definitions: the unit of registration in the gate registry.
//!
//! A [`GateDef`] describes a named gate family — its arity, its real
//! parameters, and how to produce its unitary over the amplitude algebra
//! `S`. Research gates implement this trait (usually via the convenience
//! types [`FixedGate`] and [`ParamGate`]) and are registered on a
//! [`GateRegistry`](crate::registry::GateRegistry) alongside the standard
//! library.

pub mod standard;

use crate::error::{Error, Result};
use crate::math::GateMatrix;
use crate::scalar::Scalar;

/// Single-qubit Pauli operators, used for expectation values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pauli {
    /// Identity.
    I,
    /// Pauli X.
    X,
    /// Pauli Y (requires an algebra containing `i`).
    Y,
    /// Pauli Z.
    Z,
}

/// A named gate family over the amplitude algebra `S`.
pub trait GateDef<S: Scalar>: Send + Sync {
    /// Canonical gate name (lookup keys may also include aliases).
    fn name(&self) -> &str;
    /// Number of qubits the gate acts on.
    fn arity(&self) -> usize;
    /// Number of real parameters.
    fn param_count(&self) -> usize;
    /// One-line human description.
    fn description(&self) -> &str {
        ""
    }
    /// Produce the unitary for the given parameters.
    ///
    /// Callers guarantee `params.len() == self.param_count()` (the registry
    /// and circuit binder validate this); implementations may still return
    /// [`Error::UnsupportedForAlgebra`] when specific parameter values do
    /// not embed in `S`.
    fn matrix(&self, params: &[f64]) -> Result<GateMatrix<S>>;
}

/// A parameterless gate defined by a fixed matrix.
pub struct FixedGate<S: Scalar> {
    name: String,
    description: String,
    matrix: GateMatrix<S>,
}

impl<S: Scalar> FixedGate<S> {
    /// Build a fixed gate. The registry validates unitarity on registration.
    pub fn new(name: impl Into<String>, description: impl Into<String>, matrix: GateMatrix<S>) -> Self {
        FixedGate { name: name.into(), description: description.into(), matrix }
    }
}

impl<S: Scalar> GateDef<S> for FixedGate<S> {
    fn name(&self) -> &str {
        &self.name
    }
    fn arity(&self) -> usize {
        self.matrix.num_qubits()
    }
    fn param_count(&self) -> usize {
        0
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn matrix(&self, params: &[f64]) -> Result<GateMatrix<S>> {
        if !params.is_empty() {
            return Err(Error::ParamCountMismatch {
                gate: self.name.clone(),
                expected: 0,
                got: params.len(),
            });
        }
        Ok(self.matrix.clone())
    }
}

/// A parametric gate defined by a builder closure.
pub struct ParamGate<S: Scalar> {
    name: String,
    description: String,
    arity: usize,
    param_count: usize,
    build: Box<dyn Fn(&[f64]) -> Result<GateMatrix<S>> + Send + Sync>,
}

impl<S: Scalar> ParamGate<S> {
    /// Build a parametric gate from a closure producing its unitary.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        arity: usize,
        param_count: usize,
        build: impl Fn(&[f64]) -> Result<GateMatrix<S>> + Send + Sync + 'static,
    ) -> Self {
        ParamGate {
            name: name.into(),
            description: description.into(),
            arity,
            param_count,
            build: Box::new(build),
        }
    }
}

impl<S: Scalar> GateDef<S> for ParamGate<S> {
    fn name(&self) -> &str {
        &self.name
    }
    fn arity(&self) -> usize {
        self.arity
    }
    fn param_count(&self) -> usize {
        self.param_count
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn matrix(&self, params: &[f64]) -> Result<GateMatrix<S>> {
        if params.len() != self.param_count {
            return Err(Error::ParamCountMismatch {
                gate: self.name.clone(),
                expected: self.param_count,
                got: params.len(),
            });
        }
        (self.build)(params)
    }
}
