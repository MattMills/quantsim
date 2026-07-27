//! The [`Simulator`] facade: a gate registry plus a backend registry, with
//! one-call circuit execution.

use crate::backend::{Backend, BackendRegistry};
use crate::circuit::Circuit;
use crate::error::Result;
use crate::registry::GateRegistry;
use crate::scalar::{Scalar, C64};

/// Owns a [`GateRegistry`] and a [`BackendRegistry`] over one amplitude
/// algebra, and runs circuits against them.
///
/// ```
/// use quantsim::prelude::*;
///
/// let sim: Simulator = Simulator::new(); // Simulator<C64>
/// let state = sim.run(&library::bell()).unwrap();
/// assert!((state.probability(0b00) - 0.5).abs() < 1e-12);
/// assert!((state.probability(0b11) - 0.5).abs() < 1e-12);
/// ```
///
/// For a different algebra, name it: `Simulator::<Quaternion>::new()`.
pub struct Simulator<S: Scalar = C64> {
    registry: GateRegistry<S>,
    backends: BackendRegistry<S>,
}

impl<S: Scalar> Default for Simulator<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> Simulator<S> {
    /// A simulator with the standard gate library (as far as the algebra
    /// supports it) and the built-in backends.
    pub fn new() -> Self {
        Simulator { registry: GateRegistry::standard(), backends: BackendRegistry::standard() }
    }

    /// Build from explicit registries.
    pub fn with_registries(registry: GateRegistry<S>, backends: BackendRegistry<S>) -> Self {
        Simulator { registry, backends }
    }

    /// The gate registry.
    pub fn registry(&self) -> &GateRegistry<S> {
        &self.registry
    }

    /// Mutable gate registry (register research gates here).
    pub fn registry_mut(&mut self) -> &mut GateRegistry<S> {
        &mut self.registry
    }

    /// The backend registry.
    pub fn backends(&self) -> &BackendRegistry<S> {
        &self.backends
    }

    /// Mutable backend registry (register research backends here).
    pub fn backends_mut(&mut self) -> &mut BackendRegistry<S> {
        &mut self.backends
    }

    /// Bind and run `circuit` on the default `"dense"` backend, returning the
    /// final state.
    pub fn run(&self, circuit: &Circuit<S>) -> Result<Box<dyn Backend<S>>> {
        self.run_on("dense", circuit)
    }

    /// Bind and run `circuit` on the named backend, returning the final
    /// state.
    pub fn run_on(&self, backend: &str, circuit: &Circuit<S>) -> Result<Box<dyn Backend<S>>> {
        let mut state = self.backends.create(backend, circuit.num_qubits())?;
        let bound = circuit.bind(&self.registry)?;
        bound.run(state.as_mut())?;
        Ok(state)
    }

    /// Apply a single registry gate directly to an existing state — useful
    /// for classically controlled corrections after a mid-circuit
    /// [`Backend::measure`].
    pub fn apply(
        &self,
        state: &mut dyn Backend<S>,
        gate: &str,
        params: &[f64],
        qubits: &[usize],
    ) -> Result<()> {
        let def = self.registry.resolve(gate)?;
        if def.arity() != qubits.len() {
            return Err(crate::error::Error::ArityMismatch {
                gate: gate.to_string(),
                expected: def.arity(),
                got: qubits.len(),
            });
        }
        if def.param_count() != params.len() {
            return Err(crate::error::Error::ParamCountMismatch {
                gate: gate.to_string(),
                expected: def.param_count(),
                got: params.len(),
            });
        }
        let matrix = def.matrix(params)?;
        state.apply(&matrix, qubits)
    }
}
