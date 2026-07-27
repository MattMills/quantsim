//! The gate registry: a name → [`GateDef`] catalog, per amplitude algebra.
//!
//! The registry is the extension point for research gates: implement
//! [`GateDef`] (or use the [`register_fixed`](GateRegistry::register_fixed) /
//! [`register_parametric`](GateRegistry::register_parametric) helpers), and
//! circuits can immediately reference the new gate by name. Registration
//! validates dimensions and unitarity at probe parameters, so typos in a
//! matrix fail fast rather than corrupting simulations.
//!
//! ```
//! use quantsim::prelude::*;
//!
//! let mut reg: GateRegistry = GateRegistry::standard();
//! // An "XY" interaction gate — a rotation in the {|01⟩, |10⟩} subspace.
//! reg.register_parametric("xy", "XY interaction", 2, 1, |p| {
//!     let (c, s) = ((p[0] / 2.0).cos(), (p[0] / 2.0).sin());
//!     let (o, l) = (c64(1.0, 0.0), c64(0.0, 0.0));
//!     GateMatrix::try_from_c64s(4, &[
//!         o, l, l, l,
//!         l, c64(c, 0.0), c64(0.0, -s), l,
//!         l, c64(0.0, -s), c64(c, 0.0), l,
//!         l, l, l, o,
//!     ]).ok_or(Error::UnsupportedForAlgebra { gate: "xy".into(), algebra: "R".into() })
//! }).unwrap();
//! assert!(reg.contains("xy"));
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::gates::standard::{install_standard, PROBE_PARAMS};
use crate::gates::{FixedGate, GateDef, ParamGate};
use crate::math::{GateMatrix, UNITARY_TOL};
use crate::scalar::{Scalar, C64};

/// A catalog of gate definitions keyed by name (and aliases).
///
/// Generic over the amplitude algebra: `GateRegistry::<f64>::standard()`
/// contains only the real-matrix subset of the standard library, while the
/// default `GateRegistry<C64>` (and ℍ, 𝕆, ... — anything ℂ embeds into)
/// carries all of it.
pub struct GateRegistry<S: Scalar = C64> {
    gates: HashMap<String, Arc<dyn GateDef<S>>>,
}

impl<S: Scalar> Default for GateRegistry<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> GateRegistry<S> {
    /// An empty registry.
    pub fn new() -> Self {
        GateRegistry { gates: HashMap::new() }
    }

    /// A registry pre-loaded with every standard gate the algebra supports.
    pub fn standard() -> Self {
        let mut reg = Self::new();
        install_standard(&mut reg).expect("standard gate installation cannot conflict");
        reg
    }

    /// Register a gate definition under its canonical name.
    ///
    /// Validation: the name must be new; the matrix produced at probe
    /// parameters must have dimension `2^arity` and be unitary within
    /// [`UNITARY_TOL`]. A definition that reports
    /// [`Error::UnsupportedForAlgebra`] at the probe is rejected with that
    /// error (register it on an algebra that supports it instead).
    pub fn register(&mut self, def: Arc<dyn GateDef<S>>) -> Result<()> {
        let name = def.name().to_string();
        if name.is_empty() {
            return Err(Error::UnknownGate(String::new()));
        }
        if self.gates.contains_key(&name) {
            return Err(Error::DuplicateGate(name));
        }
        // Probe at generic non-special angles (cycled if a research gate
        // takes more parameters than the probe list).
        let mut probe = Vec::with_capacity(def.param_count());
        for i in 0..def.param_count() {
            probe.push(PROBE_PARAMS[i % PROBE_PARAMS.len()]);
        }
        let m = def.matrix(&probe)?;
        let expected = 1usize << def.arity();
        if m.dim() != expected {
            return Err(Error::BadDimension { expected, got: m.dim() });
        }
        let dev = m.unitarity_deviation();
        if dev > UNITARY_TOL {
            return Err(Error::NotUnitary { label: name, deviation: dev });
        }
        self.gates.insert(name, def);
        Ok(())
    }

    /// Register a parameterless gate from a fixed matrix.
    pub fn register_fixed(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        matrix: GateMatrix<S>,
    ) -> Result<()> {
        self.register(Arc::new(FixedGate::new(name, description, matrix)))
    }

    /// Register a parametric gate from a builder closure.
    pub fn register_parametric(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        arity: usize,
        param_count: usize,
        build: impl Fn(&[f64]) -> Result<GateMatrix<S>> + Send + Sync + 'static,
    ) -> Result<()> {
        self.register(Arc::new(ParamGate::new(name, description, arity, param_count, build)))
    }

    /// Add an alias for an existing gate.
    pub fn alias(&mut self, alias: impl Into<String>, existing: &str) -> Result<()> {
        let alias = alias.into();
        if self.gates.contains_key(&alias) {
            return Err(Error::DuplicateGate(alias));
        }
        let def = self.resolve(existing)?;
        self.gates.insert(alias, def);
        Ok(())
    }

    /// Look up a gate by name or alias.
    pub fn get(&self, name: &str) -> Option<Arc<dyn GateDef<S>>> {
        self.gates.get(name).cloned()
    }

    /// Look up a gate, failing with [`Error::UnknownGate`].
    pub fn resolve(&self, name: &str) -> Result<Arc<dyn GateDef<S>>> {
        self.get(name).ok_or_else(|| Error::UnknownGate(name.to_string()))
    }

    /// Whether a gate (or alias) with this name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.gates.contains_key(name)
    }

    /// All registered names (including aliases), sorted.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.gates.keys().cloned().collect();
        names.sort();
        names
    }

    /// Number of registered names (including aliases).
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }
}
