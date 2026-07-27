//! The research-extension workflow end to end: register a custom gate
//! family and a custom (instrumented) backend, then use both by name.
//!
//! Run with `cargo run --example research_extension`.

use quantsim::prelude::*;

/// A research backend: dense simulation plus an execution trace.
struct TracingBackend {
    inner: DenseState<C64>,
    trace: Vec<String>,
}

impl Backend<C64> for TracingBackend {
    fn name(&self) -> &str {
        "tracing"
    }
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }
    fn apply(&mut self, m: &GateMatrix<C64>, qubits: &[usize]) -> Result<()> {
        self.trace
            .push(format!("{}q gate on {qubits:?}", m.num_qubits()));
        self.inner.apply(m, qubits)
    }
    fn amplitude(&self, index: u64) -> C64 {
        self.inner.amplitude(index)
    }
    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, C64)) {
        self.inner.for_each_nonzero(f)
    }
    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        self.trace
            .push(format!("project q{qubit} -> {}", outcome as u8));
        self.inner.project(qubit, outcome, renorm)
    }
    fn reset(&mut self) {
        self.trace.clear();
        self.inner.reset()
    }
    fn load(&mut self, entries: &[(u64, C64)]) -> Result<()> {
        self.inner.load(entries)
    }
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn main() -> Result<()> {
    let mut sim: Simulator = Simulator::new();

    // 1. A research gate: the XY(θ) interaction (a rotation in the
    //    {|01⟩, |10⟩} subspace), registered like any standard gate.
    //    Registration validates unitarity at probe parameters.
    sim.registry_mut()
        .register_parametric("xy", "XY interaction", 2, 1, |p| {
            let (c, s) = ((p[0] / 2.0).cos(), (p[0] / 2.0).sin());
            let (o, l) = (c64(1.0, 0.0), c64(0.0, 0.0));
            GateMatrix::from_vec(
                4,
                vec![
                    o,
                    l,
                    l,
                    l,
                    l,
                    c64(c, 0.0),
                    c64(0.0, -s),
                    l,
                    l,
                    c64(0.0, -s),
                    c64(c, 0.0),
                    l,
                    l,
                    l,
                    l,
                    o,
                ],
            )
        })?;

    // 2. A research backend, registered beside dense/sparse/adaptive.
    sim.backends_mut().register("tracing", |n| {
        Ok(Box::new(TracingBackend {
            inner: DenseState::new(n)?,
            trace: Vec::new(),
        }))
    })?;
    println!("backends: {:?}", sim.backends().names());
    println!("registry has 'xy': {}", sim.registry().contains("xy"));

    // 3. Use both by name.
    let mut c = Circuit::new(3);
    c.h(0)
        .gate("xy", vec![std::f64::consts::FRAC_PI_2], vec![0, 1])
        .cx(1, 2)
        .gate("xy", vec![1.0], vec![1, 2]);
    let state = sim.run_on("tracing", &c)?;

    println!("\nfinal probabilities:");
    for (index, p) in state.probabilities() {
        println!("  |{index:03b}⟩ p = {p:.4}");
    }
    let tracing = state.as_any().downcast_ref::<TracingBackend>().unwrap();
    println!("\nexecution trace ({} ops):", tracing.trace.len());
    for line in &tracing.trace {
        println!("  {line}");
    }
    Ok(())
}
