//! Print the evidence, not the test names: runs the same verifications the
//! test suite asserts — conformance sweeps, interference ledgers, physical
//! routing logs, and verified kernel selection — and shows the raw results.
//!
//! Run with `cargo run --release --example verify_models`.

use quantsim::prelude::*;

/// Deliberately fast-and-wrong kernel (skips every gate): selection must
/// reject it on measured deviation, never on suspicion.
struct NoOpBackend {
    inner: DenseState<C64>,
}

impl Backend<C64> for NoOpBackend {
    fn name(&self) -> &str {
        "noop"
    }
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }
    fn apply(&mut self, _: &GateMatrix<C64>, _: &[usize]) -> Result<()> {
        Ok(())
    }
    fn amplitude(&self, index: u64) -> C64 {
        self.inner.amplitude(index)
    }
    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, C64)) {
        self.inner.for_each_nonzero(f)
    }
    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        self.inner.project(qubit, outcome, renorm)
    }
    fn reset(&mut self) {
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
    sim.backends_mut()
        .register("interference", |n| Ok(Box::new(InterferenceState::new(n)?)))?;
    sim.backends_mut().register("device-linear", |n| {
        Ok(Box::new(DeviceState::new(
            Topology::linear(n),
            DurationModel::default(),
            ArityPolicy::default(),
        )?))
    })?;
    sim.backends_mut().register("device-ring", |n| {
        Ok(Box::new(DeviceState::new(
            Topology::ring(n),
            DurationModel::default(),
            ArityPolicy::default(),
        )?))
    })?;
    sim.backends_mut().register("noop", |n| {
        Ok(Box::new(NoOpBackend {
            inner: DenseState::new(n)?,
        }))
    })?;
    sim.backends_mut().register("framed-sparse", |n| {
        Ok(Box::new(FramedState::new(Box::new(
            SparseState::<C64>::new(n)?,
        ))))
    })?;
    sim.backends_mut().register("clifford-framed", |n| {
        Ok(Box::new(CliffordFramedState::<C64>::new(n)?))
    })?;

    // ════════════════════ 1. Conformance, printed ════════════════════
    // Every shipped representation — including the frame wrappers and the
    // hierarchical backend — against the dense reference, over the whole
    // registry. Same harness, no exceptions.
    println!("════ 1. conformance sweeps (live, not cached) ════\n");
    let cfg = ConformanceConfig::default();
    for backend in [
        "sparse",
        "adaptive",
        "factored",
        "mps",
        "mera",
        "interference",
        "device-linear",
        "device-ring",
        "framed-sparse",
        "clifford-framed",
    ] {
        let report = verify_backend(&sim, backend, &cfg)?;
        print!("{report}");
    }

    // The algebra axis goes through the same harness: the Ball scalar
    // (certified midpoint ± radius) runs the identical sweeps — midpoint
    // physics is C64 physics, radii ride along.
    let ball_sim: Simulator<Ball> = Simulator::new();
    for backend in ["sparse", "mera"] {
        let report = verify_backend(&ball_sim, backend, &cfg)?;
        print!("{report}");
    }

    // ════════════════════ 2. Interference ledger ════════════════════
    println!("\n════ 2. interference: independent constructive/destructive accounting ════\n");
    {
        let mut state = InterferenceState::<C64>::new(1)?;
        let h = sim.registry().resolve("h")?.matrix(&[])?;
        state.apply(&h, &[0])?;
        state.apply(&h, &[0])?;
        println!("H·H on |0⟩ — per-gate ledger:");
        for r in state.records() {
            println!(
                "  gate {}: path weight {:.6}, net {:.6}, destroyed {:.6}, constructive fraction {:.3}",
                r.gate_index,
                r.path_weight,
                r.net_weight,
                r.destroyed(),
                r.constructive_fraction()
            );
        }
        println!(
            "  destruction landed at: |0⟩ → {:.6}, |1⟩ → {:.6}",
            state.destruction_map()[0],
            state.destruction_map()[1]
        );
        println!(
            "  final amplitudes: |0⟩ = {}, |1⟩ = {}\n",
            state.amplitude(0),
            state.amplitude(1)
        );
    }
    {
        let circuit = library::grover(3, 5, 2)?;
        let state = sim.run_on("interference", &circuit)?;
        let interference = state
            .as_any()
            .downcast_ref::<InterferenceState<C64>>()
            .unwrap();
        let labels: Vec<String> = circuit
            .ops()
            .iter()
            .map(|op| match op {
                Op::Named { name, qubits, .. } => format!("{name}{qubits:?}"),
                Op::Diagonal { label, qubits, .. } => format!("{label}{qubits:?}"),
                Op::Raw { label, qubits, .. } => format!("{label}{qubits:?}"),
            })
            .collect();
        println!("Grover(n=3, marked=|101⟩, 2 iterations) — per-gate interference:");
        println!(
            "  {:<22} {:>10} {:>10} {:>10}",
            "gate", "path", "net", "destroyed"
        );
        for (label, r) in labels.iter().zip(interference.records()) {
            println!(
                "  {:<22} {:>10.4} {:>10.4} {:>10.4}",
                label,
                r.path_weight,
                r.net_weight,
                r.destroyed()
            );
        }
        println!(
            "  total destroyed: {:.4};  P(|101⟩) = {:.7}  (closed form 0.9453125)\n",
            interference.total_destroyed(),
            state.probability(5)
        );
    }

    // ════════════════ 3. Device: physical operation order ════════════════
    println!("════ 3. device model: routing, latency, physical order ════\n");
    {
        let mut c = Circuit::new(6);
        c.h(0).cx(0, 5);
        let state = sim.run_on("device-linear", &c)?;
        let device = state.as_any().downcast_ref::<DeviceState<C64>>().unwrap();
        println!("h(0); cx(0,5) on a 6-site linear chain — physical log:");
        for op in device.physical_log() {
            println!(
                "  t{:>3}–{:<3} {:<8} at sites {:?}",
                op.start, op.end, op.label, op.sites
            );
        }
        println!(
            "  swaps: {}, elapsed: {} ticks, logical→physical mapping: {:?}",
            device.swap_count(),
            device.elapsed(),
            device.mapping()
        );
        println!(
            "  logical result (must be the Bell pair on qubits 0,5): amp|000000⟩ = {}, amp|100001⟩ = {}",
            state.amplitude(0),
            state.amplitude(0b100001)
        );
        let mut c2 = Circuit::new(6);
        c2.h(0).cx(0, 5).cx(0, 5);
        let s2 = sim.run_on("device-linear", &c2)?;
        let d2 = s2.as_any().downcast_ref::<DeviceState<C64>>().unwrap();
        println!(
            "  …repeating cx(0,5): total swaps still {} — locality persisted\n",
            d2.swap_count()
        );
    }
    {
        let mut c = Circuit::new(8);
        c.h(0).cx(0, 7);
        let linear = sim.run_on("device-linear", &c)?;
        let ring = sim.run_on("device-ring", &c)?;
        let linear = linear.as_any().downcast_ref::<DeviceState<C64>>().unwrap();
        let ring = ring.as_any().downcast_ref::<DeviceState<C64>>().unwrap();
        println!("cx(0,7) on 8 qubits — topology changes cost, not physics:");
        println!(
            "  linear chain: {} swaps, {} ticks;  ring: {} swaps, {} ticks",
            linear.swap_count(),
            linear.elapsed(),
            ring.swap_count(),
            ring.elapsed()
        );
        let deviation =
            max_amplitude_deviation(linear as &dyn Backend<C64>, ring as &dyn Backend<C64>);
        println!("  amplitude deviation between the two runs: {deviation:.3e}\n");
    }

    // ════════════════ 4. VOLK-style verified selection ════════════════
    println!("════ 4. runtime kernel selection with fidelity gate ════\n");
    let workload = Workload::ghz(16);
    let report = select_backend(
        &sim,
        &workload,
        &["dense", "sparse", "factored", "mps", "mera", "noop"],
        &BenchConfig::default(),
        SelectionCriterion::Time,
        1e-9,
    )?;
    println!("workload ghz-16, criterion Time:");
    print!("{}", report.bench);
    for (name, reason) in &report.rejected {
        println!("  REJECTED {name}: {reason}");
    }
    println!(
        "  chosen: {:?} (verified against reference: {})\n",
        report.chosen, report.verified
    );

    let report = select_backend(
        &sim,
        &Workload::qft(10),
        &["dense", "sparse", "adaptive", "mps", "mera"],
        &BenchConfig::default(),
        SelectionCriterion::Time,
        1e-9,
    )?;
    println!(
        "workload qft-10, criterion Time → chosen: {:?} (verified: {})",
        report.chosen, report.verified
    );

    let mut pairs: Circuit = Circuit::new(34);
    for pair in 0..17 {
        pairs
            .ry(2 * pair, 0.2 + 0.05 * pair as f64)
            .cx(2 * pair, 2 * pair + 1);
    }
    let report = select_backend(
        &sim,
        &Workload::from_circuit("local-pairs-34", pairs),
        &["sparse", "factored", "mps", "mera"],
        &BenchConfig::default(),
        SelectionCriterion::Memory,
        1e-9,
    )?;
    println!(
        "workload local-pairs-34 (reference cannot run at width 34), criterion Memory → chosen: {:?} (verified: {})",
        report.chosen, report.verified
    );
    Ok(())
}
