//! The research stack in one run:
//!
//! 1. **simultaneous gate loops** on one register, driven by the evented
//!    scheduler (compute loop + memory-refresh loop + interaction events +
//!    measurement feedback),
//! 2. **quantum memory interacting with computation** on the factored
//!    backend — idle cells cost two amplitudes; interactions merge them
//!    into the compute geometry and measurement releases them,
//! 3. **transparent signal threads** — an n-wide insertion that adds a
//!    parity signal at one point of the circuit graph and removes it at
//!    another, verified computation-neutral as a single unit, with its
//!    geometric cost measured.
//!
//! Run with `cargo run --release --example evented_memory`.

use quantsim::prelude::*;

fn main() -> Result<()> {
    let sim: Simulator = Simulator::new();

    // ── A 36-qubit machine: compute region 0..6, memory bank 6..36 ──────
    // Dense would need 1 TiB at this width; the factored backend needs a
    // few KiB as long as entanglement stays where we put it.
    let n = 36;
    let mut schedule: Schedule = Schedule::new(n, 60);

    // Memory prep: 15 Bell cells (pairs) storing entangled values.
    for cell in 0..15 {
        let (a, b) = (6 + 2 * cell, 7 + 2 * cell);
        schedule.at(0, "h", Vec::new(), vec![a]);
        schedule.at(1, "cx", Vec::new(), vec![a, b]);
    }
    // Compute loop: period-3 entangling block on the compute region.
    schedule.add_loop(GateLoop {
        name: "compute".into(),
        start: 2,
        period: 3,
        iterations: None,
        body: vec![
            TimedOp::gate(0, "ry", vec![0.37], vec![0]),
            TimedOp::gate(0, "ry", vec![0.81], vec![2]),
            TimedOp::gate(1, "cx", Vec::new(), vec![0, 1]),
            TimedOp::gate(1, "cx", Vec::new(), vec![2, 3]),
            TimedOp::gate(2, "cp", vec![0.55], vec![1, 3]),
        ],
    });
    // Memory refresh loop, simultaneous with compute: gentle phase
    // maintenance on the first three cells (diagonal → never entangles).
    schedule.add_loop(GateLoop {
        name: "refresh".into(),
        start: 2,
        period: 5,
        iterations: None,
        body: vec![
            TimedOp::gate(0, "rz", vec![0.01], vec![6]),
            TimedOp::gate(0, "rz", vec![-0.01], vec![8]),
            TimedOp::gate(0, "rz", vec![0.01], vec![10]),
        ],
    });
    // Interaction: load memory cell 0 into the computation, use it, then
    // measure it back out (with feedback marking the outcome on cell 1).
    schedule.at(20, "cx", Vec::new(), vec![6, 4]);
    schedule.at(25, "cp", vec![0.9], vec![4, 5]);
    schedule.measure_at(
        40,
        4,
        vec![],
        vec![TimedOp::gate(1, "x", Vec::new(), vec![8])],
    );

    let (state, trace) = schedule.run_on(&sim, "factored", 7)?;
    let factored = state.as_any().downcast_ref::<FactoredState<C64>>().unwrap();

    println!("== evented run on {n} qubits (factored backend) ==");
    println!(
        "events applied: {}, measurements: {:?}",
        trace.events.len(),
        trace.measurements
    );
    println!(
        "final factor geometry ({} factors):",
        factored.factor_count()
    );
    for factor in factored.factors() {
        println!("  {factor:?}");
    }
    let (peak_amps, peak_width) = factored.peak();
    println!(
        "memory now: {} B; peak: {peak_amps} amplitudes, widest factor {peak_width} qubits",
        state.memory_bytes()
    );
    let dense_tib = 16.0 * (n as f64).exp2() / (1u64 << 40) as f64;
    println!("(dense at width {n} would be {dense_tib:.1} TiB)\n");

    // ── Transparent signal thread on the compute region ────────────────
    let mut base = Circuit::new(4);
    base.h(0).h(1); // 0, 1
    base.rz(0, 0.7).cp(0, 1, 1.1).rz(1, 0.4); // 2, 3, 4: diagonal segment
    base.h(0).h(1); // 5, 6

    let thread = Insertion::new("parity-rail")
        .gate(2, "cx", Vec::<f64>::new(), vec![0, 3])
        .gate(2, "cx", Vec::<f64>::new(), vec![1, 3])
        .gate(5, "cx", Vec::<f64>::new(), vec![1, 3])
        .gate(5, "cx", Vec::<f64>::new(), vec![0, 3]);
    let report = verify_transparent(&sim, &base, &thread, 1e-9)?;
    println!("== signal thread '{}' ==", report.name);
    println!(
        "transparent: {} (final deviation {:.1e}, weight drift {:.1e})",
        report.transparent, report.final_deviation, report.weight_drift
    );
    println!(
        "factored peaks — without: {:?}, with: {:?} (the thread's measured geometric cost)",
        report.peak_without, report.peak_with
    );

    // ── Point-stabilizer discovery on the current compute state ────────
    let compute_state = sim.run(&base)?;
    let reg = sim.registry();
    let crz = reg.resolve("crz")?;
    let grid: Vec<Vec<f64>> = (1..=4).map(|k| vec![k as f64 * 0.6]).collect();
    // q2 is untouched (|0⟩): any rotation controlled on it stabilizes.
    let found = discover_stabilizers(compute_state.as_ref(), crz.as_ref(), &[2, 0], &grid, 1e-9)?;
    println!("\n== stabilizer discovery ==");
    println!(
        "crz with control on idle q2: {}/{} parameter draws stabilize (insertion points found)",
        found.len(),
        grid.len()
    );
    let rz = reg.resolve("rz")?;
    let found = discover_stabilizers(compute_state.as_ref(), rz.as_ref(), &[0], &grid, 1e-9)?;
    println!(
        "rz on active q0: {}/{} stabilize (as expected, structure costs something here)",
        found.len(),
        grid.len()
    );
    Ok(())
}
