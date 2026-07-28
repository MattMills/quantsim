//! The dimensional lift, evidenced: a Clifford+T circuit rewritten as a
//! measurement-feedback loop in an enlarged Clifford space, run through
//! the Clifford frame, with correctness verified exactly and the cost
//! location *measured* — including the finding that the relocation does
//! not come out the way the textbook picture suggests.
//!
//! Run with `cargo run --release --example clifford_lift`.

use quantsim::prelude::*;

/// Seeded Clifford+T circuit: H-layer, Clifford stream, T/Tdg spliced.
fn clifford_t_circuit(n: usize, cliffords: usize, t: usize, seed: u64) -> Circuit {
    let mut circuit: Circuit = Circuit::new(n);
    for q in 0..n {
        circuit.h(q);
    }
    let mut rng = Prng::new(seed);
    let stride = cliffords / (t + 1);
    let mut placed = 0;
    for g in 0..cliffords {
        let q = (rng.next_u64() % n as u64) as usize;
        let r = ((q + 1) + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
        match rng.next_u64() % 6 {
            0 => circuit.h(q),
            1 => circuit.s(q),
            2 => circuit.x(q),
            3 => circuit.z(q),
            4 => circuit.cx(q, r),
            _ => circuit.cz(q, r),
        };
        if placed < t && (g + 1) % stride == 0 {
            let tq = (rng.next_u64() % n as u64) as usize;
            if placed % 2 == 0 {
                circuit.t(tq);
            } else {
                circuit.tdg(tq);
            }
            placed += 1;
        }
    }
    circuit
}

fn main() -> Result<()> {
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut().register("clifford-framed", |n| {
        Ok(Box::new(CliffordFramedState::<C64>::new(n)?))
    })?;
    // The same backend with measurement repair disabled: the pre-repair
    // drift, kept runnable so the finding stays a live comparison.
    sim.backends_mut().register("clifford-framed-drift", |n| {
        let mut state = CliffordFramedState::<C64>::new(n)?;
        state.set_measure_repair(false);
        Ok(Box::new(state))
    })?;

    let n = 6;
    let t = 8;
    let circuit = clifford_t_circuit(n, 60, t, 11);
    println!("original circuit: {n} qubits, ~60 Cliffords, {t} T/Tdg gates");
    println!(
        "lift: {} qubits — every unitary Clifford, T executed by",
        n + t
    );
    println!("measurement + conditioned Clifford correction (gate teleportation)\n");

    let mut dense = DenseState::<C64>::new(n)?;
    let reg = GateRegistry::<C64>::standard();
    circuit.bind(&reg)?.run(&mut dense)?;

    let mut direct = CliffordFramedState::<C64>::new(n)?;
    circuit.bind(&reg)?.run(&mut direct)?;
    let direct_peak = direct.peak_stored_support();

    println!(
        "  {:<28} {:>12} {:>14} {:>10}",
        "execution", "peak support", "deviation", "mechanism"
    );
    println!(
        "  {:<28} {:>12} {:>14} {:>10}",
        "direct on clifford-framed",
        direct_peak,
        format!("{:.1e}", max_amplitude_deviation(&dense, &direct)),
        "rotations"
    );

    for prep in [ResourcePrep::Upfront, ResourcePrep::JustInTime] {
        // Pre-repair drift peak, for the comparison row.
        let lifted = lift::to_clifford_feedback(&circuit, prep)?;
        let (drift_state, _) = lifted.schedule.run_on(&sim, "clifford-framed-drift", 77)?;
        let drift_peak = drift_state
            .as_any()
            .downcast_ref::<CliffordFramedState<C64>>()
            .unwrap()
            .peak_stored_support();
        println!(
            "  {:<28} {:>12} {:>14} {:>10}",
            format!("lifted, {prep:?} (drift)"),
            drift_peak,
            "-",
            "feedback"
        );

        let lifted = lift::to_clifford_feedback(&circuit, prep)?;
        let (state, trace) = lifted.schedule.run_on(&sim, "clifford-framed", 77)?;
        let framed = state
            .as_any()
            .downcast_ref::<CliffordFramedState<C64>>()
            .unwrap();
        let stats = framed.stats();
        let peak = framed.peak_stored_support();
        assert_eq!(stats.flushes, 0, "the loop itself never flushes");
        assert_eq!(stats.native_measurements, t);

        // Exact verification: divide out the known outcome phase.
        let mut outcomes = vec![false; lifted.ancillas];
        for &(_, qubit, bit) in &trace.measurements {
            outcomes[qubit - n] = bit;
        }
        let phase = lifted.outcome_phase(&outcomes);
        let mut anc_pattern = 0u64;
        for (i, &o) in outcomes.iter().enumerate() {
            if o {
                anc_pattern |= 1u64 << (n + i);
            }
        }
        let mut worst = 0.0f64;
        for idx in 0..(1u64 << n) {
            let got = state.amplitude(idx | anc_pattern) / phase;
            worst = worst.max((got - dense.amplitude(idx)).norm());
        }
        println!(
            "  {:<28} {:>12} {:>14} {:>10}",
            format!("lifted, {prep:?} (repaired)"),
            peak,
            format!("{worst:.1e}"),
            "feedback"
        );
    }

    println!("\nwhat the numbers say:");
    println!("  * the lift IS the described mechanism: after resource prep, the");
    println!("    dynamics are 100% Clifford absorption + native measurements +");
    println!("    conditioned Clifford corrections — zero flushes, physics exact");
    println!("    including the per-outcome e^(+/-i pi/4) bookkeeping.");
    println!("  * frame repair on measurement (the true tableau update, C <- C*V");
    println!("    after each projection) is what made consumption cheap: with");
    println!("    just-in-time prep at most one |T> is in flight and the loop");
    println!("    peaks near the direct route. The (drift) rows above rerun the");
    println!("    identical loop with repair disabled — the pre-repair finding,");
    println!("    kept live: both orderings drift far above the direct run.");
    println!("  * upfront prep still pays 2^t: the cost of HOLDING all t");
    println!("    resource states in one sparse register at once — a holding");
    println!("    cost, not drift. Frame-aligned sums (stabilizer-rank storage)");
    println!("    and frames over factored inners are the remaining rungs.");

    // The resource itself is cheap in a product-structured representation:
    // |T>^(x)t in the factored backend is linear in t — the composition
    // target for frames over factored inners.
    let h = reg.resolve("h")?.matrix(&[])?;
    let tg = reg.resolve("t")?.matrix(&[])?;
    print!("\n|T>^t resource alone, factored backend: ");
    for tt in [4usize, 8, 16, 32] {
        let mut f = FactoredState::<C64>::new(tt)?;
        for a in 0..tt {
            f.apply(&h, &[a])?;
            f.apply(&tg, &[a])?;
        }
        print!("t={tt}: {}B  ", f.memory_bytes());
    }
    println!("(linear — cheap to HOLD there; repair made it cheap to CONSUME");
    println!(" just-in-time, and composing the two is the factored-inner rung)");
    Ok(())
}
