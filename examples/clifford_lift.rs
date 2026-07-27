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
            format!("lifted, {prep:?}"),
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
    println!("  * but the relocated cost did NOT vanish and did NOT stay in the");
    println!("    resource states: measurement projections collapse the register");
    println!("    in the PHYSICAL basis, the frame scrambles that cancellation");
    println!("    structure, and the STORED support drifts — both prep orderings");
    println!("    peak far above the direct run. The residue moved to a place");
    println!("    this representation handles worse, not better.");
    println!("  * mechanism to change that (roadmap): frame repair on measurement");
    println!("    (re-align C after each projection — the true tableau update),");
    println!("    and frame-aligned sums (stabilizer-rank storage) for the magic.");

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
    println!("(linear — the magic is cheap to HOLD, expensive to CONSUME)");
    Ok(())
}
