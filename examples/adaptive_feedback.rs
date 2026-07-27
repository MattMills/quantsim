//! Adaptive measurement trees, live: recursive feedback in the evented
//! scheduler (measurements whose branches measure again), replayed
//! identically on the dense reference and on the Clifford frame — where
//! native measurement plus frame repair keeps unbounded adaptive
//! sequences flat.
//!
//! Run with `cargo run --release --example adaptive_feedback`.

use quantsim::prelude::*;

fn sim_with_clifford() -> Simulator {
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("clifford-framed", |n| {
            Ok(Box::new(CliffordFramedState::<C64>::new(n)?))
        })
        .unwrap();
    sim
}

fn main() -> Result<()> {
    let sim = sim_with_clifford();

    // ══ 1. a bounded repeat-until-success ladder, traced ══
    println!("══ 1. repeat-until-success ladder (depth 3), traced per seed ══\n");
    let rung3 = vec![FeedbackOp::gate(1, "x", [], [0])];
    let rung2 = vec![
        FeedbackOp::gate(1, "x", [], [0]),
        FeedbackOp::gate(2, "h", [], [0]),
        FeedbackOp::measure(3, 0, vec![], rung3),
    ];
    let rung1 = vec![
        FeedbackOp::gate(1, "x", [], [0]),
        FeedbackOp::gate(2, "h", [], [0]),
        FeedbackOp::measure(3, 0, vec![], rung2),
    ];
    let mut shown = [false; 3];
    for seed in 0..64u64 {
        let mut schedule: Schedule = Schedule::new(1, 100);
        schedule.at(0, "h", [], [0]);
        schedule.measure_branching_at(1, 0, vec![], rung1.clone());
        let (state, trace) = schedule.run_on(&sim, "dense", seed)?;
        let attempts = trace.measurements.len();
        if shown[attempts - 1] {
            continue; // show one representative path per retry depth
        }
        shown[attempts - 1] = true;
        let outcomes: Vec<u8> = trace
            .measurements
            .iter()
            .map(|&(_, _, o)| o as u8)
            .collect();
        let path: Vec<String> = trace
            .events
            .iter()
            .map(|e| format!("t{}:{}", e.time, e.label))
            .collect();
        println!(
            "  seed {seed}: {attempts} attempt(s), outcomes {outcomes:?}, P(|0⟩) = {:.1}",
            state.probability(0)
        );
        println!("    {}", path.join(" → "));
        if shown.iter().all(|&s| s) {
            break;
        }
    }
    println!("  (a failure enqueues reset + retry + a nested measurement — a finite");
    println!("   tree, so termination is structural, not hoped for)\n");

    // ══ 2. outcome-dependent measurement choice on GHZ ══
    println!("══ 2. nested feedback on GHZ-3: measure q1 only if q0 read 1 ══\n");
    let build = || {
        let mut schedule: Schedule = Schedule::new(3, 100);
        schedule.at(0, "h", [], [0]);
        schedule.at(1, "cx", [], [0, 1]);
        schedule.at(2, "cx", [], [1, 2]);
        schedule.measure_branching_at(
            10,
            0,
            vec![],
            vec![FeedbackOp::measure(
                1,
                1,
                vec![],
                vec![FeedbackOp::gate(1, "x", [], [2])],
            )],
        );
        schedule
    };
    let mut branch_shown = [false; 2];
    for seed in 0..64u64 {
        let (dense_state, dense_trace) = build().run_on(&sim, "dense", seed)?;
        let outer = dense_trace.measurements[0].2;
        if branch_shown[outer as usize] {
            continue; // one representative seed per outer outcome
        }
        branch_shown[outer as usize] = true;
        let (framed_state, framed_trace) = build().run_on(&sim, "clifford-framed", seed)?;
        assert_eq!(dense_trace.measurements, framed_trace.measurements);
        let outcomes: Vec<u8> = dense_trace
            .measurements
            .iter()
            .map(|&(_, _, o)| o as u8)
            .collect();
        let framed = framed_state
            .as_any()
            .downcast_ref::<CliffordFramedState<C64>>()
            .unwrap();
        let stats = framed.stats();
        println!(
            "  seed {seed}: outcomes {outcomes:?} ({} measurement(s) fired) — dense and frame agree",
            outcomes.len()
        );
        println!(
            "    frame: {} native measurements, {} repairs, {} flushes, peak stored support {}",
            stats.native_measurements,
            stats.frame_repairs,
            stats.flushes,
            framed.peak_stored_support()
        );
        let expect = if outcomes[0] == 1 { 0b011 } else { 0b000 };
        println!(
            "    post-state P(|{expect:03b}⟩) = {:.1} on both\n",
            dense_state.probability(expect)
        );
        if branch_shown.iter().all(|&s| s) {
            break;
        }
    }

    // ══ 3. repair keeps long adaptive sequences flat ══
    println!("══ 3. width 40: a 300-gate Clifford scramble, then measure every qubit ══\n");
    let n = 40;
    let mut c = Circuit::new(n);
    let mut rng = Prng::new(23);
    for _ in 0..300 {
        let q = (rng.next_u64() % n as u64) as usize;
        let r = ((q + 1) + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
        match rng.next_u64() % 6 {
            0 => c.h(q),
            1 => c.s(q),
            2 => c.x(q),
            3 => c.z(q),
            4 => c.cx(q, r),
            _ => c.cz(q, r),
        };
    }
    let reg = GateRegistry::<C64>::standard();
    let bound = c.bind(&reg)?;
    println!(
        "  {:>10} {:>14} {:>14} {:>14} {:>10}",
        "repair", "measurements", "peak support", "final support", "repairs"
    );
    for (repair, m) in [(false, 8usize), (true, 8), (true, 40)] {
        let mut state = CliffordFramedState::<C64>::new(n)?;
        state.set_measure_repair(repair);
        bound.run(&mut state)?;
        let mut rng = Prng::new(41);
        for q in 0..m {
            state.measure(q % n, &mut rng)?;
        }
        println!(
            "  {:>10} {:>14} {:>14} {:>14} {:>10}",
            if repair { "on" } else { "off" },
            m,
            state.peak_stored_support(),
            state.stored_nonzero_count(),
            state.stats().frame_repairs
        );
    }
    println!("\n  off: projection growth compounds (2^m — the pre-repair envelope, kept");
    println!("  measurable). on: each projection is repaired into the frame (C ← C·V),");
    println!("  so the same seeded outcomes cost a flat peak of 2 at ANY sequence length —");
    println!("  the Gottesman–Knill measurement sector through the frame mechanism.");
    Ok(())
}
