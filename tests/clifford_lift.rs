//! The dimensional lift: a Clifford+T circuit rewritten as a feedback
//! loop in an (n+t)-qubit Clifford space — every unitary Clifford, the
//! non-Cliffordness relocated into resource states, T gates executed by
//! measurement + conditioned Clifford correction. Correctness is exact
//! (outcome phases divided out, not waved away), and *where* the cost
//! condenses is measured, not assumed.

mod common;

use common::*;
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
    assert_eq!(placed, t);
    circuit
}

/// Run a lifted circuit and compare the data register **exactly** against
/// the unlifted circuit on dense: ancillas collapse to their measured
/// outcomes, and each outcome-1 gadget contributes a known e^{±iπ/4}
/// which `outcome_phase` removes. Returns the framed state's stats and
/// peak stored support (read before verification flushed anything).
fn verify_lift(circuit: &Circuit, prep: ResourcePrep, seed: u64) -> (CliffordFrameStats, usize) {
    let sim = sim_with_clifford();
    let lifted = lift::to_clifford_feedback(circuit, prep).unwrap();
    let n = lifted.data_qubits;
    let (state, trace) = lifted
        .schedule
        .run_on(&sim, "clifford-framed", seed)
        .unwrap();

    // Representation costs BEFORE observation flushes anything.
    let framed = state
        .as_any()
        .downcast_ref::<CliffordFramedState<C64>>()
        .unwrap();
    let stats = framed.stats();
    let peak = framed.peak_stored_support();

    // Ancilla outcomes in ancilla order (trace records (tick, qubit, bit)).
    let mut outcomes = vec![false; lifted.ancillas];
    for &(_, qubit, bit) in &trace.measurements {
        assert!(qubit >= n, "only ancillas are measured in the lift");
        outcomes[qubit - n] = bit;
    }
    assert_eq!(trace.measurements.len(), lifted.ancillas);
    let phase = lifted.outcome_phase(&outcomes);

    // The joint state is |data⟩ ⊗ |outcome bits⟩ (measured ancillas are
    // definite): compare every data amplitude at the ancilla pattern.
    let mut anc_pattern = 0u64;
    for (i, &o) in outcomes.iter().enumerate() {
        if o {
            anc_pattern |= 1u64 << (n + i);
        }
    }
    let dense = run_dense(circuit);
    let mut worst = 0.0f64;
    for idx in 0..(1u64 << n) {
        let got = state.amplitude(idx | anc_pattern) / phase;
        let want = dense.amplitude(idx);
        worst = worst.max((got - want).norm());
    }
    assert!(
        worst < 1e-9,
        "lifted ({prep:?}) deviates from unlifted dense by {worst}"
    );
    // Nothing may live outside the ancilla pattern.
    assert_close(state.total_weight(), 1.0, 1e-9);
    let mut on_pattern = 0.0;
    state.for_each_nonzero(&mut |i, a| {
        if i >> n == anc_pattern >> n {
            on_pattern += a.born_weight();
        }
    });
    assert_close(on_pattern, 1.0, 1e-9);
    (stats, peak)
}

#[test]
fn lifted_circuits_reproduce_the_unlifted_physics_exactly() {
    for (n, cliffs, t, seed) in [
        (4usize, 40usize, 3usize, 5u64),
        (6, 60, 5, 6),
        (5, 50, 4, 7),
    ] {
        let circuit = clifford_t_circuit(n, cliffs, t, seed);
        for prep in [ResourcePrep::Upfront, ResourcePrep::JustInTime] {
            let (stats, _) = verify_lift(&circuit, prep, 1000 + seed);
            // The lifted dynamics were *entirely* Clifford + measurement:
            // the only amplitude work was the t resource rotations.
            assert_eq!(stats.axis_rotations, t, "{prep:?}: t resource preps");
            assert_eq!(stats.native_measurements, t, "{prep:?}");
            assert_eq!(stats.flushes, 0, "{prep:?}: the loop never flushed");
            assert_eq!(stats.raw_fallbacks, 0, "{prep:?}");
            assert_eq!(stats.zyz_decompositions, 0, "{prep:?}");
        }
    }
}

#[test]
fn corrections_actually_fire_and_are_clifford() {
    // Drive one gadget to each outcome by seed hunting: outcome 1 must
    // trigger the S correction (absorbed, Clifford), outcome 0 must not.
    let mut circuit: Circuit = Circuit::new(1);
    circuit.h(0).t(0);
    let sim = sim_with_clifford();
    let mut seen = [false, false];
    for seed in 0..32u64 {
        let lifted = lift::to_clifford_feedback(&circuit, ResourcePrep::JustInTime).unwrap();
        let (_, trace) = lifted
            .schedule
            .run_on(&sim, "clifford-framed", seed)
            .unwrap();
        let outcome = trace.measurements[0].2;
        seen[outcome as usize] = true;
        // Correction S shows up in the trace exactly on outcome 1.
        let corrections = trace
            .events
            .iter()
            .filter(|e| e.label == "s" && e.qubits == vec![0])
            .count();
        assert_eq!(corrections, usize::from(outcome), "seed {seed}");
        if seen[0] && seen[1] {
            return;
        }
    }
    panic!("32 seeds never produced both outcomes");
}

#[test]
fn lift_cost_location_is_measured_not_assumed() {
    // The prediction this test originally carried — just-in-time prep
    // keeps the peak near the direct run because each measurement
    // "collapses" its ancilla — is FALSE in this representation, and the
    // measurement below is the finding: the gadget collapse happens in
    // the *physical* basis, the frame scrambles its cancellation
    // structure, and the *stored* support drifts regardless of prep
    // order (measured: both orderings peak at 8192 = 2^13 on a
    // 14-qubit lift whose direct run peaks ≤ 64). The lift is exact and
    // its dynamics are all-Clifford, but in an amplitude-backed frame
    // the relocated cost lands on stored-basis drift under projection —
    // the measured motivation for frame *repair* on measurement and for
    // frame-aligned-sum (stabilizer-rank) storage, both roadmap rungs.
    let n = 6;
    let t = 8;
    let circuit = clifford_t_circuit(n, 60, t, 11);
    let (_, upfront_peak) = verify_lift(&circuit, ResourcePrep::Upfront, 77);
    let (_, jit_peak) = verify_lift(&circuit, ResourcePrep::JustInTime, 77);

    // Direct (unlifted) run on the same framed backend for reference.
    let reg = GateRegistry::<C64>::standard();
    let mut direct = CliffordFramedState::<C64>::new(n).unwrap();
    circuit.bind(&reg).unwrap().run(&mut direct).unwrap();
    let direct_peak = direct.peak_stored_support();

    // What is actually true, pinned: the direct route is dimension-capped
    // (≤ 2^n) and beat both lifts here; the lifts stayed within the
    // enlarged space; and the drift is real — this seeded instance pays
    // far more lifted than direct. If a change ever makes the lift beat
    // the direct run, this assertion failing is the *discovery*, not a
    // regression.
    assert!(direct_peak <= 1 << n);
    assert!(upfront_peak <= 1 << (n + t));
    assert!(jit_peak <= 1 << (n + t));
    assert!(
        upfront_peak > direct_peak && jit_peak > direct_peak,
        "measured drift: lifted ({upfront_peak}/{jit_peak}) vs direct ({direct_peak})"
    );
    // And the resource state alone is *linear* in t for a representation
    // with product structure — the composition target for frames over
    // factored inners.
    let mut factored = FactoredState::<C64>::new(16).unwrap();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let tg = reg.resolve("t").unwrap().matrix(&[]).unwrap();
    for a in 0..16 {
        factored.apply(&h, &[a]).unwrap();
        factored.apply(&tg, &[a]).unwrap();
    }
    let mem16 = factored.memory_bytes();
    let mut factored8 = FactoredState::<C64>::new(8).unwrap();
    for a in 0..8 {
        factored8.apply(&h, &[a]).unwrap();
        factored8.apply(&tg, &[a]).unwrap();
    }
    let mem8 = factored8.memory_bytes();
    assert!(
        mem16 < 3 * mem8,
        "|T⟩^⊗t memory must scale linearly in a factored representation: {mem8} → {mem16}"
    );
}

#[test]
fn lift_validates_width_and_malformed_ops() {
    // n + t must fit the sparse mask width.
    let mut c: Circuit = Circuit::new(60);
    c.t(0).t(1).t(2).t(3).t(4);
    assert!(matches!(
        lift::to_clifford_feedback(&c, ResourcePrep::Upfront),
        Err(Error::TooManyQubits { .. })
    ));
    // A t with parameters is malformed and reported as such.
    let mut c: Circuit = Circuit::new(2);
    c.gate("t", vec![0.3], vec![0]);
    assert!(matches!(
        lift::to_clifford_feedback(&c, ResourcePrep::JustInTime),
        Err(Error::ParamCountMismatch { .. })
    ));
}
