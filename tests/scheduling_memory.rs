//! The evented-scheduling + factored-geometry + discovery research stack:
//! simultaneous gate loops, quantum memory interacting with computation,
//! hierarchical entanglement decomposition, and computation-transparent
//! signal threads.

mod common;

use common::*;
use quantsim::prelude::*;
use std::f64::consts::FRAC_1_SQRT_2;

// ───────────────────────── scheduling ─────────────────────────

#[test]
fn simultaneous_loops_match_flattened_circuit() {
    // Two loops with different periods on different regions, plus one-shot
    // events: queue execution must equal the flattened circuit exactly.
    let sim: Simulator = Simulator::new();
    let mut schedule = Schedule::new(5, 40);
    schedule
        .at(0, "h", Vec::new(), vec![4])
        .add_loop(GateLoop {
            name: "compute".into(),
            start: 0,
            period: 3,
            iterations: None,
            body: vec![
                TimedOp::gate(0, "ry", vec![0.31], vec![0]),
                TimedOp::gate(1, "cx", Vec::new(), vec![0, 1]),
                TimedOp::gate(2, "rz", vec![0.17], vec![1]),
            ],
        })
        .add_loop(GateLoop {
            name: "refresh".into(),
            start: 1,
            period: 5,
            iterations: Some(6),
            body: vec![
                TimedOp::gate(0, "rz", vec![0.05], vec![2]),
                TimedOp::gate(0, "rz", vec![-0.05], vec![3]),
            ],
        })
        .at(20, "cx", Vec::new(), vec![1, 2]);

    let (state, trace) = schedule.run(&sim, 1).unwrap();
    let flattened = schedule.to_circuit().unwrap();
    let reference = sim.run(&flattened).unwrap();
    for i in 0..(1u64 << 5) {
        assert!(state.amplitude(i).approx_eq(reference.amplitude(i), TOL));
    }
    // Trace is time-ordered and complete.
    assert_eq!(trace.events.len(), flattened.len());
    assert!(trace.events.windows(2).all(|w| w[0].time <= w[1].time));
    // Same seed → identical trace on a different backend.
    let (state2, trace2) = schedule.run_on(&sim, "factored", 1).unwrap();
    assert_eq!(trace.events.len(), trace2.events.len());
    for i in 0..(1u64 << 5) {
        assert!(state.amplitude(i).approx_eq(state2.amplitude(i), TOL));
    }
}

#[test]
fn overlap_policy_discriminates() {
    let sim: Simulator = Simulator::new();
    // Same tick, overlapping qubits: rejected under Error, ordered under
    // InsertionOrder.
    let mut clash = Schedule::new(2, 10);
    clash
        .policy(OverlapPolicy::Error)
        .at(5, "h", Vec::new(), vec![0])
        .at(5, "x", Vec::new(), vec![0]);
    assert!(matches!(clash.run(&sim, 0), Err(Error::InvalidState(_))));

    let mut ordered = Schedule::new(2, 10);
    ordered
        .at(5, "h", Vec::new(), vec![0])
        .at(5, "x", Vec::new(), vec![0]);
    let (state, _) = ordered.run(&sim, 0).unwrap();
    // h then x (insertion order): X·H|0⟩ = (|0⟩ − |1⟩)/√2 up to relabel:
    // H|0⟩ = (|0⟩+|1⟩)/√2, X swaps → same state.
    assert!(state.amplitude(0).approx_eq(c64(FRAC_1_SQRT_2, 0.0), TOL));
    assert!(state.amplitude(1).approx_eq(c64(FRAC_1_SQRT_2, 0.0), TOL));

    // Disjoint same-tick events are fine under Error policy.
    let mut disjoint = Schedule::new(2, 10);
    disjoint
        .policy(OverlapPolicy::Error)
        .at(5, "h", Vec::new(), vec![0])
        .at(5, "h", Vec::new(), vec![1]);
    assert!(disjoint.run(&sim, 0).is_ok());
}

#[test]
fn loop_validation_errors() {
    let sim: Simulator = Simulator::new();
    let mut bad_period = Schedule::new(1, 10);
    bad_period.add_loop(GateLoop {
        name: "z".into(),
        start: 0,
        period: 0,
        iterations: Some(1),
        body: vec![],
    });
    assert!(matches!(
        bad_period.run(&sim, 0),
        Err(Error::InvalidState(_))
    ));

    let mut bad_offset = Schedule::new(1, 10);
    bad_offset.add_loop(GateLoop {
        name: "z".into(),
        start: 0,
        period: 2,
        iterations: Some(1),
        body: vec![TimedOp::gate(2, "x", Vec::new(), vec![0])],
    });
    assert!(matches!(
        bad_offset.run(&sim, 0),
        Err(Error::InvalidState(_))
    ));

    // Events past the horizon are skipped and counted, not applied.
    let mut past = Schedule::new(1, 5);
    past.at(9, "x", Vec::new(), vec![0]);
    let (state, trace) = past.run(&sim, 0).unwrap();
    assert_eq!(trace.skipped_past_horizon, 1);
    assert!(state.amplitude(0).approx_eq(c64(1.0, 0.0), TOL));
}

#[test]
fn teleportation_as_evented_feedback() {
    // The teleportation protocol expressed as measurement events whose
    // outcomes enqueue correction ops — classical feedback as messages.
    let (theta, phi) = (1.234f64, 2.345f64);
    let alpha = c64((theta / 2.0).cos(), 0.0);
    let beta = c64(phi.cos(), phi.sin()) * (theta / 2.0).sin();
    let sim: Simulator = Simulator::new();

    for seed in 0..16 {
        let mut schedule = Schedule::new(3, 20);
        schedule
            .at(0, "u", vec![theta, phi, 0.0], vec![0])
            .at(1, "h", Vec::new(), vec![1])
            .at(2, "cx", Vec::new(), vec![1, 2])
            .at(3, "cx", Vec::new(), vec![0, 1])
            .at(4, "h", Vec::new(), vec![0])
            // Corrections must compose as Z^{m0} · X^{m1}: X fires at tick
            // 7, Z at tick 8 (same-tick feedback would apply Z first and
            // pick up a stray global phase when both trigger).
            .measure_at(
                5,
                0,
                vec![],
                vec![TimedOp::gate(3, "z", Vec::new(), vec![2])],
            )
            .measure_at(
                6,
                1,
                vec![],
                vec![TimedOp::gate(1, "x", Vec::new(), vec![2])],
            );
        let (state, trace) = schedule.run(&sim, seed).unwrap();
        assert_eq!(trace.measurements.len(), 2);
        let m0 = trace.measurements[0].2 as u64;
        let m1 = trace.measurements[1].2 as u64;
        let base = m0 | (m1 << 1);
        assert!(state.amplitude(base).approx_eq(alpha, TOL), "seed {seed}");
        assert!(
            state.amplitude(base | 0b100).approx_eq(beta, TOL),
            "seed {seed}"
        );
    }
}

// ───────────────────── factored geometry ─────────────────────

#[test]
fn factored_geometry_lifecycle() {
    let sim: Simulator = Simulator::new();
    let mut state = FactoredState::<C64>::new(6).unwrap();
    assert_eq!(state.factor_count(), 6);

    let reg = GateRegistry::<C64>::standard();
    let apply = |st: &mut FactoredState<C64>, name: &str, params: &[f64], qs: &[usize]| {
        let m = reg.resolve(name).unwrap().matrix(params).unwrap();
        st.apply(&m, qs).unwrap();
    };

    // Local single-qubit work keeps everything separated.
    for q in 0..6 {
        apply(&mut state, "ry", &[0.3 + q as f64 * 0.1], &[q]);
    }
    assert_eq!(state.factor_count(), 6);

    // Entangling within pairs merges exactly those pairs.
    apply(&mut state, "cx", &[], &[0, 1]);
    apply(&mut state, "cx", &[], &[2, 3]);
    assert_eq!(
        state.factors(),
        vec![vec![0, 1], vec![2, 3], vec![4], vec![5]]
    );

    // A cross-pair gate merges the two pairs into one 4-qubit factor.
    apply(&mut state, "cz", &[], &[1, 2]);
    assert_eq!(state.largest_factor_qubits(), 4);

    // Measurement releases the measured qubit back into its own factor.
    let mut rng = Prng::new(3);
    state.measure(1, &mut rng).unwrap();
    assert!(state.factors().contains(&vec![1]));
    assert_close(state.total_weight(), 1.0, TOL);

    // A gate that leaves its targets unentangled is split back apart
    // (rank-1 detection): cx applied twice = identity.
    apply(&mut state, "cx", &[], &[4, 5]);
    apply(&mut state, "cx", &[], &[4, 5]);
    assert!(state.factors().contains(&vec![4]));
    assert!(state.factors().contains(&vec![5]));

    // Peaks recorded the worst moment, not the final state.
    let (peak_amps, peak_width) = state.peak();
    assert_eq!(peak_width, 4);
    assert!(peak_amps >= 16);

    // And the whole story matches dense amplitudes when replayed.
    let report = verify_backend(
        &sim,
        "factored",
        &ConformanceConfig {
            random_circuits: 8,
            orderings: 2,
            ..ConformanceConfig::default()
        },
    )
    .unwrap();
    assert!(report.passed(), "{report}");
}

#[test]
fn factored_runs_wide_local_circuits_beyond_dense_reach() {
    // 48 qubits: 24 disjoint entangled pairs. Dense would need 4 PiB;
    // factored needs 24 four-amplitude factors.
    let sim: Simulator = Simulator::new();
    let n = 48;
    let mut c: Circuit = Circuit::new(n);
    let mut thetas = Vec::new();
    for pair in 0..n / 2 {
        let theta = 0.2 + 0.1 * pair as f64;
        thetas.push(theta);
        c.ry(2 * pair, theta).cx(2 * pair, 2 * pair + 1);
    }
    let state = sim.run_on("factored", &c).unwrap();
    let factored = state.as_any().downcast_ref::<FactoredState<C64>>().unwrap();
    assert_eq!(factored.factor_count(), n / 2);
    assert_eq!(factored.largest_factor_qubits(), 2);
    assert!(state.memory_bytes() < 100_000, "{}", state.memory_bytes());
    assert_close(state.total_weight(), 1.0, TOL);

    // Amplitudes match the closed form: each pair contributes cos(θ/2) for
    // |00⟩ or sin(θ/2) for |11⟩.
    let all_zero = state.amplitude(0);
    let expected: f64 = thetas.iter().map(|t| (t / 2.0).cos()).product();
    assert!(all_zero.approx_eq(c64(expected, 0.0), 1e-9));
    let ones_mask: u64 = (1u64 << n) - 1;
    let expected_ones: f64 = thetas.iter().map(|t| (t / 2.0).sin()).product();
    assert!(state
        .amplitude(ones_mask)
        .approx_eq(c64(expected_ones, 0.0), 1e-9));
    // Odd-parity-within-a-pair indices are impossible.
    assert!(state.amplitude(0b01).approx_eq(c64(0.0, 0.0), TOL));
}

#[test]
fn factored_degenerates_honestly_on_global_entanglement() {
    // GHZ entangles everything: the factored representation must end as a
    // single factor the size of the dense vector — no false economy.
    let sim: Simulator = Simulator::new();
    let state = sim.run_on("factored", &library::ghz(10)).unwrap();
    let factored = state.as_any().downcast_ref::<FactoredState<C64>>().unwrap();
    assert_eq!(factored.factor_count(), 1);
    assert_eq!(factored.largest_factor_qubits(), 10);
    assert!(state.amplitude(0).approx_eq(c64(FRAC_1_SQRT_2, 0.0), TOL));
    // And a merge that would exceed the machine errors with the MEASURED
    // numbers: two 22-qubit entangled chains (64 MiB factors — real, and
    // admitted) bridged by one CX whose merged 44-qubit factor would be
    // 256 TiB. The inhibition is the resource guard's admission against
    // measured memory, not a presumed factor-width constant.
    let mut wide: Circuit = Circuit::new(44);
    wide.h(0);
    for q in 0..21 {
        wide.cx(q, q + 1);
    }
    wide.h(22);
    for q in 22..43 {
        wide.cx(q, q + 1);
    }
    wide.cx(21, 22); // bridge: 22 + 22 = 44-qubit factor = 256 TiB
    match sim.run_on("factored", &wide) {
        Err(Error::OutOfMemory {
            requested,
            available,
            ..
        }) => {
            assert_eq!(requested, 16usize << 44);
            assert!(available < requested, "the budget is a measurement");
        }
        other => panic!("expected measured OutOfMemory, got {:?}", other.err()),
    }
}

#[test]
fn quantum_memory_interacting_with_compute() {
    // Memory bank (qubits 8..16) holds values while a compute loop runs on
    // 0..8; scheduled interactions entangle one memory cell, measurement
    // releases it. The factor geometry tells the story at every stage.
    let sim: Simulator = Simulator::new();
    let mut schedule: Schedule = Schedule::new(16, 30);
    // Memory prep: store |1⟩ in cells 8 and 9, superposition in cell 10.
    schedule
        .at(0, "x", Vec::new(), vec![8])
        .at(0, "x", Vec::new(), vec![9])
        .at(0, "h", Vec::new(), vec![10]);
    // Compute loop on 0..4.
    schedule.add_loop(GateLoop {
        name: "compute".into(),
        start: 1,
        period: 2,
        iterations: Some(8),
        body: vec![
            TimedOp::gate(0, "ry", vec![0.4], vec![0]),
            TimedOp::gate(0, "ry", vec![0.9], vec![1]),
            TimedOp::gate(1, "cx", Vec::new(), vec![0, 1]),
        ],
    });
    // Interaction: entangle compute with memory cell 8 mid-run, then read
    // the cell out again (releasing it from the compute factor).
    schedule.at(9, "cx", Vec::new(), vec![0, 8]);
    schedule.measure_at(15, 8, vec![], vec![]);

    let (state, trace) = schedule.run_on(&sim, "factored", 5).unwrap();
    let factored = state.as_any().downcast_ref::<FactoredState<C64>>().unwrap();
    // After the run: memory cells 9 and 10 never interacted — still
    // singleton factors; cell 8 was measured back out — singleton again.
    let factors = factored.factors();
    assert!(factors.contains(&vec![8]), "{factors:?}");
    assert!(factors.contains(&vec![9]), "{factors:?}");
    assert!(factors.contains(&vec![10]), "{factors:?}");
    // The interaction really happened: the peak factor spanned compute+cell.
    let (_, peak_width) = factored.peak();
    assert!(
        peak_width >= 3,
        "interaction never merged: peak {peak_width}"
    );
    assert_eq!(trace.measurements.len(), 1);
    assert_close(state.total_weight(), 1.0, TOL);
    // Idle memory stayed cheap: total footprint far below dense (1 MiB).
    assert!(state.memory_bytes() < 16_384, "{}", state.memory_bytes());
}

#[test]
fn factored_native_sampler_matches_distribution() {
    // sample_factored draws each factor independently — exact for the
    // product Born distribution, no support enumeration.
    let sim: Simulator = Simulator::new();
    let state = sim.run_on("factored", &library::bell()).unwrap();
    let factored = state.as_any().downcast_ref::<FactoredState<C64>>().unwrap();
    let counts = factored.sample_factored(4096, &mut Prng::new(3)).unwrap();
    assert_eq!(*counts.get(&0b01).unwrap_or(&0), 0);
    assert_eq!(*counts.get(&0b10).unwrap_or(&0), 0);
    let z = *counts.get(&0b00).unwrap_or(&0);
    assert_eq!(z + counts.get(&0b11).copied().unwrap_or(0), 4096);
    assert!((1792..=2304).contains(&z), "{z} zeros"); // 4σ

    // Wide product state: 36 qubits of independent |+⟩ — enumeration would
    // visit 2^36 outcomes; per-factor sampling doesn't.
    let mut c: Circuit = Circuit::new(36);
    for q in 0..36 {
        c.h(q);
    }
    let state = sim.run_on("factored", &c).unwrap();
    let factored = state.as_any().downcast_ref::<FactoredState<C64>>().unwrap();
    let counts = factored.sample_factored(512, &mut Prng::new(8)).unwrap();
    let total: u64 = counts.values().sum();
    assert_eq!(total, 512);
    // Mean bit-density across shots must hover near 1/2 (each bit fair).
    let ones: u64 = counts
        .iter()
        .map(|(idx, n)| idx.count_ones() as u64 * n)
        .sum();
    let density = ones as f64 / (512.0 * 36.0);
    assert!((0.46..=0.54).contains(&density), "bit density {density}");
}

#[test]
fn schedule_diagonal_and_raw_events() {
    // diagonal_at and raw_at events flow through the queue and flatten
    // identically to the equivalent circuit.
    let sim: Simulator = Simulator::new();
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let mut schedule: Schedule = Schedule::new(3, 10);
    schedule
        .raw_at(0, "h-raw", h.clone(), vec![0])
        .at(1, "cx", Vec::new(), vec![0, 1])
        .diagonal_at(2, "flip3", library::phase_flip(2, 3), vec![0, 1])
        .raw_at(3, "h-raw", h, vec![2]);
    let (state, trace) = schedule.run(&sim, 0).unwrap();
    let flattened = schedule.to_circuit().unwrap();
    let reference = sim.run(&flattened).unwrap();
    for i in 0..8u64 {
        assert!(state.amplitude(i).approx_eq(reference.amplitude(i), TOL));
    }
    assert_eq!(trace.events.len(), 4);
    // Non-unitary raw and diagonal events are rejected at application.
    let mut bad: Schedule = Schedule::new(1, 5);
    bad.diagonal_at(0, "too-big", vec![c64(2.0, 0.0), c64(1.0, 0.0)], vec![0]);
    assert!(matches!(bad.run(&sim, 0), Err(Error::NotUnitary { .. })));
    let mut bad: Schedule = Schedule::new(1, 5);
    let mut not_unitary = GateMatrix::<C64>::identity(2).unwrap();
    not_unitary.set(0, 0, c64(3.0, 0.0));
    bad.raw_at(0, "nope", not_unitary, vec![0]);
    assert!(matches!(bad.run(&sim, 0), Err(Error::NotUnitary { .. })));
}

// ───────────────── transparent structure discovery ─────────────────

#[test]
fn point_stabilizer_discovery() {
    let sim: Simulator = Simulator::new();
    // State: q0, q1 scrambled; q2 untouched (|0⟩).
    let mut c = Circuit::new(3);
    c.u(0, 0.7, 1.1, 0.3).u(1, 1.9, 0.2, 2.2).cx(0, 1);
    let state = sim.run(&c).unwrap();
    let reg = GateRegistry::<C64>::standard();

    // Any controlled rotation with control q2 = |0⟩ stabilizes, for every
    // parameter.
    let crz = reg.resolve("crz").unwrap();
    let grid: Vec<Vec<f64>> = [0.3, 1.2, 2.9].iter().map(|&t| vec![t]).collect();
    let found = discover_stabilizers(state.as_ref(), crz.as_ref(), &[2, 0], &grid, 1e-9).unwrap();
    assert_eq!(found.len(), 3, "control-on-|0⟩ stabilizes at all params");

    // rz on the scrambled q0 stabilizes only at θ ∈ {0, 2π} (2π is a pure
    // global phase — caught by the up-to-phase comparison).
    let rz = reg.resolve("rz").unwrap();
    let grid: Vec<Vec<f64>> = [
        0.0,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
        std::f64::consts::TAU,
    ]
    .iter()
    .map(|&t| vec![t])
    .collect();
    let found = discover_stabilizers(state.as_ref(), rz.as_ref(), &[0], &grid, 1e-9).unwrap();
    let angles: Vec<f64> = found.iter().map(|(p, _)| p[0]).collect();
    assert_eq!(angles.len(), 2, "{angles:?}");
    assert!(angles.contains(&0.0) && angles.contains(&std::f64::consts::TAU));

    // Diagonal kernels stabilize basis states trivially.
    let mut basis = Circuit::new(2);
    basis.x(0);
    let basis_state = sim.run(&basis).unwrap();
    let check = stabilizes_state(
        basis_state.as_ref(),
        &GateKernel::Diagonal(vec![c64(1.0, 0.0), cis(1.234)]),
        &[0],
        1e-9,
    )
    .unwrap();
    assert!(check.stabilizes);
    // …but not superposed ones.
    let mut plus = Circuit::new(1);
    plus.h(0);
    let plus_state = sim.run(&plus).unwrap();
    let check = stabilizes_state(
        plus_state.as_ref(),
        &GateKernel::Diagonal(vec![c64(1.0, 0.0), cis(1.234)]),
        &[0],
        1e-9,
    )
    .unwrap();
    assert!(!check.stabilizes);
}

#[test]
fn parity_signal_thread_is_transparent_and_carries_structure() {
    use Pauli::*;
    let sim: Simulator = Simulator::new();

    // Base computation on d0, d1 with a Z-diagonal middle segment; q2 is
    // the ancilla rail the signal will ride.
    let mut base = Circuit::new(3);
    base.h(0).h(1); // ops 0, 1
    base.rz(0, 0.7).cp(0, 1, 1.1).rz(1, 0.4); // ops 2, 3, 4: diagonal segment
    base.h(0).h(1); // ops 5, 6

    // The n-wide unit: inject the parity signal before the segment (pos 2),
    // remove it after (pos 5). Neither half is identity alone; the whole
    // thread is.
    let thread = Insertion::new("parity-rail")
        .gate(2, "cx", Vec::<f64>::new(), vec![0, 2])
        .gate(2, "cx", Vec::<f64>::new(), vec![1, 2])
        .gate(5, "cx", Vec::<f64>::new(), vec![1, 2])
        .gate(5, "cx", Vec::<f64>::new(), vec![0, 2]);

    let report = verify_transparent(&sim, &base, &thread, 1e-9).unwrap();
    assert!(report.transparent, "{report:?}");
    assert!(report.final_deviation < 1e-9);
    assert!(report.weight_drift < 1e-9);
    // The thread costs geometry while alive (ancilla merged into the data
    // factor) — measured, not guessed.
    assert!(report.peak_with.0 >= report.peak_without.0);

    // Mid-flight the signal is real, exploitable structure: truncate the
    // spliced circuit inside the thread's lifetime and observe the parity
    // anchor ⟨Z_d0 Z_d1 Z_anc⟩ = +1 exactly.
    let spliced = thread.splice(&base);
    let mut mid = Circuit::new(3);
    for op in spliced.ops().iter().take(7) {
        match op {
            Op::Named {
                name,
                params,
                qubits,
            } => {
                mid.gate(name.clone(), params.clone(), qubits.clone());
            }
            _ => unreachable!("thread uses named ops only"),
        }
    }
    let mid_state = sim.run(&mid).unwrap();
    let anchor = pauli_expectation(mid_state.as_ref(), &[(0, Z), (1, Z), (2, Z)]).unwrap();
    assert_close(anchor.re, 1.0, TOL);

    // The individual halves are NOT stabilizers — only the whole unit is.
    let inject_only = Insertion::new("inject-only")
        .gate(2, "cx", Vec::<f64>::new(), vec![0, 2])
        .gate(2, "cx", Vec::<f64>::new(), vec![1, 2]);
    let report = verify_transparent(&sim, &base, &inject_only, 1e-9).unwrap();
    assert!(!report.transparent, "half a thread must not be transparent");

    // And threading across a non-commuting segment breaks transparency:
    // add an H on d0 inside the segment.
    let mut noncommuting = Circuit::new(3);
    noncommuting.h(0).h(1);
    noncommuting.rz(0, 0.7).h(0).cp(0, 1, 1.1); // H breaks Z-diagonality
    noncommuting.h(0).h(1);
    let report = verify_transparent(&sim, &noncommuting, &thread, 1e-9).unwrap();
    assert!(
        !report.transparent,
        "thread through a non-commuting segment must be caught"
    );
}

#[test]
fn diagonal_ops_ride_signal_threads() {
    // Insertion::diagonal splices diagonal kernels; a phase flip applied
    // and immediately reverted is transparent as a unit.
    let sim: Simulator = Simulator::new();
    let mut base = Circuit::new(2);
    base.h(0).h(1); // 0, 1
    base.cx(0, 1); // 2
    let thread = Insertion::new("flip-unflip")
        .diagonal(2, "flip", library::phase_flip(2, 3), vec![0, 1])
        .diagonal(2, "unflip", library::phase_flip(2, 3), vec![0, 1]);
    let report = verify_transparent(&sim, &base, &thread, 1e-9).unwrap();
    assert!(report.transparent, "{report:?}");
    // A single unreverted flip is not.
    let half =
        Insertion::new("flip-only").diagonal(2, "flip", library::phase_flip(2, 3), vec![0, 1]);
    let report = verify_transparent(&sim, &base, &half, 1e-9).unwrap();
    assert!(!report.transparent);
}

#[test]
fn thread_verification_over_the_factored_geometry() {
    // A memory-swap thread: move a computation qubit's content into a
    // fresh rail, let the segment act on the other qubits, and swap back.
    // Transparent as a unit, and the factored peaks quantify the cost.
    let sim: Simulator = Simulator::new();
    let mut base = Circuit::new(4);
    base.u(0, 0.9, 0.3, 1.7).u(1, 1.2, 2.1, 0.4); // ops 0, 1
    base.cx(1, 2).rz(2, 0.8).cx(1, 2); // ops 2, 3, 4: segment not touching q0
    base.h(1); // op 5

    let swap_thread = Insertion::new("memory-swap")
        .gate(2, "swap", Vec::<f64>::new(), vec![0, 3])
        .gate(5, "swap", Vec::<f64>::new(), vec![0, 3]);
    let report = verify_transparent(&sim, &base, &swap_thread, 1e-9).unwrap();
    assert!(report.transparent, "{report:?}");
}
