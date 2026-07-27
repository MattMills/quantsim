//! Recursive measurement feedback: adaptive trees in the evented scheduler.
//!
//! A [`FeedbackOp`] branch may contain further measurements, so a schedule
//! expresses outcome-dependent measurement *sequences* — bounded
//! repeat-until-success ladders, conditional syndrome reads — with
//! structural termination (branches are finite trees). These tests pin the
//! semantics: nested events fire exactly when their parent outcome selects
//! them, replay is deterministic per seed, and the horizon prunes whole
//! subtrees.

mod common;

use common::{assert_close, sim};
use quantsim::prelude::*;

/// GHZ-3, then an adaptive chain: measure q0; only on outcome 1 measure
/// q1; only on that outcome 1 flip q2. GHZ correlations make the inner
/// branch deterministic once the outer fires.
#[test]
fn nested_measurement_fires_only_on_selecting_outcome() {
    let sim = sim();
    let mut seen = [false, false];
    for seed in 0..64u64 {
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
        let (state, trace) = schedule.run_on(&sim, "dense", seed).unwrap();
        let outer = trace.measurements[0].2;
        seen[outer as usize] = true;
        if outer {
            // GHZ: q1 must read 1, so the X on q2 must have fired.
            assert_eq!(trace.measurements.len(), 2, "seed {seed}: inner fired");
            assert!(trace.measurements[1].2, "seed {seed}: GHZ correlation");
            assert_eq!(trace.measurements[1].0, 11, "inner tick = outer + 1");
            assert!(
                trace
                    .events
                    .iter()
                    .any(|e| e.label == "x" && e.qubits == vec![2]),
                "seed {seed}: correction fired"
            );
            assert_close(state.probability(0b011), 1.0, 1e-9);
        } else {
            assert_eq!(trace.measurements.len(), 1, "seed {seed}: inner pruned");
            assert_close(state.probability(0b000), 1.0, 1e-9);
        }
        if seen[0] && seen[1] {
            return;
        }
    }
    panic!("64 seeds never produced both outer outcomes");
}

/// A bounded repeat-until-success ladder, depth 3: H then measure; outcome
/// 0 is success (post-state |0⟩), outcome 1 resets (X) and retries. The
/// trace must show exactly the retries the outcomes forced, and the final
/// state is |0⟩ on every path (the last rung gives up with a bare reset).
#[test]
fn bounded_repeat_until_success_ladder() {
    let sim = sim();
    // Innermost rung: no further retry — just reset to |0⟩ on failure.
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

    let mut attempts_seen = [false; 3];
    for seed in 0..256u64 {
        let mut schedule: Schedule = Schedule::new(1, 100);
        schedule.at(0, "h", [], [0]);
        schedule.measure_branching_at(1, 0, vec![], rung1.clone());
        let (state, trace) = schedule.run_on(&sim, "dense", seed).unwrap();

        let attempts = trace.measurements.len();
        assert!((1..=3).contains(&attempts), "seed {seed}");
        attempts_seen[attempts - 1] = true;
        // Every measurement before the last must have failed (outcome 1);
        // the last either succeeded or was the final give-up rung.
        for &(_, _, outcome) in &trace.measurements[..attempts - 1] {
            assert!(outcome, "seed {seed}: early rungs only fire on failure");
        }
        // Each failure enqueues exactly one x and (except the last rung)
        // one h retry.
        let xs = trace.events.iter().filter(|e| e.label == "x").count();
        let failures = trace
            .measurements
            .iter()
            .filter(|&&(_, _, o)| o)
            .count();
        assert_eq!(xs, failures, "seed {seed}: one reset per failure");
        // All paths end in |0⟩.
        assert_close(state.probability(0), 1.0, 1e-9);
    }
    assert!(
        attempts_seen.iter().all(|&s| s),
        "256 seeds should exercise 1, 2 and 3 attempts: {attempts_seen:?}"
    );
}

/// Same-tick feedback: branch ops share a tick and apply in branch order
/// (gate before nested measurement here), deterministically.
#[test]
fn same_tick_feedback_applies_in_branch_order() {
    let sim = sim();
    for seed in 0..16u64 {
        let mut schedule: Schedule = Schedule::new(2, 50);
        schedule.at(0, "x", [], [0]); // q0 = |1⟩ so the outer outcome is certain
        schedule.measure_branching_at(
            1,
            0,
            vec![],
            vec![
                // Both at relative offset 2 — the X must land before the
                // nested measurement reads q1.
                FeedbackOp::gate(2, "x", [], [1]),
                FeedbackOp::measure(2, 1, vec![], vec![]),
            ],
        );
        let (_, trace) = schedule.run_on(&sim, "dense", seed).unwrap();
        assert_eq!(trace.measurements.len(), 2);
        assert!(trace.measurements[0].2, "outer reads 1");
        assert!(
            trace.measurements[1].2,
            "seed {seed}: nested measure must see the same-tick X already applied"
        );
    }
}

/// The horizon prunes a nested measurement (and silently its whole
/// subtree), counted once in `skipped_past_horizon`.
#[test]
fn horizon_prunes_nested_subtrees() {
    let sim = sim();
    let mut schedule: Schedule = Schedule::new(1, 10);
    schedule.at(0, "x", [], [0]);
    schedule.measure_branching_at(
        1,
        0,
        vec![],
        vec![FeedbackOp::measure(
            100, // past the horizon
            0,
            vec![FeedbackOp::gate(1, "x", [], [0])],
            vec![FeedbackOp::gate(1, "x", [], [0])],
        )],
    );
    let (state, trace) = schedule.run_on(&sim, "dense", 3).unwrap();
    assert_eq!(trace.measurements.len(), 1, "nested measure never fired");
    assert_eq!(trace.skipped_past_horizon, 1);
    assert_close(state.probability(1), 1.0, 1e-9);
}

/// Replay determinism: identical (schedule, seed) pairs reproduce the
/// identical adaptive path, event for event, across backends.
#[test]
fn adaptive_replay_is_deterministic_across_backends() {
    let sim = sim();
    let build = || {
        let mut schedule: Schedule = Schedule::new(2, 60);
        schedule.at(0, "h", [], [0]);
        schedule.at(0, "h", [], [1]);
        schedule.measure_branching_at(
            5,
            0,
            vec![FeedbackOp::measure(
                1,
                1,
                vec![FeedbackOp::gate(1, "z", [], [1])],
                vec![FeedbackOp::gate(1, "x", [], [1])],
            )],
            vec![FeedbackOp::gate(1, "h", [], [1])],
        );
        schedule
    };
    for seed in 0..32u64 {
        let (sa, ta) = build().run_on(&sim, "dense", seed).unwrap();
        let (sb, tb) = build().run_on(&sim, "sparse", seed).unwrap();
        assert_eq!(ta.measurements, tb.measurements, "seed {seed}");
        assert_eq!(ta.events.len(), tb.events.len(), "seed {seed}");
        for (ea, eb) in ta.events.iter().zip(&tb.events) {
            assert_eq!((ea.time, &ea.label, &ea.qubits), (eb.time, &eb.label, &eb.qubits));
        }
        for i in 0..4u64 {
            let (x, y) = (sa.amplitude(i), sb.amplitude(i));
            assert!(x.approx_eq(y, 1e-9), "seed {seed}, amp {i}: {x} vs {y}");
        }
    }
}
