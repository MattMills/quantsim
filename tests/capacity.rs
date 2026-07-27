//! The resource guard: over-scale inhibition as a *library* property,
//! measured — not presumed by width constants, not delegated to callers.
//!
//! Every test here manipulates the process-global guard configuration,
//! so they serialize on one mutex (other test binaries are separate
//! processes and unaffected). Machine-independent determinism comes from
//! either an explicit configured limit or requests so far past any real
//! machine (hundreds of TiB) that measurement always refuses them.

mod common;

use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use common::assert_close;
use quantsim::backend::{MeraConfig, MeraState};
use quantsim::exact::ExactState;
use quantsim::prelude::*;

/// Serialize guard-configuration tests; restore defaults on drop.
struct GuardLock(#[allow(dead_code)] MutexGuard<'static, ()>);

fn lock() -> GuardLock {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let guard = match LOCK.get_or_init(|| Mutex::new(())).lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard::set_memory_limit(None);
    guard::set_time_budget(None);
    GuardLock(guard)
}

impl Drop for GuardLock {
    fn drop(&mut self) {
        guard::set_memory_limit(None);
        guard::set_time_budget(None);
    }
}

#[test]
fn over_scale_allocations_are_refused_by_measurement_not_presumption() {
    let _lock = lock();
    // No explicit limit: the guard measures the machine. 256 TiB of
    // dense amplitudes cannot be admitted anywhere real, and the error
    // carries both sides of the measurement.
    for (what, result) in [
        ("dense", DenseState::<C64>::new(44).err()),
        ("exact", ExactState::new(42).err()),
    ] {
        match result {
            Some(Error::OutOfMemory {
                requested,
                available,
                ..
            }) => {
                assert!(requested > available, "{what}: {requested} vs {available}");
                assert!(available > 0, "{what}: availability is measured, not zero");
            }
            other => panic!("{what}: expected measured OutOfMemory, got {other:?}"),
        }
    }
    // The structural bound (u64 indexing) is separate and still explicit.
    assert!(matches!(
        DenseState::<C64>::new(64),
        Err(Error::TooManyQubits { .. })
    ));
}

#[test]
fn explicit_limits_inhibit_exactly_and_release() {
    let _lock = lock();
    // A 1 MiB budget refuses a 16 MiB dense state with the configured
    // numbers…
    guard::set_memory_limit(Some(1 << 20));
    match DenseState::<C64>::new(20) {
        Err(Error::OutOfMemory {
            requested,
            available,
            ..
        }) => {
            assert_eq!(requested, 16 << 20);
            assert_eq!(available, 1 << 20);
        }
        other => panic!("expected OutOfMemory, got {other:?}"),
    }
    // …and the same width works the moment the limit lifts.
    guard::set_memory_limit(None);
    let state = DenseState::<C64>::new(20).unwrap();
    assert_eq!(state.num_qubits(), 20);
}

#[test]
fn adaptive_promotion_is_capacity_aware() {
    let _lock = lock();
    // Under a tight budget the dense vector is inadmissible, so the
    // adaptive backend must stay sparse INSTEAD of failing — same
    // circuit, same physics, policy derived from measured capacity.
    // H on 16 of 18 qubits puts the support exactly at the ¼-density
    // promotion threshold (2^16 of 2^18) while the sparse map itself
    // (~2 MiB, transients included) stays admissible under a 2.5 MiB
    // budget; the 4 MiB dense vector does not.
    guard::set_memory_limit(Some(5 << 19));
    let n = 18;
    let mut c: Circuit = Circuit::new(n);
    for q in 0..16 {
        c.h(q);
    }
    let sim: Simulator = Simulator::new();
    let state = sim.run_on("adaptive", &c).unwrap();
    let adaptive = state.as_any().downcast_ref::<AdaptiveState<C64>>().unwrap();
    assert!(
        !adaptive.is_dense(),
        "inadmissible dense vector: adaptive must stay sparse"
    );
    assert_close(state.probability(0), 1.0 / (1u64 << 16) as f64, 1e-12);

    guard::set_memory_limit(None);
    let promoted = sim.run_on("adaptive", &c).unwrap();
    let adaptive = promoted
        .as_any()
        .downcast_ref::<AdaptiveState<C64>>()
        .unwrap();
    assert!(adaptive.is_dense(), "with room, the same state promotes");
}

#[test]
fn sparse_growth_is_admitted_against_the_budget() {
    let _lock = lock();
    guard::set_memory_limit(Some(1 << 20));
    let n = 20;
    let mut c: Circuit = Circuit::new(n);
    for q in 0..n {
        c.h(q); // would fill 2^20 map entries ≈ 29 MiB
    }
    let sim: Simulator = Simulator::new();
    match sim.run_on("sparse", &c).err() {
        Some(Error::OutOfMemory { what, .. }) => {
            assert!(what.contains("sparse"), "{what}");
        }
        other => panic!("expected sparse growth refusal, got {other:?}"),
    }
}

#[test]
fn mera_blocks_and_factored_merges_are_admitted_not_presumed() {
    let _lock = lock();
    // A register-spanning gate at width 44 needs a 2^44-element block /
    // merged factor: refused by measurement on any real machine, with
    // the requested bytes in the error.
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let cx = reg.resolve("cx").unwrap().matrix(&[]).unwrap();

    let mut mera = MeraState::<C64>::with_config(44, MeraConfig::default()).unwrap();
    mera.apply(&h, &[0]).unwrap();
    assert!(matches!(
        mera.apply(&cx, &[0, 43]),
        Err(Error::OutOfMemory { .. })
    ));

    // The explicit policy cap still exists for callers who want refusal
    // below the machine's real limit — as configuration, not presumption.
    let mut capped = MeraState::<C64>::with_config(
        44,
        MeraConfig {
            max_block: 24,
            ..MeraConfig::default()
        },
    )
    .unwrap();
    capped.apply(&h, &[0]).unwrap();
    assert!(matches!(
        capped.apply(&cx, &[0, 43]),
        Err(Error::TooManyQubits {
            requested: 44,
            max: 24
        })
    ));
}

#[test]
fn time_budget_aborts_mid_gate_with_measured_elapsed() {
    let _lock = lock();
    // A single dense gate at width 24 sweeps 16M amplitude pairs — far
    // more than a 5 ms budget. The abort must come from INSIDE the gate
    // (the checkpointed kernel), with elapsed ≥ budget measured.
    let n = 24;
    let mut c: Circuit = Circuit::new(n);
    c.h(0).h(1).h(2);
    let reg = GateRegistry::<C64>::standard();
    let bound = c.bind(&reg).unwrap();
    let mut state = DenseState::<C64>::new(n).unwrap();

    guard::set_time_budget(Some(Duration::from_millis(5)));
    let started = std::time::Instant::now();
    match bound.run(&mut state) {
        Err(Error::Timeout {
            budget_ms,
            elapsed_ms,
        }) => {
            assert_eq!(budget_ms, 5);
            assert!(elapsed_ms >= budget_ms);
        }
        other => panic!("expected Timeout, got {other:?}"),
    }
    // The abort happened promptly — checkpoints fire within the sweep,
    // not after the whole run would have finished.
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "abort must be prompt: {:?}",
        started.elapsed()
    );

    // With the budget lifted the identical run completes.
    guard::set_time_budget(None);
    let mut fresh = DenseState::<C64>::new(n).unwrap();
    bound.run(&mut fresh).unwrap();
    assert_close(fresh.probability(0), 1.0 / 8.0, 1e-12);
}

#[test]
fn time_budget_bounds_svd_and_schedule_runs_too() {
    let _lock = lock();
    let sim: Simulator = Simulator::new();

    // The mera root-crossing GHZ chain at width 22 spends its time in
    // Jacobi SVD sweeps; the checkpoint inside the sweep loop must abort
    // it long before completion.
    guard::set_time_budget(Some(Duration::from_millis(50)));
    let started = std::time::Instant::now();
    match sim.run_on("mera", &library::ghz(22)).err() {
        Some(Error::Timeout { elapsed_ms, .. }) => assert!(elapsed_ms >= 50),
        other => panic!("expected Timeout from the SVD path, got {other:?}"),
    }
    assert!(started.elapsed() < Duration::from_secs(10));

    // Scheduled runs inherit the same scope through their event loop.
    let mut schedule: Schedule = Schedule::new(24, 10);
    schedule.at(0, "h", [], [0]);
    schedule.at(1, "h", [], [1]);
    schedule.at(2, "h", [], [2]);
    guard::set_time_budget(Some(Duration::from_millis(5)));
    match schedule.run(&sim, 7).err() {
        Some(Error::Timeout { .. }) => {}
        other => panic!("expected Timeout from the schedule, got {other:?}"),
    }
    guard::set_time_budget(None);
}

#[test]
fn a_timed_out_state_is_reported_torn() {
    let _lock = lock();
    // The documented contract: a Timeout aborts mid-operation, so the
    // state is torn exactly like after any failed apply — the norm need
    // not survive. What MUST hold: the error surfaces, and a fresh state
    // is unaffected.
    let n = 24;
    let mut c: Circuit = Circuit::new(n);
    c.h(0);
    let reg = GateRegistry::<C64>::standard();
    let bound = c.bind(&reg).unwrap();
    let mut state = DenseState::<C64>::new(n).unwrap();
    guard::set_time_budget(Some(Duration::from_millis(2)));
    assert!(matches!(bound.run(&mut state), Err(Error::Timeout { .. })));
    guard::set_time_budget(None);
    let mut fresh = DenseState::<C64>::new(n).unwrap();
    bound.run(&mut fresh).unwrap();
    assert_close(fresh.probability(0), 0.5, 1e-12);
}
