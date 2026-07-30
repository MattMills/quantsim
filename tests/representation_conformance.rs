//! The two new representations put through the same harness every other
//! representation in this crate has to pass: registry-wide conformance
//! against the dense reference, and the boundary atlas pricing their
//! cost on families that hold their assumption and families that break
//! it.
//!
//! Nothing here is specific to the representations' own mathematics —
//! that lives in `tests/padic_interference.rs` and
//! `tests/braided_boundary.rs`. This file is the safety net.

mod common;

use quantsim::conformance::{verify_backend, ConformanceConfig};
use quantsim::prelude::*;

fn sim_with_new_backends() -> Simulator {
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("phase-field", |n| Ok(Box::new(PhaseFieldState::new(n)?)))
        .unwrap();
    sim.backends_mut()
        .register("braided", |n| Ok(Box::new(BraidedState::new(n)?)))
        .unwrap();
    sim
}

// ── the safety net: every registered gate, against dense ─────────────

#[test]
fn phase_field_conforms_over_the_whole_registry() {
    let sim = sim_with_new_backends();
    let report = verify_backend(&sim, "phase-field", &ConformanceConfig::default()).unwrap();
    assert!(
        report.failures.is_empty(),
        "conformance failures: {:?}",
        report.failures
    );
    assert_eq!(report.sampling_mismatches, 0);
    assert_eq!(report.collapse_violations, 0);
    assert!(report.random_circuit_cases >= 24);
    // The phase polynomial is integer arithmetic until an amplitude is
    // asked for, so agreement with dense is not merely within tolerance.
    assert_eq!(
        report.max_amplitude_deviation, 0.0,
        "phase-field must agree with dense exactly, got {:.3e}",
        report.max_amplitude_deviation
    );
    assert!(report.max_weight_drift < 1e-14);
    // and every gate in the registry was actually swept
    assert!(report.gate_checks.len() >= 30);
    for g in &report.gate_checks {
        assert_eq!(g.max_deviation, 0.0, "gate {} deviated", g.gate);
    }
}

#[test]
fn braided_conforms_over_the_whole_registry() {
    let sim = sim_with_new_backends();
    let report = verify_backend(&sim, "braided", &ConformanceConfig::default()).unwrap();
    assert!(
        report.failures.is_empty(),
        "conformance failures: {:?}",
        report.failures
    );
    assert_eq!(report.sampling_mismatches, 0);
    assert_eq!(report.collapse_violations, 0);
    assert_eq!(report.max_amplitude_deviation, 0.0);
}

// ── the families that hold each assumption, and the ones that break it ──

/// An IQP core: a Hadamard layer, then all-pairs `cz` and a `t` on every
/// qubit. Diagonal throughout after the first layer, so the phase-field
/// class holds.
fn iqp_core(n: usize) -> Circuit {
    let mut c: Circuit = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    for a in 0..n {
        for b in a + 1..n {
            c.cz(a, b);
        }
    }
    for q in 0..n {
        c.t(q);
    }
    c
}

/// The same, plus the closing Hadamard layer — the interference step,
/// which is exactly where the phase-field class ends.
fn iqp_full(n: usize) -> Circuit {
    let mut c = iqp_core(n);
    for q in 0..n {
        c.h(q);
    }
    c
}

/// A pure braid family: nothing but Majorana generators, at fixed depth.
fn braid_family(n: usize) -> Circuit {
    let gens = Realization::Majorana.generators(n).unwrap();
    let all: Vec<usize> = (0..n).collect();
    let mut c: Circuit = Circuit::new(n);
    for k in 0..40 {
        c.raw(
            format!("sigma{}", k % gens.len()),
            gens[k % gens.len()].clone(),
            all.clone(),
        );
    }
    c
}

fn axis<'a>(scan: &'a FamilyScan, name: &str) -> &'a AxisScan {
    scan.axes
        .iter()
        .find(|a| a.axis == name)
        .unwrap_or_else(|| panic!("axis {name} missing from the scan"))
}

#[test]
fn the_phase_field_axis_certifies_the_iqp_core_and_fails_on_the_full_circuit() {
    let core = advantage_scan("iqp-core", iqp_core, &[6, 8, 10, 12]);
    let pf = axis(&core, "phase-field");
    // polynomial in BOTH memory and wall-clock — the atlas's own bar
    assert!(
        matches!(
            pf.law,
            Some(Law::Polynomial { .. }) | Some(Law::Constant)
        ),
        "phase-field memory on the IQP core: {:?}",
        pf.law
    );
    assert!(
        matches!(
            pf.time_law,
            Some(Law::Polynomial { .. }) | Some(Law::Constant)
        ),
        "phase-field time on the IQP core: {:?}",
        pf.time_law
    );
    // so the verdict names it as a reason the family is classical
    match &core.verdict {
        Verdict::Classical { via } => assert!(
            via.iter().any(|v| v == "phase-field"),
            "phase-field should certify the IQP core, verdict via {via:?}"
        ),
        other => panic!("IQP core should be classical, got {other:?}"),
    }
    // and it is the cheapest axis on that family
    let dense = axis(&core, "dense");
    let pf_max = pf.costs.iter().flatten().max().unwrap();
    let dense_max = dense.costs.iter().flatten().max().unwrap();
    assert!(
        pf_max * 20 < *dense_max,
        "phase-field {pf_max} vs dense {dense_max}"
    );

    // The closing Hadamard layer is the class boundary, and the axis
    // reports it by going exponential rather than by quietly lying.
    let full = advantage_scan("iqp-full", iqp_full, &[6, 8, 10, 12]);
    let pf_full = axis(&full, "phase-field");
    assert!(
        matches!(pf_full.law, Some(Law::Exponential { .. })),
        "phase-field memory after the final H layer: {:?}",
        pf_full.law
    );
    if let Verdict::Classical { via } = &full.verdict {
        assert!(
            !via.iter().any(|v| v == "phase-field"),
            "phase-field must not certify the full IQP circuit"
        );
    }
}

#[test]
fn the_braided_axis_is_flat_in_memory_and_exponential_in_time() {
    let scan = advantage_scan("braid", braid_family, &[4, 5, 6, 7]);
    let br = axis(&scan, "braided");
    // the word is the storage, and it does not grow with the width
    assert!(
        matches!(br.law, Some(Law::Constant)),
        "braided memory on a braid family: {:?}",
        br.law
    );
    let costs: Vec<usize> = br.costs.iter().flatten().copied().collect();
    assert_eq!(costs.len(), 4, "braided hit a wall: {:?}", br.costs);
    assert!(
        costs.iter().all(|&c| c == costs[0]),
        "braided costs should be identical across widths: {costs:?}"
    );
    // but replaying the word to read an amplitude is exponential, so the
    // atlas must NOT certify it
    assert!(
        matches!(br.time_law, Some(Law::Exponential { .. })),
        "braided time on a braid family: {:?}",
        br.time_law
    );
    if let Verdict::Classical { via } = &scan.verdict {
        assert!(
            !via.iter().any(|v| v == "braided"),
            "braided must not certify: it is memory-flat but time-exponential"
        );
    }

    // the shared alphabet is exponential and is reported, not hidden
    let mut alphabet = Vec::new();
    for n in [4usize, 5, 6] {
        let s = BraidedState::new(n).unwrap();
        assert!(s.memory_bytes() < 200, "empty word should be tiny");
        alphabet.push(s.alphabet_bytes());
    }
    for w in alphabet.windows(2) {
        assert!(w[1] > 4 * w[0], "alphabet must grow ~4^n: {alphabet:?}");
    }
}

// ── the class boundaries, stated as behaviour ────────────────────────

#[test]
fn the_phase_field_class_is_exactly_where_it_says_it_is() {
    let reg = GateRegistry::<C64>::standard();
    let n = 8;

    // diagonal-with-root-of-unity entries over a freed subcube: in class
    let mut s = PhaseFieldState::new(n).unwrap();
    iqp_core(n).bind(&reg).unwrap().run(&mut s).unwrap();
    assert!(s.is_field());
    assert_eq!(s.escapes(), 0);
    assert_eq!(s.free_qubits(), Some(n));
    // n·(n+1)/2 monomials: the pair terms plus the single-qubit t terms
    assert_eq!(s.monomials(), Some(n * (n + 1) / 2));
    // the modulus grew by lcm to hold both the cz (order 2) and t (8)
    assert_eq!(s.modulus(), Some(8));

    // and it is exactly right, against dense
    let mut d = DenseState::<C64>::new(n).unwrap();
    iqp_core(n).bind(&reg).unwrap().run(&mut d).unwrap();
    let dev = max_amplitude_deviation(&s as &dyn Backend<C64>, &d as &dyn Backend<C64>);
    assert!(dev < 1e-14, "phase-field vs dense on the IQP core: {dev:.3e}");

    // one more Hadamard on an already-free qubit leaves the class
    let mut leave: Circuit = Circuit::new(n);
    leave.h(0);
    leave.bind(&reg).unwrap().run(&mut s).unwrap();
    assert!(!s.is_field());
    assert_eq!(s.escapes(), 1);
    assert_eq!(s.monomials(), None);

    // a non-root-of-unity rotation also leaves it, from a fresh state
    let mut r = PhaseFieldState::new(4).unwrap();
    let mut c: Circuit = Circuit::new(4);
    c.h(0).rz(0, 0.371);
    c.bind(&reg).unwrap().run(&mut r).unwrap();
    assert!(!r.is_field(), "an irrational rotation must leave the class");

    // an eighth-turn rotation does not
    let mut r8 = PhaseFieldState::new(4).unwrap();
    let mut c8: Circuit = Circuit::new(4);
    c8.h(0).rz(0, std::f64::consts::FRAC_PI_4);
    c8.bind(&reg).unwrap().run(&mut r8).unwrap();
    assert!(r8.is_field(), "an eighth-turn rotation must stay in class");
}

#[test]
fn the_braided_class_is_exactly_the_realization_generators() {
    let n = 3;
    let gens = Realization::Majorana.generators(n).unwrap();
    let all: Vec<usize> = (0..n).collect();
    let mut s = BraidedState::new(n).unwrap();
    assert_eq!(s.realization(), Realization::Majorana);

    for k in 0..100 {
        s.apply(&gens[k % gens.len()], &all).unwrap();
    }
    assert!(s.is_word());
    assert_eq!(s.word().unwrap().len(), 100);
    assert!(!s.is_materialized(), "no amplitude was asked for");

    // the word replays to exactly the dense evolution
    let mut d = DenseState::<C64>::new(n).unwrap();
    for k in 0..100 {
        d.apply(&gens[k % gens.len()], &all).unwrap();
    }
    let dev = max_amplitude_deviation(&s as &dyn Backend<C64>, &d as &dyn Backend<C64>);
    assert!(dev < 1e-12, "braided word vs dense: {dev:.3e}");
    assert!(s.is_materialized(), "reading amplitudes materializes");

    // dropping the cache returns to word-only storage
    s.forget_amplitudes();
    assert!(!s.is_materialized());

    // anything outside the generator set leaves the class
    let mut t = BraidedState::new(n).unwrap();
    let reg = GateRegistry::<C64>::standard();
    let mut c: Circuit = Circuit::new(n);
    c.h(0);
    c.bind(&reg).unwrap().run(&mut t).unwrap();
    assert!(!t.is_word());
    assert_eq!(t.escapes(), 1);

    // the Fibonacci realization exists only at its native width
    assert!(BraidedState::with_realization(1, Realization::Fibonacci).is_ok());
    assert!(BraidedState::with_realization(3, Realization::Fibonacci).is_err());
    assert!(BraidedState::new(0).is_err());
}

// ── both representations compete in the ordinary selection machinery ──

#[test]
fn the_new_representations_take_part_in_backend_selection() {
    let sim = sim_with_new_backends();
    let names = sim.backends().names();
    assert!(names.iter().any(|n| n == "phase-field"));
    assert!(names.iter().any(|n| n == "braided"));

    // and a benchmark run reproduces the reference on both
    // width 8 = 16 Majorana strands, the widest the realization builds
    let workload = Workload::new("iqp-core-8", || iqp_core(8));
    let report = compare_backends(
        &sim,
        &[workload],
        &["dense", "phase-field", "braided"],
        &BenchConfig::default(),
    )
    .unwrap();
    assert_eq!(report.records.len(), 3);
    for r in &report.records {
        assert!(r.error.is_none(), "{} errored: {:?}", r.backend, r.error);
        if let Some(dev) = r.deviation {
            assert!(dev < 1e-12, "{} deviated by {dev:.3e}", r.backend);
        }
    }
    // and the phase-field record is by far the cheapest of the three
    let pf = report.record("iqp-core-8", "phase-field").unwrap();
    let de = report.record("iqp-core-8", "dense").unwrap();
    assert!(
        pf.memory_bytes * 4 < de.memory_bytes,
        "phase-field {} vs dense {}",
        pf.memory_bytes,
        de.memory_bytes
    );

    // Past its structural width the braided realization refuses, and the
    // harness records the refusal instead of a wrong answer — the same
    // contract every other representation's wall has.
    let wide = Workload::new("iqp-core-10", || iqp_core(10));
    let wide_report = compare_backends(
        &sim,
        &[wide],
        &["dense", "phase-field", "braided"],
        &BenchConfig::default(),
    )
    .unwrap();
    let br = wide_report.record("iqp-core-10", "braided").unwrap();
    assert!(
        br.error.is_some(),
        "braided should refuse 20 strands, not answer"
    );
    assert!(br.error.as_ref().unwrap().contains("strand"));
    // while the others answer normally
    for name in ["dense", "phase-field"] {
        let r = wide_report.record("iqp-core-10", name).unwrap();
        assert!(r.error.is_none(), "{name} errored: {:?}", r.error);
    }
}
