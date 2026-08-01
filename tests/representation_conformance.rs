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
    // Not exactly 0.0 like phase-field: braided now evolves through the
    // Clifford frame's tableau + sparse arithmetic rather than replaying
    // exact matrices, so it carries the frame's numerics.
    assert!(
        report.max_amplitude_deviation < 1e-14,
        "braided deviation {:.3e}",
        report.max_amplitude_deviation
    );
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
/// Each is a weight-≤2 local Clifford gate, at any width.
fn braid_family(n: usize) -> Circuit {
    let count = Realization::Majorana.generator_count(n);
    let mut c: Circuit = Circuit::new(n);
    for k in 0..40 {
        let (g, t) = Realization::Majorana.local_gate(k % count, n).unwrap();
        c.raw(format!("sigma{}", k % count), g, t);
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
    let core = advantage_scan("iqp-core", iqp_core, &[5, 6, 8, 10]);
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
    // The wall-clock law is deliberately NOT asserted, and the reason is
    // a resolution limit rather than jitter: over a sweep this short a
    // steep polynomial and a shallow exponential (measured base 1.36) are
    // numerically adjacent, and which label the fit picks moves with
    // machine load. Asserted instead as measured growth, which does not.
    // The verdict — which needs BOTH ledgers sub-exponential, and so
    // inherits the timing fit's fragility — is demonstrated at full width
    // by `examples/advantage_bounds.rs` rather than pinned here.
    let growth = pf
        .measured_time_growth()
        .expect("phase-field finished every probe width");
    let dense_growth = axis(&core, "dense")
        .measured_time_growth()
        .expect("dense finished every probe width");
    assert!(
        growth < dense_growth,
        "phase-field time grew {growth:.2}x against dense's {dense_growth:.2}x"
    );
    // and it is the cheapest axis on that family
    let dense = axis(&core, "dense");
    let pf_max = pf.costs.iter().flatten().max().unwrap();
    let dense_max = dense.costs.iter().flatten().max().unwrap();
    // As above: the constant tracks the sweep's top width, the law
    // separation is the claim.
    assert!(
        pf_max * 5 < *dense_max,
        "phase-field {pf_max} vs dense {dense_max}"
    );

    // The closing Hadamard layer is the class boundary, and the axis
    // reports it by going exponential rather than by quietly lying.
    let full = advantage_scan("iqp-full", iqp_full, &[5, 6, 8, 10]);
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
fn the_braided_axis_certifies_a_braid_family_at_every_width() {
    // The generators are weight-≤2 Clifford rotations, so nothing here
    // builds a 2^n object. The sweep runs at the smallest widths that
    // still resolve a law — the point is the *shape*, and dense has to
    // be run at every probe width for the comparison, which is what
    // costs. The wide demonstration is `examples/braided_boundary.rs`.
    let scan = advantage_scan("braid", braid_family, &[6, 8, 10, 12]);
    let br = axis(&scan, "braided");

    assert!(
        matches!(br.law, Some(Law::Constant) | Some(Law::Polynomial { .. })),
        "braided memory on a braid family: {:?}",
        br.law
    );
    // As in the phase-field test: the wall-clock law, and therefore the
    // verdict that depends on it, is a timing fit over a short sweep and
    // moves with machine load. The robust statement is the measured
    // growth against dense on the same family.
    let growth = br
        .measured_time_growth()
        .expect("braided finished every probe width");
    let dense_growth = axis(&scan, "dense")
        .measured_time_growth()
        .expect("dense finished every probe width");
    assert!(
        growth < dense_growth,
        "braided time grew {growth:.2}x against dense's {dense_growth:.2}x"
    );

    // and it is vastly below dense, which is exponential on the same family
    let dense = axis(&scan, "dense");
    assert!(matches!(dense.law, Some(Law::Exponential { .. })));
    let br_max = br.costs.iter().flatten().max().unwrap();
    let dense_max = dense.costs.iter().flatten().max().unwrap();
    // The gap itself grows with width, so the constant here is a
    // function of the sweep and not a property of the representation —
    // the property is the law separation asserted above. This pins the
    // gap's direction and order, not its size.
    assert!(
        br_max * 4 < *dense_max,
        "braided {br_max} vs dense {dense_max} at the widest probe"
    );
}

#[test]
fn the_braided_footprint_is_flat_in_width_and_the_support_stays_one() {
    let run = |qubits: usize, steps: usize| {
        let mut s = BraidedState::new(qubits).unwrap();
        let count = Realization::Majorana.generator_count(qubits);
        for k in 0..steps {
            let (g, t) = Realization::Majorana.local_gate(k % count, qubits).unwrap();
            s.apply(&g, &t).unwrap();
        }
        s
    };
    // A 7.9× wider register for a few percent more memory, with the
    // stored support pinned at 1 and no flush anywhere.
    let narrow = run(8, 200);
    let wide = run(63, 200);
    for s in [&narrow, &wide] {
        assert_eq!(s.stored_support(), 1);
        assert_eq!(s.frame_stats().flushes, 0);
        assert!(s.is_pure_braid());
    }
    assert!(
        (wide.memory_bytes() as f64) < 1.3 * narrow.memory_bytes() as f64,
        "width 8 → 63: {} → {} bytes",
        narrow.memory_bytes(),
        wide.memory_bytes()
    );

    // The Fibonacci realization is universal, so the frame cannot absorb
    // it — the Gottesman–Knill boundary, from the braid-group side.
    let mut fib = BraidedState::with_realization(1, Realization::Fibonacci).unwrap();
    let (g, t) = Realization::Fibonacci.local_gate(0, 1).unwrap();
    for _ in 0..20 {
        fib.apply(&g, &t).unwrap();
    }
    assert!(fib.is_pure_braid(), "Fibonacci generators are still generators");
    assert!(BraidedState::with_realization(3, Realization::Fibonacci).is_err());
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
    let n = 12;
    let count = Realization::Majorana.generator_count(n);
    assert_eq!(count, 2 * n - 1);
    let mut s = BraidedState::new(n).unwrap();
    assert_eq!(s.realization(), Realization::Majorana);

    // 300 generators at width 12 — no dense matrix exists anywhere here
    let mut d = DenseState::<C64>::new(n).unwrap();
    for k in 0..300 {
        let (g, t) = Realization::Majorana.local_gate(k % count, n).unwrap();
        s.apply(&g, &t).unwrap();
        d.apply(&g, &t).unwrap();
    }
    assert!(s.is_pure_braid());
    assert_eq!(s.word().len(), 300);
    assert_eq!(s.stored_support(), 1);

    let dev = max_amplitude_deviation(&s as &dyn Backend<C64>, &d as &dyn Backend<C64>);
    assert!(dev < 1e-12, "braided word vs dense: {dev:.3e}");

    // canonicalizing drops the word; the tableau still holds the element
    s.canonicalize();
    assert!(s.word().is_empty());
    let dev = max_amplitude_deviation(&s as &dyn Backend<C64>, &d as &dyn Backend<C64>);
    assert!(dev < 1e-12, "canonicalized braided vs dense: {dev:.3e}");

    // anything outside the generator set is recorded as an escape, and
    // the frame still evolves it correctly
    let mut t = BraidedState::new(4).unwrap();
    let reg = GateRegistry::<C64>::standard();
    let mut c: Circuit = Circuit::new(4);
    c.h(0).t(1).cx(0, 2);
    c.bind(&reg).unwrap().run(&mut t).unwrap();
    assert_eq!(t.escapes(), 3);
    assert!(!t.is_pure_braid());

    assert!(BraidedState::new(0).is_err());
}

// ── the stateless query: an observable, with no state ────────────────

#[test]
fn the_heisenberg_query_matches_dense_without_a_state_vector() {
    use quantsim::backend::PauliString;

    // One Trotter layer of the transverse-field Ising model at the
    // self-dual Clifford point: exp(iπ/4 Z_k) then exp(iπ/4 X_k X_k+1).
    // Every gate is a braid generator, by construction.
    let layer = |n: usize| -> Vec<(GateMatrix<C64>, Vec<usize>)> {
        let mut v = Vec::new();
        for k in 0..n {
            v.push(Realization::Majorana.local_gate(2 * k, n).unwrap());
        }
        for k in 0..n.saturating_sub(1) {
            v.push(Realization::Majorana.local_gate(2 * k + 1, n).unwrap());
        }
        v
    };

    let mut worst = 0.0f64;
    let mut checked = 0usize;
    for n in 2..=8usize {
        let gates = layer(n);
        let mut b = BraidedState::new(n).unwrap();
        let mut d = DenseState::<C64>::new(n).unwrap();
        for _ in 0..6 {
            for (g, t) in &gates {
                b.apply(g, t).unwrap();
                d.apply(g, t).unwrap();
            }
            for q in 0..n {
                for (p, ops) in [
                    (PauliString { x: 0, z: 1 << q, negative: false }, vec![(q, Pauli::Z)]),
                    (PauliString { x: 1 << q, z: 0, negative: false }, vec![(q, Pauli::X)]),
                ] {
                    let mine = b.expectation(p);
                    // quantized on a stabilizer state: exactly 0 or ±1
                    assert!(
                        mine == 0.0 || mine == 1.0 || mine == -1.0,
                        "expectation {mine} is not quantized"
                    );
                    let reference =
                        pauli_expectation(&d as &dyn Backend<C64>, &ops).unwrap().re;
                    worst = worst.max((mine - reference).abs());
                    checked += 1;
                }
            }
        }
        assert!(b.is_pure_braid());
        assert_eq!(b.stored_support(), 1);
    }
    assert!(checked >= 400, "only {checked} checks");
    assert!(worst < 1e-14, "stateless expectation deviated by {worst:.3e}");
}

#[test]
fn the_observable_light_cone_is_ballistic_and_needs_no_state() {
    use quantsim::backend::PauliString;
    let n = 63;
    let mut s = BraidedState::new(n).unwrap();
    let centre = 31usize;
    let z = PauliString { x: 0, z: 1 << centre, negative: false };
    assert_eq!(s.observable_weight(z), 1);

    for t in 1..=12usize {
        for k in 0..n {
            let (g, tg) = Realization::Majorana.local_gate(2 * k, n).unwrap();
            s.apply(&g, &tg).unwrap();
        }
        for k in 0..n - 1 {
            let (g, tg) = Realization::Majorana.local_gate(2 * k + 1, n).unwrap();
            s.apply(&g, &tg).unwrap();
        }
        // the support grows by exactly one site each way per layer:
        // Lieb-Robinson velocity 1, read off the conjugated Pauli
        let c = s.conjugated(z);
        let support = c.x | c.z;
        assert_eq!(c.weight(), 2 * t + 1, "weight at layer {t}");
        assert_eq!(support.trailing_zeros() as usize, centre - t);
        assert_eq!(63 - support.leading_zeros() as usize, centre + t);
    }
    // and none of that ever built an amplitude
    assert_eq!(s.stored_support(), 1);
    assert_eq!(s.frame_stats().flushes, 0);
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

    // No structural width limit any more: the generators are weight-≤2
    // Clifford rotations, so the wider workload runs on every backend.
    let wide = Workload::new("iqp-core-14", || iqp_core(14));
    let wide_report = compare_backends(
        &sim,
        &[wide],
        &["dense", "phase-field", "braided"],
        &BenchConfig::default(),
    )
    .unwrap();
    for name in ["dense", "phase-field", "braided"] {
        let r = wide_report.record("iqp-core-14", name).unwrap();
        assert!(r.error.is_none(), "{name} errored: {:?}", r.error);
    }
    // and at width 14 the phase field is still tiny where dense is not
    let pf14 = wide_report.record("iqp-core-14", "phase-field").unwrap();
    let de14 = wide_report.record("iqp-core-14", "dense").unwrap();
    assert!(
        pf14.memory_bytes * 50 < de14.memory_bytes,
        "phase-field {} vs dense {} at width 14",
        pf14.memory_bytes,
        de14.memory_bytes
    );
}
