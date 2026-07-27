//! The operational-model backends and runtime selection:
//! interference accounting, physical device order/routing/latency — both
//! held to reference fidelity by the conformance suite — and VOLK-style
//! verified backend selection.

mod common;

use common::*;
use quantsim::prelude::*;
use std::f64::consts::FRAC_1_SQRT_2;

fn sim_with_models() -> Simulator {
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("interference", |n| Ok(Box::new(InterferenceState::new(n)?)))
        .unwrap();
    sim.backends_mut()
        .register("device-linear", |n| {
            Ok(Box::new(DeviceState::new(
                Topology::linear(n),
                DurationModel::default(),
                ArityPolicy::default(),
            )?))
        })
        .unwrap();
    sim.backends_mut()
        .register("device-ring", |n| {
            Ok(Box::new(DeviceState::new(
                Topology::ring(n),
                DurationModel::default(),
                ArityPolicy::default(),
            )?))
        })
        .unwrap();
    sim
}

// ───────────────────── fidelity against the reference ─────────────────────

#[test]
fn model_backends_conform_over_the_full_registry() {
    // The user-facing guarantee: instrumented interference accounting and
    // physically-routed device execution reproduce the dense reference on
    // every registered gate, random circuits, sampling, and collapse.
    let sim = sim_with_models();
    let cfg = ConformanceConfig {
        random_circuits: 10,
        orderings: 2,
        ..ConformanceConfig::default()
    };
    for backend in ["interference", "device-linear", "device-ring"] {
        let report = verify_backend(&sim, backend, &cfg).unwrap();
        assert!(report.passed(), "{report}");
        assert!(report.max_amplitude_deviation < 1e-9, "{report}");
        assert_eq!(report.sampling_mismatches, 0, "{backend}");
        assert_eq!(report.collapse_violations, 0, "{backend}");
    }
}

// ───────────────────────── interference model ─────────────────────────

#[test]
fn hadamard_pair_destroys_exactly_one_unit() {
    // H then H on |0⟩: the second H sends the |1⟩ output to zero by exact
    // cancellation — path weight 1 destroyed; the |0⟩ output is fully
    // constructive.
    let mut state = InterferenceState::<C64>::new(1).unwrap();
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    state.apply(&h, &[0]).unwrap();
    state.apply(&h, &[0]).unwrap();
    let records = state.records();
    assert_eq!(records.len(), 2);
    // First H: single incoming contribution per output — nothing to cancel.
    assert_close(records[0].destroyed(), 0.0, 1e-12);
    // Second H: |0⟩ gets 0.5 + 0.5 (constructive), |1⟩ gets 0.5 − 0.5.
    assert_close(records[1].path_weight, 2.0, 1e-12);
    assert_close(records[1].net_weight, 1.0, 1e-12);
    assert_close(records[1].destroyed(), 1.0, 1e-12);
    assert_close(records[1].constructive_fraction(), 0.5, 1e-12);
    // The destruction landed at basis state |1⟩, not |0⟩.
    assert_close(state.destruction_map()[0], 0.0, 1e-12);
    assert_close(state.destruction_map()[1], 1.0, 1e-12);
    // And the amplitudes are still exactly right.
    assert!(state.amplitude(0).approx_eq(c64(1.0, 0.0), 1e-12));
}

#[test]
fn diagonal_gates_never_interfere() {
    let sim = sim_with_models();
    let mut c = Circuit::new(3);
    c.h(0).h(1).h(2);
    c.diagonal("flip5", library::phase_flip(3, 5), vec![0, 1, 2]);
    c.rz(0, 0.7).cp(1, 2, 1.1);
    let state = sim.run_on("interference", &c).unwrap();
    let interference = state
        .as_any()
        .downcast_ref::<InterferenceState<C64>>()
        .unwrap();
    let records = interference.records();
    assert_eq!(records.len(), 6);
    // The diagonal ops (indices 3, 4, 5) move phases but sum nothing.
    for record in &records[3..] {
        assert_close(record.destroyed(), 0.0, 1e-12);
        assert_close(record.constructive_fraction(), 1.0, 1e-12);
    }
}

#[test]
fn grover_interference_profile() {
    // Grover is an interference engine: the oracle (diagonal) destroys
    // nothing; the H-layers of the diffusion do all the destructive work,
    // concentrating amplitude constructively on the marked state — while
    // the instrumented run still reproduces the closed-form probability.
    let sim = sim_with_models();
    let circuit = library::grover(3, 5, 2).unwrap();
    let state = sim.run_on("interference", &circuit).unwrap();
    assert_close(state.probability(5), 0.9453125, 1e-9);

    let interference = state
        .as_any()
        .downcast_ref::<InterferenceState<C64>>()
        .unwrap();
    let records = interference.records();
    // Ops: 3 initial H, then per iteration [oracle, 3 H, flip0, 3 H] × 2.
    assert_eq!(records.len(), 3 + 2 * 8);
    let oracle_indices = [3usize, 11];
    for &i in &oracle_indices {
        assert_close(records[i].destroyed(), 0.0, 1e-12);
    }
    let total = interference.total_destroyed();
    assert!(
        total > 1.0,
        "diffusion must cancel substantial amplitude, got {total}"
    );
    // Everything destroyed came from non-diagonal (H) layers.
    let h_destroyed: f64 = records
        .iter()
        .enumerate()
        .filter(|(i, _)| !oracle_indices.contains(i) && ![7usize, 15].contains(i))
        .map(|(_, r)| r.destroyed())
        .sum();
    assert_close(h_destroyed, total, 1e-9);
}

// ───────────────────────── device model ─────────────────────────

#[test]
fn routing_moves_entanglement_stepwise_through_adjacency() {
    // cx(0, 5) on a 6-site linear chain: operand 0 must physically walk
    // 0→1→2→3→4 (four swaps), then interact with site 5.
    let sim = sim_with_models();
    let mut c = Circuit::new(6);
    c.h(0).cx(0, 5);
    let state = sim.run_on("device-linear", &c).unwrap();
    let device = state.as_any().downcast_ref::<DeviceState<C64>>().unwrap();
    assert_eq!(device.swap_count(), 4);

    // The log shows the stepwise walk: every swap is on adjacent sites,
    // marching monotonically toward the target.
    let swaps: Vec<&PhysicalOp> = device
        .physical_log()
        .iter()
        .filter(|op| op.label == "swap")
        .collect();
    assert_eq!(swaps.len(), 4);
    for (step, op) in swaps.iter().enumerate() {
        assert_eq!(
            op.sites,
            vec![step, step + 1],
            "step {step} not adjacent-forward"
        );
        assert!(device.topology().adjacent(op.sites[0], op.sites[1]));
    }
    // Routing left logical 0 living at physical 4 — mapping is persistent.
    assert_eq!(device.mapping()[0], 4);

    // And the logical result is exactly the reference Bell pair on (0, 5).
    assert!(state.amplitude(0).approx_eq(c64(FRAC_1_SQRT_2, 0.0), TOL));
    assert!(state
        .amplitude(0b100001)
        .approx_eq(c64(FRAC_1_SQRT_2, 0.0), TOL));
    assert_eq!(state.nonzero_count(), 2);

    // A second cx(0, 5): the operands are already adjacent — no new swaps.
    let mut c2 = Circuit::new(6);
    c2.h(0).cx(0, 5).cx(0, 5);
    let state2 = sim.run_on("device-linear", &c2).unwrap();
    let device2 = state2.as_any().downcast_ref::<DeviceState<C64>>().unwrap();
    assert_eq!(device2.swap_count(), 4, "locality persists after routing");
}

#[test]
fn topology_changes_the_cost_of_the_same_circuit() {
    // cx(0, 7) on 8 qubits: 6 swaps on a chain, 0 on a ring (adjacent).
    let sim = sim_with_models();
    let mut c = Circuit::new(8);
    c.h(0).cx(0, 7);
    let linear = sim.run_on("device-linear", &c).unwrap();
    let ring = sim.run_on("device-ring", &c).unwrap();
    let linear = linear.as_any().downcast_ref::<DeviceState<C64>>().unwrap();
    let ring = ring.as_any().downcast_ref::<DeviceState<C64>>().unwrap();
    assert_eq!(linear.swap_count(), 6);
    assert_eq!(ring.swap_count(), 0);
    assert!(linear.elapsed() > ring.elapsed());
    // Same physics either way.
    for i in 0..256u64 {
        assert!(linear.amplitude(i).approx_eq(ring.amplitude(i), TOL));
    }
}

#[test]
fn latency_model_parallelizes_independent_ops() {
    let sim = sim_with_models();
    // A layer of H on all qubits: independent sites advance in parallel —
    // elapsed is one 1q duration, not n of them.
    let mut layer = Circuit::new(5);
    for q in 0..5 {
        layer.h(q);
    }
    let state = sim.run_on("device-linear", &layer).unwrap();
    let device = state.as_any().downcast_ref::<DeviceState<C64>>().unwrap();
    assert_eq!(device.elapsed(), DurationModel::default().one_q);

    // Serial ops on one qubit accumulate.
    let mut serial = Circuit::new(5);
    serial.h(0).h(0).h(0);
    let state = sim.run_on("device-linear", &serial).unwrap();
    let device = state.as_any().downcast_ref::<DeviceState<C64>>().unwrap();
    assert_eq!(device.elapsed(), 3 * DurationModel::default().one_q);

    // Routed cx(0, 3): three sequential swaps… wait — path 0-1-2-3 has two
    // intermediate hops (swaps at (0,1), (1,2)), then the 2q gate.
    let mut routed = Circuit::new(5);
    routed.cx(0, 3);
    let state = sim.run_on("device-linear", &routed).unwrap();
    let device = state.as_any().downcast_ref::<DeviceState<C64>>().unwrap();
    let d = DurationModel::default();
    assert_eq!(device.swap_count(), 2);
    assert_eq!(device.elapsed(), 2 * d.swap + d.two_q);
}

#[test]
fn grid_topology_and_strict_arity_policy() {
    // 3×3 grid adjacency: site 4 (center) touches 1, 3, 5, 7.
    let grid = Topology::grid(3, 3);
    assert!(
        grid.adjacent(4, 1) && grid.adjacent(4, 3) && grid.adjacent(4, 5) && grid.adjacent(4, 7)
    );
    assert!(!grid.adjacent(0, 4) && !grid.adjacent(0, 8));
    assert_eq!(grid.shortest_path(0, 8).unwrap().len(), 5); // 4 hops

    // Strict policy refuses non-native wide gates instead of pretending.
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("device-strict", |n| {
            Ok(Box::new(DeviceState::new(
                Topology::linear(n),
                DurationModel::default(),
                ArityPolicy::Reject,
            )?))
        })
        .unwrap();
    let mut c = Circuit::new(3);
    c.ccx(0, 1, 2);
    assert!(matches!(
        sim.run_on("device-strict", &c),
        Err(Error::InvalidState(_))
    ));
    // Its decomposition (arity ≤ 2) runs fine and matches the reference.
    let mut decomposed = Circuit::new(3);
    let (a, b, t) = (0usize, 1usize, 2usize);
    decomposed
        .h(t)
        .cx(b, t)
        .tdg(t)
        .cx(a, t)
        .t(t)
        .cx(b, t)
        .tdg(t)
        .cx(a, t)
        .t(b)
        .t(t)
        .h(t)
        .cx(a, b)
        .t(a)
        .tdg(b)
        .cx(a, b);
    let mut with_x = Circuit::new(3);
    with_x.x(0).x(1).append(&decomposed, &[0, 1, 2]);
    let state = sim.run_on("device-strict", &with_x).unwrap();
    assert_close(state.probability(0b111), 1.0, 1e-9);
}

#[test]
fn disconnected_topology_reports_no_path() {
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("device-split", |n| {
            // Two islands: 0-1 and 2-3 (requires n = 4).
            assert_eq!(n, 4);
            Ok(Box::new(DeviceState::new(
                Topology::custom(4, &[(0, 1), (2, 3)])?,
                DurationModel::default(),
                ArityPolicy::default(),
            )?))
        })
        .unwrap();
    let mut c = Circuit::new(4);
    c.cx(0, 2);
    assert!(matches!(
        sim.run_on("device-split", &c),
        Err(Error::InvalidState(_))
    ));
}

// ───────────────────── VOLK-style runtime selection ─────────────────────

/// Fast and wrong: skips every gate. Selection must reject it on fidelity.
struct NoOpBackend {
    inner: DenseState<C64>,
}

impl Backend<C64> for NoOpBackend {
    fn name(&self) -> &str {
        "noop"
    }
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }
    fn apply(&mut self, _: &GateMatrix<C64>, _: &[usize]) -> Result<()> {
        Ok(())
    }
    fn amplitude(&self, index: u64) -> C64 {
        self.inner.amplitude(index)
    }
    fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, C64)) {
        self.inner.for_each_nonzero(f)
    }
    fn project(&mut self, qubit: usize, outcome: bool, renorm: f64) {
        self.inner.project(qubit, outcome, renorm)
    }
    fn reset(&mut self) {
        self.inner.reset()
    }
    fn load(&mut self, entries: &[(u64, C64)]) -> Result<()> {
        self.inner.load(entries)
    }
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn selection_picks_fast_verified_kernels_and_rejects_wrong_ones() {
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("noop", |n| {
            Ok(Box::new(NoOpBackend {
                inner: DenseState::new(n)?,
            }))
        })
        .unwrap();

    // Concentrated workload: sparse/factored dominate dense; noop is the
    // fastest of all but wrong — fidelity must gate it out.
    let workload = Workload::ghz(16);
    let report = select_backend(
        &sim,
        &workload,
        &["dense", "sparse", "factored", "noop"],
        &BenchConfig {
            repetitions: 2,
            ..BenchConfig::default()
        },
        SelectionCriterion::Time,
        1e-9,
    )
    .unwrap();
    let chosen = report.chosen.clone().expect("a kernel must win");
    assert_ne!(chosen, "dense", "dense cannot win a GHZ time race");
    assert_ne!(chosen, "noop", "unfaithful kernels must never win");
    assert!(report.verified);
    assert!(
        report
            .rejected
            .iter()
            .any(|(name, reason)| name == "noop" && reason.contains("deviates")),
        "{:?}",
        report.rejected
    );

    // Memory criterion on the same workload: the winner must be one of the
    // 2-amplitude representations.
    let by_memory = select_backend(
        &sim,
        &workload,
        &["dense", "sparse", "factored"],
        &BenchConfig {
            repetitions: 1,
            ..BenchConfig::default()
        },
        SelectionCriterion::Memory,
        1e-9,
    )
    .unwrap();
    let chosen = by_memory.chosen.clone().unwrap();
    assert!(chosen == "sparse" || chosen == "factored", "{chosen}");

    // Saturating workload: dense-family kernels win the time race.
    let saturating = Workload::qft(10);
    let report = select_backend(
        &sim,
        &saturating,
        &["dense", "sparse", "adaptive"],
        &BenchConfig {
            repetitions: 2,
            ..BenchConfig::default()
        },
        SelectionCriterion::Time,
        1e-9,
    )
    .unwrap();
    let chosen = report.chosen.clone().unwrap();
    assert_ne!(chosen, "sparse", "saturated sparse cannot win qft");
}

#[test]
fn selection_survives_an_unrunnable_reference() {
    // 34 qubits of local pair circuits: past the dense width cap, so the
    // reference cannot run; selection still returns a winner but reports
    // it as unverified.
    let sim: Simulator = Simulator::new();
    let mut pairs: Circuit = Circuit::new(34);
    for pair in 0..17 {
        pairs
            .ry(2 * pair, 0.2 + 0.05 * pair as f64)
            .cx(2 * pair, 2 * pair + 1);
    }
    let report = select_backend(
        &sim,
        &Workload::from_circuit("local-pairs-34", pairs),
        &["sparse", "factored"],
        &BenchConfig {
            repetitions: 1,
            ..BenchConfig::default()
        },
        SelectionCriterion::Time,
        1e-9,
    )
    .unwrap();
    assert!(report.chosen.is_some());
    assert!(
        !report.verified,
        "no reference at this width — must be flagged"
    );
    // Memory criterion at this width: factored's 17 four-amplitude factors
    // beat sparse's 2^17-entry map decisively.
    let by_memory = select_backend(
        &sim,
        &Workload::from_circuit("local-pairs-34-mem", {
            let mut c: Circuit = Circuit::new(34);
            for pair in 0..17 {
                c.ry(2 * pair, 0.2 + 0.05 * pair as f64)
                    .cx(2 * pair, 2 * pair + 1);
            }
            c
        }),
        &["sparse", "factored"],
        &BenchConfig {
            repetitions: 1,
            ..BenchConfig::default()
        },
        SelectionCriterion::Memory,
        1e-9,
    )
    .unwrap();
    assert_eq!(by_memory.chosen.as_deref(), Some("factored"));
}
