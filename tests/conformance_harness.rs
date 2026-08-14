//! The research-mode workflow, tested end to end: registry-wide backend
//! conformance (including custom gate sets), detection of a deliberately
//! broken backend, and the benchmark harness's correctness-checked
//! performance numbers.

mod common;

use quantsim::prelude::*;

/// A stand-in custom research gate set ("bbq-37"): an fSim-style two-qubit
/// gate and a three-qubit controlled-controlled-phase. Registered on top of
/// the standard library.
fn register_bbq37(sim: &mut Simulator) {
    sim.registry_mut()
        .register_parametric(
            "bbq_fsim",
            "fSim(θ, φ): XY rotation + conditional phase",
            2,
            2,
            |p| {
                let (theta, phi) = (p[0], p[1]);
                let (c, s) = (theta.cos(), theta.sin());
                let (o, l) = (c64(1.0, 0.0), c64(0.0, 0.0));
                GateMatrix::from_vec(
                    4,
                    vec![
                        o,
                        l,
                        l,
                        l,
                        l,
                        c64(c, 0.0),
                        c64(0.0, -s),
                        l,
                        l,
                        c64(0.0, -s),
                        c64(c, 0.0),
                        l,
                        l,
                        l,
                        l,
                        cis(-phi),
                    ],
                )
            },
        )
        .unwrap();
    sim.registry_mut()
        .register_parametric("bbq_ccphase", "doubly controlled phase", 3, 1, |p| {
            let mut m = GateMatrix::identity(8)?;
            m.set(7, 7, C64::try_from_c64(cis(p[0])).unwrap());
            Ok(m)
        })
        .unwrap();
}

#[test]
fn shipped_backends_conform_over_the_full_standard_registry() {
    // Every one of the 39 registry names, plus registry-drawn random
    // circuits and invariants — this is the direct answer to "does dense
    // match sparse match adaptive" for the whole gate set.
    let sim: Simulator = Simulator::new();
    let cfg = ConformanceConfig {
        random_circuits: 12,
        orderings: 2,
        ..ConformanceConfig::default()
    };
    for backend in ["sparse", "adaptive", "factored", "mps", "branched"] {
        let report = verify_backend(&sim, backend, &cfg).unwrap();
        assert!(report.passed(), "{report}");
        assert_eq!(report.gate_checks.len(), sim.registry().len());
        assert!(report.gate_checks.iter().all(|g| g.cases > 0));
        assert!(report.max_amplitude_deviation < 1e-9, "{report}");
        assert_eq!(report.sampling_mismatches, 0);
        assert_eq!(report.collapse_violations, 0);
    }
    // The reference trivially conforms to itself.
    let self_report = verify_backend(&sim, "dense", &cfg).unwrap();
    assert!(self_report.passed());
}

#[test]
fn custom_gate_set_is_swept_automatically() {
    let mut sim: Simulator = Simulator::new();
    register_bbq37(&mut sim);
    let cfg = ConformanceConfig {
        random_circuits: 8,
        orderings: 2,
        ..ConformanceConfig::default()
    };
    let report = verify_backend(&sim, "sparse", &cfg).unwrap();
    assert!(report.passed(), "{report}");
    // The custom gates were discovered from the registry and exercised.
    for name in ["bbq_fsim", "bbq_ccphase"] {
        let check = report
            .gate_checks
            .iter()
            .find(|g| g.gate == name)
            .unwrap_or_else(|| panic!("{name} missing from sweep"));
        assert!(check.cases > 0, "{name} not exercised");
        assert!(check.max_deviation < 1e-9);
    }
}

#[test]
fn conformance_works_for_exotic_algebras() {
    // Registry-driven means the real subset over ℝ and the full set over ℍ
    // are swept without any per-algebra code.
    let cfg = ConformanceConfig {
        random_circuits: 6,
        orderings: 2,
        ..ConformanceConfig::default()
    };
    let real_report = verify_backend(&Simulator::<f64>::new(), "sparse", &cfg).unwrap();
    assert!(real_report.passed(), "{real_report}");
    assert_eq!(real_report.gate_checks.len(), 17);

    let quat_report = verify_backend(&Simulator::<Quaternion>::new(), "sparse", &cfg).unwrap();
    assert!(quat_report.passed(), "{quat_report}");
    assert_eq!(quat_report.gate_checks.len(), 39);
}

/// A deliberately broken backend: reverses the qubit order of every
/// three-qubit gate (moving controls onto targets). Three-qubit only, so
/// the conformance scrambler (1q/2q) stays honest and the per-gate report
/// must localize the damage. Conformance must catch it.
struct SabotagedBackend {
    inner: DenseState<C64>,
}

impl Backend<C64> for SabotagedBackend {
    fn name(&self) -> &str {
        "sabotaged"
    }
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }
    fn apply(&mut self, m: &GateMatrix<C64>, qubits: &[usize]) -> Result<()> {
        if qubits.len() == 3 {
            let flipped: Vec<usize> = qubits.iter().rev().copied().collect();
            self.inner.apply(m, &flipped)
        } else {
            self.inner.apply(m, qubits)
        }
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
fn conformance_catches_a_broken_backend() {
    // A harness that has never been seen to fail proves nothing. The
    // sabotage reverses 3q gate qubit order; asymmetric 3q gates (ccx,
    // cswap and their aliases) must light up while the fully symmetric ccz
    // — and every 1q/2q gate — stays green. That localization is exactly
    // what a real debugging session needs from the report.
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("sabotaged", |n| {
            Ok(Box::new(SabotagedBackend {
                inner: DenseState::new(n)?,
            }))
        })
        .unwrap();
    let cfg = ConformanceConfig {
        random_circuits: 4,
        orderings: 2,
        ..ConformanceConfig::default()
    };
    let report = verify_backend(&sim, "sabotaged", &cfg).unwrap();
    assert!(!report.passed(), "sabotage must be detected");
    let deviation_of = |name: &str| {
        report
            .gate_checks
            .iter()
            .find(|g| g.gate == name)
            .unwrap()
            .max_deviation
    };
    assert!(deviation_of("ccx") > 0.1, "reversed ccx must deviate");
    assert!(deviation_of("toffoli") > 0.1, "alias swept independently");
    assert!(deviation_of("cswap") > 0.1, "reversed cswap must deviate");
    assert!(
        deviation_of("ccz") < 1e-9,
        "ccz is fully symmetric — must stay green"
    );
    assert!(
        deviation_of("cx") < 1e-9,
        "2q gates untouched by the sabotage"
    );
    assert!(
        deviation_of("h") < 1e-9,
        "1q gates untouched by the sabotage"
    );
    // The report names the offenders worst-first.
    assert!(report.worst_gates(3).iter().all(|g| g.max_deviation > 0.1));
    // And the failure text localizes them by name.
    assert!(report
        .failures
        .iter()
        .any(|f| f.contains("ccx") || f.contains("cswap")));
}

#[test]
fn harness_measures_speed_memory_and_correctness_together() {
    let mut sim: Simulator = Simulator::new();
    register_bbq37(&mut sim);
    let workloads = vec![
        Workload::ghz(16),
        Workload::random(8, 60, 0xBB9),
        Workload::from_circuit(
            "bbq37-mix",
            random_registry_circuit(sim.registry(), 6, 40, 37),
        ),
    ];
    let report = compare_backends(
        &sim,
        &workloads,
        &["dense", "sparse", "adaptive"],
        &BenchConfig {
            repetitions: 2,
            ..BenchConfig::default()
        },
    )
    .unwrap();

    // Every combination ran, was verified, and stayed correct.
    assert_eq!(report.records.len(), 9);
    assert!(report.records.iter().all(|r| r.error.is_none()));
    assert!(
        report.max_deviation() < 1e-9,
        "harness caught a mismatch:\n{report}"
    );
    assert!(report
        .records
        .iter()
        .all(|r| (r.total_weight - 1.0).abs() < 1e-9));

    // The efficiency story is captured: sparse crushes dense on GHZ memory.
    let memx = report.memory_ratio("ghz-16", "sparse").unwrap();
    assert!(
        memx > 100.0,
        "expected ≫100× memory advantage, got {memx:.1}×"
    );
    let ghz_sparse = report.record("ghz-16", "sparse").unwrap();
    assert_eq!(ghz_sparse.nonzeros, 2);
    // Speedups are computed and positive (magnitudes are machine-dependent).
    assert!(report.speedup("ghz-16", "sparse").unwrap() > 0.0);
    // The report renders.
    assert!(format!("{report}").contains("ghz-16"));
}

#[test]
fn harness_degrades_gracefully_past_the_reference_limit() {
    // GHZ at 40 qubits: dense (the reference) cannot exist, sparse can.
    let sim: Simulator = Simulator::new();
    let workloads = vec![Workload::ghz(40)];
    let report = compare_backends(
        &sim,
        &workloads,
        &["sparse", "dense"],
        &BenchConfig {
            repetitions: 1,
            ..BenchConfig::default()
        },
    )
    .unwrap();
    let sparse = report.record("ghz-40", "sparse").unwrap();
    assert!(sparse.error.is_none());
    assert_eq!(sparse.nonzeros, 2);
    assert!(
        sparse.deviation.is_none(),
        "no reference available at this width"
    );
    let dense = report.record("ghz-40", "dense").unwrap();
    assert!(dense.error.is_some(), "dense must report its width failure");
}

#[test]
fn registry_random_circuits_respect_width_and_params() {
    let sim: Simulator = Simulator::new();
    // Width 1: only 1q gates are eligible; circuit still builds and runs.
    let c = random_registry_circuit(sim.registry(), 1, 25, 5);
    assert_eq!(c.len(), 25);
    let state = sim.run(&c).unwrap();
    assert!((state.total_weight() - 1.0).abs() < 1e-9);
    // Empty registry yields an empty circuit rather than panicking.
    let empty = GateRegistry::<C64>::new();
    assert!(random_registry_circuit(&empty, 3, 10, 1).is_empty());
}
