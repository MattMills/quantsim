//! Both E8×E8 systems as first-class backends, in the same conformance
//! and performance frameworks as every other representation: the
//! compound-register substrate that carries the 240-level co-boundary
//! state enters `verify_backend` and `compare_backends` as
//! `"compound-binary"`, the root-keyed representation as `"e8-rep"`
//! (verified at its native width 8), and the co-boundary protocol
//! itself becomes a benchmark workload every backend must reproduce.

mod common;

use quantsim::e8::rep::E8RepState;
use quantsim::prelude::*;

fn register_e8_backends(sim: &mut Simulator) {
    sim.backends_mut()
        .register("compound-binary", |n| {
            Ok(Box::new(CompoundBackend::new(n)?))
        })
        .unwrap();
    sim.backends_mut()
        .register("e8-rep", |n| Ok(Box::new(E8RepState::new(n)?)))
        .unwrap();
}

/// The co-boundary protocol at d = 16 (two 16-level sites in 8 qubits:
/// site A = qubits 0–3, site B = qubits 4–7): prepare
/// Σ_a g(a)|a,a⟩/√16, then uncompute the pairing and interfere with
/// F†, leaving Σ_k ĝ(k)|k,0⟩ — the DFT of the stored field. The same
/// protocol the 240-level system runs, at a width every qubit
/// representation can hold.
fn coboundary16_circuit(field: &[C64]) -> Circuit {
    let d = 16usize;
    let dim = d * d;
    let f = fourier_d(d);
    let fdag: Vec<C64> = (0..d * d).map(|i| f[(i % d) * d + i / d].conj()).collect();
    let mut diag = vec![c64(0.0, 0.0); d * d];
    for (j, &g) in field.iter().enumerate() {
        diag[j * d + j] = g;
    }
    let mut unpair = vec![c64(0.0, 0.0); dim * dim];
    for a in 0..d {
        for b in 0..d {
            let from = a + d * b;
            let to = a + d * ((b + d - a) % d);
            unpair[to * dim + from] = c64(1.0, 0.0);
        }
    }
    let site_a: Vec<usize> = (0..4).collect();
    let site_b: Vec<usize> = (4..8).collect();
    let both: Vec<usize> = (0..8).collect();
    let mut c: Circuit = Circuit::new(8);
    c.raw("f16", GateMatrix::from_vec(d, f).unwrap(), site_a.clone());
    c.raw(
        "pair16",
        GateMatrix::from_vec(dim, cshift(d, d)).unwrap(),
        both.clone(),
    );
    c.raw("field16", GateMatrix::from_vec(d, diag).unwrap(), site_b);
    c.raw("unpair16", GateMatrix::from_vec(dim, unpair).unwrap(), both);
    c.raw("f16dag", GateMatrix::from_vec(d, fdag).unwrap(), site_a);
    c
}

fn field16() -> Vec<C64> {
    (0..16usize)
        .map(|a| cis(std::f64::consts::TAU * ((a * a + 3 * a) % 17) as f64 / 17.0))
        .collect()
}

#[test]
fn compound_binary_conforms_over_the_full_registry() {
    // The co-boundary system's substrate — volumes, guard-admitted
    // merges, mixed-radix packing — as a standard qubit backend, swept
    // over every registered gate at randomized parameters, orderings
    // and widths, plus registry-drawn random circuits, sampling
    // equality and measurement collapse, all against dense.
    let mut sim: Simulator = Simulator::new();
    register_e8_backends(&mut sim);
    let report = verify_backend(&sim, "compound-binary", &ConformanceConfig::default()).unwrap();
    assert!(report.passed(), "{report}");
    assert_eq!(report.gate_checks.len(), sim.registry().len());
    assert!(report.gate_checks.iter().all(|g| g.cases > 0));
    assert!(report.max_amplitude_deviation < 1e-9, "{report}");
    assert_eq!(report.sampling_mismatches, 0);
    assert_eq!(report.collapse_violations, 0);

    // The packed-index width wall is a measured refusal, same as
    // sparse: 63 qubits construct, 64 refuse.
    assert!(CompoundBackend::new(63).is_ok());
    assert!(matches!(
        CompoundBackend::new(64),
        Err(Error::TooManyQubits { max: 63, .. })
    ));
}

#[test]
fn e8_rep_conforms_including_its_native_width_8() {
    // The root-keyed representation over the full registry — once at
    // the default widths, once with widths pushed to 8 so every gate
    // is exercised on the complete E8×E8 point set (256 basis states =
    // all 240 + 16 paired spinor points of the two copies).
    let mut sim: Simulator = Simulator::new();
    register_e8_backends(&mut sim);
    let report = verify_backend(&sim, "e8-rep", &ConformanceConfig::default()).unwrap();
    assert!(report.passed(), "{report}");

    let wide = ConformanceConfig {
        extra_widths: vec![0, 7], // width clamps to max_gate_width = 8
        random_circuits: 12,
        orderings: 2,
        ..ConformanceConfig::default()
    };
    let report = verify_backend(&sim, "e8-rep", &wide).unwrap();
    assert!(report.passed(), "{report}");
    assert_eq!(report.sampling_mismatches, 0);
    assert_eq!(report.collapse_violations, 0);

    // Past 8 qubits the representation refuses with the structural
    // reason: its point set IS the 8-qubit basis.
    assert!(matches!(
        E8RepState::new(9),
        Err(Error::TooManyQubits {
            requested: 9,
            max: 8
        })
    ));

    // And the storage really is root-keyed, not bits in disguise: GHZ
    // run through the simulator lands as exactly one antipodal pair
    // of spinor roots on copy 0.
    let state = sim.run_on("e8-rep", &library::ghz(8)).unwrap();
    let rep = state
        .as_any()
        .downcast_ref::<E8RepState>()
        .expect("run_on used the registered representation");
    let points = rep.stored_points();
    assert_eq!(points.len(), 2);
    assert_eq!((points[0].0, points[1].0), (0, 0), "both on the even copy");
    let negated: quantsim::e8::Root = std::array::from_fn(|i| -points[0].1[i]);
    assert_eq!(points[1].1, negated, "GHZ is stored as an antipodal pair");
}

#[test]
fn both_e8_systems_enter_the_benchmark_harness() {
    // The performance frame: every workload (the co-boundary protocol
    // included) on the standard representations AND the two E8×E8
    // systems, every run verified against dense in the same table.
    let mut sim: Simulator = Simulator::new();
    register_e8_backends(&mut sim);
    let workloads = [
        Workload::from_circuit("ghz-8", library::ghz(8)),
        Workload::from_circuit("qft-8", library::qft(8)),
        Workload::from_circuit("rainbow-8", library::rainbow(8)),
        Workload::from_circuit("coboundary-16", coboundary16_circuit(&field16())),
    ];
    let backends = [
        "dense",
        "sparse",
        "adaptive",
        "factored",
        "mps",
        "mera",
        "compound-binary",
        "e8-rep",
    ];
    let report = compare_backends(&sim, &workloads, &backends, &BenchConfig::default()).unwrap();
    assert_eq!(report.records.len(), workloads.len() * backends.len());
    assert!(report.max_deviation() < 1e-9, "{report}");

    // The two E8 systems run everything — and so does every standard
    // representation except MPS (measured below): MERA's fixed tree
    // has no gate-window cap at this width.
    for backend in [
        "dense",
        "sparse",
        "adaptive",
        "factored",
        "mera",
        "compound-binary",
        "e8-rep",
    ] {
        for w in ["ghz-8", "qft-8", "rainbow-8", "coboundary-16"] {
            let r = report.record(w, backend).unwrap();
            assert!(r.error.is_none(), "{backend} on {w}: {:?}", r.error);
            assert!((r.total_weight - 1.0).abs() < 1e-9);
        }
    }
    // MPS hits its measured gate-window wall (5 qubits) on the
    // 8-qubit pairing permutation — an honest structural refusal,
    // recorded in the same table as everyone else's numbers.
    let r = report.record("coboundary-16", "mps").unwrap();
    assert!(r.error.is_some(), "mps must refuse the 8-qubit gate");
    assert!(
        report.record("ghz-8", "mps").unwrap().error.is_none(),
        "mps runs the ordinary workloads"
    );

    // Measured structure: GHZ stays 2-point in both E8 systems, well
    // under dense's 2^8 amplitudes.
    for backend in ["compound-binary", "e8-rep"] {
        let r = report.record("ghz-8", backend).unwrap();
        assert_eq!(r.nonzeros, 2, "{backend}");
        assert!(r.memory_bytes < report.record("ghz-8", "dense").unwrap().memory_bytes);
        assert!(report.speedup("ghz-8", backend).unwrap() > 0.0);
        assert!(report.memory_ratio("ghz-8", backend).unwrap() > 1.0);
    }
    // QFT saturates: both systems hold the full 256-point support,
    // exactly like dense — representation changes storage keys, not
    // physics.
    for backend in ["compound-binary", "e8-rep"] {
        assert_eq!(report.record("qft-8", backend).unwrap().nonzeros, 256);
    }
}

#[test]
fn the_native_mixed_arity_protocol_matches_its_qubit_encoding() {
    // Cross-framework conformance for the co-boundary system itself:
    // the d = 16 protocol run natively on two 16-level compound sites
    // must equal, amplitude by amplitude, the same protocol run as an
    // 8-qubit circuit on the dense reference — and both must equal the
    // independently computed DFT of the stored field.
    let d = 16usize;
    let field = field16();
    let sim: Simulator = Simulator::new();
    let dense = sim.run(&coboundary16_circuit(&field)).unwrap();

    let mut reg = CompoundRegister::new(&[d, d]).unwrap();
    reg.apply_1(0, &fourier_d(d)).unwrap();
    reg.apply_2_permutation(0, 1, &|a, b| (a, (b + a) % d))
        .unwrap();
    reg.apply_1_diagonal(1, &field).unwrap();
    reg.apply_2_permutation(0, 1, &|a, b| (a, (b + d - a) % d))
        .unwrap();
    let f = fourier_d(d);
    let fdag: Vec<C64> = (0..d * d).map(|i| f[(i % d) * d + i / d].conj()).collect();
    reg.apply_1(0, &fdag).unwrap();

    let mut worst: f64 = 0.0;
    for a in 0..d {
        for b in 0..d {
            let native = reg.amplitude(&[a, b]).unwrap();
            let encoded = dense.amplitude((a + d * b) as u64);
            worst = worst.max((native - encoded).norm());
        }
    }
    assert!(
        worst < 1e-9,
        "native vs qubit-encoded deviate by {worst:.2e}"
    );

    // Both equal the independent DFT of the field.
    for (k, _) in field.iter().enumerate() {
        let mut expected = c64(0.0, 0.0);
        for (alpha, phase) in field.iter().enumerate() {
            let angle = -std::f64::consts::TAU * (k * alpha % d) as f64 / d as f64;
            expected += *phase * c64(angle.cos(), angle.sin());
        }
        expected /= d as f64;
        let got = dense.amplitude(k as u64);
        common::assert_close((got - expected).norm(), 0.0, 1e-9);
    }
}
