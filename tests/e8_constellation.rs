//! The infinite E8 constellation, measured: E8/2E8 has exactly 256
//! coset classes — the origin, the 120 antipodal root pairs on the
//! √2-sphere, and 135 sixteen-vector frames on the 2-sphere — so one
//! byte positions "this" E8 on the shells of its parent at doubled
//! scale. The scale tower is a verified bijection (self-similar under
//! doubling), and the backend built on it expands the E8
//! representation to any width up to the u64 index wall: a 63-qubit
//! GHZ is literally two lattice points.

mod common;

use common::assert_close;
use quantsim::e8::constellation::{self, E8ConstellationState, Point, Shell};
use quantsim::prelude::*;
use std::collections::{HashMap, HashSet};

fn register_constellation(sim: &mut Simulator) {
    sim.backends_mut()
        .register("e8-constellation", |n| {
            Ok(Box::new(E8ConstellationState::new(n)?))
        })
        .unwrap();
}

#[test]
fn the_256_coset_classes_are_measured_with_their_spheres() {
    // The norm-2 shell: constructed by shape, 2160 vectors, verified
    // on the doubled-16 sphere.
    let n4 = constellation::norm4_vectors();
    assert_eq!(n4.len(), 2160);
    for v in n4.iter().step_by(89) {
        assert_eq!(v.iter().map(|c| c * c).sum::<i64>(), 16);
    }

    // Bucket the origin, the 240 roots and the 2160 norm-2 vectors by
    // coset class: exactly 256 classes with the exact structure
    // 1 origin + 120 root pairs + 135 sixteen-frames.
    let mut buckets: HashMap<u8, Vec<Point>> = HashMap::new();
    buckets.insert(constellation::class_of(&[0; 8]).unwrap(), vec![[0; 8]]);
    for r in quantsim::e8::roots() {
        let p: Point = std::array::from_fn(|k| r[k] as i64);
        buckets
            .entry(constellation::class_of(&p).unwrap())
            .or_default()
            .push(p);
    }
    for p in &n4 {
        buckets
            .entry(constellation::class_of(p).unwrap())
            .or_default()
            .push(*p);
    }
    assert_eq!(buckets.len(), 256, "E8/2E8 ≅ F₂⁸ — one byte exactly");
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for members in buckets.values() {
        *sizes.entry(members.len()).or_insert(0) += 1;
    }
    assert_eq!(sizes.get(&1), Some(&1), "one origin class");
    assert_eq!(sizes.get(&2), Some(&120), "120 antipodal root pairs");
    assert_eq!(sizes.get(&16), Some(&135), "135 frames of sixteen");
    assert_eq!(sizes.len(), 3);

    // The shells are the class representatives' spheres: radius 0,
    // √2, 2 — with the same 1/120/135 census.
    let mut shell_counts = [0usize; 3];
    for c in 0u16..256 {
        let c = c as u8;
        let rep = constellation::representative(c);
        let norm: i64 = rep.iter().map(|x| x * x).sum();
        let shell = constellation::shell_of(c);
        // Doubled dot = 4 × real norm².
        assert_eq!(norm, 4 * shell.radius_sqr(), "rep sits on its shell");
        assert_eq!(
            constellation::class_of(&rep),
            Some(c),
            "rep is in its class"
        );
        match shell {
            Shell::Center => shell_counts[0] += 1,
            Shell::RootSphere => shell_counts[1] += 1,
            Shell::FrameSphere => shell_counts[2] += 1,
        }
    }
    assert_eq!(shell_counts, [1, 120, 135]);
    assert_eq!(constellation::representative(0), [0; 8]);

    // The address map is LINEAR — it is the coordinate map of
    // E8/2E8 ≅ F₂⁸, so classes XOR: class(x + y) = class(x) ⊕
    // class(y), measured across root pairs.
    let rs = quantsim::e8::roots();
    for (i, a) in rs.iter().enumerate().step_by(37) {
        for b in rs.iter().skip(i + 1).step_by(41) {
            let pa: Point = std::array::from_fn(|k| a[k] as i64);
            let pb: Point = std::array::from_fn(|k| b[k] as i64);
            let sum: Point = std::array::from_fn(|k| pa[k] + pb[k]);
            let expect =
                constellation::class_of(&pa).unwrap() ^ constellation::class_of(&pb).unwrap();
            assert_eq!(constellation::class_of(&sum), Some(expect));
        }
    }

    // Non-lattice coordinates refuse.
    assert_eq!(constellation::class_of(&[1, 0, 0, 0, 0, 0, 0, 0]), None);
}

#[test]
fn the_scale_tower_is_a_bijection_and_self_similar() {
    // Depth 2: all 65 536 digit strings compose to DISTINCT lattice
    // points and decompose back exactly — the bijection onto E8/4E8,
    // measured in full.
    let mut seen = HashSet::new();
    for hi in 0u16..256 {
        for lo in 0u16..256 {
            let digits = [lo as u8, hi as u8];
            let p = constellation::compose(&digits);
            assert_eq!(
                constellation::decompose(&p, 2).as_deref(),
                Some(&digits[..]),
                "roundtrip at ({lo}, {hi})"
            );
            seen.insert(p);
        }
    }
    assert_eq!(seen.len(), 65_536, "compose is injective at depth 2");

    // Deep towers roundtrip (depth 7 ≈ a 56-qubit address).
    let mut rng = Prng::new(0xE8);
    for _ in 0..200 {
        let digits: Vec<u8> = (0..7).map(|_| rng.next_u64() as u8).collect();
        let p = constellation::compose(&digits);
        assert_eq!(constellation::decompose(&p, 7), Some(digits));
    }

    // Self-similarity: doubling a point prepends digit 0 — the
    // constellation contains itself at every scale.
    let digits = [137u8, 42, 200];
    let p = constellation::compose(&digits);
    let doubled: Point = std::array::from_fn(|k| 2 * p[k]);
    assert_eq!(
        constellation::decompose(&doubled, 4),
        Some(vec![0, 137, 42, 200])
    );

    // Non-lattice points refuse to decompose.
    assert_eq!(constellation::decompose(&[1, 0, 0, 0, 0, 0, 0, 0], 1), None);
}

#[test]
fn the_constellation_backend_conforms_at_every_width() {
    // Full-registry conformance at the default widths and with widths
    // pushed to 8, exactly like every other backend.
    let mut sim: Simulator = Simulator::new();
    register_constellation(&mut sim);
    let report = verify_backend(&sim, "e8-constellation", &ConformanceConfig::default()).unwrap();
    assert!(report.passed(), "{report}");
    assert_eq!(report.gate_checks.len(), sim.registry().len());
    assert_eq!(report.sampling_mismatches, 0);
    assert_eq!(report.collapse_violations, 0);
    let wide = ConformanceConfig {
        extra_widths: vec![0, 2, 7],
        random_circuits: 12,
        orderings: 2,
        ..ConformanceConfig::default()
    };
    let report = verify_backend(&sim, "e8-constellation", &wide).unwrap();
    assert!(report.passed(), "{report}");

    // Multi-block widths — past any single E8 copy: registry-drawn
    // random circuits at 10 and 12 qubits, amplitude-checked against
    // dense.
    for (n, seed) in [(10usize, 0xA1u64), (12, 0xA2)] {
        let circuit = random_registry_circuit(sim.registry(), n, 40, seed);
        let dense = sim.run_on("dense", &circuit).unwrap();
        let constellation = sim.run_on("e8-constellation", &circuit).unwrap();
        let dev = max_amplitude_deviation(dense.as_ref(), constellation.as_ref());
        assert!(dev < 1e-9, "width {n}: deviation {dev:.2e}");
        assert!((constellation.total_weight() - 1.0).abs() < 1e-9);
    }

    // The width wall is the trait's u64 index (same as sparse): 63
    // constructs, 64 refuses.
    assert!(E8ConstellationState::new(63).is_ok());
    assert!(matches!(
        E8ConstellationState::new(64),
        Err(Error::TooManyQubits { max: 63, .. })
    ));
}

#[test]
fn the_harness_prices_the_constellation_beside_the_others() {
    // Wide workloads through compare_backends: the constellation runs
    // everything the standard representations run, while the
    // single-copy e8-rep records its measured native-8 wall in the
    // same table.
    let mut sim: Simulator = Simulator::new();
    register_constellation(&mut sim);
    sim.backends_mut()
        .register("e8-rep", |n| {
            Ok(Box::new(quantsim::e8::rep::E8RepState::new(n)?))
        })
        .unwrap();
    let workloads = [
        Workload::from_circuit("ghz-16", library::ghz(16)),
        Workload::from_circuit("rainbow-16", library::rainbow(16)),
        Workload::from_circuit("qft-12", library::qft(12)),
    ];
    let backends = [
        "dense",
        "sparse",
        "factored",
        "mps",
        "e8-rep",
        "e8-constellation",
    ];
    let report = compare_backends(&sim, &workloads, &backends, &BenchConfig::default()).unwrap();
    assert_eq!(report.records.len(), workloads.len() * backends.len());
    assert!(report.max_deviation() < 1e-9, "{report}");
    for w in ["ghz-16", "rainbow-16", "qft-12"] {
        for backend in ["dense", "sparse", "factored", "mps", "e8-constellation"] {
            let r = report.record(w, backend).unwrap();
            assert!(r.error.is_none(), "{backend} on {w}: {:?}", r.error);
        }
        let walled = report.record(w, "e8-rep").unwrap();
        assert!(walled.error.is_some(), "e8-rep is 8-qubit-native");
    }

    // Measured structure: GHZ-16 is two constellation points, far
    // below dense's 2^16 amplitudes; saturated QFT-12 pays the honest
    // price — full support in 80-byte lattice keys costs MORE memory
    // than dense's flat vector.
    let ghz = report.record("ghz-16", "e8-constellation").unwrap();
    assert_eq!(ghz.nonzeros, 2);
    assert!(ghz.memory_bytes * 100 < report.record("ghz-16", "dense").unwrap().memory_bytes);
    let qft = report.record("qft-12", "e8-constellation").unwrap();
    assert_eq!(qft.nonzeros, 4096);
    assert!(
        qft.memory_bytes > report.record("qft-12", "dense").unwrap().memory_bytes,
        "saturated support in point keys is priced honestly"
    );
}

#[test]
fn a_63_qubit_ghz_is_two_constellation_points() {
    // Dense cannot even construct 63 qubits — measured refusal — while
    // the constellation stores the state as exactly two lattice
    // points of E8 at resolution 2⁸.
    assert!(DenseState::<C64>::new(63).is_err());

    let mut sim: Simulator = Simulator::new();
    register_constellation(&mut sim);
    let state = sim.run_on("e8-constellation", &library::ghz(63)).unwrap();
    assert_eq!(state.nonzero_count(), 2);
    let amp0 = state.amplitude(0);
    let amp1 = state.amplitude((1u64 << 63) - 1);
    assert_close(amp0.re, std::f64::consts::FRAC_1_SQRT_2, 1e-12);
    assert_close(amp1.re, std::f64::consts::FRAC_1_SQRT_2, 1e-12);

    let rep = state
        .as_any()
        .downcast_ref::<E8ConstellationState>()
        .unwrap();
    let points = rep.stored_points();
    assert!(points.contains(&[0i64; 8]), "|0…0⟩ is the origin");
    let ones = points.iter().find(|&&p| p != [0; 8]).unwrap();
    // |1…1⟩ = digits 0xff on seven full blocks + 0x7f on the 7-bit
    // top block — one point, eight scale levels deep.
    let mut digits = vec![0xffu8; 7];
    digits.push(0x7f);
    assert_eq!(*ones, constellation::compose(&digits));
    assert_eq!(constellation::decompose(ones, 8), Some(digits));
    // Measured geography: at every level the support is one center
    // digit (the origin branch) and one root-sphere digit (the ones
    // branch) — both 0xff and 0x7f are root classes.
    assert_eq!(rep.shell_census(), vec![[1, 1, 0]; 8]);
    // Two 80-byte points instead of 2^63 amplitudes.
    assert!(state.memory_bytes() < 1024, "{}", state.memory_bytes());
}
