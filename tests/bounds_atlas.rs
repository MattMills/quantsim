//! The boundary atlas, pinned: every representation's measured growth
//! law on its home fragment and on its escape family, the known
//! complexity results rediscovered from measurement alone, and the
//! advantage verdicts.
//!
//! Nothing here is asserted from theory: each law is classified from
//! measured byte costs over a size sweep; the theory is what the
//! numbers are then checked against.

mod common;

use quantsim::prelude::*;

fn base_of(law: &Law) -> f64 {
    match law {
        Law::Exponential { base } => *base,
        other => panic!("expected an exponential law, measured {other}"),
    }
}

fn axis<'a>(scan: &'a FamilyScan, name: &str) -> &'a AxisScan {
    scan.axes.iter().find(|a| a.axis == name).unwrap()
}

#[test]
fn the_law_classifier_is_calibrated() {
    // Synthetic ground truths first: the classifier must read flat,
    // polynomial and exponential data correctly before any physics
    // hangs off it.
    let sizes = [6usize, 8, 10, 12, 14];
    let flat: Vec<usize> = sizes.iter().map(|_| 125).collect();
    assert_eq!(classify_law(&sizes, &flat), Law::Constant);

    let cubic: Vec<usize> = sizes.iter().map(|&s| 3 * s * s * s).collect();
    match classify_law(&sizes, &cubic) {
        Law::Polynomial { degree } => assert!((degree - 3.0).abs() < 0.2, "{degree}"),
        other => panic!("cubic read as {other}"),
    }
    let doubling: Vec<usize> = sizes.iter().map(|&s| 7 << s).collect();
    match classify_law(&sizes, &doubling) {
        Law::Exponential { base } => assert!((base - 2.0).abs() < 0.05, "{base}"),
        other => panic!("doubling read as {other}"),
    }
}

#[test]
fn known_fragments_are_rediscovered_by_measurement() {
    // GHZ: classical via sparse (support 2 at every width, byte-flat),
    // while factored honestly pays 2^n for its one giant cluster.
    let ghz = advantage_scan("ghz", library::ghz, &[8, 10, 12, 14]);
    match &ghz.verdict {
        Verdict::Classical { via } => assert!(via.iter().any(|v| v == "sparse"), "{via:?}"),
        v => panic!("{v:?}"),
    }
    assert_eq!(axis(&ghz, "sparse").law, Some(Law::Constant));
    let factored_base = base_of(axis(&ghz, "factored").law.as_ref().unwrap());
    assert!((factored_base - 2.0).abs() < 0.15, "{factored_base}");

    // QFT on the |0…0⟩ boundary: the bond across every cut stays tiny
    // — the measured Aharonov-style result that basis-state QFT is
    // classically easy — while sparse pays the full 2^n support.
    let qft = advantage_scan("qft", library::qft, &[6, 8, 10, 12]);
    match &qft.verdict {
        Verdict::Classical { via } => assert!(via.iter().any(|v| v == "mps"), "{via:?}"),
        v => panic!("{v:?}"),
    }
    assert!(axis(&qft, "mps").law.as_ref().unwrap().is_subexponential());
    let sparse_base = base_of(axis(&qft, "sparse").law.as_ref().unwrap());
    assert!((sparse_base - 2.0).abs() < 0.1, "{sparse_base}");

    // Rainbow: classical via clustering; the fixed tree pays ~2^n at
    // its root and sparse pays exactly √2 per qubit (2^{n/2} support).
    let rainbow = advantage_scan("rainbow", library::rainbow, &[8, 10, 12, 14]);
    match &rainbow.verdict {
        Verdict::Classical { via } => {
            assert!(via.iter().any(|v| v == "factored"), "{via:?}");
        }
        v => panic!("{v:?}"),
    }
    let sqrt2 = base_of(axis(&rainbow, "sparse").law.as_ref().unwrap());
    assert!((sqrt2 - std::f64::consts::SQRT_2).abs() < 0.06, "{sqrt2}");
    let mera_base = base_of(axis(&rainbow, "mera").law.as_ref().unwrap());
    assert!(mera_base > 1.8, "the fixed tree pays the root: {mera_base}");

    // Clifford brickwork: Gottesman–Knill measured — the frame's cost
    // is polynomial (replay log), exact, at widths where sparse pays
    // 2^n.
    let clifford = advantage_scan(
        "clifford-brickwork",
        |n| library::brickwork(n, n, &(0..n).collect::<Vec<_>>()),
        &[8, 10, 12],
    );
    match &clifford.verdict {
        Verdict::Classical { via } => {
            assert!(via.iter().any(|v| v == "clifford-framed"), "{via:?}");
        }
        v => panic!("{v:?}"),
    }
    let cf = axis(&clifford, "clifford-framed");
    assert!(cf.exact && cf.law.as_ref().unwrap().is_subexponential());
}

#[test]
fn random_circuits_escape_every_assumption_at_once() {
    // The candidate family: universal random circuits at Θ(n²) gates.
    // EVERY representation's measured law is exponential — no
    // assumption in this crate holds — and every probe stayed exact,
    // so the escape is not an artifact of truncation.
    let scan = advantage_scan(
        "random",
        |n| library::random_circuit(n, 3 * n * n, 7),
        &[6, 8, 10, 12],
    );
    assert_eq!(scan.verdict, Verdict::Candidate);
    for a in &scan.axes {
        assert!(a.exact, "{} truncated — the escape must be exact", a.axis);
        let base = base_of(a.law.as_ref().unwrap());
        assert!(
            base > 1.4,
            "{} must grow exponentially on the candidate family: {base}",
            a.axis
        );
    }
    // Where the advantage would have to live, made operational: a new
    // representation whose axis stays sub-exponential HERE — while
    // exact — would be a discovered sub-exponential simulation. The
    // scan is the detector.
}

#[test]
fn each_representation_is_exponential_exactly_in_its_own_resource() {
    // The per-axis escape laws, measured directly.
    let reg = GateRegistry::<C64>::standard();

    // Sparse: cost doubles per branching qubit (H layer of size k on a
    // fixed 16-qubit register).
    let sizes = [6usize, 8, 10, 12];
    let costs: Vec<usize> = sizes
        .iter()
        .map(|&k| {
            let mut c: Circuit = Circuit::new(16);
            for q in 0..k {
                c.h(q);
            }
            let mut s = SparseState::<C64>::new(16).unwrap();
            c.bind(&reg).unwrap().run(&mut s).unwrap();
            s.memory_bytes()
        })
        .collect();
    let b = base_of(&classify_law(&sizes, &costs));
    assert!((b - 2.0).abs() < 0.1, "sparse escapes at 2^support: {b}");

    // Factored: cost doubles per qubit of the largest entangled
    // cluster (a bridged chain of size c inside a 16-qubit register;
    // sizes start at 8 so the fixed per-factor overhead is small
    // against the 2^c payload).
    let sizes = [8usize, 10, 12, 14];
    let costs: Vec<usize> = sizes
        .iter()
        .map(|&c_size| {
            let mut c: Circuit = Circuit::new(16);
            c.h(0);
            for q in 0..c_size - 1 {
                c.cx(q, q + 1);
            }
            c.h(15); // an untouched second cluster stays factored
            let mut s = FactoredState::<C64>::new(16).unwrap();
            c.bind(&reg).unwrap().run(&mut s).unwrap();
            assert_eq!(s.largest_factor_qubits(), c_size);
            s.memory_bytes()
        })
        .collect();
    let b = base_of(&classify_law(&sizes, &costs));
    assert!((b - 2.0).abs() < 0.15, "factored escapes at 2^cluster: {b}");

    // MPS: rotation + brick layers on 14 qubits — genuine volume
    // growth. A brick straddles a given cut only every other layer, so
    // the bond doubles per PAIR of layers (measured: 2, 2, 4, 4, …);
    // sweep double-layers and the law is cleanly exponential.
    let blocks = [1usize, 2, 3, 4];
    let mut bonds = Vec::new();
    let costs: Vec<usize> = blocks
        .iter()
        .map(|&b| {
            let n = 14;
            let mut c: Circuit = Circuit::new(n);
            for layer in 0..2 * b {
                for q in 0..n {
                    c.ry(q, 0.7 + 0.1 * (layer * n + q) as f64);
                }
                let start = layer % 2;
                let mut q = start;
                while q + 1 < n {
                    c.cx(q, q + 1);
                    q += 2;
                }
            }
            let mut s = MpsState::<C64>::with_config(
                n,
                MpsConfig {
                    max_bond: 4096,
                    trunc_tol: 1e-14,
                },
            )
            .unwrap();
            c.bind(&reg).unwrap().run(&mut s).unwrap();
            bonds.push(s.peak().0);
            s.memory_bytes()
        })
        .collect();
    assert_eq!(bonds, vec![2, 4, 8, 16], "bond doubles per double-layer");
    match classify_law(&blocks, &costs) {
        Law::Exponential { base } => assert!(base > 2.0, "{base}"),
        other => panic!("mps volume growth read as {other}"),
    }

    // Clifford frame: each `h(q); t(q)` on a fresh qubit conjugates the
    // T into an X-axis rotation on |0⟩ — a deterministic scatter, so
    // the stored support is exactly 2^t (the frame's 2^t bound
    // saturated from below). Note the contrast measured while building
    // this: T's placed where the frame's conjugation leaves them
    // diagonal scatter NOT AT ALL — the assumption doing its job.
    let ks = [2usize, 4, 6, 8];
    let n = 12;
    let supports: Vec<usize> = ks
        .iter()
        .map(|&t| {
            let mut c: Circuit = Circuit::new(n);
            for q in 0..t {
                c.h(q).t(q);
            }
            let mut s = CliffordFramedState::<C64>::new(n).unwrap();
            c.bind(&reg).unwrap().run(&mut s).unwrap();
            s.peak_stored_support()
        })
        .collect();
    assert_eq!(supports, vec![4, 16, 64, 256], "2^t exactly: {supports:?}");
    match classify_law(&ks, &supports) {
        Law::Exponential { base } => assert!((base - 2.0).abs() < 0.05, "{base}"),
        other => panic!("T-scatter read as {other}"),
    }

    // Ball: the certified radius grows by √2 per Hadamard — precision
    // is a resource with its own measured exponential law.
    let mut radii = Vec::new();
    for depth in [4usize, 8, 12, 16] {
        let mut c: Circuit<Ball> = Circuit::new(1);
        for _ in 0..depth {
            c.h(0);
        }
        let state = Simulator::<Ball>::new().run(&c).unwrap();
        radii.push(state.amplitude(0).rad);
    }
    for w in radii.windows(2) {
        let ratio = (w[1] / w[0]).powf(0.25); // per single H
        assert!(
            (ratio - std::f64::consts::SQRT_2).abs() < 0.25,
            "radius grows ~√2 per H: {ratio}"
        );
    }
}

#[test]
fn time_laws_join_the_verdict() {
    // An axis certifies a family classical only when BOTH its memory
    // and its wall-clock law stay sub-exponential — a memory-cheap but
    // time-exponential representation no longer slips through.
    let ghz = advantage_scan("ghz", library::ghz, &[8, 10, 12, 14]);
    let sparse = axis(&ghz, "sparse");
    assert!(sparse.time_law.as_ref().unwrap().is_subexponential());
    assert!(sparse.certifies_classical());
    let dense_time = base_of(axis(&ghz, "dense").time_law.as_ref().unwrap());
    assert!(dense_time > 1.4, "dense pays time too: {dense_time}");

    let random = advantage_scan(
        "random",
        |n| library::random_circuit(n, 3 * n * n, 7),
        &[6, 8, 10, 12],
    );
    assert_eq!(random.verdict, Verdict::Candidate);
    for name in ["dense", "sparse"] {
        let t = base_of(axis(&random, name).time_law.as_ref().unwrap());
        assert!(t > 1.3, "{name} time on the candidate family: {t}");
    }
    // The measured case FOR the rule: long-range IQP below has an axis
    // (mps) whose time law is polynomial while its memory law is
    // exponential — sub-exponential on one ledger only, certifying
    // nothing.
}

#[test]
fn interaction_range_flips_the_iqp_verdict() {
    // Same commuting IQP core, one structural knob: nearest-neighbour
    // couplings keep the linear-cut assumption alive; long-range
    // couplings escape every representation at once.
    let long = advantage_scan(
        "iqp-long",
        |n| library::iqp(n, 2 * n, true, 5),
        &[6, 8, 10, 12],
    );
    assert_eq!(long.verdict, Verdict::Candidate);
    let mps_long = axis(&long, "mps");
    assert!(
        !mps_long.law.as_ref().unwrap().is_subexponential(),
        "long-range couplings defeat the cut assumption"
    );

    let nn = advantage_scan(
        "iqp-nn",
        |n| library::iqp(n, 2 * n, false, 5),
        &[6, 8, 10, 12],
    );
    match &nn.verdict {
        Verdict::Classical { via } => {
            assert!(via.iter().any(|v| v == "mps" || v == "factored"), "{via:?}")
        }
        v => panic!("{v:?}"),
    }
    assert!(axis(&nn, "mps").certifies_classical());
}

#[test]
fn magic_doping_meets_the_frames_measured_reach() {
    // t = O(log n) doping: the stabilizer frame certifies, at constant
    // measured cost, while clustering/cut axes read polynomial and
    // sparse/dense escape.
    let light = advantage_scan(
        "doped-log",
        |n| library::doped_clifford(n, 5 * n, n.ilog2() as usize, 9),
        &[8, 10, 12, 14],
    );
    match &light.verdict {
        Verdict::Classical { via } => {
            assert!(via.iter().any(|v| v == "clifford-framed"), "{via:?}")
        }
        v => panic!("{v:?}"),
    }
    assert_eq!(axis(&light, "clifford-framed").law, Some(Law::Constant));

    // t = n/2 doping, measured honestly: the frame STILL certifies —
    // and at these sizes it is the ONLY axis that does — because T's
    // dropped into a random Clifford stream mostly land where the
    // frame's conjugation keeps them diagonal (zero scatter). The 2^t
    // escape needs the deterministic h;t scatter pinned in
    // each_representation_is_exponential_exactly_in_its_own_resource —
    // the boundary of the frame's assumption is about WHERE the magic
    // sits, not how much of it there is.
    let heavy = advantage_scan(
        "doped-heavy",
        |n| library::doped_clifford(n, 5 * n, n / 2, 9),
        &[8, 10, 12],
    );
    match &heavy.verdict {
        Verdict::Classical { via } => {
            assert!(
                via.iter().any(|v| v == "clifford-framed"),
                "the frame certifies heavy random doping: {via:?}"
            );
        }
        v => panic!("{v:?}"),
    }
    assert_eq!(
        axis(&heavy, "clifford-framed").law,
        Some(Law::Constant),
        "measured: random-stream T's cost the frame nothing"
    );
}

#[test]
fn shallow_2d_assumptions_fail_slowly() {
    // Constant-depth 2D brickwork over growing area: entanglement
    // across a cut grows with the region BOUNDARY, not its volume, so
    // the cut-rank axes read as low-degree polynomial at probe sizes
    // (the 2^√n regime — sub-exponential here, and visibly not flat)
    // while dense and sparse pay the full 2^n.
    let scan = advantage_scan(
        "shallow-2d",
        |n| library::brickwork_2d(n / 3, 3, 3, 4),
        &[9, 12, 15, 18],
    );
    match &scan.verdict {
        Verdict::Classical { via } => {
            assert!(via.iter().any(|v| v == "mps" || v == "factored"), "{via:?}")
        }
        v => panic!("{v:?}"),
    }
    match axis(&scan, "mps").law.as_ref().unwrap() {
        Law::Polynomial { degree } => {
            assert!(*degree > 1.0, "the boundary law is not flat: {degree}")
        }
        Law::Exponential { base } => {
            assert!(
                *base < 1.5,
                "if it reads exponential the base is small: {base}"
            )
        }
        Law::Constant => panic!("depth-3 2D entanglement is not free"),
    }
    let dense_base = base_of(axis(&scan, "dense").law.as_ref().unwrap());
    assert!((dense_base - 2.0).abs() < 0.1, "{dense_base}");
}

#[test]
fn selection_extrapolates_measured_laws_and_verifies() {
    // Selection by extrapolated scaling, checked against a holdout run
    // the fit never saw.
    let sel = select_by_scaling("ghz", library::ghz, &[8, 10, 12, 14], 40);
    assert_eq!(sel.best().axis, "sparse");
    assert!(sel.subexponential);
    let reg = GateRegistry::<C64>::standard();
    let mut holdout = SparseState::<C64>::new(40).unwrap();
    library::ghz(40)
        .bind(&reg)
        .unwrap()
        .run(&mut holdout)
        .unwrap();
    let measured = holdout.memory_bytes() as f64;
    let predicted = sel.best().predicted_bytes;
    assert!(
        (predicted - measured).abs() / measured < 0.5,
        "prediction {predicted:.0} vs holdout {measured}"
    );

    // QFT: some sub-exponential axis wins, and the first ranked choice
    // that completes the holdout verifies within a small factor. (A
    // choice can wall at the holdout size — mera's runtime cliff is
    // nonlinear — and the ranked list absorbs that honestly.)
    let sel = select_by_scaling("qft", library::qft, &[6, 8, 10, 12], 16);
    assert!(sel.subexponential, "{:?}", sel.best());
    let holdout_profile = resource_profile(&library::qft(16));
    let verified = sel.choices.iter().find_map(|choice| {
        holdout_profile
            .axes
            .iter()
            .find(|a| a.axis == choice.axis)
            .and_then(|a| a.cost)
            .map(|measured| (choice, measured as f64))
    });
    let (choice, measured) = verified.expect("some ranked choice completes the holdout");
    assert!(choice.law.is_subexponential(), "{:?}", choice);
    let ratio = choice.predicted_bytes / measured;
    assert!(
        (0.2..=5.0).contains(&ratio),
        "{}: predicted {:.0} vs measured {measured} (ratio {ratio:.2})",
        choice.axis,
        choice.predicted_bytes
    );

    // The candidate family: no assumption holds, and the selection
    // says so — the winner is only least-bad, and its prediction still
    // lands within a small factor of the dense truth.
    let sel = select_by_scaling(
        "random",
        |n| library::random_circuit(n, 3 * n * n, 7),
        &[6, 8, 10, 12],
        16,
    );
    assert!(
        !sel.subexponential,
        "no assumption holds on random circuits"
    );
    let truth = ((1u64 << 16) * 16) as f64;
    let ratio = sel.best().predicted_bytes / truth;
    assert!(
        (0.2..=5.0).contains(&ratio),
        "least-bad prediction sanity: {ratio:.2}"
    );
}

#[test]
fn the_walls_are_measured_refusals() {
    // The dense wall: 2^34 amplitudes is 256 GiB — the guard refuses
    // with the measured request and the measured availability, not a
    // precomputed skip.
    match DenseState::<C64>::new(34) {
        Err(Error::OutOfMemory {
            requested,
            available,
            ..
        }) => {
            assert_eq!(requested, (1usize << 34) * 16);
            assert!(available > 0, "availability is measured");
            assert!(available < requested);
        }
        other => panic!("expected a measured refusal, got {other:?}"),
    }
    // The indexing wall is a different kind: structural, not resource.
    assert!(SparseState::<C64>::new(63).is_ok());
    assert!(SparseState::<C64>::new(64).is_err());
    // And a profile records walls instead of failing: at width 34 the
    // dense axis reports its refusal while sparse proceeds.
    let profile = resource_profile(&library::ghz(34));
    let dense = profile.axes.iter().find(|a| a.axis == "dense").unwrap();
    assert!(dense.cost.is_none());
    assert!(dense.note.as_ref().unwrap().contains("out of memory"));
    let sparse = profile.axes.iter().find(|a| a.axis == "sparse").unwrap();
    assert!(sparse.cost.is_some());
    assert_eq!(sparse.parameter, "support 2");
}
