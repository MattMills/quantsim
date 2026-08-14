//! The boundary atlas, pinned: every representation's measured growth
//! law on its home fragment and on its escape family, the known
//! complexity results rediscovered from measurement alone, and the
//! advantage verdicts.
//!
//! Nothing here is asserted from theory: each law is classified from
//! measured byte costs over a size sweep; the theory is what the
//! numbers are then checked against.
//!
//! The sweeps here run at the **smallest widths that still resolve a
//! law** — these tests pin the atlas's logic, not its reach. The atlas
//! itself is a measurement program and belongs in a runnable: the
//! full-width sweep over the same families, printed as tables, is
//! `examples/advantage_bounds.rs`.

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
    let ghz = advantage_scan("ghz", library::ghz, &[6, 8, 10, 12]);
    match &ghz.verdict {
        Verdict::Classical { via } => assert_eq!(
            via.iter().any(|v| v == "sparse"),
            axis(&ghz, "sparse").certifies_classical(),
            "the verdict names exactly the axes that certify: {via:?}"
        ),
        v => panic!("{v:?}"),
    }
    assert_eq!(axis(&ghz, "sparse").law, Some(Law::Constant));
    let factored_base = base_of(axis(&ghz, "factored").law.as_ref().unwrap());
    assert!((factored_base - 2.0).abs() < 0.15, "{factored_base}");

    // QFT on the |0…0⟩ boundary: the bond across every cut stays tiny
    // — the measured Aharonov-style result that basis-state QFT is
    // classically easy — while sparse pays the full 2^n support.
    let qft = advantage_scan("qft", library::qft, &[5, 6, 8, 10]);
    assert!(
        matches!(qft.verdict, Verdict::Classical { .. }),
        "some assumption must hold on basis-state QFT: {:?}",
        qft.verdict
    );
    // The bond claim is about MEMORY, so that is asserted unconditionally.
    // Membership in `via` additionally needs a sub-exponential wall-clock
    // law, which is a fitted shape over four sizes and moves with machine
    // load — gated on the timing being variance-robust.
    let mps = axis(&qft, "mps");
    assert!(mps.law.as_ref().unwrap().is_subexponential());
    assert!(mps.exact, "and the bond never truncated");
    if mps.time_law_is_variance_robust() {
        match &qft.verdict {
            Verdict::Classical { via } => assert_eq!(
                via.iter().any(|v| v == "mps"),
                axis(&qft, "mps").certifies_classical(),
                "the verdict names exactly the axes that certify: {via:?}"
            ),
            v => panic!("{v:?}"),
        }
    }
    let sparse_base = base_of(axis(&qft, "sparse").law.as_ref().unwrap());
    assert!((sparse_base - 2.0).abs() < 0.1, "{sparse_base}");

    // Rainbow: classical via clustering; the fixed tree pays ~2^n at
    // its root and sparse pays exactly √2 per qubit (2^{n/2} support).
    let rainbow = advantage_scan("rainbow", library::rainbow, &[6, 8, 10, 12]);
    match &rainbow.verdict {
        Verdict::Classical { via } => {
            assert_eq!(
                via.iter().any(|v| v == "factored"),
                axis(&rainbow, "factored").certifies_classical(),
                "the verdict names exactly the axes that certify: {via:?}"
            );
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
        &[6, 8, 10],
    );
    // The Gottesman–Knill claim is about DESCRIPTION SIZE, so that is what
    // gets asserted unconditionally: the frame's memory law is
    // sub-exponential and every probe exact, at widths where sparse pays
    // 2^n. The `via` list additionally requires a sub-exponential
    // wall-clock law, which is a fitted shape over three sizes and can be
    // flipped by machine load — so it is asserted only where the timing
    // measurement is variance-robust.
    let cf = axis(&clifford, "clifford-framed");
    assert!(
        cf.exact,
        "the frame must be exact for the claim to mean anything"
    );
    assert!(
        cf.law.as_ref().unwrap().is_subexponential(),
        "the frame's memory law is the Gottesman-Knill statement: {:?}",
        cf.law
    );
    let sparse_base = base_of(axis(&clifford, "sparse").law.as_ref().unwrap());
    assert!(
        sparse_base > 1.3,
        "and sparse pays exponentially on the same family: {sparse_base}"
    );
    // The atlas's rule, asserted as the IMPLICATION rather than as the
    // timing fit that feeds it. Whether this machine's wall-clock fit
    // lands sub-exponential over a four-point sweep moves with load —
    // that is a resolution limit, and a variance-robustness guard is not
    // enough to hide it, because a fit can be robustly wrong. What
    // cannot move is the contract: an axis is named in the verdict
    // exactly when both of its ledgers certify.
    match &clifford.verdict {
        Verdict::Classical { via } => assert_eq!(
            via.iter().any(|v| v == "clifford-framed"),
            cf.certifies_classical(),
            "the verdict must name exactly the axes that certify: {via:?}"
        ),
        v => panic!("{v:?}"),
    }
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
        &[5, 6, 8, 10],
    );
    assert_eq!(scan.verdict, Verdict::Candidate);
    let mut priced = 0usize;
    let mut declined = 0usize;
    for a in &scan.axes {
        if a.costs.iter().any(|c| c.is_none()) {
            // A representation that cannot hold this family at all has
            // not *truncated* it — it has declined it, and said so. The
            // graph-state bundle does exactly that: a random universal
            // circuit is full of non-Clifford gates and it refuses them
            // by name. Only a silent approximation would undermine the
            // escape, so that is what the exactness check is for.
            declined += 1;
            continue;
        }
        priced += 1;
        assert!(a.exact, "{} truncated — the escape must be exact", a.axis);
        let base = base_of(a.law.as_ref().unwrap());
        assert!(
            base > 1.4,
            "{} must grow exponentially on the candidate family: {base}",
            a.axis
        );
    }
    assert!(
        priced >= 8,
        "only {priced} axes priced the candidate family"
    );
    assert!(
        declined <= 2,
        "{declined} axes declined; the candidate family should be holdable by most"
    );
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
    let ghz = advantage_scan("ghz", library::ghz, &[6, 8, 10, 12]);
    let sparse = axis(&ghz, "sparse");
    assert!(sparse.time_law.as_ref().unwrap().is_subexponential());
    assert!(sparse.certifies_classical());
    // Dense pays time as well as memory. Asserted as measured GROWTH, not
    // as a fitted base: over four sizes a steep polynomial and a shallow
    // exponential are numerically adjacent, and which label the fit picks
    // moves with machine load. The growth ratio does not.
    let dense_time = axis(&ghz, "dense")
        .measured_time_growth()
        .expect("dense finished every size");
    assert!(
        dense_time > 4.0,
        "dense should pay several-fold more time from 6 to 12 qubits: {dense_time:.2}x"
    );

    let random = advantage_scan(
        "random",
        |n| library::random_circuit(n, 3 * n * n, 7),
        &[5, 6, 8, 10],
    );
    assert_eq!(random.verdict, Verdict::Candidate);
    for name in ["dense", "sparse"] {
        let scan = axis(&random, name);
        let growth = scan
            .measured_time_growth()
            .expect("both finished every size");
        assert!(
            growth > 3.0,
            "{name} time on the candidate family should climb steeply from 5 to 10 \
             qubits: {growth:.2}x"
        );
        // The *shape* is deliberately not asserted, and the reason is a
        // resolution limit rather than jitter: over a sweep spanning 5 to
        // 10 qubits, `size^5.8` is a factor of 56 and `2^size` is a factor
        // of 32. The data does not separate them, and the classifier reads
        // one or the other *robustly* depending on the machine. The verdict
        // above does not depend on it — the memory law carries `Candidate`
        // — so the honest test asserts growth and leaves shape alone.
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
        &[5, 6, 8, 10],
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
        &[5, 6, 8, 10],
    );
    match &nn.verdict {
        Verdict::Classical { via } => assert_eq!(
            via.iter().any(|v| v == "mps" || v == "factored"),
            axis(&nn, "mps").certifies_classical() || axis(&nn, "factored").certifies_classical(),
            "the verdict names exactly the axes that certify: {via:?}"
        ),
        v => panic!("{v:?}"),
    }
    // The cut assumption surviving is a statement about MEMORY, and the
    // memory ledger is bytes rather than wall-clock, so it is the half of
    // the claim that can be pinned outright.
    assert!(
        axis(&nn, "mps").law.as_ref().unwrap().is_subexponential(),
        "nearest-neighbour couplings keep the cut assumption alive: {:?}",
        axis(&nn, "mps").law
    );
}

#[test]
fn magic_doping_meets_the_frames_measured_reach() {
    // t = O(log n) doping: the stabilizer frame certifies, at constant
    // measured cost, while clustering/cut axes read polynomial and
    // sparse/dense escape.
    let light = advantage_scan(
        "doped-log",
        |n| library::doped_clifford(n, 5 * n, n.ilog2() as usize, 9),
        &[6, 8, 10],
    );
    match &light.verdict {
        Verdict::Classical { via } => {
            assert_eq!(
                via.iter().any(|v| v == "clifford-framed"),
                axis(&light, "clifford-framed").certifies_classical(),
                "the verdict names exactly the axes that certify: {via:?}"
            )
        }
        v => panic!("{v:?}"),
    }
    // The memory ledger is deterministic — bytes, not wall-clock — so
    // this is the part of the claim that can be pinned outright.
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
        &[5, 6, 8],
    );
    match &heavy.verdict {
        Verdict::Classical { via } => {
            assert_eq!(
                via.iter().any(|v| v == "clifford-framed"),
                axis(&heavy, "clifford-framed").certifies_classical(),
                "the verdict names exactly the axes that certify: {via:?}"
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
        &[6, 9, 12],
    );
    match &scan.verdict {
        Verdict::Classical { via } => assert_eq!(
            via.iter().any(|v| v == "mps" || v == "factored"),
            axis(&scan, "mps").certifies_classical()
                || axis(&scan, "factored").certifies_classical(),
            "the verdict names exactly the axes that certify: {via:?}"
        ),
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
    let sel = select_by_scaling("ghz", library::ghz, &[6, 8, 10, 12], 40);
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
    let sel = select_by_scaling("qft", library::qft, &[5, 6, 8, 10], 12);
    assert!(sel.subexponential, "{:?}", sel.best());
    let holdout_profile = resource_profile(&library::qft(12));
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
        &[5, 6, 8, 10],
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
fn time_laws_carry_error_bars_and_shapes_join_the_scan() {
    // Variance-aware timing: each probe's law comes with min/max
    // envelope laws, and a classification is variance-robust when the
    // envelopes agree with the median.
    //
    // This test deliberately does NOT assert that *this machine* produced
    // a robust timing measurement. Under CPU contention no wall-clock
    // measurement is robust, and a test asserting otherwise measures the
    // load rather than the library — that is exactly how this suite used
    // to flake. What is asserted instead is the mechanism: the envelope is
    // always present, the predicate is exactly what it claims to be, and a
    // *clean* measurement must come out robust.
    let ghz = advantage_scan("ghz", library::ghz, &[6, 8, 10, 12]);
    let sparse = axis(&ghz, "sparse");
    let dense = axis(&ghz, "dense");

    for scan in [sparse, dense] {
        assert!(
            scan.time_law_bounds.is_some(),
            "{}: a finished axis always carries its envelope",
            scan.axis
        );
        // The predicate is load-independent: it is a statement about three
        // laws the scan already holds.
        let (median, (lo, hi)) = (
            scan.time_law.as_ref().unwrap(),
            scan.time_law_bounds.as_ref().unwrap(),
        );
        let expected = median.is_subexponential() == lo.is_subexponential()
            && median.is_subexponential() == hi.is_subexponential();
        assert_eq!(
            scan.time_law_is_variance_robust(),
            expected,
            "{}: variance-robustness must be exactly envelope agreement",
            scan.axis
        );
        // And when the timing really was clean, robustness must follow.
        let jitter = scan
            .worst_time_spread_ratio()
            .expect("a finished axis has a spread at every size");
        if jitter < 2.0 {
            assert!(
                scan.time_law_is_variance_robust(),
                "{}: a {jitter:.2}x timing spread is clean, so the classification \
                 should be robust: {:?}",
                scan.axis,
                scan.time_law_bounds
            );
        }
    }

    // Register SHAPES are first-class axes now: the hierarchical
    // splits appear in every profile with their own measured laws —
    // constant on GHZ (site support 2), exponential on the candidate
    // family, exact throughout.
    for name in ["algebraic-h", "algebraic-o"] {
        let a = axis(&ghz, name);
        assert!(a.exact);
        assert_eq!(a.law, Some(Law::Constant), "{name} on ghz");
    }
    let random = advantage_scan(
        "random",
        |n| library::random_circuit(n, 3 * n * n, 7),
        &[5, 6, 8, 10],
    );
    assert_eq!(random.verdict, Verdict::Candidate);
    for name in ["algebraic-h", "algebraic-o"] {
        let base = base_of(axis(&random, name).law.as_ref().unwrap());
        assert!(base > 1.4, "{name} escapes with everything else: {base}");
    }

    // And the scan MEASURES when a shape beats its flat counterpart:
    // a family whose algebra sector is dense while the site sector
    // stays sparse stores fewer, fatter entries — the algebraic axis
    // is cheaper than plain sparse, axis against axis.
    let mixed = |n: usize| -> Circuit {
        let mut c: Circuit = Circuit::new(n);
        for q in 0..5 {
            c.h(q);
        }
        for q in n - 2..n {
            c.h(q);
        }
        c.cx(0, n - 1).cx(1, n - 2);
        c
    };
    let profile = resource_profile(&mixed(14));
    let sparse_cost = profile
        .axes
        .iter()
        .find(|a| a.axis == "sparse")
        .unwrap()
        .cost
        .unwrap();
    let shaped_cost = profile
        .axes
        .iter()
        .find(|a| a.axis == "algebraic-o")
        .unwrap()
        .cost
        .unwrap();
    assert!(
        shaped_cost < sparse_cost,
        "the shape wins on mixed-sector states: {shaped_cost} vs {sparse_cost}"
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
    // That a *profile* records the wall instead of failing on it is the
    // same guard seen from one level up — but reading it costs a full
    // sweep of every axis at width 34, which is a measurement and not an
    // assertion. It is printed by `examples/advantage_bounds.rs`.
}
