//! Clifford frames: the Clifford part of a circuit becomes metadata
//! (tableau + replay log), rotations conjugate through the tableau onto
//! native sparse Pauli rotations, and the measured cost currency of the
//! stored state is the T-count — not the width, not the gate count.

mod common;

use common::*;
use quantsim::backend::{CLIFFORD_DIAGONAL_MAX, CLIFFORD_RECOGNITION_MAX};
use quantsim::prelude::*;
use std::f64::consts::FRAC_1_SQRT_2;

fn sim_with_clifford() -> Simulator {
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("clifford-framed", |n| {
            Ok(Box::new(CliffordFramedState::<C64>::new(n)?))
        })
        .unwrap();
    sim
}

/// Pauli-string matrix over C64: `labels[b]` acts on sub-index bit `b`.
fn pauli_matrix(labels: &[char]) -> GateMatrix<C64> {
    let single = |ch: char| -> GateMatrix<C64> {
        let entries = match ch {
            'i' => vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 0.0)],
            'x' => vec![c64(0.0, 0.0), c64(1.0, 0.0), c64(1.0, 0.0), c64(0.0, 0.0)],
            'y' => vec![c64(0.0, 0.0), c64(0.0, -1.0), c64(0.0, 1.0), c64(0.0, 0.0)],
            'z' => vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(-1.0, 0.0)],
            _ => unreachable!("unknown pauli label"),
        };
        GateMatrix::from_vec(2, entries).unwrap()
    };
    let mut acc = single(labels[labels.len() - 1]);
    for b in (0..labels.len() - 1).rev() {
        acc = acc.kron(&single(labels[b]));
    }
    acc
}

/// `exp(−iθ/2·P)` as a dense matrix: `cos(θ/2)·I − i·sin(θ/2)·P`.
fn rotation_matrix(labels: &[char], theta: f64) -> GateMatrix<C64> {
    let p = pauli_matrix(labels);
    let d = p.dim();
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    let mut m = GateMatrix::<C64>::zeros(d).unwrap();
    for r in 0..d {
        for col in 0..d {
            let ident = if r == col { c64(c, 0.0) } else { c64(0.0, 0.0) };
            m.set(r, col, ident + c64(0.0, -s) * p.get(r, col));
        }
    }
    m
}

fn masks(labels: &[char], qubits: &[usize]) -> (u64, u64) {
    let (mut x, mut z) = (0u64, 0u64);
    for (b, &ch) in labels.iter().enumerate() {
        match ch {
            'x' => x |= 1 << qubits[b],
            'y' => {
                x |= 1 << qubits[b];
                z |= 1 << qubits[b];
            }
            'z' => z |= 1 << qubits[b],
            _ => {}
        }
    }
    (x, z)
}

#[test]
fn native_pauli_rotations_match_dense_matrices() {
    // The kernel underneath the whole backend: exp(−iθ/2·σP) applied
    // natively to the sparse map must equal the dense matrix applied to
    // the same scrambled state, for every Pauli mixture and both signs.
    let n = 4;
    let reference = run_dense(&scrambler(n));
    let mut entries = Vec::new();
    reference.for_each_nonzero(&mut |i, a| entries.push((i, a)));

    let cases: &[(&[char], &[usize])] = &[
        (&['x'], &[2]),
        (&['y'], &[0]),
        (&['z'], &[3]),
        (&['x', 'z'], &[0, 2]),
        (&['y', 'y'], &[1, 3]),
        (&['z', 'z'], &[0, 3]),
        (&['x', 'y', 'z'], &[0, 1, 3]),
        (&['y', 'x', 'z', 'y'], &[0, 1, 2, 3]),
    ];
    for (case, &(labels, qubits)) in cases.iter().enumerate() {
        for negate in [false, true] {
            let theta = 0.37 + 0.41 * case as f64;
            let mut sparse = SparseState::<C64>::new(n).unwrap();
            sparse.load(&entries).unwrap();
            let (x, z) = masks(labels, qubits);
            sparse.apply_pauli_rotation(theta, x, z, negate).unwrap();

            let mut dense = DenseState::<C64>::new(n).unwrap();
            dense.load(&entries).unwrap();
            let signed = if negate { -theta } else { theta };
            dense
                .apply(&rotation_matrix(labels, signed), qubits)
                .unwrap();

            let dev = max_amplitude_deviation(&dense, &sparse);
            assert!(
                dev < 1e-12,
                "{labels:?} on {qubits:?} negate={negate}: deviation {dev}"
            );
        }
    }
}

#[test]
fn pauli_rotations_over_reals_exist_exactly_when_the_string_is_real() {
    // exp(−iθ/2·P) is a real matrix iff P has an odd number of Y factors
    // (the −i·sin coefficient meets i^{odd}). The rebit backend must get
    // native ry/ryyy but refuse rx/rz — same rule as the gate registry.
    let mut s = SparseState::<f64>::new(2).unwrap();
    s.apply_pauli_rotation(0.7, 0b01, 0b01, false).unwrap(); // Y on q0
    let mut d = DenseState::<f64>::new(2).unwrap();
    let th: f64 = 0.7;
    let (c, sn) = ((th / 2.0).cos(), (th / 2.0).sin());
    let ry = GateMatrix::from_vec(2, vec![c, -sn, sn, c]).unwrap();
    d.apply(&ry, &[0]).unwrap();
    assert!(max_amplitude_deviation(&d, &s) < 1e-12);

    let mut s = SparseState::<f64>::new(2).unwrap();
    assert!(matches!(
        s.apply_pauli_rotation(0.7, 0b01, 0, false), // X: needs −i·sin
        Err(Error::UnsupportedForAlgebra { .. })
    ));
    assert!(matches!(
        s.apply_pauli_rotation(0.7, 0, 0b01, false), // Z: needs e^{±iθ/2}
        Err(Error::UnsupportedForAlgebra { .. })
    ));
}

#[test]
fn clifford_framed_conforms_over_the_full_registry() {
    // Recognition, tableau conjugation, Walsh splitting, ZYZ fallback and
    // flush-on-observe must be invisible to physics over every registered
    // gate — including the ones that force raw fallbacks (ccx, cswap).
    let sim = sim_with_clifford();
    let cfg = ConformanceConfig {
        random_circuits: 10,
        orderings: 2,
        ..ConformanceConfig::default()
    };
    let report = verify_backend(&sim, "clifford-framed", &cfg).unwrap();
    assert!(report.passed(), "{report}");
    assert!(report.max_amplitude_deviation < 1e-9, "{report}");
    assert_eq!(report.sampling_mismatches, 0);
    assert_eq!(report.collapse_violations, 0);
}

#[test]
fn clifford_circuits_stay_support_one_at_any_width() {
    // Gottesman–Knill through the frame mechanism: a pure Clifford stream
    // never touches the stored state, at widths far past dense.
    let n = 40;
    let reg = GateRegistry::<C64>::standard();
    let mut circuit: Circuit = Circuit::new(n);
    let mut rng = Prng::new(11);
    for _ in 0..200 {
        let q = (rng.next_u64() % n as u64) as usize;
        let r = {
            let mut r = (rng.next_u64() % n as u64) as usize;
            while r == q {
                r = (rng.next_u64() % n as u64) as usize;
            }
            r
        };
        match rng.next_u64() % 6 {
            0 => circuit.h(q),
            1 => circuit.s(q),
            2 => circuit.x(q),
            3 => circuit.z(q),
            4 => circuit.cx(q, r),
            _ => circuit.cz(q, r),
        };
    }
    let bound = circuit.bind(&reg).unwrap();
    let mut state = CliffordFramedState::<C64>::new(n).unwrap();
    bound.run(&mut state).unwrap();
    assert_eq!(
        state.stored_nonzero_count(),
        1,
        "stored state never touched"
    );
    assert_eq!(state.peak_stored_support(), 1);
    assert_eq!(state.frame_gates(), 200);
    let stats = state.stats();
    assert_eq!(stats.absorbed_clifford, 200);
    assert_eq!(stats.flushes, 0);
    assert_eq!(stats.raw_fallbacks, 0);

    // Same mechanism on a verifiable circuit: GHZ(40) exact amplitudes.
    let mut ghz = CliffordFramedState::<C64>::new(n).unwrap();
    library::ghz::<C64>(n)
        .bind(&reg)
        .unwrap()
        .run(&mut ghz)
        .unwrap();
    assert_eq!(ghz.stored_nonzero_count(), 1, "pre-flush support is 1");
    let a0 = ghz.amplitude(0); // observation flushes the 40-gate log
    let a1 = ghz.amplitude((1u64 << n) - 1);
    assert!(a0.approx_eq(c64(FRAC_1_SQRT_2, 0.0), TOL));
    assert!(a1.approx_eq(c64(FRAC_1_SQRT_2, 0.0), TOL));
    assert_eq!(ghz.stats().replayed_gates, n, "flush replayed the log");

    // The physics is right at verifiable width: same stream shape at n=10
    // against dense, amplitude for amplitude.
    let n = 10;
    let mut circuit: Circuit = Circuit::new(n);
    let mut rng = Prng::new(12);
    for _ in 0..120 {
        let q = (rng.next_u64() % n as u64) as usize;
        let r = ((q + 1) + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
        match rng.next_u64() % 6 {
            0 => circuit.h(q),
            1 => circuit.s(q),
            2 => circuit.x(q),
            3 => circuit.z(q),
            4 => circuit.cx(q, r),
            _ => circuit.cz(q, r),
        };
    }
    let bound = circuit.bind(&reg).unwrap();
    let mut framed = CliffordFramedState::<C64>::new(n).unwrap();
    bound.run(&mut framed).unwrap();
    let mut dense = DenseState::<C64>::new(n).unwrap();
    bound.run(&mut dense).unwrap();
    let dev = max_amplitude_deviation(&dense, &framed);
    assert!(dev < 1e-9, "clifford stream deviates: {dev}");
}

/// Independent Clifford-membership ground truth, sharing **no code** with
/// the backend's recognizer: `M` is Clifford iff it normalizes the Pauli
/// group, iff every conjugated generator `M†·P_b·M` equals ±(some Pauli
/// string) — checked here by dense matrix products and entrywise
/// comparison against every signed candidate.
fn is_clifford_dense(m: &GateMatrix<C64>, k: usize) -> bool {
    let d = 1usize << k;
    let labels_of = |px: usize, pz: usize| -> Vec<char> {
        (0..k)
            .map(|b| match ((px >> b) & 1, (pz >> b) & 1) {
                (0, 0) => 'i',
                (1, 0) => 'x',
                (0, 1) => 'z',
                _ => 'y',
            })
            .collect()
    };
    let dag = m.dagger();
    for b in 0..k {
        for (px, pz) in [(1usize << b, 0usize), (0, 1usize << b)] {
            let gen = pauli_matrix(&labels_of(px, pz));
            let a = dag.matmul(&gen).matmul(m);
            let mut matched = false;
            'candidates: for cx in 0..d {
                for cz in 0..d {
                    let cand = pauli_matrix(&labels_of(cx, cz));
                    for sign in [1.0, -1.0] {
                        let mut ok = true;
                        'entries: for r in 0..d {
                            for c in 0..d {
                                if (a.get(r, c) - cand.get(r, c).scale(sign)).norm() > 1e-9 {
                                    ok = false;
                                    break 'entries;
                                }
                            }
                        }
                        if ok {
                            matched = true;
                            break 'candidates;
                        }
                    }
                }
            }
            if !matched {
                return false;
            }
        }
    }
    true
}

#[test]
fn absorption_is_exactly_the_clifford_subgroup() {
    // The honesty proof behind the Gottesman–Knill label, gate side: the
    // metadata-free sector (absorption) accepts a gate IFF an independent
    // dense ground truth says it is Clifford — bidirectional, over every
    // registered gate, with membership *computed*, not read off a list.
    // The recognizer cannot exceed the Clifford group by construction
    // (its acceptance test is Pauli-normalizer membership); this pins the
    // implementation to that construction.
    let reg = GateRegistry::<C64>::standard();
    let generic = [0.7365, 1.2113, -0.5871]; // far from Clifford angles
    let mut absorbed_names = Vec::new();
    let mut rejected_names = Vec::new();
    for name in reg.names() {
        let def = reg.resolve(&name).unwrap();
        let k = def.arity();
        if k > CLIFFORD_RECOGNITION_MAX {
            continue; // outside the recognizer's stated scope
        }
        let m = def.matrix(&generic[..def.param_count()]).unwrap();
        let truth = is_clifford_dense(&m, k);

        let mut state = CliffordFramedState::<C64>::new(k).unwrap();
        state.apply(&m, &(0..k).collect::<Vec<_>>()).unwrap();
        let absorbed = state.stats().absorbed_clifford == 1;
        assert_eq!(
            absorbed, truth,
            "{name}: recognizer ({absorbed}) disagrees with dense ground truth ({truth})"
        );
        if absorbed {
            absorbed_names.push(name);
        } else {
            rejected_names.push(name);
        }
    }
    // Spot-check both buckets against the textbook.
    for name in ["h", "s", "sdg", "sx", "x", "y", "z", "cx", "cz", "swap"] {
        assert!(
            absorbed_names.iter().any(|n| n == name),
            "{name} must absorb"
        );
    }
    for name in ["t", "tdg", "rz", "rx", "u", "cp", "ccx", "ccz"] {
        assert!(
            rejected_names.iter().any(|n| n == name),
            "{name} must not absorb at generic parameters"
        );
    }

    // Membership is semantic, not name-based: the same parametric gates
    // absorb exactly at Clifford angles — and the ground truth agrees.
    use std::f64::consts::{FRAC_PI_2, PI};
    for (name, params) in [
        ("rz", vec![FRAC_PI_2]),
        ("rx", vec![PI]),
        ("cp", vec![PI]),
        ("p", vec![-FRAC_PI_2]),
    ] {
        let def = reg.resolve(name).unwrap();
        let m = def.matrix(&params).unwrap();
        assert!(
            is_clifford_dense(&m, def.arity()),
            "{name}({params:?}) is Clifford by ground truth"
        );
        let mut state = CliffordFramedState::<C64>::new(def.arity()).unwrap();
        state
            .apply(&m, &(0..def.arity()).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(
            state.stats().absorbed_clifford,
            1,
            "{name}({params:?}) must absorb"
        );
    }
}

#[test]
fn clifford_evolution_cost_is_polynomial_in_width_and_depth() {
    // GK's evolution sector, cost side: a Clifford stream's only costs are
    // the tableau (linear in width) and the replay log (linear in depth).
    // The amplitude side stays at support 1 at every width the masks
    // allow — no hidden exponential anywhere in the free sector.
    let reg = GateRegistry::<C64>::standard();
    let stream = |n: usize, gates: usize, seed: u64| -> Circuit {
        let mut c: Circuit = Circuit::new(n);
        let mut rng = Prng::new(seed);
        for _ in 0..gates {
            let q = (rng.next_u64() % n as u64) as usize;
            let r = ((q + 1) + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
            match rng.next_u64() % 6 {
                0 => c.h(q),
                1 => c.s(q),
                2 => c.x(q),
                3 => c.z(q),
                4 => c.cx(q, r),
                _ => c.cz(q, r),
            };
        }
        c
    };
    for n in [16usize, 32, 48, 63] {
        let mut state = CliffordFramedState::<C64>::new(n).unwrap();
        stream(n, 300, 3)
            .bind(&reg)
            .unwrap()
            .run(&mut state)
            .unwrap();
        assert_eq!(state.stored_nonzero_count(), 1, "width {n}");
        assert_eq!(state.peak_stored_support(), 1, "width {n}");
        assert!(
            state.memory_bytes() < 256 * 1024,
            "width {n}: {} bytes is not width-flat",
            state.memory_bytes()
        );
    }
    let mut state = CliffordFramedState::<C64>::new(63).unwrap();
    stream(63, 4000, 4)
        .bind(&reg)
        .unwrap()
        .run(&mut state)
        .unwrap();
    assert_eq!(state.frame_gates(), 4000, "log is the depth-linear cost");
    assert_eq!(state.stored_nonzero_count(), 1);
    assert!(
        state.memory_bytes() < 2 * 1024 * 1024,
        "depth 4000: {} bytes is not log-linear",
        state.memory_bytes()
    );
}

#[test]
fn the_boundary_no_free_magic_and_amplitude_extraction_flushes() {
    // Two-sided boundary for the GK label. Side 1 — no free magic: T
    // cannot absorb (T†XT = (X−Y)/√2 leaves the Pauli group), so the cost
    // of a non-Clifford gate lands on amplitudes the moment it arrives.
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let t = reg.resolve("t").unwrap().matrix(&[]).unwrap();
    let mut state = CliffordFramedState::<C64>::new(1).unwrap();
    state.apply(&h, &[0]).unwrap();
    state.apply(&t, &[0]).unwrap();
    let stats = state.stats();
    assert_eq!(stats.absorbed_clifford, 1, "only H absorbed");
    assert_eq!(stats.axis_rotations, 1, "T went to the amplitude side");
    assert_eq!(
        state.stored_nonzero_count(),
        2,
        "T through an H-frame scatters the stored state (support 1 → 2)"
    );

    // Side 2 — with measurement now native (see
    // measurement_is_native_and_the_frame_survives), the remaining flush
    // surface is full amplitude-vector extraction: asking for stored
    // amplitudes of a scrambled stabilizer state materializes the frame
    // and pays the physical support. That is not a Gottesman–Knill
    // capability (GK simulates *measurement*, not 2^n-amplitude dumps),
    // but it is a cost of this representation and stays pinned here.
    let n = 14;
    let mut circuit: Circuit = Circuit::new(n);
    for q in 0..n {
        circuit.h(q);
    }
    let mut rng = Prng::new(31);
    for _ in 0..200 {
        let q = (rng.next_u64() % n as u64) as usize;
        let r = ((q + 1) + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
        match rng.next_u64() % 6 {
            0 => circuit.h(q),
            1 => circuit.s(q),
            2 => circuit.x(q),
            3 => circuit.z(q),
            4 => circuit.cx(q, r),
            _ => circuit.cz(q, r),
        };
    }
    let mut state = CliffordFramedState::<C64>::new(n).unwrap();
    circuit.bind(&reg).unwrap().run(&mut state).unwrap();
    assert_eq!(state.stored_nonzero_count(), 1, "evolution was metadata");
    assert_eq!(state.peak_stored_support(), 1);

    let _ = state.amplitude(0); // one readout query
    let stats = state.stats();
    assert_eq!(stats.flushes, 1);
    assert_eq!(stats.replayed_gates, n + 200);
    assert!(
        state.peak_stored_support() >= 1 << (n - 2),
        "readout materialized the stabilizer state: {}",
        state.peak_stored_support()
    );
}

/// Scaffold: an H-layer, then a seeded Clifford stream with `t` T gates
/// spliced in at even spacing.
fn clifford_t_circuit(n: usize, cliffords: usize, t: usize, seed: u64) -> Circuit {
    let mut circuit: Circuit = Circuit::new(n);
    for q in 0..n {
        circuit.h(q);
    }
    let mut rng = Prng::new(seed);
    let stride = cliffords / (t + 1);
    let mut placed = 0;
    for g in 0..cliffords {
        let q = (rng.next_u64() % n as u64) as usize;
        let r = ((q + 1) + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
        match rng.next_u64() % 6 {
            0 => circuit.h(q),
            1 => circuit.s(q),
            2 => circuit.x(q),
            3 => circuit.z(q),
            4 => circuit.cx(q, r),
            _ => circuit.cz(q, r),
        };
        if placed < t && (g + 1) % stride == 0 {
            circuit.t((rng.next_u64() % n as u64) as usize);
            placed += 1;
        }
    }
    assert_eq!(placed, t, "scaffold must place every T");
    circuit
}

#[test]
fn t_count_bounds_the_stored_support_not_width_or_gate_count() {
    // The claim behind the whole backend: Clifford gates are metadata, so
    // the stored support is bounded by 2^t in the T-count t — while the
    // raw sparse representation pays 2^n for the same circuit. Physics
    // verified against dense at the end, exactly, including phases.
    let n = 12;
    let reg = GateRegistry::<C64>::standard();
    for t in [2usize, 5] {
        let circuit = clifford_t_circuit(n, 100, t, 21 + t as u64);
        let bound = circuit.bind(&reg).unwrap();

        let mut framed = CliffordFramedState::<C64>::new(n).unwrap();
        bound.run(&mut framed).unwrap();
        // Read representation costs BEFORE any observation flushes.
        let stats = framed.stats();
        let peak = framed.peak_stored_support();
        assert_eq!(stats.flushes, 0, "nothing may force a flush");
        assert_eq!(stats.raw_fallbacks, 0);
        assert_eq!(stats.axis_rotations, t, "each T is one axis rotation");
        assert!(
            peak <= 1 << t,
            "t={t}: peak stored support {peak} exceeds 2^t"
        );
        assert!(peak > 1, "t={t}: T gates must actually scatter");

        // Raw sparse pays the width, not the T-count.
        let mut raw = SparseState::<C64>::new(n).unwrap();
        let mut raw_peak = raw.nonzero_count();
        for gate in bound.gates() {
            match &gate.kernel {
                GateKernel::Matrix(m) => raw.apply(m, &gate.qubits).unwrap(),
                GateKernel::Diagonal(d) => raw.apply_diagonal(d, &gate.qubits).unwrap(),
            }
            raw_peak = raw_peak.max(raw.nonzero_count());
        }
        assert!(
            raw_peak >= 1 << (n - 2),
            "raw sparse should blow up: peak {raw_peak}"
        );
        assert!(peak < raw_peak, "frames must beat raw support");

        let mut dense = DenseState::<C64>::new(n).unwrap();
        bound.run(&mut dense).unwrap();
        let dev = max_amplitude_deviation(&dense, &framed);
        assert!(dev < 1e-9, "t={t}: deviation {dev}");
    }
}

#[test]
fn recognition_routes_each_gate_to_its_mechanism() {
    let reg = GateRegistry::<C64>::standard();
    let mut state = CliffordFramedState::<C64>::new(4).unwrap();
    let m = |name: &str, params: &[f64]| reg.resolve(name).unwrap().matrix(params).unwrap();

    // Cliffords absorb — including sx and the two-qubit ones.
    for (name, qs) in [("h", vec![0]), ("s", vec![1]), ("sx", vec![2])] {
        state.apply(&m(name, &[]), &qs).unwrap();
    }
    state.apply(&m("cx", &[]), &[0, 1]).unwrap();
    state.apply(&m("swap", &[]), &[2, 3]).unwrap();
    assert_eq!(state.stats().absorbed_clifford, 5);
    assert_eq!(state.stored_nonzero_count(), 1);

    // Pauli-axis rotations conjugate through the tableau.
    state.apply(&m("rz", &[0.31]), &[0]).unwrap();
    state.apply(&m("rx", &[0.52]), &[1]).unwrap();
    state.apply(&m("t", &[]), &[2]).unwrap();
    state.apply(&m("rxx", &[0.43]), &[0, 2]).unwrap();
    assert_eq!(state.stats().axis_rotations, 4);

    // Non-axis diagonals split into Z-string rotations (Walsh).
    state.apply(&m("cp", &[0.61]), &[1, 3]).unwrap();
    state.apply(&m("ccz", &[]), &[0, 1, 2]).unwrap();
    let stats = state.stats();
    assert!(
        stats.diagonal_rotations >= 3 + 7 - 2,
        "cp and ccz decompose into Z-strings: {}",
        stats.diagonal_rotations
    );

    // Generic 1q gates split as ZYZ; nothing above has flushed.
    state.apply(&m("u", &[0.7, 1.1, -0.4]), &[3]).unwrap();
    assert_eq!(state.stats().zyz_decompositions, 1);
    assert_eq!(state.stats().flushes, 0);

    // ccx is none of the above: the frame flushes and the gate goes raw.
    state.apply(&m("ccx", &[]), &[0, 1, 2]).unwrap();
    let stats = state.stats();
    assert_eq!(stats.raw_fallbacks, 1);
    assert_eq!(stats.flushes, 1);
    assert_eq!(
        stats.replayed_gates, 5,
        "the five absorbed Cliffords replay"
    );
    assert_eq!(state.frame_gates(), 0, "frame is identity after fallback");

    // Everything still means the same physics as dense.
    let sim = sim_with_clifford();
    let mut c = Circuit::new(4);
    c.h(0)
        .s(1)
        .sx(2)
        .cx(0, 1)
        .swap(2, 3)
        .rz(0, 0.31)
        .rx(1, 0.52)
        .t(2)
        .rxx(0, 2, 0.43)
        .cp(1, 3, 0.61)
        .ccz(0, 1, 2)
        .u(3, 0.7, 1.1, -0.4)
        .ccx(0, 1, 2);
    let framed = sim.run_on("clifford-framed", &c).unwrap();
    let dense = sim.run(&c).unwrap();
    let dev = max_amplitude_deviation(dense.as_ref(), framed.as_ref());
    assert!(dev < 1e-9, "{dev}");
}

#[test]
fn flush_replays_the_log_in_absorption_order() {
    // Interleaved absorb/rotate is the trap: stored holds conjugated
    // rotations, and C·r₂′·r₁′ = r₂·g₂·r₁·g₁ only if the log replays in
    // absorption order. Exact comparison (global phase included) — the
    // p-gate's e^{iθ/2} and T's e^{iπ/8} must survive.
    let sim = sim_with_clifford();
    let mut c = Circuit::new(3);
    c.h(0)
        .t(0)
        .h(0)
        .t(0)
        .cx(0, 1)
        .p(1, 0.9)
        .h(2)
        .s(2)
        .rzz(1, 2, 0.7)
        .h(1)
        .t(1)
        .h(0);
    let framed = sim.run_on("clifford-framed", &c).unwrap();
    let dense = sim.run(&c).unwrap();
    let dev = max_amplitude_deviation(dense.as_ref(), framed.as_ref());
    assert!(dev < 1e-9, "interleaved replay deviates: {dev}");

    // flush() is idempotent and reset() clears the frame wholesale.
    let reg = GateRegistry::<C64>::standard();
    let mut state = CliffordFramedState::<C64>::new(2).unwrap();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    state.apply(&h, &[0]).unwrap();
    state.flush().unwrap();
    let flushes = state.stats().flushes;
    state.flush().unwrap();
    assert_eq!(state.stats().flushes, flushes, "empty flush is free");
    state.apply(&h, &[1]).unwrap();
    state.reset();
    assert_eq!(state.frame_gates(), 0);
    assert_eq!(state.stats(), CliffordFrameStats::default());
    assert!(state.amplitude(0).approx_eq(c64(1.0, 0.0), TOL));
}

#[test]
fn measurement_is_native_and_the_frame_survives() {
    // Measuring Z_q projects the *stored* state with the conjugated
    // string: no flush, the 3-gate GHZ frame outlives the measurement
    // (grown by the repair Clifford, never replayed), and the collapsed
    // physics is still exact.
    let sim = sim_with_clifford();
    let mut c = Circuit::new(3);
    c.h(0).cx(0, 1).cx(1, 2);
    let mut state = sim.run_on("clifford-framed", &c).unwrap();
    let outcome = state.measure(1, &mut Prng::new(5)).unwrap();
    let framed = state
        .as_any()
        .downcast_ref::<CliffordFramedState<C64>>()
        .unwrap();
    assert!(
        framed.frame_gates() >= 3,
        "frame survives measurement (plus its repair gates)"
    );
    let stats = framed.stats();
    assert_eq!(stats.flushes, 0);
    assert_eq!(stats.native_measurements, 1);
    assert_eq!(stats.frame_repairs, 1, "a random outcome triggers repair");
    assert_close(state.total_weight(), 1.0, TOL);
    // GHZ collapse: all three qubits agree (probability() flushes — after
    // the fact, as an observation should).
    let expect = if outcome { (1u64 << 3) - 1 } else { 0 };
    assert_close(state.probability(expect), 1.0, TOL);

    // Outcomes and collapsed states are seed-identical with dense, which
    // measures by the flat default path.
    let n = 8;
    let mut c = Circuit::new(n);
    let mut rng = Prng::new(9);
    for _ in 0..80 {
        let q = (rng.next_u64() % n as u64) as usize;
        let r = ((q + 1) + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
        match rng.next_u64() % 6 {
            0 => c.h(q),
            1 => c.s(q),
            2 => c.x(q),
            3 => c.z(q),
            4 => c.cx(q, r),
            _ => c.cz(q, r),
        };
    }
    let reg = GateRegistry::<C64>::standard();
    let bound = c.bind(&reg).unwrap();
    let mut framed = CliffordFramedState::<C64>::new(n).unwrap();
    bound.run(&mut framed).unwrap();
    let mut dense = DenseState::<C64>::new(n).unwrap();
    bound.run(&mut dense).unwrap();
    let (mut rng_f, mut rng_d) = (Prng::new(17), Prng::new(17));
    for q in 0..n {
        let of = framed.measure(q, &mut rng_f).unwrap();
        let od = dense.measure(q, &mut rng_d).unwrap();
        assert_eq!(of, od, "outcome diverged at qubit {q}");
    }
    let dev = max_amplitude_deviation(&dense, &framed);
    assert!(dev < 1e-9, "collapsed states diverge: {dev}");

    // The adaptive-sequence envelope at width past dense — measured on
    // both sides of the frame-repair switch. Without repair, projection's
    // ≤2× growth COMPOUNDS: the register collapses physically while the
    // stored side becomes C†|b⟩, so m measurements are bounded by 2^m
    // (the pre-repair finding, kept measurable for the record). With
    // repair (the default — the true tableau measurement update, C ← C·V
    // re-aligning the stored basis after each projection), the same
    // sequence stays FLAT: peak stored support 2, final support 1, at any
    // sequence length. That is the Gottesman–Knill measurement sector
    // reproduced through the frame mechanism — and it is asserted, not
    // hoped for.
    let n = 40;
    let mut c = Circuit::new(n);
    let mut rng = Prng::new(23);
    for _ in 0..300 {
        let q = (rng.next_u64() % n as u64) as usize;
        let r = ((q + 1) + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
        match rng.next_u64() % 6 {
            0 => c.h(q),
            1 => c.s(q),
            2 => c.x(q),
            3 => c.z(q),
            4 => c.cx(q, r),
            _ => c.cz(q, r),
        };
    }
    let bound = c.bind(&reg).unwrap();

    // Pre-repair envelope, pinned for the record (m kept small: the cost
    // is genuinely exponential in m).
    let m_off = 6;
    let mut drifting = CliffordFramedState::<C64>::new(n).unwrap();
    drifting.set_measure_repair(false);
    bound.run(&mut drifting).unwrap();
    let mut rng_off = Prng::new(41);
    let mut outcomes_off = Vec::new();
    for q in 0..m_off {
        outcomes_off.push(drifting.measure(q, &mut rng_off).unwrap());
    }
    assert_eq!(drifting.stats().flushes, 0);
    assert_eq!(drifting.stats().frame_repairs, 0);
    assert!(
        drifting.peak_stored_support() <= 1 << m_off,
        "unrepaired support bounded by 2^measurements: {}",
        drifting.peak_stored_support()
    );
    assert!(
        drifting.peak_stored_support() > 1,
        "the unrepaired drift is real — the stored state left the frame"
    );

    // Repaired run: same circuit, same seed — identical outcomes (repair
    // happens after the draw and never touches the physics), flat cost,
    // and a long sequence (every qubit) stays flat too.
    let m_on = n;
    let mut repaired = CliffordFramedState::<C64>::new(n).unwrap();
    bound.run(&mut repaired).unwrap();
    let mut rng_on = Prng::new(41);
    let mut outcomes_on = Vec::new();
    for q in 0..m_on {
        outcomes_on.push(repaired.measure(q, &mut rng_on).unwrap());
    }
    assert_eq!(&outcomes_on[..m_off], &outcomes_off[..], "same physics");
    let stats = repaired.stats();
    assert_eq!(stats.native_measurements, m_on);
    assert_eq!(stats.flushes, 0, "prep + evolve + measure + repair, no flush");
    assert!(stats.frame_repairs > 0, "random outcomes trigger repairs");
    assert!(
        repaired.peak_stored_support() <= 2,
        "repaired adaptive sequences stay flat: peak {}",
        repaired.peak_stored_support()
    );
    assert_eq!(
        repaired.stored_nonzero_count(),
        1,
        "after measuring all 40 qubits the stored state is one basis state"
    );
    // (No observation flush here on purpose: materializing the physical
    // amplitudes of the drifted run would cost ~2^n; the weight-1
    // invariant is already asserted on the cheap widths above.)
}

#[test]
fn algebra_gating_rejects_non_complex_scalars() {
    // Recognition and rotation coefficients live in ℂ: the wrapper wants a
    // commutative division algebra containing i. CComplex (Cayley–Dickson
    // ℂ) passes; ℝ, quaternions and split-complex are rejected up front.
    assert!(matches!(
        CliffordFramedState::<f64>::new(2),
        Err(Error::UnsupportedForAlgebra { .. })
    ));
    assert!(matches!(
        CliffordFramedState::<Quaternion>::new(2),
        Err(Error::UnsupportedForAlgebra { .. })
    ));
    assert!(matches!(
        CliffordFramedState::<SplitComplex>::new(2),
        Err(Error::UnsupportedForAlgebra { .. })
    ));

    let reg = GateRegistry::<CComplex>::standard();
    let mut state = CliffordFramedState::<CComplex>::new(2).unwrap();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let cx = reg.resolve("cx").unwrap().matrix(&[]).unwrap();
    state.apply(&h, &[0]).unwrap();
    state.apply(&cx, &[0, 1]).unwrap();
    assert_eq!(state.stored_nonzero_count(), 1, "bell prep absorbed");
    let a = state.amplitude(0);
    assert!(a.approx_eq(CComplex::from_re(FRAC_1_SQRT_2), TOL));
    let b = state.amplitude(3);
    assert!(b.approx_eq(CComplex::from_re(FRAC_1_SQRT_2), TOL));
}

#[test]
fn edge_paths_stay_correct() {
    let reg = GateRegistry::<C64>::standard();

    // Rotation masks are bounds-checked like everything else.
    let mut s = SparseState::<C64>::new(3).unwrap();
    assert!(matches!(
        s.apply_pauli_rotation(0.4, 1 << 5, 0, false),
        Err(Error::QubitOutOfRange { .. })
    ));

    // A Clifford arriving as a *diagonal kernel* absorbs, and the flush
    // replays the diagonal form of the log entry.
    let mut state = CliffordFramedState::<C64>::new(2).unwrap();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    state.apply(&h, &[0]).unwrap();
    state
        .apply_diagonal(
            &[c64(1.0, 0.0), c64(1.0, 0.0), c64(1.0, 0.0), c64(-1.0, 0.0)],
            &[0, 1],
        )
        .unwrap();
    assert_eq!(
        state.stats().absorbed_clifford,
        2,
        "cz-as-diagonal absorbed"
    );
    assert_eq!(state.name(), "clifford-framed");
    // H then CZ on |00⟩ = (|00⟩+|10⟩)/√2; observation replays both.
    assert!(state.amplitude(1).approx_eq(c64(FRAC_1_SQRT_2, 0.0), TOL));
    assert_eq!(state.nonzero_count(), 2);
    assert!(state.peak_inner_memory() > 0);

    // The tableau introspection API: after H(0), C†·X₀·C = Z₀.
    let mut state = CliffordFramedState::<C64>::new(2).unwrap();
    state.apply(&h, &[0]).unwrap();
    let x0 = PauliString {
        x: 1,
        z: 0,
        negative: false,
    };
    let img = state.conjugated(x0);
    assert_eq!(
        img,
        PauliString {
            x: 0,
            z: 1,
            negative: false
        }
    );

    // A non-unitary diagonal cannot be Walsh-split or absorbed: it must
    // fall back raw and still act exactly (sparse applies any diagonal).
    let mut state = CliffordFramedState::<C64>::new(1).unwrap();
    state.apply(&h, &[0]).unwrap();
    state
        .apply_diagonal(&[c64(1.0, 0.0), c64(0.5, 0.0)], &[0])
        .unwrap();
    assert_eq!(state.stats().raw_fallbacks, 1);
    assert!(state
        .amplitude(1)
        .approx_eq(c64(0.5 * FRAC_1_SQRT_2, 0.0), TOL));

    // A 2q unitary that is neither Clifford, axis, nor diagonal (an
    // fSim-style mixture) forces the same fallback for matrices.
    let (th, ph) = (0.37f64, 0.83f64);
    let (c_, s_) = (th.cos(), th.sin());
    let fsim = GateMatrix::from_vec(
        4,
        vec![
            c64(1.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(c_, 0.0),
            c64(0.0, -s_),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, -s_),
            c64(c_, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            cis(-ph),
        ],
    )
    .unwrap();
    let mut framed = CliffordFramedState::<C64>::new(2).unwrap();
    let mut dense = DenseState::<C64>::new(2).unwrap();
    for state in [&mut framed as &mut dyn Backend<C64>, &mut dense] {
        state.apply(&h, &[0]).unwrap();
        state.apply(&h, &[1]).unwrap();
        state.apply(&fsim, &[0, 1]).unwrap();
    }
    assert_eq!(framed.stats().raw_fallbacks, 1);
    assert!(max_amplitude_deviation(&dense, &framed) < 1e-12);

    // project and load keep the Backend contract through the wrapper.
    let mut state = CliffordFramedState::<C64>::new(2).unwrap();
    state.apply(&h, &[0]).unwrap();
    state.project(0, true, std::f64::consts::SQRT_2);
    assert!(state.amplitude(1).approx_eq(c64(1.0, 0.0), TOL));
    state
        .load(&[(0, c64(0.6, 0.0)), (3, c64(0.8, 0.0))])
        .unwrap();
    assert_close(state.probability(3), 0.64, TOL);
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn caps_are_wired() {
    // The recognizers only see gates up to their caps; a 6-qubit diagonal
    // exceeds CLIFFORD_DIAGONAL_MAX and must fall back (still correct).
    assert!(CLIFFORD_RECOGNITION_MAX <= CLIFFORD_DIAGONAL_MAX);
    let sim = sim_with_clifford();
    let n = CLIFFORD_DIAGONAL_MAX + 1;
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    c.diagonal(
        "flip",
        library::phase_flip(n, 3),
        (0..n).collect::<Vec<_>>(),
    );
    for q in 0..n {
        c.h(q);
    }
    let framed = sim.run_on("clifford-framed", &c).unwrap();
    let stats = framed
        .as_any()
        .downcast_ref::<CliffordFramedState<C64>>()
        .unwrap()
        .stats();
    assert_eq!(stats.raw_fallbacks, 1, "wide diagonal went raw");
    let dense = sim.run(&c).unwrap();
    let dev = max_amplitude_deviation(dense.as_ref(), framed.as_ref());
    assert!(dev < 1e-9, "{dev}");
}
