//! Property-based tests (proptest): invariants that must hold for *random*
//! circuits, parameters and states — the broadest net in the suite.

mod common;

use proptest::prelude::*;
use quantsim::prelude::*;

/// Gate pool for random circuits: name, arity, param count.
const POOL: &[(&str, usize, usize)] = &[
    ("h", 1, 0),
    ("x", 1, 0),
    ("y", 1, 0),
    ("z", 1, 0),
    ("s", 1, 0),
    ("t", 1, 0),
    ("sx", 1, 0),
    ("rx", 1, 1),
    ("ry", 1, 1),
    ("rz", 1, 1),
    ("p", 1, 1),
    ("u", 1, 3),
    ("cx", 2, 0),
    ("cz", 2, 0),
    ("ch", 2, 0),
    ("swap", 2, 0),
    ("iswap", 2, 0),
    ("cp", 2, 1),
    ("rxx", 2, 1),
    ("ryy", 2, 1),
    ("rzz", 2, 1),
    ("ccx", 3, 0),
    ("ccz", 3, 0),
    ("cswap", 3, 0),
];

#[derive(Clone, Debug)]
struct Spec {
    name: &'static str,
    params: Vec<f64>,
    qubits: Vec<usize>,
}

fn arb_spec(n: usize) -> impl Strategy<Value = Spec> {
    let max_arity = n.min(3);
    let pool: Vec<&'static (&'static str, usize, usize)> =
        POOL.iter().filter(|e| e.1 <= max_arity).collect();
    let len = pool.len();
    (0..len).prop_flat_map(move |i| {
        let &(name, arity, np) = pool[i];
        let params = proptest::collection::vec(-6.3f64..6.3, np);
        let qubits =
            proptest::sample::subsequence((0..n).collect::<Vec<usize>>(), arity).prop_shuffle();
        (params, qubits).prop_map(move |(params, qubits)| Spec {
            name,
            params,
            qubits,
        })
    })
}

fn arb_circuit() -> impl Strategy<Value = (usize, Vec<Spec>)> {
    (1usize..=5).prop_flat_map(|n| (Just(n), proptest::collection::vec(arb_spec(n), 0..25)))
}

fn build<S: Scalar>(n: usize, specs: &[Spec]) -> Circuit<S> {
    let mut c = Circuit::new(n);
    for s in specs {
        c.gate(s.name, s.params.clone(), s.qubits.clone());
    }
    c
}

proptest! {
    /// Unitary evolution preserves total Born weight over ℂ.
    #[test]
    fn norm_preserved_dense((n, specs) in arb_circuit()) {
        let state = Simulator::<C64>::new().run(&build(n, &specs)).unwrap();
        prop_assert!((state.total_weight() - 1.0).abs() < 1e-9);
    }

    /// Dense, sparse and adaptive representations agree amplitude-for-
    /// amplitude on arbitrary circuits — three implementations, one answer.
    #[test]
    fn representations_agree((n, specs) in arb_circuit()) {
        let sim = Simulator::<C64>::new();
        let c = build(n, &specs);
        let dense = sim.run(&c).unwrap();
        let sparse = sim.run_on("sparse", &c).unwrap();
        let adaptive = sim.run_on("adaptive", &c).unwrap();
        for i in 0..(1u64 << n) {
            let d = dense.amplitude(i);
            prop_assert!(d.approx_eq(sparse.amplitude(i), 1e-9), "sparse idx {}", i);
            prop_assert!(d.approx_eq(adaptive.amplitude(i), 1e-9), "adaptive idx {}", i);
        }
    }

    /// Running a circuit then its inverse restores |0…0⟩ exactly (global
    /// phase included — daggering is an exact inverse over ℂ).
    #[test]
    fn inverse_uncomputes((n, specs) in arb_circuit()) {
        let reg = GateRegistry::<C64>::standard();
        let bound = build(n, &specs).bind(&reg).unwrap();
        let mut state = DenseState::<C64>::new(n).unwrap();
        bound.run(&mut state).unwrap();
        bound.inverse().run(&mut state).unwrap();
        prop_assert!(state.amplitude(0).approx_eq(c64(1.0, 0.0), 1e-7));
    }

    /// Every registry gate is unitary at arbitrary parameters (not just the
    /// registration probe).
    #[test]
    fn all_gates_unitary_at_random_params(raw in proptest::collection::vec(-10.0f64..10.0, 8)) {
        let reg = GateRegistry::<C64>::standard();
        for name in reg.names() {
            let def = reg.resolve(&name).unwrap();
            let params: Vec<f64> = (0..def.param_count()).map(|i| raw[i % raw.len()]).collect();
            let m = def.matrix(&params).unwrap();
            prop_assert!(m.unitarity_deviation() < 1e-9, "{} at {:?}", name, params);
        }
    }

    /// Gates on disjoint qubits commute.
    #[test]
    fn disjoint_gates_commute(
        (pa, pb) in (proptest::collection::vec(-6.3f64..6.3, 3),
                     proptest::collection::vec(-6.3f64..6.3, 3)),
        qa in 0usize..4,
        qshift in 1usize..4,
    ) {
        let qb = (qa + qshift) % 4;
        let mut ab: Circuit = common::scrambler(4);
        ab.u(qa, pa[0], pa[1], pa[2]).u(qb, pb[0], pb[1], pb[2]);
        let mut ba: Circuit = common::scrambler(4);
        ba.u(qb, pb[0], pb[1], pb[2]).u(qa, pa[0], pa[1], pa[2]);
        let (sa, sb) = (common::run_dense(&ab), common::run_dense(&ba));
        for i in 0..16u64 {
            prop_assert!(sa.amplitude(i).approx_eq(sb.amplitude(i), 1e-9));
        }
    }

    /// Rotation additivity: R(a)·R(b) = R(a+b) for rx, ry, rz on a generic
    /// state (exact in the half-angle convention).
    #[test]
    fn rotation_additivity(a in -6.3f64..6.3, b in -6.3f64..6.3, which in 0usize..3) {
        let apply = |c: &mut Circuit, angle: f64| {
            match which {
                0 => c.rx(0, angle),
                1 => c.ry(0, angle),
                _ => c.rz(0, angle),
            };
        };
        let mut two: Circuit = common::scrambler(2);
        apply(&mut two, a);
        apply(&mut two, b);
        let mut one: Circuit = common::scrambler(2);
        apply(&mut one, a + b);
        let (sa, sb) = (common::run_dense(&two), common::run_dense(&one));
        for i in 0..4u64 {
            prop_assert!(sa.amplitude(i).approx_eq(sb.amplitude(i), 1e-9));
        }
    }

    /// Measurement collapses: the measured qubit's opposite branch has zero
    /// probability and the state stays normalized.
    #[test]
    fn measurement_collapse((n, specs) in arb_circuit(), q_raw in 0usize..5, seed in 0u64..1000) {
        let q = q_raw % n;
        let mut state = Simulator::<C64>::new().run(&build(n, &specs)).unwrap();
        let outcome = state.measure(q, &mut Prng::new(seed)).unwrap();
        let mut opposite = 0.0;
        state.for_each_nonzero(&mut |i, a| {
            if ((i >> q) & 1 == 1) != outcome {
                opposite += a.born_weight();
            }
        });
        prop_assert!(opposite < 1e-18);
        prop_assert!((state.total_weight() - 1.0).abs() < 1e-9);
    }

    /// Sampling is representation-independent for a given seed.
    #[test]
    fn sampling_deterministic_across_backends((n, specs) in arb_circuit(), seed in 0u64..1000) {
        let sim = Simulator::<C64>::new();
        let c = build(n, &specs);
        let a = sim.run(&c).unwrap().sample(64, &mut Prng::new(seed)).unwrap();
        let b = sim.run_on("sparse", &c).unwrap().sample(64, &mut Prng::new(seed)).unwrap();
        prop_assert_eq!(a, b);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// The Cayley–Dickson ℂ (CD<f64>) and num-complex ℂ give identical
    /// simulations — the doubling construction is wired in correctly.
    #[test]
    fn cd_complex_simulates_identically((n, specs) in arb_circuit()) {
        let zc = Simulator::<C64>::new().run(&build(n, &specs)).unwrap();
        let cd = Simulator::<CComplex>::new().run(&build(n, &specs)).unwrap();
        for i in 0..(1u64 << n) {
            let z = zc.amplitude(i);
            let c = cd.amplitude(i).coeffs();
            prop_assert!((z.re - c[0]).abs() < 1e-9 && (z.im - c[1]).abs() < 1e-9);
        }
    }

    /// Simulating a complex circuit over ℍ stays inside the embedded ℂ and
    /// matches the ℂ result (algebra embeddings are homomorphisms).
    #[test]
    fn quaternion_embedding_faithful((n, specs) in arb_circuit()) {
        let zc = Simulator::<C64>::new().run(&build(n, &specs)).unwrap();
        let hq = Simulator::<Quaternion>::new().run(&build(n, &specs)).unwrap();
        for i in 0..(1u64 << n) {
            let z = zc.amplitude(i);
            let q = hq.amplitude(i).coeffs();
            prop_assert!((z.re - q[0]).abs() < 1e-9 && (z.im - q[1]).abs() < 1e-9);
            prop_assert!(q[2].abs() < 1e-9 && q[3].abs() < 1e-9);
        }
        prop_assert!((hq.total_weight() - 1.0).abs() < 1e-9);
    }

    /// Same for 𝕆 — including that norm is still conserved (division
    /// algebra), despite non-associativity.
    #[test]
    fn octonion_embedding_faithful((n, specs) in arb_circuit()) {
        let zc = Simulator::<C64>::new().run(&build(n, &specs)).unwrap();
        let oc = Simulator::<Octonion>::new().run(&build(n, &specs)).unwrap();
        for i in 0..(1u64 << n) {
            let z = zc.amplitude(i);
            let o = oc.amplitude(i).coeffs();
            prop_assert!((z.re - o[0]).abs() < 1e-9 && (z.im - o[1]).abs() < 1e-9);
            prop_assert!(o[2..].iter().all(|x| x.abs() < 1e-9));
        }
        prop_assert!((oc.total_weight() - 1.0).abs() < 1e-9);
    }

    /// Scalar algebra laws under random elements: conjugation is an
    /// anti-homomorphism and scale commutes with everything, for every
    /// shipped algebra.
    #[test]
    fn scalar_algebra_laws(coeffs in proptest::collection::vec(-2.0f64..2.0, 32), k in -3.0f64..3.0) {
        fn laws<S: Scalar>(c: &[f64], k: f64) {
            let x = S::from_coeffs(&c[..S::DIM]);
            let y = S::from_coeffs(&c[S::DIM..2 * S::DIM]);
            // conj(x·y) = conj(y)·conj(x)
            assert!((x * y).conj().approx_eq(y.conj() * x.conj(), 1e-9));
            // conj involutive; re invariant
            assert!(x.conj().conj().approx_eq(x, 1e-12));
            assert!((x.conj().re() - x.re()).abs() < 1e-12);
            // scale is central: (k·x)·y = k·(x·y) = x·(k·y)
            assert!((x.scale(k) * y).approx_eq((x * y).scale(k), 1e-9));
            assert!((x * y.scale(k)).approx_eq((x * y).scale(k), 1e-9));
            // abs_sqr additive over coordinates and conj-invariant
            assert!((x.conj().abs_sqr() - x.abs_sqr()).abs() < 1e-9);
            // one is the identity
            assert!((x * S::one()).approx_eq(x, 1e-12) && (S::one() * x).approx_eq(x, 1e-12));
        }
        laws::<f64>(&coeffs, k);
        laws::<C64>(&coeffs, k);
        laws::<CComplex>(&coeffs, k);
        laws::<Quaternion>(&coeffs, k);
        laws::<Octonion>(&coeffs, k);
        laws::<Sedenion>(&coeffs, k);
        laws::<SplitComplex>(&coeffs, k);
    }

    /// Quaternion simulation with a genuinely quaternionic gate still
    /// conserves Born weight (ℍ is a division algebra), even though the
    /// state leaves the complex subalgebra.
    #[test]
    fn quaternionic_gates_conserve_weight((n, specs) in arb_circuit(), slot in 0usize..4) {
        let mut sim = Simulator::<Quaternion>::new();
        let j = Quaternion::basis(2);
        let mut m = GateMatrix::<Quaternion>::identity(2).unwrap();
        m.set(1, 1, j);
        sim.registry_mut().register_fixed("jphase", "diag(1, j)", m).unwrap();
        let mut c = build(n, &specs);
        c.gate("jphase", Vec::new(), vec![slot % n]);
        let state = sim.run(&c).unwrap();
        prop_assert!((state.total_weight() - 1.0).abs() < 1e-9);
    }
}
