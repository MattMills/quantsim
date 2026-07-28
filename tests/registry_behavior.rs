//! Behavior of the gate and backend registries: registration, validation,
//! aliasing, error paths, and the research-extension workflow end to end.

mod common;

use common::*;
use quantsim::prelude::*;
use std::sync::Arc;

fn xy_matrix(theta: f64) -> GateMatrix<C64> {
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
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
            o,
        ],
    )
    .unwrap()
}

#[test]
fn research_gate_registration_and_use() {
    let mut sim: Simulator = Simulator::new();
    sim.registry_mut()
        .register_parametric("xy", "XY interaction", 2, 1, |p| Ok(xy_matrix(p[0])))
        .unwrap();

    let theta = 1.618;
    let mut with_gate = Circuit::new(2);
    with_gate.h(0).gate("xy", vec![theta], vec![0, 1]);
    let mut with_raw = Circuit::new(2);
    with_raw.h(0).raw("xy-raw", xy_matrix(theta), vec![0, 1]);

    let (sa, sb) = (sim.run(&with_gate).unwrap(), sim.run(&with_raw).unwrap());
    for i in 0..4u64 {
        assert!(sa.amplitude(i).approx_eq(sb.amplitude(i), TOL));
    }
    // XY(π) swaps |01⟩ and |10⟩ with a −i phase.
    let mut c = Circuit::new(2);
    c.x(0).gate("xy", vec![std::f64::consts::PI], vec![0, 1]);
    let s = sim.run(&c).unwrap();
    assert_amp(s.as_ref(), 0b10, 0.0, -1.0);
}

#[test]
fn duplicate_and_unknown_gates_error() {
    let mut reg: GateRegistry = GateRegistry::standard();
    let err = reg
        .register_fixed("h", "clash", GateMatrix::identity(2).unwrap())
        .unwrap_err();
    assert!(matches!(err, Error::DuplicateGate(name) if name == "h"));

    let err = reg.alias("cx", "swap").unwrap_err();
    assert!(matches!(err, Error::DuplicateGate(_)));
    let err = reg.alias("brand_new", "no_such_gate").unwrap_err();
    assert!(matches!(err, Error::UnknownGate(_)));

    let mut c = Circuit::new(1);
    c.gate("nonexistent", Vec::new(), vec![0]);
    let err = c.bind(&reg).unwrap_err();
    assert!(matches!(err, Error::UnknownGate(name) if name == "nonexistent"));
}

#[test]
fn non_unitary_matrices_rejected() {
    let mut reg: GateRegistry = GateRegistry::new();
    let mut bad = GateMatrix::<C64>::identity(2).unwrap();
    bad.set(0, 0, c64(2.0, 0.0));
    let err = reg
        .register_fixed("bad", "not unitary", bad.clone())
        .unwrap_err();
    assert!(matches!(err, Error::NotUnitary { deviation, .. } if deviation > 1.0));

    // Raw matrices are checked at bind time too.
    let mut c = Circuit::new(1);
    c.raw("bad", bad, vec![0]);
    let err = c.bind(&reg).unwrap_err();
    assert!(matches!(err, Error::NotUnitary { .. }));
}

#[test]
fn dimension_and_arity_validation() {
    let mut reg: GateRegistry = GateRegistry::new();
    // Claims arity 2 but produces a 2×2 matrix.
    let err = reg
        .register_parametric("wrong_dim", "", 2, 0, |_| GateMatrix::identity(2))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::BadDimension {
            expected: 4,
            got: 2
        }
    ));

    let reg: GateRegistry = GateRegistry::standard();
    let mut c = Circuit::new(3);
    c.gate("cx", Vec::new(), vec![0, 1, 2]);
    assert!(matches!(
        c.bind(&reg).unwrap_err(),
        Error::ArityMismatch {
            expected: 2,
            got: 3,
            ..
        }
    ));

    let mut c = Circuit::new(1);
    c.gate("rx", Vec::new(), vec![0]);
    assert!(matches!(
        c.bind(&reg).unwrap_err(),
        Error::ParamCountMismatch {
            expected: 1,
            got: 0,
            ..
        }
    ));

    let mut c = Circuit::new(2);
    c.gate("cx", Vec::new(), vec![0, 5]);
    assert!(matches!(
        c.bind(&reg).unwrap_err(),
        Error::QubitOutOfRange { qubit: 5, .. }
    ));

    let mut c = Circuit::new(2);
    c.gate("cx", Vec::new(), vec![1, 1]);
    assert!(matches!(
        c.bind(&reg).unwrap_err(),
        Error::DuplicateQubits { .. }
    ));
}

#[test]
fn gate_def_trait_objects_work_directly() {
    struct MyGate;
    impl GateDef<C64> for MyGate {
        fn name(&self) -> &str {
            "my_z"
        }
        fn arity(&self) -> usize {
            1
        }
        fn param_count(&self) -> usize {
            0
        }
        fn matrix(&self, _: &[f64]) -> Result<GateMatrix<C64>> {
            GateMatrix::try_from_c64s(
                2,
                &[c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(-1.0, 0.0)],
            )
            .ok_or(Error::UnsupportedForAlgebra {
                gate: "my_z".into(),
                algebra: "?".into(),
            })
        }
    }
    let mut reg: GateRegistry = GateRegistry::new();
    reg.register(Arc::new(MyGate)).unwrap();
    assert_eq!(reg.names(), vec!["my_z"]);
    assert!(reg.resolve("my_z").is_ok());
}

/// A research backend: wraps the dense state and counts gate applications.
struct CountingBackend {
    inner: DenseState<C64>,
    applied: usize,
}

impl Backend<C64> for CountingBackend {
    fn name(&self) -> &str {
        "counting"
    }
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }
    fn apply(&mut self, m: &GateMatrix<C64>, qubits: &[usize]) -> Result<()> {
        self.applied += 1;
        self.inner.apply(m, qubits)
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
        self.applied = 0;
        self.inner.reset()
    }
    fn load(&mut self, entries: &[(u64, C64)]) -> Result<()> {
        self.inner.load(entries)
    }
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes() + std::mem::size_of::<usize>()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn research_backend_via_registry() {
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("counting", |n| {
            Ok(Box::new(CountingBackend {
                inner: DenseState::new(n)?,
                applied: 0,
            }))
        })
        .unwrap();
    assert_eq!(
        sim.backends().names(),
        vec!["adaptive", "counting", "dense", "factored", "mera", "mps", "sparse"]
    );

    let c = library::ghz(4);
    let state = sim.run_on("counting", &c).unwrap();
    let counting = state.as_any().downcast_ref::<CountingBackend>().unwrap();
    assert_eq!(counting.applied, c.len());
    // And it simulates correctly while counting.
    assert!(state
        .amplitude(0)
        .approx_eq(c64(std::f64::consts::FRAC_1_SQRT_2, 0.0), TOL));

    let err = sim
        .backends_mut()
        .register("dense", |n| Ok(Box::new(DenseState::new(n)?)));
    assert!(matches!(err.unwrap_err(), Error::DuplicateBackend(_)));
    match sim.run_on("no_such_backend", &c) {
        Err(Error::UnknownBackend(name)) => assert_eq!(name, "no_such_backend"),
        other => panic!("expected UnknownBackend, got {:?}", other.err()),
    }
}

#[test]
fn default_apply_diagonal_materializes_for_custom_backends() {
    // CountingBackend does not override apply_diagonal, so a diagonal op
    // exercises the trait's default (materialize the matrix, then apply) —
    // custom research backends must keep working without knowing about
    // structured kernels.
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("counting", |n| {
            Ok(Box::new(CountingBackend {
                inner: DenseState::new(n)?,
                applied: 0,
            }))
        })
        .unwrap();
    let mut c = Circuit::new(3);
    c.h(0)
        .h(1)
        .h(2)
        .diagonal("flip5", library::phase_flip(3, 5), vec![0, 1, 2])
        .h(2);
    let via_counting = sim.run_on("counting", &c).unwrap();
    let via_dense = sim.run(&c).unwrap();
    for i in 0..8u64 {
        assert!(
            via_counting
                .amplitude(i)
                .approx_eq(via_dense.amplitude(i), TOL),
            "idx {i}"
        );
    }
    // The fallback still routed through apply(): all 5 ops counted.
    let counting = via_counting
        .as_any()
        .downcast_ref::<CountingBackend>()
        .unwrap();
    assert_eq!(counting.applied, 5);
    // And the default validates dimensions before materializing.
    let mut bad = DenseState::<C64>::new(2).unwrap();
    struct Plain(DenseState<C64>);
    impl Backend<C64> for Plain {
        fn name(&self) -> &str {
            "plain"
        }
        fn num_qubits(&self) -> usize {
            self.0.num_qubits()
        }
        fn apply(&mut self, m: &GateMatrix<C64>, q: &[usize]) -> Result<()> {
            self.0.apply(m, q)
        }
        fn amplitude(&self, i: u64) -> C64 {
            self.0.amplitude(i)
        }
        fn for_each_nonzero(&self, f: &mut dyn FnMut(u64, C64)) {
            self.0.for_each_nonzero(f)
        }
        fn project(&mut self, q: usize, o: bool, r: f64) {
            self.0.project(q, o, r)
        }
        fn reset(&mut self) {
            self.0.reset()
        }
        fn load(&mut self, e: &[(u64, C64)]) -> Result<()> {
            self.0.load(e)
        }
        fn memory_bytes(&self) -> usize {
            self.0.memory_bytes()
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
    let mut plain = Plain(std::mem::replace(&mut bad, DenseState::new(1).unwrap()));
    let err = plain
        .apply_diagonal(&[c64(1.0, 0.0); 3], &[0, 1])
        .unwrap_err();
    assert!(matches!(
        err,
        Error::BadDimension {
            expected: 4,
            got: 3
        }
    ));
}

#[test]
fn gate_def_param_validation_direct() {
    // The defensive checks inside GateDef implementations, hit directly
    // (bind normally validates first, so these arms need explicit coverage).
    let fixed = FixedGate::new(
        "fg",
        "a fixed gate",
        GateMatrix::<C64>::identity(2).unwrap(),
    );
    assert_eq!(fixed.name(), "fg");
    assert_eq!(fixed.description(), "a fixed gate");
    assert_eq!(fixed.arity(), 1);
    assert_eq!(fixed.param_count(), 0);
    assert!(fixed.matrix(&[]).is_ok());
    assert!(matches!(
        fixed.matrix(&[1.0]).unwrap_err(),
        Error::ParamCountMismatch {
            expected: 0,
            got: 1,
            ..
        }
    ));

    let param = ParamGate::new("pg", "a param gate", 1, 2, |p: &[f64]| {
        let _ = p;
        GateMatrix::<C64>::identity(2)
    });
    assert_eq!(param.name(), "pg");
    assert_eq!(param.description(), "a param gate");
    assert_eq!(param.arity(), 1);
    assert_eq!(param.param_count(), 2);
    assert!(param.matrix(&[0.1, 0.2]).is_ok());
    assert!(matches!(
        param.matrix(&[0.1]).unwrap_err(),
        Error::ParamCountMismatch {
            expected: 2,
            got: 1,
            ..
        }
    ));
}

#[test]
fn simulator_apply_single_gates() {
    let sim: Simulator = Simulator::new();
    let mut state = DenseState::<C64>::new(2).unwrap();
    sim.apply(&mut state, "h", &[], &[0]).unwrap();
    sim.apply(&mut state, "cx", &[], &[0, 1]).unwrap();
    assert!((state.probability(0b11) - 0.5).abs() < TOL);
    assert!(matches!(
        sim.apply(&mut state, "rx", &[], &[0]).unwrap_err(),
        Error::ParamCountMismatch { .. }
    ));
    assert!(matches!(
        sim.apply(&mut state, "cx", &[], &[0]).unwrap_err(),
        Error::ArityMismatch { .. }
    ));
}

#[test]
fn quaternionic_research_gate() {
    // Over ℍ a gate can use the j unit: diag(1, j) is quaternion-unitary
    // (conj(j)·j = 1). This is exactly the kind of gate the registry's
    // per-algebra design exists for.
    let j = Quaternion::basis(2);
    let mut m = GateMatrix::<Quaternion>::identity(2).unwrap();
    m.set(1, 1, j);
    let mut sim = Simulator::<Quaternion>::new();
    sim.registry_mut()
        .register_fixed("jphase", "diag(1, j)", m)
        .unwrap();

    let mut c: Circuit<Quaternion> = Circuit::new(1);
    c.h(0).gate("jphase", Vec::new(), vec![0]);
    let state = sim.run(&c).unwrap();
    // Amplitude of |1⟩ is j/√2: coefficients [0, 0, 1/√2, 0].
    let coeffs = state.amplitude(1).coeffs();
    assert!((coeffs[2] - std::f64::consts::FRAC_1_SQRT_2).abs() < TOL);
    assert!(
        (state.total_weight() - 1.0).abs() < TOL,
        "j-phase preserves Born weight"
    );
}
