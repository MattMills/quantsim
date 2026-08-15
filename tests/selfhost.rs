//! The self-hosting object, measured: each layer of E8 volumes buys
//! exactly one degree, the diagonal becomes linear, the route is exact
//! against the registry's own gates, and the cost it trades is counted
//! rather than asserted.

use quantsim::prelude::*;
use quantsim::selfhost::{
    amortization, full_degree, recursive_expansion, verify, Diagonal, SelfHostedStack,
    VOLUME_COORDINATES,
};

fn cz() -> Diagonal {
    let mut d = Diagonal::new(1).expect("bits in range");
    d.term(0b011, 1);
    d
}

fn ccz() -> Diagonal {
    let mut d = Diagonal::new(1).expect("bits in range");
    d.term(0b111, 1);
    d
}

fn t() -> Diagonal {
    let mut d = Diagonal::new(3).expect("bits in range");
    d.term(0b001, 1);
    d
}

#[test]
fn each_layer_of_volumes_buys_exactly_one_degree() {
    let sim = Simulator::<C64>::new();
    let reports = recursive_expansion(&sim, "sparse", 8, 6).expect("sweep");
    assert_eq!(reports.len(), 5, "degrees 2 through 6");
    for r in &reports {
        assert_eq!(
            r.depth,
            r.original_degree as usize - 1,
            "degree {} should need exactly {} layers, got {}",
            r.original_degree,
            r.original_degree - 1,
            r.depth
        );
        assert_eq!(
            r.linear_degree, 1,
            "the whole point is that the diagonal comes out linear"
        );
        assert_eq!(
            r.deviation, 0.0,
            "the stack route must reproduce the substrate diagonal exactly, \
             not merely within tolerance"
        );
        assert!(r.exact());
        // One monomial per degree, so one volume per layer.
        assert_eq!(r.volumes, r.depth);
        assert_eq!(r.width, 8 + r.depth * VOLUME_COORDINATES);
    }
    // The recursion is progressive: width and depth both climb by one
    // volume per degree, and nothing collapses.
    let depths: Vec<usize> = reports.iter().map(|r| r.depth).collect();
    assert_eq!(depths, vec![1, 2, 3, 4, 5]);
}

#[test]
fn a_ccz_becomes_a_character_and_a_t_does_not() {
    let sim = Simulator::<C64>::new();

    // ccz: degree 3, genuinely entangling, and not a character of the
    // bit group — the residual is the maximum a phase can be off by.
    let report = verify(&sim, "sparse", 6, &ccz()).expect("verify ccz");
    assert_eq!(report.original_degree, 3);
    assert_eq!(report.linear_degree, 1);
    assert!(
        report.substrate_character_residual > 1.9,
        "ccz is far from a character on its own: {:.3e}",
        report.substrate_character_residual
    );
    assert!(
        report.became_character(),
        "on the stack it should BE one: residual {:.3e}",
        report.stack_character_residual
    );
    assert!(report.exact());

    // t: already degree 1, so the object correctly does nothing — no
    // layers, no volumes, no cost. But it does NOT become a character:
    // its phases are eighth roots of unity, not signs.
    let report = verify(&sim, "sparse", 6, &t()).expect("verify t");
    assert_eq!(report.depth, 0);
    assert_eq!(report.volumes, 0);
    assert_eq!(report.toffoli_equivalents, 0);
    assert_eq!(report.width, 6);
    assert!(report.exact());
    assert!(
        !report.became_character(),
        "a t phase factorizes into single-qubit gates without becoming ±1 valued"
    );
    // And the residual is the same sqrt(2) the constellation measures for
    // a t through the qubit path (tests/e8_across.rs) — the two modules
    // are pointing the same instrument at the same obstruction.
    assert!(
        (report.stack_character_residual - std::f64::consts::SQRT_2).abs() < 1e-9,
        "expected the sqrt(2) carry defect, got {:.6}",
        report.stack_character_residual
    );
}

#[test]
fn the_linearized_diagonal_agrees_with_the_registrys_own_ccz_gate() {
    let sim = Simulator::<C64>::new();
    let diagonal = ccz();
    let stack = SelfHostedStack::plan(3, &diagonal).expect("plan");
    let linear = stack.linearize(&diagonal).expect("linearize");
    assert!(linear.is_linear());

    // Ground truth: the registry's ccz on a uniform superposition.
    let mut direct = Circuit::new(3);
    for q in 0..3 {
        direct.h(q);
    }
    direct.gate("ccz", vec![], vec![0, 1, 2]);
    let truth = sim.run(&direct).expect("run ccz");

    // The stack route: expand a uniform substrate, apply single-qubit
    // phases only, and read the substrate amplitudes back.
    let amp = (1.0f64 / 8.0).sqrt();
    let mut state = sim
        .backends()
        .create("sparse", stack.width())
        .expect("stack width");
    state.reset();
    let entries: Vec<(u64, C64)> = (0..8u64)
        .map(|x| (stack.expand(x), C64::new(amp, 0.0)))
        .collect();
    state.load(&entries).expect("load");
    let (_, angles) = linear.single_qubit_angles().expect("linear");
    assert_eq!(angles.len(), 1, "one monomial, one phase gate");
    linear
        .kernel_circuit(stack.width())
        .expect("kernel")
        .bind(sim.registry())
        .expect("bind")
        .run(state.as_mut())
        .expect("run");

    let mut worst = 0.0f64;
    for x in 0..8u64 {
        let got = state.amplitude(stack.expand(x));
        worst = worst.max((got - truth.amplitude(x)).norm());
    }
    assert!(
        worst < 1e-12,
        "a single-qubit phase on the stack must equal the three-qubit ccz \
         on the substrate; worst deviation {worst:.3e}"
    );
}

#[test]
fn the_object_trades_entangling_gates_per_use_for_a_one_off_write() {
    let sim = Simulator::<C64>::new();
    let report = amortization(&sim, "sparse", 6, &ccz(), &[1, 2, 4, 8, 16]).expect("amortize");
    assert_eq!(report.direct_max_arity, 3, "ccz is a three-qubit diagonal");
    assert_eq!(
        report.stack_steady_max_arity, 1,
        "after the write, the steady state is single-qubit phases only"
    );
    assert_eq!(
        report.direct_entangling,
        vec![1, 2, 4, 8, 16],
        "the direct route pays its entangling gate every single use"
    );
    assert_eq!(
        report.stack_entangling,
        vec![1, 1, 1, 1, 1],
        "the stack pays once, no matter how often the diagonal is applied"
    );
    assert_eq!(
        report.counted_crossover,
        Some(2),
        "so the counted crossover is the second use"
    );
    assert_eq!(report.direct_nanos.len(), 5);
    assert_eq!(report.stack_nanos.len(), 5);
}

#[test]
fn the_per_use_slope_is_lower_even_where_the_constant_is_not() {
    // Honest counterpoint to the counted crossover: a classical
    // simulator applies a diagonal kernel in O(support) regardless of
    // arity, so the stack's win does not show up as raw speed at small
    // repetition counts — it shows up in the SLOPE, because the steady
    // state is cheaper per use than a multi-controlled phase.
    //
    // Asserted on the COUNTED entangling operations rather than on
    // nanoseconds. The two slopes differ by about 6% in wall-clock,
    // which is inside this machine's scheduling noise, so the timed
    // version flips ordering under parallel load — it is a measurement,
    // and `examples/selfhosted_stack.rs` prints it as one, beside these
    // counts.
    let sim = Simulator::<C64>::new();
    let reps = [8usize, 64, 256];
    let r = amortization(&sim, "sparse", 6, &ccz(), &reps).expect("amortize");
    let slope = |v: &[usize]| {
        (v[v.len() - 1] as f64 - v[0] as f64) / (reps[reps.len() - 1] - reps[0]) as f64
    };
    let (direct, stack) = (slope(&r.direct_entangling), slope(&r.stack_entangling));
    assert!(
        stack < direct,
        "the stack's per-use cost should be the cheaper one: {stack:.3} vs {direct:.3} per use"
    );
    assert_eq!(
        stack, 0.0,
        "and the steady state should cost no entangling operations at all"
    );
    assert_eq!(
        r.direct_entangling, r.repetitions,
        "while the direct route pays one multi-controlled phase per use"
    );
    // The other half of the original claim — that the stack's CONSTANT is
    // the more expensive one — has no counted form at all: the one-off
    // self-computation is a single entangling operation, so by this
    // ledger the stack is ahead from the first use. It is expensive only
    // in wall-clock, and that is exactly why it is measured in the
    // example rather than asserted here.
}

#[test]
fn volumes_are_quantized_to_eight_coordinates_and_occupancy_says_so() {
    // A dense degree-2 diagonal on six qubits has C(6,2) = 15 monomials,
    // which does not fit one E8 volume's eight coordinates.
    let diagonal = full_degree(6, 2, 1).expect("full degree");
    assert_eq!(diagonal.terms().len(), 15);
    let stack = SelfHostedStack::plan(6, &diagonal).expect("plan");
    assert_eq!(stack.depth(), 1, "one degree, one layer");
    assert_eq!(stack.volumes(), 2, "fifteen monomials need two volumes");
    assert_eq!(stack.width(), 6 + 2 * VOLUME_COORDINATES);
    assert_eq!(
        stack.layers()[0].spare(),
        1,
        "sixteen coordinates, fifteen used"
    );
    assert!((stack.occupancy() - 15.0 / 16.0).abs() < 1e-12);

    // A single monomial still reserves a whole volume, and the occupancy
    // reports that honestly rather than hiding the quantization.
    let sparse_stack = SelfHostedStack::plan(6, &cz()).expect("plan");
    assert_eq!(sparse_stack.volumes(), 1);
    assert_eq!(sparse_stack.layers()[0].spare(), 7);
    assert!((sparse_stack.occupancy() - 1.0 / 8.0).abs() < 1e-12);
}

#[test]
fn the_self_computation_is_the_feedback_and_it_round_trips() {
    let diagonal = ccz();
    let stack = SelfHostedStack::plan(4, &diagonal).expect("plan");
    let slot = stack.slot(0b111).expect("the degree-3 monomial is held");
    assert!(slot >= 4, "coordinates live above the substrate");
    for x in 0..16u64 {
        let y = stack.expand(x);
        assert_eq!(y & 0b1111, x, "the substrate is carried through unchanged");
        let expected = u64::from(x & 0b111 == 0b111);
        assert_eq!(
            (y >> slot) & 1,
            expected,
            "the coordinate must hold the monomial's own value"
        );
        assert!(
            stack.consistent(y),
            "an expanded index is on the object's image by construction"
        );
        // Flipping a coordinate leaves the image, and the object says so.
        assert!(!stack.consistent(y ^ (1 << slot)));
    }
}

#[test]
fn a_nonlinear_diagonal_refuses_to_factorize_and_an_unheld_monomial_refuses_to_linearize() {
    let diagonal = ccz();
    assert!(
        diagonal.single_qubit_angles().is_err(),
        "a degree-3 diagonal is not a product of single-qubit phases, and \
         saying so is the honest failure"
    );

    // A stack planned for cz cannot linearize ccz: it holds no
    // coordinate for the degree-3 monomial.
    let small = SelfHostedStack::plan(4, &cz()).expect("plan");
    assert!(small.linearize(&diagonal).is_err());
    assert!(small.linearize(&cz()).is_ok());

    // And a monomial outside the substrate is refused at planning time.
    let mut wide = Diagonal::new(1).expect("bits");
    wide.term(0b1_0000, 1);
    assert!(SelfHostedStack::plan(4, &wide).is_err());
}

#[test]
fn the_diagonal_representation_is_canonical_and_degree_is_honest() {
    let mut d = Diagonal::new(1).expect("bits");
    d.term(0b111, 1);
    assert_eq!(d.degree(), 3);
    // A coefficient that reduces to zero must drop the monomial, or the
    // degree would count a term contributing nothing.
    d.term(0b111, 1);
    assert_eq!(d.terms().len(), 0, "1 + 1 = 0 mod 2");
    assert_eq!(d.degree(), 0);
    assert!(d.is_linear());

    // Degree counts the largest monomial, not the term count.
    let mut mixed = Diagonal::new(3).expect("bits");
    mixed.term(0b001, 1).term(0b010, 3).term(0b110, 2);
    assert_eq!(mixed.degree(), 2);
    assert_eq!(mixed.nonlinear_monomials(), vec![0b110]);
    assert_eq!(mixed.multicontrolled_gates(), 1);
    // The empty monomial is a global phase, and is not a nonlinear term.
    mixed.term(0, 5);
    assert_eq!(mixed.multicontrolled_gates(), 1);
    assert_eq!(mixed.degree(), 2);
}

#[test]
fn the_route_is_exact_on_dense_as_well_as_sparse() {
    let sim = Simulator::<C64>::new();
    // Substrate 3 keeps the stack at 3 + 16 = 19 qubits, which dense can
    // hold — so the exactness claim is not a property of one backend.
    for backend in ["dense", "sparse", "adaptive"] {
        let report = verify(&sim, backend, 3, &ccz()).expect("verify");
        assert!(
            report.exact(),
            "{backend}: deviation {:.3e}",
            report.deviation
        );
        assert!(report.became_character());
    }
}

#[test]
fn a_mixed_degree_diagonal_fills_layers_by_degree() {
    // Degrees 2 and 3 together: two layers, each holding only its own
    // degree, so the depth is still degree - 1 rather than the term count.
    let mut d = Diagonal::new(1).expect("bits");
    d.term(0b0011, 1).term(0b0101, 1).term(0b0111, 1);
    let stack = SelfHostedStack::plan(4, &d).expect("plan");
    assert_eq!(stack.depth(), 2);
    assert_eq!(stack.layers()[0].degree, 2);
    assert_eq!(stack.layers()[0].monomials, vec![0b0011, 0b0101]);
    assert_eq!(stack.layers()[1].degree, 3);
    assert_eq!(stack.layers()[1].monomials, vec![0b0111]);

    let sim = Simulator::<C64>::new();
    let report = verify(&sim, "sparse", 4, &d).expect("verify");
    assert!(report.exact());
    assert!(report.became_character());
    assert_eq!(report.phase_gates, 3, "three monomials, three phases");
    // 1-controlled + 2-controlled writes: 1 + 1 + 3 Toffoli-equivalents.
    assert_eq!(report.toffoli_equivalents, 1 + 1 + 3);
}
