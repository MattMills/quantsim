//! Resolvers: representations defined by what they answer rather than by
//! what they store. That the three disagree about nothing but cost, that
//! the cone's cost tracks depth and not width, that a causally
//! disconnected question is answered exactly and for free — and where the
//! whole idea stops helping.

use quantsim::heisenberg::{tfim_trotter, Rotation};
use quantsim::prelude::*;
use quantsim::query::*;

fn registry() -> GateRegistry<C64> {
    GateRegistry::<C64>::standard()
}

/// Nearest-neighbour brickwork: the cone grows about two qubits a layer.
fn brickwork(n: usize, depth: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for d in 0..depth {
        for q in (d % 2..n.saturating_sub(1)).step_by(2) {
            c.gate("rzz", vec![0.31 + 0.01 * d as f64], vec![q, q + 1]);
            c.gate("rx", vec![0.47], vec![q]);
            c.gate("rx", vec![0.41], vec![q + 1]);
        }
    }
    c
}

/// Every pair coupled in every layer — the shape with no locality to use.
fn all_to_all(n: usize, depth: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for d in 0..depth {
        for a in 0..n {
            for b in (a + 1)..n {
                c.gate("rzz", vec![0.21 + 0.01 * d as f64], vec![a, b]);
            }
        }
        for q in 0..n {
            c.gate("rx", vec![0.37], vec![q]);
        }
    }
    c
}

/// The same TFIM circuit as rotations and as a `Circuit`.
fn tfim_both(n: usize, steps: usize) -> (Vec<Rotation>, Circuit<C64>) {
    let rots = tfim_trotter(n, 1.0, 0.7, 0.35, steps);
    let mut c = Circuit::new(n);
    for r in &rots {
        let s = r.support();
        if s.len() == 1 {
            if r.axis.0 != 0 {
                c.gate("rx", vec![r.theta], vec![s[0]]);
            } else {
                c.gate("rz", vec![r.theta], vec![s[0]]);
            }
        } else {
            c.gate("rzz", vec![r.theta], vec![s[0], s[1]]);
        }
    }
    (rots, c)
}

// ── they agree about everything but cost ─────────────────────────────

#[test]
fn three_resolvers_answer_the_same_question_in_three_currencies() {
    let reg = registry();
    for &(n, steps) in &[(8usize, 3usize), (12, 4), (14, 3)] {
        let (rots, circuit) = tfim_both(n, steps);
        for q in [0usize, n / 3, n / 2] {
            let ops = [(q, Pauli::Z)];
            let mut state = StateResolver::new(circuit.clone(), &reg, Inner::Dense);
            let mut cone = ConeResolver::new(circuit.clone(), &reg, Inner::Dense);
            let mut heis = HeisenbergResolver::new(rots.clone(), n);

            let a = state.expectation(&ops).unwrap();
            let b = cone.expectation(&ops).unwrap();
            let c = heis.expectation(&ops).unwrap();
            assert!((a.value - b.value).abs() < 1e-11, "n={n} q={q}: state vs cone");
            assert!(
                (a.value - c.value).abs() < 1e-11,
                "n={n} q={q}: state vs heisenberg"
            );

            // and the costs are in genuinely different currencies
            assert_eq!(a.cost.qubits, n, "the state resolver pays the full width");
            assert!(b.cost.qubits <= n);
            assert_eq!(
                c.cost.qubits, 0,
                "the Heisenberg resolver instantiates no register at all"
            );
        }
    }
}

#[test]
fn a_resolver_that_could_not_be_a_backend_still_answers() {
    // The structural claim of the module: `HeisenbergResolver` has no
    // amplitudes, no enumerable support and no basis to load into, so no
    // refactoring of `Backend` admits it — and it answers anyway.
    let n = 10;
    let (rots, _) = tfim_both(n, 4);
    let mut heis = HeisenbergResolver::new(rots, n);
    let a = heis.expectation(&[(5, Pauli::Z)]).unwrap();
    assert_eq!(a.cost.qubits, 0);
    assert_eq!(a.cost.dim(), 1, "2^0 — no space was simulated");
    assert!(a.cost.bytes > 0, "it costs something, in another currency");
    assert!(a.value.abs() <= 1.0 + 1e-12);
    assert_eq!(heis.last_error_bound, 0.0, "an exact walk discards nothing");
}

// ── the cost tracks depth, not width ─────────────────────────────────

#[test]
fn the_cones_cost_is_flat_in_width_at_fixed_depth() {
    // The sub-exponentiality claim, stated as a measurement: the same
    // local question on a wider register costs the same, because the
    // cone did not get wider — only the register did.
    let reg = registry();
    let depth = 4;
    // A FIXED site across every width. Not `n/2` — brickwork alternates
    // which pairs couple per layer, so an even and an odd site sit in
    // genuinely different local environments and their answers differ
    // for reasons that have nothing to do with the register's width.
    let site = 6usize;
    let mut widths = Vec::new();
    for &n in &[12usize, 20, 30, 40] {
        let c = brickwork(n, depth);
        let mut cone = ConeResolver::new(c, &reg, Inner::Dense);
        let a = cone.expectation(&[(site, Pauli::Z)]).unwrap();
        widths.push((n, a.cost.qubits, a.value));
    }
    let first = widths[0].1;
    for &(n, cone_qubits, _) in &widths {
        assert_eq!(
            cone_qubits, first,
            "n={n}: the cone should not widen with the register"
        );
    }
    // the answers agree too — same physics, same cone
    for w in widths.windows(2) {
        assert!(
            (w[0].2 - w[1].2).abs() < 1e-11,
            "the answer moved with the width: {widths:?}"
        );
    }
    // and the compression that buys
    let c = brickwork(40, depth);
    let mut cone = ConeResolver::new(c, &reg, Inner::Dense);
    let a = cone.expectation(&[(20, Pauli::Z)]).unwrap();
    assert!(
        a.cost.compression() > 1e9,
        "compression only {:.0}x",
        a.cost.compression()
    );
}

#[test]
fn the_cone_widens_with_depth_which_is_the_real_variable() {
    let reg = registry();
    let n = 40;
    let mut last = 0usize;
    for depth in [2usize, 4, 6, 8] {
        let c = brickwork(n, depth);
        let cone = ConeResolver::new(c, &reg, Inner::Dense);
        let w = cone.cone_width(&[n / 2]);
        assert!(
            w >= last,
            "the cone cannot shrink with depth: {w} after {last}"
        );
        last = w;
    }
    assert!(last > 8, "depth 8 should have opened the cone: {last}");
}

#[test]
fn a_query_beyond_dense_reach_is_answered_anyway() {
    // 48 qubits: a full state vector is 2^48 amplitudes. The cone is not.
    let reg = registry();
    let n = 48;
    let c = brickwork(n, 6);
    let mut cone = ConeResolver::new(c, &reg, Inner::Dense);
    assert!(
        DenseState::<C64>::new(n).is_err(),
        "dense must refuse this width, or the test proves nothing"
    );
    let a = cone.expectation(&[(24, Pauli::Z)]).unwrap();
    assert!(a.cost.qubits <= 16, "cone {} qubits", a.cost.qubits);
    assert!(a.value.abs() <= 1.0 + 1e-12);
    assert!(a.cost.compression() > 1e10);
}

// ── free answers, and refusals ───────────────────────────────────────

#[test]
fn a_causally_disconnected_perturbation_is_exactly_zero_for_free() {
    let reg = registry();
    let n = 40;
    let c = brickwork(n, 4);
    let mut cone = ConeResolver::new(c, &reg, Inner::Dense);
    let ops = [(20usize, Pauli::Z)];

    let mut free = 0usize;
    for site in [26usize, 30, 35, 39, 0, 5] {
        let rot = Rotation {
            theta: std::f64::consts::PI,
            axis: (1u64 << site, 0),
        };
        let r = cone.response(0, site, rot, &ops).unwrap();
        assert_eq!(r.value, 0.0, "site {site} is outside the cone");
        assert!(r.cost.is_free(), "site {site} cost {:?}", r.cost);
        free += 1;
    }
    assert_eq!(free, 6);
}

#[test]
fn a_connected_perturbation_moves_the_answer_and_matches_the_full_state() {
    let reg = registry();
    let n = 14;
    let c = brickwork(n, 4);
    let ops = [(7usize, Pauli::Z)];
    let mut cone = ConeResolver::new(c.clone(), &reg, Inner::Dense);
    let mut state = StateResolver::new(c, &reg, Inner::Dense);

    let mut moved = 0usize;
    for site in [7usize, 6, 8] {
        let rot = Rotation {
            theta: std::f64::consts::PI,
            axis: (1u64 << site, 0),
        };
        let a = cone.response(0, site, rot, &ops).unwrap();
        let b = state.response(0, site, rot, &ops).unwrap();
        assert!(
            (a.value - b.value).abs() < 1e-11,
            "site {site}: cone {} vs state {}",
            a.value,
            b.value
        );
        if a.value.abs() > 1e-9 {
            moved += 1;
        }
        assert!(!a.cost.is_free(), "a connected site must cost something");
    }
    assert!(moved > 0, "no perturbation inside the cone did anything");
}

#[test]
fn a_cone_wider_than_the_budget_is_refused_and_says_so() {
    let reg = registry();
    let c = brickwork(30, 12);
    let mut cone = ConeResolver::new(c, &reg, Inner::Dense).with_max_cone(8);
    match cone.expectation(&[(15, Pauli::Z)]) {
        Err(Error::TooManyQubits { requested, max }) => {
            assert!(requested > 8);
            assert_eq!(max, 8);
        }
        other => panic!("expected a measured refusal, got {other:?}"),
    }
}

// ── where it stops helping ───────────────────────────────────────────

#[test]
fn all_to_all_coupling_leaves_the_cone_nothing_to_cut() {
    // The honest boundary. One layer of all-to-all gates puts every qubit
    // in every cone, so the cone resolver simulates the whole register
    // and buys exactly nothing. Locality is the resource; without it
    // there is no saving, and the report says so rather than implying
    // otherwise.
    let reg = registry();
    let n = 12;
    let c = all_to_all(n, 2);
    let mut cone = ConeResolver::new(c.clone(), &reg, Inner::Dense);
    let a = cone.expectation(&[(6, Pauli::Z)]).unwrap();
    assert_eq!(a.cost.qubits, n, "an all-to-all layer has no outside");
    assert_eq!(a.cost.compression(), 1.0, "and therefore no compression");

    // the answer is still right — it just costs the full width
    let mut state = StateResolver::new(c, &reg, Inner::Dense);
    assert!((a.value - state.expectation(&[(6, Pauli::Z)]).unwrap().value).abs() < 1e-11);
}

#[test]
fn a_wide_observable_widens_its_own_cone() {
    // Cost follows the question, not just the circuit: asking about many
    // sites at once unions their cones.
    let reg = registry();
    let n = 30;
    let c = brickwork(n, 4);
    let cone = ConeResolver::new(c, &reg, Inner::Dense);
    let one = cone.cone_width(&[15]);
    let spread = cone.cone_width(&[3, 15, 27]);
    assert!(
        spread > one,
        "three separated sites should cost more than one: {spread} vs {one}"
    );
    assert!(spread <= 3 * one, "and no more than their union");
}

#[test]
fn nothing_runs_until_a_question_arrives() {
    // Constructing a resolver on a circuit far past dense reach must not
    // touch a single amplitude — the whole point of recording rather
    // than evolving.
    let reg = registry();
    let c = brickwork(60, 8);
    let cone = ConeResolver::new(c, &reg, Inner::Dense);
    assert_eq!(cone.width(), 60);
    // asking about the cone's size is structural, not a simulation
    assert!(cone.cone_width(&[30]) < 60);
}
