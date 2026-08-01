//! Self-embedded evolution: that the drill really is the Jacobian of the
//! drive, that a recorded run is provably not a circuit, that gradient
//! control converges where proportional control cannot — and where the
//! second loop earns its place and where it does not.

use quantsim::circuit::{BoundGate, GateKernel};
use quantsim::prelude::*;

fn bind(c: &Circuit<C64>, reg: &GateRegistry<C64>) -> Vec<BoundGate<C64>> {
    c.bind(reg).unwrap().gates().to_vec()
}

fn run_gates(st: &mut DenseState<C64>, gates: &[BoundGate<C64>]) {
    for g in gates {
        match &g.kernel {
            GateKernel::Matrix(m) => st.apply(m, &g.qubits).unwrap(),
            GateKernel::Diagonal(d) => st.apply_diagonal(d, &g.qubits).unwrap(),
        }
    }
}

fn layer(n: usize, reg: &GateRegistry<C64>) -> Vec<BoundGate<C64>> {
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.gate("h", vec![], vec![q]);
    }
    for q in (0..n - 1).step_by(2) {
        c.gate("cx", vec![], vec![q, q + 1]);
    }
    bind(&c, reg)
}

/// A state with genuine phases. A *real* state sits at a stationary
/// point of every diagonal drive, so a real preparation would make the
/// whole problem degenerate — see the test that pins that.
fn prepared(n: usize, reg: &GateRegistry<C64>, seed: u64) -> DenseState<C64> {
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.gate("ry", vec![0.3 + 0.1 * (rng.next_u64() % 7) as f64], vec![q]);
    }
    for q in 0..n {
        c.gate("rz", vec![0.2 + 0.13 * (rng.next_u64() % 9) as f64], vec![q]);
    }
    for q in 0..n - 1 {
        c.gate("cx", vec![], vec![q, q + 1]);
    }
    let mut st = DenseState::<C64>::new(n).unwrap();
    run_gates(&mut st, &bind(&c, reg));
    st
}

// ── the actuator ─────────────────────────────────────────────────────

#[test]
fn the_diagonal_built_from_harmonics_is_the_gate_it_stands_for() {
    let n = 7usize;
    let em = Emission {
        event: 0,
        coefficients: vec![(1, 0.3), (5, -0.7), (6, 1.1), (127, 0.05)],
        fired: 4,
        error: 0.0,
        gain: 0.0,
    };
    let d = em.diagonal(n);
    for x in 0..1u64 << n {
        let mut phi = 0.0;
        for &(s, t) in &em.coefficients {
            phi += if (x & s).count_ones() % 2 == 0 { t } else { -t };
        }
        let (want_re, want_im) = (phi.cos(), phi.sin());
        let got = d[x as usize];
        assert!(
            (got.re - want_re).abs() < 1e-12 && (got.im - want_im).abs() < 1e-12,
            "basis state {x}"
        );
        assert!(
            (got.re * got.re + got.im * got.im - 1.0).abs() < 1e-12,
            "the actuator must stay unitary"
        );
    }
}

// ── the drill is the Jacobian ────────────────────────────────────────

#[test]
fn the_drill_is_the_jacobian_of_the_drive() {
    // The identity the whole design rests on: the derivative of a pure
    // string with respect to the drive is the *mixed* string, so the
    // sensor and the Jacobian are one transform. Checked against finite
    // differences rather than trusted from the derivation.
    let reg = GateRegistry::<C64>::standard();
    let n = 7usize;
    let target = Target::x(0b000_0011, 0.5);
    let mut m = Reflexive::from_state(prepared(n, &reg, 1));
    let view = m.sense().unwrap();
    let law = Gradient::new(target, 0.0);
    let jac = law.jacobian(&view);
    let base = view.x_string(target.a);

    let eps = 1e-6;
    let mut worst = 0.0f64;
    for s in [1u64, 3, 5, 11, 32, 64, 100, 127] {
        let em = Emission {
            event: 0,
            coefficients: vec![(s, eps)],
            fired: 0,
            error: 0.0,
            gain: 0.0,
        };
        let mut probe = Reflexive::from_state(prepared(n, &reg, 1));
        probe.apply(&em).unwrap();
        let fd = (probe.sense().unwrap().x_string(target.a) - base) / eps;
        worst = worst.max((fd - jac[s as usize]).abs());
    }
    assert!(worst < 1e-4, "analytic vs numeric Jacobian: {worst:e}");
}

#[test]
fn a_real_state_has_no_first_order_response_to_a_phase_drive() {
    // Found by measurement, not by thinking: with real amplitudes the
    // mixed strings are real, the Jacobian is their imaginary part, and
    // the drive can do nothing at first order. A preparation without
    // phases makes the control problem degenerate.
    let reg = GateRegistry::<C64>::standard();
    let n = 6usize;
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.gate("ry", vec![0.5 + 0.1 * q as f64], vec![q]);
    }
    for q in 0..n - 1 {
        c.gate("cx", vec![], vec![q, q + 1]);
    }
    let mut st = DenseState::<C64>::new(n).unwrap();
    run_gates(&mut st, &bind(&c, &reg));

    let mut m = Reflexive::from_state(st);
    let view = m.sense().unwrap();
    let law = Gradient::new(Target::x(0b000_011, 0.5), 0.2);
    for (s, g) in law.jacobian(&view).iter().enumerate() {
        assert!(g.abs() < 1e-12, "harmonic {s} moved a real state: {g}");
    }
    // and the law emits nothing rather than thrashing
    let mut law = law;
    assert!(
        law.phases(&view, &view.fired(1e-12), 0).is_empty(),
        "an unsteerable target must produce no drive"
    );
}

// ── it is not a circuit ──────────────────────────────────────────────

#[test]
fn a_recorded_run_is_not_a_circuit_and_the_difference_is_measured() {
    // `replay` turns the emitted gates into an ordinary fixed circuit.
    // From the state the run started in it reproduces the run exactly;
    // from any other state it does not, because the gates were computed
    // from a state that is no longer there. That is the whole claim,
    // and it is a measurement.
    let reg = GateRegistry::<C64>::standard();
    let n = 7usize;
    let lay = layer(n, &reg);
    let target = Target::x(0b000_0011, 0.5);

    let mut m = Reflexive::from_state(prepared(n, &reg, 1));
    let mut law = Gradient::new(target, 0.4);
    let run = m.run(&mut law, &lay, 10).unwrap();

    let same = replay(&run.trace, &lay, prepared(n, &reg, 1)).unwrap();
    assert_eq!(
        deviation(&same, m.state()).unwrap(),
        0.0,
        "from its own initial state the record IS the program"
    );

    let mut diverged = 0usize;
    for seed in [2u64, 3, 4, 5] {
        let replayed = replay(&run.trace, &lay, prepared(n, &reg, seed)).unwrap();
        let mut other = Reflexive::from_state(prepared(n, &reg, seed));
        let mut l = Gradient::new(target, 0.4);
        other.run(&mut l, &lay, 10).unwrap();
        let d = deviation(&replayed, other.state()).unwrap();
        assert!(
            d > 1e-3,
            "seed {seed}: the compiled circuit matched the program, so nothing was proved"
        );
        diverged += 1;
    }
    assert_eq!(diverged, 4);
}

// ── the control laws ─────────────────────────────────────────────────

#[test]
fn gradient_control_converges_where_proportional_control_cannot() {
    let reg = GateRegistry::<C64>::standard();
    let n = 7usize;
    let lay = layer(n, &reg);
    let target = Target::x(0b000_0011, 0.5);
    let events = 40;

    // Proportional control drives along ⟨X_S⟩, which is not the
    // derivative of anything — it never settles at any gain.
    for gain in [0.1f64, 0.6, 1.6] {
        let mut m = Reflexive::from_state(prepared(n, &reg, 1));
        let mut l = Proportional::new(target, gain);
        let r = m.run(&mut l, &lay, events).unwrap();
        assert!(
            r.settled_at(0.05).is_none(),
            "proportional at gain {gain} unexpectedly settled — the comparison is void"
        );
    }

    // Gradient control drives along the measured Jacobian and does.
    let mut m = Reflexive::from_state(prepared(n, &reg, 1));
    let mut l = Gradient::new(target, 0.4);
    let r = m.run(&mut l, &lay, events).unwrap();
    assert!(
        r.settled_at(0.05).is_some(),
        "gradient control failed to settle: final error {}",
        r.final_error()
    );
    assert!(r.final_error() < 0.02, "final error {}", r.final_error());
}

#[test]
fn the_second_loop_rescues_a_gain_that_could_not_have_been_chosen() {
    // The honest scope of the second-order layer. It does NOT beat a
    // well-tuned fixed gain. What it does is recover from a badly
    // chosen one — and the gain cannot be chosen in advance, because
    // the right value depends on the state, which is exactly what a
    // reflexive program does not know before it runs.
    let reg = GateRegistry::<C64>::standard();
    let n = 7usize;
    let lay = layer(n, &reg);
    let target = Target::x(0b000_0011, 0.5);
    let events = 40;
    let starved = 0.05;

    let mut m = Reflexive::from_state(prepared(n, &reg, 1));
    let mut first = Gradient::new(target, starved);
    let flat = m.run(&mut first, &lay, events).unwrap();
    assert!(
        flat.settled_at(0.05).is_none(),
        "the starved gain should not settle, or there is nothing to rescue"
    );

    let mut m = Reflexive::from_state(prepared(n, &reg, 1));
    let mut second = Adaptive::new(Gradient::new(target, starved), 0.2, 0.5, 4.0);
    let adapted = m.run(&mut second, &lay, events).unwrap();
    assert!(
        adapted.settled_at(0.05).is_some(),
        "the second loop did not rescue it: final error {}",
        adapted.final_error()
    );
    assert!(
        adapted.final_error() < flat.final_error(),
        "adapted {} against flat {}",
        adapted.final_error(),
        flat.final_error()
    );
    assert!(
        second.gain() > starved,
        "the controller should have climbed off the starved gain: {}",
        second.gain()
    );
}

#[test]
fn a_controller_with_no_patience_pins_its_own_gain_at_the_floor() {
    // Kept as a test because it is how the first version failed:
    // descent through a mixing layer is a noisy signal, and cutting on
    // every single rise drives the gain down and never lets it back.
    let reg = GateRegistry::<C64>::standard();
    let n = 7usize;
    let lay = layer(n, &reg);
    let target = Target::x(0b000_0011, 0.5);

    let mut m = Reflexive::from_state(prepared(n, &reg, 1));
    let mut impatient =
        Adaptive::new(Gradient::new(target, 0.4), 0.2, 0.5, 4.0).with_patience(0);
    m.run(&mut impatient, &lay, 40).unwrap();

    let mut m = Reflexive::from_state(prepared(n, &reg, 1));
    let mut patient = Adaptive::new(Gradient::new(target, 0.4), 0.2, 0.5, 4.0);
    m.run(&mut patient, &lay, 40).unwrap();

    assert!(
        patient.gain() > impatient.gain(),
        "patience {} should end above impatience {}",
        patient.gain(),
        impatient.gain()
    );
}

// ── what the loop costs ──────────────────────────────────────────────

#[test]
fn the_drive_cannot_inflate_its_own_cost() {
    // The mass is blind to phases, so the fired set is invariant under
    // the drive: the loop cannot make itself more expensive. Only the
    // mixing layers, which the program did not choose, can.
    let reg = GateRegistry::<C64>::standard();
    let n = 7usize;
    let mut m = Reflexive::from_state(prepared(n, &reg, 1));
    let before = m.sense().unwrap().fired(1e-12).len();
    let mut law = Gradient::new(Target::x(0b000_0011, 0.5), 0.6);
    for e in 0..5 {
        m.event(&mut law, e).unwrap();
        assert_eq!(
            m.sense().unwrap().fired(1e-12).len(),
            before,
            "event {e} changed the drive's own support"
        );
    }
}

#[test]
fn the_drive_lives_on_half_the_harmonics_by_a_selection_rule() {
    // `d⟨X_A⟩/dθ_S` vanishes whenever `|A ∧ S|` is even, so half the
    // harmonics are structurally unreachable — a parity selection rule,
    // not a property of the state.
    let reg = GateRegistry::<C64>::standard();
    let n = 7usize;
    let mut m = Reflexive::from_state(prepared(n, &reg, 1));
    let view = m.sense().unwrap();
    let target = Target::x(0b000_0011, 0.5);
    let jac = Gradient::new(target, 0.2).jacobian(&view);
    for (s, g) in jac.iter().enumerate() {
        if (target.a & s as u64).count_ones() % 2 == 0 {
            assert_eq!(*g, 0.0, "even overlap {s} should be structurally silent");
        }
    }
    let live = jac.iter().filter(|g| g.abs() > 1e-12).count();
    assert!(live <= 1 << (n - 1), "{live} exceeds the parity half");
    assert!(live > 0, "nothing was steerable, so the test proves nothing");
}
