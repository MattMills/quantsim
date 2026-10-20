//! Krein registers, measured: Pauli-string metrics and their split
//! signatures; pseudo-unitary gates accepted and ordinary unitaries that
//! break the metric refused; the ghost pump (conserved η-norm, growing
//! path weight); twistor space as two qubits with `Φ = X` on chirality;
//! `su(2, 2)` solved for and identified by its Killing form; and the
//! infinity twistor cutting it down to Poincaré, AdS₄ or dS₄, with the
//! AdS₄ isometries running as two-qubit gates.

mod common;

use quantsim::gates::Pauli;
use quantsim::krein::{
    exponentiate, infinity_twistor, killing_signature, stabilizer, su22_basis, transpose,
    twistor_form,
};
use quantsim::prelude::*;

const TOL: f64 = 1e-9;

fn z(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn random_state(n: usize, seed: u64) -> Box<dyn Backend<C64>> {
    let mut rng = Prng::new(seed);
    let entries: Vec<(u64, C64)> = (0..1u64 << n)
        .map(|i| (i, z(rng.next_f64() - 0.5, rng.next_f64() - 0.5)))
        .collect();
    let mut s = DenseState::<C64>::new(n).unwrap();
    s.load(&entries).unwrap();
    Box::new(s)
}

fn sig(s: FormSignature) -> (usize, usize, usize) {
    (s.positive, s.negative, s.null)
}

fn diff(a: &GateMatrix<C64>, b: &GateMatrix<C64>) -> f64 {
    a.data()
        .iter()
        .zip(b.data())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).norm()))
}

// ── metrics ──────────────────────────────────────────────────────────

#[test]
fn pauli_metrics_have_split_signature() {
    let eta = KreinMetric::new(&[(0, Pauli::Z), (2, Pauli::X)]).unwrap();
    assert_eq!(eta.signature(3), (4, 4));
    assert_eq!(KreinMetric::new(&[]).unwrap().signature(3), (8, 0));
    assert_eq!(KreinMetric::twistor(1).signature(2), (2, 2));
    // Identity entries drop out; duplicates are refused.
    assert_eq!(KreinMetric::new(&[(1, Pauli::I)]).unwrap().ops().len(), 0);
    assert!(KreinMetric::new(&[(0, Pauli::Z), (0, Pauli::X)]).is_err());
}

#[test]
fn the_twistor_metric_is_phi() {
    let restricted = KreinMetric::twistor(1).restricted(&[0, 1]).unwrap();
    assert!(restricted.approx_eq(&twistor_form(), 0.0));
}

#[test]
fn the_pairing_agrees_with_the_ledger() {
    for (k, eta) in [
        KreinMetric::twistor(1),
        KreinMetric::new(&[(0, Pauli::Y), (1, Pauli::Z)]).unwrap(),
    ]
    .into_iter()
    .enumerate()
    {
        let psi = random_state(2, 10 + k as u64);
        let pair = eta.pair(psi.as_ref(), psi.as_ref()).unwrap();
        let reg = KreinRegister::new(psi, eta).unwrap();
        let led = reg.ledger().unwrap();
        common::assert_close(pair.re, led.eta, TOL);
        assert!(pair.im.abs() < TOL, "η is Hermitian");
        common::assert_close(led.positive + led.negative, led.path, TOL);
        assert!(led.positive >= -TOL && led.negative >= -TOL);
    }
}

// ── the register ─────────────────────────────────────────────────────

#[test]
fn the_ghost_pump_conserves_the_eta_norm_and_grows_the_path_weight() {
    let eta = KreinMetric::new(&[(0, Pauli::Z)]).unwrap();
    let boost = eta.boost(&[(0, Pauli::X)], 0.8).unwrap();
    assert!(boost.unitarity_deviation() > 0.1, "a boost is not unitary");
    assert!(eta.pseudo_unitarity_deviation(&boost, &[0]).unwrap() < TOL);

    let mut reg = KreinRegister::new(random_state(3, 1), eta).unwrap();
    let start = reg.ledger().unwrap();
    let mut last = start;
    for step in 0..5 {
        reg.apply(&boost, &[0]).unwrap();
        let led = reg.ledger().unwrap();
        common::assert_close(led.eta, start.eta, 1e-8 * led.path);
        if step >= 1 {
            assert!(led.path > last.path, "path weight should grow");
            assert!(led.negative > last.negative, "ghost weight grows with it");
        }
        last = led;
    }
}

#[test]
fn the_register_refuses_gates_that_break_the_metric_and_passes_the_rest() {
    let eta = KreinMetric::new(&[(0, Pauli::Z)]).unwrap();
    // A boost whose generator commutes with η is not a Krein boost.
    assert!(eta.boost(&[(0, Pauli::Z)], 0.3).is_err());
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let h = &GateMatrix::from_vec(2, vec![z(r, 0.0), z(r, 0.0), z(r, 0.0), z(-r, 0.0)]).unwrap();
    let mut reg = KreinRegister::new(random_state(2, 2), eta).unwrap();
    // H on the metric qubit mixes the sectors: refused, state untouched.
    let before = reg.state().amplitude(1);
    let err = reg.apply(h, &[0]).unwrap_err();
    assert!(format!("{err}").contains("not pseudo-unitary"));
    assert_eq!(reg.state().amplitude(1), before);
    // H on the other qubit is fine; so is a phase on the metric qubit.
    reg.apply(h, &[1]).unwrap();
    let s =
        GateMatrix::from_vec(2, vec![z(1.0, 0.0), z(0.0, 0.0), z(0.0, 0.0), z(0.0, 1.0)]).unwrap();
    reg.apply(&s, &[0]).unwrap();
}

// ── twistor space: su(2,2) and the infinity twistor ──────────────────

#[test]
fn su22_is_fifteen_dimensional_and_its_killing_form_names_so_4_2() {
    let basis = su22_basis();
    assert_eq!(basis.len(), 15);
    let phi = twistor_form();
    for x in &basis {
        // X†Φ + ΦX = 0, traceless.
        let a = x.dagger().matmul(&phi);
        let b = phi.matmul(x);
        let s: Vec<C64> = a.data().iter().zip(b.data()).map(|(p, q)| p + q).collect();
        assert!(s.iter().all(|v| v.norm() < TOL));
        let tr: C64 = (0..4).map(|k| x.get(k, k)).sum();
        assert!(tr.norm() < TOL);
    }
    // so(4,2): maximal compact so(4) ⊕ so(2) is 7-dimensional.
    assert_eq!(sig(killing_signature(&basis).unwrap()), (8, 7, 0));
}

#[test]
fn the_infinity_twistor_decides_the_spacetime() {
    let cases = [
        (z(0.0, 0.0), 10, (3, 3, 4)),  // Minkowski: Poincaré
        (z(-1.0, 0.0), 10, (6, 4, 0)), // AdS₄: so(3,2)
        (z(-3.0, 0.0), 10, (6, 4, 0)),
        (z(1.0, 0.0), 10, (4, 6, 0)), // dS₄: so(4,1)
        (z(2.5, 0.0), 10, (4, 6, 0)),
        (z(0.0, 1.0), 6, (3, 3, 0)), // breaks reality: only Lorentz
    ];
    for (lambda, dim, killing) in cases {
        let alg = stabilizer(&infinity_twistor(lambda));
        assert_eq!(alg.len(), dim, "λ = {lambda}");
        assert_eq!(
            sig(killing_signature(&alg).unwrap()),
            killing,
            "λ = {lambda}"
        );
    }
}

fn translation(sigma: [[C64; 2]; 2]) -> GateMatrix<C64> {
    // ω ↦ ω + i a π: the generator [[0, i a], [0, 0]].
    let o = z(0.0, 0.0);
    let i = z(0.0, 1.0);
    let data = vec![
        o,
        o,
        i * sigma[0][0],
        i * sigma[0][1],
        o,
        o,
        i * sigma[1][0],
        i * sigma[1][1],
        o,
        o,
        o,
        o,
        o,
        o,
        o,
        o,
    ];
    GateMatrix::from_vec(4, data).unwrap()
}

fn preserves(x: &GateMatrix<C64>, i: &GateMatrix<C64>) -> bool {
    let a = transpose(x).matmul(i);
    let b = i.matmul(x);
    a.data()
        .iter()
        .zip(b.data())
        .all(|(p, q)| (p + q).norm() < TOL)
}

#[test]
fn translations_preserve_the_flat_infinity_twistor_and_not_the_ads_one() {
    let (o, l) = (z(0.0, 0.0), z(1.0, 0.0));
    let paulis = [
        [[l, o], [o, l]],
        [[o, l], [l, o]],
        [[o, z(0.0, -1.0)], [z(0.0, 1.0), o]],
        [[l, o], [o, -l]],
    ];
    let flat = infinity_twistor(o);
    let ads = infinity_twistor(-l);
    for s in paulis {
        let t = translation(s);
        assert!(preserves(&t, &flat));
        assert!(!preserves(&t, &ads));
    }
    // The dilation preserves neither: it rescales the infinity twistor.
    let h = z(0.5, 0.0);
    let mut d = GateMatrix::<C64>::zeros(4).unwrap();
    for (k, v) in [h, h, -h, -h].into_iter().enumerate() {
        d.set(k, k, v);
    }
    assert!(!preserves(&d, &flat) && !preserves(&d, &ads));
}

#[test]
fn ads4_isometries_run_on_a_twistor_register_and_ordinary_unitarity_refuses_some() {
    let i_ads = infinity_twistor(z(-1.0, 0.0));
    let alg = stabilizer(&i_ads);
    // Spinor qubit 0, chirality qubit 1; a third bystander qubit.
    let mut reg = KreinRegister::new(random_state(3, 3), KreinMetric::twistor(1)).unwrap();
    let start = reg.ledger().unwrap();
    let mut non_unitary = 0;
    for (k, x) in alg.iter().enumerate() {
        let g = exponentiate(x, 0.3 + 0.1 * k as f64).unwrap();
        // The group element preserves Φ and the infinity twistor.
        assert!(
            reg.metric()
                .pseudo_unitarity_deviation(&g, &[0, 1])
                .unwrap()
                < 1e-8
        );
        assert!(diff(&transpose(&g).matmul(&i_ads).matmul(&g), &i_ads) < 1e-8);
        if g.unitarity_deviation() > 1e-6 {
            non_unitary += 1;
        }
        reg.apply(&g, &[0, 1]).unwrap();
    }
    // so(3,2) is non-compact: elements off its compact so(3) ⊕ so(2)
    // exponentiate to non-unitaries, which is why an ordinary register
    // cannot run AdS₄.
    assert!(non_unitary >= 6, "{non_unitary} non-unitary generators");
    let end = reg.ledger().unwrap();
    common::assert_close(end.eta, start.eta, 1e-7 * end.path);
    assert!(
        (end.path - start.path).abs() > 1e-3,
        "non-compact gates move the path weight"
    );
}

#[test]
fn a_null_twistor_stays_null() {
    // Φ(Z) = 0: a real light ray. (ω, π) = (1, 0, 0, i)/√2 has
    // Φ = ω⁰π̄₀ + ω¹π̄₁ + c.c. = 0.
    let mut s = DenseState::<C64>::new(2).unwrap();
    let r = std::f64::consts::FRAC_1_SQRT_2;
    s.load(&[(0, z(r, 0.0)), (3, z(0.0, r))]).unwrap();
    let mut reg = KreinRegister::new(Box::new(s), KreinMetric::twistor(1)).unwrap();
    assert!(reg.ledger().unwrap().eta.abs() < TOL);
    for (k, x) in su22_basis().iter().enumerate() {
        let g = exponentiate(x, 0.2 * (k as f64 + 1.0)).unwrap();
        reg.apply(&g, &[0, 1]).unwrap();
    }
    let led = reg.ledger().unwrap();
    assert!(led.eta.abs() < 1e-7 * led.path, "Φ = {}", led.eta);
}
