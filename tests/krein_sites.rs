//! Krein metrics over mixed-radix spaces, measured: unequal signatures a
//! Pauli string cannot reach; the product rule for signatures; agreement
//! with the qubit Krein register on qubit spaces; `so(p, q)` on one site
//! with its Killing form, cross-checked against the twistor algebras;
//! AdS₄ as a state on a `(3, 2)` site moved along its hyperboloid; the
//! qudit ghost pump; and refusal of gates that break the metric.

mod common;

use quantsim::gates::Pauli;
use quantsim::krein::su22_basis;
use quantsim::krein::{infinity_twistor, killing_signature, killing_signature_sites, stabilizer};
use quantsim::krein_sites::{exponentiate_site, so_algebra};
use quantsim::prelude::*;
use quantsim::space::fourier;

const TOL: f64 = 1e-9;

fn z(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn state_on(space: &Space, seed: u64) -> Box<dyn Backend<C64>> {
    let mut rng = Prng::new(seed);
    let dim = space.dim() as u64;
    let entries: Vec<(u64, C64)> = (0..dim)
        .map(|i| (i, z(rng.next_f64() - 0.5, rng.next_f64() - 0.5)))
        .collect();
    let mut s = BackendRegistry::<C64>::standard()
        .create_with_space("dense", space)
        .unwrap();
    s.load(&entries).unwrap();
    s
}

fn split_signs(p: usize, q: usize) -> Vec<i8> {
    (0..p + q).map(|k| if k < p { 1 } else { -1 }).collect()
}

fn sig(s: FormSignature) -> (usize, usize, usize) {
    (s.positive, s.negative, s.null)
}

// ── signatures ───────────────────────────────────────────────────────

#[test]
fn a_qudit_site_reaches_the_signatures_pauli_strings_cannot() {
    for (p, q) in [(3, 2), (3, 1), (4, 1), (3, 3), (4, 4), (5, 0)] {
        assert_eq!(
            SiteMetric::split(p, q).unwrap().signature(),
            (p as u128, q as u128)
        );
    }
    // Every non-trivial Pauli string splits the register in half.
    for n in 1..6 {
        let (a, b) = KreinMetric::new(&[(0, Pauli::Z)]).unwrap().signature(n);
        assert_eq!(a, b);
    }
}

#[test]
fn signatures_multiply_across_sites() {
    // (3,2) ⊗ (3,1) = (3·3 + 2·1, 3·1 + 2·3) = (11, 9).
    let space = Space::new(&[5, 4]).unwrap();
    let m = SiteMetric::new(space.clone())
        .signs(0, &split_signs(3, 2))
        .unwrap()
        .signs(1, &split_signs(3, 1))
        .unwrap();
    assert_eq!(m.signature(), (11, 9));
    // A definite factor leaves the other's signature scaled.
    let m = SiteMetric::new(Space::new(&[5, 3]).unwrap())
        .signs(0, &split_signs(3, 2))
        .unwrap();
    assert_eq!(m.signature(), (9, 6));
    // Diagonal: the restricted matrix carries the product signs.
    let eta = SiteMetric::new(space)
        .signs(0, &split_signs(3, 2))
        .unwrap()
        .signs(1, &split_signs(3, 1))
        .unwrap()
        .restricted(&[0, 1])
        .unwrap();
    let pos = (0..20).filter(|&k| eta.get(k, k).re > 0.0).count();
    assert_eq!(pos, 11);
}

#[test]
fn bad_factors_are_refused() {
    let space = Space::new(&[3]).unwrap();
    assert!(SiteMetric::new(space.clone()).signs(0, &[1, -1]).is_err());
    assert!(SiteMetric::new(space.clone())
        .signs(0, &[1, 2, -1])
        .is_err());
    let not_invol = SiteMatrix::from_vec(
        3,
        vec![
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 0.0),
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(1.0, 0.0),
        ],
    )
    .unwrap();
    assert!(SiteMetric::new(space).involution(0, not_invol).is_err());
}

#[test]
fn on_a_qubit_space_it_agrees_with_the_pauli_register() {
    // X on qubit 1 and Z on qubit 2, as site involutions and as a Pauli
    // string: same η-norm on the same state.
    let o = z(0.0, 0.0);
    let l = z(1.0, 0.0);
    let x = SiteMatrix::from_vec(2, vec![o, l, l, o]).unwrap();
    let space = Space::qubits(3);
    let site = SiteMetric::new(space.clone())
        .involution(1, x)
        .unwrap()
        .signs(2, &[1, -1])
        .unwrap();
    let pauli = KreinMetric::new(&[(1, Pauli::X), (2, Pauli::Z)]).unwrap();
    let a = SiteKreinRegister::new(state_on(&space, 1), site).unwrap();
    let b = KreinRegister::new(state_on(&space, 1), pauli).unwrap();
    let (la, lb) = (a.ledger().unwrap(), b.ledger().unwrap());
    common::assert_close(la.eta, lb.eta, TOL);
    common::assert_close(la.path, lb.path, TOL);
}

#[test]
fn a_non_diagonal_factor_is_evaluated_exactly() {
    // The involution swapping levels 0 and 1 of a qutrit: signature (2,1),
    // ⟨ψ|η|ψ⟩ = 2 Re(ψ̄₀ψ₁) + |ψ₂|².
    let o = z(0.0, 0.0);
    let l = z(1.0, 0.0);
    let swap01 = SiteMatrix::from_vec(3, vec![o, l, o, l, o, o, o, o, l]).unwrap();
    let space = Space::new(&[3]).unwrap();
    let m = SiteMetric::new(space.clone())
        .involution(0, swap01)
        .unwrap();
    assert_eq!(m.signature(), (2, 1));
    let s = state_on(&space, 2);
    let (a0, a1, a2) = (s.amplitude(0), s.amplitude(1), s.amplitude(2));
    let expect = 2.0 * (a0.conj() * a1).re + a2.norm_sqr();
    common::assert_close(m.eta_norm(s.as_ref()).unwrap(), expect, TOL);
}

// ── so(p, q) on one site ─────────────────────────────────────────────

#[test]
fn the_killing_form_names_each_pseudo_orthogonal_algebra() {
    let cases = [
        ((3, 1), (3, 3, 0)),
        ((2, 2), (4, 2, 0)),
        ((3, 2), (6, 4, 0)),
        ((4, 1), (4, 6, 0)),
        ((4, 2), (8, 7, 0)),
        ((3, 3), (9, 6, 0)),
        ((4, 4), (16, 12, 0)),
    ];
    for ((p, q), killing) in cases {
        let alg = so_algebra(&split_signs(p, q)).unwrap();
        assert_eq!(alg.len(), (p + q) * (p + q - 1) / 2);
        assert_eq!(
            sig(killing_signature_sites(&alg).unwrap()),
            killing,
            "so({p},{q})"
        );
    }
}

#[test]
fn the_qudit_vector_and_the_twistor_agree_on_ads_ds_and_conformal() {
    // so(3,2) on a 5-level site ≅ sp(4,ℝ) on twistors (rank-4, λ < 0);
    // so(4,1) ≅ the dS stabilizer; so(4,2) ≅ su(2,2).
    let vector =
        |p, q| sig(killing_signature_sites(&so_algebra(&split_signs(p, q)).unwrap()).unwrap());
    let twistor =
        |l: f64| sig(killing_signature(&stabilizer(&infinity_twistor(z(l, 0.0)))).unwrap());
    assert_eq!(vector(3, 2), twistor(-1.0));
    assert_eq!(vector(4, 1), twistor(1.0));
    assert_eq!(vector(4, 2), sig(killing_signature(&su22_basis()).unwrap()));
}

#[test]
fn ads4_is_a_state_and_so_3_2_moves_it_along_the_hyperboloid() {
    let metric = SiteMetric::split(3, 2).unwrap();
    let space = metric.space().clone();
    // A real point with ⟨x, x⟩_η = 0.5² + 0.3² + 0.2² − 1.0² − 0.4² = −0.78.
    let x = [0.5, 0.3, 0.2, 1.0, 0.4];
    let mut s = BackendRegistry::<C64>::standard()
        .create_with_space("dense", &space)
        .unwrap();
    let entries: Vec<(u64, C64)> = x
        .iter()
        .enumerate()
        .map(|(k, &v)| (k as u64, z(v, 0.0)))
        .collect();
    s.load(&entries).unwrap();
    let mut reg = SiteKreinRegister::new(s, metric).unwrap();
    let r2 = reg.ledger().unwrap().eta;
    common::assert_close(r2, -0.78, 1e-12);

    let alg = so_algebra(&split_signs(3, 2)).unwrap();
    let mut non_unitary = 0;
    for (k, gen) in alg.iter().enumerate() {
        let g = exponentiate_site(gen, 0.4 + 0.05 * k as f64).unwrap();
        if g.unitarity_deviation() > 1e-6 {
            non_unitary += 1;
        }
        reg.apply_sites(&g, &[0]).unwrap();
    }
    // The 6 mixed space–time generators are the boosts; the 4 within
    // the (3) and (2) blocks are rotations.
    assert_eq!(non_unitary, 6);
    let end = reg.ledger().unwrap();
    common::assert_close(end.eta, r2, 1e-9 * end.path);
    // Still a real point: the gates are real.
    for k in 0..5 {
        assert!(reg.state().amplitude(k).im.abs() < 1e-12);
    }
    assert!(
        (end.path - 1.54).abs() > 1e-2,
        "boosts move it along the hyperboloid"
    );
}

#[test]
fn the_qudit_ghost_pump_and_its_refusals() {
    let metric = SiteMetric::split(3, 2).unwrap();
    // Levels 0 (+) and 3 (−) can be boosted; 0 and 1 (both +) cannot.
    assert!(metric.level_boost(0, 0, 1, 0.5).is_err());
    let boost = metric.level_boost(0, 0, 3, 0.7).unwrap();
    assert!(boost.unitarity_deviation() > 0.1);

    // On a mixed space [5, 2], with a bystander qubit.
    let space = Space::new(&[5, 2]).unwrap();
    let metric = SiteMetric::new(space.clone())
        .signs(0, &split_signs(3, 2))
        .unwrap();
    let mut reg = SiteKreinRegister::new(state_on(&space, 3), metric).unwrap();
    let start = reg.ledger().unwrap();
    let mut last = start;
    for _ in 0..4 {
        reg.apply_sites(&boost, &[0]).unwrap();
        let led = reg.ledger().unwrap();
        common::assert_close(led.eta, start.eta, 1e-9 * led.path);
        last = led;
    }
    assert!(last.negative > start.negative && last.path > start.path);

    // The Fourier transform on the metric site mixes the sectors: refused,
    // state untouched. On the definite qubit any unitary passes.
    let before = reg.state().amplitude(4);
    let err = reg
        .apply_sites(&fourier::<C64>(5).unwrap(), &[0])
        .unwrap_err();
    assert!(format!("{err}").contains("not pseudo-unitary"));
    assert_eq!(reg.state().amplitude(4), before);
    reg.apply_sites(&fourier::<C64>(2).unwrap(), &[1]).unwrap();
}

#[test]
fn the_register_refuses_a_state_on_another_space() {
    let metric = SiteMetric::split(3, 2).unwrap();
    let wrong = state_on(&Space::new(&[4]).unwrap(), 4);
    assert!(SiteKreinRegister::new(wrong, metric).is_err());
}
