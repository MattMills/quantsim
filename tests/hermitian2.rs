//! The 2×2 Hermitian register over every amplitude algebra, measured: the
//! determinant's signature per algebra, the ℂ case certified against the
//! dense backend (density matrix, Bloch vector, Born overlaps), the null
//! cone of pure spinors and exactly where it breaks (the sedenions), the
//! det-preserving shears and boosts in every algebra, and the split
//! algebras' indefinite trace form.

mod common;

use quantsim::backend::pauli_expectation;
use quantsim::hermitian2::signature;
use quantsim::prelude::*;

const TOL: f64 = 1e-10;

fn rand_scalar<K: Scalar>(rng: &mut Prng) -> K {
    let c: Vec<f64> = (0..K::DIM).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
    K::from_coeffs(&c)
}

fn rand_h2<K: Scalar>(rng: &mut Prng) -> Hermitian2<K> {
    let a = 2.0 * rng.next_f64() - 1.0;
    let b = 2.0 * rng.next_f64() - 1.0;
    Hermitian2::new(a, b, rand_scalar(rng))
}

fn sig<K: Scalar>() -> (usize, usize, usize) {
    let s = signature::<K>();
    assert_eq!(s.dim(), K::DIM + 2);
    (s.positive, s.negative, s.null)
}

// ── the spacetime each algebra makes ─────────────────────────────────

#[test]
fn the_signature_of_det_is_set_by_the_algebra() {
    assert_eq!(sig::<f64>(), (1, 2, 0));
    assert_eq!(sig::<C64>(), (1, 3, 0));
    assert_eq!(sig::<CComplex>(), (1, 3, 0));
    assert_eq!(sig::<SplitComplex>(), (2, 2, 0));
    assert_eq!(sig::<Quaternion>(), (1, 5, 0));
    assert_eq!(sig::<SplitQuaternion>(), (3, 3, 0));
    assert_eq!(sig::<Octonion>(), (1, 9, 0));
    assert_eq!(sig::<Sedenion>(), (1, 17, 0));
}

fn norm_forms_agree<K: Scalar>(seed: u64) {
    // det uses born_weight; the shears and the trace form use the
    // algebra's products. They must be the same quadratic form.
    let mut rng = Prng::new(seed);
    for _ in 0..50 {
        let z: K = rand_scalar(&mut rng);
        common::assert_close(z.born_weight(), (z * z.conj()).re(), TOL);
        let (x, y) = (rand_h2::<K>(&mut rng), rand_h2::<K>(&mut rng));
        common::assert_close(
            x.trace_pairing(&y),
            x.trace() * y.trace() - 2.0 * x.det_pairing(&y),
            1e-9,
        );
        common::assert_close(x.det_pairing(&x), x.det(), 1e-12);
    }
}

#[test]
fn born_weight_is_the_algebras_own_norm_in_every_algebra() {
    norm_forms_agree::<f64>(1);
    norm_forms_agree::<C64>(2);
    norm_forms_agree::<SplitComplex>(3);
    norm_forms_agree::<Quaternion>(4);
    norm_forms_agree::<SplitQuaternion>(5);
    norm_forms_agree::<Octonion>(6);
    norm_forms_agree::<Sedenion>(7);
}

#[test]
fn coordinates_round_trip() {
    let mut rng = Prng::new(11);
    let x: Hermitian2<Octonion> = rand_h2(&mut rng);
    let back = Hermitian2::<Octonion>::from_coords(&x.coords());
    assert!((back.a() - x.a()).abs() < 1e-12 && (back.b() - x.b()).abs() < 1e-12);
    assert!(back.z().approx_eq(x.z(), 1e-12));
}

// ── over ℂ it is the qubit, certified against the dense backend ──────

fn one_qubit(theta: f64, phi: f64, lam: f64) -> Box<dyn Backend<C64>> {
    let mut c = Circuit::new(1);
    c.u(0, theta, phi, lam);
    common::run_dense(&c)
}

#[test]
fn over_c_a_pure_state_is_a_null_vector_with_the_bloch_vector_as_space() {
    for (k, &(th, ph, la)) in [
        (0.0, 0.0, 0.0),
        (0.7, 0.3, -1.1),
        (2.1, -0.4, 0.9),
        (std::f64::consts::PI, 1.3, 0.2),
    ]
    .iter()
    .enumerate()
    {
        let state = one_qubit(th, ph, la);
        let rho = Hermitian2::from_spinor(state.amplitude(0), state.amplitude(1));
        common::assert_close(rho.trace(), 1.0, TOL);
        assert_eq!(rho.causal_class(TOL), CausalClass::Null, "case {k}");

        let ex = pauli_expectation(state.as_ref(), &[(0, Pauli::X)])
            .unwrap()
            .re;
        let ey = pauli_expectation(state.as_ref(), &[(0, Pauli::Y)])
            .unwrap()
            .re;
        let ez = pauli_expectation(state.as_ref(), &[(0, Pauli::Z)])
            .unwrap()
            .re;
        let r = rho.bloch_vector();
        common::assert_close(r[0], ez, TOL);
        common::assert_close(r[1], ex, TOL);
        common::assert_close(r[2], -ey, TOL);
        common::assert_close(r.iter().map(|c| c * c).sum::<f64>(), 1.0, 1e-9);
    }
}

#[test]
fn over_c_the_trace_pairing_is_the_born_overlap() {
    let pairs = [
        ((0.7, 0.3, -1.1), (2.1, -0.4, 0.9)),
        ((0.0, 0.0, 0.0), (1.2, 0.5, 0.5)),
        ((3.0, 2.0, 1.0), (3.0, 2.0, 1.0)),
    ];
    for ((t1, p1, l1), (t2, p2, l2)) in pairs {
        let (s1, s2) = (one_qubit(t1, p1, l1), one_qubit(t2, p2, l2));
        let (a0, a1) = (s1.amplitude(0), s1.amplitude(1));
        let (b0, b1) = (s2.amplitude(0), s2.amplitude(1));
        let overlap = (a0.conj() * b0 + a1.conj() * b1).norm_sqr();
        let rho = Hermitian2::from_spinor(a0, a1);
        let sigma = Hermitian2::from_spinor(b0, b1);
        common::assert_close(rho.trace_pairing(&sigma), overlap, 1e-12);
    }
}

#[test]
fn the_maximally_mixed_state_is_the_time_axis() {
    let mixed = Hermitian2::from_spacetime(0.5, 0.0, C64::new(0.0, 0.0));
    assert_eq!(mixed.causal_class(TOL), CausalClass::Timelike);
    assert!(mixed.bloch_vector().iter().all(|c| c.abs() < TOL));
    let space = Hermitian2::from_spacetime(0.0, 1.0, C64::new(0.0, 0.0));
    assert_eq!(space.causal_class(TOL), CausalClass::Spacelike);
}

// ── the null cone of pure spinors, and where it breaks ───────────────

fn max_spinor_det<K: Scalar>(seed: u64) -> f64 {
    let mut rng = Prng::new(seed);
    (0..200)
        .map(|_| {
            let x = Hermitian2::from_spinor(rand_scalar::<K>(&mut rng), rand_scalar(&mut rng));
            x.det().abs()
        })
        .fold(0.0, f64::max)
}

#[test]
fn pure_spinors_are_null_exactly_in_the_composition_algebras() {
    assert!(max_spinor_det::<f64>(21) < 1e-12);
    assert!(max_spinor_det::<C64>(22) < 1e-12);
    assert!(max_spinor_det::<SplitComplex>(23) < 1e-12);
    assert!(max_spinor_det::<Quaternion>(24) < 1e-12);
    assert!(max_spinor_det::<SplitQuaternion>(25) < 1e-12);
    // Octonions: still null — the norm is multiplicative through 𝕆.
    assert!(max_spinor_det::<Octonion>(26) < 1e-12);
    // Sedenions: the norm stops being multiplicative and the cone breaks.
    let sed = max_spinor_det::<Sedenion>(27);
    assert!(
        sed > 1e-3,
        "sedenion spinors should leave the cone, got {sed}"
    );
}

// ── the Lorentz action, in every algebra ─────────────────────────────

fn shears_and_boosts_preserve_det<K: Scalar>(seed: u64) {
    let mut rng = Prng::new(seed);
    for _ in 0..50 {
        let x: Hermitian2<K> = rand_h2(&mut rng);
        let w: K = rand_scalar(&mut rng);
        let lam = 0.5 + rng.next_f64();
        let d = x.det();
        common::assert_close(x.shear_upper(w).det(), d, 1e-9);
        common::assert_close(x.shear_lower(w).det(), d, 1e-9);
        common::assert_close(x.boost(lam).det(), d, 1e-9);
        let chain = x.shear_upper(w).boost(lam).shear_lower(w.conj());
        common::assert_close(chain.det(), d, 1e-9);
    }
}

#[test]
fn shears_and_boosts_preserve_det_in_every_algebra() {
    shears_and_boosts_preserve_det::<f64>(31);
    shears_and_boosts_preserve_det::<C64>(32);
    shears_and_boosts_preserve_det::<SplitComplex>(33);
    shears_and_boosts_preserve_det::<Quaternion>(34);
    shears_and_boosts_preserve_det::<SplitQuaternion>(35);
    shears_and_boosts_preserve_det::<Octonion>(36);
    shears_and_boosts_preserve_det::<Sedenion>(37);
}

#[test]
fn octonionic_pure_states_stay_on_the_cone_under_the_lorentz_action() {
    let mut rng = Prng::new(41);
    for _ in 0..20 {
        let x = Hermitian2::from_spinor(rand_scalar::<Octonion>(&mut rng), rand_scalar(&mut rng));
        let y = x
            .shear_upper(rand_scalar(&mut rng))
            .boost(1.7)
            .shear_lower(rand_scalar(&mut rng));
        assert_eq!(y.causal_class(1e-9), CausalClass::Null);
    }
}

fn shear_matches_conjugate<K: Scalar>(seed: u64) {
    let mut rng = Prng::new(seed);
    for _ in 0..20 {
        let x: Hermitian2<K> = rand_h2(&mut rng);
        let w: K = rand_scalar(&mut rng);
        let (o, z) = (K::one(), K::zero());
        let up = x.conjugate([[o, w], [z, o]], 1e-9).unwrap();
        let lo = x.conjugate([[o, z], [w, o]], 1e-9).unwrap();
        for (got, want) in [(up, x.shear_upper(w)), (lo, x.shear_lower(w))] {
            common::assert_close(got.a(), want.a(), 1e-9);
            common::assert_close(got.b(), want.b(), 1e-9);
            assert!(got.z().approx_eq(want.z(), 1e-9));
        }
    }
}

#[test]
fn the_entrywise_shears_are_the_matrix_action_where_that_is_defined() {
    shear_matches_conjugate::<C64>(51);
    shear_matches_conjugate::<SplitComplex>(52);
    shear_matches_conjugate::<Quaternion>(53);
    shear_matches_conjugate::<SplitQuaternion>(54);
}

#[test]
fn over_c_conjugation_scales_det_by_the_squared_modulus_of_det_m() {
    let mut rng = Prng::new(61);
    for _ in 0..30 {
        let x: Hermitian2<C64> = rand_h2(&mut rng);
        let m: [[C64; 2]; 2] = [
            [rand_scalar(&mut rng), rand_scalar(&mut rng)],
            [rand_scalar(&mut rng), rand_scalar(&mut rng)],
        ];
        let det_m = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        let y = x.conjugate(m, 1e-9).unwrap();
        common::assert_close(y.det(), det_m.norm_sqr() * x.det(), 1e-9);
    }
}

#[test]
fn the_general_action_refuses_non_associative_algebras() {
    let x = Hermitian2::new(1.0, 0.0, Octonion::zero());
    let id = [
        [Octonion::one(), Octonion::zero()],
        [Octonion::zero(), Octonion::one()],
    ];
    assert!(matches!(
        x.conjugate(id, 1e-9),
        Err(Error::UnsupportedForAlgebra { .. })
    ));
}

// ── probabilities need a definite algebra ────────────────────────────

fn min_self_trace_pairing<K: Scalar>(seed: u64) -> f64 {
    let mut rng = Prng::new(seed);
    (0..200)
        .map(|_| {
            let x: Hermitian2<K> = rand_h2(&mut rng);
            x.trace_pairing(&x)
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn the_trace_form_is_positive_only_over_definite_algebras() {
    assert!(min_self_trace_pairing::<C64>(71) >= 0.0);
    assert!(min_self_trace_pairing::<Quaternion>(72) >= 0.0);
    assert!(min_self_trace_pairing::<Octonion>(73) >= 0.0);
    assert!(min_self_trace_pairing::<SplitComplex>(74) < 0.0);
    assert!(min_self_trace_pairing::<SplitQuaternion>(75) < 0.0);
}
