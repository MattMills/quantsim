//! The split octonions 𝕆ₛ, measured: a composition algebra that is
//! alternative but not associative, with an indefinite norm of signature
//! `(4, 4)`; the 2×2 Hermitian register over it as `ℝ^{5,5}`, whose pure
//! spinors are null; the full standard gate set, under which net and path
//! weight are separately conserved; and the norm-1 boost that pumps path
//! weight at conserved net weight.

#![allow(clippy::assertions_on_constants)] // algebra-flag pins are intentional

mod common;

use quantsim::hermitian2::signature;
use quantsim::prelude::*;

type OS = SplitOctonion;

const TOL: f64 = 1e-10;

fn rand_scalar<K: Scalar>(rng: &mut Prng) -> K {
    let c: Vec<f64> = (0..K::DIM).map(|_| 2.0 * rng.next_f64() - 1.0).collect();
    K::from_coeffs(&c)
}

/// The polarization of the Born form, `B(u, v)`.
fn polar<K: Scalar>(u: K, v: K) -> f64 {
    0.5 * ((u + v).born_weight() - u.born_weight() - v.born_weight())
}

// ── the algebra ──────────────────────────────────────────────────────

#[test]
fn flags_and_dimension() {
    assert_eq!(OS::DIM, 8);
    assert!(!OS::COMMUTATIVE);
    assert!(!OS::ASSOCIATIVE);
    assert!(!OS::DIVISION);
    assert!(!OS::TRIVIAL_CONJ);
}

#[test]
fn the_norm_is_the_split_form_of_signature_4_4() {
    // The basis is orthogonal for the polar form; count the signs.
    let (mut pos, mut neg) = (0, 0);
    for p in 0..8 {
        for q in (p + 1)..8 {
            assert!(polar(OS::basis(p), OS::basis(q)).abs() < TOL);
        }
        let n = OS::basis(p).born_weight();
        if n > 0.0 {
            pos += 1;
        } else if n < 0.0 {
            neg += 1;
        }
    }
    assert_eq!((pos, neg), (4, 4));
    // And it is the algebra's own norm: N(z) = Re(z·z̄).
    let mut rng = Prng::new(1);
    for _ in 0..50 {
        let z: OS = rand_scalar(&mut rng);
        common::assert_close(z.born_weight(), (z * z.conj()).re(), TOL);
    }
}

#[test]
fn it_is_a_composition_algebra_and_alternative_but_not_associative() {
    let mut rng = Prng::new(2);
    let mut assoc_defect = 0.0f64;
    for _ in 0..200 {
        let (x, y, z): (OS, OS, OS) = (
            rand_scalar(&mut rng),
            rand_scalar(&mut rng),
            rand_scalar(&mut rng),
        );
        // Multiplicative norm: the reason pure spinors stay null.
        common::assert_close(
            (x * y).born_weight(),
            x.born_weight() * y.born_weight(),
            1e-9,
        );
        // Alternative laws.
        assert!(((x * x) * y).approx_eq(x * (x * y), 1e-9));
        assert!(((y * x) * x).approx_eq(y * (x * x), 1e-9));
        // Conjugation reverses products.
        assert!((x * y).conj().approx_eq(y.conj() * x.conj(), 1e-9));
        assoc_defect = assoc_defect.max(((x * y) * z - x * (y * z)).abs_sqr().sqrt());
    }
    assert!(assoc_defect > 1e-3, "split octonions must not associate");
}

#[test]
fn null_vectors_are_zero_divisors() {
    // (1 + ℓ) is null; it annihilates (1 − ℓ) from either side.
    let u = OS::one() + OS::ell();
    let v = OS::one() - OS::ell();
    assert_eq!(u.born_weight(), 0.0);
    assert!(u.abs_sqr() > 0.0 && v.abs_sqr() > 0.0);
    assert!((u * v).is_zero(TOL) && (v * u).is_zero(TOL));
}

// ── the Hermitian register: ℝ^{5,5} ──────────────────────────────────

#[test]
fn the_hermitian_register_is_r_5_5_with_null_pure_spinors() {
    let s = signature::<OS>();
    assert_eq!((s.positive, s.negative, s.null), (5, 5, 0));

    let mut rng = Prng::new(3);
    for _ in 0..100 {
        let x = Hermitian2::from_spinor(rand_scalar::<OS>(&mut rng), rand_scalar(&mut rng));
        assert!(x.det().abs() < 1e-9, "pure spinor not null: {}", x.det());
        // The always-valid generators keep it on the cone.
        let w: OS = rand_scalar(&mut rng);
        let y = x.shear_upper(w).boost(1.3).shear_lower(w.conj());
        assert!(y.det().abs() < 1e-8);
        // And they preserve det off the cone too.
        let h = Hermitian2::new(rng.next_f64(), rng.next_f64(), rand_scalar::<OS>(&mut rng));
        common::assert_close(h.shear_upper(w).det(), h.det(), 1e-9);
    }
}

#[test]
fn the_split_row_of_the_hermitian_ladder() {
    // One split doubling after another adds (1, 1) to the signature:
    // (2,2) → (3,3) → (5,5) as the base goes ℝ → ℂ → ℍ.
    let sig = |s: quantsim::hermitian2::FormSignature| (s.positive, s.negative);
    assert_eq!(sig(signature::<SplitCD<f64>>()), (2, 2));
    assert_eq!(sig(signature::<SplitCD<C64>>()), (3, 3));
    assert_eq!(sig(signature::<OS>()), (5, 5));
    assert_eq!(sig(signature::<SplitComplex>()), (2, 2));
    assert_eq!(sig(signature::<SplitQuaternion>()), (3, 3));
}

// ── circuits over 𝕆ₛ ─────────────────────────────────────────────────

fn workload<S: Scalar>() -> Circuit<S> {
    let mut c: Circuit<S> = Circuit::new(3);
    c.h(0).cx(0, 1).t(1).ry(2, 0.7).cz(1, 2).h(2).s(0).cx(2, 0);
    c
}

#[test]
fn it_carries_the_full_standard_gate_set() {
    assert_eq!(
        GateRegistry::<OS>::standard().names(),
        GateRegistry::<C64>::standard().names()
    );
}

#[test]
fn from_a_real_start_a_standard_circuit_reproduces_the_complex_run() {
    let reference = Simulator::<C64>::new().run(&workload()).unwrap();
    let state = Simulator::<OS>::new().run(&workload()).unwrap();
    for idx in 0..8u64 {
        let (r, s) = (reference.amplitude(idx), state.amplitude(idx).coeffs());
        assert!((s[0] - r.re).abs() < TOL && (s[1] - r.im).abs() < TOL);
        assert!(s[2..].iter().all(|c| c.abs() < TOL), "leaked off ℂ");
    }
}

#[test]
fn standard_gates_conserve_net_and_path_weight_separately() {
    // Load a state with weight in both halves.
    let mut rng = Prng::new(4);
    let mut entries = Vec::new();
    for idx in 0..8u64 {
        entries.push((idx, rand_scalar::<OS>(&mut rng)));
    }
    let mut state = DenseState::<OS>::new(3).unwrap();
    state.load(&entries).unwrap();
    let (net0, path0) = (state.total_weight(), state.total_abs_sqr());
    let reg = GateRegistry::<OS>::standard();
    workload::<OS>()
        .bind(&reg)
        .unwrap()
        .run(&mut state)
        .unwrap();
    common::assert_close(state.total_weight(), net0, 1e-9);
    common::assert_close(state.total_abs_sqr(), path0, 1e-9);
}

#[test]
fn the_boost_pumps_path_weight_at_conserved_net_weight() {
    let mut rng = Prng::new(5);
    let z: OS = rand_scalar(&mut rng);
    let mut last_path = z.abs_sqr();
    for k in 1..=6 {
        let theta = 0.5 * k as f64;
        let b = OS::boost(theta);
        common::assert_close(b.born_weight(), 1.0, 1e-9);
        let w = b * z;
        common::assert_close(
            w.born_weight(),
            z.born_weight(),
            1e-6 * theta.cosh().powi(2),
        );
        if k >= 3 {
            assert!(w.abs_sqr() > last_path, "path weight should grow");
        }
        last_path = w.abs_sqr();
    }
}
