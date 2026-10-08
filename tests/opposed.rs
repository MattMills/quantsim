//! quantsim ↔ Opposed Mathematics: the bridge is exact where both sides
//! are, and the two libraries agree on states, gates, Born weights,
//! observables and no-signalling.
#![cfg(feature = "opposed-math")]

mod common;

use quantsim::exact::{DOmega, ExactState};
use quantsim::om::om_core::bucket::Bucket;
use quantsim::om::om_core::cyclotomic::Cyclo;
use quantsim::om::om_hamiltonian::hamiltonian::{
    heisenberg_chain, transverse_field_ising, Hamiltonian,
};
use quantsim::om::om_hamiltonian::hubbard::HubbardModel;
use quantsim::om::om_ipg::state::State;
use quantsim::om::om_quantum::born::{marginal_weights_exact, same_distribution, weights_exact};
use quantsim::om::om_quantum::gates::{
    character_gate, diagonal_gate, hadamard, shift_gate, sum_gate,
};
use quantsim::opposed::*;
use quantsim::prelude::*;

/// `√2 = ζ + ζ⁻¹` in `ℤ[ζ_8]`.
fn sqrt2() -> Cyclo<8> {
    Cyclo::zeta_pow(1).add(&Cyclo::zeta_pow(-1))
}

fn random_domega(rng: &mut Prng) -> DOmega {
    let mut c = [0i128; 4];
    for x in &mut c {
        *x = (rng.next_u64() % 41) as i128 - 20;
    }
    DOmega::from_parts(c, (rng.next_u64() % 6) as u32)
}

#[test]
fn numbers_cross_exactly() {
    let mut rng = Prng::new(1);
    for _ in 0..2000 {
        let (a, b) = (random_domega(&mut rng), random_domega(&mut rng));
        let (za, ka) = domega_to_cyclo(a);
        let (zb, kb) = domega_to_cyclo(b);
        assert_eq!(cyclo_to_domega(&za, ka), a);
        let (ba, k) = domega_to_bucket(a).unwrap();
        assert_eq!(bucket_to_domega(&ba, k), a);
        assert_eq!(
            ba.clone().reduced(),
            ba,
            "the bucket of a numerator is reduced"
        );
        assert_eq!(ba.is_nil(), a.is_zero());
        // Products and sums, over a common denominator.
        assert_eq!(cyclo_to_domega(&za.mul(&zb), ka + kb), a.mul(b).unwrap());
        let up = |z: &Cyclo<8>, by: u32| (0..by).fold(z.clone(), |acc, _| acc.mul(&sqrt2()));
        let k = ka.max(kb);
        let sum = up(&za, k - ka).add(&up(&zb, k - kb));
        assert_eq!(cyclo_to_domega(&sum, k), a.add(b).unwrap());
    }
}

/// One Clifford+T step, applied to both representations.
#[derive(Clone, Copy, Debug)]
enum Step {
    H(usize),
    X(usize),
    Z(usize),
    S(usize),
    T(usize),
    Cx(usize, usize),
}

fn random_steps(n: usize, len: usize, rng: &mut Prng) -> Vec<Step> {
    (0..len)
        .map(|_| {
            let q = (rng.next_u64() % n as u64) as usize;
            let r = (q + 1 + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
            match rng.next_u64() % 6 {
                0 => Step::H(q),
                1 => Step::X(q),
                2 => Step::Z(q),
                3 => Step::S(q),
                4 => Step::T(q),
                _ => Step::Cx(q, r),
            }
        })
        .collect()
}

fn circuit(n: usize, steps: &[Step]) -> Circuit {
    let mut c = Circuit::new(n);
    for &s in steps {
        match s {
            Step::H(q) => c.h(q),
            Step::X(q) => c.x(q),
            Step::Z(q) => c.z(q),
            Step::S(q) => c.s(q),
            Step::T(q) => c.t(q),
            Step::Cx(a, b) => c.cx(a, b),
        };
    }
    c
}

/// The steps run with OM's Vol. E gates from `|0…0⟩`; OM's Hadamard is the
/// unnormalised `[[1, 1], [1, −1]]`, so each one adds a `√2` to `k`.
fn om_run(n: usize, steps: &[Step]) -> (State<Mu8>, u32) {
    let mut s = State::empty();
    s.add_amp(qubit_point(0, n), Mu8::new(0), 1);
    let mut k = 0;
    for &step in steps {
        s = match step {
            Step::H(q) => {
                k += 1;
                hadamard(&s, q)
            }
            Step::X(q) => shift_gate(&s, q, 1),
            Step::Z(q) => character_gate(&s, q, 1),
            Step::S(q) => diagonal_gate(&s, q, &[Mu8::new(0), Mu8::new(2)]),
            Step::T(q) => diagonal_gate(&s, q, &[Mu8::new(0), Mu8::new(1)]),
            Step::Cx(a, b) => sum_gate(&s, a, b),
        };
    }
    (s, k)
}

fn assert_same_amplitudes(st: &ExactState, om: &State<Mu8>, k: u32) {
    let n = st.num_qubits();
    for i in 0..1u64 << n {
        let b = om.bucket(&qubit_point(i, n));
        assert_eq!(bucket_to_domega(&b, k), st.amplitude_exact(i), "index {i}");
    }
}

#[test]
fn om_gates_and_quantsim_gates_build_the_same_exact_state() {
    let mut rng = Prng::new(2);
    for trial in 0..60 {
        let n = 2 + trial % 4;
        let steps = random_steps(n, 30, &mut rng);
        let st = ExactState::run(&circuit(n, &steps)).unwrap();
        let (om, k) = om_run(n, &steps);
        assert_same_amplitudes(&st, &om, k);
        // exact_to_om agrees with the gate-built state.
        let (conv, kc) = exact_to_om(&st).unwrap();
        assert_same_amplitudes(&st, &conv, kc);
    }
}

#[test]
fn born_weights_agree_exactly_and_states_load_into_backends() {
    let mut rng = Prng::new(3);
    for trial in 0..30 {
        let n = 2 + trial % 4;
        let c = circuit(n, &random_steps(n, 25, &mut rng));
        let st = ExactState::run(&c).unwrap();
        let (om, k) = exact_to_om(&st).unwrap();
        // |ψ(i)/√2^k|² = w(i)/2^k against quantsim's (int + sqrt2·√2)/2^{k'}.
        let weights = weights_exact(&om);
        for i in 0..1u64 << n {
            let w = weights
                .get(&qubit_point(i, n))
                .cloned()
                .unwrap_or_else(Cyclo::zero);
            let p = st.probability_exact(i).unwrap();
            let p_num = Cyclo::from_int(p.int).add(&sqrt2().scale(p.sqrt2));
            assert_eq!(w.scale(1 << p.k), p_num.scale(1 << k), "index {i}");
        }
        // Loaded into float backends, the state matches the reference run.
        let reference = common::run_dense(&c);
        for backend in ["dense", "sparse", "adaptive"] {
            let mut b = Simulator::<C64>::new()
                .run_on(backend, &Circuit::new(n))
                .unwrap();
            load_om_state(b.as_mut(), &om, k).unwrap();
            assert!(max_amplitude_deviation(b.as_ref(), reference.as_ref()) < 1e-12);
        }
    }
}

/// `⟨ψ|H|ψ⟩` from OM's exact dense matrix.
fn dense_oracle(h: &Hamiltonian, n: u32, state: &dyn Backend<C64>) -> C64 {
    let m = h.to_dense(n);
    let amps: Vec<C64> = (0..1u64 << n).map(|i| state.amplitude(i)).collect();
    let mut total = C64::new(0.0, 0.0);
    for (r, ar) in amps.iter().enumerate() {
        for (col, ac) in amps.iter().enumerate() {
            let (re, im) = m.get(r, col);
            if re != 0 || im != 0 {
                total += ar.conj() * C64::new(re as f64, im as f64) * ac;
            }
        }
    }
    total
}

fn models() -> Vec<(&'static str, Hamiltonian, u32)> {
    vec![
        ("tfim", transverse_field_ising(6, 2, 3, true), 6),
        ("heisenberg", heisenberg_chain(6, true), 6),
        (
            "hubbard 2x2",
            HubbardModel::new(2, 2, 1, 4, false).hamiltonian(),
            8,
        ),
    ]
}

#[test]
fn hamiltonian_expectations_agree_on_every_backend() {
    for (label, h, n) in models() {
        let c = common::scrambler(n as usize);
        let reference = dense_oracle(&h, n, common::run_dense(&c).as_ref());
        for backend in ["dense", "sparse", "adaptive", "factored", "mps"] {
            let state = common::run_named(backend, &c);
            let e = expectation(&h, state.as_ref()).unwrap();
            assert!(
                (e - reference).norm() < 1e-9,
                "{label} on {backend}: {e} vs {reference}"
            );
        }
        assert_eq!(pauli_sum(&h).len(), h.pauli_terms());
    }
}

#[test]
fn exact_expectations_match_float_ones() {
    let mut rng = Prng::new(4);
    for (label, h, n) in models() {
        for _ in 0..3 {
            let c = circuit(n as usize, &random_steps(n as usize, 40, &mut rng));
            let exact = expectation_exact(&h, &ExactState::run(&c).unwrap()).unwrap();
            assert_eq!(exact.conj(), exact, "{label}: a Hermitian energy is real");
            let float = expectation(&h, common::run_dense(&c).as_ref()).unwrap();
            assert!(
                (exact.to_c64() - float).norm() < 1e-9,
                "{label}: {exact:?} vs {float}"
            );
        }
    }
}

/// Whether two circuits give qubits 2 and 3 the same distribution, decided
/// exactly from OM's marginal weights.
fn same_b_distribution(a: &Circuit, b: &Circuit) -> bool {
    let marginal = |c: &Circuit| {
        let (om, _) = exact_to_om(&ExactState::run(c).unwrap()).unwrap();
        marginal_weights_exact(&om, &[2, 3])
    };
    same_distribution(&marginal(a), &marginal(b))
}

#[test]
fn local_gates_do_not_signal_exactly() {
    let mut rng = Prng::new(5);
    let mut b_side_moved = 0;
    for _ in 0..20 {
        let prep = random_steps(4, 30, &mut rng);
        let before = circuit(4, &prep);
        // Gates on qubits 0 and 1 only leave the distribution of 2, 3.
        let local = random_steps(2, 15, &mut rng);
        assert!(same_b_distribution(
            &before,
            &circuit(4, &[prep.clone(), local].concat())
        ));
        // A Hadamard on qubit 2 is not local to A, and usually shows.
        if !same_b_distribution(&before, &circuit(4, &[prep, vec![Step::H(2)]].concat())) {
            b_side_moved += 1;
        }
    }
    assert!(b_side_moved > 0);
}

#[test]
fn opposition_mass_of_a_bucket_is_its_numerator_l1() {
    let mut rng = Prng::new(6);
    for _ in 0..500 {
        let d = random_domega(&mut rng);
        let (b, _): (Bucket<Mu8>, u32) = domega_to_bucket(d).unwrap();
        let l1: i128 = d.parts().0.iter().map(|c| c.abs()).sum();
        assert_eq!(i128::from(b.opposition_mass()), l1);
    }
}
