//! Algebraic gate identities and known decompositions, verified as circuit
//! equivalences on a scrambled generic state. These reproduce standard
//! results from the literature (Nielsen & Chuang §4) and double as deep
//! cross-checks of the apply machinery: each side exercises different code
//! paths (1q fast path vs general path vs raw matrices).

mod common;

use common::*;
use quantsim::prelude::*;

#[test]
fn pauli_conjugations_by_h() {
    // HXH = Z, HZH = X, HYH = −Y (up to global phase −1 on the Y case: use
    // exact equality — HYH = −Y holds exactly, so compare against y + phase).
    assert_equiv(
        1,
        |c| {
            c.h(0).x(0).h(0);
        },
        |c| {
            c.z(0);
        },
    );
    assert_equiv(
        1,
        |c| {
            c.h(0).z(0).h(0);
        },
        |c| {
            c.x(0);
        },
    );
    assert_equiv_up_to_phase(
        1,
        |c| {
            c.h(0).y(0).h(0);
        },
        |c| {
            c.y(0);
        },
    );
}

#[test]
fn s_conjugation_turns_x_into_y() {
    // S X S† = Y.
    assert_equiv(
        1,
        |c| {
            c.sdg(0).x(0).s(0);
        },
        |c| {
            c.y(0);
        },
    );
}

#[test]
fn phase_gate_powers() {
    // T² = S, S² = Z, √X² = X, (√X)† = sxdg.
    assert_equiv(
        1,
        |c| {
            c.t(0).t(0);
        },
        |c| {
            c.s(0);
        },
    );
    assert_equiv(
        1,
        |c| {
            c.s(0).s(0);
        },
        |c| {
            c.z(0);
        },
    );
    assert_equiv(
        1,
        |c| {
            c.sx(0).sx(0);
        },
        |c| {
            c.x(0);
        },
    );
    assert_equiv(
        1,
        |c| {
            c.sx(0).sxdg(0);
        },
        |_| {},
    );
    assert_equiv(
        1,
        |c| {
            c.t(0).tdg(0);
        },
        |_| {},
    );
    // S = diag(1, i) is *not* self-inverse: S⁴ = I.
    assert_equiv(
        1,
        |c| {
            c.s(0).s(0).s(0).s(0);
        },
        |_| {},
    );
}

#[test]
fn swap_is_three_cnots() {
    assert_equiv(
        2,
        |c| {
            c.swap(0, 1);
        },
        |c| {
            c.cx(0, 1).cx(1, 0).cx(0, 1);
        },
    );
}

#[test]
fn cz_is_h_conjugated_cx() {
    assert_equiv(
        2,
        |c| {
            c.cz(0, 1);
        },
        |c| {
            c.h(1).cx(0, 1).h(1);
        },
    );
    // CZ is symmetric in its qubits.
    assert_equiv(
        2,
        |c| {
            c.cz(0, 1);
        },
        |c| {
            c.cz(1, 0);
        },
    );
}

#[test]
fn cx_direction_reversal_via_hadamards() {
    assert_equiv(
        2,
        |c| {
            c.cx(1, 0);
        },
        |c| {
            c.h(0).h(1).cx(0, 1).h(0).h(1);
        },
    );
}

#[test]
fn iswap_decomposition() {
    // iSWAP = SWAP · (S ⊗ S) · CZ  (circuit order: cz, s, s, swap).
    assert_equiv(
        2,
        |c| {
            c.iswap(0, 1);
        },
        |c| {
            c.cz(0, 1).s(0).s(1).swap(0, 1);
        },
    );
}

#[test]
fn toffoli_standard_decomposition() {
    // Nielsen & Chuang Fig. 4.9: Toffoli from {H, T, T†, CX, S-free}.
    assert_equiv(
        3,
        |c| {
            c.ccx(0, 1, 2);
        },
        |c| {
            let (a, b, t) = (0, 1, 2);
            c.h(t)
                .cx(b, t)
                .tdg(t)
                .cx(a, t)
                .t(t)
                .cx(b, t)
                .tdg(t)
                .cx(a, t)
                .t(b)
                .t(t)
                .h(t)
                .cx(a, b)
                .t(a)
                .tdg(b)
                .cx(a, b);
        },
    );
}

#[test]
fn ccz_from_toffoli() {
    // CCZ = H(t) · CCX · H(t), and is symmetric under any qubit ordering.
    assert_equiv(
        3,
        |c| {
            c.ccz(0, 1, 2);
        },
        |c| {
            c.h(2).ccx(0, 1, 2).h(2);
        },
    );
    assert_equiv(
        3,
        |c| {
            c.ccz(0, 1, 2);
        },
        |c| {
            c.ccz(2, 0, 1);
        },
    );
}

#[test]
fn rotation_composition_and_conjugation() {
    let (a, b) = (0.87, -1.93);
    // Additivity (exact for the exp(−iθP/2) convention).
    assert_equiv(
        1,
        |c| {
            c.rz(0, a).rz(0, b);
        },
        |c| {
            c.rz(0, a + b);
        },
    );
    assert_equiv(
        1,
        |c| {
            c.rx(0, a).rx(0, b);
        },
        |c| {
            c.rx(0, a + b);
        },
    );
    assert_equiv(
        1,
        |c| {
            c.ry(0, a).ry(0, b);
        },
        |c| {
            c.ry(0, a + b);
        },
    );
    // Basis change: RX(θ) = H RZ(θ) H.
    assert_equiv(
        1,
        |c| {
            c.rx(0, a);
        },
        |c| {
            c.h(0).rz(0, a).h(0);
        },
    );
    // RZ vs phase gate: p(θ) = e^{iθ/2} rz(θ).
    assert_equiv_up_to_phase(
        1,
        |c| {
            c.p(0, a);
        },
        |c| {
            c.rz(0, a);
        },
    );
}

#[test]
fn rzz_via_cnot_conjugation() {
    let theta = 1.234;
    // exp(−iθ ZZ/2) = CX(a,b) · (I ⊗ RZ_b(θ)) · CX(a,b).
    assert_equiv(
        2,
        |c| {
            c.rzz(0, 1, theta);
        },
        |c| {
            c.cx(0, 1).rz(1, theta).cx(0, 1);
        },
    );
    // rxx via Hadamard conjugation of rzz.
    assert_equiv(
        2,
        |c| {
            c.rxx(0, 1, theta);
        },
        |c| {
            c.h(0).h(1).rzz(0, 1, theta).h(0).h(1);
        },
    );
}

#[test]
fn u_gate_zyz_decomposition() {
    let (theta, phi, lam) = (0.973, 2.611, -0.442);
    // u(θ, φ, λ) = e^{i(φ+λ)/2} RZ(φ) RY(θ) RZ(λ)  (global phase only).
    assert_equiv_up_to_phase(
        1,
        |c| {
            c.u(0, theta, phi, lam);
        },
        |c| {
            c.rz(0, lam).ry(0, theta).rz(0, phi);
        },
    );
}

#[test]
fn controlled_rotations_match_controlled_matrix() {
    // crz must equal the controlled() construction of rz, applied as a raw
    // matrix — ties the registry's controlled gates to GateMatrix::controlled.
    let theta = 0.777;
    let reg: GateRegistry = GateRegistry::standard();
    let rz = reg.resolve("rz").unwrap().matrix(&[theta]).unwrap();
    let raw = rz.controlled();
    assert_equiv(
        2,
        |c| {
            c.crz(0, 1, theta);
        },
        move |c| {
            c.raw("c(rz)", raw, vec![0, 1]);
        },
    );
}

#[test]
fn controlled_gate_with_control_zero_is_identity() {
    // Prepare q0 = |0⟩ definitively, scramble only q1: controlled ops do
    // nothing.
    let mut with = Circuit::new(2);
    with.u(1, 0.9, 0.4, 1.7)
        .cx(0, 1)
        .cz(0, 1)
        .crz(0, 1, 2.2)
        .ch(0, 1);
    let mut without = Circuit::new(2);
    without.u(1, 0.9, 0.4, 1.7);
    let (sa, sb) = (run_dense(&with), run_dense(&without));
    for i in 0..4u64 {
        assert!(sa.amplitude(i).approx_eq(sb.amplitude(i), TOL));
    }
}

#[test]
fn diagonal_kernel_matches_dense_gates() {
    // cz and cp as diagonal kernels must equal the registry gates —
    // pins the little-endian sub-indexing of diagonal entries.
    let theta = 1.234f64;
    let cp_diag: Vec<C64> = vec![
        c64(1.0, 0.0),
        c64(1.0, 0.0),
        c64(1.0, 0.0),
        c64(theta.cos(), theta.sin()),
    ];
    assert_equiv(
        2,
        |c| {
            c.cz(0, 1).cp(0, 1, theta);
        },
        move |c| {
            c.diagonal(
                "cz",
                vec![c64(1.0, 0.0); 3]
                    .into_iter()
                    .chain([c64(-1.0, 0.0)])
                    .collect(),
                vec![0, 1],
            )
            .diagonal("cp", cp_diag, vec![0, 1]);
        },
    );
    // rz as a 1q diagonal, on a non-adjacent target among 3 qubits.
    let rz: Vec<C64> = vec![
        c64((theta / 2.0).cos(), -(theta / 2.0).sin()),
        c64((theta / 2.0).cos(), (theta / 2.0).sin()),
    ];
    assert_equiv(
        3,
        |c| {
            c.rz(1, theta);
        },
        move |c| {
            c.diagonal("rz", rz, vec![1]);
        },
    );
}

#[test]
fn diagonal_inverse_and_backend_agreement() {
    let reg: GateRegistry = GateRegistry::standard();
    // A random-phase diagonal on qubits [2, 0] of four (order matters).
    let mut prng = Prng::new(31);
    let entries: Vec<C64> = (0..4)
        .map(|_| {
            let t = prng.next_f64() * std::f64::consts::TAU;
            c64(t.cos(), t.sin())
        })
        .collect();
    let mut c: Circuit = scrambler(4);
    c.diagonal("dphase", entries, vec![2, 0]);
    let bound = c.bind(&reg).unwrap();

    // All three representations agree.
    let sim = Simulator::<C64>::new();
    let dense = sim.run(&c).unwrap();
    let sparse = sim.run_on("sparse", &c).unwrap();
    let adaptive = sim.run_on("adaptive", &c).unwrap();
    for i in 0..16u64 {
        assert!(
            dense.amplitude(i).approx_eq(sparse.amplitude(i), TOL),
            "sparse {i}"
        );
        assert!(
            dense.amplitude(i).approx_eq(adaptive.amplitude(i), TOL),
            "adaptive {i}"
        );
    }

    // Bound inverse uncomputes the diagonal too.
    let mut state = DenseState::<C64>::new(4).unwrap();
    bound.run(&mut state).unwrap();
    bound.inverse().run(&mut state).unwrap();
    assert!(state.amplitude(0).approx_eq(c64(1.0, 0.0), 1e-8));
}

#[test]
fn diagonal_validation_at_bind() {
    let reg: GateRegistry = GateRegistry::standard();
    // Wrong length.
    let mut c: Circuit = Circuit::new(2);
    c.diagonal("bad_len", vec![c64(1.0, 0.0); 3], vec![0, 1]);
    assert!(matches!(
        c.bind(&reg).unwrap_err(),
        Error::BadDimension {
            expected: 4,
            got: 3
        }
    ));
    // Non-unit entry.
    let mut c: Circuit = Circuit::new(1);
    c.diagonal("too_big", vec![c64(1.0, 0.0), c64(2.0, 0.0)], vec![0]);
    assert!(matches!(
        c.bind(&reg).unwrap_err(),
        Error::NotUnitary { .. }
    ));
    // Over split-complex, |d| = 1 is NOT enough: conj(j)·j = −1, so diag(1, j)
    // must be rejected even though every coefficient has unit magnitude.
    let reg_s = GateRegistry::<SplitComplex>::standard();
    let mut c: Circuit<SplitComplex> = Circuit::new(1);
    c.diagonal(
        "j_diag",
        vec![SplitComplex::one(), SplitComplex::basis(1)],
        vec![0],
    );
    assert!(matches!(
        c.bind(&reg_s).unwrap_err(),
        Error::NotUnitary { .. }
    ));
}

#[test]
fn circuit_inverse_uncomputes() {
    let reg: GateRegistry = GateRegistry::standard();
    let circuit = library::random_circuit(4, 40, 0xC0FFEE);
    let bound = circuit.bind(&reg).unwrap();
    let mut state = DenseState::<C64>::new(4).unwrap();
    bound.run(&mut state).unwrap();
    bound.inverse().run(&mut state).unwrap();
    assert!(state.amplitude(0).approx_eq(c64(1.0, 0.0), 1e-8));
    assert_close(state.total_weight(), 1.0, 1e-8);
}

#[test]
fn qft_followed_by_inverse_is_identity() {
    let reg: GateRegistry = GateRegistry::standard();
    let n = 5;
    // Random state via a scrambler, then QFT, then library iqft.
    let mut c: Circuit = scrambler(n);
    c.append(&library::qft(n), &(0..n).collect::<Vec<_>>());
    c.append(&library::iqft(n), &(0..n).collect::<Vec<_>>());
    let just_scrambled = scrambler(n);
    let (sa, sb) = (run_dense(&c), run_dense(&just_scrambled));
    for i in 0..(1u64 << n) {
        assert!(sa.amplitude(i).approx_eq(sb.amplitude(i), 1e-8));
    }
    // And the BoundCircuit::inverse route agrees with the explicit iqft.
    let qft_bound = library::qft::<C64>(n).bind(&reg).unwrap();
    let iqft_bound = library::iqft::<C64>(n).bind(&reg).unwrap();
    let mut s1 = DenseState::<C64>::new(n).unwrap();
    let mut s2 = DenseState::<C64>::new(n).unwrap();
    scrambler(n).bind(&reg).unwrap().run(&mut s1).unwrap();
    scrambler(n).bind(&reg).unwrap().run(&mut s2).unwrap();
    qft_bound.inverse().run(&mut s1).unwrap();
    iqft_bound.run(&mut s2).unwrap();
    for i in 0..(1u64 << n) {
        assert!(s1.amplitude(i).approx_eq(s2.amplitude(i), 1e-8));
    }
}
