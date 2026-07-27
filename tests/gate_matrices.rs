//! Compare every standard gate's matrix against literature values, and pin
//! which gates each amplitude algebra supports.

mod common;

use quantsim::prelude::*;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4, PI};

fn mat(reg: &GateRegistry, name: &str, params: &[f64]) -> GateMatrix<C64> {
    reg.resolve(name).unwrap().matrix(params).unwrap()
}

fn expect(name: &str, got: &GateMatrix<C64>, entries: &[(f64, f64)]) {
    let dim = got.dim();
    assert_eq!(entries.len(), dim * dim, "test data for {name}");
    for r in 0..dim {
        for c in 0..dim {
            let (re, im) = entries[r * dim + c];
            let g = got.get(r, c);
            assert!(
                g.approx_eq(c64(re, im), 1e-12),
                "{name}[{r},{c}] = {g}, expected {re}{im:+}i"
            );
        }
    }
}

#[test]
fn fixed_single_qubit_gates_match_literature() {
    let reg: GateRegistry = GateRegistry::standard();
    let s = FRAC_1_SQRT_2;
    expect(
        "id",
        &mat(&reg, "id", &[]),
        &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
    );
    expect(
        "x",
        &mat(&reg, "x", &[]),
        &[(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)],
    );
    expect(
        "y",
        &mat(&reg, "y", &[]),
        &[(0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.0, 0.0)],
    );
    expect(
        "z",
        &mat(&reg, "z", &[]),
        &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-1.0, 0.0)],
    );
    expect(
        "h",
        &mat(&reg, "h", &[]),
        &[(s, 0.0), (s, 0.0), (s, 0.0), (-s, 0.0)],
    );
    expect(
        "s",
        &mat(&reg, "s", &[]),
        &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 1.0)],
    );
    expect(
        "sdg",
        &mat(&reg, "sdg", &[]),
        &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, -1.0)],
    );
    expect(
        "t",
        &mat(&reg, "t", &[]),
        &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (s, s)],
    );
    expect(
        "tdg",
        &mat(&reg, "tdg", &[]),
        &[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (s, -s)],
    );
    expect(
        "sx",
        &mat(&reg, "sx", &[]),
        &[(0.5, 0.5), (0.5, -0.5), (0.5, -0.5), (0.5, 0.5)],
    );
    expect(
        "sxdg",
        &mat(&reg, "sxdg", &[]),
        &[(0.5, -0.5), (0.5, 0.5), (0.5, 0.5), (0.5, -0.5)],
    );
}

#[test]
fn parametric_gates_at_special_angles() {
    let reg: GateRegistry = GateRegistry::standard();
    let s = FRAC_1_SQRT_2;
    // rx(π/2) = 1/√2 [[1, −i], [−i, 1]]
    expect(
        "rx(π/2)",
        &mat(&reg, "rx", &[FRAC_PI_2]),
        &[(s, 0.0), (0.0, -s), (0.0, -s), (s, 0.0)],
    );
    // ry(π/2) = 1/√2 [[1, −1], [1, 1]]
    expect(
        "ry(π/2)",
        &mat(&reg, "ry", &[FRAC_PI_2]),
        &[(s, 0.0), (-s, 0.0), (s, 0.0), (s, 0.0)],
    );
    // rz(π) = diag(−i, i)
    expect(
        "rz(π)",
        &mat(&reg, "rz", &[PI]),
        &[(0.0, -1.0), (0.0, 0.0), (0.0, 0.0), (0.0, 1.0)],
    );
    // p(π) = Z, p(π/2) = S, p(π/4) = T
    assert!(mat(&reg, "p", &[PI]).approx_eq(&mat(&reg, "z", &[]), 1e-12));
    assert!(mat(&reg, "p", &[FRAC_PI_2]).approx_eq(&mat(&reg, "s", &[]), 1e-12));
    assert!(mat(&reg, "p", &[FRAC_PI_4]).approx_eq(&mat(&reg, "t", &[]), 1e-12));
    // u(π/2, 0, π) = H; u(θ, −π/2, π/2) = rx(θ); u(θ, 0, 0) = ry(θ)
    assert!(mat(&reg, "u", &[FRAC_PI_2, 0.0, PI]).approx_eq(&mat(&reg, "h", &[]), 1e-12));
    let theta = 0.9137;
    assert!(mat(&reg, "u", &[theta, -FRAC_PI_2, FRAC_PI_2])
        .approx_eq(&mat(&reg, "rx", &[theta]), 1e-12));
    assert!(mat(&reg, "u", &[theta, 0.0, 0.0]).approx_eq(&mat(&reg, "ry", &[theta]), 1e-12));
}

#[test]
fn two_qubit_gates_match_literature() {
    let reg: GateRegistry = GateRegistry::standard();
    let o = (1.0, 0.0);
    let l = (0.0, 0.0);
    // Little-endian CX (control = qubit 0 = low bit): swaps |01⟩ ↔ |11⟩.
    expect(
        "cx",
        &mat(&reg, "cx", &[]),
        &[o, l, l, l, l, l, l, o, l, l, o, l, l, o, l, l],
    );
    expect(
        "cz",
        &mat(&reg, "cz", &[]),
        &[o, l, l, l, l, o, l, l, l, l, o, l, l, l, l, (-1.0, 0.0)],
    );
    expect(
        "swap",
        &mat(&reg, "swap", &[]),
        &[o, l, l, l, l, l, o, l, l, o, l, l, l, l, l, o],
    );
    expect(
        "iswap",
        &mat(&reg, "iswap", &[]),
        &[
            o,
            l,
            l,
            l,
            l,
            l,
            (0.0, 1.0),
            l,
            l,
            (0.0, 1.0),
            l,
            l,
            l,
            l,
            l,
            o,
        ],
    );
    // rzz(θ) = diag(e^{−iθ/2}, e^{+iθ/2}, e^{+iθ/2}, e^{−iθ/2})
    let th: f64 = 0.81;
    let (c2, s2) = ((th / 2.0).cos(), (th / 2.0).sin());
    expect(
        "rzz",
        &mat(&reg, "rzz", &[th]),
        &[
            (c2, -s2),
            l,
            l,
            l,
            l,
            (c2, s2),
            l,
            l,
            l,
            l,
            (c2, s2),
            l,
            l,
            l,
            l,
            (c2, -s2),
        ],
    );
    // rxx(π/2) has 1/√2 on the diagonal and −i/√2 on the anti-diagonal.
    let s = FRAC_1_SQRT_2;
    expect(
        "rxx(π/2)",
        &mat(&reg, "rxx", &[FRAC_PI_2]),
        &[
            (s, 0.0),
            l,
            l,
            (0.0, -s),
            l,
            (s, 0.0),
            (0.0, -s),
            l,
            l,
            (0.0, -s),
            (s, 0.0),
            l,
            (0.0, -s),
            l,
            l,
            (s, 0.0),
        ],
    );
    // ryy(π/2): +i/√2 in the outer anti-diagonal corners, −i/√2 inner.
    expect(
        "ryy(π/2)",
        &mat(&reg, "ryy", &[FRAC_PI_2]),
        &[
            (s, 0.0),
            l,
            l,
            (0.0, s),
            l,
            (s, 0.0),
            (0.0, -s),
            l,
            l,
            (0.0, -s),
            (s, 0.0),
            l,
            (0.0, s),
            l,
            l,
            (s, 0.0),
        ],
    );
    // cp(π) = cz.
    assert!(mat(&reg, "cp", &[PI]).approx_eq(&mat(&reg, "cz", &[]), 1e-12));
}

#[test]
fn three_qubit_gates_are_correct_permutations() {
    let reg: GateRegistry = GateRegistry::standard();
    let ccx = mat(&reg, "ccx", &[]);
    // Permutation: identity except 3 ↔ 7 (both controls set flips target).
    for col in 0..8u64 {
        let expected_row = match col {
            3 => 7,
            7 => 3,
            other => other,
        };
        for row in 0..8u64 {
            let want = if row == expected_row { 1.0 } else { 0.0 };
            assert!(
                ccx.get(row as usize, col as usize)
                    .approx_eq(c64(want, 0.0), 1e-12),
                "ccx[{row},{col}]"
            );
        }
    }
    let ccz = mat(&reg, "ccz", &[]);
    for i in 0..8 {
        let want = if i == 7 { -1.0 } else { 1.0 };
        assert!(ccz.get(i, i).approx_eq(c64(want, 0.0), 1e-12));
    }
    // cswap: control bit 0; swaps sub-bits 1 and 2 → 3 ↔ 5.
    let cswap = mat(&reg, "cswap", &[]);
    for col in 0..8u64 {
        let expected_row = match col {
            3 => 5,
            5 => 3,
            other => other,
        };
        assert!(
            cswap
                .get(expected_row as usize, col as usize)
                .approx_eq(c64(1.0, 0.0), 1e-12),
            "cswap[{expected_row},{col}]"
        );
    }
}

#[test]
fn aliases_resolve_to_the_same_gates() {
    let reg: GateRegistry = GateRegistry::standard();
    for (alias, canon) in [
        ("cnot", "cx"),
        ("toffoli", "ccx"),
        ("fredkin", "cswap"),
        ("phase", "p"),
        ("not", "x"),
        ("u3", "u"),
        ("cphase", "cp"),
    ] {
        let pc = reg.resolve(canon).unwrap().param_count();
        let params: Vec<f64> = (0..pc).map(|i| 0.3 + i as f64).collect();
        assert!(
            mat(&reg, alias, &params).approx_eq(&mat(&reg, canon, &params), 1e-12),
            "{alias} != {canon}"
        );
    }
}

#[test]
fn every_registered_gate_is_unitary_at_probe_params() {
    // The registry validates this on registration; re-check here at several
    // parameter draws, for every gate including aliases.
    let reg: GateRegistry = GateRegistry::standard();
    let mut rng = Prng::new(99);
    for name in reg.names() {
        let def = reg.resolve(&name).unwrap();
        for _ in 0..5 {
            let params: Vec<f64> = (0..def.param_count())
                .map(|_| rng.next_f64() * 12.0 - 6.0)
                .collect();
            let m = def.matrix(&params).unwrap();
            assert_eq!(m.dim(), 1 << def.arity(), "{name} dimension");
            let dev = m.unitarity_deviation();
            assert!(dev < 1e-9, "{name} deviates from unitarity by {dev}");
        }
    }
}

#[test]
fn algebra_gate_support_matrix() {
    // ℂ carries the full library (32 canonical + 7 aliases)...
    let full: Vec<String> = GateRegistry::<C64>::standard().names();
    assert_eq!(full.len(), 39, "complex registry: {full:?}");
    // ...and so do the quaternions and octonions, because ℂ embeds.
    assert_eq!(GateRegistry::<Quaternion>::standard().names(), full);
    assert_eq!(GateRegistry::<Octonion>::standard().names(), full);
    assert_eq!(GateRegistry::<Sedenion>::standard().names(), full);

    // ℝ gets exactly the real-matrix subset.
    let real = GateRegistry::<f64>::standard().names();
    let expected = vec![
        "ccx", "ccz", "ch", "cnot", "cry", "cswap", "cx", "cz", "fredkin", "h", "id", "not", "ry",
        "swap", "toffoli", "x", "z",
    ];
    assert_eq!(real, expected, "real registry: {real:?}");
    // Split-complex has no i either → same real subset.
    assert_eq!(GateRegistry::<SplitComplex>::standard().names(), real);
    // CD<f64> *is* ℂ, so it must carry the full library.
    assert_eq!(GateRegistry::<CComplex>::standard().names(), full);
}

#[test]
fn unsupported_gate_reports_algebra() {
    let mut reg = GateRegistry::<f64>::new();
    let err = reg
        .register_parametric("s_real", "phase S over R", 1, 0, |_| {
            GateMatrix::<f64>::try_from_c64s(
                2,
                &[c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(0.0, 1.0)],
            )
            .ok_or(Error::UnsupportedForAlgebra {
                gate: "s_real".into(),
                algebra: "R".into(),
            })
        })
        .unwrap_err();
    assert!(
        matches!(err, Error::UnsupportedForAlgebra { .. }),
        "{err:?}"
    );
}
