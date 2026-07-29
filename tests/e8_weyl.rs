//! The cross-scale E8 Weyl pair, measured: one E8 as position, the
//! DUAL E8 as momentum (the same lattice — self-duality is measured,
//! not cited), coupled through the scale tower's 2-adic phase ladder.
//! Translations, modulations, W(E8) reflections and coordinate
//! Fouriers are native constellation gates; their algebra (Heisenberg
//! commutation, Clifford covariance, F⁴ = 1, exact support
//! uncertainty) is pinned by measurement, and structured coset states
//! interact — validly quantum interference included — at widths where
//! the dense representation measurably cannot exist.

mod common;

use quantsim::e8::constellation::{self, E8ConstellationState, Point};
use quantsim::prelude::*;

/// Largest amplitude deviation between two constellation states after
/// multiplying `b` by `phase` — 0 means `a = phase·b` exactly.
fn max_dev(a: &E8ConstellationState, b: &E8ConstellationState, phase: C64) -> f64 {
    let mut dev: f64 = 0.0;
    a.for_each_nonzero(&mut |i, amp| {
        dev = dev.max((amp - b.amplitude(i) * phase).norm());
    });
    b.for_each_nonzero(&mut |i, amp| {
        dev = dev.max((a.amplitude(i) - amp * phase).norm());
    });
    dev
}

fn weight(s: &E8ConstellationState) -> f64 {
    let mut t = 0.0;
    s.for_each_nonzero(&mut |_, a| t += a.norm_sqr());
    t
}

fn dup(s: &E8ConstellationState, n: usize) -> E8ConstellationState {
    let mut entries = Vec::new();
    s.for_each_nonzero(&mut |i, a| entries.push((i, a)));
    let mut out = E8ConstellationState::new(n).unwrap();
    out.load(&entries).unwrap();
    out
}

/// Basis vector j reconstructed through the measured duality:
/// `bⱼ = Σᵢ Gⱼᵢ·b*ᵢ`.
fn basis_row(j: usize) -> Point {
    let g = constellation::gram();
    let duals = constellation::dual_basis();
    let mut v = [0i64; 8];
    for (i, dual) in duals.iter().enumerate() {
        for (slot, &coord) in v.iter_mut().zip(dual) {
            *slot += g[j][i] * coord;
        }
    }
    v
}

fn as_point(r: &quantsim::e8::Root) -> Point {
    std::array::from_fn(|k| r[k] as i64)
}

fn reflect_point(x: &Point, alpha: &Point) -> Point {
    let inner = constellation::pdot(x, alpha) / 4;
    std::array::from_fn(|k| x[k] - (inner as i64) * alpha[k])
}

#[test]
fn self_duality_is_measured_exactly() {
    // dual_basis() itself asserts det(Gram) = 1; here the content:
    // every dual vector is an E8 point (E8* ⊆ E8, and E8 ⊆ E8* since
    // the lattice is integral — so E8* = E8, measured), the pairing is
    // exactly δᵢⱼ, and a point's basis coordinates ARE its inner
    // products with the dual vectors.
    let duals = constellation::dual_basis();
    for (i, dual) in duals.iter().enumerate() {
        assert!(
            constellation::class_of(dual).is_some(),
            "b*{i} must be a lattice point"
        );
        for j in 0..8 {
            let expected = if i == j { 4 } else { 0 }; // doubled = 4·real
            assert_eq!(
                constellation::pdot(dual, &basis_row(j)),
                expected,
                "⟨b*{i}, b{j}⟩"
            );
        }
    }
    // Basis rows reconstructed through the dual really are the basis:
    // coordinates are unit vectors.
    for j in 0..8 {
        let coords = constellation::coords_of(&basis_row(j)).unwrap();
        let unit: [i64; 8] = std::array::from_fn(|k| i64::from(k == j));
        assert_eq!(coords, unit);
    }
    // Coordinate duality on assorted points, roots included.
    let samples = [
        constellation::compose(&[137, 42]),
        constellation::compose(&[255, 1, 90]),
        as_point(&quantsim::e8::roots()[7]),
    ];
    for p in &samples {
        let coords = constellation::coords_of(p).unwrap();
        for (i, dual) in duals.iter().enumerate() {
            assert_eq!(i128::from(coords[i]), constellation::pdot(dual, p) / 4);
        }
    }
}

#[test]
fn the_weyl_pair_commutes_up_to_the_cross_scale_character() {
    // M_q T_v = χ_q(v)·T_v M_q with χ_q(v) = e^{2πi⟨q,v⟩/2^m}: the
    // measured Heisenberg law of the pair. With v = 2ʲα and q = 2ⁱβ at
    // ⟨α,β⟩ = 1, the phase is e^{2πi·2^{i+j}/2^m} — the 2-adic ladder:
    // scales interact when i + j < m and have already resolved past
    // each other (phase exactly 1) when i + j ≥ m.
    let rs = quantsim::e8::roots();
    let alpha = as_point(&rs[0]);
    let beta = rs
        .iter()
        .map(as_point)
        .find(|b| constellation::pdot(&alpha, b) == 4)
        .expect("roots at inner product +1 exist");

    let n = 24; // m = 3 levels
    let modulus = 4i128 << 3;
    for i in 0..3u32 {
        for j in 0..3u32 {
            let v: Point = alpha.map(|x| x << j);
            let q: Point = beta.map(|x| x << i);
            let build = |t_first: bool| {
                let mut s = E8ConstellationState::new(n).unwrap();
                s.coordinate_fourier(0).unwrap();
                s.coordinate_fourier(4).unwrap();
                if t_first {
                    s.translate(&v).unwrap();
                    s.modulate(&q).unwrap();
                } else {
                    s.modulate(&q).unwrap();
                    s.translate(&v).unwrap();
                }
                s
            };
            let mt = build(true);
            let tm = build(false);
            let r = constellation::pdot(&q, &v).rem_euclid(modulus);
            let angle = std::f64::consts::TAU * r as f64 / modulus as f64;
            let chi = c64(angle.cos(), angle.sin());
            assert!(max_dev(&mt, &tm, chi) < 1e-12, "({i},{j}): law violated");
            if i + j < 3 {
                // The character is genuinely nontrivial below the
                // horizon: r = 4·2^{i+j} ≠ 0 mod 2^{m+2}, and the
                // orderings measurably differ.
                assert_eq!(r, 4 << (i + j), "({i},{j}): the 2-adic ladder");
                assert!(
                    max_dev(&mt, &tm, c64(1.0, 0.0)) > 1e-6,
                    "({i},{j}): scales must still interact"
                );
            } else {
                assert_eq!(r, 0, "({i},{j}): past the horizon");
                assert!(
                    max_dev(&mt, &tm, c64(1.0, 0.0)) < 1e-12,
                    "({i},{j}): scales past the horizon commute exactly"
                );
            }
        }
    }
}

#[test]
fn the_weyl_group_is_a_measured_clifford_symmetry() {
    // W(E8) reflections permute the residue basis; measured: they are
    // involutions, preserve Born weight, and conjugate the Weyl pair
    // covariantly — s T_v s = T_{s(v)} and s M_q s = M_{s(q)} — which
    // is exactly what "Clifford" means for this Heisenberg group.
    let rs = quantsim::e8::roots();
    let mirror = as_point(&rs[17]);
    let v = as_point(&rs[3]).map(|x| x * 2); // cross-scale label
    let q = as_point(&rs[91]);

    let base = {
        let mut s = E8ConstellationState::new(16).unwrap();
        s.coordinate_fourier(1).unwrap();
        s.translate(&as_point(&rs[40])).unwrap();
        s
    };

    // Involution + weight.
    let mut twice = dup(&base, 16);
    twice.reflect(&rs[17]).unwrap();
    assert!((weight(&twice) - 1.0).abs() < 1e-9);
    twice.reflect(&rs[17]).unwrap();
    assert!(max_dev(&twice, &base, c64(1.0, 0.0)) < 1e-12, "s² = 1");

    // s T_v s = T_{s(v)}.
    let mut left = dup(&base, 16);
    left.reflect(&rs[17]).unwrap();
    left.translate(&v).unwrap();
    left.reflect(&rs[17]).unwrap();
    let mut right = dup(&base, 16);
    right.translate(&reflect_point(&v, &mirror)).unwrap();
    assert!(max_dev(&left, &right, c64(1.0, 0.0)) < 1e-12);

    // s M_q s = M_{s(q)}.
    let mut left = dup(&base, 16);
    left.reflect(&rs[17]).unwrap();
    left.modulate(&q).unwrap();
    left.reflect(&rs[17]).unwrap();
    let mut right = dup(&base, 16);
    right.modulate(&reflect_point(&q, &mirror)).unwrap();
    assert!(max_dev(&left, &right, c64(1.0, 0.0)) < 1e-12);

    // Only roots reflect.
    let mut s = E8ConstellationState::new(16).unwrap();
    assert!(s.reflect(&[2, 0, 0, 0, 0, 0, 0, 0]).is_err());
}

#[test]
fn fourier_conjugates_position_into_the_dual_momentum_e8() {
    // The measured conversion law between the two E8 objects:
    // F_dir ∘ T_{B_dir} ∘ F_dir⁻¹ = M_{−b*_dir} — a position-side
    // translation becomes a momentum-side modulation labeled by the
    // DUAL basis vector. F⁴ = 1 (so F⁻¹ = F³), measured first.
    let duals = constellation::dual_basis();
    let start = |n: usize| {
        let mut s = E8ConstellationState::new(n).unwrap();
        s.translate(&constellation::compose(&[99, 5])).unwrap();
        s
    };
    let mut cycled = start(16);
    for _ in 0..4 {
        cycled.coordinate_fourier(2).unwrap();
    }
    assert!(max_dev(&cycled, &start(16), c64(1.0, 0.0)) < 1e-9, "F⁴ = 1");

    for dir in [0usize, 5, 7] {
        let mut lhs = start(16);
        lhs.coordinate_fourier(dir).unwrap();
        lhs.translate(&basis_row(dir)).unwrap();
        for _ in 0..3 {
            lhs.coordinate_fourier(dir).unwrap();
        }
        let mut rhs = start(16);
        let neg: Point = duals[dir].map(|x| -x);
        rhs.modulate(&neg).unwrap();
        assert!(max_dev(&lhs, &rhs, c64(1.0, 0.0)) < 1e-9, "direction {dir}");
    }

    // Support uncertainty is EXACT on coset states: rank k in position
    // (2^{mk} points) meets rank 8−k in momentum, and the product is
    // the full group order 2^{8m} every time.
    for (k, expected_pos, expected_mom) in [(0usize, 1, 65536), (1, 4, 16384), (2, 16, 4096)] {
        let mut s = E8ConstellationState::new(16).unwrap();
        for dir in 0..k {
            s.coordinate_fourier(dir).unwrap();
        }
        assert_eq!(s.nonzero_count(), expected_pos);
        for dir in 0..8 {
            s.coordinate_fourier(dir).unwrap();
        }
        assert_eq!(s.nonzero_count(), expected_mom);
        assert_eq!(expected_pos * expected_mom, 1 << 16);
        assert!((weight(&s) - 1.0).abs() < 1e-9);
    }
}

#[test]
fn structured_sets_interact_beyond_the_dense_wall_without_blowup() {
    // 40 qubits: dense measurably cannot exist (memory admission), the
    // structured E8 objects interact in microseconds. The protocol
    // weaves all four natives: a rank-1 position line, a cross-scale
    // translation, a dual-E8 modulation, a W(E8) round trip, then the
    // inverse Fourier — closed form: the modulation scale 2² lands the
    // support on the single point with coordinate ≡ 4 (mod 32).
    assert!(DenseState::<C64>::new(40).is_err(), "the measured wall");

    let rs = quantsim::e8::roots();
    let duals = constellation::dual_basis();
    let started = std::time::Instant::now();
    let mut s = E8ConstellationState::new(40).unwrap(); // m = 5, d = 32
    s.coordinate_fourier(0).unwrap();
    assert_eq!(s.nonzero_count(), 32, "a rank-1 line, not 2^40 amplitudes");
    s.translate(&basis_row(0).map(|x| x * 8)).unwrap(); // 2³·B₀
    assert_eq!(s.nonzero_count(), 32, "the line is closed under its shifts");
    s.modulate(&duals[0].map(|x| x * 4)).unwrap(); // 2²·b*₀ on the dual side
    s.reflect(&rs[17]).unwrap();
    s.reflect(&rs[17]).unwrap(); // W(E8) round trip inside the protocol
    for _ in 0..3 {
        s.coordinate_fourier(0).unwrap(); // F⁻¹
    }
    let elapsed = started.elapsed();
    assert_eq!(s.nonzero_count(), 1, "interference collapsed the line");
    let point = s.stored_points()[0];
    let coords = constellation::coords_of(&point).unwrap();
    assert_eq!(
        coords[0].rem_euclid(32),
        4,
        "the modulation scale, read out"
    );
    assert_eq!(&coords[1..], &[0; 7], "no leakage into other directions");
    assert!((weight(&s) - 1.0).abs() < 1e-9);
    assert!(elapsed.as_secs_f64() < 5.0, "{elapsed:?}");

    // Native gates are residue-group operations: partial blocks and
    // non-lattice labels refuse.
    let mut partial = E8ConstellationState::new(12).unwrap();
    assert!(partial.translate(&basis_row(0)).is_err());
    let mut full = E8ConstellationState::new(16).unwrap();
    assert!(full.translate(&[1, 0, 0, 0, 0, 0, 0, 0]).is_err());
    assert!(full.modulate(&[1, 0, 0, 0, 0, 0, 0, 0]).is_err());
    assert!(full.coordinate_fourier(8).is_err());
}

#[test]
fn natives_reduce_to_qubit_gates_at_depth_one() {
    // At m = 1 the residue group IS F₂⁸, so the natives must land on
    // ordinary qubit gates: translate(v) is the X-string on the bits
    // of class(v), and modulate(q) is the diagonal (−1)^{⟨q,p⟩} — both
    // checked amplitude-by-amplitude against the standard framework on
    // a scrambled 8-qubit state.
    let sim: Simulator = Simulator::new();
    let scrambled = sim.run(&library::random_circuit(8, 60, 5)).unwrap();
    let mut entries = Vec::new();
    scrambled.for_each_nonzero(&mut |i, a| entries.push((i, a)));

    let v = as_point(&quantsim::e8::roots()[123]);
    let class = constellation::class_of(&v).unwrap();
    let mut native = E8ConstellationState::new(8).unwrap();
    native.load(&entries).unwrap();
    native.translate(&v).unwrap();
    let mut c: Circuit = Circuit::new(8);
    for bit in 0..8 {
        if (class >> bit) & 1 == 1 {
            c.x(bit);
        }
    }
    let mut xstring = DenseState::<C64>::new(8).unwrap();
    xstring.load(&entries).unwrap();
    c.bind(sim.registry()).unwrap().run(&mut xstring).unwrap();
    let mut dev: f64 = 0.0;
    native.for_each_nonzero(&mut |i, a| dev = dev.max((a - xstring.amplitude(i)).norm()));
    assert!(
        dev < 1e-12,
        "translate at depth 1 is the X-string: {dev:.2e}"
    );

    // modulate(q) vs the independently computed sign pattern.
    let q = as_point(&quantsim::e8::roots()[201]);
    let mut native = E8ConstellationState::new(8).unwrap();
    native.load(&entries).unwrap();
    native.modulate(&q).unwrap();
    for &(bits, amp) in &entries {
        let p = constellation::compose(&[bits as u8]);
        let sign = if constellation::pdot(&q, &p).rem_euclid(8) == 0 {
            1.0
        } else {
            -1.0 // doubled dot ≡ 4 (mod 8) ⟺ real ⟨q,p⟩ odd
        };
        let got = native.amplitude(bits);
        assert!((got - amp * sign).norm() < 1e-12);
    }
}
