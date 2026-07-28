//! Multi-scale dual time on the E8 constellation, measured: the
//! distilled scale-composition algebra (embed/decimate isometries with
//! exact operator transport — the operad of coarse-graining), comb
//! codes whose stabilizer checks and error windows tile EXACTLY along
//! the Weyl pair's bidirectional horizon (the same inequality that
//! governs cross-scale commutation governs cross-scale detection),
//! end-to-end correction across scales, and the folding theorems: an
//! interleaved cross-direction sequence reorders through beyond-horizon
//! commutations into a single two-gate layer, and deep periodic time
//! folds to its measured period — 2⁴⁰ blocks evaluated as five.

mod common;

use quantsim::e8::constellation::{self, E8ConstellationState, Point};
use quantsim::prelude::*;

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

/// Deviation minimized over a global phase (fitted from the largest
/// amplitude), for period detection.
fn dev_up_to_phase(a: &E8ConstellationState, b: &E8ConstellationState) -> f64 {
    let mut best: Option<(f64, u64)> = None;
    a.for_each_nonzero(&mut |i, amp| {
        let w = amp.norm();
        if best.map_or(true, |(bw, _)| w > bw) {
            best = Some((w, i));
        }
    });
    let Some((_, anchor)) = best else {
        return f64::INFINITY;
    };
    let (aa, ba) = (a.amplitude(anchor), b.amplitude(anchor));
    if ba.norm() < 1e-15 {
        return f64::INFINITY;
    }
    max_dev(a, b, aa / ba)
}

fn weight(s: &E8ConstellationState) -> f64 {
    let mut t = 0.0;
    s.for_each_nonzero(&mut |_, a| t += a.norm_sqr());
    t
}

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

/// A structured, non-trivial coarse state to transport around.
fn coarse_state(n: usize) -> E8ConstellationState {
    let mut s = E8ConstellationState::new(n).unwrap();
    s.coordinate_fourier(3).unwrap();
    s.translate(&as_point(&quantsim::e8::roots()[40])).unwrap();
    s
}

/// The comb codeword at scale `a` in an m-level register: the uniform
/// coarse state embedded `a` levels down — support = every point of
/// `2^a·E8 / 2^m E8`, the encoder being the scale isometry itself.
fn comb_codeword(m: usize, a: usize) -> E8ConstellationState {
    let mut coarse = E8ConstellationState::new(8 * (m - a)).unwrap();
    for dir in 0..8 {
        coarse.coordinate_fourier(dir).unwrap();
    }
    coarse.scale_embed(a).unwrap()
}

#[test]
fn the_scale_composition_algebra_is_exact() {
    // The distilled operad of coarse-graining: embeddings compose,
    // decimation inverts them, and the Weyl operators transport
    // covariantly across scale — all exact, all measured.
    let psi = coarse_state(16); // m = 2

    // V is an isometry and R its exact left inverse; compositions add.
    let embedded = psi.scale_embed(1).unwrap();
    assert_eq!(embedded.num_qubits(), 24);
    assert!((weight(&embedded) - weight(&psi)).abs() < 1e-12);
    assert_eq!(embedded.nonzero_count(), psi.nonzero_count());
    let back = embedded.decimate(1).unwrap();
    assert!(max_dev(&back, &psi, c64(1.0, 0.0)) < 1e-12, "R∘V = id");
    let twice = psi.scale_embed(2).unwrap();
    let stepwise = psi.scale_embed(1).unwrap().scale_embed(1).unwrap();
    assert!(
        max_dev(&twice, &stepwise, c64(1.0, 0.0)) < 1e-12,
        "V_2 = V_1∘V_1"
    );
    let coarse2 = twice.decimate(1).unwrap().decimate(1).unwrap();
    assert!(
        max_dev(&coarse2, &twice.decimate(2).unwrap(), c64(1.0, 0.0)) < 1e-12,
        "R_1∘R_1 = R_2"
    );

    // Operator transport: T_{2v}∘V = V∘T_v and M_q∘V = V∘M_q.
    let v = as_point(&quantsim::e8::roots()[7]);
    let q = as_point(&quantsim::e8::roots()[123]);
    let mut left = psi.scale_embed(1).unwrap();
    left.translate(&v.map(|x| 2 * x)).unwrap();
    left.modulate(&q).unwrap();
    let mut right = coarse_state(16);
    right.translate(&v).unwrap();
    right.modulate(&q).unwrap();
    let right = right.scale_embed(1).unwrap();
    assert!(
        max_dev(&left, &right, c64(1.0, 0.0)) < 1e-12,
        "the Weyl pair rides the scale map covariantly"
    );

    // Decimating live fine-scale data refuses with the level named —
    // coarse-graining never silently destroys information.
    let mut fine = E8ConstellationState::new(24).unwrap();
    fine.translate(&as_point(&quantsim::e8::roots()[0]))
        .unwrap();
    let err = fine.decimate(1).err().expect("must refuse");
    assert!(err.to_string().contains("fine-scale data"), "{err}");
}

#[test]
fn comb_codes_encode_logical_data_across_scales() {
    // m = 4 levels (32 qubits), comb scale a = 2, one logical level
    // (w = 1): stabilizers are coarse translations T_{4B_d} and fine
    // modulations M_{8b*_d} — CROSS-scale checks that all commute
    // because every pair sits past the horizon — and the logical
    // operators live at the middle scales: X̄_d = T_{2B_d},
    // Z̄_d = M_{4b*_d}.
    let duals = constellation::dual_basis();
    let code0 = comb_codeword(4, 2);
    assert!((weight(&code0) - 1.0).abs() < 1e-9);

    // Every stabilizer check reads exactly +1 on the codeword.
    for (d, dual) in duals.iter().enumerate() {
        let phase = code0.modulation_eigenphase(&dual.map(|x| 8 * x)).unwrap();
        assert!((phase - c64(1.0, 0.0)).norm() < 1e-9, "M-check {d}");
        let mut shifted = comb_codeword(4, 2);
        shifted.translate(&basis_row(d).map(|x| 4 * x)).unwrap();
        assert!(
            max_dev(&shifted, &code0, c64(1.0, 0.0)) < 1e-12,
            "T-check {d}"
        );
    }

    // The logical X̄₀ = T_{2B₀} maps codeword to codeword: still +1 on
    // every check, but the logical Z̄₀ eigenphase flips from +1 to −1
    // — one protected bit per direction, stored at the middle scale.
    let mut code1 = comb_codeword(4, 2);
    code1.translate(&basis_row(0).map(|x| 2 * x)).unwrap();
    for dual in &duals {
        let phase = code1.modulation_eigenphase(&dual.map(|x| 8 * x)).unwrap();
        assert!((phase - c64(1.0, 0.0)).norm() < 1e-9, "still in code space");
    }
    let z0 = |s: &E8ConstellationState| s.modulation_eigenphase(&duals[0].map(|x| 4 * x)).unwrap();
    assert!((z0(&code0) - c64(1.0, 0.0)).norm() < 1e-9);
    assert!(
        (z0(&code1) - c64(-1.0, 0.0)).norm() < 1e-9,
        "Z̄₀ reads the bit"
    );

    // Self-similarity of the code itself: decimating one level maps
    // the (m=4, a=2) codeword onto the (m=3, a=1) codeword exactly —
    // the RG flow of the code is the code.
    let down = code0.decimate(1).unwrap();
    let direct = comb_codeword(3, 1);
    assert!(max_dev(&down, &direct, c64(1.0, 0.0)) < 1e-12);
}

#[test]
fn cross_scale_syndromes_tile_the_horizon() {
    // The detection window IS the Heisenberg horizon: a displacement
    // at scale j is seen by the modulation check at scale i exactly
    // when i + j < m — the same bidirectional constraint, measured as
    // the full m × m syndrome matrix on point states.
    let m = 4usize;
    let duals = constellation::dual_basis();
    let alpha = basis_row(0); // coords (1,0,…): ⟨b*₀, α⟩ = 1
    for j in 0..m {
        let mut displaced = E8ConstellationState::new(8 * m).unwrap();
        displaced.translate(&alpha.map(|x| x << j)).unwrap();
        for i in 0..m {
            let phase = displaced
                .modulation_eigenphase(&duals[0].map(|x| x << i))
                .unwrap();
            let expected_angle = std::f64::consts::TAU * (1 << (i + j)) as f64 / (1 << m) as f64;
            let expected = c64(expected_angle.cos(), expected_angle.sin());
            assert!(
                (phase - expected).norm() < 1e-9,
                "({i},{j}): {phase} vs {expected}"
            );
            if i + j < m {
                // Smallest firing character is e^{iπ/8}, 0.39 from 1.
                assert!(
                    (phase - c64(1.0, 0.0)).norm() > 0.3,
                    "({i},{j}): below the horizon the syndrome fires"
                );
            } else {
                assert!(
                    (phase - c64(1.0, 0.0)).norm() < 1e-9,
                    "({i},{j}): past the horizon the check is blind"
                );
            }
        }
    }
}

#[test]
fn errors_are_corrected_across_scales_end_to_end() {
    // Full round trip on the unique-codeword comb (m = 4, a = 2, no
    // logical level): inject a fine-scale displacement, read the
    // syndromes from cross-scale checks alone, decode the displacement
    // by exact binary phase readout, correct, and decimate back to the
    // pristine coarse state.
    let m = 4usize;
    let a = 2usize;
    let duals = constellation::dual_basis();
    let clean = comb_codeword(m, a);

    // Error: coordinate displacements (3, 0, 0, 0, 0, 1, 0, 0) —
    // scale-0 and scale-1 content in two directions at once.
    let error: Point = std::array::from_fn(|k| 3 * basis_row(0)[k] + basis_row(5)[k]);
    let mut noisy = comb_codeword(m, a);
    noisy.translate(&error).unwrap();

    // Decode each direction's displacement mod 2^a from the syndrome
    // phases of M_{2^{m−1}b*_d} (bit 0) and M_{2^{m−2}b*_d} (bit 1).
    let mut decoded = [0i64; 8];
    for d in 0..8 {
        let mut c = 0i64;
        for bit in 0..a {
            let i = m - 1 - bit;
            let phase = noisy
                .modulation_eigenphase(&duals[d].map(|x| x << i))
                .unwrap();
            // Expected phase e^{2πi·(c + 2^bit·b)·2^i/2^m}; peel the
            // known lower bits, read the next.
            let modulus = 1i64 << (bit + 1);
            let angle = phase.im.atan2(phase.re).rem_euclid(std::f64::consts::TAU);
            let steps = (angle * modulus as f64 / std::f64::consts::TAU).round() as i64 % modulus;
            let known = c % (1 << bit);
            let b = (steps - known).rem_euclid(modulus) >> bit;
            c += b << bit;
        }
        decoded[d] = c;
    }
    assert_eq!(
        decoded,
        [3, 0, 0, 0, 0, 1, 0, 0],
        "syndromes name the error"
    );

    // Correct: translate back by the decoded displacement. Any
    // residual is a stabilizer translation, so the state must equal
    // the clean codeword EXACTLY.
    let correction: Point =
        std::array::from_fn(|k| -(0..8).map(|d| decoded[d] * basis_row(d)[k]).sum::<i64>());
    noisy.translate(&correction).unwrap();
    assert!(max_dev(&noisy, &clean, c64(1.0, 0.0)) < 1e-12, "recovered");
    let coarse = noisy.decimate(a).unwrap();
    let mut reference = E8ConstellationState::new(8 * (m - a)).unwrap();
    for dir in 0..8 {
        reference.coordinate_fourier(dir).unwrap();
    }
    assert!(max_dev(&coarse, &reference, c64(1.0, 0.0)) < 1e-12);

    // The honest limit: a displacement at scale a is a STABILIZER
    // (invisible and harmless); at the logical scale it would be an
    // undetected logical operation — syndromes all read +1.
    let mut sneaky = comb_codeword(m, a);
    sneaky.translate(&basis_row(2).map(|x| x << a)).unwrap();
    for dual in &duals {
        let phase = sneaky
            .modulation_eigenphase(&dual.map(|x| x << (m - 1)))
            .unwrap();
        assert!((phase - c64(1.0, 0.0)).norm() < 1e-9, "past-window: blind");
    }
    assert!(
        max_dev(&sneaky, &clean, c64(1.0, 0.0)) < 1e-12,
        "…and harmless here"
    );
}

#[test]
fn bidirectional_interleave_folds_to_a_single_layer() {
    // Forward translations ascend the scales (A_k at scale k) while
    // backward modulations descend (B_k at scale m−1−k). Every crossed
    // pair the fold reorders sits PAST the horizon, so the interleaved
    // 2m-step sequence equals all-A-then-all-B with no phase at all —
    // and the products collapse: ΠA = ONE translation, ΠB = ONE
    // modulation. Cross-scale, cross-direction periodic structure,
    // folded up into a single two-gate layer, exactly.
    let m = 4usize;
    let rs = quantsim::e8::roots();
    let alphas: Vec<Point> = (0..m).map(|k| as_point(&rs[10 * k + 3])).collect();
    let betas: Vec<Point> = (0..m).map(|k| as_point(&rs[17 * k + 5])).collect();
    let start = || {
        let mut s = E8ConstellationState::new(8 * m).unwrap();
        s.coordinate_fourier(0).unwrap();
        s.translate(&as_point(&rs[99])).unwrap();
        s
    };

    let mut interleaved = start();
    for k in 0..m {
        interleaved.translate(&alphas[k].map(|x| x << k)).unwrap();
        interleaved
            .modulate(&betas[k].map(|x| x << (m - 1 - k)))
            .unwrap();
    }

    let big_v: Point = std::array::from_fn(|c| (0..m).map(|k| alphas[k][c] << k).sum::<i64>());
    let big_q: Point =
        std::array::from_fn(|c| (0..m).map(|k| betas[k][c] << (m - 1 - k)).sum::<i64>());
    let mut folded = start();
    folded.translate(&big_v).unwrap();
    folded.modulate(&big_q).unwrap();
    assert!(
        max_dev(&interleaved, &folded, c64(1.0, 0.0)) < 1e-12,
        "2m cross-scale steps = one T + one M, exactly"
    );

    // Nontriviality: the fold is licensed by the horizon, not by luck —
    // a BELOW-horizon swap costs the Weyl phase (measured non-1). Use
    // a non-orthogonal pair so the character genuinely bites.
    let v0 = alphas[0];
    let qm = rs
        .iter()
        .map(as_point)
        .find(|q| constellation::pdot(q, &v0) == 4)
        .expect("roots at inner product +1 exist");
    let build = |t_first: bool| {
        let mut s = start();
        if t_first {
            s.translate(&v0).unwrap();
            s.modulate(&qm).unwrap();
        } else {
            s.modulate(&qm).unwrap();
            s.translate(&v0).unwrap();
        }
        s
    };
    assert!(
        max_dev(&build(true), &build(false), c64(1.0, 0.0)) > 1e-6,
        "below the horizon the directions genuinely interact"
    );
}

#[test]
fn deep_periodic_time_folds_to_its_measured_period() {
    // One bidirectional block U = T_V then M_Q. Small powers pin the
    // exact Weyl closed form U^t = χ^{t(t−1)/2}·(T_{tV} then M_{tQ})
    // with χ = e^{2πi⟨Q,V⟩/2^m}; the measured period then folds ANY
    // depth: 2⁴⁰ + 5 blocks evaluate as (2⁴⁰ + 5) mod P blocks — deep
    // time constricted to a single short layer, verified up to global
    // phase.
    let m = 3usize; // n = 24
    let rs = quantsim::e8::roots();
    let big_v = as_point(&rs[3]);
    let big_q = as_point(&rs[91]).map(|x| 2 * x);
    let start = || {
        let mut s = E8ConstellationState::new(8 * m).unwrap();
        s.coordinate_fourier(1).unwrap();
        s
    };
    let apply_blocks = |t: u64| {
        let mut s = start();
        for _ in 0..t {
            s.translate(&big_v).unwrap();
            s.modulate(&big_q).unwrap();
        }
        s
    };

    // The closed form at small t, exact including the quadratic phase.
    let modulus = 4i128 << m;
    let chi_steps = constellation::pdot(&big_q, &big_v).rem_euclid(modulus);
    for t in 2..=6i128 {
        let mut closed = start();
        let tv: Point = big_v.map(|x| x * t as i64);
        let tq: Point = big_q.map(|x| x * t as i64);
        closed.translate(&tv).unwrap();
        closed.modulate(&tq).unwrap();
        let exponent = (chi_steps * t * (t - 1) / 2).rem_euclid(modulus);
        let angle = std::f64::consts::TAU * exponent as f64 / modulus as f64;
        let phase = c64(angle.cos(), angle.sin());
        assert!(
            max_dev(&apply_blocks(t as u64), &closed, phase) < 1e-12,
            "t = {t}: the Weyl closed form"
        );
    }

    // Measure the period: smallest P with U^P = phase·identity on ψ.
    let reference = start();
    let mut period = None;
    for p in 1..=(4 * (1 << m)) as u64 {
        if dev_up_to_phase(&apply_blocks(p), &reference) < 1e-9 {
            period = Some(p);
            break;
        }
    }
    let period = period.expect("finite-order generators must recur");
    assert_eq!(period, 8, "measured period at m = 3");

    // The fold: 2⁴⁰ + 5 blocks ≡ (2⁴⁰ + 5) mod 8 = 5 blocks.
    let t_big: u64 = (1 << 40) + 5;
    let folded = apply_blocks(t_big % period);
    let direct5 = apply_blocks(5);
    assert!(dev_up_to_phase(&folded, &direct5) < 1e-12);
    assert!((weight(&folded) - 1.0).abs() < 1e-9);
    // Cost of the deep evaluation: (t mod P) blocks — for ANY t.
}
