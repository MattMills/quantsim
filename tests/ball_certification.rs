//! Ball-arithmetic certification: the radii are *sound*, proven against
//! the exact D[ω] evaluator — at full resolution, and under deliberately
//! coarse-grained (quantized) gates.
//!
//! The soundness relation is containment: for every final amplitude, the
//! exact ring value must lie inside the computed ball. Full-resolution
//! runs certify that the conservative rounding inflation covers the real
//! float error; quantized-gate runs certify that deliberate coarse
//! graining stays sound while the radii honestly grow — and shrink again
//! when the resolution is refined. That is "compute at coarse resolution,
//! then refine" as a *checked* property of the number type.

mod common;

use quantsim::exact::ExactState;
use quantsim::prelude::*;

/// The exact-fragment circuit family (Clifford+T plus snapped rotations),
/// mirrored from the exact-reference suite.
fn exact_fragment_circuit(n: usize, gates: usize, seed: u64) -> Circuit<Ball> {
    let pi4 = std::f64::consts::FRAC_PI_4;
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for _ in 0..gates {
        let q = (rng.next_u64() % n as u64) as usize;
        let r = ((q + 1) + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
        match rng.next_u64() % 10 {
            0 => c.h(q),
            1 => c.t(q),
            2 => c.tdg(q),
            3 => c.s(q),
            4 => c.sx(q),
            5 => c.rz(q, 2.0 * pi4 * (1 + rng.next_u64() % 7) as f64),
            6 => c.cx(q, r),
            7 => c.cz(q, r),
            8 => c.cp(q, r, pi4 * (1 + rng.next_u64() % 7) as f64),
            _ => c.ry(q, 2.0 * pi4 * (1 + rng.next_u64() % 7) as f64),
        };
    }
    c
}

/// The same op list, retyped for the exact evaluator.
fn retype(c: &Circuit<Ball>) -> Circuit<C64> {
    let mut out = Circuit::new(c.num_qubits());
    for op in c.ops() {
        if let Op::Named {
            name,
            params,
            qubits,
        } = op
        {
            out.gate(name.clone(), params.clone(), qubits.clone());
        } else {
            panic!("fragment circuits are name-only");
        }
    }
    out
}

#[test]
fn radii_contain_the_exact_values_at_full_resolution() {
    for (n, gates, seed) in [(5usize, 120usize, 4u64), (7, 200, 5)] {
        let circuit = exact_fragment_circuit(n, gates, seed);
        let exact = ExactState::run(&retype(&circuit)).unwrap();

        let sim: Simulator<Ball> = Simulator::new();
        let state = sim.run(&circuit).unwrap();
        let mut worst_ratio = 0.0f64;
        let mut max_rad = 0.0f64;
        for i in 0..(1u64 << n) {
            let ball = state.amplitude(i);
            let truth = exact.amplitude_c64(i);
            assert!(
                ball.contains(truth),
                "n={n} seed={seed} amp {i}: exact {truth} outside ball ({}, r={:.3e})",
                ball.mid,
                ball.rad
            );
            let err = (truth - ball.mid).norm();
            if ball.rad > 0.0 {
                worst_ratio = worst_ratio.max(err / ball.rad);
            }
            max_rad = max_rad.max(ball.rad);
        }
        // The bound is meaningful, not vacuous — but interval arithmetic
        // pays the classic dependency cost: radii can amplify by ~√2 per
        // Hadamard-like gate on a worst path, so conservatism grows with
        // depth (measured: 2.4e-10 at 120 gates/n=5, 1.4e-7 at 200
        // gates/n=7, against ~1e-15 true error). Still four-plus orders
        // below amplitude scale here; when zero-slack truth is needed,
        // that is the exact module's job.
        assert!(
            max_rad < 1e-5,
            "full-resolution radii stay far below amplitude scale: {max_rad:.3e}"
        );
        assert!(worst_ratio <= 1.0, "containment ratio {worst_ratio}");
    }
}

#[test]
fn ball_midpoints_reproduce_the_complex_simulator_exactly() {
    // Same circuit, Simulator<Ball> vs Simulator<C64>: midpoints are
    // bit-identical (the physics does not change by carrying error bars).
    let ball_circuit = exact_fragment_circuit(6, 150, 6);
    let c64_circuit = retype(&ball_circuit);
    let ball_state = Simulator::<Ball>::new().run(&ball_circuit).unwrap();
    let c64_state = Simulator::<C64>::new().run(&c64_circuit).unwrap();
    for i in 0..(1u64 << 6) {
        assert_eq!(ball_state.amplitude(i).mid, c64_state.amplitude(i));
    }
}

#[test]
fn quantized_gates_stay_certified_and_refine_monotonically() {
    // Coarse-grain the *gates* to a dyadic grid: computation runs at a
    // deliberately reduced resolution, every output still certifies its
    // own error (containment of the exact values), and refining the grid
    // shrinks the certified radii — the resolution dial, measured.
    let n = 4;
    let base = exact_fragment_circuit(n, 60, 7);
    let exact = ExactState::run(&retype(&base)).unwrap();

    // A quantized gate is deliberately non-unitary at the midpoint level
    // (the radius carries the deficit), so it must bypass bind-time
    // unitarity validation: quantize each resolved matrix and apply it to
    // the backend directly.
    let run_quantized = |scale: f64| -> Vec<Ball> {
        let reg = GateRegistry::<Ball>::standard();
        let mut state = DenseState::<Ball>::new(n).unwrap();
        for op in base.ops() {
            if let Op::Named {
                name,
                params,
                qubits,
            } = op
            {
                let m = reg.resolve(name).unwrap().matrix(params).unwrap();
                let d = m.dim();
                let mut data = Vec::with_capacity(d * d);
                for r in 0..d {
                    for c in 0..d {
                        data.push(m.get(r, c).quantize(scale));
                    }
                }
                let quantized = GateMatrix::from_vec(d, data).unwrap();
                state.apply(&quantized, qubits).unwrap();
            }
        }
        (0..(1u64 << n)).map(|i| state.amplitude(i)).collect()
    };

    let coarse = run_quantized(1.0 / 64.0);
    let fine = run_quantized(1.0 / 65536.0);

    let mut coarse_max = 0.0f64;
    let mut fine_max = 0.0f64;
    for i in 0..(1usize << n) {
        let truth = exact.amplitude_c64(i as u64);
        assert!(
            coarse[i].contains(truth),
            "coarse amp {i}: exact {truth} outside ball ({}, r={:.3e})",
            coarse[i].mid,
            coarse[i].rad
        );
        assert!(fine[i].contains(truth), "fine amp {i}");
        coarse_max = coarse_max.max(coarse[i].rad);
        fine_max = fine_max.max(fine[i].rad);
    }
    assert!(
        coarse_max > 10.0 * fine_max,
        "refining the grid must shrink certified radii: coarse {coarse_max:.3e} vs fine {fine_max:.3e}"
    );
    assert!(
        coarse_max > 1e-4,
        "the coarse run is genuinely coarse: {coarse_max:.3e}"
    );
}
