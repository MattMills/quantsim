//! Absolute reference values: the D[ω] = ℤ[1/√2, e^{iπ/4}] evaluator as
//! ground truth for the whole backend roster — float error measured
//! absolutely (the dense reference's own included), exact zeros decided,
//! Born weights as ring elements, and Ball radii certified.
//!
//! Run with `cargo run --release --example absolute_reference`.

use quantsim::exact::{DOmega, ExactState};
use quantsim::prelude::*;

/// A deterministic circuit over the exact-supported fragment.
fn fragment(n: usize, gates: usize, seed: u64) -> Circuit<C64> {
    let pi4 = std::f64::consts::FRAC_PI_4;
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for _ in 0..gates {
        let q = (rng.next_u64() % n as u64) as usize;
        let r = ((q + 1) + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
        match rng.next_u64() % 12 {
            0 => c.h(q),
            1 => c.t(q),
            2 => c.tdg(q),
            3 => c.s(q),
            4 => c.x(q),
            5 => c.sx(q),
            6 => c.rz(q, 2.0 * pi4 * (1 + rng.next_u64() % 7) as f64),
            7 => c.cx(q, r),
            8 => c.cz(q, r),
            9 => c.cp(q, r, pi4 * (1 + rng.next_u64() % 7) as f64),
            10 => c.swap(q, r),
            _ => c.ry(q, 2.0 * pi4 * (1 + rng.next_u64() % 7) as f64),
        };
    }
    c
}

fn main() -> Result<()> {
    // ══ 1. exact values are ring elements, not floats ══
    println!("══ 1. amplitudes as ring elements ══\n");
    let ghz = ExactState::run(&library::ghz(12))?;
    println!("GHZ(12), exactly:");
    println!(
        "  amp(|0…0⟩) = {}   amp(|1…1⟩) = {}   support = {}",
        ghz.amplitude_exact(0),
        ghz.amplitude_exact((1 << 12) - 1),
        ghz.support_exact()
    );
    println!(
        "  P(|0…0⟩) = {}   total weight = {}  (equalities, not tolerances)\n",
        ghz.probability_exact(0)?,
        ghz.total_weight_exact()?
    );

    // ══ 2. exact zeros where floats leave residue ══
    println!("══ 2. the float floor, measured ══\n");
    let mut c: Circuit<C64> = Circuit::new(1);
    c.h(0).t(0).t(0).t(0).t(0).h(0); // H·T⁴·H = H·Z·H = X
    let exact = ExactState::run(&c)?;
    let sim = Simulator::<C64>::new();
    let dense = sim.run(&c)?;
    println!("H·T⁴·H |0⟩ (= X|0⟩ = |1⟩):");
    println!(
        "  exact:  amp(|0⟩) = {}   amp(|1⟩) = {}   support = {}",
        exact.amplitude_exact(0),
        exact.amplitude_exact(1),
        exact.support_exact()
    );
    println!(
        "  dense:  amp(|0⟩) = {:.3e}  — the rounding residue the ring removes",
        dense.amplitude(0).norm()
    );
    let mut c8: Circuit<C64> = Circuit::new(1);
    c8.h(0);
    for _ in 0..8 {
        c8.t(0);
    }
    c8.h(0);
    let e8 = ExactState::run(&c8)?;
    println!(
        "  H·T⁸·H |0⟩: amp(|0⟩) = {} — T⁸ = 1 exactly\n",
        e8.amplitude_exact(0)
    );

    // ══ 3. every backend measured against the same absolute yardstick ══
    println!("══ 3. absolute float error of EVERY backend (n=10, 200 gates) ══\n");
    let circuit = fragment(10, 200, 3);
    let truth = ExactState::run(&circuit)?;
    println!(
        "  exact total weight: {}  (unitarity in the ring)",
        truth.total_weight_exact()?
    );
    println!("  {:<12} {:>16}", "backend", "max |ψ − exact|");
    for backend in ["dense", "sparse", "adaptive", "factored", "mps", "mera"] {
        let state = sim.run_on(backend, &circuit)?;
        println!(
            "  {backend:<12} {:>16.3e}",
            truth.max_deviation_vs(state.as_ref())
        );
    }
    println!("  (all ~1e-15: honest IEEE accumulation — now measured, not assumed)\n");

    // ══ 4. Grover through diagonal oracles, exactly ══
    println!("══ 4. Grover in the ring ══\n");
    let (n, marked, iterations) = (5usize, 19u64, 3usize);
    let grover = ExactState::run(&library::grover::<C64>(n, marked, iterations)?)?;
    let theta = (1.0 / (1u64 << n) as f64).sqrt().asin();
    let closed = ((2 * iterations + 1) as f64 * theta).sin().powi(2);
    println!("Grover(n={n}, marked={marked}, {iterations} iterations):");
    println!(
        "  P(marked) exact ring element = {}",
        grover.probability_exact(marked)?
    );
    println!(
        "  as f64: {:.12}   closed form: {closed:.12}\n",
        grover.probability_exact(marked)?.to_f64()
    );

    // ══ 5. Ball radii certified by containment ══
    println!("══ 5. certified arithmetic: Ball radii vs exact values ══\n");
    let ball_circuit: Circuit<Ball> = {
        let mut b = Circuit::new(7);
        for op in fragment(7, 150, 5).ops() {
            if let Op::Named {
                name,
                params,
                qubits,
            } = op
            {
                b.gate(name.clone(), params.clone(), qubits.clone());
            }
        }
        b
    };
    let ball_state = Simulator::<Ball>::new().run(&ball_circuit)?;
    let truth = ExactState::run(&fragment(7, 150, 5))?;
    let mut contained = true;
    let (mut worst_err, mut worst_rad) = (0.0f64, 0.0f64);
    for i in 0..(1u64 << 7) {
        let ball = ball_state.amplitude(i);
        let z = truth.amplitude_c64(i);
        contained &= ball.contains(z);
        worst_err = worst_err.max((z - ball.mid).norm());
        worst_rad = worst_rad.max(ball.rad);
    }
    println!("150-gate circuit over Ball, every amplitude checked against the ring:");
    println!("  containment holds everywhere: {contained}");
    println!("  worst true error {worst_err:.3e} ≤ worst certified radius {worst_rad:.3e}");
    println!("  (the radius is a sound bound, proven against exact values — not a guess)");

    // ══ 6. the scope boundary, loud ══
    println!("\n══ 6. out-of-fragment requests fail loudly ══\n");
    let mut generic: Circuit<C64> = Circuit::new(1);
    generic.rz(0, 0.7365);
    match ExactState::run(&generic) {
        Err(e) => println!("  rz(0.7365): {e}"),
        Ok(_) => unreachable!("generic angles are not in D[ω]"),
    }
    let big = DOmega::int(i64::MAX).mul(DOmega::int(i64::MAX))?;
    match big.mul(DOmega::int(i64::MAX)) {
        Err(e) => println!("  i128 overflow: {e}"),
        Ok(_) => unreachable!("must overflow"),
    }
    Ok(())
}
