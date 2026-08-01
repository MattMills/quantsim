//! A machine whose next gate is a function of the state it is in.
//!
//! Every other circuit in this crate is a linear map fixed before it
//! runs — which is what makes it compilable, and also what makes its
//! cost a property of the circuit rather than of the state. This gives
//! up the first to get at the second.
//!
//! The loop is three moves: sense the state's own harmonics with a
//! Walsh-Hadamard pair, compute a drive from them, actuate by building
//! one gate across its diagonal. The sensor is H^(x)n, so the second
//! computation runs on the same substrate as the first.
//!
//! The identity that makes it work rather than flail: under a diagonal
//! drive, the derivative of a pure Pauli string is the MIXED string, so
//! the sensor's drill IS the Jacobian. Sensing and differentiating are
//! one transform.

use quantsim::circuit::{BoundGate, GateKernel};
use quantsim::prelude::*;

fn bind(c: &Circuit<C64>, reg: &GateRegistry<C64>) -> Result<Vec<BoundGate<C64>>> {
    Ok(c.bind(reg)?.gates().to_vec())
}

fn layer(n: usize, reg: &GateRegistry<C64>) -> Result<Vec<BoundGate<C64>>> {
    let mut c = Circuit::new(n);
    for q in 0..n { c.gate("h", vec![], vec![q]); }
    for q in (0..n - 1).step_by(2) { c.gate("cx", vec![], vec![q, q + 1]); }
    bind(&c, reg)
}

fn prepared(n: usize, reg: &GateRegistry<C64>, seed: u64) -> Result<DenseState<C64>> {
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for q in 0..n { c.gate("ry", vec![0.3 + 0.1 * (rng.next_u64() % 7) as f64], vec![q]); }
    // A real state sits at a stationary point of every diagonal drive,
    // so the phases are what make the problem non-degenerate.
    for q in 0..n { c.gate("rz", vec![0.2 + 0.13 * (rng.next_u64() % 9) as f64], vec![q]); }
    for q in 0..n - 1 { c.gate("cx", vec![], vec![q, q + 1]); }
    let mut st = DenseState::<C64>::new(n)?;
    for g in bind(&c, reg)? {
        match &g.kernel {
            GateKernel::Matrix(m) => st.apply(m, &g.qubits)?,
            GateKernel::Diagonal(d) => st.apply_diagonal(d, &g.qubits)?,
        }
    }
    Ok(st)
}

fn main() -> Result<()> {
    let reg = GateRegistry::<C64>::standard();
    let n = 8usize;
    let lay = layer(n, &reg)?;
    let tgt = Target::x(0b000_0011, 0.5);
    let events = 40;

    println!("== self-embedded, event-driven evolution ==\n");

    // is the drill really the Jacobian? finite differences say.
    {
        let mut m = Reflexive::from_state(prepared(n, &reg, 1)?);
        let v = m.sense()?;
        let g = Gradient::new(tgt, 0.0);
        let jac = g.jacobian(&v);
        let eps = 1e-6;
        let mut worst = 0.0f64;
        for s in [1u64, 3, 5, 11, 64, 129, 200, 255] {
            let em = Emission { event: 0, coefficients: vec![(s, eps)], fired: 0, error: 0.0, gain: 0.0 };
            let mut probe = Reflexive::from_state(prepared(n, &reg, 1)?);
            probe.apply(&em)?;
            let after = probe.sense()?.x_string(tgt.a);
            let fd = (after - v.x_string(tgt.a)) / eps;
            worst = worst.max((fd - jac[s as usize]).abs());
        }
        println!("== self-embedded, event-driven evolution ==\n");
        println!("── is the drill really the Jacobian? ──\n");
        println!("  d<X_A>/dtheta_S = +-2<X_A Z_S> for |A&S| odd, 0 otherwise.");
        println!("  Checked against finite differences rather than trusted:\n");
        println!("   worst |analytic - numeric| over 8 harmonics: {worst:.3e}");
        println!("   (that is the finite difference's own accuracy at eps=1e-6,");
        println!("   not an error in the derivation)\n");
    }

    println!("── first order, driving along the wrong direction ──\n");
    println!("  Proportional control drives along <X_S>, which is not the");
    println!("  derivative of anything. It never settles at any gain:\n");
    println!("   gain    final |err|   best |err|   settled at   verdict");
    for gain in [0.1f64, 0.3, 0.6, 1.0, 1.6, 2.4] {
        let mut m = Reflexive::from_state(prepared(n, &reg, 1)?);
        let mut l = Proportional::new(tgt, gain);
        let r = m.run(&mut l, &lay, events)?;
        let s = r.settled_at(0.05);
        println!("  {gain:5}  {:11.6} {:12.6}   {:>11}   {}",
            r.final_error(), r.best_error(),
            s.map(|i| i.to_string()).unwrap_or_else(|| "never".into()),
            if s.is_some() { "converged" } else { "oscillates" });
    }

    println!("\n── first order, driving along the measured Jacobian ──\n");
    println!("   gain    final |err|   best |err|   settled at   drive support");
    for gain in [0.05f64, 0.1, 0.2, 0.4, 0.8] {
        let mut m = Reflexive::from_state(prepared(n, &reg, 1)?);
        let mut l = Gradient::new(tgt, gain);
        let r = m.run(&mut l, &lay, events)?;
        let s = r.settled_at(0.05);
        println!("  {gain:5}  {:11.6} {:12.6}   {:>11}   {:>7}",
            r.final_error(), r.best_error(),
            s.map(|i| i.to_string()).unwrap_or_else(|| "never".into()),
            r.trace.last().map(|e| e.coefficients.len()).unwrap_or(0));
    }

    println!("\n  Same loop, same state, same events — only the direction changed.\n");
    println!("── second order: a controller over the controller ──\n");
    println!("  The honest scope: this does NOT beat a well-tuned fixed gain.");
    println!("  What it does is recover from a badly chosen one, and the gain");
    println!("  cannot be chosen in advance because the right value depends on");
    println!("  the state — which is exactly what a reflexive program does not");
    println!("  know before it runs. Read the 0.05 row against the 0.05 row above.\n");
    println!("   start gain  climb  cut    final |err|   settled at   final gain");
    for (g0, climb, cut) in [(0.05f64, 0.2f64, 0.5f64), (0.1, 0.2, 0.5), (0.2, 0.2, 0.5), (0.8, 0.2, 0.5)] {
        let mut m = Reflexive::from_state(prepared(n, &reg, 1)?);
        let mut l = Adaptive::new(Gradient::new(tgt, g0), climb, cut, 4.0);
        let r = m.run(&mut l, &lay, events)?;
        let s = r.settled_at(0.05);
        println!("  {g0:10}  {climb:5}  {cut:5}  {:11.6}   {:>11}   {:10.4}",
            r.final_error(),
            s.map(|i| i.to_string()).unwrap_or_else(|| "never".into()),
            l.gain());
    }

    println!("\n── what the loop costs ──\n");
    println!("  The drive lives on at most half the harmonics: d<X_A>/dtheta_S");
    println!("  vanishes for even |A&S|, which is a parity selection rule and not");
    println!("  a property of the state. Below that half it tracks steerability —");
    println!("  a zero drive means the target cannot be moved from here at all,");
    println!("  which the loop reports by emitting nothing rather than thrashing.\n");
    println!("   state                       fired / 2^n   sparsity    drive support");
    // generic entangled
    let mut m = Reflexive::from_state(prepared(n, &reg, 1)?);
    let v = m.sense()?;
    let mut gl = Gradient::new(tgt, 0.2);
    println!("   generic entangled           {:5} / {:<6} {:7.3}%        {:4} / {}",
        v.fired(1e-12).len(), 1usize<<n, 100.0*v.sparsity(1e-12),
        gl.phases(&v, &v.fired(1e-12), 0).len(), 1usize<<n);
    // GHZ
    let mut c = Circuit::new(n);
    c.gate("h", vec![], vec![0]);
    for q in 1..n { c.gate("cx", vec![], vec![0, q]); }
    let mut st = DenseState::<C64>::new(n)?;
    for g in bind(&c, &reg)? {
        if let GateKernel::Matrix(mm) = &g.kernel { st.apply(mm, &g.qubits)?; }
    }
    let mut mg = Reflexive::from_state(st);
    let vg = mg.sense()?;
    println!("   GHZ                         {:5} / {:<6} {:7.3}%        {:4} / {}",
        vg.fired(1e-12).len(), 1usize<<n, 100.0*vg.sparsity(1e-12),
        gl.phases(&vg, &vg.fired(1e-12), 0).len(), 1usize<<n);
    // product
    let mut c = Circuit::new(n);
    for q in 0..n { c.gate("ry", vec![0.7], vec![q]); }
    let mut st = DenseState::<C64>::new(n)?;
    for g in bind(&c, &reg)? {
        if let GateKernel::Matrix(mm) = &g.kernel { st.apply(mm, &g.qubits)?; }
    }
    let mut mp = Reflexive::from_state(st);
    let vp = mp.sense()?;
    println!("   product                     {:5} / {:<6} {:7.3}%        {:4} / {}",
        vp.fired(1e-12).len(), 1usize<<n, 100.0*vp.sparsity(1e-12),
        gl.phases(&vp, &vp.fired(1e-12), 0).len(), 1usize<<n);

    // invariance under the drive
    let mut m2 = Reflexive::from_state(prepared(n, &reg, 1)?);
    let before = m2.sense()?.fired(1e-12).len();
    let mut l2 = Proportional::new(tgt, 0.6);
    m2.event(&mut l2, 0)?;
    let after = m2.sense()?.fired(1e-12).len();
    println!("\n   fired before drive {before}, after drive {after}");

    println!("\n── is the recorded run a circuit? ──\n");
    println!("  `replay` turns the emitted gates into an ordinary fixed circuit.");
    println!("  From the state the run started in it reproduces the run exactly.");
    println!("  From any other state it does not, because those gates were");
    println!("  computed from a state that is no longer there:\n");
    let mut m = Reflexive::from_state(prepared(n, &reg, 1)?);
    let mut l = Proportional::new(tgt, 0.6);
    let run = m.run(&mut l, &lay, 12)?;
    let same = replay(&run.trace, &lay, prepared(n, &reg, 1)?)?;
    println!("   replayed from its OWN initial state:   deviation {:.3e}", deviation(&same, m.state())?);
    for seed in [2u64, 3, 4] {
        let rep = replay(&run.trace, &lay, prepared(n, &reg, seed)?)?;
        let mut o = Reflexive::from_state(prepared(n, &reg, seed)?);
        let mut ll = Proportional::new(tgt, 0.6);
        o.run(&mut ll, &lay, 12)?;
        println!("   replayed from a DIFFERENT state ({seed}):  deviation {:.3e}",
            deviation(&rep, o.state())?);
    }
    println!("\n── what this is, and is not ──\n");
    println!("  This is measurement-fed adaptive control: sense the register,");
    println!("  compute a drive, apply it. The evolution is NONLINEAR in the");
    println!("  state, as mean-field and feedback-steered dynamics are. It is not");
    println!("  unitary quantum mechanics and it does not simulate unitary quantum");
    println!("  mechanics more cheaply — the sensor reads amplitudes, so the whole");
    println!("  loop is O(2^n). What it buys is that the parameterization is exact");
    println!("  and is computed from the state rather than assumed about it.");
    Ok(())
}
