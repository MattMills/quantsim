//! The reflexive loop with no register at all.
//!
//! `examples/reflexive_loop.rs` holds a dense state vector, because it
//! was written sensor-first and the sensor read amplitudes. That was
//! backwards: a loop whose whole justification is that cost should track
//! the state's structure has no business allocating 2^n amplitudes to
//! find that structure out.
//!
//! Two facts remove the register, and neither is an approximation:
//!
//!   * a diagonal gate with harmonic support k IS k commuting Z-string
//!     rotations, so the actuator is k rotations appended to a program,
//!     not 2^n entries;
//!   * d<O>/dtheta_S = 2i<O.Z_S>, zero unless they anticommute, so
//!     sensing the target and its whole Jacobian is a handful of Pauli
//!     expectations -- each one a backward walk to the vacuum.
//!
//! What it costs is what everything honest in this crate costs: the
//! operator's spread, reported per query, with truncation accounted in a
//! certified L1.

use quantsim::prelude::*;
use quantsim::reflexive::Program;
use quantsim::support::{Support, WideConfig, WidePauli, WideRotation};
use std::time::Instant;

fn zs(qs: &[usize]) -> WidePauli {
    WidePauli { x: Support::empty(), z: qs.iter().copied().collect() }
}
fn xa(qs: &[usize]) -> WidePauli {
    WidePauli { x: qs.iter().copied().collect(), z: Support::empty() }
}

/// A nearest-neighbour program: rzz couplings + rx field, plus the
/// Z-harmonics the drive is allowed to use.
fn mixing(n: usize) -> Vec<WideRotation> {
    let mut v = Vec::new();
    for q in 0..n - 1 { v.push(WideRotation::rzz(q, q + 1, 0.35)); }
    for q in 0..n { v.push(WideRotation::rx(q, 0.42)); }
    for q in 0..n { v.push(WideRotation::rz(q, 0.27)); }
    for q in 0..n { v.push(WideRotation::rx(q, 0.31)); }
    v
}

fn main() -> Result<()> {
    let cfg = WideConfig { threshold: 1e-10, max_terms: Some(20_000) };

    // 1. does the register-free walk give the right expectation?
    println!("── against the dense reference, where dense can still run ──\n");
    println!("      n   observable   register-free        dense     deviation");
    for n in [4usize, 6, 8, 10, 12] {
        let mut p = Program::new(n);
        p.mix(&mixing(n));

        // dense reference
        let mut c = Circuit::new(n);
        for q in 0..n - 1 { c.gate("rzz", vec![0.35], vec![q, q + 1]); }
        for q in 0..n { c.gate("rx", vec![0.42], vec![q]); }
        for q in 0..n { c.gate("rz", vec![0.27], vec![q]); }
        for q in 0..n { c.gate("rx", vec![0.31], vec![q]); }
        let sim: Simulator = Simulator::new();
        let st = sim.run(&c)?;
        for (label, wp, ops) in [
            ("X_0  ", xa(&[0]), vec![(0usize, Pauli::X)]),
            ("X_01 ", xa(&[0, 1]), vec![(0, Pauli::X), (1, Pauli::X)]),
            ("Y_0  ", WidePauli { x: [0usize].into_iter().collect(), z: [0usize].into_iter().collect() }, vec![(0, Pauli::Y)]),
            ("Z_0  ", zs(&[0]), vec![(0, Pauli::Z)]),
            ("Y_0Z_1", WidePauli { x: [0usize].into_iter().collect(), z: [0usize, 1].into_iter().collect() }, vec![(0, Pauli::Y), (1, Pauli::Z)]),
        ] {
            let (got, _) = p.expectation(&wp, &cfg);
            let want = pauli_expectation(&*st, &ops)?;
            println!("  {n:5}   {label}     {got:14.10} {:12.10}  {:.3e}", want.re, (got - want.re).abs());
        }
    }

    // is the analytic Jacobian the real derivative? finite differences.
    {
        let n = 8usize;
        println!("\n── the Jacobian against finite differences ──\n");
        println!("   target      worst |analytic - numeric|");
        for (label, t) in [("X_0  ", xa(&[0])), ("X_01 ", xa(&[0, 1])),
                           ("Y_0Z_1", WidePauli { x: [0usize].into_iter().collect(),
                                                  z: [0usize, 1].into_iter().collect() })] {
            let mut p = Program::new(n);
            p.mix(&mixing(n));
            let (base, _) = p.expectation(&t, &cfg);
            let cands = p.harmonics();
            let (jac, _) = p.jacobian(&t, &cands, &cfg);
            let eps = 1e-6;
            let mut worst = 0.0f64;
            for (sup, g) in &jac {
                let mut q = Program::new(n);
                q.mix(&mixing(n));
                q.drive(&[(sup.clone(), eps)]);
                let (after, _) = q.expectation(&t, &cfg);
                worst = worst.max(((after - base) / eps - g).abs());
            }
            println!("   {label}      {worst:.3e}");
        }
    }

    // which targets are actually steerable here?
    {
        let n = 10usize;
        println!("\n── which targets are steerable? ──\n");
        println!("   target        value      live Jacobian entries   |grad|");
        for (label, t) in [("X_0", xa(&[0])), ("X_01", xa(&[0,1])), ("X_012", xa(&[0,1,2])),
                           ("Y_0", WidePauli{x: [0usize].into_iter().collect(), z: [0usize].into_iter().collect()}),
                           ("X_0 Z_1", WidePauli{x: [0usize].into_iter().collect(), z: [1usize].into_iter().collect()})] {
            let mut p = Program::new(n);
            p.mix(&mixing(n));
            let (v, _) = p.expectation(&t, &cfg);
            let cands = p.harmonics();
            let (j, _) = p.jacobian(&t, &cands, &cfg);
            let nz = j.iter().filter(|(_, g)| g.abs() > 1e-12).count();
            let norm: f64 = j.iter().map(|(_, g)| g * g).sum::<f64>().sqrt();
            println!("   {label:12} {v:9.6}   {nz:20}   {norm:.6}");
        }
    }

    // 2. the loop, register-free
    println!("\n── the loop with no register ──\n");
    for (label, remix) in [("prepare once, then drive", false), ("re-mix every event", true)] {
        println!("   {label}\n");
        println!("   event     error   rotations   peak terms   discarded L1   walks");
        let n = 12usize;
        let mut p = Program::new(n);
        p.mix(&mixing(n));
        let target = xa(&[0, 1]);
        for e in 0..10 {
            let (err, cost) = p.event(&target, 0.5, 0.4, &cfg);
            if remix { p.mix(&mixing(n)); }
            println!("  {e:6} {err:9.6} {:11} {:12} {:14.3e} {:7}",
                p.len(), cost.peak_terms, cost.discarded_l1, cost.walks);
        }
        println!();
    }

    // 3. past every register
    println!("\n── and where no register exists ──\n");
    println!("      n   qubits built   events   final error   peak terms   discarded L1        time");
    for n in [64usize, 256, 1024, 4096, 16384] {
        let mut p = Program::new(n);
        p.mix(&mixing(n));
        let target = xa(&[n / 2, n / 2 + 1]);
        let t0 = Instant::now();
        let mut err = 0.0;
        let mut cost = quantsim::reflexive::Cost::default();
        for _ in 0..8 {
            let (e, c) = p.event(&target, 0.5, 0.4, &cfg);
            err = e;
            cost = c;
        }
        println!("  {n:5} {:14} {:8} {err:13.6} {:12} {:14.3e} {:>11.1?}",
            0, 8, cost.peak_terms, cost.discarded_l1, t0.elapsed());
    }
    let _ = zs(&[0]);
    Ok(())
}
