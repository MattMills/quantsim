//! Representations defined by what they ANSWER, not by what they store.
//!
//! `Backend` requires `amplitude(index)`, so every representation
//! registered against it must be able to write itself out in the `2^n`
//! computational basis — which makes the exponential object the ground
//! truth and every alternative a compression of it. A `Resolver` is
//! asked only to answer questions, and reports what answering cost.

use quantsim::heisenberg::tfim_trotter;
use quantsim::prelude::*;
use quantsim::query::*;

fn brickwork(n: usize, depth: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for d in 0..depth {
        for q in (d % 2..n.saturating_sub(1)).step_by(2) {
            c.gate("rzz", vec![0.31 + 0.01 * d as f64], vec![q, q + 1]);
            c.gate("rx", vec![0.47], vec![q]);
            c.gate("rx", vec![0.41], vec![q + 1]);
        }
    }
    c
}

fn main() -> Result<()> {
    let reg = GateRegistry::<C64>::standard();
    println!("== a query costs what the question needs, not what the register is ==\n");
    println!("  Nothing runs until a question arrives. Then the query's backward cone");
    println!("  is relabelled onto a compact register and only that is simulated.");
    println!("  (causal_diamond already pruned the GATES outside the cone, but left a");
    println!("  circuit n qubits wide — so the simulation was still 2^n. The relabel");
    println!("  is the half that was missing.)\n");
    println!("  n   depth  cone   full dim         simulated       compression   <Z_mid>");
    for &(n, depth) in &[(12usize, 4usize), (20, 4), (30, 4), (40, 4), (40, 8), (48, 6), (64, 6)] {
        let c = brickwork(n, depth);
        let mut cone = ConeResolver::new(c.clone(), &reg, Inner::Dense);
        let ops = [(n / 2, Pauli::Z)];
        let a = cone.expectation(&ops)?;
        println!("{n:4} {depth:6} {:6} {:16} {:15} {:11.0}x  {:+.9}",
            a.cost.qubits, a.cost.full_dim(), a.cost.dim(), a.cost.compression(), a.value);
        if n <= 20 {
            let mut full = StateResolver::new(c, &reg, Inner::Dense);
            assert!((a.value - full.expectation(&ops)?.value).abs() < 1e-12);
        }
    }

    println!("\n  Answers cross-checked against the full-width state wherever that");
    println!("  is computable at all. At n=64 it is not, and the answer is the same");
    println!("  4096-dimensional calculation it was at n=12.\n");
    println!("== three resolvers, one question, three currencies ==\n");
    println!("  n=20, TFIM depth 5, <Z_10>:");
    let n = 20usize;
    let rots = tfim_trotter(n, 1.0, 0.7, 0.35, 5);
    let mut c = Circuit::new(n);
    for r in &rots {
        let s = r.support();
        if s.len() == 1 {
            if r.axis.0 != 0 { c.gate("rx", vec![r.theta], vec![s[0]]); }
            else { c.gate("rz", vec![r.theta], vec![s[0]]); }
        } else {
            c.gate("rzz", vec![r.theta], vec![s[0], s[1]]);
        }
    }
    let ops = [(n / 2, Pauli::Z)];
    let mut full = StateResolver::new(c.clone(), &reg, Inner::Dense);
    let mut cone = ConeResolver::new(c, &reg, Inner::Dense);
    let mut heis = HeisenbergResolver::new(rots, n);
    println!("    resolver      value            qubits   bytes");
    for (name, a) in [("state", full.expectation(&ops)?), ("cone", cone.expectation(&ops)?),
                      ("heisenberg", heis.expectation(&ops)?)] {
        println!("    {name:<12}  {:+.12}  {:6}  {:8}", a.value, a.cost.qubits, a.cost.bytes);
    }
    println!("\n  The Heisenberg resolver instantiates ZERO qubits — not few, zero.");
    println!("  It has no amplitudes to hand out, no support to enumerate and no");
    println!("  basis to load into, so no refactoring of `Backend` would admit it.");
    println!("  It answers anyway, which is the whole argument for the trait.\n");

    println!("== and where locality stops paying ==\n");
    let n = 12usize;
    let mut a2a = Circuit::new(n);
    for d in 0..2 {
        for a in 0..n { for b in (a + 1)..n {
            a2a.gate("rzz", vec![0.21 + 0.01 * d as f64], vec![a, b]);
        }}
        for q in 0..n { a2a.gate("rx", vec![0.37], vec![q]); }
    }
    let mut cone = ConeResolver::new(a2a, &reg, Inner::Dense);
    let a = cone.expectation(&[(6, Pauli::Z)])?;
    println!("  all-to-all, n={n}: cone {} qubits, compression {:.0}x", a.cost.qubits, a.cost.compression());
    println!("\n  One all-to-all layer puts every qubit in every cone, so there is no");
    println!("  outside left to cut and the resolver pays the full width. Locality is");
    println!("  the resource; the report says so rather than implying otherwise.");
    Ok(())
}
