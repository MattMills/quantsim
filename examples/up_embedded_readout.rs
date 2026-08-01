//! Up-embedded readout: what it costs to answer a question about a
//! circuit whose cone has nothing left to cut.
//!
//! `examples/query_resolvers.rs` ends on an honest failure: one
//! all-to-all layer puts every qubit in every cone, so the cone resolver
//! simulates the whole register and reports `1.0×` compression. Locality
//! was the resource, and there wasn't any.
//!
//! This runnable takes the other move. Each magic event is relocated to
//! a fresh wire, after which the whole dynamics is Clifford and every
//! Pauli line transports to exactly one line — nonlocal circuits
//! included. What readout costs is then the observable's *magic
//! adjacency*, which does not know how wide the register is.

use std::time::Instant;

use quantsim::gates::Pauli;
use quantsim::prelude::*;
use quantsim::query::{ConeResolver, Inner, Resolver, StateResolver};
use quantsim::upembed::{self, UpEmbedResolver};

/// Every pair coupled in every layer, with magic interleaved — the shape
/// with no locality left for a cone to cut.
fn all_to_all(n: usize, t: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.gate("h", vec![], vec![q]);
    }
    for k in 0..t / 2 {
        c.gate("t", vec![], vec![(k * 3) % n]);
    }
    for a in 0..n {
        for b in (a + 1)..n {
            c.gate("cz", vec![], vec![a, b]);
        }
    }
    for k in 0..t - t / 2 {
        c.gate("t", vec![], vec![(k * 5 + 1) % n]);
    }
    for a in 0..n {
        for b in (a + 1)..n {
            c.gate("cz", vec![], vec![a, b]);
        }
    }
    for q in 0..n {
        c.gate("h", vec![], vec![q]);
    }
    c
}

/// Independent neighbourhoods, each with its own magic.
fn blocked(n: usize, block: usize, t_per_block: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for lo in (0..n).step_by(block) {
        let hi = (lo + block).min(n);
        for _ in 0..3 {
            for a in lo..hi {
                for b in (a + 1)..hi {
                    c.gate("cz", vec![], vec![a, b]);
                }
            }
            for q in lo..hi {
                c.gate("h", vec![], vec![q]);
            }
        }
        for k in 0..t_per_block {
            c.gate("t", vec![], vec![lo + k % (hi - lo)]);
        }
    }
    c
}

/// Nearest-neighbour brickwork with magic sprinkled through it.
fn brickwork(n: usize, depth: usize, t_per_layer: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for d in 0..depth {
        for q in (d % 2..n.saturating_sub(1)).step_by(2) {
            c.gate("cx", vec![], vec![q, q + 1]);
            c.gate("h", vec![], vec![q]);
            c.gate("s", vec![], vec![q + 1]);
        }
        for k in 0..t_per_layer {
            c.gate("t", vec![], vec![(k * 7 + d) % n]);
        }
    }
    c
}

fn main() -> Result<()> {
    println!("== dis-entangling magic into a higher dimension ==\n");

    // ── it is the same expectation ───────────────────────────────────
    println!("── first, that it is the same answer ──\n");
    println!("  Against the dense backend, over random Clifford+T words drawn");
    println!("  from h/s/sdg/x/y/z/sx/cx/cz/t/tdg, on every Pauli axis.\n");
    let sim: Simulator = Simulator::new();
    let mut worst = 0.0f64;
    let mut checks = 0usize;
    for seed in 0..60u64 {
        for n in 2..=4usize {
            let mut rng = Prng::new(seed * 13 + n as u64);
            let mut c = Circuit::new(n);
            for _ in 0..12 {
                let a = (rng.next_u64() % n as u64) as usize;
                let b = (rng.next_u64() % n as u64) as usize;
                match rng.next_u64() % 12 {
                    0 => c.gate("h", vec![], vec![a]),
                    1 => c.gate("s", vec![], vec![a]),
                    2 => c.gate("sdg", vec![], vec![a]),
                    3 => c.gate("x", vec![], vec![a]),
                    4 => c.gate("y", vec![], vec![a]),
                    5 => c.gate("z", vec![], vec![a]),
                    6 => c.gate("sx", vec![], vec![a]),
                    7 | 8 if a != b => c.gate("cx", vec![], vec![a, b]),
                    9 if a != b => c.gate("cz", vec![], vec![a, b]),
                    10 => c.gate("tdg", vec![], vec![a]),
                    _ => c.gate("t", vec![], vec![a]),
                };
            }
            let state = sim.run(&c)?;
            for q in 0..n {
                for p in [Pauli::X, Pauli::Y, Pauli::Z] {
                    let want = pauli_expectation(&*state, &[(q, p)])?;
                    worst = worst.max((want.re - upembed::expectation(&c, &[(q, p)])?.value).abs());
                    checks += 1;
                }
            }
        }
    }
    println!("    {checks} expectations, worst deviation {worst:.3e}");
    println!("    (and that is the ℤ[ω]/√2^k → f64 conversion; the readout itself");
    println!("    has no floating point in it — a vanishing expectation comes back");
    println!("    as the ring's zero, decided rather than rounded)\n");

    // ── where the cone stops and this does not ───────────────────────
    println!("── the boundary the cone resolver hits ──\n");
    println!("  One all-to-all layer and every qubit is in every cone. The cone");
    println!("  resolver is not wrong about that — it is exactly right, and what");
    println!("  it is right about is that the frame has nothing to offer:\n");
    let reg = GateRegistry::<C64>::standard();
    println!("      n    t   cone qubits   cone gain   up-embed exponent   up-embed gain");
    for &(n, t) in &[(8usize, 6usize), (10, 6), (12, 6), (14, 6)] {
        let c = all_to_all(n, t);
        let ops = [(n / 2, Pauli::Z)];
        let mut cone = ConeResolver::new(c.clone(), &reg, Inner::Dense);
        let a = cone.expectation(&ops)?;
        let mut up = UpEmbedResolver::new(c.clone());
        let b = up.expectation(&ops)?;
        let mut state = StateResolver::new(c, &reg, Inner::Dense);
        let truth = state.expectation(&ops)?;
        assert!((b.value - truth.value).abs() < 1e-11);
        println!(
            "  {n:5} {t:4} {:13} {:11.1}x {:19} {:14.0}x",
            a.cost.qubits,
            a.cost.compression(),
            b.cost.qubits,
            b.cost.compression()
        );
    }
    println!("\n  Same circuits, same questions, same answers. The cone pays the");
    println!("  full width because there is no outside; the up-embedding pays the");
    println!("  observable's magic adjacency, which is a property of where the T");
    println!("  gates sit rather than of how many wires there are.\n");

    // ── and the width genuinely stops mattering ──────────────────────
    println!("── holding the magic fixed and growing the register ──\n");
    println!("      n   wires   clusters              terms   transports        time");
    for n in [16usize, 64, 256, 1024, 4096] {
        let c = all_to_all(n, 6);
        let t0 = Instant::now();
        let r = upembed::expectation(&c, &[(n / 2, Pauli::Z)])?;
        println!(
            "  {n:5} {:7}   {:<20} {:5} {:12}   {:>9.1?}",
            upembed::gadgetize(&c)?.wires(),
            format!("{:?}", r.clusters),
            r.terms,
            r.transports,
            t0.elapsed()
        );
    }
    println!("\n  The register grows 256-fold and the cost does not move. At n = 4096");
    println!("  the state vector has 2^4096 amplitudes, which is not a large number");
    println!("  so much as a number with no physical referent; the readout is still");
    println!("  13 boundary contractions and 7 transports.\n");
    println!("  The wall-clock does grow, and it is worth being exact about why: a");
    println!("  transport walks the circuit once, and an all-to-all circuit on n");
    println!("  qubits *has* n^2 gates. So the time is linear in the circuit and");
    println!("  quadratic in n only because the circuit is. Nothing exponential is");
    println!("  happening — the exponent is the `terms` column, and it is flat.\n");
    println!("  There is no width ceiling in this module, deliberately. The crate's");
    println!("  Pauli strings are u64 and cap a register at 64 wires, which is a");
    println!("  perfectly reasonable limit for a representation whose cost is 2^n");
    println!("  and an indefensible one for a representation whose whole claim is");
    println!("  that cost has stopped tracking n. The transport here runs on");
    println!("  unbounded bitsets and is pinned against the crate's u64 version");
    println!("  wherever both can run.\n");

    // ── what the factorization is worth ──────────────────────────────
    println!("── the subset sum, and why it splits ──\n");
    println!("  The sum ranges over all 2^t ancilla subsets. But the boundary state");
    println!("  is a product state, so the contraction factorizes wire by wire —");
    println!("  and therefore over the connected components of the transported");
    println!("  lines. Magic in separate neighbourhoods costs a product of small");
    println!("  sums instead of one large one:\n");
    println!("      n   block    t   clusters                    2^t     terms        gain");
    for &(n, b, tb) in &[
        (24usize, 6usize, 3usize),
        (32, 8, 3),
        (48, 8, 4),
        (64, 8, 4),
        (96, 8, 5),
    ] {
        let c = blocked(n, b, tb);
        let r = upembed::expectation(&c, &[(b / 2, Pauli::Z)])?;
        let t: usize = r.clusters.iter().sum();
        println!(
            "  {n:5} {b:7} {t:4}   {:<24} {:9.2e} {:7}   {:9.3e}x",
            format!("{:?}", &r.clusters[..r.clusters.len().min(6)]),
            2f64.powi(t as i32),
            r.terms,
            r.factorization_gain()
        );
    }
    println!("\n  And where the magic is *not* separable, the report says so rather");
    println!("  than implying otherwise — a brickwork circuit with T gates spread");
    println!("  through every layer builds one cluster and pays for it:\n");
    println!("      n   depth    t   clusters              terms        time");
    for &(n, d, tl) in &[
        (32usize, 8usize, 1usize),
        (32, 8, 2),
        (48, 10, 1),
        (48, 10, 2),
    ] {
        let c = brickwork(n, d, tl);
        let t0 = Instant::now();
        let r = upembed::expectation(&c, &[(n / 2, Pauli::Z)])?;
        let t: usize = r.clusters.iter().sum();
        println!(
            "  {n:5} {d:7} {t:4}   {:<20} {:6}   {:>9.1?}",
            format!("{:?}", &r.clusters[..r.clusters.len().min(5)]),
            r.terms,
            t0.elapsed()
        );
    }
    println!("\n  Locality is not what this frame rewards; *separability of the");
    println!("  magic* is. Those are different properties, and a circuit can have");
    println!("  either without the other. That is the honest statement of what was");
    println!("  bought: not a free lunch, a different resource — and one the report");
    println!("  measures rather than assumes.\n");

    // ── the edge ─────────────────────────────────────────────────────
    println!("── where the fragment ends ──\n");
    for (label, build) in [
        ("ccz          ", 0usize),
        ("rz(0.3)      ", 1),
        ("rz(pi/8)     ", 2),
        ("rz(pi/4) = T ", 3),
    ] {
        let mut c = Circuit::new(3);
        c.gate("h", vec![], vec![0]);
        match build {
            0 => c.gate("ccz", vec![], vec![0, 1, 2]),
            1 => c.gate("rz", vec![0.3], vec![0]),
            2 => c.gate("rz", vec![std::f64::consts::FRAC_PI_8], vec![0]),
            _ => c.gate("rz", vec![std::f64::consts::FRAC_PI_4], vec![0]),
        };
        match upembed::gadgetize(&c) {
            Ok(e) => println!(
                "    {label} accepted, {} magic event{}",
                e.magic(),
                if e.magic() == 1 { "" } else { "s" }
            ),
            Err(e) => println!("    {label} refused: {e}"),
        }
    }
    println!("\n  The construction rests on the dynamics being *exactly* Clifford");
    println!("  after gadgetization, so a rotation that is only nearly T would");
    println!("  silently void it. Those are refused by name rather than rounded.");
    Ok(())
}
