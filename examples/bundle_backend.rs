//! The polarity bundle as a **registered backend**: Clifford gate action
//! on the graph description itself, priced against dense.
//!
//! Run with `cargo run --release --example bundle_backend`.

use quantsim::bundle::{vop, PolarityBundle};
use quantsim::prelude::*;
use quantsim::rng::Prng;

fn dev_up_to_phase(a: &dyn Backend<C64>, b: &dyn Backend<C64>) -> f64 {
    let mut anchor = (0u64, 0.0f64);
    a.for_each_nonzero(&mut |i, z| {
        if z.norm() > anchor.1 {
            anchor = (i, z.norm());
        }
    });
    if anchor.1 == 0.0 {
        return max_amplitude_deviation(a, b);
    }
    let y = b.amplitude(anchor.0);
    if y.norm() < 1e-12 {
        return f64::INFINITY;
    }
    let phase = a.amplitude(anchor.0) / y;
    let mut dev = 0.0f64;
    a.for_each_nonzero(&mut |i, z| dev = dev.max((z - b.amplitude(i) * phase).norm()));
    b.for_each_nonzero(&mut |i, z| dev = dev.max((a.amplitude(i) - z * phase).norm()));
    dev
}

fn main() -> Result<()> {
    let sim: Simulator = Simulator::new();

    // ── 1. It is in the registry ──────────────────────────────────────
    println!("== the backend registry ==");
    println!("  {:?}", sim.backends().names());
    println!("  so `sim.run_on(\"bundle\", &circuit)` works like any other representation");
    let fresh = sim.backends().create("bundle", 4)?;
    println!(
        "  a fresh one is |0…0⟩ as the Backend contract requires: p(0) = {:.6}",
        fresh.probability(0)
    );

    // ── 2. The vertex operators ───────────────────────────────────────
    println!();
    println!("== what a fiber's frame becomes once gates run ==");
    println!("  the single-qubit Clifford group, generated from H and S by closure:");
    println!("    order {}", vop::count());
    println!(
        "    H∘H = identity: {}   S⁴ = identity: {}",
        vop::compose(vop::hadamard(), vop::hadamard()) == vop::IDENTITY,
        {
            let s = vop::phase();
            vop::compose(vop::compose(s, s), vop::compose(s, s)) == vop::IDENTITY
        }
    );
    println!(
        "    diagonal elements (the ones a cz commutes with): {}",
        (0..vop::count() as u8)
            .filter(|&v| vop::is_diagonal(v))
            .count()
    );

    // ── 3. Local complementation ──────────────────────────────────────
    println!();
    println!("== local complementation: the description moves, the state does not ==");
    let mut bundle = PolarityBundle::new(5)?;
    for (a, b) in [(0u32, 1u32), (1, 2), (2, 3), (3, 4), (0, 2)] {
        bundle.link(a, b)?;
    }
    bundle.apply_vop(1, vop::hadamard())?;
    let before = bundle.to_state()?;
    let signature = bundle.link_signature();
    let listing = |b: &PolarityBundle| -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for a in 0..b.sites() as u32 {
            for &c in b.neighbours(a) {
                if a < c {
                    out.push((a, c));
                }
            }
        }
        out
    };
    let links_before = listing(&bundle);
    bundle.local_complement(2)?;
    let after = bundle.to_state()?;
    println!("  links before: {links_before:?}");
    println!("  links after : {:?}", listing(&bundle));
    println!(
        "  link signature changed: {}   state deviation: {:.2e}",
        signature != bundle.link_signature(),
        dev_up_to_phase(before.as_ref(), after.as_ref())
    );

    // ── 4. Against dense, gate for gate ───────────────────────────────
    println!();
    println!("== random Clifford circuits: bundle vs dense ==");
    let registry = GateRegistry::<C64>::standard();
    let clifford = ["h", "s", "sdg", "x", "y", "z", "cx", "cz", "swap"];
    let (mut ran, mut refused, mut worst) = (0usize, 0usize, 0.0f64);
    for seed in 0..40u64 {
        let n = 5;
        let mut rng = Prng::new(seed);
        let mut circuit: Circuit = Circuit::new(n);
        for _ in 0..30 {
            let name = clifford[(rng.next_f64() * clifford.len() as f64) as usize % clifford.len()];
            let arity = registry.resolve(name)?.arity();
            let a = (rng.next_f64() * n as f64) as usize % n;
            if arity == 1 {
                circuit.gate(name, Vec::new(), vec![a]);
            } else {
                let mut b = (rng.next_f64() * n as f64) as usize % n;
                if b == a {
                    b = (a + 1) % n;
                }
                circuit.gate(name, Vec::new(), vec![a, b]);
            }
        }
        let dense = sim.run_on("dense", &circuit)?;
        match sim.run_on("bundle", &circuit) {
            Ok(state) => {
                ran += 1;
                worst = worst.max(dev_up_to_phase(state.as_ref(), dense.as_ref()));
            }
            Err(_) => refused += 1,
        }
    }
    println!("  40 circuits of 30 gates on 5 qubits:");
    println!("    ran {ran}, refused {refused}");
    println!("    worst deviation over the ones that ran: {worst:.2e}");
    println!("  the refusals are the vertex-operator reduction not converging when the two cz");
    println!("  endpoints are each other's only handle. They refuse BY NAME; they never return");
    println!("  a wrong state, and the incompleteness is a pinned test assertion.");

    // ── 5. Named circuits, and the footprint ──────────────────────────
    println!();
    println!("== named circuits, and what the description costs ==");
    println!(
        "  {:>4} {:>12} {:>14} {:>14} {:>10}",
        "n", "deviation", "bundle bytes", "dense bytes", "dense/bundle"
    );
    for n in [4usize, 8, 12, 16, 20] {
        let ghz = library::ghz(n);
        let bundle = sim.run_on("bundle", &ghz)?;
        let dense = sim.run_on("dense", &ghz)?;
        println!(
            "  {:>4} {:>12.2e} {:>14} {:>14} {:>9.2}x",
            n,
            dev_up_to_phase(bundle.as_ref(), dense.as_ref()),
            bundle.memory_bytes(),
            dense.memory_bytes(),
            dense.memory_bytes() as f64 / bundle.memory_bytes() as f64
        );
    }
    println!("  Below n = 12 the bundle costs MORE, because its footprint includes the journal");
    println!("  of every gate applied and dense's does not. The crossover is where 2^n starts");
    println!("  to matter; checkpoint() drops the journal if the audit trail is not wanted.");

    // ── 6. What it refuses ────────────────────────────────────────────
    println!();
    println!("== the refusals are by name, never by silence ==");
    let mut non_clifford: Circuit = Circuit::new(3);
    non_clifford.h(0).t(0);
    println!(
        "  a T gate: {}",
        sim.run_on("bundle", &non_clifford)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    );
    let mut b = PolarityBundle::new(3)?;
    println!(
        "  loading amplitudes: {}",
        Backend::<C64>::load(&mut b, &[(0u64, C64::new(1.0, 0.0))])
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    );
    let wide = sim.backends().create("bundle", 40)?;
    println!(
        "  a 40-site bundle exists ({} bytes) — holding the state is O(n + |E|) —",
        wide.memory_bytes()
    );
    println!(
        "  but there are 2^40 amplitudes to read, so enumeration reports none: \
         nonzero_count = {}",
        wide.nonzero_count()
    );
    println!("  the description has no width limit; reading amplitudes out of it does.");
    Ok(())
}
