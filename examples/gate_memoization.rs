//! Progressive gate-result memoization: qubit operations as `n`-wide
//! operation objects, the computational geometry memoized away into a
//! shared entry table, and a journal that unwinds and rewinds the state so
//! variants can be re-explored over a shared prefix.
//!
//! Run with `cargo run --release --example gate_memoization`.

use quantsim::memo::{explore, Explorer, MemoConfig, MemoPlan};
use quantsim::prelude::*;

fn brickwork(n: usize, depth: usize) -> Circuit {
    let mut c = Circuit::new(n);
    for layer in 0..depth {
        for q in (layer % 2..n.saturating_sub(1)).step_by(2) {
            c.h(q);
            c.cx(q, q + 1);
            c.gate("rz", vec![0.7], vec![q + 1]);
        }
    }
    c
}

fn main() -> Result<()> {
    let sim = Simulator::<C64>::new();

    println!("== Plan statistics ==");
    println!(
        "  {:<22} {:>6} {:>6} {:>8} {:>8} {:>7} {:>7} {:>10}",
        "circuit", "gates", "ops", "entries", "reuse", "hit%", "fusion", "entry B"
    );
    let cases: Vec<(String, Circuit)> = vec![
        ("ghz-12".into(), library::ghz(12)),
        ("qft-8".into(), library::qft(8)),
        ("brickwork-10x6".into(), brickwork(10, 6)),
        ("rainbow-10".into(), library::rainbow(10)),
        (
            "random-10x120".into(),
            library::random_circuit(10, 120, 0xABC),
        ),
    ];
    for (name, circuit) in &cases {
        let plan = MemoPlan::from_circuit(circuit, sim.registry(), &MemoConfig::default())?;
        let s = plan.stats();
        println!(
            "  {:<22} {:>6} {:>6} {:>8} {:>8.2} {:>6.1}% {:>7.2} {:>10}",
            name,
            s.gates_in,
            s.ops_out,
            s.entries,
            s.reuse(),
            100.0 * s.hit_rate(),
            s.fusion(),
            s.entry_bytes
        );
    }

    println!("\n== Exactness against the unmemoized circuit ==");
    for (name, circuit) in &cases {
        let truth = sim.run(circuit)?;
        let plan = MemoPlan::from_circuit(circuit, sim.registry(), &MemoConfig::default())?;
        let mut state = sim.backends().create("dense", circuit.num_qubits())?;
        state.reset();
        plan.run(state.as_mut())?;
        let dev = max_amplitude_deviation(truth.as_ref(), state.as_ref());
        println!("  {name:<22} deviation {dev:.3e}");
    }

    println!("\n== max_fuse dial on brickwork-10x6 ==");
    let circuit = brickwork(10, 6);
    let truth = sim.run(&circuit)?;
    for max_fuse in 1..=6 {
        let cfg = MemoConfig {
            max_fuse,
            ..Default::default()
        };
        let plan = MemoPlan::from_circuit(&circuit, sim.registry(), &cfg)?;
        let mut state = sim.backends().create("dense", circuit.num_qubits())?;
        state.reset();
        plan.run(state.as_mut())?;
        let dev = max_amplitude_deviation(truth.as_ref(), state.as_ref());
        let s = plan.stats();
        println!(
            "  max_fuse={max_fuse} ops={:<4} entries={:<4} reuse={:<5.2} widest={} bytes={:<7} dev={dev:.1e}",
            s.ops_out, s.entries, s.reuse(), s.widest_support, s.entry_bytes
        );
    }

    println!("\n== Re-exploring configurations over a shared prefix ==");
    println!(
        "  {:<10} {:>8} {:>22} {:>26} {:>7} {:>10}",
        "variants", "prefix", "operations applied", "wall clock (ns)", "ckpt", "deviation"
    );
    let prefix = brickwork(10, 6);
    for count in [2usize, 4, 8, 16] {
        let variants: Vec<Circuit> = (0..count)
            .map(|k| {
                let mut c = Circuit::new(10);
                c.gate("rz", vec![0.1 * (k as f64 + 1.0)], vec![0]);
                c.cx(0, 1);
                c
            })
            .collect();
        let r = explore(&sim, "dense", &prefix, &variants, &MemoConfig::default())?;
        println!(
            "  {:<10} {:>8} {:>9} -> {:<4} {:>5.2}x {:>11} -> {:<9} {:>5.2}x {:>4} {:>10.1e}",
            r.variants,
            r.prefix_ops,
            r.applied_naive,
            r.applied_memoized,
            r.counted_speedup(),
            r.nanos_naive,
            r.nanos_memoized,
            r.measured_speedup(),
            r.checkpoints,
            r.deviation
        );
    }
    println!(
        "\n  Sharing is verified, not assumed: the deviation column is the worst\n  \
         amplitude difference against running every variant from scratch, and the\n  \
         naive route gets the BETTER plan (it may fuse across the prefix boundary,\n  \
         which the shared route cannot) and still loses."
    );

    println!("\n== Unwinding and rewinding a single path ==");
    let plan = MemoPlan::from_circuit(&brickwork(8, 5), sim.registry(), &MemoConfig::default())?;
    let mut explorer = Explorer::new(&sim, "dense", &plan, 3)?;
    explorer.advance()?;
    println!(
        "  advanced to {} of {} ops, {} checkpoints holding {} amplitudes",
        explorer.cursor(),
        plan.ops().len(),
        explorer.checkpoints(),
        explorer.journal_amplitudes()
    );
    for target in [8usize, 4, 1, 0] {
        explorer.rewind(target)?;
        println!(
            "  rewound to {:<3} cursor={:<3} support={:<5} applied={} (replays included)",
            target,
            explorer.cursor(),
            explorer.state().nonzero_count(),
            explorer.applied()
        );
    }
    Ok(())
}
