//! The time system inside the register, end to end: the scale-aligned
//! clock qudit over the bulk register's depth axis (Page–Wootters
//! conditioning, the tick, scale interferometry) and the
//! representation payoff of going up a dimension — a selector qudit
//! over representation-heterogeneous slices, priced against every
//! single-backend alternative.
//!
//! Run with `cargo run --release --example scale_time`.

use quantsim::backend::{BulkState, DenseState, FactoredState, MpsState, SparseState};
use quantsim::prelude::*;

fn graph_circuit(n: usize, edges: usize, seed: u64) -> Circuit<C64> {
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    let mut placed = 0;
    while placed < edges {
        let a = (rng.next_u64() % n as u64) as usize;
        let b = (rng.next_u64() % n as u64) as usize;
        if a != b {
            c.gate("cz", [], [a, b]);
            placed += 1;
        }
    }
    c
}

fn main() -> Result<()> {
    let sim = Simulator::<C64>::new();
    let reg = GateRegistry::<C64>::standard();

    // ══ 1. the scale clock: one static state, every resolution ══
    println!("══ 1. the scale clock: Page–Wootters over the depth axis ══\n");
    let mut c: Circuit<C64> = Circuit::new(8);
    for q in [0usize, 2, 4, 6] {
        c.h(q).cx(q, q + 1);
    }
    c.t(1).cx(1, 2).cx(5, 6).cx(3, 4).s(4);
    let state = sim.run_on("bulk", &c)?;
    let bulk = state.as_any().downcast_ref::<BulkState<C64>>().unwrap();
    let scales = bulk.depth() + 1;
    let w = c64(1.0 / (scales as f64).sqrt(), 0.0);
    let hist = scale_history(bulk, &vec![w; scales])?;
    println!(
        "  {} scales in one clock qudit; selector↔system Schmidt rank = {} (mandatory)",
        scales,
        hist.selector_schmidt_rank()?
    );
    for level in 0..scales {
        let mut h = scale_history(bulk, &vec![w; scales])?;
        let p = h.condition(level)?;
        let snap = bulk.scale_snapshot(level)?;
        let dev = max_amplitude_deviation(&h, &snap);
        println!(
            "  condition clock = {level}: p = {p:.4}, slice ≡ scale-{level} view (dev {dev:.1e})"
        );
    }

    // ══ 2. the tick: internal time is coarse-to-fine flow ══
    println!("\n══ 2. the tick: |ℓ⟩⟨ℓ| ⊗ V_ℓ, |ℓ⟩ → |ℓ+1⟩ ══\n");
    let prog = bulk.unfold_program()?;
    let mut w3: Vec<C64> = vec![c64(1.0 / 3f64.sqrt(), 0.0); 3];
    w3.resize(scales, c64(0.0, 0.0));
    let mut hist = scale_history(bulk, &w3)?;
    let before = hist.selector_probabilities()?;
    tick(&mut hist, &prog)?;
    let after = hist.selector_probabilities()?;
    println!("  clock distribution before: {before:.3?}");
    println!("  clock distribution after:  {after:.3?}");
    println!("  every occupied scale advanced one level of refinement, in register.");

    // ══ 3. scale interferometry: the speed of the flow ══
    println!("\n══ 3. scale interferometry ══\n");
    let f = std::f64::consts::FRAC_1_SQRT_2;
    let two = c64(f, 0.0);
    let h_mix = vec![two, two, two, -two];
    for (label, source) in [
        ("structured 8q state", bulk.clone()),
        ("|0…0⟩ (nothing to refine)", BulkState::<C64>::new(8)?),
    ] {
        let mut pair = scale_history(&source, &[two, two])?;
        pair.selector_mix(&h_mix)?;
        let probs = pair.selector_probabilities()?;
        let visibility = probs[0] - probs[1];
        println!(
            "  {label}: p₊ = {:.4}, p₋ = {:.4} → ⟨scale ℓ|scale ℓ+1⟩ = {visibility:.4}",
            probs[0], probs[1]
        );
    }
    println!("  the clock's Born statistics measure how far one RG step moves the state.");

    // ══ 4. the payoff: more dimension above, less structure below ══
    println!("\n══ 4. four slices, four representations, one register ══\n");
    let n = 16;
    let ghz_c = library::ghz::<C64>(n);
    let rainbow_c = library::rainbow::<C64>(n);
    let brick_c = library::brickwork::<C64>(n, 3, &(0..n).collect::<Vec<_>>());
    let graph_c = graph_circuit(n, 24, 9);
    let run_in = |circuit: &Circuit<C64>, state: &mut dyn Backend<C64>| {
        circuit.bind(&reg).unwrap().run(state).unwrap();
    };
    let mut ghz_b: Box<dyn Backend<C64>> = Box::new(SparseState::<C64>::new(n)?);
    let mut rainbow_b: Box<dyn Backend<C64>> = Box::new(FactoredState::<C64>::new(n)?);
    let mut brick_b: Box<dyn Backend<C64>> = Box::new(MpsState::<C64>::new(n)?);
    let mut graph_b = sim.backends().create("bundle", n)?;
    run_in(&ghz_c, ghz_b.as_mut());
    run_in(&rainbow_c, rainbow_b.as_mut());
    run_in(&brick_c, brick_b.as_mut());
    run_in(&graph_c, graph_b.as_mut());
    println!(
        "  slices: GHZ→sparse {} B, rainbow→factored {} B, brickwork→mps {} B, graph→bundle {} B",
        ghz_b.memory_bytes(),
        rainbow_b.memory_bytes(),
        brick_b.memory_bytes(),
        graph_b.memory_bytes()
    );
    let wq = c64(0.5, 0.0);
    let branched = BranchedRegister::from_branches(
        n,
        4,
        vec![
            (0, wq, ghz_b),
            (1, wq, rainbow_b),
            (2, wq, brick_b),
            (3, wq, graph_b),
        ],
    )?;
    println!(
        "  the flagged register: {} B total, selector rank {} — the qudit carries the clash",
        branched.memory_bytes(),
        branched.selector_schmidt_rank()?
    );
    let mut ghz_f = FactoredState::<C64>::new(n)?;
    run_in(&ghz_c, &mut ghz_f);
    let mut brick_s = SparseState::<C64>::new(n)?;
    run_in(&brick_c, &mut brick_s);
    let mut graph_f = FactoredState::<C64>::new(n)?;
    run_in(&graph_c, &mut graph_f);
    let dense_bytes = DenseState::<C64>::new(n)?.memory_bytes();
    println!("  the single-representation alternatives, on their clashing slices:");
    println!("    ghz on factored:    {:>9} B", ghz_f.memory_bytes());
    println!("    brickwork on sparse:{:>9} B", brick_s.memory_bytes());
    println!("    graph on factored:  {:>9} B", graph_f.memory_bytes());
    println!("    dense (any slice):  {:>9} B", dense_bytes);
    println!("\n  the sum that no single structure holds cheaply is additive under");
    println!("  the selector — new qudit dimension above the problem, lower");
    println!("  representational structure below it.");
    Ok(())
}
