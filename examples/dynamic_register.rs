//! The dynamically scaled bulk–boundary register, end to end: a live
//! entangled register growing and shrinking at runtime, the state
//! stored across depth as isometries under an explicit bulk record,
//! truncation certified by a per-depth projective ledger, and the
//! depth store unfolded into a width-growing circuit whose wires are
//! entangled only along their structural sequences.
//!
//! Run with `cargo run --release --example dynamic_register`.

use quantsim::backend::{BulkConfig, BulkState, MeraConfig, MeraState};
use quantsim::prelude::*;

fn l2(dense: &dyn Backend<C64>, other: &dyn Backend<C64>, n: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..(1u64 << n) {
        sum += (dense.amplitude(i) - other.amplitude(i)).norm_sqr();
    }
    sum.sqrt()
}

fn main() -> Result<()> {
    let sim = Simulator::<C64>::new();
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h")?.matrix(&[])?;
    let cx = reg.resolve("cx")?.matrix(&[])?;

    // ══ 1. Growth: a live GHZ register scaled 2 → 32 ══
    println!("══ 1. dynamic growth: GHZ scaled 2 → 32 while entangled ══\n");
    let mut s = BulkState::<C64>::new(2)?;
    s.apply(&h, &[0])?;
    s.apply(&cx, &[0, 1])?;
    while s.num_qubits() < 32 {
        let q = s.num_qubits();
        s.grow(1)?;
        s.apply(&cx, &[q - 1, q])?;
    }
    println!(
        "  width {} (capacity {}), {} re-roots for 30 grows — amortized doubling",
        s.num_qubits(),
        s.capacity(),
        s.reroots()
    );
    println!(
        "  exact = {}, bond {}, memory {} bytes, ‖ψ‖² from the record = {:.12}",
        s.is_exact(),
        s.max_bond_dimension(),
        s.memory_bytes(),
        s.total_abs_sqr()
    );
    println!(
        "  amp|0…0⟩ = {:.12}  amp|1…1⟩ = {:.12}\n",
        s.amplitude(0).re,
        s.amplitude((1u64 << 32) - 1).re
    );

    // ══ 2. Release: verified, never assumed ══
    println!("══ 2. release: refused with the measured leakage, honest after measurement ══\n");
    match s.release(1) {
        Err(e) => println!("  entangled qubit: {e}"),
        Ok(_) => unreachable!(),
    }
    let mut rng = Prng::new(11);
    let outs = s.release_measured(4, &mut rng)?;
    println!(
        "  measured-released 4 qubits: outcomes {outs:?} (GHZ-correlated), width now {}, ‖ψ‖² = {:.9}\n",
        s.num_qubits(),
        s.total_abs_sqr()
    );

    // ══ 3. A stream wider than the register ══
    println!("══ 3. streaming: 48 logical qubits through a width-2 register ══\n");
    let mut src = BulkState::<C64>::new(1)?;
    src.apply(&h, &[0])?;
    let mut rng = Prng::new(3);
    let mut outcomes = Vec::new();
    for _ in 0..48 {
        src.grow(1)?;
        src.apply(&cx, &[0, 1])?;
        outcomes.extend(src.release_measured(1, &mut rng)?);
    }
    println!(
        "  48 emissions, all equal to the first: {} — peak width {} the whole run\n",
        outcomes.iter().all(|&o| o == outcomes[0]),
        src.peak_width()
    );

    // ══ 4. Truncation: environment-weighted, certified per depth ══
    println!("══ 4. truncation: the certified projective ledger, vs block-local ══\n");
    for seed in [0u64, 1] {
        let n = 8;
        let c = library::random_circuit(n, 40, seed);
        let dense = sim.run(&c)?;
        let mut bulk = BulkState::<C64>::with_config(
            n,
            BulkConfig {
                max_bond: 2,
                ..BulkConfig::default()
            },
        )?;
        let mut mera = MeraState::<C64>::with_config(
            n,
            MeraConfig {
                max_bond: 2,
                trunc_tol: 1e-12,
                max_block: 63,
            },
        )?;
        let bound_c = c.bind(&reg)?;
        bound_c.run(&mut bulk)?;
        bound_c.run(&mut mera)?;
        println!(
            "  seed {seed}, cap 2: measured L2 error {:.4} ≤ certified bound {:.4}; \
             block-local (mera) {:.4}",
            l2(dense.as_ref(), &bulk, n),
            bulk.l2_error_bound(),
            l2(dense.as_ref(), &mera, n),
        );
        println!(
            "           per-depth ledger {:?}, ‖top‖ = {:.6}",
            bulk.discarded_by_depth()
                .iter()
                .map(|x| (x * 1e4).round() / 1e4)
                .collect::<Vec<_>>(),
            bulk.total_abs_sqr().sqrt()
        );
    }
    println!();

    // ══ 5. The unfold: structured entanglement, one wire at a time ══
    println!("══ 5. unfold: the depth store as a width-growing circuit ══\n");
    let mut c: Circuit<C64> = Circuit::new(8);
    for q in [0usize, 2, 4, 6] {
        c.h(q).cx(q, q + 1);
    }
    c.t(1).cx(1, 2).cx(5, 6).cx(3, 4).s(4);
    let state = sim.run_on("bulk", &c)?;
    let bulk = state.as_any().downcast_ref::<BulkState<C64>>().unwrap();
    let prog = bulk.unfold_program()?;
    println!(
        "  {} dilated steps, widest {} wires, seed dim {}",
        prog.steps.len(),
        prog.max_step_qubits(),
        prog.seed.len()
    );
    let mut dense = DenseState::<C64>::new(8)?;
    prog.run_on(&mut dense)?;
    println!(
        "  replay deviation vs the stored state: {:.3e}",
        max_amplitude_deviation(bulk, &dense)
    );
    for q in 0..8 {
        println!(
            "  wire {q}: injected at depth {:?}, sequence {:?}",
            prog.injection_depth(q).unwrap(),
            prog.sequence(q)
        );
    }
    println!("\n  each wire is entangled ONLY through that sequence — the");
    println!("  channel crossing a tree cut carries all of the cut's rank.");
    Ok(())
}
