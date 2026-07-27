//! The hierarchical register, end to end: coarse views of the gross
//! register at every scale, refinement down to physical amplitudes, the
//! truncation dial measured, and the composition with certified (Ball)
//! and exact (D[ω]) arithmetic — representation resolution and numeric
//! resolution as two dials on the same machine.
//!
//! Run with `cargo run --release --example coarse_register`.

use quantsim::exact::ExactState;
use quantsim::prelude::*;

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
        }
    }
    out
}

fn main() -> Result<()> {
    let sim = Simulator::<C64>::new();

    // ══ 1. GHZ(16): the gross register at every resolution ══
    println!("══ 1. coarse views: GHZ(16) through the hierarchy ══\n");
    let n = 16;
    let state = sim.run_on("mera", &library::ghz(n))?;
    let mera = state.as_any().downcast_ref::<MeraState<C64>>().unwrap();
    println!(
        "GHZ({n}) stored hierarchically: {} resting, bond {} at every level, exact = {}\n",
        state.memory_bytes(),
        mera.max_bond_dimension(),
        mera.is_exact()
    );
    for depth in 0..=mera.depth() {
        let dims: Vec<usize> = mera.coarse_dims(depth).iter().map(|&(_, _, b)| b).collect();
        let (_, coeffs) = mera.coarse_state(depth)?;
        let support = coeffs.iter().filter(|a| a.norm_sqr() > 1e-18).count();
        let weight: f64 = coeffs.iter().map(|a| a.norm_sqr()).sum();
        println!(
            "  depth {depth}: {:>2} super-site(s), dims {dims:?} — support {support}, weight {weight:.6}",
            dims.len()
        );
    }
    let (dims, coeffs) = mera.coarse_state(1)?;
    println!(
        "\nthe gross register at depth 1 ({}×{} super-sites) IS a Bell pair:",
        dims[0], dims[1]
    );
    for (i, a) in coeffs.iter().enumerate() {
        if a.norm_sqr() > 1e-18 {
            println!(
                "  |site₁={}, site₀={}⟩  amp = {a}",
                i / dims[0],
                i % dims[0]
            );
        }
    }
    let (up, l, r, _) = mera.refine_basis(1, 0)?;
    println!("refinement dictionary of super-site 0: bond {up} expands to {l}×{r} one level down,");
    println!(
        "…and fully refined (leaf level): P(|0…0⟩) = {:.4}, P(|1…1⟩) = {:.4}, support {}\n",
        state.probability(0),
        state.probability((1 << n) - 1),
        state.nonzero_count()
    );

    // ══ 2. the truncation dial, measured ══
    println!("══ 2. resolution as a dial: bond cap × measured error ══\n");
    let circuit = library::random_circuit(8, 60, 33);
    let reg = GateRegistry::<C64>::standard();
    let bound = circuit.bind(&reg)?;
    let dense = sim.run(&circuit)?;
    println!("random 8-qubit, 60-gate circuit vs dense reference:");
    println!(
        "{:>10} {:>14} {:>16} {:>12}",
        "max_bond", "deviation", "discarded Σσ²", "exact?"
    );
    for cap in [64usize, 8, 4, 2] {
        let mut coarse = MeraState::<C64>::with_config(
            8,
            MeraConfig {
                max_bond: cap,
                ..MeraConfig::default()
            },
        )?;
        bound.run(&mut coarse)?;
        println!(
            "{cap:>10} {:>14.3e} {:>16.3e} {:>12}",
            max_amplitude_deviation(dense.as_ref(), &coarse),
            coarse.discarded_weight(),
            coarse.is_exact()
        );
    }
    println!("(coarser cap ⇒ more discarded weight ⇒ larger measured deviation — never silent)\n");

    // ══ 3. width past dense, entanglement hierarchy-local ══
    println!("══ 3. width 40, hierarchy-local entanglement ══\n");
    let mut wide = MeraState::<C64>::new(40)?;
    let h = reg.resolve("h")?.matrix(&[])?;
    let t = reg.resolve("t")?.matrix(&[])?;
    let cx = reg.resolve("cx")?.matrix(&[])?;
    for block in 0..8 {
        let base = block * 5;
        wide.apply(&h, &[base])?;
        wide.apply(&cx, &[base, base + 1])?;
        wide.apply(&t, &[base + 1])?;
        wide.apply(&cx, &[base + 1, base + 2])?;
        wide.apply(&cx, &[base + 2, base + 3])?;
        wide.apply(&cx, &[base + 3, base + 4])?;
    }
    let dims: Vec<usize> = wide.coarse_dims(3).iter().map(|&(_, _, b)| b).collect();
    println!(
        "40 qubits, 8 scrambled 5-qubit blocks: {} bytes resting (dense would be 16 TiB),",
        wide.memory_bytes()
    );
    println!(
        "peak transient block {} elements, weight {:.6}, gross register at depth 3: dims {dims:?}",
        wide.peak_block_elements(),
        wide.total_weight()
    );
    println!(
        "(bond 1 everywhere: the scrambled blocks are mutually unentangled, so at this\n resolution the gross register is a product of pure super-sites — geometry as data)\n"
    );

    // ══ 4. certified coarseness: Ball through the hierarchy ══
    println!("══ 4. Ball × mera: certified numbers through the SVD splits ══\n");
    let mut bc: Circuit<Ball> = Circuit::new(6);
    bc.h(0)
        .cx(0, 1)
        .t(1)
        .cx(1, 2)
        .cx(2, 3)
        .s(3)
        .cx(3, 4)
        .cx(4, 5);
    let ball_state = Simulator::<Ball>::new().run_on("mera", &bc)?;
    let exact = ExactState::run(&retype(&bc))?;
    let mut worst_err = 0.0f64;
    let mut worst_rad = 0.0f64;
    let mut contained = true;
    for i in 0..(1u64 << 6) {
        let ball = ball_state.amplitude(i);
        let truth = exact.amplitude_c64(i);
        contained &= ball.contains(truth);
        worst_err = worst_err.max((truth - ball.mid).norm());
        worst_rad = worst_rad.max(ball.rad);
    }
    println!("6-qubit Clifford+T circuit on mera over Ball, certified against exact D[ω]:");
    println!(
        "  every amplitude ball contains the exact ring value: {contained}\n  worst |exact − midpoint| = {worst_err:.3e} ≤ worst certified radius = {worst_rad:.3e}"
    );
    println!("  (representation truncation and numeric radius are independent dials:");
    println!("   discarded weight bounds the first, Ball radii certify the second.)");
    Ok(())
}
