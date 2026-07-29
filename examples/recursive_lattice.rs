//! Recursive quantum systems: a site that is either a point or an
//! entire lattice of the same kind, and a computation that participates
//! in itself.
//!
//! Four qubits bonded in a square; then the same structure again,
//! bonded laterally; then the measurements that decide whether the
//! substitution means anything — the block-spin flow and its fixed
//! point, the spectral trial with its resolution horizon, the harmonic
//! lattice where the duality is exact, the same statement executed on a
//! real backend, and finally the self-consistent loop in which a
//! block's own answer is the field it answers in.
//!
//! Run with `cargo run --release --example recursive_lattice`.

use quantsim::prelude::*;
use quantsim::recursive::*;

fn main() -> Result<()> {
    // ── 1. The structure ─────────────────────────────────────────────
    println!("── four qubits in a square, then the same structure again");
    for depth in 1..=3 {
        let l = RecursiveLattice::nest(Shape::SQUARE, depth)?;
        let bonds = l.bonds();
        let lateral = bonds.iter().filter(|b| b.depth == 0).count();
        let t = l.topology()?;
        println!(
            "  depth {depth}: {:>2} qubits, {:>2} bonds ({lateral} at the top level), \
             horizon {}, mean dist {:.2}",
            l.width(),
            bonds.len(),
            t.diameter(),
            t.mean_distance()
        );
    }
    let sq2 = RecursiveLattice::nest(Shape::SQUARE, 2)?;
    println!("  square of squares: {}", sq2.describe());
    println!("  blocks at depth 1: {:?}", sq2.blocks(1));
    println!(
        "  lateral bonds: {:?}  — each joins one block's corner to the next block's",
        sq2.bonds()
            .iter()
            .filter(|b| b.depth == 0)
            .map(|b| (b.a, b.b))
            .collect::<Vec<_>>()
    );
    let cube = RecursiveLattice::nest(Shape::CUBE, 2)?;
    println!(
        "  and the 2×2×2 volume nested the same way: {} qubits, {} bonds \
         (its block is the E8 coordinate cube)",
        cube.width(),
        cube.bonds().len()
    );

    // ── 2. A block reads its own effective description ───────────────
    println!("\n── the block-spin step: a lattice measuring itself as a point");
    let block = RecursiveLattice::new(Shape::SQUARE)?;
    println!(
        "  {:>6}  {:>10} {:>10} {:>10} {:>10}   g' = J'/h'",
        "J/h", "gap", "mu", "J'", "h'"
    );
    for g in [0.2, 0.5, 1.0, 2.0, 4.0] {
        let r = block_rg(&block, g, 1.0)?;
        println!(
            "  {g:>6.2}  {:>10.6} {:>10.6} {:>10.6} {:>10.6}   {:.6}",
            r.gap,
            r.port_element,
            r.renorm_coupling,
            r.renorm_field,
            r.renorm_coupling / r.renorm_field
        );
    }
    println!(
        "  (both numbers are measured: h' is half the block's own gap, \
         J' is J times its own boundary element squared)"
    );

    // ── 3. The flow and its fixed point ──────────────────────────────
    println!("\n── the flow: repeated substitution of a block for a point");
    for start in [0.3, 1.5] {
        let flow = rg_flow(Shape::SQUARE, start, 1.0, 10)?;
        let trail: Vec<String> = flow
            .steps
            .iter()
            .map(|s| format!("{:.3e}", s.ratio))
            .collect();
        println!("  g0 = {start}: {}", trail.join(" → "));
        if let Some(reason) = flow.terminated {
            println!("    stopped: {reason}");
        }
    }
    println!("\n── the fixed point: where a lattice IS its own point");
    println!(
        "  {:<14} {:>10} {:>12} {:>10} {:>10}",
        "block", "g*", "residual", "lambda", "nu"
    );
    for (shape, b) in [
        (Shape::Chain(2), 2.0),
        (Shape::Chain(3), 3.0),
        (Shape::SQUARE, 2.0),
    ] {
        match rg_fixed_point(shape, b) {
            Ok(fp) => println!(
                "  {:<14} {:>10.6} {:>12.2e} {:>10.6} {:>10.4}",
                format!("{:?}", shape),
                fp.ratio,
                fp.residual,
                fp.eigenvalue,
                fp.exponent
            ),
            Err(e) => println!("  {:<14} refused: {e}", format!("{:?}", shape)),
        }
    }
    println!(
        "  (Chain(2) against the exactly known nu = 1 of the 1D transverse Ising chain: \
         the block-spin approximation, measured rather than hidden)"
    );

    // ── 4. The substitution on trial ─────────────────────────────────
    println!("\n── two blocks bonded laterally vs two points at (J', h')");
    println!(
        "  {:>6} {:>11} {:>11} {:>10} {:>11} {:>7}",
        "J/h", "internal gap", "in-band dev", "all levels", "in band", "resid"
    );
    for g in [0.2, 0.5, 1.0, 2.0, 3.0] {
        let r = substitution_report(Shape::SQUARE, g, 1.0, 3)?;
        println!(
            "  {g:>6.2} {:>11.4} {:>11.3e} {:>10.3e} {:>11} {:>7.1e}",
            r.internal_gap, r.deviation_in_band, r.deviation, r.levels_in_band, r.residual
        );
    }
    let r = substitution_report(Shape::SQUARE, 0.2, 1.0, 3)?;
    println!(
        "  at J/h = 0.2 the fine gaps are {:?}",
        r.fine_gaps.iter().map(|x| round6(*x)).collect::<Vec<_>>()
    );
    println!(
        "                 the coarse gaps {:?}",
        r.coarse_gaps.iter().map(|x| round6(*x)).collect::<Vec<_>>()
    );
    println!(
        "  the third coarse level ({:.3}) sits above the block's internal gap ({:.3}): \
         the duality has a resolution horizon, and it is measured",
        r.coarse_gaps[2], r.internal_gap
    );

    // ── 5. Phonons: where the duality is exact ───────────────────────
    println!("\n── the harmonic block: a collective mode that IS the point's mode");
    let pb = phonon_block(Shape::SQUARE, 1.0, 0.3)?;
    println!(
        "  collective frequency {:.15} (bare omega 1.0) — shift {:.2e}",
        pb.collective_frequency, pb.frequency_shift
    );
    println!(
        "  port participation {:.6}, internal gap {:.6}, renormalized spring {:.6}",
        pb.port_participation, pb.internal_gap, pb.renorm_spring
    );
    println!("\n── substituting a block for a point, both bonding regimes");
    println!("  {:<9} {:>10} {:>14}", "bonding", "lateral", "deviation");
    for bonding in [Bonding::Uniform, Bonding::Port] {
        for lateral in [0.3, 0.03, 0.003] {
            let s = phonon_substitution(Shape::SQUARE, 1.0, 0.5, lateral, bonding)?;
            println!(
                "  {:<9} {lateral:>10.3} {:>14.4e}",
                format!("{:?}", bonding),
                s.deviation
            );
        }
    }
    println!(
        "  uniform bonding leaves the collective subspace exactly invariant — \
         the substitution is exact at 1e-16;"
    );
    println!(
        "  port bonding mixes it into the internal modes, second order in the \
         lateral spring (10x weaker, 100x smaller)"
    );

    // ── 6. The same statement, on a real backend ─────────────────────
    println!("\n── one phonon walking the recursive lattice on the dense backend");
    println!(
        "  {:>8} {:>7} {:>10} {:>12} {:>14} {:>12}",
        "lateral", "steps", "leakage", "trotter dev", "substitution", "coarse dev"
    );
    for lateral in [0.1, 0.01] {
        for steps in [40usize, 160, 640] {
            let w = phonon_walk(Shape::SQUARE, 1.0, lateral, 20.0 / steps as f64, steps)?;
            println!(
                "  {lateral:>8.3} {steps:>7} {:>10.2e} {:>12.4e} {:>14.4e} {:>12.4e}",
                w.leakage, w.trotter_deviation, w.substitution_deviation, w.coarse_deviation
            );
        }
    }
    println!(
        "  the leakage column is a conservation law measured, not assumed: the xy gate \
         never leaves the one-phonon sector"
    );

    // ── 7. The computation that participates in itself ───────────────
    println!("\n── a block that is its own environment");
    println!(
        "  {:>6} {:>8} {:>14} {:>12} {:>7}",
        "J", "rounds", "m*", "field", "resid"
    );
    for j in [0.2, 0.3, 0.5, 1.0, 2.0] {
        let p = self_participation(&block, j, 1.0, 2, 0.5, 1e-10, 400)?;
        println!(
            "  {j:>6.2} {:>8} {:>14.9} {:>12.6} {:>7.1e}",
            p.rounds.len(),
            p.magnetization,
            p.field,
            p.residual
        );
    }
    let trace = self_participation(&block, 0.5, 1.0, 2, 0.5, 1e-10, 400)?;
    let path: Vec<String> = trace
        .rounds
        .iter()
        .take(8)
        .map(|r| format!("{:.6}", r.magnetization))
        .collect();
    println!("  J = 0.5 trajectory: 0.500000 → {}", path.join(" → "));

    let t = participation_transition(&block, 1.0, 2, 1e-3, (0.05, 3.0), 40)?;
    println!(
        "\n  critical coupling {:.6} (bracket {:.1e}): m = {:.6} at 1.1x, {:.2e} at 0.9x",
        t.coupling, t.bracket, t.magnetization_above, t.magnetization_below
    );
    println!(
        "  below it the only self-consistent answer is zero — the computation talks \
         itself down to nothing"
    );

    // The block may itself be a lattice: same call, one scale deeper.
    let deep = RecursiveLattice::nest(Shape::Chain(3), 2)?;
    let dp = self_participation(&deep, 1.0, 1.0, 2, 0.5, 1e-9, 200)?;
    println!(
        "\n  and with the block itself a lattice ({}, {} qubits): m* = {:.6} in {} rounds",
        deep.describe(),
        deep.width(),
        dp.magnetization,
        dp.rounds.len()
    );

    Ok(())
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}
