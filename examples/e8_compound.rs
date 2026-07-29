//! The E8 compound qudit, live: the root system constructed and
//! verified, the four arity frames (binary/ternary/quaternary/quintary)
//! embedded with their measured forced correlation, and a mixed-arity
//! register running over exactly that fabric — representation,
//! interaction, and flow as three separately measured layers.
//!
//! Run with `cargo run --release --example e8_compound`.

use quantsim::e8;
use quantsim::mixed::{cshift, fabric, fourier_d, CompoundRegister};
use quantsim::prelude::*;

fn main() -> Result<()> {
    // ── 1. The E8 root system, built and checked ─────────────────────
    println!("── E8, constructed (never quoted)");
    e8::verify()?;
    println!("  240 roots generated: 112 integer + 128 even-sign spinor, every norm²");
    println!("  = 2, closed under negation — all verified programmatically.");

    let first = e8::find_a_chain(4, &[]).unwrap();
    let second = e8::find_a_chain(4, &first).unwrap();
    let ortho = first
        .iter()
        .all(|a| second.iter().all(|b| e8::dot(a, b) == 0));
    println!("  su(5) × su(5): a second A4 chain orthogonal to the first exists —");
    println!("  found by search, orthogonality checked: {ortho}.\n");

    // ── 2. The rank obstruction: correlation by necessity ────────────
    println!("── the four arity frames cannot be independent (measured)");
    let mut avoid = Vec::new();
    for k in [1usize, 2, 3] {
        avoid.extend(e8::find_a_chain(k, &avoid).unwrap());
    }
    let impossible = e8::find_a_chain(4, &avoid).is_none();
    println!("  A1 ⊥ A2 ⊥ A3 placed (rank 6); the search for an orthogonal A4 then");
    println!("  EXHAUSTS: {impossible} — rank 1+2+3+4 = 10 > 8. Binary, ternary,");
    println!("  quaternary and quintary frames must share directions in E8: the");
    println!("  cross-correlation of the horizontal set is geometric necessity.\n");

    let emb = ChainEmbedding::canonical()?;
    println!("── the canonical embedding and its measured coupling fabric");
    for (i, chain) in emb.chains.iter().enumerate() {
        println!(
            "  d = {} frame: A{} chain of {} roots (su({}))",
            i + 2,
            i + 1,
            chain.len(),
            i + 2
        );
    }
    let coupling = emb.coupling();
    println!("  coupling matrix (true = frames share root directions):");
    for (i, row) in coupling.iter().enumerate() {
        println!("    d={}: {row:?}", i + 2);
    }
    let overlap: i32 = emb.gram(2, 3).iter().flatten().map(|g| g.abs()).sum();
    println!("  the geometry couples exactly the quaternary–quintary pair (total");
    println!("  doubled Gram overlap {overlap}); binary and ternary stay independent.\n");

    // ── 3. The compound register over the E8 fabric ──────────────────
    println!("── a (2, 3, 4, 5) compound register over the E8 fabric");
    let dims = [2usize, 3, 4, 5];
    let bonds: Vec<(usize, usize)> = (0..4)
        .flat_map(|i| ((i + 1)..4).map(move |j| (i, j)))
        .filter(|&(i, j)| coupling[i][j])
        .collect();
    let report = fabric(&dims, &bonds)?;
    println!(
        "  E8-derived bonds: {:?}  (density {:.2}, cross-arity {} — swap cannot",
        report.bonds, report.density, report.cross_arity_bonds
    );
    println!("  exist between unequal arities, so every E8 bond here is native;");
    println!("  swap classes: {:?})", report.swap_classes);

    let mut reg = CompoundRegister::new(&dims)?;
    for (s, &d) in dims.iter().enumerate() {
        reg.apply_1(s, &fourier_d(d))?;
    }
    println!(
        "  local fouriers: {} volumes (horizontal set intact), largest {} states,",
        reg.volumes(),
        reg.largest_volume_states()
    );
    println!(
        "  stored entries {} = 2+3+4+5 (each volume its own arity)",
        reg.stored_entries()
    );

    reg.apply_2(2, 3, &cshift(4, 5))?;
    println!(
        "  cshift(4,5) along the E8 bond: {} volumes, largest {} states, weight {:.6}",
        reg.volumes(),
        reg.largest_volume_states(),
        reg.born_weight()
    );
    let merges = reg.merge_timeline().iter().filter(|i| i.merged).count();
    println!("  merge timeline records {merges} correlation event; cones:");
    for (s, &d) in dims.iter().enumerate() {
        println!(
            "    reachable from site {s} (d={d}): {:?}",
            reg.reachable_from(s)
        );
    }

    let counts = reg.sample(2000, &mut Prng::new(5))?;
    println!(
        "  Born-sampled 2000 shots: {} distinct outcomes over the 120-state space",
        counts.len()
    );

    // A cross-arity Bell pair: binary control, quintary target.
    let mut bell = CompoundRegister::new(&[2, 5])?;
    bell.apply_1(0, &fourier_d(2))?;
    bell.apply_2(0, 1, &cshift(2, 5))?;
    let counts = bell.sample(1000, &mut Prng::new(9))?;
    println!(
        "\n  a binary–quintary Bell pair samples exactly {} outcomes:",
        counts.len()
    );
    let mut rows: Vec<(Vec<usize>, u64)> = counts.into_iter().collect();
    rows.sort();
    for (outcome, n) in rows {
        println!("    {outcome:?}: {n}");
    }
    println!("\n  representation = horizontal volumes; interaction = the measured E8");
    println!("  fabric; flow = the recorded merge/cone structure. Three layers, three");
    println!("  separate measurements, one register.");
    Ok(())
}
