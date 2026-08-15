//! Two of the library's newer registers, shown working on DCS with
//! every decision printed rather than summarized.
//!
//! **Mosaic** holds each region of the register in its own backend and
//! merges two regions when a gate couples them, recording every merge
//! with its cause. The question it can answer that a single-backend
//! atlas cannot: DCS is 96% Clifford with its `T` gates on particular
//! wires, so does a partition that puts the magic-free wires in a
//! Clifford representation survive the brickwork, and for how long? The
//! full ledger is printed, not the first line of it.
//!
//! **Branched** holds a weighted superposition of slices at the *sum*
//! of their costs, and measures selector↔system entanglement from the
//! pairwise Gram of the slices — polynomially, never by enumeration.
//! [`cutsim`] writes `|ψ⟩ = Σ_p |ψ_A^p⟩ ⊗ |ψ_B^p⟩` over `2^k` branches,
//! which is exactly that object. So the branches go in and
//! `selector_schmidt_rank` answers what `cutsim` cannot ask itself: how
//! many of the `2^k` branches are linearly **independent**. Earlier
//! measurement found the branch *weights* exactly flat; flat weights do
//! not imply independent slices, and the gap between count and rank is
//! how much the branch sum is redundant.
//!
//! `cargo run --release --example dcs_mosaic`

use quantsim::backend::{CliffordFramedState, DenseState, MosaicPolicy, MosaicState, SparseState};
use quantsim::clock::BranchedRegister;
use quantsim::cutsim;
use quantsim::dcs::Dcs;
use quantsim::prelude::*;

fn rule() {
    println!("{}", "─".repeat(78));
}

/// Qubits carrying at least one `T`, read off the doping plan — no
/// simulation, just the construction's own site list.
fn magic_qubits(d: Dcs) -> Vec<bool> {
    let mut hot = vec![false; d.qubits];
    for site in d.doping() {
        hot[site.qubit] = true;
    }
    hot
}

/// Contiguous runs of equal flag: the coarsest partition separating
/// magic-carrying wires from magic-free ones.
fn bands(hot: &[bool]) -> Vec<(Vec<usize>, bool)> {
    let mut out: Vec<(Vec<usize>, bool)> = Vec::new();
    for (q, &h) in hot.iter().enumerate() {
        match out.last_mut() {
            Some((qs, flag)) if *flag == h => qs.push(q),
            _ => out.push((vec![q], h)),
        }
    }
    out
}

fn mosaic_run(d: Dcs, tag: &str, policy: MosaicPolicy, verbose: bool) {
    let n = d.qubits;
    let hot = magic_qubits(d);
    let parts = bands(&hot);

    let regions: Vec<(Vec<usize>, Box<dyn Backend<C64>>)> = parts
        .iter()
        .map(|(qs, is_hot)| {
            let w = qs.len();
            let b: Box<dyn Backend<C64>> = if *is_hot {
                Box::new(SparseState::<C64>::new(w).unwrap())
            } else {
                Box::new(CliffordFramedState::<C64>::new(w).unwrap())
            };
            (qs.clone(), b)
        })
        .collect();

    let seeded: Vec<String> = parts
        .iter()
        .map(|(qs, h)| {
            format!(
                "{}..{}:{}",
                qs[0],
                qs[qs.len() - 1],
                if *h { "sparse" } else { "frame" }
            )
        })
        .collect();

    let mut mosaic = MosaicState::<C64>::with_regions(n, regions, policy).unwrap();
    let reg = GateRegistry::<C64>::standard();
    let circuit = d.circuit();
    circuit.bind(&reg).unwrap().run(&mut mosaic).unwrap();

    let layout = mosaic.layout();
    let mem: usize = layout.iter().map(|(_, _, b)| b).sum();
    let mut dense = DenseState::<C64>::new(n).unwrap();
    circuit.bind(&reg).unwrap().run(&mut dense).unwrap();

    println!(
        "  {tag:<13} n={n:<3} t={:<3} seeded [{}]",
        d.t_gates,
        seeded.join(" ")
    );
    println!(
        "                final {} region(s), {} event(s), memory {mem} vs dense {} ({:.2}x)",
        layout.len(),
        mosaic.events().len(),
        dense.memory_bytes(),
        mem as f64 / dense.memory_bytes() as f64
    );
    if verbose {
        for (i, e) in mosaic.events().iter().enumerate() {
            println!(
                "                  [{i}] {:<8} {:>2} qubits  {:<26} → {:<16} {}",
                e.kind,
                e.qubits.len(),
                e.from,
                e.to,
                e.cause
            );
        }
    }
}

fn main() {
    rule();
    println!("A. MOSAIC — the merge ledger in full");
    rule();
    println!("  Magic-free bands seeded on the Clifford frame, magic-carrying bands on");
    println!("  sparse. Every decision the mosaic made, in order, with its stated cause.");
    println!();
    mosaic_run(
        Dcs::scaled(12),
        "sparse-elect",
        MosaicPolicy::default(),
        true,
    );
    println!();
    mosaic_run(
        Dcs::scaled(12),
        "frame-elect",
        MosaicPolicy {
            fallbacks: vec!["clifford-framed".into(), "sparse".into(), "dense".into()],
            ..MosaicPolicy::default()
        },
        true,
    );
    println!();
    println!("  Same circuit, same seed partition. The only difference is whether the");
    println!("  merge election may reach `clifford-framed`, and the ledger shows the");
    println!("  cost each choice actually incurred.");

    rule();
    println!("B. BRANCHED — are the cut's branches independent?");
    rule();
    println!("  cutsim writes |ψ⟩ = Σ_p |ψ_A^p⟩⊗|ψ_B^p⟩ over 2^k branches. Loading them");
    println!("  into a selector qudit lets `selector_schmidt_rank` measure how many are");
    println!("  linearly independent — from the pairwise Gram, not by enumeration.");
    println!();
    println!("     n    k   branches   selector rank   unique states   independent?");
    for n in [8usize, 10, 12, 14] {
        let d = Dcs::scaled(n);
        let circuit = d.circuit();
        let plan = cutsim::plan(&circuit, n / 2).unwrap();
        let branches = cutsim::branch_states(&circuit, &plan).unwrap();

        let loaded: Vec<(usize, C64, Box<dyn Backend<C64>>)> = branches
            .iter()
            .enumerate()
            .map(|(i, (w, v))| {
                let mut st = SparseState::<C64>::new(n).unwrap();
                let entries: Vec<(u64, C64)> = v
                    .iter()
                    .enumerate()
                    .filter(|(_, z)| z.norm_sqr() > 0.0)
                    .map(|(x, z)| (x as u64, *z))
                    .collect();
                st.load(&entries).unwrap();
                let b: Box<dyn Backend<C64>> = Box::new(st);
                (i, C64::new(*w, 0.0), b)
            })
            .collect();

        let count = loaded.len();
        let br = BranchedRegister::<C64>::from_branches(n, count, loaded).unwrap();
        let rank = br.selector_schmidt_rank().unwrap();
        println!(
            "  {n:>4}  {:>3}   {count:>8}   {rank:>13}   {:>13}   {}",
            plan.crossings.len(),
            br.unique_state_count(),
            if rank == count {
                "fully — no redundancy to exploit"
            } else {
                "NO — the sum is compressible"
            }
        );
    }
    rule();
}
