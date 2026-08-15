//! Region-wise representation on the DCS circuit, driven by where the
//! magic actually is.
//!
//! [`MosaicState`] holds each region of the register in its own backend
//! and merges two regions when a gate couples them — recording every
//! merge with its cause. That makes it the instrument for a question
//! the single-backend atlas cannot ask: the DCS circuit is 96% Clifford
//! with its `T` gates confined to particular wires, so does a partition
//! that puts the magic-free wires in a Clifford representation *survive*
//! the brickwork, and if not, for how long?
//!
//! The merge ledger is the answer either way. A partition that collapses
//! immediately says the brickwork couples everything at once; one that
//! survives `k` layers puts a number on the magic-locality horizon.
//!
//! `cargo run --release --example dcs_mosaic`

use quantsim::backend::{CliffordFramedState, DenseState, MosaicPolicy, MosaicState, SparseState};
use quantsim::dcs::{Dcs, Doping};
use quantsim::prelude::*;

/// Qubits carrying at least one `T`, from the doping plan — no
/// simulation, just the construction's own site list.
fn magic_qubits(d: Dcs) -> Vec<bool> {
    let mut hot = vec![false; d.qubits];
    for site in d.doping() {
        hot[site.qubit] = true;
    }
    hot
}

/// Contiguous runs of equal flag — the coarsest partition that
/// separates magic-carrying wires from magic-free ones.
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

fn run(d: Dcs, label: &str, policy: MosaicPolicy) {
    let n = d.qubits;
    let hot = magic_qubits(d);
    let parts = bands(&hot);
    let hot_count = hot.iter().filter(|&&h| h).count();

    // Magic-free bands go to the Clifford frame, which factors out
    // exactly what they contain; magic-carrying bands go to sparse.
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

    let mut mosaic =
        MosaicState::<C64>::with_regions(n, regions, policy).unwrap();
    let reg = GateRegistry::<C64>::standard();
    let circuit = d.circuit();
    circuit.bind(&reg).unwrap().run(&mut mosaic).unwrap();

    let layout = mosaic.layout();
    let widest = layout.iter().map(|(qs, _, _)| qs.len()).max().unwrap_or(0);
    let mem: usize = layout.iter().map(|(_, _, b)| b).sum();

    // A single-region baseline on the same circuit.
    let mut dense = DenseState::<C64>::new(n).unwrap();
    circuit.bind(&reg).unwrap().run(&mut dense).unwrap();

    println!(
        "  {label:<22} n={n:<3} t={:<4}  {} magic wires, {} seed regions",
        d.t_gates,
        hot_count,
        parts.len()
    );
    println!(
        "      after the run: {} region(s), widest {widest}, {} merge event(s), {} collapse(s)",
        layout.len(),
        mosaic.events().len(),
        mosaic.collapses()
    );
    println!(
        "      memory  mosaic {mem}  vs dense {}   ({:.2}x)",
        dense.memory_bytes(),
        mem as f64 / dense.memory_bytes() as f64
    );
    // Where did it stop being a partition? The first merge event that
    // takes the region count to one is the horizon.
    if let Some(first) = mosaic.events().first() {
        println!("      first event: {first:?}");
    }
}

fn main() {
    println!("{}", "─".repeat(78));
    println!("REGION-WISE REPRESENTATION ON DCS, PARTITIONED BY WHERE THE MAGIC IS");
    println!("{}", "─".repeat(78));
    println!("  Magic-free wires seeded on the Clifford frame, magic-carrying wires");
    println!("  on sparse. Mosaic merges regions when a gate couples them and says so.");
    println!();
    for n in [10, 12, 14, 16] {
        for (tag, policy) in [
            ("sparse-elect", MosaicPolicy::default()),
            (
                "frame-elect ",
                MosaicPolicy {
                    fallbacks: vec!["clifford-framed".into(), "sparse".into(), "dense".into()],
                    ..MosaicPolicy::default()
                },
            ),
        ] {
            run(Dcs::scaled(n), tag, policy);
        }
    }
    println!();
    println!("  The separable profile from `dcs_separability`, where the magic really");
    println!("  does fall into small components:");
    for n in [12, 16] {
        run(
            Dcs::scaled(n).with_doping(Doping::Banded {
                width: 2,
                gap: 6,
                layers: 2,
                late: false,
            }),
            "early banded",
            MosaicPolicy::default(),
        );
    }
    println!("{}", "─".repeat(78));
}
