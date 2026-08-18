//! Does the coupling **topology** change how separable the magic is?
//!
//! `dcs_separability` asks where the magic should *sit*: a `T`'s reach is
//! its cone, so the doping profile decides the factorization. This asks
//! the orthogonal question. Hold the width, the T-count, the T *sites*,
//! the bond count and the Clifford depth all fixed, and vary only
//! **which pairs the bonds join** — a nested block network
//! (`recursive`'s lattice) against random graphs of the same size, and
//! against a line as the trivially-local control.
//!
//! The `upembed` readout costs `Σ_components 2^{ancillas}`, so the
//! question is whether the topology keeps that sum small as Clifford
//! depth grows.
//!
//! `cargo run --release --example block_magic_separability`

use quantsim::gates::Pauli;
use quantsim::prelude::*;
use quantsim::recursive::{RecursiveLattice, Shape};
use quantsim::upembed::{cluster_readout, gadgetize};

/// H wall, bonds, magic on the given sites, then `rounds` further
/// Clifford layers over the same bonds — the layers are what let a
/// transported axis spread at all, since `cz` alone is diagonal and
/// commutes with the `Z` the gadget transports.
fn circuit(n: usize, bonds: &[(usize, usize)], magic: &[usize], rounds: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    for &(a, b) in bonds {
        c.gate("cz", [], [a, b]);
    }
    for &p in magic {
        c.t(p);
    }
    for _ in 0..rounds {
        for q in 0..n {
            c.h(q);
        }
        for &(a, b) in bonds {
            c.gate("cz", [], [a, b]);
        }
    }
    c
}

fn random_graph(n: usize, edges: usize, seed: u64) -> Vec<(usize, usize)> {
    let mut rng = Prng::new(seed);
    let mut out = Vec::with_capacity(edges);
    while out.len() < edges {
        let a = (rng.next_u64() % n as u64) as usize;
        let b = (rng.next_u64() % n as u64) as usize;
        if a != b {
            out.push((a.min(b), a.max(b)));
        }
    }
    out
}

fn row(label: &str, n: usize, bonds: &[(usize, usize)], magic: &[usize], rounds: usize) {
    let c = circuit(n, bonds, magic, rounds);
    let emb = gadgetize(&c).expect("gadgetize");
    match cluster_readout(&emb, &[(0usize, Pauli::Z)]) {
        Ok(r) => println!(
            "    {label:24} max cluster {:>3}   terms {:>9}   gain {:>9.0}x",
            r.max_cluster(),
            r.terms,
            r.factorization_gain()
        ),
        Err(e) => println!("    {label:24} refused: {e}"),
    }
}

fn main() {
    let lat = RecursiveLattice::nest(Shape::CUBE, 2).unwrap();
    let n = lat.width();
    let nested = lat.bond_pairs();
    // Magic on the lateral ports: the sites the blocks touch through.
    let mut magic: Vec<usize> = lat
        .bonds()
        .iter()
        .filter(|b| b.depth == 0)
        .flat_map(|b| [b.a, b.b])
        .collect();
    magic.sort_unstable();
    magic.dedup();
    let line: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();

    println!(
        "{n} qubits, {} T gates, {} bonds — only the bond SET varies",
        magic.len(),
        nested.len()
    );
    println!(
        "cost is Σ_components 2^ancillas; flat would be 2^{}\n",
        magic.len()
    );

    for rounds in [2usize, 4, 6, 8] {
        println!("  Clifford rounds after the magic: {rounds}");
        row("nested blocks", n, &nested, &magic, rounds);
        for seed in [7u64, 11, 13, 17] {
            let g = random_graph(n, nested.len(), seed);
            row(&format!("random graph seed {seed}"), n, &g, &magic, rounds);
        }
        row("line (local control)", n, &line, &magic, rounds);
        println!();
    }

    println!("The nested network's cluster size does not grow with depth — it");
    println!("oscillates, because the lattice's regularity makes transported");
    println!("lines re-cancel. Random graphs of identical size grow theirs to");
    println!("near-saturation. The line stays flat too, but a line has diameter");
    println!(
        "{} and no reach; the block network has this magic-locality at",
        n - 1
    );
    println!("diameter {}.", lat.topology().unwrap().diameter());
    println!("\nOne caveat, visible above: seed 13 stays flat like the structured");
    println!("case. The separation is a strong tendency across seeds, not a law.");
}
