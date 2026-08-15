//! Does *where* the magic sits change how separable it is?
//!
//! A `T` gate's reach is its forward light cone. On a chain of depth `D`
//! a cone opening at layer `L` is `2(D − L)` qubits wide at the output,
//! so magic placed late cannot reach far. Every factorization in this
//! crate keys off exactly that: the `upembed` readout factorizes over
//! connected components of the transported lines' *overlap* graph, and
//! overlap is cone overlap.
//!
//! So separability is not a fixed property of "468 T gates". It is a
//! property of the doping profile, and the profile is something the
//! construction chooses.
//!
//! `cargo run --release --example dcs_separability`

use quantsim::dcs::{Dcs, Doping};
use quantsim::gates::Pauli;
use quantsim::pathsum::Mask;
use quantsim::upembed::{cluster_spectrum, gadgetize, magic_axes};
use std::time::Instant;

/// Components of the magic's own overlap graph, restricted to the data
/// register: two `T` gates are linked when their transported axes touch
/// a common qubit. This is the magic's *intrinsic* separability, with
/// the observable left out — a weight-1 observable at the end of a
/// depth-70 circuit has a backward cone covering everything, so
/// including it glues every component together and measures the
/// observable's reach rather than the magic's.
fn magic_components(axes: &[(Mask, Mask)], n: usize) -> Vec<usize> {
    let supports: Vec<Mask> = axes
        .iter()
        .map(|(x, z)| {
            let mut m = Mask::zero();
            for q in 0..n {
                if x.bit(q) || z.bit(q) {
                    m.set(q);
                }
            }
            m
        })
        .collect();
    let m = supports.len();
    let mut parent: Vec<usize> = (0..m).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for i in 0..m {
        for j in (i + 1)..m {
            if supports[i].intersects(&supports[j]) {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                parent[a] = b;
            }
        }
    }
    let mut sizes = std::collections::HashMap::new();
    for i in 0..m {
        let r = find(&mut parent, i);
        *sizes.entry(r).or_insert(0usize) += 1;
    }
    let mut out: Vec<usize> = sizes.into_values().collect();
    out.sort_unstable_by(|a, b| b.cmp(a));
    out
}

fn cost_log2(sizes: &[usize]) -> f64 {
    sizes
        .iter()
        .map(|&s| 2f64.powi(s as i32))
        .sum::<f64>()
        .log2()
}

fn main() {
    println!("{}", "─".repeat(78));
    println!("MAGIC SEPARABILITY vs DOPING PROFILE   (70 qubits, depth 70, 468 T)");
    println!("{}", "─".repeat(78));
    println!("  Same skeleton, same 468 T gates, same observable. Only the layers");
    println!("  the magic is allowed to occupy change.");
    println!();
    println!("  profile                 components   largest   Σ2^cluster   measured");
    let e = Dcs::experiment();
    let obs = vec![(0usize, Pauli::Z)];
    let mut rows = Vec::new();
    for (label, doping) in [
        ("uniform                ", Doping::Uniform),
        ("last 32 layers         ", Doping::Late { layers: 32 }),
        ("last 16 layers         ", Doping::Late { layers: 16 }),
        ("last 8 layers          ", Doping::Late { layers: 8 }),
        ("last 4 layers          ", Doping::Late { layers: 4 }),
        ("last 2 layers          ", Doping::Late { layers: 2 }),
        ("first 8 layers         ", Doping::Early { layers: 8 }),
        ("first 2 layers         ", Doping::Early { layers: 2 }),
        // Cones from the last L layers are 2L wide, so a gap wider than
        // that cannot be bridged and the component must split.
        (
            "bands 8, gap 8, last 2 ",
            Doping::Banded {
                width: 8,
                gap: 8,
                layers: 2,
                late: true,
            },
        ),
        (
            "bands 8, gap 8, FIRST 2",
            Doping::Banded {
                width: 8,
                gap: 8,
                layers: 2,
                late: false,
            },
        ),
        (
            "bands 8, gap 8, FIRST 4",
            Doping::Banded {
                width: 8,
                gap: 8,
                layers: 4,
                late: false,
            },
        ),
        (
            "bands 4, gap 12, FIRST 4",
            Doping::Banded {
                width: 4,
                gap: 12,
                layers: 4,
                late: false,
            },
        ),
        (
            "bands 2, gap 16, FIRST 6",
            Doping::Banded {
                width: 2,
                gap: 16,
                layers: 6,
                late: false,
            },
        ),
    ] {
        let d = e.with_doping(doping);
        let c = d.circuit();
        let t0 = Instant::now();
        let emb = gadgetize(&c).unwrap();
        let sizes = cluster_spectrum(&emb, &obs).unwrap();
        let placed = c
            .ops()
            .iter()
            .filter(|op| matches!(op, quantsim::circuit::Op::Named { name, .. } if name == "t"))
            .count();
        let mag = magic_components(&magic_axes(&emb), 70);
        println!(
            "  {label}  {:>10}   {:>7}   2^{:<9.1}   {:>8.2?}   ({placed} T)",
            sizes.len(),
            sizes.first().copied().unwrap_or(0),
            cost_log2(&sizes),
            t0.elapsed()
        );
        println!(
            "                            magic alone: {:>4} component(s), largest {:>4}, Σ2^c = 2^{:.1}",
            mag.len(),
            mag.first().copied().unwrap_or(0),
            cost_log2(&mag)
        );
        rows.push((label, mag));
    }
    println!();
    println!("  Magic-only spectra (descending, first 16):");
    for (label, sizes) in &rows {
        let head: Vec<String> = sizes.iter().take(16).map(|s| s.to_string()).collect();
        println!(
            "    {label} [{}{}]",
            head.join(", "),
            if sizes.len() > 16 { ", …" } else { "" }
        );
    }
    println!("{}", "─".repeat(78));
}
