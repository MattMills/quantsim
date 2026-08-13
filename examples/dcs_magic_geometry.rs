//! Where the magic actually sits in the IBM doped-Clifford-sampling
//! circuit — measured on the real 70-qubit, 468-T instance.
//!
//! The paper's hardness analysis prices four axes: entanglement (MPS
//! bond `2^30`), magic (stabilizer rank `2^185.5`), the hybrid of the
//! two (CAMPS, `2^0.41t` after the first `n` rotations disentangle), and
//! noise. All four are *scalar* resource counts.
//!
//! [`upembed`](quantsim::upembed) prices a different thing. Gadgetize
//! every `T` onto its own ancilla and the dynamics become Clifford; the
//! readout is then a subset sum over the `t` ancillas, and that sum
//! **factorizes over the connected components of the transported lines'
//! overlap graph**:
//!
//! ```text
//!   cost = Σ_components 2^{ancillas in component}
//! ```
//!
//! The exponent is neither the width nor the T-count. It is the
//! *magic-adjacency* of the observable — how the 468 pieces of magic
//! reach each other once the entanglement has been rotated away. A
//! circuit with 468 T gates whose magic falls into clusters of, say, 30
//! reads out at `16 · 2^30`, not `2^468`, no matter how entangled it is.
//!
//! Nobody has computed that number for this circuit, and it is cheap:
//! `t + 1` line transports and a union-find, with no `2^k` anywhere.
//! This asks.
//!
//! `cargo run --release --example dcs_magic_geometry`

use quantsim::dcs::Dcs;
use quantsim::gates::Pauli;
use quantsim::pathsum::Mask;
use quantsim::upembed::{cluster_spectrum, gadgetize};
use std::time::Instant;

/// GF(2) rank of the axes' symplectic vectors `(x ‖ z)`, by elimination
/// over shared bitsets. The Pauli group they generate has `2^rank`
/// elements, so this bounds every Pauli expansion of the magic.
fn symplectic_rank(axes: &[(Mask, Mask)], wires: usize) -> usize {
    // One row per axis: x bits, then z bits shifted past them.
    let mut rows: Vec<Mask> = axes
        .iter()
        .map(|(x, z)| {
            let mut r = x.clone();
            for i in z.iter() {
                r.set(wires + i);
            }
            r
        })
        .collect();
    let mut rank = 0;
    let mut pivots: Vec<(usize, Mask)> = Vec::new();
    for row in rows.iter_mut() {
        for (p, prow) in &pivots {
            if row.bit(*p) {
                *row = row.xor(prow);
            }
        }
        if let Some(p) = row.lowest() {
            pivots.push((p, row.clone()));
            rank += 1;
        }
    }
    rank
}

/// Connected components of the anticommutation graph: an edge wherever
/// two axes anticommute. Distinct components are symplectically
/// orthogonal and the evolution factorizes across them.
fn anticommutation_components(axes: &[(Mask, Mask)]) -> Vec<usize> {
    let n = axes.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let anti = (axes[i].1.and(&axes[j].0).count() + axes[i].0.and(&axes[j].1).count()) % 2
                == 1;
            if anti {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                parent[a] = b;
            }
        }
    }
    let mut sizes = std::collections::HashMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        *sizes.entry(r).or_insert(0usize) += 1;
    }
    let mut out: Vec<usize> = sizes.into_values().collect();
    out.sort_unstable_by(|a, b| b.cmp(a));
    out
}

fn rule() {
    println!("{}", "─".repeat(76));
}

fn report(label: &str, sizes: &[usize], t: usize) {
    let total: usize = sizes.iter().sum();
    let terms: f64 = sizes.iter().map(|&s| 2f64.powi(s as i32)).sum();
    let biggest = sizes.first().copied().unwrap_or(0);
    println!("  {label}");
    println!(
        "    components {:>5}   largest {:>4}   ancillas in components {total} of {t}",
        sizes.len(),
        biggest
    );
    println!(
        "    readout cost Σ2^cluster = 2^{:.1}   against a flat 2^{t} — a factor 2^{:.1}",
        terms.log2(),
        (t as f64) - terms.log2()
    );
    let head: Vec<String> = sizes.iter().take(12).map(|s| s.to_string()).collect();
    println!("    spectrum: [{}{}]", head.join(", "), if sizes.len() > 12 { ", …" } else { "" });
}

fn main() {
    rule();
    println!("MAGIC GEOMETRY OF THE 70-QUBIT DCS INSTANCE");
    rule();

    for n in [12, 20, 32, 48, 70] {
        let d = Dcs::scaled(n);
        let d = if n == 70 { Dcs::experiment() } else { d };
        let circuit = d.circuit();
        let t0 = Instant::now();
        let emb = match gadgetize(&circuit) {
            Ok(e) => e,
            Err(e) => {
                println!("n = {n}: gadgetize refused: {e}");
                continue;
            }
        };
        println!(
            "n = {n}, depth {}, {} CZ, {} T  →  {} wires, {} Clifford steps  ({:?})",
            d.depth,
            d.two_qubit_gates(),
            d.t_gates,
            emb.wires(),
            emb.steps().len(),
            t0.elapsed()
        );

        // Three observables, from the cheapest question to the one the
        // sampling task actually asks.
        let single = vec![(0usize, Pauli::Z)];
        let pair = vec![(0usize, Pauli::Z), (n / 2, Pauli::Z)];
        let full: Vec<(usize, Pauli)> = (0..n).map(|q| (q, Pauli::Z)).collect();

        // The two sharper partitions, on the rotation axes over the
        // *data* register — the Clifford ∘ Pauli-rotations form.
        let t2 = Instant::now();
        let axes = quantsim::dcs::rotation_axes(&circuit).expect("Clifford+T");
        let rank = symplectic_rank(&axes, n);
        let x_rank = symplectic_rank(
            &axes
                .iter()
                .map(|(x, _)| (x.clone(), Mask::zero()))
                .collect::<Vec<_>>(),
            n,
        );
        let anti = anticommutation_components(&axes);
        println!(
            "  {} rotation axes over {n} qubits   ({:?})",
            axes.len(),
            t2.elapsed()
        );
        println!(
            "    X-component rank  {x_rank:>4}  → kernel {:>4}   (paper §S9.3 reports {} − {n} for n=70)",
            axes.len() - x_rank,
            d.t_gates
        );
        println!(
            "    symplectic rank   {rank:>4}  of at most 2n = {}   → the Pauli group the magic",
            2 * n
        );
        println!(
            "                              spans has 2^{rank} elements, against 2^{} subsets",
            d.t_gates
        );
        println!(
            "    anticommutation   {:>4} component(s), largest {}",
            anti.len(),
            anti.first().copied().unwrap_or(0)
        );

        // Where the paper's T-count laws cross the width ceiling.
        // No n-qubit state has stabilizer rank above 2^n (the
        // computational basis is a stabilizer decomposition), and its
        // stabilizer extent obeys the same bound: Σ|c_x| ≤ √(2^n·Σ|c_x|²)
        // = 2^{n/2}, so ξ = (Σ|c_x|)² ≤ 2^n. Likewise no n-qubit
        // operator has more than 4^n Pauli coefficients — which is the
        // saturated symplectic rank above, reached and then stuck.
        for (name, alpha) in [("extent (cat)", 0.3963), ("QuiZX measured", 0.327)] {
            let crossover = n as f64 / alpha;
            println!(
                "    {name:<16} 2^{{{alpha}·t}} = 2^{:.0} at t = {}, but a {n}-qubit state caps at 2^{n};",
                alpha * d.t_gates as f64,
                d.t_gates
            );
            println!(
                "                     the law crosses that ceiling at t = {crossover:.0}, and this instance runs at t = {} ({:.1}× past it)",
                d.t_gates,
                d.t_gates as f64 / crossover
            );
        }

        for (label, ops) in [
            ("Z on qubit 0                ", single),
            ("Z on qubits 0 and n/2       ", pair),
            ("Z on every qubit (parity)   ", full),
        ] {
            let t1 = Instant::now();
            match cluster_spectrum(&emb, &ops) {
                Ok(sizes) => {
                    report(label, &sizes, d.t_gates);
                    println!("    measured in {:?}", t1.elapsed());
                }
                Err(e) => println!("  {label} refused: {e}"),
            }
        }
        rule();
    }
}
