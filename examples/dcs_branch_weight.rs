//! Are the cut's branches worth the same, or does the weight concentrate?
//!
//! `cargo run --release --example dcs_branch_weight`

use quantsim::cutsim;
use quantsim::dcs::Dcs;
use quantsim::prelude::*;

fn fidelity(a: &[C64], b: &[C64]) -> f64 {
    let mut ov = C64::new(0.0, 0.0);
    let (mut na, mut nb) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        ov += x.conj() * *y;
        na += x.norm_sqr();
        nb += y.norm_sqr();
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    ov.norm_sqr() / (na * nb)
}

fn main() {
    println!("{}", "─".repeat(78));
    println!("BRANCH WEIGHT CONCENTRATION IN THE CUT SUM");
    println!("{}", "─".repeat(78));
    println!("  If every branch carries the same weight the sum cannot be truncated");
    println!("  and 2^k is the honest price. If the weight concentrates, a task that");
    println!("  needs only fidelity F needs only the branches carrying F of it.");
    println!();
    for n in [10, 12, 14, 16] {
        let d = Dcs::scaled(n);
        let c = d.circuit();
        let p = cutsim::plan(&c, n / 2).unwrap();
        let (w, exact) = cutsim::branch_weights_and_vector(&c, &p, usize::MAX).unwrap();
        let total: f64 = w.iter().sum();
        let mut sorted = w.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let top = sorted[0];
        let flat = total / w.len() as f64;
        println!(
            "  n={n:>3}  k={:>2}  branches={:<5}  heaviest/mean = {:.3}   max/min = {:.3}",
            p.crossings.len(),
            w.len(),
            top / flat,
            top / sorted.last().copied().unwrap_or(f64::NAN)
        );
        print!("        fidelity vs branches kept: ");
        for frac in [1, 2, 4, 8] {
            let keep = (w.len() / frac).max(1);
            let (_, trunc) = cutsim::branch_weights_and_vector(&c, &p, keep).unwrap();
            print!("{}/{} → {:.3}   ", keep, w.len(), fidelity(&exact, &trunc));
        }
        println!();
    }
    println!("{}", "─".repeat(78));
}
