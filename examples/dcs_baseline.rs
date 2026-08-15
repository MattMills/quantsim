//! How far up the DCS Clifford skeleton the tableau-free path sum
//! reaches, and what it costs.
//!
//! The skeleton is the experiment's *trusted classical baseline*: the
//! fidelity certificate in arXiv:2607.25941 is built by comparing the
//! doped circuit against the undoped one, which is a stabilizer circuit
//! and therefore efficiently simulable. `h*` is zero for every Clifford
//! circuit at any width, so this crate holds that baseline with no
//! tableau — the only question is what reduction *costs* to get there,
//! and that is what this sweep measures.
//!
//! Long-running by design. `cargo run --release --example dcs_baseline`.

use quantsim::dcs::{self, Dcs};
use quantsim::pathsum::{Mask, PathSum};
use std::time::{Duration, Instant};

fn main() {
    let budget: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4 * 3600);
    println!("per-instance budget: {budget} s\n");
    println!("    n      ops    walls     h*    terms     splits        reduce   status");
    let mut sizes: Vec<usize> = vec![24, 28, 32, 36, 40, 48, 56, 64, 70];
    sizes.retain(|_| true);
    for n in sizes {
        let d = Dcs::scaled(n).with_t(0);
        let c = d.skeleton();
        let p = match dcs::probe(&c, Duration::from_secs(budget)) {
            Ok(p) => p,
            Err(e) => {
                println!("  {n:>3}  refused: {e}");
                continue;
            }
        };
        println!(
            "  {n:>3}  {:>7}  {:>7}  {:>5}  {:>7}  {:>9}  {:>12.1?}   {}",
            c.len(),
            p.walls,
            p.h_star,
            p.terms,
            p.splits,
            p.elapsed,
            if p.cut_short {
                "TIMEOUT — h* is an over-estimate"
            } else {
                "reduced dry"
            }
        );
        if p.cut_short {
            println!("\n  stopped: the sweep does not reach past width {n} inside the budget.");
            return;
        }
        if p.h_star == 0 && n == dcs::EXPERIMENT_QUBITS {
            // The payoff: an amplitude of the experiment's own baseline.
            let t0 = Instant::now();
            let ps = PathSum::from_circuit(&c).expect("skeleton reduces");
            let mut bits = Mask::zero();
            for q in (0..n).step_by(3) {
                bits.set(q);
            }
            let amp = ps.amplitude_mask(&bits);
            println!(
                "\n  amplitude of the 70-qubit depth-70 skeleton at |x⟩ = every third qubit set:"
            );
            println!("    ⟨x|ψ⟩ = {amp:?}");
            println!(
                "    |⟨x|ψ⟩|² · 2^70 = {:.6}",
                amp.norm_sqr() * 2f64.powi(70)
            );
            println!("    (second reduction {:?})", t0.elapsed());
        }
    }
}
