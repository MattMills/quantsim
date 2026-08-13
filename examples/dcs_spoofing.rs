//! Reproducing the paper's own anti-spoofing analysis on the doped
//! Clifford sampling family (arXiv:2607.25941 §S9.1 and §S9.5).
//!
//! Both arguments are *falsifiable predictions* about small instances,
//! which is what makes them worth re-running here rather than quoting:
//!
//! * **§S9.5, Cliffordization.** Swapping a `T` for an `S` — the nearest
//!   Clifford — keeps a process fidelity of `cos²(π/8) = 0.8536`. If the
//!   swaps behaved independently the state fidelity after `k` of them
//!   would be `0.8536^k`, which crosses the experiment's measured 0.32
//!   between `k = 7` and `k = 8`. The paper reports exactly that
//!   ceiling: "only a maximum of 7 gates can be converted into `S` gates
//!   before the fidelity drops below the experimentally measured value".
//!
//! * **§S9.1, flat Schmidt spectra.** A truncated MPS at bond `χ` on a
//!   state of Schmidt rank `R` obeys `F < χ/R`. For Haar-random circuits
//!   the singular values decay exponentially and truncation buys fidelity
//!   cheaply; for *stabilizer* states the spectrum is flat and the bound
//!   is tight, so fidelity is **linear** in the bond budget. That is the
//!   difference between a family MPS can approximate and one it cannot,
//!   and it is measurable at 12 qubits.
//!
//! `cargo run --release --example dcs_spoofing`

use quantsim::dcs::{self, Dcs};
use quantsim::prelude::*;
use quantsim::sampling::mps_spoof_curve;

fn rule() {
    println!("{}", "─".repeat(74));
}

/// `|⟨a|b⟩|²` between two states of the same width.
fn fidelity(a: &dyn Backend<C64>, b: &dyn Backend<C64>) -> f64 {
    let mut acc = C64::new(0.0, 0.0);
    a.for_each_nonzero(&mut |i, amp| {
        acc += amp.conj() * b.amplitude(i);
    });
    acc.norm_sqr()
}

fn main() {
    let sim = Simulator::<C64>::new();

    rule();
    println!("A. CLIFFORDIZATION SPOOFING  (paper §S9.5)");
    rule();
    println!(
        "  Swap k of the T gates for S. Prediction: fidelity ≈ {:.4}^k,",
        dcs::CLIFFORDIZED_T_FIDELITY
    );
    println!("  crossing the experiment's measured 0.32 between k = 7 and k = 8.");
    println!();
    for n in [12, 14] {
        let d = Dcs::scaled(n);
        let full = d.circuit();
        let truth = sim.run(&full).unwrap();
        println!("  n = {n}, {} T gates in the instance", d.t_gates);
        println!("     k   measured F   {:.4}^k   ratio", dcs::CLIFFORDIZED_T_FIDELITY);
        for k in 0..=d.t_gates.min(12) {
            let spoof = sim.run(&dcs::cliffordize(&full, k, 99)).unwrap();
            let f = fidelity(truth.as_ref(), spoof.as_ref());
            let pred = dcs::CLIFFORDIZED_T_FIDELITY.powi(k as i32);
            println!(
                "   {k:>3}   {f:>10.4}   {pred:>8.4}   {:>6.3}{}",
                if pred > 0.0 { f / pred } else { f64::NAN },
                if f < 0.32 { "   < 0.32" } else { "" }
            );
        }
        println!();
    }

    rule();
    println!("B. TRUNCATED-MPS SPOOFING  (paper §S9.1: flat spectrum ⇒ F < χ/R)");
    rule();
    let n = 12;
    let reg = GateRegistry::<C64>::standard();
    let caps: Vec<usize> = vec![1, 2, 4, 8, 16, 32, 64, 128];
    println!("  Same brickwork, same {n} CZ layers; the only difference is whether the");
    println!("  single-qubit layer is Clifford. Max bond at n = {n} is 2^{} = {}.", n / 2, 1 << (n / 2));
    println!();
    for (label, circuit) in [
        ("DCS doped Clifford ", Dcs::scaled(n).circuit()),
        ("Haar-random control", Dcs::scaled(n).haar_control()),
    ] {
        let pts = mps_spoof_curve(&circuit, &reg, &caps, 4000, 5).unwrap();
        println!("  {label}  (n = {n})");
        println!("      χ   peak   normalized XEB   bytes");
        for p in &pts {
            println!(
                "   {:>4}   {:>4}   {:>14}   {:>7}",
                p.max_bond,
                p.peak_bond,
                p.score
                    .normalized
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "—".into()),
                p.memory
            );
        }
        // The discriminator is where the spoofer first reaches the
        // experiment's own measured fidelity. A decaying spectrum gets
        // there at small χ; a flat one only at full rank, which is
        // exponential in the width.
        let full_rank = 1usize << (n / 2);
        match pts
            .iter()
            .find(|p| p.score.normalized.unwrap_or(0.0) >= 0.32)
        {
            Some(p) => println!(
                "   first χ reaching the experiment's 0.32 fidelity: {}  = {:.0}% of full rank {}",
                p.max_bond,
                100.0 * p.max_bond as f64 / full_rank as f64,
                full_rank
            ),
            None => println!("   never reached 0.32 within the sweep"),
        }
        println!();
    }
    rule();
}
