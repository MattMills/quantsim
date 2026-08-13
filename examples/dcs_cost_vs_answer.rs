//! What the full-scale magic-geometry measurements actually computed —
//! and, just as importantly, what they did not.
//!
//! Reporting "one cluster of 468, measured in 68 ms" invites the obvious
//! objection: if something about a 70-qubit, 468-T circuit was computed
//! in 68 ms, surely a piece of the hard problem was solved. It was not,
//! and the distinction is the whole design of the up-embedded readout:
//!
//! * **Transport is Clifford.** Gadgetizing every `T` onto its own
//!   ancilla makes the dynamics Clifford, and conjugating a Pauli line
//!   through a Clifford circuit is Gottesman–Knill — `O(steps)` per
//!   line, exact, no amplitudes anywhere. 469 lines through 24 987 steps
//!   is ~12 M bitset operations. That is the 68 ms.
//! * **Contraction is not.** The answer is a subset sum over the
//!   ancillas, `Σ_components 2^{ancillas}` boundary terms. On this
//!   circuit that is `2^468`, and nothing ran it.
//!
//! So the cheap part computes the *shape* of the exponential and the
//! expensive part would compute its *value*. The shape is a structural
//! property of the Clifford frame, which is why it is polynomial; the
//! value is the sampling problem, which is why it is not.
//!
//! This example proves both halves rather than asserting them:
//!
//! 1. at a width where `2^cluster` is affordable, the same machinery
//!    computes the expectation and it is checked against the dense
//!    state — so the method is genuinely computing the right quantity,
//!    not producing a number that happens to be cheap;
//! 2. at the experiment's width, the estimate runs and the readout is
//!    priced, so the gap between "knows the cost" and "pays it" is
//!    visible in one table.
//!
//! `cargo run --release --example dcs_cost_vs_answer`

use quantsim::dcs::Dcs;
use quantsim::gates::Pauli;
use quantsim::prelude::*;
use quantsim::rng::Prng;
use quantsim::upembed::{cluster_spectrum, expectation, gadgetize};
use std::time::Instant;

fn rule() {
    println!("{}", "─".repeat(76));
}

/// `⟨ψ|P|ψ⟩` for a Pauli string, from the dense state — the independent
/// ground truth. Built by applying `P` to the state and taking the
/// overlap, so it exercises no shared code with the up-embedded path.
fn dense_pauli(circuit: &Circuit<C64>, ops: &[(usize, Pauli)]) -> f64 {
    let sim = Simulator::<C64>::new();
    let psi = sim.run(circuit).unwrap();
    let mut with_p: Circuit<C64> = circuit.clone();
    for &(q, p) in ops {
        match p {
            Pauli::I => {}
            Pauli::X => {
                with_p.x(q);
            }
            Pauli::Y => {
                with_p.y(q);
            }
            Pauli::Z => {
                with_p.z(q);
            }
        }
    }
    let p_psi = sim.run(&with_p).unwrap();
    let mut acc = C64::new(0.0, 0.0);
    psi.for_each_nonzero(&mut |i, amp| {
        acc += amp.conj() * p_psi.amplitude(i);
    });
    acc.re
}

fn main() {
    rule();
    println!("1. THE MACHINERY COMPUTES THE RIGHT QUANTITY  (where it can afford to)");
    rule();
    println!("  Same code path as the 70-qubit run, at widths where the subset");
    println!("  sum is affordable. Checked against the dense state vector.");
    println!();
    println!("  A doped graph state kills most Pauli expectations outright — only");
    println!("  strings related to the stabilizer group survive — so the probes are");
    println!("  drawn at random and the ones with signal are shown.");
    println!();
    println!("     n     t   observable       clusters   terms         upembed ⟨P⟩             dense ⟨P⟩   |Δ|");
    let mut worst = 0.0f64;
    let mut checked = 0usize;
    let mut nonzero = 0usize;
    for n in [6, 8, 10, 12] {
        let d = Dcs::scaled(n);
        let circuit = d.circuit();
        let mut rng = Prng::new(0xDC5 ^ n as u64);
        let mut shown = 0;
        for _ in 0..400 {
            let ops: Vec<(usize, Pauli)> = (0..n)
                .map(|q| {
                    (
                        q,
                        match rng.next_u64() % 4 {
                            0 => Pauli::I,
                            1 => Pauli::X,
                            2 => Pauli::Y,
                            _ => Pauli::Z,
                        },
                    )
                })
                .filter(|(_, p)| !matches!(p, Pauli::I))
                .collect();
            if ops.is_empty() {
                continue;
            }
            let r = expectation(&circuit, &ops).unwrap();
            let truth = dense_pauli(&circuit, &ops);
            worst = worst.max((r.value - truth).abs());
            checked += 1;
            if truth.abs() > 1e-6 {
                nonzero += 1;
                if shown < 3 {
                    shown += 1;
                    let label: String = ops
                        .iter()
                        .map(|(q, p)| format!("{}{q}", match p {
                            Pauli::X => "X",
                            Pauli::Y => "Y",
                            Pauli::Z => "Z",
                            Pauli::I => "I",
                        }))
                        .collect::<Vec<_>>()
                        .join("");
                    let short: String = label.chars().take(14).collect();
                    println!(
                        "  {n:>4}  {:>4}   {short:<15}  {:>8?}   {:>5}   {:>17.13}   {:>17.13}   {:.1e}",
                        d.t_gates,
                        r.clusters,
                        r.terms,
                        r.value,
                        truth,
                        (r.value - truth).abs()
                    );
                }
            }
        }
    }
    println!();
    println!(
        "  {checked} random Pauli strings checked against dense, {nonzero} with nonzero signal."
    );
    println!("  Worst deviation anywhere: {worst:.2e}");

    rule();
    println!("2. WHAT RAN AT 70 QUBITS, AND WHAT DID NOT");
    rule();
    let d = Dcs::experiment();
    let circuit = d.circuit();
    let t0 = Instant::now();
    let emb = gadgetize(&circuit).unwrap();
    let gadget_time = t0.elapsed();
    let t1 = Instant::now();
    let sizes = cluster_spectrum(&emb, &[(0, Pauli::Z)]).unwrap();
    let estimate_time = t1.elapsed();

    let lines = emb.magic() + 1;
    let steps = emb.steps().len();
    let bit_ops = lines as u128 * steps as u128;
    let contractions: f64 = sizes.iter().map(|&s| 2f64.powi(s as i32)).sum();

    println!("  RAN — polynomial, exact, Clifford:");
    println!("    gadgetize the circuit                    {gadget_time:?}");
    println!(
        "    transport {lines} Pauli lines through {steps} Clifford steps",
    );
    println!(
        "      = {:.1} M line-step updates                 {estimate_time:?}",
        bit_ops as f64 / 1e6
    );
    println!(
        "    union-find over {} line pairs",
        lines * (lines - 1) / 2
    );
    println!("    → cluster spectrum {sizes:?}");
    println!();
    println!("  DID NOT RUN — this is the sampling problem:");
    println!(
        "    boundary contractions the readout would need: Σ2^cluster = 2^{:.0}",
        contractions.log2()
    );
    println!(
        "    at 10^9 contractions/s that is 10^{:.0} seconds",
        contractions.log10() - 9.0
    );
    println!();
    println!("  The first block is Gottesman–Knill on the Clifford frame: it says");
    println!("  how the 468 pieces of magic reach each other, which is a property");
    println!("  of the frame and not of the amplitudes. The second block is the");
    println!("  amplitudes. Knowing the exponent is cheap; paying it is not.");
    rule();
}
