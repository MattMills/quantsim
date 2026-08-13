//! What a `discovery::Insertion` looks like on a DCS circuit, and what
//! `verify_transparent` actually reports for one.
//!
//! Three threads, all spliced into `Dcs::scaled(n)`:
//!
//! * `null`      — `X·X` at the same point: individually identity, the
//!                 control that the harness itself costs nothing;
//! * `z-thread`  — `Z_q` before a `CZ` layer and `Z_q` after it. Neither
//!                 half is identity; the pair is, because everything in
//!                 between (`CZ`, `T`) commutes with `Z`;
//! * `x-thread`  — `X_q` before the `CZ` layer, and `X_q·Z_p` after it,
//!                 `p` the brickwork partner. The signal is *transformed*
//!                 in flight — `CZ·X_q·CZ = X_q Z_p` — so the removal is
//!                 not a copy of the insertion. This is the shape the
//!                 module means by a signal thread.
//!
//! `verify_transparent` runs both circuits on the **dense** backend, so
//! the width here is bounded by `2^n` amplitudes, not by anything about
//! the thread.

use quantsim::circuit::{Circuit, Op};
use quantsim::dcs::Dcs;
use quantsim::discovery::{verify_transparent, Insertion};
use quantsim::scalar::C64;
use quantsim::sim::Simulator;

/// `(first index, one past last)` of each brickwork layer's `CZ`/`T` run.
fn cz_runs(circuit: &Circuit<C64>) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start: Option<usize> = None;
    for (i, op) in circuit.ops().iter().enumerate() {
        let Op::Named { name, .. } = op else { continue };
        let is_cz_run = name == "cz" || name == "t";
        match (is_cz_run, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                runs.push((s, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        runs.push((s, circuit.ops().len()));
    }
    runs
}

/// The brickwork partner of `q` inside a run, and whether a `T` lands on
/// `q` inside it (which would break an `X` thread: `T X T†` is not Pauli).
fn partner_and_clean(circuit: &Circuit<C64>, run: (usize, usize), q: usize) -> (Option<usize>, bool) {
    let mut partner = None;
    let mut clean = true;
    for op in &circuit.ops()[run.0..run.1] {
        let Op::Named { name, qubits, .. } = op else {
            continue;
        };
        if name == "cz" && qubits.contains(&q) {
            partner = Some(if qubits[0] == q { qubits[1] } else { qubits[0] });
        }
        if name == "t" && qubits[0] == q {
            clean = false;
        }
    }
    (partner, clean)
}

fn main() {
    let sim = Simulator::<C64>::new();
    println!("SIGNAL THREADS SPLICED INTO A DCS CIRCUIT");
    println!(
        "  peak = FactoredState::peak() = (max total stored amplitudes, max single-factor qubits)"
    );
    println!();
    println!(
        "   n  thread     | ops  spliced | final deviation  weight drift  transparent | \
         peak without   peak with"
    );
    println!("  {}", "-".repeat(110));

    for n in [8usize, 12, 16, 20] {
        let d = Dcs::scaled(n);
        let base = d.circuit();
        let runs = cz_runs(&base);
        // A layer in the middle of the circuit, so the thread is not a
        // boundary special case.
        let run = runs[runs.len() / 2];

        // A qubit with a partner in this layer and no T on it inside.
        let pick = (0..n).find_map(|q| match partner_and_clean(&base, run, q) {
            (Some(p), true) => Some((q, p)),
            _ => None,
        });
        let Some((q, p)) = pick else {
            println!("  {n:>2}  no clean qubit in the chosen layer");
            continue;
        };

        let threads = vec![
            Insertion::<C64>::new("null")
                .gate(run.0, "x", vec![], vec![q])
                .gate(run.0, "x", vec![], vec![q]),
            Insertion::<C64>::new("z-thread")
                .gate(run.0, "z", vec![], vec![q])
                .gate(run.1, "z", vec![], vec![q]),
            Insertion::<C64>::new("x-thread")
                .gate(run.0, "x", vec![], vec![q])
                .gate(run.1, "x", vec![], vec![q])
                .gate(run.1, "z", vec![], vec![p]),
        ];

        for thread in &threads {
            let spliced = thread.splice(&base);
            match verify_transparent(&sim, &base, thread, 1e-9) {
                Ok(r) => println!(
                    "  {:>2}  {:<10} | {:>3}  {:>7} | {:>15.2e}  {:>12.2e}  {:>11} | \
                     {:>4} / {:<6}  {:>4} / {:<6}",
                    n,
                    r.name,
                    base.len(),
                    spliced.len(),
                    r.final_deviation,
                    r.weight_drift,
                    r.transparent,
                    r.peak_without.0,
                    r.peak_without.1,
                    r.peak_with.0,
                    r.peak_with.1,
                ),
                Err(e) => println!("  {n:>2}  {:<10} | refused: {e}", thread.name),
            }
        }
        println!(
            "      (layer run = ops {}..{}, qubit {q} paired with {p})",
            run.0, run.1
        );
    }
}
