//! Reproducing IBM/UChicago doped Clifford sampling (arXiv:2607.25941)
//! and measuring it against every technique in this crate.
//!
//! Run with `cargo run --release --example ibm_dcs`.

use quantsim::bounds::{advantage_scan, classify_law, resource_profile};
use quantsim::dcs::{self, Dcs};
use quantsim::pathsum::Mask;
use std::time::Duration;

/// Wall-clock allowed to any single reduction. Every probe in this file
/// is bounded: an over-scale instance reports a measured refusal.
const BUDGET: Duration = Duration::from_secs(20);

fn rule() {
    println!("{}", "─".repeat(78));
}

fn main() {
    // ── 1. The system, as specified ─────────────────────────────────
    rule();
    println!("1. THE SYSTEM  (arXiv:2607.25941, IBM + UChicago, July 2026)");
    rule();
    let e = Dcs::experiment();
    let census = e.census();
    println!(
        "  logical qubits          {:>8}   (paper {})",
        census.qubits,
        dcs::EXPERIMENT_QUBITS
    );
    println!("  brickwork CZ depth      {:>8}", census.depth);
    println!(
        "  two-qubit gates         {:>8}   (paper {})   {}",
        census.two_qubit_gates,
        dcs::EXPERIMENT_TWO_QUBIT_GATES,
        if census.two_qubit_gates == dcs::EXPERIMENT_TWO_QUBIT_GATES {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    println!(
        "  T gates                 {:>8}   (paper {})   {}",
        census.t_gates,
        dcs::EXPERIMENT_T_GATES,
        if census.t_gates == dcs::EXPERIMENT_T_GATES {
            "MATCH"
        } else {
            "MISMATCH"
        }
    );
    println!(
        "  doping rate             {:>8.4}   T per two-qubit gate (paper quotes r = 0.19)",
        e.doping_rate()
    );
    println!("  total operations        {:>8}", census.ops);
    println!(
        "  Hadamard walls          {:>8}   ceiling on h*: only a wall allocates a path variable",
        census.walls
    );
    println!(
        "  candidate doping sites  {:>8}   wires immediately following a CZ",
        e.dope_sites().len()
    );
    println!(
        "  physical qubits         {:>8}   = {} logical + {} spacetime-code ancillas (not modelled)",
        dcs::EXPERIMENT_PHYSICAL_QUBITS,
        dcs::EXPERIMENT_QUBITS,
        dcs::EXPERIMENT_ANCILLAS
    );

    // ── 2. The Clifford skeleton at full scale ──────────────────────
    rule();
    println!("2. THE CLIFFORD SKELETON AT FULL SCALE  (70 qubits, depth 70)");
    rule();
    println!("  The experiment's own trusted baseline. h* = 0 would mean this");
    println!("  crate holds the undoped 70-qubit state with no tableau at all.");
    println!();
    println!("   n   depth      ops    walls     h*    terms    splits    reduce      status");
    for n in [8, 12, 16, 20, 24, 28, 32, 48, 70] {
        let d = Dcs::scaled(n).with_t(0);
        let c = d.skeleton();
        match dcs::probe(&c, BUDGET) {
            Ok(p) => println!(
                "  {n:>3}  {:>6}  {:>7}  {:>7}  {:>5}  {:>7}  {:>8}  {:>8.2?}   {}",
                d.depth,
                c.len(),
                p.walls,
                p.h_star,
                p.terms,
                p.splits,
                p.elapsed,
                if p.cut_short {
                    "TIMEOUT (h* is an over-estimate)"
                } else {
                    "reduced dry"
                }
            ),
            Err(err) => println!("  {n:>3}  refused: {err}"),
        }
    }

    // ── 3. What doping costs ────────────────────────────────────────
    rule();
    println!("3. WHAT DOPING COSTS  (h* against T count, skeleton held byte-identical)");
    rule();
    for n in [8, 12, 16] {
        let base = Dcs::scaled(n);
        let full = base.t_gates;
        println!(
            "  n = {n}, depth {}, {} CZ, experiment-rate doping would be t = {full}",
            base.depth,
            base.two_qubit_gates()
        );
        println!("       t     h*    h*/t   h*/walls    terms    align    reduce");
        for t in [0, 1, 2, 4, 8, 16, 32, full] {
            if t > base.dope_sites().len() {
                continue;
            }
            let c = base.with_t(t).circuit();
            match dcs::probe(&c, BUDGET) {
                Ok(p) => println!(
                    "    {t:>4}  {:>5}  {:>6}  {:>9.3}  {:>7}  {:>7}  {:>8.2?}{}",
                    p.h_star,
                    if t > 0 {
                        format!("{:.3}", p.h_star as f64 / t as f64)
                    } else {
                        "—".into()
                    },
                    p.h_star as f64 / p.walls as f64,
                    p.terms,
                    p.alignment_stalls,
                    p.elapsed,
                    if p.cut_short { "  TIMEOUT" } else { "" }
                ),
                Err(err) => println!("    {t:>4}  refused: {err}"),
            }
        }
        println!();
    }

    // ── 4. The family scaled, both resources moving ─────────────────
    rule();
    println!("4. THE FAMILY SCALED  (depth = n and t at the experiment's rate, as run)");
    rule();
    println!("     n    CZ      t   walls     h*    h*/t   h*/n    reduce      status");
    let mut law_n: Vec<usize> = Vec::new();
    let mut law_h: Vec<usize> = Vec::new();
    let mut law_t: Vec<usize> = Vec::new();
    for n in [6, 8, 10, 12, 14, 16, 18, 20] {
        let d = Dcs::scaled(n);
        let c = d.circuit();
        match dcs::probe(&c, BUDGET) {
            Ok(p) => {
                println!(
                    "  {n:>4}  {:>4}  {:>5}  {:>6}  {:>5}  {:>6.3}  {:>5.2}  {:>8.2?}   {}",
                    d.two_qubit_gates(),
                    d.t_gates,
                    p.walls,
                    p.h_star,
                    p.h_star as f64 / d.t_gates as f64,
                    p.h_star as f64 / n as f64,
                    p.elapsed,
                    if p.cut_short {
                        "TIMEOUT"
                    } else {
                        "reduced dry"
                    }
                );
                if !p.cut_short {
                    law_n.push(n);
                    law_h.push(p.h_star.max(1));
                    law_t.push(d.t_gates);
                }
            }
            Err(err) => println!("  {n:>4}  refused: {err}"),
        }
    }
    if law_n.len() >= 3 {
        println!();
        println!("  h* against width  : {:?}", classify_law(&law_n, &law_h));
        println!("  h* against T count: {:?}", classify_law(&law_t, &law_h));
    }

    // ── 5. Every other representation ───────────────────────────────
    rule();
    println!("5. EVERY OTHER REPRESENTATION  (the boundary atlas on this family)");
    rule();
    let n = 12;
    let prof = resource_profile(&Dcs::scaled(n).circuit());
    println!(
        "  n = {n}, depth {n}, {} CZ, {} T",
        Dcs::scaled(n).two_qubit_gates(),
        Dcs::scaled(n).t_gates
    );
    println!("   axis             bytes      time   exact   driving parameter");
    for a in &prof.axes {
        match (a.cost, a.nanos) {
            (Some(b), Some(t)) => println!(
                "   {:<14} {:>8}  {:>8.2?}   {:<5}  {}",
                a.axis,
                b,
                Duration::from_nanos(t as u64),
                a.exact,
                a.parameter
            ),
            _ => println!(
                "   {:<14}   refused: {}",
                a.axis,
                a.note.as_deref().unwrap_or("—")
            ),
        }
    }

    rule();
    println!("6. THE VERDICT ACROSS THE FAMILY");
    rule();
    let scan = advantage_scan("dcs", |n| Dcs::scaled(n).circuit(), &[6, 8, 10, 12]);
    println!("  sizes {:?}", scan.sizes);
    for a in &scan.axes {
        println!(
            "   {:<14} memory {:<28} time {:<28} {}",
            a.axis,
            format!("{:?}", a.law),
            format!("{:?}", a.time_law),
            if a.certifies_classical() {
                "CERTIFIES CLASSICAL"
            } else {
                ""
            }
        );
    }
    println!("  verdict: {:?}", scan.verdict);

    // ── 7. One amplitude of the real thing ──────────────────────────
    rule();
    println!("7. ONE AMPLITUDE OF THE REAL 70-QUBIT SKELETON");
    rule();
    let skel = Dcs::experiment().with_t(0);
    match dcs::probe(&skel.skeleton(), Duration::from_secs(120)) {
        Ok(p) if !p.cut_short => {
            let ps = quantsim::pathsum::PathSum::from_circuit(&skel.skeleton()).unwrap();
            let mut bits = Mask::zero();
            for q in (0..70).step_by(3) {
                bits.set(q);
            }
            let amp = ps.amplitude_mask(&bits);
            println!("  h* = {}, amplitude = {amp:?}", p.h_star);
        }
        Ok(p) => println!(
            "  reduction did not finish inside 120 s (h* ≤ {} at cut-off, {} walls)",
            p.h_star, p.walls
        ),
        Err(err) => println!("  refused: {err}"),
    }
    rule();
}
