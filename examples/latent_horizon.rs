//! The latent past and the latent future, drawn.
//!
//! A light cone is a set of cells. What the Heisenberg walk actually
//! carries is a *field* over those cells, and the field is what this
//! renders — beside its dual, the forward state's local commitment, on
//! the same sheet.
//!
//! Run with `--json` to emit the sheet as JSON instead of ASCII.

use quantsim::heisenberg::*;
use quantsim::horizon::{self, Config, Forward, Horizon};
use quantsim::prelude::*;

const SHADES: [char; 10] = [' ', '·', ':', '-', '=', '+', '*', '#', '%', '@'];

fn shade(v: f64) -> char {
    SHADES[((v.clamp(0.0, 1.0) * 9.0).round() as usize).min(9)]
}

/// A field map: rows are qubits, columns are circuit positions.
fn draw(h: &Horizon, title: &str, cell: impl Fn(usize, usize) -> f64) {
    println!("  {title}");
    print!("        ");
    for (c, &g) in h.cuts.iter().enumerate() {
        print!("{}", if c % 4 == 0 { char::from_digit((g * 10 / h.gates.max(1)) as u32 % 10, 10).unwrap() } else { ' ' });
    }
    println!("   gate ⟶");
    for q in 0..h.qubits {
        print!("   q{q:<2}  ");
        for c in 0..h.cuts.len() {
            print!("{}", shade(cell(c, q)));
        }
        println!();
    }
    println!("        {}", "─".repeat(h.cuts.len()));
}

fn bar(v: f64, max: f64, width: usize) -> String {
    let n = if max > 0.0 {
        ((v / max) * width as f64).round() as usize
    } else {
        0
    };
    format!("{}{}", "█".repeat(n.min(width)), " ".repeat(width - n.min(width)))
}

fn emit_json(h: &Horizon, truncated: &Horizon) {
    let f = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    println!("{{");
    println!("  \"qubits\": {}, \"gates\": {},", h.qubits, h.gates);
    println!(
        "  \"cuts\": [{}],",
        h.cuts
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("  \"value\": [{}],", f(&h.value));
    println!("  \"l1\": [{}],", f(&h.l1));
    println!("  \"stake\": [{}],", f(&h.stake()));
    println!("  \"visible\": [{}],", f(&h.visible()));
    println!("  \"settled\": [{}],", f(&h.settled()));
    println!(
        "  \"terms\": [{}],",
        h.terms
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "  \"maxWeight\": [{}],",
        h.max_weight
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    let grid = |name: &str, cell: &dyn Fn(usize, usize) -> f64, last: bool| {
        let rows: Vec<String> = (0..h.qubits)
            .map(|q| {
                format!(
                    "[{}]",
                    (0..h.cuts.len())
                        .map(|c| format!("{:.6}", cell(c, q)))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect();
        println!(
            "  \"{name}\": [{}]{}",
            rows.join(","),
            if last { "" } else { "," }
        );
    };
    grid("influence", &|c, q| h.influence_max(c, q), false);
    grid("commitment", &|c, q| h.commitment[c][q], false);
    grid("overlap", &|c, q| h.overlap(c, q), false);
    grid("blame", &|c, q| truncated.blame[c][q], false);
    println!(
        "  \"truncatedValue\": [{}]",
        f(&truncated.value)
    );
    println!("}}");
}

fn main() -> Result<()> {
    let json = std::env::args().any(|a| a == "--json");
    let n = 12;
    let rots = tfim_trotter(n, 1.0, 0.7, 0.35, 7);
    let obs = (0u64, 1u64 << 6);
    let cfg = Config {
        samples: 48,
        threshold: 0.0,
        forward: Forward::Dense,
    };
    let h = horizon::horizon(obs, &rots, n, &cfg)?;

    let truncated = horizon::horizon(
        obs,
        &rots,
        n,
        &Config {
            threshold: 1e-3,
            ..cfg
        },
    )?;
    if json {
        emit_json(&h, &truncated);
        return Ok(());
    }

    println!("== the latent past and the latent future, as fields ==\n");
    println!("  n={n}, {} rotations, observable Z_6, {} cuts sampled, exact.\n", h.gates, h.cuts.len());
    println!("  A light cone is a SET of cells the answer can reach. The Heisenberg");
    println!("  walk carries more than that: at each cut the observable is a weighted");
    println!("  sum of Paulis, so how much of that weight touches a qubit is a NUMBER.");
    println!("  The cone is only this field's support.\n");

    draw(&h, "latent future — Φ(t,q), the L1 weight that can still feel qubit q", |c, q| {
        h.influence_max(c, q)
    });
    println!();
    draw(&h, "latent past — r(t,q), the Bloch length the forward state has committed", |c, q| {
        h.commitment[c][q]
    });
    println!();
    draw(&h, "their overlap — visible AND committed: what the answer is assembled from", |c, q| {
        h.overlap(c, q)
    });

    println!("\n  The two run opposite ways in time. The future's field opens backward");
    println!("  from the observable; the past's decays forward from |0..0>, which is");
    println!("  fully committed (every Bloch length exactly 1) and holds nothing");
    println!("  jointly. Their contraction is the answer — and it is the SAME NUMBER");
    println!("  at every cut:\n");
    println!("    value at each cut, spread {:.2e} across all {} of them",
        h.value_spread(), h.cuts.len());
    println!("    first {:.12}   last {:.12}", h.value[0], h.value[h.value.len() - 1]);

    println!("\n  What the field says that the cone cannot:\n");
    let fill = h.cone_fill();
    println!("    cone cells                     {}", h.cone_cells());
    println!("    cells above 10% influence      {}", h.count_cells(0.10));
    println!("    cells above 50% influence      {}", h.count_cells(0.50));
    println!("    cone fill (mean Φ inside it)   {fill:.3}   — a cone drawn as a set claims 1.000");
    println!("    cells where Φ FELL             {}   — a cone can never do this", h.non_monotone_cells());
    println!("\n  That last number is the interesting one. A cone only ever grows.");
    println!("  The field shrinks wherever anticommuting terms cancel against each");
    println!("  other, so a qubit's grip on the answer weakens WITHOUT it leaving");
    println!("  the cone. That is interference, in operator space.\n");

    println!("  the front — the last gate at which each qubit still matters:");
    for (q, f) in h.front().iter().enumerate() {
        match f {
            Some(g) => println!("    q{q:<2}  gate {g:<4} {}", bar(*g as f64, h.gates as f64, 40)),
            None => println!("    q{q:<2}  never reached"),
        }
    }

    println!("\n  and the readout neither field gives alone:\n");
    println!("    cut    visible   stake   settled");
    let (visible, stake, settled) = (h.visible(), h.stake(), h.settled());
    for c in (0..h.cuts.len()).step_by(h.cuts.len() / 8) {
        println!(
            "    {:4}   {:7.3}  {:6.3}   {:.3}  {}",
            h.cuts[c],
            visible[c],
            stake[c],
            settled[c],
            bar(settled[c], 1.0, 30)
        );
    }
    println!("\n  `settled` is the share of what the future can still see that the past");
    println!("  has committed to single-qubit marginals. It starts at exactly 1.000 —");
    println!("  the input is a product state — and falls. The cone keeps covering the");
    println!("  same qubits; what it sees stops being ABOUT them individually, because");
    println!("  the information migrates into correlations. Locality drains away under");
    println!("  the future's own gaze, and neither field alone shows it.\n");

    println!("  and retrodictively, where a truncation spends its precision:\n");
    let t = &truncated;
    let total: f64 = t.blame[0].iter().sum();
    println!("    threshold 1e-3: discarded Σ|c| = {total:.4}, invariant spread {:.2e}", t.value_spread());
    println!("    (the exact walk's spread was {:.2e} — truncation is exactly the", h.value_spread());
    println!("     invariant ceasing to be invariant, which is a self-check no");
    println!("     single-cut method can perform on itself)\n");
    let worst = (0..n).max_by(|&a, &b| t.blame[0][a].partial_cmp(&t.blame[0][b]).unwrap()).unwrap();
    for q in 0..n {
        println!(
            "    q{q:<2}  {:7.4}  {}{}",
            t.blame[0][q],
            bar(t.blame[0][q], total.max(1e-30) / 2.0, 34),
            if q == worst { "  ← most" } else { "" }
        );
    }
    println!("\n  Run with --json to emit the sheet for a plotter.");
    Ok(())
}
