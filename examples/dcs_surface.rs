//! What actually crosses the bond, and which description of it is
//! cheapest — SVD as an entrant rather than the referee.
//!
//! The sweep's peak on the 70-qubit experiment is `2^36`, set by the 35
//! legs on a bond. That is the surface's *index count*, which is not
//! the same question as how many numbers it takes to write down, nor
//! the same question as what it costs to read one back. Eight
//! techniques answer all three on the same object.
//!
//! `cargo run --release --example dcs_surface`

use std::time::Duration;

use quantsim::dcs::Dcs;
use quantsim::prelude::*;
use quantsim::surface::{self, CompeteConfig, Technique};
use quantsim::sweep;

fn rule() {
    println!("{}", "─".repeat(78));
}

const TOL: f64 = 1e-9;

/// A cost, or a dash when the technique did not apply.
fn cell(c: &surface::Competition, t: Technique, read: bool) -> String {
    match c.get(t) {
        Some(e) if e.verified(TOL) => {
            let v = if read { e.read_cost } else { e.scalars };
            if v > 1 << 20 {
                format!("2^{:.0}", (v as f64).log2())
            } else {
                format!("{v}")
            }
        }
        Some(e) if e.applicable && e.checked == 0 => "?".into(),
        _ => "—".into(),
    }
}

fn main() -> Result<()> {
    let cfg = CompeteConfig {
        symbolic_budget: Duration::from_secs(8),
        ..CompeteConfig::default()
    };

    rule();
    println!("A. THE SURFACE — what the sweep carries across one bond");
    rule();
    println!("  A bond's legs are its CZ gates, in circuit-time order. The sweep holds");
    println!("  one surface at a time; its index count is 2^legs.");
    println!();
    println!("     n   depth   legs on the middle bond   raw indices");
    for n in [8usize, 12, 16, 20, 24, 70] {
        let d = if n == 70 {
            Dcs::experiment()
        } else {
            Dcs::scaled(n)
        };
        let plan = sweep::plan(&d.circuit())?;
        let mid = plan.legs_per_bond[n / 2 - 1];
        println!("  {n:>4}   {:>5}   {mid:>23}   2^{mid}", d.depth);
    }

    rule();
    println!("B. THE COMPETITION — one surface, every technique, two axes");
    rule();
    println!("  `numbers` is what must cross the bond: a stored index counts as one and");
    println!("  a stored amplitude counts as one, so nobody hides bookkeeping in indices.");
    println!("  `read` is the work to get ONE entry back. `checked` is how many entries");
    println!("  the deviation was actually measured on — a claim with 0 is not a result.");
    println!();
    let d = Dcs::scaled(20);
    let circuit = d.circuit();
    let surfaces = sweep::surfaces(&circuit, 0)?;
    let mid = &surfaces[surfaces.len() / 2];
    let comp = surface::compete(mid, Some(&circuit), 0, &cfg)?;
    println!(
        "  n = 20, bond after qubit {}, {} legs, raw {} indices",
        comp.after_qubit, comp.legs, comp.raw
    );
    println!();
    println!("     technique      numbers          read   checked   deviation   what it found");
    for e in &comp.entries {
        let num = if e.applicable {
            format!("{}", e.scalars)
        } else {
            "—".into()
        };
        let read = if e.applicable {
            if e.read_cost > 1 << 20 {
                format!("2^{:.0}", (e.read_cost as f64).log2())
            } else {
                format!("{}", e.read_cost)
            }
        } else {
            "—".into()
        };
        let dev = if e.checked > 0 {
            format!("{:.2e}", e.error)
        } else {
            "—".into()
        };
        println!(
            "  {:<12}  {:>9}  {:>12}  {:>8}  {:>10}   {}",
            e.technique.name(),
            num,
            read,
            e.checked,
            dev,
            e.detail
        );
    }
    if let Some(w) = comp.winner(TOL) {
        println!();
        println!(
            "  shortest verified description: {} at {} numbers",
            w.technique.name(),
            w.scalars
        );
    }
    if let Some(f) = comp.fastest(TOL) {
        println!(
            "  cheapest verified read:        {} at {} operations",
            f.technique.name(),
            f.read_cost
        );
    }

    rule();
    println!("C. EVERY BOND — where each description lives and where it dies");
    rule();
    println!("  Numbers to describe the surface, bond by bond, at n = 20.");
    println!();
    println!(
        "     bond   legs    dense  support    walsh  mps-svd  phase-poly  path-sum   read(ps)"
    );
    let mut runs = Vec::new();
    for s in &surfaces {
        let c = surface::compete(s, Some(&circuit), 0, &cfg)?;
        println!(
            "  {:>7}   {:>4}  {:>7}  {:>7}  {:>7}  {:>7}  {:>10}  {:>8}  {:>9}",
            c.after_qubit,
            c.legs,
            cell(&c, Technique::Dense, false),
            cell(&c, Technique::Support, false),
            cell(&c, Technique::Walsh, false),
            cell(&c, Technique::Mps, false),
            cell(&c, Technique::PhasePoly, false),
            cell(&c, Technique::PathSum, false),
            cell(&c, Technique::PathSum, true),
        );
        runs.push(c);
    }
    println!();
    let t = surface::tally(&runs, TOL);
    let mut order: Vec<_> = t.into_iter().collect();
    order.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (tech, count) in order {
        println!(
            "  shortest description on {count} of {} bonds: {}",
            runs.len(),
            tech.name()
        );
    }
    println!();
    println!("  Why phase-poly stops, in its own words:");
    for s in [4usize, 9, 18] {
        if let Some(c) = runs.iter().find(|c| c.after_qubit == s) {
            if let Some(e) = c.get(Technique::PhasePoly) {
                println!("    bond {s:>2}: {}", e.detail);
            }
        }
    }

    rule();
    println!("E. ARITY — the same surface read at higher polarity");
    rule();
    println!("  Everything above is bipolar. A bipartition is a ± split, the Walsh");
    println!("  characters are the ±1 characters of F2^m, and an independent volume is");
    println!("  priced at TWO surfaces. Arity two is a choice, not a fact about the");
    println!("  object. A leg is one CZ — one entangling event at one time — so grouping");
    println!("  b consecutive legs makes an inclusion/exclusion axis of arity d = 2^b out");
    println!("  of b consecutive entangling events, whose harmonics are the Z_d");
    println!("  characters and NOT the Z2^b characters on the same legs.");
    println!();
    println!("  Coarse-to-fine, on three bonds at n = 24 (12 legs, 4096 indices):");
    println!();
    let d24 = Dcs::scaled(24);
    let c24 = d24.circuit();
    let s24 = sweep::surfaces(&c24, 0)?;
    for bond in [0usize, 5, 22] {
        let s = &s24[bond];
        let grains = surface::grain_scan(s, surface::DEFAULT_EPSILON)?;
        println!(
            "    bond {bond:>2}  ({} legs, {} raw)",
            s.legs.len(),
            1usize << s.legs.len()
        );
        println!("        b   arity   digits   direct   walsh   Z_d chars   chi   entropy");
        for g in &grains {
            println!(
                "     {:>4}  {:>6}  {:>7}  {:>7}  {:>6}  {:>10}  {:>4}  {:>8.2}",
                g.legs_per_digit,
                g.radix,
                g.sites,
                g.direct_nnz,
                g.walsh_nnz,
                g.cyclic_nnz,
                g.chi,
                g.entropy
            );
        }
        println!();
    }
    println!("  `direct` and `walsh` do not move with b — they are properties of the");
    println!("  surface, not of the grouping. `Z_d chars` is the arity-d reading, and it");
    println!("  is the column that answers whether compound polarity buys anything.");

    rule();
    println!("D. SCALING — which descriptions double with the surface and which do not");
    rule();
    println!();
    println!("     n   legs        raw   support     walsh   mps-svd  path-sum   h*");
    for n in [8usize, 10, 12, 14, 16, 18, 20, 22, 24] {
        let d = Dcs::scaled(n);
        let circuit = d.circuit();
        let ss = sweep::surfaces(&circuit, 0)?;
        let s = &ss[ss.len() / 2];
        let cfg = CompeteConfig {
            cp_max_legs: 0,
            ..cfg.clone()
        };
        let c = surface::compete(s, Some(&circuit), 0, &cfg)?;
        let hstar = surface::symbolic(&circuit, s.after_qubit, 0)
            .map(|x| x.h_star.to_string())
            .unwrap_or_else(|_| "—".into());
        println!(
            "  {n:>4}   {:>4}   {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>3}",
            c.legs,
            c.raw,
            cell(&c, Technique::Support, false),
            cell(&c, Technique::Walsh, false),
            cell(&c, Technique::Mps, false),
            cell(&c, Technique::PathSum, false),
            hstar,
        );
    }
    rule();
    Ok(())
}
