//! A register that holds only engineered entanglement, and the exact
//! price the magic charges it.
//!
//! A `T` gate is diagonal, so `T = a·I + b·Z` with both arms Clifford.
//! A volume with `t` of them is a weighted sum of `2^t` Clifford
//! volumes, each of which the engineered register holds as one affine
//! coset carrying a quadratic phase. Nothing is truncated; the sum is
//! checked against the true surface.
//!
//! The question is what the register actually pays, which is not `2^t`
//! but the dimension of the branches' span.
//!
//! `cargo run --release --example dcs_engineered`

use quantsim::dcs::Dcs;
use quantsim::engineered;
use quantsim::prelude::*;
use quantsim::sweep;

fn rule() {
    println!("{}", "─".repeat(78));
}

fn main() -> Result<()> {
    rule();
    println!("A. THE BRANCH SET — is it really Clifford, and does it really sum?");
    rule();
    println!("  T = a·I + b·Z, a = (1+ω)/2, b = (1−ω)/2, ω = e^{{iπ/4}}. Both arms are");
    println!("  Clifford, so every branch must be a single stabilizer term and the");
    println!("  weighted sum must be the surface itself. Both are checked, not assumed.");
    println!();
    println!("     n   bond   legs    T   branches   all stabilizer?   Σ branches vs truth");
    for n in [12usize, 16] {
        let d = Dcs::scaled(n);
        let c = d.circuit();
        for bond in [0usize, 2, 4] {
            if let Ok(e) = engineered::engineered_surface(&c, bond, 0) {
                println!(
                    "  {n:>4}   {:>4}   {:>4}  {:>3}   {:>8}   {:>15}   {:>19.2e}",
                    e.after_qubit,
                    e.legs,
                    e.t_gates,
                    e.branches,
                    if e.stabilizer_branches == e.branches {
                        "yes, all"
                    } else {
                        "NO"
                    },
                    e.deviation
                );
            }
        }
    }

    rule();
    println!("B. WHAT THE REGISTER PAYS — branches against the span they occupy");
    rule();
    println!("  `supports` counts the distinct affine cosets among the branches, and");
    println!("  `span` is the dimension they span. The first says how much of the");
    println!("  geometry is shared; the second is what must actually be stored.");
    println!();
    println!("     n   bond   legs   dense    T   branches   supports   span   verdict");
    for n in [12usize, 16, 20] {
        let d = Dcs::scaled(n);
        let c = d.circuit();
        for s in sweep::surfaces(&c, 0)? {
            if engineered::magic_behind(&c, s.after_qubit) > 11 {
                println!("  {n:>4}   {:>4}   past what this display will spend", s.after_qubit);
                break;
            }
            match engineered::engineered_surface(&c, s.after_qubit, 0) {
                Ok(e) => println!(
                    "  {n:>4}   {:>4}   {:>4}  {:>6}  {:>3}   {:>8}   {:>8}   {:>4}   {}",
                    e.after_qubit,
                    e.legs,
                    e.dense(),
                    e.t_gates,
                    e.branches,
                    e.distinct_supports,
                    e.span_rank,
                    if e.span_rank as u128 >= e.dense() {
                        "saturated — dense is cheaper"
                    } else if e.distilled() {
                        "below the branch count"
                    } else {
                        "branches independent"
                    }
                ),
                Err(_) => {
                    println!("  {n:>4}   {:>4}   past the branch budget", s.after_qubit);
                    break;
                }
            }
        }
        println!();
    }

    rule();
    println!("C. THE BOUNDARY — where this representation stops");
    rule();
    println!("  A term is an affine coset with a quadratic phase: O(m^2) numbers, not one.");
    println!("  So the register is cheaper than writing the surface out while");
    println!();
    println!("      span x (m+1)^2  <  2^m,   and span = min(2^t, 2^m)");
    println!();
    println!("  which is a counting question needing no branch sweeps at all — so it can");
    println!("  be asked of the experiment directly, at its real size.");
    println!();
    let exp = Dcs::experiment();
    let c = exp.circuit();
    let plan = sweep::plan(&c)?;
    println!("     bond   legs   T behind   span (bounded)      cost   dense   engineered");
    let mut last_covered = None;
    for bond in [0usize, 1, 2, 3, 4, 5, 6, 10, 34, 68] {
        let legs = plan.legs_per_bond[bond];
        let t = engineered::magic_behind(&c, bond);
        let dense = 1u128 << legs.min(100);
        let span = if t >= legs { dense } else { 1u128 << t };
        let cost = span.saturating_mul((legs as u128 + 1).pow(2));
        let covered = cost < dense;
        if covered {
            last_covered = Some(bond);
        }
        println!(
            "  {bond:>7}   {legs:>4}   {t:>8}   {:>14}   {:>7}   {dense:>5}   {}",
            if span >= dense {
                format!("2^{legs} (capped)")
            } else {
                format!("2^{t}")
            },
            if cost > 1 << 20 {
                format!("2^{:.0}", (cost as f64).log2())
            } else {
                format!("{cost}")
            },
            if covered { "cheaper" } else { "saturated" }
        );
    }
    println!();
    let covered: Vec<usize> = (0..exp.qubits - 1)
        .filter(|&b| {
            let legs = plan.legs_per_bond[b];
            let t = engineered::magic_behind(&c, b);
            let dense = 1u128 << legs.min(100);
            let span = if t >= legs { dense } else { 1u128 << t };
            span.saturating_mul((legs as u128 + 1).pow(2)) < dense
        })
        .collect();
    println!();
    match last_covered {
        Some(_) => println!(
            "  The engineered register is cheaper on {} of the {} bonds — {:?} — and",
            covered.len(),
            exp.qubits - 1,
            covered
        ),
        None => println!("  The engineered register is cheaper on no bond of the experiment, and is"),
    }
    println!("  saturated on the rest. The magic is spread uniformly, so roughly");
    println!(
        "  {:.1} T gates sit behind each additional world-line while the legs on a",
        exp.t_gates as f64 / exp.qubits as f64
    );
    println!(
        "  bond stay at {}. The two curves cross early and never come back.",
        plan.legs_per_bond[0]
    );
    println!();
    println!("  What does hold everywhere: the branches share their affine geometry.");
    println!("  Up to four cosets across thousands of branches, so the engineered part");
    println!("  of the entanglement is stored once and the magic lives entirely in the");
    println!("  phase — which is where the branches are linearly independent.");
    rule();
    Ok(())
}
