//! How fast does the Pauli sum actually grow on the 70-qubit DCS
//! instance?
//!
//! `support::propagate_wide` is the crate's only register-unbounded
//! Pauli walk — everything else caps at 64 through `PauliKey = (u64,u64)`
//! — so it is the one mechanism that can be pointed at the experiment's
//! full width without a rewrite. Its cost is the **term count**, and the
//! term count is not `2^t`: rotations only branch a term they
//! anticommute with, and branches that land on the same Pauli string
//! merge.
//!
//! Two ceilings bound it from above and they are far apart:
//!
//! * `2^k` after `k` rotations, if nothing ever merged;
//! * `2^140`, because the 468 axes have symplectic rank exactly `2n`, so
//!   the sum lives in a single coset of a group with `2^140` elements
//!   and cannot hold more distinct terms than that.
//!
//! What the walk actually does between those is a measurement, and it is
//! what this runs. Exact — no threshold, no cap, so the numbers are the
//! representation's real size and not a truncation artifact.
//!
//! `cargo run --release --example dcs_pauli_growth`

use quantsim::dcs::{self, Dcs};
use quantsim::pathsum::Mask;
use quantsim::rng::Prng;
use quantsim::support::{propagate_wide, WideConfig, WidePauli, WidePauliSum, WideRotation};
use std::time::{Duration, Instant};

/// The axes as wide rotations, in circuit order.
fn wide_axes(d: Dcs) -> Vec<WideRotation> {
    let circuit = d.circuit();
    dcs::rotation_axes(&circuit)
        .expect("Clifford+T")
        .iter()
        .map(|(x, z)| WideRotation {
            // A T gate is exp(-i(pi/8)P) up to phase: a quarter-turn
            // rotation in the exp(-i theta P / 2) convention.
            theta: std::f64::consts::FRAC_PI_4,
            axis: WidePauli::from_masks(x, z),
        })
        .collect()
}

/// A starting observable. The physical one is `C† P C` for the frame
/// Clifford `C`, and a Clifford maps a Pauli to a single Pauli — so any
/// single Pauli string is a faithful starting point. Weight is the
/// parameter that matters, so both ends are measured.
fn observable(n: usize, weight: usize, seed: u64) -> WidePauliSum {
    let mut rng = Prng::new(seed);
    let (mut x, mut z) = (Mask::zero(), Mask::zero());
    let mut placed = 0;
    while placed < weight {
        let q = (rng.next_u64() % n as u64) as usize;
        if x.bit(q) || z.bit(q) {
            continue;
        }
        match rng.next_u64() % 3 {
            0 => x.set(q),
            1 => z.set(q),
            _ => {
                x.set(q);
                z.set(q);
            }
        }
        placed += 1;
    }
    let mut sum = WidePauliSum::zero();
    sum.add(WidePauli::from_masks(&x, &z), quantsim::C64::new(1.0, 0.0));
    sum
}

fn main() {
    let e = Dcs::experiment();
    let axes = wide_axes(e);
    println!("{}", "─".repeat(78));
    println!("PAULI-SUM GROWTH ON THE 70-QUBIT INSTANCE");
    println!("{}", "─".repeat(78));
    println!(
        "  {} rotation axes over {} qubits; mean weight {:.1}, max {}",
        axes.len(),
        e.qubits,
        axes.iter().map(|r| r.axis.weight()).sum::<usize>() as f64 / axes.len() as f64,
        axes.iter().map(|r| r.axis.weight()).max().unwrap_or(0)
    );
    println!(
        "  symplectic rank {} = 2n, so the sum lives in a group of 2^{} elements",
        dcs::symplectic_rank(
            &dcs::rotation_axes(&e.circuit()).unwrap(),
            e.qubits
        ),
        140
    );
    println!();
    println!("  Exact walk (no threshold, no cap) over the LAST k rotations:");
    println!("       k     terms      2^k     log2(terms)   bits/rotation      time");

    let exact = WideConfig {
        threshold: 0.0,
        max_terms: None,
    };
    let obs = observable(e.qubits, 20, 11);
    let mut pts: Vec<(usize, f64)> = Vec::new();
    for k in [4usize, 8, 12, 16, 20, 24, 28, 32, 36, 40] {
        let tail = &axes[axes.len() - k..];
        let t0 = Instant::now();
        let r = quantsim::guard::with_time_budget(Duration::from_secs(90), || {
            propagate_wide(&obs, tail, &exact)
        });
        let el = t0.elapsed();
        let lg = (r.sum.len() as f64).log2();
        println!(
            "    {k:>4}  {:>8}  {:>7}   {:>10.2}   {:>12.3}   {:>8.2?}",
            r.sum.len(),
            if k < 63 {
                format!("2^{k}")
            } else {
                "—".into()
            },
            lg,
            lg / k as f64,
            el
        );
        pts.push((k, lg));
        if el > Duration::from_secs(60) {
            println!("    (stopping: the next step would exceed the budget)");
            break;
        }
    }

    // Fit log2(terms) = a·k + b over the second half of the sweep, where
    // the early transient has passed.
    if pts.len() >= 4 {
        let tail = &pts[pts.len() / 2..];
        let m = tail.len() as f64;
        let sx: f64 = tail.iter().map(|p| p.0 as f64).sum();
        let sy: f64 = tail.iter().map(|p| p.1).sum();
        let sxx: f64 = tail.iter().map(|p| (p.0 * p.0) as f64).sum();
        let sxy: f64 = tail.iter().map(|p| p.0 as f64 * p.1).sum();
        let a = (m * sxy - sx * sy) / (m * sxx - sx * sx);
        let b = (sy - a * sx) / m;
        println!();
        println!("  fit over the last {} points: log2(terms) = {a:.4}·k + {b:.2}", tail.len());
        println!(
            "  extrapolated to all 468 rotations: 2^{:.1}",
            (a * 468.0 + b).min(140.0)
        );
        println!("  against the group ceiling 2^140 and the amplitude ceiling 2^70.");
    }

    println!();
    println!("  Observable weight sweep at k = 24 (does the starting Pauli matter?):");
    println!("     weight     terms");
    for w in [1usize, 5, 20, 50] {
        let o = observable(e.qubits, w, 7 + w as u64);
        let tail = &axes[axes.len() - 24..];
        let r = quantsim::guard::with_time_budget(Duration::from_secs(90), || {
            propagate_wide(&o, tail, &exact)
        });
        println!("     {w:>6}  {:>8}", r.sum.len());
    }
    println!("{}", "─".repeat(78));
}
