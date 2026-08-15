//! The interface, not the T-count: cutting the DCS chain.
//!
//! `cargo run --release --example dcs_cut`

use quantsim::cutsim;
use quantsim::dcs::Dcs;
use quantsim::prelude::*;
use std::time::Instant;

fn main() {
    println!("{}", "─".repeat(74));
    println!("A. THE INTERFACE IS THIN  (structural, no simulation)");
    println!("{}", "─".repeat(74));
    println!("     n  depth      CZ    T   best cut   crossing   interior   1-qubit");
    for n in [12, 16, 24, 48, 70] {
        let d = if n == 70 {
            Dcs::experiment()
        } else {
            Dcs::scaled(n)
        };
        let c = d.circuit();
        let p = cutsim::best_plan(&c).unwrap();
        println!(
            "  {n:>4}  {:>5}  {:>6}  {:>4}   {:>8}   {:>8}   {:>8}   {:>7}",
            d.depth,
            d.two_qubit_gates(),
            d.t_gates,
            p.cut,
            p.crossings.len(),
            p.interior_two_qubit,
            p.single_qubit
        );
    }

    println!("{}", "─".repeat(74));
    println!("B. THE DECOMPOSITION IS EXACT  (checked against dense)");
    println!("{}", "─".repeat(74));
    println!("  Every amplitude of the circuit, from one pass over the branches.");
    println!();
    println!("     n   cut   crossings   paths   amplitudes   worst |Δ| vs dense    time");
    for n in [8, 10, 12, 14, 16, 18] {
        let d = Dcs::scaled(n);
        let c = d.circuit();
        let p = cutsim::plan(&c, n / 2).unwrap();
        let dense = Simulator::<C64>::new().run(&c).unwrap();
        let t0 = Instant::now();
        let targets: Vec<u64> = (0..(1u64 << n)).collect();
        let got = cutsim::amplitudes_at(&c, &p, &targets).unwrap();
        let worst = targets
            .iter()
            .zip(&got)
            .map(|(&x, g)| (*g - dense.amplitude(x)).norm())
            .fold(0.0f64, f64::max);
        println!(
            "  {n:>4}  {:>4}   {:>9}   {:>5}   {:>10}   {:>17.2e}   {:>8.2?}",
            p.cut,
            p.crossings.len(),
            p.paths(),
            targets.len(),
            worst,
            t0.elapsed()
        );
    }
    println!("{}", "─".repeat(74));
    println!("C. THE COST LAW  (a fixed batch of targets, as sampling would ask)");
    println!("{}", "─".repeat(74));
    println!("  256 amplitudes per run. Cost is 2^k branches x 2^(n/2) per half,");
    println!("  so the exponent is k + n/2 and the T count does not appear in it.");
    println!();
    println!("     n   k = crossings   k + n/2      time      s/branch");
    let mut sizes = Vec::new();
    let mut times = Vec::new();
    for n in [12, 14, 16, 18, 20, 22] {
        let d = Dcs::scaled(n);
        let c = d.circuit();
        let p = cutsim::plan(&c, n / 2).unwrap();
        let targets: Vec<u64> = (0..256u64)
            .map(|i| i.wrapping_mul(0x9E3779B97F4A7C15))
            .collect();
        let t0 = Instant::now();
        cutsim::amplitudes_at(&c, &p, &targets).unwrap();
        let el = t0.elapsed();
        println!(
            "  {n:>4}   {:>13}   {:>7}   {:>9.2?}   {:>11.2e}",
            p.crossings.len(),
            p.cost_exponent(),
            el,
            el.as_secs_f64() / p.paths() as f64
        );
        sizes.push(p.cost_exponent() as f64);
        times.push(el.as_secs_f64());
    }
    // Fit log2(time) = a * exponent + b over the measured points.
    let m = sizes.len() as f64;
    let sx: f64 = sizes.iter().sum();
    let sy: f64 = times.iter().map(|t| t.log2()).sum();
    let sxx: f64 = sizes.iter().map(|x| x * x).sum();
    let sxy: f64 = sizes.iter().zip(&times).map(|(x, t)| x * t.log2()).sum();
    let a = (m * sxy - sx * sy) / (m * sxx - sx * sx);
    let b = (sy - a * sx) / m;
    let at70 = a * 70.0 + b;
    println!();
    println!("  fit: log2(seconds) = {a:.3}*(k + n/2) + {b:.2}");
    println!("  extrapolated to the experiment (k = 35, n/2 = 35, exponent 70):");
    println!(
        "    single core: 2^{at70:.1} s = 10^{:.1} s",
        at70 * 2f64.log10()
    );
    println!("    the branches are independent, so across N cores this divides by N;");
    println!(
        "    at 10^6 cores that is 10^{:.1} s.",
        at70 * 2f64.log10() - 6.0
    );
    println!();
    println!("{}", "─".repeat(74));
    println!("D. THE T GATES ARE FREE  (same circuit, doping swept)");
    println!("{}", "─".repeat(74));
    println!("  If the cost is set by the interface and not by the magic, then");
    println!("  sweeping the T count at a fixed skeleton must not move the clock.");
    println!();
    println!("      t   crossings      time     vs t=0");
    let n = 18;
    let base = Dcs::scaled(n);
    let c0 = base.with_t(0).circuit();
    let p0 = cutsim::plan(&c0, n / 2).unwrap();
    let targets: Vec<u64> = (0..256u64)
        .map(|i| i.wrapping_mul(0x9E3779B97F4A7C15))
        .collect();
    let t0 = Instant::now();
    cutsim::amplitudes_at(&c0, &p0, &targets).unwrap();
    let baseline = t0.elapsed().as_secs_f64();
    for t in [0, 8, 32, 128, 512, 1024] {
        let d = base.with_t(t);
        if t > d.dope_sites().len() {
            continue;
        }
        let c = d.circuit();
        let p = cutsim::plan(&c, n / 2).unwrap();
        let t1 = Instant::now();
        cutsim::amplitudes_at(&c, &p, &targets).unwrap();
        let el = t1.elapsed().as_secs_f64();
        println!(
            "  {t:>5}   {:>9}   {el:>7.3} s     {:>5.2}x",
            p.crossings.len(),
            el / baseline
        );
    }
    println!();
    println!("  The experiment's own doping rate puts t = 468 on 70 qubits. Against");
    println!("  this decomposition that is the same circuit as t = 0 — and t = 0 is");
    println!("  the Clifford skeleton the paper itself calls efficiently simulable.");
    println!();
    println!("{}", "─".repeat(74));
    println!("  For comparison, the paper's own extrapolations for this instance:");
    println!("    MPS (quimb, measured + fitted)          10^25 s");
    println!("    stabilizer decomposition (QuiZX)        10^42 s");
    println!("{}", "─".repeat(74));
}
