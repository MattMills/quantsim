//! The sequential sweep on DCS: what does the interface actually cost?
use quantsim::dcs::Dcs;
use quantsim::sweep;

fn main() {
    println!("{}", "─".repeat(76));
    println!("A. EXACTNESS  (every amplitude against dense)");
    println!("{}", "─".repeat(76));
    println!("     n    t   peak legs   peak amps   worst |Δ| vs dense");
    for n in [6usize, 8, 10, 12] {
        let d = Dcs::scaled(n);
        let c = d.circuit();
        let p = sweep::plan(&c).unwrap();
        let dev = sweep::max_deviation_vs_dense(&c).unwrap();
        println!(
            "  {n:>4} {:>4}   {:>9}   {:>9}   {:>17.2e}",
            d.t_gates,
            p.peak_live_legs,
            p.peak_amplitudes(),
            dev
        );
    }
    println!("{}", "─".repeat(76));
    println!("B. WHAT IT COSTS AT SCALE  (structure only, nothing run)");
    println!("{}", "─".repeat(76));
    println!("     n  depth      CZ     T   legs/bond   PEAK legs   peak amps   vs 2^n");
    for n in [12usize, 16, 24, 48, 70] {
        let d = if n == 70 { Dcs::experiment() } else { Dcs::scaled(n) };
        let c = d.circuit();
        let p = sweep::plan(&c).unwrap();
        let maxbond = p.legs_per_bond.iter().max().copied().unwrap_or(0);
        println!(
            "  {n:>4}  {:>5}  {:>6}  {:>4}   {:>9}   {:>9}   2^{:<9}   2^{}",
            d.depth,
            d.two_qubit_gates(),
            d.t_gates,
            maxbond,
            p.peak_live_legs,
            p.peak_live_legs + 1,
            n
        );
    }
    println!("{}", "─".repeat(76));
    println!("C. MEASURED WALL-CLOCK  (one amplitude, single core)");
    println!("{}", "─".repeat(76));
    println!("     n   peak legs        time     s / 2^peak");
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for n in [10usize, 12, 14, 16, 18, 20, 22] {
        let d = Dcs::scaled(n);
        let c = d.circuit();
        let p = sweep::plan(&c).unwrap();
        let t0 = std::time::Instant::now();
        sweep::amplitude(&c, 0x5555_5555_5555_5555 & ((1u64 << n) - 1)).unwrap();
        let el = t0.elapsed();
        println!(
            "  {n:>4}   {:>9}   {:>9.2?}   {:>12.2e}",
            p.peak_live_legs,
            el,
            el.as_secs_f64() / (1u64 << p.peak_live_legs) as f64
        );
        xs.push(p.peak_live_legs as f64);
        ys.push(el.as_secs_f64().log2());
    }
    let m = xs.len() as f64;
    let (sx, sy): (f64, f64) = (xs.iter().sum(), ys.iter().sum());
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| x * y).sum();
    let a = (m * sxy - sx * sy) / (m * sxx - sx * sx);
    let b = (sy - a * sx) / m;
    println!();
    println!("  fit: log2(seconds) = {a:.3}·(peak legs) + {b:.2}");
    println!("  at the experiment's peak of 36 legs: 2^{:.1} s = {:.1e} s single core",
        a * 36.0 + b, 2f64.powf(a * 36.0 + b));
    println!();
    println!("  For comparison, the paper's own extrapolations for this instance:");
    println!("    MPS (quimb, measured + fitted)      10^25 s");
    println!("    stabilizer decomposition (QuiZX)    10^42 s");
    println!("{}", "─".repeat(76));
}
