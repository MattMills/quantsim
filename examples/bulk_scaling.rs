//! Separating width from support: what the `bulk` register actually costs.
//!
//! Run with `cargo run --release --example bulk_scaling`.
//!
//! `examples/factoring_scale.rs` reports a "time in w" fit per
//! representation, and every one of them is **confounded**: along a ladder
//! ordered by the multiplicative order `r`, the width `w` rises with it, so
//! neither marginal fit isolates its own variable. `bulk` came back
//! polynomial in `w` where the others came back exponential, which is
//! exactly the shape a confounded fit produces when a representation
//! terminates early and sees only a short stretch of the ladder.
//!
//! This measures the two axes apart.
//!
//! * **Fixed support, every width.** `a = N−1` satisfies `a² ≡ 1` for every
//!   `N`, so it has order exactly 2 at any width. The register then holds
//!   two amplitudes whatever `w` is, and the time is the representation's
//!   own per-round, per-width overhead with the state held constant.
//! * **Fixed width, swept support.** One modulus, many bases, orders
//!   spanning several decades.
//! * **The joint fit.** `log t = c + α·log r + β·w` by least squares over
//!   the whole grid, against the two marginals, so the size of the
//!   confounding is itself a number.

use std::time::{Duration, Instant};

use quantsim::bounds::fit_law;
use quantsim::guard;
use quantsim::prelude::*;
use quantsim::shor::{self, OrderFinder, PhaseForm};

const PROBED: [&str; 7] = [
    "bulk",
    "mera",
    "mps",
    "sparse",
    "mosaic",
    "phase-field",
    "factored",
];

fn semiprime(w: usize) -> Option<(u64, u64, u64)> {
    if !(4..=62).contains(&w) {
        return None;
    }
    let (lo, hi) = (1u64 << (w - 1), (1u64 << w) - 1);
    let mut p = (hi as f64).sqrt() as u64 + 1;
    while p >= 2 {
        if shor::is_prime(p) {
            let mut q = p + 1;
            while p.checked_mul(q).is_some_and(|n| n <= hi) {
                if shor::is_prime(q) && p * q >= lo {
                    return Some((p * q, p, q));
                }
                q += 1;
            }
        }
        p -= 1;
    }
    None
}

fn factor_small(mut m: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut d = 2u64;
    while d.saturating_mul(d) <= m {
        if m % d == 0 {
            while m % d == 0 {
                m /= d;
            }
            out.push(d);
        }
        d += if d == 2 { 1 } else { 2 };
    }
    if m > 1 {
        out.push(m);
    }
    out
}

fn order_via_lambda(a: u64, n: u64, p: u64, q: u64) -> Option<u64> {
    if quantsim::padic::gcd(a % n, n) != 1 {
        return None;
    }
    let g = quantsim::padic::gcd(p - 1, q - 1);
    let lambda = (p - 1) / g * (q - 1);
    let mut primes = factor_small(p - 1);
    primes.extend(factor_small(q - 1));
    primes.sort_unstable();
    primes.dedup();
    let mut r = lambda;
    for pr in primes {
        while r % pr == 0 && shor::pow_mod(a, r / pr, n) == 1 {
            r /= pr;
        }
    }
    (shor::pow_mod(a, r, n) == 1).then_some(r)
}

fn human(nanos: u64) -> String {
    match nanos {
        n if n < 10_000 => format!("{n}ns"),
        n if n < 10_000_000 => format!("{:.1}µs", n as f64 / 1e3),
        n if n < 10_000_000_000 => format!("{:.1}ms", n as f64 / 1e6),
        n => format!("{:.2}s", n as f64 / 1e9),
    }
}

fn short(e: &quantsim::Error) -> String {
    if let quantsim::Error::Timeout { elapsed_ms, .. } = e {
        return format!("over budget at {elapsed_ms}ms");
    }
    let s = e.to_string();
    let s = s.split(';').next().unwrap_or(&s).trim().to_string();
    s.chars().take(38).collect()
}

fn run(sim: &Simulator, backend: &str, n: u64, a: u64, budget: Duration) -> Result<(u64, usize)> {
    let f = OrderFinder::new(n, a)?.on_backend(backend);
    guard::with_time_budget(budget, || {
        let t = Instant::now();
        let e = f.estimate(sim, PhaseForm::Semiclassical, &mut Prng::new(7))?;
        Ok((t.elapsed().as_nanos() as u64, e.peak_support))
    })
}

/// Least squares for `log t = c + α·log r + β·w`.
fn joint_fit(points: &[(f64, f64, f64)]) -> Option<(f64, f64)> {
    if points.len() < 4 {
        return None;
    }
    // Normal equations for the design [1, log r, w].
    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];
    for &(lr, w, lt) in points {
        let row = [1.0, lr, w];
        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += row[i] * row[j];
            }
            atb[i] += row[i] * lt;
        }
    }
    // Gaussian elimination with partial pivoting.
    let mut m = [[0.0f64; 4]; 3];
    for i in 0..3 {
        m[i][..3].copy_from_slice(&ata[i]);
        m[i][3] = atb[i];
    }
    for col in 0..3 {
        let piv =
            (col..3).max_by(|&x, &y| m[x][col].abs().partial_cmp(&m[y][col].abs()).unwrap())?;
        m.swap(col, piv);
        if m[col][col].abs() < 1e-12 {
            return None;
        }
        for row in 0..3 {
            if row != col {
                let factor = m[row][col] / m[col][col];
                let pivot = m[col];
                for (k, cell) in m[row].iter_mut().enumerate().skip(col) {
                    *cell -= factor * pivot[k];
                }
            }
        }
    }
    Some((m[1][3] / m[1][1], m[2][3] / m[2][2]))
}

fn main() -> Result<()> {
    // The whole study holds *two* amplitudes. A representation that needs
    // more than 128 MiB for two amplitudes has already answered the
    // question, and the limit turns that into a named refusal instead of
    // an uninterruptible dense run — `phase_field::load` escapes to dense
    // by construction ("an arbitrary amplitude list is outside the class"),
    // and dense element loops carry no checkpoint, so the time budget
    // cannot reach them.
    guard::set_memory_limit(Some(128 << 20));
    let mut sim: Simulator = Simulator::new();
    {
        use quantsim::backend::PhaseFieldState;
        sim.backends_mut()
            .register("phase-field", |n| Ok(Box::new(PhaseFieldState::new(n)?)))?;
    }
    let budget = Duration::from_millis(250);

    println!("budget {budget:?}/run, memory limit 128MiB");

    println!();
    println!("== A. fixed support (r = 2, a = N−1), width swept: the width cost alone ==");
    let widths: Vec<usize> = (4..=40).step_by(2).collect();
    print!("  {:<13}", "backend / w");
    for w in &widths {
        print!("{w:>7}");
    }
    println!("   fit in w");
    let mut width_only: Vec<(&str, Vec<(usize, u64)>)> = Vec::new();
    // A backend that cannot hold width w with two amplitudes cannot hold
    // it with more, so sections B and C skip past its ceiling instead of
    // re-discovering it one timeout at a time.
    let mut ceiling: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for backend in PROBED {
        print!("  {backend:<13}");
        let mut pts: Vec<(usize, u64)> = Vec::new();
        let mut stop = String::new();
        for &w in &widths {
            let Some((n, _, _)) = semiprime(w) else {
                continue;
            };
            match run(&sim, backend, n, n - 1, budget) {
                Ok((t, sup)) => {
                    debug_assert_eq!(sup.min(2), 2.min(sup));
                    print!("{:>7.2}", t as f64 / 1e6);
                    pts.push((w, t));
                }
                Err(e) => {
                    if stop.is_empty() {
                        stop = short(&e);
                    }
                    print!("{:>7}", "—");
                }
            }
        }
        let law = if pts.len() >= 3 {
            let (xs, ys): (Vec<usize>, Vec<usize>) =
                pts.iter().map(|&(w, t)| (w, t as usize)).unzip();
            format!("{:?}", fit_law(&xs, &ys).law)
        } else {
            "—".into()
        };
        println!("   {law}  {stop}");
        ceiling.insert(backend, pts.last().map_or(0, |e| e.0));
        width_only.push((backend, pts));
    }
    println!("  (ms; support is 2 in every cell — a·a ≡ 1 mod N for a = N−1)");

    println!();
    println!("== B. fixed width, support swept: the support cost alone ==");
    for &w in &[12usize, 20] {
        let Some((n, p, q)) = semiprime(w) else {
            continue;
        };
        let mut orders: Vec<(u64, u64)> = (2..400u64)
            .filter_map(|a| order_via_lambda(a, n, p, q).map(|r| (r, a)))
            .collect();
        orders.sort_unstable();
        orders.dedup_by_key(|x| x.0);
        // Thin to a geometric spread.
        let mut ladder: Vec<(u64, u64)> = Vec::new();
        let mut last = 0f64;
        for e in orders {
            if e.0 as f64 >= last * 4.0 && ladder.len() < 6 {
                last = e.0 as f64;
                ladder.push(e);
            }
        }
        println!(
            "  w = {w}, N = {n}, {} orders: {:?}",
            ladder.len(),
            ladder.iter().map(|e| e.0).collect::<Vec<_>>()
        );
        println!("    {:<13} {:>10}  fit in r", "backend", "max r");
        for backend in PROBED {
            if ceiling.get(backend).copied().unwrap_or(0) < w {
                println!(
                    "    {backend:<13} {:>10}  past its fixed-support ceiling",
                    "—"
                );
                continue;
            }
            let mut pts: Vec<(usize, usize)> = Vec::new();
            let mut best = 0u64;
            let mut stop = String::new();
            for &(r, a) in &ladder {
                match run(&sim, backend, n, a, budget) {
                    Ok((t, _)) => {
                        pts.push((r as usize, t as usize));
                        best = r;
                    }
                    Err(e) => {
                        stop = short(&e);
                        break;
                    }
                }
            }
            let law = if pts.len() >= 3 {
                let (xs, ys): (Vec<usize>, Vec<usize>) = pts.into_iter().unzip();
                format!("{:?}", fit_law(&xs, &ys).law)
            } else {
                "—".into()
            };
            println!("    {backend:<13} {best:>10}  {law}  {stop}");
        }
    }

    println!();
    println!("== C. joint fit over the whole grid: log t = c + α·log r + β·w ==");
    println!(
        "  {:<13} {:>7} {:>9} {:>9}   marginal-in-w law (confounded)",
        "backend", "points", "α (in r)", "β (per w)"
    );
    for backend in PROBED {
        let mut grid: Vec<(f64, f64, f64)> = Vec::new();
        let mut marg: Vec<(usize, usize)> = Vec::new();
        for w in (8..=16).step_by(2) {
            if ceiling.get(backend).copied().unwrap_or(0) < w {
                break;
            }
            let Some((n, p, q)) = semiprime(w) else {
                continue;
            };
            let mut orders: Vec<(u64, u64)> = (2..200u64)
                .filter_map(|a| order_via_lambda(a, n, p, q).map(|r| (r, a)))
                .collect();
            orders.sort_unstable();
            orders.dedup_by_key(|x| x.0);
            let mut last = 0f64;
            for (r, a) in orders {
                if (r as f64) < last * 8.0 {
                    continue;
                }
                last = r as f64;
                if let Ok((t, _)) = run(&sim, backend, n, a, budget) {
                    grid.push(((r as f64).ln(), w as f64, (t as f64).ln()));
                    marg.push((w, t as usize));
                }
            }
        }
        let marginal = if marg.len() >= 3 {
            let mut m = marg.clone();
            m.sort_unstable();
            m.dedup_by_key(|x| x.0);
            if m.len() >= 3 {
                let (xs, ys): (Vec<usize>, Vec<usize>) = m.into_iter().unzip();
                format!("{:?}", fit_law(&xs, &ys).law)
            } else {
                "—".into()
            }
        } else {
            "—".into()
        };
        // Three parameters: a fit on fewer than six points has almost no
        // residual degrees of freedom and is flagged rather than trusted.
        // Section A measures the same slope with r held constant and is
        // the reliable source for it.
        let weak = if grid.len() < 6 { " (n<6, weak)" } else { "" };
        match joint_fit(&grid) {
            Some((alpha, beta)) => println!(
                "  {backend:<13} {:>7} {alpha:>9.3} {beta:>9.4}   {marginal}{weak}",
                grid.len()
            ),
            None => println!(
                "  {backend:<13} {:>7} {:>9} {:>9}   {marginal}",
                grid.len(),
                "—",
                "—"
            ),
        }
    }
    println!("  α is the exponent of r; β is the natural log-slope per qubit, so");
    println!("  exp(β) is the per-qubit multiplier at fixed support.");

    println!();
    println!("== D. bulk against sparse at fixed support ==");
    println!(
        "  {:>4} {:>12} {:>12} {:>11}",
        "w", "bulk", "sparse", "ratio"
    );
    let bulk_pts = width_only
        .iter()
        .find(|c| c.0 == "bulk")
        .map(|c| c.1.clone())
        .unwrap_or_default();
    let sparse_pts = width_only
        .iter()
        .find(|c| c.0 == "sparse")
        .map(|c| c.1.clone())
        .unwrap_or_default();
    for (w, bt) in &bulk_pts {
        if let Some((_, st)) = sparse_pts.iter().find(|e| e.0 == *w) {
            println!(
                "  {w:>4} {:>12} {:>12} {:>10.1}x",
                human(*bt),
                human(*st),
                *bt as f64 / (*st).max(1) as f64
            );
        }
    }
    Ok(())
}
