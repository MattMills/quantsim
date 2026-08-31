//! Scale each factoring mechanism until it stops, and report where.
//!
//! Run with `cargo run --release --example factoring_scale`.
//!
//! The cost of semiclassical order finding is `O(t · r)` — rounds times
//! support — and the support *is* the multiplicative order `r`. The width
//! `w` enters only through `t = 2w+1`, and `r` jumps by large, erratic
//! factors between adjacent widths. So the ladder here is over `r`, not
//! over `w`: climbing `w` measures the ladder rather than the mechanism,
//! and a hard budget then quantizes every representation to the same
//! width step regardless of its constant factor.
//!
//! Orders are computed from `λ(N) = lcm(p−1, q−1)` and its factorization
//! rather than by iteration, which is `O(log λ)` instead of `O(r)` and is
//! what makes a ladder reaching `r > 10^7` constructible at all.
//!
//! `dense` is absent: it holds `2^n` slots occupied or not, so its limit
//! is arithmetic rather than a measurement. The `xdense` column reports
//! every other row's bytes as a multiple of that vector, because a label
//! is not a measurement: a representation sitting at `1.00` has
//! materialized whatever it is called, and the wall it then hits is its
//! own escape hatch rather than the computation's cost.

use std::time::{Duration, Instant};

use quantsim::bounds::fit_law;
use quantsim::guard;
use quantsim::prelude::*;
use quantsim::regev::{self, ExpSchedule, Regev};
use quantsim::shor::{self, OrderFinder, PhaseForm};

const BACKENDS: [&str; 18] = [
    "sparse",
    "adaptive",
    "factored",
    "mps",
    "mera",
    "bulk",
    "mosaic",
    "bundle",
    "branched",
    "e8-constellation",
    "e8-rep",
    "phase-field",
    "clifford-frame",
    "interference",
    "braided",
    "logical",
    "device-linear",
    "framed-sparse",
];

fn register_every_representation(sim: &mut Simulator) -> Result<()> {
    use quantsim::backend::{
        ArityPolicy, BraidedState, CliffordFramedState, DeviceState, DurationModel, FramedState,
        InterferenceState, PhaseFieldState, SparseState, Topology,
    };
    use quantsim::logical::LogicalState;

    let r = sim.backends_mut();
    r.register_e8()?;
    r.register("phase-field", |n| Ok(Box::new(PhaseFieldState::new(n)?)))?;
    r.register("clifford-frame", |n| {
        Ok(Box::new(CliffordFramedState::new(n)?))
    })?;
    r.register("interference", |n| Ok(Box::new(InterferenceState::new(n)?)))?;
    r.register("braided", |n| Ok(Box::new(BraidedState::new(n)?)))?;
    r.register("logical", |n| Ok(Box::new(LogicalState::tiled(n)?)))?;
    r.register("device-linear", |n| {
        Ok(Box::new(DeviceState::new(
            Topology::linear(n),
            DurationModel::default(),
            ArityPolicy::default(),
        )?))
    })?;
    r.register("framed-sparse", |n| {
        Ok(Box::new(FramedState::new(Box::new(SparseState::new(n)?))))
    })?;
    Ok(())
}

fn semiprime(w: usize) -> Option<(u64, u64, u64)> {
    if !(4..=62).contains(&w) {
        return None;
    }
    let lo = 1u64 << (w - 1);
    let hi = (1u64 << w) - 1;
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

/// Trial-division factorization; only ever called on `p−1`, `q−1`, both
/// below `2^32`, so the `√m` bound is cheap.
fn factor_small(mut m: u64) -> Vec<(u64, u32)> {
    let mut out = Vec::new();
    let mut d = 2u64;
    while d.saturating_mul(d) <= m {
        if m % d == 0 {
            let mut e = 0;
            while m % d == 0 {
                m /= d;
                e += 1;
            }
            out.push((d, e));
        }
        d += if d == 2 { 1 } else { 2 };
    }
    if m > 1 {
        out.push((m, 1));
    }
    out
}

/// `ord_N(a)` for `N = p·q`, from `λ(N)` and its factorization: divide
/// `λ` by each prime while the reduced exponent still gives 1. `O(log λ)`
/// modular exponentiations rather than `O(r)` multiplications, which is
/// the difference between a ladder that reaches `10^7` and one that does
/// not.
fn order_via_lambda(a: u64, n: u64, p: u64, q: u64) -> Option<u64> {
    if quantsim::padic::gcd(a % n, n) != 1 {
        return None;
    }
    let g = quantsim::padic::gcd(p - 1, q - 1);
    let lambda = (p - 1) / g * (q - 1);
    let mut primes: Vec<u64> = factor_small(p - 1)
        .into_iter()
        .chain(factor_small(q - 1))
        .map(|(pr, _)| pr)
        .collect();
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

fn first_coprime(n: u64) -> u64 {
    (2..n)
        .find(|&a| quantsim::padic::gcd(a, n) == 1)
        .unwrap_or(2)
}

fn human(nanos: u64) -> String {
    match nanos {
        n if n < 10_000 => format!("{n}ns"),
        n if n < 10_000_000 => format!("{:.1}µs", n as f64 / 1e3),
        n if n < 10_000_000_000 => format!("{:.1}ms", n as f64 / 1e6),
        n => format!("{:.2}s", n as f64 / 1e9),
    }
}

fn bytes(b: usize) -> String {
    match b {
        v if v < 10_000 => format!("{v}B"),
        v if v < 10_000_000 => format!("{:.1}KB", v as f64 / 1e3),
        v if v < 10_000_000_000 => format!("{:.1}MB", v as f64 / 1e6),
        v => format!("{:.2}GB", v as f64 / 1e9),
    }
}

fn wall(e: &quantsim::Error) -> String {
    if let quantsim::Error::Timeout { elapsed_ms, .. } = e {
        return format!("over budget at {elapsed_ms}ms");
    }
    let s = e.to_string();
    let s = s.split(';').next().unwrap_or(&s).trim().to_string();
    if s.chars().count() > 44 {
        format!("{}…", s.chars().take(43).collect::<String>())
    } else {
        s
    }
}

/// One rung: a modulus and base whose order is close to a target.
#[derive(Clone, Copy)]
struct Rung {
    r: u64,
    n: u64,
    a: u64,
    w: usize,
}

/// Every `(N, a)` this program can build, one per distinct order, sorted.
fn order_ladder(step: f64, max_r: u64) -> Vec<Rung> {
    let mut pool: Vec<Rung> = Vec::new();
    for w in 4..=62usize {
        let Some((n, p, q)) = semiprime(w) else {
            continue;
        };
        for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            if a >= n {
                continue;
            }
            if let Some(r) = order_via_lambda(a, n, p, q) {
                if r >= 2 && r <= max_r {
                    pool.push(Rung { r, n, a, w });
                }
            }
        }
    }
    pool.sort_unstable_by_key(|x| (x.r, x.w));
    pool.dedup_by_key(|x| x.r);
    let mut ladder: Vec<Rung> = Vec::new();
    let mut last = 0f64;
    for rung in pool {
        if rung.r as f64 >= last * step {
            last = rung.r as f64;
            ladder.push(rung);
        }
    }
    ladder
}

struct Point {
    rung: Rung,
    nanos: u64,
    bytes: usize,
    support: usize,
    qubits: usize,
}

/// Reported bytes as a multiple of the dense `2^n` vector for the same
/// width — the number that says whether a representation is actually
/// compressing anything.
///
/// It is worth a column because a label is not a measurement.
/// `phase_field::load` escapes to dense by construction, so a row named
/// "phase-field" is `DenseState` from the first permutation kernel
/// onward, and it sits at exactly 1.00x here — including where the
/// support is 28 of 2^15. The wall it then reports (137 GB at 33 qubits)
/// is that escape, not the computation: the identical case completes on
/// sparse, adaptive, mosaic and e8-constellation, at 1.6 MB and 19 ms.
fn dense_ratio(p: &Point) -> Option<f64> {
    if p.qubits >= 60 {
        return None;
    }
    let dense = (1u64 << p.qubits) as f64 * std::mem::size_of::<C64>() as f64;
    Some(p.bytes as f64 / dense)
}

fn ratio_cell(p: &Point) -> String {
    match dense_ratio(p) {
        None => "—".into(),
        Some(r) if r < 0.01 => format!("{r:.4}"),
        Some(r) => format!("{r:.2}"),
    }
}

fn climb(
    ladder: &[Rung],
    budget: Duration,
    mut run: impl FnMut(Rung) -> Result<Point>,
) -> (Vec<Point>, String) {
    let mut points = Vec::new();
    for &rung in ladder {
        match guard::with_time_budget(budget, || run(rung)) {
            Ok(p) => points.push(p),
            Err(e) => return (points, wall(&e)),
        }
    }
    (points, "ladder exhausted".into())
}

fn fit(points: &[Point], key: impl Fn(&Point) -> usize) -> String {
    if points.len() < 3 {
        return "—".into();
    }
    let mut pairs: Vec<(usize, usize)> =
        points.iter().map(|p| (key(p), p.nanos as usize)).collect();
    pairs.sort_unstable();
    pairs.dedup_by_key(|p| p.0);
    if pairs.len() < 3 {
        return "—".into();
    }
    let (a, b): (Vec<usize>, Vec<usize>) = pairs.into_iter().unzip();
    format!("{:?}", fit_law(&a, &b).law)
}

fn main() -> Result<()> {
    let available = guard::measured_available();
    // Well below the machine: at a one-second budget nothing gets near
    // it, and a limit the process cannot actually honour is how the
    // previous cut of this file got OOM-killed rather than refused.
    let limit = (4usize << 30).min(available / 2);
    guard::set_memory_limit(Some(limit));
    let mut sim: Simulator = Simulator::new();
    register_every_representation(&mut sim)?;

    let budget = Duration::from_secs(1);
    // A sparse entry is ~29 B; cap the ladder where the map alone would
    // fill the budget, so the wall found is time or a refusal, not a
    // ladder that ran out.
    let max_r = (limit / 29) as u64;
    let ladder = order_ladder(1.15, max_r);

    println!(
        "memory {} available, limit {}, ladder r = {}..{} over {} rungs (x1.15), max_r cap {}",
        bytes(available),
        bytes(limit),
        ladder.first().map_or(0, |r| r.r),
        ladder.last().map_or(0, |r| r.r),
        ladder.len(),
        max_r
    );

    println!();
    println!("== semiclassical order finding, scaled on r, budget {budget:?}/run ==");
    println!(
        "  {:<17} {:>12} {:>20} {:>4} {:>3} {:>9} {:>9} {:>12} {:>7}  stopped by",
        "backend", "max r", "N", "a", "w", "time", "bytes", "support", "xdense"
    );
    let mut curves: Vec<(&str, Vec<Point>)> = Vec::new();
    for name in BACKENDS {
        let (points, stop) = climb(&ladder, budget, |rung| {
            let f = OrderFinder::new(rung.n, rung.a)?.on_backend(name);
            let t = Instant::now();
            let e = f.estimate(&sim, PhaseForm::Semiclassical, &mut Prng::new(7))?;
            Ok(Point {
                rung,
                nanos: t.elapsed().as_nanos() as u64,
                bytes: e.peak_bytes,
                support: e.peak_support,
                qubits: e.qubits,
            })
        });
        match points.last() {
            Some(p) => println!(
                "  {name:<17} {:>12} {:>20} {:>4} {:>3} {:>9} {:>9} {:>12} {:>7}  {stop}",
                p.rung.r,
                p.rung.n,
                p.rung.a,
                p.rung.w,
                human(p.nanos),
                bytes(p.bytes),
                p.support,
                ratio_cell(p)
            ),
            None => println!(
                "  {name:<17} {:>12} {:>20} {:>4} {:>3} {:>9} {:>9} {:>12} {:>7}  {stop}",
                "—", "—", "—", "—", "—", "—", "—", "—"
            ),
        }
        curves.push((name, points));
    }

    println!();
    println!("== fitted laws over each climb ==");
    println!(
        "  {:<17} {:>6}  {:<38} time in w",
        "backend", "rungs", "time in r"
    );
    for (name, points) in &curves {
        if points.len() < 3 {
            continue;
        }
        println!(
            "  {name:<17} {:>6}  {:<38} {}",
            points.len(),
            fit(points, |p| p.rung.r as usize),
            fit(points, |p| p.rung.w)
        );
    }

    println!();
    println!("== full-register order finding, scaled on w ==");
    println!(
        "  {:<17} {:>5} {:>7} {:>12} {:>9} {:>10} {:>14}  stopped by",
        "backend", "max w", "qubits", "N", "time", "bytes", "support"
    );
    for name in [
        "sparse",
        "adaptive",
        "mosaic",
        "phase-field",
        "framed-sparse",
    ] {
        let mut last: Option<(usize, usize, u64, u64, usize, usize)> = None;
        let mut stop = String::from("width ceiling");
        for w in 4..=62usize {
            let Some((n, _, _)) = semiprime(w) else {
                continue;
            };
            let f = match OrderFinder::new(n, first_coprime(n)) {
                Ok(f) => f.on_backend(name),
                Err(e) => {
                    stop = wall(&e);
                    break;
                }
            };
            let t = Instant::now();
            match guard::with_time_budget(budget, || {
                f.estimate(&sim, PhaseForm::FullRegister, &mut Prng::new(7))
            }) {
                Ok(e) => {
                    last = Some((
                        w,
                        e.qubits,
                        n,
                        t.elapsed().as_nanos() as u64,
                        e.peak_bytes,
                        e.peak_support,
                    ))
                }
                Err(e) => {
                    stop = wall(&e);
                    break;
                }
            }
        }
        match last {
            Some((w, q, n, t, b, sup)) => println!(
                "  {name:<17} {w:>5} {q:>7} {n:>12} {:>9} {:>10} {:>14}  {stop}",
                human(t),
                bytes(b),
                sup
            ),
            None => println!("  {name:<17} {:>5}  {stop}", "—"),
        }
    }

    println!();
    println!("== regev, each schedule scaled on w (d = 2, R = w) ==");
    println!(
        "  {:<12} {:<12} {:>5} {:>12} {:>7} {:>9} {:>10} {:>12} {:>6}  stopped by",
        "schedule", "backend", "max w", "N", "qubits", "time", "bytes", "support", "mults"
    );
    for schedule in [
        ExpSchedule::Sequential,
        ExpSchedule::Regev,
        ExpSchedule::Fibonacci,
    ] {
        for name in ["sparse", "mosaic"] {
            let mut last: Option<(usize, u64, usize, u64, usize, usize, usize)> = None;
            let mut stop = String::from("width ceiling");
            for w in 4..=62usize {
                let Some((n, _, _)) = semiprime(w) else {
                    continue;
                };
                let cfg = match Regev::new(n, 2) {
                    Ok(c) => c.with_schedule(schedule).on_backend(name),
                    Err(e) => {
                        stop = wall(&e);
                        break;
                    }
                };
                let q = cfg.layout().qubits;
                if q > regev::MAX_WIDTH {
                    stop = format!("{q} qubits past the u64 basis index");
                    break;
                }
                let t = Instant::now();
                match guard::with_time_budget(budget, || cfg.sample(&sim, &mut Prng::new(7))) {
                    Ok(s) => {
                        last = Some((
                            w,
                            n,
                            q,
                            t.elapsed().as_nanos() as u64,
                            s.cost.peak_bytes,
                            s.cost.peak_support,
                            s.cost.full_multiplications,
                        ))
                    }
                    Err(e) => {
                        stop = wall(&e);
                        break;
                    }
                }
            }
            match last {
                Some((w, n, q, t, b, sup, mults)) => println!(
                    "  {schedule:<12} {name:<12} {w:>5} {n:>12} {q:>7} {:>9} {:>10} {:>12} {mults:>6}  {stop}",
                    human(t),
                    bytes(b),
                    sup
                ),
                None => println!("  {schedule:<12} {name:<12} {:>5}  {stop}", "—"),
            }
        }
    }

    println!();
    println!("== structural limits ==");
    println!("  shor::MAX_MODULUS      {} = 2^62", shor::MAX_MODULUS);
    println!("  shor::MAX_PHASE_BITS   {}", shor::MAX_PHASE_BITS);
    println!("  regev::MAX_WIDTH       {}", regev::MAX_WIDTH);
    println!("  sparse basis index     63 qubits");
    Ok(())
}
