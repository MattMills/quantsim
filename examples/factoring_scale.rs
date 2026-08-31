//! Scale each factoring mechanism until it stops, and report where.
//!
//! Run with `cargo run --release --example factoring_scale`.
//!
//! Every (mechanism, representation) pair climbs the width on its own
//! until a run refuses or overruns its budget; nothing is capped at a
//! fixed width. Both endings are recorded — a refusal names the wall, an
//! overrun names the budget — and the last width that completed is the
//! measurement.
//!
//! `dense` is absent: it holds `2^n` slots occupied or not, so its limit
//! is arithmetic rather than a measurement.

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

/// The balanced semiprime `p·q` of exactly `w` bits with `p` as large as
/// the range allows.
fn semiprime(w: usize) -> Option<(u64, u64, u64)> {
    if !(4..=63).contains(&w) {
        return None;
    }
    let lo = 1u64 << (w - 1);
    let hi = (1u64 << w) - 1;
    // Largest `p` whose partner still lands in `[lo, hi]`; `q > p`, so a
    // perfect square is never returned.
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
        v => format!("{:.1}MB", v as f64 / 1e6),
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

/// One completed run at one width.
#[derive(Clone)]
struct Point {
    w: usize,
    n: u64,
    r: usize,
    qubits: usize,
    nanos: u64,
    bytes: usize,
    support: usize,
}

/// Climb `w` from `start` until `run` fails, returning every completed
/// point and the message that stopped it.
fn climb(
    start: usize,
    budget: Duration,
    mut run: impl FnMut(u64, usize) -> Result<Point>,
) -> (Vec<Point>, String) {
    let mut points = Vec::new();
    for w in start..=63 {
        let Some((n, _, _)) = semiprime(w) else {
            return (points, "no semiprime at this width".into());
        };
        match guard::with_time_budget(budget, || run(n, w)) {
            Ok(p) => points.push(p),
            Err(e) => return (points, wall(&e)),
        }
    }
    (points, "width ceiling".into())
}

fn fit(points: &[Point], key: impl Fn(&Point) -> usize) -> String {
    if points.len() < 3 {
        return "—".into();
    }
    let xs: Vec<usize> = points.iter().map(&key).collect();
    let ys: Vec<usize> = points.iter().map(|p| p.nanos as usize).collect();
    let mut pairs: Vec<(usize, usize)> = xs.into_iter().zip(ys).collect();
    pairs.sort_unstable();
    pairs.dedup_by_key(|p| p.0);
    if pairs.len() < 3 {
        return "—".into();
    }
    let (a, b): (Vec<usize>, Vec<usize>) = pairs.into_iter().unzip();
    format!("{:?}", fit_law(&a, &b).law)
}

fn main() -> Result<()> {
    guard::set_memory_limit(Some(1 << 30));
    let mut sim: Simulator = Simulator::new();
    register_every_representation(&mut sim)?;
    let budget = Duration::from_secs(2);

    println!("budget {budget:?}/run, memory limit 1GiB, 1 run per width");

    println!();
    println!("== semiclassical order finding, each representation to its own wall ==");
    println!(
        "  {:<17} {:>5} {:>11} {:>9} {:>9} {:>9} {:>9}  stopped by",
        "backend", "max w", "N", "r", "time", "bytes", "support"
    );
    let mut curves: Vec<(&str, Vec<Point>)> = Vec::new();
    for name in BACKENDS {
        let (points, stop) = climb(4, budget, |n, _| {
            let f = OrderFinder::new(n, first_coprime(n))?.on_backend(name);
            let t = Instant::now();
            let e = f.estimate(&sim, PhaseForm::Semiclassical, &mut Prng::new(7))?;
            Ok(Point {
                w: f.work_bits(),
                n,
                r: e.peak_orbit,
                qubits: e.qubits,
                nanos: t.elapsed().as_nanos() as u64,
                bytes: e.peak_bytes,
                support: e.peak_support,
            })
        });
        match points.last() {
            Some(p) => println!(
                "  {name:<17} {:>5} {:>11} {:>9} {:>9} {:>9} {:>9}  {stop}",
                p.w,
                p.n,
                p.r,
                human(p.nanos),
                bytes(p.bytes),
                p.support
            ),
            None => println!(
                "  {name:<17} {:>5} {:>11} {:>9} {:>9} {:>9} {:>9}  {stop}",
                "—", "—", "—", "—", "—", "—"
            ),
        }
        curves.push((name, points));
    }

    println!();
    println!("== the same curves, width by width (ms) ==");
    let deepest = curves.iter().map(|c| c.1.len()).max().unwrap_or(0);
    print!("  {:<17}", "backend / w");
    let widths: Vec<usize> = curves
        .iter()
        .max_by_key(|c| c.1.len())
        .map(|c| c.1.iter().map(|p| p.w).collect())
        .unwrap_or_default();
    for w in &widths {
        print!("{w:>8}");
    }
    println!("   time law in w      time law in r");
    for (name, points) in &curves {
        if points.is_empty() {
            continue;
        }
        print!("  {name:<17}");
        for i in 0..deepest {
            match points.get(i) {
                Some(p) => print!("{:>8.1}", p.nanos as f64 / 1e6),
                None => print!("{:>8}", "—"),
            }
        }
        println!("   {:<18} {}", fit(points, |p| p.w), fit(points, |p| p.r));
    }

    println!();
    println!("== full-register order finding, to its own wall ==");
    println!(
        "  {:<17} {:>5} {:>7} {:>11} {:>9} {:>9} {:>9} {:>11}  stopped by",
        "backend", "max w", "qubits", "N", "r", "time", "bytes", "support"
    );
    for name in [
        "sparse",
        "adaptive",
        "mosaic",
        "phase-field",
        "framed-sparse",
    ] {
        let (points, stop) = climb(4, budget, |n, _| {
            let f = OrderFinder::new(n, first_coprime(n))?.on_backend(name);
            let t = Instant::now();
            let e = f.estimate(&sim, PhaseForm::FullRegister, &mut Prng::new(7))?;
            Ok(Point {
                w: f.work_bits(),
                n,
                r: e.peak_orbit,
                qubits: e.qubits,
                nanos: t.elapsed().as_nanos() as u64,
                bytes: e.peak_bytes,
                support: e.peak_support,
            })
        });
        match points.last() {
            Some(p) => println!(
                "  {name:<17} {:>5} {:>7} {:>11} {:>9} {:>9} {:>9} {:>11}  {stop}",
                p.w,
                p.qubits,
                p.n,
                p.r,
                human(p.nanos),
                bytes(p.bytes),
                p.support
            ),
            None => println!("  {name:<17} {:>5}   {stop}", "—"),
        }
    }

    println!();
    println!("== regev, each schedule to its own wall (d = 2, R = w) ==");
    println!(
        "  {:<12} {:<12} {:>5} {:>9} {:>7} {:>9} {:>9} {:>9} {:>6}  stopped by",
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
            for w in 4..=63usize {
                let Some((n, _, _)) = semiprime(w) else {
                    stop = "no semiprime at this width".into();
                    break;
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
                        ));
                    }
                    Err(e) => {
                        stop = wall(&e);
                        break;
                    }
                }
            }
            match last {
                Some((w, n, q, t, b, sup, mults)) => println!(
                    "  {schedule:<12} {name:<12} {w:>5} {n:>9} {q:>7} {:>9} {:>9} {:>9} {mults:>6}  {stop}",
                    human(t),
                    bytes(b),
                    sup
                ),
                None => println!(
                    "  {schedule:<12} {name:<12} {:>5}  {stop}",
                    "—"
                ),
            }
        }
    }

    println!();
    println!("== structural limits reached above, not measured ==");
    println!("  shor::MAX_MODULUS      {}", shor::MAX_MODULUS);
    println!("  phase bits <= 62       w <= 30 for 2w+1 phase bits");
    println!("  regev::MAX_WIDTH       {}", regev::MAX_WIDTH);
    println!("  sparse basis index     63 qubits");
    Ok(())
}
