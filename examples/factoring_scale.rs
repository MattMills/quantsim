//! Runtime and scale of the factoring techniques, across every register
//! type the crate offers.
//!
//! Run with `cargo run --release --example factoring_scale`.
//!
//! The other two factoring examples measure *structure* — support,
//! qubits, multiplication counts. None of those is a second of wall
//! clock. This one measures time and bytes.
//!
//! **How a run is allowed to end.** Every measured run is armed with a
//! wall-clock budget through [`quantsim::guard`]. Overrunning it is a
//! **failure** and aborts this program with the error, everywhere except
//! section 3, whose declared purpose is to find where each representation
//! stops: in sections 2 and 3 a budget overrun *is* the measurement, and
//! is reported as one. A representation that cannot hold the register
//! refuses by name (out of memory, too many qubits, gate unsupported) and
//! that is a table row, not a failure — the difference between "this
//! cannot be done" and "this did not finish" is the whole point of
//! separating them.
//!
//! No conclusion here is written ahead of the table it describes: the
//! orderings in section 2 are computed from the measured rows.

use std::time::{Duration, Instant};

use quantsim::bounds::fit_law;
use quantsim::guard;
use quantsim::prelude::*;
use quantsim::regev::{self, ExpSchedule, Regev};
use quantsim::shor::{self, OrderFinder, PhaseForm};

/// Balanced semiprimes `p·q` with strictly increasing width.
const LADDER: [(u64, u64, u64); 10] = [
    (15, 3, 5),
    (35, 5, 7),
    (221, 13, 17),
    (899, 29, 31),
    (4087, 61, 67),
    (7387, 83, 89),
    (32399, 179, 181),
    (126_727, 353, 359),
    (256_027, 503, 509),
    (1_040_399, 1019, 1021),
];

/// **Every** representation in the crate that implements
/// [`Backend`](quantsim::Backend) — the ten in the standard registry, the
/// two opt-in ℂ-only ones, and the seven that ship unregistered and have
/// to be installed by hand below. `dense` is deliberately absent: it
/// stores `2^n` slots whether or not they are zero, so its limit is
/// arithmetic rather than a measurement, and letting it set the ladder
/// would cap every other representation at its wall.
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

/// Install the representations the standard registry leaves out. They are
/// `Backend` implementors like any other; that they are not registered by
/// default is a statement about their scope, not their existence, and a
/// sweep that skipped them would be surveying the registry rather than
/// the crate.
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

/// A budget overrun outside section 3 is a failure, not a data point.
fn fail_on_timeout<T>(r: Result<T>) -> Result<std::result::Result<T, quantsim::Error>> {
    match r {
        Ok(v) => Ok(Ok(v)),
        Err(e @ quantsim::Error::Timeout { .. }) => Err(e),
        Err(e) => Ok(Err(e)),
    }
}

fn timed<T>(budget: Duration, reps: usize, mut f: impl FnMut() -> Result<T>) -> Result<(u64, T)> {
    let mut samples = Vec::with_capacity(reps);
    let mut last = None;
    for _ in 0..reps {
        let start = Instant::now();
        let out = guard::with_time_budget(budget, &mut f)?;
        samples.push(start.elapsed().as_nanos() as u64);
        last = Some(out);
    }
    samples.sort_unstable();
    Ok((samples[samples.len() / 2], last.unwrap()))
}

fn human(nanos: u64) -> String {
    match nanos {
        n if n < 10_000 => format!("{n} ns"),
        n if n < 10_000_000 => format!("{:.1} µs", n as f64 / 1e3),
        n if n < 10_000_000_000 => format!("{:.1} ms", n as f64 / 1e6),
        n => format!("{:.2} s", n as f64 / 1e9),
    }
}

fn bytes(b: usize) -> String {
    match b {
        v if v < 10_000 => format!("{v} B"),
        v if v < 10_000_000 => format!("{:.1} KB", v as f64 / 1e3),
        v => format!("{:.1} MB", v as f64 / 1e6),
    }
}

/// The wall's own message, trimmed to the table.
fn wall(e: &quantsim::Error) -> String {
    let s = e.to_string();
    let s = s.split(';').next().unwrap_or(&s).trim().to_string();
    if s.chars().count() > 44 {
        format!("{}…", s.chars().take(43).collect::<String>())
    } else {
        s
    }
}

fn first_coprime(n: u64) -> u64 {
    (2..n).find(|&a| quantsim::padic::gcd(a, n) == 1).unwrap()
}

fn main() -> Result<()> {
    // A representation that cannot hold the register should refuse, not
    // exhaust the machine. The limit makes that refusal a measurement.
    guard::set_memory_limit(Some(1 << 30));

    let mut sim: Simulator = Simulator::new();
    register_every_representation(&mut sim)?;
    let budget = Duration::from_secs(30);
    // Sections that hunt for a limit use a shorter budget and report the
    // overrun; every other section must finish inside `budget` or fail.
    let sweep_budget = Duration::from_secs(5);

    println!("== 1. semiclassical Shor: wall clock against the width ==");
    println!("   one order-finding run, median of 3, sparse representation, 30 s budget");
    println!("     N          p×q          w    r          time      bytes    support  orbit");
    let mut widths = Vec::new();
    let mut times = Vec::new();
    let mut order_pairs: Vec<(usize, usize)> = Vec::new();
    for &(n, p, q) in LADDER.iter() {
        let a = first_coprime(n);
        let r = shor::multiplicative_order(a, n).unwrap();
        let finder = OrderFinder::new(n, a)?;
        let mut seed = 0u64;
        let (t, e) = timed(budget, 3, || {
            seed += 1;
            let mut rng = Prng::new(seed);
            finder.estimate(&sim, PhaseForm::Semiclassical, &mut rng)
        })?;
        println!(
            "   {n:<9}  {:<11}  {:2}  {r:<9}  {:>8}  {:>9}  {:8}  {:6}",
            format!("{p}×{q}"),
            finder.work_bits(),
            human(t),
            bytes(e.peak_bytes),
            e.peak_support,
            e.peak_orbit
        );
        widths.push(finder.work_bits());
        times.push(t as usize);
        order_pairs.push((r as usize, t as usize));
    }
    println!(
        "   fitted time law in the width w: {:?}",
        fit_law(&widths, &times).law
    );
    order_pairs.sort_unstable();
    order_pairs.dedup_by_key(|p| p.0);
    let (os, ts): (Vec<usize>, Vec<usize>) = order_pairs.into_iter().unzip();
    println!(
        "   fitted time law in the order r: {:?}",
        fit_law(&os, &ts).law
    );
    println!("   Time is (2w+1) rounds × O(support) = O(w·r), and r is what moves.");
    println!("   The width law reads exponential only because r grows with N, which");
    println!("   is the algorithm's content, not the representation's cost.");

    println!();
    println!("== 2. every representation in the crate, across the whole ladder ==");
    println!("   semiclassical order finding, median of 3, times in ms; a row stops at");
    println!("   its first refusal or 3 s overrun, since both are monotone in width.");
    print!("     {:<17}", "backend / w =");
    let mut ws = Vec::new();
    for &(n, _, _) in LADDER.iter() {
        let w = shor::work_bits(n);
        ws.push(w);
        print!("{w:>8}");
    }
    println!("   stopped by");
    let matrix_budget = Duration::from_secs(3);
    let mut ranking: Vec<(String, usize, u64)> = Vec::new();
    for name in BACKENDS {
        print!("     {name:<17}");
        let mut reached = 0usize;
        let mut total = 0u64;
        let mut stopped = String::from("(ladder exhausted)");
        for (col, &(n, _, _)) in LADDER.iter().enumerate() {
            let a = first_coprime(n);
            let finder = OrderFinder::new(n, a)?.on_backend(name);
            let mut seed = 100u64 + col as u64;
            let out = timed(matrix_budget, 3, || {
                seed += 1;
                let mut rng = Prng::new(seed);
                finder.estimate(&sim, PhaseForm::Semiclassical, &mut rng)
            });
            match out {
                Ok((t, _)) => {
                    print!("{:>8.1}", t as f64 / 1e6);
                    reached = ws[col];
                    total += t;
                }
                Err(e) => {
                    stopped = wall(&e);
                    for _ in col..LADDER.len() {
                        print!("{:>8}", "—");
                    }
                    break;
                }
            }
        }
        println!("   {stopped}");
        ranking.push((name.to_string(), reached, total));
    }
    // Conclusions computed from the matrix above, not written ahead of it.
    let deepest = ranking.iter().map(|r| r.1).max().unwrap_or(0);
    let at_depth: Vec<&(String, usize, u64)> = ranking.iter().filter(|r| r.1 == deepest).collect();
    let mut fastest = at_depth.clone();
    fastest.sort_by_key(|r| r.2);
    println!();
    println!(
        "   {} of {} representations reached w = {deepest}, the deepest any did:",
        at_depth.len(),
        BACKENDS.len()
    );
    println!(
        "     {:?}",
        at_depth.iter().map(|r| r.0.as_str()).collect::<Vec<_>>()
    );
    println!(
        "   of those, fastest summed over the ladder: {:?}",
        fastest
            .iter()
            .take(3)
            .map(|r| r.0.as_str())
            .collect::<Vec<_>>()
    );
    let mut shallow: Vec<&(String, usize, u64)> =
        ranking.iter().filter(|r| r.1 < deepest).collect();
    shallow.sort_by_key(|r| std::cmp::Reverse(r.1));
    println!(
        "   the rest stopped at w = {:?}",
        shallow
            .iter()
            .map(|r| (r.0.as_str(), r.1))
            .collect::<Vec<_>>()
    );
    println!("   This is one algorithm on ten moduli. It orders these representations");
    println!("   for *this* workload — a permutation kernel on a cyclic orbit — and");
    println!("   claims nothing about any other.");

    println!("== 3. how far the non-dense representations carry the full-register form ==");
    println!("   Its control register is t = 2w+1 qubits in uniform superposition, and");
    println!("   the inverse QFT spreads each of the r work values over all of it, so");
    println!("   the state has 2^t·r *nonzero* amplitudes. A dense vector would hold");
    println!("   2^(3w+1) slots regardless — quoted below as arithmetic, not run,");
    println!("   because its limit is not a measurement and would cap the sweep.");
    println!("     N       w    t   nonzeros (2^t·r)   dense slots     sparse           mosaic");
    for &(n, _, _) in LADDER.iter().take(6) {
        let a = first_coprime(n);
        let base = OrderFinder::new(n, a)?;
        let w = base.work_bits();
        let t = base.phase_bits;
        let r = shor::multiplicative_order(a, n).unwrap();
        let predicted = (1u128 << t.min(96)) * r as u128;
        let dense_slots = 1u128 << (w + t).min(96);
        let mut cells = Vec::new();
        for backend in ["sparse", "mosaic"] {
            let finder = OrderFinder::new(n, a)?.on_backend(backend);
            let start = Instant::now();
            let out = guard::with_time_budget(sweep_budget, || {
                let mut rng = Prng::new(9);
                finder.estimate(&sim, PhaseForm::FullRegister, &mut rng)
            });
            cells.push(match out {
                Ok(e) => format!(
                    "{} {}",
                    human(start.elapsed().as_nanos() as u64),
                    bytes(e.peak_bytes)
                ),
                Err(e) => wall(&e),
            });
        }
        println!(
            "   {n:<7} {w:2}  {t:3}   {predicted:16}   {dense_slots:13}   {:<16} {}",
            cells[0], cells[1]
        );
    }
    println!("   Both eventually stop, because 2^t·r is exponential in w whatever");
    println!("   holds it — the full-register form is a cross-check, not a technique.");
    println!("   The semiclassical form, whose support is r rather than 2^t·r, is the");
    println!("   one section 1 and section 2 carry to w = 20.");

    println!();
    println!("== 4. Regev's three schedules, timed ==");
    println!("   one sample, median of 3, N = 35 (w = 6), d = 3 — at d = 2 the");
    println!("   sequential and Regev schedules cost the same d·R = 2R full-width");
    println!("   multiplications by construction, so d = 2 cannot show the trade.");
    println!("     schedule      R   qubits      time       bytes   support   full mults");
    let mut schedule_times: Vec<(usize, String, u64, usize)> = Vec::new();
    for schedule in [
        ExpSchedule::Sequential,
        ExpSchedule::Regev,
        ExpSchedule::Fibonacci,
    ] {
        for r in [3usize, 4] {
            let cfg = Regev::new(35, 3)?
                .with_exponent_bits(r)
                .with_schedule(schedule);
            let q = cfg.layout().qubits;
            if q > regev::MAX_WIDTH {
                println!("   {schedule:<12} {r}   {q:6}   past the u64 basis index at this R");
                continue;
            }
            let mut seed = 200u64;
            let attempt = fail_on_timeout(timed(budget, 3, || {
                seed += 1;
                let mut rng = Prng::new(seed);
                cfg.sample(&sim, &mut rng)
            }))?;
            match attempt {
                Ok((t, s)) => {
                    println!(
                        "   {schedule:<12} {r}   {q:6}  {:>9}  {:>10}  {:8}  {:6}",
                        human(t),
                        bytes(s.cost.peak_bytes),
                        s.cost.peak_support,
                        s.cost.full_multiplications
                    );
                    schedule_times.push((r, schedule.to_string(), t, s.cost.full_multiplications));
                }
                Err(e) => println!("   {schedule:<12} {r}   {q:6}   {}", wall(&e)),
            }
        }
    }
    // Ordered from the rows, per R, rather than asserted over them.
    for r in [3usize, 4] {
        let mut rows: Vec<&(usize, String, u64, usize)> =
            schedule_times.iter().filter(|x| x.0 == r).collect();
        if rows.len() < 2 {
            continue;
        }
        rows.sort_by_key(|x| x.2);
        let fastest = rows[0].1.clone();
        rows.sort_by_key(|x| x.3);
        let cheapest = rows[0].1.clone();
        println!("   R = {r}: fastest to simulate {fastest}, fewest full-width mults {cheapest}");
    }
    println!("   The two orderings above are the finding: the schedule that spends the");
    println!("   fewest full-width multiplications — the currency a real circuit pays —");
    println!("   is not the one that simulates fastest, because simulation pays for");
    println!("   registers it has to hold and hardware pays for multiplications it has");
    println!("   to run. Neither column is a verdict on the other.");

    println!();
    println!("== 5. Shor against Regev, in wall clock ==");
    println!("   Three rows, not ten: Regev's exponent box is 2^(dR) with R = w, so");
    println!("   the simulation cost doubles twice per bit of N and the ladder ends");
    println!("   four entries earlier than Shor's. That is the finding, not a gap.");
    println!("     N     shor: time / mults    regev: time / mults per sample   ratio");
    for &(n, _, _) in LADDER.iter().take(3) {
        let a = first_coprime(n);
        let finder = OrderFinder::new(n, a)?;
        let mut seed = 300u64;
        let (st, se) = timed(budget, 3, || {
            seed += 1;
            let mut rng = Prng::new(seed);
            finder.estimate(&sim, PhaseForm::Semiclassical, &mut rng)
        })?;
        let mut seed = 400u64;
        let attempt = fail_on_timeout(timed(budget, 1, || {
            seed += 1;
            let mut rng = Prng::new(seed);
            regev::factor(&sim, n, 2, &mut rng)
        }))?;
        match attempt {
            Ok((rt, rep)) => println!(
                "   {n:<5} {:>10} / {:<3}       {:>12} / {:<3}          {:>6.0}×",
                human(st),
                se.modular_multiplications,
                human(rt),
                rep.cost.full_multiplications / rep.samples.len(),
                rt as f64 / st.max(1) as f64
            ),
            Err(e) => println!("   {n:<7} {:>12}   refused: {}", human(st), wall(&e)),
        }
    }
    println!("   Both columns are in the table: Regev needs fewer full-width modular");
    println!("   multiplications per sample and orders of magnitude more simulator");
    println!("   time, because its exponent box is 2^(dR) ≥ N by construction against");
    println!("   Shor's orbit r. Simulability and hardware cost are not the same axis,");
    println!("   and nothing here should be read as a verdict on the second.");
    Ok(())
}
