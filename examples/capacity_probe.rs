//! Probe the REAL capacity walls, and verify the library's automatic
//! inhibition against them.
//!
//! Each attempt runs in an isolated child process (this binary re-execs
//! itself), walked up the width axis until it actually fails:
//!
//! * **guarded** attempts run with the library's resource guard in its
//!   default state (memory measured at allocation time) plus an armed
//!   time budget — over-scale work is refused with measured numbers or
//!   aborted mid-kernel by the deadline;
//! * at the first guarded memory refusal, the same width is re-run
//!   **raw** (admission disabled) so the *actual* wall is measured too:
//!   the child is expected to die — allocator refusal, or an OOM kill
//!   the parent detects by signal — with its last observed RSS
//!   recorded;
//! * every child runs under a parent watchdog that kills and records a
//!   hard timeout rather than waiting forever.
//!
//! Nothing here decides "too big" by arithmetic: limits are discovered
//! by running into them, and the guard's job is to convert those walls
//! into clean, measured errors.
//!
//! Run with `cargo run --release --example capacity_probe`.
//! Environment: `QS_PROBE_BUDGET_SECS` (guarded time budget for the
//! runtime axes, default 10), `QS_PROBE_WATCHDOG_SECS` (parent kill,
//! default 150 — generous enough that slow-but-fitting widths finish
//! and the walk genuinely reaches the memory wall).

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use quantsim::exact::ExactState;
use quantsim::prelude::*;

fn fmt_bytes(b: usize) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut value = b as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{b} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn proc_status_kb(pid: Option<u32>, key: &str) -> Option<usize> {
    let path = match pid {
        Some(pid) => format!("/proc/{pid}/status"),
        None => "/proc/self/status".to_string(),
    };
    let text = std::fs::read_to_string(path).ok()?;
    let line = text.lines().find(|l| l.starts_with(key))?;
    Some(line.split_whitespace().nth(1)?.parse::<usize>().ok()? * 1024)
}

// ───────────────────────── child side ─────────────────────────

/// One attempt, in-process. Prints a single machine-readable line:
/// `OK <elapsed_ms> <estimate_bytes> <peak_rss_bytes>` or
/// `ERR <library error>`; exits 0 either way (hard deaths are the
/// parent's to observe).
fn child(axis: &str, n: usize, raw: bool) {
    if raw {
        // Admission off: walk straight into the machine's real wall.
        guard::set_memory_limit(Some(usize::MAX));
        guard::set_time_budget(None);
    } else if matches!(axis, "mera-ghz" | "dense-qft") {
        // Runtime axes: the wall under test is the armed budget itself.
        // Memory axes run un-budgeted (the parent watchdog still bounds
        // them) so their wall is the measured memory refusal.
        let budget: u64 = std::env::var("QS_PROBE_BUDGET_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        guard::set_time_budget(Some(Duration::from_secs(budget)));
    }
    let started = Instant::now();
    let outcome: Result<usize> = attempt(axis, n);
    let elapsed = started.elapsed().as_millis();
    match outcome {
        Ok(estimate) => {
            let rss = proc_status_kb(None, "VmHWM:").unwrap_or(0);
            println!("OK {elapsed} {estimate} {rss}");
        }
        Err(e) => println!("ERR {e}"),
    }
}

fn attempt(axis: &str, n: usize) -> Result<usize> {
    let mut sim = Simulator::<C64>::new();
    sim.backends_mut().register("compound-binary", |n| {
        Ok(Box::new(CompoundBackend::new(n)?))
    })?;
    sim.backends_mut().register("e8-rep", |n| {
        Ok(Box::new(quantsim::e8::rep::E8RepState::new(n)?))
    })?;
    sim.backends_mut().register("e8-constellation", |n| {
        Ok(Box::new(
            quantsim::e8::constellation::E8ConstellationState::new(n)?,
        ))
    })?;
    sim.backends_mut().register("algebraic-h", |n| {
        let k = 1usize.min(n.saturating_sub(1));
        Ok(Box::new(
            quantsim::qudit::AlgebraicRegister::<Quaternion>::new(
                n - k,
                k,
                Box::new(SparseState::new(n - k)?),
            )?,
        ))
    })?;
    match axis {
        // Allocate 2^n dense amplitudes and touch them all with one gate.
        "dense" => {
            let mut c: Circuit = Circuit::new(n);
            c.h(0);
            let state = sim.run(&c)?;
            Ok(state.memory_bytes())
        }
        // Dense QFT: the runtime axis (n² gates over 2^n amplitudes).
        "dense-qft" => {
            let state = sim.run(&library::qft(n))?;
            Ok(state.memory_bytes())
        }
        // Fill the sparse map to 2^n entries.
        "sparse-fill" => {
            let mut c: Circuit = Circuit::new(n);
            for q in 0..n {
                c.h(q);
            }
            let state = sim.run_on("sparse", &c)?;
            Ok(state.memory_bytes())
        }
        // Exact D[ω] register (68-byte elements).
        "exact" => {
            let mut c: Circuit = Circuit::new(n);
            c.h(0);
            let state = ExactState::run(&c)?;
            Ok((1usize << state.num_qubits()) * std::mem::size_of::<DOmega>())
        }
        // Two n/2-qubit chains bridged into one n-qubit factor.
        "factored-bridge" => {
            let half = n / 2;
            let mut c: Circuit = Circuit::new(n);
            c.h(0);
            for q in 0..half - 1 {
                c.cx(q, q + 1);
            }
            c.h(half);
            for q in half..n - 1 {
                c.cx(q, q + 1);
            }
            c.cx(half - 1, half);
            let state = sim.run_on("factored", &c)?;
            Ok(state.memory_bytes())
        }
        // GHZ chain through the mera hierarchy: the SVD runtime axis.
        "mera-ghz" => {
            let state = sim.run_on("mera", &library::ghz(n))?;
            Ok(state.memory_bytes())
        }
        // Fill the hierarchical register's sparse site sector to 2^(n−1).
        "algebraic-fill" => {
            let mut c: Circuit = Circuit::new(n);
            for q in 0..n {
                c.h(q);
            }
            let state = sim.run_on("algebraic-h", &c)?;
            Ok(state.memory_bytes())
        }
        // H-layer then CX chain: every horizontal volume merges into one,
        // filling it to 2^n entries — the merge/growth walls.
        "compound-fill" => {
            let mut c: Circuit = Circuit::new(n);
            for q in 0..n {
                c.h(q);
            }
            for q in 0..n - 1 {
                c.cx(q, q + 1);
            }
            let state = sim.run_on("compound-binary", &c)?;
            Ok(state.memory_bytes())
        }
        // GHZ on the compound register: two entries at any width — the
        // wall is the packed-index ceiling, not memory.
        "compound-ghz" => {
            let state = sim.run_on("compound-binary", &library::ghz(n))?;
            Ok(state.memory_bytes())
        }
        // The single-copy E8 representation: the wall is structural
        // (the 256 sector points ARE the 8-qubit basis).
        "e8-rep" => {
            let mut c: Circuit = Circuit::new(n);
            c.h(0);
            let state = sim.run_on("e8-rep", &c)?;
            Ok(state.memory_bytes())
        }
        // Fill the constellation's lattice-point map to 2^n keys
        // (80-byte points: the memory wall arrives before sparse's).
        "constellation-fill" => {
            let mut c: Circuit = Circuit::new(n);
            for q in 0..n {
                c.h(q);
            }
            let state = sim.run_on("e8-constellation", &c)?;
            Ok(state.memory_bytes())
        }
        // GHZ on the constellation: two lattice points at any width —
        // the wall is the u64 basis-index ceiling.
        "constellation-ghz" => {
            let state = sim.run_on("e8-constellation", &library::ghz(n))?;
            Ok(state.memory_bytes())
        }
        other => Err(Error::InvalidState(format!("unknown axis '{other}'"))),
    }
}

// ───────────────────────── parent side ─────────────────────────

enum Verdict {
    Ok {
        elapsed_ms: u128,
        estimate: usize,
        rss: usize,
    },
    Refused(String),
    Killed {
        signal: Option<i32>,
        elapsed: Duration,
        last_rss: usize,
    },
    Watchdog {
        after: Duration,
        last_rss: usize,
    },
}

fn run_attempt(axis: &str, n: usize, raw: bool, watchdog: Duration) -> Verdict {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = Command::new(exe);
    cmd.arg("attempt")
        .arg(axis)
        .arg(n.to_string())
        .arg(if raw { "raw" } else { "guarded" })
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let started = Instant::now();
    let mut chd = cmd.spawn().expect("spawn probe child");
    let pid = chd.id();
    let mut last_rss = 0usize;
    let status = loop {
        if let Some(rss) = proc_status_kb(Some(pid), "VmHWM:") {
            last_rss = last_rss.max(rss);
        }
        match chd.try_wait().expect("try_wait") {
            Some(status) => break status,
            None => {
                if started.elapsed() > watchdog {
                    let _ = chd.kill();
                    let _ = chd.wait();
                    return Verdict::Watchdog {
                        after: started.elapsed(),
                        last_rss,
                    };
                }
                std::thread::sleep(Duration::from_millis(15));
            }
        }
    };
    let mut out = String::new();
    if let Some(mut stdout) = chd.stdout.take() {
        let _ = stdout.read_to_string(&mut out);
    }
    let line = out.lines().last().unwrap_or("").trim().to_string();
    if status.success() {
        if let Some(rest) = line.strip_prefix("OK ") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() == 3 {
                return Verdict::Ok {
                    elapsed_ms: parts[0].parse().unwrap_or(0),
                    estimate: parts[1].parse().unwrap_or(0),
                    rss: parts[2].parse().unwrap_or(0),
                };
            }
        }
        if let Some(err) = line.strip_prefix("ERR ") {
            return Verdict::Refused(err.to_string());
        }
    }
    // Hard death: pull the signal if the platform exposes it.
    #[cfg(unix)]
    let signal = std::os::unix::process::ExitStatusExt::signal(&status);
    #[cfg(not(unix))]
    let signal = None;
    Verdict::Killed {
        signal,
        elapsed: started.elapsed(),
        last_rss,
    }
}

fn describe(v: &Verdict) -> String {
    match v {
        Verdict::Ok {
            elapsed_ms,
            estimate,
            rss,
        } => format!(
            "ok        {:>7} ms   est {:>10}   peak RSS {:>10}",
            elapsed_ms,
            fmt_bytes(*estimate),
            fmt_bytes(*rss)
        ),
        Verdict::Refused(err) => format!("refused   {err}"),
        Verdict::Killed {
            signal,
            elapsed,
            last_rss,
        } => format!(
            "KILLED    signal {:?} after {:.1?}, last RSS {} — the real wall",
            signal,
            elapsed,
            fmt_bytes(*last_rss)
        ),
        Verdict::Watchdog { after, last_rss } => format!(
            "WATCHDOG  killed after {:.1?}, last RSS {}",
            after,
            fmt_bytes(*last_rss)
        ),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 5 && args[1] == "attempt" {
        let n: usize = args[3].parse().expect("width");
        child(&args[2], n, args[4] == "raw");
        return;
    }

    let watchdog = Duration::from_secs(
        std::env::var("QS_PROBE_WATCHDOG_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(150),
    );
    let mem_total = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| {
            let line = t.lines().find(|l| l.starts_with("MemTotal:"))?;
            Some(line.split_whitespace().nth(1)?.parse::<usize>().ok()? * 1024)
        })
        .unwrap_or(0);
    println!("machine context (measured, not assumed):");
    println!(
        "  MemTotal {}   MemAvailable {}   guard admission budget {}",
        fmt_bytes(mem_total),
        fmt_bytes(guard::measured_available()),
        fmt_bytes(guard::admission_budget()),
    );
    println!(
        "  guarded time budget {}s, parent watchdog {}s\n",
        std::env::var("QS_PROBE_BUDGET_SECS").unwrap_or_else(|_| "10".into()),
        watchdog.as_secs()
    );

    let axes: Vec<(&str, Vec<usize>, &str)> = vec![
        (
            "dense",
            vec![26, 28, 29, 30, 31, 32],
            "2^n complex amplitudes, all touched",
        ),
        (
            "exact",
            vec![22, 24, 26, 27, 28],
            "2^n × 68-byte D[ω] ring elements",
        ),
        (
            "sparse-fill",
            vec![22, 24, 25, 26, 27],
            "2^n hash-map entries via H-layers",
        ),
        (
            "factored-bridge",
            vec![26, 28, 30, 32, 36],
            "two n/2 chains bridged into one n-qubit factor",
        ),
        (
            "mera-ghz",
            vec![16, 18, 20, 22],
            "root-crossing SVD: the runtime wall",
        ),
        (
            "dense-qft",
            vec![18, 20, 22, 24],
            "n² gates over 2^n amplitudes: the runtime wall",
        ),
        (
            "algebraic-fill",
            vec![22, 24, 25, 26, 27],
            "hierarchical register: sparse site sector fills to 2^(n−1)",
        ),
        (
            "compound-fill",
            vec![22, 24, 25, 26, 27],
            "horizontal volumes merged into one 2^n-entry volume",
        ),
        (
            "constellation-fill",
            vec![20, 22, 24, 25, 26],
            "2^n lattice-point keys (80 B each): the E8 tower's memory wall",
        ),
        (
            "compound-ghz",
            vec![62, 63, 64],
            "two entries at any width: the packed-index ceiling",
        ),
        (
            "constellation-ghz",
            vec![62, 63, 64],
            "two lattice points at any width: the u64 index ceiling",
        ),
        (
            "e8-rep",
            vec![7, 8, 9],
            "single-copy E8×E8: the structural 256-point ceiling",
        ),
    ];

    for (axis, widths, blurb) in axes {
        println!("── axis: {axis} — {blurb}");
        for &n in &widths {
            let verdict = run_attempt(axis, n, false, watchdog);
            println!("  n={n:<3} {}", describe(&verdict));
            let stop = !matches!(verdict, Verdict::Ok { .. });
            // At the first guarded MEMORY refusal, measure the real wall
            // raw at the same width (runtime axes stay guarded: their
            // wall is the budget itself, already measured).
            if let Verdict::Refused(err) = &verdict {
                if err.contains("out of memory") {
                    let raw = run_attempt(axis, n, true, watchdog);
                    println!("  n={n:<3} raw: {}", describe(&raw));
                }
            }
            if stop {
                break;
            }
        }
        println!();
    }
    println!("every stop above is a measured event — a guard refusal with the");
    println!("numbers, a deadline abort, or an observed kill — never a");
    println!("precomputed skip.");
}
