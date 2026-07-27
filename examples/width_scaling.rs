//! Width scaling measured in memory, in actuality: prints the reported
//! footprint of each representation as qubits are added, the per-algebra
//! cost, and (on Linux) the process RSS delta actually observed when a big
//! dense state is allocated.
//!
//! Run with `cargo run --release --example width_scaling`.

use quantsim::prelude::*;

fn fmt_bytes(b: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
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

/// Current process resident set size, if the platform exposes it.
fn rss_bytes() -> Option<usize> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
    let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

fn main() {
    let sim = Simulator::<C64>::new();

    println!("== Representation scaling with width (C64 amplitudes) ==");
    println!(
        "{:>6} {:>14} {:>16} {:>16} {:>16}",
        "qubits", "dense", "sparse (GHZ)", "sparse (H-layer)", "adaptive (GHZ)"
    );
    for n in (4..=24).step_by(2) {
        let dense = if n <= 24 {
            fmt_bytes(DenseState::<C64>::new(n).unwrap().memory_bytes())
        } else {
            "-".into()
        };
        let sparse_ghz = fmt_bytes(
            sim.run_on("sparse", &library::ghz(n))
                .unwrap()
                .memory_bytes(),
        );
        // Uniform superposition saturates sparse; cap the width to keep the
        // example fast and small.
        let sparse_uniform = if n <= 16 {
            let mut c = Circuit::new(n);
            for q in 0..n {
                c.h(q);
            }
            fmt_bytes(sim.run_on("sparse", &c).unwrap().memory_bytes())
        } else {
            "(skipped)".into()
        };
        let adaptive_ghz = fmt_bytes(
            sim.run_on("adaptive", &library::ghz(n))
                .unwrap()
                .memory_bytes(),
        );
        println!("{n:>6} {dense:>14} {sparse_ghz:>16} {sparse_uniform:>16} {adaptive_ghz:>16}");
    }

    println!();
    println!("== Sparse GHZ far past the dense limit ==");
    println!("{:>6} {:>14} {:>10}", "qubits", "sparse", "nonzeros");
    for n in [32usize, 40, 48, 56, 63] {
        let state = sim.run_on("sparse", &library::ghz(n)).unwrap();
        println!(
            "{n:>6} {:>14} {:>10}",
            fmt_bytes(state.memory_bytes()),
            state.nonzero_count()
        );
    }

    println!();
    println!("== Algebra dimension multiplies the footprint (dense, n = 16) ==");
    println!("{:>24} {:>6} {:>14}", "algebra", "dim", "memory");
    fn algebra_row<S: Scalar>() {
        let mem = DenseState::<S>::new(16).unwrap().memory_bytes();
        println!(
            "{:>24} {:>6} {:>14}",
            S::algebra_name(),
            S::DIM,
            fmt_bytes(mem)
        );
    }
    algebra_row::<f64>();
    algebra_row::<C64>();
    algebra_row::<SplitComplex>();
    algebra_row::<Quaternion>();
    algebra_row::<Octonion>();
    algebra_row::<Sedenion>();

    println!();
    println!("== Estimate vs process RSS, in actuality (dense C64) ==");
    match rss_bytes() {
        Some(_) => {
            println!("{:>6} {:>14} {:>14}", "qubits", "estimate", "RSS delta");
            for n in [20usize, 22, 24] {
                let before = rss_bytes().unwrap();
                let state = DenseState::<C64>::new(n).unwrap();
                // Touch the vector so lazily mapped pages become resident.
                let mut checksum = 0.0;
                state.for_each_nonzero(&mut |_, a| checksum += a.abs_sqr());
                let after = rss_bytes().unwrap();
                println!(
                    "{n:>6} {:>14} {:>14}   (norm² = {checksum})",
                    fmt_bytes(state.memory_bytes()),
                    fmt_bytes(after.saturating_sub(before)),
                );
                drop(state);
            }
            println!("(RSS deltas are noisy at small sizes; zero-page mapping means the");
            println!(" untouched-amplitude part may not be resident until written.)");
        }
        None => println!("(process RSS not available on this platform; estimates above)"),
    }
}
