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

/// Run an attempt and render its outcome — the measured footprint, or
/// the guard's measured refusal / deadline abort. No presumed skips.
fn attempt(run: impl FnOnce() -> Result<usize>) -> String {
    match run() {
        Ok(bytes) => fmt_bytes(bytes),
        Err(Error::OutOfMemory {
            requested,
            available,
            ..
        }) => format!(
            "refused ({} > {})",
            fmt_bytes(requested),
            fmt_bytes(available)
        ),
        Err(Error::Timeout { elapsed_ms, .. }) => format!("aborted @{elapsed_ms}ms"),
        Err(other) => format!("error: {other}"),
    }
}

fn main() {
    let sim = Simulator::<C64>::new();
    // Every cell below is a real run: over-scale attempts are inhibited
    // by the library's resource guard (measured memory admission) or by
    // this armed per-attempt time budget — never pre-skipped on
    // arithmetic.
    guard::set_time_budget(Some(std::time::Duration::from_secs(10)));

    println!("== Representation scaling with width (C64 amplitudes) ==");
    println!(
        "{:>6} {:>14} {:>16} {:>20} {:>16}",
        "qubits", "dense", "sparse (GHZ)", "sparse (H-layer)", "adaptive (GHZ)"
    );
    for n in (4..=24).step_by(2) {
        let dense = attempt(|| Ok(DenseState::<C64>::new(n)?.memory_bytes()));
        let sparse_ghz = fmt_bytes(
            sim.run_on("sparse", &library::ghz(n))
                .unwrap()
                .memory_bytes(),
        );
        // Uniform superposition saturates sparse: attempted at every
        // width, bounded by the guard, not by a width switch.
        let sparse_uniform = attempt(|| {
            let mut c = Circuit::new(n);
            for q in 0..n {
                c.h(q);
            }
            Ok(sim.run_on("sparse", &c)?.memory_bytes())
        });
        let adaptive_ghz = fmt_bytes(
            sim.run_on("adaptive", &library::ghz(n))
                .unwrap()
                .memory_bytes(),
        );
        println!("{n:>6} {dense:>14} {sparse_ghz:>16} {sparse_uniform:>20} {adaptive_ghz:>16}");
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
    algebra_row::<Ball>();
    algebra_row::<SplitComplex>();
    algebra_row::<Quaternion>();
    algebra_row::<Octonion>();
    algebra_row::<Sedenion>();
    println!("  (Ball = C64 midpoint + certified radius: 1.5× the complex footprint)");

    println!();
    println!("== Factored: cost tracks entanglement clusters, not register width ==");
    println!();
    println!("(a) width sweep, entanglement held local (pairs): linear in width");
    println!(
        "{:>6} {:>16} {:>18} {:>14} {:>14} {:>14}",
        "qubits", "factored", "largest cluster", "sparse", "mps", "mera"
    );
    for n in [8usize, 16, 24, 32, 40, 48, 56] {
        let mut pairs = Circuit::new(n);
        for pair in 0..n / 2 {
            pairs
                .ry(2 * pair, 0.3 + 0.05 * pair as f64)
                .cx(2 * pair, 2 * pair + 1);
        }
        let state = sim.run_on("factored", &pairs).unwrap();
        let factored = state.as_any().downcast_ref::<FactoredState<C64>>().unwrap();
        // Sparse support is 2^(n/2) here — exponential in width; attempt
        // every row and let the guard's admission or the armed budget
        // stop it, measurably.
        let sparse = attempt(|| Ok(sim.run_on("sparse", &pairs)?.memory_bytes()));
        let mps = sim.run_on("mps", &pairs).unwrap();
        let mera = sim.run_on("mera", &pairs).unwrap();
        println!(
            "{n:>6} {:>16} {:>12} qubits {:>14} {:>14} {:>14}",
            fmt_bytes(state.memory_bytes()),
            factored.largest_factor_qubits(),
            sparse,
            fmt_bytes(mps.memory_bytes()),
            fmt_bytes(mera.memory_bytes())
        );
    }
    println!();
    println!("(b) cluster sweep, register width held at 24: exponential in cluster");
    println!(
        "{:>16} {:>16} {:>18} {:>14} {:>10}",
        "cluster qubits", "factored", "factor count", "mps", "max bond"
    );
    for cluster in [2usize, 6, 10, 14, 18, 22] {
        let mut c = Circuit::new(24);
        c.h(0);
        for q in 0..cluster - 1 {
            c.cx(q, q + 1); // one GHZ cluster of `cluster` qubits
        }
        for q in cluster..24 {
            c.ry(q, 0.4); // the rest stay separated
        }
        let state = sim.run_on("factored", &c).unwrap();
        let factored = state.as_any().downcast_ref::<FactoredState<C64>>().unwrap();
        let mps_state = sim.run_on("mps", &c).unwrap();
        let mps = mps_state.as_any().downcast_ref::<MpsState<C64>>().unwrap();
        println!(
            "{cluster:>16} {:>16} {:>18} {:>14} {:>10}",
            fmt_bytes(state.memory_bytes()),
            factored.factor_count(),
            fmt_bytes(mps_state.memory_bytes()),
            mps.max_bond_dimension()
        );
    }
    println!();
    println!("(c) the degenerate quadrants: GHZ(20) — global cluster, tiny support");
    let ghz = library::ghz(20);
    let f = sim.run_on("factored", &ghz).unwrap();
    let s = sim.run_on("sparse", &ghz).unwrap();
    let d = DenseState::<C64>::new(20).unwrap();
    let m = sim.run_on("mps", &ghz).unwrap();
    println!(
        "  dense {} | factored {} (one 20-qubit cluster) | sparse {} (2 nonzeros) | mps {} (bond 2)",
        fmt_bytes(d.memory_bytes()),
        fmt_bytes(f.memory_bytes()),
        fmt_bytes(s.memory_bytes()),
        fmt_bytes(m.memory_bytes())
    );
    let mera16 = sim.run_on("mera", &library::ghz(16)).unwrap();
    let mera_state = mera16.as_any().downcast_ref::<MeraState<C64>>().unwrap();
    println!(
        "  mera holds GHZ(16) at {} resting (hierarchy bond {}; the chain's one\n  root-crossing gate transiently paid a {}-element block — the rung-1 cost)",
        fmt_bytes(mera16.memory_bytes()),
        mera_state.max_bond_dimension(),
        mera_state.peak_block_elements()
    );
    println!("  — four orthogonal compression axes: clusters, support, Schmidt rank, hierarchy.");

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
