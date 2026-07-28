//! The boundary atlas, live: every representation's measured bound,
//! the assumption behind it, the family that escapes it, and the
//! operational meaning of quantum advantage in this crate.
//!
//! Run with `cargo run --release --example advantage_bounds`.

use quantsim::prelude::*;
use std::time::Duration;

fn fmt_bytes(b: usize) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{b} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

fn show(scan: &FamilyScan) {
    println!("  {} over sizes {:?}:", scan.name, scan.sizes);
    for a in &scan.axes {
        let law = a
            .law
            .as_ref()
            .map(|l| l.to_string())
            .unwrap_or_else(|| "walled".into());
        let last = a
            .costs
            .last()
            .and_then(|c| *c)
            .map(fmt_bytes)
            .unwrap_or_else(|| "—".into());
        println!(
            "    {:<16} {:<12} (largest run {}, exact {})",
            a.axis, law, last, a.exact
        );
    }
    match &scan.verdict {
        Verdict::Classical { via } => {
            println!(
                "    verdict: CLASSICAL — the {} assumption held",
                via.join(" / ")
            )
        }
        Verdict::Candidate => {
            println!("    verdict: CANDIDATE — every measured assumption escaped at once")
        }
    }
    println!();
}

fn main() -> Result<()> {
    println!("── the assumptions (what each representation bets on)");
    println!("  dense            no bet: 2^n always — the universal reference");
    println!("  sparse           basis-state support stays small");
    println!("  factored         entangled clusters stay small");
    println!("  mps              entanglement across linear cuts stays small");
    println!("  mera             entanglement across tree cuts stays small");
    println!("  clifford-framed  non-Clifford content stays small");
    println!("  each is efficient exactly while its bet holds; the scan below measures");
    println!("  which bet holds per family, and classifies every axis's growth law.\n");

    // ── 1. Known fragments, rediscovered by measurement ──────────────
    println!("── circuit families × every representation (measured laws + verdicts)");
    show(&advantage_scan("ghz", library::ghz, &[8, 10, 12, 14]));
    show(&advantage_scan("qft |0…0⟩", library::qft, &[6, 8, 10, 12]));
    show(&advantage_scan(
        "rainbow",
        library::rainbow,
        &[8, 10, 12, 14],
    ));
    show(&advantage_scan(
        "clifford brickwork",
        |n| library::brickwork(n, n, &(0..n).collect::<Vec<_>>()),
        &[8, 10, 12],
    ));
    show(&advantage_scan(
        "random universal (3n² gates)",
        |n| library::random_circuit(n, 3 * n * n, 7),
        &[6, 8, 10, 12],
    ));
    println!("  the atlas rediscovers the known complexity results from bytes alone:");
    println!("  GHZ is classical via support, basis-input QFT via bond, rainbow via");
    println!("  clustering (while the fixed tree pays ~2^n at its root and sparse pays");
    println!("  exactly √2^n), Clifford circuits via the frame (Gottesman–Knill at");
    println!("  size^2), and random universal circuits escape EVERY assumption at");
    println!("  once — the candidate. Advantage lives only where all axes blow up.\n");

    // ── 2. The precision axis ────────────────────────────────────────
    println!("── the precision bound (Ball): certified radius vs depth");
    let mut prev: Option<f64> = None;
    for depth in [4usize, 8, 12, 16] {
        let mut c: Circuit<Ball> = Circuit::new(1);
        for _ in 0..depth {
            c.h(0);
        }
        let state = Simulator::<Ball>::new().run(&c)?;
        let rad = state.amplitude(0).rad;
        let per_h = prev.map(|p| (rad / p).powf(0.25));
        println!(
            "  depth {depth:>2}: certified radius {rad:.3e}{}",
            per_h
                .map(|r| format!("   (×{r:.3} per H — the √2 interval law)"))
                .unwrap_or_default()
        );
        prev = Some(rad);
    }
    println!("  certified precision is a resource too: it decays exponentially with");
    println!("  depth at measured base ≈ √2 unless the grid is refined — the exact");
    println!("  D[ω] evaluator is the escape (absolute values, no radius at all).\n");

    // ── 3. The measured walls ────────────────────────────────────────
    println!("── the walls (measured refusals, never precomputed skips)");
    println!(
        "  guard admission budget right now: {}",
        fmt_bytes(guard::admission_budget())
    );
    match DenseState::<C64>::new(34) {
        Err(e) => println!("  dense at 34 qubits: {e}"),
        Ok(_) => println!("  dense at 34 qubits: fit on this machine"),
    }
    println!(
        "  sparse at 64 qubits: {}",
        SparseState::<C64>::new(64)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    );
    let slow = guard::with_time_budget(Duration::from_secs(2), || {
        Simulator::<C64>::new().run_on("mera", &library::ghz(30))
    });
    match slow {
        Err(e) => println!("  mera ghz-30 under the guard (2 s budget armed): {e}"),
        Ok(_) => println!("  mera ghz-30 under the guard (2 s budget armed): finished"),
    }
    println!("  (whichever wall arrives first — the measured memory refusal or the");
    println!("  armed deadline — is the one reported; nothing is skipped on arithmetic.)");
    println!("  (the hierarchical register clears the indexing wall: 67 logical qubits,");
    println!("  demonstrated in examples/algebraic_qudits.rs.)\n");

    // ── 4. The physical tax on the candidate family ──────────────────
    println!("── the physical tax: the candidate family on real geometry");
    let n = 12;
    let candidate = library::random_circuit(n, 2 * n * n, 7);
    let reg = GateRegistry::<C64>::standard();
    let bound = candidate.bind(&reg)?;
    let mut hex = DeviceState::with_latency(
        Topology::heavy_hex_falcon_corner(n)?,
        LatencyMap::uniform(DurationModel::ibm_falcon_like()),
        ArityPolicy::default(),
        Box::new(SparseState::<C64>::new(n)?),
    )?;
    bound.run(&mut hex)?;
    let mut ion = DeviceState::with_latency(
        Topology::complete(n),
        LatencyMap::uniform(DurationModel::ibm_falcon_like()),
        ArityPolicy::default(),
        Box::new(SparseState::<C64>::new(n)?),
    )?;
    bound.run(&mut ion)?;
    println!(
        "  heavy-hex corner: {} swaps, elapsed {:.1} µs; all-to-all: {} swaps, {:.1} µs",
        hex.swap_count(),
        hex.elapsed() as f64 / 1e3,
        ion.swap_count(),
        ion.elapsed() as f64 / 1e3,
    );
    println!(
        "  geometry tax ×{:.2} — an advantage claim must survive its coupling map.\n",
        hex.elapsed() as f64 / ion.elapsed() as f64
    );

    // ── 5. What would DISCOVER a sub-exponential advantage ───────────
    println!("── the discovery instrument");
    println!("  register any new representation as a backend and re-run the scan on the");
    println!("  candidate family: an axis that stays sub-exponential — while exact —");
    println!("  where every axis above grows exponentially IS a discovered");
    println!("  sub-exponential simulation, and the conformance suite will hold it to");
    println!("  the same physics. Until then, the measured content of quantum");
    println!("  advantage here is precisely: every assumption in the table fails at");
    println!("  once, and nothing cheaper than the 2^n reference survives.");
    Ok(())
}
