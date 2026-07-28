//! The hierarchical qubit register, live: break the flat n-wide
//! register into a varied qudit structure — a site sector under any
//! backend plus algebra qubits carried inside each Cayley–Dickson
//! scalar — with gates on the algebra sector synthesized from the
//! dual-algebra (left × right multiplication), and a logical width no
//! flat u64-indexed register can reach.
//!
//! Run with `cargo run --release --example algebraic_qudits`.

use quantsim::prelude::*;

fn fmt_bytes(b: usize) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
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

fn main() -> Result<()> {
    let reg = GateRegistry::<C64>::standard();

    // ── 1. The dual-algebra operator space, measured per doubling ────
    println!("── the dual-algebra: left × right multiplication, measured per doubling level");
    println!(
        "  {:<26} {:>9} {:>12} {:>13} {:>16}",
        "scalar (qudit width)", "rank", "of End_R", "residual", "embedded ℂ-linear"
    );
    let h = dual_algebra_report::<Quaternion>()?;
    let o = dual_algebra_report::<Octonion>()?;
    let s = dual_algebra_report::<Sedenion>()?;
    for (name, r) in [
        ("H  (1 algebra qubit)", &h),
        ("O  (2 algebra qubits)", &o),
        ("S  (3 algebra qubits)", &s),
    ] {
        println!(
            "  {:<26} {:>9} {:>12} {:>13.2e} {:>16}",
            name, r.sandwich_rank, r.operator_space, r.max_linear_residual, r.embedded_linear
        );
    }
    println!("  full rank at every level: EVERY C-linear qudit gate is exactly a sum of");
    println!("  (a·x)·b terms — the algebra acting on itself from both sides resolves the");
    println!("  whole endomorphism ring, zero divisors (S) and non-associativity (O, S)");
    println!("  notwithstanding. The embedded-C twist past H is measured, and routes site");
    println!("  gates through the component path there.");

    // A concrete synthesis: Hadamard on the quaternion qudit.
    let f = std::f64::consts::FRAC_1_SQRT_2;
    let (op, residual) =
        synthesize_sandwich::<Quaternion>(&[c64(f, 0.0), c64(f, 0.0), c64(f, 0.0), c64(-f, 0.0)])?;
    println!(
        "\n  H on the H-qudit = {} sandwich terms, residual {residual:.1e}:",
        op.terms.len()
    );
    for t in &op.terms {
        println!(
            "    x ↦ ({:+.4}·[{}]) · x · [{}]",
            t.left
                .coeffs()
                .iter()
                .cloned()
                .fold(0.0f64, |a, b| if b.abs() > a.abs() { b } else { a }),
            t.left
                .coeffs()
                .iter()
                .position(|c| c.abs() > 1e-9)
                .map(|i| ["1", "i", "j", "k"][i])
                .unwrap_or("0"),
            t.right
                .coeffs()
                .iter()
                .position(|c| c.abs() > 1e-9)
                .map(|i| ["1", "i", "j", "k"][i])
                .unwrap_or("0"),
        );
    }

    // ── 2. One logical register, many shapes ─────────────────────────
    println!("\n── ghz-20 at every split of the same logical width (site sector sparse)");
    println!(
        "  {:<34} {:>12} {:>14} {:>12}",
        "shape", "site support", "stored memory", "max |Δamp|"
    );
    let n = 20;
    let ghz = library::ghz(n);
    let bound = ghz.bind(&reg)?;
    let flat = Simulator::<C64>::new().run_on("sparse", &ghz)?;
    println!(
        "  {:<34} {:>12} {:>14} {:>12}",
        "flat: 20 qubit sites",
        flat.nonzero_count(),
        fmt_bytes(flat.memory_bytes()),
        "reference"
    );
    let mut h20 =
        AlgebraicRegister::<Quaternion>::new(19, 1, Box::new(SparseState::<Quaternion>::new(19)?))?;
    bound.run(&mut h20)?;
    let mut o20 =
        AlgebraicRegister::<Octonion>::new(18, 2, Box::new(SparseState::<Octonion>::new(18)?))?;
    bound.run(&mut o20)?;
    let mut s20 =
        AlgebraicRegister::<Sedenion>::new(17, 3, Box::new(SparseState::<Sedenion>::new(17)?))?;
    bound.run(&mut s20)?;
    for (label, dev, support, mem) in [
        (
            "19 sites × H-qudit  (19+1)",
            max_amplitude_deviation(flat.as_ref(), &h20),
            h20.site_support(),
            h20.memory_bytes(),
        ),
        (
            "18 sites × O-qudit  (18+2)",
            max_amplitude_deviation(flat.as_ref(), &o20),
            o20.site_support(),
            o20.memory_bytes(),
        ),
        (
            "17 sites × S-qudit  (17+3)",
            max_amplitude_deviation(flat.as_ref(), &s20),
            s20.site_support(),
            s20.memory_bytes(),
        ),
    ] {
        println!(
            "  {:<34} {:>12} {:>14} {:>12.1e}",
            label,
            support,
            fmt_bytes(mem),
            dev
        );
    }
    println!("  the same physics at every split — the register's SHAPE is a free knob,");
    println!("  and the site sector keeps its sparse structure (support counts sites).");

    // Native vs component routing, measured.
    let stats_h = h20.stats();
    let stats_o = o20.stats();
    println!(
        "  routing (measured embedded-linearity): H split ran {} gates native / {} component;",
        stats_h.native_site_gates, stats_h.component_gates
    );
    println!(
        "  O split ran {} native / {} component (the twist forbids the native path).",
        stats_o.native_site_gates, stats_o.component_gates
    );

    // ── 3. Beyond the flat indexing ceiling ──────────────────────────
    println!("\n── 66 logical qubits: past the u64 ceiling of every flat register");
    println!(
        "  flat sparse at 66 qubits: {}",
        SparseState::<C64>::new(66)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    );
    let mut wide =
        AlgebraicRegister::<Sedenion>::new(63, 3, Box::new(SparseState::<Sedenion>::new(63)?))?;
    library::ghz(66).bind(&reg)?.run(&mut wide)?;
    let all_sites = (1u64 << 63) - 1;
    println!(
        "  hierarchical (63 sites × S-qudit): ghz-66 runs; ⟨0…0|ψ⟩ = {:.6}, ⟨1…1|ψ⟩ = {:.6}",
        wide.amplitude_parts(0, 0).re,
        wide.amplitude_parts(all_sites, 0b111).re
    );
    println!(
        "  site support {}, stored memory {} — 66 exact logical qubits in a register",
        wide.site_support(),
        fmt_bytes(wide.memory_bytes())
    );
    println!("  whose flat basis index would not fit a machine word.");

    // ── 4. Conformance, priced by the harness ────────────────────────
    println!("\n── the hierarchical registers through the workload harness (verified in-run)");
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut().register("algebraic-h", |n| {
        let k = 1.min(n.saturating_sub(1));
        Ok(Box::new(AlgebraicRegister::<Quaternion>::new(
            n - k,
            k,
            Box::new(SparseState::<Quaternion>::new(n - k)?),
        )?))
    })?;
    sim.backends_mut().register("algebraic-o", |n| {
        let k = 2.min(n.saturating_sub(1));
        Ok(Box::new(AlgebraicRegister::<Octonion>::new(
            n - k,
            k,
            Box::new(SparseState::<Octonion>::new(n - k)?),
        )?))
    })?;
    let workloads = [
        Workload::from_circuit("ghz-10", library::ghz(10)),
        Workload::from_circuit("ranged-5", library::ranged_pairs(10, 5)),
        Workload::from_circuit("qft-10", library::qft(10)),
    ];
    let report = compare_backends(
        &sim,
        &workloads,
        &["dense", "sparse", "algebraic-h", "algebraic-o"],
        &BenchConfig::default(),
    )?;
    println!("{report}");
    Ok(())
}
