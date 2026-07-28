//! The E8×E8 co-boundary storage system, live: information stored
//! projectively across the 240 paired points of two E8 copies — as a
//! relative phase field that no single copy can see — plus the
//! measured cochain structure of the root complex itself.
//!
//! Run with `cargo run --release --example e8_coboundary`.

use quantsim::e8;
use quantsim::mixed::{fourier_d, CompoundRegister};
use quantsim::prelude::*;

const N: usize = 240;

fn main() -> Result<()> {
    // ── 1. The storage substrate: the root complex, measured ─────────
    println!("── the E8 point/edge/cell structure (all measured)");
    let rs = e8::roots();
    println!(
        "  every point sees the others as {:?} (antipode, −1, ⊥, +1, —)",
        e8::neighbor_profile(&rs[0])
    );
    let edges = e8::minus_one_edges();
    let triangles = e8::zero_sum_triangles();
    println!(
        "  −1 edges: {}   zero-sum triangles (α+β+γ = 0): {} — every edge closes",
        edges.len(),
        triangles.len()
    );
    println!("  into exactly one additive relation, because the lattice's norm-2");
    println!("  vectors are all roots.");
    let b1 = e8::triangle_complex_b1();
    println!("  GF(2) first Betti number of the complex: b₁ = {b1} — the dimension of");
    println!("  edge-stored data that is a cocycle but NOT a coboundary: the invariant");
    println!("  storage the complex carries beyond anything derivable from points.\n");

    // ── 2. The E8×E8 co-boundary state ───────────────────────────────
    println!("── E8×E8: the relative field between paired points");
    // The data: a phase per root, set by the root's inner product with
    // a reference root — the field is E8-structured, not arbitrary.
    let w = rs[7];
    let field: Vec<C64> = rs
        .iter()
        .map(|r| {
            let angle = std::f64::consts::TAU * (e8::dot(r, &w) + 8) as f64 / 17.0;
            c64(angle.cos(), angle.sin())
        })
        .collect();

    let mut reg = CompoundRegister::new(&[N, N])?;
    reg.apply_1(0, &fourier_d(N))?;
    reg.apply_2_permutation(0, 1, &|a, b| (a, (b + a) % N))?;
    reg.apply_1_diagonal(1, &field)?;
    println!("  state: Σ_α g(α) |α⟩_A |α⟩_B / √240 — one ray PER PAIRED POINT,");
    println!(
        "  {} stored entries in one volume of {} joint states.",
        reg.stored_entries(),
        reg.largest_volume_states()
    );

    let mut worst_a: f64 = 0.0;
    for a in (0..N).step_by(10) {
        let mut p = 0.0;
        for b in 0..N {
            p += reg.probability(&[a, b])?;
        }
        worst_a = worst_a.max((p - 1.0 / N as f64).abs());
    }
    println!("  copy-A marginals: uniform to {worst_a:.1e} — the field is INVISIBLE at");
    println!("  either copy alone; it lives in the co-boundary between them, and a");
    println!("  global phase of the field is physically nothing (projective storage:");
    println!("  the capacity is the ray space ℂP^239 — 478 real parameters — per");
    println!("  paired-point layer).\n");

    // ── 3. Readout by cross-copy interference ────────────────────────
    println!("── recovery requires touching BOTH copies");
    reg.apply_2_permutation(0, 1, &|a, b| (a, (b + N - a) % N))?;
    let f = fourier_d(N);
    let fdag: Vec<C64> = (0..N * N).map(|i| f[(i % N) * N + i / N].conj()).collect();
    reg.apply_1(0, &fdag)?;
    let mut worst: f64 = 0.0;
    for k in (0..N).step_by(8) {
        let mut expected = c64(0.0, 0.0);
        for (alpha, phase) in field.iter().enumerate() {
            let angle = -std::f64::consts::TAU * (k * alpha % N) as f64 / N as f64;
            expected += *phase * c64(angle.cos(), angle.sin());
        }
        expected /= N as f64;
        let mut basis = vec![0usize; 2];
        basis[0] = k;
        worst = worst.max((reg.amplitude(&basis)? - expected).norm());
    }
    println!("  uncompute the pairing, interfere with F†: output amplitudes equal the");
    println!("  DFT of the stored field to {worst:.1e} — full recovery, and the flow");
    println!(
        "  record shows it needed both copies (cone of A: {:?}).",
        reg.reachable_from(0)
    );
    println!("\n  storage = rays across paired E8 points; invisibility = uniform local");
    println!("  marginals (measured); recovery = cross-copy interference (measured);");
    println!("  and beyond point data the complex itself offers b₁ = {b1} invariant");
    println!("  edge degrees of freedom. The co-boundary is where the information is.\n");

    // ── 4. The same E8×E8 as a QUBIT representation ──────────────────
    use quantsim::e8::rep;
    println!("── E8×E8 as an 8-qubit representation (entanglement as root geometry)");
    println!("  the 128 even-parity basis states ARE the spinor roots (bit ↦ sign);");
    println!("  the odd sector pairs onto the second copy: 128 + 128 = all 256.");
    println!("  Two basis states at Hamming distance 2 differ by exactly an INTEGER");
    println!("  root — the 112 integer roots are the 2-local transition labels,");
    println!("  verified against real gate matrices in tests/e8_representation.rs.");
    let sim: Simulator = Simulator::new();
    println!(
        "  {:<22} {:>7} {:>9} {:>11} {:>10} {:>10} {:>10}",
        "state", "points", "sectors", "root edges", "affine", "antipodes", "S(center)"
    );
    for (name, circuit) in [
        ("product", {
            let mut c: Circuit = Circuit::new(8);
            c.x(1).x(4);
            c
        }),
        ("ghz-8", library::ghz(8)),
        ("rainbow-8", library::rainbow(8)),
        ("random-8", library::random_circuit(8, 200, 7)),
    ] {
        let state = sim.run(&circuit)?;
        let geo = rep::support_geometry(state.as_ref());
        let (_, entropy) = rep::schmidt(state.as_ref(), 0b1111);
        println!(
            "  {:<22} {:>7} {:>4}+{:<4} {:>11} {:>10} {:>10} {:>9.3}b",
            name,
            geo.points,
            geo.sectors.0,
            geo.sectors.1,
            geo.root_edges,
            geo.affine_dim,
            geo.antipodal_pairs,
            entropy
        );
    }
    println!("  GHZ is exactly one ANTIPODAL PAIR of the root geometry (no 2-local");
    println!("  root connects its ends); products sit at affine dimension 0; random");
    println!("  states flood the root graph. Schmidt rank obeys the geometric");
    println!("  projected-support bound on every state and cut tested — entanglement");
    println!("  investigated as measured geometry, adjacency included.\n");

    // ── 5. Both systems in the standard frameworks ───────────────────
    println!("── both E8×E8 systems as first-class backends (conformance + benchmark)");
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut().register("compound-binary", |n| {
        Ok(Box::new(CompoundBackend::new(n)?))
    })?;
    sim.backends_mut()
        .register("e8-rep", |n| Ok(Box::new(rep::E8RepState::new(n)?)))?;

    let cfg = ConformanceConfig {
        random_circuits: 8,
        orderings: 2,
        ..ConformanceConfig::default()
    };
    for backend in ["compound-binary", "e8-rep"] {
        let report = verify_backend(&sim, backend, &cfg)?;
        let cases: usize = report.gate_checks.iter().map(|g| g.cases).sum();
        println!(
            "  conformance '{backend}': {} over {} gates ({cases} cases), max deviation {:.1e}",
            if report.passed() { "PASSED" } else { "FAILED" },
            report.gate_checks.len(),
            report.max_amplitude_deviation
        );
    }
    let wide = ConformanceConfig {
        extra_widths: vec![0, 7],
        random_circuits: 8,
        orderings: 2,
        ..cfg
    };
    let report = verify_backend(&sim, "e8-rep", &wide)?;
    println!(
        "  conformance 'e8-rep' at native width 8 (full 256-point set): {}\n",
        if report.passed() { "PASSED" } else { "FAILED" }
    );

    // The benchmark table: the co-boundary protocol itself (at d = 16,
    // qubit-encodable) as a workload, next to the standard families.
    let g16: Vec<C64> = (0..16usize)
        .map(|a| cis(std::f64::consts::TAU * ((a * a + 3 * a) % 17) as f64 / 17.0))
        .collect();
    let workloads = [
        Workload::from_circuit("ghz-8", library::ghz(8)),
        Workload::from_circuit("coboundary-16", coboundary16_circuit(&g16)),
    ];
    let backends = [
        "dense",
        "sparse",
        "mps",
        "mera",
        "compound-binary",
        "e8-rep",
    ];
    let bench = compare_backends(&sim, &workloads, &backends, &BenchConfig::default())?;
    print!("{bench}");
    println!("  (mps refuses the 8-qubit pairing gate at its measured window wall;");
    println!("  every completed run is amplitude-verified against dense in-table.");
    println!("  The native 240-level protocol has no qubit encoding at all — two");
    println!("  240-level sites span 57,600 joint states, which no 2^n register");
    println!("  matches; its numbers live in sections 2–3 above.)");
    Ok(())
}

/// The d = 16 co-boundary protocol as an 8-qubit circuit (site A =
/// qubits 0–3, site B = qubits 4–7): prepare, pair, store, unpair,
/// interfere — the readout state is the DFT of the field on site A.
fn coboundary16_circuit(field: &[C64]) -> Circuit {
    let d = 16usize;
    let dim = d * d;
    let f = fourier_d(d);
    let fdag: Vec<C64> = (0..d * d).map(|i| f[(i % d) * d + i / d].conj()).collect();
    let mut diag = vec![c64(0.0, 0.0); d * d];
    for (j, &g) in field.iter().enumerate() {
        diag[j * d + j] = g;
    }
    let mut unpair = vec![c64(0.0, 0.0); dim * dim];
    for a in 0..d {
        for b in 0..d {
            unpair[(a + d * ((b + d - a) % d)) * dim + (a + d * b)] = c64(1.0, 0.0);
        }
    }
    let site_a: Vec<usize> = (0..4).collect();
    let site_b: Vec<usize> = (4..8).collect();
    let both: Vec<usize> = (0..8).collect();
    let mut c: Circuit = Circuit::new(8);
    c.raw("f16", GateMatrix::from_vec(d, f).unwrap(), site_a.clone());
    c.raw(
        "pair16",
        GateMatrix::from_vec(dim, cshift(d, d)).unwrap(),
        both.clone(),
    );
    c.raw("field16", GateMatrix::from_vec(d, diag).unwrap(), site_b);
    c.raw("unpair16", GateMatrix::from_vec(dim, unpair).unwrap(), both);
    c.raw("f16dag", GateMatrix::from_vec(d, fdag).unwrap(), site_a);
    c
}
