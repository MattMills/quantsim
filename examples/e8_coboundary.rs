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
    println!("  edge degrees of freedom. The co-boundary is where the information is.");
    Ok(())
}
