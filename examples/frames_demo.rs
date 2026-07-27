//! Local frames, evidenced: the same transverse-field circuit run raw and
//! framed, with the representation costs printed side by side — plus the
//! adopt-frame primitive rewriting a representation without touching
//! physics.
//!
//! Run with `cargo run --release --example frames_demo`.

use quantsim::prelude::*;

fn fmt_bytes(b: usize) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = b as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}

fn main() -> Result<()> {
    let n = 18;
    let layers = 4;
    let reg = GateRegistry::<C64>::standard();

    // h-layer; L × { rx everywhere, rxx on the chain }; h-layer.
    let mut circuit: Circuit = Circuit::new(n);
    for q in 0..n {
        circuit.h(q);
    }
    for l in 0..layers {
        for q in 0..n {
            circuit.rx(q, 0.3 + 0.07 * (l * n + q) as f64);
        }
        for q in 0..n - 1 {
            circuit.rxx(q, q + 1, 0.5 + 0.05 * l as f64);
        }
    }
    for q in 0..n {
        circuit.h(q);
    }
    let bound = circuit.bind(&reg)?;
    println!(
        "transverse-field circuit: {n} qubits, {layers} layers, {} gates\n",
        bound.gates().len()
    );

    // Raw sparse: track the peak it pays in the computational basis.
    let mut raw = SparseState::<C64>::new(n)?;
    let mut raw_peak = raw.memory_bytes();
    let start = std::time::Instant::now();
    for gate in bound.gates() {
        match &gate.kernel {
            GateKernel::Matrix(m) => raw.apply(m, &gate.qubits)?,
            GateKernel::Diagonal(d) => raw.apply_diagonal(d, &gate.qubits)?,
        }
        raw_peak = raw_peak.max(raw.memory_bytes());
    }
    let raw_time = start.elapsed();

    // Framed sparse: frames absorb h/rx and diagonalize rxx.
    let mut framed = FramedState::new(Box::new(SparseState::<C64>::new(n)?));
    let start = std::time::Instant::now();
    bound.run(&mut framed)?;
    let framed_time = start.elapsed();
    let stats = framed.stats();
    let stored_support = framed.stored_nonzero_count();
    let framed_peak = framed.peak_inner_memory();

    println!("               {:>14} {:>14}", "raw sparse", "framed sparse");
    println!(
        "peak memory    {:>14} {:>14}",
        fmt_bytes(raw_peak),
        fmt_bytes(framed_peak)
    );
    println!(
        "wall time      {:>12.2?} {:>12.2?}",
        raw_time, framed_time
    );
    println!("stored support {:>14} {:>14}", "-", stored_support);
    println!(
        "\nframed execution: {} absorbed 1q, {} diagonalized 2q, {} dense-conjugated, {} flushes",
        stats.absorbed_1q, stats.diagonalized, stats.conjugated, stats.flushes
    );
    println!(
        "peak advantage: {:.0}× smaller\n",
        raw_peak as f64 / framed_peak as f64
    );

    // Same physics: check a few amplitudes against dense.
    let mut dense = DenseState::<C64>::new(n)?;
    bound.run(&mut dense)?;
    let deviation = max_amplitude_deviation(&dense, &framed);
    println!("deviation vs dense reference after flush: {deviation:.2e}\n");

    // adopt_frame: rewrite the representation of an existing state.
    let h = reg.resolve("h")?.matrix(&[])?;
    let mut state = FramedState::new(Box::new(SparseState::<C64>::new(12)?));
    for q in 0..12 {
        state.apply(&h, &[q])?;
    }
    state.flush()?; // |+…+⟩ materialized: support 4096
    println!("adopt_frame demo (12 qubits, |+…+⟩):");
    println!("  stored support before adoption: {}", state.stored_nonzero_count());
    for q in 0..12 {
        state.adopt_frame(q, &h)?;
    }
    println!("  stored support after adopting H frames: {}", state.stored_nonzero_count());
    println!(
        "  physical amplitude |0…0⟩ still {:.6} (= 1/√4096 = {:.6})",
        state.amplitude(0).re,
        1.0 / 64.0
    );
    Ok(())
}
