//! Clifford space, evidenced: the same Clifford+T circuits run raw and
//! through the Clifford frame, showing that the frame's stored cost is a
//! function of the T-count — not the width, not the gate count.
//!
//! Run with `cargo run --release --example clifford_space`.

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

/// H-layer, then a seeded Clifford stream with `t` T gates spliced in.
fn clifford_t_circuit(n: usize, cliffords: usize, t: usize, seed: u64) -> Circuit {
    let mut circuit: Circuit = Circuit::new(n);
    for q in 0..n {
        circuit.h(q);
    }
    let mut rng = Prng::new(seed);
    let stride = cliffords / (t + 1);
    let mut placed = 0;
    for g in 0..cliffords {
        let q = (rng.next_u64() % n as u64) as usize;
        let r = ((q + 1) + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
        match rng.next_u64() % 6 {
            0 => circuit.h(q),
            1 => circuit.s(q),
            2 => circuit.x(q),
            3 => circuit.z(q),
            4 => circuit.cx(q, r),
            _ => circuit.cz(q, r),
        };
        if placed < t && (g + 1) % stride == 0 {
            circuit.t((rng.next_u64() % n as u64) as usize);
            placed += 1;
        }
    }
    circuit
}

fn main() -> Result<()> {
    let reg = GateRegistry::<C64>::standard();

    // ── 1. Gottesman–Knill through the frame: pure Clifford, 50 qubits ──
    let n = 50;
    let circuit = clifford_t_circuit(n, 400, 0, 7);
    let bound = circuit.bind(&reg)?;
    let mut state = CliffordFramedState::<C64>::new(n)?;
    let start = std::time::Instant::now();
    bound.run(&mut state)?;
    let elapsed = start.elapsed();
    println!(
        "pure Clifford stream, {n} qubits, {} gates:",
        bound.gates().len()
    );
    println!(
        "  stored support {} | frame holds {} gates | memory {} | {:.2?}",
        state.stored_nonzero_count(),
        state.frame_gates(),
        fmt_bytes(state.memory_bytes()),
        elapsed
    );
    println!("  (dense would need 2^50 amplitudes = 16 PiB)\n");

    // ── 2. The exchange rate: cost = f(T-count), width and gates fixed ──
    let n = 20;
    let cliffords = 300;
    println!("Clifford+T scaffold, {n} qubits, {cliffords} Clifford gates, t T gates:");
    println!(
        "  {:>3} {:>18} {:>10} {:>16} {:>14} {:>10}",
        "t", "peak stored supp", "2^t bound", "max axis weight", "stored memory", "time"
    );
    for t in [0usize, 2, 4, 6, 8, 10] {
        let bound = clifford_t_circuit(n, cliffords, t, 40 + t as u64).bind(&reg)?;
        let mut state = CliffordFramedState::<C64>::new(n)?;
        let start = std::time::Instant::now();
        bound.run(&mut state)?;
        let elapsed = start.elapsed();
        let stats = state.stats();
        println!(
            "  {:>3} {:>18} {:>10} {:>16} {:>14} {:>9.2?}",
            t,
            state.peak_stored_support(),
            1usize << t,
            stats.max_axis_weight,
            fmt_bytes(state.peak_inner_memory()),
            elapsed
        );
        assert_eq!(stats.flushes, 0);
        assert_eq!(stats.axis_rotations, t);
    }

    // Raw sparse on the same scaffold (t is irrelevant to it): pays 2^n.
    let bound = clifford_t_circuit(n, cliffords, 10, 50).bind(&reg)?;
    let mut raw = SparseState::<C64>::new(n)?;
    let mut raw_peak = raw.nonzero_count();
    let mut raw_mem = raw.memory_bytes();
    let start = std::time::Instant::now();
    for gate in bound.gates() {
        match &gate.kernel {
            GateKernel::Matrix(m) => raw.apply(m, &gate.qubits)?,
            GateKernel::Diagonal(d) => raw.apply_diagonal(d, &gate.qubits)?,
        }
        raw_peak = raw_peak.max(raw.nonzero_count());
        raw_mem = raw_mem.max(raw.memory_bytes());
    }
    println!(
        "  raw sparse, same t=10 circuit: peak support {} (2^{n} = {}), memory {}, {:.2?}\n",
        raw_peak,
        1u64 << n,
        fmt_bytes(raw_mem),
        start.elapsed()
    );

    // ── 3. Same physics: verify against dense where dense exists ──
    let n = 12;
    let bound = clifford_t_circuit(n, 120, 6, 9).bind(&reg)?;
    let mut framed = CliffordFramedState::<C64>::new(n)?;
    bound.run(&mut framed)?;
    let peak = framed.peak_stored_support();
    let mut dense = DenseState::<C64>::new(n)?;
    bound.run(&mut dense)?;
    let deviation = max_amplitude_deviation(&dense, &framed);
    println!("verification at {n} qubits, t=6: peak stored support {peak}, deviation vs dense {deviation:.2e}");
    println!("\nThe frame walks the Clifford part of the circuit through");
    println!("Sp(2n, F2) as metadata; what the amplitudes pay for is the");
    println!("non-Clifford residue. The cost currency is the T-count.");
    Ok(())
}
