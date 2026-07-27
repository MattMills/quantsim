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

    // ── 1. Gottesman–Knill's *evolution* sector through the frame ──
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
    println!("  (dense would need 2^50 amplitudes = 16 PiB; note: evolution");
    println!("   only — no amplitude was read. Readout flushes; see section 4.)\n");

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

    // ── 4. The honest boundary: exactly Gottesman–Knill, never more ──
    // 4a. The free sector is the Pauli normalizer and nothing else. The
    //     machine partitions the whole registry by the mechanism it
    //     actually used (absorption is checked against an independent
    //     dense ground truth in tests/clifford_frames.rs).
    let generic = [0.7365, 1.2113, -0.5871];
    let mut buckets: [(&str, Vec<String>); 5] = [
        ("absorbed (Clifford, metadata only)", Vec::new()),
        ("axis rotation (amplitudes touched)", Vec::new()),
        ("Walsh Z-strings (amplitudes touched)", Vec::new()),
        ("ZYZ split (amplitudes touched)", Vec::new()),
        ("flush + raw", Vec::new()),
    ];
    let mut names = reg.names();
    names.sort();
    for name in names {
        let def = reg.resolve(&name)?;
        let k = def.arity();
        let m = def.matrix(&generic[..def.param_count()])?;
        let mut probe = CliffordFramedState::<C64>::new(k)?;
        probe.apply(&m, &(0..k).collect::<Vec<_>>())?;
        let s = probe.stats();
        let bucket = if s.absorbed_clifford == 1 {
            0
        } else if s.axis_rotations == 1 {
            1
        } else if s.diagonal_rotations > 0 {
            2
        } else if s.zyz_decompositions == 1 {
            3
        } else {
            4
        };
        buckets[bucket].1.push(name);
    }
    println!("\nregistry partition at generic parameters (the machine's own free sector):");
    for (label, members) in &buckets {
        println!("  {label}: {}", members.join(" "));
    }

    // 4b. Why T can never absorb: its conjugation leaves the Pauli group.
    let e = cis(std::f64::consts::FRAC_PI_4);
    let (a01, a10) = (e, e.conj()); // A = T†XT has these off-diagonals
    let c_x = (a01 + a10) / 2.0;
    let c_y = (c64(0.0, 1.0) * (a01 - a10)) / 2.0;
    println!(
        "\nT†·X·T = {:+.3}·X {:+.3}·Y — coefficients not in {{±1}}, so T is outside",
        c_x.re, c_y.re
    );
    println!("the Pauli normalizer: absorption *must* refuse it (unit-tested), and its");
    println!("cost lands on amplitudes. No free magic — else this loop would prove BQP=BPP.");

    // 4c. The label's scope: evolution is polynomial; readout is not yet.
    let n = 16;
    let bound = clifford_t_circuit(n, 300, 0, 13).bind(&reg)?;
    let mut state = CliffordFramedState::<C64>::new(n)?;
    bound.run(&mut state)?;
    let before = state.peak_stored_support();
    let start = std::time::Instant::now();
    let _ = state.amplitude(0); // one readout query
    let flush_time = start.elapsed();
    println!("\nreadout boundary at {n} qubits: peak stored support {before} during evolution,");
    println!(
        "  {} after one amplitude() call ({:.2?} flush) — Pauli MEASUREMENT is",
        state.peak_stored_support(),
        flush_time
    );
    println!("  native and repaired (measure_pauli: no flush, adaptive sequences");
    println!("  stay flat), but FULL AMPLITUDE EXTRACTION still materializes the");
    println!("  physical support — extracting 2^n numbers is not a Gottesman-Knill");
    println!("  capability and never was. The tests pin both sides of the boundary.");

    println!("\nThe frame walks the Clifford part of the circuit through");
    println!("Sp(2n, F2) as metadata; what the amplitudes pay for is the");
    println!("non-Clifford residue. The cost currency is the T-count.");
    Ok(())
}
