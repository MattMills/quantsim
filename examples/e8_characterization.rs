//! The full scaling characterization of the two E8 backends, and the
//! measured answer to "can you compute *across* a set of E8 volumes":
//! width ceilings, the gate-fidelity census over the whole registry,
//! the min/max performance envelope, the fitted memory and time laws,
//! and the reach-per-cost of the native cross-scale operators.
//!
//! Run with `cargo run --release --example e8_characterization`.

use quantsim::characterize::{characterize, CharacterizeConfig};
use quantsim::e8::across::{self, Native};
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

/// A layer of Hadamards: maximal support, the worst case for anything
/// that stores amplitudes.
fn h_layer(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    c
}

/// A depth-3 Clifford brickwork: entangling but support-concentrated.
fn clifford_brickwork(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    for layer in 0..3 {
        for q in (layer % 2..n.saturating_sub(1)).step_by(2) {
            c.h(q);
            c.cx(q, q + 1);
        }
    }
    c
}

fn main() -> Result<()> {
    // Armed so the over-scale cells in the sweep below abort with a
    // measured elapsed time instead of running unbounded — every wall
    // printed is a real refusal.
    guard::set_time_budget(Some(std::time::Duration::from_secs(8)));

    let mut backends = BackendRegistry::<C64>::standard();
    backends.register_e8()?;
    let sim = Simulator::with_registries(GateRegistry::standard(), backends);

    let families: [(&str, &dyn Fn(usize) -> Circuit); 3] = [
        ("ghz", &|n| library::ghz(n)),
        ("h-layer", &h_layer),
        ("clifford", &clifford_brickwork),
    ];

    println!("== E8 backends: full characterization ==\n");
    for backend in ["e8-rep", "e8-constellation", "sparse", "dense"] {
        let cfg = CharacterizeConfig {
            widths: vec![4, 8, 12, 16, 20, 24],
            fidelity: true,
            envelope_width: 8,
            sweep_max: 24,
            ..Default::default()
        };
        match characterize(&sim, backend, &families, &cfg) {
            Ok(report) => println!("{report}"),
            Err(e) => println!("== {backend} ==\n  characterization refused: {e}\n"),
        }
    }

    println!("== Computing ACROSS a set of E8 volumes ==\n");
    println!("One native operation, measured on probe basis states:");
    println!(
        "  {:<12} {:>6} {:>8} {:>9} {:>7} {:>6} {:>7} {:>9}",
        "op", "qubits", "touched", "involved", "copies", "comps", "2q-min", "nanos"
    );
    let ops = [
        across::cross_scale_translation(),
        Native::Modulate(quantsim::e8::constellation::basis(0)),
        Native::Permute([1, 2, 3, 4, 5, 6, 7, 0]),
        Native::Fourier(0),
    ];
    for n in [8usize, 24, 40] {
        for op in &ops {
            match across::reach(n, op) {
                Ok(r) => println!(
                    "  {:<12} {:>6} {:>8} {:>9} {:>7} {:>6} {:>7} {:>9}",
                    r.op,
                    r.num_qubits,
                    r.qubits_touched,
                    r.qubits_involved.map_or("—".to_string(), |v| v.to_string()),
                    format!("{}/{}", r.copies_touched, r.copies),
                    r.influence_components
                        .map_or("—".to_string(), |c| c.to_string()),
                    r.two_qubit_lower_bound
                        .map_or("—".to_string(), |b| b.to_string()),
                    r.native_nanos
                ),
                Err(e) => println!("  {:<12} {:>6} refused: {e}", op.label(), n),
            }
        }
    }

    println!("\nReach and cost against the number of E8 copies:");
    let scaling = across::across_scaling(6, &across::cross_scale_translation())?;
    println!(
        "  copies      {:?}\n  touched     {:?}\n  2q-min      {:?}\n  native ns   {:?}",
        scaling.copies, scaling.qubits_touched, scaling.two_qubit_lower_bound, scaling.native_nanos
    );
    println!(
        "  per extra copy: {} qubits touched, {} two-qubit gates replaced",
        scaling.per_copy_reach, scaling.per_copy_advantage
    );
    println!(
        "  reach law over copies:     {:?}\n  advantage law over copies: {:?}\n  \
         native cost law:           {:?}",
        scaling.reach_law, scaling.advantage_law, scaling.cost_law
    );

    println!("\nThe native direct DFT, the one support-growing native op:");
    let fourier = across::fourier_cost(7, 0)?;
    println!(
        "  copies      {:?}\n  support     {:?}\n  nanos       {:?}",
        fourier.copies, fourier.support, fourier.nanos
    );
    println!(
        "  support law: {:?}\n  cost law:    {:?}",
        fourier.support_law, fourier.cost_law
    );

    println!("\n== What the native operator set can reach ==\n");
    // Prepare a genuinely spread state first: a size-1 support is a
    // coset with a linear character for trivial reasons and would prove
    // nothing about the class.
    let prep = [
        Native::Fourier(0),
        Native::Fourier(1),
        across::cross_scale_translation(),
        Native::Modulate(quantsim::e8::constellation::basis(1)),
    ];
    for op in &ops {
        match across::class_preservation(16, &prep, op) {
            Ok(p) => println!(
                "  {:<12} in-class before {} → after {} (coset {}, |ψ| uniform {}, χ residual {:.1e})",
                p.op,
                p.before.in_class,
                p.after.in_class,
                p.after.is_coset,
                p.after.uniform_modulus,
                p.after.character_residual
            ),
            Err(e) => println!("  {:<12} refused: {e}", op.label()),
        }
    }
    for (gate, params, qubits) in [
        ("h", vec![], vec![0usize]),
        ("t", vec![], vec![0usize]),
        ("cx", vec![], vec![0usize, 9]),
    ] {
        match across::qubit_gate_class(16, &prep, gate, &params, &qubits) {
            Ok(c) => println!(
                "  qubit '{gate}'   in-class {} (size {}, coset {}, uniform {}, χ residual {:.1e})",
                c.in_class, c.size, c.is_coset, c.uniform_modulus, c.character_residual
            ),
            Err(e) => println!("  qubit '{gate}'   refused: {e}"),
        }
    }

    println!("\n== Footprint per amplitude, measured ==");
    for backend in ["dense", "sparse", "e8-constellation"] {
        let env = quantsim::characterize::perf_envelope(&sim, backend, 8, 3)?;
        println!(
            "  {:<18} {:.1}..{:.1} bytes/amplitude, support {}..{}, {} timed / {} refused",
            backend,
            env.bytes_per_amplitude.0,
            env.bytes_per_amplitude.1,
            env.support_range.0,
            env.support_range.1,
            env.timed,
            env.refused
        );
    }

    let ghz = library::ghz::<C64>(24);
    for backend in ["sparse", "e8-constellation"] {
        let state = sim.run_on(backend, &ghz)?;
        println!(
            "  GHZ-24 on {:<16} {} over {} amplitudes",
            backend,
            fmt_bytes(state.memory_bytes()),
            state.nonzero_count()
        );
    }
    Ok(())
}
