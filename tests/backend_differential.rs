//! Differential testing of every shipped backend over the *whole*
//! `Backend` trait, not just gate application.
//!
//! The conformance harness (`tests/conformance_harness.rs`) sweeps the gate
//! registry through `Simulator::run_on`; this suite drives each backend
//! directly through random interleavings of everything else a caller can
//! do — `apply`, `apply_diagonal`, `apply_permutation`, seeded `measure`,
//! `pauli_expectation`, `sample`, `load` and `reset` — and compares against
//! a reference after every step. Three contracts are held:
//!
//! 1. **Same state**: amplitudes agree to 1e-9 after every operation (up
//!    to a global phase for the graph-state bundle, which defines vertex
//!    operators only up to one).
//! 2. **Same answers**: measurement outcomes, expectation values and
//!    sampled counts are seed-for-seed identical to the reference.
//! 3. **Refusals are clean**: an operation a backend refuses (by name, or
//!    by the resource guard) leaves its state untouched. This is the
//!    contract an MPS `load` used to break — it reset the qubit layout
//!    before its buffer was refused, scrambling any routed state.
//!
//! The narrow sweep runs every backend against dense at widths 1..=9 over
//! the full standard registry; the wide sweep runs the width-scalable
//! backends against sparse at 31..=63 qubits (the `u32`/`u64` shift
//! boundaries) over a support-bounded Clifford+T pool.

use quantsim::prelude::*;

/// Deviation minimized over a global phase, fitted from the largest
/// amplitude of `a`.
fn dev_up_to_phase(a: &dyn Backend<C64>, b: &dyn Backend<C64>) -> f64 {
    let mut anchor = (0u64, 0.0f64);
    a.for_each_nonzero(&mut |i, z| {
        if z.norm() > anchor.1 {
            anchor = (i, z.norm());
        }
    });
    let y = b.amplitude(anchor.0);
    if anchor.1 == 0.0 || y.norm() < 1e-12 {
        return max_amplitude_deviation(a, b);
    }
    let phase = a.amplitude(anchor.0) / y;
    let mut dev = 0.0f64;
    a.for_each_nonzero(&mut |i, z| dev = dev.max((z - b.amplitude(i) * phase).norm()));
    b.for_each_nonzero(&mut |i, z| dev = dev.max((a.amplitude(i) - z * phase).norm()));
    dev
}

fn pick(rng: &mut Prng, n: usize, k: usize) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (rng.next_u64() as usize) % (n - i);
        pool.swap(i, j);
    }
    pool[..k].to_vec()
}

fn snapshot(n: usize, state: &dyn Backend<C64>) -> SparseState<C64> {
    let mut entries = Vec::new();
    state.for_each_nonzero(&mut |i, a| entries.push((i, a)));
    let mut snap = SparseState::new(n).unwrap();
    snap.load(&entries).unwrap();
    snap
}

/// One operation's observable result, compared across backends:
/// `Ok(values)` or the refusal message.
type Outcome = std::result::Result<Vec<f64>, String>;

fn unit(r: Result<()>) -> Outcome {
    r.map(|_| Vec::new()).map_err(|e| e.to_string())
}

struct Sweep<'a> {
    reference: &'a str,
    widths: &'a [usize],
    /// Gate names to draw from (the full registry for the narrow sweep).
    gates: Vec<String>,
    trials: u64,
    steps: usize,
}

#[derive(Default)]
struct Findings {
    /// Disagreements: the suite's failures.
    problems: Vec<String>,
    /// Operations the backend refused cleanly, keyed by operation.
    refusals: std::collections::BTreeMap<&'static str, usize>,
    steps: usize,
}

fn drive(sim: &Simulator, backend: &str, sweep: &Sweep) -> Findings {
    let phase_loose = backend == "bundle";
    let dev = |a: &dyn Backend<C64>, b: &dyn Backend<C64>| {
        if phase_loose {
            dev_up_to_phase(a, b)
        } else {
            max_amplitude_deviation(a, b)
        }
    };
    let mut out = Findings::default();
    'trials: for trial in 0..sweep.trials {
        let mut rng = Prng::new(0xD1FF ^ trial.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let n = sweep.widths[(rng.next_u64() as usize) % sweep.widths.len()];
        let full = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
        let mut r = sim.backends().create(sweep.reference, n).unwrap();
        let mut t = sim.backends().create(backend, n).unwrap();
        for step in 0..sweep.steps {
            let before = snapshot(n, t.as_ref());
            let (op, rr, rt): (&'static str, Outcome, Outcome) = match rng.next_u64() % 10 {
                0..=3 => {
                    let name = &sweep.gates[(rng.next_u64() as usize) % sweep.gates.len()];
                    let def = sim.registry().resolve(name).unwrap();
                    if def.arity() > n {
                        continue;
                    }
                    let qs = pick(&mut rng, n, def.arity());
                    let ps: Vec<f64> = (0..def.param_count())
                        .map(|_| (rng.next_f64() * 2.0 - 1.0) * std::f64::consts::PI)
                        .collect();
                    let m = def.matrix(&ps).unwrap();
                    ("apply", unit(r.apply(&m, &qs)), unit(t.apply(&m, &qs)))
                }
                4 => {
                    let k = 1 + (rng.next_u64() as usize) % n.min(3);
                    let qs = pick(&mut rng, n, k);
                    let e: Vec<C64> = (0..1usize << k)
                        .map(|_| C64::from_polar(1.0, rng.next_f64() * std::f64::consts::TAU))
                        .collect();
                    let a = unit(r.apply_diagonal(&e, &qs));
                    ("apply_diagonal", a, unit(t.apply_diagonal(&e, &qs)))
                }
                5 => {
                    let (mask, add) = (rng.next_u64() & full, rng.next_u64() & full);
                    let f = move |i: u64| (i ^ mask).wrapping_add(add) & full;
                    let a = unit(r.apply_permutation("xor-add", &f));
                    (
                        "apply_permutation",
                        a,
                        unit(t.apply_permutation("xor-add", &f)),
                    )
                }
                6 => {
                    let q = (rng.next_u64() as usize) % n;
                    let s = rng.next_u64();
                    let m = |st: &mut Box<dyn Backend<C64>>| {
                        st.measure(q, &mut Prng::new(s))
                            .map(|b| vec![b as u8 as f64])
                            .map_err(|e| e.to_string())
                    };
                    ("measure", m(&mut r), m(&mut t))
                }
                7 => {
                    let k = 1 + (rng.next_u64() as usize) % n.min(4);
                    let ops: Vec<(usize, Pauli)> = pick(&mut rng, n, k)
                        .into_iter()
                        .map(|q| {
                            let p = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];
                            (q, p[(rng.next_u64() % 4) as usize])
                        })
                        .collect();
                    let e = |st: &dyn Backend<C64>| {
                        pauli_expectation(st, &ops)
                            .map(|v| vec![v.re, v.im])
                            .map_err(|e| e.to_string())
                    };
                    ("pauli_expectation", e(r.as_ref()), e(t.as_ref()))
                }
                8 => {
                    let s = rng.next_u64();
                    let draw = |st: &dyn Backend<C64>| {
                        st.sample(64, &mut Prng::new(s))
                            .map(|c| {
                                let mut v: Vec<(u64, u64)> = c.into_iter().collect();
                                v.sort_unstable();
                                v.into_iter()
                                    .flat_map(|(i, c)| [i as f64, c as f64])
                                    .collect()
                            })
                            .map_err(|e| e.to_string())
                    };
                    ("sample", draw(r.as_ref()), draw(t.as_ref()))
                }
                _ if rng.next_u64() % 4 == 0 => {
                    r.reset();
                    t.reset();
                    ("reset", Ok(Vec::new()), Ok(Vec::new()))
                }
                _ => {
                    let count = 1 + (rng.next_u64() as usize) % 4;
                    let mut idx: Vec<u64> = (0..count).map(|_| rng.next_u64() & full).collect();
                    idx.sort_unstable();
                    idx.dedup();
                    let norm = (idx.len() as f64).sqrt();
                    let entries: Vec<(u64, C64)> = idx
                        .iter()
                        .map(|&i| (i, C64::from_polar(1.0 / norm, rng.next_f64() * 6.0)))
                        .collect();
                    ("load", unit(r.load(&entries)), unit(t.load(&entries)))
                }
            };
            out.steps += 1;
            let at = format!("[{backend}] trial {trial} n={n} step {step} {op}");
            match (rr, rt) {
                (Err(e), _) => panic!("{at}: the reference refused: {e}"),
                (Ok(a), Ok(b)) => {
                    let same =
                        a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-9);
                    if !same {
                        out.problems.push(format!("{at}: reference {a:?} vs {b:?}"));
                        continue 'trials;
                    }
                }
                (Ok(_), Err(_)) => {
                    // A clean refusal: the state must be exactly as it was.
                    *out.refusals.entry(op).or_default() += 1;
                    let moved = dev(&before, t.as_ref());
                    if moved > 1e-12 {
                        out.problems
                            .push(format!("{at}: refused, but the state moved by {moved:.3e}"));
                        continue 'trials;
                    }
                    // Rewind the reference to the untouched state (the
                    // bundle refuses `load`, so resync in this direction)
                    // and carry on.
                    let mut entries = Vec::new();
                    before.for_each_nonzero(&mut |i, a| entries.push((i, a)));
                    r.load(&entries).unwrap();
                }
            }
            let d = dev(r.as_ref(), t.as_ref());
            if d.is_nan() || d > 1e-9 {
                out.problems
                    .push(format!("{at}: amplitudes deviate by {d:.3e}"));
                continue 'trials;
            }
        }
    }
    out
}

#[test]
fn every_backend_agrees_with_dense_over_the_whole_trait() {
    let sim: Simulator = Simulator::new();
    let sweep = Sweep {
        reference: "dense",
        widths: &[1, 2, 3, 4, 5, 6, 7, 8, 9],
        gates: sim.registry().names(),
        trials: 48,
        steps: 40,
    };
    for backend in [
        "sparse", "adaptive", "factored", "mps", "mera", "bulk", "mosaic", "branched", "bundle",
    ] {
        let f = drive(&sim, backend, &sweep);
        assert!(f.problems.is_empty(), "{:#?}", f.problems);
        assert!(f.steps > 1000, "{backend}: only {} steps ran", f.steps);
        if backend == "bundle" {
            // The Clifford-only description refuses non-Clifford gates and
            // arbitrary amplitudes by name — and every refusal was clean.
            assert!(f.refusals["apply"] > 0, "{:?}", f.refusals);
        } else {
            // The general-purpose backends refuse nothing at these widths.
            assert!(f.refusals.is_empty(), "{backend}: {:?}", f.refusals);
        }
    }
}

#[test]
fn width_scalable_backends_agree_with_sparse_at_the_shift_boundaries() {
    let sim: Simulator = Simulator::new();
    let gates = [
        "h", "s", "sdg", "t", "x", "y", "cx", "cz", "swap", "rz", "ccx",
    ];
    let sweep = Sweep {
        reference: "sparse",
        widths: &[31, 32, 33, 47, 62, 63],
        gates: gates.iter().map(|g| g.to_string()).collect(),
        trials: 16,
        steps: 40,
    };
    for backend in ["adaptive", "mps", "bulk", "mosaic", "branched", "factored"] {
        let f = drive(&sim, backend, &sweep);
        assert!(f.problems.is_empty(), "{:#?}", f.problems);
        if backend == "mps" {
            // Sparse loads (and the permutations built on them) compile
            // through the support's prefix tree: nothing at 63 qubits
            // needs a 2^n buffer any more.
            assert!(f.refusals.is_empty(), "mps: {:?}", f.refusals);
        }
    }
}
