//! Backend conformance: prove that *any* state representation matches the
//! reference on *any* gate set, before you trust a single benchmark number.
//!
//! This is the research-mode safety net. [`verify_backend`] takes a
//! [`Simulator`] — with whatever research gates its registry holds — and a
//! backend name, and sweeps:
//!
//! 1. **every registered gate** (aliases and custom sets included,
//!    discovered from the registry) at randomized parameters, qubit
//!    orderings and widths, comparing amplitudes against the reference
//!    backend on scrambled entangled states;
//! 2. **random circuits drawn from the registry itself** — register a
//!    custom gate set and it is automatically part of the sweep;
//! 3. **invariants**: Born-weight conservation (for division algebras),
//!    sampling equality against the reference, and measurement-collapse
//!    consistency.
//!
//! The result is a structured [`ConformanceReport`] with per-gate maximum
//! deviations, so a new backend (a matrix-product state, say) or a new gate
//! set either passes measurably or fails with the offending gate named.
//!
//! Scrambling uses raw real-valued Ry/CX matrices rather than registry
//! gates, so conformance works even for registries that carry only an
//! exotic gate set, and over every amplitude algebra.

use std::fmt;

use crate::backend::max_amplitude_deviation;
use crate::circuit::Circuit;
use crate::error::Result;
use crate::math::GateMatrix;
use crate::registry::GateRegistry;
use crate::rng::Prng;
use crate::scalar::Scalar;
use crate::sim::Simulator;

/// Knobs for [`verify_backend`]. `Default` gives a thorough-but-quick run.
#[derive(Debug, Clone)]
pub struct ConformanceConfig {
    /// Backend treated as ground truth (default `"dense"`).
    pub reference: String,
    /// Extra qubits beyond each gate's arity for the per-gate sweep
    /// (each entry is one width variant; default `[0, 2]`).
    pub extra_widths: Vec<usize>,
    /// Distinct random qubit orderings per gate and width (default 3).
    pub orderings: usize,
    /// Number of random registry-drawn circuits (default 24).
    pub random_circuits: usize,
    /// Gates per random circuit (default 30).
    pub gates_per_circuit: usize,
    /// Shots for the sampling-equality check (default 256).
    pub shots: u64,
    /// Widest gate the per-gate sweep will attempt (default 8 qubits;
    /// wider gates are skipped and noted in the report).
    pub max_gate_width: usize,
    /// Amplitude tolerance (default 1e-9).
    pub tolerance: f64,
    /// Seed for the whole sweep (default 0xBB9); same seed → same sweep.
    pub seed: u64,
}

impl Default for ConformanceConfig {
    fn default() -> Self {
        ConformanceConfig {
            reference: "dense".to_string(),
            extra_widths: vec![0, 2],
            orderings: 3,
            random_circuits: 24,
            gates_per_circuit: 30,
            shots: 256,
            max_gate_width: 8,
            tolerance: 1e-9,
            seed: 0xBB9,
        }
    }
}

/// Per-gate result of the conformance sweep.
#[derive(Debug, Clone)]
pub struct GateCheck {
    /// Gate name (registry key, including aliases).
    pub gate: String,
    /// Number of (width × ordering × parameter) cases exercised.
    pub cases: usize,
    /// Worst amplitude deviation from the reference across all cases.
    pub max_deviation: f64,
}

/// Structured outcome of [`verify_backend`].
#[derive(Debug, Clone)]
pub struct ConformanceReport {
    /// Backend under test.
    pub backend: String,
    /// Reference backend.
    pub reference: String,
    /// Amplitude algebra label.
    pub algebra: String,
    /// One entry per registry name.
    pub gate_checks: Vec<GateCheck>,
    /// Random registry-circuit cases run.
    pub random_circuit_cases: usize,
    /// Worst amplitude deviation observed anywhere.
    pub max_amplitude_deviation: f64,
    /// Worst `|total Born weight − 1|` (division algebras only; 0 otherwise).
    pub max_weight_drift: f64,
    /// Random circuits whose sampled counts differed from the reference.
    pub sampling_mismatches: usize,
    /// Measurement-collapse invariant violations.
    pub collapse_violations: usize,
    /// Human-readable failure descriptions; empty means the backend passed.
    pub failures: Vec<String>,
}

impl ConformanceReport {
    /// Whether every check stayed within tolerance.
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    /// The gates with the largest deviations, worst first.
    pub fn worst_gates(&self, count: usize) -> Vec<&GateCheck> {
        let mut sorted: Vec<&GateCheck> = self.gate_checks.iter().collect();
        sorted.sort_by(|a, b| b.max_deviation.total_cmp(&a.max_deviation));
        sorted.truncate(count);
        sorted
    }
}

impl fmt::Display for ConformanceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "conformance: '{}' vs reference '{}' over {} — {}",
            self.backend,
            self.reference,
            self.algebra,
            if self.passed() { "PASSED" } else { "FAILED" }
        )?;
        let cases: usize = self.gate_checks.iter().map(|g| g.cases).sum();
        writeln!(
            f,
            "  {} gates ({cases} cases), {} random circuits; max deviation {:.2e}, weight drift {:.2e}",
            self.gate_checks.len(),
            self.random_circuit_cases,
            self.max_amplitude_deviation,
            self.max_weight_drift
        )?;
        writeln!(
            f,
            "  sampling mismatches: {}, collapse violations: {}",
            self.sampling_mismatches, self.collapse_violations
        )?;
        for failure in &self.failures {
            writeln!(f, "  FAIL: {failure}")?;
        }
        Ok(())
    }
}

/// A deterministic random circuit drawn from a registry's own gates:
/// whatever is registered — standard, aliases, research sets — is eligible
/// (gates wider than `n` are skipped). Parameters are uniform in [−π, π].
pub fn random_registry_circuit<S: Scalar>(
    registry: &GateRegistry<S>,
    n: usize,
    gates: usize,
    seed: u64,
) -> Circuit<S> {
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    let eligible: Vec<String> = registry
        .names()
        .into_iter()
        .filter(|name| {
            registry
                .get(name)
                .map(|d| d.arity() >= 1 && d.arity() <= n)
                .unwrap_or(false)
        })
        .collect();
    if eligible.is_empty() {
        return c;
    }
    for _ in 0..gates {
        let name = &eligible[(rng.next_u64() as usize) % eligible.len()];
        let def = registry.get(name).expect("filtered above");
        let qubits = pick_distinct(&mut rng, n, def.arity());
        let params: Vec<f64> = (0..def.param_count())
            .map(|_| (rng.next_f64() * 2.0 - 1.0) * std::f64::consts::PI)
            .collect();
        c.gate(name.clone(), params, qubits);
    }
    c
}

/// Verify `backend` against the configured reference across the simulator's
/// entire gate registry plus random circuits and invariants.
pub fn verify_backend<S: Scalar>(
    sim: &Simulator<S>,
    backend: &str,
    cfg: &ConformanceConfig,
) -> Result<ConformanceReport> {
    // Fail fast on unknown backend names.
    sim.backends().create(&cfg.reference, 1)?;
    sim.backends().create(backend, 1)?;

    let mut rng = Prng::new(cfg.seed);
    let mut report = ConformanceReport {
        backend: backend.to_string(),
        reference: cfg.reference.clone(),
        algebra: S::algebra_name(),
        gate_checks: Vec::new(),
        random_circuit_cases: 0,
        max_amplitude_deviation: 0.0,
        max_weight_drift: 0.0,
        sampling_mismatches: 0,
        collapse_violations: 0,
        failures: Vec::new(),
    };

    // 1. Per-gate sweep over the whole registry.
    for name in sim.registry().names() {
        let def = sim.registry().resolve(&name)?;
        let arity = def.arity();
        let mut check = GateCheck {
            gate: name.clone(),
            cases: 0,
            max_deviation: 0.0,
        };
        if arity > cfg.max_gate_width {
            report.failures.push(format!(
                "gate '{name}': arity {arity} exceeds max_gate_width — not swept"
            ));
            report.gate_checks.push(check);
            continue;
        }
        for &extra in &cfg.extra_widths {
            let width = (arity + extra).min(cfg.max_gate_width.max(arity));
            for _ in 0..cfg.orderings.max(1) {
                let qubits = pick_distinct(&mut rng, width, arity);
                let params: Vec<f64> = (0..def.param_count())
                    .map(|_| (rng.next_f64() * 2.0 - 1.0) * std::f64::consts::PI)
                    .collect();
                let mut circuit = Circuit::new(width);
                raw_scrambler(&mut circuit, width, &mut rng);
                circuit.gate(name.clone(), params.clone(), qubits.clone());
                raw_scrambler(&mut circuit, width, &mut rng);

                let reference = sim.run_on(&cfg.reference, &circuit)?;
                let under_test = sim.run_on(backend, &circuit)?;
                let dev = max_amplitude_deviation(reference.as_ref(), under_test.as_ref());
                check.max_deviation = check.max_deviation.max(dev);
                check.cases += 1;
                if dev > cfg.tolerance {
                    report.failures.push(format!(
                        "gate '{name}' on {qubits:?} (width {width}, params {params:?}): \
                         deviation {dev:.3e} > {:.1e}",
                        cfg.tolerance
                    ));
                }
            }
        }
        report.max_amplitude_deviation = report.max_amplitude_deviation.max(check.max_deviation);
        report.gate_checks.push(check);
    }

    // 2. Random circuits drawn from the registry, with invariants.
    for case in 0..cfg.random_circuits {
        let n = 2 + (rng.next_u64() as usize % 4); // widths 2..=5
        let circuit_seed = rng.next_u64();
        let circuit =
            random_registry_circuit(sim.registry(), n, cfg.gates_per_circuit, circuit_seed);
        let reference = sim.run_on(&cfg.reference, &circuit)?;
        let under_test = sim.run_on(backend, &circuit)?;
        report.random_circuit_cases += 1;

        let dev = max_amplitude_deviation(reference.as_ref(), under_test.as_ref());
        report.max_amplitude_deviation = report.max_amplitude_deviation.max(dev);
        if dev > cfg.tolerance {
            report.failures.push(format!(
                "random circuit #{case} (n={n}, seed {circuit_seed:#x}): deviation {dev:.3e}"
            ));
        }

        if S::DIVISION {
            let drift = (under_test.total_weight() - 1.0).abs();
            report.max_weight_drift = report.max_weight_drift.max(drift);
            if drift > cfg.tolerance.max(1e-9) * 100.0 {
                report.failures.push(format!(
                    "random circuit #{case}: Born weight drifted by {drift:.3e} on a division algebra"
                ));
            }
        }

        // Sampling equality (basis-ordered accumulation makes this exact
        // across conforming backends).
        let sample_seed = rng.next_u64();
        let counts_ref = reference.sample(cfg.shots, &mut Prng::new(sample_seed));
        let counts_test = under_test.sample(cfg.shots, &mut Prng::new(sample_seed));
        match (counts_ref, counts_test) {
            (Ok(a), Ok(b)) => {
                if a != b {
                    report.sampling_mismatches += 1;
                    report
                        .failures
                        .push(format!("random circuit #{case}: sampled counts differ"));
                }
            }
            (Err(_), Err(_)) => {} // both refused (e.g. exotic algebra): consistent
            _ => {
                report.sampling_mismatches += 1;
                report.failures.push(format!(
                    "random circuit #{case}: one backend sampled, the other refused"
                ));
            }
        }

        // Measurement collapse on the backend under test.
        let mut collapsed = sim.run_on(backend, &circuit)?;
        let qubit = (rng.next_u64() as usize) % n;
        let measure_seed = rng.next_u64();
        if let Ok(outcome) = collapsed.measure(qubit, &mut Prng::new(measure_seed)) {
            let mut opposite = 0.0;
            collapsed.for_each_nonzero(&mut |i, a| {
                if ((i >> qubit) & 1 == 1) != outcome {
                    opposite += a.born_weight().abs();
                }
            });
            let renorm_ok = !S::DIVISION || (collapsed.total_weight() - 1.0).abs() < 1e-6;
            if opposite > cfg.tolerance || !renorm_ok {
                report.collapse_violations += 1;
                report.failures.push(format!(
                    "random circuit #{case}: collapse invariant violated on q{qubit} \
                     (opposite branch {opposite:.3e})"
                ));
            }
        }
    }

    Ok(report)
}

fn pick_distinct(rng: &mut Prng, width: usize, count: usize) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..width).collect();
    for i in 0..count {
        let j = i + (rng.next_u64() as usize) % (width - i);
        pool.swap(i, j);
    }
    pool.truncate(count);
    pool
}

/// Entangling scrambler from raw real Ry/CX matrices: registry-independent
/// (works even if the registry holds only an exotic gate set) and valid
/// over every amplitude algebra (all entries are real).
fn raw_scrambler<S: Scalar>(circuit: &mut Circuit<S>, width: usize, rng: &mut Prng) {
    for q in 0..width {
        let theta = (rng.next_f64() * 2.0 - 1.0) * std::f64::consts::PI;
        circuit.raw("scramble-ry", ry_raw::<S>(theta), vec![q]);
    }
    for q in 0..width.saturating_sub(1) {
        circuit.raw("scramble-cx", cx_raw::<S>(), vec![q, q + 1]);
    }
    for q in 0..width {
        let theta = (rng.next_f64() * 2.0 - 1.0) * std::f64::consts::PI;
        circuit.raw("scramble-ry", ry_raw::<S>(theta), vec![q]);
    }
}

fn ry_raw<S: Scalar>(theta: f64) -> GateMatrix<S> {
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    GateMatrix::from_vec(
        2,
        vec![S::from_re(c), S::from_re(-s), S::from_re(s), S::from_re(c)],
    )
    .expect("2x2 is a valid dimension")
}

fn cx_raw<S: Scalar>() -> GateMatrix<S> {
    let (o, l) = (S::one(), S::zero());
    GateMatrix::from_vec(4, vec![o, l, l, l, l, l, l, o, l, l, o, l, l, o, l, l])
        .expect("4x4 is a valid dimension")
}
