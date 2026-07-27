//! The research benchmark harness: quantify what a backend swap buys you —
//! wall time, memory, support size — with correctness checked against a
//! reference in the same run, so a "10× faster" claim can never silently be
//! "10× faster and wrong".
//!
//! Workflow (see `examples/research_mode.rs`):
//!
//! 1. register your gate set and/or backend on a [`Simulator`],
//! 2. [`crate::conformance::verify_backend`] until it passes,
//! 3. [`compare_backends`] over named [`Workload`]s and read speedups and
//!    memory ratios off the [`BenchmarkReport`].
//!
//! Timing uses `std::time::Instant` (min over repetitions) so it works from
//! any test or research script; use the criterion benches in `benches/` for
//! statistically rigorous micro-benchmarks.

use std::fmt;
use std::time::Instant;

use crate::backend::max_amplitude_deviation;
use crate::circuit::Circuit;
use crate::error::Result;
use crate::library;
use crate::scalar::Scalar;
use crate::sim::Simulator;

/// A named circuit family to benchmark.
pub struct Workload<S: Scalar> {
    name: String,
    build: Box<dyn Fn() -> Circuit<S> + Send + Sync>,
}

impl<S: Scalar> Workload<S> {
    /// Build from a closure producing the circuit.
    pub fn new(
        name: impl Into<String>,
        build: impl Fn() -> Circuit<S> + Send + Sync + 'static,
    ) -> Self {
        Workload {
            name: name.into(),
            build: Box::new(build),
        }
    }

    /// Wrap an already-built circuit.
    pub fn from_circuit(name: impl Into<String>, circuit: Circuit<S>) -> Self {
        Self::new(name, move || circuit.clone())
    }

    /// GHZ preparation on `n` qubits (concentrated support).
    pub fn ghz(n: usize) -> Self {
        Self::new(format!("ghz-{n}"), move || library::ghz(n))
    }

    /// Quantum Fourier transform on `n` qubits (saturating support).
    pub fn qft(n: usize) -> Self {
        Self::new(format!("qft-{n}"), move || library::qft(n))
    }

    /// Deterministic random circuit over the standard pool.
    pub fn random(n: usize, gates: usize, seed: u64) -> Self {
        Self::new(format!("random-{n}q-{gates}g"), move || {
            library::random_circuit(n, gates, seed)
        })
    }

    /// Workload name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Materialize the circuit.
    pub fn circuit(&self) -> Circuit<S> {
        (self.build)()
    }
}

/// Configuration for [`compare_backends`].
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Backend used for correctness deviations. `None` skips verification;
    /// a reference that cannot run a workload (e.g. dense at 40 qubits)
    /// yields `deviation: None` for that workload instead of failing.
    pub reference: Option<String>,
    /// Timed repetitions per (workload, backend); minimum is reported.
    pub repetitions: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        BenchConfig {
            reference: Some("dense".to_string()),
            repetitions: 3,
        }
    }
}

/// One (workload, backend) measurement.
#[derive(Debug, Clone)]
pub struct RunRecord {
    /// Workload name.
    pub workload: String,
    /// Backend name.
    pub backend: String,
    /// Best-of-repetitions wall time in seconds (`None` if the run failed).
    pub seconds: Option<f64>,
    /// Final-state memory footprint in bytes.
    pub memory_bytes: usize,
    /// Final-state stored nonzeros.
    pub nonzeros: usize,
    /// Final total Born weight.
    pub total_weight: f64,
    /// Max amplitude deviation vs the reference (`None` when unavailable).
    pub deviation: Option<f64>,
    /// Error message if the backend could not run the workload.
    pub error: Option<String>,
}

/// All measurements from one [`compare_backends`] call.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// The reference backend, if any.
    pub reference: Option<String>,
    /// One record per (workload, backend), in run order.
    pub records: Vec<RunRecord>,
}

impl BenchmarkReport {
    /// Record lookup.
    pub fn record(&self, workload: &str, backend: &str) -> Option<&RunRecord> {
        self.records
            .iter()
            .find(|r| r.workload == workload && r.backend == backend)
    }

    /// `reference time / backend time` for a workload — > 1 means the
    /// backend is faster than the reference.
    pub fn speedup(&self, workload: &str, backend: &str) -> Option<f64> {
        let reference = self.reference.as_deref()?;
        let ref_secs = self.record(workload, reference)?.seconds?;
        let secs = self.record(workload, backend)?.seconds?;
        (secs > 0.0).then(|| ref_secs / secs)
    }

    /// `reference memory / backend memory` for a workload — > 1 means the
    /// backend is smaller than the reference.
    pub fn memory_ratio(&self, workload: &str, backend: &str) -> Option<f64> {
        let reference = self.reference.as_deref()?;
        let ref_mem = self.record(workload, reference)?.memory_bytes;
        let mem = self.record(workload, backend)?.memory_bytes;
        (mem > 0).then(|| ref_mem as f64 / mem as f64)
    }

    /// Worst deviation across all records that have one.
    pub fn max_deviation(&self) -> f64 {
        self.records
            .iter()
            .filter_map(|r| r.deviation)
            .fold(0.0, f64::max)
    }
}

fn fmt_seconds(s: f64) -> String {
    if s >= 1.0 {
        format!("{s:.2} s")
    } else if s >= 1e-3 {
        format!("{:.2} ms", s * 1e3)
    } else {
        format!("{:.2} µs", s * 1e6)
    }
}

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

impl fmt::Display for BenchmarkReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut workloads: Vec<&str> = Vec::new();
        for r in &self.records {
            if !workloads.contains(&r.workload.as_str()) {
                workloads.push(&r.workload);
            }
        }
        for workload in workloads {
            writeln!(f, "workload {workload}:")?;
            writeln!(
                f,
                "  {:<12} {:>12} {:>12} {:>10} {:>10} {:>10} {:>12}",
                "backend", "time", "memory", "nonzeros", "speedup", "mem×", "deviation"
            )?;
            for r in self.records.iter().filter(|r| r.workload == workload) {
                if let Some(err) = &r.error {
                    writeln!(f, "  {:<12} failed: {err}", r.backend)?;
                    continue;
                }
                let speedup = self
                    .speedup(workload, &r.backend)
                    .map(|s| format!("{s:.2}×"))
                    .unwrap_or_else(|| "-".into());
                let memx = self
                    .memory_ratio(workload, &r.backend)
                    .map(|m| format!("{m:.1}×"))
                    .unwrap_or_else(|| "-".into());
                let dev = r
                    .deviation
                    .map(|d| format!("{d:.1e}"))
                    .unwrap_or_else(|| "-".into());
                writeln!(
                    f,
                    "  {:<12} {:>12} {:>12} {:>10} {:>10} {:>10} {:>12}",
                    r.backend,
                    r.seconds.map(fmt_seconds).unwrap_or_else(|| "-".into()),
                    fmt_bytes(r.memory_bytes),
                    r.nonzeros,
                    speedup,
                    memx,
                    dev
                )?;
            }
        }
        Ok(())
    }
}

/// Run every workload on every backend, timing each and checking the final
/// state against the reference where the reference can run it.
pub fn compare_backends<S: Scalar>(
    sim: &Simulator<S>,
    workloads: &[Workload<S>],
    backends: &[&str],
    cfg: &BenchConfig,
) -> Result<BenchmarkReport> {
    let mut report = BenchmarkReport {
        reference: cfg.reference.clone(),
        records: Vec::new(),
    };
    for workload in workloads {
        let circuit = workload.circuit();
        let reference_state = match &cfg.reference {
            Some(name) => sim.run_on(name, &circuit).ok(),
            None => None,
        };
        for &backend in backends {
            let mut best: Option<f64> = None;
            let mut final_state = None;
            let mut error = None;
            for _ in 0..cfg.repetitions.max(1) {
                let start = Instant::now();
                match sim.run_on(backend, &circuit) {
                    Ok(state) => {
                        let elapsed = start.elapsed().as_secs_f64();
                        best = Some(best.map_or(elapsed, |b: f64| b.min(elapsed)));
                        final_state = Some(state);
                    }
                    Err(e) => {
                        error = Some(e.to_string());
                        break;
                    }
                }
            }
            let record = match (final_state, error) {
                (Some(state), _) => {
                    let deviation = reference_state
                        .as_ref()
                        .map(|r| max_amplitude_deviation(r.as_ref(), state.as_ref()));
                    RunRecord {
                        workload: workload.name().to_string(),
                        backend: backend.to_string(),
                        seconds: best,
                        memory_bytes: state.memory_bytes(),
                        nonzeros: state.nonzero_count(),
                        total_weight: state.total_weight(),
                        deviation,
                        error: None,
                    }
                }
                (None, error) => RunRecord {
                    workload: workload.name().to_string(),
                    backend: backend.to_string(),
                    seconds: None,
                    memory_bytes: 0,
                    nonzeros: 0,
                    total_weight: 0.0,
                    deviation: None,
                    error,
                },
            };
            report.records.push(record);
        }
    }
    Ok(report)
}
