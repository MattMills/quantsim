//! Full performance characterization of a state representation: the
//! measured width ceiling, the measured gate-fidelity census, and the
//! measured minimum/maximum performance envelope, in one report.
//!
//! [`conformance`](crate::conformance) answers *is this backend
//! correct*, and [`bounds`](crate::bounds) answers *how does this
//! circuit family scale across every backend*. Neither answers the
//! question you ask of a new representation first: **how wide can it
//! go, how much of the gate set does it actually reproduce, and what
//! are its best and worst cases** — the numbers that go in a table
//! next to `dense` and `sparse`.
//!
//! Everything here is measured by running the backend. A ceiling is
//! the width at which a real construction or a real gate application
//! refused, carrying the refusal's own message
//! ([`Wall`]); a fidelity count is a comparison against the reference
//! backend gate by gate; an envelope is wall-clock timing and reported
//! footprint at the widths that ran. Nothing is inferred from
//! arithmetic about what *should* fit.

use std::fmt;
use std::time::Instant;

use crate::bounds::{fit_law, Law};
use crate::circuit::Circuit;
use crate::conformance::ConformanceReport;
use crate::error::Result;
use crate::scalar::Scalar;
use crate::sim::Simulator;

/// Why a width sweep stopped — always a refusal the library actually
/// produced, or the sweep's own limit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Wall {
    /// The constructor refused at the next width: this is the
    /// representation's declared ceiling, and the string is its own
    /// message (`TooManyQubits`, `OutOfMemory`, …).
    Construction(String),
    /// Construction succeeded but an operation refused: a guard
    /// admission, a deadline abort, or an unsupported gate.
    Operation(String),
    /// The sweep reached its requested limit without any refusal — the
    /// representation is not the binding constraint here.
    SweepLimit,
}

impl fmt::Display for Wall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Wall::Construction(m) => write!(f, "construction refused: {m}"),
            Wall::Operation(m) => write!(f, "operation refused: {m}"),
            Wall::SweepLimit => write!(f, "no refusal up to the sweep limit"),
        }
    }
}

/// The widest register a backend will *construct*, and the refusal one
/// qubit past it.
#[derive(Debug, Clone)]
pub struct ConstructionCeiling {
    /// Backend name.
    pub backend: String,
    /// Widest width that constructed (0 if even one qubit refused).
    pub widest: usize,
    /// The wall one qubit past `widest`.
    pub wall: Wall,
}

/// The widest width at which a backend ran a whole circuit family, and
/// what it cost there.
#[derive(Debug, Clone)]
pub struct WidthCeiling {
    /// Backend name.
    pub backend: String,
    /// Family label.
    pub family: String,
    /// Widest width that ran to completion (0 if none did).
    pub reached: usize,
    /// Reported footprint at `reached`.
    pub bytes: usize,
    /// Stored amplitude count at `reached`.
    pub support: usize,
    /// The wall one width past `reached`.
    pub wall: Wall,
}

/// How much of the gate registry a backend reproduces, counted.
#[derive(Debug, Clone)]
pub struct FidelityCensus {
    /// Backend under test.
    pub backend: String,
    /// Reference backend.
    pub reference: String,
    /// Registry entries swept (names, so aliases count separately).
    pub gates_total: usize,
    /// Gates reproducing the reference to the last bit — deviation
    /// exactly `0.0`, not merely within tolerance.
    pub bit_exact: usize,
    /// Gates within tolerance but not bit-exact.
    pub within_tolerance: usize,
    /// Gates outside tolerance — measured disagreement.
    pub outside_tolerance: usize,
    /// Gates the sweep could not exercise (arity past its width cap).
    pub not_swept: usize,
    /// The tolerance the census was taken at.
    pub tolerance: f64,
    /// Worst gate and its deviation.
    pub worst: (String, f64),
    /// Total (width × ordering × parameter) cases run.
    pub cases: usize,
    /// Random registry circuits compared.
    pub random_circuits: usize,
    /// Random circuits whose sampled counts disagreed.
    pub sampling_mismatches: usize,
    /// Measurement-collapse invariant violations.
    pub collapse_violations: usize,
}

impl FidelityCensus {
    /// Gates matching the reference at the census tolerance.
    pub fn matching(&self) -> usize {
        self.bit_exact + self.within_tolerance
    }

    /// Whether every swept gate matched.
    pub fn complete(&self) -> bool {
        self.outside_tolerance == 0 && self.not_swept == 0
    }
}

/// Count a [`ConformanceReport`] into a census. `tolerance` should be
/// the one the report was produced with.
pub fn fidelity_census(report: &ConformanceReport, tolerance: f64) -> FidelityCensus {
    let mut census = FidelityCensus {
        backend: report.backend.clone(),
        reference: report.reference.clone(),
        gates_total: report.gate_checks.len(),
        bit_exact: 0,
        within_tolerance: 0,
        outside_tolerance: 0,
        not_swept: 0,
        tolerance,
        worst: (String::new(), 0.0),
        cases: 0,
        random_circuits: report.random_circuit_cases,
        sampling_mismatches: report.sampling_mismatches,
        collapse_violations: report.collapse_violations,
    };
    for check in &report.gate_checks {
        census.cases += check.cases;
        if check.cases == 0 {
            census.not_swept += 1;
            continue;
        }
        if check.max_deviation == 0.0 {
            census.bit_exact += 1;
        } else if check.max_deviation <= tolerance {
            census.within_tolerance += 1;
        } else {
            census.outside_tolerance += 1;
        }
        if check.max_deviation >= census.worst.1 {
            census.worst = (check.gate.clone(), check.max_deviation);
        }
    }
    census
}

/// One gate's measured cost on one backend at one width.
#[derive(Debug, Clone)]
pub struct GateCost {
    /// Registry name.
    pub gate: String,
    /// Median nanoseconds per application over the repetitions.
    pub nanos: u128,
    /// Stored amplitude count after the application.
    pub support: usize,
    /// Reported footprint after the application.
    pub bytes: usize,
}

/// The measured extremes of a backend's behaviour at one width: the
/// cheapest and most expensive gate in the registry, and the range of
/// footprint per stored amplitude.
#[derive(Debug, Clone)]
pub struct PerfEnvelope {
    /// Backend name.
    pub backend: String,
    /// Width the envelope was measured at.
    pub width: usize,
    /// Gates timed (the rest refused, and are counted in `refused`).
    pub timed: usize,
    /// Gates that refused at this width.
    pub refused: usize,
    /// Cheapest gate measured.
    pub fastest: Option<GateCost>,
    /// Most expensive gate measured.
    pub slowest: Option<GateCost>,
    /// Smallest and largest support reached over the sweep.
    pub support_range: (usize, usize),
    /// Smallest and largest reported bytes per stored amplitude — the
    /// representation's per-amplitude overhead, measured rather than
    /// computed from `size_of`.
    pub bytes_per_amplitude: (f64, f64),
}

impl PerfEnvelope {
    /// Ratio of the slowest to the fastest measured gate — the
    /// dynamic range of the representation's gate cost.
    pub fn dynamic_range(&self) -> Option<f64> {
        let (fast, slow) = (self.fastest.as_ref()?, self.slowest.as_ref()?);
        if fast.nanos == 0 {
            return None;
        }
        Some(slow.nanos as f64 / fast.nanos as f64)
    }
}

/// Measure every registry gate on `backend` at `width`, one gate at a
/// time on a freshly reset register, and return the extremes.
///
/// Each gate is applied `reps` times on separate fresh states and the
/// median is kept, so a cold allocation does not become the cost. A
/// gate that refuses (arity past the width, guard admission, backend
/// restriction) is counted in [`PerfEnvelope::refused`] rather than
/// silently skipped.
pub fn perf_envelope<S: Scalar>(
    sim: &Simulator<S>,
    backend: &str,
    width: usize,
    reps: usize,
) -> Result<PerfEnvelope> {
    let reps = reps.max(1);
    let mut envelope = PerfEnvelope {
        backend: backend.to_string(),
        width,
        timed: 0,
        refused: 0,
        fastest: None,
        slowest: None,
        support_range: (usize::MAX, 0),
        bytes_per_amplitude: (f64::MAX, 0.0),
    };
    for name in sim.registry().names() {
        let def = sim.registry().resolve(&name)?;
        if def.arity() > width {
            envelope.refused += 1;
            continue;
        }
        let qubits: Vec<usize> = (0..def.arity()).collect();
        // A fixed non-special angle: no parameter accidentally lands on
        // an identity or a permutation and reads as free.
        let params: Vec<f64> = (0..def.param_count()).map(|_| 0.7).collect();
        let mut circuit = Circuit::new(width);
        circuit.gate(name.clone(), params, qubits);
        let bound = match circuit.bind(sim.registry()) {
            Ok(bound) => bound,
            Err(_) => {
                envelope.refused += 1;
                continue;
            }
        };
        let mut samples = Vec::with_capacity(reps);
        let mut last: Option<(usize, usize)> = None;
        for _ in 0..reps {
            let mut state = match sim.backends().create(backend, width) {
                Ok(state) => state,
                Err(_) => break,
            };
            state.reset();
            let start = Instant::now();
            let outcome = bound.run(state.as_mut());
            let elapsed = start.elapsed().as_nanos();
            if outcome.is_err() {
                break;
            }
            samples.push(elapsed);
            last = Some((state.nonzero_count(), state.memory_bytes()));
        }
        let (Some((support, bytes)), false) = (last, samples.is_empty()) else {
            envelope.refused += 1;
            continue;
        };
        samples.sort_unstable();
        let cost = GateCost {
            gate: name,
            nanos: samples[samples.len() / 2],
            support,
            bytes,
        };
        envelope.timed += 1;
        envelope.support_range.0 = envelope.support_range.0.min(support);
        envelope.support_range.1 = envelope.support_range.1.max(support);
        if support > 0 {
            let per = bytes as f64 / support as f64;
            envelope.bytes_per_amplitude.0 = envelope.bytes_per_amplitude.0.min(per);
            envelope.bytes_per_amplitude.1 = envelope.bytes_per_amplitude.1.max(per);
        }
        if envelope
            .fastest
            .as_ref()
            .map_or(true, |f| cost.nanos < f.nanos)
        {
            envelope.fastest = Some(cost.clone());
        }
        if envelope
            .slowest
            .as_ref()
            .map_or(true, |s| cost.nanos > s.nanos)
        {
            envelope.slowest = Some(cost);
        }
    }
    if envelope.support_range.0 == usize::MAX {
        envelope.support_range.0 = 0;
    }
    if envelope.bytes_per_amplitude.0 == f64::MAX {
        envelope.bytes_per_amplitude.0 = 0.0;
    }
    Ok(envelope)
}

/// Find the widest register `backend` will construct, at most
/// `sweep_max`, together with the refusal one qubit past it.
///
/// Constructibility is monotone in width for every shipped
/// representation, so this walks up from 1 and reports the first
/// refusal — the message is the representation's own.
pub fn construction_ceiling<S: Scalar>(
    sim: &Simulator<S>,
    backend: &str,
    sweep_max: usize,
) -> ConstructionCeiling {
    let mut widest = 0;
    let mut wall = Wall::SweepLimit;
    for n in 1..=sweep_max {
        match sim.backends().create(backend, n) {
            Ok(_) => widest = n,
            Err(e) => {
                wall = Wall::Construction(e.to_string());
                break;
            }
        }
    }
    ConstructionCeiling {
        backend: backend.to_string(),
        widest,
        wall,
    }
}

/// Run `family` at increasing widths on `backend` and report the widest
/// that finished, what it cost there, and the wall one width past it.
///
/// The family is built at each width and run in full, so the ceiling is
/// the *circuit's* ceiling on this representation, not the
/// constructor's: a backend that constructs 63 qubits and then refuses
/// the first Hadamard layer at 26 reports 25 here, with the guard's
/// message.
pub fn width_ceiling<S: Scalar>(
    sim: &Simulator<S>,
    backend: &str,
    family_name: &str,
    family: impl Fn(usize) -> Circuit<S>,
    widths: &[usize],
) -> WidthCeiling {
    let mut ceiling = WidthCeiling {
        backend: backend.to_string(),
        family: family_name.to_string(),
        reached: 0,
        bytes: 0,
        support: 0,
        wall: Wall::SweepLimit,
    };
    for &n in widths {
        let mut state = match sim.backends().create(backend, n) {
            Ok(state) => state,
            Err(e) => {
                ceiling.wall = Wall::Construction(e.to_string());
                break;
            }
        };
        state.reset();
        let circuit = family(n);
        let bound = match circuit.bind(sim.registry()) {
            Ok(bound) => bound,
            Err(e) => {
                ceiling.wall = Wall::Operation(e.to_string());
                break;
            }
        };
        match bound.run(state.as_mut()) {
            Ok(()) => {
                ceiling.reached = n;
                ceiling.bytes = state.memory_bytes();
                ceiling.support = state.nonzero_count();
            }
            Err(e) => {
                ceiling.wall = Wall::Operation(e.to_string());
                break;
            }
        }
    }
    ceiling
}

/// A family's measured memory and time laws on one backend, fitted
/// over the widths that ran.
#[derive(Debug, Clone)]
pub struct FamilyLaws {
    /// Family label.
    pub family: String,
    /// Widths that ran to completion.
    pub widths: Vec<usize>,
    /// Reported footprint at each width.
    pub bytes: Vec<usize>,
    /// Median wall-clock nanoseconds at each width.
    pub nanos: Vec<usize>,
    /// Fitted memory law (`None` with fewer than three points).
    pub memory_law: Option<Law>,
    /// Fitted wall-clock law (`None` with fewer than three points).
    pub time_law: Option<Law>,
}

/// Run `family` at each width, three times per width, and fit the
/// measured memory and wall-clock laws over the widths that finished.
///
/// Stops at the first width that refuses, so the laws are fitted over
/// real runs only — a walled width contributes nothing rather than a
/// fabricated cost.
pub fn family_laws<S: Scalar>(
    sim: &Simulator<S>,
    backend: &str,
    family_name: &str,
    family: impl Fn(usize) -> Circuit<S>,
    widths: &[usize],
) -> FamilyLaws {
    let mut laws = FamilyLaws {
        family: family_name.to_string(),
        widths: Vec::new(),
        bytes: Vec::new(),
        nanos: Vec::new(),
        memory_law: None,
        time_law: None,
    };
    for &n in widths {
        let circuit = family(n);
        let Ok(bound) = circuit.bind(sim.registry()) else {
            break;
        };
        let mut samples = Vec::new();
        let mut footprint = None;
        for _ in 0..3 {
            let Ok(mut state) = sim.backends().create(backend, n) else {
                break;
            };
            state.reset();
            let start = Instant::now();
            if bound.run(state.as_mut()).is_err() {
                samples.clear();
                break;
            }
            samples.push(start.elapsed().as_nanos() as usize);
            footprint = Some(state.memory_bytes());
        }
        let (Some(bytes), false) = (footprint, samples.is_empty()) else {
            break;
        };
        samples.sort_unstable();
        laws.widths.push(n);
        laws.bytes.push(bytes);
        laws.nanos.push(samples[samples.len() / 2].max(1));
    }
    if laws.widths.len() >= 3 {
        laws.memory_law = Some(fit_law(&laws.widths, &laws.bytes).law);
        laws.time_law = Some(fit_law(&laws.widths, &laws.nanos).law);
    }
    laws
}

/// A named circuit family: the label a report shows and the builder that
/// produces the circuit at a given width.
pub type Family<'a, S> = (&'a str, &'a dyn Fn(usize) -> Circuit<S>);

/// Knobs for [`characterize`]. `Default` gives a run that finishes in
/// seconds on every shipped backend.
#[derive(Debug, Clone)]
pub struct CharacterizeConfig {
    /// Reference backend for the fidelity census (default `"dense"`).
    pub reference: String,
    /// Highest width the construction sweep will try (default 32).
    pub sweep_max: usize,
    /// Widths each family is swept over (default 2, 4, …, 20).
    pub widths: Vec<usize>,
    /// Width the performance envelope is measured at (default 8).
    pub envelope_width: usize,
    /// Timing repetitions per gate in the envelope (default 5).
    pub envelope_reps: usize,
    /// Fidelity tolerance (default 1e-9).
    pub tolerance: f64,
    /// Whether to run the conformance sweep for the fidelity census
    /// (default true). Set false for a backend whose reference cannot
    /// be constructed at the sweep's widths.
    pub fidelity: bool,
}

impl Default for CharacterizeConfig {
    fn default() -> Self {
        CharacterizeConfig {
            reference: "dense".to_string(),
            sweep_max: 32,
            widths: (1..=10).map(|k| 2 * k).collect(),
            envelope_width: 8,
            envelope_reps: 5,
            tolerance: 1e-9,
            fidelity: true,
        }
    }
}

/// Characterize `backend` end to end: construction ceiling, fidelity
/// census against the reference, per-family width ceilings and fitted
/// laws, and the performance envelope.
///
/// Every number in the returned [`Characterization`] came from a run.
/// A backend that refuses a family contributes that refusal's message,
/// not an absence.
pub fn characterize<S: Scalar>(
    sim: &Simulator<S>,
    backend: &str,
    families: &[Family<'_, S>],
    cfg: &CharacterizeConfig,
) -> Result<Characterization> {
    // Fail fast on an unknown name rather than reporting a ceiling of 0.
    sim.backends().create(backend, 1)?;
    let construction = construction_ceiling(sim, backend, cfg.sweep_max);
    let fidelity = if cfg.fidelity {
        let conf = crate::conformance::ConformanceConfig {
            reference: cfg.reference.clone(),
            tolerance: cfg.tolerance,
            ..Default::default()
        };
        let report = crate::conformance::verify_backend(sim, backend, &conf)?;
        Some(fidelity_census(&report, cfg.tolerance))
    } else {
        None
    };
    let mut ceilings = Vec::with_capacity(families.len());
    let mut laws = Vec::with_capacity(families.len());
    for (name, family) in families {
        ceilings.push(width_ceiling(sim, backend, name, family, &cfg.widths));
        laws.push(family_laws(sim, backend, name, family, &cfg.widths));
    }
    let envelope = perf_envelope(sim, backend, cfg.envelope_width, cfg.envelope_reps).ok();
    Ok(Characterization {
        backend: backend.to_string(),
        algebra: S::algebra_name(),
        construction,
        fidelity,
        ceilings,
        laws,
        envelope,
    })
}

/// Everything measurable about one representation in one report: the
/// construction ceiling, the fidelity census, the per-family width
/// ceilings and laws, and the performance envelope.
#[derive(Debug, Clone)]
pub struct Characterization {
    /// Backend name.
    pub backend: String,
    /// Amplitude algebra label.
    pub algebra: String,
    /// The widest register that constructs.
    pub construction: ConstructionCeiling,
    /// Gate-set agreement against the reference (`None` when the
    /// backend was not conformance-tested in this run).
    pub fidelity: Option<FidelityCensus>,
    /// Per-family width ceilings.
    pub ceilings: Vec<WidthCeiling>,
    /// Per-family fitted laws.
    pub laws: Vec<FamilyLaws>,
    /// The performance envelope, at the width it was measured.
    pub envelope: Option<PerfEnvelope>,
}

impl Characterization {
    /// The widest width reached on any family — the representation's
    /// measured practical ceiling, as distinct from its declared one.
    pub fn best_reach(&self) -> Option<&WidthCeiling> {
        self.ceilings.iter().max_by_key(|c| c.reached)
    }

    /// The narrowest ceiling over the families — the worst case.
    pub fn worst_reach(&self) -> Option<&WidthCeiling> {
        self.ceilings.iter().min_by_key(|c| c.reached)
    }
}

impl fmt::Display for Characterization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "== {} ({} amplitudes) ==", self.backend, self.algebra)?;
        writeln!(
            f,
            "  constructs up to  : {} qubits ({})",
            self.construction.widest, self.construction.wall
        )?;
        if let Some(cen) = &self.fidelity {
            writeln!(
                f,
                "  gate fidelity     : {}/{} match vs {} ({} bit-exact, {} within {:.0e}, \
                 {} outside, {} not swept) over {} cases",
                cen.matching(),
                cen.gates_total,
                cen.reference,
                cen.bit_exact,
                cen.within_tolerance,
                cen.tolerance,
                cen.outside_tolerance,
                cen.not_swept,
                cen.cases
            )?;
            writeln!(
                f,
                "  worst gate        : {} at {:.2e}",
                cen.worst.0, cen.worst.1
            )?;
        }
        for c in &self.ceilings {
            writeln!(
                f,
                "  {:<18}: {} qubits, {} B, support {} — {}",
                c.family, c.reached, c.bytes, c.support, c.wall
            )?;
        }
        for l in &self.laws {
            let mem = l
                .memory_law
                .as_ref()
                .map_or("too few points".to_string(), |law| format!("{law:?}"));
            let time = l
                .time_law
                .as_ref()
                .map_or("too few points".to_string(), |law| format!("{law:?}"));
            writeln!(
                f,
                "  laws {:<13}: memory {mem}, time {time} over {:?}",
                l.family, l.widths
            )?;
        }
        if let Some(env) = &self.envelope {
            writeln!(
                f,
                "  envelope @ n={}   : {} gates timed, {} refused",
                env.width, env.timed, env.refused
            )?;
            if let (Some(fast), Some(slow)) = (&env.fastest, &env.slowest) {
                writeln!(
                    f,
                    "    fastest         : {} at {} ns (support {})",
                    fast.gate, fast.nanos, fast.support
                )?;
                writeln!(
                    f,
                    "    slowest         : {} at {} ns (support {})",
                    slow.gate, slow.nanos, slow.support
                )?;
            }
            writeln!(
                f,
                "    support range   : {}..{}",
                env.support_range.0, env.support_range.1
            )?;
            writeln!(
                f,
                "    bytes/amplitude : {:.1}..{:.1}",
                env.bytes_per_amplitude.0, env.bytes_per_amplitude.1
            )?;
        }
        Ok(())
    }
}
