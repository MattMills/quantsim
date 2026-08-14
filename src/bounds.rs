//! The measured boundary atlas: every state representation's efficient
//! fragment, the resource its cost is exponential in, and an
//! operational scan for quantum advantage.
//!
//! Each representation in this crate is a bet — an *assumption* about
//! structure — and is efficient exactly while its assumption holds:
//!
//! | representation    | assumption (resource it is exponential in)   |
//! |-------------------|----------------------------------------------|
//! | dense             | none — always `2^n` (the universal reference)|
//! | sparse            | basis-state support stays small              |
//! | factored          | entangled clusters stay small                |
//! | mps               | entanglement across linear cuts stays small  |
//! | mera              | entanglement across tree cuts stays small    |
//! | clifford-framed   | non-Clifford content stays small             |
//!
//! [`resource_profile`] runs one circuit on every representation under
//! the resource guard, recording each one's measured cost (bytes), the
//! parameter that drove it, whether the run stayed exact, and — when a
//! representation hits its wall — the measured refusal instead of a
//! number. [`advantage_scan`] does that across a circuit *family* and
//! classifies each axis's measured growth law ([`Law`]): a family is
//! **classically representable** the moment any assumption holds
//! (some axis stays constant or polynomial while exact), and is an
//! **advantage candidate** only when every measured axis grows
//! exponentially at once. That is the operational meaning of quantum
//! advantage in this crate — and the instrument that would *discover*
//! a sub-exponential simulation: register a new representation, run
//! the scan, and watch whether its axis stays flat where the others
//! blow up.

use crate::backend::{
    Backend, CliffordFramedState, FactoredState, InterferenceState, MeraConfig, MeraState,
    MosaicState, MpsConfig, MpsState, SparseState,
};
use crate::circuit::Circuit;
use crate::error::Result;
use crate::registry::GateRegistry;
use crate::scalar::C64;

/// A measured growth law over a size sweep.
#[derive(Debug, Clone, PartialEq)]
pub enum Law {
    /// Cost stays within noise of flat.
    Constant,
    /// Cost grows like `size^degree` (log–log fit).
    Polynomial {
        /// Fitted exponent.
        degree: f64,
    },
    /// Cost grows like `base^size` (semi-log fit).
    Exponential {
        /// Fitted per-unit-size factor.
        base: f64,
    },
}

impl std::fmt::Display for Law {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Law::Constant => write!(f, "constant"),
            Law::Polynomial { degree } => write!(f, "size^{degree:.1}"),
            Law::Exponential { base } => write!(f, "{base:.2}^size"),
        }
    }
}

impl Law {
    /// Whether this measured law is compatible with efficient classical
    /// simulation (not exponential).
    pub fn is_subexponential(&self) -> bool {
        !matches!(self, Law::Exponential { .. })
    }
}

fn linear_fit(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let syy: f64 = ys.iter().map(|y| (y - my).powi(2)).sum();
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let intercept = my - slope * mx;
    let ss_res: f64 = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| (y - (intercept + slope * x)).powi(2))
        .sum();
    let r2 = if syy > 0.0 { 1.0 - ss_res / syy } else { 1.0 };
    (slope, intercept, r2)
}

/// A classified law together with its fitted coefficients, so measured
/// growth can be *extrapolated*: [`LawFit::predict`] evaluates the fit
/// at any size — the basis of [`select_by_scaling`].
#[derive(Debug, Clone)]
pub struct LawFit {
    /// The classified law.
    pub law: Law,
    ln_intercept: f64,
    slope: f64,
}

impl LawFit {
    /// Predicted cost at `size` under the fitted law.
    pub fn predict(&self, size: usize) -> f64 {
        let x = size as f64;
        match self.law {
            Law::Constant => self.ln_intercept.exp(),
            Law::Polynomial { .. } => (self.ln_intercept + self.slope * x.ln()).exp(),
            Law::Exponential { .. } => (self.ln_intercept + self.slope * x).exp(),
        }
    }
}

/// Fit the measured growth of `costs` over `sizes` (≥ 3 strictly
/// positive points at increasing sizes): flat within a generous noise
/// band is [`Law::Constant`]; otherwise a semi-log (exponential) and a
/// log–log (polynomial) fit compete on residuals, with a fitted
/// exponential base below 1.25 per unit size read as polynomial.
pub fn fit_law(sizes: &[usize], costs: &[usize]) -> LawFit {
    fit_law_with(sizes, costs, 1.25)
}

/// [`fit_law`] with an explicit exponential floor — wall-clock laws use
/// a wider polynomial band (1.35) because timing jitter at microsecond
/// scales can lift a genuinely polynomial axis just past the default.
fn fit_law_with(sizes: &[usize], costs: &[usize], exp_floor: f64) -> LawFit {
    assert!(sizes.len() >= 3 && sizes.len() == costs.len());
    let max = *costs.iter().max().unwrap() as f64;
    let min = *costs.iter().min().unwrap() as f64;
    let xs: Vec<f64> = sizes.iter().map(|&s| s as f64).collect();
    let lys: Vec<f64> = costs.iter().map(|&c| (c as f64).ln()).collect();
    if max / min <= 2.0 {
        let mean = lys.iter().sum::<f64>() / lys.len() as f64;
        return LawFit {
            law: Law::Constant,
            ln_intercept: mean,
            slope: 0.0,
        };
    }
    let (exp_slope, exp_intercept, exp_r2) = linear_fit(&xs, &lys);
    let lxs: Vec<f64> = xs.iter().map(|x| x.ln()).collect();
    let (poly_slope, poly_intercept, poly_r2) = linear_fit(&lxs, &lys);
    let base = exp_slope.exp();
    if base < exp_floor || poly_r2 > exp_r2 + 1e-9 {
        LawFit {
            law: Law::Polynomial { degree: poly_slope },
            ln_intercept: poly_intercept,
            slope: poly_slope,
        }
    } else {
        LawFit {
            law: Law::Exponential { base },
            ln_intercept: exp_intercept,
            slope: exp_slope,
        }
    }
}

/// Classify the measured growth of `costs` over `sizes` — the law of
/// [`fit_law`] without the coefficients.
pub fn classify_law(sizes: &[usize], costs: &[usize]) -> Law {
    fit_law(sizes, costs).law
}

/// One representation's measured probe on one circuit.
#[derive(Debug, Clone)]
pub struct AxisProbe {
    /// Representation name.
    pub axis: &'static str,
    /// Measured cost in bytes (`None` = the representation hit a wall
    /// — a measured refusal, timeout, or unsupported operation).
    pub cost: Option<usize>,
    /// Measured wall-clock nanoseconds for the run (median of three,
    /// so a cold first pass does not masquerade as scaling).
    pub nanos: Option<usize>,
    /// The (min, max) envelope of the timing repetitions — the error
    /// bar the time law carries.
    pub nanos_spread: Option<(usize, usize)>,
    /// The structural parameter that drove the cost, for display.
    pub parameter: String,
    /// Whether the run was exact (no truncation, no approximation).
    pub exact: bool,
    /// The wall's own message, when `cost` is `None`.
    pub note: Option<String>,
}

/// Every representation's measured cost on one circuit.
#[derive(Debug, Clone)]
pub struct ResourceProfile {
    /// Circuit width.
    pub width: usize,
    /// Operation count.
    pub ops: usize,
    /// Per-representation probes, fixed order.
    pub axes: Vec<AxisProbe>,
    /// Destructive-interference weight (a diagnostic of the
    /// computation, not a representation cost).
    pub destroyed: Option<f64>,
}

fn probe<S, F, P>(
    axis: &'static str,
    make: F,
    circuit: &Circuit,
    reg: &GateRegistry<C64>,
    param: P,
) -> AxisProbe
where
    S: Backend<C64>,
    F: Fn(usize) -> Result<S>,
    P: FnOnce(&S) -> (String, bool),
{
    // Every probe runs under a scoped wall-clock budget: an axis whose
    // representation cannot finish in bounded time reports a measured
    // timeout as its wall instead of stalling the whole profile. Time
    // is measured best-of-two so allocation warm-up does not read as
    // scaling.
    let run = |make: &dyn Fn(usize) -> Result<S>| -> Result<(S, usize)> {
        crate::guard::with_time_budget(std::time::Duration::from_secs(10), || {
            let t = std::time::Instant::now();
            let mut state = make(circuit.num_qubits())?;
            circuit.bind(reg)?.run(&mut state)?;
            Ok((state, t.elapsed().as_nanos() as usize))
        })
    };
    match run(&make) {
        Ok((state, first)) => {
            let mut times = vec![first];
            for _ in 0..2 {
                if let Ok((_, t)) = run(&make) {
                    times.push(t);
                }
            }
            times.sort_unstable();
            let median = times[times.len() / 2];
            let spread = (times[0], *times.last().unwrap());
            let (parameter, exact) = param(&state);
            AxisProbe {
                axis,
                cost: Some(state.memory_bytes()),
                nanos: Some(median.max(1)),
                nanos_spread: Some((spread.0.max(1), spread.1.max(1))),
                parameter,
                exact,
                note: None,
            }
        }
        Err(e) => AxisProbe {
            axis,
            cost: None,
            nanos: None,
            nanos_spread: None,
            parameter: "—".into(),
            exact: false,
            note: Some(e.to_string()),
        },
    }
}

/// Run `circuit` on every representation under the resource guard and
/// record each one's measured cost, driving parameter, exactness, and
/// walls. Approximating configs are widened so probes are exact at
/// research sizes (MPS/mera bond caps 4096); a probe that cannot stay
/// within the guard reports its refusal rather than a number.
pub fn resource_profile(circuit: &Circuit) -> ResourceProfile {
    let reg = GateRegistry::<C64>::standard();
    let mut axes = Vec::new();

    axes.push(probe(
        "dense",
        crate::backend::DenseState::<C64>::new,
        circuit,
        &reg,
        |_s: &crate::backend::DenseState<C64>| ("2^n amplitudes".into(), true),
    ));
    axes.push(probe(
        "sparse",
        SparseState::<C64>::new,
        circuit,
        &reg,
        |s: &SparseState<C64>| (format!("support {}", s.nonzero_count()), true),
    ));
    axes.push(probe(
        "factored",
        FactoredState::<C64>::new,
        circuit,
        &reg,
        |s: &FactoredState<C64>| {
            (
                format!(
                    "{} clusters, largest {}",
                    s.factor_count(),
                    s.largest_factor_qubits()
                ),
                true,
            )
        },
    ));
    axes.push(probe(
        "mps",
        |n| {
            MpsState::<C64>::with_config(
                n,
                MpsConfig {
                    max_bond: 4096,
                    trunc_tol: 1e-14,
                },
            )
        },
        circuit,
        &reg,
        |s: &MpsState<C64>| {
            let (bond, _) = s.peak();
            (format!("peak bond {bond}"), bond < 4096)
        },
    ));
    axes.push(probe(
        "mera",
        |n| {
            MeraState::<C64>::with_config(
                n,
                MeraConfig {
                    max_bond: 4096,
                    trunc_tol: 1e-14,
                    max_block: 63,
                },
            )
        },
        circuit,
        &reg,
        |s: &MeraState<C64>| {
            (
                format!("peak block {}", s.peak_block_elements()),
                s.is_exact(),
            )
        },
    ));
    // The bulk register is deliberately NOT an axis yet: its dyadic
    // capacity tree makes memory a sawtooth in width (skewed cuts at
    // non-power-of-two sizes cap rank by the short side — measured:
    // ~2× under mera at n = 10, 12 while equal at n = 8), and a
    // four-point fit across an octave misclassifies the sawtooth as
    // polynomial, leaving certification to the contention-fragile time
    // law. The axis needs octave-aligned sampling first — see the
    // roadmap's clock/bulk rungs.
    //
    // The branched register is in the standard registry but likewise
    // not an axis: with its trivial selector the flat contracted view
    // prices as sparse plus bookkeeping, and the selector's real payoff
    // — branch sharing across scale histories — has no gate-level
    // trigger a circuit harness could exercise.
    //
    // The mosaic: the multi-representation register from singleton
    // regions, structure sculpted by the gates, merges chosen from
    // measured predictions. Its assumption is that the circuit's
    // portions each fit *some* lens in its policy (sparse/dense by
    // default) — the composite axis whose certification would mean a
    // family every fixed lens loses is still classically held by the
    // partition. No capacity tree, so no sawtooth: the fit sees the
    // policy's honest costs.
    axes.push(probe(
        "mosaic",
        MosaicState::<C64>::new,
        circuit,
        &reg,
        |s: &MosaicState<C64>| {
            (
                format!("{} regions, {} events", s.layout().len(), s.events().len()),
                true,
            )
        },
    ));
    // The phase-field representation: cost is the phase polynomial's
    // monomial count, and the assumption is that the circuit stays
    // diagonal-with-root-of-unity-entries over an affine subcube. A gate
    // that leaves the class materializes and the axis honestly reads
    // dense from then on.
    axes.push(probe(
        "phase-field",
        crate::backend::PhaseFieldState::new,
        circuit,
        &reg,
        |s: &crate::backend::PhaseFieldState| match s.monomials() {
            Some(m) => (format!("{m} monomials"), true),
            None => (format!("materialized ({} escapes)", s.escapes()), true),
        },
    ));

    // The braided representation: the word names the state and a
    // stabilizer frame holds it, because every Majorana braid generator
    // is a weight-≤2 Clifford rotation. Cost is the tableau plus the
    // word, at any width.
    axes.push(probe(
        "braided",
        crate::backend::BraidedState::new,
        circuit,
        &reg,
        |s: &crate::backend::BraidedState| {
            (
                format!(
                    "word {}, stored support {}",
                    s.word().len(),
                    s.stored_support()
                ),
                true,
            )
        },
    ));

    axes.push(probe(
        "clifford-framed",
        CliffordFramedState::<C64>::new,
        circuit,
        &reg,
        |s: &CliffordFramedState<C64>| {
            (format!("stored support {}", s.peak_stored_support()), true)
        },
    ));

    // The graph-state bundle: its cost is the link count, and it holds
    // only the Clifford sector — a non-Clifford gate reports as a
    // measured refusal rather than a number, which is exactly what an
    // assumption failing looks like.
    axes.push(probe(
        "bundle",
        |n| {
            let mut bundle = crate::bundle::PolarityBundle::new(n)?;
            <crate::bundle::PolarityBundle as Backend<C64>>::reset(&mut bundle);
            Ok(bundle)
        },
        circuit,
        &reg,
        |s: &crate::bundle::PolarityBundle| {
            let profile = s.profile();
            (
                format!("{} links, {} clusters", profile.links, profile.components),
                true,
            )
        },
    ));
    // The E8 scale tower: cost is the stored lattice-point count, and
    // its own wall is the u64 index rather than the amplitude count.
    axes.push(probe(
        "e8-constellation",
        crate::e8::constellation::E8ConstellationState::new,
        circuit,
        &reg,
        |s: &crate::e8::constellation::E8ConstellationState| {
            (format!("{} lattice points", s.stored_points().len()), true)
        },
    ));

    // Register SHAPES are axes too: the hierarchical splits (k logical
    // qubits carried inside each scalar over a sparse site sector)
    // compete in every scan and selection on the same measured terms.
    axes.push(probe(
        "algebraic-h",
        |n| {
            let k = 1usize.min(n.saturating_sub(1));
            crate::qudit::AlgebraicRegister::<crate::scalar::Quaternion>::new(
                n - k,
                k,
                Box::new(SparseState::new(n - k)?),
            )
        },
        circuit,
        &reg,
        |s: &crate::qudit::AlgebraicRegister<crate::scalar::Quaternion>| {
            (format!("site support {}", s.site_support()), true)
        },
    ));
    axes.push(probe(
        "algebraic-o",
        |n| {
            let k = 2usize.min(n.saturating_sub(1));
            crate::qudit::AlgebraicRegister::<crate::scalar::Octonion>::new(
                n - k,
                k,
                Box::new(SparseState::new(n - k)?),
            )
        },
        circuit,
        &reg,
        |s: &crate::qudit::AlgebraicRegister<crate::scalar::Octonion>| {
            (format!("site support {}", s.site_support()), true)
        },
    ));

    let destroyed = (|| -> Result<f64> {
        let mut intf = InterferenceState::new(circuit.num_qubits())?;
        circuit.bind(&reg)?.run(&mut intf)?;
        Ok(intf.total_destroyed())
    })()
    .ok();

    ResourceProfile {
        width: circuit.num_qubits(),
        ops: circuit.len(),
        axes,
        destroyed,
    }
}

/// The verdict of an [`advantage_scan`].
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Some representation's assumption held: the family is
    /// classically representable, and these axes prove it.
    Classical {
        /// The representations whose measured law stayed
        /// sub-exponential (and exact).
        via: Vec<String>,
    },
    /// Every measured axis grew exponentially (or walled): within this
    /// crate's assumptions, the family is an advantage candidate.
    Candidate,
}

/// One axis of a family scan: name, measured costs and times per size
/// (walls as `None`), the classified laws over the measured points,
/// and whether every run was exact.
#[derive(Debug, Clone)]
pub struct AxisScan {
    /// Representation name.
    pub axis: &'static str,
    /// Measured memory costs per family size.
    pub costs: Vec<Option<usize>>,
    /// Measured wall-clock nanoseconds per family size.
    pub nanos: Vec<Option<usize>>,
    /// Classified memory law (`None` when any size walled).
    pub law: Option<Law>,
    /// Classified time law from the per-size medians (`None` when any
    /// size walled). A representation that is memory-cheap but
    /// time-exponential fails its assumption here — both laws must
    /// stay sub-exponential for the axis to certify a family
    /// classical.
    pub time_law: Option<Law>,
    /// The time law's error bar: the laws of the minimum and maximum
    /// timing envelopes. When both envelopes classify the same way as
    /// the median the law is variance-robust; when they straddle, the
    /// classification is jitter-limited and says so.
    pub time_law_bounds: Option<(Law, Law)>,
    /// Per-size `(min, max)` timing envelope — the raw jitter behind
    /// [`AxisScan::time_law_bounds`], kept so a caller can tell a noisy
    /// measurement from a clean one before trusting either law.
    pub time_spread: Vec<Option<(usize, usize)>>,
    /// Every probe exact.
    pub exact: bool,
}

impl AxisScan {
    /// Whether this axis certifies the family classical: every probe
    /// finished, exactly, with BOTH the memory and the wall-clock law
    /// sub-exponential.
    pub fn certifies_classical(&self) -> bool {
        self.exact
            && self.law.as_ref().is_some_and(Law::is_subexponential)
            && self.time_law.as_ref().is_some_and(Law::is_subexponential)
    }

    /// The worst per-size timing spread, `max / min` — how noisy the
    /// wall-clock measurement was. Near 1.0 is a clean measurement; a
    /// large value means the machine, not the algorithm, dominated the
    /// numbers, and neither the time law nor its envelope should be read
    /// as a property of the code.
    pub fn worst_time_spread_ratio(&self) -> Option<f64> {
        let mut worst: Option<f64> = None;
        for entry in &self.time_spread {
            let (lo, hi) = (*entry)?;
            if lo == 0 {
                return None;
            }
            let ratio = hi as f64 / lo as f64;
            worst = Some(worst.map_or(ratio, |w: f64| w.max(ratio)));
        }
        worst
    }

    /// Measured memory growth across the swept sizes: the last cost over
    /// the first. `None` if any size walled.
    ///
    /// Growth *without* a fitted shape. A fit has to choose between
    /// `Polynomial` and `Exponential`, and over a short sweep those two
    /// are numerically adjacent — a steep polynomial and a shallow
    /// exponential differ by less than timing jitter. This ratio is the
    /// part of the claim that jitter cannot flip, so an assertion about
    /// *how much* a representation pays belongs here, and one about the
    /// *shape* belongs behind [`AxisScan::time_law_is_variance_robust`].
    pub fn measured_growth(&self) -> Option<f64> {
        Self::ratio(&self.costs)
    }

    /// Measured wall-clock growth across the swept sizes, last over
    /// first. `None` if any size walled.
    pub fn measured_time_growth(&self) -> Option<f64> {
        Self::ratio(&self.nanos)
    }

    fn ratio(values: &[Option<usize>]) -> Option<f64> {
        let first = (*values.first()?)? as f64;
        let last = (*values.last()?)? as f64;
        if first <= 0.0 {
            return None;
        }
        Some(last / first)
    }

    /// Whether the time classification is variance-robust: the minimum
    /// and maximum timing envelopes classify the same way
    /// (sub-exponential or not) as the median. A law that flips
    /// between its envelopes is jitter-limited, not measured.
    pub fn time_law_is_variance_robust(&self) -> bool {
        match (&self.time_law, &self.time_law_bounds) {
            (Some(med), Some((lo, hi))) => {
                med.is_subexponential() == lo.is_subexponential()
                    && med.is_subexponential() == hi.is_subexponential()
            }
            _ => false,
        }
    }
}

/// A circuit family scanned against every representation.
#[derive(Debug, Clone)]
pub struct FamilyScan {
    /// Family label.
    pub name: String,
    /// The sizes swept.
    pub sizes: Vec<usize>,
    /// Per-representation measured scan.
    pub axes: Vec<AxisScan>,
    /// The verdict.
    pub verdict: Verdict,
}

/// Sweep a circuit family over `sizes`, profile every size on every
/// representation, classify each axis's measured growth law, and
/// return the verdict: classical the moment any assumption holds,
/// candidate only when every axis escapes.
pub fn advantage_scan(
    name: impl Into<String>,
    family: impl Fn(usize) -> Circuit,
    sizes: &[usize],
) -> FamilyScan {
    assert!(sizes.len() >= 3, "a law needs at least three sizes");
    let profiles: Vec<ResourceProfile> = sizes
        .iter()
        .map(|&s| resource_profile(&family(s)))
        .collect();
    let axis_count = profiles[0].axes.len();
    let mut axes = Vec::with_capacity(axis_count);
    for a in 0..axis_count {
        let costs: Vec<Option<usize>> = profiles.iter().map(|p| p.axes[a].cost).collect();
        let nanos: Vec<Option<usize>> = profiles.iter().map(|p| p.axes[a].nanos).collect();
        let exact = profiles.iter().all(|p| p.axes[a].exact);
        let fit = |values: &[Option<usize>], floor: f64| -> Option<Law> {
            if values.iter().all(|c| c.is_some()) {
                let cs: Vec<usize> = values.iter().map(|c| c.unwrap()).collect();
                Some(fit_law_with(sizes, &cs, floor).law)
            } else {
                None
            }
        };
        let law = fit(&costs, 1.25);
        let time_law = fit(&nanos, 1.35);
        let spreads: Vec<Option<(usize, usize)>> =
            profiles.iter().map(|p| p.axes[a].nanos_spread).collect();
        let time_law_bounds = if spreads.iter().all(|s| s.is_some()) {
            let lows: Vec<Option<usize>> = spreads.iter().map(|s| Some(s.unwrap().0)).collect();
            let highs: Vec<Option<usize>> = spreads.iter().map(|s| Some(s.unwrap().1)).collect();
            match (fit(&lows, 1.35), fit(&highs, 1.35)) {
                (Some(lo), Some(hi)) => Some((lo, hi)),
                _ => None,
            }
        } else {
            None
        };
        axes.push(AxisScan {
            axis: profiles[0].axes[a].axis,
            costs,
            nanos,
            law,
            time_law,
            time_law_bounds,
            time_spread: spreads,
            exact,
        });
    }
    let via: Vec<String> = axes
        .iter()
        .filter(|a| a.certifies_classical())
        .map(|a| a.axis.to_string())
        .collect();
    let verdict = if via.is_empty() {
        Verdict::Candidate
    } else {
        Verdict::Classical { via }
    };
    FamilyScan {
        name: name.into(),
        sizes: sizes.to_vec(),
        axes,
        verdict,
    }
}

/// One representation's extrapolated prediction at a target size.
#[derive(Debug, Clone)]
pub struct ScalingChoice {
    /// Representation name.
    pub axis: &'static str,
    /// The fitted memory law behind the prediction.
    pub law: Law,
    /// Predicted bytes at the target size.
    pub predicted_bytes: f64,
    /// Every probe stayed exact.
    pub exact: bool,
}

/// Backend selection by *extrapolated measured scaling* rather than a
/// benchmark at one size: the family is probed at small sizes, every
/// representation's memory law fitted, and the predictions ranked at
/// the target size.
#[derive(Debug, Clone)]
pub struct ScalingSelection {
    /// Family label.
    pub name: String,
    /// The size the selection is for.
    pub target: usize,
    /// Predictions, cheapest first (walled axes excluded).
    pub choices: Vec<ScalingChoice>,
    /// Whether the winning law is sub-exponential — when false, no
    /// assumption holds and the choice is only least-bad.
    pub subexponential: bool,
}

impl ScalingSelection {
    /// The cheapest predicted representation.
    pub fn best(&self) -> &ScalingChoice {
        &self.choices[0]
    }
}

/// Choose a representation for `family` at `target` by fitting each
/// axis's measured memory growth over `probe_sizes` and extrapolating —
/// the scaling-law counterpart of benchmark-based
/// [`select_backend`](crate::harness::select_backend). Exact axes are
/// preferred: an inexact axis is ranked only if no exact one exists.
pub fn select_by_scaling(
    name: impl Into<String>,
    family: impl Fn(usize) -> Circuit,
    probe_sizes: &[usize],
    target: usize,
) -> ScalingSelection {
    assert!(probe_sizes.len() >= 3, "a fit needs at least three sizes");
    let profiles: Vec<ResourceProfile> = probe_sizes
        .iter()
        .map(|&s| resource_profile(&family(s)))
        .collect();
    let mut choices = Vec::new();
    for a in 0..profiles[0].axes.len() {
        let costs: Vec<Option<usize>> = profiles.iter().map(|p| p.axes[a].cost).collect();
        if costs.iter().any(|c| c.is_none()) {
            continue;
        }
        let cs: Vec<usize> = costs.iter().map(|c| c.unwrap()).collect();
        let fit = fit_law(probe_sizes, &cs);
        choices.push(ScalingChoice {
            axis: profiles[0].axes[a].axis,
            predicted_bytes: fit.predict(target),
            law: fit.law,
            exact: profiles.iter().all(|p| p.axes[a].exact),
        });
    }
    let any_exact = choices.iter().any(|c| c.exact);
    choices.sort_by(|x, y| {
        (any_exact && !x.exact)
            .cmp(&(any_exact && !y.exact))
            .then(x.predicted_bytes.partial_cmp(&y.predicted_bytes).unwrap())
    });
    let subexponential = choices
        .first()
        .map(|c| c.law.is_subexponential())
        .unwrap_or(false);
    ScalingSelection {
        name: name.into(),
        target,
        choices,
        subexponential,
    }
}
