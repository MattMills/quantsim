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
    MpsConfig, MpsState, SparseState,
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

/// Classify the measured growth of `costs` over `sizes` (≥ 3 strictly
/// positive points at increasing sizes): flat within a generous noise
/// band is [`Law::Constant`]; otherwise a semi-log (exponential) and a
/// log–log (polynomial) fit compete on residuals, with a fitted
/// exponential base below 1.25 per unit size read as polynomial.
pub fn classify_law(sizes: &[usize], costs: &[usize]) -> Law {
    assert!(sizes.len() >= 3 && sizes.len() == costs.len());
    let max = *costs.iter().max().unwrap() as f64;
    let min = *costs.iter().min().unwrap() as f64;
    if max / min <= 2.0 {
        return Law::Constant;
    }
    let xs: Vec<f64> = sizes.iter().map(|&s| s as f64).collect();
    let lys: Vec<f64> = costs.iter().map(|&c| (c as f64).ln()).collect();
    let (exp_slope, _, exp_r2) = linear_fit(&xs, &lys);
    let lxs: Vec<f64> = xs.iter().map(|x| x.ln()).collect();
    let (poly_slope, _, poly_r2) = linear_fit(&lxs, &lys);
    let base = exp_slope.exp();
    if base < 1.25 || poly_r2 > exp_r2 + 1e-9 {
        Law::Polynomial { degree: poly_slope }
    } else {
        Law::Exponential { base }
    }
}

/// One representation's measured probe on one circuit.
#[derive(Debug, Clone)]
pub struct AxisProbe {
    /// Representation name.
    pub axis: &'static str,
    /// Measured cost in bytes (`None` = the representation hit a wall
    /// — a measured refusal, timeout, or unsupported operation).
    pub cost: Option<usize>,
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
    F: FnOnce(usize) -> Result<S>,
    P: FnOnce(&S) -> (String, bool),
{
    // Every probe runs under a scoped wall-clock budget: an axis whose
    // representation cannot finish in bounded time reports a measured
    // timeout as its wall instead of stalling the whole profile.
    let run = || -> Result<S> {
        crate::guard::with_time_budget(std::time::Duration::from_secs(10), || {
            let mut state = make(circuit.num_qubits())?;
            circuit.bind(reg)?.run(&mut state)?;
            Ok(state)
        })
    };
    match run() {
        Ok(state) => {
            let (parameter, exact) = param(&state);
            AxisProbe {
                axis,
                cost: Some(state.memory_bytes()),
                parameter,
                exact,
                note: None,
            }
        }
        Err(e) => AxisProbe {
            axis,
            cost: None,
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
    axes.push(probe(
        "clifford-framed",
        CliffordFramedState::<C64>::new,
        circuit,
        &reg,
        |s: &CliffordFramedState<C64>| {
            (format!("stored support {}", s.peak_stored_support()), true)
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

/// One axis of a family scan: name, measured costs per size (walls as
/// `None`), the classified law over the measured points, and whether
/// every run was exact.
#[derive(Debug, Clone)]
pub struct AxisScan {
    /// Representation name.
    pub axis: &'static str,
    /// Measured costs per family size.
    pub costs: Vec<Option<usize>>,
    /// Classified law (`None` when any size walled — no clean fit).
    pub law: Option<Law>,
    /// Every probe exact.
    pub exact: bool,
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
        let exact = profiles.iter().all(|p| p.axes[a].exact);
        let law = if costs.iter().all(|c| c.is_some()) {
            let cs: Vec<usize> = costs.iter().map(|c| c.unwrap()).collect();
            Some(classify_law(sizes, &cs))
        } else {
            None
        };
        axes.push(AxisScan {
            axis: profiles[0].axes[a].axis,
            costs,
            law,
            exact,
        });
    }
    let via: Vec<String> = axes
        .iter()
        .filter(|a| a.exact && a.law.as_ref().is_some_and(|l| l.is_subexponential()))
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
