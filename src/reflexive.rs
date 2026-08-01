//! Self-embedded evolution: a machine whose next gate is a function of
//! the state it is currently in.
//!
//! Every circuit in this crate is a **linear map fixed before it runs**.
//! That is what makes it compilable, and it is also what makes its cost
//! a property of the circuit rather than of the state. This module gives
//! up the first to get the second.
//!
//! A reflexive program is a loop of three moves:
//!
//! ```text
//! sense    W(ψ) → |·|² → W        the state's own harmonics, exactly
//! compute  θ = law(spectrum)      the second computation
//! actuate  W(θ) → diag e^{iφ}     one gate, built across its diagonal
//! ```
//!
//! and the sensor is `H^⊗n` — the second computation runs *on the same
//! substrate as the first*, which is what "self-embedded" means here
//! rather than metaphorically. The actuator is the inverse of the same
//! transform, so the loop's two halves are one operation used in both
//! directions.
//!
//! ## Why it cannot be pre-compiled
//!
//! Because `θ` is read off `ψ`, the emitted gate sequence is a function
//! of the initial state. [`replay`] takes the gates a run actually
//! emitted and applies them as an ordinary circuit; starting that
//! circuit from a *different* state gives a different answer than
//! running the program from that state. The program is therefore not
//! equal to any circuit, and
//! `a_recorded_run_is_not_a_circuit_and_the_difference_is_measured`
//! measures the gap instead of asserting it.
//!
//! This is a real physical model, not a loophole: it is measurement-fed
//! adaptive control — sense the register, compute a drive, apply it —
//! and the resulting evolution is **nonlinear in the state**, as
//! mean-field and feedback-steered dynamics are. It is not unitary
//! quantum mechanics, and nothing here should be read as simulating
//! unitary quantum mechanics more cheaply.
//!
//! ## What the loop's cost tracks
//!
//! The drive's support is the **fired difference set** of
//! [`CrossView`], which is exactly the set
//! of harmonics the state actually uses. Two facts make that a usable
//! cost model rather than a heuristic:
//!
//! * an unfired difference contributes *exactly* zero, proved rather
//!   than thresholded;
//! * the fired set is **invariant under the drive** — the mass depends
//!   only on amplitude moduli, and a diagonal gate does not move those —
//!   so the loop cannot inflate its own cost. Only the mixing layers
//!   between events can, and those are the part the program did not
//!   choose.
//!
//! ## Second order
//!
//! [`Proportional`] is the first-order law: correct the error along the
//! harmonics the state occupies. [`Adaptive`] wraps it and steers its
//! *gain* from whether the error actually fell — a controller over the
//! controller, which is the second loop. Whether that helps is measured
//! in `examples/reflexive_loop.rs`, not assumed here.

use crate::backend::{Backend, DenseState};
use crate::circuit::{BoundGate, GateKernel};
use crate::crossview::{fwht, CrossView};
use crate::error::{Error, Result};
use crate::scalar::C64;

/// What one event of the loop emitted.
#[derive(Clone, Debug)]
pub struct Emission {
    /// Event index.
    pub event: usize,
    /// The phase coefficients on the `Z`-harmonics, `(S, θ_S)`.
    pub coefficients: Vec<(u64, f64)>,
    /// Differences the state fired when this was computed — the drive's
    /// support, and the loop's cost for this event.
    pub fired: usize,
    /// The error the law was correcting when this was emitted.
    pub error: f64,
    /// Gain in force when this was emitted, for reading a second-order
    /// run back.
    pub gain: f64,
}

impl Emission {
    /// The `2^n` diagonal this emission stands for.
    ///
    /// One transform: the coefficients live on the harmonics and the
    /// gate lives on the basis, and `W` is the map between them. This is
    /// the sense in which the gate is built *across its diagonal* rather
    /// than gate by gate.
    pub fn diagonal(&self, qubits: usize) -> Vec<C64> {
        let n = 1usize << qubits;
        let mut spectrum = vec![0.0f64; n];
        for &(s, theta) in &self.coefficients {
            spectrum[s as usize % n] += theta;
        }
        fwht(&mut spectrum);
        spectrum
            .into_iter()
            .map(|phi| C64::new(phi.cos(), phi.sin()))
            .collect()
    }
}

/// An observable held at a value: what the loop is steering *to*.
///
/// A controller without a setpoint is not a controller, only a
/// disturbance — so the law is defined against a target rather than
/// against an open-ended "objective".
#[derive(Clone, Copy, Debug)]
pub struct Target {
    /// `X`-support of the steered Pauli string.
    pub a: u64,
    /// `Z`-support of the steered Pauli string.
    pub b: u64,
    /// The value it is being held at.
    pub value: f64,
}

impl Target {
    /// Hold `⟨X_A⟩` at `value`.
    pub fn x(a: u64, value: f64) -> Target {
        Target { a, b: 0, value }
    }

    /// The string's present value.
    pub fn read(&self, view: &CrossView) -> f64 {
        if self.b == 0 {
            view.x_string(self.a)
        } else {
            view.mixed(self.a, self.b)
        }
    }
}

/// The rule that turns a sensed spectrum into a drive.
pub trait DriveLaw {
    /// Name, for reports.
    fn name(&self) -> &str;

    /// What is being steered.
    fn target(&self) -> Target;

    /// Signed error: how far the steered string is from its setpoint.
    fn error(&self, view: &CrossView) -> f64 {
        let t = self.target();
        t.value - t.read(view)
    }

    /// Phase coefficients on the `Z`-harmonics for this event.
    fn phases(&mut self, view: &CrossView, fired: &[u64], event: usize) -> Vec<(u64, f64)>;

    /// Gain currently in force, for the trace.
    fn gain(&self) -> f64;
}

/// First order: proportional control, driven through the state's own
/// harmonics.
///
/// The drive on harmonic `S` is `gain · error · ⟨X_S⟩`, so the
/// correction is applied along the directions the state actually
/// occupies rather than a fixed axis — the sensed spectrum is both the
/// error signal and the actuator basis. `⟨X_S⟩` is not known until the
/// state arrives, which is exactly why this is not a gate.
#[derive(Clone, Debug)]
pub struct Proportional {
    /// What is being held, and where.
    pub target: Target,
    /// Drive strength.
    pub gain: f64,
    /// Drive only the steered harmonic instead of the whole fired set.
    pub narrow: bool,
    name: String,
}

impl Proportional {
    /// Proportional control at the given gain.
    pub fn new(target: Target, gain: f64) -> Proportional {
        Proportional {
            target,
            gain,
            narrow: false,
            name: "proportional".into(),
        }
    }

    /// Drive only the harmonic being steered.
    pub fn narrow(mut self) -> Proportional {
        self.narrow = true;
        self.name = "proportional (narrow)".into();
        self
    }
}

impl DriveLaw for Proportional {
    fn name(&self) -> &str {
        &self.name
    }

    fn target(&self) -> Target {
        self.target
    }

    fn phases(&mut self, view: &CrossView, fired: &[u64], _event: usize) -> Vec<(u64, f64)> {
        let err = self.error(view);
        if self.narrow {
            return vec![(self.target.a, self.gain * err)];
        }
        let norm: f64 = fired.iter().map(|&a| view.x_string(a).abs()).sum();
        if norm <= f64::EPSILON {
            return Vec::new();
        }
        fired
            .iter()
            .filter(|&&a| a != 0)
            .map(|&a| (a, self.gain * err * view.x_string(a) / norm))
            .collect()
    }

    fn gain(&self) -> f64 {
        self.gain
    }
}

/// Gradient control: drive along the target's actual derivative.
///
/// Under a diagonal drive `φ(x) = Σ_S θ_S(−1)^{|x∧S|}` the phase
/// difference across a difference `A` is
/// `−2 Σ_{S: |A∧S| odd} θ_S(−1)^{|x∧S|}`, so
///
/// ```text
/// d⟨X_A⟩/dθ_S = ±2·⟨X_A Z_S⟩     for |A∧S| odd,   0 otherwise
/// ```
///
/// — **the gradient of a pure string with respect to the drive is the
/// mixed string**, and the mixed strings for a fixed `A` are exactly one
/// [`CrossView::drill`](crate::crossview::CrossView::drill). So the
/// sensor and the Jacobian are the same transform, and the second
/// computation is not an approximation of a derivative but the
/// derivative itself.
///
/// `the_drill_is_the_jacobian_of_the_drive` checks this against finite
/// differences rather than trusting the derivation.
#[derive(Clone, Debug)]
pub struct Gradient {
    /// What is being held, and where.
    pub target: Target,
    /// Step size along the gradient.
    pub gain: f64,
    /// Coefficients below this fraction of the largest are dropped —
    /// the drive's support is then the state's realized structure.
    pub floor: f64,
    name: String,
}

impl Gradient {
    /// Gradient control at the given step size.
    pub fn new(target: Target, gain: f64) -> Gradient {
        Gradient {
            target,
            gain,
            floor: 1e-12,
            name: "gradient".into(),
        }
    }

    /// `d⟨X_A⟩/dθ_S` for every `S`, from one drill.
    pub fn jacobian(&self, view: &CrossView) -> Vec<f64> {
        let d = view.drill(self.target.a);
        (0..d.len())
            .map(|s| match (self.target.a & s as u64).count_ones() % 4 {
                1 => 2.0 * d[s],
                3 => -2.0 * d[s],
                _ => 0.0,
            })
            .collect()
    }
}

impl DriveLaw for Gradient {
    fn name(&self) -> &str {
        &self.name
    }

    fn target(&self) -> Target {
        self.target
    }

    fn phases(&mut self, view: &CrossView, _fired: &[u64], _event: usize) -> Vec<(u64, f64)> {
        let err = self.error(view);
        let grad = self.jacobian(view);
        let norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if norm <= f64::EPSILON {
            return Vec::new();
        }
        let cut = self.floor * grad.iter().fold(0.0f64, |m, g| m.max(g.abs()));
        grad.iter()
            .enumerate()
            .filter(|(_, g)| g.abs() > cut)
            .map(|(s, g)| (s as u64, self.gain * err * g / norm))
            .collect()
    }

    fn gain(&self) -> f64 {
        self.gain
    }
}

/// Second order: a controller over the controller.
///
/// The inner law reads the state. This reads *the inner law's effect on
/// the state* — whether the error it is correcting actually shrank —
/// and moves the gain accordingly: creep up while the error is falling,
/// cut hard when it rises. That is the second loop, and it is the part
/// that genuinely cannot be scheduled ahead of time, because it depends
/// on a difference between two states neither of which exists yet.
///
/// The measured reason it exists: first-order proportional control on
/// this system is unstable above a gain that depends on the state, and
/// the state is not known when the gain would have to be chosen.
#[derive(Clone, Debug)]
pub struct Adaptive {
    inner: Gradient,
    /// How fast the gain creeps up while the error is falling.
    pub climb: f64,
    /// How hard the gain is cut when the error rises.
    pub cut: f64,
    /// Gain the controller will not exceed.
    pub ceiling: f64,
    /// Gain the controller will not fall below.
    ///
    /// Without one the loop kills itself: gradient descent through a
    /// scrambling layer legitimately sees the error rise on some events,
    /// and a rule that only ever cuts on a rise drives the gain to zero
    /// and then cannot recover. Measured — the first version of this
    /// controller did exactly that.
    pub floor: f64,
    /// Consecutive rises tolerated before the gain is cut.
    ///
    /// One is too few. Descent through a mixing layer is a noisy
    /// signal, and reacting to every single rise pins the gain at the
    /// floor — measured, and the reason this field exists.
    pub patience: usize,
    rises: usize,
    last_error: Option<f64>,
    name: String,
}

impl Adaptive {
    /// Wrap a gradient law with gain adaptation.
    pub fn new(inner: Gradient, climb: f64, cut: f64, ceiling: f64) -> Adaptive {
        let floor = inner.gain * 0.05;
        Adaptive {
            inner,
            climb,
            cut,
            ceiling,
            floor,
            patience: 3,
            rises: 0,
            last_error: None,
            name: "adaptive".into(),
        }
    }

    /// Set how many consecutive rises are tolerated before cutting.
    pub fn with_patience(mut self, patience: usize) -> Adaptive {
        self.patience = patience;
        self
    }

    /// Set the gain floor explicitly.
    pub fn with_floor(mut self, floor: f64) -> Adaptive {
        self.floor = floor;
        self
    }
}

impl DriveLaw for Adaptive {
    fn name(&self) -> &str {
        &self.name
    }

    fn target(&self) -> Target {
        self.inner.target
    }

    fn phases(&mut self, view: &CrossView, fired: &[u64], event: usize) -> Vec<(u64, f64)> {
        let err = self.error(view).abs();
        if let Some(prev) = self.last_error {
            if err <= prev {
                self.rises = 0;
                self.inner.gain = (self.inner.gain * (1.0 + self.climb)).min(self.ceiling);
            } else {
                self.rises += 1;
                if self.rises > self.patience {
                    self.rises = 0;
                    self.inner.gain = (self.inner.gain * self.cut).max(self.floor);
                }
            }
        }
        self.last_error = Some(err);
        self.inner.phases(view, fired, event)
    }

    fn gain(&self) -> f64 {
        self.inner.gain
    }
}

/// What a reflexive run did.
#[derive(Clone, Debug)]
pub struct Run {
    /// Gates the program emitted, in order — the record that
    /// [`replay`] turns back into an ordinary circuit.
    pub trace: Vec<Emission>,
    /// Signed error before each event's drive.
    pub error: Vec<f64>,
    /// Fired differences at each event — the loop's cost profile.
    pub fired: Vec<usize>,
    /// Transforms performed across the whole run.
    pub transforms: usize,
}

impl Run {
    /// The largest drive support the run ever needed, against the `2^n`
    /// it could have.
    pub fn peak_support(&self) -> usize {
        self.fired.iter().copied().max().unwrap_or(0)
    }

    /// Absolute error at the end of the run.
    pub fn final_error(&self) -> f64 {
        self.error.last().copied().unwrap_or(0.0).abs()
    }

    /// Smallest absolute error the run ever reached.
    pub fn best_error(&self) -> f64 {
        self.error
            .iter()
            .map(|e| e.abs())
            .fold(f64::INFINITY, f64::min)
    }

    /// First event at which the absolute error stayed below `tol` for
    /// the rest of the run — `None` if it never settled.
    pub fn settled_at(&self, tol: f64) -> Option<usize> {
        (0..self.error.len())
            .find(|&i| self.error[i..].iter().all(|e| e.abs() <= tol))
    }
}

/// A state carried through a reflexive program.
///
/// Holds a dense register because the sensor needs amplitudes; this is
/// an `O(2^n)` object and makes no claim otherwise. What it buys is that
/// the *drive* costs the state's structure and the parameterization is
/// exact.
pub struct Reflexive {
    qubits: usize,
    state: DenseState<C64>,
    transforms: usize,
}

impl Reflexive {
    /// Start from `|0…0⟩`.
    pub fn new(qubits: usize) -> Result<Reflexive> {
        Ok(Reflexive {
            qubits,
            state: DenseState::<C64>::new(qubits)?,
            transforms: 0,
        })
    }

    /// Start from a prepared register.
    pub fn from_state(state: DenseState<C64>) -> Reflexive {
        Reflexive {
            qubits: state.num_qubits(),
            state,
            transforms: 0,
        }
    }

    /// Register width.
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    /// The register, for inspection.
    pub fn state(&self) -> &DenseState<C64> {
        &self.state
    }

    /// Sense the current state.
    pub fn sense(&mut self) -> Result<CrossView> {
        let view = CrossView::of(&self.state as &dyn Backend<C64>)?;
        self.transforms += view.transforms();
        Ok(view)
    }

    /// Apply a mixing layer — the part of the program that is an
    /// ordinary circuit, and the only part that can grow the drive's
    /// support.
    pub fn mix(&mut self, layer: &[BoundGate<C64>]) -> Result<()> {
        for g in layer {
            match &g.kernel {
                GateKernel::Matrix(m) => self.state.apply(m, &g.qubits)?,
                GateKernel::Diagonal(d) => self.state.apply_diagonal(d, &g.qubits)?,
            }
        }
        Ok(())
    }

    /// Apply a recorded emission, without sensing — the actuator alone.
    pub fn apply(&mut self, emission: &Emission) -> Result<()> {
        let all: Vec<usize> = (0..self.qubits).collect();
        self.transforms += 1;
        self.state.apply_diagonal(&emission.diagonal(self.qubits), &all)
    }

    /// One event: sense, compute, actuate.
    pub fn event(&mut self, law: &mut dyn DriveLaw, event: usize) -> Result<Emission> {
        let view = self.sense()?;
        let fired = view.fired(1e-12);
        let error = law.error(&view);
        let gain = law.gain();
        let coefficients = law.phases(&view, &fired, event);
        let emission = Emission {
            event,
            coefficients,
            fired: fired.len(),
            error,
            gain,
        };
        let diag = emission.diagonal(self.qubits);
        self.transforms += 1;
        let all: Vec<usize> = (0..self.qubits).collect();
        self.state.apply_diagonal(&diag, &all)?;
        Ok(emission)
    }

    /// Run `events` iterations, applying `layer` between them.
    pub fn run(
        &mut self,
        law: &mut dyn DriveLaw,
        layer: &[BoundGate<C64>],
        events: usize,
    ) -> Result<Run> {
        let mut trace = Vec::with_capacity(events);
        let mut error = Vec::with_capacity(events);
        let mut fired = Vec::with_capacity(events);
        for e in 0..events {
            let em = self.event(law, e)?;
            error.push(em.error);
            fired.push(em.fired);
            trace.push(em);
            self.mix(layer)?;
        }
        Ok(Run {
            trace,
            error,
            fired,
            transforms: self.transforms,
        })
    }
}

/// Replay a recorded run as an ordinary circuit.
///
/// The emissions become fixed diagonal gates and the mixing layer is
/// unchanged, so this *is* a compiled circuit — a linear map, decided
/// before it runs. Starting it from the state the run started in
/// reproduces the run exactly; starting it from any other state does
/// not, which is the whole point.
pub fn replay(
    trace: &[Emission],
    layer: &[BoundGate<C64>],
    initial: DenseState<C64>,
) -> Result<DenseState<C64>> {
    let qubits = initial.num_qubits();
    let all: Vec<usize> = (0..qubits).collect();
    let mut state = initial;
    for em in trace {
        state.apply_diagonal(&em.diagonal(qubits), &all)?;
        for g in layer {
            match &g.kernel {
                GateKernel::Matrix(m) => state.apply(m, &g.qubits)?,
                GateKernel::Diagonal(d) => state.apply_diagonal(d, &g.qubits)?,
            }
        }
    }
    Ok(state)
}

/// How far two registers differ, amplitude by amplitude.
pub fn deviation(a: &DenseState<C64>, b: &DenseState<C64>) -> Result<f64> {
    if a.num_qubits() != b.num_qubits() {
        return Err(Error::InvalidState("reflexive: width mismatch".into()));
    }
    Ok((0..1u64 << a.num_qubits())
        .map(|i| {
            let (x, y) = (a.amplitude(i), b.amplitude(i));
            ((x.re - y.re).powi(2) + (x.im - y.im).powi(2)).sqrt()
        })
        .fold(0.0f64, f64::max))
}
