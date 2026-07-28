//! # quantsim
//!
//! An in-house research quantum simulator built around two swappable axes:
//!
//! 1. **The amplitude algebra** ([`scalar::Scalar`]): ℝ, ℂ (default),
//!    quaternions ℍ, octonions 𝕆, sedenions — via a generic Cayley–Dickson
//!    construction — plus non-Cayley–Dickson algebras like split-complex
//!    and [`scalar::Ball`] (certified midpoint ± radius arithmetic).
//!    Every gate matrix, state representation and measurement rule is
//!    generic over it. The [`exact`] module supplies float-free absolute
//!    reference values for the Clifford+T fragment.
//! 2. **The state representation** ([`backend::Backend`]): dense state
//!    vector (the BQP reference), sparse hash-map state, adaptive
//!    sparse→dense promotion, factored entanglement clusters, matrix
//!    product states, and the hierarchical `mera` tree with coarse views
//!    at every scale; further representations register by name.
//!
//! Gates live in a [`registry::GateRegistry`] — research gates are
//! first-class: implement [`gates::GateDef`] or hand the registry a closure,
//! and circuits can use the new name immediately, with unitarity validated
//! at registration.
//!
//! ## Quick start
//!
//! ```
//! use quantsim::prelude::*;
//!
//! // Bell pair on the default complex-amplitude dense backend.
//! let mut c = Circuit::new(2);
//! c.h(0).cx(0, 1);
//!
//! let sim: Simulator = Simulator::new();
//! let state = sim.run(&c)?;
//!
//! let amp = state.amplitude(0b11);
//! assert!((amp.re - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
//!
//! // Deterministic sampling.
//! let counts = state.sample(1000, &mut Prng::new(7))?;
//! assert_eq!(counts.get(&0b01), None); // only |00⟩ and |11⟩ appear
//! # Ok::<(), quantsim::Error>(())
//! ```
//!
//! ## Conventions
//!
//! * **Little-endian**: qubit 0 is the least significant bit of a basis
//!   index (Qiskit-style). `Circuit::cx(c, t)` has its control at `c`.
//! * Multi-qubit matrices: sub-index bit `b` corresponds to `qubits[b]`;
//!   controlled standard gates carry the control at `qubits[0]`.
//! * Rotations use half angles: `rx/ry/rz(θ) = exp(−iθP/2)`.
//! * States are left modules: gates multiply amplitudes from the left
//!   (matters for ℍ and beyond); inner products conjugate the left factor.
//!
//! ## BQP
//!
//! The standard registry contains a universal gate set (`h`, `t`, `cx`, and
//! friends), so any BQP circuit family can be expressed and simulated
//! exactly on the dense backend — at `O(2^n)` memory in the width `n`,
//! which is precisely what the width benchmarks (`benches/width.rs`,
//! `examples/width_scaling.rs`) measure.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod backend;
pub mod bounds;
pub mod causal;
pub mod circuit;
pub mod conformance;
pub mod discovery;
pub mod e8;
pub mod error;
pub mod exact;
pub mod gates;
pub mod guard;
pub mod harness;
pub mod library;
pub mod lift;
pub mod math;
pub mod mixed;
pub mod qudit;
pub mod registry;
pub mod rng;
pub mod sampling;
pub mod scalar;
pub mod schedule;
pub mod sim;

pub use backend::{AdaptiveState, Backend, BackendRegistry, DenseState, SparseState};
pub use circuit::{BoundCircuit, BoundGate, Circuit, GateKernel, Op};
pub use error::{Error, Result};
pub use math::GateMatrix;
pub use registry::GateRegistry;
pub use rng::Prng;
pub use scalar::{Scalar, C64};
pub use sim::Simulator;

/// One-stop imports for typical use.
pub mod prelude {
    pub use crate::backend::{
        max_amplitude_deviation, pauli_expectation, AdaptiveState, ArityPolicy, Backend,
        BackendRegistry, CliffordFrameStats, CliffordFramedState, DenseState, DeviceState,
        DurationModel, FactoredState, FrameStats, FramedState, InterferenceRecord,
        InterferenceState, LatencyMap, MeraConfig, MeraState, MpsConfig, MpsState, PauliString,
        PhysicalOp, SparseState, Topology,
    };
    pub use crate::bounds::{
        advantage_scan, classify_law, fit_law, resource_profile, select_by_scaling, AxisProbe,
        AxisScan, FamilyScan, Law, LawFit, ResourceProfile, ScalingChoice, ScalingSelection,
        Verdict,
    };
    pub use crate::causal::{
        backward_cone, causal_diamond, dual_time_amplitude, DiamondReport, DualTimeResolution,
    };
    pub use crate::circuit::{BoundCircuit, BoundGate, Circuit, GateKernel, Op};
    pub use crate::conformance::{
        random_registry_circuit, verify_backend, ConformanceConfig, ConformanceReport,
    };
    pub use crate::discovery::{
        discover_stabilizers, stabilizes_state, state_deviation_up_to_phase, verify_transparent,
        Insertion, StabilizerCheck, TransparencyReport,
    };
    pub use crate::e8::{self, ChainEmbedding};
    pub use crate::error::{Error, Result};
    pub use crate::exact::{DOmega, ExactReal, ExactState};
    pub use crate::gates::{FixedGate, GateDef, ParamGate, Pauli};
    pub use crate::guard;
    pub use crate::harness::{
        compare_backends, select_backend, BenchConfig, BenchmarkReport, SelectionCriterion,
        SelectionReport, Workload,
    };
    pub use crate::library;
    pub use crate::lift::{self, LiftedCircuit, ResourcePrep};
    pub use crate::math::{c64, cis, GateMatrix};
    pub use crate::mixed::{
        clock_d, cshift, fabric, fourier_d, shift_d, swap_dd, CompoundRegister, FabricReport,
        Interaction,
    };
    pub use crate::qudit::{
        algebra_capacity, dual_algebra_report, synthesize_sandwich, AlgebraicRegister,
        DualAlgebraReport, QuditStats, SandwichOp, SandwichTerm,
    };
    pub use crate::registry::GateRegistry;
    pub use crate::rng::Prng;
    pub use crate::sampling::{
        clifford_sample, ideal_xeb, linear_xeb, linear_xeb_exact, mps_spoof_curve, score_samples,
        spoof_decay, DecayPoint, SpoofPoint, XebScore,
    };
    pub use crate::scalar::{
        Ball, CComplex, DirectSum, Octonion, Quaternion, Scalar, Sedenion, SplitComplex,
        Trigintaduonion, C64, CD,
    };
    pub use crate::schedule::{
        FeedbackOp, GateLoop, MeasureEvent, OverlapPolicy, Schedule, ScheduleTrace,
        ScheduledKernel, TimedOp,
    };
    pub use crate::sim::Simulator;
}
