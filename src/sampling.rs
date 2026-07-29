//! Sampling-task hardness: the advantage claims are about *sampling*,
//! and this module measures that task directly.
//!
//! The score is the linear cross-entropy benchmark (XEB): for samples
//! `x₁…x_S` scored against ideal probabilities `p(x)`,
//! `raw = 2^n · mean(p(xᵢ)) − 1`. A uniform sampler scores ≈ 0; the
//! ideal sampler's own expectation is the *ceiling*
//! `2^n · Σ_x p(x)² − 1` (≈ 1 for Porter–Thomas outputs, very
//! different for structured families — the reason scores here are
//! reported normalized by the measured ceiling, not against an assumed
//! 1). The reference probabilities come from the dense state, or —
//! on the Clifford+T fragment — from the exact D\[ω\] evaluator, so the
//! task can be scored against absolute values with no float in the
//! reference path.
//!
//! Three samplers close the loop with the boundary atlas:
//!
//! * any [`Backend`]'s own Born sampling (`sample`) — exact whenever
//!   the representation is, so a family certified classical by an axis
//!   is *sampled* at that axis's cost;
//! * [`clifford_sample`] — per-shot native tableau measurement, the
//!   Gottesman–Knill sampling story: polynomial cost per shot on
//!   Clifford(+light doping) circuits, no state-vector flush;
//! * [`mps_spoof_curve`] — the real-world spoofing strategy: a
//!   truncated tensor network buys XEB fidelity with bond dimension,
//!   and the curve (χ vs score vs cost) is the measured price of
//!   faking the task. [`spoof_decay`] fixes χ and sweeps the family
//!   size: on candidate families the score decays toward 0 — the
//!   sampling task inherits the state bounds.

use crate::backend::{Backend, CliffordFramedState, MpsConfig, MpsState, PauliString};
use crate::circuit::Circuit;
use crate::error::Result;
use crate::exact::ExactState;
use crate::registry::GateRegistry;
use crate::rng::Prng;
use crate::scalar::C64;
use crate::sim::Simulator;
use std::collections::HashMap;

/// `2^n · mean(p_ideal(x)) − 1` over the sampled bitstrings, with the
/// ideal probabilities read from `ideal`.
pub fn linear_xeb(ideal: &dyn Backend<C64>, samples: &HashMap<u64, u64>) -> f64 {
    let n = ideal.num_qubits();
    let shots: u64 = samples.values().sum();
    if shots == 0 {
        return 0.0;
    }
    let mean_p: f64 = samples
        .iter()
        .map(|(&x, &c)| ideal.probability(x) * c as f64)
        .sum::<f64>()
        / shots as f64;
    (1u64 << n) as f64 * mean_p - 1.0
}

/// The ideal sampler's expected score — the ceiling scores are
/// normalized by: `2^n · Σ_x p(x)² − 1`. Zero for a uniform output
/// distribution (no XEB signal exists there), ≈ 1 for Porter–Thomas.
pub fn ideal_xeb(ideal: &dyn Backend<C64>) -> f64 {
    let n = ideal.num_qubits();
    let mut sum_p2 = 0.0;
    ideal.for_each_nonzero(&mut |_, amp| {
        let p = amp.norm_sqr();
        sum_p2 += p * p;
    });
    (1u64 << n) as f64 * sum_p2 - 1.0
}

/// A sampler's XEB against a reference, raw and normalized.
#[derive(Debug, Clone)]
pub struct XebScore {
    /// `2^n · mean(p_ideal) − 1` over the samples.
    pub raw: f64,
    /// The ideal sampler's own expected score on this circuit.
    pub ceiling: f64,
    /// `raw / ceiling` — the fidelity proxy (`None` when the ceiling
    /// vanishes: a uniform output distribution carries no XEB signal).
    pub normalized: Option<f64>,
    /// Shots scored.
    pub shots: u64,
}

/// Score `samples` against `ideal`.
pub fn score_samples(ideal: &dyn Backend<C64>, samples: &HashMap<u64, u64>) -> XebScore {
    let raw = linear_xeb(ideal, samples);
    let ceiling = ideal_xeb(ideal);
    let normalized = if ceiling.abs() > 1e-9 {
        Some(raw / ceiling)
    } else {
        None
    };
    XebScore {
        raw,
        ceiling,
        normalized,
        shots: samples.values().sum(),
    }
}

/// XEB with the reference probabilities taken from the exact D\[ω\]
/// evaluator — no float in the reference path. Only the Clifford+T
/// fragment evaluates exactly; other gates error.
pub fn linear_xeb_exact(exact: &ExactState, samples: &HashMap<u64, u64>) -> Result<f64> {
    let n = exact.num_qubits();
    let shots: u64 = samples.values().sum();
    if shots == 0 {
        return Ok(0.0);
    }
    let mut acc = 0.0;
    for (&x, &c) in samples {
        acc += exact.probability_exact(x)?.to_f64() * c as f64;
    }
    Ok((1u64 << n) as f64 * (acc / shots as f64) - 1.0)
}

/// Sample a circuit by running it on the Clifford frame once per shot
/// and measuring every qubit natively through the tableau
/// (`measure_pauli`, no state-vector flush): the Gottesman–Knill
/// sampling path, polynomial per shot while the doped support stays
/// small.
pub fn clifford_sample(
    circuit: &Circuit,
    registry: &GateRegistry<C64>,
    shots: u64,
    rng: &mut Prng,
) -> Result<HashMap<u64, u64>> {
    let n = circuit.num_qubits();
    let bound = circuit.bind(registry)?;
    let mut counts: HashMap<u64, u64> = HashMap::new();
    for _ in 0..shots {
        let mut state = CliffordFramedState::<C64>::new(n)?;
        bound.run(&mut state)?;
        let mut index = 0u64;
        for q in 0..n {
            let one = state.measure_pauli(
                PauliString {
                    x: 0,
                    z: 1 << q,
                    negative: false,
                },
                rng,
            )?;
            if one {
                index |= 1 << q;
            }
        }
        *counts.entry(index).or_insert(0) += 1;
    }
    Ok(counts)
}

/// One point of a truncated-MPS spoofing curve.
#[derive(Debug, Clone)]
pub struct SpoofPoint {
    /// The bond cap the spoofer paid for.
    pub max_bond: usize,
    /// The peak bond it actually reached.
    pub peak_bond: usize,
    /// Its memory bill.
    pub memory: usize,
    /// Wall-clock nanoseconds for run + sampling.
    pub nanos: u128,
    /// The score its samples earned against the ideal reference.
    pub score: XebScore,
}

/// The measured price of faking the sampling task: run `circuit` on
/// MPS at each bond cap (truncating — deliberately inexact), sample
/// with the native conditional sweep, and score against the dense
/// ideal.
pub fn mps_spoof_curve(
    circuit: &Circuit,
    registry: &GateRegistry<C64>,
    bond_caps: &[usize],
    shots: u64,
    seed: u64,
) -> Result<Vec<SpoofPoint>> {
    let ideal = Simulator::<C64>::new().run(circuit)?;
    let bound = circuit.bind(registry)?;
    let mut points = Vec::new();
    for &chi in bond_caps {
        let started = std::time::Instant::now();
        let mut mps = MpsState::<C64>::with_config(
            circuit.num_qubits(),
            MpsConfig {
                max_bond: chi,
                trunc_tol: 1e-12,
            },
        )?;
        bound.run(&mut mps)?;
        let samples = mps.sample_mps(shots, &mut Prng::new(seed))?;
        let nanos = started.elapsed().as_nanos();
        points.push(SpoofPoint {
            max_bond: chi,
            peak_bond: mps.peak().0,
            memory: mps.memory_bytes(),
            nanos,
            score: score_samples(ideal.as_ref(), &samples),
        });
    }
    Ok(points)
}

/// One point of a fixed-χ fidelity-decay sweep.
#[derive(Debug, Clone)]
pub struct DecayPoint {
    /// Family size.
    pub size: usize,
    /// The fixed-χ spoofer's normalized score at this size.
    pub normalized: Option<f64>,
}

/// Fix the spoofer's bond cap and sweep the family size: on candidate
/// families the normalized score decays toward zero — the sampling
/// task inherits the state bounds.
pub fn spoof_decay(
    family: impl Fn(usize) -> Circuit,
    sizes: &[usize],
    bond_cap: usize,
    shots: u64,
    seed: u64,
) -> Result<Vec<DecayPoint>> {
    let registry = GateRegistry::<C64>::standard();
    let mut out = Vec::new();
    for &n in sizes {
        let circuit = family(n);
        let points = mps_spoof_curve(&circuit, &registry, &[bond_cap], shots, seed)?;
        out.push(DecayPoint {
            size: n,
            normalized: points[0].score.normalized,
        });
    }
    Ok(out)
}
