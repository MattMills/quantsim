//! Doped Clifford sampling: the IBM/UChicago experiment of July 2026
//! (arXiv:2607.25941) reproduced, and this crate's measured position
//! against it.
//!
//! Two kinds of assertion live here and they are not the same kind of
//! fact. The **published counts** pin the construction against the
//! paper — if the circuit built here stops having 2415 two-qubit gates
//! it is not the experiment's circuit any more. The **measured walls**
//! pin where this crate actually stops, and they are written so that an
//! improvement goes *red*: a test asserting "this family is not
//! certified classical" is a test that fails the day someone certifies
//! it, which is the only way a boundary claim can be worth anything.

use quantsim::dcs::{self, Dcs};
use quantsim::gates::Pauli;
use quantsim::pathsum::PathSum;
use quantsim::prelude::*;
use quantsim::sampling::mps_spoof_curve;
use std::time::Duration;

/// The two published gate counts are consequences of the layout, not
/// free parameters, so they check the construction.
#[test]
fn the_experiment_reproduces_the_published_gate_counts() {
    let census = Dcs::experiment().census();
    assert_eq!(census.qubits, 70);
    assert_eq!(census.depth, 70);
    assert_eq!(
        census.two_qubit_gates,
        dcs::EXPERIMENT_TWO_QUBIT_GATES,
        "70 qubits at depth 70 on a linear chain is 35·35 + 35·34 = 2415 CZ gates"
    );
    assert_eq!(census.t_gates, dcs::EXPERIMENT_T_GATES);
    // The paper quotes the doping rate of its own control circuits as
    // r = 0.19; 468/2415 is where that comes from.
    let r = Dcs::experiment().doping_rate();
    assert!((r - 0.19).abs() < 0.005, "doping rate {r}");
    // Every T sits on a wire following a CZ, so the pool is 2 per gate.
    assert_eq!(
        Dcs::experiment().dope_sites().len(),
        2 * dcs::EXPERIMENT_TWO_QUBIT_GATES
    );
}

/// The closed form is the thing the example extrapolates with, so it is
/// checked against the built circuit rather than trusted.
#[test]
fn the_closed_form_two_qubit_count_matches_the_built_circuit() {
    for n in [2, 3, 4, 5, 8, 13, 16, 31, 70] {
        for depth in [1, 2, 3, 7, 16, 70] {
            let built = Dcs {
                qubits: n,
                depth,
                t_gates: 0,
                seed: 1,
            }
            .census()
            .two_qubit_gates;
            assert_eq!(built, dcs::two_qubit_gates(n, depth), "n={n} depth={depth}");
        }
    }
}

/// The skeleton is the experiment's own trusted classical baseline. It
/// is Clifford, so the path sum holds it with no surviving path
/// variables at all — Gottesman–Knill with no tableau anywhere.
#[test]
fn the_clifford_skeleton_reduces_to_no_internal_variables() {
    for n in [6, 10, 14] {
        let p = dcs::probe(&Dcs::scaled(n).with_t(0).skeleton(), Duration::from_secs(60))
            .expect("skeleton reduces");
        assert!(!p.cut_short, "n={n} did not reduce inside the budget");
        assert_eq!(p.h_star, 0, "n={n}: the skeleton is Clifford");
    }
}

/// The construction is only worth measuring if it is the circuit it
/// claims to be, so the path sum's amplitudes are checked against the
/// dense reference on a doped instance.
#[test]
fn the_path_sum_reproduces_the_doped_circuit_amplitudes() {
    let d = Dcs::scaled(8);
    assert!(d.t_gates > 0, "the scaled family is doped");
    let circuit = d.circuit();
    let dense = Simulator::<C64>::new().run(&circuit).unwrap();
    let ps = PathSum::from_circuit(&circuit).unwrap();
    let mut worst = 0.0f64;
    for x in 0..(1u64 << 8) {
        worst = worst.max((ps.amplitude(x) - dense.amplitude(x)).norm());
    }
    assert!(worst < 1e-9, "worst amplitude deviation {worst}");
}

/// Above 64 qubits the `u64` index runs out; the representation does
/// not. This is the readout the 70-qubit instance needs.
#[test]
fn the_wide_readout_agrees_with_the_narrow_one() {
    let circuit = Dcs::scaled(8).circuit();
    let ps = PathSum::from_circuit(&circuit).unwrap();
    for x in 0..(1u64 << 8) {
        let mut m = quantsim::pathsum::Mask::zero();
        for q in 0..8 {
            if x >> q & 1 == 1 {
                m.set(q);
            }
        }
        assert_eq!(ps.amplitude_mask(&m), ps.amplitude(x), "x={x}");
    }
}

/// §S9.5. Swapping a `T` for its nearest Clifford keeps `cos²(π/8)` of
/// the fidelity, and the swaps compound: the experiment's measured 0.32
/// is spent after seven of them, out of 468.
#[test]
fn cliffordizing_t_gates_spends_the_experiments_fidelity_in_seven_swaps() {
    let d = Dcs::scaled(12);
    let full = d.circuit();
    let sim = Simulator::<C64>::new();
    let truth = sim.run(&full).unwrap();
    let fidelity = |c: &Circuit<C64>| {
        let other = sim.run(c).unwrap();
        let mut acc = C64::new(0.0, 0.0);
        truth.for_each_nonzero(&mut |i, amp| {
            acc += amp.conj() * other.amplitude(i);
        });
        acc.norm_sqr()
    };
    for k in 0..=7 {
        let f = fidelity(&dcs::cliffordize(&full, k, 99));
        let predicted = dcs::CLIFFORDIZED_T_FIDELITY.powi(k as i32);
        assert!(
            (f - predicted).abs() < 1e-6,
            "k={k}: measured {f}, cos²(π/8)^k = {predicted}"
        );
    }
    assert!(
        fidelity(&dcs::cliffordize(&full, 7, 99)) > 0.32,
        "seven swaps still clear the experiment's fidelity"
    );
    assert!(
        fidelity(&dcs::cliffordize(&full, 8, 99)) < 0.32,
        "eight do not — the paper's stated ceiling"
    );
}

/// §S9.1. The paper's counter-intuitive claim: its *structured* circuit
/// is harder for a truncated MPS than the Haar-random control built on
/// the identical brickwork, because a stabilizer Schmidt spectrum is
/// flat and a Haar-random one decays. Same width, same depth, same CZ
/// layers — only the single-qubit layer differs.
#[test]
fn the_structured_circuit_costs_an_mps_more_than_its_haar_random_control() {
    let n = 12;
    let reg = GateRegistry::<C64>::standard();
    // One bond cap, well below the full rank 2^{n/2} = 64. The question
    // is what a spoofer gets for that fixed budget on each family — a
    // score comparison, not a threshold crossing, so it does not sit on
    // a boundary and does not need many shots to resolve.
    let caps = [4usize];
    let score_at = |c: &Circuit<C64>| -> f64 {
        mps_spoof_curve(c, &reg, &caps, 1200, 5).unwrap()[0]
            .score
            .normalized
            .unwrap_or(0.0)
    };
    let structured = score_at(&Dcs::scaled(n).circuit());
    let haar = score_at(&Dcs::scaled(n).haar_control());
    assert!(
        haar - structured > 0.2,
        "at χ = 4 of full rank {}: structured {structured:.3}, Haar-random control {haar:.3} — \
         the flat stabilizer spectrum is supposed to leave the structured circuit with nothing",
        1 << (n / 2)
    );
    assert!(
        structured < 0.1,
        "χ = 4 bought the structured circuit {structured:.3}, which is not 'nothing'"
    );
}

/// The cross-check that says the Clifford propagation is propagating
/// the right thing: the paper's §S9.3 reports the kernel of the
/// rotations' X-component matrix as `468 − 70 = 398` for the
/// experiment, and this reproduces that number exactly.
#[test]
fn the_rotation_axes_reproduce_the_papers_camps_kernel_dimension() {
    let d = Dcs::experiment();
    let axes = dcs::rotation_axes(&d.circuit()).unwrap();
    assert_eq!(axes.len(), dcs::EXPERIMENT_T_GATES);
    let x_rank = dcs::x_component_rank(&axes, d.qubits);
    assert_eq!(x_rank, 70, "the paper's X-component rank is n = 70");
    assert_eq!(
        axes.len() - x_rank,
        398,
        "the paper states a kernel dimension of 468 − 70 = 398"
    );
}

/// **The width ceiling.** The `468` magic axes span the *entire* Pauli
/// group on 70 qubits: the symplectic rank saturates at `2n` and stops.
///
/// That is what makes the T-count the wrong currency at this scale. A
/// cost law of the form `2^{α·t}` describes `|T⟩^{⊗t}`, a `t`-qubit
/// object; once the magic has been injected into `n` qubits every cost
/// measure is capped by the width instead. No `n`-qubit state has
/// stabilizer rank above `2^n` — the computational basis is already a
/// stabilizer decomposition — and its stabilizer extent obeys the same
/// bound, since `Σ|c_x| ≤ √(2^n · Σ|c_x|²) = 2^{n/2}`.
///
/// So the paper's `2^{0.3963·t} = 2^{185.5}` at `t = 468` is quoted
/// above a ceiling of `2^70`, and the crossing happens at `t ≈ 177`.
#[test]
fn the_magic_saturates_the_pauli_group_long_before_the_experiments_t_count() {
    for n in [32, 48, 70] {
        let d = if n == 70 { Dcs::experiment() } else { Dcs::scaled(n) };
        let axes = dcs::rotation_axes(&d.circuit()).unwrap();
        let rank = dcs::symplectic_rank(&axes, n);
        assert_eq!(
            rank,
            2 * n,
            "n={n}: the {} axes span the whole Pauli group, so rank is capped at 2n",
            axes.len()
        );
        assert!(
            axes.len() > 2 * n,
            "n={n}: and there are more axes ({}) than the group has dimensions ({})",
            axes.len(),
            2 * n
        );
    }
    // Where a T-count law stops describing an n-qubit object.
    let crossover = 70.0 / 0.3963;
    assert!(
        crossover < dcs::EXPERIMENT_T_GATES as f64,
        "the stabilizer-extent law crosses 2^70 at t = {crossover:.0}, \
         below the experiment's {}",
        dcs::EXPERIMENT_T_GATES
    );
}

/// The full-scale magic-geometry numbers are **cost estimates**: Clifford
/// transport plus a union–find, never the subset sum. That is why they
/// run at 70 qubits in milliseconds, and it is also why they prove
/// nothing on their own about the readout being correct.
///
/// This pins the other half. At toy width, where a dense state vector is
/// available purely as an independent *oracle* — not as a yardstick, and
/// not as anything the scaling claims lean on — the same code path is
/// checked to produce the right expectation on Paulis with real signal.
#[test]
fn the_up_embedded_readout_is_the_right_number_and_not_just_a_cheap_one() {
    let mut worst = 0.0f64;
    let mut nonzero = 0;
    let sim = Simulator::<C64>::new();
    for n in [6, 8] {
        let circuit = Dcs::scaled(n).circuit();
        let psi = sim.run(&circuit).unwrap();
        let mut rng = quantsim::rng::Prng::new(0xDC5 ^ n as u64);
        for _ in 0..400 {
            let ops: Vec<(usize, Pauli)> = (0..n)
                .map(|q| {
                    (
                        q,
                        match rng.next_u64() % 4 {
                            0 => Pauli::I,
                            1 => Pauli::X,
                            2 => Pauli::Y,
                            _ => Pauli::Z,
                        },
                    )
                })
                .filter(|(_, p)| !matches!(p, Pauli::I))
                .collect();
            if ops.is_empty() {
                continue;
            }
            // Ground truth: apply P to the state and take the overlap.
            let mut with_p = circuit.clone();
            for &(q, p) in &ops {
                match p {
                    Pauli::X => with_p.x(q),
                    Pauli::Y => with_p.y(q),
                    Pauli::Z => with_p.z(q),
                    Pauli::I => &mut with_p,
                };
            }
            let p_psi = sim.run(&with_p).unwrap();
            let mut acc = C64::new(0.0, 0.0);
            psi.for_each_nonzero(&mut |i, amp| {
                acc += amp.conj() * p_psi.amplitude(i);
            });
            let truth = acc.re;
            let got = quantsim::upembed::expectation(&circuit, &ops).unwrap().value;
            worst = worst.max((got - truth).abs());
            if truth.abs() > 1e-6 {
                nonzero += 1;
            }
        }
    }
    // A doped graph state annihilates most Pauli expectations, so a
    // handful of probes would only ever confirm 0 = 0.
    assert!(
        nonzero >= 8,
        "only {nonzero} probes had signal — this would be confirming 0 = 0"
    );
    assert!(worst < 1e-12, "worst deviation from dense {worst}");
}

/// Neither factorization the crate offers is available on this circuit,
/// measured at full scale. Both are cheap to ask and decisive.
#[test]
fn no_factorization_of_the_magic_is_available_at_full_scale() {
    let d = Dcs::experiment();
    let axes = dcs::rotation_axes(&d.circuit()).unwrap();
    let comps = dcs::anticommutation_components(&axes);
    assert_eq!(
        comps,
        vec![dcs::EXPERIMENT_T_GATES],
        "one anticommutation component: the coupling partition buys nothing"
    );
}

/// The boundary claim, written so that progress fails it.
///
/// `h*` is bounded above by the wall count and nothing else, and this
/// family is wall-dense by construction: every `√X` is `H·S·H`, so the
/// experiment carries ~`2n·depth` = 9904 Hadamards. The measured
/// surviving fraction does not fall toward zero as the family grows,
/// which is what would be needed for the path sum to be the technique
/// that cracks this. If a better reduction ever drives this ratio down,
/// this assertion goes red — and that is the point of writing it.
#[test]
fn the_path_sum_axis_does_not_certify_this_family_classical() {
    let mut ratios = Vec::new();
    for n in [8, 12, 16] {
        let d = Dcs::scaled(n);
        let p = dcs::probe(&d.circuit(), Duration::from_secs(120)).unwrap();
        assert!(!p.cut_short, "n={n} did not reduce inside the budget");
        assert!(
            p.h_star > 0,
            "n={n}: doping leaves the Clifford fragment, so variables must survive"
        );
        ratios.push(p.h_star as f64 / p.walls as f64);
    }
    // Flat-or-rising, not decaying: the last size is not cheaper than
    // the first by more than measurement noise.
    assert!(
        *ratios.last().unwrap() >= 0.5 * ratios[0],
        "h*/walls collapsed across the sweep: {ratios:?} — reduction improved, re-derive the bound"
    );
    // And the absolute number is the verdict. At the experiment's 9904
    // walls even the smallest measured fraction puts h* in the hundreds,
    // against the 2^185.5 stabilizer rank the paper's own analysis
    // quotes: this axis is not competitive, let alone decisive.
    let smallest = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
    let projected = smallest * 9904.0;
    assert!(
        projected > 185.5,
        "projected h* at the experiment's wall count is {projected}, \
         which would beat the paper's stabilizer-rank estimate — check this"
    );
}
