//! The path sum: a state held as its circuit's closed form rather than
//! as amplitudes. That it agrees with dense to the last bit including
//! global phase, that its exponent is the *measured* `h*` and not the
//! width, that Gottesman–Knill falls out of reduction with no tableau
//! anywhere — and where the fragment ends and it says so.

use quantsim::pathsum::{
    turn_from_dyadic, turn_from_radians, PathSum, EIGHTH, HALF, QUARTER, THREE_QUARTER,
};
use quantsim::prelude::*;

/// A random Clifford+T word, as a `Circuit` so dense can run the same one.
fn word(n: usize, gates: usize, t_share: u64, seed: u64) -> Circuit<C64> {
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for _ in 0..gates {
        let a = (rng.next_u64() % n as u64) as usize;
        let b = (rng.next_u64() % n as u64) as usize;
        match rng.next_u64() % (6 + t_share) {
            0 => c.gate("h", vec![], vec![a]),
            1 => c.gate("s", vec![], vec![a]),
            2 => c.gate("x", vec![], vec![a]),
            3 => c.gate("z", vec![], vec![a]),
            4 => c.gate("y", vec![], vec![a]),
            5 => {
                if a != b {
                    c.gate("cx", vec![], vec![a, b])
                } else {
                    c.gate("h", vec![], vec![a])
                }
            }
            _ => c.gate("t", vec![], vec![a]),
        };
    }
    c
}

fn dense_of(c: &Circuit<C64>) -> Vec<C64> {
    let sim: Simulator = Simulator::new();
    let s = sim.run(c).unwrap();
    (0..1u64 << c.num_qubits()).map(|b| s.amplitude(b)).collect()
}

fn worst_deviation(c: &Circuit<C64>) -> f64 {
    let ps = PathSum::from_circuit(c).unwrap();
    let want = dense_of(c);
    let got = ps.to_dense();
    want.iter()
        .zip(&got)
        .map(|(a, b)| ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt())
        .fold(0.0f64, f64::max)
}

// ── it is the same state, not the same probabilities ─────────────────

#[test]
fn a_composed_path_sum_reproduces_dense_amplitudes_including_global_phase() {
    // Amplitude for amplitude, not modulus for modulus: `y` and the
    // rotations carry phases a probability-only check would never see.
    let mut worst = 0.0f64;
    for seed in 0..40u64 {
        for n in 2..=4usize {
            worst = worst.max(worst_deviation(&word(n, 14, 2, seed * 7 + n as u64)));
        }
    }
    assert!(worst < 1e-12, "worst amplitude deviation {worst:e}");
}

#[test]
fn the_dyadic_rotations_reduce_to_the_same_fragment() {
    // rz/rx/rzz/cp/sx/ccx are not primitive letters — they are compiled
    // into the fragment, and the compilation has to be phase-exact.
    let mut c = Circuit::new(3);
    c.gate("h", vec![], vec![0]);
    c.gate("rx", vec![std::f64::consts::FRAC_PI_2], vec![1]);
    c.gate("rz", vec![std::f64::consts::FRAC_PI_4], vec![0]);
    c.gate("rzz", vec![std::f64::consts::FRAC_PI_2], vec![0, 2]);
    c.gate("cp", vec![std::f64::consts::FRAC_PI_2], vec![1, 2]);
    c.gate("sx", vec![], vec![2]);
    c.gate("ccx", vec![], vec![0, 1, 2]);
    c.gate("sxdg", vec![], vec![0]);
    c.gate("swap", vec![], vec![0, 1]);
    let want = dense_of(&c);
    let got = PathSum::from_circuit(&c).unwrap().to_dense();
    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
        assert!(
            (a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12,
            "amplitude {i}: dense {a:?} vs path sum {b:?}"
        );
    }
}

#[test]
fn a_state_that_cancels_is_proved_zero_rather_than_summed_to_zero() {
    // H·H on |0⟩ leaves |1⟩ with amplitude exactly 0. Reduction should
    // *know* that, not discover it by cancelling floats.
    let mut c = Circuit::new(1);
    c.gate("h", vec![], vec![0]).gate("h", vec![], vec![0]);
    let ps = PathSum::from_circuit(&c).unwrap();
    assert_eq!(ps.amplitude(0).re, 1.0);
    assert_eq!(ps.amplitude(0).im, 0.0);
    let one = ps.amplitude(1);
    assert_eq!((one.re, one.im), (0.0, 0.0), "not merely small — zero");
}

// ── Gottesman–Knill, from reduction alone ────────────────────────────

#[test]
fn every_clifford_circuit_reduces_to_no_internal_variables() {
    // The headline. No stabilizer tableau exists anywhere in this
    // module; the three rewrite rules consume every variable a wall
    // introduces, on their own.
    for &(n, gates) in &[(6usize, 120usize), (16, 300), (32, 600)] {
        for seed in 0..3u64 {
            let ps = PathSum::from_circuit(&word(n, gates, 0, seed)).unwrap();
            assert_eq!(
                ps.internal_vars(),
                0,
                "n={n} gates={gates} seed={seed}: {} of {} walls survived",
                ps.internal_vars(),
                ps.allocated_vars()
            );
            assert!(ps.allocated_vars() > n, "the circuit had walls to consume");
        }
    }
}

#[test]
fn readout_costs_two_to_the_h_star_and_not_two_to_the_width() {
    // 32 qubits is 4·10^9 dense amplitudes. h* = 0 means one term.
    let ps = PathSum::from_circuit(&word(32, 600, 0, 11)).unwrap();
    assert_eq!(ps.qubits(), 32);
    assert_eq!(ps.internal_vars(), 0);
    let a = ps.amplitude(0);
    assert!(a.re.is_finite() && a.im.is_finite());
    assert!(
        quantsim::DenseState::<C64>::new(32).is_err(),
        "dense must refuse this width, or the comparison proves nothing"
    );
}

// ── the exponent is the T-count, not the width ───────────────────────

/// Layers of walls with T gates *between* them, so a wall follows every
/// T. T gates applied after the last wall are only a diagonal phase and
/// reduce for free — an easy way to measure a flattering `h*` that means
/// nothing.
fn doped(n: usize, layers: usize, t_per_layer: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for l in 0..layers {
        for q in 0..n {
            c.gate("h", vec![], vec![q]);
        }
        for q in (0..n.saturating_sub(1)).step_by(2) {
            c.gate("cx", vec![], vec![q, q + 1]);
        }
        for k in 0..t_per_layer {
            c.gate("t", vec![], vec![(k * 3 + l) % n]);
        }
    }
    for q in 0..n {
        c.gate("h", vec![], vec![q]);
    }
    c
}

#[test]
fn h_star_is_flat_in_the_width_at_a_fixed_t_count() {
    // The sub-exponentiality claim as a measurement: hold the magic
    // fixed, grow the register eightfold, and watch the exponent not
    // move. It is the *circuit's* magic that costs, not the register.
    let mut seen = Vec::new();
    for n in [8usize, 16, 32, 64] {
        let ps = PathSum::from_circuit(&doped(n, 3, 2)).unwrap();
        seen.push((n, ps.internal_vars()));
    }
    let worst = seen.iter().map(|&(_, h)| h).max().unwrap();
    assert!(
        worst <= 12,
        "h* should stay bounded as the width grows: {seen:?}"
    );
    let widest = seen.last().unwrap();
    assert!(
        widest.1 < widest.0,
        "at n={} the exponent {} is not below the width — the claim fails",
        widest.0,
        widest.1
    );
}

#[test]
fn h_star_grows_with_the_t_count_at_a_fixed_width() {
    let none = PathSum::from_circuit(&doped(10, 3, 0)).unwrap();
    assert_eq!(none.internal_vars(), 0, "no magic, no exponent");
    let heavy = PathSum::from_circuit(&doped(10, 3, 6)).unwrap();
    assert!(
        heavy.internal_vars() > 0,
        "magic has to cost something: h* = {}",
        heavy.internal_vars()
    );
    // and it is measured after the fact, never predicted: h* is not
    // monotone in the T-count, so the only sound assertion is the pair
    // of endpoints above plus the bound below.
    for t in 0..=6usize {
        let ps = PathSum::from_circuit(&doped(10, 3, t)).unwrap();
        assert!(
            ps.internal_vars() <= 3 * t,
            "h* = {} exceeded the walls the {t} T gates could strand",
            ps.internal_vars()
        );
    }
}

// ── the reduction path itself ────────────────────────────────────────

#[test]
fn reduction_is_deterministic_in_its_normal_form_and_its_cost() {
    // `HashMap` seeds its hasher per instance, so the same circuit
    // reduced twice in one process exercises two different iteration
    // orders. Rule V picks its victim greedily rather than by whichever
    // key came first, so both runs must land on the same normal form —
    // and, just as importantly, take the same number of splits to get
    // there. Before that fix the same 32-qubit circuit reduced to
    // anywhere between 82 and 103 monomials.
    let c = word(32, 600, 0, 3);
    let first = PathSum::from_circuit(&c).unwrap();
    for _ in 0..4 {
        let again = PathSum::from_circuit(&c).unwrap();
        assert_eq!(again.terms(), first.terms(), "normal form moved");
        assert_eq!(again.splits(), first.splits(), "reduction cost moved");
        assert_eq!(again.internal_vars(), first.internal_vars());
    }
}

#[test]
fn reduction_cost_tracks_the_split_count_not_the_width() {
    // Rules E and G each retire a variable, so they are bounded by the
    // wall count; only V can fire repeatedly. A wider register at the
    // same gate count therefore does not cost more — a *denser* one
    // does, and the split counter is what says which.
    let sparse = PathSum::from_circuit(&word(64, 600, 0, 5)).unwrap();
    let dense = PathSum::from_circuit(&word(16, 600, 0, 5)).unwrap();
    assert_eq!(sparse.internal_vars(), 0);
    assert_eq!(dense.internal_vars(), 0);
    assert!(
        sparse.splits() <= dense.splits().max(1) * 4,
        "quadrupling the width should not blow up the work: {} vs {}",
        sparse.splits(),
        dense.splits()
    );
}

// ── exact arithmetic, and the edge of the fragment ───────────────────

#[test]
fn turns_are_exact_integers_and_not_rounded_angles() {
    assert_eq!(turn_from_dyadic(1, 1).unwrap(), HALF);
    assert_eq!(turn_from_dyadic(1, 2).unwrap(), QUARTER);
    assert_eq!(turn_from_dyadic(3, 2).unwrap(), THREE_QUARTER);
    assert_eq!(turn_from_dyadic(1, 3).unwrap(), EIGHTH);
    // eight T gates are the identity, exactly — the sum wraps to zero
    // with no residue, which a float angle could not promise
    let eight = EIGHTH.wrapping_mul(8);
    assert_eq!(eight, 0, "8·(1/8) of a turn must be exactly a whole turn");
    // and the same statement at the level of a state
    let mut c = Circuit::new(1);
    c.gate("h", vec![], vec![0]);
    for _ in 0..8 {
        c.gate("t", vec![], vec![0]);
    }
    let ps = PathSum::from_circuit(&c).unwrap();
    assert_eq!(ps.terms(), 0, "the phase polynomial should be empty again");
}

#[test]
fn radians_convert_when_dyadic_and_are_refused_when_not() {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};
    assert_eq!(turn_from_radians(0.0).unwrap(), 0);
    assert_eq!(turn_from_radians(PI).unwrap(), HALF);
    assert_eq!(turn_from_radians(FRAC_PI_2).unwrap(), QUARTER);
    assert_eq!(turn_from_radians(FRAC_PI_4).unwrap(), EIGHTH);
    assert_eq!(turn_from_radians(TAU).unwrap(), 0);
    assert_eq!(turn_from_radians(-FRAC_PI_2).unwrap(), THREE_QUARTER);
    // A third of a turn is not dyadic. It is refused, not rounded —
    // this representation is exact on its fragment or it says nothing.
    assert!(turn_from_radians(TAU / 3.0).is_err());
    assert!(turn_from_radians(1.0).is_err());
    assert!(turn_from_radians(f64::NAN).is_err());
}

#[test]
fn a_gate_outside_the_fragment_is_refused_by_name() {
    let mut c = Circuit::new(2);
    c.gate("h", vec![], vec![0]);
    c.gate("rz", vec![0.3], vec![0]); // 0.3 rad is not dyadic
    match PathSum::from_circuit(&c) {
        Err(Error::InvalidState(msg)) => assert!(
            msg.contains("dyadic"),
            "the refusal should say why: {msg}"
        ),
        other => panic!("expected a refusal, got {other:?}"),
    }

    let mut c = Circuit::new(2);
    c.gate("iswap", vec![], vec![0, 1]);
    match PathSum::from_circuit(&c) {
        Err(Error::InvalidState(msg)) => assert!(
            msg.contains("iswap"),
            "the refusal should name the gate: {msg}"
        ),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_qubit_outside_the_register_is_refused() {
    let mut ps = PathSum::new(3);
    assert!(ps.h(3).is_err());
    assert!(ps.cnot(0, 0).is_err(), "a cnot needs two distinct qubits");
    assert!(ps.cz(0, 5).is_err());
}
