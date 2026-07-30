//! Computing *across* a set of E8 volumes, measured: what one native
//! operation reaches through the scale tower, what that is worth against
//! the qubit path, and which states the native operator set can reach at
//! all.

use quantsim::e8::across::{self, Native};
use quantsim::e8::constellation::{self, E8ConstellationState};
use quantsim::prelude::*;

fn spread() -> [Native; 4] {
    // A genuinely spread coset state: a size-1 support is a coset with a
    // linear character for trivial reasons and would test nothing.
    [
        Native::Fourier(0),
        Native::Fourier(1),
        across::cross_scale_translation(),
        Native::Modulate(constellation::basis(1)),
    ]
}

#[test]
fn one_native_translation_moves_every_e8_copy_in_the_tower() {
    for copies in [2usize, 3, 5] {
        let r = across::reach(8 * copies, &across::cross_scale_translation()).expect("full blocks");
        assert_eq!(
            r.copies_touched, copies,
            "a finest-scale translation must carry into every copy, not just its own"
        );
        assert_eq!(
            r.qubits_touched,
            8 * copies - 7,
            "the carry reaches all but the seven bits of the finest digit it cannot flip"
        );
        assert_eq!(r.max_support_out, 1, "a translation is a basis permutation");
    }
}

#[test]
fn a_single_copy_has_no_carries_at_all() {
    // E8/2E8 is F_2^8 and class_of is linear, so at one copy the
    // translation is a bitwise XOR — the cross-scale reach is created by
    // having more than one copy, and this is where it comes from.
    let r = across::reach(8, &across::cross_scale_translation()).expect("one copy");
    assert!(r.exhaustive, "one copy is swept over all 256 residues");
    assert_eq!(r.probes, 256);
    assert_eq!(r.copies, 1);
    assert_eq!(r.qubits_touched, 1, "XOR by a single-bit class label");
    assert_eq!(
        r.two_qubit_lower_bound,
        Some(0),
        "one X gate needs no two-qubit gate"
    );
    assert_eq!(r.influence_components, Some(8), "eight independent bits");
}

#[test]
fn the_carry_makes_the_influence_graph_connected_across_the_whole_register() {
    for copies in [2usize, 3, 4, 5] {
        let n = 8 * copies;
        let r = across::reach(n, &across::cross_scale_translation()).expect("full blocks");
        assert_eq!(
            r.qubits_involved,
            Some(n),
            "every qubit is either moved or feeds a carry"
        );
        assert_eq!(
            r.influence_components,
            Some(1),
            "the carry chain ties the whole register into one component"
        );
        assert_eq!(
            r.two_qubit_lower_bound,
            Some(n - 1),
            "a connected influence pattern over n qubits costs at least n-1 two-qubit gates"
        );
    }
}

#[test]
fn the_advantage_over_the_qubit_path_is_linear_in_the_copies_not_exponential() {
    let scaling = across::across_scaling(6, &across::cross_scale_translation()).expect("sweep");
    assert_eq!(scaling.copies, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(
        scaling.per_copy_reach, 8.0,
        "each added E8 copy adds exactly its eight qubits to the reach"
    );
    assert_eq!(
        scaling.per_copy_advantage, 8.0,
        "and exactly eight more two-qubit gates to the bound it replaces"
    );
    let law = scaling.advantage_law.expect("three sizes with a bound");
    assert!(
        law.is_subexponential(),
        "the advantage is a linear factor, not an exponential one: {law:?}"
    );
    match law {
        Law::Polynomial { degree } => assert!(
            (degree - 1.0).abs() < 0.15,
            "degree {degree} should be one: the bound is 8m-1"
        ),
        other => panic!("expected a polynomial advantage law, got {other:?}"),
    }
    // The native operation's own cost is not what grows.
    assert!(scaling
        .cost_law
        .as_ref()
        .is_some_and(Law::is_subexponential));
}

#[test]
fn a_modulation_changes_no_qubit_value_and_has_no_influence_graph() {
    let r = across::reach(24, &Native::Modulate(constellation::basis(0))).expect("full blocks");
    assert_eq!(
        r.qubits_touched, 0,
        "a character is diagonal: it changes phases, never basis labels"
    );
    assert_eq!(r.copies_touched, 0);
    assert_eq!(
        r.influence_components, None,
        "a non-permutation has no single output pattern to build a graph from"
    );
    assert_eq!(r.two_qubit_lower_bound, None);
    assert_eq!(r.qubits_involved, None);
}

#[test]
fn the_native_direct_dft_costs_exponentially_in_tower_depth() {
    let f = across::fourier_cost(7, 0).expect("sweep");
    assert_eq!(
        f.support,
        vec![2, 4, 8, 16, 32, 64, 128],
        "the transform along one direction spreads over the 2^m residues of that coordinate"
    );
    match f.support_law {
        Some(Law::Exponential { base }) => assert!(
            (base - 2.0).abs() < 1e-6,
            "support doubles per copy; fitted base {base}"
        ),
        other => panic!("expected exponential support growth, got {other:?}"),
    }
    let cost = f.cost_law.expect("three sizes");
    assert!(
        !cost.is_subexponential(),
        "this is a direct DFT, not a fast one — its cost is the honest limit \
         of the native set: {cost:?}"
    );
}

#[test]
fn every_native_operation_stays_inside_the_coset_class() {
    let prep = spread();
    for op in [
        across::cross_scale_translation(),
        Native::Modulate(constellation::basis(0)),
        Native::Permute([1, 2, 3, 4, 5, 6, 7, 0]),
        Native::Reflect(quantsim::e8::roots()[0]),
        Native::Fourier(0),
    ] {
        let p = across::class_preservation(16, &prep, &op).expect("full blocks");
        assert!(
            p.before.in_class,
            "the prepared state must be in the class to begin with"
        );
        assert!(p.before.size > 1, "and must actually be spread");
        assert!(
            p.preserved(),
            "{} left the coset-with-linear-character class: coset {}, uniform {}, \
             character residual {:.3e}",
            p.op,
            p.after.is_coset,
            p.after.uniform_modulus,
            p.after.character_residual
        );
    }
}

#[test]
fn a_t_gate_through_the_qubit_path_leaves_the_class_the_natives_cannot() {
    let prep = spread();
    let t = across::qubit_gate_class(16, &prep, "t", &[], &[0]).expect("t on the tower");
    assert!(
        t.is_coset,
        "T is diagonal, so it cannot move the support off the coset"
    );
    assert!(t.uniform_modulus, "nor change any modulus");
    assert!(
        t.character_residual > 0.1,
        "but its phase is not a linear character of the coset: residual {:.3e}",
        t.character_residual
    );
    assert!(
        !t.in_class,
        "so the qubit path reaches states no sequence of native operators can"
    );
    // Clifford gates on this state happen to stay inside; the class is
    // not a Clifford/non-Clifford boundary and the test does not claim it
    // is. What matters is that something reachable through the qubit path
    // is not reachable natively.
    let cx = across::qubit_gate_class(16, &prep, "cx", &[], &[0, 9]).expect("cx on the tower");
    assert!(cx.in_class);
}

#[test]
fn the_class_measurement_rejects_a_support_that_is_not_a_coset() {
    // Two points whose difference has order four in the residue group
    // cannot be a coset of a subgroup: {0, d} is closed under addition
    // only when 2d = 0.
    let mut state = E8ConstellationState::new(16).expect("two copies");
    let a = constellation::compose(&[0, 0]);
    let b = constellation::compose(&[1, 0]);
    let bits = |p: &[i64; 8]| -> u64 {
        let digits = constellation::decompose(p, 2).expect("lattice point");
        u64::from(digits[0]) | (u64::from(digits[1]) << 8)
    };
    let amp = C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    state
        .load(&[(bits(&a), amp), (bits(&b), amp)])
        .expect("two points");
    let class = across::support_class(&state);
    assert_eq!(class.size, 2);
    assert!(
        !class.is_coset,
        "the difference has order four, so the pair is not an affine coset"
    );
    assert!(!class.in_class);
    assert!(class.uniform_modulus, "both amplitudes have equal modulus");
}

#[test]
fn reach_refuses_a_partial_block_rather_than_embedding_a_subset() {
    for n in [0usize, 1, 7, 9, 20] {
        let err = across::reach(n, &across::cross_scale_translation());
        assert!(
            err.is_err(),
            "the native operators act on the residue group, which {n} qubits is not"
        );
    }
    assert!(across::reach(16, &across::cross_scale_translation()).is_ok());
}
