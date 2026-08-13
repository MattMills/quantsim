//! Cutting the circuit instead of compressing the state.
//!
//! The claim under test is narrow and checkable: on a
//! linear-nearest-neighbour brickwork, the two halves of the chain are
//! joined by only `depth/2` gates, so the whole state is a sum of
//! `2^{depth/2}` product states — and every single-qubit gate, `T`
//! included, sits inside a half and costs nothing.
//!
//! That matters because it prices a circuit by a feature completely
//! disjoint from the one stabilizer-rank methods price. Where those pay
//! `2^{αt}` and nothing else, this pays for the interface and ignores
//! `t` entirely.

use quantsim::cutsim;
use quantsim::dcs::{self, Dcs};
use quantsim::prelude::*;
use std::time::Instant;

/// The decomposition is exact, not a truncation: every amplitude of a
/// doped instance, against the dense reference.
#[test]
fn the_cut_decomposition_reproduces_every_amplitude_exactly() {
    for n in [8, 10, 12] {
        let d = Dcs::scaled(n);
        assert!(d.t_gates > 0, "the family is doped at n={n}");
        let circuit = d.circuit();
        let plan = cutsim::plan(&circuit, n / 2).unwrap();
        let dense = Simulator::<C64>::new().run(&circuit).unwrap();
        let targets: Vec<u64> = (0..(1u64 << n)).collect();
        let got = cutsim::amplitudes_at(&circuit, &plan, &targets).unwrap();
        let worst = targets
            .iter()
            .zip(&got)
            .map(|(&x, g)| (*g - dense.amplitude(x)).norm())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-12, "n={n}: worst deviation {worst}");
    }
}

/// The interface of the experiment's own circuit: 35 crossing gates out
/// of 2415, and every one of the 468 `T` gates on the interior.
///
/// `depth/2` is the whole story — a brickwork bond is active every other
/// layer, so the number of gates joining the halves is set by the depth
/// and not by the 2415 the circuit contains.
#[test]
fn the_experiments_chain_is_joined_by_thirty_five_gates() {
    let d = Dcs::experiment();
    let circuit = d.circuit();
    let plan = cutsim::best_plan(&circuit).unwrap();
    assert_eq!(
        plan.crossings.len(),
        dcs::EXPERIMENT_DEPTH / 2,
        "a brickwork bond is active every other layer"
    );
    assert_eq!(plan.crossings.len(), 35);
    assert_eq!(
        plan.interior_two_qubit,
        dcs::EXPERIMENT_TWO_QUBIT_GATES - 35,
        "everything else is local to a half"
    );
    // The governing exponent is crossings + the wider half, not the
    // T-count and not the two-qubit count.
    assert_eq!(plan.cost_exponent(), 70);
    // And every T gate is interior, by construction: they are
    // single-qubit gates, and no single-qubit gate can cross a cut.
    let single_qubit_ops = circuit
        .ops()
        .iter()
        .filter(|op| op.qubits().len() == 1)
        .count();
    assert_eq!(plan.single_qubit, single_qubit_ops);
    assert!(
        single_qubit_ops > dcs::EXPERIMENT_T_GATES,
        "all {} T gates are among the {single_qubit_ops} interior single-qubit gates",
        dcs::EXPERIMENT_T_GATES
    );
}

/// The crux. If the cost were set by the magic, sweeping the doping at a
/// byte-identical skeleton would move the clock. It does not: the extra
/// time is the linear cost of applying more gates, not an exponential in
/// how many of them are non-Clifford.
#[test]
fn the_cost_does_not_move_when_the_doping_does() {
    let n = 14;
    let base = Dcs::scaled(n);
    let targets: Vec<u64> = (0..64u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
    let run = |t: usize| -> f64 {
        let c = base.with_t(t).circuit();
        let p = cutsim::plan(&c, n / 2).unwrap();
        // Best of three: this is a timing assertion, so warm-up and
        // scheduler noise must not be what it measures.
        (0..3)
            .map(|_| {
                let t0 = Instant::now();
                cutsim::amplitudes_at(&c, &p, &targets).unwrap();
                t0.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min)
    };
    let clifford = run(0);
    let doped = run(256);
    assert!(
        doped < 2.0 * clifford,
        "256 T gates cost {doped:.4}s against {clifford:.4}s for none — \
         a factor {:.2}, which is not the 2^(0.33·256) a stabilizer \
         decomposition would pay",
        doped / clifford
    );
    // And the branch count — the thing that is exponential — is
    // identical, because doping cannot add a crossing gate.
    let a = cutsim::plan(&base.with_t(0).circuit(), n / 2).unwrap();
    let b = cutsim::plan(&base.with_t(256).circuit(), n / 2).unwrap();
    assert_eq!(a.crossings.len(), b.crossings.len());
}

/// The refusal is honest: a crossing gate this cannot cut into two terms
/// is named rather than silently repriced.
#[test]
fn a_crossing_gate_that_is_not_a_cz_is_refused() {
    let mut c: Circuit<C64> = Circuit::new(4);
    c.h(0).cx(1, 2);
    let plan = cutsim::plan(&c, 2).unwrap();
    assert_eq!(plan.crossings.len(), 1);
    let err = cutsim::amplitude(&c, &plan, 0).unwrap_err();
    assert!(
        err.to_string().contains("not a CZ"),
        "expected a named refusal, got {err}"
    );
}
