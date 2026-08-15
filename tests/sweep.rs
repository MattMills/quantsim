//! The sequential sweep, checked against dense and against its own plan.

use quantsim::dcs::Dcs;
use quantsim::prelude::*;
use quantsim::sweep;

/// Exactness is the whole claim: this is a contraction, not a
/// truncation, so it must agree with dense to machine precision on
/// every amplitude of a doped instance.
#[test]
fn the_sweep_reproduces_every_amplitude_exactly() {
    for n in [6, 8, 10, 12] {
        let d = Dcs::scaled(n);
        assert!(d.t_gates > 0, "the family is doped at n={n}");
        let dev = sweep::max_deviation_vs_dense(&d.circuit()).unwrap();
        assert!(dev < 1e-12, "n={n}: worst deviation {dev}");
    }
}

/// The peak is `depth/2 + 1`, not the width and not the depth: a
/// brickwork bond is active every other layer, so the legs crossing it
/// are half the depth, and incoming and outgoing alternate so the live
/// set never holds both in full.
#[test]
fn the_peak_is_set_by_the_bond_not_by_the_register() {
    let d = Dcs::experiment();
    let p = sweep::plan(&d.circuit()).unwrap();
    assert_eq!(
        p.legs_per_bond.iter().max().copied().unwrap(),
        35,
        "a bond of a depth-70 brickwork carries depth/2 legs"
    );
    assert_eq!(p.peak_live_legs, 36);
    assert_eq!(p.peak_amplitudes(), 1u128 << 37);
    // Against the register it never holds.
    assert!(p.peak_amplitudes() < (1u128 << 70) / (1u128 << 32));
    // And every single-qubit gate — all 468 T among them — is free.
    assert!(p.single_qubit > 7000);
}

/// The T count does not enter the plan at all: same skeleton, doping
/// swept, identical peak.
#[test]
fn the_peak_does_not_move_when_the_doping_does() {
    let base = Dcs::experiment();
    let bare = sweep::plan(&base.with_t(0).skeleton()).unwrap();
    let doped = sweep::plan(&base.circuit()).unwrap();
    assert_eq!(bare.peak_live_legs, doped.peak_live_legs);
    assert_eq!(bare.legs_per_bond, doped.legs_per_bond);
    assert!(
        doped.single_qubit > bare.single_qubit,
        "doping did add gates"
    );
}

/// The price of independence, and that it is flat.
///
/// A volume that cannot see its left neighbour holds both of its
/// surfaces for its whole life, so it costs the *sum* of the two leg
/// counts however many world-lines sit between them. One world-line and
/// thirty-five cost the same, which is why the register's volumes
/// compose sequentially or not at all.
#[test]
fn an_independent_volume_costs_two_surfaces_whatever_its_width() {
    let c = Dcs::experiment().circuit();
    let sequential = sweep::plan(&c).unwrap().peak_live_legs;
    assert_eq!(sequential, 36);

    let mut peaks = Vec::new();
    for width in [1usize, 2, 5, 7, 10, 14] {
        let vp = sweep::plan_volumes(&c, width).unwrap();
        assert_eq!(vp.parallelism(), 70usize.div_ceil(width));
        // Interior volumes see a full surface on each side.
        let mid = vp.volumes.len() / 2;
        assert_eq!(vp.incoming[mid], 35);
        assert_eq!(vp.outgoing[mid], 35);
        peaks.push(vp.peak());
    }
    for p in &peaks {
        assert!(
            (70..=71).contains(p),
            "an independent volume peaked at {p} legs, not the two surfaces"
        );
        // The two surfaces are 35 legs each; the sweep's 36 is one of
        // them plus the qubit in hand.
        assert!(
            *p >= 2 * (sequential - 1),
            "independence came for free at {p}"
        );
    }

    // The whole register as one volume has no surfaces, so it is the
    // sweep itself and must agree with it exactly.
    let whole = sweep::plan_volumes(&c, 70).unwrap();
    assert_eq!(whole.parallelism(), 1);
    assert_eq!(whole.incoming[0], 0);
    assert_eq!(whole.outgoing[0], 0);
    assert_eq!(whole.peak(), sequential);
}

/// The scope is refused by name. A long-range leg stays open across
/// every qubit between its ends, which is the cost the sweep exists to
/// avoid, so it is an error rather than a silent reprice.
#[test]
fn long_range_and_non_cz_couplings_are_refused() {
    let mut far: Circuit<C64> = Circuit::new(4);
    far.h(0).cz(0, 3);
    let e = sweep::plan(&far).unwrap_err().to_string();
    assert!(e.contains("nearest-neighbour"), "got {e}");

    let mut cx: Circuit<C64> = Circuit::new(3);
    cx.h(0).cx(0, 1);
    let e = sweep::plan(&cx).unwrap_err().to_string();
    assert!(e.contains("not a CZ"), "got {e}");
}
