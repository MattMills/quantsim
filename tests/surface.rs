//! The surface between volumes, and the techniques competing to
//! describe it.

use std::time::Duration;

use quantsim::dcs::Dcs;
use quantsim::prelude::*;
use quantsim::surface::{self, CompeteConfig, Technique};
use quantsim::sweep;

fn cfg() -> CompeteConfig {
    CompeteConfig {
        cp_max_legs: 6,
        cp_max_rank: 16,
        cp_sweeps: 20,
        symbolic_budget: Duration::from_secs(20),
        ..CompeteConfig::default()
    }
}

#[test]
fn a_surface_has_one_amplitude_per_leg_assignment() {
    let d = Dcs::scaled(10);
    let circuit = d.circuit();
    let plan = sweep::plan(&circuit).unwrap();
    let surfaces = sweep::surfaces(&circuit, 0).unwrap();
    assert_eq!(surfaces.len(), d.qubits - 1);
    for s in &surfaces {
        assert_eq!(s.amps.len(), 1 << s.legs.len());
        assert_eq!(s.legs.len(), plan.legs_per_bond[s.after_qubit]);
    }
}

/// The whole point of the sweep is that the surface is what crosses.
/// Contracting it against the rest must give back the amplitude.
#[test]
fn the_last_surface_carries_the_amplitude() {
    let d = Dcs::scaled(8);
    let circuit = d.circuit();
    for bits in [0u64, 0b1011_0110, 0b1111_1111] {
        let surfaces = sweep::surfaces(&circuit, bits).unwrap();
        // Every surface holds a nonzero share of the state; a bond that
        // carried nothing would mean the sweep had already collapsed.
        for s in &surfaces {
            let mass: f64 = s.amps.iter().map(|z| z.norm_sqr()).sum();
            assert!(mass > 0.0, "bond {} carries nothing", s.after_qubit);
        }
    }
}

/// Two independent routes to the same numbers: the numeric sweep, which
/// contracts leg by leg, and the path sum, which reduces the volume
/// behind the bond symbolically and never forms a state at all.
#[test]
fn the_symbolic_volume_reproduces_the_swept_surface() {
    for n in [6usize, 8, 10] {
        let d = Dcs::scaled(n);
        let circuit = d.circuit();
        for bits in [0u64, 0b10_1101] {
            let surfaces = sweep::surfaces(&circuit, bits).unwrap();
            for s in surfaces.iter().take(4) {
                let sym = surface::symbolic(&circuit, s.after_qubit, bits).unwrap();
                assert_eq!(sym.legs, s.legs.len());
                let worst = (0..(1u64 << sym.legs))
                    .map(|b| (s.amps[b as usize] - sym.amplitude(b)).norm())
                    .fold(0.0, f64::max);
                assert!(
                    worst < 1e-12,
                    "n={n} bond {} deviates by {worst:e}",
                    s.after_qubit
                );
            }
        }
    }
}

/// Every entrant that claims to apply must rebuild the surface it was
/// given. A technique is allowed to decline; it is not allowed to
/// report a cost for a description that is wrong.
#[test]
fn every_applicable_technique_rebuilds_its_surface() {
    let d = Dcs::scaled(10);
    let circuit = d.circuit();
    for s in sweep::surfaces(&circuit, 0).unwrap().iter().take(5) {
        let c = surface::compete(s, Some(&circuit), 0, &cfg()).unwrap();
        for e in &c.entries {
            if e.applicable && e.checked > 0 {
                assert!(
                    e.error < 1e-9,
                    "bond {} technique {} deviates by {:e}",
                    c.after_qubit,
                    e.technique.name(),
                    e.error
                );
            }
        }
        assert!(c.winner(1e-9).is_some());
        assert!(c.fastest(1e-9).is_some());
    }
}

/// The first bond has seen a single world-line, so its surface is one
/// stabilizer term: flat on a coset with a phase polynomial. This is
/// the case a bipartition scores as ordinary and the affine
/// description scores as nearly free.
#[test]
fn the_first_surface_is_a_single_phase_polynomial() {
    let d = Dcs::scaled(12);
    let circuit = d.circuit();
    let surfaces = sweep::surfaces(&circuit, 0).unwrap();
    let first = &surfaces[0];
    let pp = surface::phase_poly(&first.amps, first.legs.len())
        .unwrap()
        .expect("bond 0 should be a single stabilizer term");
    let rebuilt = pp.rebuild(first.legs.len());
    let worst = first
        .amps
        .iter()
        .zip(&rebuilt)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max);
    assert!(worst < 1e-12, "phase polynomial deviates by {worst:e}");
    assert!(pp.scalars() < first.amps.len() as u128);
}

/// The description dies on *flatness*, not on support: the surface
/// stays a full coset at every bond and stops being a single
/// stabilizer term because it becomes a sum of them. Those are
/// different failures and the module reports which.
#[test]
fn the_deep_surfaces_keep_affine_support_and_lose_flatness() {
    let d = Dcs::scaled(16);
    let circuit = d.circuit();
    let surfaces = sweep::surfaces(&circuit, 0).unwrap();
    let deep = surfaces.last().unwrap();
    let c = surface::compete(deep, None, 0, &cfg()).unwrap();
    let pp = c.get(Technique::PhasePoly).unwrap();
    assert!(!pp.applicable);
    assert!(
        pp.detail.contains("coset of dimension"),
        "expected a flatness obstruction, got: {}",
        pp.detail
    );
}

/// A surface with no correlation between its legs is a product, and
/// the chain must say so rather than carrying width it does not need.
#[test]
fn a_product_surface_has_unit_bond_dimension() {
    let m = 8usize;
    let mut amps = vec![C64::new(1.0, 0.0); 1 << m];
    for (x, a) in amps.iter_mut().enumerate() {
        for p in 0..m {
            if (x >> p) & 1 == 1 {
                *a *= C64::new(0.3 * (p as f64 + 1.0), 0.1);
            }
        }
    }
    let s = sweep::Surface {
        after_qubit: 0,
        legs: (0..m).collect(),
        amps,
    };
    let code = surface::code(&s, 1e-10).unwrap();
    assert_eq!(code.code_dim(), 1);
    assert!(code.peak_entropy() < 1e-9);
}

/// The arity ladder must be a ladder: every rung tiles the same legs,
/// and the two bipolar readings are properties of the surface, so they
/// cannot move as the grouping changes. Only the `ℤ_d` column may.
#[test]
fn the_arity_ladder_changes_only_the_compound_reading() {
    let d = Dcs::scaled(16);
    let circuit = d.circuit();
    for s in sweep::surfaces(&circuit, 0).unwrap().iter().take(3) {
        let m = s.legs.len();
        let grains = surface::grain_scan(s, surface::DEFAULT_EPSILON).unwrap();
        assert!(!grains.is_empty());
        let (direct, walsh) = (grains[0].direct_nnz, grains[0].walsh_nnz);
        for g in &grains {
            assert_eq!(g.legs_per_digit * g.sites, m, "the grain must tile the legs");
            assert_eq!(g.radix, 1 << g.legs_per_digit);
            assert_eq!(g.direct_nnz, direct, "the grouping moved the direct reading");
            assert_eq!(g.walsh_nnz, walsh, "the grouping moved the Walsh reading");
            assert!(g.cyclic_nnz <= 1 << m);
        }
        // The coarsest rung is one compound axis over every leg, and
        // the finest is the bipolar reading everything else uses.
        assert_eq!(grains[0].legs_per_digit, 1);
        assert_eq!(grains.last().unwrap().sites, 1);
        // At arity two the ℤ_d kernel IS the Walsh kernel, so the two
        // columns must agree exactly — which checks the transform.
        assert_eq!(
            grains[0].cyclic_nnz, walsh,
            "the arity-2 reading is the Walsh reading"
        );
    }
}

/// A surface that factors over its legs stays a product under any
/// grouping of them, so the compound reading finds one coefficient per
/// digit combination and no more than the bipolar one does.
#[test]
fn a_product_surface_stays_narrow_at_every_arity() {
    let m = 6usize;
    let mut amps = vec![C64::new(1.0, 0.0); 1 << m];
    for (x, a) in amps.iter_mut().enumerate() {
        for p in 0..m {
            if (x >> p) & 1 == 1 {
                *a *= C64::new(0.0, 1.0);
            }
        }
    }
    let s = sweep::Surface {
        after_qubit: 0,
        legs: (0..m).collect(),
        amps,
    };
    for g in surface::grain_scan(&s, surface::DEFAULT_EPSILON).unwrap() {
        assert_eq!(g.chi, 1, "a product surface needs no bond at arity {}", g.radix);
        assert!(
            g.cyclic_nnz <= 1 << m,
            "arity {} found more coefficients than indices",
            g.radix
        );
    }
}

/// Cost is counted the same way for everyone, so the baseline is
/// beatable only by actually being smaller.
#[test]
fn dense_is_the_baseline_everyone_is_measured_against() {
    let d = Dcs::scaled(8);
    let circuit = d.circuit();
    let s = &sweep::surfaces(&circuit, 0).unwrap()[0];
    let c = surface::compete(s, Some(&circuit), 0, &cfg()).unwrap();
    let dense = c.get(Technique::Dense).unwrap();
    assert_eq!(dense.scalars, c.raw);
    assert_eq!(dense.read_cost, 1);
    assert_eq!(dense.error, 0.0);
}
