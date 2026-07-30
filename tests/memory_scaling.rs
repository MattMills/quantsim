//! Width scaling = memory scaling, asserted "in actuality": the reported
//! footprints must follow the predicted asymptotics per representation and
//! per amplitude algebra. The companion `examples/width_scaling.rs` prints
//! the same numbers (plus process RSS) as a table.

mod common;

use quantsim::backend::{Backend, DenseState, SparseState};
use quantsim::prelude::*;

fn h_layer(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    c
}

#[test]
fn dense_memory_doubles_per_qubit() {
    let amp = std::mem::size_of::<C64>();
    assert_eq!(amp, 16);
    let mut previous = 0usize;
    for n in 4..=16 {
        let state = DenseState::<C64>::new(n).unwrap();
        let mem = state.memory_bytes();
        let ideal = amp << n;
        assert!(mem >= ideal, "n={n}: {mem} < {ideal}");
        assert!(
            mem <= ideal + 512,
            "n={n}: overhead too large: {mem} vs {ideal}"
        );
        // Doubling ratio, once past the widths where the constant struct
        // overhead is comparable to the amplitude vector itself.
        if n > 8 {
            let ratio = mem as f64 / previous as f64;
            assert!((1.9..=2.1).contains(&ratio), "n={n}: ratio {ratio}");
        }
        previous = mem;
    }
}

#[test]
fn dense_memory_scales_with_algebra_dimension() {
    // The same width costs 8× more over 𝕆 (dim 8) than over ℝ (dim 1).
    let n = 10;
    let real = DenseState::<f64>::new(n).unwrap().memory_bytes();
    let complex = DenseState::<C64>::new(n).unwrap().memory_bytes();
    let quat = DenseState::<Quaternion>::new(n).unwrap().memory_bytes();
    let oct = DenseState::<Octonion>::new(n).unwrap().memory_bytes();
    let sed = DenseState::<Sedenion>::new(n).unwrap().memory_bytes();
    let approx = |a: usize, b: usize, factor: f64| {
        let r = b as f64 / a as f64;
        assert!(
            (r / factor - 1.0).abs() < 0.05,
            "{a} vs {b}: ratio {r}, expected {factor}"
        );
    };
    approx(real, complex, 2.0);
    approx(complex, quat, 2.0);
    approx(quat, oct, 2.0);
    approx(oct, sed, 2.0);
}

#[test]
fn sparse_ghz_memory_is_width_independent() {
    // Two amplitudes regardless of width: memory stays flat while the dense
    // equivalent doubles every qubit (and stops existing past 32).
    let sim = Simulator::<C64>::new();
    let mut footprints = Vec::new();
    for n in [8usize, 16, 24, 32, 40, 48, 56, 63] {
        let state = sim.run_on("sparse", &library::ghz(n)).unwrap();
        assert_eq!(state.nonzero_count(), 2, "n={n}");
        footprints.push(state.memory_bytes());
    }
    let max = *footprints.iter().max().unwrap();
    let min = *footprints.iter().min().unwrap();
    assert_eq!(
        max, min,
        "sparse GHZ footprint must not grow with width: {footprints:?}"
    );
    assert!(max < 4096, "two-amplitude state should be tiny, got {max}");
}

#[test]
fn sparse_saturates_to_dense_on_uniform_superposition() {
    let sim = Simulator::<C64>::new();
    for n in [6usize, 8, 10] {
        let state = sim.run_on("sparse", &h_layer(n)).unwrap();
        assert_eq!(state.nonzero_count(), 1 << n, "n={n}");
        // Hash-map overhead makes saturated sparse strictly worse than dense.
        let dense = DenseState::<C64>::new(n).unwrap().memory_bytes();
        assert!(
            state.memory_bytes() > dense,
            "saturated sparse ({}) should exceed dense ({})",
            state.memory_bytes(),
            dense
        );
    }
}

#[test]
fn adaptive_tracks_the_cheaper_representation() {
    let sim = Simulator::<C64>::new();
    let n = 12;
    // Concentrated state: adaptive ≈ sparse ≪ dense.
    let ghz = sim.run_on("adaptive", &library::ghz(n)).unwrap();
    let dense_cost = DenseState::<C64>::new(n).unwrap().memory_bytes();
    assert!(
        ghz.memory_bytes() * 100 < dense_cost,
        "GHZ via adaptive should be ≪ dense"
    );
    // Saturated state: adaptive = dense exactly (plus enum wrapper).
    let uniform = sim.run_on("adaptive", &h_layer(n)).unwrap();
    assert!(uniform.memory_bytes() >= dense_cost);
    assert!(uniform.memory_bytes() <= dense_cost + 64);
}

#[test]
fn sparse_tracks_support_size_through_a_circuit() {
    // Support grows only when superposition-creating gates act: X/CX layers
    // keep it at 1; each H at most doubles it; diagonal gates never grow it.
    let mut state = SparseState::<C64>::new(20).unwrap();
    let reg = GateRegistry::<C64>::standard();
    let apply = |st: &mut SparseState<C64>, name: &str, qs: &[usize]| {
        let m = reg.resolve(name).unwrap().matrix(&[]).unwrap();
        st.apply(&m, qs).unwrap();
    };
    for q in 0..20 {
        apply(&mut state, "x", &[q]);
    }
    assert_eq!(state.nonzero_count(), 1);
    apply(&mut state, "h", &[0]);
    assert_eq!(state.nonzero_count(), 2);
    apply(&mut state, "h", &[1]);
    assert_eq!(state.nonzero_count(), 4);
    for q in 2..8 {
        apply(&mut state, "cx", &[1, q]);
    }
    assert_eq!(state.nonzero_count(), 4, "CX must not grow support");
    apply(&mut state, "z", &[3]);
    apply(&mut state, "s", &[4]);
    apply(&mut state, "t", &[5]);
    assert_eq!(
        state.nonzero_count(),
        4,
        "diagonal gates must not grow support"
    );
}

#[test]
fn interference_can_shrink_sparse_support() {
    // H·H = I: support returns to 1, pruning removed the cancelled branch.
    let mut state = SparseState::<C64>::new(4).unwrap();
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    state.apply(&h, &[2]).unwrap();
    assert_eq!(state.nonzero_count(), 2);
    state.apply(&h, &[2]).unwrap();
    assert_eq!(
        state.nonzero_count(),
        1,
        "destructive interference must prune"
    );
}

// ── the structure-storing representations ────────────────────────────

/// The graph-state bundle's footprint is linear in the register, not
/// exponential — and the boundary atlas says so, now that it is a
/// registered backend rather than a description alongside one.
#[test]
fn the_bundle_scales_linearly_on_ghz_and_the_atlas_classifies_it() {
    let sim: Simulator = Simulator::new();
    let mut bytes = Vec::new();
    for n in [8usize, 12, 16, 20, 24] {
        let state = sim.run_on("bundle", &library::ghz(n)).unwrap();
        let bundle = state
            .as_any()
            .downcast_ref::<quantsim::bundle::PolarityBundle>()
            .unwrap();
        // GHZ is a star: exactly n − 1 links, whatever the width.
        assert_eq!(bundle.profile().links, n - 1);
        bytes.push(state.memory_bytes());
    }
    // Doubling the width from 12 to 24 must not square the cost.
    let growth = bytes[4] as f64 / bytes[1] as f64;
    assert!(
        growth < 6.0,
        "bytes {bytes:?} grew by {growth}x from n=12 to 24"
    );
    // And it is dwarfed by dense at the top of the sweep.
    let dense = DenseState::<C64>::new(24).unwrap().memory_bytes();
    assert!(
        bytes[4] * 1000 < dense,
        "bundle {} vs dense {dense}",
        bytes[4]
    );

    // The atlas classifies it sub-exponential, which is the whole point
    // of it being a backend.
    let scan = advantage_scan("ghz", library::ghz, &[8, 10, 12, 14]);
    let axis = scan
        .axes
        .iter()
        .find(|a| a.axis == "bundle")
        .expect("the bundle is an atlas axis");
    let law = axis.law.clone().expect("no width walled");
    assert!(
        !matches!(law, Law::Exponential { .. }),
        "bundle memory law {law:?}"
    );
}

/// An H-layer is the reverse workload: every amplitude is occupied, so
/// support-based representations pay the most and the bundle pays
/// nothing at all — it has no links to store.
#[test]
fn an_h_layer_is_free_for_the_bundle_and_expensive_for_everything_else() {
    let sim: Simulator = Simulator::new();
    let n = 20;
    let mut layer: Circuit = Circuit::new(n);
    for q in 0..n {
        layer.h(q);
    }
    let state = sim.run_on("bundle", &layer).unwrap();
    let bundle = state
        .as_any()
        .downcast_ref::<quantsim::bundle::PolarityBundle>()
        .unwrap();
    assert_eq!(bundle.profile().links, 0, "|+…+⟩ has no entanglement");
    assert_eq!(bundle.profile().components, n);

    let dense = sim.run_on("dense", &layer).unwrap().memory_bytes();
    let sparse = sim.run_on("sparse", &layer).unwrap().memory_bytes();
    assert!(state.memory_bytes() * 1000 < dense);
    assert!(state.memory_bytes() * 1000 < sparse);
    // Sparse is worse than dense here: the support is the whole space.
    assert!(sparse > dense);
}

/// The E8 scale tower is opt-in, because it stores complex amplitudes
/// natively and cannot be generic over the algebra. On GHZ it is two
/// lattice points at any width; on an H-layer it is the whole space.
#[test]
fn the_e8_tower_is_two_lattice_points_on_ghz_at_every_width() {
    let mut sim: Simulator = Simulator::new();
    // Not in `standard()`, and the registrar says why.
    assert!(!sim.backends().contains("e8-constellation"));
    sim.backends_mut().register_e8().unwrap();
    assert!(sim.backends().contains("e8-constellation"));
    assert!(sim.backends().contains("e8-rep"));

    let mut bytes = Vec::new();
    for n in [8usize, 16, 24, 32] {
        let state = sim.run_on("e8-constellation", &library::ghz(n)).unwrap();
        let tower = state
            .as_any()
            .downcast_ref::<quantsim::e8::constellation::E8ConstellationState>()
            .unwrap();
        assert_eq!(tower.stored_points().len(), 2, "GHZ is two points");
        common::assert_close(state.probability(0), 0.5, 1e-12);
        bytes.push(state.memory_bytes());
    }
    // Flat: the same two points however wide the register.
    assert_eq!(bytes[0], bytes[3], "bytes {bytes:?}");

    // And the reverse workload costs it the whole space.
    let n = 16;
    let mut layer: Circuit = Circuit::new(n);
    for q in 0..n {
        layer.h(q);
    }
    let full = sim.run_on("e8-constellation", &layer).unwrap();
    let tower = full
        .as_any()
        .downcast_ref::<quantsim::e8::constellation::E8ConstellationState>()
        .unwrap();
    assert_eq!(tower.stored_points().len(), 1 << n);
    assert!(full.memory_bytes() > 1_000_000);
}
