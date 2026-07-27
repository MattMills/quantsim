//! The hierarchical (MERA-family) register backend: conformance,
//! coarse-view semantics, refinement, truncation honesty, and the
//! Ball-scalar composition.

mod common;

use common::{assert_close, sim};
use quantsim::backend::{MeraConfig, MeraState};
use quantsim::prelude::*;

#[test]
fn mera_conforms_over_the_full_registry() {
    let report = verify_backend(&sim(), "mera", &ConformanceConfig::default()).unwrap();
    assert!(report.passed(), "{report}");
    assert!(report.max_amplitude_deviation < 1e-9, "{report}");
    assert_eq!(report.sampling_mismatches, 0);
    assert_eq!(report.collapse_violations, 0);
}

#[test]
fn ghz_coarse_view_is_a_maximally_entangled_super_site_pair() {
    // The 16-qubit GHZ seen at depth 1 — the gross register of two
    // 8-qubit super-sites — is a two-super-site maximally entangled pair:
    // bond dimension 2 per site, and the 2×2 coefficient matrix has both
    // singular values 1/√2 (Schmidt degeneracy makes the basis free; the
    // spectrum is the invariant statement).
    let n = 16;
    let state = sim().run_on("mera", &library::ghz(n)).unwrap();
    let mera = state.as_any().downcast_ref::<MeraState<C64>>().unwrap();

    assert!(mera.is_exact(), "GHZ compresses without truncation");
    assert_eq!(mera.max_bond_dimension(), 2, "bond 2 at every level");

    let dims: Vec<usize> = mera.coarse_dims(1).iter().map(|&(_, _, b)| b).collect();
    assert_eq!(dims, vec![2, 2]);
    let (cdims, coeffs) = mera.coarse_state(1).unwrap();
    assert_eq!(cdims, vec![2, 2]);
    assert_eq!(coeffs.len(), 4);
    // Singular values of the 2×2 coarse matrix M[l][r] = coeffs[r*2 + l].
    let m: Vec<C64> = vec![coeffs[0], coeffs[1], coeffs[2], coeffs[3]];
    let gram00 = (m[0] * m[0].conj() + m[2] * m[2].conj()).re;
    let gram11 = (m[1] * m[1].conj() + m[3] * m[3].conj()).re;
    let gram01 = m[0] * m[1].conj() + m[2] * m[3].conj();
    assert_close(gram00, 0.5, 1e-9);
    assert_close(gram11, 0.5, 1e-9);
    assert!(
        gram01.norm() < 1e-9,
        "coarse pair is maximally entangled: off-diagonal {gram01}"
    );

    // Refinement dictionaries expand a super-site one level down; the
    // physical amplitudes at the leaves are the fully refined view.
    let (up, l, r, w) = mera.refine_basis(1, 0).unwrap();
    assert_eq!(up, 2);
    assert_eq!(w.len(), up * l * r);
    assert_close(state.probability(0), 0.5, 1e-9);
    assert_close(state.probability((1 << n) - 1), 0.5, 1e-9);
    assert_eq!(state.nonzero_count(), 2);
}

#[test]
fn coarse_views_refine_from_root_to_leaves() {
    // Depth 0 is the trivial summary; leaf depth is the full state; the
    // total Born weight is the same at every resolution (the isometry
    // tree redistributes description, not weight — exact here since
    // nothing was truncated).
    let mut c = Circuit::new(6);
    c.h(0).cx(0, 1).t(1).cx(1, 2).h(3).cx(3, 4).s(4).cx(4, 5).cx(2, 3);
    let state = sim().run_on("mera", &c).unwrap();
    let mera = state.as_any().downcast_ref::<MeraState<C64>>().unwrap();
    assert!(mera.is_exact());
    for depth in 0..=mera.depth() {
        let (dims, coeffs) = mera.coarse_state(depth).unwrap();
        let weight: f64 = coeffs.iter().map(|a| a.norm_sqr()).sum();
        assert_close(weight, 1.0, 1e-9);
        let prod: usize = dims.iter().product();
        assert_eq!(coeffs.len(), prod);
    }
    // The deepest coarse view has one site per qubit with bond ≤ 2.
    let leaf_dims = mera.coarse_dims(mera.depth());
    assert_eq!(leaf_dims.len(), 6);
    for &(lo, hi, bond) in &leaf_dims {
        assert_eq!(hi - lo, 1);
        assert!(bond <= 2);
    }
}

#[test]
fn truncation_degrades_measurably_never_silently() {
    // A tight bond cap on an entangling circuit must (a) report discarded
    // weight, (b) deviate measurably from dense — never silently.
    let c = library::random_circuit(8, 60, 33);
    let reg = GateRegistry::<C64>::standard();
    let bound = c.bind(&reg).unwrap();

    let mut generous = MeraState::<C64>::with_config(8, MeraConfig::default()).unwrap();
    bound.run(&mut generous).unwrap();
    assert!(generous.is_exact(), "cap 64 is exact at width 8");
    let dense = sim().run(&c).unwrap();
    let dev = max_amplitude_deviation(dense.as_ref(), &generous);
    assert!(dev < 1e-9, "generous cap deviates: {dev}");

    let mut tight = MeraState::<C64>::with_config(
        8,
        MeraConfig {
            max_bond: 2,
            ..MeraConfig::default()
        },
    )
    .unwrap();
    bound.run(&mut tight).unwrap();
    assert!(!tight.is_exact(), "cap 2 must discard on this circuit");
    assert!(tight.discarded_weight() > 1e-6);
    let dev = max_amplitude_deviation(dense.as_ref(), &tight);
    assert!(
        dev > 1e-6,
        "truncation must show up as measured deviation, got {dev}"
    );
}

#[test]
fn gate_cost_is_the_spanning_subtree_and_caps_are_honest() {
    // Gates inside one block never inflate the peak; a register-spanning
    // gate materializes the root block (honest, instrumented); spans past
    // max_block error out.
    let n = 12;
    let mut local = MeraState::<C64>::new(n).unwrap();
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let cx = reg.resolve("cx").unwrap().matrix(&[]).unwrap();
    // Blocks at depth 2 span 3 qubits: gates within [0,3) stay small.
    local.apply(&h, &[0]).unwrap();
    local.apply(&cx, &[0, 1]).unwrap();
    local.apply(&cx, &[1, 2]).unwrap();
    assert!(
        local.peak_block_elements() <= 1 << 3,
        "block-local gates stay block-local: {}",
        local.peak_block_elements()
    );
    // A root-spanning gate pays the register width once.
    local.apply(&cx, &[0, n - 1]).unwrap();
    assert!(local.peak_block_elements() >= 1 << n);

    // Past max_block, the cost is refused, not silently paid.
    let mut wide = MeraState::<C64>::with_config(
        40,
        MeraConfig {
            max_block: 24,
            ..MeraConfig::default()
        },
    )
    .unwrap();
    wide.apply(&h, &[0]).unwrap();
    assert!(matches!(
        wide.apply(&cx, &[0, 39]),
        Err(Error::TooManyQubits {
            requested: 40,
            max: 24
        })
    ));
}

#[test]
fn wide_registers_run_when_entanglement_stays_hierarchical() {
    // Width 40: scrambling within 5-qubit-aligned subtrees only. Dense
    // storage would be 16 TiB; the hierarchy holds it in kilobytes.
    let n = 40;
    let mut state = MeraState::<C64>::new(n).unwrap();
    let reg = GateRegistry::<C64>::standard();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let t = reg.resolve("t").unwrap().matrix(&[]).unwrap();
    let cx = reg.resolve("cx").unwrap().matrix(&[]).unwrap();
    for block in 0..8 {
        let base = block * 5;
        state.apply(&h, &[base]).unwrap();
        state.apply(&cx, &[base, base + 1]).unwrap();
        state.apply(&t, &[base + 1]).unwrap();
        state.apply(&cx, &[base + 1, base + 2]).unwrap();
        state.apply(&cx, &[base + 2, base + 3]).unwrap();
        state.apply(&cx, &[base + 3, base + 4]).unwrap();
    }
    assert_close(state.total_weight(), 1.0, 1e-9);
    assert!(
        state.memory_bytes() < 64 * 1024,
        "hierarchical memory: {} bytes",
        state.memory_bytes()
    );
    assert!(state.peak_block_elements() <= 1 << 5);
    // The coarse view over the eight 5-qubit super-sites is available
    // even though the full state vector could never be materialized.
    let dims = state.coarse_dims(3);
    assert!(dims.len() >= 8);
}

#[test]
fn mera_composes_with_ball_certified_scalars() {
    // The hierarchy over Ball: SVD splits run on midpoints while radii
    // ride along — coarse representation and certified numbers compose.
    let mut c: Circuit<Ball> = Circuit::new(6);
    c.h(0).cx(0, 1).cx(1, 2).t(2).cx(2, 3).cx(3, 4).cx(4, 5);
    let state = Simulator::<Ball>::new().run_on("mera", &c).unwrap();
    assert_close(state.total_weight(), 1.0, 1e-9);
    let a = state.amplitude(0);
    assert!(a.rad >= 0.0 && a.rad < 1e-9);
    assert_close(a.mid.norm_sqr(), 0.5, 1e-9);
}
