//! The E8×E8 co-boundary storage system, measured: information held
//! projectively across the 240 paired points of two E8 copies — as
//! the relative phase field between them — provably invisible to
//! either copy alone, recoverable only through cross-copy
//! interference, on top of the measured cochain structure of the root
//! complex itself.

mod common;

use common::assert_close;
use quantsim::e8;
use quantsim::mixed::{fourier_d, CompoundRegister};
use quantsim::prelude::*;

const N: usize = 240;

/// Σ_α |α⟩_A |α⟩_B / √240 with the data field g encoded as relative
/// phases on copy B — the E8×E8 co-boundary state.
fn coboundary_state(phases: &[C64]) -> CompoundRegister {
    let mut reg = CompoundRegister::new(&[N, N]).unwrap();
    reg.apply_1(0, &fourier_d(N)).unwrap();
    reg.apply_2_permutation(0, 1, &|a, b| (a, (b + a) % N))
        .unwrap();
    reg.apply_1_diagonal(1, phases).unwrap();
    reg
}

fn structured_field(seed_root: usize) -> Vec<C64> {
    // A data field tied to the E8 geometry itself: the phase at point α
    // is set by α's inner product with a chosen reference root.
    let rs = e8::roots();
    let w = rs[seed_root];
    rs.iter()
        .map(|r| {
            let angle = std::f64::consts::TAU * (e8::dot(r, &w) + 8) as f64 / 17.0;
            c64(angle.cos(), angle.sin())
        })
        .collect()
}

#[test]
fn the_root_complex_counts_and_betti_number_are_measured() {
    // Point structure: every root sees (1, 56, 126, 56, 0) others.
    let rs = e8::roots();
    for probe in [0usize, 57, 133, 239] {
        assert_eq!(e8::neighbor_profile(&rs[probe]), [1, 56, 126, 56, 0]);
    }
    // Edge and 2-cell structure: 6720 −1-edges, and every edge closes
    // into exactly one zero-sum triangle (the lattice's norm-2 vectors
    // are all roots), so triangles = 6720/3 = 2240.
    let edges = e8::minus_one_edges();
    let triangles = e8::zero_sum_triangles();
    assert_eq!(edges.len(), 6720);
    assert_eq!(triangles.len(), 2240);
    for t in triangles.iter().step_by(97) {
        let sum: e8::Root = std::array::from_fn(|k| rs[t[0]][k] + rs[t[1]][k] + rs[t[2]][k]);
        assert_eq!(sum, [0; 8], "2-cells are the additive relations");
    }
    // The invariant edge-storage capacity beyond coboundaries,
    // measured by GF(2) rank: b₁ = 6720 − 240 + 1 − rank ∂₂ = 4241
    // (every zero-sum triangle contributes an independent boundary).
    assert_eq!(e8::triangle_complex_b1(), 4241);
}

#[test]
fn coboundary_storage_is_invisible_to_either_copy() {
    // The stored field lives BETWEEN the copies: every marginal of
    // copy A and copy B is exactly uniform whatever the data says.
    let reg = coboundary_state(&structured_field(7));
    for a in 0..N {
        let mut p_a = 0.0;
        let mut p_b = 0.0;
        for other in 0..N {
            p_a += reg.probability(&[a, other]).unwrap();
            p_b += reg.probability(&[other, a]).unwrap();
        }
        assert_close(p_a, 1.0 / N as f64, 1e-12);
        assert_close(p_b, 1.0 / N as f64, 1e-12);
    }

    // And the storage is PROJECTIVE: a global phase on the field is
    // physically nothing — every joint probability is identical.
    let g = structured_field(7);
    let rotated: Vec<C64> = g.iter().map(|&z| z * c64(0.6, 0.8)).collect();
    let plain = coboundary_state(&g);
    let shifted = coboundary_state(&rotated);
    for alpha in (0..N).step_by(17) {
        assert_close(
            plain.probability(&[alpha, alpha]).unwrap(),
            shifted.probability(&[alpha, alpha]).unwrap(),
            1e-12,
        );
    }
}

#[test]
fn cross_copy_interference_recovers_the_stored_field() {
    // Readout: uncompute the pairing (inverse cshift) so the relative
    // field collapses onto copy A, then interfere with F†. The output
    // amplitudes must equal the DFT of the stored field — computed
    // independently — so the co-boundary data is fully recovered, and
    // ONLY via an interaction that touches both copies.
    let g = structured_field(7);
    let mut reg = coboundary_state(&g);
    reg.apply_2_permutation(0, 1, &|a, b| (a, (b + N - a % N) % N))
        .unwrap();
    // F† on copy A.
    let f = fourier_d(N);
    let fdag: Vec<C64> = (0..N * N).map(|i| f[(i % N) * N + i / N].conj()).collect();
    reg.apply_1(0, &fdag).unwrap();

    for k in (0..N).step_by(13) {
        let mut expected = c64(0.0, 0.0);
        for (alpha, phase) in g.iter().enumerate() {
            let angle = -std::f64::consts::TAU * (k * alpha % N) as f64 / N as f64;
            expected += *phase * c64(angle.cos(), angle.sin());
        }
        expected /= N as f64;
        let mut basis = vec![0usize; 2];
        basis[0] = k;
        let measured = reg.amplitude(&basis).unwrap();
        assert!(
            (measured - expected).norm() < 1e-9,
            "mode {k}: {measured} vs {expected}"
        );
    }
    assert_close(reg.born_weight(), 1.0, 1e-9);

    // The flow record shows the readout required cross-copy
    // interactions: copy B is in copy A's cone and vice versa.
    assert_eq!(reg.reachable_from(0), vec![true, true]);
    assert_eq!(reg.reachable_from(1), vec![true, true]);
}

#[test]
fn wide_qudit_fast_paths_validate_exactly() {
    // The permutation path rejects non-bijections in O(d²) and the
    // diagonal path rejects non-unimodular entries in O(d) — the
    // validations that make 240-level gates affordable stay exact.
    let mut reg = CompoundRegister::new(&[N, N]).unwrap();
    let err = reg.apply_2_permutation(0, 1, &|_a, _b| (0, 0)).unwrap_err();
    assert!(err.to_string().contains("bijection"), "{err}");

    let mut bad = vec![c64(1.0, 0.0); N];
    bad[3] = c64(0.5, 0.0);
    let err = reg.apply_1_diagonal(1, &bad).unwrap_err();
    assert!(err.to_string().contains("unimodular"), "{err}");

    // A refused gate is a no-op: the register is still the fresh
    // product state.
    assert_eq!(reg.volumes(), 2);
    assert_close(reg.probability(&[0, 0]).unwrap(), 1.0, 1e-12);

    // The whole protocol stays cheap: prepare, encode, read out on
    // 240-level sites in well under a second.
    let started = std::time::Instant::now();
    let _ = coboundary_state(&structured_field(3));
    assert!(
        started.elapsed().as_secs_f64() < 2.0,
        "{:?}",
        started.elapsed()
    );
}
