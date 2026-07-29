//! The cross-scale comb codes under displacement noise, measured: the
//! CombCode object reproduces the dual-scale wave's facts, min-norm
//! decoding corrects everything inside its window and fails honestly
//! at the boundary (the mod-2 tie measured as the degenerate limit),
//! and seeded noise trajectories yield the logical-vs-physical error
//! curves — larger comb scales measurably suppress logical failure at
//! the same physical rate, the repo's first fault-tolerance
//! architecture experiment.

mod common;

use quantsim::e8::constellation::{self, CombCode, E8ConstellationState, Point};
use quantsim::prelude::*;

fn max_dev(a: &E8ConstellationState, b: &E8ConstellationState) -> f64 {
    let mut dev: f64 = 0.0;
    a.for_each_nonzero(&mut |i, amp| {
        dev = dev.max((amp - b.amplitude(i)).norm());
    });
    b.for_each_nonzero(&mut |i, amp| {
        dev = dev.max((a.amplitude(i) - amp).norm());
    });
    dev
}

fn displacement(coords: &[i64; 8]) -> Point {
    std::array::from_fn(|k| {
        (0..8)
            .map(|d| coords[d] * constellation::basis(d)[k])
            .sum::<i64>()
    })
}

#[test]
fn the_code_object_reproduces_the_measured_wave() {
    // CombCode(m=4, a=2, w=1) must be exactly the code pinned in
    // tests/e8_dual_scale.rs: codeword +1 on every check, logical X̄
    // flipping logical Z̄ while staying in code space, and the decoder
    // window mod 2^{a−w} = 2.
    let code = CombCode::new(4, 2, 1).unwrap();
    assert_eq!(code.num_qubits(), 32);
    let word = code.codeword().unwrap();
    for d in 0..8 {
        let phase = word
            .modulation_eigenphase(&code.modulation_check(d))
            .unwrap();
        assert!((phase - c64(1.0, 0.0)).norm() < 1e-9, "M-check {d}");
        let mut shifted = code.codeword().unwrap();
        shifted.translate(&code.translation_check(d)).unwrap();
        assert!(max_dev(&shifted, &word) < 1e-12, "T-check {d}");
        assert_eq!(code.logical_readout(&word, d).unwrap(), 0);
        assert_eq!(code.read_displacement(&word, d).unwrap(), 0);
    }
    let mut flipped = code.codeword().unwrap();
    flipped.translate(&code.logical_x(0)).unwrap();
    assert_eq!(code.logical_readout(&flipped, 0).unwrap(), 1);
    assert_eq!(code.logical_readout(&flipped, 3).unwrap(), 0);
    // A logical operation is invisible to every syndrome.
    for d in 0..8 {
        assert_eq!(code.read_displacement(&flipped, d).unwrap(), 0);
    }

    // Parameter validation: w ≤ a < m and the register wall.
    assert!(CombCode::new(4, 4, 1).is_err(), "a < m required");
    assert!(CombCode::new(4, 2, 3).is_err(), "w ≤ a required");
    assert!(CombCode::new(8, 2, 1).is_err(), "64 qubits refuse");
}

#[test]
fn min_norm_decoding_corrects_the_window_and_fails_honestly_at_its_edge() {
    // CombCode(m=5, a=4, w=1): window mod 8, min-norm cell
    // {−3…+4}. Every in-window displacement (mixed directions, mixed
    // signs) is decoded and corrected EXACTLY; a displacement of the
    // window size itself is a logical X̄ — corrected state is clean but
    // the logical value has moved: the honest failure mode, measured.
    let code = CombCode::new(5, 4, 1).unwrap();
    let clean = code.codeword().unwrap();

    let cases: [[i64; 8]; 3] = [
        [1, 0, 0, -2, 0, 0, 3, 0],
        [-3, 1, 0, 0, 2, 0, 0, -1],
        [0, 0, 4, 0, 0, 0, 0, 0], // +4 is the tie: decoded to +4, corrected
    ];
    for coords in &cases {
        let mut noisy = code.codeword().unwrap();
        noisy.translate(&displacement(coords)).unwrap();
        let signed = code.correct(&mut noisy).unwrap();
        for d in 0..8 {
            let m = coords[d].rem_euclid(8);
            let expect = if 2 * m > 8 { m - 8 } else { m };
            assert_eq!(signed[d], expect, "min-norm decode dir {d}");
        }
        assert!(max_dev(&noisy, &clean) < 1e-12, "corrected exactly");
        for d in 0..8 {
            assert_eq!(code.logical_readout(&noisy, d).unwrap(), 0);
        }
    }

    // Window-sized displacement = logical X̄: syndromes silent,
    // correction a no-op, logical moved. Detection cannot see it —
    // that IS the code distance, measured.
    let mut sneaky = code.codeword().unwrap();
    sneaky.translate(&code.logical_x(2)).unwrap();
    let signed = code.correct(&mut sneaky).unwrap();
    assert_eq!(signed, [0; 8], "no syndrome fires");
    assert_eq!(
        code.logical_readout(&sneaky, 2).unwrap(),
        1,
        "logical moved"
    );

    // The degenerate smallest window (a−w = 1, mod-2): a ±1 kick is a
    // TIE — decoded to +1 by the documented break — so half of all
    // ±1 displacements are mis-corrected into a logical flip. Measured
    // on both signs.
    let tiny = CombCode::new(3, 2, 1).unwrap();
    for (kick, expect_logical) in [(1i64, 0i64), (-1, 1)] {
        let mut noisy = tiny.codeword().unwrap();
        let mut coords = [0i64; 8];
        coords[0] = kick;
        noisy.translate(&displacement(&coords)).unwrap();
        tiny.correct(&mut noisy).unwrap();
        assert_eq!(
            tiny.logical_readout(&noisy, 0).unwrap(),
            expect_logical,
            "mod-2 tie: +1 corrected, −1 becomes logical"
        );
    }
}

#[test]
fn noise_trajectories_yield_logical_error_curves() {
    // Seeded displacement noise: per round and direction, six ±1
    // kicks each firing with probability p, syndrome-decode-correct
    // every round, four rounds, then read all eight logicals. The
    // measured curve: the mod-2 code (a=2) fails catastrophically, a=3
    // fails when a round's net kick reaches 2, a=4 only at 4 — logical
    // failure falls with comb scale at every physical rate, and p = 0
    // never fails.
    let trials = 200u64;
    let rounds = 4;
    let rates = [0.0f64, 0.05, 0.15, 0.30];
    let mut failures = [[0u32; 4]; 3]; // [code a−2][rate]
    for (ci, a) in [2usize, 3, 4].iter().enumerate() {
        let code = CombCode::new(a + 1, *a, 1).unwrap();
        for (ri, &p) in rates.iter().enumerate() {
            let mut rng = Prng::new(0xC0DE + *a as u64 * 1000 + ri as u64);
            for _ in 0..trials {
                let mut state = code.codeword().unwrap();
                for _ in 0..rounds {
                    let mut coords = [0i64; 8];
                    for c in coords.iter_mut() {
                        for _ in 0..6 {
                            let r = rng.next_f64();
                            if r < p {
                                *c += 1;
                            } else if r < 2.0 * p {
                                *c -= 1;
                            }
                        }
                    }
                    state.translate(&displacement(&coords)).unwrap();
                    code.correct(&mut state).unwrap();
                }
                let failed = (0..8).any(|d| code.logical_readout(&state, d).unwrap() != 0);
                if failed {
                    failures[ci][ri] += 1;
                }
            }
        }
    }

    // p = 0: no code ever fails.
    for row in &failures {
        assert_eq!(row[0], 0, "zero noise, zero logical failures");
    }
    // The degenerate mod-2 code saturates: ≥ 90% failure at EVERY
    // nonzero rate (the tie makes each odd kick a coin flip, and 32
    // direction-rounds make an odd kick near-certain). a=2 and a=3
    // are statistically indistinguishable where both saturate — the
    // honest reading of independent seeds — so the ordering claims
    // below stick to the measured wide margins.
    for ri in 1..4 {
        assert!(
            failures[0][ri] * 10 >= trials as u32 * 9,
            "degenerate window saturates: {failures:?}"
        );
    }
    // The genuine suppression curve is a=4: strictly rising in p and
    // far below the smaller combs at low and middle rates.
    assert!(
        failures[2][1] < failures[2][2] && failures[2][2] < failures[2][3],
        "a=4 rises with the physical rate: {failures:?}"
    );
    assert!(
        failures[2][1] * 4 < failures[1][1].max(1) * 4 && failures[2][1] + 20 < failures[1][1],
        "a=4 ≪ a=3 at the low rate: {failures:?}"
    );
    assert!(
        failures[2][2] * 4 < failures[1][2],
        "a=4 ≪ a=3 at the middle rate: {failures:?}"
    );
    assert!(
        failures[2][2] * 4 < failures[0][2],
        "a=4 ≪ a=2 at the middle rate: {failures:?}"
    );
}
