//! Mixed-arity compound qudits, measured: the generalized qudit gate
//! relations, conformance against the qubit reference where dimensions
//! coincide, horizontal volumes correlating only on interaction, the
//! swap-class decomposition of mixed fabrics, the E8 root construction
//! with its measured rank obstruction, recorded information flow, and
//! guard-admitted volume merges.

mod common;

use common::assert_close;
use quantsim::e8;
use quantsim::mixed::{clock_d, cshift, fabric, fourier_d, shift_d, swap_dd, CompoundRegister};
use quantsim::prelude::*;

fn mat_mul(d: usize, a: &[C64], b: &[C64]) -> Vec<C64> {
    let mut out = vec![c64(0.0, 0.0); d * d];
    for r in 0..d {
        for c in 0..d {
            let mut acc = c64(0.0, 0.0);
            for k in 0..d {
                acc += a[r * d + k] * b[k * d + c];
            }
            out[r * d + c] = acc;
        }
    }
    out
}

fn mat_close(a: &[C64], b: &[C64], tol: f64) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).norm() < tol)
}

#[test]
fn qudit_gates_satisfy_the_generalized_relations() {
    for d in 2..=5usize {
        // X^d = I, Z^d = I.
        let (x, z) = (shift_d(d), clock_d(d));
        let mut xp = x.clone();
        let mut zp = z.clone();
        for _ in 1..d {
            xp = mat_mul(d, &x, &xp);
            zp = mat_mul(d, &z, &zp);
        }
        let mut identity = vec![c64(0.0, 0.0); d * d];
        for j in 0..d {
            identity[j * d + j] = c64(1.0, 0.0);
        }
        assert!(mat_close(&xp, &identity, 1e-12), "X_{d}^{d} = 1");
        assert!(mat_close(&zp, &identity, 1e-12), "Z_{d}^{d} = 1");

        // The Weyl relation Z X = ω X Z.
        let zx = mat_mul(d, &z, &x);
        let xz = mat_mul(d, &x, &z);
        let angle = std::f64::consts::TAU / d as f64;
        let omega = c64(angle.cos(), angle.sin());
        let scaled: Vec<C64> = xz.iter().map(|&e| omega * e).collect();
        assert!(
            mat_close(&zx, &scaled, 1e-12),
            "Z_{d} X_{d} = ω X_{d} Z_{d}"
        );

        // F diagonalizes the shift: F X F† = Z.
        let f = fourier_d(d);
        let fdag: Vec<C64> = (0..d * d).map(|i| f[(i % d) * d + i / d].conj()).collect();
        let fxf = mat_mul(d, &mat_mul(d, &f, &x), &fdag);
        assert!(mat_close(&fxf, &z, 1e-12), "F X F† = Z at d = {d}");
    }
    // F₂ is exactly the Hadamard.
    let h = std::f64::consts::FRAC_1_SQRT_2;
    assert!(mat_close(
        &fourier_d(2),
        &[c64(h, 0.0), c64(h, 0.0), c64(h, 0.0), c64(-h, 0.0)],
        1e-15
    ));
}

#[test]
fn mixed_registers_conform_to_the_qubit_reference() {
    // All-binary dims replay a qubit circuit exactly: fourier_2 = h,
    // cshift(2,2) = cx, amplitudes agree basis-by-basis.
    let mut mixed = CompoundRegister::new(&[2, 2, 2]).unwrap();
    mixed.apply_1(0, &fourier_d(2)).unwrap();
    mixed.apply_2(0, 1, &cshift(2, 2)).unwrap();
    mixed.apply_2(1, 2, &cshift(2, 2)).unwrap();
    let mut c: Circuit = Circuit::new(3);
    c.h(0).cx(0, 1).cx(1, 2);
    let dense = Simulator::<C64>::new().run(&c).unwrap();
    for idx in 0..8u64 {
        let basis = [
            (idx & 1) as usize,
            ((idx >> 1) & 1) as usize,
            ((idx >> 2) & 1) as usize,
        ];
        let m = mixed.amplitude(&basis).unwrap();
        let q = dense.amplitude(idx);
        assert!((m - q).norm() < 1e-12, "basis {idx}: {m} vs {q}");
    }

    // One quaternary site IS two qubits: fourier_d(4) equals the
    // 2-qubit QFT under the little-endian identification.
    let mut quat = CompoundRegister::new(&[4]).unwrap();
    quat.apply_1(0, &shift_d(4)).unwrap(); // |1⟩: a non-trivial input
    quat.apply_1(0, &fourier_d(4)).unwrap();
    let mut c: Circuit = Circuit::new(2);
    c.x(0);
    let qft = library::qft(2);
    c.append(&qft, &[0, 1]);
    let dense = Simulator::<C64>::new().run(&c).unwrap();
    for j in 0..4u64 {
        let m = quat.amplitude(&[j as usize]).unwrap();
        let q = dense.amplitude(j);
        assert!((m - q).norm() < 1e-12, "level {j}: {m} vs {q}");
    }
}

#[test]
fn volumes_stay_horizontal_until_interaction_correlates_them() {
    // The representational layer: local gates never merge volumes; the
    // first cross-volume gate merges exactly the pair it touches, and
    // the event is recorded.
    let mut reg = CompoundRegister::new(&[2, 3, 4, 5]).unwrap();
    for s in 0..4 {
        let d = reg.dims()[s];
        reg.apply_1(s, &fourier_d(d)).unwrap();
    }
    assert_eq!(reg.volumes(), 4, "local gates keep the horizontal set");
    assert_eq!(reg.largest_volume_states(), 5);
    assert!(reg.merge_timeline().iter().all(|i| !i.merged));

    reg.apply_2(2, 3, &cshift(4, 5)).unwrap();
    assert_eq!(reg.volumes(), 3);
    assert_eq!(reg.largest_volume_states(), 20);
    assert_eq!(
        reg.merge_timeline().iter().filter(|i| i.merged).count(),
        1,
        "exactly one correlation event"
    );
    assert_close(reg.born_weight(), 1.0, 1e-12);

    // Cross-arity Bell pair: binary control, ternary target.
    let mut bell = CompoundRegister::new(&[2, 3]).unwrap();
    bell.apply_1(0, &fourier_d(2)).unwrap();
    bell.apply_2(0, 1, &cshift(2, 3)).unwrap();
    let h = std::f64::consts::FRAC_1_SQRT_2;
    assert_close(bell.amplitude(&[0, 0]).unwrap().re, h, 1e-12);
    assert_close(bell.amplitude(&[1, 1]).unwrap().re, h, 1e-12);
    assert_close(bell.probability(&[1, 0]).unwrap(), 0.0, 1e-12);
    assert_close(bell.probability(&[0, 2]).unwrap(), 0.0, 1e-12);
    let counts = bell.sample(2000, &mut Prng::new(7)).unwrap();
    assert_eq!(counts.len(), 2, "a 2×3 Bell pair has two outcomes");
    assert_eq!(counts.values().sum::<u64>(), 2000);
}

#[test]
fn swap_exists_only_between_equal_arities() {
    // The structural fact: swap is dimensionally well-formed only for
    // equal dims, so a mixed fabric decomposes into swap classes and
    // cross-arity bonds are forced native.
    let dims = [2usize, 3, 3, 2, 5];
    let bonds = [(0usize, 1), (1, 2), (2, 3), (3, 0), (2, 4)];
    let report = fabric(&dims, &bonds).unwrap();
    assert_eq!(report.cross_arity_bonds, 3, "(0,1), (2,3), (2,4)");
    assert!((report.density - 0.5).abs() < 1e-12, "{}", report.density);
    // Equal-dim sub-fabric: {1,2} join (3-level bond); {0,3} join
    // (2-level bond); 4 alone.
    assert!(report.swap_classes.contains(&vec![0, 3]));
    assert!(report.swap_classes.contains(&vec![1, 2]));
    assert!(report.swap_classes.contains(&vec![4]));

    // Applying a mis-sized "swap" across arities is a dimension error,
    // not a semantic one — it cannot even be stated.
    let mut reg = CompoundRegister::new(&[3, 5]).unwrap();
    let err = reg.apply_2(0, 1, &swap_dd(3)).unwrap_err();
    assert!(err.to_string().contains("expected 225"), "{err}");

    // Equal-dim swap works and is its own inverse.
    let mut reg = CompoundRegister::new(&[3, 3]).unwrap();
    reg.apply_1(0, &shift_d(3)).unwrap(); // |1, 0⟩
    reg.apply_2(0, 1, &swap_dd(3)).unwrap();
    assert_close(reg.probability(&[0, 1]).unwrap(), 1.0, 1e-12);
}

#[test]
fn e8_is_constructed_verified_and_obstructed() {
    // The root system is built and checked, not quoted.
    e8::verify().unwrap();

    // su(5) × su(5): two A4 chains, mutually orthogonal, both genuine
    // A-chains — exhibited by search.
    let first = e8::find_a_chain(4, &[]).unwrap();
    let second = e8::find_a_chain(4, &first).unwrap();
    assert!(e8::is_a_chain(&first) && e8::is_a_chain(&second));
    for a in &first {
        for b in &second {
            assert_eq!(e8::dot(a, b), 0, "the two su(5) frames are independent");
        }
    }

    // The rank obstruction, measured: after mutually orthogonal
    // A1 ⊥ A2 ⊥ A3 (rank 6), NO orthogonal A4 exists — the search
    // exhausts. Four arities cannot be mutually independent in E8.
    let mut avoid = Vec::new();
    for k in [1usize, 2, 3] {
        let chain = e8::find_a_chain(k, &avoid).unwrap();
        assert!(e8::is_a_chain(&chain));
        avoid.extend(chain);
    }
    assert!(
        e8::find_a_chain(4, &avoid).is_none(),
        "rank 1+2+3+4 = 10 > 8: independence is impossible, measured"
    );

    // The canonical embedding therefore carries a forced correlation —
    // and only where the geometry demands it: the quaternary and
    // quintary frames share directions, the binary and ternary stay
    // independent of everything.
    let emb = ChainEmbedding::canonical().unwrap();
    for (i, chain) in emb.chains.iter().enumerate() {
        assert_eq!(chain.len(), i + 1);
        assert!(e8::is_a_chain(chain));
    }
    let coupling = emb.coupling();
    let expected = [
        [false, false, false, false],
        [false, false, false, false],
        [false, false, false, true],
        [false, false, true, false],
    ];
    for (row, exp) in coupling.iter().zip(&expected) {
        assert_eq!(row, exp);
    }
    let overlap: i32 = emb.gram(2, 3).iter().flatten().map(|g| g.abs()).sum();
    assert!(
        overlap > 0,
        "the forced correlation is measurable: {overlap}"
    );
}

#[test]
fn flow_is_recorded_and_replayed_as_cones() {
    // dims (2, 3, 4, 5) over the E8 fabric: interact only along the
    // coupled pair, then bridge the binary in — the cones follow the
    // recorded interactions exactly.
    let mut reg = CompoundRegister::new(&[2, 3, 4, 5]).unwrap();
    for s in 0..4 {
        reg.apply_1(s, &fourier_d(reg.dims()[s])).unwrap();
    }
    reg.apply_2(2, 3, &cshift(4, 5)).unwrap();
    let reach = reg.reachable_from(2);
    assert_eq!(reach, vec![false, false, true, true]);
    assert_eq!(reg.reachable_from(0), vec![true, false, false, false]);

    reg.apply_2(0, 2, &cshift(2, 4)).unwrap();
    assert_eq!(reg.volumes(), 2, "the binary joined the correlated volume");
    // Site 0's cone now includes 3 (through the earlier 2–3 bond? No:
    // cones respect ORDER — the 0–2 interaction happened after 2–3, so
    // influence from 0 reaches 2 but nothing propagates 2→3 without a
    // LATER interaction).
    assert_eq!(reg.reachable_from(0), vec![true, false, true, false]);
    // Whereas 3's cone reached everything 2 later touched.
    assert_eq!(reg.reachable_from(3), vec![true, false, true, true]);

    reg.apply_2(2, 3, &cshift(4, 5)).unwrap();
    assert_eq!(
        reg.reachable_from(0),
        vec![true, false, true, true],
        "the later 2–3 interaction extends 0's cone"
    );
}

#[test]
fn volume_merges_are_guard_admitted() {
    // Two volumes of 10⁵ entries each: the merged volume would be
    // 10¹⁰ entries ≈ 240 GB — the merge must refuse with measured
    // numbers, leaving the register intact.
    let mut reg = CompoundRegister::new(&[10; 10]).unwrap();
    for s in 0..10 {
        reg.apply_1(s, &fourier_d(10)).unwrap();
    }
    for half in [0usize, 5] {
        for s in half..half + 4 {
            reg.apply_2(s, s + 1, &cshift(10, 10)).unwrap();
        }
    }
    assert_eq!(reg.volumes(), 2);
    assert_eq!(reg.stored_entries(), 200_000);
    let err = reg.apply_2(4, 5, &cshift(10, 10)).unwrap_err();
    match err {
        Error::OutOfMemory {
            requested,
            available,
            ..
        } => {
            assert!(requested >= 100_000usize.pow(2).saturating_mul(24));
            assert!(available < requested);
        }
        other => panic!("expected a measured refusal, got {other}"),
    }
    assert_eq!(reg.volumes(), 2, "a refused merge changes nothing");
}

#[test]
fn sampling_matches_the_stated_probabilities() {
    // A ternary site in uniform superposition: frequencies track 1/3
    // within statistical tolerance, deterministic under the seed.
    let mut reg = CompoundRegister::new(&[2, 3]).unwrap();
    reg.apply_1(1, &fourier_d(3)).unwrap();
    let counts = reg.sample(6000, &mut Prng::new(41)).unwrap();
    assert_eq!(counts.len(), 3);
    for (outcome, count) in &counts {
        assert_eq!(outcome[0], 0, "the binary site never branched");
        let freq = *count as f64 / 6000.0;
        assert!(
            (freq - 1.0 / 3.0).abs() < 0.03,
            "{outcome:?}: {freq} vs 1/3"
        );
    }
    // Full-superposition compound: mean weight of sampled outcomes
    // matches uniform over 2·3·4·5 = 120 states.
    let mut full = CompoundRegister::new(&[2, 3, 4, 5]).unwrap();
    for s in 0..4 {
        full.apply_1(s, &fourier_d(full.dims()[s])).unwrap();
    }
    let counts = full.sample(6000, &mut Prng::new(43)).unwrap();
    assert!(counts.len() > 100, "most of the 120 outcomes appear");
    for outcome in counts.keys() {
        let p = full.probability(outcome).unwrap();
        assert_close(p, 1.0 / 120.0, 1e-9);
    }
}

#[test]
fn mixed_arity_scaling_is_priced_by_level_count_not_sites() {
    // The representational cost of a volume is its level product: a
    // quintary site carries log2(5) ≈ 2.32 qubits of space — measured
    // as stored entries after full superposition and full correlation.
    let mut reg = CompoundRegister::new(&[5, 5, 5]).unwrap();
    for s in 0..3 {
        reg.apply_1(s, &fourier_d(5)).unwrap();
    }
    reg.apply_2(0, 1, &cshift(5, 5)).unwrap();
    reg.apply_2(1, 2, &cshift(5, 5)).unwrap();
    assert_eq!(reg.volumes(), 1);
    assert_eq!(reg.stored_entries(), 125, "5³ states in one volume");
    assert_eq!(reg.largest_volume_states(), 125);
    // The same construction over mixed dims prices each site at its
    // own arity.
    let mut mixed = CompoundRegister::new(&[2, 3, 4]).unwrap();
    for s in 0..3 {
        mixed.apply_1(s, &fourier_d(mixed.dims()[s])).unwrap();
    }
    mixed.apply_2(0, 1, &cshift(2, 3)).unwrap();
    mixed.apply_2(1, 2, &cshift(3, 4)).unwrap();
    assert_eq!(mixed.stored_entries(), 24, "2·3·4 states");
}

#[test]
fn k_site_gates_and_digit_collapse_work_across_mixed_arities() {
    // apply_k on three sites of DIFFERENT arities equals the site-wise
    // factor product: M₂ ⊗ M₁ ⊗ M₀ applied in one call (sites[0] low)
    // must match applying the three factors separately.
    let dims = [2usize, 3, 4];
    let factors = [fourier_d(2), fourier_d(3), fourier_d(4)];
    let dim = 24usize;
    let mut kron = vec![c64(0.0, 0.0); dim * dim];
    for r in 0..dim {
        for c in 0..dim {
            let (r0, r1, r2) = (r % 2, (r / 2) % 3, r / 6);
            let (c0, c1, c2) = (c % 2, (c / 2) % 3, c / 6);
            kron[r * dim + c] =
                factors[0][r0 * 2 + c0] * factors[1][r1 * 3 + c1] * factors[2][r2 * 4 + c2];
        }
    }
    let mut joint = CompoundRegister::new(&dims).unwrap();
    // Correlate first so the k-site gate acts inside one genuine
    // volume, then apply the joint matrix.
    joint.apply_2(0, 1, &cshift(2, 3)).unwrap();
    joint.apply_2(1, 2, &cshift(3, 4)).unwrap();
    joint.apply_k(&[0, 1, 2], &kron).unwrap();
    let mut site_wise = CompoundRegister::new(&dims).unwrap();
    site_wise.apply_2(0, 1, &cshift(2, 3)).unwrap();
    site_wise.apply_2(1, 2, &cshift(3, 4)).unwrap();
    for (s, f) in factors.iter().enumerate() {
        site_wise.apply_1(s, f).unwrap();
    }
    for a in 0..2 {
        for b in 0..3 {
            for c in 0..4 {
                let x = joint.amplitude(&[a, b, c]).unwrap();
                let y = site_wise.amplitude(&[a, b, c]).unwrap();
                assert!((x - y).norm() < 1e-12, "({a},{b},{c}): {x} vs {y}");
            }
        }
    }
    // One call = one recorded interaction, whatever the width.
    let k_gate = joint.merge_timeline().last().unwrap();
    assert_eq!(k_gate.sites, vec![0, 1, 2]);
    assert!(!k_gate.merged, "sites were already correlated");

    // Digit collapse on a ternary site: F₃|0⟩ has weight 1/3 per
    // digit; projecting digit 1 with renorm √3 renormalizes exactly,
    // drops the other digits, and is logged as a single-site event.
    let mut reg = CompoundRegister::new(&[3]).unwrap();
    reg.apply_1(0, &fourier_d(3)).unwrap();
    reg.project_digit(0, 1, 3.0_f64.sqrt()).unwrap();
    assert_close(reg.born_weight(), 1.0, 1e-12);
    assert_close(reg.probability(&[1]).unwrap(), 1.0, 1e-12);
    assert_close(reg.probability(&[0]).unwrap(), 0.0, 1e-12);
    assert_eq!(reg.merge_timeline().last().unwrap().sites, vec![0]);
    // Out-of-range digits refuse.
    assert!(reg.project_digit(0, 3, 1.0).is_err());
}
