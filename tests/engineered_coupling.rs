//! Engineered coupling structure: that the anticommutation partition is
//! the real one, that inert components are exactly droppable, that a
//! Clifford frame puts the live components on disjoint qubits, and that
//! the whole pipeline reproduces dense simulation to machine precision.

use quantsim::backend::{pauli_expectation, CliffordStep};
use quantsim::coupling::*;
use quantsim::heisenberg::*;
use quantsim::prelude::*;

/// Evolve `|0…0⟩` through the rotations on a dense backend.
fn dense_run(rotations: &[Rotation], n: usize) -> DenseState<C64> {
    let mut d = DenseState::<C64>::new(n).unwrap();
    for r in rotations {
        let (m, s) = r.gate().unwrap();
        d.apply(&m, &s).unwrap();
    }
    d
}

/// `⟨ψ| X^x Z^z |ψ⟩` for the **raw** key, matching the convention
/// [`propagate_engineered`] uses for its observable.
fn dense_expectation(d: &DenseState<C64>, key: PauliKey, n: usize) -> f64 {
    let ops: Vec<(usize, Pauli)> = (0..n)
        .filter_map(|q| match (key.0 >> q & 1, key.1 >> q & 1) {
            (1, 0) => Some((q, Pauli::X)),
            (0, 1) => Some((q, Pauli::Z)),
            (1, 1) => Some((q, Pauli::Y)),
            _ => None,
        })
        .collect();
    let herm = pauli_expectation(d as &dyn Backend<C64>, &ops).unwrap();
    (herm / axis_operator_phase(key)).re
}

/// An observable with one Z per block — the only kind that activates
/// more than one block at all, since the walk starts from the
/// observable's own support.
fn one_per_block(k: usize, w: usize) -> PauliKey {
    (0, (0..k).fold(0u64, |m, b| m | 1u64 << (b * w)))
}

/// `k` independent blocks of `w` qubits, each a little TFIM chain — the
/// easy case, where even the support partition finds the structure.
fn blocked(k: usize, w: usize, steps: usize) -> Vec<Rotation> {
    let mut rots = Vec::new();
    for s in 0..steps {
        for b in 0..k {
            let base = b * w;
            for q in 0..w - 1 {
                rots.push(Rotation::rzz(base + q, base + q + 1, 0.3 + 0.05 * s as f64));
            }
            for q in 0..w {
                rots.push(Rotation::rx(base + q, 0.4 - 0.03 * s as f64));
            }
        }
    }
    rots
}

// ── the partition ────────────────────────────────────────────────────

#[test]
fn commuting_axes_never_couple_however_they_overlap() {
    // Every pair of qubits carries a ZZ rotation: all-to-all in the
    // wiring diagram, and a single block to any support-based rule.
    let n = 8;
    let mut rots = Vec::new();
    for a in 0..n {
        for b in (a + 1)..n {
            rots.push(Rotation::rzz(a, b, 0.21 + 0.01 * (a * n + b) as f64));
        }
    }
    let obs = (0u64, 1u64 << 3); // Z_3

    assert_eq!(
        support_blocks(obs, &rots),
        1,
        "the support rule must see one block here — that is the point"
    );

    let c = coupling_of(obs, &rots);
    // Z-type axes all commute, with each other and with Z_3, so nothing
    // in the circuit is live: the observable feels none of it.
    assert_eq!(c.live_components(), 0);
    assert_eq!(
        c.inert_rotations(),
        rots.len(),
        "every rotation commutes with the observable and must be dropped"
    );
    assert!(c.live_rotations(&rots).is_empty());

    let r = propagate_engineered(obs, &rots, n).unwrap();
    let d = dense_run(&rots, n);
    assert!(
        (r.value - dense_expectation(&d, obs, n)).abs() < 1e-12,
        "engineered {} vs dense {}",
        r.value,
        dense_expectation(&d, obs, n)
    );
    assert_eq!(r.inert_rotations, rots.len());
    assert_eq!(r.support_blocks, 1);
}

#[test]
fn inert_components_are_dropped_exactly_not_approximately() {
    // Two disjoint chains; the observable lives on the first. The second
    // chain is arbitrary, non-Clifford, and completely irrelevant.
    let n = 10;
    let mut rots = blocked(2, 5, 3);
    // deliberately interleave so the deletion cannot be positional
    rots.extend(blocked(2, 5, 1));
    let obs = (0u64, 1u64 << 1); // Z_1, in the first chain

    let c = coupling_of(obs, &rots);
    assert!(
        c.inert_rotations() > 0,
        "the second chain must be recognized as inert"
    );

    let engineered = propagate_engineered(obs, &rots, n).unwrap();
    let d = dense_run(&rots, n);
    let reference = dense_expectation(&d, obs, n);
    assert!(
        (engineered.value - reference).abs() < 1e-12,
        "dropping inert rotations changed the answer: {} vs {reference}",
        engineered.value
    );

    // and the same answer comes from simulating only the live half
    let live = c.live_rotations(&rots);
    let dl = dense_run(&live, n);
    assert!(
        (dense_expectation(&dl, obs, n) - reference).abs() < 1e-12,
        "the pruned circuit must give the same expectation"
    );
}

#[test]
fn the_partition_is_invariant_under_a_clifford_scramble() {
    // Conjugating by a Clifford preserves the symplectic form, so the
    // anticommutation partition cannot move — while the supports spread
    // to the whole register and the support partition collapses.
    let (k, w) = (4, 4);
    let n = k * w;
    let rots = blocked(k, w, 2);
    let obs = one_per_block(k, w);

    let plain = coupling_of(obs, &rots);
    assert_eq!(support_blocks(obs, &rots), k, "the easy case, unscrambled");

    let (scr, skey, _) = scramble(&rots, obs, n);
    let scrambled = coupling_of(skey, &scr);

    assert_eq!(
        plain.live_components(),
        scrambled.live_components(),
        "a Clifford cannot change how many independent problems there are"
    );
    assert_eq!(
        support_blocks(skey, &scr),
        1,
        "…while the support rule now sees a single all-to-all block"
    );
    let widths: Vec<usize> = scr.iter().map(|r| r.weight()).collect();
    assert!(
        widths.iter().any(|&w| w >= n / 2),
        "the scramble should produce genuinely wide axes, got {widths:?}"
    );
}

// ── the frame ────────────────────────────────────────────────────────

#[test]
fn the_frame_puts_every_component_on_its_own_qubits() {
    let (k, w) = (4, 4);
    let n = k * w;
    let rots = blocked(k, w, 2);
    let obs = one_per_block(k, w);
    let (scr, skey, _) = scramble(&rots, obs, n);

    let frame = decoupling_frame(skey, &scr, n).unwrap();
    let blocks = frame.blocks();
    assert_eq!(blocks.len(), k, "one block per live component");

    // disjoint, and inside the register
    let mut seen = 0u64;
    for &b in blocks {
        assert_eq!(b & seen, 0, "blocks must be disjoint");
        assert_eq!(b >> n, 0, "blocks must fit the register");
        seen |= b;
    }

    // every framed *axis* lands inside one block. The framed observable
    // need not: it is already a tensor product of single-site factors,
    // so it distributes over the blocks rather than fusing them.
    let framed = frame.rewrite(&scr);
    for key in framed.iter().map(|r| r.axis) {
        let m = key.0 | key.1;
        assert!(m != 0, "an axis collapsed to the identity");
        assert!(
            blocks.iter().any(|&b| m & !b == 0),
            "axis mask {m:#x} straddles the engineered blocks {blocks:x?}"
        );
    }
    let (fkey, _) = frame.conjugate(skey);
    assert!(
        (fkey.0 | fkey.1) & !seen != 0 || (fkey.0 | fkey.1) & seen != 0,
        "the observable has to live somewhere"
    );
}

#[test]
fn the_frame_conjugation_is_a_genuine_clifford_on_the_pauli_group() {
    // The map must preserve the symplectic form and be injective — the
    // two properties the decoupling argument leans on.
    let n = 6;
    let rots = blocked(2, 3, 2);
    let obs = one_per_block(2, 3);
    let (scr, skey, _) = scramble(&rots, obs, n);
    let frame = decoupling_frame(skey, &scr, n).unwrap();

    let keys: Vec<PauliKey> = (0..n)
        .flat_map(|q| [(1u64 << q, 0u64), (0, 1u64 << q), (1u64 << q, 1u64 << q)])
        .collect();
    let images: Vec<PauliKey> = keys.iter().map(|&k| frame.conjugate(k).0).collect();
    for i in 0..keys.len() {
        for j in 0..keys.len() {
            assert_eq!(
                commutes(keys[i], keys[j]),
                commutes(images[i], images[j]),
                "the frame moved the symplectic form on {:?},{:?}",
                keys[i],
                keys[j]
            );
        }
    }
    let mut sorted = images.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), images.len(), "the frame is not injective");
}

#[test]
fn a_trivial_circuit_gets_a_trivial_frame() {
    let rots = vec![Rotation::rz(0, 0.4), Rotation::rz(1, 0.9)];
    let obs = (0u64, 1u64); // Z_0
    let c = coupling_of(obs, &rots);
    assert_eq!(c.live_components(), 0, "both axes commute with Z_0");
    let frame = decoupling_frame(obs, &c.live_rotations(&rots), 2).unwrap();
    assert!(frame.is_empty(), "nothing to steer, {:?}", frame.steps());
    assert_eq!(frame.blocks(), &[1u64], "the observable's own site");
}

// ── the stabilizer contraction ───────────────────────────────────────

#[test]
fn the_identity_frame_reproduces_the_zero_state() {
    let n = 5;
    let frame = DecouplingFrame::identity(n);
    let stab = frame.input_stabilizer();
    assert_eq!(stab.rank(), n);
    // Z-type strings are stabilized, anything with X support is traceless
    assert_eq!(stab.expectation((0, 0b10110)), 1.0);
    assert_eq!(stab.expectation((0b1, 0)), 0.0);
    assert_eq!(stab.expectation((0b1, 0b1)), 0.0);
    assert!(stab.separable(&[0b111], n), "|0…0⟩ is a product state");
}

#[test]
fn separability_detects_an_entangled_input() {
    // A Bell pair across the boundary of a two-block partition is the
    // canonical non-separable case: H on 0 then CX 0→1 gives stabilizers
    // XX and ZZ, neither of which is block-local.
    let n = 2;
    let bell = Stabilizer::new(vec![
        PauliString {
            x: 0b11,
            z: 0,
            negative: false,
        },
        PauliString {
            x: 0,
            z: 0b11,
            negative: false,
        },
    ]);
    assert_eq!(bell.rank(), 2);
    assert!(!bell.separable(&[0b01, 0b10], n));
    assert!(bell.separable(&[0b11], n), "one block holds both halves");
    assert_eq!(bell.expectation((0b11, 0)), 1.0);
    assert_eq!(bell.expectation((0, 0b11)), 1.0);
    // XX·ZZ = −YY, so ⟨YY⟩ = −1
    assert_eq!(bell.expectation((0b11, 0b11)), -1.0);
    assert_eq!(bell.expectation((0b01, 0)), 0.0);
}

// ── the whole pipeline against dense ─────────────────────────────────

#[test]
fn the_engineered_walk_is_exact_on_scrambled_all_to_all_circuits() {
    let mut worst: f64 = 0.0;
    let mut checks = 0usize;
    let mut best_ratio: f64 = 1.0;
    for &(k, w, steps) in &[(2usize, 2usize, 2usize), (2, 3, 2), (3, 2, 3), (2, 4, 1)] {
        let n = k * w;
        let rots = blocked(k, w, steps);
        let spanning = one_per_block(k, w);
        for q in 0..n {
            for obs in [
                (0u64, 1u64 << q),
                (1u64 << q, 0),
                (1u64 << q, 1u64 << q),
                spanning,
            ] {
                let (scr, skey, _) = scramble(&rots, obs, n);
                let r = propagate_engineered(skey, &scr, n).unwrap();
                let d = dense_run(&scr, n);
                let reference = dense_expectation(&d, skey, n);
                worst = worst.max((r.value - reference).abs());
                checks += 1;
                assert_eq!(
                    r.support_blocks, 1,
                    "the scrambled circuit must look fully coupled"
                );
                best_ratio = best_ratio.max(r.factor_saving());
            }
        }
    }
    assert!(checks >= 60, "only {checks} checks");
    assert!(worst < 1e-11, "engineered propagation deviated by {worst:.3e}");
    assert!(
        best_ratio > 1.0,
        "the factorization never paid: best ratio {best_ratio}"
    );
}

#[test]
fn the_engineered_value_matches_the_unscrambled_problem() {
    // The scrambler is built from CX only, so it fixes |0…0⟩ and the two
    // problems have literally the same expectation up to the coefficient
    // the conjugation puts on the observable.
    let (k, w) = (3, 3);
    let n = k * w;
    let rots = blocked(k, w, 2);
    for q in 0..n {
        let obs = (0u64, 1u64 << q);
        let (scr, skey, coeff) = scramble(&rots, obs, n);
        let engineered = propagate_engineered(skey, &scr, n).unwrap();
        let plain = dense_expectation(&dense_run(&rots, n), obs, n);
        let recovered = (C64::new(engineered.value, 0.0) * coeff).re;
        assert!(
            (recovered - plain).abs() < 1e-11,
            "q={q}: engineered {recovered} vs direct {plain}"
        );
    }
}

#[test]
fn the_engineered_walk_agrees_with_the_factored_walk_where_both_apply() {
    // Unscrambled, the two partitions coincide, so the frame must be a
    // no-op on the answer.
    let (k, w) = (3, 4);
    let n = k * w;
    let rots = blocked(k, w, 2);
    for q in 0..n {
        let obs = (0u64, 1u64 << q);
        let a = propagate_factored(obs, &rots).unwrap();
        let b = propagate_engineered(obs, &rots, n).unwrap();
        assert!(
            (a.value - b.value).abs() < 1e-12,
            "q={q}: factored {} vs engineered {}",
            a.value,
            b.value
        );
    }
    // and with an observable that actually reaches every block, both
    // partitions find the same `k`
    let spanning = one_per_block(k, w);
    let a = propagate_factored(spanning, &rots).unwrap();
    let b = propagate_engineered(spanning, &rots, n).unwrap();
    assert_eq!(a.blocks.len(), k);
    assert_eq!(b.engineered_blocks, k);
    assert!((a.value - b.value).abs() < 1e-12);
}

#[test]
fn the_engineered_partition_is_never_coarser_than_the_support_partition() {
    // Anticommutation implies overlapping support, so every engineered
    // block is contained in a support block: the engineered count can
    // only be larger or equal.
    for &(k, w, steps) in &[(2usize, 3usize, 2usize), (3, 3, 1), (4, 2, 2)] {
        let n = k * w;
        let rots = blocked(k, w, steps);
        for obs in (0..n)
            .map(|q| (0u64, 1u64 << q))
            .chain(std::iter::once(one_per_block(k, w)))
        {
            for (label, (rs, key)) in [
                ("plain", (rots.clone(), obs)),
                ("scrambled", {
                    let (s, kk, _) = scramble(&rots, obs, n);
                    (s, kk)
                }),
            ] {
                let r = propagate_engineered(key, &rs, n).unwrap();
                assert!(
                    r.engineered_blocks >= r.support_blocks,
                    "{label} n={n} obs={obs:?}: engineered {} < support {}",
                    r.engineered_blocks,
                    r.support_blocks
                );
            }
        }
    }
}

#[test]
fn the_frame_reports_whether_the_input_still_factors() {
    // The honest accounting: the operator always decouples, the pair
    // (operator, input) does not have to. Whichever way it goes the
    // value must be right, so run both and check the flag is consulted.
    let (k, w) = (3, 3);
    let n = k * w;
    let rots = blocked(k, w, 2);
    let mut separable = 0;
    let mut entangled = 0;
    for q in 0..n {
        let obs = (0u64, one_per_block(k, w).1 | 1u64 << q);
        let (scr, skey, _) = scramble(&rots, obs, n);
        let r = propagate_engineered(skey, &scr, n).unwrap();
        if r.separable_input {
            separable += 1;
        } else {
            entangled += 1;
        }
        let reference = dense_expectation(&dense_run(&scr, n), skey, n);
        assert!(
            (r.value - reference).abs() < 1e-11,
            "q={q} separable={}: {} vs {reference}",
            r.separable_input,
            r.value
        );
    }
    assert_eq!(separable + entangled, n);
}

#[test]
fn the_walk_stays_exact_when_the_circuit_is_not_clifford() {
    // Nothing above depends on the angles; pin that with deliberately
    // irrational ones and a mixed observable.
    let n = 6;
    let mut rots = Vec::new();
    for s in 0..3 {
        for q in 0..n - 1 {
            rots.push(Rotation::rzz(q, q + 1, 0.1 * (s + 1) as f64 + 0.037));
        }
        for q in 0..n {
            rots.push(Rotation::rx(q, 0.234 * (q + 1) as f64));
        }
    }
    let d = dense_run(&rots, n);
    let mut worst: f64 = 0.0;
    for obs in [
        (0u64, 0b000101u64),
        (0b000011u64, 0),
        (0b001100u64, 0b001100u64),
        (0b010000u64, 0b000001u64),
    ] {
        let r = propagate_engineered(obs, &rots, n).unwrap();
        worst = worst.max((r.value - dense_expectation(&d, obs, n)).abs());
    }
    assert!(worst < 1e-11, "deviated by {worst:.3e}");
}

// ── where the cost went ──────────────────────────────────────────────

/// A pseudorandom Clifford scrambler over the full generating set. Unlike
/// [`scramble`] this does **not** fix `|0…0⟩`, which is exactly what
/// makes it the interesting case.
fn random_frame(seed: u64, n: usize, len: usize) -> DecouplingFrame {
    let mut rng = Prng::new(seed);
    let mut steps = Vec::new();
    while steps.len() < len {
        let a = (rng.next_u64() % n as u64) as usize;
        match rng.next_u64() % 3 {
            0 => steps.push(CliffordStep::H(a)),
            1 => steps.push(CliffordStep::S(a)),
            _ => {
                let b = (rng.next_u64() % n as u64) as usize;
                if a != b {
                    steps.push(CliffordStep::Cx(a, b));
                }
            }
        }
    }
    DecouplingFrame::from_steps(steps, n)
}

#[test]
fn a_general_clifford_frame_moves_the_entanglement_into_the_input() {
    // The operator side always decouples. The pair (operator, input)
    // does not: a scrambler that fails to fix |0…0⟩ leaves `V†|0…0⟩`
    // entangled across the very blocks the operator just shed. Both
    // outcomes must produce the right number.
    let (k, w) = (3usize, 3usize);
    let n = k * w;
    let rots = blocked(k, w, 2);
    let obs = one_per_block(k, w);

    let (mut separable, mut entangled) = (0usize, 0usize);
    let mut worst: f64 = 0.0;
    for seed in 0..40u64 {
        let f = random_frame(seed, n, 20);
        let scr = f.rewrite(&rots);
        let (skey, _) = f.conjugate(obs);
        let r = propagate_engineered(skey, &scr, n).unwrap();
        assert_eq!(
            r.engineered_blocks, k,
            "seed {seed}: the operator must decouple regardless of the input"
        );
        if r.separable_input {
            separable += 1;
        } else {
            entangled += 1;
        }
        let reference = dense_expectation(&dense_run(&scr, n), skey, n);
        worst = worst.max((r.value - reference).abs());
    }
    assert!(worst < 1e-11, "deviated by {worst:.3e}");
    assert!(
        entangled > 0,
        "the entangled-input branch was never exercised"
    );
    assert!(
        separable > 0,
        "the product branch was never exercised: {separable}/{entangled}"
    );
}

#[test]
fn a_cx_only_frame_always_leaves_the_input_factored() {
    // The complementary fact, and the reason `scramble` is built from CX:
    // a Clifford that fixes |0…0⟩ cannot have put entanglement there.
    for &(k, w) in &[(2usize, 3usize), (3, 3), (4, 2)] {
        let n = k * w;
        let rots = blocked(k, w, 2);
        let obs = one_per_block(k, w);
        let (scr, skey, _) = scramble(&rots, obs, n);
        let r = propagate_engineered(skey, &scr, n).unwrap();
        assert!(r.separable_input, "k={k} w={w}");
        assert_eq!(r.engineered_blocks, k);
        assert_eq!(r.support_blocks, 1);
    }
}

#[test]
fn an_invented_partner_is_orthogonal_to_everything_it_should_be() {
    // The property the radical rounds depend on: without it, one
    // component's pivot becomes visible to another and the blocks merge.
    let (k, w) = (4usize, 4usize);
    let n = k * w;
    let rots = blocked(k, w, 3);
    let obs = one_per_block(k, w);
    let (scr, skey, _) = scramble(&rots, obs, n);
    let r = propagate_engineered(skey, &scr, n).unwrap();
    assert_eq!(
        r.engineered_blocks, k,
        "radical rounds leaked: {} blocks",
        r.engineered_blocks
    );
    assert_eq!(r.support_blocks, 1);
    assert!(
        r.peak_flat > 50 * r.peak_stored as u128,
        "flat {} vs stored {}",
        r.peak_flat,
        r.peak_stored
    );
}

#[test]
fn the_saving_grows_with_the_block_count_not_the_width() {
    // Stored terms are the sum over blocks and the flat count is the
    // product, so adding an independent block must move one linearly and
    // the other geometrically.
    let mut rows = Vec::new();
    for k in 2..=5usize {
        let n = k * 4;
        let rots = blocked(k, 4, 3);
        let obs = one_per_block(k, 4);
        let (scr, skey, _) = scramble(&rots, obs, n);
        let r = propagate_engineered(skey, &scr, n).unwrap();
        assert_eq!(r.engineered_blocks, k);
        rows.push((k, r.peak_stored, r.peak_flat));
    }
    for pair in rows.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        let stored_ratio = b.1 as f64 / a.1 as f64;
        let flat_ratio = b.2 as f64 / a.2 as f64;
        assert!(
            stored_ratio < 2.0,
            "stored grew {stored_ratio:.2}× from k={} to k={}",
            a.0,
            b.0
        );
        assert!(
            flat_ratio > 2.0,
            "flat grew only {flat_ratio:.2}× from k={} to k={}",
            a.0,
            b.0
        );
    }
}
