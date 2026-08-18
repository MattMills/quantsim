//! Up-embedded readout. That relocating each magic event to a fresh
//! wire makes the whole dynamics Clifford, that the readout then costs
//! the observable's *magic adjacency* rather than its cone, that this
//! answers exactly where the cone resolver buys nothing at all — and
//! that none of it knows how wide the register is.

use quantsim::backend::{conjugate_by_step, CliffordStep, PauliString};
use quantsim::gates::Pauli;
use quantsim::prelude::*;
use quantsim::query::{ConeResolver, Inner, Resolver, StateResolver};
use quantsim::recursive::{RecursiveLattice, Shape};
use quantsim::upembed::{self, UpEmbedResolver};

fn word(n: usize, gates: usize, t_share: u64, seed: u64) -> Circuit<C64> {
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for _ in 0..gates {
        let a = (rng.next_u64() % n as u64) as usize;
        let b = (rng.next_u64() % n as u64) as usize;
        match rng.next_u64() % (8 + t_share) {
            0 => c.gate("h", vec![], vec![a]),
            1 => c.gate("s", vec![], vec![a]),
            2 => c.gate("x", vec![], vec![a]),
            3 => c.gate("z", vec![], vec![a]),
            4 => c.gate("y", vec![], vec![a]),
            5 => c.gate("sdg", vec![], vec![a]),
            6 => {
                if a != b {
                    c.gate("cz", vec![], vec![a, b])
                } else {
                    c.gate("h", vec![], vec![a])
                }
            }
            7 => {
                if a != b {
                    c.gate("cx", vec![], vec![a, b])
                } else {
                    c.gate("sx", vec![], vec![a])
                }
            }
            k => {
                if k % 2 == 0 {
                    c.gate("t", vec![], vec![a])
                } else {
                    c.gate("tdg", vec![], vec![a])
                }
            }
        };
    }
    c
}

/// Every pair coupled in every layer — the shape with no locality left
/// for a cone to cut, with magic interleaved between the layers.
fn all_to_all(n: usize, t: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.gate("h", vec![], vec![q]);
    }
    for k in 0..t / 2 {
        c.gate("t", vec![], vec![(k * 3) % n]);
    }
    for a in 0..n {
        for b in (a + 1)..n {
            c.gate("cz", vec![], vec![a, b]);
        }
    }
    for k in 0..t - t / 2 {
        c.gate("t", vec![], vec![(k * 5 + 1) % n]);
    }
    for a in 0..n {
        for b in (a + 1)..n {
            c.gate("cz", vec![], vec![a, b]);
        }
    }
    for q in 0..n {
        c.gate("h", vec![], vec![q]);
    }
    c
}

/// Independent blocks, each with its own magic — disjoint magic
/// neighbourhoods, which is what the factorization actually rewards.
fn blocked(n: usize, block: usize, t_per_block: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for lo in (0..n).step_by(block) {
        let hi = (lo + block).min(n);
        for _ in 0..3 {
            for a in lo..hi {
                for b in (a + 1)..hi {
                    c.gate("cz", vec![], vec![a, b]);
                }
            }
            for q in lo..hi {
                c.gate("h", vec![], vec![q]);
            }
        }
        for k in 0..t_per_block {
            c.gate("t", vec![], vec![lo + k % (hi - lo)]);
        }
    }
    c
}

// ── it is the same expectation ───────────────────────────────────────

#[test]
fn the_up_embedded_readout_agrees_with_dense() {
    let sim: Simulator = Simulator::new();
    let mut worst = 0.0f64;
    for seed in 0..50u64 {
        for n in 2..=4usize {
            let c = word(n, 12, 4, seed * 13 + n as u64);
            let state = sim.run(&c).unwrap();
            for q in 0..n {
                for p in [Pauli::X, Pauli::Y, Pauli::Z] {
                    let want = pauli_expectation(&*state, &[(q, p)]).unwrap();
                    let got = upembed::expectation(&c, &[(q, p)]).unwrap();
                    worst = worst.max((want.re - got.value).abs());
                }
            }
            // a two-site observable, which mixes the line algebra
            let ops = [(0usize, Pauli::Z), (n - 1, Pauli::X)];
            let want = pauli_expectation(&*state, &ops).unwrap();
            let got = upembed::expectation(&c, &ops).unwrap();
            worst = worst.max((want.re - got.value).abs());
        }
    }
    assert!(worst < 1e-12, "worst deviation {worst:e}");
}

#[test]
fn the_wide_transport_agrees_with_the_crates_own_u64_transport() {
    // Two independent implementations of the same Clifford conjugation:
    // this module's unbounded-bitset tableau, and the crate's `u64`
    // `conjugate_by_step`. They must agree wire for wire wherever both
    // can run — that is what licences replacing the bounded one.
    let mut rng = Prng::new(99);
    let mut cases = 0usize;
    for trial in 0..300u64 {
        let n = 3 + (trial as usize % 8);
        let emb = upembed::gadgetize(&word(n, 30, 0, trial)).unwrap();
        for _ in 0..5 {
            let mask = (1u64 << n) - 1;
            let (x, z) = (rng.next_u64() & mask, rng.next_u64() & mask);
            let want = emb.steps().iter().rev().fold(
                PauliString {
                    x,
                    z,
                    negative: false,
                },
                |acc, &s: &CliffordStep| conjugate_by_step(acc, s),
            );
            let got = upembed::debug_transport(x, z, emb.steps());
            assert_eq!(
                (want.x, want.z, want.negative),
                got,
                "trial {trial}: transporting X^{x:b} Z^{z:b}"
            );
            cases += 1;
        }
    }
    assert_eq!(cases, 1500);
}

#[test]
fn an_exactly_zero_expectation_is_decided_rather_than_approximated() {
    // The readout lives in ℤ[ω]/√2^k, so a vanishing expectation comes
    // back as the ring's zero — a decision, not a small double.
    // T|+⟩ points along the X–Y diagonal: ⟨Z⟩ is exactly nothing, and
    // ⟨X⟩ is exactly 1/√2, which is a ring element and not a rounding.
    let mut c = Circuit::new(1);
    c.gate("h", vec![], vec![0]).gate("t", vec![], vec![0]);
    let z = upembed::expectation(&c, &[(0, Pauli::Z)]).unwrap();
    assert!(z.exact.is_zero(), "value was {}", z.value);
    assert_eq!(z.value, 0.0, "not merely small — zero");
    for p in [Pauli::X, Pauli::Y] {
        let r = upembed::expectation(&c, &[(0, p)]).unwrap();
        assert!(!r.exact.is_zero());
        assert!((r.value - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15);
    }

    // And the same decision survives at scale, where the cancellation
    // is between hundreds of contracted lines rather than two.
    let big = upembed::expectation(&all_to_all(12, 6), &[(6, Pauli::X)]).unwrap();
    assert!(big.exact.is_zero(), "value was {}", big.value);
    assert!(
        !upembed::expectation(&all_to_all(12, 6), &[(6, Pauli::Y)])
            .unwrap()
            .exact
            .is_zero(),
        "the same circuit must not vanish along every axis, or the test proves nothing"
    );
}

// ── where the cone buys nothing ──────────────────────────────────────

#[test]
fn where_the_cone_buys_nothing_the_up_embedding_still_does() {
    // The whole reason this module exists. An all-to-all layer puts
    // every qubit in every cone, so `ConeResolver` simulates the full
    // width and reports 1.0× — honestly, but uselessly. Going up a
    // dimension pays the observable's magic adjacency instead, and that
    // does not care that the circuit is nonlocal.
    let reg = GateRegistry::<C64>::standard();
    let (n, t) = (12usize, 6usize);
    let c = all_to_all(n, t);
    let ops = [(n / 2, Pauli::Z)];

    let mut cone = ConeResolver::new(c.clone(), &reg, Inner::Dense);
    let cheap = cone.expectation(&ops).unwrap();
    assert_eq!(cheap.cost.qubits, n, "an all-to-all layer has no outside");
    assert_eq!(
        cheap.cost.compression(),
        1.0,
        "and therefore no compression"
    );

    let mut up = UpEmbedResolver::new(c.clone());
    let a = up.expectation(&ops).unwrap();
    assert!(
        (a.value - cheap.value).abs() < 1e-11,
        "same question, different frame: {} vs {}",
        a.value,
        cheap.value
    );
    assert!(
        a.cost.qubits <= t,
        "the exponent should be the magic, not the width: {}",
        a.cost.qubits
    );
    assert!(
        a.cost.compression() > 60.0,
        "compression only {:.1}x",
        a.cost.compression()
    );
}

#[test]
fn the_cost_does_not_know_how_wide_the_register_is() {
    // Hold the magic fixed, grow the register 16-fold, and the cluster
    // spectrum does not move — while the cone's exponent tracks the
    // width exactly, because on this shape the cone *is* the width.
    let t = 6usize;
    let mut seen = Vec::new();
    for n in [16usize, 32, 64, 128, 256] {
        let r = upembed::expectation(&all_to_all(n, t), &[(n / 2, Pauli::Z)]).unwrap();
        seen.push((n, r.clusters.clone(), r.terms));
    }
    let first = seen[0].clone();
    for (n, clusters, terms) in &seen {
        assert_eq!(*clusters, first.1, "n={n}: the clusters moved");
        assert_eq!(*terms, first.2, "n={n}: the cost moved");
    }
    assert!(
        first.2 <= 1 << t,
        "and the cost is bounded by the magic: {} terms",
        first.2
    );
}

#[test]
fn there_is_no_width_ceiling() {
    // 200 data qubits plus their ancillas — past the crate's `u64`
    // Pauli representation, and 2^200 past a state vector. The claim of
    // this module is that cost stops tracking the width; a module making
    // that claim has no business capping the width.
    let n = 200usize;
    let r = upembed::expectation(&all_to_all(n, 4), &[(100, Pauli::Z)]).unwrap();
    assert!(r.value.abs() <= 1.0 + 1e-12);
    assert!(r.terms <= 16, "{} terms at n={n}", r.terms);
    let emb = upembed::gadgetize(&all_to_all(n, 4)).unwrap();
    assert_eq!(emb.wires(), n + 4);
    assert!(emb.wires() > 64, "the point is to be past the u64 ceiling");
}

// ── the factorization ────────────────────────────────────────────────

#[test]
fn disjoint_magic_neighbourhoods_factorize_the_subset_sum() {
    // The subset sum ranges over 2^t, but `|Φ⟩` is a product state, so
    // it splits over the connected components of the transported lines.
    // When the magic sits in separate neighbourhoods that is the whole
    // difference between a product of small sums and one large one.
    let r = upembed::expectation(&blocked(48, 8, 4), &[(4, Pauli::Z)]).unwrap();
    let t: usize = r.clusters.iter().sum();
    assert!(t >= 20, "the circuit should carry real magic: t = {t}");
    assert!(
        r.clusters.len() > 1,
        "the magic should have split into neighbourhoods: {:?}",
        r.clusters
    );
    assert!(
        r.terms < (1u128 << t) / 100,
        "{} terms against a flat 2^{t}",
        r.terms
    );
    assert!(
        r.factorization_gain() > 100.0,
        "gain only {:.1}x",
        r.factorization_gain()
    );
}

#[test]
fn a_clifford_circuit_needs_no_ancillas_and_one_contraction() {
    let r = upembed::expectation(&word(24, 400, 0, 4), &[(12, Pauli::Z)]).unwrap();
    assert_eq!(r.clusters, vec![0], "no magic, one empty cluster");
    assert_eq!(r.terms, 1, "and a single boundary contraction");
    assert_eq!(r.transports, 1, "only the observable is transported");
}

#[test]
fn only_one_transport_per_ancilla_however_large_the_sum() {
    // The subset sum has 2^t terms but the *transports* are t + 1,
    // because conjugation is multiplicative — that is what keeps the
    // Clifford work linear in the magic.
    let c = all_to_all(16, 8);
    let r = upembed::expectation(&c, &[(8, Pauli::Z)]).unwrap();
    let emb = upembed::gadgetize(&c).unwrap();
    assert_eq!(r.transports, emb.magic() + 1);
    assert!(r.terms >= 1 << 4, "the sum itself is much larger");
}

// ── as a resolver, and the edge ──────────────────────────────────────

#[test]
fn the_resolver_answers_what_the_state_resolver_answers() {
    let reg = GateRegistry::<C64>::standard();
    for &(n, t) in &[(6usize, 4usize), (10, 6), (12, 4)] {
        let c = all_to_all(n, t);
        for q in [0usize, n / 3, n / 2] {
            let ops = [(q, Pauli::Z)];
            let mut state = StateResolver::new(c.clone(), &reg, Inner::Dense);
            let mut up = UpEmbedResolver::new(c.clone());
            let a = state.expectation(&ops).unwrap();
            let b = up.expectation(&ops).unwrap();
            assert!(
                (a.value - b.value).abs() < 1e-11,
                "n={n} q={q}: {} vs {}",
                a.value,
                b.value
            );
            assert_eq!(a.cost.qubits, n, "the state resolver pays the full width");
            assert!(b.cost.qubits <= t, "the up-embedding pays the magic");
        }
    }
}

#[test]
fn a_clifford_perturbation_is_absorbed_and_a_magic_one_is_refused() {
    let reg = GateRegistry::<C64>::standard();
    let c = all_to_all(8, 4);
    let ops = [(4usize, Pauli::Z)];
    let mut up = UpEmbedResolver::new(c.clone());
    let mut state = StateResolver::new(c, &reg, Inner::Dense);

    // A half turn about a single-qubit axis is a Pauli, hence Clifford,
    // hence something this frame simply absorbs.
    let pauli = Rotation {
        theta: std::f64::consts::PI,
        axis: (1u64 << 3, 0),
    };
    let a = up.response(5, 3, pauli, &ops).unwrap();
    let b = state.response(5, 3, pauli, &ops).unwrap();
    assert!(
        (a.value - b.value).abs() < 1e-11,
        "{} vs {}",
        a.value,
        b.value
    );

    // An eighth turn is magic, and the frame says so rather than
    // rounding it into the nearest Clifford.
    let magic = Rotation {
        theta: std::f64::consts::FRAC_PI_4,
        axis: (1u64 << 3, 0),
    };
    match up.response(5, 3, magic, &ops) {
        Err(Error::InvalidState(msg)) => {
            assert!(
                msg.contains("Clifford"),
                "the refusal should say why: {msg}"
            )
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_circuit_outside_clifford_plus_t_is_refused_by_name() {
    let mut c = Circuit::new(3);
    c.gate("h", vec![], vec![0]);
    c.gate("ccz", vec![], vec![0, 1, 2]);
    match upembed::gadgetize(&c) {
        Err(Error::InvalidState(msg)) => assert!(msg.contains("ccz"), "{msg}"),
        other => panic!("expected a refusal, got {other:?}"),
    }

    let mut c = Circuit::new(2);
    c.gate("rz", vec![0.3], vec![0]);
    match upembed::gadgetize(&c) {
        Err(Error::InvalidState(msg)) => assert!(msg.contains("π/4"), "{msg}"),
        other => panic!("expected a refusal, got {other:?}"),
    }

    // Dyadic but finer than π/4 is still outside Clifford+T, and gets
    // its own reason rather than being lumped in with the above.
    let mut c = Circuit::new(2);
    c.gate("rz", vec![std::f64::consts::FRAC_PI_8], vec![0]);
    match upembed::gadgetize(&c) {
        Err(Error::InvalidState(msg)) => assert!(msg.contains("finer"), "{msg}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn an_observable_off_the_register_is_refused() {
    let emb = upembed::gadgetize(&all_to_all(4, 2)).unwrap();
    assert!(upembed::cluster_readout(&emb, &[(4, Pauli::Z)]).is_err());
}

// ── up-embedding magic until the dynamics is Clifford ─────────────────

use quantsim::pathsum;

fn magic_circuit(n: usize, k: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for i in 0..k {
        c.gate("h", vec![], vec![i % n]);
        c.gate("t", vec![], vec![i % n]);
        c.gate("cx", vec![], vec![i % n, (i + 1) % n]);
        c.gate("t", vec![], vec![(i + 1) % n]);
    }
    c
}

fn cancelling(n: usize, k: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for i in 0..k {
        c.gate("t", vec![], vec![i % n]);
        c.gate("cx", vec![], vec![i % n, (i + 1) % n]);
        c.gate("cx", vec![], vec![i % n, (i + 1) % n]);
        c.gate("tdg", vec![], vec![i % n]);
        c.gate("h", vec![], vec![(i + 2) % n]);
    }
    c
}

fn concat(a: &Circuit<C64>, b: &Circuit<C64>) -> Circuit<C64> {
    let mut c = Circuit::new(a.num_qubits().max(b.num_qubits()));
    for src in [a, b] {
        for op in src.ops() {
            if let quantsim::circuit::Op::Named {
                name,
                params,
                qubits,
            } = op
            {
                c.gate(name, params.clone(), qubits.clone());
            }
        }
    }
    c
}

#[test]
fn the_up_embedded_dynamics_is_clifford_by_gate_set_and_has_free_amplitudes() {
    // The dynamics really is Clifford, but the reason is the emitted
    // gate set: `to_circuit` produces nothing but `h`, `s` and `cx`.
    //
    // An earlier version of this test read `h* = 0` from `PathSum` as an
    // independent *certificate* of Clifford-ness. That inference is
    // invalid — h* counts the variables left summed after the output is
    // fixed, so it is the one-amplitude exponent, and a lone `T`, a
    // `CCZ`, and every diagonal circuit sit at h* = 0 while being
    // non-Clifford. See `tests/magic_relocation.rs`, which holds a dense
    // Clifford oracle against exactly those cases. What h* = 0 does say
    // here is true and worth asserting: one amplitude of the up-embedded
    // dynamics costs O(1).
    for c in [
        magic_circuit(3, 4),
        magic_circuit(4, 8),
        magic_circuit(5, 16),
        cancelling(3, 16),
    ] {
        let emb = upembed::gadgetize(&c).unwrap();
        assert!(emb.magic() > 0, "there should be magic to relocate");
        let cliff = emb.to_circuit();
        for op in cliff.ops() {
            let quantsim::circuit::Op::Named { name, .. } = op else {
                panic!("up-embedding emitted a non-registry op")
            };
            assert!(
                matches!(name.as_str(), "h" | "s" | "cx"),
                "up-embedding emitted {name}, outside the Clifford generators \
                 the construction promises"
            );
        }
        assert_eq!(
            pathsum::operator(&cliff).unwrap().internal_vars(),
            0,
            "and one amplitude of the up-embedded dynamics should cost O(1)"
        );
    }
}

#[test]
fn gadgets_drive_the_exponent_down_and_far_fewer_are_needed_than_t() {
    // Spend one wire per magic event and ask a reducer what is left.
    // Full gadgetization always works — it is Clifford by construction —
    // but it says nothing about how many wires were *needed*.
    //
    // What "needed" means here is precise and narrower than it first
    // looks: the count at which ONE AMPLITUDE of the remaining dynamics
    // becomes free. It is not the point at which the remaining unitary
    // becomes Clifford — h* = 0 does not imply that.
    for (c, expect_max) in [(magic_circuit(3, 4), 4usize), (magic_circuit(3, 8), 12)] {
        let t = upembed::magic_events(&c).unwrap();
        let bare = pathsum::operator(&c).unwrap().internal_vars();
        assert!(bare > 0, "this circuit should carry irreducible magic");

        let mut first_zero = None;
        for g in 0..=t {
            let partial = upembed::gadgetize_partial(&c, g).unwrap();
            let h = pathsum::operator(&partial).unwrap().internal_vars();
            if h == 0 {
                first_zero = Some(g);
                break;
            }
        }
        let need = first_zero.expect("full gadgetization must reach a free amplitude");
        assert!(need > 0, "a magic circuit needs at least one gadget");
        assert!(
            need <= expect_max && need < t,
            "needed {need} of {t} gadgets, expected fewer than {expect_max}"
        );
        assert!(
            need >= bare,
            "the exponent {bare} should not exceed the gadgets needed {need}"
        );
    }
}

#[test]
fn a_circuit_whose_magic_cancels_needs_no_gadgets_at_all() {
    // The case that matters. Naive gadgetization spends one wire per T
    // gate; here the magic cancels, so one amplitude is already free and
    // the right number of wires is zero.
    //
    // This family *is* also genuinely Clifford — `tests/magic_relocation.rs`
    // confirms it against a dense oracle — but that is a separate fact
    // about these circuits, not something the h* = 0 below establishes.
    for k in [4usize, 8, 32] {
        let c = cancelling(3, k);
        let t = upembed::magic_events(&c).unwrap();
        assert_eq!(
            t,
            2 * k,
            "the gate list really does carry {} T gates",
            2 * k
        );
        assert_eq!(
            pathsum::operator(&c).unwrap().internal_vars(),
            0,
            "and none of them survives reduction"
        );
        // so gadgetizing nothing already leaves Clifford dynamics
        let none = upembed::gadgetize_partial(&c, 0).unwrap();
        assert_eq!(pathsum::operator(&none).unwrap().internal_vars(), 0);
        assert_eq!(none.num_qubits(), 3, "and no wires were spent");
        // while full gadgetization would have spent one per T gate
        assert_eq!(upembed::gadgetize(&c).unwrap().wires(), 3 + t);
    }
}

#[test]
fn h_star_is_neither_sub_nor_super_additive_under_composition() {
    // The obvious hypothesis — that magic is a resource which adds — is
    // false in BOTH directions, which is why the join has to be reduced
    // rather than estimated from its parts.
    let u = magic_circuit(3, 4);
    let hu = pathsum::operator(&u).unwrap().internal_vars();
    assert!(hu > 0);

    // composing with the inverse CANCELS: h*(U·U†) = 0 < 2·h*(U)
    let mut inv = Circuit::<C64>::new(3);
    for op in u.ops().iter().rev() {
        if let quantsim::circuit::Op::Named {
            name,
            params,
            qubits,
        } = op
        {
            let n = match name.as_str() {
                "t" => "tdg",
                "tdg" => "t",
                "s" => "sdg",
                "sdg" => "s",
                o => o,
            };
            inv.gate(n, params.clone(), qubits.clone());
        }
    }
    let joined = concat(&u, &inv);
    assert_eq!(
        pathsum::operator(&joined).unwrap().internal_vars(),
        0,
        "U·U† must reduce away entirely"
    );

    // composing with ITSELF COMPOUNDS: h*(U·U) > 2·h*(U)
    let doubled = concat(&u, &u);
    let hd = pathsum::operator(&doubled).unwrap().internal_vars();
    assert!(
        hd > 2 * hu,
        "h*(U·U) = {hd} should exceed 2·h*(U) = {}: magic that reduced inside \
         each block stops reducing once the join entangles it",
        2 * hu
    );
}

#[test]
fn h_star_is_an_amplitude_exponent_and_not_a_magic_monotone() {
    // The limit on how far h* can be read as a resource measure. True
    // magic is invariant under composition with a Clifford, so if h*
    // were a magic monotone this would not move. It does: composing
    // with `cancelling` — independently verified Clifford against a
    // dense oracle in `tests/magic_relocation.rs` — raises h*.
    //
    // The reason is simply that h* is not a magic measure at all: it is
    // the exponent for one amplitude, and composition genuinely changes
    // that. Reading it as magic is the mistake this test exists to
    // prevent.
    let u = magic_circuit(3, 4);
    let hu = pathsum::operator(&u).unwrap().internal_vars();
    assert_eq!(hu, 1);

    let v = cancelling(3, 4);
    assert_eq!(
        pathsum::operator(&v).unwrap().internal_vars(),
        0,
        "the right factor must have a free amplitude, or the point does not stand"
    );

    let huv = pathsum::operator(&concat(&u, &v)).unwrap().internal_vars();
    assert!(
        huv > hu,
        "h* {huv} should rise above {hu} even though the right factor is Clifford"
    );
}

// ── the equator: the grid was the π/4 slice all along ────────────────

#[test]
fn the_equatorial_gadget_reproduces_dense_at_any_angle() {
    // Continuous rx/ry/rz/cp at arbitrary angles, each one ancilla
    // after its Clifford quarter-turns peel off — against dense,
    // amplitude-level, on every single-site observable. The cluster
    // spectrum is priced first so the sum is only paid where payable
    // (measured: every one of these fits).
    let sim: Simulator = Simulator::new();
    let mut worst = 0.0f64;
    for n in 3..=5usize {
        for seed in 0..4u64 {
            let c = library::random_circuit(n, 3 * n, seed);
            let emb = upembed::gadgetize_equatorial(&c).unwrap();
            let state = sim.run(&c).unwrap();
            for q in 0..n {
                for p in [Pauli::X, Pauli::Y, Pauli::Z] {
                    let ops = [(q, p)];
                    let spec = upembed::cluster_spectrum(&emb, &ops).unwrap();
                    assert!(
                        spec.first().copied().unwrap_or(0) <= 24,
                        "n={n} seed={seed}: cluster outgrew the measured bound"
                    );
                    let got = upembed::cluster_readout_equatorial(&emb, &ops).unwrap();
                    let want = pauli_expectation(&*state, &ops).unwrap();
                    worst = worst.max((want.re - got.value).abs());
                }
            }
        }
    }
    assert!(worst < 1e-12, "worst deviation {worst:e}");
}

#[test]
fn the_grid_is_the_quarter_slice_of_the_equator() {
    // On a Clifford+T circuit the two readouts are the same physics:
    // same clusters, same value, and both match dense.
    let sim: Simulator = Simulator::new();
    let c = word(3, 14, 3, 41);
    let state = sim.run(&c).unwrap();
    for q in 0..3 {
        let ops = [(q, Pauli::Z)];
        let ex = upembed::expectation(&c, &ops).unwrap();
        let eq = upembed::expectation_equatorial(&c, &ops).unwrap();
        let want = pauli_expectation(&*state, &ops).unwrap().re;
        assert_eq!(ex.clusters, eq.clusters, "same geometry either way");
        assert!((ex.value - eq.value).abs() < 1e-12);
        assert!((eq.value - want).abs() < 1e-12);
    }

    // The ancilla economics differ: a dyadic rz is a T-chain on the
    // grid but a single ancilla on the equator, and both are right.
    let mut d = Circuit::new(2);
    d.h(0)
        .gate("rz", vec![3.0 * std::f64::consts::FRAC_PI_4], vec![0])
        .cx(0, 1);
    let g = upembed::gadgetize(&d).unwrap();
    let ge = upembed::gadgetize_equatorial(&d).unwrap();
    assert_eq!(g.magic(), 3, "3π/4 is three eighths on the grid");
    assert_eq!(ge.magic(), 1, "and one ancilla after the S² peels off");
    // The quarter-turn peel landed the remainder exactly on ∓π/4, so
    // the equatorial embedding is *still grid* — one T† where the
    // T-chain spent three T's — and the exact ring applies to both.
    assert!(g.is_grid() && ge.is_grid());
    let state_d = sim.run(&d).unwrap();
    let want = pauli_expectation(&*state_d, &[(1, Pauli::X)]).unwrap().re;
    let via_grid = upembed::cluster_readout(&g, &[(1, Pauli::X)]).unwrap();
    let via_peel = upembed::cluster_readout(&ge, &[(1, Pauli::X)]).unwrap();
    let via_eq = upembed::cluster_readout_equatorial(&ge, &[(1, Pauli::X)]).unwrap();
    assert!((via_grid.value - want).abs() < 1e-12);
    assert!((via_peel.value - want).abs() < 1e-12);
    assert!((via_eq.value - want).abs() < 1e-12);

    // Refusals stay loud in both directions: the exact readout will
    // not evaluate cos θ it cannot hold, and the grid gadgetizer will
    // not approximate an angle it cannot name.
    let mut cc = Circuit::new(2);
    cc.gate("rz", vec![0.37], vec![0]);
    let emb = upembed::gadgetize_equatorial(&cc).unwrap();
    assert!(!emb.is_grid(), "0.37 rad is genuinely off the grid");
    match upembed::cluster_readout(&emb, &[(0, Pauli::Z)]) {
        Err(Error::InvalidState(msg)) => {
            assert!(
                msg.contains("equatorial"),
                "the refusal should say why: {msg}"
            )
        }
        other => panic!("expected a loud refusal, got {other:?}"),
    }
    assert!(upembed::gadgetize(&cc).is_err());
}

#[test]
fn the_qft_magic_adjacency_is_total_where_the_path_sum_holds_zero() {
    // Two instruments, one circuit, opposite exponents — and both are
    // measurements. The QFT's controlled phases put every ancilla in
    // ONE overlap cluster (3·n(n−1)/2 of them, through the shared
    // wires), so the up-embedded readout prices it at 2^{3n(n−1)/2};
    // the path sum reduces the same operator to h* = 0. Clusters price
    // magic adjacency; h* prices magic structure. The QFT has total
    // adjacency and no structure that survives reduction — which is
    // why the phase field holds it and this frame does not.
    for n in [4usize, 6, 8] {
        let emb = upembed::gadgetize_equatorial(&library::qft(n)).unwrap();
        let t = 3 * n * (n - 1) / 2;
        assert_eq!(emb.magic(), t, "three ancillas per controlled phase");
        let spec = upembed::cluster_spectrum(&emb, &[(0, Pauli::Z)]).unwrap();
        assert_eq!(spec, vec![t], "one blob at width {n}");
        assert_eq!(
            pathsum::operator(&library::qft(n)).unwrap().internal_vars(),
            0,
            "h* = 0 on the same circuit"
        );
    }
    // Where the blob is still payable, the value is right: after the
    // QFT on |0…0⟩ the register is |+…+⟩, so ⟨Z₀⟩ = 0 and ⟨X₀⟩ = 1.
    let emb = upembed::gadgetize_equatorial(&library::qft(4)).unwrap();
    let z = upembed::cluster_readout_equatorial(&emb, &[(0, Pauli::Z)]).unwrap();
    let x = upembed::cluster_readout_equatorial(&emb, &[(0, Pauli::X)]).unwrap();
    assert!(z.value.abs() < 1e-12);
    assert!((x.value - 1.0).abs() < 1e-12);
    assert_eq!(z.max_cluster(), 18);
}

#[test]
fn the_spectrum_is_priced_before_the_sum_is_paid() {
    // cluster_spectrum is t + 1 transports and a union-find — no
    // enumeration — so the exponent of a hopeless readout is measured
    // in microseconds. The random 3n² regime is one blob (108 ancillas
    // at n = 8), and the readout itself refuses past MAX_CLUSTER
    // rather than starting a sum it cannot finish.
    let c = library::random_circuit(8, 192, 3);
    let emb = upembed::gadgetize_equatorial(&c).unwrap();
    let spec = upembed::cluster_spectrum(&emb, &[(0, Pauli::Z)]).unwrap();
    assert_eq!(spec.first().copied().unwrap(), 108, "measured blob size");
    match upembed::cluster_readout_equatorial(&emb, &[(0, Pauli::Z)]) {
        Err(Error::TooManyQubits { requested, max }) => {
            assert_eq!(requested, 108);
            assert_eq!(max, upembed::MAX_CLUSTER);
        }
        other => panic!("expected the cluster wall, got {other:?}"),
    }
}

// ── the coupling topology as a separability axis ─────────────────────

/// H wall, bonds, magic, then `rounds` further Clifford layers over the
/// same bonds. The layers are what let an axis spread at all: `cz` is
/// diagonal, so it commutes with the `Z` the gadget transports and a
/// graph-state circuit alone never widens a cone.
fn topology_circuit(
    n: usize,
    bonds: &[(usize, usize)],
    magic: &[usize],
    rounds: usize,
) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    for &(a, b) in bonds {
        c.gate("cz", [], [a, b]);
    }
    for &p in magic {
        c.t(p);
    }
    for _ in 0..rounds {
        for q in 0..n {
            c.h(q);
        }
        for &(a, b) in bonds {
            c.gate("cz", [], [a, b]);
        }
    }
    c
}

fn readout_terms(n: usize, bonds: &[(usize, usize)], magic: &[usize], rounds: usize) -> u128 {
    let c = topology_circuit(n, bonds, magic, rounds);
    let emb = upembed::gadgetize(&c).unwrap();
    upembed::cluster_readout(&emb, &[(0usize, Pauli::Z)])
        .map(|r| r.terms)
        .unwrap_or(u128::MAX)
}

/// **Separability is a property of the coupling topology, not only of
/// the doping profile.** `dcs_separability` varies *where* the magic
/// sits; this varies only *which pairs the bonds join* — the width, the
/// T-count, the T sites, the bond count and the Clifford depth are all
/// held equal.
///
/// A nested block network keeps its magic clusters bounded as Clifford
/// depth grows: the lattice is regular enough that transported lines
/// re-cancel, so the cluster size oscillates instead of climbing.
/// Random graphs of identical size climb toward saturation, and their
/// readout cost with them — four orders of magnitude apart by depth 6.
///
/// Stated as a tendency, because it is one: the assertion below allows a
/// random control to stay flat (seed 13 does), and requires only that
/// most of them do not.
#[test]
fn the_block_topology_keeps_magic_separable_where_random_coupling_does_not() {
    let lat = RecursiveLattice::nest(Shape::CUBE, 2).unwrap();
    let n = lat.width();
    let nested = lat.bond_pairs();
    let mut magic: Vec<usize> = lat
        .bonds()
        .iter()
        .filter(|b| b.depth == 0)
        .flat_map(|b| [b.a, b.b])
        .collect();
    magic.sort_unstable();
    magic.dedup();
    assert_eq!((n, magic.len()), (64, 24));

    // The structured network stays cheap at every depth measured, and
    // nowhere near the flat 2^24 the T-count alone would predict.
    let flat = 1u128 << magic.len();
    for rounds in [2usize, 4, 6, 8] {
        let terms = readout_terms(n, &nested, &magic, rounds);
        assert!(
            terms <= 128,
            "nested network cost {terms} at depth {rounds}; it is meant to stay bounded"
        );
        assert!(terms * 100_000 < flat, "and far below the flat {flat}");
    }

    // Same width, same bond count, same magic, same depth — random
    // coupling instead. Most seeds blow up.
    let rounds = 6;
    let nested_terms = readout_terms(n, &nested, &magic, rounds);
    let mut blew_up = 0;
    for seed in [7u64, 11, 13, 17] {
        let mut rng = Prng::new(seed);
        let mut bonds: Vec<(usize, usize)> = Vec::with_capacity(nested.len());
        while bonds.len() < nested.len() {
            let a = (rng.next_u64() % n as u64) as usize;
            let b = (rng.next_u64() % n as u64) as usize;
            if a != b {
                bonds.push((a.min(b), a.max(b)));
            }
        }
        if readout_terms(n, &bonds, &magic, rounds) > nested_terms * 1000 {
            blew_up += 1;
        }
    }
    assert!(
        blew_up >= 3,
        "only {blew_up} of 4 random controls exceeded the block network by 1000x; \
         the separation is supposed to be a strong tendency"
    );
}
