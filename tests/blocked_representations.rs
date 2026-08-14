//! One representation per engineered block: that each block really is an
//! independent problem, that any backend can be pointed at it, that they
//! all agree with dense, and that the cheapest choice differs per block.

use quantsim::backend::{pauli_expectation, CliffordStep};
use quantsim::blocks::*;
use quantsim::coupling::*;
use quantsim::heisenberg::*;
use quantsim::prelude::*;

fn dense_run(rotations: &[Rotation], n: usize) -> DenseState<C64> {
    let mut d = DenseState::<C64>::new(n).unwrap();
    for r in rotations {
        let (m, s) = r.gate().unwrap();
        d.apply(&m, &s).unwrap();
    }
    d
}

/// `⟨ψ| X^x Z^z |ψ⟩` for the raw key — the convention the propagators use.
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

/// `k` independent TFIM chains of `w` qubits.
fn blocked(k: usize, w: usize, steps: usize) -> Vec<Rotation> {
    let mut rots = Vec::new();
    for s in 0..steps {
        for b in 0..k {
            let base = b * w;
            for q in 0..w - 1 {
                rots.push(Rotation::rzz(base + q, base + q + 1, 0.3 + 0.05 * s as f64));
            }
            for q in 0..w {
                rots.push(Rotation::rx(base + q, 0.44 - 0.03 * s as f64));
            }
        }
    }
    rots
}

/// One Z per block, so the observable actually reaches all of them.
fn one_per_block(k: usize, w: usize) -> PauliKey {
    (0, (0..k).fold(0u64, |m, b| m | 1u64 << (b * w + 1)))
}

// ── preparing a block's share of the input ───────────────────────────

#[test]
fn a_preparation_circuit_really_makes_the_state_its_generators_describe() {
    // Build a stabilizer state by running a Clifford, read its group off
    // the frame, synthesize a preparation from that group alone, and
    // check every generator is stabilized by what comes out.
    let n = 5;
    for seed in 0..12u64 {
        let mut rng = Prng::new(seed);
        let mut steps = Vec::new();
        while steps.len() < 14 {
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
        let frame = DecouplingFrame::from_steps(steps, n);
        let stab = frame.input_stabilizer();
        let gens = stab.block_generators((1u64 << n) - 1);
        assert_eq!(
            gens.len(),
            n,
            "seed {seed}: a full-width group has n generators"
        );

        let prep = preparation(&gens, n).unwrap();
        let mut state = DenseState::<C64>::new(n).unwrap();
        prep.apply(&mut state).unwrap();

        for g in &gens {
            let ops: Vec<(usize, Pauli)> = (0..n)
                .filter_map(|q| match (g.x >> q & 1, g.z >> q & 1) {
                    (1, 0) => Some((q, Pauli::X)),
                    (0, 1) => Some((q, Pauli::Z)),
                    (1, 1) => Some((q, Pauli::Y)),
                    _ => None,
                })
                .collect();
            let want = if g.negative { -1.0 } else { 1.0 };
            let got = pauli_expectation(&state as &dyn Backend<C64>, &ops)
                .unwrap()
                .re;
            assert!(
                (got - want).abs() < 1e-12,
                "seed {seed}: generator {g:?} has expectation {got}, wanted {want}"
            );
        }
    }
}

#[test]
fn a_dependent_generating_set_is_refused_rather_than_half_prepared() {
    let g = PauliString {
        x: 0,
        z: 0b011,
        negative: false,
    };
    // g, g: the second is dependent on the first.
    assert!(preparation(&[g, g], 3).is_err());
}

// ── every representation, same answer ────────────────────────────────

#[test]
fn every_block_solver_reproduces_dense_on_a_scrambled_circuit() {
    let mut worst: f64 = 0.0;
    let mut checks = 0usize;
    for &(k, w) in &[(2usize, 3usize), (3, 3), (2, 4), (3, 4)] {
        let n = k * w;
        let rots = blocked(k, w, 3);
        let obs = one_per_block(k, w);
        let (scr, skey, _) = scramble(&rots, obs, n);
        let reference = dense_expectation(&dense_run(&scr, n), skey, n);
        for solver in [
            BlockSolver::Pauli,
            BlockSolver::Dense,
            BlockSolver::Sparse,
            BlockSolver::Mps { max_bond: 64 },
        ] {
            let r = propagate_blocked(skey, &scr, n, solver).unwrap();
            assert_eq!(r.blocks.len(), k, "{}: wrong block count", solver.label());
            assert!(r.blocks.iter().all(|b| b.solver == solver));
            worst = worst.max((r.value - reference).abs());
            checks += 1;
        }
    }
    assert!(checks >= 16, "only {checks} checks");
    assert!(worst < 1e-11, "a block solver deviated by {worst:.3e}");
}

#[test]
fn the_blocked_product_agrees_with_the_engineered_pauli_walk() {
    for &(k, w) in &[(2usize, 4usize), (3, 3)] {
        let n = k * w;
        let rots = blocked(k, w, 4);
        let obs = one_per_block(k, w);
        let (scr, skey, _) = scramble(&rots, obs, n);
        let engineered = propagate_engineered(skey, &scr, n).unwrap();
        let blocked = propagate_blocked(skey, &scr, n, BlockSolver::Dense).unwrap();
        assert!(
            (engineered.value - blocked.value).abs() < 1e-11,
            "k={k} w={w}: engineered {} vs blocked {}",
            engineered.value,
            blocked.value
        );
        assert_eq!(engineered.engineered_blocks, blocked.blocks.len());
    }
}

#[test]
fn the_blocks_carry_only_their_own_qubits_and_their_own_gates() {
    let (k, w) = (3usize, 4usize);
    let n = k * w;
    let rots = blocked(k, w, 3);
    let obs = one_per_block(k, w);
    let (scr, skey, _) = scramble(&rots, obs, n);
    let r = propagate_blocked(skey, &scr, n, BlockSolver::Dense).unwrap();

    let mut seen = 0u64;
    let mut rotations = 0usize;
    for b in &r.blocks {
        assert_eq!(b.mask & seen, 0, "blocks must be disjoint");
        seen |= b.mask;
        assert_eq!(b.qubits, b.mask.count_ones() as usize);
        assert_eq!(b.qubits, w, "each block should be one original chain");
        rotations += b.rotations;
    }
    assert_eq!(
        rotations,
        scr.len(),
        "every gate must land in exactly one block"
    );
    assert_eq!(r.widest_block, w);
    // and the product of the per-block values is the answer
    let product: f64 = r.blocks.iter().map(|b| b.value).product();
    assert!((product - r.value).abs() < 1e-12);
}

// ── the point: the frame narrows the gates too ───────────────────────

#[test]
fn the_frame_restores_the_gate_widths_the_scrambler_destroyed() {
    // The scrambler makes every axis wide — wide enough that a windowed
    // backend refuses the circuit outright. The frame puts the widths
    // back where they started, which is why an MPS can be pointed at a
    // block at all.
    for &(k, w) in &[(3usize, 4usize), (4, 5), (4, 8)] {
        let n = k * w;
        let rots = blocked(k, w, 3);
        let obs = one_per_block(k, w);
        let lab_max = rots.iter().map(|r| r.weight()).max().unwrap();
        assert_eq!(lab_max, 2, "the lab circuit is two-local");

        let (scr, skey, _) = scramble(&rots, obs, n);
        let scrambled_max = scr.iter().map(|r| r.weight()).max().unwrap();
        assert!(
            scrambled_max > quantsim::backend::MPS_MAX_WINDOW,
            "n={n}: the scramble must exceed what a windowed backend accepts, got {scrambled_max}"
        );

        let frame = decoupling_frame(skey, &scr, n).unwrap();
        let framed_max = frame
            .rewrite(&scr)
            .iter()
            .map(|r| r.weight())
            .max()
            .unwrap();
        assert_eq!(
            framed_max, lab_max,
            "n={n}: the frame should recover the original two-locality, got {framed_max}"
        );
    }
}

// ── the choice is per block ──────────────────────────────────────────

/// One wide-but-shallow block and one narrow-but-deep block: no single
/// representation suits both.
fn heterogeneous(wide: usize, deep_w: usize, shallow: usize, deep: usize) -> Vec<Rotation> {
    let mut rots = Vec::new();
    for s in 0..shallow {
        for q in 0..wide - 1 {
            rots.push(Rotation::rzz(q, q + 1, 0.3 + 0.02 * s as f64));
        }
        for q in 0..wide {
            rots.push(Rotation::rx(q, 0.44));
        }
    }
    for s in 0..deep {
        for q in 0..deep_w - 1 {
            rots.push(Rotation::rzz(
                wide + q,
                wide + q + 1,
                0.37 + 0.01 * s as f64,
            ));
        }
        for q in 0..deep_w {
            rots.push(Rotation::rx(wide + q, 0.29 + 0.005 * s as f64));
        }
    }
    rots
}

#[test]
fn choosing_per_block_beats_every_uniform_choice() {
    let (wide, deep_w, shallow, deep) = (12usize, 5usize, 2usize, 24usize);
    let n = wide + deep_w;
    let rots = heterogeneous(wide, deep_w, shallow, deep);
    let obs = (0u64, (1u64 << 1) | (1u64 << (wide + 1)));
    let (scr, skey, _) = scramble(&rots, obs, n);

    let uniform_pauli = propagate_blocked(skey, &scr, n, BlockSolver::Pauli).unwrap();
    let uniform_dense = propagate_blocked(skey, &scr, n, BlockSolver::Dense).unwrap();
    // A narrow block's whole Hilbert space is smaller than a deep walk's
    // Pauli sum; a wide block's is not. That is the entire policy.
    let mixed = propagate_blocked_with(skey, &scr, n, |_, mask, rots| {
        if mask.count_ones() as usize <= 8 && rots > 40 {
            BlockSolver::Dense
        } else {
            BlockSolver::Pauli
        }
    })
    .unwrap();

    for other in [&uniform_pauli, &uniform_dense] {
        assert!(
            (mixed.value - other.value).abs() < 1e-10,
            "the representations must agree: {} vs {}",
            mixed.value,
            other.value
        );
    }
    assert!(
        mixed.total_bytes() < uniform_pauli.total_bytes(),
        "mixed {} vs uniform pauli {}",
        mixed.total_bytes(),
        uniform_pauli.total_bytes()
    );
    // The gap against uniform dense is `2^wide`, so the constant here
    // tracks the test's width and not the method; it is the direction
    // that is the claim.
    assert!(
        mixed.total_bytes() * 20 < uniform_dense.total_bytes(),
        "mixed {} vs uniform dense {}",
        mixed.total_bytes(),
        uniform_dense.total_bytes()
    );
    // and it really did use two different representations
    let solvers: Vec<BlockSolver> = mixed.blocks.iter().map(|b| b.solver).collect();
    assert!(
        solvers.contains(&BlockSolver::Pauli) && solvers.contains(&BlockSolver::Dense),
        "{solvers:?}"
    );
}

// ── past what anything global can hold ───────────────────────────────

#[test]
fn three_representations_agree_where_no_global_state_vector_fits() {
    // 40 qubits: 2^40 amplitudes is 16 TiB, and the scrambled circuit's
    // axes are far too wide for any windowed backend to accept. Blocked,
    // it is four independent ten-qubit problems, and three unrelated
    // representations of them agree.
    let (k, w) = (4usize, 10usize);
    let n = k * w;
    assert!(DenseState::<C64>::new(n).is_err(), "dense must refuse {n}");

    let rots = blocked(k, w, 3);
    let obs = one_per_block(k, w);
    let (scr, skey, _) = scramble(&rots, obs, n);

    let p = propagate_blocked(skey, &scr, n, BlockSolver::Pauli).unwrap();
    let d = propagate_blocked(skey, &scr, n, BlockSolver::Dense).unwrap();
    let m = propagate_blocked(skey, &scr, n, BlockSolver::Mps { max_bond: 64 }).unwrap();

    assert!(
        (p.value - d.value).abs() < 1e-11,
        "{} vs {}",
        p.value,
        d.value
    );
    assert!(
        (m.value - d.value).abs() < 1e-11,
        "{} vs {}",
        m.value,
        d.value
    );
    assert_eq!(p.blocks.len(), k);
    assert_eq!(p.widest_block, w);
    // the whole computation held less than a single 24-qubit vector would
    assert!(
        d.total_bytes() < (1usize << 24) * 16,
        "blocked dense held {} bytes",
        d.total_bytes()
    );
}

#[test]
fn an_entangled_framed_input_is_refused_rather_than_multiplied_out() {
    // A scrambler drawn from the full Clifford generating set does not
    // fix |0…0⟩, so the frame can leave the input entangled across the
    // blocks — and then there is no per-block problem to pose.
    let (k, w) = (3usize, 3usize);
    let n = k * w;
    let rots = blocked(k, w, 2);
    let obs = one_per_block(k, w);

    let mut refused = 0usize;
    let mut posed = 0usize;
    for seed in 0..40u64 {
        let mut rng = Prng::new(seed);
        let mut steps = Vec::new();
        while steps.len() < 20 {
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
        let f = DecouplingFrame::from_steps(steps, n);
        let scr = f.rewrite(&rots);
        let (skey, _) = f.conjugate(obs);
        let engineered = propagate_engineered(skey, &scr, n).unwrap();
        match propagate_blocked(skey, &scr, n, BlockSolver::Dense) {
            Ok(b) => {
                posed += 1;
                assert!(
                    engineered.separable_input,
                    "seed {seed}: posed a per-block problem on an entangled input"
                );
                assert!((b.value - engineered.value).abs() < 1e-11);
            }
            Err(e) => {
                refused += 1;
                assert!(
                    !engineered.separable_input,
                    "seed {seed}: refused a separable input"
                );
                assert!(format!("{e}").contains("entangled"), "{e}");
            }
        }
    }
    assert!(refused > 0, "the refusal branch was never exercised");
    assert!(
        posed > 0,
        "the product branch was never exercised: {posed}/{refused}"
    );
}
