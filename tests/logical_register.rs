//! The constraint-compression backend: the code space as a
//! representation bet. That encoded computations run at logical cost
//! with amplitude-exact reads, that faults and their syndromes live in
//! the frame with no state work, that everything outside the
//! vocabulary escapes honestly — and that the compression is measured,
//! not asserted.

use quantsim::backend::{Backend, DenseState, SparseState};
use quantsim::logical::LogicalState;
use quantsim::prelude::*;
use quantsim::retro::{syndrome_bits, ToricCode};

fn run(c: &Circuit<C64>, state: &mut dyn Backend<C64>, reg: &GateRegistry<C64>) {
    c.bind(reg).unwrap().run(state).unwrap();
}

#[test]
fn encoded_computations_are_amplitude_exact_at_logical_cost() {
    let reg = GateRegistry::<C64>::standard();
    for seed in 0..5u64 {
        let c = library::logical_random::<C64>(2, 25, seed);
        let mut log = LogicalState::tiled(16).unwrap();
        run(&c, &mut log, &reg);
        let mut dense = DenseState::<C64>::new(16).unwrap();
        run(&c, &mut dense, &reg);
        assert!(log.is_logical(), "seed {seed}: the bet must hold");
        assert_eq!(log.stats().escapes, 0);
        let mut dev = 0.0f64;
        for i in 0..(1u64 << 16) {
            let d = dense.amplitude(i) - log.amplitude(i);
            dev = dev.max((d.re * d.re + d.im * d.im).sqrt());
        }
        assert!(dev < 1e-12, "seed {seed}: dev {dev:e}");
        // The compression is real: the inner register is 4 wires.
        assert_eq!(log.logical_qubits(), 4);
        assert!(
            log.memory_bytes() < dense.memory_bytes() / 100,
            "logical {} B vs dense {} B",
            log.memory_bytes(),
            dense.memory_bytes()
        );
    }
}

#[test]
fn faults_and_syndromes_live_in_the_frame_without_state_work() {
    // A physical Pauli fault is one frame update; its syndrome is the
    // frame's signature — read with no amplitudes touched — and it
    // agrees with the state-level reading on a materialized copy.
    let reg = GateRegistry::<C64>::standard();
    let c = library::logical_random::<C64>(2, 20, 3);
    let mut log = LogicalState::tiled(16).unwrap();
    run(&c, &mut log, &reg);
    let mut fault = Circuit::<C64>::new(16);
    fault.gate("y", vec![], vec![9]);
    run(&fault, &mut log, &reg);
    assert!(log.is_logical(), "a fault is not an escape");

    let node_b = ToricCode::new(2, 8).unwrap();
    let node_a = ToricCode::new(2, 0).unwrap();
    let from_frame_b = log.frame_syndrome(&node_b);
    let from_frame_a = log.frame_syndrome(&node_a);
    assert!(from_frame_b.iter().any(|&s| s), "the fault fired node B");
    assert!(from_frame_a.iter().all(|&s| !s), "node A stayed clean");

    // The state-level reading agrees exactly.
    let mut flat = SparseState::<C64>::new(16).unwrap();
    log.for_each_nonzero(&mut |i, a| {
        let _ = (i, a);
    });
    let mut entries = Vec::new();
    log.for_each_nonzero(&mut |i, a| entries.push((i, a)));
    flat.load(&entries).unwrap();
    assert_eq!(syndrome_bits(&flat, &node_b, 1e-9).unwrap(), from_frame_b);

    // Retrocorrection in bookkeeping space: absorb the correction and
    // the syndrome clears — still no amplitudes touched.
    log.correct(quantsim::backend::PauliString {
        x: 1 << 9,
        z: 1 << 9,
        negative: false,
    });
    assert!(log.frame_syndrome(&node_b).iter().all(|&s| !s));
}

#[test]
fn outside_the_vocabulary_the_escape_is_honest() {
    let reg = GateRegistry::<C64>::standard();
    let c = library::logical_random::<C64>(2, 10, 1);
    let mut log = LogicalState::tiled(16).unwrap();
    run(&c, &mut log, &reg);
    let mut extra = Circuit::<C64>::new(16);
    extra.t(3).h(3);
    run(&extra, &mut log, &reg);
    assert!(!log.is_logical(), "a t gate is outside the bet");
    assert_eq!(log.stats().escapes, 1);
    // Still exact after materialization.
    let mut dense = DenseState::<C64>::new(16).unwrap();
    let mut full = c.clone();
    full.t(3).h(3);
    run(&full, &mut dense, &reg);
    let mut dev = 0.0f64;
    for i in 0..(1u64 << 16) {
        let d = dense.amplitude(i) - log.amplitude(i);
        dev = dev.max((d.re * d.re + d.im * d.im).sqrt());
    }
    assert!(dev < 1e-12, "dev {dev:e}");
}

#[test]
fn the_compression_scales_with_nodes_not_qubits() {
    // The measured claim: over 1..4 nodes (8..32 physical qubits), the
    // logical register's memory tracks the 2·nodes logical wires while
    // a flat register tracks the physical support. Dense at 32 qubits
    // is 64 GiB and cannot even be asked.
    let reg = GateRegistry::<C64>::standard();
    let mut logical_bytes = Vec::new();
    let mut sparse_bytes = Vec::new();
    for m in 1..=4usize {
        let c = library::logical_random::<C64>(m, 15 * m, 7);
        let mut log = LogicalState::tiled(8 * m).unwrap();
        run(&c, &mut log, &reg);
        assert!(log.is_logical(), "m={m}");
        let mut flat = SparseState::<C64>::new(8 * m).unwrap();
        run(&c, &mut flat, &reg);
        logical_bytes.push(log.memory_bytes());
        sparse_bytes.push(flat.memory_bytes());
    }
    // The logical ledger stays within a small envelope while the flat
    // one multiplies by the per-node orbit every time.
    assert!(
        logical_bytes[3] < logical_bytes[0] * 8,
        "logical growth {logical_bytes:?}"
    );
    assert!(
        sparse_bytes[3] > sparse_bytes[0] * 64,
        "sparse growth {sparse_bytes:?}"
    );
    assert!(
        logical_bytes[3] * 50 < sparse_bytes[3],
        "at 4 nodes: logical {} B vs sparse {} B",
        logical_bytes[3],
        sparse_bytes[3]
    );
}
