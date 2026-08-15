//! Retrocorrection: the surface code as a constraint on the record.
//! That syndromes are deterministic reads in a simulator, that the
//! decoder's table is measured from the code's own generators, that a
//! correction decoded once at the end of the record repairs every
//! stored slice through Clifford transport — and that the logical
//! entanglement graph's history survives, signs and all.

use quantsim::backend::{pauli_expectation, DenseState, PauliString, SparseState};
use quantsim::coupling::Stabilizer;
use quantsim::prelude::*;
use quantsim::retro::*;

fn run(c: &Circuit<C64>, state: &mut dyn Backend<C64>, reg: &GateRegistry<C64>) {
    c.bind(reg).unwrap().run(state).unwrap();
}

fn exp_of(state: &dyn Backend<C64>, p: PauliString) -> f64 {
    let mut ops = Vec::new();
    let mut rest = p.x | p.z;
    while rest != 0 {
        let q = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        let bit = 1u64 << q;
        ops.push((
            q,
            match (p.x & bit != 0, p.z & bit != 0) {
                (true, false) => quantsim::gates::Pauli::X,
                (false, true) => quantsim::gates::Pauli::Z,
                _ => quantsim::gates::Pauli::Y,
            },
        ));
    }
    pauli_expectation(state, &ops).unwrap().re
}

fn pauli(x: u64, z: u64) -> PauliString {
    PauliString {
        x,
        z,
        negative: false,
    }
}

#[test]
fn the_code_is_a_code_and_the_encoder_lands_in_it() {
    let reg = GateRegistry::<C64>::standard();
    for d in [3usize, 5] {
        let code = SurfaceCode::new(d, 0).unwrap();
        let gens = code.generators();
        assert_eq!(gens.len(), d * d - 1, "d={d}: n − k generators");
        for (i, a) in gens.iter().enumerate() {
            for b in gens.iter().skip(i + 1) {
                assert!(a.commutes_with(*b), "d={d}: generators must commute");
            }
        }
        assert_eq!(
            Stabilizer::new(gens.clone()).rank(),
            d * d - 1,
            "d={d}: independent, so exactly one logical qubit"
        );
        let (lx, lz) = (code.logical_x(), code.logical_z());
        for g in &gens {
            assert!(g.commutes_with(lx) && g.commutes_with(lz));
        }
        assert!(!lx.commutes_with(lz), "the logical pair anticommutes");

        // The encoder lands exactly in the code space, both bases.
        for (x_bar, logical) in [(false, lz), (true, lx)] {
            let mut s = SparseState::<C64>::new(d * d).unwrap();
            run(&code.encoder(d * d, x_bar), &mut s, &reg);
            for v in syndromes(&s, &code).unwrap() {
                assert!((v - 1.0).abs() < 1e-9, "d={d}: syndrome {v}");
            }
            assert!((exp_of(&s, logical) - 1.0).abs() < 1e-9);
        }
    }
}

#[test]
fn syndromes_are_deterministic_reads_and_prediction_equals_reading() {
    // The simulator inversion: no ancillas, no randomness, nothing
    // disturbed — and the syndrome a Pauli fault will produce is
    // computable from the string alone, before any state exists.
    let reg = GateRegistry::<C64>::standard();
    let code = SurfaceCode::new(3, 0).unwrap();
    let enc = code.encoder(9, true);
    let decoder = Decoder::new(&code);
    let mut worst = 0.0f64;
    for q in 0..9u64 {
        let bit = 1u64 << q;
        for e in [pauli(bit, 0), pauli(0, bit), pauli(bit, bit)] {
            let mut clean = DenseState::<C64>::new(9).unwrap();
            run(&enc, &mut clean, &reg);
            let mut hurt = DenseState::<C64>::new(9).unwrap();
            run(&enc, &mut hurt, &reg);
            apply_pauli(&mut hurt, e, &reg).unwrap();
            let read = syndrome_bits(&hurt, &code, 1e-9).unwrap();
            assert_eq!(read, signature(&code, e), "prediction is the reading");
            // Decode and restore — exactly, degenerate corrections
            // included (c·e is then a +1 stabilizer element).
            let c = decoder.decode(&read).unwrap();
            apply_pauli(&mut hurt, c, &reg).unwrap();
            for i in 0..(1u64 << 9) {
                let d = clean.amplitude(i) - hurt.amplitude(i);
                worst = worst.max((d.re * d.re + d.im * d.im).sqrt());
            }
        }
    }
    // Measured across all 27 weight-1 faults: bit-exact restoration.
    assert_eq!(worst, 0.0, "restoration is exact, not merely close");
    // 27 faults, 23 distinct signatures: the 4 collisions are the
    // weight-2 boundary-check degeneracies, and either correction of a
    // degenerate pair restores exactly (asserted above).
    assert_eq!(decoder.len(), 23);
}

#[test]
fn refusals_stay_loud_at_every_boundary() {
    let reg = GateRegistry::<C64>::standard();
    let code = SurfaceCode::new(3, 0).unwrap();

    // A state off the Pauli deformation set is not a syndrome.
    let mut s = SparseState::<C64>::new(9).unwrap();
    run(&code.encoder(9, false), &mut s, &reg);
    let mut c = Circuit::new(9);
    c.gate("ry", vec![0.3], vec![4]);
    run(&c, &mut s, &reg);
    match syndrome_bits(&s, &code, 1e-9) {
        Err(Error::InvalidState(msg)) => {
            assert!(msg.contains("not a Pauli deformation"), "{msg}")
        }
        other => panic!("expected a refusal, got {other:?}"),
    }

    // A syndrome beyond the measured table is refused, never guessed.
    let decoder = Decoder::new(&code);
    let far = pauli(1 | (1 << 8), 0); // X0·X8 — beyond weight 1
    let sig = signature(&code, far);
    match decoder.decode(&sig) {
        Err(Error::InvalidState(msg)) => assert!(msg.contains("refused"), "{msg}"),
        Err(other) => panic!("wrong refusal shape: {other:?}"),
        Ok(c) => {
            // If the composite syndrome collides with a weight-1 entry,
            // the decoded correction must still restore exactly — the
            // degeneracy is only acceptable when it is invisible.
            let mut clean = DenseState::<C64>::new(9).unwrap();
            run(&code.encoder(9, true), &mut clean, &reg);
            let mut hurt = DenseState::<C64>::new(9).unwrap();
            run(&code.encoder(9, true), &mut hurt, &reg);
            apply_pauli(&mut hurt, far, &reg).unwrap();
            apply_pauli(&mut hurt, c, &reg).unwrap();
            let mut dev = 0.0f64;
            for i in 0..(1u64 << 9) {
                let d = clean.amplitude(i) - hurt.amplitude(i);
                dev = dev.max((d.re * d.re + d.im * d.im).sqrt());
            }
            assert_eq!(dev, 0.0, "a colliding decode must be a degeneracy");
        }
    }

    // Transport through magic is refused by name.
    let mut t = Circuit::<C64>::new(2);
    t.gate("t", vec![], vec![0]);
    match compile_clifford(&t) {
        Err(Error::InvalidState(msg)) => assert!(msg.contains("magic"), "{msg}"),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn the_logical_entanglement_graph_is_built_from_physical_gates() {
    // Two patches, transversal CX: a logical Bell edge, verified from
    // logical stabilizer expectations while all 16 physical generators
    // stay clean — and held sparse at a fraction of dense.
    let reg = GateRegistry::<C64>::standard();
    let a = SurfaceCode::new(3, 0).unwrap();
    let b = SurfaceCode::new(3, 9).unwrap();
    let mut c = a.encoder(18, true);
    c.append(&b.encoder(18, false), &(0..18).collect::<Vec<usize>>());
    a.transversal_cx(&b, &mut c).unwrap();
    let mut bell = SparseState::<C64>::new(18).unwrap();
    run(&c, &mut bell, &reg);
    for code in [&a, &b] {
        for v in syndromes(&bell, code).unwrap() {
            assert!((v - 1.0).abs() < 1e-9);
        }
    }
    let xx = a.logical_x().times(b.logical_x()).unwrap();
    let zz = a.logical_z().times(b.logical_z()).unwrap();
    assert!((exp_of(&bell, xx) - 1.0).abs() < 1e-12, "X̄X̄ edge");
    assert!((exp_of(&bell, zz) - 1.0).abs() < 1e-12, "Z̄Z̄ edge");
    assert!(exp_of(&bell, a.logical_x()).abs() < 1e-12, "X̄₁ alone is 0");
    // Measured: 25.6 KiB sparse against 4 MiB dense at width 18.
    assert!(bell.memory_bytes() * 100 < DenseState::<C64>::new(18).unwrap().memory_bytes());
}

#[test]
fn one_reading_at_the_end_of_the_record_repairs_every_slice() {
    // The retrocorrection claim, end to end. A three-slice record of a
    // two-patch register: encode, entangle (transversal CX), a logical
    // Pauli era. A Y fault lands on the target patch's centre right
    // after slice 1 is written, and the CX then spreads it onto the
    // control patch — so the end-of-record syndromes fire on BOTH
    // patches. One decode at the end, corrections conjugated backward
    // through the intervening segments, slice-addressed surgery on the
    // branched record — and the whole history matches the clean run to
    // machine epsilon, logical graph signs included.
    let reg = GateRegistry::<C64>::standard();
    let a = SurfaceCode::new(3, 0).unwrap();
    let b = SurfaceCode::new(3, 9).unwrap();
    let map: Vec<usize> = (0..18).collect();

    let mut seg1 = a.encoder(18, true);
    seg1.append(&b.encoder(18, false), &map);
    let mut seg2: Circuit<C64> = Circuit::new(18);
    a.transversal_cx(&b, &mut seg2).unwrap();
    let mut seg3: Circuit<C64> = Circuit::new(18);
    for q in [0usize, 3, 6] {
        seg3.gate("x", vec![], vec![q]); // X̄ on a
    }
    for q in [9usize, 10, 11] {
        seg3.gate("z", vec![], vec![q]); // Z̄ on b
    }
    let steps2 = compile_clifford(&seg2).unwrap();
    let steps3 = compile_clifford(&seg3).unwrap();

    // The clean history, recomputed independently.
    let clean = |upto: usize| -> SparseState<C64> {
        let mut s = SparseState::<C64>::new(18).unwrap();
        for (i, seg) in [&seg1, &seg2, &seg3].iter().enumerate() {
            if i < upto {
                run(seg, &mut s, &reg);
            }
        }
        s
    };

    // The lived record: the fault lands after slice 1.
    let err = pauli(1 << 13, 1 << 13);
    let lived = |upto: usize| -> Box<dyn Backend<C64>> {
        let mut s = SparseState::<C64>::new(18).unwrap();
        run(&seg1, &mut s, &reg);
        if upto >= 2 {
            apply_pauli(&mut s, err, &reg).unwrap();
            run(&seg2, &mut s, &reg);
        }
        if upto >= 3 {
            run(&seg3, &mut s, &reg);
        }
        Box::new(s)
    };
    let w = c64(1.0 / 3f64.sqrt(), 0.0);
    let mut record = BranchedRegister::from_branches(
        18,
        3,
        vec![(0, w, lived(1)), (1, w, lived(2)), (2, w, lived(3))],
    )
    .unwrap();

    // Read the end of the record only.
    let unflag = |record: &BranchedRegister<C64>, sel: usize| -> SparseState<C64> {
        let mut entries = Vec::new();
        for i in 0..(1u64 << 18) {
            let amp = record.flagged_amplitude(sel, i);
            if amp.abs_sqr() > 0.0 {
                entries.push((i, amp * c64(3f64.sqrt(), 0.0)));
            }
        }
        let mut s = SparseState::<C64>::new(18).unwrap();
        s.load(&entries).unwrap();
        s
    };
    let end = unflag(&record, 2);
    let syn_a = syndrome_bits(&end, &a, 1e-9).unwrap();
    let syn_b = syndrome_bits(&end, &b, 1e-9).unwrap();
    assert!(
        syn_a.iter().any(|&s| s),
        "the CX spread the fault to patch a"
    );
    assert!(syn_b.iter().any(|&s| s));

    // Prediction: transport the QUESTION to when the error lived —
    // each generator pulled back through the later segments
    // anticommutes with the raw fault exactly where the end-of-record
    // syndrome fires.
    let later = vec![steps2.clone(), steps3.clone()];
    for (code, syn) in [(&a, &syn_a), (&b, &syn_b)] {
        let predicted: Vec<bool> = code
            .generators()
            .iter()
            .map(|g| !transport_back(*g, &later).commutes_with(err))
            .collect();
        assert_eq!(&predicted, syn, "the code predicts the record");
    }

    // One decode at the end; both patches contribute weight 1.
    let c_end = Decoder::new(&a)
        .decode(&syn_a)
        .unwrap()
        .times(Decoder::new(&b).decode(&syn_b).unwrap())
        .unwrap();
    assert_eq!(
        (c_end.x | c_end.z).count_ones(),
        2,
        "one fault, two patches"
    );

    // Retrocorrect: the end directly, the past slice through seg3.
    let c_past = transport_back(c_end, std::slice::from_ref(&steps3));
    record.apply_at(2, |s| apply_pauli(s, c_end, &reg)).unwrap();
    record
        .apply_at(1, |s| apply_pauli(s, c_past, &reg))
        .unwrap();

    // Every slice of the record now matches the clean history.
    for sel in 0..3usize {
        let want = clean(sel + 1);
        let mut dev = 0.0f64;
        want.for_each_nonzero(&mut |i, w_amp| {
            let got = record.flagged_amplitude(sel, i) * c64(3f64.sqrt(), 0.0);
            let d = w_amp - got;
            dev = dev.max((d.re * d.re + d.im * d.im).sqrt());
        });
        assert!(dev < 1e-12, "slice {sel} restored: dev {dev:e}");
    }

    // The logical entanglement graph's history, signs and all: the
    // Bell edge is +1/+1 after entangling and −1/−1 after the logical
    // Pauli era (X̄ₐ flips Z̄Z̄, Z̄_b flips X̄X̄ — product +1: still
    // maximally correlated, bookkeeping intact).
    let xx = a.logical_x().times(b.logical_x()).unwrap();
    let zz = a.logical_z().times(b.logical_z()).unwrap();
    for (sel, want_xx, want_zz) in [(1usize, 1.0, 1.0), (2, -1.0, -1.0)] {
        let s = unflag(&record, sel);
        assert!((exp_of(&s, xx) - want_xx).abs() < 1e-9, "slice {sel} X̄X̄");
        assert!((exp_of(&s, zz) - want_zz).abs() < 1e-9, "slice {sel} Z̄Z̄");
    }

    // Surgery through a shared wall is refused. A selector mix puts
    // the same underlying state under both flags — retrocorrecting one
    // slice would silently rewrite the other's history, and the guard
    // says so instead of doing it.
    let mut tiny = BranchedRegister::<C64>::from_branches(
        4,
        2,
        vec![
            (
                0,
                c64(0.5f64.sqrt(), 0.0),
                Box::new(SparseState::<C64>::new(4).unwrap()) as Box<dyn Backend<C64>>,
            ),
            (
                1,
                c64(0.5f64.sqrt(), 0.0),
                Box::new(SparseState::<C64>::new(4).unwrap()) as Box<dyn Backend<C64>>,
            ),
        ],
    )
    .unwrap();
    // Distinct states per selector: surgery is allowed.
    tiny.apply_at(0, |_| Ok(())).unwrap();
    // Mix the selector: every state now lives under both flags.
    let h = c64(0.5f64.sqrt(), 0.0);
    tiny.selector_mix(&[h, h, h, -h]).unwrap();
    match tiny.apply_at(0, |_| Ok(())) {
        Err(Error::InvalidState(msg)) => assert!(msg.contains("unshare"), "{msg}"),
        other => panic!("expected the shared-wall refusal, got {other:?}"),
    }
}

// ── the torus: two logical wires per physical set ────────────────────

#[test]
fn the_torus_carries_two_logical_qubits_and_knows_its_distance() {
    let reg = GateRegistry::<C64>::standard();
    for l in [2usize, 3] {
        let t = ToricCode::new(l, 0).unwrap();
        let gens = quantsim::retro::Code::generators(&t);
        for (i, a) in gens.iter().enumerate() {
            for b in gens.iter().skip(i + 1) {
                assert!(a.commutes_with(*b), "L={l}: generators commute");
            }
        }
        assert_eq!(
            Stabilizer::new(gens.clone()).rank(),
            2 * l * l - 2,
            "L={l}: one redundant check per type → k = 2"
        );
        for i in 0..2 {
            for g in &gens {
                assert!(g.commutes_with(t.logical_x(i)) && g.commutes_with(t.logical_z(i)));
            }
            assert!(!t.logical_x(i).commutes_with(t.logical_z(i)));
            assert!(t.logical_x(i).commutes_with(t.logical_z(1 - i)));
        }
        // Encoders land in the code space in every basis combination.
        for plus in [[false, false], [true, false], [true, true]] {
            let mut s = SparseState::<C64>::new(t.qubits()).unwrap();
            run(&t.encoder(t.qubits(), plus), &mut s, &reg);
            for v in syndromes(&s, &t).unwrap() {
                assert!((v - 1.0).abs() < 1e-9, "L={l} {plus:?}");
            }
            for (i, &in_plus) in plus.iter().enumerate() {
                let logical = if in_plus {
                    t.logical_x(i)
                } else {
                    t.logical_z(i)
                };
                assert!((exp_of(&s, logical) - 1.0).abs() < 1e-9);
            }
        }
    }

    // The decoder measures the distance dichotomy itself. At L = 2
    // every weight-1 X and Z fault is LOGICALLY ambiguous — two faults
    // separated by a logical share each signature — and the decoder
    // refuses them by name: distance 2 detects and never corrects.
    // The Y faults decode: their X- and Z-side collisions are each
    // stabilizer-degenerate, so the combination is invisible.
    let t2 = ToricCode::new(2, 0).unwrap();
    let dec2 = Decoder::new(&t2);
    for q in 0..t2.qubits() {
        let bit = 1u64 << q;
        for p in [pauli(bit, 0), pauli(0, bit)] {
            match dec2.decode(&signature(&t2, p)) {
                Err(Error::InvalidState(msg)) => {
                    assert!(msg.contains("ambiguous"), "{msg}")
                }
                other => panic!("L=2 weight-1 X/Z must be ambiguous, got {other:?}"),
            }
        }
        assert!(dec2.decode(&signature(&t2, pauli(bit, bit))).is_ok());
    }
    // At L = 3 every weight-1 fault decodes.
    let t3 = ToricCode::new(3, 0).unwrap();
    let dec3 = Decoder::new(&t3);
    for q in 0..t3.qubits() {
        let bit = 1u64 << q;
        for p in [pauli(bit, 0), pauli(0, bit), pauli(bit, bit)] {
            assert!(dec3.decode(&signature(&t3, p)).is_ok());
        }
    }
}

#[test]
fn the_selector_qudit_carries_the_logical_network() {
    // Two toric nodes, one transversal CX: TWO logical Bell links from
    // one physical operation. The same state, two ways: flat (physical
    // entanglement across the node cut) and Schmidt-branched (each
    // branch a PRODUCT of per-node code states, the selector carrying
    // the links). They agree amplitude for amplitude; the selector's
    // Schmidt rank is 2^links; and each mosaic branch holds the nodes
    // as separate regions because no branch circuit ever crosses the
    // cut.
    let reg = GateRegistry::<C64>::standard();
    let a = ToricCode::new(2, 0).unwrap();
    let b = ToricCode::new(2, 8).unwrap();
    let map: Vec<usize> = (0..16).collect();

    let mut flat_c = a.encoder(16, [true, true]);
    flat_c.append(&b.encoder(16, [false, false]), &map);
    a.transversal_cx(&b, &mut flat_c).unwrap();
    let mut flat = SparseState::<C64>::new(16).unwrap();
    run(&flat_c, &mut flat, &reg);
    for i in 0..2 {
        let xx = a.logical_x(i).times(b.logical_x(i)).unwrap();
        let zz = a.logical_z(i).times(b.logical_z(i)).unwrap();
        assert!((exp_of(&flat, xx) - 1.0).abs() < 1e-9, "link {i} X̄X̄");
        assert!((exp_of(&flat, zz) - 1.0).abs() < 1e-9, "link {i} Z̄Z̄");
    }

    let w = c64(0.5, 0.0);
    let mut branches: Vec<(usize, C64, Box<dyn Backend<C64>>)> = Vec::new();
    for sel in 0..4usize {
        let mut c = a.encoder(16, [false, false]);
        c.append(&b.encoder(16, [false, false]), &map);
        for (i, on) in [(0usize, sel & 1 != 0), (1, sel & 2 != 0)] {
            if on {
                for code in [&a, &b] {
                    let mut rest = code.logical_x(i).x;
                    while rest != 0 {
                        let q = rest.trailing_zeros() as usize;
                        rest &= rest - 1;
                        c.gate("x", vec![], vec![q]);
                    }
                }
            }
        }
        let mut m = quantsim::backend::MosaicState::<C64>::new(16).unwrap();
        run(&c, &mut m, &reg);
        assert_eq!(
            m.layout().len(),
            2,
            "branch {sel}: the node cut is never crossed"
        );
        branches.push((sel, w, Box::new(m)));
    }
    let net = BranchedRegister::from_branches(16, 4, branches).unwrap();
    assert_eq!(net.selector_schmidt_rank().unwrap(), 4, "2^(two links)");
    let mut dev = 0.0f64;
    for i in 0..(1u64 << 16) {
        let d = flat.amplitude(i) - net.amplitude(i);
        dev = dev.max((d.re * d.re + d.im * d.im).sqrt());
    }
    assert!(dev < 1e-12, "the selector holds the links exactly: {dev:e}");
    assert!(
        net.memory_bytes() < flat.memory_bytes(),
        "products under a selector undercut the flat cut: {} vs {}",
        net.memory_bytes(),
        flat.memory_bytes()
    );
}

#[test]
fn distributed_retrocorrection_across_the_logical_network() {
    // Sequential separate physical sets with integrated logical
    // entanglement, and a fault repaired across the whole record. Four
    // slices: encode both nodes; transversal CX (both links up);
    // node-A logical era; node-B logical era — after the link is made,
    // NO segment touches both nodes. A Y fault lands on node B after
    // slice 2. The end-of-record syndromes fire on node B ONLY (node-
    // local dynamics kept it node-local), node B's own decoder names
    // the fault, the correction transports back through the B-era
    // untouched by node A's — and every slice returns to the clean
    // history with the link signs' bookkeeping intact.
    let reg = GateRegistry::<C64>::standard();
    let a = ToricCode::new(2, 0).unwrap();
    let b = ToricCode::new(2, 8).unwrap();
    let map: Vec<usize> = (0..16).collect();

    let mut seg1 = a.encoder(16, [true, true]);
    seg1.append(&b.encoder(16, [false, false]), &map);
    let mut seg2: Circuit<C64> = Circuit::new(16);
    a.transversal_cx(&b, &mut seg2).unwrap();
    let mut seg3: Circuit<C64> = Circuit::new(16); // node A only: X̄₀ᴬ
    {
        let mut rest = a.logical_x(0).x;
        while rest != 0 {
            let q = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            seg3.gate("x", vec![], vec![q]);
        }
    }
    let mut seg4: Circuit<C64> = Circuit::new(16); // node B only: Z̄₁ᴮ
    {
        let mut rest = b.logical_z(1).z;
        while rest != 0 {
            let q = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            seg4.gate("z", vec![], vec![q]);
        }
    }
    let segs = [&seg1, &seg2, &seg3, &seg4];
    let steps4 = compile_clifford(&seg4).unwrap();

    let err = pauli(1 << 9, 1 << 9); // Y on a node-B edge
    let lived = |upto: usize| -> Box<dyn Backend<C64>> {
        let mut s = SparseState::<C64>::new(16).unwrap();
        for (i, seg) in segs.iter().enumerate() {
            if i < upto {
                run(seg, &mut s, &reg);
            }
            if i == 1 && upto > 2 {
                apply_pauli(&mut s, err, &reg).unwrap();
            }
        }
        Box::new(s)
    };
    let clean = |upto: usize| -> SparseState<C64> {
        let mut s = SparseState::<C64>::new(16).unwrap();
        for (i, seg) in segs.iter().enumerate() {
            if i < upto {
                run(seg, &mut s, &reg);
            }
        }
        s
    };
    let w = c64(0.5, 0.0);
    let mut record = BranchedRegister::from_branches(
        16,
        4,
        vec![
            (0, w, lived(1)),
            (1, w, lived(2)),
            (2, w, lived(3)),
            (3, w, lived(4)),
        ],
    )
    .unwrap();

    let unflag = |record: &BranchedRegister<C64>, sel: usize| -> SparseState<C64> {
        let mut entries = Vec::new();
        for i in 0..(1u64 << 16) {
            let amp = record.flagged_amplitude(sel, i);
            if amp.abs_sqr() > 0.0 {
                entries.push((i, amp * c64(2.0, 0.0)));
            }
        }
        let mut s = SparseState::<C64>::new(16).unwrap();
        s.load(&entries).unwrap();
        s
    };

    // Node A's syndromes are clean at the end — the fault stayed
    // node-local because nothing after the link touched both nodes.
    let end = unflag(&record, 3);
    let syn_a = syndrome_bits(&end, &a, 1e-9).unwrap();
    let syn_b = syndrome_bits(&end, &b, 1e-9).unwrap();
    assert!(syn_a.iter().all(|&s| !s), "node A never saw the fault");
    assert!(syn_b.iter().any(|&s| s));

    // Node B's own decoder names it (a Y at L = 2 decodes), and the
    // correction transported back through node B's era is what acts on
    // slice 3; node A's era never enters the transport at all.
    let c_end = Decoder::new(&b).decode(&syn_b).unwrap();
    let c_past = transport_back(c_end, std::slice::from_ref(&steps4));
    record.apply_at(3, |s| apply_pauli(s, c_end, &reg)).unwrap();
    record
        .apply_at(2, |s| apply_pauli(s, c_past, &reg))
        .unwrap();

    // Syndromes never see a sign: the B-era's Z̄₁ᴮ shares an edge with
    // the fault, so conjugation flipped the residual to −Y — and the
    // sign-blind correction leaves a global −1 on the repaired slices,
    // which a branched record makes PHYSICAL (the phase-faithfulness
    // contract). The syndrome cannot know the sign; the record can:
    // each slice must equal its predecessor pushed through the
    // segment, the uncorrupted slice anchors the chain, and one
    // amplitude comparison per slice names the sign.
    for sel in [2usize, 3] {
        let mut pushed = unflag(&record, sel - 1);
        run(segs[sel], &mut pushed, &reg);
        let slice = unflag(&record, sel);
        let mut probe: Option<(u64, C64)> = None;
        pushed.for_each_nonzero(&mut |i, amp| {
            if probe.is_none() {
                probe = Some((i, amp));
            }
        });
        let (i, want) = probe.unwrap();
        let got = slice.amplitude(i);
        if (want + got).norm() < (want - got).norm() {
            record
                .apply_at(sel, |s| {
                    let m = c64(-1.0, 0.0);
                    s.apply_diagonal(&[m, m], &[0])
                })
                .unwrap();
        }
    }

    for sel in 0..4usize {
        let want = clean(sel + 1);
        let mut dev = 0.0f64;
        want.for_each_nonzero(&mut |i, w_amp| {
            let got = record.flagged_amplitude(sel, i) * c64(2.0, 0.0);
            let d = w_amp - got;
            dev = dev.max((d.re * d.re + d.im * d.im).sqrt());
        });
        assert!(dev < 1e-12, "slice {sel} restored: {dev:e}");
    }

    // The links' sign history across the record, exactly as the
    // logical algebra demands: X̄₀ᴬ flips link 0's Z̄Z̄ at slice 3,
    // Z̄₁ᴮ flips link 1's X̄X̄ at slice 4, products stay +1.
    let link = |i: usize| {
        (
            a.logical_x(i).times(b.logical_x(i)).unwrap(),
            a.logical_z(i).times(b.logical_z(i)).unwrap(),
        )
    };
    let (xx0, zz0) = link(0);
    let (xx1, zz1) = link(1);
    for (sel, e_xx0, e_zz0, e_xx1, e_zz1) in [
        (1usize, 1.0, 1.0, 1.0, 1.0),
        (2, 1.0, -1.0, 1.0, 1.0),
        (3, 1.0, -1.0, -1.0, 1.0),
    ] {
        let s = unflag(&record, sel);
        assert!(
            (exp_of(&s, xx0) - e_xx0).abs() < 1e-9,
            "slice {sel} link0 X̄X̄"
        );
        assert!(
            (exp_of(&s, zz0) - e_zz0).abs() < 1e-9,
            "slice {sel} link0 Z̄Z̄"
        );
        assert!(
            (exp_of(&s, xx1) - e_xx1).abs() < 1e-9,
            "slice {sel} link1 X̄X̄"
        );
        assert!(
            (exp_of(&s, zz1) - e_zz1).abs() < 1e-9,
            "slice {sel} link1 Z̄Z̄"
        );
    }

    // And the honest wall: the same fault landing BEFORE the link
    // spreads through the CX onto node A as a weight-1 Z — which at
    // distance 2 is logically ambiguous, and node A's decoder refuses
    // to guess.
    let mut early = SparseState::<C64>::new(16).unwrap();
    run(&seg1, &mut early, &reg);
    apply_pauli(&mut early, err, &reg).unwrap();
    run(&seg2, &mut early, &reg);
    let syn_a_early = syndrome_bits(&early, &a, 1e-9).unwrap();
    assert!(syn_a_early.iter().any(|&s| s), "the CX spread it to node A");
    match Decoder::new(&a).decode(&syn_a_early) {
        Err(Error::InvalidState(msg)) => assert!(msg.contains("ambiguous"), "{msg}"),
        other => panic!("distance 2 must refuse the spread fault, got {other:?}"),
    }
}

#[test]
fn decoding_past_weight_one_is_minimum_weight_and_names_its_own_break_even() {
    // The weight-bounded decoder: minimum-weight by enumeration,
    // lowest weight first, the same degeneracy-vs-ambiguity honesty at
    // every weight. On the L = 3 torus (distance 3), weight 2 is past
    // the guarantee ⌊(d−1)/2⌋ = 1 — the instrument measures exactly
    // how far past it can still speak.
    let reg = GateRegistry::<C64>::standard();
    let t = ToricCode::new(3, 0).unwrap();
    let deep = Decoder::new(&t); // weight 1
    let deeper = Decoder::to_weight(&t, 2);
    assert!(
        deeper.len() > deep.len(),
        "weight 2 must widen the table: {} vs {}",
        deeper.len(),
        deep.len()
    );

    // Census over every weight-2 X/Z-type pair: decoded or refused,
    // never guessed. (Deterministic; the counts are pins.)
    let mut decoded = 0usize;
    let mut refused = 0usize;
    let n = 18usize;
    for a in 0..n {
        for b in (a + 1)..n {
            for (pa, pb) in [(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
                let mk = |q: usize, kind: u8| -> PauliString {
                    let bit = 1u64 << q;
                    if kind == 0 {
                        pauli(bit, 0)
                    } else {
                        pauli(0, bit)
                    }
                };
                let e = mk(a, pa).times(mk(b, pb)).unwrap();
                let sig = signature(&t, e);
                if sig.iter().all(|&s| !s) {
                    continue; // a stabilizer or logical: no syndrome at all
                }
                match deeper.decode(&sig) {
                    Ok(_) => decoded += 1,
                    Err(_) => refused += 1,
                }
            }
        }
    }
    assert!(decoded > 0 && refused > 0, "both outcomes must exist");

    // Every decode that is offered restores exactly: the correction
    // differs from the fault by a +1 stabilizer element or is it.
    let enc = t.encoder(18, [true, false]);
    let mut checked = 0usize;
    let mut rng = Prng::new(41);
    while checked < 25 {
        let a = (rng.next_u64() as usize) % n;
        let b = (rng.next_u64() as usize) % n;
        if a == b {
            continue;
        }
        let bit_a = 1u64 << a;
        let bit_b = 1u64 << b;
        let e = pauli(
            if rng.next_u64() % 2 == 0 { bit_a } else { 0 } | bit_b,
            if rng.next_u64() % 2 == 0 { bit_a } else { 0 },
        );
        if (e.x | e.z).count_ones() != 2 {
            continue;
        }
        let sig = signature(&t, e);
        let Ok(c) = deeper.decode(&sig) else {
            continue; // refused: honesty, not failure
        };
        let mut clean = SparseState::<C64>::new(18).unwrap();
        run(&enc, &mut clean, &reg);
        let mut hurt = SparseState::<C64>::new(18).unwrap();
        run(&enc, &mut hurt, &reg);
        apply_pauli(&mut hurt, e, &reg).unwrap();
        apply_pauli(&mut hurt, c, &reg).unwrap();
        let mut dev = 0.0f64;
        clean.for_each_nonzero(&mut |i, w_amp| {
            let d = w_amp - hurt.amplitude(i);
            dev = dev.max((d.re * d.re + d.im * d.im).sqrt());
        });
        assert!(dev < 1e-12, "offered decode must restore: {dev:e}");
        checked += 1;
    }
    println!("weight-2 census on the L = 3 torus: {decoded} decoded, {refused} refused");
}
