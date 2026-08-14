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
