//! Retrocorrection, end to end: the surface code as a constraint on
//! the record. Error correction inverted by the simulator — syndromes
//! as deterministic reads, the decoder table measured from the code's
//! own generators, and a correction decoded once at the end of the
//! record conjugated backward to repair every stored slice of a
//! branched clock register, logical entanglement graph included.
//!
//! Run with `cargo run --release --example retrocorrection`.

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

fn main() -> quantsim::Result<()> {
    let reg = GateRegistry::<C64>::standard();

    // ══ 1. the code is a constraint set ══
    println!("══ 1. the rotated surface code, d = 3: 9 qubits, 8 constraints ══\n");
    let code = SurfaceCode::new(3, 0)?;
    let gens = code.generators();
    println!(
        "  {} generators, rank {} → one logical qubit; X̄·Z̄ anticommute: {}",
        gens.len(),
        Stabilizer::new(gens.clone()).rank(),
        !code.logical_x().commutes_with(code.logical_z())
    );
    let mut zero = SparseState::<C64>::new(9)?;
    run(&code.encoder(9, false), &mut zero, &reg);
    println!(
        "  |0̄⟩ in {} gates: every syndrome +1, ⟨Z̄⟩ = {:+.4}, {} B sparse",
        code.encoder(9, false).len(),
        exp_of(&zero, code.logical_z()),
        zero.memory_bytes()
    );

    // ══ 2. syndromes are deterministic reads, prediction is exact ══
    println!("\n══ 2. simulator syndromes: no ancillas, no randomness ══\n");
    let enc = code.encoder(9, true);
    let decoder = Decoder::new(&code);
    let mut worst = 0.0f64;
    for q in 0..9u64 {
        let bit = 1u64 << q;
        for e in [
            PauliString {
                x: bit,
                z: 0,
                negative: false,
            },
            PauliString {
                x: 0,
                z: bit,
                negative: false,
            },
            PauliString {
                x: bit,
                z: bit,
                negative: false,
            },
        ] {
            let mut clean = DenseState::<C64>::new(9)?;
            run(&enc, &mut clean, &reg);
            let mut hurt = DenseState::<C64>::new(9)?;
            run(&enc, &mut hurt, &reg);
            apply_pauli(&mut hurt, e, &reg)?;
            let read = syndrome_bits(&hurt, &code, 1e-9)?;
            assert_eq!(read, signature(&code, e));
            apply_pauli(&mut hurt, decoder.decode(&read)?, &reg)?;
            for i in 0..(1u64 << 9) {
                let d = clean.amplitude(i) - hurt.amplitude(i);
                worst = worst.max((d.re * d.re + d.im * d.im).sqrt());
            }
        }
    }
    println!("  all 27 weight-1 faults: signature(fault) == syndrome(state), every time");
    println!("  decode → restore, worst amplitude deviation: {worst:e} (exact)");
    println!(
        "  decoder table: {} signatures, measured from the generators, refusing beyond",
        decoder.len()
    );

    // ══ 3. the logical entanglement graph ══
    println!("\n══ 3. two patches, one logical edge ══\n");
    let a = SurfaceCode::new(3, 0)?;
    let b = SurfaceCode::new(3, 9)?;
    let map: Vec<usize> = (0..18).collect();
    let mut seg1 = a.encoder(18, true);
    seg1.append(&b.encoder(18, false), &map);
    let mut seg2: Circuit<C64> = Circuit::new(18);
    a.transversal_cx(&b, &mut seg2)?;
    let mut bell = SparseState::<C64>::new(18)?;
    run(&seg1, &mut bell, &reg);
    run(&seg2, &mut bell, &reg);
    let xx = a.logical_x().times(b.logical_x()).unwrap();
    let zz = a.logical_z().times(b.logical_z()).unwrap();
    println!(
        "  transversal CX = logical CX: ⟨X̄X̄⟩ = {:+.4}, ⟨Z̄Z̄⟩ = {:+.4}, ⟨X̄₁⟩ = {:+.4}",
        exp_of(&bell, xx),
        exp_of(&bell, zz),
        exp_of(&bell, a.logical_x())
    );
    println!(
        "  16 physical constraints clean; {} B sparse vs {} B dense",
        bell.memory_bytes(),
        DenseState::<C64>::new(18)?.memory_bytes()
    );

    // ══ 4. the record, and one reading that repairs all of it ══
    println!("\n══ 4. retrocorrection: the past is corrected, not compensated ══\n");
    let mut seg3: Circuit<C64> = Circuit::new(18);
    for q in [0usize, 3, 6] {
        seg3.gate("x", vec![], vec![q]);
    }
    for q in [9usize, 10, 11] {
        seg3.gate("z", vec![], vec![q]);
    }
    let steps2 = compile_clifford(&seg2)?;
    let steps3 = compile_clifford(&seg3)?;

    // The fault: Y on the target patch's centre, right after slice 1.
    let err = PauliString {
        x: 1 << 13,
        z: 1 << 13,
        negative: false,
    };
    let lived = |upto: usize| -> quantsim::Result<Box<dyn Backend<C64>>> {
        let mut s = SparseState::<C64>::new(18)?;
        run(&seg1, &mut s, &reg);
        if upto >= 2 {
            apply_pauli(&mut s, err, &reg)?;
            run(&seg2, &mut s, &reg);
        }
        if upto >= 3 {
            run(&seg3, &mut s, &reg);
        }
        Ok(Box::new(s))
    };
    let w = c64(1.0 / 3f64.sqrt(), 0.0);
    let mut record = BranchedRegister::from_branches(
        18,
        3,
        vec![(0, w, lived(1)?), (1, w, lived(2)?), (2, w, lived(3)?)],
    )?;

    let unflag =
        |record: &BranchedRegister<C64>, sel: usize| -> quantsim::Result<SparseState<C64>> {
            let mut entries = Vec::new();
            for i in 0..(1u64 << 18) {
                let amp = record.flagged_amplitude(sel, i);
                if amp.abs_sqr() > 0.0 {
                    entries.push((i, amp * c64(3f64.sqrt(), 0.0)));
                }
            }
            let mut s = SparseState::<C64>::new(18)?;
            s.load(&entries)?;
            Ok(s)
        };
    let end = unflag(&record, 2)?;
    let syn_a = syndrome_bits(&end, &a, 1e-9)?;
    let syn_b = syndrome_bits(&end, &b, 1e-9)?;
    println!(
        "  fault: one Y after slice 1; the CX spread it — syndromes fire on a: {}, b: {}",
        syn_a.iter().filter(|&&s| s).count(),
        syn_b.iter().filter(|&&s| s).count()
    );

    // Prediction: transport the question to when the error lived.
    let later = vec![steps2.clone(), steps3.clone()];
    let ok = [(&a, &syn_a), (&b, &syn_b)].iter().all(|(code, syn)| {
        code.generators()
            .iter()
            .map(|g| !transport_back(*g, &later).commutes_with(err))
            .collect::<Vec<bool>>()
            == **syn
    });
    println!("  prediction (generators transported to the fault's time) == reading: {ok}");

    let c_end = Decoder::new(&a)
        .decode(&syn_a)?
        .times(Decoder::new(&b).decode(&syn_b)?)
        .unwrap();
    let c_past = transport_back(c_end, std::slice::from_ref(&steps3));
    record.apply_at(2, |s| apply_pauli(s, c_end, &reg))?;
    record.apply_at(1, |s| apply_pauli(s, c_past, &reg))?;
    println!("  one decode at the end; corrections transported back, applied in register");

    let clean = |upto: usize| -> quantsim::Result<SparseState<C64>> {
        let mut s = SparseState::<C64>::new(18)?;
        for (i, seg) in [&seg1, &seg2, &seg3].iter().enumerate() {
            if i < upto {
                run(seg, &mut s, &reg);
            }
        }
        Ok(s)
    };
    for sel in 0..3usize {
        let want = clean(sel + 1)?;
        let mut dev = 0.0f64;
        want.for_each_nonzero(&mut |i, w_amp| {
            let got = record.flagged_amplitude(sel, i) * c64(3f64.sqrt(), 0.0);
            let d = w_amp - got;
            dev = dev.max((d.re * d.re + d.im * d.im).sqrt());
        });
        println!("  slice {sel}: matches the clean history, dev {dev:e}");
    }
    for (sel, label) in [(1usize, "post-entangling"), (2, "post-logical-era")] {
        let s = unflag(&record, sel)?;
        println!(
            "  slice {sel} ({label}): logical graph ⟨X̄X̄⟩ = {:+.4}, ⟨Z̄Z̄⟩ = {:+.4}",
            exp_of(&s, xx),
            exp_of(&s, zz)
        );
    }
    println!(
        "\n  the record register: {} B for a 3-slice history of 18 qubits\n  (dense would hold {} B per slice); the constraint set diagnosed the\n  past, predicted the present, and repaired both.",
        record.memory_bytes(),
        DenseState::<C64>::new(18)?.memory_bytes()
    );
    Ok(())
}
