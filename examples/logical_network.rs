//! The logical network: toroidal nodes, selector-carried links, and
//! distributed retrocorrection over the record.
//!
//! Two toric-code nodes — separate physical qubit sets that share no
//! gate after their links are raised — computed on sequentially, held
//! in a branched clock register whose selector carries the logical
//! entanglement, with a fault on one node repaired across the whole
//! record by that node's own decoder.
//!
//! Run with `cargo run --release --example logical_network`.

use quantsim::backend::{pauli_expectation, DenseState, MosaicState, PauliString, SparseState};
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

fn pauli_gates(c: &mut Circuit<C64>, p: PauliString, gate: &str) {
    let mut rest = if gate == "x" { p.x } else { p.z };
    while rest != 0 {
        let q = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        c.gate(gate, vec![], vec![q]);
    }
}

fn main() -> quantsim::Result<()> {
    let reg = GateRegistry::<C64>::standard();

    // ══ 1. the node currency: a torus is two logical wires ══
    println!("══ 1. toric nodes: one physical set, two logical wires ══\n");
    let a = ToricCode::new(2, 0)?;
    let b = ToricCode::new(2, 8)?;
    println!(
        "  node = L×L torus: {} qubits, k = 2 (the two non-contractible cycle pairs)",
        a.qubits()
    );
    let dec_b = Decoder::new(&b);
    println!(
        "  the decoder measures its own distance: at L = 2 every weight-1 X/Z\n  \
         syndrome is LOGICALLY ambiguous (refused by name — distance 2 detects,\n  \
         never corrects) while every weight-1 Y decodes ({} unambiguous signatures).",
        dec_b.len()
    );

    // ══ 2. raising the links: one transversal CX, two Bell links ══
    println!("\n══ 2. the links: transversal CX raises both wires at once ══\n");
    let map: Vec<usize> = (0..16).collect();
    let mut seg1 = a.encoder(16, [true, true]);
    seg1.append(&b.encoder(16, [false, false]), &map);
    let mut seg2: Circuit<C64> = Circuit::new(16);
    a.transversal_cx(&b, &mut seg2)?;
    let mut flat = SparseState::<C64>::new(16)?;
    run(&seg1, &mut flat, &reg);
    run(&seg2, &mut flat, &reg);
    for i in 0..2 {
        let xx = a.logical_x(i).times(b.logical_x(i)).unwrap();
        let zz = a.logical_z(i).times(b.logical_z(i)).unwrap();
        println!(
            "  link {i}: ⟨X̄X̄⟩ = {:+.4}, ⟨Z̄Z̄⟩ = {:+.4}",
            exp_of(&flat, xx),
            exp_of(&flat, zz)
        );
    }

    // ══ 3. the selector carries the links ══
    println!("\n══ 3. the selector qudit IS the logical entanglement ══\n");
    let w4 = c64(0.5, 0.0);
    let mut branches: Vec<(usize, C64, Box<dyn Backend<C64>>)> = Vec::new();
    for sel in 0..4usize {
        let mut c = a.encoder(16, [false, false]);
        c.append(&b.encoder(16, [false, false]), &map);
        for (i, on) in [(0usize, sel & 1 != 0), (1, sel & 2 != 0)] {
            if on {
                for code in [&a, &b] {
                    pauli_gates(&mut c, code.logical_x(i), "x");
                }
            }
        }
        let mut m = MosaicState::<C64>::new(16)?;
        run(&c, &mut m, &reg);
        branches.push((sel, w4, Box::new(m)));
    }
    let net = BranchedRegister::from_branches(16, 4, branches)?;
    let mut dev = 0.0f64;
    for i in 0..(1u64 << 16) {
        let d = flat.amplitude(i) - net.amplitude(i);
        dev = dev.max((d.re * d.re + d.im * d.im).sqrt());
    }
    println!(
        "  4 branches, each a PRODUCT of per-node code states (mosaic: 2 regions,\n  \
         the node cut never crossed); contracted == flat to {dev:e}"
    );
    println!(
        "  selector rank {} = 2^(links); branched {} B vs flat {} B vs dense {} B",
        net.selector_schmidt_rank()?,
        net.memory_bytes(),
        flat.memory_bytes(),
        DenseState::<C64>::new(16)?.memory_bytes()
    );

    // ══ 4. sequential node-local compute, and a fault in the record ══
    println!("\n══ 4. distributed retrocorrection over the record ══\n");
    let mut seg3: Circuit<C64> = Circuit::new(16); // node A era only
    pauli_gates(&mut seg3, a.logical_x(0), "x");
    let mut seg4: Circuit<C64> = Circuit::new(16); // node B era only
    pauli_gates(&mut seg4, b.logical_z(1), "z");
    let segs = [&seg1, &seg2, &seg3, &seg4];
    let steps4 = compile_clifford(&seg4)?;

    let err = PauliString {
        x: 1 << 9,
        z: 1 << 9,
        negative: false,
    };
    let lived = |upto: usize| -> quantsim::Result<Box<dyn Backend<C64>>> {
        let mut s = SparseState::<C64>::new(16)?;
        for (i, seg) in segs.iter().enumerate() {
            if i < upto {
                run(seg, &mut s, &reg);
            }
            if i == 1 && upto > 2 {
                apply_pauli(&mut s, err, &reg)?;
            }
        }
        Ok(Box::new(s))
    };
    let mut record = BranchedRegister::from_branches(
        16,
        4,
        vec![
            (0, w4, lived(1)?),
            (1, w4, lived(2)?),
            (2, w4, lived(3)?),
            (3, w4, lived(4)?),
        ],
    )?;
    println!("  4-slice record: encode | link | node-A era | node-B era; Y fault on a");
    println!("  node-B edge after slice 2 — after the link, no segment touches both nodes.");

    let unflag =
        |record: &BranchedRegister<C64>, sel: usize| -> quantsim::Result<SparseState<C64>> {
            let mut entries = Vec::new();
            for i in 0..(1u64 << 16) {
                let amp = record.flagged_amplitude(sel, i);
                if amp.abs_sqr() > 0.0 {
                    entries.push((i, amp * c64(2.0, 0.0)));
                }
            }
            let mut s = SparseState::<C64>::new(16)?;
            s.load(&entries)?;
            Ok(s)
        };
    let end = unflag(&record, 3)?;
    let syn_a = syndrome_bits(&end, &a, 1e-9)?;
    let syn_b = syndrome_bits(&end, &b, 1e-9)?;
    println!(
        "  end-of-record syndromes: node A {} fired (the fault stayed node-local),\n  \
         node B {} fired — node B's own decoder names it.",
        syn_a.iter().filter(|&&s| s).count(),
        syn_b.iter().filter(|&&s| s).count()
    );
    let c_end = dec_b.decode(&syn_b)?;
    let c_past = transport_back(c_end, std::slice::from_ref(&steps4));
    record.apply_at(3, |s| apply_pauli(s, c_end, &reg))?;
    record.apply_at(2, |s| apply_pauli(s, c_past, &reg))?;

    // The sign the syndrome cannot see, named by the record itself.
    for sel in [2usize, 3] {
        let mut pushed = unflag(&record, sel - 1)?;
        run(segs[sel], &mut pushed, &reg);
        let slice = unflag(&record, sel)?;
        let mut probe: Option<(u64, C64)> = None;
        pushed.for_each_nonzero(&mut |i, amp| {
            if probe.is_none() {
                probe = Some((i, amp));
            }
        });
        let (i, want) = probe.unwrap();
        if (want + slice.amplitude(i)).norm() < (want - slice.amplitude(i)).norm() {
            record.apply_at(sel, |s| {
                let m = c64(-1.0, 0.0);
                s.apply_diagonal(&[m, m], &[0])
            })?;
            println!("  slice {sel}: the syndrome-blind sign, named by record consistency");
        }
    }

    let clean = |upto: usize| -> quantsim::Result<SparseState<C64>> {
        let mut s = SparseState::<C64>::new(16)?;
        for (i, seg) in segs.iter().enumerate() {
            if i < upto {
                run(seg, &mut s, &reg);
            }
        }
        Ok(s)
    };
    for sel in 0..4usize {
        let want = clean(sel + 1)?;
        let mut dev = 0.0f64;
        want.for_each_nonzero(&mut |i, w_amp| {
            let d = w_amp - record.flagged_amplitude(sel, i) * c64(2.0, 0.0);
            dev = dev.max((d.re * d.re + d.im * d.im).sqrt());
        });
        println!("  slice {sel}: restored, dev {dev:e}");
    }
    for (sel, label) in [(1usize, "linked"), (2, "post A-era"), (3, "post B-era")] {
        let s = unflag(&record, sel)?;
        let xx0 = a.logical_x(0).times(b.logical_x(0)).unwrap();
        let zz0 = a.logical_z(0).times(b.logical_z(0)).unwrap();
        let xx1 = a.logical_x(1).times(b.logical_x(1)).unwrap();
        let zz1 = a.logical_z(1).times(b.logical_z(1)).unwrap();
        println!(
            "  slice {sel} ({label}): link0 ({:+.0},{:+.0})  link1 ({:+.0},{:+.0})",
            exp_of(&s, xx0),
            exp_of(&s, zz0),
            exp_of(&s, xx1),
            exp_of(&s, zz1)
        );
    }
    println!(
        "\n  {} B for the 4-slice record of two 8-qubit nodes; each node decoded\n  \
         its own past, the links survived with their sign bookkeeping, and the\n  \
         selector held the network the whole time.",
        record.memory_bytes()
    );
    Ok(())
}
