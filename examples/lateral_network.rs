//! Distributed quantum computing through cross-lateralization and
//! error correction: four toric nodes that never share a clock, an
//! inter-node channel carrying nothing but 𝔽₂, and a repair that
//! crosses the quantum code against the network's parity.
//!
//! Run with `cargo run --release --example lateral_network`.

use quantsim::backend::{DenseState, PauliString};
use quantsim::lateral::*;
use quantsim::prelude::*;
use quantsim::retro::{apply_pauli, syndrome_bits, Code, Decoder, SurfaceCode, ToricCode};

fn rule(title: &str) {
    println!("\n══ {title} ══\n");
}

fn main() -> quantsim::Result<()> {
    let reg = GateRegistry::<C64>::standard();

    // ══ 1. the link is coordinatewise ══
    rule("1. the cross-lateral link: qubit i to qubit i, and nothing else");
    let a = ToricCode::new(2, 0)?;
    let b = ToricCode::new(2, 8)?;
    let link = LateralLink::new(&a, &b)?;
    println!(
        "  two L=2 toric nodes, {} qubits each, joined by {} physical CX",
        a.qubits(),
        link.width()
    );
    let mut worst = 0usize;
    let mut rng = Prng::new(0x51DE);
    for _ in 0..200_000 {
        let p = PauliString {
            x: rng.next_u64() & window_mask(0, 16),
            z: rng.next_u64() & window_mask(0, 16),
            negative: rng.next_u64() & 1 == 1,
        };
        if link.push_frame(p) != link.push_frame_by_gates(p) {
            worst += 1;
        }
    }
    println!("  frame map:  x_B ^= x_A   z_A ^= z_B   σ ^= |x_A & z_B & !(x_B ^ z_A)|");
    println!("  two XORs and a popcount against {worst} disagreements with gate-by-gate");
    println!("  conjugation over 200,000 random strings — signs included.");
    for i in [0usize, 3, 7] {
        let out = link.push_frame(PauliString {
            x: 1 << i,
            z: 1 << i,
            negative: false,
        });
        let sites: Vec<usize> = (0..16).filter(|q| (out.x | out.z) >> q & 1 == 1).collect();
        println!(
            "  Y at site {i:>2} lands on {sites:?} — its own aligned pair, never a third site"
        );
    }

    // ══ 2. the syndrome is linear, so the wire can be 𝔽₂ ══
    rule("2. the frame is the wire");
    let net = LateralNetwork::new(
        vec![(0, 8), (8, 8)],
        vec![link],
        DelayGeometry::new(2, vec![0.0, 3.0, 3.0, 0.0])?,
    )?;
    let fault_a = PauliString {
        x: 1 << 1,
        z: 1 << 5,
        negative: false,
    };
    let fault_b = PauliString {
        x: 0,
        z: 1 << 10,
        negative: false,
    };
    let program = vec![
        LateralOp::Local {
            node: 0,
            pauli: fault_a,
        },
        LateralOp::Local {
            node: 1,
            pauli: fault_b,
        },
        LateralOp::Cross { link: 0 },
    ];
    let run = net.run(&program)?;

    let mut st = DenseState::<C64>::new(16)?;
    for enc in [a.encoder(16, [false, false]), b.encoder(16, [false, false])] {
        enc.bind(&reg)?.run(&mut st)?;
    }
    apply_pauli(&mut st, fault_a, &reg)?;
    apply_pauli(&mut st, fault_b, &reg)?;
    let mut cx = Circuit::<C64>::new(16);
    a.transversal_cx(&b, &mut cx)?;
    cx.bind(&reg)?.run(&mut st)?;

    for (name, code) in [("A", &a), ("B", &b)] {
        let map = SyndromeMap::new(code)?;
        let measured = syndrome_bits(&st, code, 1e-9)?;
        let predicted = map.unpack(map.bits(run.frame));
        let bits = |v: &[bool]| {
            v.iter()
                .map(|&b| if b { '1' } else { '0' })
                .collect::<String>()
        };
        println!(
            "  node {name}: measured {}   predicted {}   {}",
            bits(&measured),
            bits(&predicted),
            if measured == predicted {
                "equal"
            } else {
                "DIVERGED"
            }
        );
    }
    println!(
        "\n  {} B crossed the wire ({} deltas × {DELTA_WIRE_LEN} B). Shipping the state",
        run.wire_bytes,
        run.deltas.len()
    );
    println!(
        "  would have cost {} B — a factor of {:.0}, and the factor grows as 2^n.",
        net.amplitude_bytes(),
        net.amplitude_bytes() as f64 / run.wire_bytes as f64
    );

    // ══ 3. erasures buy exactly a factor of two ══
    rule("3. knowing where the loss happened is worth a factor of two");
    println!("  code            d   errors ⌊(d−1)/2⌋   erasures (measured)   witness");
    for (name, gens, er, d) in [
        (
            "toric L=2",
            ToricCode::new(2, 0)?.generators().len(),
            ErasureDecoder::new(&ToricCode::new(2, 0)?),
            2usize,
        ),
        (
            "toric L=3",
            ToricCode::new(3, 0)?.generators().len(),
            ErasureDecoder::new(&ToricCode::new(3, 0)?),
            3,
        ),
        (
            "surface d=3",
            SurfaceCode::new(3, 0)?.generators().len(),
            ErasureDecoder::new(&SurfaceCode::new(3, 0)?),
            3,
        ),
        (
            "surface d=5",
            SurfaceCode::new(5, 0)?.generators().len(),
            ErasureDecoder::new(&SurfaceCode::new(5, 0)?),
            5,
        ),
    ] {
        let cap = er.certified_capacity(d + 1);
        let witness = er.first_uncorrectable(d + 1).unwrap_or_default();
        println!(
            "  {name:<14} {d}   {:^15}   {cap:^19}   {witness:?}",
            (d - 1) / 2
        );
        let _ = gens;
    }
    let l2 = ToricCode::new(2, 0)?;
    let dec = Decoder::new(&l2);
    let er = ErasureDecoder::new(&l2);
    let map = SyndromeMap::new(&l2)?;
    let fault = PauliString {
        x: 1 << 3,
        z: 0,
        negative: false,
    };
    println!("\n  the sharp case, L = 2 (distance 2 detects and never corrects):");
    match dec.decode(&quantsim::retro::signature(&l2, fault)) {
        Ok(_) => println!("    as an error:   decoded (unexpected)"),
        Err(e) => println!(
            "    as an error:   refused — {}",
            e.to_string().split(';').next().unwrap_or("").trim()
        ),
    }
    println!(
        "    as an erasure: {:?} recovered exactly from the same syndrome",
        er.decode(&[3], map.bits(fault))?
    );

    // ══ 4. the geometry sets the floor, the horizon is a budget ══
    rule("4. the network has a metric, and it is not the one you drew");
    let geo = DelayGeometry::new(
        4,
        vec![
            0.0, 2.0, 9.0, 11.0, //
            2.0, 0.0, 3.0, 7.0, //
            9.0, 3.0, 0.0, 3.0, //
            11.0, 7.0, 3.0, 0.0,
        ],
    )?;
    println!(
        "  four nodes, {} ordered pairs whose direct path is slower than a relay",
        geo.triangle_violations()
    );
    let tight = geo.tighten();
    println!(
        "  direct d(0,3) = {:.0} ticks, best relayed = {:.0}",
        geo.delay(0, 3),
        tight.delay(0, 3)
    );
    println!(
        "  temporal diameter {:.0} → {:.0} after tightening; min horizon H = {}",
        geo.temporal_diameter(),
        tight.temporal_diameter(),
        geo.min_horizon()
    );
    let h = geo.min_horizon();
    for k in [2usize, 4, 8] {
        println!(
            "  a (k = {k}) window + one one-way hop of 2 ticks against H = {h}: {}",
            if fits_in_horizon(k, 2.0, h) {
                "fits — the repair lands inside the barrier and the loss is invisible"
            } else {
                "over budget — the parity arrives after the tick has sealed"
            }
        );
    }

    let mut barrier = Barrier::new(4, h)?;
    for p in 0..4 {
        barrier.deliver(p, 0)?;
    }
    barrier.advance(1);
    println!("\n  tick 0: {:?}", barrier.try_seal());
    barrier.deliver(0, 1)?;
    barrier.deliver(1, 1)?;
    barrier.deliver(3, 1)?;
    barrier.advance(3);
    println!("  tick 1: {:?}", barrier.try_seal());
    barrier.reconstruct(2, 1)?;
    println!(
        "  tick 1: {:?}   (parity landed inside H)",
        barrier.try_seal()
    );
    barrier.advance(400);
    println!("  tick 2: {:?}", barrier.try_seal());

    // ══ 5. the dual tower, crossed ══
    rule("5. the quantum code is a layer of the network's code");
    const NODES: usize = 4;
    const TICKS: usize = 6;
    let codes: Vec<ToricCode> = (0..NODES)
        .map(|n| ToricCode::new(2, 8 * n).unwrap())
        .collect();
    let fine: Vec<ErasureDecoder> = codes.iter().map(ErasureDecoder::new).collect();
    let syn: Vec<SyndromeMap> = codes
        .iter()
        .map(|c| SyndromeMap::new(c).unwrap())
        .collect::<Vec<_>>();
    let mut rng = Prng::new(0xFACE);
    let mut truth = vec![vec![PauliString::identity(); TICKS]; NODES];
    let mut supports = vec![vec![Vec::new(); TICKS]; NODES];
    for (n, (trow, srow)) in truth.iter_mut().zip(supports.iter_mut()).enumerate() {
        for (t, (cell, sup)) in trow.iter_mut().zip(srow.iter_mut()).enumerate() {
            let _ = t;
            let q = 8 * n + (rng.next_u64() % 8) as usize;
            let kind = rng.next_u64() % 3;
            *cell = PauliString {
                x: if kind != 1 { 1 << q } else { 0 },
                z: if kind != 0 { 1 << q } else { 0 },
                negative: false,
            };
            *sup = vec![q];
        }
    }
    let lost = [(0usize, 2usize), (0, 3), (1, 2), (1, 3), (2, 2)];
    let layer = DualLayer::new(&fine, &syn, 2)?;
    println!(
        "  {NODES} nodes × {TICKS} ticks, coarse code ({NODES}, {}) across the nodes at",
        layer.coarse().parity_shards()
    );
    println!(
        "  each tick — {:.0}% bandwidth. The fine layer is the code itself: free.",
        layer.coarse().overhead() * 100.0
    );
    println!("\n  losses: node0@{{2,3}}  node1@{{2,3}}  node2@{{2}}");
    println!("    tick 2 loses 3 nodes against 2 parity shards  → coarse blocked");
    println!("    nodes 0 and 1 each hold 2 unknowns, 1 equation → fine blocked\n");

    let build = |parity: bool| -> quantsim::Result<RepairWindow> {
        let shards = if parity { 2 } else { 1 };
        let mut w = RepairWindow::new(NODES, TICKS, 0, shards)?;
        for t in 0..TICKS {
            if parity {
                let deltas: Vec<FrameDelta> = (0..NODES)
                    .map(|n| FrameDelta {
                        node: n,
                        tick: t as u64,
                        delta: truth[n][t],
                    })
                    .collect();
                for (s, shard) in layer.encode_tick(&deltas)?.into_iter().enumerate() {
                    w.set_parity(t as u64, s, shard)?;
                }
            }
            for n in 0..NODES {
                w.set_support(n, t as u64, supports[n][t].clone())?;
                if !lost.contains(&(n, t)) {
                    w.deliver(n, t as u64, truth[n][t])?;
                }
            }
        }
        for n in 0..NODES {
            let residual = lost
                .iter()
                .filter(|&&(ln, _)| ln == n)
                .fold(0u64, |acc, &(_, t)| acc ^ syn[n].bits(truth[n][t]));
            w.set_residual(n, residual)?;
        }
        Ok(w)
    };

    // coarse alone: strip the schedule so the fine layer has nothing to say
    let mut w = build(true)?;
    for n in 0..NODES {
        w.set_support(n, 2, Vec::new())?;
        w.set_support(n, 3, Vec::new())?;
    }
    let r = layer.repair(&mut w)?;
    println!(
        "  coarse alone: {} repaired, {} left {:?}",
        r.coarse_repairs,
        r.unresolved.len(),
        r.unresolved
    );

    // fine alone: no parity at all
    let solo = DualLayer::new(&fine, &syn, 1)?;
    let mut w = build(false)?;
    let r = solo.repair(&mut w)?;
    println!(
        "  fine   alone: {} repaired, {} left {:?}",
        r.fine_repairs,
        r.unresolved.len(),
        r.unresolved
    );

    // crossed
    let mut w = build(true)?;
    let r = layer.repair(&mut w)?;
    let mut exact = 0;
    for (n, row) in truth.iter().enumerate() {
        for (t, want) in row.iter().enumerate() {
            if w.slot(n, t as u64)? == Some(*want) {
                exact += 1;
            }
        }
    }
    println!(
        "  crossed:      {} coarse + {} fine over {} rounds, {} left — {}/{} slots byte-exact",
        r.coarse_repairs,
        r.fine_repairs,
        r.rounds,
        r.unresolved.len(),
        exact,
        NODES * TICKS
    );

    println!(
        "\n  Each layer's successes shrink the other's unknown count. Neither reaches\n  \
         the pattern alone; the iteration does — and the layer that costs nothing is\n  \
         the code that was already protecting the qubits."
    );
    Ok(())
}
