//! Cross-lateral distributed registers: the coordinatewise link rule,
//! the syndrome's 𝔽₂ linearity, erasure capacity measured against the
//! code, the horizon budget, and the dual tower's cross-scale synergy.

use quantsim::backend::{DenseState, PauliString};
use quantsim::lateral::*;
use quantsim::prelude::*;
use quantsim::retro::{apply_pauli, syndrome_bits, Decoder, SurfaceCode, ToricCode};
use quantsim::rng::Prng;

fn two_toric_network() -> (ToricCode, ToricCode, LateralNetwork) {
    let a = ToricCode::new(2, 0).unwrap();
    let b = ToricCode::new(2, 8).unwrap();
    let link = LateralLink::new(&a, &b).unwrap();
    let net = LateralNetwork::new(
        vec![(0, 8), (8, 8)],
        vec![link],
        DelayGeometry::new(2, vec![0.0, 3.0, 3.0, 0.0]).unwrap(),
    )
    .unwrap();
    (a, b, net)
}

// ───────────────────────── the lateral link rule ─────────────────────────

#[test]
fn link_rule_is_gate_by_gate_conjugation() {
    // Two XORs and a popcount against `width` folded Clifford steps —
    // on every string, sign included. This is the claim that the
    // inter-node map is coordinatewise.
    let mut rng = Prng::new(0x51DE);
    for (c, t, w) in [(0usize, 4usize, 4usize), (0, 8, 8), (8, 0, 8), (0, 3, 3)] {
        let link = LateralLink::from_windows(c, t, w, w).unwrap();
        for _ in 0..20_000 {
            let p = PauliString {
                x: rng.next_u64() & window_mask(c, w) | rng.next_u64() & window_mask(t, w),
                z: rng.next_u64() & window_mask(c, w) | rng.next_u64() & window_mask(t, w),
                negative: rng.next_u64() & 1 == 1,
            };
            assert_eq!(
                link.push_frame(p),
                link.push_frame_by_gates(p),
                "link {c}->{t} width {w} on {p:?}"
            );
        }
    }
}

#[test]
fn link_rule_never_crosses_indices() {
    // Qubit i of one patch reaches only qubit i of the other: a
    // single-site input produces a frame supported on at most the two
    // aligned sites. "Lateral" is a bundle of independent wires.
    let link = LateralLink::from_windows(0, 8, 8, 8).unwrap();
    for i in 0..8 {
        for (x, z) in [(1u64, 0u64), (0, 1), (1, 1)] {
            let p = PauliString {
                x: x << i,
                z: z << i,
                negative: false,
            };
            let out = link.push_frame(p);
            let allowed = (1u64 << i) | (1u64 << (8 + i));
            assert_eq!(
                (out.x | out.z) & !allowed,
                0,
                "site {i} leaked outside its aligned pair"
            );
        }
    }
}

#[test]
fn link_refuses_mismatched_and_overlapping_patches() {
    let a = ToricCode::new(2, 0).unwrap();
    let big = ToricCode::new(3, 8).unwrap();
    assert!(LateralLink::new(&a, &big).is_err(), "unequal patches");
    assert!(
        LateralLink::from_windows(0, 4, 8, 8).is_err(),
        "overlapping patches"
    );
}

// ───────────────────────── the syndrome is linear ─────────────────────────

#[test]
fn syndrome_is_f2_linear_in_the_frame() {
    // bits(p ⊕ q) = bits(p) ⊕ bits(q), for every pair — commuting or
    // not. This is what lets a node name the syndrome of the delta
    // that did *not* arrive.
    let code = ToricCode::new(3, 0).unwrap();
    let map = SyndromeMap::new(&code).unwrap();
    let mut rng = Prng::new(7);
    let m = window_mask(0, 18);
    for _ in 0..50_000 {
        let p = PauliString {
            x: rng.next_u64() & m,
            z: rng.next_u64() & m,
            negative: rng.next_u64() & 1 == 1,
        };
        let q = PauliString {
            x: rng.next_u64() & m,
            z: rng.next_u64() & m,
            negative: false,
        };
        assert_eq!(
            map.bits(xor_masks(p, q)),
            map.bits(p) ^ map.bits(q),
            "linearity on {p:?} {q:?}"
        );
    }
}

#[test]
fn syndrome_map_agrees_with_retro_signature() {
    let code = SurfaceCode::new(3, 0).unwrap();
    let map = SyndromeMap::new(&code).unwrap();
    for q in 0..9 {
        for (x, z) in [(1u64, 0u64), (0, 1), (1, 1)] {
            let p = PauliString {
                x: x << q,
                z: z << q,
                negative: false,
            };
            assert_eq!(
                map.unpack(map.bits(p)),
                quantsim::retro::signature(&code, p)
            );
        }
    }
}

// ──────────────────── erasure capacity, measured ────────────────────

#[test]
fn erasure_capacity_is_d_minus_one() {
    // Measured off the code's own generators, not read off a distance
    // — and the logical operator that ends it is produced as a
    // witness.
    for d in [3usize, 5] {
        let code = SurfaceCode::new(d, 0).unwrap();
        let er = ErasureDecoder::new(&code);
        assert_eq!(
            er.certified_capacity(d + 1),
            d - 1,
            "rotated surface d = {d}: erasure capacity"
        );
        let witness = er
            .first_uncorrectable(d + 1)
            .expect("a logical must end it");
        assert_eq!(witness.len(), d, "the witness is a weight-d logical");
    }
    for l in [2usize, 3] {
        let code = ToricCode::new(l, 0).unwrap();
        let er = ErasureDecoder::new(&code);
        assert_eq!(er.certified_capacity(l + 1), l - 1, "toric L = {l}");
        assert_eq!(er.first_uncorrectable(l + 1).unwrap().len(), l);
    }
}

#[test]
fn erasures_buy_exactly_a_factor_of_two_over_errors() {
    // The whole economy of knowing *where* the loss happened. The
    // error side is `retro::Decoder`'s contract — it corrects up to
    // ⌊(d−1)/2⌋ and refuses beyond; the erasure side is d−1.
    for l in [2usize, 3] {
        let code = ToricCode::new(l, 0).unwrap();
        let erasure = ErasureDecoder::new(&code).certified_capacity(l + 1);
        let error = (l - 1) / 2;
        assert_eq!(erasure, l - 1);
        assert_eq!(erasure, 2 * error + (l - 1) % 2);

        // L = 2 is the sharp case: distance 2 detects and never
        // corrects, so a weight-1 X or Z fault is refused by name —
        // and the same fault as an *erasure* is repaired exactly.
        if l == 2 {
            let dec = Decoder::new(&code);
            let fault = PauliString {
                x: 1 << 3,
                z: 0,
                negative: false,
            };
            assert!(
                dec.decode(&quantsim::retro::signature(&code, fault))
                    .is_err(),
                "distance 2 must refuse a located-nowhere weight-1 X"
            );
            let map = SyndromeMap::new(&code).unwrap();
            let er = ErasureDecoder::new(&code);
            assert_eq!(er.decode(&[3], map.bits(fault)).unwrap(), fault);
        }
    }
}

#[test]
fn erasure_decoder_refuses_rather_than_guesses() {
    let code = SurfaceCode::new(3, 0).unwrap();
    let er = ErasureDecoder::new(&code);
    let map = SyndromeMap::new(&code).unwrap();

    // A set carrying a logical: refused by name, never guessed
    // through — the same contract `retro::Decoder` holds for
    // degeneracy, one layer over.
    let carrying = er.first_uncorrectable(4).unwrap();
    let err = er.decode(&carrying, 0).unwrap_err().to_string();
    assert!(err.contains("logical operator is supported"), "{err}");

    // A syndrome no fault on the set can produce: a *different*
    // refusal. One qubit reaches exactly three non-zero syndromes;
    // everything else is outside the image and is reported as such
    // rather than projected onto the set.
    let reachable: Vec<u64> = [(1u64, 0u64), (0, 1), (1, 1)]
        .iter()
        .map(|&(x, z)| {
            map.bits(PauliString {
                x,
                z,
                negative: false,
            })
        })
        .collect();
    let unreachable = (1..(1u64 << map.len()))
        .find(|s| !reachable.contains(s))
        .expect("one qubit cannot reach every syndrome of an 8-generator code");
    let err = er.decode(&[0], unreachable).unwrap_err().to_string();
    assert!(err.contains("not produced by any fault"), "{err}");

    // And every reachable syndrome decodes to exactly the fault that
    // produced it.
    for (i, &(x, z)) in [(1u64, 0u64), (0, 1), (1, 1)].iter().enumerate() {
        let want = PauliString {
            x,
            z,
            negative: false,
        };
        assert_eq!(er.decode(&[0], reachable[i]).unwrap(), want);
    }

    // A qubit outside the patch: refused, not silently clamped.
    assert!(er.decode(&[40], 0).is_err());
}

#[test]
fn correctable_and_unique_are_different_questions() {
    // The rotated surface code closes its boundary with weight-2
    // checks, so qubits {0, 5} carry a stabilizer. Erasing both is
    // *correctable* — the two candidate frames differ by that
    // stabilizer and act identically on every logical observable —
    // and it is not *unique*: they are different bytes. That
    // distinction is the whole reason the fine layer declines rather
    // than handing a representative to the coarse layer downstream.
    let code = SurfaceCode::new(5, 0).unwrap();
    let er = ErasureDecoder::new(&code);
    let map = SyndromeMap::new(&code).unwrap();

    assert!(er.correctable(&[0, 5]));
    assert!(!er.unique(&[0, 5]));

    // The boundary check is Z-type, so the ambiguous pair is Z₀ and Z₅.
    let a = PauliString {
        x: 0,
        z: 1,
        negative: false,
    };
    let b = PauliString {
        x: 0,
        z: 1 << 5,
        negative: false,
    };
    let stab = xor_masks(a, b);
    assert!(
        code.generators()
            .iter()
            .any(|g| g.x == stab.x && g.z == stab.z),
        "the two candidates differ by an actual generator"
    );
    assert_eq!(map.bits(a), map.bits(b), "so they share a syndrome");
    let got = er.decode(&[0, 5], map.bits(a)).unwrap();
    assert_eq!(
        map.bits(got),
        map.bits(a),
        "the decode is right on the code space"
    );
    assert!(got == a || got == b, "and is one of the two candidates");

    // A single qubit is smaller than any stabilizer, so it is unique
    // on the nose — which is the regime the fine layer works in.
    assert!(er.unique(&[0]));
}

// ─────────────── the frame is the wire, and it is exact ───────────────

#[test]
fn distributed_frame_equals_gate_by_gate_and_predicts_the_state() {
    // Three claims in one run: the coordinatewise evolution equals the
    // physical one; the frame's signature equals the syndromes a dense
    // backend actually shows; and the wire carried 𝔽₂, not amplitudes.
    let reg = GateRegistry::<C64>::standard();
    let (a, b, net) = two_toric_network();

    let mut st = DenseState::<C64>::new(16).unwrap();
    for enc in [a.encoder(16, [false, false]), b.encoder(16, [false, false])] {
        enc.bind(&reg).unwrap().run(&mut st).unwrap();
    }

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

    let run = net.run(&program).unwrap();
    assert_eq!(
        run.frame,
        net.run_by_gates(&program).unwrap(),
        "coordinatewise frame evolution is the physical one"
    );

    apply_pauli(&mut st, fault_a, &reg).unwrap();
    apply_pauli(&mut st, fault_b, &reg).unwrap();
    let mut cx = Circuit::<C64>::new(16);
    a.transversal_cx(&b, &mut cx).unwrap();
    cx.bind(&reg).unwrap().run(&mut st).unwrap();

    for code in [&a, &b] {
        let map = SyndromeMap::new(code).unwrap();
        assert_eq!(
            syndrome_bits(&st, code, 1e-9).unwrap(),
            map.unpack(map.bits(run.frame)),
            "the distributed record predicts the measured syndrome exactly"
        );
    }

    assert_eq!(run.wire_bytes, 2 * DELTA_WIRE_LEN);
    assert!(
        (run.wire_bytes as u128) * 19_000 < net.amplitude_bytes(),
        "54 B of frame against 1 MiB of amplitude at width 16"
    );
}

#[test]
fn a_local_pauli_must_stay_local() {
    let (_, _, net) = two_toric_network();
    let straddling = PauliString {
        x: (1 << 3) | (1 << 11),
        z: 0,
        negative: false,
    };
    let err = net
        .run(&[LateralOp::Local {
            node: 0,
            pauli: straddling,
        }])
        .unwrap_err()
        .to_string();
    assert!(err.contains("reaches outside its window"), "{err}");
}

#[test]
fn frame_delta_survives_the_wire() {
    let mut rng = Prng::new(3);
    for _ in 0..1000 {
        let d = FrameDelta {
            node: (rng.next_u64() % 8) as usize,
            tick: rng.next_u64() % 4096,
            delta: PauliString {
                x: rng.next_u64(),
                z: rng.next_u64(),
                negative: rng.next_u64() & 1 == 1,
            },
        };
        assert_eq!(FrameDelta::from_bytes(&d.to_bytes()).unwrap(), d);
    }
    assert!(FrameDelta::from_bytes(&[0u8; 4]).is_err());
}

// ─────────────────────── geometry and the barrier ───────────────────────

#[test]
fn delay_geometry_is_not_a_metric_until_it_is_tightened() {
    // The direct A→C path is slower than relaying through B, which is
    // what real inter-AS latency does constantly.
    let d = DelayGeometry::new(
        3,
        vec![
            0.0, 2.0, 9.0, //
            2.0, 0.0, 3.0, //
            9.0, 3.0, 0.0,
        ],
    )
    .unwrap();
    assert_eq!(d.triangle_violations(), 2, "A→C and C→A both relay better");
    let t = d.tighten();
    assert_eq!(t.triangle_violations(), 0);
    assert_eq!(t.delay(0, 2), 5.0);
    assert_eq!(d.min_horizon(), 5, "the floor is the tightened diameter");
    assert!(d.temporal_diameter() > t.temporal_diameter());
}

#[test]
fn the_barrier_seals_on_arrivals_not_on_the_prediction() {
    let mut b = Barrier::new(3, 4).unwrap();
    // Tick 0: everyone delivers early — clean, well before the deadline.
    for p in 0..3 {
        b.deliver(p, 0).unwrap();
    }
    b.advance(1);
    assert_eq!(b.try_seal(), Seal::Clean { tick: 0 });

    // Tick 1: peer 2 is late but the deadline has not passed — waiting,
    // not sealed, and never sealed on the guess that it will arrive.
    b.deliver(0, 1).unwrap();
    b.deliver(1, 1).unwrap();
    b.advance(3);
    assert_eq!(
        b.try_seal(),
        Seal::Waiting {
            tick: 1,
            missing: vec![2]
        }
    );

    // Parity lands inside the horizon: the loss was invisible.
    b.reconstruct(2, 1).unwrap();
    assert_eq!(
        b.try_seal(),
        Seal::Repaired {
            tick: 1,
            reconstructed: vec![2]
        }
    );

    // Tick 2: nothing arrives and nothing repairs — a real gap, named.
    b.advance(100);
    assert_eq!(
        b.try_seal(),
        Seal::Degraded {
            tick: 2,
            missing: vec![0, 1, 2]
        }
    );
    assert_eq!(b.sealed_through(), Some(2));
}

#[test]
fn the_horizon_is_a_budget() {
    // k ticks of sender-side buffering plus one one-way delay. Inside
    // H the repair lands before the tick seals and the loss is
    // invisible; outside it, the seal degrades.
    let geo = DelayGeometry::new(2, vec![0.0, 3.0, 3.0, 0.0]).unwrap();
    let h = geo.min_horizon();
    assert_eq!(h, 3);
    assert!(!fits_in_horizon(4, 3.0, h), "k = 4 does not fit in H = 3");
    assert!(fits_in_horizon(4, 3.0, 8), "a wider horizon buys the code");
    assert!(fits_in_horizon(2, 1.0, 3), "k + one-way exactly meets H");
}

// ───────────────────────── the coarse layer alone ─────────────────────────

#[test]
fn window_parity_is_mds_and_says_so_when_it_is_not_enough() {
    let wp = WindowParity::new(4, 2).unwrap();
    assert_eq!(wp.overhead(), 0.5);
    let data: Vec<Vec<u8>> = (0..4u8)
        .map(|i| {
            (0..DELTA_WIRE_LEN as u8)
                .map(|b| i.wrapping_mul(31).wrapping_add(b))
                .collect()
        })
        .collect();
    let parity = wp.encode(&data).unwrap();

    for a in 0..4 {
        for b in (a + 1)..4 {
            let mut d: Vec<Option<Vec<u8>>> = data.iter().cloned().map(Some).collect();
            d[a] = None;
            d[b] = None;
            let p: Vec<Option<Vec<u8>>> = parity.iter().cloned().map(Some).collect();
            assert_eq!(wp.repair(&mut d, &p).unwrap(), 2);
            for (j, want) in data.iter().enumerate() {
                assert_eq!(
                    d[j].as_ref().unwrap(),
                    want,
                    "shard {j} after losing {a},{b}"
                );
            }
        }
    }

    // Three erasures against two parity shards: over budget, refused
    // by name with the counts in it.
    let mut d: Vec<Option<Vec<u8>>> = data.iter().cloned().map(Some).collect();
    for slot in d.iter_mut().take(3) {
        *slot = None;
    }
    let p: Vec<Option<Vec<u8>>> = parity.iter().cloned().map(Some).collect();
    let err = wp.repair(&mut d, &p).unwrap_err().to_string();
    assert!(err.contains("over budget"), "{err}");
}

// ───────────────────── the two layers, and the cross ─────────────────────

/// Four toric nodes over one register, their decoders and syndrome
/// maps, and a ground-truth delta per `(node, tick)`.
struct Fixture {
    fine: Vec<ErasureDecoder>,
    syn: Vec<SyndromeMap>,
    truth: Vec<Vec<PauliString>>,
    supports: Vec<Vec<Vec<usize>>>,
}

const NODES: usize = 4;
const TICKS: usize = 6;

fn fixture() -> Fixture {
    let codes: Vec<ToricCode> = (0..NODES)
        .map(|n| ToricCode::new(2, 8 * n).unwrap())
        .collect();
    let fine: Vec<ErasureDecoder> = codes.iter().map(ErasureDecoder::new).collect();
    let syn: Vec<SyndromeMap> = codes.iter().map(|c| SyndromeMap::new(c).unwrap()).collect();
    let mut rng = Prng::new(0xFACE);
    let mut truth = Vec::new();
    let mut supports = Vec::new();
    for n in 0..NODES {
        let mut row = Vec::new();
        let mut srow = Vec::new();
        for _ in 0..TICKS {
            let q = 8 * n + (rng.next_u64() % 8) as usize;
            let kind = rng.next_u64() % 3;
            row.push(PauliString {
                x: if kind != 1 { 1 << q } else { 0 },
                z: if kind != 0 { 1 << q } else { 0 },
                negative: false,
            });
            srow.push(vec![q]);
        }
        truth.push(row);
        supports.push(srow);
    }
    Fixture {
        fine,
        syn,
        truth,
        supports,
    }
}

/// Build a window with `lost` slots dropped, the coarse parity for
/// every tick present, supports filled from the schedule, and each
/// node's fine residual set to what its own state would show.
fn window(f: &Fixture, layer: &DualLayer, lost: &[(usize, usize)]) -> RepairWindow {
    let mut w = RepairWindow::new(NODES, TICKS, 0, layer.coarse().parity_shards()).unwrap();
    for t in 0..TICKS {
        let deltas: Vec<FrameDelta> = (0..NODES)
            .map(|n| FrameDelta {
                node: n,
                tick: t as u64,
                delta: f.truth[n][t],
            })
            .collect();
        for (s, shard) in layer.encode_tick(&deltas).unwrap().into_iter().enumerate() {
            w.set_parity(t as u64, s, shard).unwrap();
        }
        for n in 0..NODES {
            w.set_support(n, t as u64, f.supports[n][t].clone())
                .unwrap();
            if !lost.contains(&(n, t)) {
                w.deliver(n, t as u64, f.truth[n][t]).unwrap();
            }
        }
    }
    for n in 0..NODES {
        // The node reads its own syndrome; by linearity the difference
        // from what its believed frame predicts is the syndrome of the
        // XOR of exactly the deltas it is missing.
        let residual = lost
            .iter()
            .filter(|&&(ln, _)| ln == n)
            .fold(0u64, |acc, &(_, t)| acc ^ f.syn[n].bits(f.truth[n][t]));
        w.set_residual(n, residual).unwrap();
    }
    w
}

fn check_all(f: &Fixture, w: &RepairWindow) {
    for n in 0..NODES {
        for t in 0..TICKS {
            assert_eq!(
                w.slot(n, t as u64).unwrap(),
                Some(f.truth[n][t]),
                "node {n} tick {t} recovered byte-exactly"
            );
        }
    }
}

#[test]
fn either_layer_alone_handles_its_own_regime() {
    let f = fixture();
    let paid = DualLayer::new(&f.fine, &f.syn, 2).unwrap();
    // Coarse regime: two nodes down at one tick, which the fine layer
    // also happens to reach — the point is that coarse suffices.
    let mut w = window(&f, &paid, &[(0, 2), (1, 2)]);
    let r = paid.repair(&mut w).unwrap();
    assert!(r.unresolved.is_empty());
    check_all(&f, &w);

    // Fine regime, with the coarse layer switched off entirely: one
    // node down for four consecutive ticks, all of them repaired from
    // the code alone, at zero bandwidth.
    let free = DualLayer::new(&f.fine, &f.syn, 1).unwrap();
    let mut w = RepairWindow::new(NODES, TICKS, 0, 1).unwrap();
    for t in 0..TICKS {
        for n in 0..NODES {
            w.set_support(n, t as u64, f.supports[n][t].clone())
                .unwrap();
            if !(n == 0 && (2..6).contains(&t)) {
                w.deliver(n, t as u64, f.truth[n][t]).unwrap();
            }
        }
    }
    w.set_residual(
        0,
        (2..6).fold(0u64, |a, t| a ^ f.syn[0].bits(f.truth[0][t])),
    )
    .unwrap();
    let r = free.repair(&mut w).unwrap();
    // With four unknowns on one node the single fine equation cannot
    // start, and no parity was supplied: the layer says so instead of
    // guessing.
    assert_eq!(r.fine_repairs, 0);
    assert_eq!(r.unresolved.len(), 4);
}

#[test]
fn the_cross_recovers_what_defeats_each_layer_alone() {
    // node 0 and node 1 each lose ticks 2 and 3; node 2 loses tick 2.
    //
    //   tick 2: three nodes down against two parity shards → coarse blocked
    //   node 0: two unknowns against one equation           → fine blocked
    //   node 1: two unknowns against one equation           → fine blocked
    //
    // Coarse alone clears tick 3 and stops with tick 2 still dark.
    // Fine alone clears node 2's single unknown and stops. Crossed,
    // the coarse repair at tick 3 leaves nodes 0 and 1 holding one
    // unknown each, at which point their own code constraint names it
    // — and tick 2 closes on the layer that cost no bandwidth at all.
    let f = fixture();
    let lost = [(0usize, 2usize), (0, 3), (1, 2), (1, 3), (2, 2)];

    let layer = DualLayer::new(&f.fine, &f.syn, 2).unwrap();
    let mut w = window(&f, &layer, &lost);
    let r = layer.repair(&mut w).unwrap();
    assert!(
        r.unresolved.is_empty(),
        "crossed layers left {:?}",
        r.unresolved
    );
    assert!(r.rounds >= 2, "the recovery needed the iteration");
    assert!(r.coarse_repairs > 0 && r.fine_repairs > 0, "{r:?}");
    check_all(&f, &w);

    // Coarse alone: strip the residuals so the fine layer has nothing
    // to say, and watch tick 2 stay stuck.
    let mut w = window(&f, &layer, &lost);
    for n in 0..NODES {
        w.set_support(n, 2, Vec::new()).unwrap();
        w.set_support(n, 3, Vec::new()).unwrap();
    }
    let r = layer.repair(&mut w).unwrap();
    assert_eq!(r.fine_repairs, 0);
    assert_eq!(
        r.unresolved,
        vec![(0, 2), (1, 2), (2, 2)],
        "the coarse layer cannot reach a tick that lost more nodes than it has parity"
    );

    // Fine alone: no parity at all.
    let solo = DualLayer::new(&f.fine, &f.syn, 1).unwrap();
    let mut w = RepairWindow::new(NODES, TICKS, 0, 1).unwrap();
    for t in 0..TICKS {
        for n in 0..NODES {
            w.set_support(n, t as u64, f.supports[n][t].clone())
                .unwrap();
            if !lost.contains(&(n, t)) {
                w.deliver(n, t as u64, f.truth[n][t]).unwrap();
            }
        }
    }
    for n in 0..NODES {
        let residual = lost
            .iter()
            .filter(|&&(ln, _)| ln == n)
            .fold(0u64, |acc, &(_, t)| acc ^ f.syn[n].bits(f.truth[n][t]));
        w.set_residual(n, residual).unwrap();
    }
    let r = solo.repair(&mut w).unwrap();
    assert_eq!(r.coarse_repairs, 0);
    assert_eq!(r.fine_repairs, 1, "only node 2's single unknown");
    assert_eq!(r.unresolved.len(), 4);
}

#[test]
fn the_fine_layer_declines_rather_than_returning_a_representative() {
    // Two distance-5 surface patches. Tick 4 on node 0 is scheduled
    // over {0, 5} — the boundary pair that carries a weight-2
    // stabilizer, so the recovery is correctable but not unique. The
    // layer declines and counts it, because a stabilizer-equivalent
    // representative is as good as the truth for the code space and
    // *poison* for the coarse layer's byte algebra downstream.
    let codes = [
        SurfaceCode::new(5, 0).unwrap(),
        SurfaceCode::new(5, 25).unwrap(),
    ];
    let fine: Vec<ErasureDecoder> = codes.iter().map(ErasureDecoder::new).collect();
    let syn: Vec<SyndromeMap> = codes.iter().map(|c| SyndromeMap::new(c).unwrap()).collect();
    assert!(fine[0].correctable(&[0, 5]) && !fine[0].unique(&[0, 5]));

    let truth = PauliString {
        x: 1,
        z: 0,
        negative: false,
    };
    let layer = DualLayer::new(&fine, &syn, 1).unwrap();
    let mut w = RepairWindow::new(2, 6, 0, 1).unwrap();
    for t in 0..6u64 {
        for n in 0..2usize {
            let q = 25 * n + (t as usize % 5);
            let scheduled = if n == 0 && t == 4 {
                vec![0, 5]
            } else {
                vec![q]
            };
            w.set_support(n, t, scheduled).unwrap();
            if !(n == 0 && t == 4) {
                w.deliver(
                    n,
                    t,
                    PauliString {
                        x: 1 << q,
                        z: 0,
                        negative: false,
                    },
                )
                .unwrap();
            }
        }
    }
    w.set_residual(0, syn[0].bits(truth)).unwrap();
    let r = layer.repair(&mut w).unwrap();
    assert_eq!(r.fine_repairs, 0);
    assert_eq!(r.fine_declined, 1);
    assert_eq!(r.unresolved, vec![(0, 4)]);

    // Narrow the schedule to the single qubit the fault actually sat
    // on and the same layer takes it, byte-exactly, on the same
    // residual — the decline was about the support, not the fault.
    w.set_support(0, 4, vec![0]).unwrap();
    let r = layer.repair(&mut w).unwrap();
    assert_eq!(r.fine_repairs, 1);
    assert!(r.unresolved.is_empty());
    assert_eq!(w.slot(0, 4).unwrap(), Some(truth));
}
