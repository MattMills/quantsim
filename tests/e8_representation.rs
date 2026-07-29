//! E8×E8 as a qubit representation, measured: the spinor bijection
//! (128 + 128 = the full 8-qubit basis), the theorem that 2-local
//! transitions ARE integer roots (verified against actual gate
//! matrices, not asserted), and entanglement structure read off as
//! root geometry — GHZ is an antipodal pair, product states are
//! affine-dimension-0, and the projected-support bound holds on every
//! state and cut tested.

mod common;

use common::assert_close;
use quantsim::e8::{self, rep};
use quantsim::prelude::*;
use std::collections::HashSet;

#[test]
fn the_spinor_sectors_are_the_full_qubit_basis() {
    // 128 even-parity strings ↔ 128 spinor roots, bijectively; the
    // odd sector pairs through qubit 0; together: all 256 basis
    // states on E8×E8.
    let rs = e8::roots();
    let spinors: HashSet<e8::Root> = rs
        .iter()
        .filter(|r| r.iter().all(|&c| c == 1 || c == -1))
        .copied()
        .collect();
    assert_eq!(spinors.len(), 128);

    let mut seen = HashSet::new();
    let mut per_sector = [0usize; 2];
    for bits in 0u16..256 {
        let bits = bits as u8;
        let (copy, root) = rep::sector_of(bits);
        per_sector[copy] += 1;
        assert!(spinors.contains(&root), "sector image is a spinor root");
        if copy == 0 {
            assert_eq!(rep::bits_of_spinor(&root), Some(bits), "roundtrip");
            assert_eq!(rep::spinor_of_bits(bits), Some(root));
        } else {
            assert_eq!(rep::spinor_of_bits(bits), None, "odd strings are copy 1");
        }
        seen.insert((copy, root));
    }
    assert_eq!(per_sector, [128, 128]);
    assert_eq!(seen.len(), 256, "the pairing is injective");
}

#[test]
fn two_local_transitions_are_exactly_the_integer_roots() {
    // Geometry side: every same-parity Hamming-2 pair differs by an
    // integer root, and the transition set covers ALL 112 integer
    // roots.
    let rs = e8::roots();
    let integer_roots: HashSet<e8::Root> = rs
        .iter()
        .filter(|r| r.iter().all(|&c| c % 2 == 0))
        .copied()
        .collect();
    assert_eq!(integer_roots.len(), 112);
    let mut covered = HashSet::new();
    for x in 0u16..256 {
        for y in 0u16..256 {
            let (x, y) = (x as u8, y as u8);
            match rep::transition_root(x, y) {
                Some(root) => {
                    assert_eq!((x ^ y).count_ones(), 2);
                    assert!(integer_roots.contains(&root), "{root:?}");
                    covered.insert(root);
                }
                None => {
                    assert!((x ^ y).count_ones() != 2 || x.count_ones() % 2 != y.count_ones() % 2)
                }
            }
        }
    }
    assert_eq!(covered.len(), 112, "every integer root is a transition");

    // Gate side, the actual physics: the XX(i,j) matrix element
    // ⟨y|XX|x⟩ is nonzero EXACTLY when y = x with bits i,j flipped —
    // i.e. exactly when the transition root with support {i,j}
    // connects them. Verified against the real gate matrix over every
    // pair and every basis state.
    let reg = GateRegistry::<C64>::standard();
    for i in 0..8usize {
        for j in (i + 1)..8 {
            let mut c: Circuit = Circuit::new(8);
            c.gate("rxx", [std::f64::consts::PI], [i, j]);
            let bound = c.bind(&reg).unwrap();
            for x in 0u64..256 {
                let mut state = DenseState::<C64>::new(8).unwrap();
                state.load(&[(x, c64(1.0, 0.0))]).unwrap();
                bound.run(&mut state).unwrap();
                let mut outputs = Vec::new();
                state.for_each_nonzero(&mut |y, amp| {
                    if amp.norm() > 1e-9 {
                        outputs.push(y as u8);
                    }
                });
                assert_eq!(outputs.len(), 1, "rxx(π) is a flip up to phase");
                let y = outputs[0];
                assert_eq!(u64::from(y ^ (x as u8)), (1 << i) | (1 << j));
                let root = rep::transition_root(x as u8, y);
                assert!(root.is_some(), "the gate moved along a root");
                let r = root.unwrap();
                let support: Vec<usize> = (0..8)
                    .filter(|&k| r[k] != 0 && r[k] % 2 == 0 && r[k].abs() == 2)
                    .collect();
                assert_eq!(support, vec![i, j], "the root's support is the gate's");
            }
        }
    }
}

#[test]
fn entanglement_structure_reads_as_root_geometry() {
    let sim: Simulator = Simulator::new();

    // A product state: one support point, affine dimension 0, rank 1
    // across every cut.
    let mut c: Circuit = Circuit::new(8);
    c.x(1).x(4);
    let product = sim.run(&c).unwrap();
    let geo = rep::support_geometry(product.as_ref());
    assert_eq!((geo.points, geo.affine_dim, geo.root_edges), (1, 0, 0));
    for cut in [0b1u8, 0b1111, 0b1010101] {
        assert_eq!(rep::schmidt(product.as_ref(), cut).0, 1);
    }

    // GHZ-8: the support is EXACTLY one antipodal pair of the root
    // geometry (|0…0⟩ ↔ all-plus spinor, |1…1⟩ ↔ its negation), and
    // the entropy across every nontrivial cut is exactly 1 bit.
    let ghz = sim.run(&library::ghz(8)).unwrap();
    let geo = rep::support_geometry(ghz.as_ref());
    assert_eq!(geo.points, 2);
    assert_eq!(geo.antipodal_pairs, 1, "GHZ is an antipode in E8");
    assert_eq!(geo.sectors, (2, 0), "both ends live on the even copy");
    assert_eq!(geo.root_edges, 0, "no 2-local move connects the ends");
    assert_eq!(geo.affine_dim, 1);
    for cut in [0b1u8, 0b11, 0b1111, 0b1010101] {
        let (rank, entropy) = rep::schmidt(ghz.as_ref(), cut);
        assert_eq!(rank, 2);
        assert_close(entropy, 1.0, 1e-9);
    }

    // Rainbow-8 (nested Bell pairs): entropy across the center cut is
    // 4 bits (every pair crosses), across a pair-respecting cut it is
    // 0 — and the support geometry shows the affine structure of a
    // pair product (dimension 4, one flip vector per pair).
    let rainbow = sim.run(&library::rainbow(8)).unwrap();
    let geo = rep::support_geometry(rainbow.as_ref());
    assert_eq!(geo.points, 16);
    assert_eq!(geo.affine_dim, 4);
    let center = 0b00001111u8; // qubits 0..3 vs 4..7: crosses all pairs
    let (rank, entropy) = rep::schmidt(rainbow.as_ref(), center);
    assert_eq!(rank, 16);
    assert_close(entropy, 4.0, 1e-9);
    let respecting = 0b10000001u8; // pair (0,7) together
    let (rank, entropy) = rep::schmidt(rainbow.as_ref(), respecting);
    assert_eq!(rank, 1);
    assert_close(entropy, 0.0, 1e-9);

    // A dense random state spreads: many root edges, full affine
    // dimension, near-maximal rank across the center.
    let random = sim.run(&library::random_circuit(8, 200, 7)).unwrap();
    let geo = rep::support_geometry(random.as_ref());
    assert_eq!(geo.points, 256);
    assert_eq!(geo.affine_dim, 8);
    assert!(geo.root_edges > 3000, "{}", geo.root_edges);
    let (rank, entropy) = rep::schmidt(random.as_ref(), 0b1111);
    assert_eq!(rank, 16);
    assert!(entropy > 3.0, "{entropy}");
}

#[test]
fn the_projected_support_bound_holds_everywhere() {
    // Schmidt rank ≤ min(|projection of support onto either side|) —
    // the geometric entanglement bound, measured on every family and
    // cut, tight for GHZ and rainbow.
    let sim: Simulator = Simulator::new();
    let states = [
        sim.run(&library::ghz(8)).unwrap(),
        sim.run(&library::rainbow(8)).unwrap(),
        sim.run(&library::brickwork(8, 4, &[3])).unwrap(),
        sim.run(&library::random_circuit(8, 200, 7)).unwrap(),
    ];
    for state in &states {
        for cut in [0b1u8, 0b11, 0b111, 0b1111, 0b1010101, 0b1100110] {
            let (rank, _) = rep::schmidt(state.as_ref(), cut);
            let bound = rep::projected_support_bound(state.as_ref(), cut);
            assert!(
                rank <= bound,
                "rank {rank} exceeds the geometric bound {bound}"
            );
        }
    }
    // Tightness where the geometry is exact.
    let ghz = sim.run(&library::ghz(8)).unwrap();
    assert_eq!(rep::projected_support_bound(ghz.as_ref(), 0b1111), 2);
    assert_eq!(rep::schmidt(ghz.as_ref(), 0b1111).0, 2);
}
