//! The address algebra: canonical forms as composable addresses, and
//! what closing the address space under composition actually buys.

use quantsim::address::{Addr, AddressSpace, Node, Scale};
use quantsim::exact::ExactState;
use quantsim::pathsum::{PathSum, Pivot};
use quantsim::{Circuit, Scalar, Simulator, C64};

/// A 2D graph state with `layers` rounds of CZ and a T+H layer between —
/// local coupling, genuine magic, and a residual that does not vanish.
fn grid(side: usize, layers: usize) -> Circuit<C64> {
    let n = side * side;
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    for _ in 0..layers {
        for y in 0..side {
            for x in 0..side {
                let s = y * side + x;
                if x + 1 < side {
                    c.cz(s, s + 1);
                }
                if y + 1 < side {
                    c.cz(s, s + side);
                }
            }
        }
        for q in 0..n {
            c.t(q);
            c.h(q);
        }
    }
    c
}

fn ghz(n: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    c.h(0);
    for q in 1..n {
        c.cx(q - 1, q);
    }
    c
}

#[test]
fn an_address_evaluates_to_the_amplitude() {
    // Against the dense reference, against the existing merged route,
    // and against the exact ring — the address is a different route to
    // the same number, not a different number.
    let sim: Simulator = Simulator::new();
    let mut worst = 0.0f64;
    for c in [grid(2, 1), grid(2, 2), grid(3, 1), ghz(4)] {
        let n = c.num_qubits();
        let ps = PathSum::from_circuit(&c).unwrap();
        let dense = sim.run(&c).unwrap();
        let exact = ExactState::run(&c).unwrap();
        let mut sp = AddressSpace::new();
        for b in 0..(1u64 << n) {
            let a = sp.address(&ps, b, 1_000_000).unwrap();
            let v = sp.value(a).unwrap();
            let (m, _) = ps.amplitude_merged(b, 1_000_000).unwrap();
            let e = sp.value_exact(a).unwrap();
            worst = worst
                .max((v - dense.amplitude(b)).abs_sqr().sqrt())
                .max((v - m).abs_sqr().sqrt())
                .max(
                    (e.to_c64() - exact.amplitude_exact(b).to_c64())
                        .abs_sqr()
                        .sqrt(),
                );
        }
    }
    assert!(worst < 1e-12, "worst deviation {worst:e}");
}

#[test]
fn the_exact_route_decides_zero_rather_than_testing_a_tolerance() {
    // h z h = X, so ⟨00|ψ⟩ is exactly zero — and `is_zero` says so from
    // the ring, not from a threshold.
    let mut c: Circuit<C64> = Circuit::new(2);
    c.h(0);
    c.z(0);
    c.h(0);
    c.cx(0, 1);
    let ps = PathSum::from_circuit(&c).unwrap();
    let mut sp = AddressSpace::new();
    let zeros: Vec<u64> = (0..4)
        .filter(|&b| {
            let a = sp.address(&ps, b, 10_000).unwrap();
            sp.is_zero(a).unwrap()
        })
        .collect();
    assert_eq!(zeros, vec![0, 1, 2], "only |11⟩ survives");
    let a = sp.address(&ps, 3, 10_000).unwrap();
    assert!(!sp.is_zero(a).unwrap());
    // Exactly 1, with no floating point anywhere in the evaluation.
    let (coeffs, k) = sp.value_exact(a).unwrap().parts();
    assert_eq!((coeffs, k), ([1, 0, 0, 0], 0));
}

#[test]
fn the_space_outlives_the_query() {
    // The move the old `CanonKey -> C64` memo cannot make: it is scoped
    // to one call. A content-addressed space answers the second
    // amplitude out of what the first one built.
    let c = grid(3, 2);
    let ps = PathSum::from_circuit(&c).unwrap();
    let queries = 64u64;

    let mut shared = AddressSpace::new();
    let mut first = 0u64;
    let mut last = 0u64;
    for b in 0..queries {
        let before = shared.stats().builds;
        shared.address(&ps, b, 5_000_000).unwrap();
        let cost = shared.stats().builds - before;
        if b == 0 {
            first = cost;
        }
        last = cost;
    }

    let mut isolated = 0u64;
    for b in 0..queries {
        let mut fresh = AddressSpace::new();
        fresh.address(&ps, b, 5_000_000).unwrap();
        isolated += fresh.stats().builds;
    }

    let reuse = isolated as f64 / shared.stats().builds as f64;
    assert!(
        reuse > 5.0,
        "reuse only {reuse:.2}x ({isolated} vs {})",
        shared.stats().builds
    );
    assert!(
        last * 10 < first,
        "the last query should cost a fraction of the first: {first} then {last}"
    );
    assert!(shared.stats().form_hits > shared.stats().builds / 2);
}

#[test]
fn the_same_symbol_at_every_width_is_one_address() {
    // A GHZ amplitude is √2^-1 whatever the width. The address algebra
    // says so structurally: four circuits, four widths, one node.
    let mut sp = AddressSpace::new();
    let mut addrs: Vec<Addr> = Vec::new();
    for n in [4usize, 8, 12, 16] {
        let ps = PathSum::from_circuit(&ghz(n)).unwrap();
        addrs.push(sp.address(&ps, 0, 10_000).unwrap());
    }
    assert!(addrs.windows(2).all(|w| AddressSpace::same(w[0], w[1])));
    assert_eq!(sp.len(), 1, "the whole space is one address");
    assert_eq!(
        *sp.node(addrs[0]),
        Node::Product {
            scale: Scale { turn: 0, half: -1 },
            parts: Vec::new(),
        },
        "h* = 0 means a closed leaf: no parts to sum at all"
    );
}

#[test]
fn equal_addresses_are_equal_amplitudes_with_no_arithmetic() {
    // Two different circuits with the same amplitude at a basis state
    // reach the same address, and the test performs no arithmetic.
    let mut sp = AddressSpace::new();
    let mut a: Circuit<C64> = Circuit::new(2);
    a.h(0);
    a.cx(0, 1);
    let mut b: Circuit<C64> = Circuit::new(2);
    b.h(1);
    b.cx(1, 0);
    let pa = PathSum::from_circuit(&a).unwrap();
    let pb = PathSum::from_circuit(&b).unwrap();
    let x = sp.address(&pa, 0, 1000).unwrap();
    let y = sp.address(&pb, 0, 1000).unwrap();
    assert!(AddressSpace::same(x, y));
    // And the values agree, which is the property the identity stands for.
    assert_eq!(sp.value(x).unwrap(), sp.value(y).unwrap());
}

#[test]
fn the_constructors_form_an_algebra_over_addresses() {
    let mut sp = AddressSpace::new();
    let z = sp.zero();
    let one = sp.scalar(Scale::ONE);
    let half = sp.scalar(Scale { turn: 0, half: -1 });

    // Interning: the same node built twice is one address.
    assert!(AddressSpace::same(one, sp.scalar(Scale::ONE)));
    assert!(sp.stats().node_hits >= 1);

    // Products commute on addresses.
    let p1 = sp.product(Scale::ONE, vec![one, half]);
    let p2 = sp.product(Scale::ONE, vec![half, one]);
    assert!(
        AddressSpace::same(p1, p2),
        "order is not part of the symbol"
    );

    // The zero identities, exercised directly.
    assert!(AddressSpace::same(sp.product(Scale::ONE, vec![one, z]), z));
    assert!(AddressSpace::same(sp.sum(z, half), half));
    assert!(AddressSpace::same(sp.sum(half, z), half));
    assert_eq!(sp.stats().zero_products, 1);
    assert_eq!(sp.stats().zero_sums, 2);

    // A branch is a sum, and it evaluates as one.
    let s = sp.sum(half, half);
    assert!((sp.value(s).unwrap() - C64::new(2f64.sqrt(), 0.0)).abs_sqr() < 1e-24);
    assert!(sp.is_zero(z).unwrap());
}

#[test]
fn every_pivot_policy_reaches_the_same_value() {
    let c = grid(2, 2);
    let ps = PathSum::from_circuit(&c).unwrap();
    for b in 0..16u64 {
        let mut vals = Vec::new();
        for p in [
            Pivot::First,
            Pivot::MaxDegree,
            Pivot::MinRemainder,
            Pivot::Elected,
        ] {
            let mut sp = AddressSpace::new();
            let a = sp
                .address_by(&ps, &|q| b >> q & 1 == 1, 1_000_000, p)
                .unwrap();
            vals.push(sp.value(a).unwrap());
        }
        for v in &vals[1..] {
            assert!((*v - vals[0]).abs_sqr() < 1e-24, "policies disagree at {b}");
        }
    }
}

#[test]
fn the_budget_refuses_by_name() {
    let c = grid(4, 2);
    let ps = PathSum::from_circuit(&c).unwrap();
    let mut sp = AddressSpace::new();
    let err = sp.address(&ps, 0, 50).unwrap_err();
    let text = format!("{err}");
    assert!(text.contains("composition budget 50"), "{text}");
    assert!(
        text.contains("refuses rather than quietly enumerating"),
        "{text}"
    );
}

#[test]
fn a_turn_outside_the_ring_refuses_exact_evaluation_rather_than_rounding() {
    // The symbol is exact for any dyadic turn; *exact evaluation* is
    // the eighth-turn fragment, and anything else is named, not rounded.
    let s = Scale {
        turn: 1 << 58,
        half: 0,
    };
    assert!(!s.is_eighth());
    let err = format!("{}", s.to_exact().unwrap_err());
    assert!(err.contains("not in D[ω]"), "{err}");
    // The float route still works, and agrees with the angle.
    let v = s.to_c64();
    let want = std::f64::consts::TAU / 64.0;
    assert!((v.re - want.cos()).abs() < 1e-12 && (v.im - want.sin()).abs() < 1e-12);
}

#[test]
fn a_structurally_zero_output_costs_no_composition_at_all() {
    // An output form that is a constant disagreeing with the bit asked
    // for: the amplitude is zero before any residual is looked at.
    let mut c: Circuit<C64> = Circuit::new(2);
    c.x(0);
    let ps = PathSum::from_circuit(&c).unwrap();
    let mut sp = AddressSpace::new();
    let a = sp.address(&ps, 0b00, 1000).unwrap();
    assert_eq!(*sp.node(a), Node::Zero);
    assert_eq!(sp.stats().builds, 0, "nothing was composed");
    assert!(sp.is_zero(a).unwrap());
    let b = sp.address(&ps, 0b01, 1000).unwrap();
    assert!(!sp.is_zero(b).unwrap());
}

#[test]
fn a_branch_child_can_vanish_and_the_sum_identity_fires() {
    // The module previously claimed the reduction consumed the
    // vanishing pattern before any branch could produce it as a child.
    // It does not. This eleven-gate circuit is the shortest witness a
    // 60,000-circuit search found, and it is here so the claim cannot
    // quietly come back.
    let mut c: Circuit<C64> = Circuit::new(2);
    c.h(0);
    c.t(0);
    c.h(0);
    c.h(1);
    c.cz(0, 1);
    c.h(0);
    c.h(1);
    c.s(1);
    c.t(0);
    c.h(1);
    c.h(0);
    let ps = PathSum::from_circuit(&c).unwrap();
    let mut sp = AddressSpace::new();
    for b in 0..4u64 {
        sp.address(&ps, b, 100_000).unwrap();
    }
    assert_eq!(sp.stats().zero_sums, 2, "0 + x = x fired on a real circuit");
    assert_eq!(sp.stats().zero_products, 0, "x · 0 did not");

    // And it is still the right answer.
    let sim: Simulator = Simulator::new();
    let dense = sim.run(&c).unwrap();
    for b in 0..4u64 {
        let a = sp.address(&ps, b, 100_000).unwrap();
        assert!((sp.value(a).unwrap() - dense.amplitude(b)).abs_sqr() < 1e-24);
    }
}

#[test]
fn one_query_costs_exactly_what_the_merge_solver_costs() {
    // The checkable form of "the cost is unchanged": the address route
    // is the same recursion under the same pivot, so a single query
    // performs one composition per merge-solver node. Everything the
    // module adds is reuse *between* queries, not within one.
    for c in [grid(2, 2), grid(3, 1), grid(3, 2), grid(4, 1)] {
        let ps = PathSum::from_circuit(&c).unwrap();
        let n = c.num_qubits().min(6);
        for b in 0..(1u64 << n) {
            let mut sp = AddressSpace::new();
            sp.address(&ps, b, 5_000_000).unwrap();
            let (_, st) = ps.amplitude_merged(b, 5_000_000).unwrap();
            assert_eq!(
                sp.stats().builds,
                st.nodes,
                "one query, one composition per node ({} qubits, basis {b})",
                c.num_qubits()
            );
        }
    }
}
