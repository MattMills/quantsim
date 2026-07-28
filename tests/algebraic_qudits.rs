//! The hierarchical qubit register: a varied qudit structure over the
//! Cayley–Dickson tower, with dual-algebra gate synthesis — measured,
//! conformance-swept, and certified exact.
//!
//! * the qudit coordinates are honest (roundtrip, bit order);
//! * the dual-algebra (left × right multiplication) operator space is
//!   *measured* per doubling level, and where it resolves the full
//!   ℂ-linear gate space, synthesized sandwich gates agree with the
//!   matrix they synthesize;
//! * the register conforms over the FULL gate registry against dense,
//!   at several site/algebra splits, with the native/component routing
//!   chosen by the measured embedded-linearity of the scalar;
//! * encodings are exact across the site↔algebra boundary and certified
//!   against the D[ω] ring on the Clifford family;
//! * the logical width ceiling of every flat backend (u64 indexing,
//!   63 qubits) is exceeded with exact amplitudes;
//! * the site sector keeps its structure: sparse support counts sites,
//!   the algebra sector rides inside each entry.

mod common;

use common::assert_close;
use quantsim::exact::ExactState;
use quantsim::prelude::*;
use quantsim::qudit::{components, embedded_action_is_component_linear, from_components};

const TOL: f64 = 1e-12;

fn algebraic_over_sparse<A: Scalar>(total: usize, algebra: usize) -> Result<AlgebraicRegister<A>> {
    let k = algebra
        .min(algebra_capacity::<A>())
        .min(total.saturating_sub(1));
    let site = total - k;
    AlgebraicRegister::new(site, k, Box::new(SparseState::<A>::new(site)?))
}

#[test]
fn qudit_coordinates_roundtrip_with_the_documented_bit_order() {
    // Component c of CD^k pairs real coeffs (2c, 2c+1); component bit
    // j is algebra qubit j, the top bit selecting the outer half.
    fn probe<A: Scalar>() {
        let cap = A::DIM / 2;
        for c in 0..cap {
            let mut comps = vec![c64(0.0, 0.0); cap];
            comps[c] = c64(0.25, -0.75);
            let x: A = from_components(&comps);
            let co = x.coeffs();
            assert_eq!(co[2 * c], 0.25);
            assert_eq!(co[2 * c + 1], -0.75);
            assert!(co
                .iter()
                .enumerate()
                .all(|(i, v)| (i == 2 * c || i == 2 * c + 1) || *v == 0.0));
            let back = components::<A>(x);
            assert_eq!(back, comps);
        }
    }
    probe::<C64>();
    probe::<Quaternion>();
    probe::<Octonion>();
    probe::<Sedenion>();
    assert_eq!(algebra_capacity::<C64>(), 0);
    assert_eq!(algebra_capacity::<Quaternion>(), 1);
    assert_eq!(algebra_capacity::<Octonion>(), 2);
    assert_eq!(algebra_capacity::<Sedenion>(), 3);
}

#[test]
fn the_dual_algebra_operator_space_is_measured_per_doubling() {
    // ℍ ⊗ ℍ^op resolves the FULL real operator space of its qudit
    // (16 of 16): every 1-algebra-qubit gate is a sum of left×right
    // multiplications, and the embedded-ℂ action is component-linear.
    let h = dual_algebra_report::<Quaternion>().unwrap();
    assert_eq!(h.operator_space, 16);
    assert_eq!(h.sandwich_rank, 16);
    assert!(h.max_linear_residual < 1e-9, "{}", h.max_linear_residual);
    assert!(h.embedded_linear);

    // 𝕆: the twist breaks embedded component-linearity (measured, the
    // reason site gates route through the component path there) — the
    // sandwich span is measured against the 64-dim operator space.
    let o = dual_algebra_report::<Octonion>().unwrap();
    assert_eq!(o.operator_space, 64);
    assert!(!o.embedded_linear);
    assert_eq!(o.sandwich_rank, 64, "octonion sandwiches span End_R");
    assert!(o.max_linear_residual < 1e-9, "{}", o.max_linear_residual);

    // 𝕊 has zero divisors — and the sandwich span STILL measures full
    // (256 of 256, residual ~1e-15): zero divisors kill division, not
    // the multiplication algebra. Pinned so a regression is loud.
    let s = dual_algebra_report::<Sedenion>().unwrap();
    assert_eq!(s.operator_space, 256);
    assert!(!s.embedded_linear);
    assert_eq!(s.sandwich_rank, 256, "sedenion sandwiches span End_R too");
    assert!(s.max_linear_residual < 1e-9, "{}", s.max_linear_residual);

    // Where the residual vanishes the synthesis is OPERATIONAL: a
    // sandwich-applied gate equals its matrix on random qudits.
    fn verify_synthesis<A: Scalar>(matrix: &[C64]) {
        let (op, residual) = synthesize_sandwich::<A>(matrix).unwrap();
        assert!(residual < 1e-9, "residual {residual}");
        let cap = A::DIM / 2;
        let mut seed = 41u64;
        for _ in 0..6 {
            let comps: Vec<C64> = (0..cap)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    c64(
                        ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5,
                        ((seed >> 13) as f64 % 1024.0) / 1024.0 - 0.5,
                    )
                })
                .collect();
            let x: A = from_components(&comps);
            let via_algebra = components::<A>(op.apply(x));
            for r in 0..cap {
                let mut direct = c64(0.0, 0.0);
                for c in 0..cap {
                    direct += matrix[r * cap + c] * comps[c];
                }
                assert!(
                    (via_algebra[r] - direct).norm() < 1e-9,
                    "sandwich disagrees with its matrix at {r}"
                );
            }
        }
    }
    // Hadamard on the ℍ qudit's one algebra qubit.
    let f = std::f64::consts::FRAC_1_SQRT_2;
    verify_synthesis::<Quaternion>(&[c64(f, 0.0), c64(f, 0.0), c64(f, 0.0), c64(-f, 0.0)]);
    // CX on the 𝕆 qudit's two algebra qubits (control = bit 0).
    let mut cx = vec![c64(0.0, 0.0); 16];
    for (r, c) in [(0, 0), (2, 2), (1, 3), (3, 1)] {
        cx[r * 4 + c] = c64(1.0, 0.0);
    }
    verify_synthesis::<Octonion>(&cx);
}

#[test]
fn algebraic_registers_conform_over_the_full_registry() {
    // The register as a first-class backend: the complete gate registry
    // swept against dense — random parameters, orderings, widths,
    // registry-drawn random circuits, sampling and collapse — at an
    // ℍ split (native site path, measured) and an 𝕆 split (component
    // path everywhere, measured).
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("algebraic-h", |n| {
            Ok(Box::new(algebraic_over_sparse::<Quaternion>(n, 1)?))
        })
        .unwrap();
    sim.backends_mut()
        .register("algebraic-o", |n| {
            Ok(Box::new(algebraic_over_sparse::<Octonion>(n, 2)?))
        })
        .unwrap();
    for name in ["algebraic-h", "algebraic-o"] {
        let report = verify_backend(&sim, name, &ConformanceConfig::default()).unwrap();
        assert!(report.passed(), "{name}: {report}");
    }
    assert!(embedded_action_is_component_linear::<Quaternion>());
    assert!(!embedded_action_is_component_linear::<Octonion>());
}

#[test]
fn the_encoding_is_exact_across_the_boundary_and_certified() {
    // Bell pair straddling the site↔algebra boundary; the causal
    // Clifford family checked amplitude-by-amplitude against the exact
    // D[ω] ring at every split of the same logical width.
    let reg = GateRegistry::<C64>::standard();
    let n = 8;
    for (name, circuit) in [
        ("ranged-2", library::ranged_pairs(n, 2)),
        ("rainbow", library::rainbow(n)),
        ("brickwork", library::brickwork(n, 4, &[3])),
        ("ghz", library::ghz(n)),
    ] {
        let exact = ExactState::run(&circuit).unwrap();
        let bound = circuit.bind(&reg).unwrap();
        for algebra in [1usize, 2, 3] {
            let mut state = algebraic_over_sparse::<Sedenion>(n, algebra).unwrap();
            bound.run(&mut state).unwrap();
            for idx in 0..(1u64 << n) {
                let dev = (state.amplitude(idx) - exact.amplitude_c64(idx)).norm();
                assert!(dev < TOL, "{name} split k={algebra} at {idx}: {dev}");
            }
        }
    }

    // The boundary Bell pair, explicitly: site qubit 0 and algebra
    // qubit 0 of a (1 + 1) register.
    let mut c: Circuit = Circuit::new(2);
    c.h(0).cx(0, 1);
    let mut bell = algebraic_over_sparse::<Quaternion>(2, 1).unwrap();
    c.bind(&reg).unwrap().run(&mut bell).unwrap();
    let f = std::f64::consts::FRAC_1_SQRT_2;
    assert_close(bell.amplitude(0b00).re, f, TOL);
    assert_close(bell.amplitude(0b11).re, f, TOL);
    assert_close(bell.probability(0b01) + bell.probability(0b10), 0.0, TOL);
    assert_eq!(bell.site_support(), 2);
}

#[test]
fn measurement_and_feedback_work_on_algebra_qubits() {
    // Projection on an algebra qubit collapses the joint state exactly
    // like a site projection: measure one half of a boundary Bell pair
    // and the other half follows, deterministically per seed.
    let reg = GateRegistry::<C64>::standard();
    let mut c: Circuit = Circuit::new(4);
    c.h(0).cx(0, 3).h(1).cx(1, 2);
    let mut state = algebraic_over_sparse::<Octonion>(4, 2).unwrap();
    c.bind(&reg).unwrap().run(&mut state).unwrap();

    // P(algebra qubit 3 = 1) is exactly 1/2; project and renormalize.
    let p1: f64 = {
        let mut p = 0.0;
        state.for_each_nonzero(&mut |idx, amp| {
            if (idx >> 3) & 1 == 1 {
                p += amp.norm_sqr();
            }
        });
        p
    };
    assert_close(p1, 0.5, TOL);
    state.project(3, true, (1.0 / p1).sqrt());
    // Site qubit 0 must have collapsed with it.
    assert_close(state.probability(0b1001), 0.5, TOL);
    assert_close(state.probability(0b0000), 0.0, TOL);
    let mut total = 0.0;
    state.for_each_nonzero(&mut |_, amp| total += amp.norm_sqr());
    assert_close(total, 1.0, 1e-9);
}

#[test]
fn the_register_exceeds_flat_indexing_width() {
    // Every flat backend in the crate indexes basis states by u64 and
    // refuses beyond 63 qubits. The hierarchical register reaches 66
    // logical qubits — 63 sparse sites × a 3-qubit sedenion qudit —
    // with exact amplitudes addressed as (site, component) parts.
    assert!(SparseState::<C64>::new(66).is_err());
    assert!(DenseState::<C64>::new(66).is_err());

    let n = 66;
    let mut state = algebraic_over_sparse::<Sedenion>(n, 3).unwrap();
    assert_eq!(state.logical_qubits(), 66);
    let reg = GateRegistry::<C64>::standard();
    library::ghz(n).bind(&reg).unwrap().run(&mut state).unwrap();

    let f = std::f64::consts::FRAC_1_SQRT_2;
    assert_close(state.amplitude_parts(0, 0).re, f, TOL);
    let all_sites = (1u64 << 63) - 1;
    assert_close(state.amplitude_parts(all_sites, 0b111).re, f, TOL);
    assert_eq!(state.site_support(), 2);
    assert!(state.memory_bytes() < 4096, "{}", state.memory_bytes());

    // The full nonzero support of the 66-qubit state, via wide parts.
    let mut seen = Vec::new();
    state.for_each_nonzero_parts(&mut |site, comp, amp| {
        seen.push((site, comp, amp.re));
    });
    seen.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(seen.len(), 2);
    assert_eq!((seen[0].0, seen[0].1), (0, 0));
    assert_eq!((seen[1].0, seen[1].1), (all_sites, 0b111));
}

#[test]
fn site_structure_survives_the_algebra_sector() {
    // Superpose 4 site qubits AND all 3 algebra qubits: a flat sparse
    // register stores 2^7 = 128 entries; the hierarchical register
    // stores 16 site entries with the 8-component qudit riding inside
    // each — the site sector's structure is what the store scales
    // with.
    let total = 11;
    let reg = GateRegistry::<C64>::standard();
    let mut c: Circuit = Circuit::new(total);
    for q in 0..4 {
        c.h(q);
    }
    for q in 8..11 {
        c.h(q);
    }
    // Entangle the sectors so this is not a product state.
    c.cx(0, 8).cx(1, 9).cx(2, 10);
    let bound = c.bind(&reg).unwrap();

    let flat = Simulator::<C64>::new().run_on("sparse", &c).unwrap();
    assert_eq!(flat.nonzero_count(), 128);

    let mut hier = algebraic_over_sparse::<Sedenion>(total, 3).unwrap();
    bound.run(&mut hier).unwrap();
    assert_eq!(hier.site_support(), 16, "entries count sites");
    assert_eq!(hier.nonzero_count(), 128, "logical support is unchanged");
    assert!(max_amplitude_deviation(flat.as_ref(), &hier) < TOL);

    // Routing is measured, not assumed: with ℍ the site gates ran
    // natively, the boundary gate took the component path, and the
    // algebra-only gate executed as a dual-algebra sandwich.
    let mut routed = algebraic_over_sparse::<Quaternion>(5, 1).unwrap();
    let mut small: Circuit = Circuit::new(5);
    small.h(0).h(1).cx(0, 1).cx(3, 4).h(4);
    small.bind(&reg).unwrap().run(&mut routed).unwrap();
    let stats = routed.stats();
    assert_eq!(
        stats.native_site_gates, 3,
        "h(0), h(1), cx(0,1) are site-native"
    );
    assert_eq!(stats.component_gates, 1, "cx(3,4) straddles the boundary");
    assert_eq!(
        stats.sandwich_gates, 1,
        "h(4) runs as two-sided multiplication"
    );
}

#[test]
fn sandwich_native_execution_is_real_and_identical() {
    // Algebra-sector gates run as actual two-sided multiplications on
    // the stored scalars. Same circuit, native on vs off: identical
    // amplitudes, and the counters prove which path ran.
    let reg = GateRegistry::<C64>::standard();
    let mut c: Circuit = Circuit::new(5);
    c.h(0).cx(0, 1).cx(1, 3); // sites + boundary
    c.h(3).t(3).cx(3, 4).h(4).s(4).cx(4, 3); // algebra sector
    let bound = c.bind(&reg).unwrap();
    let dense = Simulator::<C64>::new().run(&c).unwrap();

    let mut native = algebraic_over_sparse::<Octonion>(5, 2).unwrap();
    bound.run(&mut native).unwrap();
    assert!(
        native.stats().sandwich_gates >= 6,
        "the algebra-sector gates execute as sandwiches: {:?}",
        native.stats()
    );
    assert!(max_amplitude_deviation(dense.as_ref(), &native) < TOL);

    let mut component = algebraic_over_sparse::<Octonion>(5, 2).unwrap();
    component.set_sandwich_native(false);
    bound.run(&mut component).unwrap();
    assert_eq!(component.stats().sandwich_gates, 0);
    assert!(max_amplitude_deviation(dense.as_ref(), &component) < TOL);
    assert!(max_amplitude_deviation(&native, &component) < TOL);

    // The synthesis is cached: replaying the same gates re-counts
    // sandwiches without re-solving (observable as cheap, identical
    // behavior — pinned by the counter doubling).
    let before = native.stats().sandwich_gates;
    bound.run(&mut native).unwrap();
    assert_eq!(native.stats().sandwich_gates, 2 * before);
}

#[test]
fn direct_sum_blocks_bound_the_synthesis() {
    // DirectSum<H, H>: two independent 1-qubit blocks in one scalar.
    // The blocks multiply independently, so the dual-algebra span is
    // EXACTLY the block-diagonal operator algebra: half the operator
    // space, with cross-block gates measurably outside it.
    type HH = DirectSum<Quaternion, Quaternion>;
    assert_eq!(algebra_capacity::<HH>(), 2);
    let report = dual_algebra_report::<HH>().unwrap();
    assert_eq!(report.operator_space, 64);
    assert_eq!(
        report.sandwich_rank, 32,
        "blockwise multiplication spans exactly the block-diagonals"
    );
    assert!(
        report.max_linear_residual > 0.1,
        "cross-block gates are outside the span: {}",
        report.max_linear_residual
    );
    assert!(
        report.embedded_linear,
        "diagonal embedding scales both blocks"
    );

    // Block-diagonal gates synthesize exactly: CZ over (bit 0 = within
    // block, bit 1 = which block) touches no cross-block entry.
    let mut cz = vec![c64(0.0, 0.0); 16];
    for (i, phase) in [1.0, 1.0, 1.0, -1.0].iter().enumerate() {
        cz[i * 4 + i] = c64(*phase, 0.0);
    }
    let (_, cz_residual) = synthesize_sandwich::<HH>(&cz).unwrap();
    assert!(cz_residual < 1e-10, "{cz_residual}");

    // SWAP(bit0, bit1) moves weight between the blocks: unreachable.
    let mut swap = vec![c64(0.0, 0.0); 16];
    for (r, c) in [(0, 0), (1, 2), (2, 1), (3, 3)] {
        swap[r * 4 + c] = c64(1.0, 0.0);
    }
    let (_, swap_residual) = synthesize_sandwich::<HH>(&swap).unwrap();
    assert!(
        swap_residual > 0.1,
        "the block boundary is measurable: {swap_residual}"
    );

    // The boundary costs routing, never correctness: the register over
    // H ⊕ H conforms over the full registry (cross-block gates run
    // through the component path).
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("algebraic-hh", |n| {
            Ok(Box::new(algebraic_over_sparse::<HH>(n, 2)?))
        })
        .unwrap();
    let conf = verify_backend(&sim, "algebraic-hh", &ConformanceConfig::default()).unwrap();
    assert!(conf.passed(), "{conf}");
}

#[test]
#[ignore = "the 32-dim basis SVD takes ~17 s; run with --ignored to reproduce"]
fn the_fourth_doubling_keeps_the_span_full() {
    // CD⟨S⟩ — trigintaduonions, 32-dim, a 4-qubit qudit per scalar:
    // the sandwich span measures FULL even here (1024 of 1024,
    // residual ~1e-15).
    let t = dual_algebra_report::<Trigintaduonion>().unwrap();
    assert_eq!(t.operator_space, 1024);
    assert_eq!(t.sandwich_rank, 1024);
    assert!(t.max_linear_residual < 1e-9, "{}", t.max_linear_residual);
    assert!(!t.embedded_linear);
}

#[test]
fn wide_sampling_beyond_the_ceiling() {
    // 67 logical qubits — 63 sparse sites × a 4-qubit trigintaduonion
    // qudit — sampled by Born weight over (site, component) parts.
    assert_eq!(algebra_capacity::<Trigintaduonion>(), 4);
    let n = 67;
    let mut state = algebraic_over_sparse::<Trigintaduonion>(n, 4).unwrap();
    assert_eq!(state.logical_qubits(), 67);
    let reg = GateRegistry::<C64>::standard();
    library::ghz(n).bind(&reg).unwrap().run(&mut state).unwrap();

    let f = std::f64::consts::FRAC_1_SQRT_2;
    let all_sites = (1u64 << 63) - 1;
    assert_close(state.amplitude_parts(0, 0).re, f, TOL);
    assert_close(state.amplitude_parts(all_sites, 0b1111).re, f, TOL);
    assert_eq!(state.site_support(), 2);
    assert_close(state.probability_parts(0, 0), 0.5, TOL);

    let counts = state.sample_parts(2000, &mut Prng::new(11)).unwrap();
    assert_eq!(counts.len(), 2, "GHZ samples exactly two outcomes");
    let zeros = counts.get(&(0, 0)).copied().unwrap_or(0);
    let ones = counts.get(&(all_sites, 0b1111)).copied().unwrap_or(0);
    assert_eq!(zeros + ones, 2000);
    assert!(
        (700..=1300).contains(&zeros),
        "fair coin within tolerance: {zeros}"
    );
}

#[test]
fn the_workload_harness_prices_the_hierarchical_register() {
    // The causal workload family through the benchmark harness with
    // the hierarchical registers alongside the flat ones — every run
    // verified against dense in-run.
    let n = 10;
    let mut sim: Simulator = Simulator::new();
    sim.backends_mut()
        .register("algebraic-h", |n| {
            Ok(Box::new(algebraic_over_sparse::<Quaternion>(n, 1)?))
        })
        .unwrap();
    sim.backends_mut()
        .register("algebraic-o", |n| {
            Ok(Box::new(algebraic_over_sparse::<Octonion>(n, 2)?))
        })
        .unwrap();
    let workloads = [
        Workload::from_circuit("ghz", library::ghz(n)),
        Workload::from_circuit("ranged-5", library::ranged_pairs(n, 5)),
        Workload::from_circuit("qft", library::qft(n)),
    ];
    let backends = ["dense", "sparse", "algebraic-h", "algebraic-o"];
    let report = compare_backends(&sim, &workloads, &backends, &BenchConfig::default()).unwrap();
    assert_eq!(report.records.len(), workloads.len() * backends.len());
    assert!(report.max_deviation() < 1e-9, "harness-verified physics");
    // GHZ stays sparse in the site sector: hierarchical memory is far
    // below dense.
    let mem = |b: &str| report.record("ghz", b).unwrap().memory_bytes;
    assert!(mem("algebraic-h") < mem("dense") / 8);
}
