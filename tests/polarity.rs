//! Systems of `n` twisted polarities, measured: the structure theorem
//! that ties the twist rank to the locally-storable sector, the Pauli
//! group as the maximally twisted example (and its stabilizer groups as
//! the isotropic one), `Polarity<N>` as an amplitude algebra, and the
//! sharp obstruction — two orthogonal states with identical pairwise
//! data.

#![allow(clippy::assertions_on_constants)] // algebra-flag pins are intentional

mod common;

use quantsim::conformance::{verify_backend, ConformanceConfig};
use quantsim::polarity::*;
use quantsim::prelude::*;
use quantsim::scalar::MAX_POLARITIES;

// ── the structure theorem ────────────────────────────────────────────

#[test]
fn the_algebra_factors_into_a_local_part_and_a_matrix_block() {
    for generators in [4usize, 6, 8] {
        for pairs in 0..=generators / 2 {
            let system = PolaritySystem::partial(generators, pairs).unwrap();
            let s = system.structure();
            assert_eq!(s.twist_rank, 2 * pairs);
            // The identity that carries the whole story: what is
            // storable as independent sign bits times what is not, is
            // the whole algebra.
            assert_eq!(s.isotropic_size * s.matrix_block, s.dimension);
            assert_eq!(s.local_bits, generators - pairs);
            assert_eq!(s.matrix_block, 1 << pairs);
        }
    }
}

#[test]
fn the_structure_is_brute_force_verified_not_quoted() {
    for system in [
        PolaritySystem::untwisted(5).unwrap(),
        PolaritySystem::partial(6, 1).unwrap(),
        PolaritySystem::partial(6, 3).unwrap(),
        PolaritySystem::fully_twisted(4).unwrap(),
        PolaritySystem::fully_twisted(5).unwrap(),
        PolaritySystem::pauli(3).unwrap(),
    ] {
        let s = system.structure();
        // Centre, counted by testing every monomial against every
        // generator.
        assert_eq!(system.center().len(), s.center_dimension);
        // Maximal commuting set, built by exhaustive greedy search.
        let isotropic = system.maximal_isotropic();
        assert_eq!(isotropic.len(), s.isotropic_size);
        // The form is bilinear, so the commuting set is a subgroup:
        // closed under XOR of generator masks.
        for &a in &isotropic {
            for &b in &isotropic {
                assert!(system.commutes(a, b));
                assert!(isotropic.contains(&(a ^ b)), "{a} ^ {b} left the set");
            }
        }
    }
}

#[test]
fn the_untwisted_system_is_entirely_local_and_the_full_twist_entirely_not() {
    let flat = PolaritySystem::untwisted(6).unwrap().structure();
    assert_eq!(flat.twist_rank, 0);
    assert_eq!(flat.matrix_block, 1); // nothing to pay
    assert_eq!(flat.local_bits, 6);
    assert_eq!(flat.center_dimension, flat.dimension); // all central

    let full = PolaritySystem::fully_twisted(6).unwrap().structure();
    assert_eq!(full.twist_rank, 6);
    assert_eq!(full.matrix_block, 8);
    assert_eq!(full.local_bits, 3);
    // Rank is even, always: an alternating form has no odd rank.
    for n in 2..=8 {
        assert_eq!(
            PolaritySystem::fully_twisted(n).unwrap().twist_rank() % 2,
            0
        );
    }
}

#[test]
fn the_pauli_group_is_the_maximally_twisted_system_and_its_isotropic_part_is_stabilizers() {
    for n in 1..=5 {
        let system = PolaritySystem::pauli(n).unwrap();
        let s = system.structure();
        assert_eq!(s.polarities, 2 * n);
        // Maximally twisted: rank is the full generator count, so the
        // centre is trivial.
        assert_eq!(s.twist_rank, 2 * n);
        assert_eq!(s.center_dimension, 1);
        // ...and a maximal commuting subgroup has exactly 2^n elements
        // — a stabilizer group, described by n independent sign bits.
        assert_eq!(s.local_bits, n);
        assert_eq!(s.isotropic_size, 1 << n);
        assert_eq!(system.maximal_isotropic().len(), 1 << n);
        // The price of leaving it is the same 2^n.
        assert_eq!(s.matrix_block, 1 << n);
    }
}

#[test]
fn products_are_associative_and_the_twist_is_what_it_says() {
    let system = PolaritySystem::pauli(2).unwrap();
    let dim = system.dimension() as u64;
    for a in 0..dim {
        for b in 0..dim {
            let (ab, sab) = system.product(a, b);
            let (ba, sba) = system.product(b, a);
            assert_eq!(ab, ba, "monomials multiply to the same subset");
            // Commuting is exactly agreeing on sign.
            assert_eq!(system.commutes(a, b), sab == sba);
            for c in 0..dim {
                let (left, sl) = system.product(ab, c);
                let (bc, sbc) = system.product(b, c);
                let (right, sr) = system.product(a, bc);
                assert_eq!(left, right);
                common::assert_close(sab * sl, sbc * sr, 1e-15);
            }
        }
    }
}

#[test]
fn malformed_systems_are_refused_by_reason() {
    assert!(PolaritySystem::new(0, &[], &[]).is_err());
    assert!(PolaritySystem::new(3, &[(0, 5)], &[]).is_err());
    assert!(PolaritySystem::new(3, &[(1, 1)], &[]).is_err());
    let err = PolaritySystem::partial(4, 3).unwrap_err();
    assert!(format!("{err}").contains("need 6 generators"), "{err}");
}

// ── the amplitude algebra ────────────────────────────────────────────

#[test]
fn polarity_two_is_the_split_quaternions() {
    // 1↔1, i↔j₀j₁, j↔j₀, k↔−j₁, checked entrywise on the whole basis.
    let to_polarity = |q: SplitQuaternion| {
        let c = q.coeffs(); // [1, i, j, k]
        Polarity::<2>::from_coeffs(&[c[0], c[2], -c[3], c[1]]) // [1, j₀, j₁, j₀j₁]
    };
    let mut worst_product = 0.0f64;
    let mut worst_born = 0.0f64;
    for a in 0..4 {
        for b in 0..4 {
            let (x, y) = (SplitQuaternion::basis(a), SplitQuaternion::basis(b));
            let transported = to_polarity(x * y);
            let multiplied = to_polarity(x) * to_polarity(y);
            worst_product = worst_product.max((transported - multiplied).abs_sqr().sqrt());
            worst_born = worst_born.max((x.born_weight() - to_polarity(x).born_weight()).abs());
        }
    }
    assert_eq!(worst_product, 0.0);
    assert_eq!(worst_born, 0.0);
    assert_eq!(Polarity::<2>::DIM, SplitQuaternion::DIM);
}

#[test]
fn two_polarities_are_needed_before_complex_numbers_embed() {
    // One polarity is the split-complex numbers: no i, real gates only.
    assert!(Polarity::<1>::COMMUTATIVE);
    assert_eq!(
        GateRegistry::<Polarity<1>>::standard().names(),
        GateRegistry::<f64>::standard().names()
    );
    // Two or more: j₀j₁ squares to −1, so the full library exists.
    let full = GateRegistry::<C64>::standard().names();
    assert_eq!(GateRegistry::<Polarity<2>>::standard().names(), full);
    assert_eq!(GateRegistry::<Polarity<3>>::standard().names(), full);
    assert_eq!(GateRegistry::<Polarity<4>>::standard().names(), full);
    assert!(!Polarity::<3>::COMMUTATIVE);
    assert!(!Polarity::<3>::DIVISION);
}

#[test]
fn the_born_form_is_inclusion_minus_exclusion_by_degree_parity() {
    let mut c = vec![0.0; 8];
    c[0] = 1.0; // degree 0
    c[0b101] = 2.0; // degree 2
    c[0b010] = 3.0; // degree 1
    c[0b111] = 4.0; // degree 3
    let q = Polarity::<3>::from_coeffs(&c);
    common::assert_close(q.inclusion_weight(), 1.0 + 4.0, 1e-15);
    common::assert_close(q.exclusion_weight(), 9.0 + 16.0, 1e-15);
    common::assert_close(q.born_weight(), 5.0 - 25.0, 1e-15);
    common::assert_close(q.abs_sqr(), 30.0, 1e-15);
    assert_eq!(q.degree_profile(), vec![1.0, 9.0, 4.0, 16.0]);
}

#[test]
fn the_backends_conform_over_a_three_polarity_amplitude() {
    let sim = Simulator::<Polarity<3>>::new();
    let cfg = ConformanceConfig::default();
    for backend in ["sparse", "adaptive", "factored"] {
        let report = verify_backend(&sim, backend, &cfg).unwrap();
        assert!(report.passed(), "{backend}: {:?}", report.failures);
        assert_eq!(report.algebra, "Pol3");
    }
    // And a canonical state comes out right.
    let state = sim.run(&library::ghz(4)).unwrap();
    common::assert_close(state.probability(0), 0.5, 1e-12);
    common::assert_close(state.probability(15), 0.5, 1e-12);
    common::assert_close(state.total_weight(), 1.0, 1e-12);
}

#[test]
fn the_storage_padding_is_stated_rather_than_hidden() {
    // Const-generic arrays cannot be sized 2^N on stable, so every
    // Polarity<N> stores 2^MAX_POLARITIES slots. The algebraic
    // dimension is honest; the footprint is padded, and the exactly
    // sized two-polarity case is SplitQuaternion.
    assert_eq!(Polarity::<1>::DIM, 2);
    assert_eq!(Polarity::<4>::DIM, 1 << MAX_POLARITIES);
    assert_eq!(
        std::mem::size_of::<Polarity<1>>(),
        std::mem::size_of::<Polarity<4>>()
    );
    assert!(std::mem::size_of::<SplitQuaternion>() < std::mem::size_of::<Polarity<2>>());
}

// ── the obstruction ──────────────────────────────────────────────────

#[test]
fn pairwise_data_is_complete_for_two_qubits_and_blind_from_three_up() {
    // Two qubits: ⟨XX⟩ separates the Bell signs, so pairwise data is
    // enough and the report says so.
    let two = ghz_sign_obstruction(2).unwrap();
    assert!(!two.pairwise_blind);
    common::assert_close(two.signature_deviation, 2.0, 1e-12);

    // Three and up: the two states are orthogonal, and every one- and
    // two-body Pauli expectation agrees exactly. Nothing pairwise can
    // tell them apart.
    for n in 3..=8 {
        let o = ghz_sign_obstruction(n).unwrap();
        assert!(o.pairwise_blind, "n = {n}");
        assert_eq!(o.signature_deviation, 0.0, "n = {n}");
        assert!(o.overlap < 1e-15, "n = {n}: overlap {}", o.overlap);
        // What does separate them is the n-body correlator.
        common::assert_close(o.global_correlator.0, 1.0, 1e-12);
        common::assert_close(o.global_correlator.1, -1.0, 1e-12);
    }
}

#[test]
fn the_blindness_is_structural_not_a_shortage_of_numbers() {
    // At three qubits the pairwise signature holds MORE real numbers
    // than the state vector does — 36 against 16 — and is still blind.
    // The failure is in what pairwise data can express, not how much
    // of it there is.
    let o = ghz_sign_obstruction(3).unwrap();
    assert_eq!(o.pairwise_values, 3 * 3 + 9 * 3);
    assert_eq!(o.state_values, 16);
    assert!(o.pairwise_values > o.state_values);
    assert!(o.pairwise_blind);

    // The signature really is the complete two-body content: identical
    // to what a direct measurement of both states returns.
    let top = (1u64 << 3) - 1;
    let amp = std::f64::consts::FRAC_1_SQRT_2;
    let mut plus = DenseState::<C64>::new(3).unwrap();
    plus.load(&[(0, C64::new(amp, 0.0)), (top, C64::new(amp, 0.0))])
        .unwrap();
    let mut minus = DenseState::<C64>::new(3).unwrap();
    minus
        .load(&[(0, C64::new(amp, 0.0)), (top, C64::new(-amp, 0.0))])
        .unwrap();
    let sig_plus = pairwise_signature(&plus).unwrap();
    let sig_minus = pairwise_signature(&minus).unwrap();
    assert_eq!(sig_plus.max_deviation(&sig_minus), 0.0);
    assert_eq!(sig_plus.values(), 36);
}

#[test]
fn the_efficient_description_is_quadratic_in_size_but_not_pairwise_in_structure() {
    // Both GHZ signs are stabilizer states, so this is not pairwise
    // data failing on something exotic — it fails inside the sector
    // that IS efficiently describable. The reason is the weight-n
    // generator: an O(n²) description that is not a pairwise one.
    for n in [3usize, 4, 8] {
        let weights = stabilizer_weight_profile(n).unwrap();
        assert_eq!(weights.len(), n);
        assert_eq!(weights[0], n, "GHZ-{n} needs a weight-{n} generator");
        assert!(weights[1..].iter().all(|&w| w == 2));
    }
    assert!(stabilizer_weight_profile(1).is_err());
    assert!(ghz_sign_obstruction(1).is_err());
}

#[test]
fn a_product_state_is_determined_by_its_pairwise_data() {
    // The contrast that makes the obstruction meaningful: where
    // pairwise data does suffice, it agrees exactly.
    let mut c: Circuit = Circuit::new(4);
    c.ry(0, 0.4).ry(1, 1.1).ry(2, 2.0).ry(3, 0.7);
    let a = Simulator::new().run(&c).unwrap();
    let b = Simulator::new().run(&c).unwrap();
    let sig = pairwise_signature(a.as_ref()).unwrap();
    assert_eq!(
        sig.max_deviation(&pairwise_signature(b.as_ref()).unwrap()),
        0.0
    );

    // ...and a genuinely different product state separates.
    let mut d: Circuit = Circuit::new(4);
    d.ry(0, 0.4).ry(1, 1.1).ry(2, 2.0).ry(3, 1.9);
    let e = Simulator::new().run(&d).unwrap();
    assert!(sig.max_deviation(&pairwise_signature(e.as_ref()).unwrap()) > 0.1);
}
