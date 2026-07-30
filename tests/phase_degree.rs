//! The polynomial degree of a phase function, measured — and the
//! resolution of the √2 that `e8_across` and `selfhost` each ran into from
//! a different direction.
//!
//! Three degrees turn out to be in play, and keeping them apart is the
//! whole point:
//!
//! 1. the **multilinear** degree of a diagonal's phase polynomial in the
//!    register bits, which `selfhost` reduces to one;
//! 2. the **phase degree on the bit group**, which decides whether the
//!    diagonal is a character of that group;
//! 3. the **phase degree on the E8 residue group**, which is what the
//!    constellation's native operators are characters of.

use quantsim::e8::constellation::{self, residue_add};
use quantsim::phase::{bit_group_elements, phase_degree, PhaseGroup};
use quantsim::prelude::*;
use quantsim::selfhost::{Diagonal, SelfHostedStack};

fn xor(a: u64, b: u64) -> u64 {
    a ^ b
}

fn diag(bits: u32, terms: &[(u64, i64)]) -> Diagonal {
    let mut d = Diagonal::new(bits).expect("bits in range");
    for &(mask, coeff) in terms {
        d.term(mask, coeff);
    }
    d
}

/// Measure a diagonal's phase degree on the bit group of `n` qubits.
fn bit_degree(d: &Diagonal, n: usize, max_order: u32) -> quantsim::phase::PhaseDegree {
    let elements = bit_group_elements(n);
    let add: &dyn Fn(u64, u64) -> u64 = &xor;
    let group = PhaseGroup {
        elements: &elements,
        complete: true,
        add,
    };
    let phase = |index: u64| d.phase(index);
    phase_degree(&group, &phase, max_order, 40_000, 1e-9)
}

#[test]
fn degree_one_is_exactly_being_a_character() {
    // A character satisfies f(p+q) = f(p)f(q), which says the first
    // derivative does not depend on the base point — so the second
    // derivative vanishes and the degree is one. The instrument must agree
    // with that definition rather than approximate it.
    let z = diag(1, &[(0b1, 1)]);
    let measured = bit_degree(&z, 3, 5);
    assert_eq!(measured.certified_degree(), Some(1));
    assert!(measured.is_character());

    // And a genuine character composed of several single-qubit factors is
    // still degree one.
    let many = diag(1, &[(0b1, 1), (0b10, 1), (0b100, 1)]);
    let measured = bit_degree(&many, 3, 5);
    assert_eq!(measured.certified_degree(), Some(1));

    // Whereas a cz is not a character, and the witness is order two.
    let cz = diag(1, &[(0b11, 1)]);
    let measured = bit_degree(&cz, 3, 5);
    assert!(!measured.is_character());
    assert_eq!(measured.certified_degree(), Some(2));
    assert!(measured.exceeds(1), "order two must hold a witness");
}

#[test]
fn the_bit_group_degree_is_multilinear_degree_plus_log_denominator_minus_one() {
    // Measured over eleven diagonals spanning both dials independently:
    // the multilinear degree (how many qubits a monomial couples) and the
    // phase denominator (how fine the roots of unity are).
    let cases: Vec<(&str, Diagonal, usize)> = vec![
        ("z", diag(1, &[(0b1, 1)]), 3),
        ("s", diag(2, &[(0b1, 1)]), 3),
        ("t", diag(3, &[(0b1, 1)]), 3),
        ("t-half", diag(4, &[(0b1, 1)]), 3),
        ("cz", diag(1, &[(0b11, 1)]), 3),
        ("cs", diag(2, &[(0b11, 1)]), 3),
        ("ct", diag(3, &[(0b11, 1)]), 3),
        ("ccz", diag(1, &[(0b111, 1)]), 3),
        ("cccz", diag(1, &[(0b1111, 1)]), 4),
        ("z-times-z", diag(1, &[(0b1, 1), (0b10, 1)]), 3),
        ("t-times-t", diag(3, &[(0b1, 1), (0b10, 1)]), 3),
    ];
    for (name, d, n) in &cases {
        let predicted = d.degree() + d.bits() - 1;
        let measured = bit_degree(d, *n, 7);
        assert_eq!(
            measured.degree(),
            Some(predicted),
            "{name}: multilinear {} + log2(denominator) {} - 1 = {predicted}, measured {:?} \
             (residuals {:?})",
            d.degree(),
            d.bits(),
            measured.degree(),
            measured.order_residuals
        );
    }
    // And the law's two consequences, stated as the tests they are:
    // a diagonal is a character iff BOTH dials are at their minimum.
    for (name, d, n) in &cases {
        let is_character = bit_degree(d, *n, 7).is_character();
        assert_eq!(
            is_character,
            d.degree() == 1 && d.bits() == 1,
            "{name}: being a character must be exactly multilinear-degree-one \
             AND plus-or-minus-one valued"
        );
    }
}

#[test]
fn the_degree_ladder_reproduces_the_clifford_hierarchy_for_diagonals() {
    // Rediscovered from measurement, not asserted from theory: the degree
    // sorts the diagonal gates into the levels they are known to occupy.
    let level = |d: &Diagonal, n: usize| bit_degree(d, n, 7).degree().expect("a degree");
    // Level one: Pauli-Z-like, the ±1 characters.
    assert_eq!(level(&diag(1, &[(0b1, 1)]), 3), 1);
    // Level two: Clifford.
    assert_eq!(level(&diag(2, &[(0b1, 1)]), 3), 2, "s");
    assert_eq!(level(&diag(1, &[(0b11, 1)]), 3), 2, "cz");
    // Level three: the first non-Clifford diagonals.
    assert_eq!(level(&diag(3, &[(0b1, 1)]), 3), 3, "t");
    assert_eq!(level(&diag(2, &[(0b11, 1)]), 3), 3, "cs");
    assert_eq!(level(&diag(1, &[(0b111, 1)]), 3), 3, "ccz");
    // Level four.
    assert_eq!(level(&diag(3, &[(0b11, 1)]), 4), 4, "ct");
    assert_eq!(level(&diag(1, &[(0b1111, 1)]), 4), 4, "cccz");
}

#[test]
fn one_e8_volume_is_the_bit_group_and_two_are_not() {
    // E8/2E8 ≅ F₂⁸ and class_of is linear, so at one copy the residue
    // group's addition IS bitwise XOR — checked over all 256 × 256 pairs,
    // not sampled.
    let mut mismatches = 0usize;
    for a in 0..256u64 {
        for b in 0..256u64 {
            if residue_add(1, a, b) != a ^ b {
                mismatches += 1;
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "one E8 volume must be exactly the bit group under XOR"
    );

    // Two copies is a different group: the division that extracts higher
    // digits carries, so the tower is not (F₂⁸)².
    let mut differing = 0usize;
    for a in 0..512u64 {
        for b in 0..512u64 {
            if residue_add(2, a, b) != a ^ b {
                differing += 1;
            }
        }
    }
    assert!(
        differing > 100_000,
        "two copies must differ from XOR on most pairs, got {differing} of {}",
        512 * 512
    );
}

#[test]
fn the_native_modulation_is_a_character_of_the_residue_group_and_not_of_the_bits() {
    // This is the crux. The constellation's native operators and the qubit
    // path are characters of DIFFERENT groups, and each looks
    // high-degree from the other's point of view.
    let levels = 2usize;
    let elements: Vec<u64> = (0..1u64 << (8 * levels)).collect();
    let add_residue: &dyn Fn(u64, u64) -> u64 = &|a, b| residue_add(levels, a, b);
    let add_xor: &dyn Fn(u64, u64) -> u64 = &xor;

    for q in [
        constellation::basis(0),
        constellation::basis(1),
        constellation::representative(1),
    ] {
        let phase = |bits: u64| constellation::modulation_phase(levels, &q, bits);

        let residue_group = PhaseGroup {
            elements: &elements,
            complete: true,
            add: add_residue,
        };
        let on_residue = phase_degree(&residue_group, &phase, 6, 40_000, 1e-9);
        assert_eq!(
            on_residue.degree(),
            Some(1),
            "a modulation is a character of the residue group by construction: \
             residuals {:?}",
            on_residue.order_residuals
        );

        let bit_group = PhaseGroup {
            elements: &elements,
            complete: true,
            add: add_xor,
        };
        let on_bits = phase_degree(&bit_group, &phase, 6, 40_000, 1e-9);
        assert!(
            on_bits.exceeds(3),
            "and is nowhere near a character of the bits: degree {:?}, residuals {:?}",
            on_bits.degree(),
            on_bits.order_residuals
        );
    }
}

#[test]
fn the_t_phase_has_degree_three_on_both_groups_and_its_order_two_residual_is_the_sqrt_two() {
    // The t phase depends only on qubit 0, and digit 0 of a residue sum is
    // the XOR of the operands' digit 0 because class_of is linear — so t
    // sees the same group structure either way, and measures degree 3 on
    // both.
    let omega = std::f64::consts::FRAC_1_SQRT_2;
    let t_phase = |bits: u64| {
        if bits & 1 == 1 {
            C64::new(omega, omega)
        } else {
            C64::new(1.0, 0.0)
        }
    };
    for levels in [1usize, 2] {
        let elements: Vec<u64> = (0..1u64 << (8 * levels)).collect();
        let add_residue: &dyn Fn(u64, u64) -> u64 = &|a, b| residue_add(levels, a, b);
        let add_xor: &dyn Fn(u64, u64) -> u64 = &xor;
        for add in [add_residue, add_xor] {
            let group = PhaseGroup {
                elements: &elements,
                complete: true,
                add,
            };
            let measured = phase_degree(&group, &t_phase, 6, 40_000, 1e-9);
            assert_eq!(measured.degree(), Some(3), "levels={levels}");
            // The second-order residual IS the number both other modules
            // reported: |i − 1| = √2. The obstruction they each hit is the
            // failure of t at the character order, and this is it.
            assert!(
                (measured.order_residuals[1] - std::f64::consts::SQRT_2).abs() < 1e-9,
                "levels={levels}: expected √2 at order two, got {}",
                measured.order_residuals[1]
            );
            // The first-order residual is |e^{iπ/4} − 1| = 2 sin(π/8).
            let expected = 2.0 * (std::f64::consts::PI / 8.0).sin();
            assert!((measured.order_residuals[0] - expected).abs() < 1e-9);
        }
    }
}

#[test]
fn linearizing_cannot_lower_the_phase_degree_below_the_denominator() {
    // Why `selfhost` could turn a ccz into a character but could do nothing
    // for a t. The stack reduces the MULTILINEAR term of the degree; the
    // log-denominator term is untouched, so a diagonal whose degree comes
    // entirely from its denominator is beyond it — however many layers.
    let sim = Simulator::<C64>::new();

    // ccz: degree 3 = multilinear 3 + 1 − 1. Linearizing drives the
    // multilinear term to one, and the phase degree follows to one.
    let ccz = diag(1, &[(0b111, 1)]);
    assert_eq!(bit_degree(&ccz, 3, 7).degree(), Some(3));
    let stack = SelfHostedStack::plan(3, &ccz).expect("plan");
    let linear = stack.linearize(&ccz).expect("linearize");
    assert_eq!(linear.degree(), 1, "multilinear degree reduced");
    let after = bit_degree(&linear, stack.width().min(16), 7);
    assert_eq!(
        after.degree(),
        Some(1),
        "so the linearized ccz is a character: residuals {:?}",
        after.order_residuals
    );
    assert!(after.is_character());

    // t: degree 3 = multilinear 1 + 3 − 1. There is no multilinear term to
    // reduce, the stack correctly does nothing, and the degree is unmoved.
    let t = diag(3, &[(0b1, 1)]);
    assert_eq!(bit_degree(&t, 3, 7).degree(), Some(3));
    let stack = SelfHostedStack::plan(3, &t).expect("plan");
    assert_eq!(stack.depth(), 0, "nothing to linearize");
    let linear = stack.linearize(&t).expect("linearize");
    assert_eq!(linear, t, "and the diagonal comes back unchanged");
    assert_eq!(
        bit_degree(&linear, 3, 7).degree(),
        Some(3),
        "so a t is never a character, at any depth"
    );
    // The floor, stated: a diagonal over 2^b-th roots has phase degree at
    // least b, because linearizing can only reach multilinear degree one.
    for bits in 1..=4u32 {
        let d = diag(bits, &[(0b1, 1)]);
        let floor = bits;
        assert_eq!(bit_degree(&d, 3, 7).degree(), Some(floor));
    }
    let _ = sim;
}

#[test]
fn a_sampled_sweep_certifies_nothing_but_still_witnesses() {
    // The asymmetry the API exists to keep visible.
    let ccz = diag(1, &[(0b111, 1)]);
    let elements = bit_group_elements(8);
    let add: &dyn Fn(u64, u64) -> u64 = &xor;
    let group = PhaseGroup {
        elements: &elements,
        complete: true,
        add,
    };
    let phase = |index: u64| ccz.phase(index);
    // 256 elements: order 1 fits the exhaustive budget, order 2 does not.
    let measured = phase_degree(&group, &phase, 6, 5_000, 1e-9);
    assert!(
        !measured.exhaustive,
        "256^3 tuples exceed the budget, so the sweep sampled"
    );
    assert_eq!(
        measured.certified_degree(),
        None,
        "a sampled sweep must not certify a degree — under-sampling can only \
         miss a violation"
    );
    assert_eq!(measured.degree(), Some(3), "but it still reports the bound");
    assert!(
        measured.exceeds(2),
        "and a non-vanishing derivative is a witness whether sampled or not"
    );
}

#[test]
fn a_constant_phase_has_degree_zero_and_the_sweep_stops_early() {
    let identity = Diagonal::new(1).expect("bits");
    let measured = bit_degree(&identity, 3, 6);
    assert_eq!(measured.certified_degree(), Some(0));
    assert_eq!(
        measured.order_residuals.len(),
        1,
        "the sweep stops at the first vanishing order: higher derivatives of \
         a vanishing one vanish too"
    );
    assert!(
        !measured.is_character(),
        "degree zero is constant, not degree one"
    );
    assert!(!measured.exceeds(0));
}
