//! The polynomial degree of a phase function, and the resolution of the √2
//! that `e8_across` and `selfhosted_stack` each hit from a different side.
//!
//! Prints the measured degree of every diagonal against the conjectured law
//! `multilinear + log₂(denominator) − 1`, the Clifford hierarchy read off
//! the degrees, and the two groups the E8 constellation is caught between.
//!
//! Run with `cargo run --release --example phase_degree`.

use quantsim::e8::constellation::{self, residue_add};
use quantsim::phase::{bit_group_elements, compare_degrees, phase_degree, PhaseGroup};
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

fn bits_degree(d: &Diagonal, n: usize) -> quantsim::phase::PhaseDegree {
    let elements = bit_group_elements(n);
    let add: &dyn Fn(u64, u64) -> u64 = &xor;
    let group = PhaseGroup {
        elements: &elements,
        complete: true,
        add,
    };
    let phase = |index: u64| d.phase(index);
    phase_degree(&group, &phase, 7, 40_000, 1e-9)
}

fn main() -> Result<()> {
    println!("== A phase function's degree, on the bit group ==\n");
    println!(
        "  {:<10} {:>6} {:>7} {:>10} {:>9} {:>6} {:>7}  order residuals",
        "gate", "mldeg", "log2 N", "predicted", "measured", "char?", "certif"
    );
    let cases: Vec<(&str, Diagonal, usize)> = vec![
        ("z", diag(1, &[(0b1, 1)]), 3),
        ("s", diag(2, &[(0b1, 1)]), 3),
        ("t", diag(3, &[(0b1, 1)]), 3),
        ("t^(1/2)", diag(4, &[(0b1, 1)]), 3),
        ("cz", diag(1, &[(0b11, 1)]), 3),
        ("cs", diag(2, &[(0b11, 1)]), 3),
        ("ct", diag(3, &[(0b11, 1)]), 4),
        ("ccz", diag(1, &[(0b111, 1)]), 3),
        ("cccz", diag(1, &[(0b1111, 1)]), 4),
    ];
    let mut all_match = true;
    for (name, d, n) in &cases {
        let measured = bits_degree(d, *n);
        let cmp = compare_degrees(*name, d.degree(), d.bits(), &measured);
        all_match &= cmp.matches_prediction();
        println!(
            "  {:<10} {:>6} {:>7} {:>10} {:>9} {:>6} {:>7}  {:?}",
            cmp.label,
            cmp.multilinear_degree,
            cmp.denominator_log,
            cmp.predicted(),
            cmp.phase_degree
                .map_or("none".to_string(), |v| v.to_string()),
            cmp.is_character,
            cmp.certified,
            measured
                .order_residuals
                .iter()
                .map(|r| format!("{r:.2e}"))
                .collect::<Vec<_>>()
        );
    }
    println!(
        "\n  Every case matched `multilinear + log2(denominator) - 1`: {all_match}.\n  \
         Degree 1 IS being a character, so a diagonal is a character exactly when\n  \
         both dials sit at their minimum: multilinear degree one AND +-1 valued."
    );

    println!("\n== The degree ladder is the Clifford hierarchy, rediscovered ==\n");
    for (level, names) in [
        (1u32, "z (the +-1 characters)"),
        (2, "s, cz (Clifford)"),
        (3, "t, cs, ccz (the first non-Clifford diagonals)"),
        (4, "ct, cccz"),
    ] {
        println!("  degree {level}: {names}");
    }

    println!("\n== The two groups the constellation is caught between ==\n");
    for levels in [1usize, 2] {
        let n = 8 * levels;
        let elements: Vec<u64> = (0..1u64 << n).collect();
        let add_residue: &dyn Fn(u64, u64) -> u64 = &|a, b| residue_add(levels, a, b);
        let add_xor: &dyn Fn(u64, u64) -> u64 = &xor;
        let mut differing = 0usize;
        let probe = 512.min(elements.len());
        for &a in elements.iter().take(probe) {
            for &b in elements.iter().take(probe) {
                if residue_add(levels, a, b) != a ^ b {
                    differing += 1;
                }
            }
        }
        println!(
            "  {levels} volume(s), n={n}: residue addition differs from XOR on {differing} \
             of {} probed pairs",
            probe * probe
        );
        let q = constellation::basis(0);
        let modulation = |bits: u64| constellation::modulation_phase(levels, &q, bits);
        let omega = std::f64::consts::FRAC_1_SQRT_2;
        let t_phase = |bits: u64| {
            if bits & 1 == 1 {
                C64::new(omega, omega)
            } else {
                C64::new(1.0, 0.0)
            }
        };
        for (label, phase) in [
            ("native modulate", &modulation as &dyn Fn(u64) -> C64),
            ("qubit t", &t_phase as &dyn Fn(u64) -> C64),
        ] {
            for (gname, add) in [("residue", add_residue), ("bits/XOR", add_xor)] {
                let group = PhaseGroup {
                    elements: &elements,
                    complete: true,
                    add,
                };
                let r = phase_degree(&group, phase, 6, 40_000, 1e-9);
                println!(
                    "    {label:<16} on {gname:<9} degree {:>5}   residuals {:?}",
                    r.degree().map_or("none".to_string(), |v| v.to_string()),
                    r.order_residuals
                        .iter()
                        .map(|x| format!("{x:.2e}"))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    println!(
        "\n  The native operators and the qubit path are characters of DIFFERENT\n  \
         groups. A modulation is degree 1 on the residue group and high-degree on\n  \
         the bits; a t is degree 3 on both. At one volume the two groups coincide\n  \
         exactly (E8/2E8 is F2^8 and class_of is linear); from two volumes on they\n  \
         do not, because extracting higher digits carries."
    );

    println!("\n== Why the self-hosted stack could fix a ccz but not a t ==\n");
    for (name, d) in [("ccz", diag(1, &[(0b111, 1)])), ("t", diag(3, &[(0b1, 1)]))] {
        let before = bits_degree(&d, 3);
        let stack = SelfHostedStack::plan(3, &d)?;
        let linear = stack.linearize(&d)?;
        let after = bits_degree(&linear, stack.width().min(16));
        println!(
            "  {name:<4} degree {:?} -> {:?}   layers {}   multilinear {} -> {}   character after: {}",
            before.degree(),
            after.degree(),
            stack.depth(),
            d.degree(),
            linear.degree(),
            after.is_character()
        );
    }
    println!(
        "\n  The stack reduces the MULTILINEAR term. The log-denominator term is\n  \
         untouched, so a diagonal whose degree comes entirely from its denominator\n  \
         is beyond it at any depth. That is the floor: a diagonal over 2^b-th roots\n  \
         has phase degree at least b, and only b = 1 is a character.\n\n  \
         And the sqrt(2) both other modules reported is this instrument's order-two\n  \
         residual for a t: |i - 1| = {:.6}.",
        std::f64::consts::SQRT_2
    );
    Ok(())
}
