//! What is the amplitude, as a function?
//!
//! Every other probe in this directory measures the size of some
//! representation — amplitudes, path variables, Pauli terms, stabilizer
//! terms, branches, bond dimension. Each one is documented as
//! "exponential in X", so measuring X and finding it large confirms the
//! premise it started from. That is circular, and it never asks the
//! direct question.
//!
//! The direct question is about `⟨x|ψ⟩` as a map `𝔽₂^n → ℂ`, with no
//! representation mentioned:
//!
//! * how many distinct **magnitudes** does it take? A graph state takes
//!   exactly one. A generic state takes `2^n`.
//! * how many nonzero **Walsh–Hadamard coefficients** does `p(x)` have?
//!   That is `⟨Z_S⟩` over all `S`, and it is the honest measure of how
//!   much of the function is actually there — a function with few
//!   nonzero coefficients is cheap however large a state vector is.
//! * what is the **algebraic degree** of the phase over `𝔽₂`? If the
//!   phase is a bounded-degree polynomial in the bits, the amplitude has
//!   a closed form and nothing needs to be summed at all.
//!
//! The Clifford skeleton is the existence proof that all three can come
//! back trivial: one magnitude, and a degree-2 phase. Whether doping
//! destroys that or merely hides it from a rewrite system is a question
//! about the function, and this asks it of the function.
//!
//! `cargo run --release --example dcs_function`

use quantsim::dcs::Dcs;
use quantsim::prelude::*;
use std::collections::BTreeMap;

/// Walsh–Hadamard transform in place.
fn fwht(v: &mut [f64]) {
    let n = v.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// Möbius transform over 𝔽₂: turn a truth table into the coefficients of
/// its algebraic normal form. The top nonzero coefficient's popcount is
/// the algebraic degree.
fn anf_degree(truth: &[bool]) -> usize {
    let mut f: Vec<bool> = truth.to_vec();
    let n = f.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                f[j + h] ^= f[j];
            }
        }
        h *= 2;
    }
    (0..n)
        .filter(|&i| f[i])
        .map(|i| (i as u64).count_ones() as usize)
        .max()
        .unwrap_or(0)
}

fn main() {
    println!("{}", "─".repeat(78));
    println!("THE AMPLITUDE AS A FUNCTION  (no representation size anywhere below)");
    println!("{}", "─".repeat(78));
    println!("     n     t   |amp| values   phase values   nonzero WH coeffs of p(x)   support");
    for n in [6, 8, 10, 12, 14] {
        for (label, d) in [
            ("skeleton", Dcs::scaled(n).with_t(0)),
            ("doped   ", Dcs::scaled(n)),
        ] {
            let state = Simulator::<C64>::new().run(&d.circuit()).unwrap();
            let dim = 1usize << n;

            // Distinct magnitudes and phases, bucketed at 1e-9.
            let mut mags: BTreeMap<i64, usize> = BTreeMap::new();
            let mut phases: BTreeMap<i64, usize> = BTreeMap::new();
            let mut probs = vec![0.0f64; dim];
            let mut support = 0usize;
            for x in 0..dim as u64 {
                let a = state.amplitude(x);
                probs[x as usize] = a.norm_sqr();
                let m = a.norm();
                if m > 1e-12 {
                    support += 1;
                    *mags.entry((m * 1e9).round() as i64).or_insert(0) += 1;
                    let ph = a.im.atan2(a.re) / std::f64::consts::TAU;
                    *phases.entry((ph * 1e6).round() as i64).or_insert(0) += 1;
                }
            }
            let mut wh = probs.clone();
            fwht(&mut wh);
            let nz = wh.iter().filter(|c| c.abs() > 1e-9).count();

            println!(
                "  {label} {n:>3}  {:>4}  {:>12}  {:>13}  {:>25}  {:>8}",
                d.t_gates,
                mags.len(),
                phases.len(),
                format!("{nz} of {dim}"),
                support
            );

            // If every amplitude shares one magnitude the state is a
            // "phase state": all the content is in the phase, and the
            // phase's algebraic degree is then the whole question.
            if mags.len() == 1 {
                // Phase in eighths of a turn, if it lands there.
                let mut eighth_ok = true;
                let mut bits: Vec<Vec<bool>> = vec![vec![false; dim]; 3];
                for x in 0..dim {
                    let a = state.amplitude(x as u64);
                    let turns = a.im.atan2(a.re) / std::f64::consts::TAU;
                    let e = (turns * 8.0).rem_euclid(8.0);
                    if (e - e.round()).abs() > 1e-6 {
                        eighth_ok = false;
                        break;
                    }
                    let e = e.round() as usize % 8;
                    for (b, plane) in bits.iter_mut().enumerate() {
                        plane[x] = (e >> b) & 1 == 1;
                    }
                }
                if eighth_ok {
                    let degs: Vec<usize> = bits.iter().map(|p| anf_degree(p)).collect();
                    println!(
                        "            one magnitude, phase in eighths of a turn; \
                         algebraic degree per bit-plane (1/8, 1/4, 1/2) = {degs:?}"
                    );
                } else {
                    println!("            one magnitude, but the phase is not on the eighth grid");
                }
            }
        }
    }
    println!("{}", "─".repeat(78));
}
