//! Exact evaluation past Clifford+T: the QFT and phase estimation in
//! D[ζ_N], where the D[ω] evaluator stops at eighth turns.
//!
//! `cargo run --release --example exact_qft`

use quantsim::cyclotomic::{CyclotomicState, DZeta};
use quantsim::library::{iqft, qft};
use quantsim::prelude::*;
use std::f64::consts::PI;

/// The QFT of a basis state on `n` qubits, exact in D[ζ_N] (`N = 2^n`),
/// against the dense backend: the float error is measured, not estimated.
fn qft_error<const N: u32>(n: usize) -> Result<()> {
    let mut c = Circuit::new(n);
    c.x(0).x(n - 1);
    c.append(&qft(n), &(0..n).collect::<Vec<_>>());
    let exact = CyclotomicState::<N>::run(&c)?;
    let dense = Simulator::<C64>::new().run(&c)?;
    println!(
        "  n = {n:>2}  D[ζ_{N:<4}]  exact support {:>5}  dense max |error| {:.1e}",
        exact.support_exact(),
        exact.max_deviation_vs(dense.as_ref())
    );
    Ok(())
}

/// Phase estimation of `p(φ)` on `|1⟩` with `t = 5` counting qubits and
/// `φ = 2π·m/2^8`: on the counting grid when 8 | m, between bins otherwise.
fn phase_estimation(m: u64) -> Result<()> {
    const T: usize = 5;
    let phi = 2.0 * PI * m as f64 / 256.0;
    let mut c = Circuit::new(T + 1);
    c.x(T);
    for j in 0..T {
        c.h(j).cp(j, T, phi * (1u64 << j) as f64);
    }
    c.append(&iqft(T), &(0..T).collect::<Vec<_>>());
    let s = CyclotomicState::<256>::run(&c)?;
    let probs = (0..1u64 << T)
        .map(|y| s.probability_exact(y | 1 << T))
        .collect::<Result<Vec<DZeta<256>>>>()?;
    let (best, p) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.to_c64().re.total_cmp(&b.1.to_c64().re))
        .expect("nonempty");
    let zeros = probs.iter().filter(|w| w.is_zero()).count();
    println!(
        "  φ = 2π·{m:>2}/256  peak y = {best:>2}  P = {:.12}  outcomes with P exactly 0: {zeros:>2}  ΣP = 1 exactly: {}",
        p.to_c64().re,
        s.total_weight_exact()? == DZeta::int(1)
    );
    Ok(())
}

fn main() -> Result<()> {
    println!("QFT |x⟩, exact vs dense:");
    qft_error::<8>(3)?;
    qft_error::<16>(4)?;
    qft_error::<64>(6)?;
    qft_error::<256>(8)?;
    qft_error::<1024>(10)?;
    println!("Phase estimation, exact outcome distribution:");
    for m in [40, 41, 44] {
        phase_estimation(m)?;
    }
    Ok(())
}
