//! What the path sum's surviving variables are actually stalled ON.
//!
//! `h*` counts variables the rewrite system could not eliminate. That is
//! a cost number, and cost numbers have been the wrong thing to look at.
//! `stall_census` reports the *reason* each one survived, and the reasons
//! are not equivalent:
//!
//! * `Shape` / `Coupling` / `Pivot` are structural — the variable sits
//!   inside a compound monomial, or a coupling is off a half turn. No
//!   choice of coefficient ring touches them.
//! * `Alignment` is not structural. The shape is exactly what rule G
//!   wants; the sum `Σ_y ω^{c·y + ½·y·L}` simply fails to close because
//!   the self-coefficient `c` is off the four points where the elliptic
//!   Gauss sum has a closed form (0, ¼, ½, ¾). A `T` gate contributes an
//!   eighth, which is not one of them.
//!
//! An `Alignment` stall is therefore a statement about the *ring the
//! coefficients live in*, not about the circuit. Closing one means
//! summing `1 + e^{iπ/4}(−1)^L`, whose magnitude is `2cos(π/8)` or
//! `2sin(π/8)` depending on the parity `L` — a factor of `(1+√2)^{∓1}`,
//! the fundamental unit of `ℤ[√2]`. The crate's magnitude bookkeeping
//! counts only integer powers of `√2`, so it cannot hold that, and the
//! rule declines to fire.
//!
//! This measures how much of `h*` is that, on the experiment's own
//! family.
//!
//! `cargo run --release --example dcs_stall_ring`

use quantsim::dcs::Dcs;
use quantsim::pathsum::{PathSum, Stall};
use std::time::{Duration, Instant};

fn main() {
    println!("{}", "─".repeat(78));
    println!("WHY THE SURVIVING VARIABLES SURVIVED  (DCS at the experiment's doping rate)");
    println!("{}", "─".repeat(78));
    println!("     n     t     h*    shape  coupling  pivot   ALIGNMENT   align%     time");
    for n in [6, 8, 10, 12, 14, 16, 18] {
        let d = Dcs::scaled(n);
        let c = d.circuit();
        let mut ps = PathSum::new(n);
        if ps.apply_circuit(&c, false).is_err() {
            continue;
        }
        let t0 = Instant::now();
        quantsim::guard::with_time_budget(Duration::from_secs(120), || {
            let _ = ps.try_reduce();
        });
        if ps.reduction_cut_short() {
            println!("  {n:>4}  {:>4}   (reduction did not reach a fixpoint)", d.t_gates);
            continue;
        }
        let census = ps.stall_census();
        let mut shape = 0;
        let mut coupling = 0;
        let mut pivot = 0;
        let mut align = 0;
        for (_, s) in &census {
            match s {
                Stall::Shape { .. } => shape += 1,
                Stall::Coupling(_) => coupling += 1,
                Stall::Pivot => pivot += 1,
                Stall::Alignment(_) => align += 1,
            }
        }
        let h = ps.internal_vars();
        let deg = ps.degree_profile();
        println!(
            "  {n:>4}  {:>4}  {h:>5}  {shape:>7}  {coupling:>8}  {pivot:>5}  {align:>10}  {:>6.1}%  {:>8.2?}",
            d.t_gates,
            if h > 0 { 100.0 * align as f64 / h as f64 } else { 0.0 },
            t0.elapsed()
        );
        let profile: Vec<String> = deg.iter().map(|(k, v)| format!("deg {k}: {v}")).collect();
        println!("           surviving monomials by degree — {}", profile.join(", "));
    }
    println!();
    println!("  Alignment stalls sit on odd eighths — the T gate's own coefficient —");
    println!("  and closing them is a question about the coefficient ring, not about");
    println!("  the circuit's size. Structural stalls are immune to that choice.");
    println!("{}", "─".repeat(78));
}
