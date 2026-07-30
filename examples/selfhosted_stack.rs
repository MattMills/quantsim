//! A geometric object that computes itself: a stack of E8 volumes where
//! each layer holds the layers below it as its own coordinates, so the
//! diagonal becomes linear — and a linear diagonal is a product of
//! single-qubit phases.
//!
//! Prints the measured recursive expansion (one layer per degree), the
//! exactness of the route against the registry's own gates, the point at
//! which the object becomes a character of its F₂ group, and what the
//! trade costs.
//!
//! Run with `cargo run --release --example selfhosted_stack`.

use quantsim::prelude::*;
use quantsim::selfhost::{
    amortization, full_degree, recursive_expansion, verify, Diagonal, SelfHostedStack,
};

fn diagonal(bits: u32, terms: &[(u64, i64)]) -> Result<Diagonal> {
    let mut d = Diagonal::new(bits)?;
    for &(mask, coeff) in terms {
        d.term(mask, coeff);
    }
    Ok(d)
}

fn main() -> Result<()> {
    let sim = Simulator::<C64>::new();

    println!("== The object, on three diagonals ==\n");
    println!(
        "  {:<6} {:>4} {:>6} {:>6} {:>8} {:>6} {:>9} {:>9} {:>7}",
        "gate", "deg", "layers", "vols", "occupancy", "width", "chi(sub)", "chi(stack)", "char?"
    );
    let cases = [
        ("cz", diagonal(1, &[(0b011, 1)])?),
        ("ccz", diagonal(1, &[(0b111, 1)])?),
        ("cccz", diagonal(1, &[(0b1111, 1)])?),
        ("t", diagonal(3, &[(0b001, 1)])?),
    ];
    for (name, d) in &cases {
        let stack = SelfHostedStack::plan(6, d)?;
        let r = verify(&sim, "sparse", 6, d)?;
        println!(
            "  {:<6} {:>4} {:>6} {:>6} {:>8.3} {:>6} {:>9.2e} {:>9.2e} {:>7}",
            name,
            d.degree(),
            stack.depth(),
            stack.volumes(),
            stack.occupancy(),
            stack.width(),
            r.substrate_character_residual,
            r.stack_character_residual,
            r.became_character()
        );
    }
    println!(
        "\n  Every route above reproduced the substrate diagonal exactly \
         (deviation {:.1e}).",
        cases
            .iter()
            .map(|(_, d)| verify(&sim, "sparse", 6, d).map(|r| r.deviation))
            .collect::<Result<Vec<f64>>>()?
            .into_iter()
            .fold(0.0, f64::max)
    );
    println!(
        "  A `t` is the honest exception: degree 1 already, so the object does\n  \
         nothing — and its phases are eighth roots, so it never becomes a\n  \
         character. That residual is the same sqrt(2) = {:.6} the constellation\n  \
         measures for a t through the qubit path (tests/e8_across.rs).",
        std::f64::consts::SQRT_2
    );

    println!("\n== Recursive expansion: each layer buys one degree ==\n");
    println!(
        "  {:>6} {:>7} {:>6} {:>6} {:>8} {:>9} {:>10}",
        "degree", "layers", "vols", "width", "linear", "toffoli", "deviation"
    );
    for r in recursive_expansion(&sim, "sparse", 8, 7)? {
        println!(
            "  {:>6} {:>7} {:>6} {:>6} {:>8} {:>9} {:>10.1e}",
            r.original_degree,
            r.depth,
            r.volumes,
            r.width,
            r.linear_degree,
            r.toffoli_equivalents,
            r.deviation
        );
    }
    println!("\n  layers = degree - 1, exactly, and the linearized degree is 1 throughout.");

    println!("\n== A dense phase polynomial, where the volumes fill up ==\n");
    println!(
        "  {:>6} {:>7} {:>7} {:>6} {:>6} {:>9} {:>9} {:>10}",
        "degree", "terms", "layers", "vols", "width", "occupancy", "toffoli", "deviation"
    );
    for degree in 2u32..=4 {
        let d = full_degree(6, degree, 1)?;
        let stack = SelfHostedStack::plan(6, &d)?;
        let r = verify(&sim, "sparse", 6, &d)?;
        println!(
            "  {:>6} {:>7} {:>7} {:>6} {:>6} {:>9.3} {:>9} {:>10.1e}",
            degree,
            d.terms().len(),
            stack.depth(),
            stack.volumes(),
            stack.width(),
            stack.occupancy(),
            r.toffoli_equivalents,
            r.deviation
        );
    }

    println!("\n== What the trade costs ==\n");
    let ccz = diagonal(1, &[(0b111, 1)])?;
    let reps = [1usize, 2, 8, 32, 128, 256, 512, 1024];
    let a = amortization(&sim, "sparse", 6, &ccz, &reps)?;
    println!("  ccz on a 6-qubit substrate, stack width {}:", a.width);
    println!("  {:>12} {:?}", "uses", a.repetitions);
    println!(
        "  {:>12} {:?}   (max arity {})",
        "direct 2q+", a.direct_entangling, a.direct_max_arity
    );
    println!(
        "  {:>12} {:?}   (steady arity {})",
        "stack 2q+", a.stack_entangling, a.stack_steady_max_arity
    );
    println!("  {:>12} {:?}", "direct ns", a.direct_nanos);
    println!("  {:>12} {:?}", "stack ns", a.stack_nanos);
    println!(
        "\n  counted crossover  : {:?} uses — the direct route pays an entangling\n  \
         {:>19} gate every time; the object pays once.\n  \
         measured crossover : {:?} uses — a classical simulator applies a diagonal\n  \
         {:>19} kernel in O(support) whatever its arity, so the win shows up in\n  \
         {:>19} the slope, not the constant.",
        a.counted_crossover, "", a.measured_crossover, "", ""
    );

    let slope = |v: &[usize]| {
        (v[v.len() - 1] as f64 - v[0] as f64) / (reps[reps.len() - 1] - reps[0]) as f64
    };
    println!(
        "\n  per-use slope      : direct {:.1} ns, stack {:.1} ns",
        slope(&a.direct_nanos),
        slope(&a.stack_nanos)
    );
    Ok(())
}
