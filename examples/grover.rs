//! Grover search demo: find a marked state in an unstructured space with
//! ~√N oracle calls, and watch the success probability match theory.
//!
//! Run with `cargo run --example grover`.

use quantsim::prelude::*;

fn main() -> Result<()> {
    let n = 4;
    let marked = 0b1011u64;
    let sim: Simulator = Simulator::new();

    println!("searching {} states for |{marked:04b}⟩:", 1 << n);
    let theta = (1.0 / (1u64 << n) as f64).sqrt().asin();
    for iterations in 0..=5 {
        let circuit = library::grover(n, marked, iterations)?;
        let state = sim.run(&circuit)?;
        let p = state.probability(marked);
        let theory = ((2 * iterations + 1) as f64 * theta).sin().powi(2);
        println!("  {iterations} iteration(s): P(marked) = {p:.6}   closed form: {theory:.6}");
    }

    let best = library::grover(n, marked, 3)?;
    let counts = sim.run(&best)?.sample(1024, &mut Prng::new(7))?;
    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
    println!("1024 shots after 3 iterations (top outcomes):");
    for (index, count) in sorted.into_iter().take(4) {
        println!("  |{index:04b}⟩ × {count}");
    }
    Ok(())
}
