//! Hello, entanglement: build a Bell pair, inspect it, sample it.
//!
//! Run with `cargo run --example bell`.

use quantsim::prelude::*;

fn main() -> Result<()> {
    let mut circuit = Circuit::new(2);
    circuit.h(0).cx(0, 1);

    let sim: Simulator = Simulator::new();
    let state = sim.run(&circuit)?;

    println!("amplitudes:");
    for (index, weight) in state.probabilities() {
        println!(
            "  |{index:02b}⟩  amp = {}  p = {weight:.4}",
            state.amplitude(index)
        );
    }

    use quantsim::gates::Pauli::*;
    println!(
        "⟨ZZ⟩ = {:+.4}",
        pauli_expectation(state.as_ref(), &[(0, Z), (1, Z)])?.re()
    );
    println!(
        "⟨XX⟩ = {:+.4}",
        pauli_expectation(state.as_ref(), &[(0, X), (1, X)])?.re()
    );

    let counts = state.sample(1000, &mut Prng::new(2026))?;
    println!("1000 shots: {counts:?}");
    Ok(())
}
