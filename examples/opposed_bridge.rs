//! Opposed Mathematics on quantsim: a Fermi–Hubbard Hamiltonian built in OM,
//! its conservation laws checked exactly as cancellation, its energy
//! evaluated on several backends and, on a Clifford+T state, exactly.
//!
//! `cargo run --release --example opposed_bridge`

use quantsim::om::om_hamiltonian::hubbard::HubbardModel;
use quantsim::opposed::{exact_to_om, expectation, expectation_exact, pauli_sum};
use quantsim::prelude::*;

fn main() -> quantsim::Result<()> {
    // 2×2 lattice, t = 1, U = 4, Jordan–Wigner onto 8 qubits. OM builds
    // 4·H so that every coefficient is an integer.
    let model = HubbardModel::new(2, 2, 1, 4, false);
    let h = model.hamiltonian();
    println!("2×2 Hubbard: {} Pauli terms", pauli_sum(&h).len());
    let (n, sz) = (model.number_operator(), model.spin_z_operator());
    println!(
        "  [H, N]  reduces to nil: {}",
        h.commutator_with(&n).is_nil()
    );
    println!(
        "  [H, Sz] reduces to nil: {}",
        h.commutator_with(&sz).is_nil()
    );

    let mut c = Circuit::new(8);
    for q in 0..8 {
        c.ry(q, 0.3 + 0.2 * q as f64);
    }
    for q in 0..7 {
        c.cx(q, q + 1);
    }
    let sim: Simulator = Simulator::new();
    for backend in ["dense", "sparse", "mps"] {
        let e = expectation(&h, sim.run_on(backend, &c)?.as_ref())?;
        println!("  <H> on {backend:<6} {:+.12}", e.re / 4.0);
    }

    let mut ct = Circuit::new(8);
    for q in 0..8 {
        ct.h(q).t(q);
    }
    for q in 0..7 {
        ct.cx(q, q + 1);
    }
    let exact = ExactState::run(&ct)?;
    let e = expectation_exact(&h, &exact)?;
    let (coeffs, k) = e.parts();
    println!(
        "  <4H> on a Clifford+T state, exactly: {coeffs:?} / sqrt2^{k}, so <H> = {}",
        e.to_c64().re / 4.0
    );
    let (om, k) = exact_to_om(&exact)?;
    println!(
        "  that state in OM: {} points over sqrt2^{k}, opposition mass {}",
        om.distinct_points(),
        om.opposition_mass()
    );
    Ok(())
}
