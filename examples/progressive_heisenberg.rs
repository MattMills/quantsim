//! Progressive stateless computation, measured live: an observable
//! propagated backwards as a sum of Paulis, with the cost tracking the
//! question, the journal saying retrodictively where the precision went,
//! and one shared walk carrying a whole energy basis.
//!
//! Run with `cargo run --release --example progressive_heisenberg`.

use quantsim::heisenberg::*;
use quantsim::prelude::*;
use std::time::Instant;

fn dense_reference(rots: &[Rotation], n: usize, q: usize) -> Result<f64> {
    let mut d = DenseState::<C64>::new(n)?;
    for r in rots {
        let (m, s) = r.gate()?;
        d.apply(&m, &s)?;
    }
    Ok(pauli_expectation(&d as &dyn Backend<C64>, &[(q, Pauli::Z)])?.re)
}

/// A ladder: nearest-neighbour plus rung couplings, so the entanglement
/// is not confined to a line.
fn ladder(n: usize, rungs: usize, dt: f64, steps: usize) -> Vec<Rotation> {
    let mut out = Vec::new();
    for _ in 0..steps {
        for k in 0..n - 1 {
            out.push(Rotation::rzz(k, k + 1, -0.5 * dt));
        }
        for k in 0..n.saturating_sub(rungs) {
            out.push(Rotation::rzz(k, k + rungs, -0.5 * dt));
        }
        for k in 0..n {
            out.push(Rotation::rx(k, -1.4 * dt));
        }
    }
    out
}

fn main() -> Result<()> {
    println!("== the cost tracks the question, not the circuit ==\n");
    let (n, q) = (40usize, 20usize);
    let rots = tfim_trotter(n, 1.0, 0.7, 0.25, 8);
    println!("  TFIM n={n}, {} rotations, observable Z_{q}", rots.len());
    println!("  (dense would need {} bytes)\n", (1u128 << n) * 16);
    println!("  threshold   terms    skips   <Z>             bound      time");
    for &th in &[1e-1f64, 1e-3, 1e-5, 1e-7] {
        let cfg = Config { threshold: th, max_terms: None, checkpoint_every: 64, exclusion: true, retire_frozen: true };
        let t0 = Instant::now();
        let p = propagate(&PauliSum::z(q), &rots, &cfg)?;
        println!(
            "  {th:9.0e}  {:6}   {:5}   {:+.10}   {:.2e}   {:?}",
            p.sum.len(), p.commuting_skips, p.expectation(), p.error_bound(), t0.elapsed()
        );
    }
    println!("\n  the skips are the backward light cone, counted while walking");
    println!("  rather than computed beforehand.");

    println!("\n== retrodiction: which gate spent my precision ==\n");
    let cfg = Config { threshold: 1e-6, max_terms: None, checkpoint_every: 32, exclusion: true, retire_frozen: true };
    let p = propagate(&PauliSum::z(q), &rots, &cfg)?;
    println!("  {} steps journalled, {} checkpoints, {} KiB",
        p.journal.steps().len(), p.journal.checkpoint_count(), p.journal.memory_bytes() / 1024);
    println!("\n  error budget   blamed gate   (binary search on a monotone total)");
    for &b in &[1.0f64, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5] {
        match p.journal.blame_gate(b) {
            Some(g) => println!("  {b:11.0e}   gate {g:4} of {}", rots.len()),
            None => println!("  {b:11.0e}   never exceeded"),
        }
    }

    println!("\n== one shared walk over a whole energy basis ==\n");
    println!("   n   terms(H)   observables   peak shared   peak separate   sharing");
    for n in [20usize, 40, 60] {
        let rots = tfim_trotter(n, 1.0, 0.7, 0.25, 6);
        let (obs, w) = tfim_energy_basis(n, 1.0, 0.7);
        let cfg = Config { threshold: 1e-5, max_terms: None, checkpoint_every: 0, exclusion: true, retire_frozen: true };
        let r = propagate_basis(&obs, &w, &rots, &cfg, true)?;
        println!("  {n:2}   E={:+9.4}   {:11}   {:11}   {:13}   {:.1}×",
            r.total, obs.len(), r.peak_terms, r.separate_peak_total.unwrap(),
            r.sharing_factor().unwrap());
    }

    println!("\n== both directions of time, meeting in the middle ==\n");
    let n = 14;
    let mut rots = Vec::new();
    for _ in 0..6 { for k in 0..n { rots.push(Rotation::rx(k, 0.41)); } }
    for _ in 0..5 {
        for k in 0..n - 1 { rots.push(Rotation::rzz(k, k + 1, 0.37)); }
        for k in 0..n { rots.push(Rotation::rx(k, 0.29)); }
    }
    println!("  n={n}, {} rotations: a low-entanglement but NON-Clifford prefix,", rots.len());
    println!("  then an entangling suffix — the shape where the forward half's");
    println!("  resource and the backward half's are genuinely different.\n");
    let cfg = Config { threshold: 1e-7, max_terms: None, checkpoint_every: 0,
                       exclusion: true, retire_frozen: false };
    let cuts: Vec<usize> = (0..=8).map(|i| i * rots.len() / 8).collect();
    for (label, fwd) in [("sparse", Forward::Sparse),
                         ("mps χ=64", Forward::Mps { max_bond: 64 })] {
        println!("  forward = {label}");
        println!("     cut   fwd bytes   back peak   meeting bytes   value");
        for m in cut_sweep(&PauliSum::z(n / 2), &rots, n, &cuts, fwd, &cfg)? {
            println!("    {:4}   {:9}   {:9}   {:13}   {:+.9}",
                m.cut, m.forward_bytes, m.backward_peak, m.meeting_cost, m.value);
        }
        let best = auto_cut(&PauliSum::z(n / 2), &rots, n, 8, fwd, &cfg)?;
        println!("    auto_cut picks {} at {} bytes\n", best.cut, best.meeting_cost);
    }
    println!("  Sparse saturates within two layers, so its best cut is 0 and meeting");
    println!("  in the middle buys nothing. MPS has a genuine interior optimum. The");
    println!("  saving is real but it is conditional: it exists only because the two");
    println!("  halves are exponential in DIFFERENT resources.\n");
    println!("  And a trap worth naming: both exclusions are BOUNDARY conditions, not");
    println!("  circuit properties — they encode \"this walk ends at |0..0>\". Applying");
    println!("  them at an interior cut gave 0.18 absolute error before it was caught;");
    println!("  they are switched off for any cut past zero.");

    println!("\n== against the real competitor: MPS, on a ladder it dislikes ==\n");
    let (n, q) = (20usize, 10usize);
    let rots = ladder(n, 5, 0.3, 6);
    let truth = dense_reference(&rots, n, q)?;
    println!("  n={n}, {} rotations, ground truth <Z_{q}> = {truth:+.9}\n", rots.len());
    println!("  method            value           error      time");
    for bond in [16usize, 32] {
        let t0 = Instant::now();
        let mut m = MpsState::<C64>::with_config(n, MpsConfig { max_bond: bond, trunc_tol: 1e-14 })?;
        for r in &rots {
            let (g, s) = r.gate()?;
            m.apply(&g, &s)?;
        }
        let v = pauli_expectation(&m as &dyn Backend<C64>, &[(q, Pauli::Z)])?.re;
        println!("  mps χ={bond:<11}  {v:+.9}   {:.2e}   {:?}", (v - truth).abs(), t0.elapsed());
    }
    for &th in &[1e-4f64, 1e-6] {
        let cfg = Config { threshold: th, max_terms: None, checkpoint_every: 0, exclusion: true, retire_frozen: true };
        let t0 = Instant::now();
        let p = propagate(&PauliSum::z(q), &rots, &cfg)?;
        println!("  heisenberg {th:.0e}  {:+.9}   {:.2e}   {:?}",
            p.expectation(), (p.expectation() - truth).abs(), t0.elapsed());
    }
    println!("\n  Caveats, so the table is not read as more than it is: this crate's");
    println!("  MPS runs a dependency-free Jacobi SVD and is slow as implementations");
    println!("  go, so this compares what is IN the crate, not the two algorithms at");
    println!("  their best. And the L1 error bound is rigorous but loose — often");
    println!("  100-1000x the actual error — so it certifies, it does not estimate.");
    Ok(())
}
