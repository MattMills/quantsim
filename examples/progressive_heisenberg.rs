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
        let cfg = Config {
            threshold: th,
            max_terms: None,
            checkpoint_every: 64,
            exclusion: true,
            retire_frozen: true,
        };
        let t0 = Instant::now();
        let p = propagate(&PauliSum::z(q), &rots, &cfg)?;
        println!(
            "  {th:9.0e}  {:6}   {:5}   {:+.10}   {:.2e}   {:?}",
            p.sum.len(),
            p.commuting_skips,
            p.expectation(),
            p.error_bound(),
            t0.elapsed()
        );
    }
    println!("\n  the skips are the backward light cone, counted while walking");
    println!("  rather than computed beforehand.");

    println!("\n== retrodiction: which gate spent my precision ==\n");
    let cfg = Config {
        threshold: 1e-6,
        max_terms: None,
        checkpoint_every: 32,
        exclusion: true,
        retire_frozen: true,
    };
    let p = propagate(&PauliSum::z(q), &rots, &cfg)?;
    println!(
        "  {} steps journalled, {} checkpoints, {} KiB",
        p.journal.steps().len(),
        p.journal.checkpoint_count(),
        p.journal.memory_bytes() / 1024
    );
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
        let cfg = Config {
            threshold: 1e-5,
            max_terms: None,
            checkpoint_every: 0,
            exclusion: true,
            retire_frozen: true,
        };
        let r = propagate_basis(&obs, &w, &rots, &cfg, true)?;
        println!(
            "  {n:2}   E={:+9.4}   {:11}   {:11}   {:13}   {:.1}×",
            r.total,
            obs.len(),
            r.peak_terms,
            r.separate_peak_total.unwrap(),
            r.sharing_factor().unwrap()
        );
    }

    println!("\n== factoring, rather than cutting ==\n");
    println!("  A cut is positional. Factoring discovers where the circuit does NOT");
    println!("  couple regions and holds the observable as a product of blocks, so");
    println!("  k independent blocks cost the SUM of their sizes, not the product.\n");
    println!("  circuit                blocks  stored  flat terms       merges   saving");
    {
        let n = 12;
        let rots = tfim_trotter(n, 1.0, 0.7, 0.3, 4);
        let f = propagate_factored((0, 1u64 << 6), &rots)?;
        println!(
            "  tfim n={n} (all coupled)     {:2}  {:6}  {:14}   {:6}   {:.1}×",
            f.blocks.len(),
            f.peak_stored,
            f.peak_flat,
            f.merges,
            f.factor_saving()
        );
    }
    for k in 4..=6usize {
        let (bs, n) = (4usize, k * 4);
        let mut rots = Vec::new();
        for _ in 0..4 {
            for b in 0..k {
                for q in b * bs..(b + 1) * bs - 1 {
                    rots.push(Rotation::rzz(q, q + 1, 0.3));
                }
            }
            for q in 0..n {
                rots.push(Rotation::rx(q, 0.44));
            }
        }
        let mut z = 0u64;
        for b in 0..k {
            z |= 1u64 << (b * bs + 1);
        }
        let f = propagate_factored((0, z), &rots)?;
        println!(
            "  {k} blocks of {bs}, n={n:<2}        {:2}  {:6}  {:14}   {:6}   {:.0}×",
            f.blocks.len(),
            f.peak_stored,
            f.peak_flat,
            f.merges,
            f.factor_saving()
        );
    }
    println!("\n  Stored grows linearly in the block count, flat grows exponentially,");
    println!("  and merges stay at 0 because a decoupled circuit never forces the");
    println!("  partition to break. TFIM collapses to one block at exactly 1.0× —");
    println!("  there is nothing to factor and the report says so.");

    println!("\n== engineering the structure, rather than accepting it ==\n");
    println!("  The honest limit of the table above is that it buys what the QUBIT");
    println!("  partition allows. But that partition is a statement about the basis:");
    println!("  what governs coupling is the symplectic form, so the real partition");
    println!("  is the components of the axes' ANTICOMMUTATION graph. Below, each");
    println!("  circuit is the same k decoupled blocks conjugated by a Clifford");
    println!("  scrambler until every axis is wide — all-to-all to any support rule.\n");
    println!("  circuit             axis wt  support  engineered  stored   flat terms        saving  sep");
    for k in 2..=6usize {
        let (bs, n) = (4usize, k * 4);
        let mut rots = Vec::new();
        for _ in 0..4 {
            for b in 0..k {
                for q in b * bs..(b + 1) * bs - 1 {
                    rots.push(Rotation::rzz(q, q + 1, 0.3));
                }
            }
            for q in 0..n {
                rots.push(Rotation::rx(q, 0.44));
            }
        }
        let mut z = 0u64;
        for b in 0..k {
            z |= 1u64 << (b * bs + 1);
        }
        let (scr, key, _) = scramble(&rots, (0, z), n);
        let wt = scr.iter().map(|r| r.weight()).max().unwrap_or(0);
        let r = propagate_engineered(key, &scr, n)?;
        println!(
            "  {k} blocks of {bs}, n={n:<2}    {wt:4}   {:6}    {:8}  {:6}  {:12}  {:11.0}×  {}",
            r.support_blocks,
            r.engineered_blocks,
            r.peak_stored,
            r.peak_flat,
            r.factor_saving(),
            r.separable_input
        );
    }
    println!("\n  Support sees ONE block at every size; the frame recovers all k.");
    println!("  Stored grows linearly, flat geometrically — the saving is in the");
    println!("  terms that never had to exist, not in a tighter truncation.\n");
    println!("  Two things this does NOT claim. The frame is a heuristic where two");
    println!("  components share a fully-commuting direction, so the block count is");
    println!("  MEASURED from the framed axes, never assumed. And the entanglement");
    println!("  is conserved, not destroyed: the frame moves it into the input state");
    println!("  |phi> = V+|0..0>, and the block product is only valid when |phi>");
    println!("  still factors. The `sep` column is that check. A CX-only scrambler");
    println!("  fixes |0..0> so it always holds; drawn from the full Clifford");
    println!("  generating set it fails about half the time, and then the");
    println!("  contraction is done flat and the report says so rather than");
    println!("  returning a wrong number.\n");
    {
        // the exact reduction that needs no frame at all
        let n = 8;
        let mut rots = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                rots.push(Rotation::rzz(a, b, 0.21));
            }
        }
        let c = coupling_of((0, 1u64 << 3), &rots);
        let r = propagate_engineered((0, 1u64 << 3), &rots, n)?;
        println!(
            "  And the cheapest case of all: {} all-to-all ZZ rotations on n={n},",
            rots.len()
        );
        println!(
            "  every one of which commutes with <Z_3>. {} of them are dropped",
            c.inert_rotations()
        );
        println!(
            "  outright — exactly, not approximately — and the answer {:+.1} falls",
            r.value
        );
        println!("  out with no propagation at all. A support rule sees one block of 8.");
    }

    println!("\n== one representation per block, chosen per block ==\n");
    println!("  Once the blocks are independent there is no reason for them to");
    println!("  SHARE a representation. In the engineered frame the whole thing is");
    println!("  <0..0|U+PU|0..0> = prod_b <phi_b| U_b+ P_b U_b |phi_b> — k complete");
    println!("  simulations, each on its own qubits, its own input, its own gates.\n");
    println!("  And the frame does something to the gates that is worth naming: it");
    println!("  puts back the locality the scrambler destroyed.\n");
    println!("  n     lab max/avg   scrambled max/avg   framed max/avg");
    for &(k, bs) in &[(3usize, 4usize), (4, 5), (4, 8), (5, 10)] {
        let n = k * bs;
        let mut rots = Vec::new();
        for _ in 0..3 {
            for b in 0..k {
                for q in b * bs..(b + 1) * bs - 1 {
                    rots.push(Rotation::rzz(q, q + 1, 0.3));
                }
            }
            for q in 0..n {
                rots.push(Rotation::rx(q, 0.44));
            }
        }
        let mut z = 0u64;
        for b in 0..k {
            z |= 1u64 << (b * bs + 1);
        }
        let (scr, key, _) = scramble(&rots, (0, z), n);
        let framed = decoupling_frame(key, &scr, n)?.rewrite(&scr);
        let wt = |r: &[Rotation]| {
            let mx = r.iter().map(|x| x.weight()).max().unwrap_or(0);
            (
                mx,
                r.iter().map(|x| x.weight()).sum::<usize>() as f64 / r.len() as f64,
            )
        };
        let (a, b) = wt(&rots);
        let (c, d) = wt(&scr);
        let (e, f) = wt(&framed);
        println!("  {n:3}     {a:2} / {b:.2}          {c:3} / {d:5.2}         {e:2} / {f:.2}");
    }
    println!("\n  The scrambled circuit has axes too wide for any windowed backend to");
    println!(
        "  ACCEPT (mps refuses past {}), so the only global option is dense.",
        quantsim::backend::MPS_MAX_WINDOW
    );
    println!("  Framed, it is two-local again — which is why an MPS can be pointed");
    println!("  at a block at all.\n");

    println!("  Four ten-qubit chains, scrambled to all-to-all — n=40, where a global");
    println!("  state vector is 16 TiB and every gate is too wide for a window:\n");
    {
        let (k, bs) = (4usize, 10usize);
        let n = k * bs;
        let mut rots = Vec::new();
        for _ in 0..3 {
            for b in 0..k {
                for q in b * bs..(b + 1) * bs - 1 {
                    rots.push(Rotation::rzz(q, q + 1, 0.3));
                }
            }
            for q in 0..n {
                rots.push(Rotation::rx(q, 0.44));
            }
        }
        let mut z = 0u64;
        for b in 0..k {
            z |= 1u64 << (b * bs + 1);
        }
        let (scr, key, _) = scramble(&rots, (0, z), n);
        println!("  solver        value                bytes    time         per block");
        for solver in [
            BlockSolver::Pauli,
            BlockSolver::Dense,
            BlockSolver::Mps { max_bond: 64 },
        ] {
            let t0 = Instant::now();
            let r = propagate_blocked(key, &scr, n, solver)?;
            println!(
                "  {:<12}  {:+.12}  {:9}  {:>10?}   {:?}",
                solver.label(),
                r.value,
                r.total_bytes(),
                t0.elapsed(),
                r.blocks
                    .iter()
                    .map(|b| b.detail.clone())
                    .collect::<Vec<_>>()
            );
        }
        println!("\n  Three unrelated representations, same twelve digits, none of them");
        println!("  ever holding more than ten qubits.\n");
    }

    println!("  And the cheapest one differs BY BLOCK. One 18-qubit block at depth 2");
    println!("  beside one 6-qubit block at depth 30 — nothing suits both:\n");
    {
        let (wide, deep_w, shallow, deep) = (18usize, 6usize, 2usize, 30usize);
        let n = wide + deep_w;
        let mut rots = Vec::new();
        for s in 0..shallow {
            for q in 0..wide - 1 {
                rots.push(Rotation::rzz(q, q + 1, 0.3 + 0.02 * s as f64));
            }
            for q in 0..wide {
                rots.push(Rotation::rx(q, 0.44));
            }
        }
        for s in 0..deep {
            for q in 0..deep_w - 1 {
                rots.push(Rotation::rzz(
                    wide + q,
                    wide + q + 1,
                    0.37 + 0.01 * s as f64,
                ));
            }
            for q in 0..deep_w {
                rots.push(Rotation::rx(wide + q, 0.29 + 0.005 * s as f64));
            }
        }
        let obs = (0u64, (1u64 << 1) | (1u64 << (wide + 1)));
        let (scr, key, _) = scramble(&rots, obs, n);
        println!("  plan               value                bytes    time         per block");
        let show = |label: &str, r: &BlockedReport, t: std::time::Duration| {
            println!(
                "  {label:<17}  {:+.12}  {:9}  {:>10?}   {:?}",
                r.value,
                r.total_bytes(),
                t,
                r.blocks
                    .iter()
                    .map(|b| format!("{}q {} {}", b.qubits, b.solver.label(), b.detail))
                    .collect::<Vec<_>>()
            );
        };
        for (label, s) in [
            ("uniform pauli", BlockSolver::Pauli),
            ("uniform dense", BlockSolver::Dense),
        ] {
            let t0 = Instant::now();
            let r = propagate_blocked(key, &scr, n, s)?;
            show(label, &r, t0.elapsed());
        }
        let t0 = Instant::now();
        let r = propagate_blocked_with(key, &scr, n, |_, mask, rots| {
            // a narrow block's whole Hilbert space is smaller than a deep
            // walk's Pauli sum; a wide block's is not
            if mask.count_ones() as usize <= 10 && rots > 50 {
                BlockSolver::Dense
            } else {
                BlockSolver::Pauli
            }
        })?;
        show("per-block choice", &r, t0.elapsed());
    }
    println!("\n  The mixed plan beats the better uniform one 4x and the worse one by");
    println!("  three orders of magnitude, on the same twelve digits. That is the");
    println!("  whole idea: not one representation that is good everywhere, but each");
    println!("  piece of the problem held where it is cheapest.\n");
    println!("  Two honest limits. The product form needs V+|0..0> to still factor");
    println!("  across the blocks; when it does not, propagate_blocked REFUSES rather");
    println!("  than multiplying numbers that are not independent. And this crate's");
    println!("  MPS runs a dependency-free Jacobi SVD, so its wall-clock above is a");
    println!("  statement about this implementation, not about MPS.\n");

    println!("\n== both directions of time, meeting in the middle ==\n");
    let n = 14;
    let mut rots = Vec::new();
    for _ in 0..6 {
        for k in 0..n {
            rots.push(Rotation::rx(k, 0.41));
        }
    }
    for _ in 0..5 {
        for k in 0..n - 1 {
            rots.push(Rotation::rzz(k, k + 1, 0.37));
        }
        for k in 0..n {
            rots.push(Rotation::rx(k, 0.29));
        }
    }
    println!(
        "  n={n}, {} rotations: a low-entanglement but NON-Clifford prefix,",
        rots.len()
    );
    println!("  then an entangling suffix — the shape where the forward half's");
    println!("  resource and the backward half's are genuinely different.\n");
    let cfg = Config {
        threshold: 1e-7,
        max_terms: None,
        checkpoint_every: 0,
        exclusion: true,
        retire_frozen: false,
    };
    let cuts: Vec<usize> = (0..=8).map(|i| i * rots.len() / 8).collect();
    for (label, fwd) in [
        ("sparse", Forward::Sparse),
        ("mps χ=64", Forward::Mps { max_bond: 64 }),
    ] {
        println!("  forward = {label}");
        println!("     cut   fwd bytes   back peak   meeting bytes   value");
        for m in cut_sweep(&PauliSum::z(n / 2), &rots, n, &cuts, fwd, &cfg)? {
            println!(
                "    {:4}   {:9}   {:9}   {:13}   {:+.9}",
                m.cut, m.forward_bytes, m.backward_peak, m.meeting_cost, m.value
            );
        }
        let best = auto_cut(&PauliSum::z(n / 2), &rots, n, 8, fwd, &cfg)?;
        println!(
            "    auto_cut picks {} at {} bytes\n",
            best.cut, best.meeting_cost
        );
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
    println!(
        "  n={n}, {} rotations, ground truth <Z_{q}> = {truth:+.9}\n",
        rots.len()
    );
    println!("  method            value           error      time");
    for bond in [16usize, 32] {
        let t0 = Instant::now();
        let mut m = MpsState::<C64>::with_config(
            n,
            MpsConfig {
                max_bond: bond,
                trunc_tol: 1e-14,
            },
        )?;
        for r in &rots {
            let (g, s) = r.gate()?;
            m.apply(&g, &s)?;
        }
        let v = pauli_expectation(&m as &dyn Backend<C64>, &[(q, Pauli::Z)])?.re;
        println!(
            "  mps χ={bond:<11}  {v:+.9}   {:.2e}   {:?}",
            (v - truth).abs(),
            t0.elapsed()
        );
    }
    for &th in &[1e-4f64, 1e-6] {
        let cfg = Config {
            threshold: th,
            max_terms: None,
            checkpoint_every: 0,
            exclusion: true,
            retire_frozen: true,
        };
        let t0 = Instant::now();
        let p = propagate(&PauliSum::z(q), &rots, &cfg)?;
        println!(
            "  heisenberg {th:.0e}  {:+.9}   {:.2e}   {:?}",
            p.expectation(),
            (p.expectation() - truth).abs(),
            t0.elapsed()
        );
    }
    println!("\n  Caveats, so the table is not read as more than it is: this crate's");
    println!("  MPS runs a dependency-free Jacobi SVD and is slow as implementations");
    println!("  go, so this compares what is IN the crate, not the two algorithms at");
    println!("  their best. And the L1 error bound is rigorous but loose — often");
    println!("  100-1000x the actual error — so it certifies, it does not estimate.");
    Ok(())
}
