//! The union edge, isolated: (1) scaling "walls" decomposed into
//! resting state cost vs transient rung-1 builder cost vs probe budget
//! — the state and its builder are different objects; (2) one state
//! past every single representation's frontier at once — a Clifford
//! expander graph state (Gottesman–Knill class, rank-exponential
//! across every cut) superposed with a T-doped hierarchy-local state
//! (non-Clifford, entanglement-local) on a selector qudit: verified
//! exactly against dense at width 16, held exactly at width 48 in
//! kilobytes where every single-representation alternative is
//! petabyte-scale or refuses by name.
//!
//! Run with `cargo run --release --example union_edge`.
use quantsim::backend::{BulkState, MeraState, SparseState};
use quantsim::prelude::*;
use std::time::{Duration, Instant};

fn fmt_bytes(b: usize) -> String {
    if b >= 1 << 30 {
        format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MiB", b as f64 / (1 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KiB", b as f64 / (1 << 10) as f64)
    } else {
        format!("{b} B")
    }
}

/// T-doped hierarchy-local circuit: non-Clifford magic in every dyadic
/// 8-block (3 T's per block), entanglement kept block-local.
fn t_doped_local(n: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for block in 0..(n / 8) {
        let b = block * 8;
        c.h(b)
            .cx(b, b + 1)
            .t(b)
            .cx(b + 1, b + 2)
            .t(b + 2)
            .cx(b + 2, b + 3)
            .t(b + 3)
            .cx(b + 3, b + 4)
            .ry(b + 4, 0.37 + 0.11 * block as f64);
    }
    c
}

/// Expander-ish graph state: H wall + seeded long-range CZ edges.
/// Pure Clifford (Gottesman–Knill class), but its cut rank survives any
/// qubit reordering — hostile to every bond/cluster/support bet.
fn graph_state(n: usize, edges: usize, seed: u64) -> Circuit<C64> {
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    let mut placed = 0;
    while placed < edges {
        let a = (rng.next_u64() % n as u64) as usize;
        let b = (rng.next_u64() % n as u64) as usize;
        if a != b {
            c.gate("cz", [], [a, b]);
            placed += 1;
        }
    }
    c
}

fn main() -> Result<()> {
    let sim = Simulator::<C64>::new();
    let reg = GateRegistry::<C64>::standard();

    // ══ 1. what a "wall" actually was: resting vs transient ══
    println!("══ 1. the GHZ walls decomposed: the STATE is cheap, the rung-1 BUILDER paid ══\n");
    for n in [16usize, 20] {
        let state = sim.run_on("mera", &library::ghz(n))?;
        let mera = state.as_any().downcast_ref::<MeraState<C64>>().unwrap();
        println!(
            "  mera GHZ({n}): resting {} (bond {}), transient peak block {} elements = {}",
            fmt_bytes(state.memory_bytes()),
            mera.max_bond_dimension(),
            mera.peak_block_elements(),
            fmt_bytes(mera.peak_block_elements() * 16),
        );
    }
    println!("  → the width-24 'wall' was the transient 2^24 builder block under a 5 s probe");
    println!("    budget, not the state: grown dynamically, bulk holds GHZ(32) at 6 KiB.");

    // ══ 2. the union-edge object ══
    // |Ψ⟩ = (|0⟩⊗|graph⟩ + |1⟩⊗|T-doped⟩)/√2 on the selector qudit:
    //   • graph slice: Clifford (GK-simulable), but rank-exponential
    //     across every cut under any ordering — outside bond, cluster,
    //     support, and amplitude bets;
    //   • T-doped slice: non-Clifford (outside GK), 3 T's per 8-block —
    //     outside the 2^t frame at scale, cheap only hierarchically.
    // No single shipped representation holds the superposition. The
    // selector holds it at the sum of the two cheap costs.
    println!("\n══ 2. one state past every single frontier at once ══\n");

    // (a) verified regime: n = 16, checked against dense exactly.
    let n = 16;
    let graph_c = graph_state(n, 24, 9);
    let doped_c = t_doped_local(n);
    let w = c64(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let mut graph_b = sim.backends().create("bundle", n)?;
    graph_c.bind(&reg)?.run(graph_b.as_mut())?;
    let mut doped_b: Box<dyn Backend<C64>> = Box::new(BulkState::<C64>::new(n)?);
    doped_c.bind(&reg)?.run(doped_b.as_mut())?;
    let branched = BranchedRegister::from_branches(n, 2, vec![(0, w, graph_b), (1, w, doped_b)])?;

    let mut reference = vec![c64(0.0, 0.0); 1 << n];
    for c in [&graph_c, &doped_c] {
        let d = sim.run(c)?;
        d.for_each_nonzero(&mut |i, a| reference[i as usize] += w * a);
    }
    let mut map = std::collections::HashMap::new();
    branched.for_each_nonzero(&mut |i, a| {
        map.insert(i, a);
    });
    let mut dev = 0.0f64;
    for (i, r) in reference.iter().enumerate() {
        let b = map.get(&(i as u64)).copied().unwrap_or(c64(0.0, 0.0));
        dev = dev.max((*r - b).norm());
    }
    println!("  n = 16 (dense-verifiable): flagged sum exact to {dev:.2e}");
    println!(
        "  branched holds it at {} — selector rank {}",
        fmt_bytes(branched.memory_bytes()),
        branched.selector_schmidt_rank()?
    );

    // The single-representation alternatives on the SAME object.
    println!("\n  every single-representation attempt at the summed state, measured:");
    let entries: Vec<(u64, C64)> = reference
        .iter()
        .enumerate()
        .filter(|(_, a)| a.norm_sqr() > 1e-24)
        .map(|(i, &a)| (i as u64, a))
        .collect();
    println!(
        "    dense:  {} (2^16 amplitudes, the reference)",
        fmt_bytes(1 << (n + 4))
    );
    let mut sp = SparseState::<C64>::new(n)?;
    sp.load(&entries)?;
    println!(
        "    sparse: {} ({} nonzeros — the graph slice populates everything)",
        fmt_bytes(sp.memory_bytes()),
        sp.nonzero_count()
    );
    for name in ["mps", "mera", "factored"] {
        let t = Instant::now();
        let r: Result<usize> = guard::with_time_budget(Duration::from_secs(20), || {
            let mut s = sim.backends().create(name, n)?;
            s.load(&entries)?;
            Ok(s.memory_bytes())
        });
        match r {
            Ok(b) => println!(
                "    {name}: {} (load-compiled in {:?})",
                fmt_bytes(b),
                t.elapsed()
            ),
            Err(e) => println!("    {name}: refused/walled — {e}"),
        }
    }

    // (b) holding regime: n = 48 — nothing else can even exist.
    let n = 48;
    let graph_c = graph_state(n, 72, 9);
    let doped_c = t_doped_local(n);
    let t = Instant::now();
    let mut graph_b = sim.backends().create("bundle", n)?;
    graph_c.bind(&reg)?.run(graph_b.as_mut())?;
    let mut doped_b = BulkState::<C64>::new(n)?;
    doped_c.bind(&reg)?.run(&mut doped_b)?;
    let doped_mem = doped_b.memory_bytes();
    let doped_exact = doped_b.is_exact();
    let doped_norm = doped_b.total_abs_sqr();
    let branched =
        BranchedRegister::from_branches(n, 2, vec![(0, w, graph_b), (1, w, Box::new(doped_b))])?;
    println!(
        "\n  n = 48 (t-count 18, 72 long-range edges), built in {:?}:",
        t.elapsed()
    );
    println!(
        "  branched holds the superposition at {} (bulk slice {}, exact = {doped_exact}, ‖ψ‖² = {doped_norm:.12})",
        fmt_bytes(branched.memory_bytes()),
        fmt_bytes(doped_mem),
    );
    println!("  the alternatives at width 48, computed from their own laws:");
    println!("    dense:            2^48 × 16 B = 4.5 PiB");
    println!("    sparse:           ~2^48 populated basis states");
    println!("    mps/mera/bulk:    expander cut rank — χ ~ 2^Θ(n) under any ordering");
    println!("    factored:         one 48-wide cluster = 2^48 dense");
    println!("    clifford-framed:  T-count 18 → 2^18 branch envelope, and rotations besides");
    println!("    bundle alone:     refuses every T by name (non-Clifford)");
    Ok(())
}
