//! The dyadic cone: the two-directional structure recursively, at every
//! scale, with the two directions pruning each other.
//!
//! Bisect the gate axis; bisect each half; repeat. Every node carries
//! what the past can have reached at its left edge and what the future
//! can still see at its right, and contracts the answer at its own
//! midpoint — so the tree checks itself `2^{d+1} − 1` ways instead of
//! one.

use quantsim::backend::pauli_expectation;
use quantsim::dyadic::{self, Config};
use quantsim::heisenberg::*;
use quantsim::horizon::Forward;
use quantsim::prelude::*;

fn ops_of(key: PauliKey, n: usize) -> Vec<(usize, Pauli)> {
    (0..n)
        .filter_map(|q| match (key.0 >> q & 1, key.1 >> q & 1) {
            (1, 0) => Some((q, Pauli::X)),
            (0, 1) => Some((q, Pauli::Z)),
            (1, 1) => Some((q, Pauli::Y)),
            _ => None,
        })
        .collect()
}

fn dense_at(rotations: &[Rotation], n: usize) -> Result<DenseState<C64>> {
    let mut d = DenseState::<C64>::new(n)?;
    for r in rotations {
        let (m, s) = r.gate()?;
        d.apply(&m, &s)?;
    }
    Ok(d)
}

fn row(vals: &[f64], places: usize) -> String {
    vals.iter()
        .map(|v| format!("{v:.places$}", places = places))
        .collect::<Vec<_>>()
        .join("  ")
}

fn main() -> Result<()> {
    let (n, depth) = (12usize, 5usize);
    let rots = tfim_trotter(n, 1.0, 0.7, 0.35, 6);
    let obs = (0u64, 1u64 << 6);
    let cfg = Config {
        depth,
        threshold: 0.0,
        forward: Forward::Dense,
        two_sided_max_terms: 0,
    };
    let c = dyadic::dyadic_cone(obs, &rots, n, &cfg)?;

    println!("== a recursive, two-directional, interacting cone ==\n");
    println!("  n={n}, {} rotations, observable Z_6, bisected {depth} times.", c.gates);
    println!("  {} nodes: level k holds 2^k segments of {} / 2^k gates each.\n", c.nodes.len(), c.gates);

    println!("  Every node contracts the answer at its OWN midpoint, so the tree");
    println!("  self-checks {} ways rather than one:\n", c.nodes.len());
    println!("    spread across all {} nodes   {:.3e}", c.nodes.len(), c.value_spread());
    println!("    the answer itself             {:+.12}\n", c.nodes[0].value);

    println!("── what the recursion conserves, and what it resolves ──\n");
    println!("  Children partition their parent's gates exactly. So EVERY additive");
    println!("  quantity is identical at every level — not a result, a check the");
    println!("  recursion performs on itself. If these drift, cells were lost.\n");
    println!("    level              {}", (0..=depth).map(|l| format!("{l:>6}")).collect::<Vec<_>>().join(""));
    println!("    segments           {}", (0..=depth).map(|l| format!("{:>6}", 1usize << l)).collect::<Vec<_>>().join(""));
    println!("    gates per segment  {}", (0..=depth).map(|l| format!("{:>6}", c.gates / (1usize << l))).collect::<Vec<_>>().join(""));
    println!();
    println!("    live-area density  {}   conserved", row(&c.density_by_level(), 4));
    println!("    front velocity     {}   conserved", row(&c.velocity_by_level(), 4));
    println!();
    println!("  And what a coarse window averages away, a finer one separates. A");
    println!("  long segment holds the front's bursts together with the stretches");
    println!("  where it is saturated and cannot move, so the peak it reports is");
    println!("  too low — and climbs every time the window halves:\n");
    println!("    PEAK velocity      {}   resolves", row(&c.peak_velocity_by_level(), 4));
    println!("    mean diamond (q)   {}   resolves", row(&c.mean_diamond_by_level(), 2));
    println!();
    println!("    resolution gain    {:.2}x  — the finest scale against the coarsest", c.resolution_gain());
    println!("\n  That is what the recursion is for. The conserved quantities are the");
    println!("  invariants it can be audited against; only the extremal ones carry");
    println!("  information that a single scale did not already have.\n");

    println!("── where the two directions interact ──\n");
    println!("  A backward walk alone can only bound a dropped term by |c|, because");
    println!("  |<psi|P|psi>| <= 1 is all that is known without the state. There is no");
    println!("  sound refinement of that, and reaching for one is a mistake — I made");
    println!("  it earlier in this work and had to retract it. But AT A CUT the state");
    println!("  is in hand, so |c|.|<psi|P|psi>| is computable, still rigorous, and");
    println!("  never worse:\n");
    println!("    level   nodes   terms    one-sided S|c|   two-sided   tighter by");
    for l in 0..=depth {
        let nodes = c.level(l);
        let one: f64 = nodes.iter().map(|x| x.l1).sum::<f64>() / nodes.len() as f64;
        let two: f64 = nodes.iter().filter_map(|x| x.two_sided_l1).sum::<f64>()
            / nodes.len() as f64;
        let terms: usize = nodes.iter().map(|x| x.terms).sum::<usize>() / nodes.len();
        println!(
            "    {l:5}   {:5}   {terms:5}   {one:14.4}   {two:9.4}   {:.2}x",
            nodes.len(),
            if two > 0.0 { one / two } else { f64::INFINITY }
        );
    }
    println!("\n    best node {:.2}x, mean {:.2}x\n", c.tightening(), c.mean_tightening());

    println!("  And it is a bound, not a heuristic. Prune to a budget at the middle");
    println!("  cut, contract what is left, and compare against dense:\n");
    let cut = rots.len() / 2;
    let back = propagate(
        &PauliSum::from_key(obs),
        &rots[cut..],
        &quantsim::heisenberg::Config {
            threshold: 0.0,
            max_terms: None,
            checkpoint_every: 0,
            exclusion: false,
            retire_frozen: false,
        },
    )?
    .sum;
    let fwd = dense_at(&rots[..cut], n)?;
    let state = &fwd as &dyn Backend<C64>;
    let truth = {
        let d = dense_at(&rots, n)?;
        (pauli_expectation(&d as &dyn Backend<C64>, &ops_of(obs, n))?
            / axis_operator_phase(obs))
        .re
    };
    println!("    budget     kept / total    certified     actual error   inside?");
    for budget in [0.0f64, 1e-4, 1e-3, 1e-2, 1e-1, 0.3] {
        let (kept, dropped, spent) = dyadic::prune_two_sided(&back, state, budget)?;
        let mut acc = C64::new(0.0, 0.0);
        for (key, coeff) in kept.terms() {
            acc += coeff * (pauli_expectation(state, &ops_of(key, n))?
                / axis_operator_phase(key));
        }
        let actual = (acc.re - truth).abs();
        println!(
            "    {budget:<9.0e}  {:5} / {:<5}    {spent:.3e}     {actual:.3e}      {}",
            back.len() - dropped,
            back.len(),
            if actual <= spent + 1e-11 { "yes" } else { "NO" }
        );
    }
    println!("\n  Read the 1e-1 row: most of the operator discarded, and the answer");
    println!("  still lands inside a bound that was computed before it was checked.");
    println!("  The one-directional bound cannot make that claim at any budget,");
    println!("  because it has to price every dropped term at its full weight.\n");

    println!("── the tree itself ──\n");
    println!("  level  seg   span   diamond   live cells   advance   terms");
    for l in 0..=depth {
        let nodes = c.level(l);
        let show = nodes.len().min(4);
        for (i, nd) in nodes.iter().take(show).enumerate() {
            let tag = if i == 0 { format!("{l:5}") } else { "     ".into() };
            println!(
                "  {tag}  {:3}   {:4}   {:5}q    {:9}   {:7}   {:5}",
                nd.index,
                nd.width(),
                nd.diamond.count_ones(),
                nd.live_cells,
                nd.advance,
                nd.terms
            );
        }
        if nodes.len() > show {
            println!("         ...  {} more", nodes.len() - show);
        }
    }
    println!("\n  A node's diamond is where the past has already arrived AND the");
    println!("  future can still see — computed per gate, so the children's live");
    println!("  cells sum to their parent's exactly. Outside it nothing that");
    println!("  happens can reach the answer, at any scale.");
    Ok(())
}
