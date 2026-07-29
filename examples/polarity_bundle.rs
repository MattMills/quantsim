//! Polarity as a fibered, re-orderable, journalled co-bundle:
//! entanglement held as an explicit, budgeted, auditable resource — and
//! the correction to what `examples/polarity_systems.rs` measured.
//!
//! Run with `cargo run --release --example polarity_bundle`.

use quantsim::bundle::*;
use quantsim::polarity::pairwise_signature;
use quantsim::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    // ── 1. The sign lives in a fiber ─────────────────────────────────
    println!("== the GHZ sign is fiber data, not base data ==");
    println!(
        "  {:>4} {:>8} {:>16} {:>12} {:>18} {:>14}",
        "n", "links", "same base?", "overlap", "2-body marginals", "fiber read"
    );
    for n in [3usize, 4, 6, 8] {
        let plus = ghz_bundle(n, false)?;
        let minus = ghz_bundle(n, true)?;
        let sp = plus.to_state()?;
        let sm = minus.to_state()?;
        let mut ov = C64::new(0.0, 0.0);
        sp.for_each_nonzero(&mut |i, a| ov += a.conj() * sm.amplitude(i));
        let marginals =
            pairwise_signature(sp.as_ref())?.max_deviation(&pairwise_signature(sm.as_ref())?);
        let read = plus.verify_against(sm.as_ref())?;
        println!(
            "  {:>4} {:>8} {:>16} {:>12.1e} {:>18.1e} {:>14}",
            n,
            plus.profile().links,
            plus.link_signature() == minus.link_signature(),
            ov.norm(),
            marginals,
            format!("spin0 = {}", read.spins[0])
        );
    }
    println!();
    println!("  Identical base, orthogonal states. Two-body marginals cannot tell them apart");
    println!("  — that measurement stands — but the bundle does not store marginals. It stores");
    println!("  a base and a fiber over each site, and the sign is in a fiber. Reading it back");
    println!("  off the state is exact: the minus state reads as a flipped fiber, deviation");
    println!("  ~1e-16 on every stabilizer generator.");

    // ── 2. Twist + order = chirality ─────────────────────────────────
    println!();
    println!("== re-ordering is not free: chirality ==");
    let mut bundle = PolarityBundle::new(5)?;
    for (a, b) in [(0u32, 1u32), (1, 2), (0, 3), (2, 4)] {
        bundle.link(a, b)?;
    }
    println!(
        "  {:>22} {:>11} {:>22}",
        "order", "chirality", "forward links by position"
    );
    for order in [
        vec![0u32, 1, 2, 3, 4],
        vec![4, 3, 2, 1, 0],
        vec![2, 0, 4, 1, 3],
        vec![1, 3, 0, 2, 4],
    ] {
        let mut b = PolarityBundle::new(5)?;
        for (x, y) in [(0u32, 1u32), (1, 2), (0, 3), (2, 4)] {
            b.link(x, y)?;
        }
        b.reorder(&order)?;
        println!(
            "  {:>22} {:>11} {:>22}",
            format!("{order:?}"),
            b.chirality(),
            format!("{:?}", b.forward_census())
        );
    }
    println!("  Swapping two LINKED sites flips the sign; unlinked ones are free. This is the");
    println!("  same sign a product of anticommuting polarity generators picks up, and the");
    println!("  tests check it against polarity::PolaritySystem rather than asserting it.");

    // ── 3. Entanglement with a budget ────────────────────────────────
    println!();
    println!("== entanglement as a budgeted resource ==");
    let mut big = scaling_bundle(20_000, 2)?;
    let start = big.profile();
    println!(
        "  start: {} live fibers, {} links, max degree {}, {} clusters, largest {}",
        start.live_sites, start.links, start.max_degree, start.components, start.largest_component
    );
    println!(
        "  {:>10} {:>8} {:>10} {:>11} {:>12} {:>12} {:>11}",
        "budget", "merges", "absorbed", "collapsed", "links after", "live fibers", "max weight"
    );
    for budget in [30_000usize, 10_000, 2_000, 300] {
        let r = big.coarse_grain_to_budget(budget)?;
        println!(
            "  {:>10} {:>8} {:>10} {:>11} {:>12} {:>12} {:>11}",
            budget,
            r.merges,
            r.absorbed_links,
            r.collapsed_links,
            r.links_after,
            big.profile().live_sites,
            r.max_weight
        );
    }
    println!("  Every link removed is on one of two lines: absorbed (became internal to a");
    println!("  super-fiber) or collapsed (parallel links merged). Nothing vanishes silently,");
    println!("  and the surviving fibers still account for all 20 000 original sites.");

    // ── 4. Interaction confined to commonality ───────────────────────
    println!();
    println!("== two bundles interact only where they have something in common ==");
    let mut a = PolarityBundle::new(6)?;
    for (x, y) in [(0u32, 1u32), (1, 2), (3, 4)] {
        a.link(x, y)?;
    }
    a.set_frame(4, Frame::X)?;
    let mut b = PolarityBundle::new(6)?;
    for (x, y) in [(1u32, 2u32), (2, 3), (4, 5)] {
        b.link(x, y)?;
    }
    b.set_frame(4, Frame::Y)?;
    b.set_spin(0, true)?;

    let common = a.commonality(&b);
    println!(
        "  shared sites {}, shared links {}, divergent {}, compatible fibers {}, incompatible {}",
        common.shared_sites,
        common.shared_links,
        common.divergent_links,
        common.compatible_fibers,
        common.incompatible_fibers
    );
    println!("  interactable sites: {:?}", common.interactable);
    let before = (a.linked(1, 2), a.linked(2, 3), a.linked(4, 5));
    a.interact(&b)?;
    println!(
        "  1–2 {} → {}   2–3 {} → {}   4–5 {} → {}  (site 4 carries X against Y: excluded)",
        before.0,
        a.linked(1, 2),
        before.1,
        a.linked(2, 3),
        before.2,
        a.linked(4, 5)
    );

    // ── 5. The journal ───────────────────────────────────────────────
    println!();
    println!("== the journal is the state ==");
    let mut j = PolarityBundle::new(6)?;
    j.link(0, 1)?;
    j.set_frame(2, Frame::X)?;
    j.link(2, 3)?;
    j.set_spin(1, true)?;
    j.coarse_grain(&[2, 3])?;
    println!("  {} journalled operations", j.journal().len());
    for steps in [1usize, 3, 5] {
        let past = j.rewind(steps)?;
        let p = past.profile();
        println!(
            "  after {steps} ops: {} links, {} live fibers, site1 spin {}",
            p.links,
            p.live_sites,
            past.inspect(1)?.fiber.spin
        );
    }

    // ── 6. Scale ─────────────────────────────────────────────────────
    println!();
    println!("== where no amplitude exists ==");
    println!(
        "  {:>10} {:>10} {:>10} {:>11} {:>11} {:>13} {:>13}",
        "sites", "links", "build ms", "profile ms", "reorder ms", "structure MB", "journal MB"
    );
    for sites in [10_000usize, 100_000, 1_000_000] {
        let t0 = Instant::now();
        let mut sb = scaling_bundle(sites, 2)?;
        let build = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let p = sb.profile();
        let profile = t1.elapsed().as_secs_f64() * 1e3;
        let reversed: Vec<u32> = (0..sites as u32).rev().collect();
        let t2 = Instant::now();
        sb.reorder(&reversed)?;
        let reorder = t2.elapsed().as_secs_f64() * 1e3;
        println!(
            "  {:>10} {:>10} {:>10.1} {:>11.1} {:>11.1} {:>13.1} {:>13.1}",
            sites,
            p.links,
            build,
            profile,
            reorder,
            (p.bytes - p.journal_bytes) as f64 / 1e6,
            p.journal_bytes as f64 / 1e6
        );
    }
    println!("  A 100 000-site state vector has 2^100000 amplitudes and does not exist. The");
    println!("  bundle is tens of megabytes and every operation stays linear. The journal is");
    println!("  the dominant term — checkpoint() gives it back when you do not need the");
    println!("  audit trail.");
    println!();
    println!("  Stated plainly: a bundle denotes a graph state dressed by local frames, which");
    println!("  is a known classically-tractable sector. Nothing here moves that boundary.");
    println!("  What is new is that entanglement in it is countable, orderable, coarse-");
    println!("  grainable and auditable — at a scale where the amplitude picture is absent.");
    Ok(())
}
