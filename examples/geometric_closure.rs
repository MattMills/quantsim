//! A journalled state representation whose errors are found by
//! geometric closure, retrodictively.
//!
//! The loops a geometry happens to have are checks nobody declared.
//! Breaking one localizes in space; breaking it at some moment
//! localizes in time; and the history is read backwards to find when.
//!
//! Run with `cargo run --release --example geometric_closure`.

use quantsim::bundle::PolarityBundle;
use quantsim::closure::*;
use quantsim::prelude::*;
use std::time::Instant;

fn grid(w: usize, h: usize) -> Result<PolarityBundle> {
    let mut b = PolarityBundle::new(w * h)?;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as u32;
            if x + 1 < w {
                b.link(i, i + 1)?;
            }
            if y + 1 < h {
                b.link(i, i + w as u32)?;
            }
        }
    }
    Ok(b)
}

fn ring(n: usize) -> Result<PolarityBundle> {
    let mut b = PolarityBundle::new(n)?;
    for i in 0..n as u32 {
        b.link(i, (i + 1) % n as u32)?;
    }
    Ok(b)
}

fn main() -> Result<()> {
    // ── 1. Checks nobody declared ────────────────────────────────────
    println!("== what the geometry supplies without being asked ==");
    println!(
        "  {:>10} {:>7} {:>7} {:>7} {:>9} {:>14}",
        "geometry", "sites", "links", "loops", "bridges", "total looplen"
    );
    let mut tree = PolarityBundle::new(5)?;
    for (a, b) in [(0u32, 1u32), (1, 2), (1, 3), (3, 4)] {
        tree.link(a, b)?;
    }
    for (name, bundle) in [
        ("tree-5", tree),
        ("ring-6", ring(6)?),
        ("grid 4×4", grid(4, 4)?),
        ("grid 6×6", grid(6, 6)?),
    ] {
        let basis = closure_basis(&bundle);
        println!(
            "  {:>10} {:>7} {:>7} {:>7} {:>9} {:>14}",
            name,
            bundle.sites(),
            bundle.profile().links,
            basis.rank,
            basis.bridges.len(),
            basis.total_length
        );
    }
    println!("  rank = |E| − |V| + components, exactly. No code was designed and no");
    println!("  redundancy was added: these checks are what having loops already means.");
    println!("  A tree has none, so a tree can see nothing — reported, not hidden.");

    // ── 2. A break is geometric ──────────────────────────────────────
    println!();
    println!("== a break is exactly the loops that pass through ==");
    let bundle = grid(4, 4)?;
    let basis = closure_basis(&bundle);
    for site in [0u32, 5, 10] {
        let mut history = ClosureHistory::new(bundle.clone())?;
        history.perturb_spin(site)?;
        let broken = history.break_report()?;
        let through: Vec<usize> = (0..basis.rank)
            .filter(|&i| basis.loops[i].passes_through(site))
            .collect();
        println!(
            "  site {site:>2}: loops broken {:?} == loops through it {:?}",
            broken.spin, through
        );
    }

    // ── 3. Localization, and its honest limits ───────────────────────
    println!();
    println!("== where the geometry can name the site, and where it declines ==");
    for (name, bundle) in [
        ("ring-6", ring(6)?),
        ("grid 4×4", grid(4, 4)?),
        ("grid 6×6", grid(6, 6)?),
    ] {
        let (mut unique, mut ambiguous, mut invisible) = (0, 0, 0);
        for site in 0..bundle.sites() as u32 {
            let mut history = ClosureHistory::new(bundle.clone())?;
            history.perturb_spin(site)?;
            let l = history.localize(&history.break_report()?);
            if l.undetectable {
                invisible += 1;
            } else if l.unique {
                unique += 1;
            } else {
                ambiguous += 1;
            }
        }
        println!(
            "  {name:>10}: {unique:>3} named exactly, {ambiguous:>3} ambiguous, \
             {invisible:>3} invisible  (of {})",
            bundle.sites()
        );
    }
    println!("  One loop can tell that something moved but not which of its sites moved.");
    println!("  The ambiguity is reported with every candidate listed — the true site is");
    println!("  always among them — and correction refuses rather than picking one.");

    // ── 4. Retrodiction ──────────────────────────────────────────────
    println!();
    println!("== reading the history backwards for the moment closure broke ==");
    let mut history = ClosureHistory::new(grid(6, 6)?)?;
    history.stamp()?;
    for step in 0..40u32 {
        history.set_spin(step % 36, step % 3 == 0)?;
        if step % 4 == 3 {
            history.stamp()?;
        }
    }
    println!(
        "  {} legitimate steps, {} stamps, {} bits of stamp in total",
        history.step(),
        history.stamps(),
        history.stamp_bits()
    );
    history.perturb_spin(14)?;
    let injected_at = history.step();
    println!("  something happens at step {injected_at} on site 14 — not recorded as anything");
    for step in 0..40u32 {
        history.set_spin((step * 7) % 36, step % 2 == 0)?;
        if step % 4 == 3 {
            history.stamp()?;
        }
    }
    history.stamp()?;

    let retro = history.retrodict()?;
    println!(
        "  retrodicted: closure last held at step {:?}, was already broken by step {:?}",
        retro.last_intact_step, retro.step
    );
    println!(
        "  the interval brackets it: {}",
        retro.last_intact_step.is_some_and(|t| t < injected_at)
            && retro.step.is_some_and(|t| t >= injected_at)
    );
    println!(
        "  and the same act names the site: {:?} (unique = {}), from {} broken loops \
         against {} intact",
        retro.localization.sites,
        retro.localization.unique,
        retro.localization.broken.loops().len(),
        retro.localization.intact_loops
    );
    println!(
        "  {} closure comparisons, where a scan would have taken {}",
        retro.comparisons, retro.linear_comparisons
    );
    println!("  No state was replayed. Nothing was ever recorded as an error.");

    let report = history.correct(&retro)?;
    println!(
        "  correction at site {}: {} broken loops → {}, closure restored = {}",
        report.site, report.broken_before, report.broken_after, report.restored
    );

    // ── 5. The state itself comes back ───────────────────────────────
    println!();
    println!("== correction restores the denoted state, not merely the bookkeeping ==");
    let mut clean = grid(4, 4)?;
    clean.set_spin(6, true)?;
    clean.set_spin(9, true)?;
    let target = (0..16u32)
        .find(|&site| {
            let mut probe = ClosureHistory::new(clean.clone()).unwrap();
            probe.perturb_spin(site).unwrap();
            let l = probe.localize(&probe.break_report().unwrap());
            l.unique && l.sites[0] == site
        })
        .expect("some site is named exactly");
    let mut h = ClosureHistory::new(clean.clone())?;
    h.stamp()?;
    h.perturb_spin(target)?;
    h.stamp()?;
    let r = h.retrodict()?;
    let rep = h.correct(&r)?;
    let repaired = h.into_bundle();
    let before = clean.to_state()?;
    let after = repaired.to_state()?;
    println!(
        "  perturbed site {target}, corrected site {}, state deviation {:.2e}",
        rep.site,
        max_amplitude_deviation(before.as_ref(), after.as_ref())
    );

    // ── 6. The second channel ────────────────────────────────────────
    println!();
    println!("== the other thing that can fail to come back: chirality ==");
    let mut h2 = ClosureHistory::new(grid(4, 4)?)?;
    println!("  intact at the start: {}", h2.break_report()?.intact());
    for position in 0..4 {
        h2.swap_order(position)?;
    }
    println!(
        "  after four deliberate re-orderings: {}",
        h2.break_report()?.intact()
    );
    h2.perturb_order(5)?;
    let broken = h2.break_report()?;
    let l = h2.localize(&broken);
    println!(
        "  after one unrecorded re-ordering: spin breaks {:?}, chirality breaks {:?}",
        broken.spin, broken.chirality
    );
    println!(
        "  which names a LINK, not a site: {:?} (unique = {})",
        l.links, l.unique
    );
    println!("  and correction declines: the link is determined, the transposition is not.");

    // ── 7. Scale ─────────────────────────────────────────────────────
    println!();
    println!("== the cost of being able to retrodict ==");
    println!(
        "  {:>8} {:>8} {:>8} {:>14} {:>10} {:>10} {:>16}",
        "sites", "links", "loops", "total looplen", "basis ms", "stamp ms", "bits per stamp"
    );
    for (w, h) in [(30usize, 30usize), (60, 60), (120, 120)] {
        let t0 = Instant::now();
        let mut hist = ClosureHistory::new(grid(w, h)?)?;
        let basis_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        hist.stamp()?;
        let stamp_ms = t1.elapsed().as_secs_f64() * 1e3;
        println!(
            "  {:>8} {:>8} {:>8} {:>14} {:>10.1} {:>10.1} {:>16}",
            w * h,
            hist.bundle().profile().links,
            hist.basis().rank,
            hist.basis().total_length,
            basis_ms,
            stamp_ms,
            hist.stamp_bits()
        );
    }
    println!("  A 120×120 geometry retrodicts from 28 322 bits — about 3.5 kB a stamp — and");
    println!("  the number of stamps is the caller's choice. Stated plainly: the cost of");
    println!("  EVALUATING closure is the total loop length, not the loop count, and a");
    println!("  spanning-forest basis does not keep that small. A short-cycle basis would;");
    println!("  the number is printed above rather than left out of the accounting.");
    Ok(())
}
