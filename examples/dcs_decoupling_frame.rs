//! Is the DCS instance's single anticommutation component an artefact of
//! the *frame*?
//!
//! `coupling::decoupling_frame` builds a Clifford `V` by symplectic
//! Gram–Schmidt over the rotation axes, ordered by anticommutation
//! component, and then *measures* the qubit-support partition of the
//! framed axes. If the lab-frame axes really carry `k` independent
//! problems, the frame is supposed to put them on `k` disjoint qubit
//! blocks however much they overlap in the lab.
//!
//! This runnable feeds it the doped-Clifford-sampling family
//! (`Dcs::scaled(n)`) at every width where `(u64, u64)` Pauli keys fit,
//! and reports the partition before and after. Three doping profiles:
//! the experiment's `Uniform`, and the two profiles `upembed` already
//! showed are separable in the lab frame, kept as positive controls so a
//! null result cannot be blamed on the code path.

use quantsim::coupling::{coupling_of, decoupling_frame, propagate_engineered, support_blocks};
use quantsim::dcs::{self, Dcs, Doping};
use quantsim::heisenberg::{commutes, PauliKey, Rotation};
use quantsim::pathsum::Mask;
use std::f64::consts::FRAC_PI_4;
use std::time::Instant;

/// `dcs::rotation_axes` speaks unbounded `Mask`; `coupling` speaks
/// `(u64, u64)`. Only widths under 64 can make the trip.
fn key_of(a: &(Mask, Mask)) -> PauliKey {
    let f = |m: &Mask| {
        m.iter().fold(0u64, |acc, i| {
            assert!(i < 64, "axis touches qubit {i}: past the PauliKey ceiling");
            acc | 1u64 << i
        })
    };
    (f(&a.0), f(&a.1))
}

/// Support-overlap components of a set of qubit masks — the partition
/// `FactoredPauliSum` discovers, and the one `DecouplingFrame::blocks`
/// reports. (`coupling`'s copy is private; this is the same rule.)
fn overlap_blocks(masks: impl IntoIterator<Item = u64>) -> Vec<u64> {
    let mut blocks: Vec<u64> = Vec::new();
    for m in masks {
        if m == 0 {
            continue;
        }
        let hit: Vec<usize> = (0..blocks.len()).filter(|&i| blocks[i] & m != 0).collect();
        let mut merged = m;
        for &i in hit.iter().rev() {
            merged |= blocks.remove(i);
        }
        blocks.push(merged);
    }
    blocks
}

/// Sizes of the anticommutation graph's components, descending.
fn anti_components(keys: &[PauliKey]) -> Vec<usize> {
    let m = keys.len();
    let mut parent: Vec<usize> = (0..m).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for i in 0..m {
        for j in (i + 1)..m {
            if !commutes(keys[i], keys[j]) {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                parent[a] = b;
            }
        }
    }
    let mut sizes = std::collections::HashMap::new();
    for i in 0..m {
        let r = find(&mut parent, i);
        *sizes.entry(r).or_insert(0usize) += 1;
    }
    let mut out: Vec<usize> = sizes.into_values().collect();
    out.sort_unstable_by(|a, b| b.cmp(a));
    out
}

fn widest(blocks: &[u64]) -> usize {
    blocks
        .iter()
        .map(|m| m.count_ones() as usize)
        .max()
        .unwrap_or(0)
}

struct Row {
    n: usize,
    axes: usize,
    distinct: usize,
    anti_before: usize,
    anti_largest: usize,
    lab_blocks: usize,
    lab_widest: usize,
    frame_steps: usize,
    /// Support-overlap blocks of the **framed axes alone** — the honest
    /// comparison against `lab_blocks`, which is also axes-only.
    eng_blocks: usize,
    eng_widest: usize,
    /// `DecouplingFrame::blocks()`, which additionally seeds one
    /// singleton per site of the framed observable.
    eng_blocks_obs: usize,
    anti_after: usize,
    live: usize,
    inert: usize,
    secs: f64,
}

fn measure(n: usize, doping: Doping) -> Option<Row> {
    let d = Dcs::scaled(n).with_doping(doping);
    let circuit = d.circuit();
    let axes = dcs::rotation_axes(&circuit).expect("Clifford+T fragment");
    let keys: Vec<PauliKey> = axes.iter().map(key_of).collect();
    let rots: Vec<Rotation> = keys
        .iter()
        .map(|&axis| Rotation {
            theta: FRAC_PI_4,
            axis,
        })
        .collect();
    // The sampling readout: Z on every qubit. A single-site observable
    // would make `support_blocks` trivially 1, so the comparison is run
    // on the widest thing the experiment asks for.
    let obs: PauliKey = (0, if n >= 64 { u64::MAX } else { (1u64 << n) - 1 });

    let mut distinct: Vec<PauliKey> = keys.clone();
    distinct.sort_unstable();
    distinct.dedup();

    let before = anti_components(&keys);
    let lab = overlap_blocks(keys.iter().map(|k| k.0 | k.1));

    let coupling = coupling_of(obs, &rots);
    let t0 = Instant::now();
    let frame = match decoupling_frame(obs, &rots, n) {
        Ok(f) => f,
        Err(e) => {
            println!("n = {n:>3}: decoupling_frame refused: {e}");
            return None;
        }
    };
    let secs = t0.elapsed().as_secs_f64();

    let framed = frame.rewrite(&rots);
    let framed_keys: Vec<PauliKey> = framed.iter().map(|r| r.axis).collect();
    let after = anti_components(&framed_keys);
    assert_eq!(
        before.len(),
        after.len(),
        "the anticommutation partition is a Clifford invariant"
    );
    let eng = overlap_blocks(framed_keys.iter().map(|k| k.0 | k.1));
    assert!(
        eng.len() <= after.len(),
        "support blocks can never be finer than anticommutation components"
    );

    Some(Row {
        n,
        axes: keys.len(),
        distinct: distinct.len(),
        anti_before: before.len(),
        anti_largest: before.first().copied().unwrap_or(0),
        lab_blocks: lab.len(),
        lab_widest: widest(&lab),
        frame_steps: frame.len(),
        eng_blocks: eng.len(),
        eng_widest: widest(&eng),
        eng_blocks_obs: frame.blocks().len(),
        anti_after: after.len(),
        live: coupling.live_components(),
        inert: coupling.inert_rotations(),
        secs,
    })
}

fn header(title: &str) {
    println!();
    println!("{title}");
    println!(
        "   n  axes  dist | anti-comp  largest | lab blocks widest | frame | eng blocks widest \
         (+obs) | live inert |   secs"
    );
    println!("  {}", "-".repeat(108));
}

fn emit(r: &Row) {
    println!(
        "  {:>2}  {:>4}  {:>4} | {:>9}  {:>7} | {:>10} {:>6} | {:>5} | {:>10} {:>6} {:>6} | \
         {:>4} {:>5} | {:>6.2}",
        r.n,
        r.axes,
        r.distinct,
        r.anti_before,
        r.anti_largest,
        r.lab_blocks,
        r.lab_widest,
        r.frame_steps,
        r.eng_blocks,
        r.eng_widest,
        r.eng_blocks_obs,
        r.live,
        r.inert,
        r.secs
    );
    debug_assert_eq!(r.anti_before, r.anti_after);
}

/// Exhaust-ish search: many *random* Clifford frames, to test the
/// hypothesis "some other frame decouples it" by measurement rather than
/// by the invariance argument alone.
fn random_frame_search(n: usize, trials: usize, seed: u64) {
    use quantsim::backend::CliffordStep;
    use quantsim::coupling::DecouplingFrame;
    use quantsim::rng::Prng;

    let d = Dcs::scaled(n);
    let axes = dcs::rotation_axes(&d.circuit()).expect("Clifford+T fragment");
    let rots: Vec<Rotation> = axes
        .iter()
        .map(|a| Rotation {
            theta: FRAC_PI_4,
            axis: key_of(a),
        })
        .collect();
    let keys: Vec<PauliKey> = rots.iter().map(|r| r.axis).collect();
    let anti = anti_components(&keys).len();

    let mut rng = Prng::new(seed);
    let mut best = 0usize;
    let mut best_widest = n;
    let mut anti_stable = true;
    for _ in 0..trials {
        // A random Clifford word: 6n elementary steps drawn from
        // {H, S, CX}, which mixes far past the point where the axes are
        // all-to-all.
        let steps: Vec<CliffordStep> = (0..6 * n)
            .map(|_| {
                let q = (rng.next_u64() % n as u64) as usize;
                match rng.next_u64() % 3 {
                    0 => CliffordStep::H(q),
                    1 => CliffordStep::S(q),
                    _ => {
                        let mut p = (rng.next_u64() % n as u64) as usize;
                        if p == q {
                            p = (p + 1) % n;
                        }
                        CliffordStep::Cx(q, p)
                    }
                }
            })
            .collect();
        let frame = DecouplingFrame::from_steps(steps, n);
        let framed: Vec<PauliKey> = frame.rewrite(&rots).iter().map(|r| r.axis).collect();
        anti_stable &= anti_components(&framed).len() == anti;
        let blocks = overlap_blocks(framed.iter().map(|k| k.0 | k.1));
        if blocks.len() > best {
            best = blocks.len();
            best_widest = widest(&blocks);
        }
    }
    println!(
        "  n = {n:>2}: {trials} random Clifford frames → best support partition {best} block(s), \
         widest {best_widest};  anticommutation components {anti} in every frame: {anti_stable}"
    );
}

fn main() {
    println!("DOES A CLIFFORD FRAME DECOUPLE THE DCS INSTANCE?");
    println!(
        "  anti-comp = anticommutation components of the axes (Clifford invariant)\n  \
         lab blocks = support-overlap components of the axes in the natural frame\n  \
         eng blocks = the same rule applied to V†-conjugated axes, i.e. DecouplingFrame::blocks\n  \
         frame      = elementary Cliffords in V; observable = Z on every qubit"
    );

    let widths = [8usize, 12, 16, 20, 24, 32, 40, 48, 56, 60, 64];

    header("EXPERIMENT'S DOPING (Doping::Uniform, the family Dcs::scaled(n))");
    let mut uniform = Vec::new();
    for &n in &widths {
        if let Some(r) = measure(n, Doping::Uniform) {
            emit(&r);
            uniform.push(r);
        }
    }

    header("POSITIVE CONTROL A: Doping::Early { layers: 2 }");
    for &n in &widths {
        if let Some(r) = measure(n, Doping::Early { layers: 2 }) {
            emit(&r);
        }
    }

    header("POSITIVE CONTROL B: Doping::Banded { width: 2, gap: 6, layers: 2, late: false }");
    for &n in &widths {
        if let Some(r) = measure(
            n,
            Doping::Banded {
                width: 2,
                gap: 6,
                layers: 2,
                late: false,
            },
        ) {
            emit(&r);
        }
    }

    println!();
    println!("TREND (Uniform): engineered blocks against width");
    for r in &uniform {
        println!(
            "  n = {:>2}: {:>3} axes → {} anticommutation component(s) = ceiling → {} engineered \
             block(s), widest {} of {} qubits",
            r.n, r.axes, r.anti_before, r.eng_blocks, r.eng_widest, r.n
        );
    }

    println!();
    println!("RANDOM-FRAME SEARCH (Uniform): can *any* frame beat the constructed one?");
    for n in [16usize, 24, 32, 48] {
        random_frame_search(n, 200, 0xC0FFEE ^ n as u64);
    }

    // How much magic would have to come out before the ceiling rises
    // above 1? The Clifford skeleton is byte-identical along this sweep
    // — only the T count moves.
    println!();
    println!("DOPING SWEEP: components against T count at fixed width (Uniform sites)");
    for n in [32usize, 48, 64] {
        let full = Dcs::scaled(n).t_gates;
        let mut last_multi = None;
        let mut line = String::new();
        for t in 1..=full {
            let d = Dcs::scaled(n).with_t(t);
            let axes = dcs::rotation_axes(&d.circuit()).expect("Clifford+T fragment");
            let keys: Vec<PauliKey> = axes.iter().map(key_of).collect();
            let c = anti_components(&keys);
            if c.len() > 1 {
                last_multi = Some((t, c.len()));
            }
            if t % (full / 8).max(1) == 0 || t == full {
                line.push_str(&format!("  t={t}:{} ", c.len()));
            }
        }
        println!("  n = {n:>2} (experiment's rate → t = {full}):{line}");
        match last_multi {
            Some((t, c)) => println!(
                "        last t with more than one component: t = {t} ({c} components) — \
                 {:.0}% of the experiment's doping",
                100.0 * t as f64 / full as f64
            ),
            None => println!("        one component at every t ≥ 1"),
        }
    }

    // The ceiling is a Clifford invariant, so it can be read at the
    // experiment's own width even though the frame machinery cannot run
    // there: `dcs::anticommutation_components` speaks unbounded `Mask`.
    println!();
    println!("THE EXPERIMENT ITSELF (n = 70 — past the (u64,u64) ceiling)");
    let t0 = Instant::now();
    let exp_axes = dcs::rotation_axes(&Dcs::experiment().circuit()).expect("Clifford+T fragment");
    let comps = dcs::anticommutation_components(&exp_axes);
    println!(
        "  {} axes → {} anticommutation component(s), largest {}   ({:?})",
        exp_axes.len(),
        comps.len(),
        comps.first().copied().unwrap_or(0),
        t0.elapsed()
    );
    let dummy = vec![Rotation::rz(0, FRAC_PI_4)];
    match decoupling_frame((0, 1), &dummy, 70) {
        Ok(_) => println!("  decoupling_frame accepted 70 qubits"),
        Err(e) => println!("  decoupling_frame(_, _, 70): {e}"),
    }

    // The whole pipeline, where the factored walk still fits: how many
    // blocks the walk ends with and whether V†|0…0⟩ survives as a
    // product across them. (The axes carry no signs — `rotation_axes`
    // drops them — so `value` is the expectation of a sign-scrambled
    // sibling of the circuit, not the experiment's own amplitude. The
    // block structure and separability are sign-independent.)
    println!();
    println!("END-TO-END propagate_engineered (small n only: the walk is 2^t terms)");
    println!("   n  t | support blocks  engineered blocks | peak stored  peak flat | separable input");
    println!("  {}", "-".repeat(88));
    for n in [8usize, 10, 12, 14] {
        let d = Dcs::scaled(n);
        let circuit = d.circuit();
        let axes = dcs::rotation_axes(&circuit).expect("Clifford+T fragment");
        let rots: Vec<Rotation> = axes
            .iter()
            .map(|a| Rotation {
                theta: FRAC_PI_4,
                axis: key_of(a),
            })
            .collect();
        let obs: PauliKey = (0, (1u64 << n) - 1);
        let sb = support_blocks(obs, &rots);
        match propagate_engineered(obs, &rots, n) {
            Ok(rep) => println!(
                "  {:>2} {:>2} | {:>14}  {:>17} | {:>11}  {:>9} | {}",
                n,
                rots.len(),
                sb,
                rep.engineered_blocks,
                rep.peak_stored,
                rep.peak_flat,
                rep.separable_input
            ),
            Err(e) => println!("  {:>2} {:>2} | refused: {e}", n, rots.len()),
        }
    }
}
