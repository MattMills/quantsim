//! The composite register, measured against the native stores on the
//! same space paths: where a product of regions wins (superposed sites
//! served by workspaces that are grown, used and released), where the
//! sparse store is the better choice (a stream that measures as it goes,
//! so the state stays basis-concentrated), and where neither structure
//! exists (a path that entangles everything, so the regions merge into
//! one store and the composite pays dense plus its bookkeeping).
//!
//! Run with `cargo run --release --example composite_register`.

use std::time::Instant;

use quantsim::backend::CompositeState;
use quantsim::prelude::*;
use quantsim::space::{cshift, fourier};

/// Grow a qutrit chain to `n` sites: each new site is entangled with its
/// predecessor (controlled shift, then Fourier), and — when `measure` —
/// the predecessor is measured. Returns (peak memory, seconds).
fn stream(state: &mut dyn Backend<C64>, n: usize, measure: bool) -> Result<(usize, f64)> {
    let f3 = fourier::<C64>(3)?;
    let cs = cshift::<C64>(3, 3);
    let mut rng = Prng::new(2026);
    let t0 = Instant::now();
    state.apply_sites(&f3, &[0])?;
    let mut peak = state.memory_bytes();
    for k in 1..n {
        state.transition(&Transition::Grow { at: k, radix: 3 }, false)?;
        state.apply_sites(&cs, &[k - 1, k])?;
        state.apply_sites(&f3, &[k])?;
        if measure {
            state.measure_site(k - 1, &mut rng)?;
        }
        peak = peak.max(state.memory_bytes());
    }
    Ok((peak, t0.elapsed().as_secs_f64()))
}

/// `n` qutrits in uniform superposition, served in turn by **one**
/// workspace qutrit grown at the start: copy qutrit `k` in, phase the
/// copy, uncompute — the phase kicks back (`|d⟩ ↦ ω^d |d⟩`) and the
/// workspace returns to `|0⟩` — then the next qutrit; the workspace is
/// released at the end. The register stays a product state of full
/// support throughout. Returns (peak memory, seconds).
fn workspace(state: &mut dyn Backend<C64>, n: usize) -> Result<(usize, f64)> {
    let f3 = fourier::<C64>(3)?;
    let cs = cshift::<C64>(3, 3);
    let z = quantsim::space::clock::<C64>(3)?;
    let t0 = Instant::now();
    for k in 0..n {
        state.apply_sites(&f3, &[k])?;
    }
    state.transition(&Transition::Grow { at: n, radix: 3 }, false)?;
    let mut peak = state.memory_bytes();
    for k in 0..n {
        state.apply_sites(&cs, &[k, n])?;
        state.apply_sites(&z, &[n])?;
        state.apply_sites(&cs, &[k, n])?;
        state.apply_sites(&cs, &[k, n])?;
        peak = peak.max(state.memory_bytes());
    }
    state.transition(&Transition::Release { site: n }, false)?;
    Ok((peak, t0.elapsed().as_secs_f64()))
}

fn kib(b: usize) -> String {
    if b < 1 << 20 {
        format!("{:.1} KiB", b as f64 / 1024.0)
    } else {
        format!("{:.1} MiB", b as f64 / (1 << 20) as f64)
    }
}

fn main() -> Result<()> {
    let one = Space::new(&[3])?;

    println!("══ 1. superposed qutrits served in turn by one grown workspace ══\n");
    println!("  qutrits  radix-dense            radix-sparse           composite              composite, no re-separation");
    for n in [6, 10, 12, 39] {
        let start = Space::new(&vec![3; n])?;
        let mut cells = Vec::new();
        for name in ["radix-dense", "radix-sparse", "composite", "no-split"] {
            let fits = n <= 12 || name == "composite";
            if name == "no-split" && n > 12 {
                cells.push(format!("{:<22}", "(merges to 3^(n+1))"));
                continue;
            }
            if !fits {
                cells.push(format!("{:<22}", "(3^n · 16 B)"));
                continue;
            }
            let mut state: Box<dyn Backend<C64>> = match name {
                "radix-dense" => Box::new(RadixDenseState::<C64>::new(start.clone())?),
                "radix-sparse" => Box::new(RadixSparseState::<C64>::new(start.clone())?),
                _ => {
                    let mut c = CompositeState::<C64>::new(start.clone())?;
                    c.set_auto_split(name == "composite");
                    Box::new(c)
                }
            };
            let (peak, secs) = workspace(state.as_mut(), n)?;
            cells.push(format!(
                "{:<22}",
                format!("{} {:.2} ms", kib(peak), secs * 1e3)
            ));
        }
        println!("  {n:>7}  {}", cells.join(" "));
    }

    println!("\n══ 2. a measured stream: grow, entangle with the predecessor, measure it ══\n");
    println!("  sites  virtual dim     radix-dense            radix-sparse           composite");
    for n in [6, 10, 13, 40] {
        let dim = 3u128.pow(n as u32);
        let mut cells = Vec::new();
        for name in ["radix-dense", "radix-sparse", "composite"] {
            if name == "radix-dense" && n > 13 {
                cells.push(format!(
                    "{:<22}",
                    format!("({:.1e} B needed)", dim as f64 * 16.0)
                ));
                continue;
            }
            let mut state: Box<dyn Backend<C64>> = match name {
                "radix-dense" => Box::new(RadixDenseState::<C64>::new(one.clone())?),
                "radix-sparse" => Box::new(RadixSparseState::<C64>::new(one.clone())?),
                _ => Box::new(CompositeState::<C64>::new(one.clone())?),
            };
            let (peak, secs) = stream(state.as_mut(), n, true)?;
            cells.push(format!(
                "{:<22}",
                format!("{} {:.2} ms", kib(peak), secs * 1e3)
            ));
        }
        println!("  {n:>5}  {dim:<14.3e}  {}", cells.join(" "));
    }
    let mut comp = CompositeState::<C64>::new(one.clone())?;
    stream(&mut comp, 40, true)?;
    let (peak_dim, peak_stored) = comp.peak();
    println!(
        "\n  composite at 40 sites: {} regions, peak region dim {peak_dim}, peak stored \
         nonzeros {peak_stored}, {} merges / {} separations ledgered",
        comp.region_count(),
        comp.events().iter().filter(|e| e.kind == "merge").count(),
        comp.events()
            .iter()
            .filter(|e| e.kind == "separate")
            .count(),
    );

    println!("\n══ 3. the same path without measurement: entanglement spans the register ══\n");
    println!(
        "  sites  radix-dense            radix-sparse           composite              regions"
    );
    for n in [6, 9, 12] {
        let mut cells = Vec::new();
        let mut regions = 0;
        for name in ["radix-dense", "radix-sparse", "composite"] {
            let mut state: Box<dyn Backend<C64>> = match name {
                "radix-dense" => Box::new(RadixDenseState::<C64>::new(one.clone())?),
                "radix-sparse" => Box::new(RadixSparseState::<C64>::new(one.clone())?),
                _ => Box::new(CompositeState::<C64>::new(one.clone())?),
            };
            let (peak, secs) = stream(state.as_mut(), n, false)?;
            if let Some(c) = state.as_any().downcast_ref::<CompositeState<C64>>() {
                regions = c.region_count();
            }
            cells.push(format!(
                "{:<22}",
                format!("{} {:.2} ms", kib(peak), secs * 1e3)
            ));
        }
        println!("  {n:>5}  {} {regions}", cells.join(" "));
    }
    Ok(())
}
