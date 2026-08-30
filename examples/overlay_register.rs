//! The overlay as a register: the layout as a constraint on the state.
//!
//! Run with `cargo run --release --example overlay_register`.

use quantsim::curve::{GridOrder, Order, Overlay};
use quantsim::overlay::OverlayRegister;
use quantsim::{GateMatrix, GateRegistry, Result, C64};

fn edges(side: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for y in 0..side {
        for x in 0..side {
            let s = y * side + x;
            if x + 1 < side {
                out.push((s, s + 1));
            }
            if y + 1 < side {
                out.push((s, s + side));
            }
        }
    }
    out
}

/// Row-local on the left half, 4×4-patch-local on the right: two
/// different localities in one circuit.
fn strips_and_patches(side: usize, q: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let half = side / 2;
    for y in 0..side {
        for bx in (0..half).step_by(q) {
            for x in bx..(bx + q).min(half) - 1 {
                out.push((y * side + x, y * side + x + 1));
            }
        }
    }
    for by in (0..side).step_by(q) {
        for bx in (half..side).step_by(q) {
            for y in by..by + q {
                for x in bx..bx + q {
                    let s = y * side + x;
                    if x + 1 < bx + q {
                        out.push((s, s + 1));
                    }
                    if y + 1 < by + q {
                        out.push((s, s + side));
                    }
                }
            }
        }
    }
    out
}

fn placement_total(side: usize, ov: &Overlay) -> Result<usize> {
    let r = OverlayRegister::<C64>::new(side, ov)?;
    let mut total = 0;
    for (a, b) in edges(side) {
        total += r.admits(&[a, b])?.width();
    }
    Ok(total)
}

fn main() -> Result<()> {
    let reg: GateRegistry<C64> = GateRegistry::standard();
    let gate = |n: &str| -> Result<GateMatrix<C64>> { reg.get(n).unwrap().matrix(&[]) };
    let (h, cz) = (gate("h")?, gate("cz")?);

    for line in [
        "A register whose regions must be blocks of an ordering.",
        "A gate that straddles two regions migrates into the smallest block that holds",
        "both, swallows whatever that block cuts into, and pays the difference.",
        "",
        "1. What a nearest-neighbour layer costs to *place*, summed over lattice edges",
        "   on a fresh register — the layout's own number, before anything commits.",
    ] {
        println!("{line}");
    }
    println!();
    println!(
        "  {:>4}  {:>12} {:>12} {:>12} {:>12}",
        "side", "row-major", "row-major D4", "hilbert", "hilbert D4"
    );
    for side in [4usize, 8, 16, 32, 64] {
        let one = |o| -> Result<usize> {
            placement_total(side, &Overlay::new(vec![GridOrder::new(side, o)?])?)
        };
        let fam = |o| -> Result<usize> { placement_total(side, &Overlay::family(side, o)?) };
        println!(
            "  {side:>4}  {:>12} {:>12} {:>12} {:>12}",
            one(Order::RowMajor)?,
            fam(Order::RowMajor)?,
            one(Order::Hilbert)?,
            fam(Order::Hilbert)?
        );
    }
    for line in [
        "",
        "   Exact, at every size:  one ordering  s²(s+1)·log₂s     |  3s²(s−1)",
        "                          the D4 family 2s²·log₂s = n·log₂n |  2s²(s−1)",
        "",
        "   So the curve's family saves exactly a third, always; the rows' family saves",
        "   (s+1)/2, without bound — and two members are the whole of it, an ordering",
        "   and its transpose. As one ordering the curve wins. As an overlay it loses.",
        "   The overlay's value is the disagreement between its members, and a",
        "   self-similar family agrees with itself.",
        "",
        "2. A run: a graph state, row-local on half the lattice and patch-local on the",
        "   other half. No single family serves both.",
    ] {
        println!("{line}");
    }

    for side in [8usize, 16] {
        let ops = strips_and_patches(side, 4);
        println!(
            "\n  side {side}, {} two-site gates, cap 16 sites",
            ops.len()
        );
        println!(
            "  {:>14}  {:>9} {:>5} {:>7} {:>11} {:>6}",
            "overlay", "completed", "peak", "padding", "peak memory", "elect"
        );
        for (name, ov) in [
            (
                "row-major",
                Overlay::new(vec![GridOrder::new(side, Order::RowMajor)?])?,
            ),
            (
                "hilbert",
                Overlay::new(vec![GridOrder::new(side, Order::Hilbert)?])?,
            ),
            ("row-major D4", Overlay::family(side, Order::RowMajor)?),
            ("hilbert D4", Overlay::family(side, Order::Hilbert)?),
            (
                "mixed",
                Overlay::families(side, &[Order::RowMajor, Order::Hilbert])?,
            ),
        ] {
            let mut r = OverlayRegister::<C64>::new(side, &ov)?;
            r.set_region_cap(16);
            for s in 0..r.sites() {
                r.apply(&h, &[s])?;
            }
            let done = ops
                .iter()
                .take_while(|&&(a, b)| r.apply(&cz, &[a, b]).is_ok())
                .count();
            let l = r.ledger();
            println!(
                "  {name:>14}  {:>4}/{:<4} {:>5} {:>7} {:>11} {:>6}{}",
                done,
                ops.len(),
                l.peak_width(),
                l.total_padding(),
                l.peak_memory(),
                l.elections(),
                if done == ops.len() { "" } else { "  refused" }
            );
        }
    }

    for line in [
        "",
        "   The mixed overlay completes what the rows refuse and holds less than either",
        "   family — because it contains a family that suits each part of the circuit.",
        "",
        "3. Padding is borrowed, not always spent. A graph-state layer applied twice is",
        "   the identity; the register has to notice and give the lattice back.",
    ] {
        println!("{line}");
    }
    let side = 8;
    let mut r = OverlayRegister::<C64>::mixed(side, &[Order::RowMajor, Order::Hilbert])?;
    r.set_region_cap(16);
    for s in 0..r.sites() {
        r.apply(&h, &[s])?;
    }
    let ops = strips_and_patches(side, 4);
    for &(a, b) in &ops {
        r.apply(&cz, &[a, b])?;
    }
    println!(
        "\n  after the layer:   {} regions, widest {}, holding {} amplitudes",
        r.region_count(),
        r.widest(),
        r.memory_amplitudes()
    );
    for &(a, b) in ops.iter().rev() {
        r.apply(&cz, &[a, b])?;
    }
    println!(
        "  after undoing it:  {} regions, widest {}, holding {} amplitudes",
        r.region_count(),
        r.widest(),
        r.memory_amplitudes()
    );
    println!(
        "  the ledger keeps the history: peak {} sites, {} migrations, {} splits, {} sites reclaimed",
        r.ledger().peak_width(),
        r.ledger().merge_count(),
        r.ledger().splits(),
        r.ledger().reclaimed()
    );
    println!(
        "\n  A dense register of {} qubits would be 2^{} amplitudes and does not exist.",
        r.sites(),
        r.sites()
    );
    Ok(())
}
