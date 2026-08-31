//! What decides whether a block fits: the machine, measured — not a
//! width constant in this module.
//!
//! Every test here configures the process-global resource guard, so
//! they serialize on one mutex; other test binaries are separate
//! processes and unaffected. The point they exist to make is that the
//! *demand* a layout puts on the register is a property of the layout
//! and the budget is a property of the machine, and the two are
//! independent — raising the budget changes the outcome and never the
//! number.

use std::sync::{Mutex, MutexGuard, OnceLock};

use quantsim::curve::{GridOrder, Order, Overlay};
use quantsim::overlay::{OverlayRegister, MAX_REGION_SITES};
use quantsim::{guard, Error, GateMatrix, GateRegistry, C64};

struct GuardLock(#[allow(dead_code)] MutexGuard<'static, ()>);

fn lock() -> GuardLock {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let g = match LOCK.get_or_init(|| Mutex::new(())).lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard::set_memory_limit(None);
    GuardLock(g)
}

impl Drop for GuardLock {
    fn drop(&mut self) {
        guard::set_memory_limit(None);
    }
}

fn gates() -> (GateMatrix<C64>, GateMatrix<C64>, GateMatrix<C64>) {
    let reg: GateRegistry<C64> = GateRegistry::standard();
    let m = |n: &str| reg.get(n).unwrap().matrix(&[]).unwrap();
    (m("h"), m("cz"), m("cx"))
}

fn patch_edges(side: usize, q: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for by in (0..side).step_by(q) {
        for bx in (0..side).step_by(q) {
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

#[test]
fn a_block_the_machine_cannot_hold_is_refused_by_measurement() {
    let _lock = lock();
    let side = 8;
    let (h, _, cx) = gates();
    let mut row = OverlayRegister::<C64>::single(side, Order::RowMajor).unwrap();
    row.apply(&h, &[0]).unwrap();

    // One vertical edge forces a sixteen-site block: 65536 amplitudes,
    // one mebibyte of `C64`. Under a 64 KiB budget that cannot be
    // admitted, and the error carries both sides of the measurement.
    guard::set_memory_limit(Some(64 << 10));
    match row.apply(&cx, &[0, side]).unwrap_err() {
        Error::OutOfMemory {
            requested,
            available,
            ref what,
        } => {
            assert_eq!(requested, 65536 * std::mem::size_of::<C64>());
            assert_eq!(available, 64 << 10);
            assert!(what.contains("16 sites"), "{what}");
        }
        other => panic!("expected a measured refusal, got {other}"),
    }

    // A refused gate mutates nothing: the partition is untouched, no
    // migration was ledgered, and the register still reports the block
    // it wanted.
    assert_eq!(row.region_count(), side * side - 1 + 1);
    assert_eq!(row.ledger().merge_count(), 0);
    assert_eq!(row.projected(&[0, side]).unwrap().unwrap().width(), 16);

    // Raise the budget and the same gate goes through. The layout's
    // demand did not change — only what the machine could hold.
    guard::set_memory_limit(Some(4 << 20));
    row.apply(&cx, &[0, side]).unwrap();
    assert_eq!(row.widest(), 16);
    assert_eq!(row.ledger().merges()[0].width(), 16);
}

#[test]
fn the_demand_is_the_layouts_and_the_budget_is_the_machines() {
    // The claim this file exists for. A register run under two very
    // different budgets reports the *same* demand at the point it
    // stops; only whether it stops differs.
    let _lock = lock();
    let side = 8;
    let (h, cz, _) = gates();
    let ops = patch_edges(side, 4);
    let mut seen = Vec::new();
    for budget in [1usize << 20, 1 << 24, 1 << 28] {
        guard::set_memory_limit(Some(budget));
        let mut r = OverlayRegister::<C64>::single(side, Order::RowMajor).unwrap();
        for s in 0..r.sites() {
            r.apply(&h, &[s]).unwrap();
        }
        let mut stopped = None;
        for &(a, b) in &ops {
            if r.apply(&cz, &[a, b]).is_err() {
                stopped = Some(r.projected(&[a, b]).unwrap().unwrap().width());
                break;
            }
        }
        seen.push(stopped);
    }
    // Every budget stops on a demand it cannot meet, and the demand is
    // the same number each time across a 256× range of budgets —
    // 32 sites, 2^32 amplitudes, 64 GiB. The layout's cost is not a
    // function of the machine.
    assert_eq!(seen, vec![Some(32); 3], "{seen:?}");
    assert_eq!(
        (1u128 << 32) * std::mem::size_of::<C64>() as u128,
        68_719_476_736
    );
}

#[test]
fn the_curve_runs_the_same_circuit_the_rows_cannot_at_any_real_budget() {
    // 256 MiB: far past what the curve needs and far short of what the
    // rows demand. This is not a policy in the module — it is the
    // machine, and the module has no say in it.
    let _lock = lock();
    let side = 8;
    let (h, cz, _) = gates();
    let ops = patch_edges(side, 4);
    guard::set_memory_limit(Some(256 << 20));
    for (order, expect_done) in [
        (Order::Hilbert, ops.len()),
        (Order::RowMajor, 8),
        (Order::Snake, 8),
    ] {
        let mut r = OverlayRegister::<C64>::single(side, order).unwrap();
        for s in 0..r.sites() {
            r.apply(&h, &[s]).unwrap();
        }
        let done = ops
            .iter()
            .take_while(|&&(a, b)| r.apply(&cz, &[a, b]).is_ok())
            .count();
        assert_eq!(done, expect_done, "{order:?}");
        if done == ops.len() {
            assert_eq!(r.peak_memory_bytes(), 4 << 20, "the curve: four mebibytes");
        }
    }
}

#[test]
fn the_only_width_bound_in_the_module_is_structural() {
    // 256 sites cannot be a local basis index, so it is refused as a
    // structural impossibility rather than a capacity judgement — the
    // same bound `FACTOR_MAX_QUBITS` puts on a factor. Note the
    // unlimited budget: this refusal is not the guard's.
    let _lock = lock();
    guard::set_memory_limit(Some(usize::MAX));
    let side = 16;
    let (h, cz, _) = gates();
    let mut reg = OverlayRegister::<C64>::single(side, Order::Hilbert).unwrap();
    reg.apply(&h, &[0]).unwrap();
    reg.apply(&h, &[side * side - 1]).unwrap();
    assert_eq!(
        reg.projected(&[0, side * side - 1])
            .unwrap()
            .unwrap()
            .width(),
        256
    );
    assert!(matches!(
        reg.apply(&cz, &[0, side * side - 1]),
        Err(Error::TooManyQubits {
            requested: 256,
            max: MAX_REGION_SITES
        })
    ));
}

#[test]
fn the_mixed_overlay_completes_under_a_budget_that_stops_both_families() {
    // The heterogeneity result, stated where the budget is explicit:
    // 8 MiB runs the mixed overlay and stops the row-major family.
    let _lock = lock();
    let side = 8;
    let (h, cz, _) = gates();
    let ops = patch_edges(side, 4);
    guard::set_memory_limit(Some(8 << 20));
    let run = |ov: &Overlay| {
        let mut r = OverlayRegister::<C64>::new(side, ov).unwrap();
        for s in 0..r.sites() {
            r.apply(&h, &[s]).unwrap();
        }
        ops.iter()
            .take_while(|&&(a, b)| r.apply(&cz, &[a, b]).is_ok())
            .count()
    };
    assert_eq!(
        run(&Overlay::new(vec![GridOrder::new(side, Order::RowMajor).unwrap()]).unwrap()),
        8
    );
    assert_eq!(run(&Overlay::family(side, Order::RowMajor).unwrap()), 8);
    assert_eq!(
        run(&Overlay::families(side, &[Order::RowMajor, Order::Hilbert]).unwrap()),
        ops.len()
    );
}
