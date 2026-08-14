//! Width benchmarks: how run time scales as qubits are added, per
//! representation. The companion memory story — the actual point of width
//! scaling — is asserted in `tests/memory_scaling.rs` and printed by
//! `examples/width_scaling.rs`; this bench captures the time dimension.
//!
//! Expected shapes: dense doubles per added qubit (O(2^n) per gate); sparse
//! on concentrated states is flat in width and only tracks support size.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use quantsim::prelude::*;
use std::time::Duration;

fn bench_dense_hlayer(c: &mut Criterion) {
    let reg = GateRegistry::<C64>::standard();
    let mut group = c.benchmark_group("width_dense_hlayer");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    for n in [10usize, 12, 14, 16, 18, 20, 22] {
        let mut circuit = Circuit::new(n);
        for q in 0..n {
            circuit.h(q);
        }
        let bound = circuit.bind(&reg).unwrap();
        group.throughput(Throughput::Elements((n as u64) << n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut state = DenseState::<C64>::new(n).unwrap();
            b.iter(|| {
                state.reset();
                bound.run(&mut state).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_dense_qft(c: &mut Criterion) {
    let reg = GateRegistry::<C64>::standard();
    let mut group = c.benchmark_group("width_dense_qft");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    for n in [8usize, 10, 12, 14, 16] {
        let bound = library::qft::<C64>(n).bind(&reg).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut state = DenseState::<C64>::new(n).unwrap();
            b.iter(|| {
                state.reset();
                bound.run(&mut state).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_sparse_ghz(c: &mut Criterion) {
    // Far past the dense width limit: sparse cost tracks the 2-amplitude
    // support, not 2^n.
    let reg = GateRegistry::<C64>::standard();
    let mut group = c.benchmark_group("width_sparse_ghz");
    for n in [10usize, 20, 30, 40, 50, 60] {
        let bound = library::ghz::<C64>(n).bind(&reg).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut state = SparseState::<C64>::new(n).unwrap();
            b.iter(|| {
                state.reset();
                bound.run(&mut state).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_adaptive_vs_dense_grover(c: &mut Criterion) {
    // Grover keeps a dense-ish state; adaptive promotes during the first
    // H layer and should track dense within a small constant factor
    // thereafter — the cost of promotion, quantified.
    let reg = GateRegistry::<C64>::standard();
    let mut group = c.benchmark_group("width_grover_3iters");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    for n in [8usize, 10, 12, 14] {
        let bound = library::grover::<C64>(n, 3, 3).unwrap().bind(&reg).unwrap();
        group.bench_with_input(BenchmarkId::new("dense", n), &n, |b, &n| {
            let mut state = DenseState::<C64>::new(n).unwrap();
            b.iter(|| {
                state.reset();
                bound.run(&mut state).unwrap();
            });
        });
        group.bench_with_input(BenchmarkId::new("adaptive", n), &n, |b, &n| {
            let mut state = AdaptiveState::<C64>::new(n).unwrap();
            b.iter(|| {
                state.reset();
                bound.run(&mut state).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_mps_ghz(c: &mut Criterion) {
    // Bond dimension stays 2: time is linear in width, far past dense.
    let reg = GateRegistry::<C64>::standard();
    let mut group = c.benchmark_group("width_mps_ghz");
    for n in [10usize, 20, 30, 40, 50, 60] {
        let bound = library::ghz::<C64>(n).bind(&reg).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut state = MpsState::<C64>::new(n).unwrap();
            b.iter(|| {
                state.reset();
                bound.run(&mut state).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_factored_pairs(c: &mut Criterion) {
    // Disjoint entangled pairs: factor sizes stay 4 — linear in width.
    let reg = GateRegistry::<C64>::standard();
    let mut group = c.benchmark_group("width_factored_pairs");
    for n in [8usize, 16, 24, 32, 40, 48] {
        let mut circuit = Circuit::new(n);
        for pair in 0..n / 2 {
            circuit
                .ry(2 * pair, 0.3 + 0.05 * pair as f64)
                .cx(2 * pair, 2 * pair + 1);
        }
        let bound = circuit.bind(&reg).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut state = FactoredState::<C64>::new(n).unwrap();
            b.iter(|| {
                state.reset();
                bound.run(&mut state).unwrap();
            });
        });
    }
    group.finish();
}

/// The era family from the mosaic suite: an `na`-qubit expander graph
/// tile (bundle) beside an `na`-qubit half that lives through a
/// Clifford → T → entangling era sequence. The mosaic's time is the
/// tiles' time; the sparse-fixed comparison pays the expander's 2^na
/// support on every gate that touches it.
fn era_on_mosaic(na: usize) -> quantsim::backend::MosaicState<C64> {
    use quantsim::backend::{MosaicPolicy, MosaicState};
    let n = 2 * na;
    let sim: Simulator = Simulator::new();
    let reg = GateRegistry::<C64>::standard();
    let mut rng = Prng::new(9);
    let mut ga = Circuit::new(na);
    for q in 0..na {
        ga.h(q);
    }
    let mut placed = 0;
    while placed < 3 * na / 2 {
        let a = (rng.next_u64() % na as u64) as usize;
        let b = (rng.next_u64() % na as u64) as usize;
        if a != b {
            ga.gate("cz", [], [a, b]);
            placed += 1;
        }
    }
    let mut a = sim.backends().create("bundle", na).unwrap();
    ga.bind(&reg).unwrap().run(a.as_mut()).unwrap();
    let b1 = sim.backends().create("bundle", na / 2).unwrap();
    let b2 = sim.backends().create("bundle", na - na / 2).unwrap();
    let mut m = MosaicState::with_regions(
        n,
        vec![
            ((0..na).collect(), a),
            ((na..na + na / 2).collect(), b1),
            ((na + na / 2..n).collect(), b2),
        ],
        MosaicPolicy::default(),
    )
    .unwrap();
    let h = reg.resolve("h").unwrap().matrix(&[]).unwrap();
    let cx = reg.resolve("cx").unwrap().matrix(&[]).unwrap();
    let t = reg.resolve("t").unwrap().matrix(&[]).unwrap();
    for (q0, width) in [(na, na / 2), (na + na / 2, na - na / 2)] {
        m.apply(&h, &[q0]).unwrap();
        for q in q0..q0 + width - 1 {
            m.apply(&cx, &[q, q + 1]).unwrap();
        }
    }
    m.apply(&t, &[na + 1]).unwrap();
    m.apply(&t, &[na + na / 2 + 1]).unwrap();
    m.apply(&cx, &[na + 1, na + na / 2 + 1]).unwrap();
    m
}

fn bench_mosaic_era(c: &mut Criterion) {
    // The multi-representation register: whole-scenario time (tile
    // prep, era gates, one migration, one merge) as width doubles.
    let mut group = c.benchmark_group("width_mosaic_era");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    for na in [8usize, 12, 16, 20] {
        group.bench_with_input(BenchmarkId::from_parameter(2 * na), &na, |b, &na| {
            b.iter(|| era_on_mosaic(na));
        });
    }
    group.finish();
}

fn bench_sparse_era(c: &mut Criterion) {
    // The best fixed lens on the same family: sparse carries the
    // expander's 2^na support through every gate.
    let reg = GateRegistry::<C64>::standard();
    let mut group = c.benchmark_group("width_sparse_era");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    for na in [6usize, 8, 10, 12] {
        let n = 2 * na;
        let mut rng = Prng::new(9);
        let mut full = Circuit::new(n);
        for q in 0..na {
            full.h(q);
        }
        let mut placed = 0;
        while placed < 3 * na / 2 {
            let a = (rng.next_u64() % na as u64) as usize;
            let b = (rng.next_u64() % na as u64) as usize;
            if a != b {
                full.gate("cz", [], [a, b]);
                placed += 1;
            }
        }
        for (q0, width) in [(na, na / 2), (na + na / 2, na - na / 2)] {
            full.h(q0);
            for q in q0..q0 + width - 1 {
                full.cx(q, q + 1);
            }
        }
        full.t(na + 1);
        full.t(na + na / 2 + 1);
        full.cx(na + 1, na + na / 2 + 1);
        let bound = full.bind(&reg).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut state = SparseState::<C64>::new(n).unwrap();
            b.iter(|| {
                state.reset();
                bound.run(&mut state).unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dense_hlayer,
    bench_dense_qft,
    bench_sparse_ghz,
    bench_mps_ghz,
    bench_factored_pairs,
    bench_adaptive_vs_dense_grover,
    bench_mosaic_era,
    bench_sparse_era
);
criterion_main!(benches);
