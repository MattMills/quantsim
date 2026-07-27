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

criterion_group!(
    benches,
    bench_dense_hlayer,
    bench_dense_qft,
    bench_sparse_ghz,
    bench_adaptive_vs_dense_grover
);
criterion_main!(benches);
