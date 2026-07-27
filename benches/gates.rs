//! Per-gate and per-circuit throughput benchmarks at a fixed width.
//!
//! Run with `cargo bench --bench gates`. Elements-per-second figures are
//! amplitudes touched per gate application (2^n), so ns/element compares
//! fairly across widths.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use quantsim::prelude::*;

const N: usize = 16;

fn gate(reg: &GateRegistry, name: &str, params: &[f64]) -> GateMatrix<C64> {
    reg.resolve(name).unwrap().matrix(params).unwrap()
}

fn bench_single_gates(c: &mut Criterion) {
    let reg = GateRegistry::<C64>::standard();
    let mut group = c.benchmark_group("gate_apply_n16");
    group.throughput(Throughput::Elements(1 << N));

    // Unitary gates applied repeatedly to an evolving state: dense gate
    // cost is state-independent, so no per-iteration reset is needed.
    let cases: Vec<(&str, GateMatrix<C64>, Vec<usize>)> = vec![
        ("h_1q", gate(&reg, "h", &[]), vec![7]),
        ("x_1q", gate(&reg, "x", &[]), vec![7]),
        ("rz_1q", gate(&reg, "rz", &[0.731]), vec![7]),
        ("u_1q", gate(&reg, "u", &[0.7, 1.1, 2.3]), vec![7]),
        ("cx_2q", gate(&reg, "cx", &[]), vec![3, 11]),
        ("cp_2q", gate(&reg, "cp", &[0.911]), vec![3, 11]),
        ("rxx_2q", gate(&reg, "rxx", &[1.42]), vec![3, 11]),
        ("swap_2q", gate(&reg, "swap", &[]), vec![3, 11]),
        ("ccx_3q", gate(&reg, "ccx", &[]), vec![2, 9, 14]),
    ];
    for (name, matrix, qubits) in cases {
        let mut state = DenseState::<C64>::new(N).unwrap();
        let mut warm = Circuit::new(N);
        for q in 0..N {
            warm.h(q);
        }
        warm.bind(&reg).unwrap().run(&mut state).unwrap();
        group.bench_function(name, |b| {
            b.iter(|| state.apply(&matrix, &qubits).unwrap());
        });
    }
    group.finish();
}

fn bench_circuits(c: &mut Criterion) {
    let reg = GateRegistry::<C64>::standard();
    let mut group = c.benchmark_group("circuit_run");
    group.sample_size(20);

    let qft = library::qft::<C64>(12).bind(&reg).unwrap();
    group.bench_function("qft_12_dense", |b| {
        let mut state = DenseState::<C64>::new(12).unwrap();
        b.iter(|| {
            state.reset();
            qft.run(&mut state).unwrap();
        });
    });

    let grover = library::grover::<C64>(10, 137, 8)
        .unwrap()
        .bind(&reg)
        .unwrap();
    group.bench_function("grover_10_8iters_dense", |b| {
        let mut state = DenseState::<C64>::new(10).unwrap();
        b.iter(|| {
            state.reset();
            grover.run(&mut state).unwrap();
        });
    });

    let random = library::random_circuit::<C64>(14, 200, 0xBEEF)
        .bind(&reg)
        .unwrap();
    group.bench_function("random_14q_200gates_dense", |b| {
        let mut state = DenseState::<C64>::new(14).unwrap();
        b.iter(|| {
            state.reset();
            random.run(&mut state).unwrap();
        });
    });

    // Dense vs sparse on the same concentrated-support circuit.
    let ghz = library::ghz::<C64>(20).bind(&reg).unwrap();
    group.bench_function("ghz_20_dense", |b| {
        let mut state = DenseState::<C64>::new(20).unwrap();
        b.iter(|| {
            state.reset();
            ghz.run(&mut state).unwrap();
        });
    });
    group.bench_function("ghz_20_sparse", |b| {
        let mut state = SparseState::<C64>::new(20).unwrap();
        b.iter(|| {
            state.reset();
            ghz.run(&mut state).unwrap();
        });
    });
    group.finish();
}

fn bench_algebras(c: &mut Criterion) {
    // The same real circuit (Ry/CX ladder) over increasingly wide algebras:
    // the per-amplitude cost of swapping the scalar, measured directly.
    fn ladder<S: Scalar>(n: usize) -> BoundCircuit<S> {
        let reg = GateRegistry::<S>::standard();
        let mut c = Circuit::new(n);
        for q in 0..n {
            c.ry(q, 0.3 + q as f64 * 0.1);
        }
        for q in 0..n - 1 {
            c.cx(q, q + 1);
        }
        for q in 0..n {
            c.ry(q, 1.1 - q as f64 * 0.05);
        }
        c.bind(&reg).unwrap()
    }
    let n = 12;
    let mut group = c.benchmark_group("algebra_ladder_n12");
    group.throughput(Throughput::Elements((3 * n as u64 - 1) * (1 << n)));

    macro_rules! case {
        ($label:literal, $ty:ty) => {
            let bound = ladder::<$ty>(n);
            group.bench_function($label, |b| {
                let mut state = DenseState::<$ty>::new(n).unwrap();
                b.iter(|| {
                    state.reset();
                    bound.run(&mut state).unwrap();
                });
            });
        };
    }
    case!("real_f64", f64);
    case!("complex_c64", C64);
    case!("complex_cd", CComplex);
    case!("quaternion", Quaternion);
    case!("octonion", Octonion);
    case!("sedenion", Sedenion);
    group.finish();
}

criterion_group!(benches, bench_single_gates, bench_circuits, bench_algebras);
criterion_main!(benches);
