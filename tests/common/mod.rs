//! Shared helpers for the integration test suite.
#![allow(dead_code)]

use quantsim::prelude::*;

pub const TOL: f64 = 1e-9;

pub fn sim() -> Simulator {
    Simulator::new()
}

pub fn run_dense(c: &Circuit) -> Box<dyn Backend<C64>> {
    sim().run(c).expect("circuit should run")
}

pub fn run_named(backend: &str, c: &Circuit) -> Box<dyn Backend<C64>> {
    sim().run_on(backend, c).expect("circuit should run")
}

pub fn assert_amp(state: &dyn Backend<C64>, idx: u64, re: f64, im: f64) {
    let a = state.amplitude(idx);
    assert!(
        (a.re - re).abs() < TOL && (a.im - im).abs() < TOL,
        "amplitude[{idx}] = {a}, expected {re}{im:+}i"
    );
}

pub fn assert_close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() < tol, "{a} vs {b} (tol {tol})");
}

/// A fixed entangling prefix so identities are exercised on a generic state
/// rather than |0…0⟩.
pub fn scrambler(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.u(q, 0.3 + 0.4 * q as f64, 1.1 - 0.2 * q as f64, 0.7 + 0.15 * q as f64);
    }
    for q in 0..n.saturating_sub(1) {
        c.cx(q, q + 1);
    }
    for q in 0..n {
        c.u(q, 0.5 + 0.2 * q as f64, 0.9, 1.3 - 0.1 * q as f64);
    }
    c
}

/// Assert two op sequences act identically on a scrambled generic state.
pub fn assert_equiv(n: usize, fa: impl FnOnce(&mut Circuit), fb: impl FnOnce(&mut Circuit)) {
    let mut ca = scrambler(n);
    fa(&mut ca);
    let mut cb = scrambler(n);
    fb(&mut cb);
    let sa = run_dense(&ca);
    let sb = run_dense(&cb);
    for i in 0..(1u64 << n) {
        let (x, y) = (sa.amplitude(i), sb.amplitude(i));
        assert!(x.approx_eq(y, TOL), "amplitude {i} differs: {x} vs {y}");
    }
}

/// Assert equivalence up to one global complex phase.
pub fn assert_equiv_up_to_phase(
    n: usize,
    fa: impl FnOnce(&mut Circuit),
    fb: impl FnOnce(&mut Circuit),
) {
    let mut ca = scrambler(n);
    fa(&mut ca);
    let mut cb = scrambler(n);
    fb(&mut cb);
    let sa = run_dense(&ca);
    let sb = run_dense(&cb);
    // Phase from the largest amplitude of a.
    let mut best = (0u64, 0.0f64);
    for i in 0..(1u64 << n) {
        let m = sa.amplitude(i).abs_sqr();
        if m > best.1 {
            best = (i, m);
        }
    }
    assert!(best.1 > 1e-12, "state a vanished");
    let phase = sb.amplitude(best.0) / sa.amplitude(best.0);
    assert_close(phase.norm(), 1.0, 1e-6);
    for i in 0..(1u64 << n) {
        let (x, y) = (sa.amplitude(i) * phase, sb.amplitude(i));
        assert!(x.approx_eq(y, 1e-7), "amplitude {i} differs up to phase: {x} vs {y}");
    }
}
