//! Pin the crate's bit-ordering and control conventions with explicit
//! basis-state behavior. If any of these break, every downstream user's
//! mental model breaks — they are the most load-bearing tests in the suite.
//!
//! Convention: little-endian (qubit 0 = least significant index bit),
//! controls at the first listed qubit.

mod common;

use common::*;
use quantsim::prelude::*;
use std::f64::consts::FRAC_1_SQRT_2;

fn basis(n: usize, index: u64) -> Circuit {
    let mut c = Circuit::new(n);
    for q in 0..n {
        if (index >> q) & 1 == 1 {
            c.x(q);
        }
    }
    c
}

#[test]
fn x_on_qubit_zero_sets_lsb() {
    let mut c = Circuit::new(2);
    c.x(0);
    let s = run_dense(&c);
    assert_amp(s.as_ref(), 0b01, 1.0, 0.0);
}

#[test]
fn x_on_qubit_one_sets_msb() {
    let mut c = Circuit::new(2);
    c.x(1);
    let s = run_dense(&c);
    assert_amp(s.as_ref(), 0b10, 1.0, 0.0);
}

#[test]
fn cx_control_first_argument() {
    // Control q0 set: |01⟩ → |11⟩.
    let mut c = basis(2, 0b01);
    c.cx(0, 1);
    assert_amp(run_dense(&c).as_ref(), 0b11, 1.0, 0.0);

    // Control q1 clear: |01⟩ with cx(1, 0) is untouched.
    let mut c = basis(2, 0b01);
    c.cx(1, 0);
    assert_amp(run_dense(&c).as_ref(), 0b01, 1.0, 0.0);

    // Control q1 set: |10⟩ with cx(1, 0) → |11⟩.
    let mut c = basis(2, 0b10);
    c.cx(1, 0);
    assert_amp(run_dense(&c).as_ref(), 0b11, 1.0, 0.0);
}

#[test]
fn swap_exchanges_bits() {
    let mut c = basis(2, 0b01);
    c.swap(0, 1);
    assert_amp(run_dense(&c).as_ref(), 0b10, 1.0, 0.0);
}

#[test]
fn iswap_adds_i_on_exchange() {
    let mut c = basis(2, 0b01);
    c.iswap(0, 1);
    assert_amp(run_dense(&c).as_ref(), 0b10, 0.0, 1.0);
    // |11⟩ passes through unchanged.
    let mut c = basis(2, 0b11);
    c.iswap(0, 1);
    assert_amp(run_dense(&c).as_ref(), 0b11, 1.0, 0.0);
}

#[test]
fn toffoli_controls_first() {
    let mut c = basis(3, 0b011);
    c.ccx(0, 1, 2);
    assert_amp(run_dense(&c).as_ref(), 0b111, 1.0, 0.0);
    // One control clear: unchanged.
    let mut c = basis(3, 0b001);
    c.ccx(0, 1, 2);
    assert_amp(run_dense(&c).as_ref(), 0b001, 1.0, 0.0);
}

#[test]
fn fredkin_swaps_when_control_set() {
    // cswap(control=0, a=1, b=2) on |q2 q1 q0⟩ = |011⟩ → |101⟩.
    let mut c = basis(3, 0b011);
    c.cswap(0, 1, 2);
    assert_amp(run_dense(&c).as_ref(), 0b101, 1.0, 0.0);
    // Control clear: |110⟩ unchanged.
    let mut c = basis(3, 0b110);
    c.cswap(0, 1, 2);
    assert_amp(run_dense(&c).as_ref(), 0b110, 1.0, 0.0);
}

#[test]
fn cp_phases_only_the_11_component() {
    let theta = 0.777;
    let mut c = Circuit::new(2);
    c.h(0).h(1).cp(0, 1, theta);
    let s = run_dense(&c);
    assert_amp(s.as_ref(), 0b00, 0.5, 0.0);
    assert_amp(s.as_ref(), 0b01, 0.5, 0.0);
    assert_amp(s.as_ref(), 0b10, 0.5, 0.0);
    assert_amp(s.as_ref(), 0b11, 0.5 * theta.cos(), 0.5 * theta.sin());
}

#[test]
fn ch_applies_h_to_target_when_control_set() {
    let mut c = basis(2, 0b01);
    c.ch(0, 1);
    let s = run_dense(&c);
    assert_amp(s.as_ref(), 0b01, FRAC_1_SQRT_2, 0.0);
    assert_amp(s.as_ref(), 0b11, FRAC_1_SQRT_2, 0.0);
}

#[test]
fn hadamard_signs() {
    // H|1⟩ = (|0⟩ − |1⟩)/√2.
    let mut c = basis(1, 1);
    c.h(0);
    let s = run_dense(&c);
    assert_amp(s.as_ref(), 0, FRAC_1_SQRT_2, 0.0);
    assert_amp(s.as_ref(), 1, -FRAC_1_SQRT_2, 0.0);
}

#[test]
fn bell_state_amplitudes() {
    let s = run_dense(&library::bell());
    assert_amp(s.as_ref(), 0b00, FRAC_1_SQRT_2, 0.0);
    assert_amp(s.as_ref(), 0b01, 0.0, 0.0);
    assert_amp(s.as_ref(), 0b10, 0.0, 0.0);
    assert_amp(s.as_ref(), 0b11, FRAC_1_SQRT_2, 0.0);
}

#[test]
fn phase_gates_act_on_one_only() {
    // S|1⟩ = i|1⟩, T|1⟩ = e^{iπ/4}|1⟩; |0⟩ untouched.
    let mut c = basis(1, 1);
    c.s(0);
    assert_amp(run_dense(&c).as_ref(), 1, 0.0, 1.0);
    let mut c = basis(1, 1);
    c.t(0);
    assert_amp(run_dense(&c).as_ref(), 1, FRAC_1_SQRT_2, FRAC_1_SQRT_2);
    let mut c = basis(1, 0);
    c.s(0).t(0);
    assert_amp(run_dense(&c).as_ref(), 0, 1.0, 0.0);
}

#[test]
fn y_gate_phases() {
    // Y|0⟩ = i|1⟩, Y|1⟩ = −i|0⟩.
    let mut c = basis(1, 0);
    c.y(0);
    assert_amp(run_dense(&c).as_ref(), 1, 0.0, 1.0);
    let mut c = basis(1, 1);
    c.y(0);
    assert_amp(run_dense(&c).as_ref(), 0, 0.0, -1.0);
}

#[test]
fn rotations_at_pi() {
    // rx(π) = −iX, ry(π)|0⟩ = |1⟩, rz(π)|0⟩ = −i|0⟩.
    let pi = std::f64::consts::PI;
    let mut c = basis(1, 0);
    c.rx(0, pi);
    assert_amp(run_dense(&c).as_ref(), 1, 0.0, -1.0);
    let mut c = basis(1, 0);
    c.ry(0, pi);
    assert_amp(run_dense(&c).as_ref(), 1, 1.0, 0.0);
    let mut c = basis(1, 0);
    c.rz(0, pi);
    assert_amp(run_dense(&c).as_ref(), 0, 0.0, -1.0);
}

#[test]
fn u_gate_columns() {
    let (theta, phi, lam) = (1.234, 0.567, 2.101);
    // Column 0: U|0⟩ = [cos(θ/2), e^{iφ} sin(θ/2)].
    let mut c = basis(1, 0);
    c.u(0, theta, phi, lam);
    let s = run_dense(&c);
    let (ct, st) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    assert_amp(s.as_ref(), 0, ct, 0.0);
    assert_amp(s.as_ref(), 1, phi.cos() * st, phi.sin() * st);
    // Column 1: U|1⟩ = [−e^{iλ} sin(θ/2), e^{i(φ+λ)} cos(θ/2)].
    let mut c = basis(1, 1);
    c.u(0, theta, phi, lam);
    let s = run_dense(&c);
    assert_amp(s.as_ref(), 0, -lam.cos() * st, -lam.sin() * st);
    assert_amp(s.as_ref(), 1, (phi + lam).cos() * ct, (phi + lam).sin() * ct);
}

#[test]
fn gates_only_touch_their_targets() {
    // A gate on qubits {1, 3} of five must leave the marginal state of the
    // others alone: check by applying to a basis state.
    let mut c = basis(5, 0b10101);
    c.cx(1, 3); // control q1 = 0 → no-op on this basis state
    assert_amp(run_dense(&c).as_ref(), 0b10101, 1.0, 0.0);
    let mut c = basis(5, 0b10111);
    c.cx(1, 3); // control q1 = 1 → flips q3
    assert_amp(run_dense(&c).as_ref(), 0b11111, 1.0, 0.0);
}
