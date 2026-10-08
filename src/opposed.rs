//! Bridge to Opposed Mathematics (OM), whose crates are re-exported as
//! [`crate::om`].
//!
//! OM computes with exact phases: an amplitude is a bucket `ℕ[μ_N]` of
//! `N`-th roots of unity in which `χ` and `−χ` cancel, and a state is a
//! sparse map from points of `∏ Z_m` to buckets. This module converts
//! between that and quantsim's representations.
//!
//! * **Numbers.** The numerator ring `ℤ[ω]` of [`DOmega`], `ω = e^{iπ/4}`,
//!   is OM's `ℤ[ζ_8]`: the coefficients on `1, ω, ω², ω³` are the power
//!   basis of [`Cyclo<8>`], and a reduced `Bucket<μ_8>` is the same element
//!   with each negative coefficient carried by `−ω^j = ω^{j+4}`
//!   ([`domega_to_cyclo`], [`domega_to_bucket`] and back).
//! * **States.** An [`ExactState`] becomes an OM state on `n` factors `Z_2`
//!   over one common denominator `√2^k` ([`exact_to_om`]); an OM state on
//!   qubit factors loads into any [`Backend`] through `ι : μ_N → U(1)`
//!   ([`load_om_state`]). Qubit `q` is factor `q`, and basis index
//!   `Σ x_q 2^q`, matching quantsim's little-endian convention.
//! * **Observables.** An OM [`Hamiltonian`] is keyed by `(x, z)` for
//!   `X^x Z^z` with a Gaussian-integer coefficient — quantsim's
//!   [`PauliKey`] basis — so it becomes a [`PauliSum`] term for term
//!   ([`pauli_sum`]). [`expectation`] evaluates it on any backend, and
//!   [`expectation_exact`] on an [`ExactState`] in `D[ω]`.
//!
//! ```
//! use quantsim::om::om_hamiltonian::hamiltonian::heisenberg_chain;
//! use quantsim::opposed::{expectation, expectation_exact};
//! use quantsim::prelude::*;
//!
//! // Singlet on qubits 0, 1: ⟨X₀X₁ + Y₀Y₁ + Z₀Z₁⟩ = −3, exactly.
//! let mut c = Circuit::new(2);
//! c.x(0).h(0).cx(0, 1).x(1);
//! let h = heisenberg_chain(2, false);
//! let energy = expectation_exact(&h, &ExactState::run(&c)?)?;
//! assert_eq!(energy, DOmega::int(-3));
//! let sim: Simulator = Simulator::new();
//! assert!((expectation(&h, sim.run(&c)?.as_ref())?.re + 3.0).abs() < 1e-12);
//! # Ok::<(), quantsim::Error>(())
//! ```

use crate::backend::{pauli_expectation, Backend};
use crate::error::{Error, Result};
use crate::exact::{DOmega, ExactState};
use crate::gates::Pauli;
use crate::heisenberg::{PauliKey, PauliSum};
use crate::scalar::C64;
use opposed_mathematics::om_core::bucket::Bucket;
use opposed_mathematics::om_core::cyclic::Zm;
use opposed_mathematics::om_core::cyclotomic::Cyclo;
use opposed_mathematics::om_core::xi::MuN;
use opposed_mathematics::om_hamiltonian::hamiltonian::Hamiltonian;
use opposed_mathematics::om_ipg::point::Point;
use opposed_mathematics::om_ipg::state::State;
use opposed_mathematics::om_quantum::embed::{bucket_value, Embed};

/// The eighth roots of unity, the phases of the Clifford+T fragment.
pub type Mu8 = MuN<8>;

/// The numerator of `d` in `ℤ[ζ_8]` and the power `k` of its `√2`
/// denominator: `d = z / √2^k`.
pub fn domega_to_cyclo(d: DOmega) -> (Cyclo<8>, u32) {
    let (c, k) = d.parts();
    let z = c.iter().enumerate().fold(Cyclo::zero(), |acc, (j, &cj)| {
        acc.add(&Cyclo::zeta_pow(j as i64).scale(cj))
    });
    (z, k)
}

/// `z / √2^k`.
pub fn cyclo_to_domega(z: &Cyclo<8>, k: u32) -> DOmega {
    let c = z.coefficients();
    DOmega::from_parts([c[0], c[1], c[2], c[3]], k)
}

/// The numerator of `d` as a reduced bucket — `c_j` copies of `ω^j` for
/// `c_j > 0`, `−c_j` copies of `ω^{j+4}` for `c_j < 0` — and the power `k`
/// of its `√2` denominator.
///
/// # Errors
///
/// [`Error::InvalidState`] if a coefficient does not fit a `u64` count.
pub fn domega_to_bucket(d: DOmega) -> Result<(Bucket<Mu8>, u32)> {
    let (c, k) = d.parts();
    let mut b = Bucket::nil();
    for (j, &cj) in c.iter().enumerate() {
        let count = u64::try_from(cj.unsigned_abs()).map_err(|_| {
            Error::InvalidState(format!("D[ω] coefficient {cj} exceeds a u64 count"))
        })?;
        let e = if cj < 0 { j + 4 } else { j };
        b.add(Mu8::new(e as u32), count);
    }
    Ok((b, k))
}

/// The value of a `μ_8` bucket divided by `√2^k`.
pub fn bucket_to_domega(b: &Bucket<Mu8>, k: u32) -> DOmega {
    cyclo_to_domega(&Cyclo::from_bucket(b), k)
}

/// The point of `Z_2^n` with coordinates the bits of `index`, qubit `q`
/// first.
pub fn qubit_point(index: u64, n: usize) -> Point {
    Point::new((0..n).map(|q| Zm::new(index >> q & 1, 2)).collect())
}

/// The basis index `Σ x_q 2^q` of a point of `Z_2^n`.
///
/// # Errors
///
/// [`Error::InvalidState`] if a factor is not `Z_2` or there are more than
/// 64 factors.
pub fn qubit_index(p: &Point) -> Result<u64> {
    if p.arity() > 64 {
        return Err(Error::InvalidState(format!(
            "a point with {} factors has no u64 basis index",
            p.arity()
        )));
    }
    p.coords().iter().enumerate().try_fold(0u64, |acc, (q, z)| {
        if z.modulus() == 2 {
            Ok(acc | z.value() << q)
        } else {
            Err(Error::InvalidState(format!(
                "factor {q} is Z_{}, not a qubit",
                z.modulus()
            )))
        }
    })
}

/// `state` as an OM state on `n` factors `Z_2` and the power `k` of `√2`
/// dividing it: `amplitude(i) = ψ(i) / √2^k`, with `k` the largest
/// denominator among the amplitudes.
///
/// # Errors
///
/// [`Error::InvalidState`] on `i128` overflow or a coefficient beyond a
/// `u64` count.
pub fn exact_to_om(state: &ExactState) -> Result<(State<Mu8>, u32)> {
    let n = state.num_qubits();
    let amps: Vec<(u64, DOmega)> = (0..1u64 << n)
        .map(|i| (i, state.amplitude_exact(i)))
        .filter(|(_, d)| !d.is_zero())
        .collect();
    let k = amps.iter().map(|(_, d)| d.parts().1).max().unwrap_or(0);
    let sqrt2 = DOmega::omega_pow(1).add(DOmega::omega_pow(-1))?;
    let mut out = State::empty();
    for (i, d) in amps {
        // d = c / √2^{k_d} = c · √2^{k − k_d} / √2^k.
        let (c, kd) = d.parts();
        let mut num = DOmega::from_parts(c, 0);
        for _ in kd..k {
            num = num.mul(sqrt2)?;
        }
        let (b, _) = domega_to_bucket(num)?;
        let p = qubit_point(i, n);
        for (chi, a) in b.iter() {
            out.add_amp(p.clone(), *chi, a);
        }
    }
    Ok((out, k))
}

/// Replace `backend`'s state with `ι(ψ) / √2^k`, each point of `Z_2^n`
/// at its basis index.
///
/// # Errors
///
/// [`Error::InvalidState`] for a factor that is not `Z_2`,
/// [`Error::WidthMismatch`] if the factor count differs from the backend
/// width, or whatever [`Backend::load`] reports.
pub fn load_om_state<X: Embed>(
    backend: &mut dyn Backend<C64>,
    state: &State<X>,
    k: u32,
) -> Result<()> {
    let scale = 0.5f64.powf(f64::from(k) / 2.0);
    let n = backend.num_qubits();
    let mut entries = Vec::new();
    for (p, b) in state.iter() {
        if p.arity() != n {
            return Err(Error::WidthMismatch {
                circuit: p.arity(),
                backend: n,
            });
        }
        let v = bucket_value(&b) * scale;
        if v != C64::new(0.0, 0.0) {
            entries.push((qubit_index(p)?, v));
        }
    }
    backend.load(&entries)
}

/// `h` as a [`PauliSum`]: the coefficient of `X^x Z^z` is `h`'s Gaussian
/// integer at `(x, z)`.
pub fn pauli_sum(h: &Hamiltonian) -> PauliSum {
    let mut out = PauliSum::zero();
    for (&(x, z), _) in h.iter() {
        let (re, im) = h.gaussian_coefficient(x, z);
        out.add((x, z), C64::new(re as f64, im as f64));
    }
    out
}

/// `X^x Z^z = (−i)^{|x∧z|} ⊗_q P_q` with `P_q ∈ {X, Y, Z}`, as quantsim's
/// `(qubit, Pauli)` list and the phase.
fn hermitian_factors(key: PauliKey) -> (Vec<(usize, Pauli)>, C64) {
    let (x, z) = key;
    let ops = (0..64)
        .filter_map(|q| match (x >> q & 1, z >> q & 1) {
            (1, 1) => Some((q, Pauli::Y)),
            (1, 0) => Some((q, Pauli::X)),
            (0, 1) => Some((q, Pauli::Z)),
            _ => None,
        })
        .collect();
    let phase = [
        C64::new(1.0, 0.0),
        C64::new(0.0, -1.0),
        C64::new(-1.0, 0.0),
        C64::new(0.0, 1.0),
    ][((x & z).count_ones() % 4) as usize];
    (ops, phase)
}

/// `⟨ψ|H|ψ⟩` on any backend, by [`pauli_expectation`] term by term; the
/// energy when the state is normalized.
///
/// # Errors
///
/// [`Error::QubitOutOfRange`] if a term acts beyond the register.
pub fn expectation(h: &Hamiltonian, state: &dyn Backend<C64>) -> Result<C64> {
    let mut total = C64::new(0.0, 0.0);
    for (key, g) in pauli_sum(h).terms() {
        let (ops, phase) = hermitian_factors(key);
        total += g * phase * pauli_expectation(state, &ops)?;
    }
    Ok(total)
}

/// `⟨ψ|H|ψ⟩` exactly in `D[ω]`, using
/// `X^x Z^z |b⟩ = (−1)^{|z∧b|} |b ⊕ x⟩`: `O(terms · 2^n)` ring operations.
///
/// # Errors
///
/// [`Error::QubitOutOfRange`] if a term acts beyond the register, or
/// [`Error::InvalidState`] on `i128` overflow.
pub fn expectation_exact(h: &Hamiltonian, state: &ExactState) -> Result<DOmega> {
    let n = state.num_qubits();
    let width = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
    let mut total = DOmega::zero();
    for (&(x, z), _) in h.iter() {
        if (x | z) & !width != 0 {
            return Err(Error::QubitOutOfRange {
                qubit: 63 - (x | z).leading_zeros() as usize,
                num_qubits: n,
            });
        }
        let (re, im) = h.gaussian_coefficient(x, z);
        let g = DOmega::from_parts([re, 0, im, 0], 0);
        let mut term = DOmega::zero();
        for b in 0..=width {
            let psi = state.amplitude_exact(b);
            if psi.is_zero() {
                continue;
            }
            let bra = state.amplitude_exact(b ^ x).conj().mul(psi)?;
            term = if (z & b).count_ones() % 2 == 0 {
                term.add(bra)?
            } else {
                term.sub(bra)?
            };
        }
        total = total.add(g.mul(term)?)?;
    }
    Ok(total)
}
