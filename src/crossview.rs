//! The cross-sparsified view: every Pauli string's expectation from
//! three transforms, and a drill that costs the state's structure.
//!
//! Asking a state for one Pauli expectation costs `O(2^n)`. Asking it
//! for all `4^n` of them the obvious way costs `O(8^n)`. But the
//! expectations are not independent numbers — they are *harmonics of the
//! label group*, and a Walsh–Hadamard transform is the map that produces
//! all of them at once.
//!
//! ```text
//! ⟨Z_S⟩      = W(|ψ|²)[S]                 one transform, every S
//! ⟨X_A⟩      = W(|W(ψ)|²)[A] / N          two transforms, every A
//! ⟨X_A Z_B⟩  = W(g_A)[B],  g_A(x) = ψ_x · conj(ψ_{x⊕A})
//! ```
//!
//! So `2·2^n` pure strings come out of three transforms — `O(n·2^n)`
//! where the naive route is `O(4^n)` — and every *mixed* string is one
//! further transform per difference `A`.
//!
//! ## The sparsification, and why it is exact
//!
//! There are `2^n` differences and most states do not use them. The
//! selector is the **mass**
//!
//! ```text
//! mass(A) = Σ_x |ψ_x|·|ψ_{x⊕A}|          = W(|W(|ψ|)|²)[A] / N
//! ```
//!
//! which is non-negative and vanishes **exactly** when `g_A ≡ 0` — and
//! `g_A ≡ 0` forces `⟨X_A Z_B⟩ = 0` for every `B` at once. So the
//! differences with zero mass can be skipped without approximating
//! anything: this is a proof that `2^n` numbers are zero, not a
//! threshold below which they are assumed small. A GHZ state on 13
//! qubits fires 2 of 8192.
//!
//! What that buys is a readout whose cost tracks *how structured the
//! state is* rather than how large it is, which is what makes it usable
//! as a sensor inside an evolution rather than only at the end of one.
//!
//! ## What this is not
//!
//! It is not sub-exponential. Three transforms of a `2^n` vector is
//! `O(n·2^n)`, and the state has to be dense enough to transform. The
//! win is a factor of `2^n/n` against the naive route and a further
//! factor of `2^n / |fired|` on the mixed strings — large, exact, and
//! still exponential. Modules that claim otherwise are elsewhere in this
//! crate; this one is honest about being a better constant.

use crate::backend::Backend;
use crate::error::{Error, Result};
use crate::gates::Pauli;
use crate::scalar::C64;

/// In-place unnormalized Walsh–Hadamard transform.
///
/// `W(v)[s] = Σ_x (−1)^{|x ∧ s|} v[x]`, and `W(W(v)) = N·v`. The length
/// must be a power of two.
pub fn fwht(v: &mut [f64]) {
    let n = v.len();
    let mut len = 1;
    while len < n {
        for block in (0..n).step_by(len << 1) {
            for i in block..block + len {
                let (a, b) = (v[i], v[i + len]);
                v[i] = a + b;
                v[i + len] = a - b;
            }
        }
        len <<= 1;
    }
}

/// The same transform on complex data, componentwise — the transform is
/// real and orthogonal, so this is exactly `W` on each part.
pub fn fwht_c(v: &mut [C64]) {
    let n = v.len();
    let mut len = 1;
    while len < n {
        for block in (0..n).step_by(len << 1) {
            for i in block..block + len {
                let (a, b) = (v[i], v[i + len]);
                v[i] = a + b;
                v[i + len] = a - b;
            }
        }
        len <<= 1;
    }
}

/// Every pure Pauli-string expectation of a state, plus the selector
/// that says which mixed ones can be nonzero.
#[derive(Clone, Debug)]
pub struct CrossView {
    qubits: usize,
    amplitudes: Vec<C64>,
    x: Vec<f64>,
    z: Vec<f64>,
    mass: Vec<f64>,
    transforms: usize,
}

impl CrossView {
    /// Build the view from raw amplitudes.
    pub fn of_amplitudes(amplitudes: &[C64]) -> Result<CrossView> {
        let n = amplitudes.len();
        if n == 0 || !n.is_power_of_two() {
            return Err(Error::InvalidState(format!(
                "crossview: {n} amplitudes is not a power of two"
            )));
        }
        let qubits = n.trailing_zeros() as usize;
        let scale = n as f64;

        // ⟨Z_S⟩ for every S: one transform of the probability diagonal.
        let mut z: Vec<f64> = amplitudes.iter().map(|a| a.re * a.re + a.im * a.im).collect();
        fwht(&mut z);

        // ⟨X_A⟩ for every A: the XOR-autocorrelation of ψ, which the
        // convolution theorem turns into two transforms.
        let mut re: Vec<f64> = amplitudes.iter().map(|a| a.re).collect();
        let mut im: Vec<f64> = amplitudes.iter().map(|a| a.im).collect();
        fwht(&mut re);
        fwht(&mut im);
        let mut x: Vec<f64> = re.iter().zip(&im).map(|(r, i)| r * r + i * i).collect();
        fwht(&mut x);
        for v in x.iter_mut() {
            *v /= scale;
        }

        // The mass: the same autocorrelation on |ψ| rather than ψ, so it
        // cannot cancel. Zero mass is a *proof* that the difference
        // contributes nothing to any mixed string.
        let mut m: Vec<f64> = amplitudes
            .iter()
            .map(|a| (a.re * a.re + a.im * a.im).sqrt())
            .collect();
        fwht(&mut m);
        let mut mass: Vec<f64> = m.iter().map(|v| v * v).collect();
        fwht(&mut mass);
        for v in mass.iter_mut() {
            *v /= scale;
        }

        Ok(CrossView {
            qubits,
            amplitudes: amplitudes.to_vec(),
            x,
            z,
            mass,
            transforms: 5,
        })
    }

    /// Build the view from any backend that can report its amplitudes.
    pub fn of(state: &dyn Backend<C64>) -> Result<CrossView> {
        let n = state.num_qubits();
        if n >= 63 {
            return Err(Error::TooManyQubits {
                requested: n,
                max: 62,
            });
        }
        let amps: Vec<C64> = (0..1u64 << n).map(|i| state.amplitude(i)).collect();
        CrossView::of_amplitudes(&amps)
    }

    /// Register width.
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    /// Transforms performed so far — the cost, in the only unit that
    /// matters here.
    pub fn transforms(&self) -> usize {
        self.transforms
    }

    /// `⟨X_A⟩` for the qubit subset `A`.
    pub fn x_string(&self, a: u64) -> f64 {
        self.x[a as usize]
    }

    /// `⟨Z_S⟩` for the qubit subset `S`.
    pub fn z_string(&self, s: u64) -> f64 {
        self.z[s as usize]
    }

    /// Every `⟨X_A⟩`, indexed by `A`.
    pub fn x_strings(&self) -> &[f64] {
        &self.x
    }

    /// Every `⟨Z_S⟩`, indexed by `S`.
    pub fn z_strings(&self) -> &[f64] {
        &self.z
    }

    /// The mass of a difference: `Σ_x |ψ_x||ψ_{x⊕A}|`.
    ///
    /// Zero exactly when every mixed string `⟨X_A Z_B⟩` vanishes.
    pub fn mass(&self, a: u64) -> f64 {
        self.mass[a as usize]
    }

    /// The differences that can contribute anything, ascending.
    ///
    /// `tol` guards against the transform's own rounding, not against
    /// small-but-real structure: the underlying statement is exact, so
    /// `tol = 0` is meaningful and only floating point argues for more.
    pub fn fired(&self, tol: f64) -> Vec<u64> {
        (0..self.mass.len() as u64)
            .filter(|&a| self.mass[a as usize] > tol)
            .collect()
    }

    /// How much of the difference space the state actually uses.
    pub fn sparsity(&self, tol: f64) -> f64 {
        self.fired(tol).len() as f64 / self.mass.len() as f64
    }

    /// `⟨X_A Z_B⟩` for every `B` at once, in the crate's Hermitian
    /// normalization — one further transform.
    ///
    /// The returned index `B` is the Z-support; the Pauli acted on qubit
    /// `q` is `X` if `A` has bit `q`, `Z` if `B` does, and `Y` if both.
    pub fn drill(&self, a: u64) -> Vec<f64> {
        let n = self.amplitudes.len();
        let mut g: Vec<C64> = (0..n)
            .map(|x| {
                let p = self.amplitudes[x];
                let q = self.amplitudes[x ^ a as usize];
                // ψ_x · conj(ψ_{x⊕A})
                C64::new(p.re * q.re + p.im * q.im, p.im * q.re - p.re * q.im)
            })
            .collect();
        fwht_c(&mut g);
        // ⟨i^{|A∧B|} X_A Z_B⟩ is the Hermitian one, and is real.
        (0..n)
            .map(|b| {
                let v = g[b];
                match (a & b as u64).count_ones() % 4 {
                    0 => v.re,
                    1 => -v.im,
                    2 => -v.re,
                    _ => v.im,
                }
            })
            .collect()
    }

    /// One mixed string, `⟨i^{|A∧B|} X_A Z_B⟩`, without the full drill.
    pub fn mixed(&self, a: u64, b: u64) -> f64 {
        let n = self.amplitudes.len();
        let mut acc = C64::new(0.0, 0.0);
        for x in 0..n {
            let p = self.amplitudes[x];
            let q = self.amplitudes[x ^ a as usize];
            let g = C64::new(p.re * q.re + p.im * q.im, p.im * q.re - p.re * q.im);
            if (x as u64 & b).count_ones() % 2 == 0 {
                acc += g;
            } else {
                acc -= g;
            }
        }
        match (a & b).count_ones() % 4 {
            0 => acc.re,
            1 => -acc.im,
            2 => -acc.re,
            _ => acc.im,
        }
    }

    /// The Pauli operator `(A, B)` names, for handing to
    /// [`pauli_expectation`](crate::backend::pauli_expectation).
    pub fn ops(&self, a: u64, b: u64) -> Vec<(usize, Pauli)> {
        (0..self.qubits)
            .filter_map(|q| match (a >> q & 1, b >> q & 1) {
                (1, 0) => Some((q, Pauli::X)),
                (0, 1) => Some((q, Pauli::Z)),
                (1, 1) => Some((q, Pauli::Y)),
                _ => None,
            })
            .collect()
    }

    /// The connected (cumulant) two-site correlation on the `Z` axis:
    /// `⟨Z_a Z_b⟩ − ⟨Z_a⟩⟨Z_b⟩`.
    pub fn connected_z(&self, a: usize, b: usize) -> f64 {
        let (ma, mb) = (1u64 << a, 1u64 << b);
        self.z_string(ma | mb) - self.z_string(ma) * self.z_string(mb)
    }

    /// The **entanglement complex** on the `Z` axis: the qubit pairs
    /// whose connected correlation exceeds `tol`.
    ///
    /// Built from the spectrum already in hand, so it costs nothing
    /// beyond the transforms — the point being that the *topology* of a
    /// state's correlations is readable without ever forming a reduced
    /// density matrix.
    pub fn entanglement_complex(&self, tol: f64) -> Vec<(usize, usize, f64)> {
        let mut out = Vec::new();
        for a in 0..self.qubits {
            for b in (a + 1)..self.qubits {
                let c = self.connected_z(a, b);
                if c.abs() > tol {
                    out.push((a, b, c));
                }
            }
        }
        out
    }
}
