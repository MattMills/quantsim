//! The standard gate library.
//!
//! Every gate is defined once as a complex matrix (little-endian sub-index
//! convention; controls on the **low** sub-index bits, i.e. `qubits[0]`) and
//! projected into the target algebra entry-by-entry via
//! [`Scalar::try_from_c64`]. Gates whose matrix does not embed in an algebra
//! simply do not exist there: over ℝ the registry ends up with the real
//! subset (`x`, `z`, `h`, `ry`, `cx`, ...), while ℂ, ℍ and 𝕆 carry the full
//! library because ℂ embeds in each.
//!
//! Rotation conventions: `rx/ry/rz(θ) = exp(−i θ P / 2)` (half-angle, like
//! OpenQASM/Qiskit); `p(λ) = diag(1, e^{iλ})`;
//! `u(θ, φ, λ)` is the generic single-qubit gate
//! `[[cos(θ/2), −e^{iλ} sin(θ/2)], [e^{iφ} sin(θ/2), e^{i(φ+λ)} cos(θ/2)]]`.

use std::f64::consts::FRAC_1_SQRT_2;

use crate::error::{Error, Result};
use crate::math::{c64, cis, GateMatrix};
use crate::registry::GateRegistry;
use crate::scalar::{Scalar, C64};

/// Non-special probe angles used when deciding whether a parametric gate
/// exists over an algebra (avoids values like 0 or π where an otherwise
/// complex matrix degenerates to a real one).
pub(crate) const PROBE_PARAMS: [f64; 3] = [0.734_912_648_1, 1.892_346_501_7, 2.577_180_932_3];

// ---------------------------------------------------------------------------
// Complex matrix builders (row-major).
// ---------------------------------------------------------------------------

/// Controlled version with the new control on sub-index bit 0.
fn ctl(u: &[C64], d: usize) -> Vec<C64> {
    let nd = 2 * d;
    let mut out = vec![c64(0.0, 0.0); nd * nd];
    for r in 0..d {
        out[(2 * r) * nd + (2 * r)] = c64(1.0, 0.0);
        for c in 0..d {
            out[(2 * r + 1) * nd + (2 * c + 1)] = u[r * d + c];
        }
    }
    out
}

fn m_id(_: &[f64]) -> Vec<C64> {
    vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(1.0, 0.0)]
}
fn m_x(_: &[f64]) -> Vec<C64> {
    vec![c64(0.0, 0.0), c64(1.0, 0.0), c64(1.0, 0.0), c64(0.0, 0.0)]
}
fn m_y(_: &[f64]) -> Vec<C64> {
    vec![c64(0.0, 0.0), c64(0.0, -1.0), c64(0.0, 1.0), c64(0.0, 0.0)]
}
fn m_z(_: &[f64]) -> Vec<C64> {
    vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(-1.0, 0.0)]
}
fn m_h(_: &[f64]) -> Vec<C64> {
    let s = FRAC_1_SQRT_2;
    vec![c64(s, 0.0), c64(s, 0.0), c64(s, 0.0), c64(-s, 0.0)]
}
fn m_s(_: &[f64]) -> Vec<C64> {
    vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(0.0, 1.0)]
}
fn m_sdg(_: &[f64]) -> Vec<C64> {
    vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(0.0, -1.0)]
}
fn m_t(_: &[f64]) -> Vec<C64> {
    vec![
        c64(1.0, 0.0),
        c64(0.0, 0.0),
        c64(0.0, 0.0),
        cis(std::f64::consts::FRAC_PI_4),
    ]
}
fn m_tdg(_: &[f64]) -> Vec<C64> {
    vec![
        c64(1.0, 0.0),
        c64(0.0, 0.0),
        c64(0.0, 0.0),
        cis(-std::f64::consts::FRAC_PI_4),
    ]
}
fn m_sx(_: &[f64]) -> Vec<C64> {
    vec![c64(0.5, 0.5), c64(0.5, -0.5), c64(0.5, -0.5), c64(0.5, 0.5)]
}
fn m_sxdg(_: &[f64]) -> Vec<C64> {
    vec![c64(0.5, -0.5), c64(0.5, 0.5), c64(0.5, 0.5), c64(0.5, -0.5)]
}
fn m_rx(p: &[f64]) -> Vec<C64> {
    let (c, s) = ((p[0] / 2.0).cos(), (p[0] / 2.0).sin());
    vec![c64(c, 0.0), c64(0.0, -s), c64(0.0, -s), c64(c, 0.0)]
}
fn m_ry(p: &[f64]) -> Vec<C64> {
    let (c, s) = ((p[0] / 2.0).cos(), (p[0] / 2.0).sin());
    vec![c64(c, 0.0), c64(-s, 0.0), c64(s, 0.0), c64(c, 0.0)]
}
fn m_rz(p: &[f64]) -> Vec<C64> {
    vec![
        cis(-p[0] / 2.0),
        c64(0.0, 0.0),
        c64(0.0, 0.0),
        cis(p[0] / 2.0),
    ]
}
fn m_p(p: &[f64]) -> Vec<C64> {
    vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), cis(p[0])]
}
fn m_u(p: &[f64]) -> Vec<C64> {
    let (theta, phi, lam) = (p[0], p[1], p[2]);
    let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    vec![c64(c, 0.0), -cis(lam) * s, cis(phi) * s, cis(phi + lam) * c]
}
fn m_cx(p: &[f64]) -> Vec<C64> {
    ctl(&m_x(p), 2)
}
fn m_cy(p: &[f64]) -> Vec<C64> {
    ctl(&m_y(p), 2)
}
fn m_cz(p: &[f64]) -> Vec<C64> {
    ctl(&m_z(p), 2)
}
fn m_ch(p: &[f64]) -> Vec<C64> {
    ctl(&m_h(p), 2)
}
fn m_cp(p: &[f64]) -> Vec<C64> {
    ctl(&m_p(p), 2)
}
fn m_crx(p: &[f64]) -> Vec<C64> {
    ctl(&m_rx(p), 2)
}
fn m_cry(p: &[f64]) -> Vec<C64> {
    ctl(&m_ry(p), 2)
}
fn m_crz(p: &[f64]) -> Vec<C64> {
    ctl(&m_rz(p), 2)
}
fn m_swap(_: &[f64]) -> Vec<C64> {
    let (o, l) = (c64(1.0, 0.0), c64(0.0, 0.0));
    vec![o, l, l, l, l, l, o, l, l, o, l, l, l, l, l, o]
}
fn m_iswap(_: &[f64]) -> Vec<C64> {
    let (o, l, i) = (c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 1.0));
    vec![o, l, l, l, l, l, i, l, l, i, l, l, l, l, l, o]
}
fn m_rxx(p: &[f64]) -> Vec<C64> {
    let (c, s) = ((p[0] / 2.0).cos(), (p[0] / 2.0).sin());
    let (cc, is, l) = (c64(c, 0.0), c64(0.0, -s), c64(0.0, 0.0));
    vec![cc, l, l, is, l, cc, is, l, l, is, cc, l, is, l, l, cc]
}
fn m_ryy(p: &[f64]) -> Vec<C64> {
    let (c, s) = ((p[0] / 2.0).cos(), (p[0] / 2.0).sin());
    let (cc, mis, pis, l) = (c64(c, 0.0), c64(0.0, -s), c64(0.0, s), c64(0.0, 0.0));
    vec![cc, l, l, pis, l, cc, mis, l, l, mis, cc, l, pis, l, l, cc]
}
fn m_rzz(p: &[f64]) -> Vec<C64> {
    let (em, ep, l) = (cis(-p[0] / 2.0), cis(p[0] / 2.0), c64(0.0, 0.0));
    vec![em, l, l, l, l, ep, l, l, l, l, ep, l, l, l, l, em]
}
fn m_ccx(p: &[f64]) -> Vec<C64> {
    ctl(&ctl(&m_x(p), 2), 4)
}
fn m_ccz(p: &[f64]) -> Vec<C64> {
    ctl(&ctl(&m_z(p), 2), 4)
}
fn m_cswap(p: &[f64]) -> Vec<C64> {
    ctl(&m_swap(p), 4)
}

// ---------------------------------------------------------------------------
// The catalog.
// ---------------------------------------------------------------------------

/// One standard gate: canonical name, aliases, arity, parameter count,
/// description, and the complex matrix builder.
pub(crate) struct StdGate {
    pub(crate) name: &'static str,
    pub(crate) aliases: &'static [&'static str],
    pub(crate) arity: usize,
    pub(crate) params: usize,
    pub(crate) description: &'static str,
    pub(crate) build: fn(&[f64]) -> Vec<C64>,
}

/// All standard gates shipped with the crate.
pub(crate) const STANDARD_GATES: &[StdGate] = &[
    StdGate {
        name: "id",
        aliases: &[],
        arity: 1,
        params: 0,
        description: "identity",
        build: m_id,
    },
    StdGate {
        name: "x",
        aliases: &["not"],
        arity: 1,
        params: 0,
        description: "Pauli X",
        build: m_x,
    },
    StdGate {
        name: "y",
        aliases: &[],
        arity: 1,
        params: 0,
        description: "Pauli Y",
        build: m_y,
    },
    StdGate {
        name: "z",
        aliases: &[],
        arity: 1,
        params: 0,
        description: "Pauli Z",
        build: m_z,
    },
    StdGate {
        name: "h",
        aliases: &[],
        arity: 1,
        params: 0,
        description: "Hadamard",
        build: m_h,
    },
    StdGate {
        name: "s",
        aliases: &[],
        arity: 1,
        params: 0,
        description: "phase S = sqrt(Z)",
        build: m_s,
    },
    StdGate {
        name: "sdg",
        aliases: &[],
        arity: 1,
        params: 0,
        description: "S dagger",
        build: m_sdg,
    },
    StdGate {
        name: "t",
        aliases: &[],
        arity: 1,
        params: 0,
        description: "T = sqrt(S)",
        build: m_t,
    },
    StdGate {
        name: "tdg",
        aliases: &[],
        arity: 1,
        params: 0,
        description: "T dagger",
        build: m_tdg,
    },
    StdGate {
        name: "sx",
        aliases: &[],
        arity: 1,
        params: 0,
        description: "sqrt(X)",
        build: m_sx,
    },
    StdGate {
        name: "sxdg",
        aliases: &[],
        arity: 1,
        params: 0,
        description: "sqrt(X) dagger",
        build: m_sxdg,
    },
    StdGate {
        name: "rx",
        aliases: &[],
        arity: 1,
        params: 1,
        description: "X rotation exp(-i θ X/2)",
        build: m_rx,
    },
    StdGate {
        name: "ry",
        aliases: &[],
        arity: 1,
        params: 1,
        description: "Y rotation exp(-i θ Y/2)",
        build: m_ry,
    },
    StdGate {
        name: "rz",
        aliases: &[],
        arity: 1,
        params: 1,
        description: "Z rotation exp(-i θ Z/2)",
        build: m_rz,
    },
    StdGate {
        name: "p",
        aliases: &["phase"],
        arity: 1,
        params: 1,
        description: "phase gate diag(1, e^{iλ})",
        build: m_p,
    },
    StdGate {
        name: "u",
        aliases: &["u3"],
        arity: 1,
        params: 3,
        description: "generic 1q gate u(θ, φ, λ)",
        build: m_u,
    },
    StdGate {
        name: "cx",
        aliases: &["cnot"],
        arity: 2,
        params: 0,
        description: "controlled X (control first)",
        build: m_cx,
    },
    StdGate {
        name: "cy",
        aliases: &[],
        arity: 2,
        params: 0,
        description: "controlled Y",
        build: m_cy,
    },
    StdGate {
        name: "cz",
        aliases: &[],
        arity: 2,
        params: 0,
        description: "controlled Z",
        build: m_cz,
    },
    StdGate {
        name: "ch",
        aliases: &[],
        arity: 2,
        params: 0,
        description: "controlled H",
        build: m_ch,
    },
    StdGate {
        name: "cp",
        aliases: &["cphase"],
        arity: 2,
        params: 1,
        description: "controlled phase",
        build: m_cp,
    },
    StdGate {
        name: "crx",
        aliases: &[],
        arity: 2,
        params: 1,
        description: "controlled X rotation",
        build: m_crx,
    },
    StdGate {
        name: "cry",
        aliases: &[],
        arity: 2,
        params: 1,
        description: "controlled Y rotation",
        build: m_cry,
    },
    StdGate {
        name: "crz",
        aliases: &[],
        arity: 2,
        params: 1,
        description: "controlled Z rotation",
        build: m_crz,
    },
    StdGate {
        name: "swap",
        aliases: &[],
        arity: 2,
        params: 0,
        description: "swap",
        build: m_swap,
    },
    StdGate {
        name: "iswap",
        aliases: &[],
        arity: 2,
        params: 0,
        description: "imaginary swap",
        build: m_iswap,
    },
    StdGate {
        name: "rxx",
        aliases: &[],
        arity: 2,
        params: 1,
        description: "XX rotation exp(-i θ XX/2)",
        build: m_rxx,
    },
    StdGate {
        name: "ryy",
        aliases: &[],
        arity: 2,
        params: 1,
        description: "YY rotation exp(-i θ YY/2)",
        build: m_ryy,
    },
    StdGate {
        name: "rzz",
        aliases: &[],
        arity: 2,
        params: 1,
        description: "ZZ rotation exp(-i θ ZZ/2)",
        build: m_rzz,
    },
    StdGate {
        name: "ccx",
        aliases: &["toffoli"],
        arity: 3,
        params: 0,
        description: "Toffoli (controls first)",
        build: m_ccx,
    },
    StdGate {
        name: "ccz",
        aliases: &[],
        arity: 3,
        params: 0,
        description: "doubly controlled Z",
        build: m_ccz,
    },
    StdGate {
        name: "cswap",
        aliases: &["fredkin"],
        arity: 3,
        params: 0,
        description: "Fredkin (control first)",
        build: m_cswap,
    },
];

/// Build a standard gate's matrix over `S`, if the algebra supports it.
pub(crate) fn build_std<S: Scalar>(g: &StdGate, params: &[f64]) -> Result<GateMatrix<S>> {
    let entries = (g.build)(params);
    GateMatrix::try_from_c64s(1 << g.arity, &entries).ok_or_else(|| Error::UnsupportedForAlgebra {
        gate: g.name.to_string(),
        algebra: S::algebra_name(),
    })
}

/// Install every standard gate that exists over `S` into `reg`, with aliases.
/// Gates that do not embed in the algebra are silently skipped; use
/// [`GateRegistry::names`] to see what survived.
pub fn install_standard<S: Scalar>(reg: &mut GateRegistry<S>) -> Result<()> {
    for g in STANDARD_GATES {
        // Probe at generic angles: if the matrix does not embed there, the
        // gate family is considered unsupported over S.
        if build_std::<S>(g, &PROBE_PARAMS[..g.params]).is_err() {
            continue;
        }
        let build = g.build;
        let (name, arity) = (g.name, g.arity);
        reg.register_parametric(g.name, g.description, g.arity, g.params, move |p| {
            let entries = build(p);
            GateMatrix::try_from_c64s(1 << arity, &entries).ok_or_else(|| {
                Error::UnsupportedForAlgebra {
                    gate: name.to_string(),
                    algebra: S::algebra_name(),
                }
            })
        })?;
        for alias in g.aliases {
            reg.alias(*alias, g.name)?;
        }
    }
    Ok(())
}
