//! Up-embedded readout: dis-entangling magic into a higher dimension.
//!
//! [`query::ConeResolver`](crate::query::ConeResolver) buys exactly
//! nothing on an all-to-all circuit — one such layer puts every qubit in
//! every cone, so the "compression" is `1.0×` and the report says so.
//! That is an honest boundary, but it is a boundary of the *frame*, not
//! of the problem. If an observable's cone can swallow the whole
//! register, the frame is wrong.
//!
//! The corrective move is to go **up**. Each magic event gets one fresh
//! wire via the T-gadget,
//!
//! ```text
//! T|ψ⟩ = √2 · ⟨0|_a CX_{q→a} (|ψ⟩ ⊗ |A⟩),      |A⟩ = T|+⟩
//! ```
//!
//! after which the entire dynamics on `m + t` wires is **Clifford**.
//! Every Pauli line transports to exactly one Pauli line; support never
//! branches, nonlocal circuits included. The rotation that split lines
//! downstairs is, one dimension up, a plain frame-preserving move — the
//! magic has been relocated out of the dynamics and into *boundary data*
//! on the new wires.
//!
//! ## What readout then costs
//!
//! Post-selecting the ancillas is `Π_a (I + Z_a)/2`, so
//!
//! ```text
//! ⟨O⟩ = Σ_{S ⊆ ancillas} ⟨Φ| V†(O · Z_S) V |Φ⟩,   |Φ⟩ = |0^m⟩ ⊗ |A⟩^t
//! ```
//!
//! and conjugation is multiplicative, so `V†(O·Z_S)V` is a *product* of
//! transported lines. Only `t + 1` single-line transports are ever
//! performed — `O` once and each ancilla's `Z` once — no matter how many
//! subsets the sum ranges over.
//!
//! `|Φ⟩` is a product state, so the contraction factorizes wire by wire,
//! and the subset sum therefore splits over the **connected components
//! of the transported lines' overlap graph**:
//!
//! ```text
//! cost = Σ_{components} 2^{ancillas in component}
//! ```
//!
//! That exponent is neither the width nor the T-count: it is the
//! *magic-adjacency* of the observable in the dis-entangled geometry. A
//! circuit whose naive cone covers everything still reads out cheaply
//! when its clusters stay small, and the cluster spectrum is itself a
//! measurement of how the magic entangles.
//!
//! The arithmetic is float-free. Every quantity in the readout lives in
//! `ℤ[ω]/√2^k` ([`exact::DOmega`](crate::exact::DOmega)), so the answer
//! is an exact ring element that is only converted to a double when a
//! caller asks for one.
//!
//! Ported from the `octonion_triality` research package's `upembed`
//! module.

use std::collections::HashMap;

use crate::backend::CliffordStep;
use crate::circuit::{Circuit, Op};
use crate::error::{Error, Result};
use crate::exact::DOmega;
use crate::gates::Pauli;
use crate::heisenberg::Rotation;
use crate::pathsum::{turn_from_radians, Mask, EIGHTH};
use crate::query::{Answer, Cost, Resolver};
use crate::scalar::C64;

/// The largest magic cluster this module will enumerate.
///
/// **This module has no ceiling on the register.** The register is not
/// what this bounds — a cluster of `MAX_CLUSTER` ancillas already asks
/// for `2^62` boundary contractions, so the refusal is a statement about
/// the *observable's magic separability*, which no amount of width
/// changes in either direction. `n` may be 4096 or 10^6; only the
/// clusters have a maximum, and hitting it means the answer is genuinely
/// hard rather than that the representation ran out of bits.
///
/// It is named and public so that `grep 'pub const MAX'` turns up every
/// ceiling in the crate, this one included. A limit that cannot be found
/// is worse than a limit.
pub const MAX_CLUSTER: usize = 62;

// ── Pauli lines in the raw convention ────────────────────────────────

/// A Pauli line: `ω^{2q} · X^x Z^z`, with the phase in quarter turns.
///
/// The *raw* convention, deliberately — not the Hermitian
/// `i^{|x∧z|}X^xZ^z` the rest of the crate uses. The boundary
/// contraction below evaluates `⟨A|XZ|A⟩` per wire, so carrying the
/// factor separately is what makes the per-wire factorization a plain
/// lookup instead of a normalization argument.
///
/// The supports are unbounded bitsets rather than `u64`. That is not
/// incidental: the crate's `PauliString` caps a register at 64, and a
/// module whose entire claim is that cost stops tracking the width has
/// no business inheriting a ceiling on the width. Nothing here is
/// `O(2^n)` or even `O(n)` per step, so there is no reason for `n` to
/// have a maximum.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Line {
    x: Mask,
    z: Mask,
    /// Phase in quarter turns, `0..4`.
    quarters: u32,
}

/// Set or clear one bit.
fn put(m: &mut Mask, i: usize, on: bool) {
    if on {
        m.set(i);
    } else {
        m.clear(i);
    }
}

impl Line {
    fn identity() -> Line {
        Line {
            x: Mask::zero(),
            z: Mask::zero(),
            quarters: 0,
        }
    }

    fn support(&self) -> Mask {
        self.x.or(&self.z)
    }

    /// `X^{x1}Z^{z1} · X^{x2}Z^{z2} = (−1)^{|z1∧x2|} X^{x1⊕x2}Z^{z1⊕z2}`.
    fn times(&self, other: &Line) -> Line {
        let swaps = self.z.and(&other.x).count() as u32;
        Line {
            x: self.x.xor(&other.x),
            z: self.z.xor(&other.z),
            quarters: (self.quarters + other.quarters + 2 * swaps) % 4,
        }
    }

    /// `V† · self · V` for `V = steps[0] … steps[last]` applied in order.
    ///
    /// Because every step is Clifford the line never branches — which is
    /// the whole point of having gone up a dimension — so this is a
    /// constant amount of bit-twiddling per step, whatever the width.
    ///
    /// Works in the Hermitian convention `i^{|x∧z|}X^xZ^z` with a `±`
    /// sign, the same one [`conjugate_by_step`] uses, so the two agree
    /// wire for wire wherever both apply; the raw phase is converted
    /// back out at the end. `contract_transport_matches_the_crates_own`
    /// pins that agreement against the crate's `u64` implementation.
    fn transport(&self, steps: &[CliffordStep]) -> Line {
        let (mut x, mut z) = (self.x.clone(), self.z.clone());
        let mut neg = false;
        let before = x.and(&z).count() as u32;
        for step in steps.iter().rev() {
            match *step {
                // H X H = Z, H Z H = X, H Y H = −Y.
                CliffordStep::H(a) => {
                    let (xa, za) = (x.bit(a), z.bit(a));
                    neg ^= xa && za;
                    put(&mut x, a, za);
                    put(&mut z, a, xa);
                }
                // S†XS = −Y, S†YS = X, S†ZS = Z.
                CliffordStep::S(a) => {
                    let (xa, za) = (x.bit(a), z.bit(a));
                    neg ^= xa && !za;
                    put(&mut z, a, za ^ xa);
                }
                // The standard tableau update; CX is self-inverse, so
                // the direction of conjugation does not matter here.
                CliffordStep::Cx(c, t) => {
                    let (xc, zc) = (x.bit(c), z.bit(c));
                    let (xt, zt) = (x.bit(t), z.bit(t));
                    neg ^= xc && zt && (xt == zc);
                    put(&mut x, t, xt ^ xc);
                    put(&mut z, c, zc ^ zt);
                }
            }
        }
        let after = x.and(&z).count() as u32;
        Line {
            x,
            z,
            quarters: (self.quarters + after + 4 - before % 4 + if neg { 2 } else { 0 }) % 4,
        }
    }
}

/// The wide transport on a `u64`-sized input, for the conformance test
/// that pins it against the crate's own `conjugate_by_step`.
#[doc(hidden)]
pub fn debug_transport(x: u64, z: u64, steps: &[CliffordStep]) -> (u64, u64, bool) {
    let mut line = Line::identity();
    for i in 0..64 {
        if x >> i & 1 == 1 { line.x.set(i); }
        if z >> i & 1 == 1 { line.z.set(i); }
    }
    let before = line.x.and(&line.z).count() as u32;
    let out = line.transport(steps);
    let after = out.x.and(&out.z).count() as u32;
    let mut xo = 0u64;
    let mut zo = 0u64;
    for i in out.x.iter() { xo |= 1 << i; }
    for i in out.z.iter() { zo |= 1 << i; }
    // undo the raw-convention correction to recover the Hermitian sign
    let neg = (out.quarters + 4 - (after + 4 - before % 4) % 4) % 4 == 2;
    (xo, zo, neg)
}

// ── the embedding ────────────────────────────────────────────────────

/// One ancilla wire, and which magic event it absorbed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Ancilla {
    /// Wire index in the enlarged register.
    pub wire: usize,
    /// Data qubit whose `T` this gadget replaced.
    pub site: usize,
    /// Whether the event was `T†` rather than `T` — the boundary state
    /// is the conjugate one.
    pub dagger: bool,
}

/// A Clifford+T circuit re-expressed as an all-Clifford circuit one
/// dimension up, plus boundary data on the new wires.
#[derive(Clone, Debug)]
pub struct UpEmbedding {
    data: usize,
    steps: Vec<CliffordStep>,
    ancillas: Vec<Ancilla>,
}

impl UpEmbedding {
    /// Data qubits — the width the caller asked about.
    pub fn data(&self) -> usize {
        self.data
    }

    /// Total wires after up-embedding, `m + t`.
    pub fn wires(&self) -> usize {
        self.data + self.ancillas.len()
    }

    /// Magic events relocated to the boundary.
    pub fn magic(&self) -> usize {
        self.ancillas.len()
    }

    /// The ancilla wires and what they absorbed.
    pub fn ancillas(&self) -> &[Ancilla] {
        &self.ancillas
    }

    /// Elementary Clifford steps of the enlarged circuit.
    pub fn steps(&self) -> &[CliffordStep] {
        &self.steps
    }

}

/// Re-express a Clifford+T circuit one dimension up.
///
/// Every `T` becomes a `CX` onto a fresh wire, so what comes back is
/// pure Clifford. Anything outside Clifford+T — a continuously
/// parameterised rotation, a gate whose angle is not a multiple of `π/4`
/// — is **refused by name** rather than approximated: the whole
/// construction rests on the dynamics being exactly Clifford after
/// gadgetization, and a rotation that is only nearly `T` would silently
/// void that.
pub fn gadgetize(circuit: &Circuit<C64>) -> Result<UpEmbedding> {
    let data = circuit.num_qubits();
    let mut steps: Vec<CliffordStep> = Vec::new();
    let mut ancillas: Vec<Ancilla> = Vec::new();

    // `T^k` for a dyadic phase, as gadgets plus Clifford remainder.
    fn eighths_of(theta: f64, name: &str) -> Result<u32> {
        let turn = turn_from_radians(theta).map_err(|_| {
            Error::InvalidState(format!(
                "upembed: `{name}` at {theta} rad is not a multiple of π/4, \
                 so it is not Clifford+T; refused rather than approximated"
            ))
        })?;
        if turn % EIGHTH != 0 {
            return Err(Error::InvalidState(format!(
                "upembed: `{name}` at {theta} rad is dyadic but finer than π/4, \
                 so it is outside Clifford+T"
            )));
        }
        Ok((turn / EIGHTH) as u32 % 8)
    }

    let magic = |steps: &mut Vec<CliffordStep>,
                     ancillas: &mut Vec<Ancilla>,
                     q: usize,
                     dagger: bool|
     -> Result<()> {
        let wire = data + ancillas.len();
        steps.push(CliffordStep::Cx(q, wire));
        ancillas.push(Ancilla {
            wire,
            site: q,
            dagger,
        });
        Ok(())
    };

    for op in circuit.ops() {
        let Op::Named {
            name,
            params,
            qubits: qs,
        } = op
        else {
            return Err(Error::InvalidState(
                "upembed: only named registry gates gadgetize".into(),
            ));
        };
        let check = |q: usize| -> Result<usize> {
            if q >= data {
                Err(Error::InvalidState(format!(
                    "upembed: qubit {q} outside a {data}-qubit register"
                )))
            } else {
                Ok(q)
            }
        };
        for &q in qs {
            check(q)?;
        }
        let theta = params.first().copied().unwrap_or(0.0);
        match (name.as_str(), qs.len()) {
            ("id", _) => {}
            ("h", 1) => steps.push(CliffordStep::H(qs[0])),
            ("s", 1) => steps.push(CliffordStep::S(qs[0])),
            ("sdg", 1) => steps.extend([CliffordStep::S(qs[0]); 3]),
            ("z", 1) => steps.extend([CliffordStep::S(qs[0]); 2]),
            ("x" | "not", 1) => steps.extend([
                CliffordStep::H(qs[0]),
                CliffordStep::S(qs[0]),
                CliffordStep::S(qs[0]),
                CliffordStep::H(qs[0]),
            ]),
            // Y ∝ X·Z, and a global phase never reaches an expectation.
            ("y", 1) => steps.extend([
                CliffordStep::S(qs[0]),
                CliffordStep::S(qs[0]),
                CliffordStep::H(qs[0]),
                CliffordStep::S(qs[0]),
                CliffordStep::S(qs[0]),
                CliffordStep::H(qs[0]),
            ]),
            // √X ∝ H·S·H.
            ("sx", 1) => steps.extend([
                CliffordStep::H(qs[0]),
                CliffordStep::S(qs[0]),
                CliffordStep::H(qs[0]),
            ]),
            ("sxdg", 1) => steps.extend([
                CliffordStep::H(qs[0]),
                CliffordStep::S(qs[0]),
                CliffordStep::S(qs[0]),
                CliffordStep::S(qs[0]),
                CliffordStep::H(qs[0]),
            ]),
            ("t", 1) => magic(&mut steps, &mut ancillas, qs[0], false)?,
            ("tdg", 1) => magic(&mut steps, &mut ancillas, qs[0], true)?,
            // A dyadic Z-rotation is S and T powers; the leading global
            // phase of `rz` drops out of every expectation.
            ("p" | "phase" | "rz", 1) => {
                for _ in 0..eighths_of(theta, name)? {
                    magic(&mut steps, &mut ancillas, qs[0], false)?;
                }
            }
            ("cx" | "cnot", 2) => steps.push(CliffordStep::Cx(qs[0], qs[1])),
            ("cz", 2) => steps.extend([
                CliffordStep::H(qs[1]),
                CliffordStep::Cx(qs[0], qs[1]),
                CliffordStep::H(qs[1]),
            ]),
            ("swap", 2) => steps.extend([
                CliffordStep::Cx(qs[0], qs[1]),
                CliffordStep::Cx(qs[1], qs[0]),
                CliffordStep::Cx(qs[0], qs[1]),
            ]),
            _ => {
                return Err(Error::InvalidState(format!(
                    "upembed: `{name}` on {} qubits is outside Clifford+T",
                    qs.len()
                )))
            }
        }
    }
    Ok(UpEmbedding {
        data,
        steps,
        ancillas,
    })
}

// ── the boundary ─────────────────────────────────────────────────────

/// `⟨Φ| X^x Z^z |Φ⟩` for `|Φ⟩ = |0^m⟩ ⊗ |A⟩^t`, wire by wire.
///
/// Every factor is exact: `⟨0|X|0⟩ = 0`, `⟨0|Z|0⟩ = 1`, `⟨A|X|A⟩ =
/// 1/√2`, `⟨A|Z|A⟩ = 0`, `⟨A|XZ|A⟩ = ∓i/√2`. A single vanishing wire
/// kills the whole line, which is why most of the subset sum costs
/// nothing.
fn boundary(line: &Line, emb: &UpEmbedding) -> Result<DOmega> {
    // Any X on a data wire kills the line: ⟨0|X|0⟩ = 0.
    if line.x.any_below(emb.data) {
        return Ok(DOmega::zero());
    }
    let mut val = DOmega::omega_pow(2 * line.quarters as i64);
    for a in &emb.ancillas {
        match (line.x.bit(a.wire), line.z.bit(a.wire)) {
            (false, false) => {}
            // ⟨A|Z|A⟩ = 0 — the ancilla's Bloch vector has no z part.
            (false, true) => return Ok(DOmega::zero()),
            // ⟨A|X|A⟩ = 1/√2.
            (true, false) => val = val.mul(DOmega::inv_sqrt2_pow(1))?,
            // ⟨A|XZ|A⟩ = −i/√2 for T, +i/√2 for T†.
            _ => {
                let sign = if a.dagger { 2 } else { 6 };
                val = val
                    .mul(DOmega::omega_pow(sign))?
                    .mul(DOmega::inv_sqrt2_pow(1))?;
            }
        }
    }
    Ok(val)
}

// ── the readout ──────────────────────────────────────────────────────

/// What an up-embedded readout cost and how the magic clustered.
#[derive(Clone, Debug)]
pub struct Readout {
    /// The expectation, exactly.
    pub exact: DOmega,
    /// The expectation as a double.
    pub value: f64,
    /// Ancillas per connected component of the transported lines'
    /// overlap graph, descending — the observable's magic-adjacency
    /// spectrum in the dis-entangled geometry.
    pub clusters: Vec<usize>,
    /// Boundary contractions performed: `Σ_components 2^{ancillas}`.
    /// The number that decides whether this was sub-exponential.
    pub terms: u128,
    /// Single-line Clifford transports performed — always `t + 1`,
    /// however large the subset sum.
    pub transports: usize,
    /// Elementary Clifford steps each transport walked.
    pub steps: usize,
}

impl Readout {
    /// The governing exponent: the largest cluster.
    pub fn max_cluster(&self) -> usize {
        self.clusters.first().copied().unwrap_or(0)
    }

    /// How many times smaller the subset sum was than the flat
    /// `2^t` it would have been without the factorization.
    pub fn factorization_gain(&self) -> f64 {
        let flat = 2f64.powi(self.clusters.iter().sum::<usize>() as i32);
        flat / (self.terms.max(1) as f64)
    }
}

/// Transport the observable and every ancilla `Z` through the lifted
/// Clifford frame, then partition the resulting lines into the connected
/// components of their overlap graph.
///
/// Returns the transported observable, all lines (index 0 is the
/// observable), and the components as index groups.
fn transport_and_partition(obs: Line, emb: &UpEmbedding) -> (Line, Vec<Line>, Vec<Vec<usize>>) {
    let obs = obs.transport(&emb.steps);
    let mut lines: Vec<Line> = vec![obs.clone()];
    lines.extend(emb.ancillas.iter().map(|a| {
        Line {
            x: Mask::zero(),
            z: Mask::single(a.wire),
            quarters: 0,
        }
        .transport(&emb.steps)
    }));
    let mut parent: Vec<usize> = (0..lines.len()).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let supports: Vec<Mask> = lines.iter().map(|l| l.support()).collect();
    for i in 0..lines.len() {
        for j in (i + 1)..lines.len() {
            if supports[i].intersects(&supports[j]) {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                parent[a] = b;
            }
        }
    }
    let mut comps: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..lines.len() {
        let r = find(&mut parent, i);
        comps.entry(r).or_default().push(i);
    }
    (obs, lines, comps.into_values().collect())
}

/// The magic-adjacency spectrum of an observable **without paying for
/// the readout**: transports and partitions, then stops.
///
/// [`cluster_readout`] returns the same spectrum, but only after
/// contracting `Σ 2^{cluster}` boundary terms — so on a circuit whose
/// clusters are large it cannot report *how* large before running out
/// of time. This is the estimate before the cost, and it is cheap:
/// `t + 1` line transports and a union–find, with no `2^k` anywhere.
///
/// Returns the cluster sizes descending. `Σ 2^{size}` is what a readout
/// would cost, and the largest entry is the governing exponent.
pub fn cluster_spectrum(emb: &UpEmbedding, ops: &[(usize, Pauli)]) -> Result<Vec<usize>> {
    let mut obs = Line::identity();
    for &(q, p) in ops {
        if q >= emb.data {
            return Err(Error::InvalidState(format!(
                "upembed: observable on qubit {q} outside a {}-qubit register",
                emb.data
            )));
        }
        let bit = Mask::single(q);
        let factor = match p {
            Pauli::I => continue,
            Pauli::X => Line {
                x: bit,
                z: Mask::zero(),
                quarters: 0,
            },
            Pauli::Z => Line {
                x: Mask::zero(),
                z: bit,
                quarters: 0,
            },
            Pauli::Y => Line {
                x: bit.clone(),
                z: bit,
                quarters: 1,
            },
        };
        obs = obs.times(&factor);
    }
    let (_, _, members) = transport_and_partition(obs, emb);
    let mut sizes: Vec<usize> = members
        .into_iter()
        .map(|g| g.iter().filter(|&&i| i > 0).count())
        .collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    Ok(sizes)
}

/// The circuit's **magic axes**: each `T`'s rotation axis transported
/// through the lifted Clifford frame, as `(x, z)` supports over the
/// lifted register.
///
/// This is the raw geometry the readout factorization is computed from,
/// exposed because the factorization is not the only thing worth asking
/// of it. Two other partitions live on the same set of lines and are
/// strictly sharper than support-overlap:
///
/// * the **symplectic span** — the `𝔽₂` rank of the lines' `(x‖z)`
///   vectors. The Pauli group generated by `t` axes has `2^rank`
///   elements and never more than `4^n`, so this rank, not `t`, bounds
///   how many distinct Paulis any expansion of `∏ exp(−iθQ_a)` can
///   reach;
/// * the **anticommutation graph** — an edge wherever two axes
///   anticommute. Distinct components are symplectically orthogonal and
///   the evolution factorizes across them however much their supports
///   overlap ([`crate::coupling`]), which support-overlap cannot see.
///
/// `t + 1` transports, no `2^k` anywhere.
pub fn magic_axes(emb: &UpEmbedding) -> Vec<(Mask, Mask)> {
    emb.ancillas
        .iter()
        .map(|a| {
            let l = Line {
                x: Mask::zero(),
                z: Mask::single(a.wire),
                quarters: 0,
            }
            .transport(&emb.steps);
            (l.x, l.z)
        })
        .collect()
}

/// Components of the magic's **own** overlap graph over the data
/// register — separability with no observable in it.
///
/// [`cluster_spectrum`] answers "what does this readout cost", and the
/// observable is part of that: a weight-1 `Z` at the end of a deep
/// circuit has a backward cone covering every qubit, so it joins every
/// component and the spectrum collapses to one. That is the honest cost
/// of *that question*, but it hides whether the magic itself is
/// separable.
///
/// This drops the observable and asks only whether the `T` gates reach
/// each other. Two are linked when their transported axes share a data
/// qubit, and `Σ 2^{component}` is what the magic would cost given an
/// observable local enough not to bridge them.
///
/// The reach is **backward**. The frame carries `V† Z_a V`, so a `T`'s
/// axis propagates toward the circuit's *input*: magic at layer `L` has
/// a cone about `2L` wide, and it is EARLY magic that is narrow. Narrow
/// is necessary and not sufficient — on a line, magic on every qubit
/// chains into one component however narrow each cone is — so
/// separation needs a spatial gap wider than `2L` as well. Both halves
/// of that rule are measured in `examples/dcs_separability.rs`.
pub fn magic_components(emb: &UpEmbedding, data_qubits: usize) -> Vec<usize> {
    let axes = magic_axes(emb);
    let supports: Vec<Mask> = axes
        .iter()
        .map(|(x, z)| {
            let mut m = Mask::zero();
            for q in 0..data_qubits {
                if x.bit(q) || z.bit(q) {
                    m.set(q);
                }
            }
            m
        })
        .collect();
    let m = supports.len();
    let mut parent: Vec<usize> = (0..m).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for i in 0..m {
        for j in (i + 1)..m {
            if supports[i].intersects(&supports[j]) {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                parent[a] = b;
            }
        }
    }
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for i in 0..m {
        let r = find(&mut parent, i);
        *sizes.entry(r).or_insert(0) += 1;
    }
    let mut out: Vec<usize> = sizes.into_values().collect();
    out.sort_unstable_by(|a, b| b.cmp(a));
    out
}

/// `⟨0^m| U† O U |0^m⟩` through the up-embedded frame.
///
/// `t + 1` Clifford transports, then a contraction factorized over the
/// connected components of the transported lines. Exact — no dense
/// state and no floating point until the caller asks for a double.
pub fn cluster_readout(emb: &UpEmbedding, ops: &[(usize, Pauli)]) -> Result<Readout> {
    let mut obs = Line::identity();
    for &(q, p) in ops {
        if q >= emb.data {
            return Err(Error::InvalidState(format!(
                "upembed: observable on qubit {q} outside a {}-qubit register",
                emb.data
            )));
        }
        // Y = i·X·Z, so it carries a quarter turn of its own.
        let bit = Mask::single(q);
        let factor = match p {
            Pauli::I => continue,
            Pauli::X => Line {
                x: bit,
                z: Mask::zero(),
                quarters: 0,
            },
            Pauli::Z => Line {
                x: Mask::zero(),
                z: bit,
                quarters: 0,
            },
            Pauli::Y => Line {
                x: bit.clone(),
                z: bit,
                quarters: 1,
            },
        };
        obs = obs.times(&factor);
    }

    let (obs, lines, mut members) = transport_and_partition(obs, emb);

    // Each component contributes its own subset sum, and the product of
    // the components is the whole sum — the supports are disjoint, so
    // `⟨Φ|·|Φ⟩` factorizes across them with no cross terms.
    let mut total = DOmega::int(1);
    let mut clusters = Vec::new();
    let mut terms: u128 = 0;
    members.sort();
    for group in members {
        let ancs: Vec<usize> = group.iter().copied().filter(|&i| i > 0).collect();
        if ancs.len() > MAX_CLUSTER {
            return Err(Error::TooManyQubits {
                requested: ancs.len(),
                max: MAX_CLUSTER,
            });
        }
        clusters.push(ancs.len());
        terms += 1u128 << ancs.len();
        let mut part = DOmega::zero();
        for subset in 0..(1u64 << ancs.len()) {
            let mut cur = if group.contains(&0) {
                obs.clone()
            } else {
                Line::identity()
            };
            for (b, &idx) in ancs.iter().enumerate() {
                if subset >> b & 1 == 1 {
                    cur = cur.times(&lines[idx]);
                }
            }
            part = part.add(boundary(&cur, emb)?)?;
        }
        total = total.mul(part)?;
    }
    clusters.sort_unstable_by(|a, b| b.cmp(a));

    Ok(Readout {
        value: total.to_c64().re,
        exact: total,
        clusters,
        terms,
        transports: lines.len(),
        steps: emb.steps.len(),
    })
}

/// `⟨0^m| U† O U |0^m⟩` straight from a circuit.
pub fn expectation(circuit: &Circuit<C64>, ops: &[(usize, Pauli)]) -> Result<Readout> {
    cluster_readout(&gadgetize(circuit)?, ops)
}

// ── as a resolver ────────────────────────────────────────────────────

/// The up-embedded frame as a [`Resolver`], so it can be priced in the
/// same currency as every other way of answering the question.
///
/// This is the one that answers where
/// [`ConeResolver`](crate::query::ConeResolver) cannot: on an all-to-all
/// circuit the cone reports `1.0×` compression because there is no
/// outside, while this resolver's exponent is the observable's magic
/// adjacency, which does not know how wide the register is.
pub struct UpEmbedResolver {
    circuit: Circuit<C64>,
    embedding: Option<UpEmbedding>,
    /// The last readout's cluster spectrum, for reports.
    pub last_clusters: Vec<usize>,
    name: String,
}

impl UpEmbedResolver {
    /// Record a circuit. Nothing is gadgetized and nothing is
    /// transported until a question arrives.
    pub fn new(circuit: Circuit<C64>) -> Self {
        UpEmbedResolver {
            circuit,
            embedding: None,
            last_clusters: Vec::new(),
            name: "upembed".into(),
        }
    }

    fn embedding(&mut self) -> Result<&UpEmbedding> {
        if self.embedding.is_none() {
            self.embedding = Some(gadgetize(&self.circuit)?);
        }
        Ok(self.embedding.as_ref().expect("just built"))
    }

    /// The magic-adjacency spectrum for an observable, without
    /// contracting anything — the cost estimate before the cost.
    pub fn clusters(&mut self, ops: &[(usize, Pauli)]) -> Result<Vec<usize>> {
        let emb = self.embedding()?.clone();
        Ok(cluster_readout(&emb, ops)?.clusters)
    }

    fn answer(&mut self, ops: &[(usize, Pauli)], circuit: Option<Circuit<C64>>) -> Result<Answer> {
        let emb = match circuit {
            Some(c) => gadgetize(&c)?,
            None => self.embedding()?.clone(),
        };
        let r = cluster_readout(&emb, ops)?;
        let cost = Cost {
            qubits: r.max_cluster(),
            width: emb.data(),
            gates: r.transports * r.steps,
            bytes: (r.terms as usize).saturating_mul(std::mem::size_of::<DOmega>()),
        };
        self.last_clusters = r.clusters;
        Ok(Answer {
            value: r.value,
            cost,
        })
    }
}

/// A perturbation this frame can absorb: a rotation by a multiple of a
/// half turn about a single-qubit axis, which is Clifford.
fn clifford_gate_for(rot: &Rotation) -> Option<(&'static str, usize)> {
    let (x, z) = rot.axis;
    if (x | z).count_ones() != 1 {
        return None;
    }
    let q = (x | z).trailing_zeros() as usize;
    // exp(−iθP/2) at θ = π is P itself up to phase; at θ = π/2 it is the
    // square root, which is still Clifford.
    let quarter = (rot.theta / std::f64::consts::FRAC_PI_2).round();
    if (rot.theta - quarter * std::f64::consts::FRAC_PI_2).abs() > 1e-12 {
        return None;
    }
    let axis = match (x >> q & 1, z >> q & 1) {
        (1, 0) => "x",
        (0, 1) => "z",
        _ => "y",
    };
    match quarter.rem_euclid(4.0) as u32 {
        0 => Some(("id", q)),
        2 => Some((axis, q)),
        // √X is `sx`; √Z is `s`; √Y has no standard letter here.
        1 | 3 if axis == "x" => Some(("sx", q)),
        1 | 3 if axis == "z" => Some(("s", q)),
        _ => None,
    }
}

impl Resolver for UpEmbedResolver {
    fn name(&self) -> &str {
        &self.name
    }

    fn width(&self) -> usize {
        self.circuit.num_qubits()
    }

    fn expectation(&mut self, ops: &[(usize, Pauli)]) -> Result<Answer> {
        self.answer(ops, None)
    }

    fn perturbed_expectation(
        &mut self,
        at_gate: usize,
        site: usize,
        perturbation: Rotation,
        ops: &[(usize, Pauli)],
    ) -> Result<Answer> {
        let Some((gate, q)) = clifford_gate_for(&perturbation) else {
            return Err(Error::InvalidState(format!(
                "upembed: a perturbation at θ = {} about a {}-site axis is not Clifford; \
                 this frame absorbs only Clifford perturbations",
                perturbation.theta,
                (perturbation.axis.0 | perturbation.axis.1).count_ones()
            )));
        };
        if q != site {
            return Err(Error::InvalidState(format!(
                "upembed: perturbation axis sits on qubit {q}, not the named site {site}"
            )));
        }
        let mut c = Circuit::new(self.circuit.num_qubits());
        for (i, op) in self.circuit.ops().iter().enumerate() {
            if i == at_gate {
                c.gate(gate, vec![], vec![q]);
            }
            match op {
                Op::Named {
                    name,
                    params,
                    qubits,
                } => {
                    c.gate(name, params.clone(), qubits.clone());
                }
                _ => {
                    return Err(Error::InvalidState(
                        "upembed: only named registry gates gadgetize".into(),
                    ))
                }
            }
        }
        if at_gate >= self.circuit.ops().len() {
            c.gate(gate, vec![], vec![q]);
        }
        self.answer(ops, Some(c))
    }
}

// ── handing the Clifford result to another representation ────────────

impl UpEmbedding {
    /// The up-embedded dynamics as an ordinary [`Circuit`] — all
    /// Clifford, on [`UpEmbedding::wires`] wires.
    ///
    /// The point of being able to hand it over: a *second* representation
    /// can then check the claim. `PathSum` reduces every Clifford circuit
    /// to `h* = 0` from rewrite rules alone, so running this through it
    /// certifies "the dynamics is Clifford" without trusting this
    /// module's own bookkeeping.
    pub fn to_circuit(&self) -> Circuit<C64> {
        let mut c = Circuit::new(self.wires());
        for step in &self.steps {
            match *step {
                CliffordStep::H(a) => c.gate("h", vec![], vec![a]),
                CliffordStep::S(a) => c.gate("s", vec![], vec![a]),
                CliffordStep::Cx(a, b) => c.gate("cx", vec![], vec![a, b]),
            };
        }
        c
    }
}

/// Gadgetize only the **first `count`** magic events, leaving the rest
/// in place — the partial move.
///
/// Full [`gadgetize`] relocates every `T` to a fresh wire and the
/// dynamics becomes Clifford by construction, which is true but says
/// nothing about how many wires were *needed*. Gadgetizing a prefix and
/// asking a reducer what is left turns that into a measurement: spend
/// wires one at a time and watch the surviving magic fall.
///
/// Returns the circuit on `qubits + min(count, t)` wires. Ancillas carry
/// the magic as boundary data and are not prepared here — this is the
/// dynamics, which is the part whose Clifford-ness is in question.
pub fn gadgetize_partial(circuit: &Circuit<C64>, count: usize) -> Result<Circuit<C64>> {
    let data = circuit.num_qubits();
    let total = magic_events(circuit)?;
    let take = count.min(total);
    let mut out = Circuit::new(data + take);
    let mut used = 0usize;
    for op in circuit.ops() {
        let Op::Named {
            name,
            params,
            qubits: qs,
        } = op
        else {
            return Err(Error::InvalidState(
                "upembed: only named registry gates gadgetize".into(),
            ));
        };
        let is_magic = matches!(name.as_str(), "t" | "tdg") && qs.len() == 1;
        if is_magic && used < take {
            out.gate("cx", vec![], vec![qs[0], data + used]);
            used += 1;
        } else {
            out.gate(name, params.clone(), qs.clone());
        }
    }
    Ok(out)
}

/// Magic events in a circuit — the `T` and `T†` gates a full
/// gadgetization would each spend a wire on.
pub fn magic_events(circuit: &Circuit<C64>) -> Result<usize> {
    let mut n = 0usize;
    for op in circuit.ops() {
        if let Op::Named { name, qubits, .. } = op {
            if matches!(name.as_str(), "t" | "tdg") && qubits.len() == 1 {
                n += 1;
            }
        }
    }
    Ok(n)
}
