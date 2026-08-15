//! **Doped Clifford sampling** — the IBM/UChicago logical-qubit
//! experiment of July 2026 ([arXiv:2607.25941]) built as a circuit
//! family, so every representation in this crate can be measured
//! against the thing itself rather than against a caricature of it.
//!
//! [arXiv:2607.25941]: https://arxiv.org/abs/2607.25941
//!
//! ## The system, as specified
//!
//! The experiment is *not* random circuit sampling. It is a **structured**
//! family, and the structure is the whole point — it is what admits the
//! spacetime code that makes the fidelity certifiable. Per the paper's
//! Figure S1, the logical circuit on `n` qubits laid out as a
//! linear-nearest-neighbour chain is:
//!
//! 1. `H` on every qubit;
//! 2. `depth` repetitions of — a **brickwork `CZ` layer** (even layer:
//!    pairs `(0,1),(2,3),…`; odd layer: `(1,2),(3,4),…`), then a layer of
//!    random single-qubit `√X` or `S·√X`;
//! 3. a final layer of random `S` or `H`, rotating the stabilizer state
//!    into graph-state form.
//!
//! That skeleton is **Clifford**, hence efficiently simulable, and the
//! experiment leans on exactly that: the undoped circuit is the trusted
//! classical baseline against which the doped circuit's fidelity is
//! certified. Hardness is then bought by **doping** — `T` gates placed on
//! wires immediately following a `CZ`, chosen to commute with the
//! spacetime code's checks so the error-detection structure survives.
//!
//! At the experiment's size the counts are forced, and they are pinned
//! here as a falsifiable check on the construction:
//!
//! ```text
//!   n = 70, depth = 70   →   2415 two-qubit gates      (paper: 2,415)
//!   doping rate 0.1938   →    468 T gates              (paper:   468)
//! ```
//!
//! `2415 = 35·35 + 35·34`: thirty-five even layers of 35 `CZ`s and
//! thirty-five odd layers of 34. The published two-qubit-gate count is a
//! *consequence* of the layout, which is what makes it worth checking —
//! a construction that misses it is not the same circuit.
//!
//! ## What is reproduced here and what is not
//!
//! The **logical** circuit is reproduced exactly. The 27 ancillas and the
//! spacetime-code checks are not: they buy fidelity on hardware and play
//! no part in the classical cost, which is what this module measures.
//! Doping sites are therefore drawn uniformly from the post-`CZ` wires
//! rather than filtered by check-commutation. That is the paper's own
//! control family — its Figure S14 compares against "the same random
//! Clifford circuit, doped at a rate of `r = 0.19` which matches our
//! experiment's doping rate" — and the paper measures the difference the
//! filter makes: constraining doping to check-valid sites moves the
//! 2-stabilizer Rényi entropy from 18.00 to 17.66 on `20×20` instances,
//! so the unconstrained family is if anything *slightly* harder, and the
//! substitution cannot flatter a classical method.
//!
//! ## Why this crate has anything to say
//!
//! The paper's own hardness analysis (its §S9) rules out the methods
//! that assume **low entanglement** (MPS: needs bond dimension `2^30`),
//! **low magic** (stabilizer decomposition: rank `2^185.5`, `2^0.327·t`
//! measured for QuiZX), the **hybrid** of the two (CAMPS: the first `n`
//! rotations disentangle free, then `2^0.41t`), and **noise** (needs
//! `ε ≈ 4·10⁻⁴`).
//!
//! [`crate::pathsum`] is on none of those axes. It holds no amplitudes,
//! no tableau and no tensors — the state is the circuit's closed form as
//! a sum over path variables, and its cost currency is `h*`, the
//! variables that survive reduction. `h*` is **zero for every Clifford
//! circuit at any width**, so the DCS skeleton is free by construction
//! rather than by a Gottesman–Knill special case, and doping enters only
//! through how much it obstructs elimination. That makes "what does
//! `h*` do on this family?" a question with a measured answer and no
//! entry in the paper's table. [`probe`] measures it, and the answer is
//! **no**: this circuit is Hadamard-dense by construction — every `√X`
//! is `H·S·H`, so the experiment carries 9904 walls — and `h*` lands in
//! the hundreds, far worse than the stabilizer rank. The axis is not
//! competitive, which the tests assert so that an improvement fails
//! them.
//!
//! ## What is worth reporting: the width ceiling
//!
//! [`rotation_axes`] rewrites the circuit as `C · ∏_a exp(−iπ/8·P_a)`
//! and asks what those 468 axes actually span. Two numbers come back in
//! milliseconds at the full 70-qubit size:
//!
//! * the rank of their **X-components** is 70, so the kernel is
//!   `468 − 70 = 398` — exactly the figure the paper's §S9.3 reports,
//!   which is the cross-check that this is the same circuit;
//! * their **symplectic rank saturates at `2n = 140`**, and has done
//!   since `n = 32`. The 468 axes generate the entire Pauli group on 70
//!   qubits.
//!
//! The second one is the finding, because it says the `T`-count has
//! stopped being the currency. A law of the form `2^{α·t}` is a
//! statement about `|T⟩^{⊗t}`, a `t`-qubit object. Once the magic is
//! injected into `n` qubits, every cost measure it feeds is capped by
//! the width instead: no `n`-qubit state has stabilizer rank above
//! `2^n`, since the computational basis is already a stabilizer
//! decomposition, and its stabilizer **extent** obeys the same bound
//! because `Σ|c_x| ≤ √(2^n · Σ|c_x|²) = 2^{n/2}`.
//!
//! So the `2^{0.3963·t} = 2^{185.5}` the paper quotes at `t = 468` sits
//! above a ceiling of `2^70`; the law crosses that ceiling at
//! `t ≈ 177`, and the QuiZX-measured `2^{0.327·t}` crosses it at
//! `t ≈ 214`. The experiment runs at more than twice either. Past the
//! crossing those extrapolations are measuring how a `T`-counting
//! algorithm degrades, not how hard the state is — and the saturated
//! symplectic rank is the structural reason why.
//!
//! This does not make the experiment classically feasible: `2^70`
//! amplitudes is still ~18 zettabytes, and the claim survives. It makes
//! the *stated margin* smaller than advertised, by about `2^115`, and
//! it says adding `T` gates past ~200 buys the construction nothing at
//! this width.
//!
//! [`Dcs::experiment`] is the system.

use crate::circuit::Circuit;
use crate::pathsum::{Mask, PathSum};
use crate::rng::Prng;
use crate::scalar::C64;
use std::time::{Duration, Instant};

/// Logical qubits in the experiment.
pub const EXPERIMENT_QUBITS: usize = 70;
/// Brickwork `CZ` depth in the experiment.
pub const EXPERIMENT_DEPTH: usize = 70;
/// Logical two-qubit operations reported by the paper.
pub const EXPERIMENT_TWO_QUBIT_GATES: usize = 2415;
/// `T` gates reported by the paper.
pub const EXPERIMENT_T_GATES: usize = 468;
/// Ancillas carrying the spacetime code (not modelled here — they buy
/// hardware fidelity, not classical cost).
pub const EXPERIMENT_ANCILLAS: usize = 27;
/// Physical qubits used on the device.
pub const EXPERIMENT_PHYSICAL_QUBITS: usize = 97;

/// Paper's stated `T`-per-two-qubit-gate doping rate (`468 / 2415`), the
/// dial its own control circuits quote as `r = 0.19`.
pub const DOPING_RATE: f64 = EXPERIMENT_T_GATES as f64 / EXPERIMENT_TWO_QUBIT_GATES as f64;

/// Two-qubit gates in an `n`-wide, `depth`-deep brickwork on a
/// linear-nearest-neighbour chain, in closed form.
///
/// Even layers pair `(0,1),(2,3),…` — `⌊n/2⌋` gates; odd layers pair
/// `(1,2),(3,4),…` — `⌊(n−1)/2⌋`. At `n = depth = 70` this is 2415.
pub fn two_qubit_gates(n: usize, depth: usize) -> usize {
    let even_layers = depth.div_ceil(2);
    let odd_layers = depth / 2;
    even_layers * (n / 2) + odd_layers * ((n.saturating_sub(1)) / 2)
}

/// The `CZ` pairs of one brickwork layer.
pub fn layer_pairs(n: usize, layer: usize) -> impl Iterator<Item = (usize, usize)> {
    let start = layer % 2;
    (0..).map_while(move |k| {
        let a = start + 2 * k;
        (a + 1 < n).then_some((a, a + 1))
    })
}

/// A doping location: the wire immediately after a `CZ`, named by the
/// `CZ`'s position in circuit order and which of its two qubits carries
/// the rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DopeSite {
    /// Brickwork layer the `CZ` sits in.
    pub layer: usize,
    /// The qubit the `T` acts on.
    pub qubit: usize,
    /// Index of the `CZ` in circuit order.
    pub cz_index: usize,
}

/// Where in the circuit the `T` gates are allowed to sit.
///
/// The paper places doping only at spacetime-code-commuting wires, and
/// records what that does to the distribution: "when constraining to the
/// locations allowed by spacetime checks, doping sites are sparse in the
/// majority of the layers, but increase in density especially in the
/// last few layers."
///
/// That is not a detail. A `T` gate's reach is its **forward light
/// cone**, and on a linear chain a cone opening at layer `L` of a
/// depth-`D` circuit is only `2(D − L)` qubits wide at the output. Magic
/// placed late cannot reach far, so late magic is *separable* magic —
/// and separability is what every factorization in this crate is
/// measured against. The profile is therefore a first-class parameter,
/// not a seed detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doping {
    /// Uniform over every post-`CZ` wire — the paper's Figure S14
    /// control, and the worst case for separability.
    Uniform,
    /// Confined to the last `layers` brickwork layers, which is the
    /// direction the paper says its check constraint pushes.
    Late {
        /// How many trailing layers may carry magic.
        layers: usize,
    },
    /// Confined to the first `layers` — the opposite extreme, kept so
    /// the comparison has both ends and not just the flattering one.
    Early {
        /// How many leading layers may carry magic.
        layers: usize,
    },
    /// Confined to `width`-wide bands of qubits separated by `gap`
    /// idle qubits, and to the last `layers` layers.
    ///
    /// The knob that actually governs separability, and it has two
    /// halves. A `T`'s axis is transported *backwards* — the frame
    /// carries `V† Z_a V`, so the reach is the **backward** cone to the
    /// circuit's input, and it is EARLY magic that is narrow, not late.
    /// Narrow is not enough on its own: on a line, magic on every qubit
    /// chains into one component however narrow each cone is. Separating
    /// needs both — early (`late = false`, cone width `2·layers`) and a
    /// spatial `gap` wider than that cone.
    Banded {
        /// Qubits per doped band.
        width: usize,
        /// Undoped qubits between bands.
        gap: usize,
        /// Layers that may carry magic, counted from `end`.
        layers: usize,
        /// Count `layers` from the circuit's end rather than its start.
        late: bool,
    },
}

/// A doped Clifford sampling instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dcs {
    /// Logical qubits (chain length).
    pub qubits: usize,
    /// Brickwork `CZ` depth.
    pub depth: usize,
    /// `T` gates doped in.
    pub t_gates: usize,
    /// Seed for the single-qubit Clifford layers and the doping draw.
    pub seed: u64,
    /// Where the magic is allowed to sit.
    pub doping: Doping,
}

impl Dcs {
    /// The experiment as run: 70 qubits, depth 70, 468 `T` gates.
    pub fn experiment() -> Dcs {
        Dcs {
            qubits: EXPERIMENT_QUBITS,
            depth: EXPERIMENT_DEPTH,
            t_gates: EXPERIMENT_T_GATES,
            seed: 2607_25941,
            doping: Doping::Uniform,
        }
    }

    /// The experiment's *shape* at width `n`: square (`depth = n`, as the
    /// paper's LNN argument requires) and doped at the experiment's rate.
    ///
    /// This is the family to scale, because both of the experiment's
    /// resources move with `n`: the two-qubit count as `n²/2` and the
    /// `T` count with it. Holding either fixed measures something the
    /// experiment is not.
    pub fn scaled(n: usize) -> Dcs {
        let t = (DOPING_RATE * two_qubit_gates(n, n) as f64).round() as usize;
        Dcs {
            qubits: n,
            depth: n,
            t_gates: t,
            seed: 2607_25941,
            doping: Doping::Uniform,
        }
    }

    /// The same instance with the doping count overridden — the magic
    /// dial at a byte-identical Clifford skeleton.
    pub fn with_t(self, t_gates: usize) -> Dcs {
        Dcs { t_gates, ..self }
    }

    /// The same instance under a different seed.
    pub fn with_seed(self, seed: u64) -> Dcs {
        Dcs { seed, ..self }
    }

    /// The same instance with the magic confined differently.
    pub fn with_doping(self, doping: Doping) -> Dcs {
        Dcs { doping, ..self }
    }

    /// Two-qubit gates in this instance.
    pub fn two_qubit_gates(&self) -> usize {
        two_qubit_gates(self.qubits, self.depth)
    }

    /// `T` gates per two-qubit gate.
    pub fn doping_rate(&self) -> f64 {
        self.t_gates as f64 / self.two_qubit_gates() as f64
    }

    /// Every wire immediately following a `CZ` — the candidate doping
    /// locations, in circuit order.
    pub fn dope_sites(&self) -> Vec<DopeSite> {
        let mut out = Vec::with_capacity(2 * self.two_qubit_gates());
        let mut cz_index = 0;
        for layer in 0..self.depth {
            for (a, b) in layer_pairs(self.qubits, layer) {
                out.push(DopeSite {
                    layer,
                    qubit: a,
                    cz_index,
                });
                out.push(DopeSite {
                    layer,
                    qubit: b,
                    cz_index,
                });
                cz_index += 1;
            }
        }
        out
    }

    /// The doping locations actually used, drawn without replacement from
    /// [`dope_sites`](Self::dope_sites) under `seed`.
    pub fn doping(&self) -> Vec<DopeSite> {
        let mut pool = self.dope_sites();
        match self.doping {
            Doping::Uniform => {}
            Doping::Late { layers } => {
                let first = self.depth.saturating_sub(layers);
                pool.retain(|s| s.layer >= first);
            }
            Doping::Early { layers } => pool.retain(|s| s.layer < layers),
            Doping::Banded {
                width,
                gap,
                layers,
                late,
            } => {
                let period = width + gap;
                let first = self.depth.saturating_sub(layers);
                pool.retain(|s| {
                    let in_time = if late {
                        s.layer >= first
                    } else {
                        s.layer < layers
                    };
                    in_time && period > 0 && s.qubit % period < width
                });
            }
        }
        let t = self.t_gates.min(pool.len());
        // Partial Fisher–Yates: the first `t` entries after the shuffle
        // are a uniform sample without replacement.
        let mut rng = Prng::new(self.seed ^ 0xD0FE_5175_u64);
        for i in 0..t {
            let j = i + (rng.next_u64() % (pool.len() - i) as u64) as usize;
            pool.swap(i, j);
        }
        pool.truncate(t);
        pool.sort_by_key(|s| (s.layer, s.cz_index, s.qubit));
        pool
    }

    /// The Clifford skeleton — the circuit before doping, and the
    /// experiment's own trusted classical baseline.
    pub fn skeleton(&self) -> Circuit<C64> {
        self.build(&[])
    }

    /// The doped circuit: the hard instance.
    pub fn circuit(&self) -> Circuit<C64> {
        let doping = self.doping();
        self.build(&doping)
    }

    fn build(&self, doping: &[DopeSite]) -> Circuit<C64> {
        let n = self.qubits;
        let mut rng = Prng::new(self.seed);
        let mut c: Circuit<C64> = Circuit::new(n);
        for q in 0..n {
            c.h(q);
        }
        let mut cz_index = 0;
        let mut next_dope = 0;
        for layer in 0..self.depth {
            for (a, b) in layer_pairs(n, layer) {
                c.cz(a, b);
                // Doping sites sit on the wires *immediately following*
                // this CZ, which is where the paper places them.
                while next_dope < doping.len() && doping[next_dope].cz_index == cz_index {
                    c.t(doping[next_dope].qubit);
                    next_dope += 1;
                }
                cz_index += 1;
            }
            // Random single-qubit layer: √X or S·√X.
            for q in 0..n {
                if rng.next_u64() & 1 == 1 {
                    c.s(q);
                }
                c.sx(q);
            }
        }
        // Rotate the stabilizer state into graph-state form.
        for q in 0..n {
            if rng.next_u64() & 1 == 1 {
                c.h(q);
            } else {
                c.s(q);
            }
        }
        debug_assert_eq!(next_dope, doping.len(), "every doping site was placed");
        c
    }

    /// The paper's own control family: the identical brickwork with the
    /// `{√X, S·√X}` layers replaced by **Haar-random** single-qubit
    /// gates (its Figures S13 and S14).
    ///
    /// The control isolates exactly one variable. Same width, same
    /// depth, same 2415 `CZ`s, same causal structure — only the
    /// single-qubit layer stops being Clifford. The paper's claim is
    /// that this makes the circuit *easier*, not harder, because a
    /// Haar-random layer produces exponentially decaying Schmidt
    /// spectra that an MPS can truncate cheaply, whereas the stabilizer
    /// spectrum is flat and truncation buys fidelity only in proportion
    /// to the bond budget. Being beaten by your own random control is
    /// the counter-intuitive claim worth re-measuring.
    pub fn haar_control(&self) -> Circuit<C64> {
        let n = self.qubits;
        let mut rng = Prng::new(self.seed);
        let mut c: Circuit<C64> = Circuit::new(n);
        for q in 0..n {
            c.h(q);
        }
        for layer in 0..self.depth {
            for (a, b) in layer_pairs(n, layer) {
                c.cz(a, b);
            }
            for q in 0..n {
                // Haar measure on SU(2): θ = 2·arccos(√u).
                let u = rng.next_f64();
                let theta = 2.0 * u.sqrt().clamp(0.0, 1.0).acos();
                let phi = std::f64::consts::TAU * rng.next_f64();
                let lam = std::f64::consts::TAU * rng.next_f64();
                c.u(q, theta, phi, lam);
            }
        }
        c
    }

    /// Structural counts, without building anything exponential.
    pub fn census(&self) -> Census {
        let c = self.circuit();
        let mut walls = 0;
        let mut cz = 0;
        let mut t = 0;
        for op in c.ops() {
            if let crate::circuit::Op::Named { name, .. } = op {
                match name.as_str() {
                    // √X = H·S·H exactly, so it allocates two path
                    // variables; the initial and final layers add one each.
                    "sx" => walls += 2,
                    "h" => walls += 1,
                    "cz" => cz += 1,
                    "t" => t += 1,
                    _ => {}
                }
            }
        }
        Census {
            qubits: self.qubits,
            depth: self.depth,
            ops: c.len(),
            two_qubit_gates: cz,
            t_gates: t,
            walls,
        }
    }
}

/// The circuit rewritten as **Clifford ∘ Pauli rotations**: the `(x, z)`
/// support of each `T` gate's axis after it is pushed to the front of
/// the circuit.
///
/// Every Clifford+T circuit factors as `U = C · ∏_a exp(−iπ/8 · P_a)`,
/// where `P_a = B_a† Z_q B_a` is the `a`-th `T`'s axis conjugated by the
/// Clifford prefix `B_a` standing before it. That form is the common
/// ground of several methods — it is what CAMPS tracks, what the
/// [`crate::coupling`] partition acts on, and what any Pauli expansion
/// of the magic expands.
///
/// Signs are dropped: every question asked of these axes here — `𝔽₂`
/// rank, commutation, span — is a question about supports, and carrying
/// signs would only invite them to be trusted for something else.
///
/// Costs `O(n · gates)` with no `2^k` anywhere: a tableau of the prefix
/// Clifford is carried forward and read off at each `T`.
pub fn rotation_axes(circuit: &Circuit<C64>) -> crate::error::Result<Vec<(Mask, Mask)>> {
    let n = circuit.num_qubits();
    // `xim[q]`, `zim[q]` hold the supports of `B† X_q B` and `B† Z_q B`
    // for the Clifford prefix `B` walked so far.
    let mut xim: Vec<(Mask, Mask)> = (0..n).map(|q| (Mask::single(q), Mask::zero())).collect();
    let mut zim: Vec<(Mask, Mask)> = (0..n).map(|q| (Mask::zero(), Mask::single(q))).collect();
    fn mul(a: &(Mask, Mask), b: &(Mask, Mask)) -> (Mask, Mask) {
        (a.0.xor(&b.0), a.1.xor(&b.1))
    }
    let mut axes = Vec::new();
    for op in circuit.ops() {
        let crate::circuit::Op::Named { name, qubits, .. } = op else {
            return Err(crate::error::Error::InvalidState(
                "dcs::rotation_axes: only named registry gates".into(),
            ));
        };
        match (name.as_str(), qubits.len()) {
            // Not Clifford: the rotation leaves the prefix untouched and
            // is recorded at the axis the prefix has carried Z_q to.
            ("t" | "tdg", 1) => axes.push(zim[qubits[0]].clone()),
            // H swaps the two images.
            ("h", 1) => {
                let a = qubits[0];
                xim.swap(a, a);
                std::mem::swap(&mut xim[a], &mut zim[a]);
            }
            // S: X ↦ ∓Y, Z fixed. (Z and X themselves only move signs,
            // which are not carried here, so they are no-ops.)
            ("s" | "sdg", 1) => {
                let a = qubits[0];
                xim[a] = mul(&xim[a], &zim[a]);
            }
            // √X: X fixed, Z ↦ ∓Y.
            ("sx" | "sxdg", 1) => {
                let a = qubits[0];
                zim[a] = mul(&xim[a], &zim[a]);
            }
            ("z" | "x" | "y" | "id", 1) => {}
            ("cz", 2) => {
                let (a, b) = (qubits[0], qubits[1]);
                let (za, zb) = (zim[a].clone(), zim[b].clone());
                xim[a] = mul(&xim[a], &zb);
                xim[b] = mul(&xim[b], &za);
            }
            ("cx", 2) => {
                let (c, t) = (qubits[0], qubits[1]);
                let (xt, zc) = (xim[t].clone(), zim[c].clone());
                xim[c] = mul(&xim[c], &xt);
                zim[t] = mul(&zim[t], &zc);
            }
            ("swap", 2) => {
                let (a, b) = (qubits[0], qubits[1]);
                xim.swap(a, b);
                zim.swap(a, b);
            }
            _ => {
                return Err(crate::error::Error::InvalidState(format!(
                    "dcs::rotation_axes: `{name}` is not in the Clifford+T fragment"
                )))
            }
        }
    }
    Ok(axes)
}

/// `𝔽₂` rank of a set of Pauli axes' symplectic vectors `(x ‖ z)` over
/// an `n`-qubit register.
///
/// The Pauli group the axes generate has `2^rank` elements, and `rank`
/// can never exceed `2n` however many axes there are. That ceiling is
/// the point: it is what turns "how much magic is there" into "how much
/// magic *fits*".
pub fn symplectic_rank(axes: &[(Mask, Mask)], n: usize) -> usize {
    let mut pivots: Vec<(usize, Mask)> = Vec::new();
    for (x, z) in axes {
        let mut row = x.clone();
        for i in z.iter() {
            row.set(n + i);
        }
        for (p, prow) in &pivots {
            if row.bit(*p) {
                row = row.xor(prow);
            }
        }
        if let Some(p) = row.lowest() {
            pivots.push((p, row));
        }
    }
    pivots.len()
}

/// The rank of the axes' **X-components alone** — the quantity the
/// paper's §S9.3 reports as `t − n`, the CAMPS disentangling metric.
///
/// Reproducing it is the cross-check on [`rotation_axes`]: the paper
/// states 468 − 70 = 398 for the experiment, and a propagation that
/// misses that number is propagating something else.
pub fn x_component_rank(axes: &[(Mask, Mask)], n: usize) -> usize {
    let x_only: Vec<(Mask, Mask)> = axes
        .iter()
        .map(|(x, _)| (x.clone(), Mask::zero()))
        .collect();
    symplectic_rank(&x_only, n)
}

/// Sizes of the connected components of the axes' **anticommutation
/// graph**, descending.
///
/// Distinct components are symplectically orthogonal, so the evolution
/// factorizes across them however much their qubit supports overlap
/// ([`crate::coupling`]) — a strictly sharper partition than the
/// support-overlap one [`crate::upembed`] uses for its readout. One
/// component means neither factorization is available.
pub fn anticommutation_components(axes: &[(Mask, Mask)]) -> Vec<usize> {
    let m = axes.len();
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
            let anti =
                (axes[i].1.and(&axes[j].0).count() + axes[i].0.and(&axes[j].1).count()) % 2 == 1;
            if anti {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                parent[a] = b;
            }
        }
    }
    let mut sizes: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for i in 0..m {
        let r = find(&mut parent, i);
        *sizes.entry(r).or_insert(0) += 1;
    }
    let mut out: Vec<usize> = sizes.into_values().collect();
    out.sort_unstable_by(|a, b| b.cmp(a));
    out
}

/// Process fidelity of an `S` gate standing in for a `T`:
/// `cos²(π/8) = (2 + √2)/4`.
///
/// The paper's §S9.5 spoofing route is to Cliffordize — swap `T` gates
/// for the nearest Clifford and simulate what is left — and `S` is that
/// nearest Clifford. The paper reports this constant as 0.8536.
pub const CLIFFORDIZED_T_FIDELITY: f64 = 0.853_553_390_593_273_8;

/// Replace `k` of `circuit`'s `T` gates with `S` gates, chosen uniformly
/// under `seed`.
///
/// The spoofing attack of §S9.5: every swap moves the circuit toward the
/// Clifford fragment, where it is free, and the question is how much
/// fidelity that buys back. The paper's answer is that it does not —
/// fidelity falls exponentially in `k`, and the experiment's own
/// measured fidelity is reached after only a handful of swaps, long
/// before the residual `T` count is tractable.
pub fn cliffordize(circuit: &Circuit<C64>, k: usize, seed: u64) -> Circuit<C64> {
    let t_positions: Vec<usize> = circuit
        .ops()
        .iter()
        .enumerate()
        .filter(|(_, op)| matches!(op, crate::circuit::Op::Named { name, .. } if name == "t"))
        .map(|(i, _)| i)
        .collect();
    let mut pool = t_positions;
    let k = k.min(pool.len());
    let mut rng = Prng::new(seed);
    for i in 0..k {
        let j = i + (rng.next_u64() % (pool.len() - i) as u64) as usize;
        pool.swap(i, j);
    }
    let swap: std::collections::HashSet<usize> = pool[..k].iter().copied().collect();
    let mut out: Circuit<C64> = Circuit::new(circuit.num_qubits());
    for (i, op) in circuit.ops().iter().enumerate() {
        match op {
            crate::circuit::Op::Named { name, qubits, .. } if name == "t" && swap.contains(&i) => {
                out.s(qubits[0]);
            }
            crate::circuit::Op::Named {
                name,
                params,
                qubits,
            } => {
                out.gate(name.clone(), params.clone(), qubits.clone());
            }
            other => panic!("dcs circuits are built from named gates only, found {other:?}"),
        }
    }
    out
}

/// Structural counts for an instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Census {
    /// Register width.
    pub qubits: usize,
    /// Brickwork depth.
    pub depth: usize,
    /// Total operations.
    pub ops: usize,
    /// `CZ` gates.
    pub two_qubit_gates: usize,
    /// `T` gates.
    pub t_gates: usize,
    /// Hadamards, counting the two inside each `√X` — the *a-priori*
    /// ceiling on `h*`, since only a wall allocates a path variable.
    pub walls: usize,
}

/// What the path sum made of an instance.
#[derive(Debug, Clone)]
pub struct Probe {
    /// `h*` — internal variables surviving reduction. Readout costs
    /// `2^{h*}` per amplitude.
    pub h_star: usize,
    /// Walls in the circuit: the ceiling `h*` is measured against.
    pub walls: usize,
    /// Monomials in the surviving phase polynomial.
    pub terms: usize,
    /// Rule-V splits performed — what reduction actually cost.
    pub splits: u64,
    /// Surviving variables that stalled only on where the
    /// self-coefficient landed (the `T` gate's odd eighth), not on
    /// monomial structure.
    pub alignment_stalls: usize,
    /// Wall-clock of the reduction.
    pub elapsed: Duration,
    /// Reduction hit the time budget rather than running dry, in which
    /// case `h_star` is an over-estimate.
    pub cut_short: bool,
}

/// Reduce `circuit` as a path sum under a wall-clock `budget` and report
/// what survived.
///
/// The budget is armed through [`crate::guard`], so an over-scale
/// instance comes back as a **measured refusal** — `cut_short` set,
/// elapsed time recorded — rather than stalling. That distinction
/// matters here: `h*` and reduction *time* are different resources on
/// this family, and only one of them is the published claim.
pub fn probe(circuit: &Circuit<C64>, budget: Duration) -> crate::error::Result<Probe> {
    let mut walls = 0;
    for op in circuit.ops() {
        if let crate::circuit::Op::Named { name, .. } = op {
            match name.as_str() {
                "sx" | "sxdg" => walls += 2,
                "h" => walls += 1,
                "rx" => walls += 2,
                _ => {}
            }
        }
    }
    let mut ps = PathSum::new(circuit.num_qubits());
    ps.apply_circuit(circuit, false)?;
    let t0 = Instant::now();
    crate::guard::with_time_budget(budget, || {
        let _ = ps.try_reduce();
    });
    let elapsed = t0.elapsed();
    let stalls = ps.stall_census();
    Ok(Probe {
        h_star: ps.internal_vars(),
        walls,
        terms: ps.terms(),
        splits: ps.splits(),
        alignment_stalls: stalls.iter().filter(|(_, s)| s.is_alignment()).count(),
        elapsed,
        cut_short: ps.reduction_cut_short(),
    })
}
