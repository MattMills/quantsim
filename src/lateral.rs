//! Cross-lateral distributed registers: the frame is the wire.
//!
//! [`logical`](crate::logical) holds a register as `P · Enc · |l⟩` and
//! [`retro`](crate::retro) proves the frame's
//! [`signature`](crate::retro::signature) **is** the syndrome — a fault
//! is one bookkeeping update and no state work at all. The
//! `logical_network` example puts two toric nodes side by side and
//! shows the links living in a selector. Everything there happens
//! inside one process, on one clock, over a channel that cannot drop
//! anything.
//!
//! This module is what happens when the nodes are actually apart.
//!
//! ## The observation
//!
//! Give each node its own code patch and couple nodes only
//! **transversally** — physical qubit `i` of node A to physical qubit
//! `i` of node B, which for a CSS code is the logical CX. Then three
//! things are true at once, and together they are the whole technique:
//!
//! 1. **The inter-node map is coordinatewise.** A transversal CX
//!    conjugates the Pauli frame by
//!    ```text
//!    x_B ^= x_A        z_A ^= z_B        σ ^= |x_A & z_B & !(x_B ^ z_A)| mod 2
//!    ```
//!    — two XORs of *aligned* bit-vectors and a popcount, with no term
//!    coupling qubit `i` of one node to qubit `j ≠ i` of the other.
//!    [`LateralLink::push_frame`] is that rule; `tests/lateral.rs`
//!    pins it against gate-by-gate
//!    [`conjugate_by_step`] on
//!    random strings. "Lateral" is the claim that the link is a
//!    *bundle of independent wires at the same level*, not a mixing
//!    matrix.
//! 2. **The syndrome is 𝔽₂-linear in the frame.** The symplectic form
//!    is bilinear, so [`SyndromeMap::bits`] satisfies
//!    `bits(p ⊕ q) = bits(p) ⊕ bits(q)` exactly, for *any* pair of
//!    strings, commuting or not. A node holding the total syndrome
//!    (an ancilla-free read, [`retro::syndromes`](crate::retro::syndromes))
//!    and the deltas that arrived can therefore compute the syndrome
//!    of the delta that did **not** arrive, by one XOR.
//! 3. **So the wire carries nothing but 𝔽₂.** The amplitudes never
//!    move. Per tick a node ships its frame delta: two masks and a
//!    sign — 16 bytes for a 64-qubit patch, independent of `2ⁿ`.
//!
//! ## What that makes a dropped packet
//!
//! A Pauli fault and a lost datagram become **the same object**: an
//! unknown vector in 𝔽₂^{2n} to be recovered from `H·e = s`. The
//! difference is that the network *knows where its loss happened* —
//! the schedule says which qubits the tick's operation touched — and
//! the physics does not.
//!
//! Known locations turn error decoding into **erasure** decoding, and
//! for a stabilizer code that is worth exactly a factor of two:
//!
//! | recovery | condition | capacity for distance `d` |
//! |---|---|---|
//! | [`Decoder`](crate::retro::Decoder) — unknown location | minimum-weight, distinguishable | `⌊(d−1)/2⌋` |
//! | [`ErasureDecoder`] — known location | no logical supported on the erased set | `d − 1` |
//!
//! [`ErasureDecoder::certified_capacity`] does not take that on
//! authority. It enumerates erasure sets against the code's own
//! generators and reports the largest size at which *every* set is
//! correctable, and [`ErasureDecoder::decode`] refuses a set carrying
//! a logical **by name** rather than guessing — the same contract
//! [`retro::Decoder`](crate::retro::Decoder) holds for degeneracy.
//!
//! ## What that makes the horizon
//!
//! A distributed run has to seal ticks that every node agrees on,
//! without any node waiting on the others in real time: authoritative
//! work happens at `now − H`, and because every node uses the same
//! rule on the same sealed inputs they agree exactly. [`Barrier`]
//! tracks *actual* arrivals per peer per tick and seals either
//! [`Seal::Clean`] or [`Seal::Degraded`] naming the missing peers —
//! never on a prediction. [`DelayGeometry::min_horizon`] is the floor
//! no protocol beats: the network's temporal diameter, over delays
//! [`tightened`](DelayGeometry::tighten) first, because measured
//! inter-node latency routinely violates the triangle inequality and
//! reasoning that assumes a metric is quietly wrong.
//!
//! And then the horizon stops being a tax. Repair-from-parity costs
//! `k` ticks of sender-side buffering; if `H` already exceeds
//! `k +` one-way delay the repair lands *inside* the barrier and the
//! loss is invisible — the tick seals clean, on time, with no
//! retransmission. [`fits_in_horizon`] is that inequality.
//!
//! ## The two layers, and why both
//!
//! [`DualLayer`] cross-decodes a **fine** layer against a **coarse**
//! one:
//!
//! * **fine, local, free** — the stabilizer code itself. Costs *zero
//!   bandwidth*: the syndrome is already there. Repairs a lost delta
//!   whose scheduled support is within the erasure capacity, however
//!   many ticks in a row are lost.
//! * **coarse, global, paid** — [`WindowParity`], a systematic
//!   MDS code over GF(256) on a window of `k` ticks with `m` parity
//!   shards. Repairs *any* `m` losses per window regardless of
//!   support, and costs `m/k` in bandwidth.
//!
//! Each layer has a loss pattern that defeats it: a burst longer than
//! `m` defeats the coarse layer, a wide-support tick defeats the fine
//! one. [`DualLayer::repair`] iterates them, and each layer's
//! successes shrink the other's unknown count until neither can move.
//! `tests/lateral.rs` pins a pattern that defeats each layer alone and
//! falls to the pair.
//!
//! ## Honest scope
//!
//! * The distributed vocabulary is **Clifford**: physical Paulis (any
//!   weight, absorbed into the frame) and transversal CX between
//!   patches. A `t` does not transport through a Pauli frame and
//!   [`compile_clifford`](crate::retro::compile_clifford) already
//!   refuses it by name; nothing here weakens that.
//! * Frames are bookkeeping, so the distributed run reproduces the
//!   monolithic one **exactly** — not to a tolerance. That is a
//!   statement about 𝔽₂ arithmetic, not about the simulator being
//!   good, and `tests/lateral.rs` asserts equality rather than a
//!   bound.
//! * This is a *simulation* of a distributed run: the loss, the
//!   delays and the clocks are injected, not measured off a socket.
//!   What is real is the algebra — the coordinatewise link rule, the
//!   syndrome's linearity, the erasure capacity measured off the
//!   code, and the arithmetic of the repair.
//! * It buys **fault tolerance and bandwidth**, not speed. Splitting
//!   a register across nodes does not shrink `2ⁿ`; it shrinks what has
//!   to cross the wire, from amplitudes to 16 bytes a tick.

use std::collections::BTreeMap;

use crate::backend::{conjugate_by_step, CliffordStep, PauliString};
use crate::error::{Error, Result};
use crate::retro::Code;

/// Widest patch a lateral link will align, set by [`PauliString`]'s
/// 63-qubit masks.
pub const MAX_PATCH: usize = 63;

// ───────────────────────────── 𝔽₂ helpers ─────────────────────────────

/// XOR two strings' masks, dropping the sign — the abelian group the
/// syndrome is linear over. Pauli multiplication is *not* this map
/// (it carries a symplectic phase), but the syndrome cannot see the
/// phase, so this is the right group for talking about syndromes.
pub fn xor_masks(a: PauliString, b: PauliString) -> PauliString {
    PauliString {
        x: a.x ^ b.x,
        z: a.z ^ b.z,
        negative: a.negative ^ b.negative,
    }
}

/// An 𝔽₂ row space over `(x‖z)` vectors, kept in echelon form.
#[derive(Clone, Debug, Default)]
struct F2Rows {
    rows: Vec<(u64, u64)>,
    pivots: Vec<usize>,
}

impl F2Rows {
    /// Leading set bit of `(x‖z)`, X bits first.
    fn leading(x: u64, z: u64) -> Option<usize> {
        if x != 0 {
            Some(x.trailing_zeros() as usize)
        } else if z != 0 {
            Some(64 + z.trailing_zeros() as usize)
        } else {
            None
        }
    }

    fn reduce(&self, mut x: u64, mut z: u64) -> (u64, u64) {
        while let Some(lead) = Self::leading(x, z) {
            match self.pivots.iter().position(|&p| p == lead) {
                Some(i) => {
                    x ^= self.rows[i].0;
                    z ^= self.rows[i].1;
                }
                None => break,
            }
        }
        (x, z)
    }

    fn insert(&mut self, x: u64, z: u64) {
        let (x, z) = self.reduce(x, z);
        if let Some(lead) = Self::leading(x, z) {
            self.rows.push((x, z));
            self.pivots.push(lead);
        }
    }

    fn contains(&self, x: u64, z: u64) -> bool {
        self.reduce(x, z) == (0, 0)
    }
}

// ─────────────────────────── the lateral link ───────────────────────────

/// A transversal CX between two equal-sized code patches: physical
/// `CX(control_offset + i → target_offset + i)` for every `i`, which
/// for CSS codes is the **logical** CX — and for the toric code, two
/// logical links at once.
///
/// The point of the type is [`push_frame`](LateralLink::push_frame):
/// the frame map is coordinatewise in `i`, so the whole inter-node
/// coupling is two XORs of aligned bit-vectors. That is what "lateral"
/// means here — a bundle of independent wires at one level, not a
/// mixing matrix — and it is why the wire carries `2n` bits instead of
/// `2ⁿ` amplitudes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LateralLink {
    control: usize,
    target: usize,
    width: usize,
}

impl LateralLink {
    /// Align two patches. They must be the same size, disjoint, and
    /// within [`MAX_PATCH`].
    pub fn new(control: &impl Code, target: &impl Code) -> Result<Self> {
        Self::from_windows(
            control.offset(),
            target.offset(),
            control.qubits(),
            target.qubits(),
        )
    }

    /// The same, from raw windows — for patches that are not [`Code`]
    /// implementors.
    pub fn from_windows(
        control: usize,
        target: usize,
        control_qubits: usize,
        target_qubits: usize,
    ) -> Result<Self> {
        if control_qubits != target_qubits {
            return Err(Error::InvalidState(format!(
                "lateral link: transversal CX needs equal patches, got {control_qubits} \
                 and {target_qubits} — a lateral link is qubit i to qubit i and there \
                 is no i-to-i map between unequal sets"
            )));
        }
        let width = control_qubits;
        if width == 0 || width > MAX_PATCH {
            return Err(Error::InvalidState(format!(
                "lateral link: patch width {width} outside 1..={MAX_PATCH}"
            )));
        }
        let (lo, hi) = if control < target {
            (control, target)
        } else {
            (target, control)
        };
        if lo + width > hi {
            return Err(Error::InvalidState(format!(
                "lateral link: patches at {control} and {target} of width {width} overlap \
                 — a link couples two registers, not a register to itself"
            )));
        }
        if control.max(target) + width > MAX_PATCH + 1 {
            return Err(Error::InvalidState(format!(
                "lateral link: patch at {} extends past qubit {MAX_PATCH}",
                control.max(target)
            )));
        }
        Ok(LateralLink {
            control,
            target,
            width,
        })
    }

    /// First register index of the control patch.
    pub fn control(&self) -> usize {
        self.control
    }

    /// First register index of the target patch.
    pub fn target(&self) -> usize {
        self.target
    }

    /// Qubits per patch — and the number of physical CX gates the link
    /// costs.
    pub fn width(&self) -> usize {
        self.width
    }

    fn mask(&self) -> u64 {
        if self.width == 64 {
            u64::MAX
        } else {
            (1u64 << self.width) - 1
        }
    }

    /// The physical gates the link is made of.
    pub fn steps(&self) -> Vec<CliffordStep> {
        (0..self.width)
            .map(|i| CliffordStep::Cx(self.control + i, self.target + i))
            .collect()
    }

    /// Conjugate a frame through the link, coordinatewise.
    ///
    /// With `a` the control window and `b` the target window,
    /// `CX† P CX` is
    ///
    /// ```text
    /// x_b ^= x_a       z_a ^= z_b       σ ^= |x_a & z_b & !(x_b ^ z_a)| mod 2
    /// ```
    ///
    /// (the sign popcount reading the *pre-update* windows). Every
    /// term is a bitwise operation between aligned masks: no qubit of
    /// one patch ever reaches a differently-indexed qubit of the
    /// other. The result equals folding the `width` physical CX gates
    /// through [`conjugate_by_step`] one at a time, which is what
    /// `tests/lateral.rs` checks on random strings.
    pub fn push_frame(&self, p: PauliString) -> PauliString {
        let m = self.mask();
        let xa = (p.x >> self.control) & m;
        let za = (p.z >> self.control) & m;
        let xb = (p.x >> self.target) & m;
        let zb = (p.z >> self.target) & m;

        let sign = (xa & zb & !(xb ^ za) & m).count_ones() & 1 == 1;

        let xb_new = xb ^ xa;
        let za_new = za ^ zb;

        PauliString {
            x: (p.x & !(m << self.target)) | (xb_new << self.target),
            z: (p.z & !(m << self.control)) | (za_new << self.control),
            negative: p.negative ^ sign,
        }
    }

    /// The same map by brute force — every physical gate folded
    /// through [`conjugate_by_step`]. Kept public because it is the
    /// oracle the fast rule is measured against, and because a reader
    /// who does not believe the mask algebra can run it.
    pub fn push_frame_by_gates(&self, p: PauliString) -> PauliString {
        self.steps().into_iter().fold(p, conjugate_by_step)
    }
}

// ───────────────────────────── the syndrome ─────────────────────────────

/// The syndrome as an 𝔽₂-linear map, packed into a word so it can be
/// XORed.
///
/// [`retro::signature`](crate::retro::signature) answers the same
/// question as a `Vec<bool>`; this is the same numbers in the form the
/// distributed protocol needs, plus the guarantee that makes the
/// protocol work at all:
///
/// ```text
/// bits(p ⊕ q) = bits(p) ⊕ bits(q)
/// ```
///
/// which holds because `ω(g, ·)` is 𝔽₂-bilinear — for *every* pair of
/// strings, commuting or not, since the syndrome cannot see the phase
/// that non-commutation would produce. So a node that reads the total
/// syndrome off its own state and knows the deltas that arrived can
/// name the syndrome of the delta that did not, by one XOR.
#[derive(Clone, Debug)]
pub struct SyndromeMap {
    gens: Vec<PauliString>,
}

impl SyndromeMap {
    /// Read a code's generators. At most 64 (the packed word); codes
    /// with more are refused rather than truncated.
    pub fn new(code: &impl Code) -> Result<Self> {
        let gens = code.generators();
        if gens.len() > 64 {
            return Err(Error::InvalidState(format!(
                "syndrome map: {} generators exceeds the 64-bit packed syndrome; \
                 refused rather than truncated",
                gens.len()
            )));
        }
        Ok(SyndromeMap { gens })
    }

    /// Generator count — the syndrome's bit width.
    pub fn len(&self) -> usize {
        self.gens.len()
    }

    /// Whether the code had no generators (it never does).
    pub fn is_empty(&self) -> bool {
        self.gens.is_empty()
    }

    /// The packed syndrome: bit `i` set where the string anticommutes
    /// with generator `i`.
    pub fn bits(&self, p: PauliString) -> u64 {
        let mut s = 0u64;
        for (i, g) in self.gens.iter().enumerate() {
            if !g.commutes_with(p) {
                s |= 1 << i;
            }
        }
        s
    }

    /// The unpacked form, matching
    /// [`retro::signature`](crate::retro::signature) and
    /// [`retro::syndrome_bits`](crate::retro::syndrome_bits).
    pub fn unpack(&self, bits: u64) -> Vec<bool> {
        (0..self.gens.len()).map(|i| bits & (1 << i) != 0).collect()
    }

    /// Pack an unpacked syndrome.
    pub fn pack(&self, bits: &[bool]) -> Result<u64> {
        if bits.len() != self.gens.len() {
            return Err(Error::InvalidState(format!(
                "syndrome map: {} bits for a {}-generator code",
                bits.len(),
                self.gens.len()
            )));
        }
        Ok(bits
            .iter()
            .enumerate()
            .fold(0u64, |a, (i, &b)| if b { a | (1 << i) } else { a }))
    }
}

// ──────────────────────────── erasure decoding ────────────────────────────

/// Recovery of a Pauli whose **location** is known and whose value is
/// not — the network's problem, as opposed to the physics'.
///
/// [`Decoder`](crate::retro::Decoder) inverts the syndrome over all
/// low-weight faults and runs out at `⌊(d−1)/2⌋`, because two faults of
/// equal weight separated by a logical are indistinguishable. Told
/// *which* qubits could be involved, that ambiguity mostly evaporates:
/// the decode is unique up to the stabilizer group exactly when no
/// logical operator is supported inside the erased set, and a logical
/// has weight at least `d`. So the capacity is `d − 1`.
///
/// This type does not assume that. [`correctable`](Self::correctable)
/// asks the code — is every zero-syndrome Pauli on this set a
/// stabilizer? — and [`certified_capacity`](Self::certified_capacity)
/// enumerates until the answer is no.
#[derive(Clone, Debug)]
pub struct ErasureDecoder {
    gens: Vec<PauliString>,
    stabilizer: F2Rows,
    offset: usize,
    qubits: usize,
}

impl ErasureDecoder {
    /// Read a code's generators and build its stabilizer row space.
    pub fn new(code: &impl Code) -> Self {
        let gens = code.generators();
        let mut stabilizer = F2Rows::default();
        for g in &gens {
            stabilizer.insert(g.x, g.z);
        }
        ErasureDecoder {
            gens,
            stabilizer,
            offset: code.offset(),
            qubits: code.qubits(),
        }
    }

    /// First register index of the patch.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Data qubits in the patch.
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    /// The syndrome column of a single-qubit generator, as an 𝔽₂
    /// vector over the code's generators.
    fn column(&self, q: usize, x: bool) -> u64 {
        let bit = 1u64 << q;
        let p = PauliString {
            x: if x { bit } else { 0 },
            z: if x { 0 } else { bit },
            negative: false,
        };
        let mut c = 0u64;
        for (i, g) in self.gens.iter().enumerate() {
            if !g.commutes_with(p) {
                c |= 1 << i;
            }
        }
        c
    }

    /// The `2|E|` columns of the erasure system, paired with the
    /// single-qubit generator each stands for.
    fn system(&self, erased: &[usize]) -> Result<Vec<(u64, PauliString)>> {
        // The elimination tracks column combinations in a `u128`, so
        // the system is capped at 64 erased qubits — which is past
        // what `PauliString`'s 63-qubit masks can address anyway.
        if erased.len() > 64 {
            return Err(Error::InvalidState(format!(
                "erasure decode: {} erased qubits exceeds the 64 the column tracker \
                 addresses",
                erased.len()
            )));
        }
        let mut cols = Vec::with_capacity(2 * erased.len());
        for &q in erased {
            if q > MAX_PATCH {
                return Err(Error::InvalidState(format!(
                    "erasure decode: qubit {q} is past the {MAX_PATCH}-qubit mask"
                )));
            }
            if q < self.offset || q >= self.offset + self.qubits {
                return Err(Error::InvalidState(format!(
                    "erasure decode: qubit {q} is not in this patch ({}..{})",
                    self.offset,
                    self.offset + self.qubits
                )));
            }
            let bit = 1u64 << q;
            cols.push((
                self.column(q, true),
                PauliString {
                    x: bit,
                    z: 0,
                    negative: false,
                },
            ));
            cols.push((
                self.column(q, false),
                PauliString {
                    x: 0,
                    z: bit,
                    negative: false,
                },
            ));
        }
        Ok(cols)
    }

    /// A basis for the Paulis supported on `erased` that produce no
    /// syndrome — the kernel of the erasure system.
    fn kernel(&self, cols: &[(u64, PauliString)]) -> Vec<PauliString> {
        // Row-reduce `[column | which columns combined to make it]`.
        let n = cols.len();
        let mut rows: Vec<(u64, u128)> = cols
            .iter()
            .enumerate()
            .map(|(j, (c, _))| (*c, 1u128 << j))
            .collect();
        let mut pivots: Vec<usize> = Vec::new();
        let mut kernel = Vec::new();
        for j in 0..n {
            let (mut val, mut track) = rows[j];
            for (i, &p) in pivots.iter().enumerate() {
                if val & (1 << p) != 0 {
                    val ^= rows[i].0;
                    track ^= rows[i].1;
                }
            }
            if val == 0 {
                // A dependent column: `track` names a zero-syndrome combination.
                let mut p = PauliString::identity();
                for (b, (_, gen)) in cols.iter().enumerate() {
                    if track & (1u128 << b) != 0 {
                        p = PauliString {
                            x: p.x ^ gen.x,
                            z: p.z ^ gen.z,
                            negative: false,
                        };
                    }
                }
                kernel.push(p);
            } else {
                let lead = val.trailing_zeros() as usize;
                // `pivots.len() <= j` always, so this only ever
                // overwrites a column already consumed.
                rows[pivots.len()] = (val, track);
                pivots.push(lead);
            }
        }
        kernel
    }

    /// Is every zero-syndrome Pauli on `erased` a stabilizer?
    ///
    /// If so, two frames on this set with the same syndrome differ by
    /// an element of the stabilizer group and therefore act
    /// identically on the code space — the decode is unique where it
    /// matters. If not, some logical operator hides inside the set and
    /// a decode would silently rewrite the logical state.
    pub fn correctable(&self, erased: &[usize]) -> bool {
        let Ok(cols) = self.system(erased) else {
            return false;
        };
        self.kernel(&cols)
            .into_iter()
            .all(|p| self.stabilizer.contains(p.x, p.z))
    }

    /// Is the recovery on `erased` unique *on the nose* — is there no
    /// non-identity Pauli at all on this set with zero syndrome?
    ///
    /// [`correctable`](Self::correctable) is the weaker and more
    /// useful physical condition: the ambiguity, if any, is a
    /// stabilizer, and a stabilizer-equivalent frame acts identically
    /// on every logical observable. But a *byte-exact* recovery is
    /// what a downstream linear code needs, and that asks for the
    /// kernel to be trivial — which it is whenever the erased set is
    /// smaller than the code's lightest stabilizer.
    pub fn unique(&self, erased: &[usize]) -> bool {
        match self.system(erased) {
            Ok(cols) => self.kernel(&cols).is_empty(),
            Err(_) => false,
        }
    }

    /// Recover the frame delta supported on `erased` from its packed
    /// syndrome.
    ///
    /// Refuses — never guesses — in two distinguishable ways: a
    /// syndrome no fault on this set can produce, and a set that
    /// carries a logical.
    pub fn decode(&self, erased: &[usize], syndrome: u64) -> Result<PauliString> {
        let cols = self.system(erased)?;
        if !self.correctable(erased) {
            return Err(Error::InvalidState(format!(
                "erasure decode: a logical operator is supported inside the {} erased \
                 qubit(s) — two frames with this syndrome differ by a logical, and a \
                 guess would silently rewrite the logical state; refused",
                erased.len()
            )));
        }
        // Solve `M v = syndrome` by forward elimination with tracked
        // column combinations.
        let mut rows: Vec<(u64, u128)> = Vec::new();
        let mut pivots: Vec<usize> = Vec::new();
        for (j, (c, _)) in cols.iter().enumerate() {
            let (mut val, mut track) = (*c, 1u128 << j);
            for (i, &p) in pivots.iter().enumerate() {
                if val & (1 << p) != 0 {
                    val ^= rows[i].0;
                    track ^= rows[i].1;
                }
            }
            if val != 0 {
                let lead = val.trailing_zeros() as usize;
                rows.push((val, track));
                pivots.push(lead);
            }
        }
        let (mut rest, mut combo) = (syndrome, 0u128);
        for (i, &p) in pivots.iter().enumerate() {
            if rest & (1 << p) != 0 {
                rest ^= rows[i].0;
                combo ^= rows[i].1;
            }
        }
        if rest != 0 {
            return Err(Error::InvalidState(format!(
                "erasure decode: syndrome {syndrome:#x} is not produced by any fault on \
                 the erased set — the loss is not where the schedule says it is, or the \
                 syndrome itself is wrong; refused rather than projected onto the set"
            )));
        }
        let mut p = PauliString::identity();
        for (b, (_, gen)) in cols.iter().enumerate() {
            if combo & (1u128 << b) != 0 {
                p = PauliString {
                    x: p.x ^ gen.x,
                    z: p.z ^ gen.z,
                    negative: false,
                };
            }
        }
        Ok(p)
    }

    /// The largest `e ≤ max` such that **every** erasure set of size
    /// `e` inside the patch is correctable — measured by enumeration
    /// against the code's own generators, not read off a distance.
    ///
    /// Enumeration is `C(qubits, e)` per size and says so by taking
    /// `max` as a parameter.
    pub fn certified_capacity(&self, max: usize) -> usize {
        let mut best = 0;
        for e in 1..=max.min(self.qubits) {
            let mut all = true;
            for_each_combination(self.qubits, e, &mut |idx| {
                let set: Vec<usize> = idx.iter().map(|&i| self.offset + i).collect();
                if self.correctable(&set) {
                    true
                } else {
                    all = false;
                    false
                }
            });
            if all {
                best = e;
            } else {
                break;
            }
        }
        best
    }

    /// The smallest erasure set this decoder refuses — the *witness*
    /// for the capacity, so a reader can see the logical that ended
    /// it rather than trust the number. `None` when no set up to
    /// `max` is refused.
    pub fn first_uncorrectable(&self, max: usize) -> Option<Vec<usize>> {
        for e in 1..=max.min(self.qubits) {
            let mut found = None;
            for_each_combination(self.qubits, e, &mut |idx| {
                let set: Vec<usize> = idx.iter().map(|&i| self.offset + i).collect();
                if self.correctable(&set) {
                    true
                } else {
                    found = Some(set);
                    false
                }
            });
            if found.is_some() {
                return found;
            }
        }
        None
    }
}

/// Visit every `k`-subset of `0..n` in lexicographic order, stopping
/// early the first time `f` returns `false`.
fn for_each_combination(n: usize, k: usize, f: &mut impl FnMut(&[usize]) -> bool) {
    if k == 0 || k > n {
        return;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        if !f(&idx) {
            return;
        }
        let mut i = k;
        loop {
            if i == 0 {
                return;
            }
            i -= 1;
            if idx[i] != i + n - k {
                idx[i] += 1;
                for j in i + 1..k {
                    idx[j] = idx[j - 1] + 1;
                }
                break;
            }
        }
    }
}

// ──────────────────────────── the wire payload ────────────────────────────

/// Bytes one frame delta occupies on the wire: two 64-bit masks and a
/// sign, plus the node and tick it belongs to. Fixed-size, because the
/// coarse repair layer codes across equal-length shards.
pub const DELTA_WIRE_LEN: usize = 2 + 8 + 8 + 8 + 1;

/// One node's frame change for one tick — the entire inter-node
/// payload of a distributed encoded computation.
///
/// This is what replaces shipping amplitudes: 27 bytes, independent of
/// the register's width, carrying a Pauli frame delta that the
/// receiving node XORs into its own window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameDelta {
    /// Which node produced it.
    pub node: usize,
    /// Which tick it belongs to.
    pub tick: u64,
    /// The change, as masks over the whole register.
    pub delta: PauliString,
}

impl FrameDelta {
    /// Serialize to the fixed wire form (little-endian).
    pub fn to_bytes(&self) -> [u8; DELTA_WIRE_LEN] {
        let mut b = [0u8; DELTA_WIRE_LEN];
        b[0..2].copy_from_slice(&(self.node as u16).to_le_bytes());
        b[2..10].copy_from_slice(&self.tick.to_le_bytes());
        b[10..18].copy_from_slice(&self.delta.x.to_le_bytes());
        b[18..26].copy_from_slice(&self.delta.z.to_le_bytes());
        b[26] = u8::from(self.delta.negative);
        b
    }

    /// Parse the wire form.
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        if b.len() != DELTA_WIRE_LEN {
            return Err(Error::InvalidState(format!(
                "frame delta: {} bytes on the wire, expected {DELTA_WIRE_LEN}",
                b.len()
            )));
        }
        Ok(FrameDelta {
            node: u16::from_le_bytes([b[0], b[1]]) as usize,
            tick: u64::from_le_bytes(b[2..10].try_into().unwrap()),
            delta: PauliString {
                x: u64::from_le_bytes(b[10..18].try_into().unwrap()),
                z: u64::from_le_bytes(b[18..26].try_into().unwrap()),
                negative: b[26] != 0,
            },
        })
    }
}

// ───────────────────────────── delay geometry ─────────────────────────────

/// One-way delays between nodes, in ticks.
///
/// Nodes are laid out in *delay*, not in space, and that layout need
/// not be a metric: routing through a third node is frequently faster
/// than the direct path, so `d(i,k) > d(i,j) + d(j,k)` happens
/// constantly on real networks and any reasoning that assumes a
/// triangle inequality is quietly wrong.
/// [`triangle_violations`](Self::triangle_violations) counts them and
/// [`tighten`](Self::tighten) replaces every direct delay with the
/// best relayed one, which *is* a metric.
///
/// The temporal diameter of the tightened geometry is the hard floor
/// on the barrier: no protocol seals a fully-coupled system closer to
/// now than that. (The construction is `cliff`'s; the arithmetic is
/// small enough to restate exactly.)
#[derive(Clone, Debug)]
pub struct DelayGeometry {
    n: usize,
    delay: Vec<f64>,
}

impl DelayGeometry {
    /// A geometry from a row-major `n × n` delay matrix in ticks.
    /// Diagonal entries must be zero; off-diagonal entries finite and
    /// non-negative.
    pub fn new(n: usize, delay: Vec<f64>) -> Result<Self> {
        if delay.len() != n * n {
            return Err(Error::InvalidState(format!(
                "delay geometry: {} entries for {n} nodes, expected {}",
                delay.len(),
                n * n
            )));
        }
        for i in 0..n {
            for j in 0..n {
                let d = delay[i * n + j];
                if i == j {
                    if d != 0.0 {
                        return Err(Error::InvalidState(format!(
                            "delay geometry: self-delay d({i},{i}) = {d} is not zero"
                        )));
                    }
                } else if !d.is_finite() || d < 0.0 {
                    return Err(Error::InvalidState(format!(
                        "delay geometry: d({i},{j}) = {d} is not a finite non-negative delay"
                    )));
                }
            }
        }
        Ok(DelayGeometry { n, delay })
    }

    /// Node count.
    pub fn nodes(&self) -> usize {
        self.n
    }

    /// One-way delay from `i` to `j`, in ticks.
    pub fn delay(&self, i: usize, j: usize) -> f64 {
        self.delay[i * self.n + j]
    }

    /// Ordered pairs whose direct delay beats no relayed path — the
    /// measured triangle violations.
    pub fn triangle_violations(&self) -> usize {
        let mut v = 0;
        for i in 0..self.n {
            for j in 0..self.n {
                if i == j {
                    continue;
                }
                for k in 0..self.n {
                    if k == i || k == j {
                        continue;
                    }
                    if self.delay(i, j) > self.delay(i, k) + self.delay(k, j) {
                        v += 1;
                        break;
                    }
                }
            }
        }
        v
    }

    /// Every direct delay replaced by the best relayed one — the
    /// shortest-path closure, which satisfies the triangle inequality
    /// by construction.
    pub fn tighten(&self) -> DelayGeometry {
        let mut d = self.delay.clone();
        let n = self.n;
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let via = d[i * n + k] + d[k * n + j];
                    if via < d[i * n + j] {
                        d[i * n + j] = via;
                    }
                }
            }
        }
        DelayGeometry { n, delay: d }
    }

    /// The furthest any other node is from `i`, after tightening.
    pub fn temporal_radius(&self, i: usize) -> f64 {
        (0..self.n).map(|j| self.delay(i, j)).fold(0.0f64, f64::max)
    }

    /// The largest tightened shortest-path delay in the network.
    pub fn temporal_diameter(&self) -> f64 {
        (0..self.n)
            .map(|i| self.temporal_radius(i))
            .fold(0.0, f64::max)
    }

    /// The smallest horizon a fully-coupled system can use: the
    /// tightened temporal diameter, rounded up to whole ticks. This is
    /// the honest answer to "how close to now can we get" — exactly as
    /// close as the slowest necessary path allows, and not one tick
    /// closer.
    pub fn min_horizon(&self) -> u64 {
        self.tighten().temporal_diameter().ceil() as u64
    }
}

// ───────────────────────────────  barrier  ───────────────────────────────

/// Can a `(k, m)` window code repair a loss without exceeding the
/// barrier's horizon?
///
/// The repair path is: wait `k` ticks for the window to close, send the
/// parity, wait one one-way delay for it to arrive. If that total fits
/// inside `H`, the receiver never notices the loss — the tick seals
/// clean, on time, with no retransmission. That is what the horizon is
/// *for*: not a delay to be tolerated, a budget, and erasure coding is
/// what it buys. (The inequality is `cliff`'s `fits_in_horizon`.)
pub fn fits_in_horizon(k: usize, one_way_ticks: f64, horizon_ticks: u64) -> bool {
    (k as f64) + one_way_ticks <= horizon_ticks as f64
}

/// How a tick sealed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Seal {
    /// Every required peer delivered, before the deadline.
    Clean {
        /// The tick that sealed.
        tick: u64,
    },
    /// Peers were missing at the deadline, but repair filled them in —
    /// the erasure code earning its place.
    Repaired {
        /// The tick that sealed.
        tick: u64,
        /// The peers whose contribution was reconstructed, not received.
        reconstructed: Vec<usize>,
    },
    /// The deadline passed with peers still missing, and it names
    /// them. A real gap in the shared history, never a silent one.
    Degraded {
        /// The tick that failed to seal cleanly.
        tick: u64,
        /// Peers with nothing delivered and nothing reconstructed.
        missing: Vec<usize>,
    },
    /// The deadline has not passed and peers are still outstanding.
    Waiting {
        /// The tick being waited on.
        tick: u64,
        /// Peers not yet heard from.
        missing: Vec<usize>,
    },
}

/// The latency barrier: authoritative work at `now − H`, tracked
/// against **actual arrivals** rather than against the prediction that
/// produced `H`.
///
/// A horizon is only a guess about arrival, and sealing on a guess is
/// how consistency breaks. So this seals a tick either cleanly (every
/// required peer delivered, possibly well before the deadline), by
/// repair (reconstructed from parity — the loss was invisible), or
/// degraded, naming the peers. Too short a horizon costs degraded
/// seals and repair work; it never costs silent divergence.
#[derive(Clone, Debug)]
pub struct Barrier {
    horizon: u64,
    peers: usize,
    now: u64,
    sealed_through: Option<u64>,
    delivered: BTreeMap<u64, u64>,
    reconstructed: BTreeMap<u64, u64>,
}

impl Barrier {
    /// A barrier over `peers` contributors with horizon `H` ticks.
    pub fn new(peers: usize, horizon: u64) -> Result<Self> {
        if peers == 0 || peers > 64 {
            return Err(Error::InvalidState(format!(
                "barrier: {peers} peers outside 1..=64 (the arrival bitset)"
            )));
        }
        Ok(Barrier {
            horizon,
            peers,
            now: 0,
            sealed_through: None,
            delivered: BTreeMap::new(),
            reconstructed: BTreeMap::new(),
        })
    }

    /// The horizon in ticks.
    pub fn horizon(&self) -> u64 {
        self.horizon
    }

    /// The last tick sealed, if any.
    pub fn sealed_through(&self) -> Option<u64> {
        self.sealed_through
    }

    /// Advance wall-clock.
    pub fn advance(&mut self, now: u64) {
        self.now = self.now.max(now);
    }

    /// Record a peer's contribution for a tick as received.
    pub fn deliver(&mut self, peer: usize, tick: u64) -> Result<()> {
        self.mark(peer, tick, false)
    }

    /// Record a peer's contribution as reconstructed from parity
    /// rather than received — the distinction the seal reports.
    pub fn reconstruct(&mut self, peer: usize, tick: u64) -> Result<()> {
        self.mark(peer, tick, true)
    }

    fn mark(&mut self, peer: usize, tick: u64, repaired: bool) -> Result<()> {
        if peer >= self.peers {
            return Err(Error::InvalidState(format!(
                "barrier: peer {peer} outside 0..{}",
                self.peers
            )));
        }
        let map = if repaired {
            &mut self.reconstructed
        } else {
            &mut self.delivered
        };
        *map.entry(tick).or_insert(0) |= 1 << peer;
        Ok(())
    }

    fn outstanding(&self, tick: u64) -> Vec<usize> {
        let have = self.delivered.get(&tick).copied().unwrap_or(0)
            | self.reconstructed.get(&tick).copied().unwrap_or(0);
        (0..self.peers).filter(|&p| have & (1 << p) == 0).collect()
    }

    /// The next tick to seal, and how.
    pub fn try_seal(&mut self) -> Seal {
        let tick = self.sealed_through.map_or(0, |t| t + 1);
        let missing = self.outstanding(tick);
        if missing.is_empty() {
            self.sealed_through = Some(tick);
            let repaired = self.reconstructed.get(&tick).copied().unwrap_or(0);
            if repaired == 0 {
                Seal::Clean { tick }
            } else {
                Seal::Repaired {
                    tick,
                    reconstructed: (0..self.peers)
                        .filter(|&p| repaired & (1 << p) != 0)
                        .collect(),
                }
            }
        } else if self.now >= tick + self.horizon {
            self.sealed_through = Some(tick);
            Seal::Degraded { tick, missing }
        } else {
            Seal::Waiting { tick, missing }
        }
    }
}

// ─────────────────────────── GF(256), exactly ───────────────────────────

/// `GF(2⁸)` under `x⁸ + x⁴ + x³ + x² + 1`, with `2` primitive.
///
/// Integer arithmetic, no floats, no tolerance — a repair is either
/// exactly right or refused. (`exceptional-galois`' discipline: over a
/// finite field every statement is an exact fact about integers, and
/// there is nothing to round.)
mod gf256 {
    /// `α^i` for `i` in `0..510`, `α = 2`, so a product needs no
    /// modular reduction on the exponent.
    pub struct Tables {
        exp: [u8; 512],
        log: [u8; 256],
    }

    impl Tables {
        pub const fn poly() -> u16 {
            0x11d
        }

        pub fn new() -> Self {
            let mut exp = [0u8; 512];
            let mut log = [0u8; 256];
            let mut x: u16 = 1;
            for (i, slot) in exp.iter_mut().enumerate().take(255) {
                *slot = x as u8;
                log[x as usize] = i as u8;
                x <<= 1;
                if x & 0x100 != 0 {
                    x ^= Self::poly();
                }
            }
            for i in 255..512 {
                exp[i] = exp[i - 255];
            }
            Tables { exp, log }
        }

        pub fn mul(&self, a: u8, b: u8) -> u8 {
            if a == 0 || b == 0 {
                0
            } else {
                self.exp[self.log[a as usize] as usize + self.log[b as usize] as usize]
            }
        }

        pub fn inv(&self, a: u8) -> u8 {
            debug_assert!(a != 0, "GF(256): zero has no inverse");
            self.exp[255 - self.log[a as usize] as usize]
        }
    }
}

/// A systematic MDS code across a window: `k` data shards, `m` parity
/// shards, any `k` of the `k + m` recovering everything.
///
/// The generator is a **Cauchy** matrix over GF(256),
/// `C[i][j] = (x_i ⊕ y_j)⁻¹` with the `x` and `y` sets disjoint. Every
/// square submatrix of a Cauchy matrix is invertible, so "any `k` of
/// `n`" is a property of the construction rather than a hope about the
/// particular evaluation points — which is what makes the repair
/// either exact or a named refusal.
#[derive(Clone, Debug)]
pub struct WindowParity {
    k: usize,
    m: usize,
    /// Row-major `m × k`.
    cauchy: Vec<u8>,
}

impl WindowParity {
    /// A `(k, m)` code. `k + m ≤ 256`, since the evaluation points are
    /// distinct field elements.
    pub fn new(k: usize, m: usize) -> Result<Self> {
        if k == 0 || m == 0 || k + m > 256 {
            return Err(Error::InvalidState(format!(
                "window parity: (k, m) = ({k}, {m}) — both must be positive and \
                 k + m ≤ 256 distinct GF(256) evaluation points"
            )));
        }
        let t = gf256::Tables::new();
        let mut cauchy = vec![0u8; m * k];
        for i in 0..m {
            for j in 0..k {
                let x = i as u8;
                let y = (m + j) as u8;
                cauchy[i * k + j] = t.inv(x ^ y);
            }
        }
        Ok(WindowParity { k, m, cauchy })
    }

    /// Data shards per window.
    pub fn data_shards(&self) -> usize {
        self.k
    }

    /// Parity shards per window.
    pub fn parity_shards(&self) -> usize {
        self.m
    }

    /// Bandwidth overhead as a fraction of the data stream.
    pub fn overhead(&self) -> f64 {
        self.m as f64 / self.k as f64
    }

    /// The `m` parity shards for `k` equal-length data shards.
    pub fn encode(&self, data: &[Vec<u8>]) -> Result<Vec<Vec<u8>>> {
        if data.len() != self.k {
            return Err(Error::InvalidState(format!(
                "window parity: {} data shards for a k = {} code",
                data.len(),
                self.k
            )));
        }
        let len = data[0].len();
        if data.iter().any(|d| d.len() != len) {
            return Err(Error::InvalidState(
                "window parity: shards must be equal length — the wire form is fixed \
                 size for exactly this reason"
                    .into(),
            ));
        }
        let t = gf256::Tables::new();
        let mut out = vec![vec![0u8; len]; self.m];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, d) in data.iter().enumerate() {
                let c = self.cauchy[i * self.k + j];
                if c == 0 {
                    continue;
                }
                for (o, &v) in row.iter_mut().zip(d.iter()) {
                    *o ^= t.mul(c, v);
                }
            }
        }
        Ok(out)
    }

    /// Fill every missing data shard from whatever data and parity
    /// arrived.
    ///
    /// Refuses by name when there is not enough parity — the caller
    /// learns *which* shards it still does not have, which is what the
    /// cross-layer iteration needs.
    pub fn repair(
        &self,
        data: &mut [Option<Vec<u8>>],
        parity: &[Option<Vec<u8>>],
    ) -> Result<usize> {
        if data.len() != self.k || parity.len() != self.m {
            return Err(Error::InvalidState(format!(
                "window parity: {} data / {} parity slots for a ({}, {}) code",
                data.len(),
                parity.len(),
                self.k,
                self.m
            )));
        }
        let missing: Vec<usize> = (0..self.k).filter(|&j| data[j].is_none()).collect();
        if missing.is_empty() {
            return Ok(0);
        }
        let have: Vec<usize> = (0..self.m).filter(|&i| parity[i].is_some()).collect();
        if have.len() < missing.len() {
            return Err(Error::InvalidState(format!(
                "window parity: {} shard(s) missing with {} parity shard(s) present — \
                 an MDS code repairs exactly as many erasures as it has parity, and \
                 this window is over budget",
                missing.len(),
                have.len()
            )));
        }
        let len = data
            .iter()
            .flatten()
            .chain(parity.iter().flatten())
            .map(|s| s.len())
            .next()
            .unwrap_or(0);
        let t = gf256::Tables::new();
        let f = missing.len();

        // rhs[r] = parity_row - Σ_{known j} C[row][j]·data[j]
        let mut a = vec![vec![0u8; f]; f];
        let mut rhs = vec![vec![0u8; len]; f];
        for (r, &row) in have.iter().take(f).enumerate() {
            for (c, &j) in missing.iter().enumerate() {
                a[r][c] = self.cauchy[row * self.k + j];
            }
            let p = parity[row].as_ref().unwrap();
            if p.len() != len {
                return Err(Error::InvalidState(
                    "window parity: parity shard length differs from data shard length".into(),
                ));
            }
            rhs[r].copy_from_slice(p);
            for (j, slot) in data.iter().enumerate() {
                if let Some(d) = slot {
                    let c = self.cauchy[row * self.k + j];
                    if c == 0 {
                        continue;
                    }
                    for b in 0..len {
                        rhs[r][b] ^= t.mul(c, d[b]);
                    }
                }
            }
        }

        // Gauss–Jordan over GF(256); the Cauchy submatrix is invertible
        // by construction, so a zero pivot column would be a bug, not a
        // data condition — and it is reported as a refusal either way.
        for col in 0..f {
            let pivot = (col..f).find(|&r| a[r][col] != 0).ok_or_else(|| {
                Error::InvalidState(
                    "window parity: singular Cauchy submatrix — every square submatrix of \
                     a Cauchy matrix is invertible, so this is a construction bug, not a \
                     loss pattern"
                        .into(),
                )
            })?;
            a.swap(col, pivot);
            rhs.swap(col, pivot);
            let inv = t.inv(a[col][col]);
            for v in a[col][col..f].iter_mut() {
                *v = t.mul(*v, inv);
            }
            for v in rhs[col].iter_mut() {
                *v = t.mul(*v, inv);
            }
            let pivot_row = a[col][col..f].to_vec();
            let pivot_rhs = rhs[col].clone();
            for r in 0..f {
                if r == col || a[r][col] == 0 {
                    continue;
                }
                let factor = a[r][col];
                for (v, &pv) in a[r][col..f].iter_mut().zip(pivot_row.iter()) {
                    *v ^= t.mul(factor, pv);
                }
                for (v, &pv) in rhs[r].iter_mut().zip(pivot_rhs.iter()) {
                    *v ^= t.mul(factor, pv);
                }
            }
        }

        for (c, &j) in missing.iter().enumerate() {
            data[j] = Some(rhs[c].clone());
        }
        Ok(f)
    }
}

// ──────────────────────── the two layers, crossed ────────────────────────

/// One window of the distributed record: `nodes × ticks` frame-delta
/// slots, the coarse parity that covers each tick across the nodes,
/// the scheduled support of each slot, and each node's fine residual.
#[derive(Clone, Debug)]
pub struct RepairWindow {
    nodes: usize,
    ticks: usize,
    base: u64,
    slots: Vec<Option<PauliString>>,
    parity: Vec<Vec<Option<Vec<u8>>>>,
    supports: Vec<Vec<usize>>,
    residual: Vec<u64>,
}

impl RepairWindow {
    /// An empty window of `ticks` ticks starting at `base`, over
    /// `nodes` nodes, with `parity_shards` coarse shards per tick.
    pub fn new(nodes: usize, ticks: usize, base: u64, parity_shards: usize) -> Result<Self> {
        if nodes == 0 || ticks == 0 {
            return Err(Error::InvalidState(
                "repair window: a window needs at least one node and one tick".into(),
            ));
        }
        Ok(RepairWindow {
            nodes,
            ticks,
            base,
            slots: vec![None; nodes * ticks],
            parity: vec![vec![None; parity_shards]; ticks],
            supports: vec![Vec::new(); nodes * ticks],
            residual: vec![0; nodes],
        })
    }

    fn index(&self, node: usize, tick: u64) -> Result<usize> {
        let t = tick.checked_sub(self.base).map(|d| d as usize);
        match (node < self.nodes, t) {
            (true, Some(t)) if t < self.ticks => Ok(node * self.ticks + t),
            _ => Err(Error::InvalidState(format!(
                "repair window: (node {node}, tick {tick}) outside {} nodes × ticks \
                 {}..{}",
                self.nodes,
                self.base,
                self.base + self.ticks as u64
            ))),
        }
    }

    /// Nodes in the window.
    pub fn nodes(&self) -> usize {
        self.nodes
    }

    /// Ticks in the window.
    pub fn ticks(&self) -> usize {
        self.ticks
    }

    /// Record a delta that arrived.
    pub fn deliver(&mut self, node: usize, tick: u64, delta: PauliString) -> Result<()> {
        let i = self.index(node, tick)?;
        self.slots[i] = Some(delta);
        Ok(())
    }

    /// Record the qubits the schedule says this slot's operation could
    /// touch. These are the *erasure locations*: known from the
    /// program, not from the packet, which is exactly why the loss is
    /// an erasure and not an error.
    pub fn set_support(&mut self, node: usize, tick: u64, support: Vec<usize>) -> Result<()> {
        let i = self.index(node, tick)?;
        self.supports[i] = support;
        Ok(())
    }

    /// Store one coarse parity shard for a tick.
    pub fn set_parity(&mut self, tick: u64, shard: usize, bytes: Vec<u8>) -> Result<()> {
        let t = tick
            .checked_sub(self.base)
            .map(|d| d as usize)
            .filter(|&t| t < self.ticks)
            .ok_or_else(|| {
                Error::InvalidState(format!(
                    "repair window: parity for tick {tick} out of range"
                ))
            })?;
        if shard >= self.parity[t].len() {
            return Err(Error::InvalidState(format!(
                "repair window: parity shard {shard} beyond the {} configured",
                self.parity[t].len()
            )));
        }
        self.parity[t][shard] = Some(bytes);
        Ok(())
    }

    /// Set a node's **fine residual**: the syndrome its own state
    /// shows, XOR the syndrome its believed frame predicts.
    ///
    /// By [`SyndromeMap`]'s linearity this equals the syndrome of the
    /// XOR of the deltas the node is missing — one 𝔽₂ equation per
    /// node, obtained by an ancilla-free local read and costing **no
    /// bandwidth at all**.
    pub fn set_residual(&mut self, node: usize, residual: u64) -> Result<()> {
        if node >= self.nodes {
            return Err(Error::InvalidState(format!(
                "repair window: node {node} outside 0..{}",
                self.nodes
            )));
        }
        self.residual[node] = residual;
        Ok(())
    }

    /// The delta in a slot, if it is known.
    pub fn slot(&self, node: usize, tick: u64) -> Result<Option<PauliString>> {
        Ok(self.slots[self.index(node, tick)?])
    }

    /// Every `(node, tick)` still unknown.
    pub fn missing(&self) -> Vec<(usize, u64)> {
        let mut out = Vec::new();
        for n in 0..self.nodes {
            for t in 0..self.ticks {
                if self.slots[n * self.ticks + t].is_none() {
                    out.push((n, self.base + t as u64));
                }
            }
        }
        out
    }
}

/// What a cross-layer repair did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RepairReport {
    /// Iterations before neither layer could move.
    pub rounds: usize,
    /// Slots filled by the coarse (paid, per-tick, across-nodes) layer.
    pub coarse_repairs: usize,
    /// Slots filled by the fine (free, per-node, across-ticks) layer.
    pub fine_repairs: usize,
    /// Slots the fine layer *could* have addressed but declined
    /// because the recovery was not unique on the nose — see
    /// [`DualLayer::repair`].
    pub fine_declined: usize,
    /// Slots neither layer resolved.
    pub unresolved: Vec<(usize, u64)>,
}

/// The dual tower, opposed in scale: a **fine** layer that is local to
/// a node and spans ticks, crossed with a **coarse** layer that is
/// local to a tick and spans nodes.
///
/// * **fine** — the stabilizer code itself. One 𝔽₂ equation per node,
///   read off the node's own state with no ancillas and no bandwidth.
///   Resolves a node whose unknowns have narrowed to one, whatever the
///   window boundaries are.
/// * **coarse** — [`WindowParity`] across the nodes at one tick.
///   Resolves up to `m` missing nodes per tick, whatever their
///   supports are, and costs `m/nodes` in bandwidth.
///
/// Each has a loss pattern the other finds trivial. A tick that loses
/// more than `m` nodes at once defeats the coarse layer; a node that
/// loses several consecutive ticks defeats the fine one. Crossed, each
/// layer's successes shrink the other's unknown count, and patterns
/// that defeat both *individually* fall to the iteration. (The
/// construction is `holochron`'s dual tower, with the quantum code
/// standing in for the fine field-scale layer — which is the point:
/// the code protecting the qubits is already a layer of the code
/// protecting the network.)
pub struct DualLayer<'a> {
    fine: &'a [ErasureDecoder],
    syn: &'a [SyndromeMap],
    coarse: WindowParity,
}

impl<'a> DualLayer<'a> {
    /// A dual layer over per-node decoders and syndrome maps, with a
    /// coarse code of `parity_shards` across the nodes.
    pub fn new(
        fine: &'a [ErasureDecoder],
        syn: &'a [SyndromeMap],
        parity_shards: usize,
    ) -> Result<Self> {
        if fine.len() != syn.len() {
            return Err(Error::InvalidState(format!(
                "dual layer: {} decoders against {} syndrome maps",
                fine.len(),
                syn.len()
            )));
        }
        let coarse = WindowParity::new(fine.len(), parity_shards)?;
        Ok(DualLayer { fine, syn, coarse })
    }

    /// The coarse code.
    pub fn coarse(&self) -> &WindowParity {
        &self.coarse
    }

    /// Encode one tick's coarse parity across the nodes.
    pub fn encode_tick(&self, deltas: &[FrameDelta]) -> Result<Vec<Vec<u8>>> {
        let shards: Vec<Vec<u8>> = deltas.iter().map(|d| d.to_bytes().to_vec()).collect();
        self.coarse.encode(&shards)
    }

    /// Cross-decode a window until neither layer can move.
    ///
    /// **The clause that governs the fine layer.** An erasure decode
    /// is unique only up to the stabilizers supported on the erased
    /// set. For the *code space* a stabilizer-equivalent frame is as
    /// good as the truth — it acts identically on every logical
    /// observable. For the *coarse layer downstream* it is not: that
    /// layer is doing linear algebra on the bytes, and a
    /// stabilizer-equivalent representative would poison every slot it
    /// then solves for. So the fine layer fills a slot only where the
    /// recovery is unique on the nose
    /// ([`ErasureDecoder::unique`]), and counts the rest in
    /// [`RepairReport::fine_declined`] rather than guessing.
    ///
    /// The second clause is [`retro`](crate::retro)'s, one level up: a
    /// syndrome is **sign-blind**. The fine layer recovers a delta's
    /// masks and assumes the sign is positive, which is exact for the
    /// deltas this protocol puts on the wire — a node's own physical
    /// fault is a Hermitian Pauli with no sign, and every sign in the
    /// system is generated *locally* by [`LateralLink::push_frame`]
    /// from windows both endpoints already hold. A signed delta is
    /// outside what a syndrome can name, there as here.
    pub fn repair(&self, w: &mut RepairWindow) -> Result<RepairReport> {
        if w.nodes != self.fine.len() {
            return Err(Error::InvalidState(format!(
                "dual layer: window has {} nodes, the layer has {}",
                w.nodes,
                self.fine.len()
            )));
        }
        if w.parity
            .iter()
            .any(|p| p.len() != self.coarse.parity_shards())
        {
            return Err(Error::InvalidState(format!(
                "dual layer: window carries parity slots the coarse code did not size — \
                 expected {} per tick",
                self.coarse.parity_shards()
            )));
        }
        let mut report = RepairReport::default();
        loop {
            report.rounds += 1;
            let mut progress = false;

            // ── coarse: per tick, across nodes ──
            for t in 0..w.ticks {
                let mut data: Vec<Option<Vec<u8>>> = (0..w.nodes)
                    .map(|n| {
                        w.slots[n * w.ticks + t].map(|p| {
                            FrameDelta {
                                node: n,
                                tick: w.base + t as u64,
                                delta: p,
                            }
                            .to_bytes()
                            .to_vec()
                        })
                    })
                    .collect();
                if data.iter().all(Option::is_some) {
                    continue;
                }
                if self.coarse.repair(&mut data, &w.parity[t]).is_err() {
                    continue;
                }
                for (n, shard) in data.iter().enumerate() {
                    if w.slots[n * w.ticks + t].is_none() {
                        let d = FrameDelta::from_bytes(shard.as_ref().unwrap())?;
                        if d.node != n || d.tick != w.base + t as u64 {
                            return Err(Error::InvalidState(format!(
                                "dual layer: the coarse layer reconstructed a shard \
                                 labelled (node {}, tick {}) into the slot for (node {n}, \
                                 tick {}) — the window and the parity disagree about what \
                                 they cover",
                                d.node,
                                d.tick,
                                w.base + t as u64
                            )));
                        }
                        w.slots[n * w.ticks + t] = Some(d.delta);
                        w.residual[n] ^= self.syn[n].bits(d.delta);
                        report.coarse_repairs += 1;
                        progress = true;
                    }
                }
            }

            // ── fine: per node, across ticks ──
            for n in 0..w.nodes {
                let miss: Vec<usize> = (0..w.ticks)
                    .filter(|&t| w.slots[n * w.ticks + t].is_none())
                    .collect();
                if miss.len() != 1 {
                    continue;
                }
                let t = miss[0];
                let support = w.supports[n * w.ticks + t].clone();
                if support.is_empty() || !self.fine[n].unique(&support) {
                    if !support.is_empty() {
                        report.fine_declined += 1;
                    }
                    continue;
                }
                let Ok(p) = self.fine[n].decode(&support, w.residual[n]) else {
                    continue;
                };
                w.slots[n * w.ticks + t] = Some(p);
                w.residual[n] ^= self.syn[n].bits(p);
                report.fine_repairs += 1;
                progress = true;
            }

            if !progress {
                break;
            }
        }
        report.unresolved = w.missing();
        Ok(report)
    }
}

// ────────────────────────────── the network ──────────────────────────────

/// One step of a distributed encoded program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LateralOp {
    /// A physical Pauli on one node — a deliberate logical operation
    /// or a fault; the frame does not distinguish them, which is the
    /// whole economy of a frame. Absorbed at any weight, in
    /// bookkeeping, with no state work.
    Local {
        /// Which node.
        node: usize,
        /// The Pauli, as masks over the whole register.
        pauli: PauliString,
    },
    /// A transversal CX across a link — the cross-lateral operation,
    /// and the only thing in the vocabulary that couples nodes.
    Cross {
        /// Index into [`LateralNetwork::links`].
        link: usize,
    },
}

/// What one distributed run produced.
#[derive(Clone, Debug)]
pub struct DistributedRun {
    /// The global frame at the end.
    pub frame: PauliString,
    /// One delta per [`LateralOp::Local`], in program order.
    pub deltas: Vec<FrameDelta>,
    /// Bytes that crossed the wire — `deltas.len() · DELTA_WIRE_LEN`,
    /// plus coarse parity if any was configured.
    pub wire_bytes: usize,
    /// The physical CX gates the program's cross operations compile
    /// to, in order — the link fabric, and the only place two nodes'
    /// qubits ever meet. Local Paulis are absent by construction: a
    /// frame absorbs them without a gate.
    pub steps: Vec<CliffordStep>,
}

/// A register split across nodes, coupled only laterally.
///
/// The nodes hold disjoint windows of one register. The only inter-node
/// operation is a transversal CX ([`LateralLink`]), so the only
/// inter-node traffic is 𝔽₂: [`FrameDelta`]s of
/// [`DELTA_WIRE_LEN`] bytes each, independent of the register's width.
/// [`amplitude_bytes`](Self::amplitude_bytes) prices the alternative.
#[derive(Clone, Debug)]
pub struct LateralNetwork {
    patches: Vec<(usize, usize)>,
    links: Vec<LateralLink>,
    geometry: DelayGeometry,
}

impl LateralNetwork {
    /// A network over `patches` — `(offset, qubits)` per node —
    /// with the given links and delay geometry.
    pub fn new(
        patches: Vec<(usize, usize)>,
        links: Vec<LateralLink>,
        geometry: DelayGeometry,
    ) -> Result<Self> {
        if patches.is_empty() {
            return Err(Error::InvalidState("lateral network: no nodes".into()));
        }
        if geometry.nodes() != patches.len() {
            return Err(Error::InvalidState(format!(
                "lateral network: {} patches against a {}-node delay geometry",
                patches.len(),
                geometry.nodes()
            )));
        }
        for (i, &(off, q)) in patches.iter().enumerate() {
            for (j, &(off2, q2)) in patches.iter().enumerate().skip(i + 1) {
                if off < off2 + q2 && off2 < off + q {
                    return Err(Error::InvalidState(format!(
                        "lateral network: node {i} at {off}..{} overlaps node {j} at \
                         {off2}..{} — nodes hold disjoint windows",
                        off + q,
                        off2 + q2
                    )));
                }
            }
        }
        for (i, l) in links.iter().enumerate() {
            let known = |o: usize| patches.iter().any(|&(off, _)| off == o);
            if !known(l.control()) || !known(l.target()) {
                return Err(Error::InvalidState(format!(
                    "lateral network: link {i} joins offsets {} and {} — neither is a \
                     node in this network",
                    l.control(),
                    l.target()
                )));
            }
        }
        Ok(LateralNetwork {
            patches,
            links,
            geometry,
        })
    }

    /// Node count.
    pub fn nodes(&self) -> usize {
        self.patches.len()
    }

    /// `(offset, qubits)` of each node.
    pub fn patches(&self) -> &[(usize, usize)] {
        &self.patches
    }

    /// The links.
    pub fn links(&self) -> &[LateralLink] {
        &self.links
    }

    /// The delay geometry.
    pub fn geometry(&self) -> &DelayGeometry {
        &self.geometry
    }

    /// Total register width.
    pub fn width(&self) -> usize {
        self.patches
            .iter()
            .map(|&(off, q)| off + q)
            .max()
            .unwrap_or(0)
    }

    /// Bytes a dense amplitude vector of this register would occupy —
    /// what a naive "ship the state" protocol would move per
    /// synchronization, against the [`DELTA_WIRE_LEN`] a frame delta
    /// costs. Saturates rather than overflowing, because at these
    /// widths the number is the point.
    pub fn amplitude_bytes(&self) -> u128 {
        1u128
            .checked_shl(self.width() as u32)
            .map_or(u128::MAX, |states| states.saturating_mul(16))
    }

    /// Which node owns a qubit.
    pub fn owner(&self, qubit: usize) -> Option<usize> {
        self.patches
            .iter()
            .position(|&(off, q)| qubit >= off && qubit < off + q)
    }

    /// Run a program as a distributed frame evolution: local Paulis
    /// XOR into the frame, cross operations apply
    /// [`LateralLink::push_frame`]. Every step is 𝔽₂ mask algebra;
    /// no amplitude is touched and nothing but [`FrameDelta`]s crosses
    /// the wire.
    pub fn run(&self, program: &[LateralOp]) -> Result<DistributedRun> {
        let mut frame = PauliString::identity();
        let mut deltas = Vec::new();
        let mut steps = Vec::new();
        for (tick, op) in program.iter().enumerate() {
            let tick = tick as u64;
            match *op {
                LateralOp::Local { node, pauli } => {
                    let &(off, q) = self.patches.get(node).ok_or_else(|| {
                        Error::InvalidState(format!(
                            "lateral run: node {node} outside 0..{}",
                            self.patches.len()
                        ))
                    })?;
                    let window = window_mask(off, q);
                    if (pauli.x | pauli.z) & !window != 0 {
                        return Err(Error::InvalidState(format!(
                            "lateral run: node {node}'s Pauli reaches outside its window \
                             {off}..{} — a local operation is local, and a frame that \
                             silently spanned nodes would make the wire a lie",
                            off + q
                        )));
                    }
                    frame = xor_masks(frame, pauli);
                    deltas.push(FrameDelta {
                        node,
                        tick,
                        delta: pauli,
                    });
                }
                LateralOp::Cross { link } => {
                    let l = self.links.get(link).ok_or_else(|| {
                        Error::InvalidState(format!(
                            "lateral run: link {link} outside 0..{}",
                            self.links.len()
                        ))
                    })?;
                    frame = l.push_frame(frame);
                    steps.extend(l.steps());
                }
            }
        }
        let wire_bytes = deltas.len() * DELTA_WIRE_LEN;
        Ok(DistributedRun {
            frame,
            deltas,
            wire_bytes,
            steps,
        })
    }

    /// The same program with every cross operation expanded into its
    /// physical CX gates and folded one at a time — the oracle the
    /// coordinatewise rule is measured against.
    pub fn run_by_gates(&self, program: &[LateralOp]) -> Result<PauliString> {
        let mut frame = PauliString::identity();
        for op in program {
            match *op {
                LateralOp::Local { pauli, .. } => frame = xor_masks(frame, pauli),
                LateralOp::Cross { link } => {
                    let l = self.links.get(link).ok_or_else(|| {
                        Error::InvalidState(format!("lateral run: link {link} out of range"))
                    })?;
                    frame = l.push_frame_by_gates(frame);
                }
            }
        }
        Ok(frame)
    }
}

/// The register mask of a `(offset, qubits)` window.
pub fn window_mask(offset: usize, qubits: usize) -> u64 {
    if qubits == 0 || offset >= 64 {
        return 0;
    }
    let width = qubits.min(64 - offset);
    let m = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    m << offset
}
