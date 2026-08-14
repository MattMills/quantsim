//! Retrocorrection: the code as a constraint on the record.
//!
//! Error correction on hardware is a forward struggle: syndromes are
//! noisy ancilla measurements, decoding is statistical, and the past is
//! gone. In a **simulator** every one of those constraints inverts:
//!
//! * **Syndromes are deterministic reads.** A state in (or one Pauli
//!   off) the code space is an exact eigenstate of every stabilizer
//!   generator, so [`syndromes`] is a set of `±1` expectations computed
//!   without ancillas, without projection, and without disturbing the
//!   state — non-demolition because nothing is demolished.
//! * **The record exists.** A branched register holds the history as
//!   slices. A correction decoded at the *end* of the record conjugates
//!   backward through the intervening Clifford dynamics
//!   ([`transport_back`]) and repairs every stored slice — the past is
//!   corrected, not merely compensated. One reading fixes the whole
//!   history.
//! * **Prediction is constrained.** The transported error's
//!   anticommutation pattern against the generators *is* the syndrome
//!   the final slice must show — computable from the string alone,
//!   before any state is read. The code predicts the record.
//!
//! The code here is the rotated surface code ([`SurfaceCode`]): `d²`
//! data qubits per patch, `d² − 1` weight-≤4 CSS generators, one
//! logical qubit, distance `d`. Patches take a qubit `offset`, so
//! several live side by side in one register and transversal CX
//! between patches is a *logical* CX — the logical entanglement graph
//! is built from physical gates and verified from logical stabilizer
//! expectations.
//!
//! The decoder's lookup table is not hard-coded: it is **measured from
//! the code's own generators** at construction — every weight-1 Pauli's
//! syndrome signature, enumerated and inverted. A syndrome outside the
//! table is refused by name, never guessed at: this instrument decodes
//! up to the distance and says so.
//!
//! Transport is Clifford-only and [`compile_clifford`] refuses anything
//! else by name (a `t` would gadgetize an ancilla — the honest boundary
//! of exact Pauli-frame retrocorrection).
//!
//! One clause the syndrome can never supply: the **sign**. Dynamics
//! that anticommute with a fault flip its residual to `−P`, and every
//! generator expectation is blind to that — on hardware the sign is an
//! unobservable global phase, but a branched record makes slice phases
//! physical (the clock's phase-faithfulness contract). The record
//! resolves what the syndrome cannot: each slice must equal its
//! predecessor pushed through the segment, the last uncorrupted slice
//! anchors the chain, and one amplitude comparison per slice names the
//! sign. Syndromes name the correction up to sign; the record's own
//! consistency names the sign.
//!
//! The torus ([`ToricCode`]) is the multi-node currency: `2L²` qubits,
//! **two** logical wires per physical set, so one transversal CX
//! between two nodes raises two logical Bell links at once — and a
//! logical link is a rank-2 Schmidt decomposition across the node cut,
//! which is exactly what a selector qudit holds: branches that are
//! *products* of per-node code states, the selector carrying the
//! entanglement, `log₂(selector rank)` = links.

use crate::backend::{conjugate_by_step, Backend, CliffordStep, PauliString};
use crate::circuit::Circuit;
use crate::coupling::Stabilizer;
use crate::error::{Error, Result};
use crate::gates::Pauli;
use crate::registry::GateRegistry;
use crate::scalar::C64;
use crate::upembed;

/// What the syndrome and decoding machinery asks of a code: its
/// generators and where its qubits live. [`SurfaceCode`] and
/// [`ToricCode`] both answer; everything downstream — [`syndromes`],
/// [`signature`], [`Decoder`] — is generic over it.
pub trait Code {
    /// All stabilizer generators (redundant generators welcome — the
    /// syndrome just grows a consistent bit).
    fn generators(&self) -> Vec<PauliString>;
    /// First register index of this code's qubits.
    fn offset(&self) -> usize;
    /// Number of data qubits.
    fn qubits(&self) -> usize;
}

/// One rotated surface-code patch: `d²` data qubits starting at
/// `offset` in the enclosing register.
#[derive(Clone, Debug)]
pub struct SurfaceCode {
    d: usize,
    offset: usize,
    x_gens: Vec<PauliString>,
    z_gens: Vec<PauliString>,
    logical_x: PauliString,
    logical_z: PauliString,
}

fn x_string(mask: u64) -> PauliString {
    PauliString {
        x: mask,
        z: 0,
        negative: false,
    }
}

fn z_string(mask: u64) -> PauliString {
    PauliString {
        x: 0,
        z: mask,
        negative: false,
    }
}

impl SurfaceCode {
    /// The rotated distance-`d` surface code (odd `d ≥ 3`), qubit
    /// `(r, c)` at register index `offset + r·d + c`.
    ///
    /// Bulk faces checkerboard between X- and Z-type; weight-2 boundary
    /// checks close the pattern on alternating edges. Generator count
    /// is `d² − 1`, so exactly one logical qubit survives; the test
    /// suite verifies commutativity, rank, and the logical pair's
    /// anticommutation rather than trusting this constructor.
    pub fn new(d: usize, offset: usize) -> Result<Self> {
        if d < 3 || d % 2 == 0 {
            return Err(Error::InvalidState(format!(
                "surface code: distance must be odd and ≥ 3, got {d}"
            )));
        }
        if offset + d * d > 64 {
            return Err(Error::InvalidState(format!(
                "surface code: patch [{offset}, {}) exceeds the 64-qubit Pauli mask",
                offset + d * d
            )));
        }
        let at = |r: usize, c: usize| -> u64 { 1u64 << (offset + r * d + c) };
        let mut x_gens = Vec::new();
        let mut z_gens = Vec::new();
        // Bulk faces: corners {(r,c),(r,c+1),(r+1,c),(r+1,c+1)},
        // X-type when r + c is even.
        for r in 0..d - 1 {
            for c in 0..d - 1 {
                let mask = at(r, c) | at(r, c + 1) | at(r + 1, c) | at(r + 1, c + 1);
                if (r + c) % 2 == 0 {
                    x_gens.push(x_string(mask));
                } else {
                    z_gens.push(z_string(mask));
                }
            }
        }
        // Boundary weight-2 checks: X pairs on the top and bottom rows
        // over the columns whose bulk neighbour is Z-type, Z pairs on
        // the left and right columns likewise.
        for c in (1..d - 1).step_by(2) {
            x_gens.push(x_string(at(0, c) | at(0, c + 1)));
        }
        for c in (0..d - 1).step_by(2) {
            x_gens.push(x_string(at(d - 1, c) | at(d - 1, c + 1)));
        }
        for r in (0..d - 1).step_by(2) {
            z_gens.push(z_string(at(r, 0) | at(r + 1, 0)));
        }
        for r in (1..d - 1).step_by(2) {
            z_gens.push(z_string(at(r, d - 1) | at(r + 1, d - 1)));
        }
        // Logical X crosses the Z boundaries (left column); logical Z
        // crosses the X boundaries (top row).
        let mut lx = 0u64;
        let mut lz = 0u64;
        for k in 0..d {
            lx |= at(k, 0);
            lz |= at(0, k);
        }
        Ok(SurfaceCode {
            d,
            offset,
            x_gens,
            z_gens,
            logical_x: x_string(lx),
            logical_z: z_string(lz),
        })
    }

    /// Code distance.
    pub fn distance(&self) -> usize {
        self.d
    }

    /// First register index of this patch.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Data qubits in this patch.
    pub fn qubits(&self) -> usize {
        self.d * self.d
    }

    /// All stabilizer generators, X-type first.
    pub fn generators(&self) -> Vec<PauliString> {
        let mut all = self.x_gens.clone();
        all.extend(self.z_gens.iter().copied());
        all
    }

    /// The logical X̄ (weight `d`, left column).
    pub fn logical_x(&self) -> PauliString {
        self.logical_x
    }

    /// The logical Z̄ (weight `d`, top row).
    pub fn logical_z(&self) -> PauliString {
        self.logical_z
    }

    /// The encoding circuit for `|0̄⟩` from `|0…0⟩` (append `x_bar:
    /// true` to prepare `|+̄⟩` instead: the same construction over the
    /// X-generators extended by X̄).
    ///
    /// CSS preparation by F2 row reduction: bring the X-type support
    /// rows to reduced echelon form; each row with pivot `p` and rest
    /// `R` becomes `H(p)` then `CX(p → r)` for `r ∈ R`. `|0…0⟩` is
    /// already a `+1` eigenstate of every Z-type generator and of Z̄,
    /// and the spread makes it one of every X-type generator too.
    pub fn encoder(&self, num_qubits: usize, x_bar: bool) -> Circuit<C64> {
        let mut rows: Vec<u64> = self.x_gens.iter().map(|g| g.x).collect();
        if x_bar {
            rows.push(self.logical_x.x);
        }
        css_encoder(rows, num_qubits)
    }

    /// Transversal CX onto another patch: physical `CX(self_q → other_q)`
    /// pairwise, which is the **logical** CX for CSS codes.
    pub fn transversal_cx(&self, other: &SurfaceCode, c: &mut Circuit<C64>) -> Result<()> {
        if self.d != other.d {
            return Err(Error::InvalidState(format!(
                "transversal cx: distances differ ({} vs {})",
                self.d, other.d
            )));
        }
        for q in 0..self.qubits() {
            c.cx(self.offset + q, other.offset + q);
        }
        Ok(())
    }
}

impl Code for SurfaceCode {
    fn generators(&self) -> Vec<PauliString> {
        SurfaceCode::generators(self)
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn qubits(&self) -> usize {
        SurfaceCode::qubits(self)
    }
}

/// The toric code on an `L × L` torus: `2L²` qubits on edges, every
/// vertex an X-check and every face a Z-check (one of each redundant),
/// so **two** logical qubits survive — the two non-contractible cycle
/// pairs of the torus. One physical set, two logical wires: the node
/// currency of a logical network.
///
/// Edge indexing at `offset`: the horizontal edge east of vertex
/// `(r, c)` is `offset + 2(rL + c)`, the vertical edge south of it is
/// `offset + 2(rL + c) + 1`.
#[derive(Clone, Debug)]
pub struct ToricCode {
    l: usize,
    offset: usize,
    x_gens: Vec<PauliString>,
    z_gens: Vec<PauliString>,
    logical_x: [PauliString; 2],
    logical_z: [PauliString; 2],
}

impl ToricCode {
    /// The `L × L` toric code (`L ≥ 2`) at qubit `offset`.
    pub fn new(l: usize, offset: usize) -> Result<Self> {
        if l < 2 {
            return Err(Error::InvalidState(format!(
                "toric code: side must be ≥ 2, got {l}"
            )));
        }
        if offset + 2 * l * l > 64 {
            return Err(Error::InvalidState(format!(
                "toric code: patch [{offset}, {}) exceeds the 64-qubit Pauli mask",
                offset + 2 * l * l
            )));
        }
        let h = |r: usize, c: usize| -> u64 { 1u64 << (offset + 2 * ((r % l) * l + (c % l))) };
        let v = |r: usize, c: usize| -> u64 { 1u64 << (offset + 2 * ((r % l) * l + (c % l)) + 1) };
        let mut x_gens = Vec::new();
        let mut z_gens = Vec::new();
        for r in 0..l {
            for c in 0..l {
                // Vertex (r, c): its east/west horizontal and
                // south/north vertical edges.
                x_gens.push(x_string(
                    h(r, c) | h(r, c + l - 1) | v(r, c) | v(r + l - 1, c),
                ));
                // Face south-east of vertex (r, c): its bounding edges.
                z_gens.push(z_string(h(r, c) | h(r + 1, c) | v(r, c) | v(r, c + 1)));
            }
        }
        // Homology, orientation by orientation: an X̄ is a cycle of the
        // DUAL lattice (it must cross every plaquette 0 or 2 times; X
        // commutes with the vertex checks for free), a Z̄ is a cycle of
        // the direct lattice (even at every vertex; free against the
        // plaquettes). Pair 1 lives on horizontal edges — Z̄₁ the
        // direct row-0 cycle, X̄₁ the dual column-0 cut — and pair 2
        // transposes onto vertical edges. Each pair overlaps in exactly
        // one edge (anticommutes); everything crosses orientations
        // disjointly (commutes).
        let mut x1 = 0u64;
        let mut z1 = 0u64;
        let mut x2 = 0u64;
        let mut z2 = 0u64;
        for k in 0..l {
            z1 |= h(0, k);
            x1 |= h(k, 0);
            z2 |= v(k, 0);
            x2 |= v(0, k);
        }
        Ok(ToricCode {
            l,
            offset,
            x_gens,
            z_gens,
            logical_x: [x_string(x1), x_string(x2)],
            logical_z: [z_string(z1), z_string(z2)],
        })
    }

    /// Torus side; the code distance is `L`.
    pub fn side(&self) -> usize {
        self.l
    }

    /// Logical X̄ᵢ for `i ∈ {0, 1}`.
    pub fn logical_x(&self, i: usize) -> PauliString {
        self.logical_x[i]
    }

    /// Logical Z̄ᵢ for `i ∈ {0, 1}`.
    pub fn logical_z(&self, i: usize) -> PauliString {
        self.logical_z[i]
    }

    /// The encoding circuit from `|0…0⟩`: `|0̄0̄⟩`, with `plus[i]`
    /// selecting `|+̄⟩` for logical `i` (the CSS construction over the
    /// vertex generators extended by the chosen X̄s).
    pub fn encoder(&self, num_qubits: usize, plus: [bool; 2]) -> Circuit<C64> {
        let mut rows: Vec<u64> = self.x_gens.iter().map(|g| g.x).collect();
        for (i, &p) in plus.iter().enumerate() {
            if p {
                rows.push(self.logical_x[i].x);
            }
        }
        css_encoder(rows, num_qubits)
    }

    /// Transversal CX onto another torus: pairwise physical CX, which
    /// is the logical CX on **both** logical wires at once — one
    /// physical operation, two network links.
    pub fn transversal_cx(&self, other: &ToricCode, c: &mut Circuit<C64>) -> Result<()> {
        if self.l != other.l {
            return Err(Error::InvalidState(format!(
                "transversal cx: sides differ ({} vs {})",
                self.l, other.l
            )));
        }
        for q in 0..self.qubits() {
            c.cx(self.offset + q, other.offset + q);
        }
        Ok(())
    }
}

impl Code for ToricCode {
    fn generators(&self) -> Vec<PauliString> {
        let mut all = self.x_gens.clone();
        all.extend(self.z_gens.iter().copied());
        all
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn qubits(&self) -> usize {
        2 * self.l * self.l
    }
}

/// CSS preparation by F2 row reduction: bring the X-type support rows
/// to reduced echelon form (dependent rows fall out on their own);
/// each surviving row with pivot `p` and rest `R` becomes `H(p)` then
/// `CX(p → r)` for `r ∈ R`.
fn css_encoder(rows: Vec<u64>, num_qubits: usize) -> Circuit<C64> {
    let mut reduced: Vec<u64> = Vec::new();
    for mut row in rows {
        loop {
            if row == 0 {
                break;
            }
            let pivot = row.trailing_zeros();
            match reduced.iter().position(|&r| r.trailing_zeros() == pivot) {
                Some(i) => row ^= reduced[i],
                None => {
                    reduced.push(row);
                    break;
                }
            }
        }
    }
    for i in 0..reduced.len() {
        let pivot_bit = 1u64 << reduced[i].trailing_zeros();
        for j in 0..reduced.len() {
            if i != j && reduced[j] & pivot_bit != 0 {
                reduced[j] ^= reduced[i];
            }
        }
    }
    let mut c = Circuit::new(num_qubits);
    for row in reduced {
        let pivot = row.trailing_zeros() as usize;
        c.h(pivot);
        let mut rest = row & !(1u64 << pivot);
        while rest != 0 {
            let q = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            c.cx(pivot, q);
        }
    }
    c
}

/// A Pauli mask pair as per-qubit observable ops for
/// [`pauli_expectation`](crate::backend::pauli_expectation).
fn ops_of(p: PauliString) -> Vec<(usize, Pauli)> {
    let mut ops = Vec::new();
    let support = p.x | p.z;
    let mut rest = support;
    while rest != 0 {
        let q = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        let bit = 1u64 << q;
        let pauli = match (p.x & bit != 0, p.z & bit != 0) {
            (true, false) => Pauli::X,
            (false, true) => Pauli::Z,
            (true, true) => Pauli::Y,
            (false, false) => unreachable!("support bit without content"),
        };
        ops.push((q, pauli));
    }
    ops
}

/// Every generator's expectation on `state` — deterministic `±1` for a
/// state that is one Pauli away from the code space. The simulator's
/// syndrome extraction: no ancillas, no randomness, nothing disturbed.
pub fn syndromes(state: &dyn Backend<C64>, code: &impl Code) -> Result<Vec<f64>> {
    code.generators()
        .iter()
        .map(|g| Ok(crate::backend::pauli_expectation(state, &ops_of(*g))?.re))
        .collect()
}

/// [`syndromes`] hardened to bits: `false` for `+1`, `true` for `−1`,
/// refusing (by value) any expectation that is not within `tol` of
/// `±1` — a state that is not one Pauli off the code space is not a
/// syndrome, and this instrument will not round it into one.
pub fn syndrome_bits(state: &dyn Backend<C64>, code: &impl Code, tol: f64) -> Result<Vec<bool>> {
    syndromes(state, code)?
        .into_iter()
        .map(|s| {
            if (s - 1.0).abs() <= tol {
                Ok(false)
            } else if (s + 1.0).abs() <= tol {
                Ok(true)
            } else {
                Err(Error::InvalidState(format!(
                    "syndrome expectation {s:.6} is not ±1: the state is not \
                     a Pauli deformation of the code space"
                )))
            }
        })
        .collect()
}

/// The syndrome signature a Pauli string would produce: one bit per
/// generator, set where the string anticommutes. This is the
/// *prediction* half of the module — computable from the string alone,
/// before any state is read.
pub fn signature(code: &impl Code, p: PauliString) -> Vec<bool> {
    code.generators()
        .iter()
        .map(|g| !g.commutes_with(p))
        .collect()
}

/// A lookup decoder whose table is measured from the code itself:
/// every weight-1 Pauli on the patch, enumerated through
/// [`signature`] and inverted — and the decoder **knows the difference
/// between degeneracy and ambiguity**. Two faults sharing a signature
/// whose product lies in the stabilizer group are one correction
/// (either restores exactly); two faults whose product is a *logical*
/// poison that signature, and decoding it refuses by name — that is
/// what running out of distance actually means, stated instead of
/// guessed through.
pub struct Decoder {
    table: Vec<(Vec<bool>, Option<PauliString>)>,
}

impl Decoder {
    /// Build the measured table for one code.
    pub fn new(code: &impl Code) -> Self {
        let group = Stabilizer::new(code.generators());
        let mut table: Vec<(Vec<bool>, Option<PauliString>)> = Vec::new();
        for q in 0..code.qubits() {
            let bit = 1u64 << (code.offset() + q);
            for p in [
                x_string(bit),
                z_string(bit),
                PauliString {
                    x: bit,
                    z: bit,
                    negative: false,
                },
            ] {
                let sig = signature(code, p);
                if sig.iter().all(|&b| !b) {
                    continue;
                }
                match table.iter_mut().find(|(s, _)| *s == sig) {
                    None => table.push((sig, Some(p))),
                    Some((_, entry)) => {
                        if let Some(prior) = entry {
                            // Same signature: degenerate only if the
                            // difference is a stabilizer element.
                            let product = (p.x ^ prior.x, p.z ^ prior.z);
                            if group.expectation(product) == 0.0 {
                                *entry = None; // a logical separates them
                            }
                        }
                    }
                }
            }
        }
        Decoder { table }
    }

    /// Unambiguously correctable syndrome signatures in the table.
    pub fn len(&self) -> usize {
        self.table.iter().filter(|(_, e)| e.is_some()).count()
    }

    /// Whether the table is empty (it never is for a valid code).
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// The correction for a syndrome; a loud refusal for a syndrome
    /// beyond the table, and a *different* loud refusal for one the
    /// code's distance cannot disambiguate — never a guess either way.
    pub fn decode(&self, syndrome: &[bool]) -> Result<PauliString> {
        if syndrome.iter().all(|&b| !b) {
            return Ok(PauliString {
                x: 0,
                z: 0,
                negative: false,
            });
        }
        match self.table.iter().find(|(s, _)| s == syndrome) {
            Some((_, Some(p))) => Ok(*p),
            Some((_, None)) => Err(Error::InvalidState(
                "decode: this syndrome is logically ambiguous at this \
                 distance — two faults separated by a logical operator \
                 produce it, and a guess would silently rewrite the \
                 logical state; refused"
                    .into(),
            )),
            None => Err(Error::InvalidState(
                "decode: syndrome outside the weight-1 table — the fault \
                 exceeds what distance and this decoder certify; refused \
                 rather than guessed"
                    .into(),
            )),
        }
    }
}

/// Compile a **Clifford** circuit to elementary steps for Pauli
/// transport, refusing anything else by name: a non-Clifford gate
/// would gadgetize magic ancillas, and a Pauli frame does not
/// transport through magic.
pub fn compile_clifford(circuit: &Circuit<C64>) -> Result<Vec<CliffordStep>> {
    let emb = upembed::gadgetize(circuit)?;
    if emb.magic() > 0 {
        return Err(Error::InvalidState(format!(
            "retro transport: circuit carries {} magic event(s); Pauli-frame \
             retrocorrection is exact on the Clifford sector only",
            emb.magic()
        )));
    }
    Ok(emb.steps().to_vec())
}

/// Conjugate a Pauli backward through later dynamics: `V† P V` for
/// `V` the composition of `later_segments` in order. A correction
/// decoded at the end of the record, transported back to slice `t` by
/// folding every later segment, acts on slice `t` exactly as the
/// end-of-record correction acts on the end.
pub fn transport_back(p: PauliString, later_segments: &[Vec<CliffordStep>]) -> PauliString {
    let mut cur = p;
    for segment in later_segments.iter().rev() {
        for &step in segment.iter().rev() {
            cur = conjugate_by_step(cur, step);
        }
    }
    cur
}

/// Apply a Pauli string to a state as physical gates — `y` where both
/// masks overlap (carrying its own `i`), `x` and `z` elsewhere, and
/// the string's sign as a global `−1`. Exact to the amplitude,
/// including phase.
pub fn apply_pauli(
    state: &mut dyn Backend<C64>,
    p: PauliString,
    reg: &GateRegistry<C64>,
) -> Result<()> {
    let x_gate = reg.resolve("x")?.matrix(&[])?;
    let y_gate = reg.resolve("y")?.matrix(&[])?;
    let z_gate = reg.resolve("z")?.matrix(&[])?;
    let mut rest = p.x | p.z;
    while rest != 0 {
        let q = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        let bit = 1u64 << q;
        match (p.x & bit != 0, p.z & bit != 0) {
            (true, true) => state.apply(&y_gate, &[q])?,
            (true, false) => state.apply(&x_gate, &[q])?,
            (false, true) => state.apply(&z_gate, &[q])?,
            (false, false) => unreachable!(),
        }
    }
    if p.negative {
        let minus = C64::new(-1.0, 0.0);
        state.apply_diagonal(&[minus, minus], &[0])?;
    }
    Ok(())
}
