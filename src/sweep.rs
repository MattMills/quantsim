//! **Sequential sweep**: the circuit contracted one qubit at a time,
//! carrying only the legs that cross between the processed part and the
//! rest.
//!
//! [`crate::cutsim`] splits the circuit at one cut and pays `2^k` for
//! the `k` gates crossing it, holding two halves of `2^{n/2}` each. That
//! prices the *interface* rather than the register, which is the right
//! currency — but it still materializes half the register twice.
//!
//! This goes the whole way. Process qubits left to right. A `CZ` between
//! qubit `k` and `k+1` decomposes as
//!
//! ```text
//!   CZ = Σ_{b ∈ {0,1}} |b⟩⟨b|_k ⊗ Z^b_{k+1}
//! ```
//!
//! so from qubit `k`'s side it **emits** a leg carrying `b`, and from
//! qubit `k+1`'s side it **consumes** that leg as a `Z^b`. A leg is
//! open only between the two ends that use it, and the moment the second
//! end applies its `Z^b` the leg is summed out — `b` appears nowhere
//! else in the network.
//!
//! ## What that makes the peak
//!
//! While qubit `k` is being processed the live object is
//!
//! ```text
//!   (incoming legs not yet consumed) × (outgoing legs already emitted) × (qubit k: 2)
//! ```
//!
//! On a brickwork the two alternate — qubit `k` couples left, then
//! right, then left — so consumed and produced advance together and the
//! live width stays near the bond's own leg count. For a brickwork on a
//! line that count is `depth/2`, because a bond is active every *other*
//! layer. The whole state is never held: the "block" is one qubit's
//! world-line.
//!
//! ```text
//!   memory ≈ 2^{legs per bond + 1}      time ≈ n · depth · 2^{legs per bond}
//! ```
//!
//! and neither depends on the register's width except linearly. Nothing
//! about that is specific to `T` gates — every single-qubit gate acts
//! inside one world-line, so the `T`-count does not appear at all.
//!
//! ## Scope, stated plainly
//!
//! Two-qubit gates must be `CZ` between **adjacent** qubits: the sweep
//! is what a linear-nearest-neighbour layout buys, and a long-range gate
//! would hold its leg open across every qubit in between, which is
//! exactly the cost this avoids. Both are refused by name rather than
//! silently repriced. Single-qubit gates are unrestricted.

use crate::backend::Backend;
use crate::circuit::{Circuit, Op};
use crate::error::{Error, Result};
use crate::registry::GateRegistry;
use crate::scalar::C64;

/// What the sweep will cost on a circuit, from its structure alone.
#[derive(Debug, Clone)]
pub struct SweepPlan {
    /// Register width.
    pub qubits: usize,
    /// `CZ` gates on each bond `(k, k+1)`, in qubit order.
    pub legs_per_bond: Vec<usize>,
    /// Widest live leg set the sweep will hold, measured by walking the
    /// schedule — not estimated.
    pub peak_live_legs: usize,
    /// Single-qubit gates, which cost the sweep nothing structural.
    pub single_qubit: usize,
}

impl SweepPlan {
    /// `2^{peak+1}` amplitudes — the sweep's memory in units of one
    /// scalar.
    pub fn peak_amplitudes(&self) -> u128 {
        1u128 << (self.peak_live_legs + 1).min(127)
    }
}

/// One event on a qubit's world-line, in circuit order.
#[derive(Debug, Clone)]
enum Event {
    /// A single-qubit gate, by its position in the circuit.
    Local(usize),
    /// Consume the leg emitted by the qubit below: apply `Z^b`, then sum
    /// `b` out. Carries the leg's identity.
    Consume(usize),
    /// Emit a leg to the qubit above, carrying this qubit's value.
    Emit(usize),
}

/// Per-qubit world-lines, and the plan they imply.
fn schedule(circuit: &Circuit<C64>) -> Result<(Vec<Vec<Event>>, SweepPlan)> {
    let n = circuit.num_qubits();
    let mut lines: Vec<Vec<Event>> = vec![Vec::new(); n];
    let mut legs_per_bond = vec![0usize; n.saturating_sub(1)];
    let mut single_qubit = 0usize;
    let mut leg_id = 0usize;
    for (i, op) in circuit.ops().iter().enumerate() {
        let Op::Named { name, qubits, .. } = op else {
            return Err(Error::InvalidState(
                "sweep: only named registry gates contract".into(),
            ));
        };
        match qubits.len() {
            1 => {
                single_qubit += 1;
                lines[qubits[0]].push(Event::Local(i));
            }
            2 => {
                if name != "cz" {
                    return Err(Error::InvalidState(format!(
                        "sweep: two-qubit gate `{name}` is not a CZ; only a rank-2 \
                         diagonal splits into one leg, and anything else is refused \
                         rather than silently repriced"
                    )));
                }
                let (lo, hi) = (qubits[0].min(qubits[1]), qubits[0].max(qubits[1]));
                if hi != lo + 1 {
                    return Err(Error::InvalidState(format!(
                        "sweep: CZ({lo}, {hi}) is not nearest-neighbour; a long-range \
                         leg stays open across every qubit between its ends, which is \
                         the cost this contraction exists to avoid"
                    )));
                }
                legs_per_bond[lo] += 1;
                lines[lo].push(Event::Emit(leg_id));
                lines[hi].push(Event::Consume(leg_id));
                leg_id += 1;
            }
            k => {
                return Err(Error::InvalidState(format!(
                    "sweep: {k}-qubit gates do not decompose into single legs"
                )))
            }
        }
    }
    // Walk the schedule to measure the peak rather than assume it.
    let mut peak = 0usize;
    let mut live = 0usize;
    for line in &lines {
        // Entering a qubit, its incoming legs are all live.
        let incoming = line
            .iter()
            .filter(|e| matches!(e, Event::Consume(_)))
            .count();
        let mut here = incoming;
        peak = peak.max(here.max(live));
        for e in line {
            match e {
                Event::Consume(_) => here -= 1,
                Event::Emit(_) => here += 1,
                Event::Local(_) => {}
            }
            peak = peak.max(here);
        }
        live = here;
    }
    Ok((
        lines,
        SweepPlan {
            qubits: n,
            legs_per_bond,
            peak_live_legs: peak,
            single_qubit,
        },
    ))
}

/// Plan the sweep without running it: the leg counts and the measured
/// peak, from the circuit's structure alone.
pub fn plan(circuit: &Circuit<C64>) -> Result<SweepPlan> {
    Ok(schedule(circuit)?.1)
}

/// What it costs to cut the register into **independent** volumes of
/// world-lines instead of sweeping it in one pass.
///
/// A volume that does not know what is to its left cannot sum its
/// incoming legs out — they are free arguments of the tensor it
/// produces, open from its first gate to its last. Its outgoing legs
/// are open too, from the moment they are emitted. So an independent
/// volume holds *both* of its surfaces at once, where the sweep holds
/// one.
#[derive(Debug, Clone)]
pub struct VolumePlan {
    /// Register width.
    pub qubits: usize,
    /// World-lines per volume, as asked for.
    pub width: usize,
    /// Volumes as inclusive qubit ranges, in order.
    pub volumes: Vec<(usize, usize)>,
    /// Legs on each volume's left surface — free arguments throughout.
    pub incoming: Vec<usize>,
    /// Legs on each volume's right surface.
    pub outgoing: Vec<usize>,
    /// Widest live leg set inside each volume, measured by walking it.
    pub peak_live_legs: Vec<usize>,
}

impl VolumePlan {
    /// The binding volume's peak.
    pub fn peak(&self) -> usize {
        self.peak_live_legs.iter().copied().max().unwrap_or(0)
    }

    /// `2^{peak+1}` amplitudes — the memory one volume needs.
    pub fn peak_amplitudes(&self) -> u128 {
        1u128 << (self.peak() + 1).min(127)
    }

    /// Volumes that can be contracted at the same time.
    pub fn parallelism(&self) -> usize {
        self.volumes.len()
    }
}

/// Measure [`VolumePlan`] for a given volume width.
///
/// `width == qubits` is the whole register as one volume, which has no
/// surfaces and so reduces to the sweep's own peak.
pub fn plan_volumes(circuit: &Circuit<C64>, width: usize) -> Result<VolumePlan> {
    if width == 0 {
        return Err(Error::InvalidState(
            "sweep: a volume of zero world-lines contracts nothing".into(),
        ));
    }
    let (lines, base) = schedule(circuit)?;
    let n = base.qubits;
    let mut volumes = Vec::new();
    let mut a = 0usize;
    while a < n {
        let b = (a + width - 1).min(n - 1);
        volumes.push((a, b));
        a = b + 1;
    }

    let mut incoming = Vec::new();
    let mut outgoing = Vec::new();
    let mut peaks = Vec::new();
    for &(a, b) in &volumes {
        let in_legs = if a == 0 { 0 } else { base.legs_per_bond[a - 1] };
        let out_legs = if b + 1 >= n { 0 } else { base.legs_per_bond[b] };
        // Incoming legs are open for the whole volume; outgoing ones
        // from the moment they are emitted; internal ones open and
        // close exactly as the sweep makes them.
        let mut internal = 0usize;
        let mut emitted = 0usize;
        let mut peak = in_legs;
        for (q, line) in lines.iter().enumerate().take(b + 1).skip(a) {
            peak = peak.max(in_legs + internal + emitted);
            for e in line {
                match e {
                    Event::Consume(_) => {
                        // A leg consumed by the volume's first world-line
                        // came from outside and stays an argument.
                        if q > a {
                            internal -= 1;
                        }
                    }
                    Event::Emit(_) => {
                        if q == b && b + 1 < n {
                            emitted += 1;
                        } else {
                            internal += 1;
                        }
                    }
                    Event::Local(_) => {}
                }
                peak = peak.max(in_legs + internal + emitted);
            }
        }
        debug_assert_eq!(emitted, out_legs);
        incoming.push(in_legs);
        outgoing.push(out_legs);
        peaks.push(peak);
    }

    Ok(VolumePlan {
        qubits: n,
        width,
        volumes,
        incoming,
        outgoing,
        peak_live_legs: peaks,
    })
}

/// The live tensor: amplitudes indexed by `(legs << 1) | s`, where `s`
/// is the current qubit's own value and `legs` runs over the open leg
/// bits in `order`.
struct Live {
    amps: Vec<C64>,
    /// Leg identity per bit position, low bit first.
    order: Vec<usize>,
}

impl Live {
    /// Sum out the leg at bit position `p`, having already used it.
    fn close(&mut self, p: usize) {
        let stride = 1usize << (p + 1);
        let mut out = vec![C64::new(0.0, 0.0); self.amps.len() / 2];
        for (i, a) in self.amps.iter().enumerate() {
            let legs = i >> 1;
            let s = i & 1;
            let lo = legs & (stride / 2 - 1);
            let hi = legs >> (p + 1);
            out[(((hi << p) | lo) << 1) | s] += *a;
        }
        self.amps = out;
        self.order.remove(p);
    }

    /// Open a leg carrying this qubit's current value: the new top bit
    /// equals `s`, so exactly half the expanded table is populated.
    fn open(&mut self, leg: usize) {
        let old = self.amps.len();
        let mut out = vec![C64::new(0.0, 0.0); old * 2];
        for (i, a) in self.amps.iter().enumerate() {
            let s = i & 1;
            // new index: the fresh bit sits above the existing legs.
            out[(s << (self.order.len() + 1)) | i] = *a;
        }
        self.amps = out;
        self.order.push(leg);
    }
}

/// The open-leg amplitudes standing between one qubit and the next.
///
/// This is the whole of what crosses the bond, and it is *not* a
/// register in the space direction: bit `p` of an index is leg
/// `legs[p]`, and the legs arrive in circuit order, so a surface is a
/// register whose sites are **times**. The sweep walks space carrying a
/// temporal register — two geometric modes, composed.
#[derive(Debug, Clone)]
pub struct Surface {
    /// The bond sits between this qubit and the one above it.
    pub after_qubit: usize,
    /// Leg identities, low bit first: one `CZ` each, in time order.
    pub legs: Vec<usize>,
    /// `2^{legs.len()}` amplitudes.
    pub amps: Vec<C64>,
}

/// `⟨bits|C|0…0⟩`, contracted one qubit at a time.
///
/// Exact. Costs `2^{peak_live_legs + 1}` amplitudes of memory and one
/// pass per world-line event, with the `T` count entering neither.
pub fn amplitude(circuit: &Circuit<C64>, bits: u64) -> Result<C64> {
    run(circuit, bits, |_| {})
}

/// Every bond's [`Surface`], recorded as the sweep crosses it.
///
/// Exponential in the legs on a bond — this is the instrument for
/// asking how much of the surface is actually occupied, not a way to
/// run the sweep.
pub fn surfaces(circuit: &Circuit<C64>, bits: u64) -> Result<Vec<Surface>> {
    let mut out = Vec::new();
    run(circuit, bits, |s| out.push(s))?;
    Ok(out)
}

/// The sweep proper: `observe` sees each bond's surface in turn.
fn run(circuit: &Circuit<C64>, bits: u64, mut observe: impl FnMut(Surface)) -> Result<C64> {
    let (lines, _) = schedule(circuit)?;
    let reg = GateRegistry::<C64>::standard();
    let bound = circuit.bind(&reg)?;
    let gates = bound.gates();

    // Legs carried between one qubit and the next, and the amplitudes
    // over them. Before qubit 0 there are none, so the carry is the
    // empty product: a single scalar 1.
    let mut carry: Vec<C64> = vec![C64::new(1.0, 0.0)];
    let mut carry_order: Vec<usize> = Vec::new();

    for (q, line) in lines.iter().enumerate() {
        crate::guard::checkpoint()?;
        // Qubit q starts in |0⟩, tensored onto the incoming legs.
        let mut live = Live {
            amps: {
                let mut v = vec![C64::new(0.0, 0.0); carry.len() * 2];
                for (i, a) in carry.iter().enumerate() {
                    v[i << 1] = *a;
                }
                v
            },
            order: carry_order.clone(),
        };

        for e in line {
            match *e {
                Event::Local(op_index) => {
                    // Apply the 2×2 to the `s` index of every leg block.
                    // `bind` emits one BoundGate per op in order, so
                    // the circuit index addresses the bound gate.
                    let m = gates.get(op_index).ok_or_else(|| {
                        Error::InvalidState("sweep: gate index out of range".into())
                    })?;
                    let mat = m.matrix()?;
                    for blk in 0..(live.amps.len() / 2) {
                        let (a0, a1) = (live.amps[blk * 2], live.amps[blk * 2 + 1]);
                        live.amps[blk * 2] = mat[0] * a0 + mat[1] * a1;
                        live.amps[blk * 2 + 1] = mat[2] * a0 + mat[3] * a1;
                    }
                }
                Event::Consume(leg) => {
                    let p = live
                        .order
                        .iter()
                        .position(|&l| l == leg)
                        .ok_or_else(|| Error::InvalidState("sweep: leg not open".into()))?;
                    // Z^b on this qubit: sign when both the leg bit and
                    // this qubit's value are set.
                    for (i, a) in live.amps.iter_mut().enumerate() {
                        if (i & 1 == 1) && ((i >> 1) >> p) & 1 == 1 {
                            *a = -*a;
                        }
                    }
                    live.close(p);
                }
                Event::Emit(leg) => live.open(leg),
            }
        }

        // Read out this qubit against the target bit and drop `s`.
        let want = ((bits >> q) & 1) as usize;
        let mut next = vec![C64::new(0.0, 0.0); live.amps.len() / 2];
        for (i, a) in live.amps.iter().enumerate() {
            if i & 1 == want {
                next[i >> 1] = *a;
            }
        }
        carry = next;
        carry_order = live.order;
        if q + 1 < lines.len() {
            observe(Surface {
                after_qubit: q,
                legs: carry_order.clone(),
                amps: carry.clone(),
            });
        }
    }

    if !carry_order.is_empty() {
        return Err(Error::InvalidState(
            "sweep: legs left open at the end of the register".into(),
        ));
    }
    Ok(carry[0])
}

/// The gate matrix behind a bound single-qubit gate, as `[m00, m01, m10, m11]`.
trait SingleQubitMatrix {
    fn matrix(&self) -> Result<[C64; 4]>;
}

impl SingleQubitMatrix for crate::circuit::BoundGate<C64> {
    fn matrix(&self) -> Result<[C64; 4]> {
        match &self.kernel {
            crate::circuit::GateKernel::Matrix(m) => {
                let d = m.data();
                if d.len() != 4 {
                    return Err(Error::InvalidState(
                        "sweep: expected a single-qubit matrix".into(),
                    ));
                }
                Ok([d[0], d[1], d[2], d[3]])
            }
            crate::circuit::GateKernel::Diagonal(d) => {
                if d.len() != 2 {
                    return Err(Error::InvalidState(
                        "sweep: expected a single-qubit diagonal".into(),
                    ));
                }
                Ok([d[0], C64::new(0.0, 0.0), C64::new(0.0, 0.0), d[1]])
            }
        }
    }
}

/// Every amplitude, for checking the sweep against a dense reference.
/// Exponential by construction — the verification path, not the point.
pub fn to_dense(circuit: &Circuit<C64>) -> Result<Vec<C64>> {
    let n = circuit.num_qubits();
    (0..(1u64 << n)).map(|x| amplitude(circuit, x)).collect()
}

/// Deviation of the sweep from a dense run of the same circuit.
pub fn max_deviation_vs_dense(circuit: &Circuit<C64>) -> Result<f64> {
    let reg = GateRegistry::<C64>::standard();
    let mut dense = crate::backend::DenseState::<C64>::new(circuit.num_qubits())?;
    circuit.bind(&reg)?.run(&mut dense)?;
    let mut worst = 0.0f64;
    for x in 0..(1u64 << circuit.num_qubits()) {
        worst = worst.max((amplitude(circuit, x)? - dense.amplitude(x)).norm());
    }
    Ok(worst)
}
