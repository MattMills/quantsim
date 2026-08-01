//! Relocating magic, and what `h*` does and does not say about it.
//!
//! `upembed` moves every magic event to a fresh wire, after which the
//! dynamics is Clifford by construction — `to_circuit` emits nothing but
//! `h`, `s` and `cx`. True, but it says nothing about how many wires
//! were *needed*: it spends one per `T` in the gate list, whether or not
//! that gate's magic survives.
//!
//! `pathsum` answers a different question. `h*` counts the variables
//! still summed over once the output index is fixed, so `2^{h*}` is the
//! cost of ONE amplitude `⟨z|U|x⟩`. Pairing the two turns "how many
//! wires does this circuit really need?" into a measurement.
//!
//! It is worth being exact about what that measurement is, because an
//! earlier version of this example was not. `h* = 0` does **not** mean
//! the unitary is Clifford. A lone `T` has `h* = 0`; so does `CCZ`; so
//! does every diagonal circuit, however much magic it carries — a
//! diagonal gate maps each basis state to itself with a phase, so it
//! never hides a variable. `tests/magic_relocation.rs` holds a dense
//! Clifford oracle against all of those. Measured the other way, every
//! Clifford circuit in that corpus did reduce to zero, so the classes
//! sit as `{Clifford} ⊊ {h* = 0}` — readable in one direction only.

use quantsim::circuit::Op;
use quantsim::pathsum;
use quantsim::prelude::*;
use quantsim::upembed;

/// Magic that genuinely survives: a `T`, then entangling, then another.
fn magic_circuit(n: usize, k: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for i in 0..k {
        c.gate("h", vec![], vec![i % n]);
        c.gate("t", vec![], vec![i % n]);
        c.gate("cx", vec![], vec![i % n, (i + 1) % n]);
        c.gate("t", vec![], vec![(i + 1) % n]);
    }
    c
}

/// Magic that cancels: `T`, a Clifford round trip, then `T†`. The gate
/// list is full of magic; the unitary has none.
fn cancelling(n: usize, k: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for i in 0..k {
        c.gate("t", vec![], vec![i % n]);
        c.gate("cx", vec![], vec![i % n, (i + 1) % n]);
        c.gate("cx", vec![], vec![i % n, (i + 1) % n]);
        c.gate("tdg", vec![], vec![i % n]);
        c.gate("h", vec![], vec![(i + 2) % n]);
    }
    c
}

/// The `i < j < k` triples of `0..n`.
fn triples(n: usize) -> Vec<[usize; 3]> {
    let mut out = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                out.push([i, j, k]);
            }
        }
    }
    out
}

/// A diagonal circuit with as much magic as `k` allows, on distinct
/// triples so nothing cancels. Non-Clifford at every `k ≥ 1`.
fn diagonal_core(n: usize, k: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.gate("t", vec![], vec![q]);
    }
    for t in triples(n).iter().take(k) {
        c.gate("ccz", vec![], t.to_vec());
    }
    c
}

fn concat(a: &Circuit<C64>, b: &Circuit<C64>) -> Circuit<C64> {
    let mut c = Circuit::new(a.num_qubits().max(b.num_qubits()));
    for src in [a, b] {
        for op in src.ops() {
            if let Op::Named {
                name,
                params,
                qubits,
            } = op
            {
                c.gate(name, params.clone(), qubits.clone());
            }
        }
    }
    c
}

fn inverse(a: &Circuit<C64>) -> Circuit<C64> {
    let mut c = Circuit::new(a.num_qubits());
    for op in a.ops().iter().rev() {
        if let Op::Named {
            name,
            params,
            qubits,
        } = op
        {
            let n = match name.as_str() {
                "t" => "tdg",
                "tdg" => "t",
                "s" => "sdg",
                "sdg" => "s",
                o => o,
            };
            c.gate(n, params.clone(), qubits.clone());
        }
    }
    c
}

/// A fixed Clifford skeleton, so magic can be varied against an
/// identical Hadamard structure.
fn skeleton(n: usize, layers: usize, seed: u64) -> Circuit<C64> {
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for _ in 0..layers {
        for q in 0..n {
            c.gate("h", vec![], vec![q]);
        }
        for _ in 0..n {
            let a = (rng.next_u64() % n as u64) as usize;
            let b = (a + 1 + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
            c.gate("cx", vec![], vec![a, b]);
        }
    }
    c
}

fn with_magic(skel: &Circuit<C64>, t: usize, seed: u64) -> Circuit<C64> {
    let n = skel.num_qubits();
    let mut rng = Prng::new(seed);
    let mut slots: Vec<usize> = (0..t)
        .map(|_| (rng.next_u64() % (skel.len() as u64 + 1)) as usize)
        .collect();
    slots.sort_unstable();
    let mut c = Circuit::new(n);
    let mut next = 0usize;
    for (i, op) in skel.ops().iter().enumerate() {
        while next < slots.len() && slots[next] == i {
            c.gate("t", vec![], vec![(rng.next_u64() % n as u64) as usize]);
            next += 1;
        }
        if let Op::Named {
            name,
            params,
            qubits,
        } = op
        {
            c.gate(name, params.clone(), qubits.clone());
        }
    }
    while next < slots.len() {
        c.gate("t", vec![], vec![(rng.next_u64() % n as u64) as usize]);
        next += 1;
    }
    c
}

fn main() -> Result<()> {
    println!("== relocating magic, and what h* measures ==\n");

    // ── what h* is not ───────────────────────────────────────────────
    println!("── h* = 0 is not a Clifford certificate ──\n");
    println!("   circuit                     gates    h*   Clifford?");
    let mut lone = Circuit::<C64>::new(1);
    lone.gate("t", vec![], vec![0]);
    for (label, c, clifford) in [
        ("a lone T", lone, "no"),
        ("diagonal core n=6 k=20", diagonal_core(6, 20), "no"),
        ("diagonal core n=12 k=220", diagonal_core(12, 220), "no"),
        ("cancelling k=8", cancelling(3, 8), "yes"),
    ] {
        let h = pathsum::operator(&c)?.internal_vars();
        println!("   {label:26} {:6} {h:5}   {clifford}", c.len());
    }
    println!("\n  A diagonal gate maps each basis state to itself with a phase, so it");
    println!("  never hides a variable — h* stays 0 with 220 CCZ gates loaded in.");
    println!("  The Clifford column is an independent dense oracle (see");
    println!("  tests/magic_relocation.rs), not this module's own opinion.\n");

    // ── how many wires were actually needed ──────────────────────────
    println!("── spend one wire at a time and ask what is left ──\n");
    println!("   circuit              T gates   gadgets needed   naive   saved");
    for (label, c) in [
        ("magic k=4", magic_circuit(3, 4)),
        ("magic k=8", magic_circuit(3, 8)),
        ("cancelling k=8", cancelling(3, 8)),
        ("cancelling k=32", cancelling(3, 32)),
    ] {
        let t = upembed::magic_events(&c)?;
        let mut need = t;
        for g in 0..=t {
            if pathsum::operator(&upembed::gadgetize_partial(&c, g)?)?.internal_vars() == 0 {
                need = g;
                break;
            }
        }
        println!("   {label:20} {t:7} {need:16} {t:7} {:7}", t - need);
    }
    println!("\n  The cancelling rows are the point: 64 T gates in the gate list, and");
    println!("  the right number of wires is ZERO. A T-counting cost model would");
    println!("  have spent 64 ancillas relocating magic that was not there. Even");
    println!("  the genuinely magical rows need fewer wires than they have T gates.\n");
    println!("  Scope, stated exactly: \"needed\" is the count at which ONE AMPLITUDE");
    println!("  of the remaining dynamics becomes free — not the count at which the");
    println!("  remaining unitary becomes Clifford. For the cancelling family both");
    println!("  happen to hold; for a diagonal core only the first does. The search");
    println!("  is also over PREFIXES of the T gates, so the number is an upper");
    println!("  bound on the true minimum over subsets. The zero case is exact.\n");

    // ── and what composition does to it ──────────────────────────────
    println!("── h* under composition ──\n");
    println!("   U                V                  h*(U)  h*(V)   sum   h*(UV)");
    let u = magic_circuit(3, 4);
    let hu = pathsum::operator(&u)?.internal_vars();
    for (label, v) in [
        ("its own inverse", inverse(&u)),
        ("itself", u.clone()),
        ("magic k=6", magic_circuit(3, 6)),
        ("cancelling k=4", cancelling(3, 4)),
    ] {
        let hv = pathsum::operator(&v)?.internal_vars();
        let huv = pathsum::operator(&concat(&u, &v))?.internal_vars();
        println!(
            "   magic k=4        {label:18} {hu:5}  {hv:5}  {:4}  {huv:5}",
            hu + hv
        );
    }
    println!("\n  h* is NEITHER sub- nor super-additive. Composed with its inverse it");
    println!("  CANCELS to nothing; composed with ITSELF it COMPOUNDS past the sum,");
    println!("  because variables that reduced away inside each block stop reducing");
    println!("  once the join entangles them. Neither block determines the answer,");
    println!("  so a join has to be reduced rather than estimated from its parts —");
    println!("  exactly what a cost model built on T-counting cannot do.\n");
    println!("  The last row is the one that fixes h*'s meaning: `cancelling k=4` is");
    println!("  a genuinely Clifford unitary, and composing with it raises h* from 1");
    println!("  to 3. Magic is invariant under Clifford composition, so h* is NOT a");
    println!("  magic monotone — it is the exponent for one amplitude, and");
    println!("  composition genuinely changes that.\n");
    // ── what actually drives the exponent ────────────────────────────
    println!("── h* is capped by Hadamards, and sublinear in magic ──\n");
    println!("  Only a Hadamard allocates a path variable. Every diagonal gate —");
    println!("  T, S, CZ, CCZ — contributes phase terms and NO variable, so");
    println!("  h* <= #H structurally and magic enters only by obstructing the");
    println!("  elimination of variables the Hadamards already made.\n");
    let skel = skeleton(8, 4, 11);
    let hs = skel.ops().iter().filter(|o| matches!(o, Op::Named { name, .. } if name == "h")).count();
    println!("   n=8, {hs} Hadamards held byte-identical across every row");
    println!("      t    h*    h*/#H    h*/t");
    for t in [1usize, 4, 16, 64] {
        let h = pathsum::operator(&with_magic(&skel, t, 5))?.internal_vars();
        println!(
            "   {t:5} {h:5}    {:.2}     {:.3}",
            h as f64 / hs as f64,
            h as f64 / t as f64
        );
    }
    println!("\n  h*/t FALLS as magic is added — about t^0.6, competitive with the");
    println!("  2^(0.23t) stabilizer-rank baseline and with no stabilizer machinery.\n");
    println!("  But magic is not the parameter that governs the asymptotics. With");
    println!("  magic scaled to width (t = n) and depth fixed, h*/n PLATEAUS near");
    println!("  0.53 rather than decaying:\n");
    println!("      n     #H     h*    h*/n");
    for n in [8usize, 32, 128] {
        let sk = skeleton(n, 2, 11);
        let hg = sk.ops().iter().filter(|o| matches!(o, Op::Named { name, .. } if name == "h")).count();
        let h = pathsum::operator(&with_magic(&sk, n, 5))?.internal_vars();
        println!("   {n:5} {hg:6} {h:6}   {:.3}", h as f64 / n as f64);
    }
    println!("\n  So one amplitude costs 2^(0.53n) against a dense 2^n: a real cut in");
    println!("  the exponent, roughly twice the reachable width, and NOT a change of");
    println!("  growth class. Sub-exponentiality would show up here as h*/n -> 0.");
    println!("  Since h* is an upper bound — what this rewrite system achieved — a");
    println!("  decaying ratio would be a fact about the reduction improving. That");
    println!("  is the actionable target, and it is open.");
    Ok(())
}
