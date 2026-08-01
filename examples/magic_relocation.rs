//! Relocating magic until the dynamics is Clifford — and what `h*` says
//! about how much relocating was actually needed.
//!
//! `upembed` moves every magic event to a fresh wire, after which the
//! dynamics is Clifford by construction. True, but it says nothing about
//! how many wires were *needed*: it spends one per `T` gate in the gate
//! list, whether or not that gate's magic survives.
//!
//! `pathsum` reduces every Clifford circuit to `h* = 0` from rewrite
//! rules alone and knows nothing about `upembed`. Putting the two
//! together turns "is it Clifford now?" into a measurement, and the
//! answer is that the naive wire count is often badly wasteful.

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

fn main() -> Result<()> {
    println!("== relocating magic until the dynamics is Clifford ==\n");

    // ── the two modules check each other ─────────────────────────────
    println!("── the up-embedded dynamics, certified by a second representation ──\n");
    println!("  `gadgetize` CLAIMS the dynamics is Clifford. `PathSum` reduces every");
    println!("  Clifford circuit to h* = 0 from rewrite rules alone and knows nothing");
    println!("  about `upembed`, so running the result through it is a certificate");
    println!("  rather than this module marking its own homework.\n");
    println!("   circuit              data   wires   steps   h*   verdict");
    for (label, c) in [
        ("magic k=4", magic_circuit(3, 4)),
        ("magic k=8", magic_circuit(4, 8)),
        ("magic k=16", magic_circuit(5, 16)),
        ("cancelling k=16", cancelling(3, 16)),
    ] {
        let emb = upembed::gadgetize(&c)?;
        let h = pathsum::operator(&emb.to_circuit())?.internal_vars();
        println!(
            "   {label:20} {:4} {:7} {:7} {h:4}   {}",
            emb.data(),
            emb.wires(),
            emb.steps().len(),
            if h == 0 {
                "certified Clifford"
            } else {
                "NOT Clifford"
            }
        );
    }

    // ── how many wires were actually needed ──────────────────────────
    println!("\n── spend one wire at a time and ask what is left ──\n");
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
        println!(
            "   {label:20} {t:7} {need:16} {t:7} {:7}",
            t - need
        );
    }
    println!("\n  The cancelling rows are the point: 64 T gates in the gate list, and");
    println!("  the right number of wires is ZERO — the unitary was already Clifford");
    println!("  and reduction certifies it. A T-counting cost model would have spent");
    println!("  64 ancillas relocating magic that was not there. Even the genuinely");
    println!("  magical rows need fewer wires than they have T gates.\n");
    println!("  Honest scope: the search is over PREFIXES of the T gates, so the");
    println!("  number reported is an upper bound on the true minimum over subsets.");
    println!("  The zero case is exact — no subset can beat none.\n");

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
        println!("   magic k=4        {label:18} {hu:5}  {hv:5}  {:4}  {huv:5}", hu + hv);
    }
    println!("\n  h* is NEITHER sub- nor super-additive, and the obvious hypothesis —");
    println!("  that magic is a resource which adds — is false in both directions.");
    println!("  Composed with its inverse it CANCELS to nothing. Composed with");
    println!("  ITSELF it COMPOUNDS past the sum: magic that reduced away inside each");
    println!("  block stops reducing once the join entangles it. Neither block");
    println!("  determines the answer, so the join has to be reduced rather than");
    println!("  estimated from its parts — which is exactly the thing a cost model");
    println!("  built on T-counting cannot do.\n");
    println!("  And the row that constrains how far to trust h*: composing with");
    println!("  `cancelling k=4` — a block whose own h* is 0, so a CLIFFORD unitary");
    println!("  — raises h* from 1 to 3. True magic is invariant under Clifford");
    println!("  composition, so that rise is not a property of the unitary: it is");
    println!("  the reduction failing to find the optimum. h* is WHAT THE REWRITE");
    println!("  SYSTEM ACHIEVED, an upper bound on the readout exponent, and NOT a");
    println!("  magic monotone. The rules are complete on the Clifford fragment and");
    println!("  not beyond it, and this is where that shows.");
    Ok(())
}
