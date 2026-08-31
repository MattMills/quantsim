//! Canonical residual forms as composable addresses: what closing the
//! address space under composition buys.
//!
//! Run with `cargo run --release --example address_algebra`.

use quantsim::address::{AddressSpace, Node, Scale};
use quantsim::pathsum::PathSum;
use quantsim::{Circuit, Result, C64};

fn rule(t: &str) {
    println!("\n══ {t} ══\n");
}

fn grid(side: usize, layers: usize) -> Circuit<C64> {
    let n = side * side;
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    for _ in 0..layers {
        for y in 0..side {
            for x in 0..side {
                let s = y * side + x;
                if x + 1 < side {
                    c.cz(s, s + 1);
                }
                if y + 1 < side {
                    c.cz(s, s + side);
                }
            }
        }
        for q in 0..n {
            c.t(q);
            c.h(q);
        }
    }
    c
}

fn ghz(n: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    c.h(0);
    for q in 1..n {
        c.cx(q - 1, q);
    }
    c
}

fn main() -> Result<()> {
    for line in [
        "The merge solver already memoizes on a canonical form, but its memo has the",
        "type CanonKey -> C64: an address maps to a VALUE, so nothing can be done with",
        "one except decode it, and it is rebuilt from empty on every query.",
        "",
        "Closing the space under composition gives three constructors --",
        "",
        "   Zero",
        "   Product { scale, parts: [Addr] }     omega^turn . sqrt2^half . prod parts",
        "   Branch  { zero: Addr, one: Addr }    the sum of the two children",
        "",
        "-- which are the solver's own three moves written as algebra instead of as",
        "control flow. Addresses reference addresses; a value is a fold done once.",
    ] {
        println!("{line}");
    }

    rule("1. a Clifford amplitude is one symbol, at every width");
    let mut sp = AddressSpace::new();
    let mut addrs = Vec::new();
    for n in [4usize, 8, 12, 16] {
        let ps = PathSum::from_circuit(&ghz(n))?;
        addrs.push(sp.address(&ps, 0, 10_000)?);
    }
    println!(
        "  GHZ at n = 4, 8, 12, 16  ->  addresses {:?}",
        addrs.iter().map(|a| a.index()).collect::<Vec<_>>()
    );
    println!("  the whole space holds {} address, and it is", sp.len());
    println!("    {:?}", sp.node(addrs[0]));
    for line in [
        "",
        "  h* = 0 means a closed leaf: no parts to sum at all, and the symbol is the",
        "  scale itself. Four circuits at four widths are one node -- equal addresses,",
        "  decided by identity, with no arithmetic performed on either side.",
    ] {
        println!("{line}");
    }

    rule("2. the space outlives the query");
    println!("  A CanonKey -> C64 memo is scoped to one call of amplitude_merged. A");
    println!("  content-addressed space answers the second amplitude out of what the");
    println!("  first one built -- Gosper's hashlife move, on the phase polynomial.\n");
    println!(
        "  {:<16} {:>7} {:>10} {:>10} {:>8} {:>9} {:>9}",
        "circuit", "queries", "shared", "isolated", "reuse", "1st query", "last"
    );
    for (name, c, queries) in [
        ("grid 2x2 L=2", grid(2, 2), 16usize),
        ("grid 3x3 L=1", grid(3, 1), 64),
        ("grid 3x3 L=2", grid(3, 2), 64),
        ("grid 4x4 L=1", grid(4, 1), 64),
    ] {
        let ps = PathSum::from_circuit(&c)?;
        let mut shared = AddressSpace::new();
        let (mut first, mut last) = (0u64, 0u64);
        for b in 0..queries as u64 {
            let before = shared.stats().builds;
            shared.address(&ps, b, 5_000_000)?;
            let cost = shared.stats().builds - before;
            if b == 0 {
                first = cost;
            }
            last = cost;
        }
        let mut isolated = 0u64;
        for b in 0..queries as u64 {
            let mut fresh = AddressSpace::new();
            fresh.address(&ps, b, 5_000_000)?;
            isolated += fresh.stats().builds;
        }
        println!(
            "  {name:<16} {queries:>7} {:>10} {:>10} {:>7.2}x {:>9} {:>9}",
            shared.stats().builds,
            isolated,
            isolated as f64 / shared.stats().builds as f64,
            first,
            last
        );
    }
    for line in [
        "",
        "  `shared` is one space answering every query; `isolated` is a fresh space per",
        "  query, which is what the old memo does. The last column is the point: the",
        "  64th amplitude of a 3x3 grid costs single-digit composition calls because",
        "  almost every form it needs is already addressed.",
    ] {
        println!("{line}");
    }

    rule("3. evaluation is exact, and zero is decided rather than tested");
    let mut c: Circuit<C64> = Circuit::new(2);
    c.h(0);
    c.z(0);
    c.h(0);
    c.cx(0, 1);
    let ps = PathSum::from_circuit(&c)?;
    let mut sp = AddressSpace::new();
    println!("  h z h cx  (which is X on the control, so only |11> survives)\n");
    for b in 0..4u64 {
        let a = sp.address(&ps, b, 10_000)?;
        let zero = sp.is_zero(a)?;
        let (coeffs, k) = sp.value_exact(a)?.parts();
        println!(
            "    |{b:02b}>  exact zero {zero:<5}  ring value (a,b,c,d)/sqrt2^k = {coeffs:?}/sqrt2^{k}"
        );
    }
    for line in [
        "",
        "  The scale is kept as the symbol it always was -- a dyadic turn and a",
        "  half-power of two -- so the whole DAG evaluates in the Clifford+T ring with",
        "  no floating point anywhere, and `is_zero` is a decision.",
    ] {
        println!("{line}");
    }

    rule("4. measured, versus merely not yet observed");
    let s = Scale {
        turn: 1 << 58,
        half: 0,
    };
    for line in [
        "  On cost, the checkable claim: for ONE query the address route performs",
        "  exactly as many compositions as the merge solver performs nodes -- the same",
        "  recursion under the same pivot -- so the per-query cost IS the merge",
        "  solver's, and everything here is reuse between queries. Whether the growth",
        "  law across a circuit family changes is not measured, and not claimed.",
        "",
        "  0 + x = x FIRES. A branch child can reduce to the zero polynomial; the",
        "  shortest witness a 60,000-circuit search found is eleven gates, and it is a",
        "  test. It is rare -- 10 of 60,000 random Clifford+T circuits at 2-4 qubits --",
        "  and an earlier smaller search finding none led this module to assert a",
        "  MECHANISM for why it could not happen. The wider search refuted that. The",
        "  rate is one sample; nothing here bounds it.",
        "",
        "  x . 0 = 0 has NOT been observed firing in any of those 60,000 circuits.",
        "  That is an observation with a sample attached, not a claim that it cannot.",
        "",
        "  The symbol is exact for any dyadic turn; exact EVALUATION is the eighth-turn",
        "  fragment, and anything else is refused by name rather than rounded:",
    ] {
        println!("{line}");
    }
    println!("    turn 2^58 is an eighth turn: {}", s.is_eighth());
    println!("    -> {}", s.to_exact().unwrap_err());
    let a = sp.address(&ps, 0, 10)?;
    assert!(matches!(sp.node(a), Node::Zero));
    Ok(())
}
