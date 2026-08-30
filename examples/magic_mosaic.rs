//! The residual as a mosaic: what collapses, what does not, and why the
//! branching election has to be measured rather than chosen.
//!
//! Run with `cargo run --release --example magic_mosaic`.

use quantsim::pathsum::{PathSum, Pivot};
use std::time::Instant;

const BUDGET: u64 = 20_000_000;

fn rule(t: &str) {
    println!("\n══ {t} ══\n");
}

fn ht_chain(k: usize) -> PathSum {
    let mut p = PathSum::new(1);
    for _ in 0..k {
        p.h(0).unwrap();
        p.t(0).unwrap();
    }
    p.h(0).unwrap();
    p.reduce();
    p
}

fn chain_family(n: usize, layers: usize) -> PathSum {
    let mut p = PathSum::new(n);
    for _ in 0..layers {
        for q in 0..n {
            p.h(q).unwrap();
        }
        for q in 0..n.saturating_sub(1) {
            p.cz(q, q + 1).unwrap();
        }
        for q in 0..n {
            p.t(q).unwrap();
        }
    }
    p.reduce();
    p
}

fn grid_family(side: usize, layers: usize) -> PathSum {
    let n = side * side;
    let mut p = PathSum::new(n);
    for _ in 0..layers {
        for q in 0..n {
            p.h(q).unwrap();
        }
        for r in 0..side {
            for c in 0..side {
                let q = r * side + c;
                if c + 1 < side {
                    p.cz(q, q + 1).unwrap();
                }
                if r + 1 < side {
                    p.cz(q, q + side).unwrap();
                }
            }
        }
        for q in 0..n {
            p.t(q).unwrap();
        }
    }
    p.reduce();
    p
}

fn nodes(p: &PathSum, pv: Pivot) -> String {
    match p.amplitude_merged_with(0, BUDGET, pv) {
        Ok((_, s)) => format!("{}", s.nodes),
        Err(_) => "wall".into(),
    }
}

fn main() -> quantsim::Result<()> {
    rule("1. the Clifford fragment never reaches the solver");
    let mut cl = PathSum::new(12);
    cl.h(0)?;
    for q in 0..11 {
        cl.cnot(q, q + 1)?;
    }
    for q in 0..12 {
        cl.s(q)?;
    }
    cl.reduce();
    let (_, st) = cl.amplitude_merged(0, BUDGET)?;
    println!(
        "  12-qubit GHZ + S layer: h* = {}, {} node, {} leaf, {} branched forms",
        cl.internal_vars(),
        st.nodes,
        st.leaves,
        st.distinct_forms
    );
    println!(
        "  The rewrite rules are complete on Clifford, so there is no residual to\n  \
         mosaic. Everything below is what is left when they are not enough."
    );

    rule("2. thin magic: h* = k, and the work is logarithmic in it");
    println!("  (HT)^k H      h*    enumerate    nodes   forms    hits    time");
    for k in [8usize, 16, 32, 64, 128] {
        let p = ht_chain(k);
        let t0 = Instant::now();
        let (_, s) = p.amplitude_merged(0, BUDGET)?;
        println!(
            "  k = {k:<9} {:>3}    2^{:<9} {:>6}  {:>6}  {:>6}  {:>7.1?}",
            p.internal_vars(),
            p.internal_vars(),
            s.nodes,
            s.distinct_forms,
            s.memo_hits,
            t0.elapsed()
        );
    }
    println!(
        "\n  Cutting the residual path in the middle leaves two halves that factor —\n  \
         and that are then the *same polynomial up to renaming*, so the second is a\n  \
         memo hit. Nodes go 19, 29, 39, 51, 63 as k doubles — 2^128 is 63 nodes."
    );

    rule("3. width is innocent; the coupling geometry is not");
    println!("  same qubit count, T-count and depth — only the CZ pattern differs\n");
    println!("  side  qubits    t   h*      1D nodes   2D nodes   ratio");
    for side in 2..=6usize {
        let n = side * side;
        let line = chain_family(n, 2);
        let grid = grid_family(side, 2);
        let ln = line.amplitude_merged(0, BUDGET)?.1.nodes;
        let gn = grid.amplitude_merged(0, BUDGET)?.1.nodes;
        println!(
            "  {side:>4}  {n:>6}  {:>3}  {:>3}      {ln:>8}   {gn:>8}   {:>5.1}x",
            n * 2,
            line.internal_vars(),
            gn as f64 / ln as f64
        );
    }
    println!(
        "\n  The 1D twin stays linear out to 36 qubits and t = 72. The obstruction is\n  \
         separator growth in the coupling graph, not the size of the register."
    );

    rule("4. the square family: the growth law, not the constant");
    println!("  n qubits, n layers, t = n², treewidth ~ n = √t\n");
    println!("    n     t    h*   enumerate       nodes     forms   log2(nodes)/n      time");
    for n in 3..=10usize {
        let p = chain_family(n, n);
        let t0 = Instant::now();
        let (_, s) = p.amplitude_merged(0, BUDGET)?;
        println!(
            "  {n:>3}  {:>4}  {:>4}   2^{:<9}  {:>10}  {:>8}   {:>13.2}   {:>7.1?}",
            n * n,
            p.internal_vars(),
            p.internal_vars(),
            s.nodes,
            s.distinct_forms,
            (s.nodes as f64).log2() / n as f64,
            t0.elapsed()
        );
    }
    println!(
        "\n  log2(nodes)/n settles at ~1.5, so the cost is 2^{{1.5√t}} where enumeration\n  \
         is 2^t. At n = 10 that is 31,513 nodes against 2^90 ≈ 1.2e27 — the growth\n  \
         *law* changed, not just the constant. It is still exponential, in √t."
    );

    rule("5. the election: no single signal wins");
    println!(
        "  {:<24} {:>9} {:>9} {:>9} {:>9}",
        "family", "First", "MaxDeg", "MinRem", "Elected"
    );
    for (label, p) in [
        ("(HT)^32 H  (a chain)".to_string(), ht_chain(32)),
        ("(HT)^64 H  (a chain)".to_string(), ht_chain(64)),
        ("grid 5x5 L=2 (shallow)".to_string(), grid_family(5, 2)),
        ("grid 6x6 L=2 (shallow)".to_string(), grid_family(6, 2)),
        ("square n=7 (deep)".to_string(), chain_family(7, 7)),
        ("square n=8 (deep)".to_string(), chain_family(8, 8)),
    ] {
        println!(
            "  {label:<24} {:>9} {:>9} {:>9} {:>9}",
            nodes(&p, Pivot::First),
            nodes(&p, Pivot::MaxDegree),
            nodes(&p, Pivot::MinRemainder),
            nodes(&p, Pivot::Elected)
        );
    }
    println!(
        "\n  Connectivity is the whole game on a chain and *flat* on a shallow grid —\n  \
         removing any one variable disconnects nothing there, so degree is the only\n  \
         signal left. Each parent policy is 2-45x worse than the other somewhere.\n  \
         Composing them wins in every regime, which is the mosaic register's finding\n  \
         on a different axis: the right lens is a property of the part, not of the\n  \
         solver, so the election belongs per component and has to be measured."
    );

    rule("6. what this is and is not");
    println!(
        "  Established, and reproduced here from a rule set that shares no code with\n  \
         the engine that first measured it: thin and structured magic collapses;\n  \
         width alone is free; the surviving exponential is indexed by the coupling\n  \
         graph's separators, not by the T-count.\n\n  \
         Open, and untouched by any of this: whether the grown-separator regime\n  \
         collapses under any representation. The square family is still exponential\n  \
         here — in √t rather than t, which is a strict reduction and not a removal.\n  \
         Every number above is a measurement, and the budget refuses by name rather\n  \
         than quietly enumerating when the merges run out."
    );
    Ok(())
}
