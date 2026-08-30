//! Laying a lattice onto a chain: what a space-filling curve costs, and
//! why the obvious expectation was backwards.
//!
//! Run with `cargo run --release --example curve_orders`.

use quantsim::curve::*;
use quantsim::polar::PolarSpace;

fn rule(t: &str) {
    println!("\n══ {t} ══\n");
}

fn main() -> quantsim::Result<()> {
    rule("1. three ways to lay a side x side lattice onto a chain");
    println!(
        "  side  order      continuous   max-dil   mean-dil   adj%   cutwidth   crossings    swaps"
    );
    for side in [4usize, 8, 16, 32, 64] {
        for order in [Order::RowMajor, Order::Snake, Order::Hilbert] {
            let g = GridOrder::new(side, order)?;
            let d = g.dilation();
            println!(
                "  {side:>4}  {:<10} {:>10}   {:>7}   {:>8.2}   {:>4.0}%   {:>8}   {:>9}   {:>6}",
                format!("{order:?}"),
                g.is_continuous(),
                d.max,
                d.mean(),
                d.adjacent_fraction() * 100.0,
                g.cutwidth(),
                g.total_crossings(),
                d.routing_cost()
            );
        }
        println!();
    }

    rule("2. the closed forms, pinned to side 128");
    println!("  ordering    max dilation             cutwidth");
    println!("  RowMajor    side                     side + 1");
    println!("  Snake       2*side - 1               side + 1");
    println!("  Hilbert     (10*4^(k-1) - 1)/3       2*side - 2      (side = 2^k)\n");
    for k in 1..=7u32 {
        let side = 1usize << k;
        let h = GridOrder::new(side, Order::Hilbert)?;
        println!(
            "  side {side:>4}: hilbert max-dil {:>6} = law {:>6}   cutwidth {:>4} = law {:>4}",
            h.dilation().max,
            Order::Hilbert.max_dilation_law(side).unwrap(),
            h.cutwidth(),
            Order::Hilbert.cutwidth_law(side).unwrap_or(0)
        );
    }

    rule("3. the refutation");
    println!("  The expectation this was written to test: a curve that preserves");
    println!("  locality should lay a lattice onto a chain better than reading rows.\n");
    for side in [16usize, 32, 64] {
        let r = GridOrder::new(side, Order::RowMajor)?;
        let h = GridOrder::new(side, Order::Hilbert)?;
        println!(
            "  side {side:>3}:  cutwidth {} vs {}  ({:.2}x)   swaps {} vs {}  ({:.2}x)   worst edge {} vs {}",
            h.cutwidth(), r.cutwidth(), h.cutwidth() as f64 / r.cutwidth() as f64,
            h.dilation().routing_cost(), r.dilation().routing_cost(),
            h.dilation().routing_cost() as f64 / r.dilation().routing_cost() as f64,
            h.dilation().max, r.dilation().max
        );
    }
    println!(
        "\n  It is worse on every measure a chain register cares about, by exact laws:\n  \
         cutwidth exactly 2x, worst dilation Theta(N) against Theta(sqrt N).\n\n  \
         The reason is a DIRECTION error, and the intuition is natural enough to be\n  \
         worth naming. A space-filling curve preserves locality from CURVE TO PLANE:\n  \
         nearby indices are nearby points. That is what makes it right for spatial\n  \
         indexing and cache layout. A chain register needs the OTHER direction --\n  \
         nearby points must get nearby indices -- and the Hilbert curve's inverse has\n  \
         unbounded dilation. Reading the rows is bad at the first and optimal at the\n  \
         second.\n\n  \
         Half the expectation did survive: cutwidth is Theta(side) under EVERY\n  \
         ordering, a property of the grid rather than of the curve, so no relabelling\n  \
         beats it asymptotically. The other half is recorded as refuted."
    );

    rule("4. keeping the recursion instead of flattening it");
    println!("  The Hilbert construction is a ROTOR PER CELL applied recursively --");
    println!("  each quadrant entered under a dihedral symmetry. The rule is:");
    for (i, r) in hilbert_rotors().iter().enumerate() {
        println!(
            "    quadrant {i}: transpose {:<5} quarter-turns {}   {}",
            r.transpose,
            r.quarter_turns,
            if r.is_reflection() {
                "reflection"
            } else {
                "rotation"
            }
        );
    }
    println!("  Two reflections, two rotations. A linear index keeps only the order");
    println!("  the recursion happens to visit cells in, and throws the rest away.\n");
    println!("  side   chain total   tree total    ratio   max separator (chain / tree)");
    for side in [8usize, 16, 32, 64, 128] {
        let b = Bisection::quadrants(side)?;
        let chain = GridOrder::new(side, Order::RowMajor)?;
        println!(
            "  {side:>4}   {:>11}   {:>10}   {:>5.1}x   {:>7} / {}",
            chain.total_crossings(),
            b.total_separator(),
            chain.total_crossings() as f64 / b.total_separator() as f64,
            chain.cutwidth(),
            b.max_separator()
        );
    }
    for line in [
        "",
        "  chain (best ordering) = side^3 - side      = Theta(N^1.5)",
        "  tree                  = 2*side^2 - 2*side  = Theta(N)",
        "  ratio                 = (side + 1) / 2     = Theta(sqrt N)",
        "",
        "  The WORST cut is identical -- the grid's own bound, which no layout beats.",
        "  The TOTAL differs by an order, and the gap grows without limit. A",
        "  representation paying per cut over a hierarchy (mera, bulk) pays the second",
        "  column; one paying over a chain's cuts pays the first.",
        "",
        "  So the curve was never the wrong idea -- FLATTENING it was. The refutation",
        "  in section 3 is real and narrow: it refutes the curve as a LINEAR ORDER,",
        "  which is exactly the use that discards the recursion.",
        "",
        "  One shape worth not guessing: the tree's cost is not at the root. Its",
        "  per-level profile is side * 2^(level/2), doubling every two levels, so the",
        "  total sits at the DEEPEST cuts. Separators shrink per node; node count",
        "  grows faster.",
        "",
        "  Not shown here, and worth separating from what is: whether an OVERLAY of",
        "  several distinct rotor assignments, used together as independent addressing",
        "  bits rather than one at a time, improves on plain recursive bisection. That",
        "  is a stronger claim than anything measured above and it stays open.",
    ] {
        println!("{line}");
    }

    rule("5. where the rotor earns its place");
    println!("  The bisection tree's separators do not depend on the rotor at all --");
    println!("  quadrants are quadrants whichever symmetry you enter them under. So if");
    println!("  the tree is what is useful, what is the rotor FOR?\n");
    println!("  A balanced tree over a chain has the contiguous 2^k blocks as its");
    println!("  subtrees, and a hierarchy's cost at a subtree is that block's BOUNDARY:\n");
    for side in [16usize, 32] {
        println!("  side {side}");
        println!("    block   RowMajor max/mean    Snake max/mean    Hilbert max/mean");
        let r = GridOrder::new(side, Order::RowMajor)?;
        let sn = GridOrder::new(side, Order::Snake)?;
        let h = GridOrder::new(side, Order::Hilbert)?;
        for ((rb, sb), hb) in r
            .block_boundaries()
            .iter()
            .zip(sn.block_boundaries().iter())
            .zip(h.block_boundaries().iter())
        {
            println!(
                "    {:>5}   {:>5} /{:>7.1}      {:>5} /{:>7.1}     {:>5} /{:>7.1}",
                rb.size,
                rb.max,
                rb.mean(),
                sb.max,
                sb.mean(),
                hb.max,
                hb.mean()
            );
        }
        println!();
    }
    for line in [
        "  Hilbert is better at EVERY block size and never worse -- the opposite",
        "  verdict from section 3, on the same three orderings. The reason is the",
        "  mechanism: a row-major block of `side` positions IS one entire row, so its",
        "  boundary is 2*side; the rotor makes the curve's blocks compact regions.",
        "",
        "  So the rotor is exactly what makes a LINEAR index's contiguous blocks",
        "  coincide with the TREE's spatial regions. Both results hold at once:",
        "    - as a chain layout (prefix cuts, dilation, routing): worse, by exact laws",
        "    - as the leaf ordering of a hierarchy (cost per subtree): better at every",
        "      scale",
        "",
        "  Not shown: this is combinatorial, and it is the cost model MeraState",
        "  documents rather than a measurement of that backend. Running the three",
        "  orderings through MeraState at a capped bond on a 4x4 grid did NOT",
        "  reproduce the advantage, and the comparison is confounded -- relabelling",
        "  the qubits also reorders the gate stream, so the truncation schedules",
        "  differ and the layout is not isolated. Recorded as an attempted",
        "  measurement that did not separate them, not as a result either way.",
    ] {
        println!("{line}");
    }

    rule("6. the snake gets both halves");
    for side in [16usize, 32, 64] {
        let r = GridOrder::new(side, Order::RowMajor)?;
        let s = GridOrder::new(side, Order::Snake)?;
        println!(
            "  side {side:>3}:  snake cutwidth {} = row {}   swaps {} = {}   continuous {} vs {}   adjacent {:.0}% vs {:.0}%",
            s.cutwidth(), r.cutwidth(),
            s.dilation().routing_cost(), r.dilation().routing_cost(),
            s.is_continuous(), r.is_continuous(),
            s.dilation().adjacent_fraction() * 100.0, r.dilation().adjacent_fraction() * 100.0
        );
    }
    println!(
        "\n  Continuous like the curve, and exactly as cheap as the rows. The property\n  \
         the Hilbert curve was wanted for comes free from reversing alternate rows."
    );

    rule("7. the census, reached from the geometry");
    println!("   n   generators   stabilizer states   |Sp(2n,2)|              v2   n^2");
    for n in 1..=6usize {
        let w = PolarSpace::new(n)?;
        println!(
            "  {n:>2}   {:>10}   {:>17}   {:>20}   {:>3}   {:>3}",
            w.generators(),
            w.stabilizer_states(),
            w.symplectic_order(),
            w.symplectic_dyadic_valuation(),
            n * n
        );
    }
    println!(
        "\n  6, 60, 1080, 36720 are the stabilizer-state counts, reached here by\n  \
         counting maximal isotropic flats and multiplying by the 2^n sign choices --\n  \
         a sibling program's phase-space census arrives at the same sequence from the\n  \
         other side. And v2|Sp(2n,2)| = n^2 EXACTLY, second differences identically 2:\n  \
         a quadratic law that is, here, just the 2^{{n^2}} factor of the polar space's\n  \
         own symmetry group. Two routes, one sequence."
    );
    Ok(())
}
