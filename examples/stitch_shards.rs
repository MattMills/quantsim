//! Sharding a register by its own symplectic form: the radical carves,
//! the hyperbolic pairs partition, and no node holds an exponential.
//!
//! Run with `cargo run --release --example stitch_shards`.

use quantsim::backend::{DenseState, PauliString};
use quantsim::prelude::*;
use quantsim::retro::{Code, SurfaceCode, ToricCode};
use quantsim::stitch::*;

fn rule(t: &str) {
    println!("\n══ {t} ══\n");
}

fn full_span(gens: &[PauliString], logicals: &[PauliString], n: usize) -> Volume {
    let all: Vec<PauliString> = gens.iter().chain(logicals).copied().collect();
    Volume::span(n, &all).unwrap()
}

fn toric_all(l: usize) -> (ToricCode, Vec<PauliString>, Vec<PauliString>) {
    let c = ToricCode::new(l, 0).unwrap();
    let g = c.generators();
    let mut lg = Vec::new();
    for i in 0..2 {
        lg.push(c.logical_x(i));
        lg.push(c.logical_z(i));
    }
    (c, g, lg)
}

fn main() -> quantsim::Result<()> {
    let reg = GateRegistry::<C64>::standard();

    rule("1. the normal form reads the code off its own generators");
    println!("  code          n    rank   radical r   pairs h   region 2^(n-r)   slices 2^h");
    for (label, n, g, lg) in [
        ("toric L=2", 8usize, toric_all(2).1, toric_all(2).2),
        ("toric L=3", 18, toric_all(3).1, toric_all(3).2),
        (
            "surface d=3",
            9,
            SurfaceCode::new(3, 0)?.generators(),
            vec![
                SurfaceCode::new(3, 0)?.logical_x(),
                SurfaceCode::new(3, 0)?.logical_z(),
            ],
        ),
        (
            "surface d=5",
            25,
            SurfaceCode::new(5, 0)?.generators(),
            vec![
                SurfaceCode::new(5, 0)?.logical_x(),
                SurfaceCode::new(5, 0)?.logical_z(),
            ],
        ),
    ] {
        let st = orthogonalize(&full_span(&g, &lg, n));
        println!(
            "  {label:<12} {n:>2}   {:>4}   {:>9}   {:>7}   {:>13}   {:>10}",
            st.rank(),
            st.radical_rank(),
            st.witt(),
            st.region_dimension(),
            st.slices()
        );
    }
    println!(
        "\n  Nothing told it r or h. The radical IS the stabilizer group and the\n  \
         hyperbolic pairs ARE the logical qubits — symplectic Gram-Schmidt found\n  \
         both, in O(rank^2 . n), with no 2^n anywhere."
    );

    rule("2. the pairs cannot be held in one place, which is why they partition");
    let (code, g, lg) = toric_all(2);
    let st = orthogonalize(&full_span(&g, &lg, 8));
    for (i, (e, f)) in st.pairs().iter().enumerate() {
        println!(
            "  pair {i}: e and f anticommute ({}) — no abelian subgroup holds both",
            !e.commutes_with(*f)
        );
    }
    let sels = st.selections()?;
    println!(
        "\n  so each pair forces a binary choice: {} selections",
        sels.len()
    );
    for (i, s) in sels.iter().enumerate() {
        println!(
            "    slice {i}: rank {} (= r + h = n), maximal isotropic {}, carves dimension {}",
            s.rank(),
            s.is_maximal_isotropic(),
            s.carves()
        );
    }
    let mut worst_join = true;
    for (i, a) in sels.iter().enumerate() {
        for b in sels.iter().skip(i + 1) {
            worst_join &= !a.join(b)?.is_isotropic();
        }
    }
    println!(
        "\n  every pair of slices, joined, stops commuting: {worst_join}\n  \
         — so nothing is fixed by two of them, the regions meet in zero,\n  \
         and {} x {} = {} exactly ({}).",
        st.slices(),
        st.slice_dimension(),
        st.slices() * st.slice_dimension(),
        if st.closes() {
            "closes"
        } else {
            "DOES NOT CLOSE"
        }
    );

    rule("3. the slices, on amplitudes");
    let mut states: Vec<Vec<C64>> = Vec::new();
    for s in &sels {
        let mut plus = [false; 2];
        for (i, p) in plus.iter_mut().enumerate() {
            *p = s.contains(code.logical_x(i));
        }
        let mut dev = DenseState::<C64>::new(8)?;
        code.encoder(8, plus).bind(&reg)?.run(&mut dev)?;
        states.push((0..256u64).map(|i| dev.amplitude(i)).collect());
        println!(
            "  slice {}: basis ({}, {}), tableau {} rows x 16 bits = {} B, no amplitudes",
            states.len() - 1,
            if plus[0] { "X̄₀" } else { "Z̄₀" },
            if plus[1] { "X̄₁" } else { "Z̄₁" },
            s.rank(),
            s.rank() * 2
        );
    }
    let ip =
        |a: &Vec<C64>, b: &Vec<C64>| -> C64 { a.iter().zip(b).map(|(x, y)| x.conj() * y).sum() };
    println!("\n  Gram matrix (they are a frame, not an orthogonal basis):");
    for a in &states {
        let row: Vec<String> = states
            .iter()
            .map(|b| format!("{:>6.3}", ip(a, b).re))
            .collect();
        println!("    [{}]", row.join(" "));
    }
    println!(
        "\n  4 lines, pairwise non-orthogonal, linearly independent, spanning the\n  \
         4-dimensional code space: 2^h . 2^(n-r-h) = 2^(n-r) on amplitudes.\n  \
         tests/stitch.rs projects a random vector into the code space and\n  \
         reassembles it from the four slices to < 1e-9."
    );

    rule("4. where the exponential went");
    println!("     n     r     h |       nodes | per node |      network |     monolithic |  per-node ratio");
    for (n, r, h) in [
        (18usize, 16usize, 2usize),
        (30, 20, 5),
        (30, 20, 10),
        (40, 20, 20),
    ] {
        let p = ShardPlan::new(&Stitch::from_parts(n, r, h)?);
        println!(
            "  {n:>4}  {r:>4}  {h:>4} | {:>11} | {:>6} B | {:>12} | {:>14} | {:>13.3e}",
            p.nodes(),
            p.per_node_bytes(),
            p.network_bytes(),
            p.monolithic_bytes(),
            p.per_node_ratio()
        );
    }
    println!(
        "\n  Per node: polynomial in n, and it does not mention the node count —\n  \
         1024 nodes and 32 nodes on the same register differ by 40 bytes each.\n  \
         Network total: 2^h . O(n^2), exponential in the LOGICAL count, not in\n  \
         the register width. The code already pulled the exponent from n to\n  \
         k = n - r; the shard plan spends it as machines instead of as memory.\n\n  \
         The price, stated: each node carries a whole tableau to hold one\n  \
         coefficient, so the network holds O(n^2) bytes per coefficient where a\n  \
         bare 2^k amplitude vector holds 16. What that buys is a per-node object\n  \
         that is polynomial, closed under Clifford evolution with no\n  \
         communication, and independent of every other node."
    );

    rule("5. h is hidden in the dimension");
    let a = Stitch::from_parts(20, 10, 3)?;
    let b = Stitch::from_parts(20, 10, 7)?;
    println!(
        "  two stitches, region dimension {} both — slices {} and {}.",
        a.region_dimension(),
        a.slices(),
        b.slices()
    );
    println!(
        "  Measuring the object's size says nothing about how many pieces it is\n  \
         in: hidden_bits = {} vs {}.",
        a.hidden_bits(),
        b.hidden_bits()
    );
    Ok(())
}
