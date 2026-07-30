//! E8 as a 2×2×2 cube volume, and the scale tower as a cube of cubes —
//! with interaction directed inward (into a volume's own interior) and
//! outward (to its siblings), measured on live lattice states.
//!
//! Run with `cargo run --release --example e8_cube`.

use quantsim::e8::constellation::E8ConstellationState;
use quantsim::e8::cube;
use quantsim::prelude::*;
use quantsim::recursive::{RecursiveLattice, Shape};

fn main() -> Result<()> {
    // ── 1. Eight coordinates, eight vertices ─────────────────────────
    println!("── the volume: E8's eight ambient coordinates on a 2×2×2 cube");
    println!("  vertices  : 0..7, vertex v is coordinate v (three bits = three axes)");
    println!("  edges (12): {:?}", cube::edges());
    for axis in 0..3 {
        println!("  axis {axis} matching: {:?}", cube::axis_pairs(axis)?);
    }

    // ── 2. The lattice condition as a law on the volume ──────────────
    println!("\n── the lattice condition, read as a parity law on the volume");
    let (on_cube, centred) = cube::sector_census();
    println!(
        "  of the 240 roots: {on_cube} live wholly on the cube (all vertex values \
         integral),"
    );
    println!(
        "                    {centred} on the body-centred copy (all half-integral) — \
         the checkerboard packing"
    );
    let bad = cube::sector(&[1, 2, 0, 0, 0, 0, 0, 0]).is_none();
    println!("  a volume with mixed vertex parities is in neither sector: refused = {bad}");
    let all_even = quantsim::e8::roots()
        .iter()
        .all(|r| cube::vertex_parity(&std::array::from_fn(|k| i64::from(r[k]))) == 0);
    println!("  and every root's eight vertex values sum to a multiple of four: {all_even}");

    // ── 3. The cube's symmetry inside the lattice's ──────────────────
    println!("\n── the cube's symmetry group, inside W(E8)");
    let (cube_syms, also_lattice) = cube::cube_symmetry_order();
    println!(
        "  of the 40320 coordinate permutations, {cube_syms} preserve the cube's twelve \
         edges ({also_lattice} of those"
    );
    println!(
        "  verified as E8 automorphisms against all 240 roots) — Z2^3 x S3: eight vertex \
         translations, six axis relabelings"
    );
    let shear = cube::linear_map([1, 3, 4]).expect("invertible");
    println!(
        "  a shear {shear:?} is a lattice automorphism ({}) but NOT a cube symmetry ({}) — \
         it sends edges to diagonals",
        cube::is_lattice_automorphism(&shear),
        cube::preserves_cube(&shear)
    );

    // The cube symmetries act on real states.
    let mut state = E8ConstellationState::new(16)?;
    state.load(&[(0, C64::new(0.6, 0.0)), (0x0105, C64::new(0.8, 0.0))])?;
    let before = state.stored_points();
    state.permute_coordinates(&cube::translation(5))?;
    println!(
        "\n  translating the volume by vertex 5 moves its support {:?} → {:?}",
        before,
        state.stored_points()
    );
    state.permute_coordinates(&cube::translation(5))?;
    println!(
        "  and again returns it exactly: {:?} (a vertex translation is an involution)",
        state.stored_points()
    );

    // ── 4. The tower: a cube of cubes ────────────────────────────────
    println!("\n── the scale tower: a cube whose every vertex is a cube");
    let digits = [3u8, 17, 200];
    let p = cube::from_tower(&digits);
    println!("  cube tower {digits:?} is the lattice point {p:?}");
    println!(
        "  its vertex values      : {:?}",
        cube::vertex_values(&p).unwrap()
    );
    println!(
        "  read back as a tower   : {:?}",
        cube::tower(&p, 3).unwrap()
    );
    let doubled: [i64; 8] = p.map(|c| c << 1);
    println!(
        "  doubled (one scale out): {:?} — an empty finest cube prepended, the \
         self-similarity of the tower",
        cube::tower(&doubled, 4).unwrap()
    );

    // ── 5. Inward and outward ────────────────────────────────────────
    println!("\n── inward and outward: displacing a volume's interior vs its siblings");
    println!(
        "  a volume at scale 2, vertex 0: inward {:?}, outward {:?}",
        cube::inward(2, 0)?,
        cube::outward(2, 0)?
    );
    println!(
        "  at the finest scale there is no interior: {}",
        cube::inward(0, 0).unwrap_err()
    );

    for qubits in [16usize, 24, 32, 40] {
        let m = qubits / 8;
        println!("\n  depth-{m} tower ({qubits} qubits), all pairs along vertex 0:");
        print!("      T\\M ");
        for k in 0..m {
            print!("{k:>12}");
        }
        println!();
        for j in 0..m {
            print!("  T@{j:<5} ");
            for k in 0..m {
                let i = cube::interaction(qubits, (j, 0), (k, 0))?;
                if i.commutes {
                    print!("{:>12}", ".");
                } else {
                    print!("{:>12.4}", i.coupling);
                }
            }
            println!();
        }
        match cube::self_reference_depth(qubits, 0)? {
            Some(d) => println!("    self-reference depth: j+k up to {d}"),
            None => println!("    self-reference depth: none — the volume cannot reach itself"),
        }
    }
    println!("\n  '.' is an EXACT commutation, not a small number. The measured law: an inward");
    println!(
        "  displacement at level j and an outward modulation at level k interact iff \
         j + k < m - 2,"
    );
    println!("  with commutator phase exp(2*pi*i * 2^(j+k+2-m)) — exactly -1 on the horizon");
    println!("  (coupling 2), a quarter turn one level inside it (1.4142), an eighth turn one");
    println!("  further (0.7654). Deeper participation is FINER participation. Every entry is");
    println!("  read off two evolved states and cross-checked against the integer pairing.");

    println!("\n── and across different vertices");
    for (j, k, v) in [(0usize, 0usize, 1usize), (0, 1, 3), (1, 2, 7)] {
        let i = cube::interaction(24, (j, 0), (k, v))?;
        println!(
            "  T@{j} vertex 0 against M@{k} vertex {v}: coupling {:.1e}, commutes {}",
            i.coupling, i.commutes
        );
    }
    println!(
        "  the eight vertices are eight exactly independent channels — the ambient \
         coordinates are orthogonal,"
    );
    println!("  so a volume talks to its interior only along the direction it displaces.");

    // ── 6. The bridge to the qubit register ──────────────────────────
    println!("\n── the same volume as a qubit register");
    println!(
        "  Shape::CUBE bonds == e8::cube::edges(): {}",
        Shape::CUBE.bonds() == cube::edges()
    );
    for depth in 1..=2 {
        let l = RecursiveLattice::nest(Shape::CUBE, depth)?;
        println!(
            "  nest(CUBE, {depth}): {:>2} qubits, {:>3} bonds, horizon {}",
            l.width(),
            l.bonds().len(),
            l.topology()?.diameter()
        );
    }
    println!(
        "  so the tower's cube of cubes and the recursive lattice's are the same object, \
         reached from two directions"
    );

    Ok(())
}
