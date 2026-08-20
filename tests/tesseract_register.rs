//! The tesseract as a register: the dimensional lift, the hypercube's two
//! readings (a coupling fabric *and* a qudit's operator basis), the edge
//! colouring that decides what each coupler generates, and the shell that
//! holds a register while its interior stays separable.
//!
//! Everything here is exact — edge sets compared as sets, ranks over
//! `GF(2)`, dimensions counted. Nothing is a tolerance.

use quantsim::backend::Topology;
use quantsim::cut::CutGraph;
use quantsim::recursive::{RecursiveLattice, Shape};
use quantsim::scalar::C64;

// ── helpers ──────────────────────────────────────────────────────────

fn edge_set(t: &Topology) -> Vec<(usize, usize)> {
    let n = t.num_sites();
    let mut e = Vec::new();
    for a in 0..n {
        for b in (a + 1)..n {
            if t.adjacent(a, b) {
                e.push((a, b));
            }
        }
    }
    e.sort_unstable();
    e
}

/// The symplectic form on `F₂^{2n}`: 0 = the two Paulis commute, 1 = they
/// anticommute.
fn omega(n: usize, a: usize, b: usize) -> usize {
    let mask = (1usize << n) - 1;
    let (ax, az) = (a & mask, (a >> n) & mask);
    let (bx, bz) = (b & mask, (b >> n) & mask);
    ((ax & bz).count_ones() + (az & bx).count_ones()) as usize % 2
}

/// Vertex `v` of `Q_{2n}` as the anti-Hermitian generator `i·P`, where `P`
/// is the n-qubit Pauli in the **Hermitian** convention
/// `P = i^{x·z} X^x Z^z` — without that phase `XZ` is anti-Hermitian and
/// `i·XZ` leaves `su(2^n)` entirely.
fn generator(n: usize, v: usize) -> Vec<C64> {
    let d = 1usize << n;
    let zero = C64::new(0.0, 0.0);
    let one = C64::new(1.0, 0.0);
    let mut m = vec![zero; d * d];
    for i in 0..d {
        m[i * d + i] = one;
    }
    for q in 0..n {
        let (x, z) = ((v >> q) & 1, (v >> (n + q)) & 1);
        let s: [[C64; 2]; 2] = match (x, z) {
            (0, 0) => [[one, zero], [zero, one]],
            (1, 0) => [[zero, one], [one, zero]],
            (0, 1) => [[one, zero], [zero, C64::new(-1.0, 0.0)]],
            _ => [[zero, C64::new(0.0, -1.0)], [C64::new(0.0, 1.0), zero]],
        };
        let mut out = vec![zero; d * d];
        for r in 0..d {
            for c in 0..d {
                let (rb, cb) = ((r >> q) & 1, (c >> q) & 1);
                let f = s[rb][cb];
                if f == zero {
                    continue;
                }
                let rr = r & !(1 << q);
                let cc = c & !(1 << q);
                out[r * d + c] += f * m[(rr | (rb << q)) * d + (cc | (rb << q))];
            }
        }
        m = out;
    }
    let i = C64::new(0.0, 1.0);
    m.iter().map(|&e| i * e).collect()
}

fn bracket(d: usize, a: &[C64], b: &[C64]) -> Vec<C64> {
    let zero = C64::new(0.0, 0.0);
    let mut ab = vec![zero; d * d];
    let mut ba = vec![zero; d * d];
    for i in 0..d {
        for k in 0..d {
            let (x, y) = (a[i * d + k], b[i * d + k]);
            if x != zero {
                for j in 0..d {
                    ab[i * d + j] += x * b[k * d + j];
                }
            }
            if y != zero {
                for j in 0..d {
                    ba[i * d + j] += y * a[k * d + j];
                }
            }
        }
    }
    (0..d * d).map(|i| ab[i] - ba[i]).collect()
}

/// A real Lie algebra lives in a real vector space: `d²` complex entries
/// become `2d²` reals.
fn realify(m: &[C64]) -> Vec<f64> {
    let mut v = Vec::with_capacity(m.len() * 2);
    for c in m {
        v.push(c.re);
        v.push(c.im);
    }
    v
}

fn add_row(basis: &mut Vec<Vec<f64>>, mut v: Vec<f64>) -> bool {
    for b in basis.iter() {
        let p = b.iter().position(|x| x.abs() > 1e-9).unwrap();
        if v[p].abs() > 1e-9 {
            let f = v[p] / b[p];
            for (x, y) in v.iter_mut().zip(b.iter()) {
                *x -= f * y;
            }
        }
    }
    if v.iter().all(|x| x.abs() < 1e-9) {
        return false;
    }
    let p = v.iter().position(|x| x.abs() > 1e-9).unwrap();
    let pv = v[p];
    for x in v.iter_mut() {
        *x /= pv;
    }
    basis.push(v);
    basis.sort_by_key(|b| b.iter().position(|x| x.abs() > 1e-9).unwrap());
    true
}

/// The dynamical Lie algebra dimension: the real span closed under the
/// bracket. For an already-closed algebra it returns its own dimension,
/// which is the correctness anchor.
fn dla(n: usize, gens: &[Vec<C64>]) -> usize {
    let d = 1usize << n;
    let mut mats: Vec<Vec<C64>> = Vec::new();
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for g in gens {
        if add_row(&mut basis, realify(g)) {
            mats.push(g.clone());
        }
    }
    loop {
        let snapshot = mats.clone();
        let mut grew = false;
        for i in 0..snapshot.len() {
            for j in (i + 1)..snapshot.len() {
                let c = bracket(d, &snapshot[i], &snapshot[j]);
                if add_row(&mut basis, realify(&c)) {
                    mats.push(c);
                    grew = true;
                }
            }
        }
        if !grew {
            return basis.len();
        }
    }
}

fn graph(n: usize, bonds: &[(usize, usize)]) -> CutGraph {
    let mut g = CutGraph::uniform(n, 2).unwrap();
    for &(a, b) in bonds {
        g.bond(a, b).unwrap();
    }
    g
}

// ── the lift ─────────────────────────────────────────────────────────

/// **The lift is the next hypercube, and it is conservative.** `Q_d □ K₂`
/// is `Q_{d+1}` on the nose — every existing coupler survives, each site
/// gains exactly one partner, and diameter and degree each rise by one.
/// A cubic register is *tesseractable* in exactly this sense.
#[test]
fn the_lift_is_the_next_hypercube_and_it_is_conservative() {
    for d in 1..=5usize {
        let base = Topology::hypercube(d);
        let lifted = base.lift().unwrap();
        let want = Topology::hypercube(d + 1);

        assert_eq!(
            edge_set(&lifted),
            edge_set(&want),
            "Q{d} □ K2 must be Q{}",
            d + 1
        );
        assert_eq!(lifted.num_sites(), 2 * base.num_sites());
        assert_eq!(lifted.num_edges(), 2 * base.num_edges() + base.num_sites());

        // conservative: the register that was there is still there
        for a in 0..base.num_sites() {
            for b in 0..base.num_sites() {
                if base.adjacent(a, b) {
                    assert!(lifted.adjacent(a, b), "the lift dropped coupler ({a},{b})");
                }
            }
        }
        // and each site gained exactly one partner
        for v in 0..base.num_sites() {
            assert!(lifted.adjacent(v, v + base.num_sites()));
        }
        assert_eq!(lifted.diameter(), base.diameter() + 1);
        assert_eq!(lifted.max_degree(), base.max_degree() + 1);
    }

    // the shape-level lift agrees with the fabric-level one
    assert_eq!(Shape::CUBE.lift().unwrap(), Shape::Hypercube(4));
    let tess = RecursiveLattice::new(Shape::CUBE.lift().unwrap()).unwrap();
    assert_eq!((tess.width(), tess.bonds().len()), (16, 32));
}

/// The lift refuses shapes it would carry out of their own family: `G □ K₂`
/// of a cycle is not a cycle, so `Shape::lift` says so by name rather than
/// returning something mislabelled.
#[test]
fn the_shape_lift_refuses_what_leaves_its_family() {
    for s in [
        Shape::SQUARE,
        Shape::Cycle(5),
        Shape::Clique(4),
        Shape::Chain(3),
    ] {
        let e = s.lift().unwrap_err().to_string();
        assert!(e.contains("not a hypercube"), "unexpected refusal: {e}");
    }
    // but the fabric-level lift is defined for every topology
    let ring = Topology::ring(5);
    let lifted = ring.lift().unwrap();
    assert_eq!(lifted.num_sites(), 10);
    assert_eq!(lifted.num_edges(), 2 * ring.num_edges() + 5);
}

// ── the tesseract is a torus ─────────────────────────────────────────

/// **`Q_{2n}` is the n-axis, 4-site torus** `C₄^□n`, edge set for edge set —
/// so the tesseract is already a 4×4 torus and `Q₆` is already the 4×4×4
/// triaxial one. No projection is added; the identity is exact.
#[test]
fn the_even_hypercube_is_a_four_site_torus() {
    // C4 cycle order on a bit pair: 00 - 01 - 11 - 10 - 00.
    const CYCLE: [usize; 4] = [0b00, 0b01, 0b11, 0b10];
    for axes in 1..=4usize {
        let sites = 4usize.pow(axes as u32);
        let encode = |coords: &[usize]| -> usize {
            let mut v = 0usize;
            for (j, &c) in coords.iter().enumerate() {
                let bits = CYCLE[c % 4];
                if bits & 1 != 0 {
                    v |= 1 << j;
                }
                if bits & 2 != 0 {
                    v |= 1 << (axes + j);
                }
            }
            v
        };
        let mut torus = Vec::new();
        for idx in 0..sites {
            let mut coords = vec![0usize; axes];
            let mut t = idx;
            for c in coords.iter_mut() {
                *c = t % 4;
                t /= 4;
            }
            let from = encode(&coords);
            for j in 0..axes {
                for step in [1usize, 3] {
                    let mut nb = coords.clone();
                    nb[j] = (nb[j] + step) % 4;
                    let to = encode(&nb);
                    if from < to {
                        torus.push((from, to));
                    }
                }
            }
        }
        torus.sort_unstable();
        torus.dedup();
        assert_eq!(
            torus,
            edge_set(&Topology::hypercube(2 * axes)),
            "Q{} must be the {axes}-axis 4-site torus",
            2 * axes
        );
    }
}

/// **One toroidal axis is one qubit.** Axis `j` is the bit pair
/// `(x_j, z_j)`, and going around it walks that qubit's four Paulis
/// `I → Z → Y → X → I`. Every edge moves exactly one qubit, by one step.
#[test]
fn each_toroidal_axis_is_one_qubit() {
    const CYCLE: [usize; 4] = [0b00, 0b01, 0b11, 0b10];
    let n = 3usize;
    let q = Topology::hypercube(2 * n);
    let pauli = |v: usize, j: usize| ((v >> j) & 1) | (((v >> (n + j)) & 1) << 1);
    let pos = |p: usize| CYCLE.iter().position(|&c| c == p).unwrap();
    for (a, b) in edge_set(&q) {
        let changed: Vec<usize> = (0..n).filter(|&j| pauli(a, j) != pauli(b, j)).collect();
        assert_eq!(changed.len(), 1, "edge ({a},{b}) moved more than one qubit");
        let j = changed[0];
        let step = (pos(pauli(a, j)) + 4 - pos(pauli(b, j))) % 4;
        assert!(
            step == 1 || step == 3,
            "edge ({a},{b}) is not one cyclic step"
        );
    }
}

// ── the register / qudit duality ─────────────────────────────────────

/// **`Q_{2n}` is simultaneously a register geometry and one qudit's
/// operator basis**: `4ⁿ` vertices, and `dim u(2ⁿ) = 4ⁿ`. Vertex 0 is the
/// identity — the `u(1)` trace direction `su(2ⁿ)` quotients out. The odd
/// hypercubes are not operator bases at all, which is why the tesseract is
/// the right object and the cube is not.
#[test]
fn the_even_hypercubes_are_qudit_operator_bases() {
    for n in 1..=4u32 {
        let verts = Topology::hypercube(2 * n as usize).num_sites();
        let dim_u = (1usize << n) * (1usize << n);
        assert_eq!(verts, 4usize.pow(n));
        assert_eq!(verts, dim_u, "Q_{} must carry dim u(2^{n})", 2 * n);
    }
    // odd cubes: the vertex count is not a perfect square, so it indexes
    // no qudit's operators
    for d in [3usize, 5, 7] {
        let v = 1usize << d;
        let r = (v as f64).sqrt() as usize;
        assert_ne!(r * r, v, "Q{d} unexpectedly indexed an operator basis");
    }
    // vertex 0 is the identity, at every width
    for n in 1..=3usize {
        let d = 1usize << n;
        let g = generator(n, 0);
        // generator() returns i·P, so vertex 0 is i·I
        for r in 0..d {
            for c in 0..d {
                let want = if r == c {
                    C64::new(0.0, 1.0)
                } else {
                    C64::new(0.0, 0.0)
                };
                assert!(
                    (g[r * d + c] - want).norm() < 1e-12,
                    "vertex 0 is not the identity"
                );
            }
        }
    }
}

/// **The edge colouring is exactly half, at every width.** The symplectic
/// form 2-colours the hypercube's edges into commuting and anticommuting,
/// with no edge ambiguous and the split exactly even.
#[test]
fn the_edge_colouring_is_exactly_half() {
    for n in 1..=4usize {
        let q = Topology::hypercube(2 * n);
        let (mut comm, mut anti) = (0usize, 0usize);
        for (a, b) in edge_set(&q) {
            if omega(n, a, b) == 0 {
                comm += 1
            } else {
                anti += 1
            }
        }
        assert_eq!(comm, anti, "the colouring must split evenly at n={n}");
        assert_eq!(comm + anti, q.num_edges());
    }
}

/// **The colouring decides what each edge generates.** An anticommuting
/// edge generates an `su(2)`; a commuting one generates nothing new. All
/// fifteen traceless vertices close on `su(4)` — the anchor that says the
/// instrument is trustworthy, since a closed algebra must return its own
/// dimension.
#[test]
fn the_edge_colouring_decides_what_an_edge_generates() {
    let n = 2usize;
    // anchor: a closed algebra returns its own dimension
    let all: Vec<Vec<C64>> = (1..16).map(|v| generator(n, v)).collect();
    assert_eq!(dla(n, &all), 15, "the 15 traceless vertices are su(4)");

    let q = Topology::hypercube(4);
    let (mut comm_seen, mut anti_seen) = (0usize, 0usize);
    for (a, b) in edge_set(&q) {
        if a == 0 || b == 0 {
            continue; // the identity vertex generates nothing
        }
        let d = dla(n, &[generator(n, a), generator(n, b)]);
        if omega(n, a, b) == 0 {
            assert_eq!(d, 2, "a commuting edge ({a},{b}) must stay abelian");
            comm_seen += 1;
        } else {
            assert_eq!(d, 3, "an anticommuting edge ({a},{b}) must generate su(2)");
            anti_seen += 1;
        }
    }
    assert_eq!((comm_seen, anti_seen), (12, 16));

    // every edge at the identity vertex is inert
    for v in [1usize, 2, 4, 8] {
        assert_eq!(omega(n, 0, v), 0, "the identity commutes with everything");
    }
    // and single-qubit axes never leave the product structure
    let axes: Vec<Vec<C64>> = [1usize, 2, 4, 8].iter().map(|&v| generator(n, v)).collect();
    assert_eq!(
        dla(n, &axes),
        6,
        "X0,Z0,X1,Z1 generate su(2) + su(2), not su(4)"
    );
    assert_eq!(dla(n, &[generator(n, 1)]), 1);
}

// ── the shell ────────────────────────────────────────────────────────

/// **The shell is the lift, and its cut is a perfect matching.** Attaching
/// a mirror copy of the cube plus the corner matching gives exactly
/// `hypercube(4)`, the cut's crossing edges are vertex-disjoint, and the
/// Schmidt rank across it is `2⁸ = 256` — the Schur bound attained, not
/// merely bounded.
#[test]
fn the_shell_is_the_lift_and_its_cut_is_a_perfect_matching() {
    let cube = RecursiveLattice::new(Shape::CUBE).unwrap();
    let shelled = cube.topology().unwrap().lift().unwrap();
    assert_eq!(edge_set(&shelled), edge_set(&Topology::hypercube(4)));

    let g = graph(16, &edge_set(&shelled));
    let cut = CutGraph::cut_of(&(0..8).collect::<Vec<_>>());
    assert_eq!(g.crossing(&cut).len(), 8, "one crossing edge per data site");
    assert!(
        g.is_matching(&cut),
        "the crossing edges must be vertex-disjoint"
    );
    assert_eq!(g.exact_rank(&cut).unwrap(), 256, "Schur is attained: 2^8");
    assert_eq!(g.schur_bound(&cut), 256);
}

/// **The shell holds the whole register while the interior stays
/// separable.** Over a nested block network the data|shell cut is maximal
/// (rank 64 on 64 data qubits) and the interior's own worst cut stays at
/// the area-law rank 7. Both are true at once, which is the point.
#[test]
fn the_shell_holds_the_register_while_the_interior_stays_area_law() {
    let lat = RecursiveLattice::nest(Shape::CUBE, 2).unwrap();
    let data = lat.bond_pairs();
    assert_eq!(lat.width(), 64);

    let shelled = lat.topology().unwrap().lift().unwrap();
    assert_eq!(shelled.num_sites(), 128);
    let g = graph(128, &edge_set(&shelled));
    let cut = CutGraph::cut_of(&(0..64).collect::<Vec<_>>());
    assert_eq!(g.crossing(&cut).len(), 64);
    assert!(g.is_matching(&cut));
    assert_eq!(
        g.gf2_cut_rank(&g.natural_order(), 64),
        64,
        "the shell must hold the whole register"
    );

    let interior = graph(64, &data);
    let order = interior.natural_order();
    let worst = (1..64)
        .map(|p| interior.gf2_cut_rank(&order, p))
        .max()
        .unwrap();
    assert_eq!(
        worst, 7,
        "the interior stays area-law while the shell is maximal"
    );
}

// ── repositioning, and the two scaling laws ──────────────────────────

/// **Repositioning is the intersection of geometry and dynamics, and it is
/// small.** Of the tesseract's 384 automorphisms, 8 preserve the
/// symplectic form; of `Sp(4,F₂)`'s 720 elements, the same 8 are
/// hypercube automorphisms. No translation survives — `ω` is
/// nondegenerate, so `ω(·,t) ≡ 0` forces `t = 0`.
#[test]
fn repositioning_is_the_small_intersection_of_geometry_and_dynamics() {
    let n = 2usize;
    let nv = 16usize;

    // permutations of the 4 bit positions
    let mut perms: Vec<Vec<usize>> = Vec::new();
    let mut idx: Vec<usize> = (0..4).collect();
    fn rec(i: usize, idx: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if i == idx.len() {
            out.push(idx.clone());
            return;
        }
        for j in i..idx.len() {
            idx.swap(i, j);
            rec(i + 1, idx, out);
            idx.swap(i, j);
        }
    }
    rec(0, &mut idx, &mut perms);
    assert_eq!(perms.len() * nv, 384, "|Aut(Q4)| = 4! x 2^4");

    let mut realizable = 0usize;
    let mut with_translation = 0usize;
    for p in &perms {
        for t in 0..nv {
            let f = |v: usize| {
                let mut w = 0usize;
                for (b, &pb) in p.iter().enumerate() {
                    if (v >> b) & 1 == 1 {
                        w |= 1 << pb;
                    }
                }
                w ^ t
            };
            if (0..nv).all(|a| (0..nv).all(|b| omega(n, f(a), f(b)) == omega(n, a, b))) {
                realizable += 1;
                if t != 0 {
                    with_translation += 1;
                }
            }
        }
    }
    assert_eq!(realizable, 8, "only 8 of 384 automorphisms are realizable");
    assert_eq!(
        with_translation, 0,
        "no translation preserves a nondegenerate form"
    );

    // and the other direction: how much of Clifford keeps the geometry
    let (mut sp, mut both) = (0usize, 0usize);
    for m in 0..(1u32 << 16) {
        let row = |i: usize| ((m >> (4 * i)) & 0xF) as usize;
        let apply = |v: usize| {
            let mut w = 0usize;
            for i in 0..4 {
                if (row(i) & v).count_ones() % 2 == 1 {
                    w |= 1 << i;
                }
            }
            w
        };
        let mut seen = vec![false; nv];
        if (0..nv).any(|v| {
            let w = apply(v);
            let dup = seen[w];
            seen[w] = true;
            dup
        }) {
            continue;
        }
        if !(0..nv).all(|a| (0..nv).all(|b| omega(n, apply(a), apply(b)) == omega(n, a, b))) {
            continue;
        }
        sp += 1;
        if (0..4).all(|b| apply(1 << b).count_ones() == 1) {
            both += 1;
        }
    }
    assert_eq!(sp, 720, "|Sp(4,F2)| = 720");
    assert_eq!(both, 8, "the same 8 elements, from the other side");
}

/// **Nesting is area-law where lifting is volume-law.** The nested block
/// network's exact Schmidt exponent grows by a bounded amount per level
/// (logarithmic in width, so polynomial bond dimension); the hypercube of
/// the same width sits at the maximum `n/2`.
#[test]
fn nesting_is_area_law_where_lifting_is_volume_law() {
    // arity-3 nesting: rank climbs one per level, width triples
    let ranks: Vec<usize> = (1..=4)
        .map(|d| {
            let lat = RecursiveLattice::nest(Shape::Cycle(3), d).unwrap();
            let g = graph(lat.width(), &lat.bond_pairs());
            let o = g.natural_order();
            (1..lat.width())
                .map(|p| g.gf2_cut_rank(&o, p))
                .max()
                .unwrap()
        })
        .collect();
    assert_eq!(
        ranks,
        vec![1, 2, 3, 4],
        "one rung of rank per level of nesting"
    );

    // at 64 qubits: nested vs lifted
    let nested = RecursiveLattice::nest(Shape::CUBE, 2).unwrap();
    let g = graph(64, &nested.bond_pairs());
    let o = g.natural_order();
    let nested_rank = (1..64).map(|p| g.gf2_cut_rank(&o, p)).max().unwrap();

    let q6 = graph(64, &edge_set(&Topology::hypercube(6)));
    let o6 = q6.natural_order();
    let lifted_rank = (1..64).map(|p| q6.gf2_cut_rank(&o6, p)).max().unwrap();

    assert_eq!(nested_rank, 7, "the nested network is area-law");
    assert_eq!(lifted_rank, 32, "the hypercube is volume-law: n/2");
    assert!(lifted_rank > 4 * nested_rank);

    // and the geometry trade that buys it
    let t = Topology::hypercube(6);
    assert_eq!((t.diameter(), t.max_degree()), (6, 6));
    let nt = nested.topology().unwrap();
    assert_eq!((nt.diameter(), nt.max_degree()), (11, 4));
}
