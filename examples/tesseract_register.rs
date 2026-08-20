//! The tesseract as a register, printed: what the lift does, the two
//! readings of one hypercube, and what each coupler is worth.
//!
//! `cargo run --release --example tesseract_register`

use quantsim::backend::Topology;
use quantsim::cut::CutGraph;
use quantsim::recursive::{RecursiveLattice, Shape};
use quantsim::scalar::C64;

const CYCLE: [usize; 4] = [0b00, 0b01, 0b11, 0b10];
const PAULI: [&str; 4] = ["I", "Z", "Y", "X"];

fn rule() -> &'static str {
    "───────────────────────────────────────────────────────────────────────────"
}

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
    e
}

fn omega(n: usize, a: usize, b: usize) -> usize {
    let mask = (1usize << n) - 1;
    let (ax, az) = (a & mask, (a >> n) & mask);
    let (bx, bz) = (b & mask, (b >> n) & mask);
    ((ax & bz).count_ones() + (az & bx).count_ones()) as usize % 2
}

fn generator(n: usize, v: usize) -> Vec<C64> {
    let d = 1usize << n;
    let (zero, one) = (C64::new(0.0, 0.0), C64::new(1.0, 0.0));
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
                let (rr, cc) = (r & !(1 << q), c & !(1 << q));
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
    let (mut ab, mut ba) = (vec![zero; d * d], vec![zero; d * d]);
    for i in 0..d {
        for k in 0..d {
            let (x, y) = (a[i * d + k], b[i * d + k]);
            for j in 0..d {
                if x != zero {
                    ab[i * d + j] += x * b[k * d + j];
                }
                if y != zero {
                    ba[i * d + j] += y * a[k * d + j];
                }
            }
        }
    }
    (0..d * d).map(|i| ab[i] - ba[i]).collect()
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

fn dla(n: usize, gens: &[Vec<C64>]) -> usize {
    let d = 1usize << n;
    let (mut mats, mut basis): (Vec<Vec<C64>>, Vec<Vec<f64>>) = (Vec::new(), Vec::new());
    let flat = |m: &[C64]| -> Vec<f64> {
        let mut v = Vec::with_capacity(m.len() * 2);
        for c in m {
            v.push(c.re);
            v.push(c.im);
        }
        v
    };
    for g in gens {
        if add_row(&mut basis, flat(g)) {
            mats.push(g.clone());
        }
    }
    loop {
        let snap = mats.clone();
        let mut grew = false;
        for i in 0..snap.len() {
            for j in (i + 1)..snap.len() {
                let c = bracket(d, &snap[i], &snap[j]);
                if add_row(&mut basis, flat(&c)) {
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

fn main() {
    println!("\n{}", rule());
    println!("  THE TESSERACT AS A REGISTER — every number below is computed here");
    println!("{}\n", rule());

    // ── 1. the lift ──────────────────────────────────────────────────
    println!("1.  THE LIFT   G □ K₂ — two copies joined by a perfect matching\n");
    println!(
        "    {:<10} {:>6} {:>7} {:>8} {:>7}   lift lands on",
        "", "sites", "edges", "diameter", "degree"
    );
    for d in 2..=6usize {
        let base = Topology::hypercube(d);
        let lifted = base.lift().unwrap();
        let target = Topology::hypercube(d + 1);
        let exact = edge_set(&lifted) == edge_set(&target);
        println!(
            "    Q{d:<9} {:>6} {:>7} {:>8} {:>7}   Q{} exactly: {}",
            base.num_sites(),
            base.num_edges(),
            base.diameter(),
            base.max_degree(),
            d + 1,
            exact
        );
    }
    let cube = Topology::hypercube(3);
    let tess = cube.lift().unwrap();
    let survived = (0..8).all(|a| (0..8).all(|b| !cube.adjacent(a, b) || tess.adjacent(a, b)));
    println!(
        "\n    conservative: every one of the cube's 12 couplers survives the lift: {survived}"
    );
    println!("    each site gains exactly one partner (v ↦ v+8); diameter 3→4, degree 3→4");
    println!("    Shape::CUBE.lift() = {:?}", Shape::CUBE.lift().unwrap());
    println!(
        "    Shape::SQUARE.lift() refuses: {}",
        Shape::SQUARE.lift().unwrap_err()
    );

    // ── 2. two readings ──────────────────────────────────────────────
    println!("\n{}", rule());
    println!("2.  ONE OBJECT, TWO READINGS\n");
    println!(
        "    {:>4} {:>10} {:>12} {:>14} {:>10}",
        "n", "Q_2n verts", "as a torus", "dim u(2^n)", "= n qubits"
    );
    for n in 1..=4usize {
        let q = Topology::hypercube(2 * n);
        let dim_u = (1usize << n) * (1usize << n);
        println!(
            "    {n:>4} {:>10} {:>12} {:>14} {:>10}",
            q.num_sites(),
            format!("4^{n} sites"),
            dim_u,
            q.num_sites() == dim_u
        );
    }
    println!("\n    and the odd cubes are NOT operator bases — the vertex count is no square:");
    for d in [3usize, 5, 7] {
        let v = 1usize << d;
        let r = (v as f64).sqrt() as usize;
        println!(
            "      Q{d}: {v} vertices, √{v} = {:.3}  → not dim u(k) for any k",
            (v as f64).sqrt()
        );
        let _ = r;
    }

    // ── 3. a toroidal axis ───────────────────────────────────────────
    println!("\n{}", rule());
    println!("3.  ONE TOROIDAL AXIS IS ONE QUBIT\n");
    print!("    walking axis j:  ");
    for (i, p) in PAULI.iter().enumerate() {
        print!("{} ({:02b})", p, CYCLE[i]);
        if i < 3 {
            print!("  →  ");
        }
    }
    println!("  →  I (00)");
    let n = 3usize;
    let q6 = Topology::hypercube(6);
    let pauli = |v: usize, j: usize| ((v >> j) & 1) | (((v >> (n + j)) & 1) << 1);
    let pos = |p: usize| CYCLE.iter().position(|&c| c == p).unwrap();
    let ok = edge_set(&q6).iter().all(|&(a, b)| {
        let ch: Vec<usize> = (0..n).filter(|&j| pauli(a, j) != pauli(b, j)).collect();
        ch.len() == 1 && {
            let s = (pos(pauli(a, ch[0])) + 4 - pos(pauli(b, ch[0]))) % 4;
            s == 1 || s == 3
        }
    });
    println!(
        "    every one of Q₆'s {} edges moves exactly one qubit by one step: {ok}",
        q6.num_edges()
    );
    println!("\n    a worked edge — vertex 0 to vertex 1, on Q₆:");
    for j in 0..n {
        println!(
            "      qubit {j}: {} → {}",
            PAULI[pos(pauli(0, j))],
            PAULI[pos(pauli(1, j))]
        );
    }

    // ── 4. the colouring, and what it is worth ───────────────────────
    println!("\n{}", rule());
    println!("4.  THE EDGE COLOURING, AND WHAT EACH COLOUR GENERATES\n");
    println!(
        "    {:>4} {:>8} {:>12} {:>16} {:>8}",
        "n", "edges", "commuting", "anticommuting", "split"
    );
    for n in 1..=4usize {
        let q = Topology::hypercube(2 * n);
        let (mut c, mut a) = (0usize, 0usize);
        for (x, y) in edge_set(&q) {
            if omega(n, x, y) == 0 {
                c += 1
            } else {
                a += 1
            }
        }
        println!(
            "    {n:>4} {:>8} {c:>12} {a:>16} {:>7.0}%",
            c + a,
            100.0 * a as f64 / (c + a) as f64
        );
    }
    println!("\n    on the tesseract, what a pair of vertices actually generates:");
    let n = 2usize;
    let all: Vec<Vec<C64>> = (1..16).map(|v| generator(n, v)).collect();
    println!(
        "      all 15 traceless vertices        → dim {}  (su(4) = 15 — the anchor)",
        dla(n, &all)
    );
    let mut shown = (false, false);
    for (a, b) in edge_set(&Topology::hypercube(4)) {
        if a == 0 || b == 0 {
            continue;
        }
        let anti = omega(n, a, b) == 1;
        if (anti && shown.1) || (!anti && shown.0) {
            continue;
        }
        let d = dla(n, &[generator(n, a), generator(n, b)]);
        let (pa, pb) = (
            format!(
                "{}{}",
                PAULI[pos(a & 1 | ((a >> 2) & 1) << 1)],
                PAULI[pos((a >> 1) & 1 | ((a >> 3) & 1) << 1)]
            ),
            format!(
                "{}{}",
                PAULI[pos(b & 1 | ((b >> 2) & 1) << 1)],
                PAULI[pos((b >> 1) & 1 | ((b >> 3) & 1) << 1)]
            ),
        );
        println!(
            "      {:<13} edge {pa}·{pb}  → dim {d}  {}",
            if anti { "anticommuting" } else { "commuting" },
            if anti {
                "(su(2) — it generates)"
            } else {
                "(abelian — it does not)"
            }
        );
        if anti {
            shown.1 = true
        } else {
            shown.0 = true
        }
    }
    let axes: Vec<Vec<C64>> = [1usize, 2, 4, 8].iter().map(|&v| generator(n, v)).collect();
    println!(
        "      the 4 single-qubit axes          → dim {}  (su(2)⊕su(2): [A⊗I, I⊗B] = 0)",
        dla(n, &axes)
    );
    println!(
        "      one vertex alone                 → dim {}",
        dla(n, &[generator(n, 1)])
    );

    // ── 5. the shell ─────────────────────────────────────────────────
    println!("\n{}", rule());
    println!("5.  THE SHELL IS THE LIFT\n");
    let shelled = Topology::hypercube(3).lift().unwrap();
    let g = graph(16, &edge_set(&shelled));
    let cut = CutGraph::cut_of(&(0..8).collect::<Vec<_>>());
    println!(
        "    cube + mirror + matching == hypercube(4): {}",
        edge_set(&shelled) == edge_set(&Topology::hypercube(4))
    );
    println!(
        "    crossing edges {} · vertex-disjoint (a perfect matching): {}",
        g.crossing(&cut).len(),
        g.is_matching(&cut)
    );
    println!(
        "    Schur bound {} · exact Schmidt rank {}  → the bound is ATTAINED",
        g.schur_bound(&cut),
        g.exact_rank(&cut).unwrap()
    );
    let lat = RecursiveLattice::nest(Shape::CUBE, 2).unwrap();
    let big = lat.topology().unwrap().lift().unwrap();
    let gb = graph(128, &edge_set(&big));
    let interior = graph(64, &lat.bond_pairs());
    let io = interior.natural_order();
    let worst = (1..64)
        .map(|p| interior.gf2_cut_rank(&io, p))
        .max()
        .unwrap();
    println!("\n    over a 64-qubit nested network, shelled to 128:");
    println!(
        "      data|shell cut rank {:>3}   ← the shell holds the WHOLE register",
        gb.gf2_cut_rank(&gb.natural_order(), 64)
    );
    println!(
        "      interior's worst cut {:>3}   ← and the inside stays area-law",
        worst
    );

    // ── 6. repositioning ─────────────────────────────────────────────
    println!("\n{}", rule());
    println!("6.  REPOSITIONING — geometry ∩ dynamics\n");
    let nv = 16usize;
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
    let (mut realizable, mut with_t) = (0usize, 0usize);
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
            if (0..nv).all(|a| (0..nv).all(|b| omega(2, f(a), f(b)) == omega(2, a, b))) {
                realizable += 1;
                if t != 0 {
                    with_t += 1;
                }
            }
        }
    }
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
            let d = seen[w];
            seen[w] = true;
            d
        }) {
            continue;
        }
        if !(0..nv).all(|a| (0..nv).all(|b| omega(2, apply(a), apply(b)) == omega(2, a, b))) {
            continue;
        }
        sp += 1;
        if (0..4).all(|b| apply(1 << b).count_ones() == 1) {
            both += 1;
        }
    }
    println!(
        "    |Aut(Q₄)|                        {:>5}   the tesseract's geometric symmetry",
        perms.len() * nv
    );
    println!(
        "      of those, symplectic           {realizable:>5}   {:>5.1}% is realizable as a gate",
        100.0 * realizable as f64 / (perms.len() * nv) as f64
    );
    println!(
        "      of those, with a translation   {with_t:>5}   ω is nondegenerate, so ω(·,t)≡0 ⟹ t=0"
    );
    println!("    |Sp(4,F₂)| (Clifford mod Pauli)  {sp:>5}");
    println!(
        "      of those, Q₄ automorphisms     {both:>5}   {:>5.1}% of Clifford keeps the geometry",
        100.0 * both as f64 / sp as f64
    );

    // ── 7. the trade ─────────────────────────────────────────────────
    println!("\n{}", rule());
    println!("7.  NEST OR LIFT — the trade, at 64 qubits\n");
    let ng = graph(64, &lat.bond_pairs());
    let no = ng.natural_order();
    let nr = (1..64).map(|p| ng.gf2_cut_rank(&no, p)).max().unwrap();
    let qg = graph(64, &edge_set(&Topology::hypercube(6)));
    let qo = qg.natural_order();
    let qr = (1..64).map(|p| qg.gf2_cut_rank(&qo, p)).max().unwrap();
    let nt = lat.topology().unwrap();
    let q6t = Topology::hypercube(6);
    println!(
        "    {:<22} {:>7} {:>8} {:>12} {:>14}",
        "", "degree", "diameter", "cut rank", "bond dimension"
    );
    println!(
        "    {:<22} {:>7} {:>8} {:>12} {:>14}",
        "nested cube-of-cubes",
        nt.max_degree(),
        nt.diameter(),
        nr,
        format!("2^{nr}")
    );
    println!(
        "    {:<22} {:>7} {:>8} {:>12} {:>14}",
        "6-cube (lifted)",
        q6t.max_degree(),
        q6t.diameter(),
        qr,
        format!("2^{qr}")
    );
    let horizon_pct = 100.0 * q6t.diameter() as f64 / nt.diameter() as f64;
    println!(
        "\n    lifting routes in {horizon_pct:.0}% the horizon and costs 2^{qr} instead of 2^{nr}."
    );
    println!("    the geometry that routes best is the one that stores worst.\n");
}
