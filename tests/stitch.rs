//! The stitch as the sharding rule: the symplectic normal form of a
//! code, the `2^h` maximal isotropic extensions it forces, and the
//! check that those extensions are a genuine direct-sum decomposition
//! of the code space — on amplitudes, not on dimensions.

use quantsim::backend::PauliString;
use quantsim::prelude::*;
use quantsim::retro::{Code, SurfaceCode, ToricCode};
use quantsim::rng::Prng;
use quantsim::stitch::*;

/// Apply a Hermitian Pauli string to a raw amplitude vector:
/// `σ · i^{|x∧z|} X^x Z^z`, so `out[k] = phase(k⊕x) · v[k⊕x]`.
fn apply(p: PauliString, v: &[C64]) -> Vec<C64> {
    let i_pow = (p.x & p.z).count_ones() & 3;
    let base = match i_pow {
        0 => C64::new(1.0, 0.0),
        1 => C64::new(0.0, 1.0),
        2 => C64::new(-1.0, 0.0),
        _ => C64::new(0.0, -1.0),
    };
    let base = if p.negative { -base } else { base };
    (0..v.len())
        .map(|k| {
            let j = k ^ (p.x as usize);
            let s = if (j as u64 & p.z).count_ones() & 1 == 1 {
                -base
            } else {
                base
            };
            s * v[j]
        })
        .collect()
}

fn inner(a: &[C64], b: &[C64]) -> C64 {
    a.iter().zip(b).map(|(x, y)| x.conj() * y).sum()
}

fn norm(v: &[C64]) -> f64 {
    inner(v, v).re.sqrt()
}

fn normalize(v: &mut [C64]) {
    let n = norm(v);
    for a in v.iter_mut() {
        *a /= C64::new(n, 0.0);
    }
}

fn state_vec(c: &Circuit<C64>, n: usize, reg: &GateRegistry<C64>) -> Vec<C64> {
    let mut st = quantsim::backend::DenseState::<C64>::new(n).unwrap();
    c.bind(reg).unwrap().run(&mut st).unwrap();
    (0..1u64 << n).map(|i| st.amplitude(i)).collect()
}

/// Solve `G c = b` for a small Hermitian Gram system by Gaussian
/// elimination with partial pivoting.
fn solve(mut g: Vec<Vec<C64>>, mut b: Vec<C64>) -> Option<Vec<C64>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&a, &c| {
            g[a][col]
                .norm_sqr()
                .partial_cmp(&g[c][col].norm_sqr())
                .unwrap()
        })?;
        if g[piv][col].norm_sqr() < 1e-24 {
            return None;
        }
        g.swap(col, piv);
        b.swap(col, piv);
        let d = g[col][col];
        for v in g[col][col..n].iter_mut() {
            *v /= d;
        }
        b[col] /= d;
        let pivot: Vec<C64> = g[col][col..n].to_vec();
        let pivot_b = b[col];
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = g[r][col];
            if f.norm_sqr() == 0.0 {
                continue;
            }
            for (v, &pv) in g[r][col..n].iter_mut().zip(pivot.iter()) {
                *v -= f * pv;
            }
            b[r] -= f * pivot_b;
        }
    }
    Some(b)
}

// ───────────────────── the form, and what it decomposes into ─────────────────────

#[test]
fn the_normal_form_reads_the_code_off_its_own_generators() {
    // r is the stabilizer rank and h is the logical count, and neither
    // is told to the decomposition — both come out of the symplectic
    // Gram–Schmidt.
    for (label, gens, logicals, n, want_r, want_h) in [
        (
            "toric L=2",
            ToricCode::new(2, 0).unwrap().generators(),
            (0..2)
                .flat_map(|i| {
                    let c = ToricCode::new(2, 0).unwrap();
                    [c.logical_x(i), c.logical_z(i)]
                })
                .collect::<Vec<_>>(),
            8usize,
            6usize,
            2usize,
        ),
        (
            "toric L=3",
            ToricCode::new(3, 0).unwrap().generators(),
            (0..2)
                .flat_map(|i| {
                    let c = ToricCode::new(3, 0).unwrap();
                    [c.logical_x(i), c.logical_z(i)]
                })
                .collect::<Vec<_>>(),
            18,
            16,
            2,
        ),
        (
            "surface d=3",
            SurfaceCode::new(3, 0).unwrap().generators(),
            vec![
                SurfaceCode::new(3, 0).unwrap().logical_x(),
                SurfaceCode::new(3, 0).unwrap().logical_z(),
            ],
            9,
            8,
            1,
        ),
        (
            "surface d=5",
            SurfaceCode::new(5, 0).unwrap().generators(),
            vec![
                SurfaceCode::new(5, 0).unwrap().logical_x(),
                SurfaceCode::new(5, 0).unwrap().logical_z(),
            ],
            25,
            24,
            1,
        ),
    ] {
        let all: Vec<PauliString> = gens.iter().chain(logicals.iter()).copied().collect();
        let v = Volume::span(n, &all).unwrap();
        let st = orthogonalize(&v);

        assert_eq!(st.radical_rank(), want_r, "{label}: stabilizer rank");
        assert_eq!(st.witt(), want_h, "{label}: logical qubits = lateral axes");
        assert_eq!(
            st.rank(),
            v.rank(),
            "{label}: orthogonalising preserves rank"
        );
        assert_eq!(st.rank(), want_r + 2 * want_h, "{label}: r + 2h");

        // The radical is a stabilizer: it commutes with itself and with
        // the whole span.
        let rad = Volume::span(n, &st.radical()).unwrap();
        assert!(rad.is_isotropic(), "{label}: the radical is abelian");
        for r in st.radical() {
            for p in &all {
                assert!(
                    r.commutes_with(*p),
                    "{label}: radical is central in the span"
                );
            }
        }

        // Each pair genuinely anticommutes — that is what makes it
        // impossible to hold in one place.
        for (e, f) in st.pairs() {
            assert!(!e.commutes_with(f), "{label}: hyperbolic pair anticommutes");
        }

        // And the stitch closes: 2^h · 2^{n−r−h} = 2^{n−r}, exactly.
        assert!(st.closes(), "{label}");
        assert_eq!(st.region_dimension(), 1u128 << (n - want_r));
        assert_eq!(st.slices(), 1u128 << want_h);
    }
}

#[test]
fn every_selection_is_maximal_isotropic_and_no_two_can_merge() {
    // The 2^h extensions are each a maximal commuting set — one state
    // apiece — and any two of them, joined, stop commuting: that is
    // the exact sense in which no node can hold two slices.
    let code = ToricCode::new(2, 0).unwrap();
    let mut all = code.generators();
    for i in 0..2 {
        all.push(code.logical_x(i));
        all.push(code.logical_z(i));
    }
    let st = orthogonalize(&Volume::span(8, &all).unwrap());
    let sels = st.selections().unwrap();
    assert_eq!(sels.len(), 4);

    for (i, s) in sels.iter().enumerate() {
        assert_eq!(s.rank(), 8, "selection {i} has rank r + h = n");
        assert!(s.is_maximal_isotropic(), "selection {i}");
        assert_eq!(s.carves(), 1, "selection {i} fixes a single state");
    }
    for (i, a) in sels.iter().enumerate() {
        for (j, b) in sels.iter().enumerate().skip(i + 1) {
            let joined = a.join(b).unwrap();
            assert!(
                !joined.is_isotropic(),
                "selections {i} and {j} joined must contain an anticommuting pair"
            );
            // What they share is the radical plus the choices they agree
            // on — never the whole of either.
            let m = a.meet(b).unwrap();
            assert!(
                m.rank() < 8,
                "selections {i},{j} share less than everything"
            );
            assert!(
                m.rank() >= st.radical_rank(),
                "they always share the radical"
            );
        }
    }
}

#[test]
fn the_centraliser_is_the_normalizer() {
    // The radical's centraliser is the code's normalizer: rank
    // n + k = n + h, containing every stabilizer and every logical,
    // computed as an F₂ kernel with no enumeration.
    let code = ToricCode::new(2, 0).unwrap();
    let rad = Volume::span(8, &code.generators()).unwrap();
    assert_eq!(rad.rank(), 6);
    let cen = rad.centraliser().unwrap();
    assert_eq!(cen.rank(), 8 + 2, "n + k");
    for g in code.generators() {
        assert!(cen.contains(g), "a stabilizer is in its own normalizer");
    }
    for i in 0..2 {
        assert!(cen.contains(code.logical_x(i)));
        assert!(cen.contains(code.logical_z(i)));
    }
    // And it is not everything: a weight-1 fault leaves the normalizer.
    let fault = PauliString {
        x: 1,
        z: 0,
        negative: false,
    };
    assert!(!cen.contains(fault) || rad.contains(fault));
}

// ──────────── the part that decides it: slices on real amplitudes ────────────

#[test]
fn the_slices_are_a_direct_sum_of_the_code_space() {
    // Each of the 2^h slices is prepared as an actual state; they are
    // shown to lie in the code space, to be pairwise distinct, to be
    // linearly independent, and to span the code space exactly. That
    // is `2^h · 2^{n−r−h} = 2^{n−r}` instantiated on amplitudes.
    let reg = GateRegistry::<C64>::standard();
    let code = ToricCode::new(2, 0).unwrap();
    let n = 8usize;
    let mut all = code.generators();
    for i in 0..2 {
        all.push(code.logical_x(i));
        all.push(code.logical_z(i));
    }
    let st = orthogonalize(&Volume::span(n, &all).unwrap());
    let sels = st.selections().unwrap();

    // Read each selection's basis choice off the volume rather than
    // assuming the pair order: which of X̄ᵢ / Z̄ᵢ did it take?
    let mut states: Vec<Vec<C64>> = Vec::new();
    for s in &sels {
        let mut plus = [false; 2];
        for (i, p) in plus.iter_mut().enumerate() {
            let has_x = s.contains(code.logical_x(i));
            let has_z = s.contains(code.logical_z(i));
            assert!(
                has_x ^ has_z,
                "a selection takes exactly one of X̄/Z̄ per axis"
            );
            *p = has_x;
        }
        states.push(state_vec(&code.encoder(n, plus), n, &reg));
    }

    // Every slice is in the code space.
    for (i, v) in states.iter().enumerate() {
        for g in code.generators() {
            let e = inner(v, &apply(g, v)).re;
            assert!(
                (e - 1.0).abs() < 1e-9,
                "slice {i} stabilizer expectation {e}"
            );
        }
    }

    // Pairwise distinct, and — being different bases of the same code
    // space — *not* orthogonal. Disjointness here is trivial
    // intersection of lines, not orthogonality, and the test says which.
    let gram: Vec<Vec<C64>> = states
        .iter()
        .map(|a| states.iter().map(|b| inner(a, b)).collect())
        .collect();
    let mut off_diagonal_nonzero = 0;
    for (i, row) in gram.iter().enumerate() {
        for (j, g) in row.iter().enumerate() {
            if i != j && g.norm() > 1e-9 {
                off_diagonal_nonzero += 1;
            }
        }
    }
    assert!(
        off_diagonal_nonzero > 0,
        "the slices are a frame, not an orthogonal basis"
    );

    // Linearly independent: the Gram matrix is invertible. Four lines
    // in a four-dimensional code space, summing directly.
    let id: Vec<C64> = (0..4)
        .map(|i| {
            if i == 0 {
                C64::new(1.0, 0.0)
            } else {
                C64::new(0.0, 0.0)
            }
        })
        .collect();
    assert!(
        solve(gram.clone(), id).is_some(),
        "the 2^h slices are linearly independent"
    );

    // And they span the code space: project a random vector into the
    // code space, then reconstruct it from the slices alone.
    let mut rng = Prng::new(0xC0DE);
    let mut v: Vec<C64> = (0..1usize << n)
        .map(|_| {
            C64::new(
                rng.next_u64() as f64 / u64::MAX as f64 - 0.5,
                rng.next_u64() as f64 / u64::MAX as f64 - 0.5,
            )
        })
        .collect();
    // P = (1/2^r) Σ_{g ∈ S} g, summed over the whole stabilizer group.
    let rad = st.radical();
    let mut proj = vec![C64::new(0.0, 0.0); v.len()];
    for m in 0..(1u64 << rad.len()) {
        let mut g = PauliString::identity();
        let mut bits = m;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            g = g.times(rad[i]).expect("the radical commutes");
        }
        for (acc, a) in proj.iter_mut().zip(apply(g, &v)) {
            *acc += a;
        }
    }
    v = proj;
    assert!(norm(&v) > 1e-6, "the random vector had code-space content");
    normalize(&mut v);
    for g in code.generators() {
        assert!((inner(&v, &apply(g, &v)).re - 1.0).abs() < 1e-9);
    }

    let b: Vec<C64> = states.iter().map(|s| inner(s, &v)).collect();
    let c = solve(gram, b).expect("the frame is invertible");
    let mut recon = vec![C64::new(0.0, 0.0); v.len()];
    for (ci, s) in c.iter().zip(states.iter()) {
        for (acc, a) in recon.iter_mut().zip(s.iter()) {
            *acc += *ci * *a;
        }
    }
    let dev = v
        .iter()
        .zip(recon.iter())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0f64, f64::max);
    assert!(
        dev < 1e-9,
        "an arbitrary code state reassembles from its 2^h slices: dev {dev:e}"
    );
}

// ──────────────────────────── the accounting ────────────────────────────

#[test]
fn the_exponential_moves_into_the_node_count() {
    // The honest column-by-column: per node polynomial, node count
    // exponential, network total larger than monolithic. That last one
    // is the price, and it is stated rather than hidden.
    let code = ToricCode::new(3, 0).unwrap();
    let mut all = code.generators();
    for i in 0..2 {
        all.push(code.logical_x(i));
        all.push(code.logical_z(i));
    }
    let st = orthogonalize(&Volume::span(18, &all).unwrap());
    let plan = ShardPlan::new(&st);

    assert_eq!(plan.nodes(), 4, "2^h nodes");
    assert!(
        plan.per_node_bytes() < 200,
        "one node holds a tableau, not amplitudes: {} B",
        plan.per_node_bytes()
    );
    assert_eq!(plan.monolithic_bytes(), (1u128 << 18) * 16);
    assert!(
        plan.per_node_ratio() > 39_000.0,
        "per-node ratio {}",
        plan.per_node_ratio()
    );
    assert!(
        plan.network_bytes() < plan.monolithic_bytes(),
        "at k = 2 the network total still beats monolithic"
    );

    // The shape that matters: per-node cost does not move when the node
    // count grows. Same register, more logical axes.
    let a = ShardPlan::new(&Stitch::from_parts(30, 20, 5).unwrap());
    let b = ShardPlan::new(&Stitch::from_parts(30, 20, 10).unwrap());
    assert_eq!(b.nodes(), a.nodes() * 32);
    assert!(
        b.per_node_bytes() < 2 * a.per_node_bytes(),
        "node count × 32, per-node cost barely moves"
    );
}

#[test]
fn h_is_hidden_in_the_dimension() {
    // Two stitches of the same region dimension and different slice
    // counts: measuring the object's size says nothing about how many
    // pieces it is in.
    let a = Stitch::from_parts(20, 10, 3).unwrap();
    let b = Stitch::from_parts(20, 10, 7).unwrap();
    assert_eq!(a.region_dimension(), b.region_dimension());
    assert_ne!(a.hidden_bits(), b.hidden_bits());
    assert_eq!(a.slices(), 8);
    assert_eq!(b.slices(), 128);
    assert!(a.closes() && b.closes());
}

#[test]
fn selections_refuse_to_expand_an_exponent_by_default() {
    let st = Stitch::from_parts(60, 20, 20).unwrap();
    let err = st.selections().unwrap_err().to_string();
    assert!(err.contains("is an exponent"), "{err}");
}

#[test]
fn the_network_total_is_exponential_in_k_not_in_n() {
    // The claim that matters and the price that comes with it. A code
    // with r generators has already pulled the exponent from n down to
    // k = n − r; the shard plan spends that exponent as machines. So
    // the network total tracks 2^h, and widening the register at fixed
    // h costs only the tableau.
    let narrow = ShardPlan::new(&Stitch::from_parts(30, 20, 10).unwrap());
    let wide = ShardPlan::new(&Stitch::from_parts(40, 30, 10).unwrap());
    assert_eq!(narrow.nodes(), wide.nodes());
    assert!(
        wide.network_bytes() < 2 * narrow.network_bytes(),
        "ten more qubits at the same k costs a tableau, not a factor of 1024"
    );
    assert!(
        wide.monolithic_bytes() == narrow.monolithic_bytes() * 1024,
        "the monolithic register pays the full 2^10"
    );

    // And the price: each node carries a tableau to hold one
    // coefficient, so the network holds more than a bare 2^k amplitude
    // vector would. Stated, not hidden.
    let bare = narrow.nodes() * 16;
    assert!(
        narrow.network_bytes() > bare,
        "the tableau overhead per coefficient is real"
    );
    assert!(
        narrow.network_bytes() < 20 * bare,
        "and it is O(n^2/64), not a new exponential: {} vs {bare}",
        narrow.network_bytes()
    );
}
