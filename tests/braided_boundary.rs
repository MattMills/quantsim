//! Braided boundary encoding: the mutual encoding across a cut, the
//! faithful Artin action, the periodic-word combinatorics, the bracket
//! as measured geometric residue, and which realizations close.

use quantsim::braided::{
    bch_residual, bracket, cayley_ball, commutator_residual, distinct_orbits, fibonacci_generators,
    free_lie_dim, lyndon_count, lyndon_factorization, majorana_bilinears, majorana_generators,
    majorana_operators, mat_exp, mutual_encoding, necklace_count, orbit_closure, orbit_junction,
    projective_order, realized_rank, run_path, verify_relations, BraidWord, FreeWord, PeriodicPath,
};
use quantsim::prelude::*;

fn ghz(n: usize) -> Circuit {
    let mut c: Circuit = Circuit::new(n);
    c.h(0);
    for q in 1..n {
        c.cx(0, q);
    }
    c
}

fn max_gap(a: &GateMatrix<C64>, b: &GateMatrix<C64>) -> f64 {
    a.data()
        .iter()
        .zip(b.data())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0, f64::max)
}

// ── the free group and the Artin action ──────────────────────────────

#[test]
fn free_words_reduce_and_invert() {
    let w = FreeWord::from_letters(&[(0, 1), (1, 1), (1, -1), (0, -1), (2, 1)]);
    assert_eq!(w.letters(), &[(2, 1)]);
    assert_eq!(w.len(), 1);
    assert!(w.times(&w.inverse()).is_empty());
    assert!(FreeWord::identity().is_empty());
    // substitution is a homomorphism on the nose
    let images = vec![
        FreeWord::from_letters(&[(1, 1), (0, 1)]),
        FreeWord::generator(2),
        FreeWord::generator(0),
    ];
    let a = FreeWord::from_letters(&[(0, 1), (1, 1)]);
    let b = FreeWord::from_letters(&[(1, -1), (0, -1)]);
    assert_eq!(
        a.times(&b).substitute(&images),
        a.substitute(&images).times(&b.substitute(&images))
    );
}

#[test]
fn the_artin_action_realizes_the_braid_group() {
    for n in 3..=5 {
        // braid relations
        for i in 0..n - 2 {
            let a =
                BraidWord::from_letters(n, &[i as i32 + 1, i as i32 + 2, i as i32 + 1]).unwrap();
            let b =
                BraidWord::from_letters(n, &[i as i32 + 2, i as i32 + 1, i as i32 + 2]).unwrap();
            assert!(a.equals(&b), "braid relation at {i} on {n} strands");
        }
        // far commutation
        for i in 0..n - 1 {
            for j in i + 2..n - 1 {
                let a = BraidWord::from_letters(n, &[i as i32 + 1, j as i32 + 1]).unwrap();
                let b = BraidWord::from_letters(n, &[j as i32 + 1, i as i32 + 1]).unwrap();
                assert!(a.equals(&b), "far commutation ({i},{j}) on {n} strands");
            }
        }
        // adjacent generators do NOT commute — the action is not blind
        let a = BraidWord::from_letters(n, &[1, 2]).unwrap();
        let b = BraidWord::from_letters(n, &[2, 1]).unwrap();
        assert!(!a.equals(&b));
        // and a word times its inverse is the identity element
        let w = BraidWord::from_letters(n, &[1, 2, -1, 2, 2, -1]).unwrap();
        assert!(w.times(&w.inverse()).unwrap().is_trivial());
    }
}

#[test]
fn a_word_trivial_only_by_the_braid_relation_is_seen_as_trivial() {
    // σ1σ2σ1σ2⁻σ1⁻σ2⁻ is freely irreducible but equals the identity
    let w = BraidWord::from_letters(3, &[1, 2, 1, -2, -1, -2]).unwrap();
    assert_eq!(w.len(), 6, "the word must not free-reduce");
    assert!(w.is_trivial());
    assert_eq!(w.permutation(), vec![0, 1, 2]);
    // while the commutator is not trivial
    let c = BraidWord::from_letters(3, &[1, 2, -1, -2]).unwrap();
    assert_eq!(c.len(), 4);
    assert!(!c.is_trivial());
}

#[test]
fn braid_words_validate_their_strand_count() {
    assert!(BraidWord::identity(1).is_err());
    assert!(BraidWord::generator(3, 2, 1).is_err());
    assert!(BraidWord::from_letters(3, &[0]).is_err());
    let a = BraidWord::identity(3).unwrap();
    let b = BraidWord::identity(4).unwrap();
    assert!(a.times(&b).is_err());
}

#[test]
fn the_cayley_ball_grows_and_is_deduplicated() {
    let b3 = cayley_ball(3, 6).unwrap();
    assert_eq!(b3, vec![1, 5, 17, 47, 115, 263, 577]);
    let b4 = cayley_ball(4, 4).unwrap();
    assert_eq!(b4, vec![1, 7, 33, 131, 469]);
    // strictly growing, and always below the free-group word count
    for w in b3.windows(2) {
        assert!(w[1] > w[0]);
    }
    let free_words = |r: u32| 1 + 4 * ((3u64.pow(r) - 1) / 2);
    assert!(
        (b3[3] as u64) < free_words(3),
        "relations must collapse words"
    );
}

// ── periodic navigation words ────────────────────────────────────────

#[test]
fn necklace_and_lyndon_counts_match_enumeration() {
    for k in 2..=4u64 {
        for n in 1..=6u64 {
            let total = (k as usize).pow(n as u32);
            let paths: Vec<PeriodicPath> = (0..total)
                .map(|code| {
                    let mut w = Vec::with_capacity(n as usize);
                    let mut c = code;
                    for _ in 0..n {
                        w.push(c % k as usize);
                        c /= k as usize;
                    }
                    PeriodicPath::new(k as usize, w).unwrap()
                })
                .collect();
            assert_eq!(
                distinct_orbits(&paths) as u128,
                necklace_count(k, n),
                "necklaces k={k} n={n}"
            );
            let primitive_necklaces = paths
                .iter()
                .filter(|p| p.is_primitive() && p.necklace() == *p.period())
                .count();
            assert_eq!(
                primitive_necklaces as u128,
                lyndon_count(k, n),
                "lyndon k={k} n={n}"
            );
            // the Witt formula is the same number
            assert_eq!(free_lie_dim(k, n), lyndon_count(k, n));
        }
    }
}

#[test]
fn a_periodic_path_is_an_infinite_path_from_a_finite_period() {
    let p = PeriodicPath::new(3, vec![0, 2, 1]).unwrap();
    assert_eq!(p.prefix(8), vec![0, 2, 1, 0, 2, 1, 0, 2]);
    assert_eq!(p.step(3_000_000), p.step(0));
    assert!(p.is_primitive());
    assert!(!PeriodicPath::new(2, vec![0, 1, 0, 1])
        .unwrap()
        .is_primitive());
    // rotations are the same orbit, and only rotations are
    let a = PeriodicPath::new(3, vec![0, 2, 1]).unwrap();
    let b = PeriodicPath::new(3, vec![2, 1, 0]).unwrap();
    let c = PeriodicPath::new(3, vec![0, 1, 2]).unwrap();
    assert!(a.same_orbit(&b));
    assert!(!a.same_orbit(&c));
    assert_eq!(a.necklace(), b.necklace());
}

#[test]
fn periodic_paths_validate_their_alphabet() {
    assert!(PeriodicPath::new(1, vec![0]).is_err());
    assert!(PeriodicPath::new(2, vec![]).is_err());
    assert!(PeriodicPath::new(2, vec![0, 2]).is_err());
}

#[test]
fn duval_factorization_is_a_non_increasing_lyndon_factorization() {
    let mut rng = Prng::new(17);
    for _ in 0..200 {
        let n = 1 + (rng.next_u64() % 12) as usize;
        let word: Vec<usize> = (0..n).map(|_| (rng.next_u64() % 3) as usize).collect();
        let f = lyndon_factorization(&word);
        // concatenates back to the word
        let joined: Vec<usize> = f.iter().flatten().copied().collect();
        assert_eq!(joined, word);
        // each factor is Lyndon: strictly least among its own rotations
        for factor in &f {
            let p = PeriodicPath::new(3, factor.clone()).unwrap();
            assert!(
                p.is_primitive() && p.necklace() == *factor,
                "{factor:?} is not Lyndon"
            );
        }
        // and the sequence is non-increasing
        for w in f.windows(2) {
            assert!(w[0] >= w[1], "factors out of order: {:?}", f);
        }
    }
}

#[test]
fn an_orbit_junction_is_either_free_or_total_and_lexicographic_order_decides() {
    let mut rng = Prng::new(23);
    let mut seamless = 0usize;
    let mut merged = 0usize;
    let mut longest_interface = 0usize;
    for _ in 0..400 {
        let pa: Vec<usize> = (0..1 + rng.next_u64() % 4)
            .map(|_| (rng.next_u64() % 3) as usize)
            .collect();
        let pb: Vec<usize> = (0..1 + rng.next_u64() % 4)
            .map(|_| (rng.next_u64() % 3) as usize)
            .collect();
        let a = PeriodicPath::new(3, pa.clone()).unwrap();
        let b = PeriodicPath::new(3, pb.clone()).unwrap();
        let (la, lb) = (24usize, 24usize);
        let j = orbit_junction(&a, la, &b, lb).unwrap();

        // the factorization really is of the joined word, and is legal
        let joined: Vec<usize> = j.factors.iter().flatten().copied().collect();
        assert_eq!(joined, j.word);
        assert_eq!(j.word.len(), la + lb);
        for w in j.factors.windows(2) {
            assert!(w[0] >= w[1], "factors out of order for {pa:?}|{pb:?}");
        }
        assert_eq!(
            j.kept_from_first + j.interface.len() + j.kept_from_second,
            j.factors.len()
        );

        // the regime is decided by comparing the two boundary factors
        let fa = lyndon_factorization(&a.prefix(la));
        let fb = lyndon_factorization(&b.prefix(lb));
        let concatenates = fa.last().unwrap() >= fb.first().unwrap();
        assert_eq!(
            j.seamless(),
            concatenates,
            "{pa:?}|{pb:?}: seamless={} but last(a) >= first(b) is {concatenates}",
            j.seamless()
        );
        // and when it is seamless, both sides kept everything they had
        if j.seamless() {
            seamless += 1;
            assert_eq!(j.kept_from_first, fa.len());
            assert_eq!(j.kept_from_second, fb.len());
            assert_eq!(j.interface_len, 0);
        } else {
            merged += 1;
            assert!(j.interface_len > 0);
            longest_interface = longest_interface.max(j.interface_len);
        }
    }
    // both regimes actually occur in the sample
    assert!(
        seamless > 20 && merged > 20,
        "{seamless} free, {merged} merged"
    );
    // and the merge is genuinely unbounded, not a local repair
    assert!(
        longest_interface > 12,
        "longest interface was only {longest_interface} letters"
    );

    // Even joining an orbit to *itself* is not automatically free, and
    // what decides it is which rotation the period happens to be written
    // in: seamless exactly when the period is its own canonical rotation
    // (its necklace), which is a labelling choice with no physical
    // content whatsoever.
    for period in [
        vec![0usize, 1, 1],
        vec![1, 1, 0],
        vec![0, 0, 1],
        vec![1, 0],
        vec![0, 1, 0, 1],
        vec![1, 0, 1, 0],
        vec![0, 1, 1, 0, 1],
    ] {
        let p = PeriodicPath::new(2, period.clone()).unwrap();
        // a whole number of periods: a truncated cut is a different seam
        let len = 6 * period.len();
        let j = orbit_junction(&p, len, &p, len).unwrap();
        assert_eq!(
            j.seamless(),
            period == p.necklace(),
            "self-join of {period:?}: seamless={} but canonical={}",
            j.seamless(),
            period == p.necklace()
        );
    }

    // and cutting mid-period is itself a different seam: the same orbit
    // joined to itself at a truncated length stops being free.
    let canonical = PeriodicPath::new(2, vec![0, 1, 1]).unwrap();
    assert!(orbit_junction(&canonical, 18, &canonical, 18)
        .unwrap()
        .seamless());
    assert!(!orbit_junction(&canonical, 20, &canonical, 20)
        .unwrap()
        .seamless());

    // the pinned asymmetry: swapping the two orbits changes the regime,
    // so a junction is a property of the *ordered* pair
    let hi = PeriodicPath::new(2, vec![0, 1]).unwrap();
    let lo = PeriodicPath::new(2, vec![0, 0, 1]).unwrap();
    let forward = orbit_junction(&hi, 12, &lo, 12).unwrap();
    let backward = orbit_junction(&lo, 12, &hi, 12).unwrap();
    assert!(forward.seamless(), "[0,1] then [0,0,1] should be free");
    assert!(!backward.seamless(), "[0,0,1] then [0,1] should merge");

    // and a merge really can swallow everything past the seam
    let big = PeriodicPath::new(2, vec![1, 1, 0]).unwrap();
    let small = PeriodicPath::new(2, vec![0, 1]).unwrap();
    let j = orbit_junction(&big, 12, &small, 12).unwrap();
    assert!(!j.seamless());
    assert_eq!(j.interface_len, 13);
    assert_eq!(j.kept_from_second, 0, "the second orbit kept nothing");

    // orbits over different alphabets have no shared word
    let q = PeriodicPath::new(3, vec![0, 2]).unwrap();
    assert!(orbit_junction(&lo, 4, &q, 4).is_err());
}

// ── the mutual boundary encoding ─────────────────────────────────────

#[test]
fn each_boundary_carries_the_other_side_exactly() {
    let sim: Simulator = Simulator::new();
    for (label, circuit, cut) in [
        ("ghz", ghz(8), vec![0, 1, 2, 3]),
        ("ghz-1|7", ghz(8), vec![0]),
        ("product", Circuit::new(6), vec![0, 2, 4]),
        (
            "random",
            random_registry_circuit(&GateRegistry::standard(), 8, 60, 9),
            vec![0, 1, 2, 3],
        ),
    ] {
        let state = sim.run(&circuit).unwrap();
        let e = mutual_encoding(state.as_ref(), &cut).unwrap();
        assert!(
            e.spectrum_deviation < 1e-13,
            "{label}: ρ_A and ρ_B spectra differ by {:.3e}",
            e.spectrum_deviation
        );
        assert!(
            e.purification_deviation < 1e-13,
            "{label}: purification of ρ_A differs by {:.3e}",
            e.purification_deviation
        );
        assert!(
            e.purified_spectrum_deviation < 1e-13,
            "{label}: purified B spectrum differs by {:.3e}",
            e.purified_spectrum_deviation
        );
        assert!(
            e.reconstruction_deviation < 1e-13,
            "{label}: rebuild differs by {:.3e}",
            e.reconstruction_deviation
        );
        assert!(e.rank >= 1 && e.rank <= e.max_rank());
        // the spectrum is a probability distribution on the cut
        let total: f64 = e.spectrum.iter().map(|s| s * s).sum();
        assert!((total - 1.0).abs() < 1e-12, "{label}: weight {total}");
    }
}

#[test]
fn the_storage_claim_is_reported_not_assumed() {
    let sim: Simulator = Simulator::new();
    // GHZ is rank 2 and the encoding pays
    let g = mutual_encoding(sim.run(&ghz(8)).unwrap().as_ref(), &[0, 1, 2, 3]).unwrap();
    assert_eq!(g.rank, 2);
    assert!(g.compression() > 3.0, "ghz compression {}", g.compression());

    // a random state is full rank and the encoding costs more than dense
    let r = mutual_encoding(
        sim.run(&random_registry_circuit(
            &GateRegistry::standard(),
            8,
            60,
            9,
        ))
        .unwrap()
        .as_ref(),
        &[0, 1, 2, 3],
    )
    .unwrap();
    assert_eq!(r.rank, 16);
    assert_eq!(r.rank, r.max_rank());
    assert!(
        r.compression() < 1.0,
        "a full-rank cut must not claim a saving: {}",
        r.compression()
    );
}

#[test]
fn a_cut_must_be_a_real_cut() {
    let sim: Simulator = Simulator::new();
    let state = sim.run(&ghz(4)).unwrap();
    assert!(mutual_encoding(state.as_ref(), &[]).is_err());
    assert!(mutual_encoding(state.as_ref(), &[0, 1, 2, 3]).is_err());
    assert!(mutual_encoding(state.as_ref(), &[0, 0]).is_err());
    assert!(mutual_encoding(state.as_ref(), &[9]).is_err());
}

// ── the unitary realizations ─────────────────────────────────────────

#[test]
fn majorana_operators_satisfy_the_clifford_algebra() {
    for m in 1..=3 {
        let g = majorana_operators(m).unwrap();
        assert_eq!(g.len(), 2 * m);
        let d = 1usize << m;
        let id = GateMatrix::<C64>::identity(d).unwrap();
        for a in 0..g.len() {
            for b in 0..g.len() {
                let mut anti = g[a].matmul(&g[b]);
                let other = g[b].matmul(&g[a]);
                for r in 0..d {
                    for c in 0..d {
                        anti.set(r, c, anti.get(r, c) + other.get(r, c));
                    }
                }
                let want = if a == b {
                    let mut t = id.clone();
                    for r in 0..d {
                        t.set(r, r, C64::new(2.0, 0.0));
                    }
                    t
                } else {
                    GateMatrix::<C64>::zeros(d).unwrap()
                };
                assert!(
                    max_gap(&anti, &want) < 1e-14,
                    "{{γ_{a}, γ_{b}}} wrong on {m} qubits"
                );
            }
        }
    }
    assert!(majorana_operators(0).is_err());
    assert!(majorana_operators(99).is_err());
}

#[test]
fn both_realizations_satisfy_the_braid_relations_exactly() {
    for strands in [4usize, 6, 8] {
        let g = majorana_generators(strands).unwrap();
        assert_eq!(g.len(), strands - 1);
        let r = verify_relations(&g).unwrap();
        assert!(
            r.hold(1e-13),
            "majorana {strands}: unitarity {:.2e} braid {:.2e} far {:.2e}",
            r.unitarity,
            r.braid,
            r.far_commutation
        );
    }
    let f = fibonacci_generators().unwrap();
    assert_eq!(f.len(), 2);
    let r = verify_relations(&f).unwrap();
    assert!(
        r.hold(1e-13),
        "fibonacci: unitarity {:.2e} braid {:.2e}",
        r.unitarity,
        r.braid
    );
    // the odd/oversized strand counts are refused, not silently rounded
    assert!(majorana_generators(5).is_err());
    assert!(majorana_generators(0).is_err());
    assert!(majorana_generators(100).is_err());
    assert!(verify_relations(&[]).is_err());
}

#[test]
fn the_generators_are_the_exponentials_of_the_bilinears() {
    for strands in [4usize, 6] {
        let bil = majorana_bilinears(strands).unwrap();
        let gen = majorana_generators(strands).unwrap();
        assert_eq!(bil.len(), gen.len());
        for (b, g) in bil.iter().zip(&gen) {
            let d = b.dim();
            let mut scaled = GateMatrix::<C64>::zeros(d).unwrap();
            for r in 0..d {
                for c in 0..d {
                    scaled.set(
                        r,
                        c,
                        b.get(r, c) * C64::new(std::f64::consts::FRAC_PI_4, 0.0),
                    );
                }
            }
            let e = mat_exp(&scaled).unwrap();
            assert!(
                max_gap(&e, g) < 1e-12,
                "exp(π/4 γγ) ≠ σ on {strands} strands: {:.3e}",
                max_gap(&e, g)
            );
        }
    }
}

#[test]
fn the_ising_image_is_finite_and_the_fibonacci_one_does_not_close() {
    let m4 = orbit_closure(&majorana_generators(4).unwrap(), 20, 200_000).unwrap();
    assert_eq!(m4.closed_at, Some(192));
    assert_eq!(m4.closure_radius, Some(7));
    assert!(m4.finite());
    assert!(!m4.hit_cap);

    let m6 = orbit_closure(&majorana_generators(6).unwrap(), 25, 200_000).unwrap();
    assert_eq!(m6.closed_at, Some(23_040));
    assert_eq!(m6.closure_radius, Some(16));

    let fib = orbit_closure(&fibonacci_generators().unwrap(), 12, 200_000).unwrap();
    assert!(!fib.finite(), "fibonacci must not report closure");
    assert!(fib.closure_radius.is_none());
    assert!(
        fib.growth > 1.5,
        "fibonacci ball is still growing: {}",
        fib.growth
    );
    // and the ball is strictly increasing throughout
    for w in fib.ball_sizes.windows(2) {
        assert!(w[1] > w[0]);
    }
    assert!(orbit_closure(&[], 3, 10).is_err());
}

#[test]
fn projective_order_is_exact_where_it_answers() {
    let f = fibonacci_generators().unwrap();
    // (σ1σ2)^3 is central in B_3, so it is a scalar in an irrep
    let g = f[1].matmul(&f[0]);
    assert_eq!(projective_order(&g, 1000, 1e-9), Some(3));
    // measured per-path orders in the Fibonacci realization
    let cases: [(&[usize], Option<usize>); 5] = [
        (&[0, 1], Some(3)),
        (&[0, 0, 1], Some(2)),
        (&[0, 1, 1, 0], Some(5)),
        (&[0, 1, 0, 1, 1], Some(10)),
        (&[0, 1, 1, 1, 0, 0, 1], None),
    ];
    for (period, want) in cases {
        let mut m = GateMatrix::<C64>::identity(2).unwrap();
        for &l in period {
            m = f[l].matmul(&m);
        }
        assert_eq!(
            projective_order(&m, 20_000, 1e-9),
            want,
            "period {period:?}"
        );
    }
    // the identity has order 1
    assert_eq!(
        projective_order(&GateMatrix::<C64>::identity(4).unwrap(), 10, 1e-12),
        Some(1)
    );
}

// ── the bracket ──────────────────────────────────────────────────────

#[test]
fn mat_exp_agrees_with_the_braid_generators_and_is_unitary() {
    let bil = majorana_bilinears(4).unwrap();
    for b in &bil {
        let d = b.dim();
        let mut s = GateMatrix::<C64>::zeros(d).unwrap();
        for r in 0..d {
            for c in 0..d {
                s.set(r, c, b.get(r, c) * C64::new(0.37, 0.0));
            }
        }
        let e = mat_exp(&s).unwrap();
        assert!(e.unitarity_deviation() < 1e-12, "exp of anti-Hermitian");
    }
    // exp(0) = I
    let z = GateMatrix::<C64>::zeros(4).unwrap();
    assert!(
        max_gap(
            &mat_exp(&z).unwrap(),
            &GateMatrix::<C64>::identity(4).unwrap()
        ) < 1e-15
    );
}

#[test]
fn the_bracket_is_the_leading_geometric_residue() {
    let mk = |x: f64, y: f64, z: f64| {
        GateMatrix::<C64>::from_vec(
            2,
            vec![
                C64::new(0.0, z),
                C64::new(y, x),
                C64::new(-y, x),
                C64::new(0.0, -z),
            ],
        )
        .unwrap()
    };
    let a = mk(1.0, 0.3, 0.0);
    let b = mk(0.0, 0.7, 1.0);
    let eps = [0.2, 0.1, 0.05, 0.025, 0.0125];

    let cr = commutator_residual(&a, &b, &eps).unwrap();
    assert!(
        (cr.order - 3.0).abs() < 0.15,
        "commutator residual order {} (want 3)",
        cr.order
    );
    // residuals shrink monotonically with ε
    for w in cr.residuals.windows(2) {
        assert!(w[1] < w[0]);
    }

    let br = bch_residual(&a, &b, &eps).unwrap();
    assert!(
        (br.order - 4.0).abs() < 0.15,
        "BCH residual order {} (want 4)",
        br.order
    );
    // and BCH-3 is a strictly better approximation than the bare bracket
    for i in 0..eps.len() {
        assert!(br.residuals[i] < cr.residuals[i]);
    }
}

#[test]
fn the_su2_pair_has_a_vanishing_fourth_order_term() {
    // For A = iX, B = iZ the order-4 BCH term [B,[A,[A,B]]] is exactly
    // zero, so the measured exponent must be 5, not 4. The fit catching
    // that is what makes it a measurement rather than a formality.
    let a = GateMatrix::<C64>::from_vec(
        2,
        vec![
            C64::new(0.0, 0.0),
            C64::new(0.0, 1.0),
            C64::new(0.0, 1.0),
            C64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let b = GateMatrix::<C64>::from_vec(
        2,
        vec![
            C64::new(0.0, 1.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, -1.0),
        ],
    )
    .unwrap();
    let inner = bracket(&b, &bracket(&a, &bracket(&a, &b).unwrap()).unwrap()).unwrap();
    assert!(
        inner.data().iter().all(|z| z.norm() < 1e-14),
        "[B,[A,[A,B]]] is not zero"
    );
    let br = bch_residual(&a, &b, &[0.2, 0.1, 0.05, 0.025, 0.0125]).unwrap();
    assert!(
        (br.order - 5.0).abs() < 0.15,
        "degenerate BCH order {} (want 5)",
        br.order
    );
}

#[test]
fn the_realization_collapses_the_free_algebra_onto_its_own_lie_algebra() {
    // 6 strands: 5 bilinears generating so(6), dimension 15.
    let bil = majorana_bilinears(6).unwrap();
    let ra = realized_rank(&bil, 5).unwrap();
    assert_eq!(ra.free_cumulative, vec![5, 15, 55, 205, 829]);
    assert_eq!(ra.realized, vec![5, 9, 12, 14, 15]);
    assert_eq!(*ra.realized.last().unwrap(), 15, "so(6) has dimension 15");
    assert_eq!(ra.ambient, 2 * 8 * 8);
    // it never exceeds the free algebra, and past degree 1 it is strictly below
    for (r, f) in ra.realized.iter().zip(&ra.free_cumulative) {
        assert!(*r as u128 <= *f);
    }
    assert!((ra.realized[2] as u128) < ra.free_cumulative[2]);

    // 4 strands: so(4), dimension 6, saturated by degree 3
    let ra4 = realized_rank(&majorana_bilinears(4).unwrap(), 5).unwrap();
    assert_eq!(ra4.realized, vec![3, 5, 6, 6, 6]);
    assert_eq!(ra4.saturated_at, Some(3));
    assert!(realized_rank(&[], 3).is_err());
}

// ── the ledger ───────────────────────────────────────────────────────

#[test]
fn every_path_ascends_exactly() {
    let fib = fibonacci_generators().unwrap();
    for (gens, period, alphabet) in [
        (majorana_generators(4).unwrap(), vec![0usize, 1, 2, 1], 3),
        (majorana_generators(6).unwrap(), vec![0usize, 2, 4, 1], 5),
        (fib.clone(), vec![0usize, 1, 1, 1, 0, 0, 1], 2),
    ] {
        let path = PeriodicPath::new(alphabet, period.clone()).unwrap();
        let qubits = gens[0].dim().trailing_zeros() as usize;
        let cut: Vec<usize> = if qubits >= 2 { vec![0] } else { vec![] };
        let l = run_path(&gens, &path, 200, &cut).unwrap();
        assert_eq!(l.steps.len(), 200);
        assert!(
            l.ascent_deviation < 1e-11,
            "period {period:?} ascended to {:.3e}",
            l.ascent_deviation
        );
        // the journal records every step's letter, in path order
        for (k, s) in l.steps.iter().enumerate() {
            assert_eq!(s.letter, path.step(k));
            assert_eq!(s.depth, k + 1);
            assert!(s.rank >= 1);
        }
        assert_eq!(l.path_bytes, 200);
    }
}

#[test]
fn the_cycle_order_decides_closure_where_a_state_count_cannot() {
    let fib = fibonacci_generators().unwrap();
    for (period, want) in [
        (vec![0usize, 1], Some(3)),
        (vec![0usize, 1, 1, 0], Some(5)),
        (vec![0usize, 1, 0, 1, 1], Some(10)),
        (vec![0usize, 1, 1, 1, 0, 0, 1], None),
    ] {
        let path = PeriodicPath::new(2, period.clone()).unwrap();
        let l = run_path(&fib, &path, 400, &[]).unwrap();
        assert_eq!(l.cycle_order, want, "period {period:?}");
        // where it closes, the state count is bounded by the order + the
        // start; where it does not, every step is a new state
        match want {
            Some(k) => assert!(
                l.distinct_states <= k * period.len() + 1,
                "period {period:?}: {} states for order {k}",
                l.distinct_states
            ),
            // No closure: the count keeps growing. It does not reach the
            // full 401, because at the key's rounding two nearby states
            // collided — which is exactly why the count cannot be the
            // closure test and the order is.
            None => assert!(
                l.distinct_states >= 390,
                "non-closing orbit only reached {} states",
                l.distinct_states
            ),
        }
    }
    // in the Ising realization every path closes, because the group does
    let gens = majorana_generators(4).unwrap();
    for period in [vec![0usize, 1, 2, 1], vec![0, 1, 1, 2, 0], vec![2, 1, 0]] {
        let path = PeriodicPath::new(3, period.clone()).unwrap();
        let l = run_path(&gens, &path, 200, &[0]).unwrap();
        assert!(
            l.cycle_order.is_some(),
            "period {period:?} did not close in a finite group"
        );
    }
}

#[test]
fn run_path_validates_its_inputs() {
    let gens = fibonacci_generators().unwrap();
    let too_wide = PeriodicPath::new(4, vec![0, 3]).unwrap();
    assert!(run_path(&gens, &too_wide, 10, &[]).is_err());
    assert!(run_path(&[], &PeriodicPath::new(2, vec![0]).unwrap(), 10, &[]).is_err());
}
