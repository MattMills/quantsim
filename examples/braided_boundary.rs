//! Braided boundary encoding, measured live: the mutual encoding across
//! a cut, the Artin action that makes path identity decidable, the
//! Cayley graph the periodic word navigates, the bracket as the measured
//! geometric residue, and which realizations let the path *be* the
//! storage.
//!
//! Run with `cargo run --release --example braided_boundary`.

use quantsim::braided::{
    bch_residual, cayley_ball, commutator_residual, distinct_orbits, fibonacci_generators,
    free_lie_dim, lyndon_count, lyndon_factorization, majorana_bilinears, majorana_generators,
    mutual_encoding, necklace_count, orbit_closure, orbit_junction, realized_rank, run_path,
    verify_relations, BraidWord, PeriodicPath,
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

fn main() -> Result<()> {
    println!("== the mutual boundary encoding ==\n");
    let sim: Simulator = Simulator::new();
    println!("  state          |A|  rank  max  spec dev   purif dev  purif spec  rebuild   stored/dense");
    for (label, circuit, cut) in [
        ("GHZ-8", ghz(8), vec![0, 1, 2, 3]),
        ("GHZ-8 (1|7)", ghz(8), vec![0]),
        ("product-8", Circuit::new(8), vec![0, 1, 2, 3]),
        (
            "random-8",
            random_registry_circuit(&GateRegistry::standard(), 8, 60, 9),
            vec![0, 1, 2, 3],
        ),
    ] {
        let state = sim.run(&circuit)?;
        let e = mutual_encoding(state.as_ref(), &cut)?;
        println!(
            "  {label:<14} {:3}  {:4}  {:3}  {:9.2e}  {:9.2e}  {:9.2e}  {:7.1e}  {:6}/{:<6}",
            e.region.len(),
            e.rank,
            e.max_rank(),
            e.spectrum_deviation,
            e.purification_deviation,
            e.purified_spectrum_deviation,
            e.reconstruction_deviation,
            e.stored_entries,
            e.dense_entries
        );
    }
    println!("\n  ρ_A and ρ_B are built from different-sized matrices and their");
    println!("  spectra agree to machine precision; ρ_A alone purifies to a state");
    println!("  whose complement is spectrally the true B. The storage column is");
    println!("  the honest part: GHZ is rank 2 and cheap, a random state is full");
    println!("  rank and the boundary encoding buys nothing at all.");

    println!("\n== the Artin action: path identity is decidable ==\n");
    let a = BraidWord::from_letters(4, &[1, 2, 1])?;
    let b = BraidWord::from_letters(4, &[2, 1, 2])?;
    let c = BraidWord::from_letters(4, &[1, 3])?;
    let d = BraidWord::from_letters(4, &[3, 1])?;
    println!("  σ1σ2σ1 == σ2σ1σ2 : {}", a.equals(&b));
    println!("  σ1σ3   == σ3σ1   : {}  (far commutation)", c.equals(&d));
    println!("  σ1σ2   == σ2σ1   : {}",
        BraidWord::from_letters(4, &[1, 2])?.equals(&BraidWord::from_letters(4, &[2, 1])?));
    let w = BraidWord::from_letters(3, &[1, 2, 1, -2, -1, -2])?;
    println!(
        "  σ1σ2σ1σ2⁻σ1⁻σ2⁻ is trivial: {}  — length 6, freely irreducible,\n    and trivial only through the braid relation. permutation {:?}",
        w.is_trivial(),
        w.permutation()
    );
    let nw = BraidWord::from_letters(3, &[1, 2, -1, -2])?;
    println!(
        "  σ1σ2σ1⁻σ2⁻ is trivial: {}   (the commutator is not)",
        nw.is_trivial()
    );
    println!("\n  x_1 under σ1 is {:?} — one boundary conjugated by the other,",
        BraidWord::generator(3, 0, 1)?.artin_images()[0].letters());
    println!("  which is the mutual encoding written as a substitution.");

    println!("\n== the infinite graph the path navigates ==\n");
    for strands in 3..=4 {
        let radius = if strands == 3 { 6 } else { 5 };
        let ball = cayley_ball(strands, radius)?;
        let ratios: Vec<String> = ball
            .windows(2)
            .map(|w| format!("{:.2}", w[1] as f64 / w[0] as f64))
            .collect();
        println!("  B_{strands} ball sizes {ball:?}");
        println!("        growth   {}", ratios.join("  "));
    }

    println!("\n== periodic words: how many distinct orbits ==\n");
    println!("  alphabet  len   necklaces  Lyndon (= free Lie dim)   enumerated");
    for k in [2u64, 3] {
        for n in [4u64, 6] {
            let total = (k as usize).pow(n as u32);
            let paths: Vec<PeriodicPath> = (0..total)
                .map(|code| {
                    let mut w = Vec::new();
                    let mut c = code;
                    for _ in 0..n {
                        w.push(c % k as usize);
                        c /= k as usize;
                    }
                    PeriodicPath::new(k as usize, w).unwrap()
                })
                .collect();
            println!(
                "  {k:8}  {n:3}   {:9}  {:22}   {:10}",
                necklace_count(k, n),
                lyndon_count(k, n),
                distinct_orbits(&paths)
            );
            assert_eq!(lyndon_count(k, n), free_lie_dim(k, n));
        }
    }
    let word = [0usize, 1, 1, 0, 1, 0, 0, 1];
    println!(
        "\n  Duval factorization of {word:?}:\n    {:?}",
        lyndon_factorization(&word)
    );

    println!("\n== what happens where two orbits meet ==\n");
    println!("  first    second   factors  kept L  interface        kept R");
    for (pa, pb) in [
        (vec![0usize, 1], vec![0usize, 1]),
        (vec![0usize, 1], vec![0usize, 0, 1]),
        (vec![0usize, 1, 1], vec![0usize, 0, 0, 1]),
        (vec![1usize, 1, 0], vec![0usize, 1]),
    ] {
        let a = PeriodicPath::new(2, pa.clone())?;
        let b = PeriodicPath::new(2, pb.clone())?;
        let j = orbit_junction(&a, 12, &b, 12)?;
        println!(
            "  {:<8} {:<8} {:7}  {:6}  {:<15}  {:5}",
            format!("{pa:?}"),
            format!("{pb:?}"),
            j.factors.len(),
            j.kept_from_first,
            format!("{:?} ({})", j.interface, j.interface_len),
            j.kept_from_second
        );
    }
    println!("\n  A junction needs no machinery of its own — Duval's factorization is");
    println!("  unique and non-increasing, so whatever the seam does is recorded in");
    println!("  the same structure the orbits are. But it is not a boundary *layer*:");
    println!("  it is two sharp regimes. When the first orbit's last factor is ≥ the");
    println!("  second's first factor the two factorizations already concatenate");
    println!("  legally, so the join costs exactly nothing. Otherwise the smaller");
    println!("  tail absorbs the larger head and can swallow everything past the");
    println!("  seam — the last row keeps 0 factors from its second orbit.");
    println!("  A junction is therefore free or total, never partial; it is a");
    println!("  property of the *ordered* pair; and it is settled by comparing");
    println!("  words, with nothing about the geometry entering. It even depends on");
    println!("  where the periods were cut — the same orbit joined to itself is");
    println!("  free at 18 letters and merges at 20.");

    println!("\n== the realizations, and their relations on the actual matrices ==\n");
    println!("  realization        dim  unitarity   braid      far comm");
    for strands in [4usize, 6] {
        let g = majorana_generators(strands)?;
        let r = verify_relations(&g)?;
        println!(
            "  Majorana/{strands}-strand   {:3}  {:9.2e}  {:9.2e}  {:9.2e}",
            g[0].dim(),
            r.unitarity,
            r.braid,
            r.far_commutation
        );
    }
    let fib = fibonacci_generators()?;
    let rf = verify_relations(&fib)?;
    println!(
        "  Fibonacci/B_3      {:3}  {:9.2e}  {:9.2e}  {:9.2e}",
        fib[0].dim(),
        rf.unitarity,
        rf.braid,
        rf.far_commutation
    );

    println!("\n== does the path close? ==\n");
    for (label, gens, radius, cap) in [
        ("Majorana 4-strand", majorana_generators(4)?, 20usize, 200_000usize),
        ("Majorana 6-strand", majorana_generators(6)?, 20, 200_000),
        ("Fibonacci B_3", fib.clone(), 12, 200_000),
    ] {
        let o = orbit_closure(&gens, radius, cap)?;
        println!(
            "  {label:<18} balls {:?}",
            &o.ball_sizes[..o.ball_sizes.len().min(9)]
        );
        match o.closed_at {
            Some(n) => println!(
                "  {:<18} CLOSED at radius {} — {n} projective elements, so an\n{:<20}arbitrarily long path is stored in O(1).",
                "", o.closure_radius.unwrap(), ""
            ),
            None => println!(
                "  {:<18} did NOT close within radius {radius} (growth {:.2}, cap hit: {}).\n{:<20}That is evidence, not proof — the report never says \"infinite\".",
                "", o.growth, o.hit_cap, ""
            ),
        }
    }

    println!("\n== the bracket is the measured geometric residue ==\n");
    let eps = [0.2, 0.1, 0.05, 0.025, 0.0125];
    let generic = generic_pair();
    let su2 = su2_generators();
    let cr = commutator_residual(&generic[0], &generic[1], &eps)?;
    let br = bch_residual(&generic[0], &generic[1], &eps)?;
    let br_su2 = bch_residual(&su2[0], &su2[1], &eps)?;
    println!("    ε      commutator vs exp(ε²[A,B])   BCH-3 (generic)   BCH-3 (iX, iZ)");
    for (i, e) in eps.iter().enumerate() {
        println!(
            "  {e:7.4}   {:24.3e}   {:15.3e}   {:14.3e}",
            cr.residuals[i], br.residuals[i], br_su2.residuals[i]
        );
    }
    println!(
        "\n  fitted orders: commutator {:.2} (expected 3), BCH {:.2} (expected 4)",
        cr.order, br.order
    );
    println!(
        "  the (iX, iZ) pair fits {:.2}, not 4 — its order-4 BCH term",
        br_su2.order
    );
    println!("  [B,[A,[A,B]]] vanishes identically, and the fit sees that.");

    println!("\n== the free algebra, and how far a realization carries it ==\n");
    let bil = majorana_bilinears(6)?;
    let ra = realized_rank(&bil, 5)?;
    println!("  5 infinitesimal folds γ_iγ_{{i+1}} on 6 strands (σ_i = exp(π/4 · γ_iγ_{{i+1}}))\n");
    println!("  degree            1  2  3  4  5");
    println!(
        "  free (new dims)  {}",
        ra.free_dims
            .iter()
            .map(|d| format!("{d:2} "))
            .collect::<String>()
    );
    println!(
        "  free cumulative  {}",
        ra.free_cumulative
            .iter()
            .map(|d| format!("{d:2} "))
            .collect::<String>()
    );
    println!(
        "  realized rank    {}",
        ra.realized
            .iter()
            .map(|d| format!("{d:2} "))
            .collect::<String>()
    );
    println!(
        "\n  the realization stops at {} — so(6), whose dimension is 15 — inside an",
        ra.realized.last().unwrap()
    );
    println!(
        "  ambient real dimension of {}. The free path algebra is infinite and no",
        ra.ambient
    );
    println!("  realization is; the collapse is where the computation's bound lives.");

    println!("\n== the ledger: descend a periodic path, then ascend it ==\n");
    println!("  realization        period            cycle order  distinct  ascent dev");
    for (label, gens, period) in [
        ("Majorana 4-strand", majorana_generators(4)?, vec![0usize, 1, 2, 1]),
        ("Majorana 6-strand", majorana_generators(6)?, vec![0usize, 2, 4, 1]),
        ("Fibonacci B_3", fib.clone(), vec![0usize, 1]),
        ("Fibonacci B_3", fib.clone(), vec![0usize, 1, 1, 0]),
        ("Fibonacci B_3", fib.clone(), vec![0usize, 1, 0, 1, 1]),
        ("Fibonacci B_3", fib.clone(), vec![0usize, 1, 1, 1, 0, 0, 1]),
    ] {
        let alphabet = period.iter().max().unwrap() + 1;
        let path = PeriodicPath::new(alphabet.max(2), period.clone())?;
        let qubits = gens[0].dim().trailing_zeros() as usize;
        let cut: Vec<usize> = if qubits >= 2 { vec![0] } else { vec![] };
        let l = run_path(&gens, &path, 400, &cut)?;
        println!(
            "  {label:<18} {:<16}  {:>11}  {:8}  {:10.2e}",
            format!("{:?}", path.period()),
            l.cycle_order
                .map(|d| d.to_string())
                .unwrap_or_else(|| "does not".into()),
            l.distinct_states,
            l.ascent_deviation,
        );
    }
    println!("\n  The cycle order is the decisive test and the state count is not:");
    println!("  a count at finite tolerance saturates for a dense orbit too, so it");
    println!("  can only ever agree with closure, never establish it. 0111001 is");
    println!("  the case in point: it does not close, its count keeps climbing —");
    println!("  and it still lands one short of 401, because two nearby states");
    println!("  collided at the key's rounding.");
    println!("\n  Every path ascends exactly: nothing along the descent was erased,");
    println!("  because the recursion never flattened the geometry. Whether the");
    println!("  *storage* is O(1) is a separate, measured question, and it is");
    println!("  answered by whether the orbit closes — not by the recursion.");

    Ok(())
}

/// A generic anti-Hermitian pair — no vanishing BCH term.
fn generic_pair() -> Vec<GateMatrix<C64>> {
    // i(v·σ) for two non-orthogonal, non-axis-aligned directions
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
    vec![mk(1.0, 0.3, 0.0), mk(0.0, 0.7, 1.0)]
}

/// Two anti-Hermitian su(2) generators to measure brackets against.
fn su2_generators() -> Vec<GateMatrix<C64>> {
    let x = GateMatrix::<C64>::from_vec(
        2,
        vec![
            C64::new(0.0, 0.0),
            C64::new(0.0, 1.0),
            C64::new(0.0, 1.0),
            C64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let z = GateMatrix::<C64>::from_vec(
        2,
        vec![
            C64::new(0.0, 1.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, -1.0),
        ],
    )
    .unwrap();
    vec![x, z]
}
