//! A falsification harness for magic relocation.
//!
//! This file was written to test whether relocating magic could make a
//! universal family cheap — which, if it held, would mean `BQP = BPP`.
//! It found instead that the quantity being measured is not the one the
//! claim needs, and the tests below now pin that down so it cannot be
//! re-asserted.
//!
//! **What `h*` is.** [`PathSum::internal_vars`] counts active variables
//! appearing in no output form: the variables still summed over *after
//! the output index is fixed*. So `2^{h*}` is the cost of ONE amplitude
//! `⟨z|U|x⟩`. That is a real and useful quantity, and it is the only
//! thing `h*` measures.
//!
//! **What `h*` is not.** It is not a Clifford certificate. A single `T`
//! has `h* = 0`; so does `CCZ`; so does any diagonal circuit, including
//! one carrying 220 `CCZ` gates on distinct triples. All are
//! non-Clifford, and [`h_star_zero_does_not_certify_clifford`] holds an
//! independent dense oracle against them. Measured the other way,
//! every Clifford circuit in the corpus *did* reduce to zero, so the
//! classes sit as `{Clifford} ⊊ {h* = 0}` — a strict superset, readable
//! in one direction only.
//!
//! **Why the complexity reading fails twice.** First, `h*` does grow on
//! random Clifford+T — measured at `2.19^T` for one amplitude — so
//! nothing collapses in the first place. Second, and independently:
//! cheap amplitudes are not cheap sampling. The diagonal core has
//! `h* = 0` at any depth, yet putting it between Hadamard layers — which
//! is just measuring it in the X basis — gives IQP, believed hard, and
//! `h*` correctly rises to `n` there. Driving `h*` to zero on a core
//! buys nothing, because the exponent that governs simulation includes
//! the basis change.
//!
//! Both failures are structural, not a matter of tuning. The suite
//! asserts the ordinary outcome so that the extraordinary one would turn
//! it red rather than pass quietly.

use quantsim::bounds::{fit_law, Law};
use quantsim::circuit::Op;
use quantsim::pathsum;
use quantsim::prelude::*;
use quantsim::upembed;

// ─────────────────────────────────────────────────────────────────────
// An independent Clifford oracle.
//
// Nothing here touches `pathsum`. A unitary is Clifford exactly when it
// maps every Pauli generator to a Pauli, so build the matrix by dense
// simulation and check conjugation directly. This is the ground truth
// that every Clifford claim in this crate is measured against.
// ─────────────────────────────────────────────────────────────────────

/// The circuit's unitary, column by column, by dense simulation.
/// Returns `u[column][row]`.
fn unitary(c: &Circuit<C64>) -> Vec<Vec<C64>> {
    let n = c.num_qubits();
    let sim = Simulator::<C64>::new();
    let bound = c.bind(sim.registry()).expect("bind");
    (0..1usize << n)
        .map(|j| {
            let mut st = DenseState::<C64>::new(n).expect("dense state");
            st.load(&[(j as u64, C64::new(1.0, 0.0))]).expect("load");
            bound.run(&mut st).expect("run");
            st.amplitudes().to_vec()
        })
        .collect()
}

/// `U (X^a Z^b) U†` as `m[row][col]`.
///
/// `(X^a Z^b)|j⟩ = (-1)^{b·j}|j ⊕ a⟩`, so the conjugation is a direct sum
/// over the basis with no matrix library needed.
fn conjugate(u: &[Vec<C64>], a: usize, b: usize) -> Vec<Vec<C64>> {
    let dim = u.len();
    let mut m = vec![vec![C64::new(0.0, 0.0); dim]; dim];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            let mut acc = C64::new(0.0, 0.0);
            for j in 0..dim {
                let sign = if ((b & j).count_ones() & 1) == 1 {
                    -1.0
                } else {
                    1.0
                };
                acc += u[j ^ a][r] * sign * u[j][c].conj();
            }
            *cell = acc;
        }
    }
    m
}

/// Is `m` a Pauli string times a global phase?
///
/// A Pauli `X^a Z^b` is monomial with unit entries, its column-to-row map
/// is XOR by a constant, and its phase pattern is a *character* —
/// `λ_c/λ_0 = (-1)^{b·c}`. All three are checked; anything else is not a
/// Pauli however close it looks.
fn is_pauli_up_to_phase(m: &[Vec<C64>]) -> bool {
    let dim = m.len();
    let n = dim.trailing_zeros() as usize;
    let mut offset: Option<usize> = None;
    let mut lambda = vec![C64::new(0.0, 0.0); dim];
    for c in 0..dim {
        let mut nz: Option<usize> = None;
        for (r, row) in m.iter().enumerate() {
            if row[c].norm() > 1e-9 {
                if nz.is_some() {
                    return false; // more than one nonzero in a column
                }
                nz = Some(r);
            }
        }
        let Some(r) = nz else { return false };
        match offset {
            None => offset = Some(r ^ c),
            Some(o) if o == r ^ c => {}
            _ => return false, // not XOR by a constant
        }
        lambda[c] = m[r][c];
    }
    if lambda.iter().any(|l| (l.norm() - 1.0).abs() > 1e-7) {
        return false;
    }
    let l0 = lambda[0];
    let mut b = 0usize;
    for k in 0..n {
        let ratio = lambda[1 << k] / l0;
        if (ratio - C64::new(1.0, 0.0)).norm() < 1e-7 {
        } else if (ratio + C64::new(1.0, 0.0)).norm() < 1e-7 {
            b |= 1 << k;
        } else {
            return false; // phase pattern is not ±1
        }
    }
    lambda.iter().enumerate().all(|(c, &l)| {
        let sign = if ((b & c).count_ones() & 1) == 1 {
            -1.0
        } else {
            1.0
        };
        (l - l0 * sign).norm() <= 1e-7
    })
}

/// Ground truth: is this circuit's unitary Clifford? Dense simulation
/// only, `O(8^n)`, so it is the small-`n` anchor everything else is
/// calibrated against.
fn is_clifford(c: &Circuit<C64>) -> bool {
    let u = unitary(c);
    (0..c.num_qubits()).all(|q| {
        is_pauli_up_to_phase(&conjugate(&u, 1 << q, 0))
            && is_pauli_up_to_phase(&conjugate(&u, 0, 1 << q))
    })
}

fn h_star(c: &Circuit<C64>) -> usize {
    pathsum::operator(c)
        .expect("operator path sum")
        .internal_vars()
}

// ─────────────────────────────────────────────────────────────────────
// Families.
// ─────────────────────────────────────────────────────────────────────

fn clifford_block(c: &mut Circuit<C64>, n: usize, gates: usize, rng: &mut Prng) {
    for _ in 0..gates {
        let a = (rng.next_u64() % n as u64) as usize;
        let b = if n > 1 {
            (a + 1 + (rng.next_u64() % (n as u64 - 1)) as usize) % n
        } else {
            a
        };
        match rng.next_u64() % 5 {
            0 => c.gate("h", vec![], vec![a]),
            1 => c.gate("s", vec![], vec![a]),
            2 => c.gate("sdg", vec![], vec![a]),
            3 if n > 1 => c.gate("cx", vec![], vec![a, b]),
            _ if n > 1 => c.gate("cz", vec![], vec![a, b]),
            _ => c.gate("h", vec![], vec![a]),
        };
    }
}

/// Random Clifford+T: `t` magic gates each separated by a random Clifford
/// block. Magic has no reason to cancel, which is why this is the family
/// the complexity question rides on.
fn clifford_t(n: usize, t: usize, seed: u64) -> Circuit<C64> {
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    clifford_block(&mut c, n, 3 * n, &mut rng);
    for _ in 0..t {
        let q = (rng.next_u64() % n as u64) as usize;
        c.gate("t", vec![], vec![q]);
        clifford_block(&mut c, n, 3 * n, &mut rng);
    }
    c
}

/// The `i < j < k` triples of `0..n`, in lexicographic order.
fn triples(n: usize) -> Vec<[usize; 3]> {
    let mut out = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                out.push([i, j, k]);
            }
        }
    }
    out
}

/// A purely diagonal non-Clifford circuit: one `T` per qubit and `k`
/// `CCZ` gates on **distinct** triples.
///
/// Distinctness is what makes it a usable witness. An earlier version
/// cycled `(i, i+1, i+2) mod n`, which repeats every `n` gates — so at
/// `k = 16, n = 4` every `CCZ` appeared an even number of times, `CCZ² =
/// I`, and the circuit really was Clifford. The dense oracle caught it.
/// With each triple used once the cubic terms cannot cancel, so the
/// circuit is non-Clifford at every `k ≥ 1` by construction.
///
/// Diagonal operators map each basis state to itself with a phase, so no
/// variable is ever hidden: `h*` is zero however much magic is loaded in.
fn diagonal_core(n: usize, k: usize) -> Circuit<C64> {
    let ts = triples(n);
    assert!(k <= ts.len(), "only {} distinct triples on {n} qubits", ts.len());
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.gate("t", vec![], vec![q]);
    }
    for t in ts.iter().take(k) {
        c.gate("ccz", vec![], t.to_vec());
    }
    c
}

/// The same core between Hadamard layers — which is just measuring it in
/// the X basis. This is IQP, believed hard to sample classically.
fn iqp(n: usize, k: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.gate("h", vec![], vec![q]);
    }
    let core = diagonal_core(n, k.min(triples(n).len()));
    c.append(&core, &(0..n).collect::<Vec<_>>());
    for q in 0..n {
        c.gate("h", vec![], vec![q]);
    }
    c
}

/// Magic that cancels: `T`, a Clifford round trip, then `T†`. The gate
/// list is full of magic; the unitary has none.
fn cancelling(n: usize, k: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for i in 0..k {
        c.gate("t", vec![], vec![i % n]);
        c.gate("cx", vec![], vec![i % n, (i + 1) % n]);
        c.gate("cx", vec![], vec![i % n, (i + 1) % n]);
        c.gate("tdg", vec![], vec![i % n]);
        c.gate("h", vec![], vec![(i + 2) % n]);
    }
    c
}

fn magic_circuit(n: usize, k: usize) -> Circuit<C64> {
    let mut c = Circuit::new(n);
    for i in 0..k {
        c.gate("h", vec![], vec![i % n]);
        c.gate("t", vec![], vec![i % n]);
        c.gate("cx", vec![], vec![i % n, (i + 1) % n]);
        c.gate("t", vec![], vec![(i + 1) % n]);
    }
    c
}

fn dagger(a: &Circuit<C64>) -> Circuit<C64> {
    let mut c = Circuit::new(a.num_qubits());
    for op in a.ops().iter().rev() {
        let Op::Named {
            name,
            params,
            qubits,
        } = op
        else {
            panic!("dagger: only named registry gates")
        };
        let inv = match name.as_str() {
            "t" => "tdg",
            "tdg" => "t",
            "s" => "sdg",
            "sdg" => "s",
            same @ ("h" | "x" | "y" | "z" | "cx" | "cz" | "ccz" | "swap") => same,
            other => panic!("dagger: unhandled gate {other}"),
        };
        c.gate(inv, params.clone(), qubits.clone());
    }
    c
}

/// `C U C†` for a random Clifford `C`.
fn conjugated(n: usize, inner: &Circuit<C64>, depth: usize, seed: u64) -> Circuit<C64> {
    let mut rng = Prng::new(seed);
    let mut front = Circuit::new(n);
    clifford_block(&mut front, n, depth, &mut rng);
    let id: Vec<usize> = (0..n).collect();
    let mut c = Circuit::new(n);
    c.append(&front, &id);
    c.append(inner, &(0..inner.num_qubits()).collect::<Vec<_>>());
    c.append(&dagger(&front), &id);
    c
}

// ─────────────────────────────────────────────────────────────────────
// 1. What `h*` is not. The correction, locked in.
// ─────────────────────────────────────────────────────────────────────

/// `h* = 0` does **not** mean the unitary is Clifford, and this test
/// exists so that claim cannot come back.
///
/// The strongest witness is the diagonal core: sixty-four `CCZ` and
/// sixty-four `T` gates, unbounded magic, `h* = 0` at every depth.
/// Diagonal operators never hide a variable — the output index fixes the
/// input — so the one-amplitude exponent is zero no matter how much
/// magic is present. The dense oracle confirms every one of these is
/// non-Clifford.
#[test]
fn h_star_zero_does_not_certify_clifford() {
    let mut lone_t = Circuit::<C64>::new(1);
    lone_t.gate("t", vec![], vec![0]);
    let mut ccz = Circuit::<C64>::new(3);
    ccz.gate("ccz", vec![], vec![0, 1, 2]);
    let mut ht = Circuit::<C64>::new(1);
    ht.gate("h", vec![], vec![0]);
    ht.gate("t", vec![], vec![0]);

    for (label, c) in [
        ("a lone T", lone_t),
        ("CCZ", ccz),
        ("H then T", ht),
        ("diagonal core n=4 k=4", diagonal_core(4, 4)),
        ("diagonal core n=5 k=10", diagonal_core(5, 10)),
        ("diagonal core n=6 k=20", diagonal_core(6, 20)),
    ] {
        assert_eq!(
            h_star(&c),
            0,
            "{label}: expected the one-amplitude exponent to be zero"
        );
        assert!(
            !is_clifford(&c),
            "{label}: the dense oracle says this IS Clifford, which would \
             break the counterexample"
        );
    }

    // And it holds as the magic content grows without bound — past the
    // width where the dense oracle can follow, which is the whole point.
    for (n, k) in [(8usize, 56usize), (10, 120), (12, 220)] {
        let c = diagonal_core(n, k);
        assert_eq!(
            h_star(&c),
            0,
            "diagonal core n={n} k={k}: h* must stay zero however much magic \
             is added"
        );
        assert_eq!(c.len(), n + k, "and the magic really is all still there");
    }
}

/// The other direction, measured: every Clifford circuit in the corpus
/// *did* reduce to zero.
///
/// Together with [`h_star_zero_does_not_certify_clifford`] this places
/// the classes precisely: `{h* = 0}` is a **strict superset** of the
/// Clifford group. Reduction is complete on Cliffords here — no witness
/// with `h* > 0` was found in 120 attempts — but it also reduces plenty
/// of non-Clifford circuits, every diagonal one among them. So `h* = 0`
/// can never be read backwards as a Clifford certificate.
///
/// This is a corpus measurement, not a proof of completeness. What it
/// must not do is silently change: a nonzero `worst` here would mean the
/// rewrite system is incomplete on the Clifford fragment.
#[test]
fn every_clifford_circuit_in_the_corpus_reduces_to_zero() {
    let mut worst = 0usize;
    let mut seen = 0usize;
    for seed in 0..40u64 {
        for depth in [4usize, 8, 12] {
            let mut rng = Prng::new(seed ^ ((depth as u64) << 32));
            let mut c = Circuit::<C64>::new(3);
            clifford_block(&mut c, 3, depth, &mut rng);
            let q = (rng.next_u64() % 3) as usize;
            c.gate("t", vec![], vec![q]);
            clifford_block(&mut c, 3, depth, &mut rng);
            c.gate("tdg", vec![], vec![q]);
            if is_clifford(&c) {
                seen += 1;
                worst = worst.max(h_star(&c));
            }
        }
    }
    assert!(seen > 0, "no Clifford witnesses were generated");
    println!("Clifford witnesses: {seen}, worst h* = {worst}");
    assert_eq!(
        worst, 0,
        "a Clifford circuit reduced to h* = {worst} > 0, so the rewrite system \
         is incomplete on the Clifford fragment — which is a finding, but it \
         contradicts the {seen}-witness measurement on record"
    );
}

// ─────────────────────────────────────────────────────────────────────
// 2. What `h*` is — and that reduction is sound.
// ─────────────────────────────────────────────────────────────────────

/// Every amplitude of the reduced path sum against the dense reference.
/// A reduction that "removes magic" by dropping a term would look like a
/// breakthrough; this is what catches it.
#[test]
fn reduction_preserves_every_amplitude_against_dense() {
    for seed in 0..12u64 {
        for t in [0usize, 2, 5, 9] {
            let c = clifford_t(4, t, seed);
            let sim = Simulator::<C64>::new();
            let out = sim.run(&c).expect("dense run");
            let mut ps = pathsum::PathSum::from_circuit(&c).expect("path sum");
            ps.reduce();
            for (i, a) in ps.to_dense().iter().enumerate() {
                let b = out.amplitude(i as u64);
                assert!(
                    (a - b).norm() < 1e-9,
                    "amplitude {i} diverged after reduction (t={t}, seed={seed}): \
                     path sum {a} vs dense {b}"
                );
            }
        }
    }
}

/// `h*` is the exponent for one amplitude, so a circuit with `h* = 0`
/// must yield exact amplitudes with no summation left — including the
/// deep diagonal cores, where a wrong reading of `h*` would otherwise go
/// unnoticed because the answer is never computed.
#[test]
fn a_zero_exponent_means_one_amplitude_costs_nothing_and_is_still_correct() {
    for k in [1usize, 10, 20] {
        let c = diagonal_core(6, k);
        assert_eq!(h_star(&c), 0);
        let sim = Simulator::<C64>::new();
        let out = sim.run(&c).expect("dense run");
        let mut ps = pathsum::PathSum::from_circuit(&c).expect("path sum");
        ps.reduce();
        for (i, a) in ps.to_dense().iter().enumerate() {
            let b = out.amplitude(i as u64);
            assert!(
                (a - b).norm() < 1e-9,
                "diagonal core k={k}: amplitude {i} wrong ({a} vs {b})"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// 3. The Clifford claim, established the way it actually holds.
// ─────────────────────────────────────────────────────────────────────

/// The up-embedded dynamics really is Clifford — but by *construction*,
/// because `to_circuit` emits nothing but `h`, `s` and `cx`. Running it
/// through `pathsum` was never evidence of that, since `h* = 0` holds for
/// plenty of non-Clifford circuits. This checks the two things that do
/// establish it: the emitted gate set, and the dense oracle.
#[test]
fn the_up_embedding_is_clifford_by_gate_set_and_by_oracle() {
    for c in [magic_circuit(3, 4), magic_circuit(3, 6), cancelling(3, 4)] {
        let emb = upembed::gadgetize(&c).expect("gadgetize");
        assert!(emb.magic() > 0, "there should be magic to relocate");
        let cliff = emb.to_circuit();

        for op in cliff.ops() {
            let Op::Named { name, .. } = op else {
                panic!("up-embedding emitted a non-registry op")
            };
            assert!(
                matches!(name.as_str(), "h" | "s" | "cx"),
                "up-embedding emitted {name}, which is outside the Clifford \
                 generators the construction promises"
            );
        }
    }

    // And confirm the gate-set argument against ground truth at a width
    // the dense oracle can reach.
    let small = upembed::gadgetize(&magic_circuit(2, 2))
        .expect("gadgetize")
        .to_circuit();
    assert!(
        is_clifford(&small),
        "the up-embedded dynamics is not Clifford by the dense oracle"
    );
}

// ─────────────────────────────────────────────────────────────────────
// 4. The complexity question, which fails twice.
// ─────────────────────────────────────────────────────────────────────

/// First failure: nothing collapses. On random Clifford+T the
/// one-amplitude exponent grows with `T` count and the cost stays
/// exponential.
///
/// This asserts the ordinary outcome, so the extraordinary one turns the
/// suite red instead of passing quietly.
#[test]
fn h_star_grows_with_t_count_on_random_clifford_t() {
    let ts = [4usize, 8, 12, 16, 20, 24];
    let seeds = 6u64;
    let means: Vec<usize> = ts
        .iter()
        .map(|&t| (0..seeds).map(|s| h_star(&clifford_t(6, t, s))).sum::<usize>() / seeds as usize)
        .collect();
    println!("random Clifford+T on n=6:");
    for (t, h) in ts.iter().zip(&means) {
        println!(
            "   T={t:3}   mean h* = {h:3}   one amplitude costs 2^h* = {:.3e}",
            (*h as f64).exp2()
        );
    }
    assert!(
        means.last().unwrap() > means.first().unwrap(),
        "h* did not grow between T=4 and T=24 on random Clifford+T. If real, \
         that is a claim about universal families; check \
         `reduction_preserves_every_amplitude_against_dense` first."
    );
    let costs: Vec<usize> = means.iter().map(|&h| 1usize << h).collect();
    let fit = fit_law(&ts, &costs);
    println!("   fitted amplitude-cost law: {}", fit.law);
    assert!(
        matches!(fit.law, Law::Exponential { .. }),
        "amplitude cost on random Clifford+T fitted as {} rather than \
         exponential; verify against dense before treating it as a result",
        fit.law
    );
}

/// Second failure, and the independent one: **cheap amplitudes are not
/// cheap sampling.**
///
/// Even if some rewrite drove `h*` to zero on every circuit, `BQP = BPP`
/// would not follow. The diagonal core is the standing witness: `h* = 0`
/// at any depth with unbounded magic, yet putting it between Hadamard
/// layers — which is just measuring it in the X basis — gives IQP, which
/// is believed hard to sample. The exponent that governs simulation
/// includes the basis change, and `h*` correctly rises to `n` once it is
/// included.
#[test]
fn a_zero_exponent_core_becomes_expensive_the_moment_it_is_measured() {
    for k in [4usize, 10, 20] {
        assert_eq!(
            h_star(&diagonal_core(6, k)),
            0,
            "the core must stay free however deep"
        );
    }
    let mut prev = 0usize;
    for n in 4..8usize {
        let h = h_star(&iqp(n, 6));
        println!("   IQP n={n}: h* = {h}");
        assert!(
            h >= prev,
            "h* fell from {prev} to {h} as the IQP instance grew"
        );
        prev = h;
    }
    assert!(
        prev >= 6,
        "h* reached only {prev} on an IQP instance at n=7. A bounded exponent \
         for sampling-hard IQP would be the extraordinary claim, not the \
         zero-exponent core it is built from."
    );
}

// ─────────────────────────────────────────────────────────────────────
// 5. What the gadget counts actually measure.
// ─────────────────────────────────────────────────────────────────────

/// Spend gadgets one at a time and report the first count whose remaining
/// dynamics has a free amplitude.
fn gadgets_needed(c: &Circuit<C64>) -> usize {
    let t = upembed::magic_events(c).expect("magic events");
    (0..=t)
        .find(|&g| h_star(&upembed::gadgetize_partial(c, g).expect("partial")) == 0)
        .unwrap_or(t)
}

/// The "0 gadgets needed" rows are real, but they mean *the remaining
/// dynamics has a free amplitude* — not *the unitary was Clifford*. On
/// the cancelling family both happen to be true; on a diagonal core only
/// the first is, and that is the distinction the original framing lost.
#[test]
fn a_zero_gadget_count_means_free_amplitudes_not_a_clifford_unitary() {
    for k in [4usize, 8] {
        let c = cancelling(3, k);
        assert_eq!(gadgets_needed(&c), 0, "the control family stopped reducing");
        assert!(
            is_clifford(&c),
            "cancelling k={k} should also happen to be Clifford"
        );
    }
    // Same verdict, opposite truth: zero gadgets, and emphatically not Clifford.
    let core = diagonal_core(5, 10);
    assert_eq!(gadgets_needed(&core), 0);
    assert!(
        !is_clifford(&core),
        "the diagonal core must remain the counterexample"
    );
}

/// And the saving must not extend to families where magic survives — a
/// bounded gadget count on random Clifford+T would be the complexity
/// claim in disguise.
#[test]
fn the_gadget_count_grows_on_random_clifford_t() {
    let mut counts = Vec::new();
    for t in [4usize, 8, 12] {
        let c = clifford_t(4, t, 3);
        counts.push((t, upembed::magic_events(&c).expect("events"), gadgets_needed(&c)));
    }
    println!("random Clifford+T gadget counts (T, events, needed):");
    for (t, e, n) in &counts {
        println!("   T={t:3}  events={e:3}  needed={n:3}");
    }
    assert!(
        counts[counts.len() - 1].2 > counts[0].2,
        "gadgets needed did not grow with T count ({} -> {})",
        counts[0].2,
        counts[counts.len() - 1].2
    );
}

// ─────────────────────────────────────────────────────────────────────
// 6. Clifford conjugation.
// ─────────────────────────────────────────────────────────────────────

/// Magic is invariant under Clifford conjugation, so a magic measure
/// would give `h*(C U C†) = h*(U)`. It does not — which is consistent,
/// because `h*` measures amplitude cost and conjugation genuinely changes
/// that. This pins the size of the effect rather than reading it as
/// magic.
#[test]
fn conjugating_by_a_clifford_moves_the_amplitude_exponent() {
    let mut inner = Circuit::<C64>::new(3);
    inner.gate("t", vec![], vec![0]);
    let base = h_star(&inner);
    let mut worst = base;
    let mut moved = 0usize;
    let mut total = 0usize;
    for seed in 0..12u64 {
        for depth in [2usize, 6, 10] {
            let h = h_star(&conjugated(3, &inner, depth, seed));
            total += 1;
            if h != base {
                moved += 1;
            }
            worst = worst.max(h);
        }
    }
    println!(
        "Clifford conjugation of a single T: base h* = {base}, worst = {worst}, \
         moved in {moved}/{total} cases"
    );
    assert!(
        moved > 0,
        "conjugation never moved h*, so this test measures nothing"
    );
}
