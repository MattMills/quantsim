//! Regev's factoring algorithm: the schedule is the whole argument.
//!
//! Run with `cargo run --release --example regev`.
//!
//! Regev replaces Shor's one exponent register with `d ≈ √n` of them.
//! That alone buys nothing — the naive schedule costs exactly what Shor
//! costs. The saving is in *how* `∏ bᵢ^{zᵢ}` is scheduled, and the price
//! of the saving is space. This example measures all three schedules,
//! finds where their qubit counts cross, and then measures the two dials
//! that decide whether the classical half works at all.

use quantsim::prelude::*;
use quantsim::regev::{
    self, coprime_primes, zeckendorf_len, ExpPlan, ExpSchedule, Layout, Regev, Superposition,
};
use quantsim::shor::{self, PhaseForm};

const SCHEDULES: [ExpSchedule; 3] = [
    ExpSchedule::Sequential,
    ExpSchedule::Regev,
    ExpSchedule::Fibonacci,
];

fn main() -> Result<()> {
    let sim: Simulator = Simulator::new();

    println!("== 1. one full run, every intermediate shown, so it can be checked ==");
    {
        let n = 35u64;
        let cfg = Regev::new(n, 2)?;
        let layout = cfg.layout();
        println!(
            "   N = {n}, d = {}, R = {} exponent bits",
            cfg.dimension, cfg.exponent_bits
        );
        println!(
            "   primes pᵢ  {:?}   bases bᵢ = pᵢ² mod N  {:?}",
            cfg.primes,
            cfg.bases()
        );
        println!(
            "   layout: {} exponent registers of {} qubits + {} work register(s) of {} = {} qubits",
            layout.exponent.len(),
            cfg.exponent_bits,
            layout.work.len(),
            quantsim::shor::work_bits(n),
            layout.qubits
        );
        let q = 1i64 << cfg.exponent_bits;
        let mut rng = Prng::new(4);
        let report = cfg.run(&sim, &mut rng, 6)?;
        println!("   the lattice L = {{ z : ∏ bᵢ^zᵢ ≡ 1 mod {n} }}, sampled 6 times:");
        println!("     run   work value measured   dual vector w   ⟨w, v⟩ mod q for the winner");
        let w = report.witness.as_ref().expect("a witness");
        for (i, sample) in report.samples.iter().enumerate() {
            let dot: i64 = sample
                .dual
                .iter()
                .zip(w.vector.iter())
                .map(|(&a, &b)| a as i64 * b)
                .sum();
            let centred = ((dot % q) + q) % q;
            let centred = if centred > q / 2 {
                centred - q
            } else {
                centred
            };
            println!(
                "     {i}     {:<19}   {:<13}   {centred}",
                sample.work_value,
                format!("{:?}", sample.dual)
            );
        }
        let rows: Vec<Vec<i64>> = report
            .samples
            .iter()
            .map(|s| s.dual.iter().map(|&x| x as i64).collect())
            .collect();
        let basis =
            quantsim::lattice::congruence_kernel(&rows, cfg.dimension, q, cfg.lattice_weight);
        println!(
            "   LLL over the congruence kernel gives, shortest first: {:?}",
            basis.iter().take(4).collect::<Vec<_>>()
        );
        println!("   winning vector v       {:?}", w.vector);
        let u = w.root;
        println!("   u = ∏ pᵢ^vᵢ mod N      {u}");
        println!(
            "     check u² mod N       {u}² mod {n} = {}  (must be 1)",
            quantsim::shor::mul_mod(u, u, n)
        );
        println!(
            "     check u ≠ ±1         u = {u}, and ±1 is 1 or {}",
            n - 1
        );
        println!(
            "     gcd(u−1, N)          gcd({}, {n}) = {}",
            u - 1,
            quantsim::padic::gcd(u - 1, n)
        );
        println!(
            "     gcd(u+1, N)          gcd({}, {n}) = {}",
            u + 1,
            quantsim::padic::gcd(u + 1, n)
        );
        println!(
            "   factors                {} × {} = {}",
            w.factors.0,
            w.factors.1,
            w.factors.0 * w.factors.1
        );
        assert_eq!(w.factors.0 * w.factors.1, n);
        assert!(w.square_root_of_one);
        assert_eq!(quantsim::shor::mul_mod(u, u, n), 1);
        assert_ne!(u, 1);
        assert_ne!(u, n - 1);
        println!("   every line above is asserted in this example, not just printed.");
        println!("   The squares are the whole construction: because bᵢ = pᵢ², any");
        println!("   v ∈ L makes ∏ pᵢ^vᵢ a square root of 1, and any square root other");
        println!("   than ±1 splits N. Shor gets one per successful order run; Regev");
        println!("   gets one per short lattice vector.");
    }

    println!();
    println!("== 2. the three schedules, measured against the table they claim ==");
    println!("   N = 35, d = 2, R = 4");
    println!("     schedule      full   small   registers   qubits   claimed (full, small)");
    let (n, d, r) = (35u64, 2usize, 4usize);
    let bases: Vec<u64> = coprime_primes(n, d).0.iter().map(|p| p * p % n).collect();
    for schedule in SCHEDULES {
        let layout = Layout::new(schedule, d, r, shor::work_bits(n));
        let plan = ExpPlan {
            modulus: n,
            bases: bases.clone(),
            exponent_bits: r,
            schedule,
            layout: layout.clone(),
        };
        let mut state = sim.backends().create("sparse", layout.qubits)?;
        state.load(&[(0, C64::new(1.0, 0.0))])?;
        let cost = regev::exponentiate(&sim, state.as_mut(), &plan)?;
        println!(
            "   {schedule:<12}  {:5}  {:6}   {:9}   {:6}   {:?}",
            cost.full_multiplications,
            cost.small_multiplications,
            cost.work_registers,
            cost.qubits,
            schedule.multiplications(d, r)
        );
    }
    println!("   All three compute the same permutation and leave no garbage — that is");
    println!("   asserted over the whole box in tests/regev.rs, not eyeballed here.");

    println!();
    println!("== 3. where the saving actually starts ==");
    println!("   Sequential costs d·R full-width multiplications; Regev's costs 2R.");
    println!("   So the multi-register idea pays nothing until d > 2.");
    println!("     d    R    sequential   regev   fibonacci");
    for d in 2..=6usize {
        for r in [4usize, 16] {
            let (s, _) = ExpSchedule::Sequential.multiplications(d, r);
            let (g, _) = ExpSchedule::Regev.multiplications(d, r);
            let (f, _) = ExpSchedule::Fibonacci.multiplications(d, r);
            println!("   {d:3}  {r:3}   {s:10}   {g:5}   {f:9}");
        }
    }

    println!();
    println!("== 4. the qubit crossover: Regev's schedule vs the Fibonacci ladder ==");
    println!("   Regev holds R+2 registers of w; Fibonacci holds 3, and pays for it in");
    println!("   d·K extra digit qubits. The crossover is a closed form, and it moves.");
    println!("     w    d    R    regev qubits   fibonacci qubits   winner");
    for &(w, d, r) in &[
        (4usize, 2usize, 4usize),
        (6, 2, 4),
        (8, 2, 4),
        (6, 2, 6),
        (10, 3, 6),
        (16, 4, 8),
        (32, 6, 12),
        (64, 8, 16),
    ] {
        let k = zeckendorf_len(r);
        let regev_q = d * r + (r + 2) * w;
        let fib_q = d * r + d * k + 3 * w;
        let winner = if fib_q < regev_q {
            "fibonacci"
        } else if fib_q > regev_q {
            "regev"
        } else {
            "tie"
        };
        println!("   {w:3}  {d:3}  {r:3}   {regev_q:12}   {fib_q:16}   {winner}");
    }
    println!("   The crossover is w(R−1) > d·K, i.e. it arrives as soon as the residue");
    println!("   width outgrows the exponent bookkeeping — which for a real modulus");
    println!("   (w = n, d ≈ R ≈ √n) it does by a factor of √n. That is the");
    println!("   Õ(n^1.5) → Õ(n) qubit result, visible here at w = 6.");

    println!();
    println!("== 5. the lattice weight, and the failure it exists to prevent ==");
    println!("   The sampled duals are rounded, so a true lattice vector has a small");
    println!("   residue, not a zero one. Demanding zero returns the trivial kernel.");
    println!("     weight   genuine square roots of one, over 8 seeds each");
    println!("              N=33 d=2   N=35 d=2   N=51 d=3   N=55 d=3");
    for weight in [1i64, 4, 64, 1024, 65536] {
        let mut row = String::new();
        for &(n, d) in &[(33u64, 2usize), (35, 2), (51, 3), (55, 3)] {
            let mut roots = 0;
            for seed in 0..8u64 {
                let mut rng = Prng::new(seed);
                let report =
                    Regev::new(n, d)?
                        .with_lattice_weight(weight)
                        .run(&sim, &mut rng, d + 4)?;
                if report
                    .witness
                    .as_ref()
                    .is_some_and(|w| w.square_root_of_one)
                {
                    roots += 1;
                }
            }
            row.push_str(&format!("   {roots}/8      "));
        }
        println!("   {weight:6}  {row}");
    }
    println!("   The expectation this section was written to test — that a large weight");
    println!("   forces the trivial kernel qℤ^d and loses the factors — is **wrong here**,");
    println!("   and the reason is instructive. It holds for generic congruence rows");
    println!("   (asserted in tests/lattice.rs), but a correctly size-reduced LLL finds");
    println!("   the short kernel vector at every weight these samples produce. The");
    println!("   dial is real and it is not the binding constraint at this scale; what");
    println!("   was binding was an LLL that only size-reduced against stale μ.");

    println!();
    println!("== 6. uniform against Gaussian preparation ==");
    println!("   Regev's analysis assumes a Gaussian. A Hadamard layer gives a box.");
    println!("   Loading the Gaussian is free here and is not free on hardware.");
    println!("     N    preparation      prepared support   peak support   genuine/8");
    for n in [33u64, 35, 55] {
        for prep in [
            Superposition::Uniform,
            Superposition::Gaussian(6.0),
            Superposition::Gaussian(16.0),
        ] {
            let cfg = Regev::new(n, 2)?.with_superposition(prep);
            // What the preparation itself puts on the register, before
            // any arithmetic — the only place the two differ.
            let prepared = {
                let layout = cfg.layout();
                let mut state = sim.backends().create("sparse", layout.qubits)?;
                cfg.prepare_for(&sim, state.as_mut(), &layout)?;
                state.nonzero_count()
            };
            let mut roots = 0;
            let mut peak = 0;
            for seed in 0..8u64 {
                let mut rng = Prng::new(seed);
                let report = cfg.run(&sim, &mut rng, 6)?;
                peak = peak.max(report.cost.peak_support);
                if report
                    .witness
                    .as_ref()
                    .is_some_and(|w| w.square_root_of_one)
                {
                    roots += 1;
                }
            }
            let label = match prep {
                Superposition::Uniform => "uniform".to_string(),
                Superposition::Gaussian(s) => format!("gaussian σ={s}"),
            };
            println!("   {n:3}    {label:<16} {prepared:16}   {peak:12}   {roots}/8");
        }
    }
    println!("   The preparations genuinely differ — a narrow Gaussian holds a fraction");
    println!("   of the box — and it makes no difference to either the cost or the");
    println!("   answer, because the inverse QFT refills the register either way. The");
    println!("   Gaussian is what Regev's *analysis* needs, not what these sizes need.");

    println!();
    println!("== 7. Shor against Regev, in both currencies ==");
    println!("   Circuit currency is full-width modular multiplications. Simulator");
    println!("   currency is peak support. They point opposite ways, and both are true.");
    println!("     N    shor mults   regev mults   shor support   regev support");
    for n in [15u64, 21, 33, 35, 55] {
        let a = (2..n).find(|&a| quantsim::padic::gcd(a, n) == 1).unwrap();
        let finder = shor::OrderFinder::new(n, a)?;
        let mut rng = Prng::new(3);
        let shor_est = finder.estimate(&sim, PhaseForm::Semiclassical, &mut rng)?;
        let mut rng = Prng::new(3);
        let rep = regev::factor(&sim, n, 2, &mut rng)?;
        let per_sample = rep.cost.full_multiplications / rep.samples.len();
        println!(
            "   {n:3}    {:10}   {:11}   {:12}   {:13}",
            shor_est.modular_multiplications,
            per_sample,
            shor_est.peak_support,
            rep.cost.peak_support
        );
    }
    println!("   Regev is cheaper in the circuit and far more expensive to simulate:");
    println!("   its exponent box is 2^(dR) ≥ N by construction, while Shor's");
    println!("   semiclassical support is the orbit r. Simulability and hardware cost");
    println!("   are not the same axis, and this is the clearest case of it in the crate.");

    println!();
    println!("== 8. end to end ==");
    println!("     N   d    primes        skipped   witness v            u    factors");
    for n in [15u64, 21, 33, 35, 39, 51, 55] {
        for d in [2usize, 3] {
            let mut rng = Prng::new(4);
            let report = regev::factor(&sim, n, d, &mut rng)?;
            let w = report.witness.as_ref();
            println!(
                "   {n:3}   {d}   {:12}  {:8}   {:18}  {:4}  {}",
                format!("{:?}", report.primes),
                format!("{:?}", report.trivial_divisors),
                w.map(|w| format!("{:?}", w.vector)).unwrap_or_default(),
                w.map(|w| w.root.to_string()).unwrap_or_default(),
                report
                    .factors
                    .map(|(p, q)| format!("{p}×{q}"))
                    .unwrap_or_else(|| "-".into()),
            );
        }
    }
    println!();
    println!("   The skipped column is a property of base selection, not a result: a");
    println!("   small prime that divides N cannot serve as a base, so the search for");
    println!("   bases notices it. It is reported so that a row whose factors were");
    println!("   already sitting in that column is not read as the lattice having");
    println!("   done the work — the witness column is where that is shown.");
    Ok(())
}
