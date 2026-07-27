//! Tour of the swappable amplitude algebras: the same simulator machinery
//! over ℝ, ℂ, ℍ, 𝕆, 𝕊 and the split-complex numbers, showing what each
//! algebra keeps and what it gives up.
//!
//! Run with `cargo run --example exotic_algebras`.

use quantsim::prelude::*;

fn main() -> Result<()> {
    println!("== Gate availability per algebra ==");
    survey_all();

    println!();
    println!("== ℝ: rebit GHZ (real gate subset only) ==");
    let state = Simulator::<f64>::new().run(&library::ghz(3))?;
    for (index, p) in state.probabilities() {
        println!(
            "  |{index:03b}⟩ amp = {:+.4}, p = {p:.4}",
            state.amplitude(index)
        );
    }

    println!();
    println!("== ℍ: a genuinely quaternionic gate ==");
    let mut sim = Simulator::<Quaternion>::new();
    let j = Quaternion::basis(2);
    let mut m = GateMatrix::<Quaternion>::identity(2)?;
    m.set(1, 1, j);
    sim.registry_mut()
        .register_fixed("jphase", "diag(1, j)", m)?;
    let mut c: Circuit<Quaternion> = Circuit::new(1);
    c.h(0).gate("jphase", Vec::new(), vec![0]);
    let state = sim.run(&c)?;
    println!("  H|0⟩ then diag(1, j):");
    for index in 0..2u64 {
        println!(
            "    |{index}⟩ coeffs [1,i,j,k] = {:?}",
            state.amplitude(index).coeffs()
        );
    }
    println!(
        "  total Born weight = {:.6} (ℍ is a division algebra: conserved)",
        state.total_weight()
    );

    println!();
    println!("== 𝕊: sedenion zero divisors break weight conservation ==");
    let (x, (p, q, r, sd)) = find_sedenion_zero_divisor();
    println!("  found zero divisor: (e{p}+e{q})·(e{r}−e{sd}) = 0, both factors nonzero");
    let u = (Sedenion::basis(5) + Sedenion::basis(14)).scale(std::f64::consts::FRAC_1_SQRT_2);
    let mut m = GateMatrix::<Sedenion>::identity(2)?;
    m.set(1, 1, u);
    let mut state = DenseState::<Sedenion>::new(1)?;
    state.load(&[(0, x.scale(0.5)), (1, x.scale(0.5))])?;
    println!("  before: total weight = {:.6}", state.total_weight());
    state.apply(&m, &[0])?;
    println!(
        "  after unit-norm 'phase' along a zero-divisor direction: {:.6}",
        state.total_weight()
    );

    println!();
    println!("== split-ℂ: indefinite Born form ==");
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let mut state = DenseState::<SplitComplex>::new(1)?;
    state.load(&[
        (0, SplitComplex::new(s, 0.0)),
        (1, SplitComplex::new(0.0, s)),
    ])?;
    println!("  state (|0⟩ + j|1⟩)/√2:");
    println!(
        "    born(|0⟩) = {:+.3}  born(|1⟩) = {:+.3}",
        state.probability(0),
        state.probability(1)
    );
    println!(
        "    total = {:+.3} — a 'probability' distribution with negative parts;",
        state.total_weight()
    );
    println!(
        "    measurement errors out rather than pretending: {:?}",
        state
            .measure(0, &mut Prng::new(1))
            .err()
            .map(|e| e.to_string())
    );
    Ok(())
}

fn find_sedenion_zero_divisor() -> (Sedenion, (usize, usize, usize, usize)) {
    for p in 1..16 {
        for q in (p + 1)..16 {
            for r in 1..16 {
                for s in (r + 1)..16 {
                    let x = Sedenion::basis(p) + Sedenion::basis(q);
                    let y = Sedenion::basis(r) - Sedenion::basis(s);
                    if (x * y).is_zero(1e-12) {
                        return (x, (p, q, r, s));
                    }
                }
            }
        }
    }
    unreachable!("sedenions contain zero divisors");
}

fn survey_all() {
    fn survey<S: Scalar>() {
        let names = GateRegistry::<S>::standard().names();
        println!(
            "{:>24}: dim {:>2}, division {:>5}, associative {:>5} — {:>2} gates",
            S::algebra_name(),
            S::DIM,
            S::DIVISION,
            S::ASSOCIATIVE,
            names.len()
        );
    }
    survey::<f64>();
    survey::<C64>();
    survey::<CComplex>();
    survey::<SplitComplex>();
    survey::<Quaternion>();
    survey::<Octonion>();
    survey::<Sedenion>();
}
