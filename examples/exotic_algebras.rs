//! Tour of the swappable amplitude algebras: the same simulator machinery
//! over ℝ, ℂ, ℍ, 𝕆, 𝕊, the split-complex numbers and the split
//! quaternions, showing what each algebra keeps and what it gives up.
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

    split_quaternion_tour()?;
    Ok(())
}

/// Split quaternions as an inclusion/exclusion pair: one amplitude
/// carrying a constructive and a destructive complex component, with
/// measurement reading their difference.
fn split_quaternion_tour() -> Result<()> {
    type SQ = SplitQuaternion;
    let c = |re: f64, im: f64| C64::new(re, im);

    println!();
    println!("== split-ℍ: an inclusion/exclusion pair ==");
    println!(
        "  ℂ embeds (i is present), so unlike split-ℂ this carries the FULL gate set: {} gates",
        GateRegistry::<SQ>::standard().names().len()
    );
    println!("  ...while still being neither commutative nor a division algebra.");

    // A standard circuit drives both channels with the same complex
    // matrix and never mixes them: one run, two ordinary registers.
    let build = |n: usize| {
        let mut c: Circuit<SQ> = Circuit::new(n);
        c.h(0).cx(0, 1).t(1).cz(0, 1).h(1);
        c
    };
    let mut joint = DenseState::<SQ>::new(2)?;
    joint.load(&[
        (0, SQ::pair(c(0.8, 0.0), c(0.3, 0.0))),
        (3, SQ::pair(c(0.0, 0.0), c(0.5, 0.0))),
    ])?;
    build(2).bind(&GateRegistry::standard())?.run(&mut joint)?;
    println!();
    println!("  one run carrying ψ in inclusion and φ in exclusion:");
    println!(
        "  {:>6} {:>22} {:>22} {:>10}",
        "state", "inclusion", "exclusion", "net"
    );
    for i in 0..4u64 {
        let q = joint.amplitude(i);
        println!(
            "  |{i:02b}⟩  {:>+10.6}{:>+10.6}i {:>+10.6}{:>+10.6}i {:>+10.6}",
            q.inclusion().re,
            q.inclusion().im,
            q.exclusion().re,
            q.exclusion().im,
            q.born_weight()
        );
    }
    println!(
        "  net total {:+.6}, path total {:.6} — the Born rule reads the difference,",
        joint.total_weight(),
        joint.total_abs_sqr()
    );
    println!("  abs_sqr reads the sum. Two ledgers, tracked independently, in one pass.");

    // The unit-norm elements are SL(2,R): non-compact, so a legitimately
    // unitary gate can pump path weight without bound at fixed net.
    println!();
    println!("  the boost cosh t + sinh t·j — unitary, net-preserving, path-pumping:");
    println!(
        "  {:>6} {:>16} {:>14} {:>14}",
        "t", "unitarity dev", "net weight", "path weight"
    );
    for t in [0.0f64, 0.5, 1.0, 2.0] {
        let mut m = GateMatrix::<SQ>::identity(2)?;
        m.set(0, 0, SQ::boost(t));
        m.set(1, 1, SQ::boost(t));
        let mut st = DenseState::<SQ>::new(1)?;
        st.load(&[(0, SQ::included(c(1.0, 0.0)))])?;
        st.apply(&m, &[0])?;
        println!(
            "  {t:>6.2} {:>16.2e} {:>14.12} {:>14.6}",
            m.unitarity_deviation(),
            st.total_weight(),
            st.total_abs_sqr()
        );
    }
    println!("  (path weight is cosh 2t exactly: constructive and destructive amplitude");
    println!("   created in matched pairs, at zero net cost — interference as a gate.)");

    // The exchange j has norm -1: it turns constructive into destructive.
    let mut jx = GateMatrix::<SQ>::identity(2)?;
    jx.set(0, 0, SQ::exchange());
    jx.set(1, 1, SQ::exchange());
    println!();
    println!(
        "  the exchange j swaps the ledgers: N(j) = {:+.0}, unitarity deviation {:.0}",
        SQ::exchange().born_weight(),
        jx.unitarity_deviation()
    );
    println!("  so the registry refuses it — turning inclusion into exclusion outright");
    println!("  is exactly a non-unitary act, and the simulator says so.");

    // The null cone: exact cancellation is decidable.
    let mut null = DenseState::<SQ>::new(1)?;
    null.load(&[(0, SQ::pair(c(0.5, 0.0), c(0.5, 0.0)))])?;
    println!();
    println!(
        "  balanced ledgers: net {:.1e}, path {:.3} — 'everything cancelled' is",
        null.total_weight(),
        null.total_abs_sqr()
    );
    println!("  distinguishable from 'nothing was there', which a complex amplitude cannot do.");
    println!(
        "  measurement refuses it: {:?}",
        null.measure(0, &mut Prng::new(1))
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
    survey::<SplitQuaternion>();
    survey::<Polarity<3>>();
    survey::<Quaternion>();
    survey::<Octonion>();
    survey::<Sedenion>();
}
