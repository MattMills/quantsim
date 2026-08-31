//! What the register is embedded in: PG(2n-1,2), its polar space
//! W(2n-1,2), the doily at n=2, and the effective geometry an assembly
//! of frames closes onto.
//!
//! Run with `cargo run --release --example polar_embedding`.

use quantsim::backend::{CliffordStep, PauliString};
use quantsim::polar::*;
use quantsim::retro::{Code, SurfaceCode, ToricCode};
use quantsim::stitch::Volume;
use quantsim::volqudit::VolQudit;

fn ps(x: u64, z: u64) -> PauliString {
    PauliString {
        x,
        z,
        negative: false,
    }
}

fn rule(t: &str) {
    println!("\n══ {t} ══\n");
}

fn name(p: &PauliString, n: usize) -> String {
    (0..n)
        .map(|q| match (p.x >> q & 1 == 1, p.z >> q & 1 == 1) {
            (true, true) => 'Y',
            (true, false) => 'X',
            (false, true) => 'Z',
            _ => '.',
        })
        .collect()
}

fn main() -> quantsim::Result<()> {
    rule("1. the embedding, and its closed forms");
    println!("   n   space        coords   points     ti-lines   lines/pt   generators   doilies");
    for n in 1..=6usize {
        let w = PolarSpace::new(n)?;
        let enumerated = if n <= 4 {
            format!("{}", w.enumerate_lines()?.len())
        } else {
            "-".into()
        };
        println!(
            "  {n:>2}   PG({:>2},2)      {:>3}   {:>8}   {:>8}   {:>8}   {:>10}   {:>7}",
            w.projective_dimension(),
            w.homogeneous_coordinates(),
            w.points(),
            w.totally_isotropic_lines(),
            w.lines_through_a_point(),
            w.generators(),
            w.doily_count()
        );
        if n <= 4 {
            assert_eq!(enumerated, format!("{}", w.totally_isotropic_lines()));
        }
    }
    println!(
        "\n  The Pauli group mod phase IS the point set of PG(2n-1,2), and commutation\n  \
         makes it the polar space W(2n-1,2). A `Volume` is a flat of it; an isotropic\n  \
         Volume is a flat of the polar space; a maximal one is a generator. Every\n  \
         line count above is checked against enumeration up to n = 4."
    );

    rule("2. n = 2 is the doily, and that is the unit");
    let w = PolarSpace::new(2)?;
    println!(
        "  W(3,2): {} points, {} totally isotropic lines,",
        w.points(),
        w.totally_isotropic_lines()
    );
    println!(
        "          3 points per line, {} lines per point,",
        w.lines_through_a_point()
    );
    println!("          triangles: {}", w.triangles()?);
    println!(
        "          GQ axiom (exactly one collinear point on every non-incident line): {}",
        w.gq_axiom()?
    );
    println!("  => the generalized quadrangle GQ(2,2). Four homogeneous coordinates.\n");
    println!("  the 15 points:");
    let pts = w.enumerate_points()?;
    for chunk in pts.chunks(5) {
        println!(
            "    {}",
            chunk
                .iter()
                .map(|p| name(p, 2))
                .collect::<Vec<_>>()
                .join("  ")
        );
    }
    println!("\n  a few of the 15 lines (each a maximal commuting set):");
    for l in w.enumerate_lines()?.iter().take(5) {
        println!(
            "    {{ {} }}",
            l.iter().map(|p| name(p, 2)).collect::<Vec<_>>().join(", ")
        );
    }

    rule("3. a wider register is a combinatorial space of doilies");
    for n in 2..=3usize {
        let w = PolarSpace::new(n)?;
        println!(
            "  n = {n}: {} doilies by enumeration, {} by closed form — {}",
            w.doilies()?,
            w.doily_count(),
            if w.doilies()? as u128 == w.doily_count() {
                "agree"
            } else {
                "DISAGREE"
            }
        );
    }
    println!(
        "  n = 4: {} by closed form (enumeration refused past 3)",
        PolarSpace::new(4)?.doily_count()
    );
    println!(
        "\n  And the doily is special: W(5,2) has {} triangles, so it is a polar space\n  \
         but not a generalized quadrangle. The GQ claim stops where it was checked.",
        PolarSpace::new(3)?.triangles()?
    );

    rule("4. every code in the crate is a flat of it");
    println!("  code            n    flat rank   flat points   ambient points   levels");
    for (label, n, gens) in [
        ("toric L=2", 8usize, ToricCode::new(2, 0)?.generators()),
        ("toric L=3", 18, ToricCode::new(3, 0)?.generators()),
        ("surface d=3", 9, SurfaceCode::new(3, 0)?.generators()),
        ("surface d=5", 25, SurfaceCode::new(5, 0)?.generators()),
    ] {
        let v = Volume::span(n, &gens)?;
        let q = VolQudit::new(n, v.clone())?;
        println!(
            "  {label:<14} {n:>2}   {:>9}   {:>11}   {:>14}   {:>6}",
            v.rank(),
            (1u128 << v.rank()) - 1,
            PolarSpace::new(n)?.points(),
            q.levels()
        );
    }

    rule("5. effective geometry: over the representation, not the state");
    let a = Assembly::new(
        4,
        vec![
            Volume::span(4, &[ps(0, 0b0011)])?,
            Volume::span(4, &[ps(0, 0b0110)])?,
            Volume::span(4, &[ps(0, 0b1100)])?,
        ],
    )?;
    println!("  three single-constraint frames on 4 qubits");
    println!(
        "    local logical ranks   {:?}   (sum {})",
        a.local_ranks()?,
        a.local_ranks()?.iter().sum::<usize>()
    );
    println!("    joined constraint     rank {}", a.joined()?.rank());
    println!(
        "    Im Γ = (⋁Vᵢ)^⊥        rank {}   idempotent {}",
        a.closure()?.rank(),
        a.is_idempotent()?
    );
    println!("    effective dimension   {}", a.effective_rank()?);
    println!(
        "    closure deficit       {}   <- what closing them together destroys",
        a.closure_deficit()?
    );
    println!(
        "    throat bound min hᵢ   {}   nested {}",
        a.throat()?,
        a.is_nested()
    );

    rule("6. the throat law, where it actually holds");
    let nested = Assembly::new(
        4,
        vec![
            Volume::span(4, &[ps(0, 0b0001)])?,
            Volume::span(4, &[ps(0, 0b0001), ps(0, 0b0010)])?,
            Volume::span(4, &[ps(0, 0b0001), ps(0, 0b0010), ps(0, 0b0100)])?,
        ],
    )?;
    println!(
        "  nested chain:  local {:?}  throat {}  effective {}  -> {}",
        nested.local_ranks()?,
        nested.throat()?,
        nested.effective_rank()?,
        if nested.effective_rank()? == nested.throat()? {
            "TIGHT"
        } else {
            "slack"
        }
    );
    let crossed = Assembly::new(
        4,
        vec![
            Volume::span(4, &[ps(0, 0b0011), ps(0, 0b1100)])?,
            Volume::span(4, &[ps(0b0011, 0), ps(0b1100, 0)])?,
        ],
    )?;
    println!(
        "  crossed pair:  local {:?}  throat {}  effective {}  -> {}",
        crossed.local_ranks()?,
        crossed.throat()?,
        crossed.effective_rank()?,
        if crossed.effective_rank()? == crossed.throat()? {
            "tight"
        } else {
            "SLACK"
        }
    );
    println!(
        "\n  The hourglass law is a BOUND in general and an equality exactly on a\n  \
         nested chain, where each stage's constraint already contains the last.\n  \
         Reported as an inequality rather than asserted as a slogan."
    );

    rule("7. which invariants survive the closure");
    let x = Assembly::new(
        3,
        vec![
            Volume::span(3, &[ps(0, 0b011)])?,
            Volume::span(3, &[ps(0, 0b110)])?,
        ],
    )?;
    let y = Assembly::new(3, vec![Volume::span(3, &[ps(0, 0b011), ps(0, 0b110)])?])?;
    println!(
        "  two assemblies with the same closure: {}",
        x.closure()?.is_same(&y.closure()?)
    );
    println!(
        "    effective rank    {} vs {}  -> factors through Γ, globally effective",
        x.effective_rank()?,
        y.effective_rank()?
    );
    println!(
        "    Σ local ranks     {} vs {}  -> does NOT factor, a local invariant only",
        x.local_ranks()?.iter().sum::<usize>(),
        y.local_ranks()?.iter().sum::<usize>()
    );

    rule("8. dynamic sufficiency");
    let v = Volume::span(3, &[ps(0, 0b011), ps(0, 0b110)])?;
    let q = VolQudit::new(3, v.clone())?;
    for (label, steps) in [
        ("identity", vec![]),
        (
            "Cx(0,1) Cx(1,0) Cx(0,1)",
            vec![
                CliffordStep::Cx(0, 1),
                CliffordStep::Cx(1, 0),
                CliffordStep::Cx(0, 1),
            ],
        ),
        ("H(0)", vec![CliffordStep::H(0)]),
        (
            "H(0) H(1) H(2)",
            vec![CliffordStep::H(0), CliffordStep::H(1), CliffordStep::H(2)],
        ),
    ] {
        let phi = dynamic_obstruction(&v, &steps)?;
        println!("  Φ = {phi}   loop {}   {label}", q.closes(&steps)?);
    }
    println!(
        "\n  Φ(P,U) = P U (I-P) is zero exactly when the frame is dynamically closed —\n  \
         the same predicate `VolQudit::closes` answers, now with the frame-geometry\n  \
         name and a rank attached instead of a bool. A nonzero Φ names the directions\n  \
         the frame does not carry but the dynamics will bring back into it."
    );
    Ok(())
}
