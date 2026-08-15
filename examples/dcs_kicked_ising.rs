//! Is the DCS brickwork the **self-dual kicked Ising model**?
//!
//! The claim to test: `Dcs`'s two-qubit gate is the Ising coupling at
//! `J = π/4` and its single-qubit gate is the transverse kick at
//! `b = π/4` — the self-dual point of the kicked Ising model. Everything
//! here is checked against the matrices the crate actually ships in
//! `src/gates/standard.rs`, pulled through the registry rather than
//! retyped.

use quantsim::math::{c64, cis, GateMatrix};
use quantsim::registry::GateRegistry;
use quantsim::scalar::C64;
use std::f64::consts::{FRAC_PI_4, PI};

fn g(reg: &GateRegistry<C64>, name: &str) -> GateMatrix<C64> {
    reg.resolve(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .matrix(&[])
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// `exp(−iθP)` for an involution `P` (`P² = I`): `cos θ·I − i sin θ·P`.
fn expp(p: &GateMatrix<C64>, theta: f64) -> GateMatrix<C64> {
    let d = p.dim();
    let (c, s) = (theta.cos(), theta.sin());
    let mut data = Vec::with_capacity(d * d);
    for r in 0..d {
        for col in 0..d {
            let id = if r == col { c64(c, 0.0) } else { c64(0.0, 0.0) };
            data.push(id + c64(0.0, -s) * p.get(r, col));
        }
    }
    GateMatrix::from_vec(d, data).expect("dim is a power of two")
}

fn scale(m: &GateMatrix<C64>, z: C64) -> GateMatrix<C64> {
    GateMatrix::from_vec(m.dim(), m.data().iter().map(|&e| z * e).collect()).expect("same dim")
}

/// Largest entrywise `|a − b|`.
fn dev(a: &GateMatrix<C64>, b: &GateMatrix<C64>) -> f64 {
    a.data()
        .iter()
        .zip(b.data())
        .fold(0.0f64, |w, (&x, &y)| w.max((x - y).norm()))
}

/// Largest entrywise deviation after aligning the best global phase —
/// so "equal as gates" is tested, not "equal as matrices".
fn dev_up_to_phase(a: &GateMatrix<C64>, b: &GateMatrix<C64>) -> f64 {
    let (mut best, mut idx) = (0.0f64, 0usize);
    for (i, e) in a.data().iter().enumerate() {
        if e.norm() > best {
            best = e.norm();
            idx = i;
        }
    }
    if best == 0.0 {
        return dev(a, b);
    }
    let u = b.data()[idx] * a.data()[idx].conj() / (best * best);
    dev(&scale(a, u), b)
}

fn ident() -> GateMatrix<C64> {
    GateMatrix::identity(2).expect("2 is a power of two")
}

fn main() {
    let reg = GateRegistry::<C64>::standard();
    let (cz, sx, s, sdg, t) = (
        g(&reg, "cz"),
        g(&reg, "sx"),
        g(&reg, "s"),
        g(&reg, "sdg"),
        g(&reg, "t"),
    );
    let (x, z) = (g(&reg, "x"), g(&reg, "z"));
    let zz = z.kron(&z);
    let i2 = ident();

    println!("IS THE DCS BRICKWORK THE SELF-DUAL KICKED ISING MODEL?");
    println!("  gates read from the registry (src/gates/standard.rs)");
    println!();
    println!("  cz   = diag({:?})", diag(&cz));
    println!("  sx   = {:?}", rows(&sx));
    println!(
        "  involutions: |X²−I| = {:.1e},  |(Z⊗Z)²−I| = {:.1e}",
        dev(&x.matmul(&x), &i2),
        dev(&zz.matmul(&zz), &GateMatrix::identity(4).unwrap())
    );

    // ── the kick: b = π/4 ────────────────────────────────────────────
    println!();
    println!("1. THE KICK.  √X  ?=  e^{{iπ/4}} · exp(−i b X)   at b = π/4");
    let kick = expp(&x, FRAC_PI_4);
    println!(
        "   |√X − e^{{iπ/4}}exp(−i(π/4)X)|            = {:.3e}",
        dev(&sx, &scale(&kick, cis(FRAC_PI_4)))
    );
    println!("   b scan (deviation up to global phase):");
    for k in 1..=8 {
        let b = PI * k as f64 / 16.0;
        println!(
            "     b = {:>6.4} ({:>4}π): {:.3e}{}",
            b,
            format!("{}/16", k),
            dev_up_to_phase(&sx, &expp(&x, b)),
            if (b - FRAC_PI_4).abs() < 1e-12 {
                "   ← self-dual value"
            } else {
                ""
            }
        );
    }

    // ── the coupling: J = π/4 ────────────────────────────────────────
    println!();
    println!("2. THE COUPLING.  CZ  ?=  e^{{iπ/4}} · (S†⊗S†) · exp(−i J Z⊗Z)   at J = π/4");
    let ising = expp(&zz, FRAC_PI_4);
    let rebuilt = scale(&sdg.kron(&sdg).matmul(&ising), cis(FRAC_PI_4));
    println!(
        "   |CZ − e^{{iπ/4}}(S†⊗S†)exp(−i(π/4)Z⊗Z)|  = {:.3e}",
        dev(&cz, &rebuilt)
    );
    println!(
        "   and the inverse reading: |exp(−i(π/4)Z⊗Z) − e^{{−iπ/4}}(S⊗S)·CZ| = {:.3e}",
        dev(&ising, &scale(&s.kron(&s).matmul(&cz), cis(-FRAC_PI_4)))
    );

    // Which J admit *any* single-qubit dressing at all? CZ·exp(+iJZ⊗Z)
    // is diagonal; a diagonal d factors as A⊗B exactly when
    // d₀₀d₁₁ = d₀₁d₁₀. That defect vanishes only at J = π/4 mod π/2.
    println!();
    println!("   J scan — CZ·exp(+i J Z⊗Z) factors as A⊗B iff d₀₀d₁₁ = d₀₁d₁₀:");
    for k in 0..=8 {
        let j = PI * k as f64 / 16.0;
        let d = cz.matmul(&expp(&zz, -j));
        let defect = (d.get(0, 0) * d.get(3, 3) - d.get(1, 1) * d.get(2, 2)).norm();
        println!(
            "     J = {:>6.4} ({:>5}π): factorization defect {:.3e}{}",
            j,
            format!("{}/16", k),
            defect,
            if defect < 1e-12 {
                "   ← the only J that works (mod π/2)"
            } else {
                ""
            }
        );
    }

    // ── one whole DCS layer against the KIM Floquet operator ─────────
    println!();
    println!("3. ONE WHOLE DCS LAYER (2 qubits) vs the KIM Floquet operator");
    println!("   DCS layer, circuit order: CZ(0,1);  then per qubit  [S if r_q]  ·  √X");
    println!("   KIM: U = exp(−i b ΣX)·exp(−i Σ h_q Z_q)·exp(−i J Z⊗Z),  J = b = π/4");
    println!("   r_q  h_0     h_1    | deviation up to phase");
    for r0 in 0..2 {
        for r1 in 0..2 {
            // Matrix order is the reverse of circuit order.
            let sq = |r: usize| if r == 1 { s.clone() } else { i2.clone() };
            let layer = sx.kron(&sx).matmul(&sq(r1).kron(&sq(r0))).matmul(&cz);
            // S = e^{iπ/4}exp(−i(π/4)Z) and CZ carries an S† per qubit,
            // so the residual longitudinal kick is S^{r−1}: none when
            // r = 1, and −π/4 when r = 0.
            let h = |r: usize| if r == 1 { 0.0 } else { -FRAC_PI_4 };
            let long = expp(&z, h(r1)).kron(&expp(&z, h(r0)));
            let kim = expp(&x, FRAC_PI_4)
                .kron(&expp(&x, FRAC_PI_4))
                .matmul(&long)
                .matmul(&expp(&zz, FRAC_PI_4));
            println!(
                "   {r0}{r1}   {:>6.4}  {:>6.4} | {:.3e}",
                h(r0),
                h(r1),
                dev_up_to_phase(&layer, &kim)
            );
        }
    }

    // ── the dopant is a longitudinal field too ───────────────────────
    println!();
    println!("4. THE DOPANT.  T ?= e^{{iπ/8}}·exp(−i(π/8)Z) — a longitudinal kick of π/8");
    println!(
        "   |T − e^{{iπ/8}}exp(−i(π/8)Z)|             = {:.3e}",
        dev(&t, &scale(&expp(&z, PI / 8.0), cis(PI / 8.0)))
    );
    println!(
        "   |S − e^{{iπ/4}}exp(−i(π/4)Z)|             = {:.3e}   (the Clifford kick is twice it)",
        dev(&s, &scale(&expp(&z, FRAC_PI_4), cis(FRAC_PI_4)))
    );
}

fn diag(m: &GateMatrix<C64>) -> Vec<String> {
    (0..m.dim()).map(|i| fmt(m.get(i, i))).collect()
}

fn rows(m: &GateMatrix<C64>) -> Vec<String> {
    m.data().iter().map(|&e| fmt(e)).collect()
}

fn fmt(z: C64) -> String {
    format!("{:+.3}{:+.3}i", z.re, z.im)
}
