//! Krein-collision stability, measured: the eigensolver against known
//! spectra; refusal of generators that are not η-self-adjoint; the
//! two-mode squeezing instability band as an opposite-signature
//! collision and the beam splitter's immunity; the Bogoliubov ghost
//! weight against an independent Fock-space simulation; the stability
//! margin tracking the opposite-signature gap, not the spectral gap; the
//! PT dimer and its exceptional point on a one-qubit Krein register; and
//! sp(2n, ℝ) read off the Killing form, at n = 2 the AdS₄ algebra.

mod common;

use quantsim::gates::Pauli;
use quantsim::krein::{infinity_twistor, killing_signature, killing_signature_sites, stabilizer};
use quantsim::krein_sites::so_algebra;
use quantsim::krein_stability::{
    bogoliubov, bogoliubov_algebra, bogoliubov_metric, bogoliubov_signs, eigenvalues, evolution,
    heisenberg_generator, instability_threshold, krein_spectrum, KreinType, Stability,
    STABILITY_TOL,
};
use quantsim::prelude::*;

fn z(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn mat(d: usize, entries: &[C64]) -> SiteMatrix<C64> {
    SiteMatrix::from_vec(d, entries.to_vec()).unwrap()
}

fn diag(v: &[f64]) -> SiteMatrix<C64> {
    let d = v.len();
    let mut data = vec![z(0.0, 0.0); d * d];
    for (k, &x) in v.iter().enumerate() {
        data[k * d + k] = z(x, 0.0);
    }
    mat(d, &data)
}

fn zeros(d: usize) -> SiteMatrix<C64> {
    mat(d, &vec![z(0.0, 0.0); d * d])
}

fn sig(s: FormSignature) -> (usize, usize, usize) {
    (s.positive, s.negative, s.null)
}

/// `h = [[ωa, κ], [κ̄, ωb]]`, `g = [[ga, gab], [gab, gb]]`.
fn two_mode(wa: f64, wb: f64, k: C64, ga: C64, gb: C64, gab: C64) -> SiteMatrix<C64> {
    let h = mat(2, &[z(wa, 0.0), k, k.conj(), z(wb, 0.0)]);
    let g = mat(2, &[ga, gab, gab, gb]);
    bogoliubov(&h, &g).unwrap()
}

fn operator_norm(p: &SiteMatrix<C64>) -> f64 {
    let d = p.dim();
    let pd = mat(
        d,
        &(0..d * d)
            .map(|k| p.get(k % d, k / d).conj())
            .collect::<Vec<_>>(),
    );
    let ev = eigenvalues(&pd.matmul(p).unwrap()).unwrap();
    ev.iter().fold(0.0f64, |m, x| m.max(x.re)).sqrt()
}

/// Every value of `want` is matched by a distinct value of `got`.
fn assert_same_spectrum(got: &[C64], want: &[C64], tol: f64) {
    assert_eq!(got.len(), want.len());
    let mut used = vec![false; got.len()];
    for w in want {
        let (k, dist) = got
            .iter()
            .enumerate()
            .filter(|(k, _)| !used[*k])
            .map(|(k, g)| (k, (g - w).norm()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert!(
            dist < tol,
            "{w} unmatched (nearest at {dist:.3e}) in {got:?}"
        );
        used[k] = true;
    }
}

// ── the eigensolver ──────────────────────────────────────────────────

#[test]
fn eigenvalues_match_known_spectra() {
    // Upper triangular: the diagonal.
    let d = [
        z(1.0, 2.0),
        z(-0.5, 0.0),
        z(3.0, -1.0),
        z(0.0, 0.25),
        z(2.0, 2.0),
    ];
    let mut data = vec![z(0.0, 0.0); 25];
    let mut rng = Prng::new(7);
    for r in 0..5 {
        data[r * 5 + r] = d[r];
        for c in r + 1..5 {
            data[r * 5 + c] = z(rng.next_f64() - 0.5, rng.next_f64() - 0.5);
        }
    }
    // Conjugate by a random dense similarity so nothing is triangular.
    let s = mat(
        5,
        &(0..25)
            .map(|k| {
                z(
                    rng.next_f64() + if k % 6 == 0 { 2.0 } else { 0.0 },
                    rng.next_f64(),
                )
            })
            .collect::<Vec<_>>(),
    );
    let t = mat(5, &data);
    assert_same_spectrum(&eigenvalues(&t).unwrap(), &d, 1e-12);
    // ST and TS share their spectrum.
    let st = eigenvalues(&s.matmul(&t).unwrap()).unwrap();
    let ts = eigenvalues(&t.matmul(&s).unwrap()).unwrap();
    assert_same_spectrum(&st, &ts, 1e-9);

    // Companion matrix of Π(x − r_k): the roots.
    let roots = [z(1.0, 1.0), z(1.0, -1.0), z(-2.0, 0.0), z(0.5, 3.0)];
    let mut coef = vec![z(1.0, 0.0)];
    for r in roots {
        let mut next = vec![z(0.0, 0.0); coef.len() + 1];
        for (k, &ck) in coef.iter().enumerate() {
            next[k] += ck;
            next[k + 1] -= ck * r;
        }
        coef = next;
    }
    let n = roots.len();
    let mut comp = vec![z(0.0, 0.0); n * n];
    for k in 0..n {
        comp[k] = -coef[k + 1];
        if k + 1 < n {
            comp[(k + 1) * n + k] = z(1.0, 0.0);
        }
    }
    assert_same_spectrum(&eigenvalues(&mat(n, &comp)).unwrap(), &roots, 1e-10);

    // Random Hermitian: real spectrum, trace preserved.
    let n = 8;
    let mut h = vec![z(0.0, 0.0); n * n];
    for r in 0..n {
        for c in r..n {
            let v = if r == c {
                z(rng.next_f64() * 4.0 - 2.0, 0.0)
            } else {
                z(rng.next_f64() - 0.5, rng.next_f64() - 0.5)
            };
            h[r * n + c] = v;
            h[c * n + r] = v.conj();
        }
    }
    let ev = eigenvalues(&mat(n, &h)).unwrap();
    let tr: f64 = (0..n).map(|k| h[k * n + k].re).sum();
    assert!(ev.iter().all(|x| x.im.abs() < 1e-10));
    common::assert_close(ev.iter().map(|x| x.re).sum::<f64>(), tr, 1e-10);
}

// ── refusals and the trivial cases ───────────────────────────────────

#[test]
fn a_generator_that_breaks_the_metric_is_refused() {
    let eta = bogoliubov_metric(1);
    // i·σ_x *is* σ_z-self-adjoint (its evolution is a boost); i·1 is not.
    let boost = mat(2, &[z(0.0, 0.0), z(0.0, 1.0), z(0.0, 1.0), z(0.0, 0.0)]);
    assert!(krein_spectrum(&boost, &eta, STABILITY_TOL).is_ok());
    let bad = mat(2, &[z(0.0, 1.0), z(0.0, 0.0), z(0.0, 0.0), z(0.0, 1.0)]);
    let err = krein_spectrum(&bad, &eta, STABILITY_TOL).unwrap_err();
    assert!(format!("{err}").contains("not η-self-adjoint"));
    assert!(bogoliubov(
        &diag(&[1.0, 2.0]),
        &mat(2, &[z(0.0, 0.0), z(1.0, 0.0), z(0.0, 0.0), z(0.0, 0.0)])
    )
    .is_err());
}

#[test]
fn the_zero_generator_on_an_indefinite_metric_is_only_weakly_stable() {
    let eta = bogoliubov_metric(2);
    let spec = krein_spectrum(&zeros(4), &eta, STABILITY_TOL).unwrap();
    assert_eq!(spec.stability, Stability::Weak);
    assert_eq!(spec.clusters.len(), 1);
    let k = &spec.clusters[0];
    assert_eq!(
        (k.multiplicity, k.geometric, k.positive, k.negative),
        (4, 4, 2, 2)
    );
    assert_eq!(k.kind, KreinType::Mixed);
    // The same generator with a definite metric is strongly stable.
    let spec = krein_spectrum(&zeros(4), &diag(&[1.0; 4]), STABILITY_TOL).unwrap();
    assert_eq!(
        spec.stability,
        Stability::Strong {
            opposite_gap: f64::INFINITY
        }
    );
}

// ── squeezing and beam splitting ─────────────────────────────────────

#[test]
fn the_two_mode_squeezing_band_is_an_opposite_signature_collision() {
    let eta = bogoliubov_metric(2);
    let o = z(0.0, 0.0);
    let g = 0.4;
    let mut unstable_seen = 0;
    for k in 0..41 {
        let sum = -1.2 + 0.06 * k as f64; // ω_a + ω_b
        let (wa, wb) = (0.9, sum - 0.9);
        let d = two_mode(wa, wb, o, o, o, z(g, 0.0));
        let spec = krein_spectrum(&d, &eta, STABILITY_TOL).unwrap();
        let half = 0.5 * sum;
        if half.abs() < g - 1e-9 {
            unstable_seen += 1;
            let want = (g * g - half * half).sqrt();
            match spec.stability {
                Stability::Unstable { growth_rate } => {
                    common::assert_close(growth_rate, want, 1e-9)
                }
                s => panic!("Σω = {sum}: expected unstable, got {s:?}"),
            }
        } else if half.abs() > g + 1e-9 {
            assert!(
                matches!(spec.stability, Stability::Strong { .. }),
                "Σω = {sum}: {:?}",
                spec.stability
            );
            // Two positive and two negative clusters.
            let pos = spec
                .clusters
                .iter()
                .filter(|c| c.kind == KreinType::Positive)
                .count();
            let neg = spec
                .clusters
                .iter()
                .filter(|c| c.kind == KreinType::Negative)
                .count();
            assert_eq!((pos, neg), (2, 2));
        }
    }
    assert_eq!(unstable_seen, 27);
}

#[test]
fn the_beam_splitter_never_destabilizes_even_on_a_degeneracy() {
    let eta = bogoliubov_metric(2);
    let o = z(0.0, 0.0);
    for wb in [0.3, 1.0, 1.0 + 1e-9, 2.5] {
        for k in 0..20 {
            let kappa = z(0.2 * k as f64, 0.1 * k as f64);
            let d = two_mode(1.0, wb, kappa, o, o, o);
            let spec = krein_spectrum(&d, &eta, STABILITY_TOL).unwrap();
            // With large κ one normal mode goes negative in frequency but
            // keeps its positive signature: no collision of opposite
            // signature happens, so nothing goes unstable.
            assert!(
                matches!(spec.stability, Stability::Strong { .. }),
                "ω_b = {wb}, κ = {kappa}: {:?}",
                spec.stability
            );
        }
    }
    // At exact degeneracy and κ = 0 the two + modes share a cluster: a
    // definite double eigenvalue, still strongly stable.
    let spec = krein_spectrum(&two_mode(1.0, 1.0, o, o, o, o), &eta, STABILITY_TOL).unwrap();
    let top = spec.clusters.last().unwrap();
    assert_eq!(
        (top.multiplicity, top.positive, top.kind),
        (2, 2, KreinType::Positive)
    );
    assert_eq!(spec.stability, Stability::Strong { opposite_gap: 2.0 });
}

// ── the ghost sector is a particle number ───────────────────────────

/// `⟨a†a⟩(t)` from the vacuum, by brute force in a truncated two-mode
/// Fock space (an independent representation of the same Hamiltonian).
fn fock_occupation(wa: f64, wb: f64, k: C64, ga: C64, gb: C64, gab: C64, t: f64) -> f64 {
    let n = 12;
    let dim = n * n;
    let idx = |p: usize, q: usize| p * n + q;
    let mut h = vec![z(0.0, 0.0); dim * dim];
    let mut add = |r: usize, c: usize, v: C64| h[r * dim + c] += v;
    for p in 0..n {
        for q in 0..n {
            let col = idx(p, q);
            add(col, col, z(wa * p as f64 + wb * q as f64, 0.0));
            // κ a†b + κ̄ b†a
            if q >= 1 && p + 1 < n {
                let amp = ((p + 1) as f64 * q as f64).sqrt();
                add(idx(p + 1, q - 1), col, k * amp);
                add(col, idx(p + 1, q - 1), k.conj() * amp);
            }
            // ½ga a†² + ½gb b†² + gab a†b† and h.c.
            if p + 2 < n {
                let amp = 0.5 * ((p + 1) as f64 * (p + 2) as f64).sqrt();
                add(idx(p + 2, q), col, ga * amp);
                add(col, idx(p + 2, q), ga.conj() * amp);
            }
            if q + 2 < n {
                let amp = 0.5 * ((q + 1) as f64 * (q + 2) as f64).sqrt();
                add(idx(p, q + 2), col, gb * amp);
                add(col, idx(p, q + 2), gb.conj() * amp);
            }
            if p + 1 < n && q + 1 < n {
                let amp = ((p + 1) as f64 * (q + 1) as f64).sqrt();
                add(idx(p + 1, q + 1), col, gab * amp);
                add(col, idx(p + 1, q + 1), gab.conj() * amp);
            }
        }
    }
    let u = evolution(&mat(dim, &h), t).unwrap();
    // Column 0 is U|vac⟩.
    (0..dim)
        .map(|r| (r / n) as f64 * u.get(r, 0).norm_sqr())
        .sum()
}

#[test]
fn the_ghost_weight_is_the_number_of_created_quanta() {
    let (wa, wb) = (1.0, 0.7);
    let (k, ga, gb, gab) = (z(0.3, 0.1), z(0.15, 0.0), z(0.0, 0.1), z(0.25, -0.05));
    let d = two_mode(wa, wb, k, ga, gb, gab);
    let space = Space::new(&[4]).unwrap();
    let metric = SiteMetric::new(space.clone())
        .signs(0, &bogoliubov_signs(2))
        .unwrap();
    let mut s = BackendRegistry::<C64>::standard()
        .create_with_space("dense", &space)
        .unwrap();
    s.load(&[(0, z(1.0, 0.0))]).unwrap();
    let mut reg = SiteKreinRegister::new(s, metric).unwrap();
    let dt = 0.2;
    let step = evolution(&heisenberg_generator(&d), dt).unwrap();
    assert!(step.unitarity_deviation() > 1e-3, "a non-unitary gate");
    for k_step in 1..=5 {
        reg.apply_sites(&step, &[0]).unwrap();
        let led = reg.ledger().unwrap();
        common::assert_close(led.eta, 1.0, 1e-10);
        let n_a = fock_occupation(wa, wb, k, ga, gb, gab, dt * k_step as f64);
        assert!(n_a > 1e-3, "quanta are created");
        common::assert_close(led.negative, n_a, 1e-9);
    }
}

#[test]
fn the_register_grows_at_twice_the_classified_rate() {
    let o = z(0.0, 0.0);
    let d = two_mode(0.3, -0.1, o, o, o, z(0.5, 0.0));
    let eta = bogoliubov_metric(2);
    let rate = match krein_spectrum(&d, &eta, STABILITY_TOL).unwrap().stability {
        Stability::Unstable { growth_rate } => growth_rate,
        s => panic!("{s:?}"),
    };
    common::assert_close(rate, (0.25f64 - 0.01).sqrt(), 1e-10);
    let space = Space::new(&[4]).unwrap();
    let metric = SiteMetric::new(space.clone())
        .signs(0, &bogoliubov_signs(2))
        .unwrap();
    let mut s = BackendRegistry::<C64>::standard()
        .create_with_space("dense", &space)
        .unwrap();
    s.load(&[(0, z(1.0, 0.0))]).unwrap();
    let mut reg = SiteKreinRegister::new(s, metric)
        .unwrap()
        .with_tolerance(1e-7);
    let step = evolution(&heisenberg_generator(&d), 1.0).unwrap();
    let mut paths = Vec::new();
    for _ in 0..20 {
        reg.apply_sites(&step, &[0]).unwrap();
        let led = reg.ledger().unwrap();
        common::assert_close(led.eta, 1.0, 1e-9 * led.path);
        paths.push(led.path);
    }
    let fitted = (paths[19] / paths[9]).ln() / 10.0;
    common::assert_close(fitted, 2.0 * rate, 1e-6);
}

// ── the margin ───────────────────────────────────────────────────────

fn random_direction(rng: &mut Prng) -> SiteMatrix<C64> {
    let mut r = || z(rng.next_f64() - 0.5, rng.next_f64() - 0.5);
    let k = r();
    let h = mat(2, &[z(r().re, 0.0), k, k.conj(), z(r().re, 0.0)]);
    let gab = r();
    let g = mat(2, &[r(), gab, gab, r()]);
    let p = bogoliubov(&h, &g).unwrap();
    let norm = operator_norm(&p);
    mat(4, &p.data().iter().map(|x| x / norm).collect::<Vec<_>>())
}

#[test]
fn the_stability_margin_is_the_opposite_signature_gap() {
    let eta = bogoliubov_metric(2);
    let o = z(0.0, 0.0);
    let squeeze_ab = two_mode(0.0, 0.0, o, o, o, z(1.0, 0.0));
    common::assert_close(operator_norm(&squeeze_ab), 1.0, 1e-12);
    let mut rng = Prng::new(2026);
    let directions: Vec<SiteMatrix<C64>> = (0..64).map(|_| random_direction(&mut rng)).collect();
    for delta in [0.5, 1e-1, 1e-2, 1e-3] {
        // Family A: two + modes δ apart (tiny spectral gap), opposite
        // gap 2. Family B: ω_b = −1 + δ, opposite gap δ.
        for (wb, gap) in [(1.0 + delta, 2.0), (-1.0 + delta, delta)] {
            let d0 = two_mode(1.0, wb, o, o, o, o);
            match krein_spectrum(&d0, &eta, STABILITY_TOL).unwrap().stability {
                Stability::Strong { opposite_gap } => common::assert_close(opposite_gap, gap, 1e-9),
                s => panic!("{s:?}"),
            }
            let mut least = f64::INFINITY;
            for p in &directions {
                if let Some(e) = instability_threshold(&d0, p, &eta, 3.0 * gap, 24).unwrap() {
                    least = least.min(e);
                }
            }
            assert!(
                least >= 0.5 * gap * (1.0 - 1e-6),
                "ω_b = {wb}: a ray destabilized at {least} < gap/2 = {}",
                0.5 * gap
            );
            // The closest opposite pair, pair-squeezed, attains gap/2.
            let tight = if wb > 0.0 {
                two_mode(0.0, 0.0, o, z(1.0, 0.0), o, o) // a† a† at ±ω_a
            } else {
                squeeze_ab.clone()
            };
            let e = instability_threshold(&d0, &tight, &eta, 3.0 * gap, 60)
                .unwrap()
                .unwrap();
            // (Detection needs Im λ above the 1e-6 real-axis tolerance,
            // which moves the measured threshold by ~1e-12/gap.)
            common::assert_close(e, 0.5 * gap, 1e-5 * gap);
        }
    }
}

// ── the PT dimer ─────────────────────────────────────────────────────

fn pt_dimer(gamma: f64, kappa: f64) -> SiteMatrix<C64> {
    mat(
        2,
        &[z(0.0, gamma), z(kappa, 0.0), z(kappa, 0.0), z(0.0, -gamma)],
    )
}

#[test]
fn the_pt_dimer_three_phases_and_its_exceptional_point() {
    let x = mat(2, &[z(0.0, 0.0), z(1.0, 0.0), z(1.0, 0.0), z(0.0, 0.0)]);
    let kappa = 1.0;
    for gamma in [0.0, 0.3, 0.9, 0.999] {
        let spec = krein_spectrum(&pt_dimer(gamma, kappa), &x, STABILITY_TOL).unwrap();
        let w = (kappa * kappa - gamma * gamma).sqrt();
        match spec.stability {
            Stability::Strong { opposite_gap } => common::assert_close(opposite_gap, 2.0 * w, 1e-9),
            s => panic!("γ = {gamma}: {s:?}"),
        }
        let kinds: Vec<KreinType> = spec.clusters.iter().map(|c| c.kind).collect();
        assert!(kinds.contains(&KreinType::Positive) && kinds.contains(&KreinType::Negative));
    }
    let ep = krein_spectrum(&pt_dimer(1.0, kappa), &x, STABILITY_TOL).unwrap();
    assert_eq!(ep.stability, Stability::Marginal);
    let c = &ep.clusters[0];
    assert_eq!(
        (c.multiplicity, c.geometric, c.positive, c.negative),
        (2, 1, 1, 1)
    );
    for gamma in [1.01, 1.5, 3.0] {
        match krein_spectrum(&pt_dimer(gamma, kappa), &x, STABILITY_TOL)
            .unwrap()
            .stability
        {
            Stability::Unstable { growth_rate } => {
                common::assert_close(growth_rate, (gamma * gamma - 1.0).sqrt(), 1e-9)
            }
            s => panic!("γ = {gamma}: {s:?}"),
        }
    }

    // On a one-qubit Krein register with η = X: at the exceptional point
    // exp(−iHt) = 1 − iHt, and from |0⟩ the path weight is 1 + 2t + 2t².
    let metric = KreinMetric::new(&[(0, Pauli::X)]).unwrap();
    let mut s = BackendRegistry::<C64>::standard()
        .create("dense", 1)
        .unwrap();
    s.load(&[(0, z(1.0, 0.0))]).unwrap();
    let mut reg = KreinRegister::new(s, metric).unwrap();
    let dt = 0.25;
    let gate = evolution(&pt_dimer(1.0, kappa), dt)
        .unwrap()
        .to_gate()
        .unwrap();
    for k in 1..=12 {
        reg.apply(&gate, &[0]).unwrap();
        let t = dt * k as f64;
        let led = reg.ledger().unwrap();
        common::assert_close(led.path, 1.0 + 2.0 * t + 2.0 * t * t, 1e-9);
        common::assert_close(led.eta, 0.0, 1e-9);
    }
}

// ── sp(2n, ℝ) and AdS₄ ───────────────────────────────────────────────

#[test]
fn the_bogoliubov_algebra_is_sp_2n_and_at_two_modes_the_ads4_algebra() {
    for n in 1..=3 {
        let alg = bogoliubov_algebra(n).unwrap();
        assert_eq!(alg.len(), n * (2 * n + 1));
        let eta = bogoliubov_metric(n);
        for x in &alg {
            // −iD is η-anti-self-adjoint: exp(θ·x) is pseudo-unitary.
            let ix = mat(
                2 * n,
                &x.data().iter().map(|v| v * z(0.0, 1.0)).collect::<Vec<_>>(),
            );
            assert!(
                quantsim::krein_stability::eta_selfadjoint_deviation(&ix, &eta).unwrap() < 1e-12
            );
        }
        assert_eq!(
            sig(killing_signature_sites(&alg).unwrap()),
            (n * (n + 1), n * n, 0),
            "n = {n}"
        );
    }
    let two = sig(killing_signature_sites(&bogoliubov_algebra(2).unwrap()).unwrap());
    let ads4 = sig(killing_signature(&stabilizer(&infinity_twistor(z(-1.0, 0.0)))).unwrap());
    let so32 = sig(killing_signature_sites(&so_algebra(&[1, 1, 1, -1, -1]).unwrap()).unwrap());
    assert_eq!(two, (6, 4, 0));
    assert_eq!(two, ads4);
    assert_eq!(two, so32);
    // And n = 1 is su(1,1) ≅ so(2,1): AdS₂'s algebra.
    let one = sig(killing_signature_sites(&bogoliubov_algebra(1).unwrap()).unwrap());
    let so21 = sig(killing_signature_sites(&so_algebra(&[1, 1, -1]).unwrap()).unwrap());
    assert_eq!(one, so21);
}
