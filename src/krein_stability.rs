//! Krein-collision stability: which evolutions of a Krein register stay
//! bounded, why, and how far a bounded one is from blowing up.
//!
//! [`crate::krein`] and [`crate::krein_sites`] build registers whose
//! inner product is indefinite and whose gates are pseudo-unitary,
//! `M†ηM = η`. A continuous evolution `M(t) = exp(−iHt)` is pseudo-unitary
//! for every `t` exactly when the generator is **`η`-self-adjoint**,
//! `ηH = H†η`. Such an `H` need not be Hermitian, and its spectrum need
//! not be real: it is symmetric about the real axis, and a pair
//! `λ, λ̄` off the axis makes the path weight grow like `e^{2 Im λ t}`
//! while the `η`-norm stays exactly conserved. Whether a register's
//! evolution is bounded is therefore a spectral question, and Krein's
//! theory answers it with one extra datum per real eigenvalue: its
//! **Krein signature**, the inertia of `η` restricted to its generalized
//! eigenspace.
//!
//! * A real eigenvalue whose eigenspace is `η`-definite (all `+` or all
//!   `−`) cannot leave the real axis under a small `η`-self-adjoint
//!   perturbation. Eigenvalues of the **same** signature may meet and pass
//!   (or repel); they cannot split into a complex pair.
//! * Only a collision of **opposite** signatures can produce a complex
//!   pair. The complex eigenvalues that result are `η`-**neutral**.
//! * So a generator whose real eigenvalues are all definite is
//!   *strongly stable*: a whole neighbourhood of it is stable, and the
//!   size of that neighbourhood is governed by the distance between
//!   eigenvalues of opposite signature — not by the spectral gap.
//!
//! ## What is here
//!
//! * [`eigenvalues`] — the complex spectrum of any square [`SiteMatrix`]
//!   (Householder–Hessenberg reduction, then shifted QR with Wilkinson
//!   shifts and Givens rotations). Dependency-free; meant for the small
//!   generators a register's site carries.
//! * [`krein_spectrum`] — clusters the spectrum, and for each cluster
//!   reports algebraic and geometric multiplicity and the inertia of `η`
//!   on its generalized eigenspace; then classifies the generator as a
//!   [`Stability`]: `Strong` (with the opposite-signature gap),
//!   `Weak` (semisimple but sitting on an opposite-signature collision),
//!   `Marginal` (a Jordan block on the real axis: polynomial growth, an
//!   exceptional point) or `Unstable` (with the growth rate `max Im λ`).
//!   Generators that are not `η`-self-adjoint are refused by name.
//! * [`bogoliubov`] — the dynamical matrix `D = [[h, g], [−ḡ, −h̄]]` of a
//!   quadratic bosonic Hamiltonian `Σ h_ij a_i†a_j + ½Σ(g_ij a_i†a_j† +
//!   h.c.)`, `η = diag(1ₙ, −1ₙ)`: `η`-self-adjoint, so its evolution lives
//!   in `U(n, n)` (in fact in `Sp(2n, ℝ)`), and [`heisenberg_generator`]
//!   (`Dᵀ`) evolves the coefficient vector of `a_i(t)` in the basis
//!   `(a_j, a_j†)` as a state of a [`crate::krein_sites::SiteKreinRegister`]
//!   on one site of radix `2n`.
//! * [`bogoliubov_algebra`] — the real Lie algebra `{−iD}` as a basis,
//!   for [`crate::krein::killing_signature_sites`].
//! * [`instability_threshold`] — the smallest `ε ≥ 0` along a ray
//!   `H + εP` at which the classification turns `Unstable`, by scan and
//!   bisection.
//!
//! ## What is measured (`tests/krein_stability.rs`)
//!
//! * **The ghost sector is a particle number.** In the Bogoliubov
//!   reading, a register started on `a_i` and evolved by
//!   [`heisenberg_generator`] has `η`-norm 1 for all time and negative
//!   ("ghost") sector weight `Σ_j |V_ij|²`, which **is** `⟨a_i†a_i⟩(t)`
//!   from the vacuum. Checked against an independent truncated Fock-space
//!   simulation of the same two-mode Hamiltonian (single- and two-mode
//!   squeezing, beam splitter and detuning at once) to `1e-9`. So on this
//!   reading the `krein` module's open question has an answer: the
//!   negative-sector weight is not a probability but the expected number
//!   of quanta created, and it is unbounded.
//! * **The instability band is an opposite-signature collision.** Two-mode
//!   squeezing `g(a†b† + ab)` between modes at `ω_a, ω_b` couples the
//!   `+` mode at `ω_a` to the `−` mode at `−ω_b`; the classifier reports
//!   `Unstable` exactly inside `|ω_a + ω_b| < 2g`, with the growth rate
//!   `√(g² − ((ω_a+ω_b)/2)²)`, and the register's path weight grows at
//!   twice that rate. A beam splitter `g(a†b + ab†)` couples two `+`
//!   modes; it is stable for every `g` and detuning, including on top of
//!   an exact same-signature degeneracy.
//! * **The stability margin is the opposite-signature gap.** For a
//!   diagonal two-mode generator, the threshold along every one of 200
//!   seeded random Bogoliubov directions (normalised in operator norm) is
//!   at least half the opposite-signature gap, and pair-squeezing the
//!   closest opposite pair attains it exactly. Shrinking the
//!   same-signature gap from `1` to `10⁻⁶` leaves the margin unchanged;
//!   shrinking the opposite gap by the same factor takes the margin with
//!   it. The spectral gap is the wrong number.
//! * **The PT dimer.** `H = [[iγ, κ], [κ, −iγ]]` is `X`-self-adjoint, so
//!   it runs on a one-qubit [`crate::krein::KreinRegister`] with the
//!   metric `X`: strongly stable for `γ < κ` (eigenvalues
//!   `±√(κ² − γ²)` of opposite signature, gap `2√(κ² − γ²)`), `Marginal`
//!   at the exceptional point `γ = κ` (path weight grows quadratically,
//!   measured), `Unstable` beyond with rate `√(γ² − κ²)`.
//! * **The two-mode Bogoliubov group is the AdS₄ group.**
//!   [`bogoliubov_algebra`]`(n)` has dimension `n(2n+1)` and Killing
//!   inertia `(n(n+1), n², 0)` — `sp(2n, ℝ)`. At `n = 2` that is
//!   `(6, 4, 0)`, the inertia the twistor stabilizer of a rank-4 infinity
//!   twistor (AdS₄) and the `so(3, 2)` qudit already measured: the
//!   squeezing/beam-splitter group of two bosonic modes and the isometry
//!   group of AdS₄ are one real form, read off the same Killing form.
//!
//! ## Boundaries
//!
//! * The eigensolver is dense and `O(d³)` per sweep; it is for site-sized
//!   generators, not registers. Eigenvalues of a Jordan block of size `k`
//!   are only resolved to `ε^{1/k}`; the clustering tolerance (default
//!   `1e-6` relative) is what decides that a split pair is "one
//!   eigenvalue", and growth rates below it are reported as `Marginal`
//!   rather than `Unstable`. The tolerance is a dial, stated, not hidden.
//! * [`instability_threshold`] measures one ray. A minimum over seeded
//!   rays is an upper bound on the true margin, not the margin; the tests
//!   pair it with a ray that attains the lower bound.
//! * Strong stability is a statement about `η`-self-adjoint perturbations.
//!   A perturbation that breaks `η`-self-adjointness leaves the Krein
//!   register altogether (it is refused there), and no claim is made
//!   about it.

use crate::error::{Error, Result};
use crate::krein_sites::exponentiate_site;
use crate::recursive::sym_eigen;
use crate::scalar::C64;
use crate::space::SiteMatrix;

fn c(re: f64) -> C64 {
    C64::new(re, 0.0)
}

fn dagger(m: &SiteMatrix<C64>) -> SiteMatrix<C64> {
    let d = m.dim();
    let data = (0..d * d).map(|k| m.get(k % d, k / d).conj()).collect();
    SiteMatrix::from_vec(d, data).expect("square")
}

fn max_abs(m: &SiteMatrix<C64>) -> f64 {
    m.data().iter().fold(0.0f64, |a, z| a.max(z.norm()))
}

/// `max |(ηH − H†η)_{rc}|`: zero exactly when `H` is `η`-self-adjoint.
pub fn eta_selfadjoint_deviation(h: &SiteMatrix<C64>, eta: &SiteMatrix<C64>) -> Result<f64> {
    let lhs = eta.matmul(h)?;
    let rhs = dagger(h).matmul(eta)?;
    Ok(lhs
        .data()
        .iter()
        .zip(rhs.data())
        .fold(0.0f64, |m, (x, y)| m.max((x - y).norm())))
}

// ── the eigensolver ──────────────────────────────────────────────────

/// The eigenvalues of a square complex matrix, with multiplicity, in no
/// particular order: Householder reduction to upper Hessenberg form, then
/// shifted QR (Wilkinson shifts, with an exceptional shift on stalls)
/// with deflation. See the module docs for the accuracy boundary.
pub fn eigenvalues(m: &SiteMatrix<C64>) -> Result<Vec<C64>> {
    let n = m.dim();
    let mut a: Vec<C64> = m.data().to_vec();
    let at = |r: usize, col: usize| r * n + col;
    // Hessenberg reduction.
    for k in 0..n.saturating_sub(2) {
        let x: Vec<C64> = (k + 1..n).map(|r| a[at(r, k)]).collect();
        let norm = x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if norm < 1e-300 {
            continue;
        }
        let phase = if x[0].norm() > 0.0 {
            x[0] / x[0].norm()
        } else {
            c(1.0)
        };
        let mut v = x.clone();
        v[0] += phase * norm;
        let vn = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        if vn < 1e-300 {
            continue;
        }
        for z in v.iter_mut() {
            *z /= vn;
        }
        // Left: rows k+1.. ← (1 − 2vv†) rows.
        for col in 0..n {
            let mut dot = c(0.0);
            for (i, vi) in v.iter().enumerate() {
                dot += vi.conj() * a[at(k + 1 + i, col)];
            }
            for (i, vi) in v.iter().enumerate() {
                a[at(k + 1 + i, col)] -= vi * dot * 2.0;
            }
        }
        // Right: columns k+1.. ← columns (1 − 2vv†).
        for r in 0..n {
            let mut dot = c(0.0);
            for (i, vi) in v.iter().enumerate() {
                dot += a[at(r, k + 1 + i)] * vi;
            }
            for (i, vi) in v.iter().enumerate() {
                a[at(r, k + 1 + i)] -= dot * vi.conj() * 2.0;
            }
        }
    }
    let scale = a.iter().fold(0.0f64, |s, z| s.max(z.norm())).max(1e-300);
    let mut out = Vec::with_capacity(n);
    let mut hi = n as isize - 1;
    let mut stall = 0usize;
    while hi >= 0 {
        let h = hi as usize;
        if h == 0 {
            out.push(a[at(0, 0)]);
            break;
        }
        // Find the start of the active unreduced block.
        let mut l = h;
        while l > 0 {
            let sub = a[at(l, l - 1)].norm();
            let diag = a[at(l, l)].norm() + a[at(l - 1, l - 1)].norm();
            if sub <= f64::EPSILON * diag.max(scale * 1e-3) {
                a[at(l, l - 1)] = c(0.0);
                break;
            }
            l -= 1;
        }
        if l == h {
            out.push(a[at(h, h)]);
            hi -= 1;
            stall = 0;
            continue;
        }
        stall += 1;
        if stall > 200 * n {
            return Err(Error::InvalidState(
                "eigenvalues: shifted QR did not converge".to_string(),
            ));
        }
        // Wilkinson shift from the trailing 2×2 of the active block.
        let (p, q, r, s) = (
            a[at(h - 1, h - 1)],
            a[at(h - 1, h)],
            a[at(h, h - 1)],
            a[at(h, h)],
        );
        let tr = p + s;
        let det = p * s - q * r;
        let disc = (tr * tr * 0.25 - det).sqrt();
        let (e1, e2) = (tr * 0.5 + disc, tr * 0.5 - disc);
        let mut mu = if (e1 - s).norm() < (e2 - s).norm() {
            e1
        } else {
            e2
        };
        if stall % 11 == 10 {
            // Exceptional shift: break a symmetric stall.
            mu = s + c(0.75 * a[at(h, h - 1)].norm());
        }
        // One QR step on the active block [l, h]: A − μ = QR, A ← RQ + μ.
        for k in l..=h {
            a[at(k, k)] -= mu;
        }
        let mut rots: Vec<(C64, C64)> = Vec::with_capacity(h - l);
        for k in l..h {
            let x = a[at(k, k)];
            let y = a[at(k + 1, k)];
            let rr = (x.norm_sqr() + y.norm_sqr()).sqrt();
            let (cs, sn) = if rr < 1e-300 {
                (c(1.0), c(0.0))
            } else {
                (x / rr, y / rr)
            };
            for col in k..=h {
                let (u, w) = (a[at(k, col)], a[at(k + 1, col)]);
                a[at(k, col)] = cs.conj() * u + sn.conj() * w;
                a[at(k + 1, col)] = -sn * u + cs * w;
            }
            rots.push((cs, sn));
        }
        for (i, &(cs, sn)) in rots.iter().enumerate() {
            let k = l + i;
            for row in l..=(k + 2).min(h) {
                let (u, w) = (a[at(row, k)], a[at(row, k + 1)]);
                a[at(row, k)] = u * cs + w * sn;
                a[at(row, k + 1)] = -u * sn.conj() + w * cs.conj();
            }
        }
        for k in l..=h {
            a[at(k, k)] += mu;
        }
    }
    Ok(out)
}

// ── Krein signatures ─────────────────────────────────────────────────

/// Eigen-decomposition of a Hermitian `d × d` matrix through its real
/// symmetric embedding `[[Re A, −Im A], [Im A, Re A]]`: eigenvalues
/// ascending (each once) and orthonormal complex eigenvectors.
fn hermitian_eigen(d: usize, a: &[C64]) -> (Vec<f64>, Vec<Vec<C64>>) {
    let n = 2 * d;
    let mut s = vec![0.0f64; n * n];
    for r in 0..d {
        for col in 0..d {
            let z = a[r * d + col];
            s[r * n + col] = z.re;
            s[(r + d) * n + col + d] = z.re;
            s[r * n + col + d] = -z.im;
            s[(r + d) * n + col] = z.im;
        }
    }
    let (vals, vecs) = sym_eigen(n, &s);
    // Each eigenvalue appears twice; pick a complex-independent set.
    let mut values = Vec::with_capacity(d);
    let mut basis: Vec<Vec<C64>> = Vec::with_capacity(d);
    for k in 0..n {
        let mut z: Vec<C64> = (0..d)
            .map(|i| C64::new(vecs[i * n + k], vecs[(i + d) * n + k]))
            .collect();
        for b in &basis {
            let dot: C64 = b.iter().zip(&z).map(|(x, y)| x.conj() * y).sum();
            for (zi, bi) in z.iter_mut().zip(b) {
                *zi -= bi * dot;
            }
        }
        let norm = z.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if norm > 1e-6 && basis.len() < d {
            for zi in z.iter_mut() {
                *zi /= norm;
            }
            basis.push(z);
            values.push(vals[k]);
        }
    }
    (values, basis)
}

/// The Krein type of a cluster of eigenvalues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KreinType {
    /// Real, and `η` is positive definite on its generalized eigenspace.
    Positive,
    /// Real, and `η` is negative definite there.
    Negative,
    /// Real, and `η` is indefinite there: an opposite-signature collision.
    Mixed,
    /// Off the real axis: the generalized eigenspace is `η`-neutral.
    Neutral,
}

/// One cluster of (numerically) coincident eigenvalues.
#[derive(Debug, Clone, PartialEq)]
pub struct KreinCluster {
    /// The cluster's mean eigenvalue.
    pub value: C64,
    /// Algebraic multiplicity (eigenvalues in the cluster).
    pub multiplicity: usize,
    /// Geometric multiplicity (nullity of `H − λ`).
    pub geometric: usize,
    /// Positive directions of `η` on the generalized eigenspace.
    pub positive: usize,
    /// Negative directions of `η` there.
    pub negative: usize,
    /// The type, from the above.
    pub kind: KreinType,
}

/// The stability class of an `η`-self-adjoint generator. See the module
/// docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Stability {
    /// Real spectrum, every eigenvalue `η`-definite. `opposite_gap` is
    /// the least distance between a positive and a negative eigenvalue
    /// (`∞` if the signatures never meet).
    Strong {
        /// Least distance between eigenvalues of opposite signature.
        opposite_gap: f64,
    },
    /// Real, semisimple spectrum with at least one mixed eigenvalue:
    /// bounded, but arbitrarily small perturbations can destabilize it.
    Weak,
    /// Real spectrum with a Jordan block: polynomial growth (an
    /// exceptional point).
    Marginal,
    /// A complex pair: exponential growth at rate `max Im λ`.
    Unstable {
        /// `max Im λ` over the spectrum.
        growth_rate: f64,
    },
}

/// The Krein-resolved spectrum of a generator.
#[derive(Debug, Clone, PartialEq)]
pub struct KreinSpectrum {
    /// Eigenvalue clusters, sorted by real then imaginary part.
    pub clusters: Vec<KreinCluster>,
    /// The resulting classification.
    pub stability: Stability,
}

/// Default relative tolerance for clustering and the real-axis test.
pub const STABILITY_TOL: f64 = 1e-6;

/// The Krein-resolved spectrum of `h` with respect to the Hermitian
/// involution `eta`, at relative tolerance `tol` (see
/// [`STABILITY_TOL`]). Refuses `h` that is not `η`-self-adjoint.
pub fn krein_spectrum(
    h: &SiteMatrix<C64>,
    eta: &SiteMatrix<C64>,
    tol: f64,
) -> Result<KreinSpectrum> {
    let d = h.dim();
    if eta.dim() != d {
        return Err(Error::InvalidState(format!(
            "krein_spectrum: a {d}-dim generator with a {}-dim metric",
            eta.dim()
        )));
    }
    let scale = max_abs(h).max(1.0);
    let dev = eta_selfadjoint_deviation(h, eta)?;
    if dev > 1e-9 * scale {
        return Err(Error::InvalidState(format!(
            "krein_spectrum: generator is not η-self-adjoint (deviation {dev:.3e})"
        )));
    }
    let atol = tol * scale;
    let mut eig = eigenvalues(h)?;
    eig.sort_by(|x, y| x.re.total_cmp(&y.re).then(x.im.total_cmp(&y.im)));
    // Single-linkage clustering.
    let mut label: Vec<usize> = (0..d).collect();
    fn find(l: &mut [usize], i: usize) -> usize {
        let mut i = i;
        while l[i] != i {
            l[i] = l[l[i]];
            i = l[i];
        }
        i
    }
    for i in 0..d {
        for j in (i + 1)..d {
            if (eig[i] - eig[j]).norm() < atol {
                let (a, b) = (find(&mut label, i), find(&mut label, j));
                label[a.max(b)] = a.min(b);
            }
        }
    }
    let mut groups: Vec<Vec<C64>> = Vec::new();
    let mut roots: Vec<usize> = Vec::new();
    for (i, &e) in eig.iter().enumerate() {
        let r = find(&mut label, i);
        match roots.iter().position(|&x| x == r) {
            Some(k) => groups[k].push(e),
            None => {
                roots.push(r);
                groups.push(vec![e]);
            }
        }
    }
    let mut clusters = Vec::with_capacity(groups.len());
    for g in &groups {
        let m = g.len();
        let lambda: C64 = g.iter().sum::<C64>() / m as f64;
        let real = lambda.im.abs() <= atol;
        let shift = if real { c(lambda.re) } else { lambda };
        let mut a = h.data().to_vec();
        for k in 0..d {
            a[k * d + k] -= shift;
        }
        let a_m = SiteMatrix::from_vec(d, a.clone())?;
        // Geometric multiplicity: singular values of H − λ below atol.
        let gram1 = dagger(&a_m).matmul(&a_m)?;
        let (s1, _) = hermitian_eigen(d, gram1.data());
        let geometric = s1.iter().filter(|&&x| x.max(0.0).sqrt() <= atol).count();
        // Generalized eigenspace: the m least singular directions of
        // (H − λ)^m.
        let mut pow = a_m.clone();
        for _ in 1..m {
            pow = pow.matmul(&a_m)?;
        }
        let gram = dagger(&pow).matmul(&pow)?;
        let (_, vecs) = hermitian_eigen(d, gram.data());
        let v: Vec<&Vec<C64>> = vecs.iter().take(m).collect();
        // η restricted: G = V†ηV.
        let mut gm = vec![c(0.0); m * m];
        for i in 0..m {
            for j in 0..m {
                let mut acc = c(0.0);
                for r in 0..d {
                    let ev: C64 = v[j]
                        .iter()
                        .enumerate()
                        .map(|(k, x)| eta.get(r, k) * x)
                        .sum();
                    acc += v[i][r].conj() * ev;
                }
                gm[i * m + j] = acc;
            }
        }
        let (gv, _) = hermitian_eigen(m, &gm);
        let cut = 1e-6;
        let positive = gv.iter().filter(|&&x| x > cut).count();
        let negative = gv.iter().filter(|&&x| x < -cut).count();
        let kind = if !real {
            KreinType::Neutral
        } else if negative == 0 {
            KreinType::Positive
        } else if positive == 0 {
            KreinType::Negative
        } else {
            KreinType::Mixed
        };
        clusters.push(KreinCluster {
            value: if real { c(lambda.re) } else { lambda },
            multiplicity: m,
            geometric,
            positive,
            negative,
            kind,
        });
    }
    clusters.sort_by(|x, y| {
        x.value
            .re
            .total_cmp(&y.value.re)
            .then(x.value.im.total_cmp(&y.value.im))
    });
    let growth = eig.iter().fold(f64::NEG_INFINITY, |a, z| a.max(z.im));
    let stability = if clusters.iter().any(|k| k.kind == KreinType::Neutral) {
        Stability::Unstable {
            growth_rate: growth,
        }
    } else if clusters.iter().any(|k| k.geometric < k.multiplicity) {
        Stability::Marginal
    } else if clusters.iter().any(|k| k.kind == KreinType::Mixed) {
        Stability::Weak
    } else {
        let mut gap = f64::INFINITY;
        for x in &clusters {
            for y in &clusters {
                if x.kind == KreinType::Positive && y.kind == KreinType::Negative {
                    gap = gap.min((x.value.re - y.value.re).abs());
                }
            }
        }
        Stability::Strong { opposite_gap: gap }
    };
    Ok(KreinSpectrum {
        clusters,
        stability,
    })
}

/// The smallest `ε` in `[0, eps_max]` at which `h + εp` is classified
/// `Unstable`, located by a uniform scan of `steps` points and then
/// bisection to `1e-10`; `None` if the ray stays out of the unstable
/// region. `p` must itself be `η`-self-adjoint.
pub fn instability_threshold(
    h: &SiteMatrix<C64>,
    p: &SiteMatrix<C64>,
    eta: &SiteMatrix<C64>,
    eps_max: f64,
    steps: usize,
) -> Result<Option<f64>> {
    let d = h.dim();
    let at = |e: f64| -> Result<bool> {
        let data = h
            .data()
            .iter()
            .zip(p.data())
            .map(|(x, y)| x + y * e)
            .collect();
        let m = SiteMatrix::from_vec(d, data)?;
        Ok(matches!(
            krein_spectrum(&m, eta, STABILITY_TOL)?.stability,
            Stability::Unstable { .. }
        ))
    };
    let steps = steps.max(1);
    let mut prev = 0.0;
    if at(0.0)? {
        return Ok(Some(0.0));
    }
    for k in 1..=steps {
        let e = eps_max * k as f64 / steps as f64;
        if at(e)? {
            let (mut lo, mut hi) = (prev, e);
            while hi - lo > 1e-10 {
                let mid = 0.5 * (lo + hi);
                if at(mid)? {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            return Ok(Some(hi));
        }
        prev = e;
    }
    Ok(None)
}

// ── Bogoliubov ───────────────────────────────────────────────────────

/// The Bogoliubov metric `diag(+1 ×n, −1 ×n)` as signs, for
/// [`crate::krein_sites::SiteMetric::signs`] on a site of radix `2n`.
pub fn bogoliubov_signs(n: usize) -> Vec<i8> {
    (0..2 * n).map(|k| if k < n { 1 } else { -1 }).collect()
}

/// The Bogoliubov metric as a matrix.
pub fn bogoliubov_metric(n: usize) -> SiteMatrix<C64> {
    let d = 2 * n;
    let mut data = vec![c(0.0); d * d];
    for k in 0..d {
        data[k * d + k] = c(if k < n { 1.0 } else { -1.0 });
    }
    SiteMatrix::from_vec(d, data).expect("square")
}

/// The dynamical matrix `D = [[h, g], [−ḡ, −h̄]]` of
/// `Σ h_ij a_i†a_j + ½Σ(g_ij a_i†a_j† + h.c.)`: `i d(a, a†)/dt = D (a, a†)`.
/// `h` must be Hermitian and `g` symmetric (refused otherwise).
pub fn bogoliubov(h: &SiteMatrix<C64>, g: &SiteMatrix<C64>) -> Result<SiteMatrix<C64>> {
    let n = h.dim();
    if g.dim() != n {
        return Err(Error::InvalidState(
            "bogoliubov: h and g differ in size".to_string(),
        ));
    }
    for i in 0..n {
        for j in 0..n {
            if (h.get(i, j) - h.get(j, i).conj()).norm() > 1e-12 {
                return Err(Error::InvalidState(
                    "bogoliubov: h is not Hermitian".to_string(),
                ));
            }
            if (g.get(i, j) - g.get(j, i)).norm() > 1e-12 {
                return Err(Error::InvalidState(
                    "bogoliubov: g is not symmetric".to_string(),
                ));
            }
        }
    }
    let d = 2 * n;
    let mut data = vec![c(0.0); d * d];
    for i in 0..n {
        for j in 0..n {
            data[i * d + j] = h.get(i, j);
            data[i * d + n + j] = g.get(i, j);
            data[(n + i) * d + j] = -g.get(i, j).conj();
            data[(n + i) * d + n + j] = -h.get(i, j).conj();
        }
    }
    SiteMatrix::from_vec(d, data)
}

/// `Dᵀ`: the generator of the coefficient vector of `a_i(t)` in the
/// basis `(a_j, a_j†)` (`i dc/dt = Dᵀc`). `η`-self-adjoint for the
/// Bogoliubov metric, so `exp(−iDᵀt)` is a gate on a
/// [`crate::krein_sites::SiteKreinRegister`]; started on `e_i`, the
/// register's negative-sector weight is `⟨a_i†a_i⟩(t)` from the vacuum.
pub fn heisenberg_generator(d: &SiteMatrix<C64>) -> SiteMatrix<C64> {
    let n = d.dim();
    let data = (0..n * n).map(|k| d.get(k % n, k / n)).collect();
    SiteMatrix::from_vec(n, data).expect("square")
}

/// `exp(−iHt)`.
pub fn evolution(h: &SiteMatrix<C64>, t: f64) -> Result<SiteMatrix<C64>> {
    let x = SiteMatrix::from_vec(
        h.dim(),
        h.data().iter().map(|z| z * C64::new(0.0, -1.0)).collect(),
    )?;
    exponentiate_site(&x, t)
}

/// A real basis of the Lie algebra `{−iD}` of `n`-mode Bogoliubov
/// generators: `n²` from Hermitian `h` and `n(n+1)` from symmetric `g`,
/// `n(2n+1)` in all — `sp(2n, ℝ)` in its `U(n, n)` form.
pub fn bogoliubov_algebra(n: usize) -> Result<Vec<SiteMatrix<C64>>> {
    let zero = || SiteMatrix::from_vec(n, vec![c(0.0); n * n]);
    let set = |m: &SiteMatrix<C64>, i: usize, j: usize, z: C64| -> SiteMatrix<C64> {
        let mut data = m.data().to_vec();
        data[i * n + j] = z;
        SiteMatrix::from_vec(n, data).expect("square")
    };
    let mut out = Vec::with_capacity(n * (2 * n + 1));
    let mi = |d: SiteMatrix<C64>| -> Result<SiteMatrix<C64>> {
        SiteMatrix::from_vec(
            d.dim(),
            d.data().iter().map(|z| z * C64::new(0.0, -1.0)).collect(),
        )
    };
    for i in 0..n {
        for j in i..n {
            let reals: &[C64] = if i == j {
                &[C64 { re: 1.0, im: 0.0 }]
            } else {
                &[C64 { re: 1.0, im: 0.0 }, C64 { re: 0.0, im: 1.0 }]
            };
            for &z in reals {
                let h = set(&set(&zero()?, i, j, z), j, i, z.conj());
                out.push(mi(bogoliubov(&h, &zero()?)?)?);
            }
            for z in [c(1.0), C64::new(0.0, 1.0)] {
                let g = set(&set(&zero()?, i, j, z), j, i, z);
                out.push(mi(bogoliubov(&zero()?, &g)?)?);
            }
        }
    }
    Ok(out)
}
