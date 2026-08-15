//! A small library of named circuit constructions, used by the tests,
//! benchmarks and examples — and as executable documentation of the crate's
//! conventions.
//!
//! Constructions are generic over the amplitude algebra; whether a given
//! circuit *binds* over an algebra depends on which gates the registry could
//! install there (e.g. [`qft`] needs `cp`, so it binds over ℂ/ℍ/𝕆 but not ℝ,
//! while [`ghz`] binds everywhere).

use crate::circuit::Circuit;
use crate::error::Result;
use crate::math::GateMatrix;
use crate::rng::Prng;
use crate::scalar::Scalar;

/// Bell pair preparation: `H(0); CX(0, 1)` giving `(|00⟩ + |11⟩)/√2`.
pub fn bell<S: Scalar>() -> Circuit<S> {
    let mut c = Circuit::new(2);
    c.h(0).cx(0, 1);
    c
}

/// GHZ state on `n ≥ 1` qubits: `(|0…0⟩ + |1…1⟩)/√2`.
pub fn ghz<S: Scalar>(n: usize) -> Circuit<S> {
    assert!(n >= 1, "GHZ needs at least one qubit");
    let mut c = Circuit::new(n);
    c.h(0);
    for q in 1..n {
        c.cx(q - 1, q);
    }
    c
}

/// Brickwork light-cone circuit: `H` on each qubit in `seeds`, then
/// `depth` alternating layers of adjacent `CX(q, q+1)` — even-`q` pairs
/// on even layers, odd-`q` pairs on odd layers. All-Clifford.
///
/// This is the canonical *causally local* workload: each layer is a set
/// of disjoint nearest-neighbour gates, so information spreads at most
/// one site per layer and the causal cone of the seed set widens
/// linearly with depth. Everything outside the cone stays exactly |0⟩.
pub fn brickwork<S: Scalar>(n: usize, depth: usize, seeds: &[usize]) -> Circuit<S> {
    let mut c = Circuit::new(n);
    for &s in seeds {
        c.h(s);
    }
    for layer in 0..depth {
        let start = layer % 2;
        let mut q = start;
        while q + 1 < n {
            c.cx(q, q + 1);
            q += 2;
        }
    }
    c
}

/// A layer of `n/2` disjoint Bell pairs, each spanning interaction
/// distance `d`: `H(a); CX(a, a+d)` for every pair. Requires
/// `n % (2·d) == 0`; pairs tile in blocks of `2d`
/// (`(b·2d + j, b·2d + j + d)` for `j < d`), so every qubit belongs to
/// exactly one pair. All-Clifford.
///
/// The family holds the gate count, arity profile and output state
/// *shape* (a product of `n/2` Bell pairs) fixed while varying only the
/// causal range `d` — the knob for measuring how each representation
/// and each device geometry prices interaction distance.
pub fn ranged_pairs<S: Scalar>(n: usize, d: usize) -> Circuit<S> {
    assert!(d >= 1 && n % (2 * d) == 0, "pairs at range d must tile n");
    let mut c = Circuit::new(n);
    for block in 0..n / (2 * d) {
        for j in 0..d {
            let a = block * 2 * d + j;
            c.h(a).cx(a, a + d);
        }
    }
    c
}

/// The rainbow state on even `n`: `H(i); CX(i, n−1−i)` for `i < n/2` —
/// `n/2` disjoint Bell pairs nested around the centre, at interaction
/// distances `n−1, n−3, …, 1`. All-Clifford.
///
/// Every pair crosses the central cut, so any representation that pays
/// per *linear or hierarchical cut* (MPS bonds, mera's root) faces rank
/// `2^{n/2}` there, while the state remains a product of pairs that a
/// clustering representation stores in `O(n)`.
pub fn rainbow<S: Scalar>(n: usize) -> Circuit<S> {
    assert!(n >= 2 && n % 2 == 0, "rainbow needs even n");
    let mut c = Circuit::new(n);
    for i in 0..n / 2 {
        c.h(i).cx(i, n - 1 - i);
    }
    c
}

/// An IQP-style circuit: an `H` wall, a commuting diagonal core (a
/// `t` on every qubit and `pairs` seeded `cz` couplings), and a second
/// `H` wall. The `long_range` flag is the structural knob: `false`
/// draws the `cz` pairs nearest-neighbour (the fragment known — and
/// here measured — to stay easy across linear cuts), `true` draws them
/// uniformly at random (the sampling-hardness regime, which escapes
/// every representation in the crate at once).
pub fn iqp<S: Scalar>(n: usize, pairs: usize, long_range: bool, seed: u64) -> Circuit<S> {
    assert!(n >= 2, "iqp needs at least two qubits");
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    for q in 0..n {
        c.t(q);
    }
    for _ in 0..pairs {
        if long_range {
            let a = (rng.next_u64() % n as u64) as usize;
            let b = ((a + 1) + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
            c.cz(a, b);
        } else {
            let a = (rng.next_u64() % (n as u64 - 1)) as usize;
            c.cz(a, a + 1);
        }
    }
    for q in 0..n {
        c.h(q);
    }
    c
}

/// A depth-`depth` brickwork over a `width × height` grid (row-major
/// sites): a seeded `ry` rotation layer, then alternating row-neighbour
/// and column-neighbour `cx` bricks per layer. At fixed shallow depth
/// the entanglement any cut sees grows with the *boundary* of the
/// region, not its volume — the constant-depth-2D regime where the
/// cut-rank assumptions fail slowly instead of exponentially.
pub fn brickwork_2d<S: Scalar>(width: usize, height: usize, depth: usize, seed: u64) -> Circuit<S> {
    let n = width * height;
    let site = |r: usize, c: usize| r * width + c;
    let mut rng = Prng::new(seed);
    let mut circuit = Circuit::new(n);
    for layer in 0..depth {
        for q in 0..n {
            circuit.ry(q, 0.4 + rng.next_f64());
        }
        if layer % 2 == 0 {
            for r in 0..height {
                let mut c = r % 2;
                while c + 1 < width {
                    circuit.cx(site(r, c), site(r, c + 1));
                    c += 2;
                }
            }
        } else {
            for c in 0..width {
                let mut r = c % 2;
                while r + 1 < height {
                    circuit.cx(site(r, c), site(r + 1, c));
                    r += 2;
                }
            }
        }
    }
    circuit
}

/// A `t`-doped Clifford circuit: `cliffords` seeded Clifford gates over
/// an `H` wall, with `t` T gates spread evenly through the stream. The
/// doping level is the magic resource dial: at `t = O(log n)` the
/// stabilizer-frame assumption still holds (stored support `≤ 2^t` is
/// polynomial) however entangling the Clifford bulk is; at `t = Θ(n)`
/// it fails with everything else.
pub fn doped_clifford<S: Scalar>(n: usize, cliffords: usize, t: usize, seed: u64) -> Circuit<S> {
    assert!(n >= 2, "doped_clifford needs at least two qubits");
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    let stride = cliffords / (t + 1);
    let mut placed = 0;
    for g in 0..cliffords {
        let a = (rng.next_u64() % n as u64) as usize;
        let b = ((a + 1) + (rng.next_u64() % (n as u64 - 1)) as usize) % n;
        match rng.next_u64() % 6 {
            0 => c.h(a),
            1 => c.s(a),
            2 => c.x(a),
            3 => c.z(a),
            4 => c.cx(a, b),
            _ => c.cz(a, b),
        };
        if placed < t && stride > 0 && (g + 1) % stride == 0 {
            c.t((rng.next_u64() % n as u64) as usize);
            placed += 1;
        }
    }
    while placed < t {
        c.t((rng.next_u64() % n as u64) as usize);
        placed += 1;
    }
    c
}

/// Quantum Fourier transform on `n` qubits (little-endian):
/// `|x⟩ → 2^{-n/2} Σ_y e^{2πi x y / 2^n} |y⟩`, including the final qubit
/// reversal swaps. Invert with [`crate::circuit::BoundCircuit::inverse`].
pub fn qft<S: Scalar>(n: usize) -> Circuit<S> {
    let mut c = Circuit::new(n);
    for j in (0..n).rev() {
        c.h(j);
        for m in (0..j).rev() {
            let angle = std::f64::consts::PI / (1u64 << (j - m)) as f64;
            c.cp(m, j, angle);
        }
    }
    for k in 0..n / 2 {
        c.swap(k, n - 1 - k);
    }
    c
}

/// The banded (approximate) QFT: [`qft`] with every controlled phase
/// below `min_angle` pruned — exactly the tail-radix web
/// [`radixweb::phase_web`](crate::radixweb::phase_web) derives and
/// verifies for the binary profile, whose measured bandwidth saturates
/// at fixed `min_angle` while the full triangle grows quadratically
/// (Coppersmith's construction, from radix arithmetic). `min_angle =
/// 0.0` is the exact [`qft`], op for op; the surviving `cp` count
/// equals the web's off-diagonal coupling count, asserted in the
/// radixweb suite.
pub fn aqft<S: Scalar>(n: usize, min_angle: f64) -> Circuit<S> {
    let mut c = Circuit::new(n);
    for j in (0..n).rev() {
        c.h(j);
        for m in (0..j).rev() {
            let angle = std::f64::consts::PI / (1u64 << (j - m)) as f64;
            if angle < min_angle {
                break; // angles only shrink as the control recedes
            }
            c.cp(m, j, angle);
        }
    }
    for k in 0..n / 2 {
        c.swap(k, n - 1 - k);
    }
    c
}

/// Nearest-neighbour variant of [`random_circuit`]: the same
/// twelve-gate pool with every two-qubit operand pair drawn adjacent —
/// the connectivity dial. Interaction range flipped the IQP verdict
/// (`iqp(n, 2n, long_range)`); this applies the same knob to the
/// universal pool.
pub fn random_nn<S: Scalar>(n: usize, gates: usize, seed: u64) -> Circuit<S> {
    assert!(n >= 2, "random_nn needs at least two qubits");
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    let tau = std::f64::consts::TAU;
    for _ in 0..gates {
        let q0 = (rng.next_u64() % (n as u64 - 1)) as usize;
        let q1 = q0 + 1;
        match rng.next_u64() % 12 {
            0 => c.h(q0),
            1 => c.x(q0),
            2 => c.t(q0),
            3 => c.s(q0),
            4 => c.sx(q0),
            5 => c.rx(q0, rng.next_f64() * tau),
            6 => c.ry(q0, rng.next_f64() * tau),
            7 => c.rz(q0, rng.next_f64() * tau),
            8 => c.cx(q0, q1),
            9 => c.cz(q0, q1),
            10 => c.cp(q0, q1, rng.next_f64() * tau),
            _ => c.swap(q0, q1),
        };
    }
    c
}

/// Dyadic-angle variant of [`random_circuit`]: the same pool and
/// connectivity with every parametric angle snapped to the dyadic grid
/// `π·k/32` — the angle dial that turns "refused by name" into
/// "measurable" for the exact-fragment instruments (the path sum, the
/// `D[ω]` ring, up-embedded readout).
pub fn random_dyadic<S: Scalar>(n: usize, gates: usize, seed: u64) -> Circuit<S> {
    assert!(n >= 2, "random_dyadic needs at least two qubits");
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    let dyadic = |rng: &mut Prng| std::f64::consts::PI * (rng.next_u64() % 64) as f64 / 32.0;
    for _ in 0..gates {
        let mut pick_q = || (rng.next_u64() % n as u64) as usize;
        let q0 = pick_q();
        let mut q1 = pick_q();
        while q1 == q0 {
            q1 = pick_q();
        }
        match rng.next_u64() % 12 {
            0 => c.h(q0),
            1 => c.x(q0),
            2 => c.t(q0),
            3 => c.s(q0),
            4 => c.sx(q0),
            5 => {
                let a = dyadic(&mut rng);
                c.rx(q0, a)
            }
            6 => {
                let a = dyadic(&mut rng);
                c.ry(q0, a)
            }
            7 => {
                let a = dyadic(&mut rng);
                c.rz(q0, a)
            }
            8 => c.cx(q0, q1),
            9 => c.cz(q0, q1),
            10 => {
                let a = dyadic(&mut rng);
                c.cp(q0, q1, a)
            }
            _ => c.swap(q0, q1),
        };
    }
    c
}

/// Inverse quantum Fourier transform on `n` qubits: the exact op-by-op
/// reversal of [`qft`] with negated phases.
pub fn iqft<S: Scalar>(n: usize) -> Circuit<S> {
    let mut c = Circuit::new(n);
    for k in (0..n / 2).rev() {
        c.swap(k, n - 1 - k);
    }
    for j in 0..n {
        for m in 0..j {
            let angle = -std::f64::consts::PI / (1u64 << (j - m)) as f64;
            c.cp(m, j, angle);
        }
        c.h(j);
    }
    c
}

/// A multi-controlled Z on all `n` qubits as a dense matrix
/// (`diag(1, …, 1, −1)`); real-valued, so it exists over every algebra.
/// Costs `O(4^n)` storage — for anything beyond a handful of qubits use
/// [`phase_flip`] with [`Circuit::diagonal`] instead.
pub fn mcz_matrix<S: Scalar>(n: usize) -> Result<GateMatrix<S>> {
    let dim = 1usize << n;
    let mut m = GateMatrix::identity(dim)?;
    m.set(dim - 1, dim - 1, -S::one());
    Ok(m)
}

/// Diagonal entries of an `n`-qubit phase flip about basis state `index`:
/// identity except `−1` at `index`. `O(2^n)` storage; real-valued, so it
/// exists over every algebra. Feed to [`Circuit::diagonal`].
pub fn phase_flip<S: Scalar>(n: usize, index: u64) -> Vec<S> {
    let dim = 1usize << n;
    assert!(index < dim as u64, "phase_flip: index out of range");
    let mut d = vec![S::one(); dim];
    d[index as usize] = -S::one();
    d
}

/// Grover search on `n` qubits for the basis state `marked`, running
/// `iterations` rounds of oracle + diffusion. The optimal iteration count is
/// roughly `π/4 · √(2^n)`.
///
/// The oracle is a diagonal phase flip at `marked`; the diffusion is
/// `H⊗n · (phase flip at 0) · H⊗n` (equal to the textbook reflection
/// `2|s⟩⟨s| − I` up to a global phase). Both use [`Circuit::diagonal`], so
/// circuit memory is `O(2^n)`, not `O(4^n)`.
pub fn grover<S: Scalar>(n: usize, marked: u64, iterations: usize) -> Result<Circuit<S>> {
    assert!((1..=63).contains(&n), "grover: width must fit u64 indices");
    assert!(marked < (1u64 << n), "grover: marked state out of range");
    let oracle = phase_flip::<S>(n, marked);
    let flip_zero = phase_flip::<S>(n, 0);
    let all: Vec<usize> = (0..n).collect();
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    for _ in 0..iterations {
        c.diagonal("oracle", oracle.clone(), all.clone());
        for q in 0..n {
            c.h(q);
        }
        c.diagonal("flip|0…0⟩", flip_zero.clone(), all.clone());
        for q in 0..n {
            c.h(q);
        }
    }
    Ok(c)
}

/// A deterministic pseudo-random circuit over a hardware-ish gate pool,
/// intended for benchmarks and cross-backend agreement tests. Uses the
/// full standard library, so it binds over algebras containing ℂ.
pub fn random_circuit<S: Scalar>(n: usize, gates: usize, seed: u64) -> Circuit<S> {
    assert!(n >= 2, "random_circuit needs at least two qubits");
    let mut rng = Prng::new(seed);
    let mut c = Circuit::new(n);
    let tau = std::f64::consts::TAU;
    for _ in 0..gates {
        let mut pick_q = || (rng.next_u64() % n as u64) as usize;
        let q0 = pick_q();
        let mut q1 = pick_q();
        while q1 == q0 {
            q1 = pick_q();
        }
        match rng.next_u64() % 12 {
            0 => c.h(q0),
            1 => c.x(q0),
            2 => c.t(q0),
            3 => c.s(q0),
            4 => c.sx(q0),
            5 => c.rx(q0, rng.next_f64() * tau),
            6 => c.ry(q0, rng.next_f64() * tau),
            7 => c.rz(q0, rng.next_f64() * tau),
            8 => c.cx(q0, q1),
            9 => c.cz(q0, q1),
            10 => c.cp(q0, q1, rng.next_f64() * tau),
            _ => c.swap(q0, q1),
        };
    }
    c
}

/// An encoded computation over `nodes` L = 2 toric patches: every
/// node's `|0̄0̄⟩` encoder, then `gates` random **logical** operations
/// compiled to physical gates — logical Paulis (X̄/Z̄ strings on a
/// random wire) and transversal CX blocks between random node pairs.
/// The family the constraint-compression backend bets on: `8·nodes`
/// physical qubits computing on `2·nodes` logical wires.
pub fn logical_random<S: Scalar>(nodes: usize, gates: usize, seed: u64) -> Circuit<S> {
    let n = 8 * nodes;
    let mut c = Circuit::new(n);
    let codes: Vec<crate::retro::ToricCode> = (0..nodes)
        .map(|i| crate::retro::ToricCode::new(2, 8 * i).expect("tiling fits"))
        .collect();
    for code in &codes {
        for op in code.encoder(n, [false, false]).ops() {
            if let crate::circuit::Op::Named { name, qubits, .. } = op {
                c.gate(name.as_str(), vec![], qubits.clone());
            }
        }
    }
    let mut rng = Prng::new(seed);
    let emit_string = |c: &mut Circuit<S>, mask: u64, gate: &str| {
        let mut rest = mask;
        while rest != 0 {
            let q = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            c.gate(gate, vec![], vec![q]);
        }
    };
    for _ in 0..gates {
        let arm = rng.next_u64() % 5;
        if arm < 3 || nodes < 2 {
            let node = (rng.next_u64() as usize) % nodes;
            let wire = (rng.next_u64() as usize) % 2;
            if rng.next_u64() % 2 == 0 {
                emit_string(&mut c, codes[node].logical_x(wire).x, "x");
            } else {
                emit_string(&mut c, codes[node].logical_z(wire).z, "z");
            }
        } else {
            let a = (rng.next_u64() as usize) % nodes;
            let mut b = (rng.next_u64() as usize) % nodes;
            if b == a {
                b = (b + 1) % nodes;
            }
            use crate::retro::Code;
            for q in 0..Code::qubits(&codes[a]) {
                c.cx(codes[a].offset() + q, codes[b].offset() + q);
            }
        }
    }
    c
}
