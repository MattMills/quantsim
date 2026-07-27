# quantsim

An in-house research quantum simulator in Rust, built so the interesting
parts are swappable:

- **Amplitude algebra** — every gate matrix, state and measurement rule is
  generic over a [`Scalar`](src/scalar/mod.rs) trait. Shipped: ℝ, ℂ
  (default), a generic **Cayley–Dickson doubling** `CD<T>` giving
  quaternions ℍ, octonions 𝕆 and sedenions 𝕊, plus **split-complex** as a
  first non-Cayley–Dickson algebra. Planned (see [ROADMAP](ROADMAP.md)):
  truncated p-adics, dual numbers, split-quaternions, Clifford scalars.
- **State representation** — a [`Backend<S>`](src/backend/mod.rs) trait with
  three shipped implementations: **dense** state vector (the BQP reference),
  **sparse** hash-map state, and an **adaptive** backend that starts sparse
  and promotes itself to dense at ¼ density. Matrix product states are
  designed-for but not yet implemented.
- **Gates** — a registry (`name → GateDef`) with a 32-gate standard library
  (plus aliases), defined once over ℂ and projected into each algebra;
  over ℝ you automatically get the real subset. Research gates are a
  closure away and are unitarity-validated at registration.

BQP support: the standard registry contains a universal set (`h`, `t`, `cx`,
…), so any BQP circuit family runs exactly on the dense backend — at the
unavoidable `O(2^n)` memory cost in width `n`, which the width benchmarks
measure rather than hide.

## Quick start

```rust
use quantsim::prelude::*;

fn main() -> Result<()> {
    // Bell pair.
    let mut c = Circuit::new(2);
    c.h(0).cx(0, 1);

    let sim: Simulator = Simulator::new();           // Simulator<C64>
    let state = sim.run(&c)?;                        // dense by default
    assert!((state.probability(0b11) - 0.5).abs() < 1e-12);

    // Deterministic, backend-independent sampling.
    let counts = state.sample(1000, &mut Prng::new(7))?;
    println!("{counts:?}");                          // {0: ~500, 3: ~500}

    // Same circuit, sparse representation, 40 qubits — try that dense.
    let ghz = library::ghz(40);
    let state = sim.run_on("sparse", &ghz)?;
    assert_eq!(state.nonzero_count(), 2);
    Ok(())
}
```

### Registering a research gate

```rust
use quantsim::prelude::*;

let mut sim: Simulator = Simulator::new();
sim.registry_mut().register_parametric("xy", "XY interaction", 2, 1, |p| {
    let (c, s) = ((p[0] / 2.0).cos(), (p[0] / 2.0).sin());
    let (o, l) = (c64(1.0, 0.0), c64(0.0, 0.0));
    GateMatrix::from_vec(4, vec![
        o, l,              l,              l,
        l, c64(c, 0.0),    c64(0.0, -s),   l,
        l, c64(0.0, -s),   c64(c, 0.0),    l,
        l, l,              l,              o,
    ])
})?;                       // rejected here if not unitary

let mut c = Circuit::new(2);
c.h(0).gate("xy", vec![1.57], vec![0, 1]);   // usable by name immediately
# Ok::<(), quantsim::Error>(())
```

Backends register the same way (`sim.backends_mut().register("mps", ...)`)
and become selectable via `sim.run_on("mps", &circuit)`. See
[`examples/research_extension.rs`](examples/research_extension.rs) for a
complete custom-gate + custom-backend program, and
[`examples/exotic_algebras.rs`](examples/exotic_algebras.rs) for the algebra
tour (quaternionic gates, sedenion Born-weight drift, split-complex negative
"probabilities").

### Swapping the amplitude algebra

```rust
use quantsim::prelude::*;

// Quaternionic amplitudes: full standard gate set (ℂ embeds in ℍ),
// plus whatever j/k-flavored research gates you register.
let sim = Simulator::<Quaternion>::new();

// Real amplitudes: the registry holds exactly the real-matrix subset
// (x, z, h, ry, cx, cz, ccx, ...), and binding an `s` gate fails loudly.
let sim = Simulator::<f64>::new();
```

## Conventions

| topic | convention |
|---|---|
| bit order | little-endian: qubit 0 is the least significant index bit (Qiskit-style) |
| multi-qubit matrices | sub-index bit `b` ↔ `qubits[b]`; `cx(c, t)` has the control at `c` = `qubits[0]` |
| rotations | half-angle: `rx/ry/rz(θ) = exp(−iθP/2)`; `p(λ) = diag(1, e^{iλ})` |
| module structure | states are left modules: gates multiply amplitudes from the left (matters for ℍ onward) |
| inner products | `⟨φ\|ψ⟩ = Σ conj(φᵢ)ψᵢ`, products left-to-right |
| randomness | in-crate xoshiro256++, seeded — bit-reproducible across platforms and backends |

Two magnitude notions are deliberately distinct on `Scalar`: `abs_sqr`
(Euclidean, always ≥ 0, used for numerics) and `born_weight` (the algebra's
quadratic form, used for measurement). They coincide on ℝ/ℂ/ℍ/𝕆 and diverge
on purpose beyond — sedenion zero divisors make "unitary" gates leak Born
weight, and split-complex states can carry negative weight; the simulator
surfaces both instead of papering over them.

## Testing

`cargo test` runs 122 tests across nine suites plus doctests:

- **conventions** — bit order and control placement pinned on basis states.
- **gate_matrices** — every standard gate vs literature values; exact
  per-algebra gate-support lists.
- **gate_identities** — HXH = Z and friends, SWAP = 3·CX, the Nielsen–Chuang
  Toffoli decomposition, iSWAP = SWAP·(S⊗S)·CZ, ZYZ decomposition, rotation
  additivity, QFT∘IQFT = 1.
- **property_tests** (proptest) — on random circuits: norm preservation;
  dense/sparse/adaptive amplitude agreement; inverse uncomputation;
  unitarity at random parameters; disjoint-gate commutation; measurement
  collapse; **CD⟨f64⟩ ≅ ℂ** as simulators; faithfulness of the ℂ ⊂ ℍ and
  ℂ ⊂ 𝕆 embeddings; scalar algebra laws for all seven algebras.
- **reproductions** — QFT against the DFT formula; exact quantum phase
  estimation; Grover success probabilities against the closed form
  (94.53% at n=3, k=2); Deutsch–Jozsa; Bernstein–Vazirani; superdense
  coding; teleportation with real mid-circuit measurement and feed-forward;
  GHZ/Mermin correlations; sedenion Born-drift by construction.
- **backends_behavior / registry_behavior / memory_scaling** — measurement
  statistics, sampling determinism across representations, adaptive
  promotion, research-extension workflows, and the memory assertions below.

## Benchmarks and width (memory) scaling

```sh
cargo bench                                  # criterion: gates.rs + width.rs
cargo run --release --example width_scaling  # memory table, incl. actual RSS
```

`benches/gates.rs` measures per-gate throughput (ns/amplitude at n=16), full
circuits (QFT, Grover, random), dense-vs-sparse on concentrated states, and
the cost of swapping the algebra (same Ry/CX ladder over ℝ, ℂ, CD⟨ℝ⟩, ℍ, 𝕆,
𝕊). `benches/width.rs` sweeps width: dense doubles per qubit; sparse GHZ is
flat out to 60+ qubits; adaptive tracks dense within noise on Grover.

Sample figures from this machine (`--quick` run, debug-free `bench`
profile):

| benchmark | result |
|---|---|
| 1q gate, dense n=16 | ~165 µs ≈ 2.5 ns/amplitude |
| 2q gate, dense n=16 | ~620 µs |
| 3q gate (`ccx`), dense n=16 | ~985 µs |
| QFT(12), dense | 2.79 ms |
| GHZ(20): dense vs sparse | 203 ms vs **5.7 µs** |
| Ry/CX ladder n=12: ℝ / ℂ / CD⟨ℝ⟩ / ℍ / 𝕆 / 𝕊 | 0.37 / 0.67 / 0.70 / 2.0 / 11.4 / 43.7 ms |

The CD⟨ℝ⟩ column is the built-from-scratch Cayley–Dickson complex running
within ~4% of `num_complex` — the generic-algebra machinery is essentially
free; wider algebras pay only their arithmetic.

Memory is asserted, not just plotted (`tests/memory_scaling.rs`): dense is
`2^n · sizeof(S)` + O(1) and doubles per qubit *and* per algebra-dimension
doubling; sparse GHZ is width-independent (125 B out to 63 qubits); saturated
sparse is strictly worse than dense; adaptive ends within 64 B of whichever
representation is cheaper. Sample of the example's output on this machine —
note the estimate matching the measured RSS delta once states are large:

```
qubits          dense     sparse (GHZ)   adaptive (GHZ)
    16        1.0 MiB            125 B            125 B
    24      256.0 MiB            125 B            125 B

qubits       estimate      RSS delta      (dense C64)
    22       64.0 MiB       64.0 MiB
    24      256.0 MiB      256.0 MiB

algebra (dense, n=16):  R 512 KiB · C 1 MiB · H 2 MiB · O 4 MiB · S 8 MiB
```

## Layout

```
src/
  scalar/        Scalar trait; f64, C64, CD<T> (ℍ/𝕆/𝕊), split-complex
  math.rs        GateMatrix<S>: matmul, dagger, controlled, kron, unitarity
  gates/         GateDef trait, FixedGate/ParamGate, standard library
  registry.rs    GateRegistry<S>: validated registration, aliases
  circuit.rs     Circuit<S> (chainable builders, raw matrices, append),
                 BoundCircuit<S> (bind-time validation, inverse())
  backend/       Backend<S> trait + dense / sparse / adaptive,
                 BackendRegistry<S>, pauli_expectation
  library.rs     bell, ghz, qft, iqft, grover, mcz, random_circuit
  sim.rs         Simulator<S>: registries + one-call execution
  rng.rs         deterministic xoshiro256++
tests/           nine integration suites (see Testing)
benches/         criterion: gates.rs, width.rs
examples/        bell, grover, exotic_algebras, research_extension,
                 width_scaling
```

Dependencies are deliberately light: `num-complex` and `rustc-hash` at
runtime; `proptest` and `criterion` for development.

## Where this is going

See [ROADMAP.md](ROADMAP.md): matrix product states (the `Backend` trait is
already shaped for them, including overridable native sampling), truncated
p-adic amplitudes (the `scale`/`born_weight` split is the designed seam),
dual numbers and other non-Cayley–Dickson scalars, mid-circuit measurement
as circuit ops, noise channels, and gate fusion (gated on
`Scalar::ASSOCIATIVE`, which is `false` from octonions onward for a reason).
