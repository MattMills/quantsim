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
  five shipped implementations: **dense** state vector (the BQP reference),
  **sparse** hash-map state, an **adaptive** backend that promotes sparse →
  dense at ¼ density, the **factored** backend (product of dense factors
  over qubit regions — memory tracks entanglement *clusters*), and **MPS**
  (matrix product states on a dependency-free Jacobi SVD — memory tracks
  Schmidt rank / *bond dimension*). Three orthogonal compression axes —
  support, clusters, bonds — all conformance-verified against dense.
- **Gates** — a registry (`name → GateDef`) with a 32-gate standard library
  (plus aliases), defined once over ℂ and projected into each algebra;
  over ℝ you automatically get the real subset. Research gates are a
  closure away and are unitarity-validated at registration. Beyond dense
  matrices, circuits accept **diagonal kernels** (`Circuit::diagonal`) for
  phase oracles and multi-controlled Z — `O(2^k)` storage and `O(states)`
  application instead of `O(4^k)`, which is the difference between a 262 KiB
  and a 4 GiB MCZ at k = 14.

- **Research mode** — a [`conformance`](src/conformance.rs) module that
  verifies *any* backend against the reference across *every* registered
  gate (custom sets discovered automatically), and a
  [`harness`](src/harness.rs) that benchmarks backends on named workloads
  with correctness re-checked in the same run. New representations and gate
  sets either pass measurably or fail with the offending gate named.
- **Evented scheduling** ([`schedule`](src/schedule.rs)) — simultaneous
  gate *loops* with periods and phases, one-shot events, and measurement
  events whose outcomes enqueue further ops: circuit time as a message
  queue, with deterministic same-tick ordering and an optional
  must-be-disjoint overlap policy. Measurement-free schedules flatten to
  circuits for equivalence testing.
- **Factored geometry** ([`FactoredState`](src/backend/factored.rs)) — the
  state as a product of factors over qubit regions: gates merge factors
  only when they couple them, measurement splits them exactly, rank-1
  detection re-separates disentangled qubits. Memory is the *sum* of
  factor sizes — non-exponential while entanglement stays hierarchically
  local (and honestly dense when it doesn't). The live factor partition and
  lifetime peak costs are inspectable: entanglement geometry as data.
- **Structure discovery** ([`discovery`](src/discovery.rs)) — find gates
  that stabilize the current state (identity up to phase), and verify
  n-wide **signal threads**: ops spliced at several points of the circuit
  graph (inject a signal here, remove it there) that are
  computation-neutral only as a whole unit, with their geometric cost
  measured on the factored backend.
- **Operational models** — [`InterferenceState`](src/backend/interference.rs)
  evolves exactly like dense while accounting constructive and destructive
  interference *independently* (per-gate path-weight vs net ledger, per-state
  destruction map; diagonals provably destroy 0; H·H destroys exactly 1.0);
  [`DeviceState`](src/backend/device.rs) reproduces the physical operation
  order of a real machine — coupling [`Topology`] (linear/ring/grid/custom),
  SWAP-routing that walks entanglement stepwise through adjacency with a
  persistent logical→physical mapping, per-qubit latency clocks, and a full
  physical op log — while answering in logical indices identical to dense.
- **VOLK-style selection** ([`harness::select_backend`]) — profile candidate
  backends on your workload on *this* machine and pick the best by time or
  memory, with fidelity as a hard gate: a fast-but-wrong kernel is rejected
  on measured deviation, never chosen.

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

`cargo test` runs 173 tests (169 across fourteen suites + 4 doctests); line
coverage is 92.6% (94.1% region) via `cargo llvm-cov`, with the remaining
gap almost entirely trivial accessors and defensive guards:

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
- **backends_behavior / registry_behavior / memory_scaling /
  error_display** — measurement statistics, sampling determinism across
  representations, adaptive promotion, research-extension workflows
  (including the trait-default `apply_diagonal` fallback for custom
  backends), every error variant's rendered message, and the memory
  assertions below.

## Benchmarks and width (memory) scaling

```sh
cargo bench                                  # criterion: gates.rs + width.rs
cargo run --release --example width_scaling  # memory table, incl. actual RSS
```

`benches/gates.rs` measures per-gate throughput (ns/amplitude at n=16), full
circuits (QFT, Grover, random), dense-vs-sparse on concentrated states, and
the cost of swapping the algebra (same Ry/CX ladder over ℝ, ℂ, CD⟨ℝ⟩, ℍ, 𝕆,
𝕊). `benches/width.rs` sweeps width: dense doubles per qubit; sparse GHZ is
flat out to 60+ qubits; adaptive tracks dense within noise on Grover
(5.6 ms vs 6.1 ms at n=14 — the sparse warm-up phase pays for itself).

Sample figures from this machine (`--quick` run, debug-free `bench`
profile):

| benchmark | result |
|---|---|
| 1q gate, dense n=16 | ~165 µs ≈ 2.5 ns/amplitude |
| 2q gate, dense n=16 | ~620 µs |
| 3q gate (`ccx`), dense n=16 | ~985 µs |
| QFT(12), dense | 2.9 ms |
| Grover(10), 8 iterations (diagonal-kernel oracle) | 0.73 ms |
| Grover(16), 100 iterations, end to end | 1.3 s |
| GHZ(20): dense vs sparse | 216 ms vs **5.7 µs** |
| Grover(14), 3 iterations: dense vs adaptive | 6.1 ms vs 5.6 ms |
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

## Research mode: verify, then measure

The workflow for a new gate set or a new backend (an MPS, a p-adic
representation, your `bbq-custom-gate-set-37`):

```rust
use quantsim::prelude::*;

let mut sim: Simulator = Simulator::new();
sim.registry_mut().register_parametric("bbq_fsim", /* … */)?;   // your set
sim.backends_mut().register("mps", |n| /* … */)?;               // your backend

// 1. Conformance: sweeps EVERY registered gate (yours included) at random
//    params/orderings/widths against the reference, plus registry-drawn
//    random circuits, Born-weight conservation, sampling equality and
//    measurement collapse. Structured report, per-gate deviations.
let report = verify_backend(&sim, "mps", &ConformanceConfig::default())?;
assert!(report.passed(), "{report}");

// 2. Benchmark: named workloads × backends → time, memory, support size,
//    speedup and memory ratios — with amplitudes re-verified in the run.
let bench = compare_backends(&sim, &[Workload::ghz(20), Workload::qft(12)],
                             &["dense", "sparse", "mps"], &BenchConfig::default())?;
println!("{bench}");
# Ok::<(), quantsim::Error>(())
```

Real output from `cargo run --release --example research_mode` (fSim-style
"bbq-37" set layered over the standard library):

```
conformance: 'sparse' vs reference 'dense' over C — PASSED
  41 gates (246 cases), 24 random circuits; max deviation 0.00e0, weight drift 6.66e-16

workload ghz-20:
  backend              time       memory   nonzeros    speedup       mem×    deviation
  dense           211.47 ms     16.0 MiB          2      1.00×       1.0×        0.0e0
  sparse           16.42 µs      125.0 B          2  12876.67×  134218.0×        0.0e0
workload bbq37-random-10q:
  dense           880.25 µs     16.0 KiB       1024      1.00×       1.0×        0.0e0
  sparse            1.37 ms     50.0 KiB       1024      0.64×       0.3×        0.0e0
  adaptive        559.98 µs     16.0 KiB       1024      1.57×       1.0×        0.0e0
```

The harness reports losses as plainly as wins (sparse is *worse* on
saturated states; adaptive tracks the better side), and the conformance
suite is itself tested against a deliberately sabotaged backend — the
report must localize the corruption to the exact gates it breaks
(`tests/conformance_harness.rs`).

## Layout

```
src/
  scalar/        Scalar trait; f64, C64, CD<T> (ℍ/𝕆/𝕊), split-complex
  math.rs        GateMatrix<S>: matmul, dagger, controlled, kron, unitarity
  gates/         GateDef trait, FixedGate/ParamGate, standard library
  registry.rs    GateRegistry<S>: validated registration, aliases
  circuit.rs     Circuit<S> (chainable builders, raw + diagonal kernels,
                 append), BoundCircuit<S> (bind-time validation, inverse())
  backend/       Backend<S> trait + dense / sparse / adaptive / factored,
                 BackendRegistry<S>, pauli_expectation
  schedule.rs    evented scheduler: simultaneous loops, events, feedback
                 (backends also: factored, interference, device, mps)
  conformance.rs registry-wide backend verification (research safety net)
  harness.rs     workload benchmarking with in-run correctness checks
  discovery.rs   point stabilizers, signal threads, transparency reports
  library.rs     bell, ghz, qft, iqft, grover, phase_flip, random_circuit
  sim.rs         Simulator<S>: registries + one-call execution
  rng.rs         deterministic xoshiro256++
tests/           fourteen integration suites (see Testing)
benches/         criterion: gates.rs, width.rs
examples/        bell, grover, exotic_algebras, research_extension,
                 research_mode, evented_memory, width_scaling
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
