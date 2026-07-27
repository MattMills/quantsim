# quantsim

An in-house research quantum simulator in Rust, built so the interesting
parts are swappable:

- **Amplitude algebra** — every gate matrix, state and measurement rule is
  generic over a [`Scalar`](src/scalar/mod.rs) trait. Shipped: ℝ, ℂ
  (default), a generic **Cayley–Dickson doubling** `CD<T>` giving
  quaternions ℍ, octonions 𝕆 and sedenions 𝕊, **split-complex** as a
  first non-Cayley–Dickson algebra, and **[`Ball`](src/scalar/ball.rs)** —
  coarse-grained certified arithmetic (complex midpoint ± certified
  radius; midpoints track `C64` bit-for-bit, radii propagate soundly, and
  `quantize` is a deliberate resolution dial). Planned (see
  [ROADMAP](ROADMAP.md)): truncated p-adics, dual numbers,
  split-quaternions, Clifford scalars.
- **State representation** — a [`Backend<S>`](src/backend/mod.rs) trait with
  six shipped implementations: **dense** state vector (the BQP reference),
  **sparse** hash-map state, an **adaptive** backend that promotes sparse →
  dense at ¼ density, the **factored** backend (product of dense factors
  over qubit regions — memory tracks entanglement *clusters*), **MPS**
  (matrix product states on a dependency-free Jacobi SVD — memory tracks
  Schmidt rank / *bond dimension*), and **mera** (a hierarchical
  isometry tree — memory tracks *renormalization structure*; see below).
  Four orthogonal compression axes — support, clusters, bonds,
  hierarchy — all conformance-verified against dense.
- **Exact reference** ([`exact`](src/exact.rs)) — a `D[ω] = ℤ[1/√2, e^{iπ/4}]`
  evaluator (checked `i128` coefficients) for the Clifford+T fragment and
  all standard rotations at eighth-turn angles: **absolute** reference
  values with no float error at all. Unitarity holds *exactly*, exact
  zeros are decidable (`H·T⁴·H` leaves a measured ~1e-16 residue on the
  float path and exactly 0 here), and `max_deviation_vs` turns any
  backend's output into an absolute error measurement — including the
  dense reference's own, and the certification of `Ball` radii.
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
  events whose outcomes enqueue further [`FeedbackOp`]s: gates, or
  **further measurements with their own branches, recursively** — adaptive
  measurement trees of any bounded depth (outcome-dependent measurement
  choices, bounded repeat-until-success ladders) inside one schedule,
  with *structural* termination (branches are finite owned trees).
  Circuit time as a message queue, with deterministic same-tick ordering
  and an optional must-be-disjoint overlap policy. Measurement-free
  schedules flatten to circuits for equivalence testing.
- **Factored geometry** ([`FactoredState`](src/backend/factored.rs)) — the
  state as a product of factors over qubit regions: gates merge factors
  only when they couple them, measurement splits them exactly, rank-1
  detection re-separates disentangled qubits. Memory is the *sum* of
  factor sizes — non-exponential while entanglement stays hierarchically
  local (and honestly dense when it doesn't). The live factor partition and
  lifetime peak costs are inspectable: entanglement geometry as data.
- **Hierarchical register** ([`MeraState`](src/backend/mera.rs)) — the
  state as a **renormalization hierarchy**: a balanced binary isometry
  tree (the MERA family's isometry layer — a tree tensor network, with
  disentanglers as the named next rung), bonds capped per super-site,
  gates costing exactly the smallest subtree spanning their targets.
  The coarse-evaluate-then-refine API is the point:
  `coarse_state(depth)` is the exact state on the super-site basis at
  any scale — the 16-qubit GHZ at depth 1 **is** a two-super-site
  maximally entangled pair (both coarse singular values exactly 1/√2,
  pinned in the tests) — and `refine_basis` expands any super-site one
  level, down to physical amplitudes. Truncation is a measured dial
  (`discarded_weight`, `is_exact`), width-40 registers run in kilobytes
  while entanglement stays hierarchy-local, and the whole thing composes
  with `Ball` so coarse representation and certified coarse arithmetic
  stack.
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
  order of a real machine — coupling [`Topology`] with real register
  geometries (`heavy_hex_falcon27`, `sycamore_like`, `complete` all-to-all,
  plus linear/ring/grid/custom), SWAP-routing that walks entanglement
  stepwise through adjacency with a persistent logical→physical mapping,
  per-qubit latency clocks driven by a [`LatencyMap`] (era-representative
  `DurationModel` presets + per-site/per-edge calibration overrides), an
  injectable inner representation (chip-scale geometry over a sparse
  inner: GHZ across all 54 Sycamore sites), a full physical op log, and
  `elapsed`/`serial_time` as the measured parallelism ratio — while
  answering in logical indices identical to dense.
- **Local frames** ([`FramedState`](src/backend/frames.rs)) — deferred
  per-qubit basis changes as representation metadata over *any* inner
  backend: 1q gates absorb for free (inverse pairs cancel without touching
  amplitudes), multi-qubit gates are conjugated into the frame and take the
  diagonal fast path when the frame diagonalizes them, observation flushes
  lazily. Entanglement is local-unitary-invariant — frames buy the
  *basis-dependent* costs: an `h`/`rx`/`rxx` transverse-field bulk keeps
  stored support at **1** where raw sparse pays `2^n` peak (tested at
  >100×). `adopt_frame` rewrites representation without touching physics —
  the concrete mechanism behind computation-transparent structure insertion.
- **Clifford frames** ([`CliffordFramedState`](src/backend/clifford_frame.rs))
  — the frame group upgraded from ⊗U(2) to the full Clifford group, held as
  a stabilizer tableau + replay log over a sparse core. Gates are
  *numerically recognized* and routed: Cliffords absorb into the frame
  (pure metadata — Gottesman–Knill's **evolution sector** falls out:
  50-qubit Clifford streams at stored support 1; the absorbed set is
  proven **exactly** the Clifford subgroup by a bidirectional sweep
  against an independent dense Pauli-normalizer check, so nothing
  non-Clifford rides free), Pauli measurements run natively through the
  tableau (no flush, `measure_pauli`) **with frame repair** — after each
  projection the frame becomes `C·V` for a repair Clifford chosen so the
  measured string is Z-type in the new stored basis, the true tableau
  measurement update, so projection growth no longer compounds (measured:
  peak stored support 2 across 40 sequential measurements at width 40,
  where the unrepaired path — still selectable, still pinned in the
  tests — pays `2^m`), Pauli-axis rotations conjugate through the tableau
  onto **native sparse Pauli-string rotations** (`O(support)`, ≤2× growth,
  weight-independent), diagonals Walsh-decompose into Z-string rotations,
  generic 1q gates split ZYZ; anything else flushes and goes raw. The
  stored cost is bounded by `2^t` in the **T-count** `t` — not width, not
  gate count (measured: t=10 at n=20 peaks at 128 amplitudes where raw
  sparse pays 2^20) — making "magic" the measured currency left over once
  the Clifford part of a circuit becomes free.
- **The dimensional lift** ([`lift`](src/lift.rs)) — rewrite a Clifford+T
  circuit as a **feedback loop in an enlarged Clifford space**: `n + t`
  qubits, every unitary Clifford, each T executed by gate teleportation
  (resource ancilla → CX → native measurement → outcome-conditioned
  Clifford correction, as an evented `Schedule`). Exact including
  per-outcome phases, and honestly instrumented — twice. The original
  finding: without frame repair, projections drift the stored basis and
  both prep orderings peaked at 8192 where the direct run peaked 16.
  Frame repair (the roadmap rung that finding motivated) landed, the
  pinned assertion **fired as designed**, and the measured story now
  reads: just-in-time prep runs the whole loop at a peak comparable to
  the direct route (one `|T⟩` in flight, each measurement repaired
  away — consumption is *cheap*), while upfront prep pays `2^t` for
  *holding* all resource states at once. The magic stays linear to hold
  in the factored backend; composing that with the frame is the
  remaining rung, alongside stabilizer-rank storage.
- **VOLK-style selection** ([`harness::select_backend`]) — profile candidate
  backends on your workload on *this* machine and pick the best by time or
  memory, with fidelity as a hard gate: a fast-but-wrong kernel is rejected
  on measured deviation, never chosen.
- **The resource guard** ([`guard`](src/guard.rs)) — over-scale inhibition
  as a property of the library, automatic for every representation.
  Large allocations are **admitted against measured capacity** (cgroup
  limit / `MemAvailable`, read at allocation time) with fallible
  reservation as backstop: an inadmissible request fails with
  `OutOfMemory { requested, available }` — both numbers measured, never
  a presumed width constant. The former capacity constants
  (`DENSE_MAX_QUBITS` & co.) are structural index bounds only; adaptive
  promotion consults real capacity and stays sparse when dense wouldn't
  fit *this* machine right now. `guard::set_time_budget` arms a
  wall-clock budget that the long kernels (dense/sparse/exact sweeps,
  Jacobi SVD, merges) checkpoint *inside* their loops: an over-scale
  run aborts mid-gate with `Timeout { budget, elapsed }` instead of
  being pre-skipped on a cost estimate.
  [`capacity_probe`](examples/capacity_probe.rs) verifies the guard
  against reality: subprocess-isolated width walks per axis until the
  actual wall — guard refusals with measured numbers, deadline aborts,
  and (with admission disabled) real OOM kills observed by signal with
  peak RSS recorded.

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

`cargo test` runs 250 tests (43 unit + 203 across twenty-two integration
suites + 4 doctests);
line coverage is 90%+ via `cargo llvm-cov`, with the remaining gap almost
entirely trivial accessors and defensive guards:

- **conventions** — bit order and control placement pinned on basis states.
- **clifford_frames** — the native sparse Pauli-string rotation kernel vs
  dense matrices over every Pauli mixture and sign (and the ℝ-subset rule:
  odd-Y strings are real); full-registry conformance through the frame;
  the `2^t` T-count bound with raw-sparse comparison; replay-log ordering
  under interleaved absorb/rotate, exact to global phase. Plus the
  **Gottesman–Knill honesty proofs**: absorption ⟺ Clifford membership,
  bidirectionally, against an independent dense Pauli-normalizer ground
  truth over the whole registry (and at Clifford angles of parametric
  gates); width/depth cost of the free sector asserted polynomial
  (support 1 at width 63, log-linear at depth 4000); and the boundary
  pinned from both sides — T scatters amplitudes the moment it arrives,
  native measurement is seed-identical with dense, **frame repair holds
  adaptive sequences flat** (peak 2 across 40 measurements at width 40;
  the unrepaired `2^m` envelope stays pinned alongside, with same-seed
  outcome equality between the two), and full amplitude extraction
  flushes. The repair steps' conjugation rules are unit-tested against
  the dense Pauli-decomposition ground truth.
- **clifford_lift** — the dimensional lift verified exactly (data
  register vs unlifted dense, per-outcome phases divided out, both
  resource orderings); the feedback loop shown all-Clifford (zero
  flushes, corrections fire exactly on outcome 1); the cost *location*
  measured **twice**: the original drift finding was pinned by an
  assertion designed to fail loudly the day a representation change made
  the lift win — frame repair made it fire as intended, and the test now
  pins the repaired story (JIT ≈ direct; upfront = the `2^t` holding
  cost; the drift preserved on a repair-off backend for comparison).
- **adaptive_feedback** — recursive measurement trees in the scheduler:
  nested events fire exactly on the selecting outcome (GHZ-correlated),
  a depth-3 repeat-until-success ladder with per-seed retry accounting,
  same-tick branch ordering, horizon pruning of whole subtrees, and
  adaptive replay determinism across backends.
- **exact_reference** — the dense reference's float error measured
  against the `D[ω]` ring (absolute, not relative); exact unitarity;
  exact zeros where floats leave ~1e-16 residue; GHZ amplitudes exactly
  1/√2; Grover through diagonal oracles vs the closed form; overflow
  fails loudly.
- **ball_certification** — every final amplitude ball *contains* the
  exact ring value, at full resolution and under quantized (coarse)
  gates; refining the grid shrinks certified radii; midpoints reproduce
  the `C64` simulator bit-for-bit; the interval dependency growth is
  measured and documented, not hidden.
- **mera** — full-registry conformance; the GHZ coarse view **is** a
  maximally entangled super-site pair; coarse views conserve weight at
  every depth; truncation degrades measurably, never silently; gate cost
  is the spanning subtree with honest caps; width-40 hierarchy-local
  circuits in kilobytes; composition with `Ball`.
- **capacity** — the resource guard as behavior: over-scale allocations
  refused by *measurement* (requested vs available bytes in the error,
  auto-measured and under explicit limits); adaptive stays sparse when
  dense is inadmissible and promotes when the limit lifts; sparse
  growth, mera blocks and factored merges all admitted not presumed;
  armed time budgets abort a single dense gate mid-sweep, the mera SVD
  path and scheduled runs — promptly, with measured elapsed times — and
  the identical runs complete once the budget lifts.
- **device_geometries** — real machines reproduced structurally
  (Falcon-27 heavy-hex: 27 sites, 28 couplers, degree ≤ 3, the known
  adjacencies; Sycamore-class 54-site diagonal lattice; ion-trap
  all-to-all) and operationally: swap cost is a property of the coupling
  map while the amplitudes stay bit-for-bit on the reference; a
  per-coupler latency override on the critical path moves the clock by
  *exactly* the override and off-path overrides move nothing;
  parallelism is the measured `serial_time/elapsed` ratio; the three era
  clocks rescale an *identical* physical op sequence; chip-scale
  geometry (GHZ across all 54 Sycamore sites) runs over a sparse inner
  in under a megabyte.
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
representation is cheaper. Capacity itself is tested as behavior
(`tests/capacity.rs`): over-scale allocations refuse with measured
requested/available bytes, adaptive stays sparse under a tight budget,
and armed time budgets abort dense sweeps, SVDs and schedules mid-kernel
with measured elapsed times. `cargo run --release --example
capacity_probe` walks every axis to its real wall on your machine. Sample of the example's output on this machine —
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
  scalar/        Scalar trait; f64, C64, CD<T> (ℍ/𝕆/𝕊), split-complex,
                 Ball (certified midpoint ± radius, quantize dial)
  math.rs        GateMatrix<S>: matmul, dagger, controlled, kron, unitarity
  exact.rs       D[ω] ring + ExactState: absolute Clifford+T reference
  guard.rs       resource guard: measured memory admission, time budgets
  gates/         GateDef trait, FixedGate/ParamGate, standard library
  registry.rs    GateRegistry<S>: validated registration, aliases
  circuit.rs     Circuit<S> (chainable builders, raw + diagonal kernels,
                 append), BoundCircuit<S> (bind-time validation, inverse())
  backend/       Backend<S> trait + dense / sparse / adaptive / factored /
                 mps / mera / interference / device / frames /
                 clifford_frame, BackendRegistry<S>, pauli_expectation
  schedule.rs    evented scheduler: simultaneous loops, events, recursive
                 measurement feedback (adaptive trees)
  lift.rs        Clifford+T → measurement-feedback loop on n+t qubits
  conformance.rs registry-wide backend verification (research safety net)
  harness.rs     workload benchmarking with in-run correctness checks
  discovery.rs   point stabilizers, signal threads, transparency reports
  library.rs     bell, ghz, qft, iqft, grover, phase_flip, random_circuit
  sim.rs         Simulator<S>: registries + one-call execution
  rng.rs         deterministic xoshiro256++
tests/           twenty-two integration suites (see Testing)
benches/         criterion: gates.rs, width.rs
examples/        bell, grover, exotic_algebras, research_extension,
                 research_mode, evented_memory, width_scaling,
                 verify_models, frames_demo, clifford_space, clifford_lift,
                 coarse_register (mera + Ball), absolute_reference (D[ω]
                 vs every backend), adaptive_feedback (recursive trees +
                 frame repair), capacity_probe (real walls, measured),
                 device_reproduction (real geometries × latency maps)
```

Dependencies are deliberately light: `num-complex` and `rustc-hash` at
runtime; `proptest` and `criterion` for development.

## Where this is going

See [ROADMAP.md](ROADMAP.md): stabilizer-rank compression for the Clifford
frame (the crude `2^t` product bound is not the ≈`2^{0.4t}` state of the
art — the gap is measurable here) and frames over factored inners (the
lift's remaining `2^t` is a *holding* cost, measured), the MERA completion
(disentanglers, path updates that replace block materialization, gauge
maintenance so truncation bounds certify, ascending superoperators so
operators renormalize instead of blocks), per-factor and MPS-bond-gauge
frames, truncated p-adic amplitudes (the `scale`/`born_weight` split is
the designed seam), dual numbers and other non-Cayley–Dickson scalars,
classical registers for the scheduler's feedback trees, noise channels,
and gate fusion (gated on `Scalar::ASSOCIATIVE`, which is `false` from
octonions onward for a reason).
