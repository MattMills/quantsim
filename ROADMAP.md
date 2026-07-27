# Roadmap

The crate is organized around two swappable axes — the **amplitude algebra**
(`Scalar`) and the **state representation** (`Backend<S>`) — so most planned
work is "fill in another cell of the matrix":

| representation \ algebra | ℝ | ℂ | Ball | split-ℂ | ℍ | 𝕆 | 𝕊 | ℚ_p |
|--------------------------|---|---|------|---------|---|---|---|-----|
| dense state vector       | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | design below |
| sparse state vector      | ✅ | ✅ | ✅ (midpoint pruning) | ✅ | ✅ | ✅ | ✅ | design below |
| adaptive (sparse→dense)  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| matrix product state     | ✅ | ✅ | ✅ | — | open (noted below) | open | — | open |
| hierarchy (mera)         | ✅ | ✅ | ✅ | — | open | open | — | open |

## Matrix product states (MPS) — SHIPPED (core)

`MpsState` is live: site-tensor chain over a dependency-free one-sided
Jacobi SVD (`math::jacobi_svd`/`svd_thin`), contiguous gate windows to
arity 5 with SWAP routing for non-adjacent operands, truncation by
relative tolerance under a hard bond cap (approximation as a *measured*
dial — see `tests/mps.rs::truncation_degrades_measurably_never_silently`),
environment-based measurement and norm, a native conditional sampler
(`sample_mps`), and bond-dimension instrumentation. Registered as `"mps"`,
swept by the conformance suite over the full registry. Remaining MPS work:

- canonical-form maintenance (environments are currently recomputed on
  demand — `O(n·χ³)` per measurement instead of `O(χ³)`);
- quaternionic MPS (the Jacobi rotations assume commuting scalars;
  construction over ℍ fails loudly today);
- smarter routing (current greedy adjacent-swap walk) and two-site
  variational compression;
- `load` past 12 qubits (state compilation currently materializes the
  dense vector).

Original design notes, kept for the record — the `Backend<S>` trait was
shaped with MPS in mind:

- `apply` maps to one-site (arity 1) and two-site (arity 2, adjacent after
  swap-routing) tensor updates; gates of arity ≥ 3 decompose or fall back to
  contraction.
- `amplitude(index)` is an `O(n·χ²)` chain contraction — cheap.
- `for_each_nonzero` is documented as potentially expensive precisely so an
  MPS backend can implement it by enumeration only when a caller insists.
- `sample` is a provided method that an MPS backend should **override** with
  the standard conditional one-qubit sweep (`O(shots · n · χ²)`), which needs
  no global enumeration.
- `memory_bytes` reports `Σ bond dimensions`, which is the whole point:
  width benchmarks will show polynomial memory on low-entanglement circuits.

The validation and measurement side is already in place: when an MPS
backend lands it registers by name and goes through
`conformance::verify_backend` (registry-wide gate sweep + invariants,
tested to catch sabotaged backends) and `harness::compare_backends`
(time/memory/support vs reference with in-run correctness checks) with no
new harness code.

Blockers and plan:

1. **SVD, dependency-free.** Truncation needs a singular value
   decomposition. Plan: one-sided Jacobi over ℝ/ℂ (simple, numerically
   robust, no external linear-algebra dependency — consistent with the
   crate's zero-heavy-deps stance).
2. **Per-algebra feasibility.** Jacobi SVD generalizes to ℍ (quaternion SVD
   is standard). Octonions do not have a classical SVD (non-associativity);
   an 𝕆-MPS is a research question — exactly the kind the crate wants to
   make askable, but it will not block ℝ/ℂ/ℍ MPS.
3. Two-site gates on non-adjacent qubits via swap networks first; smarter
   routing later.

## p-adic amplitudes (ℚ_p)

A truncated p-adic scalar (`p` fixed, `N` digits, arithmetic mod `p^N`) fits
the `Scalar` shape *except* for the two places the trait currently assumes an
ℝ-algebra:

- `scale(k: f64)` — meaningless over ℚ_p. Plan: split a `RealScale`
  sub-trait out of `Scalar`; renormalization-dependent provided methods
  (`measure`) move behind it, and p-adic backends override measurement
  semantics entirely.
- `born_weight` — the natural candidate is the p-adic absolute value
  `|x|_p = p^{-v(x)}` mapped into `f64` (it is real-valued by definition),
  following the p-adic QM literature (Vladimirov–Volovich). Whether that
  yields a sensible measurement theory is the research question; the hook is
  already separate from `abs_sqr` for exactly this reason.
- Unitarity: validated against the chosen norm rather than `U†U = I` over ℝ —
  registry validation becomes a `Scalar`-provided hook.

Sparse-first: p-adic states of interest tend to be supported on few basis
states, and the sparse backend needs nothing but the scalar to work.

## Geometric decomposition research (on top of `FactoredState`)

The factored backend ships the *hierarchical separation* substrate: exact
merge-on-demand, exact post-measurement splits, rank-1 re-separation, and
peak-cost instrumentation. The open research directions it was built to
host:

- **Geometric overdefinition** — overlapping regions holding redundant
  marginals, so that entanglement across a cut can live in more than one
  factor at once (the current partition is strict). Requires a consistency
  rule between overlapping factors; the natural first target is
  tree-structured overlaps (which is also the bridge to tree tensor
  networks / MPS below).
- **Beyond rank-1 splits** — Schmidt-rank-k factor boundaries (bond
  indices), at which point `FactoredState` *becomes* a general tensor
  network; needs the Jacobi SVD planned for MPS.
- **Local frames/gauges — per-qubit SHIPPED** (`FramedState`): deferred
  1q basis metadata over any inner backend, absorption/conjugation/lazy
  flush, `adopt_frame`/`release_frame`, stats and peak-inner-memory
  instrumentation; conformance-verified over sparse/dense/mps inners.
  Measured payoff: transverse-field bulk at stored support 1 vs 2^n raw
  (>100× peak memory).
- **Clifford frames — SHIPPED** (`CliffordFramedState`): the frame group
  upgraded to the full Clifford group as a stabilizer tableau (images of
  `X_q`/`Z_q` under `C†(·)C` as signed u64-mask Pauli strings) plus a
  replay log for flush — synthesis of a minimal Clifford circuit from the
  tableau is deliberately deferred (see next rungs). Gates are recognized
  numerically (Pauli-basis decomposition, `O(8^k)` for k ≤ 3): Clifford →
  absorb; single-axis → native sparse Pauli rotation through the tableau
  (`SparseState::apply_pauli_rotation`, `O(support)`, ≤2× growth,
  weight-independent); diagonal k ≤ 5 → Walsh–Hadamard Z-string split;
  generic 1q → ZYZ; else flush + raw. Measured: pure-Clifford streams at
  stored support 1 (Gottesman–Knill via frames, 50 qubits, µs); Clifford+T
  scaffolds bounded by 2^t in T-count t with t=10/n=20 peaking at 128 vs
  2^20 raw sparse; deviation vs dense at machine precision. The
  Gottesman–Knill label is proven honest from both sides in
  `tests/clifford_frames.rs`: the absorbed set is *exactly* the Clifford
  subgroup (bidirectional sweep against an independent dense
  Pauli-normalizer check — the free sector cannot exceed GK by
  construction), full amplitude-vector extraction still flushes (not a GK
  capability), **native Pauli measurement is SHIPPED**
  (`measure_pauli` / `measure` project the stored state through the
  tableau, no flush, outcomes seed-identical with dense), and **frame
  repair on measurement is SHIPPED**: after each projection the frame
  right-composes with a repair Clifford (S on Y bits, CX fold, one H)
  so the measured string is Z-type in the new stored basis — the true
  tableau measurement update. The pre-repair envelope (`m` measurements
  ⇒ ≤ 2^m stored support, caught by the instrumentation after the
  original poly over-claim) is repaired flat: peak 2 across 40
  sequential measurements at width 40, final support 1, with the
  unrepaired path still selectable (`set_measure_repair`) and its
  envelope still pinned for comparison.
- **Dimensional lift — SHIPPED** (`lift::to_clifford_feedback`): any
  Clifford+T circuit rewritten as an evented feedback loop on `n + t`
  qubits — resource ancillas, all-Clifford unitaries, native
  measurements, outcome-conditioned Clifford corrections; exact
  including per-outcome phases; `Upfront` vs `JustInTime` resource
  scheduling. The measured finding here has moved once, exactly as the
  instrumentation intended. Original (`tests/clifford_lift.rs`,
  pre-repair): stored-basis drift under projection made *both* orderings
  peak far above the direct rotation route (8192 vs 16 on the seeded
  reference instance). **Frame repair on measurement — SHIPPED** closed
  it: the designed-to-fail assertion fired, and the repaired story is
  pinned instead — `JustInTime` runs the whole loop at a peak
  comparable to the direct route (one `|T⟩` in flight, every gadget
  measurement repaired away: consumption is *cheap*), while `Upfront`
  pays `2^t` for *holding* every resource state in one sparse register
  at once. Remaining rungs, each with measured motivation:
  **stabilizer-rank compression / frame-aligned sums** (the stored state
  as a short sum of frame-aligned terms, ≈2^{0.4t} vs the crude 2^t
  product bound — and the natural home for magic-state consumption),
  **frames over factored inners** (`|T⟩^⊗t` is linear to hold in the
  factored backend today — 1.6 KiB at t=32 — composing that with the
  tableau would make the resource cheap to hold *and* consume in the
  same run, collapsing the Upfront/JustInTime gap),
  **log compaction** (tableau →
  minimal Clifford circuit synthesis, replacing replay of the full log —
  measurement repairs now prepend to the log, so long adaptive runs
  raise its value),
  **per-factor multi-qubit frames** (frames over a factor's whole region —
  can absorb CX-like inject/remove pairs, making parity signal threads
  representation-free), and **MPS bond gauges** (the tensor-network
  analogue).
- **Scheduling-aware geometry** — the evented scheduler knows *when*
  regions interact; a lookahead pass could pre-plan merges/splits (or
  memory swap-outs) to minimize peak factor width over the whole schedule.

## The hierarchical register (MERA program)

**Rung 1 — SHIPPED** (`MeraState`, registered `"mera"`): the register as
a balanced binary isometry tree (the MERA family's isometry layer — a
tree tensor network, stated honestly), bonds capped per super-site,
gates costing the smallest subtree spanning their targets
(materialize-block up to `max_block`, apply, re-compress by recursive
Schmidt splits with the discarded weight ledgered), and the
coarse-evaluate-then-refine API: `coarse_state(depth)` — the exact
state on the super-site basis at any scale (measured flagship: GHZ(16)
at depth 1 *is* a maximally entangled super-site pair, both coarse
singular values exactly 1/√2) — with `refine_basis` as the per-site
descent dictionary. Conformance-swept over the full registry;
truncation degrades measurably, never silently; composes with `Ball`
so representation resolution and numeric resolution stack. The
remaining rungs, in dependency order:

- **Path updates instead of block materialization** — apply a cross-cut
  2q gate as its operator-Schmidt sum (rank ≤ 4) of single-site terms,
  then hierarchical rounding along the tree path (bond direct sums +
  SVD truncation): removes the `2^{block}` transient, making GHZ-across-
  the-root bond-2 *during* the gate, not just after.
- **Gauge maintenance** — keep the tree root-canonical so per-node
  truncation weights are environment-correct and the discarded ledger
  becomes a *certified* global L2 bound (today it is block-local, like
  the MPS backend's, and labeled as such).
- **Disentanglers** — the u-layer between levels (the MERA proper):
  variationally chosen to minimize truncation across cuts; the
  measured payoff target is bond growth on critical/area-law states
  where the plain tree pays χ inflation.
- **Ascending superoperators** — renormalize *operators* through the
  isometry (and eventually disentangler) layers so expectation values
  and gate effects can be *evaluated at a chosen depth* without
  touching the leaves at all: "any operation at coarse resolution"
  moving from state queries to full operator flow.
- **Progressive residual refinement** — keep truncated components as
  addressable residuals so a coarse run can be *continued* to finer
  resolution without re-simulating from scratch — the last step of
  "refine infinitely", currently approximated by re-running at a finer
  dial (larger `max_bond`, finer `Ball::quantize`).

## Further non-Cayley–Dickson explorations

`SplitComplex` establishes the pattern (indefinite Born form surfaced through
`born_weight`, `DIVISION = false` capping expectations). Natural next
entries, each a small self-contained `Scalar` impl plus tests:

- **Dual numbers** (`ε² = 0`): nilpotents; automatic-differentiation-flavored
  simulation (state and its parameter-derivative propagate together).
- **Split-quaternions** (`CD<SplitComplex>` — already constructible today;
  needs tests and a written-up example).
- **Bicomplex / tessarines**: commutative with zero divisors.
- **Clifford-algebra scalars** Cl(p, q): connects to stabilizer-adjacent
  research.
- **Group algebras** ℝ[G] for small finite G.

## Operational-model extensions

- **Device realism** — the `DeviceState` routing/latency model is exact and
  noiseless; natural next steps are per-edge gate fidelities and idle
  decoherence (needs the density/trajectory machinery below), calibration
  data import, and smarter routing (lookahead / SABRE-style) measured
  against the current greedy BFS walk by `swap_count`/`elapsed`.
- **Interference histories** — `InterferenceState` aggregates per gate and
  per output state; a Feynman-path variant keeping (bounded) contribution
  histories would let destructive interference be attributed to *pairs of
  paths*, at exponential cost in tracked paths — a good fit for the sparse
  representation where path counts stay small.
- **Selection persistence** — `select_backend` profiles per call; caching
  choices per (workload shape, width, machine) à la VOLK profiles is a
  small serialization feature once serde lands.

## Reference and certification

- **Exact D[ω] evaluator — SHIPPED** (`exact::ExactState`): absolute
  reference values for the Clifford+T fragment and eighth-turn
  rotations, with checked `i128` coefficients. Natural next entries:
  a bigger-integer feature (arbitrary-precision coefficients behind a
  feature flag, lifting the depth ceiling), exact evaluation of the
  scheduler's feedback runs (outcome-conditioned exact branches), and
  extending the ring (e.g. `D[ζ_{16}]` for π/8-family gates — the
  Clifford-hierarchy next level).
- **Ball arithmetic — SHIPPED** (`scalar::Ball`): certified
  midpoint-radius amplitudes, quantization as a coarse-graining dial,
  containment-tested against the exact evaluator. Next: directed
  rounding (replacing conservative ε-inflation with true IEEE bounds),
  affine arithmetic to tame the measured ~√2-per-H dependency growth,
  and radius-aware sparse pruning (today pruning consults midpoints —
  dense is the certified path, and the docs say so).

## Simulator features

- **Recursive measurement feedback — SHIPPED** in the evented scheduler
  (`FeedbackOp`: branches carry gates *and further measurements*,
  adaptive trees of any bounded depth with structural termination).
  Still open on top of it: mid-circuit measurement **as circuit
  operations** (classical registers, conditions on *functions* of
  several outcomes, feed-forward in `Circuit` itself rather than only
  `Schedule`), and unbounded repeat-until-success loops (cyclic
  feedback bounded by the horizon rather than by tree depth).
- Noise channels (Kraus operators) — likely a `DensityBackend<S>` or
  trajectory sampling over the existing pure-state backends.
- More structured kernels beyond `GateKernel::Diagonal` (shipped):
  permutation kernels (X/CX/Toffoli families as index maps — no arithmetic
  at all) and controlled-sparse kernels, all behind the same
  `BoundCircuit`/`Backend` seam.
- Gate fusion (adjacent 1q gates; 1q-into-2q) for the dense backend. Note:
  fusion is an *associativity* optimization and must stay disabled for
  non-associative algebras — the `Scalar::ASSOCIATIVE` flag exists partly
  for this.
- Multithreading the dense apply loop (rayon or hand-rolled scoped threads)
  behind a feature flag.
- Circuit serialization (serde feature) once the op format stabilizes.

## Explicitly rejected for now

- GPU backends: worth doing only after the CPU dense path stops being the
  bottleneck for the widths we care about.
- A *standalone* tableau-only stabilizer backend: the hybrid shipped
  instead (`CliffordFramedState` — tableau metadata over amplitude
  storage), which degrades gracefully on non-Clifford gates rather than
  refusing them. A pure group-theoretic backend only wins once circuits
  are Clifford-only end to end, where the hybrid already stores one
  amplitude.
