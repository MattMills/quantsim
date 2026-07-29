# Roadmap

(Machinery lives here; the staged catalog of target *computations* — with
honest advantage labels and their alternate-landscape axes — lives in
[LANDSCAPES.md](LANDSCAPES.md).)

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
- `load` without dense materialization (compilation currently builds the
  full vector, guard-admitted against measured memory).

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

## Hierarchical algebraic registers — SHIPPED (core)

`qudit::AlgebraicRegister` breaks the flat register into a varied qudit
structure: a site sector under any backend plus an algebra sector — up
to 3 further logical qubits carried inside each Cayley–Dickson scalar —
with dual-algebra gate synthesis (`synthesize_sandwich`: gates as sums
of `(a·x)·b`) and measured routing (embedded-ℂ component-linearity per
scalar decides native vs component paths). Measured findings worth
keeping:

- the sandwich span `{L_{eᵢ} R_{eⱼ}}` is FULL at every doubling level
  (16/16 on ℍ, 64/64 on 𝕆, 256/256 on 𝕊, residuals ~1e-15): neither
  non-associativity nor zero divisors cost the multiplication algebra
  anything — every ℂ-linear qudit gate is a two-sided multiplication
  sum. A Hadamard on the ℍ-qudit is exactly two terms.
- the embedded-ℂ action stops being component-linear at 𝕆 (the CD
  twist conjugates a coordinate) — measured at construction, never
  assumed; ℍ splits run site gates natively on the inner backend.
- 66 exact logical qubits (63 sparse sites × 𝕊-qudit) — past the u64
  flat-indexing ceiling of every conventional register — addressed as
  (site, component) parts, conformance-swept and D[ω]-certified.

Second wave — SHIPPED:

- **the fifth doubling measured**: `Trigintaduonion = CD⟨𝕊⟩` (32-dim,
  4 algebra qubits) keeps the sandwich span FULL — 1024/1024, residual
  ~5e-15 (`#[ignore]`d test, ~17 s; run with `-- --ignored`). The
  conjecture this data supports: the CD tower's multiplication algebra
  is full `End_ℝ` at *every* level.
- **sandwich-native execution**: algebra-sector gates now EXECUTE as
  two-sided multiplications on the stored scalars — cached synthesis
  per (gate, bits), machine-precision dust snapped (relative 1e-13) so
  exact zeros stay exact, `QuditStats::sandwich_gates` counting, a
  toggle for A/B against the component path, and a dimension cap
  (default 16) so 32-dim scalars don't pay the 17 s solver implicitly.
- **multi-block algebra sectors**: `DirectSum<T, U>` — several
  independent qudit blocks per scalar. The measured boundary: the span
  of blockwise sandwiches is exactly the block-diagonal algebra (32/64
  on ℍ⊕ℍ; cross-block gates residual ~1) — algebra structure IS the
  synthesis boundary, and the register routes such gates through the
  component path with full conformance. Initialization corrected for
  non-CD scalars (one() = (1,1) is not e₀ — detected and re-seeded).
- **wide measurement statistics**: `sample_parts`/`probability_parts` —
  deterministic Born sampling over (site, component) parts at any
  logical width (ghz-67 sampled at 2000 shots).

Remaining rungs:

- **structured site sectors without materialization**: the component
  path gathers and reloads, which flattens factored/MPS inners;
  incremental per-entry updates would let cluster and bond structure
  survive algebra-sector gates.
- CD⟨CD⟨𝕊⟩⟩ and beyond: does the span stay full at 64-dim (5 algebra
  qubits)? The basis SVD is 4096² — needs a smarter rank probe than
  dense Jacobi.
- mixed direct sums of unequal blocks (ℍ⊕𝕆 has 6 complex components —
  a non-power-of-two qudit; the register currently requires 2^k).

## The boundary atlas — SHIPPED (core)

`bounds::advantage_scan` makes the crate's central question operational:
each representation is an assumption about structure, measured to be
exponential exactly in its own resource, and a circuit family is an
advantage candidate precisely when every measured axis grows
exponentially at once (random universal circuits: bases 1.7–2.0, all
probes exact). The scan rediscovers GHZ/QFT/rainbow/Clifford as
classical from bytes alone and is the standing detector for a
sub-exponential simulation: any new registered representation that
keeps a flat, exact axis on the candidate family has found one.

Second wave — SHIPPED:

- **time-cost axes**: every probe is timed (best-of-two under the
  scoped deadline) and a family is certified classical by an axis only
  when BOTH its memory and its wall-clock law stay sub-exponential.
  The measured case for the rule arrived immediately: MPS on
  long-range IQP is time-polynomial (`size^4.5`) but
  memory-exponential (`2.14^n`). Time laws use a wider polynomial
  band (base < 1.35) so microsecond jitter doesn't misread flat axes.
- **assumption dials** (`library::iqp`, `library::doped_clifford`,
  `library::brickwork_2d`): the same IQP core flips
  candidate ↔ classical on the interaction-range knob; random
  T-doping costs the frame NOTHING even at t = n/2 (measured — T's in
  a random Clifford stream land where conjugation keeps them diagonal;
  the 2^t escape needs deliberately scattered magic, so the frame's
  boundary is about WHERE the magic sits, not how much); depth-3 2D
  brickwork reads `size^2.0` — the boundary (2^√n) law failing slowly
  at probe sizes, exactly as an almost-holding assumption should.
- **selection by extrapolated scaling** (`select_by_scaling` /
  `LawFit::predict`): fit each axis's measured law at probe sizes,
  rank predictions at the target, holdout-verified (GHZ at width 40:
  predicted 125 B from fits that never saw it, measured 125 B), and
  explicit when no assumption holds (the candidate family's winner is
  flagged least-bad, not good).

Third wave — SHIPPED:

- **sampling-task hardness** (`sampling`): linear XEB against the
  dense or exact D[ω] reference (agreement 1e-15 on shared samples),
  normalized by the measured ceiling `2^n Σp² − 1` (uniform outputs
  have no signal and score `None`, never a fake fidelity). Verdicts
  carry to the task: GHZ sampled at the ceiling from 125 B,
  `clifford_sample` doing per-shot tableau measurement (width 20, 256
  shots, ~23 ms). Spoofing economics measured: truncated-MPS caps
  collapse below 0.3 normalized with only full rank scoring — a cliff,
  not a slope — and a fixed-χ budget decays to noise as the family
  grows. The sampling task inherits the state bounds, reproducibly.
- **variance-aware time laws**: three timing repetitions per probe,
  median-classified with min/max envelope laws
  (`AxisScan::time_law_bounds`, `time_law_is_variance_robust`) — a
  time classification whose envelopes disagree is jitter-limited and
  says so.
- **register shapes as scan axes**: the hierarchical splits
  (`algebraic-h`, `algebraic-o` over sparse sites) join every profile,
  scan and selection — constant on GHZ, escaping with everything else
  on the candidate family, and measurably beating plain sparse on
  mixed-sector states (axis against axis).

Next rungs:

- **spoof-cost frontier**: for each family size, the *cheapest* χ that
  reaches a target normalized XEB — the measured classical cost of the
  task at fixed fidelity, the quantity advantage experiments actually
  argue about.
- **noisy-sampler models**: depolarizing/readout noise channels on the
  sampler side, so measured XEB decay can be compared against the
  noise budget the way hardware claims are.
- **per-shot cost laws**: `clifford_sample` and `sample_mps` costs
  classified by the same law machinery as state costs (per-shot time
  vs width per family).

## Mixed-arity compound qudits — SHIPPED (core)

`mixed::CompoundRegister` + `e8`: the representation/interaction/flow
separation for non-binary registers. Sites of any arity held as
horizontal volumes (independent until interaction, guard-admitted
merges), generalized gates with tested relations, conformance to the
qubit reference where dims coincide, the fabric's measured swap-class
decomposition (swap exists only between equal arities — cross-arity
bonds are forced native), and order-respecting interaction cones. The
E8 anchor is constructed and verified programmatically: 240 roots, the
su(2)…su(5) arity chains found by search, `su(5)×su(5)` exhibited
orthogonal, and the rank obstruction MEASURED (after A1⊥A2⊥A3 the
orthogonal-A4 search exhausts; 1+2+3+4 = 10 > 8) — so the four arity
frames must share directions, and the canonical embedding's measured
Gram overlap (quaternary–quintary, 44) becomes the compound qudit's
coupling fabric.

Second wave — SHIPPED: **both E8×E8 systems in the standard
frameworks**. `CompoundRegister` grew a general k-site `apply_k`
(mixed-radix, one logged interaction per call; `apply_1`/`apply_2` are
now thin delegations) and exact `project_digit` collapse; the
all-binary case is `CompoundBackend` (`"compound-binary"`), a full
`Backend<C64>` with product-support enumeration over horizontal
volumes and the 63/64-qubit packed-index wall measured. The E8×E8
representation is `e8::rep::E8RepState` (`"e8-rep"`): amplitudes keyed
by (copy, spinor root), 8-qubit-native with a structural refusal past
8, GHZ downcast-verified to be stored as an antipodal root pair. Both
pass `verify_backend` over the full registry (e8-rep also at native
width 8 across the whole 256-point set) and run through
`compare_backends` beside dense/sparse/adaptive/factored/mps/mera,
with the d = 16 co-boundary protocol as a workload: every completed
run amplitude-verified against dense, MPS refusing the 8-qubit pairing
gate at its measured window wall, and the native two-site 16-level
protocol equal amplitude-by-amplitude to its qubit encoding.

Third wave — SHIPPED: **the infinite E8 constellation**
(`e8::constellation`). The measured coset theorem E8/2E8 ≅ F₂⁸ (the
origin + 240 roots + 2160 norm-2 vectors bucket into exactly 256
classes: 1 + 120 antipodal pairs + 135 sixteen-frames, on shells of
radius 0/√2/2 — verified by exact integer arithmetic over a
triangular doubled basis with |det| = 2⁸ and its verified adjugate)
makes one byte the identity position of an E8 on its parent's
spheres. `compose`/`decompose` realize the scale tower
Σ 2ᵏ·rep(digitₖ) as a measured bijection onto E8/2^m E8, self-similar
under doubling; `E8ConstellationState` (`"e8-constellation"`) keys
amplitudes by lattice points at any width to the u64 wall, passes
full-registry conformance at default/widened/multi-block widths, and
is priced in the harness beside the standard backends (GHZ-63 = two
80-byte points where dense refuses to construct; saturated QFT-12
honestly costs more than dense).

Fourth wave — SHIPPED: **multi-scale dual time**
(`e8::constellation` scale methods + `tests/e8_dual_scale.rs`). The
distilled scale operad: `scale_embed`/`decimate` isometries with exact
composition (`V_a∘V_b = V_{a+b}`, `R∘V = id`), covariant Weyl
transport (`T_{2v}∘V = V∘T_v`, `M_q∘V = V∘M_q`), and the honest
divisibility subtlety pinned (componentwise evenness is NOT lattice
divisibility — decimation demands digit-zero levels and refuses live
fine data with the level named). Cross-scale comb codes: coarse
translation checks + fine modulation checks, all commuting past the
horizon; logical operators at the middle scales; the m×m syndrome
matrix tiling EXACTLY the Weyl commutation inequality (detection
window = bidirectional horizon — one measured matrix unifies QEC and
the Heisenberg ladder); end-to-end correction with exact binary
phase-readout decoding and honest past-window blindness; code
self-similarity (decimated (m=4,a=2) code = (m=3,a=1) code, state
identity). Folding theorems: the interleaved ascending-T/descending-M
sequence reorders through beyond-horizon commutations only into one
two-gate layer (below-horizon control measurably refuses); deep
periodic time folds to the measured period (P = 8 at m = 3) with the
exact quadratic Weyl phase `χ^{t(t−1)/2}` pinned at small t — 2⁴⁰+5
blocks evaluated as 5 in ~100 µs. Noise trajectories — SHIPPED
(`CombCode` + `tests/e8_comb_noise.rs`): the codes as first-class
objects with min-norm decoding, window-sized displacements measured as
the code distance (silent logical operations), the mod-2 tie's
fail-half pinned, and seeded displacement-noise trajectories yielding
the logical-vs-physical error curves (degenerate window saturating,
a=4 at 0% where a=3 fails 61% at p=0.05, threshold-shaped suppression
across five rates). Remaining rungs: syndrome extraction as physical
ancilla interferometry (the eigenphase read is simulator-direct), and
the mod-2^m tableau that would make all of it polynomial.

Next rungs:

- **auto-split on disentanglement**: volumes currently merge and stay
  merged; detecting product structure (Schmidt-1 across a site) would
  restore horizontality after uncomputation, like factored's split.
- **mixed-arity circuits/registry**: the conformance/harness machinery
  now sweeps the compound register through its all-binary backend;
  still open is the genuinely mixed-arity `Circuit`-level description
  (named qudit gates over non-binary sites with bind-time validation)
  so mixed-dim registers get registry-drawn random sweeps too.
- **richer E8 embeddings**: search for minimal-total-overlap
  placements of all four chains (the canonical one is greedy), and
  weight the compound fabric by the Gram magnitudes rather than a
  boolean coupling.
- **arity-mixed device model**: `Topology`/latency over mixed sites
  with the swap-class constraint enforced by the router (equal-dim
  corridors, native cross-arity bonds).
- **constellation-native gates — SHIPPED as the cross-scale Weyl
  pair**: `translate`/`modulate`/`reflect`/`coordinate_fourier` act
  directly on lattice keys (no bit-domain conversion), with measured
  self-duality (det Gram = 1, dual basis in-lattice, coordinates =
  dual inner products), the Heisenberg law and its 2-adic scale ladder
  (interaction below the resolution horizon, exact commutation past
  it), W(E8) reflections as measured Cliffords, `F⁴ = 1` with
  `F T_B F⁻¹ = M_{−b*}`, exact support uncertainty
  (`|pos|·|mom| = 2^{8m}` on coset states), depth-1 reduction to
  X-strings/sign diagonals against the standard framework, and
  structured interaction at 40 qubits under 32-point support where
  dense measurably refuses. Still open on top: a generator-based
  stabilizer TABLEAU over ℤ/2^m (the coset family is closed under the
  native set at 2^{km} points — an 8th-root-of-dense compression;
  tableaux would make it polynomial), full W(E8) generator sets beyond
  single reflections, and deeper digit alphabets from the next shells
  (norm-6/8 vectors give E8/3E8 and beyond) for non-binary
  constellation levels that would meet the mixed-arity register.
- **edge-level co-boundary storage**: the E8×E8 system stores rays at
  paired POINTS (ℂP²³⁹, measured invisible/recoverable); the measured
  b₁ = 4241 says the complex carries that much invariant EDGE data
  beyond points — encode qudit fields on the 6720 edges modulo the
  2240 triangle relations and build the readout interferometry for
  the cocycle classes.
- **Weyl-symmetric storage bases**: decompose stored fields over the
  root graph's spectrum (the −1 adjacency is highly symmetric) so the
  co-boundary payload is addressed by symmetry sector rather than raw
  point index.

## Recursive systems — SHIPPED (core)

`recursive` and `e8::cube` are live: a `Site` is a point *or* a
`RecursiveLattice` of the same kind, `refine`/`nest` grow the structure,
and one scale-free rule produces both the intra-block bonds and the
lateral bonds joining whole sub-lattices corner to corner. On top of the
structure sit the measurements that decide whether the substitution
means anything — `block_rg`, `rg_flow`, `rg_fixed_point`,
`substitution_report`, `phonon_block`, `phonon_substitution`,
`phonon_walk`, `self_participation`, `participation_transition` — and
`e8::cube` reads an E8 point as a 2×2×2 volume whose scale tower is a
cube of cubes, with `inward`/`outward`/`interaction` measuring how far a
volume participates in its own interior. Remaining work:

- **The isometry as a representation, not just a report.** `block_rg`
  measures the two-state isometry and throws it away. Keeping it — a
  `RecursiveState` backend whose stored object is a tower of block
  isometries with the residual weight tracked per level — would make
  the substitution a *storage* strategy rather than an analysis, in the
  same relationship to `MeraState` that `FactoredState` has to dense.
  The measured discarded weight is already the natural error dial.
- **Variational isometries.** The two lowest eigenvectors are the
  simplest possible choice of block basis and demonstrably not the best
  one (the in-band deviation is a few percent). Optimizing the isometry
  against the *bonded* environment rather than the isolated block —
  one sweep of a DMRG-style environment update — should shrink it, and
  the existing `substitution_report` is the ready-made scorecard.
- **Disentanglers between blocks.** The lateral bonds are exactly where
  the block-spin truncation loses the most, and exactly what a MERA
  disentangler layer is for. This is the same rung the mera backend is
  waiting on; doing it once should serve both.
- **The phonon lattice as a real bosonic register.** `phonon_walk`
  works in the single-excitation sector, where the truncated boson
  lattice *is* its hopping matrix. Lifting it to `CompoundRegister`
  with `d`-level sites would make the substitution claim at finite
  phonon number, where the collective coordinate stops being exactly
  protected and the deviation becomes a function of occupation — a
  genuinely different measurement, not a wider version of this one.
- **Self-participation beyond mean field.** The block currently sees
  its neighbours as a field. Letting it see them as a *state* — the
  environment being another instance of the same computation, with the
  two exchanging boundary density matrices — is the honest version of
  "a computation that participates in itself", and the convergence
  trajectory would be measurable the same way.
- **Inward/outward as a gate set.** The cube's ladder is measured but
  passive. Registering `inward`/`outward` displacements as named gates
  over `E8ConstellationState`, with the horizon as a validity
  condition, would let a circuit *use* the finite self-reference depth
  — a computation whose available interactions are a function of which
  scale it is addressing.
- **Deeper cube towers.** `self_reference_depth` grows one level per
  8 qubits, so measuring it past m = 7 needs the 63-qubit wall lifted
  (u128 keys, or the tower held symbolically). The law is linear and
  boring; what is not is whether the *phase ladder* stays exactly
  2-adic that deep.

## Operational-model extensions

- **Device realism — geometry + latency maps SHIPPED**: `Topology` now
  reproduces real register geometries (`heavy_hex_falcon27` — the IBM
  Falcon-r4 lattice, 27 sites / 28 couplers / degree ≤ 3;
  `sycamore_like` diagonal lattices; `complete` trapped-ion all-to-all)
  and `LatencyMap` carries their operation-latency maps: era-preset
  `DurationModel`s (`ibm_falcon_like` / `sycamore_like` /
  `ion_trap_like`, ns ticks) plus per-site and per-edge calibration
  overrides. `DeviceState::with_latency` injects the inner
  representation (chip-scale geometry over a sparse inner), and
  `serial_time`/`elapsed` measures how much parallelism the geometry
  admitted (`tests/device_geometries.rs`,
  `examples/device_reproduction.rs`). The model stays exact and
  noiseless; next rungs, in order of leverage:
  - **latency-aware routing** — routing is still latency-blind BFS by
    edge count; the ring demonstration in `device_reproduction`
    measures the cost (an equal-hop detour around a 10×-slow coupler
    goes unused). A Dijkstra walk over `LatencyMap` swap costs, then
    lookahead / SABRE-style, measured against the BFS baseline by
    `swap_count`/`elapsed`.
  - calibration data import (per-coupler CSV/JSON → `LatencyMap`).
  - per-edge gate fidelities and idle decoherence (needs the
    density/trajectory machinery below).
- **Causal register geometry + dual time — SHIPPED** (`causal`,
  `Topology::hierarchical` / `Topology::hypercube`,
  `tests/causal_geometry.rs`, `examples/causal_geometry.rs`): registers
  whose coupling fabric is organized by causal scale, the causal-metric
  probes (`distances_from` / `diameter` / `mean_distance` /
  `ball_sizes` / `pair_availability`), backward cones and causal
  diamonds (observation-identical pruning), and dual-time resolution
  (`dual_time_amplitude`: forward from preparation, backward from
  observation, resolving at a cut; `2^{D/2}` a side at the balanced cut
  vs `2^D` one-way, measured). Findings worth carrying forward:
  - the *homogeneous* causal fabric (hypercube) improves both routing
    volume and wall-clock on the QFT; the *hub-concentrated* hierarchy
    improves routing volume but pays it back in hub serialization —
    availability and parallelism are separate axes, and only the
    measured schedule tells you the net;
  - the dual-time meeting surface can be far smaller than either
    direction's support (interface 1 on Clifford staircases) —
    suggesting cut *selection* (min-interface, not just balanced) as a
    cheap optimization;
  - next rungs: latency-weighted causal metrics (Dijkstra distances so
    availability reflects heterogeneous calibration), diamond-restricted
    device scheduling (only schedule the cone of the declared
    observables), congestion-aware fabric variants (hierarchies with
    replicated hubs), and multi-surface resolution (more than two
    opposed directions over a cut tree).
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

## Resource governance

- **Resource guard — SHIPPED** (`guard`): memory admission against
  *measured* capacity (cgroup/`MemAvailable` at allocation time,
  fallible reservation as backstop) for every large allocation in every
  backend — the mixed-arity compound register (merge AND per-gate
  growth) and the E8 constellation's lattice-key growth included, so
  the new representations refuse over-scale work with the same measured
  numbers as sparse; wall-clock budgets checkpointed inside the long
  kernels; capacity-aware adaptive promotion; the former width-constant
  caps demoted to structural index bounds. Verified against reality by
  `examples/capacity_probe.rs` (subprocess-isolated walks to real OOM
  kills and deadline aborts, now across all twelve axes: the
  algebraic/compound/constellation fills and the structural ceilings —
  compound/constellation at 63, e8-rep at its native 8 — beside the
  original dense/exact/sparse/factored/mera walls). The probe's raw
  mode is honest again: `Some(usize::MAX)` used to collide with the
  auto-measure sentinel (so "raw" rows were admission refusals in
  disguise); the limit is now flag+value, admission-disabled requests
  reach the allocator and fail there with the at-failure measured
  availability, distinctly labeled — pinned by a regression test. Next rungs: cooperative *degradation*
  instead of abort (a backend that receives OutOfMemory could spill —
  factored → mps handoff), per-scope (non-global) budgets once a
  session/context type exists, allocation accounting of the crate's own
  live states (admission currently measures the process from outside),
  and reserving the transient double-buffer in sparse/mera rebuilds
  ahead of time so mid-operation refusal can roll back instead of
  tearing.

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
