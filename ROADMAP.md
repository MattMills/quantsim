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
  **log compaction** (the storage half is done: kernels are interned —
  each distinct gate matrix stored once and shared across the log —
  which corrected the frame's log-diluted memory fit from 1.75^n to an
  honest 1.85^n on random 3n² and flipped Clifford brickwork's memory
  law from size^2 to constant, 19.3 KiB → 4.8 KiB. The replay half —
  tableau → minimal Clifford circuit synthesis, replacing replay of the
  full log — remains, and measurement repairs prepending to the log
  keep raising its value for long adaptive runs; note synthesis
  reproduces the tableau only up to global phase, which replay
  currently preserves exactly),
  **packed stored state** (the other constant: at saturated support the
  sparse map's per-entry overhead runs ~6× a flat array — the frame's
  remaining byte gap to dense on random circuits, 412 KiB vs 64 KiB at
  n = 12, lives here, not in the log any more),
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
so representation resolution and numeric resolution stack.

**Rung 2 — SHIPPED** (`BulkState`, registered `"bulk"`,
`tests/bulk_register.rs`, `examples/dynamic_register.rs`): the
hierarchy in its **projective form** — every node tensor kept
isometric, all state weight in one explicit bulk record (`top`), the
state stored across depth as a stack of isometries under the record.
What the form buys, measured:

- **the gauge-maintenance rung, delivered** — not by center transport
  but by *environment-weighted rebuilds*: the record's Gram factor
  (`X` with `X†X = ρ`) is transported down the path before every
  re-compression, so every discarded singular value is **global**
  Schmidt weight; the ledger resolves **per depth**
  (`discarded_by_depth`) and certifies
  `‖ψ_ideal − ψ_stored‖ ≤ Σ√ε` (asserted against dense across seeds;
  *equality* on single-event runs; a pinned seed where the weighted
  rebuild beats the block-local one by > 4× — and no dominance claim:
  greedy sequential truncations do not commute, seeds exist where the
  orderings trade places, the certified bound is the invariant).
  `‖ψ‖² = ‖top‖²` identically — norm at width 40 is an `O(χ)` read.
- **dynamic width** — `grow` appends boundary qubits in amortized
  `O(1)` (within capacity: bookkeeping over a pristine dormant suffix;
  past it: the register **re-roots**, becoming the left *site* of a
  register twice its size — `recursive`'s point-is-a-lattice move
  applied to the representation; a live GHZ grown 2 → 32 with exactly
  4 re-roots, amplitudes exact throughout). `release` detaches
  boundary qubits, refusing with the **measured leakage** unless the
  qubit is verifiably `|0⟩`; `release_measured` measures first. The
  flagship: 48 logical qubits streamed through a register with peak
  width 2, outcomes GHZ-correlated — width tracks *live entanglement*,
  not problem size. The price, also measured: the capacity tree is
  dyadic, so hierarchy-locality means power-of-two alignment (a
  5-block circuit that was tree-aligned on `mera`'s div-ceil tree
  crosses dyadic boundaries here, and the guard refuses the block).
- **the structured unfold** — `unfold_program` compiles the depth
  store into a seed plus one Stinespring-dilated unitary per node;
  replayed on dense it reproduces the state to `3e-17`; stopped after
  `ℓ` levels it **is** `coarse_state(ℓ)` on the channel wires with
  every un-injected wire exactly `|0⟩` (asserted per depth); each
  wire's `sequence` is the complete list of interactions it ever has
  (nested spans down its tree path), and measured Schmidt rank across
  any tree-aligned cut never exceeds the crossing channel's bond —
  entanglement relocated into nameable, sequenced channel wires, the
  up-a-dimension move `lift`/`upembed` make for magic, made for
  entanglement.

The remaining rungs, in dependency order:

- **Path updates instead of block materialization** — apply a cross-cut
  2q gate as its operator-Schmidt sum (rank ≤ 4) of single-site terms,
  then hierarchical rounding along the tree path (bond direct sums +
  SVD truncation): removes the `2^{block}` transient, making GHZ-across-
  the-root bond-2 *during* the gate, not just after. (On `bulk` the
  environment factors for the path are already in hand.)
- **Gauge maintenance on `mera` itself** — the rung is delivered on
  `bulk`; back-porting either the weighted rebuild or true center
  transport to `MeraState` (or retiring the distinction) remains, as
  does a non-greedy truncation order (the pinned seed where orderings
  trade places is the test case).
- **The atlas axis for `bulk` needs octave-aligned sampling.** A probe
  was measured and backed out: the dyadic capacity tree makes memory a
  **sawtooth** in width — skewed cuts at non-power-of-two sizes cap
  rank by the short side (random family: ~2× under `mera` at
  n = 10, 12; equal within 1% at n = 4, 8) — and a four-point fit
  across an octave misreads the sawtooth as `size^4.7` while the true
  per-step factor is 1.7×, leaving certification to the
  contention-fragile time law (the verdict flipped under parallel test
  execution; solo it held). Either the scan samples the axis at
  power-of-two sizes or the fit learns sawtooth envelopes; both keep
  the finding: bulk at pow2 widths **is** mera's cost, and the skew
  discount between octaves is real structure, not error.
- **The post-reroot crossing ceiling, measured.** Growth is O(1)
  (4 µs for grow #31 at width 32), but the *first entangling gate*
  across a freshly created root cut materializes the live block —
  2^33 elements refused by the guard at width 33 — so streaming
  growth patterns must keep entanglement inside the pre-reroot span,
  and the path-updates rung above is what removes the ceiling.
- **Interior release / renumbering** — `release` is boundary-only
  (stack discipline); releasing an interior qubit means renumbering
  the leaf map, and the honest cost of the re-alignment should be
  measured, not assumed.
- **Streaming unfold on the dynamic register** — the unfold currently
  replays onto a fixed-width backend; running it *on a `BulkState`
  that grows as wires are injected* would make the width trajectory
  literal (peak width = channel count at the widest level), and the
  dilated steps are capped at `2^6` wires — factorized dilations
  (cascades of two-wire isometries) would lift the cap.
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

## The scale-time register (clock program) — SHIPPED (core)

`clock` is live: internal, relational time as a **selector qudit**
whose entanglement with the system is mandatory, with scale as its
axis. `BranchedRegister` (selector over shared,
representation-heterogeneous branches; lazy interference; polynomial
Gram machinery for Born statistics, conditioning and the
selector↔system Schmidt rank), `scale_history` (the Page–Wootters
register over the bulk depth axis, every slice an `O(tree)`
`scale_snapshot` — no gate applied, width-32 six-scale histories in
kilobytes), `tick` (`|ℓ⟩⟨ℓ|⊗V_ℓ` — one unit of internal time is one
level of coarse-to-fine information flow), and scale interferometry
(inter-scale overlap as clock Born statistics; the structured test
state moves by exactly `1/√2` per RG step). The measured payoff: four
width-16 slices, each cheap in its own representation and hostile to
the others', held additively at 3,845 B against 68–850× for every
single-representation alternative — new qudit dimension above the
problem, lower representational structure below. Remaining rungs:

- **The native bulk tick.** `tick` applies the level's dilated
  unitaries through `Backend::apply`, which on a `bulk` branch
  materializes the root-crossing block. The refinement isometries are
  tree-structured by construction, so a bulk-native controlled
  refinement (splice one level of the stored program directly into the
  snapshot's tree, the way `scale_snapshot` already splices basis
  embeddings) would make ticking as free as snapshotting — internal
  time at width 32+ with no dense transient anywhere.
- **Branching histories.** The clock is currently a line
  (`ℓ → ℓ+1`). A *tree* of ticks — different refinement or gate
  futures in superposition, selector states labelling paths — would
  let the register hold alternative coarse-to-fine routes coherently,
  with the same Gram machinery pricing their interference. The memo
  journal's unwind/rewind is the classical shadow of this.
- **Phase pinning for phase-loose branches.** The measured boundary:
  superposition makes branch global phases relative, so the
  graph-state bundle (defined up to global phase; vertex-operator
  reductions rotate it — classified in development at exactly
  `e^{−iπ/4}` on the pinned case) is sound as a static slice but not
  under broadcast dynamics. The vop reduction *knows* the phase it
  drops; journaling it would upgrade the bundle to phase-faithful and
  open the door to graph-state slices under full dynamics.
- **Selector structure as an atlas axis.** Branch count and unique
  states are honest resources; registering a policy-constructed
  contracted view would let `advantage_scan` classify selector growth
  laws next to support, clusters, bonds and hierarchy.
- **Selector gates through the scheduler.** Clock operations are
  method calls today; routing them through the registry/schedule (the
  `mixed` module's `clock_d`/`shift_d` are the natural names) would
  let evented circuits steer the internal time system, including
  measurement-conditioned ticks — dynamics that choose their own
  resolution.
- **Exotic clock weights.** Selector weights already live in the
  amplitude algebra; a split-complex clock (signed slice weights, net
  vs path statistics) would connect the time system to the
  inclusion–exclusion program.

## The mosaic register (multi-representation program) — SHIPPED (core)

`MosaicState` (registered `"mosaic"`) is live: the register
partitioned into regions, each held by its own backend — the factored
backend's geometry with factors generalized from dense blocks to
arbitrary representations. Gates inside a region run natively; gates
across regions merge with the representation chosen from measured
predictions (sparse support-product vs dense `2^w`, both ledgered);
migration is refusal-driven and **partial before total** — a bundle
region hit by its first T first fractures along its own graph
components (`PolarityBundle::graph_components` + `restrict`, exact by
the product structure of graph states), so only the touched component
leaves and the rest stay graphs — then converts down the policy's
candidate list and retries, once, with every split, merge and
migration ledgered with its cause. Regions can themselves be mosaics
(tested), the flagship holds a width-40 register at the sum of ideal
costs through a Clifford→T→entangling era sequence (dense-verified at
16), and the phase contract is the measured dual of the clock's:
products forgive phase-loose tiles, superpositions do not. Remaining
rungs:

- **Cross-representation bonds.** The real frontier: two regions
  *entangled* while each keeps its own lens. The braided module's
  mutual (Schmidt/purification) encoding across a cut is the natural
  machinery — a bond as a shared index between a bundle tile and an
  mps tile would remove the merge-on-entangle limitation that makes
  rung 1 a product mosaic.
- **Generic partial structure.** The bundle fractures because its
  graph is readable; other representations should carry a coupling
  history (union–find over the entangling gates the mosaic routed into
  them) so any region can split along provable product boundaries, and
  factored-style rank-1 detection should re-separate merged regions
  when gates disentangle them.
- **Policy from recognition, not just refusal.** Today the bundle's
  refusal is the era detector. Recognizing Clifford gates on the way
  in (the frame's `recognize` machinery) would let regions *return* to
  cheap representations when a magic era ends, and bond/hierarchy
  predictors would let merges target mps/bulk instead of only
  sparse/dense.
- **The mosaic as an atlas axis — SHIPPED, and it taught the atlas
  something.** The first probe flipped the candidate verdict to
  `Classical via ["mosaic"]`: not an advantage discovery but
  **prefactor aliasing** — the policy elected sparse at one size and
  dense at its neighbours, and a four-point fit read the jumping
  constant (≈50 B/entry vs 16 B/amplitude on the same 2^n states) as
  `size^4`. The atlas assumes each axis has one stable cost model; a
  policy-composite axis violates that unless its elections track the
  state. The fix was owed anyway: **memory-pressure re-election**
  (`MosaicPolicy::pressure_factor`/`pressure_floor` — the adaptive
  backend's promotion rule generalized, hysteresis + floor, every
  re-election ledgered with measured-vs-predicted bytes). With it the
  axis reads `1.89^size` on the candidate family, the escape verdict
  stands, and the scan now shows the composite certifying exactly
  where its lenses reach: constant on ghz (349 B) and rainbow
  (896 B), `size^3.3` on nearest-neighbour IQP, `size^5.1` on
  t = n/2 doped Clifford — and honestly exponential on qft, brickwork,
  random universal and long-range IQP. Still open here: elections
  beyond the sparse/dense pair (bond/hierarchy predictors), and
  recognition-driven *return* to bundle when a magic era ends.
- **Mosaic × clock — SHIPPED (default).** The branched register's
  default branch is now a mosaic: representation election below the
  selector, qudit dimension above it. Measured in the atlas, the
  `branched` axis moved from sparse-tracking (three verdict
  memberships) to mosaic-tracking plus ~80 B of selector bookkeeping
  (nine: ghz 437 B, qft 872 B constant where the sparse branch paid
  200 KiB exponential, rainbow, qft|x⟩, nearest-neighbour IQP,
  t = n/2 doped, shallow-2D, aqft, linear-budget random).
  scale_time §5 measures the election against the hand-picked
  four-slice register and reports the boundary honestly: election
  matches assignment where the slice's structure is support or
  product (GHZ 373 B, rainbow 1016 B — the region partition *is* the
  factored lens), and pays 2^region where the ideal lens is bond
  (brickwork: 1 MiB vs mps's 1.5 KiB) or stabilizer (graph state:
  525 KiB vs bundle's 1 KiB) — those lenses are not yet in the
  mosaic's within-region vocabulary, which makes the "elections
  beyond the sparse/dense pair" rung above the composition's
  measured bottleneck, not a nice-to-have.

## Retrocorrection (surface code over the record) — SHIPPED (core)

`retro` inverts error correction with the simulator's own powers: the
rotated surface code as a constraint set ([`SurfaceCode`], any odd
distance, patch offsets so several share a register), syndromes as
**deterministic reads** (`±1` generator expectations — no ancillas, no
randomness, nothing disturbed), a decoder whose lookup table is
**measured from the code's own generators** and refuses beyond it, and
the retrocorrection theorem made operational: a correction decoded once
at the end of the record, conjugated backward through the intervening
Clifford segments (`transport_back`), repairs every stored slice of a
branched clock register via slice-addressed surgery
(`BranchedRegister::apply_at`, which refuses through shared walls).
Measured end to end (tests/retro.rs, examples/retrocorrection.rs): a Y
fault after slice 1 of a two-patch record spreads through the
transversal logical CX onto both patches (syndromes 2 + 4), the
generators transported to the fault's time predict the end-of-record
reading exactly, one decode yields weight-1 corrections per patch, and
every slice returns to the clean history at machine epsilon — with the
logical entanglement graph's sign history (+1/+1 → −1/−1 across the
logical Pauli era) intact. 77 KB for the 3-slice 18-qubit record
against 4.2 MB per dense slice.

The toric extension — SHIPPED — makes it a **logical network**: the
`ToricCode` carries two logical wires per physical set (the torus's
non-contractible cycle pairs, X̄ on dual cuts and Z̄ on direct cycles),
one transversal CX between nodes raises two logical Bell links at
once, and the links live in the selector: the Schmidt-branched form
(four mosaic branches, each a *product* of per-node code states, the
node cut never crossed) equals the flat state to 1.4e-17 with selector
rank 4 = 2^links at 4.7 KB against 12.8 KB flat sparse and 1 MB dense.
Sequential node-local eras keep faults node-local — the end-of-record
syndromes fire on the faulted node only, its own decoder names the
correction, transport through the other node's era is the identity —
and the four-slice record restores to exactly 0e0. Two honest clauses
came out of the build, both now instrumented: the decoder distinguishes
**degeneracy from logical ambiguity** by asking the stabilizer group
(at L = 2 every weight-1 X/Z syndrome is ambiguous and refused by name
— distance 2 detects, never corrects — while every weight-1 Y decodes),
and syndromes are **sign-blind** (dynamics anticommuting with a fault
flip its residual to −P, physical in a branched record) — resolved by
the record itself: each slice must equal its predecessor pushed through
the segment, the last clean slice anchors the chain, one amplitude
comparison names the sign.

Constraint-driven compression — SHIPPED (phase 1). The `logical`
backend (src/logical.rs) holds the register as P·Enc·|l⟩: toric nodes
tiled over the width, encoders recognized op by op, every physical
Pauli absorbed into the frame (a fault is one bookkeeping update, its
syndrome the frame's signature — no state work), transversal CX blocks
committed as logical CXs on a 2-wires-per-node inner register, reads
answered exactly by F2 coset membership, anything else escaping to an
exact materialization. Measured in the atlas on the new encoded-toric
family over widths 8–32: logical mem size^0.6 at 1.7 KiB where sparse
pays 1.29^size (200 KiB), dense/factored/mps/mera/phase-field wall
outright, and the family reads CLASSICAL via logical / braided /
clifford-framed / bundle — the family is Clifford, so the
Gottesman–Knill axes also hold it (at 10–100× the bytes); the logical
axis's distinct content is the constant and the constraint structure.
One convention scar worth remembering: the crate's PauliString is
Hermitian (i^{|x∧z|}X^xZ^z) while the frame is raw, and conjugation
changes the Y-overlap — the raw phase picks up i^{Δy} beside the sign,
measured as a per-codeword sign error before the fix.

Remaining rungs: weight->1 decoding (matching over the syndrome
graph), measurement-conditioned records (retrocorrection through frame
repairs), logical vocabulary growth (logical magic escapes today; the
frame-vs-logical split should let a t on the inner wires cost 2^t at
the LOGICAL width), a non-sparse inner for the logical layer (a graph
bundle holding the LINK GRAPH rather than its Schmidt expansion), and
the non-Clifford boundary (transport through magic via the
up-embedding's ancilla frame rather than refusal).

## Cross-lateral distributed registers — SHIPPED (core)

`lateral` takes the logical network apart and puts the nodes on
different machines. It rests on one observation with three parts, each
measured rather than asserted:

**1. The inter-node map is coordinatewise.** Couple nodes only
transversally — physical qubit `i` of node A to qubit `i` of node B,
which for a CSS code is the logical CX — and the Pauli frame
conjugates by `x_B ^= x_A`, `z_A ^= z_B`, and a sign popcount
`σ ^= |x_A & z_B & !(x_B ^ z_A)|`. Two XORs of *aligned* bit-vectors
with no term coupling qubit `i` of one node to qubit `j ≠ i` of the
other. `LateralLink::push_frame` is that rule and
`push_frame_by_gates` is the oracle it is pinned against: **zero
disagreements over 200,000 random strings**, signs included, across
four window layouts. A `Y` at site 3 lands on `[3, 11]` and nowhere
else — "lateral" is the claim that a link is a bundle of independent
wires at one level, not a mixing matrix.

**2. The syndrome is 𝔽₂-linear in the frame.** `SyndromeMap::bits`
satisfies `bits(p ⊕ q) = bits(p) ⊕ bits(q)` exactly, for every pair,
commuting or not — the symplectic form is bilinear and the syndrome
cannot see the phase that non-commutation would produce (50,000 random
pairs on an L = 3 torus). So a node holding the total syndrome — an
ancilla-free deterministic read, `retro::syndromes`' whole point — and
the deltas that arrived can name the syndrome of the delta that did
**not**, by one XOR.

**3. So the wire carries nothing but 𝔽₂.** 27 bytes per node per tick
(two masks, a sign, a node and a tick), independent of `2^n`. Measured
end to end on two L = 2 toric nodes: the coordinatewise frame after a
`Y`-flavoured fault on A, a `Z` on B and one transversal CX **equals**
the gate-by-gate frame, and its signature equals the syndromes a dense
16-qubit register actually shows, on both nodes, bit for bit — 54 B
across the wire against 1,048,576 B of amplitude, and the ratio grows
as `2^n`.

**What that makes a dropped packet.** A Pauli fault and a lost datagram
become the same object: an unknown vector in 𝔽₂^{2n} recovered from
`H·e = s`. The difference is that the network *knows where its loss
happened* — the schedule names the tick's support — and the physics
does not. Known locations turn error decoding into erasure decoding,
worth exactly a factor of two, and `ErasureDecoder` measures it off the
code's own generators rather than reading it off a distance:

| code | `d` | errors `⌊(d−1)/2⌋` | erasures (measured) | witness |
|---|---|---|---|---|
| toric L=2 | 2 | 0 | **1** | `[0, 2]` |
| toric L=3 | 3 | 1 | **2** | `[0, 2, 4]` |
| surface d=3 | 3 | 1 | **2** | `[0, 1, 2]` |
| surface d=5 | 5 | 2 | **4** | `[0, 1, 2, 3, 4]` |

`certified_capacity` enumerates erasure sets until one carries a
logical and `first_uncorrectable` produces that logical as a witness —
in every case a weight-`d` operator, which is why the capacity is
`d − 1`. L = 2 is the sharp case: `retro::Decoder` refuses a weight-1
`X` by name (distance 2 detects and never corrects) and
`ErasureDecoder` recovers it exactly from the same syndrome. And the
refusals stay distinguishable — a set carrying a logical and a syndrome
outside the set's image are different errors, never a guess either way.

**Two honest clauses came out of the build.** *Correctable* and
*unique* are different questions: on the rotated surface code the
boundary pair `{0, 5}` carries a weight-2 stabilizer, so `Z₀` and `Z₅`
share a syndrome — correctable (they act identically on every logical
observable) and not unique (they are different bytes). A
stabilizer-equivalent representative is as good as the truth for the
code space and *poison* for a downstream linear code doing algebra on
the bytes, so the fine layer fills only where the recovery is unique on
the nose and counts the rest as declined. The second clause is
`retro`'s, one level up: a syndrome is **sign-blind**. That costs
nothing here because every sign in the system is generated *locally* by
the link rule from windows both endpoints already hold — what crosses
the wire is a node's own physical fault, a Hermitian Pauli with no
sign.

**The horizon, and the two layers.** `DelayGeometry` measures what the
network actually is: nodes laid out in delay, not space, with a
"distance" that routinely violates the triangle inequality (6 ordered
pairs out of 12 on the example's four-node fabric, direct `d(0,3) = 11`
against a best relay of 8). `tighten` makes it a metric and its
temporal diameter is the horizon floor no protocol beats. `Barrier`
then seals on **actual arrivals** rather than on the prediction that
produced `H` — clean, repaired (parity landed inside the barrier, the
loss was invisible), or degraded *naming the peers*; and
`fits_in_horizon` is the inequality that decides which.

`DualLayer` crosses a **fine** layer that is local to a node and spans
ticks — the stabilizer code itself, one 𝔽₂ equation per node, read off
its own state at **zero bandwidth** — against a **coarse** layer local
to a tick and spanning nodes: a systematic Cauchy/GF(256) MDS code,
`m` repairs per tick for `m/nodes` bandwidth. Each has a pattern the
other finds trivial. Measured on 4 toric nodes × 6 ticks with node0
and node1 losing ticks {2,3} and node2 losing tick 2:

```
coarse alone (2 parity):  2 repaired, 3 left  [(0,2), (1,2), (2,2)]
fine   alone (no parity): 1 repaired, 4 left  [(0,2), (0,3), (1,2), (1,3)]
crossed:                  2 coarse + 3 fine over 2 rounds, 0 left,
                          24/24 slots byte-exact
```

Tick 2 loses three nodes against two parity shards, so the coarse layer
is blocked; nodes 0 and 1 each hold two unknowns against one equation,
so the fine layer is blocked. The coarse repair at tick 3 leaves them
holding one unknown each, at which point their own code constraint
names it — and the layer that closes the pattern is the one that cost
no bandwidth at all, because it is the code that was already protecting
the qubits. (The construction is `holochron`'s dual tower with the
quantum code standing in for the fine field-scale layer; the barrier,
the delay geometry and `fits_in_horizon` are `cliff`'s.)

**Honest scope, stated.** The distributed vocabulary is Clifford —
physical Paulis at any weight and transversal CX; a `t` does not
transport through a Pauli frame and `compile_clifford` already refuses
it by name. This is a *simulation* of a distributed run: the loss, the
delays and the clocks are injected, not measured off a socket. What is
real is the algebra — the coordinatewise link rule, the syndrome's
linearity, the erasure capacity measured off the code, the exact
GF(256) repair. And the scope of *this* module is the **control
plane**: it distributes the frame and leaves each node holding its own
patch. Sharding the register itself is the next section, and it comes
out of the same symplectic form.

Remaining rungs, in the order they matter:

- **A real socket under it.** `DelayGeometry` takes injected delays and
  `Barrier` takes injected arrivals. Driving both from a live
  transport — measuring the delay tails rather than declaring them, and
  letting `HorizonPolicy`-style estimation set `H` — is what turns the
  measured algebra into a measured system. The interface is already the
  right shape: everything above consumes arrivals, not sockets.
- **Speculation, and rollback.** The barrier gives a sealed history;
  the other half of running in real time without being in real time is
  a speculative present rebuilt forward from the seal each tick, rolled
  back when the real inputs land. For a Pauli frame that rollback is
  free — re-XOR the deltas in the new order — which is a strong hint
  that the frame is the right thing to speculate on.
- **Measurement outcomes on the wire.** Today the payload is the frame.
  A measurement-conditioned distributed run also has to replicate
  outcomes, and those are *not* 𝔽₂-linear in the frame; the natural
  move is the schedule's `FeedbackOp` tree as the shared program with
  outcomes as extra shards, which the coarse layer already codes over
  unchanged.
- **Erasure decoding past the union bound.** The fine layer resolves a
  node when its unknowns narrow to one. A node with several unknowns
  whose supports are *disjoint* is still determined by the single
  residual equation when the combined support is correctable; taking
  that would strictly widen the fine layer at no bandwidth cost.
- **Distance from the geometry.** `min_horizon` and
  `ErasureDecoder::certified_capacity` are two numbers about the same
  network — how long a loss can take to repair, and how much loss the
  code absorbs. Choosing the code distance *from* the measured delay
  tail (rather than picking both independently) is the design question
  this module makes askable and does not yet answer.

## Sharding by the symplectic form (the stitch) — SHIPPED (core)

`lateral` distributes the control plane. `stitch` distributes the
register, and the mechanism was already in the algebra: the phase-free
Pauli group is an 𝔽₂ space under an **alternating** form, and every
alternating form has a symplectic normal form
`radical ⊥ H₁ ⊥ … ⊥ H_h`. `orthogonalize` computes it by symplectic
Gram–Schmidt in `O(rank²·n)`, with no `2ⁿ` anywhere.

**The radical carves; the pairs cannot; so the pairs partition.** `r`
mutually commuting directions cut the `2ⁿ` module to `2^{n−r}` — that
is stabilizer encoding, stated as linear algebra. A hyperbolic pair
names no region at all, because `eᵢ` and `fᵢ` anticommute and **no
abelian subgroup, hence no node, can hold both**. Each pair therefore
forces a binary choice, giving `2^h` maximal isotropic extensions of
rank `r + h`; two distinct ones contain both `eᵢ` and `fᵢ` for some
`i`, and nothing is fixed by both, so their regions meet in zero. The
sum is direct and the arithmetic is exact:

```
2^h · 2^{n−r−h} = 2^{n−r}
```

**Read off the generators, not told.** Nothing hands `orthogonalize`
the code's parameters; it recovers them. Measured on the span of
generators ∪ logicals: toric L=2 → `(r, h) = (6, 2)`, L=3 → `(16, 2)`,
surface d=3 → `(8, 1)`, d=5 → `(24, 1)`. The radical **is** the
stabilizer group (isotropic, and central in the whole span — both
asserted), the hyperbolic pairs **are** the logical `(X̄, Z̄)` conjugate
pairs (anticommutation asserted per pair), and `Volume::centraliser`
returns the normalizer at rank `n + k`, computed as an 𝔽₂ kernel with
no enumeration.

**Checked on amplitudes, not on dimensions.** The four toric L=2
slices are prepared as actual states (each selection's basis choice
read *off the volume* by `contains`, not assumed from the pair order),
shown to sit in the code space to 1e-9, shown to be a **frame rather
than an orthogonal basis** — Gram `1, 1/√2, 1/2`, exactly the tensor
of `|0⟩/|+⟩` per axis, determinant 1/4 — and a random vector projected
into the code space by the full `2^r`-element stabilizer sum
reassembles from the four slices to `< 1e-9`. Every pair of slices,
joined, stops commuting; every selection is maximal isotropic of rank
`n` and carves dimension exactly 1.

**Where the exponential goes.**

```
   n    r    h |     nodes | per node |     network |    monolithic
  18   16    2 |         4 |    106 B |       424 B |       4.19 MB
  30   20    5 |        32 |    216 B |     6.91 kB |      17.18 GB
  30   20   10 |      1024 |    256 B |    262.1 kB |      17.18 GB
  40   20   20 |   1048576 |    416 B |    436.2 MB |      17.59 TB
```

Per node the cost is polynomial in `n` and **does not mention the node
count** — 1024 nodes and 32 nodes on the same register differ by 40
bytes each. The network total is `2^h · O(n²)`: exponential in the
*logical* count, not the register width, because the code has already
pulled the exponent from `n` down to `k = n − r` and the shard plan
spends it as machines instead of as memory. The price, stated rather
than hidden: each node carries a whole tableau to hold one
coefficient, `O(n²)` bytes where a bare `2^k` amplitude vector holds
16 — bought in exchange for a per-node object that is polynomial,
closed (a Clifford acts on a slice's tableau locally, no
communication), and independent of every other node.

And `h` is **hidden in the dimension**: two stitches of region
dimension 1024 can be 8 slices or 128. Measuring the object's size
says nothing about how many pieces it is in — which is what makes `h`
usable as structure rather than as an accounting quantity.
`selections()` refuses above 16 pairs by name, because `h` is an
exponent and expanding it is a decision.

(The construction is `cliff-core`'s `stitch` and `volume`, whose
documentation states the correspondence outright — the twist form is
the Pauli group's commutator phase, an isotropic volume *is* a
stabilizer, a centraliser *is* a normalizer, `2^h` regions reassemble
to `D/2^r`. What is added here is the decomposition over
`PauliString`, where the slices can be instantiated as states and the
direct sum checked on amplitudes.)

Remaining rungs:

- **The Clifford action on the frame.** A Clifford permutes the
  symplectic space, so it carries one selection to another and the
  coefficient update is a permutation-plus-phase across nodes. Making
  that explicit — which node's coefficient lands where, and what the
  communication pattern of a given gate is — turns the static
  decomposition into an evolution, and is the single most valuable
  thing missing.
- **Non-maximal slices.** When `r + h < n` a slice is a region of
  dimension `2^{n−r−h}` rather than a single state, so a node holds a
  small register instead of a tableau. That is the knob between "many
  tiny nodes" and "few larger ones", and nothing currently exercises
  it.
- **Signs.** `Volume` is phase-free by construction, which is right for
  the 𝔽₂ layer and means a selection names a *basis*, not a state,
  until signs are chosen. The `2^{r+h}` sign choices per selection are
  the rest of the decomposition, and `lateral`'s frame is exactly the
  object that carries them.
- **The dual polarity pass.** `cliff`'s `stitch` carries the other half
  of the construction — slice in the fine polarity where the residue
  is carvable, re-integrate in the coarse one where only the
  reassembled region has a name, with the seam being the 2-adic
  valuation. There is no analogue here yet, and `quantsim`'s `padic`
  module is the natural place for one.

## The residual as a mosaic (path-sum merges) — SHIPPED (core)

`pathsum` reduces a circuit to `h*`, its irreducible path content —
zero for every Clifford circuit at any width, growing with the T-count
rather than the register. Readout then enumerated `2^{h*}`, whatever
the residual looked like. `amplitude_merged` declines that the way
`MosaicState` declines a single representation: the residual is a
**partition**, and each part gets the lens it deserves.

Three lenses, and the third is the one enumeration cannot reach:

* **factor** — a disconnected interaction graph makes the sum a
  *product*, so each component is a separate, separately-priced
  problem. The same move `FactoredState` makes on the register.
* **merge** — two components that are the same polynomial *up to
  variable renaming* have the same sum, so the second costs a lookup.
  These are the merges; `MergeStats::distinct_forms` is what the route
  pays in place of `2^{h*}`. Canonicalization is a deterministic
  relabelling refined by an invariant, which makes the memo **sound by
  construction** (equal keys are equal polynomials up to renaming) and
  incomplete by degree (a weak refinement misses merges, never returns
  a wrong one).
* **branch and reduce** — pin one variable and run [E]/[G]/[V] again on
  each child. A residual that stalls under the rules frequently
  *unstalls* once one variable is fixed, and the child collapses to a
  closed form instead of to two more branches.

The answer is identical to `amplitude` — asserted at 1e-16 over 200
random circuits × every amplitude × every branching policy, not
approximated — and the budget is a **named refusal**, never a silent
fallback to enumeration.

**What it measures.** Three regimes, and the third is the one that
matters:

```
(HT)^k H          h* = k        enumerate 2^k       nodes 19/29/39/51/63
                                                    at k = 8/16/32/64/128

same qubits/t/depth, 1D vs 2D:
  side  qubits    t   h*    1D nodes   2D nodes   ratio
     3       9   18    9          17         25     1.5x
     4      16   32   16          29         59     2.0x
     5      25   50   25          37        259     7.0x
     6      36   72   36          45        853    19.0x

square family (n qubits, n layers, t = n²):
    n     t    h*   enumerate      nodes   log2(nodes)/n
    5    25    20   2^20              97            1.32
    7    49    42   2^42            1773            1.54
    8    64    56   2^56            4361            1.51
   10   100    90   2^90           31513            1.49
```

**Thin magic is logarithmic** — `2^128` by enumeration is 63 nodes.
**Width alone is free** — the 1D twin stays linear to 36 qubits and
t = 72 while its 2D twin, with identical qubit count, T-count and
depth, pulls away by 19×; the obstruction is *separator growth in the
coupling graph*, not register size. And the surviving exponential is
in **√t, not t**: `log2(nodes)/n` settles at ~1.5 across n = 3..10, so
31,513 nodes stand against `2^90 ≈ 1.2e27`. That is a strict reduction
of the growth law and **not** a removal of it, and the module says so.

**The election, and why it is the mosaic's finding.** Once a component
stalls, which variable to pin is a cost decision with an order of
magnitude in it. Measured head to head:

```
family                     First    MaxDeg    MinRem   Elected
(HT)^64 H  (a chain)         255       131        51        51
grid 6x6 L=2 (shallow)      3263      4469      1813       853
square n=8 (deep)          21321    197485     11345      4361
```

Connectivity is the whole game on a chain — cutting the middle turns
`2^k` into `O(log k)` — and **flat** on a shallow grid, where removing
any one variable disconnects nothing and degree is the only signal
left. Each single signal is 2–45× worse than the other somewhere.
`Pivot::Elected` composes them (connectivity first, degree on the ties)
and is the best or tied-best in every regime, strictly better than both
parents where they disagree most. That is `MosaicState`'s result on a
different axis: *the right lens is a property of the part, not of the
solver*, so the election belongs per component and has to be measured.
Every policy returns the same amplitude; the choice is cost only.

**Cross-validation, which is the point.** These laws were first
measured by a sibling engine over `ℤ[ζ₈]` with an entirely different
rewrite system (ℤ₈ cofactor cases on variable-monomials, against this
crate's [E]/[G]/[V] rules on products of parities). Two independent
implementations, no shared code, same three regimes and the same
`√t` law — including the same 35-node count for `(HT)^24 H` under a
connectivity-first pivot. A growth law reproduced across rule sets is
a property of the circuits.

**Where this sits, honestly.** Everything that collapses here is a
collapse theory already predicts: thin/1D magic is matrix-product
simulable, and the surviving wall is the treewidth law
(Markov–Shi) arrived at from canonical-form memoization rather than
from tensor contraction. The instrument keeps drawing the boundary
where independent theory draws it, which validates the instrument and
means no collapse found so far is a new one. Whether the
grown-separator regime collapses under *any* representation is open —
`BQP` vs `BPP` is open, `BQP ⊆ PSPACE` is all that is proved, and no
artifact here bears on it either way.

Remaining rungs:

- **A stronger canonicalizer.** The refinement is three rounds of
  colour propagation with the original index breaking ties, so
  isomorphic residuals presented in different orders are missed. Affine
  changes of variable and local-complementation moves are the known
  next family, and the 1D column's residual polynomial growth
  (`~w²`–`w³` rather than flat) is the visible symptom of exactly this
  — a solver artifact to attack, not a wall.
- **Per-component lens election, not just per-component pivot.** The
  election currently chooses a *variable*; the mosaic's full contract
  chooses a *representation*, with both predictions ledgered. A
  component that is a bounded-treewidth graph wants tensor
  contraction; one that is a low-rank quadratic form wants the Gauss
  sum in closed form; one that is dense wants neither. Making the lens
  itself the elected thing is the next structural rung.
- **Merges across components, not only within.** The memo recognizes a
  component it has seen before. It does not recognize that two
  *different* components have proportional sums, which is a strictly
  larger merge relation and the natural place a stabilizer-rank-style
  decomposition would enter.
- **The composition question.** Every fragment here is closed and
  every collapse is a structured-instance collapse. The open case is
  not any single fragment but their *composition* — the reason the QFT
  alone is easy, modular exponentiation alone is easy, and Shor is
  neither. A tool that priced a composition by its parts' invariants
  rather than by re-reducing the whole would be the first thing here
  to speak to that question at all.

## Geometric qudits (the obstruction and its boundary) — SHIPPED (core)

`volqudit` builds the three geometry classes of the supplied object /
frame / effective formalism on one ambient object: `stitch::Volume`,
the phase-free Pauli group as an 𝔽₂ space under its symplectic form,
whose meet, join and centraliser cost `O(rank · n)` and never mention
`2ⁿ`.

**The qudit is the obstruction.** Put an isotropic flat `V` into that
space — an abelian subgroup, a stabilizer, a *constraint* — and three
things follow at once, none of them declared:

* `V` is the **position**: a flat, `O(r·n)` bits, movable;
* `V^⊥ = centraliser(V)` is the **bounded boundary**, rank `2n − r`,
  always containing `V`;
* `V^⊥/V` carries a non-degenerate form of rank `2h`, `h = n − r`, so
  `levels = 2^h`.

That is the sheet-and-obstruction picture made exact, and the numbers
come off the codes rather than off the claim: toric L=2 and L=3 both
`h = 2`, surface d=3 and d=5 both `h = 1`, boundary rank `2n − r` in
every case, `V ⊆ V^⊥` asserted, and each conjugate pair asserted to
anticommute with its partner while commuting with the whole frame.
`orthogonalize` on the boundary returns the frame as its radical and
the `h` hyperbolic pairs as the qudit's logicals — *found*, not
supplied. A non-isotropic frame is refused by name: it carves nothing
and bounds nothing, and is not projected onto the nearest legal flat.

**`A_V ⊊ End(V)` is forced, not chosen.** The geometry admits the
logical Pauli group and nothing else, so naming an admissible
operation costs `2h` **bits** while naming a general element of `End`
on the same level space costs `4^h` **complex numbers** — 16 bits
against 65,536 amplitudes for the same 256-level qudit. The theory's
"large relational state volume, small admissible operator volume" is
the shape, and `OperativeCost` reports both columns in their own units.

**Motion, and holonomy from motion.** A Clifford is a symplectic map,
so `transport` carries flats to flats: the frame moves and the
signature does not, which is the frame/object covariance condition,
asserted rather than assumed. A transport that returns the frame is a
**loop**, and what the frame bounds need not come back with it.
`holonomy` reports the induced action on `V^⊥/V`, `curvature` is
`H(γ) − I`, and `holonomy_search` finds the non-flat loops by
enumeration rather than by construction — which matters, because
constructing one needs a symmetry you may not have while enumerating
needs only a budget. On a free qubit the search recovers the whole of
`SL(2,𝔽₂) ≅ S₃` — orders 2 and 3 — from motion alone; on the GHZ frame,
which far fewer words fix, it still finds a curvature-1 loop of order
2. Flat loops report curvature 0 and are excluded; the holonomy of an
open path is refused, because it is not defined.

**Frame in frame.** `nest` makes one qudit's ambient another's level
space, with the inner frame priced in the outer's logical coordinates:
`O(h)` bits rather than `O(n)`, because the outer geometry already
paid for the reduction. **Promotion** exposes only the interface, and
two structurally different frames with equal signatures promote to the
same atom. Interface sufficiency is a conjecture in the theory and
stays one here — what this adds is that it is now *checkable* at this
scale rather than only stated.

**The interior, with its control.** Split the ambient into slots and
the face map `d_i` asks the **coset** — is there any representative of
this logical class missing slot `i`? — so the answer cannot depend on
how the basis happened to be spelled. Measured:

```
compound              h   faces        skeleton   interior
3 free qubits         3   [4, 4, 4]           6          0
GHZ frame             1   [1, 1, 1]           1          1
toric L=2, halves     2   [2, 2]              2          2
```

The first row is the control and it is the important one: independent
qudits side by side have every logical class representable on a single
slot, so every face sees it, the skeleton is all of `V^⊥/V`, and the
vol is **empty**. The GHZ frame keeps exactly one dimension no face
can see — one member of its conjugate pair pushes onto a single qubit
by multiplying in a stabilizer, and its partner `XXX` cannot be pushed
off any qubit at all. That surviving dimension is the vol, in the
theory's sense, computed rather than asserted. `is_brunnian` tests the
stronger condition (nonzero interior, *every* face empty) and reports
false on all three — the faces here are incomplete, not empty.

Everything above is 𝔽₂ mask algebra: a `2^36`-level qudit on 40 qubits
is 320 bits and 69 µs, and nothing anywhere materializes `2ⁿ`.

**Honest scope.** The level space is a *Pauli* geometry, so the
admissible algebra is the logical Pauli group and not the full logical
unitary group; magic lives outside it, exactly as it does in `retro`
and `logical`. Nothing here simulates a state — this is the geometry
of where a qudit is and what it bounds, not an amplitude carrier.

Remaining rungs:

- **A Brunnian frame.** The construction can express the condition and
  nothing tested so far satisfies it. Searching the frame lattice for
  one — nonzero interior with every proper face empty — is the direct
  test of the theory's operative-interiority claim, and its absence at
  small `n` would itself be a result.
- **The operative interior.** `I_op(V) = ∩ ker ∂_i^op` with
  `∂_i^op(T) = d_i ∘ T ∘ s_i` is the operator-level analogue of the
  vol, and it is not built. A logical operation that acts at full
  arity and is invisible on every face is the object the braid
  material points at, and this module now has the faces to define it.
- **Effective geometry — SHIPPED**, in the section below: `polar`
  carries the closure `Im Γ = (⋁Vᵢ)^⊥`, the deficit, the throat as a
  bound, and `Φ(P, U)` as a rank. What remains on this axis is a
  closure over *heterogeneous* representations rather than over
  frames alone — the mosaic's axis, not the polar space's.
- **The operative tower.** `V_{k+1} = ΠΓ(V_k ⊠ V_k^∨)` and the fixed
  point `σ(op_atom(V)) ≅ σ(V)`. `nest` gives one direction of this;
  the self-compound with the dual, and whether any frame is a fixed
  point of it, is the recursion the theory is actually named for.

## The embedding, and effective geometry — SHIPPED (core)

`stitch` works with 𝔽₂ subspaces under a symplectic form and
`volqudit` puts qudits into them, and neither said what the ambient
**is**. It is a projective space, and naming it brings the counts and
the incidence structure with it.

**The embedding.** The phase-free Pauli group on `n` qubits is
`𝔽₂^{2n}` minus the origin — scalars already quotiented out — so its
points are the points of `PG(2n−1, 2)`, and the commutation form makes
it the polar space `W(2n−1, 2)` whose totally isotropic flats are
exactly the abelian subgroups: the stabilizers, and therefore the
frames. A `Volume` is a flat; an isotropic one is a flat of the polar
space; a maximal one is a **generator**. The closed forms are

```
points                  2^{2n} − 1
lines through a point   2^{2n−2} − 1
totally isotropic lines (2^{2n} − 1)(2^{2n−2} − 1) / 3
generators              ∏_{i=1}^{n} (2^i + 1)
```

and each is **checked against enumeration** up to `n = 4` rather than
quoted. (The generator count matching `∏(2^i + 1)` is the same
identity `cliff-core`'s `volume` module states for maximal commuting
sets — two independent statements of one fact.)

**`n = 2` is the doily, and that is the unit.** `W(3,2)` has 15
points, 15 totally isotropic lines, 3 points on every line, 3 lines
through every point, **zero triangles**, and the
generalized-quadrangle axiom holds — so it is `GQ(2,2)`, on four
homogeneous coordinates, and the module prints its 15 points and its
lines as maximal commuting sets. The two properties that *earn* the
name are the ones asserted: a polar space that merely had the right
counts would not be a quadrangle.

**A wider register is a combinatorial space of doilies.** The
non-degenerate 4-dimensional subspaces of `W(2n−1,2)` are copies of
`W(3,2)` sitting inside it:

```
n = 2      1 doily
n = 3    336
n = 4  91392        2^{4(n−2)}(2^{2n−2} − 1)(2^{2n} − 1) / 45
```

enumerated at `n ≤ 3` and matched against the closed form, which
answers at any width and **saturates** past `u128` rather than
wrapping — an honest "too large to name" instead of a wrong number.
And the claim stops where it was checked: `W(5,2)` has triangles and
is *not* a generalized quadrangle, which the suite asserts so the
doily's specialness cannot spread by association.

**Effective geometry closes over the representation.** For an
`Assembly` of frames in one ambient, an operation is admissible for
member `i` exactly when it commutes with `Vᵢ`, so the systemic closure
is the constraint intersection `Im Γ = ⋂ Vᵢ^⊥ = (⋁ Vᵢ)^⊥` —
idempotence checked, not assumed, and the closure contains every frame
it was built from. The finding is that **effective dimension is not
the sum of the parts**: three single-constraint frames on four qubits,
each leaving `h = 3` alone, close onto `d_eff = 1` — a deficit of 8.
That gap is precisely why effective geometry is a separate class from
assembled object geometry.

**The throat is a bound, not a slogan.** `d_eff ≤ min_i hᵢ` holds in
general and is an **equality exactly on a nested chain** `V₁ ⊆ V₂ ⊆ …`,
where each stage's constraint already contains the last. Measured:
nested `local [3,2,1]`, throat 1, effective 1 — tight; crossed
`local [2,2]`, throat 2, effective 0 — slack. Reported as an
inequality with both cases measured.

**Local versus globally effective, concretely.** An invariant is
globally effective when `I = Ī ∘ Γ`. Two assemblies with the *same*
closure agree on `d_eff` — it factors — and disagree on `Σ hᵢ`, 4
against 1. So the sum of local logical ranks is a perfectly meaningful
local quantity that the closure erases, which is §6 of the formalism
instantiated rather than restated. Assemblies with *different*
closures are reported as incomparable rather than answered.

**Dynamic sufficiency.** `Φ(P, U) = PU(I − P)` becomes
`dynamic_obstruction`: the rank by which a transport pushes a frame
out of itself, zero exactly when the transport is a loop — the same
predicate `VolQudit::closes` already answered, now with the
frame-geometry name and a number instead of a bool. Measured on the
GHZ frame: identity and a CX conjugation give `Φ = 0`, a single `H`
gives 1, transversal `H` gives 2.

**Checked against a sibling projective-geometry library.** `twistorDB`
builds `PG(n,q)` generically (`pg.rs`: `ProjectiveSpace`, `Flat` with
span/meet/`flat_count` by Gaussian binomial) and lays the Pauli
symplectic form on it per code (`extras/qec.rs`: `sympl_vector`,
`anticommutes`, the perp as a meet of half-swapped hyperplanes). **No
number disagrees.** Where the two overlap they agree — 15 points of
`W(3,2)`, the `[[5,1,3]]` and `[[7,1,3]]` parameters — and the 35-vs-15
line counts are the ambient-vs-isotropic distinction, not a conflict.
Two things are worth stating precisely because they are small: that
library *computes* the doily's 15 isotropic lines and the `GQ(2,2)`
axiom in an example and **prints** them without asserting either, and
it has no triangle count; the generator counts `∏(2^i+1)`, the
isotropic line counts 15/315/5355, and the doily counts 1/336/91392
do not appear in it at all. So the pinning here is new even though the
geometry is not.

Remaining rungs, with what the comparison sharpened:

- **Closure over heterogeneous representations.** The assembly closes
  over *frames*. The harder and more useful question is closure over
  representation *choices* — the mosaic's axis — where the invariant
  sought is what every admissible representation agrees on.
  `conformance` verifies agreement pairwise; the quotient by
  representation choice is the object that does not exist yet. Neither
  library has anything on this axis.
- **Hashable canonical flats.** `Volume` is a deliberate 𝔽₂/u64-mask
  specialization of what `twistorDB`'s `Flat` does field-generically,
  and the one thing the specialization gave up is `Flat`'s canonical
  RREF with `Hash + Eq`. That is exactly what the next two rungs need
  — flats as map keys — so it is the prerequisite rather than a
  nicety.
- **Doilies as a decomposition, not a count.** Knowing there are 336
  doilies in `W(5,2)` is not yet knowing which ones a given frame
  meets. A frame's *doily profile* — which four-coordinate
  subgeometries it touches and how — would make the combinatorial
  space usable rather than only countable.
- **The generators as a spread.** `∏(2^i + 1)` maximal commuting sets,
  of which `2^n + 1` are pairwise disjoint and cover every point
  exactly once — a symplectic spread, the natural home for a
  mutually-unbiased-bases construction. Confirmed unbuilt in both
  libraries: `twistorDB`'s only spread is the char-0 Penrose fibration
  of `CP³` into skew lines, which is rank-2 and has no 𝔽₂ or
  maximal-isotropic content.
- **The Klein correspondence.** A rank-2 frame is a *line*, and
  Plücker coordinates turn a line into a single **point** on a
  quadric — so "are these two frames compatible" becomes point
  incidence rather than subspace algebra. `twistorDB`'s `plucker.rs`
  has the whole apparatus over ℤ[i]; the 𝔽₂ version is small and
  would change the cost of every pairwise frame question.
- **Line complexes as frame constraints.** One linear equation on
  Plücker coordinates carves out an entire family of admissible
  frames — a constraint on *frames* rather than on operators, which is
  a level the current `Assembly` cannot express. `linecomplex.rs`
  notes that the non-special case is exactly the null system giving
  `W(3,q)`, which is the object this section is about.
- **A general Gram perp.** `centraliser` is hard-wired to the
  symplectic form. `twistorDB`'s `QuadricForm::polar` takes an
  arbitrary Gram matrix, which is what makes the orthogonal and
  Hermitian polar spaces free rather than a rewrite — and
  `is_maximal_isotropic` is then the *self-polarity* fixed point
  `polar(V) = V`, a framing that generalizes where the current
  predicate does not. `exceptional-galois` already builds `GF(q)`
  exactly, which is what a `q > 2` version needs.
- **Transport as one matrix product.** `dynamic_obstruction` and
  `VolQudit::transport` conjugate the frame basis element by element.
  The exterior-power (compound) action moves a whole flat in one
  product, which is the right shape once frames get wide.

## Lattice onto chain, and the census — SHIPPED (core)

An MPS, a path-sum variable list and a qubit index are all *one
dimensional*; a coupling fabric is not. The map between them is an
**ordering**, `curve` makes it a measured object, and the module exists
because the obvious expectation about it is wrong.

**The refutation.** The expectation — a locality-preserving
space-filling curve should lay a lattice onto a chain better than
reading rows — fails by an exact law, at every power-of-two side from
2 to 128:

```
ordering    max dilation          cutwidth
RowMajor    side                  side + 1
Snake       2·side − 1            side + 1
Hilbert     (10·4^{k−1} − 1)/3    2·side − 2       (side = 2^k)
```

Hilbert's cutwidth is **exactly twice** row-major's, and its worst
dilation is `Θ(N)` against `Θ(√N)`: at side 64, 126 against 65 and
3,413 against 64, with 21% more routing swaps. It is worse on every
measure a chain register cares about.

**The reason is a direction error**, and the intuition is natural
enough to be worth naming. A space-filling curve preserves locality
from *curve to plane* — nearby indices are nearby points — which is
what makes it right for spatial indexing and cache layout. A chain
register needs the *opposite*, nearby points getting nearby indices,
and the Hilbert curve's inverse has unbounded dilation. Reading the
rows is bad at the first direction and optimal at the second. Half of
the pre-stated expectation did survive: cutwidth is `Θ(side)` under
**every** ordering, a property of the grid rather than of the curve.
The **snake** turns out to get both halves — continuous like the
curve, and cutwidth, crossings and swaps identical to the rows.

**And the refutation is narrow, which is where the interest is.** It
refutes the curve *as a linear ordering*, and that is precisely the
use that throws the construction away. The Hilbert curve is a **rotor
per cell applied recursively** — each quadrant entered under a
dihedral symmetry, `Rotor` being the eight elements of `D₄` and the
Hilbert rule being transpose / identity / identity / anti-transpose,
two reflections and two rotations. A linear index keeps only the order
the recursion visits cells in. Keep the recursion instead
(`Bisection`) and it beats **every** linear ordering, by exact closed
forms:

```
chain, best ordering    side³ − side       Θ(N^{3/2})
recursive bisection     2·side² − 2·side   Θ(N)
ratio                   (side + 1) / 2     Θ(√N)
```

64.5× at side 128, and unbounded in the side. The **worst** cut is
identical — `side`, the grid's own bound, which nothing beats — so the
surviving half is confirmed from the other side; the **total** differs
by an order. A representation paying per cut over a hierarchy (`mera`,
`bulk`) pays the second row; one paying over a chain's cuts pays the
first. **So the curve was never the wrong idea; flattening it was.**

Two clauses, both stated because the first is easy to guess wrong and
the second is easy to overclaim. The tree's cost is **not** at the
root: its per-level profile is `side · 2^{⌊ℓ/2⌋}`, doubling every two
levels, so the total sits at the *deepest* cuts — separators shrink
per node and the node count grows faster. And nothing here tests
whether an **overlay** of several distinct rotor assignments, used
together as independent addressing bits rather than one at a time,
improves on plain recursive bisection. That is a stronger claim than
anything measured and it stays **open**.

**The census, from the geometry.** `PolarSpace::generators() × 2^n`
gives the stabilizer-state counts 6, 60, 1080, 36,720, 2,423,520 —
reached here by counting maximal isotropic flats and multiplying by
the `2^n` sign choices, and independently by a sibling program's
phase-space census, which computes `2^n · ∏(2^k+1)` directly. The
ratio is exactly `2^n` at every `n`, so the two are the same quantity
counted from opposite ends: states there, flats here. Also shipped:
`symplectic_order` and `symplectic_dyadic_valuation`, with
`v₂|Sp(2n,2)| = n²` and second differences identically 2.

**One honesty note on that last one, adopted from the sibling's own
caveat.** Both implementations compute `|Sp(2n,2)|` *from* the closed
form `2^{n²} ∏(4^i − 1)` rather than by enumerating the group, so what
the test verifies is that `∏(4^i − 1)` contributes no further factors
of two. That is true and it is nearly tautological, and the test says
so rather than presenting it as a discovery.

**A free cross-check that came out of the comparison.** The same
sibling computes the Freudenthal–Tits dimension as
`2 + Σ Der + 4·e₁ + 2·e₂ + ∏ dim(Aᵢ)` where `magic` uses
`3 + Σ Der + 5·e₁ + 3·e₂ + Σ_{k≥3} e_k`. The two look different and
are algebraically identical, because `∏(tᵢ+1) = Σ_k e_k` absorbs
exactly the `1`, one `e₁` and one `e₂` that separate them. Now pinned
in `tests/magic_position.rs` over arities 1–5 with an independently
written `e_k`, so either implementation drifting is caught.

**And one non-defect, checked rather than assumed.** That sibling
refutes a vendored `e8.rs`'s invariant form — `tr_W(XY) + 2β(s,t)` is
not invariant, `−4` is forced, violations ≈0.2% dense with a pinned
witness. This crate's `e8` is the **root system** (240 roots, the
112 + 128 split, the Weyl group) and builds no invariant bilinear form
on the 248-dimensional algebra at all, so the defect has nothing here
to land on. Recorded because "we don't have that bug" is only worth
saying once it has been looked for.

**What the comparison says NOT to import.** The sibling's headline
"the same degree-two boundary, found five times" is flagged by its own
library review as **prose rather than checks** — nine prose claims,
one realised as a computation — and one candidate fifth instance is
separately *refuted* as a coincidence. It is a real pattern and it is
not five measurements, and nothing here cites it as one.

Remaining rungs, sharpened by the comparison:

- **How far is the election from optimal?** `pathsum`'s
  `Pivot::Elected` is a greedy, measured choice, and the sibling
  answers exactly this question for its own greedy elimination order
  against exact search: **optimal through n = 5, and strictly weaker
  from n = 6** (612/1200 = 612/1200 at n = 5; 1023 against 1143 at
  n = 6; 594 against 649 at n = 7). The same experiment against
  `Elected` is cheap, and it would turn "no single signal wins" into
  "and here is what the best possible signal would have won".
- **The order-free width invariant.** The sibling computes the
  *minimum achievable maximum width* exactly — a bottleneck shortest
  path up the subset lattice, one DP over `2ⁿ` subsets — and measures
  the natural order overshooting it by 3–5×. That is the yardstick
  `pathsum`'s pivot election currently lacks: the solver reports what
  it paid, not what it could have paid. Its companion result is the
  one to aim at — width-bounded search whose cost is governed by *the
  answer rather than by n*, a chain at n = 60 resolving in 62 nodes
  against `2⁶⁰`, with the width decomposing over connected components
  exactly as `split_residual` already does.
- **The cubic dissolution criterion.** At degree 3 the sibling has an
  exact criterion for when a lone cubic phase fails to resolve — the
  coupling columns lying in the column space of the free–free coupling
  — verified exhaustively over 67,712 phases with zero disagreements,
  with `rank(E)` as the dial. `pathsum`'s residual is exactly where
  that would apply, and it currently has no criterion at all, only a
  measured stall.
- **The rotor overlay.** The single-rotor-assignment recursion is
  measured; a *set* of assignments used together — several curves as
  independent addressing bits rather than one as an index — is not.
  The natural measurement is whether the union of several bisection
  trees' cut families beats one tree's, which is a question about
  branch decompositions rather than about curves, and the machinery to
  ask it is now present.
- **Higher-dimensional orderings.** `curve` is two-dimensional. The
  sibling searches the general family — a base order on the `2ⁿ`
  sub-cells plus a twist per slot from `F₂ⁿ ⋊ Sₙ` — and its first
  result is a refutation worth carrying: the presentation is faithful,
  so unlike a gate set there is **no semantic quotient** to collapse
  the search onto. A 3-D ordering module would want its exact
  continuity criterion rather than expansion-testing.

## Further non-Cayley–Dickson explorations

`SplitComplex` establishes the pattern (indefinite Born form surfaced through
`born_weight`, `DIVISION = false` capping expectations). Natural next
entries, each a small self-contained `Scalar` impl plus tests:

- **Dual numbers** (`ε² = 0`): nilpotents; automatic-differentiation-flavored
  simulation (state and its parameter-derivative propagate together).
- **Split-quaternions — SHIPPED** as `SplitQuaternion`, and deliberately
  *not* as `CD<SplitComplex>`: the doubling would bury the structure that
  makes them worth having. Held as an explicit **inclusion/exclusion
  pair** `q = inc + exc·j`, the algebra's own ℤ₂ grading, with
  `born_weight = |inc|² − |exc|²` (net) against
  `abs_sqr = |inc|² + |exc|²` (path). ℂ embeds, so it is the first
  non-division algebra here to carry the full standard gate set, and the
  conformance suite applies unmodified. Remaining work:
  - **A destruction-tracking backend.** The pair currently holds what a
    caller puts in it. A backend that *routes* cancelled weight into the
    exclusion channel automatically would make the algebra do at
    amplitude level what `InterferenceState` does with a side ledger —
    and the two are then cross-checkable against each other, which is
    the measurement that would justify the representation on its own.
  - **The boost as a registered gate family.** `SplitQuaternion::boost`
    is unitary and net-preserving; parameterizing it as a research gate
    (`boost(t)` on chosen qubits) would let circuits pump path weight
    deliberately, and the atlas could then measure whether path weight
    is a resource the growth-law machinery can classify.
  - **Signed-measure sampling.** `Backend::sample` drops non-positive
    branches, so a net-negative state reports a surviving total of 0
    rather than its own −1. A sampler that handles signed weights
    honestly (importance sampling against `abs_sqr` with sign carried
    through) would make inclusion–exclusion states samplable rather
    than merely refusable.
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

## Backend characterization — SHIPPED (core)

`bounds` answers *how does this family scale across every
representation*; `conformance` answers *is this representation
correct*. `characterize` answers the question you ask of a new
representation first, and the one this repository was missing for the
E8 backends: **how wide does it go, how much of the gate set does it
reproduce, and what are its best and worst cases.**

- **ceilings, from real refusals.** `construction_ceiling` walks widths
  until the constructor refuses and keeps the refusal's own message;
  `width_ceiling` runs a whole circuit family per width and reports the
  widest that finished, so a backend that constructs 63 qubits and then
  refuses the first Hadamard layer at 21 says so with the guard's text.
  `Wall` distinguishes a construction refusal, an operation refusal, and
  a sweep that simply stopped short — never an absence dressed as a
  limit.
- **fidelity as a count.** `fidelity_census` turns a `ConformanceReport`
  into the number a table wants: gates bit-exact, gates within
  tolerance, gates outside, gates not swept, worst gate and its
  deviation. Bit-exact is tracked apart from within-tolerance on
  purpose — the shipped backends share dense's accumulation order and
  should agree to the last bit, so a gate drifting to 1e-12 is a
  finding, not a pass.
- **min/max envelope.** `perf_envelope` times every registry gate on a
  fresh register, median of repetitions, and reports the cheapest and
  most expensive gate, the support range, and **bytes per stored
  amplitude measured** rather than computed from `size_of`. That last
  number is the one that separates representations at a glance: dense
  2064–4128 B/amplitude at width 8, sparse 62.5–125, the constellation's
  eight-coordinate lattice key 112–144.
- **laws over the widths that ran.** `family_laws` fits memory and time
  laws and stops at the first refusal, so a walled width contributes
  nothing rather than a fabricated cost.

Measured on the E8 backends, which is what prompted it:

- `e8-rep` is **8 qubits, hard** — the spinor representation's own
  ceiling — and reproduces all 39 registry gates bit-exactly.
- `e8-constellation` reproduces all 39 bit-exactly at any width and
  walls on the trait's u64 index, but its **per-gate cost is 10–30×
  dense** (8.7–24.5 µs vs 0.6–3.1 µs at width 8) because every gate
  converts through the tower bijection. Its clifford-brickwork ceiling
  is set by *time*, not memory, at a support of only 2048 — the honest
  statement being that the constellation's cost is dominated by the
  lattice↔bits conversion rather than by how much state it holds.

Next rungs:

- **characterize every shipped backend in CI**, as a table checked
  against a stored baseline, so a per-gate cost regression is a test
  failure rather than folklore.
- **a native-operator cost axis**, so the constellation's group
  operations are priced by the same machinery as its qubit path (they
  are two very different cost regimes sharing one backend name).

## Computing across a set of E8 volumes — SHIPPED (measured, negative)

The question was whether a set of E8 volumes, interacted through the
scale tower, buys a non-linear computational advantage, and whether the
structure could carry full quantum computing. `e8::across` answers both
by running it, and the answers are worth having even though one is no.

- **Reach.** One native finest-scale translation carries into **every**
  copy in the tower: at `m` copies it moves `8m − 7` qubits and its
  influence graph is a single connected component, so the measured
  two-qubit lower bound for any circuit realizing the same permutation
  is `n − 1`. The bound is derived, not asserted: an output bit that
  depends on a different input bit forces a path of gates between them,
  and a graph on `k` vertices with `c` components has at least `k − c`
  edges. Under-sampling the probes can only *drop* edges, so the bound
  is conservative by construction.
- **Where the reach comes from.** At **one** copy there are no carries
  at all — `E8/2E8 ≅ F₂⁸` and the class map is linear, so the
  translation is a bitwise XOR needing zero two-qubit gates. The
  cross-scale coupling is created by having more than one copy, which is
  exactly the "across" effect and is now measured rather than argued.
- **But the advantage is linear.** Each added copy adds exactly 8 to the
  reach and exactly 8 to the lower bound (finite differences, no fit),
  and the fitted law over copies is `Polynomial { degree: 1.04 }` —
  subexponential. One native operation replaces `Θ(n)` two-qubit gates.
  That is a real and useful constant-factor-per-copy win for those
  operations; it is not a non-linear advantage, and the measurement is
  what says so.
- **The native DFT is the limit.** `coordinate_fourier` is a *direct*
  transform: its support grows as exactly `2^m` (fitted base 2.0) and
  its cost law is not subexponential. The one native operation that
  creates superposition is the one that pays exponentially for it.
- **Full quantum computing: through the qubit path only.** Every native
  operator — translate, modulate, reflect, permute, Fourier — maps a
  coset carrying a linear character to another one, verified on a
  genuinely spread state (a size-1 support satisfies the class for
  trivial reasons and proves nothing, so the check prepares spread state
  first). A `t` driven through `Backend::apply` keeps the coset and
  breaks the character (residual 1.41), so the qubit path reaches states
  no sequence of native operators can. Universality on the
  constellation is therefore real but bought entirely on the qubit path,
  which prices as sparse with a larger key — precisely the 112–144
  bytes per amplitude the envelope measures.

Next rungs:

- **A quadratic phase operator.** The native set has linear characters
  and no quadratic ones; whether a well-defined quadratic phase exists
  on `E8/2^m E8` is the sharpest open question here, because it is the
  operator that would move the native class from "affine + linear
  character" toward something strictly larger.
- **A fast native transform.** The `2^m` direct DFT is the current
  ceiling on the native side; a scale-recursive factorization would make
  the tower's self-similarity pay in the transform the way it already
  pays in the storage.
- **Native-operator circuits.** Reach is measured one operation at a
  time. What a *sequence* of native operations reaches — and whether the
  linear-per-copy advantage compounds or saturates — is unmeasured.

## Graph-state measurement — SHIPPED (core)

`bundle::project` used to be a documented no-op. That was worse than it
looked: `for_each_nonzero` materializes, so the default `measure` drew a
*correct* outcome and then left the state uncollapsed — a second
measurement of the same qubit could disagree with the first. Silent, and a
correctness bug rather than a missing feature.

`PolarityBundle::collapse` now does the update in the description:

- **An isolated site** carries the product state `V|+⟩`, so the collapse is
  a single-qubit projection and the probability comes off `V`'s matrix — it
  can be exactly 0 or 1.
- **A site with a neighbour** is rotated until its operator sends `Z` to
  `±Z`, by the same breadth-first local-complementation search `apply_cz`
  uses with a different target predicate. After that its two outcomes are
  equally likely; its edges are deleted, a `Z` goes to each former
  neighbour when the bare graph state's eigenvalue is `−1`, and the site is
  left in the state it was projected onto.
- `measure` is overridden to take its bias from the description, so it is
  `O(deg²)` rather than `O(2^n)`.
- The collapse is journalled as ONE semantic step. Its internal
  complementations are suppressed, because replay re-runs `collapse` and
  would otherwise apply them twice — the history worth keeping is "site a
  was measured and came out `outcome`", not the search that implemented it.

**A pre-existing bug fell out of it.** A site's operator is `U · Z^spin`,
so right-multiplying *that* by a generator means composing `Z^spin g Z^spin`
onto `U`. `sqrt_z` commutes with `Z` and is fine; `sqrt_x` does not — so
`local_complement` was silently wrong on any site with a spin set, as were
the two breadth-first searches that predict which word to apply. It was
latent because the `Backend` path never sets spins; the collapse rules were
the first code to exercise it. Isolating it took separating the decorations:
vops-only passed, spins-only passed, both failed.

Verified over 672 cases against dense projection, worst deviation under
1e-9, plus three-deep measurement sequences.

Next rungs:

- **Non-materializing `sample`.** `measure` is native now, but `sample`
  still inherits the materializing default. Per-shot clone-and-measure
  would make the bundle a genuine sampling backend and let it join
  `sampling`'s XEB machinery.
- **Measurement in `characterize` and the atlas.** With `project` working,
  the bundle can carry adaptive circuits and the evented scheduler, which
  is where a graph-state representation should be strongest.
- **`closure` still imports only `bundle` and `error`.** The journalled
  history and retrodiction work remains outside the simulator, and a
  measurement is exactly the kind of event a closure history should stamp.

## The √2 obstruction — RESOLVED (measured)

Two modules reported the same number from different directions:
`e8::across` measured a `t` breaking the native linear-character class by
√2, and `selfhost` measured a `t` untouched by linearization with the same
√2 residual. `phase` resolves it.

The instrument is the iterated discrete derivative
`(Δ_a f)(p) = f(p+a)/f(p)`: a phase function has **degree ≤ d** when every
`(d+1)`-fold derivative vanishes, and **degree 1 is exactly being a
character**. Measured results:

- **The degree law.** `phase degree = multilinear degree +
  log₂(denominator) − 1`, confirmed on eleven diagonals spanning both dials
  independently. A diagonal is a character iff both dials are minimal:
  multilinear degree one AND ±1 valued.
- **The ladder is the Clifford hierarchy**, rediscovered from measurement:
  degree 1 the ±1 characters, 2 Clifford (`s`, `cz`), 3 the first
  non-Clifford diagonals (`t`, `cs`, `ccz`), 4 (`ct`, `cccz`).
- **The self-host floor, derived.** The stack reduces the multilinear term
  only, so the floor is `log₂(denominator)`. A `ccz` (`b = 1`) reduces to a
  character; a `t` (`b = 3`) cannot at any depth. This explains the earlier
  measurement rather than restating it.
- **Two groups, not one.** At one E8 volume the residue group IS the bit
  group under XOR — verified over all 256 × 256 pairs, because `E8/2E8 ≅
  F₂⁸` and `class_of` is linear. From two volumes on they differ on most
  pairs, and the native modulation is degree 1 on the residue group while
  being degree 5 on the bits. A `t` is degree 3 on both. The native
  operators and the qubit path are characters of *different* groups, and
  each is high-degree from the other's side.
- **The number itself.** √2 is the order-two residual of the `t` phase,
  `|i − 1|`. The full ladder is `[2 sin(π/8), √2, 2, 0]`.

So the question "does a quadratic character exist on `E8/2^m E8`" has an
answer: yes — degree 2 is the Clifford level, and an `s`-like denominator-4
phase is one. It does not make a `t` native, because `t` is degree 3.

Next rungs:

- **Add the degree-2 and degree-3 native operators and re-measure the
  class.** `across` currently implements translations, modulations,
  reflections, permutations and the coordinate DFT — all degree 1. The
  degree ladder says what is missing at each level; adding level 2 should
  enlarge the measured class to coset-with-quadratic-character, and level 3
  should reach `t`. Whether the *support* stays a coset under those is the
  measurement to take.
- **A degree axis in the boundary atlas.** Phase degree is a structural
  resource like support or bond dimension. A family's maximum diagonal
  degree ought to be an axis parameter, so the atlas prices circuits by
  where they sit in the hierarchy.
- **Non-diagonal degree.** The instrument is defined for phase functions.
  The Clifford hierarchy is not restricted to diagonal gates, and whether
  this derivative construction extends to the general case is open.

## Progressive gate-result memoization — SHIPPED (core)

`memo` treats qubit operations as `n`-wide operation objects over a shared,
content-addressed entry table, holds the whole computational path at once,
and journals the state so branches can be re-explored.

- **The geometry is memoized away by content addressing.** A fused operator
  over an ascending support is indexed by position within that support, so
  it carries no absolute qubit information: identical structure anywhere in
  the circuit is the identical matrix and therefore one entry. The key is
  every coefficient's exact bit pattern (`Scalar::coeffs`), so a hit is an
  identity rather than a hash guess, and the mechanism is generic over the
  amplitude algebra (verified over ℍ as well as ℂ).
- **Fusion sound by disjointness.** Open supports are disjoint; touched
  groups merge when their union fits, because disjoint operators commute.
  Operations are emitted in **commit order** — a valid linearization
  because a qubit has at most one open owner and ownership transfers only
  at commit.
- **Measured reuse.** A rainbow collapses to one entry; `brickwork-10x6`
  runs 81 gates as 16 operations over 8 entries; a random circuit reuses
  almost nothing, which is the honest signal that it has no structure to
  exploit. `max_fuse` is a monotone dial trading operations (61 → 9) for
  entry bytes (448 → 169 216).
- **Journalled unwind/rewind.** `Explorer` snapshots on a stride and
  rewinds to any earlier step, landing exactly on the fresh-run state
  (asserted at every step). `explore` shares a prefix across variants:
  counted work 288 → 32 at sixteen variants (9.00×), measured 3.82×, with
  the deviation against from-scratch runs reported alongside so a speedup
  can never hide a changed answer.
- **A real bug found and pinned.** Emitting in opening rather than commit
  order let a still-open group acquire a qubit an already-committed group
  had used, applying gates out of order for a measured 1.408-amplitude
  error. The minimized nine-gate case is a regression test.

Next rungs:

- **Persist the entry table across circuits.** The table is currently
  per-plan. A table shared across a whole study would make the second
  circuit of a family cheaper than the first, which is where "progressive"
  should really pay.
- **Commutation-aware grouping.** Fusion is deliberately restricted to the
  disjointness argument, so it will not reorder a gate past a
  non-overlapping-but-commuting neighbour. A Pauli-frame or
  diagonal-commutation pass would widen the groups without weakening the
  soundness argument.
- **A rewind-cost law.** The checkpoint stride trades journal bytes against
  replay work; the crossover is currently a knob rather than a measured
  law, and `bounds`-style law fitting would make it one.
- **Content-addressed states, not just operators.** The same key idea
  applied to the state at a path node would let two branches that
  reconverge share their continuation — the natural next step for holding
  the whole path at once.

## The self-computing object — SHIPPED (core)

A geometric object that is computationally active as a feedback system
*and* on its own purpose, expanding its capacity recursively by computing
itself at each layer, linearizing the diagonal. `selfhost` builds it.

One E8 volume is `E8/2E8 ≅ F₂⁸` — one byte, eight coordinates — and the
measured F₂-linearity of the class map is the load-bearing fact: a
coordinate can hold an arbitrary F₂ function of the substrate and still be
a coordinate. A `SelfHostedStack` is a substrate (layer 0, the object on
its own purpose) under a tower of such volumes, where layer `k` holds the
degree-`k+1` monomials of the layers below it (the object computing
itself), populated by evaluating them (the feedback).

- **Linearizing the diagonal.** Any diagonal unitary is a phase
  polynomial; degree 1 is the case that factorizes into single-qubit
  phases and is a *character* of the bit group — the one diagonal an F₂
  volume applies natively. `linearize` rewrites degree `d` as degree 1 by
  substituting the coordinate that holds each higher monomial.
- **Each layer buys exactly one degree.** Measured for degrees 2 through
  7: depth = degree − 1, linearized degree 1 throughout, deviation
  exactly `0.0` against applying the diagonal directly on the substrate,
  on dense, sparse and adaptive alike.
- **A non-Clifford diagonal becomes a character.** `ccz` moves from
  character residual 2.00 (maximal) to 2.4e-16, and its single-qubit phase
  reproduces the registry's three-qubit `ccz` to under 1e-12.
- **The boundary held apart.** A `t` is degree 1 already, so the object
  does nothing — and its eighth-root phases never become ±1 valued. Its
  residual is the *same √2* `e8::across` measures for a `t` on the
  constellation. Degree reduction and character-hood are different
  properties and `LinearizationReport` reports them separately rather than
  letting one imply the other.
- **The cost, counted.** The expansion cannot be free, or the native
  operator set would manufacture non-Clifford diagonals from nothing.
  Writing a degree-`k` monomial is a `k`-controlled X — the non-native
  work — paid once against per-use. Counted crossover: the second use.
  Measured wall-clock crossover on sparse: 128 uses, because a classical
  simulator applies a diagonal kernel in `O(support)` whatever its arity,
  so the win lands in the slope (164 vs 274 ns/use) not the constant. Both
  are reported.

Next rungs:

- **The stack as a `Backend`.** The object is currently a planner and a
  verifier over other backends. As a representation in its own right it
  would join `conformance`, `characterize` and the boundary atlas, and be
  priced by the same machinery as everything else.
- **Coordinate reuse across layers.** Volumes are quantized to eight
  coordinates and a sparse polynomial wastes most of them (a `ccz` uses 2
  of 16). Sharing partial products between monomials — `x₀x₁` serving both
  `x₀x₁` and `x₀x₁x₂` — would cut both width and Toffoli count, and the
  measured occupancy is the number to drive up.
- **The quadratic character on `E8/2^m E8`.** The F₂ stack linearizes real
  diagonals into genuine characters; the `t` boundary needs a character
  valued in `2^k`-th roots, which is the open question `e8::across`
  already flags. The stack narrows it: what is needed is not a new layer
  but a coordinate group that is `Z/2^k` rather than `F₂`.

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

## Geometric closure and retrodiction — SHIPPED (core)

`closure::ClosureHistory` is a journalled representation whose past
configurations (entanglement included) replay exactly, with closure
stamps of two bits per independent loop, geometric localization from
which loops broke, and retrodiction by bisecting the stamps. Remaining
work:

- **A short-cycle basis.** Evaluating closure costs the *total loop
  length*, and a spanning-forest basis produces long loops: 1.7M total
  length at 14 400 sites against a rank of 14 161. A minimum-length
  cycle basis (or, on a lattice, the plaquettes) would make evaluation
  linear in the link count. This is the single biggest cost item and
  the number is printed in the example rather than omitted.
- **Multiple simultaneous breaks.** Localization currently assumes one
  perturbation: it intersects the broken loops and subtracts the intact
  ones. Two perturbations produce a broken set no single site explains,
  and the report correctly says *ambiguous* — but it could instead
  solve for a minimal set of sites consistent with the pattern. That is
  the honest generalization, and it is not yet done.
- **Correcting the chirality channel.** An ordering break localizes to a
  link exactly, and correction refuses because the link does not
  determine which transposition moved. Journalling position changes
  finely enough to invert them would close that gap.
- **Closure over link changes.** A twist link appearing or vanishing
  changes the basis itself rather than the stamps, so it is currently
  detected only as a rank change. Treating the basis as a stamped
  quantity in its own right would put link errors on the same footing
  as fiber and ordering errors.
- **Stamp scheduling.** Temporal resolution is exactly the stamp
  interval. Adaptive stamping — dense where closure is fragile, sparse
  where it is not — would buy resolution without buying bits, and the
  fragility is already measurable from the loop structure.

## The polarity co-bundle — SHIPPED (core)

`bundle::PolarityBundle` holds polarity as a fibered, re-orderable,
journalled structure: sparse twist links as the base, per-site fibers
carrying frame and sign, chirality from reordering, coarse-graining with
a measured cost, commonality-confined interaction, and journal replay —
running at 10^6 sites. It denotes a graph state dressed by local frames.
Remaining work:

- **Make it a `Backend` — DONE.** `PolarityBundle` implements
  `Backend<S>`, is registered as `"bundle"` in
  `BackendRegistry::standard()`, and evolves under Clifford circuits by
  vertex-operator composition, local complementation and edge toggling.
  Measured exact against dense (deviation 0.0) on the circuits it runs.
  What is left of it:
  - **Complete the vertex-operator reduction.** One configuration in
    forty random 30-gate Clifford circuits still refuses: reducing both
    `cz` endpoints into the diagonal subgroup does not converge when
    they are each other's only handle. The refusal is by name and never
    wrong, but it is incompleteness, not a design limit.
  - **Measurement.** `project` is a deliberate no-op, so `measure` and
    `sample` fail rather than collapsing wrongly. Graph-state
    measurement is a known algorithm and a separate one from gate
    action.
  - **An axis in `bounds.rs` — DONE.** `bundle` and `e8-constellation`
    are atlas axes. Adding them found `memory_bytes` counting the
    journal, which made the atlas read the audit trail's growth as the
    representation's; structure-only reporting fixed it and the law
    reads Polynomial. Remaining nearby: `e8-rep` is registered but has
    no scan family that suits its 8-qubit-native shape.
- **Non-Clifford escape as a measured budget.** A `t` gate leaves the
  sector; the bundle detects that after the fact
  (`verify_against` deviation). Carrying a small superposition of
  bundles with a magic-state budget would let it degrade gracefully and
  *report* the cost, which is the shape the rest of this crate uses.
- **Local complementation as the re-ordering group.** Chirality
  currently tracks generator order. The physically meaningful
  re-ordering on graph states is local complementation — the operation
  that preserves the entanglement class while changing the base. Adding
  it, with the journal recording each move, would make "re-orderable"
  mean something stronger than sequence order.
- **Blind link tomography.** `verify_against` recovers fiber signs given
  the base. Recovering the *base* from measurements alone — discovering
  the link set rather than confirming it — is the harder and more useful
  direction, and the one that would make "tomographically understood"
  true without qualification.
- **Compact adjacency.** `Vec<Vec<u32>>` costs 24 bytes of header per
  site. A CSR-style store with an overflow area would cut the structural
  footprint several-fold at 10^6 sites, where it is already the second
  cost after the journal.

## Polarity systems — SHIPPED (core)

`polarity::PolaritySystem` is the twisted group algebra `ℝ^τ[F₂ⁿ]` with
its structure measured by brute force (twist rank ↦ centre, maximal
isotropic subgroup, matrix block), `scalar::Polarity<N>` is the fully
twisted case as an amplitude type, and `pairwise_signature` /
`ghz_sign_obstruction` measure exactly where pairwise-local data stops
holding a state. Remaining work:

- **A partial-twist backend.** The dial is currently a statement about
  algebras, not a representation. A backend that stores the isotropic
  sector as sign bits and pays `2^{r/2}` only for the rest would put
  twist rank on the same footing as support, cluster size, bond
  dimension and T-count in `bounds.rs` — and `advantage_scan` could then
  classify it. That is the honest way to find out whether the dial buys
  anything the Clifford frame does not, and the answer might well be no.
- **Twist rank against T-count.** The Clifford frame's measured `2^t`
  wall and the polarity system's `2^{r/2}` block are suspiciously the
  same shape. Whether T-doping literally raises the effective twist rank
  is a measurable question, not a settled one, and it should be measured
  before it is claimed anywhere.
- **`k`-body signatures.** `pairwise_signature` stops at two bodies
  because that is what the locality hypothesis proposed. Generalizing to
  `k` and measuring the smallest `k` that separates a given family would
  turn the GHZ counterexample into a curve — the *correlation order* a
  state actually needs — which is a genuinely new axis rather than a
  restatement of an old one.
- **Exact storage for `Polarity<N>`.** Const-generic arrays cannot be
  sized `2^N` on stable, so every `N` pays `2^MAX_POLARITIES` slots and
  `memory_bytes` over-reports for `N < 4`. A macro-generated family of
  exactly-sized types would fix it; `SplitQuaternion` is the
  exactly-sized `N = 2` case in the meantime.

## Braided boundary encoding — SHIPPED (core)

`braided` is live: the register as `n` mutually encoding boundaries with
the navigation, not the storage, as the new object. Shipped and measured
— the mutual encoding in both directions (`ρ_A`/`ρ_B` spectra to
2.2e-16, `ρ_A` alone purifying to a spectrally correct `B`), the
faithful Artin action making path identity decidable, `cayley_ball`
growth, `PeriodicPath` with necklace/Lyndon counts and Duval
factorization, `majorana_generators`/`majorana_bilinears`/
`fibonacci_generators` with the braid relations verified on the actual
matrices, `orbit_closure` (Ising closes at 192 / 23040; Fibonacci does
not within the radius), `commutator_residual`/`bch_residual` fitting the
bracket's order from measurement, `realized_rank` measuring the collapse
onto `so(6)`, and `RecursionLedger` with `projective_order` as the
decisive per-path closure test. Remaining work:

- **A boundary-atlas axis — SHIPPED, and it certifies.** `BraidedState`
  (`src/backend/braided_state.rs`) holds the word as the address and a
  Clifford frame as the state, because every Majorana generator is a
  weight-≤2 Pauli rotation (`majorana_local_gate`, verified against the
  dense generators at every width and basis state). It is registered,
  conforms at 8.0e-16, and on a braid family the atlas reads it
  `mem Constant, time Constant` and names it in the verdict. What is
  still open: the footprint is linear in *depth* because the frame keeps
  a replay log. The tableau alone determines the state, so a frame
  variant that discards the log — accepting that it can no longer flush
  by replay — would make the representation depth-independent as well as
  width-flat. That is a change to `CliffordFramedState`, not to this
  backend.
- **Larger strand counts for the group-theory measurements.**
  `MAX_STRANDS = 16` bounds only `majorana_generators`, the dense
  matrices `orbit_closure` and `realized_rank` walk — the *backend* has
  no such limit and runs at 63 qubits (126 strands). Walking the closure
  with a Clifford tableau instead of dense matrices would lift the
  measurement side too, using exactly the local-gate form the backend
  already has.
- **The Artin action's cost.** `BraidWord::equals` compares free-group
  images, which grow with the word; it is exact and it is not cheap.
  Bringing in a normal form (Garside, or the handle reduction the
  literature uses) would make `cayley_ball` reach further, and the
  existing exact comparison is the ready-made oracle to verify it
  against.
- **Non-adjacent boundaries.** The realizations braid adjacent strands
  only, which is what `B_n` presents. A register whose boundaries are
  coupled by a general fabric would want the *loop braid* or a
  surface-braid group, and the question of what the Artin action becomes
  there is open — the honest statement today is that the module measures
  `B_n`, not an arbitrary boundary graph.
- **The recursion's second direction.** `RecursionLedger` ascends by
  applying the inverse word, which is exact but is not the same thing as
  addressing a rank configuration directly. Reading the state *at* a
  given rank without replaying the path — the "address is the content"
  claim in its strong form — needs the group element to be indexable,
  which is available exactly in the finite (Ising) case and is precisely
  what the atlas axis above would expose.

## Interference as a congruence — SHIPPED (core)

`padic` is live: phases in `ℤ/M`, the CRT diagonal, the interference
character from trailing digits, the journalled sweep priced against a
closed form, and the radix read off the geometry. Shipped and measured —
`Radix`/`crt_phase_factors`/`factored_field` (75600 amplitudes in 75,
2.3e-15, register stayed a product), `character` (cost in the prime
count, not the modulus), `CrtDiagonal` and `compare_diagonals`,
`Sweep`/`sweep`/`predicted_writes_per_point` (1.9995 predicted, 1.9995
measured), `fringe_period`, `geometry_radix`/`best_rational` with the
golden ratio's Hurwitz quantity pinned. Remaining work:

- **The amplitude, not only the character.** The module resolves the
  *congruence* structure — in-phase, antiphase, fringe period, root-of-
  unity order — and recovers the amplitude by evaluating the factored
  field. What it does not yet do is stop early on a *magnitude*
  tolerance, because the p-adic order resolves fine structure first and
  the magnitude needs the coarse end. A two-ended sweep (p-adic from
  below for the congruence, ordinary from above for the magnitude) is
  the honest shape of that, and the residual is already the natural
  dial.
- **The register as a backend — SHIPPED, in its qubit form.**
  `PhaseFieldState` (`src/backend/phase_field.rs`) is the registered
  representation: an exact phase polynomial over `ℤ/M` on an affine
  subcube, conforming at 0.0 amplitude deviation and certifying the IQP
  core on the atlas in both memory and time. What is *not* shipped is
  the rank-`k` form — the backend holds one phase field, so two
  genuinely mixed waves materialize where a rank-2 object would not.
  Growing the rank on demand, with the atlas axis being the interfering-
  component count, is the remaining rung and the natural place for
  `factored_field`'s per-wave registers to become a backend rather than
  a measurement.
- **Multi-dimensional phase arguments.** `factored_field` takes a 1-D
  argument; an `n`-D field is a product over the axes and each axis
  CRT-factors, so the construction should compose directly. It has not
  been built or measured, and the module refuses rather than pretending.
- **The QFT connection made explicit.** `ℤ/M ≅ ∏ ℤ/p_i^{n_i}` is exactly
  the Good–Thomas factorization of the DFT, and `mixed::fourier_d`
  already ships the component transforms. Measuring the full
  `QFT_M = ∏ QFT_{p_i^{n_i}}` decomposition against the crate's existing
  QFT would tie the phase register to the gate-level machinery rather
  than running beside it.
- **Irrational geometry as a first-class regime.** `geometry_radix`
  reports the convergent's error and stops. Sweeping the *sequence* of
  convergents and measuring how the interference character converges —
  which structures are stable under refinement and which flip — is the
  measurement that would say something about quasi-periodic patterns
  rather than merely pricing them.

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
