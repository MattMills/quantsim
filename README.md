# quantsim

An in-house research quantum simulator in Rust, built so the interesting
parts are swappable:

- **Amplitude algebra** — every gate matrix, state and measurement rule is
  generic over a [`Scalar`](src/scalar/mod.rs) trait. Shipped: ℝ, ℂ
  (default), a generic **Cayley–Dickson doubling** `CD<T>` giving
  quaternions ℍ, octonions 𝕆 and sedenions 𝕊, **split-complex** as a
  first non-Cayley–Dickson algebra,
  **[`SplitQuaternion`](src/scalar/split_quaternion.rs)** — the
  coquaternions held as an *inclusion/exclusion pair* (below) —
  **[`Polarity<N>`](src/scalar/polarity.rs)** — `N` twisted polarities
  in one amplitude, generalizing that pair — and
  **[`Ball`](src/scalar/ball.rs)** —
  coarse-grained certified arithmetic (complex midpoint ± certified
  radius; midpoints track `C64` bit-for-bit, radii propagate soundly, and
  `quantize` is a deliberate resolution dial). Planned (see
  [ROADMAP](ROADMAP.md)): truncated p-adics, dual numbers,
  Clifford scalars.
- **State representation** — a [`Backend<S>`](src/backend/mod.rs) trait with
  eight shipped implementations: **dense** state vector (the BQP reference),
  **sparse** hash-map state, an **adaptive** backend that promotes sparse →
  dense at ¼ density, the **factored** backend (product of dense factors
  over qubit regions — memory tracks entanglement *clusters*), **MPS**
  (matrix product states on a dependency-free Jacobi SVD — memory tracks
  Schmidt rank / *bond dimension*), **mera** (a hierarchical
  isometry tree — memory tracks *renormalization structure*; see below),
  **bulk** (the hierarchy in maintained isometric gauge with an
  explicit bulk record — and **dynamic width**: boundary qubits grown
  and released at runtime; see below), and **mosaic** (regions in
  heterogeneous representations with measured merges and
  refusal-driven migration; see below).
  Four orthogonal compression axes — support, clusters, bonds,
  hierarchy — all conformance-verified against dense, with the bulk
  register adding dynamic width and a maintained gauge on the
  hierarchy axis.
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
- **Dynamically scaled register** ([`BulkState`](src/backend/bulk.rs)) —
  the hierarchy taken to its projective form: every node tensor kept
  **isometric**, every scalar of state weight in one explicit **bulk
  record** (`top`), so the state is literally stored *across depth* as
  a stack of isometries under the record — `‖ψ‖² = ‖top‖²`
  identically, an `O(χ)` read at width 40. The register **scales at
  runtime**: `grow` appends fresh boundary qubits in amortized `O(1)`
  (within capacity it is bookkeeping; past it the register re-roots —
  the whole tree becomes the left *site* of a register twice its size,
  the `recursive` module's point-is-a-lattice move applied to the
  representation itself; a live 32-wide GHZ is grown from width 2 with
  4 re-roots, exactly), and `release` detaches boundary qubits again —
  **refusing with the measured leakage** (`5.000e-1` for a GHZ member)
  unless the qubit is verifiably `|0⟩`; `release_measured` measures
  first, so streaming works: 48 logical qubits pass through a register
  whose peak width is 2, outcomes exactly GHZ-correlated. Truncation
  is **environment-weighted**: the record's Gram factor is transported
  down before every re-compression, so discarded weight is *global*
  Schmidt weight, ledgered **per depth** with a certified bound
  `‖ψ_ideal − ψ_stored‖ ≤ Σ√ε` (asserted against dense across seeds;
  tight — equality — for single-event runs; 6× less error than
  block-local truncation on a pinned seed, no dominance claimed: the
  orderings can trade places). And the depth store **unfolds**:
  `unfold_program` compiles the register into a seed plus one dilated
  unitary per node — replayed on dense it reproduces the state to
  `3e-17`, stopped after `ℓ` levels it *is* `coarse_state(ℓ)` on the
  channel wires — with each wire's `sequence` naming the only
  interactions it ever has: entanglement between tree-aligned regions
  rides nameable channel wires with measured Schmidt rank ≤ bond.
  Dimension added above the problem separates its entanglement into
  structured, addressable channels — the same up-a-dimension move
  `lift` and `upembed` make for magic, made for entanglement.
- **The time system inside the register** ([`clock`](src/clock.rs)) —
  a **selector qudit** whose entanglement with the system is mandatory
  and structural, with scale as its axis. [`BranchedRegister`] holds a
  `d`-level selector over weighted branches that may each live in a
  **different representation** (one sparse, one factored, one MPS, one
  graph-state bundle); branch states are shared, selector rotations
  are bookkeeping (never copies, growth ledgered), interference is
  evaluated lazily, and everything selector-side — Born statistics,
  conditioning, the selector↔system Schmidt rank — is computed
  polynomially from the pairwise Gram of unique branches, never by
  enumerating the joint register. `scale_history` aligns the clock
  with recursion: the Page–Wootters register `Σ_ℓ w_ℓ|ℓ⟩⊗|ψ at scale
  ℓ⟩` over the bulk register's own depth levels, each slice an
  `O(tree)` **`scale_snapshot`** (node surgery, no gate applied —
  width-32 six-scale histories in kilobytes); conditioning the clock
  on `|ℓ⟩` **is** the scale-ℓ view (measured to 5e-16), `tick` is the
  clock-controlled refinement `|ℓ⟩⟨ℓ|⊗V_ℓ` (one unit of internal time
  = one level of coarse-to-fine information flow), and interfering the
  clock turns inter-scale overlap into Born statistics — **scale
  interferometry**: the structured test state moves by exactly `1/√2`
  per RG step, `|0…0⟩` by `1.0` (nothing to refine, full visibility).
  The payoff, measured: four width-16 slices — GHZ, rainbow, dense
  brickwork, random long-range graph state — each hostile to the
  others' representations, held **additively** in one flagged register
  at 3,845 B, while every single-representation alternative pays
  68–850× on its clashing slice (ghz-on-factored 1.0 MB,
  brickwork-on-sparse 3.3 MB, graph-on-factored 262 KB, dense 1.0 MB);
  the contracted view (selector traced against the weights) is a
  `Backend`, full-basis-conformant against the dense sum. New qudit
  dimension **above** the problem, lower representational structure
  below it. Honest boundaries, measured: enumeration-based trait
  methods cost the union support; `tick` refuses branch states shared
  across scales; and a superposition makes branch *global* phases
  relative — a representation maintained only up to global phase (the
  bundle's documented convention) is sound as a static slice but
  corrupts the contracted sum under broadcast dynamics, pinned in the
  tests as the phase-faithfulness contract.
- **The multi-representation register** ([`MosaicState`](src/backend/mosaic.rs)) —
  every representation is a *factorization lens* (sparse ↦ support,
  factored ↦ spatial products, mps ↦ linear cut rank, mera/bulk ↦
  hierarchical cut rank, bundle ↦ stabilizer structure, the frame ↦
  magic count, phase-field ↦ diagonal polynomials, branched ↦ class
  rank); the mosaic assigns every **portion** of the register to the
  lens that fits it, and lets the assignment follow the circuit.
  Regions in heterogeneous representations; gates inside a region run
  natively; gates across regions merge with the representation
  **chosen by measurement** (predicted sparse vs dense cost, both
  ledgered so mispredictions are data); migration is **refusal-driven
  and partial before total** — a bundle region hit by its first T
  first fractures along its own graph components (exact: a graph state
  is the product of its components — the graph, and the graph of
  graphs), so only the touched component leaves and the rest stay
  graphs, then the affected region converts down the policy list and
  retries, every split/merge/migration ledgered with its cause.
  Regions may themselves be mosaics (recursive composition, tested),
  and the flagship measures the point: a width-40 register whose left
  half is a 20-qubit Clifford expander (bundle — outside every
  bond/cluster/support bet) and whose right half lives through a
  Clifford→T→entangling era sequence stays at the **sum of ideal
  costs** (< 64 KiB) while the era transitions play out as ledgered
  representation changes — dense-verified at width 16. Phase contract,
  the dual of the clock module's: product composition forgives
  phase-loose tiles (a region's global phase stays global), while
  superposition composition does not — measured on both sides.
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
  plus linear/ring/grid/custom) and causally organized research fabrics
  (`hierarchical`, `hypercube`), SWAP-routing that walks entanglement
  stepwise through adjacency with a persistent logical→physical mapping,
  per-qubit latency clocks driven by a [`LatencyMap`] (era-representative
  `DurationModel` presets + per-site/per-edge calibration overrides), an
  injectable inner representation (chip-scale geometry over a sparse
  inner: GHZ across all 54 Sycamore sites), a full physical op log, and
  `elapsed`/`serial_time` as the measured parallelism ratio — while
  answering in logical indices identical to dense.
- **The boundary atlas** ([`bounds`](src/bounds.rs)) — every
  representation read as an *assumption* about structure, with its cost
  measured to be exponential exactly in its own resource (sparse ↦
  support, factored ↦ cluster size, mps ↦ cut rank, mera ↦ tree rank,
  clifford-framed ↦ T-count, Ball ↦ certified precision) and
  sub-exponential otherwise. `resource_profile` runs one circuit on
  every representation under the guard (walls report as measured
  refusals or deadline aborts, never skips); `advantage_scan` sweeps a
  circuit family, classifies every axis's measured growth law
  (`classify_law`: constant / polynomial / exponential from bytes
  alone), and returns the verdict — **classical** the moment any
  assumption holds, **advantage candidate** only when every measured
  axis grows exponentially at once, exactly. The scan rediscovers the
  known results from measurement (GHZ via support; basis-input QFT via
  bond; rainbow via clustering at measured `√2^n` sparse cost;
  Clifford circuits via the frame at `size^2` — Gottesman–Knill; random
  universal circuits escape everything at bases 1.7–2.0) and doubles as
  the *discovery instrument*: a new registered representation whose
  axis stays flat — while exact — on the candidate family is a found
  sub-exponential simulation. An axis certifies only when **both** its
  memory and its wall-clock law stay sub-exponential (measured case:
  MPS on long-range IQP is time-polynomial but memory-exponential).
  The **assumption dials** are measured families where one structural
  knob flips the verdict: the same IQP core is a candidate with
  long-range couplings and classical nearest-neighbour; random
  T-doping costs the frame *nothing* even at `t = n/2` (the 2^t escape
  needs deliberately scattered magic); depth-3 2D brickwork reads
  `size^2` — the boundary law failing slowly. `select_by_scaling`
  chooses a backend by *extrapolating* the fitted laws to a target
  size (holdout-verified: GHZ at width 40 predicted 125 B, measured
  125 B), and says plainly when no assumption holds and the choice is
  only least-bad.
- **Sampling-task hardness** ([`sampling`](src/sampling.rs)) — the
  advantage claims are about *sampling*, and the harness measures that
  task directly: linear XEB scored against the dense reference or the
  exact D[ω] ring (no float in the reference path — same samples agree
  to 1e-15), normalized by the *measured* ceiling `2^n Σp² − 1` (GHZ's
  is `2^{n−1}−1`; a uniform output has ceiling exactly 0 and the score
  is honestly `None`). The atlas verdicts carry over to the task:
  certified-easy families sample at certified cost (GHZ at the ceiling
  from 125 B; `clifford_sample` measures per-shot tableau sampling —
  Gottesman–Knill for the task, 256 shots at width 20 in ~23 ms). The
  candidate family's hardness is measured as spoofing economics:
  `mps_spoof_curve` shows every truncated bond cap collapsing below
  0.3 normalized XEB with only full rank (`χ = 2^{n/2}`) reaching the
  ceiling — a cliff, not a slope — and `spoof_decay` shows a fixed-χ
  budget decaying to noise as the family grows: the sampling task
  inherits the state bounds, as a reproducible measured artifact.
- **Mixed-arity compound qudits** ([`mixed`](src/mixed.rs) +
  [`e8`](src/e8.rs)) — the representation/interaction/flow separation
  made explicit for *non-binary* registers. Sites of any arity
  (binary/ternary/quaternary/quintary…) are held as **horizontal
  volumes** — independent until an interaction genuinely correlates
  them (lazy, guard-admitted merges; a refused merge changes nothing).
  Generalized gates with tested relations (`fourier_d` with
  `F X F† = Z`, the Weyl pair `Z X = ωXZ`, the any-arity entangler
  `cshift` — a binary–quintary Bell pair samples exactly two
  outcomes), conformance against the qubit reference where dims
  coincide (all-binary replays exactly; one quaternary site *is* the
  2-qubit QFT). The interaction fabric is measured ([`fabric`]): bond
  density, and the structural fact that **swap exists only between
  equal arities** — mixed fabrics decompose into swap classes with
  cross-arity bonds forced native. Flow is recorded, not inferred:
  merge timelines and order-respecting interaction cones. The E8
  anchor is *constructed and verified programmatically* (240 roots =
  112 + 128, norms, closure): the su(2)/su(3)/su(4)/su(5) chains of
  the four arities are found by search, the `su(5)×su(5)` orthogonal
  pair is exhibited, and the rank obstruction is **measured** — after
  A1⊥A2⊥A3 the orthogonal-A4 search exhausts (rank 10 > 8), so the
  four frames *must* share directions: the canonical embedding couples
  exactly the quaternary–quintary pair (doubled Gram overlap 44), and
  the compound register runs over precisely that fabric. The **E8×E8
  co-boundary system** builds on top: two 240-level copies, diagonally
  paired, store a data field as the *relative* rays across paired
  points — projective (a global phase is physically nothing, capacity
  ℂP²³⁹ per layer), provably invisible to either copy alone (marginals
  uniform to 1.7e-18, measured), and recovered exactly (1.9e-15) only
  through cross-copy interference. The complex's own cochain structure
  is measured too: 6720 −1-edges each closing into one zero-sum
  triangle (2240 total — the 2-cells are the additive relations
  α+β+γ = 0), and GF(2) Betti number **b₁ = 4241**: the invariant
  edge-storage the complex carries beyond anything derivable from
  points. Wide-qudit fast paths (O(d) diagonal, O(d²)-validated
  permutation gates) make the 240-level protocol run in milliseconds.
  And E8×E8 is an **8-qubit representation** outright (`e8::rep`): the
  128 even-parity basis states *are* the spinor roots, the odd sector
  pairs onto the second copy (128+128 = 256), and two basis states at
  Hamming distance 2 differ by exactly an **integer root** — the 112
  integer roots are the 2-local transition labels, verified against
  real gate matrices over every pair and basis state. Entanglement
  reads as measured root geometry: GHZ is exactly one antipodal pair
  (no 2-local root connects its ends), products sit at affine
  dimension 0, random states flood the root graph (3584 edges), and
  Schmidt rank obeys the geometric projected-support bound on every
  state and cut tested. **Both systems are first-class backends**: the
  compound substrate as `CompoundBackend` (`"compound-binary"`, with a
  general k-site `apply_k` and digit collapse) and the root-keyed
  store as `e8::rep::E8RepState` (`"e8-rep"`, amplitudes keyed by
  (copy, spinor root), 8-qubit-native, refusing wider widths with the
  structural reason) — both swept through `verify_backend` over the
  full registry and priced in `compare_backends` next to
  dense/sparse/mps/mera, with the d = 16 co-boundary protocol itself
  as a workload every representation must reproduce. And the
  representation expands to **any width** via the infinite E8
  constellation (`e8::constellation`): the measured coset theorem
  **E8/2E8 ≅ F₂⁸** — exactly 256 classes: 1 origin + 120 antipodal
  root pairs (√2-sphere) + 135 sixteen-frames (2-sphere), verified by
  exact integer arithmetic — makes one byte the identity position of
  an E8 on the shells of its parent at doubled scale. The scale tower
  `Σ 2ᵏ·rep(digitₖ)` is a measured bijection onto `E8/2^m E8`,
  self-similar under doubling, so an n-qubit basis state *is* one
  lattice point at resolution `2^⌈n/8⌉`; `E8ConstellationState`
  (`"e8-constellation"`) keys amplitudes by those points at any width
  to the u64 wall — a 63-qubit GHZ is two 80-byte lattice points
  (dense refuses 63 outright, measured), 421× faster and 4681×
  smaller than dense on GHZ-16, honestly larger and slower than dense
  on saturated QFT-12. On top of the tower sits the **cross-scale E8
  Weyl pair**: a position-E8 of native `translate` gates and a
  momentum-E8 of native `modulate` gates — the *same* lattice, because
  E8 is self-dual (det(Gram) = 1 verified; every `dual_basis` vector
  an E8 point; coordinates = dual inner products, measured) — obeying
  the measured Heisenberg law `M_q T_v = e^{2πi⟨q,v⟩/2^m} T_v M_q`
  with its 2-adic ladder (scales interact below the resolution
  horizon, commute *exactly* past it), with W(E8) reflections as
  measured Clifford symmetries (`s² = 1`, `sT_vs = T_{s(v)}`,
  `sM_qs = M_{s(q)}`), `coordinate_fourier` converting one object into
  the other (`F⁴ = 1`, `FT_BF⁻¹ = M_{−b*}`), exact support uncertainty
  (rank-k coset states: `|pos|·|mom| = 2^{8m}` on the nose), depth-1
  reduction to ordinary X-strings and sign diagonals (checked against
  the standard framework), and structured coset states interacting —
  interference, cross-scale translation, dual modulation, W(E8) —
  at 40 qubits in ~1 ms where dense's 17.6 TB is a measured refusal:
  rank-k structure costs `2^{km}` points instead of `2^{8m}`. And on
  top of the pair sits **multi-scale dual time**: the distilled scale
  operad (`scale_embed`/`decimate` isometries composing exactly, the
  Weyl pair transporting covariantly across them, decimation of live
  fine-scale data refusing with the level named), **cross-scale comb
  codes** whose coarse-translation + fine-modulation checks all
  commute past the horizon, whose logical operators live at the middle
  scales, whose syndrome windows tile *exactly* the same bidirectional
  inequality as the commutation ladder (the m×m FIRE matrix, measured),
  with end-to-end correction — inject fine displacements, decode them
  exactly from cross-scale syndrome phases alone, correct, decimate to
  the pristine coarse state — and code self-similarity (decimating the
  (m=4,a=2) code IS the (m=3,a=1) code); plus the **folding
  theorems**: an interleaved bidirectional sequence (ascending
  translations against descending modulations) reorders through
  beyond-horizon commutations into exactly one two-gate layer (the
  below-horizon control measurably refuses), and deep periodic time
  folds to its measured period — 2⁴⁰+5 blocks evaluated as 5 in
  ~100 µs, certified by the exact small-t Weyl closed form
  `U^t = χ^{t(t−1)/2}·T_{tV}M_{tQ}` plus modular arithmetic. The codes
  are a first-class object (`CombCode`: codeword, checks, logicals,
  binary-readout syndrome decoding, min-norm correction) with the
  **measured logical-vs-physical error curves**: under seeded
  displacement noise with per-round correction, the degenerate mod-2
  window saturates at any rate while each added comb level widens the
  correctable cell — a=4 at 0% logical failure where a=3 fails 61%
  (p=0.05), the threshold-shaped suppression measured across five
  rates with every trajectory priced at 256 lattice points.
- **Causal geometry** ([`causal`](src/causal.rs)) — the causality between
  register elements as an operational object: backward light cones and
  **causal diamonds** (prune a circuit to the cone of an observation
  surface — provably identical marginals, measurably fewer ops), and
  **dual-time resolution** (`dual_time_amplitude`): the preparation
  boundary evolves forward, the observation boundary evolves backward,
  and the two opposed directions resolve at a cut —
  `⟨t|U|0⟩ = ⟨U₂†t|U₁0⟩` — each paying only its own cone's support
  (measured: `2^{D/2}` a side at the balanced cut where one direction
  pays `2^D`). On the register side, [`Topology::hierarchical`] and
  [`Topology::hypercube`] build fabrics *organized by causal scale*
  (logarithmic horizons), and `diameter`/`ball_sizes`/
  `pair_availability` measure any fabric's causal metric, curvature
  signature and interaction availability.
- **The inclusion/exclusion pair**
  ([`SplitQuaternion`](src/scalar/split_quaternion.rs)) — an amplitude
  that carries a **constructive and a destructive complex component**,
  tracked independently, with measurement reading their difference. The
  split quaternions `q = z₊ + z₋·j` (`i² = −1`, `j² = +1`) are graded by
  `j`, and the product *is* the inclusion–exclusion rule: two exclusions
  make an inclusion, one of each makes an exclusion. The crate's two
  magnitudes finally separate usefully — `born_weight = |z₊|² − |z₋|²`
  is the **net**, `abs_sqr = |z₊|² + |z₋|²` is the **path weight**.
  Because ℂ embeds, this is the first non-division algebra here to carry
  the **full 39-name standard registry** (split-complex, lacking `i`,
  gets the real subset only), so the conformance suite applies
  unmodified: sparse and adaptive pass at deviation **0.0e0**, factored
  at 1.4e-16, while mps/mera/clifford-frame refuse by named structural
  reason. The first measured consequence is that an embedded-ℂ matrix
  entry acts as `m·(z₊, z₋) = (m z₊, m z₋)`: **a standard circuit drives
  both channels with the same matrix and never mixes them**, so one
  split-quaternion run carrying ψ in inclusion and φ in exclusion is
  *bit-for-bit* two ordinary complex runs — measured exactly equal, with
  the per-state net equal to `p_ψ − p_φ` and the totals to
  `A−B` and `A+B`. Unitary gates conserve the net exactly even when it
  is negative (measured drift 2.2e-16 across every registry gate on a
  net −0.28 state). Weight crosses between the ledgers through the
  non-compact part of the group: the unit-norm elements are `SL(2,ℝ)`
  (split quaternions are `M₂(ℝ)`, the norm is the determinant), so
  `SplitQuaternion::boost(t) = cosh t + sinh t·j` is a genuinely unitary
  gate (deviation ~1e-16) that **pumps path weight as `cosh 2t` while
  conserving the net exactly** — constructive and destructive amplitude
  created in matched pairs at zero net cost, interference as a gate
  rather than a ledger. The exchange `j` itself has `N(j) = −1`: it
  swaps the ledgers and negates the net, measures a unitarity deviation
  of exactly **2**, and the registry refuses it — turning inclusion into
  exclusion outright is precisely a non-unitary act. On the null cone
  `|z₊| = |z₋|` the net is exactly 0 with the path weight positive, so
  **"everything cancelled" is decidable and distinct from "nothing was
  there"**, which a complex amplitude cannot express; measurement
  refuses such a state by the existing rule rather than inventing a
  distribution. The honest bound: the grading means *no* gate can evolve
  the two channels differently — the pair is a pair, not two independent
  registers. It also works as an algebra-sector qudit
  (`dual_algebra_report` measures the sandwich span full at 16/16,
  embedded action component-linear) and benchmarks next to every other
  backend.
- **Geometric closure, retrodictively** ([`closure`](src/closure.rs)) —
  a journalled state representation whose history carries its
  entanglement, and whose errors are found by loops failing to close.
  The log *is* the state: every legitimate change is an entry, the twist
  links are part of the record, any past configuration replays. What
  makes it work is a **closure stamp** — two bits per independent loop,
  nothing more. A geometry's loops are **checks nobody declared**:
  exactly `|E| − |V| + components` of them, measured
  (grid 6×6 → rank 25), and a tree has none, so a tree sees nothing and
  says so. Each loop reports two channels: **spin closure** (the parity
  of the fiber signs it passes through) and **chirality closure** (the
  parity of the descents it makes against the current ordering).
  Breaking is *geometric*, not statistical — a perturbed site breaks
  **exactly** the loops through it and no others, asserted site by site
  across a 5×5 grid. Intersecting the broken against the intact names
  the site (grid 6×6: 24 of 36 named exactly, the rest reported
  *ambiguous* with the true site always among the candidates, and
  correction refusing rather than guessing). An ordering break names a
  **link** instead — measured as `(5,6)` after one unrecorded swap —
  and correction declines there too: the link is determined, the
  transposition is not. **Retrodiction** reads the history backwards:
  an unrepaired break persists, so the stamps are monotone and the
  first broken one is found by bisection — measured **5 comparisons
  against 22**, bracketing an injection at step 41 between "last held
  at 40" and "already broken by 45", naming site 14 uniquely in the
  same act. No state is replayed and nothing was ever recorded as an
  error. Correction then restores closure (8 broken loops → 0) *and*
  the denoted state, at deviation **0.00e0** against a directly
  simulated clean bundle. The cost of being able to retrodict a
  120×120 geometry is 28 322 bits — about 3.5 kB — a stamp; the honest
  counterweight, printed rather than omitted, is that *evaluating*
  closure costs the total loop length (1.7M at 14 400 sites), which a
  spanning-forest basis does not keep small.
- **The bundle is a backend, and an atlas axis** — `PolarityBundle`
  implements `Backend<S>` and is registered as **`"bundle"`** in
  `BackendRegistry::standard()`, so `sim.run_on("bundle", &circuit)`
  works like any other representation — and `resource_profile` /
  `advantage_scan` now carry a **`bundle`** axis beside support,
  clusters, bonds and T-count, with **`e8-constellation`** joining them
  through the new `BackendRegistry::<C64>::register_e8()` (the E8
  representations store complex amplitudes natively, so they cannot
  live in the generic `standard()` and are opt-in by that call). Adding
  them found a real defect immediately: `memory_bytes` was counting the
  *journal*, so the atlas classified the bundle's audit trail as
  **Exponential(1.40)** on GHZ. Reporting structure only — the journal
  is an optional addition whose size tracks gates applied, not state
  size — GHZ-14 falls from 67 088 to **1 552 bytes** and the law reads
  **Polynomial(1.50)**, putting `bundle` inside the atlas's
  `Classical { via: [...] }` verdict. The width sweep shows why neither
  representation dominates: on GHZ at 24 qubits, dense pays 256 MiB,
  sparse 125 B, bundle 4.0 KiB (23 links), the E8 tower 224 B (2
  lattice points); on an H-layer at 20 qubits, dense pays 16 MiB,
  sparse **50 MiB**, the E8 tower **80 MiB**, and the bundle **944 B
  with no links at all**. The bundle is the only one cheap on both, and
  on the random-universal candidate family it **declines outright**
  rather than truncating — which `tests/bounds_atlas.rs` now
  distinguishes from silent approximation explicitly. Gate action is on the graph
  itself: a single-qubit Clifford composes into the fiber's **vertex
  operator** (the full 24-element group, generated and verified — `H²=I`,
  `S⁴=I`, matrices consistent with the Pauli action at 4.4e-16), `cz`
  reduces both endpoints' operators into the diagonal subgroup by
  **local complementation** and toggles the link, `cx` and `swap`
  decompose into those. Local complementation is verified to change the
  *description* and not the state (link signature moves, state deviation
  **0.0** up to global phase). Measured against dense over 40 random
  30-gate Clifford circuits: **exact where it runs (deviation 0.0),
  39 of 40 ran**, and the one that did not **refused by name** rather
  than returning a wrong state — the vertex-operator reduction does not
  yet converge on every configuration, and that is a pinned test rather
  than a footnote. Non-Clifford gates refuse by name (`t` → "non-Clifford
  1-qubit gate ... graph-state bundle"), and `load` refuses because a
  description cannot hold arbitrary amplitudes. GHZ-20 runs at under 1%
  of dense's footprint.
- **The polarity co-bundle** ([`bundle`](src/bundle.rs)) —
  entanglement as an explicit, inspectable, budgeted resource instead of
  an implicit consequence of amplitude storage, and **the correction to
  the measurement below**. The polarity module measures that two-body
  *marginals* cannot distinguish `(|0…0⟩ ± |1…1⟩)/√2`. That stands, and
  it says nothing about a fibered representation, because a bundle does
  not store marginals. It stores a **base** — which sites are
  twist-linked — and a **fiber** over each site carrying that site's own
  polarity data, and **the GHZ sign lives in a fiber**: `ghz_bundle(n,
  ±)` produces bit-identical link signatures whose denoted states are
  *orthogonal* (measured overlap 0), with two-body marginals blind at
  deviation 0.0 exactly as before while `verify_against` reads the sign
  straight back off the state at deviation ~1e-16. The obstruction was
  an obstruction to marginals; it was never one to fibers.
  Structurally: sparse links (`O(n + |E|)`, never `2ⁿ`), a maintained
  **re-orderable** generator sequence where swapping two *linked* sites
  flips the bundle's **chirality** — the sign a product of anticommuting
  generators picks up, cross-checked against
  `PolaritySystem::product` rather than asserted — and an append-only
  **journal** where `rewind(k)` reconstructs any earlier configuration
  exactly. Entanglement is then a quantity with an owner: `profile`
  reports degree, independent clusters and bytes; `coarse_grain` merges
  fibers and reports every link removed on one of two lines (**absorbed**
  into a super-fiber, or **collapsed** parallel links);
  `coarse_grain_to_budget` drives the total under a cap and prices it
  (measured: 60 000 links → 29 601 → 9 918 → 1 518 → 0, with the
  surviving fibers still accounting for all 20 000 original sites).
  `commonality`/`interact` confine interaction to shared sites with
  *compatible* frames and leave everything else untouched. It runs where
  the amplitude picture does not exist: **100 000 sites in 30 ms and
  7 MB of structure**, a million in 320 ms — with the journal measured
  separately as the dominant term and `checkpoint()` the explicit way to
  decline paying for it. Scope, stated first: a bundle denotes a graph
  state dressed by local frames — a known classically-tractable sector,
  and nothing here moves that boundary. States that leave it are
  *reported* (`verify_against` deviation jumps from 1e-16 to >0.1 on a
  T-rotation) rather than silently approximated.
- **Polarity systems** ([`polarity`](src/polarity.rs),
  [`Polarity<N>`](src/scalar/polarity.rs)) — `n` inclusion/exclusion
  axes with a **twist**: generators `j₀ … j_{n−1}` and, for each pair, a
  choice of commuting or anticommuting. That is the twisted group
  algebra `ℝ^τ[F₂ⁿ]`, and its whole behaviour follows from the `F₂`
  **rank** `r` of the twist form — measured by brute force, not quoted:
  the centre has dimension `2^{n−r}`, a maximal pairwise-commuting set
  is a *subgroup* of size `2^{n−r/2}` (bilinearity, verified closed
  under XOR), and the algebra factors as
  `2^{n−r/2} × 2^{r/2} = 2ⁿ`. That factorization is the point: an
  abelian polarity system is simultaneously diagonalizable, so its state
  is `n − r/2` independent **sign bits** — genuinely local, `O(n)` — and
  everything outside costs `2^{r/2}`. **The twist rank is the measured
  price of non-locality**, and `PolaritySystem::partial` makes it a
  dial from untwisted (`2ⁿ` independent sectors, no correlation) to
  fully twisted (one matrix block, no local part).
  `PolaritySystem::pauli(n)` puts quantum mechanics on that dial: `2n`
  generators, measured rank `2n` — maximally twisted, centre trivial —
  with maximal isotropic subgroups of size `2ⁿ`. Those subgroups **are**
  stabilizer groups and their `n` sign bits are exactly the `O(n²)`
  description Gottesman–Knill runs on (shipped here as
  [`CliffordFramedState`](src/backend/clifford_frame.rs)), so the
  locally-storable sector is real and is already the best-known
  classical island. **Where it stops is measured too.**
  `pairwise_signature` builds the complete pairwise-local object — every
  one- and two-body Pauli expectation, i.e. all two-qubit reduced
  density matrices, `3n + 9·C(n,2)` reals — and `ghz_sign_obstruction`
  shows it is not enough, sharply: from `n = 3` up, the **orthogonal**
  states `(|0…0⟩ ± |1…1⟩)/√2` have signatures agreeing at deviation
  **exactly 0.0** with overlap 0, separated only by the `n`-body
  correlator `⟨X^{⊗n}⟩ = ±1`. At `n = 2` the same pair *is* separable by
  `⟨XX⟩`, so the measurement catches the obstruction switching on. And
  it is not a shortage of numbers: at `n = 3` the pairwise signature
  holds **36 reals against the state vector's 16** and is still blind —
  the failure is in what pairwise data can express. The sting is that
  both states are *stabilizer* states, so pairwise data fails inside the
  sector that is efficiently describable: the description is `O(n²)` in
  size but not pairwise in **structure**, because GHZ's stabilizer group
  needs a weight-`n` generator (`stabilizer_weight_profile` measures
  `[n, 2, 2, …]`). As an amplitude type, `Polarity<N>` carries the
  grading directly — `abs_sqr = Σ c_g²` is the path weight,
  `born_weight = Σ (−1)^{deg g} c_g²` the net, falling out of
  `q·conj(q)` rather than imposed — with `Polarity<1>` the
  split-complex numbers, `Polarity<2>` **exactly** the split
  quaternions (isomorphism transported entrywise at deviation 0.0), and
  ℂ embedding from two polarities up so the full registry exists and the
  conformance suite passes over `Pol3`. The cost is stated first: `N`
  polarities is `2^N` reals **per amplitude**, so one polarity per qubit
  costs `2ⁿ` per stored amplitude — the exponential does not disappear,
  it moves from the register into the scalar.
- **Recursive systems** ([`recursive`](src/recursive.rs), [`e8::cube`](src/e8.rs))
  — a site that is either a **point or an entire lattice of the same
  kind**, and a computation that participates in itself. Four qubits
  bonded in a square; `refine` turns any point into a lattice and
  `nest` does it uniformly, so a square of four points becomes a square
  of four squares — 16 qubits, four internal squares, four **lateral**
  bonds joining the blocks corner to corner by the same scale-free rule
  that joins their points (measured: depth 3 is 64 qubits, 84 bonds,
  horizon 16; the fabric is a `Topology` and runs on `DeviceState` like
  any other geometry). Whether the substitution *means* anything is
  then measured three ways, and they disagree informatively.
  **Ising**: `block_rg` diagonalizes a block and reads its effective
  description off its own spectrum — `h' = Δ/2` from the measured gap,
  `J' = J·μ²` from the measured boundary element — and `rg_fixed_point`
  finds where the flow stands still (Chain(2): `g* = 0.783243`,
  λ = 1.596, ν = 1.482 against the exactly known ν = 1, so the size of
  the Kadanoff approximation is *reported*, not hidden; residual
  4e-15). `substitution_report` puts it on trial against two blocks
  bonded laterally and finds the duality's **resolution horizon**: the
  spectra track to ~1–7% below the block's internal gap and the coarse
  description measurably invents a level above it. The ordered flow
  terminates with the reason named (the block gap falls below double
  precision — a refusal, not a silent continuation).
  **Phonons**: the duality is *exact* where it should be. A harmonic
  block's collective mode is the exact zero mode of its internal
  springs, so its frequency shift is `0.00e0` measured, and
  `phonon_substitution` separates the regimes — uniform bonding leaves
  the collective subspace exactly invariant (deviation 5.6e-16), port
  bonding is second order in the lateral spring (2.07e-2 → 2.73e-4 →
  2.80e-6 for 0.3 → 0.03 → 0.003). **On the simulator**: `phonon_walk`
  loads one phonon into a block's collective coordinate on a real
  `Backend`, Trotters the spring Laplacian through the lattice's own
  bonds, and separates Trotter error (→ 0 with the step) from
  substitution error (flat in the step, quadratic in the lateral
  spring) — with excitation-number leakage measured at `0.00e0` rather
  than assumed. And `self_participation` closes the loop: a block
  solved in the mean field its own boundary magnetization produces —
  read back off a loaded backend through `pauli_expectation` — so the
  output of the computation *is* its input. It converges to a measured
  fixed point at any recursion depth (the block may itself be a lattice
  of lattices), and `participation_transition` bisects the coupling
  where a self-feeding computation stops answering zero (`Jc =
  0.267675`; m = 0.41 at 1.1×Jc, 3.8e-10 at 0.9×).
  **E8 as a 2×2×2 cube volume** ([`e8::cube`](src/e8.rs)): the
  lattice's eight *orthogonal* ambient coordinates are the eight
  vertices of a cube, and three things fall out measured. The lattice
  condition becomes a **parity law on the volume** — of the 240 roots,
  112 live wholly on the cube and 128 on the body-centred copy, every
  one with vertex values summing to a multiple of four, and a
  mixed-parity volume is in neither sector. The cube's symmetry group
  sits **inside `W(E8)`**: exactly 48 of the 40320 coordinate
  permutations preserve the twelve edges (`Z₂³ ⋊ S₃`, brute-forced),
  all of them verified E8 automorphisms against all 240 roots, and they
  act on states through `permute_coordinates` — while a `GL(3,2)` shear
  is a lattice automorphism that is measurably *not* a cube symmetry.
  And the scale tower **already was** a cube of cubes: a digit is a
  byte, one bit per vertex, so `Σ 2ᵏ·rep(dₖ)` is a stack of volumes
  each of whose vertices resolves into another volume. On that
  structure, `inward` displaces a volume's own interior and `outward`
  its siblings, and `interaction` measures their commutator **off two
  evolved states**, cross-checked against the exact integer pairing:
  different cube vertices commute *exactly* at every scale (eight
  independent channels), while along one vertex an inward displacement
  at level `j` and an outward modulation at level `k` interact iff
  `j + k < m − 2` — with phase `exp(−2πi·2^{j+k+2−m})`, so it is
  exactly −1 on the horizon, a quarter turn one level inside, an
  eighth turn one further. **Deeper participation is finer
  participation**, and `self_reference_depth` reports the finite depth
  to which a volume can reach itself (none at m = 2, then 0, 1, 2 as
  the tower grows) — the recursion is self-referential but not
  infinitely so, and the horizon is a lattice fact rather than a
  truncation. `Shape::CUBE` closes the circle: its bonds *are*
  `e8::cube::edges()`, so the tower's cube of cubes and the recursive
  lattice's are one object reached from two directions.
- **Hierarchical algebraic registers** ([`qudit`](src/qudit.rs)) — break
  the flat n-wide register into a varied qudit structure: a **site
  sector** (any backend) plus an **algebra sector** — further logical
  qubits carried *inside every stored scalar*, using the Cayley–Dickson
  tower as the qudit space (ℍ = 1 qubit, 𝕆 = 2, 𝕊 = 3 per scalar).
  Gates on algebra qubits are synthesized from the **dual-algebra** —
  the algebra acting on itself from left and right (`A ⊗ A^op`) — and
  the sandwich span is *measured*: full operator-space rank at every
  doubling level (16/16, 64/64, 256/256, and 1024/1024 at the fifth
  doubling `Trigintaduonion`; residuals ~1e-15), zero divisors and
  non-associativity notwithstanding, so every ℂ-linear qudit gate is
  exactly a sum of `(a·x)·b` terms — and the synthesis **executes**:
  algebra-sector gates run as actual two-sided multiplications on the
  stored scalars (cached, counted, dust-snapped, with the exact
  component path as fallback and A/B toggle). The boundary is
  measurable too: a `DirectSum` scalar multiplies blockwise, so its
  span is exactly the block-diagonals (32/64 on ℍ⊕ℍ, cross-block
  residual ~1) — such gates route through the component path, costing
  routing, never correctness. Whether the embedded-ℂ action is
  component-linear is measured per scalar (true for ℍ and diagonal
  direct sums, twisted from 𝕆 on) and routes site gates native vs
  component. Conformance-swept over the full registry (including the
  ℍ⊕ℍ register); and the flat u64 indexing ceiling breaks: 66–67
  exact logical qubits (63 sparse sites × 𝕊 or `CD⟨𝕊⟩` qudits)
  addressed and *Born-sampled* as (site, component) parts.
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
  peak RSS recorded — across every representation, the new ones
  included: algebraic/compound/constellation memory fills next to
  dense/sparse/exact/factored/mera, plus the structural ceilings
  (compound and constellation at the 63-qubit packed-index wall,
  e8-rep at its native 8).

BQP support: the standard registry contains a universal set (`h`, `t`, `cx`,
…), so any BQP circuit family runs exactly on the dense backend — at the
unavoidable `O(2^n)` memory cost in width `n`, which the width benchmarks
measure rather than hide.

- **Braided boundary encoding** ([`braided`](src/braided.rs)) — the
  register as `n` mutually encoding boundaries, navigated by a periodic
  word, with the path, the address and the stored state as one object.
  The **mutual encoding** is exact and measured in both directions:
  `ρ_A` and `ρ_B` are built from different-sized matrices and their
  spectra agree to 0.0e0 on GHZ and 2.2e-16 on a random state, and `ρ_A`
  **alone** purifies to a state whose complement is spectrally the true
  `B` — the holographic claim in its checkable form. The storage claim
  is reported rather than assumed, and it fails honestly: GHZ across
  4|4 is rank 2 and stores 66 numbers against 256, while a random state
  is rank 16 and stores **528 against 256** — boundary encoding is
  dimension-reducing exactly when the state is, which is the same
  condition the boundary atlas prices every other representation
  against.
  What is new is the **navigation**. The mutual encoding written as a
  substitution *is* the Artin action `σ_i: x_i ↦ x_i x_{i+1} x_i^{-1}`,
  `x_{i+1} ↦ x_i` — one boundary conjugated by the other — and Artin's
  theorem makes it faithful, so `BraidWord::equals` decides path identity
  exactly (`σ1σ2σ1σ2⁻σ1⁻σ2⁻`, freely irreducible at length 6, is
  recognized as the identity; the commutator `σ1σ2σ1⁻σ2⁻` is not).
  `cayley_ball` measures the infinite graph the path navigates by
  deduplicating words into group elements (`B_3`: 1, 5, 17, 47, 115,
  263, 577, the ratio falling from 5.00 to 2.19 as the relations bite),
  and `PeriodicPath` turns a finite period into an infinite path whose
  orbit types are **necklaces** and whose primitive orbits are **Lyndon
  words** — Burnside and Möbius closed forms, verified against
  enumeration, with Duval's factorization tested for both properties it
  claims. That factorization also answers what happens **between** two
  orbits, and the answer is sharper than "a boundary layer":
  `orbit_junction` measures two regimes with nothing in between. When the
  first orbit's last factor is `≥` the second's first factor the two
  factorizations already concatenate legally and the join costs exactly
  nothing; otherwise the smaller tail absorbs the larger head and can
  swallow everything past the seam (`[1,1,0]` then `[0,1]`: 13 of 24
  letters, and the second orbit keeps zero factors). A junction is free
  or total, never partial; it is a property of the *ordered* pair; and it
  is settled by comparing words, with nothing about the geometry
  entering. It is not even stable under relabelling — the same orbit
  joined to itself is free exactly when its period is written in its
  canonical rotation, and free at 18 letters but merged at 20.
  The bracket is **measured, not posited**: the group commutator agrees
  with `exp(ε²[A,B])` at a fitted order of 3.02 and third-order BCH at
  3.97, and the fit is sharp enough to catch a degeneracy rather than
  paper over it — for the `su(2)` pair `iX, iZ` the order-4 term
  `[B,[A,[A,B]]]` vanishes identically and the exponent comes out 5.01.
  `majorana_bilinears` gives the infinitesimal folds `γ_iγ_{i+1}` whose
  exponentials are the braid generators (verified to 1e-12), and
  `realized_rank` measures the collapse: on 6 strands the free Lie
  algebra runs 5, 15, 55, 205, 829 through degrees 1…5 while the
  realization runs 5, 9, 12, 14, **15** and stops — the whole infinite
  path algebra lands inside `so(6)`.
  Whether the path can *be* the storage is then decided rather than
  hoped. Two realizations are built and their relations verified on the
  actual matrices (`≤ 2.2e-16`): the Majorana/Ising braiding, a genuine
  `B_n` representation at every even strand count, and the 2-dimensional
  Fibonacci `B_3` representation. The Ising image is **finite** and the
  closure is walked, not asserted — 4 strands close at 192 projective
  elements at radius 7, 6 strands at 23040 at radius 16 — so an
  arbitrarily long Ising path is stored in `O(1)`. The Fibonacci ball is
  still growing at ratio 1.88 when the search stops, which is evidence
  and not proof, and the report says exactly that. The same boundary the
  crate already measures from the other side: Ising braiding is Clifford
  and `CliffordFramedState` simulates it in `size²`; Fibonacci braiding
  is universal and nothing does.
  `RecursionLedger` closes it per path. Every path **ascends exactly**
  (3.5e-14): the descent erased nothing, because the recursion never
  flattened the geometry into symbols. Whether the storage is `O(1)` is
  a separate question and is answered by `projective_order` rather than
  by counting states — a tolerance-rounded count saturates for a dense
  orbit too. Measured in the Fibonacci realization: `01` closes at order
  3, `001` at 2, `0110` at 5, `01011` at 10, and `0111001` **does not
  close** within 10⁵ periods. Even in a universal realization many short
  orbits are finite; the property belongs to the path, not only to the
  group.

- **Both of the above as registered representations**
  ([`phase_field`](src/backend/phase_field.rs),
  [`braided_state`](src/backend/braided_state.rs)) — the two research
  modules are not only mathematics beside the library; each ships the
  `Backend` it implies, so both stand in the *same* conformance and cost
  harness as dense, sparse, MPS and the rest.
  **`phase-field`** holds the state as `scale · Σ_x ω_M^{P(x)}|x⟩` on an
  affine subcube, with `P` a multilinear polynomial over `ℤ/M` stored as
  a monomial table and the modulus grown by `lcm` as gates demand it —
  the `padic` radix doing its job inside a backend. Nothing is a float
  until an amplitude is read, so `verify_backend` over the whole registry
  returns `max_amplitude_deviation` of **exactly 0.0**, per gate and
  across 24 random circuits. On the atlas it is a **certified axis**: the
  IQP core (H layer, all-pairs `cz`, `t` layer) reads polynomial in
  *both* memory and wall-clock and the verdict names it —
  `Classical { via: ["mps", "phase-field"] }` — at 608→1976 bytes across
  widths 6→12 where dense runs 1056→65568. Its class boundary is exactly
  the physics: add the closing Hadamard layer and the axis goes
  exponential (base 1.96) and drops out of the verdict, because `h` on an
  already-free qubit is the interference step. An eighth-turn `rz` stays
  in class; `rz(0.371)` does not.
  **`braided`** holds the braid word as the address and a stabilizer
  frame as the state. The mechanism is one derivation: under
  Jordan–Wigner the Z-strings cancel inside a neighbouring Majorana pair,
  so `γ_2k γ_2k+1 = iZ_k` and `γ_2k+1 γ_2k+2 = iX_k X_k+1`, making every
  generator `σ_i = exp(iπ/4 · P)` for a Pauli `P` of weight **one or
  two** — verified against the dense generators at every width and basis
  state. A `±π/2` Pauli rotation is Clifford, so the frame absorbs it
  with zero amplitude work.
  It conforms at 8.0e-16, and on a braid family the atlas reads it
  **`mem Constant, time Constant`** and names it in the verdict —
  `Classical { via: ["mps", "braided", "clifford-framed"] }` — at 7877 →
  8453 bytes across widths 8 → 20 where dense runs 4128 → 16777248. Width
  8 → 63 costs 34837 → 38677 bytes, a 7.9× wider register for 11% more
  memory, with the stored support pinned at **1** and **zero flushes**.
  Two honest caveats. The footprint is *linear in the depth* — ~170 bytes
  per absorbed gate, which is the frame's replay log, not the state;
  `canonicalize` drops the word (sound, since the tableau already holds
  the group element it spelled) but the log is the frame's own. And
  `compare_backends` prices it *after* a flush, because verifying against
  dense extracts all `2^n` amplitudes — never a Gottesman–Knill
  capability. The time column is the honest gain there: 1963× faster than
  dense on braid-word-20.
  What this says about the representation is the point: the braided
  encoding **is** efficient, and its efficiency is the Clifford island
  reached from the braid-group side. The module measures the Majorana
  image to be finite (192 elements at 4 strands, 23040 at 6, both
  walked); the backend spends that finiteness. Fibonacci braiding is
  universal, its generators are not Clifford, and the frame cannot absorb
  them — the same boundary, stated twice.

- **Interference as a congruence** ([`padic`](src/padic.rs)) — a phase
  lives in `ℤ/M` where `M` is the modulus the source geometry supplies,
  and two structural facts carry a representation.
  **The field factors.** `ℤ/M ≅ ∏ ℤ/p_i^{n_i}`, so a phase over `ℤ/M` is
  a product of single-component phases and a wave's *entire* field is a
  product state on a mixed-arity `CompoundRegister`: at
  `M = 2^4·3^3·5^2·7 = 75600` the field is 75600 amplitudes held in 75,
  measured at 2.3e-15 against the direct sum, with the register's own
  volume count confirming no interaction ever coupled two components.
  `k` interfering waves are a rank-`k` object of size `k·Σ p_i^{n_i}`,
  not `M` — Good–Thomas read as a representation.
  **The interference character is a low-digit quantity.**
  `gcd(Δa, M) = ∏ p_i^{min(v_i, n_i)}` from the per-component p-adic
  valuations, so the *order of the phase as a root of unity* — 1 exactly
  in phase, 2 exactly antiphase and total destruction — is decided by
  **trailing** digits at `Σ_i (v_i + 1)` reads. Measured: `Σ p/(p−1)`,
  so the modulus grows a millionfold (`2^10·3^5 → 2^30·3^5`) and the
  cost does not move. That is why the p-adic direction is right here and
  backwards for magnitude: interference is a congruence condition, and
  congruences live at the fine end. `compare_diagonals` prices all four
  consumption orders on one population rather than arguing about them
  (coarsest-first 1.18 < interleaved 1.71 < shuffled 1.76 < sequential
  1.98, against a full depth of 10).
  **The sweep is priced against a closed form.** A lattice walk writes
  `Σ (1 − p^{−n})/(1 − 1/p)` digits per point *independently of depth*;
  a scrambled one writes `Σ n(1 − 1/p)`, linear in it. On a grid
  covering `ℤ/M` exactly: predicted 1.9995 / 6.0000, measured 1.9995 /
  5.9502 at `2^12`, and 1.4998 / 5.3333 predicted against 1.4998 /
  5.3126 at `3^8`. The measurement also corrects the obvious guess about
  *which* lattice order to take — every one lands on the same closed
  form, so it is the linearity that carries the claim, not the choice of
  line. The verdict cache is priced next to its hit count and does not
  pay; what does is `fringe_period`, the exact spacing `M/gcd(Δs, M)`
  read off the slope's trailing digits with the field never evaluated,
  confirmed against a brute-force scan.
  **The radix comes from the geometry**, in closed form, so nothing has
  to be computed first: `M = lcm` of the path-difference denominators.
  Commensurate slits are exact at `M = 24`; incommensurate geometry has
  no finite radix at all and pays a continued-fraction convergent, with
  the golden ratio — the worst-approximable number — paying most at
  every budget (denominators 8, 21, 89, 377, 1597, Hurwitz quantity
  pinned at 1/√5). The quasi-periodic pattern is the expensive one on
  exactly the axis that makes it quasi-periodic. Everything here is
  exact for phases that are rational multiples of `2π` — the same
  fragment `exact` and the qudit registers already live in — and
  irrational geometry enters only as a measured approximation with a
  reported error.

- **Cross-lateral distributed registers**
  ([`lateral`](src/lateral.rs)) — the logical network with the nodes on
  different machines. Couple nodes only **transversally** (qubit `i` to
  qubit `i`, the logical CX for a CSS code) and the Pauli frame
  conjugates by `x_B ^= x_A`, `z_A ^= z_B` and a sign popcount: two
  XORs of *aligned* bit-vectors, no term reaching a differently-indexed
  qubit — pinned against gate-by-gate conjugation at **0 disagreements
  in 200,000 random strings**, signs included. The syndrome is
  𝔽₂-**linear** in the frame (`bits(p ⊕ q) = bits(p) ⊕ bits(q)`, every
  pair, commuting or not), so a node that reads its own syndrome — an
  ancilla-free deterministic read — can name the syndrome of the delta
  that did *not* arrive. So the wire carries nothing but 𝔽₂: **27
  bytes** per node per tick, independent of `2^n`. Measured on two toric
  nodes, the distributed frame equals the gate-by-gate one and its
  signature equals the syndromes a dense register actually shows, bit
  for bit — 54 B across the wire against 1 MiB of amplitude.
  **And that makes a dropped packet a Pauli fault**: both are an
  unknown vector in 𝔽₂ recovered from `H·e = s`, except the network
  knows *where* its loss happened and the physics does not. Known
  locations turn error decoding into **erasure** decoding, worth
  exactly a factor of two — measured off the code's own generators, not
  read off a distance: erasure capacity `d − 1` against
  `⌊(d−1)/2⌋` (toric L=2 **1** vs 0, L=3 **2** vs 1, surface d=3 **2**
  vs 1, d=5 **4** vs 2), with the weight-`d` logical that ends it
  produced as a witness. At L = 2 the same weight-1 fault that
  [`retro::Decoder`](src/retro.rs) refuses by name is recovered
  exactly as an erasure. `DelayGeometry` measures the network's real
  metric (6 of 12 ordered pairs violate the triangle inequality on the
  example fabric; tightening drops the diameter 11 → 8) and its
  temporal diameter is the horizon floor; `Barrier` seals on **actual
  arrivals** — clean, repaired, or degraded naming the peers — never on
  the prediction that set `H`. And `DualLayer` crosses the **fine**
  layer (the stabilizer code, per node, across ticks, at *zero
  bandwidth*) against the **coarse** one (a Cauchy/GF(256) MDS code per
  tick, across nodes): on a pattern where coarse alone leaves 3 slots
  dark and fine alone leaves 4, the two crossed close all 24
  byte-exactly in 2 rounds — and the layer that finishes it is the code
  that was already protecting the qubits. Honest scope: the vocabulary
  is Clifford, and the loss and delays are injected rather than
  measured off a socket. This module distributes the **control plane**;
  sharding the register itself is [`stitch`](src/stitch.rs), below.

- **Sharding a register by its own symplectic form**
  ([`stitch`](src/stitch.rs)) — the phase-free Pauli group is an 𝔽₂
  space under an *alternating* form, so it has a symplectic normal
  form: `radical ⊥ H₁ ⊥ … ⊥ H_h`. The radical is isotropic — an
  abelian subgroup, which is to say a **stabilizer** — and each `Hᵢ` is
  a hyperbolic pair `(eᵢ, fᵢ)` that anticommutes. Two consequences,
  and together they are a sharding rule. **The radical carves**: `r`
  commuting directions cut the `2ⁿ` module to `2^{n−r}`. **The pairs
  cannot**, and that is exactly why they *partition*: `eᵢ` and `fᵢ`
  anticommute, so no abelian subgroup — and therefore no node — holds
  both, each pair forces a binary choice, and the `2^h` resulting
  maximal isotropic extensions have regions meeting in zero. Their sum
  is direct: `2^h · 2^{n−r−h} = 2^{n−r}`, exact.
  `orthogonalize` reads `r` and `h` **off the code's own generators**
  in `O(rank²·n)` with no `2ⁿ` anywhere — toric L=2 → `(r, h) = (6, 2)`,
  L=3 → `(16, 2)`, surface d=3 → `(8, 1)`, d=5 → `(24, 1)`: the radical
  *is* the stabilizer group and the pairs *are* the logical qubits,
  found rather than told. And the direct sum is checked on
  **amplitudes**, not on dimensions: the four toric slices are prepared
  as actual states, shown to lie in the code space, shown to be a frame
  rather than an orthogonal basis (Gram 1, 1/√2, 1/2 — the tensor of
  `|0⟩/|+⟩` per axis), and a random code state projected out of a random
  vector reassembles from them to < 1e-9.
  **Where the exponential goes** is the point. Per node: one tableau,
  polynomial in `n`, *not mentioning the node count* — 1024 nodes and
  32 nodes on the same register differ by 40 bytes each. Network total:
  `2^h · O(n²)`, exponential in the **logical** count, not the register
  width, because the code already pulled the exponent from `n` down to
  `k = n − r`. At `n = 30, r = 20, h = 10`: 1024 nodes × 256 B = 262 kB
  against 17.2 GB monolithic. The price, stated: each node carries a
  whole tableau to hold one coefficient, `O(n²)` bytes where a bare
  `2^k` amplitude vector holds 16 — bought in exchange for a per-node
  object that is polynomial, closed under Clifford evolution with no
  communication, and independent of every other node. And `h` is
  **hidden in the dimension**: two stitches of region dimension 1024
  can be 8 slices or 128, so the object's size says nothing about how
  many pieces it is in.

- **The residual as a mosaic** ([`pathsum`](src/pathsum.rs)) —
  [`PathSum`](src/pathsum.rs) holds a circuit's closed form and reduces
  it to `h*`, its irreducible path content, which is **zero for every
  Clifford circuit at any width** and grows with the T-count rather
  than the register. Readout then paid `2^{h*}` by enumeration.
  `amplitude_merged` declines that too, treating the residual the way
  [`MosaicState`](src/backend/mosaic.rs) treats a register — as a
  partition with a lens per part: **factor** (a disconnected
  interaction graph makes the sum a product), **merge** (two
  components that are the same polynomial up to renaming have the same
  sum, so the second costs a lookup — `distinct_forms` is what the
  route pays in place of `2^{h*}`), and **branch-and-reduce** (pin one
  variable and run the rewrite rules again; a residual that stalls
  frequently *unstalls* once one variable is fixed). Same number as
  `amplitude` — asserted, not approximated, at 1e-16 over 200 random
  circuits × every amplitude × every policy.
  What that measures: **thin magic is logarithmic**, `(HT)^k H` has
  `h* = k` and costs 19/29/39/51/**63** nodes at k = 8/16/32/64/**128**
  — `2^128` by enumeration. **Width alone is free**: at the same qubit
  count, T-count and depth, the 1D twin stays linear to 36 qubits and
  t = 72 while the 2D grid pulls away, ratio 1.0 → 19.0 as the side
  grows — the obstruction is *separator growth in the coupling graph*,
  not register size. And the surviving exponential is in **√t, not t**:
  on the square family (n qubits, n layers, `t = n²`), `log2(nodes)/n`
  settles at ~1.5 across n = 3..10, so 31,513 nodes stand against
  `2^90 ≈ 1.2e27`. A strict reduction of the growth law, and still a
  growth law — stated as such.
  The branching choice is an **election, measured**: connectivity is
  the whole game on a chain and *flat* on a shallow grid (removing any
  one variable disconnects nothing), degree is the reverse, and each
  single signal is 2–45× worse than the other somewhere. Composing
  them wins in every regime — the mosaic register's finding on a
  different axis, that the right lens is a property of the part rather
  than of the solver. Every policy returns the same amplitude; the
  budget refuses **by name** rather than quietly enumerating.

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

`cargo test` runs 975 tests (88 unit + 879 across eighty-two
integration suites + 8 doctests; one more — the 17 s measurement that
the fifth CD doubling keeps the dual-algebra span full — is `#[ignore]`d
and runs with `-- --ignored`);
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
- **bulk_register** — full-registry conformance for the dynamic
  register; the gauge invariant (norm as a record read, cross-checked
  against enumeration); `coarse_state(0)` **is** the bulk record;
  growth exact, disturbance-free, and amortized (4 re-roots for 30
  grows, pinned; grown = fresh-built against dense); release refused
  with the measured leakage and exact after uncompute; the 48-qubit
  stream through a width-2 register with GHZ-correlated outcomes and
  both stream values across seeds; the certified truncation bound
  `‖Δψ‖ ≤ Σ√ε` asserted against dense across seeds with the per-depth
  ledger summing to the total and `‖top‖` reflecting the loss; the
  pinned seed where environment weighting beats block-local truncation
  by > 4×; the unfold replayed exactly (steps verified unitary),
  snapshots at every depth equal to the coarse states with un-injected
  wires exactly `|0⟩`; wire sequences nest down the tree and measured
  Schmidt rank never exceeds the crossing channel's bond; the depth
  ledger localizing rainbow loss at the root while pair-local circuits
  stay exact; composition with `Ball` including dynamic growth.
- **clock_register** — scale snapshots equal the unfold with no gate
  applied; Page–Wootters conditioning reads out each scale exactly;
  the selector↔system Schmidt rank is the number of held scales and
  collapses to 1 on conditioning/measurement (both outcomes across
  seeds); the tick advances every occupied scale one level (verified
  against independently built snapshots) and refuses, by name, a
  topped-out clock and branch states shared across scales; scale
  interferometry equals the directly computed inter-scale overlap
  (`1/√2` per step on the structured state, `1.0` on `|0…0⟩`); the
  four-representation payoff pinned (3,845 B vs 68–850× alternatives,
  full-basis conformance of the contracted view before and after
  broadcast dynamics, no branch copied); the phase-faithfulness
  boundary made visible, never silent; width-32 six-scale histories in
  kilobytes, built the dynamic way.
- **mosaic_register** — full-registry conformance from singleton
  regions; partitions sculpted by gates with merge decisions carrying
  their predictions; refusal-driven migration exact against dense;
  **partial graphs**: a three-component bundle region fractured by its
  first T into bundle×3 with only the touched component migrating
  (split before any conversion, asserted in event order); the
  graph-of-graphs composition (a mosaic as a region of a mosaic);
  the width-40 flagship at the sum of ideal costs with era transitions
  ledgered; heterogeneous composition with the dynamic bulk register;
  merged-support saturation choosing dense with the prediction named;
  and the performance laws pinned: the mosaic's measured memory law
  sub-exponential on the era family while sparse-fixed measures
  exponential on the same family (×20+ at the shared width), and
  static structure costs zero events, zero conversions, and the parts'
  sum plus fixed bookkeeping (time scaling lives in
  `benches/width.rs`: `width_mosaic_era` vs `width_sparse_era`).
- **capacity** — the resource guard as behavior: over-scale allocations
  refused by *measurement* (requested vs available bytes in the error,
  auto-measured and under explicit limits); adaptive stays sparse when
  dense is inadmissible and promotes when the limit lifts; sparse
  growth, mera blocks and factored merges all admitted not presumed;
  compound gate/merge growth and constellation lattice-key growth
  refused under a 1 MiB budget with the representation named — while
  the same budget admits what stays small (an H-layer across
  independent horizontal volumes at 2n entries, a 20-qubit GHZ as two
  lattice points); an explicit `usize::MAX` limit is representable
  (regression: it used to collide with the auto-measure sentinel and
  silently re-enable admission), so with admission disabled an
  over-scale request reaches the ALLOCATOR and fails there, reported
  distinctly with the availability measured at failure time, while
  restoring auto-measurement re-arms up-front admission; armed time
  budgets abort a single dense gate
  mid-sweep, the mera SVD path and scheduled runs — promptly, with
  measured elapsed times — and the identical runs complete once the
  budget lifts.
- **bounds_atlas** — the boundary atlas: the law classifier calibrated
  on synthetic ground truths; the known fragments rediscovered from
  measured bytes (GHZ classical via constant sparse support while
  factored honestly pays `1.98^n`; QFT classical via bond; rainbow via
  clustering with sparse at measured base `1.40 ≈ √2` and the fixed
  tree at `1.96^n`; Clifford brickwork via the frame at `size^2`);
  random universal circuits escaping every assumption at once (bases
  1.7–2.0, all probes exact) → candidate; each representation
  exponential exactly in its own resource (support, cluster, per
  double-layer bond, `2^t` T-scatter saturated exactly, `√2`-per-H
  certified radius); and the walls as measured refusals (dense OOM
  with requested/available bytes, the 63-qubit indexing wall, profile
  axes reporting walls instead of failing); time laws joining the
  verdict (dense pays `>1.4^n` in time too, and the IQP
  memory-vs-time split is the measured case for requiring both); the
  IQP interaction-range and magic-doping verdict flips; the shallow-2D
  boundary law; selection by extrapolated scaling verified against
  holdout runs the fits never saw; variance-aware time laws (min/max
  envelope fits must agree with the median for a classification to be
  variance-robust); and register shapes as first-class axes (the
  hierarchical splits appear in every profile, constant on GHZ,
  escaping with everything else on random circuits, and measurably
  beating plain sparse on mixed-sector states).
- **sampling_hardness** — the task itself: XEB calibrated on ideal
  (≈1) and uniform (≈0) samplers with the measured-ceiling
  normalization; dense and exact D[ω] references agreeing to 1e-15 on
  the same samples; certified-easy families sampled exactly at
  certified cost; per-shot Clifford sampling with the bit convention
  pinned and the uniform-output no-signal case reported as `None`;
  the truncated-MPS spoof cliff (all caps < 0.3, full rank > 0.9);
  and fixed-budget spoofing decaying with family size.
- **algebraic_qudits** — the hierarchical register: qudit coordinates
  roundtrip with the documented bit order; the dual-algebra operator
  space measured per doubling level (full rank pinned at ℍ/𝕆/𝕊, the
  fifth doubling `#[ignore]`d at 1024/1024, synthesized sandwiches
  verified operationally against their matrices); **sandwich-native
  execution** identical to the component path with the counters proving
  which ran and the synthesis cache pinned; the **direct-sum boundary**
  (ℍ⊕ℍ span exactly the block-diagonals 32/64, cross-block SWAP
  measurably outside, register still fully conformant through the
  component path); full-registry conformance at ℍ, 𝕆 and ℍ⊕ℍ splits
  with routing chosen by measured embedded-linearity; the encoding
  exact across the site↔algebra boundary and certified against D[ω] at
  every split; projection/feedback on algebra qubits collapsing the
  joint state; 66–67 logical qubits exceeding every flat backend's u64
  ceiling, with deterministic Born sampling over (site, component)
  parts; site-sector structure surviving (support counts sites); and
  the workload harness pricing the hierarchical shapes in-run against
  dense.
- **mixed_compound** — mixed-arity compound qudits: the generalized
  gate relations (`X_d^d = Z_d^d = 1`, Weyl commutation,
  `F X F† = Z`, `F₂ = H` exactly); conformance to the qubit reference
  (all-binary bitwise-exact, a quaternary site ≡ the 2-qubit QFT);
  horizontal volumes correlating only on interaction with the event
  recorded; swap well-formed only between equal arities (the mis-sized
  cross-arity swap is a dimension error) and the fabric's swap-class
  decomposition; E8 constructed/verified with the orthogonal
  `su(5)×su(5)` pair exhibited and the rank obstruction measured by
  exhausted search; order-respecting interaction cones; guard-admitted
  merges (a 10¹⁰-entry merge refuses with measured bytes and leaves
  the register intact); sampling matching stated probabilities;
  arity-priced scaling (5³ = 125 entries vs 2·3·4 = 24); and k-site
  gates across mixed arities (a 24-dim three-arity gate equal to its
  site-wise factors, one logged interaction) with exact digit
  collapse.
- **e8_coboundary** — the E8×E8 co-boundary system: the root complex's
  measured counts ((1,56,126,56,0) at every point, 6720 edges, 2240
  zero-sum triangles verified as additive relations, b₁ = 4241 by
  GF(2) rank); the stored field invisible to either copy (all
  marginals exactly uniform) and projective (global phase changes
  nothing); recovery equal to the independently-computed DFT only via
  cross-copy interference with the flow record proving both copies
  were touched; and the wide-qudit fast paths validating exactly
  (non-bijections and non-unimodular phases refused as errors, refused
  gates leaving the register untouched).
- **e8_representation** — E8×E8 as an 8-qubit representation: the
  spinor bijection (128+128 injective over all 256 basis states); the
  transitions-are-integer-roots theorem verified both geometrically
  (all 112 covered) and against real `rxx(π)` gate matrices over all
  28 pairs × 256 states; entanglement as root geometry (GHZ an
  antipodal pair with zero root edges, rainbow affine-dimension 4 with
  exact per-cut entropies, random flooding the graph); and the
  projected-support bound on Schmidt rank holding on every family and
  cut, tight where the geometry is exact.
- **e8_backends** — both E8×E8 systems in the standard frameworks:
  `compound-binary` conformant over the full registry (per-gate
  sweeps, registry-drawn circuits, sampling equality, collapse) with
  its 63/64-qubit packed-index wall measured; `e8-rep` conformant
  including a width-8 sweep over the complete 256-point set, refusing
  9 qubits structurally, and GHZ downcast-verified to be *stored* as
  an antipodal spinor-root pair; the benchmark harness pricing both
  against dense/sparse/adaptive/factored/mps/mera on ghz/qft/rainbow
  plus the d = 16 co-boundary protocol (MPS refusing its 8-qubit
  pairing gate at the measured window wall); and the native
  mixed-arity protocol equal amplitude-by-amplitude to its
  qubit-encoded run on dense, both equal to the independent DFT.
- **e8_constellation** — the infinite constellation: the coset
  theorem measured in full (origin + 240 roots + 2160 norm-2 vectors
  bucket into exactly 256 classes sized 1/2/16 with census 1/120/135,
  representatives on their spheres, the address map linear, non-lattice
  input refused); the scale tower a bijection (all 65 536 depth-2
  strings round-trip to distinct points, depth-7 towers round-trip,
  doubling prepends digit 0); backend conformance over the full
  registry at default and widened widths plus multi-block random
  circuits at 10 and 12 qubits against dense, with the 63/64 u64
  wall; the harness pricing it beside dense/sparse/factored/mps while
  e8-rep's native-8 wall is recorded in the same table (GHZ-16 two
  points ≪ dense, saturated QFT-12 honestly larger than dense); and
  the 63-qubit GHZ stored as exactly two lattice points (digits
  pinned, census `[1,1,0]` per level, 224 bytes) where dense cannot
  construct at all.
- **e8_weyl** — the cross-scale Weyl pair: self-duality exact (dual
  basis in-lattice, pairing δᵢⱼ, coordinates = dual inner products on
  points and roots); the Heisenberg law with its 2-adic ladder pinned
  on a 3×3 scale grid (character `4·2^{i+j}` below the horizon,
  exactly 1 past it, orderings measurably differing below and agreeing
  to 1e−12 above); W(E8) reflections as involutions with exact
  `T`/`M` conjugation covariance; `F⁴ = 1` and the measured conversion
  `FT_BF⁻¹ = M_{−b*}`; support uncertainty exactly `2^{16}` at ranks
  0/1/2; the 40-qubit structured protocol (line → cross-scale
  translate → dual modulate → W(E8) round trip → F⁻¹ collapsing to
  the single point carrying the modulation scale, support ≤ 32
  throughout, dense refusing 17.6 TB in the same test); natives
  reducing at depth 1 to the X-string on `class(v)` and the
  `(−1)^{⟨q,p⟩}` diagonal, checked against the standard framework; and
  partial blocks / non-lattice labels / non-root mirrors refused.
- **e8_dual_scale** — multi-scale dual time: the scale operad exact
  (embeddings compose, R∘V = id, Weyl operators transport covariantly,
  decimation of live fine data refused with the level named — and the
  lattice-divisibility subtlety pinned: componentwise evenness is NOT
  divisibility); comb codes with cross-scale commuting checks, middle-
  scale logicals (X̄ flips Z̄'s eigenphase while every check stays +1),
  and code self-similarity under decimation; the full m×m syndrome
  matrix tiling the horizon inequality exactly; end-to-end correction
  (displacements (3,…,1,…) decoded exactly from syndrome phases alone,
  corrected to 0.0 deviation, decimated to the pristine coarse state)
  with the past-window blindness measured honestly; the interleave
  fold to one two-gate layer with its below-horizon refusal control;
  and deep periodic time folded to the measured period P = 8 with the
  exact quadratic Weyl phase pinned at small t.
- **e8_comb_noise** — the codes as objects under noise: `CombCode`
  reproducing the dual-scale wave's pinned facts with parameter
  validation; min-norm decoding correcting every in-window mixed-sign
  displacement exactly (the +4 tie included) while window-sized
  displacements are measured as silent logical operations (the code
  distance) and the mod-2 tie-break's fail-half honestly pinned on
  both signs; and 9 600 seeded noise trajectories yielding the
  logical-vs-physical curves — the degenerate window saturating ≥ 90%
  at every nonzero rate, a=4 strictly rising in p and suppressed by
  wide measured margins below a=3 and a=2.
- **split_quaternion** — the coquaternion table (`i²=−1`, `j²=k²=+1`,
  `ij=k`, `ji=−k`) and the norm measured multiplicative; the grading as
  the inclusion–exclusion rule; the full registry equal to ℂ's against
  split-complex's strict subset; conformance on sparse/adaptive/factored
  with mps/mera/clifford-frame refusing by named reason; framed, device
  and interference agreeing with dense (and `H·H` destroying exactly
  1.0 over the new algebra); the channel decomposition asserted
  **exactly equal** to two separate complex runs, per amplitude and in
  both totals; net-weight conservation across every registry gate on a
  net-negative state; the boost unitary with path weight pinned to
  `cosh 2t` and a basis-mixing boost unitary too; the exchange's
  deviation of exactly 2 with the registry refusal and the measured sign
  flip; the null cone decidable with measurement and sampling refusing,
  and a net-positive mixed state sampling at the predicted `0.64 : 1.0`
  ratio; plus the algebra-sector qudit and harness rows. The algebra
  also joins the existing `reproductions`, `property_tests` and
  `gate_matrices` sweeps.
- **closure** — cycle rank equal to `|E| − |V| + 1` on rings and grids
  with every loop verified to be a real closed walk; a tree measured to
  have no reach at all; thirty intended changes each leaving closure
  intact; a perturbation breaking **exactly** the loops through its site,
  asserted for all 25 sites of a grid; localization classified into
  named / ambiguous / invisible with the true site always among the
  candidates; retrodiction bracketing an unrecorded injection and
  naming its site, in no more than `log2(stamps) + 2` comparisons; an
  unbroken history retrodicting to nothing; stamp cost pinned at two
  bits per loop; correction restoring closure and the denoted state at
  deviation exactly 0.0; correction refusing both ambiguity and
  link-only breaks by message; the chirality channel localizing an
  ordering break to `(5,6)` while deliberate re-orderings keep closure;
  past entanglement reconstructed by rewind with the cycle rank changing
  with it; and a 14 400-site geometry stamped, perturbed and retrodicted
  inside 32 000 bits.
- **bundle** — the GHZ sign measured into a fiber (identical base
  signature, orthogonal denoted states, marginals blind at 0.0,
  tomography reading the flipped fiber at ~1e-16) across n = 3…8;
  mixed-frame tomography round-tripping through a real state; a
  T-rotated state reported as out-of-sector rather than smoothed over;
  chirality checked against `PolaritySystem::product` on four orderings
  and adjacent swaps agreeing with a wholesale reorder; coarse-graining
  accounting for every removed link on the absorbed/collapsed split with
  surviving weights still covering every original site; budgets met and
  priced; interaction confined to frame-compatible commonality with the
  incompatible fiber measurably untouched; journal replay exact at the
  end and at every prefix; a coarse-grained bundle refusing to pretend
  it still denotes a state; and 100 000 sites built, profiled, reordered
  and inspected with the structural footprint held under 128 B/site and
  `checkpoint()` measured to halve the total.
- **polarity** — the structure theorem brute-force verified (centre
  counted monomial by monomial, maximal commuting sets built by
  exhaustive search and checked closed under XOR, `isotropic × matrix =
  dimension` across the whole twist dial, rank always even); the Pauli
  system measured maximally twisted with stabilizer-sized isotropic
  subgroups; associativity and the commute/sign equivalence swept over
  every monomial pair of a 4-generator system; `Polarity<2>` transported
  onto `SplitQuaternion` at deviation exactly 0 in both product and Born
  form; the real-subset/full-registry split at one versus two
  polarities; conformance over `Pol3`; the storage padding pinned rather
  than hidden; and the obstruction — pairwise-complete at `n = 2`,
  blind from `n = 3` up with signature deviation exactly 0 against
  overlap 0, the `n`-body correlator reading ±1, the pairwise-vs-state
  value counts showing 36 > 16, the GHZ weight profile `[n, 2, …]`, and
  a product state as the contrasting case where pairwise data does
  determine the state.
- **recursive_lattice** — the structure (square = 4 bonds; square of
  squares = 16 qubits, 20 bonds, the four lateral bonds pinned by index
  and each verified to leave its block; depth 3 = 64 qubits, 84 bonds
  split 64/16/4 by depth; the fabric run as a `DeviceState` geometry
  carrying a 16-qubit GHZ); refine/coarsen as structural inverses with
  re-refinement and point-addressing both refused by message; the
  block-spin step (isometry error < 1e-12, `h' = Δ/2` and `J' = J·μ²`
  identities exact to 1e-15, a zero field refused rather than divided
  by); the flow monotone on the disordered side and terminating with
  the double-precision reason on the ordered one; the fixed point
  (`g* = 0.783243`, λ = 1.596338, ν = 1.482, residual < 1e-12); the
  substitution's resolution horizon (in-band deviation < 2% while the
  full-spectrum deviation exceeds 0.3, with the offending coarse level
  shown to sit above the block's internal gap, and in-band error held
  under 10% across the coupling range); the harmonic block's exactly
  zero frequency shift and 1/2 port participation; uniform bonding
  exact under 1e-12 against port bonding's finite error, with the
  second-order law pinned as a 50–200× ratio per decade — twice, once
  in the spectra and once in the dynamics; the phonon walk's zero
  leakage, step-independent substitution error and shrinking Trotter
  error on the dense backend; and self-participation converging to
  zero below threshold and to a self-sustaining answer above it, at
  depth 2 as readily as depth 1, with the critical coupling
  `0.267675` bisected and both unbracketed directions refused by
  message.
- **characterize** — the characterization harness measures what it
  claims: the census covers every registry name (dense bit-exact against
  itself, sparse bit-exact against dense on all 39); a construction
  ceiling is the backend's own refusal (`e8-rep` at 8, carrying its
  message) while a sweep stopping short reports no refusal at all; a
  family ceiling is allowed to be far narrower than the constructor's
  (GHZ vs an H-layer on the constellation, the latter walling on a
  measured operation refusal); a walled width contributes **no**
  fabricated cost to the fitted laws (the sweep past `e8-rep`'s ceiling
  records nothing, and dense still fits base ≈ 2 over widths that fit);
  and the envelope separates the representations by *measured* bytes per
  stored amplitude — dense ≫ 10× the sparse maps, an eight-coordinate
  lattice key strictly above a `u64` one.
- **bundle_measure** — graph-state measurement done in the description.
  The collapse is checked against projecting the *materialized* state over
  **672 cases** — seven graph shapes × four decorations (bare / vertex
  operators / spins / both) × three seeds × every site × both outcomes —
  worst deviation under 1e-9, plus sequences of three successive
  measurements on five-site graphs so the post-measurement description is
  shown to be one a further measurement can use. A GHZ bundle returns the
  same outcome on re-measuring a collapsed qubit and perfectly correlated
  outcomes across all six sites; tomography against the bundle's own
  stabilizer generators confirms the description still denotes the state it
  claims. An isolated fiber's outcome can be deterministic (probability
  exactly 1 and 0), and the impossible projection is refused by leaving the
  state alone rather than zeroing it. Native sampling matches the dense
  marginal distribution to 0.05 over 4000 shots. The collapse is journalled
  as one semantic step and replays exactly.
- **phase_degree** — degree 1 is verified to be *exactly* being a
  character, and the degree law
  `multilinear + log₂(denominator) − 1` is confirmed on eleven diagonals
  spanning both dials independently, with the two consequences asserted
  separately: a diagonal is a character iff both dials are minimal, and the
  ladder sorts `z`/`s`,`cz`/`t`,`cs`,`ccz`/`ct`,`cccz` into Clifford-hierarchy
  levels 1/2/3/4. One E8 volume is checked to be the bit group over **all**
  256 × 256 pairs; two volumes differ on most. The native modulation is
  degree 1 on the residue group and witnessed above 3 on the bits, while a
  `t` is degree 3 on both — and its order-two residual is pinned to √2 and
  its order-one to `2 sin(π/8)`, tying this module to `e8_across` and
  `selfhost` numerically rather than by narrative. Linearizing is shown
  unable to lower the degree below the denominator floor. The instrument's
  soundness asymmetry is tested too: a sampled sweep certifies nothing
  (`certified_degree` is `None`) but still witnesses (`exceeds` holds), and a
  constant phase stops the sweep at the first vanishing order.
- **memo** — progressive gate-result memoization. The whole **registry**
  survives it: 24 circuits drawn from the registry's own gates (aliases and
  parametric gates at random angles included) at three fusion widths, every
  deviation under 1e-12, and exactness holds on dense, sparse and adaptive
  and over ℍ as well as ℂ — the content key is `Scalar::coeffs`, so it is
  not a ℂ-only trick. Repeated geometry collapses to one entry regardless
  of absolute position while qubit *order* within a support stays part of
  the identity. `max_fuse` is monotone in both directions (operations never
  increase, entry bytes never decrease). A lone diagonal stays diagonal;
  fusing onto it promotes it; a gate wider than the limit keeps its own
  support and is still memoized. The accounting must add up —
  hits + misses = operations, entries = misses, and the `fused_from` counts
  sum to the source gate count. The journal rewinds to states exactly equal
  to a fresh run at every step, and refuses to advance backwards or rewind
  ahead of the cursor. **The commit-order regression is pinned** by the
  minimized nine-gate circuit that gave a 1.408 deviation under
  opening-order emission.
- **selfhost** — the self-computing object: **each layer of E8 volumes
  buys exactly one degree** (depth = degree − 1, verified for degrees 2
  through 6, every route exact at deviation `0.0`), and the linearized
  degree is 1 throughout. A `ccz` goes from character residual 2.00 on the
  substrate to 2.4e-16 on the stack and its single-qubit phase reproduces
  the registry's own three-qubit `ccz` to under 1e-12 — a non-Clifford
  diagonal became a character. A `t` is the kept-apart exception: degree 1
  already, so zero layers, zero volumes, zero cost, and a residual that is
  the **same √2** `e8_across` measures for a `t` on the constellation, so
  the two modules pin one obstruction from two directions. The trade is
  counted, not asserted: entangling gates per use (`[1,2,4,8,16]`) versus
  once (`[1,1,1,1,1]`), counted crossover at the second use, with the
  wall-clock slope measured lower even where the constant is higher.
  Volumes are quantized to eight coordinates and `occupancy` says so;
  planning refuses a monomial outside the substrate, linearizing refuses a
  monomial the stack does not hold, and a non-linear diagonal refuses to
  factorize into single-qubit phases. Exact on dense, sparse and adaptive
  alike.
- **e8_across** — computing *across* a set of E8 volumes, measured: one
  finest-scale translation carries into **every** copy (touching
  `8m − 7` qubits, support-preserving), while a single copy has no
  carries at all (`E8/2E8 ≅ F₂⁸` is linear, so the operation is a
  bitwise XOR needing zero two-qubit gates) — the cross-scale reach is
  created by having more than one copy. The carry ties the whole register
  into one influence component, so the measured two-qubit lower bound is
  `n − 1`, and the advantage over the qubit path is **linear**: exactly
  8 qubits and 8 gates per added copy, fitted degree 1.04, subexponential.
  The native direct DFT is the honest limit — support exactly
  `2^m` (fitted base 2.0) and a cost law that is *not* subexponential.
  And the class: every native operator (translate, modulate, reflect,
  permute, Fourier) stays inside coset-with-linear-character, verified on
  a genuinely spread state, while a `t` through the qubit path keeps the
  coset and breaks the character (residual 1.41) — so the qubit path
  reaches states no sequence of native operators can. A pair whose
  difference has order four is measured *not* a coset, and a partial
  8-qubit block is refused rather than silently embedded.
- **e8_cube** — the twelve edges as three axis matchings; the parity
  law over all 240 roots (112 + 128 by sector, every root summing to a
  multiple of four, mixed parities in neither sector); the symmetry
  count `(48, 48)` with every vertex translation verified against all
  240 roots, an axis relabeling accepted and a shear accepted as a
  lattice automorphism but rejected as a cube symmetry; the
  symmetries acting on live states (norm preserved, the translation an
  exact involution, non-permutations and partial blocks refused); the
  tower round-tripping and doubling prepending an empty finest cube;
  every elementary displacement verified a lattice point at every
  scale with the finest volume's missing interior refused; and the
  ladder — different vertices commuting exactly everywhere, the same
  vertex interacting exactly while `j + k < m − 2`, the phase ladder
  `exp(−2πi·2^{j+k+2−m})` pinned level by level at m = 5, and the
  self-reference depth measured none → 0 → 1 → 2 as the tower grows.
- **causal_geometry** — the causal-geometry suite across every backend:
  the register metric measured through the router on six geometries
  (swaps = graph distance − 1, the clock in exact agreement); causally
  organized fabrics (`hierarchical`, `hypercube`) collapsing horizon,
  ball growth, availability and measured QFT routing versus flat
  fabrics; causal range priced per representation geometry (mobile MPS
  and device pay time, mera's fixed tree pays rank at the crossed cut —
  saturating honestly to dense scale at maximal range — factored
  clustering and Clifford frames blind); the light cone measured at
  width 63 (marginals exactly zero outside, schedule length = causal
  depth); causal diamonds observationally identical at a quarter of the
  ops; dual-time resolution cut-invariant with `2^{D/2}` supports at
  the balanced cut and the fold-back destruction ledger matching the
  closed form `Σ(√2)^j`; everything certified against the exact D[ω]
  ring and Ball containment, and the whole family swept through the
  benchmark harness over ten backends.
- **pathsum_merge** — the residual-mosaic suite: the merged route
  asserted equal to enumeration at 1e-16 over 200 random circuits ×
  every amplitude × every branching policy; a Clifford circuit
  reaching the solver as one leaf with nothing branched; the budget
  refusing by name; thin magic logarithmic (`h* = 64` in under 100
  nodes, where enumeration is 1.8e19); width asserted innocent and the
  2D/1D gap asserted *widening*; the square family's `log2(nodes)/n`
  asserted to settle rather than grow — the `√t` law, pinned as a
  property of the circuit; and the election, with each single signal
  asserted to lose to the other somewhere and the composed one
  asserted to beat both everywhere.
- **stitch** — the symplectic sharding suite: the normal form
  recovering `(r, h)` off the generators of four codes (toric L ∈ {2,3},
  surface d ∈ {3,5}) with the radical asserted isotropic *and* central
  in the span and every hyperbolic pair asserted anticommuting; every
  selection maximal isotropic of rank `n` carving dimension exactly 1,
  every pair of selections joined asserted *non*-isotropic (nothing is
  fixed by two slices) and their meet always containing the radical
  and never everything; the centraliser returning the normalizer at
  rank `n + k` with every stabilizer and logical in it; the four toric
  slices prepared as states with each one's basis read *off the volume*
  rather than assumed, asserted in the code space, shown to be a frame
  rather than an orthogonal basis, shown linearly independent, and a
  random vector projected through the full `2^r`-element stabilizer sum
  reassembling from them to < 1e-9; and the accounting — per-node cost
  independent of node count, network total tracking `2^h` rather than
  `2^n`, with the tableau-per-coefficient overhead asserted `O(n²)`
  rather than a new exponential.
- **lateral** — the cross-lateral distributed suite: the
  coordinatewise link rule against gate-by-gate conjugation on 80,000
  random strings across four window layouts (signs included) and the
  no-third-site property per input site; the syndrome's 𝔽₂ linearity
  over 50,000 random pairs on an L = 3 torus, and agreement with
  `retro::signature`; erasure capacity enumerated off the code's own
  generators at `d − 1` for toric L ∈ {2,3} and surface d ∈ {3,5}, with
  a weight-`d` logical produced as the witness that ends it, and the
  factor of two against the error decoder pinned at L = 2 (the weight-1
  `X` that `Decoder` refuses by name, recovered exactly as an erasure);
  the decoder's two distinguishable refusals (a set carrying a logical,
  a syndrome outside the set's image) and every reachable syndrome
  decoding to the fault that produced it; *correctable* versus *unique*
  separated on the surface code's weight-2 boundary check; the
  distributed frame equal to the gate-by-gate one **and** its signature
  equal to the syndromes a dense 16-qubit register shows, on both
  nodes; a straddling "local" Pauli refused; the wire form round-tripped
  1,000 times; triangle violations found and tightened away with the
  horizon floor read off the diameter; the barrier sealing clean,
  waiting, repaired and degraded in sequence with the peers named; the
  MDS parity exact on every 2-of-4 erasure pattern and refusing 3 by
  name; and the dual tower's cross-scale synergy — a pattern that
  leaves 3 slots dark under the coarse layer alone and 4 under the fine
  layer alone, closed byte-exactly by the two crossed.
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

## Characterizing a representation fully

```sh
cargo run --release --example e8_characterization
```

`characterize` produces the numbers you ask of a new representation
before anything else — the maximum width, how much of the gate set it
actually reproduces, and its best and worst cases — and every one of
them comes from a run. A ceiling is the width at which a real
construction or a real gate application refused, carrying the library's
own message; a fidelity count is a gate-by-gate comparison against the
reference; an envelope is wall-clock timing and reported footprint at the
widths that ran. A walled width contributes **nothing** to the fitted
laws rather than a fabricated cost.

Measured on this machine (widths 4…24, 8 s armed budget):

| backend | constructs to | registry match | GHZ ceiling | H-layer ceiling | ns/gate @ n=8 | B/amplitude |
|---|---|---|---|---|---|---|
| `dense` | 24+ | 39/39 bit-exact | 24 (268 MiB) | 24 (268 MiB) | 593 – 2860 | 2064 – 4128 |
| `sparse` | 24+ | 39/39 bit-exact | 24 (**125 B**) | 24 (800 MiB) | 11379 – 14416 | 62.5 – 125 |
| `e8-rep` | **8, declared** | 39/39 bit-exact | 8 (168 B) | 8 (14 KiB) | 226 – 385 | 84 – 112 |
| `e8-constellation` | 24+ (u64 wall at 63) | 39/39 bit-exact | 24 (**224 B**) | 20, then time-walled | 8726 – 24539 | 112 – 144 |

Both E8 backends reproduce the **entire 39-gate registry bit-exactly**,
so universality on them is not in question. What the characterization
adds is the price: `e8-rep` is eight qubits by construction, and the
constellation's per-gate cost is 10–30× dense because every gate
converts through the tower bijection — its clifford-brickwork ceiling is
set by *time*, not memory, at a support of only 2048. The honest summary
is that the constellation's cost tracks the lattice↔bits conversion
rather than how much state it holds.

### What the √2 was

```sh
cargo run --release --example phase_degree
```

Two modules above hit the same number from different directions: `e8::across`
found a `t` breaking the native linear-character class by **√2**, and
`selfhost` found a `t` untouched by linearization with the same **√2**
residual. `phase` resolves it by measuring the thing neither was measuring.

For a phase function `f: G → U(1)` on the group indexing the register, take
the discrete derivative `(Δ_a f)(p) = f(p+a)/f(p)` and iterate. `f` has
**degree ≤ d** when every `(d+1)`-fold derivative is identically 1 — and
**degree 1 is exactly being a character**, since `Δ_a f` not depending on `p`
*is* `f(p+a) = f(p)f(a)`. So "is it a character" and "what is its degree" are
one question, and the degree says how far from one a diagonal is.

Measured on nine diagonals spanning both dials independently:

| gate | multilinear | log₂ N | predicted | measured | character? |
|---|---|---|---|---|---|
| `z` | 1 | 1 | 1 | **1** | yes |
| `s` | 1 | 2 | 2 | **2** | no |
| `t` | 1 | 3 | 3 | **3** | no |
| `t^½` | 1 | 4 | 4 | **4** | no |
| `cz` | 2 | 1 | 2 | **2** | no |
| `cs` | 2 | 2 | 3 | **3** | no |
| `ct` | 2 | 3 | 4 | **4** | no |
| `ccz` | 3 | 1 | 3 | **3** | no |
| `cccz` | 4 | 1 | 4 | **4** | no |

Every case matches

```
phase degree = multilinear degree + log₂(denominator) − 1
```

so a diagonal is a character exactly when **both** dials sit at their
minimum: multilinear degree one *and* ±1 valued. And the degree ladder is
the **Clifford hierarchy for diagonal gates, rediscovered from
measurement**: degree 1 the ±1 characters, degree 2 Clifford (`s`, `cz`),
degree 3 the first non-Clifford diagonals (`t`, `cs`, `ccz`), degree 4
(`ct`, `cccz`).

**Why the stack fixed a `ccz` but not a `t`.** `selfhost` reduces the
*multilinear* term. The log-denominator term is untouched, so the floor is
`log₂(denominator)`:

```
ccz  degree 3 -> 1   layers 2   multilinear 3 -> 1   character after: true
t    degree 3 -> 3   layers 0   multilinear 1 -> 1   character after: false
```

A `t`'s entire degree is its denominator, so no depth of stack can help.
That derives the earlier observation instead of restating it.

**And the two groups.** The constellation's native operators and the qubit
path are characters of *different* groups:

```
1 volume,  n=8:  residue addition differs from XOR on      0 of 65536 pairs
  native modulate  on residue  degree 1     on bits/XOR  degree 1
  qubit t          on residue  degree 3     on bits/XOR  degree 3

2 volumes, n=16: residue addition differs from XOR on 229292 of 262144 pairs
  native modulate  on residue  degree 1     on bits/XOR  degree 5
  qubit t          on residue  degree 3     on bits/XOR  degree 3
```

At one volume the two groups **coincide exactly** — `E8/2E8 ≅ F₂⁸` and
`class_of` is linear, verified over all 256 × 256 pairs. From two volumes on
they do not, because extracting higher digits carries. A modulation is
degree 1 on the residue group and degree 5 on the bits; a `t` is degree 3 on
both.

The √2 itself is this instrument's **order-two residual for a `t`**:
`|i − 1| = 1.414214`. The full ladder for `t` is
`[0.765, 1.414, 2.000, 0]` — the first entry `2 sin(π/8)`, then √2, then the
maximum a phase can be off by, then vanishing at order 4.

### An object that computes itself, one layer per degree

```sh
cargo run --release --example selfhosted_stack
```

`selfhost` builds the object as described: a substrate register, and above
it a tower of E8 volumes where **each layer holds the layers below it as
its own coordinates**. One volume is `E8/2E8 ≅ F₂⁸` — one byte, eight
coordinates — and the class map's measured F₂-linearity is what lets a
coordinate hold an arbitrary F₂ function of the substrate and still be a
coordinate.

Any diagonal unitary is a phase polynomial
`exp(2πi Σ_M c_M x_M / 2^bits)`. Degree 1 is the special case: **a
degree-1 diagonal is a product of single-qubit phase gates**, and on a
group whose coordinates are the bits it is a *character* — the one
diagonal an F₂ volume applies natively. `SelfHostedStack::linearize`
rewrites a degree-`d` diagonal as degree **1** over the stack by replacing
each higher monomial with the coordinate holding it.

Measured, on a 6-qubit substrate:

| gate | deg | layers | vols | width | χ(substrate) | χ(stack) | character? |
|---|---|---|---|---|---|---|---|
| `cz` | 2 | 1 | 1 | 14 | 2.00 | 2.4e-16 | **yes** |
| `ccz` | 3 | 2 | 2 | 22 | 2.00 | 2.4e-16 | **yes** |
| `cccz` | 4 | 3 | 3 | 30 | 2.00 | 2.4e-16 | **yes** |
| `t` | 1 | 0 | 0 | 6 | 1.41 | 1.41 | no |

χ is the failure of the diagonal to be a character of the bit group — the
same instrument `e8::across` points at the constellation. A `ccz` is
maximally far from one (2.00) and becomes one exactly; the linearized
`ccz` is a **single-qubit phase gate** that reproduces the registry's
three-qubit `ccz` on a uniform superposition to 0.0e0. A genuinely
non-Clifford diagonal has become a character.

`t` is the honest exception, and it is the *same* obstruction measured
twice: `t` is already degree 1, so the object correctly does nothing (zero
layers, zero volumes, zero cost), and its eighth-root phases never become
±1 valued. That residual is exactly the √2 the constellation reports for a
`t` through the qubit path. Degree reduction and character-hood are two
different things, and the report keeps them apart.

**The recursive expansion is exactly one layer per degree:**

```
degree  layers   vols  width   linear   toffoli  deviation
     2       1      1     16        1         1      0.0e0
     3       2      2     24        1         3      0.0e0
     4       3      3     32        1         5      0.0e0
     5       4      4     40        1         7      0.0e0
     6       5      5     48        1         9      0.0e0
     7       6      6     56        1        11      0.0e0
```

**What it costs.** The expansion cannot be free — if it were, the native
operator set would manufacture non-Clifford diagonals from nothing.
Writing a degree-`k` monomial into its coordinate is a `k`-controlled X:
precisely the non-native work. But it is paid **once**, and the diagonal is
single-qubit phases forever after:

```
uses        [1,   2,   8,    32,   128,   256,   512,   1024]
direct 2q+  [1,   2,   8,    32,   128,   256,   512,   1024]   max arity 3
stack  2q+  [1,   1,   1,    1,    1,     1,     1,     1]      steady arity 1
direct ns   [461, 632, 2243, 8604, 35947, 71730, 154954, 280923]
stack  ns   [18488, 14057, 11810, 20000, 34640, 59721, 106926, 186216]
```

Counted crossover: **2 uses**. Measured crossover: **128 uses** — a
classical simulator applies a diagonal kernel in `O(support)` whatever its
arity, so the win shows up in the slope (164 ns/use versus 274) rather
than the constant. Both numbers are reported because the counted one is
the resource the object actually trades and the measured one is what this
machine sees.

### Does computing *across* a set of E8 volumes pay?

`e8::across` measures it instead of arguing it. One native finest-scale
translation carries into **every** copy in the tower, and its influence
graph is a single connected component, so the measured two-qubit lower
bound for any circuit realizing the same permutation is `n − 1`:

```
op           qubits  touched  involved  copies  comps  2q-min     nanos
translate         8        1         8     1/1      8       0       250
translate        24       17        24     3/3      1      23       404
translate        40       33        40     5/5      1      39       586
fourier[0]       40       33         —     5/5      —       —     27401

copies      [1, 2, 3, 4, 5, 6]
touched     [1, 9, 17, 25, 33, 41]
2q-min      [0, 15, 23, 31, 39, 47]
per extra copy: 8 qubits touched, 8 two-qubit gates replaced
advantage law over copies: Polynomial { degree: 1.04 }
```

At **one** copy there are no carries at all — `E8/2E8 ≅ F₂⁸` and the
class map is linear, so the translation is a bitwise XOR needing zero
two-qubit gates. The cross-scale reach is created by having more than one
copy, which is the "across" effect, measured. But it is **linear**: 8
qubits and 8 gates per added copy, exactly, fitted degree 1.04. One
native operation replaces `Θ(n)` two-qubit gates — a real
constant-factor-per-copy win, not a non-linear advantage.

The native direct DFT is the ceiling on that side: support exactly `2^m`
(fitted base 2.0) with a cost law that is *not* subexponential. The one
native operation that creates superposition pays exponentially for it.

And the reachable class is closed: every native operator — translate,
modulate, reflect, permute, Fourier — maps a coset carrying a linear
character to another one (verified on a genuinely spread state, since a
size-1 support satisfies the class trivially and proves nothing). A `t`
driven through `Backend::apply` keeps the coset and **breaks the
character** (residual 1.4), so the qubit path reaches states no sequence
of native operators can. Full quantum computing on the constellation is
real, and it is bought entirely on the qubit path — which prices as
sparse with a bigger key.

## Progressive gate-result memoization

```sh
cargo run --release --example gate_memoization
```

`memo` treats qubit operations as **`n`-wide operation objects** rather
than "this gate, on those qubits". `MemoPlan` rewrites a circuit into
operations over a *support*, fusing consecutive gates while they fit inside
`max_fuse`, and stores each distinct operator once in a shared entry table.

The table is **content-addressed on exact coefficient bits** (via
`Scalar::coeffs`), which is what memoizes the geometry away: a fused
operator over a sorted support is indexed by position *within* that
support, so it does not know which qubits it sits on. A `cx` on `(0, 1)`
and a `cx` on `(7, 8)` are the same matrix and therefore the same entry —
nothing is canonicalized by hand, and because the key is the bit pattern
rather than a hash, a hit is an identity, never a guess. Qubit *order*
within a support stays part of the identity: `cx(0,1)` and `cx(3,2)` are
correctly distinct.

| circuit | gates | ops | entries | reuse | hit rate | fusion | entry bytes |
|---|---|---|---|---|---|---|---|
| `ghz-12` | 12 | 4 | 3 | 1.33 | 25% | 3.00 | 9 216 |
| `qft-8` | 40 | 15 | 10 | 1.50 | 33% | 2.67 | 20 224 |
| `brickwork-10x6` | 81 | 16 | 8 | 2.00 | 50% | 5.06 | 11 072 |
| `rainbow-10` | 10 | 5 | **1** | 5.00 | 80% | 2.00 | 256 |
| `random-10x120` | 120 | 21 | 18 | 1.17 | 14% | 5.71 | 40 704 |

Structure shows up as reuse and its absence shows up too: a rainbow
collapses to a single entry, a random circuit to almost none. `max_fuse` is
a measured dial trading operations for entry bytes, on `brickwork-10x6`:

```
max_fuse=1  ops=61  entries=4  widest=2  bytes=448      max_fuse=4  ops=16  entries=8  widest=4  bytes=11072
max_fuse=2  ops=44  entries=5  widest=2  bytes=1088     max_fuse=5  ops=10  entries=8  widest=5  bytes=58880
max_fuse=3  ops=16  entries=8  widest=3  bytes=5888     max_fuse=6  ops=9   entries=7  widest=6  bytes=169216
```

**Fusion is sound by disjointness.** Open supports are disjoint; a gate
merges the groups it touches when their union fits (disjoint operators
commute, so the merged operator is their product in any order), otherwise
those groups commit and the gate opens a fresh one. Operations are emitted
in **commit order**, which is a valid linearization because a qubit is
owned by at most one open group and ownership transfers only at commit.

Emitting in *opening* order is not valid, and that is not a hypothetical:
a still-open group can acquire a qubit an already-committed group used, at
which point its opening position predates gates it does not contain. That
produced a **1.408-amplitude** error on a random circuit. `tests/memo.rs`
pins the minimized nine-gate case.

### Unwinding and rewinding to re-explore

`Explorer` runs a plan while journalling state snapshots at a configurable
stride; `rewind` restores the nearest snapshot and replays forward, landing
on states indistinguishable from a fresh run (asserted exactly, at every
step, in both directions). `explore` uses that to run variants over a
shared prefix — evolved once instead of per variant:

```
variants   prefix   operations applied        wall clock (ns)          deviation
2              16    36 -> 18   2.00x      567807 -> 295553   1.92x     2.8e-17
4              16    72 -> 20   3.60x      925004 -> 391940   2.36x     2.8e-17
8              16   144 -> 24   6.00x     1956374 -> 486516   4.02x     2.8e-17
16             16   288 -> 32   9.00x     3959034 -> 1036463  3.82x     2.8e-17
```

The sharing is verified rather than assumed: the deviation column is the
worst amplitude difference against running every variant from scratch. The
naive route deliberately gets the *better* plan — it may fuse across the
prefix/variant boundary, which the shared route cannot, since the combined
plan's operation boundary is not a valid cut — and still loses.

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
                 split-quaternion (inclusion/exclusion pair),
                 polarity<N> (N twisted polarities per amplitude),
                 Ball (certified midpoint ± radius, quantize dial)
  math.rs        GateMatrix<S>: matmul, dagger, controlled, kron, unitarity
  phase.rs       polynomial degree of a phase function by iterated
                 discrete derivatives: degree 1 IS being a character, and
                 the degree ladder is the Clifford hierarchy
  memo.rs        progressive gate-result memoization: operations as n-wide
                 objects over a content-addressed entry table, with a
                 journal that unwinds and rewinds to re-explore branches
  bounds.rs      boundary atlas: measured growth laws, advantage scan
  characterize.rs full per-backend characterization: construction and
                 per-family width ceilings (carrying the library's own
                 refusal), gate-fidelity census, min/max perf envelope
  sampling.rs    sampling-task hardness: XEB, spoof curves, exact refs
  mixed.rs       mixed-arity compound qudits: volumes, fabric, flow, backend
  e8.rs          E8 roots built+verified; chains, rep, constellation,
                 cube (the 2x2x2 volume, inward/outward interaction),
                 across (reach per unit cost of the native cross-scale
                 operators; the coset class they cannot leave)
  causal.rs      backward cones, causal diamonds, dual-time resolution
  stitch.rs      sharding a register by its own symplectic form:
                 volumes as F2 subspaces (meet, join, centraliser =
                 normalizer, isotropic = stabilizer), the symplectic
                 normal form radical + hyperbolic pairs read off a
                 code's generators, the 2^h maximal isotropic
                 extensions and their exact direct sum, and where the
                 exponential goes when you spend it as machines
  lateral.rs     cross-lateral distributed registers: the transversal
                 link as a coordinatewise F2 frame map, the F2-linear
                 syndrome, erasure decoding at d-1 (measured, with the
                 logical that ends it as a witness), the delay geometry
                 and arrival-tracked barrier, and the dual tower —
                 the quantum code as the free fine layer of the
                 network's erasure code
  clock.rs       the time system inside the register: a selector qudit
                 over representation-heterogeneous branches (shared
                 states, lazy interference, polynomial Gram machinery),
                 the Page–Wootters scale history over the bulk depth
                 axis, the clock-controlled tick, scale interferometry
  bundle.rs      fibered re-orderable journalled polarity co-bundle:
                 entanglement as a budgeted, auditable resource, with
                 native graph-state measurement (collapse in the
                 description, O(deg^2) rather than O(2^n))
  closure.rs     journalled history + geometric closure: loops as
                 undeclared checks, retrodiction by bisecting stamps
  selfhost.rs    the self-computing object: a stack of E8 volumes whose
                 each layer holds the layers below as its own
                 coordinates, linearizing the diagonal one degree per
                 layer (a ccz becomes a single-qubit phase)
  polarity.rs    twisted polarity systems: twist rank, local sector,
                 the pairwise-locality obstruction measured
  recursive.rs   point-or-lattice sites, block-spin RG, phonon
                 substitution, self-participation fixed points
  braided.rs     braided boundary encoding: mutual (Schmidt/purification)
                 encoding across a cut, the faithful Artin action making
                 path identity decidable, periodic navigation words
                 (necklaces/Lyndon), the bracket as measured geometric
                 residue, and which realizations let the path be the
                 storage (Ising closes, Fibonacci does not)
  padic.rs       interference as congruence: a phase register over ℤ/M,
                 the CRT diagonal that factors the field into
                 single-component phases, the interference character read
                 from trailing digits, and a journalled sweep priced
                 against a closed form
  qudit.rs       hierarchical algebraic registers, dual-algebra synthesis
  exact.rs       D[ω] ring + ExactState: absolute Clifford+T reference
  guard.rs       resource guard: measured memory admission, time budgets
  gates/         GateDef trait, FixedGate/ParamGate, standard library
  registry.rs    GateRegistry<S>: validated registration, aliases
  circuit.rs     Circuit<S> (chainable builders, raw + diagonal kernels,
                 append), BoundCircuit<S> (bind-time validation, inverse())
  backend/       Backend<S> trait + dense / sparse / adaptive / factored /
                 mps / mera / bulk (dynamically scaled bulk–boundary
                 register: isometric gauge under an explicit record,
                 grow/release at runtime, certified per-depth ledger,
                 the structured unfold) / mosaic (regions in
                 heterogeneous representations: measured merges,
                 refusal-driven migration, partial graph splitting,
                 recursive composition) / bundle (graph-state) /
                 interference / device / frames / clifford_frame /
                 phase_field (exact phase polynomial over ℤ/M) /
                 braided_state (the braid word as the storage),
                 BackendRegistry<S>, pauli_expectation
  schedule.rs    evented scheduler: simultaneous loops, events, recursive
                 measurement feedback (adaptive trees)
  lift.rs        Clifford+T → measurement-feedback loop on n+t qubits
  conformance.rs registry-wide backend verification (research safety net)
  harness.rs     workload benchmarking with in-run correctness checks
  discovery.rs   point stabilizers, signal threads, transparency reports
  library.rs     bell, ghz, qft, iqft, grover, phase_flip, random_circuit,
                 brickwork, ranged_pairs, rainbow (causal workload family),
                 iqp, brickwork_2d, doped_clifford (assumption dials)
  sim.rs         Simulator<S>: registries + one-call execution
  rng.rs         deterministic xoshiro256++
tests/           eighty-two integration suites (see Testing)
benches/         criterion: gates.rs, width.rs
examples/        bell, grover, exotic_algebras, research_extension,
                 research_mode, evented_memory, width_scaling,
                 verify_models, frames_demo, clifford_space, clifford_lift,
                 coarse_register (mera + Ball), dynamic_register (the
                 scaled register: growth, verified release, streaming,
                 the certified ledger, the structured unfold),
                 scale_time (the clock qudit: conditioning, the tick,
                 scale interferometry, the four-representation payoff),
                 absolute_reference (D[ω]
                 vs every backend), adaptive_feedback (recursive trees +
                 frame repair), capacity_probe (real walls, measured),
                 device_reproduction (real geometries × latency maps),
                 causal_geometry (causal fabrics, diamonds, dual time),
                 algebraic_qudits (hierarchical register, dual-algebra),
                 qudit_scaling (measured scaling laws + honest costs),
                 advantage_bounds (the boundary atlas + advantage scan),
                 recursive_lattice (point-or-lattice sites, block-spin
                 RG, phonon substitution, self-participation),
                 e8_cube (E8 as a 2x2x2 volume, inward/outward ladder),
                 polarity_systems (twist dial, the locality obstruction),
                 polarity_bundle (fibers, chirality, budgets, 10^6 sites),
                 bundle_backend (Clifford gate action vs dense, priced),
                 geometric_closure (undeclared checks, retrodiction),
                 sampling_hardness (XEB, spoofing economics, exact refs),
                 e8_compound (mixed-arity qudits over the E8 fabric),
                 e8_coboundary (E8×E8 storage, representation, both
                 systems as backends in conformance + benchmark),
                 e8_constellation (the coset tower + n-qubit backend),
                 e8_weyl (the dual cross-scale Weyl pair, W(E8) Clifford),
                 e8_dual_scale (scale operad, cross-scale QEC, folding),
                 e8_comb_noise (logical-vs-physical error curves),
                 e8_characterization (width ceilings, gate-fidelity
                 census, perf envelope, and the measured answer to
                 whether computing across a set of E8s pays),
                 selfhosted_stack (one layer per degree, the diagonal
                 linearized, and what the trade costs),
                 gate_memoization (entry reuse, the fusion dial, and
                 branch re-exploration over a shared prefix),
                 phase_degree (the degree law, the Clifford hierarchy
                 rediscovered, and the sqrt(2) identified),
                 lateral_network (the coordinatewise link, the frame as
                 the wire, erasures vs errors, the horizon budget, and
                 the two layers crossed),
                 stitch_shards (the normal form read off a code, the
                 pairs that cannot be held in one place, the slices on
                 amplitudes, and where the exponential went),
                 magic_mosaic (what collapses in the path-sum residual,
                 what does not, and why the branching election has to
                 be measured)
```

Dependencies are deliberately light: `num-complex` and `rustc-hash` at
runtime; `proptest` and `criterion` for development.

## Where this is going

Two companion documents: [LANDSCAPES.md](LANDSCAPES.md) is the research
charter for *computations* — a staged catalog of pristine, valid quantum
implementations (hidden-subgroup on the E8 Weyl pair, semiclassical phase
estimation, resource-honest Shor, matchgate physics, stabilizer-code cycles,
CHSH certification, …), each with its honest advantage status and the
alternate representation/geometry landscapes to measure it across.
[ROADMAP.md](ROADMAP.md) tracks the *machinery*: stabilizer-rank compression for the Clifford
frame (the crude `2^t` product bound is not the ≈`2^{0.4t}` state of the
art — the gap is measurable here) and frames over factored inners (the
lift's remaining `2^t` is a *holding* cost, measured), the MERA completion
(disentanglers, path updates that replace block materialization,
ascending superoperators so operators renormalize instead of blocks —
gauge maintenance shipped as the bulk register's environment-weighted
rebuilds, with certified per-depth bounds), the scale-time clock's
remaining rungs (the native bulk tick, branching histories,
phase-pinned graph-state dynamics), per-factor and MPS-bond-gauge
frames, truncated p-adic amplitudes (the `scale`/`born_weight` split is
the designed seam), dual numbers and other non-Cayley–Dickson scalars,
classical registers for the scheduler's feedback trees, noise channels,
and gate fusion (gated on `Scalar::ASSOCIATIVE`, which is `false` from
octonions onward for a reason).
