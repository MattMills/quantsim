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

`cargo test` runs 371 tests (46 unit + 320 across thirty-six
integration suites + 5 doctests; one more — the 17 s measurement that
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
  bounds.rs      boundary atlas: measured growth laws, advantage scan
  sampling.rs    sampling-task hardness: XEB, spoof curves, exact refs
  mixed.rs       mixed-arity compound qudits: volumes, fabric, flow, backend
  e8.rs          E8 roots built+verified; chains, rep, constellation,
                 cube (the 2x2x2 volume, inward/outward interaction)
  causal.rs      backward cones, causal diamonds, dual-time resolution
  recursive.rs   point-or-lattice sites, block-spin RG, phonon
                 substitution, self-participation fixed points
  qudit.rs       hierarchical algebraic registers, dual-algebra synthesis
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
  library.rs     bell, ghz, qft, iqft, grover, phase_flip, random_circuit,
                 brickwork, ranged_pairs, rainbow (causal workload family),
                 iqp, brickwork_2d, doped_clifford (assumption dials)
  sim.rs         Simulator<S>: registries + one-call execution
  rng.rs         deterministic xoshiro256++
tests/           thirty-six integration suites (see Testing)
benches/         criterion: gates.rs, width.rs
examples/        bell, grover, exotic_algebras, research_extension,
                 research_mode, evented_memory, width_scaling,
                 verify_models, frames_demo, clifford_space, clifford_lift,
                 coarse_register (mera + Ball), absolute_reference (D[ω]
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
                 sampling_hardness (XEB, spoofing economics, exact refs),
                 e8_compound (mixed-arity qudits over the E8 fabric),
                 e8_coboundary (E8×E8 storage, representation, both
                 systems as backends in conformance + benchmark),
                 e8_constellation (the coset tower + n-qubit backend),
                 e8_weyl (the dual cross-scale Weyl pair, W(E8) Clifford),
                 e8_dual_scale (scale operad, cross-scale QEC, folding),
                 e8_comb_noise (logical-vs-physical error curves)
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
(disentanglers, path updates that replace block materialization, gauge
maintenance so truncation bounds certify, ascending superoperators so
operators renormalize instead of blocks), per-factor and MPS-bond-gauge
frames, truncated p-adic amplitudes (the `scale`/`born_weight` split is
the designed seam), dual numbers and other non-Cayley–Dickson scalars,
classical registers for the scheduler's feedback trees, noise channels,
and gate fusion (gated on `Scalar::ASSOCIATIVE`, which is `false` from
octonions onward for a reason).
