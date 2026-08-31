# Landscapes: pristine quantum computations for a width-unbounded simulator

This document is the research charter that sits beside [ROADMAP.md](ROADMAP.md).
The ROADMAP tracks *machinery*; this file tracks *computations* — real, useful,
physically valid quantum programs, specified pristinely enough that they can be
placed on the simulator, verified, priced, and then **re-landscaped**: moved
onto alternate representations, register geometries, arities, and algebras to
find where their cost laws bend.

The governing rules are the repo's standing ones:

1. **Validity is non-negotiable.** Every entry is honest unitary + measurement
   quantum mechanics. No entry may be "made to work" by breaking the physics;
   an implementation that cannot run at some width must *refuse measurably*
   (guard) — never approximate silently, never skip by arithmetic.
2. **Advantage claims are measured or labeled.** Each entry carries a status
   from the bounds atlas vocabulary: `CLASSICAL(via …)` when a representation
   provably carries it sub-exponentially (measured law), `CANDIDATE` when every
   measured axis stays exponential, `HEURISTIC` when the speedup itself is
   conjectural, `ORACLE` when the separation is relative to a black box.
3. **Any depth ≠ any state.** A simulator "extending to any qubit depth" means:
   *structured* states at any width (measured: a 63-qubit GHZ is 224 bytes; a
   rank-1 E8 coset line at 40 qubits is 32 points), exact reference physics at
   small width, and certification in between — never a pretense that 2^n
   amplitudes fit anywhere.

**Stages** (each entry is marked):

- `SPEC` — pristinely specified here, not yet in the tree.
- `REFERENCE` — runs verified against dense/exact at small width.
- `LANDSCAPED` — a structured representation carries it past the dense wall,
  with the law measured.
- `CERTIFIED` — invariants or exact arithmetic pin correctness at widths where
  no reference exists.

---

## I. The foundation already measured in this tree

These are not aspirations; they are the load-bearing primitives the catalog
builds on, with their measured status:

| primitive | where | status |
|---|---|---|
| Full gate registry + conformance sweeps vs dense | `conformance` | REFERENCE for every backend |
| Exact D[ω] amplitudes (Clifford+T ring) | `exact` | CERTIFIED small-width anchor |
| Ball arithmetic (certified intervals) | `scalar::Ball` | CERTIFIED numerics |
| Sparse / factored / MPS / MERA / adaptive laws | `bounds` atlas | LANDSCAPED, laws measured |
| Clifford-framed states (Gottesman–Knill hybrid) | `backend` | LANDSCAPED (`size²` law measured) |
| Measurement feedback trees (adaptive schedules) | `schedule` | REFERENCE |
| Causal diamonds, dual-time resolution | `causal` | REFERENCE + measured pruning |
| Mixed-arity compound registers, E8 fabric | `mixed`, `e8` | REFERENCE + conformant backend |
| E8 constellation (any width ≤ 63), Weyl pair | `e8::constellation` | LANDSCAPED (coset states 2^{km}) |
| Sampling hardness (XEB, spoof curves) | `sampling` | measured referee |
| Resource guard (measured OOM/deadline walls) | `guard` | the honesty layer |

---

## II. The catalog

### A. Number theory and hidden structure

#### A1. Abelian hidden-subgroup instances, constellation-native — `SPEC → near`
The cleanest *provable-speedup* family (relative to the group oracle), and the
one this tree is uniquely positioned to make pristine: the hidden-coset states
that solve abelian HSP are **exactly** the coset states the E8 Weyl pair
already manufactures and interferes at 40 qubits in ~1 ms. A pristine
implementation: hide a sublattice `H ≤ (ℤ/2^m)⁸` behind an oracle
(`translate`-conditioned function), prepare the uniform coset superposition,
`coordinate_fourier` all directions, sample — the support of the momentum-side
state *is* the annihilator `H^⊥` (support uncertainty is already measured
exact). Recover generators by exact lattice linear algebra.
**Status: ORACLE (proven separation). Landscape:** rank-k instances cost
2^{km} points — sub-exponential end to end; this is the rare entry where the
structured representation carries the *whole* algorithm, not just fragments.
**Alternate landscapes:** non-binary towers (E8/3E8 digits) for odd-order
groups; W(E8)-symmetrized oracles.

#### A2. Quantum phase estimation, semiclassical (Kitaev/iterative) — `SPEC → near`
The module underneath factoring, chemistry, and amplitude estimation — and the
form that belongs on this simulator is the **one-control-qubit** iterative QPE:
each phase bit is one ancilla, one controlled power, one measurement, one
feed-forward rotation. That is precisely the `schedule` feedback machinery.
Pristine spec: eigenstate register held in whatever representation its
structure admits; ancilla measured and *recycled*; classical accumulator drives
the corrections. **Status: proven primitive (speedup inherited from caller).
Landscape:** when the eigenstate is structured (stabilizer, product, coset,
MPS-bounded), the system register never saturates — the ancilla is 1 qubit
forever. The interference cost lives entirely in `controlled-U^{2^k}`; the
honest wall is U's own landscape, which the atlas can measure per U.
**Certification:** phases of Clifford unitaries in exact D[ω]; Ball intervals
bounding bit-decision margins.

#### A3. Shor factoring, resource-honest — `SHIPPED`
Built as specified: semiclassical (Griffiths–Niu / Kitaev) phase estimation
over modular exponentiation carried as a **basis permutation**
(`shor::controlled_mul_mod` through `Backend::apply_permutation`), never as a
compiled adder. The control register collapses to one recycled ancilla, so
the run is `w + 1` qubits against the textbook `w + 2w + 1`.

**The measured law**, and it is two numbers rather than one:

* the **orbit** — distinct residues the work register holds — is exactly
  `r`, the multiplicative order, at every width from `w = 4` to `w = 20`;
* the **support** is `r` or `2r`, the factor of two being the ancilla
  holding both branches mid-round. Which regime a case lands in is a
  property of when the orbit saturates, *not* of the width: `N = 32399`
  gives `r` for base 2 and `2r` for base 3 at one and the same width.

Neither is a function of `w`. The arithmetic is free at any width — a
permutation kernel on a basis state has support 1 at a 31-bit modulus — and
the whole cost is the period, which is what should have been true.
Fitted over the ladder: time is `Polynomial{degree ≈ 1}` in `r` and
exponential in `w` only because `r` grows with `N`. Factors 15 … 1040399,
with the lucky-gcd route counted separately so period finding is not
credited with it, and the "trial division wins at these widths" footnote
measured (10⁵–10⁵·⁶×) rather than asserted.
**Landscape:** measured across **all 18 `Backend` implementors** in the
crate, not the ten in the standard registry; eight reach `w = 20`
(sparse, adaptive, mosaic, phase-field, clifford-frame, braided, logical,
framed-sparse), MPS/MERA/bulk stop at `w = 10` — a cyclic orbit is not a
low-bond object — and the graph-state bundle refuses outright.
**Past the `u64` basis index.** Every `Backend` in the crate addresses
basis states by `u64`, so the claim above — cost is `O(support)` at any
width, support is the orbit — was untestable past 63 qubits, because the
register could not address the modulus. `wide` supplies the index type
instead (`Wide`, `Montgomery`, `WideRegister`; not a `Backend`, for the
same reason `pathsum` is not), and `shor::WideOrderFinder` runs the same
loop over it. Measured: identical outcome to the `u64` path, bit for bit,
over 16 seeds on six moduli; order finding at **4124 qubits** in 13 ms and
145 KB; and the arithmetic alone — one basis state, no interference — at
**8277 qubits** in **1126 bytes**, 96 µs per modular multiplication.
**What that shows, stated narrowly.** The wide register buys
addressability, not speed: on the *same* problem (`N = 268140589`,
`r = 212784`) it returns the identical measured value and is **3.25×
slower** than `sparse`, because the keys and the arithmetic are
multi-limb. The four-figure runs are fast because `r = 256` there *by
construction* — cost is `t·r·limbs²`, and holding `r` fixed while `w`
grows is exactly the experiment "is the width free?". It is not a claim
that a 4123-bit modulus can be factored: that needs `r ≈ 2^2000`, and `r`
is the support. What is genuinely new is only that `sparse` cannot be
*constructed* past 63 qubits at all, so the question could not previously
be asked.
**Still open:** the ℤ_N-native qudit register (`mixed::CompoundRegister`)
would drop the `y ≥ N` identity branch the padded binary register carries;
`modwidth` already prices that form exactly, and `shor::ladder_widths`
exposes the forecast with its scope stated.

#### A3b. Regev factoring, and the schedule that is the whole argument — `SHIPPED`
The `d ≈ √n`-register generalization, with the classical half (LLL over the
congruence kernel) built in `lattice`. The point the implementation makes
is that the multi-register *idea* buys nothing — the naive schedule costs
`d·R` full-width multiplications, exactly Shor's bill — and that the saving
is entirely in the **schedule**. Three are built, all three verified to
compute the same permutation over the whole box and to leave no garbage:

| schedule | full-width mults | small mults | work registers |
|---|---|---|---|
| sequential | `d·R` | 0 | 1 |
| regev | `2R` | `2dR` | `R + 2` |
| fibonacci | `2K`, `K → 1.44R` | `2dK` | 3 |

`y ↦ y²` is not injective mod `N`, which is why Regev's own schedule needs a
fresh register per squaring and lands at `Õ(n^{3/2})` qubits. Replacing it
with the reversible `(x, y) ↦ (y, x·y)` costs Fibonacci-weighted exponents
(hence Zeckendorf digits, computed reversibly) and `1.44×` the full-width
multiplications, for three registers instead of `R + 2` — the crossover is
`w(R−1) > d·K` and is visible at `w = 6`.
**Honest boundaries, measured:** the Gaussian preparation Regev's analysis
needs is indistinguishable from a uniform box here in both cost and
success; the lattice weight is a real dial that is *not* binding at this
scale (8/8 at every weight from 1 to 65536) once LLL size-reduces
correctly; and Regev is cheaper in circuit multiplications while being
~10⁴× more expensive to simulate, because its exponent box is `2^{dR} ≥ N`
by construction against Shor's orbit `r`. Simulability and hardware cost
are not the same axis, and this is the crate's clearest case of it.

#### A4. Amplitude estimation (quantum Monte Carlo) — `SPEC → near`
The *useful* quadratic speedup: estimating `⟨ψ|P|ψ⟩`-type quantities with 1/ε
instead of 1/ε² samples. Iterative AE (no QFT register, Grover powers +
maximum-likelihood on measurement records) fits the schedule machinery like A2.
**Status: proven quadratic (unconditional, oracle-relative). Landscape:**
Grover powers of structured oracles stay structured (measured already for the
library Grover); the honest study is *ε-scaling vs width-scaling* measured
jointly — an atlas family with two axes.

### B. Physics simulation (the original purpose)

#### B1. Trotterized spin-chain dynamics — `REFERENCE → LANDSCAPED`
Heisenberg/TFIM quenches under brickwork Trotter layers. Largely present
(`library::brickwork`, MPS bond-law measured: bond doubles per *pair* of brick
layers). The pristine upgrade: (i) physical observables as the deliverable —
magnetization/correlation profiles via `pauli_expectation` at every step;
(ii) **conserved-quantity certification** at widths beyond dense: total-Sz and
energy drift bounded with Ball arithmetic — correctness evidence with no
reference state; (iii) entanglement-growth laws (linear after quench ⟹
measured MPS wall arrival time predicted, then observed). **Status:
CLASSICAL(via mps) at low entanglement, CANDIDATE past the light-cone
saturation — the crossover itself is the scientific output.**

#### B2. Free fermions / matchgate circuits — `SPEC` (backend rung)
A large, physically real family (tight-binding dynamics, Kitaev chains, BdG
quasiparticles) with a *known* polynomial landscape: matchgate circuits reduce
to 2n×2n covariance-matrix evolution. This is the highest-value missing
representation in the registry: a `matchgate` backend makes an entire physics
domain CLASSICAL(via covariance) with certification against dense at small n —
and doping it with non-matchgate gates gives a second measured
"cost-of-magic" family parallel to the Clifford T-doping study already pinned.
**Status: CLASSICAL(via matchgates), measured once built.**

#### B3. Lattice gauge / fermion encodings — `SPEC`
Jordan-Wigner vs local encodings (Bravyi–Kitaev, compact encodings) as
*register-geometry experiments*: the same Hubbard-model Trotter step, measured
across encodings for locality (causal-cone width through `causal`), swap
overhead (device model), and representation cost. The deliverable is a
measured table nobody has for this simulator's geometries — encoding choice as
an empirical landscape question, on the machinery built for exactly that.

#### B4. Variational eigenstates (VQE as spec, not hype) — `SPEC`
**Status: HEURISTIC — labeled so.** Value here is not "quantum advantage" but
the landscape coupling: ansatz depth ↔ bond growth ↔ representation cost,
measured; plus gradient evaluation via parameter-shift on the exact registry.
A pristine VQE entry is an *instrument* for studying trainability landscapes
(barren-plateau variance measured across widths), not a chemistry claim.

### C. Search, walks, optimization

#### C1. Quantum walks & the glued-trees separation — `SPEC`
Continuous/discrete walks where **register geometry is the algorithm**: the
walk graph is a `Topology`, hitting distributions are the measurement record.
Glued trees is the pristine oracle-exponential example. The landscape story:
walk states on low-expansion graphs stay sparse for long times (measurable
law: support growth vs steps vs geometry), and the causal-cone machinery
prunes to the observation's diamond. **Status: ORACLE (glued trees), otherwise
polynomial-advantage instances; a geometry-first family this tree is built
for.**

#### C2. Grover on structured predicates — `REFERENCE`
Present. The pristine extension: amplitude amplification as a *library
combinator* (any state-prep + any predicate), with the measured law of oracle
structure vs support growth — Grover on a sparse predicate is the canonical
demonstration that "quadratic speedup" and "sub-exponential representation"
can coexist, ending in a measured crossover table.

#### C3. QAOA on MaxCut — `SPEC`, **HEURISTIC**
Include for honesty: depth-p QAOA over library graph families, the
p-vs-approximation-ratio surface measured, IQP-adjacency noted (the atlas
already measured the IQP range flip). No advantage claim; the value is the
measured landscape and its geometry dependence (ring vs expander vs 2D — all
existing topologies).

### D. Sampling and the advantage frontier

#### D1. Random-circuit sampling with XEB — `REFERENCE/measured`
Present end to end (XEB, exact references, spoof economics, verdict:
CANDIDATE on every measured axis). Pristine upgrades: patch/elided-circuit
spoof baselines as *measured* spoof-curve entries beside the MPS truncation
cliff already pinned.

#### D2. IQP sampling — `REFERENCE/measured`
Present with the honest range-flip verdict (long-range IQP escapes the
measured classical axes; short-range does not). Keep as the canonical example
that *one knob* moves a family across the frontier — the clearest "alternate
landscape" object lesson in the tree.

### E. Error correction and the fault-tolerant future

#### E1. Stabilizer-code cycles as feedback programs — `SPEC → near`
Repetition and small surface-code patches, written as `schedule` programs:
syndrome extraction rounds = ancilla measure + classical decode + conditional
correction — all machinery present (`CliffordFramedState` carries the
stabilizer bulk at one amplitude; frame repair on measurement is shipped).
Deliverable: logical-error-rate vs physical-error-rate curves under injected
Pauli noise (trajectory sampling), *measured* on code distance — the
simulator's first genuine architecture experiment. **Status: CLASSICAL(via
frames) for Clifford noise — which is exactly why it can run at real
distances; T-doped noise re-enters the measured magic-cost story.**

#### E2. Magic-state distillation economics — `SPEC`
15-to-1 (and friends) as circuits; the measured deliverable is the *cost
ledger*: T-count in vs fidelity out vs frame-representation cost — connecting
the shipped doped-Clifford scatter laws to the actual protocol that motivates
them.

#### E3. Lattice-code landscapes (the E8/GKP bridge) — `first rung SHIPPED`
The Weyl-pair work gives this tree a discrete cross-scale cousin of
symplectic-lattice GKP codes: stabilizer groups inside the constellation
Heisenberg group over (ℤ/2^m)⁸, with W(E8) as native Cliffords. **Shipped
(`tests/e8_dual_scale.rs`): cross-scale comb codes** — coarse-translation +
fine-modulation checks commuting past the horizon, logical operators at the
middle scales, syndrome windows tiling exactly the bidirectional commutation
inequality (one measured matrix), end-to-end displacement correction by exact
phase readout, and code self-similarity under decimation (the RG flow of the
code is the code). Also shipped beside it: the folding theorems (interleaved
bidirectional sequences collapsing to a single two-gate layer; deep periodic
time folded to its measured period with the exact quadratic Weyl phase).
**Second rung shipped (`e8_comb_noise`): the logical-vs-physical error
curves** — the first fault-tolerance architecture experiment: seeded
displacement noise with per-round syndrome-decode-correct cycles, logical
failure measurably suppressed as the comb scale grows (threshold-shaped
curves across five physical rates), window-sized displacements measured as
the code distance, every trajectory priced at constant 256-point support.
Remaining research: physical-ancilla syndrome extraction, the mod-2^m
stabilizer tableau, and distance-vs-cost against qubit codes of equal
length.

### F. Protocols: communication, nonlocality, verification

#### F1. CHSH/Bell certification — `SPEC → near`
Tiny, pristine, and load-bearing: prepare the singlet, evaluate the CHSH value
exactly via `pauli_expectation` (2√2 in exact arithmetic via D[ω]/Ball), then
*sample* finite-shot experiments and watch the bound approach — the
self-testing primitive under all device-independent futures, and a permanent
conformance anchor (any backend that mis-simulates entanglement fails CHSH
before anything else).

#### F2. Teleportation / entanglement-swapping chains — `SPEC → near`
Feedback-driven (X/Z corrections conditioned on Bell measurements) across
repeater chains of rainbow pairs — the schedule machinery again. Deliverable:
fidelity-vs-chain-length under injected noise; the network-architecture
primitive, measured. The co-boundary storage protocol (shipped) already plays
the role of an exotic quantum-memory cell beside it.

---

## III. What "any qubit depth" means as architecture

The honest architecture, distilled from everything measured so far:

1. **Representation is a scheduling decision.** The registry + `select_by_scaling`
   already choose backends by *measured extrapolated law with holdout
   verification*. The future: every catalog entry ships a **landscape
   certificate** — its measured law per representation, its verdict, and the
   width at which each wall arrives on the current machine. Programs declare
   structure; the runtime routes to the representation whose measured law
   admits it; refusals are measured, never silent.

2. **Verification is layered, and never ends.** Dense/exact anchors at small
   width → cross-representation agreement at medium width (two independent
   structured backends agreeing is evidence *neither* alone provides) →
   invariant certification at any width (conserved quantities, stabilizer
   checks, Ball-bounded drift, group-theoretic identities like the Weyl
   commutation laws). A computation without a certification story is not
   pristine — that is this document's admission bar.

3. **Structure is the commodity.** Every measured success in this tree —
   frames at one amplitude, coset states at 2^{km}, MPS at bounded bond,
   factored products, permutation registers — is one theme: *the state's
   symmetry group carried explicitly instead of its amplitudes*. The research
   direction the catalog keeps converging on: representations indexed by
   groups (Pauli, matchgate, lattice Weyl, permutation), with "magic" =
   measured cost of leaving the group. The atlas is the referee; the doped
   families are the meters.

4. **Feedback is first-class.** The semiclassical forms (A2, A3, A4, E1, F2)
   all trade register width for measurement and classical control — the single
   most architecture-relevant transformation in the catalog, and it is already
   native here (`schedule`, frame repair, recursive feedback).

5. **Geometry is a free parameter, so measure it.** Causal fabrics, device
   latency maps, mixed arities, lattice towers: every entry above gains an
   axis by re-asking its question on a different register geometry. "Alternate
   landscapes" is precisely this: same physics, different carrier, measured
   delta.

---

## IV. Staging order (proposed)

Near-term, each with acceptance criteria in the repo's measured style:

1. **A1 abelian HSP, constellation-native** — components shipped; end-to-end
   oracle + generator recovery + measured 2^{km} cost at n = 40.
2. **A2/A4 iterative QPE + amplitude estimation** — schedule-driven, one
   recycled ancilla, Ball-certified bit margins; Clifford-phase cases exact.
3. **F1 CHSH + F2 teleportation chains** — small, permanent certification
   anchors; exact 2√2.
4. **E1 repetition→surface cycles** — logical-vs-physical error curves on
   clifford-framed, trajectory noise.
5. **B2 matchgate backend** — the biggest single landscape addition; then the
   doped-matchgate cost family beside the Clifford one.
6. ~~**A3 Shor, resource-honest**~~ — shipped, with Regev beside it; the
   remaining rung is the ℤ_N-native qudit register and the interference
   step as its own atlas axis.
7. **E3 mod-2^m stabilizer tableau** — turns the Weyl pair's 8th-root
   compression into a polynomial representation; W(E8) as its Clifford group.

Each lands the same way everything here lands: spec → reference → measured
law → honest verdict — and every wall it hits must be a *measured* wall.
