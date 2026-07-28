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

#### A3. Shor factoring, resource-honest — `SPEC`
Not a toy demo: the pristine spec is semiclassical QPE (A2) over modular
exponentiation, with the honest decomposition of where quantumness lives.
Modular-exponentiation circuits are **basis permutations** — on a basis state
they are classical reversible arithmetic, and a sparse/permutation
representation carries the work register at *any* width with support = 1 until
the QFT interference begins; the superposed control collapses to one recycled
ancilla in the semiclassical form. The irreducibly quantum content is the
phase interference over the period — which is where the measured wall will
appear, and *should*. **Status: proven speedup (vs known classical).
Landscape targets:** permutation-kernel backend (ROADMAP already names
permutation kernels); factor 15/21/35 end-to-end CERTIFIED in exact D[ω] where
gates permit; measure the law of the interference step as its own atlas axis.
**Alternate landscape:** residue-tower registers — phase estimation over
(ℤ/2^m)⁸ meets the constellation group natively.

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

#### E3. Lattice-code landscapes (the E8/GKP bridge) — `SPEC / research`
The Weyl-pair work gives this tree a discrete cross-scale cousin of
symplectic-lattice GKP codes: stabilizer groups inside the constellation
Heisenberg group over (ℤ/2^m)⁸, with W(E8) as native Cliffords. Pristine
research target: define a mod-2^m stabilizer tableau (ROADMAP rung), exhibit a
small code whose logicals are W(E8) orbits, and measure its distance-vs-cost
against qubit codes of equal length. **Status: open research — the entry
exists to keep it honest and staged.**

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
6. **A3 Shor, resource-honest** — permutation kernels + semiclassical QFT;
   15/21 certified exactly; the interference step as a new atlas axis.
7. **E3 mod-2^m stabilizer tableau** — turns the Weyl pair's 8th-root
   compression into a polynomial representation; W(E8) as its Clifford group.

Each lands the same way everything here lands: spec → reference → measured
law → honest verdict — and every wall it hits must be a *measured* wall.
