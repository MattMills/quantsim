# Roadmap

The crate is organized around two swappable axes — the **amplitude algebra**
(`Scalar`) and the **state representation** (`Backend<S>`) — so most planned
work is "fill in another cell of the matrix":

| representation \ algebra | ℝ | ℂ | split-ℂ | ℍ | 𝕆 | 𝕊 | ℚ_p |
|--------------------------|---|---|---------|---|---|---|-----|
| dense state vector       | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | design below |
| sparse state vector      | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | design below |
| adaptive (sparse→dense)  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| matrix product state     | design below | design below | — | open | open | — | open |

## Matrix product states (MPS)

The `Backend<S>` trait was shaped with MPS in mind:

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

## Simulator features

- Mid-circuit measurement **as circuit operations** (classical registers and
  feed-forward), rather than only via the `Backend::measure` API.
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
- Stabilizer/Clifford fast path: valuable, but it is a *third* simulation
  paradigm (group-theoretic, not amplitude-based); it deserves its own
  design pass against the `Backend` trait rather than a bolt-on.
