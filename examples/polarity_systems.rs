//! Systems of `n` twisted polarities: what the twist buys, what it
//! costs, and exactly where pairwise-local polarity data stops being
//! able to hold a quantum state.
//!
//! Run with `cargo run --release --example polarity_systems`.

use quantsim::polarity::*;
use quantsim::prelude::*;

fn main() -> Result<()> {
    // ── 1. The twist dial ────────────────────────────────────────────
    println!("== the twist dial: eight polarities, rank 0 → 8 ==");
    println!(
        "  {:>7} {:>6} {:>9} {:>12} {:>12} {:>14}",
        "pairs", "rank", "centre", "local bits", "isotropic", "matrix block"
    );
    for pairs in 0..=4 {
        let s = PolaritySystem::partial(8, pairs)?.structure();
        println!(
            "  {:>7} {:>6} {:>9} {:>12} {:>12} {:>14}",
            pairs, s.twist_rank, s.center_dimension, s.local_bits, s.isotropic_size, s.matrix_block
        );
    }
    println!("  isotropic × matrix block = 256 on every row: the algebra splits into a part");
    println!("  storable as independent sign bits and a part that is not. The twist rank is");
    println!("  the price of non-locality, and it is a dial.");

    // ── 2. Where quantum states sit on that dial ─────────────────────
    println!();
    println!("== the Pauli group as a polarity system ==");
    println!(
        "  {:>7} {:>12} {:>6} {:>12} {:>12} {:>14}",
        "qubits", "polarities", "rank", "centre", "local bits", "isotropic"
    );
    for n in 1..=6 {
        let s = PolaritySystem::pauli(n)?.structure();
        println!(
            "  {:>7} {:>12} {:>6} {:>12} {:>12} {:>14}",
            n, s.polarities, s.twist_rank, s.center_dimension, s.local_bits, s.isotropic_size
        );
    }
    println!("  maximally twisted, trivial centre — and a maximal commuting subgroup of 2^n");
    println!("  elements described by n sign bits. That subgroup IS a stabilizer group, and");
    println!("  those n bits are the O(n²) description Gottesman–Knill runs on (this crate");
    println!("  ships it as CliffordFramedState). The locally-storable sector is real, it is");
    println!("  large, and it is already the best-known classical island.");

    // ── 3. The obstruction ───────────────────────────────────────────
    println!();
    println!("== can pairwise-local data hold a state? ==");
    println!("  (|0…0⟩ + |1…1⟩)/√2 against (|0…0⟩ − |1…1⟩)/√2 — orthogonal pure states)");
    println!(
        "  {:>4} {:>10} {:>10} {:>14} {:>10} {:>18} {:>7}",
        "n", "pairwise", "state", "sig deviation", "overlap", "⟨X^⊗n⟩  + / −", "blind"
    );
    for n in 2..=8 {
        let o = ghz_sign_obstruction(n)?;
        println!(
            "  {:>4} {:>10} {:>10} {:>14.2e} {:>10.2e} {:>8.3} /{:>8.3} {:>7}",
            o.num_qubits,
            o.pairwise_values,
            o.state_values,
            o.signature_deviation,
            o.overlap,
            o.global_correlator.0,
            o.global_correlator.1,
            o.pairwise_blind
        );
    }
    println!();
    println!("  At two qubits ⟨XX⟩ separates them and pairwise data is enough. From three up");
    println!("  every one- and two-body Pauli expectation agrees EXACTLY while the states stay");
    println!("  orthogonal: no pairwise-local representation can tell them apart. The only");
    println!("  observable that does is the n-body correlator.");
    println!();
    println!("  And it is not a shortage of numbers. At n = 3 the pairwise signature holds 36");
    println!("  reals against the state vector's 16 — more data, still blind. The failure is");
    println!("  in what pairwise data can express, not how much of it there is.");
    println!();
    for n in [3usize, 6] {
        println!(
            "  GHZ-{n} stabilizer generator weights: {:?}",
            stabilizer_weight_profile(n)?
        );
    }
    println!("  Both GHZ signs are stabilizer states, so this is not pairwise data failing on");
    println!("  something exotic — it fails inside the sector that IS efficiently describable.");
    println!("  The reason is the weight-n generator: the description is O(n²) in size but not");
    println!("  pairwise in structure, and dropping that one generator leaves two orthogonal");
    println!("  states sharing a description.");

    // ── 4. Polarity<N> as an amplitude, priced ───────────────────────
    println!();
    println!("== N polarities inside one amplitude ==");
    println!(
        "  {:>10} {:>5} {:>7} {:>9} {:>18}",
        "algebra", "dim", "gates", "commut.", "dense mem @ n = 12"
    );
    fn row<S: Scalar>() {
        let mem = DenseState::<S>::new(12).unwrap().memory_bytes();
        println!(
            "  {:>10} {:>5} {:>7} {:>9} {:>18}",
            S::algebra_name(),
            S::DIM,
            GateRegistry::<S>::standard().names().len(),
            S::COMMUTATIVE,
            mem
        );
    }
    row::<C64>();
    row::<Polarity<1>>();
    row::<Polarity<2>>();
    row::<Polarity<3>>();
    row::<Polarity<4>>();
    println!("  (storage is padded to 2^MAX_POLARITIES slots — const-generic arrays cannot be");
    println!("   sized 2^N on stable — so the footprint column is flat while the algebraic");
    println!("   dimension is not. SplitQuaternion is the exactly-sized two-polarity case.)");
    println!();
    println!("  One polarity has no i, so it carries the real gate subset only; from two up");
    println!("  j₀j₁ squares to −1 and the full library exists. But the honest headline is the");
    println!("  dim column: N polarities per amplitude is 2^N reals per amplitude. Carrying one");
    println!("  polarity per qubit of an n-qubit register costs 2^n PER STORED AMPLITUDE — the");
    println!("  exponential does not disappear, it moves from the register into the scalar.");

    let state = Simulator::<Polarity<3>>::new().run(&library::ghz(4))?;
    println!();
    println!(
        "  GHZ-4 over Pol3 all the same: p(0) = {:.6}, p(15) = {:.6}, total = {:.6}",
        state.probability(0),
        state.probability(15),
        state.total_weight()
    );
    Ok(())
}
