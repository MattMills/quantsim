//! Running the order-finding loop **sharded**, and measuring what
//! crosses between shards.
//!
//! Run with `cargo run --release --example scaling_limit`.
//!
//! The cost law is `time = O(t·r)`, `memory = O(r)` — polynomial in both
//! variables it is stated in. Whether that survives being spread over
//! machines is a question about communication, and the way to answer it
//! is to partition the state and run, not to multiply a bandwidth figure
//! by a budget.
//!
//! So: the support is partitioned across `S` shards by a generic hash of
//! the work value, the whole semiclassical loop runs on the partition,
//! and every byte that has to move between shards is counted. A
//! `SparseState` runs beside it as the reference and supplies the
//! measurement outcomes, and the sharded state is asserted equal to it
//! amplitude for amplitude after **every** round — so the traffic
//! reported is the traffic of a run that provably did the same thing.
//!
//! Not measured here: the network itself. This counts bytes that must
//! cross, and says nothing about how fast a given fabric would move them.

use rustc_hash::FxHashMap;

use quantsim::backend::Backend;
use quantsim::prelude::*;
use quantsim::shor::{self, OrderFinder, PhaseForm};

/// The state, partitioned. Shards hold disjoint sets of basis indices.
struct Sharded {
    shards: Vec<FxHashMap<u64, C64>>,
    work: Vec<usize>,
}

/// A generic hash sharding of the work value. No property of the group
/// is used, because none is available before the period is known.
fn shard_of(value: u64, shards: usize) -> usize {
    let mut h = value.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h % shards as u64) as usize
}

/// Bytes for one routed entry: the basis index plus the amplitude.
const ENTRY_BYTES: usize = std::mem::size_of::<u64>() + std::mem::size_of::<C64>();

impl Sharded {
    fn new(count: usize, work: Vec<usize>) -> Self {
        Sharded {
            shards: (0..count).map(|_| FxHashMap::default()).collect(),
            work,
        }
    }

    fn insert(&mut self, index: u64, amp: C64) {
        let s = shard_of(shor::gather(index, &self.work), self.shards.len());
        self.shards[s].insert(index, amp);
    }

    fn len(&self) -> usize {
        self.shards.iter().map(|s| s.len()).sum()
    }

    /// Largest shard over smallest non-empty — the load imbalance a real
    /// deployment would actually feel.
    fn imbalance(&self) -> f64 {
        let sizes: Vec<usize> = self.shards.iter().map(|s| s.len()).collect();
        let max = *sizes.iter().max().unwrap_or(&0) as f64;
        let mean = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        if mean > 0.0 {
            max / mean
        } else {
            1.0
        }
    }

    /// Apply a permutation. Entries whose image hashes elsewhere must be
    /// routed; the routed count is returned.
    fn permute(&mut self, map: &dyn Fn(u64) -> u64) -> usize {
        let count = self.shards.len();
        let mut outbox: Vec<Vec<(u64, C64)>> = (0..count).map(|_| Vec::new()).collect();
        let mut crossed = 0usize;
        for (home, shard) in self.shards.iter_mut().enumerate() {
            let old = std::mem::take(shard);
            for (i, a) in old {
                let j = map(i);
                let dest = shard_of(shor::gather(j, &self.work), count);
                if dest == home {
                    shard.insert(j, a);
                } else {
                    outbox[dest].push((j, a));
                    crossed += 1;
                }
            }
        }
        for (dest, batch) in outbox.into_iter().enumerate() {
            for (j, a) in batch {
                self.shards[dest].insert(j, a);
            }
        }
        crossed
    }

    /// A one-qubit gate on `qubit`. Partners differ only in that bit; when
    /// the qubit is outside the sharded field, both live on the same
    /// shard and nothing moves. Returns the routed count, which is the
    /// point.
    fn apply_1q(&mut self, m: &quantsim::math::GateMatrix<C64>, qubit: usize) -> usize {
        let sharded_field = self.work.contains(&qubit);
        let count = self.shards.len();
        let mut crossed = 0usize;
        for shard in self.shards.iter_mut() {
            let old = std::mem::take(shard);
            let mut pairs: FxHashMap<u64, [C64; 2]> = FxHashMap::default();
            for (i, a) in old {
                let bit = (i >> qubit) & 1 == 1;
                let base = i & !(1u64 << qubit);
                pairs.entry(base).or_insert([C64::new(0.0, 0.0); 2])[bit as usize] = a;
            }
            for (base, [a0, a1]) in pairs {
                let o0 = m.get(0, 0) * a0 + m.get(0, 1) * a1;
                let o1 = m.get(1, 0) * a0 + m.get(1, 1) * a1;
                if o0.abs_sqr() > 0.0 {
                    shard.insert(base, o0);
                }
                if o1.abs_sqr() > 0.0 {
                    shard.insert(base | (1u64 << qubit), o1);
                }
            }
        }
        if sharded_field {
            // Would need routing; not used by this loop, but counted
            // honestly rather than assumed away.
            crossed = self.len();
        }
        let _ = count;
        crossed
    }

    /// Collapse on `qubit` given an outcome decided elsewhere. Needs one
    /// all-reduce of two scalars for the norm; everything else is local.
    fn project(&mut self, qubit: usize, outcome: bool) -> usize {
        let mut total = 0.0f64;
        for shard in &self.shards {
            for (i, a) in shard.iter() {
                if ((i >> qubit) & 1 == 1) == outcome {
                    total += a.abs_sqr();
                }
            }
        }
        let renorm = 1.0 / total.sqrt();
        for shard in self.shards.iter_mut() {
            shard.retain(|i, _| ((i >> qubit) & 1 == 1) == outcome);
            for a in shard.values_mut() {
                *a *= C64::new(renorm, 0.0);
            }
        }
        // One f64 per shard, reduced.
        self.shards.len() * std::mem::size_of::<f64>()
    }

    fn matches(&self, reference: &dyn Backend<C64>) -> bool {
        let mut mine: Vec<(u64, C64)> = self
            .shards
            .iter()
            .flat_map(|s| s.iter().map(|(&i, &a)| (i, a)))
            .collect();
        let mut theirs: Vec<(u64, C64)> = Vec::new();
        reference.for_each_nonzero(&mut |i, a| theirs.push((i, a)));
        mine.sort_by_key(|e| e.0);
        theirs.sort_by_key(|e| e.0);
        if mine.len() != theirs.len() {
            return false;
        }
        mine.iter().zip(theirs.iter()).all(|(a, b)| {
            a.0 == b.0 && (a.1.re - b.1.re).abs() < 1e-12 && (a.1.im - b.1.im).abs() < 1e-12
        })
    }
}

struct Traffic {
    rounds: usize,
    permute_entries: usize,
    gate_entries: usize,
    reduce_bytes: usize,
    imbalance: f64,
    support: usize,
    /// Routed entries per round, in round order.
    per_round: Vec<usize>,
    /// Support entering each round.
    support_per_round: Vec<usize>,
}

/// Run the whole semiclassical loop sharded, against a `SparseState`
/// reference that supplies the measurement outcomes.
fn run_sharded(sim: &Simulator, n: u64, a: u64, shards: usize, seed: u64) -> Result<Traffic> {
    let w = shor::work_bits(n);
    let t = 2 * w + 1;
    let work: Vec<usize> = (0..w).collect();
    let anc = w;

    let mut reference = sim.backends().create("sparse", w + 1)?;
    sim.apply(reference.as_mut(), "x", &[], &[work[0]])?;
    let mut sharded = Sharded::new(shards, work.clone());
    sharded.insert(1, C64::new(1.0, 0.0));

    let k = std::f64::consts::FRAC_1_SQRT_2;
    let h = quantsim::math::GateMatrix::<C64>::from_vec(
        2,
        vec![
            C64::new(k, 0.0),
            C64::new(k, 0.0),
            C64::new(k, 0.0),
            C64::new(-k, 0.0),
        ],
    )?;

    let mut rng = Prng::new(seed);
    let mut omega = 0.0f64;
    let mut traffic = Traffic {
        rounds: t,
        permute_entries: 0,
        gate_entries: 0,
        reduce_bytes: 0,
        imbalance: 1.0,
        support: 0,
        per_round: Vec::new(),
        support_per_round: Vec::new(),
    };

    for j in (1..=t).rev() {
        sim.apply(reference.as_mut(), "h", &[], &[anc])?;
        traffic.gate_entries += sharded.apply_1q(&h, anc);

        let mult = shor::pow_mod(a, 1u64 << (j - 1), n);
        shor::controlled_mul_mod(reference.as_mut(), Some(anc), &work, mult, n)?;
        let before = sharded.len();
        let routed = sharded.permute(&|i| {
            if (i >> anc) & 1 == 0 {
                return i;
            }
            let y = shor::gather(i, &work);
            if y >= n {
                return i;
            }
            shor::scatter(i, &work, shor::mul_mod(mult, y, n))
        });
        traffic.permute_entries += routed;
        traffic.per_round.push(routed);
        traffic.support_per_round.push(before);
        // Imbalance at the round that actually holds the state: taking a
        // max over all rounds reports the opening rounds, where the
        // support is 1 and every shard but one is empty by construction.
        if sharded.len() >= traffic.support {
            traffic.support = sharded.len();
            traffic.imbalance = sharded.imbalance();
        }

        // Feedback rotation then Hadamard, both on the ancilla.
        let theta = -std::f64::consts::PI * omega;
        let e = quantsim::math::cis(theta);
        let hp = quantsim::math::GateMatrix::<C64>::from_vec(
            2,
            vec![
                C64::new(k, 0.0),
                C64::new(k * e.re, k * e.im),
                C64::new(k, 0.0),
                C64::new(-k * e.re, -k * e.im),
            ],
        )?;
        sim.apply(reference.as_mut(), "p", &[theta], &[anc])?;
        sim.apply(reference.as_mut(), "h", &[], &[anc])?;
        traffic.gate_entries += sharded.apply_1q(&hp, anc);

        let bit = reference.measure(anc, &mut rng)?;
        traffic.reduce_bytes += sharded.project(anc, bit);
        if bit {
            sim.apply(reference.as_mut(), "x", &[], &[anc])?;
            traffic.permute_entries += sharded.permute(&|i| i ^ (1u64 << anc));
        }
        omega = (if bit { 1.0 } else { 0.0 } + omega) * 0.5;

        assert!(
            sharded.matches(reference.as_ref()),
            "the sharded run diverged from the reference at round {j}"
        );
    }
    Ok(traffic)
}

fn main() -> Result<()> {
    let sim: Simulator = Simulator::new();

    println!("== the loop, actually sharded, verified against a reference every round ==");
    println!("   N = 126727 (w = 17), a = 2, r = 15752");
    println!(
        "  {:>7} {:>10} {:>14} {:>14} {:>12} {:>10}",
        "shards", "peak supp", "routed entries", "routed/round", "reduce bytes", "imbalance"
    );
    let (n, a) = (126_727u64, 2u64);
    for shards in [1usize, 2, 4, 16, 64, 256, 1024] {
        let tr = run_sharded(&sim, n, a, shards, 7)?;
        println!(
            "  {shards:>7} {:>10} {:>14} {:>14.0} {:>12} {:>10.2}",
            tr.support,
            tr.permute_entries,
            tr.permute_entries as f64 / tr.rounds as f64,
            tr.reduce_bytes,
            tr.imbalance
        );
    }

    println!();
    println!("== the same, as a fraction of the state, and in bytes ==");
    println!(
        "  {:>7} {:>16} {:>18} {:>18}",
        "shards", "routed/support", "routed bytes/run", "local gate routing"
    );
    for shards in [2usize, 16, 256, 1024] {
        let tr = run_sharded(&sim, n, a, shards, 7)?;
        println!(
            "  {shards:>7} {:>16.4} {:>18} {:>18}",
            tr.permute_entries as f64 / (tr.support * tr.rounds) as f64,
            tr.permute_entries * ENTRY_BYTES,
            tr.gate_entries
        );
    }

    println!();
    println!("== where the traffic is: routed entries by round, 256 shards ==");
    let tr = run_sharded(&sim, n, a, 256, 7)?;
    println!(
        "  {:>6} {:>12} {:>10} {:>10}",
        "round", "support in", "routed", "routed/in"
    );
    for (k, (&routed, &supp)) in tr
        .per_round
        .iter()
        .zip(tr.support_per_round.iter())
        .enumerate()
        .filter(|(k, _)| *k >= tr.per_round.len() - 12)
    {
        println!(
            "  {k:>6} {supp:>12} {routed:>10} {:>10.4}",
            routed as f64 / supp.max(1) as f64
        );
    }
    let tail: usize = tr.per_round.iter().rev().take(4).sum();
    println!(
        "  last 4 rounds carry {}/{} = {:.1}% of all routed entries",
        tail,
        tr.permute_entries,
        100.0 * tail as f64 / tr.permute_entries as f64
    );

    println!();
    println!("== how the routed fraction moves with r, at 256 shards ==");
    println!(
        "  {:>10} {:>8} {:>10} {:>16} {:>16}",
        "N", "w", "r", "routed/round", "routed/support"
    );
    for (n, a) in [
        (4087u64, 7u64),
        (14351, 5),
        (64507, 3),
        (126727, 2),
        (1040399, 2),
    ] {
        let f = OrderFinder::new(n, a)?;
        let e = f.estimate(&sim, PhaseForm::Semiclassical, &mut Prng::new(7))?;
        let tr = run_sharded(&sim, n, a, 256, 7)?;
        println!(
            "  {n:>10} {:>8} {:>10} {:>16.0} {:>16.4}",
            f.work_bits(),
            e.peak_orbit,
            tr.permute_entries as f64 / tr.rounds as f64,
            tr.permute_entries as f64 / (tr.support * tr.rounds) as f64
        );
    }
    Ok(())
}
