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
//!
//! ## The ceiling, found by running into it
//!
//! Climbing the width with the *largest-order* base at each one — the
//! honest worst case — this machine (16 GB, 4 cores) completes
//! `N = 268140589` (**28 bits**, `r = 22342320`) in **119 s and 1.5 GB**,
//! and fails `N = 1073217479` (30 bits, `r = 536575980`) on a 240 s
//! budget. Time and memory arrive together at 29–30 bits.
//!
//! Three constants come out of that, all measured, and they are what any
//! larger machine has to be reasoned about with:
//!
//! * **69 bytes** per stored amplitude at the ceiling,
//! * **~95 ns** per entry-round,
//! * `r ≈ N/4` for bases that actually split `N` — measured
//!   `log₂(N/r) = 2.0` across widths 16 to 24.
//!
//! Which gives `n_max = log₂(M / 69) + 2` for a memory budget `M`. Every
//! input is measured; the step to a larger `M` is arithmetic on measured
//! constants, not a model. Each 1000× of memory buys ten bits: 28 here,
//! ~38 on a rack, ~48 on a 10 PB datacenter.

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

fn semiprime(w: usize) -> Option<(u64, u64, u64)> {
    if !(4..=62).contains(&w) {
        return None;
    }
    let (lo, hi) = (1u64 << (w - 1), (1u64 << w) - 1);
    let mut p = (hi as f64).sqrt() as u64 + 1;
    while p >= 2 {
        if shor::is_prime(p) {
            let mut q = p + 1;
            while p.checked_mul(q).is_some_and(|n| n <= hi) {
                if shor::is_prime(q) && p * q >= lo {
                    return Some((p * q, p, q));
                }
                q += 1;
            }
        }
        p -= 1;
    }
    None
}

fn factor_small(mut m: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut d = 2u64;
    while d.saturating_mul(d) <= m {
        if m % d == 0 {
            while m % d == 0 {
                m /= d;
            }
            out.push(d);
        }
        d += if d == 2 { 1 } else { 2 };
    }
    if m > 1 {
        out.push(m);
    }
    out
}

fn order_via_lambda(a: u64, n: u64, lambda: u64, primes: &[u64]) -> Option<u64> {
    if quantsim::padic::gcd(a % n, n) != 1 {
        return None;
    }
    let mut r = lambda;
    for &p in primes {
        while r % p == 0 && shor::pow_mod(a, r / p, n) == 1 {
            r /= p;
        }
    }
    Some(r)
}

fn main() -> Result<()> {
    let sim: Simulator = Simulator::new();

    println!("== 1. r for the bases that actually split N ==");
    println!(
        "  {:>3} {:>12} {:>16} {:>10} {:>12}",
        "w", "N", "median r | split", "r/N", "log2(N/r)"
    );
    for w in [12usize, 14, 16, 18, 20, 22, 24] {
        let Some((n, p, q)) = semiprime(w) else {
            continue;
        };
        let g = quantsim::padic::gcd(p - 1, q - 1);
        let lambda = (p - 1) / g * (q - 1);
        let mut primes = factor_small(p - 1);
        primes.extend(factor_small(q - 1));
        primes.sort_unstable();
        primes.dedup();
        let mut rs: Vec<u64> = Vec::new();
        let step = (n / 20000).max(1);
        let mut a = 2u64;
        while a < n - 1 {
            if let Some(r) = order_via_lambda(a, n, lambda, &primes) {
                if shor::split_from_order(n, a, r).is_some() {
                    rs.push(r);
                }
            }
            a += step;
        }
        if rs.is_empty() {
            continue;
        }
        rs.sort_unstable();
        let med = rs[rs.len() / 2];
        println!(
            "  {w:>3} {n:>12} {med:>16} {:>10.4} {:>12.2}",
            med as f64 / n as f64,
            (n as f64 / med as f64).log2()
        );
    }

    println!();
    println!("== 2. climbing to the ceiling, largest-order base at each width ==");
    quantsim::guard::set_memory_limit(Some(12_000_000_000));
    let budget = std::time::Duration::from_secs(25);
    println!("   budget {budget:?}/run; the committed run used 240s and reached w = 28");
    println!(
        "  {:>3} {:>14} {:>12} {:>9} {:>12} {:>13}",
        "w", "N", "r", "time", "bytes", "ns/entry-rd"
    );
    let mut last: Option<(usize, u64, u64, f64, usize, f64)> = None;
    for w in [20usize, 22, 24, 26, 28, 30] {
        let Some((n, p, q)) = semiprime(w) else {
            continue;
        };
        let g = quantsim::padic::gcd(p - 1, q - 1);
        let lambda = (p - 1) / g * (q - 1);
        let mut primes = factor_small(p - 1);
        primes.extend(factor_small(q - 1));
        primes.sort_unstable();
        primes.dedup();
        let mut best = (0u64, 0u64);
        let mut a = 2u64;
        while a < n.min(4000) {
            if let Some(r) = order_via_lambda(a, n, lambda, &primes) {
                if r > best.0 {
                    best = (r, a);
                }
            }
            a += 1;
        }
        let (r, a) = best;
        if r == 0 {
            continue;
        }
        let f = OrderFinder::new(n, a)?;
        let t = std::time::Instant::now();
        match quantsim::guard::with_time_budget(budget, || {
            f.estimate(&sim, PhaseForm::Semiclassical, &mut Prng::new(7))
        }) {
            Ok(e) => {
                let secs = t.elapsed().as_secs_f64();
                let per =
                    t.elapsed().as_nanos() as f64 / (e.peak_support as f64 * f.phase_bits as f64);
                let bpe = e.peak_bytes as f64 / e.peak_support as f64;
                println!(
                    "  {w:>3} {n:>14} {r:>12} {secs:>8.1}s {:>12} {per:>13.1}",
                    e.peak_bytes
                );
                last = Some((w, n, r, secs, e.peak_bytes, bpe));
            }
            Err(err) => {
                println!("  {w:>3} {n:>14} {r:>12}   {err}");
                break;
            }
        }
    }

    if let Some((w, n, r, secs, b, bytes_per_entry)) = last {
        println!();
        println!("== 3. what a memory budget reaches, from those constants ==");
        println!(
            "   reached here: N = {n} ({w} bits), r = {r}, {secs:.1}s, {} MB, {bytes_per_entry:.0} B/entry",
            b / 1_000_000
        );
        println!("   n_max = log2(M / bytes_per_entry) + 2, the +2 from r ~ N/4 above");
        println!("  {:<26} {:>12} {:>10}", "memory budget", "max r", "max n");
        for (label, m) in [
            ("this machine (16GB)", 1.6e10f64),
            ("one rack (40 x 1TB)", 4.0e13),
            ("datacenter (10PB)", 1.0e16),
            ("1000 datacenters (10EB)", 1.0e19),
        ] {
            let max_r = m / bytes_per_entry;
            println!(
                "  {label:<26} {:>12.3e} {:>10.0}",
                max_r,
                max_r.log2() + 2.0
            );
        }
        println!("   Every constant above is measured on this machine; the only step to");
        println!("   a larger budget is that memory is linear in r, which is also measured.");
    }

    println!();
    println!("== 4. the loop run sharded, verified against a reference every round ==");
    println!("   N = 126727 (w = 17), a = 2, r = 15752");
    println!(
        "  {:>7} {:>10} {:>14} {:>14} {:>12} {:>10}",
        "shards", "peak supp", "routed", "routed/round", "routed bytes", "imbalance"
    );
    for shards in [1usize, 2, 16, 256, 1024] {
        let tr = run_sharded(&sim, 126_727, 2, shards, 7)?;
        println!(
            "  {shards:>7} {:>10} {:>14} {:>14.0} {:>12} {:>10.2}",
            tr.support,
            tr.permute_entries,
            tr.permute_entries as f64 / tr.rounds as f64,
            tr.permute_entries * ENTRY_BYTES,
            tr.imbalance
        );
        assert_eq!(tr.gate_entries, 0, "a one-qubit gate needed routing");
    }
    let tr = run_sharded(&sim, 126_727, 2, 256, 7)?;
    println!(
        "  {:>6} {:>12} {:>10} {:>10}",
        "round", "support in", "routed", "routed/in"
    );
    for (k, (&routed, &supp)) in tr
        .per_round
        .iter()
        .zip(tr.support_per_round.iter())
        .enumerate()
        .filter(|(k, _)| *k + 4 >= tr.per_round.len())
    {
        println!(
            "  {k:>6} {supp:>12} {routed:>10} {:>10.4}",
            routed as f64 / supp.max(1) as f64
        );
    }
    let tail: usize = tr.per_round.iter().rev().take(4).sum();
    println!(
        "  gates and collapse route 0; last 4 of {} rounds carry {:.1}% of the traffic",
        tr.rounds,
        100.0 * tail as f64 / tr.permute_entries as f64
    );
    Ok(())
}
