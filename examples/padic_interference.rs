//! Interference as modular arithmetic, measured live: the radix a
//! geometry supplies, the CRT factorization of the phase field, the
//! interference character read from trailing digits, and a journalled
//! sweep whose cost depends on the order the phase plane is walked in.
//!
//! Run with `cargo run --release --example padic_interference`.

use quantsim::padic::{
    character, compare_diagonals, crt_phase_factors, factored_field, fringe_period, geometry_radix,
    predicted_writes_per_point, sweep, CrtDiagonal, Order, Radix, Sweep, Wave, WaveSystem,
};
use quantsim::prelude::*;

fn main() -> Result<()> {
    println!("== the radix a geometry supplies ==\n");

    // Two slits at commensurate spacing: path differences are exact
    // rationals, so the radix is finite and small.
    let commensurate = geometry_radix(&[0.5, 0.25, 0.125, 1.0 / 3.0], 4096)?;
    println!(
        "commensurate slits  M = {:6}  radix {:?}  exact = {}  phase error {:.2e}",
        commensurate.radix.modulus(),
        commensurate.radix.components(),
        commensurate.exact,
        commensurate.max_phase_error
    );

    // Incommensurate spacing: no finite radix exists. The convergent's
    // denominator is the price, and the golden ratio — the worst
    // approximable number — pays the most at every budget.
    println!("\nincommensurate spacings — no finite radix exists, so the");
    println!("convergent's denominator is the price of the irrationality:\n");
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    println!("  budget        √2                π                 φ");
    println!("            M    err·M²      M    err·M²      M    err·M²");
    for &budget in &[8u64, 32, 128, 512, 2048] {
        let mut cells = String::new();
        for x in [2f64.sqrt().fract(), std::f64::consts::PI.fract(), phi.fract()] {
            let g = geometry_radix(&[x], budget)?;
            let m = g.radix.modulus() as f64;
            cells += &format!(" {:6} {:9.4}", g.radix.modulus(), g.max_path_error * m * m);
        }
        println!("  {budget:6}  {cells}");
    }
    println!("\n(err·M² is the Hurwitz quantity — it is bounded below by 1/√5 ≈ 0.447");
    println!(" only for φ, which is why φ's denominators must grow fastest; π is");
    println!(" unusually well approximated at 355/113 and stalls there.)");

    println!("\n== the CRT factorization of the phase field ==\n");

    // M = 2^4 · 3^3 · 5^2 · 7 = 75600
    let radix = Radix::of(16 * 27 * 25 * 7)?;
    println!(
        "M = {}  components {:?}  depth {} digits",
        radix.modulus(),
        radix.components(),
        radix.depth()
    );
    println!(
        "dense field entries {}   CRT-factored entries {}   compression {:.1}×",
        radix.dense_entries(),
        radix.factored_entries(),
        radix.dense_entries() as f64 / radix.factored_entries() as f64
    );
    let a = 31_337u64;
    println!(
        "phase e^(2πi·{a}·x/M) factors into per-component coefficients {:?}",
        crt_phase_factors(a, &radix)
    );

    // A four-wave system over a 1-D phase argument: each wave's whole
    // field over ℤ/M is one product state across the CRT components.
    let field_system = WaveSystem::new(
        radix.clone(),
        1,
        vec![
            Wave::new(0, vec![1]),
            Wave::new(1234, vec![37]),
            Wave::new(7, vec![2100]),
            Wave::new(999, vec![-11]),
        ],
    )?;
    let probes: Vec<u64> = (0..24).map(|k| (k * 3121 + 7) % radix.modulus()).collect();
    let field = factored_field(&field_system, &probes)?;
    println!(
        "\nrank {} field on a compound register of dims {:?}",
        field.rank, field.dims
    );
    println!(
        "  dense {} amplitudes  vs  factored {} ({:.1}× smaller)",
        field.dense_entries,
        field.factored_entries,
        field.compression()
    );
    println!(
        "  register volumes after writing the whole field: {:?} — stayed a product: {}",
        field.volumes,
        field.stayed_product()
    );
    println!(
        "  max deviation vs the direct interference sum over {} probes: {:.2e}",
        field.probes, field.max_deviation
    );

    println!("\n== the interference character lives in the trailing digits ==\n");
    println!("  Δa      order  coherence  valuations   digit reads");
    for &delta in &[0u64, 37_800, 25_200, 15_120, 2, 1] {
        let c = character(delta, &radix);
        println!(
            "  {delta:6}  {:6}  {:9}  {:?}  {:>6}{}",
            c.order,
            c.coherence,
            c.valuations,
            c.digit_reads,
            if c.constructive() {
                "   ← exactly in phase"
            } else if c.antiphase() {
                "   ← exact antiphase (total destruction)"
            } else {
                ""
            }
        );
    }
    println!(
        "\n(full depth is {} digits; the character never needed more than a handful)",
        radix.depth()
    );

    println!("\n== which CRT diagonal decides fastest ==\n");
    let cmp = compare_diagonals(&radix, 20_000, 0xC0FFEE);
    println!("  ordering          mean steps   worst   (full depth {})", cmp.full_depth);
    for (label, mean, worst) in &cmp.orderings {
        println!("  {label:<16}  {mean:10.3}  {worst:6}");
    }

    println!("\n== fringe spacing, from digits alone ==\n");
    println!("  the pair's fringe period is M/gcd(Δs, M) — a valuation, read off");
    println!("  the trailing digits, with the field never evaluated anywhere:\n");
    for (a, b) in [(0usize, 1usize), (0, 2), (0, 3), (1, 3)] {
        let p = fringe_period(&field_system, (a, b), 0);
        // brute-force confirmation: the first repeat of the character
        let mut scanned = None;
        let c0 = field_system.character_at(a, b, &[0]);
        for t in 1..=radix.modulus() {
            if field_system.character_at(a, b, &[t as i64]) == c0
                && field_system.delta(a, b, &[t as i64]) == field_system.delta(a, b, &[0])
            {
                scanned = Some(t);
                break;
            }
        }
        println!(
            "  waves ({a},{b})  period {:?}   scanned {:?}   agree: {}",
            p,
            scanned,
            p == scanned
        );
    }

    println!("\n== the sweep: how the phase plane is walked ==\n");
    println!("  Each grid below covers ℤ/M *exactly once*, so the measured writes");
    println!("  per point are directly comparable with the closed form.\n");

    for (p, n, side) in [(2u64, 12u32, 64i64), (3, 8, 81)] {
        let r = Radix::of(p.pow(n))?;
        let sys = WaveSystem::new(
            r.clone(),
            2,
            vec![Wave::new(0, vec![1, side]), Wave::new(0, vec![0, 0])],
        )?;
        println!(
            "  M = {}^{} = {}, depth {} digits, grid {side}×{side}",
            p,
            n,
            r.modulus(),
            r.depth()
        );
        println!(
            "    predicted: lattice {:.4}   scrambled {:.4}",
            predicted_writes_per_point(&r, true),
            predicted_writes_per_point(&r, false)
        );
        println!("    order              writes/pt  frac of depth  steps/pt");
        for (label, order) in [
            ("axis-0 lattice", Order::Axis(0)),
            ("axis-1 lattice", Order::Axis(1)),
            ("Morton (Z-order)", Order::Morton),
            ("shuffled", Order::Shuffled(7)),
        ] {
            if matches!(order, Order::Morton) && (side & (side - 1)) != 0 {
                continue; // Morton needs power-of-two extents
            }
            let rep = sweep(
                &sys,
                (0, 1),
                &[side, side],
                order,
                CrtDiagonal::Interleaved,
                false,
            )?;
            println!(
                "    {label:<17}  {:9.4}  {:13.4}  {:8.4}",
                rep.writes_per_point,
                rep.write_fraction(),
                rep.steps_per_point
            );
        }
        // the verdict cache, priced next to its hit count
        let cached = sweep(
            &sys,
            (0, 1),
            &[side, side],
            Order::Axis(0),
            CrtDiagonal::Interleaved,
            true,
        )?;
        println!(
            "    verdict cache: {} hits bought with {} probes — it does not pay,",
            cached.reused, cached.cache_probes
        );
        println!("    because the resolution it replaces is already O(#components).\n");
    }
    println!("  Every lattice order lands on the same closed form, and the direction");
    println!("  does not matter: adding *any* fixed stride to a residue moves only as");
    println!("  many digits as the carry reaches. What costs Θ(depth) is not a badly");
    println!("  chosen axis — it is having no fixed stride at all. \"Propagate linearly");
    println!("  across the surface, not chaotically\" is measured here as 1.50 vs 5.31");
    println!("  digit writes per point, and the gap widens with the depth.");

    println!("\n== the journal is resumable and rewindable ==\n");
    let deep = Radix::of(2u64.pow(12))?;
    let deep_system = WaveSystem::new(
        deep.clone(),
        2,
        vec![Wave::new(0, vec![1, 64]), Wave::new(0, vec![0, 0])],
    )?;
    let mut s = Sweep::new(
        &deep_system,
        (0, 1),
        &[32, 32],
        Order::Axis(0),
        CrtDiagonal::Interleaved,
        false,
    )?;
    let first = s.resume(&deep_system, 300);
    let mid = s.report(&deep);
    let second = s.resume(&deep_system, 10_000);
    let full = s.report(&deep);
    println!(
        "  {first} points, then {second} more: {} total, {} digit writes",
        full.points, full.digit_writes
    );
    println!(
        "  the interrupted checkpoint held {} points at {:.3} writes/pt (final {:.3})",
        mid.points, mid.writes_per_point, full.writes_per_point
    );
    s.rewind_to(100);
    println!(
        "  rewound to point 100: {} entries, {} remaining",
        s.entries().len(),
        s.remaining()
    );

    Ok(())
}
