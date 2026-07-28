//! Causal geometry of qubit registers, measured live: registers whose
//! coupling fabric is organized by causal scale, the availability and
//! wall-clock they buy, the causal diamond that bounds what must be
//! computed, and dual-time resolution — two opposed evolution
//! directions meeting at a cut.
//!
//! Run with `cargo run --release --example causal_geometry`.

use quantsim::causal::{causal_diamond, dual_time_amplitude};
use quantsim::prelude::*;

fn fmt_ns(ns: u64) -> String {
    if ns >= 1_000_000 {
        format!("{:.2} ms", ns as f64 / 1e6)
    } else if ns >= 1_000 {
        format!("{:.2} µs", ns as f64 / 1e3)
    } else {
        format!("{ns} ns")
    }
}

fn device_on(topology: Topology) -> DeviceState<C64> {
    let n = topology.num_sites();
    DeviceState::with_latency(
        topology,
        LatencyMap::uniform(DurationModel::ibm_falcon_like()),
        ArityPolicy::default(),
        Box::new(SparseState::<C64>::new(n).unwrap()),
    )
    .unwrap()
}

fn staircase(n: usize, steps: usize) -> Circuit {
    let mut c: Circuit = Circuit::new(n);
    for l in 0..steps {
        c.h(l).cx(l, l + 1);
    }
    c
}

fn main() -> Result<()> {
    let reg = GateRegistry::<C64>::standard();

    // ── 1. The causal metric of register fabrics ─────────────────────
    println!("── register fabrics and their causal metrics (32 sites)");
    let styles: Vec<(&str, Topology)> = vec![
        ("linear chain", Topology::linear(32)),
        ("ring", Topology::ring(32)),
        ("grid 4×8", Topology::grid(4, 8)),
        ("heavy-hex corner", Topology::heavy_hex_falcon_corner(27)?),
        ("hierarchical (causal)", Topology::hierarchical(32)),
        ("hypercube-5 (causal)", Topology::hypercube(5)),
        ("all-to-all", Topology::complete(32)),
    ];
    println!(
        "  {:<22} {:>8} {:>10} {:>7}   {:>21}   ball growth from 0",
        "fabric", "horizon", "mean dist", "degree", "avail @ 1/2/3 swaps"
    );
    for (name, t) in &styles {
        let ball = t.ball_sizes(0);
        let growth: Vec<String> = ball.iter().take(5).map(|b| b.to_string()).collect();
        println!(
            "  {:<22} {:>8} {:>10.2} {:>7}   {:>5.2} {:>5.2} {:>5.2}     {}",
            name,
            t.diameter(),
            t.mean_distance(),
            t.max_degree(),
            t.pair_availability(1),
            t.pair_availability(2),
            t.pair_availability(3),
            growth.join(" → "),
        );
    }
    println!("  organizing the fabric by causal scale collapses the horizon from linear");
    println!("  to logarithmic; ball growth is the curvature signature (linear = flat 1-D,");
    println!("  exponential = hyperbolic).");

    // ── 2. The router turns the metric into wall-clock ───────────────
    println!("\n── qft-16: the full schedule under each fabric (measured)");
    let qft = library::qft(16).bind(&reg)?;
    println!(
        "  {:<22} {:>6}  {:>12}  {:>12}  {:>11}",
        "fabric", "swaps", "elapsed", "serial", "parallelism"
    );
    for (name, t) in [
        ("linear chain", Topology::linear(16)),
        ("grid 4×4", Topology::grid(4, 4)),
        ("hierarchical (causal)", Topology::hierarchical(16)),
        ("hypercube-4 (causal)", Topology::hypercube(4)),
        ("all-to-all", Topology::complete(16)),
    ] {
        let mut dev = device_on(t);
        qft.run(&mut dev)?;
        println!(
            "  {:<22} {:>6}  {:>12}  {:>12}  {:>10.2}×",
            name,
            dev.swap_count(),
            fmt_ns(dev.elapsed()),
            fmt_ns(dev.serial_time()),
            dev.serial_time() as f64 / dev.elapsed() as f64,
        );
    }
    println!("  the QFT couples every pair; a causal fabric absorbs most of its routing.");
    println!("  measured honestly: the HOMOGENEOUS causal fabric (hypercube — every site a");
    println!("  hub) wins swaps AND wall-clock; the hub-concentrated hierarchy wins swaps");
    println!("  but serializes through its hubs — availability improved, parallelism paid.");

    // ── 3. The causal diamond bounds what must be computed ───────────
    println!("\n── causal diamonds of brickwork-63, depth 8, seed at site 31");
    let circuit = library::brickwork(63, 8, &[31]);
    let sim: Simulator = Simulator::new();
    let full = sim.run_on("sparse", &circuit)?;
    println!(
        "  {:<26} {:>10} {:>10} {:>12} {:>16}",
        "observation surface", "kept ops", "dropped", "cone width", "P(1) unchanged"
    );
    for (label, observed) in [
        ("edge qubit {0}", vec![0usize]),
        ("seed qubit {31}", vec![31]),
        ("window {29..34}", (29..34).collect::<Vec<_>>()),
        ("everything", (0..63).collect::<Vec<_>>()),
    ] {
        let (pruned, report) = causal_diamond(&circuit, &observed);
        let small = sim.run_on("sparse", &pruned)?;
        let q = observed[0];
        let (pf, pp) = (
            {
                let mut p = 0.0;
                full.for_each_nonzero(&mut |idx, amp| {
                    if (idx >> q) & 1 == 1 {
                        p += amp.norm_sqr();
                    }
                });
                p
            },
            {
                let mut p = 0.0;
                small.for_each_nonzero(&mut |idx, amp| {
                    if (idx >> q) & 1 == 1 {
                        p += amp.norm_sqr();
                    }
                });
                p
            },
        );
        println!(
            "  {:<26} {:>6}/{:<3} {:>10} {:>12} {:>10.6} = {:.6}",
            label,
            report.kept,
            circuit.len(),
            report.dropped,
            report.cone_qubits,
            pf,
            pp,
        );
    }
    println!("  ops outside the backward cone of the observation provably cannot move it —");
    println!("  the diamond, not the register width, is what must be computed.");

    // ── 4. Dual time: two opposed directions resolve at a cut ────────
    let n = 17;
    let steps = 16;
    let stair = staircase(n, steps);
    let dense = sim.run(&stair)?;
    let one_way = sim.run_on("sparse", &stair)?;
    println!("\n── dual-time resolution of ⟨0|U|0⟩ on the staircase (support doubles/step)");
    println!(
        "  one-directional manifold at the observation boundary: {} states",
        one_way.nonzero_count()
    );
    println!(
        "  {:>5}  {:>12} {:>12} {:>11} {:>9}   amplitude",
        "cut", "fwd support", "bwd support", "interface", "peak"
    );
    let direct = dense.amplitude(0);
    for cut in [0, 8, 16, 24, 32] {
        let r = dual_time_amplitude(&reg, &stair, cut, 0)?;
        println!(
            "  {:>5}  {:>12} {:>12} {:>11} {:>9}   {:+.6} ({}exact)",
            cut,
            r.forward_support,
            r.backward_support,
            r.interface_terms,
            r.peak_support(),
            r.amplitude.re,
            if (r.amplitude - direct).norm() < 1e-12 {
                ""
            } else {
                "IN"
            },
        );
    }
    println!("  every cut resolves the same amplitude; the balanced cut pays √manifold");
    println!("  per direction (256² = 65536). The cut chooses where the cost sits,");
    println!("  never what is true.");

    // The same opposed clocks on hardware.
    let base = DurationModel::ibm_falcon_like();
    let (front, back) = stair.split_at(steps);
    let mut full_dev = device_on(Topology::linear(n));
    stair.bind(&reg)?.run(&mut full_dev)?;
    let mut fwd_dev = device_on(Topology::linear(n));
    front.bind(&reg)?.run(&mut fwd_dev)?;
    let mut bwd_dev = device_on(Topology::linear(n));
    back.bind(&reg)?.inverse().run(&mut bwd_dev)?;
    println!(
        "  on hardware clocks ({} per step): one direction {}, opposed directions {} ∥ {}",
        fmt_ns(base.one_q + base.two_q),
        fmt_ns(full_dev.elapsed()),
        fmt_ns(fwd_dev.elapsed()),
        fmt_ns(bwd_dev.elapsed()),
    );
    println!("  — the schedule wall is max of the two opposed clocks: half.");

    // ── 5. Causal range, priced by representation geometry ───────────
    println!("\n── ranged_pairs(12, d): one causal knob, every internal geometry (measured)");
    println!(
        "  {:>3}  {:>13} {:>10} {:>11} {:>13} {:>14}",
        "d", "device swaps", "mps swaps", "mps bond", "mera memory", "factored mem"
    );
    for d in [1, 2, 3, 6] {
        let c = library::ranged_pairs(12, d);
        let bound = c.bind(&reg)?;
        let mut dev = device_on(Topology::linear(12));
        bound.run(&mut dev)?;
        let mut mps = MpsState::<C64>::new(12)?;
        bound.run(&mut mps)?;
        let mut mera = MeraState::<C64>::new(12)?;
        bound.run(&mut mera)?;
        let mut factored = FactoredState::<C64>::new(12)?;
        bound.run(&mut factored)?;
        println!(
            "  {:>3}  {:>13} {:>10} {:>11} {:>11} B {:>12} B",
            d,
            dev.swap_count(),
            mps.routing_swaps(),
            mps.max_bond_dimension(),
            mera.memory_bytes(),
            factored.memory_bytes(),
        );
    }
    println!("  mobile geometries (device, mps) pay causal range in TIME; the fixed tree");
    println!("  (mera) pays it in MEMORY at the crossed cut; clustering (factored) is");
    println!("  range-blind. The register fabric decides which currency you spend.");
    Ok(())
}
