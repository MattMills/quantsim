//! Physically reproduce existing qubit register geometries and their
//! operation-latency maps, and measure how the geometry — not the
//! physics — moves the stats.
//!
//! Four measurements, all printed from real runs:
//!
//! 1. the wiring of the reproduced machines (sites, couplers, degree);
//! 2. one chip-scale circuit over three geometries: swap count, elapsed,
//!    serial time and the parallelism ratio change with the coupling
//!    map, while the amplitudes stay bit-for-bit on the reference;
//! 3. one schedule under three era clocks: the physical op sequence is
//!    identical, only the wall clock rescales;
//! 4. a heterogeneous latency map (one slow coupler, as in real
//!    calibration data): the clock moves by exactly the override, and —
//!    because routing is latency-blind BFS — the router walks straight
//!    across the slow edge even when an equal-hop detour is free. That
//!    measured gap is the roadmap case for latency-aware routing.
//!
//! Run with `cargo run --release --example device_reproduction`.

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

fn device(topology: Topology, latency: LatencyMap, sparse: bool) -> Result<DeviceState<C64>> {
    let n = topology.num_sites();
    let inner: Box<dyn Backend<C64>> = if sparse {
        Box::new(SparseState::new(n)?)
    } else {
        Box::new(DenseState::new(n)?)
    };
    DeviceState::with_latency(topology, latency, ArityPolicy::default(), inner)
}

fn main() -> Result<()> {
    let reg = GateRegistry::<C64>::standard();

    // ── 1. The reproduced machines, structurally ─────────────────────
    println!("── reproduced register geometries");
    let machines: [(&str, Topology); 4] = [
        ("linear-27 (idealized chain)", Topology::linear(27)),
        ("ibm falcon-27 heavy-hex", Topology::heavy_hex_falcon27()),
        ("sycamore-like 6×9 diagonal", Topology::sycamore_like(6, 9)),
        ("trapped-ion-27 all-to-all", Topology::complete(27)),
    ];
    for (name, t) in &machines {
        println!(
            "  {name:<28} {:>3} sites  {:>3} couplers  max degree {:>2}  connected {}",
            t.num_sites(),
            t.num_edges(),
            t.max_degree(),
            t.is_connected()
        );
    }

    // ── 2. Same circuit, different geometry ──────────────────────────
    // GHZ across 27 qubits, sparse inner (2 nonzeros — geometry at chip
    // scale without a 2^27 vector). The physics must not move; every
    // schedule stat may.
    println!("\n── ghz-27, one circuit over three geometries (sparse inner)");
    let n = 27;
    let ghz = library::ghz(n).bind(&reg)?;
    let reference = Simulator::<C64>::new().run_on("sparse", &library::ghz(n))?;
    println!(
        "  {:<28} {:>5}  {:>12}  {:>12}  {:>11}  {:>9}",
        "geometry", "swaps", "elapsed", "serial", "parallelism", "max |Δamp|"
    );
    for (name, topology) in [
        ("linear-27", Topology::linear(n)),
        ("falcon-27 heavy-hex", Topology::heavy_hex_falcon27()),
        ("ion-27 all-to-all", Topology::complete(n)),
    ] {
        let mut dev = device(
            topology,
            LatencyMap::uniform(DurationModel::ibm_falcon_like()),
            true,
        )?;
        ghz.run(&mut dev)?;
        let deviation = max_amplitude_deviation(reference.as_ref(), &dev);
        println!(
            "  {name:<28} {:>5}  {:>12}  {:>12}  {:>10.2}×  {deviation:>9.1e}",
            dev.swap_count(),
            fmt_ns(dev.elapsed()),
            fmt_ns(dev.serial_time()),
            dev.serial_time() as f64 / dev.elapsed() as f64,
        );
    }
    println!("  the coupling map moves routing and the clock; never the state.");
    println!("  (a ghz chain is serial by construction: parallelism 1.00× everywhere.)");

    // Chip scale: all 54 Sycamore-class sites over the real diagonal
    // lattice — impossible dense (2^54 amplitudes), trivial sparse.
    let mut chip = device(
        Topology::sycamore_like(6, 9),
        LatencyMap::uniform(DurationModel::sycamore_like()),
        true,
    )?;
    library::ghz(54).bind(&reg)?.run(&mut chip)?;
    println!(
        "  ghz-54 on the full sycamore lattice: {} swaps, elapsed {}, inner {} B, {} nonzeros",
        chip.swap_count(),
        fmt_ns(chip.elapsed()),
        chip.memory_bytes(),
        chip.nonzero_count(),
    );

    // A transversal layer — 13 disjoint cx pairs — is where geometry
    // decides how much parallelism *survives*: pairs native to the
    // coupling map all fire at once; pairs that must route serialize on
    // the swap chains.
    println!("\n── transversal layer: 13 disjoint cx pairs, same three geometries");
    let mut layer: Circuit = Circuit::new(n);
    for k in 0..n / 2 {
        layer.cx(2 * k, 2 * k + 1);
    }
    let bound_layer = layer.bind(&reg)?;
    println!(
        "  {:<28} {:>5}  {:>12}  {:>12}  {:>11}",
        "geometry", "swaps", "elapsed", "serial", "parallelism"
    );
    for (name, topology) in [
        ("linear-27", Topology::linear(n)),
        ("falcon-27 heavy-hex", Topology::heavy_hex_falcon27()),
        ("ion-27 all-to-all", Topology::complete(n)),
    ] {
        let mut dev = device(
            topology,
            LatencyMap::uniform(DurationModel::ibm_falcon_like()),
            true,
        )?;
        bound_layer.run(&mut dev)?;
        println!(
            "  {name:<28} {:>5}  {:>12}  {:>12}  {:>10.2}×",
            dev.swap_count(),
            fmt_ns(dev.elapsed()),
            fmt_ns(dev.serial_time()),
            dev.serial_time() as f64 / dev.elapsed() as f64,
        );
    }
    println!("  geometry decides how much of the circuit's parallelism survives routing.");

    // ── 3. Same schedule, three era clocks ───────────────────────────
    println!("\n── qft-8 on the falcon corner, three era latency maps");
    let corner = Topology::heavy_hex_falcon_corner(8)?;
    let qft = library::qft(8).bind(&reg)?;
    let mut op_sequences = Vec::new();
    for (name, model) in [
        ("ibm-falcon-like", DurationModel::ibm_falcon_like()),
        ("sycamore-like", DurationModel::sycamore_like()),
        ("ion-trap-like", DurationModel::ion_trap_like()),
    ] {
        let mut dev = device(corner.clone(), LatencyMap::uniform(model), false)?;
        qft.run(&mut dev)?;
        op_sequences.push(
            dev.physical_log()
                .iter()
                .map(|op| (op.label.clone(), op.sites.clone()))
                .collect::<Vec<_>>(),
        );
        println!(
            "  {name:<18} 1q {:>9}  2q {:>9}  {:>4} physical ops  {:>3} swaps  elapsed {:>10}",
            fmt_ns(model.one_q),
            fmt_ns(model.two_q),
            dev.physical_log().len(),
            dev.swap_count(),
            fmt_ns(dev.elapsed()),
        );
    }
    let identical = op_sequences.windows(2).all(|w| w[0] == w[1]);
    println!("  identical physical op sequence across all three clocks: {identical}");

    // ── 4. Heterogeneous calibration and latency-blind routing ───────
    // Real calibration data is per-coupler. Slow exactly one and watch
    // the clock move by exactly that much when it sits on the critical
    // path — then catch the router walking across it anyway.
    println!("\n── per-coupler overrides (heterogeneous calibration)");
    let base = DurationModel::ibm_falcon_like();
    let mut c: Circuit = Circuit::new(4);
    c.h(0).cx(0, 1).cx(1, 2).cx(2, 3);
    let chain = c.bind(&reg)?;
    let mut uniform_dev = device(Topology::linear(4), LatencyMap::uniform(base), false)?;
    chain.run(&mut uniform_dev)?;
    let mut slowed = LatencyMap::uniform(base);
    slowed.set_two_q(1, 2, base.two_q + 1000);
    let mut slowed_dev = device(Topology::linear(4), slowed, false)?;
    chain.run(&mut slowed_dev)?;
    println!(
        "  chain cx cascade, uniform {} → coupler (1,2) +1000 ns → {} (Δ = {} ns, exact)",
        fmt_ns(uniform_dev.elapsed()),
        fmt_ns(slowed_dev.elapsed()),
        slowed_dev.elapsed() - uniform_dev.elapsed(),
    );

    // Latency-blind routing, measured. On a 6-ring, cx(0,3) has two
    // equal-hop routes: 0-1-2-3 and 0-5-4-3. BFS deterministically
    // takes 0-1-2-3. Slow that path's swap coupler vs the detour's:
    // an aware router would dodge either way and both runs would tie.
    let ring = Topology::ring(6);
    let mut c: Circuit = Circuit::new(6);
    c.h(0).cx(0, 3);
    let far = c.bind(&reg)?;
    let slow = 10 * base.swap;
    let run_ring = |slow_edge: (usize, usize)| -> Result<DeviceState<C64>> {
        let mut latency = LatencyMap::uniform(base);
        latency.set_swap(slow_edge.0, slow_edge.1, slow);
        let mut dev = device(ring.clone(), latency, false)?;
        far.run(&mut dev)?;
        Ok(dev)
    };
    let on_path = run_ring((1, 2))?;
    let off_path = run_ring((4, 5))?;
    println!("\n── latency-blind routing, measured (cx 0→3 on a 6-ring)");
    println!(
        "  slow coupler on the BFS path (1,2):  elapsed {}  route {:?}",
        fmt_ns(on_path.elapsed()),
        on_path
            .physical_log()
            .iter()
            .filter(|op| op.label == "swap")
            .map(|op| (op.sites[0], op.sites[1]))
            .collect::<Vec<_>>(),
    );
    println!(
        "  slow coupler on the detour  (4,5):  elapsed {}  same route, untouched",
        fmt_ns(off_path.elapsed()),
    );
    println!(
        "  measured cost of latency-blindness: {} — an equal-hop detour was free.",
        fmt_ns(on_path.elapsed() - off_path.elapsed()),
    );
    println!("  that gap is the roadmap case for latency-aware routing.");
    Ok(())
}
