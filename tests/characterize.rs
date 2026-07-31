//! The characterization harness measures what it claims: ceilings are
//! the library's own refusals, fidelity counts cover the whole registry,
//! and a walled width contributes no fabricated cost.

use quantsim::characterize::{
    characterize, construction_ceiling, family_laws, fidelity_census, perf_envelope, width_ceiling,
    CharacterizeConfig, Wall,
};
use quantsim::prelude::*;

fn h_layer(n: usize) -> Circuit {
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    c
}

fn sim() -> Simulator<C64> {
    Simulator::new()
}

fn e8_sim() -> Simulator<C64> {
    let mut backends = BackendRegistry::<C64>::standard();
    backends.register_e8().expect("fresh registry");
    Simulator::with_registries(GateRegistry::standard(), backends)
}

#[test]
fn the_census_counts_every_registry_name_and_dense_is_bit_exact_against_itself() {
    let sim = sim();
    let cfg = quantsim::conformance::ConformanceConfig {
        random_circuits: 2,
        ..Default::default()
    };
    let report = quantsim::conformance::verify_backend(&sim, "dense", &cfg).expect("dense");
    let census = fidelity_census(&report, cfg.tolerance);
    assert_eq!(
        census.gates_total,
        sim.registry().names().len(),
        "the census must cover every registry name, aliases included"
    );
    assert_eq!(
        census.bit_exact, census.gates_total,
        "dense against itself is bit-exact on every gate, not merely within tolerance"
    );
    assert_eq!(census.matching(), census.gates_total);
    assert_eq!(census.worst.1, 0.0);
    assert!(census.complete());
    assert_eq!(census.outside_tolerance, 0);
    assert_eq!(census.collapse_violations, 0);
}

#[test]
fn sparse_reproduces_the_whole_registry_bit_for_bit() {
    let sim = sim();
    let cfg = quantsim::conformance::ConformanceConfig {
        random_circuits: 4,
        ..Default::default()
    };
    let report = quantsim::conformance::verify_backend(&sim, "sparse", &cfg).expect("sparse");
    let census = fidelity_census(&report, cfg.tolerance);
    assert!(
        census.complete(),
        "sparse left gates unmatched: {} outside, {} not swept",
        census.outside_tolerance,
        census.not_swept
    );
    assert_eq!(
        census.bit_exact, census.gates_total,
        "same accumulation order as dense, so every gate should agree exactly"
    );
    assert!(census.cases >= census.gates_total);
}

#[test]
fn the_construction_ceiling_is_the_backends_own_declared_refusal() {
    let sim = e8_sim();
    let rep = construction_ceiling(&sim, "e8-rep", 24);
    assert_eq!(
        rep.widest, 8,
        "the E8 spinor representation is eight qubits wide by construction"
    );
    match &rep.wall {
        Wall::Construction(message) => assert!(
            message.contains('8'),
            "the wall should carry the backend's own message, got {message:?}"
        ),
        other => panic!("expected a construction refusal, got {other}"),
    }

    // The constellation's ceiling is the trait's u64 index wall, not a
    // property of the geometry — so a sweep that stops short of it
    // reports no refusal at all.
    let tower = construction_ceiling(&sim, "e8-constellation", 24);
    assert_eq!(tower.widest, 24);
    assert_eq!(tower.wall, Wall::SweepLimit);
}

#[test]
fn a_family_ceiling_can_be_far_narrower_than_the_construction_ceiling() {
    let sim = e8_sim();
    let widths: Vec<usize> = (1..=6).map(|k| 4 * k).collect();
    let ghz = width_ceiling(&sim, "e8-constellation", "ghz", library::ghz, &widths);
    // A thread-local budget so the H-layer's wall arrives as a measured
    // refusal in seconds rather than after this machine has tried to
    // store 2^24 lattice points. Nothing global is touched, so the rest
    // of the suite runs unaffected.
    let layer = guard::with_time_budget(std::time::Duration::from_secs(3), || {
        width_ceiling(&sim, "e8-constellation", "h-layer", h_layer, &widths)
    });
    assert!(
        ghz.reached > layer.reached,
        "GHZ keeps two lattice points at every width while an H-layer fills the group; \
         got ghz {} vs h-layer {}",
        ghz.reached,
        layer.reached
    );
    assert_eq!(ghz.support, 2, "GHZ is two stored points however wide");
    assert!(
        matches!(layer.wall, Wall::Operation(_)),
        "the H-layer must stop on a real operation refusal, got {}",
        layer.wall
    );
    // And the construction ceiling is wider than either: the refusal is
    // the circuit's, not the constructor's.
    assert!(construction_ceiling(&sim, "e8-constellation", 32).widest > layer.reached);
}

#[test]
fn a_walled_width_contributes_no_fabricated_cost_to_the_laws() {
    // The E8 spinor representation declares an eight-qubit ceiling, so
    // the wall is instant and exact — no need to push a dense state into
    // this machine's memory to find one.
    let sim = e8_sim();
    let widths = vec![2, 4, 6, 8, 10, 12];
    let laws = family_laws(&sim, "e8-rep", "ghz", library::ghz, &widths);
    assert_eq!(
        laws.widths,
        vec![2, 4, 6, 8],
        "the sweep must stop at the declared ceiling and record nothing past it"
    );
    assert_eq!(laws.widths.len(), laws.bytes.len());
    assert_eq!(laws.widths.len(), laws.nanos.len());
    assert!(
        laws.bytes.iter().all(|&b| b > 0) && laws.nanos.iter().all(|&t| t > 0),
        "every recorded cost came from a run that finished"
    );
    assert!(
        laws.memory_law.is_some(),
        "four surviving widths are enough to fit a law"
    );

    // And the law that does get fitted is the real one: dense doubles per
    // qubit, over widths that all fit comfortably.
    let plain = family_laws(&sim, "dense", "h-layer", h_layer, &[8, 10, 12, 14, 16]);
    assert_eq!(plain.widths.len(), 5, "all five widths should finish");
    match plain.memory_law {
        Some(Law::Exponential { base }) => assert!(
            (base - 2.0).abs() < 0.2,
            "dense doubles per qubit; fitted base {base}"
        ),
        other => panic!("expected an exponential memory law for dense, got {other:?}"),
    }
}

#[test]
fn ghz_is_constant_for_sparse_and_exponential_for_dense() {
    let sim = sim();
    let widths = vec![4, 8, 12, 16, 20];
    let sparse = family_laws(&sim, "sparse", "ghz", library::ghz, &widths);
    let dense = family_laws(&sim, "dense", "ghz", library::ghz, &widths);
    assert_eq!(sparse.memory_law, Some(Law::Constant));
    assert!(
        dense
            .memory_law
            .as_ref()
            .is_some_and(|l| !l.is_subexponential()),
        "dense pays 2^n on GHZ too: {:?}",
        dense.memory_law
    );
}

#[test]
fn the_envelope_separates_the_representations_by_bytes_per_amplitude() {
    let sim = e8_sim();
    let dense = perf_envelope(&sim, "dense", 8, 3).expect("dense envelope");
    let sparse = perf_envelope(&sim, "sparse", 8, 3).expect("sparse envelope");
    let tower = perf_envelope(&sim, "e8-constellation", 8, 3).expect("tower envelope");

    for env in [&dense, &sparse, &tower] {
        assert_eq!(
            env.timed,
            sim.registry().names().len(),
            "{} left gates untimed at width 8",
            env.backend
        );
        assert_eq!(env.refused, 0);
        assert!(env.fastest.is_some() && env.slowest.is_some());
        assert!(env.slowest.as_ref().unwrap().nanos >= env.fastest.as_ref().unwrap().nanos);
    }

    // Dense stores 2^n slots regardless of support, so its cost per
    // stored amplitude at width 8 dwarfs both sparse maps.
    assert!(
        dense.bytes_per_amplitude.0 > 10.0 * sparse.bytes_per_amplitude.1,
        "dense {:?} vs sparse {:?}",
        dense.bytes_per_amplitude,
        sparse.bytes_per_amplitude
    );
    // A lattice point key is bigger than a u64 key, and it is measured
    // rather than assumed.
    assert!(
        tower.bytes_per_amplitude.0 > sparse.bytes_per_amplitude.0,
        "an eight-coordinate lattice key costs more per amplitude than a u64: \
         tower {:?} vs sparse {:?}",
        tower.bytes_per_amplitude,
        sparse.bytes_per_amplitude
    );
    assert!(tower.bytes_per_amplitude.1 < dense.bytes_per_amplitude.0);
}

#[test]
fn the_full_characterization_of_the_constellation_holds_together() {
    let sim = e8_sim();
    let families: [(&str, &dyn Fn(usize) -> Circuit); 2] =
        [("ghz", &library::ghz), ("h-layer", &h_layer)];
    let cfg = CharacterizeConfig {
        widths: vec![4, 8, 12, 16],
        sweep_max: 20,
        envelope_width: 8,
        envelope_reps: 3,
        ..Default::default()
    };
    let report = guard::with_time_budget(std::time::Duration::from_secs(20), || {
        characterize(&sim, "e8-constellation", &families, &cfg)
    })
    .expect("characterize");

    let census = report.fidelity.as_ref().expect("fidelity was requested");
    assert!(
        census.complete(),
        "the constellation must reproduce the whole registry: {} outside tolerance",
        census.outside_tolerance
    );
    assert_eq!(census.gates_total, sim.registry().names().len());

    assert_eq!(report.construction.widest, 20);
    let best = report.best_reach().expect("families were swept");
    let worst = report.worst_reach().expect("families were swept");
    assert!(best.reached >= worst.reached);
    // Both families clear 16 qubits here — the constellation's *width*
    // ceiling is not what separates them at this scale, its support is.
    for ceiling in &report.ceilings {
        assert_eq!(ceiling.reached, 16, "{} stopped early", ceiling.family);
    }
    let ghz = report
        .ceilings
        .iter()
        .find(|c| c.family == "ghz")
        .expect("ghz swept");
    let layer = report
        .ceilings
        .iter()
        .find(|c| c.family == "h-layer")
        .expect("h-layer swept");
    assert_eq!(ghz.support, 2, "GHZ is two lattice points at any width");
    assert_eq!(
        layer.support,
        1 << 16,
        "an H-layer fills the residue group, and the report says so in support \
         rather than hiding it in a ceiling"
    );
    assert!(layer.bytes > 100 * ghz.bytes);
    assert_eq!(report.laws.len(), 2);
    assert!(report.envelope.is_some());
    // Display must not panic and must mention the counts.
    let text = report.to_string();
    assert!(text.contains("gate fidelity"));
    assert!(text.contains("bytes/amplitude"));
}

#[test]
fn characterizing_an_unknown_backend_refuses_rather_than_reporting_zero() {
    let sim = sim();
    let families: [(&str, &dyn Fn(usize) -> Circuit); 1] = [("ghz", &library::ghz)];
    let err = characterize(
        &sim,
        "no-such-backend",
        &families,
        &CharacterizeConfig::default(),
    );
    assert!(
        err.is_err(),
        "an unknown name must be an error, not a ceiling of zero"
    );
}
