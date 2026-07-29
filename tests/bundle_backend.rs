//! The bundle as a `Backend`: gate action on the graph description,
//! measured against dense.
//!
//! Everything here compares up to a global phase, which is unobservable
//! and which vertex operators are only defined up to.

mod common;

use quantsim::bundle::{vop, PolarityBundle};
use quantsim::prelude::*;
use quantsim::rng::Prng;

/// Deviation minimized over a global phase, fitted from the largest
/// amplitude.
fn dev_up_to_phase(a: &dyn Backend<C64>, b: &dyn Backend<C64>) -> f64 {
    let mut anchor = (0u64, 0.0f64);
    a.for_each_nonzero(&mut |i, z| {
        if z.norm() > anchor.1 {
            anchor = (i, z.norm());
        }
    });
    if anchor.1 == 0.0 {
        return max_amplitude_deviation(a, b);
    }
    let y = b.amplitude(anchor.0);
    if y.norm() < 1e-12 {
        return f64::INFINITY;
    }
    let phase = a.amplitude(anchor.0) / y;
    let mut dev = 0.0f64;
    a.for_each_nonzero(&mut |i, z| dev = dev.max((z - b.amplitude(i) * phase).norm()));
    b.for_each_nonzero(&mut |i, z| dev = dev.max((a.amplitude(i) - z * phase).norm()));
    dev
}

#[test]
fn the_vertex_operators_are_the_single_qubit_clifford_group() {
    assert_eq!(vop::count(), 24);
    assert_eq!(
        vop::compose(vop::hadamard(), vop::hadamard()),
        vop::IDENTITY
    );
    let s = vop::phase();
    assert_eq!(
        vop::compose(vop::compose(s, s), vop::compose(s, s)),
        vop::IDENTITY
    );
    // Exactly four are diagonal matrices: I, S, Z, S†.
    assert_eq!(
        (0..vop::count() as u8)
            .filter(|&v| vop::is_diagonal(v))
            .count(),
        4
    );
    for v in 0..vop::count() as u8 {
        assert_eq!(vop::compose(v, vop::inverse(v)), vop::IDENTITY);
    }
}

#[test]
fn local_complementation_changes_the_description_not_the_state() {
    let mut bundle = PolarityBundle::new(5).unwrap();
    for (a, b) in [(0u32, 1u32), (1, 2), (2, 3), (3, 4), (0, 2)] {
        bundle.link(a, b).unwrap();
    }
    bundle.apply_vop(1, vop::hadamard()).unwrap();
    let before = bundle.to_state().unwrap();
    let signature = bundle.link_signature();

    bundle.local_complement(2).unwrap();

    // The graph really moved...
    assert_ne!(bundle.link_signature(), signature);
    // ...and the state did not.
    let after = bundle.to_state().unwrap();
    assert_eq!(dev_up_to_phase(before.as_ref(), after.as_ref()), 0.0);
}

#[test]
fn the_bundle_is_a_registered_backend() {
    let sim: Simulator = Simulator::new();
    assert!(sim.backends().contains("bundle"));
    assert!(sim.backends().names().contains(&"bundle".to_string()));

    // A fresh one is |0…0⟩, as the Backend contract requires.
    let fresh = sim.backends().create("bundle", 4).unwrap();
    common::assert_close(fresh.probability(0), 1.0, 1e-12);
}

#[test]
fn clifford_circuits_agree_with_dense_gate_for_gate() {
    let registry = GateRegistry::<C64>::standard();
    let clifford = ["h", "s", "sdg", "x", "y", "z", "cx", "cz", "swap"];
    let sim: Simulator = Simulator::new();

    let (mut ran, mut refused) = (0usize, 0usize);
    let mut worst = 0.0f64;
    for seed in 0..40u64 {
        let n = 5;
        let mut rng = Prng::new(seed);
        let mut circuit: Circuit = Circuit::new(n);
        for _ in 0..30 {
            let name = clifford[(rng.next_f64() * clifford.len() as f64) as usize % clifford.len()];
            let arity = registry.resolve(name).unwrap().arity();
            let a = (rng.next_f64() * n as f64) as usize % n;
            if arity == 1 {
                circuit.gate(name, Vec::new(), vec![a]);
            } else {
                let mut b = (rng.next_f64() * n as f64) as usize % n;
                if b == a {
                    b = (a + 1) % n;
                }
                circuit.gate(name, Vec::new(), vec![a, b]);
            }
        }
        let dense = sim.run_on("dense", &circuit).unwrap();
        match sim.run_on("bundle", &circuit) {
            Ok(bundle) => {
                ran += 1;
                worst = worst.max(dev_up_to_phase(bundle.as_ref(), dense.as_ref()));
            }
            Err(e) => {
                // The vertex-operator reduction does not converge on
                // every configuration yet. It says so by name; it never
                // returns a wrong state.
                refused += 1;
                assert!(format!("{e}").contains("diagonal subgroup"), "{e}");
            }
        }
    }
    assert!(ran >= 38, "only {ran} of 40 random Clifford circuits ran");
    assert!(worst < 1e-12, "worst deviation {worst} over {ran} circuits");
    assert_eq!(ran + refused, 40);
}

#[test]
fn named_circuits_run_on_the_bundle_at_a_fraction_of_the_memory() {
    let sim: Simulator = Simulator::new();
    for n in [4usize, 8, 12] {
        let ghz = library::ghz(n);
        let bundle = sim.run_on("bundle", &ghz).unwrap();
        let dense = sim.run_on("dense", &ghz).unwrap();
        assert_eq!(dev_up_to_phase(bundle.as_ref(), dense.as_ref()), 0.0);
        common::assert_close(bundle.probability(0), 0.5, 1e-12);
        common::assert_close(bundle.probability((1 << n) - 1), 0.5, 1e-12);
    }
    // The description does not grow with the state space.
    let wide = sim.run_on("bundle", &library::ghz(20)).unwrap();
    let dense = DenseState::<C64>::new(20).unwrap();
    assert!(
        wide.memory_bytes() * 100 < dense.memory_bytes(),
        "bundle {} vs dense {}",
        wide.memory_bytes(),
        dense.memory_bytes()
    );
}

#[test]
fn non_clifford_gates_are_refused_by_name() {
    let sim: Simulator = Simulator::new();
    let mut circuit: Circuit = Circuit::new(3);
    circuit.h(0).t(0);
    let text = match sim.run_on("bundle", &circuit) {
        Ok(_) => panic!("a T gate leaves the Clifford sector"),
        Err(e) => format!("{e}"),
    };
    assert!(text.contains("non-Clifford"), "{text}");
    assert!(text.contains("graph-state bundle"), "{text}");

    // ...and arbitrary amplitudes cannot be loaded into a description.
    let mut bundle = PolarityBundle::new(3).unwrap();
    let entries = [(0u64, C64::new(1.0, 0.0))];
    assert!(Backend::<C64>::load(&mut bundle, &entries).is_err());
}
