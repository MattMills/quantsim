//! Mixed-radix register spaces, measured: the index algebra of a
//! [`Space`]; the qudit gate set's relations; the mixed-radix dense and
//! sparse backends certified against the independent
//! [`CompoundRegister`] (and against each other, over ℂ and ℍ); qubit
//! circuits placed on the binary sites of a mixed register; site
//! measurement statistics and collapse; and every refusal — a non-binary
//! target for a qubit operation, a backend without mixed-radix storage, a
//! swap across radices — by name.

mod common;

use quantsim::backend::{max_amplitude_deviation, pauli_expectation};
use quantsim::mixed::CompoundRegister;
use quantsim::prelude::*;
use quantsim::space::{clock, cshift, fourier, shift, swap};

const ZIGZAG: [usize; 7] = [2, 3, 4, 5, 4, 3, 2];

fn z(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

/// One random step: a random unitary on one or two random sites.
fn random_step(space: &Space, rng: &mut Prng) -> (Vec<usize>, Vec<C64>) {
    let n = space.sites();
    let a = (rng.next_f64() * n as f64) as usize % n;
    let sites = if rng.next_f64() < 0.5 {
        vec![a]
    } else {
        let mut b = (rng.next_f64() * n as f64) as usize % n;
        if b == a {
            b = (a + 1) % n;
        }
        vec![a, b]
    };
    let d = space.sub_dim(&sites);
    (sites, common::random_unitary(d, rng))
}

// ── the index algebra ────────────────────────────────────────────────

#[test]
fn indices_are_little_endian_mixed_radix() {
    let s = Space::new(&ZIGZAG).unwrap();
    assert_eq!(s.sites(), 7);
    assert_eq!(s.dim(), 2 * 3 * 4 * 5 * 4 * 3 * 2);
    assert_eq!(s.stride(0), 1);
    assert_eq!(s.stride(1), 2);
    assert_eq!(s.stride(3), 24);
    assert!(!s.is_qubits());
    let digits = [1, 2, 3, 4, 3, 2, 1];
    let idx = s.encode(&digits).unwrap();
    assert_eq!(idx, 1 + 2 * 2 + 3 * 6 + 4 * 24 + 3 * 120 + 2 * 480 + 1440);
    assert_eq!(s.decode(idx), digits);
    for (site, &want) in digits.iter().enumerate() {
        assert_eq!(s.digit(idx, site), want);
        let moved = s.with_digit(idx, site, 0);
        assert_eq!(s.digit(moved, site), 0);
    }
    assert_eq!(s.to_string(), "radices [2, 3, 4, 5, 4, 3, 2]");
}

#[test]
fn the_qubit_space_is_the_all_twos_space() {
    let q = Space::qubits(5);
    assert_eq!(q, Space::new(&[2; 5]).unwrap());
    assert!(q.is_qubits());
    assert_eq!(q.dim(), 32);
    assert_eq!(q.encode(&[1, 0, 1, 1, 0]).unwrap(), 0b01101);
    assert_eq!(q.to_string(), "5 qubits");
}

#[test]
fn malformed_spaces_and_digits_are_refused() {
    assert!(Space::new(&[2, 1, 3]).is_err());
    assert!(Space::new(&[1 << 40, 1 << 40]).is_err());
    let s = Space::new(&[3, 5]).unwrap();
    assert!(s.encode(&[3, 0]).is_err());
    assert!(s.encode(&[0]).is_err());
    assert!(s.validate_sites(&[0, 0]).is_err());
    assert!(s.validate_sites(&[2]).is_err());
}

#[test]
fn offsets_and_sub_indices_invert_each_other() {
    let s = Space::new(&ZIGZAG).unwrap();
    let sites = [5, 2, 3];
    let offsets = s.offsets(&sites);
    assert_eq!(offsets.len(), 3 * 4 * 5);
    for (sub, &off) in offsets.iter().enumerate() {
        assert_eq!(s.sub_index(off, &sites), sub);
    }
}

// ── the qudit gate set ───────────────────────────────────────────────

fn power(m: &SiteMatrix<C64>, k: usize) -> SiteMatrix<C64> {
    let mut out = SiteMatrix::identity(m.dim());
    for _ in 0..k {
        out = out.matmul(m).unwrap();
    }
    out
}

fn assert_matrix_close(a: &SiteMatrix<C64>, b: &SiteMatrix<C64>, tol: f64) {
    for (x, y) in a.data().iter().zip(b.data()) {
        assert!((x - y).norm() < tol, "{x} vs {y}");
    }
}

#[test]
fn clock_and_shift_satisfy_the_weyl_relations() {
    for d in 2..=7 {
        let x = shift::<C64>(d);
        let z = clock::<C64>(d).unwrap();
        let id = SiteMatrix::identity(d);
        assert_matrix_close(&power(&x, d), &id, 1e-12);
        assert_matrix_close(&power(&z, d), &id, 1e-12);
        // Z X = ω X Z.
        let omega = quantsim::math::cis(std::f64::consts::TAU / d as f64);
        let zx = z.matmul(&x).unwrap();
        let xz = x.matmul(&z).unwrap();
        for (a, b) in zx.data().iter().zip(xz.data()) {
            assert!((a - omega * b).norm() < 1e-12);
        }
        // The Fourier transform is unitary and carries clock to shift:
        // F Z F† = X⁻¹ in this sign convention, so F Z F† X = 1.
        let f = fourier::<C64>(d).unwrap();
        assert!(f.unitarity_deviation() < 1e-12);
        let fdag =
            SiteMatrix::from_vec(d, (0..d * d).map(|k| f.get(k % d, k / d).conj()).collect())
                .unwrap();
        let conj = f.matmul(&z).unwrap().matmul(&fdag).unwrap();
        assert_matrix_close(&conj.matmul(&x).unwrap(), &id, 1e-12);
    }
}

#[test]
fn permutation_gates_exist_over_every_algebra_and_phases_only_where_they_embed() {
    assert!(shift::<f64>(5).unitarity_deviation() < 1e-15);
    assert!(cshift::<Octonion>(3, 5).unitarity_deviation() < 1e-15);
    assert!(swap::<SplitComplex>(4).unitarity_deviation() < 1e-15);
    // Over ℝ only the radix-2 Fourier transform (the Hadamard) and clock
    // (Z) exist; radix 3 needs ω = e^{2πi/3}.
    assert!(fourier::<f64>(2).is_ok());
    assert!(clock::<f64>(2).is_ok());
    assert!(matches!(
        fourier::<f64>(3),
        Err(Error::UnsupportedForAlgebra { .. })
    ));
    assert!(fourier::<Quaternion>(5).is_ok());
}

// ── backends on mixed spaces, against an independent oracle ─────────

#[test]
fn dense_and_sparse_match_the_compound_register_on_random_mixed_circuits() {
    let space = Space::new(&ZIGZAG).unwrap();
    for seed in 0..4u64 {
        let mut rng = Prng::new(100 + seed);
        let mut dense = RadixDenseState::<C64>::new(space.clone()).unwrap();
        let mut sparse = RadixSparseState::<C64>::new(space.clone()).unwrap();
        let mut oracle = CompoundRegister::new(&ZIGZAG).unwrap();
        for _ in 0..30 {
            let (sites, m) = random_step(&space, &mut rng);
            let sm = SiteMatrix::from_vec(space.sub_dim(&sites), m.clone()).unwrap();
            dense.apply_sites(&sm, &sites).unwrap();
            sparse.apply_sites(&sm, &sites).unwrap();
            oracle.apply_k(&sites, &m).unwrap();
        }
        let mut worst = 0.0f64;
        for index in 0..space.dim() as u64 {
            let want = oracle.amplitude(&space.decode(index)).unwrap();
            worst = worst.max((dense.amplitude(index) - want).norm());
        }
        assert!(worst < 1e-10, "seed {seed}: dense vs oracle {worst}");
        assert!(max_amplitude_deviation(&dense, &sparse) < 1e-10);
        common::assert_close(dense.total_weight(), 1.0, 1e-10);
    }
}

#[test]
fn dense_and_sparse_agree_over_the_quaternions() {
    let space = Space::new(&[3, 2, 5]).unwrap();
    let mut rng = Prng::new(7);
    let mut dense = RadixDenseState::<Quaternion>::new(space.clone()).unwrap();
    let mut sparse = RadixSparseState::<Quaternion>::new(space.clone()).unwrap();
    for _ in 0..20 {
        let (sites, m) = random_step(&space, &mut rng);
        let sm = SiteMatrix::<Quaternion>::try_from_c64s("u", space.sub_dim(&sites), &m).unwrap();
        dense.apply_sites(&sm, &sites).unwrap();
        sparse.apply_sites(&sm, &sites).unwrap();
    }
    assert!(max_amplitude_deviation(&dense, &sparse) < 1e-10);
    common::assert_close(dense.total_weight(), 1.0, 1e-10);
}

#[test]
fn permutation_site_gates_cause_no_sparse_fill() {
    let space = Space::new(&ZIGZAG).unwrap();
    let mut c = SiteCircuit::<C64>::new(space.clone());
    c.shift(3)
        .unwrap()
        .cshift(3, 1)
        .unwrap()
        .swap(1, 5)
        .unwrap();
    c.cshift(5, 2).unwrap().swap(0, 6).unwrap();
    let sim: Simulator = Simulator::new();
    let state = sim.run_space("sparse", &c).unwrap();
    assert_eq!(state.nonzero_count(), 1);
    // shift(3): d3 = 1; cshift(3,1): d1 += 1 → 1; swap(1,5): d5 = 1, d1 = 0;
    // cshift(5,2): d2 += 1 → 1; swap(0,6): nothing to move.
    let want = space.encode(&[0, 0, 1, 1, 0, 1, 0]).unwrap();
    common::assert_close(state.probability(want), 1.0, 1e-12);
}

// ── the qubit case is unchanged ──────────────────────────────────────

#[test]
fn a_qubit_space_is_exactly_the_qubit_backend() {
    let mut circuit = Circuit::new(3);
    circuit.h(0).cx(0, 1).t(1).ry(2, 0.4).cz(1, 2);
    let sim: Simulator = Simulator::new();
    let want = sim.run(&circuit).unwrap();

    let bound = circuit.bind(sim.registry()).unwrap();
    let mut placed = SiteCircuit::<C64>::new(Space::qubits(3));
    placed.place(&bound, &[0, 1, 2]).unwrap();
    for backend in ["dense", "sparse", "mps", "factored"] {
        let got = sim.run_space(backend, &placed).unwrap();
        assert!(got.space().is_qubits());
        assert!(max_amplitude_deviation(got.as_ref(), want.as_ref()) < 1e-12);
    }
    // The general mixed-radix kernels, handed a qubit space directly,
    // reproduce the qubit kernels: an independent check of both.
    for got in [
        Box::new(RadixDenseState::<C64>::new(Space::qubits(3)).unwrap()) as Box<dyn Backend<C64>>,
        Box::new(RadixSparseState::<C64>::new(Space::qubits(3)).unwrap()),
    ] {
        let mut got = got;
        bound.run(got.as_mut()).unwrap();
        assert!(max_amplitude_deviation(got.as_ref(), want.as_ref()) < 1e-12);
    }
    let d = RadixDenseState::<C64>::new(Space::qubits(4)).unwrap();
    assert_eq!(d.num_qubits(), 4);
    assert_eq!(d.amplitudes().len(), 16);
}

#[test]
fn qubit_gates_run_on_the_binary_sites_of_a_mixed_register() {
    // A Bell pair across the two radix-2 ends of the zigzag.
    let space = Space::new(&ZIGZAG).unwrap();
    let sim: Simulator = Simulator::new();
    let mut bell = Circuit::new(2);
    bell.h(0).cx(0, 1);
    let bound = bell.bind(sim.registry()).unwrap();
    let mut c = SiteCircuit::<C64>::new(space.clone());
    c.fourier(3).unwrap().place(&bound, &[0, 6]).unwrap();

    let h = std::f64::consts::FRAC_1_SQRT_2;
    let mut oracle = CompoundRegister::new(&ZIGZAG).unwrap();
    oracle
        .apply_k(&[3], &quantsim::mixed::fourier_d(5))
        .unwrap();
    let hm = [z(h, 0.0), z(h, 0.0), z(h, 0.0), z(-h, 0.0)];
    oracle.apply_k(&[0], &hm).unwrap();
    let one = z(1.0, 0.0);
    let zero = z(0.0, 0.0);
    #[rustfmt::skip]
    let cx = [
        one, zero, zero, zero,
        zero, zero, zero, one,
        zero, zero, one, zero,
        zero, one, zero, zero,
    ];
    oracle.apply_k(&[0, 6], &cx).unwrap();

    for backend in ["dense", "sparse"] {
        let state = sim.run_space(backend, &c).unwrap();
        for index in 0..space.dim() as u64 {
            let want = oracle.amplitude(&space.decode(index)).unwrap();
            assert!((state.amplitude(index) - want).norm() < 1e-12, "{backend}");
        }
        let p00 = state.probability(space.encode(&[0, 0, 0, 2, 0, 0, 0]).unwrap());
        let p11 = state.probability(space.encode(&[1, 0, 0, 2, 0, 0, 1]).unwrap());
        common::assert_close(p00, 0.1, 1e-12);
        common::assert_close(p11, 0.1, 1e-12);
    }
}

// ── measurement ──────────────────────────────────────────────────────

#[test]
fn measuring_a_wide_site_collapses_it_with_born_statistics() {
    let space = Space::new(&[3, 5, 2]).unwrap();
    let mut counts = [0usize; 5];
    let shots = 2000;
    let mut rng = Prng::new(9);
    for _ in 0..shots {
        let mut st = RadixDenseState::<C64>::new(space.clone()).unwrap();
        st.apply_sites(&fourier(5).unwrap(), &[1]).unwrap();
        let d = st.measure_site(1, &mut rng).unwrap();
        counts[d] += 1;
        // Collapsed onto that digit, renormalized.
        let idx = space.encode(&[0, d, 0]).unwrap();
        common::assert_close(st.probability(idx), 1.0, 1e-12);
        common::assert_close(st.total_weight(), 1.0, 1e-12);
    }
    for &k in &counts {
        let f = k as f64 / shots as f64;
        assert!((f - 0.2).abs() < 0.04, "frequency {f} far from 1/5");
    }
}

#[test]
fn sparse_measurement_matches_dense_seed_for_seed() {
    let space = Space::new(&[4, 3, 2]).unwrap();
    let mut rng_unitary = Prng::new(31);
    let mut circuit = SiteCircuit::<C64>::new(space.clone());
    for _ in 0..10 {
        let (sites, m) = random_step(&space, &mut rng_unitary);
        let sm = SiteMatrix::from_vec(space.sub_dim(&sites), m).unwrap();
        circuit.gate("u", sm, &sites).unwrap();
    }
    let sim: Simulator = Simulator::new();
    let mut dense = sim.run_space("dense", &circuit).unwrap();
    let mut sparse = sim.run_space("sparse", &circuit).unwrap();
    let (mut ra, mut rb) = (Prng::new(5), Prng::new(5));
    for site in [0, 2, 1] {
        assert_eq!(
            dense.measure_site(site, &mut ra).unwrap(),
            sparse.measure_site(site, &mut rb).unwrap()
        );
    }
    assert!(max_amplitude_deviation(dense.as_ref(), sparse.as_ref()) < 1e-10);
}

#[test]
fn on_qubit_spaces_site_measurement_is_qubit_measurement() {
    let mut circuit = Circuit::new(3);
    circuit.h(0).h(1).cx(1, 2).ry(0, 0.3);
    let sim: Simulator = Simulator::new();
    let mut a = sim.run(&circuit).unwrap();
    let mut b = sim.run(&circuit).unwrap();
    let (mut ra, mut rb) = (Prng::new(77), Prng::new(77));
    for q in 0..3 {
        assert_eq!(
            usize::from(a.measure(q, &mut ra).unwrap()),
            b.measure_site(q, &mut rb).unwrap()
        );
    }
}

#[test]
fn sampling_a_mixed_register_returns_mixed_radix_indices() {
    let space = Space::new(&[3, 2]).unwrap();
    let mut c = SiteCircuit::<C64>::new(space.clone());
    c.fourier(0).unwrap().cshift(0, 1).unwrap();
    let sim: Simulator = Simulator::new();
    let state = sim.run_space("dense", &c).unwrap();
    let counts = state.sample(3000, &mut Prng::new(3)).unwrap();
    // |a, a mod 2⟩ for a ∈ {0, 1, 2}: digits (0,0), (1,1), (2,0).
    let mut seen: Vec<Vec<usize>> = counts.keys().map(|&i| space.decode(i)).collect();
    seen.sort();
    assert_eq!(seen, vec![vec![0, 0], vec![1, 1], vec![2, 0]]);
    for &n in counts.values() {
        assert!((n as f64 / 3000.0 - 1.0 / 3.0).abs() < 0.04);
    }
}

#[test]
fn permutations_respect_the_mixed_radix_bound() {
    let space = Space::new(&[3, 5]).unwrap();
    let mut st = RadixSparseState::<C64>::new(space.clone()).unwrap();
    st.apply_sites(&fourier(3).unwrap(), &[0]).unwrap();
    // Cycle site 1's digit: a genuine permutation of the 15 indices.
    let sp = space.clone();
    st.apply_permutation("cycle", &move |i| {
        sp.with_digit(i, 1, (sp.digit(i, 1) + 1) % 5)
    })
    .unwrap();
    for a in 0..3 {
        let idx = space.encode(&[a, 1]).unwrap();
        common::assert_close(st.probability(idx), 1.0 / 3.0, 1e-12);
    }
    // An index past 15 is outside the register, even though it fits 4 bits.
    assert!(st.apply_permutation("escape", &|i| i + 15).is_err());
}

// ── refusals ─────────────────────────────────────────────────────────

#[test]
fn qubit_operations_refuse_non_binary_sites() {
    let space = Space::new(&[2, 3, 2]).unwrap();
    let mut st = RadixDenseState::<C64>::new(space.clone()).unwrap();
    let x =
        GateMatrix::<C64>::from_vec(2, vec![z(0.0, 0.0), z(1.0, 0.0), z(1.0, 0.0), z(0.0, 0.0)])
            .unwrap();
    assert!(st.apply(&x, &[0]).is_ok());
    assert!(matches!(
        st.apply(&x, &[1]),
        Err(Error::UnsupportedForSpace { .. })
    ));
    assert!(matches!(
        st.apply_diagonal(&[z(1.0, 0.0), z(-1.0, 0.0)], &[1]),
        Err(Error::UnsupportedForSpace { .. })
    ));
    assert!(matches!(
        st.measure(1, &mut Prng::new(1)),
        Err(Error::UnsupportedForSpace { .. })
    ));
    assert!(matches!(
        pauli_expectation(&st, &[(0, Pauli::Z)]),
        Err(Error::UnsupportedForSpace { .. })
    ));
    // A qubit-only backend takes site gates through the default, which
    // routes radix-2 matrices to `apply` and refuses any other dimension.
    let mut mps = MpsState::<C64>::new(3).unwrap();
    assert!(mps.apply_sites(&shift(2), &[1]).is_ok());
    common::assert_close(mps.probability(0b010), 1.0, 1e-12);
    assert!(matches!(
        mps.apply_sites(&shift(3), &[1]),
        Err(Error::BadDimension { .. })
    ));
    let mut sparse = RadixSparseState::<C64>::new(space).unwrap();
    assert!(matches!(
        sparse.apply(&x, &[1]),
        Err(Error::UnsupportedForSpace { .. })
    ));
    assert!(matches!(
        sparse.project_digit(1, 3, 1.0),
        Err(Error::InvalidState(_))
    ));
}

#[test]
fn circuits_refuse_the_wrong_shapes() {
    let space = Space::new(&[2, 3, 5]).unwrap();
    let mut c = SiteCircuit::<C64>::new(space.clone());
    assert!(matches!(
        c.swap(1, 2),
        Err(Error::UnsupportedForSpace { .. })
    ));
    assert!(matches!(
        c.gate("wrong-dim", shift(3), &[2]),
        Err(Error::BadDimension { .. })
    ));
    let not_unitary = SiteMatrix::from_vec(3, vec![z(1.0, 0.0); 9]).unwrap();
    assert!(matches!(
        c.gate("ones", not_unitary, &[1]),
        Err(Error::NotUnitary { .. })
    ));
    let sim: Simulator = Simulator::new();
    let mut bell = Circuit::new(2);
    bell.h(0).cx(0, 1);
    let bound = bell.bind(sim.registry()).unwrap();
    assert!(matches!(
        c.place(&bound, &[0, 1]),
        Err(Error::UnsupportedForSpace { .. })
    ));
    // Backends without native mixed-radix storage are refused by name.
    c.fourier(2).unwrap();
    assert!(matches!(
        sim.run_space("mps", &c),
        Err(Error::UnsupportedForSpace { .. })
    ));
    assert!(matches!(
        sim.run_space("no-such-backend", &c),
        Err(Error::UnknownBackend(_))
    ));
    // A backend holding a different space is refused.
    let mut other = RadixDenseState::<C64>::new(Space::new(&[2, 5, 3]).unwrap()).unwrap();
    assert!(matches!(
        c.run(&mut other),
        Err(Error::UnsupportedForSpace { .. })
    ));
    assert_eq!(
        sim.backends().space_names(),
        vec!["dense".to_string(), "sparse".to_string()]
    );
}
