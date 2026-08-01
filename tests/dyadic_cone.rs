//! The dyadic cone: that every node of the tree agrees, that children
//! partition their parent so the additive quantities are conserved by
//! construction, that the extremal ones actually resolve, and — the
//! point — that the two-directional error bound is rigorous and tighter
//! than the one-directional one it replaces.

use quantsim::backend::pauli_expectation;
use quantsim::dyadic::{self, Config};
use quantsim::heisenberg::*;
use quantsim::horizon::Forward;
use quantsim::prelude::*;

fn cfg(depth: usize) -> Config {
    Config {
        depth,
        threshold: 0.0,
        forward: Forward::Dense,
        two_sided_max_terms: 0,
    }
}

fn chain(n: usize, steps: usize) -> Vec<Rotation> {
    tfim_trotter(n, 1.0, 0.7, 0.35, steps)
}

fn ops_of(key: PauliKey, n: usize) -> Vec<(usize, Pauli)> {
    (0..n)
        .filter_map(|q| match (key.0 >> q & 1, key.1 >> q & 1) {
            (1, 0) => Some((q, Pauli::X)),
            (0, 1) => Some((q, Pauli::Z)),
            (1, 1) => Some((q, Pauli::Y)),
            _ => None,
        })
        .collect()
}

fn dense_at(rotations: &[Rotation], n: usize) -> DenseState<C64> {
    let mut d = DenseState::<C64>::new(n).unwrap();
    for r in rotations {
        let (m, s) = r.gate().unwrap();
        d.apply(&m, &s).unwrap();
    }
    d
}

fn dense_value(rotations: &[Rotation], n: usize, key: PauliKey) -> f64 {
    let d = dense_at(rotations, n);
    let herm = pauli_expectation(&d as &dyn Backend<C64>, &ops_of(key, n)).unwrap();
    (herm / axis_operator_phase(key)).re
}

// ── the tree agrees with itself and with dense ───────────────────────

#[test]
fn every_node_of_the_tree_returns_the_same_number() {
    for &(n, steps, depth) in &[(7usize, 3usize, 3usize), (9, 4, 4), (8, 3, 2)] {
        let rots = chain(n, steps);
        let obs = (0u64, 1u64 << (n / 2));
        let c = dyadic::dyadic_cone(obs, &rots, n, &cfg(depth)).unwrap();
        let truth = dense_value(&rots, n, obs);

        assert_eq!(
            c.nodes.len(),
            (1usize << (depth + 1)) - 1,
            "a depth-{depth} binary tree has 2^(d+1)-1 nodes"
        );
        assert!(
            c.value_spread() < 1e-11,
            "n={n}: the tree disagreed by {:.3e} across {} independent contractions",
            c.value_spread(),
            c.nodes.len()
        );
        for node in &c.nodes {
            assert!(
                (node.value - truth).abs() < 1e-11,
                "n={n} L{} #{}: {} vs dense {truth}",
                node.level,
                node.index,
                node.value
            );
        }
    }
}

#[test]
fn children_partition_their_parents_gates_exactly() {
    let (n, depth) = (9usize, 4usize);
    let rots = chain(n, 4);
    let c = dyadic::dyadic_cone((0, 1u64 << 4), &rots, n, &cfg(depth)).unwrap();

    for level in 0..depth {
        let parents = c.level(level);
        let children = c.level(level + 1);
        for (i, p) in parents.iter().enumerate() {
            let (l, r) = (children[2 * i], children[2 * i + 1]);
            assert_eq!(l.span.0, p.span.0, "left child must start where the parent does");
            assert_eq!(r.span.1, p.span.1, "right child must end where the parent does");
            assert_eq!(l.span.1, r.span.0, "the children must meet, with no gap or overlap");
            assert_eq!(
                l.live_cells + r.live_cells,
                p.live_cells,
                "L{level} #{i}: live area must be additive across the split"
            );
            assert_eq!(l.cells + r.cells, p.cells);
            assert_eq!(l.advance + r.advance, p.advance, "the front's motion telescopes");
        }
    }
}

// ── conserved versus resolved ────────────────────────────────────────

#[test]
fn the_additive_quantities_are_conserved_at_every_scale() {
    // Not a physics result — a check the recursion performs on itself.
    // Children partition their parent's gates, so any measure summed
    // over a level cannot move with the level. Drift means the
    // bisection lost or double-counted cells.
    let (n, depth) = (10usize, 5usize);
    let rots = chain(n, 5);
    let c = dyadic::dyadic_cone((0, 1u64 << 5), &rots, n, &cfg(depth)).unwrap();

    let density = c.density_by_level();
    let velocity = c.velocity_by_level();
    assert_eq!(density.len(), depth + 1);
    for l in 1..=depth {
        assert!(
            (density[l] - density[0]).abs() < 1e-12,
            "live-area density drifted at L{l}: {density:?}"
        );
        assert!(
            (velocity[l] - velocity[0]).abs() < 1e-12,
            "front velocity drifted at L{l}: {velocity:?}"
        );
    }
    assert!(density[0] > 0.0 && density[0] < 1.0, "{density:?}");
}

#[test]
fn the_extremal_quantities_resolve_as_the_window_halves() {
    // The other half of the statement: what a coarse window averages
    // away, a finer one separates. A coarse node holds the front's
    // bursts together with the stretches where it is saturated and
    // cannot move at all, so the peak local velocity it reports is too
    // low — and climbs every time the window halves.
    let (n, depth) = (10usize, 5usize);
    let rots = chain(n, 5);
    let c = dyadic::dyadic_cone((0, 1u64 << 5), &rots, n, &cfg(depth)).unwrap();

    let peaks = c.peak_velocity_by_level();
    for l in 1..=depth {
        assert!(
            peaks[l] >= peaks[l - 1] - 1e-12,
            "a finer window cannot report a LOWER peak: {peaks:?}"
        );
    }
    assert!(
        c.resolution_gain() > 1.5,
        "the bisection resolved almost nothing: {peaks:?}"
    );
    // and the mean diamond narrows, since a shorter span overlaps fewer
    // qubits even though the total live area is fixed
    let widths = c.mean_diamond_by_level();
    assert!(
        widths[depth] < widths[0],
        "mean diamond width should fall with depth: {widths:?}"
    );
}

#[test]
fn a_deeper_tree_can_only_sharpen_the_peak() {
    let n = 10;
    let rots = chain(n, 5);
    let obs = (0u64, 1u64 << 5);
    let shallow = dyadic::dyadic_cone(obs, &rots, n, &cfg(2)).unwrap();
    let deep = dyadic::dyadic_cone(obs, &rots, n, &cfg(5)).unwrap();
    assert!(
        deep.resolution_gain() >= shallow.resolution_gain() - 1e-12,
        "shallow {:.3} vs deep {:.3}",
        shallow.resolution_gain(),
        deep.resolution_gain()
    );
    // the two trees agree on the answer regardless of how finely cut
    assert!((shallow.nodes[0].value - deep.nodes[0].value).abs() < 1e-12);
}

// ── where the two directions interact ────────────────────────────────

#[test]
fn the_two_sided_bound_is_never_looser_and_is_sometimes_much_tighter() {
    // Without the state, |⟨ψ|P|ψ⟩| ≤ 1 is all that is known, so Σ|c| is
    // the only sound bound. At a cut the state IS known, so
    // Σ|c|·|⟨ψ|P|ψ⟩| is computable — and it can only be smaller.
    let n = 9;
    let rots = chain(n, 4);
    let c = dyadic::dyadic_cone((0, 1u64 << 4), &rots, n, &cfg(4)).unwrap();

    let mut evaluated = 0usize;
    for node in &c.nodes {
        let Some(two) = node.two_sided_l1 else { continue };
        evaluated += 1;
        assert!(
            two <= node.l1 + 1e-12,
            "L{} #{}: two-sided {two} exceeded one-sided {}",
            node.level,
            node.index,
            node.l1
        );
    }
    assert!(evaluated > 0, "no node evaluated the two-sided sum");
    assert!(
        c.tightening() > 2.0,
        "the two directions bought nothing: best {:.2}x",
        c.tightening()
    );
    assert!(c.mean_tightening() >= 1.0);
}

#[test]
fn the_two_sided_prune_holds_the_error_it_certifies() {
    // The claim that makes the bound worth having: drop terms under a
    // budget, contract what is left, and the real error is inside the
    // certified one. Every budget, on a real circuit, against dense.
    let n = 9;
    let rots = chain(n, 4);
    let obs = (0u64, 1u64 << 4);
    let truth = dense_value(&rots, n, obs);
    let cut = rots.len() / 2;

    let back = propagate(
        &PauliSum::from_key(obs),
        &rots[cut..],
        &quantsim::heisenberg::Config {
            threshold: 0.0,
            max_terms: None,
            checkpoint_every: 0,
            exclusion: false,
            retire_frozen: false,
        },
    )
    .unwrap()
    .sum;
    let fwd = dense_at(&rots[..cut], n);
    let state = &fwd as &dyn Backend<C64>;

    let (one, two) = dyadic::two_sided_bound(&back, state).unwrap();
    assert!(two <= one + 1e-12, "two-sided {two} vs one-sided {one}");
    assert!(one / two > 1.5, "no tightening at this cut: {:.2}x", one / two);

    let mut ever_dropped = 0usize;
    for budget in [1e-6f64, 1e-4, 1e-2, 1e-1, 0.3] {
        let (kept, dropped, spent) = dyadic::prune_two_sided(&back, state, budget).unwrap();
        assert!(spent <= budget + 1e-12, "spent {spent} over budget {budget}");
        assert_eq!(kept.len() + dropped, back.len(), "terms went missing");
        ever_dropped += dropped;

        let mut acc = C64::new(0.0, 0.0);
        for (key, coeff) in kept.terms() {
            let e = pauli_expectation(state, &ops_of(key, n)).unwrap() / axis_operator_phase(key);
            acc += coeff * e;
        }
        let actual = (acc.re - truth).abs();
        assert!(
            actual <= spent + 1e-11,
            "budget {budget:.0e}: dropped {dropped}, certified {spent:.3e}, actual {actual:.3e}"
        );
    }
    assert!(ever_dropped > 0, "no budget ever dropped anything");
}

#[test]
fn a_budget_of_zero_drops_nothing_and_a_huge_one_drops_everything() {
    let n = 7;
    let rots = chain(n, 3);
    let obs = (0u64, 1u64 << 3);
    let cut = rots.len() / 2;
    let back = propagate(
        &PauliSum::from_key(obs),
        &rots[cut..],
        &quantsim::heisenberg::Config {
            threshold: 0.0,
            max_terms: None,
            checkpoint_every: 0,
            exclusion: false,
            retire_frozen: false,
        },
    )
    .unwrap()
    .sum;
    let fwd = dense_at(&rots[..cut], n);
    let state = &fwd as &dyn Backend<C64>;

    let (kept, dropped, spent) = dyadic::prune_two_sided(&back, state, 0.0).unwrap();
    assert_eq!(dropped, 0, "a zero budget must keep everything");
    assert_eq!(kept.len(), back.len());
    assert_eq!(spent, 0.0);

    let (_, two) = dyadic::two_sided_bound(&back, state).unwrap();
    let (kept, dropped, _) = dyadic::prune_two_sided(&back, state, two * 2.0).unwrap();
    assert_eq!(dropped, back.len(), "a budget above the total must drop all");
    assert_eq!(kept.len(), 0);
}

// ── refusals ─────────────────────────────────────────────────────────

#[test]
fn a_circuit_too_short_to_bisect_is_refused_rather_than_folded() {
    let rots = chain(4, 1);
    // 4 qubits, 1 step is 7 gates; depth 4 wants 16 segments
    assert!(dyadic::dyadic_cone((0, 1), &rots, 4, &cfg(4)).is_err());
    assert!(dyadic::dyadic_cone((0, 1), &rots, 4, &cfg(2)).is_ok());
}

#[test]
fn an_impossible_depth_or_width_is_refused() {
    let rots = chain(6, 3);
    assert!(dyadic::dyadic_cone((0, 1), &rots, 0, &cfg(2)).is_err());
    assert!(dyadic::dyadic_cone((0, 1), &rots, 65, &cfg(2)).is_err());
    assert!(dyadic::dyadic_cone((0, 1), &rots, 6, &cfg(dyadic::MAX_DEPTH + 1)).is_err());
}

#[test]
fn depth_zero_is_a_single_node_that_still_answers() {
    let n = 7;
    let rots = chain(n, 3);
    let obs = (0u64, 1u64 << 3);
    let c = dyadic::dyadic_cone(obs, &rots, n, &cfg(0)).unwrap();
    assert_eq!(c.nodes.len(), 1);
    assert_eq!(c.nodes[0].span, (0, rots.len()));
    assert!((c.nodes[0].value - dense_value(&rots, n, obs)).abs() < 1e-11);
    assert_eq!(c.value_spread(), 0.0);
    assert_eq!(c.resolution_gain(), 1.0, "one scale resolves nothing");
}

#[test]
fn a_forward_representation_that_holds_the_state_gives_the_same_tree() {
    let n = 8;
    let rots = chain(n, 3);
    let obs = (0u64, 1u64 << 4);
    let dense = dyadic::dyadic_cone(obs, &rots, n, &cfg(3)).unwrap();
    for forward in [Forward::Sparse, Forward::Mps { max_bond: 64 }] {
        let other = dyadic::dyadic_cone(
            obs,
            &rots,
            n,
            &Config {
                forward,
                ..cfg(3)
            },
        )
        .unwrap();
        for (a, b) in dense.nodes.iter().zip(&other.nodes) {
            assert!((a.value - b.value).abs() < 1e-11, "{forward:?}");
            assert_eq!(a.diamond, b.diamond);
            assert_eq!(a.live_cells, b.live_cells);
        }
    }
}

#[test]
fn truncation_shows_up_as_the_tree_ceasing_to_agree() {
    // The same self-check the line of cuts gives, but over a tree: an
    // exact walk closes to machine precision and a truncated one does
    // not, and the spread is the honest size of the error.
    let n = 8;
    let rots = chain(n, 4);
    let obs = (0u64, 1u64 << 4);
    let mut previous = f64::INFINITY;
    for &th in &[1e-2f64, 1e-3, 1e-4, 0.0] {
        let c = dyadic::dyadic_cone(
            obs,
            &rots,
            n,
            &Config {
                threshold: th,
                ..cfg(3)
            },
        )
        .unwrap();
        let spread = c.value_spread();
        assert!(
            spread <= previous + 1e-12,
            "spread grew as the threshold tightened: {spread} after {previous}"
        );
        previous = spread;
    }
    assert!(previous < 1e-11, "the exact tree should close: {previous}");
}
