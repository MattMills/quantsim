//! The tail-radix phase web: that the mixed-radix Fourier kernel really
//! does factorize over the tail, that the result is banded, and what the
//! band costs to hold exactly.

use quantsim::pathsum;
use quantsim::radixweb::{
    bandwidth, digits, dyadic_share, full_web, phase_web, reversed_digits, segment,
    verify_factorization,
};
use std::f64::consts::TAU;

/// **The identity, checked rather than assumed.** Every `x, y` in the
/// ring, against the kernel evaluated directly.
#[test]
fn the_kernel_factorizes_over_the_tail_exactly() {
    for profile in [
        vec![2usize, 2, 2],
        vec![2, 3, 4],
        vec![3, 5, 7],
        vec![2, 3, 4, 3, 2],
        vec![4, 2, 6],
    ] {
        let worst = verify_factorization(&profile).expect("within the sweep ceiling");
        assert!(
            worst < 1e-12,
            "profile {profile:?}: the tail factorization deviated from the kernel \
             by {worst} turns. The whole banding argument rests on this identity."
        );
    }
}

/// The half that vanishes, and *why*: for `i < j` the place values
/// contribute `Π_{i<k<j} d_k`, an integer, so the phase is a whole
/// number of turns.
#[test]
fn the_couplings_below_the_diagonal_carry_whole_turns() {
    let profile = [2usize, 3, 4, 5];
    let n: u128 = profile.iter().map(|&d| d as u128).product();
    for i in 0..profile.len() {
        for j in 0..profile.len() {
            if i >= j {
                continue;
            }
            // place(x_i) · place(y'_j) / N, which should be an integer.
            let place_x: u128 = profile[i + 1..].iter().map(|&d| d as u128).product();
            let place_y: u128 = profile[..j].iter().map(|&d| d as u128).product();
            let num = place_x * place_y;
            assert_eq!(
                num % n,
                0,
                "coupling i={i} j={j} did not carry a whole turn, so the web is \
                 not lower-triangular after all"
            );
        }
    }
}

/// The digit conventions round-trip, so the factorization is not being
/// verified against a bug in the indexing.
#[test]
fn the_two_place_value_conventions_reconstruct_their_inputs() {
    let profile = [2usize, 3, 4, 5];
    let n: u128 = profile.iter().map(|&d| d as u128).product();
    for v in 0..n {
        let d = digits(&profile, v);
        let mut place: u128 = n;
        let mut acc = 0u128;
        for (k, &dk) in profile.iter().enumerate() {
            place /= dk as u128;
            acc += d[k] * place;
        }
        assert_eq!(acc, v, "ordinary place value did not round-trip at {v}");

        let r = reversed_digits(&profile, v);
        let mut place = 1u128;
        let mut acc = 0u128;
        for (k, &dk) in profile.iter().enumerate() {
            acc += r[k] * place;
            place *= dk as u128;
        }
        assert_eq!(acc, v, "reversed place value did not round-trip at {v}");
    }
}

/// **Banding.** The full web is a triangle of `n(n+1)/2` couplings; above
/// any fixed angle threshold only a band survives, and the band does not
/// grow as the chain does.
#[test]
fn the_web_is_banded_and_the_band_does_not_grow_with_the_chain() {
    let min_angle = TAU / 1024.0;
    println!("   n   full triangle   banded   bandwidth");
    let mut widths = Vec::new();
    for n in [4usize, 8, 16, 32, 64] {
        let profile: Vec<usize> = (0..n).map(|_| 2usize).collect();
        let full = full_web(&profile).len();
        let banded = phase_web(&profile, min_angle).len();
        let bw = bandwidth(&profile, min_angle);
        println!("  {n:3} {full:15} {banded:8} {bw:11}");
        assert_eq!(full, n * (n + 1) / 2, "the full web should be the triangle");
        assert!(banded <= full, "banding cannot add couplings");
        widths.push(bw);
    }
    // The bandwidth saturates: the segment product passes the threshold
    // after a fixed number of steps regardless of how long the chain is.
    let tail = &widths[2..];
    assert!(
        tail.windows(2).all(|p| p[0] == p[1]),
        "bandwidth kept growing with n ({widths:?}); the whole point of the \
         radix segment product is that it saturates"
    );
    // The saving is linear-versus-quadratic, so it IMPROVES with n — that
    // is the signature, not any particular percentage at one size.
    let ratio = |n: usize| {
        let p: Vec<usize> = (0..n).map(|_| 2usize).collect();
        full_web(&p).len() as f64 / phase_web(&p, min_angle).len() as f64
    };
    let (small, large) = (ratio(16), ratio(64));
    println!("   full/banded: n=16 {small:.2}, n=64 {large:.2}");
    assert!(
        large > 2.0 * small,
        "the banding saving did not improve with n ({small:.2} -> {large:.2}); \
         banded should grow linearly while the triangle grows quadratically"
    );
    // Stated directly: banded is bounded by n times the saturated bandwidth.
    for n in [16usize, 32, 64] {
        let p: Vec<usize> = (0..n).map(|_| 2usize).collect();
        let banded = phase_web(&p, min_angle).len();
        let bw = bandwidth(&p, min_angle);
        assert!(
            banded <= n * bw,
            "n={n}: {banded} couplings exceeds n·bandwidth = {}",
            n * bw
        );
    }
}

/// A segment product is a product over every dimension in between, so it
/// grows at least like `2^span` and the angle decays at least
/// exponentially. That is the mechanism behind the band.
#[test]
fn the_segment_product_grows_at_least_exponentially_in_the_span() {
    for profile in [vec![2usize; 12], vec![2, 3, 4, 5, 4, 3, 2], vec![3usize; 8]] {
        for j in 0..profile.len() {
            for i in j..profile.len() {
                let d = segment(&profile, j, i);
                assert!(
                    d >= 1u128 << (i - j),
                    "segment {j}..{i} of {profile:?} was {d}, below 2^span"
                );
            }
        }
    }
}

/// **What the band costs to hold exactly**, which is where this meets
/// `pathsum`.
///
/// The coupling angles are `2π/D` for `D` a product of local dimensions,
/// so they are roots of unity of composite order. `pathsum` is exact over
/// dyadic angles and refuses the rest rather than rounding. On a binary
/// chain every denominator is a power of two and the whole web is
/// representable; a single odd dimension poisons every segment spanning
/// it, and the share collapses.
#[test]
fn a_binary_chain_is_wholly_dyadic_and_one_odd_dimension_wrecks_it() {
    let binary = vec![2usize; 8];
    assert_eq!(
        dyadic_share(&binary),
        1.0,
        "every segment of a binary chain is a power of two"
    );
    // Confirm against pathsum itself rather than by inspection: it must
    // accept every angle in the binary web and refuse the odd ones.
    for c in full_web(&binary) {
        assert!(
            pathsum::turn_from_radians(c.angle()).is_ok(),
            "pathsum refused a binary-chain coupling of denominator {}",
            c.denom
        );
    }

    println!("   profile                     dyadic share");
    for profile in [
        vec![2usize; 8],
        vec![2, 2, 2, 3, 2, 2, 2, 2],
        vec![2, 3, 4, 5, 4, 3, 2],
        vec![3usize; 8],
    ] {
        let share = dyadic_share(&profile);
        println!("   {profile:?}   {share:.3}");
    }

    let poisoned = vec![2usize, 2, 2, 3, 2, 2, 2, 2];
    let share = dyadic_share(&poisoned);
    assert!(
        share < 1.0,
        "one odd dimension should cost some of the web, got {share}"
    );
    // And pathsum really does refuse those.
    let refused = full_web(&poisoned)
        .iter()
        .filter(|c| pathsum::turn_from_radians(c.angle()).is_err())
        .count();
    assert!(
        refused > 0,
        "no coupling was refused, so the dyadic restriction costs nothing here"
    );
    println!(
        "   pathsum refuses {refused} of {} couplings on {poisoned:?}",
        full_web(&poisoned).len()
    );

    // An all-odd chain keeps only the trivial single-site segments that
    // happen to be powers of two — none, for dimension 3.
    assert_eq!(
        dyadic_share(&[3usize; 8]),
        0.0,
        "no segment of an all-ternary chain is a power of two"
    );
}
