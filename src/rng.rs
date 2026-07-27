//! Deterministic pseudo-random number generation.
//!
//! The crate ships its own small PRNG ([xoshiro256++] seeded via [SplitMix64])
//! rather than depending on an external `rand` version, so measurement and
//! sampling results are bit-for-bit reproducible across platforms and across
//! releases of this crate. It is **not** cryptographically secure.
//!
//! [xoshiro256++]: https://prng.di.unimi.it/
//! [SplitMix64]: https://prng.di.unimi.it/splitmix64.c

/// A small, fast, deterministic PRNG (xoshiro256++).
///
/// ```
/// use quantsim::rng::Prng;
/// let mut a = Prng::new(42);
/// let mut b = Prng::new(42);
/// assert_eq!(a.next_u64(), b.next_u64());
/// ```
#[derive(Debug, Clone)]
pub struct Prng {
    s: [u64; 4],
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Prng {
    /// Create a generator from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            *slot = splitmix64(&mut sm);
        }
        // Guard against the (practically impossible) all-zero state.
        if s == [0, 0, 0, 0] {
            s[0] = 0x9E37_79B9_7F4A_7C15;
        }
        Prng { s }
    }

    /// Next uniformly distributed 64-bit value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Next uniform `f64` in `[0, 1)` with 53 bits of precision.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_across_instances() {
        let mut a = Prng::new(0xDEAD_BEEF);
        let mut b = Prng::new(0xDEAD_BEEF);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = Prng::new(1);
        let mut b = Prng::new(2);
        let same = (0..64).filter(|_| a.next_u64() == b.next_u64()).count();
        assert!(same < 4, "seeds 1 and 2 should produce distinct streams");
    }

    #[test]
    fn f64_in_unit_interval() {
        let mut rng = Prng::new(7);
        let mut sum = 0.0;
        for _ in 0..10_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
            sum += x;
        }
        let mean = sum / 10_000.0;
        assert!((mean - 0.5).abs() < 0.02, "mean {mean} far from 0.5");
    }

    #[test]
    fn zero_seed_works() {
        let mut rng = Prng::new(0);
        // Must not get stuck at zero.
        assert_ne!(rng.next_u64() | rng.next_u64() | rng.next_u64(), 0);
    }
}
