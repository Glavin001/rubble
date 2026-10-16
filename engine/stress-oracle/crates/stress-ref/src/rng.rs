//! Deterministic, order-independent random numbers.
//!
//! Every random draw is a pure function of `(seed, keys...)`, so a bond's strength
//! does not depend on evaluation order, thread count or which other bonds exist.

/// SplitMix64 finaliser.
pub fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Hash of a seed and a list of keys.
pub fn hash(seed: u64, keys: &[u64]) -> u64 {
    keys.iter().fold(mix64(seed), |h, k| mix64(h ^ mix64(*k)))
}

/// Uniform sample in the open interval (0, 1).
pub fn uniform(seed: u64, keys: &[u64]) -> f64 {
    ((hash(seed, keys) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_in_range_and_well_spread() {
        let n = 100_000;
        let mut sum = 0.0;
        for i in 0..n {
            let u = uniform(7, &[1, i]);
            assert!(u > 0.0 && u < 1.0);
            sum += u;
        }
        assert!((sum / n as f64 - 0.5).abs() < 0.005);
        assert_ne!(uniform(7, &[1, 2]), uniform(8, &[1, 2]));
    }
}
