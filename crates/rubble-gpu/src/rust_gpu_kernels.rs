/// CPU-visible fallback representation of a rust-gpu compute kernel.
///
/// The actual rust-gpu target build is driven by a dedicated shader crate/toolchain,
/// while host tests can still validate algorithm parity through this function.
pub fn mul2_kernel_host_fallback(data: &mut [f32]) {
    for v in data {
        *v *= 2.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_fallback_matches_expected_behavior() {
        let mut data = vec![1.0, 2.0, 3.0];
        mul2_kernel_host_fallback(&mut data);
        assert_eq!(data, vec![2.0, 4.0, 6.0]);
    }
}
