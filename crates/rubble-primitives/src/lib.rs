pub fn radix_sort_u64_keyval(data: &mut [u64]) {
    data.sort_unstable_by_key(|kv| (kv >> 32) as u32);
}

pub fn exclusive_prefix_scan(values: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(values.len());
    let mut acc = 0;
    for &v in values {
        out.push(acc);
        acc += v;
    }
    out
}

pub fn stream_compact<T: Copy>(data: &[T], predicate: &[u32]) -> Vec<T> {
    assert_eq!(data.len(), predicate.len());
    data.iter()
        .zip(predicate.iter())
        .filter_map(|(d, p)| (*p != 0).then_some(*d))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix_sort_orders_by_upper_32_bits() {
        let mut pairs = vec![
            (5u64 << 32) | 1,
            (2u64 << 32) | 9,
            (2u64 << 32) | 3,
            (9u64 << 32) | 1,
        ];
        radix_sort_u64_keyval(&mut pairs);
        let keys: Vec<u32> = pairs.iter().map(|v| (v >> 32) as u32).collect();
        assert_eq!(keys, vec![2, 2, 5, 9]);
    }

    #[test]
    fn scan_ones_is_index_sequence() {
        let input = vec![1u32; 1024];
        let out = exclusive_prefix_scan(&input);
        assert_eq!(out.first(), Some(&0));
        assert_eq!(out.last(), Some(&1023));
    }

    #[test]
    fn compaction_preserves_relative_order() {
        let data = vec![10, 20, 30, 40, 50];
        let pred = vec![0, 1, 0, 1, 1];
        assert_eq!(stream_compact(&data, &pred), vec![20, 40, 50]);
    }
}
