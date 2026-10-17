//! Deterministic data parallelism for the substep's independent per-item work.
//!
//! Bond responses and chunk loads within a substep depend only on the state at the
//! start of the substep, so they are computed in parallel into a list in item order and
//! then applied sequentially in that order. Results (and every floating-point sum taken
//! from them) are therefore bit-identical for any thread count. The worker pool is
//! rayon's (`RAYON_NUM_THREADS` sets its size); with a single worker every loop runs on
//! the calling thread (handing work to the pool and waiting for it only costs time).

use rayon::prelude::*;

/// Below this many items the loop runs on the calling thread: waking the pool costs more
/// than the work. Measured on 4 cores, every benchmark scene (up to ~2000 bonds) ran as
/// fast or faster serially, so only larger structures go to the pool.
const MIN_PARALLEL_ITEMS: usize = 4096;

/// Whether the pool has more than one worker.
fn pooled() -> bool {
    rayon::current_num_threads() > 1
}

/// Whether `map` over `n` items runs on the calling thread.
pub fn serial(n: usize) -> bool {
    n < MIN_PARALLEL_ITEMS || !pooled()
}

/// `items.iter().map(f).collect()`, split over the worker pool, in item order.
pub fn map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    if serial(items.len()) {
        return items.iter().map(f).collect();
    }
    items.par_iter().with_min_len(MIN_PARALLEL_ITEMS / 4).map(f).collect()
}

/// `items.into_iter().map(f).collect()`, split over the worker pool, in item order.
pub fn map_into<T: Send, R: Send>(items: Vec<T>, f: impl Fn(T) -> R + Sync + Send) -> Vec<R> {
    if items.len() < MIN_PARALLEL_ITEMS || !pooled() {
        return items.into_iter().map(f).collect();
    }
    items.into_par_iter().with_min_len(MIN_PARALLEL_ITEMS / 4).map(f).collect()
}

/// Below this many items a loop of expensive items (tens of microseconds each, such as
/// the polytope contacts of chunk pairs) runs on the calling thread. Measured on 16
/// cores (layer contact, all threads): a frame collapse with ~20 pairs per substep spent
/// 11 s of system time waking the pool at 16 (2.6 s of work, the same wall time
/// serially); a breaching panel with ~600 pairs per substep runs 5x faster in the pool.
const MIN_PARALLEL_HEAVY: usize = 64;

/// `map_into` for expensive items: split over the pool from a few dozen items on, in
/// item order (bit-identical for any thread count).
pub fn map_into_heavy<T: Send, R: Send>(items: Vec<T>, f: impl Fn(T) -> R + Sync + Send) -> Vec<R> {
    if items.len() < MIN_PARALLEL_HEAVY || !pooled() {
        return items.into_iter().map(f).collect();
    }
    items.into_par_iter().with_min_len(2).map(f).collect()
}

/// `range.map(f).collect()`, split over the pool in runs of at least `min_len` items, in
/// order (bit-identical for any thread count); serial below two runs.
pub fn map_range<R: Send>(range: std::ops::Range<usize>, min_len: usize, f: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    if range.len() < 2 * min_len || !pooled() {
        return range.map(f).collect();
    }
    range.into_par_iter().with_min_len(min_len).map(f).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn map_keeps_item_order() {
        let items: Vec<u64> = (0..10_000).collect();
        let out = super::map(&items, |x| x * 3 + 1);
        assert!(out.iter().enumerate().all(|(i, &v)| v == 3 * i as u64 + 1));
    }
}
