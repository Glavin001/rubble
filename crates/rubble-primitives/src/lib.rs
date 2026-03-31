//! `rubble-primitives` -- GPU-accelerated prefix scan, radix sort, and stream compaction.

mod compaction;
mod prefix_scan;
mod radix_sort;

pub use compaction::GpuStreamCompaction;
pub use prefix_scan::GpuPrefixScan;
pub use radix_sort::{GpuRadixSort, RadixSortEntry};

// Re-export the internal PrefixScan used by other modules in this crate.
pub(crate) use prefix_scan::InternalPrefixScan;

#[cfg(test)]
fn test_gpu() -> rubble_gpu::GpuContext {
    pollster::block_on(rubble_gpu::GpuContext::new()).expect(
        "FATAL: No GPU adapter found. Install mesa-vulkan-drivers for lavapipe software Vulkan.",
    )
}
