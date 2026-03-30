//! `rubble-primitives` -- GPU-accelerated prefix scan, radix sort, and stream compaction.

mod compaction;
mod prefix_scan;
mod radix_sort;

pub use compaction::StreamCompaction;
pub use prefix_scan::PrefixScan;
pub use radix_sort::RadixSort;

#[cfg(test)]
fn test_gpu() -> rubble_gpu::GpuContext {
    pollster::block_on(rubble_gpu::GpuContext::new()).expect(
        "FATAL: No GPU adapter found. Install mesa-vulkan-drivers for lavapipe software Vulkan.",
    )
}
