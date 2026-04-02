//! `rubble-gpu` — thin wgpu compute abstraction for the rubble physics engine.

mod buffer;
mod context;
mod kernel;
mod multi_gpu;

pub use buffer::{GpuAtomicCounter, GpuBuffer, PingPongBuffer};
#[cfg(test)]
pub use context::test_gpu;
pub use context::GpuContext;
pub use kernel::{round_up_workgroups, ComputeKernel};
pub use multi_gpu::{GpuDevice, GpuDevicePool, MultiGpuBuffer, MultiGpuContext, WorkDistribution};

/// Errors that can occur when initialising GPU resources.
#[derive(thiserror::Error, Debug)]
pub enum GpuError {
    #[error("No GPU adapter found. Install mesa-vulkan-drivers for lavapipe software Vulkan.")]
    NoAdapter,
    #[error("Failed to request device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),
}
