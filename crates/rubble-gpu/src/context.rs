use crate::GpuError;

/// Thin wrapper around a wgpu device and queue.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Request a high-performance GPU adapter and create a device + queue.
    pub async fn new() -> Result<Self, GpuError> {
        let backends = if cfg!(target_arch = "wasm32") {
            wgpu::Backends::BROWSER_WEBGPU
        } else {
            wgpu::Backends::VULKAN
        };

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: Default::default(),
        });

        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(_) => {
                // Fallback: try software adapter (e.g. lavapipe in CI)
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::LowPower,
                        compatible_surface: None,
                        force_fallback_adapter: true,
                    })
                    .await
                    .map_err(|_| GpuError::NoAdapter)?
            }
        };

        // Clamp requested limits to what the adapter actually supports.
        // SwiftShader (used in CI/testing) may support fewer storage buffers.
        let adapter_limits = adapter.limits();
        let desired_storage_buffers: u32 = 16;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: desired_storage_buffers
                        .min(adapter_limits.max_storage_buffers_per_shader_stage),
                    ..wgpu::Limits::downlevel_defaults()
                },
                ..Default::default()
            })
            .await?;

        Ok(Self { device, queue })
    }
}

/// Create a [`GpuContext`] for tests. Panics with a clear message if no GPU
/// adapter is found.
#[cfg(test)]
pub fn test_gpu() -> GpuContext {
    pollster::block_on(GpuContext::new()).expect(
        "FATAL: No GPU adapter found. Install mesa-vulkan-drivers for lavapipe software Vulkan.",
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_gpu_context() {
        let ctx = crate::test_gpu();
        // If we got here, device and queue are valid.
        let _ = ctx.device.features();
        let _ = ctx.queue;
    }
}
