use crate::GpuError;

/// Thin wrapper around a wgpu device and queue.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Enumerate all available GPU adapters and return their info.
    pub async fn enumerate_adapters() -> Vec<wgpu::AdapterInfo> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: Default::default(),
        });

        instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .await
            .into_iter()
            .map(|a| a.get_info())
            .collect()
    }

    /// Create a [`GpuContext`] from an existing adapter.
    pub async fn new_with_adapter(adapter: &wgpu::Adapter) -> Result<Self, GpuError> {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 16,
                    ..wgpu::Limits::downlevel_defaults()
                },
                ..Default::default()
            })
            .await?;

        Ok(Self { device, queue })
    }

    /// Request a high-performance GPU adapter and create a device + queue.
    pub async fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 16,
                    ..wgpu::Limits::downlevel_defaults()
                },
                ..Default::default()
            })
            .await?;

        Ok(Self { device, queue })
    }
}

/// Try to create a [`GpuContext`] for tests. Returns `None` if no GPU
/// adapter is found (e.g. in CI without Vulkan drivers).
#[cfg(test)]
pub fn test_gpu() -> Option<GpuContext> {
    pollster::block_on(GpuContext::new()).ok()
}

/// Macro to skip a test when no GPU adapter is available.
#[macro_export]
macro_rules! skip_no_gpu {
    () => {
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU adapter found");
            return;
        };
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_gpu_context() {
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU adapter found");
            return;
        };
        // If we got here, device and queue are valid.
        let _ = ctx.device.features();
        let _ = ctx.queue;
    }
}
