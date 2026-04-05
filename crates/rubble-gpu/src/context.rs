use crate::GpuError;
use std::sync::Mutex;

/// Thin wrapper around a wgpu device and queue.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    staging_pool: Mutex<Vec<(u64, wgpu::Buffer)>>,
}

impl GpuContext {
    pub(crate) fn from_device_queue(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            staging_pool: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn acquire_staging_buffer(&self, size: u64, label: &'static str) -> wgpu::Buffer {
        let mut pool = self.staging_pool.lock().unwrap();
        if let Some(idx) = pool.iter().position(|(capacity, _)| *capacity >= size) {
            let (_, buffer) = pool.swap_remove(idx);
            buffer
        } else {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }
    }

    pub(crate) fn release_staging_buffer(&self, buffer: wgpu::Buffer) {
        let capacity = buffer.size();
        let mut pool = self.staging_pool.lock().unwrap();
        pool.push((capacity, buffer));
    }

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

        Ok(Self::from_device_queue(device, queue))
    }

    /// Request a high-performance GPU adapter and create a device + queue.
    pub async fn new() -> Result<Self, GpuError> {
        let backends = if cfg!(target_arch = "wasm32") {
            wgpu::Backends::BROWSER_WEBGPU
        } else {
            wgpu::Backends::PRIMARY
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

        Ok(Self::from_device_queue(device, queue))
    }
}

/// Try to create a [`GpuContext`] for tests. Returns `None` if no GPU
/// adapter is found (e.g. in CI without Vulkan drivers).
#[cfg(test)]
pub fn test_gpu() -> Option<GpuContext> {
    pollster::block_on(GpuContext::new()).ok()
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
