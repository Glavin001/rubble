#[cfg(feature = "rust-gpu")]
pub mod rust_gpu_kernels;

use bytemuck::Pod;
use std::marker::PhantomData;

pub struct GpuContext {
    pub device: Option<wgpu::Device>,
    pub queue: Option<wgpu::Queue>,
    pub cpu_fallback: bool,
}

impl GpuContext {
    pub async fn new_with_backends(backends: wgpu::Backends) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await;

        match adapter {
            Ok(adapter) => {
                let (device, queue) = adapter
                    .request_device(&wgpu::DeviceDescriptor {
                        label: Some("rubble-device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::Performance,
                        trace: wgpu::Trace::default(),
                    })
                    .await
                    .expect("failed to create device");
                Self {
                    device: Some(device),
                    queue: Some(queue),
                    cpu_fallback: false,
                }
            }
            Err(_) => Self {
                device: None,
                queue: None,
                cpu_fallback: true,
            },
        }
    }
}

pub fn test_gpu_context() -> GpuContext {
    pollster::block_on(GpuContext::new_with_backends(
        wgpu::Backends::NOOP | wgpu::Backends::all(),
    ))
}

pub struct GpuBuffer<T: Pod + Copy> {
    buffer: Option<wgpu::Buffer>,
    cpu_data: Vec<T>,
    len: u32,
    capacity: u32,
    usage: wgpu::BufferUsages,
    _marker: PhantomData<T>,
}

impl<T: Pod + Copy> GpuBuffer<T> {
    pub fn new(ctx: &GpuContext, initial_capacity: usize) -> Self {
        let cap = initial_capacity.max(1) as u32;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buffer = ctx.device.as_ref().map(|device| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rubble-gpu-buffer"),
                size: (cap as usize * std::mem::size_of::<T>()) as u64,
                usage,
                mapped_at_creation: false,
            })
        });
        Self {
            buffer,
            cpu_data: Vec::with_capacity(cap as usize),
            len: 0,
            capacity: cap,
            usage,
            _marker: PhantomData,
        }
    }

    pub fn upload(&mut self, ctx: &GpuContext, data: &[T]) {
        self.grow_if_needed(ctx, data.len());
        if let (Some(queue), Some(buffer)) = (&ctx.queue, &self.buffer) {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
        }
        self.cpu_data.clear();
        self.cpu_data.extend_from_slice(data);
        self.len = data.len() as u32;
    }

    pub fn download(&self, ctx: &GpuContext) -> Vec<T> {
        if ctx.cpu_fallback || self.buffer.is_none() {
            return self.cpu_data.clone();
        }
        let device = ctx.device.as_ref().expect("device missing");
        let queue = ctx.queue.as_ref().expect("queue missing");
        let buffer = self.buffer.as_ref().expect("buffer missing");
        let byte_len = self.len as u64 * std::mem::size_of::<T>() as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rubble-staging"),
            size: byte_len.max(1),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        if byte_len > 0 {
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_len);
        }
        queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..byte_len);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).ok();
        });
        let _ = device.poll(wgpu::PollType::Wait);
        rx.recv()
            .expect("map callback dropped")
            .expect("map failed");
        let view = slice.get_mapped_range();
        let out = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();
        out
    }

    pub fn len(&self) -> u32 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn grow_if_needed(&mut self, ctx: &GpuContext, required: usize) {
        if required <= self.capacity as usize {
            return;
        }
        let mut new_cap = self.capacity.max(1) as usize;
        while new_cap < required {
            new_cap *= 2;
        }
        if let Some(device) = &ctx.device {
            let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rubble-gpu-buffer-grow"),
                size: (new_cap * std::mem::size_of::<T>()) as u64,
                usage: self.usage,
                mapped_at_creation: false,
            });
            if self.len > 0 {
                if let (Some(old), Some(queue)) = (&self.buffer, &ctx.queue) {
                    let bytes = self.len as u64 * std::mem::size_of::<T>() as u64;
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    encoder.copy_buffer_to_buffer(old, 0, &new_buffer, 0, bytes);
                    queue.submit(std::iter::once(encoder.finish()));
                }
            }
            self.buffer = Some(new_buffer);
        }
        self.cpu_data.reserve(new_cap - self.capacity as usize);
        self.capacity = new_cap as u32;
    }
}

pub struct PingPongBuffer<T: Pod + Copy> {
    current: GpuBuffer<T>,
    next: GpuBuffer<T>,
}

impl<T: Pod + Copy> PingPongBuffer<T> {
    pub fn new(ctx: &GpuContext, initial_capacity: usize) -> Self {
        Self {
            current: GpuBuffer::new(ctx, initial_capacity),
            next: GpuBuffer::new(ctx, initial_capacity),
        }
    }
    pub fn current(&self) -> &GpuBuffer<T> {
        &self.current
    }
    pub fn current_mut(&mut self) -> &mut GpuBuffer<T> {
        &mut self.current
    }
    pub fn next(&self) -> &GpuBuffer<T> {
        &self.next
    }
    pub fn next_mut(&mut self) -> &mut GpuBuffer<T> {
        &mut self.next
    }
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }
}

pub fn round_up_workgroups(total: u32, workgroup_size: u32) -> u32 {
    total.div_ceil(workgroup_size)
}

pub fn run_mul2_kernel(ctx: &GpuContext, input: &[f32]) -> Vec<f32> {
    if ctx.cpu_fallback {
        let mut out = input.to_vec();
        for v in &mut out {
            *v *= 2.0;
        }
        return out;
    }

    let mut buffer = GpuBuffer::<f32>::new(ctx, input.len());
    buffer.upload(ctx, input);

    let device = ctx.device.as_ref().expect("device missing");
    let queue = ctx.queue.as_ref().expect("queue missing");
    let raw = buffer.buffer.as_ref().expect("buffer missing");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mul2"),
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage,read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
 let i = gid.x;
 if (i < arrayLength(&data)) { data[i] = data[i] * 2.0; }
}"
            .into(),
        ),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mul2-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mul2-pl"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mul2-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mul2-bg"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: raw.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(round_up_workgroups(input.len() as u32, 64), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    buffer.download(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_doubles_values() {
        let ctx = test_gpu_context();
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let out = run_mul2_kernel(&ctx, &data);
        for (i, o) in out.iter().enumerate() {
            assert_eq!(*o, data[i] * 2.0);
        }
    }

    #[test]
    fn buffer_growth_preserves_data() {
        let ctx = test_gpu_context();
        let mut buf = GpuBuffer::<u32>::new(&ctx, 2);
        buf.upload(&ctx, &[1, 2]);
        buf.grow_if_needed(&ctx, 16);
        assert!(buf.capacity() >= 16);
        assert_eq!(buf.download(&ctx), vec![1, 2]);
    }
}
