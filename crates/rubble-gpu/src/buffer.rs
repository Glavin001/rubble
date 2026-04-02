use crate::GpuContext;
use std::marker::PhantomData;

/// A GPU storage buffer holding elements of type `T`.
pub struct GpuBuffer<T: bytemuck::Pod> {
    buffer: wgpu::Buffer,
    capacity: u32,
    len: u32,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> GpuBuffer<T> {
    /// Create a new GPU buffer with the given element capacity.
    pub fn new(ctx: &GpuContext, capacity: usize) -> Self {
        let byte_size = (capacity * std::mem::size_of::<T>()) as u64;
        // wgpu requires non-zero buffer size
        let byte_size = byte_size.max(4);
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuBuffer"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            capacity: capacity as u32,
            len: 0,
            _marker: PhantomData,
        }
    }

    /// Upload data to the buffer, updating the length.
    pub fn upload(&mut self, ctx: &GpuContext, data: &[T]) {
        self.grow_if_needed(ctx, data.len());
        self.len = data.len() as u32;
        ctx.queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }

    /// Download the buffer contents (up to `len` elements) from the GPU.
    pub fn download(&self, ctx: &GpuContext) -> Vec<T> {
        if self.len == 0 {
            return Vec::new();
        }

        let byte_len = (self.len as usize * std::mem::size_of::<T>()) as u64;

        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuBuffer staging"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, byte_len);
        ctx.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let mapped = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        result
    }

    /// Set the logical length (for buffers populated by GPU compute shaders).
    pub fn set_len(&mut self, len: u32) {
        self.len = len;
    }

    /// Number of elements currently stored.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Element capacity of the buffer.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Whether the buffer has zero stored elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Grow the buffer if `required` exceeds current capacity.
    /// New capacity is `2 * required`. Existing data is copied.
    pub fn grow_if_needed(&mut self, ctx: &GpuContext, required: usize) {
        if required <= self.capacity as usize {
            return;
        }
        let new_capacity = required * 2;
        let new_byte_size = (new_capacity * std::mem::size_of::<T>()) as u64;
        let new_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuBuffer (grown)"),
            size: new_byte_size.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy old data if any
        if self.len > 0 {
            let copy_bytes = (self.len as usize * std::mem::size_of::<T>()) as u64;
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &new_buffer, 0, copy_bytes);
            ctx.queue.submit(Some(encoder.finish()));
        }

        self.buffer = new_buffer;
        self.capacity = new_capacity as u32;
    }

    /// Reference to the underlying wgpu buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Size in bytes of the allocated buffer.
    pub fn byte_size(&self) -> u64 {
        self.buffer.size()
    }
}

/// Double-buffered GPU storage for ping-pong patterns.
pub struct PingPongBuffer<T: bytemuck::Pod> {
    buffers: [GpuBuffer<T>; 2],
    current: usize,
}

impl<T: bytemuck::Pod> PingPongBuffer<T> {
    pub fn new(ctx: &GpuContext, capacity: usize) -> Self {
        Self {
            buffers: [GpuBuffer::new(ctx, capacity), GpuBuffer::new(ctx, capacity)],
            current: 0,
        }
    }

    pub fn current(&self) -> &GpuBuffer<T> {
        &self.buffers[self.current]
    }

    pub fn current_mut(&mut self) -> &mut GpuBuffer<T> {
        &mut self.buffers[self.current]
    }

    pub fn next(&self) -> &GpuBuffer<T> {
        &self.buffers[1 - self.current]
    }

    pub fn next_mut(&mut self) -> &mut GpuBuffer<T> {
        &mut self.buffers[1 - self.current]
    }

    pub fn swap(&mut self) {
        self.current = 1 - self.current;
    }

    pub fn upload(&mut self, ctx: &GpuContext, data: &[T]) {
        self.buffers[self.current].upload(ctx, data);
    }

    pub fn download(&self, ctx: &GpuContext) -> Vec<T> {
        self.buffers[self.current].download(ctx)
    }
}

/// A single `atomic<u32>` counter on the GPU.
pub struct GpuAtomicCounter {
    buffer: wgpu::Buffer,
}

impl GpuAtomicCounter {
    pub fn new(ctx: &GpuContext) -> Self {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuAtomicCounter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer }
    }

    /// Reset the counter to zero.
    pub fn reset(&self, ctx: &GpuContext) {
        ctx.queue
            .write_buffer(&self.buffer, 0, bytemuck::bytes_of(&0u32));
    }

    /// Write a specific value to the counter.
    pub fn write(&self, ctx: &GpuContext, value: u32) {
        ctx.queue
            .write_buffer(&self.buffer, 0, bytemuck::bytes_of(&value));
    }

    /// Read the counter value back from the GPU.
    pub fn read(&self, ctx: &GpuContext) -> u32 {
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuAtomicCounter staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, 4);
        ctx.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let mapped = slice.get_mapped_range();
        let value = *bytemuck::from_bytes::<u32>(&mapped);
        drop(mapped);
        staging.unmap();
        value
    }

    /// Reference to the underlying wgpu buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_upload_download() {
        let Some(ctx) = crate::test_gpu() else { eprintln!("SKIP: No GPU"); return; };
        let mut buf = GpuBuffer::<f32>::new(&ctx, 4);
        let data = [1.0f32, 2.0, 3.0, 4.0];
        buf.upload(&ctx, &data);
        let result = buf.download(&ctx);
        assert_eq!(result, data);
    }

    #[test]
    fn test_buffer_grow() {
        let Some(ctx) = crate::test_gpu() else { eprintln!("SKIP: No GPU"); return; };
        let mut buf = GpuBuffer::<f32>::new(&ctx, 4);
        let data = [1.0f32, 2.0, 3.0, 4.0];
        buf.upload(&ctx, &data);

        buf.grow_if_needed(&ctx, 100);
        assert!(buf.capacity() >= 100);
        // Old data should be preserved
        let result = buf.download(&ctx);
        assert_eq!(result, data);
    }

    #[test]
    fn test_ping_pong() {
        let Some(ctx) = crate::test_gpu() else { eprintln!("SKIP: No GPU"); return; };
        let mut pp = PingPongBuffer::<f32>::new(&ctx, 4);
        let data = [10.0f32, 20.0, 30.0, 40.0];
        pp.current_mut().upload(&ctx, &data);
        pp.swap();
        // After swap, old current is now "next"
        let result = pp.next().download(&ctx);
        assert_eq!(result, data);
    }

    #[test]
    fn test_ping_pong_upload_swap() {
        let Some(ctx) = crate::test_gpu() else { eprintln!("SKIP: No GPU"); return; };
        let mut pp = PingPongBuffer::<u32>::new(&ctx, 4);

        // Upload data to the current buffer
        let data = [1u32, 2, 3, 4];
        pp.upload(&ctx, &data);

        // Before swap: current has the data
        assert_eq!(pp.download(&ctx), data);

        // Swap buffers
        pp.swap();

        // After swap: current is the other (empty) buffer, next has the data
        assert!(pp.current().is_empty());
        let from_next = pp.next().download(&ctx);
        assert_eq!(from_next, data);

        // Upload different data to the new current
        let data2 = [5u32, 6, 7, 8];
        pp.upload(&ctx, &data2);
        assert_eq!(pp.download(&ctx), data2);

        // The two buffers hold different data
        assert_ne!(pp.current().download(&ctx), pp.next().download(&ctx));
    }

    #[test]
    fn test_atomic_counter() {
        let Some(ctx) = crate::test_gpu() else { eprintln!("SKIP: No GPU"); return; };
        let counter = GpuAtomicCounter::new(&ctx);
        counter.reset(&ctx);

        // Run a simple compute shader that increments the counter
        let wgsl = r#"
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    atomicAdd(&counter, 1u);
}
"#;
        let kernel = crate::ComputeKernel::from_wgsl(&ctx, wgsl, "main");

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: kernel.bind_group_layout(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: counter.buffer().as_entire_binding(),
            }],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(kernel.pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1); // 1 workgroup of 64 threads
        }
        ctx.queue.submit(Some(encoder.finish()));

        let value = counter.read(&ctx);
        assert_eq!(value, 64);
    }
}
