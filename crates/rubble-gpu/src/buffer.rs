use crate::GpuContext;
use std::marker::PhantomData;

/// A GPU storage buffer holding elements of type `T`.
pub struct GpuBuffer<T: bytemuck::Pod> {
    buffer: wgpu::Buffer,
    capacity: u32,
    len: u32,
    usage: wgpu::BufferUsages,
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> GpuBuffer<T> {
    /// Create a new GPU buffer with the given element capacity.
    pub fn new(ctx: &GpuContext, capacity: usize) -> Self {
        Self::new_with_usage(ctx, capacity, wgpu::BufferUsages::empty())
    }

    pub fn new_with_usage(
        ctx: &GpuContext,
        capacity: usize,
        extra_usage: wgpu::BufferUsages,
    ) -> Self {
        let byte_size = (capacity * std::mem::size_of::<T>()) as u64;
        // wgpu requires non-zero buffer size
        let byte_size = byte_size.max(4);
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST
            | extra_usage;
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuBuffer"),
            size: byte_size,
            usage,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            capacity: capacity as u32,
            len: 0,
            usage,
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

        let staging = ctx.acquire_staging_buffer(byte_len, "GpuBuffer staging");

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, byte_len);
        ctx.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..byte_len);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let mapped = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        ctx.release_staging_buffer(staging);
        result
    }

    /// Download this buffer and another buffer in a single GPU submission and wait.
    pub fn download_with<U>(&self, ctx: &GpuContext, other: &GpuBuffer<U>) -> (Vec<T>, Vec<U>)
    where
        T: bytemuck::Zeroable,
        U: bytemuck::Pod + bytemuck::Zeroable,
    {
        if self.len == 0 && other.len == 0 {
            return (Vec::new(), Vec::new());
        }

        let byte_len_self = (self.len as usize * std::mem::size_of::<T>()) as u64;
        let byte_len_other = (other.len as usize * std::mem::size_of::<U>()) as u64;

        let staging_self =
            ctx.acquire_staging_buffer(byte_len_self.max(4), "GpuBuffer staging pair A");
        let staging_other =
            ctx.acquire_staging_buffer(byte_len_other.max(4), "GpuBuffer staging pair B");

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        if byte_len_self > 0 {
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_self, 0, byte_len_self);
        }
        if byte_len_other > 0 {
            encoder.copy_buffer_to_buffer(&other.buffer, 0, &staging_other, 0, byte_len_other);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let slice_self = staging_self.slice(..byte_len_self.max(4));
        let slice_other = staging_other.slice(..byte_len_other.max(4));
        slice_self.map_async(wgpu::MapMode::Read, |_| {});
        slice_other.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let result_self = if byte_len_self == 0 {
            Vec::new()
        } else {
            let mapped = slice_self.get_mapped_range();
            let result = bytemuck::cast_slice(&mapped[..byte_len_self as usize]).to_vec();
            drop(mapped);
            result
        };
        let result_other = if byte_len_other == 0 {
            Vec::new()
        } else {
            let mapped = slice_other.get_mapped_range();
            let result = bytemuck::cast_slice(&mapped[..byte_len_other as usize]).to_vec();
            drop(mapped);
            result
        };

        staging_self.unmap();
        staging_other.unmap();
        ctx.release_staging_buffer(staging_self);
        ctx.release_staging_buffer(staging_other);
        (result_self, result_other)
    }

    /// Download this buffer and an atomic counter in a single GPU submission and wait.
    pub fn download_with_counter(
        &self,
        ctx: &GpuContext,
        counter: &GpuAtomicCounter,
        element_count: u32,
    ) -> (Vec<T>, u32)
    where
        T: bytemuck::Zeroable,
    {
        if element_count == 0 {
            return (Vec::new(), counter.read(ctx));
        }

        let byte_len_self = (element_count as usize * std::mem::size_of::<T>()) as u64;
        let staging_self =
            ctx.acquire_staging_buffer(byte_len_self.max(4), "GpuBuffer staging with counter");
        let staging_counter = ctx.acquire_staging_buffer(4, "GpuAtomicCounter staging with buffer");

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_self, 0, byte_len_self);
        encoder.copy_buffer_to_buffer(&counter.buffer, 0, &staging_counter, 0, 4);
        ctx.queue.submit(Some(encoder.finish()));

        let slice_self = staging_self.slice(..byte_len_self.max(4));
        let slice_counter = staging_counter.slice(..4);
        slice_self.map_async(wgpu::MapMode::Read, |_| {});
        slice_counter.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let mapped_self = slice_self.get_mapped_range();
        let result_self = bytemuck::cast_slice(&mapped_self[..byte_len_self as usize]).to_vec();
        drop(mapped_self);

        let mapped_counter = slice_counter.get_mapped_range();
        let count = *bytemuck::from_bytes::<u32>(&mapped_counter[..4]);
        drop(mapped_counter);

        staging_self.unmap();
        staging_counter.unmap();
        ctx.release_staging_buffer(staging_self);
        ctx.release_staging_buffer(staging_counter);
        (result_self, count)
    }

    /// Download the buffer contents asynchronously.
    /// On WASM, yields to the JS event loop while waiting for the GPU.
    /// On native, blocks via `device.poll(wait_indefinitely)`.
    pub async fn download_async(&self, ctx: &GpuContext) -> Vec<T> {
        if self.len == 0 {
            return Vec::new();
        }

        let byte_len = (self.len as usize * std::mem::size_of::<T>()) as u64;

        let staging = ctx.acquire_staging_buffer(byte_len, "GpuBuffer staging (async)");

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, byte_len);
        ctx.queue.submit(Some(encoder.finish()));

        Self::map_staging_async(ctx, staging, byte_len).await
    }

    /// Record a copy from this buffer to a staging buffer on the given encoder.
    /// Returns the staging buffer to be mapped later with `map_staging_async`.
    /// This avoids a separate queue.submit() for the copy — the caller batches it
    /// into an existing encoder.
    pub fn record_copy_to_staging(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Option<(wgpu::Buffer, u64)> {
        if self.len == 0 {
            return None;
        }
        let byte_len = (self.len as usize * std::mem::size_of::<T>()) as u64;
        let staging = ctx.acquire_staging_buffer(byte_len, "GpuBuffer staging (batched)");
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, byte_len);
        Some((staging, byte_len))
    }

    /// Await an already-submitted staging buffer copy. Call this after the
    /// encoder containing the copy has been submitted.
    /// On WASM, yields to the JS event loop while waiting; on native, blocks.
    pub async fn map_staging_async(ctx: &GpuContext, staging: wgpu::Buffer, byte_len: u64) -> Vec<T> {
        let slice = staging.slice(..byte_len);

        #[cfg(target_arch = "wasm32")]
        {
            let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let done_clone = done.clone();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                result.unwrap();
                done_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            });
            while !done.load(std::sync::atomic::Ordering::SeqCst) {
                crate::yield_now().await;
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        }

        let mapped = slice.get_mapped_range();
        let elem_count = mapped.len() / std::mem::size_of::<T>();
        let mut result = vec![T::zeroed(); elem_count];
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut result);
        dst_bytes.copy_from_slice(&mapped[..dst_bytes.len()]);
        drop(mapped);
        staging.unmap();
        ctx.release_staging_buffer(staging);
        result
    }

    /// Async variant of [`GpuBuffer::download_with`].
    pub async fn download_with_async<U>(
        &self,
        ctx: &GpuContext,
        other: &GpuBuffer<U>,
    ) -> (Vec<T>, Vec<U>)
    where
        T: bytemuck::Zeroable,
        U: bytemuck::Pod + bytemuck::Zeroable,
    {
        if self.len == 0 && other.len == 0 {
            return (Vec::new(), Vec::new());
        }

        let byte_len_self = (self.len as usize * std::mem::size_of::<T>()) as u64;
        let byte_len_other = (other.len as usize * std::mem::size_of::<U>()) as u64;

        let staging_self =
            ctx.acquire_staging_buffer(byte_len_self.max(4), "GpuBuffer staging async pair A");
        let staging_other =
            ctx.acquire_staging_buffer(byte_len_other.max(4), "GpuBuffer staging async pair B");

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        if byte_len_self > 0 {
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_self, 0, byte_len_self);
        }
        if byte_len_other > 0 {
            encoder.copy_buffer_to_buffer(&other.buffer, 0, &staging_other, 0, byte_len_other);
        }
        ctx.queue.submit(Some(encoder.finish()));

        let slice_self = staging_self.slice(..byte_len_self.max(4));
        let slice_other = staging_other.slice(..byte_len_other.max(4));

        #[cfg(target_arch = "wasm32")]
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            use std::sync::Arc;
            let done_self = Arc::new(AtomicBool::new(false));
            let done_other = Arc::new(AtomicBool::new(false));
            {
                let done = done_self.clone();
                slice_self.map_async(wgpu::MapMode::Read, move |result| {
                    result.unwrap();
                    done.store(true, Ordering::SeqCst);
                });
            }
            {
                let done = done_other.clone();
                slice_other.map_async(wgpu::MapMode::Read, move |result| {
                    result.unwrap();
                    done.store(true, Ordering::SeqCst);
                });
            }
            while !done_self.load(Ordering::SeqCst) || !done_other.load(Ordering::SeqCst) {
                crate::yield_now().await;
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            slice_self.map_async(wgpu::MapMode::Read, |_| {});
            slice_other.map_async(wgpu::MapMode::Read, |_| {});
            let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        }

        let result_self = if byte_len_self == 0 {
            Vec::new()
        } else {
            let mapped = slice_self.get_mapped_range();
            let elem_count = byte_len_self as usize / std::mem::size_of::<T>();
            let mut result = vec![T::zeroed(); elem_count];
            let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut result);
            dst_bytes.copy_from_slice(&mapped[..dst_bytes.len()]);
            drop(mapped);
            result
        };
        let result_other = if byte_len_other == 0 {
            Vec::new()
        } else {
            let mapped = slice_other.get_mapped_range();
            let elem_count = byte_len_other as usize / std::mem::size_of::<U>();
            let mut result = vec![U::zeroed(); elem_count];
            let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut result);
            dst_bytes.copy_from_slice(&mapped[..dst_bytes.len()]);
            drop(mapped);
            result
        };

        staging_self.unmap();
        staging_other.unmap();
        ctx.release_staging_buffer(staging_self);
        ctx.release_staging_buffer(staging_other);
        (result_self, result_other)
    }

    /// Async variant of [`GpuBuffer::download_with_counter`].
    pub async fn download_with_counter_async(
        &self,
        ctx: &GpuContext,
        counter: &GpuAtomicCounter,
        element_count: u32,
    ) -> (Vec<T>, u32)
    where
        T: bytemuck::Zeroable,
    {
        if element_count == 0 {
            return (Vec::new(), counter.read_async(ctx).await);
        }

        let byte_len_self = (element_count as usize * std::mem::size_of::<T>()) as u64;
        let staging_self = ctx
            .acquire_staging_buffer(byte_len_self.max(4), "GpuBuffer staging async with counter");
        let staging_counter =
            ctx.acquire_staging_buffer(4, "GpuAtomicCounter staging async with buffer");

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_self, 0, byte_len_self);
        encoder.copy_buffer_to_buffer(&counter.buffer, 0, &staging_counter, 0, 4);
        ctx.queue.submit(Some(encoder.finish()));

        let slice_self = staging_self.slice(..byte_len_self.max(4));
        let slice_counter = staging_counter.slice(..4);

        #[cfg(target_arch = "wasm32")]
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            use std::sync::Arc;
            let done_self = Arc::new(AtomicBool::new(false));
            let done_counter = Arc::new(AtomicBool::new(false));
            {
                let done = done_self.clone();
                slice_self.map_async(wgpu::MapMode::Read, move |result| {
                    result.unwrap();
                    done.store(true, Ordering::SeqCst);
                });
            }
            {
                let done = done_counter.clone();
                slice_counter.map_async(wgpu::MapMode::Read, move |result| {
                    result.unwrap();
                    done.store(true, Ordering::SeqCst);
                });
            }
            while !done_self.load(Ordering::SeqCst) || !done_counter.load(Ordering::SeqCst) {
                crate::yield_now().await;
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            slice_self.map_async(wgpu::MapMode::Read, |_| {});
            slice_counter.map_async(wgpu::MapMode::Read, |_| {});
            let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        }

        let mapped_self = slice_self.get_mapped_range();
        let elem_count = byte_len_self as usize / std::mem::size_of::<T>();
        let mut result = vec![T::zeroed(); elem_count];
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut result);
        dst_bytes.copy_from_slice(&mapped_self[..dst_bytes.len()]);
        drop(mapped_self);

        let mapped_counter = slice_counter.get_mapped_range();
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&mapped_counter[..4]);
        drop(mapped_counter);

        staging_self.unmap();
        staging_counter.unmap();
        ctx.release_staging_buffer(staging_self);
        ctx.release_staging_buffer(staging_counter);
        (result, u32::from_le_bytes(bytes))
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
            usage: self.usage,
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

    pub fn current_index(&self) -> usize {
        self.current
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

    /// Set the logical length of the current buffer.
    pub fn set_len(&mut self, len: u32) {
        self.buffers[self.current].set_len(len);
    }

    /// Async download — works on all platforms.
    pub async fn download_async(&self, ctx: &GpuContext) -> Vec<T> {
        self.buffers[self.current].download_async(ctx).await
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
        let staging = ctx.acquire_staging_buffer(4, "GpuAtomicCounter staging");

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, 4);
        ctx.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..4);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let mapped = slice.get_mapped_range();
        let value = *bytemuck::from_bytes::<u32>(&mapped);
        drop(mapped);
        staging.unmap();
        ctx.release_staging_buffer(staging);
        value
    }

    pub fn read_triplet(
        ctx: &GpuContext,
        a: &GpuAtomicCounter,
        b: &GpuAtomicCounter,
        c: &GpuAtomicCounter,
    ) -> [u32; 3] {
        let staging = ctx.acquire_staging_buffer(12, "GpuAtomicCounter triplet staging");

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&a.buffer, 0, &staging, 0, 4);
        encoder.copy_buffer_to_buffer(&b.buffer, 0, &staging, 4, 4);
        encoder.copy_buffer_to_buffer(&c.buffer, 0, &staging, 8, 4);
        ctx.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..12);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        let mapped = slice.get_mapped_range();
        let values = [
            *bytemuck::from_bytes::<u32>(&mapped[0..4]),
            *bytemuck::from_bytes::<u32>(&mapped[4..8]),
            *bytemuck::from_bytes::<u32>(&mapped[8..12]),
        ];
        drop(mapped);
        staging.unmap();
        ctx.release_staging_buffer(staging);
        values
    }

    /// Read the counter value asynchronously — works on all platforms.
    pub async fn read_async(&self, ctx: &GpuContext) -> u32 {
        let staging = ctx.acquire_staging_buffer(4, "GpuAtomicCounter staging (async)");

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, 4);
        ctx.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..4);

        #[cfg(target_arch = "wasm32")]
        {
            let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let done_clone = done.clone();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                result.unwrap();
                done_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            });
            while !done.load(std::sync::atomic::Ordering::SeqCst) {
                crate::yield_now().await;
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        }

        let mapped = slice.get_mapped_range();
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&mapped[..4]);
        drop(mapped);
        staging.unmap();
        ctx.release_staging_buffer(staging);
        u32::from_le_bytes(bytes)
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
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut buf = GpuBuffer::<f32>::new(&ctx, 4);
        let data = [1.0f32, 2.0, 3.0, 4.0];
        buf.upload(&ctx, &data);
        let result = buf.download(&ctx);
        assert_eq!(result, data);
    }

    #[test]
    fn test_buffer_grow() {
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
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
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
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
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
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
        let Some(ctx) = crate::test_gpu() else {
            eprintln!("SKIP: No GPU");
            return;
        };
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
