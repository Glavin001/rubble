//! The rigid coupling's records in an [`ExplicitBeamsGpu`] built
//! [`ExplicitBeamsGpu::with_coupling`] (`slang/kernels/explicit_beams_rigid.slang`):
//! a CPU [`ProxySet`] written as a vehicle's proxies for the next frames, and read back.
//! The World fills the same records on the GPU (its coupling plan); these are for tests
//! and hosts that plan on the CPU.

use crate::explicit_beams_gpu::ExplicitBeamsGpu;
use engine_softbody::explicit_rigid::ProxySet;

pub use engine_exec::explicit_plan::coupling::{proxy_records, CPL_BOX, CPL_CAPSULE, CPL_HULL, CPL_SPHERE};
use engine_exec::explicit_plan::coupling::decode_proxies;

/// Write vehicle `v`'s proxies (its accumulators cleared) for the next frames.
pub fn write_proxies(gpu: &ExplicitBeamsGpu, queue: &wgpu::Queue, v: usize, set: &ProxySet) -> Result<(), String> {
    let layout = gpu.coupling().ok_or("explicit coupling: built without coupling room")?;
    let recs = proxy_records(&layout, set)?;
    queue.write_buffer(
        gpu.misc_buffer(),
        u64::from(layout.vehicle(v as u32)) * 16,
        bytemuck::cast_slice(&recs),
    );
    let zeros = vec![0i32; layout.acc_per_vehicle() as usize];
    queue.write_buffer(gpu.acc_buffer(), u64::from(layout.acc(v as u32)) * 4, bytemuck::cast_slice(&zeros));
    Ok(())
}

/// Read vehicle `v`'s proxies back into `set` (blocking): the bodies' state, each shape's
/// window impulse, moment and peak.
pub fn read_proxies(gpu: &ExplicitBeamsGpu, device: &wgpu::Device, queue: &wgpu::Queue, v: usize, set: &mut ProxySet) {
    let layout = gpu.coupling().expect("coupling room");
    let n = layout.recs() as u64;
    let words: Vec<u32> = read_words(device, queue, gpu.misc_buffer(), u64::from(layout.vehicle(v as u32)) * 16, n * 16);
    decode_proxies(&layout, &words, set);
}

/// `size` bytes of `buffer` from `offset` (blocking).
pub fn read_words(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("explicit-coupling-read"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("explicit-coupling-read") });
    enc.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
    queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map").expect("map ok");
    let out = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
    staging.unmap();
    out
}
