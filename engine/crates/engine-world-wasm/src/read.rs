//! Reads that resolve later (a GPU copy and `mapAsync`), never waiting in place.

use crate::WebWorld;
use wasm_bindgen::prelude::*;

/// Floats per body in [`WebWorld::read_bodies`].
pub const BODY_READ_STRIDE: usize = 14;

#[wasm_bindgen]
impl WebWorld {
    /// The GPU state of body slots `slots` as the queue leaves it, resolved once
    /// the copy is mapped: per slot [`BODY_READ_STRIDE`] floats, the solver-frame
    /// (centre of mass) position xyz, rotation xyzw, linear velocity xyz,
    /// angular velocity xyz and 1 if awake. For checks and occasional reads; the
    /// per-frame path is [`WebWorld::poll`].
    #[wasm_bindgen(js_name = readBodies)]
    pub fn read_bodies(&self, slots: Vec<u32>) -> js_sys::Promise {
        let device = self.world.device().clone();
        let queue = self.world.queue().clone();
        let source = self.world.sim().body_buffer().clone();
        let record = std::mem::size_of::<engine_schema::BodyState>() as u64;
        wasm_bindgen_futures::future_to_promise(async move {
            let n = slots.len().max(1) as u64;
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readBodies"),
                size: n * record,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            for (k, &s) in slots.iter().enumerate() {
                enc.copy_buffer_to_buffer(&source, u64::from(s) * record, &staging, k as u64 * record, record);
            }
            queue.submit([enc.finish()]);
            let (tx, rx) = futures_channel::oneshot::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            rx.await
                .map_err(|_| crate::err("readBodies: map dropped"))?
                .map_err(|e| crate::err(format!("readBodies: {e}")))?;
            let mut out = Vec::with_capacity(slots.len() * BODY_READ_STRIDE);
            {
                let data = staging.slice(..).get_mapped_range().expect("mapped");
                for k in 0..slots.len() {
                    let b: engine_schema::BodyState =
                        bytemuck::pod_read_unaligned(&data[k * record as usize..][..record as usize]);
                    out.extend_from_slice(&b.position);
                    out.extend_from_slice(&b.orientation);
                    out.extend_from_slice(&b.linear_vel);
                    out.extend_from_slice(&b.angular_vel);
                    let asleep = b.flags & engine_schema::BodyFlags::Sleeping as u32 != 0;
                    out.push(if asleep { 0.0 } else { 1.0 });
                }
            }
            staging.unmap();
            Ok(js_sys::Float32Array::from(out.as_slice()).into())
        })
    }
}
