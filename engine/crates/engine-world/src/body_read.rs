//! A non-blocking read of bodies' GPU records: a copy queued now (after every step
//! submitted so far) and mapped later. Hosts and scene scripts that need velocities or a
//! fresher pose than the last polled frame start one and take it on a later frame, as the
//! web pages' `readBodies` does; nothing waits in place.

use crate::*;
use std::sync::{Arc, Mutex};

/// One body's GPU record: the solver frame (centre of mass) pose and velocities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyRecordRead {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub linear_velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub awake: bool,
}

type MapResult = Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>;

/// A read in flight ([`World::begin_body_read`]).
pub struct BodyRead {
    staging: wgpu::Buffer,
    n: usize,
    mapped: MapResult,
}

const RECORD: u64 = std::mem::size_of::<engine_schema::BodyState>() as u64;

impl World {
    /// Start reading `ids`' GPU records as the queue leaves them (every step submitted so
    /// far). Take the result with [`BodyRead::try_take`] on a later frame.
    pub fn begin_body_read(&self, ids: &[BodyId]) -> anyhow::Result<BodyRead> {
        for id in ids {
            anyhow::ensure!(
                self.body_is_valid(*id),
                "begin_body_read: {id:?} is not a live body"
            );
        }
        let device = self.device();
        let n = ids.len().max(1);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("begin_body_read"),
            size: n as u64 * RECORD,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let source = self.sim().body_buffer();
        let mut enc = device.create_command_encoder(&Default::default());
        for (k, id) in ids.iter().enumerate() {
            enc.copy_buffer_to_buffer(
                source,
                u64::from(id.index()) * RECORD,
                &staging,
                k as u64 * RECORD,
                RECORD,
            );
        }
        self.queue().submit([enc.finish()]);
        let mapped: MapResult = Arc::new(Mutex::new(None));
        let m = mapped.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            *m.lock().unwrap() = Some(r);
        });
        Ok(BodyRead {
            staging,
            n: ids.len(),
            mapped,
        })
    }
}

impl BodyRead {
    /// Whether the copy has landed ([`Self::try_take`] returns it).
    pub fn ready(&self, world: &World) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        let _ = world.device().poll(wgpu::PollType::Poll);
        #[cfg(target_arch = "wasm32")]
        let _ = world;
        self.mapped.lock().unwrap().is_some()
    }

    /// The records, in the order asked for, once the copy has landed (`None`: not yet;
    /// never waits).
    pub fn try_take(&self, world: &World) -> anyhow::Result<Option<Vec<BodyRecordRead>>> {
        if !self.ready(world) {
            return Ok(None);
        }
        if let Some(Err(e)) = self.mapped.lock().unwrap().as_ref() {
            anyhow::bail!("body read: {e}");
        }
        let mut out = Vec::with_capacity(self.n);
        {
            let data = self
                .staging
                .slice(..)
                .get_mapped_range()
                .map_err(|e| anyhow::anyhow!("body read: {e:?}"))?;
            for k in 0..self.n {
                let b: engine_schema::BodyState =
                    bytemuck::pod_read_unaligned(&data[k * RECORD as usize..][..RECORD as usize]);
                out.push(BodyRecordRead {
                    position: b.position,
                    rotation: b.orientation,
                    linear_velocity: b.linear_vel,
                    angular_velocity: b.angular_vel,
                    awake: b.flags & engine_schema::BodyFlags::Sleeping as u32 == 0,
                });
            }
        }
        self.staging.unmap();
        Ok(Some(out))
    }
}
