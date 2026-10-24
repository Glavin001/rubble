//! The engine's chunk colliders moved with their chunks' deformation in the resident frame
//! (`shaders/shapes.slang`), as the host-coupled frame moves them (`World::shape_poses`,
//! `World::engine_shapes_deformed`): each frame before the engine steps, from the chunk
//! state the stress solve left on the GPU, with no read. The host keeps each collider's
//! undeformed placement (`ResidentEngine::collider_rest`), uploaded only for colliders the
//! engine added or changed since.

use stress_ref::math::Vec3;

use crate::gpu::{Binding, Gpu, Kernel};
use crate::resident::ResidentEngine;
use crate::shaders;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ShapeParams {
    collider_count: u32,
    pad: [u32; 3],
}

const BINDINGS: [Binding; 5] = [
    Binding::Uniform,   // params
    Binding::Storage,   // rest
    Binding::Storage,   // collider info
    Binding::Storage,   // chunk state
    Binding::StorageRw, // engine colliders
];

/// 4 float4 per collider (shapes.slang `rest`).
type Rest = [[f32; 4]; 4];

pub struct ShapeDeform {
    kernel: Kernel,
    rest: wgpu::Buffer,
    /// Colliders the rest table has room for.
    capacity: u32,
    /// Colliders it describes.
    count: u32,
    /// The engine's `layout_version` it was made at.
    version: Option<u64>,
}

impl ShapeDeform {
    pub fn new(gpu: &Gpu) -> ShapeDeform {
        ShapeDeform { kernel: gpu.compute(&shaders::SHAPES, "shapes_deform", &BINDINGS), rest: gpu.storage("shape rest", &[[0f32; 4]; 4]), capacity: 1, count: 0, version: None }
    }

    /// The rest table current with the engine's colliders: patched where the engine says
    /// which changed, rebuilt otherwise (its capacity grown, a compaction). `center(s, c)`:
    /// chunk c of structure s's centre in its cluster frame. False when the engine gives no
    /// rest placements (its shapes stay undeformed).
    pub fn update(&mut self, gpu: &Gpu, e: &dyn ResidentEngine, center: &dyn Fn(usize, usize) -> Vec3) -> bool {
        let version = e.layout_version();
        let count = e.body_counts().1;
        if self.version == Some(version) && self.count == count {
            return true;
        }
        let records = |range: std::ops::Range<u32>| -> Option<Vec<Rest>> {
            let rest = e.collider_rest(range)?;
            Some(
                rest.iter()
                    .map(|r| match r {
                        Some(r) => {
                            let to_body = r.principal.conjugate();
                            let cb = to_body.rotate(center(r.chunk.0, r.chunk.1) - r.com);
                            let q = |q: stress_ref::math::Quat| [q.x as f32, q.y as f32, q.z as f32, q.w as f32];
                            let p = r.local_pos;
                            [[cb.x as f32, cb.y as f32, cb.z as f32, 1.0], q(to_body), [p[0], p[1], p[2], 0.0], r.local_rot]
                        }
                        None => [[0.0; 4]; 4],
                    })
                    .collect(),
            )
        };
        let entry = std::mem::size_of::<Rest>() as u64;
        let patch = match self.version {
            Some(v) if count <= self.capacity => e.collider_changes_since(v),
            _ => None,
        };
        match patch {
            Some(ranges) => {
                for range in ranges {
                    let range = range.start.min(count)..range.end.min(count);
                    if range.is_empty() {
                        continue;
                    }
                    let Some(part) = records(range.clone()) else { return false };
                    gpu.queue.write_buffer(&self.rest, entry * range.start as u64, bytemuck::cast_slice(&part));
                }
            }
            None => {
                let Some(mut all) = records(0..count) else { return false };
                // Room for the engine's collider capacity: a later split's colliders patch in.
                let capacity = e.collider_capacity().max(count).max(1);
                all.resize(capacity as usize, [[0.0; 4]; 4]);
                self.rest = gpu.storage_into("shape rest", &all, Some(&self.rest));
                self.capacity = capacity;
            }
        }
        self.version = Some(version);
        self.count = count;
        true
    }

    /// Move the colliders: `collider_info` the coupling's collider table (its chunks),
    /// `state` the solver's chunk state, `colliders` the engine's.
    pub fn encode(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, collider_info: &wgpu::Buffer, state: &wgpu::Buffer, colliders: &wgpu::Buffer) {
        let info_entries = (collider_info.size() / std::mem::size_of::<crate::ingest::ColliderInfo>() as u64) as u32;
        let n = self.count.min(self.capacity).min(info_entries);
        if n == 0 {
            return;
        }
        let params = gpu.uniform("shape params", &ShapeParams { collider_count: n, pad: [0; 3] });
        let bind = gpu.bind(&self.kernel.group, &[&params, &self.rest, collider_info, state, colliders]);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        self.kernel.dispatch(&mut pass, &bind, n as usize);
    }
}
