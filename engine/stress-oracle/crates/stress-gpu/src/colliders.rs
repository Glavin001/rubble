//! The coupling's collider table kept current on the GPU after a split the GPU laid out
//! (`shaders/colliders.slang`, engine/docs/stress-baseline-goal.md M4), in place of the
//! host's rebuild of it. `STRESS_GPU_COLLIDER_CHECK=1` compares it with the host's table
//! after every such split (`ColliderCheck`).

use crate::gpu::{groups, Binding, Gpu, Kernel};
use crate::ingest::{ColliderInfo, KIND_BODY, NONE};
use crate::shaders;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ColliderParams {
    parent_count: u32,
    chunk_count: u32,
    collider_count: u32,
    chunk_words: u32,
    island_words: u32,
    ground_modulus: f32,
    pad: [u32; 2],
}

const BINDINGS: [Binding; 16] = [
    Binding::Uniform,   // params
    Binding::Storage,   // parents
    Binding::Storage,   // work
    Binding::Storage,   // children
    Binding::Storage,   // slots
    Binding::Storage,   // part ranges
    Binding::Storage,   // old chunk statics
    Binding::Storage,   // chunk statics
    Binding::Storage,   // islands
    Binding::Storage,   // islands_of
    Binding::Storage,   // structure_base
    Binding::Storage,   // control
    Binding::Storage,   // chunk props
    Binding::StorageRw, // chunk slot
    Binding::StorageRw, // collider keys
    Binding::StorageRw, // collider info
];

/// The split pass's buffers the refresh reads (gpu_split.rs `GpuSplitter::pass`).
pub struct RefreshInputs<'a> {
    pub parent_count: u32,
    pub chunk_count: u32,
    pub parents: &'a wgpu::Buffer,
    pub work: &'a wgpu::Buffer,
    pub children: &'a wgpu::Buffer,
    pub slots: &'a wgpu::Buffer,
    pub part_ranges: &'a wgpu::Buffer,
    pub old_chunks: &'a wgpu::Buffer,
    pub chunks: &'a wgpu::Buffer,
    pub islands: &'a wgpu::Buffer,
    pub islands_of: &'a wgpu::Buffer,
    pub structure_base: &'a wgpu::Buffer,
    pub control: &'a wgpu::Buffer,
}

/// Per engine collider its stable chunk key (`structure_base[s] + c`, or NONE), per stable
/// chunk its layout index and the props the other side of a contact sees (modulus,
/// radius); the table itself is the ingest's (`EngineIngest::collider_info`).
pub struct ColliderRefresh {
    keys_kernel: Kernel,
    slots_kernel: Kernel,
    refresh_kernel: Kernel,
    keys: wgpu::Buffer,
    chunk_slot: wgpu::Buffer,
    props: wgpu::Buffer,
    /// Per structure: its first stable key.
    structure_base: Vec<u32>,
    capacity: u32,
    ground_modulus: f32,
}

impl ColliderRefresh {
    /// For structures whose chunks have `props[s][c]` = (Young's modulus, radius), an
    /// engine of up to `capacity` colliders, and static geometry of `ground_modulus`.
    pub fn new(gpu: &Gpu, props: &[Vec<[f32; 2]>], capacity: u32, ground_modulus: f32) -> ColliderRefresh {
        let k = |e: &str| gpu.compute(&shaders::COLLIDERS, e, &BINDINGS);
        let mut structure_base = Vec::with_capacity(props.len());
        let mut flat: Vec<[f32; 4]> = Vec::new();
        for st in props {
            structure_base.push(flat.len() as u32);
            flat.extend(st.iter().map(|p| [p[0], p[1], 0.0, 0.0]));
        }
        let stable = flat.len().max(1);
        if flat.is_empty() {
            flat.push([0.0; 4]);
        }
        ColliderRefresh {
            keys_kernel: k("collider_keys"),
            slots_kernel: k("collider_slots"),
            refresh_kernel: k("collider_refresh"),
            keys: gpu.storage("collider keys", &vec![NONE; capacity.max(1) as usize]),
            chunk_slot: gpu.storage("collider chunk slots", &vec![NONE; stable]),
            props: gpu.storage("collider chunk props", &flat),
            structure_base,
            capacity: capacity.max(1),
            ground_modulus,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The keys of the host's table `table` (built against the layout `layout_chunks`),
    /// for an engine of up to `capacity` colliders.
    pub fn set_keys(&mut self, gpu: &Gpu, table: &[ColliderInfo], layout_chunks: &[(usize, usize)], capacity: u32) {
        let capacity = capacity.max(table.len() as u32).max(1);
        let mut keys = vec![NONE; capacity as usize];
        for (key, info) in keys.iter_mut().zip(table) {
            if info.a[2] == KIND_BODY && info.a[3] != 0 && info.a[0] != NONE {
                let (s, c) = layout_chunks[info.a[0] as usize];
                *key = self.structure_base[s] + c as u32;
            }
        }
        self.keys = gpu.storage_into("collider keys", &keys, Some(&self.keys));
        self.capacity = capacity;
    }

    /// The keys of the host's entries `table` for the colliders from `first` on (the rest
    /// as they are).
    pub fn set_keys_range(&mut self, gpu: &Gpu, first: u32, table: &[ColliderInfo], layout_chunks: &[(usize, usize)]) {
        let keys: Vec<u32> = table
            .iter()
            .map(|info| {
                if info.a[2] == KIND_BODY && info.a[3] != 0 && info.a[0] != NONE {
                    let (s, c) = layout_chunks[info.a[0] as usize];
                    self.structure_base[s] + c as u32
                } else {
                    NONE
                }
            })
            .collect();
        if !keys.is_empty() {
            gpu.queue.write_buffer(&self.keys, 4 * first as u64, bytemuck::cast_slice(&keys));
        }
    }

    /// The refresh, after the split pass's layout, into `collider_info` (room for
    /// `capacity` colliders).
    pub fn encode(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, inputs: &RefreshInputs, total_slots: usize, collider_info: &wgpu::Buffer) {
        for (pipeline, group, size) in self.steps(gpu, inputs, total_slots, collider_info) {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(size, 1, 1);
        }
    }

    /// The refresh's dispatches (pipeline, bind group, groups), for a caller's own pass.
    pub fn steps(&self, gpu: &Gpu, inputs: &RefreshInputs, total_slots: usize, collider_info: &wgpu::Buffer) -> Vec<(wgpu::ComputePipeline, wgpu::BindGroup, u32)> {
        let params = gpu.uniform(
            "collider params",
            &ColliderParams {
                parent_count: inputs.parent_count,
                chunk_count: inputs.chunk_count,
                collider_count: self.capacity,
                chunk_words: 16,
                island_words: 25,
                ground_modulus: self.ground_modulus,
                pad: [0; 2],
            },
        );
        let bind = |k: &Kernel| {
            gpu.bind(
                &k.group,
                &[
                    &params,
                    inputs.parents,
                    inputs.work,
                    inputs.children,
                    inputs.slots,
                    inputs.part_ranges,
                    inputs.old_chunks,
                    inputs.chunks,
                    inputs.islands,
                    inputs.islands_of,
                    inputs.structure_base,
                    inputs.control,
                    &self.props,
                    &self.chunk_slot,
                    &self.keys,
                    collider_info,
                ],
            )
        };
        let steps = [
            (&self.keys_kernel, groups(total_slots.max(inputs.parent_count as usize), 64)),
            (&self.slots_kernel, groups(inputs.chunk_count as usize, 64)),
            (&self.refresh_kernel, groups(self.capacity as usize, 64)),
        ];
        steps.into_iter().map(|(k, size)| (k.pipeline.clone(), bind(k), size.max(1))).collect()
    }
}

/// What `STRESS_GPU_COLLIDER_CHECK=1` has compared: the GPU's table after a split it laid
/// out against the host's.
#[derive(Clone, Copy, Debug, Default)]
pub struct ColliderCheck {
    pub tables: u64,
    /// Of which tables patched in place (an engine change under the same layout).
    pub patches: u64,
    pub colliders: u64,
    /// Entries whose integer words (chunk, body code, kind, chunk side) differ.
    pub int_mismatch: u64,
    /// Largest relative difference of the other-side props (mass, modulus, radius).
    pub props: f64,
}

impl ColliderCheck {
    pub fn compare(&mut self, gpu_table: &[ColliderInfo], host: &[ColliderInfo]) {
        self.tables += 1;
        for (g, h) in gpu_table.iter().zip(host) {
            self.colliders += 1;
            if g.a != h.a {
                self.int_mismatch += 1;
            }
            for k in 0..3 {
                let (a, b) = (g.b[k] as f64, h.b[k] as f64);
                if a != b {
                    let d = if a.is_infinite() || b.is_infinite() { f64::INFINITY } else { (a - b).abs() / b.abs().max(1e-30) };
                    self.props = self.props.max(d);
                }
            }
        }
    }
}
