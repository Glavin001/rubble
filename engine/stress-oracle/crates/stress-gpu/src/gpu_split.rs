//! Fracture topology on the GPU (`shaders/split.slang`, engine/docs/stress-baseline-goal.md
//! M2): the children of disconnected islands, as stress-ref `split_cluster` makes them, from
//! the GPU's tables and state. `STRESS_GPU_SPLIT_CHECK=1` compares every split of the
//! resident frame with the host's (`SplitCheck`).

use crate::gpu::{groups, Binding, Gpu, Kernel};
use crate::shaders;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SplitParams {
    parent_count: u32,
    out_base: u32,
    pad: [u32; 2],
}

const BINDINGS: [Binding; 11] = [
    Binding::Uniform,   // params
    Binding::Storage,   // chunk statics
    Binding::Storage,   // index (incidence)
    Binding::Storage,   // bond statics
    Binding::Storage,   // bond records
    Binding::Storage,   // islands
    Binding::Storage,   // state
    Binding::StorageRw, // parents
    Binding::StorageRw, // work (labels, members)
    Binding::StorageRw, // out (child records)
    Binding::Storage,   // frames (each parent's rigid frame)
];

const CHILD_WORDS: usize = 8;

/// One child of a split, as the GPU computed it.
#[derive(Clone, Debug, Default)]
pub struct GpuChild {
    /// Its chunks, as layout chunk indices, in the island's order.
    pub chunks: Vec<u32>,
    pub anchored: bool,
    pub mass: f32,
    /// Centre of mass (body frame of the structure).
    pub com: [f32; 3],
    /// Inertia rows about the centre of mass (body frame).
    pub inertia: [[f32; 3]; 3],
    pub velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
}

/// How GPU splits compared with the host's (`STRESS_GPU_SPLIT_CHECK=1`).
#[derive(Clone, Copy, Debug, Default)]
pub struct SplitCheck {
    pub splits: u64,
    pub children: u64,
    /// Splits whose children (count, membership or order) differ from the host's.
    pub topology_mismatch: u64,
    /// Children whose anchoring differs.
    pub anchored_mismatch: u64,
    /// Largest relative differences (f32 against the host's f64).
    pub mass: f64,
    pub com: f64,
    pub inertia: f64,
    pub velocity: f64,
    pub angular_velocity: f64,
}

pub struct GpuSplitter {
    components: Kernel,
    children: Kernel,
}

impl GpuSplitter {
    pub fn new(gpu: &Gpu) -> GpuSplitter {
        let k = |e: &str| gpu.compute(&shaders::SPLIT, e, &BINDINGS);
        GpuSplitter { components: k("split_components"), children: k("split_children") }
    }

    /// The children of the islands `parents` (island, chunk range, rigid frame: rotation
    /// (x, y, z, w), centre-of-mass velocity, angular velocity, centre of mass in the
    /// body frame), read back.
    /// `buffers`: chunk statics, index, bond statics, bond records, islands, state.
    pub fn split(&self, gpu: &Gpu, buffers: [&wgpu::Buffer; 6], parents: &[(usize, [u32; 2], [[f32; 4]; 4])]) -> Vec<Vec<GpuChild>> {
        let mut table: Vec<[u32; 4]> = Vec::with_capacity(parents.len() + 1);
        let mut base = 0u32;
        let frames: Vec<[f32; 4]> = parents.iter().flat_map(|p| p.2).collect();
        let frames = gpu.storage("split frames", if frames.is_empty() { &[[0f32; 4]][..] } else { &frames });
        for &(island, r, _) in parents {
            table.push([island as u32, base, 0, 0]);
            base += r[1] - r[0];
        }
        table.push([0, base, 0, 0]);
        let total = base as usize;
        let parents_buf = gpu.storage("split parents", &table);
        let work = gpu.storage("split work", &vec![0u32; 2 * total.max(1)]);
        let out = gpu.storage("split out", &vec![[0f32; 4]; CHILD_WORDS * total.max(1)]);
        let params = gpu.uniform("split params", &SplitParams { parent_count: parents.len() as u32, out_base: 0, pad: [0; 2] });
        let [chunks, index, bonds, bond_dyn, islands, state] = buffers;
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for (k, size) in [(&self.components, parents.len().max(1) as u32), (&self.children, groups(total, 64))] {
            let bind = gpu.bind(&k.group, &[&params, chunks, index, bonds, bond_dyn, islands, state, &parents_buf, &work, &out, &frames]);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(size.max(1), 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        let table: Vec<[u32; 4]> = gpu.read(&parents_buf);
        let work: Vec<u32> = gpu.read(&work);
        let out: Vec<[f32; 4]> = gpu.read(&out);
        parents
            .iter()
            .enumerate()
            .map(|(p, &(_, r, _))| {
                let base = table[p][1] as usize;
                (0..table[p][2] as usize)
                    .map(|k| {
                        let rec = &out[CHILD_WORDS * (base + k)..CHILD_WORDS * (base + k + 1)];
                        let (offset, count) = (rec[0][1].to_bits() as usize, rec[0][2].to_bits() as usize);
                        GpuChild {
                            chunks: (0..count).map(|j| r[0] + work[total + base + offset + j]).collect(),
                            anchored: rec[0][3].to_bits() != 0,
                            mass: rec[1][3],
                            com: [rec[1][0], rec[1][1], rec[1][2]],
                            inertia: [[rec[2][0], rec[2][1], rec[2][2]], [rec[3][0], rec[3][1], rec[3][2]], [rec[4][0], rec[4][1], rec[4][2]]],
                            velocity: [rec[5][0], rec[5][1], rec[5][2]],
                            angular_velocity: [rec[6][0], rec[6][1], rec[6][2]],
                        }
                    })
                    .collect()
            })
            .collect()
    }
}
