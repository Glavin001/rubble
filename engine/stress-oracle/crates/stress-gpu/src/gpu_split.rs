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
    offsets: Kernel,
    spawn: Kernel,
    retire: Kernel,
}

/// A split's GPU buffers (for the spawn that follows) and its children read back.
pub struct SplitRun {
    parents: wgpu::Buffer,
    work: wgpu::Buffer,
    out: wgpu::Buffer,
    frames: wgpu::Buffer,
    firsts: wgpu::Buffer,
    parent_count: usize,
    total: usize,
    pub children: Vec<Vec<GpuChild>>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SpawnParams {
    parent_count: u32,
    first_body: u32,
    first_collider: u32,
    live: u32,
    max_bodies: u32,
    pad: [u32; 3],
}

/// The engine's own body and collider buffers, for a live spawn (bodies in two regions,
/// the second at `max_bodies`).
pub struct SpawnTarget<'a> {
    pub bodies: &'a wgpu::Buffer,
    pub colliders: &'a wgpu::Buffer,
    pub max_bodies: u32,
}

const SPAWN_BINDINGS: [Binding; 15] = [
    Binding::Uniform,   // params
    Binding::Storage,   // parents
    Binding::Storage,   // work
    Binding::Storage,   // children
    Binding::Storage,   // frames
    Binding::Storage,   // part ranges
    Binding::Storage,   // parts
    Binding::Storage,   // chunk statics
    Binding::Storage,   // state
    Binding::StorageRw, // slots
    Binding::StorageRw, // bodies
    Binding::StorageRw, // colliders
    Binding::Storage,   // islands_of (each parent's first chunk, engine body, collider range)
    Binding::StorageRw, // extras (each child's centre of mass, mass, principal rotation)
    Binding::Storage,   // structure_base
];

/// The engine's per-chunk part table on the GPU (`spawn.slang`), by structure and chunk:
/// built once (a chunk's parts do not change).
pub struct PartTable {
    pub ranges: wgpu::Buffer,
    pub parts: wgpu::Buffer,
    pub structure_base: wgpu::Buffer,
    /// Each chunk's part count, by (structure, chunk) (host copy).
    pub counts: Vec<Vec<u32>>,
}

impl PartTable {
    /// From each structure's chunks' part records (`parts[s][c]`).
    pub fn new(gpu: &Gpu, parts: &[Vec<Vec<[[f32; 4]; 8]>>]) -> PartTable {
        let mut base = Vec::with_capacity(parts.len());
        let mut ranges = Vec::new();
        let mut flat: Vec<[f32; 4]> = Vec::new();
        let mut counts = Vec::with_capacity(parts.len());
        for st in parts {
            base.push(ranges.len() as u32);
            counts.push(st.iter().map(|p| p.len() as u32).collect());
            for p in st {
                ranges.push([(flat.len() / 8) as u32, p.len() as u32, 0, 0]);
                flat.extend(p.iter().flatten());
            }
        }
        let or_one = |v: &[[f32; 4]]| if v.is_empty() { vec![[0f32; 4]] } else { v.to_vec() };
        PartTable {
            ranges: gpu.storage("spawn part ranges", if ranges.is_empty() { &[[0u32; 4]][..] } else { &ranges }),
            parts: gpu.storage("spawn parts", &or_one(&flat)),
            structure_base: gpu.storage("spawn structure base", if base.is_empty() { &[0u32][..] } else { &base }),
            counts,
        }
    }
}

/// A child's rubble body and colliders as the GPU made them (`spawn.slang`): raw
/// `BodyState` and `Collider` words.
#[derive(Clone, Debug, Default)]
pub struct GpuBody {
    pub slot: u32,
    /// Its first collider slot.
    pub first_collider: u32,
    pub body: [[f32; 4]; 7],
    pub colliders: Vec<[[f32; 4]; 4]>,
    /// Centre of mass (the structure's frame), mass.
    pub com: [f32; 4],
    /// Principal rotation (x, y, z, w): the structure's frame to the body's.
    pub principal: [f32; 4],
}

impl GpuSplitter {
    pub fn new(gpu: &Gpu) -> GpuSplitter {
        let k = |e: &str| gpu.compute(&shaders::SPLIT, e, &BINDINGS);
        let s = |e: &str| gpu.compute(&shaders::SPAWN, e, &SPAWN_BINDINGS);
        GpuSplitter { components: k("split_components"), children: k("split_children"), offsets: s("spawn_offsets"), spawn: s("spawn_bodies"), retire: s("spawn_retire") }
    }

    /// The children of the islands `parents` (island, chunk range, rigid frame: rotation
    /// (x, y, z, w), centre-of-mass velocity, angular velocity, centre of mass in the
    /// body frame, pose position), read back.
    /// `buffers`: chunk statics, index, bond statics, bond records, islands, state.
    pub fn split(&self, gpu: &Gpu, buffers: [&wgpu::Buffer; 6], parents: &[(usize, [u32; 2], [[f32; 4]; 5])]) -> SplitRun {
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
        let work_buf = gpu.storage("split work", &vec![0u32; 2 * total.max(1)]);
        let out = gpu.storage("split out", &vec![[0f32; 4]; CHILD_WORDS * total.max(1)]);
        let params = gpu.uniform("split params", &SplitParams { parent_count: parents.len() as u32, out_base: 0, pad: [0; 2] });
        let [chunks, index, bonds, bond_dyn, islands, state] = buffers;
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for (k, size) in [(&self.components, parents.len().max(1) as u32), (&self.children, groups(total, 64))] {
            let bind = gpu.bind(&k.group, &[&params, chunks, index, bonds, bond_dyn, islands, state, &parents_buf, &work_buf, &out, &frames]);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(size.max(1), 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        let table: Vec<[u32; 4]> = gpu.read(&parents_buf);
        let work: Vec<u32> = gpu.read(&work_buf);
        let out_data: Vec<[f32; 4]> = gpu.read(&out);
        let out_words = &out_data;
        let children = parents
            .iter()
            .enumerate()
            .map(|(p, &(_, r, _))| {
                let base = table[p][1] as usize;
                (0..table[p][2] as usize)
                    .map(|k| {
                        let rec = &out_words[CHILD_WORDS * (base + k)..CHILD_WORDS * (base + k + 1)];
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
            .collect();
        let firsts: Vec<[u32; 4]> = parents.iter().map(|p| [p.1[0], 0, 0, 0]).collect();
        let firsts = gpu.storage("split firsts", if firsts.is_empty() { &[[0u32; 4]][..] } else { &firsts });
        SplitRun { parents: parents_buf, work: work_buf, out, frames, firsts, parent_count: parents.len(), total, children }
    }

    /// The rubble bodies and colliders of `run`'s children (`spawn.slang`), the first at
    /// body slot `first_body` and collider slot `first_collider`, from the part table,
    /// read back by child in order. With `target` they are written into the engine's own buffers, and each
    /// split parent's body (`retire`: per parent its body and collider range) is retired
    /// there; without, into a copy (a check).
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        &self,
        gpu: &Gpu,
        run: &SplitRun,
        table: &PartTable,
        chunks: &wgpu::Buffer,
        state: &wgpu::Buffer,
        first_body: u32,
        first_collider: u32,
        target: Option<SpawnTarget>,
        retire: &[[u32; 3]],
    ) -> Vec<GpuBody> {
        let total = run.total.max(1);
        // At most every part of the children's chunks once.
        let n_parts: usize = (table.parts.size() as usize / 128).max(1);
        let slots = gpu.storage("spawn slots", &vec![[0u32; 4]; total]);
        let extras = gpu.storage("spawn extras", &vec![[0f32; 4]; 2 * total]);
        let live = target.is_some();
        let (own_bodies, own_colliders);
        let (bodies, colliders, max_bodies) = match &target {
            Some(t) => (t.bodies, t.colliders, t.max_bodies),
            None => {
                own_bodies = gpu.storage("spawn bodies", &vec![[0f32; 4]; 7 * total]);
                own_colliders = gpu.storage("spawn colliders", &vec![[0f32; 4]; 4 * n_parts]);
                (&own_bodies, &own_colliders, 0)
            }
        };
        // Each parent's first chunk with its retire record.
        let parent_table: Vec<[u32; 4]> = gpu.read(&run.parents);
        let firsts_host: Vec<[u32; 4]> = gpu.read(&run.firsts);
        let islands_of: Vec<[u32; 4]> = (0..run.parent_count)
            .map(|p| {
                let r = retire.get(p).copied().unwrap_or([0, 0, 0]);
                [firsts_host[p][0], r[0], r[1], r[2]]
            })
            .collect();
        let islands_of = gpu.storage("spawn parents", if islands_of.is_empty() { &[[0u32; 4]][..] } else { &islands_of });
        let params = gpu.uniform("spawn params", &SpawnParams { parent_count: run.parent_count as u32, first_body, first_collider, live: live as u32, max_bodies, pad: [0; 3] });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for (k, size) in [(&self.offsets, 1u32), (&self.spawn, groups(total, 64)), (&self.retire, groups(run.parent_count, 64))] {
            let bind = gpu.bind(&k.group, &[&params, &run.parents, &run.work, &run.out, &run.frames, &table.ranges, &table.parts, chunks, state, &slots, bodies, colliders, &islands_of, &extras, &table.structure_base]);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(size.max(1), 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        let slots: Vec<[u32; 4]> = gpu.read(&slots);
        let extras: Vec<[f32; 4]> = gpu.read(&extras);
        // The records written: all of a copy, or the engine's new range.
        let made: Vec<&[u32; 4]> = (0..run.parent_count).flat_map(|p| (0..parent_table[p][2] as usize).map(move |k| (p, k))).map(|(p, k)| &slots[parent_table[p][1] as usize + k]).filter(|s| s[3] != 0).collect();
        let n_bodies: u32 = made.len() as u32;
        let n_colliders: u32 = made.iter().map(|s| s[2]).sum();
        let (body_words, col_words): (Vec<[f32; 4]>, Vec<[f32; 4]>) = if live {
            let read = gpu.read_ranges(&[(bodies, 112 * first_body as u64, 112 * n_bodies as u64), (colliders, 64 * first_collider as u64, 64 * n_colliders as u64)]);
            (bytemuck::pod_collect_to_vec(&read[0]), bytemuck::pod_collect_to_vec(&read[1]))
        } else {
            (gpu.read(bodies), gpu.read(colliders))
        };
        let mut out = Vec::new();
        for p in 0..run.parent_count {
            for k in 0..parent_table[p][2] as usize {
                let slot = parent_table[p][1] as usize + k;
                let s = slots[slot];
                if s[3] == 0 {
                    continue;
                }
                let b = 7 * (s[0] - first_body) as usize;
                let mut body = [[0f32; 4]; 7];
                body.copy_from_slice(&body_words[b..b + 7]);
                let cols = (0..s[2] as usize)
                    .map(|j| {
                        let c = 4 * ((s[1] - first_collider) as usize + j);
                        [col_words[c], col_words[c + 1], col_words[c + 2], col_words[c + 3]]
                    })
                    .collect();
                out.push(GpuBody { slot: s[0], first_collider: s[1], body, colliders: cols, com: extras[2 * slot], principal: extras[2 * slot + 1] });
            }
        }
        out
    }
}

/// How GPU-made bodies compared with the host's (`STRESS_GPU_SPLIT_CHECK=1`).
#[derive(Clone, Copy, Debug, Default)]
pub struct SpawnCheck {
    pub bodies: u64,
    /// Bodies at another slot than the host's, or with another collider count.
    pub slot_mismatch: u64,
    pub collider_count_mismatch: u64,
    /// Flags, shapes, materials or shape parameters differing.
    pub record_mismatch: u64,
    /// Largest differences: body position (m), rotation (matrix entries), inverse mass
    /// and world inverse inertia (relative), velocity and spin (m/s, rad/s), collider
    /// world position (m) and rotation (matrix entries).
    pub position: f64,
    pub rotation: f64,
    pub inv_mass: f64,
    pub inv_inertia: f64,
    pub velocity: f64,
    pub spin: f64,
    pub collider_position: f64,
    pub collider_rotation: f64,
}

fn quat_mat(q: [f32; 4]) -> [[f64; 3]; 3] {
    let (x, y, z, w) = (q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

fn mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
}

fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [0, 1, 2].map(|r| [0, 1, 2].map(|c| a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c]))
}

fn v3(v: [f32; 4]) -> [f64; 3] {
    [v[0] as f64, v[1] as f64, v[2] as f64]
}

impl SpawnCheck {
    /// One body: the GPU's against the host's (slot, raw records).
    pub fn compare(&mut self, gpu: &GpuBody, host_slot: u32, host: &([[f32; 4]; 7], Vec<[[f32; 4]; 4]>)) {
        self.bodies += 1;
        if gpu.slot != host_slot {
            self.slot_mismatch += 1;
        }
        let (g, h) = (&gpu.body, &host.0);
        let dist = |a: [f64; 3], b: [f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        self.position = self.position.max(dist(v3(g[0]), v3(h[0])));
        let (rg, rh) = (quat_mat(g[1]), quat_mat(h[1]));
        for r in 0..3 {
            for c in 0..3 {
                self.rotation = self.rotation.max((rg[r][c] - rh[r][c]).abs());
            }
        }
        self.inv_mass = self.inv_mass.max((g[0][3] as f64 - h[0][3] as f64).abs() / (h[0][3] as f64).abs().max(1e-30));
        // World inverse inertia R diag(inv) R^T.
        let world = |r: &[[f64; 3]; 3], d: [f64; 3]| {
            let rd = [0, 1, 2].map(|i| [0, 1, 2].map(|j| r[i][j] * d[j]));
            let rt = [0, 1, 2].map(|i| [0, 1, 2].map(|j| r[j][i]));
            mat_mul(&rd, &rt)
        };
        let (ig, ih) = (world(&rg, v3(g[2])), world(&rh, v3(h[2])));
        let scale = ih.iter().flatten().fold(0.0f64, |a, &b| a.max(b.abs())).max(1e-30);
        for r in 0..3 {
            for c in 0..3 {
                self.inv_inertia = self.inv_inertia.max((ig[r][c] - ih[r][c]).abs() / scale);
            }
        }
        if g[2][3].to_bits() != h[2][3].to_bits() {
            self.record_mismatch += 1;
        }
        self.velocity = self.velocity.max(dist(v3(g[3]), v3(h[3])));
        self.spin = self.spin.max(dist(v3(g[4]), v3(h[4])));
        if gpu.colliders.len() != host.1.len() {
            self.collider_count_mismatch += 1;
            return;
        }
        for (gc, hc) in gpu.colliders.iter().zip(&host.1) {
            if gc[0][1].to_bits() != hc[0][1].to_bits() || gc[0][2].to_bits() != hc[0][2].to_bits() || gc[3] != hc[3] {
                self.record_mismatch += 1;
            }
            let pg = mat_vec(&rg, v3(gc[1]));
            let ph = mat_vec(&rh, v3(hc[1]));
            self.collider_position = self.collider_position.max(dist([pg[0] + g[0][0] as f64, pg[1] + g[0][1] as f64, pg[2] + g[0][2] as f64], [ph[0] + h[0][0] as f64, ph[1] + h[0][1] as f64, ph[2] + h[0][2] as f64]));
            let (cg, ch) = (mat_mul(&rg, &quat_mat(gc[2])), mat_mul(&rh, &quat_mat(hc[2])));
            for r in 0..3 {
                for c in 0..3 {
                    self.collider_rotation = self.collider_rotation.max((cg[r][c] - ch[r][c]).abs());
                }
            }
        }
    }
}
