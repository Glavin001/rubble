//! Fracture topology on the GPU (`shaders/split.slang`, engine/docs/stress-baseline-goal.md
//! M2): the children of disconnected islands, as stress-ref `split_cluster` makes them, from
//! the GPU's tables and state. `STRESS_GPU_SPLIT_CHECK=1` compares every split of the
//! resident frame with the host's (`SplitCheck`).

use crate::gpu::{groups, Binding, Gpu, Kernel};
use crate::shaders;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SplitParams {
    pub(crate) parent_count: u32,
    pub(crate) out_base: u32,
    /// 1: an anchored parent's frame from its island record (`split_frames`;
    /// `STRESS_GPU_ANCHORED_ENGINE_FRAME=1`: from its fixed engine body, as before).
    pub(crate) anchored_pose: u32,
    pub(crate) pad: u32,
}

const BINDINGS: [Binding; 14] = [
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
    Binding::StorageRw, // frames (each parent's rigid frame)
    Binding::Storage,   // engine bodies (split_frames)
    Binding::Storage,   // parent bodies (split_frames)
    Binding::StorageRw, // control (the split pass's control words)
];

pub(crate) const CHILD_WORDS: usize = 8;

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
    pub(crate) components: Kernel,
    pub(crate) children: Kernel,
    pub(crate) children_group: Kernel,
    pub(crate) frames: Kernel,
    pub(crate) offsets: Kernel,
    pub(crate) spawn: Kernel,
    pub(crate) retire: Kernel,
    pub(crate) spawn_colliders: Kernel,
    pub(crate) spawn_group: Kernel,
    /// The split pass's copies of the tables as they were (`pass`), kept between splits:
    /// a new buffer per split costs its allocation and zero fill.
    old: std::cell::RefCell<Vec<Option<wgpu::Buffer>>>,
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
pub(crate) struct SpawnParams {
    pub(crate) parent_count: u32,
    pub(crate) first_body: u32,
    pub(crate) first_collider: u32,
    pub(crate) live: u32,
    pub(crate) max_bodies: u32,
    pub(crate) max_colliders: u32,
    pub(crate) island_cap: u32,
    pub(crate) pad: u32,
}

/// The engine's own body and collider buffers, for a live spawn (bodies in two regions,
/// the second at `max_bodies`).
pub struct SpawnTarget<'a> {
    pub bodies: &'a wgpu::Buffer,
    pub colliders: &'a wgpu::Buffer,
    pub max_bodies: u32,
}

const SPAWN_BINDINGS: [Binding; 17] = [
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
    Binding::StorageRw, // control
    Binding::StorageRw, // member offsets (each member's first collider)
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

/// A spawn's child slots and extras, on the GPU (for the layout's resident tables).
pub struct SpawnBuffers {
    pub slots: wgpu::Buffer,
    pub extras: wgpu::Buffer,
    /// [0]: 1 when the split did not fit (`spawn.slang` `control`).
    pub control: wgpu::Buffer,
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
        GpuSplitter { components: k("split_components"), children: k("split_children"), children_group: k("split_children_group"), frames: k("split_frames"), offsets: s("spawn_offsets"), spawn: s("spawn_bodies"), retire: s("spawn_retire"), spawn_colliders: s("spawn_colliders"), spawn_group: s("spawn_bodies_group"), old: std::cell::RefCell::new(vec![None; 8]) }
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
        let params = gpu.uniform("split params", &SplitParams { parent_count: parents.len() as u32, out_base: 0, anchored_pose: anchored_pose(), pad: 0 });
        let [chunks, index, bonds, bond_dyn, islands, state] = buffers;
        let none_bodies = gpu.storage("split no engine bodies", &[[0f32; 4]; 7]);
        let none_parents = gpu.storage("split no parent bodies", &[[0f32; 4]; 2]);
        let control = gpu.storage("split control", &control_words(parents.len() as u32, 0, 0, 0, 0, total as u32));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for (k, size) in [(&self.components, parents.len().max(1) as u32), (&self.children, groups(total, 64))] {
            let bind = gpu.bind(&k.group, &[&params, chunks, index, bonds, bond_dyn, islands, state, &parents_buf, &work_buf, &out, &frames, &none_bodies, &none_parents, &control]);
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
    ) -> (Vec<GpuBody>, SpawnBuffers) {
        let total = run.total.max(1);
        // At most every part of the children's chunks once.
        let n_parts: usize = (table.parts.size() as usize / 128).max(1);
        let slots_buf = gpu.storage("spawn slots", &vec![[0u32; 4]; total]);
        let extras_buf = gpu.storage("spawn extras", &vec![[0f32; 4]; 2 * total]);
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
        let params = gpu.uniform("spawn params", &SpawnParams { parent_count: run.parent_count as u32, first_body, first_collider, live: live as u32, max_bodies, max_colliders: u32::MAX, island_cap: u32::MAX, pad: 0 });
        let control = gpu.storage("spawn control", &control_words(run.parent_count as u32, 0, 0, first_body, first_collider, total as u32));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let member_offsets = gpu.storage("spawn member offsets", &vec![0u32; total.max(1)]);
        let bodies_step = if split_tree() { (&self.spawn_group, total.max(1) as u32) } else { (&self.spawn, groups(total, 64)) };
        for (k, size) in [(&self.offsets, 1u32), bodies_step, (&self.spawn_colliders, groups(total, 64)), (&self.retire, run.parent_count.max(1) as u32)] {
            let bind = gpu.bind(&k.group, &[&params, &run.parents, &run.work, &run.out, &run.frames, &table.ranges, &table.parts, chunks, state, &slots_buf, bodies, colliders, &islands_of, &extras_buf, &table.structure_base, &control, &member_offsets]);
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(size.max(1), 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        let slots: Vec<[u32; 4]> = gpu.read(&slots_buf);
        let extras: Vec<[f32; 4]> = gpu.read(&extras_buf);
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
        (out, SpawnBuffers { slots: slots_buf, extras: extras_buf, control })
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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayoutParams {
    pub parent_count: u32,
    pub island_count: u32,
    pub next_id: u32,
    pub substeps: u32,
    pub chunk_words: u32,
    pub bond_words: u32,
    pub dyn_words: u32,
    pub island_words: u32,
    pub chunk_count: u32,
    pub statics_base: u32,
    pub statics_bonds: u32,
    pub resident: u32,
    pub props_base: u32,
    pub eload_base: u32,
    pub pulse_base: u32,
    pub pulse_count_base: u32,
    pub pulse_cap: u32,
    pub engine_base: u32,
    pub total: u32,
    pub gate_flag_base: u32,
    pub bond_count: u32,
    pub driven: u32,
    pub pad: [u32; 2],
}

const LAYOUT_BINDINGS: [Binding; 27] = [
    Binding::Uniform,   // params
    Binding::Storage,   // parents
    Binding::Storage,   // work
    Binding::Storage,   // children
    Binding::Storage,   // frames
    Binding::StorageRw, // plan
    Binding::Storage,   // old chunks
    Binding::StorageRw, // chunks
    Binding::Storage,   // old state
    Binding::StorageRw, // state
    Binding::Storage,   // old bonds
    Binding::StorageRw, // bonds
    Binding::Storage,   // old bond records
    Binding::StorageRw, // bond records
    Binding::Storage,   // old index
    Binding::StorageRw, // index
    Binding::Storage,   // old islands
    Binding::StorageRw, // islands
    Binding::StorageRw, // scratch
    Binding::Storage,   // old ctab
    Binding::StorageRw, // ctab
    Binding::StorageRw, // gate
    Binding::Storage,   // spawn slots
    Binding::Storage,   // spawn extras
    Binding::Storage,   // control
    Binding::Storage,   // the cascade's island copy as it was
    Binding::StorageRw, // the cascade's island copy
];

/// The tables a split re-lays out (`layout.slang`): each as (as it was, to write).
pub struct LayoutTables<'a> {
    pub chunks: (&'a wgpu::Buffer, &'a wgpu::Buffer),
    pub state: (&'a wgpu::Buffer, &'a wgpu::Buffer),
    pub bonds: (&'a wgpu::Buffer, &'a wgpu::Buffer),
    pub bond_dyn: (&'a wgpu::Buffer, &'a wgpu::Buffer),
    pub index: (&'a wgpu::Buffer, &'a wgpu::Buffer),
    pub islands: (&'a wgpu::Buffer, &'a wgpu::Buffer),
    pub scratch: &'a wgpu::Buffer,
    pub ctab: (&'a wgpu::Buffer, &'a wgpu::Buffer),
    pub gate: &'a wgpu::Buffer,
    /// The spawn's child slots, extras and control (`spawn.slang`), for the resident
    /// tables and the abort.
    pub spawn: (&'a wgpu::Buffer, &'a wgpu::Buffer, &'a wgpu::Buffer),
}

#[derive(Clone)]
pub struct GpuLayout {
    pub(crate) plan: std::rc::Rc<Kernel>,
    pub(crate) moved: std::rc::Rc<Kernel>,
    pub(crate) parent: std::rc::Rc<Kernel>,
    pub(crate) copy: std::rc::Rc<Kernel>,
}

impl GpuLayout {
    pub fn new(gpu: &Gpu) -> GpuLayout {
        let k = |e: &str| std::rc::Rc::new(gpu.compute(&shaders::LAYOUT, e, &LAYOUT_BINDINGS));
        GpuLayout { plan: k("layout_plan"), moved: k("layout_moved"), parent: k("layout_parent"), copy: k("layout_copy") }
    }

    pub fn clone_kernels(&self) -> GpuLayout {
        self.clone()
    }

    /// `run`'s split re-laid out in `tables` (`params.total`, `parent_count` from the run);
    /// `island_cap`: islands the tables hold. Returns the plan buffer (its island order).
    pub fn layout(&self, gpu: &Gpu, run: &SplitRun, tables: &LayoutTables, mut params: LayoutParams, island_cap: usize) -> wgpu::Buffer {
        params.parent_count = run.parent_count as u32;
        params.total = run.total as u32;
        let cap = params.island_count as usize + run.total;
        let words = 16 + cap + 8 * run.total + params.island_count as usize + params.bond_count as usize + 1;
        let plan = gpu.storage("layout plan", &vec![0u32; words]);
        let uniform = gpu.uniform("layout params", &params);
        // The control words the layout reads (the spawn's buffer keeps its own).
        gpu.queue.write_buffer(tables.spawn.2, 4 * CTL_PARENTS, bytemuck::cast_slice(&[params.parent_count, params.island_count, params.next_id]));
        gpu.queue.write_buffer(tables.spawn.2, 4 * CTL_TOTAL, bytemuck::cast_slice(&[params.total]));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let no_copy = gpu.storage("layout no island copy", &[[0f32; 4]; 1]);
        let no_copy_out = gpu.storage("layout no island copy out", &[[0f32; 4]; 1]);
        for (k, size) in [(&*self.plan, 1u32), (&*self.moved, groups(island_cap + 1, 64)), (&*self.parent, run.parent_count.max(1) as u32)] {
            let t = tables;
            let bind = gpu.bind(
                &k.group,
                &[&uniform, &run.parents, &run.work, &run.out, &run.frames, &plan, t.chunks.0, t.chunks.1, t.state.0, t.state.1, t.bonds.0, t.bonds.1, t.bond_dyn.0, t.bond_dyn.1, t.index.0, t.index.1, t.islands.0, t.islands.1, t.scratch, t.ctab.0, t.ctab.1, t.gate, t.spawn.0, t.spawn.1, t.spawn.2, &no_copy, &no_copy_out],
            );
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(size.max(1), 1, 1);
        }
        gpu.queue.submit([encoder.finish()]);
        plan
    }
}

/// How a GPU layout compared with the host's in-place split (`STRESS_GPU_SPLIT_CHECK=1`
/// with `STRESS_QS_SPLIT_IN_PLACE=1`): words differing per table, and the island records'
/// largest float difference (relative to each field's size).
#[derive(Clone, Copy, Debug, Default)]
pub struct LayoutCheck {
    pub layouts: u64,
    pub chunk_words: u64,
    pub state_words: u64,
    pub bond_words: u64,
    pub dyn_words: u64,
    pub index_words: u64,
    pub island_int_words: u64,
    pub island_float: f64,
}

/// A split pass's engine side (`GpuSplitter::pass`): each parent's body and frame data,
/// the engine's buffers and capacities, where its next bodies go, the parents to retire.
pub struct PassEngine<'a> {
    /// Per parent: (body as bits, centre of mass) and principal rotation (x, y, z, w).
    pub parent_bodies: Vec<[[f32; 4]; 2]>,
    pub target: SpawnTarget<'a>,
    pub max_colliders: u32,
    pub first_body: u32,
    pub first_collider: u32,
    pub retire: Vec<[u32; 3]>,
}

/// What a split pass gives the host, in one read.
pub struct PassResult {
    /// The split did not fit (nothing was made, retired or laid out).
    pub aborted: bool,
    pub children: Vec<Vec<GpuChild>>,
    pub bodies: Vec<GpuBody>,
}

impl GpuSplitter {
    /// A whole split on the GPU in one submission and one read (M4): its frames from the
    /// engine's bodies, its topology, its children's bodies made and the parents retired in
    /// the engine's buffers, and its tables laid out (`layout`, from copies made first).
    #[allow(clippy::too_many_arguments)]
    pub fn pass(
        &self,
        gpu: &Gpu,
        layout: &GpuLayout,
        buffers: [&wgpu::Buffer; 6],
        scratch: &wgpu::Buffer,
        parents: &[(usize, [u32; 2])],
        bond_ranges: &[[u32; 2]],
        engine: &PassEngine,
        table: &PartTable,
        layout_params: LayoutParams,
        island_cap: usize,
        resident: Option<(&wgpu::Buffer, [u32; 6])>,
        gate: &wgpu::Buffer,
        parts_max: usize,
        colliders: Option<(&crate::colliders::ColliderRefresh, &wgpu::Buffer)>,
        island_copy: Option<&wgpu::Buffer>,
    ) -> PassResult {
        let [chunks, index, bonds, bond_dyn, islands, state] = buffers;
        // The split's buffers (as `split`).
        let mut table_host: Vec<[u32; 4]> = Vec::with_capacity(parents.len() + 1);
        let mut base = 0u32;
        for &(island, r) in parents {
            table_host.push([island as u32, base, 0, 0]);
            base += r[1] - r[0];
        }
        table_host.push([0, base, 0, 0]);
        let total = base as usize;
        let parents_buf = gpu.storage("pass parents", &table_host);
        let work = gpu.storage("pass work", &vec![0u32; 2 * total.max(1)]);
        let out = gpu.storage("pass out", &vec![[0f32; 4]; CHILD_WORDS * total.max(1)]);
        let frames = gpu.storage("pass frames", &vec![[0f32; 4]; 5 * parents.len().max(1)]);
        let pb: Vec<[f32; 4]> = engine.parent_bodies.iter().flatten().copied().collect();
        let parent_bodies = gpu.storage("pass parent bodies", if pb.is_empty() { &[[0f32; 4]; 2][..] } else { &pb });
        let split_params = gpu.uniform("pass split params", &SplitParams { parent_count: parents.len() as u32, out_base: 0, anchored_pose: anchored_pose(), pad: 0 });
        // The spawn's.
        let slots = gpu.storage("pass slots", &vec![[0u32; 4]; total.max(1)]);
        let extras = gpu.storage("pass extras", &vec![[0f32; 4]; 2 * total.max(1)]);
        let control = gpu.storage("pass control", &control_words(parents.len() as u32, layout_params.island_count, layout_params.next_id, engine.first_body, engine.first_collider, total as u32));
        let firsts: Vec<[u32; 4]> = parents.iter().zip(&engine.retire).map(|(p, r)| [p.1[0], r[0], r[1], r[2]]).collect();
        let firsts = gpu.storage("pass parents' records", if firsts.is_empty() { &[[0u32; 4]][..] } else { &firsts });

        let spawn_params = gpu.uniform(
            "pass spawn params",
            &SpawnParams { parent_count: parents.len() as u32, first_body: engine.first_body, first_collider: engine.first_collider, live: 1, max_bodies: engine.target.max_bodies, max_colliders: engine.max_colliders, island_cap: island_cap as u32, pad: 0 },
        );
        // The layout's: copies of the tables as they are, its plan.
        let mut params = layout_params;
        params.parent_count = parents.len() as u32;
        params.total = total as u32;
        if let Some((_, r)) = &resident {
            params.resident = 1;
            [params.props_base, params.eload_base, params.pulse_base, params.pulse_count_base, params.pulse_cap, params.engine_base] = *r;
        }
        let cap = params.island_count as usize + total;
        let plan = gpu.storage("pass plan", &vec![0u32; 16 + cap + 8 * total + params.island_count as usize + params.bond_count as usize + 1]);
        let layout_uniform = gpu.uniform("pass layout params", &params);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        // Copies of the tables as they were, of only what the layout reads: each parent's
        // chunk and bond ranges (chunk statics, state, bond statics and records, the
        // resident tables' per-chunk entries) and, whole, the incidence, the islands and the
        // resident tables' island entries (a moved island is any island). Into buffers kept
        // between splits; the rest of them is never read.
        let mut cache = self.old.borrow_mut();
        let mut copy = |slot: usize, src: &wgpu::Buffer, label: &str, ranges: Option<&[(u64, u64)]>, encoder: &mut wgpu::CommandEncoder| {
            let dst = match &cache[slot] {
                Some(b) if b.size() >= src.size() => b.clone(),
                _ => {
                    let b = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: src.size(), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                    cache[slot] = Some(b.clone());
                    b
                }
            };
            match ranges {
                Some(ranges) => {
                    for &(at, len) in ranges.iter().filter(|r| r.1 > 0) {
                        encoder.copy_buffer_to_buffer(src, at, &dst, at, len);
                    }
                }
                None => encoder.copy_buffer_to_buffer(src, 0, &dst, 0, src.size()),
            }
            dst
        };
        let f4 = 16u64;
        let per_chunk = |words: u64| -> Vec<(u64, u64)> { parents.iter().map(|&(_, r)| (f4 * words * r[0] as u64, f4 * words * (r[1] - r[0]) as u64)).collect() };
        let per_bond = |words: u64| -> Vec<(u64, u64)> { bond_ranges.iter().map(|r| (f4 * words * r[0] as u64, f4 * words * (r[1] - r[0]) as u64)).collect() };
        let old: Vec<wgpu::Buffer> = vec![
            copy(0, chunks, "old chunks", Some(&per_chunk(params.chunk_words as u64)), &mut encoder),
            copy(1, state, "old state", Some(&per_chunk(4)), &mut encoder),
            copy(2, bonds, "old bonds", Some(&per_bond(params.bond_words as u64)), &mut encoder),
            copy(3, bond_dyn, "old bond records", Some(&per_bond(params.dyn_words as u64)), &mut encoder),
            copy(4, index, "old index", None, &mut encoder),
            copy(5, islands, "old islands", None, &mut encoder),
        ];
        let dummy = gpu.storage("pass no ctab", &[[0f32; 4]; 1]);
        let dummy_out = gpu.storage("pass no ctab out", &[[0f32; 4]; 1]);
        let old_ctab = resident.as_ref().map(|(c, _)| {
            let mut ranges: Vec<(u64, u64)> = Vec::new();
            for &(_, r) in parents {
                let (c0, n) = (r[0] as u64, (r[1] - r[0]) as u64);
                ranges.push((f4 * (params.props_base as u64 + c0), f4 * n));
                ranges.push((f4 * (params.eload_base as u64 + 2 * c0), f4 * 2 * n));
                ranges.push((f4 * (params.pulse_count_base as u64 + c0), f4 * n));
                ranges.push((f4 * (params.pulse_base as u64 + 3 * params.pulse_cap as u64 * c0), f4 * 3 * params.pulse_cap as u64 * n));
            }
            ranges.push((f4 * params.engine_base as u64, (f4 * 3 * (params.island_count as u64 + 1)).min(c.size().saturating_sub(f4 * params.engine_base as u64))));
            copy(6, c, "old ctab", Some(&ranges), &mut encoder)
        });
        let old_copy = island_copy.map(|c| copy(7, c, "old island copy", None, &mut encoder));
        drop(cache);
        let no_copy = gpu.storage("pass no island copy", &[[0f32; 4]; 1]);
        let no_copy_out = gpu.storage("pass no island copy out", &[[0f32; 4]; 1]);
        let (copy_old, copy_new): (&wgpu::Buffer, &wgpu::Buffer) = match (&old_copy, island_copy) {
            (Some(o), Some(c)) => (o, c),
            _ => (&no_copy, &no_copy_out),
        };
        let (ctab_old, ctab_new): (&wgpu::Buffer, &wgpu::Buffer) = match (&old_ctab, &resident) {
            (Some(o), Some((c, _))) => (o, c),
            _ => (&dummy, &dummy_out),
        };
        let split_bind = |k: &Kernel| gpu.bind(&k.group, &[&split_params, chunks, index, bonds, bond_dyn, islands, state, &parents_buf, &work, &out, &frames, engine.target.bodies, &parent_bodies, &control]);
        let member_offsets = gpu.storage("pass member offsets", &vec![0u32; total.max(1)]);
        let spawn_bind = |k: &Kernel| {
            gpu.bind(&k.group, &[&spawn_params, &parents_buf, &work, &out, &frames, &table.ranges, &table.parts, chunks, state, &slots, engine.target.bodies, engine.target.colliders, &firsts, &extras, &table.structure_base, &control, &member_offsets])
        };
        let layout_bind = |k: &Kernel| {
            gpu.bind(
                &k.group,
                &[&layout_uniform, &parents_buf, &work, &out, &frames, &plan, &old[0], chunks, &old[1], state, &old[2], bonds, &old[3], bond_dyn, &old[4], index, &old[5], islands, scratch, ctab_old, ctab_new, gate, &slots, &extras, &control, copy_old, copy_new],
            )
        };
        let n = parents.len().max(1) as u32;
        let mut steps: Vec<(&Kernel, wgpu::BindGroup, u32)> = vec![
            (&self.frames, split_bind(&self.frames), groups(parents.len(), 64)),
            (&self.components, split_bind(&self.components), n),
            (&self.children, split_bind(&self.children), groups(total, 64)),
            (&self.offsets, spawn_bind(&self.offsets), 1),
            (&self.spawn, spawn_bind(&self.spawn), groups(total, 64)),
            (&self.spawn_colliders, spawn_bind(&self.spawn_colliders), groups(total, 64)),
            (&self.retire, spawn_bind(&self.retire), n),
            (&layout.plan, layout_bind(&layout.plan), 1),
            (&layout.moved, layout_bind(&layout.moved), groups(island_cap + 1, 64)),
            (&layout.parent, layout_bind(&layout.parent), n),
        ];
        if island_copy.is_some() {
            steps.push((&layout.copy, layout_bind(&layout.copy), groups(island_cap + 1, 64)));
        }
        // `STRESS_GPU_TIMING=1`: each step's GPU time (a diagnostic read).
        let timing = (std::env::var("STRESS_GPU_TIMING").is_ok_and(|v| v == "1") && gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY))
            .then(|| gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("split pass timestamps"), ty: wgpu::QueryType::Timestamp, count: 2 * steps.len() as u32 }));
        for (i, (k, bind, size)) in steps.iter().enumerate() {
            let stamps = timing.as_ref().map(|q| wgpu::ComputePassTimestampWrites { query_set: q, beginning_of_pass_write_index: Some(2 * i as u32), end_of_pass_write_index: Some(2 * i as u32 + 1) });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: stamps });
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups((*size).max(1), 1, 1);
        }
        let timing_resolve = timing.as_ref().map(|q| {
            let b = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("split pass timestamps"), size: 8 * 2 * steps.len() as u64, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            encoder.resolve_query_set(q, 0..2 * steps.len() as u32, &b, 0);
            b
        });
        // The coupling's collider table after the layout (colliders.slang).
        if let Some((refresh, collider_info)) = colliders {
            let inputs = crate::colliders::RefreshInputs {
                parent_count: parents.len() as u32,
                chunk_count: params.chunk_count,
                parents: &parents_buf,
                work: &work,
                children: &out,
                slots: &slots,
                part_ranges: &table.ranges,
                old_chunks: &old[0],
                chunks,
                islands,
                islands_of: &firsts,
                structure_base: &table.structure_base,
                control: &control,
                member_offsets: &member_offsets,
            };
            refresh.encode(gpu, &mut encoder, &inputs, total, collider_info);
        }
        let t_submit = std::time::Instant::now();
        gpu.queue.submit([encoder.finish()]);
        // The one read.
        let body_words = 112u64;
        let col_words = 64u64;
        let bodies_max = (total as u64).min(engine.target.max_bodies.saturating_sub(engine.first_body) as u64);
        let cols_max = (parts_max as u64).min(engine.max_colliders.saturating_sub(engine.first_collider) as u64);
        let read = gpu.read_ranges(&[
            (&parents_buf, 0, 16 * (parents.len() as u64 + 1)),
            (&work, 4 * total as u64, 4 * total as u64),
            (&out, 0, 16 * (CHILD_WORDS * total) as u64),
            (&slots, 0, 16 * total as u64),
            (&extras, 0, 32 * total as u64),
            (&control, 0, 4 * CTL_WORDS as u64),
            (engine.target.bodies, body_words * engine.first_body as u64, body_words * bodies_max),
            (engine.target.colliders, col_words * engine.first_collider as u64, col_words * cols_max),
        ]);
        if std::env::var("STRESS_GPU_QS_TRACE").is_ok() {
            eprintln!("split pass: {} parents, {} chunks: GPU and read {:.1} ms", parents.len(), total, t_submit.elapsed().as_secs_f64() * 1e3);
        }
        if let Some(b) = timing_resolve {
            let t: Vec<u64> = gpu.read(&b);
            let period = gpu.queue.get_timestamp_period() as f64 * 1e-6;
            let names = ["frames", "components", "children", "offsets", "spawn", "colliders", "retire", "plan", "moved", "parent", "copy"];
            let ms: Vec<String> = (0..t.len() / 2).map(|q| format!("{} {:.2}", names.get(q).unwrap_or(&"?"), t[2 * q + 1].saturating_sub(t[2 * q]) as f64 * period)).collect();
            eprintln!("split pass GPU ms: {}", ms.join(", "));
        }
        let table_read: Vec<[u32; 4]> = bytemuck::pod_collect_to_vec(&read[0]);
        let members: Vec<u32> = bytemuck::pod_collect_to_vec(&read[1]);
        let out_words: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&read[2]);
        let slot_words: Vec<[u32; 4]> = bytemuck::pod_collect_to_vec(&read[3]);
        let extra_words: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&read[4]);
        let control_words: Vec<u32> = bytemuck::pod_collect_to_vec(&read[5]);
        let new_bodies: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&read[6]);
        let new_cols: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&read[7]);
        let aborted = control_words.first().copied().unwrap_or(0) != 0;
        let children: Vec<Vec<GpuChild>> = parents
            .iter()
            .enumerate()
            .map(|(p, &(_, r))| {
                let base = table_read[p][1] as usize;
                (0..table_read[p][2] as usize)
                    .map(|k| {
                        let rec = &out_words[CHILD_WORDS * (base + k)..CHILD_WORDS * (base + k + 1)];
                        let (offset, count) = (rec[0][1].to_bits() as usize, rec[0][2].to_bits() as usize);
                        GpuChild {
                            chunks: (0..count).map(|j| r[0] + members[base + offset + j]).collect(),
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
        let mut bodies = Vec::new();
        if !aborted {
            for p in 0..parents.len() {
                for k in 0..table_read[p][2] as usize {
                    let slot = table_read[p][1] as usize + k;
                    let s = slot_words[slot];
                    if s[3] == 0 {
                        continue;
                    }
                    let b = 7 * (s[0] - engine.first_body) as usize;
                    let mut body = [[0f32; 4]; 7];
                    body.copy_from_slice(&new_bodies[b..b + 7]);
                    let cols = (0..s[2] as usize)
                        .map(|j| {
                            let c = 4 * ((s[1] - engine.first_collider) as usize + j);
                            [new_cols[c], new_cols[c + 1], new_cols[c + 2], new_cols[c + 3]]
                        })
                        .collect();
                    bodies.push(GpuBody { slot: s[0], first_collider: s[1], body, colliders: cols, com: extra_words[2 * slot], principal: extra_words[2 * slot + 1] });
                }
            }
        }
        PassResult { aborted, children, bodies }
    }
}

/// `SplitParams::anchored_pose`: an anchored parent's frame from its island record, unless
/// `STRESS_GPU_ANCHORED_ENGINE_FRAME=1`.
pub(crate) fn anchored_pose() -> u32 {
    std::env::var("STRESS_GPU_ANCHORED_ENGINE_FRAME").map_or(1, |v| (v != "1") as u32)
}

/// The split pass's control words (common.slang `CTL_*`).
pub const CTL_PARENTS: u64 = 1;
pub const CTL_TOTAL: u64 = 6;
pub const CTL_NEW_ISLANDS: usize = 7;
pub const CTL_WORDS: usize = 16;

/// A split pass's control words as the host sets them: parents, live islands, the next
/// cluster id, the engine's bodies and colliders, child slots.
fn control_words(parents: u32, islands: u32, next_id: u32, first_body: u32, first_collider: u32, total: u32) -> [u32; CTL_WORDS] {
    let mut w = [0u32; CTL_WORDS];
    w[1] = parents;
    w[2] = islands;
    w[3] = next_id;
    w[4] = first_body;
    w[5] = first_collider;
    w[6] = total;
    w
}

/// `split_children_group` (a group per child slot) in the GPU split pass, unless
/// `STRESS_GPU_SPLIT_TREE=0` (one thread per child, as the per-split path runs it).
pub(crate) fn split_tree() -> bool {
    std::env::var("STRESS_GPU_SPLIT_TREE").map_or(true, |v| v != "0")
}
