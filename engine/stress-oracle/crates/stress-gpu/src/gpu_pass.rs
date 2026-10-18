//! A split pass whose inputs the GPU makes (`shaders/pass.slang`, engine/docs/stress-baseline-goal.md
//! M4): the parents from the cascade's status, their engine data from the resident tables,
//! the counts from GPU counters, every size-dependent dispatch indirect. It encodes into the
//! caller's encoder and needs nothing from the host, so a frame can run its cascade passes
//! and the splits between them in one submission. Its buffers are made once, for the
//! largest split the tables can hold (every island a parent, every chunk a child slot).

use crate::gpu::{groups, Binding, Gpu, Kernel};
use crate::gpu_split::{GpuBody, GpuChild, GpuLayout, GpuSplitter, LayoutParams, PartTable, PassResult, SpawnParams, SpawnTarget, SplitParams, CHILD_WORDS, CTL_WORDS};
use crate::shaders;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PassParams {
    chunk_words: u32,
    bond_words: u32,
    dyn_words: u32,
    island_words: u32,
    props_base: u32,
    eload_base: u32,
    pulse_base: u32,
    pulse_count_base: u32,
    pulse_cap: u32,
    engine_base: u32,
    resident: u32,
    resume: u32,
    island_float4s: u32,
    index_words: u32,
    engine_float4s: u32,
    copy_float4s: u32,
}

const PASS_BINDINGS: [Binding; 25] = [
    Binding::Uniform,   // params
    Binding::StorageRw, // cascade status
    Binding::Storage,   // islands
    Binding::Storage,   // ctab
    Binding::StorageRw, // control
    Binding::StorageRw, // parents
    Binding::StorageRw, // islands_of
    Binding::StorageRw, // parent bodies
    Binding::StorageRw, // indirect arguments
    Binding::StorageRw, // counters
    Binding::Storage,   // chunks
    Binding::StorageRw, // old chunks
    Binding::Storage,   // state
    Binding::StorageRw, // old state
    Binding::Storage,   // bonds
    Binding::StorageRw, // old bonds
    Binding::Storage,   // bond records
    Binding::StorageRw, // old bond records
    Binding::StorageRw, // old ctab
    Binding::StorageRw, // the cascade's island arguments
    Binding::Storage,   // index
    Binding::StorageRw, // old index
    Binding::StorageRw, // old islands
    Binding::Storage,   // the cascade's island copy
    Binding::StorageRw, // old island copy
];

const LOG_BINDINGS: [Binding; 11] = [
    Binding::Uniform,   // params
    Binding::Storage,   // control
    Binding::Storage,   // parents
    Binding::Storage,   // work
    Binding::Storage,   // children
    Binding::Storage,   // slots
    Binding::Storage,   // extras
    Binding::Storage,   // engine bodies
    Binding::Storage,   // engine colliders
    Binding::StorageRw, // log
    Binding::Storage,   // cascade status
];

/// The frame's split log (`passlog.slang`): its capacity in words, and how much of it the
/// frame's one read takes (the rest, if any, costs a read of its own).
pub const LOG_WORDS: usize = 1 << 20;
pub const LOG_READ_WORDS: usize = 1 << 16;

/// Offsets (bytes) of `args`: one group per parent; per child slot in groups of 64; per
/// parent in groups of 64; `pass_whole`.
const ARGS_PARENTS: u64 = 0;
const ARGS_SLOTS: u64 = 16;
const ARGS_PARENTS_64: u64 = 32;
const ARGS_WHOLE: u64 = 48;

/// The tables a split pass works on (the solver's), and the engine's.
pub struct PassTables<'a> {
    /// Chunk statics, index, bond statics, bond records, islands, state.
    pub buffers: [&'a wgpu::Buffer; 6],
    pub scratch: &'a wgpu::Buffer,
    pub gate: &'a wgpu::Buffer,
    /// The resident tables (ctab) and their bases: props, eloads, pulses, pulse counts,
    /// pulse capacity, engine entries.
    pub resident: (&'a wgpu::Buffer, [u32; 6]),
    /// The cascade's status record, its island copy and its dispatch arguments.
    pub status: &'a wgpu::Buffer,
    pub island_copy: &'a wgpu::Buffer,
    pub island_args: &'a wgpu::Buffer,
    pub target: SpawnTarget<'a>,
    pub max_colliders: u32,
    pub table: &'a PartTable,
    pub colliders: Option<(&'a crate::colliders::ColliderRefresh, &'a wgpu::Buffer)>,
}

/// A dispatch's size: direct, or from the pass's arguments at a byte offset.
enum Size {
    Direct(u32),
    Indirect(u64),
}

/// A split pass made ready (`GpuPass::prepare`): its dispatches, recorded in one compute
/// pass per cascade pass of a frame.
pub struct PreparedPass {
    steps: Vec<(wgpu::ComputePipeline, wgpu::BindGroup, Size)>,
    args: wgpu::Buffer,
}

impl PreparedPass {
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        for (pipeline, bind, size) in &self.steps {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            match size {
                Size::Direct(n) => pass.dispatch_workgroups(*n, 1, 1),
                Size::Indirect(at) => pass.dispatch_workgroups_indirect(&self.args, *at),
            }
        }
    }
}

pub struct GpuPass {
    parents_k: Kernel,
    snapshot_k: Kernel,
    whole_k: Kernel,
    finish_k: Kernel,
    log_k: Kernel,
    /// The frame's split log (`passlog.slang`).
    pub log: wgpu::Buffer,
    pub parents: wgpu::Buffer,
    pub work: wgpu::Buffer,
    pub out: wgpu::Buffer,
    frames: wgpu::Buffer,
    parent_bodies: wgpu::Buffer,
    pub slots: wgpu::Buffer,
    pub extras: wgpu::Buffer,
    pub control: wgpu::Buffer,
    firsts: wgpu::Buffer,
    args: wgpu::Buffer,
    plan: wgpu::Buffer,
    old: Vec<wgpu::Buffer>,
    /// Live islands, engine bodies, engine colliders, the next cluster id.
    pub counters: wgpu::Buffer,
    island_cap: usize,
    chunk_cap: usize,
}

impl GpuPass {
    /// For tables of `island_cap` islands, `chunks` chunks and `bonds` bonds; `old` gives
    /// each table the layout copies (chunk statics, state, bond statics, bond records,
    /// index, islands, ctab, the cascade's island copy) to size them.
    pub fn new(gpu: &Gpu, island_cap: usize, chunks: usize, bonds: usize, sizes: [u64; 8]) -> GpuPass {
        let k = |e: &str| gpu.compute(&shaders::PASS, e, &PASS_BINDINGS);
        let buf = |label: &str, bytes: u64, extra: wgpu::BufferUsages| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes.max(16), usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | extra, mapped_at_creation: false })
        };
        let n = chunks.max(1) as u64;
        let cap = island_cap as u64 + 1;
        let none = wgpu::BufferUsages::empty();
        let plan_words = 16 + (cap + n) + 6 * n + cap + bonds as u64 + 1;
        GpuPass {
            parents_k: k("pass_parents"),
            snapshot_k: k("pass_snapshot"),
            whole_k: k("pass_whole"),
            finish_k: k("pass_finish"),
            log_k: gpu.compute(&shaders::PASSLOG, "pass_log", &LOG_BINDINGS),
            log: buf("gpu pass log", 4 * LOG_WORDS as u64, none),
            parents: buf("gpu pass parents", 16 * (cap + 1), none),
            work: buf("gpu pass work", 4 * 2 * n, none),
            out: buf("gpu pass out", 16 * CHILD_WORDS as u64 * n, none),
            frames: buf("gpu pass frames", 16 * 5 * cap, none),
            parent_bodies: buf("gpu pass parent bodies", 16 * 2 * cap, none),
            slots: buf("gpu pass slots", 16 * n, none),
            extras: buf("gpu pass extras", 32 * n, none),
            control: buf("gpu pass control", 4 * CTL_WORDS as u64, none),
            firsts: buf("gpu pass parents' records", 16 * cap, none),
            args: buf("gpu pass arguments", 64, wgpu::BufferUsages::INDIRECT),
            plan: buf("gpu pass plan", 4 * plan_words, none),
            old: sizes.iter().map(|&b| buf("gpu pass old", b, none)).collect(),
            counters: buf("gpu pass counters", 16, none),
            island_cap,
            chunk_cap: chunks,
        }
    }

    pub fn fits(&self, island_cap: usize, chunks: usize, sizes: [u64; 8]) -> bool {
        self.island_cap == island_cap && self.chunk_cap == chunks && self.old.iter().zip(sizes).all(|(b, s)| b.size() >= s)
    }

    /// The split pass after a cascade pass, into `encoder`: a no-op unless that pass
    /// disconnected an island. `resume`: the cascade's next pass runs after the split.
    /// The log emptied, for a frame's splits (before its submission).
    pub fn reset_log(&self, gpu: &Gpu) {
        gpu.queue.write_buffer(&self.log, 0, bytemuck::cast_slice(&[16u32, 0, 0, 0]));
    }

    /// `log`: the split appended to the frame's log (`passlog.slang`).
    pub fn encode(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, splitter: &GpuSplitter, layout: &GpuLayout, t: &PassTables, lp: LayoutParams, resume: bool, log: bool) {
        self.prepare(gpu, splitter, layout, t, lp, resume, log).record(encoder);
    }

    /// The split pass's dispatches and bind groups, made once for a frame's passes
    /// (`PreparedPass::record` records them, in one compute pass).
    pub fn prepare(&self, gpu: &Gpu, splitter: &GpuSplitter, layout: &GpuLayout, t: &PassTables, mut lp: LayoutParams, resume: bool, log: bool) -> PreparedPass {
        let [chunks, index, bonds, bond_dyn, islands, state] = t.buffers;
        let (ctab, bases) = t.resident;
        lp.resident = 1;
        [lp.props_base, lp.eload_base, lp.pulse_base, lp.pulse_count_base, lp.pulse_cap, lp.engine_base] = bases;
        let cap = self.island_cap as u32;
        let pp = PassParams {
            chunk_words: lp.chunk_words,
            bond_words: lp.bond_words,
            dyn_words: lp.dyn_words,
            island_words: lp.island_words,
            props_base: lp.props_base,
            eload_base: lp.eload_base,
            pulse_base: lp.pulse_base,
            pulse_count_base: lp.pulse_count_base,
            pulse_cap: lp.pulse_cap,
            engine_base: lp.engine_base,
            resident: 1,
            resume: resume as u32,
            island_float4s: lp.island_words * (cap + 1),
            index_words: (index.size() / 4) as u32,
            engine_float4s: (3 * (cap + 1)).min((ctab.size() / 16) as u32 - lp.engine_base),
            copy_float4s: ((t.island_copy.size() / 16) as u32).min(lp.island_words * (cap + 1)),
        };
        let pass_params = gpu.uniform("gpu pass params", &pp);
        let o = &self.old;
        // Only `pass_parents` writes the arguments; the others dispatch from them.
        let no_args = gpu.storage("gpu pass no arguments", &[[0u32; 4]; 4]);
        let pass_bind_with = |k: &Kernel, args: &wgpu::Buffer| {
            gpu.bind(
                &k.group,
                &[&pass_params, t.status, islands, ctab, &self.control, &self.parents, &self.firsts, &self.parent_bodies, args, &self.counters, chunks, &o[0], state, &o[1], bonds, &o[2], bond_dyn, &o[3], &o[6], t.island_args, index, &o[4], &o[5], t.island_copy, &o[7]],
            )
        };
        let pass_bind = |k: &Kernel| pass_bind_with(k, &no_args);
        let split_params = gpu.uniform("gpu pass split params", &SplitParams { parent_count: 0, out_base: 0, anchored_pose: crate::gpu_split::anchored_pose(), pad: 0 });
        let split_bind = |k: &Kernel| gpu.bind(&k.group, &[&split_params, chunks, index, bonds, bond_dyn, islands, state, &self.parents, &self.work, &self.out, &self.frames, t.target.bodies, &self.parent_bodies, &self.control]);
        let spawn_params = gpu.uniform(
            "gpu pass spawn params",
            &SpawnParams { parent_count: 0, first_body: 0, first_collider: 0, live: 1, max_bodies: t.target.max_bodies, max_colliders: t.max_colliders, island_cap: cap, pad: 0 },
        );
        let spawn_bind = |k: &Kernel| {
            gpu.bind(&k.group, &[&spawn_params, &self.parents, &self.work, &self.out, &self.frames, &t.table.ranges, &t.table.parts, chunks, state, &self.slots, t.target.bodies, t.target.colliders, &self.firsts, &self.extras, &t.table.structure_base, &self.control])
        };
        let layout_uniform = gpu.uniform("gpu pass layout params", &lp);
        let layout_bind = |k: &Kernel| {
            gpu.bind(
                &k.group,
                &[&layout_uniform, &self.parents, &self.work, &self.out, &self.frames, &self.plan, &o[0], chunks, &o[1], state, &o[2], bonds, &o[3], bond_dyn, &o[4], index, &o[5], islands, t.scratch, &o[6], ctab, t.gate, &self.slots, &self.extras, &self.control, &o[7], t.island_copy],
            )
        };
        let steps: Vec<(&Kernel, wgpu::BindGroup, Size)> = vec![
            (&self.parents_k, pass_bind_with(&self.parents_k, &self.args), Size::Direct(1)),
            (&self.snapshot_k, pass_bind(&self.snapshot_k), Size::Indirect(ARGS_PARENTS)),
            (&self.whole_k, pass_bind(&self.whole_k), Size::Indirect(ARGS_WHOLE)),
            (&splitter.frames, split_bind(&splitter.frames), Size::Indirect(ARGS_PARENTS_64)),
            (&splitter.components, split_bind(&splitter.components), Size::Indirect(ARGS_PARENTS)),
            (&splitter.children, split_bind(&splitter.children), Size::Indirect(ARGS_SLOTS)),
            (&splitter.offsets, spawn_bind(&splitter.offsets), Size::Direct(1)),
            (&splitter.spawn, spawn_bind(&splitter.spawn), Size::Indirect(ARGS_SLOTS)),
            (&splitter.retire, spawn_bind(&splitter.retire), Size::Indirect(ARGS_PARENTS_64)),
            (&layout.plan, layout_bind(&layout.plan), Size::Direct(1)),
            (&layout.moved, layout_bind(&layout.moved), Size::Direct(groups(self.island_cap + 1, 64))),
            (&layout.parent, layout_bind(&layout.parent), Size::Indirect(ARGS_PARENTS)),
            (&layout.copy, layout_bind(&layout.copy), Size::Direct(groups(self.island_cap + 1, 64))),
        ];
        let mut prepared: Vec<(wgpu::ComputePipeline, wgpu::BindGroup, Size)> = steps.into_iter().map(|(k, b, size)| (k.pipeline.clone(), b, size)).collect();
        if let Some((refresh, collider_info)) = t.colliders {
            let inputs = crate::colliders::RefreshInputs {
                parent_count: 0,
                chunk_count: lp.chunk_count,
                parents: &self.parents,
                work: &self.work,
                children: &self.out,
                slots: &self.slots,
                part_ranges: &t.table.ranges,
                old_chunks: &o[0],
                chunks,
                islands,
                islands_of: &self.firsts,
                structure_base: &t.table.structure_base,
                control: &self.control,
            };
            prepared.extend(refresh.steps(gpu, &inputs, self.chunk_cap.max(self.island_cap), collider_info).into_iter().map(|(p, b, n)| (p, b, Size::Direct(n))));
        }
        if log {
            let lp = gpu.uniform("gpu pass log params", &[LOG_WORDS as u32, 0, 0, 0]);
            let bind = gpu.bind(&self.log_k.group, &[&lp, &self.control, &self.parents, &self.work, &self.out, &self.slots, &self.extras, t.target.bodies, t.target.colliders, &self.log, t.status]);
            prepared.push((self.log_k.pipeline.clone(), bind, Size::Direct(1)));
        }
        prepared.push((self.finish_k.pipeline.clone(), pass_bind(&self.finish_k), Size::Direct(1)));
        PreparedPass { steps: prepared, args: self.args.clone() }
    }

    /// What the pass did, read back (two reads: its control words, then what they size):
    /// as `GpuSplitter::pass` gives it.
    pub fn read_result(&self, gpu: &Gpu, target: &SpawnTarget) -> (PassResult, Vec<usize>) {
        let control: Vec<u32> = gpu.read_range(&self.control, 0, CTL_WORDS);
        let (n, total) = (control[1] as usize, control[6] as usize);
        let (first_body, first_collider) = (control[4], control[5]);
        let (bodies_after, colliders_after) = (control[8], control[9]);
        let aborted = control[0] != 0;
        let body_words = 112u64;
        let col_words = 64u64;
        let nb = bodies_after.saturating_sub(first_body) as u64;
        let nc = colliders_after.saturating_sub(first_collider) as u64;
        let read = gpu.read_ranges(&[
            (&self.parents, 0, 16 * (n as u64 + 1)),
            (&self.work, 4 * total as u64, 4 * total as u64),
            (&self.out, 0, 16 * (CHILD_WORDS * total) as u64),
            (&self.slots, 0, 16 * total as u64),
            (&self.extras, 0, 32 * total as u64),
            (target.bodies, body_words * first_body as u64, body_words * if aborted { 0 } else { nb }),
            (target.colliders, col_words * first_collider as u64, col_words * if aborted { 0 } else { nc }),
        ]);
        let table: Vec<[u32; 4]> = bytemuck::pod_collect_to_vec(&read[0]);
        let members: Vec<u32> = bytemuck::pod_collect_to_vec(&read[1]);
        let out: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&read[2]);
        let slots: Vec<[u32; 4]> = bytemuck::pod_collect_to_vec(&read[3]);
        let extras: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&read[4]);
        let new_bodies: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&read[5]);
        let new_cols: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&read[6]);
        let parent_islands: Vec<usize> = (0..n).map(|p| table[p][0] as usize).collect();
        // The parents' chunk ranges: their first chunks (`islands_of`) are not read; a
        // child's chunks are given relative to its parent's first chunk, so the caller adds it.
        let children: Vec<Vec<GpuChild>> = (0..n)
            .map(|p| {
                let base = table[p][1] as usize;
                (0..table[p][2] as usize)
                    .map(|k| {
                        let rec = &out[CHILD_WORDS * (base + k)..CHILD_WORDS * (base + k + 1)];
                        let (offset, count) = (rec[0][1].to_bits() as usize, rec[0][2].to_bits() as usize);
                        GpuChild {
                            chunks: (0..count).map(|j| members[base + offset + j]).collect(),
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
            for p in 0..n {
                for k in 0..table[p][2] as usize {
                    let slot = table[p][1] as usize + k;
                    let s = slots[slot];
                    if s[3] == 0 {
                        continue;
                    }
                    let b = 7 * (s[0] - first_body) as usize;
                    let mut body = [[0f32; 4]; 7];
                    body.copy_from_slice(&new_bodies[b..b + 7]);
                    let cols = (0..s[2] as usize)
                        .map(|j| {
                            let c = 4 * ((s[1] - first_collider) as usize + j);
                            [new_cols[c], new_cols[c + 1], new_cols[c + 2], new_cols[c + 3]]
                        })
                        .collect();
                    bodies.push(GpuBody { slot: s[0], first_collider: s[1], body, colliders: cols, com: extras[2 * slot], principal: extras[2 * slot + 1] });
                }
            }
        }
        (PassResult { aborted, children, bodies }, parent_islands)
    }
}
