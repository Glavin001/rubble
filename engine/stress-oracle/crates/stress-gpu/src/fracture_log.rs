//! The GPU fracture's change log (`shaders/fracture_log.slang`): each frame that broke a bond
//! appends what changed (broken bonds, chunks that changed body, body slots whose frame
//! changed), stamped with its step, to a ring the host acknowledges. A host reads it instead of
//! every chunk's body, bond and frame; entries wait until acknowledged, so one that skipped
//! frames' readbacks still finds every change in step order, and an overflow (more unread
//! changes than a section holds) is flagged for one full read.

use crate::gpu::{Binding, Gpu, Kernel};
use crate::shaders;

const HEADER: u32 = 8;
const BOND_WORDS: u32 = 2;
const CHUNK_WORDS: u32 = 3;
const FRAME_WORDS: u32 = 10;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct LogParams {
    chunk_count: u32,
    bond_count: u32,
    body_count: u32,
    step: u32,
    w_alive: u32,
    cap_bonds: u32,
    cap_chunks: u32,
    cap_frames: u32,
}

const BINDINGS: [Binding; 5] = [
    Binding::Uniform,
    Binding::Storage,
    Binding::Storage,
    Binding::StorageRw,
    Binding::StorageRw,
];

/// How far a host has read each section (bonds, chunks, frames): entry counts since the start.
pub type LogCursor = [u32; 3];

/// The changes in a log copy since a cursor, in step order (within a step, by index).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LogEntries {
    /// (bond, step).
    pub bonds: Vec<(u32, u32)>,
    /// (chunk, new body, step).
    pub chunks: Vec<(u32, u32, u32)>,
    /// (body slot, step, frame).
    pub frames: Vec<(u32, u32, crate::fracture::FrameRecord)>,
    /// Some entries were not kept: read the state in full.
    pub overflow: bool,
    /// Where these end (the cursor to acknowledge).
    pub cursor: LogCursor,
}

pub struct FractureLog {
    params: LogParams,
    params_buf: wgpu::Buffer,
    pub buffer: wgpu::Buffer,
    kernels: [Kernel; 3],
    bind: wgpu::BindGroup,
}

impl FractureLog {
    /// A log over `work` and `frames` (`fracture.slang`'s), starting from the state the
    /// tables describe: every bond intact, each chunk on `chunk_body`, each slot at `frames0`.
    pub fn new(
        gpu: &Gpu,
        layout: crate::fracture::Layout,
        work: &wgpu::Buffer,
        frames: &wgpu::Buffer,
        chunk_body: &[u32],
        frames0: &[crate::fracture::FrameRecord],
    ) -> FractureLog {
        let (n, e, b) = (layout.n, layout.e, layout.b);
        let params = LogParams {
            chunk_count: n,
            bond_count: e,
            body_count: b,
            step: 0,
            w_alive: layout.w_alive(),
            cap_bonds: e.clamp(1, 2048),
            cap_chunks: n.clamp(1, 2048),
            cap_frames: b.clamp(1, 512),
        };
        let mut track: Vec<u32> = vec![1; e as usize];
        track.extend_from_slice(chunk_body);
        for f in frames0 {
            for v in f {
                track.extend(v.iter().map(|x| x.to_bits()));
            }
        }
        let words = HEADER
            + BOND_WORDS * params.cap_bonds
            + CHUNK_WORDS * params.cap_chunks
            + FRAME_WORDS * params.cap_frames;
        let params_buf = gpu.uniform("fracture log params", &params);
        let track = gpu.storage("fracture log track", &track);
        let buffer = gpu.storage("fracture log", &vec![0u32; words as usize]);
        let kernels = ["log_bonds", "log_chunks", "log_frames"]
            .map(|k| gpu.compute(&shaders::FRACTURE_LOG, k, &BINDINGS));
        let bind = gpu.bind(
            &kernels[0].group,
            &[&params_buf, work, frames, &track, &buffer],
        );
        FractureLog {
            params,
            params_buf,
            buffer,
            kernels,
            bind,
        }
    }

    /// The log's bytes (one readback copy of the whole ring).
    pub fn bytes(&self) -> u64 {
        self.buffer.size()
    }

    pub fn set_step(&mut self, gpu: &Gpu, step: u32) {
        if self.params.step != step {
            self.params.step = step;
            gpu.queue
                .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        }
    }

    /// Record the frame's log after its split, sized by the fracture's gate (`args`: bonds,
    /// chunks, bodies at `arg` words 3 apart).
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        args: &wgpu::Buffer,
        arg_bonds: u64,
        arg_chunks: u64,
        arg_bodies: u64,
    ) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        for (k, arg) in self.kernels.iter().zip([arg_bonds, arg_chunks, arg_bodies]) {
            pass.set_pipeline(&k.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups_indirect(args, 12 * arg);
        }
    }

    /// The changes in a copy of the log (`bytes`) since `read`.
    pub fn decode(&self, bytes: &[u8], read: LogCursor) -> LogEntries {
        let w: &[u32] = bytemuck::cast_slice(bytes);
        let p = &self.params;
        let cursor = [w[0], w[1], w[2]];
        let caps = [p.cap_bonds, p.cap_chunks, p.cap_frames];
        let mut out = LogEntries {
            overflow: w[3] != 0,
            cursor,
            ..Default::default()
        };
        let bases = [
            HEADER,
            HEADER + BOND_WORDS * p.cap_bonds,
            HEADER + BOND_WORDS * p.cap_bonds + CHUNK_WORDS * p.cap_chunks,
        ];
        for s in 0..3 {
            let unread = cursor[s].wrapping_sub(read[s]);
            if unread > caps[s] {
                out.overflow = true;
            }
            for i in 0..unread.min(caps[s]) {
                let at = (read[s].wrapping_add(i) % caps[s]) as usize;
                match s {
                    0 => {
                        let e = bases[0] as usize + BOND_WORDS as usize * at;
                        out.bonds.push((w[e], w[e + 1]));
                    }
                    1 => {
                        let e = bases[1] as usize + CHUNK_WORDS as usize * at;
                        out.chunks.push((w[e], w[e + 1], w[e + 2]));
                    }
                    _ => {
                        let e = bases[2] as usize + FRAME_WORDS as usize * at;
                        let f = |k: usize| [0, 1, 2, 3].map(|c| f32::from_bits(w[e + k + c]));
                        out.frames.push((w[e], w[e + 1], [f(2), f(6)]));
                    }
                }
            }
        }
        out.bonds.sort_unstable_by_key(|&(j, s)| (s, j));
        out.chunks.sort_unstable_by_key(|&(c, _, s)| (s, c));
        out.frames.sort_unstable_by_key(|&(b, s, _)| (s, b));
        out
    }

    /// Acknowledge everything up to `cursor` (and clear an overflow the host has handled with a
    /// full read): the GPU may reuse those entries.
    pub fn acknowledge(&self, gpu: &Gpu, cursor: LogCursor, clear_overflow: bool) {
        gpu.queue
            .write_buffer(&self.buffer, 16, bytemuck::cast_slice(&cursor));
        if clear_overflow {
            gpu.queue
                .write_buffer(&self.buffer, 12, bytemuck::bytes_of(&0u32));
        }
    }
}
