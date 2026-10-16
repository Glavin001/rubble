//! Metal executor of the fracture lab's host plan.
//!
//! `engine_fracture::Planner` owns the buffer sizes, host words, step state and
//! the ordered dispatch schedule; this host creates the buffers and pipelines
//! (Slang -> MSL with the browser's `ENGINE_WGSL` layout, see build.rs) and
//! replays each tick's commands, exactly as `fracture/simulation.mjs` does on
//! WebGPU. All storage is shared (unified memory on Apple silicon), so
//! counters and bodies are read in place.

use super::kernels::{Kernel, KERNELS};
use engine_fracture::plan::{
    self, Command, Counters, Limits, Planner, BODY_BYTES, HOST_UPLOAD, PASS_ALIGN, UPLOAD_BYTES,
};
use engine_fracture::Scene;
use metal::*;
use std::time::Instant;

/// Host uploads per batch (<= 4 ticks): one region per tick, reused after the wait.
const RING: usize = 4;

#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    pub wall_ms: f64,
    pub encode_ms: f64,
    pub wait_ms: f64,
}

pub struct FractureSim {
    queue: CommandQueue,
    pipes: Vec<ComputePipelineState>,
    pub bufs: Vec<Buffer>,
    upload: Buffer,
    pub plan: Planner,
    pub last: Counters,
    pub timing: Timing,
    in_flight: Option<CommandBuffer>,
    batch: (Instant, u32),
}

fn shared(device: &Device, size: u64) -> Buffer {
    let buf = device.new_buffer(size.max(4), MTLResourceOptions::StorageModeShared);
    unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, size as usize) };
    buf
}

fn write(buf: &Buffer, offset: u64, bytes: &[u8]) {
    assert!(offset as usize + bytes.len() <= buf.length() as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            (buf.contents() as *mut u8).add(offset as usize),
            bytes.len(),
        )
    };
}

fn read_words(buf: &Buffer, offset: usize, count: usize) -> Vec<u32> {
    unsafe {
        std::slice::from_raw_parts(
            (buf.contents() as *const u8).add(offset) as *const u32,
            count,
        )
        .to_vec()
    }
}

impl FractureSim {
    pub fn new(device: &Device, queue: &CommandQueue, scene: &Scene) -> anyhow::Result<Self> {
        let max = device.max_buffer_length();
        let plan = Planner::new(
            scene,
            Limits {
                max_binding: max,
                max_buffer: max,
            },
        )?;
        let bufs: Vec<Buffer> = plan
            .buffer_sizes_bytes()
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                let b = shared(device, size);
                b.set_label(&format!("AVBD {i}"));
                b
            })
            .collect();
        let compile = CompileOptions::new();
        // IEEE math like the WebGPU path, so contacts stay comparable to the browser lab.
        compile.set_fast_math_enabled(false);
        let pipes = KERNELS
            .iter()
            .map(|k| {
                let lib = device
                    .new_library_with_source(k.msl, &compile)
                    .map_err(|e| anyhow::anyhow!("compile {}: {e}", k.entry))?;
                let func = lib
                    .get_function(k.entry, None)
                    .map_err(|e| anyhow::anyhow!("function {}: {e}", k.entry))?;
                device
                    .new_compute_pipeline_state_with_function(&func)
                    .map_err(|e| anyhow::anyhow!("pipeline {}: {e}", k.entry))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let sim = Self {
            queue: queue.clone(),
            pipes,
            upload: shared(device, (UPLOAD_BYTES * RING) as u64),
            last: plan.last,
            plan,
            bufs,
            timing: Timing::default(),
            in_flight: None,
            batch: (Instant::now(), 1),
        };
        write(
            &sim.bufs[plan::PASS_BUFFER as usize],
            0,
            &Planner::pass_table(),
        );
        write(
            &sim.bufs[plan::MATERIAL_BUFFER as usize],
            0,
            &sim.plan.material_table(),
        );
        let hull: Vec<u8> = scene
            .hull_data
            .iter()
            .flatten()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        write(&sim.bufs[plan::HULL_BUFFER as usize], 0, &hull);
        sim.upload_initial();
        Ok(sim)
    }

    fn upload_initial(&self) {
        write(
            &self.bufs[plan::BODY_BUFFER as usize],
            0,
            self.plan.body_bytes(),
        );
        write(
            &self.bufs[plan::COLLIDER_BUFFER as usize],
            0,
            self.plan.collider_bytes(),
        );
        write(
            &self.bufs[plan::JOINT_BUFFER as usize],
            0,
            self.plan.joint_bytes(),
        );
    }

    pub fn body_buffer(&self) -> &Buffer {
        &self.bufs[plan::BODY_BUFFER as usize]
    }
    pub fn steps(&self) -> u32 {
        self.plan.steps
    }
    pub fn shots(&self) -> u32 {
        self.plan.shots
    }
    pub fn awake(&self) -> bool {
        !self.plan.initialized || self.last.awake > 0
    }

    /// Replay one command list into `cb`: copies in a blit pass, then the
    /// dispatches in one serial compute pass (each sees the previous one's writes).
    fn replay(&self, cb: &CommandBufferRef, words: &[u32], ring: usize) {
        let base = (ring * UPLOAD_BYTES) as u64;
        write(&self.upload, base, self.plan.host_upload());
        let commands = Command::decode(words);
        let blit = cb.new_blit_command_encoder();
        for c in &commands {
            if let Command::Copy {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } = *c
            {
                let (from, offset) = if src == HOST_UPLOAD {
                    (&self.upload, base + src_offset as u64)
                } else {
                    (&self.bufs[src as usize], src_offset as u64)
                };
                blit.copy_from_buffer(
                    from,
                    offset,
                    &self.bufs[dst as usize],
                    dst_offset as u64,
                    size as u64,
                );
            }
        }
        blit.end_encoding();
        let desc = ComputePassDescriptor::new();
        desc.set_dispatch_type(MTLDispatchType::Serial);
        let e = cb.compute_command_encoder_with_descriptor(desc);
        for c in &commands {
            match *c {
                Command::Dispatch { kernel, x, y, pass } => {
                    let k = self.bind(e, kernel, pass, false);
                    e.dispatch_thread_groups(MTLSize::new(x as u64, y as u64, 1), threads(k));
                }
                Command::Indirect { kernel, slot, pass } => {
                    let k = self.bind(e, kernel, pass, true);
                    e.dispatch_thread_groups_indirect(
                        &self.bufs[plan::ARGS_BUFFER as usize],
                        slot as u64 * 12,
                        threads(k),
                    );
                }
                _ => {}
            }
        }
        e.end_encoding();
    }

    fn bind(
        &self,
        e: &ComputeCommandEncoderRef,
        kernel: u32,
        pass: u32,
        indirect: bool,
    ) -> &'static Kernel {
        let k = &KERNELS[kernel as usize];
        e.set_compute_pipeline_state(&self.pipes[kernel as usize]);
        for &(index, slot) in k.binds {
            if slot == plan::PASS_BUFFER {
                e.set_buffer(
                    index as u64,
                    Some(&self.bufs[slot as usize]),
                    (pass * PASS_ALIGN) as u64,
                );
            } else if slot == plan::ARGS_BUFFER && indirect {
                // Never bind the indirect buffer as a resource of the dispatch it sizes.
                e.set_buffer(index as u64, None, 0);
            } else {
                e.set_buffer(index as u64, Some(&self.bufs[slot as usize]), 0);
            }
        }
        k
    }

    /// Submit `count` (1-4) ticks without waiting. Fully sleeping scenes skip GPU work.
    pub fn submit(&mut self, count: u32) -> anyhow::Result<bool> {
        anyhow::ensure!(
            (1..=RING as u32).contains(&count),
            "Physics batch must have 1–4 ticks"
        );
        self.wait()?;
        if self.plan.skip_asleep(count) {
            self.timing = Timing::default();
            return Ok(false);
        }
        let start = Instant::now();
        let cb = self.queue.new_command_buffer().to_owned();
        cb.set_label("fracture physics");
        for i in 0..count as usize {
            let words = self.plan.encode_step(false).to_vec();
            self.replay(&cb, &words, i);
        }
        cb.commit();
        self.plan.mark_submitted();
        let encode = start.elapsed().as_secs_f64() * 1000.0 / count as f64;
        self.timing = Timing {
            wall_ms: encode,
            encode_ms: encode,
            wait_ms: 0.0,
        };
        self.in_flight = Some(cb);
        self.batch = (start, count);
        Ok(true)
    }

    /// Wait for the in-flight batch, then read the counters in place.
    pub fn wait(&mut self) -> anyhow::Result<()> {
        let Some(cb) = self.in_flight.take() else {
            return Ok(());
        };
        let waited = Instant::now();
        let token = self.plan.begin_read();
        cb.wait_until_completed();
        let (start, count) = self.batch;
        let end = Instant::now();
        anyhow::ensure!(
            cb.status() != MTLCommandBufferStatus::Error,
            "Metal command buffer failed"
        );
        self.last = self.plan.finish_read(
            token,
            &read_words(&self.bufs[plan::COUNTERS_BUFFER as usize], 0, 20),
        );
        // Wall includes queue execution; not isolated GPU time.
        let n = count as f64;
        self.timing.wall_ms = (end - start).as_secs_f64() * 1000.0 / n;
        self.timing.wait_ms = (end - waited).as_secs_f64() * 1000.0 / n;
        self.plan.overflow_error()
    }

    /// Step `count` ticks one at a time, waiting for each (tests, probes).
    pub fn step(&mut self, count: u32) -> anyhow::Result<()> {
        for _ in 0..count {
            self.submit(1)?;
            self.wait()?;
        }
        Ok(())
    }

    /// Fire the next projectile along a ray (the plan aims and sizes it).
    pub fn launch(&mut self, origin: [f64; 3], direction: [f64; 3]) -> anyhow::Result<u32> {
        self.wait()?;
        let (id, bytes) = self.plan.launch(origin, direction, None, None)?;
        for o in self.plan.body_offsets(id) {
            write(self.body_buffer(), o, &bytes);
        }
        self.last = self.plan.last;
        Ok(id)
    }

    /// Clear transient state and restore bodies, colliders, joints and ammunition.
    pub fn reset(&mut self) {
        let _ = self.wait();
        for (i, buf) in self.bufs.iter().enumerate() {
            if Planner::cleared_on_reset(i as u32) {
                unsafe {
                    std::ptr::write_bytes(buf.contents() as *mut u8, 0, buf.length() as usize)
                };
            }
        }
        self.upload_initial();
        self.plan.reset();
        self.last = self.plan.last;
    }

    /// Run only the collision kernels at the current poses (port of
    /// `inspectCollisions()`); returns (counters, contact metas: 8 words each).
    pub fn inspect_collisions(&mut self) -> anyhow::Result<(Counters, Vec<[u32; 8]>)> {
        self.wait()?;
        let words = self
            .plan
            .encode_inspect(0, self.plan.awake_queries)
            .to_vec();
        let cb = self.queue.new_command_buffer();
        let token = self.plan.begin_read();
        self.replay(cb, &words, 0);
        cb.commit();
        cb.wait_until_completed();
        self.plan.after_inspect();
        let counters = self.plan.count_read(
            token,
            &read_words(&self.bufs[plan::COUNTERS_BUFFER as usize], 0, 20),
        );
        anyhow::ensure!(
            counters.overflow == 0,
            "Collision overflow {}",
            counters.overflow
        );
        let base = self.plan.contact_base(false) as usize;
        let metas = read_words(&self.bufs[4], base * 32, counters.contact as usize * 8);
        Ok((
            counters,
            metas.chunks(8).map(|c| c.try_into().unwrap()).collect(),
        ))
    }

    /// Copy of the first `count` 112-byte bodies as f32 words (28 per body).
    pub fn snapshot(&mut self, count: usize) -> anyhow::Result<Vec<f32>> {
        self.wait()?;
        let words = read_words(self.body_buffer(), 0, count * BODY_BYTES / 4);
        Ok(words.into_iter().map(f32::from_bits).collect())
    }
}

fn threads(k: &Kernel) -> MTLSize {
    MTLSize::new(
        k.threads[0] as u64,
        k.threads[1] as u64,
        k.threads[2] as u64,
    )
}
