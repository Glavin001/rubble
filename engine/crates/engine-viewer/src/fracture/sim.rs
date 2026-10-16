//! Native Metal host for the fracture demo: the same Slang AVBD + convex kernels and
//! dispatch order as `engine-webgpu/fracture/simulation.mjs`, compiled Slang → MSL.
//!
//! Buffers use the browser's `BIND(slot)` numbering (0–27). All storage is shared
//! (unified memory on Apple silicon), so counters and bodies are read in place.

use super::kernels::{Kernel, KERNELS};
use engine_fracture::{collider_bounds, FractureScene};
use metal::*;
use std::collections::HashMap;
use std::time::Instant;

const MAX_WARM_HASH: u32 = 1 << 21;
const SLOTS: usize = 28;
const HOST_SLOT: usize = 6;
const PASS_SLOT: usize = 22;
const ARGS_SLOT: usize = 23;
/// Host uploads per batch (≤ 4 ticks) — one region per tick, reused after the wait.
const RING: usize = 4;

fn next_pow2(n: u32) -> u32 {
    n.max(1).next_power_of_two()
}

#[derive(Clone, Copy, Debug)]
pub struct Capacity {
    pub max_bodies: u32,
    pub max_colliders: u32,
    pub max_contacts: u32,
    pub max_candidates: u32,
    pub max_joints: u32,
    pub max_colors: u32,
    pub max_commands: u32,
    pub max_hull: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Counters {
    pub contact: u32,
    pub candidate: u32,
    pub awake: u32,
    pub color: u32,
    pub manifold: u32,
    pub overflow: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Timing {
    pub wall_ms: f64,
    pub encode_ms: f64,
    pub wait_ms: f64,
}

/// AVBD params block (208 bytes host uniform: 20 params + 32 scratch words).
#[derive(Clone, Copy)]
struct Params {
    dt: f32,
    gravity_y: f32,
    iterations: u32,
    flags: u32,
    alpha: f32,
    gamma: f32,
    beta_lin: f32,
    beta_ang: f32,
    k_start: f32,
    k_max_n: f32,
    k_max_t: f32,
    contact_offset: f32,
    sleep_lin: f32,
    sleep_ang: f32,
    sleep_time: f32,
    residual_exit: f32,
    trust_lin: f32,
    trust_ang: f32,
    pivot_floor: f32,
    grid_cell_size: f32,
}

impl Params {
    fn words(&self) -> [u32; 20] {
        let f = |x: f32| x.to_bits();
        [
            f(self.dt),
            f(self.gravity_y),
            self.iterations,
            self.flags,
            f(self.alpha),
            f(self.gamma),
            f(self.beta_lin),
            f(self.beta_ang),
            f(self.k_start),
            f(self.k_max_n),
            f(self.k_max_t),
            f(self.contact_offset),
            f(self.sleep_lin),
            f(self.sleep_ang),
            f(self.sleep_time),
            f(self.residual_exit),
            f(self.trust_lin),
            f(self.trust_ang),
            f(self.pivot_floor),
            f(self.grid_cell_size),
        ]
    }
}

struct Pipe {
    pso: ComputePipelineState,
    kernel: &'static Kernel,
}

pub struct FractureSim {
    device: Device,
    queue: CommandQueue,
    pipes: HashMap<&'static str, Pipe>,
    pub bufs: Vec<Buffer>,
    host_upload: Buffer,
    pub cap: Capacity,
    params: Params,
    csr_base: u32,
    body_bytes: Vec<u8>,
    collider_bytes: Vec<u8>,
    initial_aabbs: Vec<u8>,
    grid_levels: u32,
    planes: u32,
    body_count: u32,
    collider_count: u32,
    projectile_start: u32,
    projectile_count: u32,
    projectile_radius: f32,
    projectile_mass: f32,
    extent: f32,
    center: [f32; 3],
    pub last: Counters,
    pub steps: u32,
    parity: u32,
    pub shots: u32,
    pub timing: Timing,
    initialized: bool,
    in_flight: Option<CommandBuffer>,
    batch: (Instant, u32),
}

impl FractureSim {
    pub fn new(
        device: &Device,
        queue: &CommandQueue,
        scene: &FractureScene,
    ) -> anyhow::Result<Self> {
        let nb = scene.bodies.len() as u32;
        let nc = scene.colliders.len() as u32;
        let cap = Capacity {
            max_bodies: next_pow2(nb.max(64)),
            max_colliders: next_pow2(nc.max(64)),
            max_contacts: next_pow2((nc * 32).max(4096)),
            max_candidates: next_pow2((nc * 64).max(4096)),
            max_joints: 1,
            max_colors: 32,
            max_commands: 1,
            max_hull: (scene.hull_data.len() as u32).max(64),
        };
        let constraint_id_words = next_pow2((2 * (cap.max_contacts + cap.max_joints)).max(32768));
        let csr_base = constraint_id_words.max(131072);
        let p = &scene.params;
        let params = Params {
            dt: p.dt,
            gravity_y: p.gravity_y,
            iterations: p.iterations,
            flags: p.flags,
            alpha: p.alpha,
            gamma: 0.999,
            beta_lin: 10000.0,
            beta_ang: 100.0,
            k_start: 1000.0,
            k_max_n: 1e10,
            k_max_t: 1e6,
            contact_offset: p.contact_offset,
            sleep_lin: p.sleep_lin,
            sleep_ang: p.sleep_ang,
            sleep_time: p.sleep_time,
            residual_exit: 1e-4,
            trust_lin: 0.35,
            trust_ang: 0.5,
            pivot_floor: 1e-6,
            grid_cell_size: scene.grid_cell_size,
        };

        // Sizes follow `bufferSizes()` in simulation.mjs; the hull arena is packed
        // float3 (12-byte stride) on Metal instead of WGSL's 16-byte vec3 stride.
        let b = |n: u32, stride: u64| n as u64 * stride;
        let mut sizes = [256u64; SLOTS];
        sizes[0] = b(cap.max_bodies * 2, 112);
        sizes[1] = b(cap.max_bodies, 112);
        sizes[2] = b(cap.max_colliders, 64);
        sizes[3] = b(64, 16);
        sizes[4] = b(cap.max_contacts * 2, 128);
        sizes[5] = b(cap.max_joints, 256);
        sizes[8] = b(cap.max_commands, 64);
        sizes[9] = b(cap.max_bodies + cap.max_colliders, 32);
        sizes[10] = b(cap.max_candidates, 16);
        sizes[11] = b(cap.max_bodies, 4);
        sizes[12] = b(cap.max_bodies, 4);
        sizes[13] = b(cap.max_bodies + 1, 4);
        sizes[14] = (csr_base + cap.max_bodies + 1) as u64 * 4;
        sizes[15] = b(cap.max_colliders.max(cap.max_bodies * cap.max_colors), 4);
        sizes[16] = b(cap.max_colors, 4);
        sizes[17] = b(cap.max_colors, 4);
        sizes[18] = b(cap.max_hull, 12);
        sizes[19] = 65536 * 4;
        sizes[20] = b(cap.max_bodies, 4);
        sizes[21] = 64 * 4;
        sizes[23] = b(cap.max_colors.max(70) * 3, 4);
        sizes[24] = b(cap.max_colliders, 80);
        sizes[25] = b(cap.max_candidates, 32);
        sizes[26] = b(cap.max_bodies, 32);
        sizes[27] = b(cap.max_candidates, 32);
        let max_len = device.max_buffer_length();
        if let Some(i) = sizes.iter().position(|&s| s > max_len) {
            anyhow::bail!("GPU buffer {i} exceeds this device's limits");
        }
        let opts = MTLResourceOptions::StorageModeShared;
        let bufs: Vec<Buffer> = sizes
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let buf = device.new_buffer(s, opts);
                buf.set_label(&format!("AVBD {i}"));
                unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, s as usize) };
                buf
            })
            .collect();
        let host_upload = device.new_buffer((Self::upload_stride(&cap) * RING) as u64, opts);

        let mut pipes = HashMap::new();
        let compile = CompileOptions::new();
        // IEEE math like the WebGPU path, so contacts stay comparable to the browser demo.
        compile.set_fast_math_enabled(false);
        for kernel in KERNELS {
            let lib = device
                .new_library_with_source(kernel.msl, &compile)
                .map_err(|e| anyhow::anyhow!("compile {}: {e}", kernel.entry))?;
            let func = lib
                .get_function(kernel.entry, None)
                .map_err(|e| anyhow::anyhow!("function {}: {e}", kernel.entry))?;
            let pso = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| anyhow::anyhow!("pipeline {}: {e}", kernel.entry))?;
            pipes.insert(kernel.name, Pipe { pso, kernel });
        }

        let mut body_bytes = vec![0u8; sizes[0] as usize];
        for (i, body) in scene.bodies.iter().enumerate() {
            let bytes = body.to_bytes();
            body_bytes[i * 112..i * 112 + 112].copy_from_slice(&bytes);
            let j = cap.max_bodies as usize + i;
            body_bytes[j * 112..j * 112 + 112].copy_from_slice(&bytes);
        }
        let mut collider_bytes = vec![0u8; sizes[2] as usize];
        for (i, c) in scene.colliders.iter().enumerate() {
            collider_bytes[i * 64..i * 64 + 64].copy_from_slice(&c.to_bytes());
        }
        // Sleeping bodies skip AABB refit, so publish their initial cached bounds.
        let mut initial_aabbs = vec![0u8; sizes[9] as usize];
        for (i, c) in scene.colliders.iter().enumerate() {
            let (lo, hi) = collider_bounds(scene, c);
            let o = (cap.max_bodies as usize + i) * 32;
            let w = if c.params[3] != 0.0 {
                c.params[3]
            } else {
                c.params[0]
            };
            let words = [
                lo[0].to_bits(),
                lo[1].to_bits(),
                lo[2].to_bits(),
                w.to_bits(),
                hi[0].to_bits(),
                hi[1].to_bits(),
                hi[2].to_bits(),
                c.body,
            ];
            for (k, x) in words.iter().enumerate() {
                initial_aabbs[o + k * 4..o + k * 4 + 4].copy_from_slice(&x.to_le_bytes());
            }
        }
        let sim = Self {
            device: device.clone(),
            queue: queue.clone(),
            pipes,
            bufs,
            host_upload,
            cap,
            params,
            csr_base,
            body_bytes,
            collider_bytes,
            initial_aabbs,
            grid_levels: scene.grid_levels,
            planes: 0,
            body_count: nb,
            collider_count: nc,
            projectile_start: scene.projectile_start,
            projectile_count: scene.projectile_count,
            projectile_radius: scene.projectile_radius,
            projectile_mass: scene.projectile_mass,
            extent: scene.extent,
            center: scene.center,
            last: Counters::default(),
            steps: 0,
            parity: 0,
            shots: 0,
            timing: Timing::default(),
            initialized: false,
            in_flight: None,
            batch: (Instant::now(), 1),
        };
        let mut sim = sim;
        // Constant tables: materials (friction 0.5) and the packed hull arena.
        let mut mats = Vec::with_capacity(64 * 4);
        for _ in 0..64 {
            mats.extend([0.5f32, 0.0, 0.01, 0.0]);
        }
        sim.write(3, 0, bytemuck_f32(&mats));
        let hull: Vec<f32> = scene
            .hull_data
            .iter()
            .flat_map(|v| [v[0], v[1], v[2]])
            .collect();
        sim.write(18, 0, bytemuck_f32(&hull));
        let planes: Vec<u32> = scene
            .colliders
            .iter()
            .enumerate()
            .filter(|(_, c)| c.shape == 4)
            .map(|(i, _)| i as u32)
            .collect();
        sim.planes = planes.len() as u32;
        sim.write(19, 0, bytemuck_u32(&planes));
        sim.reset();
        Ok(sim)
    }

    fn upload_stride(cap: &Capacity) -> usize {
        (576 + cap.max_colors as usize * 4).next_multiple_of(256)
    }

    fn write(&self, slot: usize, offset: usize, bytes: &[u8]) {
        let buf = &self.bufs[slot];
        assert!(offset + bytes.len() <= buf.length() as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (buf.contents() as *mut u8).add(offset),
                bytes.len(),
            );
        }
    }

    fn read_u32(&self, slot: usize, word: usize) -> u32 {
        unsafe { *(self.bufs[slot].contents() as *const u32).add(word) }
    }

    pub fn body_buffer(&self) -> &Buffer {
        &self.bufs[0]
    }
    pub fn awake(&self) -> bool {
        !self.initialized || self.last.awake > 0
    }

    /// Clear transient state and restore bodies, colliders, bounds and ammunition.
    pub fn reset(&mut self) {
        let _ = self.wait();
        for (i, buf) in self.bufs.iter().enumerate() {
            if ![0, 2, 3, 18, 19, 22].contains(&i) {
                unsafe {
                    std::ptr::write_bytes(buf.contents() as *mut u8, 0, buf.length() as usize)
                };
            }
        }
        self.write(0, 0, &self.body_bytes.clone());
        self.write(2, 0, &self.collider_bytes.clone());
        self.write(9, 0, &self.initial_aabbs.clone());
        self.steps = 0;
        self.parity = 0;
        self.shots = 0;
        self.initialized = false;
        self.last = Counters {
            awake: self.body_count - 1,
            color: 16,
            ..Default::default()
        };
    }

    fn scratch_words(&self) -> [u32; 64] {
        let mut s = [0u32; 64];
        s[1] = self.cap.max_bodies;
        s[2] = self.cap.max_candidates;
        s[3] = self.cap.max_contacts;
        s[4] = self.cap.max_colors.min(32);
        s[7] = if self.steps == 0 {
            0
        } else {
            self.last.contact.min(self.cap.max_contacts)
        };
        s[8] = if self.parity == 0 {
            self.cap.max_contacts
        } else {
            0
        };
        // One hash entry per manifold; host field and clear dispatch must agree.
        s[9] = next_pow2((self.cap.max_contacts * 4).max(1024))
            .min(self.csr_base)
            .min(MAX_WARM_HASH);
        s[10] = self.cap.max_candidates;
        s[11] = if self.parity == 0 {
            0
        } else {
            self.cap.max_contacts
        };
        let mut log = 0;
        while (1u32 << log) < self.body_count.max(1) && log < 20 {
            log += 1;
        }
        s[12] = if log == 0 { 0 } else { (log - 1) & 1 };
        s[17] = if self.initialized {
            self.steps.max(1)
        } else {
            0
        };
        s[18] = self.grid_levels;
        s[19] = self.planes;
        s[24] = self.csr_base;
        s
    }

    /// Stage host uniforms in a ring region and blit them into the step buffers.
    fn write_host(&self, cb: &CommandBufferRef, ring: usize) {
        let stride = Self::upload_stride(&self.cap);
        let base = ring * stride;
        let s = self.scratch_words();
        let mut host = [0u32; 52];
        host[..20].copy_from_slice(&self.params.words());
        host[20..52].copy_from_slice(&s[..32]);
        let mut counters = [0u32; 16];
        counters[0] = self.body_count;
        counters[1] = self.collider_count;
        let colors = vec![0u32; self.cap.max_colors as usize];
        unsafe {
            let p = (self.host_upload.contents() as *mut u8).add(base);
            std::ptr::copy_nonoverlapping(s.as_ptr() as *const u8, p, 256);
            std::ptr::copy_nonoverlapping(host.as_ptr() as *const u8, p.add(256), 208);
            std::ptr::copy_nonoverlapping(counters.as_ptr() as *const u8, p.add(512), 64);
            std::ptr::copy_nonoverlapping(
                colors.as_ptr() as *const u8,
                p.add(576),
                colors.len() * 4,
            );
        }
        let blit = cb.new_blit_command_encoder();
        let up = &self.host_upload;
        let base = base as u64;
        blit.copy_from_buffer(up, base, &self.bufs[21], 0, 256);
        blit.copy_from_buffer(up, base + 256, &self.bufs[HOST_SLOT], 0, 208);
        // Keep the previous GPU contact count (s[7]) before counters are cleared.
        if self.steps > 0 {
            blit.copy_from_buffer(&self.bufs[7], 8, &self.bufs[HOST_SLOT], 80 + 7 * 4, 4);
        }
        blit.copy_from_buffer(up, base + 512, &self.bufs[7], 0, 64);
        blit.copy_from_buffer(
            up,
            base + 576,
            &self.bufs[17],
            0,
            self.cap.max_colors as u64 * 4,
        );
        blit.end_encoding();
    }

    fn bind(&self, enc: &ComputeCommandEncoderRef, name: &str, pass: u32, indirect: bool) -> &Pipe {
        let pipe = self
            .pipes
            .get(name)
            .unwrap_or_else(|| panic!("missing kernel {name}"));
        enc.set_compute_pipeline_state(&pipe.pso);
        for &(index, slot) in pipe.kernel.binds {
            let (index, slot) = (index as u64, slot as usize);
            if slot == PASS_SLOT {
                // Pass uniform table entry (`simulation.mjs` writes these at PASS_ALIGN).
                let v = pass_value(pass);
                let bytes = [v, 0u32, 0, 0];
                enc.set_bytes(index, 16, bytes.as_ptr() as *const _);
            } else if slot == ARGS_SLOT && indirect {
                // Never bind the indirect buffer as a shader resource on the dispatch
                // that consumes it for threadgroup counts.
                enc.set_buffer(index, None, 0);
            } else {
                enc.set_buffer(index, Some(&self.bufs[slot]), 0);
            }
        }
        pipe
    }

    /// `dispatchWorkgroups(groups(threads))`: ceil(threads/64) groups in rows of 32768.
    fn disp(&self, enc: &ComputeCommandEncoderRef, name: &str, threads: u32, pass: u32) {
        let pipe = self.bind(enc, name, pass, false);
        let n = threads.max(1).div_ceil(64) as u64;
        let t = pipe.kernel.threads;
        enc.dispatch_thread_groups(
            MTLSize::new(n.min(32768), n.div_ceil(32768), 1),
            MTLSize::new(t[0] as u64, t[1] as u64, t[2] as u64),
        );
    }

    fn disp_work(&self, enc: &ComputeCommandEncoderRef, name: &str, slot: u32, pass: u32) {
        let pipe = self.bind(enc, name, pass, true);
        let t = pipe.kernel.threads;
        enc.dispatch_thread_groups_indirect(
            &self.bufs[ARGS_SLOT],
            slot as u64 * 12,
            MTLSize::new(t[0] as u64, t[1] as u64, t[2] as u64),
        );
    }

    fn encode_step(&self, cb: &CommandBufferRef, ring: usize) {
        let n_bodies = self.body_count.max(1);
        let n_cols = self.collider_count.max(1);
        let ids_len = next_pow2((2 * (self.cap.max_contacts + self.cap.max_joints)).max(32768));
        let hash_n = next_pow2((self.cap.max_contacts.max(256) * 4).max(1024))
            .min(ids_len)
            .min(MAX_WARM_HASH);
        self.write_host(cb, ring);
        let desc = ComputePassDescriptor::new();
        desc.set_dispatch_type(MTLDispatchType::Serial);
        let e = cb.compute_command_encoder_with_descriptor(desc);
        self.disp(e, "build_aabbs", n_cols, 0);
        self.disp(e, "broadphase_clear", 131072, 0);
        self.disp(e, "broadphase_insert", n_cols, 0);
        self.disp(e, "broadphase_query", n_cols, 0);
        self.disp(e, "write_work_args", 64, 0);
        let reuse = self.params.flags & 1 != 0;
        if reuse {
            self.disp(e, "update_refs", n_bodies, 0);
            self.disp(e, "contact_hash_clear", hash_n, 0);
            self.disp_work(e, "contact_hash_insert", 68, 0);
        }
        self.disp_work(e, "prepare_candidates", 64, 0);
        self.disp(e, "write_sat_args", 1, 0);
        self.disp_work(e, "sat", 69, 0);
        self.disp_work(e, "narrowphase", 64, 0);
        self.disp(e, "write_work_args", 64, 0);
        if !reuse {
            self.disp(e, "contact_hash_clear", hash_n, 0);
            self.disp_work(e, "contact_hash_insert", 68, 0);
        }
        self.disp_work(e, "persist_contacts", 65, 0);
        self.disp(e, "build_awake_list", n_bodies, 0);
        self.disp(e, "write_work_args", 64, 0);
        self.disp(e, "csr_clear", n_bodies + 1, 0);
        self.disp_work(e, "csr_degree_contacts", 66, 0);
        if self.body_count <= 512 {
            self.disp(e, "csr_prefix", 1, 0);
        } else {
            self.disp(e, "csr_prefix_init", n_bodies, 0);
            let mut log = 0;
            while (1u32 << log) < self.body_count && log < 20 {
                self.disp(e, "csr_prefix_scan", n_bodies, 48 + log);
                log += 1;
            }
            self.disp(e, "csr_prefix_finalize", n_bodies + 1, 0);
            self.disp(e, "csr_prefix_swap", n_bodies, 0);
        }
        self.disp_work(e, "csr_scatter_contacts", 66, 0);
        self.disp_work(e, "csr_sort", 67, 0);
        self.disp_work(e, "init_colors", 67, 0);
        // Complete graph coloring before solving; the shader reports unfinished colors.
        for round in 0..32 {
            self.disp_work(e, "color_bodies", 67, round % 16);
        }
        self.disp_work(e, "pack_color_lists", 67, 0);
        self.disp(e, "write_color_args", 64, 0);
        self.disp(e, "predict_bodies", n_bodies, 0);
        // Current-step indirect arguments skip empty colors without a CPU estimate.
        let colors = self.cap.max_colors.min(32);
        for _ in 0..self.params.iterations {
            for color in 0..colors {
                self.disp_work(e, "solve_primal_color", color, color);
            }
            self.disp_work(e, "solve_dual", 65, 0);
        }
        self.disp(e, "extract_velocity_sleep", n_bodies, 0);
        for _ in 0..8 {
            self.disp_work(e, "sleep_island_gate", 66, 0);
        }
        self.disp(e, "sleep_island_commit", n_bodies, 0);
        e.end_encoding();
    }

    /// Submit `count` (1–4) ticks without waiting. Fully sleeping scenes skip GPU work.
    pub fn submit(&mut self, count: u32) -> anyhow::Result<bool> {
        anyhow::ensure!(
            (1..=RING as u32).contains(&count),
            "Physics batch must have 1–4 ticks"
        );
        self.wait()?;
        if self.initialized && self.last.awake == 0 {
            self.steps += count;
            self.timing = Timing::default();
            return Ok(false);
        }
        let start = Instant::now();
        let cb = self.queue.new_command_buffer().to_owned();
        cb.set_label("fracture physics");
        for i in 0..count {
            self.encode_step(&cb, i as usize);
            self.initialized = true;
            self.parity ^= 1;
            self.steps += 1;
        }
        cb.commit();
        let encode = start.elapsed().as_secs_f64() * 1000.0;
        self.timing = Timing {
            wall_ms: encode / count as f64,
            encode_ms: encode / count as f64,
            wait_ms: 0.0,
        };
        self.in_flight = Some(cb);
        self.batch = (start, count);
        Ok(true)
    }

    /// Wait for the in-flight batch, then read counters in place.
    pub fn wait(&mut self) -> anyhow::Result<()> {
        let Some(cb) = self.in_flight.take() else {
            return Ok(());
        };
        let waited = Instant::now();
        cb.wait_until_completed();
        let (start, count) = self.batch;
        let end = Instant::now();
        let c = |k| self.read_u32(7, k);
        self.last = Counters {
            contact: c(2),
            candidate: c(4),
            awake: c(5),
            color: c(6),
            manifold: c(12),
            overflow: c(7),
        };
        // Wall includes queue execution; not isolated GPU time.
        let n = count as f64;
        self.timing.wall_ms = (end - start).as_secs_f64() * 1000.0 / n;
        self.timing.wait_ms = (end - waited).as_secs_f64() * 1000.0 / n;
        anyhow::ensure!(
            cb.status() != MTLCommandBufferStatus::Error,
            "Metal command buffer failed"
        );
        anyhow::ensure!(
            self.last.overflow == 0,
            "Physics capacity exceeded (0x{:x}); choose a coarser level",
            self.last.overflow
        );
        Ok(())
    }

    /// Fire the next projectile along a ray; aims near the scene centre's depth.
    pub fn launch(&mut self, origin: [f32; 3], direction: [f32; 3]) -> anyhow::Result<u32> {
        anyhow::ensure!(
            self.shots < self.projectile_count,
            "All 16 projectiles fired. Reset to reload."
        );
        anyhow::ensure!(
            origin.iter().chain(&direction).all(|x| x.is_finite()),
            "Invalid projectile ray"
        );
        self.wait()?;
        let id = self.projectile_start + self.shots;
        let (m, r) = (self.projectile_mass, self.projectile_radius);
        let speed = (self.extent * 3.0).clamp(5.0, 18.0);
        let distance = r.max(
            (0..3)
                .map(|i| (self.center[i] - origin[i]) * direction[i])
                .sum(),
        );
        let mut v = direction.map(|x| x * speed);
        v[1] += 0.5 * 9.81 * distance / speed;
        let inv_i = 1.0 / (0.4 * m * r * r);
        let body = engine_fracture::Body {
            p: origin,
            q: [0.0, 0.0, 0.0, 1.0],
            v,
            inv_mass: 1.0 / m,
            inv_i: [inv_i; 3],
            flags: 9,
        };
        let bytes = body.to_bytes();
        self.write(0, id as usize * 112, &bytes);
        self.write(0, (self.cap.max_bodies + id) as usize * 112, &bytes);
        self.last.awake = self.last.awake.max(1);
        self.shots += 1;
        Ok(id)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Run only the collision kernels at the current poses without advancing them
    /// (port of `inspectCollisions()`); returns contacts as 32-word records.
    pub fn inspect_collisions(&mut self) -> anyhow::Result<Vec<[u32; 32]>> {
        self.wait()?;
        let saved = self.params.flags;
        self.params.flags = 0;
        let cb = self.queue.new_command_buffer();
        self.write_host(cb, 0);
        self.params.flags = saved;
        let e = cb.new_compute_command_encoder();
        let n = self.collider_count;
        self.disp(e, "build_aabbs", n, 0);
        self.disp(e, "broadphase_clear", 131072, 0);
        self.disp(e, "broadphase_insert", n, 0);
        self.disp(e, "broadphase_query", n, 0);
        self.disp(e, "write_work_args", 64, 0);
        self.disp_work(e, "prepare_candidates", 64, 0);
        self.disp(e, "write_sat_args", 1, 0);
        self.disp_work(e, "sat", 69, 0);
        self.disp_work(e, "narrowphase", 64, 0);
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let contacts = self.read_u32(7, 2) as usize;
        let base = if self.parity == 0 {
            0
        } else {
            self.cap.max_contacts as usize
        };
        let words = unsafe {
            std::slice::from_raw_parts(
                (self.bufs[4].contents() as *const [u32; 32]).add(base),
                contacts,
            )
        };
        Ok(words.to_vec())
    }

    /// Copy of the first `count` 112-byte bodies as f32 words (28 per body).
    pub fn snapshot(&mut self, count: usize) -> anyhow::Result<Vec<f32>> {
        self.wait()?;
        let words = unsafe {
            std::slice::from_raw_parts(self.bufs[0].contents() as *const f32, count * 28)
        };
        Ok(words.to_vec())
    }
}

/// Pass table entry: identity for 0..48, scan strides `(1<<k)|((k&1)<<31)` for 48..68.
fn pass_value(i: u32) -> u32 {
    if (48..68).contains(&i) {
        let k = i - 48;
        (1 << k) | ((k & 1) << 31)
    } else if i >= 68 {
        0
    } else {
        i
    }
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
fn bytemuck_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
