//! wgpu submission of the Slang→WGSL phase graph (same buffers/order as Metal).
//! This is the fair path vs three-avbd Chrome WebGPU.

use bytemuck::bytes_of;
use engine_exec::{ExecError, PhaseTimingsMs};
use engine_schema::*;
use std::collections::HashMap;

const BIND_COUNT: usize = 27;
const PASS_ALIGN: u64 = 256;

struct Pipe {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    dynamic_pass: bool,
}

pub struct WgpuSim {
    device: wgpu::Device,
    queue: wgpu::Queue,
    capacity: WorldCapacity,
    params: AvbdParams,
    pipes: HashMap<String, Pipe>,
    bufs: Vec<wgpu::Buffer>,
    body_count: u32,
    collider_count: u32,
    joint_count: u32,
    command_count: u32,
    sleep_island_rounds: u32,
    last_counters: StepCounters,
    last_timings: PhaseTimingsMs,
    steps_done: u32,
    contacts_parity: u32,
    host_bodies: Vec<BodyState>,
    host_cols: Vec<Collider>,
    chip: String,
}

/// WebGPU `maxComputeWorkgroupsPerDimension` default is 65535. A 2^22-thread
/// dispatch is 65536 groups and invalidates the command buffer (columns 100k).
fn groups(threads: u64) -> u32 {
    const MAX_WG: u32 = 65535;
    let g = ((threads.max(1) + 63) / 64) as u32;
    g.min(MAX_WG)
}

struct Bind {
    index: u32,
    uniform: bool,
}

fn bindings_of(wgsl: &str) -> Vec<Bind> {
    let mut out = Vec::new();
    for line in wgsl.lines() {
        let Some(i) = line.find("@binding(") else {
            continue;
        };
        let rest = &line[i + "@binding(".len()..];
        let end = rest.find(')').unwrap_or(0);
        let Ok(index) = rest[..end].trim().parse::<u32>() else {
            continue;
        };
        if out.iter().any(|b: &Bind| b.index == index) {
            continue;
        }
        out.push(Bind {
            index,
            uniform: line.contains("var<uniform"),
        });
    }
    out.sort_by_key(|b| b.index);
    out
}

/// Word offset where the CSR index lives inside `gConstraintIds` on the WGSL path.
fn constraint_id_words(c: &WorldCapacity) -> u32 {
    let words = (2 * (c.max_contacts + c.max_joints))
        .max(32768)
        .next_power_of_two() as u64;
    words.max(131072) as u32
}

impl WgpuSim {
    pub fn new(capacity: WorldCapacity) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow::anyhow!("no wgpu adapter"))?;
        let info = adapter.get_info();
        let align = adapter.limits().min_storage_buffer_offset_alignment;
        if u64::from(align) > PASS_ALIGN {
            anyhow::bail!("storage offset alignment {align} > {PASS_ALIGN}");
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("engine-webgpu-sim"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: Default::default(),
            },
            None,
        ))?;

        let sizes = buffer_sizes(&capacity);
        let mut bufs = Vec::with_capacity(BIND_COUNT);
        for (i, &bytes) in sizes.iter().enumerate() {
            let usage = wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | if i == 6 || i == 22 {
                    wgpu::BufferUsages::UNIFORM
                } else {
                    wgpu::BufferUsages::STORAGE
                }
                | if i == 23 {
                    wgpu::BufferUsages::INDIRECT
                } else {
                    wgpu::BufferUsages::empty()
                };
            bufs.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("b{i}")),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            }));
        }
        // Zero pass table slots (256-byte aligned so dynamic offsets are legal).
        let mut pass = vec![0u8; (96 * PASS_ALIGN) as usize];
        for i in 0..96u32 {
            let v = if i < 48 {
                i
            } else if i < 68 {
                let k = i - 48;
                (1u32 << k) | ((k & 1) << 31)
            } else {
                0
            };
            let o = (i as usize) * PASS_ALIGN as usize;
            pass[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }
        queue.write_buffer(&bufs[22], 0, &pass);

        let mats = [Material {
            friction: 0.5,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        }; 64];
        queue.write_buffer(&bufs[3], 0, bytemuck::cast_slice(&mats));

        let shaders = crate::WebGpuWorld::compile_shaders_report();
        let mut pipes = HashMap::new();
        for (name, r) in shaders {
            let wgsl = r.map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            let binds = bindings_of(&wgsl);
            let mut entries = Vec::new();
            for b in &binds {
                let dynamic = b.index == 22;
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: b.index,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if b.uniform {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: false }
                        },
                        has_dynamic_offset: dynamic,
                        min_binding_size: if dynamic {
                            std::num::NonZeroU64::new(16)
                        } else {
                            None
                        },
                    },
                    count: None,
                });
            }
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&name),
                entries: &entries,
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&name),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&name),
                layout: Some(&device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some(&name),
                        bind_group_layouts: &[&layout],
                        push_constant_ranges: &[],
                    },
                )),
                module: &module,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });
            let resources: Vec<wgpu::BindGroupEntry> = binds
                .iter()
                .map(|b| wgpu::BindGroupEntry {
                    binding: b.index,
                    resource: if b.index == 22 {
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &bufs[22],
                            offset: 0,
                            size: std::num::NonZeroU64::new(16),
                        })
                    } else {
                        bufs[b.index as usize].as_entire_binding()
                    },
                })
                .collect();
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&name),
                layout: &layout,
                entries: &resources,
            });
            pipes.insert(
                name,
                Pipe {
                    pipeline,
                    bind,
                    dynamic_pass: binds.iter().any(|b| b.index == 22),
                },
            );
        }

        let _ = mats;
        Ok(Self {
            device,
            queue,
            capacity,
            params: AvbdParams::default(),
            pipes,
            bufs,
            body_count: 0,
            collider_count: 0,
            joint_count: 0,
            command_count: 0,
            sleep_island_rounds: 8,
            last_counters: StepCounters::default(),
            last_timings: PhaseTimingsMs::default(),
            steps_done: 0,
            contacts_parity: 0,
            host_bodies: Vec::new(),
            host_cols: Vec::new(),
            chip: format!("{} / {}", info.name, info.backend),
        })
    }

    pub fn chip(&self) -> &str {
        &self.chip
    }

    pub fn set_params(&mut self, params: AvbdParams) {
        self.params = params;
    }

    pub fn add_body(&mut self, body: BodyState) -> u32 {
        let id = self.body_count;
        self.host_bodies.push(body);
        let stride = std::mem::size_of::<BodyState>() as u64;
        self.queue
            .write_buffer(&self.bufs[0], id as u64 * stride, bytes_of(&body));
        // WGSL packs previous poses at gBodies[maxBodies + i].
        self.queue.write_buffer(
            &self.bufs[0],
            (u64::from(self.capacity.max_bodies) + u64::from(id)) * stride,
            bytes_of(&body),
        );
        self.body_count += 1;
        id
    }

    pub fn add_collider(&mut self, col: Collider) -> u32 {
        let id = self.collider_count;
        self.host_cols.push(col);
        self.queue.write_buffer(
            &self.bufs[2],
            id as u64 * std::mem::size_of::<Collider>() as u64,
            bytes_of(&col),
        );
        self.collider_count += 1;
        id
    }

    pub fn body_y(&self, id: u32) -> f32 {
        self.read_body(id).position[1]
    }

    pub fn read_body(&self, id: u32) -> BodyState {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("body-read"),
            size: std::mem::size_of::<BodyState>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(
            &self.bufs[0],
            id as u64 * std::mem::size_of::<BodyState>() as u64,
            &staging,
            0,
            std::mem::size_of::<BodyState>() as u64,
        );
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let body: BodyState = *bytemuck::from_bytes(&data[..std::mem::size_of::<BodyState>()]);
        drop(data);
        staging.unmap();
        body
    }

    pub fn step(&mut self) -> Result<StepCounters, ExecError> {
        let wall = std::time::Instant::now();
        if self.steps_done > 0
            && self.last_counters.awake_count == 0
            && self.command_count == 0
            && self.body_count > 0
        {
            self.steps_done += 1;
            self.last_timings.total_gpu = 0.0;
            return Ok(self.last_counters);
        }
        self.refresh_cell();
        let n_bodies = self.body_count.max(1) as u64;
        let n_cols = self.collider_count.max(1) as u64;
        let n_joints = self.joint_count.max(1) as u64;
        let awake_dispatch = if self.last_counters.awake_count == 0 {
            n_bodies
        } else {
            ((self.last_counters.awake_count as u64).saturating_mul(2) + 256)
                .min(n_bodies)
                .max(1)
        };
        let prev_contacts = self.last_counters.contact_count.max(256) as u64;
        let prev_cand = self.last_counters.candidate_count.max(256) as u64;
        let narrow_n = (self.capacity.max_candidates as u64).min(
            prev_cand
                .saturating_add(prev_cand / 2)
                .saturating_add(2048)
                .max(n_cols.saturating_mul(4))
                .max(prev_contacts.saturating_add(prev_contacts / 4).saturating_add(1024)),
        );
        let contact_dispatch = (self.capacity.max_contacts as u64).min(
            prev_contacts
                .saturating_add(prev_contacts / 4)
                .saturating_add(1024)
                .max(n_cols.saturating_mul(4)),
        );
        let prev_manifolds = self.last_counters.manifold_count.max(256) as u64;
        let manifold_dispatch = narrow_n.max(
            prev_manifolds
                .saturating_add(prev_manifolds / 4)
                .saturating_add(1024),
        );

        self.write_host_uniforms();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("step"),
            });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("step"),
            timestamp_writes: None,
        });

        self.disp(&mut pass, "build_aabbs", n_cols, None);
        if self.joint_count > 0 {
            self.disp(&mut pass, "joint_hash_clear", 65536, None);
            self.disp(&mut pass, "joint_hash_insert", n_joints, None);
        }
        self.disp(&mut pass, "broadphase_clear", 131072, None);
        self.disp(&mut pass, "broadphase_insert", n_cols, None);
        self.disp(&mut pass, "broadphase_query", n_cols, None);

        let ids_len = (2 * (self.capacity.max_contacts + self.capacity.max_joints))
            .max(32768)
            .next_power_of_two();
        // Largest pow2 thread count with workgroups ≤ 65535 (2^22 → 65536 groups = invalid).
        const MAX_POW2_DISPATCH: u32 = 1 << 21;
        let hash_n = (prev_contacts.max(256).saturating_mul(4).max(1024) as u32)
            .next_power_of_two()
            .min(ids_len)
            .min(MAX_POW2_DISPATCH) as u64;
        let reuse = (self.params.flags & FLAG_REUSE_CONTACTS) != 0;
        if reuse {
            self.disp(&mut pass, "update_refs", n_bodies, None);
            self.disp(&mut pass, "contact_hash_clear", hash_n, None);
            self.disp(&mut pass, "contact_hash_insert", prev_contacts.max(1), None);
        }
        self.disp(&mut pass, "narrowphase", narrow_n, None);
        if !reuse {
            self.disp(&mut pass, "contact_hash_clear", hash_n, None);
            self.disp(&mut pass, "contact_hash_insert", prev_contacts.max(1), None);
        }
        self.disp(&mut pass, "persist_contacts", contact_dispatch, None);
        if self.joint_count > 0 {
            self.disp(&mut pass, "warmstart_joints", n_joints, None);
        }
        self.disp(&mut pass, "build_awake_list", n_bodies, None);
        self.disp(&mut pass, "csr_clear", n_bodies + 1, None);
        self.disp(&mut pass, "csr_degree_contacts", manifold_dispatch, None);
        if self.body_count <= 512 {
            self.disp(&mut pass, "csr_prefix", 1, None);
        } else {
            self.disp(&mut pass, "csr_prefix_init", n_bodies, None);
            let mut log = 0u32;
            while (1u32 << log) < self.body_count && log < 20 {
                self.disp(&mut pass, "csr_prefix_scan", n_bodies, Some(48 + log));
                log += 1;
            }
            self.disp(&mut pass, "csr_prefix_finalize", n_bodies + 1, None);
        }
        self.disp(&mut pass, "csr_scatter_contacts", manifold_dispatch, None);
        self.disp(&mut pass, "csr_sort", awake_dispatch, None);
        self.disp(&mut pass, "init_colors", awake_dispatch, None);
        // Match Metal / three-avbd: ≥16 JP rounds, colorCap ≤32 (was hard-capped at 8 for
        // body_count>8192 — under-coloured Pyramid 30 vs oracle's ~13 colours).
        let color_cap = self.capacity.max_colors.min(32);
        let color_rounds = self
            .last_counters
            .color_count
            .saturating_add(1)
            .max(16)
            .min(color_cap);
        for round in 0..color_rounds {
            self.disp(&mut pass, "color_bodies", awake_dispatch, Some(round % 16));
        }
        self.disp(&mut pass, "pack_color_lists", awake_dispatch, None);
        self.disp(&mut pass, "write_color_args", 64, None);
        self.disp(&mut pass, "predict_bodies", n_bodies, None);

        let dual_n = contact_dispatch.max(n_joints);
        let max_c = self
            .last_counters
            .color_count
            .saturating_add(1)
            .max(16)
            .min(self.capacity.max_colors.min(32));
        for _iter in 0..self.params.iterations {
            for color in 0..max_c {
                self.disp_indirect(&mut pass, color);
            }
            self.disp(&mut pass, "solve_dual", dual_n, None);
        }
        self.disp(&mut pass, "extract_velocity_sleep", n_bodies, None);
        if self.last_counters.manifold_count > 0 && self.sleep_island_rounds > 0 {
            for _ in 0..self.sleep_island_rounds.min(16) {
                self.disp(&mut pass, "sleep_island_gate", manifold_dispatch, None);
            }
        }
        self.disp(&mut pass, "sleep_island_commit", n_bodies, None);
        drop(pass);
        self.queue.submit([enc.finish()]);

        let counters = self.read_counters();
        self.contacts_parity ^= 1;
        self.steps_done = self.steps_done.saturating_add(1);
        self.command_count = 0;
        self.last_timings.total_gpu = wall.elapsed().as_secs_f64() * 1000.0;
        self.last_counters = counters;
        if counters.overflow_flags != 0 {
            return Err(ExecError::Overflow(counters.overflow_flags));
        }
        Ok(counters)
    }

    fn disp<'a>(
        &'a self,
        pass: &mut wgpu::ComputePass<'a>,
        name: &str,
        threads: u64,
        pass_idx: Option<u32>,
    ) {
        if let Ok(only) = std::env::var("WG_ONLY") {
            if !only.split(',').any(|n| n.trim() == name) {
                return;
            }
        }
        if let Ok(skip) = std::env::var("WG_SKIP") {
            if skip.split(',').any(|n| n.trim() == name) {
                return;
            }
        }
        let p = self.pipes.get(name).unwrap_or_else(|| panic!("missing {name}"));
        pass.set_pipeline(&p.pipeline);
        if p.dynamic_pass {
            let idx = pass_idx.unwrap_or(0);
            pass.set_bind_group(0, &p.bind, &[idx * PASS_ALIGN as u32]);
        } else {
            pass.set_bind_group(0, &p.bind, &[]);
        }
        pass.dispatch_workgroups(groups(threads), 1, 1);
    }

    fn disp_indirect<'a>(&'a self, pass: &mut wgpu::ComputePass<'a>, color: u32) {
        if let Ok(only) = std::env::var("WG_ONLY") {
            if !only.split(',').any(|n| n.trim() == "solve_primal_color") {
                return;
            }
        }
        if let Ok(skip) = std::env::var("WG_SKIP") {
            if skip.split(',').any(|n| n.trim() == "solve_primal_color") {
                return;
            }
        }
        let p = &self.pipes["solve_primal_color"];
        pass.set_pipeline(&p.pipeline);
        pass.set_bind_group(0, &p.bind, &[color * PASS_ALIGN as u32]);
        pass.dispatch_workgroups_indirect(&self.bufs[23], (color as u64) * 12);
    }

    fn write_host_uniforms(&self) {
        let max_c = self.capacity.max_contacts;
        let cur = if self.contacts_parity == 0 { 0 } else { max_c };
        let prev = if self.contacts_parity == 0 { max_c } else { 0 };
        let ids_len = constraint_id_words(&self.capacity);
        let prev_n = self.last_counters.contact_count.min(max_c).max(256);
        let mut s = [0u32; 64];
        s[1] = self.capacity.max_bodies;
        s[2] = self.capacity.max_candidates;
        s[3] = max_c;
        s[7] = self.last_counters.contact_count.min(max_c);
        s[8] = prev;
        const MAX_POW2_DISPATCH: u32 = 1 << 21;
        s[9] = prev_n
            .saturating_mul(4)
            .max(1024)
            .next_power_of_two()
            .min(ids_len)
            .min(MAX_POW2_DISPATCH);
        s[10] = self.capacity.max_candidates;
        s[11] = cur;
        let mut log = 0u32;
        while (1u32 << log) < self.body_count.max(1) && log < 20 {
            log += 1;
        }
        s[12] = if log == 0 { 0 } else { (log - 1) & 1 };
        s[17] = self.steps_done;
        // CSR index region starts past the hash/id words inside gConstraintIds.
        s[24] = ids_len;
        self.queue.write_buffer(&self.bufs[21], 0, bytemuck::cast_slice(&s));

        let c = StepCounters {
            body_count: self.body_count,
            collider_count: self.collider_count,
            joint_count: self.joint_count,
            ..StepCounters::default()
        };
        self.queue.write_buffer(&self.bufs[7], 0, bytes_of(&c));
        let mut packed = [[0u32; 4]; 8];
        for i in 0..32 {
            packed[i / 4][i % 4] = s[i];
        }
        let host = HostUniforms {
            params: self.params,
            s: packed,
        };
        self.queue.write_buffer(&self.bufs[6], 0, bytes_of(&host));
        let n = self.capacity.max_colors.min(64) as usize;
        let zeros = vec![0u32; n];
        self.queue
            .write_buffer(&self.bufs[17], 0, bytemuck::cast_slice(&zeros));
    }

    fn refresh_cell(&mut self) {
        let mut max_r = 0.25f32;
        for c in &self.host_cols {
            if c.shape == ShapeType::Plane as u32 {
                continue;
            }
            let body = c.body as usize;
            if body >= self.host_bodies.len() || self.host_bodies[body].inv_mass <= 0.0 {
                continue;
            }
            let r = if c.params[3] > 0.0 {
                c.params[3]
            } else {
                (c.params[0] * c.params[0] + c.params[1] * c.params[1] + c.params[2] * c.params[2])
                    .sqrt()
            };
            if r > 2.0 {
                continue;
            }
            max_r = max_r.max(r);
        }
        self.params.grid_cell_size = (2.0 * max_r * 1.0001).max(0.5);
    }

    fn read_counters(&self) -> StepCounters {
        let n = std::mem::size_of::<StepCounters>() as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("counters-read"),
            size: n,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(&self.bufs[7], 0, &staging, 0, n);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let c: StepCounters = *bytemuck::from_bytes(&data[..n as usize]);
        drop(data);
        staging.unmap();
        c
    }

    pub fn timings_ms(&self) -> f64 {
        self.last_timings.total_gpu
    }
}

/// Matches `HostUniforms` in `world_step.slang` (`AvbdParams` then `uint4 s[8]`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HostUniforms {
    params: AvbdParams,
    s: [[u32; 4]; 8],
}

fn buffer_sizes(c: &WorldCapacity) -> [u64; BIND_COUNT] {
    let b = |n: u32, stride: u64| n as u64 * stride;
    let mut s = [256u64; BIND_COUNT];
    s[0] = b(c.max_bodies.saturating_mul(2), std::mem::size_of::<engine_schema::BodyState>() as u64);
    s[1] = b(c.max_bodies, std::mem::size_of::<engine_schema::BodyState>() as u64);
    s[2] = b(c.max_colliders, 64);
    s[3] = b(64, 16);
    s[4] = b(c.max_contacts.saturating_mul(2), 128);
    s[5] = b(c.max_joints, 256);
    s[6] = 256;
    s[7] = 256;
    s[8] = b(c.max_commands, 64);
    s[9] = b(c.max_bodies + c.max_colliders, 32);
    s[10] = b(c.max_candidates, 16);
    s[11] = b(c.max_bodies, 4);
    s[12] = b(c.max_bodies, 4);
    s[13] = b(c.max_bodies + 1, 4);
    s[14] = (u64::from(constraint_id_words(c)) + u64::from(c.max_bodies) + 1) * 4;
    s[15] = b(c.max_bodies * c.max_colors, 4);
    s[16] = b(c.max_colors, 4);
    s[17] = b(c.max_colors, 4);
    s[18] = b(c.max_hull_vertices, 16);
    s[19] = 65536 * 4;
    s[20] = b(c.max_bodies, 4);
    s[21] = 64 * 4;
    s[22] = 96 * PASS_ALIGN;
    s[23] = b(c.max_colors.max(64) * 3, 4);
    s[24] = b(c.max_colliders, 80);
    s[25] = b(c.max_candidates, 32);
    s[26] = b(c.max_bodies, 32);
    s
}
