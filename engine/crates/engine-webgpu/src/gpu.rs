//! wgpu submission of the Slang→WGSL phase graph (same buffers/order as Metal).
//! This is the fair path vs three-avbd Chrome WebGPU.

use bytemuck::bytes_of;
use engine_exec::{ExecError, PhaseTimingsMs};
use engine_schema::*;
use std::collections::HashMap;

const BIND_COUNT: usize = 29;
const PASS_ALIGN: u64 = 256;

struct Pipe {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    dynamic_pass: bool,
}

pub struct WgpuSim {
    compact_contacts: bool,
    persistent_contacts: bool,
    sparse_start: u32,
    manifold_persistence: bool,
    /// Opt-in three-avbd shape: warm-hash insert per previous manifold.
    pub(crate) manifold_hash_insert: bool,
    /// Record warm hash: 16-byte slots carrying the pair key and previous run.
    pub(crate) warm_record_hash: bool,
    /// Narrowphase finishes reused copies (persistent contacts only).
    pub(crate) reuse_copy: bool,
    /// Two-pass narrowphase (8ek): reuse, then shape tests over compacted pairs.
    pub(crate) np_split: bool,
    /// `ENGINE_PAIRS=np`: the two-pass narrowphase writes solver pairs itself,
    /// so the separate pack pass is skipped (requires `np_split`).
    pub(crate) np_pairs: bool,
    /// Persist work list (8em, HOST_S(22)); requires reuse copy.
    pub(crate) persist_list: bool,
    /// Carry resting pairs across sleep (8ep, HOST_S(28)).
    pub(crate) carry_rest: bool,
    /// Local incremental color repair (8en, HOST_S(27)).
    pub(crate) local_repair: bool,
    /// Test hook: caps the persist list (HOST_S(23)); 0 uses the whole region.
    pub(crate) persist_list_limit: u32,
    /// Test hook: caps deferred pairs (HOST_S(21)); 0 uses the whole tail.
    pub(crate) np_defer_limit: u32,
    fused_persistence: bool,
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
    /// Released joint slots, reused by `add_joint`.
    free_joints: Vec<u32>,
    /// Per-body collider reach (torque arm default).
    body_reach: Vec<f32>,
    chip: String,
}

/// Must agree with Slang dispatchIndex: 2^21 threads per row, with no truncation.
fn groups(threads: u64) -> (u32, u32) {
    let g = threads.max(1).div_ceil(64);
    let x = g.min(32768);
    let y = g.div_ceil(32768);
    assert!(
        y <= 65535,
        "dispatch exceeds WebGPU's two-dimensional workgroup limit"
    );
    (x as u32, y as u32)
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
/// Split broadphase grids: awake dynamics in one table, sleepers and small
/// statics in a second (HOST_S(20)); awake queriers walk 14 cells.
fn grid_split() -> bool {
    std::env::var("ENGINE_GRID").as_deref() == Ok("split")
}

fn constraint_id_words(c: &WorldCapacity) -> u32 {
    let words = (2 * (c.max_contacts + c.max_joints))
        .max(32768)
        .next_power_of_two() as u64;
    words.max(131072) as u32
}

fn persistent_entry(name: &str) -> &str {
    match name {
        "apply_commands" => "apply_commands_persistent",
        "narrowphase" => "narrowphase_persistent",
        "contact_hash_insert" => "contact_hash_insert_persistent",
        "contact_hash_insert_manifolds" => "contact_hash_insert_manifolds_persistent",
        "persist_contacts" => "persist_contacts_persistent",
        "persist_manifolds" | "persist_manifolds_compact" => "persist_manifolds_persistent",
        "pack_solver_contacts" => "pack_solver_pairs_persistent",
        // 8el: persistent hosts pack pairs before the graph; CSR reads their body ids.
        "csr_degree_contacts" => "csr_degree_contacts_persistent",
        "csr_scatter_contacts" => "csr_scatter_contacts_persistent",
        "solve_primal_color_compact" => "solve_primal_color_persistent",
        "solve_primal_sparse_compact" => "solve_primal_sparse_persistent",
        "solve_primal_one_compact" => "solve_primal_one_persistent",
        "solve_dual_compact" => "solve_dual_persistent",
        _ => name,
    }
}

impl WgpuSim {
    pub fn new(capacity: WorldCapacity) -> anyhow::Result<Self> {
        if std::env::var("ENGINE_PERSISTENT_CONTACTS").as_deref() == Ok("1") {
            return Self::new_with_persistent_contacts(capacity);
        }
        Self::new_with_compact_contacts(
            capacity,
            std::env::var("ENGINE_COMPACT_CONTACTS").as_deref() == Ok("1"),
        )
    }

    pub fn new_with_compact_contacts(
        capacity: WorldCapacity,
        compact_contacts: bool,
    ) -> anyhow::Result<Self> {
        Self::new_with_contact_storage(capacity, compact_contacts, false)
    }

    pub fn new_with_persistent_contacts(capacity: WorldCapacity) -> anyhow::Result<Self> {
        Self::new_with_contact_storage(capacity, true, true)
    }

    fn new_with_contact_storage(
        capacity: WorldCapacity,
        compact_contacts: bool,
        persistent_contacts: bool,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            capacity.max_colliders < 0x7fff_ffff,
            "grid collider ID overlaps awake-dynamic flag or empty sentinel"
        );
        if persistent_contacts {
            anyhow::ensure!(
                capacity.max_candidates < 0x3fff_ffff,
                "persistent owner index exceeds 30 bits"
            );
        }
        anyhow::ensure!(
            capacity.max_candidates < 0x7fff_ffff,
            "compact pair index exceeds 31 bits"
        );
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

        let mut sizes = buffer_sizes(&capacity);
        if persistent_contacts {
            sizes[4] = u64::from(capacity.max_contacts) * 2 * 32;
            sizes[27] = u64::from(capacity.max_contacts) * 2 * 64;
        }
        if !compact_contacts {
            sizes[27] = 64;
            sizes[28] = 32;
        }
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
        for (name, canonical) in &shaders {
            // Persistent variants load under their canonical aliases; reuse-copy
            // entries keep their own names (they have no canonical twin).
            if name.ends_with("_persistent")
                && !(persistent_contacts && name.ends_with("_copy_persistent"))
            {
                continue;
            }
            let entry = if persistent_contacts {
                persistent_entry(name)
            } else {
                name.as_str()
            };
            let r = if entry == name {
                canonical
            } else {
                &shaders
                    .iter()
                    .find(|(n, _)| n == entry)
                    .ok_or_else(|| anyhow::anyhow!("missing persistent shader {entry}"))?
                    .1
            };
            if !compact_contacts
                && (name.ends_with("_compact") || name.ends_with("_solver_contacts"))
            {
                continue;
            }
            let wgsl = r
                .as_ref()
                .map_err(|e| anyhow::anyhow!("{entry}: {e}"))?
                .clone();
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
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&name),
                        bind_group_layouts: &[&layout],
                        push_constant_ranges: &[],
                    }),
                ),
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
                name.clone(),
                Pipe {
                    pipeline,
                    bind,
                    dynamic_pass: binds.iter().any(|b| b.index == 22),
                },
            );
        }

        let _ = mats;
        Ok(Self {
            compact_contacts,
            persistent_contacts,
            sparse_start: if capacity.max_bodies <= crate::WGSL_LOOP_CAP * 8 {
                std::env::var("ENGINE_SPARSE_COLORS")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(32)
                    .min(32)
            } else {
                32
            },
            fused_persistence: compact_contacts
                && std::env::var("ENGINE_MANIFOLD_PERSISTENCE").as_deref() == Ok("fused"),
            manifold_persistence: std::env::var("ENGINE_MANIFOLD_PERSISTENCE").as_deref()
                == Ok("1"),
            manifold_hash_insert: matches!(
                std::env::var("ENGINE_HASH_INSERT").as_deref(),
                Ok("manifold") | Ok("record")
            ),
            warm_record_hash: std::env::var("ENGINE_HASH_INSERT").as_deref() == Ok("record"),
            reuse_copy: persistent_contacts
                && std::env::var("ENGINE_REUSE").as_deref() == Ok("copy"),
            np_split: persistent_contacts
                && std::env::var("ENGINE_REUSE").as_deref() == Ok("copy")
                && std::env::var("ENGINE_NP").as_deref() == Ok("split"),
            np_pairs: persistent_contacts
                && std::env::var("ENGINE_REUSE").as_deref() == Ok("copy")
                && std::env::var("ENGINE_NP").as_deref() == Ok("split")
                && std::env::var("ENGINE_PAIRS").as_deref() == Ok("np"),
            np_defer_limit: 0,
            persist_list_limit: 0,
            local_repair: std::env::var("ENGINE_REPAIR").as_deref() == Ok("local"),
            carry_rest: std::env::var("ENGINE_REST").as_deref() == Ok("carry"),
            persist_list: persistent_contacts
                && std::env::var("ENGINE_REUSE").as_deref() == Ok("copy")
                && std::env::var("ENGINE_PERSIST").as_deref() == Ok("list"),
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
            free_joints: Vec::new(),
            body_reach: Vec::new(),
            chip: format!("{} / {}", info.name, info.backend),
        })
    }

    pub fn chip(&self) -> &str {
        &self.chip
    }

    pub fn set_params(&mut self, params: AvbdParams) {
        self.params = params;
    }

    /// Replace the material table that `Collider::material` indexes (64 entries, all
    /// friction 0.5 until set).
    pub fn set_materials(&mut self, mats: &[Material]) {
        assert!(mats.len() <= 64, "the material table holds 64 entries");
        self.queue
            .write_buffer(&self.bufs[3], 0, bytemuck::cast_slice(mats));
    }

    pub fn add_body(&mut self, body: BodyState) -> u32 {
        self.add_bodies(std::slice::from_ref(&body)).start
    }

    /// Append a scene-sized batch with one upload for current poses and one for
    /// previous poses. A write per body can exhaust Metal staging allocations
    /// before the first submission on half-million-body scenes.
    pub fn add_bodies(&mut self, bodies: &[BodyState]) -> std::ops::Range<u32> {
        let start = self.body_count;
        let count = u32::try_from(bodies.len()).expect("body batch exceeds u32 index space");
        assert!(
            count <= self.capacity.max_bodies.saturating_sub(start),
            "body upload exceeds capacity: {start} existing + {count} new > {}",
            self.capacity.max_bodies
        );
        if count > 0 {
            let stride = std::mem::size_of::<BodyState>() as u64;
            let bytes = bytemuck::cast_slice(bodies);
            self.queue
                .write_buffer(&self.bufs[0], u64::from(start) * stride, bytes);
            // WGSL packs previous poses at gBodies[maxBodies + i].
            self.queue.write_buffer(
                &self.bufs[0],
                (u64::from(self.capacity.max_bodies) + u64::from(start)) * stride,
                bytes,
            );
            self.host_bodies.extend_from_slice(bodies);
            self.body_count += count;
        }
        start..self.body_count
    }

    pub fn add_collider(&mut self, col: Collider) -> u32 {
        self.add_colliders(std::slice::from_ref(&col)).start
    }

    /// Append collider records in one contiguous upload, preserving body indices.
    pub fn add_colliders(&mut self, colliders: &[Collider]) -> std::ops::Range<u32> {
        let start = self.collider_count;
        let count = u32::try_from(colliders.len()).expect("collider batch exceeds u32 index space");
        assert!(
            count <= self.capacity.max_colliders.saturating_sub(start),
            "collider upload exceeds capacity: {start} existing + {count} new > {}",
            self.capacity.max_colliders
        );
        if count > 0 {
            self.queue.write_buffer(
                &self.bufs[2],
                u64::from(start) * std::mem::size_of::<Collider>() as u64,
                bytemuck::cast_slice(colliders),
            );
            self.host_cols.extend_from_slice(colliders);
            self.collider_count += count;
            for col in colliders {
                if let Some(r) = engine_schema::collider_reach(col) {
                    let b = col.body as usize;
                    if self.body_reach.len() <= b {
                        self.body_reach.resize(b + 1, 0.0);
                    }
                    self.body_reach[b] = self.body_reach[b].max(r);
                }
            }
        }
        start..self.collider_count
    }

    pub fn joint_count(&self) -> u32 {
        self.joint_count
    }

    /// Replay a scene built elsewhere (e.g. `MetalWorld::snapshot`).
    pub fn load_snapshot(&mut self, snap: &WorldSnapshot) {
        if let Some(p) = snap.params {
            self.set_params(p);
        }
        if !snap.materials.is_empty() {
            self.set_materials(&snap.materials[..snap.materials.len().min(64)]);
        }
        self.add_bodies(&snap.bodies);
        self.add_colliders(&snap.colliders);
        self.add_joints(&snap.joints);
    }

    fn joint_offset(id: u32) -> u64 {
        u64::from(id) * std::mem::size_of::<Joint>() as u64
    }

    /// Write a raw GPU joint, reusing a released slot when one is free.
    pub fn add_joint(&mut self, joint: Joint) -> u32 {
        let id = match self.free_joints.pop() {
            Some(id) => id,
            None => {
                let id = self.joint_count;
                assert!(id < self.capacity.max_joints, "joint capacity exceeded");
                self.joint_count += 1;
                id
            }
        };
        self.queue.write_buffer(
            &self.bufs[5],
            Self::joint_offset(id),
            bytemuck::bytes_of(&joint),
        );
        id
    }

    /// Batch upload (scene import); slots follow the current count.
    pub fn add_joints(&mut self, joints: &[Joint]) -> std::ops::Range<u32> {
        let start = self.joint_count;
        let count = joints.len() as u32;
        assert!(
            count <= self.capacity.max_joints.saturating_sub(start),
            "joint capacity exceeded"
        );
        if count > 0 {
            self.queue.write_buffer(
                &self.bufs[5],
                Self::joint_offset(start),
                bytemuck::cast_slice(joints),
            );
            self.joint_count += count;
        }
        start..self.joint_count
    }

    pub fn add_joint_desc(&mut self, desc: &JointDesc) -> u32 {
        let mut j = desc.to_gpu();
        if desc.torque_arm <= 0.0 {
            let reach = |id: u32| {
                let dynamic = self
                    .host_bodies
                    .get(id as usize)
                    .is_some_and(|b| b.inv_mass > 0.0);
                if dynamic {
                    self.body_reach.get(id as usize).copied().unwrap_or(0.0)
                } else {
                    0.0
                }
            };
            j.torque_arm =
                engine_schema::default_torque_arm(reach(desc.body_a), reach(desc.body_b));
        }
        self.add_joint(j)
    }

    pub fn download_joints(&self) -> Vec<Joint> {
        self.read_buffer_range(5, 0, self.joint_count as usize)
    }

    pub fn read_joint(&self, id: u32) -> Joint {
        self.read_buffer_range(5, Self::joint_offset(id), 1)[0]
    }

    /// Release a joint: the pair collides again and the slot is reused.
    pub fn remove_joint(&mut self, id: u32) {
        // Released = broken with both ends on body 0 (matches MetalWorld).
        let head = [0u32, 0, engine_schema::JOINT_FLAG_BROKEN];
        self.queue.write_buffer(
            &self.bufs[5],
            Self::joint_offset(id),
            bytemuck::cast_slice(&head),
        );
        self.free_joints.push(id);
    }

    /// Joints the GPU broke since the last call (one readback). Slots are released.
    pub fn drain_broken_joints(&mut self) -> Vec<u32> {
        let broken: Vec<u32> = self
            .download_joints()
            .iter()
            .enumerate()
            .filter(|(_, j)| j.is_broken() && j.body_a != j.body_b)
            .map(|(i, _)| i as u32)
            .collect();
        for &id in &broken {
            self.remove_joint(id);
        }
        broken
    }

    /// Update a drive's targets in place without touching solver state.
    pub fn set_joint_drive(&mut self, id: u32, velocity: f32, target: f32) {
        let off = Self::joint_offset(id) + std::mem::offset_of!(Joint, drive_target) as u64;
        self.queue.write_buffer(
            &self.bufs[5],
            off,
            bytemuck::cast_slice(&[target, velocity]),
        );
    }

    pub fn body_y(&self, id: u32) -> f32 {
        self.read_body(id).position[1]
    }

    pub fn read_body(&self, id: u32) -> BodyState {
        self.read_body_range(id, 1)[0]
    }

    /// Download one snapshot, avoiding a GPU synchronization per body in
    /// scale diagnostics and cross-backend comparisons.
    pub fn download_bodies(&self) -> Vec<BodyState> {
        self.read_body_range(0, self.body_count)
    }

    fn read_body_range(&self, first: u32, count: u32) -> Vec<BodyState> {
        assert!(first <= self.body_count && count <= self.body_count - first);
        self.read_buffer_range(
            0,
            first as u64 * std::mem::size_of::<BodyState>() as u64,
            count as usize,
        )
    }

    /// Slow diagnostic readback: no two active endpoints may be solved in the
    /// same colour dispatch. Report the first offending body/manifold explicitly.
    pub fn validate_contact_coloring(&self) -> Result<(), String> {
        let bodies = self.download_bodies();
        let colors: Vec<u32> = self.read_buffer_range(12, 0, self.body_count as usize);
        let active = |id: usize| bodies[id].is_dynamic() && !bodies[id].is_sleeping();
        for id in 0..bodies.len() {
            if active(id) && colors[id] >= self.capacity.max_colors {
                return Err(format!("body {id} has unfinished/out-of-range colour {}; assigning unfinished bodies to colour zero introduces concurrent constraint writes", colors[id]));
            }
        }
        let manifolds: Vec<Manifold> =
            self.read_buffer_range(25, 0, self.last_counters.manifold_count as usize);
        for (id, m) in manifolds.iter().enumerate() {
            if m.contact_count == 0 {
                continue;
            }
            let a = self.host_cols[m.collider_a as usize].body as usize;
            let b = self.host_cols[m.collider_b as usize].body as usize;
            if active(a) && active(b) && colors[a] == colors[b] {
                return Err(format!("manifold {id}: active bodies {a} and {b} share colour {} and would race in the primal solver", colors[a]));
            }
        }
        Ok(())
    }

    fn read_buffer_range<T: bytemuck::Pod>(
        &self,
        binding: usize,
        offset: u64,
        count: usize,
    ) -> Vec<T> {
        if count == 0 {
            return Vec::new();
        }
        let size = count as u64 * std::mem::size_of::<T>() as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("body-read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(&self.bufs[binding], offset, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let bodies = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        bodies
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
        self.disp(
            &mut pass,
            "broadphase_clear",
            u64::from(self.grid_hash_words()) * if grid_split() { 2 } else { 1 },
            None,
        );
        self.disp(&mut pass, "broadphase_insert", n_cols, None);
        self.disp(&mut pass, "broadphase_query", n_cols, None);
        self.disp(&mut pass, self.work_args_entry(), 64, None);
        if self.carry_rest {
            // Previous manifolds (slot 72) are intact until narrowphase; recount
            // candidates (slot 64) after the carried resting pairs join them.
            self.disp_work(&mut pass, "carry_rest_pairs", 72, None);
            self.disp(&mut pass, self.work_args_entry(), 64, None);
        }

        // Same size the uniforms publish; records clear four words per slot.
        let prev_n = self
            .last_counters
            .contact_count
            .min(self.capacity.max_contacts)
            .max(256);
        let hash_n =
            u64::from(self.warm_hash_slots(prev_n)) * if self.warm_record_hash { 4 } else { 1 };
        let reuse = (self.params.flags & FLAG_REUSE_CONTACTS) != 0;
        if reuse {
            self.disp(&mut pass, "update_refs", n_bodies, None);
            self.disp(&mut pass, "contact_hash_clear", hash_n, None);
            let insert = if self.manifold_hash_insert {
                "contact_hash_insert_manifolds"
            } else {
                "contact_hash_insert"
            };
            self.disp_work(&mut pass, insert, 68, None);
        }
        // Narrowphase overwrites gManifolds; the manifold insert must precede it.
        assert!(
            reuse || !self.manifold_hash_insert,
            "ENGINE_HASH_INSERT=manifold requires contact reuse"
        );
        if self.np_pairs {
            self.disp_work(&mut pass, "narrowphase_defer_pairs_copy_persistent", 64, None);
            self.disp_work(&mut pass, "narrowphase_fresh_pairs_copy_persistent", 64, None);
        } else if self.np_split {
            self.disp_work(&mut pass, "narrowphase_defer_copy_persistent", 64, None);
            self.disp_work(&mut pass, "narrowphase_fresh_copy_persistent", 64, None);
        } else {
            self.disp_work(
                &mut pass,
                if self.reuse_copy {
                    "narrowphase_copy_persistent"
                } else {
                    "narrowphase"
                },
                64,
                None,
            );
        }
        self.disp(&mut pass, self.work_args_entry(), 64, None);
        if !reuse {
            self.disp(&mut pass, "contact_hash_clear", hash_n, None);
            self.disp_work(&mut pass, "contact_hash_insert", 68, None);
        }
        if self.fused_persistence && !self.persistent_contacts {
            self.disp_work(&mut pass, "clear_solver_contacts", 65, None);
        }
        if self.persist_list {
            // Slot 70 covers listed contacts; slot 71 runs the full scan only
            // when the list overflowed (both written after narrowphase).
            self.disp_work(&mut pass, "persist_list_copy_persistent", 70, None);
            self.disp_work(&mut pass, "persist_contacts_copy_persistent", 71, None);
        } else {
            self.disp_work(
                &mut pass,
                if self.fused_persistence {
                    "persist_manifolds_compact"
                } else if self.manifold_persistence {
                    "persist_manifolds"
                } else if self.reuse_copy {
                    "persist_contacts_copy_persistent"
                } else {
                    "persist_contacts"
                },
                if self.fused_persistence || self.manifold_persistence {
                    66
                } else {
                    65
                },
                None,
            );
        }
        // Persistent CSR reads solver pairs, so pack them before the graph unless
        // manifold persistence already did.
        let early_pair_pack =
            self.persistent_contacts && !self.fused_persistence && !self.manifold_persistence;
        if early_pair_pack && !self.np_pairs {
            self.disp_work(&mut pass, "pack_solver_contacts", 66, None);
        }
        if self.joint_count > 0 {
            self.disp(&mut pass, "warmstart_joints", n_joints, None);
        }
        self.disp(&mut pass, "build_awake_list", n_bodies, None);
        self.disp(&mut pass, self.work_args_entry(), 64, None);
        self.disp(&mut pass, "csr_clear", n_bodies + 1, None);
        self.disp_work(&mut pass, "csr_degree_contacts", 66, None);
        if self.joint_count > 0 {
            self.disp(&mut pass, "csr_degree_joints", n_joints, None);
        }
        if self.body_count <= 512 {
            self.disp(&mut pass, "csr_prefix", 1, None);
        } else {
            self.disp(&mut pass, "csr_prefix_blocks", (n_bodies + 3) / 4, None);
            self.disp(&mut pass, "csr_prefix_block_sums", 64, None);
            self.disp(&mut pass, "csr_prefix_add", n_bodies, None);
        }
        self.disp_work(&mut pass, "csr_scatter_contacts", 66, None);
        if self.joint_count > 0 {
            self.disp(&mut pass, "csr_scatter_joints", n_joints, None);
        }
        self.disp_work(&mut pass, "csr_sort", 67, None);
        self.disp_work(&mut pass, "init_colors", 67, None);
        self.disp(&mut pass, self.work_args_entry(), 64, None);
        // A colour count is not the JP dependency depth. The active 110k
        // ring still has uncoloured bodies after 16 rounds at step 60.
        let color_rounds = 32;
        if self.local_repair {
            // Slot 69 is zero when the local pass handled the repair list.
            self.disp(&mut pass, "color_bodies_local", 64, None);
        }
        for round in 0..color_rounds {
            self.disp_work(&mut pass, "color_bodies", 69, Some(round % 16));
        }
        self.disp(&mut pass, "count_color_tiles", n_bodies, None);
        self.disp(
            &mut pass,
            "prefix_color_tiles_parallel",
            u64::from(self.capacity.max_colors.min(32)) * 64,
            None,
        );
        self.disp(&mut pass, "scatter_color_tiles", n_bodies, None);
        self.disp(&mut pass, "write_color_args", 64, None);
        self.disp(&mut pass, "predict_bodies", n_bodies, None);

        // Dispatch the full palette; empty GPU indirect buckets cost no work.
        let max_c = self.capacity.max_colors.min(32);
        if self.compact_contacts && !self.fused_persistence && !early_pair_pack {
            if !self.persistent_contacts {
                self.disp_work(&mut pass, "clear_solver_contacts", 65, None);
            }
            self.disp_work(&mut pass, "pack_solver_contacts", 66, None);
        }
        for iter in 0..self.params.iterations {
            for color in 0..max_c.min(self.sparse_start) {
                self.disp_indirect(&mut pass, color);
            }
            if self.sparse_start < max_c {
                self.disp(
                    &mut pass,
                    if self.compact_contacts {
                        "solve_primal_sparse_compact"
                    } else {
                        "solve_primal_sparse"
                    },
                    64,
                    Some(self.sparse_start),
                );
            }
            self.disp_work(
                &mut pass,
                if self.compact_contacts {
                    "solve_dual_compact"
                } else {
                    "solve_dual"
                },
                65,
                Some(u32::from(iter + 1 == self.params.iterations)),
            );
        }
        self.disp(&mut pass, "extract_velocity_sleep", n_bodies, None);
        self.disp(&mut pass, "write_sleep_args", 64, None);
        if self.sleep_island_rounds > 0 {
            for _ in 0..self.sleep_island_rounds.min(16) {
                self.disp_work(&mut pass, "sleep_island_gate", 66, None);
            }
        }
        self.disp_work(&mut pass, "sleep_island_commit", 67, None);
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
        let p = self
            .pipes
            .get(name)
            .unwrap_or_else(|| panic!("missing {name}"));
        pass.set_pipeline(&p.pipeline);
        if p.dynamic_pass {
            let idx = pass_idx.unwrap_or(0);
            pass.set_bind_group(0, &p.bind, &[idx * PASS_ALIGN as u32]);
        } else {
            pass.set_bind_group(0, &p.bind, &[]);
        }
        let (x, y) = groups(threads);
        pass.dispatch_workgroups(x, y, 1);
    }

    fn disp_work<'a>(
        &'a self,
        pass: &mut wgpu::ComputePass<'a>,
        name: &str,
        slot: u32,
        pass_idx: Option<u32>,
    ) {
        let p = &self.pipes[name];
        pass.set_pipeline(&p.pipeline);
        if p.dynamic_pass {
            pass.set_bind_group(0, &p.bind, &[pass_idx.unwrap_or(0) * PASS_ALIGN as u32]);
        } else {
            pass.set_bind_group(0, &p.bind, &[]);
        }
        pass.dispatch_workgroups_indirect(&self.bufs[23], slot as u64 * 12);
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
        let p = &self.pipes[if self.compact_contacts {
            "solve_primal_color_compact"
        } else {
            "solve_primal_color"
        }];
        pass.set_pipeline(&p.pipeline);
        pass.set_bind_group(0, &p.bind, &[color * PASS_ALIGN as u32]);
        pass.dispatch_workgroups_indirect(&self.bufs[23], (color as u64) * 12);
    }

    /// Persist-list hosts write slots 70/71 from the reuse-copy variant (8em).
    fn work_args_entry(&self) -> &'static str {
        if self.persist_list {
            "write_work_args_copy_persistent"
        } else {
            "write_work_args"
        }
    }

    /// Broadphase grid heads: at least twice the colliders (three-avbd), so
    /// 506k gables avoids ~4 colliders per bucket; small scenes keep 131072.
    /// `ENGINE_GRID=fixed` keeps the fixed table as a comparison control;
    /// `ENGINE_GRID=split` (8ej) halves the cap so both split tables fit.
    fn grid_hash_words(&self) -> u32 {
        if std::env::var("ENGINE_GRID").as_deref() == Ok("fixed") {
            return 131072;
        }
        let region = constraint_id_words(&self.capacity) / if grid_split() { 2 } else { 1 };
        let cap = 1u32 << (31 - region.leading_zeros());
        (2 * self.collider_count)
            .max(131072)
            .next_power_of_two()
            .min(cap)
    }

    /// Warm hash size in slots: one word per slot (heads), or four-word record
    /// slots at ~2x manifolds, never below 4x bodies, within the pre-CSR region.
    fn warm_hash_slots(&self, prev_contacts: u32) -> u32 {
        let ids_len = constraint_id_words(&self.capacity);
        if self.warm_record_hash {
            (prev_contacts / 2)
                .max(4 * self.body_count)
                .max(1024)
                .next_power_of_two()
                .min(ids_len / 4)
        } else {
            const MAX_WARM_HASH: u32 = 1 << 21;
            prev_contacts
                .saturating_mul(4)
                .max(1024)
                .next_power_of_two()
                .min(ids_len)
                .min(MAX_WARM_HASH)
        }
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
        s[9] = self.warm_hash_slots(prev_n);
        s[15] = u32::from(self.warm_record_hash);
        s[10] = self.capacity.max_candidates;
        s[11] = cur;
        let mut log = 0u32;
        while (1u32 << log) < self.body_count.max(1) && log < 20 {
            log += 1;
        }
        s[12] = if log == 0 { 0 } else { (log - 1) & 1 };
        s[17] = self.steps_done;
        s[26] = 1; // incremental coloring: history row + compact work slot 69
        s[16] = self.grid_hash_words();
        s[20] = u32::from(grid_split());
        s[21] = self.np_defer_limit;
        s[22] = u32::from(self.persist_list);
        s[23] = self.persist_list_limit;
        s[27] = u32::from(self.local_repair);
        s[28] = u32::from(self.carry_rest);
        if self.manifold_hash_insert || self.carry_rest {
            s[13] = self.last_counters.manifold_count;
        }
        if self.manifold_hash_insert {
            s[14] = 1;
        }
        // CSR index region starts past the hash/id words inside gConstraintIds.
        s[24] = ids_len;
        s[4] = self.capacity.max_colors.min(32);
        self.queue
            .write_buffer(&self.bufs[21], 0, bytemuck::cast_slice(&s));

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
    s[0] = b(
        c.max_bodies.saturating_mul(2),
        std::mem::size_of::<engine_schema::BodyState>() as u64,
    );
    s[1] = b(
        c.max_bodies,
        std::mem::size_of::<engine_schema::BodyState>() as u64,
    );
    s[2] = b(c.max_colliders, 64);
    s[3] = b(64, 16);
    s[4] = b(c.max_contacts.saturating_mul(2), 128);
    s[5] = b(
        c.max_joints,
        std::mem::size_of::<engine_schema::Joint>() as u64,
    );
    s[6] = 256;
    s[7] = 256;
    s[8] = b(c.max_commands, 64);
    s[9] = b(c.max_bodies + c.max_colliders, 32);
    s[10] = b(c.max_candidates, 16);
    s[11] = b(c.max_bodies, 4);
    s[12] = b(c.max_bodies, 4);
    s[13] = b(c.max_bodies + 1, 4);
    s[14] = (u64::from(constraint_id_words(c)) + u64::from(c.max_bodies) + 1) * 4;
    s[15] = b(c.max_bodies * (c.max_colors + 2), 4);
    s[16] = b(c.max_colors * c.max_bodies.div_ceil(64).max(1), 4);
    s[17] = b(c.max_colors, 4);
    s[18] = b(c.max_hull_vertices, 16);
    s[19] = 65536 * 4;
    s[20] = b(c.max_bodies, 4);
    s[21] = 64 * 4;
    s[22] = 96 * PASS_ALIGN;
    s[23] = b(c.max_colors.max(73) * 3, 4);
    s[24] = b(c.max_colliders, 80);
    s[25] = b(c.max_candidates, 32);
    s[26] = b(c.max_bodies, 32);
    s[27] = b(c.max_contacts, 64);
    s[28] = b(c.max_candidates, 32);
    s
}

#[cfg(test)]
#[path = "gpu_tests.rs"]
mod tests;
