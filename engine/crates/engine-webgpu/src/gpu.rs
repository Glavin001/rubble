//! wgpu submission of the Slang→WGSL phase graph (same buffers/order as Metal).
//! This is the fair path vs three-avbd Chrome WebGPU.

use bytemuck::bytes_of;
use engine_exec::rigid_plan;
// Kernel fixtures in gpu_tests reach these through `super::*`.
#[cfg(test)]
use engine_exec::rigid_plan::{constraint_id_words, COLOR_ROUND_LIMIT};
use engine_exec::{ExecError, PhaseTimingsMs};
use engine_schema::convex::HullArena;
use engine_schema::*;
use std::collections::HashMap;

use engine_exec::rigid_plan::{BIND_COUNT, PASS_ALIGN};

struct Pipe {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    dynamic_pass: bool,
    /// Kept so bind groups can be rebuilt when a buffer is replaced (hull growth).
    layout: wgpu::BindGroupLayout,
    bindings: Vec<u32>,
}

fn make_bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    bindings: &[u32],
    bufs: &[wgpu::Buffer],
) -> wgpu::BindGroup {
    let resources: Vec<wgpu::BindGroupEntry> = bindings
        .iter()
        .map(|&b| wgpu::BindGroupEntry {
            binding: b,
            resource: if b == 22 {
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &bufs[22],
                    offset: 0,
                    size: std::num::NonZeroU64::new(16),
                })
            } else {
                bufs[b as usize].as_entire_binding()
            },
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &resources,
    })
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
    /// Split broadphase grid (8ej, HOST_S(20)); resolved once, not per step.
    grid_split: bool,
    /// `ENGINE_GRID=fixed`: the old 131072-head table (comparison control).
    grid_fixed: bool,
    /// `ENGINE_GRID_LEVELS` override of the hierarchical grid (None: convex scenes).
    grid_levels_mode: Option<bool>,
    /// Diagnostic dispatch filters (`WG_ONLY` / `WG_SKIP`), parsed once.
    wg_only: Option<Vec<String>>,
    wg_skip: Option<Vec<String>>,
    /// Primal colours dispatched one by one: last step's colour count plus this
    /// headroom; later colours run in the sparse tail (`None`: whole palette).
    color_headroom: Option<u32>,
    /// JP rounds dispatched: last step's rounds plus this headroom, then
    /// `color_bodies_tail` (`None`: all 32 rounds).
    color_round_headroom: Option<u32>,
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
    /// Any convex hull collider: schedules the convex_* narrowphase entries.
    has_convex: bool,
    /// Size levels of the hierarchical broadphase grid (0 = uniform grid).
    grid_levels: u32,
    /// (collider count, base cell, levels): recomputed only when colliders change.
    grid_cache: Option<(u32, f32, u32)>,
    /// Any collider carries a collision filter (HOST_S(21)).
    has_filters: bool,
    /// The last step's manifolds still describe the scene (whole-step reuse).
    manifolds_valid: bool,
    /// Records the hull buffer (binding 18) holds; grows on demand.
    hull_records_capacity: u32,
    /// The host wrote body state since the last step (`write_bodies`): step even
    /// if the last step left no body awake.
    wake_pending: bool,
    /// Step-start counter values, copied into binding 7 inside the step's
    /// encoder after the previous counts were forwarded to the host uniform.
    counter_upload: wgpu::Buffer,
    /// Offset of the last step's contact rows in binding 4 (`last_contacts`).
    last_contact_offset: u32,
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
fn grid_split_env() -> bool {
    !matches!(env_str("ENGINE_GRID").as_deref(), Some("scaled") | Some("fixed"))
}

/// ENGINE_AUDIT P0: every measured win is the default on every host; an
/// `ENGINE_*` variable set to its control value restores the old path.
fn env_str(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// `true` unless `name` is set to one of the listed control values.
fn env_default_on(name: &str, controls: &[&str]) -> bool {
    match env_str(name) {
        Some(v) => !controls.contains(&v.as_str()),
        None => true,
    }
}

fn env_list(name: &str) -> Option<Vec<String>> {
    env_str(name).map(|v| v.split(',').map(|n| n.trim().to_string()).collect())
}

/// Headroom over last step's count; `ENGINE_<X>=full` dispatches everything.
fn env_headroom(name: &str, default: u32) -> Option<u32> {
    match env_str(name).as_deref() {
        Some("full") => None,
        Some(v) => v.parse().ok().or(Some(default)),
        None => Some(default),
    }
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
        "convex_classify" => "convex_classify_persistent",
        "convex_manifold" => "convex_manifold_persistent",
        "reuse_all_manifolds" => "reuse_all_manifolds_persistent",
        _ => name,
    }
}

/// A high-performance adapter's device with its full limits.
pub fn default_device() -> anyhow::Result<(wgpu::AdapterInfo, wgpu::Device, wgpu::Queue)> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::GL;
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        ..Default::default()
    }))
    .map_err(|e| anyhow::anyhow!("no wgpu adapter: {e}"))?;
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("engine-webgpu-sim"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        ..Default::default()
    }))?;
    Ok((info, device, queue))
}

impl WgpuSim {
    pub fn new(capacity: WorldCapacity) -> anyhow::Result<Self> {
        // Persistent 32 B pair + 64 B point records are the default (8cz/8dg);
        // ENGINE_CONTACTS=canonical|compact keep the controls.
        // Unit-test fixtures read the canonical 128-byte contact layout.
        if cfg!(test) {
            return Self::new_with_compact_contacts(capacity, false);
        }
        let legacy_compact = env_str("ENGINE_COMPACT_CONTACTS").as_deref() == Some("1");
        let legacy_off = env_str("ENGINE_PERSISTENT_CONTACTS").as_deref() == Some("0");
        match env_str("ENGINE_CONTACTS").as_deref() {
            Some("canonical") => Self::new_with_compact_contacts(capacity, false),
            Some("compact") => Self::new_with_compact_contacts(capacity, true),
            _ if legacy_compact => Self::new_with_compact_contacts(capacity, true),
            _ if legacy_off => Self::new_with_compact_contacts(capacity, false),
            _ => Self::new_with_persistent_contacts(capacity),
        }
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
        let (info, device, queue) = default_device()?;
        Self::with_device_storage(capacity, compact_contacts, persistent_contacts, info, device, queue)
    }

    /// The simulation on an existing device (e.g. one shared with the GPU stress solver,
    /// so both read each other's buffers without a host round trip).
    pub fn with_device(
        capacity: WorldCapacity,
        info: wgpu::AdapterInfo,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> anyhow::Result<Self> {
        Self::with_device_storage(capacity, false, false, info, device, queue)
    }

    fn with_device_storage(
        capacity: WorldCapacity,
        compact_contacts: bool,
        persistent_contacts: bool,
        info: wgpu::AdapterInfo,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> anyhow::Result<Self> {
        let align = device.limits().min_storage_buffer_offset_alignment;
        if u64::from(align) > PASS_ALIGN {
            anyhow::bail!("storage offset alignment {align} > {PASS_ALIGN}");
        }

        let sizes = rigid_plan::buffer_sizes(&capacity, compact_contacts, persistent_contacts);
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
        // Pass table slots (256-byte aligned so dynamic offsets are legal).
        queue.write_buffer(&bufs[22], 0, &rigid_plan::pass_table());

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
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    }),
                ),
                module: &module,
                entry_point: None,
                compilation_options: Default::default(),
                cache: None,
            });
            let bindings: Vec<u32> = binds.iter().map(|b| b.index).collect();
            let bind = make_bind_group(&device, name, &layout, &bindings, &bufs);
            pipes.insert(
                name.clone(),
                Pipe {
                    pipeline,
                    bind,
                    dynamic_pass: binds.iter().any(|b| b.index == 22),
                    layout,
                    bindings,
                },
            );
        }

        let _ = mats;
        let hull_records = capacity.max_hull_vertices;
        // Every arena starts with the unit cube that box-versus-hull pairs reuse.
        let arena = HullArena::new();
        queue.write_buffer(&bufs[18], 0, bytemuck::cast_slice(arena.records()));
        let counter_upload = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("counter-upload"),
            size: std::mem::size_of::<StepCounters>() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        #[allow(unused_mut)]
        let mut sim = Self {
            counter_upload,
            compact_contacts,
            persistent_contacts,
            sparse_start: if capacity.max_bodies <= crate::WGSL_LOOP_CAP * 8 {
                env_str("ENGINE_SPARSE_COLORS")
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(32)
                    .min(32)
            } else {
                32
            },
            fused_persistence: compact_contacts
                && env_str("ENGINE_MANIFOLD_PERSISTENCE").as_deref() == Some("fused"),
            manifold_persistence: env_str("ENGINE_MANIFOLD_PERSISTENCE").as_deref() == Some("1"),
            // Record warm hash (8ed) is the default; ENGINE_HASH_INSERT=contact|manifold.
            manifold_hash_insert: env_default_on("ENGINE_HASH_INSERT", &["contact"]),
            warm_record_hash: env_default_on("ENGINE_HASH_INSERT", &["contact", "manifold"]),
            reuse_copy: persistent_contacts && env_default_on("ENGINE_REUSE", &["pass"]),
            np_split: persistent_contacts
                && env_default_on("ENGINE_REUSE", &["pass"])
                && env_default_on("ENGINE_NP", &["fused"]),
            np_defer_limit: 0,
            persist_list_limit: 0,
            local_repair: env_default_on("ENGINE_REPAIR", &["rounds"]),
            carry_rest: env_default_on("ENGINE_REST", &["drop"]),
            persist_list: persistent_contacts
                && env_default_on("ENGINE_REUSE", &["pass"])
                && env_default_on("ENGINE_PERSIST", &["scan"]),
            grid_split: grid_split_env(),
            grid_fixed: env_str("ENGINE_GRID").as_deref() == Some("fixed"),
            grid_levels_mode: match env_str("ENGINE_GRID_LEVELS").as_deref() {
                Some("off") => Some(false),
                Some("on") => Some(true),
                _ => None,
            },
            wg_only: env_list("WG_ONLY"),
            wg_skip: env_list("WG_SKIP"),
            color_headroom: env_headroom("ENGINE_COLOR_HEADROOM", 4),
            color_round_headroom: env_headroom("ENGINE_COLOR_ROUND_HEADROOM", 2),
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
            has_convex: false,
            grid_levels: 0,
            grid_cache: None,
            has_filters: false,
            manifolds_valid: false,
            hull_records_capacity: hull_records,
            wake_pending: false,
            last_contact_offset: 0,
        };
        // Kernel fixtures in the unit tests set the modes they exercise on top
        // of the pre-P0 controls; integration tests run the shipped defaults.
        #[cfg(test)]
        sim.use_control_modes();
        Ok(sim)
    }

    /// The pre-ENGINE_AUDIT configuration: per-contact warm hash, separate
    /// persistence, global colouring rounds, scaled grid, full palette.
    #[cfg(test)]
    /// Dispatch the whole colour palette every iteration instead of last step's
    /// colour count plus headroom (ENGINE_COLOR_HEADROOM=full), for tests.
    pub(crate) fn use_full_color_palette(&mut self) {
        self.color_headroom = None;
    }

    pub(crate) fn use_control_modes(&mut self) {
        self.manifold_hash_insert = false;
        self.warm_record_hash = false;
        self.reuse_copy = false;
        self.np_split = false;
        self.persist_list = false;
        self.local_repair = false;
        self.carry_rest = false;
        self.grid_split = false;
        self.color_headroom = None;
        self.color_round_headroom = None;
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
        self.queue.write_buffer(&self.bufs[3], 0, bytemuck::cast_slice(mats));
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
            self.manifolds_valid = false;
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
            self.has_convex |= colliders
                .iter()
                .any(|c| c.shape == ShapeType::ConvexHull as u32);
            self.has_filters |= colliders
                .iter()
                .any(engine_schema::convex::CollisionFilter::is_set);
            self.manifolds_valid = false;
            self.queue.write_buffer(
                &self.bufs[2],
                u64::from(start) * std::mem::size_of::<Collider>() as u64,
                bytemuck::cast_slice(colliders),
            );
            self.host_cols.extend_from_slice(colliders);
            self.collider_count += count;
        }
        start..self.collider_count
    }

    /// Upload the whole hull arena (binding 18), growing the buffer if needed.
    pub fn set_hulls(&mut self, arena: &mut HullArena) {
        arena.mark_all_dirty();
        self.sync_hulls(arena);
    }

    /// Upload the records the arena changed since the last sync (inserts and
    /// first-fit reuse of released blocks). Grows the GPU buffer to the next
    /// power of two and rebuilds the bind groups that reference it.
    pub fn sync_hulls(&mut self, arena: &mut HullArena) {
        let needed = arena.len() as u32;
        if needed > self.hull_records_capacity {
            let records = needed.next_power_of_two();
            let bytes = u64::from(records) * 16;
            assert!(
                bytes <= self.device.limits().max_storage_buffer_binding_size as u64,
                "hull arena of {records} records exceeds the device storage binding limit"
            );
            let grown = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("b18"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(
                &self.bufs[18],
                0,
                &grown,
                0,
                u64::from(self.hull_records_capacity) * 16,
            );
            self.queue.submit([enc.finish()]);
            self.bufs[18] = grown;
            self.hull_records_capacity = records;
            for (name, pipe) in self.pipes.iter_mut() {
                if pipe.bindings.contains(&18) {
                    pipe.bind = make_bind_group(&self.device, name, &pipe.layout, &pipe.bindings, &self.bufs);
                }
            }
        }
        let dirty = arena.take_dirty();
        let records = arena.records();
        for r in dirty {
            self.queue.write_buffer(
                &self.bufs[18],
                u64::from(r.start) * 16,
                bytemuck::cast_slice(&records[r.start as usize..r.end as usize]),
            );
        }
        self.manifolds_valid = false;
    }

    /// Replace a collider (e.g. swap a fractured chunk's hull or its filter).
    /// Release the old hull's arena reference yourself once nothing uses it.
    pub fn set_collider(&mut self, id: u32, col: Collider) {
        self.set_colliders(id, std::slice::from_ref(&col));
    }

    /// Replace the contiguous colliders `first..first + cols.len()` in one upload.
    pub fn set_colliders(&mut self, first: u32, cols: &[Collider]) {
        let end = first as usize + cols.len();
        assert!(end <= self.collider_count as usize, "colliders {first}..{end} out of range");
        if cols.is_empty() {
            return;
        }
        self.queue.write_buffer(
            &self.bufs[2],
            u64::from(first) * std::mem::size_of::<Collider>() as u64,
            bytemuck::cast_slice(cols),
        );
        use engine_schema::convex::CollisionFilter;
        // Any filter left: only a replaced filtered collider can clear the flag, and
        // only then is every collider scanned again.
        let cleared = self.host_cols[first as usize..end].iter().any(CollisionFilter::is_set);
        self.host_cols[first as usize..end].copy_from_slice(cols);
        self.has_convex |= cols.iter().any(|c| c.shape == ShapeType::ConvexHull as u32);
        if cols.iter().any(CollisionFilter::is_set) {
            self.has_filters = true;
        } else if cleared {
            self.has_filters = self.host_cols.iter().any(CollisionFilter::is_set);
        }
        self.grid_cache = None;
        self.manifolds_valid = false;
    }

    /// Dynamic body made of convex hulls (a single hull or a compound) in one
    /// input frame placed by `pos`/`rot`. Mass and inertia are the compound's,
    /// from uniform `density`. Uploads the updated arena.
    pub fn add_convex_body(
        &mut self,
        arena: &mut HullArena,
        hulls: &[engine_schema::convex::ConvexHull],
        pos: glam::Vec3,
        rot: glam::Quat,
        density: f32,
    ) -> u32 {
        let cooked = engine_schema::convex::cook_body(hulls, density);
        let (p, q) = cooked.body_pose(pos, rot);
        let body = self.add_body(BodyState::dynamic(p, q, cooked.mass, cooked.inertia));
        let colliders: Vec<Collider> = cooked
            .children
            .iter()
            .map(|child| Collider {
                body,
                shape: ShapeType::ConvexHull as u32,
                material: 0,
                generation: 0,
                local_pos: child.local_pos.to_array(),
                density,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: engine_schema::convex::convex_collider_params(
                    arena.insert(&child.hull).expect("hull arena capacity"),
                    1.0,
                    child.hull.bounding_radius(),
                ),
            })
            .collect();
        self.add_colliders(&colliders);
        self.sync_hulls(arena);
        body
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
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().expect("mapped range");
        let bodies = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        bodies
    }

    pub fn step(&mut self) -> Result<StepCounters, ExecError> {
        self.step_inner(false)
    }

    /// Encode and submit one step without waiting for it (pipelined hosts and
    /// benchmarks). Previous-step counts reach the next step through GPU copies,
    /// as in the browser host; `drain` waits and refreshes the counters.
    pub fn step_submit(&mut self) -> Result<(), ExecError> {
        self.step_core(false, false).map(|_| ())
    }

    /// Wait for every submitted step and refresh the counters (overflow check).
    pub fn drain(&mut self) -> Result<StepCounters, ExecError> {
        let counters = self.read_counters();
        self.last_counters = counters;
        self.manifolds_valid = counters.overflow_flags == 0;
        if counters.overflow_flags != 0 {
            return Err(ExecError::Overflow(counters.overflow_flags));
        }
        Ok(counters)
    }

    /// three-avbd `reuseManifolds`: skip bounds, broadphase and narrowphase and
    /// carry every previous manifold forward (persistence re-evaluates C0). Falls
    /// back to a full step when the scene changed since the last step.
    pub fn step_reuse_manifolds(&mut self) -> Result<StepCounters, ExecError> {
        let reuse = self.manifolds_valid;
        self.step_inner(reuse)
    }

    fn step_inner(&mut self, reuse_manifolds: bool) -> Result<StepCounters, ExecError> {
        self.step_core(reuse_manifolds, true)
    }

    fn step_core(&mut self, reuse_manifolds: bool, read: bool) -> Result<StepCounters, ExecError> {
        let wall = std::time::Instant::now();
        if self.steps_done > 0
            && self.last_counters.awake_count == 0
            && self.command_count == 0
            && self.body_count > 0
            && !self.wake_pending
        {
            self.steps_done += 1;
            self.last_timings.total_gpu = 0.0;
            return Ok(self.last_counters);
        }
        self.refresh_cell();
        self.write_host_uniforms_ex(!read);
        self.wake_pending = false;
        self.last_contact_offset = if self.contacts_parity == 0 {
            0
        } else {
            self.capacity.max_contacts
        };
        let modes = self.step_modes();
        let inputs = self.step_inputs();
        let cmds = rigid_plan::encode_step(&modes, &inputs, reuse_manifolds, !read);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("step"),
            });
        for c in &cmds {
            if let rigid_plan::Cmd::Copy { src, src_offset, dst, dst_offset, size } = *c {
                let from = if src == rigid_plan::COUNTER_UPLOAD {
                    &self.counter_upload
                } else {
                    &self.bufs[src as usize]
                };
                enc.copy_buffer_to_buffer(
                    from,
                    u64::from(src_offset),
                    &self.bufs[dst as usize],
                    u64::from(dst_offset),
                    u64::from(size),
                );
            }
        }
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("step"),
            timestamp_writes: None,
        });
        for c in &cmds {
            match *c {
                rigid_plan::Cmd::Copy { .. } => {}
                rigid_plan::Cmd::Dispatch { kernel, threads, pass: p } => {
                    self.disp(&mut pass, kernel, threads, Some(p));
                }
                rigid_plan::Cmd::Indirect { kernel, slot, pass: p } => {
                    if kernel.starts_with("solve_primal_color") {
                        self.disp_indirect(&mut pass, slot);
                    } else {
                        self.disp_work(&mut pass, kernel, slot, Some(p));
                    }
                }
            }
        }
        drop(pass);
        self.queue.submit([enc.finish()]);

        if !read {
            self.contacts_parity ^= 1;
            self.steps_done = self.steps_done.saturating_add(1);
            self.command_count = 0;
            self.last_timings.total_gpu = wall.elapsed().as_secs_f64() * 1000.0;
            return Ok(self.last_counters);
        }
        let counters = self.read_counters();
        self.contacts_parity ^= 1;
        self.steps_done = self.steps_done.saturating_add(1);
        self.command_count = 0;
        self.last_timings.total_gpu = wall.elapsed().as_secs_f64() * 1000.0;
        self.last_counters = counters;
        self.manifolds_valid = counters.overflow_flags == 0;
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
        if !self.wg_allows(name) {
            return;
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
        if !self.wg_allows("solve_primal_color") {
            return;
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

    /// This step's resolved modes for the shared step graph.
    fn step_modes(&self) -> rigid_plan::StepModes {
        rigid_plan::StepModes {
            compact_contacts: self.compact_contacts,
            persistent_contacts: self.persistent_contacts,
            manifold_persistence: self.manifold_persistence,
            fused_persistence: self.fused_persistence,
            manifold_insert: self.manifold_insert(),
            copy_reuse: self.copy_reuse(),
            split_np: self.split_np(),
            listed_persist: self.listed_persist(),
            carry_rest: self.carry_rest,
            local_repair: self.local_repair,
            grid_split: self.grid_split,
            sparse_start: self.sparse_start,
            color_headroom: self.color_headroom,
            color_round_headroom: self.color_round_headroom,
            tail_ok: self.capacity.max_bodies <= crate::WGSL_LOOP_CAP * 8,
        }
    }

    fn step_inputs(&self) -> rigid_plan::StepInputs {
        // Same size the uniforms publish; records clear four words per slot.
        let prev_n = self
            .last_counters
            .contact_count
            .min(self.capacity.max_contacts)
            .max(256);
        rigid_plan::StepInputs {
            body_count: self.body_count,
            collider_count: self.collider_count,
            joint_count: self.joint_count,
            joint_hash_slots: engine_schema::joint_hash_size(self.joint_count),
            has_convex: self.has_convex,
            reuse_contacts: self.reuse_on(),
            max_colors: self.capacity.max_colors,
            iterations: self.params.iterations,
            sleep_time: self.params.sleep_time,
            sleep_island_rounds: self.sleep_island_rounds,
            grid_hash_words: self.grid_hash_words(),
            warm_hash_words: u64::from(self.warm_hash_slots(prev_n))
                * if self.record_hash() { 4 } else { 1 },
            steps_done: self.steps_done,
            last: self.last_counters,
        }
    }

    /// Contact reuse (FLAG_REUSE_CONTACTS) gates every mode built on it.
    fn reuse_on(&self) -> bool {
        (self.params.flags & FLAG_REUSE_CONTACTS) != 0
    }
    fn record_hash(&self) -> bool {
        self.warm_record_hash && self.reuse_on()
    }
    fn manifold_insert(&self) -> bool {
        self.manifold_hash_insert && self.reuse_on()
    }
    fn copy_reuse(&self) -> bool {
        self.reuse_copy
    }
    fn split_np(&self) -> bool {
        self.np_split && self.copy_reuse()
    }
    /// The persist list is written by narrowphase only; convex kernels emit
    /// contacts after it, so hull scenes take the full persistence scan.
    fn listed_persist(&self) -> bool {
        self.persist_list && !self.has_convex
    }



    fn wg_allows(&self, name: &str) -> bool {
        self.wg_only.as_ref().is_none_or(|l| l.iter().any(|n| n == name))
            && !self.wg_skip.as_ref().is_some_and(|l| l.iter().any(|n| n == name))
    }

    /// Persist-list hosts write slots 70/71 from the reuse-copy variant (8em).
    #[cfg_attr(not(test), allow(dead_code))]
    fn work_args_entry(&self) -> &'static str {
        if self.listed_persist() {
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
        rigid_plan::grid_hash_words(&self.capacity, self.collider_count, self.grid_split, self.grid_fixed)
    }

    /// Warm hash size in slots: one word per slot (heads), or four-word record
    /// slots at ~2x manifolds, never below 4x bodies, within the pre-CSR region.
    fn warm_hash_slots(&self, prev_contacts: u32) -> u32 {
        rigid_plan::warm_hash_slots(&self.capacity, self.body_count, prev_contacts, self.record_hash())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn write_host_uniforms(&self) {
        self.write_host_uniforms_ex(false);
    }

    /// `pipelined`: counters go to the upload buffer and the step's encoder
    /// forwards last step's counts before resetting them (no host readback).
    fn write_host_uniforms_ex(&self, pipelined: bool) {
        let inputs = self.step_inputs();
        let s = rigid_plan::host_words(
            &self.capacity,
            &self.step_modes(),
            &inputs,
            &rigid_plan::HostState {
                contacts_parity: self.contacts_parity,
                grid_levels: self.grid_levels,
                has_filters: self.has_filters,
                record_hash: self.record_hash(),
                np_defer_limit: self.np_defer_limit,
                persist_list_limit: self.persist_list_limit,
            },
        );
        self.queue
            .write_buffer(&self.bufs[21], 0, bytemuck::cast_slice(&s));
        let c = rigid_plan::step_counters(&inputs);
        if pipelined {
            self.queue.write_buffer(&self.counter_upload, 0, bytes_of(&c));
        } else {
            self.queue.write_buffer(&self.bufs[7], 0, bytes_of(&c));
        }
        let host = rigid_plan::host_uniforms(&self.params, &s);
        self.queue.write_buffer(&self.bufs[6], 0, bytes_of(&host));
        // Colour counts (0..63) and per-colour max degrees (64..127).
        let zeros = vec![0u32; 128];
        self.queue
            .write_buffer(&self.bufs[17], 0, bytemuck::cast_slice(&zeros));
    }

    fn refresh_cell(&mut self) {
        let hier = self.grid_levels_mode.unwrap_or(self.has_convex);
        let (cell, levels) = match self.grid_cache {
            Some((n, cell, levels)) if hier && n == self.collider_count => (cell, levels),
            _ => {
                let r = rigid_plan::grid_cell(&self.host_cols, &self.host_bodies, hier);
                if hier {
                    self.grid_cache = Some((self.collider_count, r.0, r.1));
                }
                r
            }
        };
        self.params.grid_cell_size = cell;
        self.grid_levels = levels;
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
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().expect("mapped range");
        let c: StepCounters = *bytemuck::from_bytes(&data[..n as usize]);
        drop(data);
        staging.unmap();
        c
    }

    pub fn timings_ms(&self) -> f64 {
        self.last_timings.total_gpu
    }
}




#[path = "gpu_host_state.rs"]
mod host_state;
pub use host_state::SimSnapshot;

#[cfg(test)]
#[path = "gpu_tests.rs"]
mod tests;
