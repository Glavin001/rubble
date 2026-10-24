//! wgpu submission of the Slang→WGSL phase graph (same buffers/order as Metal).
//! This is the fair path vs three-avbd Chrome WebGPU.

use bytemuck::bytes_of;
use engine_exec::rigid_plan;
// Kernel fixtures in gpu_tests reach these through `super::*`.
#[cfg(test)]
use engine_exec::rigid_plan::{constraint_id_words, HostUniforms};
use engine_exec::{ExecError, PhaseTimingsMs};
use engine_schema::convex::HullArena;
use engine_schema::*;
use engine_vehicle::{
    FunctionalState, Ray, RayHit, VehicleBackend, VehicleCapacity, VehicleDesc, VehicleEvent,
    VehicleHandle, VehicleHost, VehicleInput, VehicleState, WheelDesc,
};
use std::collections::HashMap;

use crate::features::{
    check_limits, ContactStorage, EngineFeatures, GridLayout, ManifoldPersistence, SimOptions,
    WarmHash,
};
use engine_exec::rigid_plan::{BIND_COUNT, PASS_ALIGN};
use engine_softbody::{
    SoftBackend, SoftCapacity, SoftHandle, SoftHost, SoftSettings, SoftVehicleDesc, SpawnOptions,
};

/// Bindings of the render-pose buffers, after the shared plan's `BIND_COUNT`.
pub const POSES_BINDING: usize = BIND_COUNT;
pub const POSE_HEADER_BINDING: usize = BIND_COUNT + 1;
pub const MOVED_BINDING: usize = BIND_COUNT + 2;
/// Bytes of one [`crate::PoseRecord`].
pub const POSE_RECORD_BYTES: u64 = 32;

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
    /// The look-ahead contacts' largest reach per collider (m; 0: the kernels' 0.5).
    spec_reach_max: f32,
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
    /// One dispatch per soft particle colour (comparison control).
    soft_unfused: bool,
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
    /// Run `pose_pack` each step (render poses and the moved-body list).
    pose_output: bool,
    /// The last `step*` call ran the GPU step (false: skipped, nothing awake).
    last_step_ran: bool,
    sleep_island_rounds: u32,
    last_counters: StepCounters,
    last_timings: PhaseTimingsMs,
    steps_done: u32,
    contacts_parity: u32,
    host_bodies: Vec<BodyState>,
    host_cols: Vec<Collider>,
    /// Released joint slots, reused by `add_joint`.
    free_joints: Vec<u32>,
    /// Body commands queued for the next step (`push_body_commands`).
    pending_commands: Vec<GpuCommand>,
    /// Per-body collider reach (torque arm default).
    body_reach: Vec<f32>,
    chip: String,
    /// Any convex hull collider: schedules the convex_* narrowphase entries.
    has_convex: bool,
    /// Any heightfield collider: its pairs take the convex_* entries too.
    has_height_field: bool,
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
    /// Steps the counters in `last_counters` describe (pipelined steps leave
    /// them stale until `drain`; the asleep skip needs current counters).
    counted_steps: u32,
    /// Step-start counter values, copied into binding 7 inside the step's
    /// encoder after the previous counts were forwarded to the host uniform.
    counter_upload: wgpu::Buffer,
    /// Offset of the last step's contact rows in binding 4 (`last_contacts`).
    last_contact_offset: u32,
    /// Vehicle controller pool (`vehicle_step.slang`).
    vehicles: Option<VehicleHost>,
    /// Soft-body particles and beams (`soft_step.slang`, binding 32).
    soft: Option<SoftHost>,
    /// Event ring read cursor (`poll_joint_breaks`).
    events_seen: u32,
    /// Some body carries `BODY_CCD` (the continuous-collision passes run).
    has_ccd: bool,
    /// Some body is kinematic: its motion needs every step (no asleep skip).
    has_kinematic: bool,
    /// Impulse records for the next step (`apply_impulses`).
    impulse_queue: Vec<ImpulseRecord>,
    /// The step being encoded applies impulse records.
    impulses_this_step: bool,
    /// Character / query / projectile pool (binding 33).
    char_cap: engine_schema::character::CharacterCapacity,
    /// Characters: (inner kinematic body, inner collider).
    characters: Vec<(u32, u32)>,
    query_count: u32,
    projectile_count: u32,
    /// Event ring read cursor of `poll_motion_events`.
    motion_events_seen: u32,
    /// Measurement control (`set_fast_list`): long sweeps stay in the grid.
    fast_list_off: bool,
    /// Event ring read cursor (`poll_beam_breaks`).
    beam_events_seen: u32,
    /// Event ring read cursor (`poll_mount_breaks`).
    mount_events_seen: u32,
}

/// Motion bookkeeping of uploaded bodies: CCD and kinematic flags.
pub(crate) fn body_motion_flags(bodies: &[BodyState]) -> (bool, bool) {
    let ccd = bodies.iter().any(|b| b.flags & BODY_CCD != 0);
    let kin = bodies.iter().any(|b| b.flags & (BodyFlags::Kinematic as u32 | BodyFlags::Dynamic as u32) == BodyFlags::Kinematic as u32);
    (ccd, kin)
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
        "vehicle_loads" => "vehicle_loads_persistent",
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
        // Per-phase GPU timing (`RUBBLE_GPU_TIMING=1`) needs timestamp queries.
        required_features: if std::env::var("RUBBLE_GPU_TIMING").is_ok_and(|v| v == "1") {
            adapter.features() & wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        },
        required_limits: adapter.limits(),
        ..Default::default()
    }))?;
    Ok((info, device, queue))
}

impl WgpuSim {
    /// The shipped configuration on the default device, with the `ENGINE_*`
    /// control variables applied ([`SimOptions::from_env`]).
    pub fn new(capacity: WorldCapacity) -> anyhow::Result<Self> {
        // Unit-test fixtures read the canonical 128-byte contact layout.
        if cfg!(test) {
            return Self::new_with_compact_contacts(capacity, false);
        }
        Self::new_with_options(capacity, SimOptions::from_env())
    }

    pub fn new_with_compact_contacts(
        capacity: WorldCapacity,
        compact_contacts: bool,
    ) -> anyhow::Result<Self> {
        let contacts = if compact_contacts {
            ContactStorage::Compact
        } else {
            ContactStorage::Canonical
        };
        Self::new_with_options(
            capacity,
            SimOptions {
                contacts,
                ..SimOptions::from_env()
            },
        )
    }

    pub fn new_with_persistent_contacts(capacity: WorldCapacity) -> anyhow::Result<Self> {
        Self::new_with_options(
            capacity,
            SimOptions {
                contacts: ContactStorage::Persistent,
                ..SimOptions::from_env()
            },
        )
    }

    /// Canonical contacts with an explicit vehicle pool size.
    pub fn new_with_vehicles(
        capacity: WorldCapacity,
        vehicles: VehicleCapacity,
    ) -> anyhow::Result<Self> {
        Self::new_with_options(
            capacity,
            SimOptions {
                contacts: ContactStorage::Canonical,
                vehicles,
                ..SimOptions::from_env()
            },
        )
    }

    /// Canonical contacts with a vehicle pool and a soft-body arena (soft bodies
    /// need the canonical contact layout their particle kernel reads).
    pub fn new_with_soft(
        capacity: WorldCapacity,
        vehicles: VehicleCapacity,
        soft: SoftCapacity,
    ) -> anyhow::Result<Self> {
        Self::new_with_options(
            capacity,
            SimOptions {
                contacts: ContactStorage::Canonical,
                vehicles,
                soft: Some(soft),
                ..SimOptions::from_env()
            },
        )
    }

    /// Any configuration on the default device ([`default_device`]).
    pub fn new_with_options(capacity: WorldCapacity, options: SimOptions) -> anyhow::Result<Self> {
        let (info, device, queue) = default_device()?;
        Self::with_device_options(capacity, options, info, device, queue)
    }

    /// The simulation on an existing device (e.g. one shared with the GPU stress solver,
    /// so both read each other's buffers without a host round trip). Canonical
    /// contacts, the default vehicle pool, environment-resolved features.
    pub fn with_device(
        capacity: WorldCapacity,
        info: wgpu::AdapterInfo,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> anyhow::Result<Self> {
        Self::with_device_options(
            capacity,
            SimOptions {
                contacts: ContactStorage::Canonical,
                ..SimOptions::from_env()
            },
            info,
            device,
            queue,
        )
    }

    /// Every configuration on a caller's device: the one constructor the others wrap.
    /// Fails with a clear error when the device's limits cannot hold the capacity.
    pub fn with_device_options(
        mut capacity: WorldCapacity,
        options: SimOptions,
        info: wgpu::AdapterInfo,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> anyhow::Result<Self> {
        let SimOptions {
            contacts,
            vehicles: vehicle_capacity,
            soft,
            features,
            pose_output,
        } = options;
        anyhow::ensure!(
            soft.is_none() || contacts == ContactStorage::Canonical,
            "soft bodies need ContactStorage::Canonical (got {contacts:?})"
        );
        let compact_contacts = contacts.compact();
        let persistent_contacts = contacts.persistent();
        // One tyre joint per vehicle wheel slot rides on top of the user joints.
        capacity.max_joints += vehicle_capacity.max_wheels;
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
        let mut sim = Self::with_device_storage(
            capacity,
            compact_contacts,
            persistent_contacts,
            vehicle_capacity,
            features,
            pose_output,
            info,
            device,
            queue,
        )?;
        if let Some(soft) = soft {
            let bytes = soft.bytes();
            sim.check_buffer_fits(32, bytes)?;
            sim.bufs[32] = sim.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("b32 soft"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            sim.rebind_buffer(32);
            sim.soft = Some(SoftHost::new(soft));
        }
        Ok(sim)
    }

    /// A storage binding of `bytes` must fit the device's binding and buffer limits.
    fn check_buffer_fits(&self, binding: usize, bytes: u64) -> anyhow::Result<()> {
        check_limits(&self.device.limits(), binding, bytes)
    }

    #[allow(clippy::too_many_arguments)]
    fn with_device_storage(
        capacity: WorldCapacity,
        compact_contacts: bool,
        persistent_contacts: bool,
        vehicle_capacity: VehicleCapacity,
        features: EngineFeatures,
        pose_output: bool,
        info: wgpu::AdapterInfo,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> anyhow::Result<Self> {
        let align = device.limits().min_storage_buffer_offset_alignment;
        if u64::from(align) > PASS_ALIGN {
            anyhow::bail!("storage offset alignment {align} > {PASS_ALIGN}");
        }

        let sizes = rigid_plan::buffer_sizes(&capacity, compact_contacts, persistent_contacts);
        // Collider AABBs, then a 4-entry wheel-row block per vehicle and a
        // 4-entry ground block per wheel slot (two-body tyre rows); the vehicle
        // pool itself is binding 31.
        let aabb_tail = capacity.max_bodies + capacity.max_colliders;
        let mut sizes = sizes;
        sizes[9] +=
            u64::from(4 * (vehicle_capacity.max_vehicles + vehicle_capacity.max_wheels)) * 32;
        sizes[31] = vehicle_capacity.bytes().max(16);
        let char_cap = engine_schema::character::CharacterCapacity::default();
        sizes[33] = char_cap.bytes().max(16);
        let limits = device.limits();
        for (i, &bytes) in sizes.iter().enumerate() {
            check_limits(&limits, i, bytes)?;
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
        // Render-pose buffers (pose_pack.slang), past the shared plan's bindings:
        // full size only when the host asked for pose output.
        let pose_bytes = if pose_output {
            u64::from(capacity.max_bodies.max(1)) * POSE_RECORD_BYTES
        } else {
            256
        };
        for (binding, bytes) in [
            (POSES_BINDING, pose_bytes),
            (POSE_HEADER_BINDING, 16),
            (MOVED_BINDING, pose_bytes),
        ] {
            check_limits(&limits, binding, bytes)?;
            debug_assert_eq!(bufs.len(), binding);
            bufs.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("b{binding} poses")),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
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
            counted_steps: 0,
            counter_upload,
            compact_contacts,
            persistent_contacts,
            spec_reach_max: 0.0,
            sparse_start: if capacity.max_bodies <= crate::WGSL_LOOP_CAP * 8 {
                features.sparse_colors.min(32)
            } else {
                32
            },
            fused_persistence: compact_contacts
                && features.manifold_persistence == ManifoldPersistence::Fused,
            manifold_persistence: features.manifold_persistence == ManifoldPersistence::Manifolds,
            // Record warm hash (8ed) is the default; WarmHash::{Manifold, Contact} are controls.
            manifold_hash_insert: features.warm_hash != WarmHash::Contact,
            warm_record_hash: features.warm_hash == WarmHash::Record,
            reuse_copy: persistent_contacts && features.reuse_copy,
            np_split: persistent_contacts && features.reuse_copy && features.np_split,
            np_defer_limit: 0,
            persist_list_limit: 0,
            local_repair: features.local_repair,
            carry_rest: features.carry_rest,
            persist_list: persistent_contacts && features.reuse_copy && features.persist_list,
            // Split broadphase grids: awake dynamics in one table, sleepers and small
            // statics in a second (HOST_S(20)); awake queriers walk 14 cells.
            grid_split: features.grid == GridLayout::Split,
            grid_fixed: features.grid == GridLayout::Fixed,
            grid_levels_mode: features.grid_levels,
            color_headroom: features.color_headroom,
            color_round_headroom: features.color_round_headroom,
            soft_unfused: features.soft_unfused,
            wg_only: features.wg_only,
            wg_skip: features.wg_skip,
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
            pose_output,
            last_step_ran: false,
            sleep_island_rounds: 8,
            last_counters: StepCounters::default(),
            last_timings: PhaseTimingsMs::default(),
            steps_done: 0,
            contacts_parity: 0,
            host_bodies: Vec::new(),
            host_cols: Vec::new(),
            free_joints: Vec::new(),
            pending_commands: Vec::new(),
            body_reach: Vec::new(),
            chip: format!("{} / {}", info.name, info.backend),
            has_convex: false,
            has_height_field: false,
            grid_levels: 0,
            grid_cache: None,
            has_filters: false,
            manifolds_valid: false,
            hull_records_capacity: hull_records,
            wake_pending: false,
            last_contact_offset: 0,
            vehicles: Some(VehicleHost::new(vehicle_capacity, aabb_tail)),
            soft: None,
            events_seen: 0,
            has_ccd: false,
            has_kinematic: false,
            impulse_queue: Vec::new(),
            impulses_this_step: false,
            char_cap,
            characters: Vec::new(),
            query_count: 0,
            projectile_count: 0,
            motion_events_seen: 0,
            fast_list_off: false,
            beam_events_seen: 0,
            mount_events_seen: 0,
        };
        sim.write_character_header();
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

    /// The browser host's step modes (rigid_plan::StepModes::default): the
    /// control path for checks that must match world-host.mjs.
    pub fn use_control_modes(&mut self) {
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

    /// Overwrite one of the 64 material slots (friction is combined per contact).
    pub fn set_material(&mut self, index: u32, material: Material) {
        assert!(index < 64, "material slot out of range");
        self.queue
            .write_buffer(&self.bufs[3], u64::from(index) * 16, bytes_of(&material));
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
            self.manifolds_valid = false;
            self.note_motion_flags(bodies);
            // A body added to a world that fell asleep must still step: the
            // all-asleep skip only knows the last step's awake count.
            self.wake_pending = true;
        }
        start..self.body_count
    }

    pub(crate) fn note_motion_flags(&mut self, bodies: &[BodyState]) {
        let (ccd, kin) = body_motion_flags(bodies);
        self.has_ccd |= ccd;
        self.has_kinematic |= kin;
    }

    /// Queue an impulse `j` (N·s) at world point `point` on a dynamic body, applied
    /// on the GPU at the start of the next step (`apply_impulses`): no body
    /// upload, so manifold reuse stays on. Records on one body are summed in
    /// queue order; the body wakes.
    pub fn apply_impulse(&mut self, body: u32, j: [f32; 3], point: [f32; 3]) {
        self.queue_impulses(&[ImpulseRecord::at_point(body, j, point)]);
    }

    /// Queue a linear impulse through the body's centre of mass and an angular
    /// impulse (world, N·m·s), applied on the GPU at the next step's start.
    pub fn apply_linear_angular_impulse(&mut self, body: u32, linear: [f32; 3], angular: [f32; 3]) {
        self.queue_impulses(&[ImpulseRecord::linear_angular(body, linear, angular)]);
    }

    /// Set a kinematic body's velocity (linear m/s, angular rad/s) from the next
    /// step on, on the GPU: the body moves by it each step (contacts carry what
    /// rides on it) until changed.
    pub fn set_kinematic_velocity(&mut self, body: u32, linear: [f32; 3], angular: [f32; 3]) {
        self.queue_impulses(&[ImpulseRecord::set_velocity(body, linear, angular)]);
    }

    /// Queue impulse records for the next step. Panics past `IMPULSE_CAP` records
    /// in one step (an explicit limit, never a silent drop).
    pub fn queue_impulses(&mut self, records: &[ImpulseRecord]) {
        assert!(
            self.impulse_queue.len() + records.len() <= IMPULSE_CAP as usize,
            "more than {IMPULSE_CAP} impulse records in one step"
        );
        for r in records {
            assert!(r.body < self.body_count, "impulse on body {} of {}", r.body, self.body_count);
        }
        self.impulse_queue.extend_from_slice(records);
        self.wake_pending = true;
    }

    /// Upload the queued impulse records and their count (scratch words), before
    /// the step's uniforms (which schedule `apply_impulses` from it).
    fn upload_impulses(&mut self) {
        let n = self.impulse_queue.len() as u32;
        self.impulses_this_step = n > 0;
        if n > 0 {
            // Mode word: mode | queue index << 8 (the per-body sum order).
            let keyed: Vec<ImpulseRecord> = self
                .impulse_queue
                .iter()
                .enumerate()
                .map(|(k, r)| ImpulseRecord { mode: (r.mode & 255) | ((k as u32) << 8), ..*r })
                .collect();
            self.queue.write_buffer(&self.bufs[21], u64::from(IMPULSE_BASE) * 4, bytemuck::cast_slice(&keyed));
            // IMPULSE_COUNT and IMPULSE_FIRST are adjacent words.
            self.queue.write_buffer(&self.bufs[21], u64::from(IMPULSE_COUNT) * 4, bytemuck::bytes_of(&n));
            self.queue.write_buffer(&self.bufs[21], u64::from(IMPULSE_FIRST) * 4, bytemuck::bytes_of(&n));
        }
        self.impulse_queue.clear();
    }

    /// TOI casts that fell back to look-ahead in the last step (GJK budget):
    /// a diagnostic read of the scratch word.
    pub fn ccd_fallbacks(&self) -> u32 {
        self.read_buffer_range::<u32>(21, u64::from(CCD_FALLBACKS) * 4, 1)[0]
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
            self.has_height_field |= colliders
                .iter()
                .any(|c| c.shape == ShapeType::HeightField as u32);
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
                    pipe.bind = make_bind_group(
                        &self.device,
                        name,
                        &pipe.layout,
                        &pipe.bindings,
                        &self.bufs,
                    );
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

    /// The host's copy of a collider record.
    pub fn collider(&self, id: u32) -> Collider {
        self.host_cols[id as usize]
    }

    /// Replace a collider (e.g. swap a fractured chunk's hull or its filter).
    /// Release the old hull's arena reference yourself once nothing uses it.
    pub fn set_collider(&mut self, id: u32, col: Collider) {
        self.set_colliders(id, std::slice::from_ref(&col));
    }

    /// Bodies and colliders a GPU kernel already wrote at the next slots (bodies in both
    /// regions, as `add_bodies` writes them; colliders as `add_colliders`): the host's
    /// counts and mirrors follow, with no upload. For bodies made on the GPU (a split's
    /// children, stress-gpu `spawn.slang`).
    pub fn adopt_appended(
        &mut self,
        bodies: &[BodyState],
        colliders: &[Collider],
    ) -> (std::ops::Range<u32>, std::ops::Range<u32>) {
        let body_start = self.body_count;
        let count = u32::try_from(bodies.len()).expect("body batch exceeds u32 index space");
        assert!(
            count <= self.capacity.max_bodies.saturating_sub(body_start),
            "adopted bodies exceed capacity"
        );
        if count > 0 {
            self.host_bodies.extend_from_slice(bodies);
            self.body_count += count;
            self.manifolds_valid = false;
            self.note_motion_flags(bodies);
        }
        let col_start = self.collider_count;
        let count = u32::try_from(colliders.len()).expect("collider batch exceeds u32 index space");
        assert!(
            count <= self.capacity.max_colliders.saturating_sub(col_start),
            "adopted colliders exceed capacity"
        );
        if count > 0 {
            self.has_convex |= colliders
                .iter()
                .any(|c| c.shape == ShapeType::ConvexHull as u32);
            self.has_height_field |= colliders
                .iter()
                .any(|c| c.shape == ShapeType::HeightField as u32);
            self.has_filters |= colliders
                .iter()
                .any(engine_schema::convex::CollisionFilter::is_set);
            self.manifolds_valid = false;
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
        (body_start..self.body_count, col_start..self.collider_count)
    }

    /// Bodies a GPU kernel already rewrote in place, in both regions (as `write_bodies`):
    /// the host's mirror follows, with no upload.
    pub fn mirror_bodies(&mut self, first: u32, bodies: &[BodyState]) {
        let count = bodies.len() as u32;
        assert!(
            first <= self.body_count && count <= self.body_count - first,
            "mirror_bodies outside the bodies"
        );
        if count == 0 {
            return;
        }
        self.host_bodies[first as usize..(first + count) as usize].copy_from_slice(bodies);
        self.manifolds_valid = false;
        self.wake_pending = true;
        self.note_motion_flags(bodies);
    }

    /// A body a GPU kernel retired in place (no mass, motion or flags; stress-gpu
    /// `spawn_retire`): the host's mirror follows, with no upload.
    pub fn mirror_retire(&mut self, id: u32) {
        let b = &mut self.host_bodies[id as usize];
        b.inv_mass = 0.0;
        b.inv_inertia = [0.0; 3];
        b.linear_vel = [0.0; 3];
        b.angular_vel = [0.0; 3];
        b.flags = 0;
        self.manifolds_valid = false;
        self.wake_pending = true;
    }

    /// Colliders a GPU kernel already rewrote in place (as `set_colliders`): the host's
    /// mirror and flags follow, with no upload.
    pub fn mirror_colliders(&mut self, first: u32, cols: &[Collider]) {
        let end = first as usize + cols.len();
        assert!(
            end <= self.collider_count as usize,
            "colliders {first}..{end} out of range"
        );
        if cols.is_empty() {
            return;
        }
        use engine_schema::convex::CollisionFilter;
        let cleared = self.host_cols[first as usize..end]
            .iter()
            .any(CollisionFilter::is_set);
        self.host_cols[first as usize..end].copy_from_slice(cols);
        self.has_convex |= cols.iter().any(|c| c.shape == ShapeType::ConvexHull as u32);
        self.has_height_field |= cols.iter().any(|c| c.shape == ShapeType::HeightField as u32);
        if cols.iter().any(CollisionFilter::is_set) {
            self.has_filters = true;
        } else if cleared {
            self.has_filters = self.host_cols.iter().any(CollisionFilter::is_set);
        }
        self.grid_cache = None;
        self.manifolds_valid = false;
    }

    /// Renumber the bodies and colliders, keeping `keep_bodies` and `keep_colliders` (old
    /// ids, each ascending; every kept collider's body kept) at `0..` in the same order.
    /// Between steps only. Lossless for what the next step reads: the bodies (both pose
    /// regions) and colliders move, and last step's contacts and manifolds keep their
    /// warm-start data under the new ids (the order is kept, so every pair keeps its A/B
    /// order); rows that name a dropped collider are left dead. The colour history and
    /// the separating-axis cache, hints keyed by id, are cleared. Returns the old → new
    /// maps (`u32::MAX`: dropped). For appenders that retire slots instead of reusing
    /// them (stress-rubble's fracture splits).
    /// Characters, projectiles and queued impulses are renumbered too (a
    /// character's inner capsule must be kept; a dropped ground or hit reads
    /// as none, a queued impulse on a dropped body is discarded).
    pub fn compact(&mut self, keep_bodies: &[u32], keep_colliders: &[u32]) -> (Vec<u32>, Vec<u32>) {
        assert!(
            self.joint_count == 0 && self.vehicle_count() == 0,
            "compact: joints and vehicles name bodies and are not renumbered"
        );
        let mut body_map = vec![u32::MAX; self.body_count as usize];
        for (new, &old) in keep_bodies.iter().enumerate() {
            assert!(
                (old as usize) < body_map.len() && (new == 0 || keep_bodies[new - 1] < old),
                "compact: kept bodies ascending and in range"
            );
            body_map[old as usize] = new as u32;
        }
        let mut col_map = vec![u32::MAX; self.collider_count as usize];
        for (new, &old) in keep_colliders.iter().enumerate() {
            assert!(
                (old as usize) < col_map.len() && (new == 0 || keep_colliders[new - 1] < old),
                "compact: kept colliders ascending and in range"
            );
            col_map[old as usize] = new as u32;
        }
        let stride = std::mem::size_of::<BodyState>() as u64;
        let max_bodies = u64::from(self.capacity.max_bodies);
        let now: Vec<BodyState> = self.read_buffer_range(0, 0, self.body_count as usize);
        let prev: Vec<BodyState> =
            self.read_buffer_range(0, max_bodies * stride, self.body_count as usize);
        let bodies: Vec<BodyState> = keep_bodies.iter().map(|&b| now[b as usize]).collect();
        let prev: Vec<BodyState> = keep_bodies.iter().map(|&b| prev[b as usize]).collect();
        let cols: Vec<Collider> = keep_colliders
            .iter()
            .map(|&c| {
                let mut col = self.host_cols[c as usize];
                col.body = body_map[col.body as usize];
                assert!(
                    col.body != u32::MAX,
                    "compact: collider {c} kept but its body dropped"
                );
                col
            })
            .collect();
        // Last step's contacts (the half the next step reads as its previous one) and
        // manifolds: ids remapped; a row naming a dropped collider is dead.
        let remap = |a: &mut u32, b: &mut u32| -> bool {
            let (na, nb) = (
                col_map.get(*a as usize).copied().unwrap_or(u32::MAX),
                col_map.get(*b as usize).copied().unwrap_or(u32::MAX),
            );
            let live = na != u32::MAX && nb != u32::MAX;
            (*a, *b) = if live { (na, nb) } else { (0, 0) };
            live
        };
        let prev_off = if self.contacts_parity == 0 {
            0
        } else {
            u64::from(self.capacity.max_contacts)
        };
        let contact_size = std::mem::size_of::<Contact>() as u64;
        let n_contacts = self
            .last_counters
            .contact_count
            .min(self.capacity.max_contacts) as usize;
        let mut contacts: Vec<Contact> =
            self.read_buffer_range(4, prev_off * contact_size, n_contacts);
        for c in &mut contacts {
            if !remap(&mut c.collider_a, &mut c.collider_b) {
                c.alive = 0;
            }
        }
        let n_manifolds = self
            .last_counters
            .manifold_count
            .min(self.capacity.max_contacts) as usize;
        let mut manifolds: Vec<Manifold> = self.read_buffer_range(25, 0, n_manifolds);
        for m in &mut manifolds {
            if !remap(&mut m.collider_a, &mut m.collider_b) {
                m.contact_count = 0;
            }
        }
        let q = &self.queue;
        q.write_buffer(&self.bufs[0], 0, bytemuck::cast_slice(&bodies));
        q.write_buffer(
            &self.bufs[0],
            max_bodies * stride,
            bytemuck::cast_slice(&prev),
        );
        q.write_buffer(&self.bufs[2], 0, bytemuck::cast_slice(&cols));
        q.write_buffer(
            &self.bufs[4],
            prev_off * contact_size,
            bytemuck::cast_slice(&contacts),
        );
        q.write_buffer(&self.bufs[25], 0, bytemuck::cast_slice(&manifolds));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("compact"),
            });
        enc.clear_buffer(&self.bufs[15], 0, None);
        enc.clear_buffer(&self.bufs[30], 0, None);
        q.submit([enc.finish()]);
        self.body_count = bodies.len() as u32;
        self.collider_count = cols.len() as u32;
        self.host_bodies = bodies;
        self.body_reach = vec![0.0; self.host_bodies.len()];
        for col in &cols {
            if let Some(r) = engine_schema::collider_reach(col) {
                let b = col.body as usize;
                self.body_reach[b] = self.body_reach[b].max(r);
            }
        }
        self.has_convex = cols.iter().any(|c| c.shape == ShapeType::ConvexHull as u32);
        self.has_height_field = cols.iter().any(|c| c.shape == ShapeType::HeightField as u32);
        self.has_filters = cols
            .iter()
            .any(engine_schema::convex::CollisionFilter::is_set);
        self.host_cols = cols;
        self.grid_cache = None;
        self.manifolds_valid = false;
        self.wake_pending = true;
        self.remap_motion(&body_map, &col_map);
        (body_map, col_map)
    }

    /// Replace the contiguous colliders `first..first + cols.len()` in one upload.
    pub fn set_colliders(&mut self, first: u32, cols: &[Collider]) {
        let end = first as usize + cols.len();
        assert!(
            end <= self.collider_count as usize,
            "colliders {first}..{end} out of range"
        );
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
        let cleared = self.host_cols[first as usize..end]
            .iter()
            .any(CollisionFilter::is_set);
        self.host_cols[first as usize..end].copy_from_slice(cols);
        self.has_convex |= cols.iter().any(|c| c.shape == ShapeType::ConvexHull as u32);
        self.has_height_field |= cols.iter().any(|c| c.shape == ShapeType::HeightField as u32);
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

    /// A static heightfield body at `pos`/`rot` (the field is centred on it).
    /// Returns (body, collider).
    pub fn add_height_field(
        &mut self,
        arena: &mut HullArena,
        field: &engine_schema::heightfield::HeightField,
        pos: glam::Vec3,
        rot: glam::Quat,
        material: u32,
    ) -> Result<(u32, u32), engine_schema::heightfield::HeightFieldError> {
        let base = arena.insert_height_field(field)?;
        let mut support = BodyState::static_plane_support();
        support.position = pos.to_array();
        support.orientation = rot.to_array();
        support.initial_pos = support.position;
        support.initial_rot = support.orientation;
        let body = self.add_body(support);
        let collider = self.add_collider(Collider {
            body,
            shape: ShapeType::HeightField as u32,
            material,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: engine_schema::heightfield::height_field_collider_params(base, field),
        });
        self.sync_hulls(arena);
        Ok((body, collider))
    }

    /// Upload an edited heightfield (`rows` = the sample rows that changed).
    /// The collider's bounds follow the new height range; contacts are rebuilt.
    pub fn update_height_field(
        &mut self,
        arena: &mut HullArena,
        collider: u32,
        field: &engine_schema::heightfield::HeightField,
        rows: std::ops::Range<u32>,
    ) -> Result<(), engine_schema::heightfield::HeightFieldError> {
        let mut col = self.collider(collider);
        assert_eq!(col.shape, ShapeType::HeightField as u32, "collider {collider} is not a heightfield");
        let base = engine_schema::convex::convex_params_base(col.params);
        arena.update_height_field(base, field, rows)?;
        col.params = engine_schema::heightfield::height_field_collider_params(base, field);
        self.set_colliders(collider, std::slice::from_ref(&col));
        self.sync_hulls(arena);
        // Contact reuse (FLAG_REUSE_CONTACTS) keeps a pair's manifold until either
        // body's generation passes the step it was made in, and a static body's
        // never changes: mark the terrain as moved this step, so no manifold made
        // against the old surface is reused.
        let at = u64::from(col.body) * std::mem::size_of::<BodyState>() as u64
            + std::mem::offset_of!(BodyState, generation) as u64;
        self.queue
            .write_buffer(&self.bufs[0], at, bytemuck::bytes_of(&self.steps_done));
        if let Some(b) = self.host_bodies.get_mut(col.body as usize) {
            b.generation = self.steps_done;
        }
        Ok(())
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

    /// Diagnostic: run this step's graph only up to the end of the broadphase
    /// (bounds, grid, fast list) and return its candidate pairs (lower collider
    /// first). The step is not taken. The CSR build reuses the candidate buffer
    /// later in a step, so this is the only point where the pairs are readable.
    pub fn debug_broadphase_pairs(&mut self) -> Vec<(u32, u32)> {
        self.refresh_cell();
        self.write_host_uniforms_ex(false);
        let cmds = rigid_plan::encode_step(&self.step_modes(), &self.step_inputs(), false, false);
        let cut = cmds
            .iter()
            .position(|c| matches!(c, rigid_plan::Cmd::Phase { name: "broadphase" }))
            .expect("broadphase phase marker");
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("broadphase-only") });
        self.encode_dispatches(&mut enc, &cmds[..cut]);
        self.queue.submit([enc.finish()]);
        let c = self.read_counters();
        let n = c.candidate_count.min(self.capacity.max_candidates) as usize;
        let words: Vec<u32> = self.read_buffer_range(10, 0, n * 4);
        (0..n).map(|k| (words[4 * k].min(words[4 * k + 1]), words[4 * k].max(words[4 * k + 1]))).collect()
    }

    /// Diagnostic read of `count` words of a binding from word `first` (waits
    /// for the GPU). For tests and probes; never on a host's hot path.
    pub fn read_words(&self, binding: usize, first: u64, count: usize) -> Vec<u32> {
        self.read_buffer_range(binding, first * 4, count)
    }

    pub(crate) fn pose_output_enabled(&self) -> bool {
        self.pose_output
    }

    /// Whether the last `step*` call ran the GPU step (false: it was skipped
    /// because nothing was awake and nothing was written).
    pub fn last_step_ran(&self) -> bool {
        self.last_step_ran
    }

    pub(crate) fn buffer(&self, binding: usize) -> &wgpu::Buffer {
        &self.bufs[binding]
    }

    pub(crate) fn read_binding<T: bytemuck::Pod>(
        &self,
        binding: usize,
        offset: u64,
        count: usize,
    ) -> Vec<T> {
        self.read_buffer_range(binding, offset, count)
    }

    pub(crate) fn write_binding<T: bytemuck::Pod>(&self, binding: usize, offset: u64, data: &[T]) {
        self.queue
            .write_buffer(&self.bufs[binding], offset, bytemuck::cast_slice(data));
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
        engine_gpu_wait::recv(&rx).unwrap();
        let data = slice.get_mapped_range().expect("mapped range");
        let bodies = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        bodies
    }

    /// The largest reach (m) the look-ahead contacts (`FLAG_SPEC_VELOCITY`) give one
    /// collider; 0 keeps the kernels' default (0.5 m). A fast body's contacts are found
    /// within its step's motion up to this, so it needs no substeps to meet what it hits.
    pub fn set_spec_reach_max(&mut self, reach: f32) {
        self.spec_reach_max = reach;
    }

    /// Measurement control: `false` never schedules the fast broadphase list or
    /// the oversize ownership, so long sweeps use the 27-cell stencil as before
    /// the CCD audit (they then miss pairs; see tests/broadphase_oracle.rs).
    pub fn set_fast_list(&mut self, on: bool) {
        self.fast_list_off = !on;
    }

    /// Force the hierarchical grid on or off (`None`: convex scenes use it), as
    /// `ENGINE_GRID_LEVELS=on|off` does.
    pub fn set_grid_levels(&mut self, mode: Option<bool>) {
        self.grid_levels_mode = mode;
        self.grid_cache = None;
    }

    pub fn step(&mut self) -> Result<StepCounters, ExecError> {
        self.step_inner(false)
    }

    /// Encode and submit one step without waiting for it (pipelined hosts and
    /// benchmarks). Previous-step counts reach the next step through GPU copies,
    /// as in the browser host; `drain` waits and refreshes the counters.
    /// Planning hints for a host that never reads the counters back (pipelined steps): the
    /// previous step's contacts (sizes the warm-start hash; the count itself is forwarded on
    /// the GPU) and its colours (how many get their own dispatch; the rest run in the
    /// sparse tail, which solves them the same way). Leaves the asleep skip's bookkeeping
    /// alone: a hint never skips a step.
    pub fn set_plan_hints(&mut self, contacts: u32, colors: u32) {
        self.last_counters.contact_count = contacts;
        self.last_counters.color_count = colors;
    }

    /// Steps run so far (`HOST_S(17)`: a moved body's generation is the step it moved in).
    pub fn steps_done(&self) -> u32 {
        self.steps_done
    }

    pub fn step_submit(&mut self) -> Result<(), ExecError> {
        self.step_core(false, false).map(|_| ())
    }

    /// Wait for every submitted step and refresh the counters (overflow check).
    pub fn drain(&mut self) -> Result<StepCounters, ExecError> {
        let counters = self.read_counters();
        self.drained(counters)
    }

    /// `drain` with the counters (`counters_buffer`) read by the host along with other
    /// reads, after every submitted step finished: one round trip for the host's frame.
    pub fn drained(&mut self, counters: StepCounters) -> Result<StepCounters, ExecError> {
        self.last_counters = counters;
        self.counted_steps = self.steps_done;
        // Wheel queries walk this step's broadphase grid, so vehicle worlds
        // never reuse a whole step's manifolds.
        self.manifolds_valid = counters.overflow_flags == 0 && self.vehicle_count() == 0 && self.characters.is_empty();
        if counters.overflow_flags != 0 {
            return Err(ExecError::Overflow(counters.overflow_flags));
        }
        Ok(counters)
    }

    /// three-avbd `reuseManifolds`: skip bounds, broadphase and narrowphase and
    /// carry every previous manifold forward (persistence re-evaluates C0). Falls
    /// back to a full step when the scene changed since the last step.
    pub fn step_reuse_manifolds(&mut self) -> Result<StepCounters, ExecError> {
        // CCD scenes re-run collision every step: a reused step would keep last
        // step's time-of-impact contacts and skip ccd_register (stale TOI slots).
        let reuse = self.manifolds_valid && !self.has_ccd;
        self.step_inner(reuse)
    }

    fn step_inner(&mut self, reuse_manifolds: bool) -> Result<StepCounters, ExecError> {
        self.step_core(reuse_manifolds, true)
    }

    fn step_core(&mut self, reuse_manifolds: bool, read: bool) -> Result<StepCounters, ExecError> {
        let wall = web_time::Instant::now();
        // Skip only when the counters describe the latest step: pipelined steps
        // leave them stale until `drain`.
        let vehicles_dirty = self.vehicles.as_ref().is_some_and(|v| v.dirty);
        if self.steps_done > 0
            && self.counted_steps == self.steps_done
            && self.last_counters.awake_count == 0
            && self.command_count == 0
            && self.body_count > 0
            && !self.wake_pending
            && !vehicles_dirty
            && !self.has_kinematic
            && self.projectile_count == 0
        {
            self.steps_done += 1;
            self.counted_steps = self.steps_done;
            self.last_timings.total_gpu = 0.0;
            self.last_step_ran = false;
            return Ok(self.last_counters);
        }
        let host_t = std::env::var("RUBBLE_HOST_TIMING")
            .is_ok()
            .then(web_time::Instant::now);
        self.refresh_cell();
        let t_cell = host_t.map(|t| t.elapsed().as_secs_f64() * 1e3);
        self.last_step_ran = true;
        self.flush_commands();
        self.upload_impulses();
        self.write_host_uniforms_ex(!read);
        if let (Some(t), Some(c)) = (host_t, t_cell) {
            eprintln!(
                "rubble host: grid cell {c:.2} ms, uniforms {:.2} ms",
                t.elapsed().as_secs_f64() * 1e3 - c
            );
        }
        self.wake_pending = false;
        self.last_contact_offset = if self.contacts_parity == 0 {
            0
        } else {
            self.capacity.max_contacts
        };
        let prof = std::env::var("RUBBLE_STEP_PROFILE").is_ok().then(web_time::Instant::now);
        let modes = self.step_modes();
        let inputs = self.step_inputs();
        let cmds = rigid_plan::encode_step(&modes, &inputs, reuse_manifolds, !read);
        if let Some(t) = prof {
            eprintln!("rubble step profile: inputs+plan {:.3} ms since start {:.3}", t.elapsed().as_secs_f64() * 1e3, wall.elapsed().as_secs_f64() * 1e3);
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("step"),
            });
        for c in &cmds {
            if let rigid_plan::Cmd::Copy {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } = *c
            {
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
        // `RUBBLE_GPU_TIMING=1` (diagnostic): a timestamped pass per phase of the plan,
        // each phase's GPU time printed after the step (waits for it).
        let timing = std::env::var("RUBBLE_GPU_TIMING").is_ok_and(|v| v == "1")
            && self
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY);
        if timing {
            // A phase marker follows its commands: segment k is the k-th phase.
            let mut names: Vec<&str> = Vec::new();
            for c in &cmds {
                if let rigid_plan::Cmd::Phase { name } = *c {
                    names.push(name);
                }
            }
            names.push("tail");
            let queries = self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("step phases"),
                ty: wgpu::QueryType::Timestamp,
                count: 2 * names.len() as u32,
            });
            let mut segment = 0u32;
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("step"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &queries,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            for c in &cmds {
                match *c {
                    rigid_plan::Cmd::Copy { .. } => {}
                    rigid_plan::Cmd::Phase { .. } => {
                        drop(pass);
                        segment += 1;
                        pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("step"),
                            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                                query_set: &queries,
                                beginning_of_pass_write_index: Some(2 * segment),
                                end_of_pass_write_index: Some(2 * segment + 1),
                            }),
                        });
                    }
                    rigid_plan::Cmd::Dispatch {
                        kernel,
                        threads,
                        pass: p,
                    } => self.disp(&mut pass, kernel, threads, Some(p)),
                    rigid_plan::Cmd::Indirect {
                        kernel,
                        slot,
                        pass: p,
                    } => {
                        if kernel.starts_with("solve_primal_color") {
                            self.disp_indirect(&mut pass, slot);
                        } else {
                            self.disp_work(&mut pass, kernel, slot, Some(p));
                        }
                    }
                }
            }
            drop(pass);
            let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("step phases"),
                size: 16 * names.len() as u64,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("step phases read"),
                size: 16 * names.len() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            enc.resolve_query_set(&queries, 0..2 * names.len() as u32, &resolve, 0);
            enc.copy_buffer_to_buffer(&resolve, 0, &staging, 0, 16 * names.len() as u64);
            self.queue.submit([enc.finish()]);
            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            let t: Vec<u64> =
                bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
            let period = self.queue.get_timestamp_period() as f64 * 1e-6;
            let parts: Vec<String> = names.iter().enumerate().map(|(i, n)| format!("{n} {:.2}", t[2 * i + 1].saturating_sub(t[2 * i]) as f64 * period)).collect();
            let begin = t.iter().step_by(2).copied().filter(|&x| x > 0).min().unwrap_or(0);
            let end = t.iter().skip(1).step_by(2).copied().max().unwrap_or(0);
            let span = end.saturating_sub(begin) as f64 * period;
            eprintln!("rubble step GPU ms: span {span:.2}, {}", parts.join(", "));
            return self.finish_step(read, wall);
        }
        self.encode_dispatches(&mut enc, &cmds);
        let t_enc = host_t.map(|t| t.elapsed().as_secs_f64() * 1e3);
        self.queue.submit([enc.finish()]);
        if let (Some(t), Some(e)) = (host_t, t_enc) {
            let submitted = t.elapsed().as_secs_f64() * 1e3;
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            eprintln!("rubble host: encoded {e:.2} ms, submitted {submitted:.2} ms, done {:.2} ms ({} commands)", t.elapsed().as_secs_f64() * 1e3, cmds.len());
        }
        self.finish_step(read, wall)
    }

    /// The plan's dispatches in one compute pass of `enc` (its copies are encoded before).
    fn encode_dispatches(&self, enc: &mut wgpu::CommandEncoder, cmds: &[rigid_plan::Cmd]) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("step"),
            timestamp_writes: None,
        });
        let split = std::env::var("RUBBLE_PASS_PER_PHASE").is_ok_and(|v| v == "1");
        // `RUBBLE_SKIP_KERNELS=a,b` (diagnostic, A/B timing): leave those out.
        let skip = std::env::var("RUBBLE_SKIP_KERNELS").unwrap_or_default();
        let skipped = |k: &str| !skip.is_empty() && skip.split(',').any(|s| s == k);
        for c in cmds {
            match *c {
                rigid_plan::Cmd::Dispatch { kernel, .. } | rigid_plan::Cmd::Indirect { kernel, .. } if skipped(kernel) => continue,
                _ => {}
            }
            match *c {
                rigid_plan::Cmd::Phase { .. } if split => {
                    drop(pass);
                    pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("step"), timestamp_writes: None });
                }
                rigid_plan::Cmd::Copy { .. } | rigid_plan::Cmd::Phase { .. } => {}
                rigid_plan::Cmd::Dispatch {
                    kernel,
                    threads,
                    pass: p,
                } => {
                    self.disp(&mut pass, kernel, threads, Some(p));
                }
                rigid_plan::Cmd::Indirect {
                    kernel,
                    slot,
                    pass: p,
                } => {
                    if kernel.starts_with("solve_primal_color") {
                        self.disp_indirect(&mut pass, slot);
                    } else {
                        self.disp_work(&mut pass, kernel, slot, Some(p));
                    }
                }
            }
        }
    }

    /// `step_submit` encoded into the caller's `enc` instead of a submission of its own:
    /// a host that runs other work in the same frame (the stress stage) submits the frame
    /// once. The step's host writes (its uniforms) take effect at that submission, before
    /// its commands, as they would at its own.
    pub fn step_encode(&mut self, enc: &mut wgpu::CommandEncoder) -> Result<(), ExecError> {
        let wall = web_time::Instant::now();
        let vehicles_dirty = self.vehicles.as_ref().is_some_and(|v| v.dirty);
        if self.steps_done > 0
            && self.counted_steps == self.steps_done
            && self.last_counters.awake_count == 0
            && self.command_count == 0
            && self.body_count > 0
            && !self.wake_pending
            && !vehicles_dirty
            && !self.has_kinematic
            && self.projectile_count == 0
        {
            self.steps_done += 1;
            self.counted_steps = self.steps_done;
            self.last_timings.total_gpu = 0.0;
            self.last_step_ran = false;
            return Ok(());
        }
        self.refresh_cell();
        self.last_step_ran = true;
        self.flush_commands();
        self.upload_impulses();
        self.write_host_uniforms_ex(true);
        self.wake_pending = false;
        self.last_contact_offset = if self.contacts_parity == 0 {
            0
        } else {
            self.capacity.max_contacts
        };
        let modes = self.step_modes();
        let inputs = self.step_inputs();
        let cmds = rigid_plan::encode_step(&modes, &inputs, false, true);
        for c in &cmds {
            if let rigid_plan::Cmd::Copy {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } = *c
            {
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
        self.encode_dispatches(enc, &cmds);
        self.finish_step(false, wall).map(|_| ())
    }

    /// The step's host bookkeeping after its submission (and its counters, `read`).
    fn finish_step(
        &mut self,
        read: bool,
        wall: web_time::Instant,
    ) -> Result<StepCounters, ExecError> {
        if let Some(v) = self.vehicles.as_mut() {
            v.dirty = false;
        }
        self.impulses_this_step = false;
        if !read {
            self.contacts_parity ^= 1;
            self.steps_done = self.steps_done.saturating_add(1);
            self.command_count = 0;
            self.last_timings.total_gpu = wall.elapsed().as_secs_f64() * 1000.0;
            return Ok(self.last_counters);
        }
        let t_read = web_time::Instant::now();
        let counters = self.read_counters();
        if std::env::var("RUBBLE_STEP_PROFILE").is_ok() {
            eprintln!("rubble step profile: read {:.3} ms, total {:.3} ms", t_read.elapsed().as_secs_f64() * 1e3, wall.elapsed().as_secs_f64() * 1e3);
        }
        self.contacts_parity ^= 1;
        self.steps_done = self.steps_done.saturating_add(1);
        self.command_count = 0;
        self.last_timings.total_gpu = wall.elapsed().as_secs_f64() * 1000.0;
        self.last_counters = counters;
        self.counted_steps = self.steps_done;
        self.manifolds_valid = counters.overflow_flags == 0 && self.vehicle_count() == 0 && self.characters.is_empty();
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
            pose_output: self.pose_output,
        }
    }

    /// Vehicles in the pool (0 = the vehicle passes are not encoded).
    fn vehicle_count(&self) -> u32 {
        self.vehicles.as_ref().map_or(0, |v| v.vehicle_count())
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
            has_height_field: self.has_height_field,
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
            vehicle_count: self.vehicle_count(),
            wheel_count: self.vehicles.as_ref().map_or(0, |v| v.wheel_count()),
            motion: rigid_plan::MotionInputs {
                ccd: self.has_ccd,
                kinematic: self.has_kinematic,
                look_ahead: self.params.flags & FLAG_SPEC_VELOCITY != 0,
                impulses: self.impulses_this_step,
                projectiles: self.projectile_count,
                characters: self.characters.len() as u32,
                queries: self.query_count,
                fast_list_off: self.fast_list_off,
            },
            soft: self.soft_counts(),
            command_count: self.command_count,
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
    /// contacts after it, so hull and heightfield scenes take the full
    /// persistence scan.
    fn listed_persist(&self) -> bool {
        self.persist_list && !self.has_convex && !self.has_height_field
    }

    fn wg_allows(&self, name: &str) -> bool {
        self.wg_only
            .as_ref()
            .is_none_or(|l| l.iter().any(|n| n == name))
            && !self
                .wg_skip
                .as_ref()
                .is_some_and(|l| l.iter().any(|n| n == name))
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
        rigid_plan::grid_hash_words(
            &self.capacity,
            self.collider_count,
            self.grid_split,
            self.grid_fixed,
        )
    }

    /// Warm hash size in slots: one word per slot (heads), or four-word record
    /// slots at ~2x manifolds, never below 4x bodies, within the pre-CSR region.
    fn warm_hash_slots(&self, prev_contacts: u32) -> u32 {
        rigid_plan::warm_hash_slots(
            &self.capacity,
            self.body_count,
            prev_contacts,
            self.record_hash(),
        )
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn write_host_uniforms(&self) {
        self.write_host_uniforms_ex(false);
    }

    /// `pipelined`: counters go to the upload buffer and the step's encoder
    /// forwards last step's counts before resetting them (no host readback).
    fn write_host_uniforms_ex(&self, pipelined: bool) {
        let inputs = self.step_inputs();
        let mut s = rigid_plan::host_words(
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
        // Word 31: the look-ahead contacts' reach cap (FLAG_SPEC_VELOCITY), 0: the
        // kernels' default.
        s[31] = self.spec_reach_max.to_bits();
        self.queue
            .write_buffer(&self.bufs[21], 0, bytemuck::cast_slice(&s));
        let c = rigid_plan::step_counters(&inputs);
        if pipelined {
            self.queue
                .write_buffer(&self.counter_upload, 0, bytes_of(&c));
        } else {
            self.queue.write_buffer(&self.bufs[7], 0, bytes_of(&c));
        }
        let host = rigid_plan::host_uniforms(&self.params, &s);
        self.queue.write_buffer(&self.bufs[6], 0, bytes_of(&host));
        // Colour counts (0..63) and per-colour max degrees (64..127).
        let zeros = vec![0u32; 128];
        self.queue
            .write_buffer(&self.bufs[17], 0, bytemuck::cast_slice(&zeros));
        if self.pose_output {
            // pose_pack appends from 0 and scans this step's bodies.
            self.queue.write_buffer(
                &self.bufs[POSE_HEADER_BINDING],
                0,
                bytemuck::cast_slice(&[0u32, self.body_count, 0, 0]),
            );
        }
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
        engine_gpu_wait::recv(&rx).unwrap();
        let data = slice.get_mapped_range().expect("mapped range");
        let c: StepCounters = *bytemuck::from_bytes(&data[..n as usize]);
        drop(data);
        staging.unmap();
        c
    }

    pub fn timings_ms(&self) -> f64 {
        self.last_timings.total_gpu
    }

    /// Rebuild the bind groups that use `binding` after its buffer was replaced.
    fn rebind_buffer(&mut self, binding: u32) {
        for (name, pipe) in self.pipes.iter_mut() {
            if pipe.bindings.contains(&binding) {
                pipe.bind =
                    make_bind_group(&self.device, name, &pipe.layout, &pipe.bindings, &self.bufs);
            }
        }
    }

    // ---------------------------------------------------------------------
    // Soft bodies (slang/kernels/soft_step.slang)
    // ---------------------------------------------------------------------

    fn soft_counts(&self) -> rigid_plan::SoftCounts {
        let Some(h) = self.soft.as_ref() else {
            return Default::default();
        };
        let c = h.counts();
        rigid_plan::SoftCounts {
            bodies: c.bodies,
            beams: c.beams,
            colors: c.colors,
            // EngineFeatures::soft_unfused: one dispatch per particle colour (comparison control).
            max_color: if self.soft_unfused {
                c.max_color.max(rigid_plan::SOFT_FUSED_MAX_COLOR + 1)
            } else {
                c.max_color
            },
            groups: c.groups + 1,
            particles: c.particles,
            chebyshev: h.settings.chebyshev_rho > 0.0,
            particles_total: c.particles_total,
            triangles: c.triangles,
            contacts: c.contacts,
            dual: c.dual,
            sweeps: c.sweeps,
        }
    }

    /// The world was built with a soft-body arena.
    pub fn has_soft(&self) -> bool {
        self.soft.is_some()
    }

    pub fn soft_host(&self) -> &SoftHost {
        self.soft
            .as_ref()
            .expect("soft host (WgpuSim::new_with_soft)")
    }

    /// Spawn a soft vehicle (particles, node colliders, beams and, with wheels,
    /// its rigid core carrying the raycast wheels and drivetrain).
    pub fn add_soft_vehicle(
        &mut self,
        desc: &SoftVehicleDesc,
        position: [f32; 3],
        rotation: [f32; 4],
        opts: &SpawnOptions,
    ) -> Result<SoftHandle, String> {
        let mut soft = self
            .soft
            .take()
            .ok_or("soft host (WgpuSim::new_with_soft)")?;
        let mut vehicles = self.vehicles.take();
        let r = soft.add(self, vehicles.as_mut(), desc, position, rotation, opts);
        self.vehicles = vehicles;
        self.soft = Some(soft);
        self.manifolds_valid = false;
        r
    }

    pub fn set_soft_settings(&mut self, settings: SoftSettings) {
        let mut soft = self.soft.take().expect("soft host");
        soft.set_settings(self, settings);
        self.soft = Some(soft);
    }

    /// Register a thin static obstacle (sphere or capsule collider) that soft
    /// vehicle triangles collide with.
    pub fn add_soft_obstacle(&mut self, collider: u32) -> Result<(), String> {
        let mut soft = self.soft.take().expect("soft host");
        let r = soft.add_obstacle(self, collider);
        self.soft = Some(soft);
        r
    }

    /// Steering input of a soft vehicle's hydros (−1..1).
    pub fn set_soft_steer(&mut self, h: SoftHandle, steer: f32) {
        let mut soft = self.soft.take().expect("soft host");
        soft.set_steer(self, h, steer);
        self.soft = Some(soft);
    }

    /// Current beam records of a soft vehicle (GPU readback).
    pub fn read_soft_beams(&self, h: SoftHandle) -> Vec<engine_softbody::model::Beam> {
        self.soft_host().read_beams(self, h)
    }

    /// Particle states of a soft vehicle (GPU readback).
    pub fn read_soft_particles(&self, h: SoftHandle) -> Vec<BodyState> {
        let e = self.soft_host().entry(h);
        self.read_body_range(e.first_body, e.particles)
    }

    // ---------------------------------------------------------------------
    // Vehicles (slang/kernels/vehicle_step.slang)
    // ---------------------------------------------------------------------

    fn with_vehicles<R>(&mut self, f: impl FnOnce(&mut VehicleHost, &mut Self) -> R) -> R {
        let mut host = self.vehicles.take().expect("vehicle host");
        let r = f(&mut host, self);
        self.vehicles = Some(host);
        r
    }

    pub fn vehicle_host(&self) -> &VehicleHost {
        self.vehicles.as_ref().expect("vehicle host")
    }

    /// Spawn a vehicle (chassis + parked wheel/part bodies) at a chassis pose.
    pub fn add_vehicle(
        &mut self,
        desc: &VehicleDesc,
        position: [f32; 3],
        rotation: [f32; 4],
    ) -> Result<VehicleHandle, String> {
        self.with_vehicles(|h, sim| h.add_vehicle(sim, desc, position, rotation))
    }

    pub fn set_vehicle_input(&mut self, v: VehicleHandle, input: VehicleInput) {
        self.with_vehicles(|h, sim| h.set_input(sim, v, input));
    }

    pub fn set_functional_state(&mut self, v: VehicleHandle, state: FunctionalState) {
        self.with_vehicles(|h, sim| h.set_functional_state(sim, v, state));
    }

    pub fn detach_wheel(&mut self, v: VehicleHandle, wheel: u32) {
        self.with_vehicles(|h, sim| h.detach_wheel(sim, v, wheel));
    }

    pub fn disable_wheel(&mut self, v: VehicleHandle, wheel: u32) {
        self.with_vehicles(|h, sim| h.disable_wheel(sim, v, wheel));
    }

    pub fn detach_part(&mut self, v: VehicleHandle, part: u32) {
        self.with_vehicles(|h, sim| h.detach_part(sim, v, part));
    }

    pub fn disable_part(&mut self, v: VehicleHandle, part: u32) {
        self.with_vehicles(|h, sim| h.disable_part(sim, v, part));
    }

    pub fn add_wheel(&mut self, v: VehicleHandle, wheel: &WheelDesc) -> Result<u32, String> {
        self.with_vehicles(|h, sim| h.add_wheel(sim, v, wheel))
    }

    pub fn set_wheel_params(&mut self, v: VehicleHandle, wheel: u32, desc: &WheelDesc) {
        self.with_vehicles(|h, sim| h.set_wheel_params(sim, v, wheel, desc));
    }

    /// Wheel probes scan every collider instead of walking the broadphase grid
    /// (a reference oracle; the grid walk is the default).
    pub fn set_vehicle_full_scan(&mut self, on: bool) {
        self.with_vehicles(|h, sim| {
            h.full_scan_queries = on;
            let header = h.header();
            sim.write_slots(0, &header);
        });
    }

    pub fn wake_vehicle(&mut self, v: VehicleHandle) {
        self.with_vehicles(|h, sim| h.wake(sim, v));
    }

    pub fn vehicle_state(&self, v: VehicleHandle) -> VehicleState {
        self.vehicle_host().read_state(self, v)
    }

    /// The vehicle pool (binding 31), for hosts that copy vehicle state with
    /// each step ([`Self::vehicle_state_slots`]).
    pub fn vehicle_pool_buffer(&self) -> &wgpu::Buffer {
        &self.bufs[31]
    }

    /// The 16-byte pool slots `(first, count)` holding `v`'s state: its record,
    /// then its wheels.
    pub fn vehicle_state_slots(&self, v: VehicleHandle) -> [(u32, u32); 2] {
        self.vehicle_host().state_slots(v)
    }

    /// `v`'s state from copies of its [`Self::vehicle_state_slots`].
    pub fn decode_vehicle_state(
        &self,
        v: VehicleHandle,
        vehicle: &[[u32; 4]],
        wheels: &[[u32; 4]],
    ) -> VehicleState {
        self.vehicle_host().decode_state(v, vehicle, wheels)
    }

    /// Vehicle events since the previous call (severing, functional changes,
    /// contact and slide transitions).
    pub fn poll_vehicle_events(&mut self) -> Vec<VehicleEvent> {
        self.with_vehicles(|h, sim| h.poll_events(sim))
    }

    /// Joints that broke since the previous call (read from the event ring).
    pub fn poll_joint_breaks(&mut self) -> Vec<JointBreak> {
        let words: Vec<u32> = self.read_buffer_range(
            21,
            u64::from(EVENT_RING_BASE) * 4,
            (4 + 4 * EVENT_RING_CAPACITY) as usize,
        );
        let mut out = Vec::new();
        let lost = decode_joint_breaks(&words, &mut self.events_seen, &mut out);
        if lost > 0 {
            eprintln!("event ring overflow: {lost} events lost");
        }
        out
    }

    /// Vehicle part / wheel mounts that broke since the last call (event ring).
    pub fn poll_mount_breaks(&mut self) -> Vec<MountBreak> {
        let words: Vec<u32> = self.read_buffer_range(
            21,
            u64::from(EVENT_RING_BASE) * 4,
            (4 + 4 * EVENT_RING_CAPACITY) as usize,
        );
        let mut out = Vec::new();
        let lost = decode_mount_breaks(&words, &mut self.mount_events_seen, &mut out);
        if lost > 0 {
            eprintln!("event ring overflow: {lost} mount events lost");
        }
        out
    }

    /// What each part's and wheel's mount carried on the last step (`vehicle_loads`).
    pub fn vehicle_mount_loads(&self, v: VehicleHandle) -> Vec<engine_vehicle::MountLoad> {
        self.vehicle_host().read_mount_loads(self, v)
    }

    /// Soft-body beams that broke since the last call (event ring readback).
    /// Beam breaks since the last poll. Prefer [`Self::poll_soft_events`], which
    /// also reports the pieces they cut loose; use one or the other (both read
    /// the same cursor).
    pub fn poll_beam_breaks(&mut self) -> Vec<engine_softbody::BeamBreak> {
        let words: Vec<u32> = self.read_buffer_range(
            21,
            u64::from(EVENT_RING_BASE) * 4,
            (4 + 4 * EVENT_RING_CAPACITY) as usize,
        );
        let mut out = Vec::new();
        let lost = engine_softbody::decode_beam_breaks(
            &words,
            &mut self.beam_events_seen,
            EVENT_RING_CAPACITY,
            &mut out,
        );
        if lost > 0 {
            eprintln!("event ring overflow: {lost} beam events lost");
        }
        out
    }

    /// The wgpu device and queue, for integrations that draw from the
    /// simulation's buffers (`body_buffer`) without a readback.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The event ring (binding 21 from `EVENT_RING_BASE`): buffer, byte offset and
    /// byte size, for hosts that copy it into their own readback with each step.
    pub fn event_ring_range(&self) -> (&wgpu::Buffer, u64, u64) {
        (
            &self.bufs[21],
            u64::from(EVENT_RING_BASE) * 4,
            u64::from(4 + 4 * EVENT_RING_CAPACITY) * 4,
        )
    }

    /// Every event in a copy of the event ring (`event_ring_range`) since the last
    /// poll: joint, mount and beam breaks and the soft pieces those cut loose. It
    /// advances the same cursors as `poll_joint_breaks`, `poll_mount_breaks` and
    /// `poll_soft_events`, so a host uses this or those. `lost` counts records
    /// overwritten before they were read (soft bodies are then resynchronised).
    pub fn ring_events(&mut self, words: &[u32]) -> RingEvents {
        let mut joint_breaks = Vec::new();
        let lost_joints = decode_joint_breaks(words, &mut self.events_seen, &mut joint_breaks);
        let mut mount_breaks = Vec::new();
        let lost_mounts = decode_mount_breaks(words, &mut self.mount_events_seen, &mut mount_breaks);
        let soft = self.soft_events_from(words);
        RingEvents {
            joint_breaks,
            mount_breaks,
            lost: lost_joints.max(lost_mounts).max(soft.lost),
            soft,
        }
    }

    fn soft_events_from(&mut self, words: &[u32]) -> SoftEvents {
        let mut breaks = Vec::new();
        let lost = engine_softbody::decode_beam_breaks(
            words,
            &mut self.beam_events_seen,
            EVENT_RING_CAPACITY,
            &mut breaks,
        );
        let mut pieces = Vec::new();
        if self.soft.is_some() {
            if lost > 0 {
                let n = self.soft.as_ref().map_or(0, |h| h.entries.len());
                for s in 0..n {
                    let h = engine_softbody::SoftHandle(s as u32);
                    let beams = self.read_soft_beams(h);
                    pieces.extend(self.soft.as_mut().unwrap().sync_broken(h, &beams));
                }
            } else {
                pieces = self.soft.as_mut().unwrap().note_breaks(&breaks);
            }
        }
        SoftEvents {
            breaks,
            pieces,
            lost,
        }
    }

    /// Soft-body destruction events since the last poll: beam breaks, and the
    /// pieces they cut loose (`SoftHost::note_breaks`). After an event-ring
    /// overflow every soft body is resynchronised from a beam readback, so no
    /// piece is missed.
    pub fn poll_soft_events(&mut self) -> SoftEvents {
        let words: Vec<u32> = self.read_buffer_range(
            21,
            u64::from(EVENT_RING_BASE) * 4,
            (4 + 4 * EVENT_RING_CAPACITY) as usize,
        );
        self.soft_events_from(&words)
    }

    /// Whether soft body `h` is in rigid mode right now (`SoftSettings::rigid_mode`):
    /// riding its core with the particle solve skipped.
    pub fn read_soft_rigid(&self, h: SoftHandle) -> bool {
        let l = self.soft_host().capacity.layout();
        let slot = l.body_base + engine_softbody::BODY_STRIDE * h.0 + 3;
        let w: Vec<u32> = self.read_buffer_range(32, u64::from(slot) * 16, 4);
        w[0] == 1
    }

    /// A soft body's damage: beams bent and broken, plastic work, and the
    /// per-node deformation from its rest shape (rigid motion removed).
    pub fn read_soft_damage(&self, h: SoftHandle) -> engine_softbody::SoftDamage {
        let beams = self.read_soft_beams(h);
        let rest: Vec<glam::Vec3> = self
            .soft_host()
            .entry(h)
            .desc
            .nodes
            .iter()
            .map(|n| glam::Vec3::from_array(n.position))
            .collect();
        let now: Vec<glam::Vec3> = self
            .read_soft_particles(h)
            .iter()
            .map(|b| glam::Vec3::from_array(b.position))
            .collect();
        let detached = self.soft_host().entry(h).detached.clone();
        engine_softbody::damage_summary(&beams, Some((&rest, &now)), Some(&detached))
    }

    /// Batched scene ray casts against the current poses.
    /// Write a batch of ray queries for [`Self::encode_ray_queries`] (no
    /// readback). At most the vehicle pool's `max_rays`.
    pub fn write_ray_queries(&mut self, rays: &[Ray]) -> anyhow::Result<()> {
        let max = self.vehicle_host().max_rays();
        anyhow::ensure!(
            rays.len() as u32 <= max,
            "{} ray queries exceed VehicleCapacity::max_rays ({max})",
            rays.len()
        );
        self.with_vehicles(|h, sim| h.write_rays(sim, rays));
        // The next step runs even if everything sleeps, so its frame carries
        // the hits (a step with nothing awake changes nothing).
        self.wake_pending = true;
        Ok(())
    }

    /// Trace the written rays against the current poses, into `enc` (after a
    /// step's dispatches). Hits land in [`Self::ray_query_slots`] of the pool.
    pub fn encode_ray_queries(&self, enc: &mut wgpu::CommandEncoder) {
        let rays = self.vehicle_host().ray_count();
        if rays == 0 {
            return;
        }
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ray queries"),
            timestamp_writes: None,
        });
        self.disp(
            &mut pass,
            "build_aabbs",
            u64::from(self.collider_count.max(1)),
            None,
        );
        self.disp(&mut pass, "raycast_batch", u64::from(rays) * 64, None);
    }

    /// The pool slots `(first, count)` holding the written rays' hits.
    pub fn ray_query_slots(&self) -> (u32, u32) {
        self.vehicle_host().ray_slots()
    }

    /// Hits from a copy of [`Self::ray_query_slots`].
    pub fn decode_ray_hits(&self, slots: &[[u32; 4]]) -> Vec<Option<RayHit>> {
        self.vehicle_host().decode_ray_hits(slots)
    }

    pub fn raycast_batch(&mut self, rays: &[Ray]) -> Vec<Option<RayHit>> {
        if rays.is_empty() {
            return Vec::new();
        }
        self.with_vehicles(|h, sim| h.write_rays(sim, rays));
        self.write_host_uniforms();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rays"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rays"),
                timestamp_writes: None,
            });
            self.disp(
                &mut pass,
                "build_aabbs",
                u64::from(self.collider_count.max(1)),
                None,
            );
            self.disp(&mut pass, "raycast_batch", rays.len() as u64 * 64, None);
        }
        self.queue.submit([enc.finish()]);
        self.vehicle_host().read_ray_hits(self)
    }
}

impl SoftBackend for WgpuSim {
    fn write_soft_slots(&mut self, slot: u32, data: &[[u32; 4]]) {
        if !data.is_empty() {
            self.queue.write_buffer(
                &self.bufs[32],
                u64::from(slot) * 16,
                bytemuck::cast_slice(data),
            );
        }
    }

    fn read_soft_slots(&self, slot: u32, count: u32) -> Vec<[u32; 4]> {
        self.read_buffer_range(32, u64::from(slot) * 16, count as usize)
    }

    fn host_body(&self, index: u32) -> BodyState {
        self.host_bodies[index as usize]
    }

    fn host_collider(&self, index: u32) -> Collider {
        self.host_cols[index as usize]
    }

    fn joint_from_desc(&self, desc: &JointDesc) -> Joint {
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
        j
    }
}

impl VehicleBackend for WgpuSim {
    fn body_count(&self) -> u32 {
        self.body_count
    }

    fn joint_count(&self) -> u32 {
        self.joint_count
    }

    fn append_joints(&mut self, joints: &[engine_schema::Joint]) {
        if joints.is_empty() {
            return;
        }
        assert!(
            self.joint_count + joints.len() as u32 <= self.capacity.max_joints,
            "joint capacity exhausted"
        );
        let stride = std::mem::size_of::<engine_schema::Joint>() as u64;
        self.queue.write_buffer(
            &self.bufs[5],
            u64::from(self.joint_count) * stride,
            bytemuck::cast_slice(joints),
        );
        self.joint_count += joints.len() as u32;
    }

    fn collider_count(&self) -> u32 {
        self.collider_count
    }

    fn append_bodies(&mut self, bodies: &[BodyState]) {
        self.add_bodies(bodies);
    }

    fn append_colliders(&mut self, colliders: &[Collider]) {
        self.add_colliders(colliders);
    }

    fn write_body(&mut self, index: u32, body: &BodyState) {
        let stride = std::mem::size_of::<BodyState>() as u64;
        self.queue
            .write_buffer(&self.bufs[0], u64::from(index) * stride, bytes_of(body));
        self.queue.write_buffer(
            &self.bufs[0],
            (u64::from(self.capacity.max_bodies) + u64::from(index)) * stride,
            bytes_of(body),
        );
        self.host_bodies[index as usize] = *body;
    }

    fn write_collider(&mut self, index: u32, collider: &Collider) {
        self.queue.write_buffer(
            &self.bufs[2],
            u64::from(index) * std::mem::size_of::<Collider>() as u64,
            bytes_of(collider),
        );
        self.host_cols[index as usize] = *collider;
    }

    fn write_slots(&mut self, slot: u32, data: &[[u32; 4]]) {
        if !data.is_empty() {
            self.queue.write_buffer(
                &self.bufs[31],
                u64::from(slot) * 16,
                bytemuck::cast_slice(data),
            );
        }
    }

    fn write_curve_points(&mut self, byte_offset: u64, points: &[[f32; 2]]) {
        self.queue
            .write_buffer(&self.bufs[31], byte_offset, bytemuck::cast_slice(points));
    }

    fn read_slots(&self, slot: u32, count: u32) -> Vec<[u32; 4]> {
        self.read_buffer_range(31, u64::from(slot) * 16, count as usize)
    }
}

#[path = "gpu_host_state.rs"]
mod host_state;
#[path = "gpu_characters.rs"]
mod characters;
pub use characters::{CharacterHandle, MotionEvent};
pub use host_state::SimSnapshot;

#[cfg(test)]
#[path = "gpu_tests.rs"]
mod tests;

/// [`WgpuSim::poll_soft_events`] result.
/// Everything in the event ring since the last poll ([`WgpuSim::ring_events`]).
#[derive(Clone, Debug, Default)]
pub struct RingEvents {
    pub joint_breaks: Vec<JointBreak>,
    pub mount_breaks: Vec<MountBreak>,
    pub soft: SoftEvents,
    /// Records overwritten before they were read (the largest loss of any kind).
    pub lost: u32,
}

#[derive(Clone, Debug, Default)]
pub struct SoftEvents {
    pub breaks: Vec<engine_softbody::BeamBreak>,
    pub pieces: Vec<engine_softbody::PieceDetached>,
    /// Records overwritten in the event ring before this poll (bodies were
    /// resynchronised from a readback).
    pub lost: u32,
}
