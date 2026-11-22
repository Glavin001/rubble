//! Explicit soft-body vehicles in the World (`engine/docs/explicit-vehicles-in-world.md`):
//! the 2 kHz node–beam vehicles (`ExplicitBeamsGpu`) on the World's device, colliding two
//! way with its rigid bodies and loading its threshold destructibles, all on the GPU.
//!
//! Per World step (`World::step`):
//! 1. **emit** (before the rigid step): the last window's per-shape impulses become impulse
//!    records on the bodies their colliders belong to now (after the last split);
//! 2. the rigid step applies them at its start;
//! 3. **frame**: the vehicles' bounds, the proxy plan from the bodies after the rigid step,
//!    the explicit frame over the next window (its 2 kHz steps, the bodies integrated as
//!    proxies inside it), then the window's chunk loads into the threshold ingest's slot 1;
//! 4. the fracture takes those loads with the rigid contacts' (breaks, splits) before the
//!    next step: forward-only, nothing re-simulated.
//!
//! The vehicles' clock starts one window after they are added (their first frame covers the
//! window after the first rigid step), and then runs window for window with the World.

use crate::World;
use engine_exec::explicit_plan::coupling::world as cw;
use engine_softbody::explicit_beams::ExplicitBeams;
use engine_webgpu::{CouplingCapacity, CouplingLayout, ExplicitBeamsGpu, VEHICLE_COUPLING_WGSL};

pub use cw::{wheel_reach, PROXY_SLACK};

#[path = "explicit_vehicles_portable.rs"]
mod portable;

/// How the World couples its explicit vehicles ([`World::add_explicit_vehicles`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExplicitVehiclesDef {
    /// Proxy bodies, shapes and hull planes per vehicle per window. More fails the step.
    pub capacity: CouplingCapacity,
    /// The vehicles' collision filter (categories, mask): a collider couples when the
    /// filters accept each other (the World's ground is usually left out: the vehicles
    /// keep their own ground plane).
    pub categories: u16,
    pub mask: u16,
    /// Extra reach of the proxy plan (m) past the vehicle's and the body's travel.
    pub margin: f32,
    /// Vehicle–vehicle contact between the explicit vehicles (the solver's own).
    pub vehicle_contact: bool,
    /// Vehicle sleep (`engine_softbody::explicit_sleep`, an engine-wide setting): on by
    /// default in the World (large maps: a parked vehicle costs nothing per step); `None`
    /// steps every vehicle every step, bit for bit as without it (the crash lab).
    pub sleep: Option<engine_softbody::explicit_sleep::VehicleSleep>,
    /// Bit-exact results (an engine-wide setting, the owner's decision of 2026-11-19):
    /// on, the solver's kernels keep the CPU reference's IEEE order and its results are
    /// the CPU reference's bit for bit on every machine (testing and reproduction); off
    /// (the default; `RUBBLE_BIT_EXACT=1` turns it on), the same equations, contact
    /// rules, iterations and tolerances with the compilers free to fuse and reassociate
    /// arithmetic: faster, last-bit differences between machines
    /// (`ExplicitBeamsGpu::set_bit_exact`). At run time: [`World::set_explicit_bit_exact`].
    pub bit_exact: bool,
    /// Raycast wheels hit bodies (an engine-wide setting, on by default: a car drives on
    /// crates, bricks and ramps that are World bodies, as BeamNG's wheels meet whatever
    /// they stand on): the wheels' rays also meet the vehicle's rigid proxies, the
    /// nearest surface wins, and the wheel force's reaction goes onto the body at the
    /// contact point (`engine_softbody::explicit_drive::DriveConfig::see_bodies`). The
    /// proxy plan then reaches the wheels' rays ([`wheel_reach`]). Off: every step bit
    /// for bit as without it (the wheels see the ground and the static shapes only).
    /// Live: [`World::set_explicit_wheels_hit_bodies`].
    pub wheels_hit_bodies: bool,
}

impl Default for ExplicitVehiclesDef {
    fn default() -> Self {
        Self {
            capacity: CouplingCapacity {
                bodies: 16,
                shapes: 64,
                planes: 1024,
            },
            categories: 0xFFFF,
            mask: 0xFFFF,
            margin: 0.1,
            vehicle_contact: true,
            sleep: Some(engine_softbody::explicit_sleep::VehicleSleep::default()),
            bit_exact: engine_webgpu::bit_exact_default(),
            // Off by default until the browser's non-finite state with it on is fixed
            // (explicit-vehicles-in-world.md §13, "Open: Chrome").
            wheels_hit_bodies: false,
        }
    }
}

/// The threshold ingest's records, row counts, collider table and rows per slot.
type Ingest<'a> = Option<(&'a wgpu::Buffer, &'a wgpu::Buffer, &'a wgpu::Buffer, u32)>;

pub(crate) struct ExplicitVehicles {
    pub gpu: ExplicitBeamsGpu,
    layout: CouplingLayout,
    def: ExplicitVehiclesDef,
    /// The vehicles as added (spawn state: node counts, masses).
    pub sims: Vec<ExplicitBeams>,
    bgl: wgpu::BindGroupLayout,
    /// cpl_query's own layout (the broadphase buffers).
    bgl_q: wgpu::BindGroupLayout,
    /// The module's entries, in [`cw::Entry::ALL`]'s order.
    pipes: Vec<wgpu::ComputePipeline>,
    params: wgpu::Buffer,
    work: wgpu::Buffer,
    /// The work buffer's regions (`engine_exec`'s, the kernel's).
    pub(crate) work_layout: cw::WorkLayout,
    /// Tests ([`Self::replan_words`]): cpl_plan's serial walk.
    serial_plan: bool,
    /// Per vehicle its proxy flags (out of the broadphase), as uploaded.
    flags: Vec<u32>,
    /// The last window's plan source: the World broadphase (true) or the scan.
    pub(crate) last_query: bool,
    /// Measurement: plan by the scan even when the resting table is kept.
    force_scan: bool,
    /// Stand-ins for the threshold ingest's records, row counts and collider table.
    dummy: [wgpu::Buffer; 3],
    /// For renderers: per collision triangle (global nodes a, b, c, vehicle).
    pub tris: wgpu::Buffer,
    pub tri_count: u32,
    /// The window clock, and whether the last frame's impulses wait for their emit.
    pub clock: cw::Clock,
    status: std::sync::Arc<std::sync::Mutex<Option<u32>>>,
    status_buf: wgpu::Buffer,
    status_busy: std::sync::Arc<std::sync::atomic::AtomicBool>,
    read_pending: bool,
    /// The proxied list's indirect arguments (`cw::Buf::ListArgs`).
    list_args: wgpu::Buffer,
    /// Vehicle sleep: the bits the last delivered frame reported (all awake at the start).
    pub(crate) sleep_mask: Vec<u32>,
    /// Some vehicle was awake in the last delivered frame (or none delivered yet).
    pub(crate) any_awake: bool,
}

fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// What a replay of the shared plan's commands ([`cw::Cmd`]) records them with.
struct Replay<'a> {
    pipes: &'a [wgpu::ComputePipeline],
    work: &'a wgpu::Buffer,
    list_args: &'a wgpu::Buffer,
    status: &'a wgpu::Buffer,
    group: &'a wgpu::BindGroup,
    qgroup: &'a wgpu::BindGroup,
}

impl Replay<'_> {
    fn buf(&self, b: cw::Buf) -> &wgpu::Buffer {
        match b {
            cw::Buf::Work => self.work,
            cw::Buf::ListArgs => self.list_args,
            cw::Buf::Status => self.status,
        }
    }

    /// Record `cmds` into `enc`; `frame` records the explicit frame at [`cw::Cmd::Frame`].
    fn run(
        &self,
        sim: &mut engine_webgpu::WgpuSim,
        enc: &mut wgpu::CommandEncoder,
        cmds: &[cw::Cmd],
        frame: &mut dyn FnMut(&mut engine_webgpu::WgpuSim, &mut wgpu::CommandEncoder),
    ) {
        let mut i = 0;
        while i < cmds.len() {
            match cmds[i] {
                cw::Cmd::Clear { offset, size } => enc.clear_buffer(self.work, offset, Some(size)),
                cw::Cmd::Copy { src, src_offset, dst, dst_offset, size } => {
                    enc.copy_buffer_to_buffer(self.buf(src), src_offset, self.buf(dst), dst_offset, size)
                }
                cw::Cmd::Mark(name) => {
                    if let Some(m) = sim.marks() {
                        m.mark(enc, name);
                    }
                }
                cw::Cmd::Frame => frame(sim, enc),
                cw::Cmd::Pass { label, segment, before_step } => {
                    // With the World's GPU timing on: the coupling's passes as segments
                    // of the step's GPU times (resolved and read with its readback).
                    let seg = match segment {
                        Some(name) if before_step => sim.host_segment_before_step(name),
                        Some(name) => sim.host_segment(name),
                        None => None,
                    };
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(label),
                        timestamp_writes: sim.segment_writes(seg, true, true),
                    });
                    let mut bound_query = None;
                    i += 1;
                    while let Some(&(cw::Cmd::Dispatch { entry, .. } | cw::Cmd::Indirect { entry })) = cmds.get(i) {
                        let query = entry == cw::Entry::Query;
                        if bound_query != Some(query) {
                            pass.set_bind_group(0, if query { self.qgroup } else { self.group }, &[]);
                            bound_query = Some(query);
                        }
                        pass.set_pipeline(&self.pipes[entry.index()]);
                        match cmds[i] {
                            cw::Cmd::Dispatch { groups, .. } => pass.dispatch_workgroups(groups, 1, 1),
                            _ => pass.dispatch_workgroups_indirect(self.list_args, 0),
                        }
                        i += 1;
                    }
                    continue;
                }
                cw::Cmd::Dispatch { .. } | cw::Cmd::Indirect { .. } => unreachable!("a coupling dispatch outside a pass"),
            }
            i += 1;
        }
    }
}

impl ExplicitVehicles {
    pub fn new(world: &World, vehicles: &[ExplicitBeams], def: &ExplicitVehiclesDef) -> anyhow::Result<Self> {
        anyhow::ensure!(!vehicles.is_empty(), "explicit vehicles: none given");
        let c = def.capacity;
        anyhow::ensure!(
            c.bodies <= c.shapes && c.shapes <= 1024 && c.planes < 1 << 16,
            "explicit vehicles: at most 1024 shapes (and no more bodies) and 65535 hull planes per vehicle, not {c:?}"
        );
        let device = world.sim.device();
        let queue = world.sim.queue();
        let mut gpu = ExplicitBeamsGpu::with_settings(device, queue, vehicles, Some(def.capacity), def.sleep, def.bit_exact);
        gpu.set_vehicle_contact(def.vehicle_contact);
        gpu.set_coupling_on(true);
        gpu.set_wheels_see_bodies(def.wheels_hit_bodies);
        let layout = gpu.coupling().expect("coupling room");
        let entries = |set: &[(u32, cw::Access)]| -> Vec<wgpu::BindGroupLayoutEntry> {
            set.iter()
                .map(|&(binding, access)| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: match access {
                            cw::Access::Uniform => wgpu::BufferBindingType::Uniform,
                            cw::Access::Read => wgpu::BufferBindingType::Storage { read_only: true },
                            cw::Access::Write => wgpu::BufferBindingType::Storage { read_only: false },
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect()
        };
        // cpl_query (the World broadphase's resting table and moved lists) has its own
        // layout: with them every pass would bind 14 storage buffers, past WebGPU's
        // per-stage limit (maxStorageBuffersPerShaderStage, 10 in browsers).
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vehicle-coupling"),
            entries: &entries(cw::MAIN_SET),
        });
        let bgl_q = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vehicle-coupling-query"),
            entries: &entries(cw::QUERY_SET),
        });
        let pl_q = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vehicle-coupling-query"),
            bind_group_layouts: &[Some(&bgl_q)],
            immediate_size: 0,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vehicle-coupling"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let module = engine_webgpu::explicit_shader_module(device, wgpu::ShaderModuleDescriptor {
            label: Some("vehicle_coupling"),
            source: wgpu::ShaderSource::Wgsl(engine_webgpu::uniformize_workgroup_lanes(&VEHICLE_COUPLING_WGSL).into()),
        });
        let pipes = cw::Entry::ALL
            .iter()
            .map(|e| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(e.name()),
                    layout: Some(if *e == cw::Entry::Query { &pl_q } else { &pl }),
                    module: &module,
                    entry_point: Some(e.name()),
                    compilation_options: engine_webgpu::explicit_compilation_options(device),
                    cache: None,
                })
            })
            .collect();
        use wgpu::util::DeviceExt;
        let veh: Vec<[u32; 4]> = vehicles.iter().enumerate().map(|(i, v)| cw::vehicle_row(gpu.node_offset(i), v)).collect();

        let mut tris: Vec<[u32; 4]> = Vec::new();
        for (i, v) in vehicles.iter().enumerate() {
            let off = gpu.node_offset(i);
            tris.extend(v.triangles.iter().map(|t| [off + t.nodes[0], off + t.nodes[1], off + t.nodes[2], i as u32]));
        }
        let tri_count = tris.len() as u32;
        tris.push([0; 4]);
        let tris = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("explicit-vehicle-triangles"),
            contents: bytemuck::cast_slice(&tris),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let work_layout = cw::WorkLayout::new(vehicles.len() as u32, world.sim.capacity().max_bodies, def.capacity.shapes);
        let work = storage(device, "vehicle-coupling-work", work_layout.bytes());
        // The proxied list's indirect arguments: its count copied after the plan (a buffer
        // bound for storage cannot also be a dispatch's indirect arguments).
        let list_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vehicle-coupling-list-args"),
            contents: bytemuck::cast_slice(&cw::LIST_ARGS_INIT),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        });
        queue.write_buffer(&work, work_layout.table, bytemuck::cast_slice(&veh));
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vehicle-coupling-params"),
            size: std::mem::size_of::<cw::Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let status_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vehicle-coupling-status"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            gpu,
            layout,
            def: *def,
            sims: vehicles.to_vec(),
            bgl,
            bgl_q,
            pipes,
            params,
            work,
            work_layout,
            serial_plan: false,
            flags: vec![0; vehicles.len()],
            last_query: false,
            force_scan: std::env::var("RUBBLE_COUPLING_SCAN").is_ok(),
            dummy: [0, 1, 2].map(|_| storage(device, "vehicle-coupling-dummy", cw::INGEST_DUMMY_BYTES)),
            tris,
            tri_count,
            clock: cw::Clock::default(),
            status: Default::default(),
            status_buf,
            status_busy: Default::default(),
            read_pending: false,
            list_args,
            sleep_mask: vec![0; work_layout.mask_words as usize],
            any_awake: true,
        })
    }

    /// The coupling passes' status word (work[1], byte offset): the World reads it with
    /// every step ([`Self::status_bits_error`]).
    pub(crate) fn status_word(&self) -> (&wgpu::Buffer, u64) {
        (&self.work, cw::WorkLayout::STATUS_OFFSET)
    }

    /// The coupling status bits as an error (none: Ok): a room exceeded, or a loop held
    /// to its cap or past its pass budget (ST_LOOP).
    pub(crate) fn status_bits_error(&self, bits: u32) -> Result<(), String> {
        cw::status_error(bits, self.layout.cap)
    }

    /// The vehicle sleep readback (World step's extras): the asleep bits after the frame.
    pub(crate) fn sleep_extra(&self) -> Option<(&wgpu::Buffer, u64, u64)> {
        self.gpu
            .sleep_on()
            .then(|| (&self.work, self.work_layout.sleep_mask_word * 4, 4 * self.sleep_mask.len() as u64))
    }

    /// A delivered frame's asleep bits: the vehicles that fell asleep or woke since the
    /// last one, in vehicle order.
    pub(crate) fn sleep_changes(&mut self, words: &[u32]) -> Vec<crate::ExplicitSleepEvent> {
        let nv = self.sims.len();
        let mut out = Vec::new();
        for (k, (&new, old)) in words.iter().zip(self.sleep_mask.iter_mut()).enumerate() {
            let mut diff = new ^ *old;
            while diff != 0 {
                let b = diff.trailing_zeros();
                diff &= diff - 1;
                let v = 32 * k + b as usize;
                if v < nv {
                    out.push(crate::ExplicitSleepEvent {
                        vehicle: v as u32,
                        asleep: new & (1 << b) != 0,
                    });
                }
            }
            *old = new;
        }
        let full = |k: usize| if 32 * (k + 1) <= nv { u32::MAX } else { (1u32 << (nv % 32)) - 1 };
        self.any_awake = self.sleep_mask.iter().enumerate().any(|(k, &w)| w != full(k));
        out
    }

    /// Whether this World step must run for the vehicles: some vehicle was awake when
    /// last seen, or the host wrote to one since (a wake is pending).
    pub(crate) fn needs_step(&self) -> bool {
        !self.gpu.sleep_on() || self.any_awake || self.gpu.wakes_pending()
    }

    /// The buffer behind a binding of the module ([`cw::source`]).
    fn buffer<'a>(&'a self, sim: &'a engine_webgpu::WgpuSim, ingest: Ingest<'a>, source: cw::Source) -> &'a wgpu::Buffer {
        use cw::Source as S;
        match source {
            S::Params => &self.params,
            S::RigidBodies => sim.body_buffer(),
            S::RigidColliders => sim.collider_buffer(),
            S::RigidHulls => sim.hull_arena_buffer(),
            S::RigidScratch => sim.scratch_buffer(),
            S::RestIds => sim.rest_buffers().0,
            S::RestAabbs => sim.rest_buffers().1,
            S::MovedHeader => sim.moved_header_buffer(),
            S::Moved => sim.moved_buffer(),
            S::ExplicitNodes => self.gpu.nodes_buffer(),
            S::ExplicitMisc => self.gpu.misc_buffer(),
            S::Work => &self.work,
            S::IngestRecords => ingest.map_or(&self.dummy[0], |i| i.0),
            S::IngestRows => ingest.map_or(&self.dummy[1], |i| i.1),
            S::IngestInfo => ingest.map_or(&self.dummy[2], |i| i.2),
        }
    }

    /// The bind group of `set` ([`cw::MAIN_SET`], or cpl_query's [`cw::QUERY_SET`]).
    fn bind_set(&self, sim: &engine_webgpu::WgpuSim, ingest: Ingest, query: bool) -> wgpu::BindGroup {
        let (set, layout, label) = if query {
            (cw::QUERY_SET, &self.bgl_q, "vehicle-coupling-query")
        } else {
            (cw::MAIN_SET, &self.bgl, "vehicle-coupling")
        };
        let entries: Vec<wgpu::BindGroupEntry> = set
            .iter()
            .map(|&(binding, _)| wgpu::BindGroupEntry {
                binding,
                resource: self.buffer(sim, ingest, cw::source(binding)).as_entire_binding(),
            })
            .collect();
        sim.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    }

    fn bind(&self, sim: &engine_webgpu::WgpuSim, ingest: Ingest) -> wgpu::BindGroup {
        self.bind_set(sim, ingest, false)
    }

    /// cpl_query's bind group ([`Self::bgl_q`]).
    fn bind_query(&self, sim: &engine_webgpu::WgpuSim) -> wgpu::BindGroup {
        self.bind_set(sim, None, true)
    }

    /// The proxy plan's extra reach: the def's margin, and while wheels hit bodies at
    /// least the wheels' rays' ([`wheel_reach`]).
    fn plan_margin(&self) -> f32 {
        if self.def.wheels_hit_bodies {
            self.def.margin.max(wheel_reach(&self.sims))
        } else {
            self.def.margin
        }
    }

    fn write_params(&self, sim: &engine_webgpu::WgpuSim, steps: u32, ingest_rows: u32, rest: Option<engine_webgpu::RestView>) {
        let p = cw::params(&cw::ParamsInput {
            layout: &self.layout,
            work: &self.work_layout,
            colliders: sim.collider_count(),
            steps,
            margin: self.plan_margin(),
            windows: self.clock.windows,
            categories: self.def.categories,
            mask: self.def.mask,
            ingest_rows,
            serial_plan: self.serial_plan,
            sleep_base: match (self.gpu.sleep_layout(), self.gpu.sleep_on()) {
                (Some(s), true) => s.base,
                _ => 0,
            },
            hull_geom: self.sims.iter().any(|v| v.surface_contact),
            rest: rest.map(|r| cw::RestTable {
                rest_base: r.rest_base,
                hash_size: r.hash_size,
                aabb_base: r.aabb_base,
                levels: r.levels,
                cell: r.cell,
            }),
            retire_base: self.gpu.retire_base(),
            piece_records: self.gpu.piece_window_records(),
            loop_budget: engine_webgpu::loop_budget_override(),
        });
        sim.queue().write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
    }

    /// What [`Replay::run`] records this host's commands with.
    fn replay<'a>(&'a self, group: &'a wgpu::BindGroup, qgroup: &'a wgpu::BindGroup) -> Replay<'a> {
        Replay {
            pipes: &self.pipes,
            work: &self.work,
            list_args: &self.list_args,
            status: &self.status_buf,
            group,
            qgroup,
        }
    }

    /// Tests: the plan alone (bounds, plan, share) for a window of `steps` from the
    /// current state, `serial` its lane-0 walk; returns the coupling's records and the
    /// share counts as words. Blocking.
    pub fn replan_words(&mut self, sim: &mut engine_webgpu::WgpuSim, steps: u32, ingest: Ingest, serial: bool) -> Vec<u32> {
        self.serial_plan = serial;
        self.write_params(sim, steps, ingest.map_or(0, |i| i.3), None);
        self.serial_plan = false;
        let mut enc = sim.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let group = self.bind(sim, ingest);
        self.replay(&group, &group).run(sim, &mut enc, &cw::plan_alone(&self.work_layout, false, "plan-alone"), &mut |_, _| {});
        sim.queue().submit([enc.finish()]);
        let l = self.layout;
        let mut out = engine_webgpu::explicit_coupling::read_words(
            sim.device(),
            sim.queue(),
            self.gpu.misc_buffer(),
            u64::from(l.misc_base) * 16,
            u64::from(l.misc_words()) * 16,
        );
        out.extend(engine_webgpu::explicit_coupling::read_words(sim.device(), sim.queue(), &self.work, 0, self.work_layout.cleared));
        out
    }

    /// Tests: the last window's impulse records as `cpl_emit` writes them (`serial`: on one
    /// lane, the shapes in order), sorted by key; the impulse list left as it was. Blocking.
    pub fn emit_records(&mut self, sim: &mut engine_webgpu::WgpuSim, steps: u32, ingest: Ingest, serial: bool) -> Vec<[u32; 8]> {
        use engine_schema::{IMPULSE_BASE, IMPULSE_COUNT};
        let read = |sim: &engine_webgpu::WgpuSim, at: u32, words: u32| {
            engine_webgpu::explicit_coupling::read_words(sim.device(), sim.queue(), sim.scratch_buffer(), u64::from(at) * 4, u64::from(words) * 4)
        };
        let count0 = read(sim, IMPULSE_COUNT, 1)[0];
        self.serial_plan = serial;
        self.write_params(sim, steps, ingest.map_or(0, |i| i.3), None);
        self.serial_plan = false;
        let group = self.bind(sim, ingest);
        let mut enc = sim.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.replay(&group, &group).run(sim, &mut enc, &cw::emit_alone(), &mut |_, _| {});
        sim.queue().submit([enc.finish()]);
        let count1 = read(sim, IMPULSE_COUNT, 1)[0].min(engine_schema::IMPULSE_CAP);
        let words = read(sim, IMPULSE_BASE + 8 * count0, 8 * count1.saturating_sub(count0).max(1));
        sim.queue().write_buffer(sim.scratch_buffer(), u64::from(IMPULSE_COUNT) * 4, bytemuck::bytes_of(&count0));
        let mut recs: Vec<[u32; 8]> = words.chunks_exact(8).take(count1.saturating_sub(count0) as usize).map(|c| c.try_into().unwrap()).collect();
        recs.sort_by_key(|r| (r[1], r[0]));
        recs
    }

    fn set_flag(&mut self, queue: &wgpu::Queue, v: u32, flag: u32, on: bool) {
        let f = self.flags[v as usize];
        let f = if on { f | flag } else { f & !flag };
        self.flags[v as usize] = f;
        queue.write_buffer(&self.work, self.work_layout.vehicle(v) + 12, bytemuck::bytes_of(&f));
    }

    /// Chunk records a window can make at most: a record per proxy shape.
    pub fn ingest_rows(&self) -> u32 {
        self.work_layout.ingest_rows(self.layout.cap.shapes)
    }

    /// Step 1, before the rigid step: the last window's impulses as impulse records
    /// (`cw::emit`).
    pub fn encode_emit(&mut self, sim: &mut engine_webgpu::WgpuSim, ingest: Ingest, enc: &mut wgpu::CommandEncoder) {
        let Some(wakes) = self.clock.take_emit() else {
            return;
        };
        let group = self.bind(sim, ingest);
        self.replay(&group, &group).run(sim, enc, &cw::emit(), &mut |_, _| {});
        if wakes {
            sim.mark_gpu_impulses();
        } else {
            sim.schedule_gpu_impulses();
        }
    }

    /// Step 3, after the rigid step: the next window's plan, frame and chunk loads
    /// (`cw::frame`). Returns the explicit frame's status-read flag (for
    /// [`Self::after_submit`]).
    pub fn encode_frame(&mut self, sim: &mut engine_webgpu::WgpuSim, dt: f32, ingest: Ingest, enc: &mut wgpu::CommandEncoder) -> anyhow::Result<bool> {
        if let Some(bits) = self.status.lock().unwrap().take() {
            self.status_bits_error(bits).map_err(anyhow::Error::msg)?;
        }
        // The explicit solver's own status (loop cap, vehicle-contact limits) is read
        // without waiting inside its frame (it fails the frame that finds an error).
        let steps = self.clock.window_steps(dt);
        let rows = ingest.map_or(0, |i| i.3);
        // The World broadphase's resting table when the rigid step just kept it (O(nearby)
        // per vehicle); else every collider is scanned.
        let rest = if self.force_scan { None } else { sim.rest_view() };
        self.last_query = rest.is_some();
        self.write_params(sim, steps, rows, rest);
        let group = self.bind(sim, ingest);
        let qgroup = self.bind_query(sim);
        let cmds = cw::frame(&self.work_layout, cw::FrameOpts { query: self.last_query, sleep: self.gpu.sleep_on() });
        let mut read = false;
        let replay = Replay {
            pipes: &self.pipes,
            work: &self.work,
            list_args: &self.list_args,
            status: &self.status_buf,
            group: &group,
            qgroup: &qgroup,
        };
        let gpu = &mut self.gpu;
        replay.run(sim, enc, &cmds, &mut |sim, enc| {
            // The rigid bodies push the vehicles: vehicle contact's frame bounds are
            // planned from this window's speeds.
            gpu.request_contact_replan();
            if std::env::var("RUBBLE_GPU_TIMING_EXPLICIT").is_ok() {
                gpu.set_marks(sim.marks().cloned());
            }
            // `GpuTiming::Phases`: each of the frame's dispatches timed on its own (named
            // by its kernel); `Step`: the frame as one segment.
            read = if sim.gpu_timing() == engine_webgpu::GpuTiming::Phases && steps > 0 && sim.host_segments_left() > 1 {
                let (first, cap) = (sim.next_host_segment(), sim.host_segments_left() - 1);
                let set = sim.step_query_set().expect("timing on").clone();
                let (read, labels) = gpu.frame_encode_per_dispatch(enc, steps, &set, first, cap);
                for l in labels {
                    let _ = sim.host_segment(l);
                }
                read
            } else {
                let seg = if steps > 0 { sim.host_segment("explicit_frame") } else { None };
                gpu.frame_encode_timed(enc, steps, sim.step_query_set().zip(seg.map(|k| 2 * k)))
            };
        });
        let due = self.clock.window_done(steps, self.needs_step());
        self.read_pending = false;
        if due && !self.status_busy.load(std::sync::atomic::Ordering::Acquire) {
            self.replay(&group, &qgroup).run(sim, enc, &[cw::status_read()], &mut |_, _| {});
            self.read_pending = true;
        }
        Ok(read)
    }

    /// After the submission holding [`Self::encode_frame`].
    pub fn after_submit(&mut self, read: bool) {
        self.gpu.frame_submitted(read);
        if std::mem::take(&mut self.read_pending) {
            self.status_busy.store(true, std::sync::atomic::Ordering::Release);
            let busy = self.status_busy.clone();
            let status = self.status.clone();
            let buf = self.status_buf.clone();
            self.status_buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                if r.is_ok() {
                    let bits = {
                        let data = buf.slice(..).get_mapped_range().expect("mapped");
                        let w: &[u32] = bytemuck::cast_slice(&data);
                        w[1]
                    };
                    buf.unmap();
                    if bits != 0 {
                        *status.lock().unwrap() = Some(bits);
                    }
                }
                busy.store(false, std::sync::atomic::Ordering::Release);
            });
        }
    }
}

/// A vehicle's proxy in the World broadphase ([`World::explicit_vehicle_proxy`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleProxy {
    /// Its fat box: the swept box (nodes, the band, the fastest node's travel over a
    /// World step) grown by [`PROXY_SLACK`] when the swept box last left it. Frozen while
    /// the vehicle sleeps (vehicle sleep) or is out of the broadphase.
    pub lo: glam::Vec3,
    pub hi: glam::Vec3,
    pub in_broadphase: bool,
    /// Times the fat box was rebuilt.
    pub refattens: u32,
}

impl World {
    /// Put explicit soft-body vehicles (2 kHz node–beam, `engine_softbody::explicit_beams`)
    /// in this world, on its device: they collide two way with its bodies and load its
    /// threshold destructibles (`engine/docs/explicit-vehicles-in-world.md`). Once per world;
    /// the vehicles start stepping with the next step.
    pub fn add_explicit_vehicles(&mut self, vehicles: &[ExplicitBeams], def: &ExplicitVehiclesDef) -> anyhow::Result<()> {
        anyhow::ensure!(self.explicit.is_none(), "explicit vehicles are already in this world");
        let x = ExplicitVehicles::new(self, vehicles, def)?;
        self.explicit = Some(Box::new(x));
        Ok(())
    }

    /// Put the explicit vehicles back to `vehicles`' state in place (a restart: the same
    /// fleet placed anew, at new speeds, undamaged), without making a device, buffer or
    /// pipeline: `ExplicitBeamsGpu::reset_state`, and the coupling's clock, pending
    /// impulses and proxies as after [`World::add_explicit_vehicles`]. The vehicles
    /// step again with the next step. `vehicles` must be the fleet that was added (only
    /// its state may differ); another is an error and changes nothing.
    pub fn reset_explicit_vehicles(&mut self, vehicles: &[ExplicitBeams]) -> anyhow::Result<()> {
        let queue = self.sim.queue().clone();
        let x = self.explicit.as_mut().ok_or_else(|| anyhow::anyhow!("no explicit vehicles"))?;
        anyhow::ensure!(vehicles.len() == x.sims.len(), "reset_explicit_vehicles: {} vehicles, {} were added", vehicles.len(), x.sims.len());
        x.gpu.reset_state(vehicles, Some(x.def.capacity), x.def.sleep).map_err(anyhow::Error::msg)?;
        x.gpu.set_wheels_see_bodies(x.def.wheels_hit_bodies);
        x.sims = vehicles.to_vec();
        x.clock.reset();
        x.read_pending = false;
        *x.status.lock().unwrap() = None;
        x.any_awake = true;
        for m in &mut x.sleep_mask {
            *m = 0;
        }
        queue.write_buffer(&x.work, 4 * x.work_layout.sleep_mask_word, bytemuck::cast_slice(&x.sleep_mask));
        for v in 0..vehicles.len() as u32 {
            // Its proxy in the broadphase is rebuilt at the next window.
            queue.write_buffer(&x.work, x.work_layout.proxy(v) + 28, bytemuck::bytes_of(&0u32));
        }
        Ok(())
    }

    /// The explicit vehicles' GPU state (nodes, beams, the coupling's records), once added.
    pub fn explicit_vehicles(&self) -> Option<&ExplicitBeamsGpu> {
        self.explicit.as_ref().map(|x| &x.gpu)
    }

    /// For renderers: the explicit vehicles' collision triangles, (global nodes a, b, c,
    /// vehicle) as `u32` per triangle (node positions: [`ExplicitBeamsGpu::nodes_buffer`]
    /// relative to each vehicle's origin in [`ExplicitBeamsGpu::misc_buffer`]), and their
    /// count.
    pub fn explicit_triangles(&self) -> Option<(&wgpu::Buffer, u32)> {
        self.explicit.as_ref().map(|x| (&x.tris, x.tri_count))
    }

    /// The explicit vehicles as they were added (spawn state: node counts, raycast wheels).
    pub fn explicit_vehicles_as_added(&self) -> &[ExplicitBeams] {
        self.explicit.as_ref().map_or(&[], |x| x.sims.as_slice())
    }

    /// Mutable access (velocity writes, the grab, vehicle contact settings).
    pub fn explicit_vehicles_mut(&mut self) -> Option<&mut ExplicitBeamsGpu> {
        self.explicit.as_mut().map(|x| &mut x.gpu)
    }

    /// Vehicle sleep at runtime, from the next step on, with no rebuild: `Some(thresholds)`
    /// turns it on (or changes the thresholds of a sleep already on: still counts and
    /// sleeping vehicles stand), `None` turns it off (every vehicle steps every step; a
    /// sleeping one resumes from its state at sleep). Turning it on from off starts every
    /// vehicle awake with a still count of 0. Only for vehicles added with room for sleep
    /// ([`ExplicitVehiclesDef::sleep`] set, the default); thresholds that are not finite
    /// and non-negative are an error and change nothing.
    pub fn set_explicit_vehicle_sleep(&mut self, sleep: Option<engine_softbody::explicit_sleep::VehicleSleep>) -> anyhow::Result<()> {
        let queue = self.sim.queue().clone();
        let x = self.explicit.as_mut().ok_or_else(|| anyhow::anyhow!("no explicit vehicles"))?;
        anyhow::ensure!(
            x.gpu.sleep_layout().is_some(),
            "vehicle sleep: these vehicles were added without room for it (ExplicitVehiclesDef::sleep was None)"
        );
        let Some(cfg) = sleep else {
            x.gpu.set_sleep(false);
            return Ok(());
        };
        anyhow::ensure!(
            cfg.speed.is_finite() && cfg.speed >= 0.0 && cfg.reach.is_finite() && cfg.reach >= 0.0,
            "vehicle sleep: speed {} m/s and reach {} m must be finite and not negative",
            cfg.speed,
            cfg.reach
        );
        x.gpu.set_sleep_config(cfg);
        x.def.sleep = Some(cfg);
        if !x.gpu.sleep_on() {
            x.gpu.set_sleep(true);
            x.any_awake = true;
            for m in &mut x.sleep_mask {
                *m = 0;
            }
            queue.write_buffer(&x.work, 4 * x.work_layout.sleep_mask_word, bytemuck::cast_slice(&x.sleep_mask));
        }
        Ok(())
    }

    /// Bit-exact results at runtime, from the next step on ([`ExplicitVehiclesDef::bit_exact`];
    /// `ExplicitBeamsGpu::set_bit_exact`): the solver's pipelines are made again for the
    /// setting (compiled at their first use); no state changes. No explicit vehicles: an
    /// error.
    pub fn set_explicit_bit_exact(&mut self, on: bool) -> anyhow::Result<()> {
        let x = self.explicit.as_mut().ok_or_else(|| anyhow::anyhow!("no explicit vehicles"))?;
        x.gpu.set_bit_exact(on);
        x.def.bit_exact = on;
        Ok(())
    }

    /// Whether the explicit vehicles' results are bit-exact (`None`: no explicit vehicles).
    pub fn explicit_bit_exact(&self) -> Option<bool> {
        self.explicit.as_ref().map(|x| x.gpu.bit_exact())
    }

    /// Raycast wheels hit bodies ([`ExplicitVehiclesDef::wheels_hit_bodies`]) on or off at
    /// runtime, from the next step, with no rebuild.
    pub fn set_explicit_wheels_hit_bodies(&mut self, on: bool) -> anyhow::Result<()> {
        let x = self.explicit.as_mut().ok_or_else(|| anyhow::anyhow!("no explicit vehicles"))?;
        x.def.wheels_hit_bodies = on;
        x.gpu.set_wheels_see_bodies(on);
        Ok(())
    }

    /// Whether raycast wheels hit bodies (false without explicit vehicles).
    pub fn explicit_wheels_hit_bodies(&self) -> bool {
        self.explicit.as_ref().is_some_and(|x| x.def.wheels_hit_bodies)
    }

    /// Vehicle sleep's thresholds when it is on (`None`: off, or no explicit vehicles).
    pub fn explicit_vehicle_sleep(&self) -> Option<engine_softbody::explicit_sleep::VehicleSleep> {
        let x = self.explicit.as_ref()?;
        x.gpu.sleep_on().then(|| x.gpu.sleep_layout().map(|s| s.cfg)).flatten()
    }

    /// Blocking read (tests, tools): per explicit vehicle its sleep state (still steps,
    /// asleep); empty without vehicle sleep.
    pub fn explicit_sleep_states(&self) -> Vec<(u32, bool)> {
        self.explicit.as_ref().map_or_else(Vec::new, |x| x.gpu.sleep_states())
    }

    /// Explicit steps the vehicles have run, and windows (World steps) they took.
    pub fn explicit_vehicle_steps(&self) -> Option<(u64, u64)> {
        self.explicit.as_ref().map(|x| (x.clock.steps, x.clock.windows))
    }

    /// Blocking read (tests, tools): per vehicle, its last window's proxy shapes as
    /// (collider, impulse N s, angular impulse about the shape's reference point and that
    /// point): the impulses the next step's rigid step applies.
    /// Measurement (blocking): per vehicle the coupling's candidates of the last window
    /// (cpl_query's list, duplicates included; its room is three times the shapes room).
    pub fn explicit_candidate_counts(&self) -> Vec<u32> {
        let Some(x) = self.explicit.as_ref() else {
            return Vec::new();
        };
        let nv = x.sims.len() as u64;
        engine_webgpu::explicit_coupling::read_words(self.sim.device(), self.sim.queue(), &x.work, x.work_layout.cand_counts(), 4 * nv)
    }

    pub fn explicit_window_impulses(&self) -> Vec<Vec<(u32, [f32; 3], [f32; 3], [f32; 3])>> {
        let Some(x) = self.explicit.as_ref() else {
            return Vec::new();
        };
        let l = x.layout;
        let words = engine_webgpu::explicit_coupling::read_words(
            self.sim.device(),
            self.sim.queue(),
            x.gpu.misc_buffer(),
            u64::from(l.misc_base) * 16,
            u64::from(l.misc_words()) * 16,
        );
        let rec = |r: u32| -> [u32; 4] {
            let i = ((r - l.misc_base) * 4) as usize;
            [words[i], words[i + 1], words[i + 2], words[i + 3]]
        };
        let f3 = |r: [u32; 4]| [f32::from_bits(r[0]), f32::from_bits(r[1]), f32::from_bits(r[2])];
        (0..x.sims.len() as u32)
            .map(|v| {
                let n = rec(l.vehicle(v))[1];
                (0..n)
                    .map(|k| {
                        let at = l.shape(v, k);
                        (rec(at)[2], f3(rec(at + 4)), f3(rec(at + 5)), f3(rec(at + 6)))
                    })
                    .collect()
            })
            .collect()
    }

    /// Tests only (blocking): the coupling plan alone from the current state, its
    /// lane-parallel records or (`serial`) its lane-0 walk, as words: the records and
    /// the share counts. Two calls from one state compare the two.
    #[doc(hidden)]
    pub fn explicit_replan_words(&mut self, serial: bool) -> Vec<u32> {
        let Some(mut x) = self.explicit.take() else {
            return Vec::new();
        };
        let steps = (f64::from(self.sim.params().dt) / f64::from(engine_softbody::explicit_beams::STEP)).round() as u32;
        let out = x.replan_words(&mut self.sim, steps, self.fracture.vehicle_ingest().ok().flatten(), serial);
        self.explicit = Some(x);
        out
    }

    /// Tests: the last window's impulse records from `cpl_emit` (lane-parallel, or
    /// `serial` on one lane), sorted by key, the World's impulse list left as it was.
    #[doc(hidden)]
    pub fn explicit_emit_records(&mut self, serial: bool) -> Vec<[u32; 8]> {
        let Some(mut x) = self.explicit.take() else {
            return Vec::new();
        };
        let steps = (f64::from(self.sim.params().dt) / f64::from(engine_softbody::explicit_beams::STEP)).round() as u32;
        let out = x.emit_records(&mut self.sim, steps, self.fracture.vehicle_ingest().ok().flatten(), serial);
        self.explicit = Some(x);
        out
    }

    /// Vehicle `v` out of the World broadphase (`false`: removed: no proxy, no query, no
    /// coupling) or back in (`true`, the default). Its fat box is rebuilt on return.
    pub fn set_explicit_vehicle_in_broadphase(&mut self, v: u32, inside: bool) -> anyhow::Result<()> {
        let queue = self.sim.queue().clone();
        let x = self.explicit.as_mut().ok_or_else(|| anyhow::anyhow!("no explicit vehicles"))?;
        anyhow::ensure!((v as usize) < x.sims.len(), "explicit vehicle {v} out of range");
        x.set_flag(&queue, v, cw::PROXY_OUT, !inside);
        if inside {
            // Refattened at the next window (moves 0).
            let at = x.work_layout.proxy(v) + 28;
            queue.write_buffer(&x.work, at, bytemuck::bytes_of(&0u32));
        }
        Ok(())
    }

    /// Blocking read (tests, tools): vehicle `v`'s broadphase proxy.
    pub fn explicit_vehicle_proxy(&self, v: u32) -> Option<VehicleProxy> {
        let x = self.explicit.as_ref()?;
        if v as usize >= x.sims.len() {
            return None;
        }
        let w = engine_webgpu::explicit_coupling::read_words(self.sim.device(), self.sim.queue(), &x.work, x.work_layout.proxy(v), 32);
        let f = |k: usize| f32::from_bits(w[k]);
        let flags = x.flags[v as usize];
        Some(VehicleProxy {
            lo: glam::Vec3::new(f(0), f(1), f(2)),
            hi: glam::Vec3::new(f(3), f(4), f(5)),
            in_broadphase: flags & cw::PROXY_OUT == 0,
            refattens: w[7],
        })
    }

    /// The proxies' GPU table (for a GPU consumer): the buffer, the byte offset of vehicle
    /// 0's record and the stride. Each record: fat box lo xyz, hi xyz (f32 bits), 0,
    /// refattens (u32).
    pub fn explicit_proxy_table(&self) -> Option<(&wgpu::Buffer, u64, u64)> {
        let x = self.explicit.as_ref()?;
        Some((&x.work, x.work_layout.proxy(0), 32))
    }

    /// Test hook: the coupling plan run now twice from the World's current state, from the
    /// World broadphase's resting table and by the scan of every collider; each run's
    /// proxy records (every vehicle's region, every word) and status. Errors when the last
    /// step did not keep the resting table. Overwrites the last window's records (the
    /// next step's impulses): for tests that end or compare here.
    #[doc(hidden)]
    pub fn debug_explicit_plans(&mut self) -> anyhow::Result<[(Vec<u32>, u32); 2]> {
        let rest = self.sim.rest_view().ok_or_else(|| anyhow::anyhow!("the last step kept no resting table"))?;
        let x = self.explicit.as_ref().ok_or_else(|| anyhow::anyhow!("no explicit vehicles"))?;
        let l = x.layout;
        let mut out: Vec<(Vec<u32>, u32)> = Vec::new();
        for query in [true, false] {
            x.write_params(&self.sim, 34, 0, query.then_some(rest));
            let mut enc = self.sim.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("plan-check") });
            let group = x.bind(&self.sim, None);
            let qgroup = x.bind_query(&self.sim);
            x.replay(&group, &qgroup).run(&mut self.sim, &mut enc, &cw::plan_alone(&x.work_layout, query, "plan-check"), &mut |_, _| {});
            self.sim.queue().submit([enc.finish()]);
            let words = engine_webgpu::explicit_coupling::read_words(
                self.sim.device(),
                self.sim.queue(),
                x.gpu.misc_buffer(),
                u64::from(l.misc_base) * 16,
                u64::from(l.misc_words()) * 16,
            );
            let status = engine_webgpu::explicit_coupling::read_words(self.sim.device(), self.sim.queue(), &x.work, 0, 16)[1];
            out.push((words, status));
        }
        let b = out.pop().unwrap();
        let a = out.pop().unwrap();
        Ok([a, b])
    }

    /// Measurement hook: the coupling plan (bounds, the broadphase query or the scan, the
    /// plan, the share counts) from the World's current state, `reps` times in one
    /// submission, waited for; the mean wall time per plan (ms). `scan`: every collider
    /// scanned; else the World broadphase's resting table (an error when the last step
    /// kept none). Overwrites the last window's records, like
    /// [`World::debug_explicit_plans`].
    #[doc(hidden)]
    pub fn debug_time_explicit_plan(&mut self, scan: bool, reps: u32) -> anyhow::Result<f64> {
        let rest = if scan {
            None
        } else {
            Some(self.sim.rest_view().ok_or_else(|| anyhow::anyhow!("the last step kept no resting table"))?)
        };
        let x = self.explicit.as_ref().ok_or_else(|| anyhow::anyhow!("no explicit vehicles"))?;
        x.write_params(&self.sim, 34, 0, rest);
        let group = x.bind(&self.sim, None);
        let qgroup = x.bind_query(&self.sim);
        let mut enc = self.sim.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("plan-time") });
        let cmds = cw::plan_alone(&x.work_layout, rest.is_some(), "plan-time");
        for _ in 0..reps {
            x.replay(&group, &qgroup).run(&mut self.sim, &mut enc, &cmds, &mut |_, _| {});
        }
        let t = std::time::Instant::now();
        self.sim.queue().submit([enc.finish()]);
        let _ = self.sim.device().poll(wgpu::PollType::wait_indefinitely());
        Ok(t.elapsed().as_secs_f64() * 1e3 / f64::from(reps.max(1)))
    }

    /// Measurement switch (interleaved A/B): plan every window by the scan of every
    /// collider (`true`) even when the World broadphase's resting table is kept (also
    /// `RUBBLE_COUPLING_SCAN=1`). The proxies are the same either way.
    #[doc(hidden)]
    pub fn set_explicit_plan_scan(&mut self, scan: bool) {
        if let Some(x) = self.explicit.as_mut() {
            x.force_scan = scan;
        }
    }

    /// Whether the last window's proxies came from the World broadphase's persistent
    /// resting table (true) or a scan of every collider (false: the step did not keep
    /// that table, as in a world with a GPU fracture).
    pub fn explicit_plan_used_broadphase(&self) -> Option<bool> {
        self.explicit.as_ref().map(|x| x.last_query)
    }

    /// The coupling's record layout in the explicit buffers (for reads of the proxies).
    pub fn explicit_coupling_layout(&self) -> Option<CouplingLayout> {
        self.explicit.as_ref().map(|x| x.layout)
    }

    /// `World::step`'s part before the rigid step (emit).
    pub(crate) fn explicit_before_step(&mut self, enc: &mut wgpu::CommandEncoder) -> anyhow::Result<()> {
        let Some(mut x) = self.explicit.take() else {
            return Ok(());
        };
        let r = self.fracture.vehicle_ingest().map(|ingest| x.encode_emit(&mut self.sim, ingest, enc));
        // Vehicle sleep: every vehicle asleep and nothing written to one: the vehicles
        // need no step (the rigid bodies may still run it).
        if x.needs_step() {
            self.sim.request_step();
        }
        self.explicit = Some(x);
        r
    }

    /// `World::step`'s part after the rigid step (the next window's frame).
    pub(crate) fn explicit_after_step(&mut self, enc: &mut wgpu::CommandEncoder) -> anyhow::Result<Option<bool>> {
        let Some(mut x) = self.explicit.take() else {
            return Ok(None);
        };
        let dt = self.sim.params().dt;
        let r = self
            .fracture
            .vehicle_ingest()
            .and_then(|ingest| x.encode_frame(&mut self.sim, dt, ingest, enc));
        self.explicit = Some(x);
        r.map(Some)
    }

    pub(crate) fn explicit_after_submit(&mut self, read: Option<bool>) {
        if let (Some(x), Some(read)) = (self.explicit.as_mut(), read) {
            x.after_submit(read);
        }
    }
}
