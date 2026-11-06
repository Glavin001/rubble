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
use engine_softbody::explicit_beams::ExplicitBeams;
use engine_webgpu::{CouplingCapacity, CouplingLayout, ExplicitBeamsGpu, VEHICLE_COUPLING_WGSL};

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
        }
    }
}

const STEP: f64 = engine_softbody::explicit_beams::STEP as f64;
/// Impulse record keys (after the host's, the terrain's 0x300000 + collider).
const KEY_BASE: u32 = 0x50_0000;
const ST_NAMES: [&str; 5] = [
    "more colliders than the shapes room",
    "more bodies than the bodies room",
    "more hull planes than the planes room",
    "more impulse records than IMPULSE_CAP",
    "more chunk records than the threshold ingest's rows",
];
/// Windows between status reads.
const STATUS_EVERY: u64 = 8;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    vehicles: u32,
    colliders: u32,
    max_bodies: u32,
    couple_rec: u32,
    b: u32,
    s: u32,
    planes: u32,
    recs: u32,
    misc_base: u32,
    window: f32,
    margin: f32,
    key_base: u32,
    impulse_base: u32,
    impulse_count: u32,
    impulse_first: u32,
    impulse_cap: u32,
    ingest_rows: u32,
    ingest_slot: u32,
    step: u32,
    filter: u32,
    row_cap: u32,
    /// Tests: cpl_plan's serial walk (1) instead of its lane-parallel records.
    serial_plan: u32,
    /// Vehicle sleep: gMisc record of vehicle 0's sleep record (0: off).
    sleep_base: u32,
    /// work's word of the proxied list (0: every vehicle).
    list_base: u32,
}

pub(crate) struct ExplicitVehicles {
    pub gpu: ExplicitBeamsGpu,
    layout: CouplingLayout,
    def: ExplicitVehiclesDef,
    /// The vehicles as added (spawn state: node counts, masses).
    pub sims: Vec<ExplicitBeams>,
    bgl: wgpu::BindGroupLayout,
    /// cpl_bounds, cpl_plan, cpl_share, cpl_loads, cpl_loads_end, cpl_emit, cpl_emit_end,
    /// cpl_sleep_mask.
    pipes: Vec<wgpu::ComputePipeline>,
    params: wgpu::Buffer,
    work: wgpu::Buffer,
    /// Bytes of `work` cleared each window (the vehicle table follows).
    cleared: u64,
    /// Tests ([`Self::replan_words`]): cpl_plan's serial walk.
    serial_plan: bool,
    /// Stand-ins for the threshold ingest's records, row counts and collider table.
    dummy: [wgpu::Buffer; 3],
    /// For renderers: per collision triangle (global nodes a, b, c, vehicle).
    pub tris: wgpu::Buffer,
    pub tri_count: u32,
    /// Explicit steps owed to World time (the remainder carried).
    carry: f64,
    /// A frame ran: its impulses are emitted at the next step's start.
    emit_pending: bool,
    /// That frame was encoded while some vehicle could be awake (vehicle sleep): its
    /// emit wakes the rigid step. Otherwise every vehicle was asleep with nothing
    /// written to one, so it stepped only if the plan woke it beside a moving body
    /// (then the rigid step runs anyway): the emit schedules its records without a wake.
    emit_wakes: bool,
    pub windows: u64,
    pub steps: u64,
    status: std::sync::Arc<std::sync::Mutex<Option<u32>>>,
    status_buf: wgpu::Buffer,
    status_busy: std::sync::Arc<std::sync::atomic::AtomicBool>,
    read_pending: bool,
    /// work's word of the proxied list: its count (the x of cpl_share's, cpl_loads' and
    /// cpl_emit's indirect arguments), 1, 1, 0, then the vehicles.
    list_word: u64,
    list_args: wgpu::Buffer,
    /// Vehicle sleep: work's word of the asleep bits (one word per 32 vehicles), and the
    /// bits the last delivered frame reported (all awake at the start).
    pub(crate) sleep_mask_word: u64,
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
        let mut gpu = ExplicitBeamsGpu::with_sleep(device, queue, vehicles, Some(def.capacity), def.sleep);
        gpu.set_vehicle_contact(def.vehicle_contact);
        gpu.set_coupling_on(true);
        let layout = gpu.coupling().expect("coupling room");
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let entries = [
            entry(0, wgpu::BufferBindingType::Uniform),
            entry(1, ro),
            entry(2, ro),
            entry(3, ro),
            entry(4, ro),
            entry(5, rw),
            entry(8, rw),
            entry(9, rw),
            entry(10, ro),
            entry(11, rw),
            entry(12, rw),
        ];
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vehicle-coupling"),
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vehicle-coupling"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vehicle_coupling"),
            source: wgpu::ShaderSource::Wgsl(VEHICLE_COUPLING_WGSL.into()),
        });
        let pipes = [
            "cpl_bounds",
            "cpl_plan",
            "cpl_share",
            "cpl_loads",
            "cpl_loads_end",
            "cpl_emit",
            "cpl_emit_end",
            "cpl_sleep_mask",
        ]
            .iter()
            .map(|e| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(e),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(e),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        use wgpu::util::DeviceExt;
        let veh: Vec<[u32; 4]> = vehicles
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mass: f32 = v.nodes.iter().map(|n| n.mass).sum();
                [gpu.node_offset(i), v.nodes.len() as u32, mass.to_bits(), 0]
            })
            .collect();

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
        let max_bodies = world.sim.capacity().max_bodies;
        // Cleared each window: rows, status, share counts; then the bounds (cpl_bounds
        // rewrites an awake vehicle's, a sleeping one's stand), the vehicle table, the
        // asleep bits, the proxied list (its count cleared each window).
        let nv = vehicles.len() as u64;
        let cleared = 4 * (2 + u64::from(max_bodies));
        let table = cleared + 32 * nv;
        let mask_words = nv.div_ceil(32);
        let sleep_mask_word = table / 4 + 4 * nv;
        let list_word = (sleep_mask_word + mask_words).next_multiple_of(4);
        let work = storage(device, "vehicle-coupling-work", 4 * (list_word + 4 + nv));
        // The proxied list's indirect arguments: its count copied after the plan (a buffer
        // bound for storage cannot also be a dispatch's indirect arguments).
        let list_args = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vehicle-coupling-list-args"),
            contents: bytemuck::cast_slice(&[0u32, 1, 1, 0]),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        });
        queue.write_buffer(&work, table, bytemuck::cast_slice(&veh));
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vehicle-coupling-params"),
            size: std::mem::size_of::<Params>() as u64,
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
            pipes,
            params,
            work,
            cleared,
            serial_plan: false,
            dummy: [0, 1, 2].map(|_| storage(device, "vehicle-coupling-dummy", 160)),
            tris,
            tri_count,
            carry: 0.0,
            emit_pending: false,
            emit_wakes: true,
            windows: 0,
            steps: 0,
            status: Default::default(),
            status_buf,
            status_busy: Default::default(),
            read_pending: false,
            list_word,
            list_args,
            sleep_mask_word,
            sleep_mask: vec![0; mask_words as usize],
            any_awake: true,
        })
    }

    /// The vehicle sleep readback (World step's extras): the asleep bits after the frame.
    pub(crate) fn sleep_extra(&self) -> Option<(&wgpu::Buffer, u64, u64)> {
        self.gpu
            .sleep_on()
            .then(|| (&self.work, self.sleep_mask_word * 4, 4 * self.sleep_mask.len() as u64))
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

    fn bind(
        &self,
        sim: &engine_webgpu::WgpuSim,
        ingest: Option<(&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, u32)>,
    ) -> wgpu::BindGroup {
        let (records, rows, info) = match ingest {
            Some((r, c, i, _)) => (r, c, i),
            None => (&self.dummy[0], &self.dummy[1], &self.dummy[2]),
        };
        let bufs: [(u32, &wgpu::Buffer); 11] = [
            (0, &self.params),
            (1, sim.body_buffer()),
            (2, sim.collider_buffer()),
            (3, sim.hull_arena_buffer()),
            (4, self.gpu.nodes_buffer()),
            (5, self.gpu.misc_buffer()),
            (8, sim.scratch_buffer()),
            (9, &self.work),
            (10, info),
            (11, records),
            (12, rows),
        ];
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .map(|(b, buf)| wgpu::BindGroupEntry {
                binding: *b,
                resource: buf.as_entire_binding(),
            })
            .collect();
        sim.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vehicle-coupling"),
            layout: &self.bgl,
            entries: &entries,
        })
    }

    fn write_params(&self, sim: &engine_webgpu::WgpuSim, steps: u32, ingest_rows: u32) {
        let cap = self.layout.cap;
        let p = Params {
            vehicles: self.sims.len() as u32,
            colliders: sim.collider_count(),
            max_bodies: sim.capacity().max_bodies,
            couple_rec: 0,
            b: cap.bodies,
            s: cap.shapes,
            planes: cap.planes,
            recs: self.layout.recs(),
            misc_base: self.layout.misc_base,
            window: (f64::from(steps) * STEP) as f32,
            margin: self.def.margin,
            key_base: KEY_BASE,
            impulse_base: engine_schema::IMPULSE_BASE,
            impulse_count: engine_schema::IMPULSE_COUNT,
            impulse_first: engine_schema::IMPULSE_FIRST,
            impulse_cap: engine_schema::IMPULSE_CAP,
            ingest_rows,
            ingest_slot: 1,
            step: self.windows as u32,
            filter: u32::from(self.def.categories) | u32::from(self.def.mask) << 16,
            row_cap: self.ingest_rows().min(ingest_rows),
            serial_plan: u32::from(self.serial_plan),
            sleep_base: match (self.gpu.sleep_layout(), self.gpu.sleep_on()) {
                (Some(s), true) => s.base,
                _ => 0,
            },
            list_base: self.list_word as u32,
        };
        sim.queue().write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
    }

    /// Tests: the plan alone (bounds, plan, share) for a window of `steps` from the
    /// current state, `serial` its lane-0 walk; returns the coupling's records and the
    /// share counts as words. Blocking.
    pub fn replan_words(
        &mut self,
        sim: &engine_webgpu::WgpuSim,
        steps: u32,
        ingest: Option<(&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, u32)>,
        serial: bool,
    ) -> Vec<u32> {
        self.serial_plan = serial;
        self.write_params(sim, steps, ingest.map_or(0, |i| i.3));
        self.serial_plan = false;
        let mut enc = sim.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.clear_buffer(&self.work, 0, Some(self.cleared));
        enc.clear_buffer(&self.work, 4 * self.list_word, Some(4));
        let group = self.bind(sim, ingest);
        let nv = self.sims.len() as u32;
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_bind_group(0, &group, &[]);
            for p in 0..3 {
                pass.set_pipeline(&self.pipes[p]);
                pass.dispatch_workgroups(nv, 1, 1);
            }
        }
        sim.queue().submit([enc.finish()]);
        let l = self.layout;
        let mut out = engine_webgpu::explicit_coupling::read_words(
            sim.device(),
            sim.queue(),
            self.gpu.misc_buffer(),
            u64::from(l.misc_base) * 16,
            u64::from(l.misc_words()) * 16,
        );
        out.extend(engine_webgpu::explicit_coupling::read_words(sim.device(), sim.queue(), &self.work, 0, self.cleared));
        out
    }

    /// Chunk records a window can make at most: a record per proxy shape.
    pub fn ingest_rows(&self) -> u32 {
        self.sims.len() as u32 * self.layout.cap.shapes
    }

    /// Step 1, before the rigid step: the last window's impulses as impulse records.
    pub fn encode_emit(
        &mut self,
        sim: &mut engine_webgpu::WgpuSim,
        ingest: Option<(&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, u32)>,
        enc: &mut wgpu::CommandEncoder,
    ) {
        if !std::mem::take(&mut self.emit_pending) {
            return;
        }
        let group = self.bind(sim, ingest);
        let nv = self.sims.len() as u32;
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vehicle coupling emit"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &group, &[]);
        pass.set_pipeline(&self.pipes[5]);
        let _ = nv;
        pass.dispatch_workgroups_indirect(&self.list_args, 0);
        pass.set_pipeline(&self.pipes[6]);
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);
        if self.emit_wakes {
            sim.mark_gpu_impulses();
        } else {
            sim.schedule_gpu_impulses();
        }
    }

    /// Step 3, after the rigid step: the next window's plan, frame and chunk loads.
    /// Returns the explicit frame's status-read flag (for [`Self::after_submit`]).
    pub fn encode_frame(
        &mut self,
        sim: &engine_webgpu::WgpuSim,
        dt: f32,
        ingest: Option<(&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, u32)>,
        enc: &mut wgpu::CommandEncoder,
    ) -> anyhow::Result<bool> {
        if let Some(bits) = self.status.lock().unwrap().take() {
            let names: Vec<&str> = (0..5).filter(|k| bits & (1 << k) != 0).map(|k| ST_NAMES[k]).collect();
            anyhow::bail!(
                "explicit vehicles: {} (capacity {:?}; raise ExplicitVehiclesDef::capacity)",
                names.join(", "),
                self.layout.cap
            );
        }
        // The explicit solver's own status (loop cap, vehicle-contact limits) is read
        // without waiting inside its frame (it fails the frame that finds an error).
        self.carry += f64::from(dt) / STEP;
        let steps = (self.carry + 1e-6).floor().max(0.0);
        self.carry -= steps;
        let steps = steps as u32;
        let rows = ingest.map_or(0, |i| i.3);
        self.write_params(sim, steps, rows);
        enc.clear_buffer(&self.work, 0, Some(self.cleared));
        enc.clear_buffer(&self.work, 4 * self.list_word, Some(4));
        let group = self.bind(sim, ingest);
        let nv = self.sims.len() as u32;
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vehicle coupling plan"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &group, &[]);
            pass.set_pipeline(&self.pipes[0]);
            pass.dispatch_workgroups(nv, 1, 1);
            pass.set_pipeline(&self.pipes[1]);
            pass.dispatch_workgroups(nv, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.work, 4 * self.list_word, &self.list_args, 0, 4);
        {
            // Only the vehicles with proxies (cpl_plan's list).
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vehicle coupling share"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &group, &[]);
            pass.set_pipeline(&self.pipes[2]);
            pass.dispatch_workgroups_indirect(&self.list_args, 0);
        }
        if let Some(m) = sim.marks() {
            m.mark(enc, "vehicle coupling plan");
        }
        // The rigid bodies push the vehicles: vehicle contact's frame bounds are planned
        // from this window's speeds.
        self.gpu.request_contact_replan();
        if std::env::var("RUBBLE_GPU_TIMING_EXPLICIT").is_ok() {
            self.gpu.set_marks(sim.marks().cloned());
        }
        let read = self.gpu.frame_encode(enc, steps);
        if let Some(m) = sim.marks() {
            m.mark(enc, "explicit frame");
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vehicle coupling loads"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &group, &[]);
            pass.set_pipeline(&self.pipes[3]);
            pass.dispatch_workgroups_indirect(&self.list_args, 0);
            pass.set_pipeline(&self.pipes[4]);
            pass.dispatch_workgroups(1, 1, 1);
            if self.gpu.sleep_on() {
                pass.set_pipeline(&self.pipes[7]);
                pass.dispatch_workgroups(nv.div_ceil(32).div_ceil(64), 1, 1);
            }
        }
        self.windows += 1;
        self.steps += u64::from(steps);
        self.emit_pending = steps > 0;
        self.emit_wakes = self.needs_step();
        self.read_pending = false;
        if self.windows % STATUS_EVERY == 1 && !self.status_busy.load(std::sync::atomic::Ordering::Acquire) {
            enc.copy_buffer_to_buffer(&self.work, 0, &self.status_buf, 0, 16);
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

    /// Mutable access (velocity writes, the grab, vehicle contact settings).
    pub fn explicit_vehicles_mut(&mut self) -> Option<&mut ExplicitBeamsGpu> {
        self.explicit.as_mut().map(|x| &mut x.gpu)
    }

    /// Blocking read (tests, tools): per explicit vehicle its sleep state (still steps,
    /// asleep); empty without vehicle sleep.
    pub fn explicit_sleep_states(&self) -> Vec<(u32, bool)> {
        self.explicit.as_ref().map_or_else(Vec::new, |x| x.gpu.sleep_states())
    }

    /// Explicit steps the vehicles have run, and windows (World steps) they took.
    pub fn explicit_vehicle_steps(&self) -> Option<(u64, u64)> {
        self.explicit.as_ref().map(|x| (x.steps, x.windows))
    }

    /// Blocking read (tests, tools): per vehicle, its last window's proxy shapes as
    /// (collider, impulse N s, angular impulse about the shape's reference point and that
    /// point): the impulses the next step's rigid step applies.
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
        let out = x.replan_words(&self.sim, steps, self.fracture.vehicle_ingest().ok().flatten(), serial);
        self.explicit = Some(x);
        out
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
            .and_then(|ingest| x.encode_frame(&self.sim, dt, ingest, enc));
        self.explicit = Some(x);
        r.map(Some)
    }

    pub(crate) fn explicit_after_submit(&mut self, read: Option<bool>) {
        if let (Some(x), Some(read)) = (self.explicit.as_mut(), read) {
            x.after_submit(read);
        }
    }
}
