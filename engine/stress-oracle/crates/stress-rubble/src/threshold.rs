//! The threshold profile's world on the GPU alone (engine/docs/stress-threshold-gpu.md):
//! rubble steps, its contacts become chunk loads (stress-gpu `EngineIngest`), and bonds
//! break and bodies split in place (stress-gpu `GpuFracture`), one submission a frame and
//! nothing read back. The host sets it up once and then only launches shots and, when it
//! wants them, reads counters a frame or more late without waiting ([`ThresholdWorld::poll`]).
//!
//! Set-up: the scene's world (stress-ref, once) gives the clusters as engine bodies;
//! rubble cooks them (`create_body`: one collider per chunk part). Then every other body a
//! split can make gets an inert slot, and every shot its parked sphere, so the engine's
//! body and collider counts never change again.

use std::collections::HashMap;

use engine_schema::BodyState;
use stress_gpu::fracture::{FractureTables, GpuFracture, INFO_SUPPORT};
use stress_gpu::gpu::Gpu;
use stress_gpu::ingest::{ColliderInfo, EngineIngest, KIND_BODY, KIND_STATIC, NONE};
use stress_ref::engine::{BodyGeometry, BodyKey, EngineBody, RigidEngine};
use stress_ref::math::{Pose, Quat, Vec3};
use stress_ref::scene::{ImpactorDesc, Scene, Support};

use crate::{to_rubble, Owner, RubbleEngine};

/// The shots a world holds (the fracture lab's 16).
pub const SHOTS: usize = 16;

/// Impact pulses carried past their frame per chunk (`STRESS_RESIDENT_PULSES`, default 8
/// here); a chunk that needs more is an error the world reports.
pub fn pulse_cap() -> u32 {
    std::env::var("STRESS_RESIDENT_PULSES").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
}

/// What [`ThresholdWorld::poll`] reads (a frame or more late).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ThresholdStats {
    /// Frames stepped when it was read.
    pub frame: u64,
    pub broken: u32,
    /// Bodies the structures are now (chunks' bodies, without the shots and the ground).
    pub bodies: u32,
    pub split_frames: u32,
    /// `GpuFracture` error bits (stress-gpu `fracture::Counters::error`).
    pub error: u32,
    /// rubble's counters of its last step: overflow bits, contacts, awake bodies.
    pub engine_overflow: u32,
    pub contacts: u32,
    pub awake: u32,
}

/// A shot's parked sphere: its engine body and collider.
#[derive(Clone, Copy, Debug)]
struct Shot {
    body: u32,
    collider: u32,
}

pub struct ThresholdWorld {
    /// The scene as the reference world set it up (structures, chunks: the renderer's
    /// meshes; read at set-up only, never stepped).
    pub mirror: stress_ref::solver::ReferenceSolver,
    pub scene: Scene,
    pub engine: RubbleEngine,
    pub ingest: EngineIngest,
    pub fracture: GpuFracture,
    pub tables: FractureTables,
    /// Each global chunk's (structure, chunk).
    pub chunk_of: Vec<(usize, usize)>,
    pub frame_dt: f64,
    pub frame: u64,
    shots: Vec<Shot>,
    fired: usize,
    /// Non-blocking counter reads: (buffer, frame it was copied after, mapped flag).
    ring: Vec<(wgpu::Buffer, u64, std::sync::Arc<std::sync::atomic::AtomicU8>)>,
    ring_next: usize,
    pub last: ThresholdStats,
    /// The structures' bodies at the start.
    initial_bodies: u32,
    /// The ground's Young's modulus (the ingest's static side).
    ground_modulus: f64,
    /// The shots' material's (the scene's "projectile").
    projectile_modulus: f64,
    shot_radius: f64,
}

const RING: usize = 3;
const COUNTER_BYTES: u64 = std::mem::size_of::<engine_schema::StepCounters>() as u64;
// Ring slot states.
const IDLE: u8 = 0;
const PENDING: u8 = 1;
const READY: u8 = 2;

impl ThresholdWorld {
    /// The world of `scene` (its capability profile applied, `with_derived`), with
    /// [`SHOTS`] parked spheres of `shot` (radius, mass; its material the scene's
    /// "projectile").
    pub fn new(gpu: &Gpu, scene: &Scene, shot_radius: f64, shot_mass: f64) -> Result<ThresholdWorld, String> {
        let world = stress_ref::world::World::new(scene);
        let m = &world.solver;
        let chunk_count: usize = m.structures.iter().map(|s| s.chunks.len()).sum();
        // Capacity: a body per chunk at most, the shots, the ground.
        let mut capacity = crate::capacity_for(scene);
        let bodies_needed = (chunk_count + SHOTS + 4) as u32;
        capacity.max_bodies = capacity.max_bodies.max(bodies_needed.next_power_of_two());
        let mut engine = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), capacity.clone()).map_err(|e| format!("rubble: {e}"))?;
        let profile = stress_ref::capability::Profile::Threshold;
        let (iterations, substeps) = profile.engine_policy();
        engine.set_policy(iterations, substeps);
        if let Some(g) = &scene.ground {
            engine.add_ground(g.height);
        }
        let ground_modulus = scene.ground.as_ref().map(|g| scene.material(&g.material).youngs_modulus).unwrap_or(3e10);
        // The clusters as the reference world gives them, then the shots, parked.
        let mut bodies = world.initial_engine_bodies();
        bodies.retain(|b| matches!(b.key, BodyKey::Cluster(_)));
        let projectile = scene.material("projectile");
        let park = |i: usize| Pose::new(Vec3::new(1e4 + 10.0 * i as f64, 1e4, -1e4), Quat::IDENTITY);
        // `STRESS_THRESHOLD_NO_SHOTS=1`, `STRESS_THRESHOLD_NO_SPARE=1` (diagnostics): no
        // parked shots, no spare slots (no split can take one).
        let shot_count = if std::env::var("STRESS_THRESHOLD_NO_SHOTS").is_ok_and(|v| v == "1") { 0 } else { SHOTS };
        for i in 0..shot_count {
            bodies.push(EngineBody {
                key: BodyKey::Impactor(i),
                signature: 0,
                geometry: BodyGeometry::Sphere { radius: shot_radius, mass: shot_mass },
                fixed: true,
                ccd: false,
                pose: park(i),
                velocity: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            });
        }
        engine.sync_bodies(&bodies);
        let shots: Vec<Shot> = (0..shot_count)
            .map(|i| {
                let e = &engine.bodies[&BodyKey::Impactor(i)];
                Shot { body: e.id, collider: e.colliders.start }
            })
            .collect();
        // Inert slots for every body a split can make (static, out of every pair: no
        // collider names them until a split re-parents one).
        let used = engine.sim.body_count() as usize;
        let spare = if std::env::var("STRESS_THRESHOLD_NO_SPARE").is_ok_and(|v| v == "1") { 0 } else { chunk_count.saturating_sub(m.clusters.len()) };
        let inert = BodyState { position: [0.0, -1e4, 0.0], orientation: [0.0, 0.0, 0.0, 1.0], ..bytemuck::Zeroable::zeroed() };
        engine.sim.add_bodies(&vec![inert; spare]);
        let body_count = engine.sim.body_count();

        // The chunks, numbered structure by structure.
        let mut offset = Vec::with_capacity(m.structures.len());
        let mut chunk_of = Vec::with_capacity(chunk_count);
        for (s, st) in m.structures.iter().enumerate() {
            offset.push(chunk_of.len() as u32);
            chunk_of.extend((0..st.chunks.len()).map(|c| (s, c)));
        }
        let global = |s: usize, c: usize| offset[s] + c as u32;
        // Each chunk's colliders (in part order) from rubble's owners.
        let mut chunk_cols: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
        let mut collider_body: Vec<u32> = vec![u32::MAX; engine.owners.len()];
        for (i, o) in engine.owners.iter().enumerate() {
            if let Owner::Body { key, chunk } = o {
                collider_body[i] = engine.bodies[key].id;
                if let Some(sc) = chunk {
                    chunk_cols.entry(*sc).or_default().push(i as u32);
                }
            }
        }
        let mut t = FractureTables {
            max_bodies: engine.sim.capacity().max_bodies,
            pulse_cap: pulse_cap(),
            impact_peak: true,
            band: (Vec3::from_array(scene.gravity).norm() * scene.sim.frame_dt) as f32,
            dt: scene.sim.frame_dt as f32,
            frames: vec![[[0.0; 4], [0.0, 0.0, 0.0, 1.0]]; body_count as usize],
            free: vec![0; body_count as usize],
            ..Default::default()
        };
        for s in used..body_count as usize {
            t.free[s] = 1;
        }
        // Per chunk: its parts' mass properties summed (as cook_with sums them), its
        // colliders, its material.
        for &(s, c) in &chunk_of {
            let ch = &m.structures[s].chunks[c];
            let parts = engine.chunk_parts.get(&(s, c)).ok_or_else(|| format!("chunk {s}/{c}: no parts cooked"))?;
            let cols = chunk_cols.get(&(s, c)).ok_or_else(|| format!("chunk {s}/{c}: no colliders"))?;
            if parts.len() != cols.len() {
                return Err(format!("chunk {s}/{c}: {} parts, {} colliders", parts.len(), cols.len()));
            }
            let mass: f64 = parts.iter().map(|p| p.mass).sum();
            let centre = parts.iter().fold(Vec3::ZERO, |a, p| a + p.center * p.mass) * (1.0 / mass.max(1e-300));
            let mut i = stress_ref::math::Mat3::ZERO;
            for p in parts {
                let d = p.center - centre;
                i += p.inertia + (stress_ref::math::Mat3::diag(Vec3::splat(d.norm2())) - stress_ref::math::Mat3::outer(d, d)) * p.mass;
            }
            let first = t.parts.len() as u32;
            for (p, &col) in parts.iter().zip(cols) {
                let q = p.rotation;
                t.parts.push([[p.origin.x as f32, p.origin.y as f32, p.origin.z as f32, f32::from_bits(col)], [q.x as f32, q.y as f32, q.z as f32, q.w as f32]]);
            }
            let f = |x: f64| x as f32;
            t.chunks.push([
                [f(centre.x), f(centre.y), f(centre.z), f(mass)],
                [f(i.m[0][0]), f(i.m[1][1]), f(i.m[2][2]), f(i.m[0][1])],
                [f(i.m[0][2]), f(i.m[1][2]), f32::from_bits(first), f32::from_bits(parts.len() as u32)],
                [f(ch.youngs_modulus), f(ch.poisson_ratio), f(ch.volume().cbrt()), f(ch.half_extents.min_elem())],
            ]);
            t.info.push(if ch.support != Support::None { INFO_SUPPORT } else { 0 });
            t.chunk_body.push(collider_body[cols[0] as usize]);
        }
        // Each cluster's body frame (its structure's frame: a cluster's shapes are its
        // chunks at their structure-frame poses) and its bonds.
        for cl in &m.clusters {
            let e = &engine.bodies[&BodyKey::Cluster(cl.id)];
            let p = e.principal;
            t.frames[e.id as usize] = [[e.com.x as f32, e.com.y as f32, e.com.z as f32, f32::from_bits(cl.structure as u32)], [p.x as f32, p.y as f32, p.z as f32, p.w as f32]];
            for &bi in &cl.bonds {
                let b = &m.bonds[cl.structure][bi];
                if !b.connected() {
                    continue;
                }
                let capacity = b.strength.tensile * b.geometry.area * b.weibull;
                t.bonds.push([global(cl.structure, b.geometry.a), global(cl.structure, b.geometry.b), (capacity as f32).to_bits(), 0]);
            }
        }
        // The ingest's collider table: chunks' colliders on their bodies, the shots, the
        // ground (static).
        let mut root_of_body: HashMap<u32, usize> = HashMap::new();
        for (k, &b) in t.chunk_body.iter().enumerate() {
            root_of_body.entry(b).or_insert(k);
        }
        let mut table = vec![ColliderInfo { a: [NONE, NONE, KIND_STATIC, 0], b: [f32::INFINITY, ground_modulus as f32, 1.0, 0.0] }; engine.owners.len()];
        for &(s, c) in &chunk_of {
            let k = global(s, c) as usize;
            let body = t.chunk_body[k];
            let root = root_of_body[&body];
            let inv = engine.bodies.values().find(|e| e.id == body).map_or(0.0, |e| e.inv_mass);
            let mass = if inv > 0.0 { 1.0 / inv } else { f32::INFINITY };
            let r = t.chunks[root][3];
            for &col in &chunk_cols[&(s, c)] {
                table[col as usize] = ColliderInfo { a: [k as u32, body, KIND_BODY, 1], b: [mass, r[0], r[3], 0.0] };
            }
        }
        for sh in &shots {
            table[sh.collider as usize] = ColliderInfo { a: [NONE, sh.body, KIND_BODY, 0], b: [shot_mass as f32, projectile.youngs_modulus as f32, shot_radius as f32, 0.0] };
        }
        let ingest = EngineIngest::new(gpu, capacity.max_contacts, 1, capacity.max_bodies, chunk_count as u32, &table);
        let fracture = GpuFracture::new(gpu, &t, &engine.engine_buffers_now(), &ingest)?;
        let ring = (0..RING)
            .map(|_| {
                let b = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("threshold stats"), size: 64 + COUNTER_BYTES, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                (b, 0, std::sync::Arc::new(std::sync::atomic::AtomicU8::new(IDLE)))
            })
            .collect();
        let mirror = world.solver.clone();
        Ok(ThresholdWorld { mirror, scene: scene.clone(), engine, ingest, fracture, tables: t, chunk_of, frame_dt: scene.sim.frame_dt, frame: 0, shots, fired: 0, ring, ring_next: 0, last: ThresholdStats::default(), initial_bodies: m.clusters.len() as u32, ground_modulus, projectile_modulus: projectile.youngs_modulus, shot_radius })
    }

    /// Launch the next parked shot (`desc`: the stress frame, its pose and velocity):
    /// its record written in both pose regions, its collider's mass in the ingest's table.
    /// False when all are fired.
    pub fn launch(&mut self, gpu: &Gpu, desc: &ImpactorDesc) -> bool {
        let Some(sh) = self.shots.get(self.fired).copied() else { return false };
        self.fired += 1;
        let r = match desc.shape {
            stress_ref::scene::ImpactorShape::Sphere { radius } => radius,
            stress_ref::scene::ImpactorShape::Box { half_extents } => half_extents.iter().copied().fold(f64::INFINITY, f64::min),
        };
        let i = 0.4 * desc.mass * r * r;
        let p = to_rubble(Vec3::from_array(desc.position));
        let v = to_rubble(Vec3::from_array(desc.velocity));
        let w = to_rubble(Vec3::from_array(desc.angular_velocity));
        let flags = engine_schema::BodyFlags::Dynamic as u32 | engine_schema::BodyFlags::Awake as u32;
        let record = BodyState {
            position: p,
            inv_mass: (1.0 / desc.mass) as f32,
            orientation: [0.0, 0.0, 0.0, 1.0],
            inv_inertia: [(1.0 / i) as f32; 3],
            flags,
            linear_vel: v,
            angular_vel: w,
            ..bytemuck::Zeroable::zeroed()
        };
        let stride = std::mem::size_of::<BodyState>() as u64;
        let max = self.engine.sim.capacity().max_bodies as u64;
        for region in [0, max] {
            gpu.queue.write_buffer(self.engine.sim.body_buffer(), (region + sh.body as u64) * stride, bytemuck::bytes_of(&record));
        }
        let info = ColliderInfo { a: [NONE, sh.body, KIND_BODY, 0], b: [desc.mass as f32, self.projectile_modulus as f32, r as f32, 0.0] };
        gpu.queue.write_buffer(&self.ingest.collider_info, std::mem::size_of::<ColliderInfo>() as u64 * sh.collider as u64, bytemuck::bytes_of(&info));
        true
    }

    /// One frame: rubble's step, its contacts ingested, the loads, breaks and splits. One
    /// submission; nothing waited for.
    pub fn step_frame(&mut self, gpu: &Gpu) -> Result<(), String> {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let e = self.engine.engine_buffers_now();
        self.ingest.begin_frame(&mut enc, &e);
        let mut params = self.engine.sim.params();
        params.dt = self.frame_dt as f32;
        // The resident frame's speculative contact range (`ResidentEngine::step_resident`).
        params.contact_offset = 0.01;
        self.engine.sim.set_params(params);
        // `STRESS_THRESHOLD_SUBMIT=1` (a diagnostic): rubble's step its own submission.
        if std::env::var("STRESS_THRESHOLD_SUBMIT").is_ok_and(|v| v == "1") {
            let done = std::mem::replace(&mut enc, gpu.device.create_command_encoder(&Default::default()));
            gpu.queue.submit([done.finish()]);
            self.engine.sim.step_submit().map_err(|e| format!("rubble step failed: {e:?}"))?;
        } else {
            self.engine.sim.step_encode(&mut enc).map_err(|e| format!("rubble step failed: {e:?}"))?;
        }
        let e = self.engine.engine_buffers_now();
        self.ingest.substep(gpu, &mut enc, &e, self.engine.sim.last_contact_offset(), 0, self.frame_dt);
        self.ingest.merge(gpu, &mut enc, &e, 1);
        self.fracture.encode_frame(gpu, &mut enc, self.engine.sim.steps_done());
        self.frame += 1;
        // The counters into the ring's next free slot (a read when the host polls).
        let slot = self.ring_next;
        if self.ring[slot].2.load(std::sync::atomic::Ordering::Acquire) == IDLE {
            enc.copy_buffer_to_buffer(&self.fracture.atom, 4 * self.fracture.layout.a_counters() as u64, &self.ring[slot].0, 0, 64);
            enc.copy_buffer_to_buffer(self.engine.sim.counters_buffer(), 0, &self.ring[slot].0, 64, COUNTER_BYTES);
            self.ring[slot].1 = self.frame;
            self.ring_next = (slot + 1) % RING;
            gpu.queue.submit([enc.finish()]);
            let flag = self.ring[slot].2.clone();
            flag.store(PENDING, std::sync::atomic::Ordering::Release);
            self.ring[slot].0.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                if r.is_ok() {
                    flag.store(READY, std::sync::atomic::Ordering::Release);
                }
            });
        } else {
            gpu.queue.submit([enc.finish()]);
        }
        Ok(())
    }

    /// The newest counters the GPU has finished, without waiting (`device.poll` without
    /// blocking); `last` keeps them.
    pub fn poll(&mut self, gpu: &Gpu) -> ThresholdStats {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        for (buf, frame, flag) in &self.ring {
            if flag.load(std::sync::atomic::Ordering::Acquire) != READY {
                continue;
            }
            let c: Vec<u32> = match buf.slice(..).get_mapped_range() {
                Ok(view) => bytemuck::cast_slice(&view).to_vec(),
                Err(_) => continue,
            };
            buf.unmap();
            flag.store(IDLE, std::sync::atomic::Ordering::Release);
            if *frame >= self.last.frame {
                let e: engine_schema::StepCounters = bytemuck::pod_read_unaligned(bytemuck::cast_slice(&c[16..16 + COUNTER_BYTES as usize / 4]));
                self.last = ThresholdStats {
                    frame: *frame,
                    broken: c[1],
                    bodies: self.initial_bodies + c[2],
                    split_frames: c[4],
                    error: c[3],
                    engine_overflow: e.overflow_flags,
                    contacts: e.contact_count,
                    awake: e.awake_count,
                };
            }
        }
        self.last
    }

    /// Diagnostic read (waits): every chunk's centre in the world (the stress frame), from
    /// its body's pose and the body's frame, and its body's speed (m/s).
    pub fn chunk_centres(&self, gpu: &Gpu) -> Vec<([f64; 3], f64)> {
        let bodies: Vec<BodyState> = gpu.read_range(self.engine.sim.body_buffer(), 0, self.tables.body_count() as usize);
        let frames = self.fracture.body_frames(gpu);
        let owner = self.fracture.chunk_bodies(gpu);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let axes = Quat { w: h, x: h, y: 0.0, z: 0.0 };
        (0..self.tables.chunk_count() as usize)
            .map(|k| {
                let b = &bodies[owner[k] as usize];
                let f = &frames[owner[k] as usize];
                let rot = (axes * Quat { x: b.orientation[0] as f64, y: b.orientation[1] as f64, z: b.orientation[2] as f64, w: b.orientation[3] as f64 }).normalized();
                let p = Quat { x: f[1][0] as f64, y: f[1][1] as f64, z: f[1][2] as f64, w: f[1][3] as f64 };
                let q = (rot * p.conjugate()).normalized();
                let pos = crate::from_rubble(b.position);
                let com = Vec3::new(f[0][0] as f64, f[0][1] as f64, f[0][2] as f64);
                let c = &self.tables.chunks[k][0];
                let world = pos + q.rotate(Vec3::new(c[0] as f64, c[1] as f64, c[2] as f64) - com);
                let v = Vec3::new(b.linear_vel[0] as f64, b.linear_vel[1] as f64, b.linear_vel[2] as f64).norm() + Vec3::new(b.angular_vel[0] as f64, b.angular_vel[1] as f64, b.angular_vel[2] as f64).norm() * 5.0;
                (world.to_array(), v)
            })
            .collect()
    }

    /// The shots as impactors (their meshes, for a renderer's shot objects).
    pub fn shot_impactors(&self) -> Vec<stress_ref::world::Impactor> {
        (0..self.shots.len())
            .map(|k| {
                let desc = ImpactorDesc {
                    name: format!("shot{k}"),
                    shape: stress_ref::scene::ImpactorShape::Sphere { radius: self.shot_radius },
                    mass: 1.0,
                    position: [0.0; 3],
                    orientation: [1.0, 0.0, 0.0, 0.0],
                    velocity: [0.0; 3],
                    angular_velocity: [0.0; 3],
                    material: "projectile".into(),
                    crush: None,
                };
                stress_ref::world::Impactor::from_desc(&desc, &self.scene)
            })
            .collect()
    }

    /// The view's tables for `renderer` (made from [`Self::mirror`] and
    /// [`Self::shot_impactors`]): each chunk's object, render centre, rotation and colour.
    pub fn attach_view(&mut self, gpu: &Gpu, renderer: &crate::render::Renderer) {
        let (_, _, chunk_object, impactor_base) = renderer.resident_targets();
        let mut chunks = Vec::with_capacity(self.chunk_of.len());
        for &(s, c) in &self.chunk_of {
            let ch = &self.mirror.structures[s].chunks[c];
            let q = rotation_quat(&ch.rotation);
            let col = crate::render::material_color(&ch.material);
            chunks.push([
                [ch.center.x as f32, ch.center.y as f32, ch.center.z as f32, f32::from_bits(chunk_object[s][c])],
                [q.x as f32, q.y as f32, q.z as f32, q.w as f32],
                [col[0], col[1], col[2], 0.0],
            ]);
        }
        let shot_bodies = self.shots.iter().map(|s| s.body).collect();
        self.fracture.set_view(gpu, &stress_gpu::fracture::ViewTables { chunks, shot_bodies, impactor_base });
    }

    /// The renderer's objects and (with `bonds`) bond lines from the bodies as they are,
    /// recorded into `encoder` (then `Renderer::draw_resident`). Returns the line vertices.
    pub fn encode_view(&mut self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, renderer: &crate::render::Renderer, mode: u32, bonds: bool) -> u32 {
        let (objects, lines, _, _) = renderer.resident_targets();
        let bonds = bonds && 2 * self.tables.bond_count() as usize <= renderer.line_capacity();
        self.fracture.encode_view(gpu, encoder, objects, lines, mode, bonds);
        if bonds { 2 * self.tables.bond_count() } else { 0 }
    }

    /// Shots launched so far.
    pub fn fired(&self) -> usize {
        self.fired
    }

    /// The ground's modulus (the ingest's static side).
    pub fn ground_modulus(&self) -> f64 {
        self.ground_modulus
    }
}

impl RubbleEngine {
    /// The engine's buffers as the ingest and the fracture bind them.
    pub fn engine_buffers_now(&self) -> stress_gpu::ingest::EngineBuffers<'_> {
        stress_gpu::ingest::EngineBuffers { bodies: self.sim.body_buffer(), colliders: self.sim.collider_buffer(), contacts: self.sim.contact_buffer(), counters: self.sim.counters_buffer() }
    }
}

/// A rotation matrix's quaternion.
fn rotation_quat(m: &stress_ref::math::Mat3) -> Quat {
    let r = |i: usize, j: usize| m.m[i][j];
    let glam_m = glam::DMat3::from_cols(glam::DVec3::new(r(0, 0), r(1, 0), r(2, 0)), glam::DVec3::new(r(0, 1), r(1, 1), r(2, 1)), glam::DVec3::new(r(0, 2), r(1, 2), r(2, 2)));
    let q = glam::DQuat::from_mat3(&glam_m).normalize();
    Quat { w: q.w, x: q.x, y: q.y, z: q.z }
}
