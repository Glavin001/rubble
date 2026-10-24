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
use stress_gpu::fracture::{self, FractureTables, GpuFracture, INFO_SUPPORT};
use stress_gpu::gpu::Gpu;
use stress_gpu::ingest::{ColliderInfo, EngineIngest, KIND_BODY, KIND_STATIC, NONE};
use stress_ref::capability::{LoadModel, Profile, StrikeModel};
use stress_ref::engine::{BodyGeometry, BodyKey, EngineBody, RigidEngine};
use stress_ref::math::{Pose, Quat, Vec3};
use stress_ref::scene::{ImpactorDesc, Scene, Support};

use crate::{to_rubble, Owner, RubbleEngine};

/// The shots a world holds (the fracture lab's 16).
pub const SHOTS: usize = 16;

/// Carried impact pulses each half of the pool holds: two per contact the engine can hold
/// (a pulse comes from one and lasts its Hertz duration). A frame reserves room for each
/// chunk's carried pulses and one per manifold. `STRESS_PULSE_POOL` multiplies it (a
/// measurement).
pub fn pool_cap(max_contacts: u32) -> u32 {
    let k: u32 = std::env::var("STRESS_PULSE_POOL").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    2 * max_contacts * k.max(1)
}

/// The engine's contacts for a world of `chunks` chunks: a collapsed structure has every
/// piece resting on several others at a few points each (a whole suburb in a heap: about 8 a
/// chunk; 12 a chunk here), within 2^19 and what one binding of rubble's 256-byte contact
/// records can hold on this device (`max_binding` bytes). 2^19 is WebGPU's default binding
/// (128 MiB), so this fits any WebGPU device, and every buffer the world makes stays under
/// the default limits. More contacts than that in one step is an error the world reports
/// (rubble's overflow flags).
pub fn contact_capacity(chunks: usize, floor: u32, max_binding: u64) -> u32 {
    let device = (max_binding / 256).min(1 << 19) as u32;
    ((12 * chunks) as u32).next_power_of_two().max(floor).min(prev_power_of_two(device)).max(floor.min(device))
}

fn prev_power_of_two(n: u32) -> u32 {
    if n == 0 { 0 } else { 1 << (31 - n.leading_zeros()) }
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
    /// Analytic strikes by outcome (`impact_local`), in all.
    pub bounces: u32,
    pub craters: u32,
    pub spalls: u32,
    pub perforations: u32,
    /// The load path's largest failure index so far (`load_path`; 1 is a bond's strength).
    pub peak_utilization: f32,
    /// The in-step contact fracture's strikes and the bonds they broke, in all.
    pub contact_strikes: u32,
    pub contact_breaks: u32,
}

/// A shot's parked sphere: its engine body and collider.
#[derive(Clone, Copy, Debug)]
struct Shot {
    body: u32,
    collider: u32,
}

pub struct ThresholdWorld {
    /// The capability profile it runs (`threshold`, `load_path` or `impact_local`).
    pub profile: Profile,
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
    /// `STRESS_THRESHOLD_PHASES=1`: each part's summed wall time (ms).
    pub phases: Vec<(&'static str, f64)>,
    /// The ground's Young's modulus (the ingest's static side).
    ground_modulus: f64,
    /// The shots' material's (the scene's "projectile").
    projectile_modulus: f64,
    shot_radius: f64,
    /// What strike prediction knows of each shot (`GpuFracture::set_shots`).
    strikes: Vec<fracture::Shot>,
    /// Each table bond's (structure, bond) in the mirror.
    pub bond_source: Vec<(usize, usize)>,
    /// The in-step contact fracture's sweeps (0 off; `Profile::contact_fracture`, set when
    /// the world is made).
    contact_sweeps: u32,
    /// Whether a shot's strike is predicted at the end of the frame before it and its
    /// breaks made before the engine steps (forward only: the shot meets the pieces). Off,
    /// bonds break only on the contact loads after a step ([`Self::set_strike_prediction`];
    /// `STRESS_THRESHOLD_STRIKE=0` starts it off).
    predict_strikes: bool,
}

const RING: usize = 4;
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
        ThresholdWorld::with_profile(gpu, scene, shot_radius, shot_mass, Profile::Threshold)
    }

    /// [`Self::new`] under `profile` (one the fracture world runs: `threshold`, `load_path`,
    /// `impact_local`; `Profile::fracture_models`, with their environment overrides).
    pub fn with_profile(gpu: &Gpu, scene: &Scene, shot_radius: f64, shot_mass: f64, profile: Profile) -> Result<ThresholdWorld, String> {
        ThresholdWorld::with_contact_fracture(gpu, scene, shot_radius, shot_mass, profile, None)
    }

    /// [`Self::with_profile`] with the in-step contact fracture on or off (`None`: the
    /// profile's own, `Profile::contact_fracture`). On, a body about to strike a structure
    /// within a step has the strike's load solved by the load path inside that step (where
    /// rubble's broadphase ends, before its narrowphase and solve): the bonds break by
    /// their own strength and the structure splits, or nothing breaks and the striker
    /// rebounds. The shots are then rubble CCD bodies (time-of-impact contacts:
    /// `RubbleEngine::set_toi_ccd`), and strikes are not also predicted at the end of a
    /// frame (`set_strike_prediction` turns that on again).
    pub fn with_contact_fracture(gpu: &Gpu, scene: &Scene, shot_radius: f64, shot_mass: f64, profile: Profile, contact: Option<bool>) -> Result<ThresholdWorld, String> {
        if !matches!(profile, Profile::Threshold | Profile::LoadPath | Profile::ImpactLocal) {
            return Err(format!("the fracture world runs threshold, load_path or impact_local, not {}", profile.name()));
        }
        let (load_model, strike_model) = profile.fracture_models()?;
        let contact_sweeps = match (contact, profile.contact_fracture()?) {
            (Some(false), _) => 0,
            (Some(true), 0) => stress_ref::capability::CONTACT_SWEEPS,
            (_, n) => n,
        };
        let world = stress_ref::world::World::new(scene);
        let m = &world.solver;
        let chunk_count: usize = m.structures.iter().map(|s| s.chunks.len()).sum();
        // Capacity: a body per chunk at most, the shots, the ground.
        let mut capacity = crate::capacity_for(scene);
        let bodies_needed = (chunk_count + SHOTS + 4) as u32;
        capacity.max_bodies = capacity.max_bodies.max(bodies_needed.next_power_of_two());
        capacity.max_contacts = contact_capacity(chunk_count, capacity.max_contacts, gpu.device.limits().max_storage_buffer_binding_size as u64);
        capacity.max_candidates = capacity.max_candidates.max(capacity.max_contacts);
        let mut engine = RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), capacity.clone()).map_err(|e| format!("rubble: {e}"))?;
        let (iterations, substeps) = profile.engine_policy();
        engine.set_policy(iterations, substeps);
        if contact_sweeps > 0 {
            // The shots sweep their bounds over the step (a strike is a candidate pair
            // before it lands) and meet what they strike at its time of impact.
            engine.set_toi_ccd(true);
        }
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
                // With TOI CCD (`set_toi_ccd`), BODY_CCD on the slot: a fired shot gets
                // time-of-impact contacts.
                ccd: engine.toi_ccd(),
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
            contact_sweeps,
            max_candidates: engine.sim.capacity().max_candidates,
            pool_cap: pool_cap(capacity.max_contacts),
            // The load path takes each impact's mean over the frame (a load, as Blast's
            // solver takes it); the contact test its peak.
            impact_peak: load_model == LoadModel::Threshold,
            load_model,
            strike_model,
            gravity: scene.gravity.map(|g| g as f32),
            band: (Vec3::from_array(scene.gravity).norm() * scene.sim.frame_dt) as f32,
            dt: scene.sim.frame_dt as f32,
            strike_speed: stress_ref::world::tiered_impact_speed() as f32,
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
            // Strike prediction: the chunk's box and its plane-strain modulus.
            let mat = scene.material(&ch.material);
            let e_plane = mat.youngs_modulus * scene.sim.stiffness_scale / (1.0 - mat.poisson_ratio * mat.poisson_ratio);
            let (bc, h, bq) = (ch.center, ch.half_extents, rotation_quat(&ch.rotation));
            t.geometry.push([[f(bc.x), f(bc.y), f(bc.z), f(h.norm())], [f(h.x), f(h.y), f(h.z), f(e_plane)], [f(bq.x), f(bq.y), f(bq.z), f(bq.w)]]);
            t.info.push(if ch.support != Support::None { INFO_SUPPORT } else { 0 });
            t.chunk_body.push(collider_body[cols[0] as usize]);
        }
        // Each cluster's body frame (its structure's frame: a cluster's shapes are its
        // chunks at their structure-frame poses) and its bonds (and, for the load path and
        // the analytic strike, each bond's frame, law and material).
        let ext_needed = load_model != LoadModel::Threshold || strike_model != StrikeModel::Hertz || contact_sweeps > 0;
        let mut materials = stress_gpu::joint::MaterialTable::default();
        let mut bond_data: Vec<&stress_ref::solver::RtBond> = Vec::new();
        let mut bond_source = Vec::new();
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
                t.bonds.push([global(cl.structure, b.geometry.a), global(cl.structure, b.geometry.b), (capacity as f32).to_bits(), (b.geometry.area as f32).to_bits()]);
                bond_source.push((cl.structure, bi));
                if ext_needed {
                    let g = &b.geometry;
                    let law = stress_gpu::joint::bond_law(g, &b.stiffness, &b.strength, b.rebar.as_ref(), b.weibull, materials.index(&b.strength));
                    let v = |x: Vec3| [x.x as f32, x.y as f32, x.z as f32, 0.0];
                    let u = |w: [u32; 4]| w.map(f32::from_bits);
                    t.ext.bonds.push([v(g.centroid), v(g.normal), v(g.t1), law.geom0, law.geom1, law.stiff0, law.stiff1, law.rebar0, law.rebar1, u(law.ids)]);
                    bond_data.push(b);
                }
            }
        }
        if ext_needed {
            build_ext(&mut t, &chunk_of, m, scene, &bond_data, materials);
        }
        if let LoadModel::LoadPath { .. } = load_model {
            let net = load_path_network(&t, &bond_data);
            t.lp_init = load_path_warm_start(&t, &net);
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
        let mut fracture = GpuFracture::new(gpu, &t, &engine.engine_buffers_now(), &ingest)?;
        // The punch list's sphere (world_coupling.rs `predict_punch`): bounding radius
        // 1.01 r + 1e-3, face r.
        let e_shot = projectile.youngs_modulus * scene.sim.stiffness_scale / (1.0 - projectile.poisson_ratio * projectile.poisson_ratio);
        let strikes: Vec<fracture::Shot> = shots
            .iter()
            .map(|sh| fracture::Shot { body: sh.body, bound: (1.01 * shot_radius + 1e-3) as f32, face: shot_radius as f32, mass: shot_mass as f32, modulus: e_shot as f32, launch: 0 })
            .collect();
        // `STRESS_THRESHOLD_STRIKE=0` (a measurement): contact loads only, no strikes. With the
        // in-step contact fracture, strikes are its, not predicted at the end of a frame.
        let predict_strikes = contact_sweeps == 0 && !std::env::var("STRESS_THRESHOLD_STRIKE").is_ok_and(|v| v == "0");
        fracture.set_candidates(gpu, engine.sim.candidate_buffer(), &engine.engine_buffers_now(), &ingest);
        if predict_strikes {
            fracture.set_shots(gpu, &strikes)?;
        }
        // Nothing reads rubble's counters back each step, and a plan from counters read a
        // frame or two late would differ run to run (the warm-start hash's size, which
        // colours get their own dispatch: both change results). Fixed instead: the hash
        // sized for the contact capacity, every colour its own dispatch.
        engine.sim.set_plan_hints(capacity.max_contacts, capacity.max_colors.min(32));
        let ring = (0..RING)
            .map(|_| {
                let b = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("threshold stats"), size: 64 + COUNTER_BYTES, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                (b, 0, std::sync::Arc::new(std::sync::atomic::AtomicU8::new(IDLE)))
            })
            .collect();
        let mirror = world.solver.clone();
        Ok(ThresholdWorld { profile, mirror, scene: scene.clone(), engine, ingest, fracture, tables: t, chunk_of, frame_dt: scene.sim.frame_dt, frame: 0, shots, fired: 0, ring, ring_next: 0, last: ThresholdStats::default(), initial_bodies: m.clusters.len() as u32, phases: Vec::new(), ground_modulus, projectile_modulus: projectile.youngs_modulus, shot_radius, strikes, bond_source, predict_strikes, contact_sweeps })
    }

    /// Launch a shot (`desc`: the stress frame, its pose and velocity): its record written
    /// in both pose regions, its collider's mass in the ingest's table. No limit: past the
    /// [`SHOTS`] spheres, each shot reuses the oldest. False only for a world with none.
    pub fn launch(&mut self, gpu: &Gpu, desc: &ImpactorDesc) -> bool {
        let Some(sh) = self.shots.get(self.fired % self.shots.len().max(1)).copied() else { return false };
        self.fired += 1;
        let r = match desc.shape {
            stress_ref::scene::ImpactorShape::Sphere { radius } => radius,
            stress_ref::scene::ImpactorShape::Box { half_extents } => half_extents.iter().copied().fold(f64::INFINITY, f64::min),
        };
        let i = 0.4 * desc.mass * r * r;
        let p = to_rubble(Vec3::from_array(desc.position));
        let v = to_rubble(Vec3::from_array(desc.velocity));
        let w = to_rubble(Vec3::from_array(desc.angular_velocity));
        let ccd = if self.engine.toi_ccd() { engine_schema::BODY_CCD } else { 0 };
        let flags = engine_schema::BodyFlags::Dynamic as u32 | engine_schema::BodyFlags::Awake as u32 | ccd;
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
        let k = (self.fired - 1) % self.shots.len();
        self.strikes[k].launch = self.fired as u32;
        self.strikes[k].mass = desc.mass as f32;
        self.strikes[k].face = r as f32;
        self.strikes[k].bound = match desc.shape {
            stress_ref::scene::ImpactorShape::Sphere { radius } => 1.01 * radius + 1e-3,
            stress_ref::scene::ImpactorShape::Box { half_extents } => 1.01 * Vec3::from_array(half_extents).norm() + 1e-3,
        } as f32;
        if self.predict_strikes {
            // The table is a few hundred bytes; its count is unchanged.
            let _ = self.fracture.set_shots(gpu, &self.strikes);
        }
        true
    }

    /// Strike prediction on or off (on by default): off, a shot's breaks come only from the
    /// contact loads after the step that meets the wall, so the engine's contact sees the
    /// wall whole.
    pub fn set_strike_prediction(&mut self, gpu: &Gpu, on: bool) -> Result<(), String> {
        self.predict_strikes = on;
        self.fracture.set_shots(gpu, if on { &self.strikes } else { &[] })
    }

    pub fn strike_prediction(&self) -> bool {
        self.predict_strikes
    }

    /// The in-step contact fracture's sweeps, 0 off (set when the world is made:
    /// [`Self::with_contact_fracture`]).
    pub fn contact_fracture(&self) -> u32 {
        self.contact_sweeps
    }

    /// One frame: rubble's step, its contacts ingested, the loads, breaks and splits. One
    /// submission; nothing waited for.
    pub fn step_frame(&mut self, gpu: &Gpu) -> Result<(), String> {
        // `STRESS_THRESHOLD_PHASES=1` (a diagnostic): each part submitted and waited for,
        // its wall time summed in `phases`.
        let timing = std::env::var("STRESS_THRESHOLD_PHASES").is_ok_and(|v| v == "1");
        let mut clock = web_time::Instant::now();
        let mut mark = |name: &'static str, enc: &mut wgpu::CommandEncoder, phases: &mut Vec<(&'static str, f64)>| {
            if !timing {
                return;
            }
            let done = std::mem::replace(enc, gpu.device.create_command_encoder(&Default::default()));
            gpu.queue.submit([done.finish()]);
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            let ms = clock.elapsed().as_secs_f64() * 1e3;
            match phases.iter_mut().find(|p| p.0 == name) {
                Some(p) => p.1 += ms,
                None => phases.push((name, ms)),
            }
            clock = web_time::Instant::now();
        };
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        mark("(before)", &mut enc, &mut self.phases);
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
        } else if self.contact_sweeps > 0 {
            // The in-step contact fracture where the step's broadphase ends.
            let fracture = &mut self.fracture;
            self.engine.sim.step_encode_with(&mut enc, "broadphase", &mut |e| fracture.encode_contact_fracture(e)).map_err(|e| format!("rubble step failed: {e:?}"))?;
        } else {
            self.engine.sim.step_encode(&mut enc).map_err(|e| format!("rubble step failed: {e:?}"))?;
        }
        mark("rubble step", &mut enc, &mut self.phases);
        let e = self.engine.engine_buffers_now();
        self.ingest.substep(gpu, &mut enc, &e, self.engine.sim.last_contact_offset(), 0, self.frame_dt);
        self.ingest.merge(gpu, &mut enc, &e, 1);
        mark("ingest", &mut enc, &mut self.phases);
        self.fracture.encode_frame(gpu, &mut enc, self.engine.sim.steps_done());
        mark("fracture", &mut enc, &mut self.phases);
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
                    bounces: c[7],
                    craters: c[8],
                    spalls: c[9],
                    perforations: c[10],
                    peak_utilization: f32::from_bits(c[11]),
                    contact_strikes: c[12],
                    contact_breaks: c[13],
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
    /// Diagnostic read (waits): each shot's centre and velocity in the stress frame, and
    /// whether it is in flight (not parked).
    pub fn shot_states(&self, gpu: &Gpu) -> Vec<([f64; 3], [f64; 3], bool)> {
        let bodies: Vec<BodyState> = gpu.read_range(self.engine.sim.body_buffer(), 0, self.tables.body_count() as usize);
        let z = |v: [f32; 3]| [v[0] as f64, -v[2] as f64, v[1] as f64];
        self.shots.iter().map(|sh| {
            let b = &bodies[sh.body as usize];
            (z(b.position), z(b.linear_vel), b.inv_mass > 0.0)
        }).collect()
    }

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

    /// The load path's network as the tables give it (the host's twin; stress-ref
    /// load_path.rs), its bonds in table order.
    pub fn load_path_network(&self) -> stress_ref::load_path::Network {
        let bonds: Vec<&stress_ref::solver::RtBond> = self.bond_source.iter().map(|&(s, i)| &self.mirror.bonds[s][i]).collect();
        load_path_network(&self.tables, &bonds)
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

/// The load path's and the analytic strike's per-chunk tables: each chunk's bonds (the
/// load path's adjacency) and the struck material (the entry chunk's, `impact_local`): its
/// density, elasticity and rate law from the scene's material, its strengths its bonds'
/// least (as the pack authored them), the material's where it has no bond.
fn build_ext(t: &mut FractureTables, chunk_of: &[(usize, usize)], m: &stress_ref::solver::ReferenceSolver, scene: &Scene, bonds: &[&stress_ref::solver::RtBond], materials: stress_gpu::joint::MaterialTable) {
    let n = chunk_of.len();
    let mut incident: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (i, b) in t.bonds.iter().enumerate() {
        incident[b[0] as usize].push(i as u32);
        incident[b[1] as usize].push(i as u32);
    }
    let f = |x: f64| x as f32;
    for (k, &(s, c)) in chunk_of.iter().enumerate() {
        let ch = &m.structures[s].chunks[c];
        let mat = scene.material(&ch.material);
        let first = t.ext.adjacency.len() as u32;
        t.ext.adjacency.extend(&incident[k]);
        let (mut tensile, mut compressive, mut shear) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        for &i in &incident[k] {
            let st = &bonds[i as usize].strength;
            tensile = tensile.min(st.tensile);
            compressive = compressive.min(if st.compressive.is_finite() { st.compressive } else { mat.compressive_strength });
            shear = shear.min(st.cohesion.min(st.shear_cap));
        }
        if incident[k].is_empty() {
            (tensile, compressive, shear) = (mat.tensile_strength, mat.compressive_strength, mat.cohesion.min(mat.shear_cap.unwrap_or(f64::INFINITY)));
        }
        let dif = mat.dif;
        let d = dif.unwrap_or(stress_ref::material::DifParams { reference_rate: 1.0, exponent: 0.0, transition_rate: 1.0, exponent_high: 0.0, max: 0.0 });
        t.ext.chunks.push([
            [f32::from_bits(first), f32::from_bits(incident[k].len() as u32), 0.0, 0.0],
            [f(mat.density), f(tensile), f(compressive), f(shear)],
            [f(d.reference_rate), f(d.exponent), f(d.transition_rate), f(d.exponent_high)],
            [if dif.is_some() { f(d.max) } else { 0.0 }, f(mat.poisson_ratio), f(mat.youngs_modulus * scene.sim.stiffness_scale), 0.0],
        ]);
    }
    t.ext.materials = materials.materials.iter().map(|g| [g.strength, g.energy, g.dif, g.misc, g.kind_flags.map(f32::from_bits)]).collect();
}

/// The load path's network of the tables (stress-ref load_path.rs): the chunks as the GPU
/// has them (centre, mass, inertia, body, support), the bonds in table order.
fn load_path_network(t: &FractureTables, bonds: &[&stress_ref::solver::RtBond]) -> stress_ref::load_path::Network {
    use stress_ref::load_path::{Bond, Chunk, Network};
    use stress_ref::math::Mat3;
    let chunks: Vec<Chunk> = t
        .chunks
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let i = |a: f32| a as f64;
            let inertia = Mat3 { m: [[i(c[1][0]), i(c[1][3]), i(c[2][0])], [i(c[1][3]), i(c[1][1]), i(c[2][1])], [i(c[2][0]), i(c[2][1]), i(c[1][2])]] };
            Chunk { centre: Vec3::new(i(c[0][0]), i(c[0][1]), i(c[0][2])), mass: i(c[0][3]), inertia, body: t.chunk_body[k] as usize, support: t.info[k] & INFO_SUPPORT != 0 }
        })
        .collect();
    let net_bonds = t.bonds.iter().zip(bonds).map(|(b, d)| Bond::new(b[0] as usize, b[1] as usize, &d.geometry, &d.stiffness)).collect();
    Network::new(chunks, net_bonds)
}

/// The load path's warm start: every body settled under gravity (f64, solved to 1e-9 or
/// 20000 iterations), so frame 0 starts from equilibrium.
fn load_path_warm_start(t: &FractureTables, net: &stress_ref::load_path::Network) -> Vec<[[f32; 4]; 2]> {
    let g = Vec3::new(t.gravity[0] as f64, t.gravity[1] as f64, t.gravity[2] as f64);
    let loads: Vec<[f64; 6]> = net.chunks.iter().map(|c| [g.x * c.mass, g.y * c.mass, g.z * c.mass, 0.0, 0.0, 0.0]).collect();
    let loads = net.relieve(&loads);
    let fixed = net.fixed();
    let alive = vec![true; net.bonds.len()];
    let mut x = vec![[0.0; 6]; net.chunks.len()];
    net.solve(&loads, &fixed, &alive, &mut x, 1e-9, 20_000);
    x.iter().map(|d| [[d[0] as f32, d[1] as f32, d[2] as f32, 0.0], [d[3] as f32, d[4] as f32, d[5] as f32, 0.0]]).collect()
}

/// A rotation matrix's quaternion.
fn rotation_quat(m: &stress_ref::math::Mat3) -> Quat {
    let r = |i: usize, j: usize| m.m[i][j];
    let glam_m = glam::DMat3::from_cols(glam::DVec3::new(r(0, 0), r(1, 0), r(2, 0)), glam::DVec3::new(r(0, 1), r(1, 1), r(2, 1)), glam::DVec3::new(r(0, 2), r(1, 2), r(2, 2)));
    let q = glam::DQuat::from_mat3(&glam_m).normalize();
    Quat { w: q.w, x: q.x, y: q.y, z: q.z }
}
