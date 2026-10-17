//! The scenes and the pipeline: stress-gpu's engine-coupled world (`GpuWorld::with_engine`)
//! with rubble's AVBD engine on the same device and no frame rewinds
//! (`redo_fractured_frames = false`), as `stress_bench --solver rubble-gpu` runs it.

use std::time::Instant;

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::material::Material;
use stress_ref::math::{Mat3, Pose, Quat, Vec3};
use stress_ref::scene::{ImpactorDesc, ImpactorShape, ProbeKind, Scene};
use stress_ref::world::Impactor;
use stress_rubble::scenes;

/// The viewer's scenes (keys 1-9 in this order).
pub const SCENES: &[(&str, &str)] = &[
    ("wall_ram", "free concrete wall, 200 kg steel ram at 40 m/s"),
    ("b5_v10", "b5 wall anchored at its base, 1000 kg ram at 10 m/s"),
    ("b5_v20", "b5 wall anchored at its base, 1000 kg ram at 20 m/s"),
    ("b5_v40", "b5 wall anchored at its base, 1000 kg ram at 40 m/s"),
    ("house_ram", "house-1story struck by a 400 kg box at 12 m/s"),
    ("house_idle", "house-1story, intact"),
    ("town", "16 house-1story copies, one rammed (town16_one_ram)"),
    ("town_idle", "16 house-1story copies, intact (town16_idle)"),
    ("veneer", "the cooked veneer house (veneer-house.glb, level 0)"),
];

/// Projectiles fired with a click: steel spheres.
#[derive(Clone, Copy, Debug)]
pub struct ShotSpec {
    pub radius: f64,
    pub speed: f64,
}

const SHOT_DENSITY: f64 = 7850.0;

/// Projectiles per run at most: impactors are keyed by index in the coupled world, so
/// they are never removed (R resets).
pub const MAX_SHOTS: usize = 48;

/// The scene `name` in the benchmark's real-time configuration, and its projectile.
/// Anything else is a scene JSON path.
pub fn scene(name: &str) -> Result<(Scene, ShotSpec), String> {
    // The viewer runs open-ended; `duration` only bounds batch runs.
    let rt = |s: Scene| scenes::realtime(s, 1e9);
    let wall_shot = ShotSpec { radius: 0.06, speed: 40.0 };
    let house_shot = ShotSpec { radius: 0.25, speed: 25.0 };
    let b5 = |v: f64, label: &str| (rt(stress_ref::builders::wall_impact(v, label)), wall_shot);
    Ok(match name {
        "wall_ram" => (rt(scenes::wall_ram()), wall_shot),
        "b5_v10" => b5(10.0, "b5_v10"),
        "b5_v20" => b5(20.0, "b5_v20"),
        "b5_v40" => b5(40.0, "b5_v40"),
        "house_ram" => (rt(stress_ref::scenes::rammed(scenes::house())), house_shot),
        "house_idle" => (rt(scenes::house()), house_shot),
        "town" => (rt(stress_ref::scenes::bombarded(stress_ref::scenes::pack_town(&scenes::house(), 16, 2.0), 1)), house_shot),
        "town_idle" => (rt(stress_ref::scenes::pack_town(&scenes::house(), 16, 2.0)), house_shot),
        "veneer" => (rt(veneer()?), house_shot),
        path => {
            let s = Scene::load(std::path::Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
            (rt(s), house_shot)
        }
    })
}

/// The veneer house as rubble's stress cooker reads it (`engine_fracture::cook`, which
/// writes its `.stress.bin`): the fracture GLB's level 0 through its Shardwright stress
/// export and stress-ref's import. Its chunks are compound hulls, which the GPU world
/// does not run yet (`stress_gpu::world::unsupported`): the viewer says so.
fn veneer() -> Result<Scene, String> {
    let file = "veneer-house.glb";
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../crates/engine-webgpu/fracture/assets").join(file);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let asset = engine_fracture::parse_fracture_glb(&bytes).map_err(|e| format!("{file}: {e}"))?;
    let json = engine_fracture::stress_export::shardwright_stress_json(&asset, 0);
    let scene = stress_ref::shardwright::import_value(&json, &asset.physics.name)?;
    scene.validate()?;
    Ok(scene.with_derived())
}

/// One frame's wall-clock split (milliseconds).
#[derive(Clone, Copy, Debug, Default)]
pub struct Times {
    pub total: f64,
    /// The stress solver: its GPU submissions (waited for), readbacks, buffer builds,
    /// host splits and static solves on the mirror.
    pub stress: f64,
    /// Of `stress`: waiting for its GPU work.
    pub stress_gpu: f64,
    /// rubble's engine (`stress_rubble::profile`).
    pub engine: f64,
    /// The rest: the coupled frame's host logic (islands, contact plans, write-back).
    pub host: f64,
}

pub struct Sim {
    pub name: String,
    pub scene: Scene,
    pub world: GpuWorld,
    /// The scene's scripted impactors, held until the first click (`--no-hold`: none).
    pub armed: Vec<ImpactorDesc>,
    pub shot: ShotSpec,
    shot_material: Material,
    pub shots: usize,
    pub last: Times,
    pub frames: u64,
}

impl Sim {
    pub fn new(gpu: &Gpu, name: &str, hold: bool) -> Result<Sim, String> {
        let (scene, shot) = scene(name)?;
        if let Some(why) = stress_gpu::world::unsupported(&scene) {
            return Err(format!("{name} does not run on the GPU pipeline yet: {why}"));
        }
        let mut start = scene.clone();
        let armed = if hold { std::mem::take(&mut start.impactors) } else { Vec::new() };
        // Probes on a held impactor have nothing to read until it is launched (probes
        // only record; nothing in the solve reads them).
        start.probes.retain(|p| match &p.kind {
            ProbeKind::ImpactorVelocity { impactor, .. } | ProbeKind::ImpactorPosition { impactor, .. } => !armed.iter().any(|d| &d.name == impactor),
            _ => true,
        });
        let shot_material = match scene.impactors.first() {
            Some(d) => scene.material(&d.material).clone(),
            None => scene.materials.values().max_by(|a, b| a.youngs_modulus.total_cmp(&b.youngs_modulus)).cloned().ok_or("the scene has no materials")?,
        };
        let engine = stress_rubble::RubbleEngine::with_device(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone())
            .map_err(|e| format!("rubble engine: {e}"))?;
        let mut world = GpuWorld::with_engine(gpu, &start, Box::new(engine))?;
        // rubble's production pipeline: impact islands routed before the step, no rewinds.
        world.redo_fractured_frames = false;
        stress_rubble::profile::take();
        Ok(Sim { name: name.to_string(), scene, world, armed, shot, shot_material, shots: 0, last: Times::default(), frames: 0 })
    }

    fn stress_seconds(&self) -> (f64, f64) {
        let p = &self.world.solver.profile;
        (p.build + p.gpu + p.readback + p.download + p.split + self.world.tail_seconds, p.gpu)
    }

    /// One frame (`sim.frame_dt`), timed.
    pub fn step(&mut self, gpu: &Gpu) -> Result<(), String> {
        let (s0, g0) = self.stress_seconds();
        let t = Instant::now();
        self.world.step_frame(gpu)?;
        let total = t.elapsed().as_secs_f64() * 1e3;
        // Top-level engine calls only (`step.sim` and `step.contacts` are inside `step`).
        let engine: f64 = stress_rubble::profile::take().iter().filter(|e| !e.0.contains('.')).map(|e| e.2).sum();
        let (s1, g1) = self.stress_seconds();
        let stress = (s1 - s0) * 1e3;
        self.last = Times { total, stress, stress_gpu: (g1 - g0) * 1e3, engine, host: (total - stress - engine).max(0.0) };
        self.frames += 1;
        Ok(())
    }

    pub fn frame_dt(&self) -> f64 {
        self.scene.sim.frame_dt
    }

    pub fn time(&self) -> f64 {
        self.world.solver.mirror.time
    }

    /// Clusters (one per fragment, as the observation counts them).
    pub fn fragments(&self) -> usize {
        self.world.solver.mirror.clusters.len()
    }

    /// The fastest cluster's speed (m/s).
    pub fn max_speed(&self) -> f64 {
        self.world.solver.mirror.clusters.iter().map(|c| c.velocity.norm()).fold(0.0, f64::max)
    }

    pub fn broken_bonds(&self) -> usize {
        self.world.solver.mirror.broken_bond_count()
    }

    pub fn impactor_count(&self) -> usize {
        self.world.solver.impactors.len()
    }

    /// Speed of the scene's first impactor (the ram), once launched.
    pub fn ram_speed(&self) -> Option<f64> {
        (!self.scene.impactors.is_empty() && self.armed.is_empty()).then(|| self.world.solver.impactors.first().map(|i| i.velocity.norm())).flatten()
    }

    /// Launch the held scripted impactors as the scene describes them. False if none.
    pub fn launch_armed(&mut self, gpu: &Gpu) -> Result<bool, String> {
        if self.armed.is_empty() {
            return Ok(false);
        }
        for d in std::mem::take(&mut self.armed) {
            self.world.add_impactor(gpu, Impactor::from_desc(&d, &self.scene))?;
        }
        Ok(true)
    }

    /// Fire a projectile from `origin` along `dir`.
    pub fn fire(&mut self, gpu: &Gpu, origin: Vec3, dir: Vec3) -> Result<(), String> {
        if self.shots >= MAX_SHOTS {
            return Err(format!("{MAX_SHOTS} projectiles fired: R resets"));
        }
        let r = self.shot.radius;
        let mass = SHOT_DENSITY * 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
        self.shots += 1;
        let imp = Impactor {
            name: format!("shot {}", self.shots),
            shape: ImpactorShape::Sphere { radius: r },
            mass,
            inertia: Mat3::IDENTITY * (0.4 * mass * r * r),
            pose: Pose::new(origin, Quat::IDENTITY),
            velocity: dir.normalized() * self.shot.speed,
            angular_velocity: Vec3::ZERO,
            material: self.shot_material.clone(),
            crush: None,
            crush_used: 0.0,
            crush_depth: 0.0,
            driven: false,
        };
        self.world.add_impactor(gpu, imp)?;
        Ok(())
    }

    /// The structures' bounds (chunk centres and half extents).
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let (mut lo, mut hi) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
        for c in self.scene.bodies.iter().flat_map(|b| b.chunks.iter()) {
            let (p, h) = (Vec3::from_array(c.center), Vec3::from_array(c.half_extents));
            lo = lo.component_min(p - h);
            hi = hi.component_max(p + h);
        }
        (lo, hi)
    }
}

/// `DESTRUCTION_VIEW_HOST_CHUNKS=1`: every chunk drawn from the mirror instead of the
/// GPU state (checks the host path against the GPU one).
pub fn host_chunks_only() -> bool {
    std::env::var("DESTRUCTION_VIEW_HOST_CHUNKS").is_ok_and(|v| v == "1")
}

/// A chunk the solver keeps out of its GPU tables, as `destruction_render.slang`'s
/// `HostChunk` (world box; centre w: utilization, < 0 without an unbroken bond).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HostChunk {
    pub center: [f32; 4],
    pub axes: [[f32; 4]; 3],
}

/// One vertex of a chunk's hull triangles in the chunk's own frame, as
/// `destruction_render.slang`'s `HullVertex` (pos.w: the chunk's index).
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HullVertex {
    pub pos: [f32; 4],
    pub normal: [f32; 4],
}

fn push_hull(ch: &stress_ref::structure::ChunkData, index: usize, out: &mut Vec<HullVertex>) {
    for (p, n) in stress_rubble::render::chunk_triangles(ch) {
        out.push(HullVertex { pos: [p[0], p[1], p[2], index as f32], normal: [n[0], n[1], n[2], 0.0] });
    }
}

impl Sim {
    /// Hull triangles of every chunk in the GPU tables, tagged with its table index (they
    /// change only with `host_chunk_signature`).
    pub fn chunk_hulls(&self) -> Vec<HullVertex> {
        let m = &self.world.solver.mirror;
        let mut out = Vec::new();
        for (k, &(s, c)) in self.world.solver.chunk_order().iter().enumerate() {
            push_hull(&m.structures[s].chunks[c], k, &mut out);
        }
        out
    }

    /// A signature of what the GPU tables hold and of the topology: the host chunks
    /// change only with it (excluded clusters do not move while out).
    pub fn host_chunk_signature(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.world.solver.chunk_order().hash(&mut h);
        let m = &self.world.solver.mirror;
        (m.clusters.len(), m.events.len()).hash(&mut h);
        h.finish()
    }

    /// Every active chunk not in the GPU tables, from the mirror (`world.rs chunk_box`),
    /// and their hull triangles tagged with the host chunk's index.
    pub fn host_chunks(&self) -> (Vec<HostChunk>, Vec<HullVertex>) {
        let m = &self.world.solver.mirror;
        let on_gpu: std::collections::HashSet<(usize, usize)> =
            if host_chunks_only() { Default::default() } else { self.world.solver.chunk_order().iter().copied().collect() };
        let mut out = Vec::new();
        let mut hulls = Vec::new();
        for (s, st) in m.structures.iter().enumerate() {
            if (0..st.chunks.len()).all(|c| on_gpu.contains(&(s, c))) {
                continue;
            }
            // Largest utilization of each chunk's unbroken bonds (-1: none).
            let mut util = vec![-1.0f64; st.chunks.len()];
            for b in &m.bonds[s] {
                if b.joint.connected(b.rebar.is_some()) {
                    for c in [b.geometry.a, b.geometry.b] {
                        util[c] = util[c].max(b.joint.utilization);
                    }
                }
            }
            for c in 0..st.chunks.len() {
                let state = &m.chunks[s][c];
                if on_gpu.contains(&(s, c)) || !state.active || state.removed {
                    continue;
                }
                let ch = &st.chunks[c];
                let cl = &m.clusters[state.cluster];
                let hidden = Quat::from_axis_angle(state.th, state.th.norm()).to_mat3();
                let r = cl.rotation() * hidden * ch.rotation;
                let p = m.chunk_position(s, c);
                let h = ch.half_extents;
                let half = [h.x, h.y, h.z];
                push_hull(ch, out.len(), &mut hulls);
                out.push(HostChunk {
                    center: [p.x as f32, p.y as f32, p.z as f32, util[c] as f32],
                    axes: std::array::from_fn(|k| [r.m[0][k] as f32, r.m[1][k] as f32, r.m[2][k] as f32, half[k] as f32]),
                });
            }
        }
        (out, hulls)
    }
}

/// Entry distance of a ray into an axis-aligned box (negative: the origin is inside).
pub fn ray_box(o: Vec3, d: Vec3, lo: Vec3, hi: Vec3) -> Option<f64> {
    let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
    for (o, d, lo, hi) in [(o.x, d.x, lo.x, hi.x), (o.y, d.y, lo.y, hi.y), (o.z, d.z, lo.z, hi.z)] {
        if d.abs() < 1e-12 {
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let (a, b) = ((lo - o) / d, (hi - o) / d);
        t0 = t0.max(a.min(b));
        t1 = t1.min(a.max(b));
    }
    (t1 >= t0.max(0.0)).then_some(t0)
}
