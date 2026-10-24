//! Driven vehicles (a pack's `defaults.drives`, `stress_ref::drive`) in rubble's world API:
//! the scenario host (`rubble::scenario`: the pack's structure as a `Threshold`
//! destructible, each drive a raycast vehicle from the `engine-vehicle` presets, shots as
//! projectile balls, charges cut between steps) on the page's device, drawn by the lab's
//! renderer: the chunks from the GPU (`fracture` view kernels), the vehicles' boxes and
//! wheels and the balls where the last polled step put them.

use rubble::scenario::Scenario;
use rubble::{GpuContext, StrengthModel, Transform};
use stress_gpu::gpu::Gpu;
use stress_ref::math::{Mat3, Vec3};
use stress_ref::scene::ImpactorDesc;
use stress_ref::solver::ReferenceSolver;
use stress_ref::wrecking::PackCharge;
use stress_rubble::render::{material_color, material_surface, ExtraMesh, Renderer};

pub(crate) struct DLoaded {
    pub scenario: Scenario,
    pub renderer: Renderer,
    mirror: ReferenceSolver,
    extras: Vec<ExtraMesh>,
    /// The extras' poses as of the last polled step (stress frame): each car's parts, then
    /// the balls (unit spheres from `ball_extras` on, parked until fired).
    poses: Vec<(Mat3, Vec3)>,
    ball_extras: usize,
    pub behind: f64,
    pub totals: (usize, usize, usize),
}

/// Y-up (the pack's and the world's frame) to the stress scene's Z-up.
fn zup(v: [f32; 3]) -> Vec3 {
    Vec3::new(v[0] as f64, -v[2] as f64, v[1] as f64)
}

/// A world rotation (Y-up) as the map from its body's frame into the stress frame.
fn zup_rotation(q: [f32; 4]) -> Mat3 {
    let r = glam::Mat3::from_quat(glam::Quat::from_array(q).normalize());
    // Rows of M * R, M = (x, y, z) -> (x, -z, y).
    let row = |k: usize| glam::Vec3::new(r.x_axis[k], r.y_axis[k], r.z_axis[k]);
    let (r0, r1, r2) = (row(0), -row(2), row(1));
    let mut m = Mat3::IDENTITY;
    for (i, r) in [r0, r1, r2].iter().enumerate() {
        m.m[i] = [r.x as f64, r.y as f64, r.z as f64];
    }
    m
}

fn pose(t: &Transform) -> (Mat3, Vec3) {
    (zup_rotation(t.rotation), zup(t.position))
}

/// The balls drawn at most (the page's shots and the scenario's).
fn balls() -> usize {
    crate::SHOTS.min(Scenario::MAX_BALLS)
}

/// The scenario for a pack with drives on the page's device, and its renderer.
pub(crate) fn load(gpu: &Gpu, pack: &[u8], width: u32, height: u32, format: wgpu::TextureFormat) -> Result<DLoaded, String> {
    let context = GpuContext::Shared { info: gpu.adapter_info.clone(), device: gpu.device.clone(), queue: gpu.queue.clone() };
    let scenario = Scenario::load(context, pack).map_err(|e| format!("{e:#}"))?;
    // The renderer's mirror of the structure (its chunks' meshes and the view tables).
    let asset = rubble::DestructibleAssetDesc::from_pack(pack).map_err(|e| format!("scene pack: {e:#}"))?;
    let stress = asset.stress.as_ref().ok_or("scene pack: no stress import")?;
    let solver = asset.solver_for(StrengthModel::Threshold).ok_or("scene pack: no threshold solver")?;
    let scene = stress.threshold_scene(1.0, solver.dt).map_err(|e| format!("threshold scene: {e:#}"))?;
    let mirror = stress_ref::world::World::new(&scene).solver;
    let mut extras = Vec::new();
    for car in &scenario.cars {
        for (_, half) in &car.boxes {
            extras.push(ExtraMesh::Box { half: half.map(|h| h as f64), color: material_color("sheet-steel"), surface: material_surface("sheet-steel") });
        }
        for &(radius, half_width, _) in &car.wheels {
            extras.push(ExtraMesh::Cylinder { radius: radius as f64, half_width: half_width as f64, color: material_color("wheel"), surface: material_surface("wheel") });
        }
    }
    // The shots' balls: unit spheres, posed scaled to each shot's radius.
    let ball_extras = extras.len();
    let shot = stress_rubble::render::SHOT;
    for _ in 0..balls() {
        extras.push(ExtraMesh::Sphere { radius: 1.0, color: [shot[0], shot[1], shot[2]], surface: shot[3] });
    }
    let totals = (
        mirror.structures.iter().map(|s| s.chunks.len()).sum::<usize>(),
        mirror.structures.iter().flat_map(|s| &s.chunks).map(|c| c.parts.as_ref().map_or(1, |p| p.len())).sum::<usize>(),
        mirror.structures.iter().map(|s| s.bonds.len()).sum::<usize>(),
    );
    let poses = vec![(Mat3::IDENTITY, Vec3::new(0.0, 0.0, -1e4)); extras.len()];
    let renderer = Renderer::for_target_with(gpu, &mirror, &[], &extras, width, height, format);
    let mut l = DLoaded { scenario, renderer, mirror, extras, poses, ball_extras, behind: 0.0, totals };
    l.attach_view()?;
    l.read_poses();
    Ok(l)
}

impl DLoaded {
    /// The renderer's chunk tables (as `ThresholdWorld::attach_view`), chunks numbered
    /// structure by structure as the world's threshold destructible numbers them.
    fn attach_view(&mut self) -> Result<(), String> {
        let (_, _, chunk_object, impactor_base) = self.renderer.resident_targets();
        let mut chunks = Vec::new();
        for (s, st) in self.mirror.structures.iter().enumerate() {
            for (c, ch) in st.chunks.iter().enumerate() {
                let q = stress_rubble::threshold::rotation_quat(&ch.rotation);
                let col = material_color(&ch.material);
                chunks.push([
                    [ch.center.x as f32, ch.center.y as f32, ch.center.z as f32, f32::from_bits(chunk_object[s][c])],
                    [q.x as f32, q.y as f32, q.z as f32, q.w as f32],
                    [col[0], col[1], col[2], material_surface(&ch.material)],
                ]);
            }
        }
        let view = stress_gpu::fracture::ViewTables { chunks, shot_bodies: Vec::new(), impactor_base };
        let s = &mut self.scenario;
        s.world.destructible_set_view(s.destructible, &view).map_err(|e| format!("view: {e:#}"))
    }

    /// A renderer for a resized canvas.
    pub fn resize(&mut self, gpu: &Gpu, width: u32, height: u32, format: wgpu::TextureFormat) -> Result<(), String> {
        self.renderer = Renderer::for_target_with(gpu, &self.mirror, &[], &self.extras, width, height, format);
        self.attach_view()
    }

    pub fn frame(&self) -> u64 {
        self.scenario.frame
    }

    pub fn dt(&self) -> f64 {
        self.scenario.dt
    }

    pub fn step(&mut self) -> Result<(), String> {
        self.scenario.step().map_err(|e| format!("step: {e:#}"))
    }

    pub fn shoot(&mut self, desc: &ImpactorDesc) -> Result<(), String> {
        self.scenario.shoot(desc).map_err(|e| format!("{e:#}"))
    }

    pub fn cut(&mut self, charge: &PackCharge) -> Result<usize, String> {
        self.scenario.cut(charge).map_err(|e| format!("{e:#}"))
    }

    /// Wait for the steps' readbacks, then pose the vehicles and balls for the next draw.
    pub fn poll(&mut self) -> Result<(), String> {
        self.scenario.poll().map_err(|e| format!("poll: {e:#}"))?;
        self.read_poses();
        Ok(())
    }

    fn read_poses(&mut self) {
        let mut k = 0;
        for parts in self.scenario.car_poses() {
            for t in &parts {
                self.poses[k] = pose(t);
                k += 1;
            }
        }
        for (i, (t, r)) in self.scenario.ball_poses().into_iter().enumerate().take(balls()) {
            let (mut m, p) = pose(&t);
            for row in &mut m.m {
                *row = row.map(|x| x * r as f64);
            }
            self.poses[self.ball_extras + i] = (m, p);
        }
    }

    /// Record the chunks' objects (from the GPU), the vehicles' poses and the draw.
    pub fn encode_view(&mut self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, mode: u32, bonds: bool) -> Result<u32, String> {
        let (objects, lines, _, _) = self.renderer.resident_targets();
        let capacity = self.renderer.line_capacity();
        let s = &mut self.scenario;
        let n = s.world.destructible_encode_view(s.destructible, encoder, objects, lines, capacity, mode, bonds).map_err(|e| format!("view: {e:#}"))?;
        self.renderer.set_extras(gpu, &self.poses);
        Ok(n)
    }

    /// The page's stats (the keys the other stress modes report).
    pub fn stats(&self, step_ms: f64, stepped: u32) -> serde_json::Value {
        let s = &self.scenario;
        let d = s.world.destructible_stats(s.destructible).ok();
        serde_json::json!({
            "gpuResident": true,
            "worldApi": true,
            "simSeconds": s.frame as f64 * s.dt,
            "stepMs": if stepped > 0 { step_ms / stepped as f64 } else { 0.0 },
            "chunks": self.totals.0,
            "hulls": self.totals.1,
            "bonds": self.totals.2,
            "shots": s.balls.len(),
            "warnings": Vec::<String>::new(),
            "canFire": s.balls.len() < balls(),
            "launches": Vec::<String>::new(),
            "launchesArmed": 0,
            "drives": s.cars.iter().map(|c| c.drive.name.clone()).collect::<Vec<_>>(),
            "driveKmh": s.speed_kmh(),
            "frameHz": 1.0 / s.dt,
            "profile": "threshold",
            "frames": s.frame,
            "statsFrame": s.frame,
            "bodies": d.map_or(0, |d| d.actors),
            "fragments": d.map_or(0, |d| d.actors),
            "brokenBonds": s.broken,
            "awake": s.awake,
            "contacts": s.contacts,
            "errors": 0,
            "engineOverflow": 0,
            "compactions": 0,
        })
    }
}
