//! Driven vehicles (a pack's `defaults.drives`, `stress_ref::drive`) in rubble's world API:
//! the pack's structure as a `Threshold` destructible (the threshold profile's GPU fracture,
//! inside `World`) and each drive as a raycast vehicle from the `engine-vehicle` presets,
//! its wheels driving it from its delay on and its chassis a projectile, so strike
//! prediction breaks the shop front before the step that meets it. One `World` on the
//! page's device, drawn by the lab's renderer: the chunks from the GPU (`fracture` view
//! kernels), the vehicles' boxes and wheels where the frame's readback put them.

use rubble::{
    BodyDef, DestructibleAssetDesc, DestructibleDef, DestructibleId, DestructionEvent, Geometry, GpuContext, Projectile, ShapeDef, StrengthModel, Transform, VehicleBuild, VehicleDef, VehicleInput, VehicleModel, World, WorldDef,
};
use engine_schema::ShapeType;
use stress_gpu::gpu::Gpu;
use stress_ref::drive::Drive;
use stress_ref::math::{Mat3, Vec3};
use stress_ref::solver::ReferenceSolver;
use stress_rubble::render::{material_color, material_surface, ExtraMesh, Renderer};

/// A driven vehicle: its drive, what the world built, and its drawn parts (in the
/// renderer's extras, chassis colliders first, then its wheels).
struct Car {
    drive: Drive,
    build: VehicleBuild,
    /// Its chassis's box colliders, each posed in the chassis body's frame.
    boxes: Vec<Transform>,
    wheels: usize,
}

pub(crate) struct DLoaded {
    world: World,
    destructible: DestructibleId,
    cars: Vec<Car>,
    pub renderer: Renderer,
    mirror: ReferenceSolver,
    extras: Vec<ExtraMesh>,
    pub behind: f64,
    pub totals: (usize, usize, usize),
    pub frame: u64,
    pub dt: f64,
    /// The extras' poses as of the last polled frame (stress frame).
    poses: Vec<(Mat3, Vec3)>,
    broken: usize,
    /// Each car's forward speed (km/h) as of the last polled frame.
    kmh: Vec<f64>,
    awake: u32,
    contacts: u32,
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

fn compose(a: &Transform, b: &Transform) -> Transform {
    let (qa, pa) = (glam::Quat::from_array(a.rotation), glam::Vec3::from(a.position));
    let (qb, pb) = (glam::Quat::from_array(b.rotation), glam::Vec3::from(b.position));
    Transform::new((pa + qa * pb).to_array(), (qa * qb).normalize().to_array())
}

/// The world for a pack with drives: its structure placed as the lab places it (the stress
/// frame is the asset frame less `placement`), its ground, and each drive's vehicle.
pub(crate) fn load(gpu: &Gpu, pack: &[u8], drives: &[Drive], width: u32, height: u32, format: wgpu::TextureFormat) -> Result<DLoaded, String> {
    fn e(what: &str) -> impl Fn(anyhow::Error) -> String + '_ {
        move |err| format!("{what}: {err:#}")
    }
    let asset = DestructibleAssetDesc::from_pack(pack).map_err(e("scene pack"))?;
    let stress = asset.stress.as_ref().ok_or("scene pack: no stress import")?;
    let solver = asset.solver_for(StrengthModel::Threshold).ok_or("scene pack: no threshold solver")?;
    let scene = stress.threshold_scene(1.0, solver.dt).map_err(e("threshold scene"))?;
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    // The lab's threshold capacity (`ThresholdWorld::new`), and room for the vehicles' bodies.
    let mut capacity = stress_rubble::capacity_for(&scene);
    capacity.max_bodies = capacity.max_bodies.max(((chunks + 64 + 16 * drives.len()) as u32).next_power_of_two());
    capacity.max_contacts = stress_rubble::threshold::contact_capacity(chunks, capacity.max_contacts, 1 << 30);
    capacity.max_candidates = capacity.max_candidates.max(capacity.max_contacts);
    let def = WorldDef {
        capacity,
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: false,
        contacts: rubble::ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let context = GpuContext::Shared { info: gpu.adapter_info.clone(), device: gpu.device.clone(), queue: gpu.queue.clone() };
    let mut world = World::new(context, &def).map_err(e("world"))?;
    // Placed as the lab places it (`engine-world` tests/threshold.rs): the world's frame is
    // then the pack's own, where the drives and the ground are given.
    let offset = stress.placement().map(|x| -x as f32);
    if let Some(h) = stress.ground_height() {
        let g = world.create_body(&BodyDef::default()).map_err(e("ground"))?;
        world.create_shape(g, &ShapeDef::default(), Geometry::Plane { normal: [0.0, 1.0, 0.0], offset: h }).map_err(e("ground"))?;
    }
    let a = world.create_destructible_asset(&asset).map_err(e("structure"))?;
    let destructible = world
        .create_destructible(&DestructibleDef { model: StrengthModel::Threshold, transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]), asleep: false, ..DestructibleDef::new(a) })
        .map_err(e("structure"))?;
    let mut cars = Vec::new();
    let mut extras = Vec::new();
    let body = ExtraMesh::Box { half: [0.0; 3], color: material_color("sheet-steel"), surface: material_surface("sheet-steel") };
    let tyre = (material_color("wheel"), material_surface("wheel"));
    for d in drives {
        let desc = engine_vehicle::presets::by_name(&d.vehicle).ok_or_else(|| format!("drive {}: no vehicle preset {}", d.name, d.vehicle))?;
        // The pack's frame is Y-up; the vehicle's nose is its design frame's +z.
        let (p, f) = (zup_inverse(d.at), zup_inverse(d.forward));
        let yaw = f[0].atan2(f[2]) as f32;
        let at = p.map(|x| x as f32);
        let wheels: Vec<(f64, f64)> = desc.wheels.iter().map(|w| (w.radius as f64, 0.5 * w.width as f64)).collect();
        let build = world
            .create_vehicle(&VehicleDef { model: VehicleModel::Rigid(desc), transform: Transform::new(at, glam::Quat::from_rotation_y(yaw).to_array()), parts: vec![], connections: vec![], user_data: 0 })
            .map_err(e(&format!("drive {}", d.name)))?;
        world.body_set_projectile(build.chassis, Some(Projectile::default())).map_err(e(&format!("drive {}", d.name)))?;
        let mut boxes = Vec::new();
        for (shape, pose, params) in world.body_colliders(build.chassis).map_err(e(&format!("drive {}", d.name)))? {
            if shape == ShapeType::Box {
                let ExtraMesh::Box { color, surface, .. } = body else { unreachable!() };
                extras.push(ExtraMesh::Box { half: [params[0] as f64, params[1] as f64, params[2] as f64], color, surface });
                boxes.push(pose);
            }
        }
        for &(radius, half_width) in &wheels {
            extras.push(ExtraMesh::Cylinder { radius, half_width, color: tyre.0, surface: tyre.1 });
        }
        cars.push(Car { drive: d.clone(), build, boxes, wheels: wheels.len() });
    }
    let mirror = stress_ref::world::World::new(&scene).solver;
    let totals = (
        mirror.structures.iter().map(|s| s.chunks.len()).sum::<usize>(),
        mirror.structures.iter().flat_map(|s| &s.chunks).map(|c| c.parts.as_ref().map_or(1, |p| p.len())).sum::<usize>(),
        mirror.structures.iter().map(|s| s.bonds.len()).sum::<usize>(),
    );
    let poses = vec![(Mat3::IDENTITY, Vec3::new(0.0, 0.0, -1e4)); extras.len()];
    let renderer = Renderer::for_target_with(gpu, &mirror, &[], &extras, width, height, format);
    let mut l = DLoaded { world, destructible, cars, renderer, mirror, extras, behind: 0.0, totals, frame: 0, dt: solver.dt as f64, poses, broken: 0, kmh: vec![0.0; drives.len()], awake: 0, contacts: 0 };
    l.attach_view()?;
    l.read_poses();
    Ok(l)
}

/// The stress frame's Z-up to the pack's Y-up.
fn zup_inverse(v: [f64; 3]) -> [f64; 3] {
    [v[0], v[2], -v[1]]
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
        self.world.destructible_set_view(self.destructible, &view).map_err(|e| format!("view: {e:#}"))
    }

    /// A renderer for a resized canvas.
    pub fn resize(&mut self, gpu: &Gpu, width: u32, height: u32, format: wgpu::TextureFormat) -> Result<(), String> {
        self.renderer = Renderer::for_target_with(gpu, &self.mirror, &[], &self.extras, width, height, format);
        self.attach_view()
    }

    /// One physics frame: each vehicle's input (its throttle from its delay on, braked
    /// before), then the world's step.
    pub fn step(&mut self) -> Result<(), String> {
        let t = self.frame as f64 * self.dt;
        for car in &self.cars {
            let Some(h) = car.build.vehicle else { continue };
            let input = if t >= car.drive.delay { VehicleInput { forward: car.drive.throttle as f32, ..VehicleInput::default() } } else { VehicleInput { brake: 1.0, ..VehicleInput::default() } };
            self.world.set_vehicle_input(h, input);
        }
        self.world.step().map_err(|e| format!("step: {e:#}"))?;
        self.frame += 1;
        Ok(())
    }

    /// Wait for the steps' readbacks: the bonds they broke, the counters, and the
    /// vehicles' poses for the next draw.
    pub fn poll(&mut self) -> Result<(), String> {
        let frames = self.world.poll_blocking().map_err(|e| format!("poll: {e:#}"))?;
        let mut wheels = std::collections::HashMap::new();
        for f in &frames {
            for e in &f.destruction_events {
                if let DestructionEvent::BondsBroken { bonds, .. } = e {
                    self.broken += bonds.len();
                }
            }
            if let Some(c) = &f.counters {
                self.awake = c.awake_count;
                self.contacts = c.contact_count;
                if c.overflow_flags != 0 {
                    return Err(format!("world: engine overflow {:#x} (1 contacts, 2 candidate pairs, 64 manifolds)", c.overflow_flags));
                }
            }
            for v in &f.vehicles {
                wheels.insert(v.vehicle, v.state.wheels.clone());
                if let Some(k) = self.cars.iter().position(|c| c.build.vehicle == Some(v.vehicle)) {
                    self.kmh[k] = 3.6 * v.state.forward_speed as f64;
                }
            }
        }
        self.read_poses_with(&wheels);
        Ok(())
    }

    fn read_poses(&mut self) {
        self.read_poses_with(&Default::default());
    }

    fn read_poses_with(&mut self, wheels: &std::collections::HashMap<rubble::VehicleHandle, Vec<rubble::WheelState>>) {
        let mut k = 0;
        for car in &self.cars {
            let chassis = self.world.body_transform(car.build.chassis).ok().flatten();
            for b in &car.boxes {
                if let Some(c) = &chassis {
                    let pose = compose(c, b);
                    self.poses[k] = (zup_rotation(pose.rotation), zup(pose.position));
                }
                k += 1;
            }
            let states = car.build.vehicle.and_then(|h| wheels.get(&h));
            for w in 0..car.wheels {
                if let Some(s) = states.and_then(|s| s.get(w)).filter(|s| !s.detached) {
                    self.poses[k] = (zup_rotation(s.rotation), zup(s.position));
                }
                k += 1;
            }
        }
    }

    /// Record the chunks' objects (from the GPU), the vehicles' poses and the draw.
    pub fn encode_view(&mut self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, mode: u32, bonds: bool) -> Result<u32, String> {
        let (objects, lines, _, _) = self.renderer.resident_targets();
        let capacity = self.renderer.line_capacity();
        let n = self.world.destructible_encode_view(self.destructible, encoder, objects, lines, capacity, mode, bonds).map_err(|e| format!("view: {e:#}"))?;
        self.renderer.set_extras(gpu, &self.poses);
        Ok(n)
    }

    /// The page's stats (the keys the other stress modes report).
    pub fn stats(&self, step_ms: f64, stepped: u32) -> serde_json::Value {
        let s = self.world.destructible_stats(self.destructible).ok();
        serde_json::json!({
            "gpuResident": true,
            "worldApi": true,
            "simSeconds": self.frame as f64 * self.dt,
            "stepMs": if stepped > 0 { step_ms / stepped as f64 } else { 0.0 },
            "chunks": self.totals.0,
            "hulls": self.totals.1,
            "bonds": self.totals.2,
            "shots": 0,
            "warnings": Vec::<String>::new(),
            "canFire": false,
            "charges": Vec::<String>::new(),
            "chargesArmed": 0,
            "launches": Vec::<String>::new(),
            "launchesArmed": 0,
            "drives": self.cars.iter().map(|c| c.drive.name.clone()).collect::<Vec<_>>(),
            "driveKmh": self.kmh,
            "frameHz": 1.0 / self.dt,
            "profile": "threshold",
            "frames": self.frame,
            "statsFrame": self.frame,
            "bodies": s.map_or(0, |s| s.actors),
            "fragments": s.map_or(0, |s| s.actors),
            "brokenBonds": self.broken,
            "awake": self.awake,
            "contacts": self.contacts,
            "errors": 0,
            "engineOverflow": 0,
            "compactions": 0,
        })
    }
}
