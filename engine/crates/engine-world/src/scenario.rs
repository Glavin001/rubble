//! Scenarios: a scene pack with driven vehicles (`defaults.drives`, stress-ref `drive`) as
//! one world, the host every page and test of them shares. The pack's structure is a
//! `Threshold` destructible placed as the fracture lab places it, so the world's frame is
//! the pack's own (Y-up); its ground is the lab's; each drive is a raycast vehicle from the
//! `engine-vehicle` presets, its wheels driving it from its delay on and its chassis a
//! projectile, so strike prediction breaks a wall before the step that meets it.
//!
//! The lab's other inputs: a shot is a projectile sphere of the ball's mass and radius
//! ([`Scenario::shoot`]), and a pack's charge is cut between two steps
//! ([`Scenario::cut`], `World::destructible_cut_bonds`).
//!
//! Drawing is the host's. A chunk is drawn posed at rest and moved by
//! [`Scenario::chunk_motion`] (the render contract: its piece's transform); the vehicles'
//! parts by [`Scenario::car_poses`], the balls by [`Scenario::ball_poses`].

use crate::{
    BodyDef, BodyId, DestructibleAssetDesc, DestructibleDef, DestructibleId, DestructionEvent, Geometry, GpuContext, MassData, Projectile, ShapeDef, StrengthModel, Transform, VehicleBuild, VehicleDef,
    VehicleHandle, VehicleInput, VehicleModel, World, WorldDef,
};
use engine_schema::ShapeType;
use stress_ref::drive::Drive;
use stress_ref::scene::{ImpactorDesc, ImpactorShape};
use stress_ref::wrecking::PackCharge;

/// A driven vehicle and its drawn parts, posed in its chassis body's frame.
pub struct ScenarioCar {
    pub drive: Drive,
    pub build: VehicleBuild,
    /// Its chassis's box colliders: pose and half extents.
    pub boxes: Vec<(Transform, [f32; 3])>,
    /// Its wheels: radius, half width, and its centre at full droop (where it spawns).
    pub wheels: Vec<(f32, f32, [f32; 3])>,
}

/// A pack with drives as one world (module doc).
pub struct Scenario {
    pub world: World,
    pub destructible: DestructibleId,
    /// The destructible's transform: every piece's at rest (the render contract).
    pub instance: Transform,
    pub cars: Vec<ScenarioCar>,
    /// The pack's charges (`defaults.charges`), in firing order.
    pub charges: Vec<PackCharge>,
    /// The shots fired: their bodies and radii.
    pub balls: Vec<(BodyId, f32)>,
    /// Steps taken, and their length (s).
    pub frame: u64,
    pub dt: f64,
    /// Bonds broken, awake bodies and contacts, as of the last polled step.
    pub broken: usize,
    pub awake: u32,
    pub contacts: u32,
    /// Each car's forward speed (m/s) as of the last polled step: its cruise control's.
    speed: Vec<f64>,
    /// The step last polled (`frame` then).
    polled: u64,
    /// Each car's chassis and wheels as of the last polled step (until then, as made).
    chassis: Vec<Transform>,
    wheels: Vec<Vec<Transform>>,
}

fn y_up(v: [f64; 3]) -> [f32; 3] {
    [v[0] as f32, v[2] as f32, -v[1] as f32]
}

fn compose(a: &Transform, b: &Transform) -> Transform {
    let (qa, pa) = (glam::Quat::from_array(a.rotation), glam::Vec3::from(a.position));
    let (qb, pb) = (glam::Quat::from_array(b.rotation), glam::Vec3::from(b.position));
    Transform::new((pa + qa * pb).to_array(), (qa * qb).normalize().to_array())
}

impl Scenario {
    /// The shots a scenario takes at most.
    pub const MAX_BALLS: usize = 16;

    /// The world for a pack (its JSON bytes) with drives, on `context`'s device. Waits for
    /// the GPU once (the vehicles' poses as made).
    pub fn load(context: GpuContext, pack: &[u8]) -> anyhow::Result<Scenario> {
        use anyhow::Context;
        let json: serde_json::Value = serde_json::from_slice(pack).context("scene pack is not JSON")?;
        let drives = stress_ref::drive::drives_of(&json).map_err(|e| anyhow::anyhow!("scene pack: {e}"))?;
        let charges = stress_ref::wrecking::charges_of(&json);
        let asset = DestructibleAssetDesc::from_pack(pack).context("scene pack")?;
        let stress = asset.stress.as_ref().context("scene pack: no stress import")?;
        let solver = asset.solver_for(StrengthModel::Threshold).context("scene pack: no threshold solver")?;
        let scene = stress.threshold_scene(1.0, solver.dt).context("threshold scene")?;
        let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
        // The lab's threshold capacity (`ThresholdWorld::new`), and room for the vehicles'
        // bodies (chassis, wheels, parts) and the balls.
        let mut capacity = stress_rubble::capacity_for(&scene);
        capacity.max_bodies = capacity.max_bodies.max(((chunks + 64 + 16 * drives.len() + Self::MAX_BALLS) as u32).next_power_of_two());
        capacity.max_contacts = stress_rubble::threshold::contact_capacity(chunks, capacity.max_contacts, 1 << 30);
        capacity.max_candidates = capacity.max_candidates.max(capacity.max_contacts);
        let dt = solver.dt as f64;
        let def = WorldDef { capacity, gravity: [0.0, solver.gravity_y, 0.0], solver, enable_sleep: false, contacts: crate::ContactStorage::Canonical, ..WorldDef::default() };
        let mut world = World::new(context, &def).context("world")?;
        // Placed as the lab places it (tests/threshold.rs): the world's frame is then the
        // pack's own, where the drives and the ground are given.
        let offset = stress.placement().map(|x| -x as f32);
        if let Some(h) = stress.ground_height() {
            // (Its height is the asset frame's; the structure stands at -placement.)
            let g = world.create_body(&BodyDef::default()).context("ground")?;
            world.create_shape(g, &ShapeDef::default(), Geometry::Plane { normal: [0.0, 1.0, 0.0], offset: h + offset[1] }).context("ground")?;
        }
        let a = world.create_destructible_asset(&asset).context("structure")?;
        let instance = Transform::new(offset, [0.0, 0.0, 0.0, 1.0]);
        let destructible = world.create_destructible(&DestructibleDef { model: StrengthModel::Threshold, transform: instance, asleep: false, ..DestructibleDef::new(a) }).context("structure")?;
        let mut cars = Vec::with_capacity(drives.len());
        for d in &drives {
            let what = || format!("drive {}", d.name);
            let desc = engine_vehicle::presets::by_name(&d.vehicle).with_context(|| format!("drive {}: no vehicle preset {}", d.name, d.vehicle))?;
            // The vehicle's nose is its design frame's +z; it stands on its wheels at full
            // droop over the ground point the drive gives.
            let f = y_up(d.forward);
            let yaw = f[0].atan2(f[2]);
            let mut at = y_up(d.at);
            at[1] += desc.droop_height();
            // (The chassis body's frame is the centred design frame.)
            let centred = desc.centered();
            let down = -glam::Vec3::from_array(centred.up).normalize();
            let wheels = centred
                .wheels
                .iter()
                .map(|w| {
                    let dir = w.suspension_dir.map_or(down, |s| glam::Vec3::from_array(s).normalize());
                    (w.radius, 0.5 * w.width, (glam::Vec3::from_array(w.position) + dir * w.suspension_max).to_array())
                })
                .collect();
            let build = world
                .create_vehicle(&VehicleDef { model: VehicleModel::Rigid(desc), transform: Transform::new(at, glam::Quat::from_rotation_y(yaw).to_array()), parts: vec![], connections: vec![], user_data: 0 })
                .with_context(what)?;
            world.body_set_projectile(build.chassis, Some(Projectile::default())).with_context(what)?;
            let boxes = world.body_colliders(build.chassis).with_context(what)?.into_iter().filter(|c| c.0 == ShapeType::Box).map(|(_, pose, p)| (pose, [p[0], p[1], p[2]])).collect();
            cars.push(ScenarioCar { drive: d.clone(), build, boxes, wheels });
        }
        let n = cars.len();
        let mut s = Scenario {
            world,
            destructible,
            instance,
            cars,
            charges,
            balls: Vec::new(),
            frame: 0,
            dt,
            broken: 0,
            awake: 0,
            contacts: 0,
            speed: vec![0.0; n],
            polled: 0,
            chassis: Vec::with_capacity(n),
            wheels: Vec::with_capacity(n),
        };
        // As made: each chassis read back, its wheels at full droop.
        for car in &s.cars {
            let t = s.world.read_body_state(car.build.chassis)?.transform;
            s.wheels.push(car.wheels.iter().map(|w| compose(&t, &Transform::new(w.2, [0.0, 0.0, 0.0, 1.0]))).collect());
            s.chassis.push(t);
        }
        Ok(s)
    }

    /// One step: each vehicle's input (its throttle from its delay on, up to its cruising
    /// speed as of the step before; braked before), then the world's step. A cruise
    /// control reads the last step's speed whenever the host polls (it waits for it), so
    /// a run is the same physics however its steps are batched (live play, a take, a
    /// replay's catch-up).
    pub fn step(&mut self) -> anyhow::Result<()> {
        if self.polled != self.frame && self.cars.iter().any(|c| c.drive.speed.is_some()) {
            self.poll()?;
        }
        let t = self.frame as f64 * self.dt;
        for (car, &v) in self.cars.iter().zip(&self.speed) {
            let Some(h) = car.build.vehicle else { continue };
            let input = match car.drive.throttle_at(t, v) {
                Some(throttle) => VehicleInput { forward: throttle as f32, ..VehicleInput::default() },
                None => VehicleInput { brake: 1.0, ..VehicleInput::default() },
            };
            self.world.set_vehicle_input(h, input);
        }
        self.world.step()?;
        self.frame += 1;
        Ok(())
    }

    /// Wait for the steps' readbacks: the bonds they broke, the counters, and the
    /// vehicles' speeds and poses. An engine overflow is an error.
    pub fn poll(&mut self) -> anyhow::Result<()> {
        let frames = self.world.poll_blocking()?;
        self.polled = self.frame;
        let mut wheels: std::collections::HashMap<VehicleHandle, Vec<crate::WheelState>> = Default::default();
        for f in &frames {
            for e in &f.destruction_events {
                if let DestructionEvent::BondsBroken { bonds, .. } = e {
                    self.broken += bonds.len();
                }
            }
            if let Some(c) = &f.counters {
                self.awake = c.awake_count;
                self.contacts = c.contact_count;
                anyhow::ensure!(c.overflow_flags == 0, "world: engine overflow {:#x} (1 contacts, 2 candidate pairs, 64 manifolds)", c.overflow_flags);
            }
            for v in &f.vehicles {
                wheels.insert(v.vehicle, v.state.wheels.clone());
                if let Some(k) = self.cars.iter().position(|c| c.build.vehicle == Some(v.vehicle)) {
                    self.speed[k] = v.state.forward_speed as f64;
                }
            }
        }
        for (k, car) in self.cars.iter().enumerate() {
            if let Ok(Some(t)) = self.world.body_transform(car.build.chassis) {
                self.chassis[k] = t;
            }
            if let Some(states) = car.build.vehicle.and_then(|h| wheels.get(&h)) {
                for (w, s) in states.iter().enumerate().take(self.wheels[k].len()) {
                    if !s.detached {
                        self.wheels[k][w] = Transform::new(s.position, s.rotation);
                    }
                }
            }
        }
        Ok(())
    }

    /// Fire a shot (the lab's, in its stress frame, Z-up): a dynamic sphere of its radius
    /// and mass (a solid ball's inertia) at its position, with its velocity and spin, a
    /// projectile, so its contacts and strike prediction break the structure as the lab's
    /// shots do. Between steps. An error past [`Self::MAX_BALLS`] shots.
    pub fn shoot(&mut self, desc: &ImpactorDesc) -> anyhow::Result<()> {
        anyhow::ensure!(self.balls.len() < Self::MAX_BALLS, "{} shots fired: a scenario takes at most {}", self.balls.len(), Self::MAX_BALLS);
        let ImpactorShape::Sphere { radius } = desc.shape else {
            anyhow::bail!("shot {}: only spheres", desc.name);
        };
        let b = self.world.create_body(&BodyDef { linear_velocity: y_up(desc.velocity), angular_velocity: y_up(desc.angular_velocity), ..BodyDef::dynamic(y_up(desc.position)) })?;
        self.world.create_sphere_shape(b, &ShapeDef { update_body_mass: false, ..ShapeDef::default() }, [0.0; 3], radius as f32)?;
        let i = (0.4 * desc.mass * radius * radius) as f32;
        self.world.body_set_mass_data(b, MassData { mass: desc.mass as f32, center: [0.0; 3], rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]] })?;
        self.world.body_set_projectile(b, Some(Projectile::default()))?;
        self.balls.push((b, radius as f32));
        Ok(())
    }

    /// Set off a charge: its bonds cut between this step and the next. Its debris kick
    /// is not carried yet: an error for a charge with one.
    pub fn cut(&mut self, charge: &PackCharge) -> anyhow::Result<usize> {
        anyhow::ensure!(charge.debris.is_empty() || charge.kick == [0.0; 3], "charge {}: debris kicks are not carried by a scenario yet", charge.name);
        self.world.destructible_cut_bonds(self.destructible, &charge.bonds).map_err(|e| anyhow::anyhow!("charge {}: {e:#}", charge.name))
    }

    /// Each car's forward speed (km/h) as of the last polled step.
    pub fn speed_kmh(&self) -> Vec<f64> {
        self.speed.iter().map(|v| 3.6 * v).collect()
    }

    /// Each car's parts as of the last polled step (world): its boxes, then its wheels
    /// (a wheel's axle is its local x).
    pub fn car_poses(&self) -> Vec<Vec<Transform>> {
        self.cars
            .iter()
            .enumerate()
            .map(|(k, car)| car.boxes.iter().map(|(pose, _)| compose(&self.chassis[k], pose)).chain(self.wheels[k].iter().copied()).collect())
            .collect()
    }

    /// Each ball as of the last polled step, and its radius.
    pub fn ball_poses(&self) -> Vec<(Transform, f32)> {
        self.balls.iter().filter_map(|&(b, r)| Some((self.world.body_transform(b).ok()??, r))).collect()
    }

    /// Chunk `chunk`'s motion since rest as of the last polled step: its piece's transform
    /// less the instance's, `T T0^-1` (a chunk drawn at its rest pose is drawn where it is
    /// by this). Chunks number as the pack's nodes.
    pub fn chunk_motion(&self, chunk: u32) -> anyhow::Result<Transform> {
        let body = self.world.destructible_chunk_body(self.destructible, chunk)?;
        let Some(t) = self.world.body_transform(body)? else {
            return Ok(Transform::IDENTITY);
        };
        let (q0, p0) = (glam::Quat::from_array(self.instance.rotation), glam::Vec3::from(self.instance.position));
        let inverse = Transform::new((-(q0.inverse() * p0)).to_array(), q0.inverse().to_array());
        Ok(compose(&t, &inverse))
    }
}
