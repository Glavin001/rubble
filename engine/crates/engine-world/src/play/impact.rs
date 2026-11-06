//! The meteor impact (`impact.html`) as a [`super::SceneDef`]: an impact or a buried
//! charge on a gently rolling sand plain (deformable terrain, π-scaling), its world, its
//! placement and its script: launched, it strikes after the approach; the crater bakes
//! when settled (or at a set time, or on demand); then a monster truck drives out of it.
//! The page's own numbers and order throughout (its switches as `Number` reads them).

use super::director::{note, note_with, Action, Event};
use crate::*;
use serde::{Deserialize, Serialize};

/// Gravity (m/s²).
pub const G: f64 = 9.81;
/// The flight shown before contact (s): the strike comes this long after the launch.
pub const APPROACH: f64 = 0.12;

/// `impact.html`'s switches.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct ImpactSceneDef {
    /// `impact` or `explosion` (`?mode=`).
    pub mode: String,
    /// The impactor: radius (m), speed (km/s), density (kg/m³).
    pub radius: f64,
    pub speed: f64,
    pub density: f64,
    /// The charge: 10^charge kg TNT, `depth` m down.
    pub charge: f64,
    pub depth: f64,
    /// The region's margin past the predicted crater (× its radius).
    pub margin: f64,
    /// `game` or `physical` (`engine_terrain::TerrainProfile`).
    pub profile: String,
    /// Physics frames per second.
    pub hz: f64,
    /// Cells across the transient radius (none: the profile's).
    pub cpr: Option<f64>,
    /// The bake: `settle` (when the crater rests) or `time` (at `bake_at` s).
    pub bake: String,
    pub bake_at: f64,
    /// The rest speed the settle bake waits for (m/s; none: 1% of √(gR)).
    pub settle: Option<f64>,
}

impl Default for ImpactSceneDef {
    fn default() -> Self {
        ImpactSceneDef {
            mode: "impact".into(),
            radius: 8.0,
            speed: 20.0,
            density: 3000.0,
            charge: 6.0,
            depth: 8.0,
            margin: 1.5,
            profile: "game".into(),
            hz: 120.0,
            cpr: None,
            bake: "settle".into(),
            bake_at: 12.0,
            settle: None,
        }
    }
}

/// What the page reads of a placed impact scene.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Placed {
    /// The heightfield shape and its grid (cell `sp` m, corner `x0`).
    pub ground: u64,
    pub nx: u32,
    pub nz: u32,
    pub sp: f64,
    pub x0: f64,
    /// The impact point (on the plain).
    pub impact: [f32; 3],
    /// The prediction (π-scaling): the transient crater, the region's cell, the source.
    pub radius: f64,
    pub depth: f64,
    pub final_rim_radius: f64,
    pub formation_time: f64,
    pub dx: f64,
    pub source_speed: f64,
    pub energy: f64,
    pub mass: Option<f64>,
    pub settle_speed: f32,
}

impl ImpactSceneDef {
    pub(crate) fn from_query(q: &super::scene::Query) -> Self {
        let num = |k: &str, d: f64| match q.get(k) {
            Some(v) => {
                let n = super::scene::js_number(v);
                if n.is_finite() {
                    n
                } else {
                    d
                }
            }
            None => d,
        };
        let physical = q.get("profile") == Some("physical");
        ImpactSceneDef {
            mode: if q.get("mode") == Some("explosion") {
                "explosion"
            } else {
                "impact"
            }
            .into(),
            radius: num("radius", 8.0),
            speed: num("speed", 20.0),
            density: num("density", 3000.0),
            charge: num("charge", 6.0),
            depth: num("depth", 8.0),
            margin: num("margin", 1.5),
            profile: if physical { "physical" } else { "game" }.into(),
            hz: num("hz", 120.0),
            cpr: if q.get("cpr").is_some() {
                Some(num("cpr", 12.0))
            } else if physical {
                Some(if num("margin", 1.5) >= 1.5 {
                    12.0
                } else {
                    16.0
                })
            } else {
                None
            },
            bake: if q.get("bake") == Some("time") {
                "time"
            } else {
                "settle"
            }
            .into(),
            bake_at: num("bakeAt", 12.0),
            settle: q.get("settle").map(|_| num("settle", 0.0)),
        }
    }

    fn profile(&self) -> TerrainProfile {
        if self.profile == "physical" {
            TerrainProfile::Physical
        } else {
            TerrainProfile::Game
        }
    }

    /// The terrain's settings: dry sand at physical stiffness, the browser's limits; the
    /// rest speed set from the crater (`settle_speed`).
    pub fn terrain_def(&self, settle_speed: f32) -> TerrainDef {
        let mut config = TerrainConfig::profile(self.profile());
        config.max_particles = 3 << 19;
        config.max_blocks = 8192;
        config.max_parcels = 1 << 18;
        config.max_substeps = 1024;
        config.max_columns = 1 << 17;
        config.max_field_samples = 1 << 19;
        config.max_coupled = 64;
        if let Some(c) = self.cpr {
            config.cells_per_radius = c as f32;
        }
        TerrainDef {
            config,
            // The page hands its margin over as an f32.
            region_margin: f64::from(self.margin as f32),
            settle_speed,
            settle_frames: 30,
            surface: Some(TerrainSurfaceDef {
                patch_samples: 0,
                max_debris: 0,
                ..TerrainSurfaceDef::default()
            }),
            ..TerrainDef::default()
        }
    }

    /// The crater π-scaling predicts, the region's cell, the source's peak speed, energy
    /// and (an impact's) mass.
    pub fn prediction(&self) -> anyhow::Result<(TerrainPrediction, f64, Option<f64>)> {
        let d = self.terrain_def(0.05);
        if self.mode == "impact" {
            let src = d.vertical_impact(self.radius, self.density, self.speed * 1000.0)?;
            Ok((
                d.predict_impact(&src, G)?,
                src.kinetic_energy(),
                Some(src.mass()),
            ))
        } else {
            let src = d.tnt_charge(10f64.powf(self.charge), self.depth)?;
            Ok((d.predict_explosion(&src, G)?, src.energy, None))
        }
    }
}

/// The plain: gently rolling, flat at the impact point (the scaling assumes level ground).
fn plain(x: f64, z: f64, r: f64) -> f64 {
    let rr = super::hills::js_hypot(&[x, z]) / r;
    let fade = ((rr - 0.4) / 1.2).max(0.0).min(1.0);
    let amp = 0.01 * r;
    fade * amp
        * ((x / (1.7 * r) + 0.4).sin() * (z / (2.1 * r)).cos() + 0.6 * ((x + z) / (0.9 * r)).sin())
}

/// The grid: cells of the region's `dx`, out to 6 transient radii.
fn grid(r: f64, sp: f64) -> (u32, f64) {
    let half = 6.0 * r;
    let nx = 2 * (half / sp).ceil() as u32 + 1;
    (nx, -0.5 * (nx as f64 - 1.0) * sp)
}

pub fn world_def(_def: &ImpactSceneDef) -> anyhow::Result<WorldDef> {
    Ok(WorldDef {
        gravity: [0.0, -G as f32, 0.0],
        contacts: ContactStorage::Canonical,
        capacity: WorldCapacity {
            max_bodies: 4096,
            max_colliders: 8192,
            max_contacts: 65536,
            max_joints: 64,
            max_candidates: 131072,
            max_colors: 32,
            max_commands: 2048,
            ..WorldCapacity::default()
        },
        vehicles: VehicleCapacity {
            max_vehicles: 4,
            max_wheels: 16,
            max_curve_points: 512,
            max_diffs: 8,
            max_anti_rolls: 8,
            max_parts: 16,
            max_rays: 16,
        },
        readback_frames: 4,
        ..WorldDef::default()
    })
}

pub fn place(world: &mut World, def: &ImpactSceneDef) -> anyhow::Result<Placed> {
    let (pred, energy, mass) = def.prediction()?;
    let r = pred.crater.radius;
    // The page reads the cell from JSON: the f32, widened.
    let sp = pred.dx as f64;
    let (nx, x0) = grid(r, sp);
    let nz = nx;
    let mut heights = Vec::with_capacity((nx * nz) as usize);
    for j in 0..nz {
        for i in 0..nx {
            heights.push(plain(x0 + i as f64 * sp, x0 + j as f64 * sp, r) as f32);
        }
    }
    let mut solver = world.solver();
    solver.dt = (1.0 / def.hz) as f32;
    solver.iterations = 8;
    solver.gravity_y = -G as f32;
    solver.flags = engine_schema::FLAG_SPEC_VELOCITY;
    world.set_solver(solver);
    // The material table: slot 0 (the ground and the vehicle's part colliders) and the
    // wheel rims at 0.9, the rest 0.6.
    for i in 0..64u32 {
        let friction = if i == 0 || i == engine_vehicle::pack::WHEEL_RIM_MATERIAL {
            0.9
        } else {
            0.6
        };
        world.set_material_slot(
            i,
            SurfaceMaterial {
                friction,
                ..SurfaceMaterial::default()
            },
        )?;
    }
    let ground_body = world.create_body(&BodyDef::default())?;
    let shape = ShapeDef {
        material: SurfaceMaterial {
            friction: 0.9,
            ..SurfaceMaterial::default()
        },
        ..ShapeDef::default()
    };
    let field = HeightField::new(nx, nz, sp as f32, sp as f32, heights.clone())?;
    let ground =
        world.create_height_field_shape(ground_body, &shape, &field, Transform::IDENTITY)?;
    // Rest at the crater's scale: 1% of its gravity speed √(gR).
    let settle_speed = def.settle.unwrap_or(0.01 * (G * r).sqrt()) as f32;
    world.enable_terrain(&def.terrain_def(settle_speed))?;
    let surface = surface_at(&heights, nx, nz, x0, sp, 0.0, 0.0);
    Ok(Placed {
        ground: ground.to_bits(),
        nx,
        nz,
        sp,
        x0,
        impact: [0.0, surface as f32, 0.0],
        radius: r,
        depth: pred.crater.depth,
        final_rim_radius: pred.crater.final_rim_radius,
        formation_time: pred.crater.formation_time,
        dx: pred.dx as f64,
        source_speed: pred.source_speed as f64,
        energy,
        mass,
        settle_speed,
    })
}

/// The ground's height under (x, z) on the grid's triangles, as the page interpolates.
fn surface_at(h: &[f32], nx: u32, nz: u32, x0: f64, sp: f64, x: f64, z: f64) -> f64 {
    let (fx, fz) = ((x - x0) / sp, (z - x0) / sp);
    if fx < 0.0 || fz < 0.0 || fx > (nx - 1) as f64 || fz > (nz - 1) as f64 {
        return 0.0;
    }
    let i = (fx.floor() as u32).min(nx - 2) as usize;
    let j = (fz.floor() as u32).min(nz - 2) as usize;
    let (u, v) = (fx - i as f64, fz - j as f64);
    let at = |a: usize, b: usize| h[(j + b) * nx as usize + i + a] as f64;
    if u >= v {
        at(0, 0) + u * (at(1, 0) - at(0, 0)) + v * (at(1, 1) - at(1, 0))
    } else {
        at(0, 0) + v * (at(0, 1) - at(0, 0)) + u * (at(1, 1) - at(0, 1))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    Ready,
    Approach,
    Open,
    Baked,
    Drive,
}

/// The impact's script: the strike after the approach, the timed bake, the truck.
pub struct ImpactDirector {
    def: ImpactSceneDef,
    placed: Placed,
    phase: Phase,
    /// Sim time since the strike (negative during the approach).
    t: f64,
    bake_t: Option<f64>,
    vehicle: Option<VehicleHandle>,
    chassis: Option<BodyId>,
    car_read: Option<BodyRead>,
    car: Option<([f64; 3], [f64; 3], f64)>,
    pub manual: bool,
}

impl ImpactDirector {
    pub(crate) fn new(def: &ImpactSceneDef, placed: Placed) -> Self {
        ImpactDirector {
            def: def.clone(),
            placed,
            phase: Phase::Ready,
            t: 0.0,
            bake_t: None,
            vehicle: None,
            chassis: None,
            car_read: None,
            car: None,
            manual: false,
        }
    }

    fn dt(&self) -> f64 {
        1.0 / self.def.hz
    }

    fn strike(&mut self, world: &mut World, ev: &mut Vec<Event>) -> anyhow::Result<()> {
        let ground = ShapeId::from_bits(self.placed.ground);
        let d = world
            .terrain_settings()
            .ok_or_else(|| anyhow::anyhow!("terrain is not enabled"))?
            .clone();
        let crater = if self.def.mode == "impact" {
            let src =
                d.vertical_impact(self.def.radius, self.def.density, self.def.speed * 1000.0)?;
            world.terrain_impact(ground, &src, self.placed.impact)?
        } else {
            let src = d.tnt_charge(10f64.powf(self.def.charge), self.def.depth)?;
            world.terrain_explode(ground, &src, self.placed.impact)?
        };
        self.phase = Phase::Open;
        self.t = 0.0;
        note_with(
            ev,
            world,
            "strike",
            serde_json::json!({ "radius": crater.radius }),
        );
        Ok(())
    }

    fn bake(&mut self, world: &mut World) -> anyhow::Result<()> {
        if self.phase == Phase::Open && !world.terrain_areas().is_empty() {
            world.terrain_close()?;
        }
        Ok(())
    }

    pub(crate) fn before_step(
        &mut self,
        world: &mut World,
        ev: &mut Vec<Event>,
    ) -> anyhow::Result<()> {
        if self.phase == Phase::Approach && self.t >= 0.0 {
            self.strike(world, ev)?;
        }
        // Timed bake: once nothing is in flight.
        if self.phase == Phase::Open && self.def.bake == "time" && self.t >= self.def.bake_at {
            if world.terrain_last_report().is_some_and(|r| r.airborne == 0) {
                self.bake(world)?;
            }
        }
        // The crater baked (on time, on settling or on demand): the truck a beat later.
        if self.phase == Phase::Open && world.terrain_areas().is_empty() {
            self.phase = Phase::Baked;
            self.bake_t = Some(self.t);
            note(ev, world, "baked");
        }
        if self.phase == Phase::Baked && self.bake_t.is_some_and(|b| self.t >= b + 1.5) {
            self.spawn(world, ev)?;
        }
        self.drive(world)?;
        if matches!(
            self.phase,
            Phase::Approach | Phase::Open | Phase::Baked | Phase::Drive
        ) {
            self.t += self.dt();
        }
        Ok(())
    }

    /// On the crater floor, facing out (−x): it drives up the inner wall toward the rim.
    fn spawn(&mut self, world: &mut World, ev: &mut Vec<Event>) -> anyhow::Result<()> {
        let p = &self.placed;
        let x = p.impact[0] as f64 - 0.08 * p.radius;
        let heights = match world.shape_geometry(ShapeId::from_bits(p.ground))? {
            Geometry::HeightField { field, .. } => field.heights().to_vec(),
            _ => anyhow::bail!("the ground is not a heightfield"),
        };
        let y = surface_at(&heights, p.nx, p.nz, p.x0, p.sp, x, 0.0) + 2.2;
        let desc = engine_vehicle::presets::all()
            .into_iter()
            .find(|d| d.name == "monster_truck")
            .ok_or_else(|| anyhow::anyhow!("no monster_truck preset"))?;
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let v = world.add_vehicle(
            &desc,
            [x as f32, y as f32, 0.0],
            [0.0, -h as f32, 0.0, h as f32],
        )?;
        let slot = world.sim().vehicle_host().chassis_body(v);
        self.vehicle = Some(v);
        self.chassis = world.body_at(slot);
        self.phase = Phase::Drive;
        note(ev, world, "drive");
        Ok(())
    }

    /// Autopilot: full throttle straight out along −x, held to 12 m/s.
    fn drive(&mut self, world: &mut World) -> anyhow::Result<()> {
        let Some(v) = self.vehicle else {
            return Ok(());
        };
        if let Some(r) = &self.car_read {
            if let Some(rec) = r.try_take(world)? {
                let b = rec[0];
                let q = b.rotation.map(|x| x as f64);
                let fwd = rot(q, [0.0, 0.0, 1.0]);
                let lv = b.linear_velocity.map(|x| x as f64);
                self.car = Some((
                    b.position.map(|x| x as f64),
                    fwd,
                    super::hills::js_hypot(&lv),
                ));
                self.car_read = None;
            }
        }
        if self.car_read.is_none() {
            if let Some(c) = self.chassis {
                self.car_read = Some(world.begin_body_read(&[c])?);
            }
        }
        if self.manual {
            return Ok(());
        }
        let (fast, steer) = match self.car {
            Some((_, fwd, speed)) => {
                let want = [-1.0, 0.0, 0.0];
                let side = fwd[0] * want[2] - fwd[2] * want[0];
                (speed > 12.0, (2.0 * side).clamp(-1.0, 1.0))
            }
            None => (false, 0.0),
        };
        world.set_vehicle_input(
            v,
            VehicleInput {
                forward: if fast { 0.0 } else { 1.0 },
                steer: steer as f32,
                ..VehicleInput::default()
            },
        );
        Ok(())
    }

    pub(crate) fn act(
        &mut self,
        world: &mut World,
        action: &Action,
        ev: &mut Vec<Event>,
    ) -> anyhow::Result<()> {
        match action {
            Action::Launch => {
                if self.phase == Phase::Ready {
                    self.phase = Phase::Approach;
                    self.t = -APPROACH;
                    note(ev, world, "launch");
                }
                Ok(())
            }
            Action::Bake => self.bake(world),
            Action::TakeWheel => {
                self.manual = true;
                Ok(())
            }
            Action::Reset => {
                // The host restores its saved world first; the script starts over.
                self.phase = Phase::Ready;
                self.t = 0.0;
                self.bake_t = None;
                self.vehicle = None;
                self.chassis = None;
                self.car_read = None;
                self.car = None;
                self.manual = false;
                note(ev, world, "reset");
                Ok(())
            }
            a => anyhow::bail!("the impact has no {} action", a.name()),
        }
    }

    pub(crate) fn state(&self) -> serde_json::Value {
        let phase = match self.phase {
            Phase::Ready => "ready",
            Phase::Approach => "approach",
            Phase::Open => "open",
            Phase::Baked => "baked",
            Phase::Drive => "drive",
        };
        serde_json::json!({ "phase": phase, "t": self.t, "bakeT": self.bake_t,
            "vehicle": self.vehicle.map(|v| v.0), "car": self.car.map(|c| c.0) })
    }
}

fn rot(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let [x, y, z, w] = q;
    let d = 2.0 * (x * v[0] + y * v[1] + z * v[2]);
    let s = w * w - (x * x + y * y + z * z);
    let c = [
        y * v[2] - z * v[1],
        z * v[0] - x * v[2],
        x * v[1] - y * v[0],
    ];
    [
        x * d + v[0] * s + c[0] * 2.0 * w,
        y * d + v[1] * s + c[1] * 2.0 * w,
        z * d + v[2] * s + c[2] * 2.0 * w,
    ]
}
