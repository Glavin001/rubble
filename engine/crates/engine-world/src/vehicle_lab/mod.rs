//! The vehicle test lab (`vehicle-lab.html`, `examples/vehicle_lab.rs`,
//! `engine/docs/vehicle-lab.md`): one soft-body car (the 2 kHz explicit solver) and the
//! things a tester throws at it, with the numbers that say what happened.
//!
//! - **Shoot**: rigid projectiles (box, sphere, cylinder; a density or a mass; 1–100 m/s;
//!   spin), each logged with its impact speed and the momentum it gave the car.
//! - **Compactor**: kinematic plates closing on the car from above, the sides or the
//!   ends, with force against travel and the work done.
//! - **Mountain**: an endless rocky slope ([`mountain`]) streamed around the falling car.
//! - **Drive**: obstacles put in front of the car (wall, pole, kerb, ramp, bricks).
//!
//! The lab's logic is [`Bench`]: commands and scripts in JSON, the objects' bookkeeping,
//! the statistics ([`monitor`]). It talks to its host through [`Rig`], so the same
//! scenario runs in the GPU World ([`Lab`]) and on the CPU reference ([`cpu::CpuLab`]:
//! the explicit solver with `explicit_rigid`'s proxies and no rigid solver: bodies fly
//! free between their contacts with the car).
//!
//! Nothing here changes solver behaviour: the lab places bodies, reads state and counts.

pub mod cpu;
pub mod monitor;
pub mod mountain;

use crate::explicit_scene::place_vehicle;
use crate::{
    BodyDef, BodyId, BodyType, ContactStorage, ExplicitVehiclesDef, Geometry, GpuContext,
    KillBounds, ShapeDef, Transform, World, WorldDef,
};
use engine_schema::convex::{CollisionFilter, ConvexHull};
use engine_softbody::explicit_beams::{ExplicitBeams, StaticShape, GRAVITY};
use engine_softbody::explicit_drive::DriveInput;
use engine_webgpu::{CouplingCapacity, CouplingLayout};
use glam::{DVec3, Quat, Vec3};
use monitor::{CarModel, CarRead, CarStats, ProxyFrame};
use mountain::{MountainSpec, RockGeom};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// The flat scene's ground height.
pub const GROUND: f32 = 0.0;
/// The collision category of what the car's own solver already collides with (its ground
/// plane, the arena's ramp and kerb): the coupling's mask leaves it out.
const FIXTURE_CATEGORY: u16 = 0x8000;
/// Projectiles alive at once; the oldest is removed for a new one.
pub const MAX_SHOTS: usize = 48;
/// Shots waiting in a burst.
pub const MAX_QUEUE: usize = 64;
/// Obstacle bodies alive at once (a brick stack is one body per brick).
pub const MAX_OBSTACLE_BODIES: usize = 240;

/// The lab's scene.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scene {
    Flat,
    Mountain,
}

/// How a lab is built.
#[derive(Clone, Debug, PartialEq)]
pub struct LabOptions {
    pub scene: Scene,
    /// The car's opt-in triangle-against-shape contact (`ExplicitBeams::surface_contact`).
    pub surface_contact: bool,
    /// A second copy of the car parked broadside ahead (the flat scene).
    pub parked: bool,
    /// The coupling's room per car; more in one window fails the step by name.
    pub capacity: CouplingCapacity,
    pub mountain: MountainSpec,
    /// Rigid bodies sleep after 0.5 s still.
    pub sleep: bool,
    /// The car's named parts (beam lists) for the per-part numbers.
    pub parts: Vec<(String, Vec<u32>)>,
}

impl Default for LabOptions {
    fn default() -> Self {
        Self {
            scene: Scene::Flat,
            surface_contact: false,
            parked: false,
            capacity: CouplingCapacity {
                bodies: 256,
                shapes: 256,
                planes: 4096,
            },
            mountain: MountainSpec::default(),
            sleep: true,
            parts: Vec::new(),
        }
    }
}

fn num(v: &Value, k: &str, d: f64) -> f64 {
    v.get(k).and_then(Value::as_f64).unwrap_or(d)
}

fn vec3(v: &Value, k: &str) -> Option<Vec3> {
    let a = v.get(k)?.as_array()?;
    (a.len() == 3).then(|| {
        Vec3::new(
            a[0].as_f64().unwrap_or(0.0) as f32,
            a[1].as_f64().unwrap_or(0.0) as f32,
            a[2].as_f64().unwrap_or(0.0) as f32,
        )
    })
}

impl LabOptions {
    /// `{ scene: 'flat' | 'mountain', surface, parked, shapes, planes, sleep, mountain:
    /// { seed, steepness, roughness, width, chunk, ahead, behind, launch, loose }, parts:
    /// { names, start, beams } }`, each defaulting to [`LabOptions::default`].
    pub fn from_json(v: &Value) -> Self {
        let d = Self::default();
        let m = v.get("mountain").cloned().unwrap_or(Value::Null);
        let dm = d.mountain;
        let shapes = (num(v, "shapes", f64::from(d.capacity.shapes)) as u32).clamp(8, 1024);
        let mut parts = Vec::new();
        if let Some(p) = v.get("parts") {
            let list = |k: &str| -> Vec<u64> {
                p.get(k)
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Value::as_u64).collect())
                    .unwrap_or_default()
            };
            let names: Vec<String> = p
                .get("names")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .map(|s| s.as_str().unwrap_or("").to_string())
                        .collect()
                })
                .unwrap_or_default();
            let (start, beams) = (list("start"), list("beams"));
            for (k, name) in names.iter().enumerate() {
                let (a, b) = (
                    start.get(k).copied().unwrap_or(0) as usize,
                    start.get(k + 1).copied().unwrap_or(beams.len() as u64) as usize,
                );
                if a <= b && b <= beams.len() {
                    parts.push((
                        name.clone(),
                        beams[a..b].iter().map(|&x| x as u32).collect(),
                    ));
                }
            }
        }
        Self {
            scene: if v.get("scene").and_then(Value::as_str) == Some("mountain") {
                Scene::Mountain
            } else {
                Scene::Flat
            },
            surface_contact: v
                .get("surface")
                .and_then(Value::as_bool)
                .unwrap_or(d.surface_contact),
            parked: v.get("parked").and_then(Value::as_bool).unwrap_or(d.parked),
            capacity: CouplingCapacity {
                bodies: shapes,
                shapes,
                planes: (num(v, "planes", f64::from(d.capacity.planes)) as u32).clamp(64, 65535),
            },
            mountain: MountainSpec {
                seed: m.get("seed").and_then(Value::as_u64).unwrap_or(dm.seed),
                steepness: num(&m, "steepness", f64::from(dm.steepness)) as f32,
                roughness: num(&m, "roughness", f64::from(dm.roughness)) as f32,
                width: num(&m, "width", f64::from(dm.width)) as f32,
                chunk: num(&m, "chunk", f64::from(dm.chunk)) as f32,
                ahead: num(&m, "ahead", f64::from(dm.ahead)) as u32,
                behind: num(&m, "behind", f64::from(dm.behind)) as u32,
                launch: num(&m, "launch", f64::from(dm.launch)) as f32,
                loose: num(&m, "loose", f64::from(dm.loose)) as u32,
                rockslide: m
                    .get("rockslide")
                    .and_then(Value::as_bool)
                    .unwrap_or(dm.rockslide),
            }
            .clamped(),
            sleep: v.get("sleep").and_then(Value::as_bool).unwrap_or(d.sleep),
            parts,
        }
    }
}

/// A lab object's shape, in its own frame.
#[derive(Clone, Debug, PartialEq)]
pub enum LabGeom {
    Box {
        half: Vec3,
    },
    Sphere {
        radius: f32,
    },
    /// Along local y; a 16-sided prism in the World (it has no cylinder primitive).
    Cylinder {
        radius: f32,
        half_len: f32,
    },
    /// Along local y.
    Capsule {
        radius: f32,
        half_len: f32,
    },
    /// A convex hull of points.
    Hull {
        points: Arc<Vec<Vec3>>,
    },
}

/// The cylinder's sides as a prism.
pub const CYLINDER_SIDES: usize = 16;

impl LabGeom {
    /// The cylinder's prism (its circumscribed radius the cylinder's).
    fn cylinder_points(radius: f32, half_len: f32) -> Vec<Vec3> {
        (0..CYLINDER_SIDES)
            .flat_map(|k| {
                let a = k as f32 / CYLINDER_SIDES as f32 * std::f32::consts::TAU;
                let (x, z) = (radius * a.cos(), radius * a.sin());
                [Vec3::new(x, -half_len, z), Vec3::new(x, half_len, z)]
            })
            .collect()
    }

    pub fn volume(&self) -> f32 {
        match self {
            Self::Box { half } => 8.0 * half.x * half.y * half.z,
            Self::Sphere { radius } => 4.0 / 3.0 * std::f32::consts::PI * radius.powi(3),
            // The prism's volume (what the World weighs), not the round cylinder's.
            Self::Cylinder { radius, half_len } => {
                let n = CYLINDER_SIDES as f32;
                0.5 * n * radius * radius * (std::f32::consts::TAU / n).sin() * 2.0 * half_len
            }
            Self::Capsule { radius, half_len } => {
                std::f32::consts::PI * radius * radius * (2.0 * half_len + 4.0 / 3.0 * radius)
            }
            Self::Hull { points } => ConvexHull::from_points(points)
                .map(|h| h.mass_properties(1.0).mass)
                .unwrap_or(0.0),
        }
    }

    /// Principal moments of inertia per unit mass (the CPU reference's proxies).
    fn unit_inertia(&self) -> Vec3 {
        match self {
            Self::Box { half } => {
                Vec3::new(
                    half.y * half.y + half.z * half.z,
                    half.x * half.x + half.z * half.z,
                    half.x * half.x + half.y * half.y,
                ) / 3.0
            }
            Self::Sphere { radius } => Vec3::splat(0.4 * radius * radius),
            Self::Cylinder { radius, half_len } | Self::Capsule { radius, half_len } => {
                let side = 0.25 * radius * radius + half_len * half_len / 3.0;
                Vec3::new(side, 0.5 * radius * radius, side)
            }
            Self::Hull { points } => {
                let r = points.iter().fold(0.0f32, |m, p| m.max(p.length()));
                Vec3::splat(0.4 * r * r)
            }
        }
    }

    fn radius(&self) -> f32 {
        match self {
            Self::Box { half } => half.length(),
            Self::Sphere { radius } => *radius,
            Self::Cylinder { radius, half_len } => (radius * radius + half_len * half_len).sqrt(),
            Self::Capsule { radius, half_len } => radius + half_len,
            Self::Hull { points } => points.iter().fold(0.0f32, |m, p| m.max(p.length())),
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Box { half } => format!(
                "box {:.2} × {:.2} × {:.2} m",
                2.0 * half.x,
                2.0 * half.y,
                2.0 * half.z
            ),
            Self::Sphere { radius } => format!("sphere r {radius:.2} m"),
            Self::Cylinder { radius, half_len } => {
                format!("cylinder r {radius:.2} × {:.2} m", 2.0 * half_len)
            }
            Self::Capsule { radius, half_len } => {
                format!("capsule r {radius:.2} × {:.2} m", 2.0 * half_len)
            }
            Self::Hull { points } => format!("hull of {} points", points.len()),
        }
    }
}

/// How a lab object moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Motion {
    Static,
    Kinematic,
    Dynamic,
}

/// A lab object to put in the host's world.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjDef {
    pub geom: LabGeom,
    pub pos: DVec3,
    pub rot: Quat,
    pub vel: Vec3,
    pub ang_vel: Vec3,
    pub motion: Motion,
    pub density: f32,
    pub friction: f32,
    /// Continuous collision on the rigid side (a fast projectile against other bodies).
    pub bullet: bool,
}

impl ObjDef {
    fn fixed(geom: LabGeom, pos: DVec3, rot: Quat, tag: f32) -> Self {
        Self {
            geom,
            pos,
            rot,
            vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            motion: Motion::Static,
            density: tag,
            friction: 0.6,
            bullet: false,
        }
    }
}

/// What a lab host provides: bodies in, bodies out, and the car's controls. A key names
/// a body's (one) shape in the coupling's records (the World's collider index).
pub trait Rig {
    fn spawn(&mut self, def: &ObjDef) -> anyhow::Result<u32>;
    fn set_velocity(&mut self, key: u32, vel: Vec3) -> anyhow::Result<()>;
    fn remove(&mut self, key: u32);
    fn set_drive(&mut self, input: DriveInput);
}

/// Material presets (kg/m³).
pub const MATERIALS: [(&str, f32); 5] = [
    ("steel", 7850.0),
    ("aluminium", 2700.0),
    ("concrete", 2400.0),
    ("wood", 600.0),
    ("rubber", 1100.0),
];

#[derive(Clone, Debug)]
struct ShotSpec {
    geom: LabGeom,
    density: f32,
    material: String,
    speed: f32,
    origin: DVec3,
    dir: Vec3,
    spin: Vec3,
    cross: bool,
    bullet: bool,
}

/// A projectile and what it did.
#[derive(Clone, Debug)]
pub struct Shot {
    pub id: u32,
    key: Option<u32>,
    pub label: String,
    pub material: String,
    pub mass: f32,
    pub speed: f32,
    dir: Vec3,
    fired: f64,
    v_prev: Vec3,
    t_prev: f64,
    /// The first window with an impulse on it: time and its speed going in (m/s).
    pub impact: Option<(f64, f32)>,
    v_impact: Vec3,
    v_last: Vec3,
    t_last: f64,
    /// Σ of the windows' impulses on it that were read (N s), their count, and the
    /// largest one-step force (N).
    pub impulse: Vec3,
    pub windows: u32,
    pub peak: f32,
    /// The deepest a car node was seen inside it (m) and whether its centre was seen
    /// inside the car's collision hull (then, and at the last reading).
    pub depth: f32,
    pub inside_ever: bool,
    inside_last: bool,
    seen: u64,
}

impl Shot {
    /// The momentum it gave the car (N s): its own change since the impact began, the
    /// gravity it fell under taken out.
    pub fn momentum(&self) -> Vec3 {
        match self.impact {
            Some((t, _)) => {
                self.mass
                    * (self.v_impact - self.v_last
                        + Vec3::new(0.0, GRAVITY, 0.0) * (self.t_last - t) as f32)
            }
            None => Vec3::ZERO,
        }
    }

    pub fn outcome(&self) -> &'static str {
        if self.impact.is_none() {
            return if self.key.is_some() {
                "no contact"
            } else {
                "removed"
            };
        }
        // By what it kept of its speed along the shot when it was last the car's proxy.
        let along = self.v_last.dot(self.dir) / self.speed.max(1e-3);
        if self.inside_last && self.v_last.length() < 1.0 {
            "embedded"
        } else if along > 0.5 {
            if self.inside_ever {
                "passed through"
            } else {
                "glanced"
            }
        } else if along > 0.05 {
            if self.inside_ever {
                "slowed, went through"
            } else {
                "slowed"
            }
        } else if along < -0.05 {
            "bounced"
        } else {
            "stopped"
        }
    }

    fn json(&self) -> Value {
        let p = self.momentum();
        json!({
            "id": self.id, "what": self.label, "material": self.material, "mass": self.mass, "speed": self.speed,
            "firedAt": self.fired,
            "impactAt": self.impact.map(|i| i.0), "impactSpeed": self.impact.map(|i| i.1),
            "momentum": p.length(), "momentumAlong": p.dot(self.dir), "momentumOwn": self.mass * self.speed,
            "impulseRead": self.impulse.length(), "windows": self.windows, "peakForce": self.peak,
            "exitSpeed": self.impact.map(|_| self.v_last.length()), "exitAlong": self.impact.map(|_| self.v_last.dot(self.dir)),
            "lastSeen": self.t_last, "nodeDepth": self.depth, "insideHull": self.inside_ever, "outcome": self.outcome(), "live": self.key.is_some(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlateState {
    Closing,
    Opening,
    Paused,
    Stopped,
}

/// A compactor plate.
#[derive(Clone, Debug)]
pub struct Plate {
    key: u32,
    pub name: String,
    /// The closing direction (unit) and speed (m/s).
    dir: Vec3,
    pub speed: f32,
    /// Travel so far and at the stop gap (m).
    pub travel: f32,
    pub stroke: f32,
    state: PlateState,
    resume: PlateState,
    /// The car's push back along the closing direction (N), the window's mean.
    pub force: f32,
    pub force_peak: f32,
    /// Largest one-step force (N).
    pub step_peak: f32,
    /// ∫ force d(travel) (J).
    pub work: f64,
    /// Deepest car node seen inside it (m).
    pub depth: f32,
    pub windows: u32,
    /// (travel, force) per window read.
    pub series: Vec<[f32; 2]>,
    moved: f32,
}

impl Plate {
    fn velocity(&self) -> Vec3 {
        match self.state {
            PlateState::Closing => self.dir * self.speed,
            PlateState::Opening => -self.dir * self.speed,
            _ => Vec3::ZERO,
        }
    }

    fn json(&self) -> Value {
        json!({
            "name": self.name, "speed": self.speed, "travel": self.travel, "stroke": self.stroke,
            "state": match self.state { PlateState::Closing => "closing", PlateState::Opening => "opening", PlateState::Paused => "paused", PlateState::Stopped => "stopped" },
            "force": self.force, "forcePeak": self.force_peak, "stepPeak": self.step_peak, "work": self.work,
            "nodeDepth": self.depth, "windows": self.windows, "samples": self.series.len(),
        })
    }
}

struct MountainRun {
    spec: MountainSpec,
    palette: Vec<Arc<Vec<Vec3>>>,
    live: BTreeMap<i64, Vec<u32>>,
    start: Option<DVec3>,
    made: u64,
    dropped: u64,
    /// Seconds the car has been at rest (negative: a boulder is on its way).
    rest: f32,
    boulders: Vec<(u32, f64)>,
    rockslides: u32,
}

/// The lab's logic, host independent.
pub struct Bench {
    pub opts: LabOptions,
    pub model: CarModel,
    pub car: CarStats,
    /// World time (s) and the World step (s).
    pub t: f64,
    pub dt: f32,
    pub shots: Vec<Shot>,
    queue: Vec<(f64, ShotSpec)>,
    next_shot: u32,
    pub plates: Vec<Plate>,
    obstacles: Vec<u32>,
    mountain: Option<MountainRun>,
    script: Vec<(f64, Value)>,
    script_at: usize,
    pub script_end: Option<f64>,
    kinds: HashMap<u32, (u8, usize)>,
    pub notes: Vec<String>,
    /// The car's extents about its centre of mass along (left, up, forward) at the last
    /// reading.
    ext: [Vec3; 2],
}

const KIND_SHOT: u8 = 0;
const KIND_PLATE: u8 = 1;

impl Bench {
    pub fn new(opts: LabOptions, car: &ExplicitBeams, dt: f32) -> Self {
        let mountain = (opts.scene == Scene::Mountain).then(|| MountainRun {
            spec: opts.mountain,
            palette: opts.mountain.palette().into_iter().map(Arc::new).collect(),
            live: BTreeMap::new(),
            start: None,
            made: 0,
            dropped: 0,
            rest: 0.0,
            boulders: Vec::new(),
            rockslides: 0,
        });
        Self {
            model: CarModel::new(car, opts.parts.clone()),
            opts,
            car: CarStats::default(),
            t: 0.0,
            dt,
            shots: Vec::new(),
            queue: Vec::new(),
            next_shot: 1,
            plates: Vec::new(),
            obstacles: Vec::new(),
            mountain,
            script: Vec::new(),
            script_at: 0,
            script_end: None,
            kinds: HashMap::new(),
            notes: Vec::new(),
            ext: [Vec3::splat(-1.0), Vec3::splat(1.0)],
        }
    }

    /// The car's frame at the last reading: centre of mass (world), left, forward.
    fn frame(&self) -> (DVec3, Vec3, Vec3) {
        let h = if self.car.heading.length() > 0.5 {
            self.car.heading
        } else {
            Vec3::Z
        };
        (self.car.centre, Vec3::Y.cross(h), h)
    }

    fn ground(&self) -> f64 {
        match self.opts.scene {
            Scene::Flat => f64::from(GROUND),
            Scene::Mountain => self.car.centre.y - 1.0,
        }
    }

    fn shot_spec(&self, v: &Value) -> anyhow::Result<ShotSpec> {
        let shape = v.get("shape").and_then(Value::as_str).unwrap_or("sphere");
        let size = vec3(v, "size");
        let radius = num(v, "radius", 0.1) as f32;
        let geom = match shape {
            "box" | "cuboid" => LabGeom::Box {
                half: 0.5 * size.unwrap_or(Vec3::splat(0.2)),
            },
            "sphere" => LabGeom::Sphere { radius },
            "cylinder" => LabGeom::Cylinder {
                radius,
                half_len: 0.5 * num(v, "length", 0.4) as f32,
            },
            other => {
                anyhow::bail!("vehicle lab: unknown shot shape '{other}' (box, sphere, cylinder)")
            }
        };
        let dims_ok = match &geom {
            LabGeom::Box { half } => half.min_element() >= 0.01 && half.max_element() <= 2.0,
            LabGeom::Sphere { radius } => (0.02..=1.5).contains(radius),
            LabGeom::Cylinder { radius, half_len } => {
                (0.02..=1.5).contains(radius) && (0.01..=2.0).contains(half_len)
            }
            _ => true,
        };
        anyhow::ensure!(
            dims_ok,
            "vehicle lab: shot size out of range (2 cm to 4 m): {}",
            geom.describe()
        );
        let material = v
            .get("material")
            .and_then(Value::as_str)
            .unwrap_or("steel")
            .to_string();
        let mut density = MATERIALS
            .iter()
            .find(|m| m.0 == material)
            .map(|m| m.1)
            .unwrap_or(7850.0);
        if let Some(d) = v.get("density").and_then(Value::as_f64) {
            density = d as f32;
        }
        let mut material = material;
        if let Some(m) = v.get("mass").and_then(Value::as_f64) {
            density = m as f32 / geom.volume();
            material = format!("{m} kg");
        }
        anyhow::ensure!(
            density.is_finite() && (1.0..=50_000.0).contains(&density),
            "vehicle lab: shot density {density} kg/m³ out of range (1 to 50,000)"
        );
        let speed = num(v, "speed", 20.0) as f32;
        anyhow::ensure!(
            (0.0..=100.0).contains(&speed),
            "vehicle lab: shot speed {speed} m/s out of range (0 to 100)"
        );
        let (c, left, fwd) = self.frame();
        let (origin, dir) = if let (Some(o), Some(d)) =
            (v.get("origin").and_then(Value::as_array), vec3(v, "dir"))
        {
            anyhow::ensure!(
                o.len() == 3 && d.length() > 1e-6,
                "vehicle lab: a shot needs origin [x, y, z] and a direction"
            );
            (
                DVec3::new(
                    o[0].as_f64().unwrap_or(0.0),
                    o[1].as_f64().unwrap_or(0.0),
                    o[2].as_f64().unwrap_or(0.0),
                ),
                d.normalize(),
            )
        } else {
            // { aim: { side, height (m above the ground), along (m toward the front),
            // across (m toward the left), distance (m from the car's side) } }.
            let a = v.get("aim").cloned().unwrap_or(Value::Null);
            let side = a.get("side").and_then(Value::as_str).unwrap_or("left");
            let dist = num(&a, "distance", 6.0) as f32;
            let along = num(&a, "along", 0.0) as f32;
            let across = num(&a, "across", 0.0) as f32;
            let height = num(&a, "height", 0.9);
            let (axis, reach) = match side {
                "left" => (left, self.ext[1].x),
                "right" => (-left, -self.ext[0].x),
                "front" => (fwd, self.ext[1].z),
                "rear" => (-fwd, -self.ext[0].z),
                "top" => (Vec3::Y, self.ext[1].y),
                other => anyhow::bail!(
                    "vehicle lab: unknown aim side '{other}' (left, right, front, rear, top)"
                ),
            };
            let mut o = c + (axis * (reach + dist)).as_dvec3();
            if side == "top" {
                o += (fwd * along + left * across).as_dvec3();
            } else {
                o.y = self.ground() + height;
                o += (if side == "left" || side == "right" {
                    fwd * along
                } else {
                    left * across
                })
                .as_dvec3();
            }
            (o, -axis)
        };
        let spin = vec3(v, "spin").unwrap_or(dir * num(v, "spinRate", 0.0) as f32);
        anyhow::ensure!(spin.length() <= 600.0, "vehicle lab: spin above 600 rad/s");
        Ok(ShotSpec {
            geom,
            density,
            material,
            speed,
            origin,
            dir,
            spin,
            cross: v.get("axis").and_then(Value::as_str) == Some("cross"),
            bullet: v.get("bullet").and_then(Value::as_bool).unwrap_or(true),
        })
    }

    fn launch(&mut self, rig: &mut dyn Rig, s: &ShotSpec) -> anyhow::Result<u32> {
        while self.shots.iter().filter(|x| x.key.is_some()).count() >= MAX_SHOTS {
            let old = self
                .shots
                .iter_mut()
                .find(|x| x.key.is_some())
                .expect("a live shot");
            let key = old.key.take().expect("live");
            rig.remove(key);
            self.kinds.remove(&key);
        }
        // A box flies with its local z along the shot; a cylinder end on (its axis, local
        // y, along the shot) or, `cross`, side on.
        let rot = match (&s.geom, s.cross) {
            (LabGeom::Cylinder { .. }, false) => Quat::from_rotation_arc(Vec3::Y, s.dir),
            (LabGeom::Cylinder { .. }, true) => Quat::from_rotation_arc(Vec3::Z, s.dir),
            (LabGeom::Box { .. }, _) => Quat::from_rotation_arc(Vec3::Z, s.dir),
            _ => Quat::IDENTITY,
        };
        let vel = s.dir * s.speed;
        let key = rig.spawn(&ObjDef {
            geom: s.geom.clone(),
            pos: s.origin,
            rot,
            vel,
            ang_vel: s.spin,
            motion: Motion::Dynamic,
            density: s.density,
            friction: 0.5,
            bullet: s.bullet,
        })?;
        let id = self.next_shot;
        self.next_shot += 1;
        self.kinds.insert(key, (KIND_SHOT, self.shots.len()));
        self.shots.push(Shot {
            id,
            key: Some(key),
            label: s.geom.describe(),
            material: s.material.clone(),
            mass: s.density * s.geom.volume(),
            speed: s.speed,
            dir: s.dir,
            fired: self.t,
            v_prev: vel,
            t_prev: self.t,
            impact: None,
            v_impact: vel,
            v_last: vel,
            t_last: self.t,
            impulse: Vec3::ZERO,
            windows: 0,
            peak: 0.0,
            depth: 0.0,
            inside_ever: false,
            inside_last: false,
            seen: 0,
        });
        Ok(id)
    }

    fn remove_plates(&mut self, rig: &mut dyn Rig) {
        for p in self.plates.drain(..) {
            rig.remove(p.key);
            self.kinds.remove(&p.key);
        }
    }

    fn add_plates(&mut self, rig: &mut dyn Rig, v: &Value) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.car.readings > 0,
            "vehicle lab: the compactor needs a reading of the car first"
        );
        let axis = v.get("axis").and_then(Value::as_str).unwrap_or("top");
        let speed = num(v, "speed", 0.1) as f32;
        anyhow::ensure!(
            (0.005..=2.0).contains(&speed),
            "vehicle lab: plate speed {speed} m/s out of range (0.005 to 2)"
        );
        let gap = num(v, "gap", 0.3) as f32;
        anyhow::ensure!(
            (0.0..=3.0).contains(&gap),
            "vehicle lab: plate stop gap {gap} m out of range (0 to 3)"
        );
        let clear = num(v, "clearance", 0.15) as f32;
        self.remove_plates(rig);
        let (c, left, fwd) = self.frame();
        let yaw = Quat::from_rotation_arc(Vec3::Z, fwd);
        let [lo, hi] = self.ext;
        let ground = self.ground() as f32 - c.y as f32;
        let (wide, long) = (hi.x - lo.x, hi.z - lo.z);
        let thick = 0.15f32;
        // (name, half extents in the car's frame, centre in it, closing direction, stroke)
        let mid = Vec3::new(0.5 * (lo.x + hi.x), 0.0, 0.5 * (lo.z + hi.z));
        let tall = (hi.y - ground + 0.4).max(1.0);
        let list: Vec<(&str, Vec3, Vec3, Vec3, f32)> = match axis {
            "top" => vec![(
                "top",
                Vec3::new(0.5 * wide + 0.5, thick, 0.5 * long + 0.5),
                Vec3::new(mid.x, hi.y + clear + thick, mid.z),
                -Vec3::Y,
                hi.y + clear - (ground + gap),
            )],
            "sides" => [(1.0f32, "left", hi.x), (-1.0, "right", lo.x)]
                .into_iter()
                .map(|(s, name, edge)| {
                    (
                        name,
                        Vec3::new(thick, 0.5 * tall, 0.5 * long + 0.5),
                        Vec3::new(edge + s * (clear + thick), ground + 0.5 * tall, mid.z),
                        Vec3::new(-s, 0.0, 0.0),
                        (edge - mid.x) * s + clear - 0.5 * gap,
                    )
                })
                .collect(),
            "ends" => [(1.0f32, "front", hi.z), (-1.0, "rear", lo.z)]
                .into_iter()
                .map(|(s, name, edge)| {
                    (
                        name,
                        Vec3::new(0.5 * wide + 0.5, 0.5 * tall, thick),
                        Vec3::new(mid.x, ground + 0.5 * tall, edge + s * (clear + thick)),
                        Vec3::new(0.0, 0.0, -s),
                        (edge - mid.z) * s + clear - 0.5 * gap,
                    )
                })
                .collect(),
            other => {
                anyhow::bail!("vehicle lab: unknown compactor axis '{other}' (top, sides, ends)")
            }
        };
        let world = |p: Vec3| left * p.x + Vec3::Y * p.y + fwd * p.z;
        for (name, half, at, dir, stroke) in list {
            let dir = world(dir);
            let mut plate = Plate {
                key: 0,
                name: name.into(),
                dir,
                speed,
                travel: 0.0,
                stroke: stroke.max(0.0),
                state: PlateState::Closing,
                resume: PlateState::Closing,
                force: 0.0,
                force_peak: 0.0,
                step_peak: 0.0,
                work: 0.0,
                depth: 0.0,
                windows: 0,
                series: Vec::new(),
                moved: 0.0,
            };
            plate.key = rig.spawn(&ObjDef {
                geom: LabGeom::Box { half },
                pos: c + world(at).as_dvec3(),
                rot: yaw,
                vel: plate.velocity(),
                ang_vel: Vec3::ZERO,
                motion: Motion::Kinematic,
                density: 7800.0,
                friction: 0.6,
                bullet: false,
            })?;
            self.kinds
                .insert(plate.key, (KIND_PLATE, self.plates.len()));
            self.plates.push(plate);
        }
        Ok(())
    }

    fn add_obstacle(&mut self, rig: &mut dyn Rig, v: &Value) -> anyhow::Result<()> {
        let kind = v.get("kind").and_then(Value::as_str).unwrap_or("wall");
        let (c, left, fwd) = self.frame();
        let ahead = num(v, "ahead", 12.0) as f32 + self.ext[1].z;
        let side = num(v, "side", 0.0) as f32;
        let ground = self.ground();
        let base = DVec3::new(c.x, ground, c.z) + (fwd * ahead + left * side).as_dvec3();
        let yaw = Quat::from_rotation_arc(Vec3::Z, fwd);
        let mut defs: Vec<ObjDef> = Vec::new();
        match kind {
            "wall" => {
                let half = Vec3::new(
                    0.5 * num(v, "width", 8.0) as f32,
                    0.5 * num(v, "height", 2.5) as f32,
                    0.5 * num(v, "thickness", 0.6) as f32,
                );
                defs.push(ObjDef::fixed(
                    LabGeom::Box { half },
                    base + DVec3::Y * f64::from(half.y),
                    yaw,
                    2400.0,
                ));
            }
            "pole" => {
                let (r, h) = (num(v, "radius", 0.15) as f32, num(v, "height", 4.0) as f32);
                defs.push(ObjDef::fixed(
                    LabGeom::Capsule {
                        radius: r,
                        half_len: 0.5 * h,
                    },
                    base + DVec3::Y * f64::from(0.5 * h),
                    Quat::IDENTITY,
                    7800.0,
                ));
            }
            "kerb" => {
                let half = Vec3::new(
                    0.5 * num(v, "width", 6.0) as f32,
                    0.5 * num(v, "height", 0.15) as f32,
                    0.15,
                );
                defs.push(ObjDef::fixed(
                    LabGeom::Box { half },
                    base + DVec3::Y * f64::from(half.y),
                    yaw,
                    2400.0,
                ));
            }
            "ramp" => {
                let (len, rise) = (num(v, "length", 5.0) as f32, num(v, "rise", 1.0) as f32);
                let half = Vec3::new(
                    0.5 * num(v, "width", 3.0) as f32,
                    0.2,
                    0.5 * (len * len + rise * rise).sqrt(),
                );
                let tilt = Quat::from_rotation_x(-(rise / len).atan());
                let up = tilt * Vec3::Y;
                let centre =
                    base + (fwd * (0.5 * len)).as_dvec3() + DVec3::Y * f64::from(0.5 * rise)
                        - (yaw * up * 0.2).as_dvec3();
                defs.push(ObjDef::fixed(
                    LabGeom::Box { half },
                    centre,
                    yaw * tilt,
                    2400.0,
                ));
            }
            "bricks" => {
                let (cols, rows, deep) = (
                    num(v, "cols", 6.0) as u32,
                    num(v, "rows", 5.0) as u32,
                    num(v, "deep", 1.0) as u32,
                );
                anyhow::ensure!(
                    cols * rows * deep <= 200 && cols * rows * deep > 0,
                    "vehicle lab: a brick stack of 1 to 200 bricks, not {}",
                    cols * rows * deep
                );
                let half = Vec3::new(0.5, 0.25, 0.25);
                let density = num(v, "density", 100.0) as f32;
                for d in 0..deep {
                    for r in 0..rows {
                        for k in 0..cols {
                            let off = if r % 2 == 1 { 0.5 } else { 0.0 };
                            let p = left * ((k as f32 - 0.5 * (cols as f32 - 1.0)) * 1.01 + off)
                                + Vec3::Y * (0.25 + 0.5 * r as f32)
                                + fwd * (0.51 * d as f32);
                            defs.push(ObjDef {
                                motion: Motion::Dynamic,
                                density,
                                friction: 0.5,
                                ..ObjDef::fixed(
                                    LabGeom::Box { half },
                                    base + p.as_dvec3(),
                                    yaw,
                                    density,
                                )
                            });
                        }
                    }
                }
            }
            other => anyhow::bail!(
                "vehicle lab: unknown obstacle '{other}' (wall, pole, kerb, ramp, bricks)"
            ),
        }
        anyhow::ensure!(
            self.obstacles.len() + defs.len() <= MAX_OBSTACLE_BODIES,
            "vehicle lab: more than {MAX_OBSTACLE_BODIES} obstacle bodies; clear some first"
        );
        for d in &defs {
            self.obstacles.push(rig.spawn(d)?);
        }
        Ok(())
    }

    /// One command: `{ fire: { shape, size | radius, length, material | density | mass,
    /// speed, spin | spinRate, origin + dir | aim, count, interval, axis } }`, `{ plates:
    /// { axis: top | sides | ends, speed, gap, clearance } }`, `{ plateControl: pause |
    /// resume | reverse | remove }`, `{ obstacle: { kind: wall | pole | kerb | ramp |
    /// bricks, ahead, side, … } }`, `{ clear: shots | obstacles | all }`, `{ drive: {
    /// forward, steer, brake, handbrake } }`.
    pub fn command(&mut self, rig: &mut dyn Rig, cmd: &Value) -> anyhow::Result<Value> {
        if let Some(f) = cmd.get("fire") {
            let spec = self.shot_spec(f)?;
            let count = (num(f, "count", 1.0) as usize).clamp(1, 32);
            let interval = num(f, "interval", 0.15).max(0.0);
            anyhow::ensure!(
                self.queue.len() + count <= MAX_QUEUE,
                "vehicle lab: more than {MAX_QUEUE} shots waiting"
            );
            let id = self.launch(rig, &spec)?;
            for k in 1..count {
                self.queue
                    .push((self.t + interval * k as f64, spec.clone()));
            }
            return Ok(json!({ "shot": id, "queued": count - 1 }));
        }
        if let Some(p) = cmd.get("plates") {
            self.add_plates(rig, p)?;
            return Ok(json!({ "plates": self.plates.len() }));
        }
        if let Some(c) = cmd.get("plateControl").and_then(Value::as_str) {
            match c {
                "remove" => self.remove_plates(rig),
                "pause" | "resume" | "reverse" => {
                    for p in &mut self.plates {
                        let next = match (c, p.state) {
                            ("pause", PlateState::Closing | PlateState::Opening) => {
                                p.resume = p.state;
                                PlateState::Paused
                            }
                            ("resume", PlateState::Paused) => p.resume,
                            ("reverse", PlateState::Closing) => PlateState::Opening,
                            ("reverse", PlateState::Opening) => PlateState::Closing,
                            ("reverse", PlateState::Paused) => {
                                p.resume = if p.resume == PlateState::Closing {
                                    PlateState::Opening
                                } else {
                                    PlateState::Closing
                                };
                                PlateState::Paused
                            }
                            ("reverse", PlateState::Stopped) => PlateState::Opening,
                            (_, s) => s,
                        };
                        p.state = next;
                        rig.set_velocity(p.key, p.velocity())?;
                    }
                }
                other => anyhow::bail!(
                    "vehicle lab: unknown plateControl '{other}' (pause, resume, reverse, remove)"
                ),
            }
            return Ok(Value::Null);
        }
        if let Some(o) = cmd.get("obstacle") {
            self.add_obstacle(rig, o)?;
            return Ok(json!({ "obstacles": self.obstacles.len() }));
        }
        if let Some(c) = cmd.get("clear").and_then(Value::as_str) {
            if c == "shots" || c == "all" {
                self.queue.clear();
                for s in &mut self.shots {
                    if let Some(k) = s.key.take() {
                        rig.remove(k);
                        self.kinds.remove(&k);
                    }
                }
            }
            if c == "obstacles" || c == "all" {
                for k in self.obstacles.drain(..) {
                    rig.remove(k);
                }
            }
            if c == "all" {
                self.remove_plates(rig);
            }
            return Ok(Value::Null);
        }
        if let Some(d) = cmd.get("drive") {
            let f = |k: &str| (num(d, k, 0.0) as f32).clamp(-1.0, 1.0);
            rig.set_drive(DriveInput {
                forward: f("forward"),
                steer: f("steer"),
                brake: f("brake").max(0.0),
                handbrake: f("handbrake").max(0.0),
            });
            return Ok(Value::Null);
        }
        anyhow::bail!("vehicle lab: unknown command {cmd}")
    }

    /// A scripted run: `{ events: [{ t, …command }], duration }`. Events run at their
    /// time (s of World time) in [`Bench::before_step`]; keys the lab does not know
    /// (`cam`, `slow`, `note`: the page's) are left alone.
    pub fn set_script(&mut self, script: &Value) -> anyhow::Result<()> {
        let mut events: Vec<(f64, Value)> = Vec::new();
        for e in script
            .get("events")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
        {
            let t = e
                .get("t")
                .and_then(Value::as_f64)
                .ok_or_else(|| anyhow::anyhow!("vehicle lab: a script event needs t: {e}"))?;
            if [
                "fire",
                "plates",
                "plateControl",
                "obstacle",
                "clear",
                "drive",
            ]
            .iter()
            .any(|k| e.get(k).is_some())
            {
                events.push((t, e));
            }
        }
        anyhow::ensure!(
            events.len() <= 1000,
            "vehicle lab: more than 1000 script events"
        );
        events.sort_by(|a, b| a.0.total_cmp(&b.0));
        self.script = events;
        self.script_at = 0;
        self.script_end = script.get("duration").and_then(Value::as_f64);
        Ok(())
    }

    /// Whether a script with a duration has run it out.
    pub fn script_done(&self) -> bool {
        self.script_end.is_some_and(|d| self.t >= d)
    }

    fn stream_mountain(&mut self, rig: &mut dyn Rig, all: bool) -> anyhow::Result<()> {
        let Some(m) = self.mountain.as_mut() else {
            return Ok(());
        };
        let at = if self.car.readings > 0 {
            m.spec.chunk_of(self.car.centre)
        } else {
            0
        };
        let (first, last) = (
            (at - i64::from(m.spec.behind)).max(-1),
            at + i64::from(m.spec.ahead),
        );
        let gone: Vec<i64> = m
            .live
            .keys()
            .copied()
            .filter(|k| *k < first || *k > last)
            .collect();
        for k in gone {
            for key in m.live.remove(&k).unwrap_or_default() {
                rig.remove(key);
            }
            m.dropped += 1;
        }
        // The nearest missing chunk first; one a step once the run is going.
        let mut order: Vec<i64> = (first..=last).filter(|k| !m.live.contains_key(k)).collect();
        order.sort_by_key(|k| (k - at).abs());
        for k in order {
            let chunk = m.spec.chunk(k);
            let mut keys = Vec::with_capacity(chunk.shapes.len());
            for s in &chunk.shapes {
                let geom = match &s.geom {
                    RockGeom::Box { half } => LabGeom::Box { half: *half },
                    RockGeom::Sphere { radius } => LabGeom::Sphere { radius: *radius },
                    RockGeom::Spike { palette } => LabGeom::Hull {
                        points: m.palette[*palette].clone(),
                    },
                };
                let mut def = ObjDef::fixed(geom, chunk.origin + s.pos.as_dvec3(), s.rot, s.tag);
                if s.loose {
                    def.motion = Motion::Dynamic;
                    def.density = 2300.0;
                }
                keys.push(rig.spawn(&def)?);
            }
            m.live.insert(k, keys);
            m.made += 1;
            if !all {
                break;
            }
        }
        // The rockslide: the car at rest for 2 s gets a 3.4 t boulder thrown at it from
        // 8 m up the slope (a dynamic rigid body: the World's contact with it, B1
        // included); boulders are removed after 8 s.
        let t = self.t;
        let old: Vec<u32> = m
            .boulders
            .iter()
            .filter(|b| t - b.1 > 8.0)
            .map(|b| b.0)
            .collect();
        m.boulders.retain(|b| t - b.1 <= 8.0);
        for key in old {
            rig.remove(key);
        }
        if m.spec.rockslide && m.rest > 2.0 && self.car.readings > 0 {
            let th = m.spec.steepness.to_radians();
            let (up, normal) = (
                Vec3::new(0.0, th.sin(), -th.cos()),
                Vec3::new(0.0, th.cos(), th.sin()),
            );
            let from = self.car.centre + (up * 8.0 + normal * 2.5).as_dvec3();
            let aim = (self.car.centre - from).as_vec3().normalize();
            let key = rig.spawn(&ObjDef {
                geom: LabGeom::Sphere { radius: 0.7 },
                pos: from,
                rot: Quat::IDENTITY,
                vel: aim * 14.0,
                ang_vel: Vec3::new(10.0, 0.0, 0.0),
                motion: Motion::Dynamic,
                density: 2400.0,
                friction: 0.6,
                bullet: true,
            })?;
            m.boulders.push((key, t));
            m.rockslides += 1;
            m.rest = -2.0;
            let fallen = m.start.map_or(0.0, |s| s.y - self.car.centre.y);
            self.car.events.push(monitor::Event {
                t,
                text: format!(
                    "at rest after {fallen:.0} m of fall: rockslide {}",
                    m.rockslides
                ),
            });
        }
        Ok(())
    }

    /// Before the host's first step: the scene's first objects.
    pub fn begin(&mut self, rig: &mut dyn Rig) -> anyhow::Result<()> {
        self.stream_mountain(rig, true)
    }

    /// Before each World step: due script events and queued shots, the plates' travel
    /// and stops, the mountain's chunks.
    pub fn before_step(&mut self, rig: &mut dyn Rig) -> anyhow::Result<()> {
        while self.script_at < self.script.len() && self.script[self.script_at].0 <= self.t + 1e-9 {
            let e = self.script[self.script_at].1.clone();
            self.script_at += 1;
            self.command(rig, &e)?;
        }
        let mut k = 0;
        while k < self.queue.len() {
            if self.queue[k].0 <= self.t + 1e-9 {
                let (_, spec) = self.queue.remove(k);
                self.launch(rig, &spec)?;
            } else {
                k += 1;
            }
        }
        let dt = self.dt;
        for p in &mut self.plates {
            // The step about to run moves the plate at its velocity.
            p.moved = 0.0;
            match p.state {
                PlateState::Closing => {
                    if p.travel + p.speed * dt > p.stroke {
                        p.state = PlateState::Stopped;
                        rig.set_velocity(p.key, Vec3::ZERO)?;
                    } else {
                        p.travel += p.speed * dt;
                        p.moved = p.speed * dt;
                    }
                }
                PlateState::Opening => {
                    if p.travel - p.speed * dt < 0.0 {
                        p.state = PlateState::Stopped;
                        rig.set_velocity(p.key, Vec3::ZERO)?;
                    } else {
                        p.travel -= p.speed * dt;
                        p.moved = -p.speed * dt;
                    }
                }
                _ => {}
            }
        }
        self.stream_mountain(rig, false)?;
        self.t += f64::from(dt);
        Ok(())
    }

    /// One window's reading of the car and its proxies.
    pub fn read(&mut self, car: &CarRead, frame: &ProxyFrame) {
        self.car.read(&self.model, car, frame, self.t);
        if self.car.non_finite {
            return;
        }
        // The car's extents in its own frame.
        let (c, left, fwd) = self.frame();
        let c = (c - car.origin).as_vec3();
        let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
        for p in car.pos {
            let d = *p - c;
            let e = Vec3::new(d.dot(left), d.y, d.dot(fwd));
            lo = lo.min(e);
            hi = hi.max(e);
        }
        self.ext = [lo, hi];
        if let Some(m) = self.mountain.as_mut() {
            m.start.get_or_insert(self.car.centre);
            m.rest = if self.car.speed < 0.7 || m.rest < 0.0 {
                m.rest + self.dt
            } else {
                0.0
            };
        }
        let reading = self.car.readings;
        let dt = self.dt;
        for (k, s) in frame.shapes.iter().enumerate() {
            let Some(&(kind, i)) = self.kinds.get(&s.collider) else {
                continue;
            };
            let (_, depth, centre_in) = self
                .car
                .per_shape
                .get(k)
                .copied()
                .unwrap_or((s.collider, 0.0, false));
            let body = frame.bodies[s.body];
            if kind == KIND_SHOT {
                let shot = &mut self.shots[i];
                let hit = s.impulse.length_squared() > 0.0;
                if hit && shot.impact.is_none() {
                    shot.impact = Some((self.t, shot.v_prev.length()));
                    shot.v_impact = shot.v_prev;
                    self.car.events.push(monitor::Event {
                        t: self.t,
                        text: format!(
                            "shot {} ({}, {:.1} kg) struck at {:.1} m/s",
                            shot.id,
                            shot.label,
                            shot.mass,
                            shot.v_prev.length()
                        ),
                    });
                }
                if hit {
                    shot.impulse += s.impulse;
                    shot.windows += 1;
                    shot.peak = shot.peak.max(s.peak);
                }
                shot.depth = shot.depth.max(depth);
                shot.inside_ever |= centre_in;
                shot.inside_last = centre_in;
                shot.v_last = body.vel;
                shot.t_last = self.t;
                shot.v_prev = body.vel;
                shot.t_prev = self.t;
                shot.seen = reading;
            } else {
                let p = &mut self.plates[i];
                // The reaction on the plate is the car's push: along the closing
                // direction it resists (negative impulse along it).
                let f = -s.impulse.dot(p.dir) / dt;
                p.force = f;
                p.force_peak = p.force_peak.max(f);
                p.step_peak = p.step_peak.max(s.peak);
                p.work += f64::from(f) * f64::from(p.moved);
                p.depth = p.depth.max(depth);
                p.windows += 1;
                if p.series.len() < 20_000 {
                    p.series.push([p.travel, f]);
                }
            }
        }
        // A plate with no proxy this window read no force.
        for p in &mut self.plates {
            if !frame.shapes.iter().any(|s| s.collider == p.key) {
                p.force = 0.0;
            }
        }
    }

    /// Plate `i`'s (travel, force) samples, flat.
    pub fn plate_series(&self, i: usize) -> Vec<f32> {
        self.plates
            .get(i)
            .map(|p| p.series.iter().flatten().copied().collect())
            .unwrap_or_default()
    }

    /// The lab's numbers.
    pub fn stats(&self) -> Value {
        let mountain = self.mountain.as_ref().map(|m| {
            let s = m.start.unwrap_or(self.car.centre);
            let d = self.car.centre - s;
            json!({
                "seed": m.spec.seed, "steepness": m.spec.steepness, "roughness": m.spec.roughness,
                "fallen": -d.y, "along": (d.y * d.y + d.z * d.z).sqrt(), "chunk": m.spec.chunk_of(self.car.centre),
                "chunksLive": m.live.len(), "chunksMade": m.made, "chunksDropped": m.dropped, "rockslides": m.rockslides,
                "bodies": m.live.values().map(Vec::len).sum::<usize>(),
            })
        });
        json!({
            "t": self.t,
            "scene": if self.opts.scene == Scene::Mountain { "mountain" } else { "flat" },
            "surfaceContact": self.opts.surface_contact,
            "car": self.car.json(&self.model),
            "shots": self.shots.iter().map(Shot::json).collect::<Vec<_>>(),
            "plates": self.plates.iter().map(Plate::json).collect::<Vec<_>>(),
            "obstacles": self.obstacles.len(),
            "mountain": mountain,
            "events": self.car.events.iter().map(|e| json!({ "t": e.t, "text": e.text })).collect::<Vec<_>>(),
            "damage": self.car.damage_series.iter().map(|d| json!([d.0, d.1, d.2])).collect::<Vec<_>>(),
            "limits": { "shots": MAX_SHOTS, "queue": MAX_QUEUE, "obstacleBodies": MAX_OBSTACLE_BODIES,
                "coupling": { "bodies": self.opts.capacity.bodies, "shapes": self.opts.capacity.shapes, "planes": self.opts.capacity.planes } },
            "scriptDone": self.script_done(),
            "notes": self.notes,
        })
    }

    /// A text table of the stats (the offline example's output).
    pub fn table(&self) -> String {
        use std::fmt::Write;
        let mut o = String::new();
        let c = &self.car;
        let _ = writeln!(o, "t {:.2} s   car {:.0} kg at ({:.2}, {:.2}, {:.2})   {:.1} km/h   KE {:.0} J   peak {:.1} g", self.t, self.model.mass, c.centre.x, c.centre.y, c.centre.z, c.speed * 3.6, c.kinetic, c.g_peak);
        let _ = writeln!(o, "beams {} broken, {} yielded of {}   deformation {:.3} m (peak {:.3}), {} nodes > 10 cm   attached mass {:.1} %", c.beams_broken, c.beams_yielded, c.beams_total, c.deform, c.deform_peak, c.nodes_moved, 100.0 * c.attached);
        let _ = writeln!(o, "penetration: deepest node in a rigid body {:.3} m (peak {:.3}), {} nodes inside; body points in the car's hull {} (deepest {:.3} m, peak {:.3})", c.node_depth, c.node_depth_peak, c.nodes_inside, c.body_points_inside, c.body_depth, c.body_depth_peak);
        let _ = writeln!(
            o,
            "coupling peak: {} bodies, {} shapes, {} planes of {:?}{}",
            c.proxies_peak[0],
            c.proxies_peak[1],
            c.proxies_peak[2],
            self.opts.capacity,
            if c.non_finite {
                "   NON-FINITE STATE"
            } else {
                ""
            }
        );
        if !self.shots.is_empty() {
            let _ = writeln!(
                o,
                "{:>3} {:<28} {:>9} {:>7} {:>8} {:>10} {:>10} {:>9} {:>8} {:>7}  outcome",
                "id",
                "shot",
                "mass kg",
                "m/s",
                "impact",
                "p given",
                "of own %",
                "peak kN",
                "exit m/s",
                "depth"
            );
            for s in &self.shots {
                let p = s.momentum();
                let _ = writeln!(
                    o,
                    "{:>3} {:<28} {:>9.2} {:>7.1} {:>8} {:>10.1} {:>10.1} {:>9.1} {:>8.1} {:>7.3}  {}",
                    s.id,
                    format!("{} {}", s.material, s.label),
                    s.mass,
                    s.speed,
                    s.impact.map_or("-".into(), |i| format!("{:.1}", i.1)),
                    p.length(),
                    100.0 * p.dot(s.dir) / (s.mass * s.speed).max(1e-6),
                    s.peak / 1000.0,
                    s.v_last.dot(s.dir),
                    s.depth,
                    s.outcome()
                );
            }
        }
        for p in &self.plates {
            let _ = writeln!(o, "plate {:<6} {:.3} m/s  travel {:.3} of {:.3} m  force now {:.1} kN, peak {:.1} kN (step peak {:.1} kN)  work {:.0} J  deepest node inside {:.3} m  ({} windows)", p.name, p.speed, p.travel, p.stroke, p.force / 1000.0, p.force_peak / 1000.0, p.step_peak / 1000.0, p.work, p.depth, p.windows);
        }
        if let Some(m) = &self.mountain {
            let d = c.centre - m.start.unwrap_or(c.centre);
            let _ = writeln!(o, "mountain seed {} ({}°, roughness {}): fallen {:.0} m ({:.0} m along), mechanical energy lost {:.0} kJ, chunks made {} dropped {}, rockslides {}", m.spec.seed, m.spec.steepness, m.spec.roughness, -d.y, (d.y * d.y + d.z * d.z).sqrt(), (c.mech0 - c.kinetic - c.potential) / 1000.0, m.made, m.dropped, m.rockslides);
        }
        for e in c.events.iter().take(40) {
            let _ = writeln!(o, "  {:>7.3} s  {}", e.t, e.text);
        }
        o
    }
}

/// The flat arena's fixtures, in the car's own solver (its raycast wheels see them, and
/// its nodes get the solver's static contact): a ramp to the car's left and a kerb to its
/// right, 10 m ahead of its start.
pub fn arena_fixtures() -> Vec<(Vec3, Vec3, Quat)> {
    let (len, rise) = (6.0f32, 1.0f32);
    let tilt = Quat::from_rotation_x(-(rise / len).atan());
    let half = Vec3::new(1.6, 0.2, 0.5 * (len * len + rise * rise).sqrt());
    let centre = Vec3::new(14.0, GROUND + 0.5 * rise, 12.0 + 0.5 * len) - tilt * Vec3::Y * 0.2;
    vec![
        (centre, half, tilt),
        (
            Vec3::new(-14.0, GROUND + 0.06, 12.0),
            Vec3::new(3.0, 0.06, 0.2),
            Quat::IDENTITY,
        ),
    ]
}

/// The lab's car from `template` for `opts`: placed, its ground and fixtures set, surface
/// contact as asked.
pub fn lab_car(template: &ExplicitBeams, opts: &LabOptions) -> ExplicitBeams {
    let mut car = match opts.scene {
        Scene::Flat => {
            let mut c = place_vehicle(template, Vec3::new(0.0, GROUND, 2.3), 0, 0.0, GROUND);
            for (centre, half, rot) in arena_fixtures() {
                c.shapes.push(StaticShape::oriented_box(centre, half, rot));
            }
            c
        }
        Scene::Mountain => {
            // Just off the lip, at the launch speed; its own ground plane far below (the
            // mountain is the World's static bodies).
            let len = template
                .nodes
                .iter()
                .map(|n| n.pos.z)
                .fold(f32::MIN, f32::max)
                - template
                    .nodes
                    .iter()
                    .map(|n| n.pos.z)
                    .fold(f32::MAX, f32::min);
            let mut c = place_vehicle(
                template,
                Vec3::new(0.0, 0.3, len.abs() + 0.3),
                0,
                opts.mountain.launch,
                -1.0e5,
            );
            if let Some(d) = c.drive.as_mut() {
                d.roll(opts.mountain.launch);
            }
            c
        }
    };
    car.surface_contact = opts.surface_contact;
    car
}

/// The lab's World settings: the brick ring drive's solver (1/60 s steps of 4 iterations,
/// persistent contacts) for a few thousand bodies.
pub fn lab_world_def(opts: &LabOptions) -> WorldDef {
    let mut def = crate::brick_ring_drive::ring_world_def(4096);
    def.gravity = [0.0, GRAVITY, 0.0];
    // The mountain run goes down for kilometres.
    def.kill_bounds = Some(KillBounds::below(if opts.scene == Scene::Mountain {
        -1.0e6
    } else {
        -50.0
    }));
    def.contacts = ContactStorage::Persistent;
    if opts.sleep {
        def.solver.sleep_time = 0.5;
    }
    def
}

struct WorldRig<'a> {
    world: &'a mut World,
    bodies: &'a mut HashMap<u32, BodyId>,
}

impl Rig for WorldRig<'_> {
    fn spawn(&mut self, d: &ObjDef) -> anyhow::Result<u32> {
        let body = self.world.create_body(&BodyDef {
            body_type: match d.motion {
                Motion::Static => BodyType::Static,
                Motion::Kinematic => BodyType::Kinematic,
                Motion::Dynamic => BodyType::Dynamic,
            },
            position: d.pos.as_vec3().to_array(),
            rotation: d.rot.to_array(),
            linear_velocity: d.vel.to_array(),
            angular_velocity: d.ang_vel.to_array(),
            is_bullet: d.bullet && d.motion == Motion::Dynamic,
            ..BodyDef::default()
        })?;
        let def = ShapeDef {
            material: crate::SurfaceMaterial {
                friction: d.friction,
                ..Default::default()
            },
            density: d.density,
            ..ShapeDef::default()
        };
        let shape = match &d.geom {
            LabGeom::Box { half } => self.world.create_box_shape(body, &def, half.to_array()),
            LabGeom::Sphere { radius } => self
                .world
                .create_sphere_shape(body, &def, [0.0; 3], *radius),
            LabGeom::Capsule { radius, half_len } => self
                .world
                .create_capsule_shape(body, &def, [0.0; 3], *radius, *half_len),
            LabGeom::Cylinder { radius, half_len } => {
                let hull = ConvexHull::from_points(&LabGeom::cylinder_points(*radius, *half_len))
                    .map_err(|e| anyhow::anyhow!("vehicle lab: cylinder hull: {e:?}"))?;
                self.world.create_hull_shape(body, &def, &hull)
            }
            LabGeom::Hull { points } => {
                let hull = ConvexHull::from_points(points)
                    .map_err(|e| anyhow::anyhow!("vehicle lab: hull: {e:?}"))?;
                self.world.create_shape(
                    body,
                    &def,
                    Geometry::Hull {
                        hull,
                        transform: Transform::IDENTITY,
                    },
                )
            }
        };
        let shape = match shape {
            Ok(s) => s,
            Err(e) => {
                let _ = self.world.destroy_body(body);
                return Err(e);
            }
        };
        let key = self.world.shape_collider(shape)?;
        self.bodies.insert(key, body);
        Ok(key)
    }

    fn set_velocity(&mut self, key: u32, vel: Vec3) -> anyhow::Result<()> {
        match self.bodies.get(&key) {
            Some(b) if self.world.body_is_valid(*b) => {
                self.world.body_set_linear_velocity(*b, vel.to_array())
            }
            _ => Ok(()),
        }
    }

    fn remove(&mut self, key: u32) {
        if let Some(b) = self.bodies.remove(&key) {
            if self.world.body_is_valid(b) {
                let _ = self.world.destroy_body(b);
            }
        }
    }

    fn set_drive(&mut self, input: DriveInput) {
        if let Some(x) = self.world.explicit_vehicles_mut() {
            x.set_drive_input(0, input);
        }
    }
}

/// Where a host reads the lab car's state each window: byte ranges of the explicit
/// vehicles' node, misc and beam buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadPlan {
    pub nodes: (u64, u64),
    /// The car's origin record (16 bytes).
    pub origin: (u64, u64),
    pub coupling: (u64, u64),
    pub beams: (u64, u64),
}

/// The lab in a GPU World.
pub struct Lab {
    pub bench: Bench,
    bodies: HashMap<u32, BodyId>,
    layout: CouplingLayout,
    plan: ReadPlan,
}

impl Lab {
    /// A lab World on `gpu` with the car from `template` (a vehicle on raycast wheels,
    /// facing +z). `tweak` adjusts the World's definition (GPU timing).
    pub fn build(
        gpu: GpuContext,
        template: &ExplicitBeams,
        opts: LabOptions,
        tweak: impl FnOnce(&mut WorldDef),
    ) -> anyhow::Result<(World, Lab)> {
        let mut def = lab_world_def(&opts);
        tweak(&mut def);
        let mut world = World::new(gpu, &def)?;
        let fixture = |friction: f32| ShapeDef {
            material: crate::SurfaceMaterial {
                friction,
                ..Default::default()
            },
            density: 2400.0,
            filter: CollisionFilter {
                categories: FIXTURE_CATEGORY,
                ..CollisionFilter::default()
            },
            ..ShapeDef::default()
        };
        let car = lab_car(template, &opts);
        let mut cars = vec![car.clone()];
        if opts.scene == Scene::Flat {
            // The ground the rigid bodies stand on (the car has its own plane), and the
            // arena's ramp and kerb for the rigid bodies and the renderer.
            let g = world.create_body(&BodyDef {
                position: [0.0, GROUND - 0.5, 0.0],
                ..BodyDef::default()
            })?;
            world.create_box_shape(
                g,
                &ShapeDef {
                    density: 0.0,
                    ..fixture(0.6)
                },
                [400.0, 0.5, 400.0],
            )?;
            for (centre, half, rot) in arena_fixtures() {
                let b = world.create_body(&BodyDef {
                    position: centre.to_array(),
                    rotation: rot.to_array(),
                    ..BodyDef::default()
                })?;
                world.create_box_shape(b, &fixture(0.6), half.to_array())?;
            }
            if opts.parked {
                // Broadside across the car's path.
                let mut p = place_vehicle(template, Vec3::new(2.3, GROUND, 16.0), 1, 0.0, GROUND);
                p.shapes = car.shapes.clone();
                p.surface_contact = opts.surface_contact;
                cars.push(p);
            }
        }
        world.enable_sleep_archive(None)?;
        world.add_explicit_vehicles(
            &cars,
            &ExplicitVehiclesDef {
                capacity: opts.capacity,
                mask: !FIXTURE_CATEGORY,
                sleep: None,
                ..ExplicitVehiclesDef::default()
            },
        )?;
        let layout = world
            .explicit_coupling_layout()
            .ok_or_else(|| anyhow::anyhow!("vehicle lab: no coupling layout"))?;
        let x = world.explicit_vehicles().expect("vehicles");
        let span = x.layout().spans[0];
        let plan = ReadPlan {
            nodes: (u64::from(span.node_off) * 48, u64::from(span.nodes) * 48),
            origin: (0, 16),
            coupling: (
                u64::from(layout.vehicle(0)) * 16,
                u64::from(layout.recs()) * 16,
            ),
            beams: (u64::from(span.beam_off) * 32, u64::from(span.beams) * 32),
        };
        let dt = world.solver().dt;
        let mut lab = Lab {
            bench: Bench::new(opts, &car, dt),
            bodies: HashMap::new(),
            layout,
            plan,
        };
        lab.bench.begin(&mut WorldRig {
            world: &mut world,
            bodies: &mut lab.bodies,
        })?;
        Ok((world, lab))
    }

    pub fn read_plan(&self) -> ReadPlan {
        self.plan
    }

    pub fn command(&mut self, world: &mut World, cmd: &Value) -> anyhow::Result<Value> {
        self.bench.command(
            &mut WorldRig {
                world,
                bodies: &mut self.bodies,
            },
            cmd,
        )
    }

    /// Call before each `World::step`.
    pub fn before_step(&mut self, world: &mut World) -> anyhow::Result<()> {
        self.bench.before_step(&mut WorldRig {
            world,
            bodies: &mut self.bodies,
        })
    }

    /// One window's state as the buffers hold it ([`Lab::read_plan`]'s ranges, as
    /// words): the car's nodes, its origin record, its coupling records and (when read)
    /// its beams.
    pub fn ingest(
        &mut self,
        nodes: &[u32],
        origin: &[u32],
        coupling: &[u32],
        beams: Option<&[u32]>,
    ) -> anyhow::Result<()> {
        let n = nodes.len() / 12;
        let f = |w: u32| f32::from_bits(w);
        let pos: Vec<Vec3> = (0..n)
            .map(|i| Vec3::new(f(nodes[12 * i]), f(nodes[12 * i + 1]), f(nodes[12 * i + 2])))
            .collect();
        let vel: Vec<Vec3> = (0..n)
            .map(|i| {
                Vec3::new(
                    f(nodes[12 * i + 4]),
                    f(nodes[12 * i + 5]),
                    f(nodes[12 * i + 6]),
                )
            })
            .collect();
        anyhow::ensure!(origin.len() >= 3, "vehicle lab: no origin record");
        let o = DVec3::new(
            f64::from(f(origin[0])),
            f64::from(f(origin[1])),
            f64::from(f(origin[2])),
        );
        let frame = monitor::decode_coupling(&self.layout, coupling).map_err(anyhow::Error::msg)?;
        let beams: Option<Vec<(bool, f32)>> = beams.map(|b| {
            (0..b.len() / 8)
                .map(|j| (f(b[8 * j + 6]) != 0.0, f(b[8 * j])))
                .collect()
        });
        self.bench.read(
            &CarRead {
                pos: &pos,
                vel: &vel,
                beams: beams.as_deref(),
                origin: o,
            },
            &frame,
        );
        Ok(())
    }

    /// Read the car's state now (blocking; native hosts) and take it.
    pub fn read_blocking(&mut self, world: &World, beams: bool) -> anyhow::Result<()> {
        use engine_webgpu::explicit_coupling::read_words;
        let x = world
            .explicit_vehicles()
            .ok_or_else(|| anyhow::anyhow!("vehicle lab: no vehicles"))?;
        let (d, q) = (world.device(), world.queue());
        let p = self.plan;
        let nodes = read_words(d, q, x.nodes_buffer(), p.nodes.0, p.nodes.1);
        let origin = read_words(d, q, x.misc_buffer(), p.origin.0, p.origin.1);
        let coupling = read_words(d, q, x.misc_buffer(), p.coupling.0, p.coupling.1);
        let b = beams.then(|| read_words(d, q, x.beams_buffer(), p.beams.0, p.beams.1));
        self.ingest(&nodes, &origin, &coupling, b.as_deref())
    }
}
