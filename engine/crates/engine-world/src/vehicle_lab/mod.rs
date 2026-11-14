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
use engine_softbody::explicit_debris::{DebrisTracker, DebrisTtl, Pieces};
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
    /// Anti-clip's self-collision box padded by each contacter's own travel, not the
    /// car's fastest node's (`ExplicitBeams::own_travel_reach`; on in the lab: debris
    /// falling off the mountain no longer pads every triangle's box). Live
    /// (`{ ownTravel: bool }` command).
    pub own_travel_reach: bool,
    /// The GPU's reused self-collision pair list told which nodes are off the main body
    /// (`ExplicitBeamsGpu::set_piece_labels`, from each beam reading): a piece torn off no
    /// longer invalidates the main body's list. Lossless (the same results either way).
    /// Live (`{ pieceLabels: bool }`).
    pub piece_labels: bool,
    /// With the piece labels, the coupling's window per piece
    /// (`ExplicitBeamsGpu::set_piece_windows`; lossless). Live (`{ pieceWindows: bool }`).
    pub piece_windows: bool,
    /// Small debris retired after its time to live (`DebrisTtl`; None: never, the
    /// solver's behaviour without the setting). On in the lab: 5 s, at most 10 kg and 40
    /// nodes ([`LAB_DEBRIS_TTL`]: what the mountain sheds). Live
    /// (`{ debris: { on, seconds, maxKg, maxNodes } }`; retired pieces stay retired).
    pub debris_ttl: Option<DebrisTtl>,
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
            own_travel_reach: true,
            piece_labels: true,
            piece_windows: true,
            debris_ttl: Some(LAB_DEBRIS_TTL),
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

/// The lab's debris time to live (`LabOptions::debris_ttl`).
/// Measured on the mountain (lab-mountain-perf.md §6): its fragments weigh 0.3 to 5.6 kg
/// (2 to 12 nodes); the next pieces up are a 12 kg panel, the 19 kg wheels, 30 and 45 kg
/// parts. 10 kg retires every fragment and nothing a player would miss.
pub const LAB_DEBRIS_TTL: DebrisTtl = DebrisTtl { max_mass_kg: 10.0, max_nodes: 40, seconds: 5.0 };

/// `{ on, seconds, maxKg, maxNodes }` (missing: the lab's default) as a time to live.
pub fn debris_ttl_from(x: &Value) -> Option<DebrisTtl> {
    let d = LAB_DEBRIS_TTL;
    x.get("on").and_then(Value::as_bool).unwrap_or(true).then(|| DebrisTtl {
        max_mass_kg: (num(x, "maxKg", f64::from(d.max_mass_kg)) as f32).max(0.0),
        max_nodes: (num(x, "maxNodes", f64::from(d.max_nodes)) as u32).max(1),
        seconds: (num(x, "seconds", f64::from(d.seconds)) as f32).max(0.0),
    })
}

fn debris_json(t: Option<DebrisTtl>) -> Value {
    match t {
        Some(t) => json!({ "on": true, "seconds": t.seconds, "maxKg": t.max_mass_kg, "maxNodes": t.max_nodes }),
        None => json!({ "on": false }),
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
    /// `{ scene: 'flat' | 'mountain', surface, ownTravel, parked, shapes, planes, sleep, mountain:
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
            own_travel_reach: v
                .get("ownTravel")
                .and_then(Value::as_bool)
                .unwrap_or(d.own_travel_reach),
            piece_labels: v
                .get("pieceLabels")
                .and_then(Value::as_bool)
                .unwrap_or(d.piece_labels),
            piece_windows: v
                .get("pieceWindows")
                .and_then(Value::as_bool)
                .unwrap_or(d.piece_windows),
            debris_ttl: match v.get("debris") {
                Some(x) if !x.is_null() => debris_ttl_from(x),
                _ => d.debris_ttl,
            },
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
                start: num(&m, "start", f64::from(dm.start)) as f32,
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
    /// A crash preset's stated speed (m/s), and the car's speed in the window before an
    /// obstacle first took an impulse from it.
    pub target: Option<f32>,
    pub arrival: Option<f32>,
    /// The car's extents about its centre of mass along (left, up, forward) at the last
    /// reading.
    ext: [Vec3; 2],
    /// The car's nodes at the last reading (about `last_origin`): what a click picks.
    last: Vec<Vec3>,
    last_origin: DVec3,
    /// The car's pieces at the last beam reading, and its debris time to live: which
    /// nodes are retired, and a new set for the solver (the rig applies it).
    pub pieces: Option<Pieces>,
    pub debris: DebrisTracker,
    pub retire_due: Option<Vec<bool>>,
}

const KIND_SHOT: u8 = 0;
const KIND_PLATE: u8 = 1;

impl Bench {
    pub fn new(opts: LabOptions, car: &ExplicitBeams, dt: f32) -> Self {
        let ttl = opts.debris_ttl.filter(|_| DebrisTtl::check(car).is_ok());
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
            target: None,
            arrival: None,
            ext: [Vec3::splat(-1.0), Vec3::splat(1.0)],
            last: Vec::new(),
            last_origin: DVec3::ZERO,
            pieces: None,
            debris: DebrisTracker::for_vehicle(ttl, car),
            retire_due: None,
        }
        .with_debris_check(car)
    }

    fn with_debris_check(mut self, car: &ExplicitBeams) -> Self {
        if let (Some(_), Err(e)) = (self.opts.debris_ttl, DebrisTtl::check(car)) {
            self.notes.push(format!("debris time to live off: {e}"));
        }
        self
    }

    /// The debris time to live from now on (retired pieces stay retired).
    pub fn set_debris_ttl(&mut self, ttl: Option<DebrisTtl>) {
        self.opts.debris_ttl = ttl;
        self.debris.ttl = ttl;
    }

    /// The debris time to live failed in the solver (`reason`): off from now on.
    pub fn debris_failed(&mut self, reason: &str) {
        self.debris.ttl = None;
        self.opts.debris_ttl = None;
        self.notes.push(format!("debris time to live off: {reason}"));
        self.car.events.push(monitor::Event { t: self.t, text: format!("debris time to live off: {reason}") });
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
                // A 16-sided prism (the scene renderer draws boxes, hulls and spheres).
                defs.push(ObjDef::fixed(
                    LabGeom::Cylinder {
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
            // { pick: { origin, dir } }: a shot at what lies under a view ray (the page's
            // click): launched from the ray's origin at the elevation that carries it,
            // falling under gravity, through the picked point.
            let mut aimed = f.clone();
            let mut picked = Value::Null;
            if let Some(pick) = f.get("pick") {
                let o = pick
                    .get("origin")
                    .and_then(Value::as_array)
                    .filter(|o| o.len() == 3)
                    .ok_or_else(|| anyhow::anyhow!("vehicle lab: pick needs origin [x, y, z]"))?;
                let o = DVec3::new(
                    o[0].as_f64().unwrap_or(0.0),
                    o[1].as_f64().unwrap_or(0.0),
                    o[2].as_f64().unwrap_or(0.0),
                );
                let d = vec3(pick, "dir")
                    .filter(|d| d.length() > 1e-6)
                    .ok_or_else(|| anyhow::anyhow!("vehicle lab: pick needs a direction"))?
                    .normalize();
                let (target, on) = self.pick(o, d);
                let speed = num(f, "speed", 20.0) as f32;
                let (dir, reachable) = ballistic(target - o, speed);
                aimed["origin"] = json!([o.x, o.y, o.z]);
                aimed["dir"] = json!([dir.x, dir.y, dir.z]);
                picked = json!({ "target": [target.x, target.y, target.z], "on": on, "reachable": reachable });
            }
            let f = &aimed;
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
            return Ok(json!({ "shot": id, "queued": count - 1, "pick": picked }));
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

    /// One window's reading of the car and its proxies, taken now (a host that reads at
    /// the step).
    pub fn read(&mut self, car: &CarRead, frame: &ProxyFrame) {
        self.read_at(car, frame, self.t);
    }

    /// One window's reading of the car and its proxies, copied at the lab's time `taken`
    /// (s): a page's readback arrives frames after the step it was copied at, and the
    /// pose a camera carries forward needs the time of the state, not of its arrival.
    pub fn read_at(&mut self, car: &CarRead, frame: &ProxyFrame, taken: f64) {
        let before = self.car.speed;
        self.car.read(&self.model, car, frame, taken);
        // The pieces the unbroken beams leave; small ones retired after their time.
        if let Some(b) = car.beams {
            let pieces = Pieces::from_beams(
                &self.model.masses,
                self.model.beams.iter().zip(b).map(|(e, x)| (e[0], e[1], x.0)),
            );
            if self.debris.update(&pieces, taken) {
                self.retire_due = Some(self.debris.retired.clone());
            }
            for r in self.debris.take_events() {
                self.car.events.push(monitor::Event {
                    t: r.t,
                    text: format!(
                        "debris retired: {} nodes, {:.2} kg, {:.1} s off the main body",
                        r.nodes, r.mass, r.off_for
                    ),
                });
            }
            self.pieces = Some(pieces);
        }
        if self.arrival.is_none()
            && self.car.readings > 1
            && frame
                .shapes
                .iter()
                .any(|s| s.impulse.length_squared() > 0.0)
        {
            self.arrival = Some(before);
            self.car.events.push(monitor::Event {
                t: self.t,
                text: format!("first contact at {:.1} km/h", before * 3.6),
            });
        }
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
        self.last = car.pos.to_vec();
        self.last_origin = car.origin;
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

    /// The car now, for a camera: centre (world), velocity, heading, speed, the reading's
    /// World time (s), readings taken.
    pub fn pose(&self) -> [f64; 12] {
        let c = &self.car;
        [
            c.centre.x,
            c.centre.y,
            c.centre.z,
            c.velocity.x.into(),
            c.velocity.y.into(),
            c.velocity.z.into(),
            c.heading.x.into(),
            c.heading.y.into(),
            c.heading.z.into(),
            c.speed.into(),
            c.t,
            c.readings as f64,
        ]
    }

    /// What a ray from `o` along `d` (world) meets first: the car's collision skin as last
    /// read ("car"), else the flat arena's ground ("ground"), else the point of the ray
    /// nearest the car's centre ("air").
    pub fn pick(&self, o: DVec3, d: Vec3) -> (DVec3, &'static str) {
        let ro = (o - self.last_origin).as_vec3();
        let mut best = f32::INFINITY;
        for t in &self.model.triangles {
            let [a, b, c] = t.map(|k| self.last.get(k as usize).copied().unwrap_or(Vec3::NAN));
            // Möller–Trumbore, either side.
            let (e1, e2) = (b - a, c - a);
            let h = d.cross(e2);
            let det = e1.dot(h);
            if !(det.abs() > 1e-9) {
                continue;
            }
            let s = ro - a;
            let u = s.dot(h) / det;
            let q = s.cross(e1);
            let v = d.dot(q) / det;
            let hit = e2.dot(q) / det;
            if u >= 0.0 && v >= 0.0 && u + v <= 1.0 && hit > 0.05 && hit < best {
                best = hit;
            }
        }
        if best.is_finite() {
            return (o + (d * best).as_dvec3(), "car");
        }
        if self.opts.scene == Scene::Flat && d.y < -1e-4 {
            let t = (f64::from(GROUND) - o.y) / f64::from(d.y);
            if t > 0.0 && t < 500.0 {
                return (o + d.as_dvec3() * t, "ground");
            }
        }
        let along = ((self.car.centre - o).as_vec3().dot(d)).max(2.0);
        (o + (d * along).as_dvec3(), "air")
    }

    /// Whether something the per-step readings matter for is going on: a projectile in
    /// its first seconds, a plate, a crash preset before its first contact, a script.
    pub fn busy(&self) -> bool {
        !self.plates.is_empty()
            || !self.queue.is_empty()
            || self.script_at < self.script.len()
            || (self.target.is_some() && self.arrival.is_none())
            || self
                .shots
                .iter()
                .any(|s| s.key.is_some() && self.t - s.fired < 4.0)
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
            "ownTravel": self.opts.own_travel_reach,
            "pieceLabels": self.opts.piece_labels,
            "debris": debris_json(self.debris.ttl),
            "retired": { "pieces": self.debris.pieces, "nodes": self.debris.nodes, "mass": self.debris.mass },
            "car": self.car.json(&self.model),
            "shots": self.shots.iter().map(Shot::json).collect::<Vec<_>>(),
            "plates": self.plates.iter().map(Plate::json).collect::<Vec<_>>(),
            "obstacles": self.obstacles.len(),
            "mountain": mountain,
            "events": self.car.events.iter().map(|e| json!({ "t": e.t, "text": e.text })).collect::<Vec<_>>(),
            "damage": self.car.damage_series.iter().map(|d| json!([d.0, d.1, d.2])).collect::<Vec<_>>(),
            "limits": { "shots": MAX_SHOTS, "queue": MAX_QUEUE, "obstacleBodies": MAX_OBSTACLE_BODIES,
                "coupling": { "bodies": self.opts.capacity.bodies, "shapes": self.opts.capacity.shapes, "planes": self.opts.capacity.planes } },
            "targetKmh": self.target.map(|x| x * 3.6), "arrivalKmh": self.arrival.map(|x| x * 3.6),
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
        let _ = writeln!(o, "debris retired: {} pieces, {} nodes, {:.2} kg (time to live {:?})", self.debris.pieces, self.debris.nodes, self.debris.mass, self.debris.ttl);
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

/// The launch direction that carries a projectile of `speed` (m/s), falling under gravity,
/// from the origin through `to` (the low trajectory), and whether the speed reaches it
/// (if not: straight at it).
pub fn ballistic(to: DVec3, speed: f32) -> (Vec3, bool) {
    let g = f64::from(-GRAVITY);
    let (dx, dy) = ((to.x * to.x + to.z * to.z).sqrt(), to.y);
    let v2 = f64::from(speed) * f64::from(speed);
    let disc = v2 * v2 - g * (g * dx * dx + 2.0 * dy * v2);
    if dx < 1e-6 || disc < 0.0 || speed <= 0.0 {
        return (to.as_vec3().normalize_or_zero(), dx < 1e-6 && speed > 0.0);
    }
    let tan = (v2 - disc.sqrt()) / (g * dx);
    let h = DVec3::new(to.x / dx, 0.0, to.z / dx);
    ((h + DVec3::Y * tan).normalize().as_vec3(), true)
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

/// Where and how the car starts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Start {
    /// Speed along its heading (m/s) and to its left (m/s: a sled's sideways launch).
    pub speed: f32,
    pub lateral: f32,
    /// Its heading, turned about +y from +z (rad).
    pub yaw: f32,
    /// Its centre (x, z) and how far above its standing height it starts (m: on a sled
    /// its wheels do not reach the ground until it has dropped).
    pub at: [f32; 2],
    pub lift: f32,
}

impl Default for Start {
    fn default() -> Self {
        Self {
            speed: 0.0,
            lateral: 0.0,
            yaw: 0.0,
            at: [0.0, 0.0],
            lift: 0.0,
        }
    }
}

/// The lab's car from `template` for `opts`, started as `start` (none: the scene's own
/// start: standing on the flat arena, or just off the mountain's lip at its launch
/// speed): placed, its ground and fixtures set, surface contact as asked. Every lab car
/// of one template has the same layout (the same static shapes: the arena's fixtures,
/// far overhead on the mountain), so a restart is `World::reset_explicit_vehicles`.
pub fn lab_car(template: &ExplicitBeams, opts: &LabOptions, start: Option<Start>) -> ExplicitBeams {
    let mountain = opts.scene == Scene::Mountain;
    let ground = if mountain { -1.0e5 } else { GROUND };
    // Facing +z, standing on y = 0, centred on its node box.
    let mut car = place_vehicle(template, Vec3::ZERO, 0, 0.0, 0.0);
    let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
    for n in &car.nodes {
        lo = lo.min(n.pos);
        hi = hi.max(n.pos);
    }
    let mid = 0.5 * (lo + hi);
    let mountain_start = Start {
        speed: opts.mountain.launch,
        at: [0.0, 0.5 * (hi.z - lo.z) + 0.3],
        lift: 0.3,
        ..Start::default()
    };
    let start = start.unwrap_or(if mountain {
        // Just off the lip (its rear clear of it), at the launch speed.
        Start {
            speed: opts.mountain.launch,
            at: [0.0, 0.5 * (hi.z - lo.z) + 0.3],
            lift: 0.3,
            ..Start::default()
        }
    } else {
        Start::default()
    });
    let turn = Quat::from_rotation_y(start.yaw);
    let (fwd, left) = (turn * Vec3::Z, turn * Vec3::X);
    let vel = fwd * start.speed + left * start.lateral;
    let base = if mountain { 0.0 } else { GROUND };
    for n in &mut car.nodes {
        let mut p = n.pos - Vec3::new(mid.x, 0.0, mid.z);
        if start.yaw != 0.0 {
            p = turn * p;
        }
        n.pos = p + Vec3::new(start.at[0], base + start.lift, start.at[1]);
        n.vel = vel;
    }
    car.origin = Vec3::ZERO;
    if mountain && opts.mountain.start > 0.0 && start == mountain_start {
        // A run that starts down the mountain: the car 2 m over the fall line there, its
        // nodes about its own origin (as the solver's rebase leaves them).
        car.origin = (opts.mountain.line(f64::from(opts.mountain.start))
            + DVec3::new(0.0, 2.0, 0.0))
        .as_vec3();
    }
    car.ground = ground;
    if start.speed != 0.0 && start.lateral == 0.0 {
        if let Some(d) = car.drive.as_mut() {
            d.roll(start.speed);
        }
    }
    for (centre, half, rot) in arena_fixtures() {
        let up = if mountain {
            Vec3::new(0.0, 1.0e4, 0.0)
        } else {
            Vec3::ZERO
        };
        car.shapes
            .push(StaticShape::oriented_box(centre + up, half, rot));
    }
    car.surface_contact = opts.surface_contact;
    car.own_travel_reach = opts.own_travel_reach;
    car
}

/// The lab's World settings: the brick ring drive's solver (1/60 s steps of 4 iterations,
/// persistent contacts) for a few thousand bodies.
pub fn lab_world_def(opts: &LabOptions) -> WorldDef {
    let mut def = crate::brick_ring_drive::ring_world_def(4096);
    def.gravity = [0.0, GRAVITY, 0.0];
    // The mountain run goes down for kilometres.
    def.kill_bounds = Some(KillBounds::below(-1.0e6));
    // Eight steps' results may be in flight (3 by default): a page whose frames run
    // late, or whose readbacks arrive late, can still take every step real time owes.
    def.readback_frames = 8;
    let _ = opts;
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
    /// The scene's own bodies (the flat arena's ground slab and fixtures).
    scene_bodies: Vec<BodyId>,
    template: ExplicitBeams,
    layout: CouplingLayout,
    plan: ReadPlan,
    /// Restarts since the build.
    pub restarts: u32,
    /// The car as added (its beams, triangles and contacters: the pieces' topology).
    car: ExplicitBeams,
    /// The piece labels last sent to the GPU, and new ones to send before the next step.
    labels: Vec<u8>,
    labels_due: Option<Vec<u8>>,
}

fn deg(v: &Value, k: &str, d: f64) -> f32 {
    (num(v, k, d) as f32).to_radians()
}

impl Lab {
    /// The scene's own bodies, made once (the ground slab is the World's first collider:
    /// the page's renderer leaves it to its own ground): the flat arena's slab, which the
    /// rigid bodies stand on (the car has its own plane), and its ramp and kerb for the
    /// rigid bodies and the renderer. On the mountain they stand far away.
    fn scene_bodies(world: &mut World, opts: &LabOptions) -> anyhow::Result<Vec<BodyId>> {
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
        let mut out = Vec::new();
        let poses = Self::scene_poses(opts);
        let g = world.create_body(&BodyDef {
            position: poses[0].position,
            ..BodyDef::default()
        })?;
        out.push(g);
        world.create_box_shape(
            g,
            &ShapeDef {
                density: 0.0,
                ..fixture(0.6)
            },
            [400.0, 0.5, 400.0],
        )?;
        for (k, (_, half, _)) in arena_fixtures().into_iter().enumerate() {
            let b = world.create_body(&BodyDef {
                position: poses[k + 1].position,
                rotation: poses[k + 1].rotation,
                ..BodyDef::default()
            })?;
            out.push(b);
            world.create_box_shape(b, &fixture(0.6), half.to_array())?;
        }
        Ok(out)
    }

    /// Where the scene's bodies stand: the slab, then the fixtures.
    fn scene_poses(opts: &LabOptions) -> Vec<Transform> {
        let away = if opts.scene == Scene::Flat {
            0.0
        } else {
            2.0e5
        };
        let mut out = vec![Transform::new(
            [0.0, GROUND - 0.5 - away, 0.0],
            Quat::IDENTITY.to_array(),
        )];
        for (centre, _, rot) in arena_fixtures() {
            out.push(Transform::new(
                (centre - Vec3::new(0.0, away, 0.0)).to_array(),
                rot.to_array(),
            ));
        }
        out
    }

    fn cars(
        template: &ExplicitBeams,
        opts: &LabOptions,
        start: Option<Start>,
    ) -> Vec<ExplicitBeams> {
        let car = lab_car(template, opts, start);
        let mut cars = vec![car];
        if opts.parked {
            // Broadside across the arena, 16 m ahead of the origin (far overhead on the
            // mountain: the fleet is the same in every scene).
            let far = if opts.scene == Scene::Mountain {
                1.0e4
            } else {
                0.0
            };
            let mut p = lab_car(
                template,
                &LabOptions {
                    scene: Scene::Flat,
                    ..opts.clone()
                },
                Some(Start {
                    yaw: std::f32::consts::FRAC_PI_2,
                    at: [0.0, 16.0 + far],
                    ..Start::default()
                }),
            );
            p.ground = cars[0].ground;
            p.shapes = cars[0].shapes.clone();
            cars.push(p);
        }
        cars
    }

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
        let scene_bodies = Self::scene_bodies(&mut world, &opts)?;
        let cars = Self::cars(template, &opts, None);
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
        if let Some(x) = world.explicit_vehicles_mut() {
            x.set_piece_windows(opts.piece_windows);
        }
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
            bench: Bench::new(opts, &cars[0], dt),
            bodies: HashMap::new(),
            scene_bodies,
            template: template.clone(),
            layout,
            plan,
            restarts: 0,
            car: cars[0].clone(),
            labels: Vec::new(),
            labels_due: None,
        };
        lab.bench.begin(&mut WorldRig {
            world: &mut world,
            bodies: &mut lab.bodies,
        })?;
        Ok((world, lab))
    }

    /// Start over in place, in well under a second: no device, pipeline or buffer is
    /// made. Every lab body is removed, the car goes back to its rest shape at a new
    /// start (`World::reset_explicit_vehicles`), the scene is laid out again. `v`:
    /// `{ scene: 'flat' | 'mountain', mountain: { seed, steepness, roughness, … }, start:
    /// { kmh, angle (degrees), x, z }, crash: { kind: 'wall' | 'offset' | 'pole' |
    /// 'sidepole', kmh, overlap (0–1 of the car's width, the offset wall's), offset (m,
    /// the pole's from the car's centre line; along the car for the side pole), angle
    /// (degrees, the wall's from square), gap (m to the obstacle at the start) } }`; what
    /// is left out keeps the lab's setting (the start: the scene's own). The car's
    /// surface contact and the second car stay as built.
    pub fn restart(&mut self, world: &mut World, v: &Value) -> anyhow::Result<()> {
        let mut opts = self.bench.opts.clone();
        match v.get("scene").and_then(Value::as_str) {
            Some("mountain") => opts.scene = Scene::Mountain,
            Some("flat") => opts.scene = Scene::Flat,
            _ => {}
        }
        if v.get("mountain").is_some() {
            let mut merged = json!({ "mountain": v["mountain"] });
            merged["scene"] = json!("mountain");
            let m = LabOptions::from_json(&merged).mountain;
            opts.mountain = m;
        }
        let crash = v.get("crash").filter(|c| !c.is_null());
        anyhow::ensure!(
            crash.is_none() || opts.scene == Scene::Flat,
            "vehicle lab: the crash presets are on the flat arena"
        );
        let mut start = v.get("start").filter(|c| !c.is_null()).map(|s| Start {
            speed: num(s, "kmh", 0.0) as f32 / 3.6,
            yaw: deg(s, "angle", 0.0),
            at: [num(s, "x", 0.0) as f32, num(s, "z", 0.0) as f32],
            ..Start::default()
        });
        let mut kind = "";
        if let Some(c) = crash {
            kind = c.get("kind").and_then(Value::as_str).unwrap_or("wall");
            let speed = num(c, "kmh", 50.0) as f32 / 3.6;
            anyhow::ensure!(
                (0.0..=70.0).contains(&speed),
                "vehicle lab: crash speed out of range (0 to 250 km/h)"
            );
            start = Some(match kind {
                "wall" | "offset" | "pole" => Start {
                    speed,
                    ..Start::default()
                },
                // On a sled: carried sideways with its wheels off the ground (0.5 m up,
                // past the suspension's droop), so the tyres' grip takes nothing before
                // the pole does; it starts 0.3 m from the pole and drops as it strikes.
                "sidepole" => Start {
                    lateral: speed,
                    lift: 0.5,
                    ..Start::default()
                },
                other => anyhow::bail!(
                    "vehicle lab: unknown crash '{other}' (wall, offset, pole, sidepole)"
                ),
            });
        }
        let cars = Self::cars(&self.template, &opts, start);
        // The car first: a fleet the World was not built for is refused before anything
        // else changes.
        world.reset_explicit_vehicles(&cars)?;
        for (_, b) in self.bodies.drain() {
            if world.body_is_valid(b) {
                let _ = world.destroy_body(b);
            }
        }
        for (b, pose) in self.scene_bodies.iter().zip(Self::scene_poses(&opts)) {
            world.body_set_transform(*b, pose)?;
        }
        if let Some(x) = world.explicit_vehicles_mut() {
            x.set_drive_input(0, DriveInput::default());
        }
        let dt = self.bench.dt;
        self.bench = Bench::new(opts, &cars[0], dt);
        self.restarts += 1;
        self.car = cars[0].clone();
        self.labels_due = (!self.labels.is_empty()).then(Vec::new);
        let mut rig = WorldRig {
            world,
            bodies: &mut self.bodies,
        };
        self.bench.begin(&mut rig)?;
        if let Some(c) = crash {
            // The car's box as placed (facing +z; its left is +x).
            let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
            for n in &cars[0].nodes {
                lo = lo.min(n.pos);
                hi = hi.max(n.pos);
            }
            let gap =
                (num(c, "gap", if kind == "sidepole" { 0.3 } else { 1.0 }) as f32).clamp(0.2, 50.0);
            let (wide, cx, cz) = (hi.x - lo.x, 0.5 * (lo.x + hi.x), 0.5 * (lo.z + hi.z));
            let offset = num(c, "offset", 0.0) as f32;
            let pole = |x: f32, z: f32| {
                let (r, h) = (0.127f32, 3.0f32);
                ObjDef::fixed(
                    LabGeom::Cylinder {
                        radius: r,
                        half_len: 0.5 * h,
                    },
                    DVec3::new(f64::from(x), f64::from(GROUND + 0.5 * h), f64::from(z)),
                    Quat::IDENTITY,
                    7800.0,
                )
            };
            let def = match kind {
                "wall" | "offset" => {
                    let overlap = if kind == "offset" {
                        (num(c, "overlap", 0.4) as f32).clamp(0.05, 1.0)
                    } else {
                        1.0
                    };
                    // The offset wall covers the car's left `overlap` of its width, and 4 m
                    // beyond it; the full wall 4 m either side.
                    let (x0, x1) = if kind == "offset" {
                        (hi.x - overlap * wide, hi.x + 4.0)
                    } else {
                        (lo.x - 4.0, hi.x + 4.0)
                    };
                    let half = Vec3::new(0.5 * (x1 - x0), 1.25, 0.5);
                    let yaw = Quat::from_rotation_y(deg(c, "angle", 0.0));
                    // Its face's middle stays `gap` ahead of the nose.
                    let face = Vec3::new(0.5 * (x0 + x1), 0.0, hi.z + gap);
                    let centre = face + yaw * Vec3::new(0.0, GROUND + half.y, half.z);
                    ObjDef::fixed(LabGeom::Box { half }, centre.as_dvec3(), yaw, 2400.0)
                }
                "pole" => pole(cx + offset, hi.z + gap + 0.127),
                _ => pole(hi.x + gap + 0.127, cz + offset),
            };
            self.bench.obstacles.push(rig.spawn(&def)?);
            self.bench.notes.push(format!(
                "crash preset {kind}: {:.0} km/h",
                num(c, "kmh", 50.0)
            ));
            self.bench.target = Some(num(c, "kmh", 50.0) as f32 / 3.6);
        }
        Ok(())
    }

    pub fn read_plan(&self) -> ReadPlan {
        self.plan
    }

    /// The piece labels last sent to the GPU's pair list (empty: none, all zero).
    pub fn piece_labels(&self) -> &[u8] {
        &self.labels
    }

    pub fn command(&mut self, world: &mut World, cmd: &Value) -> anyhow::Result<Value> {
        // { ownTravel: bool }: the cars' anti-clip padding by each contacter's own travel
        // (LabOptions::own_travel_reach), now and after a restart.
        if let Some(on) = cmd.get("ownTravel").and_then(Value::as_bool) {
            self.bench.opts.own_travel_reach = on;
            if let Some(x) = world.explicit_vehicles_mut() {
                for v in 0..x.vehicle_count() {
                    x.set_own_travel_reach(v, on);
                }
            }
            return Ok(json!({ "ownTravel": on }));
        }
        // { debris: { on, seconds, maxKg, maxNodes } }: the debris time to live
        // (LabOptions::debris_ttl), now and after a restart.
        if let Some(d) = cmd.get("debris") {
            let ttl = debris_ttl_from(d);
            self.bench.set_debris_ttl(ttl);
            return Ok(json!({ "debris": debris_json(ttl) }));
        }
        // { pieceWindows: bool }: the coupling's window per piece (LabOptions::piece_windows).
        if let Some(on) = cmd.get("pieceWindows").and_then(Value::as_bool) {
            self.bench.opts.piece_windows = on;
            if let Some(x) = world.explicit_vehicles_mut() {
                x.set_piece_windows(on);
            }
            return Ok(json!({ "pieceWindows": on }));
        }
        // { pieceLabels: bool }: the pair list's piece labels (LabOptions::piece_labels).
        if let Some(on) = cmd.get("pieceLabels").and_then(Value::as_bool) {
            self.bench.opts.piece_labels = on;
            if !on {
                self.labels_due = Some(Vec::new());
            }
            return Ok(json!({ "pieceLabels": on }));
        }
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
        if let Some(retired) = self.bench.retire_due.take() {
            if let Some(x) = world.explicit_vehicles_mut() {
                if let Err(e) = x.set_retired(0, &retired) {
                    self.bench.debris_failed(&e);
                }
            }
        }
        if let Some(labels) = self.labels_due.take() {
            if let Some(x) = world.explicit_vehicles_mut() {
                let zeros = vec![0u8; self.car.nodes.len()];
                x.set_piece_labels(0, if labels.is_empty() { &zeros } else { &labels });
            }
            self.labels = labels;
        }
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
        self.ingest_at(nodes, origin, coupling, beams, None)
    }

    /// [`Self::ingest`] of a state copied at the lab's time `taken` (s; none: now).
    pub fn ingest_at(
        &mut self,
        nodes: &[u32],
        origin: &[u32],
        coupling: &[u32],
        beams: Option<&[u32]>,
        taken: Option<f64>,
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
        let taken = taken.filter(|t| t.is_finite()).unwrap_or(self.bench.t);
        self.bench.read_at(
            &CarRead {
                pos: &pos,
                vel: &vel,
                beams: beams.as_deref(),
                origin: o,
            },
            &frame,
            taken,
        );
        if let (Some(b), Some(pieces), true) = (beams.as_deref(), self.bench.pieces.as_ref(), self.bench.opts.piece_labels) {
            use engine_softbody::explicit_debris::piece_labels;
            let broken: Vec<bool> = b.iter().map(|x| x.0).collect();
            let labels = if pieces.mass.len() > 1 { piece_labels(&self.car, pieces, &broken) } else { Vec::new() };
            if labels != self.labels {
                self.labels_due = Some(labels);
            }
        }
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
