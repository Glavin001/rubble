//! The soft-body demo's scenes (`softbody.html`) placed in a [`World`]: the crash yard,
//! the crash tests (wall, pole, side pole, T-bone, head-on), rollover, ramp, drop, crates,
//! the import target, the traffic derby, limbo, sandwich and crusher, with the procedural
//! sedan, the hybrid dune buggy, the hi-res JBeam car or an imported `.truck` / JBeam. One
//! definition ([`SoftSceneDef`]) gives the world's definition ([`world_def`]) and the
//! bodies ([`place`]); the page keeps its renderers, cameras, drivers and scripts (the
//! ram, the crusher's lift, imports at run time) over what [`Placed`] returns.
//!
//! On the explicit solver (`explicit`) the vehicle runs in the page's explicit sim; the
//! world holds the ground and the sim's static shapes (`explicit_boxes`), for rendering.
//!
//! Every value is the page's: the same bodies in the same order, its math in f64 (values
//! it read from JSON as their shortest decimals) cast to f32.

use crate::*;
use engine_softbody::procedural::{box_truck, car_desc, car_truck, CarParams};
use engine_softbody::{SoftSettings, SoftVehicleDesc, SpawnOptions};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The scenes: name and title.
pub const SCENES: [(&str, &str); 15] = [
    ("yard", "Crash yard (drive)"),
    ("wall", "Crash test: rigid wall, 54 km/h"),
    ("pole", "Crash test: 20 cm pole, 40 km/h"),
    ("tbone", "Crash test: 1,500 kg side impact, 40 km/h"),
    ("headon", "Crash test: head-on, 2 × 29 km/h"),
    ("sidepole", "Crash test: side pole, 32 km/h"),
    ("rollover", "Rollover: sliding sideways into a curb"),
    ("ramp", "Corkscrew ramp (one side)"),
    ("drop", "Nose-first drop"),
    ("crates", "Lattice crates drop"),
    (
        "import",
        "Crash test: your .truck / JBeam into a wall, 40 km/h",
    ),
    (
        "traffic",
        "Traffic derby: self-driving soft cars (?cars=10…100)",
    ),
    ("limbo", "Limbo bar: low clearance shears the roof off"),
    (
        "sandwich",
        "Sandwich: 18 t truck shunts the car into a parked truck",
    ),
    ("crusher", "Crusher: 10 t block dropped on the roof"),
];

/// The page's collider material slots and their friction.
pub const MAT_GROUND: u32 = 0;
pub const MAT_PROP: u32 = 1;
pub const MAT_NODE: u32 = 9;
/// The car's half width and its bumper face ahead of its centre (m).
pub const CAR_HW: f64 = 0.85;
pub const CAR_FRONT: f64 = 2.2;

fn friction(slot: u32) -> f32 {
    match slot {
        MAT_GROUND => 1.0,
        MAT_PROP => 0.6,
        MAT_NODE => 0.8,
        s if s == engine_vehicle::pack::WHEEL_RIM_MATERIAL => 0.9,
        _ => 0.6,
    }
}

/// An imported vehicle: `.truck` text, or JBeam files (name, text) and a part config.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct ImportSource {
    pub truck: Option<String>,
    pub jbeam: Option<Vec<(String, String)>>,
    pub pc: Option<String>,
}

/// A soft scene and the page's switches.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SoftSceneDef {
    pub scene: String,
    /// `sedan`, `buggy`, `hires` or `import` (with `import`).
    pub car: String,
    pub import: Option<ImportSource>,
    /// The vehicle runs on the explicit solver (the page's sim).
    pub explicit: bool,
    /// The explicit sim's static shapes: rows [centre xyz, half xyz, rotation xyzw, kind]
    /// (missing rotation: identity; kind 0/1 box, 2 vertical cylinder, 3 capsule, 4 sphere).
    pub explicit_boxes: Vec<Vec<f64>>,
    /// The buggy's mount scale (`?strength=`; 0: 1).
    pub strength: f64,
    /// `?deform=`: the procedural car rebuilt from `.truck` text with this threshold.
    pub deform: f64,
    /// Step rate, iterations, thresholds × (`?threshold=`), particle sweeps, rigid mode.
    pub hz: f64,
    pub iters: u32,
    pub threshold: f64,
    pub sweeps: u32,
    pub rigid: bool,
    /// Crash parameters by name (`speed`, `overlap`, `angle`, `offset`, `at`, `mass`,
    /// `carSpeed`, `height`, `pitch`, `bar`, `parked`, `liftAt`, `liftTo`); absent ones
    /// take each scene's standard test.
    pub params: BTreeMap<String, f64>,
    /// Extra approach distance before a crash (m).
    pub runup: f64,
    /// Traffic: how many cars (`?cars=`, 1–128, default 20).
    pub cars: f64,
    /// The import scene's target: `wall` or `car`.
    pub target: String,
    /// Traffic drivers (`?ai=`): empty, the derby (two in three wander, one chases);
    /// `cruise`, everyone wanders and brakes for cars ahead; `0`, none.
    pub ai: String,
}

impl Default for SoftSceneDef {
    fn default() -> Self {
        Self {
            scene: "yard".into(),
            car: "sedan".into(),
            import: None,
            explicit: false,
            explicit_boxes: Vec::new(),
            strength: 0.0,
            deform: 0.0,
            hz: 240.0,
            iters: 10,
            threshold: 0.0,
            sweeps: 0,
            rigid: true,
            params: BTreeMap::new(),
            runup: 0.0,
            cars: 0.0,
            target: "wall".into(),
            ai: String::new(),
        }
    }
}

impl SoftSceneDef {
    fn known(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            SCENES.iter().any(|(s, _)| *s == self.scene),
            "unknown soft scene {}",
            self.scene
        );
        anyhow::ensure!(
            ["sedan", "buggy", "hires", "import"].contains(&self.car.as_str()),
            "unknown car {}",
            self.car
        );
        anyhow::ensure!(
            self.car != "import" || self.import.is_some(),
            "car import needs its source"
        );
        Ok(())
    }

    fn p(&self, k: &str, d: f64) -> f64 {
        self.params
            .get(k)
            .copied()
            .filter(|v| v.is_finite())
            .unwrap_or(d)
    }

    /// The hybrid buggy (soft frame, rigid parts) drives the scene.
    pub fn buggy(&self) -> bool {
        self.car == "buggy" && !self.explicit
    }

    /// Traffic's car count (0 elsewhere).
    pub fn ncars(&self) -> u32 {
        if self.scene != "traffic" {
            return 0;
        }
        let n = if self.cars != 0.0 && !self.cars.is_nan() {
            self.cars
        } else {
            20.0
        };
        n.clamp(1.0, 128.0) as u32
    }

    /// The scene's vehicle (none on the explicit solver).
    pub fn car_desc(&self) -> anyhow::Result<Option<SoftVehicleDesc>> {
        if self.explicit {
            return Ok(None);
        }
        Ok(Some(match self.car.as_str() {
            "buggy" => {
                let s = if self.strength != 0.0 && !self.strength.is_nan() {
                    self.strength
                } else {
                    1.0
                };
                let k = engine_softbody::buggy::BuggyBeams {
                    mount_scale: s.max(0.01) as f32,
                    ..Default::default()
                };
                engine_softbody::buggy::dune_buggy_soft_with(&k)
            }
            "hires" => engine_softbody::jbeam::load_jbeam(&[("hires.jbeam", &hires_jbeam())], None)
                .map_err(|e| anyhow::anyhow!("hires car: {e}"))?,
            "import" => import_desc(self.import.as_ref().expect("checked"))?,
            _ if self.deform != 0.0 && !self.deform.is_nan() => {
                let d = CarParams::default();
                let deform = self.deform as f32;
                engine_softbody::truck::load_truck(&car_truck(&CarParams {
                    deform: if deform > 0.0 { deform } else { d.deform },
                    ..d
                }))
            }
            _ => car_desc(&CarParams::default()).0,
        }))
    }
}

/// The high-resolution test car as JBeam text (the explicit solver's tuning, as the page
/// loads it for both solvers).
pub fn hires_jbeam() -> String {
    use engine_softbody::hires::{hires_car_jbeam, HiresParams};
    hires_car_jbeam(&HiresParams {
        node_weight: 2.0,
        spring: 1.5e6,
        damp: 150.0,
        stiffness_length: 0.15,
        ..Default::default()
    })
    .0
}

fn import_desc(src: &ImportSource) -> anyhow::Result<SoftVehicleDesc> {
    if let Some(files) = &src.jbeam {
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        return engine_softbody::jbeam::load_jbeam(&refs, src.pc.as_deref())
            .map_err(|e| anyhow::anyhow!("JBeam import: {e}"));
    }
    let text = src
        .truck
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("import: neither .truck text nor JBeam files"))?;
    Ok(engine_softbody::truck::load_truck(text))
}

/// The yard's and the crate scene's braced lattice crate.
fn crate_desc() -> SoftVehicleDesc {
    engine_softbody::truck::load_truck(&box_truck(3, 3, 3, 0.45, 120.0, 1.5e6, 6e3, 1e5))
}

fn pow2(n: u64) -> u32 {
    n.max(1).next_power_of_two() as u32
}

/// The world a scene needs: its capacity (from the car loaded and the traffic count),
/// readback depth for its steps per frame, the vehicle and soft pools.
pub fn world_def(def: &SoftSceneDef) -> anyhow::Result<WorldDef> {
    def.known()?;
    let steps = (def.hz / 60.0).round().max(1.0) as u32;
    let ncars = def.ncars() as u64;
    let big = ncars.max(24);
    // Per-car soft capacity from the car actually loaded (an imported or hi-res car can
    // have several times the procedural sedan's beams), at least the sedan's.
    let size = if !def.explicit && (def.car == "import" || def.car == "hires") {
        let d = def.car_desc()?.expect("not explicit");
        let groups: std::collections::BTreeSet<u32> = d
            .beams
            .iter()
            .map(|b| b.break_group)
            .filter(|g| *g != 0)
            .collect();
        Some((
            d.nodes.len() as u64,
            d.beams.len() as u64,
            d.triangles.len() as u64,
            groups.len() as u64,
        ))
    } else {
        None
    };
    let (particles, beams, triangles, groups) = size.unwrap_or((0, 0, 0, 0));
    let per_beams = 1000.max(beams + 64);
    let per_particles = 170.max(particles + 16);
    let per_triangles = 400.max(triangles + 16);
    let est = 200.max(particles + 32) * big + 2000;
    let max_bodies = pow2(est.max(4096));
    let capacity = WorldCapacity {
        max_bodies,
        max_colliders: pow2(
            (220.max(particles + 48) * big + 4000)
                .max(max_bodies as u64 * 2)
                .max(8192),
        ),
        max_contacts: pow2((est * 16).max(65536)).min(1 << 21),
        max_joints: (512 + if def.buggy() { 48 } else { 16 } * big) as u32,
        max_candidates: pow2((est * 32).max(131072)).min(1 << 21),
        max_colors: 32,
        max_commands: 2048,
        max_hull_vertices: 64,
    };
    Ok(WorldDef {
        gravity: [0.0, -9.81, 0.0],
        contacts: ContactStorage::Canonical,
        capacity,
        readback_frames: 2 * steps + 2,
        vehicles: VehicleCapacity {
            max_vehicles: (big + 8) as u32,
            max_wheels: (4 * big + 32) as u32,
            max_curve_points: 4096,
            max_diffs: 256,
            max_anti_rolls: 256,
            max_parts: 512,
            max_rays: 16,
        },
        soft: Some(SoftCapacity {
            max_bodies: (big + 16) as u32,
            max_beams: (per_beams * big + 8192) as u32,
            max_particles: (per_particles * big + 2048) as u32,
            max_colors: 64,
            max_groups: 1024.max(groups * big + 64) as u32,
            max_triangles: (per_triangles * big + 4096) as u32,
            max_obstacles: 64,
            ..SoftCapacity::default()
        }),
        ..WorldDef::default()
    })
}

/// A soft body of the scene: its handle, whether the procedural car's mesh or the buggy's
/// draws it.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PlacedSoft {
    pub h: u32,
    pub car: bool,
    pub buggy: bool,
}

/// A rigid body a camera can follow: its name, slot and where the camera rides on it.
#[derive(Clone, Debug, Serialize)]
pub struct Subject {
    pub name: String,
    pub body: u32,
    pub mount: [f32; 3],
}

/// A dynamic box: its handle, shape and slot.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DynBox {
    pub body: u64,
    pub shape: u64,
    pub slot: u32,
}

/// The crusher's block and its scripted lift.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Crusher {
    pub body: u64,
    pub slot: u32,
    pub lift_at: f64,
    pub lift_to: f64,
    pub speed: f64,
}

/// What the page needs of a placed scene.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Placed {
    pub softs: Vec<PlacedSoft>,
    pub subjects: Vec<Subject>,
    /// The parked side ram (yard, T-bone), its speed (m/s) and where along the car it
    /// strikes; `pending_ram`: steps until the T-bone launches it (−1: never).
    pub ram: Option<DynBox>,
    pub ram_speed: f64,
    pub ram_at: f64,
    pub pending_ram: i32,
    pub crusher: Option<Crusher>,
    /// Parked cannonballs (static until launched).
    pub cannon: Vec<u64>,
    /// The scene's vehicle when the page needs it later (the buggy's renderer, an
    /// imported car crashed into the import scene's target).
    pub car_desc: Option<SoftVehicleDesc>,
}

const IDENT: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

fn f3(v: [f64; 3]) -> [f32; 3] {
    v.map(|x| x as f32)
}

fn f4(v: [f64; 4]) -> [f32; 4] {
    v.map(|x| x as f32)
}

fn q_axis(ax: [f64; 3], ang: f64) -> [f64; 4] {
    let s = (ang / 2.0).sin();
    [ax[0] * s, ax[1] * s, ax[2] * s, (ang / 2.0).cos()]
}

/// An f32 the page read from JSON, as the page holds it: its shortest decimal as an f64.
fn dec(x: f32) -> f64 {
    x.to_string().parse().expect("an f32 prints as a number")
}

/// `Math.hypot` of two numbers as browsers compute it (scaled, compensated sum).
fn js_hypot2(a: f64, b: f64) -> f64 {
    let max = a.abs().max(b.abs());
    if max == 0.0 {
        return 0.0;
    }
    let (mut sum, mut compensation) = (0.0f64, 0.0f64);
    for x in [a, b] {
        let n = x.abs() / max;
        let summand = n * n - compensation;
        let preliminary = sum + summand;
        compensation = (preliminary - sum) - summand;
        sum = preliminary;
    }
    sum.sqrt() * max
}

/// A desc's forward in its own frame (as `engine_softbody::pack::forward_axis`, in the
/// page's arithmetic): the reference nodes, else the steering axle, else −X like most
/// Rigs of Rods trucks.
fn forward_of(d: &SoftVehicleDesc) -> [f64; 3] {
    let p = |i: u32| d.nodes[i as usize].position.map(dec);
    let flat = |v: [f64; 3]| {
        let l = js_hypot2(v[0], v[2]);
        (l > 1e-4).then(|| [v[0] / l, 0.0, v[2] / l])
    };
    if let Some(r) = &d.refs {
        let (c, b) = (p(r.center), p(r.back));
        if let Some(f) = flat([c[0] - b[0], c[1] - b[1], c[2] - b[2]]) {
            return f;
        }
    }
    let mid = |w: &engine_softbody::SoftWheel| {
        let (a, b) = (p(w.node1), p(w.node2));
        [
            0.5 * (a[0] + b[0]),
            0.5 * (a[1] + b[1]),
            0.5 * (a[2] + b[2]),
        ]
    };
    let mean = |ws: &[&engine_softbody::SoftWheel]| {
        let n = ws.len() as f64;
        ws.iter().map(|w| mid(w)).fold([0.0; 3], |a, b| {
            [a[0] + b[0] / n, a[1] + b[1] / n, a[2] + b[2] / n]
        })
    };
    let st: Vec<_> = d.wheels.iter().filter(|w| w.steerable).collect();
    let other: Vec<_> = d.wheels.iter().filter(|w| !w.steerable).collect();
    if !st.is_empty() && !other.is_empty() {
        let (a, b) = (mean(&st), mean(&other));
        if let Some(f) = flat([a[0] - b[0], a[1] - b[1], a[2] - b[2]]) {
            return f;
        }
    }
    [-1.0, 0.0, 0.0]
}

/// The page's placement helpers over a world.
struct Builder<'w> {
    world: &'w mut World,
    out: Placed,
    car: Option<SoftVehicleDesc>,
    /// The car is turned and set down by its nodes (hi-res car, imports).
    fit: bool,
    sedan: bool,
    buggy: bool,
}

impl Builder<'_> {
    fn shape_on(&mut self, body: BodyId, geometry: Geometry, slot: u32) -> anyhow::Result<ShapeId> {
        let def = ShapeDef {
            material: SurfaceMaterial {
                friction: friction(slot),
                ..SurfaceMaterial::default()
            },
            ..ShapeDef::default()
        };
        self.world.create_shape(body, &def, geometry)
    }

    fn cuboid(half: [f64; 3]) -> Geometry {
        Geometry::Box {
            half: f3(half),
            transform: Transform::IDENTITY,
        }
    }

    /// A static box; every one but the ground is a soft obstacle (skin triangles collide
    /// with it too, so a wall's edge cannot slip between a panel's nodes).
    fn static_box(
        &mut self,
        p: [f64; 3],
        half: [f64; 3],
        q: [f64; 4],
        slot: u32,
    ) -> anyhow::Result<BodyId> {
        let body = self.world.create_body(&BodyDef {
            position: f3(p),
            rotation: f4(q),
            ..BodyDef::default()
        })?;
        let s = self.shape_on(body, Self::cuboid(half), slot)?;
        if slot != MAT_GROUND {
            self.world.soft_add_obstacle(s)?;
        }
        Ok(body)
    }

    fn dyn_box(
        &mut self,
        p: [f64; 3],
        half: [f64; 3],
        mass: f64,
        v: [f64; 3],
    ) -> anyhow::Result<DynBox> {
        let body = self.world.create_body(&BodyDef {
            body_type: BodyType::Dynamic,
            position: f3(p),
            linear_velocity: f3(v),
            ..BodyDef::default()
        })?;
        let shape = self.shape_on(body, Self::cuboid(half), MAT_PROP)?;
        let i = [
            (mass * (half[1] * half[1] + half[2] * half[2])) / 3.0,
            (mass * (half[0] * half[0] + half[2] * half[2])) / 3.0,
            (mass * (half[0] * half[0] + half[1] * half[1])) / 3.0,
        ];
        self.world
            .body_set_mass_data(body, MassData::principal(mass as f32, [0.0; 3], f3(i)))?;
        Ok(DynBox {
            body: body.to_bits(),
            shape: shape.to_bits(),
            slot: body.index(),
        })
    }

    /// A static pole (capsule along Y) that soft triangles also collide with.
    fn pole(&mut self, p: [f64; 3], r: f64, half: f64) -> anyhow::Result<()> {
        let body = self.world.create_body(&BodyDef {
            position: f3(p),
            ..BodyDef::default()
        })?;
        let g = Geometry::Capsule {
            center: [0.0; 3],
            radius: r as f32,
            half_len: half as f32,
        };
        let s = self.shape_on(body, g, MAT_PROP)?;
        self.world.soft_add_obstacle(s)
    }

    fn subject(&mut self, name: &str, slot: u32, mount: [f32; 3]) {
        self.out.subjects.push(Subject {
            name: name.into(),
            body: slot,
            mount,
        });
    }

    fn park_ram(&mut self, mass: f64) -> anyhow::Result<()> {
        let ram = self.dyn_box([-4000.0; 3], [0.6, 0.6, 0.9], mass, [0.0; 3])?;
        self.subject("side ram", ram.slot, [0.0, 1.0, 0.0]);
        // Its edges would slip between the nodes: the skin collides with it too.
        self.world
            .soft_add_obstacle(ShapeId::from_bits(ram.shape))?;
        self.out.ram = Some(ram);
        Ok(())
    }

    /// The scene's car at `p`, turned by `q`, moving at `v`.
    fn car_at(&mut self, p: [f64; 3], q: [f64; 4], v: [f64; 3]) -> anyhow::Result<u32> {
        let desc = self.car.clone().expect("a soft scene's car");
        let (mut p, mut q) = (p, q);
        if self.fit {
            // Turned so its forward axis is the scene's +Z, its front face centred where
            // the sedan's would be (CAR_FRONT ahead of the spawn point), its lowest node
            // 10 cm up.
            let f = forward_of(&desc);
            let fix = q_axis([0.0, 1.0, 0.0], -f[0].atan2(f[2]));
            let (a, b) = (q, fix);
            q = [
                a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
                a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
                a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
            ];
            let rot = |v: [f64; 3]| {
                let c = [
                    q[1] * v[2] - q[2] * v[1],
                    q[2] * v[0] - q[0] * v[2],
                    q[0] * v[1] - q[1] * v[0],
                ];
                let d = 2.0 * (q[0] * v[0] + q[1] * v[1] + q[2] * v[2]);
                let w2 = q[3] * q[3] - (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
                [0, 1, 2].map(|i| q[i] * d + v[i] * w2 + c[i] * 2.0 * q[3])
            };
            let ps: Vec<[f64; 3]> = desc
                .nodes
                .iter()
                .map(|n| rot(n.position.map(dec)))
                .collect();
            let fw = rot(f);
            let l = [fw[2], 0.0, -fw[0]];
            let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            let along = ps
                .iter()
                .map(|r| dot(*r, fw))
                .fold(f64::NEG_INFINITY, f64::max);
            let lats: Vec<f64> = ps.iter().map(|r| dot(*r, l)).collect();
            let lo = lats.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = lats.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mid = (lo + hi) / 2.0;
            let low = ps.iter().map(|r| r[1]).fold(f64::INFINITY, f64::min);
            p = [0, 1, 2].map(|i| {
                if i == 1 {
                    p[1] + 0.05 - low
                } else {
                    p[i] + fw[i] * (CAR_FRONT - along) - l[i] * mid
                }
            });
        }
        let opts = SpawnOptions {
            velocity: f3(v),
            ..SpawnOptions::default()
        };
        let h = self.world.add_soft_vehicle(&desc, f3(p), f4(q), &opts)?.0;
        self.out.softs.push(PlacedSoft {
            h,
            car: self.sedan,
            buggy: self.buggy,
        });
        Ok(h)
    }

    fn car(&mut self, p: [f64; 3], yaw: f64, v: [f64; 3]) -> anyhow::Result<u32> {
        self.car_at(p, q_axis([0.0, 1.0, 0.0], yaw), v)
    }

    /// A lattice crate (no rigid core).
    fn lattice_crate(&mut self, p: [f64; 3]) -> anyhow::Result<()> {
        let opts = SpawnOptions {
            rigid_core: false,
            ..SpawnOptions::default()
        };
        let q = q_axis([0.0, 1.0, 0.0], 0.0);
        let h = self
            .world
            .add_soft_vehicle(&crate_desc(), f3(p), f4(q), &opts)?
            .0;
        self.out.softs.push(PlacedSoft {
            h,
            car: false,
            buggy: false,
        });
        Ok(())
    }
}

/// Place scene `def` in `world` (made from [`world_def`], empty): its solver, material
/// slots and soft settings, the ground, the scene and the parked cannonballs.
pub fn place(world: &mut World, def: &SoftSceneDef) -> anyhow::Result<Placed> {
    def.known()?;
    anyhow::ensure!(
        world.body_count() == 0,
        "a soft scene is placed in an empty world"
    );
    let mut solver = world.solver();
    solver.dt = (1.0 / def.hz) as f32;
    solver.iterations = def.iters;
    solver.gravity_y = -9.81;
    solver.beta_lin = 1e7;
    world.set_solver(solver);
    for slot in 0..64 {
        world.set_material_slot(
            slot,
            SurfaceMaterial {
                friction: friction(slot),
                ..SurfaceMaterial::default()
            },
        )?;
    }
    // Deform/break thresholds scaled for the step rate (?threshold=1 keeps the file's raw
    // values); ?sweeps=N particle sweeps per iteration; rigid mode unless ?rigid=0.
    let calibrated = SoftSettings::calibrated_for(def.hz as f32);
    world.set_soft_settings(SoftSettings {
        threshold_scale: if def.threshold != 0.0 && !def.threshold.is_nan() {
            def.threshold as f32
        } else {
            calibrated.threshold_scale
        },
        sweeps: if def.sweeps != 0 {
            def.sweeps
        } else {
            calibrated.sweeps
        },
        rigid_mode: def.rigid,
        ..calibrated
    })?;
    let car = def.car_desc()?;
    let mut b = Builder {
        world,
        out: Placed {
            softs: Vec::new(),
            subjects: Vec::new(),
            ram: None,
            ram_speed: 11.1,
            ram_at: 0.0,
            pending_ram: -1,
            crusher: None,
            cannon: Vec::new(),
            car_desc: None,
        },
        car,
        fit: def.car == "hires" || def.car == "import",
        sedan: def.car == "sedan",
        buggy: def.buggy(),
    };
    b.static_box([0.0, -5.0, 0.0], [400.0, 5.0, 400.0], IDENT, MAT_GROUND)?;
    let runup = def.runup;
    let pi = std::f64::consts::PI;
    let p = |k: &str, d: f64| def.p(k, d);
    if def.explicit {
        // The explicit sim's static shapes, for rendering (it collides with its own).
        for row in &def.explicit_boxes {
            let g = |i: usize, d: f64| row.get(i).copied().unwrap_or(d);
            let (hx, hy, hz) = (g(3, 0.0), g(4, 0.0), g(5, 0.0));
            let geometry = match g(10, 0.0) as u32 {
                2 => {
                    let pts: Vec<glam::Vec3> = (0..32u32)
                        .map(|i| {
                            let a = (i >> 1) as f64 * pi / 8.0;
                            let y = if i & 1 != 0 { hy } else { -hy };
                            glam::Vec3::from_array(f3([hx * a.cos(), y, hx * a.sin()]))
                        })
                        .collect();
                    Geometry::Hull {
                        hull: ConvexHull::from_points(&pts)
                            .map_err(|e| anyhow::anyhow!("hull: {e:?}"))?,
                        transform: Transform::IDENTITY,
                    }
                }
                3 => Geometry::Capsule {
                    center: [0.0; 3],
                    radius: hx as f32,
                    half_len: hy as f32,
                },
                4 => Geometry::Sphere {
                    center: [0.0; 3],
                    radius: hx as f32,
                },
                _ => Builder::cuboid([hx, hy, hz]),
            };
            let body = b.world.create_body(&BodyDef {
                position: f3([g(0, 0.0), g(1, 0.0), g(2, 0.0)]),
                rotation: f4([g(6, 0.0), g(7, 0.0), g(8, 0.0), g(9, 1.0)]),
                ..BodyDef::default()
            })?;
            b.shape_on(body, geometry, MAT_PROP)?;
        }
    } else {
        match def.scene.as_str() {
            "yard" => {
                b.car([0.0, 0.05, 0.0], 0.0, [0.0; 3])?;
                // A wall, a row of poles, parked soft cars, crates, a ramp.
                b.static_box([0.0, 1.5, 60.0], [8.0, 1.5, 0.3], IDENT, MAT_PROP)?;
                for i in 0..5 {
                    b.pole([-14.0 + i as f64 * 3.5, 1.5, 34.0], 0.1, 1.5)?;
                }
                b.car([14.0, 0.05, 20.0], pi / 2.0, [0.0; 3])?;
                b.car([14.0, 0.05, 30.0], -pi / 2.0, [0.0; 3])?;
                b.static_box(
                    [-18.0, 0.5, 10.0],
                    [3.0, 0.3, 4.0],
                    q_axis([1.0, 0.0, 0.0], -0.25),
                    MAT_PROP,
                )?;
                for i in 0..3 {
                    b.lattice_crate([-9.0 + i as f64 * 1.6, 0.1, 22.0])?;
                }
                b.park_ram(1500.0)?;
            }
            "wall" => {
                // Rigid wall ahead of the bumper; overlap: share of the car's width it
                // covers from the left (+X); angle: the wall's yaw (oblique impact).
                let overlap = p("overlap", 1.0);
                let angle = p("angle", 0.0) * pi / 180.0;
                let edge = CAR_HW - overlap * 2.0 * CAR_HW;
                b.static_box(
                    [edge + 3.0, 1.5, 3.0],
                    [3.0, 1.5, 0.2],
                    q_axis([0.0, 1.0, 0.0], angle),
                    MAT_PROP,
                )?;
                b.car(
                    [0.0, 0.05, -2.0 - runup],
                    0.0,
                    [0.0, 0.0, p("speed", 54.0) / 3.6],
                )?;
            }
            "pole" => {
                // A 20 cm pole; offset: lateral offset from the centre line (+ = left).
                b.pole([p("offset", 0.0), 1.5, 2.8], 0.1, 1.5)?;
                b.car(
                    [0.0, 0.05, -2.0 - runup],
                    0.0,
                    [0.0, 0.0, p("speed", 40.0) / 3.6],
                )?;
            }
            "sidepole" => {
                // Sliding sideways (right side first) into a pole at the driver's door;
                // at: the pole's position along the car.
                b.pole([-(CAR_HW + 0.5 + runup), 1.5, p("at", 0.3)], 0.125, 1.5)?;
                b.car([0.0, 0.05, 0.0], 0.0, [-p("speed", 32.0) / 3.6, 0.0, 0.0])?;
            }
            "tbone" => {
                // A block of `mass` kg at `speed` into the left side, `at` m forward of the
                // car's centre; carSpeed: the struck car's own speed.
                let car_speed = p("carSpeed", 0.0);
                b.car(
                    [0.0, 0.05, -car_speed / 3.6 * 0.25],
                    0.0,
                    [0.0, 0.0, car_speed / 3.6],
                )?;
                b.park_ram(p("mass", 1500.0))?;
                b.out.ram_speed = p("speed", 40.0) / 3.6;
                b.out.ram_at = p("at", 0.0);
                b.out.pending_ram = 60;
            }
            "headon" => {
                // speed: each car's; overlap: share of the width that meets.
                let v = p("speed", 29.0) / 3.6;
                let dx = 2.0 * CAR_HW * (1.0 - p("overlap", 1.0)) + 0.3 * p("overlap", 1.0);
                b.car([0.0, 0.05, -3.0 - runup / 2.0], 0.0, [0.0, 0.0, v])?;
                b.car([dx, 0.05, 3.0 + runup / 2.0], pi, [0.0, 0.0, -v])?;
            }
            "rollover" => {
                // Sliding sideways into a 25 cm curb: the tyres and the curb trip it.
                b.static_box(
                    [-(CAR_HW + 1.2 + runup) - 0.3, 0.125, 0.0],
                    [0.3, 0.125, 4.0],
                    IDENT,
                    MAT_PROP,
                )?;
                b.car([0.0, 0.05, 0.0], 0.0, [-p("speed", 50.0) / 3.6, 0.0, 0.0])?;
            }
            "ramp" => {
                // A ramp under the left wheels only: one side launches, the car corkscrews.
                let ang = (1.1f64 / 4.0).atan();
                let y = 2.0 * ang.sin() - 0.2 * ang.cos() + 0.02;
                b.static_box(
                    [0.55, y, CAR_FRONT + 5.0 + 2.0 + runup],
                    [0.45, 0.2, 2.0],
                    q_axis([1.0, 0.0, 0.0], -ang),
                    MAT_PROP,
                )?;
                b.car([0.0, 0.05, 0.0], 0.0, [0.0, 0.0, p("speed", 70.0) / 3.6])?;
            }
            "drop" => {
                // Dropped from `height` m, nose down by `pitch` degrees.
                b.car_at(
                    [0.0, p("height", 8.0), 0.0],
                    q_axis([1.0, 0.0, 0.0], p("pitch", 35.0) * pi / 180.0),
                    [0.0; 3],
                )?;
            }
            "limbo" => {
                // A fixed bar across the road (underside at `bar` m): past the pillars'
                // strength the roof shears off.
                let y = p("bar", 1.2) + 0.1;
                b.static_box([0.0, y, 6.0], [3.2, 0.1, 0.1], IDENT, MAT_PROP)?;
                for x in [-3.3, 3.3] {
                    b.static_box([x, y / 2.0, 6.0], [0.12, y / 2.0, 0.12], IDENT, MAT_PROP)?;
                }
                b.car(
                    [0.0, 0.05, -2.0 - runup],
                    0.0,
                    [0.0, 0.0, p("speed", 80.0) / 3.6],
                )?;
            }
            "sandwich" => {
                // The car waits behind a parked 15 t truck; an 18 t truck at `speed` shunts
                // it into the parked one.
                let half = [1.25, 1.6, 6.0];
                let parked = b.dyn_box(
                    [0.0, 1.75, CAR_FRONT + 0.4 + half[2]],
                    half,
                    p("parked", 15000.0),
                    [0.0; 3],
                )?;
                b.subject("parked truck", parked.slot, [0.0, 1.2, -6.3]);
                b.world
                    .soft_add_obstacle(ShapeId::from_bits(parked.shape))?;
                b.car([0.0, 0.05, 0.0], 0.0, [0.0; 3])?;
                let truck = b.dyn_box(
                    [0.0, 1.75, -CAR_FRONT - 3.0 - half[2] - runup],
                    half,
                    p("mass", 18000.0),
                    [0.0, 0.0, p("speed", 60.0) / 3.6],
                )?;
                b.subject("18 t truck", truck.slot, [0.0, 1.3, 6.4]);
                b.world.soft_add_obstacle(ShapeId::from_bits(truck.shape))?;
            }
            "crusher" => {
                // A `mass` kg block dropped from `height` m onto the roof.
                let block = b.dyn_box(
                    [0.0, p("height", 4.0) + 1.5, 0.0],
                    [1.2, 0.5, 1.5],
                    p("mass", 10000.0),
                    [0.0; 3],
                )?;
                b.subject("crusher block", block.slot, [0.0, -1.2, 2.5]);
                b.out.crusher = Some(Crusher {
                    body: block.body,
                    slot: block.slot,
                    lift_at: p("liftAt", 2.5),
                    lift_to: p("liftTo", 4.5),
                    speed: 1.2,
                });
                b.world.soft_add_obstacle(ShapeId::from_bits(block.shape))?;
                b.car([0.0, 0.05, 0.0], 0.0, [0.0; 3])?;
            }
            "traffic" => {
                // A walled 140 m arena with concrete blocks and poles; the cars drive
                // themselves.
                let a = 70.0;
                for (x, z, hx, hz) in [
                    (0.0, a, a, 0.5),
                    (0.0, -a, a, 0.5),
                    (a, 0.0, 0.5, a),
                    (-a, 0.0, 0.5, a),
                ] {
                    b.static_box([x, 1.0, z], [hx, 1.0, hz], IDENT, MAT_PROP)?;
                }
                for i in 0..10 {
                    let (a, r) = (i as f64 * 2.39996, 18.0 + (i % 4) as f64 * 11.0);
                    b.static_box(
                        [a.cos() * r, 0.5, a.sin() * r],
                        [1.5, 0.5, 0.5],
                        q_axis([0.0, 1.0, 0.0], a),
                        MAT_PROP,
                    )?;
                }
                for i in 0..8 {
                    let a = i as f64 * 0.785 + 0.4;
                    b.pole([a.cos() * 32.0, 1.5, a.sin() * 32.0], 0.15, 1.5)?;
                }
                let n = def.ncars();
                let cols = (n as f64).sqrt().ceil() as u32;
                let c = (cols as f64 - 1.0) / 2.0;
                for i in 0..n {
                    let x = ((i % cols) as f64 - c) * 9.0;
                    let z = ((i / cols) as f64 - c) * 11.0;
                    b.car([x, 0.05, z], (i as f64 * 1.7) % 6.283, [0.0; 3])?;
                }
            }
            "import" => {
                // The target for an imported vehicle (crashed in at run time): a rigid
                // wall across z = 12, or a procedural car parked side-on there.
                if def.target == "car" {
                    b.car([0.0, 0.05, 13.0], pi / 2.0, [0.0; 3])?;
                } else {
                    b.static_box([0.0, 2.5, 12.3], [6.0, 2.5, 0.3], IDENT, MAT_PROP)?;
                }
            }
            _ => {
                // crates
                for i in 0..9u32 {
                    let p = [
                        (i % 3) as f64 * 1.4 - 1.4,
                        1.0 + (i / 3) as f64 * 1.5,
                        (i % 2) as f64 * 0.3,
                    ];
                    b.lattice_crate(p)?;
                }
            }
        }
    }
    // Cannonballs: parked static spheres (their 300 kg mass set first), launched by a
    // type change. The grid is sized for them as moving colliders.
    for i in 0..6 {
        let pos = [-3000.0 - i as f64 * 5.0, -3000.0, -3000.0];
        let body = b.world.create_body(&BodyDef {
            body_type: BodyType::Dynamic,
            position: f3(pos),
            ..BodyDef::default()
        })?;
        let g = Geometry::Sphere {
            center: [0.0; 3],
            radius: 0.5,
        };
        let s = b.shape_on(body, g, MAT_PROP)?;
        let i = (0.4f64 * 300.0 * 0.5 * 0.5) as f32;
        b.world
            .body_set_mass_data(body, MassData::principal(300.0, [0.0; 3], [i; 3]))?;
        let t = Transform::new(f3(pos), [0.0, 0.0, 0.0, 1.0]);
        b.world
            .body_set_type(body, BodyType::Static, t, [0.0; 3], [0.0; 3])?;
        b.world.soft_add_obstacle(s)?;
        b.out.cannon.push(body.to_bits());
    }
    b.world.reserve_moving_radius(0.5);
    if b.buggy || (def.scene == "import" && def.car == "import") {
        b.out.car_desc = b.car.take();
    }
    Ok(b.out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(scene: &str, car: &str) -> SoftSceneDef {
        SoftSceneDef {
            scene: scene.into(),
            car: car.into(),
            ..SoftSceneDef::default()
        }
    }

    #[test]
    fn traffic_counts_are_clamped() {
        let mut d = def("traffic", "sedan");
        assert_eq!(d.ncars(), 20);
        d.cars = 500.0;
        assert_eq!(d.ncars(), 128);
        assert_eq!(def("wall", "sedan").ncars(), 0);
    }

    #[test]
    fn the_hires_car_sizes_its_soft_pool() {
        let sedan = world_def(&def("wall", "sedan")).unwrap();
        let hires = world_def(&def("wall", "hires")).unwrap();
        let (s, h) = (sedan.soft.unwrap(), hires.soft.unwrap());
        assert!(h.max_particles > s.max_particles && h.max_beams > s.max_beams);
        assert_eq!(sedan.readback_frames, 2 * 4 + 2);
    }

    #[test]
    fn unknown_scenes_and_sourceless_imports_fail() {
        assert!(world_def(&def("moon", "sedan")).is_err());
        assert!(world_def(&def("wall", "import")).is_err());
        assert!(def("wall", "sedan").car_desc().unwrap().is_some());
        let explicit = SoftSceneDef {
            explicit: true,
            ..def("wall", "sedan")
        };
        assert!(explicit.car_desc().unwrap().is_none());
    }
}
