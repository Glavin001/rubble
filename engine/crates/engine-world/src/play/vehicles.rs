//! The vehicle demo's scenes (`vehicle.html`) placed in a [`World`]: the playground and
//! destruction yard, traffic, the two-way coupling labs (slab, seesaw, carry, bridge, tank
//! lab, side wall), the dune buggies and vibe-land's garage, the meteor range, the gauntlet
//! and the meteor run. One definition ([`VehicleSceneDef`]) gives the world's definition
//! ([`world_def`]) and the bodies ([`place`]); the page keeps its input, camera and
//! gameplay over what [`Placed`] returns.
//!
//! Every value is the page's: the same bodies in the same order, so a host placing a scene
//! here builds the world the page built.

use crate::*;
use engine_vehicle::{presets, MountDesc};
use serde::{Deserialize, Serialize};

/// The page's collider kinds (its colours; the GPU slot holds the friction).
pub const MAT_GROUND: u32 = 0;
pub const MAT_PROP: u32 = 1;
pub const MAT_ICE: u32 = 2;
pub const MAT_SKID: u32 = 3;
pub const MAT_PAD: u32 = 4;
pub const MAT_HIT: u32 = 5;
pub const MAT_MISS: u32 = 6;
pub const MAT_METEOR: u32 = 10;
pub const MAT_TRAIL: u32 = 11;
pub const MAT_FINISH: u32 = 12;
/// The hybrid buggy's node spheres' material slot.
const MAT_SOFT_NODE: u32 = 9;
/// The soft-body step rate of the hybrid buggies (240 Hz, four steps a 60 Hz frame).
pub const SOFT_HZ: f32 = 240.0;

/// Every scene, in the page's menu order, with its title.
pub const SCENES: [(&str, &str); 14] = [
    ("playground", "Playground: ramps, stacks, mixed vehicles"),
    ("destruction", "Destruction yard: sever parts"),
    ("traffic", "Traffic: many AI vehicles"),
    ("slab", "Lab: car on a slab sliding on ice"),
    ("seesaw", "Lab: seesaw plank"),
    ("carry", "Lab: kart riding on a truck"),
    ("sidewall", "Lab: crates beside the tyres"),
    ("bridge", "Lab: plank bridge with breakable bolts"),
    ("tanklab", "Lab: tank over rubble, logs and a ledge"),
    (
        "buggies",
        "Dune buggies: jumps, walls, cannon (parts break off under load)",
    ),
    (
        "vibeland",
        "vibe-land garage: all seven, jumps, walls, cannon",
    ),
    (
        "gauntlet",
        "Gauntlet: drive a shelled street (directed debris)",
    ),
    (
        "meteors",
        "Meteor range: steered vs unsteered lanes (directed debris)",
    ),
    (
        "run",
        "Meteor gauntlet: the game (meteors + directed debris)",
    ),
];

/// A facade block or building core of a street layout.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct LayoutBlock {
    pub p: [f64; 3],
    pub half: [f64; 3],
    pub mass: f64,
}

/// A parked car at the kerb.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct LayoutParked {
    pub p: [f64; 3],
    pub yaw: f64,
}

/// The meteor run's meteors.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct LayoutMeteor {
    pub count: u32,
    pub radius: f64,
    pub mass: f64,
}

/// Layout numbers are f64: the page held exactly what the JSON gave it (the gauntlet's
/// shortest decimals, the run's widened f32s), and the scene computes from those.
///
/// A street (the gauntlet's, the meteor run's): engine-vehicle-wasm `gauntlet::Layout`,
/// and the run's finish line and meteors.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StreetLayout {
    pub spawn: [f64; 3],
    pub blocks: Vec<LayoutBlock>,
    pub cores: Vec<LayoutBlock>,
    pub parked: Vec<LayoutParked>,
    #[serde(default)]
    pub finish: Option<f64>,
    #[serde(default)]
    pub meteor: Option<LayoutMeteor>,
}

/// A target pad of the meteor range.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct RangePad {
    pub p: [f64; 3],
    pub half: [f64; 3],
    pub lane: u32,
}

/// The meteor range (engine-vehicle-wasm `MeteorRange::range`).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RangeLayout {
    pub pads: Vec<RangePad>,
    pub radius: f64,
    pub mass: f64,
}

impl StreetLayout {
    fn of(l: &engine_vehicle::streets::gauntlet::Layout, num: fn(f32) -> f64) -> Self {
        let block = |b: &engine_vehicle::streets::gauntlet::Block| LayoutBlock {
            p: b.p.map(num),
            half: b.half.map(num),
            mass: num(b.mass),
        };
        StreetLayout {
            spawn: l.spawn.map(num),
            blocks: l.blocks.iter().map(block).collect(),
            cores: l.cores.iter().map(block).collect(),
            parked: l
                .parked
                .iter()
                .map(|c| LayoutParked {
                    p: c.p.map(num),
                    yaw: num(c.yaw),
                })
                .collect(),
            finish: None,
            meteor: None,
        }
    }

    /// The Gauntlet's street, as the page holds it from `Gauntlet.layout()` (its JSON's
    /// shortest decimals).
    pub fn gauntlet() -> Self {
        Self::of(&engine_vehicle::streets::gauntlet::layout(), dec)
    }

    /// The meteor run's street with its finish line and meteors, as the page holds it from
    /// `MeteorRun.layout()` (its JSON carries each f32 widened).
    pub fn run() -> Self {
        use engine_vehicle::streets::run;
        let w = |x: f32| x as f64;
        StreetLayout {
            finish: Some(w(run::FINISH)),
            meteor: Some(LayoutMeteor {
                count: run::METEORS,
                radius: w(run::METEOR_RADIUS),
                mass: w(run::METEOR_MASS),
            }),
            ..Self::of(&run::layout(), w)
        }
    }
}

impl RangeLayout {
    /// The meteor range, as the page holds it from `MeteorRange.range()` (widened f32s).
    pub fn meteors() -> Self {
        use engine_vehicle::streets::meteors;
        let w = |x: f32| x as f64;
        RangeLayout {
            pads: meteors::pads()
                .iter()
                .map(|p| RangePad {
                    p: p.p.map(w),
                    half: p.half.map(w),
                    lane: p.lane,
                })
                .collect(),
            radius: w(meteors::RADIUS),
            mass: w(meteors::MASS),
        }
    }
}

/// A vehicle scene: the scene and the page's switches.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", default)]
pub struct VehicleSceneDef {
    pub scene: String,
    /// The player's preset (none: the scene's).
    pub preset: Option<String>,
    /// Multiplies every mount's break limits (`?strength=`).
    pub strength: f32,
    /// The buggies' node–beam frames (`?build=hybrid`).
    pub hybrid: bool,
    /// Rigid mode for the hybrid buggies (`?rigid=0` turns it off).
    pub rigid_mode: bool,
    /// The player's spawn (x, z, yaw), any part of it (`?spawn=`).
    pub spawn: Vec<f64>,
    /// Traffic's AI vehicles (`?vehicles=`, 1..60).
    pub vehicles: u32,
    /// The bridge's heavy vehicle (`?heavy=`; "none": the player alone).
    pub heavy: Option<String>,
    /// The tank as before track pods (`?tracks=shells`).
    pub tracks_shells: bool,
    /// Sleep off (`?sleep=off`).
    pub sleep_off: bool,
    /// The resting-pair carry (`?rest=carry`).
    pub carry_rest: bool,
    /// Continuous collision for the cannonballs (`?ccd=0` turns it off).
    pub cannon_ccd: bool,
    /// The gauntlet's or the meteor run's street.
    pub street: Option<StreetLayout>,
    /// The meteor range's pads and meteors.
    pub range: Option<RangeLayout>,
}

impl Default for VehicleSceneDef {
    fn default() -> Self {
        Self {
            scene: "playground".into(),
            preset: None,
            strength: 1.0,
            hybrid: false,
            rigid_mode: true,
            spawn: Vec::new(),
            vehicles: 48,
            heavy: None,
            tracks_shells: false,
            sleep_off: false,
            carry_rest: false,
            cannon_ccd: true,
            street: None,
            range: None,
        }
    }
}

impl VehicleSceneDef {
    fn known(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            SCENES.iter().any(|(s, _)| *s == self.scene),
            "unknown scene {}",
            self.scene
        );
        Ok(())
    }

    /// The player's preset: the one asked for, else the scene's.
    pub fn preset_name(&self) -> String {
        if let Some(p) = &self.preset {
            return p.clone();
        }
        match self.scene.as_str() {
            "destruction" => "destructible_sedan",
            "carry" | "gauntlet" | "run" => "monster_truck",
            "tanklab" => "tank",
            "buggies" | "vibeland" => "dune_buggy",
            _ => "sedan",
        }
        .into()
    }

    fn hybrid(&self) -> bool {
        self.hybrid && (self.scene == "buggies" || self.scene == "vibeland")
    }

    fn traffic(&self) -> u32 {
        if self.scene == "traffic" {
            self.vehicles.clamp(1, 60)
        } else {
            0
        }
    }

    fn guided(&self) -> bool {
        matches!(self.scene.as_str(), "gauntlet" | "meteors" | "run")
    }

    /// The coupling labs (their skids are frictionless).
    fn labs(&self) -> bool {
        !matches!(
            self.scene.as_str(),
            "playground"
                | "destruction"
                | "traffic"
                | "buggies"
                | "vibeland"
                | "gauntlet"
                | "meteors"
                | "run"
        )
    }

    fn spawn(&self, k: usize, or: f64) -> f64 {
        self.spawn.get(k).copied().unwrap_or(or)
    }
}

/// Steps a 60 Hz frame takes: four at the hybrid buggies' 240 Hz, else one.
pub fn steps_per_frame(def: &VehicleSceneDef) -> u32 {
    if def.hybrid() {
        (SOFT_HZ / 60.0).round() as u32
    } else {
        1
    }
}

fn pow2(n: u32) -> u32 {
    n.max(1).next_power_of_two()
}

/// The world a scene needs: its capacity, gravity, vehicle and soft pools and guidance.
pub fn world_def(def: &VehicleSceneDef) -> anyhow::Result<WorldDef> {
    def.known()?;
    let n = def.traffic();
    let est = match def.scene.as_str() {
        "traffic" => 4000 + n * 40,
        "gauntlet" => 9000,
        "run" => 15000,
        _ => 6000,
    };
    let max_bodies = pow2(est.max(4096));
    let hybrid = def.hybrid();
    let capacity = WorldCapacity {
        max_bodies,
        max_colliders: pow2((est * 2).max(max_bodies * 2).max(8192)),
        max_contacts: pow2((est * 16).max(65536)).min(1 << 21),
        max_joints: if hybrid { 1024 } else { 64 },
        max_candidates: pow2((est * 32).max(131072)).min(1 << 21),
        max_colors: 32,
        max_commands: 2048,
        max_hull_vertices: 64,
    };
    let mut out = WorldDef {
        gravity: [0.0, if hybrid { -9.81 } else { -10.0 }, 0.0],
        contacts: ContactStorage::Canonical,
        capacity,
        vehicles: VehicleCapacity {
            max_vehicles: 64,
            max_wheels: 1024,
            max_curve_points: 8192,
            max_diffs: 256,
            max_anti_rolls: 256,
            max_parts: 512,
            max_rays: 64,
        },
        ..WorldDef::default()
    };
    if hybrid {
        out.soft = Some(SoftCapacity {
            max_bodies: 32,
            max_beams: 24576,
            max_particles: 4096,
            max_colors: 64,
            max_groups: 1024,
            max_triangles: 12288,
            max_obstacles: 64,
            ..SoftCapacity::default()
        });
    }
    if def.guided() {
        out.guidance = Some(GuidanceCapacity {
            max_zones: 256,
            plan_budget: 256,
        });
    }
    Ok(out)
}

/// A body parked static until launched (made dynamic with its mass first, so a launch only
/// changes its type, pose and velocity).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ParkedBody {
    pub id: u64,
    pub slot: u32,
    pub home: [f32; 3],
}

/// A vehicle of the scene: its handle, preset, AI input, soft body and trailer.
#[derive(Clone, Debug, Serialize)]
pub struct PlacedVehicle {
    pub h: u32,
    pub name: String,
    pub ai: Option<[f32; 3]>,
    pub soft: Option<u32>,
    pub trailer: Option<u32>,
}

/// A meteor range pad.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PlacedPad {
    pub body: u64,
    pub slot: u32,
    pub lane: u32,
    pub pad: u32,
    pub y: f32,
    pub z: f32,
}

/// A touchdown marker (collides with nothing).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PlacedMarker {
    pub id: u64,
    pub collider: u32,
}

/// What a page needs of a placed scene.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Placed {
    /// Each collider's page kind ([`MAT_GROUND`], ...; the vehicles' own are 0).
    pub kinds: Vec<u32>,
    pub vehicles: Vec<PlacedVehicle>,
    pub cannon: Vec<ParkedBody>,
    /// The street's facade blocks (static until a blast launches them), its parked cars.
    pub blocks: Vec<ParkedBody>,
    pub parked: Vec<u32>,
    /// The meteor range's pads, meteors and markers.
    pub pads: Vec<PlacedPad>,
    pub meteors: Vec<ParkedBody>,
    pub markers: Vec<PlacedMarker>,
    /// The meteor run's meteors.
    pub rocks: Vec<ParkedBody>,
    /// Breakable joints in the scene (the bridge's bolts).
    pub bridge_bolts: u32,
    /// The world's body and collider capacity ([`world_def`]): the page sizes its kind
    /// table and hide bitset by them.
    pub capacity: PlacedCapacity,
}

/// The capacity a page sizes its tables by.
#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlacedCapacity {
    pub max_bodies: u32,
    pub max_colliders: u32,
}

/// `q` (x, y, z, w) about axis `ax` by `ang`, in f64 as the page computes it.
fn q_axis(ax: [f64; 3], ang: f64) -> [f64; 4] {
    let s = (ang / 2.0).sin();
    [ax[0] * s, ax[1] * s, ax[2] * s, (ang / 2.0).cos()]
}

/// An f32 the page read from JSON, as the page holds it: its shortest decimal as an f64.
fn dec(x: f32) -> f64 {
    x.to_string().parse().expect("an f32 prints as a number")
}

fn dec3(v: [f32; 3]) -> [f64; 3] {
    v.map(dec)
}

fn f3(v: [f64; 3]) -> [f32; 3] {
    v.map(|x| x as f32)
}

fn f4(v: [f64; 4]) -> [f32; 4] {
    v.map(|x| x as f32)
}

fn box_i(m: f64, h: [f64; 3]) -> [f32; 3] {
    [
        (m * (h[1] * h[1] + h[2] * h[2])) / 3.0,
        (m * (h[0] * h[0] + h[2] * h[2])) / 3.0,
        (m * (h[0] * h[0] + h[1] * h[1])) / 3.0,
    ]
    .map(|x| x as f32)
}

fn ball_i(m: f64, r: f64) -> [f32; 3] {
    [(0.4 * m * r * r) as f32; 3]
}

const IDENT: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

/// The page's placement helpers over a world.
struct Builder<'w> {
    world: &'w mut World,
    friction: [f32; 64],
    out: Placed,
    descs: Vec<VehicleDesc>,
    def: VehicleSceneDef,
}

impl Builder<'_> {
    fn shape(
        &mut self,
        body: BodyId,
        geometry: Geometry,
        kind: u32,
        filter: Option<Filter>,
    ) -> anyhow::Result<u32> {
        let def = ShapeDef {
            material: SurfaceMaterial {
                friction: self.friction.get(kind as usize).copied().unwrap_or(0.6),
                ..SurfaceMaterial::default()
            },
            filter: filter.unwrap_or_default(),
            ..ShapeDef::default()
        };
        let s = self.world.create_shape(body, &def, geometry)?;
        let c = self.world.shape_collider(s)?;
        if self.out.kinds.len() <= c as usize {
            self.out.kinds.resize(c as usize + 1, 0);
        }
        self.out.kinds[c as usize] = kind;
        Ok(c)
    }

    fn box_of(half: [f64; 3], position: Option<[f64; 3]>) -> Geometry {
        Geometry::Box {
            half: f3(half),
            transform: position.map_or(Transform::IDENTITY, |p| {
                Transform::new(f3(p), [0.0, 0.0, 0.0, 1.0])
            }),
        }
    }

    fn static_box(
        &mut self,
        p: [f64; 3],
        half: [f64; 3],
        q: [f64; 4],
        kind: u32,
    ) -> anyhow::Result<BodyId> {
        let body = self.world.create_body(&BodyDef {
            position: f3(p),
            rotation: f4(q),
            ..BodyDef::default()
        })?;
        self.shape(body, Self::box_of(half, None), kind, None)?;
        Ok(body)
    }

    fn dyn_body(
        &mut self,
        p: [f64; 3],
        q: [f64; 4],
        v: [f64; 3],
        is_bullet: bool,
    ) -> anyhow::Result<BodyId> {
        Ok(self.world.create_body(&BodyDef {
            body_type: BodyType::Dynamic,
            position: f3(p),
            rotation: f4(q),
            linear_velocity: f3(v),
            is_bullet,
            ..BodyDef::default()
        })?)
    }

    fn set_mass(&mut self, body: BodyId, mass: f64, inertia: [f32; 3]) -> anyhow::Result<()> {
        self.world
            .body_set_mass_data(body, MassData::principal(mass as f32, [0.0; 3], inertia))
    }

    fn dyn_box(&mut self, p: [f64; 3], half: [f64; 3], mass: f64) -> anyhow::Result<BodyId> {
        let body = self.dyn_body(p, IDENT, [0.0; 3], false)?;
        self.shape(body, Self::box_of(half, None), MAT_PROP, None)?;
        self.set_mass(body, mass, box_i(mass, half))?;
        Ok(body)
    }

    fn dyn_ball(&mut self, p: [f64; 3], r: f64, mass: f64) -> anyhow::Result<BodyId> {
        let body = self.dyn_body(p, IDENT, [0.0; 3], false)?;
        self.shape(
            body,
            Geometry::Sphere {
                center: [0.0; 3],
                radius: r as f32,
            },
            MAT_PROP,
            None,
        )?;
        self.set_mass(body, mass, ball_i(mass, r))?;
        Ok(body)
    }

    fn parked(
        &mut self,
        p: [f64; 3],
        geometry: Geometry,
        kind: u32,
        mass: f64,
        inertia: [f32; 3],
        is_bullet: bool,
    ) -> anyhow::Result<ParkedBody> {
        let body = self.dyn_body(p, IDENT, [0.0; 3], is_bullet)?;
        self.shape(body, geometry, kind, None)?;
        self.set_mass(body, mass, inertia)?;
        self.world.body_set_type(
            body,
            BodyType::Static,
            Transform::new(f3(p), [0.0, 0.0, 0.0, 1.0]),
            [0.0; 3],
            [0.0; 3],
        )?;
        Ok(ParkedBody {
            id: body.to_bits(),
            slot: body.index(),
            home: f3(p),
        })
    }

    fn desc(&self, name: &str) -> anyhow::Result<VehicleDesc> {
        // The tank with wheel shells is its preset renamed (`?tracks=shells`).
        let shells = format!("{name}_shells");
        let d = self
            .descs
            .iter()
            .find(|d| d.name == name || (self.def.tracks_shells && d.name == shells))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "unknown preset {name}; have {}",
                    self.descs
                        .iter()
                        .map(|d| d.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
        let mut d = d.clone();
        let k = self.def.strength;
        if k != 1.0 {
            let scale = |m: &mut MountDesc| {
                m.break_force = (dec(m.break_force) * dec(k)) as f32;
                m.break_torque = (dec(m.break_torque) * dec(k)) as f32;
            };
            d.parts.iter_mut().for_each(|p| scale(&mut p.mount));
            d.wheels.iter_mut().for_each(|w| scale(&mut w.mount));
        }
        Ok(d)
    }

    fn add_vehicle(
        &mut self,
        name: &str,
        p: [f64; 3],
        yaw: f64,
        ai: Option<[f32; 3]>,
    ) -> anyhow::Result<u32> {
        let q = q_axis([0.0, 1.0, 0.0], yaw);
        if self.def.hybrid() && name == "dune_buggy" {
            // The soft frame stands on its core's wheels from the ground up.
            let k = engine_softbody::buggy::BuggyBeams {
                mount_scale: if self.def.strength > 0.0 {
                    self.def.strength
                } else {
                    1.0
                },
                ..Default::default()
            };
            let desc = engine_softbody::buggy::dune_buggy_soft_with(&k);
            let s = self.world.add_soft_vehicle(
                &desc,
                [p[0] as f32, 0.1, p[2] as f32],
                f4(q),
                &SpawnOptions::default(),
            )?;
            let h = self
                .world
                .soft_vehicle(s)
                .ok_or_else(|| anyhow::anyhow!("the soft buggy has no core"))?;
            self.out.vehicles.push(PlacedVehicle {
                h: h.0,
                name: name.into(),
                ai,
                soft: Some(s.0),
                trailer: None,
            });
            return Ok(h.0);
        }
        let desc = self.desc(name)?;
        let h = self.world.add_vehicle(&desc, f3(p), f4(q))?;
        let mut v = PlacedVehicle {
            h: h.0,
            name: name.into(),
            ai,
            soft: None,
            trailer: None,
        };
        if name == "vibe_semi" {
            // Its flatbed: a vehicle of its own, placed so the two hitch points meet (both
            // design spaces share the origin), joined there by a ball joint.
            let trailer = self.desc("vibe_semi_trailer")?;
            let host = self.world.sim().vehicle_host();
            // The page reads the origin as JSON and the centre of mass as an f32 array.
            let oc = dec3(host.origin(h));
            let ot = trailer.center_of_mass().map(|x| x as f64);
            let d = qrot(q, [ot[0] - oc[0], ot[1] - oc[1], ot[2] - oc[2]]);
            let t = self.world.add_vehicle(
                &trailer,
                f3([p[0] + d[0], p[1] + d[1], p[2] + d[2]]),
                f4(q),
            )?;
            let hitch = dec3(presets::VIBE_SEMI_HITCH);
            let host = self.world.sim().vehicle_host();
            let (ca, cb) = (host.chassis_body(h), host.chassis_body(t));
            let body = |s: u32| {
                self.world
                    .body_at(s)
                    .ok_or_else(|| anyhow::anyhow!("no body at slot {s}"))
            };
            let (a, b) = (body(ca)?, body(cb)?);
            ball_joint(
                self.world,
                a,
                b,
                f3([hitch[0] - oc[0], hitch[1] - oc[1], hitch[2] - oc[2]]),
                f3([hitch[0] - ot[0], hitch[1] - ot[1], hitch[2] - ot[2]]),
                0.0,
            )?;
            v.trailer = Some(t.0);
        }
        self.out.vehicles.push(v);
        Ok(h.0)
    }

    fn add_blocks(&mut self, g: &StreetLayout) -> anyhow::Result<()> {
        let mut r = 0f64;
        for b in &g.blocks {
            let (half, p) = (b.half, b.p);
            let parked = self.parked(
                p,
                Self::box_of(half, None),
                MAT_PROP,
                b.mass,
                box_i(b.mass, half),
                false,
            )?;
            self.out.blocks.push(parked);
            let br = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
            if br <= 2.0 {
                r = r.max(br);
            }
        }
        self.world.reserve_moving_radius(r as f32);
        Ok(())
    }
}

/// A D6 ball joint between `a` and `b` at their anchors, breaking past `break_force` N
/// (0: never), as the page's `createBallJoint` makes it.
fn ball_joint(
    world: &mut World,
    a: BodyId,
    b: BodyId,
    anchor_a: [f32; 3],
    anchor_b: [f32; 3],
    break_force: f32,
) -> anyhow::Result<JointId> {
    let big = |v: f32| if v > 0.0 { v } else { engine_schema::JOINT_BIG };
    let desc = engine_schema::JointDesc::ball(
        0,
        0,
        glam::Vec3::from(anchor_a),
        glam::Vec3::from(anchor_b),
    )
    .with_break(big(break_force), big(0.0));
    world.create_joint(&JointDef::Custom {
        body_a: a,
        body_b: b,
        desc,
        user_data: 0,
    })
}

/// `v` rotated by `q`, in f64 as the page computes it.
fn qrot(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
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

/// Place scene `def` in `world` (made from [`world_def`], empty): its solver and material
/// slots, the ground slab, the scene's bodies and vehicles, the cannonballs, the AI inputs.
pub fn place(world: &mut World, def: &VehicleSceneDef) -> anyhow::Result<Placed> {
    def.known()?;
    anyhow::ensure!(
        world.body_count() == 0,
        "a vehicle scene is placed in an empty world"
    );
    let scene = def.scene.as_str();
    // The bridge needs 20 iterations to converge its bending within a fast crossing.
    let mut solver = world.solver();
    solver.iterations = if scene == "bridge" { 20 } else { 8 };
    world.set_solver(solver);
    let hybrid = def.hybrid();
    if hybrid {
        let mut s = world.solver();
        s.dt = 1.0 / SOFT_HZ;
        s.iterations = 10;
        s.gravity_y = -9.81;
        s.beta_lin = 1e7;
        world.set_solver(s);
        // Rigid mode: a buggy that only drives rides its core; any contact turns it soft.
        world.set_soft_settings(SoftSettings {
            rigid_mode: def.rigid_mode,
            ..SoftSettings::calibrated_for(SOFT_HZ)
        })?;
    }
    if def.sleep_off {
        let mut s = world.solver();
        s.sleep_time = 1e6;
        world.set_solver(s);
    }
    if def.carry_rest {
        world.set_carry_rest(true);
    }
    // The material table (64 slots, friction 0.6 unless set): vehicle parts name slot 0,
    // wheel rims slot WHEEL_RIM_MATERIAL and soft nodes slot 9; shapes of a given friction
    // share the first slot with it.
    let mut friction = [0.6f32; 64];
    friction[MAT_GROUND as usize] = 1.0;
    friction[MAT_PROP as usize] = 0.6;
    friction[MAT_ICE as usize] = 0.08;
    friction[engine_vehicle::pack::WHEEL_RIM_MATERIAL as usize] = 0.9;
    if def.labs() {
        friction[MAT_SKID as usize] = 0.0;
    }
    if hybrid {
        friction[MAT_SOFT_NODE as usize] = 0.8;
    }
    for (i, f) in friction.iter().enumerate() {
        world.set_material_slot(
            i as u32,
            SurfaceMaterial {
                friction: *f,
                ..SurfaceMaterial::default()
            },
        )?;
    }
    let mut descs = presets::all();
    let player = def.preset_name();
    if def.tracks_shells {
        if let Some(d) = descs
            .iter_mut()
            .find(|d| d.name == player && d.tracks.is_some())
        {
            if let Some(t) = d.tracks.as_mut() {
                t.left.pod = None;
                t.right.pod = None;
            }
            if let Some(belly) = d.parts.iter().position(|p| p.name == "belly") {
                let m = d.parts.remove(belly).mass;
                d.parts[0].mass = (dec(d.parts[0].mass) + dec(m)) as f32;
            }
            d.name += "_shells";
        }
    }
    let mut b = Builder {
        world,
        friction,
        out: Placed::default(),
        descs,
        def: def.clone(),
    };
    b.desc(&player)?;
    // The ground slab (collider 0; the renderer draws it as a plane).
    b.static_box([0.0, -5.0, 0.0], [400.0, 5.0, 400.0], IDENT, MAT_GROUND)?;
    let sx = |k: usize, or: f64| def.spawn(k, or);
    let xq = |a: f64| q_axis([1.0, 0.0, 0.0], a);
    let parked_ai = Some([0.0, 0.0, 1.0]);
    match scene {
        "playground" | "destruction" => {
            b.add_vehicle(&player, [sx(0, 0.0), 2.5, sx(1, 0.0)], sx(2, 0.0), None)?;
            // Ramps (static, rotated about X), a long hill and a kicker.
            b.static_box([0.0, 0.6, 30.0], [4.0, 0.4, 6.0], xq(-0.22), MAT_PROP)?;
            b.static_box([0.0, 1.95, 41.0], [4.0, 0.4, 5.5], IDENT, MAT_PROP)?;
            b.static_box([0.0, 0.6, 52.0], [4.0, 0.4, 6.0], xq(0.22), MAT_PROP)?;
            b.static_box([-22.0, 0.9, 10.0], [3.0, 0.3, 4.0], xq(-0.32), MAT_PROP)?;
            b.static_box(
                [24.0, 0.0, -20.0],
                [10.0, 1.5, 14.0],
                q_axis([0.0, 0.0, 1.0], 0.18),
                MAT_PROP,
            )?;
            // Ice patch.
            b.static_box([-30.0, -0.02, -25.0], [12.0, 0.05, 12.0], IDENT, MAT_ICE)?;
            // Bollards.
            for i in 0..8 {
                b.static_box(
                    [-12.0 + i as f64 * 3.4, 0.6, 18.0],
                    [0.25, 0.6, 0.25],
                    IDENT,
                    MAT_PROP,
                )?;
            }
            // Box stacks.
            for (x, z) in [(8.0, 26.0), (-10.0, 32.0), (14.0, 40.0)] {
                for y in 0..5 {
                    for i in 0..4 - (y >> 1) {
                        b.dyn_box(
                            [x + (i as f64 - 1.5) * 1.05, 0.5 + y as f64 * 1.0, z],
                            [0.5, 0.5, 0.5],
                            40.0,
                        )?;
                    }
                }
            }
            // Brick wall.
            for c in 0..8 {
                for i in 0..14 {
                    b.dyn_box(
                        [
                            -20.0 + i as f64 * 1.02 + (c & 1) as f64 * 0.5,
                            0.25 + c as f64 * 0.5,
                            60.0,
                        ],
                        [0.5, 0.25, 0.25],
                        12.0,
                    )?;
                }
            }
            // Balls.
            for i in 0..6 {
                b.dyn_ball([20.0 + i as f64 * 2.2, 0.8, 6.0], 0.8, 60.0)?;
            }
            if scene == "destruction" {
                b.add_vehicle("destructible_sedan", [8.0, 2.5, -10.0], 1.2, None)?;
                b.add_vehicle("destructible_sedan", [-8.0, 2.5, -12.0], -0.6, None)?;
                for c in 0..6 {
                    for i in 0..10 {
                        b.dyn_box(
                            [
                                12.0 + i as f64 * 1.02 + (c & 1) as f64 * 0.5,
                                0.25 + c as f64 * 0.5,
                                -30.0,
                            ],
                            [0.5, 0.25, 0.25],
                            12.0,
                        )?;
                    }
                }
            } else {
                b.add_vehicle("monster_truck", [-14.0, 3.5, -12.0], 0.4, None)?;
                b.add_vehicle("kart", [6.0, 1.5, -6.0], 2.0, None)?;
                b.add_vehicle("six_wheel_truck", [18.0, 3.0, 22.0], -1.2, None)?;
            }
        }
        "buggies" | "vibeland" => {
            // Arena for vibe-land's vehicles: every part breaks off when its mount is
            // overloaded.
            b.add_vehicle(&player, [sx(0, 0.0), 1.5, sx(1, 0.0)], sx(2, 0.0), None)?;
            for (x, z, hx, hz) in [
                (0.0, 95.0, 60.0, 1.0),
                (0.0, -45.0, 60.0, 1.0),
                (60.0, 25.0, 1.0, 70.0),
                (-60.0, 25.0, 1.0, 70.0),
            ] {
                b.static_box([x, 1.0, z], [hx, 1.0, hz], IDENT, MAT_PROP)?;
            }
            // Dunes ahead: a kicker, a tabletop with its down-ramp, then a step-down.
            b.static_box([0.0, 0.55, 22.0], [3.5, 0.3, 4.0], xq(-0.3), MAT_PROP)?;
            b.static_box([0.0, 1.6, 33.0], [3.5, 0.3, 5.0], IDENT, MAT_PROP)?;
            b.static_box([0.0, 0.55, 44.0], [3.5, 0.3, 4.0], xq(0.3), MAT_PROP)?;
            b.static_box([-18.0, 0.7, 18.0], [3.0, 0.3, 5.0], xq(-0.22), MAT_PROP)?;
            b.static_box([-18.0, 1.85, 30.0], [3.0, 0.3, 7.0], IDENT, MAT_PROP)?;
            // A concrete barrier row to crash into.
            for i in 0..7 {
                b.static_box(
                    [16.0 + i as f64 * 2.2, 0.5, 8.0],
                    [1.0, 0.5, 0.3],
                    IDENT,
                    MAT_PROP,
                )?;
            }
            // Brick wall and crate stacks.
            for c in 0..6 {
                for i in 0..12 {
                    b.dyn_box(
                        [
                            -12.0 + i as f64 * 1.02 + (c & 1) as f64 * 0.5,
                            0.25 + c as f64 * 0.5,
                            70.0,
                        ],
                        [0.5, 0.25, 0.25],
                        12.0,
                    )?;
                }
            }
            for (x, z) in [(20.0, 40.0), (-30.0, 50.0)] {
                for y in 0..4 {
                    for i in 0..4 - (y >> 1) {
                        b.dyn_box(
                            [x + (i as f64 - 1.5) * 1.05, 0.5 + y as f64 * 1.0, z],
                            [0.5, 0.5, 0.5],
                            40.0,
                        )?;
                    }
                }
            }
            if scene == "buggies" {
                // Parked buggies: targets for the cannon and for ramming.
                for (x, z, yaw) in [
                    (12.0, -12.0, 1.57),
                    (-12.0, -12.0, -1.57),
                    (30.0, 20.0, 0.6),
                    (-30.0, 15.0, -2.2),
                ] {
                    b.add_vehicle("dune_buggy", [x, 1.5, z], yaw, None)?;
                }
                // A heavy ball dropped on the first one's engine and fuel tank.
                b.dyn_ball([10.8, 9.0, -12.0], 0.7, 900.0)?;
            } else {
                // The catalog, parked in a row behind the start (Tab drives the next one).
                for (i, name) in [
                    "dune_buggy",
                    "vibe_trophy",
                    "vibe_rally",
                    "vibe_monster",
                    "vibe_derby",
                    "vibe_sprint",
                    "vibe_semi",
                ]
                .into_iter()
                .enumerate()
                {
                    let y = if name == "vibe_monster" { 1.2 } else { 0.8 };
                    b.add_vehicle(name, [-24.0 + i as f64 * 8.0, y, -16.0], 0.0, None)?;
                }
            }
        }
        "meteors" => {
            // Two lanes of target pads; the left one is steered. The car only anchors the
            // camera, between the lanes.
            let g = def.range.clone().unwrap_or_else(RangeLayout::meteors);
            b.add_vehicle(&player, [0.0, 2.2, 18.0], 0.0, None)?;
            // Pads are kinematic, so they can slide as carts (moving targets).
            for p in &g.pads {
                let body = b.world.create_body(&BodyDef {
                    body_type: BodyType::Kinematic,
                    position: [p.p[0] as f32, p.half[1] as f32, p.p[2] as f32],
                    ..BodyDef::default()
                })?;
                b.shape(body, Builder::box_of(p.half, None), MAT_PAD, None)?;
                let pad = b.out.pads.iter().filter(|x| x.lane == p.lane).count() as u32;
                b.out.pads.push(PlacedPad {
                    body: body.to_bits(),
                    slot: body.index(),
                    lane: p.lane,
                    pad,
                    y: p.half[1] as f32,
                    z: p.p[2] as f32,
                });
            }
            let (r, m) = (g.radius, g.mass);
            // Parked meteors (static until launched) and touchdown markers.
            for i in 0..36 {
                let geometry = Geometry::Sphere {
                    center: [0.0; 3],
                    radius: r as f32,
                };
                let mt = b.parked(
                    [-3000.0 - i as f64 * 3.0, -3000.0, -3000.0],
                    geometry,
                    MAT_PROP,
                    m,
                    ball_i(m, r),
                    false,
                )?;
                b.out.meteors.push(mt);
            }
            for i in 0..60 {
                let p = [-3000.0 - i as f64 * 3.0, -3000.0, -2000.0];
                let body = b.world.create_body(&BodyDef {
                    position: f3(p),
                    ..BodyDef::default()
                })?;
                let filter = Filter {
                    categories: 0x8000,
                    mask: 0,
                    group: 0,
                };
                let c = b.shape(
                    body,
                    Builder::box_of([0.25, 0.06, 0.25], None),
                    MAT_MISS,
                    Some(filter),
                )?;
                b.out.markers.push(PlacedMarker {
                    id: body.to_bits(),
                    collider: c,
                });
            }
        }
        "gauntlet" | "run" => {
            // Masonry facades on both sides of a street, static cores behind them and
            // parked cars at the kerbs. Facade blocks stay static until a blast launches
            // them.
            let g = def.street.clone().unwrap_or_else(|| {
                if scene == "run" {
                    StreetLayout::run()
                } else {
                    StreetLayout::gauntlet()
                }
            });
            let sp = g.spawn;
            b.add_vehicle(&player, sp, 0.0, None)?;
            for c in &g.cores {
                b.static_box(c.p, c.half, IDENT, MAT_PROP)?;
            }
            b.add_blocks(&g)?;
            for c in &g.parked {
                let h = b.add_vehicle("sedan", c.p, c.yaw, parked_ai)?;
                b.out.parked.push(h);
            }
            if scene == "run" {
                let finish = g
                    .finish
                    .ok_or_else(|| anyhow::anyhow!("the run's layout has no finish"))?;
                // The finish gate: posts at the kerbs and a banner across, over the road.
                for x in [-7.4, 7.4] {
                    b.static_box([x, 4.0, finish], [0.3, 4.0, 0.3], IDENT, MAT_FINISH)?;
                }
                b.static_box([0.0, 8.4, finish], [7.7, 0.5, 0.15], IDENT, MAT_FINISH)?;
                // Parked meteors (continuous collision once they fly), each with its trail.
                let mt = g
                    .meteor
                    .ok_or_else(|| anyhow::anyhow!("the run's layout has no meteors"))?;
                let (r, m) = (mt.radius, mt.mass);
                for i in 0..mt.count {
                    let geometry = Geometry::Sphere {
                        center: [0.0; 3],
                        radius: r as f32,
                    };
                    let rock = b.parked(
                        [-3000.0 - i as f64 * 4.0, -3000.0, -1000.0],
                        geometry,
                        MAT_METEOR,
                        m,
                        ball_i(m, r),
                        true,
                    )?;
                    let filter = Filter {
                        categories: 0x8000,
                        mask: 0,
                        group: 0,
                    };
                    let id = BodyId::from_bits(rock.id);
                    b.shape(
                        id,
                        Builder::box_of([0.8 * r, 0.8 * r, 0.5], None),
                        MAT_TRAIL,
                        Some(filter),
                    )?;
                    b.out.rocks.push(rock);
                }
                if r <= 2.0 {
                    b.world.reserve_moving_radius(r as f32);
                }
            }
        }
        "traffic" => {
            let n = def.traffic();
            b.add_vehicle(&player, [sx(0, 0.0), 2.5, sx(1, -60.0)], sx(2, 0.0), None)?;
            let kinds = [
                "sedan",
                "sports",
                "kart",
                "jolt_default",
                "destructible_sedan",
                "monster_truck",
            ];
            let cols = (n as f64).sqrt().ceil() as u32;
            let mut rng = 7u32;
            let mut rand = || {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                rng as f64 / 4294967296.0
            };
            for i in 0..n {
                let x = ((i % cols) as f64 - cols as f64 / 2.0) * 14.0;
                let z = ((i / cols) as f64 - cols as f64 / 2.0) * 14.0;
                let kind = kinds[i as usize % kinds.len()];
                let y = if kind == "monster_truck" { 3.5 } else { 2.2 };
                let yaw = rand() * 6.28;
                let forward = 0.2 + 0.25 * rand();
                let side = if rand() < 0.5 { -1.0 } else { 1.0 };
                let steer = side * (0.15 + 0.35 * rand());
                b.add_vehicle(
                    kind,
                    [x, y, z],
                    yaw,
                    Some([forward as f32, steer as f32, 0.0]),
                )?;
            }
            for _ in 0..400 {
                let x = (rand() - 0.5) * 120.0;
                let y = 0.5 + rand() * 6.0;
                let z = (rand() - 0.5) * 120.0;
                b.dyn_box([x, y, z], [0.5, 0.5, 0.5], 30.0)?;
            }
        }
        // The two-way coupling labs: each mirrors a test in tests/vehicle.rs.
        "slab" => {
            // Heavy slab: grippy deck on frictionless skids. The car's tyres push it back.
            let (m, half) = (2500.0, [6.0, 0.15, 30.0]);
            let body = b.dyn_body([0.0, 0.25, 0.0], IDENT, [0.0; 3], false)?;
            b.shape(
                body,
                Builder::box_of(half, Some([0.0, 0.1, 0.0])),
                MAT_GROUND,
                None,
            )?;
            b.shape(
                body,
                Builder::box_of(half, Some([0.0, -0.1, 0.0])),
                MAT_SKID,
                None,
            )?;
            b.set_mass(body, m, box_i(m, [6.0, 0.25, 30.0]))?;
            // Cones along the world so the slab's motion reads on camera.
            let mut z = -40.0;
            while z <= 40.0 {
                b.static_box([8.5, 0.5, z], [0.25, 0.5, 0.25], IDENT, MAT_PROP)?;
                z += 8.0;
            }
            b.add_vehicle(&player, [sx(0, 0.0), 2.0, sx(1, -20.0)], sx(2, 0.0), None)?;
        }
        "seesaw" => {
            // Low pivot, long plank resting on its near (−Z) end; the car's weight tips it.
            b.static_box([0.0, 0.25, 0.0], [2.6, 0.25, 0.2], IDENT, MAT_PROP)?;
            let tilt = -(0.5f64 / 8.0).atan();
            let half = [2.2, 0.08, 7.0];
            let body = b.dyn_body([0.0, 0.58 - (-tilt).tan(), -1.0], xq(tilt), [0.0; 3], false)?;
            b.shape(body, Builder::box_of(half, None), MAT_GROUND, None)?;
            b.set_mass(body, 250.0, box_i(250.0, half))?;
            b.add_vehicle(&player, [sx(0, 0.0), 2.0, sx(1, -16.0)], sx(2, 0.0), None)?;
        }
        "carry" => {
            b.add_vehicle(&player, [sx(0, 0.0), 2.0, sx(1, 0.0)], sx(2, 0.0), None)?;
            // A parked kart (handbrake on) riding on the truck's roof.
            b.add_vehicle("kart", [sx(0, 0.0), 4.0, sx(1, 0.0)], sx(2, 0.0), parked_ai)?;
        }
        "bridge" => {
            // Mirrors bridge() in tests/vehicle.rs: five 2.6 m timber planks (200 kg/m) span
            // a 12 m gap 3 m up on the abutment seats; each seam has four ball-joint bolts
            // rated 255 kN.
            let (y, n, len, gap, hx, hy) = (3.0f64, 5u32, 2.6f64, 0.005f64, 2.2f64, 0.1f64);
            let hz = len / 2.0;
            let (mass, bolt_k, bolt_break) = (200.0 * len, 1e8f32, 255e3f32);
            let pitch = len + gap;
            let end_z = n as f64 * pitch - gap;
            let (seat, road_half, road_x) = (0.5, 30.0, hx + 0.3);
            let seat_top = y - 2.0 * hy;
            b.static_box(
                [0.0, y / 2.0, -gap - road_half],
                [road_x, y / 2.0, road_half],
                IDENT,
                MAT_PROP,
            )?;
            b.static_box(
                [0.0, y / 2.0, end_z + gap + road_half],
                [road_x, y / 2.0, road_half],
                IDENT,
                MAT_PROP,
            )?;
            b.static_box(
                [0.0, seat_top / 2.0, (seat - gap) / 2.0],
                [road_x, seat_top / 2.0, (seat + gap) / 2.0],
                IDENT,
                MAT_PROP,
            )?;
            b.static_box(
                [0.0, seat_top / 2.0, end_z - (seat - gap) / 2.0],
                [road_x, seat_top / 2.0, (seat + gap) / 2.0],
                IDENT,
                MAT_PROP,
            )?;
            let mut planks = Vec::new();
            for i in 0..n {
                let body =
                    b.dyn_body([0.0, y - hy, i as f64 * pitch + hz], IDENT, [0.0; 3], false)?;
                b.shape(body, Builder::box_of([hx, hy, hz], None), MAT_GROUND, None)?;
                b.set_mass(body, mass, box_i(mass, [hx, hy, hz]))?;
                planks.push(body);
            }
            for i in 0..(n - 1) as usize {
                for (bx, by) in [
                    (-0.8 * hx, -0.8 * hy),
                    (0.8 * hx, -0.8 * hy),
                    (-0.8 * hx, 0.8 * hy),
                    (0.8 * hx, 0.8 * hy),
                ] {
                    // A D6 ball bolt; its rows start at BOLT_K.
                    let j = ball_joint(
                        b.world,
                        planks[i],
                        planks[i + 1],
                        f3([bx, by, hz + gap / 2.0]),
                        f3([bx, by, -hz - gap / 2.0]),
                        bolt_break,
                    )?;
                    b.world.joint_set_linear_penalty(j, bolt_k)?;
                }
            }
            b.out.bridge_bolts = (n - 1) * 4;
            // The light car first; the heavy vehicle waits behind it.
            b.add_vehicle(
                &player,
                [sx(0, 0.0), y + 1.5, sx(1, -2.5)],
                sx(2, 0.0),
                None,
            )?;
            let heavy = def.heavy.clone().unwrap_or_else(|| "monster_truck".into());
            if heavy != "none" {
                b.add_vehicle(&heavy, [0.0, y + 2.0, -11.0], 0.0, parked_ai)?;
            }
        }
        "tanklab" => {
            // Rubble in the track paths, logs across the lane, then a 0.6 m ledge.
            for row in 0..7 {
                for x in [-1.7, 0.0, 1.7] {
                    let h = if row % 2 == 1 { 0.175 } else { 0.12 };
                    let z = 10.0 + row as f64 * 1.5;
                    let j = ((row % 3) as f64 - 1.0) * 0.15;
                    let m = 1500.0 * 8.0 * h * h * h;
                    let body = b.dyn_body([x + j, h, z], IDENT, [0.0; 3], false)?;
                    b.shape(body, Builder::box_of([h, h, h], None), MAT_PROP, None)?;
                    b.set_mass(body, m, box_i(m, [h, h, h]))?;
                }
            }
            for z in [24.0, 28.0] {
                let half = [2.6, 0.2, 0.2];
                let body = b.dyn_body([0.0, 0.2, z], IDENT, [0.0; 3], false)?;
                b.shape(body, Builder::box_of(half, None), MAT_PROP, None)?;
                b.set_mass(body, 250.0, box_i(250.0, half))?;
            }
            b.static_box([0.0, 0.3, 36.0 + 25.0], [6.0, 0.3, 25.0], IDENT, MAT_GROUND)?;
            let mut z = 0.0;
            while z <= 80.0 {
                b.static_box([7.5, 0.5, z], [0.25, 0.5, 0.25], IDENT, MAT_PROP)?;
                z += 8.0;
            }
            b.add_vehicle(&player, [sx(0, 0.0), 2.5, sx(1, 0.0)], sx(2, 0.0), None)?;
        }
        "sidewall" => {
            // Light crates in two rows that only the tyre outlines reach.
            for i in 0..8 {
                for x in [-1.15, 1.15] {
                    let half = [0.3, 0.2, 0.3];
                    let body =
                        b.dyn_body([x, 0.2, 6.0 + i as f64 * 2.5], IDENT, [0.0; 3], false)?;
                    b.shape(body, Builder::box_of(half, None), MAT_PROP, None)?;
                    b.set_mass(body, 15.0, box_i(15.0, half))?;
                }
            }
            b.add_vehicle(&player, [sx(0, 0.0), 2.0, sx(1, 0.0)], sx(2, 0.0), None)?;
        }
        _ => unreachable!("checked above"),
    }
    // Cannonballs: parked static spheres (continuous collision unless switched off).
    for i in 0..12 {
        let geometry = Geometry::Sphere {
            center: [0.0; 3],
            radius: 0.6,
        };
        let ball = b.parked(
            [-4000.0 - i as f64 * 5.0, -4000.0, -4000.0],
            geometry,
            MAT_PROP,
            400.0,
            ball_i(400.0, 0.6),
            def.cannon_ccd,
        )?;
        b.out.cannon.push(ball);
    }
    for v in &b.out.vehicles {
        if let Some([forward, steer, handbrake]) = v.ai {
            b.world.set_vehicle_input(
                VehicleHandle(v.h),
                VehicleInput {
                    forward,
                    steer,
                    handbrake,
                    ..VehicleInput::default()
                },
            );
        }
    }
    let cap = world_def(def)?.capacity;
    b.out.capacity = PlacedCapacity {
        max_bodies: cap.max_bodies,
        max_colliders: cap.max_colliders,
    };
    let colliders = b.world.sim().collider_count();
    anyhow::ensure!(
        colliders <= cap.max_colliders,
        "{colliders} colliders exceed capacity {}",
        cap.max_colliders
    );
    b.out.kinds.resize(cap.max_colliders as usize, 0);
    Ok(b.out)
}
