//! Destruction approaches side by side: the same scene and the same shots under the ways a
//! game can decide what breaks, each a setting of this engine (`rubble::lab`), so a host
//! shows what each one lets a player do.
//!
//! - [`Approach`]: what others ship, by technique, and Rubble's own. Each is a
//!   [`LabSettings`]; nothing here is a separate engine.
//! - [`Scenario`]: a player's moment (a balcony loaded until it snaps, a pillar shot out
//!   from under a canopy, one wall struck at four speeds, two shots of equal momentum),
//!   generated as a scene pack with its shots authored in it.
//! - [`Compare`]: a scenario placed under an approach, stepped, and its [`Verdict`]: the
//!   numbers a host prints under the view.
//!
//! Fracture is forward only under every approach (AGENTS.md).

use crate::lab::{Lab, RECORD_FLOATS, LabAction, LabArrangement, LabBonds, LabDef, LabPlan, LabSettings, LabSource};
use crate::{BodyId, FractureOptions, FractureReport, GpuContext, ImpactResponse, StrengthModel, StrikeRule, World};
use serde_json::{json, Value};
use std::sync::Arc;

/// How a game decides what breaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Approach {
    /// A bond breaks when the contact or impact force on its chunk beats its strength;
    /// a piece cut off from every anchor falls. Decided after the step: the engine has
    /// already bounced the shot off the whole wall.
    ImpactThreshold,
    /// `ImpactThreshold` with the weight carried down the bond graph to the anchors:
    /// overloaded members crush, a piece whose centre leaves its supports tips. No bending.
    SupportFlow,
    /// A quasi-static stress solve: gravity and the contact loads carried through the
    /// bonds, each checked in tension, shear, compression and bending. Statics only: an
    /// impact is the force the engine's step already produced.
    StressSolver,
    /// What a stress-solver pipeline commonly pairs with it for impacts: the GPU stress
    /// solver (Blast's quasi-static model) for the load, and radial damage for a hit: every
    /// bond within the weapon's radius of the point struck breaks, at any speed or mass.
    StressRadial,
    /// Rubble: the quasi-static load solve, and a strike decided before the step by the
    /// analytic local impact (ballistic limit, plug, spall, strain rate), so speed and
    /// mass decide whether a shot bounces, craters or goes through.
    Rubble,
}

impl Approach {
    pub const ALL: [Approach; 5] = [Approach::ImpactThreshold, Approach::SupportFlow, Approach::StressSolver, Approach::StressRadial, Approach::Rubble];

    pub fn name(self) -> &'static str {
        match self {
            Approach::ImpactThreshold => "impact_threshold",
            Approach::SupportFlow => "support_flow",
            Approach::StressSolver => "stress_solver",
            Approach::StressRadial => "stress_radial",
            Approach::Rubble => "rubble",
        }
    }

    pub fn parse(name: &str) -> Option<Approach> {
        Self::ALL.into_iter().find(|a| a.name() == name)
    }

    pub fn title(self) -> &'static str {
        match self {
            Approach::ImpactThreshold => "Impact threshold",
            Approach::SupportFlow => "Support graph",
            Approach::StressSolver => "Stress solver (statics)",
            Approach::StressRadial => "Stress solver + radial damage",
            Approach::Rubble => "Rubble",
        }
    }

    /// What the approach is, in a line.
    pub fn blurb(self) -> &'static str {
        match self {
            Approach::ImpactThreshold => "Hit hard enough and a bond breaks; anything still tied to an anchor stays up.",
            Approach::SupportFlow => "Adds weight flowing down to the supports: overloaded members crush, unsupported pieces tip.",
            Approach::StressSolver => "Solves the static load through every bond, bending included. Impacts are whatever force the step produced.",
            Approach::StressRadial => "A static stress solve for load, and for a hit every bond within half a metre breaks, at any speed or mass.",
            Approach::Rubble => "The static load solve plus an impact decided before the step from the shot's speed and mass.",
        }
    }

    /// The lab settings that make it, at `physics_hz`.
    pub fn settings(self, physics_hz: f32) -> LabSettings {
        let after_the_step = FractureOptions {
            contact_fracture: Some(false),
            strike_prediction: Some(false),
            strike: None,
            impact_response: None,
        };
        let (model, fracture, support_loss) = match self {
            Approach::ImpactThreshold => (StrengthModel::Threshold, after_the_step, false),
            Approach::SupportFlow => (StrengthModel::Threshold, after_the_step, true),
            Approach::StressSolver => (StrengthModel::LoadPath, after_the_step, false),
            Approach::StressRadial => (
                StrengthModel::BlastGpu,
                FractureOptions {
                    contact_fracture: Some(false),
                    strike_prediction: Some(true),
                    strike: Some(StrikeRule::Radial {
                        radius_mm: StrikeRule::RADIAL_DEFAULT_MM,
                    }),
                    impact_response: None,
                },
                false,
            ),
            Approach::Rubble => (
                StrengthModel::LoadPath,
                FractureOptions {
                    contact_fracture: Some(false),
                    strike_prediction: Some(true),
                    strike: Some(StrikeRule::Analytic),
                    // What a strike leaves in the structure, solved through its bonds: a
                    // heavy slow shot that does not get through still breaks the wall.
                    impact_response: Some(ImpactResponse::default()),
                },
                false,
            ),
        };
        LabSettings {
            bonds: LabBonds::Model(model),
            physics_hz,
            fracture,
            support_loss,
            // A predicted analytic strike resolves the shot's own contact (through with its
            // plug, or stopped on the face): a swept contact would strike the plug again.
            ccd: self != Approach::Rubble,
            ..LabSettings::default()
        }
    }
}

/// A player's moment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scenario {
    /// A concrete balcony fixed along one edge; weights are set down on it one at a time,
    /// walking out from the wall.
    Balcony,
    /// A canopy on two pillars; one pillar is shot out.
    Pillar,
    /// Four walls alike, the same ball at four speeds, a wall each.
    SpeedLadder,
    /// Two walls alike, two shots of equal momentum: light and fast, heavy and slow.
    EqualMomentum,
}

/// Density of the scenarios' concrete and steel (kg/m³).
const CONCRETE: f64 = 2400.0;
const STEEL: f64 = 7850.0;
/// A balcony weight (kg).
const BALCONY_WEIGHT: f64 = 300.0;
/// Block edge (m).
const B: f64 = 0.25;
/// The struck wall's thickness (m).
const WALL: f64 = B;

fn steel_ball(radius: f64) -> f64 {
    STEEL * 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3)
}

/// An authored shot of a scenario (a scene pack's `defaults.shots` entry).
#[derive(Clone, Debug)]
struct Shot {
    label: String,
    note: &'static str,
    radius: f64,
    mass: f64,
    origin: [f64; 3],
    velocity: [f64; 3],
    /// Seconds into the script.
    at: f64,
}

/// A scene pack of axis-aligned blocks (y up), bonded face to face.
#[derive(Default)]
struct Blocks {
    nodes: Vec<Value>,
    sizes: Vec<Value>,
    colliders: Vec<Value>,
    bonds: Vec<Value>,
    /// (centre, half extents) per node, for bonding.
    boxes: Vec<([f64; 3], [f64; 3])>,
}

impl Blocks {
    /// A block; `anchor`: a support (mass 0).
    fn block(&mut self, centre: [f64; 3], half: [f64; 3], anchor: bool) -> u32 {
        let volume = 8.0 * half[0] * half[1] * half[2];
        let mass = if anchor { 0.0 } else { CONCRETE * volume };
        self.nodes.push(json!({ "centroid": { "x": centre[0], "y": centre[1], "z": centre[2] }, "mass": mass, "volume": volume, "m": 0 }));
        self.sizes.push(json!({ "x": 2.0 * half[0], "y": 2.0 * half[1], "z": 2.0 * half[2] }));
        self.colliders.push(json!({ "kind": "cuboid", "halfExtents": { "x": half[0], "y": half[1], "z": half[2] } }));
        self.boxes.push((centre, half));
        self.nodes.len() as u32 - 1
    }

    /// A grid of `counts` cubes of edge [`B`] from `origin` (its low corner); `anchor`
    /// says which cells are supports.
    fn grid(&mut self, origin: [f64; 3], counts: [usize; 3], anchor: impl Fn([usize; 3]) -> bool) {
        for i in 0..counts[0] {
            for j in 0..counts[1] {
                for k in 0..counts[2] {
                    let c = [origin[0] + (i as f64 + 0.5) * B, origin[1] + (j as f64 + 0.5) * B, origin[2] + (k as f64 + 0.5) * B];
                    self.block(c, [B / 2.0; 3], anchor([i, j, k]));
                }
            }
        }
    }

    /// Bond every pair of blocks that share a face (their overlap its area).
    fn bond_faces(&mut self) {
        let n = self.boxes.len();
        for a in 0..n {
            for b in a + 1..n {
                let (ca, ha) = self.boxes[a];
                let (cb, hb) = self.boxes[b];
                for axis in 0..3 {
                    let gap = (cb[axis] - ca[axis]).abs() - (ha[axis] + hb[axis]);
                    if gap.abs() > 1e-6 {
                        continue;
                    }
                    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                    let overlap = |k: usize| (ca[k] + ha[k]).min(cb[k] + hb[k]) - (ca[k] - ha[k]).max(cb[k] - hb[k]);
                    let (ou, ov) = (overlap(u), overlap(v));
                    if ou <= 1e-6 || ov <= 1e-6 {
                        continue;
                    }
                    let mut centroid = [0.0; 3];
                    let sign = (cb[axis] - ca[axis]).signum();
                    centroid[axis] = ca[axis] + sign * ha[axis];
                    for k in [u, v] {
                        centroid[k] = ((ca[k] + ha[k]).min(cb[k] + hb[k]) + (ca[k] - ha[k]).max(cb[k] - hb[k])) / 2.0;
                    }
                    let mut normal = [0.0; 3];
                    normal[axis] = sign;
                    self.bonds.push(json!({
                        "node0": a, "node1": b, "m": 0, "area": ou * ov,
                        "centroid": { "x": centroid[0], "y": centroid[1], "z": centroid[2] },
                        "normal": { "x": normal[0], "y": normal[1], "z": normal[2] },
                    }));
                }
            }
        }
    }

    fn pack(mut self, key: &str, title: &str, shots: &[Shot]) -> Vec<u8> {
        self.bond_faces();
        let first = shots.first().expect("a scenario has a shot");
        let speed = first.velocity.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
        let v = |p: [f64; 3]| json!({ "x": p[0], "y": p[1], "z": p[2] });
        let shots: Vec<Value> = shots
            .iter()
            .map(|s| {
                json!({
                    "label": s.label, "note": s.note, "radius": s.radius, "mass": s.mass,
                    "massScale": 1, "speedScale": 1,
                    "origin": v(s.origin), "velocity": v(s.velocity), "launchAt": s.at,
                })
            })
            .collect();
        // Structural concrete: 3 MPa in tension, 30 MPa in compression, 4 MPa in shear.
        let pack = json!({
            "version": 2, "key": key, "title": title,
            "defaults": {
                "projectile": { "radius": first.radius, "mass": first.mass, "speed": speed },
                "solver": { "gravity": -9.81, "materialScale": 1, "materials": [{
                    "name": "concrete", "elasticModulus": 30e9,
                    "compressionElastic": 30e6, "compressionFatal": 30e6,
                    "tensionElastic": 3e6, "tensionFatal": 3e6,
                    "shearElastic": 4e6, "shearFatal": 4e6,
                }] },
                "shots": shots,
            },
            "scenario": { "nodes": self.nodes, "nodeSizes": self.sizes, "nodeColliders": self.colliders, "bonds": self.bonds },
        });
        serde_json::to_vec(&pack).expect("a pack serializes")
    }
}

/// A struck wall's width and the gap to the next (m).
const WALL_WIDTH: f64 = 2.0;
const WALL_GAP: f64 = 0.5;

/// Where wall `i` of `n` is centred along x.
fn wall_x(i: usize, n: usize) -> f64 {
    (i as f64 - 0.5 * (n as f64 - 1.0)) * (WALL_WIDTH + WALL_GAP)
}

/// The struck walls of the speed scenarios, one a shot: each 2 m wide (x), 3 m tall, 0.25 m
/// thick (z from 0), its bottom row anchored, half a metre from the next (no bond between
/// them: what one shot does to its wall leaves the others as they were).
fn walls(n: usize) -> Blocks {
    let mut b = Blocks::default();
    for i in 0..n {
        b.grid([wall_x(i, n) - 0.5 * WALL_WIDTH, 0.0, 0.0], [(WALL_WIDTH / B) as usize, 12, 1], |c| c[1] == 0);
    }
    b
}

/// A shot along -z at the wall's front face (z = WALL) at height `y`, arriving level: it
/// starts `travel` seconds away and is lifted by the drop gravity gives it on the way.
fn at_wall(label: String, note: &'static str, x: f64, y: f64, radius: f64, mass: f64, speed: f64, at: f64) -> Shot {
    // At least four steps of flight (a strike is predicted the frame before it lands).
    let travel = (3.0 / speed).clamp(4.0 / 120.0, 0.35);
    let drop = 0.5 * 9.81 * travel * travel;
    Shot {
        label,
        note,
        radius,
        mass,
        origin: [x, y + drop, WALL + radius + speed * travel],
        velocity: [0.0, 9.81 * travel * 0.0, -speed],
        at,
    }
}

impl Scenario {
    pub const ALL: [Scenario; 4] = [Scenario::Balcony, Scenario::Pillar, Scenario::SpeedLadder, Scenario::EqualMomentum];

    pub fn name(self) -> &'static str {
        match self {
            Scenario::Balcony => "balcony",
            Scenario::Pillar => "pillar",
            Scenario::SpeedLadder => "speed_ladder",
            Scenario::EqualMomentum => "equal_momentum",
        }
    }

    pub fn parse(name: &str) -> Option<Scenario> {
        Self::ALL.into_iter().find(|s| s.name() == name)
    }

    pub fn title(self) -> &'static str {
        match self {
            Scenario::Balcony => "Overload the balcony",
            Scenario::Pillar => "Shoot out the pillar",
            Scenario::SpeedLadder => "Same ball, four speeds",
            Scenario::EqualMomentum => "Fast and light, slow and heavy",
        }
    }

    /// What a player notices.
    pub fn caption(self) -> &'static str {
        match self {
            Scenario::Balcony => "Weights go onto a balcony one at a time, each further from the wall. Where the load is solved the root heats up step by step and snaps at a weight you can count; without a load solve it holds anything, forever.",
            Scenario::Pillar => "Take out one of two pillars and the canopy should come down. Without weight in the model it hangs off the other pillar.",
            Scenario::SpeedLadder => "The same ball at four walls: slow, it should stop and crack its wall; fast, it should punch a hole and keep going. Without an impact model every wall stops it; with a damage radius every shot goes through.",
            Scenario::EqualMomentum => "Two shots with the same momentum, a wall each: the fast light one holes its wall, the slow heavy one shoves its wall over. Force alone cannot tell them apart.",
        }
    }

    /// Physics steps per second.
    pub fn physics_hz(self) -> f32 {
        120.0
    }

    /// Seconds the script runs.
    pub fn seconds(self) -> f64 {
        match self {
            Scenario::Balcony => 8.0,
            Scenario::Pillar => 5.0,
            Scenario::SpeedLadder => 6.0,
            Scenario::EqualMomentum => 5.0,
        }
    }

    fn shots(self) -> Vec<Shot> {
        match self {
            Scenario::Balcony => {
                // Weights set down one at a time (from 1 mm: a sustained load, not an
                // impact), in rows of two walking out from the wall: each adds its weight
                // times its distance from the wall to the bending at the root, so the root
                // loads up step by step until it gives.
                let (r, mass) = (0.18, BALCONY_WEIGHT);
                (0..10)
                    .map(|i| {
                        let x = 0.3 + 0.4 * (i / 2) as f64;
                        Shot {
                            label: format!("weight {} ({:.1} m out)", i + 1, x),
                            note: "300 kg, set down",
                            radius: r,
                            mass,
                            origin: [x.min(1.82), 3.0 + B + r + 1e-3, if i % 2 == 0 { -0.25 } else { 0.25 }],
                            velocity: [0.0; 3],
                            at: 0.5 + 0.6 * i as f64,
                        }
                    })
                    .collect()
            }
            Scenario::Pillar => {
                let r = 0.25;
                let speed = 200.0;
                vec![Shot {
                    label: "a cannonball at the left pillar".into(),
                    note: "cuts the pillar",
                    radius: r,
                    mass: steel_ball(r),
                    origin: [-1.5, 1.25, 0.25 + r + speed * 0.05],
                    velocity: [0.0, 0.0, -speed],
                    at: 0.5,
                }]
            }
            Scenario::SpeedLadder => {
                let r = 0.12;
                [20.0, 100.0, 250.0, 500.0]
                    .into_iter()
                    .enumerate()
                    .map(|(i, speed)| at_wall(format!("{speed:.0} m/s"), "the same steel ball, a wall of its own", wall_x(i, 4), 1.6, r, steel_ball(r), speed, 0.5 + 1.25 * i as f64))
                    .collect()
            }
            Scenario::EqualMomentum => {
                let (r, speed) = (0.12, 300.0);
                let light = steel_ball(r);
                let heavy_r = 0.4;
                let heavy = steel_ball(heavy_r);
                vec![
                    at_wall(format!("{light:.0} kg at {speed:.0} m/s"), "light and fast", wall_x(0, 2), 1.6, r, light, speed, 0.5),
                    at_wall(format!("{heavy:.0} kg at {:.1} m/s", light * speed / heavy), "heavy and slow, the same momentum", wall_x(1, 2), 1.6, heavy_r, heavy, light * speed / heavy, 2.0),
                ]
            }
        }
    }

    /// The scenario's scene pack, its shots authored in it.
    pub fn pack(self) -> Vec<u8> {
        let blocks = match self {
            Scenario::Balcony => {
                // 2 m out from the wall (x), 1 m wide, 0.25 m thick, 3 m up; the column in
                // the wall is the anchor.
                let mut b = Blocks::default();
                b.grid([-B, 3.0, -0.5], [9, 1, 4], |c| c[0] == 0);
                b
            }
            Scenario::Pillar => {
                // Two 0.5 m pillars 2.5 m tall at x = -1.5 and 1.5 on anchored footings,
                // a 4 m by 1.5 m canopy 0.25 m thick across their tops.
                let mut b = Blocks::default();
                for x in [-1.75, 1.25] {
                    b.grid([x, 0.0, -0.25], [2, 11, 2], |c| c[1] == 0);
                }
                b.grid([-2.0, 2.75, -0.75], [16, 1, 6], |_| false);
                b
            }
            Scenario::SpeedLadder => walls(4),
            Scenario::EqualMomentum => walls(2),
        };
        blocks.pack(self.name(), self.title(), &self.shots())
    }

    /// The lab run of the scenario under `approach`.
    pub fn lab_def(self, approach: Approach) -> LabDef {
        LabDef {
            source: LabSource::Pack {
                bytes: Arc::new(self.pack()),
            },
            arrangement: LabArrangement::One,
            settings: approach.settings(self.physics_hz()),
        }
    }
}

/// What a shot did.
#[derive(Clone, Debug, PartialEq)]
pub struct ShotResult {
    pub label: String,
    /// Launch speed (m/s).
    pub launch_speed: f64,
    /// Its speed now (m/s), and along its launch direction (negative: it came back).
    pub speed: f64,
    pub along: f64,
    /// Past the far face of what it was fired at (the wall scenarios).
    pub through: bool,
    /// Bonds broken from its launch until the next shot's.
    pub broke: u32,
    /// The wall scenarios: its wall came down (the wall's top over where it struck dropped
    /// more than half a metre).
    pub wall_fell: bool,
}

/// A run's numbers so far.
#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    pub seconds: f64,
    pub bonds: u32,
    pub bonds_broken: u32,
    /// Seconds at the first break.
    pub first_break: Option<f64>,
    /// How far the watched part (the balcony's tip, the canopy) has dropped (m).
    pub drop: f64,
    /// The watched part came down (dropped more than half a metre).
    pub fell: bool,
    /// The fracture's own tallies (analytic strikes, the load path's peak utilization).
    pub report: Option<FractureReport>,
    pub shots: Vec<ShotResult>,
}

/// What a run's verdict is read from: the shots as they fly, the first break, the watched
/// part. Kept beside a placed [`Lab`]; [`Self::observe`] after each polled step.
pub struct Tracker {
    scenario: Scenario,
    /// Each launched shot, in launch order: its body, its last two positions, the bonds
    /// broken when it launched, where it was first seen.
    flying: Vec<(BodyId, [f32; 3], [f32; 3], u32, [f32; 3])>,
    broken: u32,
    first_break: Option<f64>,
    /// The watched chunk (destructible 0) and its height when placed.
    watch: (u32, f32),
    /// The wall scenarios: every chunk's (x, height) when placed (a shot's wall is the one
    /// whose top is over where the shot was launched).
    placed: Vec<[f32; 2]>,
}

impl Tracker {
    pub fn new(scenario: Scenario, lab: &Lab, world: &World) -> anyhow::Result<Tracker> {
        let chunks = world.destructible_stats(lab.destructibles()[0])?.chunks as usize;
        // The watched chunk: the highest, then farthest out along +x (the balcony's tip);
        // the pillar scenario watches the canopy's end over the pillar shot out (-x).
        // Chunk c is scene body 1 + c.
        let poses = lab.poses(world)?;
        let mut watch = (0u32, f32::MIN, f32::MIN);
        for c in 0..chunks {
            let o = RECORD_FLOATS * (1 + c);
            let x = if scenario == Scenario::Pillar { -poses[o] } else { poses[o] };
            if (poses[o + 1], x) > (watch.1, watch.2) {
                watch = (c as u32, poses[o + 1], x);
            }
        }
        // The wall scenarios: every chunk's (x, height) as placed.
        let placed = if matches!(scenario, Scenario::SpeedLadder | Scenario::EqualMomentum) {
            (0..chunks).map(|c| [poses[RECORD_FLOATS * (1 + c)], poses[RECORD_FLOATS * (1 + c) + 1]]).collect()
        } else {
            Vec::new()
        };
        Ok(Tracker {
            placed,
            scenario,
            flying: Vec::new(),
            broken: 0,
            first_break: None,
            watch: (watch.0, watch.1),
        })
    }

    /// The wall scenarios: the highest placed chunk nearest `x`, and its height then.
    fn top_over(&self, x: f32) -> Option<(usize, f32)> {
        let top = self.placed.iter().map(|p| p[1]).fold(f32::MIN, f32::max);
        self.placed
            .iter()
            .enumerate()
            .filter(|(_, p)| top - p[1] < 1e-3)
            .min_by(|a, b| (a.1[0] - x).abs().total_cmp(&(b.1[0] - x).abs()))
            .map(|(c, p)| (c, p[1]))
    }

    /// Take in the last polled step: new shots, each shot's motion, the breaks.
    pub fn observe(&mut self, lab: &Lab, world: &World) -> anyhow::Result<()> {
        let hz = self.scenario.physics_hz() as f64;
        let st = lab.stats(world)?;
        self.broken = st.bonds - st.bonds_intact;
        if self.broken > 0 && self.first_break.is_none() {
            self.first_break = Some(world.step_count() as f64 / hz);
        }
        for &(body, _) in lab.balls() {
            let Some(t) = world.body_transform(body)? else { continue };
            match self.flying.iter_mut().find(|f| f.0 == body) {
                Some(f) => {
                    if t.position != f.2 {
                        f.1 = f.2;
                        f.2 = t.position;
                    }
                }
                None => self.flying.push((body, t.position, t.position, self.broken, t.position)),
            }
        }
        Ok(())
    }

    /// The run's numbers as of the last [`Self::observe`]. `report`: with the fracture's
    /// own tallies (`World::destructible_fracture_report`, a blocking read: native hosts).
    pub fn verdict(&self, lab: &Lab, world: &World, report: bool) -> anyhow::Result<Verdict> {
        let hz = self.scenario.physics_hz() as f64;
        let st = lab.stats(world)?;
        let poses = lab.poses(world)?;
        let y = poses[RECORD_FLOATS * (1 + self.watch.0 as usize) + 1];
        let drop = (self.watch.1 - y) as f64;
        let report = if report { Some(world.destructible_fracture_report(lab.destructibles()[0])?) } else { None };
        let authored = lab.pack_shots();
        let wall = matches!(self.scenario, Scenario::SpeedLadder | Scenario::EqualMomentum);
        let mut shots = Vec::new();
        for (n, f) in self.flying.iter().enumerate() {
            let Some(s) = authored.get(n) else { break };
            let launch_speed = s.velocity.iter().map(|v| v * v).sum::<f64>().sqrt();
            let v: Vec<f64> = (0..3).map(|k| (f.2[k] - f.1[k]) as f64 * hz).collect();
            let speed = v.iter().map(|v| v * v).sum::<f64>().sqrt();
            let along = if launch_speed > 0.0 {
                (0..3).map(|k| v[k] * s.velocity[k]).sum::<f64>() / launch_speed
            } else {
                0.0
            };
            // The wall scenarios fire along -z at a wall WALL thick, from `origin[2]` in front
            // of its back face (the pack's frame): through once it has travelled past that.
            let travelled = (f.2[2] - f.4[2]).abs() as f64;
            let to_back = s.origin[2] + s.radius;
            shots.push(ShotResult {
                label: s.label.clone(),
                launch_speed,
                speed,
                along,
                through: wall && travelled > to_back,
                broke: self.flying.get(n + 1).map_or(self.broken, |next| next.3) - f.3,
                wall_fell: self.top_over(f.4[0]).is_some_and(|(c, y0)| y0 - poses[RECORD_FLOATS * (1 + c) + 1] > 0.5),
            });
        }
        Ok(Verdict {
            seconds: world.step_count() as f64 / hz,
            bonds: st.bonds,
            bonds_broken: self.broken,
            first_break: self.first_break,
            drop,
            fell: drop > 0.5,
            report,
            shots,
        })
    }
}

/// Queue a scenario's script on its lab: every authored shot at its time.
pub fn queue_script(scenario: Scenario, lab: &mut Lab, world: &World) -> anyhow::Result<()> {
    let hz = scenario.physics_hz() as f64;
    let shots: Vec<f64> = lab.pack_shots().iter().map(|s| s.launch_at).collect();
    for (i, at) in shots.into_iter().enumerate() {
        lab.queue(world, &LabAction::PackShot(i as u32), (at * hz).round() as u64)?;
    }
    Ok(())
}

/// A scenario placed under an approach in a new world (native hosts and tests).
pub struct Compare {
    pub scenario: Scenario,
    pub approach: Approach,
    pub lab: Lab,
    pub tracker: Tracker,
}

impl Compare {
    pub fn new(gpu: GpuContext, scenario: Scenario, approach: Approach) -> anyhow::Result<(World, Compare)> {
        let plan = LabPlan::new(&scenario.lab_def(approach))?;
        let mut world = World::new(gpu, &plan.world_def()?)?;
        let lab = Lab::place(&mut world, &plan)?;
        let tracker = Tracker::new(scenario, &lab, &world)?;
        Ok((world, Compare { scenario, approach, lab, tracker }))
    }

    pub fn queue_script(&mut self, world: &World) -> anyhow::Result<()> {
        queue_script(self.scenario, &mut self.lab, world)
    }

    /// One step, blocking for its frame, observed.
    pub fn step(&mut self, world: &mut World) -> anyhow::Result<()> {
        self.lab.step(world)?;
        world.poll_blocking()?;
        self.tracker.observe(&self.lab, world)
    }

    pub fn verdict(&self, world: &World) -> anyhow::Result<Verdict> {
        self.tracker.verdict(&self.lab, world, true)
    }
}
