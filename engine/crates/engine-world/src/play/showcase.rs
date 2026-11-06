//! The meteor run (`showcase.html`) as a [`super::SceneDef`]: a dune buggy on deformable
//! (MPM) sand, meteors striking around it. Its scenes: `drive` (the barrage beside the
//! road, near misses ahead of and behind the car, several areas open at once), `drive1`
//! (three strikes beside the road), `pass` (strikes kept clear of the track), `bts` (behind
//! the scenes: two strikes at 300 Hz, the engine's view beside the game's) and `play` (you
//! drive; a director drops meteors around you; click to drop your own).
//!
//! The page's own numbers and order throughout: its switches as `Number` reads them, the
//! values it handed the wasm terrain parser as f32 there, what it read from wasm JSON as
//! f32 widened, and its f64 script math.

use super::director::{note, note_with, Action, Event};
use super::hills::js_hypot;
use crate::*;
use engine_terrain::layout::MPM_MAX_AREAS;
use engine_terrain::surface::{
    SURFACE_VALID_WORD, SURFACE_VIEW_OPEN, SURFACE_VIEW_WINDOW, SURFACE_VIEW_WORD,
    SURFACE_VIEW_WORDS,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// Gravity (m/s²).
pub const G: f64 = 9.81;
/// The fireball's flight before contact (s): a strike lands this long after it is
/// scheduled (a play strike's `lead` instead, when it has one).
pub const APPROACH: f64 = 1.35;
/// The meteor's density (kg/m³).
const DENSITY: f64 = 3000.0;
const SCENES: [&str; 5] = ["drive", "drive1", "pass", "bts", "play"];

/// `showcase.html`'s switches: each number as the page's `num(k, d)` reads it (present and
/// finite, else none: the scene's default).
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct ShowcaseSceneDef {
    /// `drive`, `drive1`, `pass`, `bts` or `play`.
    pub scene: String,
    /// `game` or `physical` (`engine_terrain::TerrainProfile`).
    pub profile: String,
    /// `?rule=` (play: `strict`, else the local settle rule).
    pub rule: Option<String>,
    /// The vehicle preset.
    pub car: String,
    /// Play's random sequence (`?seed=`; the page picks one when absent).
    pub seed: Option<f64>,
    /// The meteor's radius (m) and speed (km/s).
    pub radius: Option<f64>,
    pub speed: Option<f64>,
    /// Cells across the transient radius.
    pub cpr: Option<f64>,
    /// The region's margin past the crater (× its radius).
    pub margin: Option<f64>,
    /// The rest speed areas bake at (m/s; none: 1% of √(gR)).
    pub settle: Option<f64>,
    /// Areas open at once (drive and play).
    pub regions: Option<f64>,
    /// Play's basin half side (m).
    pub half: Option<f64>,
    /// The track: clearance from a region box, straights' half length, turn radius.
    pub clear: Option<f64>,
    pub straight: Option<f64>,
    pub turn: Option<f64>,
    /// The drive scene's flat ground outside the loop (m).
    pub flatout: Option<f64>,
    /// The surface patches' samples across.
    pub patch: Option<f64>,
    /// The autopilot's speed (km/h).
    pub kmh: Option<f64>,
    /// The run-up before the recording (s).
    pub lead: Option<f64>,
    /// drive1: rim to track (m).
    pub gap: Option<f64>,
    /// The fixed schedules' times (s) and the second strike's turn on the first's rim.
    pub t1: Option<f64>,
    pub t2: Option<f64>,
    pub t3: Option<f64>,
    pub pass1: Option<f64>,
    pub pass2: Option<f64>,
    pub pass3: Option<f64>,
    pub turn2: Option<f64>,
    /// Play: the director keeps out of areas older than this (s).
    pub settle_after: Option<f64>,
}

impl Default for ShowcaseSceneDef {
    fn default() -> Self {
        ShowcaseSceneDef {
            scene: "drive".into(),
            profile: "game".into(),
            rule: None,
            car: "dune_buggy".into(),
            seed: None,
            radius: None,
            speed: None,
            cpr: None,
            margin: None,
            settle: None,
            regions: None,
            half: None,
            clear: None,
            straight: None,
            turn: None,
            flatout: None,
            patch: None,
            kmh: None,
            lead: None,
            gap: None,
            t1: None,
            t2: None,
            t3: None,
            pass1: None,
            pass2: None,
            pass3: None,
            turn2: None,
            settle_after: None,
        }
    }
}

/// A JS number handed to the wasm parser as an unsigned integer field (`Fields::u32`).
fn js_u32(name: &str, v: f64) -> anyhow::Result<u32> {
    anyhow::ensure!(
        v >= 0.0 && v.fract() == 0.0 && v <= 9007199254740991.0,
        "{name}: expected an unsigned integer"
    );
    Ok(v as u64 as u32)
}

/// `Math.round`: half up.
fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

/// `x >>> 0`.
fn js_to_u32(x: f64) -> u32 {
    if !x.is_finite() {
        return 0;
    }
    x.trunc().rem_euclid(4294967296.0) as u32
}

impl ShowcaseSceneDef {
    pub(crate) fn from_query(q: &super::scene::Query) -> anyhow::Result<Self> {
        let opt = |k: &str| {
            q.get(k)
                .map(super::scene::js_number)
                .filter(|n| n.is_finite())
        };
        let scene = q
            .get("scene")
            .filter(|s| SCENES.contains(s))
            .unwrap_or("drive");
        let profile = q.get("profile").unwrap_or("game");
        anyhow::ensure!(
            profile == "game" || profile == "physical",
            "TerrainConfig.profile: \"physical\" or \"game\", not {profile:?}"
        );
        Ok(ShowcaseSceneDef {
            scene: scene.into(),
            profile: profile.into(),
            rule: q.get("rule").map(Into::into),
            car: q
                .get("car")
                .filter(|s| !s.is_empty())
                .unwrap_or("dune_buggy")
                .into(),
            seed: opt("seed"),
            radius: opt("radius"),
            speed: opt("speed"),
            cpr: opt("cpr"),
            margin: opt("margin"),
            settle: opt("settle"),
            regions: opt("regions"),
            half: opt("half"),
            clear: opt("clear"),
            straight: opt("straight"),
            turn: opt("turn"),
            flatout: opt("flatout"),
            patch: opt("patch"),
            kmh: opt("kmh"),
            lead: opt("lead"),
            gap: opt("gap"),
            t1: opt("t1"),
            t2: opt("t2"),
            t3: opt("t3"),
            pass1: opt("pass1"),
            pass2: opt("pass2"),
            pass3: opt("pass3"),
            turn2: opt("turn2"),
            settle_after: opt("settleafter"),
        })
    }

    fn is(&self, s: &str) -> bool {
        self.scene == s
    }

    pub fn play(&self) -> bool {
        self.is("play")
    }

    /// The physics step (s): 300 Hz behind the scenes, else 120 Hz.
    pub fn dt(&self) -> f64 {
        if self.is("bts") {
            1.0 / 300.0
        } else {
            1.0 / 120.0
        }
    }

    fn radius(&self) -> f64 {
        self.radius.unwrap_or(1.0)
    }

    fn speed_ms(&self) -> f64 {
        self.speed.unwrap_or(20.0) * 1000.0
    }

    fn cpr(&self) -> f64 {
        self.cpr.unwrap_or(6.0)
    }

    /// The region margin as the page holds it (its scripts' number; the terrain gets it
    /// as an f32).
    fn margin(&self) -> f64 {
        self.margin.unwrap_or(1.0)
    }

    /// Areas open at once, where the scene sets it (otherwise the profile's).
    fn max_regions(&self) -> Option<f64> {
        if self.is("drive") {
            Some(self.regions.unwrap_or(8.0))
        } else if self.play() {
            Some(self.regions.unwrap_or(12.0))
        } else {
            None
        }
    }

    fn patch(&self) -> f64 {
        self.patch.unwrap_or(if self.is("drive") || self.play() {
            128.0
        } else {
            72.0
        })
    }

    fn profile(&self) -> TerrainProfile {
        if self.profile == "physical" {
            TerrainProfile::Physical
        } else {
            TerrainProfile::Game
        }
    }

    /// The terrain's settings as the page builds `TERRAIN.settings` (with `patch`: the
    /// surface patches' samples, set once the scene is known).
    pub fn terrain_def(&self, settle_speed: f32, patch: bool) -> anyhow::Result<TerrainDef> {
        let mut c = TerrainConfig::profile(self.profile());
        if let Some(r) = self.max_regions() {
            c.max_regions = js_u32("TerrainConfig.maxRegions", r)?;
        }
        c.max_particles = 1 << 20;
        c.max_blocks = 8192;
        c.max_parcels = 1 << 17;
        c.max_substeps = 1024;
        c.max_columns = 1 << 16;
        c.max_field_samples = 1 << 19;
        // The dune buggy drives into open regions: its 17 parts' colliders couple.
        c.max_coupled = 256;
        c.cells_per_radius = self.cpr() as f32;
        let mut surface = TerrainSurfaceDef {
            cells: 1 << 21,
            max_debris: 0,
            ..TerrainSurfaceDef::default()
        };
        if patch {
            surface.patch_samples = js_u32("TerrainSurfaceDef.patchSamples", self.patch())?;
        }
        let mut d = TerrainDef {
            config: c,
            region_margin: f64::from(self.margin() as f32),
            settle_speed,
            settle_frames: 60,
            surface: Some(surface),
            ..TerrainDef::default()
        };
        if self.play() {
            d.settle_rule = if self.rule.as_deref() == Some("strict") {
                SettleRule::Strict
            } else {
                SettleRule::Local
            };
        }
        Ok(d)
    }

    /// The meteor's crater (π-scaling), its mass and energy.
    pub fn prediction(&self) -> anyhow::Result<(TerrainPrediction, f64, f64)> {
        let d = self.terrain_def(0.05, false)?;
        let src = d.vertical_impact(self.radius(), DENSITY, self.speed_ms())?;
        Ok((d.predict_impact(&src, G)?, src.mass(), src.kinetic_energy()))
    }

    /// The page's `TERRAIN` (`{ meteor, settings }`, its own numbers), for its HUD and
    /// automation.
    fn terrain_json(&self, settle: f64) -> serde_json::Value {
        let mut config = serde_json::json!({
            "maxParticles": 1 << 20, "maxBlocks": 8192, "maxParcels": 1 << 17, "maxSubsteps": 1024,
            "maxColumns": 1 << 16, "maxFieldSamples": 1 << 19, "maxCoupled": 256,
            "cellsPerRadius": self.cpr(), "profile": self.profile,
        });
        if let Some(r) = self.max_regions() {
            config["maxRegions"] = r.into();
        }
        let mut settings = serde_json::json!({
            "config": config, "regionMargin": self.margin(), "settleSpeed": settle,
            "settleFrames": 60,
            "surface": { "cells": 1 << 21, "maxDebris": 0, "patchSamples": self.patch() },
        });
        if self.play() {
            settings["settleRule"] = if self.rule.as_deref() == Some("strict") {
                "strict"
            } else {
                "local"
            }
            .into();
        }
        serde_json::json!({
            "meteor": { "radius": self.radius(), "density": DENSITY, "speed": self.speed_ms() },
            "settings": settings,
        })
    }
}

pub fn world_def(_def: &ShowcaseSceneDef) -> anyhow::Result<WorldDef> {
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
            max_parts: 64,
            max_rays: 16,
        },
        readback_frames: 4,
        ..WorldDef::default()
    })
}

/// What the page reads of a placed meteor run.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Placed {
    /// The heightfield shape and its grid (cell `sp` m, corner `x0`, square).
    pub ground: u64,
    pub nx: u32,
    pub nz: u32,
    pub sp: f64,
    pub x0: f64,
    /// The car (vehicle handle).
    pub car: u32,
    /// The crater: transient radius, final rim radius, the region's cell (m); how far an
    /// open box reaches from its strike.
    pub radius: f64,
    pub rim: f64,
    pub dx: f64,
    pub reach: f64,
    /// The road: the straights' half length, the turns' radius, the clearance, the
    /// autopilot's speed (m/s); the run-up (s) and the physics step (s).
    pub straight: f64,
    pub turn: f64,
    pub clear: f64,
    pub speed: f64,
    pub lead: f64,
    pub dt: f64,
    /// The basin's half side (m), the flat ground past the loop (m).
    pub half: f64,
    pub flat_out: f64,
    /// The fireball's approach (s).
    pub approach: f64,
    /// The first strike's point (x, z), when it is fixed before the run.
    pub first_point: Option<[f64; 2]>,
    /// The page's `TERRAIN` (`{ meteor, settings }`) and the prediction (as
    /// `terrainPredictImpact` returns it).
    pub terrain: serde_json::Value,
    pub prediction: serde_json::Value,
}

/// The road and the field: everything strike points and the autopilot need.
#[derive(Clone, Debug)]
struct Layout {
    play: bool,
    r: f64,
    rim: f64,
    block: f64,
    margin: f64,
    clear: f64,
    straight: f64,
    turn: f64,
    flat_out: f64,
    half: f64,
    speed: f64,
    runup: f64,
    track: Vec<[f64; 2]>,
}

impl Layout {
    fn new(def: &ShowcaseSceneDef, r: f64, rim: f64, dx: f64) -> anyhow::Result<Layout> {
        let clear = def.clear.unwrap_or(9.0);
        let block = 4.0 * dx;
        let straight = def.straight.unwrap_or(90.0);
        let turn = def
            .turn
            .unwrap_or(((58.0 * r / 15.1 - clear) / block).floor() * block + clear + 0.5);
        anyhow::ensure!(
            turn > 0.0,
            "the track's turn radius must be positive ({turn})"
        );
        let mut track = Vec::new();
        let mut x = -straight;
        while x < straight {
            track.push([x, turn]);
            x += 1.0;
        }
        let mut a = 0.0f64;
        while a < std::f64::consts::PI {
            track.push([straight + turn * a.sin(), turn * a.cos()]);
            a += 1.0 / turn;
        }
        let mut x = straight;
        while x > -straight {
            track.push([x, -turn]);
            x -= 1.0;
        }
        let mut a = 0.0f64;
        while a < std::f64::consts::PI {
            track.push([-straight - turn * a.sin(), -turn * a.cos()]);
            a += 1.0 / turn;
        }
        Ok(Layout {
            play: def.play(),
            r,
            rim,
            block,
            margin: def.margin(),
            clear,
            straight,
            turn,
            flat_out: if def.is("drive") {
                def.flatout.unwrap_or(75.0)
            } else {
                14.0
            },
            half: if def.play() {
                def.half.unwrap_or(600.0)
            } else {
                320.0
            },
            speed: def.kmh.unwrap_or(60.0) / 3.6,
            runup: if def.is("drive") { 1.65 } else { 0.9 },
            track,
        })
    }

    /// Distance from the loop's spine (the segment between the turns' centres).
    fn spine(&self, x: f64, z: f64) -> f64 {
        js_hypot(&[x - x.min(self.straight).max(-self.straight), z])
    }

    /// The ground: dunes, flattened over the strike field; play: a wide basin rimmed by
    /// hills.
    fn dune(&self, x: f64, z: f64) -> f64 {
        let big = 2.6 * (x / 47.0 + 0.7).sin() * (z / 61.0 - 0.4).cos()
            + 1.6 * ((x + 1.7 * z) / 37.0).sin();
        let ridge = 1.2 * ((0.8 * x - 0.6 * z) / 11.0 + (z / 29.0).sin() * 2.0).sin();
        if self.play {
            let edge = ((x.abs().max(z.abs()) - (self.half - 140.0)) / 90.0)
                .max(0.0)
                .min(1.0);
            return 0.3 * (big + 0.35 * ridge)
                + edge * edge * 30.0 * (0.7 + 0.3 * (x / 41.0).sin() * (z / 37.0).cos());
        }
        let r = self.spine(x, z);
        let field = ((r - self.turn - self.flat_out) / 40.0).max(0.0).min(1.0);
        let far = ((r - self.turn - self.flat_out - 16.0) / 80.0)
            .max(0.0)
            .min(1.0)
            * 9.0
            * (0.6 + 0.4 * (x / 53.0).sin() * (z / 71.0).cos());
        field * (big + 0.35 * ridge) * 0.6 + far
    }

    /// Where the car is at sim time `t` (it reaches speed about `runup` s in).
    fn track_at(&self, t: f64) -> [f64; 2] {
        let k = js_round(self.speed * (t - self.runup).max(0.0));
        let k = k.max(0.0).min((self.track.len() - 1) as f64);
        self.track[k as usize]
    }

    /// The box a strike at (x, z) opens (`RegionPlan::around`: the padded radius rounded
    /// out to whole blocks).
    fn box_of(&self, x: f64, z: f64) -> [f64; 4] {
        let reach = self.r * (1.0 + self.margin);
        let b = self.block;
        [
            ((x - reach) / b).floor() * b,
            ((z - reach) / b).floor() * b,
            ((x + reach) / b).ceil() * b,
            ((z + reach) / b).ceil() * b,
        ]
    }

    /// How close the track comes to a box.
    fn track_clearance(&self, [x0, z0, x1, z1]: [f64; 4]) -> f64 {
        let mut d = f64::INFINITY;
        for &[x, z] in &self.track {
            d = d.min(js_hypot(&[
                (x0 - x).max(0.0).max(x - x1),
                (z0 - z).max(0.0).max(z - z1),
            ]));
        }
        d
    }
}

/// The heightfield's grid: its cells the region's, out to `half`.
fn grid(half: f64, sp: f64) -> (u32, f64) {
    let nx = 2 * (half / sp).ceil() as u32 + 1;
    (nx, -0.5 * (nx as f64 - 1.0) * sp)
}

/// The ground's height under (x, z), bilinear in the samples (the page's `groundAt`).
fn ground_at(h: &[f32], nx: u32, nz: u32, x0: f64, sp: f64, x: f64, z: f64) -> f64 {
    let fx = ((x - x0) / sp).max(0.0).min(nx as f64 - 1.001);
    let fz = ((z - x0) / sp).max(0.0).min(nz as f64 - 1.001);
    let (i, j) = (fx.floor(), fz.floor());
    let (u, v) = (fx - i, fz - j);
    let (i, j) = (i as usize, j as usize);
    let at = |a: usize, b: usize| h[(j + b) * nx as usize + i + a] as f64;
    (1.0 - v) * ((1.0 - u) * at(0, 0) + u * at(1, 0)) + v * ((1.0 - u) * at(0, 1) + u * at(1, 1))
}

fn layout_of(def: &ShowcaseSceneDef, p: &Placed) -> anyhow::Result<Layout> {
    Layout::new(def, p.radius, p.rim, p.dx)
}

pub fn place(world: &mut World, def: &ShowcaseSceneDef) -> anyhow::Result<Placed> {
    let (pred, mass, energy) = def.prediction()?;
    let r = pred.crater.radius;
    let rim = pred.crater.final_rim_radius;
    // The page reads the cell from JSON: the f32, widened.
    let dx = pred.dx as f64;
    let settle = def.settle.unwrap_or(0.01 * (G * r).sqrt());
    let layout = Layout::new(def, r, rim, dx)?;
    let sp = dx;
    let (nx, x0) = grid(layout.half, sp);
    let nz = nx;
    let mut heights = Vec::with_capacity((nx * nz) as usize);
    for j in 0..nz {
        for i in 0..nx {
            heights.push(layout.dune(x0 + i as f64 * sp, x0 + j as f64 * sp) as f32);
        }
    }
    let mut solver = world.solver();
    solver.dt = def.dt() as f32;
    solver.iterations = 8;
    solver.gravity_y = -G as f32;
    solver.flags = engine_schema::FLAG_SPEC_VELOCITY;
    world.set_solver(solver);
    // The material table: slot 0 (the ground, the patches and the buggy's part colliders)
    // and the wheel rims at 0.9, the rest 0.6.
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
    world.enable_terrain(&def.terrain_def(settle as f32, true)?)?;
    // The car: on the first straight facing +x (play: at the basin's centre).
    let [x, z] = if def.play() {
        [0.0, 0.0]
    } else {
        layout.track[0]
    };
    let y = ground_at(&heights, nx, nz, x0, sp, x, z) + 1.4;
    let desc = engine_vehicle::presets::all()
        .into_iter()
        .find(|d| d.name == def.car)
        .ok_or_else(|| anyhow::anyhow!("no vehicle preset {}", def.car))?;
    let yaw = std::f64::consts::PI / 2.0;
    let car = world.add_vehicle(
        &desc,
        [x as f32, y as f32, z as f32],
        [0.0, (yaw / 2.0).sin() as f32, 0.0, (yaw / 2.0).cos() as f32],
    )?;
    let schedule = schedule_of(def);
    let first_point = schedule.first().and_then(|s| match s.at {
        Some(a) => Some(a),
        None if s.rim_of.is_none() => strike_point(&layout, &schedule, 0, &def.scene).ok(),
        None => None,
    });
    let c = &pred.crater;
    let prediction = serde_json::json!({
        "crater": {
            "radius": c.radius, "depth": c.depth, "rimRadius": c.rim_radius,
            "finalRimRadius": c.final_rim_radius, "volume": c.volume,
            "formationTime": c.formation_time, "complex": c.complex, "piV": c.pi_v,
            "regime": format!("{:?}", c.regime),
        },
        "dx": pred.dx, "blocks": pred.blocks, "sourceSpeed": pred.source_speed,
        "mass": mass, "energy": energy,
    });
    Ok(Placed {
        ground: ground.to_bits(),
        nx,
        nz,
        sp,
        x0,
        car: car.0,
        radius: r,
        rim,
        dx,
        reach: r * (1.0 + layout.margin) + 4.0 * dx,
        straight: layout.straight,
        turn: layout.turn,
        clear: layout.clear,
        speed: layout.speed,
        lead: def.lead.unwrap_or(if def.is("bts") { 0.0 } else { 1.5 }),
        dt: def.dt(),
        half: layout.half,
        flat_out: layout.flat_out,
        approach: APPROACH,
        first_point,
        terrain: def.terrain_json(settle),
        prediction,
    })
}

// ------------------------------------------------------------------------------ script

#[derive(Clone, Copy, Debug, PartialEq)]
enum St {
    Wait,
    Approach,
    Done,
    Fizzled,
}

/// A read of the live surface's words in flight (the page's `readTerrainSurface`).
struct SurfaceRead {
    staging: wgpu::Buffer,
    mapped: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>,
}

impl SurfaceRead {
    fn begin(world: &World, word: u32, n: u32) -> SurfaceRead {
        let device = world.device();
        let bytes = 4 * u64::from(n.max(1));
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("showcase surface read"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(
            world.sim().terrain_surface_buffer(),
            4 * u64::from(word),
            &staging,
            0,
            bytes,
        );
        world.queue().submit([enc.finish()]);
        let mapped = Arc::new(Mutex::new(None));
        let m = mapped.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            *m.lock().unwrap() = Some(r);
        });
        SurfaceRead { staging, mapped }
    }

    /// The words, once landed (never waits).
    fn try_take(&self, world: &World) -> anyhow::Result<Option<Vec<u32>>> {
        #[cfg(not(target_arch = "wasm32"))]
        let _ = world.device().poll(wgpu::PollType::Poll);
        #[cfg(target_arch = "wasm32")]
        let _ = world;
        match self.mapped.lock().unwrap().as_ref() {
            None => return Ok(None),
            Some(Err(e)) => anyhow::bail!("surface read: {e}"),
            Some(Ok(())) => {}
        }
        let out = {
            let data = self
                .staging
                .slice(..)
                .get_mapped_range()
                .map_err(|e| anyhow::anyhow!("surface read: {e:?}"))?;
            bytemuck::pod_collect_to_vec::<u8, u32>(&data)
        };
        self.staging.unmap();
        Ok(Some(out))
    }
}

/// The live height read before contact: the surface's header, then the one sample.
enum LiveRead {
    Header(SurfaceRead),
    Height(SurfaceRead),
}

/// A strike on a schedule (fixed, or the play director's and the player's).
struct Strike {
    id: u32,
    /// When it is due (s), and when once regions may overlap.
    t: f64,
    overlap_t: Option<f64>,
    /// `pass`: beside where the car is then; `side` 1 inside the loop, −1 outside;
    /// `beside`: the rim that far from the track.
    pass: Option<f64>,
    side: Option<f64>,
    beside: Option<f64>,
    /// On strike `rim_of`'s rim, turned `rim_turn` from +x toward the field.
    rim_of: Option<usize>,
    rim_turn: f64,
    /// A fixed point (x, z).
    at: Option<[f64; 2]>,
    /// Play: whose (`you` or `sky`), and a longer warning than the approach.
    who: Option<&'static str>,
    lead: Option<f64>,
    state: St,
    point: Option<[f64; 3]>,
    t0: Option<f64>,
    struck_at: Option<f64>,
    /// The fireball's approach direction (cosmetic; set as it is scheduled).
    yaw: f64,
    pitch: f64,
    live_read: bool,
    read: Option<LiveRead>,
}

impl Strike {
    fn new(t: f64) -> Strike {
        Strike {
            id: 0,
            t,
            overlap_t: None,
            pass: None,
            side: None,
            beside: None,
            rim_of: None,
            rim_turn: 0.0,
            at: None,
            who: None,
            lead: None,
            state: St::Wait,
            point: None,
            t0: None,
            struck_at: None,
            yaw: 0.0,
            pitch: 0.0,
            live_read: false,
            read: None,
        }
    }
}

/// The page's `SCHEDULES[SCENE]`.
fn schedule_of(def: &ShowcaseSceneDef) -> Vec<Strike> {
    let lead = def.lead.unwrap_or(if def.is("bts") { 0.0 } else { 1.5 });
    let n = |v: Option<f64>, d: f64| v.unwrap_or(d);
    let pass = |t: f64, p: f64| Strike {
        pass: Some(p),
        ..Strike::new(t)
    };
    let rim = |t: f64, overlap: f64, of: usize, turn: f64| Strike {
        overlap_t: Some(overlap),
        rim_of: Some(of),
        rim_turn: turn,
        ..Strike::new(t)
    };
    let beside = |t: f64, p: f64, side: Option<f64>, b: f64| Strike {
        pass: Some(p),
        side,
        beside: Some(b),
        ..Strike::new(t)
    };
    let mut s = match def.scene.as_str() {
        "pass" => vec![
            pass(lead + n(def.t1, 1.2), lead + n(def.pass1, 5.0)),
            rim(lead + n(def.t2, 7.0), lead + 3.4, 0, n(def.turn2, 0.45)),
            pass(lead + n(def.t3, 13.5), lead + n(def.pass3, 17.0)),
        ],
        "drive" => [
            (0.5, 2.9, 1.0, 2.0),
            (1.9, 3.9, -1.0, -1.0),
            (4.4, 4.8, 1.0, 4.0),
            (6.5, 5.2, -1.0, 9.0),
            (6.5, 8.9, 1.0, -1.0),
            (8.0, 10.1, -1.0, 3.0),
            (11.1, 11.5, 1.0, 3.0),
            (12.5, 14.7, -1.0, 2.0),
            (14.7, 13.7, 1.0, 3.0),
            (15.5, 17.9, 1.0, 4.0),
            (17.1, 17.5, -1.0, 2.0),
        ]
        .into_iter()
        .map(|(t, p, side, b)| beside(lead + t, lead + p, Some(side), b))
        .collect(),
        "drive1" => {
            let gap = n(def.gap, 3.0);
            vec![
                beside(lead + n(def.t1, 1.0), lead + n(def.pass1, 4.6), None, gap),
                Strike {
                    overlap_t: Some(lead + 3.0),
                    ..beside(lead + n(def.t2, 7.0), lead + n(def.pass2, 10.6), None, gap)
                },
                beside(lead + n(def.t3, 13.0), lead + n(def.pass3, 16.6), None, gap),
            ]
        }
        "bts" => vec![
            Strike {
                at: Some([0.0, 20.0]),
                ..Strike::new(n(def.t1, 1.5))
            },
            rim(n(def.t2, 5.6), 2.6, 0, n(def.turn2, 0.45)),
        ],
        _ => Vec::new(),
    };
    for (k, x) in s.iter_mut().enumerate() {
        x.id = k as u32;
    }
    s
}

/// Where strike `i` lands (x, z), as the page's `strikePoint`.
fn strike_point(l: &Layout, sched: &[Strike], i: usize, scene: &str) -> anyhow::Result<[f64; 2]> {
    let s = &sched[i];
    if let Some(at) = s.at {
        return Ok(at);
    }
    let pass = || {
        s.pass
            .ok_or_else(|| anyhow::anyhow!("strike {i} has no pass time"))
    };
    if let Some(b) = s.beside {
        // Beside where the car will pass: the rim `beside` m from the track (inside the
        // loop; `side` −1: outside it, away from the spine).
        let [ax, az] = l.track_at(pass()?);
        let sx = ax.min(l.straight).max(-l.straight);
        let d = js_hypot(&[ax - sx, az]);
        let off = (l.rim + b) * s.side.unwrap_or(1.0);
        return Ok([ax + (sx - ax) / d * off, az - az / d * off]);
    }
    if let Some(r) = s.rim_of {
        let o = sched[r]
            .point
            .ok_or_else(|| anyhow::anyhow!("strike {i}: strike {r} has no point yet"))?;
        let (x, mut z) = (
            o[0] + l.rim * s.rim_turn.cos(),
            o[2] - l.rim * s.rim_turn.sin(),
        );
        if scene != "pass" {
            return Ok([x, z]);
        }
        // Into the field until its box clears the track.
        let mut k = 0;
        while k < 400 && l.track_clearance(l.box_of(x, z)) < l.clear {
            let sign = if z == 0.0 || z.is_nan() {
                1.0
            } else {
                z.signum()
            };
            z -= 0.5 * sign;
            k += 1;
        }
        return Ok([x, z]);
    }
    // From the loop's spine toward where the car passes, while the box still clears the
    // track.
    let [ax, az] = l.track_at(pass()?);
    let sx = ax.min(l.straight).max(-l.straight);
    let d = js_hypot(&[ax - sx, az]);
    let u = [(ax - sx) / d, az / d];
    let mut f = 0.0;
    while f + 0.5 < d
        && l.track_clearance(l.box_of(sx + u[0] * (f + 0.5), u[1] * (f + 0.5))) >= l.clear
    {
        f += 0.5;
    }
    Ok([sx + u[0] * f, u[1] * f])
}

/// The car as last read: position, forward, up, velocity.
#[derive(Clone, Copy, Debug)]
struct Pose {
    p: [f64; 3],
    fwd: [f64; 3],
    up: [f64; 3],
    v: [f64; 3],
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

/// mulberry32, as the page's `rand`.
fn mulberry32(seed: &mut u32) -> f64 {
    *seed = seed.wrapping_add(0x6D2B79F5);
    let mut t = *seed;
    t = (t ^ (t >> 15)).wrapping_mul(t | 1);
    t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
    (t ^ (t >> 14)) as f64 / 4294967296.0
}

fn norm3(a: [f64; 3]) -> [f64; 3] {
    let l = js_hypot(&a);
    let l = if l == 0.0 || l.is_nan() { 1.0 } else { l };
    a.map(|x| x / l)
}

/// A fireball shown: its strike, where and when it lands, where it comes from.
#[derive(Clone, Copy, Debug)]
struct Fx {
    id: u32,
    point: [f64; 3],
    t0: f64,
    yaw: f64,
    pitch: f64,
}

/// An open area as play tracks it: since when, and its box.
#[derive(Clone, Copy, Debug)]
struct Opened {
    slot: u32,
    t: f64,
    lo: [f64; 3],
    hi: [f64; 3],
}

/// Play's run: how it ends, its score and the director's clock.
#[derive(Clone, Debug, Default)]
struct Game {
    over: Option<(&'static str, f64)>,
    near: u32,
    dropped: u32,
    survived: f64,
    next: f64,
    msg: String,
    msg_t: f64,
    flip_t: f64,
    stuck_t: f64,
    throttle: bool,
    trail: Vec<[f64; 3]>,
    opened: Vec<Opened>,
}

impl Game {
    fn new() -> Game {
        Game {
            next: 3.0,
            msg_t: -1e9,
            ..Game::default()
        }
    }
}

/// The meteor run's script: the strikes on their schedule (each scheduled an approach
/// before it lands, on the live surface's height where an area is open, bakes forced when
/// only one area may be open), the autopilot round the track, and play's director, the
/// player's strikes and controls, and how a run ends.
pub struct ShowcaseDirector {
    def: ShowcaseSceneDef,
    placed: Placed,
    layout: Layout,
    dt: f64,
    schedule: Vec<Strike>,
    next_id: u32,
    /// The fireballs shown, oldest first (kept after their strike leaves the schedule,
    /// as the page's were).
    fx: Vec<Fx>,
    t: f64,
    frame: u64,
    last_strike: f64,
    vehicle: VehicleHandle,
    chassis: Option<BodyId>,
    car_read: Option<BodyRead>,
    pose: Option<Pose>,
    car_speed: f64,
    /// The terrain report as the page holds it: none while the world's last one is this
    /// (the page dropped it at a reset, or at a strike with nothing open).
    report_dropped: Option<Option<TerrainStepReport>>,
    game: Game,
    /// The player's controls (play): throttle, the steering asked for, handbrake.
    controls: (f64, f64, f64),
    steer_now: f64,
    seed: u32,
}

impl ShowcaseDirector {
    pub(crate) fn new(
        def: &ShowcaseSceneDef,
        placed: Placed,
        world: &World,
    ) -> anyhow::Result<Self> {
        let layout = layout_of(def, &placed)?;
        let schedule = schedule_of(def);
        let vehicle = VehicleHandle(placed.car);
        let slot = world.sim().vehicle_host().chassis_body(vehicle);
        Ok(ShowcaseDirector {
            dt: def.dt(),
            next_id: schedule.len() as u32,
            schedule,
            fx: Vec::new(),
            t: 0.0,
            frame: 0,
            last_strike: -1e9,
            vehicle,
            chassis: world.body_at(slot),
            car_read: None,
            pose: None,
            car_speed: 0.0,
            report_dropped: None,
            game: Game::new(),
            controls: (0.0, 0.0, 0.0),
            steer_now: 0.0,
            seed: js_to_u32(def.seed.unwrap_or(0.0)),
            def: def.clone(),
            placed,
            layout,
        })
    }

    fn play(&self) -> bool {
        self.layout.play
    }

    fn rand(&mut self) -> f64 {
        mulberry32(&mut self.seed)
    }

    fn can_overlap(world: &World) -> bool {
        world
            .terrain_settings()
            .is_some_and(|d| d.config.max_regions > 1)
    }

    fn regions(world: &World) -> usize {
        world
            .terrain_settings()
            .map_or(1, |d| d.config.max_regions as usize)
    }

    /// The terrain report as the page holds it.
    fn report(&self, world: &World) -> Option<TerrainStepReport> {
        let r = world.terrain_last_report().copied();
        match self.report_dropped {
            Some(d) if d == r => None,
            _ => r,
        }
    }

    /// The page's `heights` (the heightfield as last baked) under (x, z).
    fn ground_at(&self, world: &World, x: f64, z: f64) -> anyhow::Result<f64> {
        let p = &self.placed;
        match world.shape_geometry(ShapeId::from_bits(p.ground))? {
            Geometry::HeightField { field, .. } => {
                Ok(ground_at(field.heights(), p.nx, p.nz, p.x0, p.sp, x, z))
            }
            _ => anyhow::bail!("the ground is not a heightfield"),
        }
    }

    fn say(&mut self, m: impl Into<String>) {
        self.game.msg = m.into();
        self.game.msg_t = self.t;
    }

    pub(crate) fn before_step(
        &mut self,
        world: &mut World,
        ev: &mut Vec<Event>,
    ) -> anyhow::Result<()> {
        // What landed since the last step: the live heights, the car's state.
        self.take_live_reads(world)?;
        if let Some(r) = &self.car_read {
            if let Some(rec) = r.try_take(world)? {
                let b = rec[0];
                let q = b.rotation.map(|x| x as f64);
                let v = b.linear_velocity.map(|x| x as f64);
                self.pose = Some(Pose {
                    p: b.position.map(|x| x as f64),
                    fwd: rot(q, [0.0, 0.0, 1.0]),
                    up: rot(q, [0.0, 1.0, 0.0]),
                    v,
                });
                self.car_speed = js_hypot(&v);
                self.car_read = None;
            }
        }
        if let Some(d) = self.report_dropped {
            if d != world.terrain_last_report().copied() {
                self.report_dropped = None;
            }
        }
        if self.play() {
            self.play_tick(world, ev)?;
        }
        self.schedule(world, ev)?;
        self.drive_input(world);
        // One read of the chassis in flight.
        if self.car_read.is_none() {
            if let Some(c) = self.chassis {
                self.car_read = Some(world.begin_body_read(&[c])?);
            }
        }
        self.t += self.dt;
        self.frame += 1;
        Ok(())
    }

    /// The live surface's height at a strike's point, read just before contact: the
    /// surface's header, then the sample of the first open view over it.
    fn take_live_reads(&mut self, world: &World) -> anyhow::Result<()> {
        let (x0, sp) = (self.placed.x0, self.placed.sp);
        let fx = &mut self.fx;
        for s in &mut self.schedule {
            let Some(read) = &s.read else {
                continue;
            };
            match read {
                LiveRead::Header(r) => {
                    let Some(head) = r.try_take(world)? else {
                        continue;
                    };
                    s.read = None;
                    if head[SURFACE_VALID_WORD as usize] != 1 {
                        continue;
                    }
                    let p = s.point.unwrap_or([0.0; 3]);
                    for v in 0..MPM_MAX_AREAS as usize {
                        let at = (SURFACE_VIEW_WORD + SURFACE_VIEW_WORDS * v as u32) as usize;
                        let view = &head[at..at + SURFACE_VIEW_WORDS as usize];
                        if view[0] & SURFACE_VIEW_OPEN == 0 {
                            continue;
                        }
                        let w = SURFACE_VIEW_WINDOW as usize;
                        let [i0, j0, ni, nj] =
                            [view[w], view[w + 1], view[w + 2], view[w + 3]].map(f64::from);
                        let ti = js_round((p[0] - x0) / sp) - i0;
                        let tj = js_round((p[2] - x0) / sp) - j0;
                        if ti < 0.0 || tj < 0.0 || ti >= ni || tj >= nj {
                            continue;
                        }
                        // view[2]: the view's heights' first word.
                        let word = f64::from(view[2]) + tj * ni + ti;
                        s.read = Some(LiveRead::Height(SurfaceRead::begin(world, word as u32, 1)));
                        break;
                    }
                }
                LiveRead::Height(r) => {
                    let Some(h) = r.try_take(world)? else {
                        continue;
                    };
                    s.read = None;
                    if s.state == St::Approach {
                        if let Some(p) = &mut s.point {
                            p[1] = f32::from_bits(h[0]) as f64;
                            for f in fx.iter_mut().filter(|f| f.id == s.id) {
                                f.point = *p;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// The strikes due: scheduled an approach before (or bakes first, one area at a
    /// time), then struck.
    fn schedule(&mut self, world: &mut World, ev: &mut Vec<Event>) -> anyhow::Result<()> {
        let overlap = Self::can_overlap(world);
        for i in 0..self.schedule.len() {
            if self.schedule[i].state == St::Done {
                continue;
            }
            if self.schedule[i].state == St::Wait {
                let s = &self.schedule[i];
                let due = match s.overlap_t {
                    Some(o) if overlap => o,
                    _ => s.t,
                };
                if self.t < due - APPROACH {
                    continue;
                }
                if !world.terrain_areas().is_empty() && !overlap {
                    // One region at a time: bake the open one first, once nothing is in
                    // flight; the fireball starts next frame, on the baked heights.
                    if self.report(world).map_or(1, |r| r.airborne) > 0 {
                        continue;
                    }
                    note(ev, world, "close");
                    world.terrain_close()?;
                    continue;
                }
                let [x, z] = strike_point(&self.layout, &self.schedule, i, &self.def.scene)?;
                let point = [x, self.ground_at(world, x, z)?, z];
                // Coming in from beyond the strike as the car sees it, and from the side.
                let c = self.pose.map_or([0.0; 3], |p| p.p);
                let away = norm3([point[0] - c[0], 0.0, point[2] - c[2]]);
                let side = [-away[2], 0.0, away[0]];
                let d = [
                    0.85 * away[0] + 0.55 * side[0],
                    0.0,
                    0.85 * away[2] + 0.55 * side[2],
                ];
                let t = self.t;
                let s = &mut self.schedule[i];
                s.point = Some(point);
                s.t0 = Some(t + s.lead.unwrap_or(APPROACH));
                s.yaw = d[0].atan2(d[2]);
                s.pitch = 0.5;
                s.state = St::Approach;
                let f = Fx {
                    id: s.id,
                    point,
                    t0: t + s.lead.unwrap_or(APPROACH),
                    yaw: s.yaw,
                    pitch: s.pitch,
                };
                self.fx.push(f);
            }
            let s = &self.schedule[i];
            let (t0, point) = match (s.t0, s.point) {
                (Some(t0), Some(p)) => (t0, p),
                _ => continue,
            };
            // Over an open area the ground is the live surface (a rim still moving): its
            // height there, read back just before contact.
            if s.state == St::Approach
                && !s.live_read
                && !world.terrain_areas().is_empty()
                && self.t > t0 - 0.25
            {
                let words = SURFACE_VIEW_WORD + SURFACE_VIEW_WORDS * MPM_MAX_AREAS;
                let s = &mut self.schedule[i];
                s.live_read = true;
                s.read = Some(LiveRead::Header(SurfaceRead::begin(world, 0, words)));
            }
            if self.schedule[i].state == St::Approach && self.t >= t0 - 1e-9 {
                let areas = world.terrain_areas();
                anyhow::ensure!(
                    areas.is_empty() || overlap,
                    "a strike is due while a region is open"
                );
                if self.play() && !self.play_can_open(world, point[0], point[2]) {
                    // No area slot or pool room left by contact: it burns up.
                    let t = self.t;
                    let s = &mut self.schedule[i];
                    s.state = St::Fizzled;
                    s.struck_at = Some(t);
                    let id = s.id;
                    self.fx.retain(|f| f.id != id);
                    self.say("it burned up");
                    continue;
                }
                if areas.is_empty() {
                    self.report_dropped = Some(world.terrain_last_report().copied());
                }
                let src = world
                    .terrain_settings()
                    .ok_or_else(|| anyhow::anyhow!("terrain is not enabled"))?
                    .vertical_impact(self.def.radius(), DENSITY, self.def.speed_ms())?;
                world.terrain_impact(
                    ShapeId::from_bits(self.placed.ground),
                    &src,
                    point.map(|v| v as f32),
                )?;
                let t = self.t;
                let s = &mut self.schedule[i];
                s.state = St::Done;
                s.struck_at = Some(t);
                let who = s.who;
                self.last_strike = t;
                let region = world
                    .terrain_areas()
                    .first()
                    .map(|a| serde_json::json!({ "window": a.window, "dx": a.plan.dx }));
                note_with(
                    ev,
                    world,
                    "strike",
                    serde_json::json!({ "t": t, "point": point, "region": region, "who": who }),
                );
                if self.play() {
                    self.struck(world, point, who, ev);
                }
            }
        }
        Ok(())
    }

    /// The autopilot round the road, held to its speed (play: the player's controls).
    fn drive_input(&mut self, world: &mut World) {
        if self.play() {
            let input = self.play_input();
            world.set_vehicle_input(self.vehicle, input);
            return;
        }
        let mut inp = VehicleInput {
            forward: 1.0,
            ..VehicleInput::default()
        };
        if let Some(p) = self.pose {
            // Pure pursuit: 14 m ahead of the nearest track point.
            let track = &self.layout.track;
            let (mut best, mut bd) = (0, f64::INFINITY);
            for (k, &[x, z]) in track.iter().enumerate() {
                let d = (x - p.p[0]) * (x - p.p[0]) + (z - p.p[2]) * (z - p.p[2]);
                if d < bd {
                    bd = d;
                    best = k;
                }
            }
            let [tx, tz] = track[(best + 14) % track.len()];
            let want = [tx - p.p[0], 0.0, tz - p.p[2]];
            let l = js_hypot(&[want[0], want[2]]);
            let l = if l == 0.0 || l.is_nan() { 1.0 } else { l };
            let side = (p.fwd[0] * want[2] - p.fwd[2] * want[0]) / l;
            inp.steer = (2.2 * side).min(1.0).max(-1.0) as f32;
            let e = self.layout.speed - self.car_speed;
            inp.forward = (0.35 * e + 0.25).min(1.0).max(0.0) as f32;
            inp.brake = if e < -3.0 { (-0.2 * e).min(1.0) } else { 0.0 } as f32;
        }
        world.set_vehicle_input(self.vehicle, inp);
    }

    fn play_input(&mut self) -> VehicleInput {
        if self.game.over.is_some() {
            return VehicleInput {
                brake: 1.0,
                ..VehicleInput::default()
            };
        }
        let (fwd, target, handbrake) = self.controls;
        self.steer_now += (target - self.steer_now) * (self.dt * 8.0).min(1.0);
        self.game.throttle = fwd != 0.0;
        VehicleInput {
            forward: fwd as f32,
            steer: self.steer_now as f32,
            handbrake: handbrake as f32,
            ..VehicleInput::default()
        }
    }

    // ---- play

    fn pending(&self) -> usize {
        self.schedule
            .iter()
            .filter(|s| matches!(s.state, St::Wait | St::Approach))
            .count()
    }

    /// The pool's held slots and its parcel slots in use (fractions).
    fn pool_use(&self, world: &World) -> f64 {
        let Some(r) = self.report(world) else {
            return 0.0;
        };
        if world.terrain_areas().is_empty() {
            return 0.0;
        }
        let c = &world.terrain_settings().expect("terrain on").config;
        ((f64::from(r.live) + f64::from(r.airborne)) / f64::from(c.max_particles))
            .max(f64::from(r.parcels) / f64::from(c.max_parcels))
    }

    fn touches(b: [f64; 4], g: f64, a: &TerrainArea) -> bool {
        let (o, e) = (a.plan.origin.to_array(), a.plan.end().to_array());
        b[0] < e[0] as f64 + g
            && b[2] > o[0] as f64 - g
            && b[1] < e[2] as f64 + g
            && b[3] > o[2] as f64 - g
    }

    /// A strike whose box comes within the areas' gap of an open area grows it.
    fn touches_area(&self, world: &World, x: f64, z: f64) -> bool {
        let (b, g) = (self.layout.box_of(x, z), 2.0 * self.layout.block);
        world.terrain_areas().iter().any(|a| Self::touches(b, g, a))
    }

    /// The samples across of the window a strike at (x, z) would grow to.
    fn grown_span(&self, world: &World, x: f64, z: f64) -> f64 {
        let [mut x0, mut z0, mut x1, mut z1] = self.layout.box_of(x, z);
        let g = 2.0 * self.layout.block;
        let areas = world.terrain_areas();
        let mut used = Vec::new();
        let mut grew = true;
        while grew {
            grew = false;
            for a in &areas {
                if used.contains(&a.slot) || !Self::touches([x0, z0, x1, z1], g, a) {
                    continue;
                }
                used.push(a.slot);
                grew = true;
                let (o, e) = (a.plan.origin.to_array(), a.plan.end().to_array());
                x0 = x0.min(o[0] as f64);
                z0 = z0.min(o[2] as f64);
                x1 = x1.max(e[0] as f64);
                z1 = z1.max(e[2] as f64);
            }
        }
        (x1 - x0).max(z1 - z0) / self.placed.sp + 4.0
    }

    fn fits_patch(&self, world: &World, x: f64, z: f64) -> bool {
        self.grown_span(world, x, z) <= self.def.patch() - 8.0
    }

    /// The age of the oldest open area a strike at (x, z) would grow (0: none).
    fn touched_age(&self, world: &World, x: f64, z: f64) -> f64 {
        let (b, g) = (self.layout.box_of(x, z), 2.0 * self.layout.block);
        let mut age = 0.0f64;
        for a in world.terrain_areas() {
            if Self::touches(b, g, &a) {
                let opened = self
                    .game
                    .opened
                    .iter()
                    .find(|o| o.slot == a.slot)
                    .map_or(self.t, |o| o.t);
                age = age.max(self.t - opened);
            }
        }
        age
    }

    fn play_can_open(&self, world: &World, x: f64, z: f64) -> bool {
        self.pool_use(world) < 0.93
            && self.fits_patch(world, x, z)
            && (self.touches_area(world, x, z)
                || world.terrain_areas().len() < Self::regions(world))
    }

    fn edge(&self) -> f64 {
        self.layout.half - 170.0
    }

    fn in_field(&self, x: f64, z: f64) -> bool {
        x.abs() < self.edge() && z.abs() < self.edge()
    }

    fn push(&mut self, mut s: Strike) {
        s.id = self.next_id;
        self.next_id += 1;
        self.schedule.push(s);
    }

    fn try_strike(&mut self, world: &World, x: f64, z: f64, who: &'static str) {
        if !self.in_field(x, z) {
            return self.say("out of the basin");
        }
        if self.pool_use(world) > 0.88 {
            return self.say("too much sand in the air");
        }
        if !self.touches_area(world, x, z)
            && world.terrain_areas().len() + self.pending() >= Self::regions(world)
        {
            return self.say("too many craters still moving");
        }
        if !self.fits_patch(world, x, z) {
            return self.say("that crater field is too big already");
        }
        let t = self.t;
        self.push(Strike {
            at: Some([x, z]),
            who: Some(who),
            ..Strike::new(t)
        });
        if who == "you" {
            self.game.dropped += 1;
        }
    }

    fn flat_fwd(p: &Pose) -> [f64; 3] {
        norm3([p.fwd[0], 0.0, p.fwd[2]])
    }

    fn drop_ahead(&mut self, world: &World) {
        let Some(pose) = self.pose else {
            return;
        };
        if self.game.over.is_some() {
            return;
        }
        let f = Self::flat_fwd(&pose);
        let lead = 40.0 + self.car_speed * APPROACH;
        self.try_strike(
            world,
            pose.p[0] + f[0] * lead,
            pose.p[2] + f[2] * lead,
            "you",
        );
    }

    fn end(&mut self, why: &'static str, world: &World, ev: &mut Vec<Event>) {
        if self.game.over.is_some() {
            return;
        }
        self.game.over = Some((why, self.t));
        self.game.survived = self.t;
        note_with(
            ev,
            world,
            "over",
            serde_json::json!({ "why": why, "at": self.t }),
        );
    }

    /// Contact: a direct hit ends the run; close is a near miss.
    fn struck(&mut self, world: &World, point: [f64; 3], who: Option<&str>, ev: &mut Vec<Event>) {
        let Some(pose) = self.pose else {
            return;
        };
        if self.game.over.is_some() {
            return;
        }
        let d = js_hypot(&[pose.p[0] - point[0], pose.p[2] - point[2]]);
        if d < 0.5 * self.layout.r {
            let why = if who == Some("you") {
                "Hit by your own meteor"
            } else {
                "Direct hit"
            };
            self.end(why, world, ev);
        } else if d < self.layout.rim + 12.0 {
            self.game.near += 1;
            // `toFixed(0)`: the nearest whole metre, halves up.
            self.say(format!("near miss! {} m", d.round()));
        }
    }

    /// The director: where the car will be when a meteor lands, and round it (beside
    /// the line, ahead on it, behind, far out), and more often as you last, onto the car.
    fn direct(&mut self, world: &World) {
        let Some(pose) = self.pose else {
            return;
        };
        if self.game.over.is_some() || self.t < 2.5 || self.t < self.game.next {
            return;
        }
        let gap = (2.6 - self.t / 75.0).max(0.9);
        self.game.next = self.t + gap * (0.7 + 0.6 * self.rand());
        let regions = Self::regions(world);
        // Two kept for yours.
        if (world.terrain_areas().len() + self.pending()) as f64 >= regions as f64 - 2.0 {
            return;
        }
        let (p, v, f) = (pose.p, pose.v, Self::flat_fwd(&pose));
        let side = [f[2], 0.0, -f[0]];
        let at = [p[0] + v[0] * APPROACH, p[2] + v[2] * APPROACH];
        let sgn = if self.rand() < 0.5 { -1.0 } else { 1.0 };
        let pick = self.rand();
        let on_you = (0.08 + self.t / 600.0).min(0.3);
        // Onto the car: 2.4 s of warning, time to steer or brake out of the deadly middle.
        const ON_YOU_LEAD: f64 = 2.4;
        let rim = self.layout.rim;
        let mut lead = None;
        let (mut x, mut z);
        if pick < on_you {
            lead = Some(ON_YOU_LEAD);
            x = p[0] + v[0] * ON_YOU_LEAD + (self.rand() - 0.5) * 8.0;
            z = p[2] + v[2] * ON_YOU_LEAD + (self.rand() - 0.5) * 8.0;
        } else if pick < on_you + 0.35 {
            let a = 8.0 + 35.0 * self.rand();
            let o = sgn * (rim + 2.0 + 10.0 * self.rand());
            x = at[0] + f[0] * a + side[0] * o;
            z = at[1] + f[2] * a + side[2] * o;
        } else if pick < on_you + 0.55 {
            let a = 30.0 + 30.0 * self.rand();
            let o = sgn * 6.0 * self.rand();
            x = at[0] + f[0] * a + side[0] * o;
            z = at[1] + f[2] * a + side[2] * o;
        } else if pick < on_you + 0.75 {
            let a = 14.0 + 20.0 * self.rand();
            let o = sgn * 8.0 * self.rand();
            x = p[0] - f[0] * a + side[0] * o;
            z = p[2] - f[2] * a + side[2] * o;
        } else {
            let a = 2.0 * std::f64::consts::PI * self.rand();
            let d = 70.0 + 120.0 * self.rand();
            x = p[0] + a.cos() * d;
            z = p[2] + a.sin() * d;
        }
        let e = self.edge();
        x = x.min(e - 1.0).max(-e + 1.0);
        z = z.min(e - 1.0).max(-e + 1.0);
        // Not into an area left to settle: try again next frame.
        if self.touched_age(world, x, z) > self.def.settle_after.unwrap_or(8.0) {
            self.game.next = self.t;
            return;
        }
        if self.pool_use(world) < 0.88 && self.fits_patch(world, x, z) {
            let t = self.t;
            self.push(Strike {
                at: Some([x, z]),
                who: Some("sky"),
                lead,
                ..Strike::new(t)
            });
        }
    }

    /// Each step before the strikes: the run's end checks, the areas' ages, the director,
    /// and pruning what has played out.
    fn play_tick(&mut self, world: &World, ev: &mut Vec<Event>) -> anyhow::Result<()> {
        if let (Some(pose), None) = (self.pose, self.game.over) {
            let dt = self.dt;
            self.game.flip_t = if pose.up[1] < 0.3 {
                self.game.flip_t + dt
            } else {
                0.0
            };
            if self.game.flip_t > 2.5 {
                self.end("Rolled over", world, ev);
            }
            // Stuck: on the throttle, and under 5 m from where the car was 8 s ago.
            self.game.stuck_t = if self.game.throttle {
                self.game.stuck_t + dt
            } else {
                0.0
            };
            let every = js_round(1.0 / dt) as u64;
            if self.frame % every == 0 {
                self.game.trail.push(pose.p);
                if self.game.trail.len() > 9 {
                    self.game.trail.remove(0);
                }
            }
            if self.game.trail.len() == 9 {
                let back = self.game.trail[0];
                if self.game.stuck_t > 8.0
                    && js_hypot(&[pose.p[0] - back[0], pose.p[2] - back[2]]) < 5.0
                {
                    self.end("Stuck in the sand", world, ev);
                }
            }
        }
        // Each open area's age: an area grown over others keeps the oldest one's.
        let prev = std::mem::take(&mut self.game.opened);
        for a in world.terrain_areas() {
            let (lo, hi) = (
                a.plan.origin.to_array().map(f64::from),
                a.plan.end().to_array().map(f64::from),
            );
            let mut t = prev
                .iter()
                .find(|o| o.slot == a.slot)
                .map_or(self.t, |o| o.t);
            for o in &prev {
                if o.lo[0] < hi[0] && o.hi[0] > lo[0] && o.lo[2] < hi[2] && o.hi[2] > lo[2] {
                    t = t.min(o.t);
                }
            }
            self.game.opened.push(Opened {
                slot: a.slot,
                t,
                lo,
                hi,
            });
        }
        self.direct(world);
        let t = self.t;
        self.schedule.retain(|s| {
            !(matches!(s.state, St::Done | St::Fizzled)
                && s.struck_at.is_some_and(|a| t - a > 12.0))
        });
        if self.fx.len() > 64 {
            let ids: Vec<u32> = self.schedule.iter().map(|s| s.id).collect();
            self.fx.retain(|f| ids.contains(&f.id));
        }
        Ok(())
    }

    pub(crate) fn act(
        &mut self,
        world: &mut World,
        action: &Action,
        ev: &mut Vec<Event>,
    ) -> anyhow::Result<()> {
        match action {
            Action::Strike { at } => {
                anyhow::ensure!(
                    self.play(),
                    "only the meteor run's play scene takes strikes"
                );
                match at {
                    Some([x, z]) => self.try_strike(world, *x, *z, "you"),
                    None => self.drop_ahead(world),
                }
                Ok(())
            }
            Action::Drive {
                forward,
                steer,
                handbrake,
            } => {
                anyhow::ensure!(
                    self.play(),
                    "only the meteor run's play scene takes the wheel"
                );
                self.controls = (*forward, *steer, *handbrake);
                Ok(())
            }
            Action::Reset => {
                // The host restores its saved world first; the script starts over. (A car
                // read in flight still lands, as the page's did.)
                self.report_dropped = Some(world.terrain_last_report().copied());
                self.t = 0.0;
                self.frame = 0;
                self.fx.clear();
                self.last_strike = -1e9;
                self.pose = None;
                self.car_speed = 0.0;
                for s in &mut self.schedule {
                    s.state = St::Wait;
                    s.live_read = false;
                    s.read = None;
                }
                if self.play() {
                    self.schedule.clear();
                    let throttle = self.game.throttle;
                    self.game = Game {
                        throttle,
                        ..Game::new()
                    };
                }
                note(ev, world, "reset");
                Ok(())
            }
            a => anyhow::bail!("the meteor run has no {} action", a.name()),
        }
    }

    /// The script's state for the page: its clock, the car as last read, the schedule
    /// (each strike's state, contact time and point), the fireballs shown, play's run.
    pub(crate) fn state(&self) -> serde_json::Value {
        let schedule: Vec<_> = self
            .schedule
            .iter()
            .map(|s| {
                let mut o = serde_json::json!({ "state": match s.state {
                    St::Wait => "wait",
                    St::Approach => "approach",
                    St::Done => "done",
                    St::Fizzled => "fizzled",
                } });
                if let Some(t0) = s.t0 {
                    o["t0"] = t0.into();
                }
                if let Some(p) = s.point {
                    o["point"] = serde_json::json!(p);
                }
                if let Some(a) = s.struck_at {
                    o["struckAt"] = a.into();
                }
                if let Some(w) = s.who {
                    o["who"] = w.into();
                }
                o
            })
            .collect();
        let fx: Vec<_> = self
            .fx
            .iter()
            .map(|f| {
                serde_json::json!({ "point": f.point, "t0": f.t0,
                    "fx": { "yaw": f.yaw, "pitch": f.pitch } })
            })
            .collect();
        let g = &self.game;
        serde_json::json!({
            "t": self.t, "frame": self.frame,
            "car": self.pose.map(|p| serde_json::json!({ "p": p.p, "fwd": p.fwd, "up": p.up, "v": p.v })),
            "carSpeed": self.car_speed, "lastStrike": self.last_strike,
            "schedule": schedule, "fx": fx,
            "game": {
                "over": g.over.map(|(why, at)| serde_json::json!({ "why": why, "at": at })),
                "near": g.near, "dropped": g.dropped, "survived": g.survived,
                "msg": g.msg, "msgT": g.msg_t,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn def(q: &str) -> ShowcaseSceneDef {
        match super::super::SceneDef::from_query(super::super::Family::Showcase, q).unwrap() {
            super::super::SceneDef::Showcase(d) => d,
            _ => panic!("not the showcase"),
        }
    }

    #[test]
    fn switches_read_as_the_page_reads_them() {
        let d = def("scene=play&radius=1.5&regions=&cpr=abc&rule=strict&car=");
        assert_eq!(d.scene, "play");
        assert_eq!(d.radius, Some(1.5));
        // `num`: a blank switch is 0 (finite), a word falls back.
        assert_eq!(d.regions, Some(0.0));
        assert_eq!(d.cpr, None);
        assert_eq!(d.car, "dune_buggy");
        let t = d.terrain_def(0.1, true).unwrap();
        assert_eq!(t.settle_rule, SettleRule::Strict);
        assert_eq!(t.config.max_regions, 0);
        assert_eq!(t.surface.unwrap().patch_samples, 128);
        assert_eq!(def("scene=nope").scene, "drive");
        assert!(
            super::super::SceneDef::from_query(super::super::Family::Showcase, "profile=x")
                .is_err()
        );
        assert!(def("scene=drive&regions=2.5")
            .terrain_def(0.1, true)
            .is_err());
    }

    #[test]
    fn mulberry32_as_the_page_computes_it() {
        // The page's `rand` in its int32 steps (Math.imul, `>>>`), for comparison.
        fn page(seed: &mut u32) -> f64 {
            *seed = seed.wrapping_add(0x6D2B79F5);
            let t = *seed as i32;
            let imul = |a: i32, b: i32| a.wrapping_mul(b);
            let t = imul(t ^ ((t as u32) >> 15) as i32, t | 1);
            let t = t ^ (t.wrapping_add(imul(t ^ ((t as u32) >> 7) as i32, t | 61)));
            ((t ^ ((t as u32) >> 14) as i32) as u32) as f64 / 4294967296.0
        }
        assert_eq!(js_to_u32(-1.0), 4294967295);
        assert_eq!(js_to_u32(123456789.9), 123456789);
        let (mut a, mut b) = (12345u32, 12345u32);
        for _ in 0..1000 {
            let x = page(&mut a);
            assert_eq!(x, mulberry32(&mut b));
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn rounding_as_the_page_rounds() {
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0);
        assert_eq!(js_round(0.49999999999999994), 0.0);
        assert_eq!(js_round(-0.5), 0.0);
    }

    fn layout(q: &str) -> (ShowcaseSceneDef, Layout) {
        let d = def(q);
        let (p, _, _) = d.prediction().unwrap();
        let l = Layout::new(&d, p.crater.radius, p.crater.final_rim_radius, p.dx as f64).unwrap();
        (d, l)
    }

    #[test]
    fn the_barrage_lands_beside_the_road() {
        let (d, l) = layout("");
        let s = schedule_of(&d);
        assert_eq!(s.len(), 11);
        for (i, x) in s.iter().enumerate() {
            let [px, pz] = strike_point(&l, &s, i, &d.scene).unwrap();
            // The rim `beside` m from where the car passes.
            let [ax, az] = l.track_at(x.pass.unwrap());
            let gap = js_hypot(&[px - ax, pz - az]);
            assert!(
                (gap - (l.rim + x.beside.unwrap()).abs()).abs() < 1e-9,
                "{i}: {gap}"
            );
        }
        // The track closes on itself.
        let (a, b) = (l.track[0], *l.track.last().unwrap());
        assert!(js_hypot(&[a[0] - b[0], a[1] - b[1]]) < 2.0);
    }

    #[test]
    fn pass_strikes_go_out_from_the_spine_while_clear() {
        let (d, l) = layout("scene=pass");
        let s = schedule_of(&d);
        for i in [0, 2] {
            let [x, z] = strike_point(&l, &s, i, &d.scene).unwrap();
            // Out from the spine toward where the car passes: its box clears the track,
            // or (no step out does) it stays on the spine.
            let on_spine = z == 0.0 && x.abs() <= l.straight;
            assert!(
                on_spine || l.track_clearance(l.box_of(x, z)) >= l.clear,
                "{i}: ({x}, {z})"
            );
        }
        // The rim strike waits for its first strike's point.
        assert!(strike_point(&l, &s, 1, &d.scene).is_err());
    }

    #[test]
    fn the_play_basin_is_rimmed() {
        let (_, l) = layout("scene=play");
        assert!(l.dune(0.0, 0.0).abs() < 2.0);
        assert!(l.dune(590.0, 0.0) > 20.0);
    }
}
