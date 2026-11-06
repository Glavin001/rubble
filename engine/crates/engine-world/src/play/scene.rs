//! One scene definition for every demo family that places its content in Rust: a
//! [`SceneDef`] gives its world's definition ([`SceneDef::world_def`]) and places its
//! content in that world ([`SceneDef::place`]), whichever page, viewer or C host runs it.
//! [`SceneDef::from_query`] reads a page's URL switches, so a preset's query
//! ([`Preset::scene_def`]) is the same scene its page builds. [`Playground`] is a placed
//! scene with its world, stepped.
//!
//! The families keep their own definitions (rigid stacks, vehicles, soft bodies, motion,
//! terrain); this enum is the one entry over them.

use super::presets::{Family, Preset};
use super::{hills, motion, rigid, softbody, vehicles};
use crate::*;
use serde::{Deserialize, Serialize};

/// A rigid benchmark scene (`demo.html`): one of [`rigid::BUILT`], the pyramid's
/// `levels` (none: the scene's), continuous collision for its balls, Morton upload order.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RigidSceneDef {
    pub scene: String,
    pub levels: Option<u32>,
    pub ccd: bool,
    pub spatial: bool,
}

impl Default for RigidSceneDef {
    fn default() -> Self {
        Self {
            scene: "wall".into(),
            levels: None,
            ccd: true,
            spatial: false,
        }
    }
}

impl RigidSceneDef {
    fn table(&self) -> anyhow::Result<rigid::WebScene> {
        rigid::scenes()
            .into_iter()
            .find(|(n, _)| *n == self.scene)
            .map(|(_, s)| s)
            .ok_or_else(|| anyhow::anyhow!("unknown rigid scene {}", self.scene))
    }

    /// The world the scene needs: the table's capacity and solver constants (the page's
    /// defaults: persistent contacts, 60 Hz).
    pub fn world_def(&self) -> anyhow::Result<WorldDef> {
        let s = self.table()?;
        anyhow::ensure!(
            s.snapshot.is_none(),
            "{} is a snapshot scene (import its RUBSNAP1 records)",
            self.scene
        );
        let cap = s
            .cap
            .ok_or_else(|| anyhow::anyhow!("{} has no capacity in the scene table", self.scene))?;
        let mut def = WorldDef {
            gravity: [0.0, -10.0, 0.0],
            contacts: ContactStorage::Persistent,
            capacity: WorldCapacity {
                max_bodies: cap.max_bodies,
                max_colliders: cap.max_colliders,
                max_contacts: cap.max_contacts,
                max_joints: cap.max_joints,
                max_candidates: cap.max_candidates,
                max_colors: cap.max_colors,
                max_commands: cap.max_commands,
                ..WorldCapacity::default()
            },
            ..WorldDef::default()
        };
        let p = &mut def.solver;
        p.dt = (1.0f64 / 60.0) as f32;
        if let Some(i) = s.iterations {
            p.iterations = i;
        }
        p.flags = s.flags.unwrap_or(0);
        p.alpha = s.alpha.unwrap_or(0.99) as f32;
        p.beta_lin = s.beta_lin.unwrap_or(10000.0) as f32;
        p.sleep_lin = s.sleep_lin.unwrap_or(0.15) as f32;
        p.sleep_ang = s.sleep_ang.unwrap_or(0.15) as f32;
        p.sleep_time = s.sleep_time.unwrap_or(0.5) as f32;
        Ok(def)
    }

    pub fn place(&self, world: &mut World) -> anyhow::Result<rigid::Placed> {
        let s = self.table()?;
        let shape = ShapeDef {
            material: SurfaceMaterial {
                friction: s.friction as f32,
                ..SurfaceMaterial::default()
            },
            ..ShapeDef::default()
        };
        let levels = self.levels.or(s.levels).unwrap_or(0);
        rigid::place(world, &self.scene, levels, self.ccd, self.spatial, &shape)
    }
}

/// A scene of any family that places its content in Rust.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "family", rename_all = "camelCase")]
pub enum SceneDef {
    Rigid(RigidSceneDef),
    Vehicle(vehicles::VehicleSceneDef),
    Soft(softbody::SoftSceneDef),
    Motion(motion::MotionSceneDef),
    Terrain(hills::HillsDef),
    City(super::city_scene::CitySceneDef),
    /// Broadside or the timber yard: built by [`super::stage::Play`] from its GLB.
    Range(super::stage::RangeSceneDef),
}

/// What a placed scene hands its host: each family's handles and layout.
#[derive(Clone, Debug)]
pub enum Placed {
    Rigid(rigid::Placed),
    Vehicle(vehicles::Placed),
    Soft(softbody::Placed),
    Motion(motion::Placed),
    Terrain(hills::Placed),
    City(crate::city::City),
}

impl Placed {
    /// The family's own placement as JSON (camelCase), for hosts that read it so; the
    /// rigid scenes' handles (hundreds of thousands) go as typed arrays instead.
    pub fn to_json(&self) -> anyhow::Result<serde_json::Value> {
        Ok(match self {
            Placed::Rigid(p) => serde_json::json!({
                "bodies": p.bodies.iter().map(|b| b.to_bits()).collect::<Vec<_>>(),
                "gpuToScene": p.gpu_to_scene,
                "initial": p.initial,
            }),
            Placed::Vehicle(p) => serde_json::to_value(p)?,
            Placed::Soft(p) => serde_json::to_value(p)?,
            Placed::Motion(p) => serde_json::to_value(p)?,
            Placed::Terrain(p) => serde_json::to_value(p)?,
            Placed::City(c) => serde_json::json!({
                "houses": c.houses.iter().map(|h| h.to_bits()).collect::<Vec<_>>(),
                "lots": c.lots,
                "pitch": c.pitch,
                "ground": c.ground,
            }),
        })
    }
}

impl SceneDef {
    pub fn family(&self) -> Family {
        match self {
            SceneDef::Rigid(_) => Family::Rigid,
            SceneDef::Vehicle(_) => Family::Vehicle,
            SceneDef::Soft(_) => Family::Soft,
            SceneDef::Motion(_) => Family::Motion,
            SceneDef::Terrain(_) => Family::Terrain,
            SceneDef::City(_) => Family::City,
            SceneDef::Range(_) => Family::Range,
        }
    }

    /// The world the scene needs.
    pub fn world_def(&self) -> anyhow::Result<WorldDef> {
        match self {
            SceneDef::Rigid(d) => d.world_def(),
            SceneDef::Vehicle(d) => vehicles::world_def(d),
            SceneDef::Soft(d) => softbody::world_def(d),
            SceneDef::Motion(d) => motion::world_def(d),
            SceneDef::Terrain(d) => hills::world_def(d),
            SceneDef::City(d) => super::city_scene::world_def(d),
            SceneDef::Range(_) => {
                anyhow::bail!("a range scene builds its own world (Playground::new)")
            }
        }
    }

    /// Place the scene in `world` (made from [`Self::world_def`], empty).
    pub fn place(&self, world: &mut World) -> anyhow::Result<Placed> {
        Ok(match self {
            SceneDef::Rigid(d) => Placed::Rigid(d.place(world)?),
            SceneDef::Vehicle(d) => Placed::Vehicle(vehicles::place(world, d)?),
            SceneDef::Soft(d) => Placed::Soft(softbody::place(world, d)?),
            SceneDef::Motion(d) => Placed::Motion(motion::place(world, d)?),
            SceneDef::Terrain(d) => Placed::Terrain(hills::place(world, d)?),
            SceneDef::City(d) => Placed::City(super::city_scene::place(world, d)?),
            SceneDef::Range(_) => {
                anyhow::bail!("a range scene places itself (Playground::new)")
            }
        })
    }

    /// Steps a 60 fps frame takes (the soft scenes and hybrid buggies step faster).
    pub fn steps_per_frame(&self) -> u32 {
        match self {
            SceneDef::Vehicle(d) => vehicles::steps_per_frame(d),
            SceneDef::Soft(d) => (d.hz / 60.0).round().max(1.0) as u32,
            _ => 1,
        }
    }

    /// The scene a family's page builds from its URL query (`a=1&b=2`, with or without
    /// the leading `?`), as the page reads its switches. Families whose content is not
    /// placed here (the lab, the range, impact, the city) have none.
    pub fn from_query(family: Family, query: &str) -> anyhow::Result<SceneDef> {
        let q = Query::parse(query);
        Ok(match family {
            Family::Rigid => {
                let check = q.get("check").is_some();
                let scene = q.get("scene").unwrap_or("wall");
                // ?levels=N (2..50) for the pyramid.
                let levels = rigid::scenes()
                    .into_iter()
                    .find(|(n, _)| *n == scene && *n == "pyramid")
                    .and_then(|(_, s)| s.levels)
                    .map(|d| {
                        let d = d as f64;
                        let v = match q.get("levels").filter(|s| !s.is_empty()) {
                            Some(s) => js_number(s),
                            None => d,
                        };
                        let v = if v == 0.0 || v.is_nan() { d } else { v };
                        v.clamp(2.0, 50.0) as u32
                    });
                SceneDef::Rigid(RigidSceneDef {
                    scene: scene.into(),
                    levels,
                    ccd: match q.get("ccd") {
                        Some(v) => v != "0",
                        None => !check,
                    },
                    spatial: q.get("order") == Some("morton"),
                })
            }
            Family::Vehicle => {
                let scene = q.get("scene").unwrap_or("playground").to_string();
                let hybrid =
                    (scene == "buggies" || scene == "vibeland") && q.get("build") == Some("hybrid");
                SceneDef::Vehicle(vehicles::VehicleSceneDef {
                    preset: q.get("preset").filter(|s| !s.is_empty()).map(Into::into),
                    strength: q.or("strength", 1.0).max(0.01) as f32,
                    hybrid,
                    rigid_mode: q.get("rigid") != Some("0"),
                    spawn: q
                        .get("spawn")
                        .unwrap_or("")
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(js_number)
                        .collect(),
                    vehicles: q.or("vehicles", 48.0) as u32,
                    heavy: q.get("heavy").filter(|s| !s.is_empty()).map(Into::into),
                    tracks_shells: q.get("tracks") == Some("shells"),
                    sleep_off: q.get("sleep") == Some("off"),
                    carry_rest: q.get("rest") == Some("carry"),
                    cannon_ccd: q.get("ccd") != Some("0"),
                    street: None,
                    range: None,
                    guide: q.get("guide") != Some("0"),
                    // The menu takes `?strength=` when it names an option (else none: 0).
                    guide_strength: match q.get("strength") {
                        None | Some("") => 1,
                        Some("0") => 0,
                        Some("1") => 1,
                        Some("2") => 2,
                        _ => 0,
                    },
                    targets_moving: q.get("targets") == Some("moving"),
                    routes: q.get("routes") != Some("0"),
                    scene,
                })
            }
            Family::Soft => {
                let params = q
                    .pairs
                    .iter()
                    .filter(|(_, v)| !v.is_empty() && js_number(v).is_finite())
                    .map(|(k, v)| (k.clone(), js_number(v)))
                    .collect();
                let car = match q.get("car") {
                    Some(c @ ("sedan" | "buggy" | "hires")) => c,
                    _ => "sedan",
                };
                SceneDef::Soft(softbody::SoftSceneDef {
                    scene: q.get("scene").unwrap_or("yard").into(),
                    car: car.into(),
                    import: None,
                    explicit: q.get("solver") == Some("explicit"),
                    explicit_boxes: Vec::new(),
                    strength: q.or("strength", 0.0),
                    deform: q.or("deform", 0.0),
                    hz: q.or("hz", 240.0),
                    iters: q.or("iters", 10.0) as u32,
                    threshold: q.or("threshold", 0.0),
                    sweeps: q.or("sweeps", 0.0) as u32,
                    rigid: q.get("rigid") != Some("0"),
                    params,
                    runup: q.or("runup", 0.0),
                    cars: q.or("cars", 0.0),
                    target: if q.get("target") == Some("car") {
                        "car"
                    } else {
                        "wall"
                    }
                    .into(),
                    ai: q.get("ai").unwrap_or("").into(),
                })
            }
            Family::Motion => {
                let scene = match q.get("scene") {
                    Some(s @ ("ccd" | "character" | "bullets")) => s,
                    _ => "ccd",
                };
                SceneDef::Motion(motion::MotionSceneDef {
                    scene: scene.into(),
                    look_ahead: q.get("lookahead") != Some("0"),
                    // `Number(x) || 250`: blank, 0 and NaN fall back.
                    speed: match q.get("speed").map(js_number) {
                        Some(v) if v != 0.0 && !v.is_nan() => v,
                        _ => 250.0,
                    },
                    autofire: q.get("autofire") != Some("0"),
                })
            }
            Family::Terrain => {
                let n = q.get("n").map(js_number).unwrap_or(f64::NAN);
                let n = hills::COUNTS
                    .iter()
                    .copied()
                    .find(|c| *c as f64 == n)
                    .unwrap_or(2000);
                SceneDef::Terrain(hills::HillsDef { n })
            }
            Family::Range => SceneDef::Range(super::stage::RangeSceneDef::from_query(&q)?),
            Family::City => SceneDef::City(super::city_scene::CitySceneDef::from_query(&q)?),
            f => anyhow::bail!("the {} family is not placed through SceneDef yet", f.name()),
        })
    }
}

/// The families [`SceneDef`] places.
pub fn placed_here(family: Family) -> bool {
    matches!(
        family,
        Family::Rigid
            | Family::Vehicle
            | Family::Soft
            | Family::Motion
            | Family::Terrain
            | Family::City
            | Family::Range
    )
}

impl Preset {
    /// The scene this preset's page builds (none for families placed elsewhere, and for
    /// rigid presets that import a snapshot).
    pub fn scene_def(&self) -> Option<anyhow::Result<SceneDef>> {
        if !placed_here(self.family) {
            return None;
        }
        let def = SceneDef::from_query(self.family, self.query);
        if let Ok(SceneDef::Rigid(r)) = &def {
            if !rigid::BUILT.contains(&r.scene.as_str()) {
                return None;
            }
        }
        Some(def)
    }
}

/// `Number(s)` as a page reads a query value: blank is 0, anything not a number NaN.
pub(crate) fn js_number(s: &str) -> f64 {
    let t = s.trim();
    if t.is_empty() {
        return 0.0;
    }
    match t {
        "Infinity" | "+Infinity" => f64::INFINITY,
        "-Infinity" => f64::NEG_INFINITY,
        _ if t
            .chars()
            .any(|c| c.is_ascii_alphabetic() && c != 'e' && c != 'E') =>
        {
            f64::NAN
        }
        _ => t.parse().unwrap_or(f64::NAN),
    }
}

/// A URL query's pairs, decoded as `URLSearchParams` does (`+` is a space, `%XX` a byte).
pub(crate) struct Query {
    pairs: Vec<(String, String)>,
}

impl Query {
    fn parse(query: &str) -> Self {
        let decode = |s: &str| {
            let b = s.replace('+', " ").into_bytes();
            let mut out = Vec::with_capacity(b.len());
            let mut i = 0;
            while i < b.len() {
                let hex = |c: u8| (c as char).to_digit(16);
                if b[i] == b'%' && i + 2 < b.len() {
                    if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                        out.push((h * 16 + l) as u8);
                        i += 3;
                        continue;
                    }
                }
                out.push(b[i]);
                i += 1;
            }
            String::from_utf8_lossy(&out).into_owned()
        };
        let pairs = query
            .trim_start_matches('?')
            .split('&')
            .filter(|p| !p.is_empty())
            .map(|p| match p.split_once('=') {
                Some((k, v)) => (decode(k), decode(v)),
                None => (decode(p), String::new()),
            })
            .collect();
        Query { pairs }
    }

    /// The first value of `k` (`URLSearchParams.get`).
    pub(crate) fn get(&self, k: &str) -> Option<&str> {
        self.pairs
            .iter()
            .find(|(a, _)| a == k)
            .map(|(_, v)| v.as_str())
    }

    /// `Number(get(k)) || d`.
    fn or(&self, k: &str, d: f64) -> f64 {
        let v = self.get(k).map(js_number).unwrap_or(0.0);
        if v == 0.0 || v.is_nan() {
            d
        } else {
            v
        }
    }
}

/// What a playground runs: a scene placed in its world, or a range scene (Broadside, the
/// timber yard), which builds its own world from its baked GLB ([`super::stage::Play`]).
pub enum Stage {
    Placed { world: World, placed: Placed },
    Range(Box<super::stage::Play>),
}

/// A placed scene with its world, stepped: the one way any host runs a preset.
pub struct Playground {
    pub scene: SceneDef,
    pub director: super::Director,
    pub stage: Stage,
}

/// A playground's counts as of its last step.
#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayStats {
    pub steps: u64,
    pub bodies: u32,
    pub awake: u32,
    pub contacts: u32,
    pub overflow: u32,
}

/// The repository's baked range assets (`engine-webgpu/fracture/assets`), for native hosts.
pub fn repo_asset(name: &str) -> anyhow::Result<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../engine-webgpu/fracture/assets")
        .join(name);
    std::fs::read(&path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

impl Playground {
    /// Make `scene`'s world on `gpu` and place the scene (a range scene's GLB from the
    /// repository's assets).
    pub fn new(gpu: GpuContext, scene: SceneDef) -> anyhow::Result<Self> {
        Self::with_assets(gpu, scene, &repo_asset)
    }

    /// [`Self::new`] with the host's assets (`asset(name)`: the file's bytes).
    pub fn with_assets(
        gpu: GpuContext,
        scene: SceneDef,
        asset: &dyn Fn(&str) -> anyhow::Result<Vec<u8>>,
    ) -> anyhow::Result<Self> {
        if let SceneDef::Range(r) = &scene {
            let glb = asset(r.asset())?;
            let play = super::stage::Play::load(gpu, &glb, r.kind(), r.level, r.raze)?;
            return Ok(Playground {
                scene,
                director: super::Director::none(),
                stage: Stage::Range(Box::new(play)),
            });
        }
        let mut world = World::new(gpu, &scene.world_def()?)?;
        let placed = scene.place(&mut world)?;
        let director = super::Director::new(&scene, &placed, &mut world)?;
        Ok(Playground {
            scene,
            director,
            stage: Stage::Placed { world, placed },
        })
    }

    /// A scene already placed in `world` ([`SceneDef::place`]).
    pub fn of_placed(scene: SceneDef, mut world: World, placed: Placed) -> anyhow::Result<Self> {
        let director = super::Director::new(&scene, &placed, &mut world)?;
        Ok(Playground {
            scene,
            director,
            stage: Stage::Placed { world, placed },
        })
    }

    /// Preset `name`'s scene ([`Preset::scene_def`]), with `query`'s switches (its page's
    /// URL switches, `speed=80&overlap=0.5`; empty: none) over the preset's own.
    pub fn preset(gpu: GpuContext, name: &str, query: &str) -> anyhow::Result<Self> {
        let p = super::preset(name).ok_or_else(|| anyhow::anyhow!("unknown preset {name}"))?;
        anyhow::ensure!(
            placed_here(p.family),
            "preset {name} ({}) is not placed through SceneDef",
            p.family.name()
        );
        // A page reads a switch's first value: the caller's come first.
        let q = query.trim_start_matches('?');
        let merged = if q.is_empty() {
            p.query.to_string()
        } else {
            format!("{q}&{}", p.query)
        };
        Self::new(gpu, SceneDef::from_query(p.family, &merged)?)
    }

    pub fn world(&self) -> &World {
        match &self.stage {
            Stage::Placed { world, .. } => world,
            Stage::Range(p) => &p.scene.range().world,
        }
    }

    pub fn world_mut(&mut self) -> &mut World {
        match &mut self.stage {
            Stage::Placed { world, .. } => world,
            Stage::Range(p) => &mut p.scene.range_mut().world,
        }
    }

    /// The placement (none for a range scene, which places itself).
    pub fn placed(&self) -> Option<&Placed> {
        match &self.stage {
            Stage::Placed { placed, .. } => Some(placed),
            Stage::Range(_) => None,
        }
    }

    /// Do `action` before the next step. A range scene fires its 24-pounder along the
    /// aim (fire or cannon).
    pub fn act(&mut self, action: &super::Action) -> anyhow::Result<()> {
        match &mut self.stage {
            Stage::Placed { world, .. } => self.director.act(world, action),
            Stage::Range(p) => match action {
                super::Action::Cannon { aim } | super::Action::Fire { aim: Some(aim) } => {
                    let f = |v: [f64; 3]| v.map(|x| x as f32);
                    p.shoot_ray(f(aim.eye), f(aim.dir));
                    Ok(())
                }
                a => anyhow::bail!("range scenes have no {} action", a.name()),
            },
        }
    }

    /// One frame of `steps` steps, frame-locked: the scene's frame scripts on fresh reads,
    /// each step after its own script, every result waited on. `dt`: the frame's time (s).
    pub fn frame(&mut self, steps: u32, dt: f64) -> anyhow::Result<()> {
        if let Stage::Placed { world, .. } = &mut self.stage {
            self.director.prepare(world)?;
            while self.director.pending(world) {
                std::thread::yield_now();
            }
            self.director.before_frame(world, dt, None)?;
        }
        self.step(steps)?;
        if let Stage::Placed { world, .. } = &mut self.stage {
            self.director.after_frame(world)?;
        }
        Ok(())
    }

    /// `steps` steps (each after the scene's script), waiting for each one's results.
    pub fn step(&mut self, steps: u32) -> anyhow::Result<()> {
        match &mut self.stage {
            Stage::Placed { world, .. } => {
                for _ in 0..steps {
                    self.director.before_step(world)?;
                    world.step()?;
                    world.poll_blocking()?;
                }
                Ok(())
            }
            Stage::Range(p) => p.frame(steps),
        }
    }

    pub fn stats(&self) -> PlayStats {
        let world = self.world();
        let mut s = PlayStats {
            steps: world.step_count(),
            bodies: world.body_count() as u32,
            ..PlayStats::default()
        };
        if let Some(c) = world.counters() {
            s.awake = c.awake_count;
            s.contacts = c.contact_count;
            s.overflow = c.overflow_flags;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queries_read_as_pages_read_them() {
        let q = Query::parse("?scene=wall&speed=64&overlap=0.25&x=&name=a%20b+c");
        assert_eq!(q.get("scene"), Some("wall"));
        assert_eq!(q.get("name"), Some("a b c"));
        assert_eq!(q.or("x", 7.0), 7.0);
        assert_eq!(q.or("speed", 1.0), 64.0);
        assert!(js_number("abc").is_nan());
        assert_eq!(js_number("4e5"), 4e5);
    }

    #[test]
    fn every_placed_preset_has_a_scene() {
        let mut n = 0;
        for p in super::super::presets() {
            match p.scene_def() {
                Some(d) => {
                    let d = d.unwrap_or_else(|e| panic!("{}: {e}", p.name));
                    assert_eq!(d.family(), p.family, "{}", p.name);
                    if !matches!(d, SceneDef::Range(_)) {
                        d.world_def().unwrap_or_else(|e| panic!("{}: {e}", p.name));
                    }
                    n += 1;
                }
                None => assert!(
                    !placed_here(p.family) || p.family == Family::Rigid,
                    "{} has no scene",
                    p.name
                ),
            }
        }
        assert!(n > 30, "{n} presets placed through SceneDef");
    }

    #[test]
    fn scene_defs_round_trip_as_json() {
        let d = SceneDef::from_query(Family::Soft, "scene=tbone&speed=60&at=1.6").unwrap();
        let j = serde_json::to_value(&d).unwrap();
        assert_eq!(j["family"], "soft");
        let back: SceneDef = serde_json::from_value(j).unwrap();
        let SceneDef::Soft(s) = back else {
            panic!("not soft")
        };
        assert_eq!(s.params["speed"], 60.0);
    }
}
