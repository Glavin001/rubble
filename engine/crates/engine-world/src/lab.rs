//! The fracture lab as one Rust API on [`World`] (engine/docs/lab-world-plan.md): every
//! lab run is a scene placed under a bond model with explicit settings, and actions
//! scheduled on it. The browser page (engine-world-wasm), the C ABI and native demos call
//! the same functions; none of them decides physics.
//!
//! - **Scene** ([`LabSource`] and [`LabArrangement`]): a GLB level, the cuboid house, a
//!   scene pack (the reference rigs, the glass tower), as one copy, a sample's layout of
//!   copies, or the 2×2 tower block with its city (cars and pedestrians) and meteor.
//! - **Bond model** ([`LabBonds`]): none (loose pieces), breakable D6 joints
//!   (`StrengthModel::Bonds`), or a stress model (`Threshold`, `LoadPath`, `ImpactLocal`,
//!   `BlastGpu`) on the world's GPU fracture.
//! - **Settings** ([`LabSettings`]): physics rate, ball scales, CCD, bond strength, the
//!   stress models' fracture switches, crushing. Selecting a scene never changes them; a
//!   scene's suggested settings are [`LabDef::suggested_settings`], applied only when asked.
//! - **Actions** ([`LabAction`]): a shot along a ray, the row shot, the towers' volley,
//!   topple and meteor, a pack's authored shots; each queued at a step ([`Lab::queue`]) and
//!   launched by [`Lab::step`].
//!
//! A combination that cannot work is an error naming why, never a substitution.
//!
//! Drawing: the bond models' scene body `i` is world body slot `i` (the ground first, then
//! every piece), so a renderer binds the world's buffers in place; the stress models'
//! chunks move between bodies as they split, so [`Lab::poses`] gives a pose record per
//! scene body.

use std::sync::Arc;

use crate::*;
use engine_fracture::scene::{Scene, SHAPE_PLANE};

/// Floats a body record takes (`BodyState`, the lab renderer's `BodyPose`).
pub const RECORD_FLOATS: usize = 28;
/// Where a body not in play is drawn (far under the ground).
const PARKED: [f32; 3] = [0.0, -1.0e4, 0.0];
/// Body flags the lab renderer colours by: dynamic, asleep, awake.
const FLAG_DYNAMIC: u32 = 1;
const FLAG_SLEEPING: u32 = 4;
const FLAG_AWAKE: u32 = 8;
/// Balls the stress models keep in flight (past them the oldest goes).
pub const STRESS_SHOTS: u32 = 16;
/// Steps in flight before a host must wait for a readback (World's default is 3): a heavy
/// step (the towers volley) takes longer than a display frame to come back.
pub const READBACK_FRAMES: u32 = 8;
/// The physics rate when a host sets none: the stress models' authored frame. Every model
/// runs at the rate the settings give; none picks its own.
pub const DEFAULT_PHYSICS_HZ: f32 = 60.0;

/// How the stress view's load over strength `u` becomes the renderer's colour ramp (0 blue
/// to 1 at strength). A display choice: the solve and the breaks do not see it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StressScale {
    /// `u` itself: blue unloaded, red at strength.
    #[default]
    Absolute,
    /// Decades of `u` from 10^-8 (blue) to 1 (red, at strength): a structure loaded far
    /// below its strength (a scene authored with very strong bonds) still shows its load
    /// pattern, and a strike's rise.
    Log,
    /// `u` over the frame's largest among the drawn chunks: where the load is, whatever its
    /// size. Not comparable from frame to frame.
    Relative,
}

impl StressScale {
    pub const ALL: [StressScale; 3] = [StressScale::Absolute, StressScale::Log, StressScale::Relative];

    pub fn name(self) -> &'static str {
        match self {
            StressScale::Absolute => "absolute",
            StressScale::Log => "log",
            StressScale::Relative => "relative",
        }
    }

    pub fn parse(name: &str) -> Option<StressScale> {
        Self::ALL.into_iter().find(|s| s.name() == name)
    }

    /// The decades [`StressScale::Log`] spans below 1.
    pub const LOG_DECADES: f32 = 8.0;
}

/// How the lab's bonds hold: none, D6 joints, or a stress model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabBonds {
    /// Loose pieces: no bonds (the lab's "None").
    None,
    /// A strength model: `Bonds` (breakable D6 joints) or a stress model.
    Model(StrengthModel),
}

impl LabBonds {
    /// Every bond model, in the lab's menu order.
    pub const ALL: [LabBonds; 6] = [
        LabBonds::None,
        LabBonds::Model(StrengthModel::Bonds),
        LabBonds::Model(StrengthModel::Threshold),
        LabBonds::Model(StrengthModel::LoadPath),
        LabBonds::Model(StrengthModel::ImpactLocal),
        LabBonds::Model(StrengthModel::BlastGpu),
    ];

    /// The name the lab's menu and URLs use (`none`, `d6`, `threshold`, `load_path`,
    /// `impact_local`, `blast_gpu`).
    pub fn name(self) -> &'static str {
        match self {
            LabBonds::None => "none",
            LabBonds::Model(StrengthModel::Bonds) => "d6",
            LabBonds::Model(m) => m.name(),
        }
    }

    pub fn parse(name: &str) -> Option<LabBonds> {
        Self::ALL.into_iter().find(|b| b.name() == name)
    }

    /// The stress model, for the stress bond models.
    pub fn stress(self) -> Option<StrengthModel> {
        match self {
            LabBonds::Model(m) if m.is_stress() => Some(m),
            _ => None,
        }
    }
}

/// A stress model's own fracture switches, resolved (in-step contact fracture, strike
/// prediction): what `FractureOptions::default()` gives it. A host shows these, and sends
/// the user's choice back explicitly.
pub fn fracture_defaults(model: StrengthModel) -> anyhow::Result<(bool, bool)> {
    let def = DestructibleDef {
        model,
        ..DestructibleDef::new(DestructibleAssetId(0))
    };
    let s = crate::threshold::setup_of(&def)?;
    Ok((s.contact_sweeps > 0, s.predict_strikes))
}

/// Chunk crushing in the stress models: off, the scene's own crush parameters (a ScenePack
/// v3's `materials[].crush`), or those and the cited presets for the other materials.
/// Crushed chunks are removed (dust).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LabCrush {
    #[default]
    Off,
    Scene,
    Presets,
}

/// A lab run's settings: the same names and meanings for every bond model.
#[derive(Clone, Debug, PartialEq)]
pub struct LabSettings {
    pub bonds: LabBonds,
    /// Physics steps per simulated second (> 0).
    pub physics_hz: f32,
    /// Multiplies every bond's break strength (D6 joints and the stress models).
    pub strength_scale: f32,
    /// Multiples of the scene's ball: radius (its mass stays), mass, launch speed.
    pub ball_radius_scale: f64,
    pub ball_mass_scale: f64,
    pub ball_speed_scale: f64,
    /// Continuous collision for the balls (`BodyDef::is_bullet`).
    pub ccd: bool,
    /// The stress models' in-step contact fracture and strike prediction.
    pub fracture: FractureOptions,
    /// The stress models' support loss (`DestructibleDef::support_loss`): weight flows
    /// through the bond graph to the anchors, overloaded bonds break, unsupported pieces tip.
    pub support_loss: bool,
    /// With `support_loss`: `DestructibleDef::support_loss_dynamic_factor` (1 to 2).
    pub support_loss_dynamic_factor: f32,
    /// Chunk crushing (stress models; an error where no chunk can crush).
    pub crush: LabCrush,
    /// Time each step on the GPU.
    pub gpu_timing: GpuTiming,
}

impl Default for LabSettings {
    fn default() -> Self {
        Self {
            bonds: LabBonds::Model(StrengthModel::Bonds),
            physics_hz: DEFAULT_PHYSICS_HZ,
            strength_scale: 1.0,
            ball_radius_scale: 1.0,
            ball_mass_scale: 1.0,
            ball_speed_scale: 1.0,
            ccd: false,
            fracture: FractureOptions::default(),
            support_loss: false,
            support_loss_dynamic_factor: 1.0,
            crush: LabCrush::Off,
            gpu_timing: GpuTiming::Step,
        }
    }
}

/// What the lab builds a scene from.
#[derive(Clone, Debug)]
pub enum LabSource {
    /// A fracture GLB's hierarchy level (`max_hull_vertices`: the hull budget, 8..64).
    Glb {
        bytes: Arc<Vec<u8>>,
        level: u8,
        max_hull_vertices: Option<u32>,
    },
    /// The procedural cuboid house.
    Cuboid,
    /// A scene pack (ScenePack JSON: the reference rigs, vibe-city presets, the glass tower).
    Pack { bytes: Arc<Vec<u8>> },
    /// A lab scene a host built itself (benchmarks, synthetic bond tests), placed as given:
    /// the bond models only (it has no stress import). `settle`: its D6 bonds settle under
    /// self-weight first.
    Scene { scene: Arc<Scene>, settle: bool },
}

impl LabSource {
    /// The glass tower and the pavilion beside it (engine-fracture `glass_tower.rs`).
    pub fn glass_tower() -> anyhow::Result<LabSource> {
        let pack = serde_json::to_vec(&engine_fracture::glass_tower::felling_scene_pack())?;
        Ok(LabSource::Pack {
            bytes: Arc::new(pack),
        })
    }
}

/// How copies of the source are arranged.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum LabArrangement {
    /// One copy.
    #[default]
    One,
    /// A sample's layout of copies (a row, a grid, a suburb).
    Layout(engine_fracture::samples::Layout),
    /// The 2×2 tower block (engine-fracture `towers.rs`): four copies, 6 m streets, the
    /// city's cars and pedestrians as plain rigid bodies, and with `meteor` the 5 m,
    /// 30,000 t ball.
    TowerBlock { meteor: bool },
}

/// A lab run: scene, arrangement and settings.
#[derive(Clone, Debug)]
pub struct LabDef {
    pub source: LabSource,
    pub arrangement: LabArrangement,
    pub settings: LabSettings,
}

impl LabDef {
    /// The settings a scene suggests (the tower block: D6 bonds at 0.2× strength, CCD balls
    /// at 16× speed; the meteor at its own speed), over `base`. Never applied by placing a
    /// scene: a host applies them when its user asks.
    pub fn suggested_settings(&self, base: &LabSettings) -> LabSettings {
        match self.arrangement {
            LabArrangement::TowerBlock { meteor } => LabSettings {
                bonds: LabBonds::Model(StrengthModel::Bonds),
                strength_scale: engine_fracture::towers::BOND_STRENGTH as f32,
                ccd: true,
                ball_speed_scale: if meteor {
                    1.0
                } else {
                    engine_fracture::towers::SPEED_SCALE
                },
                ..base.clone()
            },
            _ => base.clone(),
        }
    }
}

/// A shot action, queued on a placed lab ([`Lab::queue`]).
#[derive(Clone, Debug, PartialEq)]
pub enum LabAction {
    /// A ball from `origin` along `direction` by the lab's rule (the scene's speed times the
    /// speed scale, lifted against gravity when the scene arcs its shots).
    Shot {
        origin: [f64; 3],
        direction: [f64; 3],
    },
    /// The scene's row shot (a row of houses).
    RowShot,
    /// The tower block's volley: raking shots, then shots at the towers' lower floors.
    Volley,
    /// Base shots that topple the block's first tower over its street.
    Topple,
    /// The meteor onto the block's first tower.
    Meteor,
    /// A pack's authored shot `index`, as authored (its scales do not apply).
    PackShot(u32),
}

/// A pack's authored shot.
pub use engine_fracture::pack::PackShot;
/// A sample's layout of copies (a row, a grid, a suburb).
pub use engine_fracture::samples::Layout as LabLayout;

/// A lab's counts, every number a host shows.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LabStats {
    pub steps: u64,
    pub shots: u32,
    /// Summed over the scene's destructibles.
    pub chunks: u32,
    pub bonds: u32,
    pub bonds_intact: u32,
    pub bodies: u32,
    pub crushed: u32,
    /// The last polled step's counters.
    pub awake: u32,
    pub contacts: u32,
    /// GPU time of a physics step (ms; the last timed step), where the device times it.
    pub gpu_step_ms: Option<f64>,
    /// What that allows: steps a second with the GPU doing nothing else, and the share of
    /// each wall second that real time at the physics rate takes.
    pub max_steps_per_second: Option<f64>,
    pub realtime_share: Option<f64>,
    pub physics_hz: f32,
}

/// A lab run built on the host (no GPU yet): the scene a renderer draws, the world's
/// definition and the assets to place. D6 bonds settle first ([`Self::needs_settle`]).
pub struct LabPlan {
    def: LabDef,
    /// The scene of one copy, before its arrangement (the D6 settle's input).
    one: Scene,
    /// The arranged scene: what a renderer draws (its bodies in scene order).
    scene: Scene,
    /// The stress models' asset (one copy) and its authored shots.
    stress_asset: Option<DestructibleAssetDesc>,
    pack_shots: Vec<PackShot>,
    settled: bool,
}

impl LabPlan {
    /// Build the run's scene and assets. Checks every setting; a combination that cannot
    /// work is an error naming why.
    pub fn new(def: &LabDef) -> anyhow::Result<LabPlan> {
        let s = &def.settings;
        anyhow::ensure!(
            s.physics_hz.is_finite() && s.physics_hz > 0.0,
            "physics rate {} Hz: give a positive rate",
            s.physics_hz
        );
        for (what, k) in [
            ("ball radius scale", s.ball_radius_scale),
            ("ball mass scale", s.ball_mass_scale),
            ("ball speed scale", s.ball_speed_scale),
            ("bond strength scale", s.strength_scale as f64),
        ] {
            anyhow::ensure!(k.is_finite() && k > 0.0, "{what} {k}: give a positive multiple");
        }
        let block = matches!(def.arrangement, LabArrangement::TowerBlock { .. });
        anyhow::ensure!(
            !block || matches!(def.source, LabSource::Pack { .. }),
            "the tower block is copies of a scene pack (one tower)"
        );
        anyhow::ensure!(
            s.crush == LabCrush::Off || s.bonds.stress().is_some(),
            "crushing is a stress model's (Threshold, LoadPath, ImpactLocal, BlastGpu); the {} bonds have none",
            s.bonds.name()
        );
        anyhow::ensure!(
            s.fracture == FractureOptions::default() || s.bonds.stress().is_some(),
            "the contact fracture and strike prediction switches are the stress models'; the {} bonds have none",
            s.bonds.name()
        );
        anyhow::ensure!(
            !s.support_loss || s.bonds.stress().is_some(),
            "support loss is a stress model's (Threshold, LoadPath, ImpactLocal, BlastGpu); the {} bonds have none",
            s.bonds.name()
        );
        let dt = 1.0 / s.physics_hz as f64;
        let raw = match &def.source {
            LabSource::Glb {
                bytes,
                level,
                max_hull_vertices,
            } => {
                let parsed = engine_fracture::parse_fracture_glb(bytes)?;
                engine_fracture::build_scene(&parsed, *level, max_hull_vertices.map(|v| v as usize))?
            }
            LabSource::Cuboid => engine_fracture::cuboid::build_cuboid_house(&Default::default())?,
            LabSource::Pack { bytes } => engine_fracture::pack::build_pack_scene(bytes)?.0,
            LabSource::Scene { scene, .. } => {
                anyhow::ensure!(
                    s.bonds.stress().is_none(),
                    "a prebuilt lab scene has no stress import: the {} model needs a GLB, a pack or the cuboid house",
                    s.bonds.name()
                );
                (**scene).clone()
            }
        };
        let mut one = raw;
        one.params.dt = dt as f32;
        let (stress_asset, pack_shots) = match s.bonds.stress() {
            Some(_) => {
                let (desc, shots) = match &def.source {
                    LabSource::Glb { bytes, level, .. } => {
                        (DestructibleAssetDesc::from_glb(bytes, *level)?, Vec::new())
                    }
                    LabSource::Pack { bytes } => (
                        DestructibleAssetDesc::from_pack(bytes)?,
                        engine_fracture::pack::pack_shots(bytes)?,
                    ),
                    LabSource::Cuboid => {
                        (DestructibleAssetDesc::cuboid_house(&Default::default())?, Vec::new())
                    }
                    LabSource::Scene { .. } => unreachable!("refused above"),
                };
                let stress = desc
                    .stress
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("the source has no stress import"))?;
                stress.scene()?;
                check_lab_chunk_order(&desc)?;
                (Some(desc), shots)
            }
            None => {
                let shots = match &def.source {
                    LabSource::Pack { bytes } => engine_fracture::pack::pack_shots(bytes)?,
                    _ => Vec::new(),
                };
                (None, shots)
            }
        };
        // The bond models' bonds: none, or the D6 joints at the strength scale.
        if s.bonds == LabBonds::None || s.bonds.stress().is_some() {
            one = one.without_joints();
        } else if s.strength_scale != 1.0 {
            one = one.scale_bond_strength(s.strength_scale as f64);
        }
        let mut plan = LabPlan {
            def: def.clone(),
            scene: one.clone(),
            one,
            stress_asset,
            pack_shots,
            settled: false,
        };
        plan.arrange()?;
        Ok(plan)
    }

    pub fn def(&self) -> &LabDef {
        &self.def
    }

    /// The scene a renderer draws: every body in scene order (the ground first), the
    /// balls' slots after the pieces.
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Whether D6 bonds must settle under self-weight before placing (the lab's bond
    /// preload): [`Self::settle`] natively, [`Self::begin_settle`] in a host that must
    /// not wait.
    pub fn needs_settle(&self) -> bool {
        let wanted = match &self.def.source {
            LabSource::Scene { settle, .. } => *settle,
            _ => true,
        };
        wanted
            && !self.settled
            && self.def.settings.bonds == LabBonds::Model(StrengthModel::Bonds)
            && !self.one.joints.is_empty()
            && !matches!(self.def.arrangement, LabArrangement::TowerBlock { .. })
    }

    /// Settle one copy's bonds in a helper world on `gpu` and arrange the settled copy
    /// (blocks until done). Returns whether it settled.
    pub fn settle(&mut self, gpu: GpuContext) -> anyhow::Result<bool> {
        let Some(mut job) = self.begin_settle(gpu)? else {
            return Ok(true);
        };
        loop {
            if let Some(done) = job.advance(true)? {
                self.finish_settle(&job)?;
                return Ok(done);
            }
        }
    }

    /// [`Self::settle`] as a job that never waits (`None`: nothing to settle).
    pub fn begin_settle(&self, gpu: GpuContext) -> anyhow::Result<Option<LabSettle>> {
        if !self.needs_settle() {
            return Ok(None);
        }
        let desc = DestructibleAssetDesc::from_scene(&self.one)?;
        let solver = desc
            .solver
            .ok_or_else(|| anyhow::anyhow!("the lab asset has no solver"))?;
        let mut world = World::new(gpu, &bonds_world_def(&self.one, &desc, solver, &self.def.settings))?;
        let id = world.create_destructible_asset(&desc)?;
        let job = world.begin_destructible_preload(id)?;
        Ok(Some(LabSettle {
            world,
            id,
            job,
            done: None,
        }))
    }

    /// The settled copy arranged (after [`LabSettle::advance`] reported done).
    pub fn finish_settle(&mut self, job: &LabSettle) -> anyhow::Result<()> {
        anyhow::ensure!(job.done.is_some(), "the settle has not finished");
        let desc = job.world.destructible_asset(job.id)?;
        if let Some(p) = &desc.preload {
            let s = &self.one;
            // The chunks' centres of mass where the settle left them (a body origin's pose
            // times the chunk's frame), as the body records `Scene::settled` reads.
            let mut snapshot = vec![0.0f32; RECORD_FLOATS * s.bodies.len()];
            let shots = s.projectile_start..s.projectile_start + s.projectile_count;
            let pieces = (1..s.bodies.len() as u32).filter(|b| !shots.contains(b));
            for ((b, chunk), origin) in pieces.zip(&desc.chunks).zip(&p.body_poses) {
                let t = compose(origin, &chunk.pose);
                let o = RECORD_FLOATS * b as usize;
                snapshot[o..o + 3].copy_from_slice(&t.position);
                snapshot[o + 4..o + 8].copy_from_slice(&t.rotation);
            }
            self.one = s.settled(&snapshot, p.steps, p.settled)?;
        }
        self.settled = true;
        self.arrange()
    }

    /// The arranged scene from `one`: copies, the tower block and its city, the ball scale.
    fn arrange(&mut self) -> anyhow::Result<()> {
        let s = &self.def.settings;
        let mut scene = match self.def.arrangement {
            LabArrangement::One => self.one.clone(),
            LabArrangement::Layout(l) => engine_fracture::samples::layout_scene(
                self.one.clone(),
                Some(l),
                matches!(self.def.source, LabSource::Cuboid),
            ),
            LabArrangement::TowerBlock { meteor } => {
                let (gap, columns, rows) = (engine_fracture::towers::GAP, 2, 2);
                let ball = if meteor {
                    engine_fracture::towers::METEOR
                } else {
                    engine_fracture::towers::BALL
                };
                let grid = engine_fracture::towers::towers_grid(&self.one, columns, rows, gap, Some(ball));
                engine_fracture::props::with_city(&grid)
            }
        };
        if s.ball_radius_scale != 1.0 {
            scene = scene.scale_projectile_radius(s.ball_radius_scale)?;
        }
        scene.params.dt = (1.0 / s.physics_hz as f64) as f32;
        self.scene = scene;
        Ok(())
    }

    /// The world this run needs: its solver at the physics rate, sleep and contact storage
    /// for its bond model, the capacity for every copy, the balls and the props.
    pub fn world_def(&self) -> anyhow::Result<WorldDef> {
        let s = &self.def.settings;
        match s.bonds.stress() {
            None => {
                let desc = DestructibleAssetDesc::from_scene(&self.scene)?;
                let solver = desc
                    .solver
                    .ok_or_else(|| anyhow::anyhow!("the lab asset has no solver"))?;
                Ok(bonds_world_def(&self.scene, &desc, solver, s))
            }
            Some(model) => self.stress_world_def(model),
        }
    }

    fn stress_world_def(&self, model: StrengthModel) -> anyhow::Result<WorldDef> {
        let s = &self.def.settings;
        let desc = self.stress_asset.as_ref().expect("a stress asset");
        let stress = desc.stress.as_ref().expect("a stress source");
        let mut solver = desc
            .solver_for(model)
            .ok_or_else(|| anyhow::anyhow!("no {} solver", model.name()))?;
        solver.dt = 1.0 / s.physics_hz;
        let scene = stress.stress_scene(model, s.strength_scale, solver.dt)?;
        let copies = self.copies().len();
        let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum::<usize>() * copies;
        let props = self.scene.props.len();
        let one = stress_rubble::capacity_for(&scene);
        let mut c = one.clone();
        c.max_colliders = one.max_colliders * copies as u32 + self.prop_colliders() + STRESS_SHOTS + 8;
        c.max_hull_vertices = one.max_hull_vertices * copies as u32 + self.prop_hull_vertices();
        c.max_bodies = c
            .max_bodies
            .max(((chunks + props + STRESS_SHOTS as usize + 8) as u32).next_power_of_two());
        c.max_contacts = stress_rubble::threshold::contact_capacity(
            chunks + props + STRESS_SHOTS as usize,
            c.max_contacts,
            1 << 30,
        );
        c.max_candidates = c.max_candidates.max(c.max_contacts);
        c.max_commands = c.max_commands.max(2048);
        Ok(WorldDef {
            capacity: c,
            gravity: [0.0, solver.gravity_y, 0.0],
            solver,
            enable_sleep: false,
            contacts: ContactStorage::Canonical,
            readback_frames: READBACK_FRAMES,
            guidance: Some(GuidanceCapacity {
                max_zones: 256,
                plan_budget: 256,
            }),
            gpu_timing: s.gpu_timing,
            ..WorldDef::default()
        })
    }

    fn prop_bodies(&self) -> std::collections::BTreeSet<u32> {
        self.scene.prop_bodies.iter().copied().collect()
    }

    fn prop_colliders(&self) -> u32 {
        let props = self.prop_bodies();
        self.scene.colliders.iter().filter(|c| props.contains(&c.body)).count() as u32
    }

    fn prop_hull_vertices(&self) -> u32 {
        // The props' cooked hulls (cars), at most the scene's arena.
        (self.scene.hull_data.len() as u32).min(1 << 16)
    }

    /// Each copy's instance offset in the arranged scene (asset frame): its first piece less
    /// the asset's first chunk.
    fn copies(&self) -> Vec<[f32; 3]> {
        let s = &self.scene;
        let stride = s.instance_stride.max(1) as usize;
        let copies = s.copies.max(1) as usize;
        let Some(desc) = &self.stress_asset else {
            return vec![[0.0; 3]];
        };
        let first = desc.chunks[0].pose.position;
        (0..copies)
            .map(|k| {
                let p = s.bodies[1 + k * stride].p;
                [0, 1, 2].map(|i| p[i] as f32 - first[i])
            })
            .collect()
    }
}

/// A D6 settle in progress ([`LabPlan::begin_settle`]).
pub struct LabSettle {
    world: World,
    id: DestructibleAssetId,
    job: Option<DestructiblePreload>,
    done: Option<bool>,
}

impl LabSettle {
    /// Advance it (`wait`: block on the GPU; else as far as free readback slots allow).
    /// `None` until it is decided, then whether it settled.
    pub fn advance(&mut self, wait: bool) -> anyhow::Result<Option<bool>> {
        if self.done.is_some() {
            return Ok(self.done);
        }
        let Some(job) = self.job.as_mut() else {
            self.done = Some(true);
            return Ok(self.done);
        };
        self.done = self.world.advance_destructible_preload(job, wait)?;
        Ok(self.done)
    }

    /// Settle steps checked so far.
    pub fn steps(&self) -> u32 {
        self.job.as_ref().map_or(0, |j| j.steps())
    }
}

/// The lab's surface: every collider at the scene's friction, no restitution, a 1 cm
/// contact offset.
fn lab_material(s: &Scene) -> SurfaceMaterial {
    SurfaceMaterial {
        friction: s.params.friction,
        restitution: 0.0,
        contact_offset: 0.01,
        ..Default::default()
    }
}

fn next_pow2(n: u64) -> u32 {
    n.max(1).next_power_of_two() as u32
}

fn hull_vertices(desc: &DestructibleAssetDesc) -> u32 {
    desc.chunks
        .iter()
        .flat_map(|c| &c.shapes)
        .map(|g| match g {
            Geometry::Hull { hull, .. } => hull.vertices.len() as u32,
            _ => 0,
        })
        .sum()
}

/// The bond models' world: the lab plan's capacities, sleep on, persistent contacts, the
/// physics rate.
fn bonds_world_def(s: &Scene, desc: &DestructibleAssetDesc, mut solver: AvbdParams, o: &LabSettings) -> WorldDef {
    solver.dt = 1.0 / o.physics_hz;
    let budget = s.capacity;
    let nb = s.bodies.len() as u64 + 1;
    let nc = s.colliders.len() as u64;
    let c = WorldCapacity {
        max_bodies: next_pow2(nb.max(64)),
        max_colliders: next_pow2(nc.max(64)),
        max_contacts: next_pow2((nc * budget.contacts_per_collider as u64).max(4096)),
        max_candidates: next_pow2((nc * budget.candidates_per_collider as u64).max(4096)),
        max_joints: (s.joints.len() as u32).max(1),
        max_colors: budget.max_colors.unwrap_or(32),
        // Launches and resets are commands in World.
        max_commands: 2048,
        max_hull_vertices: (hull_vertices(desc) + 64).max(1024),
    };
    WorldDef {
        capacity: c,
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: true,
        contacts: ContactStorage::Persistent,
        readback_frames: READBACK_FRAMES,
        guidance: Some(GuidanceCapacity {
            max_zones: 256,
            plan_budget: 256,
        }),
        gpu_timing: o.gpu_timing,
        ..WorldDef::default()
    }
}

/// What a scene body is drawn from in the stress models.
#[derive(Clone, Copy, Debug)]
enum Drawn {
    Ground,
    /// Chunk `chunk` of destructible `copy`, at `rest` (its lab pose) in its body.
    Chunk { copy: u32, chunk: u32, rest: Transform },
    /// A plain body (a street prop).
    Body(BodyId),
    /// A ball slot, or nothing.
    None,
}

/// The lab's shot rule (`engine_fracture::plan::shot_velocity`) and its settings.
#[derive(Clone, Copy, Debug)]
struct Shooting {
    center: [f64; 3],
    extent: f64,
    radius: f64,
    mass: f64,
    speed: Option<f64>,
    arc: bool,
    speed_scale: f64,
    ccd: bool,
    /// The pack frame's origin in the lab's (stress models' packs: the scene placement).
    placement: [f64; 3],
}

impl Shooting {
    fn of(s: &Scene, o: &LabSettings) -> Shooting {
        Shooting {
            center: s.center,
            extent: s.extent,
            radius: s.projectile_radius,
            mass: s.projectile_mass * o.ball_mass_scale,
            speed: s.projectile_speed,
            arc: s.projectile_arc,
            speed_scale: o.ball_speed_scale,
            ccd: o.ccd,
            placement: [0.0; 3],
        }
    }

    /// The scene's shot speed (m/s) before the speed scale.
    fn base_speed(&self) -> f64 {
        engine_fracture::plan::default_shot_speed(self.extent, self.speed)
    }
}

/// A shot due at a step.
#[derive(Clone, Debug)]
struct Due {
    step: u64,
    origin: [f64; 3],
    velocity: [f64; 3],
    spin: [f64; 3],
    mass: f64,
    radius: f64,
}

/// A lab run placed in a world.
pub struct Lab {
    settings: LabSettings,
    scene: Scene,
    destructibles: Vec<DestructibleId>,
    /// The street props (a loose destructible), stress models.
    props: Option<DestructibleId>,
    /// Per scene body, its world slot (`u32::MAX`: none). Bond models only.
    slot_of: Vec<u32>,
    /// Bond models: the parked ball bodies, one per ball slot.
    shots: Vec<BodyId>,
    /// Stress models: the live balls, oldest first (the shared pool, `play::balls`).
    balls: crate::play::balls::Balls,
    fired: u32,
    shooting: Shooting,
    material: SurfaceMaterial,
    /// Stress models: what each scene body is drawn from.
    drawn: Vec<Drawn>,
    pack_shots: Vec<PackShot>,
    /// Bond models: the placed scene, for [`Lab::reset`].
    saved: Option<WorldSnapshot>,
    /// Shots queued, by step.
    schedule: Vec<Due>,
    /// Directed debris: on, and its zones (scene body anchors).
    guidance: Option<(GuidanceSettings, Vec<engine_schema::guidance::ZoneDesc>)>,
    /// The stress view's colour scale ([`Lab::poses`]).
    stress_scale: StressScale,
}

impl Lab {
    /// Place a plan in `world` (made from [`LabPlan::world_def`], empty): the ground, the
    /// destructibles, the props, the ball slots.
    pub fn place(world: &mut World, plan: &LabPlan) -> anyhow::Result<Lab> {
        anyhow::ensure!(
            !plan.needs_settle(),
            "settle the D6 bonds first (LabPlan::settle)"
        );
        anyhow::ensure!(world.body_count() == 0, "the lab places its scene in an empty world");
        match plan.def.settings.bonds.stress() {
            None => Self::place_bonds(world, plan),
            Some(model) => Self::place_stress(world, plan, model),
        }
    }

    fn lab_ground(world: &mut World, s: &Scene, material: SurfaceMaterial) -> anyhow::Result<BodyId> {
        let g = &s.bodies[0];
        let ground = world.create_body(&BodyDef {
            position: f3(g.p),
            rotation: f4(g.q),
            ..BodyDef::default()
        })?;
        let def = ShapeDef {
            material,
            ..ShapeDef::default()
        };
        for c in s.colliders.iter().filter(|c| c.body == 0) {
            anyhow::ensure!(c.shape == SHAPE_PLANE, "the lab's ground is planes (collider shape {})", c.shape);
            let p = c.params;
            world.create_shape(
                ground,
                &def,
                Geometry::Plane {
                    normal: [p[0] as f32, p[1] as f32, p[2] as f32],
                    offset: p[3] as f32,
                },
            )?;
        }
        Ok(ground)
    }

    fn place_bonds(world: &mut World, plan: &LabPlan) -> anyhow::Result<Lab> {
        let s = &plan.scene;
        let material = lab_material(s);
        let desc = DestructibleAssetDesc::from_scene(s)?;
        let asset = world.create_destructible_asset(&desc)?;
        let ground = Self::lab_ground(world, s, material)?;
        anyhow::ensure!(ground.index() == 0, "the ground took slot {}", ground.index());
        let shots_range = s.projectile_start..s.projectile_start + s.projectile_count;
        // The scene's pieces start asleep (the lab's scenes do), unless every one is awake.
        let asleep = (1..s.bodies.len())
            .filter(|i| !shots_range.contains(&(*i as u32)) && s.bodies[*i].inv_mass > 0.0)
            .any(|i| s.bodies[i].flags & FLAG_SLEEPING != 0);
        let d = world.create_destructible(&DestructibleDef {
            asleep,
            shape: ShapeDef {
                material,
                update_body_mass: false,
                ..ShapeDef::default()
            },
            ..DestructibleDef::new(asset)
        })?;
        let mut slot_of = vec![u32::MAX; s.bodies.len()];
        slot_of[0] = 0;
        let pieces = (1..s.bodies.len() as u32).filter(|b| !shots_range.contains(b));
        for (k, b) in pieces.enumerate() {
            slot_of[b as usize] = world.destructible_chunk_body(d, k as u32)?.index();
        }
        // A renderer reads slots directly: the pieces before the balls are scene body =
        // slot, bond = joint.
        for b in 1..s.projectile_start.min(s.bodies.len() as u32) {
            anyhow::ensure!(slot_of[b as usize] == b, "scene body {b} took world slot {}", slot_of[b as usize]);
        }
        for k in 0..s.joints.len() as u32 {
            let j = world.destructible_bond_joint(d, k)?;
            anyhow::ensure!(j.map(|j| j.index()) == Some(k), "bond {k} is joint {:?}", j.map(|j| j.index()));
        }
        // The balls, parked static where the scene parks them until fired.
        let r = s.projectile_radius as f32;
        let mut shots = Vec::with_capacity(s.projectile_count as usize);
        for i in shots_range {
            let b = &s.bodies[i as usize];
            let body = world.create_body(&BodyDef {
                position: f3(b.p),
                rotation: f4(b.q),
                ..BodyDef::default()
            })?;
            world.create_sphere_shape(
                body,
                &ShapeDef {
                    material,
                    update_body_mass: false,
                    ..ShapeDef::default()
                },
                [0.0; 3],
                r,
            )?;
            slot_of[i as usize] = body.index();
            shots.push(body);
        }
        world.reserve_moving_radius(r);
        let saved = Some(world.save_state()?);
        Ok(Lab {
            settings: plan.def.settings.clone(),
            scene: s.clone(),
            destructibles: vec![d],
            props: None,
            slot_of,
            shots,
            balls: crate::play::balls::Balls::new(1, 1),
            fired: 0,
            shooting: Shooting::of(s, &plan.def.settings),
            material,
            drawn: Vec::new(),
            pack_shots: plan.pack_shots.clone(),
            saved,
            schedule: Vec::new(),
            guidance: None,
            stress_scale: StressScale::Absolute,
        })
    }

    fn place_stress(world: &mut World, plan: &LabPlan, model: StrengthModel) -> anyhow::Result<Lab> {
        let s = &plan.scene;
        let o = &plan.def.settings;
        let material = lab_material(s);
        let desc = plan.stress_asset.as_ref().expect("a stress asset");
        let stress = desc.stress.as_ref().expect("a stress source");
        let asset = world.create_destructible_asset(desc)?;
        if let Some(h) = stress.ground_height() {
            let g = world.create_body(&BodyDef::default())?;
            world.create_shape(
                g,
                &ShapeDef {
                    material,
                    ..ShapeDef::default()
                },
                Geometry::Plane {
                    normal: [0.0, 1.0, 0.0],
                    offset: h,
                },
            )?;
        }
        let crush = crush_for(o.crush, stress)?;
        let offsets = plan.copies();
        let mut destructibles = Vec::with_capacity(offsets.len());
        for off in &offsets {
            destructibles.push(world.create_destructible(&DestructibleDef {
                model,
                transform: Transform::new(*off, [0.0, 0.0, 0.0, 1.0]),
                strength_scale: o.strength_scale,
                asleep: false,
                crush: crush.clone(),
                fracture: o.fracture,
                support_loss: o.support_loss,
                support_loss_dynamic_factor: o.support_loss_dynamic_factor,
                shape: ShapeDef {
                    material,
                    update_body_mass: false,
                    ..ShapeDef::default()
                },
                ..DestructibleDef::new(asset)
            })?);
        }
        // Chunk c of copy k is scene body 1 + c + k * stride; the stress import's first
        // chunks are the lab scene's, in order (its ground anchors follow, undrawn).
        let stride = s.instance_stride.max(1) as usize;
        let lab_chunks = desc.chunks.len();
        let mut drawn = vec![Drawn::None; s.bodies.len()];
        drawn[0] = Drawn::Ground;
        for k in 0..offsets.len() {
            for c in 0..lab_chunks {
                let i = 1 + c + k * stride;
                if i < drawn.len() {
                    drawn[i] = Drawn::Chunk {
                        copy: k as u32,
                        chunk: c as u32,
                        rest: desc.chunks[c].pose,
                    };
                }
            }
        }
        // The street props: plain rigid bodies (a destructible of loose pieces), as the
        // scene has them.
        let mut props = None;
        if !s.prop_bodies.is_empty() {
            let (d, bodies) = place_props(world, s, material)?;
            for (&b, body) in s.prop_bodies.iter().zip(bodies) {
                drawn[b as usize] = Drawn::Body(body);
            }
            props = Some(d);
        }
        let mut sh = Shooting::of(s, o);
        sh.placement = stress.placement();
        Ok(Lab {
            settings: o.clone(),
            scene: s.clone(),
            destructibles,
            props,
            slot_of: Vec::new(),
            shots: Vec::new(),
            balls: crate::play::balls::Balls::new(
                s.projectile_count.max(1).min(STRESS_SHOTS) as usize,
                usize::MAX,
            ),
            fired: 0,
            shooting: sh,
            material,
            drawn,
            pack_shots: plan.pack_shots.clone(),
            saved: None,
            schedule: Vec::new(),
            guidance: None,
            stress_scale: StressScale::Absolute,
        })
    }

    pub fn settings(&self) -> &LabSettings {
        &self.settings
    }

    /// The scene a renderer draws.
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Whether the stress models draw from [`Self::poses`] (else the world's buffers, scene
    /// body = slot).
    pub fn draws_poses(&self) -> bool {
        self.settings.bonds.stress().is_some()
    }

    pub fn destructibles(&self) -> &[DestructibleId] {
        &self.destructibles
    }

    /// The street props' destructible (loose pieces), stress models with a city.
    pub fn props(&self) -> Option<DestructibleId> {
        self.props
    }

    /// Shots fired since placement (or the last reset).
    pub fn shots_fired(&self) -> u32 {
        self.fired
    }

    /// The scene's shot speed (m/s) before the speed scale, its mass and radius as fired.
    pub fn shot_speed(&self) -> f64 {
        self.shooting.base_speed()
    }

    pub fn shot_mass(&self) -> f64 {
        self.shooting.mass
    }

    pub fn shot_radius(&self) -> f64 {
        self.shooting.radius
    }

    /// The ball settings for the shots fired from now on.
    pub fn set_ball(&mut self, mass_scale: f64, speed_scale: f64, ccd: bool) -> anyhow::Result<()> {
        for (what, k) in [("ball mass scale", mass_scale), ("ball speed scale", speed_scale)] {
            anyhow::ensure!(k.is_finite() && k > 0.0, "{what} {k}: give a positive multiple");
        }
        self.settings.ball_mass_scale = mass_scale;
        self.settings.ball_speed_scale = speed_scale;
        self.settings.ccd = ccd;
        self.shooting.mass = self.scene.projectile_mass * mass_scale;
        self.shooting.speed_scale = speed_scale;
        self.shooting.ccd = ccd;
        Ok(())
    }

    /// The world slot of each scene body (`None`: not a body), bond models.
    pub fn slot_of(&self) -> &[u32] {
        &self.slot_of
    }

    /// Stress models: the live balls, oldest first, with the step each was fired.
    pub fn balls(&self) -> &[(BodyId, u64)] {
        self.balls.live()
    }

    /// The pack's authored shots.
    pub fn pack_shots(&self) -> &[PackShot] {
        &self.pack_shots
    }

    /// Queue `action` to start `delay` steps from now: its shots launch at their steps
    /// ([`Self::step`] or [`Self::launch_due`]). An action the scene has none of is an error.
    pub fn queue(&mut self, world: &World, action: &LabAction, delay: u64) -> anyhow::Result<()> {
        let base = world.step_count() + delay;
        let sh = self.shooting;
        let block = || -> anyhow::Result<()> {
            anyhow::ensure!(
                !self.scene.prop_bodies.is_empty() || self.scene.copies > 1,
                "{action:?}: only the tower block has it"
            );
            Ok(())
        };
        let add = |list: &mut Vec<Due>, step: u64, origin: [f64; 3], dir: [f64; 3], speed: Option<f64>, arc: Option<bool>| {
            let v = engine_fracture::plan::shot_velocity(
                sh.center,
                sh.extent,
                sh.radius,
                origin,
                dir,
                Some(speed.unwrap_or_else(|| sh.base_speed() * sh.speed_scale)),
                arc.unwrap_or(sh.arc),
            );
            list.push(Due {
                step,
                origin,
                velocity: v,
                spin: [0.0; 3],
                mass: sh.mass,
                radius: sh.radius,
            });
        };
        let mut due = Vec::new();
        match action {
            LabAction::Shot { origin, direction } => {
                anyhow::ensure!(
                    origin.iter().chain(direction).all(|x| x.is_finite()),
                    "invalid shot ray"
                );
                add(&mut due, base, *origin, *direction, None, None);
            }
            LabAction::RowShot => {
                let r = self
                    .scene
                    .row_shot
                    .ok_or_else(|| anyhow::anyhow!("the scene has no row shot"))?;
                add(&mut due, base, r.origin, r.direction, None, None);
            }
            LabAction::Volley | LabAction::Topple | LabAction::Meteor => {
                block()?;
                let shots = match action {
                    LabAction::Volley => {
                        let (raking, base_shots, every) = engine_fracture::towers::VOLLEY;
                        engine_fracture::towers::volley(&self.scene, raking, base_shots, every)
                    }
                    LabAction::Topple => engine_fracture::towers::topple(&self.scene, 0, 12, 20),
                    _ => vec![engine_fracture::towers::meteor(&self.scene)],
                };
                for s in shots {
                    // A shot with its own speed flies straight, as the lab fires them.
                    let arc = s.speed.map(|_| false);
                    add(&mut due, base + s.step as u64, s.origin, s.dir, s.speed, arc);
                }
            }
            LabAction::PackShot(i) => {
                let s = self
                    .pack_shots
                    .get(*i as usize)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("no authored shot {i} (the scene has {})", self.pack_shots.len()))?;
                let p = self.shooting.placement;
                due.push(Due {
                    step: base,
                    origin: [0, 1, 2].map(|k| s.origin[k] + p[k]),
                    velocity: s.velocity,
                    spin: s.angular_velocity,
                    mass: s.mass,
                    radius: s.radius,
                });
            }
        }
        self.schedule.extend(due);
        self.schedule.sort_by_key(|d| d.step);
        Ok(())
    }

    /// Shots queued and not yet launched.
    pub fn queued(&self) -> usize {
        self.schedule.len()
    }

    /// Clear the queue.
    pub fn clear_queue(&mut self) {
        self.schedule.clear();
    }

    /// Launch every queued shot due at the world's next step. Returns their ball slots
    /// (scene bodies), in order.
    pub fn launch_due(&mut self, world: &mut World) -> anyhow::Result<Vec<u32>> {
        let now = world.step_count();
        let n = self.schedule.iter().take_while(|d| d.step <= now).count();
        let due: Vec<Due> = self.schedule.drain(..n).collect();
        let mut out = Vec::with_capacity(due.len());
        for d in due {
            out.push(self.fire(world, d.origin, d.velocity, d.spin, d.mass, d.radius)?);
        }
        Ok(out)
    }

    /// One step: the due shots launched, then the world's step.
    pub fn step(&mut self, world: &mut World) -> anyhow::Result<Vec<u32>> {
        let fired = self.launch_due(world)?;
        world.step()?;
        Ok(fired)
    }

    /// A ball: bond models, the next parked slot launched; stress models, a new ball (the
    /// oldest of [`STRESS_SHOTS`] destroyed) registered as a projectile.
    fn fire(&mut self, world: &mut World, origin: [f64; 3], v: [f64; 3], w: [f64; 3], mass: f64, radius: f64) -> anyhow::Result<u32> {
        let (m, r) = (mass as f32, radius as f32);
        let i = 0.4 * m * r * r;
        let mass_data = MassData {
            mass: m,
            center: [0.0; 3],
            rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
        };
        let at = Transform::new(f3(origin), [0.0, 0.0, 0.0, 1.0]);
        let count = self.scene.projectile_count.max(1);
        let slot = self.scene.projectile_start + self.fired % count;
        if self.settings.bonds.stress().is_none() {
            anyhow::ensure!(!self.shots.is_empty(), "the scene has no ball slots");
            let body = self.shots[(self.fired % count) as usize];
            // Continuous collision is set on a static body (a reused slot is parked again
            // first), then it is launched.
            if world.body_type(body)? != BodyType::Static {
                let parked = world.body_initial_transform(body)?;
                world.body_set_type(body, BodyType::Static, parked, [0.0; 3], [0.0; 3])?;
            }
            world.body_set_bullet(body, self.shooting.ccd)?;
            world.body_set_type(body, BodyType::Dynamic, at, f3(v), f3(w))?;
            world.body_set_mass_data(body, mass_data)?;
        } else {
            let ball = crate::play::balls::Ball {
                mass: m,
                radius: r,
                material: self.material,
                ccd: self.shooting.ccd,
            };
            let step = world.step_count();
            self.balls.fire(world, &ball, at.position, f3(v), f3(w), step)?;
        }
        self.fired += 1;
        Ok(slot)
    }

    /// Directed debris (engine-fracture guidance.rs): `settings` (`None`: off) and its
    /// zones, anchors as scene bodies. Every piece of every destructible is re-armed (a
    /// stress model's pieces, and the slots its splits may take).
    pub fn set_guidance(
        &mut self,
        world: &mut World,
        settings: Option<GuidanceSettings>,
        zones: &[engine_schema::guidance::ZoneDesc],
    ) -> anyhow::Result<()> {
        let on = settings.is_some();
        let s = settings.clone().unwrap_or(GuidanceSettings {
            enabled: false,
            ..GuidanceSettings::default()
        });
        world.set_guidance(&s)?;
        let mut out = Vec::with_capacity(zones.len());
        for (k, z) in zones.iter().enumerate() {
            let anchor = match z.anchor_body {
                Some(b) => ZoneAnchor::Body(
                    self.scene_body(world, b)?
                        .ok_or_else(|| anyhow::anyhow!("zone {k}: scene body {b} is no body"))?,
                ),
                None => ZoneAnchor::World(z.position),
            };
            out.push(GuidanceZone {
                anchor,
                anchor_relative: z.anchor_relative,
                yaw: z.yaw,
                offset: z.offset,
                shape: z.shape,
                weight: z.weight,
                quota: z.quota,
                active: z.active,
                tags: z.tags,
                motion: z.motion.clone(),
            });
        }
        world.set_guidance_zones(&out)?;
        let mask = if on { u32::MAX } else { 0 };
        for &d in self.destructibles.iter().chain(&self.props) {
            world.destructible_set_guided(d, mask)?;
        }
        if self.draws_poses() {
            // The stress models' pieces break onto reserved slots: those are armed too.
            world.guide_fracture_spares(mask)?;
        }
        world.reset_guidance_claims()?;
        self.guidance = settings.map(|s| (s, zones.to_vec()));
        Ok(())
    }

    /// The body a scene body is now (`None`: none).
    pub fn scene_body(&self, world: &World, b: u32) -> anyhow::Result<Option<BodyId>> {
        if !self.draws_poses() {
            return Ok(self
                .slot_of
                .get(b as usize)
                .copied()
                .filter(|&s| s != u32::MAX)
                .and_then(|s| world.body_at(s)));
        }
        Ok(match self.drawn.get(b as usize) {
            Some(Drawn::Body(id)) => Some(*id),
            Some(Drawn::Chunk { copy, chunk, .. }) => {
                Some(world.destructible_chunk_body(self.destructibles[*copy as usize], *chunk)?)
            }
            _ => None,
        })
    }

    /// The bond models' reset: the scene as placed. The stress models place again.
    pub fn reset(&mut self, world: &mut World) -> anyhow::Result<()> {
        let saved = self
            .saved
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("the stress models reset by placing the scene again"))?;
        world.restore_state(saved)?;
        self.fired = 0;
        self.schedule.clear();
        Ok(())
    }

    /// Every number a host shows, from the world's last polled step.
    pub fn stats(&self, world: &World) -> anyhow::Result<LabStats> {
        let mut st = LabStats {
            steps: world.step_count(),
            shots: self.fired,
            physics_hz: self.settings.physics_hz,
            ..LabStats::default()
        };
        for &d in &self.destructibles {
            let s = world.destructible_stats(d)?;
            st.chunks += s.chunks;
            st.bonds += s.bonds;
            st.bonds_intact += s.bonds_intact;
            st.bodies += s.actors;
            st.crushed += s.crushed;
        }
        if let Some(c) = world.counters() {
            st.awake = c.awake_count;
            st.contacts = c.contact_count;
        }
        // The whole tick (rigid step and fracture) where the world times it, else the rigid
        // step's own passes.
        st.gpu_step_ms = world
            .tick_gpu_ms()
            .or_else(|| world.last_step_gpu_time().map(|t| t.total_ms));
        if let Some(ms) = st.gpu_step_ms.filter(|&ms| ms > 0.0) {
            st.max_steps_per_second = Some(1000.0 / ms);
            st.realtime_share = Some(self.settings.physics_hz as f64 * ms / 1000.0);
        }
        Ok(st)
    }

    /// Per bond of the scene (its destructibles' bonds in order), whether it holds.
    pub fn bonds_intact(&self, world: &World) -> anyhow::Result<Vec<bool>> {
        let mut out = Vec::new();
        for &d in &self.destructibles {
            let n = world.destructible_stats(d)?.bonds;
            for k in 0..n {
                out.push(world.destructible_bond_intact(d, k)?);
            }
        }
        Ok(out)
    }

    /// The stress models' draw: a [`RECORD_FLOATS`]-float record per scene body (its pose,
    /// its body's flags in word 11), as of the last polled step. Chunks at their bodies'
    /// transforms times their lab poses, props and balls at theirs, the rest parked.
    pub fn poses(&self, world: &World) -> anyhow::Result<Vec<f32>> {
        let n = self.drawn.len();
        let mut out = vec![0.0f32; RECORD_FLOATS * n];
        // Word 3: the record's rigid body slot + 1 (0: none), for the renderer's Bodies view.
        let mut put = |i: usize, t: Transform, flags: u32, body: Option<BodyId>| {
            let o = RECORD_FLOATS * i;
            out[o..o + 3].copy_from_slice(&t.position);
            out[o + 3] = f32::from_bits(body.map_or(0, |b| b.index() + 1));
            out[o + 4..o + 8].copy_from_slice(&t.rotation);
            out[o + 11] = f32::from_bits(flags);
        };
        let parked = Transform::new(PARKED, [0.0, 0.0, 0.0, 1.0]);
        let mut stress = Vec::with_capacity(self.destructibles.len());
        for &d in &self.destructibles {
            stress.push(world.destructible_chunk_stress(d)?);
        }
        let mut stressed: Vec<(usize, f32)> = Vec::new();
        for (i, d) in self.drawn.iter().enumerate() {
            match *d {
                Drawn::Ground => put(i, Transform::IDENTITY, 0, None),
                Drawn::None => put(i, parked, 0, None),
                Drawn::Body(b) => match world.body_transform(b).ok().flatten() {
                    Some(t) => put(i, t, body_flags(world, b), Some(b)),
                    None => put(i, parked, 0, None),
                },
                Drawn::Chunk { copy, chunk, rest } => {
                    let dest = self.destructibles[copy as usize];
                    let body = world.destructible_chunk_body(dest, chunk)?;
                    if !world.body_is_valid(body) {
                        // Crushed: its body is gone.
                        put(i, parked, 0, None);
                        continue;
                    }
                    let t = world.body_transform(body)?.unwrap_or(world.body_initial_transform(body)?);
                    put(i, compose(&t, &rest), body_flags(world, body), Some(body));
                    // The stress view (`World::set_stress_view`): 1 + the chunk's largest
                    // bond load over strength in word 12 (0: none solved).
                    if let Some(v) = stress.get(copy as usize).and_then(|s| s.as_ref()).and_then(|s| s.get(chunk as usize)) {
                        stressed.push((i, 1.0 + v[0].max(v[1]).max(v[2]).max(v[3]).max(0.0)));
                    }
                }
            }
        }
        let count = self.scene.projectile_count.max(1) as usize;
        let first = self.fired as usize - self.balls.len();
        for (k, &(b, _)) in self.balls.live().iter().enumerate() {
            let slot = self.scene.projectile_start as usize + (first + k) % count;
            if let Some(t) = world.body_transform(b).ok().flatten() {
                if slot < n {
                    put(slot, t, body_flags(world, b), Some(b));
                }
            }
        }
        let peak = stressed.iter().map(|s| s.1 - 1.0).fold(0.0f32, f32::max);
        for (i, u) in stressed {
            let u = u - 1.0;
            let shown = match self.stress_scale {
                StressScale::Absolute => u,
                StressScale::Log => {
                    if u >= 1.0 {
                        1.0
                    } else if u > 0.0 {
                        ((u.log10() + StressScale::LOG_DECADES) / StressScale::LOG_DECADES).clamp(0.0, 1.0)
                    } else {
                        0.0
                    }
                }
                StressScale::Relative => if peak > 0.0 { u / peak } else { 0.0 },
            };
            out[RECORD_FLOATS * i + 12] = 1.0 + shown;
        }
        Ok(out)
    }

    /// The stress view's colour scale from the next [`Lab::poses`] on.
    pub fn set_stress_scale(&mut self, scale: StressScale) {
        self.stress_scale = scale;
    }
}

/// The street props as a destructible of loose pieces (no bonds), each a dynamic body.
fn place_props(world: &mut World, s: &Scene, material: SurfaceMaterial) -> anyhow::Result<(DestructibleId, Vec<BodyId>)> {
    // A scene of the ground and the props alone (their colliders and hulls as the scene
    // has them), as one loose asset.
    let mut sub = s.clone();
    let keep: Vec<u32> = std::iter::once(0).chain(s.prop_bodies.iter().copied()).collect();
    let mut map = vec![u32::MAX; s.bodies.len()];
    for (k, &b) in keep.iter().enumerate() {
        map[b as usize] = k as u32;
    }
    sub.bodies = keep.iter().map(|&b| s.bodies[b as usize].clone()).collect();
    sub.colliders = s
        .colliders
        .iter()
        .filter(|c| map[c.body as usize] != u32::MAX)
        .map(|c| {
            let mut c = c.clone();
            c.body = map[c.body as usize];
            c
        })
        .collect();
    sub.joints = Vec::new();
    sub.projectile_start = sub.bodies.len() as u32;
    sub.projectile_count = 0;
    sub.prop_bodies = Vec::new();
    sub.props = Vec::new();
    let desc = DestructibleAssetDesc::from_scene(&sub)?;
    let asset = world.create_destructible_asset(&desc)?;
    let d = world.create_destructible(&DestructibleDef {
        asleep: true,
        shape: ShapeDef {
            material,
            update_body_mass: false,
            ..ShapeDef::default()
        },
        ..DestructibleDef::new(asset)
    })?;
    let bodies = (0..s.prop_bodies.len() as u32)
        .map(|k| world.destructible_chunk_body(d, k))
        .collect::<anyhow::Result<_>>()?;
    Ok((d, bodies))
}

/// Chunk crushing as the lab sets it: off, the scene's own parameters, or those and the
/// cited presets for the rest. Crushed chunks are removed (dust).
fn crush_for(mode: LabCrush, stress: &StressSource) -> anyhow::Result<Option<Crush>> {
    if mode == LabCrush::Off {
        return Ok(None);
    }
    let mut materials = Vec::new();
    if mode == LabCrush::Presets {
        for (name, m) in &stress.scene()?.materials {
            if m.crush.is_some() {
                continue;
            }
            let base = name.split(['#', '@']).next().unwrap_or(name);
            if let Some(p) = stress_ref::crush::presets::for_material(base) {
                if !materials.iter().any(|(n, _): &(String, CrushMaterial)| n == base) {
                    materials.push((
                        base.to_string(),
                        CrushMaterial {
                            cap_pressure: p.cap_pressure,
                            cohesion: p.cohesion,
                            friction_slope: p.friction_slope,
                            crush_energy: p.crush_energy,
                            crush_viscosity: p.crush_viscosity,
                        },
                    ));
                }
            }
        }
    }
    Ok(Some(Crush {
        mass: CrushMass::Remove,
        materials,
        max_per_step: 0,
    }))
}

/// The stress import's first chunks are the asset's chunks (the lab scene's bodies after
/// the ground), in order: each chunk's centre (stress frame, turned Y up, plus the
/// placement) on its chunk.
pub fn check_lab_chunk_order(desc: &DestructibleAssetDesc) -> anyhow::Result<()> {
    let stress = desc
        .stress
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no stress import"))?;
    let n = desc.chunks.len();
    let scene = stress.scene()?;
    let p = stress.placement();
    let rot = |q: [f64; 4], v: [f64; 3]| {
        let q = glam::DQuat::from_xyzw(q[1], q[2], q[3], q[0]);
        (q * glam::DVec3::from(v)).to_array()
    };
    let mut k = 0;
    for body in &scene.bodies {
        for ch in &body.chunks {
            if k >= n {
                return Ok(());
            }
            let c = rot(body.orientation, ch.center);
            let zup = [body.position[0] + c[0], body.position[1] + c[1], body.position[2] + c[2]];
            // Z up to Y up: (x, y, z) -> (x, z, -y).
            let at = [p[0] + zup[0], p[1] + zup[2], p[2] - zup[1]];
            let dist = |i: usize| {
                let b = desc.chunks[i - 1].pose.position;
                (0..3).map(|j| (at[j] - b[j] as f64).powi(2)).sum::<f64>().sqrt()
            };
            let d = dist(1 + k);
            // The import's centre is its box's, the lab's the mesh's centre of mass: within
            // the chunk's bounds, and nearer its own body than the bodies numbered next to it.
            let bound = ch.half_extents.iter().map(|h| h * h).sum::<f64>().sqrt();
            let neighbours = [k, k + 2].into_iter().filter(|&i| i >= 1 && i <= n).map(dist);
            anyhow::ensure!(
                d <= bound && neighbours.into_iter().all(|o| d <= o),
                "the stress import's chunk {k} is {d:.3} m from the lab scene's chunk {k} (chunk order differs)"
            );
            k += 1;
        }
    }
    anyhow::ensure!(k >= n, "the stress import has {k} chunks for the asset's {n}");
    Ok(())
}

fn body_flags(world: &World, body: BodyId) -> u32 {
    let dynamic = world.body_type(body).is_ok_and(|t| t == BodyType::Dynamic);
    let awake = world.body_is_awake(body).unwrap_or(false);
    u32::from(dynamic) * FLAG_DYNAMIC | if awake { FLAG_AWAKE } else { FLAG_SLEEPING }
}

fn f3(v: [f64; 3]) -> [f32; 3] {
    v.map(|x| x as f32)
}

fn f4(v: [f64; 4]) -> [f32; 4] {
    v.map(|x| x as f32)
}

fn compose(a: &Transform, b: &Transform) -> Transform {
    let (qa, pa) = (glam::Quat::from_array(a.rotation), glam::Vec3::from(a.position));
    let (qb, pb) = (glam::Quat::from_array(b.rotation), glam::Vec3::from(b.position));
    Transform::new((pa + qa * pb).to_array(), (qa * qb).normalize().to_array())
}
