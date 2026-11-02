//! The fracture lab (`engine-webgpu/fracture/`) on [`World`]: its scenes become
//! destructibles, its shots bodies, and the page steps and draws the world.
//!
//! - **Bond modes** (None, breakable D6 bonds): the lab's `engine_fracture::Scene` is
//!   one `StrengthModel::Bonds` destructible (`DestructibleAssetDesc::from_scene`, the
//!   lab's solver `lab_solver`). Scene body `i` is world body slot `i` (the ground first,
//!   then every chunk in order), so the lab's renderer binds the world's body and joint
//!   buffers in place; the parked shots and street props, drawn once, are remapped
//!   ([`LabScene::remap`]).
//! - **Stress mode, threshold** (the lab's v0 stress solver): the same source as a
//!   `StrengthModel::Threshold` destructible (one per copy of the cuboid houses), on the
//!   world's shared GPU fracture. Chunks move between bodies as they split, so the page
//!   draws them from poses made here each frame ([`WebWorld::lab_poses`]), as Broadside.
//!
//! The bond preload (D6 bonds settled under their own weight) runs as a job the page
//! advances each frame ([`WebWorld::lab_preload_advance`]): nothing here waits.

use crate::{err, res, WebFrame, WebWorld};
use engine_fracture::scene::{Scene, SHAPE_PLANE};
use engine_fracture_wasm::Scene as JsScene;
use rubble::*;
use wasm_bindgen::prelude::*;

/// Floats a body record takes (`BodyState`, the draw shader's `BodyPose`).
const RECORD_FLOATS: usize = 28;
/// Where a body not in play is drawn (far under the ground).
const PARKED: [f32; 3] = [0.0, -1.0e4, 0.0];
/// Body flags the draw shader colours by (`invI.w`): dynamic, awake.
const FLAG_DYNAMIC: u32 = 1;
const FLAG_SLEEPING: u32 = 4;
const FLAG_AWAKE: u32 = 8;

fn js<T: serde::Serialize>(v: &T) -> Result<JsValue, JsValue> {
    serde::Serialize::serialize(v, &serde_wasm_bindgen::Serializer::json_compatible()).map_err(err)
}

fn next_pow2(n: u64) -> u32 {
    n.max(1).next_power_of_two() as u32
}

fn f3(v: [f64; 3]) -> [f32; 3] {
    v.map(|x| x as f32)
}

fn f4(v: [f64; 4]) -> [f32; 4] {
    v.map(|x| x as f32)
}

/// A lab source as a destructible asset, before any world (`LabAsset.bonds` etc.).
#[wasm_bindgen]
pub struct LabAsset {
    desc: DestructibleAssetDesc,
    model: StrengthModel,
    /// The asset in the world it was registered in ([`WebWorld::lab_register`]).
    id: Option<DestructibleAssetId>,
    /// A pack's authored shots (threshold, scene packs).
    pack_shots: Vec<engine_fracture::pack::PackShot>,
}

#[wasm_bindgen]
impl LabAsset {
    /// The lab scene as a `Bonds` destructible: its chunks, bonds (none for the None
    /// mode's `withoutJoints` scenes) and solver.
    pub fn bonds(scene: &JsScene) -> Result<LabAsset, JsValue> {
        Ok(LabAsset {
            desc: res(DestructibleAssetDesc::from_scene(scene.scene()))?,
            model: StrengthModel::Bonds,
            id: None,
            pack_shots: Vec::new(),
        })
    }

    /// A threshold destructible from the lab's stress-mode sources: `kind` `'glb'` (the
    /// GLB at `level`), `'pack'` (a scene pack's JSON) or `'cuboid'` (one cuboid house;
    /// the page's layout places copies).
    pub fn threshold(kind: &str, bytes: &[u8], level: u8) -> Result<LabAsset, JsValue> {
        let (desc, pack_shots) = match kind {
            "glb" => (
                res(DestructibleAssetDesc::from_glb(bytes, level))?,
                Vec::new(),
            ),
            "pack" => (
                res(DestructibleAssetDesc::from_pack(bytes))?,
                res(engine_fracture::pack::pack_shots(bytes))?,
            ),
            "cuboid" => (
                res(DestructibleAssetDesc::cuboid_house(&Default::default()))?,
                Vec::new(),
            ),
            _ => return Err(err(format!("LabAsset.threshold: unknown kind {kind}"))),
        };
        if let Some(s) = &desc.stress {
            res(s.scene().map(|_| ()))?;
        }
        Ok(LabAsset {
            desc,
            model: StrengthModel::Threshold,
            id: None,
            pack_shots,
        })
    }

    /// Chunks, collision parts and bonds (the panel's fragments, hulls and bonds).
    pub fn totals(&self) -> Vec<u32> {
        let parts: usize = self.desc.chunks.iter().map(|c| c.shapes.len()).sum();
        vec![
            self.desc.chunks.len() as u32,
            parts as u32,
            self.desc.bonds.len() as u32,
        ]
    }

    /// The `WorldDef` (as `createWorld` takes it) the lab runs `scene` with: its
    /// solver, sleep and capacity as the lab's own plan sized them. `opts`:
    /// `{ copies, physicsHz, gpuTiming }` (threshold: copies of the asset placed, the frame
    /// rate; bonds: `'phases'` times each phase of the step).
    #[wasm_bindgen(js_name = worldDef)]
    pub fn world_def(&self, scene: &JsScene, opts: JsValue) -> Result<JsValue, JsValue> {
        let o = Opts::read(opts)?;
        let s = scene.scene();
        let def = match self.model {
            StrengthModel::Bonds => bonds_world(s, &self.desc, &o)?,
            StrengthModel::Threshold => threshold_world(s, &self.desc, &o)?,
        };
        Ok(def)
    }

    /// The pack's authored shots: `{ label, mass, radius, speed, launchAt, note }`.
    #[wasm_bindgen(js_name = packShots)]
    pub fn pack_shots(&self) -> Result<JsValue, JsValue> {
        let v: Vec<_> = self
            .pack_shots
            .iter()
            .map(|s| {
                let speed = s.velocity.iter().map(|x| x * x).sum::<f64>().sqrt();
                serde_json::json!({"label": s.label, "mass": s.mass, "radius": s.radius, "speed": speed, "launchAt": s.launch_at, "note": s.note})
            })
            .collect();
        js(&v)
    }
}

/// Load options (`LabAsset.worldDef`, `WebWorld.labPlace`).
#[derive(Default)]
struct Opts {
    copies: u32,
    physics_hz: f32,
    /// Threshold chunk crushing: 0 off, 1 the scene's parameters, 2 with presets.
    crush: u32,
    strength_scale: f32,
    /// `WorldDef.gpuTiming` (`'step'` by default; `'phases'` for per-phase GPU times).
    gpu_timing: Option<String>,
}

impl Opts {
    fn read(v: JsValue) -> Result<Opts, JsValue> {
        let mut o = Opts {
            copies: 1,
            strength_scale: 1.0,
            ..Opts::default()
        };
        if v.is_undefined() || v.is_null() {
            return Ok(o);
        }
        let mut f = crate::defs::Fields::of("lab options", v)?;
        f.u32("copies", &mut o.copies)?;
        f.f32("physicsHz", &mut o.physics_hz)?;
        f.u32("crush", &mut o.crush)?;
        f.f32("strengthScale", &mut o.strength_scale)?;
        o.gpu_timing = f.string("gpuTiming")?;
        f.finish()?;
        o.copies = o.copies.max(1);
        Ok(o)
    }
}

/// The lab's surface: every collider at the scene's friction, no restitution, the
/// plan's 1 cm contact offset (`Planner::material_table`).
fn lab_material(s: &Scene) -> SurfaceMaterial {
    SurfaceMaterial {
        friction: s.params.friction,
        restitution: 0.0,
        contact_offset: 0.01,
        ..Default::default()
    }
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

fn capacity_js(c: &WorldCapacity) -> serde_json::Value {
    serde_json::json!({
        "maxBodies": c.max_bodies, "maxColliders": c.max_colliders, "maxContacts": c.max_contacts,
        "maxJoints": c.max_joints, "maxCandidates": c.max_candidates, "maxColors": c.max_colors,
        "maxCommands": c.max_commands, "maxHullVertices": c.max_hull_vertices,
    })
}

fn def_js(def: serde_json::Value, solver: &AvbdParams) -> Result<JsValue, JsValue> {
    let o = js(&def)?;
    js_sys::Reflect::set(&o, &"solver".into(), &crate::defs::solver_to_js(solver))?;
    Ok(o)
}

/// The bond modes' world: the lab plan's capacities (`Planner::new`: bodies, colliders,
/// contacts and candidates by its budget, colours), sleep on, persistent contacts.
fn bonds_world(s: &Scene, desc: &DestructibleAssetDesc, o: &Opts) -> Result<JsValue, JsValue> {
    let solver = desc
        .solver
        .ok_or_else(|| err("the lab asset has no solver"))?;
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
        // Launches and resets are commands in World (the plan wrote records).
        max_commands: 2048,
        max_hull_vertices: (hull_vertices(desc) + 64).max(1024),
    };
    def_js(
        serde_json::json!({
            "gravity": [0.0, solver.gravity_y, 0.0],
            "enableSleep": true,
            "contacts": "persistent",
            "capacity": capacity_js(&c),
            "guidance": { "maxZones": 256, "planBudget": 256 },
            "readbackFrames": READBACK_FRAMES,
            "gpuTiming": o.gpu_timing.as_deref().unwrap_or("step"),
        }),
        &solver,
    )
}

/// Steps in flight before the page must wait for a readback (World's default is 3). A
/// heavy step (the towers volley, ~6 ms of GPU) takes longer than a display frame to come
/// back, and with 3 slots the page could take only ~80 of 120 ticks a second; the old JS
/// host queued its ticks without that limit.
const READBACK_FRAMES: u32 = 8;

/// Shots the stress mode keeps in flight (the lab's 16; past them the oldest goes).
const STRESS_SHOTS: u32 = 16;

/// The threshold mode's world: the lab's `ThresholdWorld` capacity for every copy, its
/// solver (never asleep, canonical contacts), the frame rate the panel set.
fn threshold_world(_s: &Scene, desc: &DestructibleAssetDesc, o: &Opts) -> Result<JsValue, JsValue> {
    let stress = desc
        .stress
        .as_ref()
        .ok_or_else(|| err("the asset has no stress import"))?;
    let mut solver = desc
        .solver_for(StrengthModel::Threshold)
        .ok_or_else(|| err("no threshold solver"))?;
    if o.physics_hz > 0.0 {
        solver.dt = 1.0 / o.physics_hz;
    }
    let scene = res(stress.threshold_scene(1.0, solver.dt))?;
    let chunks: usize =
        scene.bodies.iter().map(|b| b.chunks.len()).sum::<usize>() * o.copies as usize;
    let one = stress_rubble::capacity_for(&scene);
    let mut c = one.clone();
    c.max_colliders = one.max_colliders * o.copies;
    c.max_hull_vertices = one.max_hull_vertices * o.copies;
    c.max_bodies = c
        .max_bodies
        .max(((chunks + STRESS_SHOTS as usize + 4) as u32).next_power_of_two());
    c.max_contacts = stress_rubble::threshold::contact_capacity(
        chunks + STRESS_SHOTS as usize,
        c.max_contacts,
        1 << 30,
    );
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    c.max_commands = c.max_commands.max(2048);
    def_js(
        serde_json::json!({
            "gravity": [0.0, solver.gravity_y, 0.0],
            "enableSleep": false,
            "contacts": "canonical",
            "readbackFrames": READBACK_FRAMES,
            "capacity": capacity_js(&c),
        }),
        &solver,
    )
}

/// What a scene body is drawn from in the threshold mode.
#[derive(Clone, Copy)]
enum Drawn {
    Ground,
    /// Chunk `chunk` of destructible `copy`, at `rest` (its lab pose) in its body.
    Chunk {
        copy: u32,
        chunk: u32,
        rest: Transform,
    },
    /// A shot slot (`projectile_start + k`), or nothing.
    None,
}

/// The lab's shot rule (`engine_fracture::plan::shot_velocity`) and its settings.
#[derive(Clone, Copy)]
struct Shooting {
    center: [f64; 3],
    extent: f64,
    radius: f64,
    mass: f64,
    speed: Option<f64>,
    arc: bool,
    speed_scale: f64,
    ccd: bool,
    /// The pack frame's origin in the lab's (threshold packs: the scene placement).
    placement: [f64; 3],
}

/// A lab scene placed in a world ([`WebWorld::lab_place`]).
#[wasm_bindgen]
pub struct LabScene {
    model: StrengthModel,
    destructibles: Vec<DestructibleId>,
    /// Per scene body, its world slot (`u32::MAX`: none). Bonds only.
    slot_of: Vec<u32>,
    /// Bonds: the parked shot bodies, one per projectile slot. Threshold: the live balls,
    /// oldest first.
    shots: Vec<BodyId>,
    fired: u32,
    shooting: Shooting,
    projectile_start: u32,
    projectile_count: u32,
    material: SurfaceMaterial,
    /// Threshold: what each scene body is drawn from.
    drawn: Vec<Drawn>,
    pack_shots: Vec<engine_fracture::pack::PackShot>,
    /// Bonds: the placed scene, for the reset key.
    saved: Option<WorldSnapshot>,
}

#[wasm_bindgen]
impl LabScene {
    /// Shots fired since placement (or the last reset).
    #[wasm_bindgen(getter)]
    pub fn shots(&self) -> u32 {
        self.fired
    }

    /// The scene's shot speed (m/s) before the speed scale.
    #[wasm_bindgen(getter, js_name = shotSpeed)]
    pub fn shot_speed(&self) -> f64 {
        engine_fracture::plan::default_shot_speed(self.shooting.extent, self.shooting.speed)
    }

    #[wasm_bindgen(getter, js_name = shotMass)]
    pub fn shot_mass(&self) -> f64 {
        self.shooting.mass
    }

    #[wasm_bindgen(getter, js_name = shotRadius)]
    pub fn shot_radius(&self) -> f64 {
        self.shooting.radius
    }

    /// The mass (kg) of the shots fired from now on.
    #[wasm_bindgen(js_name = setShotMass)]
    pub fn set_shot_mass(&mut self, kg: f64) {
        if kg.is_finite() && kg > 0.0 {
            self.shooting.mass = kg;
        }
    }

    /// The speed of the shots fired from now on, as a multiple of the scene's.
    #[wasm_bindgen(js_name = setSpeedScale)]
    pub fn set_speed_scale(&mut self, k: f64) {
        if k.is_finite() && k > 0.0 {
            self.shooting.speed_scale = k;
        }
    }

    /// Continuous collision for the shots fired from now on (bond modes; the threshold
    /// mode's balls have none, as the lab's ThresholdWorld).
    #[wasm_bindgen(js_name = setCcd)]
    pub fn set_ccd(&mut self, on: bool) {
        self.shooting.ccd = on;
    }

    /// The world slot of each scene body (`-1`: none), for snapshots in scene order.
    #[wasm_bindgen(js_name = slotOf)]
    pub fn slot_of(&self) -> Vec<i32> {
        self.slot_of
            .iter()
            .map(|&s| if s == u32::MAX { -1 } else { s as i32 })
            .collect()
    }

    /// A render stream (10 floats per vertex, body id last) drawn once, with its body ids
    /// turned into world slots: the shots and the street props.
    pub fn remap(&self, mut stream: Vec<f32>) -> Vec<f32> {
        if self.model == StrengthModel::Bonds {
            for v in stream.chunks_exact_mut(10) {
                if let Some(&s) = self.slot_of.get(v[9] as usize) {
                    if s != u32::MAX {
                        v[9] = s as f32;
                    }
                }
            }
        }
        stream
    }

    /// The pack's authored shots, as `LabAsset.packShots`.
    #[wasm_bindgen(js_name = packShots)]
    pub fn pack_shots(&self) -> Result<JsValue, JsValue> {
        LabAsset {
            desc: DestructibleAssetDesc::default(),
            model: self.model,
            id: None,
            pack_shots: self.pack_shots.clone(),
        }
        .pack_shots()
    }
}

#[wasm_bindgen]
impl WebWorld {
    /// A world on `other`'s device (`World::gpu_context`): a page that rebuilds its scene
    /// keeps one device for its renderers.
    #[wasm_bindgen(js_name = createOnDeviceOf)]
    pub fn create_on_device_of(def: JsValue, other: &WebWorld) -> Result<WebWorld, JsValue> {
        let def = crate::defs::world_def(def)?;
        let world = res(World::new(other.world.gpu_context(), &def))?;
        Ok(WebWorld { world })
    }

    /// The joint records (binding 5), by joint slot: bond lines hide once broken.
    #[wasm_bindgen(js_name = jointBuffer)]
    pub fn joint_buffer(&self) -> JsValue {
        crate::gpu_buffer(self.world.sim().joint_buffer())
    }

    /// Register the asset in this world (once).
    #[wasm_bindgen(js_name = labRegister)]
    pub fn lab_register(&mut self, asset: &mut LabAsset) -> Result<(), JsValue> {
        if asset.id.is_none() {
            asset.id = Some(res(self.world.create_destructible_asset(&asset.desc))?);
        }
        Ok(())
    }

    /// Start the lab's bond preload of a registered asset (`undefined`: it has no bonds).
    #[wasm_bindgen(js_name = labPreloadBegin)]
    pub fn lab_preload_begin(&mut self, asset: &LabAsset) -> Result<Option<LabPreload>, JsValue> {
        let id = asset.id.ok_or_else(|| err("labRegister the asset first"))?;
        Ok(res(self.world.begin_destructible_preload(id))?.map(|job| LabPreload { job }))
    }

    /// Advance the preload as far as free readback slots allow (never waits):
    /// `undefined` until it is decided, then whether it settled.
    #[wasm_bindgen(js_name = labPreloadAdvance)]
    pub fn lab_preload_advance(&mut self, p: &mut LabPreload) -> Result<Option<bool>, JsValue> {
        res(self.world.advance_destructible_preload(&mut p.job, false))
    }

    /// The lab scene of a preloaded asset (`labPreloadAdvance` done): every piece asleep
    /// at its settled pose (`Scene::settled`), as the lab's bond preloading made it. The
    /// page lays out copies of it, each starting from the same settled house.
    #[wasm_bindgen(js_name = labSettledScene)]
    pub fn lab_settled_scene(&self, asset: &LabAsset, scene: &JsScene) -> Result<JsScene, JsValue> {
        let id = asset.id.ok_or_else(|| err("labRegister the asset first"))?;
        let desc = res(self.world.destructible_asset(id))?;
        let s = scene.scene();
        let Some(p) = &desc.preload else {
            return Ok(JsScene::from_scene(s.clone()));
        };
        // The chunks' centres of mass where the settle left them (a body origin's pose
        // times the chunk's principal frame), as the body records `settled` reads.
        let mut snapshot = vec![0.0f32; RECORD_FLOATS * s.bodies.len()];
        let shots = s.projectile_start..s.projectile_start + s.projectile_count;
        let pieces = (1..s.bodies.len() as u32).filter(|b| !shots.contains(b));
        for ((b, chunk), origin) in pieces.zip(&desc.chunks).zip(&p.body_poses) {
            let t = compose(origin, &chunk.pose);
            let o = RECORD_FLOATS * b as usize;
            snapshot[o..o + 3].copy_from_slice(&t.position);
            snapshot[o + 4..o + 8].copy_from_slice(&t.rotation);
        }
        Ok(JsScene::from_scene(res(
            s.settled(&snapshot, p.steps, p.settled)
        )?))
    }

    /// Place a registered asset as the lab's `scene`: the ground, the destructible(s) and
    /// the shots (the scene's ball: `withProjectileRadius` scales it). `opts`: `{ copies, crush,
    /// strengthScale }` (threshold).
    #[wasm_bindgen(js_name = labPlace)]
    pub fn lab_place(
        &mut self,
        asset: &LabAsset,
        scene: &JsScene,
        opts: JsValue,
    ) -> Result<LabScene, JsValue> {
        let o = Opts::read(opts)?;
        let id = asset.id.ok_or_else(|| err("labRegister the asset first"))?;
        let s = scene.scene();
        match asset.model {
            StrengthModel::Bonds => res(self.place_bonds(id, s)),
            StrengthModel::Threshold => res(self.place_threshold(id, asset, s, &o)),
        }
    }

    /// Fire a shot from `origin` along `direction` by the lab's rule (speed: the scene's
    /// times the scale; `arc`: lifted against gravity). Returns the scene body it is drawn
    /// as (the projectile slot).
    #[wasm_bindgen(js_name = labLaunch)]
    pub fn lab_launch(
        &mut self,
        lab: &mut LabScene,
        origin: &[f64],
        direction: &[f64],
        speed: Option<f64>,
        arc: Option<bool>,
    ) -> Result<u32, JsValue> {
        let v3 = |v: &[f64]| -> Result<[f64; 3], JsValue> {
            v.try_into().map_err(|_| err("expected 3 numbers"))
        };
        let (origin, direction) = (v3(origin)?, v3(direction)?);
        anyhow_check(
            origin.iter().chain(&direction).all(|x| x.is_finite()),
            "invalid shot ray",
        )?;
        let sh = lab.shooting;
        let v = engine_fracture::plan::shot_velocity(
            sh.center,
            sh.extent,
            sh.radius,
            origin,
            direction,
            Some(speed.unwrap_or_else(|| lab.shot_speed()) * sh.speed_scale),
            arc.unwrap_or(sh.arc),
        );
        res(self.fire(lab, origin, v, [0.0; 3], sh.mass, sh.radius))
    }

    /// Fire the pack's authored shot `index` as authored (origin, velocity, spin, mass,
    /// radius; the panel's scales do not apply).
    #[wasm_bindgen(js_name = labFirePackShot)]
    pub fn lab_fire_pack_shot(&mut self, lab: &mut LabScene, index: u32) -> Result<u32, JsValue> {
        let s = lab
            .pack_shots
            .get(index as usize)
            .cloned()
            .ok_or_else(|| err(format!("no authored shot {index}")))?;
        let p = lab.shooting.placement;
        let origin = [0, 1, 2].map(|k| s.origin[k] + p[k]);
        res(self.fire(
            lab,
            origin,
            s.velocity,
            s.angular_velocity,
            s.mass,
            s.radius,
        ))
    }

    /// Directed debris: `settings` (`GuidanceSettings`, or `null` for off) and the zones
    /// (`ZoneDesc`, anchors as scene bodies); every piece re-armed, claims cleared.
    #[wasm_bindgen(js_name = labSetGuidance)]
    pub fn lab_set_guidance(
        &mut self,
        lab: &LabScene,
        settings: JsValue,
        zones: JsValue,
    ) -> Result<(), JsValue> {
        let on = !(settings.is_null() || settings.is_undefined());
        let settings = if on {
            settings
        } else {
            js(&serde_json::json!({ "enabled": false }))?
        };
        self.set_guidance(settings)?;
        let mut v: serde_json::Value = serde_wasm_bindgen::from_value(zones).map_err(err)?;
        if let Some(list) = v.as_array_mut() {
            for z in list {
                if let Some(b) = z.get("anchorBody").and_then(|b| b.as_u64()) {
                    let slot = lab
                        .slot_of
                        .get(b as usize)
                        .copied()
                        .filter(|&s| s != u32::MAX)
                        .ok_or_else(|| err(format!("zone anchor {b} is no body")))?;
                    z["anchorBody"] = slot.into();
                }
            }
        }
        self.set_guidance_zones(js(&v)?)?;
        let mask = if on { u32::MAX } else { 0 };
        for &d in &lab.destructibles {
            res(self.world.destructible_set_guided(d, mask))?;
        }
        res(self.world.reset_guidance_claims())
    }

    /// The bond modes' reset: the scene as placed (`World::restore_state`).
    #[wasm_bindgen(js_name = labReset)]
    pub fn lab_reset(&mut self, lab: &mut LabScene) -> Result<(), JsValue> {
        let saved = lab
            .saved
            .as_ref()
            .ok_or_else(|| err("only the bond modes reset in place"))?;
        res(self.world.restore_state(saved))?;
        lab.fired = 0;
        Ok(())
    }

    /// The scene's counts: `{ chunks, bonds, bondsIntact, actors, crushed }`, summed over
    /// its destructibles.
    #[wasm_bindgen(js_name = labStats)]
    pub fn lab_stats(&self, lab: &LabScene) -> Result<JsValue, JsValue> {
        let mut t = [0u32; 5];
        for &d in &lab.destructibles {
            let s = res(self.world.destructible_stats(d))?;
            for (a, b) in t
                .iter_mut()
                .zip([s.chunks, s.bonds, s.bonds_intact, s.actors, s.crushed])
            {
                *a += b;
            }
        }
        js(
            &serde_json::json!({ "chunks": t[0], "bonds": t[1], "bondsIntact": t[2], "actors": t[3], "crushed": t[4] }),
        )
    }

    /// Per bond of the scene (its destructibles' bonds in order), 1 if intact, as of the
    /// last polled frame.
    #[wasm_bindgen(js_name = labBondsIntact)]
    pub fn lab_bonds_intact(&self, lab: &LabScene) -> Result<Vec<u8>, JsValue> {
        let mut out = Vec::new();
        for &d in &lab.destructibles {
            let n = res(self.world.destructible_stats(d))?.bonds;
            for k in 0..n {
                out.push(u8::from(res(self.world.destructible_bond_intact(d, k))?));
            }
        }
        Ok(out)
    }

    /// The threshold mode's draw: a [`RECORD_FLOATS`]-float record per scene body (its
    /// pose, and its body's flags for the sleep view) as of the last polled frame. Chunks
    /// at their bodies' transforms times their lab poses, the balls at theirs, the rest
    /// parked out of view.
    #[wasm_bindgen(js_name = labPoses)]
    pub fn lab_poses(&self, lab: &LabScene) -> Result<Vec<f32>, JsValue> {
        let n = lab.drawn.len();
        let mut out = vec![0.0f32; RECORD_FLOATS * n];
        let mut put = |i: usize, t: Transform, flags: u32| {
            let o = RECORD_FLOATS * i;
            out[o..o + 3].copy_from_slice(&t.position);
            out[o + 4..o + 8].copy_from_slice(&t.rotation);
            out[o + 11] = f32::from_bits(flags);
        };
        let parked = Transform::new(PARKED, [0.0, 0.0, 0.0, 1.0]);
        for (i, d) in lab.drawn.iter().enumerate() {
            match *d {
                Drawn::Ground => put(i, Transform::IDENTITY, 0),
                Drawn::None => put(i, parked, 0),
                Drawn::Chunk { copy, chunk, rest } => {
                    let dest = lab.destructibles[copy as usize];
                    let body = res(self.world.destructible_chunk_body(dest, chunk))?;
                    if !self.world.body_is_valid(body) {
                        // Crushed: its body is gone.
                        put(i, parked, 0);
                        continue;
                    }
                    let t = res(self.world.body_transform(body))?
                        .unwrap_or(res(self.world.body_initial_transform(body))?);
                    let flags = body_flags(&self.world, body);
                    put(i, compose(&t, &rest), flags);
                }
            }
        }
        let count = lab.projectile_count.max(1) as usize;
        let first = lab.fired as usize - lab.shots.len();
        for (k, &b) in lab.shots.iter().enumerate() {
            let slot = lab.projectile_start as usize + (first + k) % count;
            if let Some(t) = self.world.body_transform(b).ok().flatten() {
                if slot < n {
                    put(slot, t, body_flags(&self.world, b));
                }
            }
        }
        Ok(out)
    }

    /// The GPU records ([`RECORD_FLOATS`] floats, `BodyState`) of body slots `slots` as
    /// the queue leaves them now: the copy is submitted before this returns (a later step
    /// does not reach it), the Promise resolves once it is mapped. `-1`: zeros.
    #[wasm_bindgen(js_name = readBodySlots)]
    pub fn read_body_slots(&self, slots: Vec<i32>) -> js_sys::Promise {
        let device = self.world.device().clone();
        let record = std::mem::size_of::<engine_schema::BodyState>() as u64;
        let n = slots.len().max(1) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readBodySlots"),
            size: n * record,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        let source = self.world.sim().body_buffer();
        for (k, &s) in slots.iter().enumerate() {
            if s >= 0 {
                enc.copy_buffer_to_buffer(
                    source,
                    s as u64 * record,
                    &staging,
                    k as u64 * record,
                    record,
                );
            }
        }
        self.world.queue().submit([enc.finish()]);
        let (tx, rx) = futures_channel::oneshot::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        wasm_bindgen_futures::future_to_promise(async move {
            rx.await
                .map_err(|_| err("readBodySlots: map dropped"))?
                .map_err(|e| err(format!("readBodySlots: {e}")))?;
            let out = {
                let data = staging.slice(..).get_mapped_range().expect("mapped");
                js_sys::Float32Array::from(bytemuck::cast_slice::<u8, f32>(
                    &data[..(slots.len() as u64 * record) as usize],
                ))
            };
            staging.unmap();
            Ok(out.into())
        })
    }
}

/// A bond preload in progress (`labPreloadBegin`).
#[wasm_bindgen]
pub struct LabPreload {
    job: DestructiblePreload,
}

#[wasm_bindgen]
impl LabPreload {
    /// Settle steps checked so far.
    #[wasm_bindgen(getter)]
    pub fn steps(&self) -> u32 {
        self.job.steps()
    }
}

#[wasm_bindgen]
impl WebFrame {
    /// Destructible breaks in these frames: `[step, bonds broken, chunks crushed, ...]`,
    /// one triple per step that broke or crushed something.
    #[wasm_bindgen(js_name = labBreaks)]
    pub fn lab_breaks(&self) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        for (step, e) in &self.frame().destruction_events {
            let (b, c) = match e {
                DestructionEvent::BondsBroken { bonds, .. } => (bonds.len() as f64, 0.0),
                DestructionEvent::ChunkCrushed { .. } => (0.0, 1.0),
                _ => continue,
            };
            let step = *step as f64;
            match out.len().checked_sub(3) {
                Some(i) if out[i] == step => {
                    out[i + 1] += b;
                    out[i + 2] += c;
                }
                _ => out.extend([step, b, c]),
            }
        }
        out
    }
}

fn anyhow_check(ok: bool, what: &str) -> Result<(), JsValue> {
    if ok {
        Ok(())
    } else {
        Err(err(what))
    }
}

fn body_flags(world: &World, body: BodyId) -> u32 {
    let dynamic = world.body_type(body).is_ok_and(|t| t == BodyType::Dynamic);
    let awake = world.body_is_awake(body).unwrap_or(false);
    u32::from(dynamic) * FLAG_DYNAMIC | if awake { FLAG_AWAKE } else { FLAG_SLEEPING }
}

fn compose(a: &Transform, b: &Transform) -> Transform {
    let (qa, pa) = (
        glam::Quat::from_array(a.rotation),
        glam::Vec3::from(a.position),
    );
    let (qb, pb) = (
        glam::Quat::from_array(b.rotation),
        glam::Vec3::from(b.position),
    );
    Transform::new((pa + qa * pb).to_array(), (qa * qb).normalize().to_array())
}

impl WebWorld {
    /// The lab's ground: body 0's colliders (the plane y = 0) on a static body.
    fn lab_ground(&mut self, s: &Scene, material: SurfaceMaterial) -> anyhow::Result<BodyId> {
        let g = &s.bodies[0];
        let ground = self.world.create_body(&BodyDef {
            position: f3(g.p),
            rotation: f4(g.q),
            ..BodyDef::default()
        })?;
        let def = ShapeDef {
            material,
            ..ShapeDef::default()
        };
        for c in s.colliders.iter().filter(|c| c.body == 0) {
            anyhow::ensure!(
                c.shape == SHAPE_PLANE,
                "the lab's ground is planes (collider shape {})",
                c.shape
            );
            let p = c.params;
            self.world.create_shape(
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

    fn place_bonds(&mut self, asset: DestructibleAssetId, s: &Scene) -> anyhow::Result<LabScene> {
        let material = lab_material(s);
        anyhow::ensure!(
            self.world.body_count() == 0,
            "the lab places its scene in an empty world"
        );
        let ground = self.lab_ground(s, material)?;
        anyhow::ensure!(
            ground.index() == 0,
            "the ground took slot {}",
            ground.index()
        );
        let shots_range = s.projectile_start..s.projectile_start + s.projectile_count;
        // The scene's pieces start asleep (the lab's scenes do), unless every one is awake.
        let asleep = (1..s.bodies.len())
            .filter(|i| !shots_range.contains(&(*i as u32)) && s.bodies[*i].inv_mass > 0.0)
            .any(|i| s.bodies[i].flags & FLAG_SLEEPING != 0);
        let d = self.world.create_destructible(&DestructibleDef {
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
            slot_of[b as usize] = self.world.destructible_chunk_body(d, k as u32)?.index();
        }
        // The instanced streams (every piece of every copy, the bond lines) read slots
        // directly: the pieces before the shots must be scene body = slot, bond = joint.
        for b in 1..s.projectile_start.min(s.bodies.len() as u32) {
            anyhow::ensure!(
                slot_of[b as usize] == b,
                "scene body {b} took world slot {}",
                slot_of[b as usize]
            );
        }
        let joints = s.joints.len() as u32;
        for k in 0..joints {
            let j = self.world.destructible_bond_joint(d, k)?;
            anyhow::ensure!(
                j.map(|j| j.index()) == Some(k),
                "bond {k} is joint {:?}",
                j.map(|j| j.index())
            );
        }
        // The shots, parked static where the scene parks them until fired.
        let r = s.projectile_radius as f32;
        let mut shots = Vec::with_capacity(s.projectile_count as usize);
        for i in shots_range.clone() {
            let b = &s.bodies[i as usize];
            let body = self.world.create_body(&BodyDef {
                position: f3(b.p),
                rotation: f4(b.q),
                ..BodyDef::default()
            })?;
            self.world.create_sphere_shape(
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
        self.world.reserve_moving_radius(r);
        let saved = Some(self.world.save_state()?);
        Ok(LabScene {
            model: StrengthModel::Bonds,
            destructibles: vec![d],
            slot_of,
            shots,
            fired: 0,
            shooting: shooting(s),
            projectile_start: s.projectile_start,
            projectile_count: s.projectile_count,
            material,
            drawn: Vec::new(),
            pack_shots: Vec::new(),
            saved,
        })
    }

    fn place_threshold(
        &mut self,
        asset: DestructibleAssetId,
        a: &LabAsset,
        s: &Scene,
        o: &Opts,
    ) -> anyhow::Result<LabScene> {
        let material = lab_material(s);
        let stress = a
            .desc
            .stress
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no stress import"))?;
        // The asset frame is the lab's frame of one copy (`from_glb`, `from_pack` and
        // `cuboid_house` build their chunks from the lab scene of it): each copy's instance
        // is its offset, from its first chunk in the drawn scene.
        let stride = s.instance_stride.max(1) as usize;
        let copies = s.copies.max(1).min(o.copies.max(1)) as usize;
        let first = a.desc.chunks[0].pose.position;
        let offsets: Vec<[f32; 3]> = (0..copies)
            .map(|k| {
                let p = f3(s.bodies[1 + k * stride].p);
                [p[0] - first[0], p[1] - first[1], p[2] - first[2]]
            })
            .collect();
        if let Some(h) = stress.ground_height() {
            let g = self.world.create_body(&BodyDef::default())?;
            self.world.create_shape(
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
        let mut destructibles = Vec::with_capacity(copies);
        for off in &offsets {
            destructibles.push(self.world.create_destructible(&DestructibleDef {
                model: StrengthModel::Threshold,
                transform: Transform::new(*off, [0.0, 0.0, 0.0, 1.0]),
                strength_scale: o.strength_scale,
                asleep: false,
                crush: crush.clone(),
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
        let lab_chunks = a.desc.chunks.len();
        check_lab_chunk_order(&a.desc)?;
        let mut drawn = vec![Drawn::None; s.bodies.len()];
        drawn[0] = Drawn::Ground;
        for k in 0..copies {
            for c in 0..lab_chunks {
                let i = 1 + c + k * stride;
                // The chunk's pose in the asset frame: its body's frame, as drawn.
                if i < drawn.len() {
                    drawn[i] = Drawn::Chunk {
                        copy: k as u32,
                        chunk: c as u32,
                        rest: a.desc.chunks[c].pose,
                    };
                }
            }
        }
        let mut sh = shooting(s);
        sh.placement = stress.placement();
        Ok(LabScene {
            model: StrengthModel::Threshold,
            destructibles,
            slot_of: Vec::new(),
            shots: Vec::new(),
            fired: 0,
            shooting: sh,
            projectile_start: s.projectile_start,
            projectile_count: s.projectile_count.max(1),
            material,
            drawn,
            pack_shots: a.pack_shots.clone(),
            saved: None,
        })
    }

    /// A shot: bonds, the next parked projectile launched; threshold, a new ball (the
    /// oldest of [`STRESS_SHOTS`] destroyed), a projectile for the fracture.
    fn fire(
        &mut self,
        lab: &mut LabScene,
        origin: [f64; 3],
        v: [f64; 3],
        w: [f64; 3],
        mass: f64,
        radius: f64,
    ) -> anyhow::Result<u32> {
        let (m, r) = (mass as f32, radius as f32);
        let i = 0.4 * m * r * r;
        let mass_data = MassData {
            mass: m,
            center: [0.0; 3],
            rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
        };
        let at = Transform::new(f3(origin), [0.0, 0.0, 0.0, 1.0]);
        let count = lab.projectile_count.max(1);
        let slot = lab.projectile_start + lab.fired % count;
        match lab.model {
            StrengthModel::Bonds => {
                anyhow::ensure!(!lab.shots.is_empty(), "The scene has no projectile slots");
                let body = lab.shots[(lab.fired % count) as usize];
                // Continuous collision is set on a static body (a reused slot is parked
                // again first, as the plan rewrote its record), then it is launched.
                if self.world.body_type(body)? != BodyType::Static {
                    let parked = self.world.body_initial_transform(body)?;
                    self.world
                        .body_set_type(body, BodyType::Static, parked, [0.0; 3], [0.0; 3])?;
                }
                self.world.body_set_bullet(body, lab.shooting.ccd)?;
                self.world
                    .body_set_type(body, BodyType::Dynamic, at, f3(v), f3(w))?;
                self.world.body_set_mass_data(body, mass_data)?;
            }
            StrengthModel::Threshold => {
                if lab.shots.len() >= count.min(STRESS_SHOTS) as usize {
                    let old = lab.shots.remove(0);
                    self.world.destroy_body(old)?;
                }
                // No continuous collision, as the lab's ThresholdWorld (its profile's one
                // substep, no TOI CCD): the strike prediction meets the shot.
                let ball = self.world.create_body(&BodyDef {
                    linear_velocity: f3(v),
                    angular_velocity: f3(w),
                    ..BodyDef::dynamic(at.position)
                })?;
                self.world.create_sphere_shape(
                    ball,
                    &ShapeDef {
                        material: lab.material,
                        update_body_mass: false,
                        ..ShapeDef::default()
                    },
                    [0.0; 3],
                    r,
                )?;
                self.world.body_set_mass_data(ball, mass_data)?;
                self.world
                    .body_set_projectile(ball, Some(Projectile::default()))?;
                lab.shots.push(ball);
            }
        }
        lab.fired += 1;
        Ok(slot)
    }
}

fn shooting(s: &Scene) -> Shooting {
    Shooting {
        center: s.center,
        extent: s.extent,
        radius: s.projectile_radius,
        mass: s.projectile_mass,
        speed: s.projectile_speed,
        arc: s.projectile_arc,
        speed_scale: 1.0,
        ccd: true,
        placement: [0.0; 3],
    }
}

/// Chunk crushing as the stress mode's Crush menu sets it: off, the scene's own
/// parameters, or those and the cited presets for the rest. Crushed chunks are removed
/// (the lab's dust).
fn crush_for(mode: u32, stress: &StressSource) -> anyhow::Result<Option<Crush>> {
    if mode == 0 {
        return Ok(None);
    }
    let mut materials = Vec::new();
    if mode >= 2 {
        for (name, m) in &stress.scene()?.materials {
            if m.crush.is_some() {
                continue;
            }
            let base = name.split(['#', '@']).next().unwrap_or(name);
            if let Some(p) = stress_ref::crush::presets::for_material(base) {
                if !materials
                    .iter()
                    .any(|(n, _): &(String, CrushMaterial)| n == base)
                {
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
            let zup = [
                body.position[0] + c[0],
                body.position[1] + c[1],
                body.position[2] + c[2],
            ];
            // Z up to Y up (`to_y_up`): (x, y, z) -> (x, z, -y).
            let at = [p[0] + zup[0], p[1] + zup[2], p[2] - zup[1]];
            let dist = |i: usize| {
                let b = desc.chunks[i - 1].pose.position;
                (0..3)
                    .map(|j| (at[j] - b[j] as f64).powi(2))
                    .sum::<f64>()
                    .sqrt()
            };
            let d = dist(1 + k);
            // The import's centre is its box's, the lab's the mesh's centre of mass (a thin
            // pane's differ by centimetres): within the chunk's bounds, and nearer its own
            // body than the bodies numbered next to it (an offset numbering fails that).
            let bound = ch.half_extents.iter().map(|h| h * h).sum::<f64>().sqrt();
            let neighbours = [k, k + 2]
                .into_iter()
                .filter(|&i| i >= 1 && i <= n)
                .map(dist);
            anyhow::ensure!(
                d <= bound && neighbours.into_iter().all(|o| d <= o),
                "the stress import's chunk {k} is {d:.3} m from the lab scene's chunk {k} (chunk order differs)"
            );
            k += 1;
        }
    }
    anyhow::ensure!(
        k >= n,
        "the stress import has {k} chunks for the asset's {n}"
    );
    Ok(())
}
