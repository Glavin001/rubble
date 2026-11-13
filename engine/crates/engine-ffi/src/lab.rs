//! The fracture lab (`rubble::lab`) through the C ABI: a lab scene placed under a bond
//! model with explicit settings, actions queued at steps, the steps, and the stats every
//! host shows. The same Rust the browser lab and native demos run. The lab works in
//! Rubble's frame (Y up, metres), whatever a world's coordinates: its scenes are authored
//! there.

use super::*;
use rubble::lab::{Lab, LabAction, LabArrangement, LabBonds, LabCrush, LabDef, LabPlan, LabSettings, LabSource};

/// What a lab scene is built from.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RbLabSource {
    /// A fracture GLB (`bytes`) at hierarchy level `level`.
    RB_LAB_GLB = 0,
    /// The procedural cuboid house (no bytes).
    RB_LAB_CUBOID = 1,
    /// A scene pack's JSON (`bytes`).
    RB_LAB_PACK = 2,
    /// The glass tower and its pavilion (generated; no bytes).
    RB_LAB_GLASS_TOWER = 3,
}

/// How copies of the source are arranged.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RbLabArrangement {
    RB_LAB_ONE = 0,
    /// `layout_columns` × `layout_rows` copies (a row or a grid), or a suburb of them.
    RB_LAB_LAYOUT = 1,
    /// The 2×2 tower block of a pack with its city (cars, pedestrians).
    RB_LAB_TOWER_BLOCK = 2,
    /// The tower block with the meteor's ball.
    RB_LAB_METEOR_BLOCK = 3,
}

/// How the lab's bonds hold.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RbLabBonds {
    /// Loose pieces.
    RB_LAB_BONDS_NONE = 0,
    /// Breakable D6 joints (settled under self-weight before placing).
    RB_LAB_BONDS_D6 = 1,
    /// A stress model on the GPU fracture.
    RB_LAB_BONDS_THRESHOLD = 2,
    RB_LAB_BONDS_LOAD_PATH = 3,
    RB_LAB_BONDS_IMPACT_LOCAL = 4,
    RB_LAB_BONDS_BLAST_GPU = 5,
}

/// Chunk crushing (stress models).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RbLabCrush {
    RB_LAB_CRUSH_OFF = 0,
    /// The scene's own crush parameters.
    RB_LAB_CRUSH_SCENE = 1,
    /// Those and the cited presets for the other materials.
    RB_LAB_CRUSH_PRESETS = 2,
}

/// A lab run (`rbDefaultLabDef`).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RbLabDef {
    pub source: RbLabSource,
    /// The GLB or pack bytes (copied); null with length 0 for the generated sources.
    pub bytes: *const u8,
    pub length: usize,
    pub level: u8,
    pub arrangement: RbLabArrangement,
    pub layout_columns: u32,
    pub layout_rows: u32,
    pub layout_suburb: bool,
    pub bonds: RbLabBonds,
    /// Physics steps per simulated second (default 60).
    pub physics_hz: f32,
    pub strength_scale: f32,
    pub ball_radius_scale: f64,
    pub ball_mass_scale: f64,
    pub ball_speed_scale: f64,
    /// Continuous collision for the balls.
    pub ccd: bool,
    /// Stress models: the in-step contact fracture and strike prediction.
    pub contact_fracture: RbFractureSwitch,
    pub strike_prediction: RbFractureSwitch,
    pub crush: RbLabCrush,
    pub internal_value: u32,
}

/// A lab action (`rbLab_Queue`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RbLabAction {
    /// A ball from `origin` along `direction`.
    RB_LAB_SHOT = 0,
    RB_LAB_ROW_SHOT = 1,
    RB_LAB_VOLLEY = 2,
    RB_LAB_TOPPLE = 3,
    RB_LAB_METEOR = 4,
    /// The pack's authored shot `index`.
    RB_LAB_PACK_SHOT = 5,
}

/// A lab's counts (`rbLab_GetStats`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RbLabStats {
    pub steps: u64,
    pub shots: u32,
    pub chunks: u32,
    pub bonds: u32,
    pub bonds_intact: u32,
    pub bodies: u32,
    pub crushed: u32,
    pub awake: u32,
    pub contacts: u32,
    /// GPU time of a physics step (ms), what it allows, and the share of a second real
    /// time takes; negative where the device does not time steps.
    pub gpu_step_ms: f64,
    pub max_steps_per_second: f64,
    pub realtime_share: f64,
    pub physics_hz: f32,
}

/// An opaque lab: its world and its placed scene.
pub struct RbLab {
    world: World,
    lab: Lab,
}

#[no_mangle]
pub extern "C" fn rbDefaultLabDef() -> RbLabDef {
    let s = LabSettings::default();
    RbLabDef {
        source: RbLabSource::RB_LAB_CUBOID,
        bytes: std::ptr::null(),
        length: 0,
        level: 0,
        arrangement: RbLabArrangement::RB_LAB_ONE,
        layout_columns: 1,
        layout_rows: 1,
        layout_suburb: false,
        bonds: RbLabBonds::RB_LAB_BONDS_D6,
        physics_hz: s.physics_hz,
        strength_scale: s.strength_scale,
        ball_radius_scale: s.ball_radius_scale,
        ball_mass_scale: s.ball_mass_scale,
        ball_speed_scale: s.ball_speed_scale,
        ccd: s.ccd,
        contact_fracture: RbFractureSwitch::RB_FRACTURE_MODEL_DEFAULT,
        strike_prediction: RbFractureSwitch::RB_FRACTURE_MODEL_DEFAULT,
        crush: RbLabCrush::RB_LAB_CRUSH_OFF,
        internal_value: COOKIE,
    }
}

unsafe fn lab_def(d: &RbLabDef) -> anyhow::Result<LabDef> {
    check_cookie(d.internal_value, "lab def not from rbDefaultLabDef")?;
    let bytes = || -> anyhow::Result<std::sync::Arc<Vec<u8>>> {
        if d.bytes.is_null() || d.length == 0 {
            return Err(invalid("lab source bytes"));
        }
        Ok(std::sync::Arc::new(std::slice::from_raw_parts(d.bytes, d.length).to_vec()))
    };
    let source = match d.source {
        RbLabSource::RB_LAB_GLB => LabSource::Glb {
            bytes: bytes()?,
            level: d.level,
            max_hull_vertices: None,
        },
        RbLabSource::RB_LAB_CUBOID => LabSource::Cuboid,
        RbLabSource::RB_LAB_PACK => LabSource::Pack { bytes: bytes()? },
        RbLabSource::RB_LAB_GLASS_TOWER => LabSource::glass_tower()?,
    };
    let arrangement = match d.arrangement {
        RbLabArrangement::RB_LAB_ONE => LabArrangement::One,
        RbLabArrangement::RB_LAB_LAYOUT => LabArrangement::Layout(rubble::lab::LabLayout {
            suburb: d.layout_suburb,
            columns: d.layout_columns.max(1),
            rows: d.layout_rows.max(1),
        }),
        RbLabArrangement::RB_LAB_TOWER_BLOCK => LabArrangement::TowerBlock { meteor: false },
        RbLabArrangement::RB_LAB_METEOR_BLOCK => LabArrangement::TowerBlock { meteor: true },
    };
    let bonds = match d.bonds {
        RbLabBonds::RB_LAB_BONDS_NONE => LabBonds::None,
        RbLabBonds::RB_LAB_BONDS_D6 => LabBonds::Model(StrengthModel::Bonds),
        RbLabBonds::RB_LAB_BONDS_THRESHOLD => LabBonds::Model(StrengthModel::Threshold),
        RbLabBonds::RB_LAB_BONDS_LOAD_PATH => LabBonds::Model(StrengthModel::LoadPath),
        RbLabBonds::RB_LAB_BONDS_IMPACT_LOCAL => LabBonds::Model(StrengthModel::ImpactLocal),
        RbLabBonds::RB_LAB_BONDS_BLAST_GPU => LabBonds::Model(StrengthModel::BlastGpu),
    };
    Ok(LabDef {
        source,
        arrangement,
        settings: LabSettings {
            bonds,
            physics_hz: d.physics_hz,
            strength_scale: d.strength_scale,
            ball_radius_scale: d.ball_radius_scale,
            ball_mass_scale: d.ball_mass_scale,
            ball_speed_scale: d.ball_speed_scale,
            ccd: d.ccd,
            fracture: FractureOptions {
                contact_fracture: switch_in(d.contact_fracture),
                strike_prediction: switch_in(d.strike_prediction),
            },
            crush: match d.crush {
                RbLabCrush::RB_LAB_CRUSH_OFF => LabCrush::Off,
                RbLabCrush::RB_LAB_CRUSH_SCENE => LabCrush::Scene,
                RbLabCrush::RB_LAB_CRUSH_PRESETS => LabCrush::Presets,
            },
            // Not in the C ABI yet (`RbLabDef` has no fields for them): off, as before.
            support_loss: false,
            blast_loading: false,
            gpu_timing: GpuTiming::Step,
        },
    })
}

unsafe fn lab_mut<'a>(lab: *mut RbLab) -> anyhow::Result<&'a mut RbLab> {
    lab.as_mut().ok_or_else(|| invalid("null lab"))
}

/// Build and place a lab run on the default GPU: its scene, its world (D6 bonds settle
/// first, blocking). A combination that cannot work fails with its reason
/// (`rbGetLastError`).
#[no_mangle]
pub unsafe extern "C" fn rbCreateLab(def: *const RbLabDef, out: *mut *mut RbLab) -> RbResult {
    guard_args(|| {
        let def = lab_def(read(def, "null lab def")?)?;
        let mut plan = LabPlan::new(&def)?;
        let mut world = World::new(GpuContext::Default, &plan.world_def()?)?;
        if plan.needs_settle() {
            plan.settle(world.gpu_context())?;
            // The settled scene may need a different capacity: a world for it.
            world = World::new(world.gpu_context(), &plan.world_def()?)?;
        }
        let lab = Lab::place(&mut world, &plan)?;
        write(out, Box::into_raw(Box::new(RbLab { world, lab })))
    })
}

#[no_mangle]
pub unsafe extern "C" fn rbDestroyLab(lab: *mut RbLab) {
    if !lab.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(lab))));
    }
}

/// Queue `action` to start `delay` steps from now (`origin` and `direction`: a shot's ray,
/// Rubble's frame; `index`: a pack shot's). An action the scene has none of is an error.
#[no_mangle]
pub unsafe extern "C" fn rbLab_Queue(
    lab: *mut RbLab,
    action: RbLabAction,
    origin: RbVec3,
    direction: RbVec3,
    index: u32,
    delay: u64,
) -> RbResult {
    guard_args(|| {
        let l = lab_mut(lab)?;
        let d3 = |v: RbVec3| [v.x as f64, v.y as f64, v.z as f64];
        let a = match action {
            RbLabAction::RB_LAB_SHOT => LabAction::Shot {
                origin: d3(origin),
                direction: d3(direction),
            },
            RbLabAction::RB_LAB_ROW_SHOT => LabAction::RowShot,
            RbLabAction::RB_LAB_VOLLEY => LabAction::Volley,
            RbLabAction::RB_LAB_TOPPLE => LabAction::Topple,
            RbLabAction::RB_LAB_METEOR => LabAction::Meteor,
            RbLabAction::RB_LAB_PACK_SHOT => LabAction::PackShot(index),
        };
        l.lab.queue(&l.world, &a, delay)
    })
}

/// `steps` steps (each launching the shots due), waiting for each step's results.
#[no_mangle]
pub unsafe extern "C" fn rbLab_Step(lab: *mut RbLab, steps: u32) -> RbResult {
    guard_args(|| {
        let l = lab_mut(lab)?;
        for _ in 0..steps {
            l.lab.step(&mut l.world)?;
            l.world.poll_blocking()?;
        }
        Ok(())
    })
}

/// The lab's counts as of its last step.
#[no_mangle]
pub unsafe extern "C" fn rbLab_GetStats(lab: *mut RbLab, out: *mut RbLabStats) -> RbResult {
    guard_args(|| {
        let l = lab_mut(lab)?;
        let s = l.lab.stats(&l.world)?;
        write(
            out,
            RbLabStats {
                steps: s.steps,
                shots: s.shots,
                chunks: s.chunks,
                bonds: s.bonds,
                bonds_intact: s.bonds_intact,
                bodies: s.bodies,
                crushed: s.crushed,
                awake: s.awake,
                contacts: s.contacts,
                gpu_step_ms: s.gpu_step_ms.unwrap_or(-1.0),
                max_steps_per_second: s.max_steps_per_second.unwrap_or(-1.0),
                realtime_share: s.realtime_share.unwrap_or(-1.0),
                physics_hz: s.physics_hz,
            },
        )
    })
}
