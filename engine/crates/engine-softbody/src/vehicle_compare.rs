//! The vehicle comparison harness (`examples/vehicle_compare.rs`): every car
//! of a manifest through the same scenes on the explicit solver's CPU
//! reference, deterministically, with the same measurements, so an authored
//! car, a generated one and an imported community car are judged alike
//! (`engine/docs/measurements/vehicle-compare.md`).
//!
//! Nothing here reads BeamNG code; a community car's JBeam files are data the
//! caller points at (gitignored, never committed).
//!
//! **Cars.** A manifest entry is either an authored style (`style`: built at
//! Medium, its `.truck` on the solver) or a JBeam vehicle (`mod`, `vehicle`,
//! `pc`: the folder under the crash lab's `__mods/`, the vehicle's name in it
//! and its configuration). Every node set a scene measures is picked from the
//! car's geometry (its skin box, axles and ground), never from node names, so
//! a new car needs no code: one manifest line.
//!
//! **Paths.** The rest scene runs on raycast wheels, as the page drives a car
//! (for a JBeam car twice: JBeam mode, and the page's default `.truck`
//! export). The crash scenes run each car on the spawn its reference scenes
//! use (`showcase_car.rs`): an authored car's `.truck` with its soft tyres, a
//! JBeam car in JBeam mode on its pressure wheels; so the authored car's
//! numbers here cross-check the documented ones.
//!
//! **Frame.** Every car is turned to face +Z (x to its left, the driver's
//! side; y up), its skin box centred on x = z = 0, its lowest node 2 cm above
//! the ground, settled 1 s at rest; each scene starts from a copy.

use crate::authoring::skin::{deform_beams, unit_states, BeamState, ShowcaseVisual};
use crate::authoring::{build, style, BuildParams, Built, Resolution};
use crate::desc::SoftVehicleDesc;
use crate::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, ExplicitKind, JbeamWheels, StaticBox, StaticShape,
    NODE_IMMOVABLE, NODE_NO_GROUND, NODE_TYRE, STEP,
};
use crate::explicit_contact::ExplicitBeamsWorld;
use glam::{DMat3, DQuat, DVec3, Quat, Vec3};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// One manifest entry (`vehicle-compare-cars.json`).
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct CarSpec {
    pub name: String,
    #[serde(default)]
    pub label: String,
    /// An authored style (`sedan1`).
    #[serde(default)]
    pub style: Option<String>,
    /// A JBeam vehicle: its folder under `__mods/`, its name there and its
    /// `.pc` (relative to the vehicle's folder).
    #[serde(default, rename = "mod")]
    pub mod_dir: Option<String>,
    #[serde(default)]
    pub vehicle: Option<String>,
    #[serde(default)]
    pub pc: Option<String>,
    /// Skipped (reported) when its files are absent.
    #[serde(default)]
    pub optional: bool,
    /// The front door's part (JBeam part name) when the geometric pick
    /// (the left part named `door` with the frontmost centre) is wrong.
    #[serde(default)]
    pub door_part: Option<String>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Manifest {
    pub cars: Vec<CarSpec>,
}

/// The solver settings every car runs with (the reference scenes': drag on,
/// no turbulence).
/// Surface contact (jbeam-semantics-spec.md §38) for every car of a run:
/// `SURFACE_CONTACT=1` on, else off (the harness's default, unchanged).
pub fn surface_contact() -> bool {
    std::env::var("SURFACE_CONTACT").is_ok_and(|v| v == "1")
}

pub fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        turbulence: false,
        ..Default::default()
    }
}

/// What a car was built from.
pub enum Source {
    Authored(Box<Built>),
    Jbeam {
        desc: Box<SoftVehicleDesc>,
        files: Vec<(String, String)>,
        pc: Option<String>,
        dir: PathBuf,
        vehicle: String,
    },
}

/// The car's frame at rest (template coordinates).
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    /// Skin box.
    pub lo: Vec3,
    pub hi: Vec3,
    /// Axles along z.
    pub front_axle: f32,
    pub rear_axle: f32,
    /// The lowest non-tyre node's height and the ground (0).
    pub floor: f32,
}

impl Frame {
    pub fn width(&self) -> f32 {
        self.hi.x - self.lo.x
    }
    pub fn length(&self) -> f32 {
        self.hi.z - self.lo.z
    }
    pub fn wheelbase(&self) -> f32 {
        self.front_axle - self.rear_axle
    }
}

/// A car ready for the scenes.
pub struct Car {
    pub spec: CarSpec,
    pub source: Source,
    /// Settled at rest on its crash spawn (soft tyres / pressure wheels).
    pub template: ExplicitBeams,
    pub frame: Frame,
    /// Parts: (name, nodes), tyre nodes excluded.
    pub parts: Vec<(String, Vec<u32>)>,
    /// Hardware (hinges, latches; JBeam break groups): (name, node pairs).
    pub hardware: Vec<(String, Vec<(u32, u32)>)>,
    /// The collision skin's nodes.
    pub skin: Vec<u32>,
    pub visual: Option<ShowcaseVisual>,
    /// The front-left door's nodes (the door crush), when it has one.
    pub door: Option<(String, Vec<u32>)>,
    pub warnings: Vec<String>,
    /// Per part, the largest move of its nodes about the body's rigid fit over the 1 s settle (m), largest first.
    pub settle: Vec<(String, f32)>,
}

fn is_tyre(s: &ExplicitBeams, i: usize) -> bool {
    s.nodes[i].flags & NODE_TYRE != 0
}

/// Fold the origin into the node positions (origin zero).
fn flatten_origin(s: &mut ExplicitBeams) {
    let o = s.origin;
    for n in &mut s.nodes {
        n.pos += o;
    }
    s.origin = Vec3::ZERO;
}

fn skin_nodes(s: &ExplicitBeams) -> Vec<u32> {
    let mut v: Vec<u32> = s.triangles.iter().flat_map(|t| t.nodes).collect();
    v.sort_unstable();
    v.dedup();
    v.retain(|&i| !is_tyre(s, i as usize));
    v
}

/// Turn about +Y by `angle` and place: skin box centred on x = z = 0, the
/// lowest node `lift` above the ground.
fn place(s: &mut ExplicitBeams, angle: f32, lift: f32) {
    flatten_origin(s);
    let q = Quat::from_rotation_y(angle);
    for n in &mut s.nodes {
        n.pos = q * n.pos;
        n.vel = q * n.vel;
    }
    let skin = skin_nodes(s);
    let (lo, hi) = bounds(s, &skin);
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let mid = 0.5 * (lo + hi);
    for n in &mut s.nodes {
        n.pos.x -= mid.x;
        n.pos.z -= mid.z;
        n.pos.y -= low - lift;
    }
}

fn bounds(s: &ExplicitBeams, ids: &[u32]) -> (Vec3, Vec3) {
    let it: Box<dyn Iterator<Item = Vec3>> = if ids.is_empty() {
        Box::new(s.nodes.iter().map(|n| n.pos))
    } else {
        Box::new(ids.iter().map(|&i| s.nodes[i as usize].pos))
    };
    it.fold((Vec3::MAX, Vec3::MIN), |(a, b), p| (a.min(p), b.max(p)))
}

/// Every node at `v` (immovable ones still), tyres rolling.
fn set_moving(s: &mut ExplicitBeams, v: Vec3) {
    for n in &mut s.nodes {
        n.vel = if n.flags & NODE_IMMOVABLE != 0 {
            Vec3::ZERO
        } else {
            v
        };
    }
    s.roll_tyres(v);
    if let Some(d) = s.drive.as_mut() {
        let [c, b, _] = d.config.refs;
        let f = (s.nodes[c as usize].pos - s.nodes[b as usize].pos).normalize_or_zero();
        d.roll(v.dot(f));
    }
}

fn steps(s: &mut ExplicitBeams, seconds: f32) {
    for _ in 0..(seconds / STEP).round() as usize {
        s.step();
    }
}

fn axles(s: &ExplicitBeams, lo: Vec3, hi: Vec3) -> (f32, f32) {
    let mid = 0.5 * (lo.z + hi.z);
    let (mut f, mut r) = ((0.0f32, 0u32), (0.0f32, 0u32));
    let mut hubs: Vec<Vec3> = s
        .nodes
        .iter()
        .filter(|n| n.flags & NODE_TYRE != 0)
        .map(|n| n.pos)
        .collect();
    if hubs.is_empty() {
        if let Some(d) = s.drive.as_ref() {
            hubs = d
                .config
                .wheels
                .iter()
                .map(|w| 0.5 * (s.nodes[w.node1 as usize].pos + s.nodes[w.node2 as usize].pos))
                .collect();
        }
    }
    for p in hubs {
        let z = p.z - mid;
        if z > 0.0 {
            f = (f.0 + z, f.1 + 1)
        } else {
            r = (r.0 + z, r.1 + 1)
        }
    }
    let len = hi.z - lo.z;
    if f.1 == 0 || r.1 == 0 {
        return (mid + 0.3 * len, mid - 0.3 * len);
    }
    (mid + f.0 / f.1 as f32, mid + r.0 / r.1 as f32)
}

fn frame_of(s: &ExplicitBeams, skin: &[u32]) -> Frame {
    let (lo, hi) = bounds(s, skin);
    let (front_axle, rear_axle) = axles(s, lo, hi);
    let floor = (0..s.nodes.len())
        .filter(|&i| !is_tyre(s, i))
        .map(|i| s.nodes[i].pos.y)
        .fold(f32::MAX, f32::min);
    Frame {
        lo,
        hi,
        front_axle,
        rear_axle,
        floor,
    }
}

/// The manifest's default location: next to the crash lab.
pub fn default_manifest() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine-webgpu/vehicle-compare-cars.json")
}

/// The crash lab's `__mods/` folder (gitignored; `RUBBLE_COMPARE_MODS` overrides).
pub fn mods_dir() -> PathBuf {
    std::env::var("RUBBLE_COMPARE_MODS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine-webgpu/__mods"))
}

/// Whether a manifest entry's files are here.
pub fn available(spec: &CarSpec) -> bool {
    match (&spec.style, &spec.mod_dir) {
        (Some(_), _) => true,
        (None, Some(m)) => mods_dir()
            .join(m)
            .join(spec.vehicle.as_deref().unwrap_or(m))
            .is_dir(),
        _ => false,
    }
}

fn walk(d: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(d) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else {
                out.push(p);
            }
        }
    }
}

/// Spawn the car on its crash path, unplaced (desc order; tyres last).
fn crash_spawn(source: &Source) -> Result<(ExplicitBeams, f32, Vec<String>), String> {
    match source {
        Source::Authored(b) => {
            let probe = ExplicitBeams::from_truck(&b.truck, Vec3::ZERO, Vec3::ZERO, settings())?;
            let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
            let s = ExplicitBeams::from_truck(
                &b.truck,
                Vec3::new(0.0, probe.spawn_clearance() - low, 0.0),
                Vec3::ZERO,
                settings(),
            )?;
            Ok((s, 0.0, Vec::new()))
        }
        Source::Jbeam { desc, .. } => {
            let (s, w) = ExplicitBeams::from_desc_jbeam_with(
                desc,
                JbeamWheels::Pressure,
                Vec3::ZERO,
                Vec3::ZERO,
                settings(),
            )?;
            let f = crate::pack::forward_axis(desc);
            Ok((s, -f.x.atan2(f.z), w))
        }
    }
}

/// The rest scene's spawn on raycast wheels: `truck_export` for a JBeam car
/// through the page's default `.truck` export, else JBeam mode.
fn raycast_spawn(source: &Source, truck_export: bool) -> Result<ExplicitBeams, String> {
    let mut s = raycast_spawn_inner(source, truck_export)?;
    s.surface_contact = surface_contact();
    Ok(s)
}

fn raycast_spawn_inner(source: &Source, truck_export: bool) -> Result<ExplicitBeams, String> {
    let truck = |text: &str| -> Result<ExplicitBeams, String> {
        let probe = ExplicitBeams::from_truck(text, Vec3::ZERO, Vec3::ZERO, settings())?;
        let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
        let (mut s, _) = ExplicitBeams::from_truck_raycast(
            text,
            None,
            Vec3::new(0.0, probe.spawn_clearance() - low, 0.0),
            0.0,
            Vec3::ZERO,
            settings(),
        )?;
        flatten_origin(&mut s);
        Ok(s)
    };
    match source {
        Source::Authored(b) => truck(&b.truck),
        Source::Jbeam { desc, .. } if truck_export => {
            let (text, _) = crate::truck_export::desc_to_truck(
                desc,
                crate::truck_export::TyreDefaults::for_vehicle(desc),
            );
            let mut s = truck(&text)?;
            let f = crate::pack::forward_axis(desc);
            place(&mut s, -f.x.atan2(f.z), 0.02);
            Ok(s)
        }
        Source::Jbeam { desc, .. } => {
            let (mut s, _) =
                ExplicitBeams::from_desc_jbeam_raycast(desc, Vec3::ZERO, Vec3::ZERO, settings())?;
            let f = crate::pack::forward_axis(desc);
            place(&mut s, -f.x.atan2(f.z), 0.02);
            Ok(s)
        }
    }
}

impl Car {
    /// Load, spawn, settle and pick the node sets.
    pub fn load(spec: &CarSpec) -> Result<Car, String> {
        let source = if let Some(st) = &spec.style {
            let b = build(
                &style(st).ok_or_else(|| format!("no style {st}"))?,
                &BuildParams {
                    resolution: Resolution::Medium,
                    ..Default::default()
                },
            )?;
            Source::Authored(Box::new(b))
        } else {
            let m = spec
                .mod_dir
                .as_deref()
                .ok_or("a car needs `style` or `mod`")?;
            let dir = mods_dir().join(m);
            let vehicle = spec.vehicle.clone().unwrap_or_else(|| m.to_string());
            let files = crate::jbeam::read_vehicle_dir(&dir, &vehicle)?;
            let pc = match &spec.pc {
                Some(p) => Some(
                    std::fs::read_to_string(dir.join(&vehicle).join(p))
                        .map_err(|e| format!("{p}: {e}"))?,
                ),
                None => None,
            };
            let refs: Vec<(&str, &str)> = files
                .iter()
                .map(|(a, b)| (a.as_str(), b.as_str()))
                .collect();
            let desc = crate::jbeam::load_jbeam(&refs, pc.as_deref())?;
            Source::Jbeam {
                desc: Box::new(desc),
                files,
                pc,
                dir,
                vehicle,
            }
        };
        let (spawned, angle, warnings) = crash_spawn(&source)?;
        // The visual binds to the spawn's node indices (rest positions).
        let visual = match &source {
            Source::Authored(b) => crate::authoring::skin::showcase_visual(b).ok(),
            Source::Jbeam {
                files,
                pc,
                dir,
                vehicle,
                desc,
            } => {
                let refs: Vec<(&str, &str)> = files
                    .iter()
                    .map(|(a, b)| (a.as_str(), b.as_str()))
                    .collect();
                let flex = crate::jbeam::assembled_rows(&refs, pc.as_deref(), "flexbodies")?;
                let props = crate::jbeam::assembled_rows(&refs, pc.as_deref(), "props")?;
                let mut paths = Vec::new();
                walk(dir, &mut paths);
                let _ = vehicle;
                let mut meshes = Vec::new();
                let mut mats = HashMap::new();
                for p in &paths {
                    let name = p.to_string_lossy();
                    if p.extension().is_some_and(|x| x == "dae") {
                        let src =
                            String::from_utf8_lossy(&std::fs::read(p).map_err(|e| e.to_string())?)
                                .into_owned();
                        meshes.extend(crate::flexmesh::read_collada(&src)?);
                    } else if name.ends_with(".materials.json") {
                        mats.extend(crate::flexmesh::read_materials(
                            &std::fs::read_to_string(p).unwrap_or_default(),
                        ));
                    }
                }
                // A glTF render model with weighted skins (flexskin): the
                // `.rskin` and the `.glb` beside it.
                let mut skin = None;
                if let Some(sk) = paths
                    .iter()
                    .find(|p| p.extension().is_some_and(|x| x == "rskin"))
                {
                    skin = Some(crate::flexskin::Skin::decode(
                        &std::fs::read(sk).map_err(|e| e.to_string())?,
                    )?);
                    if let Some(g) = paths.iter().find(|p| {
                        p.extension().is_some_and(|x| x == "glb") && p.parent() == sk.parent()
                    }) {
                        let g = crate::glb::read(&std::fs::read(g).map_err(|e| e.to_string())?)?;
                        meshes.extend(g.meshes.iter().map(|m| crate::flexskin::glb_mesh(&g, m)));
                    }
                }
                crate::flexmesh::jbeam_visual_skinned(
                    &flex,
                    &props,
                    desc,
                    &spawned,
                    &meshes,
                    &mats,
                    [0.5, 0.05, 0.04],
                    skin.as_ref(),
                )
                .map_err(|e| eprintln!("visual: {e}"))
                .ok()
                .map(|r| r.0)
            }
        };
        let mut template = spawned;
        template.surface_contact = surface_contact();
        // A measurement (research/explicit-stability): `RUBBLE_DAMP_SCHEME=N` runs
        // every scene on that damping velocity (`ExplicitBeams::damp_scheme`).
        // (4: `ExplicitBeamsSettings::centred_damping`.)
        if let Some(k) = std::env::var("RUBBLE_DAMP_SCHEME")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
        {
            // (0: centred damping off; the BeamNG profile's default is on.)
            template.settings.centred_damping = k == 4;
            if k != 4 {
                template.damp_scheme = k;
            }
        }
        place(&mut template, angle, 0.02);
        let spawn_pos = positions(&template);
        steps(&mut template, 1.0);
        let skin = skin_nodes(&template);
        let frame = frame_of(&template, &skin);
        let n = template.nodes.len();
        // Parts and hardware.
        let (parts, hardware): (Vec<(String, Vec<u32>)>, Vec<(String, Vec<(u32, u32)>)>) =
            match &source {
                Source::Authored(b) => {
                    let sk = &b.skeleton;
                    let mut by: Vec<Vec<u32>> = vec![Vec::new(); sk.parts.len()];
                    for (i, q) in sk.nodes.iter().enumerate() {
                        by[q.part as usize].push(i as u32);
                    }
                    let parts: Vec<(String, Vec<u32>)> = sk
                        .parts
                        .iter()
                        .cloned()
                        .zip(by)
                        .filter(|(_, v)| !v.is_empty())
                        .collect();
                    let hw = b
                        .hardware
                        .iter()
                        .map(|h| {
                            (
                                h.name.clone(),
                                h.beams
                                    .iter()
                                    .map(|&j| (sk.beams[j].a, sk.beams[j].b))
                                    .collect(),
                            )
                        })
                        .collect();
                    (parts, hw)
                }
                Source::Jbeam { desc, .. } => {
                    let parts: Vec<(String, Vec<u32>)> = desc
                        .parts
                        .iter()
                        .map(|p| {
                            (
                                p.name.clone(),
                                p.nodes
                                    .iter()
                                    .copied()
                                    .filter(|&i| {
                                        (i as usize) < n && !is_tyre(&template, i as usize)
                                    })
                                    .collect::<Vec<u32>>(),
                            )
                        })
                        .filter(|(_, v)| !v.is_empty())
                        .collect();
                    let mut hw: Vec<(String, Vec<(u32, u32)>)> = desc
                        .break_groups
                        .iter()
                        .map(|g| (g.clone(), Vec::new()))
                        .collect();
                    for b in &desc.beams {
                        let g = b.break_group as usize;
                        if g > 0 && g <= hw.len() {
                            hw[g - 1].1.push((b.a, b.b));
                        }
                    }
                    hw.retain(|h| !h.1.is_empty());
                    (parts, hw)
                }
            };
        let door = pick_door(spec, &template, &parts);
        let settle = settle_moves(&template, &spawn_pos, &parts);
        Ok(Car {
            spec: spec.clone(),
            source,
            template,
            frame,
            parts,
            hardware,
            skin,
            visual,
            door,
            warnings,
            settle,
        })
    }

    pub fn mass(&self) -> f32 {
        self.template.nodes.iter().map(|n| n.mass).sum()
    }
}

/// The front-left door: the manifest's part, else the part named like a door
/// whose centre is left of the centreline, frontmost.
fn pick_door(
    spec: &CarSpec,
    s: &ExplicitBeams,
    parts: &[(String, Vec<u32>)],
) -> Option<(String, Vec<u32>)> {
    let centre =
        |ids: &[u32]| ids.iter().map(|&i| s.nodes[i as usize].pos).sum::<Vec3>() / ids.len() as f32;
    if let Some(p) = &spec.door_part {
        return parts.iter().find(|(n, _)| n == p).cloned();
    }
    if let Some(p) = parts.iter().find(|(n, _)| n == "door_f_l") {
        return Some(p.clone());
    }
    parts
        .iter()
        .filter(|(n, ids)| {
            n.to_lowercase().contains("door") && ids.len() >= 4 && centre(ids).x > 0.2
        })
        .max_by(|a, b| centre(&a.1).z.total_cmp(&centre(&b.1).z))
        .cloned()
}

impl Car {
    /// A node's name: its JBeam name, else its authored part and index.
    pub fn node_name(&self, i: usize) -> String {
        match &self.source {
            Source::Jbeam { desc, .. } => desc
                .nodes
                .get(i)
                .map_or(format!("wheel#{i}"), |n| n.name.clone()),
            Source::Authored(b) => b.skeleton.nodes.get(i).map_or(format!("tyre#{i}"), |q| {
                format!("{}#{i}", b.skeleton.parts[q.part as usize])
            }),
        }
    }
}

/// Per part, the largest move of its nodes over the settle about the whole
/// car's rigid fit (tyres excluded), largest first: a part the content
/// leaves unheld (a lid that springs open) shows here.
fn settle_moves(
    s: &ExplicitBeams,
    spawn: &[Vec3],
    parts: &[(String, Vec<u32>)],
) -> Vec<(String, f32)> {
    let now = positions(s);
    let body: Vec<u32> = (0..s.nodes.len() as u32)
        .filter(|&i| !is_tyre(s, i as usize))
        .collect();
    let fit = rigid_fit(spawn, &now, &body);
    let mut v: Vec<(String, f32)> = parts
        .iter()
        .map(|(n, ids)| {
            (
                n.clone(),
                ids.iter()
                    .map(|&i| fit.to_rest(now[i as usize]).distance(spawn[i as usize]))
                    .fold(0.0, f32::max),
            )
        })
        .collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v
}

/// Adds `deepest_node_name` beside every `deepest_node` in `v`.
pub fn name_nodes(v: &mut Value, car: &Car) {
    match v {
        Value::Object(m) => {
            if let Some(i) = m.get("deepest_node").and_then(|x| x.as_u64()) {
                m.insert("deepest_node_name".into(), json!(car.node_name(i as usize)));
            }
            for (_, x) in m.iter_mut() {
                name_nodes(x, car);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|x| name_nodes(x, car)),
        _ => {}
    }
}

// --- Geometry helpers ----------------------------------------------------------

/// A rigid fit of `now` onto `rest` over `ids`: rest ≈ c0 + R (now − c1).
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub c0: DVec3,
    pub c1: DVec3,
    /// Takes now-relative vectors to rest-relative ones.
    pub r: DMat3,
}

impl Fit {
    pub fn to_rest(&self, p: Vec3) -> Vec3 {
        (self.c0 + self.r * (p.as_dvec3() - self.c1)).as_vec3()
    }
}

/// Horn's closed form (unit quaternion, the largest eigenvector of his 4×4
/// matrix by shifted power iteration, f64).
pub fn rigid_fit(rest: &[Vec3], now: &[Vec3], ids: &[u32]) -> Fit {
    let n = ids.len().max(1) as f64;
    let c0 = ids
        .iter()
        .map(|&i| rest[i as usize].as_dvec3())
        .sum::<DVec3>()
        / n;
    let c1 = ids
        .iter()
        .map(|&i| now[i as usize].as_dvec3())
        .sum::<DVec3>()
        / n;
    // S = Σ a bᵀ with a = now − c1 (source), b = rest − c0 (target).
    let mut m = [[0.0f64; 3]; 3];
    for &i in ids {
        let a = now[i as usize].as_dvec3() - c1;
        let b = rest[i as usize].as_dvec3() - c0;
        let (aa, bb) = (a.to_array(), b.to_array());
        for r in 0..3 {
            for c in 0..3 {
                m[r][c] += aa[r] * bb[c];
            }
        }
    }
    let [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = m;
    let k = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let shift: f64 = k.iter().flatten().map(|x| x.abs()).sum::<f64>() + 1e-12;
    let mut q = [1.0f64, 0.0, 0.0, 0.0];
    for _ in 0..200 {
        let mut y = [0.0f64; 4];
        for r in 0..4 {
            y[r] = shift * q[r] + (0..4).map(|c| k[r][c] * q[c]).sum::<f64>();
        }
        let l = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if l == 0.0 {
            break;
        }
        q = y.map(|v| v / l);
    }
    let r = DMat3::from_quat(DQuat::from_xyzw(q[1], q[2], q[3], q[0]).normalize());
    Fit { c0, c1, r }
}

fn positions(s: &ExplicitBeams) -> Vec<Vec3> {
    s.nodes.iter().map(|n| s.origin + n.pos).collect()
}

/// Nodes with at least one intact beam.
/// The nodes still joined to `seeds` (the cabin) through unbroken beams: a part
/// torn off with its own beams intact is not (its nodes are excluded from the
/// crush measure, as a detached fascia or lamp lies ahead of the crushed front).
fn attached_to(s: &ExplicitBeams, seeds: &[u32]) -> Vec<bool> {
    let mut on = vec![false; s.nodes.len()];
    let mut stack: Vec<u32> = seeds.to_vec();
    for &i in seeds {
        on[i as usize] = true;
    }
    while let Some(i) = stack.pop() {
        for &b in &s.incidence[i as usize] {
            let bm = &s.beams[b as usize];
            if bm.broken {
                continue;
            }
            for o in [bm.a, bm.b] {
                if !on[o as usize] {
                    on[o as usize] = true;
                    stack.push(o);
                }
            }
        }
    }
    on
}

fn pct(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(f32::total_cmp);
    v[((v.len() - 1) as f32 * q).round() as usize]
}

/// 100 Hz two-pole Butterworth, forward and backward (the reference scenes').
fn butterworth(x: &[f32], dt: f32, hz: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let pass = |x: &[f32]| -> Vec<f32> {
        let w = (std::f32::consts::PI * hz * dt).tan();
        let n = 1.0 / (1.0 + 2f32.sqrt() * w + w * w);
        let (b0, a1, a2) = (
            w * w * n,
            2.0 * (w * w - 1.0) * n,
            (1.0 - 2f32.sqrt() * w + w * w) * n,
        );
        let (b1, b2) = (2.0 * b0, b0);
        let (mut x1, mut x2, mut y1, mut y2) = (x[0], x[0], x[0], x[0]);
        x.iter()
            .map(|&v| {
                let o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                (x2, x1, y2, y1) = (x1, v, y1, o);
                o
            })
            .collect()
    };
    let mut f = pass(x);
    f.reverse();
    let mut f = pass(&f);
    f.reverse();
    f
}

fn r1(x: f32) -> f64 {
    (f64::from(x) * 10.0).round() / 10.0
}
fn r3(x: f32) -> f64 {
    (f64::from(x) * 1000.0).round() / 1000.0
}
fn mm(x: f32) -> f64 {
    r1(x * 1e3)
}

// --- Shared measurements ---------------------------------------------------------

/// Broken hardware groups and detached parts.
fn damage_report(car: &Car, s: &ExplicitBeams) -> Value {
    let mut at: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (j, b) in s.beams.iter().enumerate() {
        at.entry((b.a.min(b.b), b.a.max(b.b))).or_default().push(j);
    }
    let broken_hw: Vec<&str> = car
        .hardware
        .iter()
        .filter(|(_, pairs)| {
            pairs.iter().any(|&(a, b)| {
                at.get(&(a.min(b), a.max(b)))
                    .is_some_and(|js| js.iter().any(|&j| s.beams[j].broken))
            })
        })
        .map(|(n, _)| n.as_str())
        .collect();
    // Connected components over intact non-tyre beams; the main one holds the
    // most mass. A part is detached when most of its mass is outside it.
    let n = s.nodes.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for b in s.beams.iter().filter(|b| !b.broken) {
        let (a, c) = (b.a as usize, b.b as usize);
        if a >= n || c >= n || is_tyre(s, a) || is_tyre(s, c) {
            continue;
        }
        let (ra, rc) = (find(&mut parent, a), find(&mut parent, c));
        if ra != rc {
            parent[ra] = rc;
        }
    }
    let mut mass: HashMap<usize, f32> = HashMap::new();
    for i in (0..n).filter(|&i| !is_tyre(s, i)) {
        let r = find(&mut parent, i);
        *mass.entry(r).or_default() += s.nodes[i].mass;
    }
    let main = mass
        .iter()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(k, _)| *k)
        .unwrap_or(0);
    let mut detached = Vec::new();
    for (name, ids) in &car.parts {
        let (mut out, mut all) = (0.0f32, 0.0f32);
        for &i in ids {
            let m = s.nodes[i as usize].mass;
            all += m;
            if find(&mut parent, i as usize) != main {
                out += m;
            }
        }
        if all > 0.0 && out > 0.5 * all {
            detached.push(name.clone());
        }
    }
    let free = (0..n)
        .filter(|&i| !is_tyre(s, i) && find(&mut parent, i) != main)
        .count();
    let (broken, bent) = s.damage();
    json!({
        "broken_beams": broken, "bent_beams": bent,
        "hardware_broken": broken_hw.len(), "hardware_total": car.hardware.len(), "hardware_broken_names": broken_hw,
        "parts_detached": detached.len(), "parts_detached_names": detached, "nodes_off_main_body": free,
        "finite": s.nodes.iter().all(|q| q.pos.is_finite()),
    })
}

/// Per part, the RMS over its nodes of √2·σ of each node's displacement
/// about the part's rigid fit, over a window of samples (m).
struct Ripple {
    parts: Vec<(String, Vec<u32>)>,
    rest: Vec<Vec3>,
    /// Per part per node: Σr, Σ|r|²; samples.
    acc: Vec<Vec<(DVec3, f64)>>,
    count: usize,
}

impl Ripple {
    fn new(car: &Car, rest: Vec<Vec3>) -> Self {
        // Panels: parts of four or more nodes, at least half of them on the collision skin.
        let skin: std::collections::HashSet<u32> = car.skin.iter().copied().collect();
        let parts: Vec<(String, Vec<u32>)> = car
            .parts
            .iter()
            .filter(|(_, v)| {
                v.len() >= 4 && 2 * v.iter().filter(|i| skin.contains(i)).count() >= v.len()
            })
            .cloned()
            .collect();
        let acc = parts
            .iter()
            .map(|(_, v)| vec![(DVec3::ZERO, 0.0); v.len()])
            .collect();
        Ripple {
            parts,
            rest,
            acc,
            count: 0,
        }
    }
    fn sample(&mut self, now: &[Vec3]) {
        for (k, (_, ids)) in self.parts.iter().enumerate() {
            let f = rigid_fit(&self.rest, now, ids);
            for (j, &i) in ids.iter().enumerate() {
                let r = (f.to_rest(now[i as usize]) - self.rest[i as usize]).as_dvec3();
                let a = &mut self.acc[k][j];
                a.0 += r;
                a.1 += r.length_squared();
            }
        }
        self.count += 1;
    }
    /// (worst part, its ripple; median part's; top five).
    fn result(&self) -> Value {
        let n = self.count.max(1) as f64;
        let mut per: Vec<(String, f64)> = self
            .parts
            .iter()
            .zip(&self.acc)
            .map(|((name, _), a)| {
                let ms = a
                    .iter()
                    .map(|(s, q)| 2.0 * (q / n - (*s / n).length_squared()).max(0.0))
                    .sum::<f64>()
                    / a.len() as f64;
                (name.clone(), ms.sqrt())
            })
            .collect();
        per.sort_by(|a, b| b.1.total_cmp(&a.1));
        let med = per.get(per.len() / 2).map_or(0.0, |p| p.1);
        json!({
            "worst_part": per.first().map(|p| p.0.clone()), "worst_mm": per.first().map_or(0.0, |p| (p.1 * 1e4).round() / 10.0),
            "median_part_mm": (med * 1e4).round() / 10.0,
            "top": per.iter().take(5).map(|(n, v)| json!([n, (v * 1e4).round() / 10.0])).collect::<Vec<_>>(),
        })
    }
}

/// Occupant points (template frame; x on the driver's side): **Choice**, a
/// mid-size sedan's package scaled by the wheelbase from the front axle.
fn occupant_points(f: &Frame) -> Vec<(&'static str, Vec3)> {
    let (fa, wb) = (f.front_axle, f.wheelbase());
    let x = 0.37;
    vec![
        ("feet", Vec3::new(x, 0.25, fa - 0.32 * wb)),
        ("knees", Vec3::new(x, 0.55, fa - 0.42 * wb)),
        ("chest", Vec3::new(x, 0.85, fa - 0.55 * wb)),
        ("hip", Vec3::new(x, 0.50, fa - 0.60 * wb)),
        ("head", Vec3::new(x, 1.15, fa - 0.64 * wb)),
    ]
}

/// Survival-space intrusion: at each occupant point the most any surrounding
/// node (0.10–0.60 m from it at rest, attached, not a tyre's) has moved
/// toward it, in the cabin's frame (`fit_ids`).
struct Intrusion {
    pts: Vec<(&'static str, Vec3, Vec<(u32, f32)>)>,
    fit_ids: Vec<u32>,
    rest: Vec<Vec3>,
    max: Vec<f32>,
    last: Vec<f32>,
}

impl Intrusion {
    fn new(car: &Car, rest: Vec<Vec3>, fit_ids: Vec<u32>) -> Self {
        let s = &car.template;
        let pts: Vec<_> = occupant_points(&car.frame)
            .into_iter()
            .map(|(n, p)| {
                let near: Vec<(u32, f32)> = (0..s.nodes.len())
                    .filter(|&i| !is_tyre(s, i))
                    .map(|i| (i as u32, rest[i].distance(p)))
                    .filter(|&(_, d)| (0.10..0.60).contains(&d))
                    .collect();
                (n, p, near)
            })
            .collect();
        let k = pts.len();
        Intrusion {
            pts,
            fit_ids,
            rest,
            max: vec![0.0; k],
            last: vec![0.0; k],
        }
    }
    fn sample(&mut self, s: &ExplicitBeams) {
        let now = positions(s);
        let att = attached_to(s, &self.fit_ids);
        let f = rigid_fit(&self.rest, &now, &self.fit_ids);
        for (k, (_, p, near)) in self.pts.iter().enumerate() {
            let v = near
                .iter()
                .filter(|(i, _)| att[*i as usize] && now[*i as usize].is_finite())
                .map(|&(i, d0)| d0 - f.to_rest(now[i as usize]).distance(*p))
                .fold(0.0f32, f32::max);
            self.last[k] = v;
            self.max[k] = self.max[k].max(v);
        }
    }
    fn result(&self) -> Value {
        let mut o = serde_json::Map::new();
        for (k, (n, _, near)) in self.pts.iter().enumerate() {
            o.insert(n.to_string(), json!({"max_mm": mm(self.max[k]), "residual_mm": mm(self.last[k]), "nodes": near.len()}));
        }
        Value::Object(o)
    }
}

/// The visual quality of the render mesh now against the settled rest
/// (`visual_metrics`).
thread_local! {
    /// `SEAM_DEBUG`: the last visual_metrics' opened pairs (vertex, vertex, gap).
    static SEAM_PAIRS: std::cell::RefCell<Vec<(usize, usize, f32)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// `SEAM_DEBUG`: per piece pair, whether the opened seams are the parts'
/// nodes separating (the gap between each vertex's heaviest node pair) or the
/// skin reaching past its nodes.
fn seam_diagnosis(
    car: &Car,
    v: &ShowcaseVisual,
    rest: &[Vec3],
    now: &[Vec3],
    pairs: &[(usize, usize, f32)],
) {
    let top = |i: usize| -> Option<u32> {
        if v.wt_start.len() != v.positions.len() + 1 {
            return Some(v.bind_nodes[i][0]);
        }
        let (a, b) = (v.wt_start[i] as usize, v.wt_start[i + 1] as usize);
        (a..b)
            .max_by(|&x, &y| v.wt_w[x][0].total_cmp(&v.wt_w[y][0]))
            .map(|k| v.wt_node[k])
    };
    let mut piece = vec![0usize; v.positions.len()];
    for (k, (_, first, count)) in v.pieces.iter().enumerate() {
        for x in *first..first + count {
            piece[x as usize] = k;
        }
    }
    let mut by: std::collections::BTreeMap<String, (usize, f32, f32, usize)> = Default::default();
    // The widest-opened node pairs per piece pair (names, node gap in mm).
    let mut worst: std::collections::BTreeMap<String, Vec<(f32, u32, u32)>> = Default::default();
    for &(i, j, g) in pairs {
        let (Some(a), Some(b)) = (top(i), top(j)) else {
            continue;
        };
        let ng =
            now[a as usize].distance(now[b as usize]) - rest[a as usize].distance(rest[b as usize]);
        let key = format!("{} / {}", v.pieces[piece[i]].0, v.pieces[piece[j]].0);
        let e = by.entry(key.clone()).or_insert((0, 0.0, 0.0, 0));
        e.0 += 1;
        e.1 += g;
        e.2 += ng;
        if ng > 0.05 {
            e.3 += 1;
        }
        let w = worst.entry(key).or_default();
        if !w.iter().any(|x| x.1 == a && x.2 == b) {
            w.push((ng, a, b));
        }
    }
    let mut v2: Vec<_> = by.into_iter().collect();
    v2.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    for (k, (n, g, ng, nb)) in v2.iter().take(16) {
        eprintln!(
            "  seam {k}: {n} pairs, skin gap mean {:.0} mm, top-node gap mean {:.0} mm, node gap > 5 cm {nb}",
            g / *n as f32 * 1e3,
            ng / *n as f32 * 1e3
        );
        if let Some(w) = worst.get_mut(k) {
            w.sort_by(|x, y| y.0.total_cmp(&x.0));
            let top: Vec<String> = w
                .iter()
                .take(3)
                .map(|&(g, a, b)| {
                    format!(
                        "{}-{} {:.0}",
                        car.node_name(a as usize),
                        car.node_name(b as usize),
                        g * 1e3
                    )
                })
                .collect();
            eprintln!("    widest node pairs (mm): {}", top.join(", "));
        }
    }
}

/// `HW_DEBUG`: per hardware group (hinges, latches, bolts) the largest
/// tension any of its beams carried and when, and whether it broke.
pub struct HwTrack {
    on: bool,
    groups: Vec<(String, Vec<usize>, f32, f32)>,
}

impl HwTrack {
    pub fn new(car: &Car, s: &ExplicitBeams) -> HwTrack {
        let on = std::env::var("HW_DEBUG").is_ok();
        let mut at: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        if on {
            for (j, b) in s.beams.iter().enumerate() {
                at.entry((b.a.min(b.b), b.a.max(b.b))).or_default().push(j);
            }
        }
        let groups = if on {
            let mut g: Vec<(String, Vec<usize>, f32, f32)> = car
                .hardware
                .iter()
                .map(|(n, pairs)| {
                    let js: Vec<usize> = pairs
                        .iter()
                        .flat_map(|&(a, b)| {
                            at.get(&(a.min(b), a.max(b))).cloned().unwrap_or_default()
                        })
                        .collect();
                    (n.clone(), js, 0.0f32, 0.0f32)
                })
                .collect();
            // Couplers (latches) by their nodes' names.
            for (j, b) in s.beams.iter().enumerate() {
                if b.kind == ExplicitKind::JbeamCoupler {
                    g.push((
                        format!(
                            "coupler {}-{}",
                            car.node_name(b.a as usize),
                            car.node_name(b.b as usize)
                        ),
                        vec![j],
                        0.0,
                        0.0,
                    ));
                }
            }
            g
        } else {
            Vec::new()
        };
        HwTrack { on, groups }
    }
    pub fn sample(&mut self, s: &ExplicitBeams, t: f32) {
        if !self.on {
            return;
        }
        for g in &mut self.groups {
            // The group's load: its beams' tensions summed (a bolt's or hinge's
            // beams share it).
            let f: f32 =
                g.1.iter()
                    .filter(|&&j| !s.beams[j].broken)
                    .map(|&j| (-s.beams[j].force).max(0.0))
                    .sum();
            if f > g.2 {
                g.2 = f;
                g.3 = t;
            }
        }
    }
    pub fn report(&self, scene: &str, s: &ExplicitBeams) {
        if !self.on {
            return;
        }
        let mut v: Vec<&(String, Vec<usize>, f32, f32)> = self.groups.iter().collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        for g in v {
            let broken = g.1.iter().any(|&j| s.beams[j].broken);
            eprintln!(
                "  hw {scene} {}: {} beams, peak tension {:.1} kN at {:.0} ms{}",
                g.0,
                g.1.len(),
                g.2 / 1e3,
                g.3 * 1e3,
                if broken { ", BROKE" } else { "" }
            );
        }
    }
}

fn visual_now(car: &Car, s: &ExplicitBeams) -> Value {
    let Some(v) = &car.visual else {
        return json!({"available": false});
    };
    let beams: Vec<BeamState> = s
        .beams
        .iter()
        .map(|b| BeamState {
            broken: b.broken,
            rest: b.rest,
        })
        .collect();
    let rest_beams: Vec<BeamState> = car
        .template
        .beams
        .iter()
        .map(|b| BeamState {
            broken: b.broken,
            rest: b.rest,
        })
        .collect();
    let units = unit_states(v, &beams);
    let rest_units = unit_states(v, &rest_beams);
    let (p0, n0) = deform_beams(
        v,
        &positions(&car.template),
        Some(&rest_units),
        Some(&rest_beams),
    );
    let (p1, n1) = deform_beams(v, &positions(s), Some(&units), Some(&beams));
    SEAM_PAIRS.with(|c| c.borrow_mut().clear());
    let out = visual_metrics(v, &p0, &n0, &p1, &n1);
    if std::env::var("SEAM_DEBUG").is_ok() {
        let pairs = SEAM_PAIRS.with(|c| c.borrow().clone());
        seam_diagnosis(car, v, &positions(&car.template), &positions(s), &pairs);
    }
    out
}

/// Render-mesh quality after a deformation, wheels excluded:
/// * **stretch**: each triangle's principal stretches (the singular values of
///   its 2D deformation gradient from rest): the extremes and the counts past
///   1.5× and below 0.5×;
/// * **flipped**: triangles whose geometric normal opposes their vertices'
///   shading normals, new since rest;
/// * **torn**: vertices the renderer drops (binding triangle torn, glass shattered);
/// * **seams**: boundary vertices of two different pieces within 1 cm at rest
///   (shut lines and joins) and how far they opened;
/// * **faceting**: the change of the dihedral angle across each interior
///   edge of a piece (a bent panel's creases).
pub fn visual_metrics(
    v: &ShowcaseVisual,
    p0: &[Vec3],
    n0: &[Vec3],
    p1: &[Vec3],
    n1: &[Vec3],
) -> Value {
    let nv = v.positions.len();
    let mut piece = vec![u32::MAX; nv];
    for (k, (_, first, count)) in v.pieces.iter().enumerate() {
        for i in *first..first + count {
            if (i as usize) < nv {
                piece[i as usize] = k as u32;
            }
        }
    }
    let wheel = |i: usize| v.wheel.get(i).is_some_and(|&w| w != u32::MAX);
    let ok = |p: &[Vec3], i: usize| p[i].is_finite();
    let tris: Vec<[usize; 3]> = v
        .indices
        .chunks_exact(3)
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
        .filter(|t| t.iter().all(|&i| !wheel(i)) && t.iter().all(|&i| ok(p0, i)))
        .collect();
    let verts: Vec<usize> = (0..nv).filter(|&i| !wheel(i) && ok(p0, i)).collect();
    let torn = verts.iter().filter(|&&i| !ok(p1, i)).count();
    // Stretch and flips.
    let (mut smax, mut smin) = (Vec::new(), Vec::new());
    let (mut over, mut under, mut flips, mut flips0, mut live) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let normal = |p: &[Vec3], t: &[usize; 3]| (p[t[1]] - p[t[0]]).cross(p[t[2]] - p[t[0]]);
    let flipped = |p: &[Vec3], n: &[Vec3], t: &[usize; 3]| {
        let g = normal(p, t);
        let sh = n[t[0]] + n[t[1]] + n[t[2]];
        g.length_squared() > 1e-14 && sh.length_squared() > 1e-6 && g.dot(sh) < 0.0
    };
    for t in &tris {
        let e1 = p0[t[1]] - p0[t[0]];
        let e2 = p0[t[2]] - p0[t[0]];
        let area = e1.cross(e2).length();
        if area < 1e-8 {
            continue;
        }
        let was_flipped = flipped(p0, n0, t);
        if was_flipped {
            flips0 += 1;
        }
        if t.iter().any(|&i| !ok(p1, i)) {
            continue;
        }
        live += 1;
        // Rest basis (u along e1, w in plane).
        let u = e1.normalize();
        let w = e1.cross(e2).cross(e1).normalize();
        let (a11, a21, a12, a22) = (e1.dot(u), e1.dot(w), e2.dot(u), e2.dot(w));
        let det = a11 * a22 - a12 * a21;
        let f1 = p1[t[1]] - p1[t[0]];
        let f2 = p1[t[2]] - p1[t[0]];
        // F = [f1 f2] A⁻¹, A = [[a11 a12],[a21 a22]].
        let c1 = (f1 * a22 - f2 * a21) / det;
        let c2 = (f2 * a11 - f1 * a12) / det;
        let (g11, g12, g22) = (c1.dot(c1), c1.dot(c2), c2.dot(c2));
        let tr = 0.5 * (g11 + g22);
        let disc = (0.25 * (g11 - g22) * (g11 - g22) + g12 * g12).sqrt();
        let (l1, l2) = ((tr + disc).max(0.0).sqrt(), (tr - disc).max(0.0).sqrt());
        smax.push(l1);
        smin.push(l2);
        if l1 > 1.5 {
            over += 1;
        }
        if l2 < 0.5 {
            under += 1;
        }
        if !was_flipped && flipped(p1, n1, t) {
            flips += 1;
        }
    }
    // Edges: interior edges within a piece (two triangles), and boundary vertices.
    let mut edges: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (k, t) in tris.iter().enumerate() {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.entry((a.min(b), a.max(b))).or_default().push(k);
        }
    }
    // Welded positions: vertices at one place are one for adjacency
    // (meshes split vertices at UV and normal seams).
    let key = |p: Vec3| {
        (
            (p.x * 2e4).round() as i64,
            (p.y * 2e4).round() as i64,
            (p.z * 2e4).round() as i64,
        )
    };
    let mut weld: HashMap<(i64, i64, i64, u32), usize> = HashMap::new();
    let wid: Vec<usize> = (0..nv)
        .map(|i| {
            let (x, y, z) = key(p0[i]);
            let l = weld.len();
            *weld.entry((x, y, z, piece[i])).or_insert(l)
        })
        .collect();
    let mut wedges: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (k, t) in tris.iter().enumerate() {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (wa, wb) = (wid[a], wid[b]);
            if wa != wb {
                wedges.entry((wa.min(wb), wa.max(wb))).or_default().push(k);
            }
        }
    }
    let _ = edges;
    let mut dihedral = Vec::new();
    let (mut d30, mut d60, mut nd) = (0usize, 0usize, 0usize);
    for ts in wedges.values().filter(|ts| ts.len() == 2) {
        let (a, b) = (&tris[ts[0]], &tris[ts[1]]);
        if piece[a[0]] != piece[b[0]] {
            continue;
        }
        let ang = |p: &[Vec3]| -> Option<f32> {
            let (na, nb) = (normal(p, a), normal(p, b));
            (na.length_squared() > 1e-16 && nb.length_squared() > 1e-16)
                .then(|| na.normalize().dot(nb.normalize()).clamp(-1.0, 1.0).acos())
        };
        let (Some(t0), true) = (ang(p0), a.iter().chain(b.iter()).all(|&i| ok(p1, i))) else {
            continue;
        };
        let Some(t1) = ang(p1) else { continue };
        let d = (t1 - t0).abs().to_degrees();
        nd += 1;
        if d > 30.0 {
            d30 += 1;
        }
        if d > 60.0 {
            d60 += 1;
        }
        dihedral.push(d);
    }
    // Seams: boundary (welded) vertices; pair each with the nearest boundary
    // vertex of another piece within 1 cm at rest.
    let mut boundary = vec![false; nv];
    let mut wboundary: HashMap<usize, bool> = HashMap::new();
    for (&(a, b), ts) in &wedges {
        if ts.len() == 1 {
            wboundary.insert(a, true);
            wboundary.insert(b, true);
        }
    }
    for &i in &verts {
        boundary[i] = wboundary.contains_key(&wid[i]);
    }
    let cell = 0.01f32;
    let ck = |p: Vec3| {
        (
            (p.x / cell).floor() as i32,
            (p.y / cell).floor() as i32,
            (p.z / cell).floor() as i32,
        )
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for &i in verts.iter().filter(|&&i| boundary[i]) {
        grid.entry(ck(p0[i])).or_default().push(i);
    }
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for &i in verts.iter().filter(|&&i| boundary[i]) {
        let (x, y, z) = ck(p0[i]);
        let mut best: Option<(f32, usize)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &j in grid
                        .get(&(x + dx, y + dy, z + dz))
                        .map(|v| v.as_slice())
                        .unwrap_or(&[])
                    {
                        if piece[j] == piece[i] {
                            continue;
                        }
                        let d = p0[i].distance(p0[j]);
                        if d < cell && best.is_none_or(|b| d < b.0) {
                            best = Some((d, j));
                        }
                    }
                }
            }
        }
        if let Some((_, j)) = best {
            if i < j || !pairs.contains(&(j, i)) {
                pairs.push((i, j));
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    let mut gaps: Vec<f32> = Vec::new();
    let (mut g1, mut g5, mut gt) = (0usize, 0usize, 0usize);
    let mut seam_by: std::collections::BTreeMap<String, usize> = Default::default();
    for &(i, j) in &pairs {
        if !ok(p1, i) || !ok(p1, j) {
            gt += 1;
            continue;
        }
        let g = p1[i].distance(p1[j]) - p0[i].distance(p0[j]);
        if g > 0.01 {
            g1 += 1;
        }
        if g > 0.05 {
            g5 += 1;
            if std::env::var("SEAM_DEBUG").is_ok() {
                let nm = |k: usize| {
                    v.pieces
                        .get(piece[k] as usize)
                        .map_or("?", |q| q.0.as_str())
                };
                *seam_by
                    .entry(format!("{} / {}", nm(i), nm(j)))
                    .or_insert(0usize) += 1;
                SEAM_PAIRS.with(|c| c.borrow_mut().push((i, j, g)));
            }
        }
        gaps.push(g);
    }
    if !seam_by.is_empty() {
        let mut v: Vec<_> = seam_by.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!(
            "seams opened > 5 cm by piece pair: {:?}",
            &v[..v.len().min(12)]
        );
    }
    // Shut lines: every vertex (not only open mesh edges) paired with the
    // nearest vertex of another piece within SHUT_LINE at rest. A BeamNG
    // panel's edges are hemmed (closed meshes), so the boundary pairs above
    // never see its hood, door or lamp lines (vehicle-gen3.md round 5).
    let (shut, _) = rest_seam_pairs(v, p0, SHUT_LINE, false);
    let (mut s5, mut sgaps) = (0usize, Vec::new());
    for &(i, j) in &shut {
        if !ok(p1, i) || !ok(p1, j) {
            continue;
        }
        let g = p1[i].distance(p1[j]) - p0[i].distance(p0[j]);
        if g > 0.05 {
            s5 += 1;
        }
        sgaps.push(g);
    }
    json!({
        "shutline_pairs": shut.len(), "shutline_open_5cm": s5,
        "shutline_gap_p99_mm": mm(pct(&mut sgaps, 0.99)),
        "available": true, "vertices": verts.len(), "triangles": tris.len(),
        "torn_vertices": torn, "torn_fraction": r3(torn as f32 / verts.len().max(1) as f32),
        "stretch_max": r3(smax.iter().copied().fold(0.0, f32::max)), "stretch_p99": r3(pct(&mut smax, 0.99)),
        "compress_min": r3(smin.iter().copied().fold(f32::MAX, f32::min)), "compress_p01": r3(pct(&mut smin, 0.01)),
        "tris_over_1_5": over, "tris_under_0_5": under, "tris_live": live,
        "flipped_new": flips, "flipped_at_rest": flips0,
        "dihedral_edges": nd, "dihedral_p99_deg": r1(pct(&mut dihedral, 0.99)), "dihedral_max_deg": r1(dihedral.iter().copied().fold(0.0, f32::max)),
        "dihedral_over_30": d30, "dihedral_over_60": d60,
        "seam_pairs": pairs.len(), "seam_open_1cm": g1, "seam_open_5cm": g5, "seam_torn_pairs": gt,
        "seam_gap_p99_mm": mm(pct(&mut gaps, 0.99)), "seam_gap_max_mm": mm(gaps.iter().copied().fold(0.0, f32::max)),
    })
}

// --- Penetration into rigid colliders ---------------------------------------------

/// A rigid collider the car meets: a static box (the wall) or a static shape
/// (the pole).
#[derive(Clone, Copy, Debug)]
enum Rigid {
    Box(StaticBox),
    Shape(StaticShape),
}

impl Rigid {
    fn all(s: &ExplicitBeams) -> Vec<Rigid> {
        s.boxes
            .iter()
            .map(|b| Rigid::Box(*b))
            .chain(s.shapes.iter().map(|q| Rigid::Shape(*q)))
            .collect()
    }
    /// World box holding it.
    fn aabb(&self) -> (Vec3, Vec3) {
        match *self {
            Rigid::Box(b) => (b.lo, b.hi),
            Rigid::Shape(q) => {
                let b = q.bounds();
                (b.lo, b.hi)
            }
        }
    }
    /// How deep `p` is inside (m, 0 outside): its distance to the nearest surface.
    fn depth(&self, p: Vec3) -> f32 {
        match *self {
            Rigid::Box(b) => {
                let d = (p - b.lo).min(b.hi - p);
                d.min_element().max(0.0)
            }
            Rigid::Shape(StaticShape::OrientedBox {
                centre, half, axes, ..
            }) => {
                let q = p - centre;
                let l = Vec3::new(q.dot(axes[0]), q.dot(axes[1]), q.dot(axes[2]));
                (half - l.abs()).min_element().max(0.0)
            }
            Rigid::Shape(StaticShape::Cylinder {
                centre,
                radius,
                half_height,
            }) => {
                let r = radius - Vec3::new(p.x - centre.x, 0.0, p.z - centre.z).length();
                let h = half_height - (p.y - centre.y).abs();
                r.min(h).max(0.0)
            }
            Rigid::Shape(StaticShape::Capsule { a, b, radius }) => {
                let ab = b - a;
                let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
                (radius - p.distance(a + ab * t)).max(0.0)
            }
            Rigid::Shape(StaticShape::Sphere { centre, radius }) => {
                (radius - p.distance(centre)).max(0.0)
            }
        }
    }
    /// The deepest point of segment a–b (a vertical cylinder exactly: the
    /// segment's nearest point to the axis in plan view, within the caps;
    /// otherwise 17 samples).
    fn depth_segment(&self, a: Vec3, b: Vec3) -> f32 {
        if let Rigid::Shape(StaticShape::Cylinder { centre, .. }) = *self {
            let (a2, d2) = (
                Vec3::new(a.x - centre.x, 0.0, a.z - centre.z),
                Vec3::new(b.x - a.x, 0.0, b.z - a.z),
            );
            let t = (-a2.dot(d2) / d2.length_squared().max(1e-12)).clamp(0.0, 1.0);
            return self
                .depth(a + (b - a) * t)
                .max(self.depth(a))
                .max(self.depth(b));
        }
        (0..=16)
            .map(|k| self.depth(a + (b - a) * (k as f32 / 16.0)))
            .fold(0.0, f32::max)
    }
    /// The deepest point of triangle p0 p1 p2 (a vertical cylinder exactly:
    /// the triangle's nearest point to the axis in plan view, if its height
    /// there is within the caps; otherwise a 6×6 barycentric grid).
    fn depth_triangle(&self, p: [Vec3; 3]) -> f32 {
        if let Rigid::Shape(StaticShape::Cylinder { centre, .. }) = *self {
            let flat = |v: Vec3| glam::Vec2::new(v.x - centre.x, v.z - centre.z);
            let (a, b, c) = (flat(p[0]), flat(p[1]), flat(p[2]));
            // Barycentric of the axis (origin) in the plan-view triangle.
            let (v0, v1, v2) = (b - a, c - a, -a);
            let den = v0.x * v1.y - v1.x * v0.y;
            let mut best = [p[0], p[1], p[2]]
                .iter()
                .map(|&q| self.depth(q))
                .fold(0.0, f32::max);
            for (x, y) in [(p[0], p[1]), (p[1], p[2]), (p[2], p[0])] {
                best = best.max(self.depth_segment(x, y));
            }
            if den.abs() > 1e-12 {
                let u = (v2.x * v1.y - v1.x * v2.y) / den;
                let v = (v0.x * v2.y - v2.x * v0.y) / den;
                if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
                    best = best.max(self.depth(p[0] + (p[1] - p[0]) * u + (p[2] - p[0]) * v));
                }
            }
            return best;
        }
        let mut best = 0.0f32;
        for i in 0..=6 {
            for j in 0..=6 - i {
                let (u, v) = (i as f32 / 6.0, j as f32 / 6.0);
                best = best.max(self.depth(p[0] + (p[1] - p[0]) * u + (p[2] - p[0]) * v));
            }
        }
        best
    }
}

/// Per sample, the deepest node, beam segment and collision-triangle point of
/// the car inside any rigid collider; the max and p99 over the samples.
struct Penetration {
    rigid: Vec<Rigid>,
    nodes: Vec<f32>,
    beams: Vec<f32>,
    tris: Vec<f32>,
    /// The deepest node seen: (depth, index, mass).
    worst: (f32, usize, f32),
}

impl Penetration {
    fn new(s: &ExplicitBeams) -> Self {
        Penetration {
            rigid: Rigid::all(s),
            nodes: Vec::new(),
            beams: Vec::new(),
            tris: Vec::new(),
            worst: (0.0, 0, 0.0),
        }
    }
    fn sample(&mut self, s: &ExplicitBeams) {
        let p = positions(s);
        let (mut dn, mut db, mut dt) = (0.0f32, 0.0f32, 0.0f32);
        for r in &self.rigid {
            let (lo, hi) = r.aabb();
            let near = |q: Vec3| {
                q.cmpge(lo - Vec3::splat(0.01)).all() && q.cmple(hi + Vec3::splat(0.01)).all()
            };
            let hit = |a: Vec3, b: Vec3| a.min(b).cmple(hi).all() && a.max(b).cmpge(lo).all();
            for (i, q) in p.iter().enumerate().filter(|(i, q)| {
                q.is_finite() && near(**q) && s.nodes[*i].flags & NODE_NO_GROUND == 0
            }) {
                let d = r.depth(*q);
                if d > self.worst.0 {
                    self.worst = (d, i, s.nodes[i].mass);
                }
                dn = dn.max(d);
            }
            for b in s.beams.iter().filter(|b| {
                !b.broken
                    && b.a != b.b
                    && (s.nodes[b.a as usize].flags | s.nodes[b.b as usize].flags) & NODE_NO_GROUND
                        == 0
            }) {
                let (a, c) = (p[b.a as usize], p[b.b as usize]);
                if a.is_finite() && c.is_finite() && hit(a, c) {
                    db = db.max(r.depth_segment(a, c));
                }
            }
            for t in &s.triangles {
                let q = t.nodes.map(|i| p[i as usize]);
                if q.iter().all(|x| x.is_finite())
                    && hit(q[0].min(q[1]).min(q[2]), q[0].max(q[1]).max(q[2]))
                {
                    dt = dt.max(r.depth_triangle(q));
                }
            }
        }
        self.nodes.push(dn);
        self.beams.push(db);
        self.tris.push(dt);
    }
    fn result(&self) -> Value {
        let st = |v: &[f32]| {
            let mut a = v.to_vec();
            let max = a.iter().copied().fold(0.0, f32::max);
            let mut hot: Vec<f32> = a.iter().copied().filter(|&x| x > 0.0).collect();
            json!({"max_mm": mm(max), "p99_mm": mm(pct(&mut a, 0.99)), "p99_in_contact_mm": mm(pct(&mut hot, 0.99)), "samples_in_contact": hot.len()})
        };
        json!({"deepest_node": self.worst.1, "deepest_node_mass": r3(self.worst.2), "colliders": self.rigid.len(), "samples": self.nodes.len(), "nodes": st(&self.nodes), "beams": st(&self.beams), "triangles": st(&self.tris)})
    }
}

// --- Census --------------------------------------------------------------------

/// The car's census on its crash spawn (tyres excluded): nodes, mass per node,
/// spring ends per node, k·L, ε_y, the share at the step's bound, beam kinds,
/// mass and centre of mass.
pub fn census(car: &Car) -> Value {
    let s = &car.template;
    let g_limit = crate::authoring::skeleton::G_LIMIT;
    let h = STEP;
    let body: Vec<usize> = (0..s.nodes.len()).filter(|&i| !is_tyre(s, i)).collect();
    let mut masses: Vec<f32> = body.iter().map(|&i| s.nodes[i].mass).collect();
    let total: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let com = s.nodes.iter().map(|n| n.pos * n.mass).sum::<Vec3>() / total;
    let mut ends = vec![0u32; s.nodes.len()];
    let (mut kl, mut len, mut eps) = (Vec::new(), Vec::new(), Vec::new());
    let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
    for b in &s.beams {
        let (a, c) = (b.a as usize, b.b as usize);
        if a == c || is_tyre(s, a) || is_tyre(s, c) {
            continue;
        }
        *kinds.entry(format!("{:?}", b.kind)).or_default() += 1;
        if b.k > 0.0 && b.k.is_finite() {
            ends[a] += 1;
            ends[c] += 1;
            let l = s.nodes[a].pos.distance(s.nodes[c].pos);
            kl.push(b.k * l);
            len.push(l);
            if b.tc.is_finite() && b.tc > 0.0 && l > 0.0 {
                eps.push(b.tc / (b.k * l));
            }
        }
    }
    let row = |i: usize| -> f32 {
        let mi = s.nodes[i].mass.max(1e-6);
        let mut r = 0.0f32;
        for &j in &s.incidence[i] {
            let bm = &s.beams[j as usize];
            if bm.broken {
                continue;
            }
            let o = if bm.a as usize == i { bm.b } else { bm.a } as usize;
            let mo = s.nodes[o].mass.max(1e-6);
            r += (h * h * bm.k + 2.0 * h * bm.c) * (1.0 / mi + 1.0 / (mi * mo).sqrt());
        }
        r / (g_limit * h * h)
    };
    let rows: Vec<f32> = body.iter().map(|&i| row(i)).collect();
    let at_bound = rows.iter().filter(|&&r| r > 0.95).count();
    let mut ends_b: Vec<f32> = body.iter().map(|&i| ends[i] as f32).collect();
    let f = &car.frame;
    let front_share = (com.z - f.rear_axle) / f.wheelbase();
    let mut kl2 = kl.clone();
    json!({
        "nodes": body.len(), "tyre_nodes": s.nodes.len() - body.len(), "beams": s.beams.len(),
        "mass_kg": r1(total), "body_mass_kg": r1(masses.iter().sum()),
        "node_mass_p50": r3(pct(&mut masses, 0.5)), "node_mass_mean": r3(masses.iter().sum::<f32>() / masses.len().max(1) as f32),
        "ends_per_node_p50": pct(&mut ends_b, 0.5), "ends_per_node_mean": r1(ends_b.iter().sum::<f32>() / ends_b.len().max(1) as f32),
        "kl_p10": pct(&mut kl, 0.1).round(), "kl_p50": pct(&mut kl, 0.5).round(), "kl_p90": pct(&mut kl, 0.9).round(),
        "kl_mean": (kl2.iter().sum::<f32>() / kl2.len().max(1) as f32).round(), "kl_max": pct(&mut kl2, 1.0).round(),
        "len_p50": r3(pct(&mut len, 0.5)), "eps_y_p50": r3(pct(&mut eps, 0.5)),
        "at_bound_share": r3(at_bound as f32 / body.len().max(1) as f32),
        "beam_kinds": kinds,
        "com": [r3(com.x), r3(com.y), r3(com.z - 0.5 * (f.front_axle + f.rear_axle))],
        "com_height_m": r3(com.y), "front_axle_share": r3(front_share),
        "length_m": r3(f.length()), "width_m": r3(f.width()), "height_m": r3(f.hi.y), "wheelbase_m": r3(f.wheelbase()),
        "skin_triangles": s.triangles.len(),
        "visual_vertices": car.visual.as_ref().map(|v| v.positions.len()), "visual_triangles": car.visual.as_ref().map(|v| v.indices.len() / 3),
        "parts": car.parts.len(), "hardware_groups": car.hardware.len(),
        "settle_top_parts_mm": car.settle.iter().take(4).map(|(n, d)| json!([n, mm(*d)])).collect::<Vec<_>>(),
    })
}

// --- Scenes ----------------------------------------------------------------------

/// Parked 10 s on raycast wheels (`truck_export`: a JBeam car through the
/// page's default `.truck` export).
pub fn scene_rest(car: &Car, truck_export: bool, seconds: f32) -> Value {
    let t0 = Instant::now();
    let mut s = match raycast_spawn(&car.source, truck_export) {
        Ok(s) => s,
        Err(e) => return json!({"error": e}),
    };
    let t_spawn = t0.elapsed().as_secs_f32();
    // `RUBBLE_DAMP_SCHEME` (a measurement) on a BeamNG profile spawn as on the template.
    if std::env::var("RUBBLE_DAMP_SCHEME").is_ok() && s.beamng_profile {
        s.damp_scheme = car.template.damp_scheme;
        s.settings.centred_damping = car.template.settings.centred_damping;
    }
    let total: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let com = |s: &ExplicitBeams| s.nodes.iter().map(|n| n.pos * n.mass).sum::<Vec3>() / total;
    let mut first_break = None;
    let mut com_half = Vec3::ZERO;
    let mut com_1 = Vec3::ZERO;
    let n_steps = (seconds / STEP).round() as usize;
    // A run that diverges (a node's position no longer finite) stops there:
    // its state is lost, and stepping NaNs costs about 50× a live step (the
    // contact pass's boxes are no longer bounded).
    let mut diverged = None;
    for k in 0..n_steps {
        s.step();
        if first_break.is_none() && k % 20 == 0 && s.beams.iter().any(|b| b.broken) {
            first_break = Some((k + 1) as f32 * STEP);
        }
        if k % 20 == 0 && !s.nodes.iter().all(|n| n.pos.is_finite()) {
            diverged = Some((k + 1) as f32 * STEP);
            break;
        }
        if k + 1 == (1.0 / STEP) as usize {
            com_1 = com(&s);
        }
        if k + 1 == n_steps / 2 {
            com_half = com(&s);
        }
    }
    if std::env::var("COMPARE_TIMES").is_ok() {
        eprintln!(
            "  rest (truck export {truck_export}): spawned in {:.1} s, {} nodes, {} beams; stepped in {:.1} s",
            t_spawn,
            s.nodes.len(),
            s.beams.len(),
            t0.elapsed().as_secs_f32() - t_spawn
        );
    }
    let c = com(&s);
    let (broken, bent) = s.damage();
    let mut by: HashMap<String, usize> = HashMap::new();
    // The parts the broken beams are in (a node's first part; desc order is
    // the spawn's for both paths' body nodes).
    let mut part_of: HashMap<u32, &str> = HashMap::new();
    for (name, ids) in &car.parts {
        for &i in ids {
            part_of.entry(i).or_insert(name.as_str());
        }
    }
    for b in s.beams.iter().filter(|b| b.broken) {
        let pa = part_of.get(&b.a).copied().unwrap_or("?");
        let pb = part_of.get(&b.b).copied().unwrap_or("?");
        let k = if pa == pb {
            pa.to_string()
        } else {
            format!("{pa}/{pb}")
        };
        *by.entry(k).or_default() += 1;
    }
    let mut by: Vec<(String, usize)> = by.into_iter().collect();
    by.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let ke: f32 = s
        .nodes
        .iter()
        .map(|n| 0.5 * n.mass * n.vel.length_squared())
        .sum();
    let vmax = s.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max);
    let wall = t0.elapsed().as_secs_f32();
    json!({
        "path": if truck_export { "truck export, raycast wheels" } else if matches!(car.source, Source::Authored(_)) { ".truck, raycast wheels" } else { "JBeam mode, raycast wheels" },
        "seconds": seconds, "diverged_s": diverged.map(r3), "broken_beams": broken, "bent_beams": bent, "first_break_s": first_break.map(r3),
        "broken_by_part": by.iter().take(6).map(|(k, n)| json!([k, n])).collect::<Vec<_>>(),
        "ke_end_j": r1(ke), "vmax_end": r3(vmax),
        "creep_mm": mm((c - com_half).length()), "sag_mm": mm(com_1.y - c.y),
        "finite": s.nodes.iter().all(|n| n.pos.is_finite()),
        "wall_s_per_sim_s": r3(wall / diverged.unwrap_or(seconds)),
    })
}

/// Frontal rigid wall, full width, at `kmh`: the pulse at the rear cabin, the
/// crush, springback, the wall force, energy, ripple, intrusion, damage and
/// the render mesh after.
pub fn scene_wall(car: &Car, kmh: f32, seconds_after: f32) -> Value {
    let t0 = Instant::now();
    let v0 = kmh / 3.6;
    let f = car.frame;
    let mut s = car.template.clone();
    let rest = positions(&s);
    set_moving(&mut s, Vec3::Z * v0);
    let front = car
        .skin
        .iter()
        .map(|&i| s.nodes[i as usize].pos.z)
        .fold(f32::MIN, f32::max);
    let gap = 0.05;
    let wall_z = front + gap;
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 2.0, wall_z + 0.5),
        Vec3::new(0.5 * f.width() + 1.0, 2.0, 0.5),
    ));
    // The rear cabin (NCAP's rear-seat accelerometers stand in the middle of
    // the cabin's rear): non-tyre nodes over the rear half of the wheelbase.
    let cabin: Vec<u32> = (0..s.nodes.len() as u32)
        .filter(|&i| {
            !is_tyre(&s, i as usize)
                && (f.rear_axle - 0.1..f.rear_axle + 0.5 * f.wheelbase())
                    .contains(&rest[i as usize].z)
        })
        .collect();
    let cm = cabin.iter().map(|&i| s.nodes[i as usize].mass).sum::<f32>();
    let vcab = |s: &ExplicitBeams| {
        cabin
            .iter()
            .map(|&i| s.nodes[i as usize].mass * s.nodes[i as usize].vel.z)
            .sum::<f32>()
            / cm
    };
    let zcab = |s: &ExplicitBeams| {
        cabin
            .iter()
            .map(|&i| s.nodes[i as usize].mass * s.nodes[i as usize].pos.z)
            .sum::<f32>()
            / cm
    };
    let pz = |s: &ExplicitBeams| {
        s.nodes
            .iter()
            .map(|n| f64::from(n.mass) * f64::from(n.vel.z))
            .sum::<f64>()
    };
    let total: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let mut pen = Penetration::new(&s);
    // Ledger (one-vehicle world).
    let mut w = ExplicitBeamsWorld::new(vec![s]);
    w.contact = false;
    w.enable_ledger();
    let e0 = w.ledger.as_ref().unwrap().start;
    let ke0 = w.ledger.as_ref().unwrap().kinetic;
    let mut ripple = Ripple::new(car, rest.clone());
    let mut ripple_late = Ripple::new(car, rest.clone());
    let mut intr = Intrusion::new(car, rest.clone(), cabin.clone());
    let (mut t_contact, mut steps_after) = (None::<usize>, 0usize);
    let (mut z_contact, mut d_dyn, mut t_m, mut rebound) = (0.0f32, 0.0f32, None::<f32>, 0.0f32);
    let mut force: Vec<f32> = Vec::new();
    let mut acc: Vec<f32> = Vec::new();
    let mut p_prev = pz(&w.vehicles[0]);
    let mut v_prev = vcab(&w.vehicles[0]);
    let (mut e_min, mut e_rise) = (f64::MAX, 0.0f64);
    let mut k = 0usize;
    let after = (seconds_after / STEP).round() as usize;
    let limit = (1.5 / STEP) as usize + after;
    let mut hw = HwTrack::new(car, &w.vehicles[0]);
    while k < limit {
        w.step();
        k += 1;
        let s = &w.vehicles[0];
        hw.sample(s, steps_after as f32 * STEP);
        if k % 2 == 0 {
            pen.sample(s);
        }
        if t_contact.is_none() {
            let reach = car
                .skin
                .iter()
                .map(|&i| s.nodes[i as usize].pos.z)
                .fold(f32::MIN, f32::max);
            // Within the contact band (surface contact stops the skin there).
            if reach >= wall_z - 0.03 {
                t_contact = Some(k);
                z_contact = zcab(s);
            }
            p_prev = pz(s);
            v_prev = vcab(s);
            continue;
        }
        steps_after += 1;
        let p = pz(s);
        force.push(((p_prev - p) / f64::from(STEP)) as f32);
        p_prev = p;
        let v = vcab(s);
        acc.push((v - v_prev) / STEP);
        v_prev = v;
        let d = zcab(s) - z_contact;
        if t_m.is_none() {
            d_dyn = d_dyn.max(d);
            if v <= 0.0 {
                t_m = Some(steps_after as f32 * STEP);
            }
        } else {
            rebound = rebound.max(-v);
        }
        let ts = steps_after as f32 * STEP;
        // `WALL_DEBUG`: the plastic work by member every 10 ms, and the force.
        if steps_after % 20 == 0 && steps_after <= 160 && std::env::var("WALL_DEBUG").is_ok() {
            eprintln!(
                "  wall {:.0} ms: force {:.0} kN; plastic {}",
                ts * 1e3,
                force.iter().rev().take(20).sum::<f32>() / 20e3,
                plastic_by_member(car, s, 14)
            );
        }
        if steps_after % 10 == 0 {
            intr.sample(s);
            if let Some(l) = &w.ledger {
                let e = l.mechanical();
                e_min = e_min.min(e);
                e_rise = e_rise.max(e - e_min);
            }
        }
        if steps_after % 2 == 0 && (0.5..0.6).contains(&ts) {
            ripple.sample(&positions(s));
        }
        if steps_after % 2 == 0 && ts >= seconds_after - 0.1 {
            ripple_late.sample(&positions(s));
        }
        if steps_after >= after {
            break;
        }
    }
    let s = &w.vehicles[0];
    hw.report("wall", s);
    let now = positions(s);
    let l = w.ledger.as_ref().unwrap();
    // The pulse (100 Hz) and force (100 Hz), 1 ms samples to 150 ms.
    let a = butterworth(&acc, STEP, 100.0);
    let peak_g = a.iter().map(|x| -x).fold(0.0f32, f32::max) / 9.81;
    let fz = butterworth(&force, STEP, 100.0);
    let (ip, peak_f) = fz
        .iter()
        .enumerate()
        .fold((0, 0.0f32), |b, (i, &x)| if x > b.1 { (i, x) } else { b });
    let curve: Vec<f64> = fz
        .iter()
        .step_by(2)
        .take(150)
        .map(|x| r1(x / 1e3))
        .collect();
    let impulse: f32 = force.iter().sum::<f32>() * STEP;
    // Residual crush at bumper height (NHTSA's six zones), in the cabin's frame.
    let fit = rigid_fit(&rest, &now, &cabin);
    let att = attached_to(s, &cabin);
    let zones = 6usize;
    let (xl, xh) = (f.lo.x, f.hi.x);
    let band: Vec<u32> = car
        .skin
        .iter()
        .copied()
        .filter(|&i| {
            let p = rest[i as usize];
            (0.25..0.75).contains(&(p.y)) && p.z > f.hi.z - 1.2
        })
        .collect();
    let mut c = vec![0.0f32; zones];
    for (z, ci) in c.iter_mut().enumerate() {
        let x0 = xl + (xh - xl) * z as f32 / zones as f32;
        let x1 = xl + (xh - xl) * (z + 1) as f32 / zones as f32;
        let ids: Vec<u32> = band
            .iter()
            .copied()
            .filter(|&i| (x0..x1).contains(&rest[i as usize].x))
            .collect();
        let r0 = ids
            .iter()
            .map(|&i| rest[i as usize].z)
            .fold(f32::MIN, f32::max);
        // A node the crush left ahead of its rest place (in the cabin's frame) by
        // more than 5 cm is a part hanging loose (a lamp or fascia on its last
        // beams swings forward as the cabin rebounds), not the crushed front: a
        // frontal wall pushes the structure back only.
        let r1_ = ids
            .iter()
            .filter(|&&i| att[i as usize] && now[i as usize].is_finite())
            .map(|&i| (i, fit.to_rest(now[i as usize]).z))
            .filter(|&(i, z)| z <= rest[i as usize].z + 0.05)
            .map(|(_, z)| z)
            .fold(f32::MIN, f32::max);
        *ci = if ids.is_empty() || r1_ == f32::MIN {
            0.0
        } else {
            (r0 - r1_).max(0.0)
        };
        if std::env::var("COMPARE_CRUSH_DEBUG").is_ok() {
            let front = ids
                .iter()
                .copied()
                .filter(|&i| {
                    att[i as usize]
                        && now[i as usize].is_finite()
                        && fit.to_rest(now[i as usize]).z <= rest[i as usize].z + 0.05
                })
                .max_by(|&a, &b| {
                    fit.to_rest(now[a as usize])
                        .z
                        .total_cmp(&fit.to_rest(now[b as usize]).z)
                });
            let lead = ids
                .iter()
                .copied()
                .max_by(|&a, &b| rest[a as usize].z.total_cmp(&rest[b as usize].z));
            eprintln!(
                "(crush) zone {z}: {} nodes, rest front {r0:.3} (node {lead:?}), now front {r1_:.3} (node {front:?}, rest z {:.3})",
                ids.len(),
                front.map_or(0.0, |i| rest[i as usize].z)
            );
        }
    }
    let c_max = c.iter().copied().fold(0.0, f32::max);
    let c_avg = (0.5 * c[0] + c[1..5].iter().sum::<f32>() + 0.5 * c[5]) / 5.0;
    let b1 = (v0 - 8.0 / 3.6) / c_avg;
    let springback = if d_dyn > 0.0 {
        (d_dyn - c_max) / d_dyn
    } else {
        f32::NAN
    };
    let tm = t_m.unwrap_or(f32::NAN);
    let ke1 = l.kinetic;
    let closure = (l.kinetic - ke0) - l.kinetic_change();
    let wall_s = t0.elapsed().as_secs_f32();
    let sim_s = k as f32 * STEP;
    json!({
        "kmh": kmh, "mass_kg": r1(total),
        "dynamic_crush_mm": mm(d_dyn), "t_m_ms": r1(tm * 1e3), "peak_g": r1(peak_g), "mean_g_to_tm": r1(v0 / tm / 9.81),
        "rebound_ms": r3(rebound), "restitution": r3(rebound / v0),
        "residual_crush_zones_mm": c.iter().map(|x| mm(*x)).collect::<Vec<_>>(), "residual_crush_max_mm": mm(c_max), "residual_crush_avg_mm": mm(c_avg),
        "campbell_b1": r1(b1), "springback_ratio": r3(springback),
        "peak_force_kn": r1(peak_f / 1e3), "peak_force_ms": r1((ip as f32 * STEP) * 1e3), "impulse_kns": r3(impulse / 1e3),
        "force_kn_1ms": curve,
        "energy": {
            "start_kj": r3((e0 / 1e3) as f32), "ke_start_kj": r3((ke0 / 1e3) as f32), "ke_end_kj": r3((ke1 / 1e3) as f32),
            "elastic_end_kj": r3((l.elastic / 1e3) as f32), "beam_dissipation_kj": r3((l.beam_dissipation() / 1e3) as f32),
            "contact_work_kj": r3(((l.w_contact_normal + l.w_contact_friction + l.w_self_contact + l.w_contact_shift) / 1e3) as f32),
            "ground_work_kj": r3(((l.w_ground_vertical + l.w_ground_horizontal) / 1e3) as f32),
            "air_work_kj": r3((l.w_air / 1e3) as f32), "discretisation_kj": r3((l.discretisation / 1e3) as f32),
            "closure_share": (closure / e0.max(1.0) * 1e6).round() / 1e6, "max_rise_j": r1(e_rise as f32),
        },
        "penetration": pen.result(), "ripple_0_5s": ripple.result(), "ripple_end": ripple_late.result(),
        "intrusion": intr.result(),
        "damage": damage_report(car, s),
        "visual": visual_now(car, s),
        "sim_s": r3(sim_s), "wall_s_per_sim_s": r3(wall_s / sim_s),
    })
}

/// The plastic work so far by member (node-name stems without side or
/// station, an upper estimate: yield force times the rest length's change),
/// its total and the `top` largest (`SIDE_DEBUG`, `WALL_DEBUG`).
pub fn plastic_by_member(car: &Car, s: &ExplicitBeams, top: usize) -> String {
    let stem = |i: u32| -> String {
        let n = car.node_name(i as usize);
        let n = n.trim_end_matches(['L', 'R']);
        let n: String = n.chars().filter(|c| !c.is_ascii_digit()).collect();
        n.trim_end_matches(['L', 'R']).to_string()
    };
    let mut by: std::collections::BTreeMap<String, f64> = Default::default();
    for b in s.beams.iter().filter(|b| b.a != b.b && !b.broken) {
        let dr = (b.rest - b.rest0) as f64;
        let fy = if dr < 0.0 { b.tc } else { b.tt } as f64;
        if !fy.is_finite() || dr == 0.0 {
            continue;
        }
        let (mut a, mut c) = (stem(b.a), stem(b.b));
        if a > c {
            std::mem::swap(&mut a, &mut c);
        }
        *by.entry(format!("{a}-{c}")).or_default() += fy * dr.abs();
    }
    let mut v: Vec<(String, f64)> = by.into_iter().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    let tot: f64 = v.iter().map(|x| x.1).sum();
    format!(
        "{:.1} kJ: {}",
        tot / 1e3,
        v.iter()
            .take(top)
            .map(|(k, e)| format!("{k} {:.2}", e / 1e3))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// The oblique pole's impact reference line aft of the front axle, as a
/// share of the wheelbase. NHTSA's 2018 Honda Accord pole tests (TP-214P,
/// reports v10841 and v10842): 1,303 and 1,313 mm of a 2,819-2,820 mm
/// wheelbase, 0.10 m behind the test weight's centre of mass. (A 2020
/// Veloster: 0.50; a 2024 Nautilus: 0.45.) Round 5's harness used the
/// generic head point, 0.64 of the wheelbase: 0.5 m behind gen3's centre of
/// mass, so the 75° impulse spun the car about the pole.
pub const POLE_LINE: f32 = 0.464;
/// The carriage's friction under the tyres: the car rides the test's
/// platform on low-friction pads (PTFE on steel, μ 0.04-0.10), towed into
/// the pole with its wheels not turning.
pub const CARRIAGE_MU: f32 = 0.1;

/// FMVSS 214's oblique pole (32 km/h, 75°, 254 mm) at the driver's head:
/// residual and dynamic crush at the pole, intrusion, damage, visual.
pub fn scene_pole(car: &Car, kmh: f32, seconds: f32) -> Value {
    scene_pole_at(car, kmh, seconds, 0.0)
}

/// [`scene_pole`] with the pole moved `dz` m along the car (+ forward): a
/// pole a few centimetres either way meets other nodes, which shows how much
/// of a residual is the lattice's place against the pole.
pub fn scene_pole_at(car: &Car, kmh: f32, seconds: f32, dz: f32) -> Value {
    let t0 = Instant::now();
    let f = car.frame;
    let a = 75.0f32.to_radians();
    let dir = Vec3::new(a.sin(), 0.0, a.cos());
    let mut s = car.template.clone();
    let rest = positions(&s);
    let old = std::env::var("POLE_OLD").is_ok();
    let r = 0.127;
    let head = if old {
        occupant_points(&f)
            .iter()
            .find(|p| p.0 == "head")
            .unwrap()
            .1
    } else {
        // The impact reference line (the driver dummy's head CG), as the
        // NHTSA tests measured it on a 2018 Accord: [`POLE_LINE`] of the
        // wheelbase aft of the front axle.
        Vec3::new(0.37, 0.0, f.front_axle - POLE_LINE * f.wheelbase())
    };
    let head = Vec3::new(head.x, 0.0, head.z + dz);
    if old {
        set_moving(&mut s, dir * kmh / 3.6);
    } else {
        // On the carriage: every node at the test velocity, the wheels not
        // turning, the tyres on the carriage's low-friction pads
        // ([`CARRIAGE_MU`]).
        for n in &mut s.nodes {
            n.vel = if n.flags & NODE_IMMOVABLE != 0 {
                Vec3::ZERO
            } else {
                dir * kmh / 3.6
            };
            if n.flags & NODE_TYRE != 0 {
                n.friction = n.friction.min(CARRIAGE_MU);
            }
        }
    }
    let gap = |c: Vec3| {
        s.nodes
            .iter()
            .map(|n| Vec3::new(n.pos.x - c.x, 0.0, n.pos.z - c.z).length())
            .fold(f32::MAX, f32::min)
    };
    let mut t = 0.0f32;
    while gap(head + dir * t) < r + 0.03 {
        t += 0.005;
    }
    let centre = head + dir * t + Vec3::Y * 2.0;
    s.shapes.push(StaticShape::cylinder(centre, r, 2.0));
    side_run(car, s, rest, Some(centre), None, seconds, t0)
}

/// The far half's frame and the struck side's crush (shared by the pole and
/// the MDB): `pole` the pole's centre (crush within ±0.15 m of its line),
/// else NHTSA's struck-side maximum over the barrier's height band.
fn side_run(
    car: &Car,
    s: ExplicitBeams,
    rest: Vec<Vec3>,
    pole: Option<Vec3>,
    mdb: Option<ExplicitBeams>,
    seconds: f32,
    t0: Instant,
) -> Value {
    let f = car.frame;
    let far: Vec<u32> = (0..s.nodes.len() as u32)
        .filter(|&i| !is_tyre(&s, i as usize) && rest[i as usize].x < -0.2 * f.width())
        .collect();
    let reach = car
        .skin
        .iter()
        .map(|&i| rest[i as usize].x)
        .fold(f32::MIN, f32::max);
    let struck: Vec<u32> = car
        .skin
        .iter()
        .copied()
        .filter(|&i| {
            let p = rest[i as usize];
            match pole {
                Some(c) => {
                    p.x > 0.3 * 0.5 * f.width()
                        && (p.z - c.z).abs() < 0.15
                        && (0.3..1.2).contains(&p.y)
                }
                None => p.x > reach - 0.15 && (0.279..=0.838).contains(&(p.y - f.floor.min(0.02))),
            }
        })
        .collect();
    let mut intr = Intrusion::new(car, rest.clone(), far.clone());
    let mut pen = Penetration::new(&s);
    let crush = |s: &ExplicitBeams| -> f32 {
        let now = positions(s);
        let fit = rigid_fit(&rest, &now, &far);
        struck
            .iter()
            .filter(|&&i| now[i as usize].is_finite())
            .map(|&i| rest[i as usize].x - fit.to_rest(now[i as usize]).x)
            .fold(0.0, f32::max)
    };
    let mut dyn_max = 0.0f32;
    let side_debug = std::env::var("SIDE_DEBUG").is_ok();
    let mut elastic_at_max = String::new();
    // The pole's deepest reach into the car: its axis in the far side's frame from the struck side's skin.
    let mut pole_reach = 0.0f32;
    let n = (seconds / STEP).round() as usize;
    let mut world = mdb.map(|m| {
        let mut w = ExplicitBeamsWorld::new(vec![s.clone(), m]);
        w.contact = true;
        w
    });
    let mut solo = if world.is_none() { Some(s) } else { None };
    let mut e_max = 0.0f64;
    let mech = |v: &ExplicitBeams| {
        v.nodes
            .iter()
            .map(|n| {
                f64::from(n.mass)
                    * (0.5 * f64::from(n.vel.length_squared()) + 9.81 * f64::from(n.pos.y))
            })
            .sum::<f64>()
    };
    let e0 = match (&world, &solo) {
        (Some(w), _) => w.vehicles.iter().map(mech).sum::<f64>(),
        (_, Some(s)) => mech(s),
        _ => 0.0,
    };
    let mut hw = HwTrack::new(
        car,
        match (&world, &solo) {
            (Some(w), _) => &w.vehicles[0],
            (_, Some(s)) => s,
            _ => unreachable!(),
        },
    );
    for k in 0..n {
        match (&mut world, &mut solo) {
            (Some(w), _) => w.step(),
            (_, Some(s)) => s.step(),
            _ => {}
        }
        hw.sample(
            match (&world, &solo) {
                (Some(w), _) => &w.vehicles[0],
                (_, Some(s)) => s,
                _ => unreachable!(),
            },
            k as f32 * STEP,
        );
        if k % 2 == 1 {
            let car_s = match (&world, &solo) {
                (Some(w), _) => &w.vehicles[0],
                (_, Some(s)) => s,
                _ => unreachable!(),
            };
            pen.sample(car_s);
        }
        if k % 10 == 9 {
            let (car_s, e) = match (&world, &solo) {
                (Some(w), _) => (&w.vehicles[0], w.vehicles.iter().map(mech).sum::<f64>()),
                (_, Some(s)) => (s, mech(s)),
                _ => unreachable!(),
            };
            e_max = e_max.max(e - e0);
            let c_now = crush(car_s);
            if side_debug && c_now > dyn_max {
                // The elastic energy at the deepest crush: what springs back.
                let mut v: Vec<(f32, usize)> = car_s
                    .beams
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| {
                        !b.broken
                            && b.a != b.b
                            && b.k > 0.0
                            && !is_tyre(car_s, b.a as usize)
                            && !is_tyre(car_s, b.b as usize)
                    })
                    .map(|(j, b)| {
                        let l = (car_s.nodes[b.a as usize].pos - car_s.nodes[b.b as usize].pos)
                            .length();
                        let e = l - b.rest;
                        // Supports carry no tension.
                        let e = if b.kind == ExplicitKind::JbeamSupport {
                            e.min(0.0)
                        } else {
                            e
                        };
                        (0.5 * b.k * e * e, j)
                    })
                    .collect();
                v.sort_by(|a, b| b.0.total_cmp(&a.0));
                let total: f32 = v.iter().map(|x| x.0).sum();
                elastic_at_max = format!(
                    "elastic {:.0} J at {:.0} mm; top: {}",
                    total,
                    c_now * 1e3,
                    v.iter()
                        .take(14)
                        .map(|&(e, j)| {
                            let b = &car_s.beams[j];
                            format!(
                                "{}-{} {:.0} J",
                                car.node_name(b.a as usize),
                                car.node_name(b.b as usize),
                                e
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            dyn_max = dyn_max.max(c_now);
            if let Some(c) = pole {
                let now = positions(car_s);
                let fit = rigid_fit(&rest, &now, &far);
                pole_reach = pole_reach.max(reach - (fit.to_rest(c).x - 0.127));
                // `POLE_TRACE`: every 50 ms the pole in the far side's frame,
                // the car's yaw and its centre of mass's velocity.
                if k % 100 == 99 && std::env::var("POLE_TRACE").is_ok() {
                    let q = fit.to_rest(c);
                    let fwd = fit.r.transpose() * glam::DVec3::Z;
                    let m: f32 = car_s.nodes.iter().map(|n| n.mass).sum();
                    let v = car_s.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>() / m;
                    eprintln!(
                        "  pole {:.0} ms: pole at x {:.3} z {:.3} (car frame), yaw {:.1}°, v ({:.2}, {:.2}, {:.2}) m/s, crush {:.0} mm",
                        (k + 1) as f32 * STEP * 1e3,
                        q.x,
                        q.z,
                        fwd.x.atan2(fwd.z).to_degrees(),
                        v.x,
                        v.y,
                        v.z,
                        c_now * 1e3
                    );
                }
            }
            intr.sample(car_s);
        }
    }
    let s = match (&world, &solo) {
        (Some(w), _) => &w.vehicles[0],
        (_, Some(s)) => s,
        _ => unreachable!(),
    };
    hw.report(if pole.is_some() { "pole" } else { "side" }, s);
    // The pole's imprint: the car slides along an oblique pole as it crushes
    // (75°: a quarter of the speed is along the car), so the crush is spread
    // along the pole's path, not only on its line at first contact. The
    // pole's final place in the far side's frame (its shift along the car)
    // and the deepest residual crush within 0.15 m of its path, first
    // contact to rest (NHTSA's crush profile takes the damage where it is).
    let imprint = pole.map(|c| {
        let now = positions(s);
        let fit = rigid_fit(&rest, &now, &far);
        let cz = fit.to_rest(c).z;
        let r = car
            .skin
            .iter()
            .copied()
            .filter(|&i| {
                let p = rest[i as usize];
                p.x > 0.3 * 0.5 * f.width()
                    && p.z > cz.min(c.z) - 0.15
                    && p.z < cz.max(c.z) + 0.15
                    && (0.3..1.2).contains(&p.y)
            })
            .filter(|&i| now[i as usize].is_finite())
            .map(|i| rest[i as usize].x - fit.to_rest(now[i as usize]).x)
            .fold(0.0, f32::max);
        (cz - c.z, r)
    });
    // `SIDE_DEBUG`: the struck nodes with the most residual crush, and the
    // plastic work by member (node-name stems, an upper estimate).
    if std::env::var("SIDE_DEBUG").is_ok() {
        eprintln!("  side plastic {}", plastic_by_member(car, s, 16));
        let now = positions(s);
        let fit = rigid_fit(&rest, &now, &far);
        let mut v: Vec<(f32, u32)> = struck
            .iter()
            .filter(|&&i| now[i as usize].is_finite())
            .map(|&i| (rest[i as usize].x - fit.to_rest(now[i as usize]).x, i))
            .collect();
        v.sort_by(|a, b| b.0.total_cmp(&a.0));
        let top: Vec<String> = v
            .iter()
            .take(8)
            .map(|&(d, i)| {
                format!(
                    "{} {:.0} (y {:.2})",
                    car.node_name(i as usize),
                    d * 1e3,
                    rest[i as usize].y
                )
            })
            .collect();
        eprintln!(
            "  side {}: struck {} nodes; residual (mm) {}\n  side {} {}",
            if pole.is_some() { "pole" } else { "mdb" },
            struck.len(),
            top.join(", "),
            if pole.is_some() { "pole" } else { "mdb" },
            elastic_at_max
        );
    }
    let wall_s = t0.elapsed().as_secs_f32();
    let ke: f32 = s
        .nodes
        .iter()
        .map(|n| 0.5 * n.mass * n.vel.length_squared())
        .sum();
    json!({
        "barrier_min_x": world.as_ref().map(|w| r3(w.vehicles[1].nodes.iter().map(|n| n.pos.x).fold(f32::MAX, f32::min))), "barrier_vx": world.as_ref().map(|w| r3(w.vehicles[1].nodes.iter().map(|n| n.mass * n.vel.x).sum::<f32>() / w.vehicles[1].nodes.iter().map(|n| n.mass).sum::<f32>())), "car_vx": r3(s.nodes.iter().map(|n| n.mass * n.vel.x).sum::<f32>() / s.nodes.iter().map(|n| n.mass).sum::<f32>()), "pole_reach_mm": pole.map(|_| mm(pole_reach)), "residual_crush_mm": mm(crush(s)), "pole_slide_mm": imprint.map(|x| mm(x.0)), "residual_crush_imprint_mm": imprint.map(|x| mm(x.1)), "dynamic_crush_mm": mm(dyn_max), "struck_nodes": struck.len(),
        "energy_rise_j": r1(e_max as f32), "ke_end_kj": r3(ke / 1e3),
        "penetration": pen.result(), "intrusion": intr.result(), "damage": damage_report(car, s), "visual": visual_now(car, s),
        "sim_s": seconds, "wall_s_per_sim_s": r3(wall_s / seconds),
    })
}

/// A side-pole sweep: square-on (90°) at `kmh` into a 254 mm pole, its line
/// every `spacing` m along the car from bumper to bumper, `seconds` each.
/// Per offset: penetration into the pole (nodes, beams, triangles), the
/// pole's deepest reach into the car in the far side's frame, and whether it
/// passed through (the car keeps more than half its momentum along the pole's
/// line, or the pole's reach passes the far side's skin).
pub fn scene_pole_sweep(car: &Car, kmh: f32, spacing: f32, seconds: f32) -> Value {
    let t0 = Instant::now();
    let f = car.frame;
    let r = 0.127;
    let mut offsets = Vec::new();
    let mut z = f.lo.z + 0.1;
    while z <= f.hi.z - 0.1 + 1e-4 {
        offsets.push(z);
        z += spacing;
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out = std::sync::Mutex::new(vec![Value::Null; offsets.len()]);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(6));
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&zc) = offsets.get(k) else { break };
                let mut s = car.template.clone();
                let rest = positions(&s);
                let v = kmh / 3.6;
                set_moving(&mut s, Vec3::X * v);
                let reach = car.skin.iter().map(|&i| rest[i as usize].x).fold(f32::MIN, f32::max);
                let far_x = car.skin.iter().map(|&i| rest[i as usize].x).fold(f32::MAX, f32::min);
                let centre = Vec3::new(reach + r + 0.03, 2.0, zc);
                s.shapes.push(StaticShape::cylinder(centre, r, 2.0));
                let far: Vec<u32> = (0..s.nodes.len() as u32).filter(|&i| !is_tyre(&s, i as usize) && rest[i as usize].x < -0.2 * f.width()).collect();
                let p0: f32 = s.nodes.iter().map(|n| n.mass * n.vel.x).sum();
                let mut pen = Penetration::new(&s);
                let mut deepest = f32::MIN;
                let n = (seconds / STEP).round() as usize;
                for j in 0..n {
                    s.step();
                    if j % 2 == 1 {
                        pen.sample(&s);
                    }
                    if j % 20 == 19 {
                        // The pole's axis in the far side's rest frame: how far in it has come.
                        let now = positions(&s);
                        let fit = rigid_fit(&rest, &now, &far);
                        deepest = deepest.max(reach - (fit.to_rest(centre).x - r));
                    }
                }
                let p1: f32 = s.nodes.iter().map(|n| n.mass * n.vel.x).sum();
                let width = reach - far_x;
                let through = p1 > 0.5 * p0 || deepest > width;
                out.lock().unwrap()[k] = json!({
                    "z": r3(zc), "along": r3((zc - f.lo.z) / f.length()), "through": through,
                    "reach_mm": mm(deepest), "momentum_left": r3(p1 / p0), "penetration": pen.result(),
                    "finite": s.nodes.iter().all(|q| q.pos.is_finite()),
                });
            });
        }
    });
    let runs = out.into_inner().unwrap();
    let get = |v: &Value, k: &str| v["penetration"][k]["max_mm"].as_f64().unwrap_or(0.0);
    let worst = |k: &str| runs.iter().map(|v| get(v, k)).fold(0.0, f64::max);
    let through = runs
        .iter()
        .filter(|v| v["through"].as_bool() == Some(true))
        .count();
    let sim = offsets.len() as f32 * seconds;
    json!({
        "kmh": kmh, "spacing_m": spacing, "offsets": runs.len(), "passed_through": through,
        "worst_node_mm": worst("nodes"), "worst_beam_mm": worst("beams"), "worst_triangle_mm": worst("triangles"),
        "runs": runs, "wall_s_per_sim_s": r3(t0.elapsed().as_secs_f32() / sim),
    })
}

/// FMVSS 214's moving deformable barrier (1,367.6 kg, 53.9 km/h, crabbed 27°)
/// into the driver's side at the impact reference line, as the crash lab's
/// `mdb214` scene places it; two-body contact.
pub fn scene_mdb(car: &Car, seconds: f32) -> Value {
    use crate::crash_rigs::{mdb214_jbeam, MDB_CRAB_DEG, MDB_FACE, MDB_SPEED};
    let t0 = Instant::now();
    let f = car.frame;
    let text = mdb214_jbeam();
    let desc = match crate::jbeam::load_jbeam(&[("mdb214.jbeam", &text)], None) {
        Ok(d) => d,
        Err(e) => return json!({"error": e}),
    };
    let Ok((mut m, _)) = ExplicitBeams::from_desc_jbeam_with(
        &desc,
        JbeamWheels::Pressure,
        Vec3::ZERO,
        Vec3::ZERO,
        settings(),
    ) else {
        return json!({"error": "MDB spawn"});
    };
    let fwd = crate::pack::forward_axis(&desc);
    let yaw = -0.5 * std::f32::consts::PI;
    place(&mut m, yaw - fwd.x.atan2(fwd.z), 0.02);
    let v = MDB_SPEED;
    let crab = MDB_CRAB_DEG.to_radians();
    let wb = f.wheelbase();
    let irl = if wb <= 2.896 {
        0.5 * (f.front_axle + f.rear_axle) + 0.94
    } else {
        f.front_axle - 0.508
    };
    let contact = Vec3::new(f.hi.x, 0.0, irl - 0.5 * MDB_FACE[0]);
    let dir = Vec3::new(-crab.cos(), 0.0, -crab.sin());
    let gap = 0.3;
    // The barrier's face middle (its most forward skin along −X) at contact − dir·gap.
    let mskin = skin_nodes(&m);
    let (lo, hi) = bounds(&m, &mskin);
    let face = Vec3::new(lo.x, 0.0, 0.5 * (lo.z + hi.z));
    let target = contact - dir * gap;
    for n in &mut m.nodes {
        n.pos.x += target.x - face.x;
        n.pos.z += target.z - face.z;
    }
    set_moving(&mut m, dir * v);
    let s = car.template.clone();
    let rest = positions(&s);
    side_run(car, s, rest, None, Some(m), seconds, t0)
}

/// A quasi-static ram ([`crate::crash_rigs::ram_rig`]): FMVSS 214's static door
/// crush (305 mm cylinder, far side held, 457 mm) or FMVSS 216a's roof
/// (platen 762 × 1,829 mm at 5° pitch and 25° roll, sills held, 127 mm).
pub fn scene_door(car: &Car) -> Value {
    let Some((name, door)) = &car.door else {
        return json!({"available": false});
    };
    let t0 = Instant::now();
    let mut s = car.template.clone();
    let lo = door
        .iter()
        .map(|&i| s.nodes[i as usize].pos.y)
        .fold(f32::MAX, f32::min);
    let (z0, z1) = door.iter().fold((f32::MAX, f32::MIN), |(a, c), &i| {
        (
            a.min(s.nodes[i as usize].pos.z),
            c.max(s.nodes[i as usize].pos.z),
        )
    });
    let zc = 0.5 * (z0 + z1);
    let outer = door
        .iter()
        .map(|&i| s.nodes[i as usize].pos.x)
        .fold(f32::MIN, f32::max);
    let r = 0.1525;
    let (y0, y1) = (lo + 0.127, lo + 0.80);
    let doorset: std::collections::HashSet<u32> = door.iter().copied().collect();
    let held: Vec<usize> = (0..s.nodes.len())
        .filter(|&i| !is_tyre(&s, i) && s.nodes[i].pos.x < -0.2 && !doorset.contains(&(i as u32)))
        .collect();
    let cx0 = outer + r + 0.01;
    let inside = move |p: Vec3, d: f32| -> Option<f32> {
        if p.y < y0 || p.y > y1 {
            return None;
        }
        let cx = cx0 - d;
        let dz = p.z - zc;
        if dz.abs() >= r {
            return None;
        }
        let face = cx - (r * r - dz * dz).sqrt();
        (p.x > face).then(|| p.x - face)
    };
    let curve = crate::crash_rigs::ram_rig(&mut s, &held, -Vec3::X, 0.457 + 0.05, 0.1, &inside);
    // Crush from the ram's first contact.
    let first = curve.iter().position(|c| c.1 > 100.0).unwrap_or(0);
    let d0 = curve.get(first).map_or(0.0, |c| c.0);
    let avg = |upto: f32| {
        let v: Vec<f32> = curve
            .iter()
            .filter(|c| c.0 - d0 <= upto && c.0 >= d0)
            .map(|c| c.1)
            .collect();
        v.iter().sum::<f32>() / v.len().max(1) as f32
    };
    let peak = curve
        .iter()
        .filter(|c| c.0 - d0 <= 0.457)
        .map(|c| c.1)
        .fold(0.0f32, f32::max);
    let weight = car.mass() * 9.81;
    let wall_s = t0.elapsed().as_secs_f32();
    json!({
        "door_part": name, "initial_kn": r1(avg(0.152) / 1e3), "intermediate_kn": r1(avg(0.305) / 1e3), "peak_kn": r1(peak / 1e3),
        "peak_required_kn": r1((2.0 * weight).min(31_138.0) / 1e3),
        "force_kn_10mm": curve.iter().step_by(2).map(|c| r1(c.1 / 1e3)).collect::<Vec<_>>(),
        "damage": damage_report(car, &s),
        "wall_s_per_sim_s": r3(wall_s / (0.507 / 0.1)),
    })
}

pub fn scene_roof(car: &Car) -> Value {
    let t0 = Instant::now();
    let f = car.frame;
    let mut s = car.template.clone();
    let skin: Vec<Vec3> = car.skin.iter().map(|&i| s.nodes[i as usize].pos).collect();
    let wb_mid = |p: &Vec3| p.z > f.rear_axle - 0.3 && p.z < f.front_axle - 0.35 * f.wheelbase();
    let top = skin
        .iter()
        .filter(|p| wb_mid(p))
        .map(|p| p.y)
        .fold(f32::MIN, f32::max);
    let roof: Vec<Vec3> = skin
        .iter()
        .copied()
        .filter(|p| wb_mid(p) && p.y > top - 0.12)
        .collect();
    if roof.is_empty() {
        return json!({"error": "no roof"});
    }
    let zmax = roof.iter().map(|p| p.z).fold(f32::MIN, f32::max);
    let xmax = roof.iter().map(|p| p.x).fold(f32::MIN, f32::max);
    let mean = |v: Vec<Vec3>| v.iter().sum::<Vec3>() / v.len().max(1) as f32;
    let header = mean(roof.iter().copied().filter(|p| p.z > zmax - 0.08).collect());
    let rail = mean(roof.iter().copied().filter(|p| p.x > xmax - 0.08).collect());
    let (pitch, roll) = (5.0f32.to_radians(), 25.0f32.to_radians());
    let push = Quat::from_rotation_x(pitch) * Quat::from_rotation_z(-roll) * -Vec3::Y;
    let along = (Quat::from_rotation_x(pitch) * Vec3::Z).normalize();
    let across = push.cross(along).normalize();
    let front = header.z + 0.254;
    let centre_z = front - 0.5 * 1.829;
    let under = move |p: Vec3| {
        let q = p - Vec3::new(rail.x, rail.y, centre_z);
        q.dot(across).abs() < 0.381 && q.dot(along).abs() < 0.9145
    };
    let start = s
        .nodes
        .iter()
        .filter(|n| under(n.pos))
        .map(|n| n.pos.dot(push))
        .fold(f32::MAX, f32::min);
    // The first nodes the platen meets (its contact, for the report).
    let mut first: Vec<(f32, usize)> = (0..s.nodes.len())
        .filter(|&i| under(s.nodes[i].pos))
        .map(|i| (s.nodes[i].pos.dot(push) - start, i))
        .collect();
    first.sort_by(|a, b| a.0.total_cmp(&b.0));
    let first: Vec<Value> = first
        .iter()
        .take(6)
        .map(|&(d, i)| {
            json!([
                car.node_name(i),
                mm(d),
                r3(s.nodes[i].mass),
                [
                    r3(s.nodes[i].pos.x),
                    r3(s.nodes[i].pos.y),
                    r3(s.nodes[i].pos.z)
                ]
            ])
        })
        .collect();
    let held: Vec<usize> = (0..s.nodes.len())
        .filter(|&i| {
            let p = s.nodes[i].pos;
            !is_tyre(&s, i)
                && p.y < f.floor + 0.15
                && p.x.abs() > 0.5 * f.width() - 0.25
                && p.z > f.rear_axle + 0.3
                && p.z < f.front_axle - 0.3
        })
        .collect();
    let inside = move |p: Vec3, d: f32| -> Option<f32> {
        if !under(p) {
            return None;
        }
        let depth = start + d - p.dot(push);
        (depth > 0.0).then_some(depth)
    };
    let curve = crate::crash_rigs::ram_rig(&mut s, &held, push, 0.127, 0.1, &inside);
    let peak = curve.iter().map(|c| c.1).fold(0.0f32, f32::max);
    let weight = car.mass() * 9.81;
    let wall_s = t0.elapsed().as_secs_f32();
    json!({
        "peak_kn": r1(peak / 1e3), "ratio": r3(peak / weight), "held_nodes": held.len(), "first_contact": first, "roof_top_m": r3(top), "header": [r3(header.x), r3(header.y), r3(header.z)], "axles": [r3(f.front_axle), r3(f.rear_axle)], "rail": [r3(rail.x), r3(rail.y), r3(rail.z)],
        "force_kn_10mm": curve.iter().step_by(2).map(|c| r1(c.1 / 1e3)).collect::<Vec<_>>(),
        "damage": damage_report(car, &s),
        "wall_s_per_sim_s": r3(wall_s / (0.127 / 0.1)),
    })
}

/// Every scene for one car (each scene on its own thread).
/// Parked on its pressure wheels (the crash template, settled 1 s) for
/// `seconds` more with the parking brake input `parking` (spec §25.1): the
/// kinetic energy at the end, the share in the car's mean motion, the centre
/// of mass's travel and broken and bent beams.
pub fn scene_rest_pressure(car: &Car, seconds: f32, parking: f32) -> Value {
    let mut s = car.template.clone();
    s.parking_brake = parking;
    let total: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let com = |s: &ExplicitBeams| s.nodes.iter().map(|n| n.pos * n.mass).sum::<Vec3>() / total;
    let c0 = com(&s);
    steps(&mut s, seconds);
    let (mut ke, mut p) = (0.0f32, Vec3::ZERO);
    for n in &s.nodes {
        ke += 0.5 * n.mass * n.vel.length_squared();
        p += n.mass * n.vel;
    }
    let v = p / total;
    json!({
        "brakes": s.brakes.len(),
        "ke_j": r3(ke), "ke_mean_motion_j": r3(0.5 * total * v.length_squared()),
        "com_travel_mm": mm((com(&s) - c0).length()),
        "broken": s.beams.iter().filter(|b| b.broken).count(),
        "bent": s.beams.iter().filter(|b| b.bent()).count(),
    })
}

pub fn run_car(car: &Car, quick: bool) -> Value {
    let rest_s = if quick { 2.0 } else { 10.0 };
    let jbeam = matches!(car.source, Source::Jbeam { .. });
    // `COMPARE_TIMES=1`: each scene's wall-clock time on stderr.
    let timed = |name: &'static str, f: &dyn Fn() -> Value| -> Value {
        let t = std::time::Instant::now();
        let v = f();
        if std::env::var("COMPARE_TIMES").is_ok() {
            eprintln!(
                "  {}: {name} in {:.1} s",
                car.spec.name,
                t.elapsed().as_secs_f32()
            );
        }
        v
    };
    std::thread::scope(|sc| {
        let rest = sc.spawn(|| timed("rest", &|| scene_rest(car, false, rest_s)));
        let rest_tx = jbeam
            .then(|| sc.spawn(|| timed("rest_truck_export", &|| scene_rest(car, true, rest_s))));
        let wall = sc.spawn(|| {
            timed("wall56", &|| {
                scene_wall(car, 56.0, if quick { 0.3 } else { 1.0 })
            })
        });
        let pole = sc.spawn(|| {
            timed("pole32", &|| {
                scene_pole(car, 32.0, if quick { 0.4 } else { 1.5 })
            })
        });
        let parked = jbeam.then(|| {
            sc.spawn(|| {
                timed("rest_pressure", &|| {
                    json!({
                        "applied": scene_rest_pressure(car, rest_s, 1.0),
                        "released": scene_rest_pressure(car, rest_s, 0.0),
                    })
                })
            })
        });
        let mdb = sc.spawn(|| timed("mdb214", &|| scene_mdb(car, if quick { 0.4 } else { 1.5 })));
        let roof = sc.spawn(|| timed("roof", &|| scene_roof(car)));
        let door = sc.spawn(|| timed("door", &|| scene_door(car)));
        let sweep = sc.spawn(|| {
            timed("pole_sweep32", &|| {
                scene_pole_sweep(
                    car,
                    32.0,
                    if quick { 1.0 } else { 0.2 },
                    if quick { 0.3 } else { 0.8 },
                )
            })
        });
        json!({
            "name": car.spec.name, "label": car.spec.label,
            "census": census(car),
            "rest": rest.join().unwrap(),
            "rest_truck_export": rest_tx.map(|h| h.join().unwrap()),
            "rest_pressure": parked.map(|h| h.join().unwrap()),
            "wall56": wall.join().unwrap(),
            "pole32": pole.join().unwrap(),
            "mdb214": mdb.join().unwrap(),
            "roof": roof.join().unwrap(),
            "door": door.join().unwrap(),
            "pole_sweep32": sweep.join().unwrap(),
            "warnings": car.warnings.len(),
        })
    })
}

// --- Seam evidence (round 5) -------------------------------------------------------

/// The shut-line pairing distance: two pieces' vertices within 1.5 cm at rest
/// (a production panel gap is 3–6 mm; the ETK's hood-to-fender vertices sit
/// 5–15 mm apart).
pub const SHUT_LINE: f32 = 0.015;

/// Boundary vertices of two different pieces and the nearest such vertex of
/// another piece within `cell` at rest (`visual_metrics`' seam pairs when
/// `cell` is 1 cm; every vertex instead of the boundary ones when not
/// `boundary_only`), and per vertex its piece. Wheels excluded.
pub fn rest_seam_pairs(
    v: &ShowcaseVisual,
    p0: &[Vec3],
    cell: f32,
    boundary_only: bool,
) -> (Vec<(usize, usize)>, Vec<u32>) {
    let nv = v.positions.len();
    let mut piece = vec![u32::MAX; nv];
    for (k, (_, first, count)) in v.pieces.iter().enumerate() {
        for i in *first..first + count {
            if (i as usize) < nv {
                piece[i as usize] = k as u32;
            }
        }
    }
    let wheel = |i: usize| v.wheel.get(i).is_some_and(|&w| w != u32::MAX);
    let tris: Vec<[usize; 3]> = v
        .indices
        .chunks_exact(3)
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
        .filter(|t| t.iter().all(|&i| !wheel(i) && p0[i].is_finite()))
        .collect();
    let key = |p: Vec3| {
        (
            (p.x * 2e4).round() as i64,
            (p.y * 2e4).round() as i64,
            (p.z * 2e4).round() as i64,
        )
    };
    let mut weld: HashMap<(i64, i64, i64, u32), usize> = HashMap::new();
    let wid: Vec<usize> = (0..nv)
        .map(|i| {
            let (x, y, z) = key(p0[i]);
            let l = weld.len();
            *weld.entry((x, y, z, piece[i])).or_insert(l)
        })
        .collect();
    let mut wedges: HashMap<(usize, usize), usize> = HashMap::new();
    for t in &tris {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (wa, wb) = (wid[a], wid[b]);
            if wa != wb {
                *wedges.entry((wa.min(wb), wa.max(wb))).or_default() += 1;
            }
        }
    }
    let mut wb = std::collections::HashSet::new();
    for (&(a, b), &n) in &wedges {
        if n == 1 {
            wb.insert(a);
            wb.insert(b);
        }
    }
    let bverts: Vec<usize> = (0..nv)
        .filter(|&i| !wheel(i) && p0[i].is_finite() && (!boundary_only || wb.contains(&wid[i])))
        .collect();
    let ck = |p: Vec3| {
        (
            (p.x / cell).floor() as i32,
            (p.y / cell).floor() as i32,
            (p.z / cell).floor() as i32,
        )
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for &i in &bverts {
        grid.entry(ck(p0[i])).or_default().push(i);
    }
    let mut pairs = Vec::new();
    for &i in &bverts {
        let (x, y, z) = ck(p0[i]);
        let mut best: Option<(f32, usize)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &j in grid
                        .get(&(x + dx, y + dy, z + dz))
                        .map(|v| v.as_slice())
                        .unwrap_or(&[])
                    {
                        if piece[j] == piece[i] {
                            continue;
                        }
                        let d = p0[i].distance(p0[j]);
                        if d < cell && best.is_none_or(|b| d < b.0) {
                            best = Some((d, j));
                        }
                    }
                }
            }
        }
        if let Some((_, j)) = best {
            pairs.push((i.min(j), i.max(j)));
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    (pairs, piece)
}

/// A render vertex's skinning nodes: its weighted nodes (a flexskin) or its
/// binding triangle (a flexbody).
fn vertex_nodes(v: &ShowcaseVisual, i: usize) -> Vec<u32> {
    if v.wt_start.len() == v.positions.len() + 1 {
        (v.wt_start[i] as usize..v.wt_start[i + 1] as usize)
            .map(|k| v.wt_node[k])
            .collect()
    } else {
        v.bind_nodes[i].to_vec()
    }
}

/// The node a vertex follows most (its heaviest weight, or its binding
/// triangle's first node).
fn vertex_top(v: &ShowcaseVisual, i: usize) -> u32 {
    if v.wt_start.len() == v.positions.len() + 1 {
        let (a, b) = (v.wt_start[i] as usize, v.wt_start[i + 1] as usize);
        (a..b)
            .max_by(|&x, &y| v.wt_w[x][0].total_cmp(&v.wt_w[y][0]))
            .map_or(v.bind_nodes[i][0], |k| v.wt_node[k])
    } else {
        v.bind_nodes[i][0]
    }
}

/// Seam evidence for one car in the wall at `kmh` (`gen3_seam_evidence`):
/// per piece pair that shares a seam at rest (boundary vertices within
/// `cell`), the flexbody groups each piece binds to, whether the two sides'
/// vertices share skinning nodes, and over the crash the widest skin gap and
/// the widest separation of the two sides' nodes; per hood piece its largest
/// rise in the cabin's frame (the tent).
pub fn seam_evidence(car: &Car, kmh: f32, seconds: f32, cell: f32) -> Value {
    use std::collections::{BTreeMap, BTreeSet};
    let Some(v) = &car.visual else {
        return json!({"available": false});
    };
    let rest_nodes = positions(&car.template);
    let rest_beams: Vec<BeamState> = car
        .template
        .beams
        .iter()
        .map(|b| BeamState {
            broken: b.broken,
            rest: b.rest,
        })
        .collect();
    let rest_units = unit_states(v, &rest_beams);
    let (p0, _) = deform_beams(v, &rest_nodes, Some(&rest_units), Some(&rest_beams));
    let (pairs, piece) = rest_seam_pairs(v, &p0, cell, std::env::var("SEAM_ALL_VERTS").is_err());
    // Flexbody groups per mesh, and each group's nodes by name.
    let mut groups_of: HashMap<String, Vec<String>> = HashMap::new();
    let mut group_nodes: HashMap<String, Vec<String>> = HashMap::new();
    if let Source::Jbeam {
        files, pc, desc, ..
    } = &car.source
    {
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        if let Ok(rows) = crate::jbeam::assembled_rows(&refs, pc.as_deref(), "flexbodies") {
            for (_, row) in rows {
                let Some(mesh) = row.get("mesh").and_then(Value::as_str) else {
                    continue;
                };
                let g: Vec<String> = match row.get("[group]").or_else(|| row.get("group")) {
                    Some(Value::Array(a)) => a
                        .iter()
                        .filter_map(|g| g.as_str().map(str::to_string))
                        .collect(),
                    Some(Value::String(s)) => vec![s.clone()],
                    _ => Vec::new(),
                };
                groups_of.insert(mesh.to_string(), g);
            }
        }
        for (g, ns) in &desc.node_group_nodes {
            group_nodes.insert(
                g.clone(),
                ns.iter()
                    .map(|&i| desc.nodes[i as usize].name.clone())
                    .collect(),
            );
        }
    }
    let pname = |k: u32| {
        v.pieces
            .get(k as usize)
            .map_or("?".to_string(), |q| q.0.clone())
    };
    let mut by: BTreeMap<(u32, u32), Vec<(usize, usize)>> = Default::default();
    for &(i, j) in &pairs {
        let (a, b) = (piece[i], piece[j]);
        if a <= b {
            by.entry((a, b)).or_default().push((i, j));
        } else {
            by.entry((b, a)).or_default().push((j, i));
        }
    }
    let piece_nodes = |k: u32| -> BTreeSet<u32> {
        let (_, first, count) = &v.pieces[k as usize];
        (*first as usize..(*first + *count) as usize)
            .flat_map(|i| vertex_nodes(v, i))
            .collect()
    };
    #[derive(Default)]
    struct Acc {
        gap: f32,
        node_gap: f32,
        widest: (f32, u32, u32),
        gap_end: f32,
        node_gap_end: f32,
        open5_end: usize,
        /// The widest pair's two nodes' moves in the cabin's frame at the end (m).
        moves: (Vec3, Vec3),
    }
    let mut acc: Vec<Acc> = by.keys().map(|_| Acc::default()).collect();
    let hoods: Vec<u32> = (0..v.pieces.len() as u32)
        .filter(|&k| pname(k).to_ascii_lowercase().contains("hood"))
        .collect();
    let mut rise = vec![0.0f32; hoods.len()];
    let f = car.frame;
    let mut s = car.template.clone();
    set_moving(&mut s, Vec3::Z * kmh / 3.6);
    let front = car
        .skin
        .iter()
        .map(|&i| s.nodes[i as usize].pos.z)
        .fold(f32::MIN, f32::max);
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 2.0, front + 0.05 + 0.5),
        Vec3::new(0.5 * f.width() + 1.0, 2.0, 0.5),
    ));
    let cabin: Vec<u32> = (0..s.nodes.len() as u32)
        .filter(|&i| {
            !is_tyre(&s, i as usize)
                && (f.rear_axle - 0.1..f.rear_axle + 0.5 * f.wheelbase())
                    .contains(&rest_nodes[i as usize].z)
        })
        .collect();
    let mut w = ExplicitBeamsWorld::new(vec![s]);
    w.contact = false;
    let n_steps = (seconds / STEP).round() as usize;
    let every = (0.005 / STEP).round() as usize;
    let mut broken_hw: Vec<String> = Vec::new();
    for k in 1..=n_steps {
        w.step();
        if k % every != 0 && k != n_steps {
            continue;
        }
        let s = &w.vehicles[0];
        let now = positions(s);
        let beams: Vec<BeamState> = s
            .beams
            .iter()
            .map(|b| BeamState {
                broken: b.broken,
                rest: b.rest,
            })
            .collect();
        let units = unit_states(v, &beams);
        let (p1, _) = deform_beams(v, &now, Some(&units), Some(&beams));
        let fit = rigid_fit(&rest_nodes, &now, &cabin);
        let end = k == n_steps;
        for (q, ps) in by.values().enumerate() {
            let a = &mut acc[q];
            let (mut g_now, mut ng_now, mut o5) = (0.0f32, 0.0f32, 0usize);
            for &(i, j) in ps {
                if !p1[i].is_finite() || !p1[j].is_finite() {
                    continue;
                }
                let g = p1[i].distance(p1[j]) - p0[i].distance(p0[j]);
                let (na, nb) = (vertex_top(v, i), vertex_top(v, j));
                let ng = now[na as usize].distance(now[nb as usize])
                    - rest_nodes[na as usize].distance(rest_nodes[nb as usize]);
                g_now = g_now.max(g);
                ng_now = ng_now.max(ng);
                if g > 0.05 {
                    o5 += 1;
                }
                if ng > a.widest.0 {
                    a.widest = (ng, na, nb);
                }
            }
            a.gap = a.gap.max(g_now);
            a.node_gap = a.node_gap.max(ng_now);
            if end {
                let mv = |n: u32| fit.to_rest(now[n as usize]) - rest_nodes[n as usize];
                a.moves = (mv(a.widest.1), mv(a.widest.2));
                a.gap_end = g_now;
                a.node_gap_end = ng_now;
                a.open5_end = o5;
            }
        }
        for (h, &k) in hoods.iter().enumerate() {
            let (_, first, count) = &v.pieces[k as usize];
            for i in *first as usize..(*first + *count) as usize {
                if p1[i].is_finite() {
                    rise[h] = rise[h].max(fit.to_rest(p1[i]).y - p0[i].y);
                }
            }
        }
        if end {
            broken_hw = car
                .hardware
                .iter()
                .filter(|(_, ps)| {
                    ps.iter().any(|&(a, b)| {
                        s.incidence[a as usize].iter().any(|&j| {
                            let bm = &s.beams[j as usize];
                            bm.broken && (bm.a.min(bm.b), bm.a.max(bm.b)) == (a.min(b), a.max(b))
                        })
                    })
                })
                .map(|(n, _)| n.clone())
                .collect();
        }
    }
    let gset = |g: &[String]| -> BTreeSet<String> {
        g.iter()
            .flat_map(|x| group_nodes.get(x).cloned().unwrap_or_default())
            .collect()
    };
    let rows: Vec<Value> = by
        .iter()
        .zip(&acc)
        .map(|(((a, b), ps), r)| {
            let (na, nb) = (piece_nodes(*a), piece_nodes(*b));
            let shared_seam = ps
                .iter()
                .filter(|&&(i, j)| {
                    let x = vertex_nodes(v, i);
                    vertex_nodes(v, j).iter().any(|n| x.contains(n))
                })
                .count();
            let same_top = ps
                .iter()
                .filter(|&&(i, j)| vertex_top(v, i) == vertex_top(v, j))
                .count();
            let mesh = |k: u32| {
                let n = pname(k);
                n.split_once(':').map_or(n.clone(), |x| x.1.to_string())
            };
            let ga = groups_of.get(&mesh(*a)).cloned().unwrap_or_default();
            let gb = groups_of.get(&mesh(*b)).cloned().unwrap_or_default();
            json!({
                "a": pname(*a), "b": pname(*b), "pairs": ps.len(),
                "groups_a": ga, "groups_b": gb,
                "shared_group_nodes": gset(&ga).intersection(&gset(&gb)).count(),
                "nodes_a": na.len(), "nodes_b": nb.len(), "shared_nodes": na.intersection(&nb).count(),
                "seam_pairs_sharing_a_node": shared_seam, "seam_pairs_same_top_node": same_top,
                "skin_gap_max_mm": mm(r.gap), "node_gap_max_mm": mm(r.node_gap),
                "widest_node_pair": format!("{}-{}", car.node_name(r.widest.1 as usize), car.node_name(r.widest.2 as usize)),
                "widest_rest": [rest_nodes[r.widest.1 as usize].to_array().map(r3), rest_nodes[r.widest.2 as usize].to_array().map(r3)],
                "widest_moves_end_mm": [r.moves.0.to_array().map(mm), r.moves.1.to_array().map(mm)],
                "skin_gap_end_mm": mm(r.gap_end), "node_gap_end_mm": mm(r.node_gap_end), "open_5cm_end": r.open5_end,
            })
        })
        .collect();
    json!({
        "car": car.spec.name, "kmh": kmh, "cell_mm": mm(cell), "seam_pairs": pairs.len(),
        "pieces": v.pieces.len(),
        "hood_rise_mm": hoods.iter().zip(&rise).map(|(&k, &r)| json!({"piece": pname(k), "rise_mm": mm(r)})).collect::<Vec<_>>(),
        "hardware_broken": broken_hw,
        "pairs": rows,
    })
}
