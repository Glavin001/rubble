//! The destruction-reactions harness (`engine/docs/destruction-reactions.md`, "Tests and
//! oracle"): run a reference scene pack as a `StrengthModel::Threshold` destructible, fire
//! one of its authored shots (or none: gravity alone), and count every reaction per frame
//! from the world's events: bonds broken, splits, pieces, support-loss breaks, crushed
//! chunks and crushed mass, plus the impactor's speed and kinetic energy and the system's
//! total mass.
//!
//! Shared by `tests/reactions.rs` and `examples/destruction_reactions.rs` (`#[path]`), so
//! it stays out of the public API.
//!
//! **Hooks for the other reaction branches.** `ChunkCrushed` (feat/dr-crush) and blast
//! loads (feat/dr-blast) are not in `DestructionEvent` yet. Their counters exist and read
//! zero; the arms that fill them compile only with the harness features
//! `reactions-crush` / `reactions-blast` (engine-world's Cargo.toml) — turn them on, and
//! fix the arm if the merged event's shape differs, once those branches land. Support-loss
//! breaks (feat/dr-support) are `BondsBroken` today and cannot be told apart from impact
//! breaks; `support_loss_breaks` stays `None` until an event carries the cause
//! (`reactions-support`).
#![allow(dead_code)]

use rubble::*;

/// The reference rigs the harness knows, by pack key (rubble owns the copies:
/// `engine-webgpu/fracture/packs`, written by `port-reference-packs.mjs`).
pub const SCENES: &[(&str, &[u8])] = &[
    (
        "crush-wall",
        include_bytes!("../../../engine-webgpu/fracture/packs/crush-wall.json"),
    ),
    (
        "reference-building",
        include_bytes!("../../../engine-webgpu/fracture/packs/reference-building.json"),
    ),
    (
        "reference-building-crush",
        include_bytes!("../../../engine-webgpu/fracture/packs/reference-building-crush.json"),
    ),
];

pub fn pack_bytes(key: &str) -> anyhow::Result<&'static [u8]> {
    SCENES
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, b)| *b)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "unknown scene {key:?}: one of {:?}",
                SCENES.iter().map(|s| s.0).collect::<Vec<_>>()
            )
        })
}

/// One run: a scene, an optional shot (an authored shot's label, e.g. `1.4x`), and how
/// long to run in total.
#[derive(Clone, Debug)]
pub struct RunSpec {
    pub scene: String,
    pub shot: Option<String>,
    pub seconds: f64,
}

/// What one frame did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameRow {
    pub frame: u32,
    pub time: f64,
    pub bonds_broken: u32,
    pub splits: u32,
    /// New actors this frame (each split's children less the parent).
    pub pieces_created: u32,
    /// Actors after the frame.
    pub actors: u32,
    /// `None` until an event reports the cause (`reactions-support`).
    pub support_loss_breaks: Option<u32>,
    pub crushed_chunks: u32,
    pub crushed_mass: f64,
    /// Bonds blast loads broke (`reactions-blast`).
    pub blast_breaks: u32,
    /// The impactor, once launched.
    pub impactor_speed: Option<f64>,
    pub impactor_ke: Option<f64>,
    /// Every chunk's actor mass plus the impactor's.
    pub system_mass: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShotInfo {
    pub label: String,
    pub mass: f64,
    pub radius: f64,
    pub speed_in: f64,
    pub ke_in: f64,
    pub launch_frame: u32,
    /// After the last frame.
    pub speed_out: f64,
    pub ke_out: f64,
    pub min_speed: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub scene: String,
    pub title: String,
    pub chunks: u32,
    pub bonds: u32,
    pub dt: f64,
    pub frames: Vec<FrameRow>,
    pub shot: Option<ShotInfo>,
    pub structure_mass_start: f64,
    pub structure_mass_end: f64,
    pub bonds_intact_end: u32,
    /// Every destruction event in order, as `frame: event` (the determinism record).
    pub events: Vec<String>,
}

impl Report {
    pub fn total_broken(&self) -> u32 {
        self.frames.iter().map(|f| f.bonds_broken).sum()
    }
    pub fn total_splits(&self) -> u32 {
        self.frames.iter().map(|f| f.splits).sum()
    }
    pub fn total_crushed(&self) -> u32 {
        self.frames.iter().map(|f| f.crushed_chunks).sum()
    }
    pub fn total_crushed_mass(&self) -> f64 {
        self.frames.iter().map(|f| f.crushed_mass).sum()
    }
    pub fn actors_end(&self) -> u32 {
        self.frames.last().map_or(0, |f| f.actors)
    }
}

/// The capacity the lab's `ThresholdWorld::new` gives the scene, plus room for the shot.
fn capacity(asset: &DestructibleAssetDesc) -> anyhow::Result<WorldCapacity> {
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let scene = stress.threshold_scene(1.0, solver.dt)?;
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut c = stress_rubble::capacity_for(&scene);
    c.max_bodies = c
        .max_bodies
        .max(((chunks + 16 + 4) as u32).next_power_of_two());
    c.max_contacts = stress_rubble::threshold::contact_capacity(chunks, c.max_contacts, 1 << 30);
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    Ok(c)
}

fn len(v: [f32; 3]) -> f64 {
    glam::Vec3::from(v).length() as f64
}

/// Sum of every distinct actor's mass (each chunk belongs to exactly one actor).
fn structure_mass(w: &World, id: DestructibleId, chunks: u32) -> anyhow::Result<f64> {
    let mut seen = vec![false; chunks as usize];
    let mut mass = 0.0;
    for c in 0..chunks {
        if seen[c as usize] {
            continue;
        }
        let a = w.destructible_actor_of(id, c)?;
        for &k in &a.chunks {
            seen[k as usize] = true;
        }
        mass += a.mass as f64;
    }
    Ok(mass)
}

/// Count one event into `row`.
fn count(e: &DestructionEvent, row: &mut FrameRow) {
    match e {
        DestructionEvent::BondsBroken { bonds, .. } => row.bonds_broken += bonds.len() as u32,
        DestructionEvent::Split { children, .. } => {
            row.splits += 1;
            row.pieces_created += children.len().saturating_sub(1) as u32;
        }
        // TODO(feat/dr-crush): the charter's event (`destruction-reactions.md` section 1).
        #[cfg(feature = "reactions-crush")]
        DestructionEvent::ChunkCrushed { mass, .. } => {
            row.crushed_chunks += 1;
            row.crushed_mass += *mass as f64;
        }
        // TODO(feat/dr-blast): blast-load breaks into `row.blast_breaks` once an event
        // carries the cause (`reactions-blast`).
        // TODO(feat/dr-support): support-loss breaks into `row.support_loss_breaks`
        // (`reactions-support`).
        _ => {}
    }
}

/// Run `spec` on the default GPU.
pub fn run(spec: &RunSpec) -> anyhow::Result<Report> {
    let bytes = pack_bytes(&spec.scene)?;
    let asset = DestructibleAssetDesc::from_pack(bytes)?;
    let shots = engine_fracture::pack::pack_shots(bytes)?;
    let shot = match &spec.shot {
        None => None,
        Some(label) => Some(
            shots
                .iter()
                .find(|s| &s.label == label)
                .cloned()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "{}: no shot {label:?}; the pack authors {:?}",
                        spec.scene,
                        shots.iter().map(|s| &s.label).collect::<Vec<_>>()
                    )
                })?,
        ),
    };
    let solver = asset
        .solver_for(StrengthModel::Threshold)
        .ok_or_else(|| anyhow::anyhow!("no threshold solver"))?;
    let stress = asset.stress.as_ref().expect("stress source");
    let placement = stress.placement();
    // World frame = the pack's own frame: the asset less its placement.
    let offset = placement.map(|x| -x as f32);
    let def = WorldDef {
        capacity: capacity(&asset)?,
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: false,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def)?;
    if let Some(h) = stress.ground_height() {
        let g = w.create_body(&BodyDef::default())?;
        w.create_shape(
            g,
            &ShapeDef::default(),
            Geometry::Plane {
                normal: [0.0, 1.0, 0.0],
                offset: h + offset[1],
            },
        )?;
    }
    let a = w.create_destructible_asset(&asset)?;
    let id = w.create_destructible(&DestructibleDef {
        model: StrengthModel::Threshold,
        transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
        asleep: false,
        ..DestructibleDef::new(a)
    })?;
    let stats = w.destructible_stats(id)?;
    let dt = solver.dt as f64;
    let frames = (spec.seconds / dt).round() as u32;
    let launch_frame = shot.as_ref().map(|s| (s.launch_at / dt).round() as u32);
    let structure_mass_start = structure_mass(&w, id, stats.chunks)?;
    let mut ball: Option<BodyId> = None;
    let mut info: Option<ShotInfo> = None;
    let mut rows = Vec::with_capacity(frames as usize);
    let mut events = Vec::new();
    for n in 0..frames {
        if Some(n) == launch_frame {
            let s = shot.as_ref().unwrap();
            let f3 = |v: [f64; 3]| v.map(|x| x as f32);
            let b = w.create_body(&BodyDef {
                linear_velocity: f3(s.velocity),
                angular_velocity: f3(s.angular_velocity),
                ..BodyDef::dynamic(f3(s.origin))
            })?;
            w.create_sphere_shape(
                b,
                &ShapeDef {
                    update_body_mass: false,
                    ..ShapeDef::default()
                },
                [0.0; 3],
                s.radius as f32,
            )?;
            let i = (0.4 * s.mass * s.radius * s.radius) as f32;
            w.body_set_mass_data(
                b,
                MassData {
                    mass: s.mass as f32,
                    center: [0.0; 3],
                    rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
                },
            )?;
            w.body_set_projectile(b, Some(Projectile::default()))?;
            let speed = len(f3(s.velocity));
            info = Some(ShotInfo {
                label: s.label.clone(),
                mass: s.mass,
                radius: s.radius,
                speed_in: speed,
                ke_in: 0.5 * s.mass * speed * speed,
                launch_frame: n,
                speed_out: speed,
                ke_out: 0.5 * s.mass * speed * speed,
                min_speed: speed,
            });
            ball = Some(b);
        }
        w.step()?;
        let mut row = FrameRow {
            frame: n,
            time: (n + 1) as f64 * dt,
            ..FrameRow::default()
        };
        for f in w.poll_blocking()? {
            for e in &f.destruction_events {
                count(e, &mut row);
                events.push(format!("{n}: {e:?}"));
            }
        }
        row.actors = w.destructible_stats(id)?.actors;
        row.system_mass = structure_mass(&w, id, stats.chunks)?;
        if let (Some(b), Some(s)) = (ball, info.as_mut()) {
            let v = len(w.read_body_state(b)?.linear_velocity);
            row.impactor_speed = Some(v);
            row.impactor_ke = Some(0.5 * s.mass * v * v);
            row.system_mass += s.mass;
            s.speed_out = v;
            s.ke_out = 0.5 * s.mass * v * v;
            s.min_speed = s.min_speed.min(v);
        }
        rows.push(row);
    }
    let end = w.destructible_stats(id)?;
    let structure_mass_end = structure_mass(&w, id, stats.chunks)?;
    Ok(Report {
        scene: spec.scene.clone(),
        title: asset.name.clone().unwrap_or_default(),
        chunks: stats.chunks,
        bonds: stats.bonds,
        dt,
        frames: rows,
        shot: info,
        structure_mass_start,
        structure_mass_end,
        bonds_intact_end: end.bonds_intact,
        events,
    })
}

/// The report as JSON (summary plus the frames that did anything).
pub fn to_json(r: &Report) -> serde_json::Value {
    use serde_json::json;
    let shot = r.shot.as_ref().map(|s| {
        json!({"label": s.label, "mass": s.mass, "radius": s.radius, "launchFrame": s.launch_frame,
               "speedIn": s.speed_in, "speedOut": s.speed_out, "minSpeed": s.min_speed,
               "keIn": s.ke_in, "keOut": s.ke_out})
    });
    let frames: Vec<_> = r
        .frames
        .iter()
        .filter(|f| f.bonds_broken + f.splits + f.crushed_chunks + f.blast_breaks > 0)
        .map(|f| {
            json!({"frame": f.frame, "time": f.time, "bondsBroken": f.bonds_broken, "splits": f.splits,
                   "piecesCreated": f.pieces_created, "actors": f.actors,
                   "supportLossBreaks": f.support_loss_breaks, "crushedChunks": f.crushed_chunks,
                   "crushedMass": f.crushed_mass, "blastBreaks": f.blast_breaks,
                   "impactorSpeed": f.impactor_speed, "impactorKe": f.impactor_ke,
                   "systemMass": f.system_mass})
        })
        .collect();
    json!({
        "scene": r.scene, "title": r.title, "chunks": r.chunks, "bonds": r.bonds, "dt": r.dt,
        "frames": r.frames.len(), "shot": shot,
        "totals": {"bondsBroken": r.total_broken(), "splits": r.total_splits(),
                   "actorsEnd": r.actors_end(), "bondsIntactEnd": r.bonds_intact_end,
                   "supportLossBreaks": serde_json::Value::Null,
                   "crushedChunks": r.total_crushed(), "crushedMass": r.total_crushed_mass(),
                   "structureMassStart": r.structure_mass_start, "structureMassEnd": r.structure_mass_end},
        "hooks": {"crush": cfg!(feature = "reactions-crush"), "blast": cfg!(feature = "reactions-blast"),
                  "support": cfg!(feature = "reactions-support")},
        "activeFrames": frames,
    })
}

/// A markdown summary row per report.
pub fn to_markdown(reports: &[Report]) -> String {
    let mut s = String::from(
        "| Scene | Shot | Mass kg | Speed in → out m/s | KE in → out MJ | Bonds broken | Splits | Pieces (actors) | Crushed (kg) | Support-loss | Structure mass start → end kg |\n|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for r in reports {
        let (label, mass, speed, ke) = match &r.shot {
            Some(x) => (
                x.label.clone(),
                format!("{:.0}", x.mass),
                format!("{:.1} → {:.1}", x.speed_in, x.speed_out),
                format!("{:.2} → {:.2}", x.ke_in / 1e6, x.ke_out / 1e6),
            ),
            None => ("gravity".into(), "—".into(), "—".into(), "—".into()),
        };
        s += &format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} ({:.0}) | {} | {:.1} → {:.1} |\n",
            r.scene,
            label,
            mass,
            speed,
            ke,
            r.total_broken(),
            r.total_splits(),
            r.actors_end(),
            r.total_crushed(),
            r.total_crushed_mass(),
            if cfg!(feature = "reactions-support") {
                "?"
            } else {
                "n/a"
            },
            r.structure_mass_start,
            r.structure_mass_end,
        );
    }
    s
}
