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
//! **Reactions.** A run turns on what it asks for (`RunSpec::crush`, `support_loss`,
//! `blast`: the destructible's `crush`, `support_loss` and `blast_loading`). Bonds broken
//! are counted by their `BreakCause` (blast and support-loss breaks apart from the rest),
//! crushes by their `CrushCause`. The harness features `reactions-crush`,
//! `reactions-blast` and `reactions-support` (engine-world's Cargo.toml) are kept for
//! scripts that pass them; every counter is on without them.
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

/// One run: a scene, an optional shot (an authored shot's label, e.g. `1.4x`), an
/// optional charge, the reactions turned on, and how long to run in total.
#[derive(Clone, Debug, Default)]
pub struct RunSpec {
    pub scene: String,
    pub shot: Option<String>,
    pub seconds: f64,
    /// A charge, detonated at its time (the pack's frame; `World::detonate`). Turns on
    /// the destructible's `blast_loading`.
    pub blast: Option<BlastSpec>,
    /// `DestructibleDef::crush` (the pack's own crush blocks, `CrushMass::Conserve`).
    pub crush: bool,
    /// `DestructibleDef::support_loss`.
    pub support_loss: bool,
    /// `DestructibleDef::blast_loading` without a charge (on with `blast`).
    pub blast_loading: bool,
}

/// A charge for [`RunSpec::blast`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlastSpec {
    /// The pack's frame (m) and TNT mass (kg).
    pub position: [f32; 3],
    pub tnt_kg: f32,
    /// When it goes off (s), after the settle as the shots launch.
    pub at: f64,
}

impl BlastSpec {
    /// `x,y,z,kg` (the example's `--blast`), at `at` seconds.
    pub fn parse(s: &str, at: f64) -> anyhow::Result<Self> {
        let v: Vec<f32> = s
            .split(',')
            .map(|x| x.trim().parse::<f32>())
            .collect::<Result<_, _>>()
            .map_err(|e| anyhow::anyhow!("--blast {s:?}: {e} (want x,y,z,kg)"))?;
        anyhow::ensure!(v.len() == 4, "--blast {s:?}: want x,y,z,kg");
        Ok(Self {
            position: [v[0], v[1], v[2]],
            tnt_kg: v[3],
            at,
        })
    }
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
    /// Bonds support loss broke (`BreakCause::SupportLoss`).
    pub support_loss_breaks: u32,
    pub crushed_chunks: u32,
    pub crushed_mass: f64,
    /// Crushes by cause: impact, blast, contact, load.
    pub crushed_by_cause: [u32; 4],
    /// Bonds blast loads broke (`BreakCause::Blast`).
    pub blast_breaks: u32,
    /// Bonds crushes broke (`BreakCause::Crush`).
    pub crush_breaks: u32,
    /// The lowest chunk centre (live chunks and crush fragments) above the ground (m);
    /// `None` without a ground.
    pub lowest_above_ground: Option<f64>,
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
    /// The charge, if any, and the frame it went off before.
    pub blast: Option<BlastSpec>,
    pub blast_frame: Option<u32>,
    /// The chunks crushed (in order) and their causes.
    pub crushed: Vec<(u32, CrushCause)>,
    /// Bonds broken per cause (the bond indices), over the run.
    pub broken_by_cause: Vec<(BreakCause, Vec<u32>)>,
    /// Where each chunk's centre starts and ends (the pack's frame); a crushed chunk's
    /// end is its fragments' mean (NaN: removed).
    pub chunk_centres_start: Vec<[f32; 3]>,
    pub chunk_centres_end: Vec<[f32; 3]>,
    pub ground: Option<f32>,
    /// The reactions turned on ("crush+support+blast", or "—").
    pub reactions: String,
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
    pub fn total_support_loss(&self) -> u32 {
        self.frames.iter().map(|f| f.support_loss_breaks).sum()
    }
    pub fn total_blast_breaks(&self) -> u32 {
        self.frames.iter().map(|f| f.blast_breaks).sum()
    }
    pub fn total_crushed_by(&self, cause: CrushCause) -> u32 {
        self.crushed.iter().filter(|c| c.1 == cause).count() as u32
    }
    /// The lowest chunk centre above the ground over the run.
    pub fn lowest_above_ground(&self) -> Option<f64> {
        self.frames
            .iter()
            .filter_map(|f| f.lowest_above_ground)
            .reduce(f64::min)
    }
}

fn cause_index(c: CrushCause) -> usize {
    match c {
        CrushCause::Impact => 0,
        CrushCause::Blast => 1,
        CrushCause::Contact => 2,
        CrushCause::Load => 3,
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

/// The crush state the harness tracks: each crushed chunk's fragments.
#[derive(Default)]
struct Crushed {
    chunks: Vec<bool>,
    fragments: Vec<(u32, Vec<BodyId>)>,
}

/// Sum of every distinct actor's mass (each live chunk belongs to exactly one actor),
/// plus every crush fragment's body mass (`CrushMass::Conserve`: they carry the crushed
/// chunks' mass).
fn structure_mass(
    w: &World,
    id: DestructibleId,
    chunks: u32,
    crushed: &Crushed,
) -> anyhow::Result<f64> {
    let mut seen = vec![false; chunks as usize];
    let mut mass = 0.0;
    for c in 0..chunks {
        if seen[c as usize] || crushed.chunks.get(c as usize).copied().unwrap_or(false) {
            continue;
        }
        let a = w.destructible_actor_of(id, c)?;
        for &k in &a.chunks {
            seen[k as usize] = true;
        }
        mass += a.mass as f64;
    }
    for (_, frs) in &crushed.fragments {
        for &b in frs {
            mass += w.body_mass_data(b)?.map_or(0.0, |m| m.mass as f64);
        }
    }
    Ok(mass)
}

/// Every chunk's centre now (the world's frame, the pack's): a live chunk's from its
/// body's transform, a crushed one's its fragments' mean (`None`: removed).
fn chunk_centres(
    w: &World,
    id: DestructibleId,
    asset: &DestructibleAssetDesc,
    crushed: &Crushed,
) -> anyhow::Result<Vec<Option<[f32; 3]>>> {
    let mut out = Vec::with_capacity(asset.chunks.len());
    for (c, ch) in asset.chunks.iter().enumerate() {
        if crushed.chunks.get(c).copied().unwrap_or(false) {
            let frs = &crushed
                .fragments
                .iter()
                .find(|f| f.0 == c as u32)
                .expect("a crushed chunk's record")
                .1;
            if frs.is_empty() {
                out.push(None);
                continue;
            }
            let mut sum = glam::Vec3::ZERO;
            for &b in frs {
                sum += glam::Vec3::from(w.read_body_state(b)?.transform.position);
            }
            out.push(Some((sum / frs.len() as f32).to_array()));
            continue;
        }
        let t = w.read_body_state(w.destructible_chunk_body(id, c as u32)?)?.transform;
        let q = glam::Quat::from_array(t.rotation);
        let p = glam::Vec3::from(t.position) + q * glam::Vec3::from(ch.pose.position);
        out.push(Some(p.to_array()));
    }
    Ok(out)
}

/// Count one event into `row`.
fn count(e: &DestructionEvent, row: &mut FrameRow) {
    match e {
        DestructionEvent::BondsBroken { bonds, cause, .. } => {
            let n = bonds.len() as u32;
            row.bonds_broken += n;
            match cause {
                BreakCause::Blast => row.blast_breaks += n,
                BreakCause::SupportLoss => row.support_loss_breaks += n,
                BreakCause::Crush => row.crush_breaks += n,
                BreakCause::Stress => {}
            }
        }
        DestructionEvent::Split { children, .. } => {
            row.splits += 1;
            row.pieces_created += children.len().saturating_sub(1) as u32;
        }
        DestructionEvent::ChunkCrushed { mass, cause, .. } => {
            row.crushed_chunks += 1;
            row.crushed_mass += *mass as f64;
            row.crushed_by_cause[cause_index(*cause)] += 1;
        }
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
        crush: spec.crush.then(Crush::default),
        support_loss: spec.support_loss,
        blast_loading: spec.blast_loading || spec.blast.is_some(),
        ..DestructibleDef::new(a)
    })?;
    let stats = w.destructible_stats(id)?;
    let dt = solver.dt as f64;
    let frames = (spec.seconds / dt).round() as u32;
    let launch_frame = shot.as_ref().map(|s| (s.launch_at / dt).round() as u32);
    let blast_frame = spec.blast.map(|b| (b.at / dt).round() as u32);
    let mut crushed = Crushed {
        chunks: vec![false; stats.chunks as usize],
        fragments: Vec::new(),
    };
    let mut crushed_list = Vec::new();
    let mut broken_by_cause: Vec<(BreakCause, Vec<u32>)> = Vec::new();
    let ground = stress.ground_height().map(|h| h + offset[1]);
    let structure_mass_start = structure_mass(&w, id, stats.chunks, &crushed)?;
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
        if Some(n) == blast_frame {
            let b = spec.blast.unwrap();
            w.detonate(&BlastDef {
                position: b.position,
                tnt_kg: b.tnt_kg,
                radial: None,
            })?;
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
                match e {
                    DestructionEvent::ChunkCrushed {
                        chunk,
                        cause,
                        fragments,
                        ..
                    } => {
                        crushed.chunks[*chunk as usize] = true;
                        crushed.fragments.push((*chunk, fragments.clone()));
                        crushed_list.push((*chunk, *cause));
                    }
                    DestructionEvent::BondsBroken { bonds, cause, .. } => {
                        match broken_by_cause.iter_mut().find(|c| c.0 == *cause) {
                            Some(c) => c.1.extend(bonds),
                            None => broken_by_cause.push((*cause, bonds.clone())),
                        }
                    }
                    _ => {}
                }
            }
        }
        row.actors = w.destructible_stats(id)?.actors;
        row.system_mass = structure_mass(&w, id, stats.chunks, &crushed)?;
        if let Some(g) = ground {
            let centres = chunk_centres(&w, id, &asset, &crushed)?;
            row.lowest_above_ground = centres
                .iter()
                .flatten()
                .map(|p| (p[1] - g) as f64)
                .reduce(f64::min);
        }
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
    let structure_mass_end = structure_mass(&w, id, stats.chunks, &crushed)?;
    let chunk_centres_end = chunk_centres(&w, id, &asset, &crushed)?
        .into_iter()
        .map(|p| p.unwrap_or([f32::NAN; 3]))
        .collect();
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
        blast: spec.blast,
        blast_frame,
        crushed: crushed_list,
        broken_by_cause,
        chunk_centres_start: asset
            .chunks
            .iter()
            .map(|c| {
                let p = c.pose.position;
                [p[0] - placement[0] as f32, p[1] - placement[1] as f32, p[2] - placement[2] as f32]
            })
            .collect(),
        chunk_centres_end,
        ground,
        reactions: {
            let on: Vec<&str> = [
                (spec.crush, "crush"),
                (spec.support_loss, "support"),
                (spec.blast_loading || spec.blast.is_some(), "blast"),
            ]
            .iter()
            .filter(|x| x.0)
            .map(|x| x.1)
            .collect();
            if on.is_empty() {
                "—".into()
            } else {
                on.join("+")
            }
        },
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
        .filter(|f| f.bonds_broken + f.splits + f.crushed_chunks > 0)
        .map(|f| {
            json!({"frame": f.frame, "time": f.time, "bondsBroken": f.bonds_broken, "splits": f.splits,
                   "piecesCreated": f.pieces_created, "actors": f.actors,
                   "supportLossBreaks": f.support_loss_breaks, "crushedChunks": f.crushed_chunks,
                   "crushedMass": f.crushed_mass, "crushedByCause": f.crushed_by_cause,
                   "blastBreaks": f.blast_breaks, "crushBreaks": f.crush_breaks,
                   "lowestAboveGround": f.lowest_above_ground,
                   "impactorSpeed": f.impactor_speed, "impactorKe": f.impactor_ke,
                   "systemMass": f.system_mass})
        })
        .collect();
    json!({
        "scene": r.scene, "title": r.title, "chunks": r.chunks, "bonds": r.bonds, "dt": r.dt,
        "frames": r.frames.len(), "shot": shot,
        "totals": {"bondsBroken": r.total_broken(), "splits": r.total_splits(),
                   "actorsEnd": r.actors_end(), "bondsIntactEnd": r.bonds_intact_end,
                   "supportLossBreaks": r.total_support_loss(), "blastBreaks": r.total_blast_breaks(),
                   "crushedChunks": r.total_crushed(), "crushedMass": r.total_crushed_mass(),
                   "crushedBlast": r.total_crushed_by(CrushCause::Blast),
                   "crushedImpact": r.total_crushed_by(CrushCause::Impact),
                   "crushedContact": r.total_crushed_by(CrushCause::Contact),
                   "crushedLoad": r.total_crushed_by(CrushCause::Load),
                   "lowestAboveGround": r.lowest_above_ground(),
                   "structureMassStart": r.structure_mass_start, "structureMassEnd": r.structure_mass_end},
        "blast": r.blast.map(|b| json!({"position": b.position, "tntKg": b.tnt_kg, "at": b.at})),
        "activeFrames": frames,
        "brokenByCause": r.broken_by_cause.iter().map(|(c, b)| json!({"cause": format!("{c:?}"), "bonds": b})).collect::<Vec<_>>(),
        "crushed": r.crushed.iter().map(|(c, k)| json!({"chunk": c, "cause": format!("{k:?}")})).collect::<Vec<_>>(),
        "chunkCentresEnd": r.chunk_centres_end,
    })
}

/// A markdown summary row per report.
pub fn to_markdown(reports: &[Report]) -> String {
    let mut s = String::from(
        "| Scene | Event | Reactions | Speed in → out m/s | KE in → out MJ | Bonds broken (blast / support loss / crush) | Splits | Pieces (actors) | Crushed: impact / blast / contact / load (kg) | Lowest above ground m | Structure mass start → end kg |\n|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    for r in reports {
        let (label, speed, ke) = match (&r.shot, &r.blast) {
            (Some(x), _) => (
                format!("{} ({:.0} kg)", x.label, x.mass),
                format!("{:.1} → {:.1}", x.speed_in, x.speed_out),
                format!("{:.2} → {:.2}", x.ke_in / 1e6, x.ke_out / 1e6),
            ),
            (None, Some(b)) => (
                format!(
                    "{} kg TNT at ({}, {}, {})",
                    b.tnt_kg, b.position[0], b.position[1], b.position[2]
                ),
                "—".into(),
                "—".into(),
            ),
            (None, None) => ("gravity".into(), "—".into(), "—".into()),
        };
        let crush_breaks: u32 = r.frames.iter().map(|f| f.crush_breaks).sum();
        let by = |c| r.total_crushed_by(c);
        s += &format!(
            "| {} | {} | {} | {} | {} | {} ({} / {} / {}) | {} | {} | {} / {} / {} / {} ({:.0}) | {} | {:.1} → {:.1} |\n",
            r.scene,
            label,
            r.reactions,
            speed,
            ke,
            r.total_broken(),
            r.total_blast_breaks(),
            r.total_support_loss(),
            crush_breaks,
            r.total_splits(),
            r.actors_end(),
            by(CrushCause::Impact),
            by(CrushCause::Blast),
            by(CrushCause::Contact),
            by(CrushCause::Load),
            r.total_crushed_mass(),
            r.lowest_above_ground()
                .map_or("—".into(), |h| format!("{h:.2}")),
            r.structure_mass_start,
            r.structure_mass_end,
        );
    }
    s
}
