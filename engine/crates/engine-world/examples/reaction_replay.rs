//! Run a destruction-reactions scenario in `World` and dump it for playback: every frame's
//! chunk and body poses, the chunk / fragment / ball shapes, and the events (`BondsBroken`
//! with the breaks support loss made, `Split`, `ChunkCrushed` with its pose and cause, the
//! blasts). `engine/tools/reaction-replay/` plays the JSON back in three.js and records it
//! (`record.mjs`), drawing dust at each crush (engine/docs/destruction-reactions.md).
//!
//! ```sh
//! cd engine
//! cargo run --release -p engine-world --example reaction_replay -- building-blast \
//!   [--seconds 6] [--tnt 25] [--crush conserve|remove|off] [--support-loss on|off] \
//!   [--factor 1.0] [--blast-loading on|off] [--out replay.json]
//! cargo run --release -p engine-world --example reaction_replay -- --list
//! ```
//!
//! Scenarios (each a reusable recipe; flags override its switches):
//! - `building-blast`: reference-building-crush, a charge on the ground floor at the bay's
//!   centre; blast loading, crushing (`Conserve`) and support loss on.
//! - `slab-columns`: a 100 t slab on four columns, one shot away (tests/support_loss.rs);
//!   support loss on (`--support-loss off` for the comparison).
//! - `block-columns`: a 40 t block on three columns, the outer one shot away: it tips.
//! - `crush-wall`: the punch-through wall's 1.4x shot, crushing with `Conserve` fragments.
//! - `pack:<scene>:<shot>`: any reference pack and authored shot (`destruction_reactions`
//!   `--list`), crushing as `--crush` says.

#[path = "../tests/support/frames.rs"]
mod frames;
#[path = "../tests/support/reactions.rs"]
mod reactions;

use rubble::*;
use serde_json::{json, Value};

/// A ball fired at `frame`.
struct ShotSpec {
    frame: u32,
    origin: [f32; 3],
    velocity: [f32; 3],
    angular_velocity: [f32; 3],
    radius: f32,
    mass: f32,
    bullet: bool,
}

struct Scenario {
    name: String,
    title: String,
    asset: DestructibleAssetDesc,
    seconds: f64,
    blast_loading: bool,
    support_loss: bool,
    factor: f32,
    crush: Option<Crush>,
    shots: Vec<ShotSpec>,
    blasts: Vec<(u32, BlastDef)>,
    /// A suggested camera: eye, target (world frame).
    camera: ([f32; 3], [f32; 3]),
    notes: Vec<String>,
}

fn crush_of(mode: &str) -> anyhow::Result<Option<Crush>> {
    Ok(match mode {
        "off" => None,
        "remove" => Some(Crush {
            mass: CrushMass::Remove,
            ..Crush::default()
        }),
        "conserve" => Some(Crush::default()),
        _ => anyhow::bail!("--crush is conserve, remove or off (got {mode})"),
    })
}

/// The pack's authored shot `label` as a ball (launched at its `launchAt`).
fn pack_shot(bytes: &[u8], label: &str, dt: f64) -> anyhow::Result<ShotSpec> {
    let shots = engine_fracture::pack::pack_shots(bytes)?;
    let s = shots.iter().find(|s| s.label == label).ok_or_else(|| {
        anyhow::anyhow!(
            "no shot {label:?}: {:?}",
            shots.iter().map(|s| &s.label).collect::<Vec<_>>()
        )
    })?;
    let f3 = |v: [f64; 3]| v.map(|x| x as f32);
    Ok(ShotSpec {
        frame: (s.launch_at / dt).round() as u32,
        origin: f3(s.origin),
        velocity: f3(s.velocity),
        angular_velocity: f3(s.angular_velocity),
        radius: s.radius as f32,
        mass: s.mass as f32,
        bullet: false,
    })
}

fn dt_of(asset: &DestructibleAssetDesc) -> f64 {
    asset.solver_for(StrengthModel::Threshold).unwrap().dt as f64
}

fn scenario(name: &str, tnt: f32) -> anyhow::Result<Scenario> {
    Ok(match name {
        "building-blast" => {
            let bytes = reactions::pack_bytes("reference-building-crush")?;
            let asset = DestructibleAssetDesc::from_pack(bytes)?;
            // The ground floor runs from the footings' tops (0.6 m) to the first slab
            // (2.1 m); the charge sits 1 m up at the 2x2 bay's centre.
            let at = [0.0, 1.0, 0.0];
            let dt = dt_of(&asset);
            Scenario {
                name: name.into(),
                title: format!("Reference building: {tnt} kg TNT on the ground floor"),
                asset,
                seconds: 6.0,
                blast_loading: true,
                support_loss: true,
                factor: 1.0,
                crush: Some(Crush::default()),
                shots: Vec::new(),
                blasts: vec![(
                    (1.0 / dt).round() as u32,
                    BlastDef {
                        position: at,
                        tnt_kg: tnt,
                        radial: None,
                    },
                )],
                camera: ([16.0, 8.0, 18.0], [0.0, 4.4, 0.0]),
                notes: vec![
                    "Blast loading, crushing (Conserve) and support loss on.".into(),
                    "Blast crushes need a charge at a crushable face: from inside, the drywall infill is in tension (tests/reactions.rs CHARGE).".into(),
                ],
            }
        }
        "slab-columns" | "block-columns" => {
            let f = if name == "slab-columns" {
                // tests/support_loss.rs `slab_row(1.2)`: the remaining three columns at 120%
                // of their capacity once the fourth is gone.
                let weight = 1.0e5 * frames::G;
                let compression = weight / 3.0 / 1.2 / frames::AREA;
                let columns = [[-3.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [3.0, 0.0]];
                frames::frame(&columns, &[0], 1.0e5, [3.6, 0.2, 0.6], compression)
            } else {
                let columns = [[-1.0, -1.0], [-1.0, 1.0], [1.6, 0.0]];
                frames::frame(&columns, &[2], 4.0e4, [2.0, 2.0, 1.4], 1.0e9)
            };
            let shot_column = if name == "slab-columns" { 0 } else { 2 };
            let asset = DestructibleAssetDesc::from_pack(&f.pack)?;
            let offset = asset
                .stress
                .as_ref()
                .unwrap()
                .placement()
                .map(|x| -x as f32);
            let p = asset.chunks[f.columns[shot_column] as usize].pose.position;
            // As the test fires it: from -z, 2.3 m off the column's middle, at 40 m/s.
            let at = [p[0] + offset[0], p[1] + offset[1], p[2] + offset[2] - 2.3];
            let dt = dt_of(&asset);
            Scenario {
                name: name.into(),
                title: if name == "slab-columns" {
                    "Support loss: a 100 t slab on four columns, one shot away".into()
                } else {
                    "Support loss: a 40 t block on three columns, the outer one shot away".into()
                },
                asset,
                seconds: 3.0,
                blast_loading: false,
                support_loss: true,
                factor: 1.0,
                crush: None,
                shots: vec![ShotSpec {
                    frame: (0.5 / dt).round() as u32,
                    origin: at,
                    velocity: [0.0, 0.0, 40.0],
                    angular_velocity: [0.0; 3],
                    radius: 0.3,
                    mass: 400.0,
                    bullet: true,
                }],
                blasts: Vec::new(),
                camera: if name == "slab-columns" {
                    ([-7.0, 4.5, -9.0], [0.0, 1.8, 0.0])
                } else {
                    ([-9.5, 6.5, -11.0], [0.0, 2.6, 0.0])
                },
                notes: vec![],
            }
        }
        "crush-wall" => pack_scenario("crush-wall", "1.4x", Some(Crush::default()))?,
        _ => {
            let rest = name
                .strip_prefix("pack:")
                .ok_or_else(|| anyhow::anyhow!("unknown scenario {name} (--list)"))?;
            let (scene, shot) = rest.split_once(':').unwrap_or((rest, "gravity"));
            pack_scenario(scene, shot, Some(Crush::default()))?
        }
    })
}

fn pack_scenario(scene: &str, shot: &str, crush: Option<Crush>) -> anyhow::Result<Scenario> {
    let bytes = reactions::pack_bytes(scene)?;
    let asset = DestructibleAssetDesc::from_pack(bytes)?;
    let dt = dt_of(&asset);
    let shots = if shot == "gravity" {
        Vec::new()
    } else {
        vec![pack_shot(bytes, shot, dt)?]
    };
    let camera = if scene == "crush-wall" {
        // From the shot's side, off its line: the ball, the hit face and the plug.
        ([-5.5, 3.4, -6.5], [0.6, 1.6, 0.2])
    } else {
        ([-11.0, 7.0, -12.0], [0.0, 4.0, 0.0])
    };
    Ok(Scenario {
        name: format!("{scene}_{shot}"),
        title: format!(
            "{} · {shot}",
            asset.name.clone().unwrap_or_else(|| scene.into())
        ),
        asset,
        seconds: 5.0,
        blast_loading: false,
        support_loss: false,
        factor: 1.0,
        crush,
        shots,
        blasts: Vec::new(),
        camera,
        notes: vec![],
    })
}

fn capacity(asset: &DestructibleAssetDesc, extra: u32) -> anyhow::Result<WorldCapacity> {
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let scene = stress.threshold_scene(1.0, solver.dt)?;
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut c = stress_rubble::capacity_for(&scene);
    c.max_bodies = c
        .max_bodies
        .max(((chunks as u32 + 16 + 4 + extra) as u32).next_power_of_two());
    c.max_contacts = stress_rubble::threshold::contact_capacity(chunks, c.max_contacts, 1 << 30);
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    Ok(c)
}

fn r4(x: f32) -> f64 {
    (x as f64 * 1e4).round() / 1e4
}

fn pose(t: &Transform) -> Value {
    json!([
        r4(t.position[0]),
        r4(t.position[1]),
        r4(t.position[2]),
        r4(t.rotation[0]),
        r4(t.rotation[1]),
        r4(t.rotation[2]),
        r4(t.rotation[3])
    ])
}

fn mul(a: &Transform, b: &Transform) -> Transform {
    let qa = glam::Quat::from_array(a.rotation);
    let p = qa * glam::Vec3::from(b.position) + glam::Vec3::from(a.position);
    let q = qa * glam::Quat::from_array(b.rotation);
    Transform::new(p.to_array(), q.normalize().to_array())
}

/// A geometry in the frame `base` (box, hull or sphere), for the player.
fn shape_json(g: &Geometry, base: &Transform) -> Option<Value> {
    Some(match g {
        Geometry::Box { half, transform } => {
            json!({"kind": "box", "half": half.map(r4), "t": pose(&mul(base, transform))})
        }
        Geometry::Hull { hull, transform } => {
            let v: Vec<f64> = hull
                .vertices
                .iter()
                .flat_map(|v| [r4(v.x), r4(v.y), r4(v.z)])
                .collect();
            json!({"kind": "hull", "vertices": v, "t": pose(&mul(base, transform))})
        }
        Geometry::Sphere { center, radius } => {
            json!({"kind": "sphere", "radius": r4(*radius), "t": pose(&mul(base, &Transform::new(*center, [0.0, 0.0, 0.0, 1.0])))})
        }
        _ => return None,
    })
}

fn cause(c: CrushCause) -> &'static str {
    match c {
        CrushCause::Impact => "impact",
        CrushCause::Blast => "blast",
        CrushCause::Contact => "contact",
        CrushCause::Load => "load",
    }
}

fn run(s: &Scenario) -> anyhow::Result<Value> {
    let asset = &s.asset;
    let solver = asset
        .solver_for(StrengthModel::Threshold)
        .ok_or_else(|| anyhow::anyhow!("no threshold solver"))?;
    let stress = asset.stress.as_ref().expect("stress source");
    let offset = stress.placement().map(|x| -x as f32);
    let budget = match s.crush.as_ref().map(|c| c.mass) {
        Some(CrushMass::Conserve { budget, .. }) => budget,
        _ => 0,
    };
    let def = WorldDef {
        capacity: capacity(asset, budget + s.shots.len() as u32)?,
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: false,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def)?;
    let ground = stress.ground_height().unwrap_or(0.0) + offset[1];
    let g = w.create_body(&BodyDef::default())?;
    w.create_shape(
        g,
        &ShapeDef::default(),
        Geometry::Plane {
            normal: [0.0, 1.0, 0.0],
            offset: ground,
        },
    )?;
    let a = w.create_destructible_asset(asset)?;
    let id = w.create_destructible(&DestructibleDef {
        model: StrengthModel::Threshold,
        transform: Transform::new(offset, [0.0, 0.0, 0.0, 1.0]),
        asleep: false,
        blast_loading: s.blast_loading,
        support_loss: s.support_loss,
        support_loss_dynamic_factor: s.factor,
        crush: s.crush.clone(),
        ..DestructibleDef::new(a)
    })?;
    let stats = w.destructible_stats(id)?;
    let n_chunks = stats.chunks as usize;
    let dt = solver.dt as f64;
    let frames = (s.seconds / dt).round() as u32;

    // Chunk shapes in the asset frame (chunk world = body transform * this).
    let chunks: Vec<Value> = asset
        .chunks
        .iter()
        .map(|c| {
            json!({
                "mass": r4(c.mass),
                "shapes": c.shapes.iter().filter_map(|g| shape_json(g, &c.pose)).collect::<Vec<_>>(),
            })
        })
        .collect();

    // Bodies other than chunks (balls, crush fragments): id, birth frame, shapes, kind.
    struct Extra {
        body: BodyId,
        born: u32,
        kind: &'static str,
        shapes: Vec<Value>,
    }
    let mut extras: Vec<Extra> = Vec::new();
    let mut crushed = vec![false; n_chunks];
    let mut out_frames = Vec::with_capacity(frames as usize);
    let mut events = Vec::new();
    let (mut total_broken, mut total_splits) = (0u32, 0u32);
    let mut by_cause = serde_json::Map::new();
    let mut crushed_mass = 0.0f64;
    let mut support_seen = 0usize;
    let mut support_bonds: Vec<u32> = Vec::new();
    for n in 0..frames {
        for sh in s.shots.iter().filter(|sh| sh.frame == n) {
            let b = w.create_body(&BodyDef {
                linear_velocity: sh.velocity,
                angular_velocity: sh.angular_velocity,
                is_bullet: sh.bullet,
                ..BodyDef::dynamic(sh.origin)
            })?;
            w.create_sphere_shape(
                b,
                &ShapeDef {
                    update_body_mass: false,
                    ..ShapeDef::default()
                },
                [0.0; 3],
                sh.radius,
            )?;
            let i = 0.4 * sh.mass * sh.radius * sh.radius;
            w.body_set_mass_data(
                b,
                MassData {
                    mass: sh.mass,
                    center: [0.0; 3],
                    rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
                },
            )?;
            w.body_set_projectile(b, Some(Projectile::default()))?;
            let sphere = Geometry::Sphere {
                center: [0.0; 3],
                radius: sh.radius,
            };
            extras.push(Extra {
                body: b,
                born: n,
                kind: "ball",
                shapes: shape_json(&sphere, &Transform::IDENTITY)
                    .into_iter()
                    .collect(),
            });
            events.push(
                json!({"frame": n, "type": "shot", "pos": sh.origin, "mass": sh.mass,
                               "speed": glam::Vec3::from(sh.velocity).length()}),
            );
        }
        for (_, b) in s.blasts.iter().filter(|(f, _)| *f == n) {
            w.detonate(b)?;
            events.push(json!({"frame": n, "type": "blast", "pos": b.position, "tnt": b.tnt_kg}));
        }
        w.step()?;
        let mut frame_broken = 0u32;
        for f in w.poll_blocking()? {
            for e in &f.destruction_events {
                match e {
                    DestructionEvent::BondsBroken { bonds, .. } => {
                        frame_broken += bonds.len() as u32;
                    }
                    DestructionEvent::Split { children, .. } => {
                        total_splits += 1;
                        events
                            .push(json!({"frame": n, "type": "split", "children": children.len()}));
                    }
                    DestructionEvent::ChunkCrushed {
                        chunk,
                        mass,
                        pose: p,
                        linear_velocity,
                        cause: c,
                        fragments,
                        peak_pressure,
                        ..
                    } => {
                        crushed[*chunk as usize] = true;
                        crushed_mass += *mass as f64;
                        let k = by_cause
                            .get(cause(*c))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        by_cause.insert(cause(*c).into(), json!(k + 1));
                        for &fb in fragments {
                            let mut shapes = Vec::new();
                            for sid in w.body_shapes(fb)? {
                                if let Some(j) =
                                    shape_json(w.shape_geometry(sid)?, &Transform::IDENTITY)
                                {
                                    shapes.push(j);
                                }
                            }
                            extras.push(Extra {
                                body: fb,
                                born: n,
                                kind: "fragment",
                                shapes,
                            });
                        }
                        events.push(json!({"frame": n, "type": "crushed", "chunk": chunk,
                            "pos": p.position.map(r4), "mass": r4(*mass), "cause": cause(*c),
                            "velocity": linear_velocity.map(r4), "fragments": fragments.len(),
                            "peakPressureMPa": r4(*peak_pressure / 1e6)}));
                    }
                    _ => {}
                }
            }
        }
        total_broken += frame_broken;
        let sl = w.destructible_support_loss_breaks(id)?;
        if sl.len() > support_seen {
            support_bonds.extend_from_slice(&sl[support_seen..]);
            events
                .push(json!({"frame": n, "type": "supportLoss", "bonds": sl.len() - support_seen}));
            support_seen = sl.len();
        }
        if frame_broken > 0 {
            events.push(json!({"frame": n, "type": "bonds", "count": frame_broken}));
        }
        // Poses: each chunk's body (null once crushed), each extra (null before birth or
        // once destroyed).
        let mut body_index: Vec<BodyId> = Vec::new();
        let mut cb = Vec::with_capacity(n_chunks);
        let mut statics = Vec::new();
        for c in 0..n_chunks as u32 {
            if crushed[c as usize] {
                cb.push(json!(null));
                continue;
            }
            let b = w.destructible_chunk_body(id, c)?;
            let k = match body_index.iter().position(|x| *x == b) {
                Some(k) => k,
                None => {
                    body_index.push(b);
                    if w.body_type(b)? == BodyType::Static {
                        statics.push(body_index.len() - 1);
                    }
                    body_index.len() - 1
                }
            };
            cb.push(json!(k));
        }
        let bodies: Vec<Value> = body_index
            .iter()
            .map(|&b| {
                w.body_transform(b)
                    .ok()
                    .flatten()
                    .map_or(json!(null), |t| pose(&t))
            })
            .collect();
        let ex: Vec<Value> = extras
            .iter()
            .map(|e| {
                if !w.body_is_valid(e.body) {
                    return json!(null);
                }
                w.body_transform(e.body)
                    .ok()
                    .flatten()
                    .map_or(json!(null), |t| pose(&t))
            })
            .collect();
        let st = w.destructible_stats(id)?;
        out_frames.push(json!({
            "b": bodies, "cb": cb, "s": statics, "x": ex,
            "hud": {"bondsBroken": total_broken, "splits": total_splits, "actors": st.actors,
                    "crushed": st.crushed, "crushedByCause": Value::Object(by_cause.clone()),
                    "crushedMass": crushed_mass.round(), "supportLossBreaks": support_seen},
        }));
    }
    let extras_json: Vec<Value> = extras
        .iter()
        .map(|e| json!({"born": e.born, "kind": e.kind, "shapes": e.shapes}))
        .collect();
    eprintln!(
        "{}: {frames} frames; {total_broken} bonds broken ({support_seen} by support loss), {total_splits} splits, {} crushed {} ({crushed_mass:.0} kg), {} fragments",
        s.name,
        w.destructible_stats(id)?.crushed,
        Value::Object(by_cause.clone()),
        extras.iter().filter(|e| e.kind == "fragment").count()
    );
    Ok(json!({
        "name": s.name, "title": s.title, "dt": dt, "frameCount": frames,
        "ground": ground, "camera": {"eye": s.camera.0, "target": s.camera.1},
        "switches": {"blastLoading": s.blast_loading, "supportLoss": s.support_loss,
                     "dynamicFactor": s.factor,
                     "crush": match s.crush.as_ref().map(|c| c.mass) {
                        None => "off".to_string(),
                        Some(CrushMass::Remove) => "remove".into(),
                        Some(CrushMass::Conserve { fragments, .. }) => format!("conserve ({fragments} fragments)"),
                     }},
        "notes": s.notes,
        "chunks": chunks, "extras": extras_json, "events": events, "frames": out_frames,
        "supportLossBonds": support_bonds,
    }))
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut name = None;
    let (mut seconds, mut tnt, mut out) = (None, 25.0f32, None);
    let (mut crush, mut support, mut blast, mut factor) = (None, None, None, None);
    let on = |v: Option<String>| -> anyhow::Result<bool> {
        match v.as_deref() {
            Some("on") => Ok(true),
            Some("off") => Ok(false),
            v => anyhow::bail!("expected on or off, got {v:?}"),
        }
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--seconds" => seconds = Some(args.next().unwrap_or_default().parse::<f64>()?),
            "--tnt" => tnt = args.next().unwrap_or_default().parse()?,
            "--out" => out = args.next(),
            "--crush" => crush = Some(crush_of(&args.next().unwrap_or_default())?),
            "--support-loss" => support = Some(on(args.next())?),
            "--blast-loading" => blast = Some(on(args.next())?),
            "--factor" => factor = Some(args.next().unwrap_or_default().parse::<f32>()?),
            "--list" => {
                println!(
                    "building-blast slab-columns block-columns crush-wall pack:<scene>:<shot>"
                );
                return Ok(());
            }
            _ => name = Some(a),
        }
    }
    let name = name.unwrap_or_else(|| "building-blast".into());
    let mut s = scenario(&name, tnt)?;
    if let Some(x) = seconds {
        s.seconds = x;
    }
    if let Some(c) = crush {
        s.crush = c;
    }
    if let Some(x) = support {
        s.support_loss = x;
    }
    if let Some(x) = blast {
        s.blast_loading = x;
    }
    if let Some(x) = factor {
        s.factor = x;
    }
    let t = std::time::Instant::now();
    let v = run(&s)?;
    let path = out.unwrap_or_else(|| format!("{}.json", s.name));
    std::fs::write(&path, serde_json::to_string(&v)?)?;
    eprintln!("wrote {path} in {:.1} s", t.elapsed().as_secs_f64());
    Ok(())
}
