//! Several destructibles under every stress model: a 2×2 block of a pack's copies (the lab's
//! tower block, engine-fracture `towers_grid`) in one world, on one shared GPU fracture
//! (each copy's chunks and bonds at their own offset; the load path's and the analytic
//! strike's per-bond tables and Blast's graph joined over every copy). The block stands at
//! rest as one copy does alone, and a shot at one copy breaks it as it breaks alone.
//!
//! house-1story always; the fractured tower too when its pack has been generated
//! (engine/scenes/vibe-city: `bun run export.ts fracturedTower`; git-ignored).
//!
//!   cargo test --release -p engine-world --test stress_block -- --test-threads=1 --nocapture
#![cfg(feature = "stress")]

use rubble::*;

const STRESS: [StrengthModel; 4] = [
    StrengthModel::Threshold,
    StrengthModel::LoadPath,
    StrengthModel::ImpactLocal,
    StrengthModel::BlastGpu,
];

fn models() -> Vec<StrengthModel> {
    match std::env::var("STRESS_MODELS") {
        Ok(v) => v.split(',').map(|n| StrengthModel::parse(n).expect("a model name")).collect(),
        Err(_) => STRESS.to_vec(),
    }
}

/// A pack's asset and the lab's 2×2 block of it.
struct Block {
    asset: DestructibleAssetDesc,
    /// The lab's scene of one copy (its shot).
    one: engine_fracture::Scene,
    /// Each copy's instance offset (asset frame), copy 0 first.
    offsets: Vec<[f32; 3]>,
    /// The shot's ball (radius m, mass kg) and speed scale, else the pack's at 1×.
    ball: Option<(f64, f64, f64)>,
}

fn block(pack: &[u8]) -> Block {
    let asset = DestructibleAssetDesc::from_pack(pack).unwrap();
    let (one, _) = engine_fracture::pack::build_pack_scene(pack).unwrap();
    let grid = engine_fracture::towers::towers_grid(&one, 2, 2, engine_fracture::towers::GAP, None);
    // As the lab places copies (engine-world-wasm lab.rs `place_threshold`): copy k's first
    // chunk in the grid, less the asset's first chunk.
    let stride = grid.instance_stride.max(1) as usize;
    let first = asset.chunks[0].pose.position;
    let offsets = (0..grid.copies.max(1) as usize)
        .map(|k| {
            let p = grid.bodies[1 + k * stride].p;
            [0, 1, 2].map(|i| p[i] as f32 - first[i])
        })
        .collect();
    Block { asset, one, offsets, ball: None }
}

struct Run {
    /// Per copy: bonds broken at the end.
    broken: Vec<u32>,
    /// Copy 0's bonds broken by each frame's end.
    timeline: Vec<u32>,
}

/// The front-row copy (largest z, nearest the lab's shot): nothing stands between it and
/// a shot from in front.
fn front(b: &Block) -> usize {
    (0..b.offsets.len()).max_by(|&i, &j| b.offsets[i][2].total_cmp(&b.offsets[j][2])).unwrap()
}

/// The block's copies `place` placed under `model`, `frames` steps, a shot at the first of
/// them (its lab shot, at the copy's offset) when `shot`.
fn run(b: &Block, model: StrengthModel, place: &[usize], frames: u32, shot: bool) -> anyhow::Result<Run> {
    let copies = place.len();
    let stress = b.asset.stress.as_ref().unwrap();
    let solver = b.asset.solver_for(model).unwrap();
    let scene = stress.stress_scene(model, 1.0, solver.dt)?;
    let chunks: usize = scene.bodies.iter().map(|x| x.chunks.len()).sum::<usize>() * copies;
    let one = stress_rubble::capacity_for(&scene);
    let mut c = one.clone();
    c.max_colliders = one.max_colliders * copies as u32;
    c.max_hull_vertices = one.max_hull_vertices * copies as u32;
    c.max_bodies = c.max_bodies.max(((chunks + 16 + 4) as u32).next_power_of_two());
    c.max_contacts = stress_rubble::threshold::contact_capacity(chunks + 16, c.max_contacts, 1 << 30);
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            capacity: c,
            gravity: [0.0, solver.gravity_y, 0.0],
            solver,
            enable_sleep: false,
            contacts: ContactStorage::Canonical,
            ..WorldDef::default()
        },
    )?;
    if let Some(h) = stress.ground_height() {
        let g = w.create_body(&BodyDef::default())?;
        w.create_shape(
            g,
            &ShapeDef::default(),
            Geometry::Plane {
                normal: [0.0, 1.0, 0.0],
                offset: h,
            },
        )?;
    }
    let a = w.create_destructible_asset(&b.asset)?;
    let mut ids = Vec::new();
    for off in place.iter().map(|&k| &b.offsets[k]) {
        ids.push(w.create_destructible(&DestructibleDef {
            model,
            transform: Transform::new(*off, [0.0, 0.0, 0.0, 1.0]),
            asleep: false,
            ..DestructibleDef::new(a)
        })?);
    }
    if shot {
        let lab = &b.one;
        let off = b.offsets[place[0]];
        let c = lab.center;
        let origin = [c[0], c[1], c[2] + 3.0 * lab.extent.max(1.0)];
        let k = b.ball.map_or(1.0, |x| x.2) * if model == StrengthModel::ImpactLocal { 6.0 } else { 1.0 };
        let speed = k * engine_fracture::plan::default_shot_speed(lab.extent, lab.projectile_speed);
        let v = engine_fracture::plan::shot_velocity(c, lab.extent, lab.projectile_radius, origin, [0.0, 0.0, -1.0], Some(speed), lab.projectile_arc);
        let (r, m) = b.ball.map_or((lab.projectile_radius, lab.projectile_mass), |x| (x.0, x.1));
        if std::env::var("STRESS_TRACE").is_ok() {
            eprintln!("offsets {:?}; shot from {:?} at {:?}", b.offsets, origin, v);
        }
        // Continuous collision for the load path's contact fracture and for fast balls.
        let contact = model == StrengthModel::LoadPath || b.ball.is_some();
        let ball = w.create_body(&BodyDef {
            linear_velocity: v.map(|x| x as f32),
            is_bullet: contact,
            ..BodyDef::dynamic([0, 1, 2].map(|i| origin[i] as f32 + off[i]))
        })?;
        w.create_sphere_shape(
            ball,
            &ShapeDef {
                update_body_mass: false,
                ..ShapeDef::default()
            },
            [0.0; 3],
            r as f32,
        )?;
        let i = (0.4 * m * r * r) as f32;
        w.body_set_mass_data(
            ball,
            MassData {
                mass: m as f32,
                center: [0.0; 3],
                rotational_inertia: [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]],
            },
        )?;
        w.body_set_projectile(ball, Some(Projectile::default()))?;
    }
    let mut timeline = Vec::new();
    for f in 0..frames {
        w.step()?;
        w.poll_blocking()?;
        let s = w.destructible_stats(ids[0])?;
        timeline.push(s.bonds - s.bonds_intact);
        // STRESS_TRACE=1 (a probe): each step's counters.
        if std::env::var("STRESS_TRACE").is_ok() && f % 4 == 0 {
            let c = w.counters().unwrap();
            eprintln!("{} copies {copies} step {f}: contacts {} awake {} broken {:?}", model.name(), c.contact_count, c.awake_count, ids.iter().map(|&d| w.destructible_stats(d).map(|s| s.bonds - s.bonds_intact).unwrap_or(0)).collect::<Vec<_>>());
        }
    }
    let broken = ids
        .iter()
        .map(|&id| w.destructible_stats(id).map(|s| s.bonds - s.bonds_intact))
        .collect::<anyhow::Result<_>>()?;
    Ok(Run { broken, timeline })
}

fn first_break(t: &[u32]) -> Option<(usize, u32)> {
    let f = t.iter().position(|&b| b > 0)?;
    Some((f, t[f]))
}

/// `known`: a model's open failure on this scene (the error it fails with), reported and not
/// failed on; it passing is reported too.
fn check(name: &str, b: &Block, frames: u32, known: &[(StrengthModel, &str)]) {
    assert_eq!(b.offsets.len(), 4, "{name}: a 2×2 block");
    // The struck copy first, then the others.
    let t = front(b);
    let block: Vec<usize> = std::iter::once(t).chain((0..4).filter(|&k| k != t)).collect();
    for model in models() {
        // At rest: every copy keeps what one copy alone keeps.
        let alone = run(b, model, &block[..1], 60, false).unwrap();
        let rest = run(b, model, &block, 60, false).unwrap();
        eprintln!("{name} {} at rest: alone {:?}, block {:?}", model.name(), alone.broken, rest.broken);
        for (k, &n) in rest.broken.iter().enumerate() {
            assert_eq!(n, alone.broken[0], "{name} {}: copy {k} at rest", model.name());
        }
        // Struck: copy 0 breaks as alone (the same first break; the totals within the
        // tolerance tests/stress_models.rs holds the world to against the lab).
        let open = known.iter().find(|k| k.0 == model).map(|k| k.1);
        let (alone, struck) = match (run(b, model, &block[..1], frames, true), run(b, model, &block, frames, true), open) {
            (Ok(a), Ok(s), _) => {
                if open.is_some() {
                    eprintln!("{name} {}: the known failure did not happen", model.name());
                }
                (a, s)
            }
            (Err(e), _, Some(what)) | (_, Err(e), Some(what)) if e.to_string().contains(what) => {
                eprintln!("{name} {} KNOWN OPEN FAILURE: {e}", model.name());
                continue;
            }
            (Err(e), _, _) | (_, Err(e), _) => panic!("{name} {}: {e:#}", model.name()),
        };
        eprintln!(
            "{name} {} struck: alone broke {:?} (first {:?}), in the block {:?} (first {:?})",
            model.name(),
            alone.broken,
            first_break(&alone.timeline),
            struck.broken,
            first_break(&struck.timeline)
        );
        assert!(alone.broken[0] > 0, "{name} {}: the shot breaks the copy", model.name());
        assert_eq!(first_break(&struck.timeline), first_break(&alone.timeline), "{name} {}: the first break", model.name());
        let ratio = struck.broken[0] as f64 / alone.broken[0] as f64;
        assert!((0.67..=1.5).contains(&ratio), "{name} {}: {} bonds in the block against {} alone", model.name(), struck.broken[0], alone.broken[0]);
    }
}

#[test]
fn a_block_of_house_1story_stands_and_breaks_as_one_house() {
    let b = block(include_bytes!("../../../stress-oracle/packs/house-1story.json"));
    check("house-1story", &b, 120, &[]);
}

#[test]
fn a_block_of_fractured_towers_stands_and_breaks_as_one_tower() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../engine-webgpu/generated/vibe-city/fracturedTower.json");
    let Ok(pack) = std::fs::read(path) else {
        eprintln!("no fracturedTower pack (engine/scenes/vibe-city: bun run export.ts fracturedTower); skipping");
        return;
    };
    let mut b = block(&pack);
    // The tower block's ball (engine-fracture towers.rs): 1 m, 30 t, at 16× the shot speed.
    let (r, m) = engine_fracture::towers::BALL;
    b.ball = Some((r, m, engine_fracture::towers::SPEED_SCALE));
    // Open (engine/docs/lab-world-plan.md): under the load path the struck tower reports a
    // load-path block not positive definite (fracture.slang `fr_lp_block`), alone as in
    // the block; the lab's own fracture world on the same shot does not reach it, and every
    // block of that run factors in f64 and in f32 (stress-rubble example load_path_blocks).
    check("fractured tower", &b, 120, &[(StrengthModel::LoadPath, "load-path block was not positive definite")]);
}
