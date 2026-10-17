//! Export Shardwright fracture assets as benchmark scenes (format v2), one per hierarchy level.
//!
//! `cargo run --release -p bench-fracture-export -- [--assets DIR] [--out DIR] [asset:level,...]`
//!
//! Geometry comes from rubble's own importer (`engine_fracture::build_scene`, the Rust port of
//! the WebGPU fracture demo's `geometry.mjs`), so every engine gets exactly the bodies and hull
//! vertices rubble's GPU kernels see: each fragment is a body at its centre of mass, turned to
//! its principal axes, with its baked convex hulls' vertices in that frame and the asset's
//! mass and principal inertia. The object stands on a ground plane at y = 0, centred in x/z.
//! Fragments start at rest at their baked poses, unbonded (bonds are inspection metadata in
//! rubble too).
//!
//! The scene adds one scripted wrecking ball (rubble's demo fires click-aimed shots instead):
//! radius 6% of the object's extent, half the object's mass, launched along −z from one extent
//! away at the mass-weighted height of the fragments on its line (an arch's crown, not its
//! opening), gravity-compensated to reach the centre after the 1 s lead-in, so the smash and
//! its debris fill the timed window.

use anyhow::{bail, Context, Result};
use engine_fracture::{build_scene, parse_fracture_glb};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const PROTOCOL: (u32, u32, u32, u32) = (2, 3, 5, 10); // rounds, warm steps, batches, per batch
const LEAD: u32 = 120; // 1 s at the importer's 1/120 s step
const STEPS: u32 = 360; // 3 s: the impact plus two seconds of debris
const FRICTION: f32 = 0.5; // rubble's fracture path is demo-wide 0.5 (asset materials unused)

/// (scene name, file) of the bundled assets.
const ASSETS: [(&str, &str); 2] = [("arch", "masonry-arch.glb"), ("house", "veneer-house.glb")];

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let bench = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut assets = bench.join("../engine/crates/engine-webgpu/fracture/assets");
    let mut out = bench.join("scenes/out");
    let mut only: Option<Vec<String>> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--assets" => assets = PathBuf::from(args.next().context("--assets DIR")?),
            "--out" => out = PathBuf::from(args.next().context("--out DIR")?),
            _ => only = Some(a.split(',').map(String::from).collect()),
        }
    }
    std::fs::create_dir_all(&out)?;
    let mut index = Vec::new();
    for (name, file) in ASSETS {
        let path = assets.join(file);
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let glb_sha = hex(&Sha256::digest(&bytes));
        let asset = parse_fracture_glb(&bytes).with_context(|| format!("parsing {}", path.display()))?;
        for level in asset.levels() {
            let id = format!("shardwright_{name}_l{level}");
            if only.as_ref().is_some_and(|o| !o.iter().any(|x| x == &id || x == name || x == &format!("{name}:{level}"))) {
                continue;
            }
            match export(&asset, level, &id, &path, &glb_sha, &out) {
                Ok(entry) => {
                    eprintln!("{id:24} {}", entry["summary"].as_str().unwrap_or(""));
                    index.push(entry);
                }
                Err(e) => {
                    eprintln!("{id:24} SKIPPED: {e:#}");
                    index.push(json!({ "id": id, "skipped": format!("{e:#}") }));
                }
            }
        }
    }
    std::fs::write(out.join("fracture-index.json"), serde_json::to_string_pretty(&json!({ "scenes": index }))? + "\n")?;
    eprintln!("wrote {}", out.display());
    Ok(())
}

fn export(asset: &engine_fracture::FractureAsset, level: u8, id: &str, glb: &Path, glb_sha: &str, out: &Path) -> Result<serde_json::Value> {
    let s = build_scene(asset, level)?;
    let fragments = 1..s.projectile_start as usize;
    if s.bodies[0].inv_mass != 0.0 || s.colliders[0].shape != 4 {
        bail!("importer layout changed: body 0 is not the static ground plane");
    }
    // Hull vertices per fragment, straight from the GPU hull arena (header, then vertices)
    let mut body_hulls: Vec<Vec<Vec<[f32; 3]>>> = vec![Vec::new(); s.bodies.len()];
    for c in s.colliders.iter().filter(|c| c.shape == 3) {
        let (base, nv) = (c.params[0] as usize, c.params[1] as usize);
        let verts = s.hull_data[base + 1..base + 1 + nv].iter().map(|v| [v[0], v[1], v[2]]).collect();
        body_hulls[c.body as usize].push(verts);
    }
    let total_mass: f64 = fragments.clone().map(|i| 1.0 / s.bodies[i].inv_mass as f64).sum();
    let ball_r = (0.06 * s.extent).clamp(0.1, 1.0);
    let ball_m = 0.5 * total_mass as f32;
    let g = s.params.gravity_y;
    let t = LEAD as f32 * s.params.dt;
    // Aim at material: the mass-weighted height of fragments near the ball's line (x ≈ 0),
    // e.g. an arch's crown rather than its opening
    let near: Vec<usize> = fragments.clone().filter(|&i| s.bodies[i].p[0].abs() < 0.15 * s.extent).collect();
    let near = if near.is_empty() { fragments.clone().collect() } else { near };
    let w: f64 = near.iter().map(|&i| 1.0 / s.bodies[i].inv_mass as f64).sum();
    let y0 = (near.iter().map(|&i| s.bodies[i].p[1] as f64 / s.bodies[i].inv_mass as f64).sum::<f64>() / w) as f32;
    let z0 = s.extent + ball_r;
    let ball_v = [0.0, -0.5 * g * t, -z0 / t];

    let mut words: Vec<u32> = Vec::new();
    let mut hull_table: Vec<u32> = Vec::new();
    let mut vertices: Vec<f32> = Vec::new();
    let mut body = |shape: u32, stat: bool, w2: [u32; 3], mass: f32, inertia: [f32; 3], p: [f32; 3], q: [f32; 4], v: [f32; 3]| {
        let f = |x: f32| x.to_bits();
        words.extend([shape, stat as u32]);
        words.extend(w2);
        words.push(f(mass));
        words.extend(inertia.map(f));
        words.push(f(FRICTION));
        words.extend(p.map(f));
        words.extend(q.map(f));
        words.extend(v.map(f));
        words.extend([0.0f32; 3].map(f));
        words.push(f(0.0)); // restitution
    };
    // Ground plane through the origin, normal +y (body 0, as in the importer)
    body(3, true, [0; 3], 0.0, [0.0; 3], [0.0; 3], [0.0, 0.0, 0.0, 1.0], [0.0; 3]);
    let (mut nh, mut compound, mut max_nv) = (0u32, 0usize, 0usize);
    for i in fragments.clone() {
        let b = &s.bodies[i];
        let hulls = &body_hulls[i];
        if hulls.is_empty() {
            bail!("fragment body {i} has no hull");
        }
        compound += (hulls.len() > 1) as usize;
        let mut radius = 0f32;
        for h in hulls {
            hull_table.extend([(vertices.len() / 3) as u32, h.len() as u32]);
            max_nv = max_nv.max(h.len());
            for v in h {
                radius = radius.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
                vertices.extend(v);
            }
        }
        body(2, false, [nh, hulls.len() as u32, radius.to_bits()], 1.0 / b.inv_mass, b.inv_i.map(|x| 1.0 / x), b.p, b.q, [0.0; 3]);
        nh += hulls.len() as u32;
    }
    let ball_i = 0.4 * ball_m * ball_r * ball_r;
    body(1, false, [ball_r.to_bits(), ball_r.to_bits(), ball_r.to_bits()], ball_m, [ball_i; 3], [0.0, y0, z0], [0.0, 0.0, 0.0, 1.0], ball_v);
    let nb = fragments.len() + 2;
    let ball = nb as u32 - 1;

    let mut bin: Vec<u32> = vec![0x4353_4252, 2, nb as u32, 0, nh, (vertices.len() / 3) as u32];
    bin.extend(&words);
    bin.extend(&hull_table);
    bin.extend(vertices.iter().map(|x| x.to_bits()));
    let bytes: Vec<u8> = bin.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sha = hex(&Sha256::digest(&bytes));

    let span = PROTOCOL.0 * (PROTOCOL.1 + PROTOCOL.2 * PROTOCOL.3);
    let speed = (ball_v[1] * ball_v[1] + ball_v[2] * ball_v[2]).sqrt();
    // 0.4 s: just before the ball reaches the object's front face (≈0.5 s)
    let pre_impact = LEAD * 2 / 5;
    let mut checkpoints = vec![0, pre_impact, LEAD, LEAD + span, STEPS];
    checkpoints.dedup();
    let sane = |at: u32| {
        vec![
            json!({ "at": at, "metric": "nonfinite", "max": 0, "why": "NaN/Inf pose or velocity" }),
            json!({ "at": at, "metric": "escaped", "max": 0, "why": "body fell through the ground plane or left the scene" }),
        ]
    };
    let mut gates = sane(STEPS);
    // Provisional envelopes until a reference run calibrates them (README "Fracture scenes")
    gates.push(json!({ "at": STEPS, "metric": "max_speed", "group": "dynamic", "max": (3.0 * speed).max(20.0), "why": "no fragment moves at 3× the ball's launch speed (energy injection)" }));
    gates.push(json!({ "at": STEPS, "metric": "pen_p99", "max": 0.03, "why": "hull contacts stay within 3 cm (provisional)" }));
    // No pre-impact "stands" gate: fragments are unbonded, so pieces only bonds held up (roofs,
    // overhangs) rightly fall; checkpoint `pre_impact` records how much each engine settles
    gates.push(json!({ "at": STEPS, "metric": "min_displacement", "group": "ball", "min": 0.25 * s.extent, "why": "the ball travelled into the object rather than stalling or tunnelling back" }));
    let meta = json!({
        "format": "rubble-bench-scene",
        "version": 2,
        "id": id,
        "tier": "fracture",
        "source": {
            "builder": "engine-fracture build_scene (rubble's Shardwright importer)",
            "asset": glb.file_name().unwrap().to_string_lossy(),
            "assetPath": glb.canonicalize().unwrap_or(glb.to_path_buf()).display().to_string(),
            "assetSha256": glb_sha,
            "level": level,
            "fragments": fragments.len(),
            "bonds": s.bond_count,
            "ball": { "radius": ball_r, "mass": ball_m, "start": [0.0, y0, z0], "velocity": ball_v },
            "note": "fragments start unbonded at their baked poses; one scripted ball replaces the demo's click-aimed shots",
        },
        "frame": "Y-up, right-handed (Shardwright meters, centred in x/z, standing on y = 0)",
        "units": "SI (m, kg, s, rad); quaternions (x, y, z, w)",
        "dt": s.params.dt,
        "gravity": [0.0, g, 0.0],
        "iterations": s.params.iterations,
        "avbd": { "alpha": s.params.alpha, "gamma": 0.999, "betaLin": 10000.0, "betaAng": 100.0 },
        "frictionCombine": "every body 0.5 (rubble's fracture path does not read Shardwright materials yet)",
        "protocol": { "lead": LEAD, "rounds": PROTOCOL.0, "warmSteps": PROTOCOL.1, "batches": PROTOCOL.2, "perBatch": PROTOCOL.3, "steps": STEPS, "checkpoints": checkpoints },
        "counts": { "bodies": nb, "dynamic": nb - 1, "static": 1, "spheres": 1, "boxes": 0, "joints": 0, "hulls": nh, "compoundBodies": compound, "maxHullVertices": max_nv },
        "bounds": { "extent": s.extent, "height": s.height },
        "bin": format!("{id}.scene.bin"),
        "binBytes": bytes.len(),
        "binSha256": sha,
        "groups": { "ball": [ball], "fragments": (1..ball).collect::<Vec<u32>>() },
        "gates": gates,
    });
    std::fs::write(out.join(format!("{id}.scene.bin")), &bytes)?;
    std::fs::write(out.join(format!("{id}.scene.json")), serde_json::to_string(&meta)? + "\n")?;
    Ok(json!({
        "id": id, "tier": "fracture", "bodies": nb, "hulls": nh, "compoundBodies": compound, "sha256": sha,
        "summary": format!("{} fragments, {nh} hulls ({compound} compound bodies, ≤{max_nv} vertices), extent {:.1} m, ball r {ball_r:.2} m {ball_m:.0} kg at {speed:.1} m/s", fragments.len(), s.extent),
    }))
}
