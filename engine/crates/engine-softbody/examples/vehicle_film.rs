//! Films a rigged design on the CPU reference: runs a scene and writes, per
//! frame, every car's skinned render mesh (the `.rskin` weights evaluated on
//! the nodes, `authoring::skin::deform`) for `tools/vehicle-gen/render_film.py`
//! (Blender) to render. No GPU in the simulation; the clip shows what the
//! reference solver does, frame-locked.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example vehicle_film -- <scene> <out dir> [design]
//! ```
//! Scenes: `drive` (raycast wheels: away, then a turn), `wall` (56 km/h into a
//! rigid wall), `pole` (32 km/h sideways into a 254 mm pole at the front
//! seat), `c2c` (two cars head on, 50 km/h each). Crash scenes are sampled
//! every 4 ms (7.5× slow motion at 30 fps) and end with the car at rest.
//! Writes `mesh.bin` (vertex and triangle counts, colours, triangles),
//! `frame_NNNN.bin` (positions, f32, Rubble's frame: +Y up, +Z forward) and
//! `scene.json` (the static obstacles, the frame interval, the camera's hint).
use engine_softbody::authoring::rig;
use engine_softbody::authoring::skin::{deform, ShowcaseVisual};
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{ExplicitBeams, StaticBox, StaticShape};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::explicit_drive::DriveInput;
use engine_softbody::vehicle_compare::settings;
use glam::Vec3;
use serde_json::json;
use std::io::Write;
use std::path::Path;

struct Loaded {
    desc: SoftVehicleDesc,
    refs: Vec<(String, String)>,
    pc: Option<String>,
    built: rig::RiggedVehicle,
}

fn load(design: &str) -> Result<Loaded, String> {
    let built = rig::rig_vehicle(design, None, None)?;
    let files = rig::mod_files(&built);
    let refs: Vec<(String, String)> = files
        .iter()
        .filter(|(p, _)| p.ends_with(".jbeam"))
        .map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned()))
        .collect();
    let pc = files
        .iter()
        .find(|(p, _)| p.ends_with(".pc"))
        .map(|(_, b)| String::from_utf8_lossy(b).into_owned());
    let r: Vec<(&str, &str)> = refs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let desc = engine_softbody::jbeam::load_jbeam(&r, pc.as_deref())?;
    Ok(Loaded { desc, refs, pc, built })
}

/// The skinned visual of a car spawned from `desc` (its nodes' indices).
fn visual(l: &Loaded, desc: &SoftVehicleDesc, sim: &ExplicitBeams) -> Result<ShowcaseVisual, String> {
    let r: Vec<(&str, &str)> = l.refs.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let flex = engine_softbody::jbeam::assembled_rows(&r, l.pc.as_deref(), "flexbodies")?;
    let g = &l.built.input.glb;
    let meshes: Vec<_> = g.meshes.iter().map(|m| engine_softbody::flexskin::glb_mesh(g, m)).collect();
    let paint = engine_softbody::glb::paint_of(g).unwrap_or([0.5, 0.05, 0.04]);
    let (v, _) = engine_softbody::flexmesh::jbeam_visual_skinned(
        &flex,
        &[],
        desc,
        sim,
        &meshes,
        &Default::default(),
        paint,
        Some(&l.built.skin),
    )?;
    Ok(v)
}

fn spawn(desc: &SoftVehicleDesc, at: Vec3, vel: Vec3) -> Result<ExplicitBeams, String> {
    // The lowest point: a node, or a raycast wheel's tyre (it has no nodes).
    let low = desc.nodes.iter().map(|n| n.position[1]).fold(f32::MAX, f32::min);
    let low = desc.wheels.iter().fold(low, |m, w| {
        let y = |i: u32| desc.nodes[i as usize].position[1];
        m.min(0.5 * (y(w.node1) + y(w.node2)) - w.radius)
    });
    let lift = 0.02 - low;
    let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(
        desc,
        at + Vec3::Y * lift,
        vel,
        settings(),
    )?;
    s.surface_contact = true;
    s.finish();
    Ok(s)
}

fn turned(desc: &SoftVehicleDesc, quarter: i32) -> SoftVehicleDesc {
    let mut d = desc.clone();
    for n in &mut d.nodes {
        let [x, y, z] = n.position;
        n.position = match quarter.rem_euclid(4) {
            1 => [z, y, -x],
            2 => [-x, y, -z],
            3 => [-z, y, x],
            _ => [x, y, z],
        };
    }
    d
}

fn write_frame(dir: &Path, k: usize, w: &ExplicitBeamsWorld, vis: &[ShowcaseVisual]) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    for (s, v) in w.vehicles.iter().zip(vis) {
        let now: Vec<Vec3> = s.nodes.iter().map(|n| n.pos).collect();
        let (p, _) = deform(v, &now);
        for q in p {
            let q = q + s.origin;
            for c in [q.x, q.y, q.z] {
                buf.extend_from_slice(&c.to_le_bytes());
            }
        }
    }
    std::fs::write(dir.join(format!("frame_{k:04}.bin")), buf)
}

fn write_mesh(dir: &Path, vis: &[ShowcaseVisual]) -> std::io::Result<()> {
    let nv: usize = vis.iter().map(|v| v.positions.len()).sum();
    let nt: usize = vis.iter().map(|v| v.indices.len() / 3).sum();
    let mut f = std::fs::File::create(dir.join("mesh.bin"))?;
    f.write_all(&(nv as u32).to_le_bytes())?;
    f.write_all(&(nt as u32).to_le_bytes())?;
    for v in vis {
        for c in &v.colors {
            for x in c {
                f.write_all(&x.to_le_bytes())?;
            }
        }
    }
    let mut base = 0u32;
    for v in vis {
        for i in &v.indices {
            f.write_all(&(i + base).to_le_bytes())?;
        }
        base += v.positions.len() as u32;
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let scene = a.first().cloned().unwrap_or_else(|| "wall".into());
    let out = std::path::PathBuf::from(a.get(1).cloned().unwrap_or_else(|| format!("out/film/{scene}")));
    let design = a.get(2).cloned().unwrap_or_else(|| "sedan".into());
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let l = load(&design)?;
    let d = &l.desc;
    let zs: Vec<f32> = d.nodes.iter().map(|n| n.position[2]).collect();
    let xs: Vec<f32> = d.nodes.iter().map(|n| n.position[0]).collect();
    let zmax = zs.iter().cloned().fold(f32::MIN, f32::max);
    let half_w = xs.iter().cloned().fold(0.0f32, |m, x| m.max(x.abs()));
    let mut obstacles = Vec::new();
    let (mut sims, mut descs) = (Vec::new(), Vec::new());
    let (dt, frames, camera): (f32, usize, &str) = match scene.as_str() {
        "drive" => {
            sims.push(spawn(d, Vec3::ZERO, Vec3::ZERO)?);
            descs.push(d.clone());
            (1.0 / 30.0, 240, "chase")
        }
        "wall" => {
            let v = 56.0 / 3.6;
            let mut s = spawn(d, Vec3::ZERO, Vec3::Z * v)?;
            let wall_z = zmax + 0.05;
            s.boxes.push(StaticBox::centered(Vec3::new(0.0, 1.0, wall_z + 0.5), Vec3::new(2.0, 1.0, 0.5)));
            obstacles.push(json!({"box": {"centre": [0.0, 1.0, wall_z + 0.5], "half": [2.0, 1.0, 0.5]}}));
            sims.push(s);
            descs.push(d.clone());
            (0.004, 160, "side")
        }
        "polefront" => {
            // Frontal, a 254 mm pole on the centre line, 50 km/h.
            let v = 50.0 / 3.6;
            let mut s = spawn(d, Vec3::ZERO, Vec3::Z * v)?;
            let centre = Vec3::new(0.0, 1.0, zmax + 0.05 + 0.127);
            s.shapes.push(StaticShape::cylinder(centre, 0.127, 2.0));
            obstacles.push(json!({"cylinder": {"centre": centre.to_array(), "radius": 0.127, "half_height": 2.0}}));
            sims.push(s);
            descs.push(d.clone());
            (0.004, 110, "side")
        }
        "offset" => {
            // A rigid barrier over 40 % of the car's width (the driver's
            // side, +x), 64 km/h.
            let v = 64.0 / 3.6;
            let mut s = spawn(d, Vec3::ZERO, Vec3::Z * v)?;
            let wall_z = zmax + 0.05;
            let edge = half_w - 0.4 * 2.0 * half_w;
            let (c, h) = (Vec3::new(edge + 1.0, 1.0, wall_z + 0.5), Vec3::new(1.0, 1.0, 0.5));
            s.boxes.push(StaticBox::centered(c, h));
            obstacles.push(json!({"box": {"centre": c.to_array(), "half": h.to_array()}}));
            sims.push(s);
            descs.push(d.clone());
            (0.004, 110, "side")
        }
        "pole" => {
            // Sideways: the car turned a quarter so its left side leads along +Z.
            let t = turned(d, 1);
            let v = 32.0 / 3.6;
            let mut s = spawn(&t, Vec3::ZERO, Vec3::Z * v)?;
            // The front seat's H-point: about 1.4 m behind the front axle.
            let side = t.nodes.iter().map(|n| n.position[2]).fold(f32::MIN, f32::max);
            let centre = Vec3::new(-1.4, 1.0, side + 0.05 + 0.127);
            s.shapes.push(StaticShape::cylinder(centre, 0.127, 2.0));
            obstacles.push(json!({"cylinder": {"centre": centre.to_array(), "radius": 0.127, "half_height": 2.0}}));
            sims.push(s);
            descs.push(t);
            (0.004, 160, "side")
        }
        "c2c" => {
            let v = 50.0 / 3.6;
            sims.push(spawn(d, Vec3::new(0.0, 0.0, -0.025 - zmax), Vec3::Z * v)?);
            descs.push(d.clone());
            let t = turned(d, 2);
            sims.push(spawn(&t, Vec3::new(0.0, 0.0, 0.025 + zmax), -Vec3::Z * v)?);
            descs.push(t);
            (0.004, 160, "side")
        }
        _ => return Err(format!("unknown scene {scene}")),
    };
    let _ = half_w;
    // Bound on a spawn at the desc's own positions (the loader matches the
    // skin's nodes by position); the same node order as the scene's spawn.
    let vis: Vec<ShowcaseVisual> = descs
        .iter()
        .map(|dd| {
            let (probe, _) =
                ExplicitBeams::from_desc_jbeam_raycast(dd, Vec3::ZERO, Vec3::ZERO, settings())?;
            visual(&l, dd, &probe)
        })
        .collect::<Result<_, String>>()?;
    write_mesh(&out, &vis).map_err(|e| e.to_string())?;
    let mut w = ExplicitBeamsWorld::new(sims);
    let per = (dt / 0.0005).round() as usize;
    for k in 0..frames {
        if scene == "drive" {
            let t = k as f32 * dt;
            let input = DriveInput {
                forward: 1.0,
                steer: if t > 4.0 { 0.35 } else { 0.0 },
                ..Default::default()
            };
            w.vehicles[0].set_drive_input(input);
        }
        write_frame(&out, k, &w, &vis).map_err(|e| e.to_string())?;
        for _ in 0..per {
            w.step();
        }
    }
    let broken: Vec<usize> = w.vehicles.iter().map(|s| s.beams.iter().filter(|b| b.broken).count()).collect();
    let speed: Vec<f32> = w
        .vehicles
        .iter()
        .map(|s| {
            let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
            (s.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>() / m).length()
        })
        .collect();
    std::fs::write(
        out.join("scene.json"),
        serde_json::to_string_pretty(&json!({
            "scene": scene, "design": design, "dt": dt, "frames": frames, "camera": camera,
            "obstacles": obstacles, "broken": broken, "final_speed": speed,
            "paint": engine_softbody::glb::paint_of(&l.built.input.glb),
        }))
        .unwrap(),
    )
    .map_err(|e| e.to_string())?;
    println!("{scene}: {frames} frames every {dt} s → {}; broken beams {broken:?}, final speed {speed:?} m/s", out.display());
    Ok(())
}
