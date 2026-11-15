//! The binding lint (`authoring::skin_lint::lint_parts`) on an imported
//! BeamNG vehicle's flexbodies as our loader binds them (flexmesh.rs): a
//! calibration of the lint against content built for BeamNG. Local data only
//! (nothing derived is written or committed). No parts: A1 checks each piece
//! against its own bound nodes; A7, B1, B4 need parts and measure nothing.
//! `cargo run --release -p engine-softbody --example skin_lint_import -- <vehicles dir> <name> [pc]`
use engine_softbody::authoring::skin_lint::lint_parts;
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
use engine_softbody::flexmesh::{jbeam_visual, read_collada, read_materials, DaeMesh};
use engine_softbody::jbeam::{assembled_rows, load_jbeam, read_vehicle_dir};
use glam::Vec3;

fn walk(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(a.get(1).ok_or("vehicles dir")?);
    let name = a.get(2).ok_or("vehicle name")?;
    let files = read_vehicle_dir(&dir, name)?;
    let refs: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let pc = a.get(3).map(|p| std::fs::read_to_string(dir.join(name).join(p)).map_err(|e| e.to_string())).transpose()?;
    let desc = load_jbeam(&refs, pc.as_deref())?;
    let flex = assembled_rows(&refs, pc.as_deref(), "flexbodies")?;
    let props = assembled_rows(&refs, pc.as_deref(), "props")?;
    let (sim, _) = ExplicitBeams::from_desc_jbeam_with(&desc, JbeamWheels::Pressure, Vec3::ZERO, Vec3::ZERO, ExplicitBeamsSettings::default())?;
    let mut meshes: Vec<DaeMesh> = Vec::new();
    let mut mats = std::collections::HashMap::new();
    let mut all = Vec::new();
    for d in [dir.join(name), dir.join("common")] {
        walk(&d, &mut all);
    }
    for e in &all {
        let ext = e.extension().and_then(|x| x.to_str()).unwrap_or("");
        if ext == "dae" {
            meshes.extend(read_collada(&String::from_utf8_lossy(&std::fs::read(e).map_err(|x| x.to_string())?))?);
        } else if e.to_string_lossy().ends_with(".materials.json") {
            mats.extend(read_materials(&std::fs::read_to_string(e).map_err(|x| x.to_string())?));
        }
    }
    let (v, rep) = jbeam_visual(&flex, &props, &desc, &sim, &meshes, &mats, [0.5, 0.05, 0.04])?;
    eprintln!("{name}: {} vertices, {} triangles, {} unbound", rep.vertices, rep.triangles, rep.unbound);
    let parts = vec![u16::MAX; v.node_rest.len()];
    let beams: Vec<(u32, u32)> = sim.beams.iter().map(|b| (b.a, b.b)).collect();
    let r = lint_parts(&v, &parts, &[], &beams, 7);
    println!(
        "lint {name}: A1 {} A3 max {:.3} over {} ({:.1} %) B3 max {:.2} p99 {:.2} A6 {} A4 max {:.1} over {} | B2 affine {:.1e} m (p99 {:.1e}) bend {:.1e} m (p99 {:.1e}) | A2 angle min {:.0}° p1 {:.0}°",
        r.cross_group, r.weight_norm_max, r.weight_norm_over, 100.0 * r.weight_norm_over as f32 / r.vertices.max(1) as f32,
        r.gain_max, r.gain_p99, r.incoherent_edges, r.reach_max, r.reach_over,
        r.affine_err, r.affine_p99, r.bend_err, r.bend_p99, r.frame_angle_min, r.frame_angle_p1
    );
    Ok(())
}
