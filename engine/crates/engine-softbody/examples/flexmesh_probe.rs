//! Load an imported BeamNG vehicle's flexbodies and props (flexmesh.rs) and
//! report what bound. Local data only (a gitignored `inspiration/` folder).
//! `cargo run --release -p engine-softbody --example flexmesh_probe -- <vehicles dir> <name> [pc]`
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
use engine_softbody::flexmesh::{jbeam_visual, read_collada, DaeMesh};
use engine_softbody::jbeam::{assembled_rows, load_jbeam, read_vehicle_dir};
use glam::Vec3;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(a.get(1).ok_or("vehicles dir")?);
    let name = a.get(2).ok_or("vehicle name")?;
    let files = read_vehicle_dir(&dir, name)?;
    let refs: Vec<(&str, &str)> = files.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let pc = a.get(3).map(|p| std::fs::read_to_string(dir.join(name).join(p)).map_err(|e| e.to_string())).transpose()?;
    let t = std::time::Instant::now();
    let desc = load_jbeam(&refs, pc.as_deref())?;
    let flex = assembled_rows(&refs, pc.as_deref(), "flexbodies")?;
    let props = assembled_rows(&refs, pc.as_deref(), "props")?;
    // The JBeam-mode solver (desc order; pressure wheels when it has them).
    let (sim, _) = ExplicitBeams::from_desc_jbeam_with(&desc, JbeamWheels::Pressure, Vec3::ZERO, Vec3::ZERO, ExplicitBeamsSettings::default())?;
    let sim_nodes: Vec<Vec3> = sim.nodes.iter().map(|n| n.pos).collect();
    let mut meshes: Vec<DaeMesh> = Vec::new();
    for e in walk(&dir.join(name)) {
        if e.extension().is_some_and(|x| x == "dae") {
            let src = String::from_utf8_lossy(&std::fs::read(&e).map_err(|x| x.to_string())?).into_owned();
            let m = read_collada(&src)?;
            eprintln!("{}: {} meshes, {} vertices", e.display(), m.len(), m.iter().map(|x| x.pos.len()).sum::<usize>());
            if std::env::var("NAMES").is_ok() {
                let mut v: Vec<(usize, &str)> = m.iter().map(|x| (x.pos.len(), x.name.as_str())).collect();
                v.sort_by(|a, b| b.cmp(a));
                eprintln!("{:?}", &v[..v.len().min(25)]);
            }
            meshes.extend(m);
        }
    }
    let mut mats = std::collections::HashMap::new();
    for e in walk(&dir.join(name)) {
        if e.to_string_lossy().ends_with(".materials.json") {
            mats.extend(engine_softbody::flexmesh::read_materials(&std::fs::read_to_string(&e).unwrap_or_default()));
        }
    }
    let (v, rep) = jbeam_visual(&flex, &props, &desc, &sim, &meshes, &mats, [0.5, 0.05, 0.04])?;
    eprintln!(
        "{name}: {} nodes ({} in the solver); {} flexbodies, {} props, {} vertices, {} triangles, {} unbound vertices, {:.1} s",
        desc.nodes.len(), sim_nodes.len(), rep.flexbodies, rep.props, rep.vertices, rep.triangles, rep.unbound, t.elapsed().as_secs_f32()
    );
    let mut miss = rep.missing.clone();
    miss.sort();
    miss.dedup();
    eprintln!("missing ({}): {:?}", miss.len(), &miss[..miss.len().min(20)]);
    // The rest mesh reproduces itself through the binding.
    let (p, _) = engine_softbody::authoring::skin::deform(&v, &sim_nodes);
    let worst = p.iter().zip(&v.positions).map(|(a, b)| a.distance(Vec3::from_array(*b))).fold(0.0f32, f32::max);
    eprintln!("rest mesh reproduced to {worst:.2e} m");
    for u in &v.units {
        eprintln!("unit {} kind {} beams {} trigger {} shards {}", u.name, u.kind, u.beams.len(), u.strain, u.shards);
    }
    Ok(())
}

fn walk(d: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(d) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(walk(&p));
            } else {
                out.push(p);
            }
        }
    }
    out
}
