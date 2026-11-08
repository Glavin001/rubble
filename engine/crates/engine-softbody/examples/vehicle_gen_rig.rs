//! Rigs a vehicle-gen design with crash physics: reads its art `.glb` and
//! `rig.json` (`engine/tools/vehicle-gen`, `build.py`), builds the gen3
//! lattice on it (`authoring::gen2::vg_input`), binds every render vertex
//! (`.rskin`) and writes a BeamNG mod folder (JBeam parts, `.pc`, the `.glb`,
//! the `.rskin`). Then it prints the car's mass, centre of mass and lattice,
//! and its rest fidelity at spawn (the skinned mesh against the design).
//!
//! ```bash
//! cargo run --release -p engine-softbody --example vehicle_gen_rig -- <design> [--params FILE] [--in DIR] [--out DIR]
//! ```
//! A design of the Rust generator (`engine-vehiclegen`: the sedan) is built
//! in memory from its parameters (`--params`, JSON; the designer page's);
//! the others read a Python build's `.glb` and `rig.json` from `--in`,
//! default `engine/tools/vehicle-gen/out/<design>` when it holds a build,
//! else `engine/scenes/vehicles/<design>`. `--out` defaults to
//! `engine/crates/engine-webgpu/__mods/<design>` (gitignored), which the page
//! loads as `softbody.html?solver=explicit&mod=/__mods/<design>&pc=<design>/<design>.pc&wheels=raycast`.
//! Guide: `engine/docs/generated-vehicles.md`.
use engine_softbody::authoring::gen2::{rig, vg_input};
use std::path::{Path, PathBuf};

fn pct(v: &mut [f32], q: f32) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f32 * q).round() as usize]
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let design = a.first().cloned().unwrap_or_else(|| "sedan".into());
    let opt = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out: PathBuf = opt("--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("crates/engine-webgpu/__mods").join(&design));
    let params = opt("--params").map(|p| std::fs::read_to_string(p).expect("--params file"));
    let input_dir = opt("--in").map(PathBuf::from);
    let t0 = std::time::Instant::now();
    let inp = vg_input::VgInput::find(&design, params.as_deref(), input_dir.as_deref())?;
    println!(
        "{design}: {} meshes, {} markers ({})",
        inp.model.parts.len(),
        inp.model.markers.len(),
        match &input_dir {
            Some(d) => d.display().to_string(),
            None if engine_vehiclegen::default_params_json(&design).is_ok() => {
                "the Rust generator, in memory".into()
            }
            None => "a Python build's rig.json".into(),
        }
    );
    let b = vg_input::build(inp)?;
    let nv: usize = b.skin.meshes.iter().map(|m| m.verts.len()).sum();
    let ne: usize = b
        .skin
        .meshes
        .iter()
        .flat_map(|m| m.verts.iter())
        .map(|v| v.w.len())
        .sum();
    println!(
        "rigged and bound in {:.1} s; skins: {nv} vertices, {ne} entries ({:.1} per vertex)",
        t0.elapsed().as_secs_f32(),
        ne as f32 / nv.max(1) as f32
    );
    let files = vg_input::files(&b);
    let mut bytes = 0usize;
    for (p, data) in &files {
        let path = out.join(p);
        std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
        std::fs::write(&path, data).map_err(|e| e.to_string())?;
        bytes += data.len();
    }
    println!(
        "wrote {} ({} files, {:.1} MB)",
        out.display(),
        files.len(),
        bytes as f32 / 1e6
    );

    let car = &b.rig.car;
    let hub = rig::hub_extra(car);
    let total: f32 = car.nodes.iter().map(|n| n.mass).sum::<f32>() + hub.values().sum::<f32>();
    let com = car
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| n.p * (n.mass + hub.get(&i).copied().unwrap_or(0.0)))
        .sum::<glam::Vec3>()
        / total;
    let wb = b.input.rig.dims.wheelbase;
    println!(
        "mass {total:.1} kg (kerb {:.0}); centre of mass {:.0} mm behind the front axle ({:.1} % of the wheelbase), {:.0} mm up, {:+.0} mm left",
        b.input.physics.kerb,
        com.y * 1e3,
        100.0 * com.y / wb,
        com.z * 1e3,
        com.x * 1e3
    );
    println!(
        "nodes {} ({} contact-only), beams {}, triangles {}; omega_max*h {:.3}",
        car.nodes.len(),
        car.nodes.iter().filter(|n| n.contact).count(),
        car.beams.len(),
        car.tris.len(),
        car.omega_h
    );
    let rows = car.rows(&hub);
    for part in engine_softbody::authoring::gen2::build::Part::ALL {
        let ids: Vec<usize> = (0..car.nodes.len())
            .filter(|&i| car.nodes[i].part == part)
            .collect();
        if ids.is_empty() {
            continue;
        }
        let kg: f32 = ids.iter().map(|&i| car.nodes[i].mass).sum();
        let mut m: Vec<f32> = ids.iter().map(|&i| car.nodes[i].mass).collect();
        let mut rw: Vec<f32> = ids.iter().map(|&i| rows[i]).collect();
        println!(
            "  {:<12} {:>4} nodes {:>7.1} kg  node kg p50 {:.2}  row p50/max {:.2}/{:.2}",
            part.name().trim_start_matches("gen2_"),
            ids.len(),
            kg,
            pct(&mut m, 0.5),
            pct(&mut rw, 0.5),
            pct(&mut rw, 1.0)
        );
    }
    Ok(())
}
