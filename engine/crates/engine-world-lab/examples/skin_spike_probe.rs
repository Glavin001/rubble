//! A generated car's weighted skin (`.rskin`) after a vehicle-lab shot, on the CPU
//! reference (`vehicle_lab::cpu`; no GPU): per piece, the vertex that ends farthest
//! from its own bound nodes past its rest distance to them (a skin spike), and at the
//! end the two worst with their entries (node, weight, how far the node moved).
//! Found the 4x4's windscreen spike (`engine/docs/generated-vehicles-lessons.md` §20).
//!
//! ```text
//! cargo run --release -p engine-world-lab --example skin_spike_probe -- --car out/gen/suv_4x4_new \
//!   [--side top|left|right|front|rear] [--radius 1.5] [--speed 6] [--along 0] [--height 0.9] \
//!   [--distance 1] [--surface 1] [--secs 3] [--every 30]
//! ```
//! `--car`: a mod folder (`vehicle_gen_rig <design> --out out/gen/<design>_new`).
use engine_softbody::explicit_beams::{ExplicitBeamsSettings, JbeamWheels};
use glam::Vec3;
use rubble::vehicle_lab::cpu::CpuLab;
use rubble::vehicle_lab::LabOptions;
use serde_json::json;

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1).cloned())
}
fn num(k: &str, d: f64) -> f64 {
    arg(k).and_then(|x| x.parse().ok()).unwrap_or(d)
}

fn main() -> anyhow::Result<()> {
    let dir = arg("--car").unwrap_or("out/gen/suv_4x4_new".into());
    let dir = std::path::Path::new(&dir);
    let list: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("files.json"))?)?;
    let mut texts = Vec::new();
    let mut pc = None;
    for f in &list {
        if f.ends_with(".jbeam") {
            texts.push((f.clone(), std::fs::read_to_string(dir.join(f))?));
        } else if f.ends_with(".pc") {
            pc = Some(std::fs::read_to_string(dir.join(f))?);
        }
    }
    let files: Vec<(&str, &str)> = texts
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        ..Default::default()
    };
    let desc =
        engine_softbody::jbeam::load_jbeam(&files, pc.as_deref()).map_err(anyhow::Error::msg)?;
    let flex = engine_softbody::jbeam::assembled_rows(&files, pc.as_deref(), "flexbodies")
        .map_err(anyhow::Error::msg)?;
    let props = engine_softbody::jbeam::assembled_rows(&files, pc.as_deref(), "props")
        .map_err(anyhow::Error::msg)?;
    let (template, _) = rubble::showcase_scene::jbeam_vehicle_on(
        &files,
        pc.as_deref(),
        settings,
        JbeamWheels::Raycast,
    )?;
    let glb_name = list
        .iter()
        .find(|f| f.ends_with(".glb") && !f.contains(".lite."))
        .unwrap();
    let skin_name = list
        .iter()
        .find(|f| f.ends_with(".rskin") && !f.contains(".lite."))
        .unwrap();
    let g = engine_softbody_authoring::glb::read(&std::fs::read(dir.join(glb_name))?)
        .map_err(anyhow::Error::msg)?;
    let all: Vec<_> = g
        .meshes
        .iter()
        .map(|m| engine_softbody_authoring::flexskin::glb_mesh(&g, m))
        .collect();
    let skin =
        engine_softbody_authoring::flexskin::Skin::decode(&std::fs::read(dir.join(skin_name))?)
            .map_err(anyhow::Error::msg)?;
    let (vis, _) = engine_softbody_authoring::flexmesh::jbeam_visual_skin(
        &flex,
        &props,
        &desc,
        &template,
        &all,
        &Default::default(),
        [0.6, 0.04, 0.03],
        Some(engine_softbody_authoring::flexskin::SkinRef::Decoded(&skin)),
    )
    .map_err(anyhow::Error::msg)?;
    let side = arg("--side").unwrap_or("top".into());
    let script = json!({
        "mode": "shoot", "scene": "flat", "surface": num("--surface", 1.0) != 0.0,
        "events": [
            { "t": 0.5, "fire": { "shape": "sphere", "radius": num("--radius", 1.5), "material": "steel",
              "speed": num("--speed", 6.0),
              "aim": { "side": side, "height": num("--height", 0.9), "along": num("--along", 0.0), "distance": num("--distance", 1.0) } } }
        ]
    });
    let opts = LabOptions::from_json(&script);
    let mut lab = CpuLab::new(&template, opts)?;
    lab.bench.set_script(&script)?;
    anyhow::ensure!(lab.sim.nodes.len() == template.nodes.len());
    let rest: Vec<Vec3> = template.nodes.iter().map(|n| n.pos).collect();
    let windows = (num("--secs", 3.0) * 60.0) as u64;
    let every = num("--every", 30.0) as u64;
    let report = |nodes: &[Vec3], t: f64, full: bool| {
        let mut worst: Vec<(f32, usize, String)> = Vec::new();
        for (name, first, count) in &vis.pieces {
            let mut best = (0.0f32, usize::MAX);
            for i in *first as usize..(*first + *count) as usize {
                let (a, b) = (vis.wt_start[i] as usize, vis.wt_start[i + 1] as usize);
                if a == b {
                    continue;
                }
                let w: Vec<(u32, f32, f32, f32)> = (a..b)
                    .map(|k| {
                        (
                            vis.wt_node[k],
                            vis.wt_w[k][0],
                            vis.wt_w[k][1],
                            vis.wt_w[k][2],
                        )
                    })
                    .collect();
                let (p, _) = engine_softbody_authoring::flexskin::eval(
                    &w,
                    vis.wt_off[i],
                    vis.wt_nrm[i],
                    nodes,
                );
                let (p0, _) = engine_softbody_authoring::flexskin::eval(
                    &w,
                    vis.wt_off[i],
                    vis.wt_nrm[i],
                    &rest,
                );
                let d = |x: Vec3, q: &[Vec3]| {
                    w.iter()
                        .filter(|e| e.1.abs() > 0.05)
                        .map(|e| (q[e.0 as usize] - x).length())
                        .fold(f32::MAX, f32::min)
                };
                let s = d(p, nodes) - d(p0, &rest);
                if s > best.0 {
                    best = (s, i);
                }
            }
            if best.1 != usize::MAX {
                worst.push((best.0, best.1, name.clone()));
            }
        }
        worst.sort_by(|x, y| y.0.total_cmp(&x.0));
        let line: Vec<String> = worst
            .iter()
            .take(5)
            .map(|w| format!("{} {:.2} m (v{})", w.2, w.0, w.1))
            .collect();
        println!(
            "t {t:5.2}: farthest past rest from own nodes: {}",
            line.join(", ")
        );
        if full {
            for w in worst.iter().take(2) {
                let i = w.1;
                let (a, b) = (vis.wt_start[i] as usize, vis.wt_start[i + 1] as usize);
                let mut e: Vec<(u32, f32)> =
                    (a..b).map(|k| (vis.wt_node[k], vis.wt_w[k][0])).collect();
                e.sort_by(|x, y| y.1.abs().total_cmp(&x.1.abs()));
                let sa: f32 = e.iter().map(|x| x.1.abs()).sum();
                println!("  {} v{i} rest {:?} sum|A| {sa:.2}", w.2, vis.positions[i]);
                for (n, aa) in e.iter().take(8) {
                    let nm = &desc.nodes[*n as usize].name;
                    let mv = nodes[*n as usize] - rest[*n as usize];
                    println!(
                        "     {nm:<12} A {aa:+.3} moved {:?} ({:.2} m)",
                        mv,
                        mv.length()
                    );
                }
            }
        }
    };
    for k in 0..windows {
        lab.step()?;
        if k % every == every - 1 || k + 1 == windows {
            let nodes: Vec<Vec3> = lab.sim.nodes.iter().map(|n| n.pos).collect();
            report(&nodes, lab.bench.t, k + 1 == windows);
        }
    }
    let broken = lab.sim.beams.iter().filter(|b| b.broken).count();
    println!("beams broken {broken}");
    Ok(())
}
