//! Do torn-off parts orbit the car? (The vehicle lab's report of 2026-11-17: on the
//! mountain a door came off and seemed to circle the tumbling body instead of
//! flying away.) Runs the lab's mountain on the GPU World with the generated sedan
//! and follows every part that comes free, from the GPU's own state.
//!
//! A part is **free** once no record that can pull joins it to the main body: the
//! pieces of every unbroken record but SUPPORT beams (push only; a stretched one
//! carries no force and breaks at its long bound, 100 lengths) and the aero and
//! pressure triangles' bookkeeping. **Held** parts are still joined by such a
//! record (a door on one hinge swings round the body: that is physics).
//!
//! For each free part, every window (1/60 s): its centre of mass relative to the
//! body's, the angle it has swept about the body's centre since it came free, and
//! its centre of mass's acceleration less gravity (what pushes or pulls it: contact
//! with the ground or the body, air). Writes a CSV and an SVG of the paths.
//!
//! ```text
//! cargo run --release -p engine-world-lab --example door_orbit -- --car <sedan_new dir> \
//!   [--secs 40] [--seed 1] [--terrain rolling|ledges] [--out engine/out/free-flight]
//! ```
//! (GPU: run under `scripts/with-gpu-lock.py --limit 600`.)

use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, ExplicitKind, GRAVITY,
};
use engine_softbody::explicit_debris::Pieces;
use engine_webgpu::explicit_coupling::read_words;
use glam::DVec3;
use rubble::vehicle_lab::{Lab, LabOptions};
use rubble::GpuContext;
use serde_json::json;
use std::fmt::Write as _;

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1).cloned())
}

struct Track {
    name: String,
    nodes: Vec<usize>,
    /// When it came free (s), and its path after: (t, rel. position, swept angle,
    /// |accel − g|, distance to the nearest body node).
    free_at: Option<f64>,
    path: Vec<(f64, DVec3, f64, f64, f64)>,
    last_rel: Option<DVec3>,
    swept: f64,
    last_v: Option<DVec3>,
}

fn main() -> anyhow::Result<()> {
    let dir = std::path::PathBuf::from(
        arg("--car").ok_or_else(|| anyhow::anyhow!("--car <dir with files.json>"))?,
    );
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
    let desc =
        engine_softbody::jbeam::load_jbeam(&files, pc.as_deref()).map_err(anyhow::Error::msg)?;
    // The page's settings: air drag and turbulence asked for (the BeamNG profile then
    // takes per-triangle aero), centred damping (always on for JBeam content).
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag: true,
        turbulence: true,
        ..Default::default()
    };
    let template: ExplicitBeams = rubble::showcase_scene::jbeam_vehicle_on(
        &files,
        pc.as_deref(),
        settings,
        engine_softbody::explicit_beams::JbeamWheels::Raycast,
    )?
    .0;
    anyhow::ensure!(template.nodes.len() == desc.nodes.len(), "node order");
    let secs: f64 = arg("--secs").and_then(|x| x.parse().ok()).unwrap_or(40.0);
    let seed: u64 = arg("--seed").and_then(|x| x.parse().ok()).unwrap_or(1);
    let terrain = arg("--terrain").unwrap_or_else(|| "rolling".into());
    let out = std::path::PathBuf::from(arg("--out").unwrap_or_else(|| "out/free-flight".into()));
    std::fs::create_dir_all(&out)?;
    // `--script <file.json>`: a lab script (`engine-webgpu/vehicle-lab-scripts/`) in place
    // of the plain mountain run (e.g. shots at the doors).
    let spec = match arg("--script") {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
        None => json!({
            "scene": "mountain",
            "mountain": { "seed": seed, "steepness": 50, "roughness": 0.6, "terrain": terrain },
            "events": []
        }),
    };
    let opts = LabOptions::from_json(&spec);
    let (mut world, mut lab) = Lab::build(GpuContext::Default, &template, opts, |_| {})?;
    lab.read_blocking(&world, true)?;
    lab.bench.set_script(&spec)?;

    let n = template.nodes.len();
    let mass: Vec<f64> = template.nodes.iter().map(|x| x.mass as f64).collect();
    let pulls = |k: ExplicitKind| {
        !matches!(
            k,
            ExplicitKind::JbeamSupport
                | ExplicitKind::JbeamAeroTri
                | ExplicitKind::JbeamAeroTri2
                | ExplicitKind::JbeamPressureTri
                | ExplicitKind::JbeamPressureTri2
        )
    };
    let mut tracks: Vec<Track> = desc
        .parts
        .iter()
        .filter(|p| p.name != "sedan_body" && p.nodes.len() >= 3)
        .map(|p| {
            // Its own nodes (in no other part).
            let own: Vec<usize> = p
                .nodes
                .iter()
                .map(|&i| i as usize)
                .filter(|&i| {
                    !desc
                        .parts
                        .iter()
                        .any(|q| q.name != p.name && q.nodes.contains(&(i as u32)))
                })
                .collect();
            Track {
                name: p.name.clone(),
                nodes: own,
                free_at: None,
                path: Vec::new(),
                last_rel: None,
                swept: 0.0,
                last_v: None,
            }
        })
        .filter(|t| !t.nodes.is_empty())
        .collect();
    let mut reported = vec![false; tracks.len()];
    let windows = (secs * 60.0).round() as u64;
    let dt = 1.0 / 60.0;
    let g = DVec3::new(0.0, GRAVITY as f64, 0.0);
    let plan = lab.read_plan();
    let mut csv =
        String::from("part,t,rel_x,rel_y,rel_z,dist,swept_deg,accel_less_g,nearest_body,body_x,body_y,body_z,body_accel_less_g\n");
    let mut spin_csv = String::from("t,lx,ly,lz,aero_tx,aero_ty,aero_tz,accel_less_g,nodes\n");
    let mut body_v: Option<DVec3> = None;
    for k in 0..windows {
        lab.before_step(&mut world)?;
        world.step()?;
        world.poll()?;
        lab.read_blocking(&world, true)?;
        let x = world
            .explicit_vehicles()
            .ok_or_else(|| anyhow::anyhow!("no vehicles"))?;
        let (d, q) = (world.device(), world.queue());
        let nodes = read_words(d, q, x.nodes_buffer(), plan.nodes.0, plan.nodes.1);
        let origin = read_words(d, q, x.misc_buffer(), plan.origin.0, plan.origin.1);
        let beams = read_words(d, q, x.beams_buffer(), plan.beams.0, plan.beams.1);
        let f = |w: u32| f32::from_bits(w) as f64;
        let o = DVec3::new(f(origin[0]), f(origin[1]), f(origin[2]));
        let pos: Vec<DVec3> = (0..n)
            .map(|i| o + DVec3::new(f(nodes[12 * i]), f(nodes[12 * i + 1]), f(nodes[12 * i + 2])))
            .collect();
        let vel: Vec<DVec3> = (0..n)
            .map(|i| {
                DVec3::new(
                    f(nodes[12 * i + 4]),
                    f(nodes[12 * i + 5]),
                    f(nodes[12 * i + 6]),
                )
            })
            .collect();
        let broken: Vec<bool> = (0..template.beams.len())
            .map(|j| f32::from_bits(beams[8 * j + 6]) != 0.0)
            .collect();
        let held = Pieces::from_beams(
            &template.nodes.iter().map(|x| x.mass).collect::<Vec<_>>(),
            template
                .beams
                .iter()
                .zip(&broken)
                .map(|(b, &x)| (b.a, b.b, x || !pulls(b.kind))),
        );
        let t = (k + 1) as f64 * dt;
        let body: Vec<usize> = (0..n).filter(|&i| held.piece[i] == held.main).collect();
        let com = |set: &[usize], v: &[DVec3]| {
            let m: f64 = set.iter().map(|&i| mass[i]).sum();
            set.iter().map(|&i| v[i] * mass[i]).sum::<DVec3>() / m
        };
        let cb = com(&body, &pos);
        let vb = com(&body, &vel);
        let body_acc = body_v.map_or(0.0, |lv: DVec3| ((vb - lv) / dt - g).length());
        body_v = Some(vb);
        // The spin ledger of the main body (the held piece): angular momentum about
        // its centre of mass, and the air's torque on it by the solver's own
        // per-triangle law at this state (`aero_tri_force`; triangles with all three
        // nodes on the body). A step in `nodes` is a piece leaving.
        {
            let on: Vec<bool> = (0..n).map(|i| held.piece[i] == held.main).collect();
            let (c, v) = (cb, vb);
            let l: DVec3 = body
                .iter()
                .map(|&i| mass[i] * (pos[i] - c).cross(vel[i] - v))
                .sum();
            let mut tq = DVec3::ZERO;
            for (j, b) in template.beams.iter().enumerate() {
                if b.kind != ExplicitKind::JbeamAeroTri {
                    continue;
                }
                let ext = &template.beam_ext[template.ext_of[j] as usize - 1];
                let ids = [b.a as usize, b.b as usize, ext.index(0) as usize];
                if !ids.iter().all(|&i| on[i]) {
                    continue;
                }
                let fa = engine_softbody::explicit_jbeam_ext::aero_tri_force(
                    ids.map(|i| (pos[i] - o).as_vec3()),
                    ids.map(|i| vel[i].as_vec3()),
                    ext,
                );
                for i in ids {
                    tq += (pos[i] - c).cross(fa.as_dvec3());
                }
            }
            let _ = writeln!(
                spin_csv,
                "{t:.4},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{body_acc:.4},{}",
                l.x,
                l.y,
                l.z,
                tq.x,
                tq.y,
                tq.z,
                body.len()
            );
        }
        for (ti, tr) in tracks.iter_mut().enumerate() {
            let free = tr.nodes.iter().all(|&i| held.piece[i] != held.main);
            if !free {
                // Held, but far from where it belongs: what holds it?
                let gap = tr
                    .nodes
                    .iter()
                    .map(|&i| {
                        body.iter()
                            .map(|&j| pos[i].distance(pos[j]))
                            .fold(f64::MAX, f64::min)
                    })
                    .fold(f64::MAX, f64::min);
                if gap > 1.0 && !reported[ti] {
                    reported[ti] = true;
                    let inside: Vec<bool> = (0..n).map(|i| tr.nodes.contains(&i)).collect();
                    let mut what = Vec::new();
                    for (j, b) in template.beams.iter().enumerate() {
                        if broken[j]
                            || !pulls(b.kind)
                            || inside[b.a as usize] == inside[b.b as usize]
                        {
                            continue;
                        }
                        let l = pos[b.a as usize].distance(pos[b.b as usize]);
                        let rest_now = f32::from_bits(beams[8 * j]) as f64;
                        what.push(format!("{j} {:?} {}–{} {l:.2} m (rest {rest_now:.3}, spawn {:.3}, k {:.0}, c {:.1})", b.kind, desc.nodes[b.a as usize].name, desc.nodes[b.b as usize].name, b.rest0, b.k, b.c));
                    }
                    println!(
                        "t {t:6.2} s: {} held {gap:.2} m from the body by: {}",
                        tr.name,
                        what.join("; ")
                    );
                }
                continue;
            }
            if tr.free_at.is_none() {
                tr.free_at = Some(t);
                println!(
                    "t {t:6.2} s: {} came free ({} nodes)",
                    tr.name,
                    tr.nodes.len()
                );
            }
            let c = com(&tr.nodes, &pos);
            let v = com(&tr.nodes, &vel);
            let rel = c - cb;
            if let Some(l) = tr.last_rel {
                tr.swept += l.angle_between(rel);
            }
            tr.last_rel = Some(rel);
            let acc = tr.last_v.map_or(0.0, |lv| ((v - lv) / dt - g).length());
            tr.last_v = Some(v);
            let near = tr
                .nodes
                .iter()
                .flat_map(|&i| body.iter().map(move |&j| (i, j)))
                .map(|(i, j)| pos[i].distance(pos[j]))
                .fold(f64::MAX, f64::min);
            tr.path.push((t, rel, tr.swept, acc, near));
            let _ = writeln!(
                csv,
                "{},{t:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{acc:.3},{near:.3},{:.4},{:.4},{:.4},{body_acc:.3}",
                tr.name,
                rel.x,
                rel.y,
                rel.z,
                rel.length(),
                tr.swept.to_degrees(),
                cb.x,
                cb.y,
                cb.z
            );
        }
        if k % 300 == 299 {
            let c = &lab.bench.car;
            println!(
                "t {t:6.2} s: car {:.1} km/h, {} broken, {} parts free",
                c.speed * 3.6,
                c.beams_broken,
                tracks.iter().filter(|t| t.free_at.is_some()).count()
            );
        }
        anyhow::ensure!(
            !lab.bench.car.non_finite,
            "non-finite node state at t = {t:.3} s"
        );
    }
    std::fs::write(out.join("door_orbit.csv"), &csv)?;
    std::fs::write(out.join("spin.csv"), &spin_csv)?;
    // Summary and the paths (top view, x–z, relative to the body's centre).
    let mut svg = String::new();
    let free: Vec<&Track> = tracks.iter().filter(|t| t.free_at.is_some()).collect();
    let span = free
        .iter()
        .flat_map(|t| t.path.iter().map(|p| p.1.x.abs().max(p.1.z.abs())))
        .fold(5.0f64, f64::max)
        .min(200.0);
    let (w, h) = (900.0, 900.0);
    let sx = |x: f64| w / 2.0 + x / span * (w / 2.0 - 30.0);
    let sz = |z: f64| h / 2.0 - z / span * (h / 2.0 - 30.0);
    let _ = writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{}" font-family="sans-serif" font-size="13"><rect width="100%" height="100%" fill="white"/>"#,
        h + 40.0
    );
    let _ = writeln!(
        svg,
        r#"<line x1="{}" y1="0" x2="{}" y2="{h}" stroke="lightgray"/><line x1="0" y1="{}" x2="{w}" y2="{}" stroke="lightgray"/><circle cx="{}" cy="{}" r="5" fill="black"/>"#,
        w / 2.0,
        w / 2.0,
        h / 2.0,
        h / 2.0,
        w / 2.0,
        h / 2.0
    );
    let colours = [
        "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf",
        "#bcbd22", "#7f7f7f",
    ];
    println!("\npart, free at (s), seconds free, swept about the body (deg), distance at free → end (m), median |accel − g| far from the body (m/s²)");
    for (i, t) in free.iter().enumerate() {
        let c = colours[i % colours.len()];
        let pts: Vec<String> = t
            .path
            .iter()
            .map(|p| format!("{:.1},{:.1}", sx(p.1.x), sz(p.1.z)))
            .collect();
        let _ = writeln!(
            svg,
            r#"<polyline points="{}" fill="none" stroke="{c}" stroke-width="1.5"/>"#,
            pts.join(" ")
        );
        let _ = writeln!(
            svg,
            r#"<text x="10" y="{}" fill="{c}">{} (free at {:.2} s, swept {:.0}°)</text>"#,
            18.0 + 16.0 * i as f64,
            t.name,
            t.free_at.unwrap(),
            t.swept.to_degrees()
        );
        let mut far: Vec<f64> = t.path.iter().filter(|p| p.4 > 0.5).map(|p| p.3).collect();
        far.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = far.get(far.len() / 2).copied().unwrap_or(f64::NAN);
        println!(
            "{}, {:.2}, {:.2}, {:.0}, {:.2} → {:.2}, {:.2}",
            t.name,
            t.free_at.unwrap(),
            t.path.last().unwrap().0 - t.free_at.unwrap(),
            t.swept.to_degrees(),
            t.path.first().unwrap().1.length(),
            t.path.last().unwrap().1.length(),
            med
        );
    }
    let _ = writeln!(
        svg,
        r#"<text x="10" y="{}">top view (x–z) relative to the body's centre of mass, ±{span:.0} m</text></svg>"#,
        h + 32.0
    );
    std::fs::write(out.join("door_orbit.svg"), svg)?;
    println!("wrote {}", out.join("door_orbit.csv").display());
    Ok(())
}
