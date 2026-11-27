//! Part outlines of a crash reference scenario on the CPU reference, for comparing shapes
//! with the reference footage (`engine_softbody::crash_reference::outline`,
//! `engine/tools/crash-reference/outlines.py`).
//!
//! `crash_outlines -- ID [--contact edge] [--size 1280x720] [--out DIR] [--no-render] [--cameras FILE]`
//!
//! Runs scenario ID (run 0, nominal) to each of its moments (ms after first contact, at the
//! exact solver step the web recorder stops on) and writes `DIR/outlines.json`: per moment,
//! per body, per part group its outline as each scenario camera sees it (pixels, the page's
//! projection) and as the ghost (its rest shape carried rigidly with the cabin and rear),
//! the hood's centreline, the wheels (centre and radius), and the same in the car's own
//! side frame (x along the car, y up, metres, origin at the rear axle on the ground).
//! Unless `--no-render`, also `DIR/<moment>-<camera>.png`: the collision surface drawn
//! from the same camera (flat shaded, each part group its colour), so the outlines line
//! up with a still exactly. Default DIR: `out/crash-reference/outlines/<ID>-<contact>`.
//! `--cameras FILE` adds explicit cameras (a JSON list of `outline::FreeCamera` with an
//! optional `width` and `height` of the image they apply to): each is projected and
//! rendered like a scenario camera, under its own name.
//! The census of a design's parts: `crash_outlines -- --census sedan`.

use engine_softbody::crash_reference::outline::{self, CarFrame, Ghost, Parts, View};
use engine_softbody::crash_reference::{self as cr, Contact, CpuRunner, Jitter, Runner, DT};
use glam::Vec3;
use serde_json::{json, Value};

/// Each group's colour in the rendered still.
fn colour(group: &str) -> [f32; 3] {
    match group {
        "hood" => [0.95, 0.55, 0.10],
        "fender_l" | "fender_r" => [0.20, 0.70, 0.95],
        "bumper_f" | "bumper_r" => [0.85, 0.25, 0.85],
        "door_fl" | "door_fr" => [0.35, 0.85, 0.35],
        "door_rl" | "door_rr" => [0.20, 0.55, 0.30],
        "trunk" => [0.80, 0.80, 0.25],
        "roof" => [0.70, 0.70, 0.95],
        "greenhouse" => [0.55, 0.55, 0.75],
        "sill_l" | "sill_r" => [0.95, 0.30, 0.30],
        _ => [0.62, 0.62, 0.62],
    }
}

fn srgb(x: f32) -> u8 {
    (x.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8
}

/// Z-buffered flat-shaded triangles (world corners, linear colour) from `v`.
fn raster(v: &View, tris: &[([Vec3; 3], [f32; 3])]) -> Vec<[u8; 3]> {
    let (w, h) = (v.width as usize, v.height as usize);
    let mut zb = vec![f32::MAX; w * h];
    let mut img: Vec<[u8; 3]> = (0..w * h)
        .map(|k| {
            let c = 0.05 + 0.06 * (1.0 - (k / w) as f32 / h as f32);
            [srgb(c * 0.7), srgb(c * 0.8), srgb(c * 1.1)]
        })
        .collect();
    let light = Vec3::new(0.4, 0.8, 0.45).normalize();
    for (p, col) in tris {
        let Some(s) = p
            .iter()
            .map(|&q| v.project(q).map(|x| (x, v.depth(q))))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let area = (s[1].0[0] - s[0].0[0]) * (s[2].0[1] - s[0].0[1])
            - (s[2].0[0] - s[0].0[0]) * (s[1].0[1] - s[0].0[1]);
        if area.abs() < 1e-9 {
            continue;
        }
        let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
        let shade = 0.3 + 0.7 * n.dot(light).abs();
        let c = [
            srgb(col[0] * shade),
            srgb(col[1] * shade),
            srgb(col[2] * shade),
        ];
        let x0 = s.iter().map(|q| q.0[0]).fold(f32::MAX, f32::min).max(0.0) as usize;
        let x1 = s
            .iter()
            .map(|q| q.0[0])
            .fold(f32::MIN, f32::max)
            .min(w as f32 - 1.0);
        let y0 = s.iter().map(|q| q.0[1]).fold(f32::MAX, f32::min).max(0.0) as usize;
        let y1 = s
            .iter()
            .map(|q| q.0[1])
            .fold(f32::MIN, f32::max)
            .min(h as f32 - 1.0);
        if x1 < 0.0 || y1 < 0.0 {
            continue;
        }
        for y in y0..=y1 as usize {
            for x in x0..=x1 as usize {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let e = |a: usize, b: usize| {
                    ((s[a].0[0] - px) * (s[b].0[1] - py) - (s[b].0[0] - px) * (s[a].0[1] - py))
                        / area
                };
                let (w0, w1) = (e(1, 2), e(2, 0));
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let d = 1.0 / (w0 / s[0].1 + w1 / s[1].1 + w2 / s[2].1);
                let k = y * w + x;
                if d < zb[k] {
                    zb[k] = d;
                    img[k] = c;
                }
            }
        }
    }
    img
}

fn write_png(img: &[[u8; 3]], w: usize, h: usize, path: &str) -> Result<(), String> {
    let f = std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
    let mut e = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    e.write_header()
        .and_then(|mut wr| wr.write_image_data(&img.concat()))
        .map_err(|e| e.to_string())
}

/// A box's 12 triangles (centre, half extents).
fn box_tris(c: Vec3, h: Vec3) -> Vec<[Vec3; 3]> {
    let k = |i: u32| {
        c + h * Vec3::new(
            if i & 1 == 1 { 1.0 } else { -1.0 },
            if i & 2 == 2 { 1.0 } else { -1.0 },
            if i & 4 == 4 { 1.0 } else { -1.0 },
        )
    };
    let faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ];
    faces
        .iter()
        .flat_map(|f| [[k(f[0]), k(f[1]), k(f[2])], [k(f[0]), k(f[2]), k(f[3])]])
        .collect()
}

fn outlines_json(
    parts: &Parts,
    now: &[Vec3],
    ghost: &[Vec3],
    proj: &dyn Fn(Vec3) -> Option<[f32; 2]>,
) -> Value {
    let mut groups = serde_json::Map::new();
    for g in &parts.groups {
        groups.insert(g.name.clone(), json!({ "now": outline::outline(g, now, proj), "ghost": outline::outline(g, ghost, proj) }));
    }
    json!({
        "parts": groups,
        "hood_centreline": { "now": outline::centreline_line(parts, now, proj), "ghost": outline::centreline_line(parts, ghost, proj) },
    })
}

fn census(design: &str) -> Result<(), String> {
    let t = engine_softbody_authoring::crash_templates::template(design)?;
    let rest: Vec<Vec3> = t
        .desc
        .nodes
        .iter()
        .map(|n| Vec3::from(n.position))
        .collect();
    let tris: Vec<[u32; 3]> = t
        .desc
        .triangles
        .iter()
        .filter(|x| x.collision)
        .map(|x| x.nodes)
        .collect();
    let p = outline::groups(&t, &rest, &tris);
    println!(
        "{design}: {} nodes, {} collision triangles",
        rest.len(),
        tris.len()
    );
    for g in &p.groups {
        println!(
            "  {:12} {:4} nodes, {} chains ({})",
            g.name,
            g.nodes.len(),
            g.chains.len(),
            if g.boundary { "boundary" } else { "hull" }
        );
    }
    println!(
        "  hood centreline rows {}: {:?}",
        p.hood_centreline.len(),
        p.hood_centreline
            .iter()
            .map(|r| r
                .iter()
                .map(|&i| t.desc.nodes[i].name.clone())
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );
    println!(
        "  wheels {:?}",
        p.wheels
            .iter()
            .map(|w| (&w.name, w.radius))
            .collect::<Vec<_>>()
    );
    println!("  rigid fit over {} nodes", p.rigid.len());
    Ok(())
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let flag = |k: &str| {
        a.iter()
            .position(|x| x == k)
            .and_then(|i| a.get(i + 1))
            .cloned()
    };
    if let Some(d) = flag("--census") {
        return census(&d);
    }
    let id = a.first().filter(|x| !x.starts_with("--")).ok_or(
        "usage: crash_outlines ID [--contact edge] [--size 1280x720] [--out DIR] [--no-render] [--cameras FILE]",
    )?;
    let s = cr::get(id).ok_or(format!("no scenario {id}"))?;
    let contact = flag("--contact").map_or(Ok(Contact::Edge), |x| {
        Contact::from_name(&x).ok_or(format!("no contact setting {x}"))
    })?;
    let size = flag("--size").unwrap_or("1280x720".into());
    let (w, h) = size
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse::<usize>().ok()?, h.parse::<usize>().ok()?)))
        .ok_or("--size WxH")?;
    let out = flag("--out").unwrap_or(format!(
        "out/crash-reference/outlines/{}-{}",
        s.id,
        contact.name()
    ));
    let render = !a.iter().any(|x| x == "--no-render");
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let templates = engine_softbody_authoring::crash_templates::templates_for(&s)?;
    let built = s.build(&templates, Jitter::run(0), contact)?;
    let mut runner = CpuRunner::new(&built);
    // Per body: its template, groups, start state.
    let bodies: Vec<(&cr::Body, Parts, Vec<Vec3>, Vec<f32>)> = built
        .layout
        .bodies
        .iter()
        .zip(&built.fleet)
        .map(|(b, f)| {
            let t = &templates[b.design];
            let rest: Vec<Vec3> = (0..f.nodes.len())
                .map(|i| {
                    t.desc
                        .nodes
                        .get(i)
                        .map_or(Vec3::ZERO, |n| Vec3::from(n.position))
                })
                .collect();
            let tris: Vec<[u32; 3]> = f
                .triangles
                .iter()
                .enumerate()
                .filter(|(k, _)| !f.tri_anti_only(*k))
                .map(|(_, x)| x.nodes)
                .collect();
            (
                b,
                outline::groups(t, &rest, &tris),
                f.positions(),
                f.nodes.iter().map(|n| n.mass).collect(),
            )
        })
        .collect();
    // The scenario's cameras, then any explicit ones (--cameras FILE: a JSON list of
    // { name, eye, target, up?, fov_y_deg?, width?, height? } in the scenario's world frame,
    // a reference image's own camera set in the Reference Studio).
    let mut views: Vec<(String, View)> = s
        .cameras
        .iter()
        .map(|c| (c.name.to_string(), View::new(c, w as f32, h as f32)))
        .collect();
    let mut extra = Vec::new();
    if let Some(file) = flag("--cameras") {
        let text = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
        let list: Vec<Value> = serde_json::from_str(&text).map_err(|e| format!("{file}: {e}"))?;
        for c in list {
            let fc: outline::FreeCamera =
                serde_json::from_value(c.clone()).map_err(|e| format!("{file}: {e}"))?;
            let cw = c["width"].as_f64().unwrap_or(w as f64) as f32;
            let ch = c["height"].as_f64().unwrap_or(h as f64) as f32;
            if views.iter().any(|(n, _)| *n == fc.name) {
                return Err(format!("{file}: camera {} is already a scenario camera", fc.name));
            }
            views.push((fc.name.clone(), View::look(&fc, cw, ch)));
            extra.push(json!({ "camera": fc, "width": cw, "height": ch }));
        }
    }
    let impact = (built.layout.impact_s / DT).round() as i64;
    let mut moments: Vec<(&cr::Moment, u32)> = s
        .moments
        .iter()
        .map(|m| {
            (
                m,
                (impact + (m.after_impact_ms * 2.0).round() as i64).max(0) as u32,
            )
        })
        .collect();
    moments.sort_by_key(|m| m.1);
    let mut at = 0u32;
    let mut out_moments = Vec::new();
    for (m, step) in moments {
        if step > at {
            runner.frame(step - at)?;
            at = step;
        }
        let sample = runner.sample()?;
        let mut jb = Vec::new();
        let mut render_tris: Vec<([Vec3; 3], [f32; 3])> = built
            .layout
            .boxes
            .iter()
            .flat_map(|&(c, hh)| box_tris(c, hh))
            .map(|t| (t, [0.32, 0.34, 0.40]))
            .collect();
        for (k, (b, parts, p0, mass)) in bodies.iter().enumerate() {
            let now = &sample.pos[k];
            let g = Ghost::fit(parts, p0, now, mass);
            let ghost: Vec<Vec3> = p0.iter().map(|&p| g.apply(p)).collect();
            let frame = CarFrame::new(parts, now, &g, b.heading());
            let side = |p: Vec3| Some(frame.side(p));
            let mut cams = serde_json::Map::new();
            for (c, v) in &views {
                let proj = |p: Vec3| v.project(p);
                let mut o = outlines_json(parts, now, &ghost, &proj);
                let wheels: Vec<Value> = parts
                    .wheels
                    .iter()
                    .map(|wh| {
                        let (c_now, c_gh) = (0.5 * (now[wh.hub[0]] + now[wh.hub[1]]), 0.5 * (ghost[wh.hub[0]] + ghost[wh.hub[1]]));
                        json!({ "name": wh.name, "now": { "c": v.project(c_now), "r": v.radius_px(c_now, wh.radius) }, "ghost": { "c": v.project(c_gh), "r": v.radius_px(c_gh, wh.radius) } })
                    })
                    .collect();
                o["wheels"] = json!(wheels);
                cams.insert(c.clone(), o);
            }
            let mut car = outlines_json(parts, now, &ghost, &side);
            car["wheels"] = json!(parts
                .wheels
                .iter()
                .map(|wh| json!({ "name": wh.name, "r": wh.radius, "now": frame.side(0.5 * (now[wh.hub[0]] + now[wh.hub[1]])), "ghost": frame.side(0.5 * (ghost[wh.hub[0]] + ghost[wh.hub[1]])) }))
                .collect::<Vec<_>>());
            // The cabin and rear's turn from the start: yaw (+ toward its left), pitch (+ nose down, rear up).
            let (yaw, pitch, _) = cr::metrics::yaw_pitch_roll(g.r, b.heading());
            jb.push(json!({ "name": b.name, "design": b.design, "cameras": cams, "car_frame": car, "fit_nodes": parts.rigid.len(), "rear_yaw_deg": yaw.to_degrees(), "rear_pitch_deg": pitch.to_degrees() }));
            // The still: every collision triangle, coloured by its group.
            let f = &built.fleet[k];
            let mut owner = vec![""; now.len()];
            for g in &parts.groups {
                for &i in &g.nodes {
                    // (the roof over the greenhouse it is part of)
                    if owner[i].is_empty() || owner[i] == "greenhouse" {
                        owner[i] = g.name.as_str();
                    }
                }
            }
            for (t, x) in f.triangles.iter().enumerate() {
                if f.tri_anti_only(t) {
                    continue;
                }
                let ids = x.nodes.map(|i| i as usize);
                let grp = if ids.iter().all(|&i| owner[i] == owner[ids[0]]) {
                    owner[ids[0]]
                } else {
                    ""
                };
                render_tris.push((ids.map(|i| now[i]), colour(grp)));
            }
        }
        if render {
            for (c, v) in &views {
                let path = format!("{out}/{}-{c}.png", m.name);
                write_png(&raster(v, &render_tris), v.width as usize, v.height as usize, &path)?;
            }
        }
        eprintln!("  {} ({} ms, step {step})", m.name, m.after_impact_ms);
        out_moments.push(json!({ "name": m.name, "after_impact_ms": m.after_impact_ms, "step": step, "t": step as f32 * DT, "bodies": jb }));
    }
    let doc = json!({
        "scenario": s.id, "title": s.title, "contact": contact.name(), "run": 0,
        "projection": { "width": w, "height": h, "fov_y_deg": outline::FOV_Y_DEG, "up": "+Y", "pixels": "x right, y down, from the top-left corner" },
        "car_frame": "x along the car's current heading (horizontal, its rear's turn), y up, metres; origin the rear axle's centre on the ground (y = 0)",
        "ghost": "rest shape carried rigidly with the cabin and rear (mass-weighted robust fit over cabin and rear section nodes)",
        "cameras": s.cameras, "explicit_cameras": extra, "moments": out_moments,
    });
    let path = format!("{out}/outlines.json");
    std::fs::write(
        &path,
        serde_json::to_string(&doc).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("{path}");
    Ok(())
}
