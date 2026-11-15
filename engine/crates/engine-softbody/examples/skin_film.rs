//! Side-by-side films of the showcase sedan's skin on the SAME node
//! trajectories (CPU only): left the node frames with the earlier stretch /
//! fold cuts (`Binding::Stretch`, what the page drew on 2026-11-13), right cut
//! only where beams broke (`Binding::Breaks`, the default). 1920 × 1080 (two
//! 960 × 1080 panels), perspective, the car filling most of the frame; smooth
//! shading from normals recomputed on the deformed mesh (as drawn: cut
//! triangles left out); the raycast wheels from the drive state; a
//! depth-tested skeleton overlay (live stiff beams: green, yellow → red by
//! plastic strain) in the orbits' second turn. Impacts in slow motion
//! (1/240 s a frame, 4× at 60 fps), the rest real time.
//!
//! `cargo run --release -p engine-softbody --example skin_film -- <clip> <out dir>`
//! clips: wall70, side60, grab. Frames: `<out>/<clip>/f00000.png`, resumable
//! (existing frames are kept). `ffmpeg -framerate 60 -i f%05d.png ...` makes
//! the MP4 (engine/docs/measurements/skin-shards.md, "Video v2").
use engine_softbody::authoring::node_frames::Binding;
use engine_softbody::authoring::skin::{
    deform_beams, raycast_visual, rigid_kind, showcase_visual_lod, unit_states, vertex_tears, wheel_vertex,
    BeamState, ShowcaseVisual, WheelPose,
};
use engine_softbody::authoring::{build, style, BuildParams, Built, Resolution};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, Grab, StaticBox, STEP};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use glam::{Quat, Vec3};

const W: usize = 960;
const H: usize = 1080;
const FOV_Y: f32 = 0.6; // radians (34°)

fn spawn(b: &Built, at: Vec3, yaw: f32, vel: Vec3) -> ExplicitBeams {
    let st = ExplicitBeamsSettings::default();
    let (p, _) = ExplicitBeams::from_truck_raycast(&b.truck, None, Vec3::ZERO, yaw, Vec3::ZERO, st.clone()).unwrap();
    let d = p.drive.as_ref().unwrap();
    let low = d
        .config
        .wheels
        .iter()
        .map(|w| 0.5 * (p.nodes[w.node1 as usize].pos.y + p.nodes[w.node2 as usize].pos.y) - w.radius)
        .fold(f32::MAX, f32::min);
    let (lo, hi) = p.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, c), n| (a.min(n.pos), c.max(n.pos)));
    let mid = 0.5 * (lo + hi);
    let pos = Vec3::new(at.x - mid.x, 0.02 - low, at.z - mid.z);
    let (mut s, _) = ExplicitBeams::from_truck_raycast(&b.truck, None, pos, yaw, vel, st).unwrap();
    let sp = vel.length();
    if let Some(d) = s.drive.as_mut() {
        let r: Vec<f32> = d.config.wheels.iter().map(|w| w.radius).collect();
        for (w, r) in d.wheels.iter_mut().zip(r) {
            w.omega = sp / r;
        }
    }
    s
}

#[derive(Clone, Copy)]
struct Cam {
    eye: Vec3,
    at: Vec3,
}

/// One frame to draw: the solver's state and the camera.
struct Snap {
    nodes: Vec<Vec3>,
    beams: Vec<BeamState>,
    k: Vec<f32>,
    rest0: Vec<f32>,
    ab: Vec<(u32, u32)>,
    wheels: Vec<WheelPose>,
    cam: Cam,
    overlay: bool,
}

fn snap(s: &ExplicitBeams, cam: Cam, overlay: bool) -> Snap {
    Snap {
        nodes: s.positions(),
        beams: s.beams.iter().map(|b| BeamState { broken: b.broken, rest: b.rest }).collect(),
        k: s.beams.iter().map(|b| b.k).collect(),
        rest0: s.beams.iter().map(|b| b.rest0).collect(),
        ab: s.beams.iter().map(|b| (b.a, b.b)).collect(),
        wheels: s
            .drive
            .as_ref()
            .map(|d| d.wheels.iter().map(|w| WheelPose { centre: w.centre, axle: w.axle, up: w.up, spin: w.spin }).collect())
            .unwrap_or_default(),
        cam,
        overlay,
    }
}

fn centroid(p: &[Vec3]) -> Vec3 {
    p.iter().copied().sum::<Vec3>() / p.len().max(1) as f32
}

/// The skin as drawn: positions, tears, smooth normals from the deformed
/// mesh (drawn triangles only, area weighted, each facing its rest side).
fn skin(v: &ShowcaseVisual, s: &Snap) -> (Vec<Vec3>, Vec<f32>, Vec<Vec3>) {
    let us = unit_states(v, &s.beams);
    let (mut pos, _) = deform_beams(v, &s.nodes, Some(&us), Some(&s.beams));
    for i in 0..pos.len() {
        let rk = v.rigid[i];
        if rk == rigid_kind::WHEEL || rk == rigid_kind::WHEEL_HUB {
            if let Some(w) = s.wheels.get(v.bind_nodes[i][0] as usize) {
                pos[i] = wheel_vertex(v.bind_coords[i], v.bind_normal[i], w, rk == rigid_kind::WHEEL_HUB).0;
            }
        }
    }
    let tear: Vec<f32> = vertex_tears(v, &s.nodes, Some(&s.beams), &pos)
        .iter()
        .enumerate()
        .map(|(i, &t)| if v.rigid[i] >= 2 { 0.0 } else { t })
        .collect();
    let mut nrm = vec![Vec3::ZERO; pos.len()];
    for t in v.indices.chunks_exact(3) {
        let ids = [t[0] as usize, t[1] as usize, t[2] as usize];
        if ids.iter().any(|&i| !pos[i].is_finite() || tear[i] >= 0.5) {
            continue;
        }
        let p = ids.map(|i| pos[i]);
        let r0 = ids.map(|i| Vec3::from_array(v.positions[i]));
        let vn: Vec3 = ids.iter().map(|&i| Vec3::from_array(v.normals[i])).sum();
        let side = if (r0[1] - r0[0]).cross(r0[2] - r0[0]).dot(vn) >= 0.0 { 1.0 } else { -1.0 };
        let f = (p[1] - p[0]).cross(p[2] - p[0]) * side;
        for i in ids {
            nrm[i] += f;
        }
    }
    for n in &mut nrm {
        *n = n.normalize_or_zero();
    }
    (pos, tear, nrm)
}

fn srgb(x: f32) -> u8 {
    (x.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8
}

/// One panel: z-buffered, the renderer's cut (tear interpolated, discarded
/// past ½), Phong-shaded with a key, a fill and a sky term; inside faces
/// dark (the page's bare panel inside); the skeleton depth-tested on top.
fn raster(v: &ShowcaseVisual, s: &Snap) -> Vec<[u8; 3]> {
    let (pos, tear, nrm) = skin(v, s);
    if std::env::var("DEBUG").is_ok() {
        let fin = pos.iter().filter(|p| p.is_finite()).count();
        let c = centroid(&pos.iter().copied().filter(|p| p.is_finite()).collect::<Vec<_>>());
        eprintln!("nodes centroid {:?} max z {:?} finite {fin}/{} centroid {c:?} cam {:?} -> {:?} tears<.5 {}", centroid(&s.nodes), s.nodes.iter().map(|p| p.z).fold(f32::MIN, f32::max), pos.len(), s.cam.eye, s.cam.at, tear.iter().filter(|&&t| t < 0.5).count());
    }
    let fwd = (s.cam.at - s.cam.eye).normalize();
    let right = fwd.cross(Vec3::Y).normalize();
    let up = right.cross(fwd);
    let f = (H as f32 * 0.5) / (FOV_Y * 0.5).tan();
    let proj = |p: Vec3| {
        let d = p - s.cam.eye;
        let z = d.dot(fwd);
        (W as f32 * 0.5 + d.dot(right) * f / z, H as f32 * 0.5 - d.dot(up) * f / z, z)
    };
    let mut zb = vec![f32::MAX; W * H];
    let mut img: Vec<[u8; 3]> = (0..W * H)
        .map(|k| {
            let y = (k / W) as f32 / H as f32;
            let c = 0.16 + 0.10 * (1.0 - y);
            [srgb(c * 0.55), srgb(c * 0.6), srgb(c * 0.7)]
        })
        .collect();
    let key = Vec3::new(0.35, 0.8, 0.5).normalize();
    let fill = Vec3::new(-0.6, 0.3, -0.4).normalize();
    for t in v.indices.chunks_exact(3) {
        let ids = [t[0] as usize, t[1] as usize, t[2] as usize];
        let p = ids.map(|i| pos[i]);
        if p.iter().any(|q| !q.is_finite()) {
            continue;
        }
        let tv = ids.map(|i| tear[i]);
        if tv.iter().all(|&x| x >= 0.5) {
            continue;
        }
        let sp = p.map(proj);
        if sp.iter().any(|q| q.2 < 0.1) {
            continue;
        }
        let x0 = sp.iter().map(|q| q.0).fold(f32::MAX, f32::min).max(0.0);
        let x1 = sp.iter().map(|q| q.0).fold(f32::MIN, f32::max).min(W as f32 - 1.0);
        let y0 = sp.iter().map(|q| q.1).fold(f32::MAX, f32::min).max(0.0);
        let y1 = sp.iter().map(|q| q.1).fold(f32::MIN, f32::max).min(H as f32 - 1.0);
        let area = (sp[1].0 - sp[0].0) * (sp[2].1 - sp[0].1) - (sp[2].0 - sp[0].0) * (sp[1].1 - sp[0].1);
        if area.abs() < 1e-9 || x0 > x1 || y0 > y1 {
            continue;
        }
        let col = v.colors[ids[0]];
        let ns = ids.map(|i| nrm[i]);
        for y in y0 as usize..=y1 as usize {
            for x in x0 as usize..=x1 as usize {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let w0 = ((sp[1].0 - px) * (sp[2].1 - py) - (sp[2].0 - px) * (sp[1].1 - py)) / area;
                let w1 = ((sp[2].0 - px) * (sp[0].1 - py) - (sp[0].0 - px) * (sp[2].1 - py)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                if w0 * tv[0] + w1 * tv[1] + w2 * tv[2] >= 0.5 {
                    continue;
                }
                // Perspective-correct depth and attributes.
                let iz = w0 / sp[0].2 + w1 / sp[1].2 + w2 / sp[2].2;
                let d = 1.0 / iz;
                let k = y * W + x;
                if d >= zb[k] {
                    continue;
                }
                zb[k] = d;
                let (a0, a1, a2) = (w0 / sp[0].2 * d, w1 / sp[1].2 * d, w2 / sp[2].2 * d);
                let n = (ns[0] * a0 + ns[1] * a1 + ns[2] * a2).normalize_or_zero();
                let q = p[0] * a0 + p[1] * a1 + p[2] * a2;
                let view = (s.cam.eye - q).normalize();
                img[k] = if n.dot(view) < 0.0 {
                    // The panel's inside: bare, dark.
                    [srgb(0.035), srgb(0.035), srgb(0.04)]
                } else {
                    let diff = 0.75 * n.dot(key).max(0.0) + 0.2 * n.dot(fill).max(0.0) + 0.18 * (0.5 + 0.5 * n.y);
                    let h = (key + view).normalize();
                    let spec = 0.35 * n.dot(h).max(0.0).powf(60.0);
                    let c = Vec3::from_array(col) * diff + Vec3::splat(spec);
                    [srgb(c.x), srgb(c.y), srgb(c.z)]
                };
            }
        }
    }
    if s.overlay {
        for (j, &(a, b)) in s.ab.iter().enumerate() {
            if s.beams[j].broken || s.k[j] <= 0.0 {
                continue;
            }
            let pl = ((s.beams[j].rest - s.rest0[j]).abs() / s.rest0[j].max(1e-6) / 0.08).min(1.0);
            let col = if pl < 0.0125 { [60u8, 230, 110] } else { [255, (215.0 * (1.0 - pl)) as u8 + 25, 20] };
            let (pa, pb) = (proj(s.nodes[a as usize]), proj(s.nodes[b as usize]));
            if pa.2 < 0.1 || pb.2 < 0.1 {
                continue;
            }
            let steps = ((pb.0 - pa.0).abs().max((pb.1 - pa.1).abs()) as usize).clamp(1, 4000);
            for i in 0..=steps {
                let t = i as f32 / steps as f32;
                let (x, y) = (pa.0 + (pb.0 - pa.0) * t, pa.1 + (pb.1 - pa.1) * t);
                // Depth-tested (perspective-correct), 3 cm in front allowed.
                let z = 1.0 / ((1.0 - t) / pa.2 + t / pb.2);
                if x < 0.0 || y < 0.0 || x as usize >= W || y as usize >= H {
                    continue;
                }
                let k = y as usize * W + x as usize;
                if z <= zb[k] + 0.03 {
                    img[k] = col;
                }
            }
        }
    }
    img
}

fn write_png(img: &[[u8; 3]], (w, h): (usize, usize), path: &str) {
    let f = std::fs::File::create(path).unwrap();
    let mut e = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    let mut wr = e.write_header().unwrap();
    wr.write_image_data(&img.concat()).unwrap();
}

/// Render the snapshots in parallel: before | after, one PNG each.
fn render(before: &ShowcaseVisual, after: &ShowcaseVisual, snaps: &[Snap], dir: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get()).min(12);
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if k >= snaps.len() {
                    break;
                }
                // ONLY=a,b: just those frames (a still check).
                if let Ok(o) = std::env::var("ONLY") {
                    if !o.split(',').any(|x| x.parse() == Ok(k)) {
                        continue;
                    }
                }
                let path = format!("{dir}/f{k:05}.png");
                if std::path::Path::new(&path).exists() {
                    continue;
                }
                let a = raster(before, &snaps[k]);
                let b = raster(after, &snaps[k]);
                let mut img = Vec::with_capacity(2 * W * H);
                for y in 0..H {
                    img.extend_from_slice(&a[y * W..(y + 1) * W]);
                    // A 4-pixel divider.
                    img.extend_from_slice(&b[y * W..(y + 1) * W]);
                }
                for y in 0..H {
                    for x in W - 2..W + 2 {
                        img[y * 2 * W + x] = [200, 200, 200];
                    }
                }
                write_png(&img, (2 * W, H), &path);
            });
        }
    });
    eprintln!("{dir}: {} frames", snaps.len());
}

/// An orbit round `c` at `dist`, `n` frames, then the same with the skeleton.
fn orbit(s: &ExplicitBeams, c: Vec3, dist: f32, height: f32, from: Vec3, n: usize, out: &mut Vec<Snap>) {
    for overlay in [false, true] {
        for k in 0..n {
            let a = std::f32::consts::TAU * k as f32 / n as f32;
            let d = Quat::from_rotation_y(a) * Vec3::new(from.x, 0.0, from.z).normalize();
            out.push(snap(s, Cam { eye: c + d * dist + Vec3::Y * height, at: c }, overlay));
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let clip = a.get(1).map(String::as_str).unwrap_or("wall70");
    let out = a.get(2).map(String::as_str).unwrap_or("out/skin-film-v2");
    let b = build(&style("sedan").unwrap(), &BuildParams { resolution: Resolution::Medium, ..Default::default() }).unwrap();
    let mut after = showcase_visual_lod(&b, 1.0).unwrap();
    raycast_visual(&mut after, &b.truck).unwrap();
    let mut before = after.clone();
    before.frames.mode = Binding::Stretch;
    let mut snaps = Vec::new();
    match clip {
        "wall70" => {
            let vel = Vec3::new(0.0, 0.0, 70.0 / 3.6);
            let mut s = spawn(&b, Vec3::ZERO, 0.0, vel);
            let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
            s.boxes.push(StaticBox::centered(Vec3::new(0.0, 1.5, front + 0.9), Vec3::new(8.0, 1.5, 0.3)));
            // Close on the grille and the bonnet's front seam, from ahead-left
            // and above, following the car; the wall is out of the way to the
            // right of the view.
            let close = |s: &ExplicitBeams| {
                // The front: the 97th percentile of the nodes' z (a detached
                // node left at the wall is not the car).
                let mut z: Vec<f32> = s.positions().iter().map(|q| q.z).collect();
                z.sort_by(f32::total_cmp);
                let fz = z[z.len() * 97 / 100];
                let at = Vec3::new(0.0, 0.75, fz - 0.45);
                Cam { eye: at + Vec3::new(-2.9, 1.5, 2.6), at }
            };
            // 0.1 s run-up real time, then the impact at 1/240 s a frame
            // (0.5 s), then 1 s real time.
            for _ in 0..6 {
                s.frame();
                snaps.push(snap(&s, close(&s), false));
            }
            for _ in 0..120 {
                for _ in 0..((1.0 / 240.0) / STEP).round() as usize {
                    s.step();
                }
                snaps.push(snap(&s, close(&s), false));
            }
            for _ in 0..60 {
                s.frame();
                snaps.push(snap(&s, close(&s), false));
            }
            let c = centroid(&s.positions()) * Vec3::new(1.0, 0.0, 1.0) + Vec3::Y * 0.7;
            orbit(&s, c, 7.2, 2.0, Vec3::new(-0.6, 0.0, 1.0), 180, &mut snaps);
        }
        "side60" => {
            let struck = spawn(&b, Vec3::ZERO, 0.0, Vec3::ZERO);
            let p0 = struck.positions();
            let rear = b.skeleton.parts.iter().position(|p| p == "door_r_l").unwrap() as u16;
            let door: Vec<Vec3> = (0..b.skeleton.nodes.len()).filter(|&i| b.skeleton.nodes[i].part == rear).map(|i| p0[i]).collect();
            let zd = door.iter().map(|p| p.z).sum::<f32>() / door.len() as f32;
            let hw = p0.iter().map(|p| p.x).fold(f32::MIN, f32::max);
            let striker = spawn(&b, Vec3::new(hw + 0.5 * b.def.profile.length + 0.3, 0.0, zd), -90.0, Vec3::new(-60.0 / 3.6, 0.0, 0.0));
            let mut w = ExplicitBeamsWorld::new(vec![struck, striker]);
            // The struck car's rear door from its left-rear, above: the
            // striker comes in from the right of the view.
            let cam = |w: &ExplicitBeamsWorld| {
                let c = centroid(&w.vehicles[0].positions());
                let at = Vec3::new(c.x + 0.5, 0.75, zd);
                Cam { eye: at + Vec3::new(3.6, 1.6, -3.4), at }
            };
            for _ in 0..4 {
                w.frame_60();
                snaps.push(snap(&w.vehicles[0], cam(&w), false));
            }
            for _ in 0..120 {
                for _ in 0..((1.0 / 240.0) / STEP).round() as usize {
                    w.step();
                }
                snaps.push(snap(&w.vehicles[0], cam(&w), false));
            }
            for _ in 0..60 {
                w.frame_60();
                snaps.push(snap(&w.vehicles[0], cam(&w), false));
            }
            let c = centroid(&w.vehicles[0].positions()) * Vec3::new(1.0, 0.0, 1.0) + Vec3::Y * 0.7;
            orbit(&w.vehicles[0], c, 7.2, 2.0, Vec3::new(1.0, 0.0, -0.8), 180, &mut snaps);
        }
        _ => {
            // The page's grab (50 kN/m): per part, its outermost node pulled
            // 0.5, 2 and 10 m outward over 0.5 s, held 0.5 s, released 1 s.
            for part in ["door_f_l", "fender_l", "bumper_f"] {
                let mut s = spawn(&b, Vec3::ZERO, 0.0, Vec3::ZERO);
                s.run(0.3);
                let pi = b.skeleton.parts.iter().position(|p| p == part).unwrap() as u16;
                let p = s.positions();
                let c = centroid(&p);
                let node = (0..b.skeleton.nodes.len())
                    .filter(|&i| b.skeleton.nodes[i].part == pi)
                    .max_by(|&x, &y| (p[x] - c).length().total_cmp(&(p[y] - c).length()))
                    .unwrap();
                let dirn = ((p[node] - c) * Vec3::new(1.0, 0.3, 1.0)).normalize();
                let k = 5.0e4;
                let m = s.nodes[node].mass;
                let cd = (2.0 * (k * m).sqrt()).min(0.5 * m / STEP);
                for dist in [0.5f32, 2.0, 10.0] {
                    let start = s.positions()[node];
                    // The panel close up, from outside it and above; pulled
                    // far, the view widens to keep the car and the panel.
                    let side = dirn.cross(Vec3::Y).normalize();
                    for f in 0..120u32 {
                        s.grab = (f < 60).then(|| Grab { node: node as u32, target: start + dirn * dist * (f as f32 / 30.0).min(1.0), k, c: cd });
                        s.frame();
                        let at = if dist > 5.0 { c + dirn * 3.0 + Vec3::Y * 0.3 } else { p[node] * 0.6 + c * 0.4 + Vec3::Y * 0.2 };
                        let back = if dist > 5.0 { 11.0 } else { 4.2 };
                        let eye = at + (dirn * 0.8 + side * 0.55 + Vec3::Y * 0.45).normalize() * back;
                        snaps.push(snap(&s, Cam { eye, at }, false));
                    }
                }
                s.grab = None;
            }
        }
    }
    render(&before, &after, &snaps, &format!("{out}/{clip}"));
}
