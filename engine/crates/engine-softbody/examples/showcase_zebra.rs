//! The showcase car's body surface under zebra light: a software render of the
//! visual pieces at rest (no skeleton, no GPU), the paint shading as reflected
//! stripes and everything else dark grey. Smooth, continuous stripes mean a
//! smooth surface; a wobble or kink in the surface shows as one in them.
//!
//! `cargo run --release -p engine-softbody --example showcase_zebra -- OUTDIR [view,...]`
//!
//! Views (as `showcase-car.html`'s `cam`/`dist`/`look`, the stripes' axis
//! last: 1 along the car, 2 across it, 3 level bands): see `VIEWS`.
//! `LOD=0.45` renders the distant level; `SHADE=1` a plain studio shade;
//! `RAW=1` the uncut body surface tinted by patch (band blue, ramp green,
//! top red), `WIRE=1` with its triangles' edges.

use engine_softbody::authoring::parts::body::BodySurface;
use engine_softbody::authoring::parts::design::{assemble, sedan_styling};
use engine_softbody::authoring::parts::with_detail;
use engine_softbody::authoring::sedan::sedan;
use glam::Vec3;

type View = (&'static str, [f32; 3], f32, [f32; 3], u8);

const VIEWS: &[View] = &[
    ("three", [0.8, 0.32, 0.75], 5.6, [0.0, 0.0, 0.0], 1),
    ("bonnet", [0.25, 0.95, 1.0], 2.9, [0.0, 0.2, 1.35], 1),
    ("apillar", [0.95, 0.55, 0.65], 2.4, [0.55, 0.3, 0.85], 1),
    ("side", [1.0, 0.12, 0.05], 5.6, [0.0, -0.15, 0.0], 1),
    ("side_bands", [1.0, 0.12, 0.05], 5.6, [0.0, -0.15, 0.0], 3),
    ("cpillar", [0.75, 0.55, -0.35], 2.3, [0.6, 0.45, -1.0], 1),
    ("deck", [-0.3, 0.95, -1.0], 2.7, [0.0, 0.25, -1.6], 2),
    ("nose", [0.35, 0.12, 1.0], 3.0, [0.0, -0.1, 1.6], 3),
    ("tail", [-0.35, 0.15, -1.0], 3.0, [0.0, -0.05, -1.6], 3),
    ("rear", [-0.6, 0.35, -0.8], 6.2, [0.0, 0.0, 0.0], 1),
    ("top", [0.3, 1.0, 0.3], 5.5, [0.0, 0.0, 0.0], 1),
    ("plan_front", [0.0, 1.0, 0.02], 2.2, [0.0, 0.2, 1.6], 1),
    ("plan_rear", [0.0, 1.0, -0.02], 2.2, [0.0, 0.3, -1.4], 1),
    ("corner", [0.7, 0.35, 0.7], 1.6, [0.6, 0.0, 2.0], 1),
    ("dome", [0.0, 1.0, 0.02], 0.9, [0.38, 0.25, 1.5], 1),
    ("face", [0.25, 0.08, 1.0], 2.2, [0.0, -0.15, 2.0], 1),
    ("dlo", [1.0, 0.5, -0.3], 0.7, [0.85, 0.42, -1.2], 1),
    ("tail_corner", [0.7, 0.4, -0.7], 1.7, [0.6, 0.15, -2.0], 1),
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).cloned().unwrap_or_else(|| "zebra".into());
    let only: Option<Vec<String>> = args
        .get(2)
        .map(|s| s.split(',').map(|x| x.to_string()).collect());
    std::fs::create_dir_all(&out).unwrap();
    let def = sedan();
    let lod: f32 = std::env::var("LOD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let t0 = std::time::Instant::now();
    let pieces = with_detail(lod, || assemble(&def, &sedan_styling(&def)));
    eprintln!("assemble: {:.0} ms", t0.elapsed().as_secs_f32() * 1e3);
    let (mut pos, mut nrm, mut mat, mut idx) = (vec![], vec![], vec![], vec![]);
    let raw = std::env::var("RAW").is_ok();
    if raw {
        let st = sedan_styling(&def);
        let mut bs = st.body.clone();
        bs.grille = st.grille.clone();
        let s = with_detail(lod, || BodySurface::build(&def.profile, &bs));
        pos = s.mesh.pos.clone();
        nrm = s.mesh.nrm.clone();
        mat = s.patch.iter().map(|p| 100 + p).collect();
        idx = s.mesh.idx.clone();
    }
    for p in pieces.iter().filter(|_| !raw) {
        if p.name.starts_with("wheel") || p.name.starts_with("caliper") {
            continue;
        }
        let base = pos.len() as u32;
        pos.extend_from_slice(&p.mesh.pos);
        nrm.extend_from_slice(&p.mesh.nrm);
        mat.extend_from_slice(&p.mesh.mat);
        idx.extend(p.mesh.idx.iter().map(|i| i + base));
    }
    eprintln!("{} vertices, {} triangles", pos.len(), idx.len() / 3);
    let centre = Vec3::new(0.0, 0.6, 0.5 * def.profile.length);
    let shade = std::env::var("SHADE").is_ok();
    for &(name, cam, dist, look, axis) in VIEWS {
        if let Some(o) = &only {
            if !o.iter().any(|x| x == name) {
                continue;
            }
        }
        let target = centre + Vec3::from_array(look);
        let eye = target + Vec3::from_array(cam).normalize() * dist;
        let path = format!("{out}/zebra_{name}.png");
        render(&path, &pos, &nrm, &mat, &idx, eye, target, axis, shade);
        eprintln!("{path}");
    }
}

#[allow(clippy::too_many_arguments)]
fn render(
    path: &str,
    pos: &[Vec3],
    nrm: &[Vec3],
    mats: &[u8],
    idx: &[u32],
    eye: Vec3,
    target: Vec3,
    axis: u8,
    shade: bool,
) {
    let wire = std::env::var("WIRE").is_ok();
    // 2× supersampled, then box-filtered down.
    let (ow, oh) = (1600usize, 900usize);
    let (w, h) = (ow * 2, oh * 2);
    let mut img = vec![Vec3::new(0.32, 0.33, 0.35); w * h];
    let mut zb = vec![f32::MAX; w * h];
    let fwd = (target - eye).normalize();
    let right = fwd.cross(Vec3::Y).normalize();
    let up = right.cross(fwd);
    let f = h as f32 / 2.0 / (0.75f32 / 2.0).tan();
    let proj = |p: Vec3| -> (f32, f32, f32) {
        let d = p - eye;
        let z = d.dot(fwd);
        (
            w as f32 / 2.0 + d.dot(right) / z * f,
            h as f32 / 2.0 - d.dot(up) / z * f,
            z,
        )
    };
    for t in idx.chunks(3) {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let (pa, pb, pc) = (proj(pos[a]), proj(pos[b]), proj(pos[c]));
        if pa.2 < 0.1 || pb.2 < 0.1 || pc.2 < 0.1 {
            continue;
        }
        let minx = pa.0.min(pb.0).min(pc.0).floor().max(0.0) as usize;
        let maxx = pa.0.max(pb.0).max(pc.0).ceil().min(w as f32 - 1.0) as usize;
        let miny = pa.1.min(pb.1).min(pc.1).floor().max(0.0) as usize;
        let maxy = pa.1.max(pb.1).max(pc.1).ceil().min(h as f32 - 1.0) as usize;
        if maxx < minx || maxy < miny {
            continue;
        }
        let area = (pb.0 - pa.0) * (pc.1 - pa.1) - (pc.0 - pa.0) * (pb.1 - pa.1);
        if area.abs() < 1e-12 {
            continue;
        }
        for y in miny..=maxy {
            for x in minx..=maxx {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let w0 = ((pb.0 - px) * (pc.1 - py) - (pc.0 - px) * (pb.1 - py)) / area;
                let w1 = ((pc.0 - px) * (pa.1 - py) - (pa.0 - px) * (pc.1 - py)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                // Perspective-correct weights.
                let (q0, q1, q2) = (w0 / pa.2, w1 / pb.2, w2 / pc.2);
                let s = q0 + q1 + q2;
                let (b0, b1, b2) = (q0 / s, q1 / s, q2 / s);
                let z = 1.0 / s;
                let k = y * w + x;
                if z >= zb[k] {
                    continue;
                }
                zb[k] = z;
                let p = pos[a] * b0 + pos[b] * b1 + pos[c] * b2;
                let mut n = (nrm[a] * b0 + nrm[b] * b1 + nrm[c] * b2).normalize_or_zero();
                let v = (eye - p).normalize();
                if n.dot(v) < 0.0 {
                    n = -n;
                }
                let r = 2.0 * n.dot(v) * n - v;
                let m = mats[a];
                let tint = match m {
                    100 => Vec3::new(0.75, 0.85, 1.0),
                    101 => Vec3::new(0.75, 1.0, 0.75),
                    102 => Vec3::new(1.0, 0.75, 0.75),
                    _ => Vec3::ONE,
                };
                let m = if m >= 100 { 0 } else { m };
                let edge = if wire && b0.min(b1).min(b2) < 0.03 {
                    0.55
                } else {
                    1.0
                };
                img[k] = edge
                    * tint
                    * if m == 0 {
                        if shade {
                            let l = Vec3::new(0.3, 0.9, 0.3).normalize();
                            let sky = 0.25 + 0.75 * (0.5 + 0.5 * r.y).powf(3.0);
                            Vec3::new(0.55, 0.05, 0.05) * (0.25 + 0.6 * n.dot(l).max(0.0))
                                + Vec3::splat(sky * 0.35 + r.dot(l).max(0.0).powf(300.0))
                        } else {
                            let c = match axis {
                                1 => r.y.atan2(r.x) * 9.0 / std::f32::consts::PI,
                                2 => r.y.atan2(r.z) * 9.0 / std::f32::consts::PI,
                                _ => r.y.clamp(-1.0, 1.0).asin() * 18.0 / std::f32::consts::PI,
                            };
                            let band = if (c - c.floor()) < 0.5 { 0.92 } else { 0.05 };
                            Vec3::splat(band)
                        }
                    } else if m == 1 || m == 25 || m == 27 {
                        Vec3::new(0.08, 0.09, 0.1)
                    } else {
                        Vec3::splat(0.18 + 0.1 * n.y.max(0.0))
                    };
            }
        }
    }
    let mut out = format!("P6\n{ow} {oh}\n255\n").into_bytes();
    for y in 0..oh {
        for x in 0..ow {
            let mut c = Vec3::ZERO;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                c += img[(y * 2 + dy) * w + x * 2 + dx];
            }
            let c = (c * 0.25).powf(1.0 / 2.2);
            for ch in c.to_array() {
                out.push((ch.clamp(0.0, 1.0) * 255.0) as u8);
            }
        }
    }
    let ppm = format!("{path}.ppm");
    std::fs::write(&ppm, out).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i", &ppm, path])
        .status();
    let _ = std::fs::remove_file(&ppm);
}
