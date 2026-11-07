//! Side-by-side wireframe video of a car on the explicit 2 kHz step (left) and
//! in the AVBD vehicle mode (right), same scene, same start
//! (`engine/docs/avbd-vehicles.md`), through ffmpeg, plus stills.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example avbd_video -- \
//!     --car etk_kc6 --config 120x50 --scene wall|pole|bump [--seconds 1.2] \
//!     [--out ~/Development/rubble-artifacts/avbd-vehicles]
//! ```
//!
//! Frames every 4 ms of simulated time (8× slow motion at 30 fps). The AVBD side
//! holds its state between its own steps (a 120 Hz step shows on two frames).
//! Beams grey, turning red with strain from spawn; broken beams disappear.
//! Stills (PNG) at the start, at 60 ms, at 150 ms and at the end.
use engine_softbody::avbd_beams::{AvbdSettings, AvbdState};
use engine_softbody::explicit_beams::{
    ExplicitBeams, StaticBox, StaticShape, NODE_IMMOVABLE, STEP,
};
use engine_softbody::vehicle_compare::{Car, Manifest};
use glam::{Mat3, Vec3};
use std::io::Write;
use std::path::PathBuf;

const W: usize = 800;
const H: usize = 600;

fn line(img: &mut [u8], x0: f32, y0: f32, x1: f32, y1: f32, c: [u8; 3], ox: usize) {
    let n = ((x1 - x0).abs().max((y1 - y0).abs()).ceil() as usize).clamp(1, 4000);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let (x, y) = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
        if !(x >= 0.0 && y >= 0.0 && x < W as f32 && y < H as f32) {
            continue;
        }
        let k = (y as usize * W * 2 + ox + x as usize) * 3;
        img[k..k + 3].copy_from_slice(&c);
    }
}

fn moving(s: &mut ExplicitBeams, v: Vec3) {
    for n in &mut s.nodes {
        n.vel = if n.flags & NODE_IMMOVABLE != 0 {
            Vec3::ZERO
        } else {
            v
        };
    }
    s.roll_tyres(v);
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| {
        a.iter()
            .position(|x| x == k)
            .and_then(|i| a.get(i + 1).cloned())
    };
    let car_name = opt("--car").unwrap_or("etk_kc6".into());
    let cfg = opt("--config").unwrap_or("120x50".into());
    let scene = opt("--scene").unwrap_or("wall".into());
    let seconds: f32 = opt("--seconds").map_or(1.2, |s| s.parse().unwrap());
    let out = opt("--out").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("HOME").unwrap())
            .join("Development/rubble-artifacts/avbd-vehicles")
    });
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let (hz, it) = cfg.split_once('x').ok_or("config <hz>x<iters>")?;
    let set = AvbdSettings {
        hz: hz.parse().unwrap(),
        iterations: it.parse().unwrap(),
        ..Default::default()
    };
    let manifest =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/avbd-vehicles-cars.json");
    let m: Manifest =
        serde_json::from_str(&std::fs::read_to_string(&manifest).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let spec = m
        .cars
        .iter()
        .find(|c| c.name == car_name)
        .ok_or("no such car")?;
    let car = Car::load(spec)?;
    let mut s = car.template.clone();
    let f = car.frame;
    let mut boxes: Vec<StaticBox> = Vec::new();
    let mut yaw = -35.0f32;
    match scene.as_str() {
        "wall" => {
            moving(&mut s, Vec3::Z * 56.0 / 3.6);
            let front = car
                .skin
                .iter()
                .map(|&i| s.nodes[i as usize].pos.z)
                .fold(f32::MIN, f32::max);
            let wz = front + 0.05;
            let b = StaticBox::centered(
                Vec3::new(0.0, 2.0, wz + 0.5),
                Vec3::new(0.5 * f.width() + 1.0, 2.0, 0.5),
            );
            s.boxes.push(b);
            boxes.push(b);
            yaw = -60.0;
        }
        "pole" => {
            let ang = 75.0f32.to_radians();
            let dir = Vec3::new(ang.sin(), 0.0, ang.cos());
            moving(&mut s, dir * 32.0 / 3.6);
            let r = 0.127;
            let mid = Vec3::new(
                0.0,
                0.0,
                0.5 * (f.front_axle + f.rear_axle) + 0.25 * (f.front_axle - f.rear_axle),
            );
            let gap = |c: Vec3| {
                s.nodes
                    .iter()
                    .map(|n| Vec3::new(n.pos.x - c.x, 0.0, n.pos.z - c.z).length())
                    .fold(f32::MAX, f32::min)
            };
            let mut t = 0.0f32;
            while gap(mid + dir * t) < r + 0.03 {
                t += 0.005;
            }
            let c = mid + dir * t + Vec3::Y * 2.0;
            s.shapes.push(StaticShape::cylinder(c, r, 2.0));
            boxes.push(StaticBox::centered(c, Vec3::new(r, 2.0, r)));
            yaw = 20.0;
        }
        "bump" => {
            for q in &mut s.nodes {
                q.vel = Vec3::ZERO;
                q.vel.y = -1.0;
            }
        }
        _ => return Err(format!("unknown scene {scene}")),
    }
    let mut r = s.clone();
    r.avbd = Some(Box::new(AvbdState::new(&r, set)?));
    let edges: Vec<(usize, usize)> = s
        .beams
        .iter()
        .filter(|b| {
            use engine_softbody::explicit_beams::ExplicitKind as K;
            !matches!(
                b.kind,
                K::JbeamAeroTri
                    | K::JbeamAeroTri2
                    | K::JbeamTorsion
                    | K::JbeamTorsionPair
                    | K::Trigger
            ) && b.a != b.b
        })
        .map(|b| (b.a as usize, b.b as usize))
        .collect();
    let keep: Vec<usize> = s
        .beams
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            use engine_softbody::explicit_beams::ExplicitKind as K;
            !matches!(
                b.kind,
                K::JbeamAeroTri
                    | K::JbeamAeroTri2
                    | K::JbeamTorsion
                    | K::JbeamTorsionPair
                    | K::Trigger
            ) && b.a != b.b
        })
        .map(|(j, _)| j)
        .collect();
    let rest: Vec<f32> = edges
        .iter()
        .map(|&(a, b)| (s.nodes[a].pos - s.nodes[b].pos).length().max(1e-4))
        .collect();
    let per = 8usize; // 4 ms
    let ticks = (seconds / STEP).round() as usize;
    let mut frames: Vec<[(Vec<Vec3>, Vec<bool>); 2]> = Vec::new();
    let snap = |s: &ExplicitBeams| -> (Vec<Vec3>, Vec<bool>) {
        (
            s.nodes.iter().map(|n| n.pos + s.origin).collect(),
            keep.iter().map(|&j| s.beams[j].broken).collect(),
        )
    };
    frames.push([snap(&s), snap(&r)]);
    for k in 0..ticks {
        s.step();
        r.step();
        if (k + 1) % per == 0 {
            frames.push([snap(&s), snap(&r)]);
        }
    }
    let pitch = 18.0f32.to_radians();
    let rot = Mat3::from_rotation_x(pitch) * Mat3::from_rotation_y(yaw.to_radians());
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    // Framed on the explicit car's start and end (parts thrown far do not shrink it).
    for fr in [&frames[0], &frames[frames.len() - 1]] {
        for q in fr[0].0.iter() {
            if q.is_finite() {
                let v = rot * *q;
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    let centre = 0.5 * (lo + hi);
    let scale = 0.7 * (W as f32 / (hi.x - lo.x)).min(H as f32 / (hi.y - lo.y));
    let project = |p: Vec3| {
        let v = rot * p - centre;
        (W as f32 * 0.5 + v.x * scale, H as f32 * 0.5 - v.y * scale)
    };
    let stem = format!("{car_name}-{scene}-explicit-vs-avbd{cfg}");
    let mp4 = out.join(format!("{stem}.mp4"));
    let font = "/System/Library/Fonts/Helvetica.ttc";
    let draw = format!(
        "drawtext=fontfile={font}:text='explicit 2 kHz (reference)':x=20:y=20:fontsize=26:fontcolor=white,\
         drawtext=fontfile={font}:text='AVBD {} Hz x {} sweeps':x={}+20:y=20:fontsize=26:fontcolor=white,\
         drawtext=fontfile={font}:text='{car_name} {scene}   t = %{{eif\\:n*4\\:d}} ms   (8x slow motion, red = beam strain)':x=20:y=h-40:fontsize=20:fontcolor=gray",
        set.hz, set.iterations, W
    );
    let mut ff = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
        ])
        .args(["-s", &format!("{}x{}", W * 2, H), "-r", "30", "-i", "-"])
        .args([
            "-vf", &draw, "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18",
        ])
        .arg(&mp4)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    let stdin = ff.stdin.as_mut().unwrap();
    for fr in &frames {
        let mut img = vec![18u8; W * 2 * H * 3];
        for (side, (p, broken)) in [(0usize, &fr[0]), (W, &fr[1])] {
            for k in -20..=20 {
                let g = k as f32;
                let (a0, b0) = (
                    project(Vec3::new(g, 0.0, -20.0)),
                    project(Vec3::new(g, 0.0, 20.0)),
                );
                line(&mut img, a0.0, a0.1, b0.0, b0.1, [40, 40, 46], side);
                let (a0, b0) = (
                    project(Vec3::new(-20.0, 0.0, g)),
                    project(Vec3::new(20.0, 0.0, g)),
                );
                line(&mut img, a0.0, a0.1, b0.0, b0.1, [40, 40, 46], side);
            }
            for b in &boxes {
                let (c, h) = (0.5 * (b.lo + b.hi), 0.5 * (b.hi - b.lo));
                for i in 0..8u32 {
                    for j in 0..3 {
                        let a0 = Vec3::new(
                            if i & 1 == 0 { -1.0 } else { 1.0 },
                            if i & 2 == 0 { -1.0 } else { 1.0 },
                            if i & 4 == 0 { -1.0 } else { 1.0 },
                        );
                        let mut b0 = a0;
                        b0[j] = -b0[j];
                        if a0[j] < 0.0 {
                            let (pa, pb) = (project(c + a0 * h), project(c + b0 * h));
                            line(&mut img, pa.0, pa.1, pb.0, pb.1, [70, 120, 220], side);
                        }
                    }
                }
            }
            for (k, &(a0, b0)) in edges.iter().enumerate() {
                if broken[k] || !p[a0].is_finite() || !p[b0].is_finite() {
                    continue;
                }
                let strain = (((p[a0] - p[b0]).length() / rest[k]) - 1.0).abs();
                let rr = (strain * 8.0).min(1.0);
                let c = [
                    (170.0 + 85.0 * rr) as u8,
                    (170.0 * (1.0 - rr)) as u8,
                    (170.0 * (1.0 - rr)) as u8,
                ];
                let (pa, pb) = (project(p[a0]), project(p[b0]));
                line(&mut img, pa.0, pa.1, pb.0, pb.1, c, side);
            }
        }
        for y in 0..H {
            let k = (y * W * 2 + W) * 3;
            img[k..k + 3].copy_from_slice(&[90, 90, 90]);
        }
        stdin.write_all(&img).map_err(|e| e.to_string())?;
    }
    drop(ff.stdin.take());
    ff.wait().map_err(|e| e.to_string())?;
    // Stills: frame k is at 4k ms.
    let last = frames.len() - 1;
    for (name, k) in [("0ms", 0usize), ("60ms", 15), ("150ms", 37), ("end", last)] {
        let png = out.join(format!("{stem}-{name}.png"));
        let st = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"])
            .arg(&mp4)
            .args([
                "-vf",
                &format!("select=eq(n\\,{})", k.min(last)),
                "-vframes",
                "1",
            ])
            .arg(&png)
            .status()
            .map_err(|e| e.to_string())?;
        if !st.success() {
            return Err("ffmpeg still".into());
        }
    }
    println!("{} frames -> {}", frames.len(), mp4.display());
    Ok(())
}
