//! Side-by-side wireframe video of two traces of the same `.truck` scene (e.g. a
//! reference simulator and Rubble's explicit beams), through ffmpeg.
//!
//! ```bash
//! cargo run -p engine-softbody --release --example compare_video -- <file.truck> \
//!   <left.trace> "Reference" <right.trace> "Rubble (GPU)" box=cx,cy,cz,hx,hy,hz out=crash.mp4
//! ```
//! Traces: u32 nodes, u32 beams, then per sample the node positions (f32 xyz) and the
//! beams' broken flags (u8), in the node and beam order of
//! `ExplicitBeams::from_truck` (`explicit_beams_check trace=` writes this).
//! Beams are grey, turning red as they shorten or stretch from spawn; broken beams
//! disappear. `fps=` the output rate (default 30: 250 Hz traces play 8× slowed),
//! `yaw=`/`pitch=` the camera (degrees).
use engine_softbody::explicit_beams::ExplicitBeams;
use glam::{Mat3, Vec3};
use std::io::{Read, Write};

const W: usize = 960;
const H: usize = 720;

struct Trace {
    nodes: usize,
    beams: usize,
    frames: Vec<(Vec<Vec3>, Vec<u8>)>,
}

fn load(path: &str) -> Trace {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    let u = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
    let (nodes, beams) = (u(0), u(4));
    let frame = nodes * 12 + beams;
    let mut frames = Vec::new();
    let mut at = 8;
    while at + frame <= bytes.len() {
        let p: Vec<Vec3> = (0..nodes)
            .map(|n| {
                let f = |k: usize| {
                    f32::from_le_bytes(
                        bytes[at + n * 12 + k * 4..at + n * 12 + k * 4 + 4]
                            .try_into()
                            .unwrap(),
                    )
                };
                Vec3::new(f(0), f(1), f(2))
            })
            .collect();
        let b = bytes[at + nodes * 12..at + frame].to_vec();
        frames.push((p, b));
        at += frame;
    }
    Trace {
        nodes,
        beams,
        frames,
    }
}

fn kv<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().find_map(|a| a.strip_prefix(&format!("{key}=")))
}

fn line(img: &mut [u8], x0: f32, y0: f32, x1: f32, y1: f32, c: [u8; 3], ox: usize) {
    let n = ((x1 - x0).abs().max((y1 - y0).abs()).ceil() as usize).max(1);
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let (x, y) = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
        if x < 0.0 || y < 0.0 || x >= W as f32 || y >= H as f32 {
            continue;
        }
        let k = (y as usize * W * 2 + ox + x as usize) * 3;
        img[k..k + 3].copy_from_slice(&c);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let text = std::fs::read_to_string(&args[0]).unwrap();
    let sim = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, Default::default())
        .expect("spawn");
    let edges: Vec<(usize, usize)> = sim
        .beams
        .iter()
        .map(|b| (b.a as usize, b.b as usize))
        .collect();
    let (left, right) = (load(&args[1]), load(&args[3]));
    assert_eq!(
        (left.nodes, left.beams),
        (sim.nodes.len(), sim.beams.len()),
        "left trace topology"
    );
    assert_eq!(
        (right.nodes, right.beams),
        (sim.nodes.len(), sim.beams.len()),
        "right trace topology"
    );
    let labels = (&args[2], &args[4]);
    let yaw = kv(&args, "yaw")
        .map_or(-35.0f32, |v| v.parse().unwrap())
        .to_radians();
    let pitch = kv(&args, "pitch")
        .map_or(20.0f32, |v| v.parse().unwrap())
        .to_radians();
    let rot = Mat3::from_rotation_x(pitch) * Mat3::from_rotation_y(yaw);
    let wall = kv(&args, "box").map(|b| {
        let c: Vec<f32> = b.split(',').map(|v| v.parse().unwrap()).collect();
        (Vec3::new(c[0], c[1], c[2]), Vec3::new(c[3], c[4], c[5]))
    });
    // Frame the whole run: every node of every sample, the wall's near face.
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for t in [&left, &right] {
        for (p, _) in t.frames.iter().step_by(4) {
            for q in p {
                let v = rot * *q;
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    let centre = 0.5 * (lo + hi);
    let scale = 0.85 * (W as f32 / (hi.x - lo.x)).min(H as f32 / (hi.y - lo.y));
    let project = |p: Vec3| {
        let v = rot * p - centre;
        (W as f32 * 0.5 + v.x * scale, H as f32 * 0.5 - v.y * scale)
    };
    let grid = left.frames[0].0.iter().fold(Vec3::ZERO, |a, p| a + *p) / left.nodes as f32;
    let grid = Vec3::new(grid.x.round(), 0.0, grid.z.round());
    let rest: Vec<f32> = edges
        .iter()
        .map(|&(a, b)| {
            (left.frames[0].0[a] - left.frames[0].0[b])
                .length()
                .max(1e-4)
        })
        .collect();
    let out = kv(&args, "out").unwrap_or("crash.mp4");
    let fps = kv(&args, "fps").unwrap_or("30");
    let font = "/System/Library/Fonts/Helvetica.ttc";
    let draw = format!(
        "drawtext=fontfile={font}:text='{}':x=20:y=20:fontsize=30:fontcolor=white,\
         drawtext=fontfile={font}:text='{}':x={}+20:y=20:fontsize=30:fontcolor=white,\
         drawtext=fontfile={font}:text='t = %{{eif\\:n*4\\:d}} ms   (8x slow motion · red = beam strain)':x=20:y=h-50:fontsize=24:fontcolor=gray",
        labels.0.replace(':', "\\:"),
        labels.1.replace(':', "\\:"),
        W
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
        .args(["-s", &format!("{}x{}", W * 2, H), "-r", fps, "-i", "-"])
        .args([
            "-vf", &draw, "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18", out,
        ])
        .stdin(std::process::Stdio::piped())
        .spawn()
        .expect("ffmpeg");
    let stdin = ff.stdin.as_mut().unwrap();
    let frames = left.frames.len().min(right.frames.len());
    for f in 0..frames {
        let mut img = vec![18u8; W * 2 * H * 3];
        for (side, t) in [(0usize, &left), (W, &right)] {
            let (p, broken) = &t.frames[f];
            // Ground grid (1 m) around the scene.
            for k in -30..=30 {
                let (gx, gz) = (grid.x + k as f32, grid.z + k as f32);
                let (a, b) = (
                    project(Vec3::new(gx, 0.0, grid.z - 30.0)),
                    project(Vec3::new(gx, 0.0, grid.z + 30.0)),
                );
                line(&mut img, a.0, a.1, b.0, b.1, [40, 40, 46], side);
                let (a, b) = (
                    project(Vec3::new(grid.x - 30.0, 0.0, gz)),
                    project(Vec3::new(grid.x + 30.0, 0.0, gz)),
                );
                line(&mut img, a.0, a.1, b.0, b.1, [40, 40, 46], side);
            }
            if let Some((c, h)) = wall {
                for i in 0..8u32 {
                    for j in 0..3 {
                        let a = Vec3::new(
                            if i & 1 == 0 { -1.0 } else { 1.0 },
                            if i & 2 == 0 { -1.0 } else { 1.0 },
                            if i & 4 == 0 { -1.0 } else { 1.0 },
                        );
                        let mut b = a;
                        b[j] = -b[j];
                        if a[j] < 0.0 {
                            let (pa, pb) = (project(c + a * h), project(c + b * h));
                            line(&mut img, pa.0, pa.1, pb.0, pb.1, [70, 120, 220], side);
                        }
                    }
                }
            }
            for (k, &(a, b)) in edges.iter().enumerate() {
                if broken[k] != 0 {
                    continue;
                }
                let strain = (((p[a] - p[b]).length() / rest[k]) - 1.0).abs();
                let r = (strain * 8.0).min(1.0);
                let c = [
                    (170.0 + 85.0 * r) as u8,
                    (170.0 * (1.0 - r)) as u8,
                    (170.0 * (1.0 - r)) as u8,
                ];
                let (pa, pb) = (project(p[a]), project(p[b]));
                line(&mut img, pa.0, pa.1, pb.0, pb.1, c, side);
            }
        }
        // Divider.
        for y in 0..H {
            let k = (y * W * 2 + W) * 3;
            img[k..k + 3].copy_from_slice(&[90, 90, 90]);
        }
        stdin.write_all(&img).unwrap();
    }
    drop(ff.stdin.take());
    ff.wait().unwrap();
    println!("{frames} frames → {out}");
}
