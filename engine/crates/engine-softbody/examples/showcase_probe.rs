//! Build a showcase vehicle and run a scene on the explicit CPU reference.
//! `cargo run --release -p engine-softbody --example showcase_probe -- <style> <res> <scene> [speed] [offset]`
use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, StaticBox, StaticShape,
};
use glam::Vec3;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let st = a.get(1).map(String::as_str).unwrap_or("sedan");
    let res = Resolution::parse(a.get(2).map(String::as_str).unwrap_or("medium")).unwrap();
    let scene = a.get(3).map(String::as_str).unwrap_or("info");
    let speed: f32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0) / 3.6;
    let offset: f32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let def = style(st).unwrap();
    let strength: f32 = std::env::var("STRENGTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let b = build(
        &def,
        &BuildParams {
            resolution: res,
            strength,
            ..Default::default()
        },
    )
    .unwrap();
    let sk = &b.skeleton;
    let (wk, wc) = sk.worst_rates();
    let wh = sk.worst_omega_h();
    eprintln!(
        "{st} {res:?}: nodes {} beams {} shocks {} skin tris {} collision tris {} mass {:.1} kg; worst k/m {:.2e} c/m {:.0} ωh {:.2} stiffness x{:.2}",
        sk.nodes.len(), sk.beams.len(), sk.shocks.len(), sk.skin.len(), sk.collision.len(), sk.total_mass(), wk, wc, wh, b.stiffness_scale
    );
    if let Ok(n) = std::env::var("NODE") {
        let i: u32 = n.parse().unwrap();
        let fl = sk.floors(1.0);
        eprintln!(
            "node {i} mass {:.3} fixed {:.3} floor {:.3} at {:?}",
            sk.nodes[i as usize].mass,
            sk.nodes[i as usize].fixed,
            fl[i as usize],
            sk.nodes[i as usize].p
        );
        for bm in sk.beams.iter().filter(|b| b.a == i || b.b == i) {
            let o = if bm.a == i { bm.b } else { bm.a };
            let l = sk.pos(i).distance(sk.pos(o));
            eprintln!(
                "  to {o} mass {:.3} L {:.3} k_at {:.0} c {:.1}",
                sk.nodes[o as usize].mass,
                l,
                bm.mat.k_at(l),
                bm.mat.damp
            );
        }
    }
    if let Ok(p) = std::env::var("TRUCK_OUT") {
        std::fs::write(&p, &b.truck).unwrap();
    }
    if scene == "info" {
        return;
    }
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        drag: std::env::var("DRAG").is_ok(),
        ..Default::default()
    };
    let probe = ExplicitBeams::from_truck(&b.truck, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let vel = match scene {
        "side" => Vec3::new(speed, 0.0, 0.0),
        _ => Vec3::new(0.0, 0.0, speed),
    };
    let lift = if scene == "drop" { 15.0 } else { 0.0 };
    let mut s = ExplicitBeams::from_truck(
        &b.truck,
        Vec3::new(0.0, 0.02 - low + lift, 0.0),
        vel,
        settings,
    )
    .unwrap();
    s.roll_tyres(vel);
    let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
    let side = s.positions().iter().map(|p| p.x).fold(f32::MIN, f32::max);
    match scene {
        "wall" => s.boxes.push(StaticBox::centered(
            Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
            Vec3::new(3.0, 2.0, 1.0),
        )),
        "pole" => s.shapes.push(StaticShape::cylinder(
            Vec3::new(offset, 2.0, front + 0.05 + 0.125),
            0.125,
            2.0,
        )),
        "side" => s.shapes.push(StaticShape::cylinder(
            Vec3::new(side + 0.05 + 0.125, 2.0, offset),
            0.125,
            2.0,
        )),
        _ => {}
    }
    let secs: f32 = std::env::var("SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if scene == "rest" { 10.0 } else { 1.5 });
    let len0 = front - s.positions().iter().map(|p| p.z).fold(f32::MAX, f32::min);
    if let Ok(n) = std::env::var("STEPS") {
        for st in 0..n.parse::<usize>().unwrap() {
            s.step();
            let vi = (0..s.nodes.len())
                .max_by(|&a, &b| s.nodes[a].vel.length().total_cmp(&s.nodes[b].vel.length()))
                .unwrap();
            let part = if vi < sk.nodes.len() {
                sk.parts[sk.nodes[vi].part as usize].clone()
            } else {
                "tyre".into()
            };
            eprintln!(
                "step {st}: fastest {vi} ({part}) v {:.2} at {:?} mass {:.3} flags {}",
                s.nodes[vi].vel.length(),
                s.nodes[vi].pos,
                s.nodes[vi].mass,
                s.nodes[vi].flags
            );
        }
        let vi = (0..s.nodes.len())
            .max_by(|&a, &b| s.nodes[a].vel.length().total_cmp(&s.nodes[b].vel.length()))
            .unwrap();
        for &bi in &s.incidence[vi] {
            let bm = &s.beams[bi as usize];
            let o = if bm.a as usize == vi { bm.b } else { bm.a };
            eprintln!("  beam {bi} to {o} at {:?} (spawn {:?}) rest {:.3} k {:.0} c {:.1} force {:.1} broken {} kind {:?}", s.nodes[o as usize].pos, sk.nodes.get(o as usize).map(|n| n.p), bm.rest, bm.k, bm.c, bm.force, bm.broken, bm.kind);
        }
        return;
    }
    let ride = |s: &ExplicitBeams| -> (f32, f32, f32) {
        let axle = sk
            .wheels
            .iter()
            .map(|w| s.nodes[w.a1 as usize].pos.y)
            .sum::<f32>()
            / sk.wheels.len() as f32;
        let g = &b.groups["sill"];
        let sill = g.iter().map(|&i| s.nodes[i as usize].pos.y).sum::<f32>() / g.len() as f32;
        let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
        (axle, sill, low)
    };
    eprintln!("spawn ride: axle {:?}", ride(&s));
    let spawn_pos = s.positions();
    let t0 = std::time::Instant::now();
    let frames = (secs * 60.0) as usize;
    let cabin: Vec<usize> = (0..sk.nodes.len())
        .filter(|&i| {
            b.groups
                .get("pillar_b")
                .is_some_and(|g| g.contains(&(i as u32)))
        })
        .collect();
    let _ = cabin;
    let mean_of = |s: &ExplicitBeams, names: &[&str]| -> Vec3 {
        let mut v = Vec3::ZERO;
        let mut n = 0.0f32;
        for nm in names {
            if let Some(g) = b.groups.get(*nm) {
                for &i in g {
                    v += s.nodes[i as usize].pos;
                    n += 1.0;
                }
            }
        }
        v / n.max(1.0)
    };
    let b0 = mean_of(&s, &["pillar_b", "pillar_b_r"]);
    let a0 = mean_of(&s, &["pillar_a", "pillar_a_r"]);
    let c0 = mean_of(&s, &["pillar_c", "pillar_c_r"]);
    let mut b_max = 0.0f32;
    let wall_z = front + 0.05;
    let fl0 = front - a0.z;
    let mut crush_max = 0.0f32;
    let sill0 = {
        let g = &b.groups["sill"];
        (s.nodes[g[0] as usize].pos.z - s.nodes[*g.last().unwrap() as usize].pos.z).abs()
    };
    let mut reported = false;
    for f in 0..frames {
        s.frame();
        if !reported && std::env::var("BROKEN").is_ok() && s.damage().0 > 0 {
            reported = true;
            eprintln!("first breaks at frame {f}:");
            for (bi, bm) in s
                .beams
                .iter()
                .enumerate()
                .filter(|(_, b)| b.broken)
                .take(12)
            {
                let (pa, pb) = (sk.nodes[bm.a as usize].p, sk.nodes[bm.b as usize].p);
                let pa_part = &sk.parts[sk.nodes[bm.a as usize].part as usize];
                let pb_part = &sk.parts[sk.nodes[bm.b as usize].part as usize];
                eprintln!("  beam {bi} {}-{} {pa_part}/{pb_part} {:?} {:?} rest0 {:.3} k {:.0} str {:.0} kind {:?}", bm.a, bm.b, pa, pb, bm.rest0, bm.k, bm.strength, bm.kind);
            }
        }
        if f % std::env::var("EVERY")
            .ok()
            .and_then(|x| x.parse().ok())
            .unwrap_or(60usize)
            == 0
            || f + 1 == frames
        {
            let p = s.positions();
            let n = p.len() as f32;
            let com = p.iter().copied().sum::<Vec3>() / n;
            let ke: f32 = s
                .nodes
                .iter()
                .map(|n| 0.5 * n.mass * n.vel.length_squared())
                .sum();
            let vmax = s.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max);
            let vi = (0..s.nodes.len())
                .max_by(|&a, &b| s.nodes[a].vel.length().total_cmp(&s.nodes[b].vel.length()))
                .unwrap();
            let vpart = if vi < sk.nodes.len() {
                sk.parts[sk.nodes[vi].part as usize].clone()
            } else {
                "tyre".into()
            };
            eprintln!("   fastest node {vi} ({vpart}) at {:?}", s.nodes[vi].pos);
            let nan = p.iter().any(|q| !q.is_finite());
            let (broken, bent) = s.damage();
            let fz = p.iter().map(|q| q.z).fold(f32::MIN, f32::max);
            let bz = mean_of(&s, &["pillar_b", "pillar_b_r"]);
            let az = mean_of(&s, &["pillar_a", "pillar_a_r"]);
            let cz = mean_of(&s, &["pillar_c", "pillar_c_r"]);
            b_max = b_max.max(bz.z - b0.z);
            let front_now = if scene == "wall" { wall_z.min(fz) } else { fz };
            let crush = fl0 - (front_now - az.z);
            crush_max = crush_max.max(crush);
            let vcom = s.nodes.iter().map(|n| n.mass * n.vel).sum::<Vec3>()
                / s.nodes.iter().map(|n| n.mass).sum::<f32>();
            let sill = |s: &ExplicitBeams| -> f32 {
                let g = &b.groups["sill"];
                let first = s.nodes[g[0] as usize].pos.z;
                let last = s.nodes[*g.last().unwrap() as usize].pos.z;
                (first - last).abs()
            };
            let _ = sill;
            eprintln!("   sill length change {:.3}", sill(&s) - sill0);
            eprintln!(
                "   front crush {crush:.3} (max {crush_max:.3}) com velocity {:.2} m/s",
                vcom.z
            );
            {
                let shell: std::collections::HashSet<u32> = b.shell.nodes.iter().copied().collect();
                let mut e = [0.0f64; 4];
                for bm in s.beams.iter().filter(|x| !x.broken) {
                    let l = s.nodes[bm.a as usize]
                        .pos
                        .distance(s.nodes[bm.b as usize].pos);
                    let d = l - bm.rest;
                    let en = 0.5 * bm.k as f64 * (d as f64).powi(2);
                    let cat = if bm.a as usize >= sk.nodes.len() || bm.b as usize >= sk.nodes.len()
                    {
                        3
                    } else if shell.contains(&bm.a) && shell.contains(&bm.b) {
                        0
                    } else if shell.contains(&bm.a) || shell.contains(&bm.b) {
                        1
                    } else {
                        2
                    };
                    e[cat] += en;
                }
                eprintln!(
                    "   elastic J: skin {:.0} skin-structure {:.0} structure {:.0} tyres {:.0}",
                    e[0], e[1], e[2], e[3]
                );
            }
            eprintln!(
                "   B-pillar travel {:.3} (max {:.3}) A–C cabin length change {:.3} A–B {:.3}",
                bz.z - b0.z,
                b_max,
                (az.z - cz.z) - (a0.z - c0.z),
                (az.z - bz.z) - (a0.z - b0.z)
            );
            let rz = p.iter().map(|q| q.z).fold(f32::MAX, f32::min);
            eprintln!("   ride (axle y, sill y, lowest) {:?}", ride(&s));
            {
                let g = &b.groups["sill"];
                let ys: Vec<String> = g
                    .iter()
                    .step_by(4)
                    .map(|&i| {
                        format!(
                            "{:.3}@{:.2}",
                            s.nodes[i as usize].pos.y, s.nodes[i as usize].pos.z
                        )
                    })
                    .collect();
                eprintln!("   sill profile {}", ys.join(" "));
                let tw: Vec<String> = sk
                    .shocks
                    .iter()
                    .map(|x| {
                        format!(
                            "top {:.3} kl {:.3} len {:.3}",
                            s.nodes[x.b as usize].pos.y,
                            s.nodes[x.a as usize].pos.y,
                            s.nodes[x.a as usize]
                                .pos
                                .distance(s.nodes[x.b as usize].pos)
                        )
                    })
                    .collect();
                eprintln!("   shocks {}", tw.join(" | "));
            }
            eprintln!(
                "t {:.2}s com ({:.3},{:.3},{:.3}) KE {:.1} J vmax {:.2} broken {broken} bent {bent} length {:.3} (crush {:.3}) nan {nan} [{:.1}s]",
                (f + 1) as f32 / 60.0, com.x, com.y, com.z, ke, vmax, fz - rz, len0 - (fz - rz), t0.elapsed().as_secs_f32()
            );
        }
    }
    if let Ok(dir) = std::env::var("DRAW") {
        let shell: std::collections::HashSet<u32> = b.shell.nodes.iter().copied().collect();
        let mut lines = Vec::new();
        for bm in s.beams.iter().filter(|x| !x.broken) {
            if bm.a as usize >= sk.nodes.len() || bm.b as usize >= sk.nodes.len() {
                lines.push((bm.a, bm.b, [40, 40, 40]));
                continue;
            }
            let sa = shell.contains(&bm.a);
            let sb = shell.contains(&bm.b);
            let c = if sa && sb {
                [200, 200, 220]
            } else if sa || sb {
                [150, 220, 150]
            } else {
                [200, 30, 30]
            };
            lines.push((bm.a, bm.b, c));
        }
        lines.sort_by_key(|l| (l.2 == [200, 30, 30]) as u8);
        draw(&format!("{dir}/end.png"), &s.positions(), &lines);
        draw(&format!("{dir}/spawn.png"), &spawn_pos, &lines);
    }
    if let Ok(dir) = std::env::var("VISUAL") {
        let vis = engine_softbody::authoring::skin::showcase_visual(&b).unwrap();
        eprintln!(
            "visual: {} vertices, {} triangles",
            vis.positions.len(),
            vis.indices.len() / 3
        );
        let nodes = s.positions();
        let spawn: Vec<Vec3> = spawn_pos.clone();
        let c = nodes.iter().copied().sum::<Vec3>() / nodes.len() as f32;
        let c0 = spawn.iter().copied().sum::<Vec3>() / spawn.len() as f32;
        for (name, np, cc) in [("rest", &spawn, c0), ("after", &nodes, c)] {
            let (p, n) = engine_softbody::authoring::skin::deform(&vis, np);
            for (vi, eye) in [
                (0, Vec3::new(3.6, 1.4, 3.4)),
                (1, Vec3::new(-3.5, 1.2, -3.6)),
                (2, Vec3::new(0.0, 1.0, 5.2)),
            ] {
                render(
                    &format!("{dir}/{scene}_{name}_{vi}.png"),
                    &p,
                    &n,
                    &vis.colors,
                    &vis.material,
                    &vis.ao,
                    &vis.indices,
                    cc + eye,
                    cc + Vec3::new(0.0, -0.05, 0.0),
                );
            }
        }
    }
    if std::env::var("CORNER").is_ok() {
        let g = &b.groups["suspension"];
        let p = s.positions();
        let names = [
            "a1", "a2", "ku", "kl", "ks", "plf", "plr", "puf", "pur", "pt", "top",
        ];
        for (k, &i) in g.iter().take(11).enumerate() {
            eprintln!(
                "  {:>4} spawn {:?} now {:?}",
                names[k], spawn_pos[i as usize], p[i as usize]
            );
            let ties: Vec<String> = s.incidence[i as usize]
                .iter()
                .map(|&bi| {
                    let bm = &s.beams[bi as usize];
                    let o = if bm.a == i { bm.b } else { bm.a };
                    format!(
                        "{}{}",
                        if g.contains(&o) { "s" } else { "B" },
                        if bm.broken { "x" } else { "" }
                    )
                })
                .collect();
            eprintln!("       beams {}", ties.join(""));
        }
    }
    if std::env::var("GROUPS").is_ok() {
        let p = s.positions();
        let mut names: Vec<&String> = b.groups.keys().collect();
        names.sort();
        for n in names {
            let g = &b.groups[n];
            let dy = g
                .iter()
                .map(|&i| p[i as usize].y - spawn_pos[i as usize].y)
                .sum::<f32>()
                / g.len() as f32;
            eprintln!("  {n:>16} dy {dy:+.3}");
        }
    }
}

/// A debug raster: views of the beams (side: z→x, y→up; front: x, y; top: z, x).
pub fn draw(path: &str, pos: &[Vec3], beams: &[(u32, u32, [u8; 3])]) {
    let (w, h) = (1500usize, 900usize);
    let mut img = vec![255u8; w * h * 3];
    let s = 140.0f32;
    let views: [(f32, f32, usize); 3] = [(80.0, 330.0, 0), (80.0, 740.0, 2), (900.0, 330.0, 1)];
    for &(ox, oy, kind) in &views {
        let proj = |p: Vec3| -> (f32, f32) {
            match kind {
                0 => (ox + (p.z + 2.6) * s, oy - p.y * s),
                1 => (ox + (p.x + 1.3) * s + 300.0, oy - p.y * s),
                _ => (ox + (p.z + 2.6) * s, oy - p.x * s),
            }
        };
        for &(a, b, c) in beams {
            let (x0, y0) = proj(pos[a as usize]);
            let (x1, y1) = proj(pos[b as usize]);
            let n = ((x1 - x0).abs().max((y1 - y0).abs()) as usize).clamp(1, 4000);
            for i in 0..=n {
                let t = i as f32 / n as f32;
                let (x, y) = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
                if x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h {
                    let k = ((y as usize) * w + x as usize) * 3;
                    img[k..k + 3].copy_from_slice(&c);
                }
            }
        }
    }
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    out.extend_from_slice(&img);
    let ppm = format!("{path}.ppm");
    std::fs::write(&ppm, out).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i", &ppm, path])
        .status();
}

/// A software render of the visual mesh (z-buffer, Lambert + a sky term).
pub fn render(
    path: &str,
    pos: &[Vec3],
    nrm: &[Vec3],
    col: &[[f32; 3]],
    mats: &[u8],
    ao: &[f32],
    idx: &[u32],
    eye: Vec3,
    target: Vec3,
) {
    let (w, h) = (1280usize, 720usize);
    let mut img = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let t = y as f32 / h as f32;
            let c = [0.75 - 0.25 * t, 0.8 - 0.25 * t, 0.85 - 0.2 * t];
            for k in 0..3 {
                img[(y * w + x) * 3 + k] = (c[k] * 255.0) as u8;
            }
        }
    }
    let mut zb = vec![f32::MAX; w * h];
    let fwd = (target - eye).normalize();
    let right = fwd.cross(Vec3::Y).normalize();
    let up = right.cross(fwd);
    let f = 1.6 * h as f32 / 2.0;
    let proj = |p: Vec3| -> (f32, f32, f32) {
        let d = p - eye;
        let z = d.dot(fwd);
        (
            w as f32 / 2.0 + d.dot(right) / z * f,
            h as f32 / 2.0 - d.dot(up) / z * f,
            z,
        )
    };
    let light = Vec3::new(0.4, 0.8, 0.35).normalize();
    for t in idx.chunks(3) {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        if !(pos[a].is_finite() && pos[b].is_finite() && pos[c].is_finite()) {
            continue;
        }
        let pa = proj(pos[a]);
        let pb = proj(pos[b]);
        let pc = proj(pos[c]);
        if pa.2 < 0.1 || pb.2 < 0.1 || pc.2 < 0.1 {
            continue;
        }
        let minx = pa.0.min(pb.0).min(pc.0).floor().max(0.0) as usize;
        let maxx = pa.0.max(pb.0).max(pc.0).ceil().min(w as f32 - 1.0) as usize;
        let miny = pa.1.min(pb.1).min(pc.1).floor().max(0.0) as usize;
        let maxy = pa.1.max(pb.1).max(pc.1).ceil().min(h as f32 - 1.0) as usize;
        if maxx < minx || maxy < miny || maxx - minx > 600 {
            continue;
        }
        let area = (pb.0 - pa.0) * (pc.1 - pa.1) - (pc.0 - pa.0) * (pb.1 - pa.1);
        if area.abs() < 1e-9 {
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
                let z = w0 * pa.2 + w1 * pb.2 + w2 * pc.2;
                let k = y * w + x;
                if z >= zb[k] {
                    continue;
                }
                zb[k] = z;
                let mut n = (nrm[a] * w0 + nrm[b] * w1 + nrm[c] * w2).normalize_or_zero();
                let p = pos[a] * w0 + pos[b] * w1 + pos[c] * w2;
                let v = (eye - p).normalize();
                if n.dot(v) < 0.0 {
                    n = -n;
                }
                let base = Vec3::from_array(col[a]);
                let m = mats[a];
                let diff = n.dot(light).max(0.0);
                let sky = 0.35 + 0.25 * n.y;
                let r = (2.0 * n.dot(v) * n - v).normalize();
                let spec_pow = match m {
                    0 => 40.0,
                    1 => 80.0,
                    2 => 30.0,
                    _ => 10.0,
                };
                let spec_k = match m {
                    0 => 0.5,
                    1 => 0.9,
                    2 => 0.9,
                    4 => 0.6,
                    _ => 0.08,
                };
                let spec = r.dot(light).max(0.0).powf(spec_pow) * spec_k;
                let env = if matches!(m, 0 | 1 | 2) {
                    0.12 * (0.5 + 0.5 * r.y)
                } else {
                    0.0
                };
                let c = base * (diff * 0.75 + sky) + Vec3::splat(spec + env);
                let occ = ao[t[0] as usize] * w0 + ao[t[1] as usize] * w1 + ao[t[2] as usize] * w2;
                let c = c * occ;
                let c = c.powf(1.0 / 2.2);
                for (i, ch) in c.to_array().iter().enumerate() {
                    img[k * 3 + i] = (ch.clamp(0.0, 1.0) * 255.0) as u8;
                }
            }
        }
    }
    // Ground line shade.
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    out.extend_from_slice(&img);
    let ppm = format!("{path}.ppm");
    std::fs::write(&ppm, out).unwrap();
    let _ = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i", &ppm, path])
        .status();
}
