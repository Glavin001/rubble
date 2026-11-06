//! Diagnostic: a car of the comparison manifest into the wall (surface contact
//! as `SURFACE_CONTACT`), and every 10 ms the greenhouse's racking (the mean
//! move of the cabin's top nodes against its floor nodes, over the middle of
//! the wheelbase), named nodes' positions and, with `PROBE_MESH=1`, the skinned
//! mesh pieces' top heights over the middle of the car.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example gen3_probe -- <car> [kmh] [node ...]
//! ```
use engine_softbody::explicit_beams::{ExplicitBeams, StaticBox};
use engine_softbody::vehicle_compare::{default_manifest, Car, Manifest};
use glam::Vec3;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let name = a.first().cloned().unwrap_or_else(|| "gen2".into());
    let kmh: f32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(56.0);
    let names: Vec<&String> = a.iter().skip(2).collect();
    let m: Manifest = serde_json::from_str(
        &std::fs::read_to_string(default_manifest()).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let spec = m
        .cars
        .iter()
        .find(|c| c.name == name)
        .ok_or("no such car")?;
    let car = Car::load(spec)?;
    let mut s = car.template.clone();
    let rest = pos(&s);
    let front = car
        .skin
        .iter()
        .map(|&i| s.nodes[i as usize].pos.z)
        .fold(f32::MIN, f32::max);
    let wall_z = front + 0.05;
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 2.0, wall_z + 0.5),
        Vec3::new(car.frame.width() * 0.5 + 1.0, 2.0, 0.5),
    ));
    for n in &mut s.nodes {
        n.vel = Vec3::Z * kmh / 3.6;
    }
    s.roll_tyres(Vec3::Z * kmh / 3.6);
    let ids: Vec<usize> = names
        .iter()
        .filter_map(|nm| (0..s.nodes.len()).find(|&i| car.node_name(i) == **nm))
        .collect();
    // The greenhouse: skin nodes over the middle of the wheelbase, the top
    // 15 cm against the floor's lowest 25 cm.
    let f = car.frame;
    let mid = 0.5 * (f.front_axle + f.rear_axle);
    let span: Vec<u32> = car
        .skin
        .iter()
        .copied()
        .filter(|&i| {
            (rest[i as usize].z - mid).abs()
                < std::env::var("PROBE_SPAN")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.45)
                    * f.wheelbase()
        })
        .collect();
    let ytop = span
        .iter()
        .map(|&i| rest[i as usize].y)
        .fold(f32::MIN, f32::max);
    let top: Vec<u32> = span
        .iter()
        .copied()
        .filter(|&i| rest[i as usize].y > ytop - 0.15)
        .collect();
    let floor: Vec<u32> = span
        .iter()
        .copied()
        .filter(|&i| rest[i as usize].y < f.floor + 0.25)
        .collect();
    let mean = |s: &ExplicitBeams, set: &[u32]| {
        set.iter()
            .map(|&i| s.origin + s.nodes[i as usize].pos - rest[i as usize])
            .sum::<Vec3>()
            / set.len().max(1) as f32
    };
    let mesh = std::env::var("PROBE_MESH").is_ok();
    let v = car.visual.as_ref();
    let piece: Vec<String> = v.map_or(Vec::new(), |v| {
        let mut p = vec![String::new(); v.positions.len()];
        for (nm, first, count) in &v.pieces {
            for k in *first..first + count {
                p[k as usize] = nm.rsplit(':').next().unwrap_or(nm).to_string();
            }
        }
        p
    });
    // `PROBE_REAR=1`: the band over the trunk (its rear 0.4 m, above 0.75 m)
    // instead of the middle of the car (above 1 m).
    let (band_z, band_y) = if std::env::var("PROBE_REAR").is_ok() {
        (f.lo.z + 0.25, 0.75)
    } else {
        (mid, 1.0)
    };
    let report = |t: f32, s: &ExplicitBeams| {
        // In the floor's own frame (its rigid fit): the greenhouse's move
        // against the floor without the car's pitch.
        let now = pos(s);
        let fit = engine_softbody::vehicle_compare::rigid_fit(&rest, &now, &floor);
        let d = top
            .iter()
            .map(|&i| fit.to_rest(now[i as usize]) - rest[i as usize])
            .sum::<Vec3>()
            / top.len().max(1) as f32;
        let _ = mean;
        let mut txt = format!(
            "t {:4.0} ms: racking top − floor dz {:+.3} dy {:+.3} m (top {} nodes, floor {})",
            t * 1e3,
            d.z,
            d.y,
            top.len(),
            floor.len()
        );
        if let (true, Some(v)) = (mesh, v) {
            let (p, _) = engine_softbody::authoring::skin::deform(v, &pos(s));
            let (p0, _) = engine_softbody::authoring::skin::deform(v, &rest);
            let mut hi: std::collections::BTreeMap<&str, f32> = Default::default();
            for k in 0..p.len() {
                if (p0[k].z - band_z).abs() < 0.2
                    && p0[k].y > rest_floor(&f) + band_y
                    && p0[k].x.abs() < 0.6
                {
                    let e = hi.entry(piece[k].as_str()).or_insert(f32::MIN);
                    *e = e.max(p[k].y);
                }
            }
            for (k, y) in hi {
                txt += &format!(" {k}={y:.3}");
            }
        }
        if std::env::var("PROBE_FAST").is_ok() {
            let mut v: Vec<(f32, usize)> = s
                .nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (n.vel.length(), i))
                .collect();
            v.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(sp, i) in v.iter().take(4) {
                txt += &format!(
                    " | {} {:.2} m/s ({:.3} kg)",
                    car.node_name(i),
                    sp,
                    s.nodes[i].mass
                );
            }
        }
        for &i in &ids {
            let q = s.origin + s.nodes[i].pos;
            txt += &format!(" | {} ({:.3},{:.3},{:.3})", car.node_name(i), q.x, q.y, q.z);
        }
        println!("{txt}");
    };
    report(0.0, &s);
    let (mut t, mut next) = (0.0f32, 0.01f32);
    let end: f32 = std::env::var("PROBE_END")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.2);
    while t < end {
        s.step();
        t += 0.0005;
        if t >= next - 1e-6 {
            report(t, &s);
            next += 0.01;
        }
    }
    if std::env::var("PROBE_ENERGY").is_ok() {
        // Plastic work by zone (yield force now × the rest length's change:
        // an upper estimate under hardening): ahead of the firewall (the
        // front axle less a fifth of the wheelbase), the cabin, behind the
        // rear axle; and the cabin's by the nodes' height.
        let fw = f.front_axle - 0.2 * f.wheelbase();
        let mut zone = [0.0f64; 3];
        let mut cab_hi = 0.0f64;
        for b in &s.beams {
            if b.broken || b.a == b.b || b.rest0 <= 0.0 {
                continue;
            }
            let dr = (b.rest - b.rest0) as f64;
            let fy = if dr < 0.0 { b.tc } else { b.tt } as f64;
            if !fy.is_finite() {
                continue;
            }
            let w = fy * dr.abs();
            let m = 0.5 * (rest[b.a as usize] + rest[b.b as usize]);
            let k = if m.z > fw {
                0
            } else if m.z > f.rear_axle {
                1
            } else {
                2
            };
            zone[k] += w;
            if k == 1 && m.y > f.floor + 0.55 {
                cab_hi += w;
            }
        }
        println!(
            "plastic work (kJ): front {:.1}, cabin {:.1} (above the belt {:.1}), rear {:.1}",
            zone[0] / 1e3,
            zone[1] / 1e3,
            cab_hi / 1e3,
            zone[2] / 1e3
        );
    }
    if std::env::var("PROBE_BENT").is_ok() {
        // The beams that yielded most (plastic strain), and the most strained
        // beams now (elastic and plastic).
        let mut v: Vec<(f32, f32, usize)> = s
            .beams
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.broken && b.rest0 > 0.02 && b.a != b.b)
            .map(|(j, b)| {
                let l = (s.nodes[b.a as usize].pos - s.nodes[b.b as usize].pos).length();
                ((b.rest - b.rest0) / b.rest0, (l - b.rest0) / b.rest0, j)
            })
            .collect();
        v.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
        for &(p, e, j) in v
            .iter()
            .filter(|q| {
                std::env::var("PROBE_MATCH").map_or(true, |m| {
                    let b = &s.beams[q.2];
                    m.split(",").any(|t| {
                        car.node_name(b.a as usize).starts_with(t)
                            || car.node_name(b.b as usize).starts_with(t)
                    })
                })
            })
            .take(40)
        {
            let b = &s.beams[j];
            println!(
                "plastic {:+.3} strain {:+.3}: {} - {} ({:.2} m)",
                p,
                e,
                car.node_name(b.a as usize),
                car.node_name(b.b as usize),
                b.rest0
            );
        }
    }
    Ok(())
}

fn rest_floor(f: &engine_softbody::vehicle_compare::Frame) -> f32 {
    f.floor
}

fn pos(s: &ExplicitBeams) -> Vec<Vec3> {
    s.nodes.iter().map(|n| s.origin + n.pos).collect()
}
