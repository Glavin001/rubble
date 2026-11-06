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
        .filter(|&i| (rest[i as usize].z - mid).abs() < 0.3 * f.wheelbase())
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
    let report = |t: f32, s: &ExplicitBeams| {
        let d = mean(s, &top) - mean(s, &floor);
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
                if (p0[k].z - mid).abs() < 0.25 && p0[k].y > rest_floor(&f) + 1.0 {
                    let e = hi.entry(piece[k].as_str()).or_insert(f32::MIN);
                    *e = e.max(p[k].y);
                }
            }
            for (k, y) in hi {
                txt += &format!(" {k}={y:.3}");
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
    Ok(())
}

fn rest_floor(f: &engine_softbody::vehicle_compare::Frame) -> f32 {
    f.floor
}

fn pos(s: &ExplicitBeams) -> Vec<Vec3> {
    s.nodes.iter().map(|n| s.origin + n.pos).collect()
}
