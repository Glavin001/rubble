//! The lab's sedan, loaded as `vehicle_lab`'s `car()` and placed as `lab_car` places it
//! on the flat arena (ground y = 0), with no World: the reference's own ground plane and
//! static shapes stand in for the World's coupling.
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
use glam::{Quat, Vec3};

pub fn default_car_dir() -> String {
    std::env::var("CAR").unwrap_or_else(|_| {
        format!("{}/Development/rubble-artifacts/sedan-live/mods/sedan_new", std::env::var("HOME").unwrap())
    })
}

/// The template (vehicle_lab `car()` for a JBeam export directory, drag on).
pub fn load_template(dir: &str, wheels: JbeamWheels) -> anyhow::Result<ExplicitBeams> {
    let dir = std::path::Path::new(dir);
    let list: Vec<String> = serde_json::from_str(&std::fs::read_to_string(dir.join("files.json"))?)?;
    let mut texts = Vec::new();
    let mut pc = None;
    for f in &list {
        if f.ends_with(".jbeam") {
            texts.push((f.clone(), std::fs::read_to_string(dir.join(f))?));
        } else if f.ends_with(".pc") {
            pc = Some(std::fs::read_to_string(dir.join(f))?);
        }
    }
    let files: Vec<(&str, &str)> = texts.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let settings = ExplicitBeamsSettings { fast_inv_sqrt: true, drag: true, turbulence: true, ..Default::default() };
    let desc = engine_softbody::jbeam::load_jbeam(&files, pc.as_deref()).map_err(anyhow::Error::msg)?;
    let (mut sim, _w) = match wheels {
        JbeamWheels::Raycast => ExplicitBeams::from_desc_jbeam_raycast(&desc, Vec3::ZERO, Vec3::ZERO, settings),
        w => ExplicitBeams::from_desc_jbeam_with(&desc, w, Vec3::ZERO, Vec3::ZERO, settings),
    }
    .map_err(anyhow::Error::msg)?;
    // showcase_scene::jbeam_vehicle_on: forward axis to +Z.
    let f = engine_softbody::pack::forward_axis(&desc);
    let angle = -f.x.atan2(f.z);
    let quarter = angle / std::f32::consts::FRAC_PI_2;
    let k = quarter.round();
    let rot = |p: Vec3| -> Vec3 {
        if (quarter - k).abs() > 1e-6 {
            return Quat::from_rotation_y(angle) * p;
        }
        match (k as i32).rem_euclid(4) {
            0 => p,
            1 => Vec3::new(p.z, p.y, -p.x),
            2 => Vec3::new(-p.x, p.y, -p.z),
            _ => Vec3::new(-p.z, p.y, p.x),
        }
    };
    for n in &mut sim.nodes {
        n.pos = rot(n.pos);
        n.vel = rot(n.vel);
    }
    Ok(sim)
}

/// Placed on the flat ground (y = 0): centred in x, z, its lowest node `lift` over the
/// ground, moving at `speed` along +z.
pub fn place(template: &ExplicitBeams, lift: f32, speed: f32) -> ExplicitBeams {
    let mut car = template.clone();
    let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
    for n in &car.nodes {
        lo = lo.min(n.pos + car.origin);
        hi = hi.max(n.pos + car.origin);
    }
    let mid = 0.5 * (lo + hi);
    for n in &mut car.nodes {
        n.pos = n.pos + car.origin - Vec3::new(mid.x, lo.y, mid.z) + Vec3::new(0.0, lift, 0.0);
        n.vel = Vec3::new(0.0, 0.0, speed);
    }
    car.origin = Vec3::ZERO;
    car.ground = 0.0;
    car
}

/// As [`place`], turned by `quarter` quarter turns about +y (exact), its node box centred
/// at (`x`, `z`), moving at `speed` along its own forward axis.
pub fn place_at(template: &ExplicitBeams, lift: f32, speed: f32, quarter: i32, x: f32, z: f32) -> ExplicitBeams {
    let mut car = place(template, lift, 0.0);
    let rot = |p: Vec3| match quarter.rem_euclid(4) {
        0 => p,
        1 => Vec3::new(p.z, p.y, -p.x),
        2 => Vec3::new(-p.x, p.y, -p.z),
        _ => Vec3::new(-p.z, p.y, p.x),
    };
    let fwd = rot(Vec3::Z);
    for n in &mut car.nodes {
        n.pos = rot(n.pos) + Vec3::new(x, 0.0, z);
        n.vel = fwd * speed;
    }
    car
}

/// The two-body scenes: (name, the bodies). Head-on: two cars nose to nose, each at
/// `speed`; side: one car at `speed` into the side of a parked one; rear: into the back of
/// a parked one.
pub fn pair_scene(template: &ExplicitBeams, name: &str, speed: f32, gap: f32) -> anyhow::Result<Vec<ExplicitBeams>> {
    let base = place(template, 0.02, 0.0);
    let (mut lo, mut hi) = (Vec3::MAX, Vec3::MIN);
    for n in &base.nodes {
        lo = lo.min(n.pos);
        hi = hi.max(n.pos);
    }
    let (len, wid) = (hi.z - lo.z, hi.x - lo.x);
    let mut cars = match name {
        "headon" => vec![
            place_at(template, 0.02, speed, 0, 0.0, -0.5 * (len + gap)),
            place_at(template, 0.02, speed, 2, 0.0, 0.5 * (len + gap)),
        ],
        "side" => vec![
            place_at(template, 0.02, speed, 0, 0.0, -0.5 * (len + gap) - 0.5 * wid),
            place_at(template, 0.02, 0.0, 1, 0.0, 0.0),
        ],
        "rear" => vec![
            place_at(template, 0.02, speed, 0, 0.0, -0.5 * (len + gap)),
            place_at(template, 0.02, 0.0, 0, 0.0, 0.5 * (len + gap)),
        ],
        "apart" => vec![
            place_at(template, 0.02, 0.0, 0, -5.0, 0.0),
            place_at(template, 0.02, 0.0, 0, 5.0, 0.0),
        ],
        other => anyhow::bail!("unknown pair scene {other} (headon, side, rear, apart)"),
    };
    for c in &mut cars {
        c.drive = None;
    }
    Ok(cars)
}
