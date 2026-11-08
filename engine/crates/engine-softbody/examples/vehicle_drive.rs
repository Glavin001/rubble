//! A rigged design driven on the CPU reference: raycast wheels (direct drive)
//! on flat ground, full throttle until the speed stops rising, then a
//! coast-down. Prints speed against time, the top speed, the drive's own
//! limit, and the coast-down's drag against ½ρ·Cd·A·v² (the triangle
//! aerodynamics, no per-node drag).
//!
//! ```bash
//! cargo run --release -p engine-softbody --example vehicle_drive -- [design] [seconds]
//! ```
use engine_softbody::authoring::rig;
use engine_softbody::explicit_beams::ExplicitBeams;
use engine_softbody::explicit_drive::DriveInput;
use engine_softbody::vehicle_compare::settings;
use glam::Vec3;

fn com(s: &ExplicitBeams) -> (Vec3, Vec3, f32) {
    let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let x = s.nodes.iter().map(|n| (s.origin + n.pos) * n.mass).sum::<Vec3>() / m;
    let v = s.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>() / m;
    (x, v, m)
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let design = a.first().cloned().unwrap_or_else(|| "sedan".into());
    let secs: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let car = rig::rig_vehicle(&design, None, None)?;
    let files = rig::mod_files(&car);
    let text: Vec<(String, String)> = files
        .iter()
        .filter(|(p, _)| p.ends_with(".jbeam") || p.ends_with(".pc"))
        .map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned()))
        .collect();
    let refs: Vec<(&str, &str)> = text
        .iter()
        .filter(|(p, _)| p.ends_with(".jbeam"))
        .map(|(p, t)| (p.as_str(), t.as_str()))
        .collect();
    let pc = text.iter().find(|(p, _)| p.ends_with(".pc")).map(|(_, t)| t.as_str());
    let desc = engine_softbody::jbeam::load_jbeam(&refs, pc)?;
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_raycast(&desc, Vec3::ZERO, Vec3::ZERO, settings())?;
    let cfg = s.drive.as_ref().ok_or("no raycast drive")?.config.clone();
    println!(
        "{design}: {} nodes, drive: {} wheels, max torque {} N m a wheel, top speed setting {:.1} m/s ({:.0} km/h)",
        s.nodes.len(),
        cfg.wheels.len(),
        cfg.max_torque,
        cfg.top_speed,
        cfg.top_speed * 3.6
    );
    for _ in 0..120 {
        s.frame();
    }
    let fw = {
        let r = cfg.refs;
        (s.nodes[r[0] as usize].pos - s.nodes[r[1] as usize].pos).normalize()
    };
    s.set_drive_input(DriveInput {
        forward: 1.0,
        ..Default::default()
    });
    let mut v = 0.0;
    let mut top = 0.0f32;
    for t in 1..=secs {
        for _ in 0..60 {
            s.frame();
        }
        v = com(&s).1.dot(fw);
        top = top.max(v);
        if t % 5 == 0 || t <= 5 {
            println!("  t {t:>3} s  v {:6.2} m/s ({:5.1} km/h)", v, v * 3.6);
        }
    }
    let broken = s.beams.iter().filter(|b| b.broken).count();
    println!("top speed {:.1} km/h after {secs} s; broken beams {broken}", top * 3.6);
    // Coast-down from the top speed: 2 s, throttle off.
    s.set_drive_input(DriveInput::default());
    let m = com(&s).2;
    let v0 = com(&s).1.dot(fw);
    for _ in 0..120 {
        s.frame();
    }
    let v1 = com(&s).1.dot(fw);
    let decel = (v0 - v1) / 2.0;
    let vm = 0.5 * (v0 + v1);
    let cd = car.input.physics.drag_cd;
    println!(
        "coast-down at {:.1} km/h: {:.3} m/s² → {:.0} N on {m:.0} kg (rolling and tyre losses included); ½ρ·Cd·v² = {:.0} N per m² of frontal area at Cd {cd}",
        vm * 3.6,
        decel,
        decel * m,
        0.5 * 1.225 * cd * vm * vm
    );
    let _ = v;
    Ok(())
}
