//! Car to car on the CPU reference: two rigged cars of a design head on, full
//! overlap, each at `km/h` (default 50; equal masses and speeds make the plane
//! between them a rigid wall for each, so each car's crush compares with the
//! rigid wall at the same speed). Raycast wheels, vehicle contact (node
//! against triangle), surface contact on. Prints each car's dynamic and
//! residual crush (bumper beam centre to the tunnel's B-pillar station along
//! the car), its cabin's peak deceleration (10 ms mean), broken beams and
//! hardware, and the momentum and energy books.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example vehicle_c2c -- [design] [km/h] [seconds]
//! ```
use engine_softbody::authoring::rig;
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::ExplicitBeams;
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_softbody::vehicle_compare::settings;
use glam::Vec3;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let design = a.first().cloned().unwrap_or_else(|| "sedan".into());
    let kmh: f32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(50.0);
    let secs: f32 = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(1.0);
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
    // The second car turned about +Y (Rubble frame: +Z forward).
    let mut turned: SoftVehicleDesc = desc.clone();
    for n in &mut turned.nodes {
        n.position = [-n.position[0], n.position[1], -n.position[2]];
    }
    let len = {
        let z: Vec<f32> = desc.nodes.iter().map(|n| n.position[2]).collect();
        z.iter().cloned().fold(f32::MIN, f32::max) - z.iter().cloned().fold(f32::MAX, f32::min)
    };
    let v = kmh / 3.6;
    // Noses 5 cm apart: car A's nose at z = −0.025 (its front-most node at
    // the desc's max z), car B's at +0.025.
    let zmax = desc.nodes.iter().map(|n| n.position[2]).fold(f32::MIN, f32::max);
    let lift = 0.02 - desc.nodes.iter().map(|n| n.position[1]).fold(f32::MAX, f32::min);
    let pa = Vec3::new(0.0, lift, -0.025 - zmax);
    let pb = Vec3::new(0.0, lift, 0.025 + zmax);
    let (mut sa, _) = ExplicitBeams::from_desc_jbeam_raycast(&desc, pa, Vec3::Z * v, settings())?;
    let (mut sb, _) =
        ExplicitBeams::from_desc_jbeam_raycast(&turned, pb, -Vec3::Z * v, settings())?;
    sa.surface_contact = true;
    sb.surface_contact = true;
    sa.finish();
    sb.finish();
    let idx = |n: &str| desc.nodes.iter().position(|x| x.name == n).ok_or(format!("no node {n}"));
    let (front, back) = (idx("bf_b02")?, idx("tn2")?);
    let cabin: Vec<usize> = (0..5).filter_map(|k| idx(&format!("tn{k}")).ok()).collect();
    let mut w = ExplicitBeamsWorld::new(vec![sa, sb]);
    let pos = |s: &ExplicitBeams, i: usize| s.origin + s.nodes[i].pos;
    let l0 = |s: &ExplicitBeams| (pos(s, front) - pos(s, back)).length();
    let base = [l0(&w.vehicles[0]), l0(&w.vehicles[1])];
    let cab_v = |s: &ExplicitBeams, fw: f32| {
        cabin.iter().map(|&i| s.nodes[i].vel.z * fw).sum::<f32>() / cabin.len() as f32
    };
    let mom = |w: &ExplicitBeamsWorld| {
        w.vehicles
            .iter()
            .flat_map(|s| s.nodes.iter().map(|n| n.vel * n.mass))
            .sum::<Vec3>()
    };
    let ke = |w: &ExplicitBeamsWorld| {
        w.vehicles
            .iter()
            .flat_map(|s| s.nodes.iter().map(|n| 0.5 * n.mass as f64 * n.vel.length_squared() as f64))
            .sum::<f64>()
    };
    let (p0, e0) = (mom(&w), ke(&w));
    let mut crush_max = [0.0f32; 2];
    let mut hist: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    let steps = (secs / 0.0005).round() as usize;
    for k in 0..steps {
        w.step();
        if k % 10 == 0 {
            for (c, fw) in [(0usize, 1.0f32), (1, -1.0)] {
                let s = &w.vehicles[c];
                crush_max[c] = crush_max[c].max(base[c] - l0(s));
                hist[c].push(cab_v(s, fw));
            }
        }
    }
    println!(
        "{design} head on, full overlap, {kmh} km/h each ({len:.2} m cars), {secs} s:"
    );
    for c in 0..2 {
        let s = &w.vehicles[c];
        // 10 ms means of the cabin's velocity (samples every 5 ms).
        let h = &hist[c];
        let peak = (2..h.len())
            .map(|k| (h[k - 2] - h[k]) / 0.010 / 9.81)
            .fold(0.0f32, f32::max);
        let broken = s.beams.iter().filter(|b| b.broken).count();
        let groups: std::collections::BTreeSet<i32> =
            s.beams.iter().filter(|b| b.broken && b.group > 0).map(|b| b.group).collect();
        println!(
            "  car {}: dynamic crush {:.0} mm, residual {:.0} mm, cabin peak {:.1} g, final cabin speed {:+.2} m/s, broken beams {broken}, break groups set off {}",
            ["A", "B"][c],
            crush_max[c] * 1e3,
            (base[c] - l0(s)) * 1e3,
            peak,
            h.last().copied().unwrap_or(0.0),
            groups.len()
        );
    }
    let p1 = mom(&w);
    println!(
        "  momentum {:.1} → {:.1} N·s (|Δ| {:.2}); kinetic energy {:.1} → {:.1} kJ",
        p0.length(),
        p1.length(),
        (p1 - p0).length(),
        e0 / 1e3,
        ke(&w) / 1e3
    );
    Ok(())
}
