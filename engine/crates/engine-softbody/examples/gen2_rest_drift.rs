//! Rest drift: how far each part's nodes sit from their designed (JBeam)
//! positions after spawn and after parking, about the body's rigid fit (the
//! hubs, rim proxies and tyres excluded from the fit), on pressure and on
//! raycast wheels. A part that sags, springs or warps at rest shows here.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example gen2_rest_drift -- \
//!     [mod-dir] [vehicle] [pc] [seconds...]
//! ```
//! Defaults: `engine/crates/engine-webgpu/__mods/gen2 gen2 gen2.pc 1 10`.
use engine_softbody::explicit_beams::{ExplicitBeams, JbeamWheels, NODE_TYRE};
use engine_softbody::vehicle_compare::{rigid_fit, settings};
use glam::Vec3;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dir = a.first().cloned().unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../engine-webgpu/__mods/gen2").into()
    });
    let vehicle = a.get(1).cloned().unwrap_or_else(|| "gen2".into());
    let pc = a.get(2).cloned().unwrap_or_else(|| "gen2.pc".into());
    let mut times: Vec<f32> = a.iter().skip(3).filter_map(|s| s.parse().ok()).collect();
    if times.is_empty() {
        times = vec![1.0, 10.0];
    }
    let dir = std::path::Path::new(&dir);
    let files = engine_softbody::jbeam::read_vehicle_dir(dir, &vehicle)?;
    let pct = std::fs::read_to_string(dir.join(&vehicle).join(&pc)).ok();
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let desc = engine_softbody::jbeam::load_jbeam(&refs, pct.as_deref())?;
    let design: Vec<Vec3> = desc.nodes.iter().map(|n| Vec3::from(n.position)).collect();
    let nd = design.len();
    let suspension = |name: &str| name.starts_with("hub") || name.starts_with("rim");
    let fit_ids: Vec<u32> = (0..nd as u32)
        .filter(|&i| !suspension(&desc.nodes[i as usize].name))
        .collect();
    for (label, wheels) in [
        ("pressure", JbeamWheels::Pressure),
        ("raycast", JbeamWheels::Raycast),
    ] {
        let (mut s, _) =
            ExplicitBeams::from_desc_jbeam_with(&desc, wheels, Vec3::ZERO, Vec3::ZERO, settings())?;
        // As the harness places a car: its lowest node 2 cm above the ground.
        let low = s
            .nodes
            .iter()
            .map(|n| s.origin.y + n.pos.y)
            .fold(f32::MAX, f32::min);
        for n in &mut s.nodes {
            n.pos.y += 0.02 - low;
        }
        let report = |s: &ExplicitBeams, when: &str| {
            let now: Vec<Vec3> = (0..nd).map(|i| s.origin + s.nodes[i].pos).collect();
            assert!(s.nodes[..nd].iter().all(|n| n.flags & NODE_TYRE == 0));
            let fit = rigid_fit(&design, &now, &fit_ids);
            let d: Vec<Vec3> = (0..nd).map(|i| fit.to_rest(now[i]) - design[i]).collect();
            println!("[{label}] {when}:");
            let mut rows: Vec<(f32, String)> = desc
                .parts
                .iter()
                .filter(|p| !p.nodes.is_empty())
                .map(|p| {
                    let mut v: Vec<(f32, usize)> = p
                        .nodes
                        .iter()
                        .filter(|&&i| {
                            (i as usize) < nd && !suspension(&desc.nodes[i as usize].name)
                        })
                        .map(|&i| (d[i as usize].length(), i as usize))
                        .collect();
                    v.sort_by(|x, y| y.0.total_cmp(&x.0));
                    let worst: Vec<String> = v
                        .iter()
                        .take(3)
                        .map(|&(m, i)| {
                            format!(
                                "{} {:.1} (dz {:+.1})",
                                desc.nodes[i].name,
                                m * 1e3,
                                d[i].y * 1e3
                            )
                        })
                        .collect();
                    let mx = v.first().map_or(0.0, |x| x.0);
                    let p50 = v.get(v.len() / 2).map_or(0.0, |x| x.0);
                    (
                        mx,
                        format!(
                            "  {:<16} max {:6.1} mm  p50 {:5.1} mm  worst: {}",
                            p.name,
                            mx * 1e3,
                            p50 * 1e3,
                            worst.join(", ")
                        ),
                    )
                })
                .collect();
            rows.sort_by(|x, y| y.0.total_cmp(&x.0));
            for (_, r) in rows {
                println!("{r}");
            }
        };
        report(&s, "spawn");
        let mut t = 0.0f32;
        for &tt in &times {
            while t < tt - 1e-6 {
                s.step();
                t += 0.0005;
            }
            report(&s, &format!("parked {tt} s"));
        }
    }
    Ok(())
}
