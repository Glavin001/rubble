//! Measurement (time only): one 34-step frame of N repo sedans apart, the in-workgroup
//! frame against the wide passes (one thread per element across every vehicle, a
//! dispatch per pass per substep), wall time a frame (median of 20 after 10 warm), and
//! the wide frame's GPU ms per pass kind (each dispatch in its own timed pass: the
//! passes add their own cost). For wide-frame-design.md.
//!
//!   cargo run --release -p engine-webgpu-lab --example wide_vs_workgroup -- 1 32 128
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

fn main() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    let counts: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let (_, device, queue) = default_device().unwrap();
    for n in if counts.is_empty() { vec![1, 32] } else { counts } {
        let fleet: Vec<ExplicitBeams> = (0..n)
            .map(|i| {
                let at = Vec3::new((i % 16) as f32 * 6.0, 0.0, (i / 16) as f32 * 12.0);
                // WIDE_CAR=<jbeam dir with files.json>: that car (JBeam mode, pressure wheels)
                // in place of the repo sedan.
                let mut s = match std::env::var("WIDE_CAR") {
                    Ok(dir) => {
                        let dir = std::path::Path::new(&dir);
                        let list = std::fs::read_to_string(dir.join("files.json")).unwrap();
                        let names: Vec<&str> = list.split('"').filter(|s| s.ends_with(".jbeam")).collect();
                        let files: Vec<(String, String)> = names.iter().map(|f| (f.to_string(), std::fs::read_to_string(dir.join(f)).unwrap())).collect();
                        let refs: Vec<(&str, &str)> = files.iter().map(|(n, t)| (n.as_str(), t.as_str())).collect();
                        let pc = std::fs::read_to_string(dir.join("sedan/sedan.pc")).ok();
                        let desc = engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref()).unwrap();
                        let settings = ExplicitBeamsSettings { fast_inv_sqrt: true, ..Default::default() };
                        let wheels = if std::env::var("WIDE_RAYCAST").is_ok() { engine_softbody::explicit_beams::JbeamWheels::Raycast } else { engine_softbody::explicit_beams::JbeamWheels::Pressure };
                        ExplicitBeams::from_desc_jbeam_with(&desc, wheels, at, Vec3::new(0.0, 0.0, 15.0), settings).unwrap().0
                    }
                    Err(_) => ExplicitBeams::from_truck(&text, at, Vec3::new(0.0, 0.0, 15.0), ExplicitBeamsSettings::default()).unwrap(),
                };
                let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
                for p in &mut s.nodes {
                    p.pos.y += 0.02 - low;
                }
                s.finish();
                s
            })
            .collect();
        let (nodes, beams) = (fleet[0].nodes.len(), fleet[0].beams.len());
        let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &fleet);
        gpu.set_vehicle_contact(false);
        let snap = gpu.snapshot();
        let mut line = format!("{n} sedans ({nodes} nodes, {beams} beams each):");
        for wide in [false, true] {
            gpu.restore(&snap);
            let on = gpu.set_wide(wide);
            for _ in 0..10 {
                gpu.frame(34);
            }
            gpu.wait();
            let mut ms = Vec::new();
            for _ in 0..20 {
                let t = std::time::Instant::now();
                gpu.frame(34);
                gpu.wait();
                ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            ms.sort_by(f64::total_cmp);
            line += &format!(" {} {:.2} ms/frame (wide {on});", if wide { "wide" } else { "workgroup" }, ms[10]);
            if on {
                let p = gpu.profile_wide_frame(34);
                line += &format!(
                    " wide passes (GPU ms a frame, timed apart): {}",
                    p.iter().filter(|(_, ms)| *ms > 0.0).map(|(k, ms)| format!("{} {ms:.2}", k.trim_start_matches("explicit_beams_wide_"))).collect::<Vec<_>>().join(", ")
                );
            }
        }
        println!("{line}");
    }
}
