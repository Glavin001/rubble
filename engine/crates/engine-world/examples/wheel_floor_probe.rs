//! The raycast wheels under a load they cannot carry (the lab's compactor, a drop):
//! per wheel, per window, the suspension length against its stop, the hub's height
//! against the wheel's radius and rim, the lowest drawn point of the tyre and of the
//! hub's own nodes, the wheel's load and the plate's force. CPU reference only.
//!
//! ```text
//! cargo run --release -p engine-world --example wheel_floor_probe -- \
//!   [--car gen|showcase|sedan] [--mod <dir with sedan/>] \
//!   [--plate 0.1] [--gap 0.05] [--secs 12] [--drop 5] [--every 30]
//! ```
//! - `--car gen` (default): the generated sedan as the rigger exports it (in memory);
//!   `--mod DIR`: an export folder (`DIR/sedan/*.jbeam`, `sedan.pc`).
//! - `--plate S`: the top plate closing at S m/s to a `--gap` stop (the compactor).
//! - `--drop H`: the car dropped from H m onto the flat ground instead.
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;
use rubble::vehicle_lab::cpu::CpuLab;
use rubble::vehicle_lab::LabOptions;
use serde_json::json;

fn arg(k: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == k)
        .and_then(|i| a.get(i + 1).cloned())
}

fn gen_desc() -> SoftVehicleDesc {
    use engine_softbody::authoring::gen2::{rig as lattice, vg_input};
    use engine_softbody::authoring::rig;
    let inp = rig::DesignInput::find("sedan", None, None).expect("the sedan's design");
    let r = lattice::rig_with(&inp);
    let files = vg_input::physics_files(&inp, &r);
    let texts: Vec<(String, String)> = files
        .iter()
        .map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned()))
        .collect();
    let refs: Vec<(&str, &str)> = texts
        .iter()
        .filter(|(p, _)| p.ends_with(".jbeam"))
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let pc = texts
        .iter()
        .find(|(p, _)| p.ends_with(".pc"))
        .map(|(_, b)| b.as_str());
    engine_softbody::jbeam::load_jbeam(&refs, pc).expect("the sedan's jbeam")
}

fn mod_desc(dir: &str) -> SoftVehicleDesc {
    let d = std::path::Path::new(dir);
    let files = engine_softbody::jbeam::read_vehicle_dir(d, "sedan").expect("mod files");
    let pc = std::fs::read_to_string(d.join("sedan/sedan.pc")).ok();
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref()).expect("mod jbeam")
}

fn main() -> anyhow::Result<()> {
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        ..Default::default()
    };
    let car = arg("--car").unwrap_or_else(|| "gen".into());
    let (template, names): (ExplicitBeams, Vec<String>) = if let Some(dir) = arg("--mod") {
        let d = mod_desc(&dir);
        let (s, _) = ExplicitBeams::from_desc_jbeam_raycast(&d, Vec3::ZERO, Vec3::ZERO, settings)
            .map_err(anyhow::Error::msg)?;
        (s, d.nodes.iter().map(|n| n.name.clone()).collect())
    } else if car == "gen" {
        let d = gen_desc();
        let (s, _) = ExplicitBeams::from_desc_jbeam_raycast(&d, Vec3::ZERO, Vec3::ZERO, settings)
            .map_err(anyhow::Error::msg)?;
        (s, d.nodes.iter().map(|n| n.name.clone()).collect())
    } else {
        let text = match car.as_str() {
            "showcase" => {
                use engine_softbody::authoring::{build, style, BuildParams, Resolution};
                build(
                    &style("sedan").unwrap(),
                    &BuildParams {
                        resolution: Resolution::Low,
                        ..BuildParams::default()
                    },
                )
                .map_err(anyhow::Error::msg)?
                .truck
            }
            _ => engine_softbody::procedural::car_truck(&engine_softbody::procedural::CarParams {
                minimass: 10.0,
                ..engine_softbody::procedural::CarParams::sedan()
            }),
        };
        let (s, _) =
            ExplicitBeams::from_truck_raycast(&text, None, Vec3::ZERO, 0.0, Vec3::ZERO, settings)
                .map_err(anyhow::Error::msg)?;
        (s, Vec::new())
    };
    let drive = template.drive.as_ref().expect("raycast wheels");
    println!(
        "car {car}: {} nodes, {:.0} kg",
        template.nodes.len(),
        template.nodes.iter().map(|n| n.mass).sum::<f32>()
    );
    for (k, w) in drive.config.wheels.iter().enumerate() {
        let m = template.nodes[w.node1 as usize].mass
            + if w.node2 != w.node1 {
                template.nodes[w.node2 as usize].mass
            } else {
                0.0
            };
        println!(
            "wheel {k}: nodes {} {} ({}), hub {:.2} kg, r {:.3} susp_min {:.3} susp_max {:.3} rest_len {:.3} k {:.0} c {:.0} bump_k {:.0} bump_c {:.0}",
            w.node1,
            w.node2,
            names.get(w.node1 as usize).cloned().unwrap_or_default(),
            m,
            w.radius,
            w.susp_min,
            w.susp_max,
            w.rest_len,
            w.k,
            w.c,
            w.bump_k,
            w.bump_c
        );
    }
    // Each wheel's own nodes (the generated car's `whe..` group: axle and rim triangle).
    let rims: Vec<Vec<usize>> = drive
        .config
        .wheels
        .iter()
        .map(|w| {
            let n1 = names.get(w.node1 as usize).cloned().unwrap_or_default();
            let pre = n1.split('_').next().unwrap_or("").to_string();
            if pre.is_empty() {
                return Vec::new();
            }
            names
                .iter()
                .enumerate()
                .filter(|(_, n)| n.starts_with(&format!("{pre}_f")))
                .map(|(i, _)| i)
                .collect()
        })
        .collect();
    let secs: f64 = arg("--secs").and_then(|x| x.parse().ok()).unwrap_or(12.0);
    let every: u64 = arg("--every").and_then(|x| x.parse().ok()).unwrap_or(30);
    let drop: f32 = arg("--drop").and_then(|x| x.parse().ok()).unwrap_or(0.0);
    let speed: f64 = arg("--plate").and_then(|x| x.parse().ok()).unwrap_or(0.1);
    let gap: f64 = arg("--gap").and_then(|x| x.parse().ok()).unwrap_or(0.05);
    let events = if drop > 0.0 {
        json!([])
    } else {
        json!([{ "t": 0.5, "plates": { "axis": "top", "speed": speed, "gap": gap } }])
    };
    let script = json!({ "mode": "compactor", "scene": "flat", "surface": arg("--surface").as_deref() == Some("1"), "events": events });
    let mut lab = CpuLab::new(&template, LabOptions::from_json(&script))?;
    lab.bench.set_script(&script)?;
    if drop > 0.0 {
        for n in &mut lab.sim.nodes {
            n.pos.y += drop;
        }
    }
    let mut worst_draw = 0.0f32;
    let mut worst_hub = f32::MAX;
    let mut worst_rimnode = f32::MAX;
    for k in 0..(secs * 60.0) as u64 {
        lab.step()?;
        let s = &lab.sim;
        let d = s.drive.as_ref().unwrap();
        let g = s.ground;
        let mut line = String::new();
        for (i, (w, st)) in d.config.wheels.iter().zip(&d.wheels).enumerate() {
            let hub_y = 0.5 * (s.nodes[w.node1 as usize].pos.y + s.nodes[w.node2 as usize].pos.y)
                + s.origin.y
                - g;
            // The drawn tyre's lowest point: its centre less r along its plane's downward.
            let a = st.axle.normalize_or_zero();
            let down = (Vec3::NEG_Y - a * a.dot(Vec3::NEG_Y)).normalize_or_zero();
            let low = (st.centre + down * w.radius).y - g;
            let rim_low = rims[i]
                .iter()
                .map(|&j| s.nodes[j].pos.y + s.origin.y - g)
                .fold(f32::MAX, f32::min);
            worst_draw = worst_draw.min(low);
            worst_hub = worst_hub.min(hub_y);
            worst_rimnode = worst_rimnode.min(rim_low);
            line += &format!(
                " | w{i} l {:+.3} hub {:.3} draw {:+.3} rim {:+.3} F {:6.1}kN {}",
                st.length,
                hub_y,
                low,
                if rim_low == f32::MAX {
                    f32::NAN
                } else {
                    rim_low
                },
                st.load / 1e3,
                if st.contact { "c" } else { "-" }
            );
        }
        if k % every == 0 {
            let p = lab.bench.plates.first();
            println!(
                "t {:5.2} plate {:.3} m {:7.1} kN broken {:4}{line}",
                lab.bench.t,
                p.map_or(0.0, |p| p.travel),
                p.map_or(0.0, |p| p.force / 1e3),
                lab.bench.car.beams_broken
            );
        }
    }
    println!(
        "worst: drawn tyre lowest {:+.3} m, hub height {:.3} m, rim node lowest {:+.3} m",
        worst_draw, worst_hub, worst_rimnode
    );
    Ok(())
}
