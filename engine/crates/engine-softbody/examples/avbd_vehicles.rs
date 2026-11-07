//! The AVBD vehicle mode against the explicit reference on the CPU
//! (`engine/docs/avbd-vehicles.md`): the same car and scenes on the explicit
//! 2 kHz step and on AVBD at several rates and sweep counts.
//!
//! ```bash
//! cargo run --release -p engine-softbody --example avbd_vehicles -- \
//!     --car etk_kc6 --configs explicit,120x20,240x20 --scenes rest,bump,wall,pole
//! ```
//!
//! A config is `explicit` or `<hz>x<iterations>[c<rho>]` (`c`: Chebyshev).
//! Results: one JSON per car/config/scene under `--out`
//! (default `engine/docs/measurements/avbd-vehicles/`), and a summary line each
//! on stdout.
use engine_softbody::avbd_beams::{AvbdSettings, AvbdState};
use engine_softbody::explicit_beams::{ExplicitBeams, STEP};
use engine_softbody::vehicle_compare::{rigid_fit, scene_pole, scene_wall, Car, Manifest};
use glam::Vec3;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Instant;

fn parse(cfg: &str) -> Option<AvbdSettings> {
    if cfg == "explicit" {
        return None;
    }
    let (hz, rest) = cfg.split_once('x').expect("config <hz>x<iters>");
    let (it, rho) = match rest.split_once('c') {
        Some((a, b)) => (a, b.parse().unwrap()),
        None => (rest, 0.0),
    };
    Some(AvbdSettings {
        hz: hz.parse().unwrap(),
        iterations: it.parse().unwrap(),
        chebyshev_rho: rho,
        ..Default::default()
    })
}

/// The body's nodes: every part's (tyres excluded) but the wheels', hubs' and
/// brakes' (they turn).
fn body_nodes(car: &Car) -> Vec<u32> {
    let mut v: Vec<u32> = car
        .parts
        .iter()
        .filter(|(n, _)| {
            let n = n.to_lowercase();
            !["wheel", "tire", "tyre", "hub", "brake", "rim", "spindle"]
                .iter()
                .any(|w| n.contains(w))
        })
        .flat_map(|(_, ids)| ids.iter().copied())
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// RMS of the nodes' displacement about the rigid fit of the body (mm).
fn deform_rms(rest: &[Vec3], now: &[Vec3], ids: &[u32]) -> f32 {
    let fit = rigid_fit(rest, now, ids);
    let mut acc = 0.0f64;
    for &i in ids {
        let d = fit.to_rest(now[i as usize]) - rest[i as usize];
        acc += f64::from(d.length_squared());
    }
    ((acc / ids.len() as f64).sqrt() * 1e3) as f32
}

fn pos(s: &ExplicitBeams) -> Vec<Vec3> {
    s.nodes.iter().map(|n| n.pos).collect()
}

fn com(s: &ExplicitBeams) -> Vec3 {
    let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
    s.nodes.iter().map(|n| n.pos * n.mass).sum::<Vec3>() / m
}

/// At rest from the settled template for `seconds`: sag, drift, the shape's
/// change, the motion left.
fn scene_rest(car: &Car, seconds: f32) -> Value {
    let t0 = Instant::now();
    let mut s = car.template.clone();
    // From still (the template keeps the last of its settle's motion).
    for q in &mut s.nodes {
        q.vel = Vec3::ZERO;
    }
    let rest = pos(&s);
    let ids = body_nodes(car);
    let c0 = com(&s);
    let n = (seconds / STEP).round() as usize;
    let mut c_half = c0;
    let mut deform_max = 0.0f32;
    for k in 0..n {
        s.step();
        if k + 1 == n / 2 {
            c_half = com(&s);
        }
        if k % 20 == 0 {
            deform_max = deform_max.max(deform_rms(&rest, &pos(&s), &ids));
        }
    }
    let c = com(&s);
    let ke: f32 = s
        .nodes
        .iter()
        .map(|q| 0.5 * q.mass * q.vel.length_squared())
        .sum();
    let (broken, bent) = s.damage();
    let mut by_part: Vec<(String, f32, Vec3)> = car
        .parts
        .iter()
        .map(|(name, ids)| {
            let ke = ids
                .iter()
                .map(|&i| 0.5 * s.nodes[i as usize].mass * s.nodes[i as usize].vel.length_squared())
                .sum::<f32>();
            let m = ids
                .iter()
                .map(|&i| s.nodes[i as usize].mass)
                .sum::<f32>()
                .max(1e-6);
            let v = ids
                .iter()
                .map(|&i| s.nodes[i as usize].vel * s.nodes[i as usize].mass)
                .sum::<Vec3>()
                / m;
            (name.clone(), ke, v)
        })
        .collect();
    by_part.sort_by(|a, b| b.1.total_cmp(&a.1));
    by_part.truncate(6);
    json!({
        "ke_by_part": by_part.iter().map(|(n, k, v)| json!([n, k, [v.x, v.y, v.z]])).collect::<Vec<_>>(),
        "sag_mm": (c0.y - c.y) * 1e3, "creep_2nd_half_mm": (c - c_half).length() * 1e3,
        "drift_xz_mm": Vec3::new(c.x - c0.x, 0.0, c.z - c0.z).length() * 1e3, "drift_mm": [(c.x - c0.x) * 1e3, (c.y - c0.y) * 1e3, (c.z - c0.z) * 1e3],
        "shape_change_rms_mm": deform_rms(&rest, &pos(&s), &ids), "shape_change_max_rms_mm": deform_max,
        "ke_end_j": ke, "vmax_end": s.nodes.iter().map(|q| q.vel.length()).fold(0.0, f32::max),
        "broken": broken, "bent": bent,
        "finite": s.nodes.iter().all(|q| q.pos.is_finite()),
        "avbd_clamped": s.avbd.as_ref().map(|a| a.stats.clamped),
        "wall_s_per_sim_s": t0.elapsed().as_secs_f32() / seconds,
    })
}

/// A landing: every node given 1 m/s down (a 5 cm drop) from rest; the body's
/// elastic response (deformation about its rigid fit, sampled each ms): peak,
/// the oscillation's zero crossings and decay (ringing), the COM bounce.
fn scene_bump(car: &Car, seconds: f32) -> Value {
    let t0 = Instant::now();
    let mut s = car.template.clone();
    let rest = pos(&s);
    let ids = body_nodes(car);
    for q in &mut s.nodes {
        q.vel.y -= 1.0;
    }
    let y0 = com(&s).y;
    let n = (seconds / STEP).round() as usize;
    let mut series = Vec::new();
    let mut ys = Vec::new();
    for k in 0..n {
        s.step();
        if k % 2 == 1 {
            series.push(deform_rms(&rest, &pos(&s), &ids));
            ys.push((com(&s).y - y0) * 1e3);
        }
    }
    let peak = series.iter().copied().fold(0.0, f32::max);
    let tail = series[series.len() * 3 / 4..].iter().sum::<f32>() / (series.len() / 4) as f32;
    // Ringing: sign changes of the COM height's velocity (bounces) after the first minimum.
    let mut crossings = 0;
    let dy: Vec<f32> = ys.windows(2).map(|w| w[1] - w[0]).collect();
    let mut last = 0.0f32;
    for &d in &dy {
        if d.abs() > 0.002 {
            if last != 0.0 && d.signum() != last {
                crossings += 1;
            }
            last = d.signum();
        }
    }
    // Time for the body deformation to settle within 10 % of its swing.
    let settle = series
        .iter()
        .rposition(|&d| (d - tail).abs() > 0.1 * (peak - tail).abs().max(0.05))
        .map_or(0.0, |i| (i + 1) as f32 * 1e-3);
    let ymin = ys.iter().copied().fold(f32::MAX, f32::min);
    json!({
        "deform_peak_mm": peak, "deform_end_mm": tail, "deform_settle_s": settle,
        "com_dip_mm": -ymin, "com_reversals": crossings,
        "deform_ms": series.iter().step_by(5).map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>(),
        "com_mm_5ms": ys.iter().step_by(5).map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>(),
        "finite": s.nodes.iter().all(|q| q.pos.is_finite()),
        "avbd_clamped": s.avbd.as_ref().map(|a| a.stats.clamped),
        "wall_s_per_sim_s": t0.elapsed().as_secs_f32() / seconds,
    })
}

fn get<'a>(v: &'a Value, path: &[&str]) -> &'a Value {
    let mut x = v;
    for p in path {
        x = &x[*p];
    }
    x
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| {
        a.iter()
            .position(|x| x == k)
            .and_then(|i| a.get(i + 1).cloned())
    };
    let car_name = opt("--car").unwrap_or("etk_kc6".into());
    let configs = opt("--configs").unwrap_or("explicit,120x20".into());
    let scenes = opt("--scenes").unwrap_or("rest,bump,wall,pole".into());
    let out = opt("--out").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/measurements/avbd-vehicles")
    });
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let manifest =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/avbd-vehicles-cars.json");
    let m: Manifest =
        serde_json::from_str(&std::fs::read_to_string(&manifest).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let spec = m
        .cars
        .iter()
        .find(|c| c.name == car_name)
        .ok_or("no such car")?;
    let t = Instant::now();
    let mut car = Car::load(spec)?;
    eprintln!("{car_name}: loaded in {:.1} s", t.elapsed().as_secs_f32());
    let base = car.template.clone();
    for cfg in configs.split(',') {
        car.template = base.clone();
        if let Some(set) = parse(cfg) {
            car.template.avbd = Some(Box::new(AvbdState::new(&car.template, set)?));
        }
        for scene in scenes.split(',') {
            let t = Instant::now();
            let r = match scene {
                "rest" => scene_rest(&car, 2.0),
                "bump" => scene_bump(&car, 1.5),
                "wall" => scene_wall(&car, 56.0, 1.0),
                "pole" => scene_pole(&car, 32.0, 1.5),
                _ => return Err(format!("unknown scene {scene}")),
            };
            let wall = t.elapsed().as_secs_f32();
            let line = match scene {
                "rest" => format!(
                    "sag {:.1} mm, drift {} mm, creep {:.2} mm, shape {:.2} mm (max {:.2}), ke {:.2} J, broken {} bent {}",
                    r["sag_mm"].as_f64().unwrap_or(f64::NAN),
                    r["drift_xz_mm"],
                    r["creep_2nd_half_mm"].as_f64().unwrap_or(f64::NAN),
                    r["shape_change_rms_mm"].as_f64().unwrap_or(f64::NAN),
                    r["shape_change_max_rms_mm"].as_f64().unwrap_or(f64::NAN),
                    r["ke_end_j"].as_f64().unwrap_or(f64::NAN),
                    r["broken"],
                    r["bent"]
                ),
                "bump" => format!(
                    "deform peak {:.2} mm end {:.2} settle {:.3} s, com dip {:.1} mm reversals {}",
                    r["deform_peak_mm"].as_f64().unwrap_or(f64::NAN),
                    r["deform_end_mm"].as_f64().unwrap_or(f64::NAN),
                    r["deform_settle_s"].as_f64().unwrap_or(f64::NAN),
                    r["com_dip_mm"].as_f64().unwrap_or(f64::NAN),
                    r["com_reversals"]
                ),
                "wall" => format!(
                    "crush dyn {} res {} springback {} peak_g {} restitution {} t_m {} ms, broken {} bent {}, ripple_end {}",
                    r["dynamic_crush_mm"], r["residual_crush_max_mm"], r["springback_ratio"], r["peak_g"],
                    r["restitution"], r["t_m_ms"], get(&r, &["damage", "broken_beams"]), get(&r, &["damage", "bent_beams"]),
                    r["ripple_end"]
                ),
                _ => {
                    let keys = ["residual_crush_mm", "dynamic_crush_mm", "pole_reach_mm", "peak_g"];
                    let mut s = String::new();
                    for k in keys {
                        if !r[k].is_null() {
                            s += &format!("{k} {} ", r[k]);
                        }
                    }
                    format!(
                        "{s}broken {} bent {}",
                        get(&r, &["damage", "broken_beams"]),
                        get(&r, &["damage", "bent_beams"])
                    )
                }
            };
            println!("{car_name} {cfg:>12} {scene:>5}: {line} [{wall:.1} s]");
            let doc = json!({"car": car_name, "config": cfg, "scene": scene, "wall_s": wall, "result": r});
            std::fs::write(
                out.join(format!("{car_name}-{cfg}-{scene}.json")),
                serde_json::to_string_pretty(&doc).unwrap(),
            )
            .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
