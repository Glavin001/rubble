//! Ball trajectory trace: why did a ball take the path it took?
//!
//! Fires balls into a lab scene as a click does (the orbit camera's aim at
//! points over the view), steps one tick at a time, and for every live ball
//! and every tick records:
//! - its position, velocity, and the velocity change beyond gravity (what the
//!   contacts and CCD did this tick);
//! - every piece its swept sphere touched this tick (exact segment–hull
//!   distance at the pieces' step-start poses), with that piece's state
//!   (asleep, awake, moved) and mass;
//! - its CCD slot record: time of impact, partner, TOI normal, whether the
//!   partner was held still (asleep), the impact response, the post-solve clamp.
//!
//! Then prints, per ball, every tick where its velocity changed by more than
//! `TRACE_DV` m/s beyond gravity: the turn angle, the vertical kick, and what
//! it touched; and flags "phantom" changes (no piece within reach) and
//! "upward" kicks off a horizontal impact. The full per-tick record goes to
//! `TRACE_OUT` (default /tmp/ball_trace.jsonl, one JSON object per line).
//!
//!   cargo test --release -p engine-viewer --test ball_trace -- --ignored --nocapture
//!   TRACE_SCENE=cuboids TRACE_CCD=0 TRACE_SHOTS=8 cargo test ...

use engine_fracture::samples::{base_scene, layout_scene, sample};
use engine_fracture::scene::Scene;
use engine_fracture::view::Orbit;
use engine_viewer::fracture::FractureSim;
use metal::Device;
use std::io::Write;

#[path = "support/phase.rs"]
mod phase;

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

type V = [f64; 3];
fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn len(a: V) -> f64 {
    dot(a, a).sqrt()
}
fn word(s: &[f32], b: usize, k: usize) -> f32 {
    s[b * 28 + k]
}
fn vec3(s: &[f32], b: usize, k: usize) -> V {
    [0, 1, 2].map(|i| word(s, b, k + i) as f64)
}

/// The scene a sample key builds (procedural ones only: no GLB needed).
fn sample_scene(key: &str) -> Scene {
    let s = sample(key).expect("sample");
    assert!(s.file.is_none(), "TRACE_SCENE must be a procedural sample (cuboid, cuboidGrid, cuboids, cuboids2x)");
    layout_scene(base_scene(None, 0, None).unwrap(), s.layout, true)
}

#[derive(Default, Clone)]
struct Slot {
    toi: Option<f32>,
    partner: Option<u32>,
    normal: V,
    held: bool,
    responded: bool,
    clamped: bool,
    clamp_mu: Option<f32>,
}

fn slots(sim: &FractureSim) -> std::collections::HashMap<u32, Slot> {
    use engine_schema::{CCD_BASE, CCD_BODY_BASE, CCD_CARRY_BASE, CCD_CARRY_WORDS};
    let words = |at: u32, k: usize| unsafe {
        std::slice::from_raw_parts((sim.bufs[21].contents() as *const u32).add(at as usize), k).to_vec()
    };
    let n = words(CCD_BASE, 1)[0] as usize;
    let mut out = std::collections::HashMap::new();
    for k in 0..n {
        let s = words(CCD_BODY_BASE + 4 * k as u32, 4);
        let c = words(CCD_CARRY_BASE + CCD_CARRY_WORDS * k as u32, 32);
        let f = |i: usize| f32::from_bits(c[i]);
        let has_toi = s[1] != 0x7F80_0000 && s[1] != u32::MAX;
        out.insert(
            s[0],
            Slot {
                toi: has_toi.then(|| f32::from_bits(s[1])),
                partner: (has_toi && s[3] & 1 != 0).then_some(c[7]),
                normal: [f(29) as f64, f(30) as f64, f(31) as f64],
                held: s[3] & 1 != 0 && c[15] != 0,
                responded: c[16] != 0,
                clamped: c[11] != 0,
                clamp_mu: (c[8] != 0x7F80_0000).then(|| f(8)),
            },
        );
    }
    out
}

#[test]
#[ignore = "diagnostic: traces balls fired into a lab scene"]
fn trace_ball_trajectories() {
    let key: String = env("TRACE_SCENE", "cuboids".to_string());
    let ccd = env("TRACE_CCD", 1u32) != 0;
    let shots = env("TRACE_SHOTS", 6u32);
    let every = env("TRACE_EVERY", 40u32);
    let ticks = env("TRACE_TICKS", 360u32);
    let speed = env("TRACE_SPEED", 1.0f64);
    let dv_min = env("TRACE_DV", 2.0f64);
    let out_path: String = env("TRACE_OUT", "/tmp/ball_trace.jsonl".to_string());
    let mut scene = sample_scene(&key);
    // TRACE_FLAGS_OFF: solver flags to clear (diagnosis), e.g. 131072 (deferred wake).
    let off: u32 = env("TRACE_FLAGS_OFF", 0u32);
    scene.params.flags &= !off;
    let n = scene.bodies.len();
    let dt = scene.params.dt as f64;
    let g = [0.0, scene.params.gravity_y as f64, 0.0];
    let start = scene.projectile_start as usize;
    let pieces = phase::pieces(&scene);
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let mut sim = FractureSim::new(&device, &queue, &scene).unwrap();
    sim.plan.set_projectile_ccd(ccd);
    sim.plan.set_projectile_speed_scale(speed).unwrap();
    let radius = scene.projectile_radius;
    println!(
        "{key}: {} pieces, ball r {radius:.2} m, {:.0} kg, {:.0} m/s, dt {dt:.4}, CCD {ccd}",
        start - 1,
        scene.projectile_mass,
        sim.plan.shot_speed() * speed
    );
    // Shots as clicks: the default camera, points over the middle of the view.
    let orbit = Orbit::new(&scene);
    let mut k: u64 = 0x5eed;
    let mut rnd = || {
        k = k.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((k >> 33) & 0xFFFF) as f64 / 65535.0
    };
    let mut out = std::fs::File::create(&out_path).unwrap();
    let mut prev = sim.snapshot(n).unwrap();
    let mut live: Vec<(usize, u32)> = Vec::new(); // (body, fired at tick)
    let mut events: Vec<String> = Vec::new();
    let (mut phantom, mut upward, mut unbacked) = (0, 0, 0);
    // Clamped ticks (the post-solve check pulled the ball back) and the
    // fraction of its step's travel they took away.
    let (mut clamps, mut clamp_lost) = (0u32, 0.0f64);
    let (mut total_impulse, mut total_absorbed) = (0.0f64, 0.0f64);
    for t in 0..ticks {
        if t % every == 0 && t / every < shots {
            let ray = orbit.aim(0.3 + 0.4 * rnd(), 0.35 + 0.3 * rnd(), 16.0 / 9.0, radius);
            let id = sim.launch(ray.origin, ray.direction).unwrap() as usize;
            live.retain(|&(b, _)| b != id);
            live.push((id, t));
            prev = sim.snapshot(n).unwrap();
        }
        sim.step(1).unwrap();
        let now = sim.snapshot(n).unwrap();
        let ccd_slots = slots(&sim);
        for &(ball, fired) in &live {
            let (p0, p1) = (vec3(&prev, ball, 0), vec3(&now, ball, 0));
            if p0[1] < -100.0 {
                continue;
            }
            let (v0, v1) = (vec3(&prev, ball, 12), vec3(&now, ball, 12));
            let dv = sub(sub(v1, v0), g.map(|x| x * dt));
            // Pieces the swept sphere touched (step-start poses; a piece the
            // solve moved is also checked at its end pose).
            let mut touched = Vec::new();
            let mut absorbed = 0.0f64;
            let seg = sub(p1, p0);
            for pc in &pieces {
                let (x0, q0) = phase::pose(&prev, pc.body);
                let (x1, q1) = phase::pose(&now, pc.body);
                let w = sub(x0, p0);
                let tt = (dot(w, seg) / dot(seg, seg).max(1e-20)).clamp(0.0, 1.0);
                let near = len(sub([p0[0] + seg[0] * tt, p0[1] + seg[1] * tt, p0[2] + seg[2] * tt], x0));
                if near > pc.bound + radius + 0.5 + len(sub(x1, x0)) {
                    continue;
                }
                let c = &scene.colliders[pc.collider];
                let d0 = phase::segment_hull_distance(p0, p1, &phase::piece_vertices(&scene, c, x0, q0));
                let d1 = phase::segment_hull_distance(p0, p1, &phase::piece_vertices(&scene, c, x1, q1));
                let d = d0.min(d1);
                if d <= radius + 0.05 {
                    let flags = word(&prev, pc.body, 11).to_bits();
                    let f1 = word(&now, pc.body, 11).to_bits();
                    let state = format!("{}{}->{}", if flags & 4 != 0 { "asleep" } else { "awake" }, if flags & 16 != 0 { "+pending" } else { "" }, if f1 & 4 != 0 { "asleep" } else if f1 & 16 != 0 { "pending" } else { "awake" });
                    let moved = len(sub(x1, x0));
                    let mass = 1.0 / (word(&prev, pc.body, 3) as f64).max(1e-12);
                    touched.push(format!("{}:{state}:moved {moved:.2}:{mass:.0}kg:d {d:.2}", pc.body));
                    absorbed += mass * len(sub(vec3(&now, pc.body, 12), vec3(&prev, pc.body, 12)));
                }
            }
            // The ground plane (y = 0) within reach.
            if p0[1].min(p1[1]) - radius < 0.05 {
                touched.push("GROUND".to_string());
            }
            let slot = ccd_slots.get(&(ball as u32)).cloned().unwrap_or_default();
            let speed0 = len(v0);
            let turn = if speed0 > 1e-3 && len(v1) > 1e-3 { (dot(v0, v1) / (speed0 * len(v1))).clamp(-1.0, 1.0).acos().to_degrees() } else { 0.0 };
            let line = format!(
                "{{\"tick\":{t},\"ball\":{ball},\"age\":{},\"p\":[{:.3},{:.3},{:.3}],\"v\":[{:.2},{:.2},{:.2}],\"dv\":[{:.2},{:.2},{:.2}],\"turn\":{turn:.1},\"touched\":{:?},\"toi\":{},\"partner\":{},\"normal\":[{:.2},{:.2},{:.2}],\"held\":{},\"responded\":{},\"clamped\":{},\"clamp_mu\":{}}}",
                t - fired, p1[0], p1[1], p1[2], v1[0], v1[1], v1[2], dv[0], dv[1], dv[2], touched,
                slot.toi.map_or("null".into(), |x| format!("{x:.3}")),
                slot.partner.map_or("null".into(), |x| x.to_string()),
                slot.normal[0], slot.normal[1], slot.normal[2],
                slot.held, slot.responded, slot.clamped,
                slot.clamp_mu.map_or("null".into(), |x| format!("{x:.3}")),
            );
            writeln!(out, "{line}").unwrap();
            // The ball's impulse beyond gravity against the momentum the pieces
            // it touched gained: the rest went into the ground or a body the
            // solve held still.
            let impulse = scene.projectile_mass * len(dv);
            let touches_ground = touched.iter().any(|t| t == "GROUND");
            if len(dv) > dv_min && !touches_ground {
                total_impulse += impulse;
                total_absorbed += absorbed.min(impulse);
                if impulse > 3.0 * absorbed + 1e4 {
                    unbacked += 1;
                }
            }
            if slot.clamped {
                clamps += 1;
                clamp_lost += 1.0 - slot.clamp_mu.unwrap_or(1.0).clamp(0.0, 1.0) as f64;
            }
            if len(dv) > dv_min {
                let horizontal_hit = slot.toi.is_some() && slot.normal[1].abs() < 0.5;
                let phantom_here = touched.is_empty();
                let upward_here = dv[1] > 0.5 * len(dv) && horizontal_hit;
                phantom += phantom_here as u32;
                upward += upward_here as u32;
                events.push(format!(
                    "ball {ball} tick {t} (age {}): impulse {:.0} kN·s vs pieces gained {:.0} | |v| {:.1}->{:.1} m/s, turn {turn:.0}°, dv beyond g ({:.1},{:.1},{:.1}), toi {} partner {} n ({:.2},{:.2},{:.2}){}{}{} | touched {}{}{}",
                    t - fired, impulse / 1e3, absorbed / 1e3, speed0, len(v1), dv[0], dv[1], dv[2],
                    slot.toi.map_or("-".into(), |x| format!("{x:.2}")),
                    slot.partner.map_or("-".into(), |x| x.to_string()),
                    slot.normal[0], slot.normal[1], slot.normal[2],
                    if slot.held { " held" } else { "" },
                    if slot.responded { " respond" } else { "" },
                    if slot.clamped { format!(" CLAMP mu {:.2}", slot.clamp_mu.unwrap_or(1.0)) } else { String::new() },
                    if touched.is_empty() { "NOTHING".to_string() } else { touched.join(", ") },
                    if phantom_here { "  <- PHANTOM" } else { "" },
                    if upward_here { "  <- UPWARD off a horizontal hit" } else { "" },
                ));
            }
        }
        prev = now;
    }
    for e in &events {
        println!("{e}");
    }
    println!("{} velocity changes over {dv_min} m/s: {phantom} with nothing touched, {upward} upward off a horizontal hit; per-tick records in {out_path}", events.len());
    println!("{clamps} clamped ticks (ball pulled back), {:.1} ticks of travel lost to them", clamp_lost);
    println!(
        "off the ground: ball impulse {:.0} kN·s, touched pieces gained {:.0} kN·s ({:.1}%); {unbacked} ticks where the ball lost over 3x what its pieces gained",
        total_impulse / 1e3,
        total_absorbed / 1e3,
        100.0 * total_absorbed / total_impulse.max(1e-9)
    );
}
