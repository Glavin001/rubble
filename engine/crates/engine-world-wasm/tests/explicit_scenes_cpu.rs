//! The crash lab's scenes on the CPU reference (`ExplicitSim::spawn_cpu` +
//! `ExplicitBeamsWorld`): a page's scene link reproduced without a GPU.
use engine_softbody::authoring::{build, style, BuildParams, Resolution};
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_world_wasm::ExplicitSim;

fn showcase_truck() -> String {
    // (TB_STYLE: `sedan1`, the one-tier build.)
    let name = std::env::var("TB_STYLE").unwrap_or_else(|_| "sedan".into());
    build(&style(&name).unwrap(), &BuildParams { resolution: Resolution::Medium, strength: 1.0, paint: None })
        .unwrap()
        .truck
}

fn env(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// `softbody.html?solver=explicit&car=showcase&xscene=tbone&speed=118&carSpeed=61&at=0&angle=90`
/// (reported 2026-11-14: the struck car's kinetic energy left every bound at
/// t = 1.17 s; on this reference a 55 g skin node of the struck car left a frame at
/// 2,810 m/s and the cars' kinetic energy rose 285 kJ in one frame). Two cars on flat
/// ground with no engine have no source of energy, so, asserted: every node stays
/// finite; the cars' kinetic energy never rises more than 1 % of its start above its
/// low so far (what stored elastic energy can give back is far less); and no node
/// ends a frame faster than 150 m/s (the cars close at 37 m/s).
#[test]
fn showcase_tbone_118_into_61_stays_bounded() {
    let spec = serde_json::json!({
        "vehicles": [{ "truck": showcase_truck(), "at": [0.0, 0.0], "front": true, "yaw": 0.0 }],
        "scene": "tbone",
        "speed": env("TB_KMH", 118.0) / 3.6,
        "carSpeed": env("TB_CAR", 61.0) / 3.6,
        "at": env("TB_AT", 0.0),
        "angle": env("TB_ANG", 90.0),
    });
    let mut cars = ExplicitSim::spawn_cpu(&spec).unwrap();
    // (TB_RULE: `ror`, `two_body` (without the dissipative normal force) or
    // `dissipative` (with it) on both cars, a measurement; the file's rule otherwise.)
    if let Ok(r) = std::env::var("TB_RULE") {
        for c in &mut cars {
            c.two_body_contact = r != "ror";
            c.dissipative_contact = r == "dissipative";
        }
    }
    eprintln!(
        "lightest node {:.3} kg; two-body {}, dissipative {}",
        cars[0].nodes.iter().map(|n| n.mass).fold(f32::MAX, f32::min),
        cars[0].two_body_contact,
        cars[0].dissipative_contact
    );
    let mut w = ExplicitBeamsWorld::new(cars);
    let kinetic = |w: &ExplicitBeamsWorld| -> Vec<f64> {
        w.vehicles
            .iter()
            .map(|v| v.nodes.iter().map(|n| 0.5 * f64::from(n.mass) * f64::from(n.vel.length_squared())).sum())
            .collect()
    };
    let start: f64 = kinetic(&w).iter().sum();
    let secs = env("TB_SECS", 2.0);
    let verbose = std::env::var("TB_LOG").is_ok();
    let mut worst = 0.0f64;
    let (mut low, mut rise, mut fastest) = (f64::MAX, 0.0f64, 0.0f32);
    // (TB_TRACE=speed m/s: step by step, the first node past that speed: its force
    // split by source the step before, and its beams.)
    let trace = std::env::var("TB_TRACE").ok().and_then(|v| v.parse::<f32>().ok());
    if trace.is_some() {
        w.enable_ledger();
    }
    let mut traced = 0;
    for f in 0..(secs * 60.0) as u32 {
        if let Some(limit) = trace {
            for v in &mut w.vehicles {
                v.check_wings();
                v.rebase_origin();
            }
            for s in 0..33 {
                let before: Vec<Vec<(glam::Vec3, glam::Vec3)>> = w.vehicles.iter().map(|v| v.nodes.iter().map(|n| (n.pos, n.vel)).collect()).collect();
                let parts: Vec<Option<[Vec<glam::Vec3>; 4]>> = w.vehicles.iter().map(|v| v.trace.as_ref().map(|t| t.parts.clone())).collect();
                let beams: Vec<Vec<engine_softbody::explicit_beams::Beam>> = w.vehicles.iter().map(|v| v.beams.clone()).collect();
                w.step();
                for (vi, v) in w.vehicles.iter().enumerate() {
                    for (i, n) in v.nodes.iter().enumerate() {
                        let dv = (n.vel - before[vi][i].1).length();
                        if n.vel.length() > limit && dv > 0.5 * limit && traced < 6 {
                            traced += 1;
                            eprintln!(
                                "frame {f} step {s} (t {:.4}): vehicle {vi} node {i} mass {:.3} flags {:#x}: v {:?} -> {:?} (dv {dv:.1}, impulse-force {:.0} N)",
                                (f64::from(f) * 33.0 + f64::from(s)) / 2000.0, n.mass, n.flags, before[vi][i].1, n.vel, dv * n.mass * 2000.0
                            );
                            eprintln!(
                                "   rules: two-body {} dissipative {} anti-clip {} BeamNG profile {}; vehicle contact on it this step {:?}",
                                v.two_body_contact, v.dissipative_contact, v.anti_clip, v.beamng_profile,
                                w.ledger.as_ref().map(|l| l.vehicle_contact(vi, i))
                            );
                            if let Some(p) = &parts[vi] {
                                eprintln!("   forces consumed [air, beams, self, slides]: {:?}", [p[0].get(i), p[1].get(i), p[2].get(i), p[3].get(i)]);
                            }
                            for &bi in &v.incidence[i] {
                                let b0 = &beams[vi][bi as usize];
                                let o = if b0.a as usize == i { b0.b } else { b0.a } as usize;
                                let len = (before[vi][i].0 - before[vi][o].0).length();
                                let rate = (before[vi][i].1 - before[vi][o].1).dot((before[vi][i].0 - before[vi][o].0) / len.max(1e-9));
                                eprintln!(
                                    "   beam {bi} to {o} (m {:.3}): kind {:?} {:?} k {:.0} c {:.0} rest {:.4} rest0 {:.4} len {len:.4} rate {rate:.1} m/s force {:.0} broken {} -> {}",
                                    v.nodes[o].mass, b0.kind, b0.ty, b0.k, b0.c, b0.rest, b0.rest0, b0.force, b0.broken, v.beams[bi as usize].broken
                                );
                            }
                        }
                    }
                }
            }
        } else {
            w.frame(33);
        }
        let k = kinetic(&w);
        let t = f64::from(f + 1) * 33.0 / 2000.0;
        if verbose {
            let fast = |v: &engine_softbody::explicit_beams::ExplicitBeams| {
                v.nodes.iter().enumerate().map(|(i, n)| (n.vel.length(), i)).fold((0.0f32, 0), |a, b| if b.0 > a.0 { b } else { a })
            };
            eprintln!(
                "t {t:.3}: KE {:.0} / {:.0} kJ; fastest node {:?} / {:?}",
                k[0] / 1e3,
                k[1] / 1e3,
                fast(&w.vehicles[0]),
                fast(&w.vehicles[1])
            );
        }
        fastest = w.vehicles.iter().flat_map(|v| &v.nodes).map(|n| n.vel.length()).fold(fastest, f32::max);
        let total: f64 = k.iter().sum();
        assert!(total.is_finite() && w.vehicles.iter().all(|v| v.nodes.iter().all(|n| n.pos.is_finite())), "not finite at t = {t:.3} s");
        worst = worst.max(total / start);
        low = low.min(total);
        rise = rise.max(total - low);
    }
    let broken: Vec<usize> = w.vehicles.iter().map(|v| v.beams.iter().filter(|b| b.broken).count()).collect();
    eprintln!("broken beams {broken:?}");
    eprintln!("largest rise of the cars' kinetic energy over its low so far {:.1} kJ; fastest node at a frame's end {fastest:.0} m/s", rise / 1e3);
    assert!(rise <= 0.01 * start, "the cars' kinetic energy rose {:.1} kJ above its low ({:.0} kJ at the start)", rise / 1e3, start / 1e3);
    assert!(fastest <= 150.0, "a node ended a frame at {fastest:.0} m/s");
    eprintln!("kinetic energy peaked at {:.3} of the start ({:.0} kJ)", worst, start / 1e3);
}
