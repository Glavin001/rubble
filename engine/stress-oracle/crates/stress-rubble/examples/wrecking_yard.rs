//! A Wrecking Yard structure (`stress_ref::wrecking`) hit by one of the yard's ball
//! presets along the structure's line of attack, natively, as the lab's stress modes run
//! it (`scenes::wrecking_yard`, the resident GpuWorld for `blast_parity`, the GPU-resident
//! threshold world for `threshold`): the hero check of the progressive collapse.
//!
//!   cargo run --release -p stress-rubble --features gpu --example wrecking_yard -- [STRUCTURE] [BALL] [FRAMES]
//!
//! STRUCTURE `wy_frame10` (default), `wy_round_tower`, `wy_brick_tower`,
//! `rr_storefront_row`, `ft_apartment_block` or `ft_glass_tower`; BALL `ball10t`,
//! `ball50t` (default), `sphere15m`, `car2t` or `jet`; FRAMES at 60 Hz (default 600). The
//! ball is fired at frame 10. `STRESS_PROFILE=threshold`: the threshold mode.
//! `STRESS_SHOT_MASS=4`, `STRESS_SHOT_SPEED=2`: the ball's mass or speed times this (the
//! lab's sliders).
//!
//! Every `EVERY` frames: bonds broken, fragments, ms per frame; then, per storey (the
//! chunks' `storey-N`, `floor-N` or `course-N` group), the frame at which a tenth of its
//! chunks had dropped a metre and how many had at the end: a collapse that climbs shows
//! later frames on higher storeys, and a standing core storeys with few dropped. Last,
//! each storey's mean displacement (m) at the end, and its tilt (degrees, the mean
//! horizontal drift over the drift of the storey below, per storey height): an upper
//! block that tilts away as one drifts more the higher it is.
use std::collections::BTreeMap;

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;
use stress_ref::wrecking;

const EVERY: usize = 30;
const FIRE_AT: usize = 10;
/// A chunk has dropped once it is this far (m) below where it stood.
const DROP: f64 = 1.0;

fn main() {
    let mut args = std::env::args().skip(1);
    let key = args.next().unwrap_or_else(|| "wy_frame10".into());
    let ball_key = args.next().unwrap_or_else(|| "ball50t".into());
    let frames: usize = args.next().map_or(600, |f| f.parse().expect("FRAMES"));
    let p = std::env::var("STRESS_PROFILE").ok().map(|v| Profile::parse(&v).unwrap_or_else(|| panic!("STRESS_PROFILE={v}: no such profile"))).unwrap_or(Profile::BlastParity);
    let env = |k: &str| std::env::var(k).ok().map_or(1.0, |v| v.parse::<f64>().unwrap_or_else(|_| panic!("{k}")));
    let mut ball = *wrecking::ball(&ball_key).unwrap_or_else(|| panic!("no ball {ball_key}"));
    ball.mass *= env("STRESS_SHOT_MASS");
    ball.speed *= env("STRESS_SHOT_SPEED");
    let structure = wrecking::structure(&key).unwrap_or_else(|| panic!("no structure {key}"));
    let scene = stress_rubble::scenes::wrecking_yard(&key, p, frames as f64 / 60.0);
    // `STRESS_WY_AIM_X=3`, `STRESS_WY_AIM_Z=12`: aim at this x or height instead (m, the
    // same direction).
    let mut aim = structure.aim;
    for (k, var) in [(0, "STRESS_WY_AIM_X"), (2, "STRESS_WY_AIM_Z")] {
        if let Ok(v) = std::env::var(var) {
            aim.target[k] = v.parse().unwrap_or_else(|_| panic!("{var}"));
        }
    }
    let shot = wrecking::launch(&ball, &aim, "ball");
    // Each chunk's storey (its group), in the scene's order.
    let storey: Vec<String> = scene.bodies[0].chunks.iter().map(|c| c.groups.iter().find(|g| g.starts_with("storey-") || g.starts_with("floor-") || g.starts_with("course-")).cloned().unwrap_or_else(|| "footing".into())).collect();
    let origin: Vec<[f64; 3]> = scene.bodies[0].chunks.iter().map(|c| c.center).collect();
    let start: Vec<f64> = origin.iter().map(|c| c[2]).collect();
    // Each chunk's displacement at the last report.
    let mut moved: Vec<[f64; 3]> = vec![[0.0; 3]; origin.len()];
    // The top storey's mean displacement and its tilt (the angle its chunks' mean
    // horizontal drift makes over its height above the ground): an upper block that tilts
    // away leans before it falls.
    let number = |s: &str| s.rsplit('-').next().and_then(|n| n.parse::<usize>().ok()).unwrap_or(0);
    let top_storey = storey.iter().map(|s| number(s)).max().unwrap_or(0);
    let top_chunks: Vec<usize> = (0..storey.len()).filter(|&k| number(&storey[k]) == top_storey && storey[k] != "footing").collect();
    let top_height = top_chunks.iter().map(|&k| origin[k][2]).sum::<f64>() / top_chunks.len().max(1) as f64;
    let top_moved = |moved: &[[f64; 3]]| -> String {
        let n = top_chunks.len().max(1) as f64;
        let d = top_chunks.iter().fold([0.0; 3], |a, &k| [a[0] + moved[k][0] / n, a[1] + moved[k][1] / n, a[2] + moved[k][2] / n]);
        format!("({:.2}, {:.2}, {:.2}) m, lean {:.1} deg", d[0], d[1], d[2], d[0].hypot(d[1]).atan2(top_height + d[2]).to_degrees())
    };
    println!("{key}: {} chunks, {} bonds, {} ({:.0} t, r {:.2} m, {:.0} m/s{}), {}, {frames} frames", start.len(), scene.bodies[0].bonds.len(), ball.key, ball.mass / 1e3, ball.radius, ball.speed, if ball.rolling { ", rolling" } else { "" }, p.name());
    let gpu = Gpu::new().expect("GPU");
    // First frame each storey had a tenth of its chunks dropped.
    let mut first: BTreeMap<String, usize> = BTreeMap::new();
    let mut note = |f: usize, z: &[f64]| -> BTreeMap<String, (usize, usize)> {
        let mut count: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        for (k, s) in storey.iter().enumerate() {
            let e = count.entry(s.clone()).or_default();
            e.1 += 1;
            if z[k] < start[k] - DROP {
                e.0 += 1;
            }
        }
        for (s, (d, n)) in &count {
            if *d * 10 >= *n && *d > 0 {
                first.entry(s.clone()).or_insert(f);
            }
        }
        count
    };
    let mut ms = Vec::with_capacity(frames);
    let mut last = BTreeMap::new();
    if p == Profile::Threshold {
        let mut w = stress_rubble::threshold::ThresholdWorld::new(&gpu, &scene, ball.radius, ball.mass).expect("threshold world");
        for f in 0..frames {
            if f == FIRE_AT {
                w.launch(&gpu, &shot);
            }
            let t = std::time::Instant::now();
            w.step_frame(&gpu).expect("frame");
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            let st = w.poll(&gpu);
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            if (f + 1) % EVERY == 0 || f + 1 == frames {
                let centres = w.chunk_centres(&gpu);
                moved = centres.iter().zip(&origin).map(|((c, _), o)| [c[0] - o[0], c[1] - o[1], c[2] - o[2]]).collect();
                let z: Vec<f64> = centres.iter().map(|(c, _)| c[2]).collect();
                last = note(f, &z);
                println!("frame {:4}: broken {:5}, bodies {:5}, {:6.2} ms/frame (last {EVERY}), top storey moved {}", f + 1, st.broken, st.bodies, mean(&ms[ms.len().saturating_sub(EVERY)..]), top_moved(&moved));
            }
            if st.error != 0 || st.engine_overflow != 0 {
                panic!("frame {f}: fracture errors {:#x}, engine overflow {:#x}", st.error, st.engine_overflow);
            }
        }
    } else {
        let mut e = stress_rubble::RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble");
        let (iterations, substeps) = p.engine_policy();
        e.set_policy(iterations, substeps);
        let mut w = GpuWorld::with_resident_engine(&gpu, &scene, e).expect("world");
        w.redo_fractured_frames = false;
        w.set_impact_islands(true);
        w.apply_profile(p);
        let trace: usize = std::env::var("STRESS_WY_TRACE").ok().map_or(0, |v| v.parse().expect("STRESS_WY_TRACE"));
        let mut was_broken = vec![false; scene.bodies[0].bonds.len()];
        for f in 0..frames {
            if f == FIRE_AT {
                let imp = stress_ref::world::Impactor::from_desc(&shot, &w.scene);
                w.add_impactor(&gpu, imp).expect("shot");
            }
            let t = std::time::Instant::now();
            w.step_frame(&gpu).expect("frame");
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            if trace > 0 && (f + 1) % trace == 0 {
                // `STRESS_WY_TRACE=N`: every N frames, the bonds broken since, by storey and
                // the kinds of chunk they joined.
                w.sync(&gpu);
                let bonds = &scene.bodies[0].bonds;
                let kind = |c: usize| scene.bodies[0].chunks[c].groups.first().cloned().unwrap_or_default();
                let mut by: BTreeMap<String, usize> = BTreeMap::new();
                for (i, b) in w.solver.mirror.bonds[0].iter().enumerate() {
                    let broken = !b.joint.connected(b.rebar.is_some());
                    if broken && !was_broken[i] {
                        let (a, c) = (bonds[i].a, bonds[i].b);
                        *by.entry(format!("{} {}-{}", storey[a], kind(a), kind(c))).or_default() += 1;
                    }
                    was_broken[i] = broken;
                }
                if !by.is_empty() {
                    println!("  frame {}: {by:?}", f + 1);
                }
            }
            if (f + 1) % EVERY == 0 || f + 1 == frames {
                w.sync(&gpu);
                let o = w.observation();
                let chunks = o.bodies.values().next().expect("the structure");
                moved = chunks.iter().map(|c| c.displacement).collect();
                let z: Vec<f64> = chunks.iter().zip(&start).map(|(c, z0)| z0 + c.displacement[2]).collect();
                last = note(f, &z);
                println!("frame {:4}: broken {:5}, fragments {:5}, {:6.2} ms/frame (last {EVERY}), top storey moved {}", f + 1, o.values["broken_bonds"], o.values["fragments"], mean(&ms[ms.len().saturating_sub(EVERY)..]), top_moved(&moved));
            }
        }
    }
    let mut sorted = ms[FIRE_AT..].to_vec();
    sorted.sort_by(f64::total_cmp);
    println!("after the shot: median {:.2} ms/frame, p95 {:.2}, worst {:.2}", sorted[sorted.len() / 2], sorted[sorted.len() * 95 / 100], sorted[sorted.len() - 1]);
    // Storeys in height order.
    let mut storeys: Vec<&String> = last.keys().collect();
    storeys.sort_by_key(|s| s.rsplit('-').next().and_then(|n| n.parse::<usize>().ok()).unwrap_or(0));
    println!("storey      dropped   first frame a tenth had dropped {DROP} m");
    for s in &storeys {
        let (d, n) = last[*s];
        println!("{s:12} {d:4}/{n:<4}  {}", first.get(*s).map_or("-".into(), |f| format!("{}", f + 1)));
    }
    // Each storey's mean displacement and height; its tilt from the drift over the storey
    // below.
    // `STRESS_WY_KIND=core`: of the chunks of that type only (`groups[0]`).
    let only = std::env::var("STRESS_WY_KIND").ok();
    let kind = |k: usize| scene.bodies[0].chunks[k].groups.first().cloned().unwrap_or_default();
    println!("storey      mean displacement (x, y, z) m       tilt over the storey below{}", only.as_ref().map_or(String::new(), |t| format!(" ({t} only)")));
    let mut below: Option<([f64; 3], f64)> = None;
    for s in storeys {
        let ks: Vec<usize> = (0..storey.len()).filter(|&k| &storey[k] == s && only.as_ref().is_none_or(|t| kind(k) == *t)).collect();
        if ks.is_empty() {
            continue;
        }
        let n = ks.len().max(1) as f64;
        let d = ks.iter().fold([0.0; 3], |a, &k| [a[0] + moved[k][0] / n, a[1] + moved[k][1] / n, a[2] + moved[k][2] / n]);
        let z = ks.iter().map(|&k| origin[k][2] + moved[k][2]).sum::<f64>() / n;
        let tilt = below.map_or("-".into(), |(b, zb): ([f64; 3], f64)| format!("{:5.1} deg", ((d[0] - b[0]).hypot(d[1] - b[1])).atan2(z - zb).to_degrees()));
        println!("{s:12} ({:7.2}, {:7.2}, {:7.2})          {tilt}", d[0], d[1], d[2]);
        below = Some((d, z));
    }
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len().max(1) as f64
}
