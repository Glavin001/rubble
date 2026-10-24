//! A Wrecking Yard structure (`stress_ref::wrecking`) hit by one of the yard's ball
//! presets along the structure's line of attack, natively, as the lab's stress modes run
//! it (`scenes::wrecking_yard`, the resident GpuWorld for `blast_parity`, the GPU-resident
//! threshold world for `threshold`): the hero check of the progressive collapse.
//!
//!   cargo run --release -p stress-rubble --features gpu --example wrecking_yard -- [STRUCTURE] [BALL] [FRAMES]
//!
//! STRUCTURE `wy_frame10` (default), `wy_round_tower`, `wy_brick_tower` or `bo_viaduct`; BALL `ball10t`,
//! `ball50t` (default) or `sphere15m`; FRAMES at 60 Hz (default 600). The ball is fired at
//! frame 10. `STRESS_PROFILE=threshold`: the threshold mode. `STRESS_SHOT_MASS=4`,
//! `STRESS_SHOT_SPEED=2`: the ball's mass or speed times this (the lab's sliders).
//!
//! Every `EVERY` frames: bonds broken, fragments, ms per frame; then, per storey (the
//! chunks' second group: `storey-N`, `floor-N`, `course-N`, the viaduct's `pier-N` and
//! `span-N`), the frame at which a tenth of its chunks had dropped a metre and how many
//! had at the end: a collapse that climbs shows later frames on higher storeys, and a
//! standing core storeys with few dropped. For the viaduct, every `EVERY` frames, each
//! span that has moved: how far its west and east ends have dropped (its deck chunks at
//! each end) and how many of its bonded chunk pairs are still at their distance (within
//! 5 cm). A span that comes down as one slab shows one end lower than the other and its
//! pairs intact until it lands.
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
    let shot = wrecking::launch(&ball, &structure.aim, "ball");
    // Each chunk's storey (its group), in the scene's order.
    let storey: Vec<String> = scene.bodies[0].chunks.iter().map(|c| c.groups.get(1).cloned().unwrap_or_else(|| "footing".into())).collect();
    let centres: Vec<Vec3> = scene.bodies[0].chunks.iter().map(|c| Vec3::from_array(c.center)).collect();
    let start: Vec<f64> = centres.iter().map(|c| c.z).collect();
    let spans = stress_ref::viaduct::span_probes(&scene.bodies[0].chunks, &scene.bodies[0].bonds);
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
                let at: Vec<Vec3> = w.chunk_centres(&gpu).iter().map(|(c, _)| Vec3::from_array(*c)).collect();
                let z: Vec<f64> = at.iter().map(|c| c.z).collect();
                last = note(f, &z);
                print_spans(&spans, &centres, &at);
                println!("frame {:4}: broken {:5}, bodies {:5}, {:6.2} ms/frame (last {EVERY})", f + 1, st.broken, st.bodies, mean(&ms[ms.len().saturating_sub(EVERY)..]));
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
                let at: Vec<Vec3> = o.bodies.values().next().expect("the structure").iter().zip(&centres).map(|(c, c0)| *c0 + Vec3::from_array(c.displacement)).collect();
                let z: Vec<f64> = at.iter().map(|c| c.z).collect();
                last = note(f, &z);
                print_spans(&spans, &centres, &at);
                println!("frame {:4}: broken {:5}, fragments {:5}, {:6.2} ms/frame (last {EVERY})", f + 1, o.values["broken_bonds"], o.values["fragments"], mean(&ms[ms.len().saturating_sub(EVERY)..]));
            }
        }
    }
    let mut sorted = ms[FIRE_AT..].to_vec();
    sorted.sort_by(f64::total_cmp);
    println!("after the shot: median {:.2} ms/frame, p95 {:.2}, worst {:.2}", sorted[sorted.len() / 2], sorted[sorted.len() * 95 / 100], sorted[sorted.len() - 1]);
    // Storeys in height order.
    let mut storeys: Vec<&String> = last.keys().collect();
    storeys.sort_by_key(|s| (s.rsplit('-').next().and_then(|n| n.parse::<usize>().ok()).unwrap_or(0), s.as_str()));
    println!("storey      dropped   first frame a tenth had dropped {DROP} m");
    for s in storeys {
        let (d, n) = last[s];
        println!("{s:12} {d:4}/{n:<4}  {}", first.get(s).map_or("-".into(), |f| format!("{}", f + 1)));
    }
}

/// Each span (`stress_ref::viaduct::span_probes`) that has moved: its ends' drops and its
/// intact pairs.
fn print_spans(spans: &[stress_ref::viaduct::SpanProbe], start: &[Vec3], at: &[Vec3]) {
    let (start, at): (Vec<[f64; 3]>, Vec<[f64; 3]>) = (start.iter().map(|v| v.to_array()).collect(), at.iter().map(|v| v.to_array()).collect());
    for s in spans {
        let st = s.state(&start, &at);
        if st.west_drop.abs().max(st.east_drop.abs()) < 0.05 {
            continue;
        }
        println!("  {}: west end down {:5.2} m, east end {:5.2} m, pairs intact {}/{}", s.name, st.west_drop, st.east_drop, st.intact, st.pairs);
    }
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len().max(1) as f64
}
