//! The Topple tower (`stress_ref::topple`) brought down by its charges, natively, as the
//! lab's stress modes run it (`scenes::topple`: the GPU-resident threshold world for
//! `threshold`, the resident GpuWorld for `blast_parity`): the check that it leans as one
//! mass, falls across the street and breaks up on landing.
//!
//!   cargo run --release -p stress-rubble --features gpu --example topple -- [FRAMES] [DETONATE_AT]
//!
//! FRAMES at 60 Hz (default 480), the charges set off before frame DETONATE_AT (default
//! 30), each at its delay. `STRESS_PROFILE=blast_parity` (default `threshold`).
//!
//! Every `TOPPLE_EVERY` frames (15): bonds broken, bodies, the tower's lean (degrees toward the
//! street, of the line from the second storey's centre to the top storey's), the top
//! storey's height and how far it has moved toward the street (-y), and the share of the
//! storeys' chunks on the largest body. At the end, per storey: the bodies its chunks are
//! on and the largest one's share (a storey-sized slab is one body holding most of it).
use std::collections::BTreeMap;

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;

/// Frames between reports (`TOPPLE_EVERY`, default 15).
fn every() -> usize {
    std::env::var("TOPPLE_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(15)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let frames: usize = args.next().map_or(480, |f| f.parse().expect("FRAMES"));
    let at: u64 = args.next().map_or(30, |f| f.parse().expect("DETONATE_AT"));
    let p = std::env::var("STRESS_PROFILE").ok().map(|v| Profile::parse(&v).unwrap_or_else(|| panic!("STRESS_PROFILE={v}: no such profile"))).unwrap_or(Profile::Threshold);
    let (scene, mut demolition) = stress_rubble::scenes::topple(p, frames as f64 / 60.0);
    let chunks = &scene.bodies[0].chunks;
    let storey: Vec<Option<usize>> = chunks.iter().map(|c| c.groups.get(1).and_then(|g| g.strip_prefix("storey-")).and_then(|n| n.parse().ok())).collect();
    let top = storey.iter().flatten().copied().max().unwrap_or(1);
    let debris: Vec<usize> = demolition.charges.iter().flat_map(|c| c.debris.clone()).collect();
    let start: Vec<Vec3> = chunks.iter().map(|c| Vec3::from_array(c.center)).collect();
    println!("topple_tower: {} chunks, {} bonds, {} storeys, charges {:?}, {}, {frames} frames, set off before frame {at}", chunks.len(), scene.bodies[0].bonds.len(), top, demolition.names(), p.name());
    let gpu = Gpu::new().expect("GPU");
    // (frame, positions, body of each chunk, broken, bodies)
    let report = |f: usize, pos: &[Vec3], body: &[u64], broken: f64, bodies: f64| {
        let centre = |s: usize| -> Vec3 {
            let ks: Vec<usize> = (0..pos.len()).filter(|&k| storey[k] == Some(s) && !debris.contains(&k)).collect();
            ks.iter().fold(Vec3::ZERO, |a, &k| a + pos[k]) * (1.0 / ks.len() as f64)
        };
        let (lo, hi) = (centre(2), centre(top));
        let axis = hi - lo;
        let lean = (-axis.y).atan2(axis.z).to_degrees();
        let tower: Vec<usize> = (0..pos.len()).filter(|&k| storey[k].is_some() && !debris.contains(&k)).collect();
        let mut on: BTreeMap<u64, usize> = BTreeMap::new();
        for &k in &tower {
            *on.entry(body[k]).or_default() += 1;
        }
        let largest = on.values().copied().max().unwrap_or(0) as f64 / tower.len() as f64;
        let top0 = (0..start.len()).filter(|&k| storey[k] == Some(top)).fold(Vec3::ZERO, |a, k| a + start[k]) * (1.0 / tower.iter().filter(|&&k| storey[k] == Some(top)).count() as f64);
        let thrown = debris.iter().map(|&k| start[k].y - pos[k].y).sum::<f64>() / debris.len().max(1) as f64;
        println!("frame {:4} ({:5.2} s): broken {broken:5}, bodies {bodies:4}, lean {lean:6.1} deg, top storey at z {:5.1} m, moved {:5.1} m toward the street, largest body {:.0}% of the storeys, charged columns {thrown:4.1} m out", f + 1, (f + 1) as f64 / 60.0, hi.z, -(hi.y - top0.y), 100.0 * largest);
    };
    let storeys_at_end = |pos: &[Vec3], body: &[u64]| {
        let _ = pos;
        println!("storey  bodies  largest share");
        for s in 1..=top {
            let ks: Vec<usize> = (0..body.len()).filter(|&k| storey[k] == Some(s) && !debris.contains(&k)).collect();
            let mut on: BTreeMap<u64, usize> = BTreeMap::new();
            for &k in &ks {
                *on.entry(body[k]).or_default() += 1;
            }
            println!("{s:6}  {:6}  {:3.0}%", on.len(), 100.0 * on.values().copied().max().unwrap_or(0) as f64 / ks.len() as f64);
        }
        // The pieces of 10 chunks or more: their chunks storey by storey.
        let mut pieces: BTreeMap<u64, BTreeMap<usize, usize>> = BTreeMap::new();
        for k in (0..body.len()).filter(|&k| storey[k].is_some() && !debris.contains(&k)) {
            *pieces.entry(body[k]).or_default().entry(storey[k].unwrap()).or_default() += 1;
        }
        let mut big: Vec<(usize, BTreeMap<usize, usize>)> = pieces.into_values().map(|m| (m.values().sum(), m)).filter(|(n, _)| *n >= 10).collect();
        big.sort_by(|a, b| b.0.cmp(&a.0));
        for (n, m) in big {
            println!("piece of {n:3} chunks: {}", m.iter().map(|(s, c)| format!("s{s}:{c}")).collect::<Vec<_>>().join(" "));
        }
    };
    if p == Profile::Threshold {
        let ball = stress_ref::wrecking::BALLS[0];
        let mut w = stress_rubble::threshold::ThresholdWorld::new(&gpu, &scene, ball.radius, ball.mass).expect("threshold world");
        let dt = w.frame_dt;
        for f in 0..frames {
            if w.frame == at {
                demolition.arm("", w.frame, dt);
            }
            for c in demolition.due(w.frame) {
                let n = w.detonate(&gpu, &c).expect("detonate");
                println!("before frame {}: {} cut, {n} bonds", w.frame + 1, c.name);
            }
            w.step_frame(&gpu).expect("frame");
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            let st = w.poll(&gpu);
            if st.error != 0 || st.engine_overflow != 0 {
                panic!("frame {f}: fracture errors {:#x}, engine overflow {:#x}", st.error, st.engine_overflow);
            }
            if (f + 1) % every() == 0 || f + 1 == frames {
                let pos: Vec<Vec3> = w.chunk_centres(&gpu).iter().map(|(c, _)| Vec3::from_array(*c)).collect();
                let body: Vec<u64> = w.fracture.chunk_bodies(&gpu).into_iter().map(u64::from).collect();
                report(f, &pos, &body, st.broken as f64, st.bodies as f64);
                if f + 1 == frames {
                    storeys_at_end(&pos, &body);
                }
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
        let dt = scene.sim.frame_dt;
        let mut was_broken = vec![false; scene.bodies[0].bonds.len()];
        for f in 0..frames {
            if w.frame == at {
                demolition.arm("", w.frame, dt);
            }
            for c in demolition.due(w.frame) {
                let bonds = c.bond_indices(&scene.bodies[0].bonds).expect("charge bonds");
                let n = w.cut_bonds(&gpu, 0, &bonds, &c.debris, c.kick).expect("detonate");
                println!("before frame {}: {} cut, {n} bonds", w.frame + 1, c.name);
            }
            w.step_frame(&gpu).expect("frame");
            if (f + 1) % every() == 0 || f + 1 == frames {
                w.sync(&gpu);
                let o = w.observation();
                let pos: Vec<Vec3> = o.bodies.values().next().expect("the structure").iter().zip(&start).map(|(c, s)| *s + Vec3::from_array(c.displacement)).collect();
                let m = &w.solver.mirror;
                let body: Vec<u64> = (0..pos.len()).map(|k| m.clusters[m.chunks[0][k].cluster].id).collect();
                report(f, &pos, &body, o.values["broken_bonds"], o.values["fragments"]);
                if std::env::var("TOPPLE_TRACE").is_ok() {
                    // The bonds broken since the last report, by storey and the kinds of
                    // chunk they joined.
                    let bonds = &scene.bodies[0].bonds;
                    let kind = |c: usize| chunks[c].groups.first().cloned().unwrap_or_default();
                    let mut by: BTreeMap<String, usize> = BTreeMap::new();
                    for (i, b) in w.solver.mirror.bonds[0].iter().enumerate() {
                        let broken = !b.joint.connected(b.rebar.is_some());
                        if broken && !was_broken[i] {
                            let (a, c) = (bonds[i].a, bonds[i].b);
                            *by.entry(format!("s{} {}-{}", storey[a].or(storey[c]).unwrap_or(0), kind(a), kind(c))).or_default() += 1;
                        }
                        was_broken[i] = broken;
                    }
                    if !by.is_empty() {
                        println!("  {by:?}");
                    }
                }
                if f + 1 == frames {
                    storeys_at_end(&pos, &body);
                }
            }
        }
    }
}
