//! A scene with bonded vehicles (`stress_ref::vehicles`) natively, as the lab's stress
//! modes run it: every launch released at frame 10 (`stress_rubble::launch::Launcher`,
//! `ThresholdWorld::launch_island` / `GpuWorld::launch_island`).
//!
//!   cargo run --release -p stress-rubble --features gpu --example vehicles -- [SCENE] [FRAMES]
//!
//! SCENE a `stress_ref::vehicles::SCENES` key (default `vh_car_barrier`); FRAMES at 60 Hz
//! (default 180). `STRESS_PROFILE=blast_parity` for the other profile (default threshold).
//!
//! Every `EVERY` frames: bonds broken (the structure's, each vehicle's), bodies, and each
//! vehicle's centre and speed (its chunks' mean). At the end, per vehicle and part kind,
//! the bonds intact of those it had, and the share of the vehicle's chunks on its largest
//! body (a cabin that stays recognisable keeps a large share).
use std::collections::BTreeMap;

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::capability::Profile;
use stress_ref::math::Vec3;

const EVERY: u64 = 10;
const AT: u64 = 10;

fn main() {
    let mut args = std::env::args().skip(1);
    let key = args.next().unwrap_or_else(|| "vh_car_barrier".into());
    let frames: u64 = args.next().map_or(180, |f| f.parse().expect("FRAMES"));
    let p = std::env::var("STRESS_PROFILE").ok().map(|v| Profile::parse(&v).unwrap_or_else(|| panic!("STRESS_PROFILE={v}: no such profile"))).unwrap_or(Profile::Threshold);
    let vs = stress_ref::vehicles::scene(&key).unwrap_or_else(|| panic!("no scene {key}"));
    let pack = vs.to_pack();
    let scene = stress_rubble::scenes::pack_scene(&pack, &key, p, frames as f64 / 60.0);
    let mut launcher = stress_rubble::launch::Launcher::of_pack(&pack).expect("launches");
    let bonds = scene.bodies[0].bonds.clone();
    let chunks = scene.bodies[0].chunks.clone();
    let owner: Vec<Option<usize>> = (0..chunks.len()).map(|k| vs.vehicles.iter().position(|(_, ids)| ids.contains(&k))).collect();
    let gpu = Gpu::new().expect("GPU");
    println!("{key} under {}: {} chunks, {} bonds; vehicles {:?}", p.name(), chunks.len(), bonds.len(), vs.vehicles.iter().map(|(n, ids)| (n.clone(), ids.len())).collect::<Vec<_>>());
    // Per frame: (positions, intact bonds, body of each chunk).
    let report = |f: u64, pos: &[Vec3], intact: &[bool], body: &[u32], start: &[Vec3], prev: &[Vec3]| {
        let broken_of = |who: Option<usize>| (0..bonds.len()).filter(|&b| !intact[b] && owner[bonds[b].a] == who && owner[bonds[b].b] == who).count();
        let mut line = format!("frame {f:4}: structure broken {:4}", broken_of(None));
        for (vi, (name, ids)) in vs.vehicles.iter().enumerate() {
            let n = ids.len() as f64;
            let c = ids.iter().fold(Vec3::ZERO, |a, &k| a + pos[k]) * (1.0 / n);
            let c0 = ids.iter().fold(Vec3::ZERO, |a, &k| a + prev[k]) * (1.0 / n);
            let s = ids.iter().fold(Vec3::ZERO, |a, &k| a + start[k]) * (1.0 / n);
            let mut on: BTreeMap<u32, usize> = BTreeMap::new();
            for &k in ids {
                *on.entry(body[k]).or_default() += 1;
            }
            let largest = *on.values().max().unwrap_or(&0) as f64 / n;
            line += &format!(" | {name}: broken {:3}, bodies {:2}, largest {:.0}%, moved {:.2?}, speed {:.1} m/s", broken_of(Some(vi)), on.len(), 100.0 * largest, (c - s).to_array(), (c - c0).norm() * 60.0 / EVERY as f64);
        }
        println!("{line}");
    };
    let kinds = |intact: &[bool], body: &[u32]| {
        for (vi, (name, ids)) in vs.vehicles.iter().enumerate() {
            let mut by: BTreeMap<String, (usize, usize)> = BTreeMap::new();
            for (b, bd) in bonds.iter().enumerate() {
                if owner[bd.a] == Some(vi) && owner[bd.b] == Some(vi) {
                    let ka = chunks[bd.a].groups.first().cloned().unwrap_or_default();
                    let kb = chunks[bd.b].groups.first().cloned().unwrap_or_default();
                    let key = if ka <= kb { format!("{ka}-{kb}") } else { format!("{kb}-{ka}") };
                    let e = by.entry(key).or_default();
                    e.1 += 1;
                    e.0 += intact[b] as usize;
                }
            }
            let mut on: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
            for &k in ids {
                on.entry(body[k]).or_default().push(k);
            }
            let main = on.values().max_by_key(|v| v.len()).cloned().unwrap_or_default();
            let mut kept: BTreeMap<String, (usize, usize)> = BTreeMap::new();
            for &k in ids {
                let e = kept.entry(chunks[k].groups.first().cloned().unwrap_or_default()).or_default();
                e.1 += 1;
                e.0 += main.contains(&k) as usize;
            }
            println!("{name}: bonds intact by joint {by:?}");
            println!("{name}: parts on its largest body {kept:?}");
        }
    };
    if p == Profile::Threshold {
        let ball = stress_ref::wrecking::BALLS[0];
        let mut w = stress_rubble::threshold::ThresholdWorld::new(&gpu, &scene, ball.radius, ball.mass).expect("threshold world");
        let pos = |w: &stress_rubble::threshold::ThresholdWorld| -> Vec<Vec3> { w.chunk_centres(&gpu).iter().map(|(c, _)| Vec3::from_array(*c)).collect() };
        let start = pos(&w);
        let mut prev = start.clone();
        let t = std::time::Instant::now();
        for _ in 0..frames {
            if w.frame == AT {
                launcher.arm("", w.frame, w.frame_dt);
            }
            for l in launcher.due(w.frame) {
                w.launch_island(&gpu, 0, &l).expect("launch");
            }
            w.step_frame(&gpu).expect("frame");
            if w.frame % EVERY == 0 {
                let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                let st = w.poll(&gpu);
                assert_eq!((st.error, st.engine_overflow), (0, 0));
                let now = pos(&w);
                report(w.frame, &now, &w.scene_bonds_intact(&gpu, 0), &w.fracture.chunk_bodies(&gpu), &start, &prev);
                prev = now;
            }
        }
        let intact = w.scene_bonds_intact(&gpu, 0);
        kinds(&intact, &w.fracture.chunk_bodies(&gpu));
        println!("{:.1} s", t.elapsed().as_secs_f64());
    } else {
        let mut e = stress_rubble::RubbleEngine::with_device_capacity(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone(), stress_rubble::capacity_for(&scene)).expect("rubble");
        let (iterations, substeps) = p.engine_policy();
        e.set_policy(iterations, substeps);
        let mut w = GpuWorld::with_resident_engine(&gpu, &scene, e).expect("world");
        w.redo_fractured_frames = false;
        w.set_impact_islands(true);
        w.apply_profile(p);
        let state = |w: &mut GpuWorld| -> (Vec<Vec3>, Vec<bool>, Vec<u32>) {
            w.sync(&gpu);
            let o = w.observation();
            let pos: Vec<Vec3> = o.bodies.values().next().unwrap().iter().zip(&chunks).map(|(c, ch)| Vec3::from_array(ch.center) + Vec3::from_array(c.displacement)).collect();
            let m = &w.solver.mirror;
            let intact = (0..bonds.len()).map(|b| m.bonds[0][b].connected()).collect();
            let body = (0..chunks.len()).map(|k| m.chunks[0][k].cluster as u32).collect();
            (pos, intact, body)
        };
        let (start, _, _) = state(&mut w);
        let mut prev = start.clone();
        let t = std::time::Instant::now();
        for _ in 0..frames {
            if w.frame == AT {
                launcher.arm("", w.frame, scene.sim.frame_dt);
            }
            for l in launcher.due(w.frame) {
                w.launch_island(&gpu, 0, &l.chunks, l.velocity, l.angular_velocity).expect("launch");
            }
            w.step_frame(&gpu).expect("frame");
            if w.frame % EVERY == 0 {
                let (now, intact, body) = state(&mut w);
                report(w.frame, &now, &intact, &body, &start, &prev);
                prev = now;
            }
        }
        let (_, intact, body) = state(&mut w);
        kinds(&intact, &body);
        println!("{:.1} s", t.elapsed().as_secs_f64());
    }
}
