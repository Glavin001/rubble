//! Continuous contact on the GPU (`docs/continuous-contact-design.md`, "The GPU port").

use super::*;

/// The generated sedans head on at 60 km/h each, 25 cm apart (surface and edge contact).
fn sedans_head_on() -> Vec<ExplicitBeams> {
    use engine_softbody_authoring::authoring::rig;
    let built = rig::rig_vehicle("sedan", None, None).expect("rig the generated sedan");
    let files = rig::mod_files(&built);
    let txt: Vec<(String, String)> = files.iter().filter(|(p, _)| p.ends_with(".jbeam")).map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned())).collect();
    let pc = files.iter().find(|(p, _)| p.ends_with(".pc")).map(|(_, b)| String::from_utf8_lossy(b).into_owned());
    let refs: Vec<(&str, &str)> = txt.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let d = engine_softbody::jbeam::load_jbeam(&refs, pc.as_deref()).expect("load");
    let settings = engine_softbody_authoring::vehicle_compare::settings();
    let mut turned = d.clone();
    turned.nodes.iter_mut().for_each(|n| n.position = [-n.position[0], n.position[1], -n.position[2]]);
    let ext = |d: &engine_softbody::desc::SoftVehicleDesc, k: usize| d.nodes.iter().map(|n| n.position[k]).fold((f32::MAX, f32::MIN), |(a, b), x| (a.min(x), b.max(x)));
    let make = |d: &engine_softbody::desc::SoftVehicleDesc, z: f32, v: f32| {
        let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(d, Vec3::new(0.0, 0.02 - ext(d, 1).0, z), Vec3::Z * v, settings).expect("spawn");
        s.surface_contact = true;
        s.edge_contact = true;
        s.finish();
        s
    };
    let speed = 60.0 / 3.6;
    vec![make(&d, -0.125 - ext(&d, 2).1, speed), make(&turned, 0.125 - ext(&turned, 2).0, -speed)]
}

/// Continuous contact through the generated sedans' head-on, GPU == CPU bit for bit every
/// step (the pass's listing, speculative contact, CCD impulses and impact zones), with the
/// split path's tests split off and on.
#[test]
fn continuous_head_on_steps_like_the_cpu() {
    let fleet = sedans_head_on();
    for tests in [false, true] {
        let Some(mut g) = gpu(&fleet) else {
            eprintln!("no GPU: skipped");
            return;
        };
        g.set_bit_exact(true);
        g.set_vehicle_contact(true);
        g.set_group_split(true);
        g.set_group_split_tests(tests);
        g.set_continuous(true).expect("the sedan keeps its clearance");
        let mut w = ExplicitBeamsWorld::new(fleet.clone());
        w.continuous = Some(engine_softbody::explicit_contact::Dat::new(0.08, 0.85));
        for f in 0..80 {
            g.frame(1);
            w.frame(1);
            let pos = g.positions();
            let (_, c) = g.continuous_counts().expect("continuous tables");
            let cpu = w.continuous.as_ref().unwrap().total.ccd;
            for k in 0..2 {
                let d = rms(&pos[k], &w.vehicles[k].positions());
                assert_eq!(d, 0.0, "tests split {tests}, step {f}, sedan {k}: GPU-CPU {d} m; GPU counters {c:?}, CPU impacts {} zones {}", cpu.hits, cpu.zones);
            }
        }
        let cpu = w.continuous.as_ref().unwrap().total;
        let (_, c) = g.continuous_counts().unwrap();
        eprintln!("tests split {tests}: GPU counters {c:?}; CPU impacts {} zones {} unresolved {}", cpu.ccd.hits, cpu.ccd.zones, cpu.ccd.unresolved);
        assert!(cpu.ccd.hits > 0, "the cars met");
    }
}

/// Velocity contact (`set_continuous_velocity`; the CPU's `Dat::velocity` with its warm starts
/// and relisting off, which the GPU's first port has not) through the generated sedans' head-on,
/// GPU == CPU bit for bit every step: the rows, the APGD solve, the rounds and the impact zones,
/// through the first torn triangles (step 26: a break group's beams torn at a frame's end as
/// within a frame, `explicit_beams_numerics.slang`'s gDropsAtFrameEnd).
#[test]
fn continuous_velocity_head_on_steps_like_the_cpu() {
    let fleet = sedans_head_on();
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    g.set_bit_exact(true);
    g.set_vehicle_contact(true);
    g.set_group_split(true);
    g.set_continuous(true).expect("the sedan keeps its clearance");
    g.set_continuous_velocity(true);
    let mut w = ExplicitBeamsWorld::new(fleet.clone());
    let mut d = engine_softbody::explicit_contact::Dat::new(0.08, 0.85);
    (d.velocity, d.warm_starts, d.relist) = (true, false, false);
    w.continuous = Some(d);
    for f in 0..80 {
        g.frame(1);
        w.frame(1);
        let pos = g.positions();
        let (_, c) = g.continuous_counts().expect("continuous tables");
        for k in 0..2 {
            let d = rms(&pos[k], &w.vehicles[k].positions());
            assert_eq!(d, 0.0, "step {f}, sedan {k}: GPU-CPU {d} m; GPU counters {c:?}");
        }
        assert_eq!(c[0], 0, "step {f}: an overflow");
    }
    let cpu = w.continuous.as_ref().unwrap().total.ccd;
    let (_, c) = g.continuous_counts().unwrap();
    eprintln!("velocity: GPU counters {c:?}; CPU impacts {} zones {} unresolved {} rounds {} row iterations {}", cpu.hits, cpu.zones, cpu.unresolved, cpu.rounds, cpu.row_iters);
    assert!(cpu.hits > 0 && cpu.zones > 0, "the cars met");
    assert_eq!([c[1], c[2], c[3], c[6], c[7]].map(|x| x as u64), [cpu.hits, cpu.zones, cpu.unresolved, cpu.rounds, cpu.row_iters], "impacts, zones, unresolved, rounds, APGD iterations");
}

/// The pass's pair list against the CPU's, every step through the hit: as many pairs
/// (point-triangle and edge-edge, between the cars and each car's own) as the CPU lists,
/// nothing overflowed.
#[test]
fn continuous_pairs_list_like_the_cpu() {
    let fleet = sedans_head_on();
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    g.set_bit_exact(true);
    g.set_vehicle_contact(true);
    g.set_group_split(true);
    g.set_continuous(true).expect("the sedan keeps its clearance");
    let mut w = ExplicitBeamsWorld::new(fleet);
    // (Both with their response: the states stay equal, so the listings must too.)
    w.continuous = Some(engine_softbody::explicit_contact::Dat::new(0.08, 0.85));
    w.continuous.as_mut().unwrap().keys = Some(Vec::new());
    for f in 0..60 {
        g.frame(1);
        w.frame(1);
        let (pairs, c) = g.continuous_counts().expect("continuous tables");
        let cpu = w.continuous.as_ref().unwrap().last;
        let total: u64 = pairs.iter().map(|&p| u64::from(p)).sum();
        assert_eq!(c[0], 0, "step {f}: the pair list or a grid overflowed");
        if std::env::var("CC_DEBUG_DEAD").is_ok() {
            eprintln!("step {f}: CPU dead triangles {:?}, broken beams {:?}", w.vehicles.iter().map(|v| v.tri_dead.iter().filter(|&&d| d).count()).collect::<Vec<_>>(), w.vehicles.iter().map(|v| v.beams.iter().filter(|b| b.broken).count()).collect::<Vec<_>>());
        }
        if total != cpu.pairs_vt + cpu.pairs_ee {
            // (Which pairs: each side's keys as sorted global node sets.)
            let first: Vec<u32> = w.vehicles.iter().scan(0u32, |a, v| { let o = *a; *a += v.nodes.len() as u32; Some(o) }).collect();
            let norm = |mut k: [u32; 4]| { k.sort_unstable(); k };
            let gk: std::collections::BTreeSet<[u32; 4]> = g.continuous_pair_keys().into_iter().map(norm).collect();
            let ck: std::collections::BTreeSet<[u32; 4]> = w.continuous.as_ref().unwrap().keys.as_ref().unwrap().iter().map(|k| norm(k.map(|(v, i)| first[v] + i))).collect();
            eprintln!("step {f}: only on the GPU {:?}; only on the CPU {:?}", gk.difference(&ck).collect::<Vec<_>>(), ck.difference(&gk).collect::<Vec<_>>());
        }
        assert_eq!(total, cpu.pairs_vt + cpu.pairs_ee, "step {f}: GPU pairs {pairs:?}, CPU vt {} ee {}", cpu.pairs_vt, cpu.pairs_ee);
    }
}

/// The same at the lab's frames (34 steps), 12 frames (0.2 s) through the crash.
#[test]
fn continuous_head_on_lab_frames_like_the_cpu() {
    let fleet = sedans_head_on();
    let Some(mut g) = gpu(&fleet) else {
        eprintln!("no GPU: skipped");
        return;
    };
    g.set_bit_exact(true);
    g.set_vehicle_contact(true);
    g.set_group_split(true);
    g.set_continuous(true).expect("the sedan keeps its clearance");
    let mut w = ExplicitBeamsWorld::new(fleet.clone());
    w.continuous = Some(engine_softbody::explicit_contact::Dat::new(0.08, 0.85));
    for f in 0..12 {
        let t = std::time::Instant::now();
        g.frame(34);
        let pos = g.positions();
        let gpu_ms = t.elapsed().as_secs_f64() * 1e3;
        w.frame(34);
        for k in 0..2 {
            let d = rms(&pos[k], &w.vehicles[k].positions());
            assert_eq!(d, 0.0, "frame {f}, sedan {k}: GPU-CPU {d} m");
        }
        let (_, c) = g.continuous_counts().unwrap();
        eprintln!("frame {f}: {gpu_ms:.1} ms (with readback), counters {c:?}");
    }
}

/// Measurement (ignored): wall time a 34-step frame without continuous contact, with its
/// position response and with velocity contact.
#[test]
#[ignore]
fn continuous_cost() {
    let fleet = sedans_head_on();
    for (on, vel) in [(false, false), (true, false), (true, true)] {
        let Some(mut g) = gpu(&fleet) else { return };
        g.set_vehicle_contact(true);
        g.set_group_split(true);
        g.set_continuous(on).expect("the sedan keeps its clearance");
        g.set_continuous_velocity(vel);
        let mut ms = Vec::new();
        for _ in 0..8 {
            let t = std::time::Instant::now();
            g.frame(34);
            let _ = g.positions();
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            if on {
                let _ = g.continuous_counts();
            }
        }
        let c = g.continuous_counts().map(|c| c.1);
        eprintln!("continuous {on}, velocity {vel}: ms a frame {:?}; counters {c:?}", ms.iter().map(|x| (x * 10.0).round() / 10.0).collect::<Vec<_>>());
    }
}
