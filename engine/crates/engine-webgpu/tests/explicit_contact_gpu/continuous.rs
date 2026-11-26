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
        g.set_continuous(true);
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

/// The pass's pair list against the CPU's, every step through the hit: as many pairs
/// (point-triangle and edge-edge, between the cars and each car's own) as the CPU lists,
/// nothing overflowed. (The pass does not move anything yet: the states stay the same.)
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
    g.set_continuous(true);
    let mut w = ExplicitBeamsWorld::new(fleet);
    // (The CPU's listing alone: no response, as the GPU's pass so far.)
    let mut d = engine_softbody::explicit_contact::Dat::new(0.08, 0.85);
    (d.iters, d.ccd_rounds) = (0, 0);
    w.continuous = Some(d);
    for f in 0..60 {
        g.frame(1);
        w.frame(1);
        let (pairs, c) = g.continuous_counts().expect("continuous tables");
        let cpu = w.continuous.as_ref().unwrap().last;
        let total: u64 = pairs.iter().map(|&p| u64::from(p)).sum();
        assert_eq!(c[0], 0, "step {f}: the pair list or a grid overflowed");
        assert_eq!(total, cpu.pairs_vt + cpu.pairs_ee, "step {f}: GPU pairs {pairs:?}, CPU vt {} ee {}", cpu.pairs_vt, cpu.pairs_ee);
    }
}
