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

/// The seam alone: the split path's members in two halves (node update, rest) with nothing
/// between them steps exactly as in one, bit for bit, through the hit.
#[test]
fn continuous_seam_alone_steps_as_one_member_dispatch() {
    let fleet = sedans_head_on();
    for tests in [false, true] {
        let (Some(mut a), Some(mut b)) = (gpu(&fleet), gpu(&fleet)) else {
            eprintln!("no GPU: skipped");
            return;
        };
        for g in [&mut a, &mut b] {
            g.set_bit_exact(true);
            g.set_vehicle_contact(true);
            g.set_group_split(true);
            g.set_group_split_tests(tests);
        }
        a.set_continuous(true);
        for f in 0..6 {
            a.frame(34);
            b.frame(34);
            let (pa, pb) = (a.positions(), b.positions());
            for k in 0..2 {
                assert_eq!(rms(&pa[k], &pb[k]), 0.0, "tests split {tests}, sedan {k}, frame {f}");
            }
        }
        a.contact_status().unwrap();
    }
}
