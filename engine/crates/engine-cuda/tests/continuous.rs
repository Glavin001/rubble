//! Continuous contact on CUDA (`docs/continuous-contact-design.md`): the generated sedans
//! head on, a 34-step frame's wall time with and without it, and the CUDA run beside the
//! CPU reference's (CUDA here is not bit exact: its parity is by tolerance, as
//! `explicit_parity.rs`). GPU tests run when `ENGINE_CUDA_DEVICE=1`; without it they skip.

use engine_cuda::CudaExplicitBeams;
use engine_softbody::explicit_beams::ExplicitBeams;
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use glam::Vec3;

fn gpu() -> bool {
    if std::env::var("ENGINE_CUDA_DEVICE").as_deref() != Ok("1") {
        eprintln!("skipped: set ENGINE_CUDA_DEVICE=1 on an NVIDIA host (built with nvcc)");
        return false;
    }
    assert!(CudaExplicitBeams::available(), "ENGINE_CUDA_DEVICE=1 but no NVIDIA driver/device loaded");
    true
}

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

/// A 34-step frame's wall time (with the positions' readback) through the crash, continuous
/// contact off and on, and the cars' distance from the CPU reference's at the end.
#[test]
fn cuda_continuous_head_on_cost() {
    if !gpu() {
        return;
    }
    let fleet = sedans_head_on();
    for on in [false, true] {
        let mut g = CudaExplicitBeams::new(&fleet).unwrap();
        g.set_vehicle_contact(true);
        g.set_group_split(true);
        g.set_continuous(on).expect("the sedan keeps its clearance");
        let mut w = ExplicitBeamsWorld::new(fleet.clone());
        if on {
            w.continuous = Some(engine_softbody::explicit_contact::Dat::new(0.08, 0.85));
        }
        let mut ms = Vec::new();
        let mut pos = Vec::new();
        for _ in 0..8 {
            let t = std::time::Instant::now();
            g.frame(34).unwrap();
            pos = g.positions().unwrap();
            ms.push((t.elapsed().as_secs_f64() * 1e4).round() / 10.0);
            w.frame(34);
        }
        let d: Vec<f32> = (0..2)
            .map(|k| {
                let c = w.vehicles[k].positions();
                (pos[k].iter().zip(&c).map(|(a, b)| a.distance_squared(*b)).sum::<f32>() / c.len() as f32).sqrt()
            })
            .collect();
        eprintln!("CUDA continuous {on}: ms a frame {ms:?}; rms from the CPU after 0.13 s {d:?} m");
        assert!(pos.iter().flatten().all(|p| p.is_finite()));
    }
}
