//! The free-flight scenes of `engine-softbody/tests/free_flight.rs` (the generated
//! sedan, the vehicle lab's `car=gen`, far above any ground, with the page's
//! settings: per-triangle aero and centred damping) on the GPU against the CPU
//! reference, bit for bit every step: every node's position and velocity, and the
//! broken flags at the end. The CPU test holds the physics (momentum, angular
//! momentum, a detached part's ballistic flight, torn parts leaving on their
//! tangent); this one holds that the kernels step the same, so the invariants hold
//! on the GPU too. Per-vehicle frame and the wide passes.
use engine_softbody::authoring::gen2::{rig as lattice, vg_input};
use engine_softbody::authoring::rig;
use engine_softbody::desc::SoftVehicleDesc;
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, ExplicitKind};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::{DVec3, Vec3};
use std::sync::OnceLock;

/// The sedan as the rigger exports it (JBeam, in memory).
fn sedan() -> &'static SoftVehicleDesc {
    static D: OnceLock<SoftVehicleDesc> = OnceLock::new();
    D.get_or_init(|| {
        let inp = rig::DesignInput::find("sedan", None, None).expect("the sedan's design");
        let r = lattice::rig_with(&inp);
        let files = vg_input::physics_files(&inp, &r);
        let texts: Vec<(String, String)> = files
            .iter()
            .map(|(p, b)| (p.clone(), String::from_utf8_lossy(b).into_owned()))
            .collect();
        let refs: Vec<(&str, &str)> = texts
            .iter()
            .filter(|(p, _)| p.ends_with(".jbeam"))
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let pc = texts
            .iter()
            .find(|(p, _)| p.ends_with(".pc"))
            .map(|(_, b)| b.as_str());
        engine_softbody::jbeam::load_jbeam(&refs, pc).expect("the sedan's jbeam")
    })
}

/// The sedan at the origin, the ground 10 km below, with the page's settings.
fn spawn() -> ExplicitBeams {
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        ..Default::default()
    };
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_raycast(sedan(), Vec3::ZERO, Vec3::ZERO, settings)
            .expect("spawn");
    assert!(s.settings.centred_damping && s.settings.centred_objective);
    s.ground = -10_000.0;
    s
}

fn centre(s: &ExplicitBeams, which: &dyn Fn(usize) -> bool) -> DVec3 {
    let (mut m, mut mx) = (0.0, DVec3::ZERO);
    for (_, n) in s.nodes.iter().enumerate().filter(|(i, _)| which(*i)) {
        m += n.mass as f64;
        mx += n.mass as f64 * n.pos.as_dvec3();
    }
    mx / m
}

/// Adds a rigid spin `w` (rad/s) about the centre of mass of `which`, and `v`.
fn spin(s: &mut ExplicitBeams, which: &dyn Fn(usize) -> bool, w: DVec3, v: DVec3) {
    let c = centre(s, which);
    for (i, n) in s.nodes.iter_mut().enumerate() {
        if which(i) {
            n.vel += (v + w.cross(n.pos.as_dvec3() - c)).as_vec3();
        }
    }
}

fn long_axis(s: &ExplicitBeams) -> DVec3 {
    let (lo, hi) = s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
        (a.min(n.pos), b.max(n.pos))
    });
    let d = hi - lo;
    if d.x > d.z {
        DVec3::X
    } else {
        DVec3::Z
    }
}

/// The CPU and the GPU one step at a time for `steps`, the GPU on its per-vehicle
/// frame and on the wide passes: every node bit for bit after each step, the
/// broken flags at the end. Returns the broken count, or `None` without an adapter.
fn bit_for_bit(label: &str, mut cpu: ExplicitBeams, steps: u32) -> Option<usize> {
    let (_, device, queue) = default_device().ok()?;
    let mut frame = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&cpu));
    let mut wide = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&cpu));
    wide.set_vehicle_contact(false);
    let wide_ok = wide.set_wide(true);
    if !wide_ok {
        eprintln!("{label}: no wide passes");
    }
    for step in 0..steps {
        cpu.step();
        frame.frame(1);
        if wide_ok {
            wide.frame(1);
        }
        let cp = cpu.positions();
        let paths = [("frame", &frame), ("wide", &wide)];
        for (path, gpu) in &paths[..1 + wide_ok as usize] {
            let (gp, gv) = (&gpu.positions()[0], &gpu.velocities()[0]);
            for i in 0..cpu.nodes.len() {
                let same = cp[i].to_array().map(f32::to_bits) == gp[i].to_array().map(f32::to_bits)
                    && cpu.nodes[i].vel.to_array().map(f32::to_bits)
                        == gv[i].to_array().map(f32::to_bits);
                assert!(
                    same,
                    "{label} ({path}): step {step} node {i}: CPU {:?} {:?} GPU {:?} {:?}",
                    cp[i], cpu.nodes[i].vel, gp[i], gv[i]
                );
            }
        }
    }
    let cb: Vec<bool> = cpu.beams.iter().map(|b| b.broken).collect();
    let paths = [("frame", &frame), ("wide", &wide)];
    for (path, gpu) in &paths[..1 + wide_ok as usize] {
        gpu.wait();
        let gb: Vec<bool> = gpu.beams()[0].iter().map(|b| b.broken).collect();
        assert_eq!(cb, gb, "{label} ({path}): broken beams");
    }
    let broken = cb.iter().filter(|&&b| b).count();
    eprintln!(
        "{label}: {steps} steps bit for bit ({}), {broken} records broken",
        if wide_ok { "frame and wide" } else { "frame" }
    );
    Some(broken)
}

/// The nodes of part `name` in no other part.
fn own_nodes(name: &str) -> Vec<bool> {
    let d = sedan();
    let mut own = vec![false; d.nodes.len()];
    for &i in &d.part(name).expect("the part").nodes {
        own[i as usize] = true;
    }
    for p in d.parts.iter().filter(|p| p.name != name) {
        for &i in &p.nodes {
            own[i as usize] = false;
        }
    }
    own
}

/// The intact car tumbling at 2 rev/s; the front left door cut free and thrown
/// off a yawing body; the car spun at 10 rev/s about its long axis until the hub
/// caps and a door tear off.
#[test]
fn free_flight_steps_bit_for_bit() {
    if default_device().is_err() {
        eprintln!("no GPU adapter: skipped");
        return;
    }
    let tau = std::f64::consts::TAU;

    let mut s = spawn();
    spin(
        &mut s,
        &|_| true,
        DVec3::new(0.3, 1.0, 0.5).normalize() * 2.0 * tau,
        DVec3::ZERO,
    );
    bit_for_bit("intact car tumbling at 2 rev/s", s, 1000);

    let mut s = spawn();
    let door = own_nodes("sedan_door_FL");
    let mut cut = 0;
    for j in 0..s.beams.len() {
        let b = s.beams[j];
        if matches!(
            b.kind,
            ExplicitKind::JbeamAeroTri
                | ExplicitKind::JbeamAeroTri2
                | ExplicitKind::JbeamPressureTri
                | ExplicitKind::JbeamPressureTri2
        ) {
            continue;
        }
        if door[b.a as usize] != door[b.b as usize] {
            s.beams[j].broken = true;
            cut += 1;
        }
    }
    assert!(cut > 0);
    let out = (centre(&s, &|i| door[i]) - centre(&s, &|i| !door[i]))
        .with_y(0.0)
        .normalize();
    for (i, n) in s.nodes.iter_mut().enumerate() {
        if door[i] {
            n.pos += (out * 0.25).as_vec3();
        }
    }
    spin(
        &mut s,
        &|i| door[i],
        DVec3::new(0.4, 1.0, 0.2).normalize() * 3.0,
        out * 3.0,
    );
    spin(&mut s, &|i| !door[i], DVec3::Y, -out * 3.0);
    bit_for_bit("door cut free and thrown", s, 1000);

    let mut s = spawn();
    let axis = long_axis(&s);
    spin(&mut s, &|_| true, axis * 10.0 * tau, DVec3::ZERO);
    let broken = bit_for_bit("spun apart at 10 rev/s", s, 1000).unwrap();
    assert!(broken > 0, "the spin tears parts off");
}
