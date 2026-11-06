//! Centred damping (`ExplicitBeamsSettings::centred_damping`,
//! `engine/docs/jbeam-semantics-spec.md` §39) on the GPU against the CPU reference,
//! bit for bit every step: every node's position and velocity. Original Rubble
//! fixtures (`engine_softbody::jbeam_fixture`); opt-in, a community vehicle that
//! does not rest without it (`RUBBLE_COMPARE_MODS`, the crash lab's `__mods/`).
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, JbeamWheels, StaticBox,
};
use engine_softbody::jbeam::load_jbeam;
use engine_softbody::jbeam_fixture::{latched_box_jbeam, NODE_COUPLERS, PART2_CAR, PART2_GROUPS};
use engine_softbody::truck_export::TyreDefaults;
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings {
        turbulence: false,
        drag: false,
        centred_damping: true,
        ..Default::default()
    }
}

fn spawn(text: &str, wheels: Option<JbeamWheels>, v: Vec3) -> ExplicitBeams {
    let d = load_jbeam(&[("t.jbeam", text)], None).unwrap();
    let wheels = wheels.unwrap_or(JbeamWheels::Soft(TyreDefaults::for_vehicle(&d)));
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_with(&d, wheels, Vec3::ZERO, v, settings()).unwrap();
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += 0.02 - low;
    }
    s.roll_tyres(v);
    s
}

/// Both engines one step at a time for `steps`; every node bit for bit after each.
/// Returns the GPU's broken count, or `None` without an adapter.
fn bit_for_bit(label: &str, mut cpu: ExplicitBeams, steps: u32) -> Option<usize> {
    let (_, device, queue) = default_device().ok()?;
    assert!(cpu.settings.centred_damping);
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&cpu));
    for step in 0..steps {
        cpu.step();
        gpu.frame(1);
        let (gp, gv) = (&gpu.positions()[0], &gpu.velocities()[0]);
        let cp = cpu.positions();
        for i in 0..cpu.nodes.len() {
            let same = cp[i].to_array().map(f32::to_bits) == gp[i].to_array().map(f32::to_bits)
                && cpu.nodes[i].vel.to_array().map(f32::to_bits)
                    == gv[i].to_array().map(f32::to_bits);
            assert!(
                same,
                "{label}: step {step} node {i}: CPU {:?} {:?} GPU {:?} {:?}",
                cp[i], cpu.nodes[i].vel, gp[i], gv[i]
            );
        }
    }
    gpu.wait();
    let gb: Vec<bool> = gpu.beams()[0].iter().map(|b| b.broken).collect();
    assert_eq!(
        cpu.beams.iter().map(|b| b.broken).collect::<Vec<_>>(),
        gb,
        "{label}: broken beams"
    );
    let broken = gb.iter().filter(|&&b| b).count();
    eprintln!("{label}: {steps} steps bit for bit, {broken} broken");
    Some(broken)
}

/// The pressure-wheel fixture car (LBEAMs, anisotropic sidewalls, the tread
/// contact) at rest and rolling.
#[test]
fn pressure_wheel_car_steps_bit_for_bit() {
    let at_rest = spawn(PART2_CAR, Some(JbeamWheels::Pressure), Vec3::ZERO);
    // The flag changes the motion (the correction is not zero).
    let (mut on, mut off) = (at_rest.clone(), at_rest.clone());
    off.settings.centred_damping = false;
    for _ in 0..200 {
        on.step();
        off.step();
    }
    assert_ne!(on.positions(), off.positions());
    if bit_for_bit("fixture car at rest", at_rest, 4000).is_none() {
        eprintln!("no GPU adapter: skipped");
        return;
    }
    let v = Vec3::new(0.0, 0.0, 10.0);
    bit_for_bit(
        "fixture car at 10 m/s",
        spawn(PART2_CAR, Some(JbeamWheels::Pressure), v),
        2000,
    );
}

/// The latched box (torsion bar, bounded beams with fast damping and a cutoff,
/// supports, a coupler latch, hinge break groups, self-collision) into a wall at
/// 20 m/s; break groups pulled apart; node couplers thrown off.
#[test]
fn breaking_elements_step_bit_for_bit() {
    let text = latched_box_jbeam();
    let mut boxed = spawn(&text, None, Vec3::new(0.0, 0.0, 20.0));
    let front = boxed
        .positions()
        .iter()
        .map(|p| p.z)
        .fold(f32::MIN, f32::max);
    boxed.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.0, front + 1.05),
        Vec3::new(4.0, 3.0, 1.0),
    ));
    let Some(broken) = bit_for_bit("latched box into a wall", boxed, 1500) else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    assert!(broken > 0, "the crash breaks something");

    for pull in ["a1", "b1"] {
        let mut s = spawn(PART2_GROUPS, None, Vec3::ZERO);
        let d = load_jbeam(&[("t.jbeam", PART2_GROUPS)], None).unwrap();
        let i = d.nodes.iter().position(|n| n.name == pull).unwrap();
        s.nodes[i].vel = Vec3::new(40.0, 0.0, 0.0);
        let b = bit_for_bit(&format!("groups pulled at {pull}"), s, 500).unwrap();
        assert!(b > 0);
    }

    let d = load_jbeam(&[("nc.jbeam", NODE_COUPLERS)], None).unwrap();
    let (mut cpu, _) = ExplicitBeams::from_desc_jbeam(
        &d,
        TyreDefaults::default(),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::ZERO,
        settings(),
    )
    .unwrap();
    for (i, n) in d.nodes.iter().enumerate() {
        if n.name.starts_with('l') {
            cpu.nodes[i].vel.y += 3.0;
        }
    }
    bit_for_bit("node couplers", cpu, 800);
}

/// Opt-in (`RUBBLE_COMPARE_MODS`): the full-size SUV (`fullsuv_tex`,
/// `mid-sport.pc`) on its hubs, which does not rest without centred damping (spec
/// §29.1): 2,000 steps bit for bit, and at rest at the end.
#[test]
fn community_suv_rests_bit_for_bit() {
    let Ok(mods) = std::env::var("RUBBLE_COMPARE_MODS") else {
        return;
    };
    let dir = std::path::Path::new(&mods).join("fullsuv_tex");
    let Ok(files) = engine_softbody::jbeam::read_vehicle_dir(&dir, "fullsuv") else {
        return;
    };
    let refs: Vec<(&str, &str)> = files
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let pc = std::fs::read_to_string(dir.join("fullsuv").join("mid-sport.pc")).unwrap();
    let mut d = load_jbeam(&refs, Some(&pc)).unwrap();
    for w in &mut d.pressure_wheels {
        w.flags.insert("hasTire".into(), false);
    }
    let (mut s, _) = ExplicitBeams::from_desc_jbeam_with(
        &d,
        JbeamWheels::Pressure,
        Vec3::ZERO,
        Vec3::ZERO,
        settings(),
    )
    .unwrap();
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += 0.02 - low;
    }
    let (_, device, queue) = default_device().unwrap();
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&s));
    for step in 0..2000 {
        s.step();
        gpu.frame(1);
        let (gp, gv) = (&gpu.positions()[0], &gpu.velocities()[0]);
        let cp = s.positions();
        for i in 0..s.nodes.len() {
            assert!(
                cp[i].to_array().map(f32::to_bits) == gp[i].to_array().map(f32::to_bits)
                    && s.nodes[i].vel.to_array().map(f32::to_bits)
                        == gv[i].to_array().map(f32::to_bits),
                "SUV: step {step} node {i}"
            );
        }
    }
    gpu.wait();
    let vmax = s.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max);
    eprintln!("SUV on its hubs: 2000 steps bit for bit, fastest node {vmax:.2} m/s");
    assert!(vmax < 1.0, "the SUV rests: {vmax} m/s");
}
