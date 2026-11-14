//! Centred damping (`ExplicitBeamsSettings::centred_damping`,
//! `engine/docs/jbeam-semantics-spec.md` §39) on the GPU against the CPU reference,
//! bit for bit every step: every node's position and velocity. Original Rubble
//! fixtures (`engine_softbody::jbeam_fixture`); opt-in, a community vehicle that
//! does not rest without it (`RUBBLE_COMPARE_MODS`, the crash lab's `__mods/`).
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, FrameClock, JbeamWheels, StaticBox,
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

/// The CPU and the GPU one step at a time for `steps`, the GPU both on its
/// per-vehicle frame and on the wide passes: every node bit for bit after each
/// step. Returns the broken count and the fastest node's speed, or `None` without
/// an adapter.
fn bit_for_bit(label: &str, mut cpu: ExplicitBeams, steps: u32) -> Option<(usize, f32)> {
    let (_, device, queue) = default_device().ok()?;
    assert!(cpu.settings.centred_damping);
    let mut frame = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&cpu));
    let mut wide = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&cpu));
    wide.set_vehicle_contact(false);
    // (A vehicle with commands, wings or shocks2 cannot take the wide passes.)
    let wide_ok = wide.set_wide(true);
    if !wide_ok {
        eprintln!("{label}: no wide passes (commands, wings or shocks2)");
    }
    for step in 0..steps {
        cpu.step();
        frame.frame(1);
        wide.frame(1);
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
    let vmax = cpu.nodes.iter().map(|n| n.vel.length()).fold(0.0, f32::max);
    eprintln!(
        "{label}: {steps} steps bit for bit ({}), {broken} broken, fastest node {vmax:.2} m/s",
        if wide_ok { "frame and wide" } else { "frame" }
    );
    Some((broken, vmax))
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
    let Some((broken, _)) = bit_for_bit("latched box into a wall", boxed, 1500) else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    assert!(broken > 0, "the crash breaks something");

    for pull in ["a1", "b1"] {
        let mut s = spawn(PART2_GROUPS, None, Vec3::ZERO);
        let d = load_jbeam(&[("t.jbeam", PART2_GROUPS)], None).unwrap();
        let i = d.nodes.iter().position(|n| n.name == pull).unwrap();
        s.nodes[i].vel = Vec3::new(40.0, 0.0, 0.0);
        let (b, _) = bit_for_bit(&format!("groups pulled at {pull}"), s, 500).unwrap();
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
    let (_, vmax) = bit_for_bit("SUV on its hubs", s, 2000).unwrap();
    assert!(vmax < 1.0, "the SUV rests: {vmax} m/s");
}

/// Opt-in timing (`RUBBLE_CENTRED_TIMING=<fleet size>`, at most 256): a fleet of
/// the pressure-wheel fixture car parked apart, 60 fps frames, interleaved A/B so
/// a shared GPU's load falls on every configuration alike: the wide passes without
/// and with centred damping, and the per-vehicle frame without and with it. Prints the median
/// frame time of each. (The community cars carry commands or shocks2, which the
/// wide passes do not run.)
#[test]
fn centred_damping_frame_cost() {
    let Ok(n) = std::env::var("RUBBLE_CENTRED_TIMING") else {
        return;
    };
    let n: usize = n.parse().unwrap();
    let fleet = |centred: bool| -> Vec<ExplicitBeams> {
        (0..n)
            .map(|k| {
                let mut s = spawn(PART2_CAR, Some(JbeamWheels::Pressure), Vec3::ZERO);
                s.settings.centred_damping = centred;
                let at = Vec3::new(4.0 * (k % 16) as f32, 0.0, 6.0 * (k / 16) as f32);
                for nd in &mut s.nodes {
                    nd.pos += at;
                }
                s
            })
            .collect()
    };
    let (_, device, queue) = default_device().unwrap();
    let (off, on) = (fleet(false), fleet(true));
    let mut runs = [
        (
            "wide, centred off",
            ExplicitBeamsGpu::new(&device, &queue, &off),
            true,
        ),
        (
            "wide, centred on",
            ExplicitBeamsGpu::new(&device, &queue, &on),
            true,
        ),
        (
            "frame, centred off",
            ExplicitBeamsGpu::new(&device, &queue, &off),
            false,
        ),
        (
            "frame, centred on",
            ExplicitBeamsGpu::new(&device, &queue, &on),
            false,
        ),
    ];
    for (label, g, wide) in &mut runs {
        g.set_vehicle_contact(false);
        assert_eq!(g.set_wide(*wide), *wide, "{label}");
    }
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); runs.len()];
    let mut clock = [FrameClock::default(); 4];
    for frame in 0..260 {
        for (k, (_, g, _)) in runs.iter_mut().enumerate() {
            let t = std::time::Instant::now();
            g.frame(clock[k].next());
            g.wait();
            if frame >= 20 {
                times[k].push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
    }
    for ((label, ..), t) in runs.iter().zip(&mut times) {
        t.sort_by(f64::total_cmp);
        eprintln!(
            "{n} fixture cars, {label}: median {:.2} ms per 60 fps frame (p10 {:.2}, p90 {:.2})",
            t[t.len() / 2],
            t[t.len() / 10],
            t[t.len() * 9 / 10]
        );
    }
}
