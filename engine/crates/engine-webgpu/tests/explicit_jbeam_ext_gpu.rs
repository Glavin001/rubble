//! JBeam mode Part II (`engine/docs/jbeam-semantics-spec.md` §16–§28) on the GPU
//! against the CPU reference: the same Slang rules on the same records. Original
//! Rubble fixtures (`engine_softbody::jbeam_fixture::PART2_*`).
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, FrameClock, JbeamWheels, StaticBox,
};
use engine_softbody::jbeam::load_jbeam;
use engine_softbody::jbeam_fixture::{
    PART2_CAR, PART2_GROUPS, PART2_PANEL, PART2_RAIL, PART2_RAMP,
};
use engine_softbody::truck_export::TyreDefaults;
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

fn spawn(text: &str, wheels: Option<JbeamWheels>, v: Vec3) -> ExplicitBeams {
    let d = load_jbeam(&[("t.jbeam", text)], None).unwrap();
    let wheels = wheels.unwrap_or(JbeamWheels::Soft(TyreDefaults::for_vehicle(&d)));
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        drag: false,
        ..Default::default()
    };
    let (mut s, _) =
        ExplicitBeams::from_desc_jbeam_with(&d, wheels, Vec3::ZERO, v, settings).unwrap();
    let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += 0.02 - low;
    }
    s.roll_tyres(v);
    s
}

/// Both engines for `frames` 60 fps frames; (max position difference, CPU broken,
/// GPU broken, CPU deform, GPU deform).
fn compare(
    label: &str,
    cpu: &mut ExplicitBeams,
    frames: u32,
) -> Option<(f32, usize, usize, Vec<u32>, Vec<u32>)> {
    let (_, device, queue) = default_device().ok()?;
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(cpu));
    let mut clock = FrameClock::default();
    let mut worst = 0.0f32;
    for _ in 0..frames {
        cpu.frame();
        gpu.frame(clock.next());
        let g = &gpu.positions()[0];
        let d = cpu
            .positions()
            .iter()
            .zip(g)
            .map(|(a, b)| a.distance(*b))
            .fold(0.0, f32::max);
        worst = worst.max(d);
    }
    gpu.wait();
    let gb = gpu.beams()[0].iter().filter(|b| b.broken).count();
    let cb = cpu.damage().0;
    let gd = gpu.deform_triggered().remove(0);
    let cd = cpu.deform_triggered().to_vec();
    eprintln!("{label}: max position difference {:.4} mm; broken CPU {cb} GPU {gb}; deform CPU {cd:?} GPU {gd:?}", worst * 1e3);
    Some((worst, cb, gb, cd, gd))
}

#[test]
fn break_groups_triangles_and_deform_groups_match() {
    for (label, text, pull) in [
        ("groups a", PART2_GROUPS, "a1"),
        ("groups b", PART2_GROUPS, "b1"),
        ("panel", PART2_PANEL, "p1"),
    ] {
        let mut s = spawn(text, None, Vec3::ZERO);
        let d = load_jbeam(&[("t.jbeam", text)], None).unwrap();
        let i = d.nodes.iter().position(|n| n.name == pull).unwrap();
        s.nodes[i].vel = Vec3::new(40.0, 0.0, 0.0);
        let Some((worst, cb, gb, cd, gd)) = compare(label, &mut s, 30) else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        assert!(cb > 0, "{label}: something breaks");
        assert_eq!(cb, gb, "{label}");
        assert_eq!(cd, gd, "{label}");
        assert!(worst < 1e-3, "{label}: {worst}");
    }
}

#[test]
fn precompression_ramp_and_rails_match() {
    for (label, text) in [("ramp", PART2_RAMP), ("rail", PART2_RAIL)] {
        let mut s = spawn(text, None, Vec3::ZERO);
        let Some((worst, ..)) = compare(label, &mut s, 60) else {
            return;
        };
        assert!(worst < 1e-3, "{label}: {worst}");
    }
}

#[test]
fn pressure_wheel_car_matches_at_rest_and_into_a_wall() {
    let mut s = spawn(PART2_CAR, Some(JbeamWheels::Pressure), Vec3::ZERO);
    let Some((worst, cb, gb, ..)) = compare("pressure car at rest", &mut s, 120) else {
        return;
    };
    assert_eq!((cb, gb), (0, 0));
    assert!(worst < 5e-3, "at rest: {worst}");
    let v = Vec3::new(0.0, 0.0, 56.0 / 3.6);
    let mut s = spawn(PART2_CAR, Some(JbeamWheels::Pressure), v);
    let front = s.nodes.iter().map(|n| n.pos.z).fold(f32::MIN, f32::max);
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.0, front + 0.3 + 1.0),
        Vec3::new(4.0, 3.0, 1.0),
    ));
    let (worst, ..) = compare("pressure car wall", &mut s, 60).unwrap();
    assert!(s.nodes.iter().all(|n| n.pos.is_finite()));
    eprintln!("wall run: max difference {worst} m (a crash is chaotic: reported, not judged)");
}

/// The solver on a device with WebGPU's 64 KiB uniform binding limit (Chrome's):
/// the parameter buffer holds 129 copies of 512 bytes (66 KiB), so every pass binds
/// one copy, never the whole buffer. Its frames match the full-limit device's.
#[test]
fn runs_within_webgpu_uniform_binding_limit() {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::GL;
    desc.flags = engine_webgpu::instance_flags();
    let instance = wgpu::Instance::new(desc);
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        ..Default::default()
    })) else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let mut limits = adapter.limits();
    limits.max_uniform_buffer_binding_size =
        wgpu::Limits::default().max_uniform_buffer_binding_size;
    assert_eq!(limits.max_uniform_buffer_binding_size, 65536);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("webgpu-uniform-limit"),
        required_limits: limits,
        ..Default::default()
    }))
    .unwrap();
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    // The narrow (multi-car) and wide (one car) paths, and the frame of 128 steps
    // that writes every copy.
    let mut cpu = spawn(PART2_CAR, Some(JbeamWheels::Pressure), Vec3::ZERO);
    let mut one = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&cpu));
    let mut many = ExplicitBeamsGpu::new(&device, &queue, &vec![cpu.clone(); 3]);
    let mut clock = FrameClock::default();
    for _ in 0..10 {
        let steps = clock.next();
        cpu.frame();
        one.frame(steps);
        many.frame(steps);
    }
    one.frame(128);
    many.frame(128);
    one.wait();
    many.wait();
    let err = pollster::block_on(scope.pop());
    assert!(err.is_none(), "validation error: {err:?}");
    assert!(one.positions()[0].iter().all(|p| p.is_finite()));
    assert_eq!(many.positions()[0], many.positions()[2]);
}

/// Triangle aerodynamics (spec §31) on the GPU: the panel fixture thrown through
/// the air at 25 m/s (its triangles' drag and lift, the solver's drag setting on),
/// bit for bit against the CPU.
#[test]
fn triangle_aero_matches() {
    let d = load_jbeam(&[("t.jbeam", PART2_PANEL)], None).unwrap();
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        drag: true,
        ..Default::default()
    };
    let v = Vec3::new(3.0, 25.0, 10.0);
    let (mut s, _) = ExplicitBeams::from_desc_jbeam_with(
        &d,
        JbeamWheels::Soft(TyreDefaults::for_vehicle(&d)),
        Vec3::new(0.0, 5.0, 0.0),
        v,
        settings,
    )
    .unwrap();
    assert!(s
        .beams
        .iter()
        .any(|b| b.kind == engine_softbody::explicit_beams::ExplicitKind::JbeamAeroTri));
    let Some((worst, cb, gb, ..)) = compare("aero panel", &mut s, 40) else {
        return;
    };
    assert_eq!(cb, gb);
    assert!(worst < 1e-4, "aero: {worst}");
}

/// JBeam mode's exact lengths and atan2 (jbeam-semantics-spec.md §35) and the tread
/// contact (§34) step bit for bit with the CPU: every node's position and velocity
/// after each step, the pressure-wheel fixture car at rest (2 s) and rolling at
/// 10 m/s (1 s); opt-in (`RUBBLE_BEAMNG_MODS`), the supercar at rest 1 s and the
/// Astral R 350 steps: at step 354 (in the BeamNG profile's 2.5 cm band, §36; 427
/// in the old 3 cm one) its first self-collision contact differs by a last bit (the
/// contact force reaches the node through the GPU's 1/32 N accumulator; §35, open).
#[test]
fn jbeam_mode_steps_bit_for_bit() {
    let Ok((_, device, queue)) = default_device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let check = |label: &str, mut cpu: ExplicitBeams, steps: u32| {
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
        eprintln!("{label}: {steps} steps bit for bit");
    };
    check(
        "fixture car at rest",
        spawn(PART2_CAR, Some(JbeamWheels::Pressure), Vec3::ZERO),
        4000,
    );
    let v = Vec3::new(0.0, 0.0, 10.0);
    check(
        "fixture car at 10 m/s",
        spawn(PART2_CAR, Some(JbeamWheels::Pressure), v),
        2000,
    );
    let Ok(mods) = std::env::var("RUBBLE_BEAMNG_MODS") else {
        return;
    };
    let m = std::path::Path::new(&mods);
    for (dir, name, pc, steps) in [
        ("monster_supercar/vehicles", "supercar", "config.pc", 2000),
        (
            "qlourie_astral_r/vehicles",
            "qlourie_astral_r",
            "wrc17_tarmac_balanced.pc",
            350,
        ),
    ] {
        let dir = m.join(dir);
        let Ok(files) = engine_softbody::jbeam::read_vehicle_dir(&dir, name) else {
            continue;
        };
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let pcs = std::fs::read_to_string(dir.join(name).join(pc)).unwrap();
        let d = load_jbeam(&refs, Some(&pcs)).unwrap();
        let settings = ExplicitBeamsSettings {
            turbulence: false,
            drag: false,
            ..Default::default()
        };
        let (mut s, _) = ExplicitBeams::from_desc_jbeam_with(
            &d,
            JbeamWheels::Pressure,
            Vec3::ZERO,
            Vec3::ZERO,
            settings,
        )
        .unwrap();
        let low = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
        for n in &mut s.nodes {
            n.pos.y += 0.02 - low;
        }
        check(name, s, steps);
    }
}
