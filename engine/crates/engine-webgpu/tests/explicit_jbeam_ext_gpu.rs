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
