//! Centred damping (`ExplicitBeamsSettings::centred_damping`,
//! `engine/docs/jbeam-semantics-spec.md` §39) on the native Metal host against the
//! CPU reference, bit for bit every step, on the per-vehicle frame and on the wide
//! passes (as on wgpu: `engine-webgpu/tests/explicit_centred_damping_gpu.rs`).
use engine_metal::MetalExplicitBeams;
use engine_softbody::explicit_beams::{
    ExplicitBeams, ExplicitBeamsSettings, JbeamWheels, StaticBox,
};
use engine_softbody::jbeam::load_jbeam;
use engine_softbody::jbeam_fixture::{latched_box_jbeam, PART2_CAR};
use engine_softbody::truck_export::TyreDefaults;
use glam::Vec3;

fn spawn(text: &str, wheels: Option<JbeamWheels>, v: Vec3) -> ExplicitBeams {
    let d = load_jbeam(&[("t.jbeam", text)], None).unwrap();
    let wheels = wheels.unwrap_or(JbeamWheels::Soft(TyreDefaults::for_vehicle(&d)));
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        drag: false,
        centred_damping: true,
        centred_objective: false,
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

fn metal(fleet: &[ExplicitBeams]) -> Option<MetalExplicitBeams> {
    match MetalExplicitBeams::new(fleet) {
        Ok(m) => Some(m),
        Err(e) if e.to_string().contains("no Metal device") => {
            eprintln!("no Metal device: skipped");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

/// CPU and Metal (frame and wide) one step at a time; every node bit for bit.
fn bit_for_bit(label: &str, mut cpu: ExplicitBeams, steps: u32) -> Option<usize> {
    let mut frame = metal(std::slice::from_ref(&cpu))?;
    let mut wide = metal(std::slice::from_ref(&cpu))?;
    wide.set_vehicle_contact(false);
    assert!(wide.set_wide(true), "{label}: the wide passes run it");
    let bits = |a: Vec3| a.to_array().map(f32::to_bits);
    for step in 0..steps {
        cpu.step();
        for (path, gpu) in [("frame", &mut frame), ("wide", &mut wide)] {
            gpu.frame(1);
            gpu.wait();
            let (p, v) = (gpu.positions(), gpu.velocities());
            let bad = (0..cpu.nodes.len())
                .filter(|&i| {
                    bits(cpu.nodes[i].pos) != bits(p[0][i])
                        || bits(cpu.nodes[i].vel) != bits(v[0][i])
                })
                .count();
            assert_eq!(
                bad, 0,
                "{label} ({path}): step {step}: {bad} nodes differ from the CPU"
            );
        }
    }
    let cb: Vec<bool> = cpu.beams.iter().map(|b| b.broken).collect();
    for (path, gpu) in [("frame", &mut frame), ("wide", &mut wide)] {
        let gb: Vec<bool> = gpu.beams()[0].iter().map(|b| b.broken).collect();
        assert_eq!(cb, gb, "{label} ({path}): broken beams");
    }
    let broken = cb.iter().filter(|&&b| b).count();
    eprintln!("{label}: {steps} steps bit for bit on Metal (frame and wide), {broken} broken");
    Some(broken)
}

#[test]
fn metal_centred_damping_steps_bit_for_bit() {
    let at_rest = spawn(PART2_CAR, Some(JbeamWheels::Pressure), Vec3::ZERO);
    if bit_for_bit("fixture car at rest", at_rest, 2000).is_none() {
        return;
    }
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
    let broken = bit_for_bit("latched box into a wall", boxed, 1500).unwrap();
    assert!(broken > 0);
}
