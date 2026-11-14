//! Raycast wheels (`engine_softbody::explicit_drive`, `slang/kernels/explicit_beams_drive.slang`)
//! on the native Metal host against the CPU reference, with the bars of
//! `explicit_parity.rs` (Metal compiles the kernels with fast math, so not bit for
//! bit): the procedural sedan on raycast wheels settles, pulls away, turns and
//! brakes; positions agree to millimetres, the speed and heading to a few percent.

use engine_metal::MetalExplicitBeams;
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, FrameClock};
use engine_softbody::explicit_drive::{Drive, DriveInput};
use engine_softbody::procedural::{car_truck, CarParams};
use glam::Vec3;

fn car() -> ExplicitBeams {
    let text = car_truck(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let settings = ExplicitBeamsSettings {
        turbulence: false,
        drag: false,
        ..Default::default()
    };
    let (mut s, _) = ExplicitBeams::from_truck_raycast(
        &text,
        None,
        Vec3::new(0.0, 0.6, 0.0),
        0.0,
        Vec3::ZERO,
        settings,
    )
    .unwrap();
    s.fixed_contact = true;
    s.finish();
    s
}

fn heading(d: &Drive) -> f32 {
    let f: Vec3 = d.wheels.iter().map(|w| w.axle.cross(w.up)).sum();
    f.x.atan2(f.z)
}

#[test]
fn metal_raycast_sedan_drives_like_the_cpu() {
    let mut cpu = car();
    let mut gpu = match MetalExplicitBeams::new(std::slice::from_ref(&cpu)) {
        Ok(m) => m,
        Err(e) if e.to_string().contains("no Metal device") => return,
        Err(e) => panic!("{e}"),
    };
    let mut clock = FrameClock::default();
    let mut g_drive = cpu.drive.as_deref().cloned().unwrap();
    let phases = [
        (60u32, DriveInput::default()),
        (
            120,
            DriveInput {
                forward: 1.0,
                ..Default::default()
            },
        ),
        (
            90,
            DriveInput {
                forward: 0.4,
                steer: -0.8,
                ..Default::default()
            },
        ),
        (
            90,
            DriveInput {
                brake: 1.0,
                ..Default::default()
            },
        ),
    ];
    for (frames, input) in phases {
        cpu.set_drive_input(input);
        gpu.set_drive_input(0, input);
        for _ in 0..frames {
            cpu.frame();
            gpu.frame(clock.next());
        }
        gpu.read_drive(0, &mut g_drive);
        let c = cpu.drive.as_deref().unwrap();
        let p = cpu.positions();
        let g = gpu.positions().remove(0);
        let max = p
            .iter()
            .zip(&g)
            .map(|(a, b)| a.distance(*b))
            .fold(0.0f32, f32::max);
        eprintln!(
            "Metal raycast sedan {input:?}: speed CPU {:.3} GPU {:.3} m/s, heading CPU {:.4} GPU {:.4} rad, max position difference {:.2} mm",
            c.speed,
            g_drive.speed,
            heading(c),
            heading(&g_drive),
            max * 1e3
        );
        assert!(
            (c.speed - g_drive.speed).abs() < 0.03 * c.speed.abs().max(1.0),
            "speed"
        );
        assert!((heading(c) - heading(&g_drive)).abs() < 0.03, "heading");
        assert!(max < 0.05, "positions {max} m apart");
        assert!(g_drive
            .wheels
            .iter()
            .zip(&c.wheels)
            .all(|(a, b)| a.contact == b.contact));
    }
}
