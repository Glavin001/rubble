//! Driving an explicit vehicle in the World (`brick-ring-drive.html`): the procedural
//! sedan on raycast wheels (`engine_softbody::explicit_drive`), its input set through
//! the World's explicit vehicles (`ExplicitBeamsGpu::set_drive_input`) while the World
//! steps, against the CPU reference (`ExplicitBeams::frame`) with the same input per
//! frame: every node's position and velocity, frame by frame, bit for bit.
//! The run crosses the origin rebase (100 m from the origin) to cover the World's
//! coupling records moving with the origin (`cplRebase`).

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_drive::DriveInput;
use engine_softbody::procedural::{car_truck, CarParams};
use glam::Vec3;
use rubble::explicit_scene::place_vehicle;
use rubble::*;

const SETTINGS: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: true,
    drag: false,
    turbulence: false,
    triangle_throttle: true,
    centred_damping: false,
    centred_objective: false,
};

/// Throttle, then a turn, the handbrake, the brakes.
fn input(f: u32) -> DriveInput {
    match f {
        0..=89 => DriveInput {
            forward: 1.0,
            ..Default::default()
        },
        90..=119 => DriveInput {
            forward: 0.6,
            steer: -0.5,
            ..Default::default()
        },
        120..=139 => DriveInput {
            forward: 0.3,
            steer: 0.4,
            handbrake: 1.0,
            ..Default::default()
        },
        _ => DriveInput {
            brake: 1.0,
            ..Default::default()
        },
    }
}

#[test]
fn driving_in_the_world_is_the_cpu_reference() -> anyhow::Result<()> {
    let text = car_truck(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let (template, _) =
        ExplicitBeams::from_truck_raycast(&text, None, Vec3::ZERO, 0.0, Vec3::ZERO, SETTINGS)
            .map_err(anyhow::Error::msg)?;
    // From 85 m out along +z at 20 m/s: the origin re-bases within the run.
    let mut car = place_vehicle(&template, Vec3::new(0.0, 0.0, 85.0), 0, 20.0, 0.0);
    car.fixed_contact = true;
    if let Some(d) = car.drive.as_mut() {
        d.roll(20.0);
    }
    car.finish();
    let Ok(mut w) = World::new(GpuContext::Default, &WorldDef::default()) else {
        eprintln!("no GPU: skipped");
        return Ok(());
    };
    let g = w.create_body(&BodyDef::default())?;
    w.create_shape(
        g,
        &ShapeDef::default(),
        Geometry::Plane {
            normal: [0.0, 1.0, 0.0],
            offset: 0.0,
        },
    )?;
    w.add_explicit_vehicles(
        std::slice::from_ref(&car),
        &ExplicitVehiclesDef {
            sleep: None,
            ..ExplicitVehiclesDef::default()
        },
    )?;
    let mut cpu = car;
    let (mut cpu_frames, mut cpu_steps) = (0u32, 0u64);
    let mut compared = 0;
    let mut worst = (0.0f32, 0.0f32);
    let mut bitwise = true;
    for _ in 0..170 {
        // The frame the next World step runs takes the input of its index.
        let (_, windows) = w.explicit_vehicle_steps().unwrap_or((0, 0));
        w.explicit_vehicles_mut()
            .expect("vehicles")
            .set_drive_input(0, input(windows as u32));
        w.step()?;
        let _ = w.poll_blocking()?;
        let (steps, _) = w.explicit_vehicle_steps().expect("vehicles");
        // The CPU reference to the same step count, frame by frame.
        while cpu_steps < steps {
            cpu.set_drive_input(input(cpu_frames));
            cpu_steps += u64::from(cpu.frame());
            cpu_frames += 1;
        }
        anyhow::ensure!(
            cpu_steps == steps,
            "frames do not line up: CPU {cpu_steps} steps, GPU {steps}"
        );
        let x = w.explicit_vehicles().expect("vehicles");
        x.kernel_status().map_err(anyhow::Error::msg)?;
        let (gp, gv) = (x.positions(), x.velocities());
        for (i, n) in cpu.nodes.iter().enumerate() {
            let (p, v) = (n.pos + cpu.origin, n.vel);
            let (dp, dv) = ((p - gp[0][i]).length(), (v - gv[0][i]).length());
            bitwise &= dp == 0.0 && dv == 0.0;
            worst = (worst.0.max(dp), worst.1.max(dv));
        }
        compared += 1;
    }
    let speed = cpu.drive.as_ref().map_or(0.0, |d| d.speed);
    let far = (cpu.nodes[0].pos + cpu.origin).length();
    println!(
        "{compared} World steps, {cpu_frames} frames; origin {:?}, node 0 {far:.1} m out; \
         speed {speed:.2} m/s; worst node difference {:.3e} m, {:.3e} m/s; bit for bit: {bitwise}",
        cpu.origin, worst.0, worst.1
    );
    assert!(cpu.origin.length() > 100.0, "the run re-based");
    // Measured (M3 Max, 2026-11-07): 170 steps bit for bit through the re-base.
    assert!(
        bitwise,
        "the World's drive left the CPU reference: {worst:?}"
    );
    Ok(())
}
