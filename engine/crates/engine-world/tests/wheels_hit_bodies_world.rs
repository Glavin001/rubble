//! Raycast wheels on World bodies (`ExplicitVehiclesDef::wheels_hit_bodies`, on by
//! default; known-bugs F7): the procedural raycast sedan parked on a 4 t dynamic box
//! platform 1 m high lying on the World's ground (the car's own ground plane at the same height)
//! drives off it at full throttle and pushes the platform back; with the setting off its
//! wheels hang above the platform and it stays put. The setting is live
//! (`World::set_explicit_wheels_hit_bodies`).

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_drive::DriveInput;
use engine_softbody::procedural::{car_truck, CarParams};
use glam::Vec3;
use rubble::*;

/// Distance driven along +z over 4 s of throttle after 1 s settling, the platform's speed
/// along z at the end, and the lowest the car's centre came below where it settled.
fn run(on: bool) -> anyhow::Result<Option<(f32, f32, f32)>> {
    let text = car_truck(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let settings = ExplicitBeamsSettings {
        drag: false,
        turbulence: false,
        ..Default::default()
    };
    let (mut car, _) = ExplicitBeams::from_truck_raycast(&text, None, Vec3::ZERO, 0.0, Vec3::ZERO, settings)
        .map_err(anyhow::Error::msg)?;
    // The car's forward along +z.
    let [c, b, _] = car.drive.as_ref().unwrap().config.refs;
    let f = car.nodes[c as usize].pos - car.nodes[b as usize].pos;
    let yaw = (-f.x).atan2(f.z);
    let (s, co) = yaw.sin_cos();
    let lo = car.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let m: f32 = car.nodes.iter().map(|n| n.mass).sum();
    let cx = car.nodes.iter().map(|n| n.pos * n.mass).sum::<Vec3>() / m;
    for n in &mut car.nodes {
        let p = n.pos - Vec3::new(cx.x, 0.0, cx.z);
        n.pos = Vec3::new(co * p.x + s * p.z, p.y - lo + 1.05, -s * p.x + co * p.z);
    }
    car.ground = 0.0;
    car.finish();
    let Ok(mut w) = World::new(GpuContext::Default, &WorldDef::default()) else {
        eprintln!("no GPU: skipped");
        return Ok(None);
    };
    let g = w.create_body(&BodyDef::default())?;
    w.create_shape(g, &ShapeDef::default(), Geometry::Plane { normal: [0.0, 1.0, 0.0], offset: 0.0 })?;
    let half = [1.2f32, 0.5, 3.0];
    let mass = 4000.0f32;
    let p = w.create_body(&BodyDef::dynamic([0.0, half[1], 0.0]))?;
    w.create_box_shape(
        p,
        &ShapeDef {
            density: mass / (8.0 * half[0] * half[1] * half[2]),
            ..ShapeDef::default()
        },
        half,
    )?;
    w.add_explicit_vehicles(
        std::slice::from_ref(&car),
        &ExplicitVehiclesDef {
            sleep: None,
            wheels_hit_bodies: on,
            ..ExplicitVehiclesDef::default()
        },
    )?;
    assert_eq!(w.explicit_wheels_hit_bodies(), on);
    let centre = |w: &World| {
        let x = w.explicit_vehicles().unwrap().positions();
        x[0].iter().zip(&car.nodes).map(|(p, n)| *p * n.mass).sum::<Vec3>() / m
    };
    let dt = w.solver().dt;
    let mut throttle = |w: &mut World, forward: f32| {
        w.explicit_vehicles_mut().unwrap().set_drive_input(0, DriveInput { forward, ..Default::default() });
    };
    for _ in 0..(1.0 / dt).round() as u32 {
        w.step()?;
        w.poll_blocking()?;
    }
    let x1 = centre(&w);
    throttle(&mut w, 1.0);
    let mut low = f32::MAX;
    for _ in 0..(4.0 / dt).round() as u32 {
        w.step()?;
        w.poll_blocking()?;
        low = low.min(centre(&w).y);
    }
    w.explicit_vehicles().unwrap().kernel_status().map_err(anyhow::Error::msg)?;
    let x2 = centre(&w);
    let vp = w.read_body_state(p)?.linear_velocity[2];
    println!("wheels on bodies {on}: settled at y {:.3}, went {:.2} m, platform {vp:.3} m/s, dropped {:.3} m", x1.y, x2.z - x1.z, x1.y - low);
    Ok(Some((x2.z - x1.z, vp, x1.y - low)))
}

#[test]
fn a_car_on_a_world_platform_drives_off_it() -> anyhow::Result<()> {
    let Some((went, _, drop)) = run(true)? else {
        return Ok(());
    };
    assert!(went > 3.5, "drove {went} m");
    assert!(drop > 0.8, "came down off the platform: {drop} m");
    let Some((stuck, _, _)) = run(false)? else {
        return Ok(());
    };
    assert!(stuck.abs() < 0.5, "without the setting the wheels hang: moved {stuck} m");
    Ok(())
}
