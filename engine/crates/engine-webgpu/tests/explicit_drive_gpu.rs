//! Raycast wheels on the GPU (`slang/kernels/explicit_beams_drive.slang`) against
//! the CPU reference (`engine_softbody::explicit_drive`), bit for bit: the procedural
//! sedan on raycast wheels driving (throttle, steer, brake, handbrake) on flat ground,
//! over a ramp (an oriented box), into a wall (a box) and into a parked car (vehicle
//! contact, a contact group); a vehicle without wheels beside a driven one steps as
//! the CPU's. The CPU sums contact forces in the GPU's fixed point (`fixed_contact`).

use engine_softbody::explicit_beams::*;
use engine_softbody::explicit_drive::DriveInput;
use engine_softbody::procedural::{car_truck, CarParams};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::{Quat, Vec3};

const SETTINGS: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: true,
    drag: false,
    turbulence: false,
    triangle_throttle: true,
    centred_damping: false,
    centred_objective: false,
};

fn sedan() -> String {
    car_truck(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    })
}

fn driven(at: Vec3, yaw: f32, speed: f32) -> ExplicitBeams {
    let v = yaw_offset(Vec3::new(0.0, 0.0, speed), yaw);
    let (mut s, _) =
        ExplicitBeams::from_truck_raycast(&sedan(), None, at, yaw, v, SETTINGS).unwrap();
    s.fixed_contact = true;
    s.finish();
    s
}

fn plain(at: Vec3, yaw: f32) -> ExplicitBeams {
    let mut s = ExplicitBeams::from_truck_config_yaw(&sedan(), None, at, yaw, Vec3::ZERO, SETTINGS)
        .unwrap();
    s.fixed_contact = true;
    s.finish();
    s
}

/// Inputs over frames: launch, steer, brake, handbrake.
fn input(f: u32) -> DriveInput {
    match f {
        0..=59 => DriveInput {
            forward: 1.0,
            ..Default::default()
        },
        60..=99 => DriveInput {
            forward: 0.6,
            steer: -0.8,
            ..Default::default()
        },
        100..=129 => DriveInput {
            forward: 0.3,
            steer: 0.5,
            handbrake: 1.0,
            ..Default::default()
        },
        _ => DriveInput {
            brake: 1.0,
            steer: 0.2,
            ..Default::default()
        },
    }
}

/// CPU and GPU frame by frame: every node's position and velocity and every wheel's
/// state bit for bit. Returns how many wheel-frames had contact.
fn compare(label: &str, mut fleet: Vec<ExplicitBeams>, frames: u32, contact: bool) -> Option<u32> {
    let (_, device, queue) = default_device().ok()?;
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &fleet);
    gpu.set_vehicle_contact(contact);
    let mut cpu_world =
        engine_softbody::explicit_contact::ExplicitBeamsWorld::new(std::mem::take(&mut fleet));
    cpu_world.contact = contact;
    let mut touched = 0;
    for f in 0..frames {
        for (v, s) in cpu_world.vehicles.iter_mut().enumerate() {
            if s.drive.is_some() {
                s.set_drive_input(input(f));
                gpu.set_drive_input(v, input(f));
            }
        }
        let steps = cpu_world.frame_60();
        gpu.frame(steps);
        let gp = gpu.positions();
        let gv = gpu.velocities();
        let drives: Vec<_> = cpu_world
            .vehicles
            .iter()
            .map(|s| s.drive.as_deref().cloned())
            .collect();
        let gd = gpu.drive_states(&drives);
        for (v, s) in cpu_world.vehicles.iter().enumerate() {
            let cp = s.positions();
            for (i, (a, b)) in cp.iter().zip(&gp[v]).enumerate() {
                assert!(
                    a.to_array().map(f32::to_bits) == b.to_array().map(f32::to_bits),
                    "{label}: frame {f} vehicle {v} node {i}: CPU {a:?} GPU {b:?}"
                );
                let (ca, cb) = (s.nodes[i].vel, gv[v][i]);
                assert!(
                    ca.to_array().map(f32::to_bits) == cb.to_array().map(f32::to_bits),
                    "{label}: frame {f} vehicle {v} node {i} velocity: CPU {ca:?} GPU {cb:?}"
                );
            }
            if let (Some(c), Some(g)) = (&s.drive, &gd[v]) {
                assert_eq!(c.wheels, g.wheels, "{label}: frame {f} vehicle {v} wheels");
                assert_eq!(
                    c.smoothed, g.smoothed,
                    "{label}: frame {f} vehicle {v} input"
                );
                touched += c.wheels.iter().filter(|w| w.contact).count() as u32;
            }
        }
    }
    let d = cpu_world.vehicles[0].drive.as_ref().unwrap();
    println!(
        "{label}: {frames} frames bit for bit; speed {:.2} m/s, wheel contacts {touched}",
        d.speed
    );
    Some(touched)
}

#[test]
fn driving_on_flat_ground_bitwise() {
    let Some(t) = compare(
        "flat",
        vec![driven(Vec3::new(0.0, 0.6, 0.0), 0.0, 0.0)],
        160,
        false,
    ) else {
        return;
    };
    assert!(t > 0);
}

#[test]
fn driving_over_a_ramp_into_a_wall_bitwise() {
    let mut s = driven(Vec3::new(0.0, 0.6, 0.0), 0.0, 8.0);
    s.shapes.push(StaticShape::oriented_box(
        Vec3::new(0.6, 0.0, 7.0),
        Vec3::new(0.5, 0.2, 2.0),
        Quat::from_rotation_x(-0.15),
    ));
    s.shapes
        .push(StaticShape::cylinder(Vec3::new(-3.0, 1.0, 9.0), 0.15, 1.0));
    s.boxes.push(StaticBox::centered(
        Vec3::new(0.0, 1.0, 16.0),
        Vec3::new(4.0, 1.0, 0.3),
    ));
    s.finish();
    compare("ramp+wall", vec![s], 160, false);
}

#[test]
fn driving_into_a_parked_car_with_a_plain_car_beside_bitwise() {
    // The driven car, a parked car ahead (vehicle contact, one contact group), and a
    // plain car on beam tyres far off: it steps as the CPU's.
    let fleet = vec![
        driven(Vec3::new(0.0, 0.6, 0.0), 0.0, 10.0),
        plain(Vec3::new(0.0, 0.6, 9.0), 90.0),
        plain(Vec3::new(30.0, 0.6, 0.0), 0.0),
    ];
    compare("into a car", fleet, 120, true);
}

/// The generated sedan (JBeam mode, its wheels parts of their own: the rim stop on)
/// dropped 1.5 m under throttle: its hubs reach their rims' radius over the ground and
/// the rim stop (`explicit_drive::rim_stop`, the kernel's `drvRimStop`) holds them,
/// bit for bit with the CPU.
#[test]
fn landing_on_the_rims_bitwise() {
    use engine_softbody::authoring::gen2::{rig as lattice, vg_input};
    use engine_softbody::authoring::rig;
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
    let d = engine_softbody::jbeam::load_jbeam(&refs, pc).expect("the sedan's jbeam");
    let make = || {
        let (mut s, _) = ExplicitBeams::from_desc_jbeam_raycast(
            &d,
            Vec3::new(0.0, 1.5, 0.0),
            Vec3::ZERO,
            SETTINGS,
        )
        .unwrap();
        s.fixed_contact = true;
        s.finish();
        s
    };
    // The CPU alone first: the stop must be reached in this run.
    let mut s = make();
    let dr = s.drive.as_ref().unwrap();
    assert!(dr.config.rim_stop, "the generated sedan's rim stop is off");
    let wheels = dr.config.wheels.clone();
    let mut reached = 0;
    for f in 0..90u32 {
        s.set_drive_input(input(f));
        s.frame();
        for w in &wheels {
            let hub = 0.5 * (s.nodes[w.node1 as usize].pos.y + s.nodes[w.node2 as usize].pos.y)
                + s.origin.y
                - s.ground;
            if hub < w.rim_radius + 1e-3 {
                reached += 1;
            }
        }
    }
    println!("landing: {reached} wheel-frames at the rim stop (CPU)");
    assert!(reached > 0, "the drop never reached the rim stop");
    compare("landing on the rims", vec![make()], 90, false);
}
