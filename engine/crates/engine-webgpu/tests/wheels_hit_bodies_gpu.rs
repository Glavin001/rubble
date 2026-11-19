//! Raycast wheels on rigid bodies on the GPU (`explicit_beams_drive.slang`
//! drvRaysBodies, drvTyreBody) against the CPU reference
//! (`engine_softbody::explicit_drive`, `DriveConfig::see_bodies`): the raycast sedan
//! resting on a free-floating 2 t crate (no ground), then at full throttle. The coupled
//! paths agree to float noise, not bit for bit (the GPU's proxy records are relative to
//! the vehicle's origin, the CPU's in the World), as the coupling's own parity test
//! (`explicit_rigid_gpu.rs`).
//! * The wheels stand on the crate on the GPU as on the CPU, the car drives forward and
//!   pushes the crate back; car and crate speeds agree within 1 %.
//! * The GPU closes horizontal momentum (car plus crate) to the fixed-point rounding.
//! * With the setting off on the GPU the frame is the frame without it bit for bit (the
//!   wheels hang above the crate).

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_drive::DriveInput;
use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet, ProxyShape, RigidProxy};
use engine_softbody::procedural::{car_truck, CarParams};
use engine_webgpu::explicit_coupling::{read_proxies, write_proxies};
use engine_webgpu::{default_device, CouplingCapacity, ExplicitBeamsGpu};
use glam::{Quat, Vec3};

const CAP: CouplingCapacity = CouplingCapacity {
    bodies: 4,
    shapes: 8,
    planes: 64,
};
const CRATE: f32 = 2000.0;
/// The drive suites' settings (explicit_drive_gpu.rs): the GPU's bit-for-bit configuration.
const SETTINGS: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: true,
    drag: false,
    turbulence: false,
    triangle_throttle: true,
    centred_damping: false,
    centred_objective: false,
};

fn car(see: bool) -> ExplicitBeams {
    let text = car_truck(&CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    });
    let settings = SETTINGS;
    let (mut s, _) = ExplicitBeams::from_truck_raycast(&text, None, Vec3::ZERO, 0.0, Vec3::ZERO, settings).unwrap();
    let lo = s.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut s.nodes {
        n.pos.y += 0.55 - lo;
    }
    s.ground = -1.0e4;
    s.fixed_contact = true;
    s.drive.as_mut().unwrap().config.see_bodies = see;
    s.finish();
    s
}

fn crate_under(s: &ExplicitBeams) -> ProxySet {
    let [c, b, _] = s.drive.as_ref().unwrap().config.refs;
    let f = s.nodes[c as usize].pos - s.nodes[b as usize].pos;
    let m: f32 = s.nodes.iter().map(|n| n.mass).sum();
    let x = s.nodes.iter().map(|n| n.pos * n.mass).sum::<Vec3>() / m;
    let half = Vec3::new(1.2, 0.25, 3.0);
    let i = Vec3::new(
        CRATE / 3.0 * (half.y * half.y + half.z * half.z),
        CRATE / 3.0 * (half.x * half.x + half.z * half.z),
        CRATE / 3.0 * (half.x * half.x + half.y * half.y),
    );
    let mut set = ProxySet {
        bodies: vec![RigidProxy {
            body: 1,
            pos: Vec3::new(x.x, 0.25, x.z),
            rot: Quat::from_rotation_y(f.x.atan2(f.z)),
            vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            inv_mass: 1.0 / CRATE,
            inv_inertia: Vec3::ONE / i,
            share: 1.0,
            count: 0,
        }],
        shapes: vec![ProxyShape::new(0, Vec3::ZERO, Quat::IDENTITY, ProxyGeom::Box { half }, 1, None)],
        frozen: false,
    };
    set.begin_window();
    set
}

fn throttle(f: u32) -> DriveInput {
    DriveInput {
        forward: if f >= 20 { 1.0 } else { 0.0 },
        ..Default::default()
    }
}

#[test]
fn wheels_on_a_crate_match_the_cpu_and_close_momentum() {
    let Ok((_, device, queue)) = default_device() else {
        eprintln!("skip: no GPU");
        return;
    };
    let sim = car(true);
    let set = crate_under(&sim);
    let mut gpu = ExplicitBeamsGpu::with_coupling(&device, &queue, std::slice::from_ref(&sim), Some(CAP));
    write_proxies(&gpu, &queue, 0, &set).unwrap();
    gpu.set_coupling_on(true);
    assert!(gpu.wheels_see_bodies());
    let mut cpu = sim.clone();
    cpu.rigid = Some(Box::new(set.clone()));
    let masses: Vec<f32> = sim.nodes.iter().map(|n| n.mass).collect();
    let m: f32 = masses.iter().sum();
    let mut contacts = (0, 0);
    let mut gset = set.clone();
    for f in 0..50 {
        cpu.set_drive_input(throttle(f));
        gpu.set_drive_input(0, throttle(f));
        let steps = cpu.clock.next();
        gpu.frame(steps);
        for _ in 0..steps {
            cpu.step();
        }
        let gd = gpu.drive_states(&[cpu.drive.as_deref().cloned()]);
        contacts.0 += cpu.drive.as_ref().unwrap().wheels.iter().filter(|w| w.contact).count();
        contacts.1 += gd[0].as_ref().unwrap().wheels.iter().filter(|w| w.contact).count();
    }
    read_proxies(&gpu, &device, &queue, 0, &mut gset);
    let cset = cpu.rigid.as_ref().unwrap();
    let gvel = gpu.velocities();
    let [c, b, _] = sim.drive.as_ref().unwrap().config.refs;
    let fw = (sim.nodes[c as usize].pos - sim.nodes[b as usize].pos) * Vec3::new(1.0, 0.0, 1.0);
    let fw = fw.normalize();
    let pg: Vec3 = gvel[0].iter().zip(&masses).map(|(v, m)| *v * *m).sum::<Vec3>() + gset.bodies[0].vel * CRATE;
    let car_g = gvel[0].iter().zip(&masses).map(|(v, m)| v.dot(fw) * m).sum::<f32>() / m;
    let car_c = cpu.nodes.iter().map(|n| n.vel.dot(fw) * n.mass).sum::<f32>() / m;
    let (box_g, box_c) = (gset.bodies[0].vel.dot(fw), cset.bodies[0].vel.dot(fw));
    let ph = Vec3::new(pg.x, 0.0, pg.z);
    println!(
        "wheel contacts CPU {} GPU {}; car GPU {car_g:.4} CPU {car_c:.4} m/s; crate GPU {box_g:.4} CPU {box_c:.4} m/s; GPU horizontal momentum {:.3} N s of {:.0}",
        contacts.0,
        contacts.1,
        ph.length(),
        car_g * m
    );
    assert!(contacts.1 > 100 && contacts.0.abs_diff(contacts.1) <= 4, "contacts {contacts:?}");
    assert!(car_g > 0.3 && box_g < -0.1, "car {car_g} crate {box_g}");
    let rel = |a: f32, b: f32| (a - b).abs() / b.abs().max(1e-3);
    assert!(rel(car_g, car_c) < 0.01, "car GPU {car_g} CPU {car_c}");
    assert!(rel(box_g, box_c) < 0.01, "crate GPU {box_g} CPU {box_c}");
    assert!(ph.length() < 0.005 * car_g * m, "GPU momentum {ph}");
}

/// The soft flatbed of `engine-softbody/tests/wheels_hit_bodies.rs` (6 × 6 m, 0.8 m deep,
/// its top face collision triangles) on its own ground at y = 0 centred at (`cx`, `cz`).
fn flatbed(cx: f32, cz: f32) -> ExplicitBeams {
    let n = 13;
    let id = |i: usize, j: usize, k: usize| (j * n + k) * n + i;
    let mut text = String::from("flatbed\n\nglobals\n2000, 0, tracks/semi\n\nminimass\n4\n\nset_beam_defaults 200000, 400, 400000, 4000000\n\nnodes\n");
    for j in 0..2 {
        for k in 0..n {
            for i in 0..n {
                text += &format!("{}, {}, {}, {}, n\n", id(i, j, k), cx - 3.0 + 0.5 * i as f32, 0.8 * j as f32, cz - 3.0 + 0.5 * k as f32);
            }
        }
    }
    text += "\nbeams\n";
    for j in 0..2usize {
        for k in 0..n {
            for i in 0..n {
                for (di, dj, dk) in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 0, -1), (1, 1, 0), (0, 1, 1), (1, -1, 0), (0, -1, 1)] {
                    let (i2, j2, k2) = (i as i32 + di, j as i32 + dj, k as i32 + dk);
                    if (0..n as i32).contains(&i2) && (0..2).contains(&j2) && (0..n as i32).contains(&k2) {
                        text += &format!("{}, {}\n", id(i, j, k), id(i2 as usize, j2 as usize, k2 as usize));
                    }
                }
            }
        }
    }
    text += "\nsubmesh\ncab\n";
    for k in 0..n - 1 {
        for i in 0..n - 1 {
            let (a, b, c, d) = (id(i, 1, k), id(i + 1, 1, k), id(i + 1, 1, k + 1), id(i, 1, k + 1));
            text += &format!("{a}, {c}, {b}, c\n{a}, {d}, {c}, c\n");
        }
    }
    text += "\nend\n";
    let settings = SETTINGS;
    let mut s = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).expect("flatbed");
    s.ground = 0.0;
    s.fixed_contact = true;
    s.finish();
    s
}

/// The raycast sedan (wheels seeing bodies) dropped onto the flatbed, then driven off it:
/// the GPU's contact group (one workgroup, or `split` across workgroups) against the CPU
/// reference's world, every node's position and velocity and every wheel's state bit for
/// bit, frame by frame. Returns the wheel-frames on the flatbed.
fn car_on_vehicle(split: bool) -> Option<u32> {
    let (_, device, queue) = default_device().ok()?;
    let mut a = car(std::env::var("WHB_SEE").as_deref() != Ok("0"));
    let lo = a.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut a.nodes {
        n.pos.y += 0.95 - lo;
    }
    a.ground = 0.0;
    let m: f32 = a.nodes.iter().map(|n| n.mass).sum();
    let x = a.nodes.iter().map(|n| n.pos * n.mass).sum::<Vec3>() / m;
    let far = if std::env::var("WHB_FAR").is_ok() { 30.0 } else { 0.0 };
    let fleet = vec![a, flatbed(x.x + far, x.z)];
    let mut gpu = ExplicitBeamsGpu::new(&device, &queue, &fleet);
    gpu.set_vehicle_contact(true);
    gpu.set_group_split(split);
    let mut cpu = engine_softbody::explicit_contact::ExplicitBeamsWorld::new(fleet);
    let mut on = 0;
    for f in 0..120u32 {
        let input = DriveInput {
            forward: if f >= 40 { 1.0 } else { 0.0 },
            ..Default::default()
        };
        cpu.vehicles[0].set_drive_input(input);
        gpu.set_drive_input(0, input);
        let steps = cpu.frame_60();
        gpu.frame(steps);
        gpu.kernel_status().unwrap();
        let (gp, gv) = (gpu.positions(), gpu.velocities());
        for (v, s) in cpu.vehicles.iter().enumerate() {
            for (i, n) in s.nodes.iter().enumerate() {
                let p = s.origin + n.pos;
                assert!(
                    p.to_array().map(f32::to_bits) == gp[v][i].to_array().map(f32::to_bits)
                        && n.vel.to_array().map(f32::to_bits) == gv[v][i].to_array().map(f32::to_bits),
                    "split {split}: frame {f} vehicle {v} node {i}: CPU {p:?} {:?} GPU {:?} {:?}",
                    n.vel,
                    gp[v][i],
                    gv[v][i]
                );
            }
        }
        let gd = gpu.drive_states(&[cpu.vehicles[0].drive.as_deref().cloned(), None]);
        let c = cpu.vehicles[0].drive.as_ref().unwrap();
        assert_eq!(c.wheels, gd[0].as_ref().unwrap().wheels, "split {split}: frame {f} wheels");
        on += c.wheels.iter().filter(|w| w.contact).count() as u32;
    }
    println!("car on a vehicle, group split {split}: 120 frames bit for bit, {on} wheel-frames in contact");
    Some(on)
}

#[test]
fn wheels_on_another_vehicle_are_the_cpu_reference_bit_for_bit() {
    for split in [false, true] {
        let Some(on) = car_on_vehicle(split) else {
            eprintln!("skip: no GPU");
            return;
        };
        assert!(on > 200, "the wheels stood on the flatbed ({on} wheel-frames)");
    }
}

#[test]
fn wheels_not_seeing_bodies_are_the_frame_without_it_bit_for_bit() {
    let Ok((_, device, queue)) = default_device() else {
        eprintln!("skip: no GPU");
        return;
    };
    // The same car built without the setting, and one built with it then turned off.
    let plain = car(false);
    let set = crate_under(&plain);
    let run = |sim: &ExplicitBeams, off_after_build: bool| {
        let mut gpu = ExplicitBeamsGpu::with_coupling(&device, &queue, std::slice::from_ref(sim), Some(CAP));
        write_proxies(&gpu, &queue, 0, &set).unwrap();
        gpu.set_coupling_on(true);
        if off_after_build {
            gpu.set_wheels_see_bodies(false);
        }
        for f in 0..30 {
            gpu.set_drive_input(0, throttle(f));
            gpu.frame(33);
        }
        (gpu.positions(), gpu.velocities())
    };
    let a = run(&plain, false);
    let b = run(&car(true), true);
    let bits = |v: &Vec<Vec<Vec3>>| v.iter().flatten().flat_map(|x| x.to_array().map(f32::to_bits)).collect::<Vec<_>>();
    assert!(bits(&a.0) == bits(&b.0) && bits(&a.1) == bits(&b.1), "off is not the frame without it");
}
