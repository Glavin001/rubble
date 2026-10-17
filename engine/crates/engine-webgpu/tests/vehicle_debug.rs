//! Manual trace of a preset's drivetrain (`PRESET=name cargo test --test vehicle_debug -- --ignored --nocapture`).
use engine_schema::{AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_vehicle::{presets, VehicleInput};
use engine_webgpu::WgpuSim;

#[test]
#[ignore]
fn trace_preset_launch() {
    let name = std::env::var("PRESET").unwrap_or_else(|_| "monster_truck".into());
    let desc = presets::by_name(&name).expect("preset");
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 32768,
        max_joints: 64,
        max_candidates: 65536,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 10;
    sim.set_params(p);
    sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    let body = sim.add_body(BodyState {
        position: [0.0, -5.0, 0.0],
        inv_mass: 0.0,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: [0.0; 3],
        flags: 0,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: [0.0, -5.0, 0.0],
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
    });
    sim.add_collider(Collider {
        body,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [400.0, 5.0, 400.0, 400.0],
    });
    let v = sim.add_vehicle(&desc, [0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("vehicle");
    for i in 0..150 {
        sim.step().expect("step");
        if i % 10 == 0 {
            let st = sim.vehicle_state(v);
            let c = sim.read_body(sim.vehicle_host().chassis_body(v));
            eprintln!(
                "settle {i}: y {:.3} q {:?} v {:?} flags {} contacts {} L {:?}",
                c.position[1],
                c.orientation,
                c.linear_vel,
                c.flags,
                st.contacts,
                st.wheels.iter().map(|w| (w.suspension_length * 1000.0).round() / 1000.0).collect::<Vec<_>>()
            );
        }
    }
    let st = sim.vehicle_state(v);
    for (i, w) in st.wheels.iter().enumerate() {
        eprintln!(
            "settled wheel {i}: contact {} L {:.3} Fs {:.0} pos {:?} disabled {}",
            w.in_contact, w.suspension_length, w.suspension_force, w.position, w.disabled
        );
    }
    eprintln!("contacts {} chassis {:?}", st.contacts, sim.read_body(sim.vehicle_host().chassis_body(v)).position);
    if std::env::var("TURN").is_ok() {
        let phases = [
            (VehicleInput { forward: 1.0, ..Default::default() }, 180),
            (VehicleInput { brake: 0.6, ..Default::default() }, 60),
            (VehicleInput { forward: 0.3, steer: std::env::var("STEER").map(|s| s.parse().unwrap()).unwrap_or(-0.4), ..Default::default() }, 240),
        ];
        for (input, n) in phases {
            sim.set_vehicle_input(v, input);
            for k in 0..n {
                sim.step().expect("step");
                if k % 15 == 14 {
                    let st = sim.vehicle_state(v);
                    let c = sim.read_body(sim.vehicle_host().chassis_body(v));
                    let q = glam::Quat::from_array(c.orientation);
                    let up = q * glam::Vec3::Y;
                    let lean = up.dot(glam::Vec3::Y.cross(q * glam::Vec3::Z).normalize()).asin().to_degrees();
                    eprintln!(
                        "lean(+left) {lean:.1}° steer {:?} fwd {:.1} st {:.1} br {:.1}: spd {:.2} lat {:.2} yaw {:.2} pos {:?} Fn {:?} fT {:?} slide {:?}",
                        st.wheels.iter().map(|w| (glam::Quat::from_array(w.rotation) * glam::Vec3::Z).x).collect::<Vec<_>>(),
                        input.forward, input.steer, input.brake, st.forward_speed, st.lateral_speed, st.yaw_rate,
                        c.position.map(|x| (x * 10.0).round() / 10.0),
                        st.wheels.iter().map(|w| w.suspension_force as i32).collect::<Vec<_>>(),
                        st.wheels.iter().map(|w| w.lateral_force as i32).collect::<Vec<_>>(),
                        st.wheels.iter().map(|w| w.sliding).collect::<Vec<_>>()
                    );
                }
            }
        }
        return;
    }
    sim.set_vehicle_input(v, VehicleInput { forward: 1.0, ..Default::default() });
    for i in 0..12 {
        for _ in 0..15 {
            sim.step().expect("step");
        }
        let st = sim.vehicle_state(v);
        let r = |x: f32| (x * 10.0).round() / 10.0;
        eprintln!(
            "t {:.2} spd {:.2} rpm {:.0} gear {} clutch {:.2} omega {:?} fL {:?} Fn {:?} slip {:?}",
            (i + 1) as f32 * 0.25,
            st.forward_speed,
            st.engine_rpm,
            st.gear,
            st.clutch,
            st.wheels.iter().map(|w| r(w.omega)).collect::<Vec<_>>(),
            st.wheels.iter().map(|w| w.longitudinal_force as i32).collect::<Vec<_>>(),
            st.wheels.iter().map(|w| w.suspension_force as i32).collect::<Vec<_>>(),
            st.wheels.iter().map(|w| r(w.slip_ratio)).collect::<Vec<_>>()
        );
    }
}

/// Raw vehicle words while a tank starts with throttle + steer (`--ignored`).
#[test]
#[ignore]
fn trace_tank_steer_start() {
    use engine_schema::vehicle::*;
    use engine_vehicle::VehicleBackend;
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 32768,
        max_joints: 64,
        max_candidates: 65536,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 10;
    sim.set_params(p);
    sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    let mut g = BodyState::dynamic(glam::Vec3::new(0.0, -5.0, 0.0), glam::Quat::IDENTITY, 1.0, glam::Vec3::ONE);
    g.inv_mass = 0.0;
    g.inv_inertia = [0.0; 3];
    g.flags = 0;
    let ground = sim.add_body(g);
    sim.add_collider(Collider {
        body: ground,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [400.0, 5.0, 400.0, 400.0],
    });
    let v = sim.add_vehicle(&presets::tank(), [0.0, 2.5, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("vehicle");
    for _ in 0..120 {
        sim.step().expect("step");
    }
    let env = |k: &str, d: f32| std::env::var(k).map(|s| s.parse().unwrap()).unwrap_or(d);
    let (steer, forward, pre) = (env("STEER", -0.4), env("FWD", 0.3), env("PRE", 0.0) as u32);
    sim.set_vehicle_input(v, VehicleInput { forward: 1.0, ..Default::default() });
    for _ in 0..pre {
        sim.step().expect("step");
    }
    sim.set_vehicle_input(v, VehicleInput { brake: 0.6, ..Default::default() });
    for _ in 0..env("BRAKE", 0.0) as u32 {
        sim.step().expect("step");
    }
    sim.set_vehicle_input(v, VehicleInput { forward, steer, ..Default::default() });
    let base = sim.vehicle_host().capacity.layout().vehicle_base;
    for i in 0..120 {
        sim.step().expect("step");
        if i % 10 == 0 {
            let s = sim.read_slots(base, VEH_STRIDE);
            let f = |k: u32| s[k as usize].map(f32::from_bits);
            let st = sim.vehicle_state(v);
            eprintln!(
                "{i}: yaw {:.2} d0 {:?} d1 {:?} trackD {:?} spd {:.2} ω {:?}",
                st.yaw_rate,
                f(V_D0),
                f(V_D1),
                f(V_TRACK_D),
                st.forward_speed,
                [st.wheels[0].omega, st.wheels[7].omega]
            );
        }
    }
}

/// A preset cruising into a stack of 40 kg crates (`PRESET`, `--ignored`): per-step
/// chassis velocity, spin and wheel loads around the impact.
#[test]
#[ignore]
fn trace_stack_impact() {
    let name = std::env::var("PRESET").unwrap_or_else(|_| "monster_truck".into());
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 32768,
        max_joints: 64,
        max_candidates: 65536,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 8;
    sim.set_params(p);
    sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    sim.set_material(1, Material { friction: 0.6, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    sim.set_material(7, Material { friction: 0.9, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    let mut g = BodyState::dynamic(glam::Vec3::new(0.0, -5.0, 0.0), glam::Quat::IDENTITY, 1.0, glam::Vec3::ONE);
    g.inv_mass = 0.0;
    g.inv_inertia = [0.0; 3];
    g.flags = 0;
    let ground = sim.add_body(g);
    sim.add_collider(Collider {
        body: ground,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [400.0, 5.0, 400.0, 400.0],
    });
    for y in 0..5 {
        for i in 0..(4 - (y >> 1)) {
            let m = 40.0f32;
            let body = sim.add_body(BodyState::dynamic(
                glam::Vec3::new((i as f32 - 1.5) * 1.05, 0.5 + y as f32, 13.0),
                glam::Quat::IDENTITY,
                m,
                glam::Vec3::splat(m * 1.0 / 6.0),
            ));
            sim.add_collider(Collider {
                body,
                shape: ShapeType::Box as u32,
                material: 1,
                generation: 0,
                local_pos: [0.0; 3],
                density: 0.0,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: [0.5, 0.5, 0.5, 0.87],
            });
        }
    }
    let desc = presets::by_name(&name).expect("preset");
    let v = sim.add_vehicle(&desc, [0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("vehicle");
    for _ in 0..60 {
        sim.step().expect("step");
    }
    let target = std::env::var("SPEED").map(|s| s.parse().unwrap()).unwrap_or(11.0f32);
    for i in 0..240 {
        let st = sim.vehicle_state(v);
        let e = target - st.forward_speed;
        sim.set_vehicle_input(v, VehicleInput { forward: (e * 0.4).clamp(0.0, 1.0), ..Default::default() });
        sim.step().expect("step");
        let c = sim.read_body(sim.vehicle_host().chassis_body(v));
        let st = sim.vehicle_state(v);
        let speed = glam::Vec3::from_array(c.linear_vel).length();
        let w = glam::Vec3::from_array(c.angular_vel).length();
        let peak = st.wheels.iter().fold(0.0f32, |m, w| m.max(w.suspension_force));
        if i % 10 == 0 || speed > 25.0 || w > 4.0 || peak > 60_000.0 {
            eprintln!(
                "{i}: p {:?} |v| {speed:.1} |w| {w:.1} contacts {} Fs {:?} hit {:?} L {:?} ny {:?}",
                c.position.map(|x| (x * 100.0).round() / 100.0),
                st.contacts,
                st.wheels.iter().map(|w| w.suspension_force as i32).collect::<Vec<_>>(),
                st.wheels.iter().map(|w| w.ground_body).collect::<Vec<_>>(),
                st.wheels.iter().map(|w| (w.suspension_length * 100.0).round() / 100.0).collect::<Vec<_>>(),
                st.wheels.iter().map(|w| (w.contact_normal[1] * 100.0).round() / 100.0).collect::<Vec<_>>()
            );
        }
        if speed > 40.0 {
            break;
        }
    }
}

/// Every preset alone on flat ground with constant throttle + steer (the traffic AI)
/// for 12 s: reports speed and the lowest chassis up.y (`--ignored`).
#[test]
#[ignore]
fn trace_constant_circles() {
    let forward: f32 = std::env::var("FWD").map(|s| s.parse().unwrap()).unwrap_or(0.7);
    let steer: f32 = std::env::var("STEER").map(|s| s.parse().unwrap()).unwrap_or(0.5);
    for desc in presets::all() {
        let mut sim = WgpuSim::new(WorldCapacity {
            max_bodies: 4096,
            max_colliders: 8192,
            max_contacts: 32768,
            max_joints: 64,
            max_candidates: 65536,
            max_colors: 32,
            max_commands: 64,
            max_hull_vertices: 64,
        })
        .expect("wgpu");
        let mut p = AvbdParams::default();
        p.iterations = 8;
        sim.set_params(p);
        sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
        sim.set_material(7, Material { friction: 0.9, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
        let mut g = BodyState::dynamic(glam::Vec3::new(0.0, -5.0, 0.0), glam::Quat::IDENTITY, 1.0, glam::Vec3::ONE);
        g.inv_mass = 0.0;
        g.inv_inertia = [0.0; 3];
        g.flags = 0;
        let ground = sim.add_body(g);
        sim.add_collider(Collider {
            body: ground,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [400.0, 5.0, 400.0, 400.0],
        });
        let v = sim.add_vehicle(&desc, [0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("vehicle");
        sim.set_vehicle_input(v, VehicleInput { forward, steer, ..Default::default() });
        let mut min_up = 1.0f32;
        let mut max_speed = 0.0f32;
        for i in 0..720u32 {
            sim.step().expect("step");
            let c = sim.read_body(sim.vehicle_host().chassis_body(v));
            let up = glam::Quat::from_array(c.orientation) * glam::Vec3::Y;
            min_up = min_up.min(up.y);
            max_speed = max_speed.max(glam::Vec3::from_array(c.linear_vel).length());
            if std::env::var("PRESET").ok().as_deref() == Some(desc.name.as_str()) && i % 30 == 0 {
                let st = sim.vehicle_state(v);
                eprintln!(
                    "  {} up.y {:.2} speed {:.1} yaw {:.2} contacts {} Fn {:?}",
                    i,
                    up.y,
                    st.forward_speed,
                    st.yaw_rate,
                    st.contacts,
                    st.wheels.iter().map(|w| w.suspension_force as i32).collect::<Vec<_>>()
                );
            }
        }
        let st = sim.vehicle_state(v);
        eprintln!("{:>20}: max speed {max_speed:5.1} m/s, min up.y {min_up:.2}, final yaw {:.2} contacts {}", desc.name, st.yaw_rate, st.contacts);
    }
}

/// Handbrake turn vs plain turn from 15 m/s (`PRESET`, `--ignored`): yaw rate,
/// slip angle and rear wheel state every 6 steps.
#[test]
#[ignore]
fn trace_handbrake_turn() {
    let name = std::env::var("PRESET").unwrap_or_else(|_| "sedan".into());
    let hbs: Vec<f32> = if std::env::var("NOHB").is_ok() { vec![0.0] } else { vec![0.0, 1.0] };
    for hb in hbs {
        let mut sim = WgpuSim::new(WorldCapacity {
            max_bodies: 4096,
            max_colliders: 8192,
            max_contacts: 32768,
            max_joints: 64,
            max_candidates: 65536,
            max_colors: 32,
            max_commands: 64,
            max_hull_vertices: 64,
        })
        .expect("wgpu");
        let mut p = AvbdParams::default();
        p.iterations = 8;
        sim.set_params(p);
        sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
        let mut g = BodyState::dynamic(glam::Vec3::new(0.0, -5.0, 0.0), glam::Quat::IDENTITY, 1.0, glam::Vec3::ONE);
        g.inv_mass = 0.0;
        g.inv_inertia = [0.0; 3];
        g.flags = 0;
        let ground = sim.add_body(g);
        sim.add_collider(Collider {
            body: ground,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [400.0, 5.0, 400.0, 400.0],
        });
        let desc = presets::by_name(&name).expect("preset");
        let v = sim.add_vehicle(&desc, [0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("vehicle");
        for _ in 0..60 {
            sim.step().expect("step");
        }
        for _ in 0..400 {
            let e = 15.0 - sim.vehicle_state(v).forward_speed;
            sim.set_vehicle_input(v, VehicleInput { forward: (e * 0.4).clamp(0.0, 1.0), ..Default::default() });
            sim.step().expect("step");
            if e.abs() < 0.2 {
                break;
            }
        }
        let env = |k: &str, d: f32| std::env::var(k).map(|s| s.parse().unwrap()).unwrap_or(d);
        sim.set_vehicle_input(v, VehicleInput { forward: env("FWD", 0.0), steer: env("STEER", -1.0), handbrake: hb, ..Default::default() });
        let mut heading = 0.0f32;
        for i in 0..72 {
            sim.step().expect("step");
            let st = sim.vehicle_state(v);
            heading += st.yaw_rate / 60.0;
            if i % 6 == 5 {
                eprintln!(
                    "hb {hb} t {:.2}: yaw {:.2} heading {:5.1}° fwd {:.1} lat {:.1} rear ω {:?} slide {:?} fT {:?} fL {:?}",
                    (i + 1) as f32 / 60.0,
                    st.yaw_rate,
                    heading.to_degrees(),
                    st.forward_speed,
                    st.lateral_speed,
                    st.wheels.iter().map(|w| (w.omega * 10.0).round() / 10.0).collect::<Vec<_>>(),
                    st.wheels.iter().map(|w| w.sliding).collect::<Vec<_>>(),
                    st.wheels.iter().map(|w| w.lateral_force as i32).collect::<Vec<_>>(),
                    st.wheels.iter().map(|w| w.longitudinal_force as i32).collect::<Vec<_>>(),
                );
            }
        }
    }
}

/// Straight-line full throttle (`PRESET`, `FWD`, `SECS`, `--ignored`): heading drift,
/// lean, wheel loads and contact every 0.5 s.
#[test]
#[ignore]
fn trace_straight_line() {
    let env = |k: &str, d: f32| std::env::var(k).map(|s| s.parse().unwrap()).unwrap_or(d);
    let name = std::env::var("PRESET").unwrap_or_else(|_| "motorcycle".into());
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 32768,
        max_joints: 64,
        max_candidates: 65536,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 8;
    sim.set_params(p);
    sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    let mut g = BodyState::dynamic(glam::Vec3::new(0.0, -5.0, 0.0), glam::Quat::IDENTITY, 1.0, glam::Vec3::ONE);
    g.inv_mass = 0.0;
    g.inv_inertia = [0.0; 3];
    g.flags = 0;
    let ground = sim.add_body(g);
    sim.add_collider(Collider {
        body: ground,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [800.0, 5.0, 800.0, 800.0],
    });
    let desc = presets::by_name(&name).expect("preset");
    let v = sim.add_vehicle(&desc, [0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("vehicle");
    for _ in 0..90 {
        sim.step().expect("step");
    }
    // Optional warm-up: a slow turn (PRE_STEER for PRE_SECS at 0.3 throttle) so the
    // straight run starts from a realistic, slightly asymmetric state.
    for i in 0..(env("PRE_SECS", 0.0) * 60.0) as u32 {
        sim.set_vehicle_input(v, VehicleInput { forward: 0.3, steer: env("PRE_STEER", 0.0), ..Default::default() });
        sim.step().expect("step");
        if i % 15 == 14 {
            let c = sim.read_body(sim.vehicle_host().chassis_body(v));
            let q = glam::Quat::from_array(c.orientation);
            let up = q * glam::Vec3::Y;
            let lean = up.dot(glam::Vec3::Y.cross(q * glam::Vec3::Z).normalize()).asin().to_degrees();
            let st = sim.vehicle_state(v);
            eprintln!(
                "pre {:4.2}: speed {:5.1} yaw {:5.2} lean {lean:5.1}° Fn {:?} slide {:?} slip {:?}",
                (i + 1) as f32 / 60.0,
                st.forward_speed,
                st.yaw_rate,
                st.wheels.iter().map(|w| w.suspension_force as i32).collect::<Vec<_>>(),
                st.wheels.iter().map(|w| w.sliding).collect::<Vec<_>>(),
                st.wheels.iter().map(|w| (w.slip_ratio * 100.0).round() / 100.0).collect::<Vec<_>>(),
            );
        }
    }
    sim.set_vehicle_input(v, VehicleInput { forward: env("FWD", 1.0), ..Default::default() });
    let steps = (env("SECS", 15.0) * 60.0) as u32;
    for i in 0..steps {
        sim.step().expect("step");
        if i % 30 == 29 {
            let c = sim.read_body(sim.vehicle_host().chassis_body(v));
            let q = glam::Quat::from_array(c.orientation);
            let fwd = q * glam::Vec3::Z;
            let up = q * glam::Vec3::Y;
            let lean = up.dot(glam::Vec3::Y.cross(fwd).normalize()).asin().to_degrees();
            let heading = fwd.x.atan2(fwd.z).to_degrees();
            let st = sim.vehicle_state(v);
            eprintln!(
                "t {:4.1}: speed {:5.1} gear {} heading {heading:6.1}° yaw {:5.2} lean {lean:5.1}° pitch {:5.1}° x {:6.2} contacts {} Fn {:?} slide {:?}",
                (i + 1) as f32 / 60.0,
                st.forward_speed,
                st.gear,
                st.yaw_rate,
                fwd.y.asin().to_degrees(),
                c.position[0],
                st.contacts,
                st.wheels.iter().map(|w| w.suspension_force as i32).collect::<Vec<_>>(),
                st.wheels.iter().map(|w| w.sliding).collect::<Vec<_>>(),
            );
        }
    }
}

/// Kart dropped on a monster truck (`--ignored`): two-body link state and loads.
#[test]
#[ignore]
fn trace_car_on_car() {
    use engine_schema::vehicle::*;
    use engine_vehicle::VehicleBackend;
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 32768,
        max_joints: 64,
        max_candidates: 65536,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 10;
    sim.set_params(p);
    sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    let mut g = BodyState::dynamic(glam::Vec3::new(0.0, -5.0, 0.0), glam::Quat::IDENTITY, 1.0, glam::Vec3::ONE);
    g.inv_mass = 0.0;
    g.inv_inertia = [0.0; 3];
    g.flags = 0;
    let ground = sim.add_body(g);
    sim.add_collider(Collider {
        body: ground,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [400.0, 5.0, 400.0, 400.0],
    });
    let t = sim.add_vehicle(&presets::monster_truck(), [0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("truck");
    for _ in 0..120 {
        sim.step().expect("step");
    }
    let roof = sim.read_body(sim.vehicle_host().chassis_body(t)).position[1];
    let k = sim.add_vehicle(&presets::kart(), [0.0, roof + 1.6, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("kart");
    let l = sim.vehicle_host().capacity.layout();
    for i in 0..240 {
        sim.step().expect("step");
        if i % 20 == 0 {
            let tc = sim.read_body(sim.vehicle_host().chassis_body(t));
            let kc = sim.read_body(sim.vehicle_host().chassis_body(k));
            let ts = sim.vehicle_state(t);
            let ks = sim.vehicle_state(k);
            // kart wheel slots follow the truck's (4) in the pool
            let tj: Vec<u32> = (0..4).map(|w| sim.read_slots(l.wheel_base + (4 + w) * WHEEL_STRIDE + W_TJ, 1)[0][2]).collect();
            eprintln!(
                "{i}: truck y {:.3} flags {} load {:.0} | kart y {:.3} flags {} contacts {} grounds {:?} twoBody {:?} kart Fs {:?}",
                tc.position[1],
                tc.flags,
                ts.wheels.iter().map(|w| w.suspension_force).sum::<f32>(),
                kc.position[1],
                kc.flags,
                ks.contacts,
                ks.wheels.iter().map(|w| w.ground_body).collect::<Vec<_>>(),
                tj,
                ks.wheels.iter().map(|w| w.suspension_force as i32).collect::<Vec<_>>(),
            );
        }
    }
}

/// Traffic-scale timing (`--ignored`): 48 vehicles + 400 crates, ms per step.
#[test]
#[ignore]
fn bench_traffic() {
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 8192,
        max_colliders: 16384,
        max_contacts: 131072,
        max_joints: 64,
        max_candidates: 262144,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 8;
    sim.set_params(p);
    sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    let mut g = BodyState::dynamic(glam::Vec3::new(0.0, -5.0, 0.0), glam::Quat::IDENTITY, 1.0, glam::Vec3::ONE);
    g.inv_mass = 0.0;
    g.inv_inertia = [0.0; 3];
    g.flags = 0;
    let ground = sim.add_body(g);
    sim.add_collider(Collider {
        body: ground,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [400.0, 5.0, 400.0, 400.0],
    });
    let kinds = ["sedan", "sports", "kart", "jolt_default", "destructible_sedan", "monster_truck"];
    let mut rng = 7u32;
    let mut rand = || {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        rng as f32 / 4294967296.0
    };
    let mut vs = Vec::new();
    for i in 0..48 {
        let (x, z) = (((i % 7) as f32 - 3.5) * 14.0, ((i / 7) as f32 - 3.5) * 14.0);
        let k = kinds[i % kinds.len()];
        let y = if k == "monster_truck" { 3.5 } else { 2.2 };
        let v = sim.add_vehicle(&presets::by_name(k).unwrap(), [x, y, z], [0.0, 0.0, 0.0, 1.0]).expect("v");
        vs.push(v);
    }
    for _ in 0..400 {
        let body = sim.add_body(BodyState::dynamic(
            glam::Vec3::new((rand() - 0.5) * 120.0, 0.5 + rand() * 6.0, (rand() - 0.5) * 120.0),
            glam::Quat::IDENTITY,
            30.0,
            glam::Vec3::splat(5.0),
        ));
        sim.add_collider(Collider {
            body,
            shape: ShapeType::Box as u32,
            material: 0,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [0.5, 0.5, 0.5, 0.87],
        });
    }
    for (i, v) in vs.iter().enumerate() {
        sim.set_vehicle_input(*v, VehicleInput { forward: 0.3 + 0.01 * i as f32, steer: if i % 2 == 0 { 0.3 } else { -0.3 }, ..Default::default() });
    }
    for _ in 0..60 {
        sim.step().expect("step");
    }
    let t0 = std::time::Instant::now();
    for _ in 0..300 {
        sim.step().expect("step");
    }
    eprintln!("BENCH traffic: {:.2} ms/step", t0.elapsed().as_secs_f64() * 1000.0 / 300.0);
}

/// Raw per-wheel query words (hit distance t, L at x⁻, normal, flags, bump K) for the
/// wheels of `v`, one line each, when any wheel load passes `min_load` N.
fn dump_wheel_rows(sim: &WgpuSim, v: engine_vehicle::VehicleHandle, step: u32, min_load: f32) -> bool {
    use engine_schema::vehicle::*;
    use engine_vehicle::VehicleBackend;
    let st = sim.vehicle_state(v);
    let peak = st.wheels.iter().fold(0.0f32, |m, w| m.max(w.suspension_force));
    if peak < min_load {
        return false;
    }
    let l = sim.vehicle_host().capacity.layout();
    let c = sim.read_body(sim.vehicle_host().chassis_body(v));
    eprintln!(
        "step {step}: chassis p {:?} v {:?} w {:?}",
        c.position.map(|x| (x * 100.0).round() / 100.0),
        c.linear_vel.map(|x| (x * 10.0).round() / 10.0),
        c.angular_vel.map(|x| (x * 10.0).round() / 10.0)
    );
    for (i, w) in st.wheels.iter().enumerate() {
        let slot = sim.vehicle_host().wheel_slot(v, i as u32);
        let s = sim.read_slots(l.wheel_base + slot * WHEEL_STRIDE, WHEEL_STRIDE);
        let f = |k: u32| s[k as usize].map(f32::from_bits);
        let r3 = |a: [f32; 4]| a.map(|x| (x * 1000.0).round() / 1000.0);
        eprintln!(
            "   w{i}: Fs {:>9.0} flags {:#x} ground {:?} col {} t {:.3} L0 {:.3} n·invCos {:?} K {:?}",
            w.suspension_force,
            s[W_C0 as usize][2],
            w.ground_body,
            s[W_C0 as usize][1] as i32,
            f(W_HIT)[3],
            f(W_C1)[3],
            r3(f(W_C2)),
            f(W_K).map(|x| x.round())
        );
    }
    true
}

/// Crate-stack wedge (`PRESET`, `SPEED`, `MIN_LOAD`, `--ignored`): the wheel rows at every
/// step whose peak wheel load passes MIN_LOAD (default 20× vehicle weight).
#[test]
#[ignore]
fn trace_crate_wedge() {
    let name = std::env::var("PRESET").unwrap_or_else(|_| "monster_truck".into());
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 32768,
        max_joints: 64,
        max_candidates: 65536,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 10;
    sim.set_params(p);
    sim.set_material(0, Material { friction: 1.0, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    sim.set_material(1, Material { friction: 0.6, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 });
    let mut g = BodyState::dynamic(glam::Vec3::new(0.0, -5.0, 0.0), glam::Quat::IDENTITY, 1.0, glam::Vec3::ONE);
    g.inv_mass = 0.0;
    g.inv_inertia = [0.0; 3];
    g.flags = 0;
    let ground = sim.add_body(g);
    sim.add_collider(Collider {
        body: ground,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [400.0, 5.0, 400.0, 400.0],
    });
    for y in 0..5 {
        for i in 0..(4 - (y >> 1)) {
            let body = sim.add_body(BodyState::dynamic(
                glam::Vec3::new((i as f32 - 1.5) * 1.05, 0.5 + y as f32, 13.0),
                glam::Quat::IDENTITY,
                40.0,
                glam::Vec3::splat(40.0 / 6.0),
            ));
            sim.add_collider(Collider {
                body,
                shape: ShapeType::Box as u32,
                material: 1,
                generation: 0,
                local_pos: [0.0; 3],
                density: 0.0,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: [0.5, 0.5, 0.5, 0.87],
            });
        }
    }
    let desc = presets::by_name(&name).expect("preset");
    let weight = (desc.parts.iter().map(|p| p.mass).sum::<f32>() + desc.wheels.iter().map(|w| w.mass).sum::<f32>()) * 10.0;
    let min_load = std::env::var("MIN_LOAD").map(|s| s.parse().unwrap()).unwrap_or(20.0 * weight);
    let v = sim.add_vehicle(&desc, [0.0, 2.5, 0.0], [0.0, 0.0, 0.0, 1.0]).expect("vehicle");
    for _ in 0..60 {
        sim.step().expect("step");
    }
    let target = std::env::var("SPEED").map(|s| s.parse().unwrap()).unwrap_or(11.0f32);
    let (mut peak, mut max_y) = (0.0f32, 0.0f32);
    for i in 0..240 {
        let e = target - sim.vehicle_state(v).forward_speed;
        sim.set_vehicle_input(v, VehicleInput { forward: (e * 0.4).clamp(0.0, 1.0), ..Default::default() });
        sim.step().expect("step");
        dump_wheel_rows(&sim, v, i, min_load);
        peak = sim.vehicle_state(v).wheels.iter().fold(peak, |m, w| m.max(w.suspension_force));
        max_y = max_y.max(sim.read_body(sim.vehicle_host().chassis_body(v)).position[1]);
    }
    eprintln!("{name}: peak wheel load {:.1}× weight, max chassis y {max_y:.2}", peak / weight);
}

// ---------------------------------------------------------------------------
// vehicle.html scenes, rebuilt body for body (same order, LCG, materials).
// ---------------------------------------------------------------------------
fn demo_world(bodies: u32) -> WgpuSim {
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: bodies,
        max_colliders: bodies * 2,
        max_contacts: bodies * 16,
        max_joints: 1024,
        max_candidates: bodies * 32,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu");
    let mut p = AvbdParams::default();
    p.iterations = 8;
    sim.set_params(p);
    let mat = |f: f32| Material { friction: f, restitution: 0.0, contact_offset: 0.01, _pad: 0.0 };
    sim.set_material(0, mat(1.0)); // ground
    sim.set_material(1, mat(0.6)); // props
    sim.set_material(2, mat(0.08)); // ice
    sim.set_material(engine_vehicle::pack::WHEEL_RIM_MATERIAL, mat(0.9));
    demo_box(&mut sim, [0.0, -5.0, 0.0], [400.0, 5.0, 400.0], [0.0, 0.0, 0.0, 1.0], 0, 0.0);
    sim
}

fn q_axis(ax: [f32; 3], ang: f32) -> [f32; 4] {
    let s = (ang * 0.5).sin();
    [ax[0] * s, ax[1] * s, ax[2] * s, (ang * 0.5).cos()]
}

/// Box body + collider; mass 0 = static.
fn demo_box(sim: &mut WgpuSim, p: [f32; 3], h: [f32; 3], q: [f32; 4], material: u32, mass: f32) -> u32 {
    let mut b = BodyState::dynamic(glam::Vec3::from_array(p), glam::Quat::from_array(q), 1.0, glam::Vec3::ONE);
    if mass > 0.0 {
        b.inv_mass = 1.0 / mass;
        b.inv_inertia = [
            3.0 / (mass * (h[1] * h[1] + h[2] * h[2])),
            3.0 / (mass * (h[0] * h[0] + h[2] * h[2])),
            3.0 / (mass * (h[0] * h[0] + h[1] * h[1])),
        ];
    } else {
        b.inv_mass = 0.0;
        b.inv_inertia = [0.0; 3];
        b.flags = 0;
    }
    b.initial_pos = p;
    b.initial_rot = q;
    let body = sim.add_body(b);
    sim.add_collider(Collider {
        body,
        shape: ShapeType::Box as u32,
        material,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [h[0], h[1], h[2], (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt()],
    });
    body
}

fn demo_vehicle(sim: &mut WgpuSim, name: &str, p: [f32; 3], yaw: f32) -> engine_vehicle::VehicleHandle {
    sim.add_vehicle(&presets::by_name(name).expect("preset"), p, q_axis([0.0, 1.0, 0.0], yaw)).expect("vehicle")
}

/// vehicle.html `scene=playground` with the player preset at (x, z, yaw).
fn playground(preset: &str, x: f32, z: f32, yaw: f32) -> (WgpuSim, engine_vehicle::VehicleHandle) {
    let mut sim = demo_world(6000);
    let v = demo_vehicle(&mut sim, preset, [x, 2.5, z], yaw);
    let id = [0.0, 0.0, 0.0, 1.0];
    demo_box(&mut sim, [0.0, 0.6, 30.0], [4.0, 0.4, 6.0], q_axis([1.0, 0.0, 0.0], -0.22), 1, 0.0);
    demo_box(&mut sim, [0.0, 1.95, 41.0], [4.0, 0.4, 5.5], id, 1, 0.0);
    demo_box(&mut sim, [0.0, 0.6, 52.0], [4.0, 0.4, 6.0], q_axis([1.0, 0.0, 0.0], 0.22), 1, 0.0);
    demo_box(&mut sim, [-22.0, 0.9, 10.0], [3.0, 0.3, 4.0], q_axis([1.0, 0.0, 0.0], -0.32), 1, 0.0);
    demo_box(&mut sim, [24.0, 0.0, -20.0], [10.0, 1.5, 14.0], q_axis([0.0, 0.0, 1.0], 0.18), 1, 0.0);
    demo_box(&mut sim, [-30.0, -0.02, -25.0], [12.0, 0.05, 12.0], id, 2, 0.0);
    for i in 0..8 {
        demo_box(&mut sim, [-12.0 + i as f32 * 3.4, 0.6, 18.0], [0.25, 0.6, 0.25], id, 1, 0.0);
    }
    for (x, z) in [(8.0f32, 26.0f32), (-10.0, 32.0), (14.0, 40.0)] {
        for y in 0..5 {
            for i in 0..(4 - (y >> 1)) {
                demo_box(&mut sim, [x + (i as f32 - 1.5) * 1.05, 0.5 + y as f32, z], [0.5; 3], id, 1, 40.0);
            }
        }
    }
    for c in 0..8 {
        for i in 0..14 {
            let p = [-20.0 + i as f32 * 1.02 + (c & 1) as f32 * 0.5, 0.25 + c as f32 * 0.5, 60.0];
            demo_box(&mut sim, p, [0.5, 0.25, 0.25], id, 1, 12.0);
        }
    }
    for i in 0..6 {
        let (r, m) = (0.8f32, 60.0f32);
        let body = sim.add_body(BodyState::dynamic(glam::Vec3::new(20.0 + i as f32 * 2.2, 0.8, 6.0), glam::Quat::IDENTITY, m, glam::Vec3::splat(0.4 * m * r * r)));
        sim.add_collider(Collider {
            body,
            shape: ShapeType::Sphere as u32,
            material: 1,
            generation: 0,
            local_pos: [0.0; 3],
            density: 0.0,
            local_rot: [0.0, 0.0, 0.0, 1.0],
            params: [r, 0.0, 0.0, r],
        });
    }
    demo_vehicle(&mut sim, "monster_truck", [-14.0, 3.5, -12.0], 0.4);
    demo_vehicle(&mut sim, "kart", [6.0, 1.5, -6.0], 2.0);
    demo_vehicle(&mut sim, "six_wheel_truck", [18.0, 3.0, 22.0], -1.2);
    (sim, v)
}

/// record-vehicle-demos.mjs `monster_stack` take (`MIN_LOAD`, `--ignored`): chassis
/// pose every 0.25 s and the wheel rows on load spikes.
#[test]
#[ignore]
fn trace_monster_stack_take() {
    let (mut sim, v) = playground("monster_truck", -10.0, 22.0, 0.0);
    let min_load: f32 = std::env::var("MIN_LOAD").map(|s| s.parse().unwrap()).unwrap_or(600_000.0);
    // (steps, cruise m/s (<0 = none), steer, brake)
    let script = [(40u32, -1.0f32, 0.0f32, 0.0f32), (150, 11.0, 0.0, 0.0), (120, 11.0, 0.0, 0.0), (150, 10.0, 1.0, 0.0), (210, 12.0, 0.15, 0.0), (100, -1.0, 0.0, 1.0)];
    let (mut step, mut min_up, mut peak) = (0u32, 1.0f32, 0.0f32);
    for (n, cruise, steer, brake) in script {
        sim.set_vehicle_input(v, VehicleInput { forward: 0.0, steer, brake, ..Default::default() });
        for _ in 0..n {
            if cruise >= 0.0 {
                let e = cruise - sim.vehicle_state(v).forward_speed;
                let brake = if e < -1.0 { (-e * 0.2).min(1.0) } else { 0.0 };
                sim.set_vehicle_input(v, VehicleInput { forward: (e * 0.4).clamp(0.0, 1.0), steer, brake, ..Default::default() });
            }
            sim.step().expect("step");
            step += 1;
            let c = sim.read_body(sim.vehicle_host().chassis_body(v));
            let up = glam::Quat::from_array(c.orientation) * glam::Vec3::Y;
            min_up = min_up.min(up.y);
            let st = sim.vehicle_state(v);
            peak = st.wheels.iter().fold(peak, |m, w| m.max(w.suspension_force));
            if !dump_wheel_rows(&sim, v, step, min_load) && step % 15 == 0 {
                eprintln!(
                    "{step}: p {:?} up.y {:.2} |v| {:.1} |w| {:.1} spd {:.1} contacts {} Fs {:?}",
                    c.position.map(|x| (x * 100.0).round() / 100.0),
                    up.y,
                    glam::Vec3::from_array(c.linear_vel).length(),
                    glam::Vec3::from_array(c.angular_vel).length(),
                    st.forward_speed,
                    st.contacts,
                    st.wheels.iter().map(|w| w.suspension_force as i32).collect::<Vec<_>>()
                );
            }
        }
    }
    let c = sim.read_body(sim.vehicle_host().chassis_body(v));
    let up = glam::Quat::from_array(c.orientation) * glam::Vec3::Y;
    eprintln!("monster_stack: final up.y {:.2} (min {min_up:.2}) p {:?} peak wheel load {peak:.0} N", up.y, c.position);
}

/// vehicle.html `scene=traffic` (record take `traffic`): the highest chassis of all 49
/// vehicles, and the wheel rows of the vehicle that hops highest (`--ignored`).
#[test]
#[ignore]
fn trace_traffic_hop() {
    let mut sim = demo_world(8192);
    let player = demo_vehicle(&mut sim, "sedan", [0.0, 2.5, 0.0], 0.0);
    let kinds = ["sedan", "sports", "kart", "jolt_default", "destructible_sedan", "monster_truck"];
    let mut rng = 7u32;
    let mut rand = || {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        rng as f32 / 4294967296.0
    };
    let n = 48;
    let cols = (n as f32).sqrt().ceil() as i32;
    let mut vs = vec![(player, "sedan", 2.5f32)];
    let mut ai = Vec::new();
    for i in 0..n {
        let (x, z) = (((i % cols) as f32 - cols as f32 / 2.0) * 14.0, ((i / cols) as f32 - cols as f32 / 2.0) * 14.0);
        let kind = kinds[i as usize % kinds.len()];
        let y = if kind == "monster_truck" { 3.5 } else { 2.2 };
        let h = demo_vehicle(&mut sim, kind, [x, y, z], rand() * 6.28);
        let input = VehicleInput { forward: 0.2 + 0.25 * rand(), steer: (if rand() < 0.5 { -1.0 } else { 1.0 }) * (0.15 + 0.35 * rand()), ..Default::default() };
        ai.push((h, input));
        vs.push((h, kind, y));
    }
    let id = [0.0, 0.0, 0.0, 1.0];
    for _ in 0..400 {
        let p = [(rand() - 0.5) * 120.0, 0.5 + rand() * 6.0, (rand() - 0.5) * 120.0];
        demo_box(&mut sim, p, [0.5; 3], id, 1, 30.0);
    }
    for (h, input) in &ai {
        sim.set_vehicle_input(*h, *input);
    }
    let watch: Option<usize> = std::env::var("WATCH").ok().map(|s| s.parse().unwrap());
    let mut best = (0.0f32, 0usize, 0u32);
    let mut settled_y = vec![0.0f32; vs.len()];
    for step in 0..720u32 {
        let input = if step < 480 { VehicleInput { forward: 0.4, steer: 0.2, ..Default::default() } } else { VehicleInput { forward: 0.6, steer: -0.3, ..Default::default() } };
        sim.set_vehicle_input(player, input);
        sim.step().expect("step");
        for (k, (h, _, _)) in vs.iter().enumerate() {
            let y = sim.read_body(sim.vehicle_host().chassis_body(*h)).position[1];
            if step == 90 {
                settled_y[k] = y;
            }
            if step > 90 && y - settled_y[k] > best.0 {
                best = (y - settled_y[k], k, step);
            }
            if watch == Some(k) && step > 90 {
                dump_wheel_rows(&sim, *h, step, 0.0);
            }
        }
    }
    let (rise, k, step) = best;
    eprintln!("highest hop: vehicle {k} ({}) rose {rise:.2} m above its settled height {:.2} at step {step}", vs[k].1, settled_y[k]);
}

/// Static ledge beside a monster truck's front-left mount (`--ignored`): wheel rows.
#[test]
#[ignore]
fn trace_ledge_beside_mount() {
    let mut sim = demo_world(4096);
    let env = |k: &str, d: f32| std::env::var(k).map(|s| s.parse().unwrap()).unwrap_or(d);
    demo_box(&mut sim, [1.45 + env("DX", 0.5) + 0.4, env("Y", 1.95), 1.75], [0.4, 0.4, 0.4], [0.0, 0.0, 0.0, 1.0], 1, 0.0);
    let v = demo_vehicle(&mut sim, "monster_truck", [0.0, 2.0, 0.0], 0.0);
    for i in 0..120 {
        sim.step().expect("step");
        if i % 20 == 0 || i < 3 {
            dump_wheel_rows(&sim, v, i, 0.0);
        }
    }
}

/// Full-lock turn from a held speed on flat asphalt (`PRESET`, `STEER`, `--ignored`):
/// min chassis up.y per entry speed, to compare with the static stability factor.
#[test]
#[ignore]
fn trace_rollover_threshold() {
    let name = std::env::var("PRESET").unwrap_or_else(|_| "monster_truck".into());
    let steer: f32 = std::env::var("STEER").map(|s| s.parse().unwrap()).unwrap_or(1.0);
    for speed in [5.0f32, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0] {
        let mut sim = demo_world(4096);
        let v = demo_vehicle(&mut sim, &name, [0.0, 2.5, -150.0], 0.0);
        for _ in 0..60 {
            sim.step().expect("step");
        }
        for _ in 0..900 {
            let e = speed - sim.vehicle_state(v).forward_speed;
            sim.set_vehicle_input(v, VehicleInput { forward: (e * 0.4).clamp(0.0, 1.0), brake: if e < -1.0 { (-e * 0.2).min(1.0) } else { 0.0 }, ..Default::default() });
            sim.step().expect("step");
            if e.abs() < 0.1 {
                break;
            }
        }
        let mut min_up = 1.0f32;
        let mut max_lat = 0.0f32;
        for _ in 0..240 {
            let e = speed - sim.vehicle_state(v).forward_speed;
            sim.set_vehicle_input(v, VehicleInput { forward: (e * 0.4).clamp(0.0, 1.0), steer, brake: if e < -1.0 { (-e * 0.2).min(1.0) } else { 0.0 }, ..Default::default() });
            sim.step().expect("step");
            let c = sim.read_body(sim.vehicle_host().chassis_body(v));
            min_up = min_up.min((glam::Quat::from_array(c.orientation) * glam::Vec3::Y).y);
            let st = sim.vehicle_state(v);
            max_lat = max_lat.max((st.forward_speed * st.yaw_rate).abs() / 10.0);
        }
        eprintln!("{name} full lock ({steer}) at {speed:4.1} m/s: min up.y {min_up:.2} (rolled: {}), peak lateral {max_lat:.2} g", min_up < 0.5);
    }
}
