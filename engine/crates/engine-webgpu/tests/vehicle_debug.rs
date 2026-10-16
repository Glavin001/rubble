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
